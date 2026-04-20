"""
Accessible RetNet — 124M Parameter Implementation
Based on "Retentive Network: A Successor to Transformer for Large Language Models"
Architecture specs from Task.md

Model Config:
  d_model     = 768
  n_layers    = 12
  n_heads     = 6
  d_head_qk   = 128   (Q and K)
  d_head_v    = 256   (V)
  d_ffn       = 1536
  vocab_size  = 50257 (GPT-2)
  max_seq_len = 512
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

class RetNetConfig:
    def __init__(self):
        self.d_model     = 768
        self.n_layers    = 12
        self.n_heads     = 6
        self.d_head_qk   = 128   # Q, K head dimension
        self.d_head_v    = 256   # V head dimension (asymmetric)
        self.d_ffn       = 1536
        self.vocab_size  = 50257
        self.max_seq_len = 512
        self.dropout     = 0.0
        # Multi-scale decay: γ_h = 1 - 2^(-(5+h)) for h in {0,...,5}
        self.gammas = [1.0 - 2.0 ** (-(5 + h)) for h in range(self.n_heads)]


# ─────────────────────────────────────────────────────────────────────────────
# xPos: Extrapolatable Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

def _build_xpos_decay(d: int, base: int = 512) -> torch.Tensor:
    """
    Build the per-dimension decay scale ζ used in xPos.
    Shape: (d//2,)
    ζ_i = ((i + d/4) / d)^(1) — gives stable extrapolation beyond training length.
    """
    half = d // 2
    # i ranges from 0 to half-1
    idx = torch.arange(half, dtype=torch.float32)
    scale = ((idx + half) / d)  # values in [0.5, 1.0)
    return scale  # (half,)


def apply_xpos(x: torch.Tensor, offset: int = 0, scale_base: int = 512) -> torch.Tensor:
    """
    Apply xPos rotary encoding to a tensor of shape (B, T, H, d_head).
    offset: starting position index (for recurrent/chunked modes).

    FP16 safety: all intermediate computations are done in float32
    and the xPos decay uses ζ^(pos/scale_base) to bound the exponent to ~[0,1].
    """
    orig_dtype = x.dtype
    B, T, H, d = x.shape
    half = d // 2

    # Always compute position encoding in float32 to avoid FP16 underflow
    pos = torch.arange(offset, offset + T, dtype=torch.float32, device=x.device)  # (T,)

    freq_seq = torch.arange(half, dtype=torch.float32, device=x.device)
    inv_freq = 1.0 / (10000.0 ** (freq_seq / half))  # (half,)
    angles = torch.outer(pos, inv_freq)  # (T, half)

    # xPos decay: ζ^(pos/scale_base) — exponent ≤ 1.0 when pos ≤ scale_base
    # This keeps all values in [ζ^1, 1.0] ⊆ [0.5, 1.0], safe for FP16.
    zeta = _build_xpos_decay(d).to(x.device)  # (half,), values in [0.5, 1.0)
    norm_pos = pos / scale_base                # (T,), values in [0, ~1]
    decay = torch.pow(zeta.unsqueeze(0), norm_pos.unsqueeze(1))  # (T, half) in [0.5, 1.0]

    cos_a = (torch.cos(angles) * decay).unsqueeze(0).unsqueeze(2)   # (1, T, 1, half)
    sin_a = (torch.sin(angles) * decay).unsqueeze(0).unsqueeze(2)

    # Cast input to float32 for the rotation, then cast back
    x_f32 = x.float()
    x1, x2 = x_f32[..., :half], x_f32[..., half:]
    out1 = x1 * cos_a - x2 * sin_a
    out2 = x1 * sin_a + x2 * cos_a

    return torch.cat([out1, out2], dim=-1).to(orig_dtype)  # (B, T, H, d)


def apply_xpos_key(x: torch.Tensor, offset: int = 0, scale_base: int = 512) -> torch.Tensor:
    """Apply xPos to keys (uses inverse scale ζ^(-pos/scale_base))."""
    orig_dtype = x.dtype
    B, T, H, d = x.shape
    half = d // 2

    pos = torch.arange(offset, offset + T, dtype=torch.float32, device=x.device)
    freq_seq = torch.arange(half, dtype=torch.float32, device=x.device)
    inv_freq = 1.0 / (10000.0 ** (freq_seq / half))
    angles = torch.outer(pos, inv_freq)

    zeta = _build_xpos_decay(d).to(x.device)
    norm_pos = pos / scale_base
    # Inverse decay for keys: ζ^(-pos/scale_base), bounded in [1.0, 1/ζ] ≈ [1.0, 2.0]
    decay = torch.pow(zeta.unsqueeze(0), -norm_pos.unsqueeze(1))

    cos_a = (torch.cos(angles) * decay).unsqueeze(0).unsqueeze(2)
    sin_a = (torch.sin(angles) * decay).unsqueeze(0).unsqueeze(2)

    x_f32 = x.float()
    x1, x2 = x_f32[..., :half], x_f32[..., half:]
    out1 = x1 * cos_a - x2 * sin_a
    out2 = x1 * sin_a + x2 * cos_a
    return torch.cat([out1, out2], dim=-1).to(orig_dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Scale Retention
# ─────────────────────────────────────────────────────────────────────────────

class MultiScaleRetention(nn.Module):
    """
    Multi-scale retention with:
    - Parallel mode for training (full sequence, causal)
    - Recurrent mode for inference (step-by-step, O(1) memory)

    Each head h uses decay γ_h = 1 - 2^(-(5+h)).
    Q,K dimension: d_head_qk=128; V dimension: d_head_v=256.
    Output: grouped-normalized concat → gated projection.
    """

    def __init__(self, config: RetNetConfig):
        super().__init__()
        self.n_heads  = config.n_heads
        self.d_model  = config.d_model
        self.d_qk     = config.d_head_qk
        self.d_v      = config.d_head_v
        self.gammas   = config.gammas
        self.max_seq  = config.max_seq_len

        # Projections
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.d_head_qk, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_heads * config.d_head_qk, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_heads * config.d_head_v,  bias=False)

        # Output: gate + proj
        # After GroupNorm on concat heads: (B, T, n_heads * d_v)
        self.gate_proj = nn.Linear(config.d_model, config.n_heads * config.d_head_v, bias=False)
        self.out_proj  = nn.Linear(config.n_heads * config.d_head_v, config.d_model, bias=False)

        # GroupNorm: one group per head
        self.group_norm = nn.GroupNorm(config.n_heads, config.n_heads * config.d_head_v,
                                       eps=1e-5, affine=True)

        # Pre-compute decay masks for max_seq_len — shape (H, T, T), registered as buffer
        # so they move to GPU automatically with .to(device)
        D_all = MultiScaleRetention._build_all_decay_masks(config.max_seq_len, config.gammas)
        self.register_buffer("decay_mask_full", D_all)

        self._init_weights()

    def _init_weights(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.gate_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)

    # ── Parallel Mode (Training) ──────────────────────────────────────────────

    @staticmethod
    def _build_all_decay_masks(T: int, gammas=None) -> torch.Tensor:
        """
        Pre-build all H causal decay matrices at once.
        Returns (H, T, T) float32 tensor.
        gammas: list of H floats. If None, uses default RetNetConfig gammas.
        """
        if gammas is None:
            gammas = [1.0 - 2.0 ** (-(5 + h)) for h in range(6)]
        H = len(gammas)
        i_idx = torch.arange(T).unsqueeze(1).float()   # (T, 1)
        j_idx = torch.arange(T).unsqueeze(0).float()   # (1, T)
        diff = i_idx - j_idx                            # (T, T)
        causal = (diff >= 0).float()                    # (T, T)

        # Stack for all heads: (H, T, T)
        gamma_t = torch.tensor(gammas, dtype=torch.float32).view(H, 1, 1)  # (H,1,1)
        # Clamp diff to [0, T] to avoid negative exponents (those are masked to 0 anyway)
        diff_clamped = diff.clamp(min=0).unsqueeze(0)   # (1, T, T)
        D = (gamma_t ** diff_clamped) * causal.unsqueeze(0)  # (H, T, T)

        row_sum = D.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # (H, T, 1)
        D_norm = D / row_sum.sqrt()                             # (H, T, T)
        return D_norm  # float32

    def _get_decay_mask(self, T: int, dtype: torch.dtype) -> torch.Tensor:
        """Slice pre-computed mask to length T and cast to model dtype."""
        return self.decay_mask_full[:, :T, :T].to(dtype)  # (H, T, T)

    def forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parallel (training) mode — fully vectorized over all heads.
        x: (B, T, d_model)
        Returns: (B, T, d_model)
        """
        B, T, _ = x.shape
        dtype = x.dtype

        # Project Q, K, V
        Q = self.q_proj(x).view(B, T, self.n_heads, self.d_qk)  # (B, T, H, d_qk)
        K = self.k_proj(x).view(B, T, self.n_heads, self.d_qk)
        V = self.v_proj(x).view(B, T, self.n_heads, self.d_v)

        # Apply xPos (float32 internally, returns to original dtype)
        Q = apply_xpos(Q)
        K = apply_xpos_key(K)

        # (B, H, T, d)
        Q = Q.permute(0, 2, 1, 3).contiguous()
        K = K.permute(0, 2, 1, 3).contiguous()
        V = V.permute(0, 2, 1, 3).contiguous()

        scale = self.d_qk ** -0.5

        # Batched scores over all heads: (B, H, T, T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, T, T)

        # Apply pre-computed causal decay mask (H, T, T) → broadcast over B
        D = self._get_decay_mask(T, dtype)          # (H, T, T)
        scores = scores * D.unsqueeze(0)            # (B, H, T, T)

        # Weighted sum over values: (B, H, T, d_v)
        out = torch.matmul(scores, V)               # (B, H, T, d_v)

        # Reshape to (B, T, H*d_v) for GroupNorm
        out = out.permute(0, 2, 1, 3).contiguous()            # (B, T, H, d_v)
        concat = out.view(B, T, self.n_heads * self.d_v)       # (B, T, H*d_v)

        # GroupNorm: input (B, H*d_v, T), output (B, H*d_v, T)
        concat_normed = self.group_norm(concat.permute(0, 2, 1)).permute(0, 2, 1)

        # Gated output
        gate = F.silu(self.gate_proj(x))        # (B, T, H*d_v)
        gated = concat_normed * gate

        return self.out_proj(gated)             # (B, T, d_model)

    # ── Recurrent Mode (Inference) ────────────────────────────────────────────

    def forward_recurrent(
        self,
        x: torch.Tensor,
        states: Optional[List[torch.Tensor]],
        position: int = 0,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Recurrent (inference) mode — single step.
        x:       (B, 1, d_model)  — one token at a time
        states:  list of H tensors, each (B, d_qk, d_v) — per-head recurrent state s_t
        position: current position index (for xPos)
        Returns: output (B, 1, d_model), new_states list
        """
        B = x.shape[0]
        device, dtype = x.device, x.dtype

        Q = self.q_proj(x).view(B, 1, self.n_heads, self.d_qk)  # (B, 1, H, d_qk)
        K = self.k_proj(x).view(B, 1, self.n_heads, self.d_qk)
        V = self.v_proj(x).view(B, 1, self.n_heads, self.d_v)

        # Apply xPos at single position
        Q = apply_xpos(Q, offset=position)
        K = apply_xpos_key(K, offset=position)

        # Squeeze time dim
        Q = Q.squeeze(1)  # (B, H, d_qk)
        K = K.squeeze(1)
        V = V.squeeze(1)  # (B, H, d_v)

        if states is None:
            states = [torch.zeros(B, self.d_qk, self.d_v, device=device, dtype=dtype)
                      for _ in range(self.n_heads)]

        new_states = []
        head_outputs = []

        for h in range(self.n_heads):
            q_h = Q[:, h]           # (B, d_qk)
            k_h = K[:, h]           # (B, d_qk)
            v_h = V[:, h]           # (B, d_v)
            gamma = self.gammas[h]
            s_prev = states[h]      # (B, d_qk, d_v)

            # s_t = γ * s_{t-1} + k_t^T ⊗ v_t
            # k_t^T shape: (B, d_qk, 1), v_t shape: (B, 1, d_v)
            # outer product: (B, d_qk, d_v)
            s_t = gamma * s_prev + torch.bmm(k_h.unsqueeze(2), v_h.unsqueeze(1))
            new_states.append(s_t)

            # o_t = q_t * s_t  →  (B, d_qk) @ (B, d_qk, d_v) = (B, d_v)
            o_t = torch.bmm(q_h.unsqueeze(1), s_t).squeeze(1)  # (B, d_v)
            head_outputs.append(o_t)

        # Concatenate: (B, H*d_v) → add time dim for GroupNorm
        concat = torch.cat(head_outputs, dim=-1).unsqueeze(1)  # (B, 1, H*d_v)
        concat_normed = self.group_norm(concat.permute(0, 2, 1)).permute(0, 2, 1)

        gate = F.silu(self.gate_proj(x.squeeze(1))).unsqueeze(1)  # (B, 1, H*d_v)
        gated = concat_normed * gate

        out = self.out_proj(gated)  # (B, 1, d_model)
        return out, new_states

    def forward(
        self,
        x: torch.Tensor,
        recurrent_states: Optional[List[torch.Tensor]] = None,
        position: int = 0,
    ):
        if recurrent_states is not None or x.shape[1] == 1:
            return self.forward_recurrent(x, recurrent_states, position)
        else:
            return self.forward_parallel(x)


# ─────────────────────────────────────────────────────────────────────────────
# Feed-Forward Network
# ─────────────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, config: RetNetConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ffn)
        self.fc2 = nn.Linear(config.d_ffn, config.d_model)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


# ─────────────────────────────────────────────────────────────────────────────
# RetNet Block
# ─────────────────────────────────────────────────────────────────────────────

class RetNetBlock(nn.Module):
    """
    Single RetNet layer:
      x → LayerNorm → MultiScaleRetention → + residual
        → LayerNorm → FeedForward → + residual
    """
    def __init__(self, config: RetNetConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.retention = MultiScaleRetention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        recurrent_states: Optional[List[torch.Tensor]] = None,
        position: int = 0,
    ):
        # Retention sub-layer
        normed = self.ln1(x)
        ret_out = self.retention(normed, recurrent_states, position)

        if isinstance(ret_out, tuple):
            ret_out, new_states = ret_out
        else:
            new_states = None

        x = x + ret_out

        # FFN sub-layer
        x = x + self.ffn(self.ln2(x))

        if new_states is not None:
            return x, new_states
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full RetNet Language Model
# ─────────────────────────────────────────────────────────────────────────────

class RetNetLM(nn.Module):
    """
    124M Retentive Network Language Model.

    Training:  model(input_ids)            → logits (B, T, vocab_size)
    Inference: model.generate(...)         → token ids using recurrent mode
    """

    def __init__(self, config: Optional[RetNetConfig] = None):
        super().__init__()
        if config is None:
            config = RetNetConfig()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([RetNetBlock(config) for _ in range(config.n_layers)])
        self.final_ln = nn.LayerNorm(config.d_model)

        # LM head — weight-tied to embedding
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # weight tying

        self._init_weights()
        print(f"[RetNet] Model initialized: {self.num_parameters():,} parameters")

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for block in self.blocks:
            # Scale residual projections by 1/sqrt(n_layers) for stability
            scale = (2 * self.config.n_layers) ** -0.5
            nn.init.normal_(block.retention.out_proj.weight, std=scale * 0.02)
            nn.init.normal_(block.ffn.fc2.weight, std=scale * 0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Parallel (training) forward pass.
        input_ids: (B, T) — token ids
        Returns: logits (B, T, vocab_size)
        """
        x = self.embedding(input_ids)  # (B, T, d_model)

        for block in self.blocks:
            x = block(x)  # parallel mode returns tensor directly

        x = self.final_ln(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

    def forward_recurrent_step(
        self,
        token_id: torch.Tensor,
        all_states: Optional[List[List[torch.Tensor]]],
        position: int,
    ) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        """
        Single-step recurrent forward for inference.
        token_id:   (B,) — single token
        all_states: list[n_layers] of list[n_heads] of (B, d_qk, d_v) tensors
        position:   current sequence position
        Returns: logits (B, vocab_size), new_all_states
        """
        x = self.embedding(token_id.unsqueeze(1))  # (B, 1, d_model)

        new_all_states = []
        for i, block in enumerate(self.blocks):
            states_i = all_states[i] if all_states is not None else None
            x, new_states_i = block(x, recurrent_states=states_i, position=position)
            new_all_states.append(new_states_i)

        x = self.final_ln(x.squeeze(1))     # (B, d_model)
        logits = self.lm_head(x)            # (B, vocab_size)
        return logits, new_all_states

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.9,
        repetition_penalty: float = 1.3,
        ngram_block: int = 3,
    ) -> torch.Tensor:
        """
        Autoregressive generation using recurrent mode (O(1) memory per step).
        prompt_ids: (B, T_prompt) — context tokens
        Returns: (B, T_prompt + max_new_tokens) — full token sequence
        """
        self.eval()
        device = prompt_ids.device
        B, T_prompt = prompt_ids.shape

        # Prefill: process prompt in parallel for efficiency, then switch to recurrent
        # For simplicity here we process token by token (correct recurrent mode)
        all_states = None
        generated = prompt_ids.clone()

        # Prefill phase: feed prompt tokens step by step to build state
        for t in range(T_prompt):
            token = generated[:, t]
            _, all_states = self.forward_recurrent_step(token, all_states, position=t)

        # Generation phase
        ngram_cache = {b: [] for b in range(B)}  # track generated token history

        for step in range(max_new_tokens):
            last_token = generated[:, -1]
            logits, all_states = self.forward_recurrent_step(
                last_token, all_states, position=T_prompt + step
            )
            # logits: (B, vocab_size)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(B):
                    for tid in set(generated[b].tolist()):
                        if logits[b, tid] > 0:
                            logits[b, tid] /= repetition_penalty
                        else:
                            logits[b, tid] *= repetition_penalty

            # N-gram blocking
            if ngram_block > 0 and generated.shape[1] >= ngram_block:
                for b in range(B):
                    last_ngram = tuple(generated[b, -(ngram_block-1):].tolist())
                    tokens_in_history = generated[b].tolist()
                    blocked = set()
                    for i in range(len(tokens_in_history) - ngram_block + 1):
                        if tuple(tokens_in_history[i:i+ngram_block-1]) == last_ngram:
                            blocked.add(tokens_in_history[i+ngram_block-1])
                    for tid in blocked:
                        logits[b, tid] = float('-inf')

            # Temperature scaling
            logits = logits / max(temperature, 1e-5)

            # Top-p (nucleus) sampling
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cum_probs = sorted_probs.cumsum(dim=-1)
            # Remove tokens with cumulative probability above top_p
            sorted_probs[cum_probs - sorted_probs > top_p] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)

            next_token_sorted = torch.multinomial(sorted_probs, num_samples=1).squeeze(1)
            next_token = sorted_idx.gather(1, next_token_sorted.unsqueeze(1)).squeeze(1)

            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

        return generated

    @torch.no_grad()
    def generate_parallel(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.9,
        repetition_penalty: float = 1.3,
        ngram_block: int = 3,
    ) -> torch.Tensor:
        """
        Autoregressive generation using the PARALLEL forward at each step.
        Uses the same GroupNorm statistics as training (computed over full T),
        so generated text matches what the model actually learned.
        Slower than recurrent mode (O(T^2)) but correct for this checkpoint.
        prompt_ids: (B, T_prompt)
        Returns: (B, T_prompt + max_new_tokens)
        """
        self.eval()
        generated = prompt_ids.clone()

        for step in range(max_new_tokens):
            # Full parallel forward on the growing sequence.
            # MultiScaleRetention routes to recurrent mode when T==1, which
            # returns a tuple and breaks LayerNorm in the next block.
            # Pad to ≥2 tokens to force parallel mode; logits[:,-1] is still correct.
            seq = generated
            if seq.shape[1] == 1:
                seq = torch.cat([torch.zeros_like(seq), seq], dim=1)
            logits = self.forward(seq)                # (B, T, vocab)
            next_logits = logits[:, -1, :].float()   # (B, vocab) — last position

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(generated.shape[0]):
                    for tid in set(generated[b].tolist()):
                        if next_logits[b, tid] > 0:
                            next_logits[b, tid] /= repetition_penalty
                        else:
                            next_logits[b, tid] *= repetition_penalty

            # N-gram blocking
            if ngram_block > 0 and generated.shape[1] >= ngram_block:
                for b in range(generated.shape[0]):
                    last_ng = tuple(generated[b, -(ngram_block - 1):].tolist())
                    hist    = generated[b].tolist()
                    blocked = {hist[i + ngram_block - 1]
                               for i in range(len(hist) - ngram_block + 1)
                               if tuple(hist[i:i + ngram_block - 1]) == last_ng}
                    for tid in blocked:
                        next_logits[b, tid] = float('-inf')

            # Temperature + top-p nucleus sampling
            next_logits = next_logits / max(temperature, 1e-5)
            probs = F.softmax(next_logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = sorted_probs.cumsum(dim=-1)
            sorted_probs[cum - sorted_probs > top_p] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
            next_tok = sorted_idx.gather(1, torch.multinomial(sorted_probs, 1))
            generated = torch.cat([generated, next_tok], dim=1)

        return generated


# ─────────────────────────────────────────────────────────────────────────────
# Quick Sanity Check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = RetNetConfig()
    print(f"[Config] Gammas: {[f'{g:.6f}' for g in config.gammas]}")

    model = RetNetLM(config)

    # Test parallel forward
    B, T = 2, 64
    ids = torch.randint(0, config.vocab_size, (B, T))
    with torch.no_grad():
        logits = model(ids)
    print(f"[Parallel] Input {tuple(ids.shape)} -> Logits {tuple(logits.shape)}")

    # Test recurrent forward
    model.eval()
    states = None
    for t in range(4):
        tok = torch.randint(0, config.vocab_size, (B,))
        with torch.no_grad():
            logits_r, states = model.forward_recurrent_step(tok, states, t)
    print(f"[Recurrent] Step logits shape: {tuple(logits_r.shape)}")

    print("[OK] Architecture verified.")
