"""
Accessible RetNet — Generation & Evaluation Script.

Loads a trained checkpoint and:
1. Evaluates perplexity on WikiText-103 test set.
2. Generates sample text using recurrent mode (O(1) memory).
3. Demonstrates the "perplexity-quality mismatch" (low PPL, imperfect semantics).

Usage:
  python generate.py --eval                    # evaluate test perplexity
  python generate.py --generate "The king"     # generate from prompt
  python generate.py --eval --generate "The"   # both
  python generate.py --checkpoint path/to.pt   # specify checkpoint
"""

import sys
import math
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F

from retnet_model import RetNetLM, RetNetConfig

# ─────────────────────────────────────────────────────────────────────────────
# Load tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def get_tokenizer():
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


# ─────────────────────────────────────────────────────────────────────────────
# Load model from checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Path, device: torch.device) -> RetNetLM:
    print(f"[Generate] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    config = RetNetConfig()
    if "config" in ckpt:
        cfg_dict = ckpt["config"]
        for k, v in cfg_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)

    model = RetNetLM(config).to(device)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    val_ppl  = ckpt.get("val_ppl",  "?")
    test_ppl = ckpt.get("test_ppl", "?")
    step     = ckpt.get("step", "?")
    print(f"  Loaded step={step} | Best Val PPL={val_ppl} | Test PPL={test_ppl}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Perplexity evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_perplexity(model: RetNetLM, device: torch.device) -> float:
    from dataset import get_dataloaders
    from torch.cuda.amp import autocast

    print("[Generate] Loading WikiText-103 test set for perplexity evaluation...")
    loaders = get_dataloaders(batch_size=8, verbose=False)
    test_loader = loaders["test"]

    model.eval()
    total_loss   = 0.0
    total_tokens = 0
    n_batches    = len(test_loader)

    print(f"  Evaluating {n_batches} batches...")

    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        with autocast():
            logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total_loss   += loss.item()
        total_tokens += y.numel()

        if (i + 1) % 50 == 0:
            running_ppl = math.exp(min(total_loss / total_tokens, 20.0))
            print(f"  [{i+1}/{n_batches}] Running PPL: {running_ppl:.2f}", flush=True)

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20.0))
    return ppl


# ─────────────────────────────────────────────────────────────────────────────
# Text generation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_text(
    model: RetNetLM,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    repetition_penalty: float = 1.3,
    ngram_block: int = 3,
) -> str:
    model.eval()
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    print(f"\n[Generate] Prompt: '{prompt}'")
    print(f"  Params: temp={temperature}, top_p={top_p}, rep_pen={repetition_penalty}, ngram_block={ngram_block}")
    print(f"  Generating {max_new_tokens} tokens (recurrent mode, O(1) memory)...")

    output_ids = model.generate(
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        ngram_block=ngram_block,
    )

    # Decode only the generated portion
    generated_ids = output_ids[0, len(prompt_ids[0]):].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Generate] Device: {device}")

    # Resolve checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_dir = Path(__file__).parent / "checkpoints"
        # Prefer best_model, then latest, then final
        for name in ["best_model.pt", "final_model.pt", "latest.pt"]:
            candidate = ckpt_dir / name
            if candidate.exists():
                ckpt_path = candidate
                break
        else:
            print("[Error] No checkpoint found in ./checkpoints/")
            print("        Run train.py first, or specify --checkpoint path")
            sys.exit(1)

    model = load_model(ckpt_path, device)

    # ── Perplexity Evaluation ──
    if args.eval:
        print("\n" + "="*55)
        print("  PERPLEXITY EVALUATION (WikiText-103 Test Set)")
        print("="*55)
        ppl = evaluate_perplexity(model, device)
        print(f"\n  Test Perplexity: {ppl:.2f}")
        print(f"  Target (paper):  ~15.2")
        if ppl <= 20.0:
            print("  Status: EXCELLENT - within target range!")
        elif ppl <= 30.0:
            print("  Status: GOOD - close to target")
        else:
            print("  Status: May need more training steps")
        print("="*55)

    # ── Text Generation ──
    if args.generate:
        tokenizer = get_tokenizer()
        prompts = args.generate if isinstance(args.generate, list) else [args.generate]

        print("\n" + "="*55)
        print("  TEXT GENERATION (Recurrent Mode)")
        print("  NOTE: Expect 'perplexity-quality mismatch' —")
        print("  low PPL but potential semantic drift without RLHF")
        print("="*55)

        for prompt in prompts:
            text = generate_text(
                model, tokenizer, prompt, device,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.rep_penalty,
                ngram_block=args.ngram_block,
            )
            print(f"\n[Prompt]    {prompt}")
            print(f"[Generated] {text}")
            print("-"*55)

    # ── Demo: multiple temperatures ──
    if args.demo:
        tokenizer = get_tokenizer()
        prompt = "The history of artificial intelligence"
        print("\n" + "="*55)
        print(f"  TEMPERATURE DEMO | Prompt: '{prompt}'")
        print("="*55)
        for temp in [0.5, 0.8, 1.0, 1.2]:
            text = generate_text(
                model, tokenizer, prompt, device,
                max_new_tokens=80, temperature=temp, top_p=0.9,
                repetition_penalty=1.3, ngram_block=3,
            )
            print(f"\n  [temp={temp}] {text[:200]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RetNet Generation & Evaluation")
    parser.add_argument("--eval",       action="store_true",  help="Evaluate test perplexity")
    parser.add_argument("--generate",   type=str, nargs="+",  help="Prompt(s) for text generation")
    parser.add_argument("--demo",       action="store_true",  help="Run temperature sweep demo")
    parser.add_argument("--checkpoint", type=str,             help="Path to checkpoint .pt file")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max new tokens (default: 200)")
    parser.add_argument("--temperature",type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    parser.add_argument("--top-p",      type=float, default=0.9,  help="Nucleus sampling p (default: 0.9)")
    parser.add_argument("--rep-penalty",type=float, default=1.3,  help="Repetition penalty (default: 1.3)")
    parser.add_argument("--ngram-block",type=int,   default=3,    help="N-gram blocking n (default: 3)")
    args = parser.parse_args()

    if not (args.eval or args.generate or args.demo):
        print("Specify at least one of: --eval, --generate 'prompt', --demo")
        parser.print_help()
        sys.exit(1)

    main(args)
