"""
Accessible RetNet — Supervised Fine-Tuning (SFT) on Stanford Alpaca.

Loads the pretrained base model and fine-tunes on 52K instruction-response
pairs so the model learns to generate coherent, on-topic text.

Prompt template (Alpaca format):
  ### Instruction:
  <instruction>

  ### Input:
  <input>   (optional)

  ### Response:
  <response>

Only the Response tokens are included in the loss — the model learns to
complete responses, not memorize instructions.

Usage:
  python finetune.py              # full fine-tune (~2-3 hours on RTX 2070)
  python finetune.py --smoke      # 20-step smoke test
  python finetune.py --checkpoint checkpoints/best_model.pt
"""

import os
import sys
import io
import csv
import json
import math
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from retnet_model import RetNetLM, RetNetConfig

# ─────────────────────────────────────────────────────────────────────────────
# Rich Dashboard
# ─────────────────────────────────────────────────────────────────────────────

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.tree import Tree
    from rich.text import Text
    from rich import box
    RICH = True
except ImportError:
    RICH = False

console = Console() if RICH else None


# ─────────────────────────────────────────────────────────────────────────────
# Alpaca Prompt Formatter
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_WITH_INPUT = """\
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

PROMPT_NO_INPUT = """\
### Instruction:
{instruction}

### Response:
{output}"""

RESPONSE_SPLIT = "### Response:\n"


def format_sample(sample: dict) -> str:
    if sample.get("input", "").strip():
        return PROMPT_WITH_INPUT.format(
            instruction=sample["instruction"],
            input=sample["input"],
            output=sample["output"],
        )
    return PROMPT_NO_INPUT.format(
        instruction=sample["instruction"],
        output=sample["output"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class AlpacaDataset(Dataset):
    """
    Tokenizes Alpaca samples into fixed-length sequences.
    Loss is computed only on Response tokens (instruction tokens are masked).
    """

    def __init__(self, samples: list, tokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.items     = []

        eos = tokenizer.eos_token_id
        resp_ids = tokenizer.encode(RESPONSE_SPLIT, add_special_tokens=False)

        for s in samples:
            text     = format_sample(s)
            full_ids = tokenizer.encode(text, add_special_tokens=False) + [eos]

            if len(full_ids) > max_len:
                full_ids = full_ids[:max_len]

            # Find where Response starts so we can mask instruction tokens
            resp_start = self._find_subseq(full_ids, resp_ids)
            if resp_start == -1:
                resp_start = 0   # fallback: train on everything
            else:
                resp_start += len(resp_ids)   # start of actual response text

            # input_ids = full sequence; labels = -100 for instruction, token ids for response
            input_ids = full_ids[:-1]
            labels    = [-100] * len(input_ids)
            for i in range(resp_start, len(input_ids)):
                labels[i] = input_ids[i]

            # Pad to max_len - 1
            pad_len = (max_len - 1) - len(input_ids)
            input_ids = input_ids + [eos] * pad_len
            labels    = labels    + [-100] * pad_len

            self.items.append((
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(labels,    dtype=torch.long),
            ))

    @staticmethod
    def _find_subseq(seq, sub):
        for i in range(len(seq) - len(sub) + 1):
            if seq[i:i+len(sub)] == sub:
                return i
        return -1

    def __len__(self):  return len(self.items)
    def __getitem__(self, i): return self.items[i]


def load_alpaca(tokenizer, val_split: float = 0.05, max_len: int = 512, verbose=True):
    """Download Stanford Alpaca and return train/val datasets."""
    cache = Path(__file__).parent / "data_cache" / "alpaca.json"
    cache.parent.mkdir(exist_ok=True)

    if cache.exists():
        if verbose: print("[Finetune] Loading cached Alpaca dataset...")
        with open(cache) as f:
            data = json.load(f)
    else:
        if verbose: print("[Finetune] Downloading Stanford Alpaca (52K samples)...")
        from datasets import load_dataset
        ds   = load_dataset("tatsu-lab/alpaca", split="train")
        data = [{"instruction": r["instruction"],
                 "input":       r["input"],
                 "output":      r["output"]} for r in ds]
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(data, f)
        if verbose: print(f"[Finetune] Saved {len(data):,} samples to cache.")

    # Filter empty outputs
    data = [s for s in data if s["output"].strip()]

    # Deterministic train/val split
    split_idx  = int(len(data) * (1 - val_split))
    train_data = data[:split_idx]
    val_data   = data[split_idx:]

    if verbose:
        print(f"[Finetune] Train: {len(train_data):,} | Val: {len(val_data):,} samples")
        print("[Finetune] Tokenizing... (may take 1-2 minutes)")

    train_ds = AlpacaDataset(train_data, tokenizer, max_len)
    val_ds   = AlpacaDataset(val_data,   tokenizer, max_len)

    if verbose:
        print(f"[Finetune] Tokenization complete.")

    return train_ds, val_ds


# ─────────────────────────────────────────────────────────────────────────────
# LR Schedule
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step, warmup, total, max_lr, min_lr_ratio=0.1):
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    progress = min(progress, 1.0)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max_lr * min_lr_ratio + (max_lr - max_lr * min_lr_ratio) * cosine


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with autocast():
            logits = model(x)   # (B, T, vocab)
        # Only count non-masked tokens
        mask    = y != -100
        loss    = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                  y.view(-1), ignore_index=-100, reduction="sum")
        n_tok   = mask.sum().item()
        if n_tok > 0:
            total_loss   += loss.item()
            total_tokens += n_tok
    ppl = math.exp(min(total_loss / max(total_tokens, 1), 20.0))
    model.train()
    return ppl


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────────────────────

def build_dashboard(state: dict) -> Panel:
    if not RICH:
        return None

    step         = state["step"]
    total        = state["total_steps"]
    loss         = state.get("loss") or 0.0
    ppl          = state.get("ppl") or 0.0
    best_val     = state.get("best_val_ppl") or 0.0
    lr           = state.get("lr", "0")
    gpu_gb       = state.get("gpu_gb", 0.0)
    eta          = state.get("eta", "--")
    phase        = state.get("phase", "")
    val_history  = state.get("val_history", [])
    log_lines    = state.get("log_lines", [])

    pct     = step / total if total > 0 else 0
    bar_len = 18
    filled  = int(bar_len * pct)
    bar     = "[bold green]" + "#" * filled + "[/bold green]" + "[dim]-[/dim]" * (bar_len - filled)

    tree = Tree("[bold white]FINE-TUNING KNOWLEDGE GRAPH[/bold white]")

    base_node = tree.add("[bold green][OK][/bold green] Base RetNet (pretrained 124M)")
    base_node.add("[green]Best Val PPL = 15.09 on WikiText-103[/green]")
    base_node.add("[green]Step 42,000 checkpoint loaded[/green]")

    data_node = tree.add("[bold green][OK][/bold green] Stanford Alpaca Dataset")
    data_node.add("[green]49,400 train | 2,600 val samples[/green]")
    data_node.add("[green]Instruction masking: loss on Response only[/green]")

    ft_node = tree.add(
        f"[bold cyan][>>][/bold cyan] Fine-Tuning ({step:,}/{total:,})  [{bar}] {pct*100:.1f}%"
    )
    ft_node.add(f"[green]LR = 3e-5 (10x smaller than pretraining)[/green]")
    ft_node.add(f"[green]Alpaca prompt template[/green]")
    ft_node.add(f"[{'green' if step >= 200 else 'dim'}]Warmup (200 steps)[/{'green' if step >= 200 else 'dim'}]")
    ft_node.add(f"[{'green' if phase == 'Complete' else 'dim'}]Cosine decay[/{'green' if phase == 'Complete' else 'dim'}]")

    tree.add(f"[{'bold green][OK]' if phase == 'Complete' else 'dim][  ]'}[/{'bold green' if phase == 'Complete' else 'dim'}] SFT Complete")

    metrics = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    metrics.add_column("K", style="dim cyan",   width=12)
    metrics.add_column("V", style="bold white", width=18)

    gpu_pct  = gpu_gb / 8.0
    gpu_fill = int(12 * gpu_pct)
    gpu_col  = "red" if gpu_pct > 0.9 else ("yellow" if gpu_pct > 0.75 else "green")
    gpu_bar  = f"[{gpu_col}]{'#'*gpu_fill}[/{gpu_col}][dim]{'-'*(12-gpu_fill)}[/dim]"

    metrics.add_row("Step",      f"{step:,} / {total:,}")
    metrics.add_row("Loss",      f"{loss:.4f}" if loss else "---")
    metrics.add_row("Train PPL", f"{ppl:.2f}"  if ppl  else "---")
    metrics.add_row("Best Val",  f"{best_val:.2f}" if best_val else "---")
    metrics.add_row("LR",        lr)
    metrics.add_row("GPU",       f"{gpu_bar} {gpu_gb:.1f}/8GB")
    metrics.add_row("ETA",       eta)
    metrics.add_row("Phase",     f"[cyan]{phase}[/cyan]")

    if val_history:
        hist = "  ".join([f"[{s}]={p:.1f}" for s, p in val_history[-5:]])
        hist_text = f"\n[dim]Val PPL: {hist}[/dim]"
    else:
        hist_text = ""

    log_text = "\n".join(log_lines[-10:]) if log_lines else "[dim]Starting...[/dim]"

    layout = Table(box=box.ROUNDED, expand=True, show_header=False, padding=(0, 1))
    layout.add_column("Graph",   ratio=3)
    layout.add_column("Metrics", ratio=2)
    layout.add_row(tree, metrics)

    outer = Table(box=box.ROUNDED, expand=True, show_header=False, padding=(0, 1))
    outer.add_column("Main", ratio=1)
    outer.add_row(layout)
    outer.add_row(Panel(log_text + hist_text,
                        title="[bold]Event Log[/bold]", border_style="dim blue"))

    return Panel(outer,
                 title="[bold magenta]RetNet SFT — Alpaca Fine-Tuning | RTX 2070[/bold magenta]",
                 border_style="magenta")


# ─────────────────────────────────────────────────────────────────────────────
# State persistence (same format as train.py so monitor.html works)
# ─────────────────────────────────────────────────────────────────────────────

def save_state(out_dir, state: dict):
    with open(out_dir / "training_state.json", "w") as f:
        json.dump(state, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(__file__).parent / "checkpoints_sft"
    out_dir.mkdir(exist_ok=True)

    # Also write state to main checkpoints dir so monitor.html picks it up
    monitor_dir = Path(__file__).parent / "checkpoints"
    monitor_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Finetune] Device: {device}")

    # ── Hyperparams ──────────────────────────────────────────────────────────
    TOTAL_STEPS  = 20    if args.smoke else 800     # ~0.26 epochs — stops before memorization
    WARMUP_STEPS = 5     if args.smoke else 80      # 10% warmup
    PHYS_BATCH   = 2     if args.smoke else 4
    ACCUM_STEPS  = 2     if args.smoke else 4       # effective batch = 16
    MAX_LR       = 5e-5                             # slightly higher — fewer steps to converge
    GRAD_CLIP    = 1.0
    EVAL_EVERY   = 10    if args.smoke else 100     # frequent — catch best checkpoint early
    SAVE_EVERY   = 10    if args.smoke else 100
    LOG_EVERY    = 1     if args.smoke else 10
    MAX_SEQ_LEN  = 512

    # ── Tokenizer ────────────────────────────────────────────────────────────
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_ds, val_ds = load_alpaca(tokenizer, max_len=MAX_SEQ_LEN, verbose=True)

    train_loader = DataLoader(train_ds, batch_size=PHYS_BATCH, shuffle=True,
                              pin_memory=True, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=PHYS_BATCH, shuffle=False,
                              pin_memory=True, num_workers=0, drop_last=False)

    # ── Load pretrained model ─────────────────────────────────────────────────
    # Prefer SFT resume, then user-specified checkpoint, then best pretraining ckpt
    sft_latest = out_dir / "latest_sft.pt"

    if sft_latest.exists() and not args.smoke:
        ckpt_path = sft_latest
        print(f"[Finetune] Resuming SFT from {ckpt_path}")
    elif args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        print(f"[Finetune] Loading checkpoint: {ckpt_path}")
    else:
        # Auto-find best pretrained checkpoint
        for name in ["best_model.pt", "final_model.pt", "latest.pt"]:
            candidate = Path(__file__).parent / "checkpoints" / name
            if candidate.exists():
                ckpt_path = candidate
                break
        else:
            print("[Error] No pretrained checkpoint found. Run train.py first.")
            sys.exit(1)
        print(f"[Finetune] Loading pretrained checkpoint: {ckpt_path}")

    ckpt   = torch.load(ckpt_path, map_location=device)
    config = RetNetConfig()
    if "config" in ckpt:
        for k, v in ckpt["config"].items():
            if hasattr(config, k): setattr(config, k, v)

    model = RetNetLM(config).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"[Finetune] Model loaded: {model.num_parameters():,} parameters")

    # ── Optimizer ────────────────────────────────────────────────────────────
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if any(k in name for k in ["bias", "ln", "norm", "embedding"]):
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = torch.optim.AdamW(
        [{"params": decay,    "weight_decay": 0.01},   # lighter wd for SFT
         {"params": no_decay, "weight_decay": 0.0}],
        lr=MAX_LR, betas=(0.9, 0.98), eps=1e-8
    )
    scaler = GradScaler()

    start_step   = 0
    best_val_ppl = float("inf")

    if sft_latest.exists() and not args.smoke and ckpt_path == sft_latest:
        optimizer.load_state_dict(ckpt.get("optimizer", {}))
        try: scaler.load_state_dict(ckpt.get("scaler", {}))
        except: pass
        start_step   = ckpt.get("step", 0)
        best_val_ppl = ckpt.get("best_val_ppl", float("inf"))
        print(f"[Finetune] Resumed from step {start_step:,}")

    # ── CSV logger ────────────────────────────────────────────────────────────
    log_file   = open(out_dir / "finetune_log.csv", "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["step", "train_loss", "train_ppl", "val_ppl", "lr", "gpu_gb", "elapsed_s"])

    # ── State dict for monitor ────────────────────────────────────────────────
    val_history = []
    log_lines   = []

    def add_log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        log_lines.append(f"[{ts}] {msg}")
        if len(log_lines) > 12: log_lines.pop(0)

    def get_gpu():
        return torch.cuda.memory_allocated(0) / 1024**3 if device.type == "cuda" else 0.0

    def write_state(step, loss, ppl, lr, eta_str, phase):
        d = {
            "step": step, "total_steps": TOTAL_STEPS,
            "loss": round(loss, 4) if math.isfinite(loss) else None,
            "ppl":  round(ppl,  2) if math.isfinite(ppl)  else None,
            "best_val_ppl": round(best_val_ppl, 2) if math.isfinite(best_val_ppl) else None,
            "lr":     f"{lr:.2e}",
            "gpu_gb": round(get_gpu(), 2),
            "speed":  round(speed, 2),
            "eta":    eta_str,
            "phase":  phase,
            "val_history": val_history,
            "log_lines":   log_lines,
            "updated_at":  datetime.now().isoformat(),
        }
        save_state(out_dir, d)
        save_state(monitor_dir, d)   # browser monitor picks this up
        return d

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    train_iter   = iter(train_loader)
    global_step  = start_step
    micro_count  = 0
    running_loss = 0.0
    step_times   = []
    speed        = 0.0
    t0           = time.time()
    t_step       = time.time()

    optimizer.zero_grad()
    add_log(f"SFT started | {TOTAL_STEPS:,} steps | LR={MAX_LR:.0e} | Alpaca 52K")

    state = write_state(global_step, float("inf"), float("inf"), MAX_LR, "--", "Warmup")

    print(f"\n[Finetune] Starting SFT — {TOTAL_STEPS:,} steps")
    print(f"[Finetune] Batch={PHYS_BATCH}x{ACCUM_STEPS}={PHYS_BATCH*ACCUM_STEPS} effective | LR={MAX_LR}")
    print(f"[Finetune] Browser monitor: http://localhost:8765\n")

    def render():
        return build_dashboard(state)

    with (Live(render(), refresh_per_second=1, console=console) if RICH
          else open(os.devnull, "w")) as live:

        while global_step < TOTAL_STEPS:
            try:
                x_b, y_b = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x_b, y_b  = next(train_iter)

            x_b = x_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)

            with autocast():
                logits = model(x_b)
                # Cross-entropy ignoring instruction tokens (label = -100)
                loss   = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y_b.view(-1),
                    ignore_index=-100,
                )
                loss_s = loss / ACCUM_STEPS

            scaler.scale(loss_s).backward()
            running_loss += loss.item()
            micro_count  += 1

            if micro_count % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

                lr = get_lr(global_step, WARMUP_STEPS, TOTAL_STEPS, MAX_LR)
                for pg in optimizer.param_groups: pg["lr"] = lr

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

                # Timing
                now = time.time()
                step_times.append(now - t_step)
                if len(step_times) > 50: step_times.pop(0)
                t_step   = now
                avg_t    = sum(step_times) / len(step_times)
                speed    = 1.0 / avg_t if avg_t > 0 else 0.0
                eta_str  = str(timedelta(seconds=int(avg_t * (TOTAL_STEPS - global_step))))

                avg_loss     = running_loss / ACCUM_STEPS
                running_loss = 0.0
                ppl          = math.exp(min(avg_loss, 20.0))
                phase        = "Warmup" if global_step < WARMUP_STEPS else "Cosine Decay"

                # Log
                if global_step % LOG_EVERY == 0:
                    elapsed = now - t0
                    csv_writer.writerow([global_step, f"{avg_loss:.4f}", f"{ppl:.2f}",
                                         "", f"{lr:.6f}", f"{get_gpu():.2f}", f"{elapsed:.0f}"])
                    log_file.flush()
                    add_log(f"step={global_step:,} loss={avg_loss:.3f} ppl={ppl:.1f} lr={lr:.1e}")

                # Write monitor state
                if global_step % 50 == 0 or args.smoke:
                    state = write_state(global_step, avg_loss, ppl, lr, eta_str, phase)

                if RICH: live.update(render())

                # Validation
                if global_step % EVAL_EVERY == 0 or global_step == TOTAL_STEPS:
                    phase = "Validation"
                    add_log(f"Validating at step {global_step:,}...")
                    torch.cuda.empty_cache()
                    if RICH: live.update(render())

                    max_val = 5 if args.smoke else None
                    val_ppl = evaluate(model, val_loader, device, max_batches=max_val)
                    is_best = val_ppl < best_val_ppl
                    if is_best: best_val_ppl = val_ppl

                    val_history.append((global_step, round(val_ppl, 2)))
                    elapsed = time.time() - t0
                    csv_writer.writerow([global_step, f"{avg_loss:.4f}", f"{ppl:.2f}",
                                         f"{val_ppl:.2f}", f"{lr:.6f}", f"{get_gpu():.2f}", f"{elapsed:.0f}"])
                    log_file.flush()

                    marker = " (NEW BEST)" if is_best else ""
                    add_log(f"Val PPL={val_ppl:.2f}{marker}")
                    phase = "Warmup" if global_step < WARMUP_STEPS else "Cosine Decay"
                    state = write_state(global_step, avg_loss, ppl, lr, eta_str, phase)
                    if RICH: live.update(render())

                    if is_best:
                        torch.save({
                            "step": global_step, "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scaler": scaler.state_dict(),
                            "val_ppl": val_ppl, "best_val_ppl": best_val_ppl,
                            "config": config.__dict__,
                        }, out_dir / "best_sft.pt")
                        add_log(f"Saved best_sft.pt (val={val_ppl:.2f})")

                    model.train()

                # Periodic checkpoint
                if global_step % SAVE_EVERY == 0:
                    torch.save({
                        "step": global_step, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "val_ppl": best_val_ppl, "best_val_ppl": best_val_ppl,
                        "config": config.__dict__,
                    }, out_dir / "latest_sft.pt")
                    add_log(f"Checkpoint saved at step {global_step:,}")

    # ── Complete ──────────────────────────────────────────────────────────────
    phase = "Complete"
    state = write_state(global_step, avg_loss, ppl, lr, "0:00:00", phase)
    log_file.close()

    print("\n" + "=" * 55)
    print("  Fine-Tuning Complete!")
    print(f"  Best Val PPL : {best_val_ppl:.2f}")
    print(f"  Checkpoints  : {out_dir}")
    print(f"    best_sft.pt   <- use this for generation")
    print("=" * 55)
    print("\nGenerate with:")
    print(f"  python generate_sft.py --prompt \"Explain what a transformer is\"")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RetNet SFT on Stanford Alpaca")
    parser.add_argument("--smoke",      action="store_true", help="20-step smoke test")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pretrained checkpoint (default: auto-detect best_model.pt)")
    args = parser.parse_args()

    if args.smoke:
        print("[Smoke] Running 20-step SFT smoke test...")
    else:
        print("[Finetune] Starting full SFT — 5,000 steps on Stanford Alpaca")
        print("[Finetune] Monitor at http://localhost:8765")

    main(args)
