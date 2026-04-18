"""
Accessible RetNet Training Script — 124M Parameters on WikiText-103.

Training Config:
  Optimizer:  AdamW  (beta1=0.9, beta2=0.98, wd=0.05)
  LR:         3e-4, warmup=1000 steps, cosine decay to 50000 steps
  Precision:  FP16 (torch.cuda.amp)
  Grad clip:  1.0
  Phys batch: 8 x 4 accum = effective batch 32
  Checkpoints every 2000 steps + best model saved

Run: python train.py
Run smoke test: python train.py --smoke
"""

import os
import sys
import csv
import json
import math
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from retnet_model import RetNetLM, RetNetConfig
from dataset import get_dataloaders

# ─────────────────────────────────────────────────────────────────────────────
# Rich Dashboard
# ─────────────────────────────────────────────────────────────────────────────

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import (
        Progress, BarColumn, TextColumn, TimeRemainingColumn,
        MofNCompleteColumn, SpinnerColumn
    )
    from rich.layout import Layout
    from rich.text import Text
    from rich.tree import Tree
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None


# ─────────────────────────────────────────────────────────────────────────────
# Training State (written to JSON every 100 steps)
# ─────────────────────────────────────────────────────────────────────────────

class TrainingState:
    def __init__(self, total_steps: int, out_dir: Path):
        self.total_steps   = total_steps
        self.out_dir       = out_dir
        self.step          = 0
        self.loss          = float("inf")
        self.ppl           = float("inf")
        self.best_val_ppl  = float("inf")
        self.lr            = 0.0
        self.gpu_gb        = 0.0
        self.gpu_total     = 8.0
        self.speed         = 0.0         # steps/sec
        self.eta_str       = "--"
        self.phase         = "Initializing"
        self.val_history   = []          # list of (step, val_ppl)
        self.log_lines     = []          # last 12 events
        self.start_time    = time.time()

        # Pipeline status flags
        self.env_done      = True
        self.data_done     = False
        self.model_done    = False
        self.warmup_done   = False
        self.training_active = False
        self.eval_done     = False
        self.gen_done      = False

    def add_log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_lines.append(f"[{ts}] {msg}")
        if len(self.log_lines) > 12:
            self.log_lines.pop(0)

    def save_json(self):
        d = {
            "step": self.step,
            "total_steps": self.total_steps,
            "loss": round(self.loss, 4) if math.isfinite(self.loss) else None,
            "ppl": round(self.ppl, 2) if math.isfinite(self.ppl) else None,
            "best_val_ppl": round(self.best_val_ppl, 2) if math.isfinite(self.best_val_ppl) else None,
            "lr": f"{self.lr:.2e}",
            "gpu_gb": round(self.gpu_gb, 2),
            "speed": round(self.speed, 2),
            "eta": self.eta_str,
            "phase": self.phase,
            "val_history": self.val_history,
            "updated_at": datetime.now().isoformat(),
        }
        with open(self.out_dir / "training_state.json", "w") as f:
            json.dump(d, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard Renderer
# ─────────────────────────────────────────────────────────────────────────────

def build_dashboard(state: TrainingState) -> Panel:
    """Build the Rich dashboard layout."""
    if not RICH_AVAILABLE:
        return None

    def status_icon(done: bool, active: bool = False) -> str:
        if active: return "[bold cyan][>>][/bold cyan]"
        if done:   return "[bold green][OK][/bold green]"
        return "[dim][  ][/dim]"

    # ── Knowledge Graph Tree ──
    tree = Tree("[bold white]PIPELINE KNOWLEDGE GRAPH[/bold white]")

    env_node = tree.add(f"{status_icon(state.env_done)} Environment Setup")
    env_node.add("[green]CUDA 11.8 + PyTorch 2.1.2[/green]")
    env_node.add("[green]RTX 2070 | 8GB VRAM | CC 7.5[/green]")

    data_node = tree.add(f"{status_icon(state.data_done)} Dataset (WikiText-103)")
    if state.data_done:
        data_node.add("[green]~103M tokens tokenized[/green]")
        data_node.add("[green]512-token chunks built[/green]")
    else:
        data_node.add("[yellow]Downloading / Tokenizing...[/yellow]")

    model_node = tree.add(f"{status_icon(state.model_done)} RetNet Model (124M params)")
    if state.model_done:
        model_node.add("[green]xPos encoding[/green]")
        model_node.add("[green]6-head multi-scale retention[/green]")
        model_node.add("[green]12 blocks + weight tying[/green]")

    training_active = state.training_active and not state.eval_done
    train_pct = state.step / state.total_steps if state.total_steps > 0 else 0
    bar_len = 16
    filled = int(bar_len * train_pct)
    bar = "[bold green]" + "#" * filled + "[/bold green]" + "[dim]-[/dim]" * (bar_len - filled)
    train_node = tree.add(
        f"{status_icon(state.eval_done, training_active)} Training "
        f"({state.step:,}/{state.total_steps:,})  [{bar}] {train_pct*100:.1f}%"
    )
    warmup_done = state.step >= 1000
    train_node.add(f"{status_icon(warmup_done)} Warmup (1k steps)")
    cosine_active = 1000 <= state.step < state.total_steps
    train_node.add(f"{status_icon(state.eval_done, cosine_active)} Cosine Decay Phase")
    train_node.add(f"{status_icon(state.eval_done)} Final Evaluation")

    tree.add(f"{status_icon(state.gen_done)} Generation Pipeline")

    # ── Metrics Table ──
    metrics = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    metrics.add_column("Key",   style="dim cyan",  width=12)
    metrics.add_column("Value", style="bold white", width=16)

    loss_str  = f"{state.loss:.4f}" if math.isfinite(state.loss) else "---"
    ppl_str   = f"{state.ppl:.1f}"  if math.isfinite(state.ppl) and state.ppl < 1e6 else "---"
    bval_str  = f"{state.best_val_ppl:.1f}" if math.isfinite(state.best_val_ppl) and state.best_val_ppl < 1e6 else "---"

    gpu_pct   = state.gpu_gb / state.gpu_total
    gpu_bar_l = 12
    gpu_fill  = int(gpu_bar_l * gpu_pct)
    gpu_bar   = "#" * gpu_fill + "-" * (gpu_bar_l - gpu_fill)
    gpu_color = "red" if gpu_pct > 0.9 else ("yellow" if gpu_pct > 0.75 else "green")
    gpu_str   = f"[{gpu_color}]{gpu_bar}[/{gpu_color}] {state.gpu_gb:.1f}/{state.gpu_total:.0f}GB"

    metrics.add_row("Step",     f"{state.step:,} / {state.total_steps:,}")
    metrics.add_row("Loss",     loss_str)
    metrics.add_row("PPL",      ppl_str)
    metrics.add_row("Best Val", bval_str)
    metrics.add_row("LR",       f"{state.lr:.2e}")
    metrics.add_row("GPU Mem",  gpu_str)
    metrics.add_row("Speed",    f"{state.speed:.2f} step/s")
    metrics.add_row("ETA",      state.eta_str)
    metrics.add_row("Phase",    f"[cyan]{state.phase}[/cyan]")

    # ── Log Panel ──
    log_text = "\n".join(state.log_lines[-10:]) if state.log_lines else "[dim]No events yet[/dim]"

    # ── Val history spark ──
    if len(state.val_history) >= 2:
        hist_str = "  ".join([f"[{s}]={p:.1f}" for s, p in state.val_history[-5:]])
        val_hist_text = f"\n[dim]Val PPL history: {hist_str}[/dim]"
    else:
        val_hist_text = ""

    # Combine into one layout using a table
    layout = Table(box=box.ROUNDED, expand=True, show_header=False, padding=(0, 1))
    layout.add_column("Graph", ratio=3)
    layout.add_column("Metrics", ratio=2)
    layout.add_row(tree, metrics)

    outer = Table(box=box.ROUNDED, expand=True, show_header=False, padding=(0, 1))
    outer.add_column("Main", ratio=1)
    outer.add_row(layout)
    outer.add_row(
        Panel(log_text + val_hist_text, title="[bold]Event Log[/bold]", border_style="dim blue")
    )

    return Panel(outer, title="[bold magenta]Accessible RetNet — 124M | RTX 2070 | WikiText-103[/bold magenta]", border_style="magenta")


# ─────────────────────────────────────────────────────────────────────────────
# LR Schedule
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(progress, 1.0)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = max_lr * 0.1
    return min_lr + (max_lr - min_lr) * cosine_decay


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, max_batches: Optional[int] = None) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with autocast():
            logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            reduction="sum"
        )
        total_loss += loss.item()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20.0))
    model.train()
    return ppl


# ─────────────────────────────────────────────────────────────────────────────
# CSV Logger
# ─────────────────────────────────────────────────────────────────────────────

class CSVLogger:
    def __init__(self, path: Path):
        self.path = path
        self.file = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["step", "train_loss", "train_ppl", "val_ppl", "lr", "gpu_gb", "elapsed_s"])
        self.file.flush()

    def log(self, step, train_loss, train_ppl, val_ppl, lr, gpu_gb, elapsed_s):
        self.writer.writerow([step, f"{train_loss:.4f}", f"{train_ppl:.2f}",
                               f"{val_ppl:.2f}" if val_ppl else "", f"{lr:.6f}",
                               f"{gpu_gb:.2f}", f"{elapsed_s:.0f}"])
        self.file.flush()

    def close(self):
        self.file.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    out_dir = Path(__file__).parent / "checkpoints"
    out_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")
    if device.type == "cuda":
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")

    # ── Hyperparams ──
    TOTAL_STEPS     = 10 if args.smoke else 50_000
    WARMUP_STEPS    = 5  if args.smoke else 1_000
    PHYS_BATCH      = 2  if args.smoke else 8
    ACCUM_STEPS     = 2  if args.smoke else 4      # effective batch = 32
    MAX_LR          = 3e-4
    GRAD_CLIP       = 1.0
    EVAL_EVERY      = 5  if args.smoke else 2_000
    SAVE_EVERY      = 5  if args.smoke else 2_000
    LOG_EVERY       = 1  if args.smoke else 50

    state = TrainingState(TOTAL_STEPS, out_dir)
    csv_logger = CSVLogger(out_dir / "training_log.csv")

    # ── Dataset ──
    state.phase = "Loading Dataset"
    state.add_log("Loading WikiText-103 dataset...")
    print("[Train] Loading WikiText-103...")
    loaders = get_dataloaders(batch_size=PHYS_BATCH, num_workers=0, verbose=True)
    state.data_done = True
    state.add_log(f"Dataset ready: {loaders['stats']['train_chunks']:,} train chunks")

    # ── Model ──
    state.phase = "Building Model"
    state.add_log("Building RetNet 124M model...")
    config = RetNetConfig()
    model = RetNetLM(config).to(device)
    n_params = model.num_parameters()
    state.model_done = True
    state.add_log(f"Model ready: {n_params:,} parameters")
    print(f"[Train] Model parameters: {n_params:,}")

    # ── Optimizer ──
    # Separate weight decay: don't decay biases, LayerNorm params, embeddings
    decay_params    = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "bias" in name or "ln" in name or "norm" in name or "embedding" in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": 0.05},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=MAX_LR, betas=(0.9, 0.98), eps=1e-8
    )
    scaler = GradScaler()

    # ── Resume from checkpoint if exists ──
    start_step = 0
    ckpt_latest = out_dir / "latest.pt"
    if ckpt_latest.exists() and not args.smoke:
        print(f"[Train] Resuming from {ckpt_latest}")
        ckpt = torch.load(ckpt_latest, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"]
        state.best_val_ppl = ckpt.get("best_val_ppl", float("inf"))
        state.add_log(f"Resumed from step {start_step:,}")
        print(f"[Train] Resumed from step {start_step:,}")

    state.training_active = True

    # ── Live Dashboard ──
    def update_gpu_stats():
        if device.type == "cuda":
            used = torch.cuda.memory_allocated(0) / 1024**3
            state.gpu_gb = used

    # Training loop
    model.train()
    train_iter = iter(loaders["train"])
    optimizer.zero_grad()

    step_times = []
    running_loss = 0.0
    micro_count  = 0
    global_step  = start_step

    state.add_log(f"Training started | {TOTAL_STEPS:,} steps | LR={MAX_LR:.0e}")
    state.save_json()

    def do_dashboard_update():
        if RICH_AVAILABLE:
            panel = build_dashboard(state)
            return panel
        return None

    with (Live(do_dashboard_update(), refresh_per_second=1, console=console) if RICH_AVAILABLE else open(os.devnull, "w")) as live:

        t_step_start = time.time()

        while global_step < TOTAL_STEPS:
            # Fetch batch
            try:
                x_batch, y_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(loaders["train"])
                x_batch, y_batch = next(train_iter)

            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            # Forward + loss
            with autocast():
                logits = model(x_batch)  # (B, T, vocab)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y_batch.view(-1)
                )
                loss_scaled = loss / ACCUM_STEPS

            scaler.scale(loss_scaled).backward()
            running_loss += loss.item()
            micro_count  += 1

            # Gradient accumulation step
            if micro_count % ACCUM_STEPS == 0:
                # Clip gradients
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

                # Update LR
                lr = get_lr(global_step, WARMUP_STEPS, TOTAL_STEPS, MAX_LR)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                global_step += 1

                # Track timing
                t_now = time.time()
                step_time = t_now - t_step_start
                step_times.append(step_time)
                if len(step_times) > 100:
                    step_times.pop(0)
                t_step_start = t_now

                avg_step_time = sum(step_times) / len(step_times) if step_times else 1.0
                steps_remaining = TOTAL_STEPS - global_step
                eta_sec = avg_step_time * steps_remaining
                eta_td  = timedelta(seconds=int(eta_sec))
                eta_str = str(eta_td)

                # Update state
                avg_loss = running_loss / ACCUM_STEPS
                running_loss = 0.0
                ppl = math.exp(min(avg_loss, 20.0))

                state.step    = global_step
                state.loss    = avg_loss
                state.ppl     = ppl
                state.lr      = lr
                state.speed   = 1.0 / avg_step_time if avg_step_time > 0 else 0.0
                state.eta_str = eta_str
                state.phase   = "Warmup" if global_step < WARMUP_STEPS else "Cosine Decay"
                if global_step == WARMUP_STEPS:
                    state.warmup_done = True
                update_gpu_stats()

                # Log to console and CSV
                if global_step % LOG_EVERY == 0:
                    elapsed = time.time() - state.start_time
                    csv_logger.log(global_step, avg_loss, ppl, None, lr, state.gpu_gb, elapsed)
                    state.add_log(f"step={global_step:,} loss={avg_loss:.3f} ppl={ppl:.1f} lr={lr:.2e}")

                # Save JSON state
                if global_step % 100 == 0 or args.smoke:
                    state.save_json()

                # Update live dashboard
                if RICH_AVAILABLE:
                    live.update(do_dashboard_update())

                # Validation
                if global_step % EVAL_EVERY == 0 or global_step == TOTAL_STEPS:
                    state.phase = "Validation"
                    state.add_log(f"Running validation at step {global_step:,}...")
                    torch.cuda.empty_cache()   # free fragmented VRAM before eval
                    if RICH_AVAILABLE:
                        live.update(do_dashboard_update())

                    max_val_batches = 10 if args.smoke else None
                    val_ppl = evaluate(model, loaders["val"], device, max_batches=max_val_batches)

                    is_best = val_ppl < state.best_val_ppl
                    if is_best:
                        state.best_val_ppl = val_ppl

                    state.val_history.append((global_step, round(val_ppl, 2)))
                    elapsed = time.time() - state.start_time
                    csv_logger.log(global_step, avg_loss, ppl, val_ppl, lr, state.gpu_gb, elapsed)

                    log_msg = f"Val PPL={val_ppl:.2f} {'(NEW BEST)' if is_best else ''}"
                    state.add_log(log_msg)
                    state.save_json()

                    if RICH_AVAILABLE:
                        live.update(do_dashboard_update())

                    # Save best checkpoint
                    if is_best:
                        torch.save({
                            "step": global_step,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scaler": scaler.state_dict(),
                            "val_ppl": val_ppl,
                            "best_val_ppl": state.best_val_ppl,
                            "config": config.__dict__,
                        }, out_dir / "best_model.pt")
                        state.add_log(f"Saved best model (val_ppl={val_ppl:.2f})")

                    model.train()
                    state.phase = "Warmup" if global_step < WARMUP_STEPS else "Cosine Decay"

                # Save periodic checkpoint
                if global_step % SAVE_EVERY == 0:
                    torch.save({
                        "step": global_step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "val_ppl": state.best_val_ppl,
                        "best_val_ppl": state.best_val_ppl,
                        "config": config.__dict__,
                    }, out_dir / "latest.pt")
                    torch.save(model.state_dict(), out_dir / f"step_{global_step:06d}.pt")
                    state.add_log(f"Checkpoint saved at step {global_step:,}")

    # ── Training Complete ──
    state.phase = "Complete"
    state.eval_done = True
    state.training_active = False

    # Final evaluation
    state.add_log("Running final test set evaluation...")
    test_ppl = evaluate(model, loaders["test"], device)
    state.add_log(f"FINAL Test PPL = {test_ppl:.2f}")
    state.save_json()

    # Save final model
    torch.save({
        "step": global_step,
        "model": model.state_dict(),
        "config": config.__dict__,
        "val_ppl": state.best_val_ppl,
        "test_ppl": test_ppl,
    }, out_dir / "final_model.pt")

    csv_logger.close()

    print("\n" + "="*60)
    print(f"  Training Complete!")
    print(f"  Best Val PPL : {state.best_val_ppl:.2f}")
    print(f"  Test PPL     : {test_ppl:.2f}")
    print(f"  Checkpoints  : {out_dir}")
    print("="*60)


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Accessible RetNet")
    parser.add_argument("--smoke", action="store_true",
                        help="Run 10-step smoke test to verify everything works")
    args = parser.parse_args()

    if args.smoke:
        print("[Smoke Test] Running 10-step sanity check...")
    else:
        print("[Train] Starting full 50k-step training run...")
        print("[Train] Expected duration: ~8-12 hours on RTX 2070")
        print("[Train] Checkpoints saved to: ./checkpoints/")
        print("[Train] Live state: ./checkpoints/training_state.json")
        print("[Train] CSV log:    ./checkpoints/training_log.csv")
        print()

    train(args)
