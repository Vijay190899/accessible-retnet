"""
WikiText-103 Dataset Pipeline for Accessible RetNet.
Downloads, tokenizes, and chunks the dataset into 512-token sequences.
Uses GPT-2 tokenizer (vocab size 50,257).
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from pathlib import Path


CACHE_DIR = Path(__file__).parent / "data_cache"
CHUNK_SIZE = 512   # max_seq_len from config


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer Wrapper
# ─────────────────────────────────────────────────────────────────────────────

def get_tokenizer():
    """Load GPT-2 tokenizer. Falls back to tiktoken if transformers unavailable."""
    try:
        from transformers import GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        return tok
    except Exception as e:
        raise RuntimeError(
            f"Could not load GPT-2 tokenizer: {e}\n"
            "Run: pip install transformers>=4.38.0"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Chunked Token Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ChunkedTokenDataset(Dataset):
    """
    Stores a flat list of token IDs and yields (input, target) pairs
    of length CHUNK_SIZE via sliding window with stride = CHUNK_SIZE (no overlap).
    input[i]  = tokens[i : i + CHUNK_SIZE]
    target[i] = tokens[i+1 : i + CHUNK_SIZE + 1]  (next-token prediction)
    """

    def __init__(self, token_ids: torch.Tensor, chunk_size: int = CHUNK_SIZE):
        self.tokens = token_ids
        self.chunk_size = chunk_size
        # Number of complete chunks
        self.n_chunks = (len(token_ids) - 1) // chunk_size

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx: int):
        start = idx * self.chunk_size
        end = start + self.chunk_size
        x = self.tokens[start:end]
        y = self.tokens[start + 1:end + 1]
        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# Build or Load Cache
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize_split(texts, tokenizer, split_name: str) -> torch.Tensor:
    """Concatenate all texts and tokenize into a flat token tensor."""
    print(f"  Tokenizing {split_name} split ({len(texts)} articles)...", flush=True)
    all_ids = []
    eos_id = tokenizer.eos_token_id

    for i, text in enumerate(texts):
        text = text.strip()
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
        all_ids.append(eos_id)  # document separator

        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{len(texts)} articles tokenized...", flush=True)

    flat = torch.tensor(all_ids, dtype=torch.long)
    print(f"  {split_name}: {len(flat):,} tokens ({len(flat)/1e6:.1f}M)", flush=True)
    return flat


def build_datasets(verbose: bool = True) -> dict:
    """
    Download WikiText-103, tokenize, chunk, and return DataLoaders.
    Caches tokenized tensors to disk so subsequent runs are instant.

    Returns dict with keys: 'train', 'val', 'test' — each a ChunkedTokenDataset.
    Also returns 'stats' dict with token counts.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / "wikitext103_gpt2.pt"

    if cache_file.exists():
        if verbose:
            print(f"[Dataset] Loading cached tokens from {cache_file}")
        data = torch.load(cache_file)
        if verbose:
            for split in ["train", "val", "test"]:
                n = len(data[split])
                print(f"  {split:6s}: {n:,} tokens ({n/1e6:.1f}M)")
    else:
        if verbose:
            print("[Dataset] Downloading WikiText-103 (first time, ~100MB)...")
        from datasets import load_dataset
        raw = load_dataset("wikitext", "wikitext-103-raw-v1")

        tokenizer = get_tokenizer()

        if verbose:
            print("[Dataset] Tokenizing with GPT-2 tokenizer...")
        data = {}
        for split_name, hf_split in [("train", "train"), ("val", "validation"), ("test", "test")]:
            texts = raw[hf_split]["text"]
            data[split_name] = _tokenize_split(texts, tokenizer, split_name)

        torch.save(data, cache_file)
        if verbose:
            print(f"[Dataset] Saved cache to {cache_file}")

    # Build datasets
    train_ds = ChunkedTokenDataset(data["train"], CHUNK_SIZE)
    val_ds   = ChunkedTokenDataset(data["val"],   CHUNK_SIZE)
    test_ds  = ChunkedTokenDataset(data["test"],  CHUNK_SIZE)

    stats = {
        "train_tokens": len(data["train"]),
        "val_tokens":   len(data["val"]),
        "test_tokens":  len(data["test"]),
        "train_chunks": len(train_ds),
        "val_chunks":   len(val_ds),
        "test_chunks":  len(test_ds),
    }

    if verbose:
        print(f"[Dataset] Train chunks: {len(train_ds):,} | Val chunks: {len(val_ds):,} | Test chunks: {len(test_ds):,}")
        print(f"[Dataset] Chunk size: {CHUNK_SIZE} tokens | Vocab: 50,257 (GPT-2)")

    return {"train": train_ds, "val": val_ds, "test": test_ds, "stats": stats}


def get_dataloaders(
    batch_size: int = 8,
    num_workers: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Returns train/val/test DataLoaders.
    num_workers=0 is safest on Windows (avoids multiprocessing issues).
    """
    datasets = build_datasets(verbose=verbose)

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False,
    )

    return {
        "train": train_loader,
        "val":   val_loader,
        "test":  test_loader,
        "stats": datasets["stats"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*55)
    print("  WikiText-103 Dataset Pipeline Test")
    print("="*55)

    loaders = get_dataloaders(batch_size=4, verbose=True)

    # Peek at a batch
    train_iter = iter(loaders["train"])
    x_batch, y_batch = next(train_iter)
    print(f"\n[Batch check]")
    print(f"  x shape: {tuple(x_batch.shape)}  (batch, seq_len)")
    print(f"  y shape: {tuple(y_batch.shape)}")
    print(f"  x[0, :8] tokens: {x_batch[0, :8].tolist()}")
    print(f"  y[0, :8] tokens: {y_batch[0, :8].tolist()} (shifted by 1)")

    stats = loaders["stats"]
    print(f"\n[Stats]")
    print(f"  Train: {stats['train_tokens']:>12,} tokens | {stats['train_chunks']:>8,} chunks")
    print(f"  Val:   {stats['val_tokens']:>12,} tokens | {stats['val_chunks']:>8,} chunks")
    print(f"  Test:  {stats['test_tokens']:>12,} tokens | {stats['test_chunks']:>8,} chunks")
    print("\n[OK] Dataset pipeline verified.")
