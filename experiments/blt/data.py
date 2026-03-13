"""
Fixed data pipeline for BLT experiments.
Downloads a subset of FineWeb-Edu, converts to raw bytes, provides iterators.

DO NOT MODIFY — this file is immutable per experiment.yaml.
"""

import os
import struct
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_FILE = os.path.join(DATA_DIR, "train.bin")
VAL_FILE = os.path.join(DATA_DIR, "val.bin")

# Dataset config
NUM_TRAIN_BYTES = 100_000_000  # 100MB of training data
NUM_VAL_BYTES = 5_000_000      # 5MB of validation data
VOCAB_SIZE = 256               # raw bytes
SEED = 42

# HuggingFace dataset source
HF_DATASET = "HuggingFaceFW/fineweb-edu-score-2"
HF_SPLIT = "train"


def _download_and_prepare():
    """Download FineWeb-Edu subset and write raw bytes to binary files."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(TRAIN_FILE) and os.path.exists(VAL_FILE):
        # Verify sizes
        train_size = os.path.getsize(TRAIN_FILE)
        val_size = os.path.getsize(VAL_FILE)
        if train_size >= NUM_TRAIN_BYTES and val_size >= NUM_VAL_BYTES:
            return

    print("Downloading FineWeb-Edu subset...")
    from datasets import load_dataset

    ds = load_dataset(
        HF_DATASET,
        split=f"{HF_SPLIT}",
        streaming=True,
    )

    total_needed = NUM_TRAIN_BYTES + NUM_VAL_BYTES
    raw_bytes = bytearray()

    for example in ds:
        text = example["text"]
        raw_bytes.extend(text.encode("utf-8"))
        if len(raw_bytes) >= total_needed + 10000:  # small buffer
            break

    if len(raw_bytes) < total_needed:
        raise RuntimeError(
            f"Only got {len(raw_bytes)} bytes, needed {total_needed}. "
            "Dataset may be too small or network issue."
        )

    # Deterministic split
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(len(raw_bytes))

    all_bytes = np.frombuffer(bytes(raw_bytes), dtype=np.uint8)
    shuffled = all_bytes[indices]

    train_bytes = shuffled[:NUM_TRAIN_BYTES]
    val_bytes = shuffled[NUM_TRAIN_BYTES:NUM_TRAIN_BYTES + NUM_VAL_BYTES]

    with open(TRAIN_FILE, "wb") as f:
        f.write(train_bytes.tobytes())
    with open(VAL_FILE, "wb") as f:
        f.write(val_bytes.tobytes())

    print(f"Wrote {NUM_TRAIN_BYTES:,} train bytes and {NUM_VAL_BYTES:,} val bytes.")


class ByteDataset(Dataset):
    """Memory-mapped byte-level dataset."""

    def __init__(self, filepath: str, seq_len: int):
        self.seq_len = seq_len
        self.data = np.memmap(filepath, dtype=np.uint8, mode="r")
        self.num_sequences = (len(self.data) - 1) // seq_len  # -1 for target shift

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 for target
        chunk = self.data[start:end].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def get_dataloaders(
    seq_len: int = 4096,
    batch_size: int = 8,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for byte-level data.

    Args:
        seq_len: Number of bytes per sequence.
        batch_size: Batch size.
        num_workers: DataLoader workers.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    _download_and_prepare()

    train_ds = ByteDataset(TRAIN_FILE, seq_len)
    val_ds = ByteDataset(VAL_FILE, seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, val_loader


def get_val_bytes(n: int = 10000) -> bytes:
    """Return first n bytes from validation set for generation quality checks."""
    _download_and_prepare()
    with open(VAL_FILE, "rb") as f:
        return f.read(n)


if __name__ == "__main__":
    _download_and_prepare()
    print(f"Train: {os.path.getsize(TRAIN_FILE):,} bytes")
    print(f"Val: {os.path.getsize(VAL_FILE):,} bytes")

    # Quick sanity check
    train_loader, val_loader = get_dataloaders(seq_len=512, batch_size=4)
    x, y = next(iter(train_loader))
    print(f"Batch shape: x={x.shape}, y={y.shape}")
    print(f"Value range: [{x.min()}, {x.max()}]")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
