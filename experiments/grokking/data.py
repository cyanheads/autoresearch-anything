"""
experiments/grokking/data.py — Modular arithmetic dataset generation.

Generates all (a, b) pairs for a binary operation mod prime,
splits into train/val sets. Immutable across experiments to ensure
consistent evaluation.
"""

import torch
from typing import Literal

Operation = Literal["add", "sub", "mul"]


def generate_dataset(
    prime: int = 113,
    operation: Operation = "add",
    data_fraction: float = 0.5,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate all (a, b) pairs for a op b mod prime, split into train/val.

    Returns (train_inputs, train_targets, val_inputs, val_targets).
    Inputs are [N, 2] tensors of (a, b) pairs.
    Targets are [N] tensors of (a op b) mod prime.
    """
    rng = torch.Generator().manual_seed(seed)

    a = torch.arange(prime).repeat_interleave(prime)
    b = torch.arange(prime).repeat(prime)

    if operation == "add":
        c = (a + b) % prime
    elif operation == "sub":
        c = (a - b) % prime
    elif operation == "mul":
        c = (a * b) % prime
    else:
        raise ValueError(f"Unknown operation: {operation}")

    inputs = torch.stack([a, b], dim=1)  # [p^2, 2]

    n = len(inputs)
    n_train = int(n * data_fraction)
    perm = torch.randperm(n, generator=rng)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    return inputs[train_idx], c[train_idx], inputs[val_idx], c[val_idx]
