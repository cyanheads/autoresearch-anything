"""
experiments/grokking/measures.py — Progress measures for grokking.

Tracks the underlying dynamics of circuit formation, not just accuracy.
Immutable across experiments for consistent measurement.

Key measures:
- Fourier analysis of embeddings (Nanda et al. 2023)
- Weight norm trajectory (Omnigrok / LU mechanism)
- Logit statistics (Softmax Collapse detection)
- Spectral entropy of representations
"""

import torch
import torch.nn as nn
import numpy as np


def compute_fourier_magnitudes(embed_weights: torch.Tensor, prime: int) -> dict:
    """Fourier analysis of the embedding matrix.

    For modular arithmetic mod p, the generalizing circuit encodes numbers
    as points on circles at specific Fourier frequencies. This tracks how
    much embedding energy is concentrated in the dominant frequencies vs
    spread uniformly (memorization).
    """
    W = embed_weights.detach().cpu().float().numpy()  # [prime, d_model]
    p = prime

    # DFT: F[k] = sum_n W[n] * exp(-2πi k n / p)
    freqs = np.arange(p)
    dft_matrix = np.exp(-2j * np.pi * np.outer(freqs, freqs) / p) / np.sqrt(p)
    F = dft_matrix @ W  # [p, d_model]

    # Power spectrum: |F[k]|^2 summed over embedding dimensions
    power = np.sum(np.abs(F) ** 2, axis=1)  # [p]
    total_power = np.sum(power)

    if total_power < 1e-10:
        return {"fourier_top5_frac": 0.0, "fourier_gini": 0.0}

    # Top-5 non-DC frequency fraction (how concentrated is the spectrum)
    non_dc = np.sort(power[1:])[::-1]
    top5_frac = float(np.sum(non_dc[:5]) / total_power)

    # Gini coefficient of power spectrum (0 = uniform, 1 = single frequency)
    sorted_power = np.sort(power[1:])  # exclude DC
    n = len(sorted_power)
    cumsum = np.cumsum(sorted_power)
    gini = float((2 * np.sum((np.arange(1, n + 1) * sorted_power)) / (n * np.sum(sorted_power))) - (n + 1) / n)

    return {
        "fourier_top5_frac": top5_frac,
        "fourier_gini": max(0.0, gini),
    }


def compute_weight_norm(model: nn.Module) -> float:
    """Total L2 norm of all parameters."""
    total = sum(p.data.norm().item() ** 2 for p in model.parameters())
    return float(total ** 0.5)


def compute_embed_norm(model: nn.Module, embed_name: str = "embed") -> float:
    """L2 norm of just the embedding matrix."""
    for name, p in model.named_parameters():
        if embed_name in name and "pos" not in name:
            return float(p.data.norm().item())
    return 0.0


def compute_logit_stats(logits: torch.Tensor) -> dict:
    """Statistics of logit magnitudes.

    Tracks Softmax Collapse (Lyu et al. 2025): if logits grow unboundedly
    after memorization, floating-point absorption errors cause gradient
    vanishing, permanently trapping the network.
    """
    with torch.no_grad():
        return {
            "logit_mean_abs": float(logits.abs().mean().item()),
            "logit_max_abs": float(logits.abs().max().item()),
            "logit_std": float(logits.std().item()),
        }


def compute_spectral_entropy(hidden: torch.Tensor) -> float:
    """Spectral entropy of hidden representations.

    High entropy = distributed/random representations (memorization).
    Low entropy = structured/low-rank representations (generalization).

    Computed from singular values of the representation matrix.
    """
    with torch.no_grad():
        h = hidden.detach().float()
        h = h - h.mean(dim=0)

        try:
            s = torch.linalg.svdvals(h)
        except Exception:
            return 0.0

        # Normalize to probability distribution
        s = s / (s.sum() + 1e-10)
        s = s[s > 1e-10]

        entropy = -(s * s.log()).sum()
        return float(entropy.item())


def compute_all_measures(
    model: nn.Module,
    prime: int,
    val_logits: torch.Tensor | None = None,
    val_hidden: torch.Tensor | None = None,
) -> dict:
    """Compute all progress measures in one call."""
    measures = {}

    # Fourier analysis of token embeddings
    embed_w = None
    for name, p in model.named_parameters():
        if "embed" in name and "pos" not in name:
            embed_w = p.data
            break
    if embed_w is not None:
        measures.update(compute_fourier_magnitudes(embed_w, prime))

    # Weight norms
    measures["weight_norm"] = compute_weight_norm(model)
    measures["embed_norm"] = compute_embed_norm(model)

    # Logit stats (if provided)
    if val_logits is not None:
        measures.update(compute_logit_stats(val_logits))

    # Spectral entropy (if provided)
    if val_hidden is not None:
        measures["spectral_entropy"] = compute_spectral_entropy(val_hidden)

    return measures
