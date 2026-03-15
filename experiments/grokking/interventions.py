"""
experiments/grokking/interventions.py — Dynamic training interventions.

Each intervention can modify training dynamics during the run:
- Adjust optimizer hyperparameters (LR, weight decay)
- Add auxiliary loss terms
- Modify gradients
- Apply explicit regularization

This is the primary experimental surface. Add new interventions here.
"""

import math
import torch
import torch.nn as nn


class Intervention:
    """Base class for training interventions."""

    def __init__(self, config: dict):
        self.config = config

    def on_step(self, model: nn.Module, optimizer, step: int, total_steps: int, metrics: dict):
        """Called after each optimizer step. Can modify model/optimizer state."""
        pass

    def extra_loss(self, model: nn.Module, logits: torch.Tensor, targets: torch.Tensor,
                   step: int, total_steps: int) -> torch.Tensor | float:
        """Return additional loss term to add to CE loss, or 0."""
        return 0.0


class NoIntervention(Intervention):
    """Baseline: standard training with no modifications."""
    pass


class WeightDecayRamp(Intervention):
    """Linearly ramp weight decay from 0 to target value.

    Hypothesis: letting the model memorize freely first, then gradually
    increasing weight decay pressure, may accelerate the transition from
    memorization to generalization.
    """
    def on_step(self, model, optimizer, step, total_steps, metrics):
        ramp_frac = self.config.get("wd_ramp_frac", 0.5)
        ramp_steps = int(total_steps * ramp_frac)
        target_wd = self.config.get("weight_decay", 1.0)
        frac = min(step / max(ramp_steps, 1), 1.0)
        for pg in optimizer.param_groups:
            pg["weight_decay"] = target_wd * frac


class WeightDecayPulse(Intervention):
    """Pulse of high weight decay at a specific training phase.

    Hypothesis: a sharp burst of weight decay after memorization can
    rapidly prune the memorization circuit, accelerating grokking.
    """
    def on_step(self, model, optimizer, step, total_steps, metrics):
        pulse_start_frac = self.config.get("pulse_start_frac", 0.25)
        pulse_duration = self.config.get("pulse_duration", 1000)
        pulse_wd = self.config.get("pulse_wd", 10.0)
        base_wd = self.config.get("weight_decay", 1.0)

        pulse_start = int(total_steps * pulse_start_frac)
        if pulse_start <= step < pulse_start + pulse_duration:
            wd = pulse_wd
        else:
            wd = base_wd
        for pg in optimizer.param_groups:
            pg["weight_decay"] = wd


class NormTarget(Intervention):
    """Regularize toward a target weight norm (Omnigrok Goldilocks zone).

    The LU mechanism shows grokking requires traversing from high to
    optimal weight norm. This directly targets the optimal norm.
    """
    def extra_loss(self, model, logits, targets, step, total_steps):
        target_norm = self.config.get("target_norm", 5.0)
        norm_weight = self.config.get("norm_weight", 0.01)
        current_norm_sq = sum(p.norm() ** 2 for p in model.parameters())
        target_sq = target_norm ** 2
        return norm_weight * (current_norm_sq - target_sq) ** 2


class LRSpike(Intervention):
    """Spike learning rate to perturb out of memorization basin.

    Hypothesis: a brief LR spike after memorization can destabilize
    the memorizing solution, allowing the optimizer to find the
    generalizing circuit faster.
    """
    def on_step(self, model, optimizer, step, total_steps, metrics):
        spike_step_frac = self.config.get("spike_step_frac", 0.33)
        spike_duration = self.config.get("spike_duration", 200)
        spike_factor = self.config.get("spike_factor", 10.0)
        base_lr = self.config.get("lr", 1e-3)

        spike_step = int(total_steps * spike_step_frac)
        if spike_step <= step < spike_step + spike_duration:
            lr = base_lr * spike_factor
        else:
            lr = base_lr
        for pg in optimizer.param_groups:
            pg["lr"] = lr


class GradientNoise(Intervention):
    """Add calibrated noise to gradients.

    Hypothesis: noise helps escape the memorization basin (sharp minimum)
    toward the generalization basin (flatter minimum).
    """
    def on_step(self, model, optimizer, step, total_steps, metrics):
        noise_scale = self.config.get("noise_scale", 0.01)
        # Anneal noise: high early, low late
        scale = noise_scale / (1 + step * 0.001)
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.add_(torch.randn_like(p.grad) * scale)


class SpectralRegularization(Intervention):
    """Penalize high spectral entropy of representations.

    Hypothesis: forcing structured/low-rank representations may
    accelerate circuit formation. The generalizing circuit produces
    lower-entropy representations than memorization.
    """
    def extra_loss(self, model, logits, targets, step, total_steps):
        reg_weight = self.config.get("spectral_weight", 0.001)
        # Compute spectral entropy of logits as a proxy
        # (cheaper than SVD of hidden states every step)
        s = torch.linalg.svdvals(logits.float())
        s = s / (s.sum() + 1e-10)
        s = s.clamp(min=1e-10)
        entropy = -(s * s.log()).sum()
        return reg_weight * entropy


# ── Registry ──────────────────────────────────────────────────────────

INTERVENTIONS = {
    "none": NoIntervention,
    "wd_ramp": WeightDecayRamp,
    "wd_pulse": WeightDecayPulse,
    "norm_target": NormTarget,
    "lr_spike": LRSpike,
    "gradient_noise": GradientNoise,
    "spectral_reg": SpectralRegularization,
}


def get_intervention(name: str, config: dict) -> Intervention:
    """Get an intervention by name. Falls back to NoIntervention."""
    cls = INTERVENTIONS.get(name)
    if cls is None:
        raise ValueError(f"Unknown intervention: {name}. Available: {list(INTERVENTIONS.keys())}")
    return cls(config)
