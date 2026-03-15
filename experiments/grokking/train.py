"""
experiments/grokking/train.py — Grokking experiment training script.

Supports multiple model architectures (standard, spherical, svd-param)
and gradient interventions (grokfast, perp_grad, etc).
"""

import argparse
import json
import math
import sys
import time
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import generate_dataset
from measures import compute_all_measures
from interventions import get_intervention


# ── Model architectures ──────────────────────────────────────────────


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, spherical: bool = False):
        super().__init__()
        self.spherical = spherical
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        if self.spherical:
            x = F.normalize(x, dim=-1) * math.sqrt(x.size(-1))
        h = self.norm2(x)
        h = self.ff(h)
        x = x + h
        if self.spherical:
            x = F.normalize(x, dim=-1) * math.sqrt(x.size(-1))
        return x


class SVDLinear(nn.Module):
    """Linear layer parameterized as U @ diag(sigma) @ V^T.

    From "Decomposed Learning" — reparameterizing weights in SVD form
    constrains the optimization geometry and prevents deep memorization
    basins. The optimizer directly updates U, sigma, V.
    """
    def __init__(self, in_features: int, out_features: int, rank: int | None = None):
        super().__init__()
        rank = rank or min(in_features, out_features)
        self.U = nn.Parameter(torch.randn(out_features, rank) * 0.02)
        self.sigma = nn.Parameter(torch.ones(rank))
        self.V = nn.Parameter(torch.randn(in_features, rank) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x @ V @ diag(sigma) @ U^T + bias
        h = x @ self.V          # [..., rank]
        h = h * self.sigma      # [..., rank]
        h = h @ self.U.T        # [..., out]
        return h + self.bias


class GrokTransformer(nn.Module):
    def __init__(
        self,
        prime: int,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 1024,
        n_layers: int = 3,
        init_scale: float = 1.0,
        arch: str = "standard",  # "standard", "spherical", "svd"
        svd_rank: int | None = None,
    ):
        super().__init__()
        self.prime = prime
        self.d_model = d_model
        self.arch = arch

        self.embed = nn.Embedding(prime, d_model)
        self.pos_embed = nn.Embedding(2, d_model)

        spherical = (arch == "spherical")
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, spherical=spherical)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        if arch == "svd":
            self.head = SVDLinear(d_model, prime, rank=svd_rank)
        else:
            self.head = nn.Linear(d_model, prime)

        if init_scale != 1.0:
            with torch.no_grad():
                for p in self.parameters():
                    p.mul_(init_scale)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.arange(x.size(1), device=x.device)
        h = self.embed(x) + self.pos_embed(pos)
        if self.arch == "spherical":
            h = F.normalize(h, dim=-1) * math.sqrt(h.size(-1))
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        last_h = h[:, -1]
        logits = self.head(last_h)
        return logits, last_h


# ── Grokfast: slow gradient amplification ────────────────────────────


class GrokfastEMA:
    """Grokfast (Lim et al. 2024): amplify slow gradient components.

    Maintains an EMA of gradients. The slow (low-frequency) component
    is the EMA itself; the fast component is grad - EMA. Amplify the
    slow component by factor alpha to accelerate circuit formation.

    This is the key insight: the generalizing circuit produces slow,
    consistent gradient signal that gets drowned out by fast memorization
    noise. Amplifying the slow signal accelerates grokking by 50-100x
    in the original paper.
    """
    def __init__(self, alpha: float = 2.0, lamb: float = 0.98):
        self.alpha = alpha
        self.lamb = lamb
        self.ema: dict[str, torch.Tensor] = {}

    def step(self, model: nn.Module):
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                if name not in self.ema:
                    self.ema[name] = torch.zeros_like(p.grad)
                self.ema[name].mul_(self.lamb).add_(p.grad, alpha=1 - self.lamb)
                # Amplify slow component: grad = grad + alpha * ema
                p.grad.add_(self.ema[name], alpha=self.alpha)


# ── Training ──────────────────────────────────────────────────────────


def evaluate(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor,
             device: torch.device) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        x = inputs.to(device)
        t = targets.to(device)
        logits, hidden = model(x)
        loss = F.cross_entropy(logits, t)
        acc = (logits.argmax(-1) == t).float().mean()
    model.train()
    return float(loss.item()), float(acc.item()), logits, hidden


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", file=sys.stderr)

    # ── Data ──
    train_x, train_y, val_x, val_y = generate_dataset(
        prime=args.prime,
        operation=args.operation,
        data_fraction=args.data_fraction,
        seed=args.data_seed,
    )
    print(f"train: {len(train_x)}, val: {len(val_x)}", file=sys.stderr)
    train_x, train_y = train_x.to(device), train_y.to(device)
    val_x, val_y = val_x.to(device), val_y.to(device)

    # ── Model ──
    model = GrokTransformer(
        prime=args.prime,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        init_scale=args.init_scale,
        arch=args.arch,
        svd_rank=args.svd_rank,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}  arch: {args.arch}", file=sys.stderr)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    # ── Grokfast ──
    grokfast = None
    if args.grokfast:
        grokfast = GrokfastEMA(alpha=args.grokfast_alpha, lamb=args.grokfast_lamb)

    # ── Intervention ──
    intervention_config = {
        "weight_decay": args.weight_decay,
        "lr": args.lr,
        **{k: v for k, v in vars(args).items()
           if k.startswith(("wd_", "pulse_", "spike_", "noise_", "target_",
                            "norm_", "spectral_", "adaptive_"))},
    }
    intervention = get_intervention(args.intervention, intervention_config)

    # ── Tracking ──
    grok_step = args.total_steps
    memorize_step = args.total_steps
    grokked = False
    memorized = False
    measures_log = []
    eval_interval = max(args.total_steps // args.n_evals, 1)
    start_time = time.time()
    peak_vram = 0

    # ── Training loop ──
    model.train()
    for step in range(1, args.total_steps + 1):
        # LR warmup
        if args.warmup_steps > 0 and step <= args.warmup_steps:
            lr = args.lr * step / args.warmup_steps
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        # Sample batch
        idx = torch.randint(0, len(train_x), (args.batch_size,), device=device)
        batch_x, batch_y = train_x[idx], train_y[idx]

        # Forward
        logits, _ = model(batch_x)
        loss = F.cross_entropy(logits, batch_y)

        # Intervention extra loss
        extra = intervention.extra_loss(model, logits, batch_y, step, args.total_steps)
        if extra != 0.0:
            loss = loss + extra

        if torch.isnan(loss):
            print("NaN detected at step", step, file=sys.stderr)
            print(f"grok_step: {args.total_steps}", flush=True)
            print(f"final_val_acc: 0.0", flush=True)
            return

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # Grokfast: amplify slow gradient components
        if grokfast is not None:
            grokfast.step(model)

        # Batch accuracy for reactive interventions
        with torch.no_grad():
            batch_acc = (logits.detach().argmax(-1) == batch_y).float().mean().item()

        # Intervention on_step
        intervention.on_step(model, optimizer, step, args.total_steps,
                             {"batch_acc": batch_acc})

        optimizer.step()

        # Track VRAM
        if device.type == "cuda":
            vram = torch.cuda.max_memory_allocated(device) / 1024 / 1024
            peak_vram = max(peak_vram, vram)

        # ── Periodic evaluation ──
        if step % eval_interval == 0 or step == 1:
            val_loss, val_acc, val_logits, val_hidden = evaluate(model, val_x, val_y, device)
            train_loss, train_acc, _, _ = evaluate(model, train_x, train_y, device)

            if not memorized and train_acc >= 0.99:
                memorize_step = step
                memorized = True
            if not grokked and val_acc >= args.grok_threshold:
                grok_step = step
                grokked = True

            measures = compute_all_measures(model, args.prime, val_logits, val_hidden)
            measures["step"] = step
            measures["train_loss"] = train_loss
            measures["train_acc"] = train_acc
            measures["val_loss"] = val_loss
            measures["val_acc"] = val_acc
            measures_log.append(measures)

            elapsed = time.time() - start_time
            print(
                f"step {step}/{args.total_steps}  "
                f"t_loss={train_loss:.4f}  t_acc={train_acc:.4f}  "
                f"v_loss={val_loss:.4f}  v_acc={val_acc:.4f}  "
                f"w={measures['weight_norm']:.1f}  "
                f"f5={measures.get('fourier_top5_frac', 0):.3f}  "
                f"{elapsed:.1f}s",
                file=sys.stderr,
            )

            if grokked and val_acc >= args.grok_threshold and step > grok_step + eval_interval * 3:
                print(f"Grokked at step {grok_step}, stable. Stopping.", file=sys.stderr)
                break

    wall_time = time.time() - start_time
    val_loss, val_acc, _, _ = evaluate(model, val_x, val_y, device)
    train_loss, train_acc, _, _ = evaluate(model, train_x, train_y, device)

    with open("measures.jsonl", "w") as f:
        for m in measures_log:
            f.write(json.dumps(m) + "\n")

    grok_delay = grok_step - memorize_step if grokked and memorized else args.total_steps
    print(f"grok_step: {grok_step}", flush=True)
    print(f"memorize_step: {memorize_step}", flush=True)
    print(f"grok_delay: {grok_delay}", flush=True)
    print(f"final_val_acc: {val_acc:.6f}", flush=True)
    print(f"final_train_acc: {train_acc:.6f}", flush=True)
    print(f"val_loss: {val_loss:.6f}", flush=True)
    print(f"train_loss: {train_loss:.6f}", flush=True)
    print(f"wall_time_s: {wall_time:.1f}", flush=True)
    print(f"peak_vram_mb: {peak_vram:.0f}", flush=True)
    print(f"grokked: {grokked}", flush=True)
    print(f"arch: {args.arch}", flush=True)
    print(f"grokfast: {args.grokfast}", flush=True)
    print(f"intervention: {args.intervention}", flush=True)
    print(f"operation: {args.operation}", flush=True)
    print(f"prime: {args.prime}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Grokking experiment")

    # Task
    p.add_argument("--prime", type=int, default=113)
    p.add_argument("--operation", type=str, default="add", choices=["add", "sub", "mul"])
    p.add_argument("--data-fraction", type=float, default=0.5)
    p.add_argument("--data-seed", type=int, default=42)

    # Architecture
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--arch", type=str, default="standard",
                   choices=["standard", "spherical", "svd"])
    p.add_argument("--svd-rank", type=int, default=None)

    # Training
    p.add_argument("--total-steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.98)
    p.add_argument("--weight-decay", type=float, default=2.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=50)

    # Grokfast
    p.add_argument("--grokfast", action="store_true", default=False)
    p.add_argument("--grokfast-alpha", type=float, default=2.0,
                   help="Slow gradient amplification factor")
    p.add_argument("--grokfast-lamb", type=float, default=0.98,
                   help="EMA decay for slow gradient estimation")

    # Grokking detection
    p.add_argument("--grok-threshold", type=float, default=0.95)
    p.add_argument("--n-evals", type=int, default=1000)

    # Intervention
    p.add_argument("--intervention", type=str, default="none")

    # Intervention-specific params
    p.add_argument("--adaptive-boost-wd", type=float, default=2.0)
    p.add_argument("--adaptive-trigger-acc", type=float, default=0.95)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
