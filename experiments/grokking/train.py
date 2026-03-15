"""
experiments/grokking/train.py — Grokking experiment training script.

Trains a small transformer on modular arithmetic and tracks grokking
dynamics: when memorization happens, when generalization happens, and
what the underlying progress measures look like throughout.
"""

import argparse
import json
import math
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import generate_dataset
from measures import compute_all_measures
from interventions import get_intervention


# ── Model ─────────────────────────────────────────────────────────────


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
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
        h = self.norm2(x)
        h = self.ff(h)
        x = x + h
        return x


class GrokTransformer(nn.Module):
    def __init__(
        self,
        prime: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 1,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.prime = prime
        self.d_model = d_model

        self.embed = nn.Embedding(prime, d_model)
        self.pos_embed = nn.Embedding(2, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, prime)

        # Omnigrok lever: scale all parameters
        if init_scale != 1.0:
            with torch.no_grad():
                for p in self.parameters():
                    p.mul_(init_scale)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, hidden_state_at_last_position)."""
        pos = torch.arange(x.size(1), device=x.device)
        h = self.embed(x) + self.pos_embed(pos)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        last_h = h[:, -1]  # [B, d_model]
        logits = self.head(last_h)  # [B, prime]
        return logits, last_h


# ── Training ──────────────────────────────────────────────────────────


def evaluate(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor,
             device: torch.device) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    """Evaluate on a full dataset. Returns (loss, accuracy, logits, hidden)."""
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
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}", file=sys.stderr)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    # ── Intervention ──
    intervention_config = {
        "weight_decay": args.weight_decay,
        "lr": args.lr,
        **{k: v for k, v in vars(args).items() if k.startswith(("wd_", "pulse_", "spike_", "noise_", "target_", "norm_", "spectral_"))},
    }
    intervention = get_intervention(args.intervention, intervention_config)

    # ── Tracking ──
    grok_threshold = args.grok_threshold
    memorize_threshold = 0.99
    grok_step = args.total_steps  # default: never grokked
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

        # NaN guard
        if torch.isnan(loss):
            print("NaN detected at step", step, file=sys.stderr)
            print(f"val_loss: nan", flush=True)
            print(f"val_accuracy: 0.0", flush=True)
            return

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # Batch accuracy for reactive interventions
        with torch.no_grad():
            batch_acc = (logits.detach().argmax(-1) == batch_y).float().mean().item()

        # Intervention on_step (may modify gradients/optimizer)
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

            # Check milestones
            if not memorized and train_acc >= memorize_threshold:
                memorize_step = step
                memorized = True
            if not grokked and val_acc >= grok_threshold:
                grok_step = step
                grokked = True

            # Compute progress measures
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
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
                f"w_norm={measures['weight_norm']:.2f}  "
                f"fourier={measures.get('fourier_top5_frac', 0):.3f}  "
                f"time={elapsed:.1f}s",
                file=sys.stderr,
            )

            # Early stop if grokked and stable
            if grokked and val_acc >= grok_threshold and step > grok_step + eval_interval * 3:
                print(f"Grokked at step {grok_step}, stable. Stopping early.", file=sys.stderr)
                break

    wall_time = time.time() - start_time

    # ── Final evaluation ──
    val_loss, val_acc, _, _ = evaluate(model, val_x, val_y, device)
    train_loss, train_acc, _, _ = evaluate(model, train_x, train_y, device)

    # ── Write measures log ──
    with open("measures.jsonl", "w") as f:
        for m in measures_log:
            f.write(json.dumps(m) + "\n")

    # ── Output metrics (stdout, for autoresearch extraction) ──
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
    print(f"intervention: {args.intervention}", flush=True)
    print(f"operation: {args.operation}", flush=True)
    print(f"prime: {args.prime}", flush=True)
    print(f"data_fraction: {args.data_fraction}", flush=True)
    print(f"init_scale: {args.init_scale}", flush=True)
    print(f"weight_decay: {args.weight_decay}", flush=True)
    print(f"lr: {args.lr}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Grokking experiment")

    # Task
    p.add_argument("--prime", type=int, default=113)
    p.add_argument("--operation", type=str, default="add", choices=["add", "sub", "mul"])
    p.add_argument("--data-fraction", type=float, default=0.5)
    p.add_argument("--data-seed", type=int, default=42)

    # Architecture
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--init-scale", type=float, default=1.0)

    # Training
    p.add_argument("--total-steps", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=6385)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.98)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Grokking detection
    p.add_argument("--grok-threshold", type=float, default=0.95)
    p.add_argument("--n-evals", type=int, default=200,
                   help="Number of evaluation points during training")

    # Intervention
    p.add_argument("--intervention", type=str, default="adaptive_wd",
                   choices=["none", "wd_ramp", "wd_pulse", "norm_target",
                            "lr_spike", "gradient_noise", "spectral_reg",
                            "adaptive_wd", "perp_grad", "perp_grad_adaptive"])

    # Intervention-specific params
    p.add_argument("--wd-ramp-frac", type=float, default=0.5)
    p.add_argument("--pulse-start-frac", type=float, default=0.25)
    p.add_argument("--pulse-duration", type=int, default=1000)
    p.add_argument("--pulse-wd", type=float, default=10.0)
    p.add_argument("--target-norm", type=float, default=5.0)
    p.add_argument("--norm-weight", type=float, default=0.01)
    p.add_argument("--spike-step-frac", type=float, default=0.33)
    p.add_argument("--spike-duration", type=int, default=200)
    p.add_argument("--spike-factor", type=float, default=10.0)
    p.add_argument("--noise-scale", type=float, default=0.01)
    p.add_argument("--spectral-weight", type=float, default=0.001)

    # Adaptive WD params
    p.add_argument("--adaptive-boost-wd", type=float, default=2.0)
    p.add_argument("--adaptive-trigger-acc", type=float, default=0.95)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
