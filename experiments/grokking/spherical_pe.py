"""
experiments/grokking/spherical_pe.py — Spherical Positional Encoding experiments.

Tests SO(d) rotational PE on a spherical residual stream.

Three PE variants:
1. Standard additive PE (baseline, breaks spherical geometry)
2. RoPE on q,k (standard, block-diagonal SO(2)^{d/2})
3. Learned-plane RoPE: learn which planes to rotate in (orthogonal Q basis)
4. Spherical PE: SO(d) rotation applied to hidden states directly
   (geometrically native — SO(d) is the isometry group of S^{d-1})

Tested on modular arithmetic (fast) and FineWeb (real language).
"""

import argparse
import math
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import generate_dataset
from measures import compute_all_measures


# ── Positional Encoding implementations ──────────────────────────────


class AdditivePositionalEncoding(nn.Module):
    """Standard learned additive PE. Breaks spherical geometry."""
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add PE to hidden states."""
        pos = torch.arange(x.size(1), device=x.device)
        return x + self.pe(pos)

    def apply_to_qk(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0):
        """No-op for additive PE (already applied to hidden states)."""
        return q, k


class RoPE(nn.Module):
    """Standard RoPE: block-diagonal SO(2)^{d/2} rotations on q,k.

    Each pair of dimensions (2i, 2i+1) rotates by position * theta_i.
    No cross-dimensional interactions. This is the baseline rotary PE.
    """
    def __init__(self, d_model: int, max_len: int = 4096, base: float = 10000.0):
        super().__init__()
        # Frequencies: theta_i = base^(-2i/d)
        freqs = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("freqs", freqs)
        # Precompute sin/cos for max_len positions
        positions = torch.arange(max_len).float()
        angles = torch.outer(positions, freqs)  # [max_len, d/2]
        self.register_buffer("cos_cache", angles.cos())
        self.register_buffer("sin_cache", angles.sin())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """No-op — RoPE doesn't modify hidden states."""
        return x

    def apply_to_qk(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0):
        """Apply rotary embeddings to queries and keys."""
        T = q.size(-2)
        d_head = q.size(-1)
        cos = self.cos_cache[offset:offset + T, :d_head // 2].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[offset:offset + T, :d_head // 2].unsqueeze(0).unsqueeze(0)

        def rotate(x):
            x1, x2 = x[..., ::2], x[..., 1::2]
            return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

        return rotate(q), rotate(k)


class LearnedPlaneRoPE(nn.Module):
    """Learned-basis RoPE: rotate in learned planes, not consecutive pairs.

    Learn an orthogonal matrix Q that defines the rotation basis.
    RoPE is then applied in Q-rotated coordinates:
      R(pos) = Q^T @ diag_rotations(pos) @ Q

    This gives cross-dimensional positional interactions through the
    basis change, while keeping O(d) rotation cost.

    From "Rethinking RoPE" (2025): the missing ingredient in RoPE is
    the choice of basis, not the rotation structure itself.
    """
    def __init__(self, d_model: int, max_len: int = 4096, base: float = 10000.0):
        super().__init__()
        # Standard RoPE frequencies
        freqs = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("freqs", freqs)
        positions = torch.arange(max_len).float()
        angles = torch.outer(positions, freqs)
        self.register_buffer("cos_cache", angles.cos())
        self.register_buffer("sin_cache", angles.sin())

        # Learnable orthogonal basis: parameterize via skew-symmetric matrix
        # Q = exp(A) where A is skew-symmetric
        self.A = nn.Parameter(torch.zeros(d_model, d_model) * 0.01)

    def _get_Q(self):
        """Get orthogonal matrix from skew-symmetric parameter."""
        A = self.A - self.A.T  # ensure skew-symmetric
        return torch.matrix_exp(A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def apply_to_qk(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0):
        T = q.size(-2)
        d_head = q.size(-1)
        Q = self._get_Q()[:d_head, :d_head]  # [d_head, d_head]
        cos = self.cos_cache[offset:offset + T, :d_head // 2].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[offset:offset + T, :d_head // 2].unsqueeze(0).unsqueeze(0)

        def rotate(x):
            x = x @ Q
            x1, x2 = x[..., ::2], x[..., 1::2]
            x = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
            x = x @ Q.T
            return x

        return rotate(q), rotate(k)


class SphericalPE(nn.Module):
    """SO(d) rotation applied to hidden states on the sphere.

    Instead of adding position vectors or rotating q/k, we rotate the
    entire hidden state by a position-dependent SO(d) element. Since
    the hidden states live on S^{d-1}, and SO(d) is its isometry group,
    this is the geometrically natural positional encoding.

    Uses GRAPE-style rank-2k generators for efficiency:
    - Learn k rotation planes (pairs of orthonormal vectors u_i, v_i)
    - Learn k frequency scales
    - Rotation at position m = product of k Rodrigues rotations

    Cost: O(k*d) per token, vs O(d) for RoPE, O(d^3) for full SO(d).
    """
    def __init__(self, d_model: int, n_planes: int = None, max_len: int = 4096,
                 base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        # Default: d/2 planes (same parameter count as RoPE)
        self.n_planes = n_planes or d_model // 2

        # Learnable plane directions: start with random orthonormal pairs
        # We parameterize via a d x (2*n_planes) matrix and orthogonalize
        self.plane_params = nn.Parameter(torch.randn(d_model, 2 * self.n_planes) * 0.02)

        # Learnable frequency scales (like RoPE's theta)
        base_freqs = 1.0 / (base ** (torch.arange(0, self.n_planes).float() / self.n_planes))
        self.freq_scale = nn.Parameter(base_freqs.log())  # learn in log-space

    def _get_planes(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get orthonormalized plane basis vectors."""
        # QR decomposition to get orthonormal columns
        Q, _ = torch.linalg.qr(self.plane_params)  # [d, 2*n_planes]
        # Split into u, v pairs
        u = Q[:, 0::2]  # [d, n_planes]
        v = Q[:, 1::2]  # [d, n_planes]
        return u, v

    def _rodrigues(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
                   theta: torch.Tensor) -> torch.Tensor:
        """Apply Rodrigues rotation in the (u, v) plane by angle theta.

        For a rank-2 skew-symmetric generator A = theta*(uv^T - vu^T):
        exp(A) @ x = x + sin(theta)*(u(v·x) - v(u·x)) + (cos(theta)-1)*((u·x)u + (v·x)v)

        This rotates x within the (u,v) plane by angle theta,
        leaving the orthogonal complement unchanged.
        Cost: O(d) per plane.
        """
        # x: [..., d], u: [d], v: [d], theta: scalar
        ux = (x * u).sum(-1, keepdim=True)  # [..., 1]
        vx = (x * v).sum(-1, keepdim=True)  # [..., 1]
        sin_t = theta.sin()
        cos_t = theta.cos()
        return x + sin_t * (u * vx - v * ux) + (cos_t - 1) * (u * ux + v * vx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply position-dependent SO(d) rotation to hidden states."""
        B, T, d = x.shape
        u, v = self._get_planes()  # [d, n_planes] each
        freqs = self.freq_scale.exp()  # [n_planes]

        positions = torch.arange(T, device=x.device).float()  # [T]

        # Apply each rotation plane sequentially
        for i in range(self.n_planes):
            theta = positions * freqs[i]  # [T]
            theta = theta.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
            ui = u[:, i]  # [d]
            vi = v[:, i]  # [d]
            x = self._rodrigues(x, ui, vi, theta)

        return x

    def apply_to_qk(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0):
        """Also apply to q,k for attention position-dependence."""
        # For spherical PE, the hidden state rotation already encodes position.
        # Optionally also rotate q,k for extra positional signal in attention.
        return q, k


class SphericalPEQK(SphericalPE):
    """SphericalPE applied to both hidden states AND q,k.

    Double positional signal: hidden states get rotated (position on sphere),
    AND q,k get rotated (position-dependent attention).
    """
    def apply_to_qk(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0):
        B, H, T, d_head = q.shape
        d = self.d_model
        # Only rotate in the first min(d_head, n_planes) planes
        u, v = self._get_planes()
        freqs = self.freq_scale.exp()
        positions = torch.arange(T, device=q.device).float() + offset

        n = min(d_head // 2, self.n_planes)
        for i in range(n):
            theta = positions * freqs[i]
            theta = theta.reshape(1, 1, T, 1)
            ui = u[:d_head, i]
            vi = v[:d_head, i]
            q = self._rodrigues(q, ui, vi, theta)
            k = self._rodrigues(k, ui, vi, theta)

        return q, k


# ── Model ─────────────────────────────────────────────────────────────


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, pe: nn.Module):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.pe = pe
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # [B, H, T, d_head]

        # Apply PE to q,k
        q, k = self.pe.apply_to_qk(q, k)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(out.transpose(1, 2).reshape(B, T, C))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 pe: nn.Module, spherical: bool = False):
        super().__init__()
        self.spherical = spherical
        self.scale = math.sqrt(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, pe)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        if self.spherical:
            x = F.normalize(x, dim=-1) * self.scale
        x = x + self.ff(self.norm2(x))
        if self.spherical:
            x = F.normalize(x, dim=-1) * self.scale
        return x


class Model(nn.Module):
    def __init__(self, prime: int, d_model: int, n_heads: int, d_ff: int,
                 n_layers: int, pe_type: str = "additive", spherical: bool = True):
        super().__init__()
        self.d_model = d_model
        self.spherical = spherical

        self.embed = nn.Embedding(prime, d_model)
        self.head = nn.Linear(d_model, prime)

        # Build PE
        if pe_type == "additive":
            self.pe = AdditivePositionalEncoding(16, d_model)
        elif pe_type == "rope":
            self.pe = RoPE(d_model)
        elif pe_type == "learned_rope":
            self.pe = LearnedPlaneRoPE(d_model)
        elif pe_type == "spherical":
            self.pe = SphericalPE(d_model)
        elif pe_type == "spherical_qk":
            self.pe = SphericalPEQK(d_model)
        else:
            raise ValueError(f"Unknown PE type: {pe_type}")

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, self.pe, spherical=spherical)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.embed(x)

        # Apply PE to hidden states
        h = self.pe(h)

        if self.spherical:
            h = F.normalize(h, dim=-1) * math.sqrt(self.d_model)

        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        logits = self.head(h[:, -1])
        return logits, h[:, -1]


# ── Training ──────────────────────────────────────────────────────────


def evaluate(model, inputs, targets, device):
    model.eval()
    with torch.no_grad():
        logits, _ = model(inputs.to(device))
        loss = F.cross_entropy(logits, targets.to(device))
        acc = (logits.argmax(-1) == targets.to(device)).float().mean()
    model.train()
    return float(loss.item()), float(acc.item())


def run_config(name: str, pe_type: str, spherical: bool, args,
               train_x, train_y, val_x, val_y, device) -> dict:
    model = Model(
        prime=args.prime, d_model=args.d_model, n_heads=args.n_heads,
        d_ff=args.d_ff, n_layers=args.n_layers,
        pe_type=pe_type, spherical=spherical,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   betas=(0.9, 0.98), weight_decay=args.weight_decay)

    grok_step = args.total_steps
    memorize_step = args.total_steps
    eval_interval = max(args.total_steps // args.n_evals, 1)
    start = time.time()

    model.train()
    for step in range(1, args.total_steps + 1):
        if args.warmup > 0 and step <= args.warmup:
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * step / args.warmup

        idx = torch.randint(0, len(train_x), (args.batch_size,), device=device)
        logits, _ = model(train_x[idx])
        loss = F.cross_entropy(logits, train_y[idx])

        if torch.isnan(loss):
            return {"name": name, "grok_step": args.total_steps, "val_acc": 0.0,
                    "memorize_step": args.total_steps, "params": n_params, "wall_s": time.time() - start}

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_interval == 0:
            vl, va = evaluate(model, val_x, val_y, device)
            tl, ta = evaluate(model, train_x, train_y, device)
            if memorize_step == args.total_steps and ta >= 0.99:
                memorize_step = step
            if grok_step == args.total_steps and va >= 0.95:
                grok_step = step

            if grok_step < args.total_steps and va >= 0.95 and step > grok_step + eval_interval * 3:
                break

    wall = time.time() - start
    vl, va = evaluate(model, val_x, val_y, device)

    del model, optimizer
    torch.cuda.empty_cache()

    return {
        "name": name, "pe_type": pe_type, "spherical": spherical,
        "grok_step": grok_step, "memorize_step": memorize_step,
        "grok_delay": grok_step - memorize_step if grok_step < args.total_steps else args.total_steps,
        "val_acc": va, "val_loss": vl, "params": n_params, "wall_s": wall,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prime", type=int, default=113)
    p.add_argument("--operation", type=str, default="add")
    p.add_argument("--data-fraction", type=float, default=0.5)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=2.0)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--total-steps", type=int, default=5000)
    p.add_argument("--n-evals", type=int, default=1000)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_x, train_y, val_x, val_y = generate_dataset(
        prime=args.prime, operation=args.operation,
        data_fraction=args.data_fraction, seed=42,
    )
    train_x, train_y = train_x.to(device), train_y.to(device)
    val_x, val_y = val_x.to(device), val_y.to(device)

    configs = [
        # Standard architecture baselines
        ("std+additive",      "additive",     False),
        ("std+rope",          "rope",         False),
        ("std+learned_rope",  "learned_rope", False),

        # Spherical architecture variants
        ("sph+additive",      "additive",     True),
        ("sph+rope",          "rope",         True),
        ("sph+learned_rope",  "learned_rope", True),
        ("sph+spherical_pe",  "spherical",    True),
        ("sph+spherical_qk",  "spherical_qk", True),
    ]

    print(f"{'name':<22} {'grok':>6} {'delay':>6} {'mem':>6} {'v_acc':>8} {'params':>10} {'wall':>6}")
    print("-" * 72)

    results = []
    for name, pe_type, spherical in configs:
        r = run_config(name, pe_type, spherical, args,
                       train_x, train_y, val_x, val_y, device)
        results.append(r)
        gs = r["grok_step"]
        gd = r["grok_delay"]
        ms = r["memorize_step"]
        va = r["val_acc"]
        np_ = r["params"]
        ws = r["wall_s"]
        print(f"{name:<22} {gs:>6} {gd:>6} {ms:>6} {va:>8.4f} {np_:>10,} {ws:>6.1f}s",
              flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
