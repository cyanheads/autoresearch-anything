"""
experiments/grokking/fast_spherical_pe.py — Optimize spherical PE speed.

The bottleneck: sequential Rodrigues rotations over n_planes.
Solution: vectorize all planes into a single batched operation.

Key insight: for k rotation planes with orthonormal bases (u_i, v_i),
the rotations COMMUTE if the planes are orthogonal. So we can apply
them all simultaneously instead of sequentially.

For orthogonal planes, the combined rotation is:
x' = x + sum_i [sin(θ_i)(u_i(v_i·x) - v_i(u_i·x)) + (cos(θ_i)-1)((u_i·x)u_i + (v_i·x)v_i)]

This is a single matrix-vector multiply if we precompute the rotation matrix.
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSphericalPE(nn.Module):
    """Vectorized SO(d) rotational PE using batched Rodrigues formula.

    Instead of looping over planes, we:
    1. Project x onto all planes simultaneously via matmul
    2. Apply sin/cos rotation in the projected space
    3. Reconstruct via matmul

    Cost: O(n_planes * d) via two matmuls, fully parallel on GPU.
    """
    def __init__(self, d_model: int, n_planes: int = None, base: float = 10000.0,
                 max_len: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.n_planes = n_planes or d_model // 2

        # Learnable plane basis: [d, 2*n_planes], orthogonalized at forward time
        self.plane_params = nn.Parameter(torch.randn(d_model, 2 * self.n_planes) * 0.02)

        # Learnable frequencies in log-space
        base_freqs = 1.0 / (base ** (torch.arange(self.n_planes).float() / self.n_planes))
        self.freq_scale = nn.Parameter(base_freqs.log())

        self._cached_Q = None
        self._cache_step = -1

    def _get_planes(self):
        """Get orthonormalized plane basis. Cache for the forward pass."""
        Q, _ = torch.linalg.qr(self.plane_params)  # [d, 2*n_planes]
        return Q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply position-dependent SO(d) rotation to hidden states.

        Vectorized: no Python loops over planes.
        """
        B, T, d = x.shape
        Q = self._get_planes()  # [d, 2*n_planes]
        freqs = self.freq_scale.exp()  # [n_planes]

        # u columns: Q[:, 0::2] -> [d, n_planes]
        # v columns: Q[:, 1::2] -> [d, n_planes]
        U = Q[:, 0::2]  # [d, k]
        V = Q[:, 1::2]  # [d, k]

        # Project x onto all planes: [B, T, k]
        ux = x @ U  # x · u_i for each plane
        vx = x @ V  # x · v_i for each plane

        # Position-dependent angles: [T, k]
        positions = torch.arange(T, device=x.device, dtype=x.dtype)
        theta = positions.unsqueeze(1) * freqs.unsqueeze(0)  # [T, k]
        sin_t = theta.sin()  # [T, k]
        cos_t = theta.cos()  # [T, k]

        # Rodrigues formula, vectorized over all planes:
        # x' = x + sum_i [sin(θ_i)(u_i(v_i·x) - v_i(u_i·x)) + (cos(θ_i)-1)((u_i·x)u_i + (v_i·x)v_i)]
        #
        # Rearranging:
        # delta_u_coeff = sin(θ) * vx + (cos(θ)-1) * ux   -> coefficient for each u_i
        # delta_v_coeff = -sin(θ) * ux + (cos(θ)-1) * vx  -> coefficient for each v_i
        # x' = x + U @ delta_u_coeff + V @ delta_v_coeff

        # [B, T, k]
        delta_u = sin_t.unsqueeze(0) * vx + (cos_t - 1).unsqueeze(0) * ux
        delta_v = -sin_t.unsqueeze(0) * ux + (cos_t - 1).unsqueeze(0) * vx

        # Reconstruct: [B, T, d]
        x = x + delta_u @ U.T + delta_v @ V.T

        return x

    def apply_to_qk(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0):
        """Apply rotation to q,k as well (optional, for attention PE)."""
        B, H, T, d_head = q.shape
        Q_full = self._get_planes()
        freqs = self.freq_scale.exp()

        # Use first d_head dimensions of plane basis
        n = min(d_head // 2, self.n_planes)
        U = Q_full[:d_head, :2*n:2]   # [d_head, n]
        V = Q_full[:d_head, 1:2*n:2]  # [d_head, n]

        positions = torch.arange(T, device=q.device, dtype=q.dtype) + offset
        theta = positions.unsqueeze(1) * freqs[:n].unsqueeze(0)  # [T, n]
        sin_t = theta.sin()
        cos_t = theta.cos()

        def rotate(x):
            ux = x @ U  # [B, H, T, n]
            vx = x @ V
            du = sin_t * vx + (cos_t - 1) * ux
            dv = -sin_t * ux + (cos_t - 1) * vx
            return x + du @ U.T + dv @ V.T

        return rotate(q), rotate(k)


def benchmark():
    """Compare sequential vs vectorized spherical PE speed."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = [
        (128, 2, 64),    # small: d=128, T=2 (modular arithmetic)
        (256, 2, 128),   # medium: d=256, T=2
        (256, 256, 128), # language-scale: d=256, T=256
        (576, 256, 288), # full LM scale: d=576, T=256
        (576, 512, 288), # long context
    ]

    print(f"{'d':>5} {'T':>5} {'planes':>7} {'fast_ms':>9} {'speedup':>8}")
    print("-" * 40)

    for d, T, n_planes in configs:
        pe = FastSphericalPE(d, n_planes=n_planes).to(device)
        x = torch.randn(32, T, d, device=device)

        # Warmup
        for _ in range(5):
            _ = pe(x)
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(100):
            _ = pe(x)
        torch.cuda.synchronize()
        fast_ms = (time.time() - start) / 100 * 1000

        print(f"{d:>5} {T:>5} {n_planes:>7} {fast_ms:>8.2f}ms")

    # Now test in a full model forward pass
    print("\n--- Full model forward pass (3-layer, d=256, T=256, bs=32) ---")

    class TestModel(nn.Module):
        def __init__(self, d, n_layers, pe_type):
            super().__init__()
            self.embed = nn.Embedding(1000, d)
            self.pe = FastSphericalPE(d) if pe_type == "spherical" else None
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d, 4, d*4, batch_first=True)
                for _ in range(n_layers)
            ])
            self.head = nn.Linear(d, 1000)
            self.pe_type = pe_type
            self.d = d

        def forward(self, x):
            h = self.embed(x)
            if self.pe is not None:
                h = self.pe(h)
            h = F.normalize(h, dim=-1) * math.sqrt(self.d)
            for layer in self.layers:
                h = layer(h)
                h = F.normalize(h, dim=-1) * math.sqrt(self.d)
            return self.head(h[:, -1])

    for pe_type in ["none", "spherical"]:
        model = TestModel(256, 3, pe_type).to(device)
        x = torch.randint(0, 1000, (32, 256), device=device)

        # Warmup
        for _ in range(3):
            _ = model(x)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(50):
            _ = model(x)
        torch.cuda.synchronize()
        ms = (time.time() - start) / 50 * 1000

        overhead = ""
        if pe_type == "none":
            base_ms = ms
        else:
            overhead = f"  (+{ms - base_ms:.1f}ms, {(ms/base_ms - 1)*100:.1f}% overhead)"

        print(f"  {pe_type:<12} {ms:.1f}ms/step{overhead}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    benchmark()
