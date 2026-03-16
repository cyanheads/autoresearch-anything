"""
experiments/grokking/spherical_pe_lm.py — Spherical PE on real language.

Compares four configurations on FineWeb-Edu:
1. Standard + additive PE (baseline)
2. Standard + RoPE (current SOTA PE)
3. Spherical + additive PE (spherical arch, standard PE)
4. Spherical + SphericalPE (fully geometric: spherical arch + SO(d) rotational PE)

The hypothesis: a geometrically consistent architecture (representations on
the sphere, positional encoding via isometry group rotations) should learn
faster and generalize better than the standard approach.
"""

import argparse
import math
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Data ──────────────────────────────────────────────────────────────


class FineWebDataset:
    def __init__(self, seq_len, device, train_tokens=50_000_000, val_tokens=1_000_000):
        import tiktoken
        from datasets import load_dataset

        self.seq_len = seq_len
        self.device = device
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab

        print(f"Loading FineWeb-Edu...", file=sys.stderr)
        ds = load_dataset("HuggingFaceFW/FineWeb-Edu", name="sample-10BT",
                          split="train", streaming=True)
        total = train_tokens + val_tokens
        tokens = []
        n = 0
        for ex in ds:
            tokens.extend(self.enc.encode_ordinary(ex["text"]))
            n = len(tokens)
            if n >= total:
                break
            if n % 5_000_000 < 1000:
                print(f"  {n:,} / {total:,}", file=sys.stderr)

        all_t = torch.tensor(tokens[:total], dtype=torch.long)
        self.train_buf = all_t[:train_tokens].to(device)
        self.val_buf = all_t[train_tokens:total].to(device)
        print(f"  train={self.train_buf.shape[0]:,} val={self.val_buf.shape[0]:,}", file=sys.stderr)

    def get_batch(self, bs, split="train"):
        buf = self.train_buf if split == "train" else self.val_buf
        starts = torch.randint(0, buf.shape[0] - self.seq_len - 1, (bs,), device=self.device)
        return torch.stack([buf[s:s + self.seq_len + 1] for s in starts])


# ── Fast Spherical PE ─────────────────────────────────────────────────


class FastSphericalPE(nn.Module):
    """Vectorized SO(d) rotational PE via batched Rodrigues formula."""
    def __init__(self, d_model, n_planes=None, base=10000.0):
        super().__init__()
        self.d_model = d_model
        self.n_planes = n_planes or d_model // 2
        self.plane_params = nn.Parameter(torch.randn(d_model, 2 * self.n_planes) * 0.02)
        base_freqs = 1.0 / (base ** (torch.arange(self.n_planes).float() / self.n_planes))
        self.freq_scale = nn.Parameter(base_freqs.log())

    def _get_planes(self):
        Q, _ = torch.linalg.qr(self.plane_params)
        return Q[:, 0::2], Q[:, 1::2]  # U, V each [d, k]

    def forward(self, x):
        B, T, d = x.shape
        U, V = self._get_planes()
        freqs = self.freq_scale.exp()

        ux = x @ U  # [B, T, k]
        vx = x @ V

        positions = torch.arange(T, device=x.device, dtype=x.dtype)
        theta = positions.unsqueeze(1) * freqs.unsqueeze(0)  # [T, k]
        sin_t = theta.sin()
        cos_t = theta.cos()

        du = sin_t.unsqueeze(0) * vx + (cos_t - 1).unsqueeze(0) * ux
        dv = -sin_t.unsqueeze(0) * ux + (cos_t - 1).unsqueeze(0) * vx

        return x + du @ U.T + dv @ V.T

    def apply_to_qk(self, q, k, offset=0):
        B, H, T, dh = q.shape
        U, V = self._get_planes()
        freqs = self.freq_scale.exp()
        n = min(dh // 2, self.n_planes)
        Uh, Vh = U[:dh, :n], V[:dh, :n]

        positions = torch.arange(T, device=q.device, dtype=q.dtype) + offset
        theta = positions.unsqueeze(1) * freqs[:n].unsqueeze(0)
        sin_t, cos_t = theta.sin(), theta.cos()

        def rot(x):
            ux = x @ Uh
            vx = x @ Vh
            du = sin_t * vx + (cos_t - 1) * ux
            dv = -sin_t * ux + (cos_t - 1) * vx
            return x + du @ Uh.T + dv @ Vh.T

        return rot(q), rot(k)


class RoPE(nn.Module):
    def __init__(self, head_dim, max_len=4096, base=10000.0):
        super().__init__()
        freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_len).float()
        angles = torch.outer(positions, freqs)
        self.register_buffer("cos_cache", angles.cos())
        self.register_buffer("sin_cache", angles.sin())

    def forward(self, x):
        return x

    def apply_to_qk(self, q, k, offset=0):
        T, dh = q.size(-2), q.size(-1)
        cos = self.cos_cache[offset:offset+T, :dh//2].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[offset:offset+T, :dh//2].unsqueeze(0).unsqueeze(0)
        def rot(x):
            x1, x2 = x[..., ::2], x[..., 1::2]
            return torch.stack([x1*cos - x2*sin, x1*sin + x2*cos], -1).flatten(-2)
        return rot(q), rot(k)


class AdditivePE(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
    def forward(self, x):
        return x + self.pe(torch.arange(x.size(1), device=x.device))
    def apply_to_qk(self, q, k, offset=0):
        return q, k


# ── Model ─────────────────────────────────────────────────────────────


class Attention(nn.Module):
    def __init__(self, d, n_heads, pe):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.pe = pe
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.pe.apply_to_qk(q, k)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(out.transpose(1, 2).reshape(B, T, C))


class FFN(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d, bias=False)
        self.w3 = nn.Linear(d, d_ff, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, d, n_heads, d_ff, pe, spherical):
        super().__init__()
        self.spherical = spherical
        self.scale = math.sqrt(d)
        self.n1 = nn.RMSNorm(d)
        self.attn = Attention(d, n_heads, pe)
        self.n2 = nn.RMSNorm(d)
        self.ff = FFN(d, d_ff)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        if self.spherical:
            x = F.normalize(x, dim=-1) * self.scale
        x = x + self.ff(self.n2(x))
        if self.spherical:
            x = F.normalize(x, dim=-1) * self.scale
        return x


class LM(nn.Module):
    def __init__(self, V, d, n_heads, n_layers, d_ff, pe_type, spherical, tie):
        super().__init__()
        self.d = d
        self.spherical = spherical
        self.tok = nn.Embedding(V, d)
        head_dim = d // n_heads

        if pe_type == "additive":
            self.pe = AdditivePE(2048, d)
        elif pe_type == "rope":
            self.pe = RoPE(head_dim)
        elif pe_type == "spherical":
            self.pe = FastSphericalPE(d)
        else:
            raise ValueError(pe_type)

        self.layers = nn.ModuleList([Block(d, n_heads, d_ff, self.pe, spherical) for _ in range(n_layers)])
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        if tie:
            self.head.weight = self.tok.weight
        nn.init.normal_(self.tok.weight, std=0.02)

    def forward(self, x):
        h = self.tok(x)
        h = self.pe(h)
        if self.spherical:
            h = F.normalize(h, dim=-1) * math.sqrt(self.d)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        logits = self.head(h)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
        return loss, logits


# ── Training ──────────────────────────────────────────────────────────


def train_one(tag, pe_type, spherical, args, dataset):
    device = torch.device("cuda")
    model = LM(dataset.vocab_size, args.d, args.heads, args.layers, args.ff,
               pe_type, spherical, tie=True).to(device)
    np_ = sum(p.numel() for p in model.parameters())
    print(f"\n[{tag}] params={np_:,} pe={pe_type} spherical={spherical}", file=sys.stderr)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                             weight_decay=args.wd)
    warmup = args.warmup
    total = args.steps
    def lr_fn(s):
        if s < warmup: return s / warmup
        return 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    start = time.time()
    log_every = max(1, total // 20)
    recent = []

    for step in range(1, total + 1):
        model.train()
        batch = dataset.get_batch(args.bs)
        with torch.autocast("cuda", torch.bfloat16):
            loss, _ = model(batch)
        if torch.isnan(loss):
            print(f"[{tag}] NaN at step {step}", file=sys.stderr)
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        recent.append(loss.item())
        if len(recent) > 100: recent.pop(0)

        if step % log_every == 0 or step == 1:
            model.eval()
            vl, vc, vt = 0, 0, 0
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                for _ in range(10):
                    vb = dataset.get_batch(args.bs, "val")
                    l, lo = model(vb)
                    vl += l.item()
                    vc += (lo[:, :-1].argmax(-1) == vb[:, 1:]).sum().item()
                    vt += vb[:, 1:].numel()
            vl /= 10
            va = vc / vt
            el = time.time() - start
            tl = sum(recent) / len(recent)
            print(f"[{tag}] {step}/{total}  t={tl:.3f} v={vl:.3f} acc={va:.4f} {el:.0f}s",
                  file=sys.stderr)

    # Final eval
    model.eval()
    vl, vc, vt = 0, 0, 0
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        for _ in range(50):
            vb = dataset.get_batch(args.bs, "val")
            l, lo = model(vb)
            vl += l.item()
            vc += (lo[:, :-1].argmax(-1) == vb[:, 1:]).sum().item()
            vt += vb[:, 1:].numel()

    wall = time.time() - start
    vram = torch.cuda.max_memory_allocated(device) / 1e6
    del model, opt, sched
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    return {"tag": tag, "val_loss": vl/50, "val_acc": vc/vt,
            "train_loss": sum(recent)/len(recent), "wall_s": wall, "vram_mb": vram, "params": np_}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=576)
    p.add_argument("--heads", type=int, default=9)
    p.add_argument("--layers", type=int, default=30)
    p.add_argument("--ff", type=int, default=1536)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--train-tokens", type=int, default=50_000_000)
    p.add_argument("--val-tokens", type=int, default=1_000_000)
    args = p.parse_args()

    device = torch.device("cuda")
    ds = FineWebDataset(args.seq, device, args.train_tokens, args.val_tokens)

    configs = [
        ("std+additive",  "additive",  False),
        ("std+rope",      "rope",      False),
        ("sph+additive",  "additive",  True),
        ("sph+rope",      "rope",      True),
        ("sph+sph_pe",    "spherical", True),
    ]

    results = []
    for tag, pe, sph in configs:
        r = train_one(tag, pe, sph, args, ds)
        results.append(r)
        print(f">>> [{r['tag']}] val_loss={r['val_loss']:.4f} acc={r['val_acc']:.4f} "
              f"wall={r['wall_s']:.0f}s vram={r['vram_mb']:.0f}MB", file=sys.stderr)

    print("\n=== RESULTS ===")
    print(f"{'config':<18} {'val_loss':>9} {'val_acc':>9} {'wall_s':>8} {'vram_MB':>8}")
    print("-" * 58)
    for r in results:
        print(f"{r['tag']:<18} {r['val_loss']:>9.4f} {r['val_acc']:>9.4f} "
              f"{r['wall_s']:>8.0f} {r['vram_mb']:>8.0f}")

    best = min(results, key=lambda r: r["val_loss"])
    print(f"\nBest: {best['tag']} (val_loss={best['val_loss']:.4f})")


if __name__ == "__main__":
    main()
