"""
experiments/grokking/spherical_lm.py — Test spherical architecture on real language.

Compares standard vs spherical transformer on FineWeb-Edu next-token prediction.
Same architecture as lm-heads experiment (30-layer, d=576, SwiGLU) but with
spherical residual stream normalization.

The question: does the topological constraint that eliminates grokking delay on
modular arithmetic also help real language modeling?
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
    def __init__(self, seq_len: int, device: torch.device,
                 train_tokens: int = 50_000_000, val_tokens: int = 1_000_000):
        import tiktoken
        from datasets import load_dataset

        self.seq_len = seq_len
        self.device = device
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab

        print(f"Loading FineWeb-Edu...", file=sys.stderr)
        ds = load_dataset("HuggingFaceFW/FineWeb-Edu", name="sample-10BT",
                          split="train", streaming=True)

        total_needed = train_tokens + val_tokens
        tokens = []
        n = 0
        for example in ds:
            encoded = self.enc.encode_ordinary(example["text"])
            tokens.extend(encoded)
            n += len(encoded)
            if n >= total_needed:
                break
            if n % 5_000_000 < len(encoded):
                print(f"  {n:,} / {total_needed:,} tokens", file=sys.stderr)

        all_tokens = torch.tensor(tokens[:total_needed], dtype=torch.long)
        self.train_buf = all_tokens[:train_tokens].to(device)
        self.val_buf = all_tokens[train_tokens:train_tokens + val_tokens].to(device)
        print(f"  train: {self.train_buf.shape[0]:,}, val: {self.val_buf.shape[0]:,}",
              file=sys.stderr)

    def get_batch(self, batch_size: int, split: str = "train") -> torch.Tensor:
        buf = self.train_buf if split == "train" else self.val_buf
        max_start = buf.shape[0] - self.seq_len - 1
        starts = torch.randint(0, max_start, (batch_size,), device=self.device)
        return torch.stack([buf[s:s + self.seq_len + 1] for s in starts])


# ── Model ─────────────────────────────────────────────────────────────


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(out.transpose(1, 2).reshape(B, T, C))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, spherical: bool = False):
        super().__init__()
        self.spherical = spherical
        self.scale = math.sqrt(d_model)
        self.attn_norm = nn.RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ff_norm = nn.RMSNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        if self.spherical:
            x = F.normalize(x, dim=-1) * self.scale
        x = x + self.ff(self.ff_norm(x))
        if self.spherical:
            x = F.normalize(x, dim=-1) * self.scale
        return x


class LMTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, spherical: bool = False,
                 tie_weights: bool = True):
        super().__init__()
        self.d_model = d_model
        self.spherical = spherical

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(1024, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, spherical=spherical)
            for _ in range(n_layers)
        ])
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)

        if self.spherical:
            h = F.normalize(h, dim=-1) * math.sqrt(self.d_model)

        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        logits = self.head(h)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1),
        )
        return loss, logits


# ── Training ──────────────────────────────────────────────────────────


def train_model(args, spherical: bool, dataset: FineWebDataset) -> dict:
    device = torch.device("cuda")
    tag = "spherical" if spherical else "standard"

    model = LMTransformer(
        vocab_size=dataset.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        spherical=spherical,
        tie_weights=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[{tag}] params: {n_params:,}", file=sys.stderr)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )

    warmup = args.warmup_steps
    total = args.steps

    def lr_fn(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    start = time.time()
    log_interval = max(1, total // 20)
    recent = []

    for step in range(1, total + 1):
        model.train()
        batch = dataset.get_batch(args.batch_size, "train")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model(batch)

        if torch.isnan(loss):
            print(f"[{tag}] NaN at step {step}", file=sys.stderr)
            return {"tag": tag, "val_loss": float("nan"), "val_acc": 0.0,
                    "train_loss": float("nan"), "wall_s": time.time() - start}

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        recent.append(loss.item())
        if len(recent) > 100:
            recent.pop(0)

        if step % log_interval == 0 or step == 1:
            model.eval()
            vloss_sum, vcorrect, vtotal = 0.0, 0, 0
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(10):
                    vb = dataset.get_batch(args.batch_size, "val")
                    vl, vlogits = model(vb)
                    vloss_sum += vl.item()
                    preds = vlogits[:, :-1].argmax(-1)
                    vcorrect += (preds == vb[:, 1:]).sum().item()
                    vtotal += vb[:, 1:].numel()
            vl_avg = vloss_sum / 10
            vacc = vcorrect / vtotal
            tl = sum(recent) / len(recent)
            el = time.time() - start
            print(f"[{tag}] step {step}/{total}  t_loss={tl:.4f}  "
                  f"v_loss={vl_avg:.4f}  v_acc={vacc:.4f}  {el:.0f}s",
                  file=sys.stderr)

    # Final eval
    model.eval()
    vloss_sum, vcorrect, vtotal = 0.0, 0, 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(50):
            vb = dataset.get_batch(args.batch_size, "val")
            vl, vlogits = model(vb)
            vloss_sum += vl.item()
            preds = vlogits[:, :-1].argmax(-1)
            vcorrect += (preds == vb[:, 1:]).sum().item()
            vtotal += vb[:, 1:].numel()

    wall = time.time() - start
    vram = torch.cuda.max_memory_allocated(device) / 1e6

    result = {
        "tag": tag,
        "val_loss": vloss_sum / 50,
        "val_acc": vcorrect / vtotal,
        "train_loss": sum(recent) / len(recent),
        "wall_s": wall,
        "vram_mb": vram,
        "params": n_params,
    }

    # Reset VRAM for next model
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d-model", type=int, default=576)
    p.add_argument("--n-heads", type=int, default=9)
    p.add_argument("--n-layers", type=int, default=30)
    p.add_argument("--d-ff", type=int, default=1536)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--train-tokens", type=int, default=50_000_000)
    p.add_argument("--val-tokens", type=int, default=1_000_000)
    args = p.parse_args()

    device = torch.device("cuda")
    dataset = FineWebDataset(args.seq_len, device,
                             train_tokens=args.train_tokens,
                             val_tokens=args.val_tokens)

    print(f"\nArchitecture: {args.n_layers}L d={args.d_model} ff={args.d_ff} "
          f"h={args.n_heads} seq={args.seq_len}", file=sys.stderr)
    print(f"Training: {args.steps} steps, bs={args.batch_size}, "
          f"lr={args.lr}, wd={args.weight_decay}", file=sys.stderr)
    print(f"Data: {args.train_tokens:,} train tokens, "
          f"{args.val_tokens:,} val tokens\n", file=sys.stderr)

    results = []
    for spherical in [False, True]:
        r = train_model(args, spherical, dataset)
        results.append(r)
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"[{r['tag']}] FINAL: val_loss={r['val_loss']:.4f}  "
              f"val_acc={r['val_acc']:.4f}  wall={r['wall_s']:.0f}s  "
              f"vram={r['vram_mb']:.0f}MB", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)

    # Summary to stdout
    print("\n=== SPHERICAL vs STANDARD on FineWeb ===")
    for r in results:
        print(f"{r['tag']:<12} val_loss={r['val_loss']:.4f}  "
              f"val_acc={r['val_acc']:.4f}  wall={r['wall_s']:.0f}s  "
              f"vram={r['vram_mb']:.0f}MB")

    std = results[0]
    sph = results[1]
    delta_loss = sph["val_loss"] - std["val_loss"]
    delta_acc = sph["val_acc"] - std["val_acc"]
    print(f"\ndelta: val_loss={delta_loss:+.4f}  val_acc={delta_acc:+.4f}")
    if delta_loss < 0:
        print(">>> Spherical WINS on real language")
    else:
        print(">>> Standard wins (spherical constraint hurts)")


if __name__ == "__main__":
    main()
