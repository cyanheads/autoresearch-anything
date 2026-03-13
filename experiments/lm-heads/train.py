"""
experiments/lm-heads/train.py

Training script for LM head gradient bottleneck experiments.
Supports two data modes:
  - spamlang: synthetic language (one token repeated) — isolates gradient bottleneck
  - fineweb: real English text from FineWeb-Edu — tests transfer to real language

Usage:
    python train.py [--head-type baseline] [--dataset spamlang|fineweb] [--steps 5000]
"""

import argparse
import math
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from heads import build_head


# ---------------------------------------------------------------------------
# Data: SpamLang (synthetic) and FineWeb-Edu (real language)
# ---------------------------------------------------------------------------

def make_spamlang_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Each sequence is one token repeated seq_len times."""
    tokens = torch.randint(0, vocab_size, (batch_size, 1), device=device)
    return tokens.expand(batch_size, seq_len).contiguous()


class FineWebDataset:
    """Streams and tokenizes FineWeb-Edu text into a token buffer.

    Pre-fills a fixed-size token buffer at init, then serves random windows.
    Separate buffers for train/val (different slices of the stream).
    """

    def __init__(self, seq_len: int, device: torch.device, train_tokens: int = 10_000_000, val_tokens: int = 500_000):
        import tiktoken
        from datasets import load_dataset

        self.seq_len = seq_len
        self.device = device
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab  # 50257

        print(f"Loading FineWeb-Edu (streaming)...", file=sys.stderr)
        ds = load_dataset("HuggingFaceFW/FineWeb-Edu", name="sample-10BT", split="train", streaming=True)

        # Tokenize into a flat buffer
        total_needed = train_tokens + val_tokens
        tokens = []
        n_tokens = 0
        for example in ds:
            encoded = self.enc.encode_ordinary(example["text"])
            tokens.extend(encoded)
            n_tokens += len(encoded)
            if n_tokens >= total_needed:
                break
            if n_tokens % 1_000_000 < len(encoded):
                print(f"  tokenized {n_tokens:,} / {total_needed:,} tokens", file=sys.stderr)

        all_tokens = torch.tensor(tokens[:total_needed], dtype=torch.long)
        self.train_buf = all_tokens[:train_tokens].to(device)
        self.val_buf = all_tokens[train_tokens:train_tokens + val_tokens].to(device)
        print(f"  train buffer: {self.train_buf.shape[0]:,} tokens, val buffer: {self.val_buf.shape[0]:,} tokens", file=sys.stderr)

    def get_batch(self, batch_size: int, split: str = "train") -> torch.Tensor:
        buf = self.train_buf if split == "train" else self.val_buf
        max_start = buf.shape[0] - self.seq_len
        starts = torch.randint(0, max_start, (batch_size,), device=self.device)
        return torch.stack([buf[s:s + self.seq_len] for s in starts])


# ---------------------------------------------------------------------------
# Minimal transformer backbone
# ---------------------------------------------------------------------------

class TransformerBackbone(nn.Module):
    """Lightweight decoder-only transformer (no KV cache — training only)."""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(1024, d_model)  # max 1024 positions
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = nn.RMSNorm(d_model)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, x: torch.Tensor, return_intermediates: list[int] | None = None) -> torch.Tensor | tuple[torch.Tensor, dict[int, torch.Tensor]]:
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(positions)
        intermediates = {}
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if return_intermediates is not None and i in return_intermediates:
                intermediates[i] = h
        h = self.norm(h)
        if return_intermediates is not None:
            return h, intermediates
        return h


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn_norm = nn.RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ff_norm = nn.RMSNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ff(self.ff_norm(x))
        return x


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
    """SwiGLU feed-forward."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Full model: backbone + head
# ---------------------------------------------------------------------------

class SpamLangModel(nn.Module):
    def __init__(self, backbone: TransformerBackbone, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head
        # multi_exit needs intermediate layer outputs
        self._return_intermediates = getattr(head, 'exit_layers', None)

    def forward(self, x: torch.Tensor):
        """Returns (loss, logits) given input token ids."""
        if self._return_intermediates is not None:
            h, intermediates = self.backbone(x, return_intermediates=self._return_intermediates)
            return self.head(h, x, intermediates=intermediates)
        h = self.backbone(x)
        return self.head(h, x)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", file=sys.stderr)

    # Model config
    d_model = args.d_model
    n_heads = args.n_heads
    n_layers = args.n_layers
    d_ff = args.d_ff
    seq_len = args.seq_len
    batch_size = args.batch_size

    # Dataset setup
    fineweb_ds = None
    if args.dataset == "fineweb":
        fineweb_ds = FineWebDataset(seq_len, device, train_tokens=args.train_tokens, val_tokens=args.val_tokens)
        vocab_size = fineweb_ds.vocab_size  # 50257 (GPT-2)
        print(f"dataset: fineweb (V={vocab_size})", file=sys.stderr)
    else:
        vocab_size = args.vocab_size
        print(f"dataset: spamlang (V={vocab_size})", file=sys.stderr)

    # Build backbone
    backbone = TransformerBackbone(vocab_size, d_model, n_heads, n_layers, d_ff)

    # Build head (pass backbone for weight tying)
    head = build_head(
        head_type=args.head_type,
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        backbone=backbone,
    )

    model = SpamLangModel(backbone, head).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_params_nonemb = n_params - backbone.tok_emb.weight.numel() - backbone.pos_emb.weight.numel()
    print(f"params: {n_params:,}  (non-embedding: {n_params_nonemb:,})", file=sys.stderr)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True,
    )

    # Cosine LR schedule with warmup
    warmup_steps = args.warmup_steps
    total_steps = args.steps

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Training
    start_time = time.time()
    best_val_loss = float("inf")
    log_interval = max(1, total_steps // 20)
    recent_losses: list[float] = []  # EMA window for smoothed train_loss

    for step in range(1, total_steps + 1):
        model.train()
        if fineweb_ds is not None:
            batch = fineweb_ds.get_batch(batch_size, split="train")
        else:
            batch = make_spamlang_batch(batch_size, seq_len, vocab_size, device)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = model(batch)

        # NaN guard
        if torch.isnan(loss):
            print("NaN detected in loss at step", step, file=sys.stderr)
            print("NaN detected")
            sys.exit(1)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        recent_losses.append(loss.item())
        if len(recent_losses) > 100:
            recent_losses.pop(0)

        if step % log_interval == 0 or step == 1:
            # Validation
            model.eval()
            val_losses = []
            val_correct = 0
            val_total = 0
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                for _ in range(10):
                    if fineweb_ds is not None:
                        vbatch = fineweb_ds.get_batch(batch_size, split="val")
                    else:
                        vbatch = make_spamlang_batch(batch_size, seq_len, vocab_size, device)
                    vloss, vlogits = model(vbatch)
                    val_losses.append(vloss.item())
                    preds = vlogits[:, :-1].argmax(dim=-1)
                    targets = vbatch[:, 1:]
                    val_correct += (preds == targets).sum().item()
                    val_total += targets.numel()

            avg_val_loss = sum(val_losses) / len(val_losses)
            val_acc = val_correct / val_total
            smooth_train = sum(recent_losses) / len(recent_losses)
            elapsed = time.time() - start_time

            print(
                f"step {step}/{total_steps}  "
                f"train_loss={smooth_train:.4f}  "
                f"val_loss={avg_val_loss:.4f}  "
                f"val_acc={val_acc:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"time={elapsed:.1f}s",
                file=sys.stderr,
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

    # Final evaluation
    model.eval()
    val_losses = []
    val_correct = 0
    val_total = 0
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(50):
            if fineweb_ds is not None:
                vbatch = fineweb_ds.get_batch(batch_size, split="val")
            else:
                vbatch = make_spamlang_batch(batch_size, seq_len, vocab_size, device)
            vloss, vlogits = model(vbatch)
            val_losses.append(vloss.item())
            preds = vlogits[:, :-1].argmax(dim=-1)
            targets = vbatch[:, 1:]
            val_correct += (preds == targets).sum().item()
            val_total += targets.numel()

    final_val_loss = sum(val_losses) / len(val_losses)
    final_val_acc = val_correct / val_total
    smooth_train = sum(recent_losses) / len(recent_losses)
    wall_time = time.time() - start_time

    peak_vram = torch.cuda.max_memory_allocated(device) / 1e6 if torch.cuda.is_available() else 0

    # Report metrics (stdout — captured by autoresearch)
    print(f"val_loss: {final_val_loss:.6f}")
    print(f"val_accuracy: {final_val_acc:.6f}")
    print(f"train_loss: {smooth_train:.6f}")
    print(f"wall_time_s: {wall_time:.1f}")
    print(f"peak_vram_mb: {peak_vram:.0f}")
    print(f"head_type: {args.head_type}")
    print(f"dataset: {args.dataset}")
    print(f"vocab_size: {vocab_size}")
    print(f"d_model: {d_model}")
    print(f"seq_len: {seq_len}")


def main():
    parser = argparse.ArgumentParser(description="LM Head Gradient Bottleneck Experiment")
    parser.add_argument("--head-type", type=str, default="baseline")
    parser.add_argument("--dataset", type=str, default="spamlang", choices=["spamlang", "fineweb"])
    parser.add_argument("--vocab-size", type=int, default=32768, help="Only used for spamlang; fineweb uses GPT-2 vocab (50257)")
    parser.add_argument("--d-model", type=int, default=576)
    parser.add_argument("--n-heads", type=int, default=9)
    parser.add_argument("--n-layers", type=int, default=30)
    parser.add_argument("--d-ff", type=int, default=1536)
    parser.add_argument("--seq-len", type=int, default=64, help="64 for spamlang, recommend 256 for fineweb")
    parser.add_argument("--batch-size", type=int, default=128, help="128 for spamlang, recommend 32 for fineweb")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--train-tokens", type=int, default=10_000_000, help="Token buffer size for fineweb training")
    parser.add_argument("--val-tokens", type=int, default=500_000, help="Token buffer size for fineweb validation")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
