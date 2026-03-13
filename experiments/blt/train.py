"""
Byte Latent Transformer (BLT) — Baseline Implementation

Architecture:
  Local Encoder (byte → patch) → Global Transformer (patch → patch) → Local Decoder (patch → byte)

This is the mutable artifact. Everything is fair game:
  - Encoder/decoder depth and width
  - Global transformer config
  - Patching strategy (fixed-size vs entropy-based)
  - N-gram hash embeddings
  - Optimizer and schedule
  - Batch size, sequence length
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from data import get_dataloaders

# ============================================================================
# Config
# ============================================================================

DEVICE = "cuda:1"  # RTX 3090

# Architecture
VOCAB_SIZE = 256          # raw bytes
PATCH_SIZE = 8            # fixed patch size (bytes per patch)
GLOBAL_DIM = 512          # global transformer hidden dim
LOCAL_DIM = 256           # local encoder/decoder hidden dim (k=2 ratio)
GLOBAL_LAYERS = 12        # global transformer depth
GLOBAL_HEADS = 8          # global transformer attention heads
ENCODER_LAYERS = 1        # local encoder depth
DECODER_LAYERS = 4        # local decoder depth
LOCAL_HEADS = 4           # local encoder/decoder attention heads
LOCAL_WINDOW = 512        # sliding window size for local models (bytes)
NGRAM_SIZES = [3, 4, 5, 6]  # hash n-gram sizes
NGRAM_BUCKETS = 50000     # hash buckets per n-gram size
DROPOUT = 0.0

# Training
SEQ_LEN = 4096            # bytes per sequence
BATCH_SIZE = 16
LR = 4e-4
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 200
MAX_STEPS = 2000          # total training steps
EVAL_INTERVAL = 200       # evaluate every N steps
GRAD_CLIP = 1.0
USE_AMP = True            # mixed precision


# ============================================================================
# N-gram Hash Embeddings
# ============================================================================

class NGramHashEmbedding(nn.Module):
    """Hash-based n-gram embeddings for byte sequences."""

    def __init__(self, ngram_sizes: list[int], num_buckets: int, embed_dim: int):
        super().__init__()
        self.ngram_sizes = ngram_sizes
        self.num_buckets = num_buckets
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_buckets, embed_dim) for _ in ngram_sizes
        ])
        self.scale = 1.0 / (len(ngram_sizes) + 1)  # +1 for base embedding

    def _rolling_hash(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """Compute rolling polynomial hash for n-grams.

        Args:
            x: (batch, seq_len) byte values
            n: n-gram size

        Returns:
            (batch, seq_len) hash indices, 0 for positions without full n-gram
        """
        B, T = x.shape
        if T < n:
            return torch.zeros(B, T, dtype=torch.long, device=x.device)

        # Simple polynomial rolling hash
        base = 257
        hashes = torch.zeros(B, T, dtype=torch.long, device=x.device)

        for i in range(n - 1, T):
            h = torch.zeros(B, dtype=torch.long, device=x.device)
            for j in range(n):
                h = h * base + x[:, i - n + 1 + j].long()
            hashes[:, i] = h % self.num_buckets

        return hashes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) byte values [0, 255]

        Returns:
            (batch, seq_len, embed_dim) n-gram embeddings (to be added to base)
        """
        total = torch.zeros(
            x.shape[0], x.shape[1], self.embeddings[0].embedding_dim,
            device=x.device, dtype=self.embeddings[0].weight.dtype,
        )

        for ngram_size, emb in zip(self.ngram_sizes, self.embeddings):
            hashes = self._rolling_hash(x, ngram_size)
            total = total + emb(hashes)

        return total * self.scale


# ============================================================================
# Transformer Components
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, window_size: int | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # each: (B, T, H, D)
        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=causal,
            dropout_p=self.dropout if self.training else 0.0,
        )

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class CrossAttention(nn.Module):
    def __init__(self, dim_q: int, dim_kv: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim_q // num_heads

        self.q_proj = nn.Linear(dim_q, dim_q, bias=False)
        self.kv_proj = nn.Linear(dim_kv, 2 * dim_q, bias=False)
        self.out = nn.Linear(dim_q, dim_q, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(context).reshape(B, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
        )

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mult)
        # SwiGLU
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, window_size: int | None = None):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = SelfAttention(dim, num_heads, dropout, window_size)
        self.norm2 = RMSNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), causal=causal)
        x = x + self.ff(self.norm2(x))
        return x


class DecoderBlock(nn.Module):
    """Transformer block with cross-attention (for the local decoder)."""

    def __init__(self, dim: int, cross_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm_cross = RMSNorm(dim)
        self.cross_attn = CrossAttention(dim, cross_dim, num_heads, dropout)
        self.norm1 = RMSNorm(dim)
        self.attn = SelfAttention(dim, num_heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor, causal: bool = True) -> torch.Tensor:
        # Cross-attention first (before self-attention, per BLT paper)
        x = x + self.cross_attn(self.norm_cross(x), context)
        x = x + self.attn(self.norm1(x), causal=causal)
        x = x + self.ff(self.norm2(x))
        return x


# ============================================================================
# BLT Model
# ============================================================================

class ByteLatentTransformer(nn.Module):
    """
    Byte Latent Transformer with fixed-size patching.

    Architecture:
        1. Byte embedding + n-gram hash embeddings
        2. Local encoder (small transformer) processes bytes
        3. Cross-attention pools bytes → patches
        4. Global transformer processes patches
        5. Local decoder (with cross-attention from patches) predicts bytes
    """

    def __init__(self):
        super().__init__()

        # Byte embedding
        self.byte_embed = nn.Embedding(VOCAB_SIZE, LOCAL_DIM)
        self.ngram_embed = NGramHashEmbedding(NGRAM_SIZES, NGRAM_BUCKETS, LOCAL_DIM)

        # Local encoder
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(LOCAL_DIM, LOCAL_HEADS, DROPOUT, LOCAL_WINDOW)
            for _ in range(ENCODER_LAYERS)
        ])

        # Encoder cross-attention: pool bytes → patches
        # Query: one per patch (initialized by mean-pooling byte embeds within patch)
        # Key/Value: byte representations from encoder
        self.encoder_pool = CrossAttention(LOCAL_DIM, LOCAL_DIM, LOCAL_HEADS, DROPOUT)
        self.encoder_pool_norm = RMSNorm(LOCAL_DIM)

        # Project local dim → global dim
        self.up_proj = nn.Linear(LOCAL_DIM, GLOBAL_DIM, bias=False)

        # Global transformer
        self.global_layers = nn.ModuleList([
            TransformerBlock(GLOBAL_DIM, GLOBAL_HEADS, DROPOUT)
            for _ in range(GLOBAL_LAYERS)
        ])
        self.global_norm = RMSNorm(GLOBAL_DIM)

        # Project global dim → local dim for decoder cross-attention
        self.down_proj = nn.Linear(GLOBAL_DIM, LOCAL_DIM, bias=False)

        # Local decoder (with cross-attention from patch representations)
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(LOCAL_DIM, LOCAL_DIM, LOCAL_HEADS, DROPOUT)
            for _ in range(DECODER_LAYERS)
        ])
        self.decoder_norm = RMSNorm(LOCAL_DIM)

        # Output head
        self.output_head = nn.Linear(LOCAL_DIM, VOCAB_SIZE, bias=False)

        # Tie byte embedding weights with output head
        self.output_head.weight = self.byte_embed.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _bytes_to_patches(self, byte_repr: torch.Tensor) -> torch.Tensor:
        """Pool byte representations into patch representations via cross-attention.

        Args:
            byte_repr: (B, T, local_dim) byte-level representations from encoder

        Returns:
            (B, T // patch_size, local_dim) patch representations
        """
        B, T, D = byte_repr.shape
        num_patches = T // PATCH_SIZE

        # Truncate to exact multiple of patch_size
        byte_repr = byte_repr[:, :num_patches * PATCH_SIZE, :]

        # Create patch queries by mean-pooling bytes within each patch
        reshaped = byte_repr.reshape(B, num_patches, PATCH_SIZE, D)
        patch_queries = reshaped.mean(dim=2)  # (B, num_patches, D)

        # Cross-attend: each patch query attends to its own bytes
        # For simplicity, we attend to all bytes (masked would be better but complex)
        patch_repr = self.encoder_pool(
            self.encoder_pool_norm(patch_queries),
            byte_repr,
        )

        return patch_repr

    def _expand_patches_to_bytes(self, patch_repr: torch.Tensor, num_bytes: int) -> torch.Tensor:
        """Expand patch representations back to byte-level for decoder cross-attention.

        Args:
            patch_repr: (B, num_patches, local_dim)
            num_bytes: target number of bytes

        Returns:
            (B, num_bytes, local_dim) — each byte gets its patch's representation
        """
        B, P, D = patch_repr.shape
        # Repeat each patch for PATCH_SIZE bytes
        expanded = patch_repr.unsqueeze(2).expand(B, P, PATCH_SIZE, D)
        expanded = expanded.reshape(B, P * PATCH_SIZE, D)
        return expanded[:, :num_bytes, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T) byte values [0, 255]

        Returns:
            (B, T, 256) logits over byte vocabulary
        """
        B, T = x.shape
        num_patches = T // PATCH_SIZE
        T_aligned = num_patches * PATCH_SIZE

        # 1. Byte embedding + n-gram
        byte_emb = self.byte_embed(x) + self.ngram_embed(x)  # (B, T, local_dim)

        # 2. Local encoder
        h = byte_emb
        for layer in self.encoder_layers:
            h = layer(h, causal=True)

        # 3. Pool bytes → patches via cross-attention
        patch_repr = self._bytes_to_patches(h)  # (B, num_patches, local_dim)

        # 4. Project up to global dim
        patch_repr = self.up_proj(patch_repr)  # (B, num_patches, global_dim)

        # 5. Global transformer
        for layer in self.global_layers:
            patch_repr = layer(patch_repr, causal=True)
        patch_repr = self.global_norm(patch_repr)

        # 6. Project down to local dim
        patch_repr = self.down_proj(patch_repr)  # (B, num_patches, local_dim)

        # 7. Expand patches back to byte positions for cross-attention
        patch_context = self._expand_patches_to_bytes(patch_repr, T_aligned)

        # 8. Local decoder with cross-attention from patches
        h_dec = byte_emb[:, :T_aligned, :]
        for layer in self.decoder_layers:
            h_dec = layer(h_dec, patch_context, causal=True)
        h_dec = self.decoder_norm(h_dec)

        # 9. Output logits
        logits = self.output_head(h_dec)  # (B, T_aligned, 256)

        # Pad back to original length if needed
        if T_aligned < T:
            pad = torch.zeros(B, T - T_aligned, VOCAB_SIZE, device=logits.device, dtype=logits.dtype)
            logits = torch.cat([logits, pad], dim=1)

        return logits


# ============================================================================
# Training
# ============================================================================

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def compute_bpb(loss: float) -> float:
    """Convert cross-entropy loss (nats) to bits-per-byte."""
    return loss / math.log(2)


@torch.no_grad()
def evaluate(model: nn.Module, val_loader, max_batches: int = 50) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)

        with autocast("cuda", dtype=torch.bfloat16, enabled=USE_AMP):
            logits = model(x)
            T_aligned = (x.shape[1] // PATCH_SIZE) * PATCH_SIZE
            loss = F.cross_entropy(
                logits[:, :T_aligned, :].reshape(-1, VOCAB_SIZE),
                y[:, :T_aligned].reshape(-1),
            )

        total_loss += loss.item()
        n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)


def train():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    print(f"Device: {DEVICE}")
    print(f"Patch size: {PATCH_SIZE}")
    print(f"Global dim: {GLOBAL_DIM}, Local dim: {LOCAL_DIM}")
    print(f"Global layers: {GLOBAL_LAYERS}, Encoder layers: {ENCODER_LAYERS}, Decoder layers: {DECODER_LAYERS}")
    print(f"Seq len: {SEQ_LEN} bytes = {SEQ_LEN // PATCH_SIZE} patches")

    # Data
    train_loader, val_loader = get_dataloaders(
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        num_workers=2,
    )

    # Model
    model = ByteLatentTransformer().to(DEVICE)
    total_params = count_params(model)
    print(f"total_params_m: {total_params / 1e6:.1f}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
        eps=1e-8,
    )

    # LR schedule: linear warmup + cosine decay
    def lr_schedule(step: int) -> float:
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    scaler = GradScaler("cuda", enabled=USE_AMP)

    # Training loop
    model.train()
    step = 0
    start_time = time.time()
    total_bytes = 0
    best_val_bpb = float("inf")

    while step < MAX_STEPS:
        for x, y in train_loader:
            if step >= MAX_STEPS:
                break

            x, y = x.to(DEVICE), y.to(DEVICE)

            with autocast("cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                logits = model(x)
                T_aligned = (x.shape[1] // PATCH_SIZE) * PATCH_SIZE
                loss = F.cross_entropy(
                    logits[:, :T_aligned, :].reshape(-1, VOCAB_SIZE),
                    y[:, :T_aligned].reshape(-1),
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            total_bytes += x.shape[0] * T_aligned
            step += 1

            train_bpb = compute_bpb(loss.item())

            if step % 50 == 0:
                elapsed = time.time() - start_time
                throughput = total_bytes / elapsed
                print(
                    f"step {step}/{MAX_STEPS} | "
                    f"train_bpb: {train_bpb:.4f} | "
                    f"lr: {scheduler.get_last_lr()[0]:.2e} | "
                    f"throughput: {throughput:.0f} bytes/s"
                )

            # Evaluate
            if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
                val_loss = evaluate(model, val_loader)
                val_bpb = compute_bpb(val_loss)

                if val_bpb < best_val_bpb:
                    best_val_bpb = val_bpb

                print(f"--- Eval step {step}: val_bpb={val_bpb:.4f} (best={best_val_bpb:.4f}) ---")

    # Final metrics
    elapsed = time.time() - start_time
    throughput = total_bytes / elapsed

    # Peak VRAM
    peak_vram_mb = torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024)

    print(f"\n=== Final Results ===")
    print(f"val_bpb: {best_val_bpb:.4f}")
    print(f"train_bpb: {train_bpb:.4f}")
    print(f"throughput_bytes_per_sec: {throughput:.0f}")
    print(f"peak_vram_mb: {peak_vram_mb:.0f}")
    print(f"total_params_m: {total_params / 1e6:.1f}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    train()
