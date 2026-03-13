# BLT Architecture Search

## What is BLT?

The Byte Latent Transformer (Meta, ACL 2025) is a tokenizer-free language model that operates directly on raw bytes. Instead of a fixed tokenizer, it uses **entropy-based dynamic patching** to group bytes into variable-length patches, spending more compute on surprising content and less on predictable content.

## Architecture

```
Raw bytes → [Local Encoder] → patches → [Global Transformer] → patches → [Local Decoder] → byte predictions
              1 layer, h_E          12 layers, h_G = 2·h_E           4 layers, h_E
```

### Local Encoder
- Byte embedding (256 vocab) + hash n-gram embeddings (n=3..6, 50K buckets each)
- 1 transformer layer with sliding window attention (w=512)
- Cross-attention pools bytes → patches (mean-pool query init, attend to own patch's bytes)
- Projects h_E → h_G

### Global Transformer
- Standard causal transformer, operates on **patches not bytes**
- Most parameters and FLOPs live here
- Sequence length = num_bytes / patch_size (e.g., 4096 bytes / 8 = 512 patches)

### Local Decoder
- Cross-attention **before** self-attention (reversed from encoder, per BLT paper)
- 4 layers (paper found decoder should be deeper than encoder: 7-9 at full scale)
- Projects h_G → h_E for cross-attention context
- Output head: Linear(h_E, 256) tied with byte embedding

## Experiment Design

### Primary metric
- **val_bpb** (validation bits-per-byte): lower is better

### Axes to explore
1. **Patch size**: 4, 6, 8, 12, 16 — larger = cheaper inference, but coarser representation
2. **Encoder/decoder depth ratio**: paper uses 1/7-9 at scale; we start at 1/4
3. **Dimension ratio k** (h_G / h_E): paper uses k=2, never swept
4. **N-gram config**: which sizes, how many buckets, embedding dim
5. **Patching strategy**: fixed-size (baseline) → entropy-based (requires training a small patcher)
6. **Global transformer config**: depth vs width, head count, FFN multiplier
7. **Alternative local architectures**: could SSM (Mamba-style) replace attention in encoder/decoder?
8. **Optimizer**: AdamW vs Muon, schedule, weight decay

### Key constraints
- Single RTX 3090 (24GB VRAM)
- ~8 minute wall-clock budget per run
- Target model size: 50-100M parameters

## Data

100MB FineWeb-Edu subset, byte-level (no tokenization). 5MB validation split.
Loaded via `data.py` (immutable).

## References

- [Byte Latent Transformer paper](https://arxiv.org/abs/2412.09871)
- [Official code](https://github.com/facebookresearch/blt)
- [Mamba-3 (ICLR 2026)](https://openreview.net/forum?id=HwCvaJOiCj) — potential SSM encoder
- [Forgetting Transformer (FoX)](https://arxiv.org/abs/2503.02130) — forget-gated attention variant
