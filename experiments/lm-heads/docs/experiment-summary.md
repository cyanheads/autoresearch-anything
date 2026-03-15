# LM Head Gradient Bottleneck: Experiment Summary

## The Problem

Standard language models predict the next token via a linear projection from hidden states to vocabulary logits:

```
logits = h @ W.T    where W in R^{V x D}, h in R^D
```

Godey & Artzi (2026) prove that for any function `f(h)` mapping D-dimensional hidden states to a loss, the Jacobian w.r.t. `h` has rank at most D (Eq. 9 of the paper):

```
nabla_H L = nabla_L L . J_f(H)    where rank(J_f) <= D
```

When `V >> D` (vocabulary size >> hidden dimension), the backward pass through the LM head projects a V-dimensional gradient into D-space via `W`, destroying the components in W's null space. The paper measures 95-99% of gradient norm lost in this projection (Fig. 6), with cosine similarity between the projected and ideal gradient at only 0.1-0.2 (Fig. 7).

This is not a modeling capacity issue — it's purely an optimization issue. The model has sufficient expressivity to solve the task, but the gradient signal can't reach the backbone effectively.

## The Testbed: SpamLang

SpamLang is a synthetic language where each sequence is one token repeated:

```
[42, 42, 42, 42, ...]    (token 42 repeated for all positions)
```

The model provably has sufficient capacity to solve this (Prop. 2.4 of the paper). Any failure is purely due to the gradient bottleneck. This makes SpamLang an ideal controlled testbed: if a head design can't learn SpamLang, it definitely can't learn real language.

Our configuration: `V = 32,768`, `D = 576`, `seq_len = 64`, 30-layer transformer (~120M non-embedding params), trained for 3,000 steps with AdamW (lr=5e-4, cosine schedule).

## The Math: Why Reducing Output Classes Works

The key insight is that the bottleneck severity depends on the ratio of output classes to hidden dimension.

**Baseline head**: Projects `R^D -> R^V`. The gradient passes through `W in R^{V x D}`. Since `D = 576 << V = 32,768`, the projection compresses a 32,768-dim gradient vector into 576 dimensions — a 57:1 compression ratio. Most gradient information is destroyed.

**Factored head**: Decomposes token prediction into two stages, each with `~sqrt(V) ~ 182` classes:

```
token_id = f1 * 182 + f2
f1 = token_id // 182    (which cluster, 182 classes)
f2 = token_id % 182     (position within cluster, 182 classes)
```

Each stage projects `R^D -> R^182`. Since `D = 576 > 182`, the Jacobian at each stage is **full rank** w.r.t. the output — no information is destroyed. The gradient passes through without compression.

**Three-level factored**: Decomposes into three stages with `~V^(1/3) ~ 32` classes each:

```
token_id = f1 * 32^2 + f2 * 32 + f3
```

Each stage projects `R^576 -> R^32`. Even less compression — the gradient has 576 dimensions of freedom to encode 32 classes of information.

**General principle**: For a K-way classification with hidden dim D, the gradient bottleneck is:
- K <= D: no bottleneck (full-rank Jacobian)
- K > D: compression ratio K/D, destroying (K-D)/K fraction of gradient info

## Taxonomy of Approaches

### Category 1: Reduce effective V per prediction stage

**These work.** By decomposing the V-class prediction into multiple stages of K-class predictions where K <= D, each stage has a full-rank Jacobian and gradient flows through without compression.

| Head | Decomposition | Classes/stage | val_accuracy | val_loss |
|------|--------------|---------------|-------------|----------|
| baseline | None (V=32768) | 32,768 | 85.5% | 6.554 |
| hierarchical | 2-level with cluster bias conditioning | ~182 | 98.0% | 0.039 |
| factored | 2-level independent | ~182 | 98.1% | 0.034 |
| factored3 | 3-level independent | ~32 | 98.2% | 0.022 |

Key observations:
- **Conditioning doesn't matter on SpamLang.** Hierarchical (with cluster bias) and factored (independent) achieve nearly identical accuracy. This is expected: SpamLang tokens are uniformly distributed, so cluster identity carries no information about within-cluster position.
- **More levels = lower per-stage loss.** factored3 has the lowest val_loss (0.022) because each of its 3 stages classifies over ~32 classes — trivially easy for a 576-dim representation. But accuracy is similar across all factored variants (~98%).
- **val_loss is NOT comparable across head types.** Baseline computes CE over 32,768 classes; factored sums CE over two 182-class problems. Different scales. val_accuracy IS comparable (always measures: did we predict the right token?).

### Category 2: Bypass the head with D-space losses

**Mixed results.** Operating entirely in D-space avoids the bottleneck, but the training signal is qualitatively different.

| Head | Approach | val_accuracy | val_loss | Status |
|------|----------|-------------|----------|--------|
| contrastive_aux | CE + InfoNCE in D-space | 85.4% | 6.197 | discard |
| embedding_pred | Cosine + InfoNCE, no V-dim projection | 98.4% | 0.015 | keep (SpamLang only) |

- **contrastive_aux failed** because the primary loss (CE through full V-dim head) still bottlenecks. The aux loss provides additional signal, but it can't compensate for the primary head's broken gradients.
- **embedding_pred works brilliantly on SpamLang** — it bypasses the bottleneck entirely by predicting in D-space. But it fundamentally **cannot work on real language** because: (1) it can't express probability distributions over multiple valid continuations, (2) semantically similar embeddings cause nearest-neighbor confusion, (3) it produces similarity scores, not calibrated probabilities needed for sampling/generation.
- **factored_emb_aux (factored + embedding aux)** — the aux loss competed for gradient bandwidth rather than helping. No improvement over plain factored (98.3% vs 98.1%).

### Category 3: Alternative gradient paths

**Partially effective.** Providing gradient shortcuts helps, but each exit still bottlenecks if it uses full V-dim projection.

| Head | Approach | val_accuracy | val_loss | Status |
|------|----------|-------------|----------|--------|
| multi_exit | V-dim heads at 3 layers, weighted sum | 96.2% | 6.595 | park |

Multi-exit improves accuracy over baseline (96.2% vs 85.5%) by providing gradient shortcuts to earlier layers. But each exit still uses a `(V, D)` projection, so each exit is individually bottlenecked. The improvement comes from the gradient paths being shorter (fewer layers to backpropagate through), not from fixing the bottleneck itself.

### Category 4: Modified backward pass

**In progress.** The `learned_backward` head replaces `W.T` in the backward pass with a learned matrix `B`, hoping to find gradient projections that preserve more information. Crashed on first attempt (bf16 dtype mismatch), fix committed, not yet re-run.

## What We've Learned

1. **The paper's theory is correct.** Reducing output classes below D eliminates the bottleneck — this is the dominant effect. Everything that keeps full V-dim output fails or partially fails.

2. **Conditioning doesn't matter on SpamLang.** Hierarchical (conditioned) and factored (independent) perform identically. This is expected for uniform token distributions, but on real language with structured co-occurrence, conditioning should matter.

3. **Auxiliary D-space losses don't help when added to a working head.** factored_emb_aux showed no improvement over plain factored. The factored head already provides good gradients, so the extra signal is redundant.

4. **More factoring levels = lower loss, same accuracy.** Going from 2-level to 3-level reduced per-stage CE loss but accuracy stayed ~98%. The remaining ~2% error is likely not due to the bottleneck but to model capacity or training duration.

5. **Embedding prediction is a SpamLang artifact.** Its success reflects the trivial structure of the task (deterministic next token, uniform distribution), not a general solution to the bottleneck.

## Limitations of SpamLang as a Testbed

SpamLang is useful for isolating the gradient bottleneck but has properties that make some results non-transferable to real language:

| Property | SpamLang | Real Language |
|----------|----------|---------------|
| Next token | Deterministic given context | Stochastic — many valid continuations |
| Token distribution | Uniform (all 32K tokens equally likely) | Highly skewed (Zipf's law) |
| Token relationships | None — arbitrary IDs | Rich co-occurrence structure |
| Cluster structure | Meaningless (arithmetic decomposition) | Semantic grouping possible |
| Sequence structure | Trivial (constant repetition) | Complex long-range dependencies |

These differences mean:
- **Factored > hierarchical on SpamLang** because conditioning is useless with uniform tokens. On real language, **hierarchical likely wins** because cluster identity is informative.
- **Embedding prediction works on SpamLang** because the answer is always a single token. On real language it can't express distributions.
- **The arbitrary f1/f2 decomposition works on SpamLang** because all tokens are interchangeable. On real language, **semantic clustering** (grouping related tokens together) would likely improve factored/hierarchical heads significantly.

## Next Direction: Real Language

To validate whether the gradient bottleneck findings transfer, we need to test on real language with:

1. **A real tokenizer and vocabulary** — actual BPE tokens with semantic structure
2. **A real text corpus** — where next-token prediction requires modeling complex distributions
3. **Semantic clustering** — group tokens by embedding similarity rather than arbitrary arithmetic, so "cluster prediction" means "what kind of word?" and "within-cluster prediction" means "which specific word?"
4. **Proper hierarchical conditioning** — which should matter now that cluster identity carries semantic information

The key question: does the factored/hierarchical head still provide a meaningful convergence speedup on real language, or was the SpamLang result purely an artifact of the extreme V/D ratio (32768/576 = 57:1)?

The paper's own large-scale experiments (2B params, D=4096, V=32000) showed a 16x convergence slowdown from the bottleneck — suggesting the effect is real but less catastrophic at higher D. Testing at our scale (D=576) with real language would reveal whether the factored head's advantage persists when the task itself is harder.
