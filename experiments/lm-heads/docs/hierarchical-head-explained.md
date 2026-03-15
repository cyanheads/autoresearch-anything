# Hierarchical Head: How It Works

## The Problem

The standard LM head is a single linear projection `W @ h` where `W` is shape `(V, D)` — mapping from hidden dimension `D=576` to vocabulary size `V=32,768`. During backpropagation, the Jacobian of any function `f(h)` has rank at most `D` w.r.t. `h` (paper Eq. 9). Since `D=576 << V=32,768`, 95-99% of gradient norm is destroyed passing through this layer.

The model has sufficient capacity to solve SpamLang (proven in Prop 2.4 of the paper), but the optimization signal can't reach the backbone through the bottleneck. Result: baseline only achieves 85.5% accuracy.

## The Hierarchical Solution

Instead of one projection into 32,768 classes, decompose the prediction into two stages of ~182 classes each.

### Decomposition

Every token ID `t` in `[0, V)` is mapped to a `(cluster, position)` pair:

```
cluster = t // cluster_size
position = t % cluster_size
```

With `V = 32,768`:

| Parameter | Value | How computed |
|-----------|-------|--------------|
| `n_clusters` | 182 | `isqrt(32768) + 1` (since `181^2 = 32,761 < 32,768`) |
| `cluster_size` | 181 | `ceil(32768 / 182)` |
| Grid coverage | 32,942 | `182 * 181` (174 unused slots at the end) |

So token `t = 5000` maps to cluster `5000 // 181 = 27`, position `5000 % 181 = 113`.

### Architecture

Three learnable parameters:

```python
self.cluster_proj = nn.Linear(d_model, n_clusters, bias=False)   # W_c: (182, 576)
self.token_proj   = nn.Linear(d_model, cluster_size, bias=False)  # W_t: (181, 576)
self.cluster_bias = nn.Parameter(torch.zeros(n_clusters, cluster_size))  # B: (182, 181)
```

- **`cluster_proj`** (`W_c`): Projects hidden state `h` to 182-dim cluster logits.
- **`token_proj`** (`W_t`): Projects hidden state `h` to 181-dim token-within-cluster logits. This projection is **shared** across all clusters.
- **`cluster_bias`** (`B`): A `(182, 181)` lookup table. Each cluster has its own bias vector added to the shared token logits. This is the **only** conditioning between the two stages.

### Forward Pass (Training)

Given backbone output `h` of shape `(B, T, 576)` and input tokens `x` of shape `(B, T)`:

```
1. Shift for next-token prediction:
   targets = x[:, 1:]              # (B, T-1)
   h_shift = h[:, :-1]             # (B, T-1, 576)

2. Compute ground-truth decomposition:
   cluster_targets = targets // 181   # which cluster
   token_targets   = targets % 181    # position within cluster

3. Stage 1 — Cluster prediction:
   cluster_logits = h_shift @ W_c.T           # (B*(T-1), 182)
   cluster_loss   = CrossEntropy(cluster_logits, cluster_targets)

4. Stage 2 — Token-within-cluster prediction:
   token_logits_base = h_shift @ W_t.T        # (B*(T-1), 181) — shared across clusters
   cluster_bias      = B[cluster_targets]      # (B*(T-1), 181) — bias for TRUE cluster
   token_logits      = token_logits_base + cluster_bias
   token_loss        = CrossEntropy(token_logits, token_targets)

5. Total loss:
   loss = cluster_loss + token_loss
```

Key detail in step 4: during **training**, the bias is looked up using the **ground-truth** cluster (`cluster_targets`), not the predicted cluster. This is teacher forcing — the model learns token-within-cluster prediction conditioned on the correct cluster assignment.

### Inference (Accuracy Measurement)

During the `torch.no_grad()` block, we reconstruct the full token prediction:

```
1. Predict cluster:      top_cluster = argmax(cluster_logits)
2. Get bias for that cluster: bias = B[top_cluster]
3. Predict token in cluster:  top_token = argmax(token_logits_base + bias)
4. Reconstruct full token:    predicted = top_cluster * 181 + top_token
```

Note: at inference time, the token prediction uses the **predicted** cluster's bias, not the ground-truth. Any error in cluster prediction cascades to token prediction.

### Why This Works: The Gradient Math

The baseline head has Jacobian `J = dL/dh` that passes through `W` of shape `(V, D) = (32768, 576)`. The gradient w.r.t. `h` lives in a space of rank at most `min(V, D) = 576`, but the loss gradient `dL/d(logits)` is a `V=32768` dimensional vector, so projecting it back through `W.T` compresses it from `V` dims into `D` dims — a 57:1 compression ratio.

The hierarchical head replaces this with two independent gradient paths:

| Path | Projection shape | Output classes | Compression ratio |
|------|-----------------|----------------|-------------------|
| Cluster | `W_c`: (182, 576) | 182 | **None** — `D > classes` |
| Token | `W_t`: (181, 576) | 181 | **None** — `D > classes` |

Each stage maps 576-dim hidden state to ~181 logits. Since `576 > 181`, the Jacobian at each stage is full-rank w.r.t. the output — **no information is destroyed**. The gradient signal passes through each projection without compression.

The cluster bias `B[c]` adds 181 learnable parameters per cluster (182 * 181 = 32,942 total). Its gradient flows directly to `B` — no bottleneck there either, since it's a direct parameter, not mediated through a D-dim projection.

### Total parameter count

| Component | Shape | Parameters |
|-----------|-------|-----------|
| `cluster_proj.weight` | (182, 576) | 104,832 |
| `token_proj.weight` | (181, 576) | 104,256 |
| `cluster_bias` | (182, 181) | 32,942 |
| **Head total** | | **242,030** |
| Baseline head | (32768, 576) | 18,874,368 |

The hierarchical head uses **78x fewer parameters** than the baseline while achieving dramatically better optimization.

### Results (SpamLang, V=32768, D=576)

| Head | val_loss | val_accuracy | wall_time |
|------|----------|-------------|-----------|
| baseline | 6.5536 | 85.5% | 1313s |
| hierarchical | 0.0385 | 97.98% | 718s |

**Important**: val_loss is NOT directly comparable between heads — the baseline computes cross-entropy over 32,768 classes while hierarchical sums two cross-entropies over ~182 classes each. The losses have fundamentally different scales.

**val_accuracy IS comparable** — it measures whether the model correctly predicts the next token, regardless of how the prediction was computed internally. The improvement from 85.5% to 98.0% is a genuine apples-to-apples comparison showing the hierarchical head enables learning that the baseline head blocks.

### Relationship to Factored Head

The factored head (`FactoredHead`) uses the same `√V` decomposition but with **no conditioning between stages** — it predicts `f1` and `f2` from `h` independently with no cluster bias. Results:

| Head | val_loss | val_accuracy | Conditioning? |
|------|----------|-------------|---------------|
| hierarchical | 0.0385 | 97.98% | Yes (cluster bias) |
| factored | 0.0343 | 98.13% | No |

On SpamLang, conditioning doesn't help (and slightly hurts). This makes sense: SpamLang tokens are uniformly random, so the cluster→token dependency structure is arbitrary. On natural language with real token co-occurrence patterns, conditioning would likely matter more.
