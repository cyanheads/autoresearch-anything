## The Constraint to Work Around

The paper proves (Eq 9) that for *any* function $f_\theta(H_\theta) = L_\theta$, the gradient w.r.t. $H$ is:

$$\nabla_H \mathcal{L} = \nabla_L \mathcal{L} \cdot J_f(H_\theta)$$

If $H \in \mathbb{R}^{C \times D}$, then $J_f$ is rank $D$ at most, and you're stuck. So the question isn't "make a better linear layer" — it's "how do we get richer gradient signal to the backbone?"

Here are the angles I see:

---

## 1. Auxiliary Losses That Bypass the Bottleneck

The most direct approach: don't route all supervision through the LM head. If you attach loss terms at intermediate layers that provide gradient signal without passing through the rank-$D$ projection, the backbone sees a fuller training signal.

**Contrastive auxiliary losses.** Instead of (or in addition to) predicting the full $V$-dimensional softmax, add a contrastive objective in the $D$-dimensional hidden space. Something like: the hidden state for position $t$ should be close to the embedding of the actual next token and far from negatives. This operates entirely in $D$-space — no rank compression. The catch is that it provides a different training signal (embedding similarity vs. probability calibration), so it's complementary rather than a replacement.

**Multi-layer prediction heads.** Attach lightweight prediction heads at multiple layers (not just the final one), each contributing to the loss. This is essentially what deep supervision does in vision architectures. Each head still has the rank-$D$ bottleneck, but different layers see gradient signal from their own head without it being filtered through all subsequent layers *and* the final LM head. CALM (Schuster et al., 2022) and early-exit training do something adjacent. The question is whether the gradient from intermediate heads compensates for the bottleneck at the final head, or just adds noise from undertrained representations.

**Target representation losses.** Predict a compressed representation of the target distribution rather than the full logit vector. For instance, train a frozen projection from the $V$-dimensional target distribution down to some $K$-dimensional space (where $K > D$ but $K \ll V$), and add an auxiliary loss in that space. You're still compressing, but you control the compression to be less lossy than the learned LM head's null-space projection.

## 2. Increasing the Effective Rank of the Projection

The paper shows that the Jacobian of $f$ is rank $D$ at most. But what if we increase the effective dimensionality that the LM head operates over?

**Block-diagonal or mixture heads.** Instead of one $W \in \mathbb{R}^{V \times D}$, use $K$ separate projections $W_1, \ldots, W_K$, each operating on a different $D/K$-dimensional slice of the hidden state, then combine logits. If the combination is additive (like Mixture of Softmaxes from Yang et al. 2018), the paper already notes this doesn't fix the rank issue in the Jacobian. But if the combination is *multiplicative* or involves gating that depends on $H$ itself, the Jacobian structure changes — it's no longer a simple left-multiplication by a fixed matrix but involves higher-order terms. The effective rank of the Jacobian could exceed $D$ if the gating function introduces sufficient nonlinearity.

**Nonlinear LM heads.** A one-hidden-layer MLP as the LM head: $f(h) = W_2 \cdot \text{ReLU}(W_1 h + b_1)$. If the intermediate dimension is $M > D$, the Jacobian at a given $h$ is $W_2 \cdot \text{diag}(\mathbb{1}[W_1 h + b_1 > 0]) \cdot W_1$. The rank is $\min(V, M, D)$, so this still bottlenecks at $D$ since $W_1 \in \mathbb{R}^{M \times D}$. You haven't actually escaped — the input dimensionality is the binding constraint. This is a dead end unless you also widen the representation entering the head.

**Local widening before projection.** What if the final Transformer layer outputs a wider representation specifically for the LM head? Insert a learned up-projection $U \in \mathbb{R}^{D' \times D}$ with $D' > D$ (maybe $D' = 2D$ or $4D$) right before the LM head, so the projection to logits is $W \cdot U \cdot h$ where $W \in \mathbb{R}^{V \times D'}$. The Jacobian w.r.t. $h$ is $W \cdot U$, still rank $D$. No help — the information has to pass through the $D$-dimensional bottleneck at $h$.

This is the fundamental issue: **the bottleneck is at the hidden representation dimensionality, not the LM head's architecture.** Any function of $h \in \mathbb{R}^D$ has a Jacobian of rank $\leq D$ w.r.t. $h$. Full stop.

## 3. Rethinking What $h$ Contains

If the problem is that $h$ is $D$-dimensional and we can't change that, maybe we can change what information the backward pass needs to convey.

**Factored vocabularies / hierarchical softmax.** Decompose the prediction into stages: first predict a cluster, then predict a token within the cluster. Each stage has a smaller effective vocabulary, so the gradient through each stage is lower-rank and loses less relative to $D$. Hierarchical softmax is an old idea, but the paper's gradient analysis gives it new motivation — it's not about computational efficiency, it's about gradient fidelity. With a two-level hierarchy where each level has $\sqrt{V}$ classes, the gradient at each stage is rank $\min(D, \sqrt{V})$ instead of $\min(D, V)$. For $V = 128K$ and $D = 4096$, that's a big difference: $\sqrt{V} \approx 358$, well within $D$.

The tradeoff: hierarchical decomposition introduces conditional independence assumptions that may not hold. And the cluster assignment itself becomes a modeling decision that could introduce its own bottlenecks. But this feels like the most promising structural direction.

**Hash-based or random projection targets.** Instead of predicting $V$ logits, predict $K$ hash-based projections of the one-hot target, where $K$ is chosen to be close to $D$. This is essentially compressed sensing applied to the output layer. The gradient is lower-dimensional by construction, so it passes through the bottleneck with less loss. At inference time, you'd need to recover the full distribution from the compressed representation — feasible if the hash functions are well-chosen, but introduces approximation error.

## 4. Gradient-Level Interventions

Rather than changing the architecture, modify the backward pass directly.

**Gradient projection correction.** After computing $\nabla_H \mathcal{L} = \nabla_L \mathcal{L} \cdot W$, you know the null-space component is lost. What if you estimate it and inject it as a correction? You can't recover the exact lost component, but you could maintain a running estimate of the null-space gradient statistics and use it as a regularizer or momentum term. This is speculative, but the paper's Fig 7 shows the lost component has structure (it's not pure noise in the full gradient — it's the tail coefficients), which suggests it might be partially recoverable from historical gradients.

**Straight-through or projected gradient estimators.** Replace the backward pass through the LM head with an estimator that preserves more of the gradient. For instance, use $W^+$ (the pseudoinverse) instead of $W$ in the backward pass, so the gradient becomes $\nabla_L \mathcal{L} \cdot W^+ \cdot W^\top$. This projects the logit gradient onto the row space of $W$ before applying the transpose — still rank $D$, but potentially better-conditioned. Or use a learned backward projection $B \neq W^\top$ optimized to minimize the angle between the projected and full gradient. This is related to feedback alignment and direct feedback alignment from the deep learning theory literature.

**Per-layer learning rate scaling.** If the gradient reaching the backbone is ~5% of its ideal norm (per Fig 6), a crude fix is to scale the learning rate for backbone parameters up by $\sim 20\times$ relative to the LM head. This doesn't fix the *direction* (which is the real problem — cosine similarity of 0.1-0.2 means the direction is nearly wrong), but it's trivially implementable and might help at the margin. Probably not a real solution given the directional misalignment.

## 5. Changing the Training Objective

**Direct logit optimization with periodic synchronization.** Treat the logits as free parameters for $K$ steps, optimizing them directly (full-rank updates). Every $K$ steps, "distill" the logit matrix back into $H \cdot W^\top$ form via low-rank approximation. The backbone sees the gradient from the distillation step rather than from the loss. This separates "what should the logits be?" from "how should the backbone represent them?" The risk is that the distillation step itself is lossy (same rank constraint), but the logits evolve along the true gradient for $K$ steps before being compressed, which might preserve more information than compressing every step.

**Non-autoregressive or latent-space objectives.** If the bottleneck is specifically about projecting to vocab-sized logits, objectives that operate in latent space (contrastive, diffusion-based, or energy-based) sidestep it entirely. This is a radical departure from standard LM training, but papers like ELECTRA (Clark et al., 2020) showed that alternative training objectives can be more sample-efficient. The gradient bottleneck analysis gives a concrete mechanistic reason why.

## Where I'd Bet

The highest-leverage directions, in my estimation:

| Approach | Feasibility | Expected Impact | Risk |
|---|---|---|---|
| Hierarchical / factored output | High — well-studied | Medium-high — directly reduces effective $V$ | Cluster quality matters |
| Auxiliary contrastive losses | High — bolts on | Medium — complementary signal | May not substitute for calibrated probabilities |
| Straight-through / learned backward | Medium — needs careful tuning | High if direction improves | Could destabilize training |
| Direct logit opt + sync | Low — engineering complexity | High in theory | Distillation step may erase gains |
| Multi-layer prediction heads | High — simple | Medium — helps intermediate layers | Final layer still bottlenecked |

The hierarchical output direction seems most immediately actionable. The paper's own analysis provides the framework: measure the effective rank of the per-stage gradient, show it fits within $D$, and demonstrate improved convergence. That's a clean follow-up experiment.

The learned backward projection is the most intellectually interesting. Feedback alignment showed that networks can learn with approximate backward passes — the question is whether a *better* approximate backward (one that preserves more of the logit gradient's informative components) translates to meaningfully faster convergence. The paper's gradient analysis tooling (null-space projection, cosine similarity measurement) would directly measure whether a proposed backward modification helps.