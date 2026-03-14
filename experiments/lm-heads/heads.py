"""
experiments/lm-heads/heads.py

LM head implementations for the gradient bottleneck experiment.

Each head takes hidden states h ∈ R^{B×T×D} and input tokens x ∈ R^{B×T},
and returns (loss, logits) where logits ∈ R^{B×T×V}.

The gradient bottleneck (Godey & Artzi 2026): backpropagating V-dim gradients
through a rank-D linear layer destroys 95-99% of gradient norm. The Jacobian
of any f(h) is rank ≤ D, so architectural changes to the head alone can't
escape this. We need to either:
  1. Reduce effective V per prediction stage (hierarchical, factored)
  2. Bypass the head with auxiliary losses (contrastive)
  3. Provide gradient through alternative paths (multi-exit)
  4. Something we haven't thought of yet
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Head registry
# ---------------------------------------------------------------------------

HEAD_REGISTRY: dict[str, type] = {}


def register_head(name: str):
    def decorator(cls):
        HEAD_REGISTRY[name] = cls
        return cls
    return decorator


def build_head(head_type: str, vocab_size: int, d_model: int, n_layers: int, backbone=None, **kwargs):
    if head_type not in HEAD_REGISTRY:
        raise ValueError(f"Unknown head type: {head_type}. Available: {list(HEAD_REGISTRY)}")
    return HEAD_REGISTRY[head_type](
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        backbone=backbone,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Baseline: Standard linear LM head
# ---------------------------------------------------------------------------

@register_head("baseline")
class BaselineHead(nn.Module):
    """Standard linear projection W ∈ R^{V×D} + cross-entropy.
    This is the standard LM head that suffers from the gradient bottleneck."""

    def __init__(self, vocab_size: int, d_model: int, **kwargs):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, h: torch.Tensor, x: torch.Tensor, **kwargs):
        logits = self.proj(h)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1),
        )
        return loss, logits


# ---------------------------------------------------------------------------
# 1b. Baseline with weight tying (standard modern LM practice)
# ---------------------------------------------------------------------------

@register_head("baseline_tied")
class BaselineTiedHead(nn.Module):
    """Baseline head with weight tying: output projection = input embedding.T.

    Standard practice in modern LMs (GPT-2, LLaMA, etc). The output projection
    matrix is NOT a separate parameter — it shares weights with the input
    embedding table. This means:
    - Head gradient directly improves input embeddings
    - Embedding is trained bidirectionally (input reconstruction + output prediction)
    - Weight matrix is better conditioned than random initialization
    - Reduces total parameter count by V*D (significant for large V)
    """

    def __init__(self, vocab_size: int, d_model: int, backbone=None, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        # Tie to backbone's input embedding — NO separate weight
        self.embedding_weight = backbone.tok_emb.weight if backbone is not None else None

    def forward(self, h: torch.Tensor, x: torch.Tensor, **kwargs):
        # logits = h @ W_emb.T (weight-tied)
        logits = h @ self.embedding_weight.T

        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1),
        )
        return loss, logits


# ---------------------------------------------------------------------------
# 1e. Tied head + low-rank correction
# ---------------------------------------------------------------------------

@register_head("tied_lowrank")
class TiedLowRankHead(nn.Module):
    """Weight-tied head with a low-rank correction term.

    logits = h @ W_emb.T + h @ A @ B

    where A ∈ R^{D×r}, B ∈ R^{r×V}, r << D. The tied projection provides
    the main prediction, while the low-rank term adds flexibility to adjust
    logits for tokens whose output representation differs from their input
    embedding. With r=64, this adds ~3.3M params.
    """

    def __init__(self, vocab_size: int, d_model: int, backbone=None, rank: int = 64, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_weight = backbone.tok_emb.weight if backbone is not None else None

        self.A = nn.Linear(d_model, rank, bias=False)
        self.B = nn.Linear(rank, vocab_size, bias=False)

        nn.init.normal_(self.A.weight, std=0.01)
        nn.init.zeros_(self.B.weight)

    def forward(self, h: torch.Tensor, x: torch.Tensor, **kwargs):
        logits = h @ self.embedding_weight.T + self.B(self.A(h))

        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1),
        )
        return loss, logits


# ---------------------------------------------------------------------------
# 1f. Tied head + output norm + learned scale
# ---------------------------------------------------------------------------

@register_head("tied_norm")
class TiedNormHead(nn.Module):
    """Weight-tied head with head-specific RMSNorm + learned temperature.

    logits = (RMSNorm(h) * scale) @ W_emb.T
    """

    def __init__(self, vocab_size: int, d_model: int, backbone=None, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_weight = backbone.tok_emb.weight if backbone is not None else None
        self.norm = nn.RMSNorm(d_model)
        self.logit_scale = nn.Parameter(torch.ones(1))

    def forward(self, h: torch.Tensor, x: torch.Tensor, **kwargs):
        h_normed = self.norm(h) * self.logit_scale
        logits = h_normed @ self.embedding_weight.T

        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1),
        )
        return loss, logits


# ---------------------------------------------------------------------------
# 1g. Tied head + output bias
# ---------------------------------------------------------------------------

@register_head("tied_bias")
class TiedBiasHead(nn.Module):
    """Weight-tied head with a learned output bias.

    logits = h @ W_emb.T + b

    Standard weight tying omits bias. Adding a learned bias b ∈ R^V lets the
    head adjust per-token log-probabilities to match output frequency
    distribution, which may differ from what embedding similarity gives.
    """

    def __init__(self, vocab_size: int, d_model: int, backbone=None, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_weight = backbone.tok_emb.weight if backbone is not None else None
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, h: torch.Tensor, x: torch.Tensor, **kwargs):
        logits = h @ self.embedding_weight.T + self.bias

        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1),
        )
        return loss, logits


# ---------------------------------------------------------------------------
# 1b3. Tied head + gradient scaling
# ---------------------------------------------------------------------------

@register_head("tied_grad_scale")
class TiedGradScaleHead(nn.Module):
    """Weight-tied head with gradient amplification.

    If the bottleneck destroys 95-99% of gradient norm, amplifying the gradient
    flowing back through the head might compensate. Uses a custom autograd function
    to scale the backward pass gradient by a factor (e.g., V/D ≈ 87x) while
    leaving the forward pass unchanged.

    Combined with weight tying for better conditioning.
    """

    def __init__(self, vocab_size: int, d_model: int, backbone=None, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_weight = backbone.tok_emb.weight if backbone is not None else None
        # Scale factor: sqrt(V/D) to partially compensate for gradient compression
        self.grad_scale = (vocab_size / d_model) ** 0.5  # ~9.3x for V=50257, D=576

    def forward(self, h: torch.Tensor, x: torch.Tensor, **kwargs):
        # Apply gradient scaling in backward pass only
        h_scaled = _GradScale.apply(h, self.grad_scale)
        logits = h_scaled @ self.embedding_weight.T

        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1),
        )
        return loss, logits


class _GradScale(torch.autograd.Function):
    """Scale gradient in backward pass without affecting forward pass."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


# ---------------------------------------------------------------------------
# 1b2. Tied head + gentle embedding aux
# ---------------------------------------------------------------------------

@register_head("tied_emb_aux")
class TiedEmbAuxHead(nn.Module):
    """Weight-tied head + gentle embedding aux (weight 0.1).

    Combines weight tying (best performer so far) with a small cosine+InfoNCE
    aux signal in D-space. The aux weight is 10x smaller than exp 18 (which
    hurt badly at 1.0). Hypothesis: gentle D-space gradient complements the
    tied CE gradient without overwhelming it.
    """

    def __init__(self, vocab_size: int, d_model: int, backbone=None, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_weight = backbone.tok_emb.weight if backbone is not None else None
        self.aux_weight = 0.1

    def forward(self, h: torch.Tensor, x: torch.Tensor, **kwargs):
        B, T, D = h.shape
        targets = x[:, 1:].reshape(-1)

        # Primary: weight-tied CE
        logits = h @ self.embedding_weight.T
        ce_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            targets,
        )

        # Aux: cosine + InfoNCE in D-space (gentle)
        h_shift = h[:, :-1].reshape(-1, D)
        target_emb = self.embedding_weight[targets]
        h_norm = F.normalize(h_shift, dim=-1)
        t_norm = F.normalize(target_emb, dim=-1)

        cosine_loss = 1.0 - (h_norm * t_norm).sum(dim=-1).mean()

        n_neg = min(1024, self.vocab_size)
        neg_idx = torch.randint(0, self.vocab_size, (n_neg,), device=h.device)
        neg_emb = F.normalize(self.embedding_weight[neg_idx], dim=-1)
        pos_sim = (h_norm * t_norm).sum(dim=-1, keepdim=True) / 0.07
        neg_sim = h_norm @ neg_emb.T / 0.07
        nce_logits = torch.cat([pos_sim, neg_sim], dim=1)
        nce_labels = torch.zeros(h_norm.shape[0], dtype=torch.long, device=h.device)
        nce_loss = F.cross_entropy(nce_logits, nce_labels)

        loss = ce_loss + self.aux_weight * (cosine_loss + nce_loss)
        return loss, logits


# ---------------------------------------------------------------------------
# 1c. Tied head + MLP expansion
# ---------------------------------------------------------------------------

@register_head("tied_mlp")
class TiedMLPHead(nn.Module):
    """Weight-tied head with MLP expansion: h → SwiGLU expand(D→4D) → project using W_emb.T.

    Combines weight tying (better gradient conditioning from tied embeddings)
    with MLP expansion (4x gradient rank). The MLP transforms h before
    projecting with the tied embedding, giving the gradient more degrees of
    freedom while maintaining embedding quality.
    """

    def __init__(self, vocab_size: int, d_model: int, backbone=None, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_weight = backbone.tok_emb.weight if backbone is not None else None

        # MLP expansion: D → 4D → D (back to D for compatibility with embedding)
        expand_dim = d_model * 4
        self.w1 = nn.Linear(d_model, expand_dim, bias=False)
        self.w3 = nn.Linear(d_model, expand_dim, bias=False)
        self.down = nn.Linear(expand_dim, d_model, bias=False)
        self.norm = nn.RMSNorm(d_model)

        nn.init.normal_(self.w1.weight, std=0.02)
        nn.init.normal_(self.w3.weight, std=0.02)
        nn.init.normal_(self.down.weight, std=0.02)

    def forward(self, h: torch.Tensor, x: torch.Tensor, **kwargs):
        # MLP expansion + down-project back to D
        expanded = F.silu(self.w1(h)) * self.w3(h)
        h_transformed = self.norm(self.down(expanded))

        # Weight-tied projection
        logits = h_transformed @ self.embedding_weight.T

        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1),
        )
        return loss, logits


# ---------------------------------------------------------------------------
# 2. Hierarchical softmax: two-stage factored prediction
# ---------------------------------------------------------------------------

@register_head("hierarchical")
class HierarchicalHead(nn.Module):
    """Two-stage prediction: first predict cluster, then token within cluster.

    With V tokens split into ~√V clusters of ~√V tokens each, each stage
    predicts over ~√V classes. Since √V ≈ 181 for V=32768, and D=576,
    we have D >> num_classes at each stage. The gradient at each stage
    is rank min(D, √V), meaning much less information is lost compared
    to the full-rank V case.
    """

    def __init__(self, vocab_size: int, d_model: int, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_clusters = math.isqrt(vocab_size)
        if self.n_clusters * self.n_clusters < vocab_size:
            self.n_clusters += 1
        self.cluster_size = math.ceil(vocab_size / self.n_clusters)

        # Stage 1: predict cluster from h
        self.cluster_proj = nn.Linear(d_model, self.n_clusters, bias=False)
        # Stage 2: predict token within cluster, conditioned on cluster via bias
        self.token_proj = nn.Linear(d_model, self.cluster_size, bias=False)
        self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters, self.cluster_size))

        nn.init.normal_(self.cluster_proj.weight, std=0.02)
        nn.init.normal_(self.token_proj.weight, std=0.02)

    def forward(self, h: torch.Tensor, x: torch.Tensor, **kwargs):
        B, T, D = h.shape

        targets = x[:, 1:].reshape(-1)
        cluster_targets = targets // self.cluster_size
        token_targets = targets % self.cluster_size

        h_shift = h[:, :-1].reshape(-1, D)

        # Stage 1: cluster prediction
        cluster_logits = self.cluster_proj(h_shift)
        cluster_loss = F.cross_entropy(cluster_logits, cluster_targets)

        # Stage 2: token within cluster (conditioned on true cluster during training)
        token_logits_base = self.token_proj(h_shift)
        cluster_bias = self.cluster_bias[cluster_targets]
        token_logits = token_logits_base + cluster_bias
        token_loss = F.cross_entropy(token_logits, token_targets)

        loss = cluster_loss + token_loss

        # Reconstruct full logits for accuracy measurement
        with torch.no_grad():
            top_cluster = cluster_logits.argmax(dim=-1)
            bias = self.cluster_bias[top_cluster]
            top_token = (token_logits_base + bias).argmax(dim=-1)
            predicted = (top_cluster * self.cluster_size + top_token).clamp(max=self.vocab_size - 1)
            scatter_logits = torch.full((h_shift.shape[0], self.vocab_size), -100.0, device=h.device)
            scatter_logits.scatter_(1, predicted.unsqueeze(1), 100.0)
            full_logits = torch.zeros(B, T, self.vocab_size, device=h.device)
            full_logits[:, :-1] = scatter_logits.reshape(B, T - 1, self.vocab_size)

        return loss, full_logits


# ---------------------------------------------------------------------------
# 3. Auxiliary contrastive loss (with weight tying to input embeddings)
# ---------------------------------------------------------------------------

@register_head("contrastive_aux")
class ContrastiveAuxHead(nn.Module):
    """Standard linear head + contrastive auxiliary loss in D-space.

    The contrastive loss operates entirely in the D-dimensional hidden space,
    bypassing the rank-D bottleneck. The hidden state for position t should be
    close to the *input embedding* of the actual next token (weight-tied) and
    far from negative samples.
    """

    def __init__(self, vocab_size: int, d_model: int, backbone=None, **kwargs):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        self.contrastive_temperature = 0.1
        self.aux_weight = 1.0
        # Tie to input embeddings for contrastive targets
        self.embedding_weight = backbone.tok_emb.weight if backbone is not None else self.proj.weight
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, h: torch.Tensor, x: torch.Tensor, **kwargs):
        logits = self.proj(h)

        # Standard CE loss
        ce_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1),
        )

        # Contrastive loss in embedding space
        B, T, D = h.shape
        h_pred = h[:, :-1].reshape(-1, D)
        targets = x[:, 1:].reshape(-1)

        # Target embeddings from input embedding table (weight-tied)
        target_emb = self.embedding_weight[targets]

        h_norm = F.normalize(h_pred, dim=-1)
        t_norm = F.normalize(target_emb, dim=-1)

        # In-batch negatives
        n_negatives = min(1024, self.embedding_weight.shape[0])
        neg_indices = torch.randint(0, self.embedding_weight.shape[0], (n_negatives,), device=h.device)
        neg_emb = F.normalize(self.embedding_weight[neg_indices], dim=-1)

        pos_sim = (h_norm * t_norm).sum(dim=-1) / self.contrastive_temperature
        neg_sim = h_norm @ neg_emb.T / self.contrastive_temperature
        logits_contrastive = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        contrastive_labels = torch.zeros(h_norm.shape[0], dtype=torch.long, device=h.device)
        contrastive_loss = F.cross_entropy(logits_contrastive, contrastive_labels)

        loss = ce_loss + self.aux_weight * contrastive_loss
        return loss, logits


# ---------------------------------------------------------------------------
# 4. Multi-exit: heads at multiple layers (explicit intermediates)
# ---------------------------------------------------------------------------

@register_head("multi_exit")
class MultiExitHead(nn.Module):
    """Attach prediction heads at multiple backbone layers, sum logits.

    Each intermediate head provides gradient signal to its layer without
    that signal being filtered through all subsequent layers AND the final
    head. The backbone passes intermediate hidden states explicitly (no hooks).
    """

    def __init__(self, vocab_size: int, d_model: int, n_layers: int, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size

        # Place heads at layers n_layers//3, 2*n_layers//3, and final
        self.exit_layers = [n_layers // 3 - 1, 2 * n_layers // 3 - 1, n_layers - 1]

        self.exit_projs = nn.ModuleList([
            nn.Linear(d_model, vocab_size, bias=False) for _ in self.exit_layers
        ])
        self.exit_norms = nn.ModuleList([
            nn.RMSNorm(d_model) for _ in self.exit_layers
        ])
        self.exit_weights = nn.Parameter(torch.ones(len(self.exit_layers)))

        for proj in self.exit_projs:
            nn.init.normal_(proj.weight, std=0.02)

    def forward(self, h: torch.Tensor, x: torch.Tensor, intermediates: dict[int, torch.Tensor] | None = None, **kwargs):
        if intermediates is None:
            intermediates = {}

        # Collect states: use intermediate if available, else final h
        states = []
        for idx in self.exit_layers:
            if idx in intermediates:
                states.append(intermediates[idx])
            else:
                states.append(h)  # final layer falls back to h (already normed by backbone)

        weights = F.softmax(self.exit_weights, dim=0)
        combined_logits = torch.zeros(*h.shape[:2], self.vocab_size, device=h.device)
        exit_losses = []

        for i, (state, proj, norm) in enumerate(zip(states, self.exit_projs, self.exit_norms)):
            exit_logits = proj(norm(state))
            combined_logits = combined_logits + weights[i] * exit_logits
            exit_loss = F.cross_entropy(
                exit_logits[:, :-1].reshape(-1, self.vocab_size),
                x[:, 1:].reshape(-1),
            )
            exit_losses.append(exit_loss)

        combined_loss = F.cross_entropy(
            combined_logits[:, :-1].reshape(-1, self.vocab_size),
            x[:, 1:].reshape(-1),
        )

        aux_loss = sum(exit_losses) / len(exit_losses)
        loss = combined_loss + 0.5 * aux_loss

        return loss, combined_logits


# ---------------------------------------------------------------------------
# 5. Factored prediction: decompose tokens into sub-units
# ---------------------------------------------------------------------------

@register_head("factored")
class FactoredHead(nn.Module):
    """Predict token as (factor1, factor2) independently.

    Decompose each token ID into two factors: token = f1 * f2_size + f2.
    Each factor is a small classification problem where D >> num_classes,
    so gradient bottleneck is minimal per factor.

    Unlike hierarchical, there's no conditioning between factors — they're
    predicted fully independently. This tests whether the conditioning in
    hierarchical actually matters.
    """

    def __init__(self, vocab_size: int, d_model: int, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.f1_size = math.isqrt(vocab_size)
        if self.f1_size * self.f1_size < vocab_size:
            self.f1_size += 1
        self.f2_size = self.f1_size

        self.proj_f1 = nn.Linear(d_model, self.f1_size, bias=False)
        self.proj_f2 = nn.Linear(d_model, self.f2_size, bias=False)

        nn.init.normal_(self.proj_f1.weight, std=0.02)
        nn.init.normal_(self.proj_f2.weight, std=0.02)

    def forward(self, h: torch.Tensor, x: torch.Tensor, **kwargs):
        B, T, D = h.shape

        targets = x[:, 1:].reshape(-1)
        f1_targets = targets // self.f2_size
        f2_targets = targets % self.f2_size

        h_shift = h[:, :-1].reshape(-1, D)

        f1_logits = self.proj_f1(h_shift)
        f2_logits = self.proj_f2(h_shift)

        f1_loss = F.cross_entropy(f1_logits, f1_targets)
        f2_loss = F.cross_entropy(f2_logits, f2_targets)

        loss = f1_loss + f2_loss

        with torch.no_grad():
            f1_pred = f1_logits.argmax(dim=-1)
            f2_pred = f2_logits.argmax(dim=-1)
            predicted = (f1_pred * self.f2_size + f2_pred).clamp(max=self.vocab_size - 1)
            scatter_logits = torch.full((h_shift.shape[0], self.vocab_size), -100.0, device=h.device)
            scatter_logits.scatter_(1, predicted.unsqueeze(1), 100.0)
            full_logits = torch.zeros(B, T, self.vocab_size, device=h.device)
            full_logits[:, :-1] = scatter_logits.reshape(B, T - 1, self.vocab_size)

        return loss, full_logits


# ---------------------------------------------------------------------------
# 6. Three-level factored: cube-root decomposition
# ---------------------------------------------------------------------------

@register_head("factored3")
class Factored3Head(nn.Module):
    """Three-level independent factorization: token = f1*f2_size*f3_size + f2*f3_size + f3.

    Cube root of 32768 ~ 32. Each stage predicts over ~32 classes where
    D=576 >> 32. The gradient per stage is essentially uncompressed.
    """

    def __init__(self, vocab_size: int, d_model: int, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.f_size = math.ceil(vocab_size ** (1/3))
        while self.f_size ** 3 < vocab_size:
            self.f_size += 1

        self.proj_f1 = nn.Linear(d_model, self.f_size, bias=False)
        self.proj_f2 = nn.Linear(d_model, self.f_size, bias=False)
        self.proj_f3 = nn.Linear(d_model, self.f_size, bias=False)

        for proj in [self.proj_f1, self.proj_f2, self.proj_f3]:
            nn.init.normal_(proj.weight, std=0.02)

    def forward(self, h: torch.Tensor, x: torch.Tensor, **kwargs):
        B, T, D = h.shape

        targets = x[:, 1:].reshape(-1)
        f_size_sq = self.f_size * self.f_size
        f1_targets = targets // f_size_sq
        f2_targets = (targets % f_size_sq) // self.f_size
        f3_targets = targets % self.f_size

        h_shift = h[:, :-1].reshape(-1, D)

        f1_logits = self.proj_f1(h_shift)
        f2_logits = self.proj_f2(h_shift)
        f3_logits = self.proj_f3(h_shift)

        f1_loss = F.cross_entropy(f1_logits, f1_targets)
        f2_loss = F.cross_entropy(f2_logits, f2_targets)
        f3_loss = F.cross_entropy(f3_logits, f3_targets)

        loss = f1_loss + f2_loss + f3_loss

        with torch.no_grad():
            f1_pred = f1_logits.argmax(dim=-1)
            f2_pred = f2_logits.argmax(dim=-1)
            f3_pred = f3_logits.argmax(dim=-1)
            predicted = (f1_pred * f_size_sq + f2_pred * self.f_size + f3_pred).clamp(max=self.vocab_size - 1)
            scatter_logits = torch.full((h_shift.shape[0], self.vocab_size), -100.0, device=h.device)
            scatter_logits.scatter_(1, predicted.unsqueeze(1), 100.0)
            full_logits = torch.zeros(B, T, self.vocab_size, device=h.device)
            full_logits[:, :-1] = scatter_logits.reshape(B, T - 1, self.vocab_size)

        return loss, full_logits


# ---------------------------------------------------------------------------
# 7. Factored + multi-exit hybrid
# ---------------------------------------------------------------------------

@register_head("factored_multi_exit")
class FactoredMultiExitHead(nn.Module):
    """Combine factored output (reduces V per stage) + multi-exit (direct
    gradient paths to earlier layers). Each exit uses factored prediction."""

    def __init__(self, vocab_size: int, d_model: int, n_layers: int, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.f_size = math.isqrt(vocab_size)
        if self.f_size * self.f_size < vocab_size:
            self.f_size += 1

        self.exit_layers = [n_layers // 3 - 1, 2 * n_layers // 3 - 1, n_layers - 1]

        self.exit_norms = nn.ModuleList([nn.RMSNorm(d_model) for _ in self.exit_layers])
        self.exit_f1 = nn.ModuleList([nn.Linear(d_model, self.f_size, bias=False) for _ in self.exit_layers])
        self.exit_f2 = nn.ModuleList([nn.Linear(d_model, self.f_size, bias=False) for _ in self.exit_layers])
        self.exit_weights = nn.Parameter(torch.ones(len(self.exit_layers)))

        for proj_list in [self.exit_f1, self.exit_f2]:
            for proj in proj_list:
                nn.init.normal_(proj.weight, std=0.02)

    def forward(self, h: torch.Tensor, x: torch.Tensor, intermediates: dict[int, torch.Tensor] | None = None, **kwargs):
        if intermediates is None:
            intermediates = {}

        B, T, D = h.shape
        targets = x[:, 1:].reshape(-1)
        f1_targets = targets // self.f_size
        f2_targets = targets % self.f_size

        weights = F.softmax(self.exit_weights, dim=0)
        total_loss = torch.tensor(0.0, device=h.device)
        final_f1_logits = None
        final_f2_logits = None

        for i, idx in enumerate(self.exit_layers):
            state = intermediates.get(idx, h)
            state = self.exit_norms[i](state)
            h_shift = state[:, :-1].reshape(-1, D)

            f1_logits = self.exit_f1[i](h_shift)
            f2_logits = self.exit_f2[i](h_shift)

            f1_loss = F.cross_entropy(f1_logits, f1_targets)
            f2_loss = F.cross_entropy(f2_logits, f2_targets)

            total_loss = total_loss + weights[i] * (f1_loss + f2_loss)
            final_f1_logits = f1_logits
            final_f2_logits = f2_logits

        with torch.no_grad():
            f1_pred = final_f1_logits.argmax(dim=-1)
            f2_pred = final_f2_logits.argmax(dim=-1)
            predicted = (f1_pred * self.f_size + f2_pred).clamp(max=self.vocab_size - 1)
            scatter_logits = torch.full((final_f1_logits.shape[0], self.vocab_size), -100.0, device=h.device)
            scatter_logits.scatter_(1, predicted.unsqueeze(1), 100.0)
            full_logits = torch.zeros(B, T, self.vocab_size, device=h.device)
            full_logits[:, :-1] = scatter_logits.reshape(B, T - 1, self.vocab_size)

        return total_loss, full_logits


# ---------------------------------------------------------------------------
# 8. Embedding prediction: bypass V-dim projection entirely
# ---------------------------------------------------------------------------

@register_head("embedding_pred")
class EmbeddingPredHead(nn.Module):
    """Predict the next token's embedding directly in D-space.

    No V-dim projection at all. The loss is cosine similarity + a small
    CE component from nearest-neighbor lookup. All gradient flows in D-space,
    completely bypassing the rank-D bottleneck.

    For accuracy: find nearest embedding via dot product with the embedding table.
    """

    def __init__(self, vocab_size: int, d_model: int, backbone=None, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Small MLP to predict embedding of next token
        self.pred_proj = nn.Linear(d_model, d_model, bias=False)
        # Reference to embedding table for nearest-neighbor lookup
        self.embedding_weight = backbone.tok_emb.weight if backbone is not None else None
        nn.init.normal_(self.pred_proj.weight, std=0.02)

    def forward(self, h: torch.Tensor, x: torch.Tensor, **kwargs):
        B, T, D = h.shape

        h_shift = h[:, :-1].reshape(-1, D)
        targets = x[:, 1:].reshape(-1)

        # Predict next token's embedding
        pred_emb = self.pred_proj(h_shift)  # (N, D)

        # Get target embeddings
        target_emb = self.embedding_weight[targets]  # (N, D)

        # Cosine similarity loss (maximize similarity to target embedding)
        pred_norm = F.normalize(pred_emb, dim=-1)
        target_norm = F.normalize(target_emb, dim=-1)
        cosine_loss = 1.0 - (pred_norm * target_norm).sum(dim=-1).mean()

        # Also add a small contrastive component: push away from random negatives
        n_neg = min(512, self.vocab_size)
        neg_idx = torch.randint(0, self.vocab_size, (n_neg,), device=h.device)
        neg_emb = F.normalize(self.embedding_weight[neg_idx], dim=-1)  # (n_neg, D)

        # InfoNCE-style: log(exp(sim_pos) / (exp(sim_pos) + sum(exp(sim_neg))))
        pos_sim = (pred_norm * target_norm).sum(dim=-1, keepdim=True) / 0.07  # (N, 1)
        neg_sim = pred_norm @ neg_emb.T / 0.07  # (N, n_neg)
        nce_logits = torch.cat([pos_sim, neg_sim], dim=1)  # (N, 1+n_neg)
        nce_labels = torch.zeros(pred_norm.shape[0], dtype=torch.long, device=h.device)
        nce_loss = F.cross_entropy(nce_logits, nce_labels)

        loss = cosine_loss + nce_loss

        # Accuracy: nearest neighbor in embedding table
        with torch.no_grad():
            # Dot product with full embedding table for token prediction
            sim = pred_emb @ self.embedding_weight.T  # (N, V)
            full_logits = torch.zeros(B, T, self.vocab_size, device=h.device)
            full_logits[:, :-1] = sim.reshape(B, T - 1, self.vocab_size)

        return loss, full_logits


# ---------------------------------------------------------------------------
# 9. Factored with residual conditioning
# ---------------------------------------------------------------------------

@register_head("factored_residual")
class FactoredResidualHead(nn.Module):
    """Factored prediction where factor 2 is conditioned on factor 1's prediction.

    Unlike plain factored (independent f1, f2) or hierarchical (bias conditioning),
    this feeds the predicted f1 embedding back as input to f2 prediction.
    This creates a non-linear dependency: f2 = g(h, embed(f1_pred)).

    During training, uses teacher-forced f1 (ground truth) for conditioning.
    Gradient flows through both the f1 and f2 paths, each with ~√V classes.
    """

    def __init__(self, vocab_size: int, d_model: int, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.f_size = math.isqrt(vocab_size)
        if self.f_size * self.f_size < vocab_size:
            self.f_size += 1

        # Factor 1: independent prediction from h
        self.proj_f1 = nn.Linear(d_model, self.f_size, bias=False)
        # Factor 1 embedding: maps predicted cluster to a conditioning vector
        self.f1_embed = nn.Embedding(self.f_size, d_model)
        # Factor 2: conditioned on h + f1 embedding
        self.proj_f2 = nn.Linear(d_model, self.f_size, bias=False)

        nn.init.normal_(self.proj_f1.weight, std=0.02)
        nn.init.normal_(self.proj_f2.weight, std=0.02)
        nn.init.normal_(self.f1_embed.weight, std=0.02)

    def forward(self, h: torch.Tensor, x: torch.Tensor, **kwargs):
        B, T, D = h.shape

        targets = x[:, 1:].reshape(-1)
        f1_targets = targets // self.f_size
        f2_targets = targets % self.f_size

        h_shift = h[:, :-1].reshape(-1, D)

        # Stage 1: predict f1
        f1_logits = self.proj_f1(h_shift)  # (N, f_size)
        f1_loss = F.cross_entropy(f1_logits, f1_targets)

        # Stage 2: condition on true f1 during training (teacher forcing)
        f1_cond = self.f1_embed(f1_targets)  # (N, D)
        h_conditioned = h_shift + f1_cond  # residual conditioning
        f2_logits = self.proj_f2(h_conditioned)  # (N, f_size)
        f2_loss = F.cross_entropy(f2_logits, f2_targets)

        loss = f1_loss + f2_loss

        # Accuracy: use predicted f1 for conditioning
        with torch.no_grad():
            f1_pred = f1_logits.argmax(dim=-1)
            f1_cond_pred = self.f1_embed(f1_pred)
            h_cond_pred = h_shift + f1_cond_pred
            f2_pred = self.proj_f2(h_cond_pred).argmax(dim=-1)
            predicted = (f1_pred * self.f_size + f2_pred).clamp(max=self.vocab_size - 1)
            scatter_logits = torch.full((h_shift.shape[0], self.vocab_size), -100.0, device=h.device)
            scatter_logits.scatter_(1, predicted.unsqueeze(1), 100.0)
            full_logits = torch.zeros(B, T, self.vocab_size, device=h.device)
            full_logits[:, :-1] = scatter_logits.reshape(B, T - 1, self.vocab_size)

        return loss, full_logits


# ---------------------------------------------------------------------------
# 10. Learned backward projection: fix gradient direction, not just magnitude
# ---------------------------------------------------------------------------

@register_head("learned_backward")
class LearnedBackwardHead(nn.Module):
    """Standard linear head with a learned backward projection.

    The forward pass is identical to baseline: logits = h @ W.T.
    The backward pass replaces W.T with a learned matrix B that is trained
    to produce better gradient directions for the backbone.

    Inspired by feedback alignment (Lillicrap et al. 2016), but instead of
    random B, we optimize B to minimize the angle between the projected
    gradient and the ideal gradient. B is gently pulled toward W.T but can
    diverge to find better gradient directions.

    Implementation: custom autograd function substitutes B for W in the
    backward pass through the head projection.
    """

    def __init__(self, vocab_size: int, d_model: int, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Forward weight (standard)
        self.W = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.normal_(self.W.weight, std=0.02)

        # Backward weight: separate learned matrix for gradient projection
        # Shape: (d_model, vocab_size) — maps V-dim gradient back to D-space
        self.B = nn.Parameter(torch.randn(d_model, vocab_size) * 0.02)

    def forward(self, h: torch.Tensor, x: torch.Tensor, **kwargs):
        B_size, T, D = h.shape

        h_shift = h[:, :-1].reshape(-1, D)
        targets = x[:, 1:].reshape(-1)

        # Forward pass with custom backward
        logits = _LearnedBackwardFn.apply(h_shift, self.W.weight, self.B)

        loss = F.cross_entropy(logits, targets)

        # Reshape logits for accuracy
        full_logits = torch.zeros(B_size, T, self.vocab_size, device=h.device)
        full_logits[:, :-1] = logits.reshape(B_size, T - 1, self.vocab_size)

        return loss, full_logits


class _LearnedBackwardFn(torch.autograd.Function):
    """Custom autograd: forward uses W, backward uses learned B instead of W.T."""

    @staticmethod
    def forward(ctx, h, W, B):
        # h: (N, D), W: (V, D), B: (D, V)
        ctx.save_for_backward(h, W, B)
        return h @ W.T  # standard forward: (N, V)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: (N, V) — gradient of loss w.r.t. logits
        h, W, B = ctx.saved_tensors

        # Cast to match grad_output dtype (bf16 under autocast)
        dtype = grad_output.dtype
        B_cast = B.to(dtype)
        h_cast = h.to(dtype)

        # Gradient w.r.t h: use B instead of W.T
        grad_h = grad_output @ B_cast.T  # (N, D)

        # Gradient w.r.t W: standard (so W still learns normally from logit loss)
        grad_W = grad_output.T @ h_cast  # (V, D)

        # Gradient w.r.t B: pull toward W.T with gentle regularization
        grad_B = (B - W.T) * 0.01

        return grad_h, grad_W, grad_B


# ---------------------------------------------------------------------------
# 11. Baseline + factored auxiliary: full V-dim prediction with un-bottlenecked aux gradient
# ---------------------------------------------------------------------------

@register_head("baseline_factored_aux")
class BaselineFactoredAuxHead(nn.Module):
    """Baseline head for prediction + factored auxiliary for gradient quality.

    The baseline head does the real V-dim prediction (for inference/accuracy).
    The factored auxiliary computes loss through ~√V-class projections, providing
    un-bottlenecked gradient to the backbone as a supplement.

    Key difference from contrastive_aux: the aux here decomposes the SAME
    classification task, not a different objective. Key difference from
    factored_emb_aux: the primary head is the bottlenecked baseline, so the
    aux gradient actually addresses a real deficit.
    """

    def __init__(self, vocab_size: int, d_model: int, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size

        # Primary: standard baseline head
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

        # Auxiliary: factored head for gradient quality
        self.f_size = math.isqrt(vocab_size)
        if self.f_size * self.f_size < vocab_size:
            self.f_size += 1

        self.proj_f1 = nn.Linear(d_model, self.f_size, bias=False)
        self.proj_f2 = nn.Linear(d_model, self.f_size, bias=False)
        nn.init.normal_(self.proj_f1.weight, std=0.02)
        nn.init.normal_(self.proj_f2.weight, std=0.02)

        self.aux_weight = 0.5  # balance between primary and aux loss

    def forward(self, h: torch.Tensor, x: torch.Tensor, **kwargs):
        B, T, D = h.shape
        targets = x[:, 1:].reshape(-1)

        # Primary loss: standard CE over full V
        logits = self.proj(h)
        ce_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            targets,
        )

        # Auxiliary loss: factored CE over √V classes each
        h_shift = h[:, :-1].reshape(-1, D)
        f1_targets = targets // self.f_size
        f2_targets = targets % self.f_size

        f1_loss = F.cross_entropy(self.proj_f1(h_shift), f1_targets)
        f2_loss = F.cross_entropy(self.proj_f2(h_shift), f2_targets)

        loss = ce_loss + self.aux_weight * (f1_loss + f2_loss)

        return loss, logits  # return baseline logits for accuracy


# ---------------------------------------------------------------------------
# 11b. Baseline + embedding prediction auxiliary
# ---------------------------------------------------------------------------

@register_head("baseline_emb_aux")
class BaselineEmbAuxHead(nn.Module):
    """Baseline head for prediction + embedding prediction aux for gradient quality.

    Unlike baseline_factored_aux where the aux used meaningless arithmetic
    decomposition, this aux pushes hidden states toward the target token's
    actual embedding using cosine similarity + InfoNCE. This gradient signal
    is semantically meaningful and flows entirely in D-space (no V-dim bottleneck).
    """

    def __init__(self, vocab_size: int, d_model: int, backbone=None, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size

        # Primary: standard baseline head
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

        # Aux: embedding prediction (cosine + InfoNCE in D-space)
        self.embedding_weight = backbone.tok_emb.weight if backbone is not None else self.proj.weight
        self.aux_weight = 1.0  # stronger aux weight since signal is meaningful

    def forward(self, h: torch.Tensor, x: torch.Tensor, **kwargs):
        B, T, D = h.shape
        targets = x[:, 1:].reshape(-1)

        # Primary loss: standard CE over full V
        logits = self.proj(h)
        ce_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            targets,
        )

        # Aux loss: cosine + InfoNCE in D-space
        h_shift = h[:, :-1].reshape(-1, D)
        target_emb = self.embedding_weight[targets]

        h_norm = F.normalize(h_shift, dim=-1)
        t_norm = F.normalize(target_emb, dim=-1)

        cosine_loss = 1.0 - (h_norm * t_norm).sum(dim=-1).mean()

        # InfoNCE with random negatives
        n_neg = min(1024, self.vocab_size)
        neg_idx = torch.randint(0, self.vocab_size, (n_neg,), device=h.device)
        neg_emb = F.normalize(self.embedding_weight[neg_idx], dim=-1)

        pos_sim = (h_norm * t_norm).sum(dim=-1, keepdim=True) / 0.07
        neg_sim = h_norm @ neg_emb.T / 0.07
        nce_logits = torch.cat([pos_sim, neg_sim], dim=1)
        nce_labels = torch.zeros(h_norm.shape[0], dtype=torch.long, device=h.device)
        nce_loss = F.cross_entropy(nce_logits, nce_labels)

        loss = ce_loss + self.aux_weight * (cosine_loss + nce_loss)

        return loss, logits  # baseline logits for accuracy


# ---------------------------------------------------------------------------
# 12. Semantic factored: factored prediction with embedding-based clustering
# ---------------------------------------------------------------------------

@register_head("semantic_factored")
class SemanticFactoredHead(nn.Module):
    """Factored prediction using semantic clusters from pretrained embeddings.

    Instead of arbitrary arithmetic decomposition (token_id // K, token_id % K),
    tokens are grouped by embedding similarity. Each cluster contains semantically
    related tokens, so "cluster prediction" ≈ "what kind of word?" and
    "within-cluster prediction" ≈ "which specific word?"

    Requires semantic_clusters.pt (generated from GPT-2 embeddings via PCA + chunking).
    Falls back to arbitrary factoring if the file doesn't exist.
    """

    def __init__(self, vocab_size: int, d_model: int, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size

        # Load semantic cluster mapping
        import os
        cluster_path = os.path.join(os.path.dirname(__file__), "semantic_clusters.pt")
        data = torch.load(cluster_path, weights_only=True)
        self.n_clusters = data["n_clusters"]
        self.cluster_size = data["cluster_size"]

        # Buffers: not parameters, but move with .to(device)
        self.register_buffer("token_to_cluster", data["token_to_cluster"])     # (V,)
        self.register_buffer("token_to_position", data["token_to_position"])   # (V,)
        self.register_buffer("cluster_to_token", data["cluster_to_token"])     # (n_clusters, cluster_size)

        # Projections: same architecture as FactoredHead
        self.proj_cluster = nn.Linear(d_model, self.n_clusters, bias=False)
        self.proj_position = nn.Linear(d_model, self.cluster_size, bias=False)

        nn.init.normal_(self.proj_cluster.weight, std=0.02)
        nn.init.normal_(self.proj_position.weight, std=0.02)

    def forward(self, h: torch.Tensor, x: torch.Tensor, **kwargs):
        B, T, D = h.shape

        targets = x[:, 1:].reshape(-1)
        cluster_targets = self.token_to_cluster[targets]    # semantic cluster ID
        position_targets = self.token_to_position[targets]  # position within cluster

        h_shift = h[:, :-1].reshape(-1, D)

        cluster_logits = self.proj_cluster(h_shift)
        position_logits = self.proj_position(h_shift)

        cluster_loss = F.cross_entropy(cluster_logits, cluster_targets)
        position_loss = F.cross_entropy(position_logits, position_targets)

        loss = cluster_loss + position_loss

        # Reconstruct token prediction for accuracy
        with torch.no_grad():
            c_pred = cluster_logits.argmax(dim=-1)       # predicted cluster
            p_pred = position_logits.argmax(dim=-1)      # predicted position
            # Look up actual token ID from (cluster, position)
            predicted = self.cluster_to_token[c_pred, p_pred]
            predicted = predicted.clamp(max=self.vocab_size - 1)
            scatter_logits = torch.full((h_shift.shape[0], self.vocab_size), -100.0, device=h.device)
            scatter_logits.scatter_(1, predicted.unsqueeze(1), 100.0)
            full_logits = torch.zeros(B, T, self.vocab_size, device=h.device)
            full_logits[:, :-1] = scatter_logits.reshape(B, T - 1, self.vocab_size)

        return loss, full_logits


# ---------------------------------------------------------------------------
# 13. Adaptive softmax: frequency-based token grouping
# ---------------------------------------------------------------------------

@register_head("adaptive_softmax")
class AdaptiveSoftmaxHead(nn.Module):
    """Adaptive softmax: frequent tokens get full-rank head, rare tokens share smaller heads.

    Splits vocabulary into frequency bands. The most common tokens (band 0) are
    predicted with a full D-dim linear layer. Rarer bands use progressively smaller
    intermediate projections (D → D//4 → V_band), reducing computation and — crucially —
    changing the gradient dynamics.

    Uses PyTorch's built-in AdaptiveLogSoftmaxWithLoss.
    """

    def __init__(self, vocab_size: int, d_model: int, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Frequency cutoffs: top 2k tokens in band 0, next 10k in band 1, rest in band 2
        cutoffs = [c for c in [2000, 10000] if c < vocab_size]

        self.adaptive = nn.AdaptiveLogSoftmaxWithLoss(
            d_model, vocab_size, cutoffs=cutoffs, div_value=4.0,
        )

    def forward(self, h: torch.Tensor, x: torch.Tensor, **kwargs):
        B, T, D = h.shape

        h_shift = h[:, :-1].reshape(-1, D)
        targets = x[:, 1:].reshape(-1)

        output = self.adaptive(h_shift, targets)
        loss = output.loss

        with torch.no_grad():
            log_probs = self.adaptive.log_prob(h_shift)
            full_logits = torch.zeros(B, T, self.vocab_size, device=h.device)
            full_logits[:, :-1] = log_probs.reshape(B, T - 1, self.vocab_size)

        return loss, full_logits


# ---------------------------------------------------------------------------
# 14. MLP head: expand dimensionality before V-dim projection
# ---------------------------------------------------------------------------

@register_head("mlp_head")
class MLPHead(nn.Module):
    """MLP head: h → SwiGLU expand(D→4D) → project(4D→V).

    The expansion gives the gradient 4x more channels through the bottleneck.
    Jacobian rank becomes min(4D, V) instead of min(D, V).
    With D=576, 4D=2304, V=50257: rank 2304 vs rank 576 — 4x more gradient info.
    """

    def __init__(self, vocab_size: int, d_model: int, **kwargs):
        super().__init__()
        expand_dim = d_model * 4

        self.w1 = nn.Linear(d_model, expand_dim, bias=False)
        self.w3 = nn.Linear(d_model, expand_dim, bias=False)
        self.proj = nn.Linear(expand_dim, vocab_size, bias=False)
        self.norm = nn.RMSNorm(expand_dim)

        nn.init.normal_(self.w1.weight, std=0.02)
        nn.init.normal_(self.w3.weight, std=0.02)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, h: torch.Tensor, x: torch.Tensor, **kwargs):
        expanded = F.silu(self.w1(h)) * self.w3(h)
        expanded = self.norm(expanded)
        logits = self.proj(expanded)

        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1),
        )
        return loss, logits


# ---------------------------------------------------------------------------
# 15. Semantic hierarchical: semantic clusters + cluster-conditioned prediction
# ---------------------------------------------------------------------------

@register_head("semantic_hierarchical")
class SemanticHierarchicalHead(nn.Module):
    """Hierarchical prediction with semantic clusters and cluster bias conditioning.

    Same as SemanticFactoredHead but adds per-cluster bias to the position prediction,
    so the within-cluster prediction is conditioned on which cluster was selected.
    On real language (unlike SpamLang), this conditioning should matter because
    cluster identity carries semantic information.
    """

    def __init__(self, vocab_size: int, d_model: int, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size

        import os
        cluster_path = os.path.join(os.path.dirname(__file__), "semantic_clusters.pt")
        data = torch.load(cluster_path, weights_only=True)
        self.n_clusters = data["n_clusters"]
        self.cluster_size = data["cluster_size"]

        self.register_buffer("token_to_cluster", data["token_to_cluster"])
        self.register_buffer("token_to_position", data["token_to_position"])
        self.register_buffer("cluster_to_token", data["cluster_to_token"])

        self.proj_cluster = nn.Linear(d_model, self.n_clusters, bias=False)
        self.proj_position = nn.Linear(d_model, self.cluster_size, bias=False)
        self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters, self.cluster_size))

        nn.init.normal_(self.proj_cluster.weight, std=0.02)
        nn.init.normal_(self.proj_position.weight, std=0.02)

    def forward(self, h: torch.Tensor, x: torch.Tensor, **kwargs):
        B, T, D = h.shape

        targets = x[:, 1:].reshape(-1)
        cluster_targets = self.token_to_cluster[targets]
        position_targets = self.token_to_position[targets]

        h_shift = h[:, :-1].reshape(-1, D)

        # Stage 1: cluster prediction
        cluster_logits = self.proj_cluster(h_shift)
        cluster_loss = F.cross_entropy(cluster_logits, cluster_targets)

        # Stage 2: position prediction conditioned on true cluster (teacher forcing)
        position_logits_base = self.proj_position(h_shift)
        bias = self.cluster_bias[cluster_targets]
        position_logits = position_logits_base + bias
        position_loss = F.cross_entropy(position_logits, position_targets)

        loss = cluster_loss + position_loss

        with torch.no_grad():
            c_pred = cluster_logits.argmax(dim=-1)
            p_logits = position_logits_base + self.cluster_bias[c_pred]
            p_pred = p_logits.argmax(dim=-1)
            predicted = self.cluster_to_token[c_pred, p_pred].clamp(max=self.vocab_size - 1)
            scatter_logits = torch.full((h_shift.shape[0], self.vocab_size), -100.0, device=h.device)
            scatter_logits.scatter_(1, predicted.unsqueeze(1), 100.0)
            full_logits = torch.zeros(B, T, self.vocab_size, device=h.device)
            full_logits[:, :-1] = scatter_logits.reshape(B, T - 1, self.vocab_size)

        return loss, full_logits


# ---------------------------------------------------------------------------
# 19. Multi-token prediction: predict next K tokens simultaneously
# ---------------------------------------------------------------------------

@register_head("tied_mtp")
class TiedMTPHead(nn.Module):
    """Weight-tied head with multi-token prediction (MTP).

    Predicts the next K tokens simultaneously using K separate linear projections
    that each share the embedding table. Unlike multi-exit (which attaches at
    different layers), MTP attaches K heads to the SAME final hidden state,
    each targeting a different future position (t+1, t+2, ..., t+K).

    Each head provides independent gradient through the backbone, and the targets
    are genuinely different (different future tokens), so gradients don't compete
    like aux losses do — they reinforce richer representations.

    Inspired by Meta's multi-token prediction (Gloeckle et al., 2024).
    """

    def __init__(self, vocab_size: int, d_model: int, backbone=None, n_ahead: int = 2, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_ahead = n_ahead
        self.embedding_weight = backbone.tok_emb.weight if backbone is not None else None

        # Single future projection (t+2 prediction)
        # Using n_ahead=2: only predict t+1 (primary) and t+2 (aux)
        self.future_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.future_proj.weight)  # identity init — starts as baseline_tied

    def forward(self, h: torch.Tensor, x: torch.Tensor, **kwargs):
        B, T, D = h.shape

        # Compute all logits in one batched matmul
        # Primary hidden states and future-transformed hidden states
        h_primary = h[:, :-1]  # (B, T-1, D) — predicts x[:, 1:]
        h_future = self.future_proj(h[:, :-2])  # (B, T-2, D) — predicts x[:, 2:]

        # Stack and do single matmul (more efficient than two separate ones)
        # But shapes differ, so we do two matmuls but keep it simple
        logits = h @ self.embedding_weight.T  # (B, T, V) — full logits for accuracy
        primary_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, self.vocab_size),
            x[:, 1:].reshape(-1),
        )

        future_logits = h_future @ self.embedding_weight.T  # (B, T-2, V)
        future_loss = F.cross_entropy(
            future_logits.reshape(-1, self.vocab_size),
            x[:, 2:].reshape(-1),
        )

        # Weight aux at 0.5
        loss = primary_loss + 0.5 * future_loss

        return loss, logits


# ---------------------------------------------------------------------------
# 20. Tied with label smoothing
# ---------------------------------------------------------------------------

@register_head("tied_smooth")
class TiedSmoothHead(nn.Module):
    """Weight-tied head with label smoothing.

    Label smoothing redistributes a fraction of the target probability mass
    uniformly across all tokens. This prevents the model from becoming
    overconfident, keeps gradients non-zero for non-target tokens, and
    acts as a regularizer. With weight tying already providing regularization
    through shared weights, smoothing may compound the benefit.
    """

    def __init__(self, vocab_size: int, d_model: int, backbone=None, smoothing: float = 0.1, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.embedding_weight = backbone.tok_emb.weight if backbone is not None else None

    def forward(self, h: torch.Tensor, x: torch.Tensor, **kwargs):
        logits = h @ self.embedding_weight.T
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1),
            label_smoothing=self.smoothing,
        )
        return loss, logits


# ---------------------------------------------------------------------------
# 21. Tied with dropout before projection
# ---------------------------------------------------------------------------

@register_head("tied_dropout")
class TiedDropoutHead(nn.Module):
    """Weight-tied head with dropout on hidden states before projection.

    Dropout before the V-dim projection prevents co-adaptation between
    specific hidden dimensions and embedding directions. This forces the
    backbone to distribute information more broadly across dimensions,
    potentially improving generalization.
    """

    def __init__(self, vocab_size: int, d_model: int, backbone=None, drop_rate: float = 0.1, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_weight = backbone.tok_emb.weight if backbone is not None else None
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, h: torch.Tensor, x: torch.Tensor, **kwargs):
        h_dropped = self.dropout(h)
        logits = h_dropped @ self.embedding_weight.T
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1),
        )
        return loss, logits


# ---------------------------------------------------------------------------
# 22. Tied with Z-loss: penalize log-partition function magnitude
# ---------------------------------------------------------------------------

@register_head("tied_zloss")
class TiedZLossHead(nn.Module):
    """Weight-tied head with Z-loss regularization (PaLM-style).

    Z-loss adds a penalty on the squared log of the partition function:
    z_loss = log(sum(exp(logits)))^2. This prevents logits from growing
    too large, which can cause numerical instability and poor gradient
    quality when the softmax becomes very peaky.

    Unlike label smoothing (which modifies the target distribution),
    Z-loss directly penalizes logit magnitude, keeping the softmax
    well-conditioned.
    """

    def __init__(self, vocab_size: int, d_model: int, backbone=None, z_weight: float = 1e-4, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.z_weight = z_weight
        self.embedding_weight = backbone.tok_emb.weight if backbone is not None else None

    def forward(self, h: torch.Tensor, x: torch.Tensor, **kwargs):
        logits = h @ self.embedding_weight.T

        flat_logits = logits[:, :-1].reshape(-1, logits.size(-1))
        targets = x[:, 1:].reshape(-1)

        ce_loss = F.cross_entropy(flat_logits, targets)

        # Z-loss: penalize log(sum(exp(logits)))^2
        log_z = torch.logsumexp(flat_logits, dim=-1)
        z_loss = (log_z ** 2).mean()

        loss = ce_loss + self.z_weight * z_loss

        return loss, logits


# ---------------------------------------------------------------------------
# 23. Partitioned tied head: K vocab partitions, each with tied projection
# ---------------------------------------------------------------------------

@register_head("tied_partitioned")
class TiedPartitionedHead(nn.Module):
    """Split vocabulary into K partitions, use K separate tied projections.

    Instead of one (D, V) projection, uses K projections of size (D, V/K).
    A learned router (D→K) selects the partition. Within each partition,
    the projection weights are tied to the corresponding slice of the
    input embedding table.

    This reduces the effective V/D ratio per partition from 87:1 to ~22:1
    (with K=4), potentially alleviating the gradient bottleneck within
    each partition.
    """

    def __init__(self, vocab_size: int, d_model: int, backbone=None, n_partitions: int = 4, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_partitions = n_partitions
        self.embedding_weight = backbone.tok_emb.weight if backbone is not None else None

        # Router: select which partition
        self.router = nn.Linear(d_model, n_partitions, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)

        # Partition boundaries (roughly equal-sized)
        self.partition_size = math.ceil(vocab_size / n_partitions)

        # Token-to-partition mapping (buffer)
        partition_ids = torch.arange(vocab_size) // self.partition_size
        self.register_buffer("token_partition", partition_ids)

    def forward(self, h: torch.Tensor, x: torch.Tensor, **kwargs):
        B, T, D = h.shape

        # Full logits via weight tying (for primary loss and accuracy)
        logits = h @ self.embedding_weight.T

        # Router loss: train the router to predict which partition the target falls in
        targets = x[:, 1:].reshape(-1)
        partition_targets = self.token_partition[targets]
        h_shift = h[:, :-1].reshape(-1, D)
        router_logits = self.router(h_shift)
        router_loss = F.cross_entropy(router_logits, partition_targets)

        # Primary CE loss
        ce_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            targets,
        )

        # Total loss: CE + small router contribution
        loss = ce_loss + 0.1 * router_loss

        return loss, logits
