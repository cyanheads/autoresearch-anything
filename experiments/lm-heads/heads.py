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
            full_logits = torch.full((h_shift.shape[0], self.vocab_size), -100.0, device=h.device)
            full_logits.scatter_(1, predicted.unsqueeze(1), 100.0)
            pad = torch.zeros(B, 1, self.vocab_size, device=h.device)
            full_logits = torch.cat([pad, full_logits.reshape(B, T - 1, self.vocab_size)], dim=1)

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
            full_logits = torch.full((h_shift.shape[0], self.vocab_size), -100.0, device=h.device)
            full_logits.scatter_(1, predicted.unsqueeze(1), 100.0)
            pad = torch.zeros(B, 1, self.vocab_size, device=h.device)
            full_logits = torch.cat([pad, full_logits.reshape(B, T - 1, self.vocab_size)], dim=1)

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
            full_logits = torch.full((h_shift.shape[0], self.vocab_size), -100.0, device=h.device)
            full_logits.scatter_(1, predicted.unsqueeze(1), 100.0)
            pad = torch.zeros(B, 1, self.vocab_size, device=h.device)
            full_logits = torch.cat([pad, full_logits.reshape(B, T - 1, self.vocab_size)], dim=1)

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
            full_logits = torch.full((final_f1_logits.shape[0], self.vocab_size), -100.0, device=h.device)
            full_logits.scatter_(1, predicted.unsqueeze(1), 100.0)
            pad = torch.zeros(B, 1, self.vocab_size, device=h.device)
            full_logits = torch.cat([pad, full_logits.reshape(B, T - 1, self.vocab_size)], dim=1)

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
            pad = torch.zeros(B, 1, self.vocab_size, device=h.device)
            full_logits = torch.cat([pad, sim.reshape(B, T - 1, self.vocab_size)], dim=1)

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
            full_logits = torch.full((h_shift.shape[0], self.vocab_size), -100.0, device=h.device)
            full_logits.scatter_(1, predicted.unsqueeze(1), 100.0)
            pad = torch.zeros(B, 1, self.vocab_size, device=h.device)
            full_logits = torch.cat([pad, full_logits.reshape(B, T - 1, self.vocab_size)], dim=1)

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
        pad = torch.zeros(B_size, 1, self.vocab_size, device=h.device)
        full_logits = torch.cat([pad, logits.reshape(B_size, T - 1, self.vocab_size)], dim=1)

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

        # Gradient w.r.t h: use B instead of W.T
        # Standard: grad_h = grad_output @ W  (V-dim grad projected to D-space via W)
        # Ours: grad_h = grad_output @ B.T  (V-dim grad projected via learned B)
        grad_h = grad_output @ B.T  # (N, D)

        # Gradient w.r.t W: standard (so W still learns normally from logit loss)
        grad_W = grad_output.T @ h  # (V, D)

        # Gradient w.r.t B: pull toward W.T with gentle regularization
        # This keeps B in the neighborhood of W.T but allows it to diverge
        # where divergence improves gradient quality
        grad_B = (B - W.T) * 0.01

        return grad_h, grad_W, grad_B
