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

    def forward(self, h: torch.Tensor, x: torch.Tensor):
        logits = self.proj(h)
        # Shift for autoregressive loss: predict position t+1 from position t
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

    Key insight from the paper: the gradient bottleneck scales with V/D.
    By reducing effective V to √V per stage, we reduce the bottleneck
    dramatically.
    """

    def __init__(self, vocab_size: int, d_model: int, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        # Choose cluster size ≈ √V
        self.n_clusters = math.isqrt(vocab_size)
        if self.n_clusters * self.n_clusters < vocab_size:
            self.n_clusters += 1
        self.cluster_size = math.ceil(vocab_size / self.n_clusters)
        # Pad vocab to exact multiple
        self.padded_vocab = self.n_clusters * self.cluster_size

        # Stage 1: predict cluster from h
        self.cluster_proj = nn.Linear(d_model, self.n_clusters, bias=False)
        # Stage 2: predict token within cluster from h
        # We use a single projection to cluster_size, conditioned on cluster
        # via a per-cluster bias (lightweight)
        self.token_proj = nn.Linear(d_model, self.cluster_size, bias=False)
        self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters, self.cluster_size))

        nn.init.normal_(self.cluster_proj.weight, std=0.02)
        nn.init.normal_(self.token_proj.weight, std=0.02)

    def forward(self, h: torch.Tensor, x: torch.Tensor):
        B, T, D = h.shape

        # Compute cluster and token-within-cluster targets
        targets = x[:, 1:].reshape(-1)  # (B*(T-1),)
        cluster_targets = targets // self.cluster_size  # which cluster
        token_targets = targets % self.cluster_size  # position within cluster

        h_shift = h[:, :-1].reshape(-1, D)  # (B*(T-1), D)

        # Stage 1: cluster prediction
        cluster_logits = self.cluster_proj(h_shift)  # (N, n_clusters)
        cluster_loss = F.cross_entropy(cluster_logits, cluster_targets)

        # Stage 2: token within cluster
        token_logits_base = self.token_proj(h_shift)  # (N, cluster_size)
        # Add cluster-specific bias
        cluster_bias = self.cluster_bias[cluster_targets]  # (N, cluster_size)
        token_logits = token_logits_base + cluster_bias
        token_loss = F.cross_entropy(token_logits, token_targets)

        loss = cluster_loss + token_loss

        # For metrics: reconstruct full logits (expensive, only for eval)
        # During training we skip this and just return approximate logits
        with torch.no_grad():
            full_logits = self._reconstruct_logits(h, h_shift, cluster_logits, token_logits_base)

        return loss, full_logits

    def _reconstruct_logits(self, h, h_shift, cluster_logits, token_logits_base):
        """Approximate full logits for accuracy measurement."""
        B, T, D = h.shape
        # Use cluster log-probs + token log-probs to get approximate full log-probs
        cluster_log_probs = F.log_softmax(cluster_logits, dim=-1)  # (N, n_clusters)
        # For each cluster, compute token log-probs
        # This is expensive for full reconstruction, so we just use top-1 cluster
        top_cluster = cluster_logits.argmax(dim=-1)  # (N,)
        bias = self.cluster_bias[top_cluster]
        token_log_probs = F.log_softmax(token_logits_base + bias, dim=-1)
        # Reconstruct: the predicted token is top_cluster * cluster_size + argmax(token_logits)
        top_token_in_cluster = token_log_probs.argmax(dim=-1)
        predicted_token = top_cluster * self.cluster_size + top_token_in_cluster
        # Build sparse logits where the predicted token gets high score
        full_logits = torch.full((h_shift.shape[0], self.vocab_size), -100.0, device=h.device)
        predicted_token = predicted_token.clamp(max=self.vocab_size - 1)
        full_logits.scatter_(1, predicted_token.unsqueeze(1), 100.0)
        # Reshape to (B, T, V) — pad the first position
        pad = torch.zeros(B, 1, self.vocab_size, device=h.device)
        return torch.cat([pad, full_logits.reshape(B, T - 1, self.vocab_size)], dim=1)


# ---------------------------------------------------------------------------
# 3. Auxiliary contrastive loss
# ---------------------------------------------------------------------------

@register_head("contrastive_aux")
class ContrastiveAuxHead(nn.Module):
    """Standard linear head + contrastive auxiliary loss in D-space.

    The key idea: the contrastive loss operates entirely in the D-dimensional
    hidden space, bypassing the rank-D bottleneck entirely. The hidden state
    for position t should be close to the embedding of the actual next token
    and far from negative samples.

    This provides a complementary gradient signal that doesn't suffer from
    the V→D compression.
    """

    def __init__(self, vocab_size: int, d_model: int, **kwargs):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        self.contrastive_temperature = 0.1
        self.aux_weight = 1.0  # weight of contrastive loss relative to CE
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, h: torch.Tensor, x: torch.Tensor):
        logits = self.proj(h)

        # Standard CE loss
        ce_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1),
        )

        # Contrastive loss in embedding space
        # h[:, :-1] should be close to embedding of x[:, 1:]
        B, T, D = h.shape
        h_pred = h[:, :-1].reshape(-1, D)  # (N, D)
        targets = x[:, 1:].reshape(-1)  # (N,)

        # Get target embeddings (reuse the LM head weights as embeddings)
        target_emb = self.proj.weight[targets]  # (N, D) — note: proj.weight is (V, D)

        # Normalize for cosine similarity
        h_norm = F.normalize(h_pred, dim=-1)
        t_norm = F.normalize(target_emb, dim=-1)

        # In-batch negatives: sample a subset to keep memory reasonable
        n_negatives = min(1024, h_norm.shape[0])
        neg_indices = torch.randint(0, self.proj.weight.shape[0], (n_negatives,), device=h.device)
        neg_emb = F.normalize(self.proj.weight[neg_indices], dim=-1)  # (K, D)

        # Positive similarity
        pos_sim = (h_norm * t_norm).sum(dim=-1) / self.contrastive_temperature  # (N,)

        # Negative similarities
        neg_sim = h_norm @ neg_emb.T / self.contrastive_temperature  # (N, K)

        # InfoNCE loss
        logits_contrastive = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (N, 1+K)
        contrastive_labels = torch.zeros(h_norm.shape[0], dtype=torch.long, device=h.device)
        contrastive_loss = F.cross_entropy(logits_contrastive, contrastive_labels)

        loss = ce_loss + self.aux_weight * contrastive_loss
        return loss, logits


# ---------------------------------------------------------------------------
# 4. Multi-exit: heads at multiple layers
# ---------------------------------------------------------------------------

@register_head("multi_exit")
class MultiExitHead(nn.Module):
    """Attach prediction heads at multiple backbone layers, sum logits.

    Each intermediate head provides gradient signal to its layer without
    that signal being filtered through all subsequent layers AND the final
    head. While each individual head still has the rank-D bottleneck,
    different layers get direct gradient paths.

    This is related to deep supervision in vision (Lee et al. 2015) and
    CALM (Schuster et al. 2022).
    """

    def __init__(self, vocab_size: int, d_model: int, n_layers: int, backbone=None, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.backbone = backbone

        # Place heads at layers n_layers//3, 2*n_layers//3, and n_layers (final)
        self.exit_layers = [n_layers // 3 - 1, 2 * n_layers // 3 - 1, n_layers - 1]

        # One projection per exit
        self.exit_projs = nn.ModuleList([
            nn.Linear(d_model, vocab_size, bias=False) for _ in self.exit_layers
        ])
        self.exit_norms = nn.ModuleList([
            nn.RMSNorm(d_model) for _ in self.exit_layers
        ])
        # Learnable weights for combining exit logits
        self.exit_weights = nn.Parameter(torch.ones(len(self.exit_layers)))

        for proj in self.exit_projs:
            nn.init.normal_(proj.weight, std=0.02)

    def forward(self, h: torch.Tensor, x: torch.Tensor):
        """h is the final hidden state. We need intermediate states too.

        IMPORTANT: This head hooks into the backbone's forward pass.
        The backbone must be modified to return intermediate states,
        or we re-run through layers here. We take the simpler approach
        of storing intermediates via hooks.
        """
        # h is already the final output; we need to collect intermediates
        # We'll use the stored intermediates from hooks (set up in _setup_hooks)
        if not hasattr(self, '_intermediates') or not self._intermediates:
            # Fallback: just use final state for all exits (first call / no hooks)
            intermediates = [h] * len(self.exit_layers)
        else:
            intermediates = [self._intermediates.get(idx, h) for idx in self.exit_layers]
            self._intermediates = {}  # clear for next forward

        # Compute weighted sum of logits from each exit
        weights = F.softmax(self.exit_weights, dim=0)
        combined_logits = torch.zeros(*h.shape[:2], self.vocab_size, device=h.device)
        exit_losses = []

        for i, (state, proj, norm) in enumerate(zip(intermediates, self.exit_projs, self.exit_norms)):
            exit_logits = proj(norm(state))
            combined_logits = combined_logits + weights[i] * exit_logits
            # Each exit also gets its own CE loss (deep supervision)
            exit_loss = F.cross_entropy(
                exit_logits[:, :-1].reshape(-1, self.vocab_size),
                x[:, 1:].reshape(-1),
            )
            exit_losses.append(exit_loss)

        # Combined loss: CE on combined logits + sum of exit losses
        combined_loss = F.cross_entropy(
            combined_logits[:, :-1].reshape(-1, self.vocab_size),
            x[:, 1:].reshape(-1),
        )

        # Weight: main loss + 0.5 * average of exit losses
        aux_loss = sum(exit_losses) / len(exit_losses)
        loss = combined_loss + 0.5 * aux_loss

        return loss, combined_logits

    def setup_hooks(self, backbone):
        """Register forward hooks on backbone layers to capture intermediates."""
        self._intermediates = {}

        def make_hook(layer_idx):
            def hook(module, input, output):
                self._intermediates[layer_idx] = output
            return hook

        self._hooks = []
        for idx in self.exit_layers[:-1]:  # last exit uses final output
            handle = backbone.layers[idx].register_forward_hook(make_hook(idx))
            self._hooks.append(handle)


# ---------------------------------------------------------------------------
# 5. Factored prediction: decompose tokens into sub-units
# ---------------------------------------------------------------------------

@register_head("factored")
class FactoredHead(nn.Module):
    """Predict token as (factor1, factor2) independently.

    Decompose each token ID into two factors: token = f1 * factor2_size + f2.
    Predict each factor from h independently. Each factor is a small
    classification problem where D >> num_classes, so gradient bottleneck
    is minimal per factor.

    The factorization is arbitrary (not semantic), but the gradient signal
    for each factor is high-quality because num_classes << D.
    """

    def __init__(self, vocab_size: int, d_model: int, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        # Factor sizes
        self.f1_size = math.isqrt(vocab_size)
        if self.f1_size * self.f1_size < vocab_size:
            self.f1_size += 1
        self.f2_size = self.f1_size

        self.proj_f1 = nn.Linear(d_model, self.f1_size, bias=False)
        self.proj_f2 = nn.Linear(d_model, self.f2_size, bias=False)

        nn.init.normal_(self.proj_f1.weight, std=0.02)
        nn.init.normal_(self.proj_f2.weight, std=0.02)

    def forward(self, h: torch.Tensor, x: torch.Tensor):
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

        # Reconstruct full logits for accuracy measurement
        with torch.no_grad():
            f1_pred = f1_logits.argmax(dim=-1)
            f2_pred = f2_logits.argmax(dim=-1)
            predicted = (f1_pred * self.f2_size + f2_pred).clamp(max=self.vocab_size - 1)
            full_logits = torch.full((h_shift.shape[0], self.vocab_size), -100.0, device=h.device)
            full_logits.scatter_(1, predicted.unsqueeze(1), 100.0)
            pad = torch.zeros(B, 1, self.vocab_size, device=h.device)
            full_logits = torch.cat([pad, full_logits.reshape(B, T - 1, self.vocab_size)], dim=1)

        return loss, full_logits
