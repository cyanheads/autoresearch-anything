# Grokking Research Context

## The Phenomenon

Grokking (Power et al. 2022): neural networks trained on small algorithmic
datasets achieve perfect training accuracy quickly, but validation accuracy
doesn't catch up until orders of magnitude more training steps. The model
memorizes first, then suddenly generalizes.

## Key Mechanisms Identified

### Circuit Formation (Nanda et al. 2023)
For modular addition mod p, the generalizing circuit uses discrete Fourier
transforms: embeddings encode numbers as rotations on circles at specific
frequencies. The circuit forms **continuously** during training — measurable
via Fourier analysis of embeddings — but test accuracy jumps **discontinuously**
when the memorization circuit is finally pruned.

Three phases: (1) Memorization, (2) Circuit formation (hidden), (3) Cleanup.

### The LU Mechanism (Liu et al. 2023 — Omnigrok)
Plot test loss vs weight norm: it forms a U shape. Plot train loss: it forms
an L. Grokking = starting at high weight norm and traversing to the U minimum.

**Key insight**: initialization scale controls grokking.
- Large init → grokking (start in memorization regime, weight decay drags to optimum)
- Goldilocks init → immediate generalization (no delay)
- Too small init → no learning at all

### Lazy-to-Rich Transition (Kumar et al. 2024)
Grokking = transition from lazy regime (kernel/memorization) to rich regime
(feature learning/generalization). Three necessary conditions:
1. Initial NTK eigenvectors misaligned with target
2. Dataset small enough for train/test decoupling
3. Network starts in lazy regime

**Grokking without weight decay is possible** — weight decay amplifies but
doesn't cause the transition.

### Softmax Collapse (Lyu et al. 2025)
Without regularization, logits grow unboundedly after memorization. Eventually
float32 absorption errors cause gradient vanishing, permanently trapping the
network. Weight decay prevents this by constraining logit growth.

**Fix**: StableMax (numerically stable softmax) enables grokking without any
regularization. PerpGrad optimizer (gradients orthogonal to weight directions)
eliminates the memorization phase entirely.

### GrokTransfer (Xu et al. 2025)
Transfer embeddings from a weak/small model to a larger one. The larger model
generalizes immediately — no memorization phase. The bottleneck is embedding
quality, not architecture capacity.

## Control Variables

| Variable | Effect on Grokking |
|----------|-------------------|
| Weight decay | Primary accelerator. Higher = faster grokking. Prevents Softmax Collapse. |
| Init scale | Large = grokking. Goldilocks = immediate generalization. Controls starting point on LU landscape. |
| Data fraction | Less data = longer delay, more dramatic grokking. Too little = never groks. |
| Learning rate | Higher LR can accelerate. Too high destabilizes. |
| Model width | Wider = can grok faster (more circuit capacity). |
| Optimizer | AdamW amplifies grokking. Muon accelerates it (Tveit 2025). |
| Float precision | float32 can prevent grokking via Softmax Collapse on small datasets. |

## What's Still Open

1. **No method reliably predicts WHEN grokking will happen** (only whether).
2. **Can you force grokking at a target step?** Nobody has demonstrated this.
3. **Dynamic interventions** (schedules, perturbations, triggered actions) are
   largely unexplored — most work uses static hyperparameters.
4. **Cross-task transfer**: Do interventions that work for addition generalize
   to multiplication, permutation groups, or other tasks?
5. **Eliminating the delay entirely**: PerpGrad and GrokTransfer hint this is
   possible, but through very different mechanisms. Can we find others?

## Our Experiment

We explore dynamic training interventions — things that happen DURING training,
not just static hyperparameter choices:

- Weight decay schedules (ramp, pulse, adaptive)
- Norm targeting (drive toward Goldilocks zone explicitly)
- LR perturbations (spike to escape memorization basin)
- Gradient noise (help escape sharp memorization minimum)
- Spectral regularization (force structured representations)
- Novel interventions discovered during experimentation

We track rich progress measures (Fourier, weight norm, spectral entropy, logit
stats) to understand WHY each intervention works or fails, not just whether
grok_step improves.
