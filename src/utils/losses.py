"""
Loss functions for CMA-Former.

Components:
  - QuantileLoss        : pinball loss across multiple quantiles
  - RegimeLoss          : cross-entropy on regime gate, with class weighting
  - SparseGateLoss      : L1 on the cross-market gating matrix
  - CMAFormerLoss       : composite loss orchestrating all of the above

The composite loss accepts the model outputs (preds, gate_probs) and the
gating matrix from cross-market layers, and returns a scalar plus a dict of
component losses for logging.

Author: CMA-Former Project
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Quantile / Pinball loss
# ---------------------------------------------------------------------

class QuantileLoss(nn.Module):
    r"""Pinball loss for probabilistic forecasting.

    For quantile q and prediction \hat{p}_q vs target p:
        L_q = max(q (p - \hat{p}_q),  (q - 1)(p - \hat{p}_q))
    Averaged across quantiles, horizon steps, markets, and batch.
    """

    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.register_buffer('quantiles',
                             torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, preds: torch.Tensor, target: torch.Tensor
                ) -> torch.Tensor:
        # preds  : (B, M, H, Q)
        # target : (B, M, H)   -> expand to (B, M, H, 1) and broadcast
        target = target.unsqueeze(-1)                                # (B, M, H, 1)
        err = target - preds                                          # (B, M, H, Q)
        # Defensive: move buffer to preds.device if mismatched (covers the
        # case where the loss module was not moved to GPU explicitly).
        q = self.quantiles.to(preds.device).view(1, 1, 1, -1)
        loss = torch.maximum(q * err, (q - 1) * err)
        return loss.mean()


# ---------------------------------------------------------------------
# Regime classification auxiliary loss
# ---------------------------------------------------------------------

class RegimeLoss(nn.Module):
    """
    Cross-entropy on regime gate vs the dominant future-window regime label.
    Default: per-batch weights inverse to class frequency, to handle the
    extreme imbalance (negative & spike are rare).
    """

    def __init__(self, n_classes: int = 3,
                 class_weights: Optional[torch.Tensor] = None,
                 dynamic_weights: bool = True):
        super().__init__()
        self.n_classes = n_classes
        self.dynamic_weights = dynamic_weights
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None  # type: ignore

    def forward(self, gate_probs: torch.Tensor,
                regime_labels: torch.Tensor) -> torch.Tensor:
        """
        gate_probs    : (B, M, K)  -- softmax outputs (NOT logits)
        regime_labels : (B, M, H)  -- per-step regime label in {0,1,2}

        Aggregation rule (revised):
          A horizon is labelled 'spike' (resp. 'negative') if AT LEAST a
          fraction `_horizon_threshold` of its steps belong to that class.
          Otherwise the horizon is 'normal'. Negative takes priority over
          spike when both thresholds are met.

          Why: the original 'any-X' rule inflates the rare-class horizon
          prevalence by ~20x (e.g. 1.5%/step spike -> 27% horizon spike),
          producing a noisy aux signal that destabilises the gate. The
          'majority' rule swings to the opposite extreme (rare events
          never reach 50% of any horizon -> 100% normal labels). The
          threshold rule keeps the prevalence proportional and meaningful
          while still providing learning signal for rare regimes.
        """
        B, M, H = regime_labels.shape
        threshold_frac = 0.25                         # 25% of horizon
        thresh_count = max(1, int(H * threshold_frac))

        n_spike = (regime_labels == 1).sum(dim=-1)                    # (B, M)
        n_neg   = (regime_labels == 2).sum(dim=-1)                    # (B, M)

        # Build target on the same device as regime_labels
        target = torch.zeros(B, M, dtype=torch.long,
                             device=regime_labels.device)
        target[n_spike >= thresh_count] = 1
        target[n_neg   >= thresh_count] = 2          # neg priority

        # Gate probs -> log probs (clamp to avoid log(0))
        log_p = torch.log(gate_probs.clamp_min(1e-9))
        # Compute weights
        if self.dynamic_weights:
            counts = torch.bincount(target.view(-1), minlength=self.n_classes
                                    ).float() + 1.0
            inv = counts.sum() / counts
            weights = inv / inv.sum() * self.n_classes               # mean=1
        elif self.class_weights is not None:
            weights = self.class_weights.to(gate_probs.device)
        else:
            weights = torch.ones(self.n_classes, device=gate_probs.device)

        loss = F.nll_loss(log_p.view(B * M, self.n_classes),
                          target.view(B * M),
                          weight=weights)
        return loss


# ---------------------------------------------------------------------
# Sparsity regularizer on coupling gates
# ---------------------------------------------------------------------

class SparseGateLoss(nn.Module):
    """
    L1 penalty on the off-diagonal of the gating matrix to encourage
    interpretable, sparse coupling structure. The diagonal is excluded since
    self-coupling is always present.
    """

    def forward(self, gates: torch.Tensor) -> torch.Tensor:
        # gates: (M, M)  -- effective coupling matrix g_{m -> m'}
        M = gates.shape[0]
        mask = 1 - torch.eye(M, device=gates.device)
        return (gates.abs() * mask).sum() / (M * (M - 1))


# ---------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------

class CMAFormerLoss(nn.Module):
    """
    L_total = L_quantile  +  α(t) * λ1 * L_regime  +  α(t) * λ2 * L_sparse

    where α(t) is a linear warm-up factor that ramps from 0 to 1 over
    `aux_warmup_steps` steps. This lets the main quantile loss settle on a
    sensible base prediction before the auxiliary losses start steering
    the gate / sparsity. Without this warm-up, the regime aux loss can
    push the gate into an early local minimum that hurts overall MAE.

    Set aux_warmup_steps=0 to disable warm-up (legacy behaviour).
    """

    def __init__(self,
                 quantiles: List[float],
                 lambda_regime: float = 0.1,
                 lambda_sparse: float = 0.01,
                 aux_warmup_steps: int = 5000):
        super().__init__()
        self.quantile_loss = QuantileLoss(quantiles)
        self.regime_loss = RegimeLoss()
        self.sparse_loss = SparseGateLoss()
        self.lambda_regime = lambda_regime
        self.lambda_sparse = lambda_sparse
        self.aux_warmup_steps = aux_warmup_steps
        self.register_buffer('_step', torch.zeros(1, dtype=torch.long))

    def _aux_factor(self) -> float:
        if self.aux_warmup_steps <= 0:
            return 1.0
        s = float(self._step.item())
        return min(1.0, s / self.aux_warmup_steps)

    def forward(self,
                preds: torch.Tensor,
                target: torch.Tensor,
                gate_probs: torch.Tensor,
                regime_labels: torch.Tensor,
                gating_matrix: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:

        l_q = self.quantile_loss(preds, target)
        l_r = self.regime_loss(gate_probs, regime_labels)

        if gating_matrix is not None:
            l_s = self.sparse_loss(gating_matrix)
        else:
            l_s = torch.tensor(0.0, device=preds.device)

        alpha = self._aux_factor()
        total = (l_q
                 + alpha * self.lambda_regime * l_r
                 + alpha * self.lambda_sparse * l_s)

        # Step counter (only count training steps; eval forward shouldn't,
        # but the trainer wraps eval in torch.no_grad() so .data add is fine
        # either way as it's tracked by mode).
        if self.training:
            self._step += 1

        return {
            'total':    total,
            'quantile': l_q.detach(),
            'regime':   l_r.detach(),
            'sparse':   l_s.detach(),
        }
