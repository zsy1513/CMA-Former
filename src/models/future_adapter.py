"""Shared target-day covariate adapter for fair baseline comparison."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class FutureCovariateAdapter(nn.Module):
    """Add the same future-covariate correction head to any neural forecaster.

    The adapter is shared across markets and horizons. It ensures every neural
    baseline receives the same target-day load, renewable and calendar inputs
    without changing the baseline's historical encoder.
    """

    def __init__(self, base_model: nn.Module, n_quantiles: int,
                 d_future_exog: int = 3, d_future_glob: int = 6,
                 hidden: int = 64):
        super().__init__()
        self.base_model = base_model
        self.n_quantiles = n_quantiles
        self.future_head = nn.Sequential(
            nn.Linear(d_future_exog + d_future_glob, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_quantiles),
        )

    @property
    def use_revin(self):
        return bool(getattr(self.base_model, 'use_revin', False))

    def forward(self, x_price, x_exog, x_glob,
                x_future_exog, x_future_glob,
                attn_dump: Optional[Dict[str, torch.Tensor]] = None):
        kwargs = {}
        forward_vars = self.base_model.forward.__code__.co_varnames
        if attn_dump is not None and 'attn_dump' in forward_vars:
            kwargs['attn_dump'] = attn_dump
        if 'x_future_exog' in forward_vars:
            kwargs['x_future_exog'] = x_future_exog
        if 'x_future_glob' in forward_vars:
            kwargs['x_future_glob'] = x_future_glob
        result = self.base_model(x_price, x_exog, x_glob, **kwargs)
        if isinstance(result, tuple) and len(result) == 3:
            prediction, gate, experts = result
        elif isinstance(result, tuple) and len(result) == 2:
            prediction, gate = result
            experts = prediction.unsqueeze(2)
        else:
            prediction, gate, experts = result, None, None

        markets = x_future_exog.shape[1]
        global_features = x_future_glob.unsqueeze(1).expand(
            -1, markets, -1, -1)
        future_features = torch.cat(
            [x_future_exog, global_features], dim=-1)
        correction = self.future_head(future_features)
        # At the D-1 noon issue time the complete D-1 day-ahead price curve
        # is already published. Use it as an operationally valid persistence
        # anchor and train every neural architecture on the remaining residual.
        anchor = x_future_exog[..., 0].unsqueeze(-1)
        prediction = anchor + prediction + correction
        if experts is not None and experts.shape[-1] == correction.shape[-1]:
            experts = experts + anchor.unsqueeze(2) + correction.unsqueeze(2)
        return prediction, gate, experts

    def get_coupling_gates(self):
        if hasattr(self.base_model, 'get_coupling_gates'):
            return self.base_model.get_coupling_gates()
        return None

    def normalize_target(self, target, stats=None):
        return self.base_model.normalize_target(target, stats)

    def denormalize_predictions(self, predictions, stats=None):
        return self.base_model.denormalize_predictions(predictions, stats)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
