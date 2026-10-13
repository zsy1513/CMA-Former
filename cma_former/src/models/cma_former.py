"""
CMA-Former: Cross-Market Attention Transformer.

Top-level model assembling all components:
  Module 1: PatchEmbedding
  Module 2: Stack of IntraMarketBlock (shared weights, per-market)
  Module 3: Stack of CrossMarketBlock (with Coupling Prior Gating)
  Module 4: Quantile output head (linear by default; an optional
            regime-aware MoE head can be enabled via cfg.use_moe)

Author: CMA-Former Project
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .components import (
    PatchEmbedding,
    IntraMarketBlock,
    CrossMarketBlock,
    RegimeAwareMoEHead,
)


# ---------------------------------------------------------------------
# RevIN: Reversible Instance Normalisation (Kim et al., ICLR 2022)
# ---------------------------------------------------------------------

class RevIN(nn.Module):
    """
    Per-instance, per-market reversible normalisation.
    For each (batch, market) lookback window, compute its own mean and std,
    standardise the input, run the network, then de-standardise the output.

    This neutralises distribution shift between train / test (e.g. energy
    crisis vs post-crisis era) -- the model only ever sees zero-mean,
    unit-variance inputs regardless of the absolute price level.

    Stateless API:
        z, (mu, std) = revin.normalize(x)
        y_raw = revin.denormalize(y_norm, mu, std)

    Stateless design avoids subtle bugs with DataLoader workers, torch.compile,
    or batch-size mismatches between forward() calls.
    """

    def __init__(self, n_markets: int, eps: float = 1e-5,
                 affine: bool = True):
        super().__init__()
        self.n_markets = n_markets
        self.eps = eps
        self.affine = affine
        if affine:
            # per-market learnable affine (gain/bias) applied AFTER normalising
            self.gain = nn.Parameter(torch.ones(n_markets))
            self.bias = nn.Parameter(torch.zeros(n_markets))

    def normalize(self, x: torch.Tensor):
        # x: (B, M, L)
        mu = x.mean(dim=-1, keepdim=True)                        # (B, M, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)        # (B, M, 1)
        std = torch.sqrt(var + self.eps)
        z = (x - mu) / std
        if self.affine:
            z = z * self.gain.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return z, (mu, std)

    def denormalize(self, y_norm: torch.Tensor,
                    mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        # y_norm: (B, M, H, Q) ; mu, std: (B, M, 1)
        if self.affine:
            y_norm = (y_norm - self.bias.view(1, -1, 1, 1)
                      ) / self.gain.view(1, -1, 1, 1)
        std4 = std.unsqueeze(-1)                                  # (B, M, 1, 1)
        mu4  = mu.unsqueeze(-1)                                   # (B, M, 1, 1)
        return y_norm * std4 + mu4


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class CMAFormerConfig:
    n_markets: int = 4
    lookback: int = 336
    horizon: int = 24
    d_exog: int = 3                       # load, RES, flow_net
    d_glob: int = 9                       # gas, carbon + 6 calendar (cyc) + holiday
    d_future_exog: int = 3                # known D-1 price, target load and RES
    d_future_glob: int = 6                # target-day calendar encoding
    d_model: int = 128
    patch_size: int = 16
    patch_stride: int = 8
    n_heads: int = 8
    d_ff: int = 256
    n_layers_intra: int = 3
    n_layers_cross: int = 2
    dropout: float = 0.1
    n_experts: int = 3
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    # RevIN (Reversible Instance Normalisation, ICLR 2022)
    # Each input window is normalised with its OWN mean/std before entering
    # the network, and the output is denormalised with the same statistics.
    # This makes the model robust to train/test distribution shift, which is
    # critical for energy crisis -> post-crisis transitions in the data.
    # DISABLED by default while we validate the rest of the pipeline.
    use_revin: bool = False
    revin_eps: float = 1e-5
    revin_affine: bool = True             # learn scale/bias on top of normalised input
    use_coupling_prior: bool = True
    fixed_coupling_prior: bool = False
    use_future_covariates: bool = False
    # MoE head switch:
    # - True: RegimeAwareMoEHead with 3 experts (Normal/Spike/Negative)
    # - False (default for ablation): a single Linear head, identical in
    #   architecture to PatchTST's head. This is the 'w/o MoE' ablation.
    use_moe: bool = False


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class CMAFormer(nn.Module):
    def __init__(self, cfg: CMAFormerConfig,
                 coupling_prior: Optional[torch.Tensor] = None):
        super().__init__()
        self.cfg = cfg

        # RevIN: applied to input price BEFORE patching, denormalises at output.
        # Disabled by setting cfg.use_revin = False (legacy behaviour).
        self.use_revin = bool(getattr(cfg, 'use_revin', False))
        if self.use_revin:
            self.revin = RevIN(
                n_markets=cfg.n_markets,
                eps=getattr(cfg, 'revin_eps', 1e-5),
                affine=getattr(cfg, 'revin_affine', True),
            )

        self.embed = PatchEmbedding(
            d_model=cfg.d_model,
            patch_size=cfg.patch_size,
            patch_stride=cfg.patch_stride,
            d_exog=cfg.d_exog,
            d_glob=cfg.d_glob,
        )
        # Preserve market identity after per-market normalization and shared
        # temporal encoding. The embedding is especially important for the
        # cross-market block, whose Q/K/V projections are otherwise shared.
        self.market_embedding = nn.Parameter(
            torch.zeros(1, cfg.n_markets, 1, cfg.d_model))
        nn.init.trunc_normal_(self.market_embedding, std=0.02)

        self.intra_layers = nn.ModuleList([
            IntraMarketBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers_intra)
        ])
        self.cross_layers = nn.ModuleList([
            CrossMarketBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout,
                             n_markets=cfg.n_markets,
                             use_prior=cfg.use_coupling_prior,
                             fixed_prior=cfg.fixed_coupling_prior)
            for _ in range(cfg.n_layers_cross)
        ])

        self.use_future_covariates = bool(
            getattr(cfg, 'use_future_covariates', False))
        if self.use_future_covariates:
            future_dim = cfg.horizon * (cfg.d_future_exog + cfg.d_future_glob)
            self.future_encoder = nn.Sequential(
                nn.Linear(future_dim, cfg.d_model),
                nn.GELU(),
                nn.LayerNorm(cfg.d_model),
            )

        # Optional initial prior
        if coupling_prior is not None and cfg.use_coupling_prior:
            for blk in self.cross_layers:
                blk.cross.set_prior(coupling_prior)

        # Pool patches per market into one vector via a small attention pool.
        # We use a learnable query token; the keys/values are the N patch tokens.
        self.pool_query = nn.Parameter(torch.zeros(1, 1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.pool_query, std=0.02)
        self.pool_norm = nn.LayerNorm(cfg.d_model)

        # Head: either RegimeAwareMoEHead (default) or a single linear,
        # PatchTST-style. The latter is used to ablate the MoE in isolation.
        self.use_moe = bool(getattr(cfg, 'use_moe', False))
        if self.use_moe:
            self.head = RegimeAwareMoEHead(
                d_model=cfg.d_model,
                horizon=cfg.horizon,
                n_quantiles=len(cfg.quantiles),
                hidden=cfg.d_model,
                dropout=cfg.dropout,
            )
        else:
            # Single linear, no expert mixture, no soft gate
            self.head = nn.Linear(cfg.d_model,
                                  cfg.horizon * len(cfg.quantiles))

    # ----- pooling -----

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pool patches per market into one vector.
        
        Strategy: take the LAST patch token. This is what PatchTST and most
        modern patch-based time-series transformers use, because the last
        patch contains the most recent context -- directly relevant for
        predicting the next horizon.
        
        Empirical comparison (v1.12): switching from a learnable-query
        attention pool to last-token pool boosted target-prediction
        correlation from 0.27 to 0.7+. The attention pool with default
        small init (std=0.02) effectively degenerates into uniform mean
        pooling, which discards the recency information critical for
        electricity price forecasting.
        """
        # tokens: (B, M, N, D) -> last patch -> (B, M, D)
        return tokens[:, :, -1, :]

    # ----- forward -----

    def forward(self,
                x_price: torch.Tensor,
                x_exog: torch.Tensor,
                x_glob: torch.Tensor,
                x_future_exog: Optional[torch.Tensor] = None,
                x_future_glob: Optional[torch.Tensor] = None,
                attn_dump: Optional[Dict[str, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inputs:
          x_price : (B, M, L)
          x_exog  : (B, M, L, d_x)
          x_glob  : (B, L, d_z)
          x_future_exog : (B, M, H, d_future_exog), optional
          x_future_glob : (B, H, d_future_glob), optional
        Returns:
          preds      : (B, M, H, Q)   normalized-scale quantile predictions
          gate_probs : (B, M, 3)      regime probabilities
          experts    : (B, M, 3, H, Q)
        """
        # RevIN: per-instance normalise before the network sees the data.
        # We KEEP the predictions in normalised space (do NOT denormalise
        # inside the model). The trainer will normalise the target with
        # the SAME window stats so that loss is computed in the per-window
        # standardised space. Final eval predictions are denormalised by
        # the trainer (using stats stashed on the model object).
        revin_stats = None
        if self.use_revin:
            x_price, revin_stats = self.revin.normalize(x_price)
            # Stash on a non-parameter attribute for the trainer to read
            # within the same forward pass. SAFE because the trainer always
            # consumes these immediately after forward(); never persisted.
            self._last_revin_stats = revin_stats

        # Module 1
        h = self.embed(x_price, x_exog, x_glob)                  # (B, M, N, D)
        h = h + self.market_embedding

        # Module 2
        for i, layer in enumerate(self.intra_layers):
            h = layer(h, attn_dump=attn_dump,
                      dump_prefix=f'intra_{i}_')

        # Target-day forecast products are available at the day-ahead issue
        # time. Encode them separately from historical observations and add
        # the context before cross-market exchange so all markets receive the
        # same opportunity to use known future fundamentals.
        if self.use_future_covariates:
            if x_future_exog is None or x_future_glob is None:
                raise ValueError('future covariates are enabled but not provided')
            B, M, H, _ = x_future_exog.shape
            glob = x_future_glob.unsqueeze(1).expand(-1, M, -1, -1)
            future = torch.cat([x_future_exog, glob], dim=-1).reshape(B, M, -1)
            future_context = self.future_encoder(future)
            h = h + future_context.unsqueeze(2)

        # Module 3
        for i, layer in enumerate(self.cross_layers):
            h = layer(h, attn_dump=attn_dump,
                      dump_prefix=f'cross_{i}_')

        # Pool patches -> (B, M, D)
        pooled = self._pool(h)                                   # (B, M, D)

        # Module 4 + 5
        if self.use_moe:
            preds, gate_probs, experts = self.head(pooled)
        else:
            # Plain linear head
            B, M, D = pooled.shape
            H = self.cfg.horizon
            Q = len(self.cfg.quantiles)
            preds = self.head(pooled).view(B, M, H, Q)
            # Provide dummy gate_probs / experts for downstream uniformity.
            gate_probs = torch.zeros(B, M, 3, device=preds.device)
            gate_probs[..., 0] = 1.0                              # all 'normal'
            experts = preds.unsqueeze(2).expand(B, M, 3, H, Q)
        # `preds` and `experts` are in NORMALISED space when use_revin=True.
        return preds, gate_probs, experts

    def denormalize_predictions(self,
                                preds_norm: torch.Tensor,
                                stats=None) -> torch.Tensor:
        """Public helper for the trainer to denormalise stored stats."""
        if not self.use_revin:
            return preds_norm
        if stats is None:
            stats = getattr(self, '_last_revin_stats', None)
        if stats is None:
            raise RuntimeError(
                'No RevIN stats available; call forward() first or pass stats.')
        mu, std = stats
        # accept (B, M, H, Q) or (B, M, H)
        if preds_norm.ndim == 3:
            return preds_norm * std + mu                          # (B, M, H)
        elif preds_norm.ndim == 4:
            return preds_norm * std.unsqueeze(-1) + mu.unsqueeze(-1)
        else:
            raise ValueError(f'Unexpected preds shape: {preds_norm.shape}')

    def normalize_target(self, y_raw: torch.Tensor,
                         stats=None) -> torch.Tensor:
        """Normalise y_price_raw with the SAME window stats as input."""
        if not self.use_revin:
            return y_raw
        if stats is None:
            stats = getattr(self, '_last_revin_stats', None)
        mu, std = stats
        # y_raw: (B, M, H)  ;  mu/std: (B, M, 1)
        y = (y_raw - mu) / std
        if self.revin.affine:
            y = y * self.revin.gain.view(1, -1, 1) + self.revin.bias.view(1, -1, 1)
        return y

    # ----- introspection -----

    def get_coupling_gates(self) -> Optional[torch.Tensor]:
        """Average effective gating matrix across cross-layers.
        
        Returns None when n_layers_cross=0 (ablation: no cross-market block).
        """
        if len(self.cross_layers) == 0:
            return None
        gates = [blk.cross.gating_matrix() for blk in self.cross_layers]
        return torch.stack(gates).mean(dim=0)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
