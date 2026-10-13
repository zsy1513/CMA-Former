"""
Statistical and shallow-DL baselines.

Implements:
  - NaiveLastDay : tomorrow = today (24h shift); simplest baseline
  - NaiveSeasonal: tomorrow = same weekday last week
  - LEAR         : Lasso Estimated AutoRegressive (Lago et al. 2021)
  - DNN          : Multi-layer perceptron baseline (Lago et al. 2021)

All models share the same I/O contract as CMA-Former:
  forward(x_price, x_exog, x_glob) -> (preds, gate=None, experts=None)
Where preds has shape (B, M, H, Q). For point-prediction baselines we
replicate the median across all Q quantiles.

Author: CMA-Former Project
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class nn:                                    # type: ignore
        class Module:
            pass


# ---------------------------------------------------------------------
# Naive: yesterday repeats
# ---------------------------------------------------------------------

class NaiveLastDay(nn.Module if HAS_TORCH else object):
    """Predicts that the next H hours equal the last H hours of input."""

    def __init__(self, n_quantiles: int = 3, horizon: int = 24,
                 n_markets: int = 4):
        super().__init__()
        self.Q = n_quantiles
        self.H = horizon
        self.M = n_markets

    def forward(self, x_price, x_exog, x_glob):
        # x_price: (B, M, L)  -> take last H as prediction
        last = x_price[:, :, -self.H:]                       # (B, M, H)
        preds = last.unsqueeze(-1).expand(-1, -1, -1, self.Q)
        return preds, None, None


class NaiveSeasonal(nn.Module if HAS_TORCH else object):
    """Predicts next 24h = same hours one week ago."""

    def __init__(self, n_quantiles: int = 3, horizon: int = 24,
                 weekly_lag: int = 168):
        super().__init__()
        self.Q = n_quantiles
        self.H = horizon
        self.lag = weekly_lag

    def forward(self, x_price, x_exog, x_glob):
        # Take from L - lag .. L - lag + H
        L = x_price.shape[-1]
        start = L - self.lag
        end = start + self.H
        last = x_price[:, :, start:end]
        preds = last.unsqueeze(-1).expand(-1, -1, -1, self.Q)
        return preds, None, None


# ---------------------------------------------------------------------
# DNN (Lago 2021): multi-market multi-horizon MLP
# ---------------------------------------------------------------------

class DNN(nn.Module if HAS_TORCH else object):
    """
    Multi-layer perceptron taking flattened past prices + exogenous as input
    and producing M*H*Q outputs.
    """

    def __init__(self,
                 lookback: int,
                 horizon: int,
                 n_markets: int,
                 d_exog: int,
                 d_glob: int,
                 hidden: int = 512,
                 n_layers: int = 3,
                 dropout: float = 0.1,
                 n_quantiles: int = 3):
        super().__init__()
        self.M = n_markets
        self.L = lookback
        self.H = horizon
        self.Q = n_quantiles
        in_dim = (n_markets * lookback +              # past prices
                  n_markets * lookback * d_exog +     # per-market exog
                  lookback * d_glob)                  # global
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, n_markets * horizon * n_quantiles))
        self.net = nn.Sequential(*layers)

    def forward(self, x_price, x_exog, x_glob):
        B = x_price.shape[0]
        feat = torch.cat([
            x_price.reshape(B, -1),
            x_exog.reshape(B, -1),
            x_glob.reshape(B, -1),
        ], dim=-1)
        out = self.net(feat).view(B, self.M, self.H, self.Q)
        return out, None, None


# ---------------------------------------------------------------------
# LEAR (Lasso Estimated AutoRegressive) -- numpy implementation
# ---------------------------------------------------------------------

class LEAR:
    """
    Lasso AR model from Lago et al. (Applied Energy 2021).

    For each market and each forecast horizon h in {0,...,H-1}, fits:
        p_{t+h+1} = sum_lags alpha * p_{t-l} + sum_exog beta * X_{t} + ...

    NOT an nn.Module; lives outside the gradient world. Trained per market
    independently, with one model per (market, horizon) pair.

    Use:
        lear = LEAR(...)
        lear.fit(train_data)
        preds = lear.predict(test_data)   # (N, M, H)
    """

    def __init__(self,
                 lookback: int = 336,
                 horizon: int = 24,
                 n_markets: int = 4,
                 alpha: float = 1e-3,
                 selected_lags: List[int] = None):
        from sklearn.linear_model import Lasso, LassoCV
        self.Lasso = Lasso
        self.LassoCV = LassoCV
        self.L = lookback
        self.H = horizon
        self.M = n_markets
        self.alpha = alpha
        # Lago's recommended set of "informative" lags (h, h+24, h+168)
        self.selected_lags = selected_lags or [0, 1, 2, 23, 24, 25,
                                               47, 48, 49, 167, 168, 169]
        self.models = None                                   # filled on fit

    @staticmethod
    def _build_features(price_arr: np.ndarray,
                        exog_arr: np.ndarray,
                        glob_arr: np.ndarray,
                        L: int, H: int,
                        selected_lags: List[int]) -> np.ndarray:
        """
        price_arr: (M, T)   ;  exog_arr: (M, T, d_x)  ;  glob_arr: (T, d_z)
        Returns a feature matrix (T - L - H + 1, n_features).
        """
        M, T = price_arr.shape
        n_samples = T - L - H + 1
        feats = []
        # Selected lag prices, all markets
        for l in selected_lags:
            for m in range(M):
                col = price_arr[m, L - 1 - l : L - 1 - l + n_samples]
                feats.append(col)
        # Global day-ahead drivers at t (last lookback step)
        if glob_arr.shape[1] > 0:
            for k in range(glob_arr.shape[1]):
                feats.append(glob_arr[L - 1 : L - 1 + n_samples, k])
        # Exog at t for each market
        for m in range(M):
            for k in range(exog_arr.shape[2]):
                feats.append(exog_arr[m, L - 1 : L - 1 + n_samples, k])
        X = np.stack(feats, axis=-1)                         # (n_samples, n_features)
        return X

    def fit(self, price_arr: np.ndarray, exog_arr: np.ndarray,
            glob_arr: np.ndarray):
        """
        price_arr: (M, T) raw prices
        exog_arr : (M, T, d_x) per-market exog
        glob_arr : (T, d_z)   global exog
        """
        L, H, M = self.L, self.H, self.M
        X = self._build_features(price_arr, exog_arr, glob_arr,
                                 L, H, self.selected_lags)
        n = X.shape[0]
        self.models = [[None] * H for _ in range(M)]
        for m in range(M):
            for h in range(H):
                y = price_arr[m, L + h : L + h + n]
                model = self.Lasso(alpha=self.alpha, max_iter=2000)
                model.fit(X, y)
                self.models[m][h] = model

    def predict(self, price_arr: np.ndarray, exog_arr: np.ndarray,
                glob_arr: np.ndarray) -> np.ndarray:
        L, H, M = self.L, self.H, self.M
        X = self._build_features(price_arr, exog_arr, glob_arr,
                                 L, H, self.selected_lags)
        n = X.shape[0]
        preds = np.zeros((n, M, H), dtype=np.float32)
        for m in range(M):
            for h in range(H):
                preds[:, m, h] = self.models[m][h].predict(X)
        return preds


class LEARTorchAdapter(nn.Module if HAS_TORCH else object):
    """
    Wraps a fitted LEAR into the torch model interface (predict-only).
    The model is fit OUTSIDE the trainer (since LEAR uses sklearn Lasso),
    then this adapter is constructed and used at inference time.
    """

    def __init__(self, lear_model: LEAR, n_quantiles: int = 3):
        super().__init__()
        self.lear = lear_model
        self.Q = n_quantiles
        # Need a dummy parameter so DataLoader.to(device) doesn't fail
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x_price, x_exog, x_glob):
        # Move to numpy
        # x_price: (B, M, L) ; LEAR expects (M, T) where T includes L+0 (we
        # treat each batch sample as a (T=L) "timeseries" so LEAR predicts
        # the next H steps). We do this per-sample.
        B, M, L = x_price.shape
        H = self.lear.H
        device = x_price.device
        x_p = x_price.detach().cpu().numpy().astype(np.float32)
        x_e = x_exog.detach().cpu().numpy().astype(np.float32)
        x_g = x_glob.detach().cpu().numpy().astype(np.float32)

        preds = np.zeros((B, M, H), dtype=np.float32)
        for b in range(B):
            # LEAR expects T = L (we pass exactly the lookback)
            X = LEAR._build_features(x_p[b], x_e[b], x_g[b],
                                     L=L, H=H,
                                     selected_lags=self.lear.selected_lags)
            # X has shape (1, n_features) here since T - L - H + 1 = 1 - H
            # (need to handle: predict step uses "last point only")
            for m in range(M):
                for h in range(H):
                    if X.shape[0] > 0:
                        preds[b, m, h] = self.lear.models[m][h].predict(X[-1:])[0]
        out = torch.from_numpy(preds).to(device)
        out = out.unsqueeze(-1).expand(-1, -1, -1, self.Q)
        return out, None, None
