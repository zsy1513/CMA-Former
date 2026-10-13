"""
Evaluation metrics for electricity-price forecasting.

Implements:
  - Point metrics:          MAE, RMSE, SMAPE, rMAE
  - Probabilistic metrics:  Pinball loss, CRPS (empirical from quantiles),
                            quantile coverage
  - Extreme-event metrics:  Spike-F1, Negative-Recall, Spike/Neg precision
  - Statistical tests:      Diebold-Mariano (DM), Giacomini-White (GW)

All functions accept numpy arrays (post-denormalization) so they can be used
both inside the trainer (after .cpu().numpy()) and in standalone analysis.

Author: CMA-Former Project
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


# =====================================================================
# Point metrics
# =====================================================================

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-3) -> float:
    """Symmetric MAPE; handles zero/negative prices via |y|+|yhat| denom."""
    denom = (np.abs(y_true) + np.abs(y_pred)).clip(min=eps)
    return float(100.0 * np.mean(2 * np.abs(y_true - y_pred) / denom))


def rmae(y_true: np.ndarray, y_pred: np.ndarray, y_naive: np.ndarray) -> float:
    """Relative MAE vs a naive (last-day) baseline; <1 means better than naive."""
    e_model = mae(y_true, y_pred)
    e_naive = mae(y_true, y_naive) + 1e-9
    return float(e_model / e_naive)


# =====================================================================
# Probabilistic metrics
# =====================================================================

def pinball_loss(y_true: np.ndarray,
                 y_quantile_preds: np.ndarray,
                 quantiles: List[float]) -> float:
    """
    y_true:           (..., 1) or broadcastable shape
    y_quantile_preds: (..., Q)
    quantiles:        list of Q quantile levels in (0,1)
    Returns mean pinball loss across all dims and quantiles.
    """
    q = np.array(quantiles).reshape((1,) * (y_quantile_preds.ndim - 1) + (-1,))
    if y_true.shape != y_quantile_preds.shape:
        y_true = np.expand_dims(y_true, axis=-1)
    err = y_true - y_quantile_preds
    loss = np.maximum(q * err, (q - 1) * err)
    return float(loss.mean())


def crps_from_quantiles(y_true: np.ndarray,
                        y_quantile_preds: np.ndarray,
                        quantiles: List[float]) -> float:
    """
    Approximate CRPS from a finite number of quantiles via the trapezoidal
    integration of the pinball loss:
        CRPS ≈ 2 * mean_{q in Q} pinball(q)
    With the quantile-derived approximation (Gneiting & Raftery, 2007 §4.2).
    """
    q = np.array(quantiles)
    if y_true.shape != y_quantile_preds.shape:
        y_true_e = np.expand_dims(y_true, axis=-1)
    else:
        y_true_e = y_true
    err = y_true_e - y_quantile_preds
    pinball = np.maximum(q * err, (q - 1) * err)             # (..., Q)
    # 2 * mean over q gives the CRPS approximation
    return float(2 * pinball.mean())


def coverage(y_true: np.ndarray,
             y_low: np.ndarray,
             y_high: np.ndarray) -> float:
    """Fraction of times y_true is in [y_low, y_high]."""
    return float(np.mean((y_true >= y_low) & (y_true <= y_high)))


def interval_width(y_low: np.ndarray, y_high: np.ndarray) -> float:
    """Mean width of a central prediction interval."""
    return float(np.mean(y_high - y_low))


def winkler_score(y_true: np.ndarray,
                  y_low: np.ndarray,
                  y_high: np.ndarray,
                  alpha: float = 0.2) -> float:
    """Winkler interval score for a nominal ``1-alpha`` interval."""
    width = y_high - y_low
    below = (y_true < y_low) * (2.0 / alpha) * (y_low - y_true)
    above = (y_true > y_high) * (2.0 / alpha) * (y_true - y_high)
    return float(np.mean(width + below + above))


def quantile_crossing_rate(y_quantile_preds: np.ndarray) -> float:
    """Share of adjacent quantile pairs that violate monotonic ordering."""
    if y_quantile_preds.shape[-1] < 2:
        return 0.0
    return float(np.mean(np.diff(y_quantile_preds, axis=-1) < 0))


# =====================================================================
# Extreme-event metrics
# =====================================================================

def _binary_metrics(y_true_bin: np.ndarray, y_pred_bin: np.ndarray
                    ) -> Tuple[float, float, float]:
    """Returns (precision, recall, f1)."""
    tp = float(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    fp = float(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = float(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return prec, rec, f1


def spike_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                  spike_threshold: float) -> Dict[str, float]:
    """Spike defined as price > spike_threshold (absolute, e.g. mean + 2 sigma)."""
    yt = (y_true > spike_threshold).astype(np.int32)
    yp = (y_pred > spike_threshold).astype(np.int32)
    p, r, f1 = _binary_metrics(yt, yp)
    return {'spike_precision': p, 'spike_recall': r, 'spike_f1': f1,
            'spike_threshold': float(spike_threshold)}


def negative_metrics(y_true: np.ndarray, y_pred: np.ndarray
                     ) -> Dict[str, float]:
    yt = (y_true < 0).astype(np.int32)
    yp = (y_pred < 0).astype(np.int32)
    p, r, f1 = _binary_metrics(yt, yp)
    return {'neg_precision': p, 'neg_recall': r, 'neg_f1': f1}


# =====================================================================
# Diebold-Mariano test
# =====================================================================

def diebold_mariano(y_true: np.ndarray,
                    y_pred_a: np.ndarray,
                    y_pred_b: np.ndarray,
                    h: int = 1,
                    loss: str = 'mae'
                    ) -> Dict[str, float]:
    """
    DM test of equal predictive accuracy.

    H0: E[d_t] = 0    where d_t = L(e_a_t) - L(e_b_t)
    H1 (two-sided): E[d_t] != 0

    A negative DM stat with small p-value -> model A significantly better.

    Args:
        y_true, y_pred_a, y_pred_b : 1-D numpy arrays of equal length.
        h    : forecast horizon (used for HAC variance lag).
        loss : 'mae' or 'mse'.
    """
    e_a = y_true - y_pred_a
    e_b = y_true - y_pred_b
    if loss == 'mae':
        d = np.abs(e_a) - np.abs(e_b)
    elif loss == 'mse':
        d = e_a ** 2 - e_b ** 2
    else:
        raise ValueError(loss)

    n = len(d)
    d_mean = d.mean()

    # HAC (Newey-West) long-run variance estimate
    lags = h - 1
    gamma_0 = ((d - d_mean) ** 2).mean()
    var_d = gamma_0
    for k in range(1, lags + 1):
        gamma_k = ((d[k:] - d_mean) * (d[:-k] - d_mean)).mean()
        var_d += 2 * (1 - k / (lags + 1)) * gamma_k
    var_d = max(var_d, 1e-12)

    dm_stat = d_mean / np.sqrt(var_d / n)

    # Harvey, Leybourne, Newbold (1997) small-sample correction
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat_hln = dm_stat * correction

    # Two-sided p-value using t-distribution with n-1 df
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat_hln), df=n - 1))

    return {'dm_stat': float(dm_stat),
            'dm_stat_hln': float(dm_stat_hln),
            'p_value': float(p_value),
            'mean_loss_diff': float(d_mean),
            'a_better': bool(d_mean < 0 and p_value < 0.05)}


def daily_loss_panel(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     loss: str = 'mae') -> np.ndarray:
    """Aggregate a day-ahead forecast to one loss per day and market.

    Inputs must have shape ``(delivery_days, markets, horizon)``. This avoids
    treating the 24 hourly errors from one bidding decision as independent
    observations.
    """
    if y_true.shape != y_pred.shape or y_true.ndim != 3:
        raise ValueError('expected equal (days, markets, horizon) arrays')
    error = y_true - y_pred
    if loss == 'mae':
        return np.mean(np.abs(error), axis=-1)
    if loss == 'mse':
        return np.mean(error ** 2, axis=-1)
    raise ValueError(loss)


def diebold_mariano_from_daily_losses(loss_a: np.ndarray,
                                      loss_b: np.ndarray,
                                      max_lag: int = None
                                      ) -> Dict[str, float]:
    """DM test on one loss differential per delivery day.

    Newey--West variance is retained, but the statistical unit is a delivery
    day rather than a flattened set of overlapping hourly forecasts.
    """
    a = np.asarray(loss_a, dtype=float).reshape(-1)
    b = np.asarray(loss_b, dtype=float).reshape(-1)
    if len(a) != len(b) or len(a) < 20:
        raise ValueError('daily loss series must have equal length >= 20')
    valid = np.isfinite(a) & np.isfinite(b)
    d = a[valid] - b[valid]
    n = len(d)
    d_mean = float(d.mean())
    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100.0) ** (2 / 9)))
    max_lag = max(0, min(int(max_lag), n - 2))

    centered = d - d_mean
    long_run_var = float(np.mean(centered ** 2))
    for lag in range(1, max_lag + 1):
        covariance = float(np.mean(centered[lag:] * centered[:-lag]))
        weight = 1.0 - lag / (max_lag + 1)
        long_run_var += 2.0 * weight * covariance
    long_run_var = max(long_run_var, 1e-12)
    statistic = d_mean / np.sqrt(long_run_var / n)
    p_value = 2 * (1 - stats.t.cdf(abs(statistic), df=n - 1))
    return {
        'dm_stat': float(statistic),
        'p_value': float(p_value),
        'mean_loss_diff': d_mean,
        'n_days': int(n),
        'nw_max_lag': int(max_lag),
        'a_better': bool(d_mean < 0 and p_value < 0.05),
    }


def moving_block_bootstrap_loss_difference(loss_a: np.ndarray,
                                           loss_b: np.ndarray,
                                           block_length: int = 7,
                                           n_resamples: int = 2000,
                                           seed: int = 42
                                           ) -> Dict[str, float]:
    """Moving-block bootstrap confidence interval for mean daily loss gain."""
    a = np.asarray(loss_a, dtype=float).reshape(-1)
    b = np.asarray(loss_b, dtype=float).reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    d = (a - b)[valid]
    n = len(d)
    if n < max(20, block_length * 2):
        raise ValueError('insufficient daily observations for block bootstrap')
    block_length = max(1, min(int(block_length), n))
    starts = np.arange(n - block_length + 1)
    n_blocks = int(np.ceil(n / block_length))
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        sample = np.concatenate([d[s:s + block_length] for s in chosen])[:n]
        estimates[i] = sample.mean()
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        'mean_loss_diff': float(d.mean()),
        'ci_2.5': float(lower),
        'ci_97.5': float(upper),
        'block_length': int(block_length),
        'n_resamples': int(n_resamples),
    }


# =====================================================================
# Giacomini-White conditional predictive ability test
# =====================================================================

def giacomini_white(y_true: np.ndarray,
                    y_pred_a: np.ndarray,
                    y_pred_b: np.ndarray,
                    test_fn: np.ndarray = None,
                    loss: str = 'mae'
                    ) -> Dict[str, float]:
    """
    GW test using a constant test function (then equivalent to DM with
    Newey-West variance). The user can pass a richer test_fn (n, k) for
    truly conditional version.
    """
    e_a = y_true - y_pred_a
    e_b = y_true - y_pred_b
    if loss == 'mae':
        d = np.abs(e_a) - np.abs(e_b)
    else:
        d = e_a ** 2 - e_b ** 2
    n = len(d)

    if test_fn is None:
        test_fn = np.ones((n, 1))                              # constant
    z = test_fn * d.reshape(-1, 1)                             # (n, k)

    z_mean = z.mean(axis=0)
    omega = np.cov(z, rowvar=False)
    if omega.ndim == 0:
        omega = omega.reshape(1, 1)
    omega = np.atleast_2d(omega)

    # GW test statistic ~ chi^2(k)
    try:
        omega_inv = np.linalg.pinv(omega)
        gw_stat = n * z_mean @ omega_inv @ z_mean
    except Exception:
        gw_stat = float('nan')

    k = z.shape[1]
    p_value = 1 - stats.chi2.cdf(gw_stat, df=k)
    return {'gw_stat': float(gw_stat),
            'p_value': float(p_value),
            'k': int(k),
            'mean_d': float(d.mean())}


# =====================================================================
# Top-level convenience: full metric panel
# =====================================================================

def compute_all_metrics(y_true: np.ndarray,
                        y_pred_median: np.ndarray,
                        y_pred_quantiles: np.ndarray = None,
                        quantiles: List[float] = None,
                        y_naive: np.ndarray = None,
                        spike_thr: float = None
                        ) -> Dict[str, float]:
    """One-shot computation of the full metric panel for a market."""
    out = {
        'mae':   mae(y_true, y_pred_median),
        'rmse':  rmse(y_true, y_pred_median),
        'smape': smape(y_true, y_pred_median),
    }
    if y_naive is not None:
        out['rmae'] = rmae(y_true, y_pred_median, y_naive)
    if y_pred_quantiles is not None and quantiles is not None:
        out['pinball'] = pinball_loss(y_true, y_pred_quantiles, quantiles)
        out['crps']    = crps_from_quantiles(y_true, y_pred_quantiles, quantiles)
        # 80% nominal coverage if quantiles include 0.1 and 0.9
        if 0.1 in quantiles and 0.9 in quantiles:
            i_lo = quantiles.index(0.1)
            i_hi = quantiles.index(0.9)
            out['coverage_80'] = coverage(y_true,
                                           y_pred_quantiles[..., i_lo],
                                           y_pred_quantiles[..., i_hi])
            out['interval_width_80'] = interval_width(
                y_pred_quantiles[..., i_lo], y_pred_quantiles[..., i_hi])
            out['winkler_80'] = winkler_score(
                y_true, y_pred_quantiles[..., i_lo],
                y_pred_quantiles[..., i_hi], alpha=0.2)
        out['quantile_crossing_rate'] = quantile_crossing_rate(
            y_pred_quantiles)
    if spike_thr is not None:
        out.update(spike_metrics(y_true, y_pred_median, spike_thr))
    out.update(negative_metrics(y_true, y_pred_median))
    return out
