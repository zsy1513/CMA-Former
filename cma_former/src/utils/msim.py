"""
Market Spillover Intensity Matrix (MSIM) analysis.

Implements the interpretability machinery that turns CMA-Former's
cross-market attention weights into human-readable spillover diagnostics:

  - aggregate_msim         : average attention into a M x M matrix
  - time_varying_msim      : rolling-window MSIM over the test set
  - regime_conditional_msim: condition on the model's regime gate
  - lagged_spillover       : per-patch-position decomposition
  - directionality_index   : net directional spillover (Diebold-Yilmaz style)
  - msim_recovery          : compare estimated vs ground-truth (synthetic only)

Inputs are the attention dumps captured by the model:
    attn_dump['cross_{l}_cross']  : (N_samples, N_patches, H_heads, M, M)
    attn_dump['cross_{l}_gate']   : (M, M)
    gate_probs                    : (N_samples, M, K)  for regime conditioning

Author: CMA-Former Project
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger('cma_former.msim')


# =====================================================================
# Aggregation primitives
# =====================================================================

def aggregate_msim(attn: np.ndarray,
                   normalize_rows: bool = True) -> np.ndarray:
    """
    Average attention weights across samples / patches / heads to produce a
    M x M matrix where row m is "the share of attention that market m's
    queries place on each market m'".

    attn shape: (N_samples, N_patches, n_heads, M, M)
    Returns: (M, M)

    The (m, m') entry can be read as: "how much does market m's price look
    to market m' to forecast itself?". If normalize_rows is True (default),
    rows sum to 1 (interpretable as probability mass).
    """
    if attn.ndim != 5:
        raise ValueError(f'expected 5D attn, got {attn.shape}')
    msim = attn.mean(axis=(0, 1, 2))                                # (M, M)
    if normalize_rows:
        msim = msim / (msim.sum(axis=-1, keepdims=True) + 1e-12)
    return msim


def aggregate_multilayer(attn_dump: Dict[str, np.ndarray],
                         normalize_rows: bool = True
                         ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Average MSIM across all cross-market layers, plus return per-layer maps.
    """
    per_layer = {}
    cross_keys = sorted(k for k in attn_dump
                        if k.startswith('cross_') and k.endswith('cross'))
    if not cross_keys:
        raise KeyError(f'No cross-market attention found. Keys: {list(attn_dump)}')
    mats = []
    for k in cross_keys:
        m = aggregate_msim(attn_dump[k], normalize_rows=normalize_rows)
        per_layer[k] = m
        mats.append(m)
    avg = np.mean(np.stack(mats, axis=0), axis=0)
    if normalize_rows:
        avg = avg / (avg.sum(axis=-1, keepdims=True) + 1e-12)
    return avg, per_layer


# =====================================================================
# Time-varying MSIM
# =====================================================================

def time_varying_msim(attn: np.ndarray,
                      timestamps: np.ndarray,
                      window: str = '30D',
                      normalize_rows: bool = True
                      ) -> Dict[pd.Timestamp, np.ndarray]:
    """
    Compute MSIM in a rolling window over the test set.

    attn: (N_samples, N_patches, H, M, M)
    timestamps: (N_samples,) datetime64 array (UTC) of the END of each
                input window (i.e., the prediction time anchor).
    window: pandas-style offset, e.g. '30D'.

    Returns: {center_timestamp -> (M, M) matrix}
    """
    df_idx = pd.to_datetime(timestamps, utc=True)
    n = len(df_idx)
    if attn.shape[0] != n:
        raise ValueError(f'attn batch ({attn.shape[0]}) != timestamps ({n})')

    # Aggregate first across patches+heads to (N, M, M) for fast rolling
    per_sample = attn.mean(axis=(1, 2))                             # (N, M, M)

    # Use pandas rolling on a flattened series, market by market
    M = per_sample.shape[-1]
    ts = pd.Series(np.arange(n), index=df_idx).sort_index()
    out: Dict[pd.Timestamp, np.ndarray] = {}

    # Step the centre by 7 days for a manageable number of points
    starts = pd.date_range(df_idx.min(), df_idx.max() - pd.Timedelta(window),
                           freq='7D')
    for start in starts:
        end = start + pd.Timedelta(window)
        mask = (df_idx >= start) & (df_idx < end)
        if mask.sum() < 24:                                         # need enough samples
            continue
        chunk = per_sample[mask]                                    # (k, M, M)
        m = chunk.mean(axis=0)
        if normalize_rows:
            m = m / (m.sum(axis=-1, keepdims=True) + 1e-12)
        out[start + (pd.Timedelta(window) / 2)] = m
    return out


# =====================================================================
# Regime-conditional MSIM
# =====================================================================

def regime_conditional_msim(attn: np.ndarray,
                            gate_probs: np.ndarray,
                            regime_names: List[str] = None
                            ) -> Dict[str, np.ndarray]:
    """
    Average attention conditional on the model's regime gate.

    attn:        (N, N_patches, H, M, M)
    gate_probs:  (N, M, K)  -- per-sample, per-market regime probabilities
    Returns: {regime_name -> (M, M) matrix}, where assignment is by argmax of
             the average gate across markets.
    """
    if regime_names is None:
        regime_names = ['normal', 'spike', 'negative']
    K = gate_probs.shape[-1]
    if K != len(regime_names):
        raise ValueError(f'gate has {K} regimes; names has {len(regime_names)}')

    # Average gate across markets for a single per-sample regime label
    avg_gate = gate_probs.mean(axis=1)                              # (N, K)
    sample_regime = avg_gate.argmax(axis=-1)                        # (N,)

    # Per-sample attention summary
    per_sample = attn.mean(axis=(1, 2))                             # (N, M, M)

    out = {}
    for k, name in enumerate(regime_names):
        mask = (sample_regime == k)
        if mask.sum() < 5:
            logger.info(f'  Regime "{name}" has only {mask.sum()} samples; skipping.')
            continue
        m = per_sample[mask].mean(axis=0)
        m = m / (m.sum(axis=-1, keepdims=True) + 1e-12)
        out[name] = m
        logger.info(f'  Regime "{name}": {int(mask.sum())} samples')
    return out


# =====================================================================
# Lagged spillover (per-patch decomposition)
# =====================================================================

def lagged_spillover(attn: np.ndarray) -> np.ndarray:
    """
    Decompose MSIM by patch position to reveal *when* in the past each
    market's information matters.

    Returns: (N_patches, M, M) -- per-patch averaged spillover.

    Patch index 0 is the OLDEST, and the last index is the most recent
    (because PatchTST builds patches with increasing time stride).
    """
    if attn.ndim != 5:
        raise ValueError('attn must be 5-D')
    per_patch = attn.mean(axis=(0, 2))                              # (N_p, M, M)
    # Row-normalize per patch
    per_patch = per_patch / (per_patch.sum(axis=-1, keepdims=True) + 1e-12)
    return per_patch


# =====================================================================
# Diebold-Yilmaz style directionality
# =====================================================================

def directionality_index(msim: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Diebold-Yilmaz spillover indices computed from MSIM (which is row-stochastic).

    For market m:
      TO-others    : sum_{m' != m} MSIM[m', m]   (how much m exports)
      FROM-others  : sum_{m' != m} MSIM[m, m']   (how much m imports)
      NET          : TO - FROM  (positive => net spillover transmitter)

    Total spillover index: average of off-diagonal mass.
    """
    M = msim.shape[0]
    eye = np.eye(M, dtype=bool)
    off = msim.copy()
    off[eye] = 0
    to_others = off.sum(axis=0)                                     # column sums
    from_others = off.sum(axis=1)                                   # row sums
    net = to_others - from_others
    total = off.sum() / M                                           # normalised
    return {
        'to_others':   to_others,
        'from_others': from_others,
        'net':         net,
        'total':       float(total),
    }


# =====================================================================
# Recovery quality (synthetic only)
# =====================================================================

def msim_recovery(estimated: np.ndarray, ground_truth: np.ndarray,
                  exclude_diag: bool = True
                  ) -> Dict[str, float]:
    """
    Compare estimated MSIM against the ground-truth coupling matrix
    (only meaningful for synthetic data where GT is known).
    Returns Pearson r, Spearman rho, and Frobenius distance.
    """
    from scipy.stats import pearsonr, spearmanr

    e = estimated.copy()
    g = ground_truth.copy()

    if exclude_diag:
        M = e.shape[0]
        mask = ~np.eye(M, dtype=bool)
        e = e[mask]
        g = g[mask]
    else:
        e = e.flatten()
        g = g.flatten()

    pr, _ = pearsonr(e, g)
    sr, _ = spearmanr(e, g)
    frob = float(np.sqrt(((e - g) ** 2).sum()))
    return {
        'pearson_r':       float(pr),
        'spearman_rho':    float(sr),
        'frobenius_dist':  frob,
    }


# =====================================================================
# Top-level reporter
# =====================================================================

@dataclass
class MSIMReport:
    msim_overall: np.ndarray
    msim_per_layer: Dict[str, np.ndarray]
    directionality: Dict[str, np.ndarray]
    time_varying: Dict[pd.Timestamp, np.ndarray]
    regime_conditional: Dict[str, np.ndarray]
    lagged: np.ndarray
    recovery: Optional[Dict[str, float]]
    markets: List[str]

    def save(self, out_dir: Path):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Overall MSIM as CSV with named columns
        pd.DataFrame(self.msim_overall, index=self.markets,
                     columns=self.markets
                     ).to_csv(out_dir / 'msim_overall.csv')

        # Per-layer
        for k, v in self.msim_per_layer.items():
            pd.DataFrame(v, index=self.markets,
                         columns=self.markets
                         ).to_csv(out_dir / f'msim_{k}.csv')

        # Directionality
        di = pd.DataFrame({
            'market': self.markets,
            'to_others': self.directionality['to_others'],
            'from_others': self.directionality['from_others'],
            'net': self.directionality['net'],
        })
        di.to_csv(out_dir / 'directionality.csv', index=False)
        with open(out_dir / 'total_spillover.txt', 'w') as f:
            f.write(f'{self.directionality["total"]:.4f}\n')

        # Time-varying
        if self.time_varying:
            tv_data = []
            for ts, m in sorted(self.time_varying.items()):
                row = {'timestamp': ts}
                for i, mi in enumerate(self.markets):
                    for j, mj in enumerate(self.markets):
                        row[f'{mi}->{mj}'] = m[i, j]
                tv_data.append(row)
            pd.DataFrame(tv_data).to_csv(out_dir / 'msim_time_varying.csv',
                                          index=False)

        # Regime conditional
        for name, m in self.regime_conditional.items():
            pd.DataFrame(m, index=self.markets, columns=self.markets
                         ).to_csv(out_dir / f'msim_regime_{name}.csv')

        # Lagged
        np.save(out_dir / 'msim_lagged.npy', self.lagged)

        # Recovery
        if self.recovery is not None:
            with open(out_dir / 'recovery.txt', 'w') as f:
                for k, v in self.recovery.items():
                    f.write(f'{k}: {v:.4f}\n')

        logger.info(f'MSIM report saved to {out_dir}')


def build_msim_report(attn_dump: Dict[str, np.ndarray],
                      gate_probs: Optional[np.ndarray],
                      timestamps: Optional[np.ndarray],
                      markets: List[str],
                      ground_truth: Optional[np.ndarray] = None
                      ) -> MSIMReport:
    """All-in-one: produce a full MSIMReport."""
    msim_avg, per_layer = aggregate_multilayer(attn_dump, normalize_rows=True)

    direction = directionality_index(msim_avg)

    # Use the LAST cross layer for time-varying & lagged (deepest semantic)
    cross_keys = sorted(k for k in attn_dump
                        if k.startswith('cross_') and k.endswith('cross'))
    last = attn_dump[cross_keys[-1]]                                # (N,Np,H,M,M)

    tv = {}
    if timestamps is not None and len(timestamps) == last.shape[0]:
        tv = time_varying_msim(last, timestamps)
    rc = {}
    if gate_probs is not None and gate_probs.shape[0] == last.shape[0]:
        rc = regime_conditional_msim(last, gate_probs)
    lag = lagged_spillover(last)

    rec = None
    if ground_truth is not None:
        rec = msim_recovery(msim_avg, ground_truth)

    return MSIMReport(
        msim_overall=msim_avg,
        msim_per_layer=per_layer,
        directionality=direction,
        time_varying=tv,
        regime_conditional=rc,
        lagged=lag,
        recovery=rec,
        markets=markets,
    )
