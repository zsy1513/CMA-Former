"""
Visualization helpers for training diagnostics and result inspection.

Provides:
  - training loss curves
  - prediction vs ground truth (sample window) per market
  - forecast error distribution per model
  - MSIM heatmaps and spillover summaries
  - recovery scatter for synthetic validation

All figures use matplotlib, publication-style: serif font, vector PDF output.
Color palette is colorblind-safe (Tol's colorscheme).

Author: CMA-Former Project
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                                      # headless
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logger = logging.getLogger('cma_former.viz')

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,                                    # editable text in PDF
    'ps.fonttype': 42,
})

# Tol's colorblind-safe qualitative scheme
TOL = ['#4477AA', '#EE6677', '#228833', '#CCBB44',
       '#66CCEE', '#AA3377', '#BBBBBB']


# =====================================================================
# Training loss curves
# =====================================================================

def plot_training_curves(history: List[Dict], out_path: Path,
                         title: str = 'Training & Validation Loss'):
    df = pd.DataFrame(history)
    epochs = df['epoch'].values

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    ax.plot(epochs, df['train_total'], color=TOL[0], label='Train (total)',
            linewidth=1.5)
    if 'val_loss' in df:
        ax.plot(epochs, df['val_loss'], color=TOL[1], label='Validation',
                linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Composite loss')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(frameon=False)

    ax = axes[1]
    components = [c for c in ['train_quantile', 'train_regime', 'train_sparse']
                  if c in df.columns]
    for i, c in enumerate(components):
        ax.plot(epochs, df[c], color=TOL[i], label=c.replace('train_', ''),
                linewidth=1.2, alpha=0.85)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Components')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(frameon=False)

    fig.suptitle(title)
    fig.savefig(out_path, format='pdf')
    plt.close(fig)


# =====================================================================
# Prediction vs ground truth
# =====================================================================

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                     y_low: np.ndarray, y_high: np.ndarray,
                     timestamps: np.ndarray,
                     market_names: List[str],
                     out_path: Path,
                     n_show_hours: int = 168):
    """
    y_true/y_pred/y_low/y_high : (T, M)  flat hourly predictions
    timestamps                 : (T,)
    """
    M = len(market_names)
    fig, axes = plt.subplots(M, 1, figsize=(11, 1.8 * M), sharex=True)
    if M == 1:
        axes = [axes]
    n = min(n_show_hours, len(timestamps))
    ts = pd.to_datetime(timestamps[:n])

    for i, name in enumerate(market_names):
        ax = axes[i]
        ax.plot(ts, y_true[:n, i], color='black', label='Actual',
                linewidth=1.0)
        ax.plot(ts, y_pred[:n, i], color=TOL[0], label='CMA-Former (q50)',
                linewidth=1.2)
        ax.fill_between(ts, y_low[:n, i], y_high[:n, i],
                        color=TOL[0], alpha=0.18, label='80% PI')
        ax.set_ylabel(f'{name}\n€/MWh')
        ax.grid(True, linestyle='--', alpha=0.4)
        if i == 0:
            ax.legend(loc='upper right', frameon=False, ncol=3)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    axes[-1].set_xlabel('Date')
    fig.savefig(out_path, format='pdf')
    plt.close(fig)


# =====================================================================
# Error distribution boxplot
# =====================================================================

def plot_error_boxplot(errors: Dict[str, np.ndarray], out_path: Path,
                       metric_name: str = 'absolute error'):
    """errors: {model_name: 1-D numpy array of per-sample errors}"""
    names = list(errors.keys())
    data = [errors[n] for n in names]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bp = ax.boxplot(data, labels=names, showfliers=False, patch_artist=True,
                    medianprops={'color': 'black', 'linewidth': 1.4})
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(TOL[i % len(TOL)])
        patch.set_alpha(0.7)
    ax.set_ylabel(f'{metric_name} (€/MWh)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.xticks(rotation=30, ha='right')
    fig.savefig(out_path, format='pdf')
    plt.close(fig)


# =====================================================================
# MSIM heatmap
# =====================================================================

def plot_msim_heatmap(msim: np.ndarray, market_names: List[str],
                      out_path: Path,
                      title: str = 'Market Spillover Intensity Matrix',
                      cmap: str = 'magma_r',
                      annotate: bool = True):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(msim, cmap=cmap, vmin=0, vmax=msim.max(),
                   aspect='auto')
    ax.set_xticks(np.arange(len(market_names)))
    ax.set_yticks(np.arange(len(market_names)))
    ax.set_xticklabels(market_names)
    ax.set_yticklabels(market_names)
    ax.set_xlabel('Source market (key)')
    ax.set_ylabel('Target market (query)')
    ax.set_title(title)

    if annotate:
        for i in range(msim.shape[0]):
            for j in range(msim.shape[1]):
                v = msim[i, j]
                color = 'white' if v > msim.max() * 0.6 else 'black'
                ax.text(j, i, f'{v:.2f}',
                        ha='center', va='center', color=color, fontsize=9)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Spillover share')
    fig.savefig(out_path, format='pdf')
    plt.close(fig)


# =====================================================================
# Time-varying total spillover index
# =====================================================================

def plot_time_varying_spillover(tv_msim: Dict, market_names: List[str],
                                out_path: Path,
                                title: str = 'Time-varying spillover'):
    if not tv_msim:
        logger.warning('No time-varying MSIM to plot.')
        return
    timestamps = sorted(tv_msim.keys())
    M = len(market_names)
    eye = np.eye(M, dtype=bool)

    total_idx = []
    for ts in timestamps:
        m = tv_msim[ts].copy()
        m[eye] = 0
        total_idx.append(m.sum() / M)
    total_idx = np.array(total_idx)

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True,
                              gridspec_kw={'height_ratios': [1.2, 1]})

    ax = axes[0]
    ax.plot(timestamps, total_idx, color=TOL[0], linewidth=1.6)
    ax.set_ylabel('Total spillover\n(off-diagonal mass)')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.4)

    # Crisis window shading: 2021-09 to 2023-03
    crisis_start = pd.Timestamp('2021-09-01', tz='UTC')
    crisis_end = pd.Timestamp('2023-03-31', tz='UTC')
    if any(t >= crisis_start and t <= crisis_end for t in timestamps):
        ax.axvspan(crisis_start, crisis_end, color='#EE6677', alpha=0.10,
                   label='Energy crisis')
        ax.legend(loc='upper right', frameon=False)

    # Pairwise: each market's net export
    ax = axes[1]
    for i, m in enumerate(market_names):
        net = []
        for ts in timestamps:
            mat = tv_msim[ts]
            off = mat.copy()
            off[eye] = 0
            net.append(off[:, i].sum() - off[i, :].sum())
        ax.plot(timestamps, net, color=TOL[i], linewidth=1.4, label=m)
    ax.set_ylabel('Net spillover\n(export – import)')
    ax.set_xlabel('Date')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(frameon=False, ncol=4, loc='upper right')
    ax.axhline(0, color='black', linewidth=0.7, alpha=0.6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    fig.savefig(out_path, format='pdf')
    plt.close(fig)


# =====================================================================
# Regime-conditional MSIM heatmaps (3-panel)
# =====================================================================

def plot_regime_conditional(rc: Dict[str, np.ndarray],
                            market_names: List[str], out_path: Path):
    if not rc:
        logger.warning('No regime-conditional MSIM to plot.')
        return
    regimes = list(rc.keys())
    n = len(regimes)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    vmax = max(m.max() for m in rc.values())

    for ax, name in zip(axes, regimes):
        m = rc[name]
        im = ax.imshow(m, cmap='magma_r', vmin=0, vmax=vmax, aspect='auto')
        ax.set_xticks(np.arange(len(market_names)))
        ax.set_yticks(np.arange(len(market_names)))
        ax.set_xticklabels(market_names)
        ax.set_yticklabels(market_names)
        ax.set_title(f'Regime: {name}')
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m[i, j]
                color = 'white' if v > vmax * 0.6 else 'black'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        color=color, fontsize=8)
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label='Spillover share')
    fig.savefig(out_path, format='pdf')
    plt.close(fig)


# =====================================================================
# Lagged spillover heatmap
# =====================================================================

def plot_lagged_spillover(lagged: np.ndarray, market_names: List[str],
                          out_path: Path,
                          patch_size: int = 16, patch_stride: int = 8):
    """
    lagged: (N_patches, M, M)
    Plots, for each (target, source) pair, how spillover varies with patch lag.
    Shows the off-diagonal element changes as a heatmap (target market on rows,
    patch lag on columns), one panel per source market.
    """
    Np, M, _ = lagged.shape
    fig, axes = plt.subplots(1, M, figsize=(4 * M, 4.5), sharey=True)
    if M == 1:
        axes = [axes]

    # Convert patch index to "hours ago"
    # Patch 0 is oldest; the centre-of-window for patch p is approx
    #   (Np - 1 - p) * patch_stride + patch_size/2 hours ago.
    hours_ago = [int((Np - 1 - p) * patch_stride + patch_size / 2)
                 for p in range(Np)]
    vmax = lagged.max()

    for k_src, ax in enumerate(axes):
        # Show row m vs patch p of lagged[p, m, k_src]
        mat = np.array([lagged[p, :, k_src] for p in range(Np)]).T  # (M, Np)
        im = ax.imshow(mat, cmap='magma_r', aspect='auto',
                       vmin=0, vmax=vmax,
                       extent=[hours_ago[0], hours_ago[-1], M - 0.5, -0.5])
        ax.set_yticks(np.arange(M))
        ax.set_yticklabels(market_names)
        ax.set_xlabel('Hours before forecast')
        ax.set_title(f'Source: {market_names[k_src]}')
        ax.invert_xaxis()                                  # most recent on the right? No, leftmost = oldest

    fig.colorbar(im, ax=axes, fraction=0.04, pad=0.04, label='Spillover share')
    fig.savefig(out_path, format='pdf')
    plt.close(fig)


# =====================================================================
# Recovery scatter
# =====================================================================

def plot_recovery_scatter(estimated: np.ndarray, ground_truth: np.ndarray,
                          market_names: List[str], out_path: Path,
                          stats: Dict = None):
    M = estimated.shape[0]
    mask = ~np.eye(M, dtype=bool)

    e = estimated[mask]
    g = ground_truth[mask]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(g, e, color=TOL[0], s=60, alpha=0.8)
    lim_max = max(g.max(), e.max()) * 1.05
    ax.plot([0, lim_max], [0, lim_max], color='gray', linestyle='--',
            label='y = x', linewidth=1)
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_xlabel('Ground-truth spillover')
    ax.set_ylabel('Estimated MSIM (off-diagonal)')
    ax.set_title('MSIM recovery vs ground truth')
    ax.grid(True, linestyle='--', alpha=0.4)

    if stats is not None:
        text = (f'Pearson r = {stats["pearson_r"]:.3f}\n'
                f'Spearman ρ = {stats["spearman_rho"]:.3f}\n'
                f'Frobenius = {stats["frobenius_dist"]:.3f}')
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                ha='left', va='top',
                bbox=dict(facecolor='white', edgecolor='black',
                          boxstyle='round,pad=0.4', alpha=0.85))

    ax.legend(loc='lower right', frameon=False)
    fig.savefig(out_path, format='pdf')
    plt.close(fig)
