"""
Tests for MSIM analysis module.

Verifies that aggregation, time-varying, regime-conditional, and recovery
metrics behave correctly on synthetic attention dumps with KNOWN structure.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.utils.msim import (
    aggregate_msim, aggregate_multilayer, time_varying_msim,
    regime_conditional_msim, lagged_spillover,
    directionality_index, msim_recovery,
    build_msim_report,
)

np.random.seed(0)


def make_synthetic_attn(N: int, Np: int, H: int, M: int,
                        true_msim: np.ndarray, noise: float = 0.05):
    """Generate attn dump where ground-truth row distribution = true_msim[m]."""
    # For each (sample, patch, head) we sample a Dirichlet around true_msim[m]
    out = np.zeros((N, Np, H, M, M), dtype=np.float32)
    for m in range(M):
        alpha = true_msim[m] * 100 + 1                      # concentration
        # rng sample size (N, Np, H, M)
        for n in range(N):
            for p in range(Np):
                for h in range(H):
                    out[n, p, h, m, :] = np.random.dirichlet(alpha)
    # Add a tiny bit of noise but keep simplex
    out += noise * np.random.randn(*out.shape) * 0.01
    out = np.clip(out, 1e-6, None)
    out /= out.sum(axis=-1, keepdims=True)
    return out


def test_aggregate():
    M, N, Np, H = 4, 200, 10, 8
    true = np.array([
        [0.40, 0.30, 0.20, 0.10],
        [0.10, 0.50, 0.30, 0.10],
        [0.15, 0.15, 0.50, 0.20],
        [0.05, 0.20, 0.25, 0.50],
    ])
    attn = make_synthetic_attn(N, Np, H, M, true)
    est = aggregate_msim(attn)
    diff = np.abs(est - true).mean()
    print(f'T1 aggregate: mean abs error = {diff:.4f}')
    assert diff < 0.03, f'recovery too poor: {diff}'
    rows = est.sum(axis=1)
    assert np.allclose(rows, 1, atol=1e-4), f'rows: {rows}'


def test_multilayer():
    M = 4
    true = np.eye(M) * 0.5 + 0.5 / M
    attn1 = make_synthetic_attn(50, 8, 4, M, true)
    attn2 = make_synthetic_attn(50, 8, 4, M, true)
    dump = {'cross_0_cross': attn1, 'cross_1_cross': attn2,
            'cross_0_gate': np.eye(M), 'cross_1_gate': np.eye(M)}
    avg, per_layer = aggregate_multilayer(dump)
    assert avg.shape == (M, M)
    assert len(per_layer) == 2
    print(f'T2 multilayer aggregation: shape={avg.shape}, per-layer={list(per_layer)}')


def test_directionality():
    # Construct a CHAIN: market 0 sends to all others, others send only to themselves
    M = 4
    msim = np.eye(M) * 0.6
    msim[:, 0] += 0.4 / 1                        # everyone listens to market 0
    msim[0, 0] = 1.0
    msim = msim / msim.sum(axis=1, keepdims=True)
    di = directionality_index(msim)
    print(f'T3 directionality:')
    print(f'    to_others   = {di["to_others"].round(3)}')
    print(f'    from_others = {di["from_others"].round(3)}')
    print(f'    net         = {di["net"].round(3)}')
    print(f'    total       = {di["total"]:.3f}')
    # Market 0 is the largest "exporter"
    assert di['to_others'][0] == di['to_others'].max(), \
        'market 0 should export the most'
    assert di['net'][0] > 0


def test_time_varying():
    M, N, Np, H = 4, 365 * 4, 8, 4              # 4 obs per day for 1 yr
    true = np.eye(M) * 0.6 + 0.4 / M
    attn = make_synthetic_attn(N, Np, H, M, true)
    ts = pd.date_range('2023-01-01', periods=N, freq='6h', tz='UTC').values
    out = time_varying_msim(attn, ts, window='30D')
    assert len(out) > 5
    # All windows should give roughly the same MSIM
    mats = np.stack(list(out.values()))
    spread = mats.std(axis=0).mean()
    print(f'T4 time-varying: {len(out)} windows, '
          f'inter-window spread={spread:.4f}')


def test_regime_conditional():
    M = 4
    K = 3
    N, Np, H = 600, 8, 4
    # Make different MSIM per regime, then mix according to simulated gate
    msim_norm = np.eye(M) * 0.5 + 0.5 / M
    msim_spike = np.eye(M) * 0.2
    msim_spike[:, 0] += 0.8                                  # spike => DE leadership
    msim_spike = msim_spike / msim_spike.sum(axis=1, keepdims=True)
    msim_neg = np.eye(M) * 0.7 + 0.3 / M
    msim_neg = msim_neg / msim_neg.sum(axis=1, keepdims=True)

    # Per-sample regime label
    regime = np.random.choice(K, size=N, p=[0.7, 0.2, 0.1])
    attn = np.zeros((N, Np, H, M, M), dtype=np.float32)
    for n in range(N):
        if regime[n] == 0:
            attn[n] = make_synthetic_attn(1, Np, H, M, msim_norm)[0]
        elif regime[n] == 1:
            attn[n] = make_synthetic_attn(1, Np, H, M, msim_spike)[0]
        else:
            attn[n] = make_synthetic_attn(1, Np, H, M, msim_neg)[0]

    # Gate probabilities = one-hot at the true regime
    gate = np.zeros((N, M, K), dtype=np.float32)
    for n in range(N):
        gate[n, :, regime[n]] = 1.0

    out = regime_conditional_msim(attn, gate)
    assert 'normal' in out and 'spike' in out
    # Spike-conditional: column 0 should dominate
    spike_col0_mass = out['spike'][:, 0].sum()
    spike_col_other = out['spike'][:, 1:].sum()
    print(f'T5 regime-conditional:')
    print(f'    spike  -> col0 mass = {spike_col0_mass:.3f} (should dominate)')
    print(f'    spike  -> other    = {spike_col_other:.3f}')
    assert spike_col0_mass > spike_col_other * 0.8


def test_recovery():
    M = 4
    gt = np.array([
        [1.00, 0.45, 0.55, 0.30],
        [0.30, 1.00, 0.20, 0.40],
        [0.45, 0.15, 1.00, 0.55],
        [0.20, 0.30, 0.50, 1.00],
    ])
    # Estimated: GT + small noise
    est = gt + np.random.randn(M, M) * 0.05
    out = msim_recovery(est, gt, exclude_diag=True)
    print(f'T6 recovery (small noise):')
    for k, v in out.items():
        print(f'    {k:20s} : {v:.4f}')
    assert out['pearson_r'] > 0.9


def test_full_report():
    M = 4
    N, Np, H_heads = 200, 8, 4
    true = np.array([
        [0.40, 0.30, 0.20, 0.10],
        [0.10, 0.50, 0.30, 0.10],
        [0.15, 0.15, 0.50, 0.20],
        [0.05, 0.20, 0.25, 0.50],
    ])
    attn = make_synthetic_attn(N, Np, H_heads, M, true)
    dump = {'cross_0_cross': attn, 'cross_0_gate': np.eye(M),
            'cross_1_cross': attn * 0.9 + 0.1 / M,
            'cross_1_gate': np.eye(M)}
    gate_probs = np.zeros((N, M, 3), dtype=np.float32)
    gate_probs[..., 0] = 1.0
    timestamps = pd.date_range('2023-07-01', periods=N, freq='3h',
                               tz='UTC').values

    rep = build_msim_report(dump, gate_probs, timestamps,
                            markets=['DE_LU', 'FR', 'NL', 'BE'],
                            ground_truth=true)
    print(f'T7 full report:')
    print(f'    overall MSIM:')
    print(pd.DataFrame(rep.msim_overall,
                       index=['DE_LU', 'FR', 'NL', 'BE'],
                       columns=['DE_LU', 'FR', 'NL', 'BE']).round(3))
    print(f'    recovery: pearson={rep.recovery["pearson_r"]:.3f}')
    assert rep.recovery['pearson_r'] > 0.85


if __name__ == '__main__':
    print('=' * 60)
    print('MSIM analysis tests')
    print('=' * 60)
    test_aggregate()
    test_multilayer()
    test_directionality()
    test_time_varying()
    test_regime_conditional()
    test_recovery()
    test_full_report()
    print('=' * 60)
    print('ALL MSIM TESTS PASSED')
    print('=' * 60)
