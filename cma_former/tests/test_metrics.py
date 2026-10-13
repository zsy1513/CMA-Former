"""Numerical tests for metrics module (no torch needed)."""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.metrics import (
    mae, rmse, smape, rmae,
    pinball_loss, crps_from_quantiles, coverage,
    interval_width, winkler_score, quantile_crossing_rate,
    spike_metrics, negative_metrics,
    diebold_mariano, giacomini_white,
    daily_loss_panel, diebold_mariano_from_daily_losses,
    moving_block_bootstrap_loss_difference,
    compute_all_metrics,
)

np.random.seed(0)


def test_point_metrics():
    n = 1000
    y = np.random.randn(n) * 50 + 100              # mean ~ 100
    y_pred = y + np.random.randn(n) * 10           # add noise
    y_naive = np.roll(y, 24)

    m = mae(y, y_pred)
    r = rmse(y, y_pred)
    s = smape(y, y_pred)
    rm = rmae(y, y_pred, y_naive)

    # Sanity: MAE ≈ noise scale * sqrt(2/pi) ≈ 10 * 0.798
    assert 6 < m < 12, f'mae={m}'
    assert r > m                                    # always RMSE >= MAE
    assert 0 < s < 100
    assert rm < 1                                   # better than naive
    print(f'T1 point metrics: MAE={m:.3f}, RMSE={r:.3f}, '
          f'SMAPE={s:.3f}, rMAE={rm:.3f}')


def test_pinball_loss():
    n = 500
    y = np.random.randn(n)
    quantiles = [0.1, 0.5, 0.9]
    # Perfect predictions: each quantile = corresponding empirical quantile
    # of the conditional distribution, but here y has no covariate -> use
    # marginal quantiles of standard normal
    from scipy.stats import norm
    q_perfect = np.array([norm.ppf(q) for q in quantiles])
    yq = np.tile(q_perfect, (n, 1))
    pl = pinball_loss(y, yq, quantiles)
    # Should be small but positive (irreducible noise)
    assert 0.1 < pl < 1.0, f'pl={pl}'

    # Compare: degenerate case (all quantiles = mean=0)
    yq_bad = np.zeros((n, 3))
    pl_bad = pinball_loss(y, yq_bad, quantiles)
    assert pl_bad > pl, 'optimal should be better than naive zero'
    print(f'T2 pinball: optimal={pl:.3f} < naive={pl_bad:.3f}')


def test_crps():
    n = 500
    y = np.random.randn(n)
    quantiles = [0.1, 0.5, 0.9]
    yq = np.tile([-1.28, 0.0, 1.28], (n, 1))         # ~ ideal quantiles
    c = crps_from_quantiles(y, yq, quantiles)
    assert c > 0
    print(f'T3 CRPS: {c:.3f}')


def test_coverage():
    n = 1000
    y = np.random.randn(n)
    lo = np.full(n, -1.28)                            # 10th percentile
    hi = np.full(n,  1.28)                            # 90th percentile
    cov = coverage(y, lo, hi)
    assert 0.75 < cov < 0.85, f'coverage={cov}'
    print(f'T4 coverage(80% nominal): {cov*100:.1f}%')


def test_spike_metrics():
    n = 1000
    y = np.random.randn(n) * 30 + 50
    y[::20] += 200                                    # 5% spikes
    pred_perfect = y.copy()
    m = spike_metrics(y, pred_perfect, spike_threshold=150)
    assert m['spike_f1'] > 0.99
    pred_zero = np.zeros_like(y)
    m2 = spike_metrics(y, pred_zero, spike_threshold=150)
    assert m2['spike_f1'] < 0.01
    print(f'T5 spike metrics: perfect F1={m["spike_f1"]:.3f}, '
          f'naive F1={m2["spike_f1"]:.3f}')


def test_dm_test():
    n = 1000
    y = np.random.randn(n)
    # Model A: noise sd=0.5  Model B: noise sd=1.0
    pred_a = y + np.random.randn(n) * 0.5
    pred_b = y + np.random.randn(n) * 1.0

    out = diebold_mariano(y, pred_a, pred_b, h=24, loss='mae')
    assert out['mean_loss_diff'] < 0, 'A should have smaller loss'
    assert out['p_value'] < 0.001, f'p-value={out["p_value"]}'
    assert out['a_better']
    print(f'T6 DM test: stat={out["dm_stat_hln"]:.3f}, '
          f'p={out["p_value"]:.5f}, a_better={out["a_better"]}')


def test_gw_test():
    n = 1000
    y = np.random.randn(n)
    pred_a = y + np.random.randn(n) * 0.3
    pred_b = y + np.random.randn(n) * 1.0

    out = giacomini_white(y, pred_a, pred_b)
    assert out['gw_stat'] > 0
    assert out['p_value'] < 0.001
    print(f'T7 GW test: stat={out["gw_stat"]:.2f}, p={out["p_value"]:.5f}')


def test_full_panel():
    n = 500
    y = np.random.randn(n) * 30 + 80
    y_pred = y + np.random.randn(n) * 5
    yq = np.stack([y_pred - 8, y_pred, y_pred + 8], axis=-1)
    y_naive = np.roll(y, 24)

    out = compute_all_metrics(y, y_pred, yq, [0.1, 0.5, 0.9], y_naive,
                              spike_thr=150)
    print('T8 full panel:')
    for k, v in out.items():
        print(f'    {k:20s} : {v:.4f}')


def test_probabilistic_interval_metrics():
    y = np.array([0.0, 1.0, 2.0])
    lo = np.array([-1.0, 0.0, 1.0])
    hi = np.array([1.0, 2.0, 3.0])
    assert interval_width(lo, hi) == 2.0
    assert winkler_score(y, lo, hi, alpha=0.2) == 2.0
    ordered = np.stack([lo, y, hi], axis=-1)
    crossed = ordered.copy()
    crossed[0, 0], crossed[0, 1] = crossed[0, 1], crossed[0, 0]
    assert quantile_crossing_rate(ordered) == 0.0
    assert quantile_crossing_rate(crossed) > 0.0


def test_day_level_dm_and_block_bootstrap():
    rng = np.random.default_rng(4)
    y = rng.normal(size=(300, 4, 24))
    pred_a = y + rng.normal(scale=0.4, size=y.shape)
    pred_b = y + rng.normal(scale=0.9, size=y.shape)
    loss_a = daily_loss_panel(y, pred_a)
    loss_b = daily_loss_panel(y, pred_b)
    result = diebold_mariano_from_daily_losses(loss_a[:, 0], loss_b[:, 0])
    assert result['n_days'] == 300
    assert result['mean_loss_diff'] < 0
    assert result['p_value'] < 0.001
    boot = moving_block_bootstrap_loss_difference(
        loss_a[:, 0], loss_b[:, 0], n_resamples=200, seed=3)
    assert boot['ci_97.5'] < 0


if __name__ == '__main__':
    print('=' * 60)
    print('Metrics module sanity tests')
    print('=' * 60)
    test_point_metrics()
    test_pinball_loss()
    test_crps()
    test_coverage()
    test_spike_metrics()
    test_dm_test()
    test_gw_test()
    test_full_panel()
    print('=' * 60)
    print('ALL METRICS TESTS PASSED')
    print('=' * 60)
