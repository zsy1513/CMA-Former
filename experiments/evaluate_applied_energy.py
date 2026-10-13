"""Aggregate multi-seed results and perform delivery-day inference."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.metrics import (
    daily_loss_panel, diebold_mariano_from_daily_losses,
    moving_block_bootstrap_loss_difference)


def _prediction_runs(root: Path):
    runs = []
    for path in root.glob('*/seed_*/predictions.npz'):
        match = re.match(r'seed_(\d+)', path.parent.name)
        if match:
            runs.append((path.parent.parent.name, int(match.group(1)), path))
    return runs


def _load_daily(path: Path):
    arrays = np.load(path)
    truth = arrays['y_true_raw']
    prediction = arrays['y_pred_raw']
    dates = arrays['delivery_date_ordinal']
    return daily_loss_panel(truth, prediction), dates


def _holm_adjust(frame: pd.DataFrame) -> pd.DataFrame:
    """Holm-adjust p-values within one reported comparison family."""
    if frame.empty:
        return frame
    result = frame.copy()
    order = np.argsort(result['p_value'].to_numpy(dtype=float))
    raw = result['p_value'].to_numpy(dtype=float)[order]
    adjusted = np.maximum.accumulate(
        (len(raw) - np.arange(len(raw))) * raw)
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result['p_value_holm'] = restored
    result['significant_holm_0.05'] = restored < 0.05
    return result


def _mean_seed_daily(run_paths):
    """Average daily loss across seeds after exact date alignment."""
    loaded = [_load_daily(path) for path in run_paths]
    common = loaded[0][1]
    for _, dates in loaded[1:]:
        common = np.intersect1d(common, dates)
    aligned = []
    for losses, dates in loaded:
        lookup = {int(day): index for index, day in enumerate(dates)}
        aligned.append(losses[[lookup[int(day)] for day in common]])
    return np.mean(np.stack(aligned, axis=0), axis=0), common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-root', default='results/applied_energy')
    parser.add_argument('--cpu-root', default='results/applied_energy_cpu')
    parser.add_argument('--out', default='results/applied_energy_analysis')
    parser.add_argument('--reference', default='cma_former')
    args = parser.parse_args()
    gpu_root, cpu_root = Path(args.gpu_root), Path(args.cpu_root)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    for metrics_path in gpu_root.glob('*/seed_*/metrics.json'):
        seed = int(metrics_path.parent.name.replace('seed_', ''))
        model = metrics_path.parent.parent.name
        with open(metrics_path, encoding='utf-8') as handle:
            metrics = json.load(handle)
        metric_rows.append({'model': model, 'seed': seed,
                            **metrics['macro']})
    for model in ('lear', 'naive_previous_day'):
        path = cpu_root / model / 'metrics.json'
        if path.exists():
            with open(path, encoding='utf-8') as handle:
                metrics = json.load(handle)
            metric_rows.append({'model': model, 'seed': 'deterministic',
                                **metrics['macro']})
    long_metrics = pd.DataFrame(metric_rows)
    long_metrics.to_csv(output / 'metrics_by_seed.csv', index=False)
    neural = long_metrics[pd.to_numeric(
        long_metrics['seed'], errors='coerce').notna()].copy()
    numeric = [c for c in neural.columns if c not in ('model', 'seed')]
    summary = neural.groupby('model')[numeric].agg(['mean', 'std'])
    summary.to_csv(output / 'metrics_mean_std.csv')
    tidy = neural.groupby('model')[numeric].agg(['mean', 'std', 'count'])
    tidy.columns = ['_'.join(column) for column in tidy.columns]
    tidy.reset_index().to_csv(output / 'metrics_summary_tidy.csv', index=False)

    run_lookup = {(model, seed): path for model, seed, path
                  in _prediction_runs(gpu_root)}
    reference_runs = {seed: path for (model, seed), path in run_lookup.items()
                      if model == args.reference}
    comparison_rows = []
    for seed, reference_path in reference_runs.items():
        ref_loss, ref_dates = _load_daily(reference_path)
        competitors = [(model, path) for (model, other_seed), path
                       in run_lookup.items() if other_seed == seed
                       and model != args.reference]
        for cpu_model in ('lear', 'naive_previous_day'):
            cpu_path = cpu_root / cpu_model / 'predictions.npz'
            if cpu_path.exists():
                competitors.append((cpu_model, cpu_path))
        for model, path in competitors:
            other_loss, other_dates = _load_daily(path)
            common, ref_idx, other_idx = np.intersect1d(
                ref_dates, other_dates, return_indices=True)
            for market_index in range(ref_loss.shape[1]):
                dm = diebold_mariano_from_daily_losses(
                    ref_loss[ref_idx, market_index],
                    other_loss[other_idx, market_index])
                bootstrap = moving_block_bootstrap_loss_difference(
                    ref_loss[ref_idx, market_index],
                    other_loss[other_idx, market_index],
                    block_length=7, n_resamples=2000, seed=seed)
                comparison_rows.append({
                    'reference': args.reference, 'other': model,
                    'seed': seed, 'market_index': market_index,
                    'n_common_days': len(common), **dm,
                    'bootstrap_ci_2.5': bootstrap['ci_2.5'],
                    'bootstrap_ci_97.5': bootstrap['ci_97.5'],
                })
    seed_comparisons = _holm_adjust(pd.DataFrame(comparison_rows))
    seed_comparisons.to_csv(
        output / 'daily_dm_and_block_bootstrap.csv', index=False)

    # Primary inference uses one loss per delivery day and market, averaged
    # across random initializations before testing. Seeds are therefore a
    # stability dimension, not extra independent observations.
    by_model = {}
    for model, seed, path in _prediction_runs(gpu_root):
        by_model.setdefault(model, []).append(path)
    if args.reference in by_model:
        reference_loss, reference_dates = _mean_seed_daily(
            by_model[args.reference])
        aggregate_rows = []
        aggregate_competitors = {
            model: _mean_seed_daily(paths)
            for model, paths in by_model.items() if model != args.reference
        }
        for cpu_model in ('lear', 'naive_previous_day'):
            cpu_path = cpu_root / cpu_model / 'predictions.npz'
            if cpu_path.exists():
                aggregate_competitors[cpu_model] = _load_daily(cpu_path)
        for model, (other_loss, other_dates) in aggregate_competitors.items():
            common, ref_idx, other_idx = np.intersect1d(
                reference_dates, other_dates, return_indices=True)
            for market_index in range(reference_loss.shape[1]):
                dm = diebold_mariano_from_daily_losses(
                    reference_loss[ref_idx, market_index],
                    other_loss[other_idx, market_index])
                bootstrap = moving_block_bootstrap_loss_difference(
                    reference_loss[ref_idx, market_index],
                    other_loss[other_idx, market_index],
                    block_length=7, n_resamples=2000, seed=42)
                aggregate_rows.append({
                    'reference': args.reference, 'other': model,
                    'market_index': market_index,
                    'reference_seeds': len(by_model[args.reference]),
                    'other_seeds': (len(by_model[model])
                                    if model in by_model else 1),
                    'n_common_days': len(common), **dm,
                    'bootstrap_ci_2.5': bootstrap['ci_2.5'],
                    'bootstrap_ci_97.5': bootstrap['ci_97.5'],
                })
        _holm_adjust(pd.DataFrame(aggregate_rows)).to_csv(
            output / 'aggregate_daily_dm_and_block_bootstrap.csv', index=False)
    print(f'Wrote analysis to {output}')


if __name__ == '__main__':
    main()
