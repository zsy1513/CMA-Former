"""Verify completeness and internal consistency of the frozen result matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MODELS = [
    'cma_former', 'patchtst', 'dlinear', 'nbeatsx', 'tft',
    'itransformer', 'temporal_gat', 'single_market_patchtst',
    'concatenated_transformer', 'cma_no_cross', 'cma_topology_prior',
    'cma_fixed_prior', 'cma_shuffled_prior', 'cma_late_future',
]
SEEDS = [11, 22, 42, 77, 123]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-root', required=True)
    parser.add_argument('--cpu-root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    gpu_root, cpu_root = Path(args.gpu_root), Path(args.cpu_root)
    reference_truth = reference_dates = None
    checks = []
    for model in MODELS:
        for seed in SEEDS:
            run = gpu_root / model / f'seed_{seed}'
            prediction_path, metrics_path = (run / 'predictions.npz',
                                             run / 'metrics.json')
            if not prediction_path.exists() or not metrics_path.exists():
                raise FileNotFoundError(f'incomplete run: {run}')
            arrays = np.load(prediction_path)
            truth = arrays['y_true_raw']
            prediction = arrays['y_pred_raw']
            quantiles = arrays['y_pred_quantiles_raw']
            dates = arrays['delivery_date_ordinal']
            if truth.shape != (544, 4, 24):
                raise ValueError(f'unexpected target shape at {run}: {truth.shape}')
            if prediction.shape != truth.shape or quantiles.shape != (544, 4, 24, 3):
                raise ValueError(f'unexpected prediction shape at {run}')
            if not (np.isfinite(truth).all() and np.isfinite(prediction).all()
                    and np.isfinite(quantiles).all()):
                raise ValueError(f'non-finite prediction content at {run}')
            if reference_truth is None:
                reference_truth, reference_dates = truth.copy(), dates.copy()
            elif not (np.array_equal(truth, reference_truth)
                      and np.array_equal(dates, reference_dates)):
                raise ValueError(f'target/date mismatch at {run}')
            macro_mae = float(np.mean([
                np.mean(np.abs(truth[:, market] - prediction[:, market]))
                for market in range(4)]))
            payload = json.loads(metrics_path.read_text(encoding='utf-8'))
            saved_mae = float(payload['macro']['mae'])
            if not np.isclose(macro_mae, saved_mae, rtol=0, atol=2e-5):
                raise ValueError(
                    f'MAE mismatch at {run}: recomputed={macro_mae}, saved={saved_mae}')
            checks.append({'model': model, 'seed': seed, 'mae': saved_mae})

    cpu_checks = []
    for model in ('lear', 'naive_previous_day'):
        run = cpu_root / model
        arrays = np.load(run / 'predictions.npz')
        payload = json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
        if not (np.array_equal(arrays['y_true_raw'], reference_truth)
                and np.array_equal(arrays['delivery_date_ordinal'], reference_dates)):
            raise ValueError(f'CPU target/date mismatch for {model}')
        macro_mae = float(np.mean([
            np.mean(np.abs(arrays['y_true_raw'][:, market]
                           - arrays['y_pred_raw'][:, market]))
            for market in range(4)]))
        saved_mae = float(payload['macro']['mae'])
        if not np.isclose(macro_mae, saved_mae, rtol=0, atol=2e-5):
            raise ValueError(f'CPU MAE mismatch for {model}')
        cpu_checks.append({'model': model, 'mae': saved_mae})

    attention = gpu_root / 'cma_former' / 'seed_42' / 'attention.npz'
    if not attention.exists() or attention.stat().st_size > 50_000_000:
        raise ValueError('compact CMA seed-42 attention is missing or unexpectedly large')
    report = {
        'status': 'pass', 'neural_runs': len(checks),
        'models': len(MODELS), 'seeds_per_model': len(SEEDS),
        'test_delivery_days': int(reference_truth.shape[0]),
        'target_shape': list(reference_truth.shape),
        'cpu_baselines': cpu_checks,
        'compact_attention_bytes': attention.stat().st_size,
        'checks': checks,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in (
        'status', 'neural_runs', 'models', 'seeds_per_model',
        'test_delivery_days', 'target_shape', 'cpu_baselines')}, indent=2))


if __name__ == '__main__':
    main()
