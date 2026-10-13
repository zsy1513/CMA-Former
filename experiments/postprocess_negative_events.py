"""Negative-price false-alarm and delivery-day event diagnostics."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _scores(truth, prediction):
    truth = np.asarray(truth, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    tp = int(np.sum(truth & prediction))
    fp = int(np.sum(~truth & prediction))
    fn = int(np.sum(truth & ~prediction))
    tn = int(np.sum(~truth & ~prediction))
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else np.nan)
    fpr = fp / (fp + tn) if fp + tn else np.nan
    return {
        'precision': precision, 'recall': recall, 'f1': f1,
        'false_positive_rate': fpr,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


def _runs(root):
    for path in Path(root).glob('*/seed_*/predictions.npz'):
        match = re.fullmatch(r'seed_(\d+)', path.parent.name)
        if match:
            yield path.parent.parent.name, int(match.group(1)), path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-root', default='results/applied_energy_final')
    parser.add_argument('--cpu-root', default='results/applied_energy_final_cpu')
    parser.add_argument('--out', default='results/applied_energy_final_analysis')
    parser.add_argument('--markets', nargs='+',
                        default=['DE_LU', 'FR', 'NL', 'BE'])
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    run_list = list(_runs(args.gpu_root))
    for model in ('lear', 'naive_previous_day'):
        path = Path(args.cpu_root) / model / 'predictions.npz'
        if path.exists():
            run_list.append((model, 'deterministic', path))
    for model, seed, path in run_list:
        arrays = np.load(path)
        truth = arrays['y_true_raw'] < 0
        prediction = arrays['y_pred_raw'] < 0
        for market_index, market in enumerate(args.markets):
            hourly = _scores(truth[:, market_index, :],
                             prediction[:, market_index, :])
            day_event = _scores(truth[:, market_index, :].any(axis=-1),
                                prediction[:, market_index, :].any(axis=-1))
            rows.append({
                'model': model, 'seed': seed, 'market': market,
                **{f'hourly_{key}': value for key, value in hourly.items()},
                **{f'day_event_{key}': value
                   for key, value in day_event.items()},
                'observed_negative_days': int(
                    truth[:, market_index, :].any(axis=-1).sum()),
                'observed_negative_hours': int(
                    truth[:, market_index, :].sum()),
            })
    detail = pd.DataFrame(rows)
    detail.to_csv(output / 'negative_price_event_metrics.csv', index=False)

    neural = detail[pd.to_numeric(detail['seed'], errors='coerce').notna()]
    numeric = [column for column in detail.columns
               if column not in ('model', 'seed', 'market')]
    summary = neural.groupby(['model', 'market'])[numeric].agg(
        ['mean', 'std', 'count'])
    summary.columns = ['_'.join(column) for column in summary.columns]
    summary.reset_index().to_csv(
        output / 'negative_price_event_summary.csv', index=False)
    print(f'Wrote {len(detail)} model-seed-market rows to {output}')


if __name__ == '__main__':
    main()
