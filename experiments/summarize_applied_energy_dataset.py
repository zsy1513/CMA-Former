"""Create traceable dataset and operational-sample summary tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.day_ahead_protocol import (
    DayAheadMarketDataset, DayAheadSpec, apply_publication_lags,
    compute_training_stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/applied_energy.yaml')
    parser.add_argument('--out', default='results/applied_energy_final_analysis')
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    data = config['data']
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    master = pd.read_parquet(data['processed_path'])
    split_bounds = {
        'train': (data['train_start'], data['train_end']),
        'validation': (data['val_start'], data['val_end']),
        'test': (data['test_start'], data['test_end']),
    }
    price_rows = []
    for split, (start, end) in split_bounds.items():
        local = master.index.tz_convert(data['timezone'])
        mask = ((local.date >= pd.Timestamp(start).date())
                & (local.date <= pd.Timestamp(end).date()))
        for market in data['markets']:
            values = master.loc[mask, f'price_{market}'].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            price_rows.append({
                'split': split, 'market': market,
                'hourly_observations': int(len(finite)),
                'mean_eur_per_mwh': float(np.mean(finite)),
                'std_eur_per_mwh': float(np.std(finite, ddof=1)),
                'minimum_eur_per_mwh': float(np.min(finite)),
                'maximum_eur_per_mwh': float(np.max(finite)),
                'negative_price_hours': int(np.sum(finite < 0)),
                'negative_price_share': float(np.mean(finite < 0)),
            })
    pd.DataFrame(price_rows).to_csv(
        output / 'dataset_price_summary.csv', index=False)

    spec = DayAheadSpec(
        lookback=data['lookback'], horizon=data['horizon'],
        market_timezone=data['timezone'],
        gate_closure_hour=data['gate_closure_hour_local'],
        flow_publication_lag_hours=data['flow_publication_lag_hours'],
        financial_publication_lag_days=data['financial_publication_lag_days'],
        skip_dst_transition_days=data['skip_dst_transition_days'])
    safe = apply_publication_lags(master, data['markets'], spec)
    stats = compute_training_stats(safe, data['train_end'], data['timezone'])
    sample_rows = []
    for split, (start, end) in split_bounds.items():
        dataset = DayAheadMarketDataset(
            safe, data['markets'], spec, stats, start, end)
        sample_rows.append({
            'split': split, 'valid_delivery_days': len(dataset),
            **{f'skipped_{key}': value
               for key, value in dataset.skipped.items()},
        })
    pd.DataFrame(sample_rows).to_csv(
        output / 'operational_sample_summary.csv', index=False)

    imputation = {}
    for column in master.columns:
        if column.startswith('imputed_'):
            imputation[column] = int(master[column].fillna(0).sum())
    with open(output / 'dataset_integrity_summary.json', 'w',
              encoding='utf-8') as handle:
        json.dump({
            'rows': int(len(master)),
            'start_utc': str(master.index.min()),
            'end_utc': str(master.index.max()),
            'duplicate_timestamps': int(master.index.duplicated().sum()),
            'imputed_values': imputation,
        }, handle, indent=2)
    print(f'Wrote dataset summaries to {output}')


if __name__ == '__main__':
    main()
