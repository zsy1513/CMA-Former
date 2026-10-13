"""Build and audit the leakage-controlled Applied Energy dataset."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.coupling_prior import build_coupling_prior
from src.data_loader import build_processed
from src.day_ahead_protocol import (
    DayAheadMarketDataset,
    DayAheadSpec,
    apply_publication_lags,
    compute_training_stats,
)


def _save_frame(frame: pd.DataFrame, path: Path) -> Path:
    try:
        frame.to_parquet(path)
        return path
    except Exception:
        fallback = path.with_suffix('.csv')
        frame.to_csv(fallback)
        return fallback


def _inventory(raw_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(raw_root.rglob('*.csv')):
        relative = path.relative_to(raw_root)
        rows.append({
            'category': relative.parts[0] if len(relative.parts) > 1 else 'root',
            'relative_path': relative.as_posix(),
            'size_bytes': path.stat().st_size,
        })
    return pd.DataFrame(rows)


def _provenance_table(markets):
    rows = []
    for market in markets:
        rows.extend([
            {'column': f'price_{market}', 'source': 'ENTSO-E day-ahead price',
             'role': 'history and target', 'availability_rule': 'delivery price'},
            {'column': f'load_{market}', 'source': 'ENTSO-E Forecasted Load',
             'role': 'history and target-day known covariate',
             'availability_rule': 'day-ahead forecast product; issuance version not archived'},
            {'column': f'res_{market}', 'source': 'ENTSO-E wind and solar forecast',
             'role': 'history and target-day known covariate',
             'availability_rule': 'day-ahead forecast product; sum of all wind/solar columns'},
            {'column': f'flow_available_{market}',
             'source': 'ENTSO-E realized cross-border physical flow',
             'role': 'historical only',
             'availability_rule': 'shifted by 24 hours'},
            {'column': f'known_day_price_{market}',
             'source': 'ENTSO-E day-ahead price for delivery day D-1',
             'role': 'known covariate for forecasting delivery day D',
             'availability_rule': 'all 24 D-1 prices cleared before D-1 issue time'},
        ])
    rows.extend([
        {'column': 'ttf_gas_available', 'source': 'TTF daily close',
         'role': 'historical only', 'availability_rule': 'shifted by 1 day'},
        {'column': 'eua_carbon_available', 'source': 'EUA daily close',
         'role': 'historical only', 'availability_rule': 'shifted by 1 day'},
    ])
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/applied_energy.yaml')
    parser.add_argument('--out', default='results/applied_energy_audit')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    data_cfg = config['data']
    prior_cfg = config['coupling_prior']
    markets = data_cfg['markets']
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    processed_dir = Path(data_cfg['processed_path']).parent

    master = build_processed(
        raw_root=Path(data_cfg['raw_root']), out_root=processed_dir,
        markets=markets, start=data_cfg['start_date'], end=data_cfg['end_date'])
    spec = DayAheadSpec(
        lookback=data_cfg['lookback'], horizon=data_cfg['horizon'],
        market_timezone=data_cfg['timezone'],
        gate_closure_hour=data_cfg['gate_closure_hour_local'],
        flow_publication_lag_hours=data_cfg['flow_publication_lag_hours'],
        financial_publication_lag_days=data_cfg['financial_publication_lag_days'],
        skip_dst_transition_days=data_cfg['skip_dst_transition_days'])
    safe = apply_publication_lags(master, markets, spec)
    safe_path = _save_frame(safe, processed_dir / 'master_available.parquet')
    stats = compute_training_stats(safe, data_cfg['train_end'],
                                   data_cfg['timezone'])

    split_specs = {
        'train': (data_cfg['train_start'], data_cfg['train_end']),
        'validation': (data_cfg['val_start'], data_cfg['val_end']),
        'test': (data_cfg['test_start'], data_cfg['test_end']),
    }
    protocol_rows = []
    split_summary = {}
    for split, (start, end) in split_specs.items():
        dataset = DayAheadMarketDataset(
            safe, markets, spec, stats,
            delivery_start=start, delivery_end=end)
        split_summary[split] = {
            'valid_delivery_days': len(dataset),
            'skipped': dataset.skipped,
        }
        for sample in dataset.samples:
            protocol_rows.append({
                'split': split,
                'delivery_date': date.fromordinal(
                    sample.delivery_date_ordinal).isoformat(),
                'issue_time_utc': pd.Timestamp(
                    sample.issue_time_utc_ns, tz='UTC').isoformat(),
                'target_start_utc': pd.Timestamp(
                    sample.target_start_utc_ns, tz='UTC').isoformat(),
            })

    prior, prior_metadata = build_coupling_prior(
        markets=markets, mode=prior_cfg['sensitivity_mode'],
        raw_root=Path(data_cfg['raw_root']) / 'entsoe' / 'ntc',
        train_end_year=prior_cfg['train_end_year'],
        export_dir=output / 'coupling_prior')

    expected_hours = len(pd.date_range(
        data_cfg['start_date'], f"{data_cfg['end_date']} 23:00",
        freq='h', tz='UTC'))
    summary = {
        'master_path': str(safe_path.as_posix()),
        'start_utc': master.index.min().isoformat(),
        'end_utc': master.index.max().isoformat(),
        'observed_hours': int(len(master)),
        'expected_hours': int(expected_hours),
        'complete_hourly_grid': bool(len(master) == expected_hours),
        'duplicate_timestamps': int(master.index.duplicated().sum()),
        'missing_values_by_column': {
            key: int(value) for key, value in safe.isna().sum().items()
        },
        'splits': split_summary,
        'prior': prior_metadata,
    }
    with open(output / 'protocol_audit.json', 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    pd.DataFrame(protocol_rows).to_csv(output / 'forecast_origins.csv',
                                       index=False)
    _inventory(Path(data_cfg['raw_root'])).to_csv(
        output / 'raw_file_inventory.csv', index=False)
    _provenance_table(markets).to_csv(output / 'variable_provenance.csv',
                                      index=False)

    report = f"""# Applied Energy protocol audit

- Hourly grid: {len(master):,} observed / {expected_hours:,} expected
- UTC range: {master.index.min()} to {master.index.max()}
- Forecast unit: one forecast per delivery day
- Market timezone: {spec.market_timezone}
- Local issue hour: {spec.gate_closure_hour}:00 on D-1
- DST transition days: excluded when the delivery day has 23 or 25 hours
- Realized physical flow: historical only, shifted {spec.flow_publication_lag_hours} h
- TTF/EUA closes: shifted {spec.financial_publication_lag_days} day
- Main coupling prior: {prior_metadata['mode']}; no realized test-period flow

## Valid delivery days

{json.dumps(split_summary, indent=2)}

## Remaining provenance limitation

ENTSO-E forecast CSVs do not store publication vintages or forecast
revisions. Vintage-level verification is therefore unavailable in the
downloaded files.
"""
    (output / 'protocol_audit.md').write_text(report, encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
