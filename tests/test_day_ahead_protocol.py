from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_load, load_wind_solar
from src.day_ahead_protocol import (
    DayAheadMarketDataset,
    DayAheadSpec,
    apply_publication_lags,
    compute_training_stats,
)


MARKETS = ['DE_LU', 'FR', 'NL', 'BE']


def _synthetic_master():
    idx = pd.date_range('2022-12-01', '2023-11-30 23:00', freq='h', tz='UTC')
    frame = pd.DataFrame(index=idx)
    x = np.arange(len(idx), dtype=float)
    for offset, market in enumerate(MARKETS):
        frame[f'price_{market}'] = 40 + offset + np.sin(x / 24)
        frame[f'load_{market}'] = 1000 + offset * 10 + np.cos(x / 24)
        frame[f'res_{market}'] = 300 + offset * 5 + np.sin(x / 12)
        frame[f'flow_net_{market}'] = offset + np.sin(x / 48)
    frame['ttf_gas'] = 50 + np.sin(x / (24 * 7))
    frame['eua_carbon'] = 80 + np.cos(x / (24 * 7))
    return frame


def test_one_non_overlapping_sample_per_regular_delivery_day():
    spec = DayAheadSpec(lookback=48)
    safe = apply_publication_lags(_synthetic_master(), MARKETS, spec)
    stats = compute_training_stats(safe, '2023-06-30', spec.market_timezone)
    ds = DayAheadMarketDataset(
        safe, MARKETS, spec, stats,
        delivery_start='2023-03-20', delivery_end='2023-04-05')

    assert ds.skipped['dst'] == 2
    assert len(ds) == 15
    target_ranges = [(s.target_start, s.target_end) for s in ds.samples]
    for left, right in zip(target_ranges, target_ranges[1:]):
        assert left[1] <= right[0]

    sample = ds.samples[0]
    issue = pd.Timestamp(sample.issue_time_utc_ns, tz='UTC')
    target = pd.Timestamp(sample.target_start_utc_ns, tz='UTC')
    assert issue < target
    assert sample.history_end - sample.history_start == spec.lookback
    assert sample.target_end - sample.target_start == spec.horizon


def test_publication_lags_are_explicit():
    frame = _synthetic_master().iloc[:100].copy()
    spec = DayAheadSpec(flow_publication_lag_hours=24,
                        financial_publication_lag_days=1)
    safe = apply_publication_lags(frame, MARKETS, spec)
    assert np.isnan(safe['flow_available_FR'].iloc[:24]).all()
    assert safe['flow_available_FR'].iloc[24] == frame['flow_net_FR'].iloc[0]
    assert safe['ttf_gas_available'].iloc[24] == frame['ttf_gas'].iloc[0]


def test_load_loader_does_not_mix_actual_and_forecast(tmp_path: Path):
    fc = pd.DataFrame({
        'timestamp': ['2024-01-01 01:00:00+01:00'],
        'Forecasted Load': [100.0],
    })
    actual = pd.DataFrame({
        'timestamp': ['2024-01-01 01:00:00+01:00'],
        'Actual Load': [999.0],
    })
    fc.to_csv(tmp_path / 'FR_LoadFc_2024.csv', index=False)
    actual.to_csv(tmp_path / 'FR_LoadAct_2024.csv', index=False)
    assert load_load('FR', tmp_path, forecast_only=True).iloc[0] == 100.0
    assert load_load('FR', tmp_path, forecast_only=False).iloc[0] == 999.0


def test_wind_solar_loader_sums_all_generation_columns(tmp_path: Path):
    raw = pd.DataFrame({
        'timestamp': ['2024-01-01 01:00:00+01:00'],
        'Solar': [10.0],
        'Wind Offshore': [20.0],
        'Wind Onshore': [30.0],
    })
    raw.to_csv(tmp_path / 'FR_WS_2024.csv', index=False)
    assert load_wind_solar('FR', tmp_path).iloc[0] == 60.0
