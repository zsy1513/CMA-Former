"""Operational day-ahead sampling for coupled European price forecasting.

The legacy experiments created one 24-step forecast every hour. This module
creates one sample per delivery day and keeps forecast issue time, delivery
date and data availability explicit. It is intentionally separate from the
legacy sliding-window dataset so previous results remain reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

    class Dataset:  # type: ignore
        pass


@dataclass(frozen=True)
class DayAheadSpec:
    lookback: int = 336
    horizon: int = 24
    market_timezone: str = 'Europe/Brussels'
    gate_closure_hour: int = 12
    flow_publication_lag_hours: int = 24
    financial_publication_lag_days: int = 1
    skip_dst_transition_days: bool = True


@dataclass(frozen=True)
class DayAheadSample:
    history_start: int
    history_end: int
    target_start: int
    target_end: int
    known_price_start: int
    known_price_end: int
    issue_time_utc_ns: int
    target_start_utc_ns: int
    delivery_date_ordinal: int


def _require_utc_hourly_index(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError('master data must have a DatetimeIndex')
    if df.index.tz is None:
        raise ValueError('master data index must be timezone-aware')
    if not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError('master data index must be sorted and unique')
    utc = df.index.tz_convert('UTC')
    gaps = utc.to_series().diff().dropna()
    if len(gaps) and not (gaps == pd.Timedelta(hours=1)).all():
        raise ValueError('master data must be a contiguous hourly UTC grid')


def apply_publication_lags(master: pd.DataFrame,
                           markets: List[str],
                           spec: DayAheadSpec) -> pd.DataFrame:
    """Create availability-safe historical covariates.

    Target-day load and renewable columns remain unchanged because they are
    ENTSO-E forecast products indexed by delivery time. Realized physical
    flows and daily financial closes are shifted before historical use.
    """
    out = master.copy()
    for market in markets:
        source = f'flow_net_{market}'
        target = f'flow_available_{market}'
        if source in out:
            out[target] = out[source].shift(spec.flow_publication_lag_hours)

    for source in ('ttf_gas', 'eua_carbon'):
        if source in out:
            out[f'{source}_available'] = out[source].shift(
                24 * spec.financial_publication_lag_days)
    return out


def compute_training_stats(master: pd.DataFrame,
                           train_end: str,
                           timezone: str) -> Dict[str, Dict[str, float]]:
    """Compute normalization statistics using delivery dates in training."""
    end_local = pd.Timestamp(train_end).tz_localize(timezone)
    end_exclusive = (end_local + pd.DateOffset(days=1)).tz_convert('UTC')
    train = master[master.index < end_exclusive]
    stats: Dict[str, Dict[str, float]] = {}
    for col in train.select_dtypes(include=[np.number]).columns:
        values = train[col].to_numpy(dtype=float)
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values))
        stats[col] = {'mean': mean, 'std': std if std >= 1e-8 else 1.0}
    return stats


class DayAheadMarketDataset(Dataset):
    """One sample per 24-hour delivery day under a fixed issue-time rule."""

    def __init__(self,
                 master: pd.DataFrame,
                 markets: List[str],
                 spec: DayAheadSpec,
                 stats: Dict[str, Dict[str, float]],
                 delivery_start: str,
                 delivery_end: str):
        _require_utc_hourly_index(master)
        self.master = master.copy()
        self.master.index = self.master.index.tz_convert('UTC')
        self.markets = list(markets)
        self.spec = spec
        self.stats = stats
        self.delivery_start = pd.Timestamp(delivery_start).date()
        self.delivery_end = pd.Timestamp(delivery_end).date()

        self.price_cols = [f'price_{m}' for m in markets]
        self.hist_exog_cols = [
            [f'load_{m}', f'res_{m}', f'flow_available_{m}']
            for m in markets
        ]
        self.future_exog_cols = [[f'load_{m}', f'res_{m}'] for m in markets]
        self.hist_global_cols = [
            c for c in ('ttf_gas_available', 'eua_carbon_available')
            if c in self.master
        ]

        required = list(self.price_cols)
        required += [c for cols in self.hist_exog_cols for c in cols]
        required += [c for cols in self.future_exog_cols for c in cols]
        missing = sorted(set(required) - set(self.master.columns))
        if missing:
            raise KeyError(f'missing day-ahead columns: {missing}')

        self.samples, self.skipped = self._build_samples()
        self._build_arrays()

    def _build_samples(self) -> Tuple[List[DayAheadSample], Dict[str, int]]:
        idx = self.master.index
        indexer = pd.Series(np.arange(len(idx)), index=idx)
        samples: List[DayAheadSample] = []
        skipped = {'dst': 0, 'missing_grid': 0, 'insufficient_history': 0,
                   'missing_values': 0}

        dates = pd.date_range(self.delivery_start, self.delivery_end, freq='D')
        for stamp in dates:
            target_date = stamp.date()
            target_local = pd.Timestamp(target_date).tz_localize(
                self.spec.market_timezone)
            next_local = (target_local + pd.DateOffset(days=1))
            target_start = target_local.tz_convert('UTC')
            target_stop = next_local.tz_convert('UTC')
            target_mask = (idx >= target_start) & (idx < target_stop)
            target_pos = np.flatnonzero(target_mask)

            known_start_local = target_local - pd.DateOffset(days=1)
            known_start = known_start_local.tz_convert('UTC')
            known_stop = target_start
            known_pos = np.flatnonzero((idx >= known_start) & (idx < known_stop))

            if (len(target_pos) != self.spec.horizon
                    or len(known_pos) != self.spec.horizon):
                if (self.spec.skip_dst_transition_days
                        and (len(target_pos) in (23, 25)
                             or len(known_pos) in (23, 25))):
                    skipped['dst'] += 1
                    continue
                skipped['missing_grid'] += 1
                continue
            if not np.all(np.diff(target_pos) == 1):
                skipped['missing_grid'] += 1
                continue
            if not np.all(np.diff(known_pos) == 1):
                skipped['missing_grid'] += 1
                continue

            issue_local = (target_local - pd.DateOffset(days=1)
                           + pd.Timedelta(hours=self.spec.gate_closure_hour))
            issue_utc = issue_local.tz_convert('UTC')
            if issue_utc not in indexer.index:
                skipped['missing_grid'] += 1
                continue
            history_end = int(indexer.loc[issue_utc])
            history_start = history_end - self.spec.lookback
            if history_start < 0:
                skipped['insufficient_history'] += 1
                continue

            target_first = int(target_pos[0])
            target_last = int(target_pos[-1] + 1)
            known_first = int(known_pos[0])
            known_last = int(known_pos[-1] + 1)
            required_cols = sorted(set(
                self.price_cols
                + [c for cols in self.hist_exog_cols for c in cols]
                + [c for cols in self.future_exog_cols for c in cols]
                + self.hist_global_cols
            ))
            history_ok = np.isfinite(
                self.master.iloc[history_start:history_end][required_cols]
                .to_numpy(dtype=float)
            ).all()
            target_ok = np.isfinite(
                self.master.iloc[target_first:target_last][
                    self.price_cols
                    + [c for cols in self.future_exog_cols for c in cols]
                ].to_numpy(dtype=float)
            ).all()
            known_ok = np.isfinite(
                self.master.iloc[known_first:known_last][self.price_cols]
                .to_numpy(dtype=float)
            ).all()
            if not history_ok or not target_ok or not known_ok:
                skipped['missing_values'] += 1
                continue

            samples.append(DayAheadSample(
                history_start=history_start,
                history_end=history_end,
                target_start=target_first,
                target_end=target_last,
                known_price_start=known_first,
                known_price_end=known_last,
                issue_time_utc_ns=int(issue_utc.value),
                target_start_utc_ns=int(target_start.value),
                delivery_date_ordinal=target_date.toordinal(),
            ))
        return samples, skipped

    def _z(self, values: np.ndarray, column: str) -> np.ndarray:
        stat = self.stats[column]
        return (values - stat['mean']) / stat['std']

    def _build_arrays(self) -> None:
        df = self.master
        self.price_raw = np.stack(
            [df[c].to_numpy(dtype=np.float32) for c in self.price_cols])
        self.price_norm = np.stack([
            self._z(df[c].to_numpy(dtype=np.float32), c)
            for c in self.price_cols
        ])

        hist = []
        future = []
        for hist_cols, future_cols in zip(self.hist_exog_cols,
                                          self.future_exog_cols):
            hist.append(np.stack([
                self._z(df[c].to_numpy(dtype=np.float32), c) for c in hist_cols
            ], axis=-1))
            future.append(np.stack([
                self._z(df[c].to_numpy(dtype=np.float32), c)
                for c in future_cols
            ], axis=-1))
        self.hist_exog = np.stack(hist, axis=0)
        self.future_exog = np.stack(future, axis=0)

        hist_global = []
        for col in self.hist_global_cols:
            hist_global.append(self._z(df[col].to_numpy(dtype=np.float32), col))
        utc = df.index
        hist_global.extend([
            np.sin(2 * np.pi * utc.hour / 24), np.cos(2 * np.pi * utc.hour / 24),
            np.sin(2 * np.pi * utc.dayofweek / 7),
            np.cos(2 * np.pi * utc.dayofweek / 7),
            np.sin(2 * np.pi * utc.month / 12),
            np.cos(2 * np.pi * utc.month / 12),
        ])
        self.hist_global = np.stack(hist_global, axis=-1).astype(np.float32)

        local = utc.tz_convert(self.spec.market_timezone)
        self.future_global = np.stack([
            np.sin(2 * np.pi * local.hour / 24),
            np.cos(2 * np.pi * local.hour / 24),
            np.sin(2 * np.pi * local.dayofweek / 7),
            np.cos(2 * np.pi * local.dayofweek / 7),
            np.sin(2 * np.pi * local.month / 12),
            np.cos(2 * np.pi * local.month / 12),
        ], axis=-1).astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        hs, he = sample.history_start, sample.history_end
        ts, te = sample.target_start, sample.target_end
        ks, ke = sample.known_price_start, sample.known_price_end
        y_raw = self.price_raw[:, ts:te]
        regimes = (y_raw < 0).astype(np.int64) * 2
        future_exog = np.concatenate([
            self.price_norm[:, ks:ke, None],
            self.future_exog[:, ts:te, :],
        ], axis=-1)
        item = {
            'x_price': self.price_norm[:, hs:he],
            'x_exog': self.hist_exog[:, hs:he, :],
            'x_glob': self.hist_global[hs:he, :],
            'x_future_exog': future_exog,
            'x_future_glob': self.future_global[ts:te, :],
            'y_price': self.price_norm[:, ts:te],
            'y_price_raw': y_raw,
            'y_regime': regimes,
            'issue_time_utc_ns': np.int64(sample.issue_time_utc_ns),
            'target_start_utc_ns': np.int64(sample.target_start_utc_ns),
            'delivery_date_ordinal': np.int64(sample.delivery_date_ordinal),
        }
        if HAS_TORCH:
            return {k: torch.from_numpy(v.copy()) if isinstance(v, np.ndarray)
                    else torch.tensor(v) for k, v in item.items()}
        return item


def make_day_ahead_dataloaders(master: pd.DataFrame,
                               markets: List[str],
                               spec: DayAheadSpec,
                               train_start: str,
                               train_end: str,
                               val_start: str,
                               val_end: str,
                               test_start: str,
                               test_end: str,
                               batch_size: int = 32,
                               num_workers: int = 0):
    if not HAS_TORCH:
        raise RuntimeError('PyTorch is required for DataLoader creation')
    safe_master = apply_publication_lags(master, markets, spec)
    stats = compute_training_stats(safe_master, train_end,
                                   spec.market_timezone)
    common = dict(master=safe_master, markets=markets, spec=spec, stats=stats)
    train = DayAheadMarketDataset(delivery_start=train_start,
                                  delivery_end=train_end, **common)
    val = DayAheadMarketDataset(delivery_start=val_start,
                                delivery_end=val_end, **common)
    test = DayAheadMarketDataset(delivery_start=test_start,
                                 delivery_end=test_end, **common)
    loaders = (
        DataLoader(train, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, drop_last=True),
        DataLoader(val, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers),
        DataLoader(test, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers),
    )
    return (*loaders, stats)
