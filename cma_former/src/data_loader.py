"""
Data loading and preprocessing for CMA-Former.

Handles:
  - ENTSO-E raw CSV (day-ahead price, load, wind/solar, cross-border flow) parsing
  - Merging with exogenous gas/carbon prices
  - Multi-market alignment (UTC harmonization)
  - Missing value handling (preserve spikes, interpolate small gaps)
  - Train/val/test split (walk-forward compatible)
  - Sliding window construction
  - PyTorch Dataset with regime label generation

Author: CMA-Former Project
"""

from __future__ import annotations

import os
import glob
import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Define stubs so the file still parses for static checking
    class Dataset:  # type: ignore
        pass

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('cma_former.data')


# =====================================================================
# Configuration
# =====================================================================

EU_MARKETS = ['DE_LU', 'FR', 'NL', 'BE']
RAW_ROOT = Path('./data/raw')
PROCESSED_ROOT = Path('./data/processed')


# =====================================================================
# Raw ENTSO-E parsers (robust to format variants)
# =====================================================================

def _parse_entsoe_csv(path: Path, value_col_keywords: List[str],
                      local_timezone: str = 'Europe/Brussels') -> pd.DataFrame:
    """
    Robust parser for ENTSO-E CSV files in two known shapes:

    Shape A -- entsoe-py output (preferred path used by entsoe_downloader.py):
        timestamp,value
        2019-01-01 00:00:00+00:00,28.32
        2019-01-01 01:00:00+00:00,10.07
        ...

    Shape B -- manual ENTSO-E web UI export:
        MTU (CET/CEST),Day-ahead Price [EUR/MWh],BZN
        "01.01.2019 00:00 - 01.01.2019 01:00 (CET/CEST)",28.32,DE-LU
        ...

    The function auto-detects the shape and returns:
        DataFrame with tz-aware DatetimeIndex (UTC) and a 'value' column.
    """
    # Try comma, then semicolon, then tab
    df = None
    for sep in [',', ';', '\t', '|']:
        try:
            df = pd.read_csv(path, sep=sep, engine='python')
            if df.shape[1] >= 2:
                break
        except Exception:
            continue
    if df is None or df.shape[1] < 2:
        # Maybe a single-column with timestamp as index?
        try:
            df = pd.read_csv(path, sep=',', engine='python')
        except Exception as e:
            raise RuntimeError(f'Could not parse {path}: {e}')

    # ----- Shape A detection (entsoe-py) -----
    cols_lower = [str(c).lower() for c in df.columns]
    if 'timestamp' in cols_lower:
        ts_col = df.columns[cols_lower.index('timestamp')]
        df['_ts'] = pd.to_datetime(df[ts_col], errors='coerce', utc=True)
        df = df.dropna(subset=['_ts'])
        # Pick the FIRST numeric column (entsoe-py's default name is 'value'
        # or the bidding zone code).
        value_col = None
        for c in df.columns:
            if c in (ts_col, '_ts'):
                continue
            try:
                pd.to_numeric(df[c].iloc[:5])
                value_col = c
                break
            except Exception:
                continue
        if value_col is None:
            raise ValueError(f'No numeric column in entsoe-py file {path}')
        df['value'] = pd.to_numeric(df[value_col], errors='coerce')
        out = df[['_ts', 'value']].rename(columns={'_ts': 'timestamp'})
        out = out.set_index('timestamp').sort_index()
        out = out[~out.index.duplicated(keep='first')]
        return out

    # ----- Shape B detection (web-UI export) -----
    time_col = None
    for c in df.columns:
        if any(k in str(c).lower() for k in ['mtu', 'time', 'date', 'period']):
            time_col = c
            break
    if time_col is None:
        raise ValueError(f'No time column found in {path}; columns={list(df.columns)}')

    def _start_ts(x: str) -> Optional[pd.Timestamp]:
        if not isinstance(x, str):
            return pd.NaT
        s = x.split('-')[0].strip()
        for fmt in ('%d.%m.%Y %H:%M', '%Y-%m-%d %H:%M', '%d/%m/%Y %H:%M',
                    '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d %H:%M:%S'):
            try:
                return pd.to_datetime(s, format=fmt)
            except Exception:
                continue
        return pd.to_datetime(s, errors='coerce')

    df['_ts'] = df[time_col].apply(_start_ts)
    df = df.dropna(subset=['_ts'])

    # Identify the value column by keyword match, else first numeric
    value_col = None
    for c in df.columns:
        if c == time_col or c == '_ts':
            continue
        if any(k.lower() in str(c).lower() for k in value_col_keywords):
            value_col = c
            break
    if value_col is None:
        # fall back: first numeric column
        for c in df.columns:
            if c in [time_col, '_ts']:
                continue
            try:
                pd.to_numeric(df[c].iloc[:5])
                value_col = c
                break
            except Exception:
                continue

    df['value'] = pd.to_numeric(df[value_col], errors='coerce')
    out = df[['_ts', 'value']].rename(columns={'_ts': 'timestamp'})
    out = out.set_index('timestamp').sort_index()

    # Manual ENTSO-E exports are expressed in CET/CEST. Localize in the
    # market timezone first and only then convert to UTC. Treating these
    # timestamps as UTC shifts every observation and silently corrupts DST
    # transition days.
    if out.index.tz is None:
        out.index = out.index.tz_localize(
            local_timezone, ambiguous='infer', nonexistent='NaT')
        out.index = out.index.tz_convert('UTC')

    out = out[~out.index.isna()]
    out = out[~out.index.duplicated(keep='first')]
    return out


def load_dam_prices(market: str, raw_dir: Path = None) -> pd.Series:
    """Load and concatenate day-ahead price CSVs for one market."""
    raw_dir = raw_dir or (RAW_ROOT / 'entsoe' / 'dayahead_prices')
    files = sorted(glob.glob(str(raw_dir / f'{market}*.csv')))
    if not files:
        raise FileNotFoundError(
            f'No DAM price files for {market} in {raw_dir}. '
            f'Expected pattern: {market}*.csv')
    pieces = [_parse_entsoe_csv(Path(f), ['day-ahead', 'price', 'eur'])
              for f in files]
    s = pd.concat(pieces).sort_index()['value']
    s = s[~s.index.duplicated(keep='first')]
    s.name = f'price_{market}'
    return s


def load_load(market: str, raw_dir: Path = None,
              forecast_only: bool = True) -> pd.Series:
    """Load forecasted or actual load without mixing the two products.

    The legacy implementation matched ``{market}*.csv`` and therefore read
    both ``LoadAct`` and ``LoadFc`` files. Duplicate removal then selected a
    product according to filename order. That made the feature provenance
    ambiguous and could silently replace the documented forecast with actual
    load. The Applied Energy protocol uses forecast files only.
    """
    raw_dir = raw_dir or (RAW_ROOT / 'entsoe' / 'load')
    product = 'LoadFc' if forecast_only else 'LoadAct'
    files = sorted(glob.glob(str(raw_dir / f'{market}_{product}_*.csv')))
    if not files:
        logger.warning(f'No {product} files for {market}; will be filled later.')
        return pd.Series(dtype=float, name=f'load_{market}')
    pieces = [_parse_entsoe_csv(Path(f), ['day-ahead', 'load', 'forecast'])
              for f in files]
    s = pd.concat(pieces).sort_index()['value']
    s = s[~s.index.duplicated(keep='first')]
    s.name = f'load_{market}'
    return s


def load_wind_solar(market: str, raw_dir: Path = None) -> pd.Series:
    """Load total day-ahead renewable forecast for a market.

    ENTSO-E files may contain separate Solar, Wind Offshore and Wind Onshore
    columns. They must be summed; selecting the first numeric column would
    reduce the feature to solar only for several markets.
    """
    raw_dir = raw_dir or (RAW_ROOT / 'entsoe' / 'wind_solar')
    files = sorted(glob.glob(str(raw_dir / f'{market}*.csv')))
    if not files:
        logger.warning(f'No wind/solar files for {market}.')
        return pd.Series(dtype=float, name=f'res_{market}')
    pieces = []
    for f in files:
        raw = pd.read_csv(f)
        cols_lower = {str(c).lower(): c for c in raw.columns}
        if 'timestamp' not in cols_lower:
            df = _parse_entsoe_csv(Path(f), ['wind', 'solar', 'forecast'])
            pieces.append(df['value'])
            continue

        ts_col = cols_lower['timestamp']
        ts = pd.to_datetime(raw[ts_col], errors='coerce', utc=True)
        generation_cols = [
            c for c in raw.columns
            if c != ts_col and any(k in str(c).lower()
                                   for k in ('wind', 'solar'))
        ]
        if not generation_cols:
            raise ValueError(f'No wind/solar columns in {f}')
        values = raw[generation_cols].apply(pd.to_numeric,
                                             errors='coerce').sum(axis=1,
                                                                  min_count=1)
        s_part = pd.Series(values.to_numpy(), index=ts).dropna()
        pieces.append(s_part)
    s = pd.concat(pieces).sort_index()
    s = s[~s.index.duplicated(keep='first')]
    s.name = f'res_{market}'
    return s


def load_cross_border_flows(raw_dir: Path = None) -> pd.DataFrame:
    """
    Load bidirectional flows for 4 markets (8 directional pairs).
    Returns DataFrame with columns flow_{from}_{to}.
    Net flow per market is computed downstream.
    """
    raw_dir = raw_dir or (RAW_ROOT / 'entsoe' / 'cross_border_flows')
    pairs = [('DE_LU', 'FR'), ('FR', 'DE_LU'),
             ('DE_LU', 'NL'), ('NL', 'DE_LU'),
             ('FR', 'BE'), ('BE', 'FR'),
             ('NL', 'BE'), ('BE', 'NL')]
    series = {}
    for a, b in pairs:
        files = sorted(glob.glob(str(raw_dir / f'{a}_to_{b}*.csv')))
        if not files:
            logger.warning(f'No flow data {a}->{b}; setting to zero.')
            series[f'flow_{a}_{b}'] = pd.Series(dtype=float)
            continue
        pieces = [_parse_entsoe_csv(Path(f), ['flow', 'mw'])
                  for f in files]
        s = pd.concat(pieces).sort_index()['value']
        s = s[~s.index.duplicated(keep='first')]
        series[f'flow_{a}_{b}'] = s

    df = pd.DataFrame(series).sort_index()
    return df


def load_exogenous(raw_dir: Path = None) -> pd.DataFrame:
    """
    Load TTF gas and EUA carbon price daily series. Forward-fill to hourly.
    """
    raw_dir = raw_dir or (RAW_ROOT / 'exogenous')
    out = {}

    # TTF
    ttf_path = raw_dir / 'ttf_gas.csv'
    if ttf_path.exists():
        ttf = pd.read_csv(ttf_path)
        # Normalize column names
        ttf.columns = [c.lower().strip() for c in ttf.columns]
        date_col = next((c for c in ttf.columns if 'date' in c), ttf.columns[0])
        price_col = next((c for c in ttf.columns
                          if any(k in c for k in ['price', 'close', 'settle'])),
                         ttf.columns[1])
        ttf['ts'] = pd.to_datetime(ttf[date_col], errors='coerce', utc=True)
        ttf = ttf.dropna(subset=['ts']).set_index('ts').sort_index()
        out['ttf_gas'] = pd.to_numeric(ttf[price_col], errors='coerce')
    else:
        logger.warning(f'TTF file not found at {ttf_path}')

    # EUA
    eua_path = raw_dir / 'eua_carbon.csv'
    if eua_path.exists():
        eua = pd.read_csv(eua_path)
        eua.columns = [c.lower().strip() for c in eua.columns]
        date_col = next((c for c in eua.columns if 'date' in c), eua.columns[0])
        price_col = next((c for c in eua.columns
                          if any(k in c for k in ['price', 'close', 'settle'])),
                         eua.columns[1])
        eua['ts'] = pd.to_datetime(eua[date_col], errors='coerce', utc=True)
        eua = eua.dropna(subset=['ts']).set_index('ts').sort_index()
        out['eua_carbon'] = pd.to_numeric(eua[price_col], errors='coerce')
    else:
        logger.warning(f'EUA file not found at {eua_path}')

    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out).sort_index()
    return df


# =====================================================================
# Master pipeline: build processed parquet
# =====================================================================

def build_processed(raw_root: Path = RAW_ROOT,
                    out_root: Path = PROCESSED_ROOT,
                    markets: List[str] = None,
                    start: str = '2019-01-01',
                    end: str = '2024-12-31') -> pd.DataFrame:
    """
    Construct the master multi-market hourly DataFrame and save to parquet.
    Columns:
      price_{market}, load_{market}, res_{market}    (per-market)
      flow_net_{market}                              (sum of inflows - outflows)
      ttf_gas, eua_carbon                            (global)
      hour, dow, month, is_holiday                   (calendar)
    """
    markets = markets or EU_MARKETS
    out_root.mkdir(parents=True, exist_ok=True)

    # Hourly UTC index. A date-only end value denotes the full final day,
    # rather than midnight at the start of that day.
    start_ts = pd.Timestamp(start, tz='UTC')
    end_ts = pd.Timestamp(end, tz='UTC')
    if isinstance(end, str) and len(end.strip()) == 10:
        end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
    idx = pd.date_range(start=start_ts, end=end_ts, freq='h')
    master = pd.DataFrame(index=idx)

    # Per-market series
    for m in markets:
        try:
            p = load_dam_prices(m, raw_root / 'entsoe' / 'dayahead_prices')
            master[f'price_{m}'] = p.reindex(idx)
        except FileNotFoundError as e:
            logger.error(str(e))
            master[f'price_{m}'] = np.nan

        ld = load_load(m, raw_root / 'entsoe' / 'load')
        master[f'load_{m}'] = ld.reindex(idx)

        rs = load_wind_solar(m, raw_root / 'entsoe' / 'wind_solar')
        master[f'res_{m}'] = rs.reindex(idx)

    # Cross-border flows -> net flow per market
    flows = load_cross_border_flows(raw_root / 'entsoe' / 'cross_border_flows')
    if not flows.empty:
        flows = flows.reindex(idx)
        for m in markets:
            inflow = sum(flows[c].fillna(0)
                         for c in flows.columns
                         if c.endswith(f'_{m}'))
            outflow = sum(flows[c].fillna(0)
                          for c in flows.columns
                          if c.startswith(f'flow_{m}_'))
            master[f'flow_net_{m}'] = inflow - outflow
    else:
        for m in markets:
            master[f'flow_net_{m}'] = 0.0

    # Exogenous global
    exog = load_exogenous(raw_root / 'exogenous')
    if not exog.empty:
        exog_h = exog.resample('h').ffill().reindex(idx, method='ffill')
        for c in exog_h.columns:
            master[c] = exog_h[c]

    # Calendar features
    master['hour'] = master.index.hour
    master['dow'] = master.index.dayofweek
    master['month'] = master.index.month

    # Holiday flag (best-effort; skip if `holidays` not installed)
    try:
        import holidays
        de_h = holidays.country_holidays('DE', years=range(2019, 2026))
        fr_h = holidays.country_holidays('FR', years=range(2019, 2026))
        nl_h = holidays.country_holidays('NL', years=range(2019, 2026))
        be_h = holidays.country_holidays('BE', years=range(2019, 2026))
        union = set(de_h) | set(fr_h) | set(nl_h) | set(be_h)
        master['is_holiday'] = master.index.normalize().date
        master['is_holiday'] = master['is_holiday'].apply(
            lambda d: 1 if d in union else 0).astype(np.int8)
    except ImportError:
        master['is_holiday'] = 0
        logger.warning('holidays package missing, is_holiday=0')

    # Missing-value treatment with availability-safe rules.
    # Prices: only short interpolation. Forecast covariates: short interpolation
    # followed by the same hour from the previous week. Financial closes:
    # forward-fill only, because the latest published close remains available.
    price_cols = [c for c in master.columns if c.startswith('price_')]
    forecast_cols = [c for c in master.columns
                     if c.startswith('load_') or c.startswith('res_')]
    financial_cols = [c for c in ('ttf_gas', 'eua_carbon') if c in master]

    for c in price_cols:
        missing = master[c].isna()
        master[c] = master[c].interpolate(method='time', limit=3,
                                          limit_direction='both')
        master[f'imputed_{c}'] = (missing & master[c].notna()).astype(np.int8)

    for c in forecast_cols:
        missing = master[c].isna()
        filled = master[c].interpolate(method='time', limit=3,
                                       limit_direction='both')
        # Repeat twice so a long gap can use the preceding two weeks without
        # borrowing information from after the gap.
        for _ in range(2):
            filled = filled.fillna(filled.shift(168))
        master[c] = filled
        master[f'imputed_{c}'] = (missing & filled.notna()).astype(np.int8)

    for c in financial_cols:
        missing = master[c].isna()
        master[c] = master[c].ffill()
        master[f'imputed_{c}'] = (missing & master[c].notna()).astype(np.int8)

    # Drop fully missing rows (rare; usually first/last few timestamps)
    n_before = len(master)
    master = master.dropna(subset=price_cols, how='all')
    n_after = len(master)
    if n_after < n_before:
        logger.info(f'Dropped {n_before - n_after} rows with no price data.')

    # Save
    out_path = out_root / 'master.parquet'
    try:
        master.to_parquet(out_path)
        logger.info(f'Saved master dataset to {out_path}: {master.shape}')
    except Exception as e:
        logger.warning(f'Parquet save failed ({e}); saving CSV instead.')
        master.to_csv(out_root / 'master.csv')

    return master


# =====================================================================
# Regime label generation
# =====================================================================

def compute_regime_labels(price: np.ndarray,
                          spike_sigma: float = 2.0,
                          negative_threshold: float = 0.0
                          ) -> np.ndarray:
    """
    Returns int array {0: normal, 1: spike, 2: negative} of same length as price.
    Statistics computed on the full series (in practice you'd compute on
    training set only and apply to all; the trainer handles that).
    """
    mu = np.nanmean(price)
    sigma = np.nanstd(price)
    labels = np.zeros_like(price, dtype=np.int64)
    labels[price > mu + spike_sigma * sigma] = 1   # spike
    labels[price < negative_threshold] = 2          # negative
    return labels


# =====================================================================
# Sliding window dataset
# =====================================================================

@dataclass
class WindowSpec:
    lookback: int = 336
    horizon: int = 24


class MultiMarketDataset(Dataset):
    """
    Yields per-sample dict:
        x_price : (M, L)       -- past prices (normalized)
        x_exog  : (M, L, d_x)  -- per-market exogenous (load, res, flow_net)
        x_glob  : (L, d_z)     -- global exogenous (ttf, eua, calendar)
        y_price : (M, H)       -- future prices (normalized)
        y_regime: (M, H)       -- future regime labels
        y_price_raw: (M, H)    -- future prices in original scale (for eval)
    """

    def __init__(self,
                 master: pd.DataFrame,
                 markets: List[str],
                 spec: WindowSpec,
                 stats: Optional[Dict[str, Dict[str, float]]] = None,
                 spike_sigma: float = 2.0,
                 neg_threshold: float = 0.0):
        self.master = master.copy()
        self.markets = markets
        self.spec = spec

        # Per-market columns
        self.price_cols = [f'price_{m}' for m in markets]
        self.exog_per_market = ['load', 'res', 'flow_net']
        self.glob_cols = [c for c in ['ttf_gas', 'eua_carbon',
                                      'hour', 'dow', 'month', 'is_holiday']
                          if c in master.columns]

        # Compute / receive standardization stats
        if stats is None:
            self.stats = self._compute_stats()
        else:
            self.stats = stats

        # Regime thresholds (mean+k*std on training stats)
        self.spike_sigma = spike_sigma
        self.neg_threshold = neg_threshold

        # Pre-extract numpy arrays for speed
        self._build_arrays()

        # Valid sample indices (need lookback + horizon contiguous)
        self.n = len(self.master) - spec.lookback - spec.horizon + 1
        if self.n <= 0:
            raise ValueError('Dataset too short for given lookback+horizon')

    # ---------------- helpers ----------------

    def _compute_stats(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        for c in self.master.columns:
            v = self.master[c].astype(float).values
            mu = float(np.nanmean(v))
            sd = float(np.nanstd(v))
            if sd < 1e-8:
                sd = 1.0
            stats[c] = {'mean': mu, 'std': sd}
        return stats

    def _norm(self, x: np.ndarray, col: str) -> np.ndarray:
        s = self.stats[col]
        return (x - s['mean']) / s['std']

    def _build_arrays(self):
        df = self.master
        # Prices: (M, T)
        price_arr = np.stack([df[f'price_{m}'].values for m in self.markets],
                             axis=0).astype(np.float32)
        self.price_raw = price_arr  # for regime + eval

        # Per-market exog: (M, T, 3)
        exog_arr = np.zeros((len(self.markets), len(df), 3), dtype=np.float32)
        for i, m in enumerate(self.markets):
            for j, ev in enumerate(self.exog_per_market):
                col = f'{ev}_{m}'
                if col in df.columns:
                    raw = df[col].values.astype(np.float32)
                    exog_arr[i, :, j] = self._norm(np.nan_to_num(raw), col)
        self.exog = exog_arr

        # Global: (T, d_z)
        glob_list = []
        for c in self.glob_cols:
            raw = df[c].values.astype(np.float32)
            if c in ('hour', 'dow', 'month', 'is_holiday'):
                # Cyclical encoding for hour/dow/month, leave is_holiday raw
                if c == 'hour':
                    glob_list.append(np.sin(2 * np.pi * raw / 24))
                    glob_list.append(np.cos(2 * np.pi * raw / 24))
                elif c == 'dow':
                    glob_list.append(np.sin(2 * np.pi * raw / 7))
                    glob_list.append(np.cos(2 * np.pi * raw / 7))
                elif c == 'month':
                    glob_list.append(np.sin(2 * np.pi * raw / 12))
                    glob_list.append(np.cos(2 * np.pi * raw / 12))
                else:
                    glob_list.append(raw)
            else:
                glob_list.append(self._norm(np.nan_to_num(raw), c))
        self.glob = np.stack(glob_list, axis=-1).astype(np.float32) \
                    if glob_list else np.zeros((len(df), 0), dtype=np.float32)

        # Normalized prices (for inputs; raw kept for outputs)
        norm_price = np.zeros_like(price_arr)
        for i, m in enumerate(self.markets):
            col = f'price_{m}'
            norm_price[i] = self._norm(np.nan_to_num(price_arr[i]), col)
        self.price_norm = norm_price

        # Regime labels per market (int8)
        regimes = np.zeros_like(price_arr, dtype=np.int64)
        for i, m in enumerate(self.markets):
            col = f'price_{m}'
            mu, sd = self.stats[col]['mean'], self.stats[col]['std']
            p = price_arr[i]
            r = np.zeros_like(p, dtype=np.int64)
            r[p > mu + self.spike_sigma * sd] = 1
            r[p < self.neg_threshold] = 2
            regimes[i] = r
        self.regimes = regimes

    # ---------------- API ----------------

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        L, H = self.spec.lookback, self.spec.horizon
        s = idx
        e = idx + L
        f = e + H

        if HAS_TORCH:
            to_t = lambda x: torch.from_numpy(x.copy())
        else:
            to_t = lambda x: x.copy()

        return {
            'x_price':    to_t(self.price_norm[:, s:e]),       # (M, L)
            'x_exog':     to_t(self.exog[:, s:e, :]),          # (M, L, 3)
            'x_glob':     to_t(self.glob[s:e, :]),             # (L, d_z)
            'y_price':    to_t(self.price_norm[:, e:f]),       # (M, H)
            'y_regime':   to_t(self.regimes[:, e:f]),          # (M, H)
            'y_price_raw': to_t(self.price_raw[:, e:f]),       # (M, H)
        }


# =====================================================================
# Splitting & DataLoader convenience
# =====================================================================

def split_by_date(master: pd.DataFrame,
                  train_end: str,
                  val_end: str
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-based train/val/test split."""
    idx = master.index
    train = master[idx <= pd.Timestamp(train_end, tz='UTC')]
    val   = master[(idx >  pd.Timestamp(train_end, tz='UTC')) &
                   (idx <= pd.Timestamp(val_end,   tz='UTC'))]
    test  = master[idx >  pd.Timestamp(val_end,   tz='UTC')]
    return train, val, test


def make_dataloaders(master: pd.DataFrame,
                     markets: List[str],
                     spec: WindowSpec,
                     train_end: str,
                     val_end: str,
                     batch_size: int = 32,
                     num_workers: int = 4,
                     spike_sigma: float = 2.0,
                     neg_threshold: float = 0.0
                     ) -> Tuple:
    if not HAS_TORCH:
        raise RuntimeError('PyTorch required for make_dataloaders')

    train_df, val_df, test_df = split_by_date(master, train_end, val_end)

    # Stats from training set ONLY (no leakage)
    train_ds = MultiMarketDataset(train_df, markets, spec,
                                  spike_sigma=spike_sigma,
                                  neg_threshold=neg_threshold)
    stats = train_ds.stats
    val_ds = MultiMarketDataset(val_df, markets, spec, stats=stats,
                                spike_sigma=spike_sigma,
                                neg_threshold=neg_threshold)
    test_ds = MultiMarketDataset(test_df, markets, spec, stats=stats,
                                 spike_sigma=spike_sigma,
                                 neg_threshold=neg_threshold)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl, test_dl, stats


# =====================================================================
# CLI
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build_processed', action='store_true',
                    help='Run the raw->processed pipeline.')
    ap.add_argument('--raw', default=str(RAW_ROOT))
    ap.add_argument('--out', default=str(PROCESSED_ROOT))
    ap.add_argument('--start', default='2019-01-01')
    ap.add_argument('--end', default='2024-12-31')
    ap.add_argument('--markets', nargs='+', default=EU_MARKETS)
    args = ap.parse_args()

    if args.build_processed:
        master = build_processed(Path(args.raw), Path(args.out),
                                 markets=args.markets,
                                 start=args.start, end=args.end)
        # Quick sanity
        print(master.describe().T.head(10))
        print(f'Saved {len(master):,} rows × {master.shape[1]} columns.')


if __name__ == '__main__':
    main()
