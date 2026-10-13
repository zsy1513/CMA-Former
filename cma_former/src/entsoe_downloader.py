"""
Batch downloader for ENTSO-E Transparency Platform.

Pulls all data needed for CMA-Former experiments:
  - Day-ahead prices (4 markets)
  - Total load (day-ahead forecast + actual)
  - Wind & solar generation forecast
  - Cross-border physical flows (8 directional pairs)

Usage:
  1. Register an account at https://transparency.entsoe.eu/
  2. Generate a personal Web API security token:
     My Account Settings -> Web API Security Token -> Generate
     (or email transparency@entsoe.eu requesting API access; takes 1-3 working days)
  3. Set the token as environment variable, or pass via --token:
        export ENTSOE_TOKEN='your-token-here'
  4. Run:
        python src/entsoe_downloader.py --start 2019-01-01 --end 2024-12-31

Notes:
  - The script chunks requests by year to avoid hitting the per-request size cap.
  - Resumes by skipping already-existing CSVs unless --force is passed.
  - Each CSV is in the same format produced by the manual web UI export, so
    `data_loader.py` reads them transparently.

Dependencies:
  pip install entsoe-py pandas tqdm

Author: CMA-Former Project
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('entsoe.downloader')


# ENTSO-E EIC area codes (must match entsoe-py's `Area` enum keys)
EIC_CODES = {
    'DE_LU': 'DE_LU',     # Germany-Luxembourg bidding zone (since Oct 2018)
    'FR':    'FR',
    'NL':    'NL',
    'BE':    'BE',
}

CROSS_BORDER_PAIRS: List[Tuple[str, str]] = [
    ('DE_LU', 'FR'), ('FR', 'DE_LU'),
    ('DE_LU', 'NL'), ('NL', 'DE_LU'),
    ('FR', 'BE'),    ('BE', 'FR'),
    ('NL', 'BE'),    ('BE', 'NL'),
]


# ---------------------------------------------------------------------
# Lazy import of entsoe-py
# ---------------------------------------------------------------------

def _get_client(token: str):
    try:
        from entsoe import EntsoePandasClient
    except ImportError:
        sys.stderr.write(
            'ERROR: entsoe-py not installed. Run: pip install entsoe-py\n')
        sys.exit(1)
    return EntsoePandasClient(api_key=token)


# ---------------------------------------------------------------------
# Per-data-type wrappers
# ---------------------------------------------------------------------

def _save_yearly(client_method, market: str, year: int, out_path: Path,
                 retries: int = 3, **kwargs):
    """
    Calls client_method(country_code=market, start=Yearstart, end=Yearend, **kwargs)
    and saves the result as CSV. Skips if file already exists.
    """
    if out_path.exists() and out_path.stat().st_size > 1024:
        logger.info(f'  [skip] {out_path.name}')
        return

    start = pd.Timestamp(f'{year}-01-01', tz='UTC')
    end = pd.Timestamp(f'{year+1}-01-01', tz='UTC')

    last_err = None
    for attempt in range(retries):
        try:
            df = client_method(country_code=market, start=start, end=end, **kwargs)
            if isinstance(df, pd.Series):
                df = df.to_frame(name='value')
            df.index.name = 'timestamp'
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path)
            logger.info(f'  [ok]   {out_path.name} ({len(df)} rows)')
            time.sleep(1.5)              # be kind to the server
            return
        except Exception as e:
            last_err = e
            wait = 5 * (attempt + 1)
            logger.warning(f'  [retry {attempt+1}/{retries}] {out_path.name}: '
                           f'{type(e).__name__}: {e}; waiting {wait}s')
            time.sleep(wait)
    logger.error(f'  [FAIL] {out_path.name}: {last_err}')


def _save_yearly_pair(client_method, m_from: str, m_to: str, year: int,
                      out_path: Path, retries: int = 3):
    """For cross-border methods that take country_code_from / country_code_to."""
    if out_path.exists() and out_path.stat().st_size > 1024:
        logger.info(f'  [skip] {out_path.name}')
        return

    start = pd.Timestamp(f'{year}-01-01', tz='UTC')
    end = pd.Timestamp(f'{year+1}-01-01', tz='UTC')

    last_err = None
    for attempt in range(retries):
        try:
            df = client_method(country_code_from=m_from,
                               country_code_to=m_to,
                               start=start, end=end)
            if isinstance(df, pd.Series):
                df = df.to_frame(name='value')
            df.index.name = 'timestamp'
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path)
            logger.info(f'  [ok]   {out_path.name} ({len(df)} rows)')
            time.sleep(1.5)
            return
        except Exception as e:
            last_err = e
            wait = 5 * (attempt + 1)
            logger.warning(f'  [retry {attempt+1}/{retries}] {out_path.name}: '
                           f'{type(e).__name__}: {e}; waiting {wait}s')
            time.sleep(wait)
    logger.error(f'  [FAIL] {out_path.name}: {last_err}')


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def download_all(token: str, raw_root: Path, years: List[int],
                 markets: List[str] = None,
                 skip_flows: bool = False):
    markets = markets or list(EIC_CODES.keys())
    client = _get_client(token)

    # 1. Day-ahead prices
    logger.info('=== Day-ahead prices ===')
    out_dir = raw_root / 'entsoe' / 'dayahead_prices'
    for m in markets:
        for y in years:
            _save_yearly(client.query_day_ahead_prices, m, y,
                         out_dir / f'{m}_DAM_{y}.csv')

    # 2. Total load (actual; we'll also pull day-ahead forecast)
    logger.info('=== Load: day-ahead forecast ===')
    out_dir = raw_root / 'entsoe' / 'load'
    for m in markets:
        for y in years:
            _save_yearly(client.query_load_forecast, m, y,
                         out_dir / f'{m}_LoadFc_{y}.csv')
    logger.info('=== Load: actual ===')
    for m in markets:
        for y in years:
            _save_yearly(client.query_load, m, y,
                         out_dir / f'{m}_LoadAct_{y}.csv')

    # 3. Wind & Solar day-ahead forecast
    logger.info('=== Wind & Solar day-ahead forecast ===')
    out_dir = raw_root / 'entsoe' / 'wind_solar'
    for m in markets:
        for y in years:
            _save_yearly(client.query_wind_and_solar_forecast, m, y,
                         out_dir / f'{m}_WS_{y}.csv',
                         psr_type=None)   # all RES psr types

    # 4. Cross-border physical flows
    if not skip_flows:
        logger.info('=== Cross-border physical flows ===')
        out_dir = raw_root / 'entsoe' / 'cross_border_flows'
        for m_from, m_to in CROSS_BORDER_PAIRS:
            if m_from not in markets or m_to not in markets:
                continue
            for y in years:
                _save_yearly_pair(client.query_crossborder_flows,
                                  m_from, m_to, y,
                                  out_dir / f'{m_from}_to_{m_to}_{y}.csv')

    # 5. Net Transfer Capacity (annual snapshot suffices for prior init)
    logger.info('=== Forecasted transfer capacity (NTC, day-ahead) ===')
    out_dir = raw_root / 'entsoe' / 'ntc'
    for m_from, m_to in CROSS_BORDER_PAIRS:
        if m_from not in markets or m_to not in markets:
            continue
        for y in years:
            _save_yearly_pair(client.query_net_transfer_capacity_dayahead,
                              m_from, m_to, y,
                              out_dir / f'NTC_{m_from}_to_{m_to}_{y}.csv')

    logger.info('All downloads complete.')


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='ENTSO-E batch downloader')
    ap.add_argument('--token', default=os.getenv('ENTSOE_TOKEN'),
                    help='ENTSO-E Web API security token. '
                         'Falls back to $ENTSOE_TOKEN env var.')
    ap.add_argument('--raw_root', default='./data/raw',
                    help='Output root directory.')
    ap.add_argument('--start', default='2019-01-01')
    ap.add_argument('--end', default='2024-12-31')
    ap.add_argument('--markets', nargs='+',
                    default=['DE_LU', 'FR', 'NL', 'BE'])
    ap.add_argument('--skip_flows', action='store_true',
                    help='Skip cross-border flows (slow).')
    ap.add_argument('--force', action='store_true',
                    help='Re-download even if file exists.')
    args = ap.parse_args()

    if not args.token:
        sys.stderr.write(
            'ERROR: provide --token or set ENTSOE_TOKEN env var.\n'
            'Get a token: https://transparency.entsoe.eu/ '
            '-> My Account -> Web API Security Token\n')
        sys.exit(1)

    if args.force:
        # Disable resume by flagging to the helpers via env var
        os.environ['ENTSOE_FORCE'] = '1'
        # Simple way: short-circuit existence checks
        global _save_yearly, _save_yearly_pair
        _orig_yr, _orig_pair = _save_yearly, _save_yearly_pair

        def _yr(*a, **kw):
            kw_path = a[3] if len(a) > 3 else kw.get('out_path')
            if kw_path and kw_path.exists():
                kw_path.unlink()
            return _orig_yr(*a, **kw)

        def _pr(*a, **kw):
            kw_path = a[4] if len(a) > 4 else kw.get('out_path')
            if kw_path and kw_path.exists():
                kw_path.unlink()
            return _orig_pair(*a, **kw)

        _save_yearly = _yr
        _save_yearly_pair = _pr

    start_y = pd.Timestamp(args.start).year
    end_y = pd.Timestamp(args.end).year
    years = list(range(start_y, end_y + 1))

    logger.info(f'Downloading {args.markets} for years {years}')
    logger.info(f'Output: {args.raw_root}')

    download_all(args.token, Path(args.raw_root), years,
                 markets=args.markets,
                 skip_flows=args.skip_flows)


if __name__ == '__main__':
    main()
