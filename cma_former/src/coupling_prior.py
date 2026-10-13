"""Traceable market-coupling priors for cross-market attention."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    HAS_TORCH = False


INTERCONNECTED_PAIRS = (
    ('DE_LU', 'FR'),
    ('DE_LU', 'NL'),
    ('DE_LU', 'BE'),
    ('FR', 'BE'),
    ('NL', 'BE'),
)


def _topology_matrix(markets: List[str]) -> np.ndarray:
    matrix = np.zeros((len(markets), len(markets)), dtype=np.float32)
    np.fill_diagonal(matrix, 1.0)
    for left, right in INTERCONNECTED_PAIRS:
        if left in markets and right in markets:
            i, j = markets.index(left), markets.index(right)
            matrix[i, j] = 1.0
            matrix[j, i] = 1.0
    return matrix


def _training_ntc_matrix(markets: List[str], raw_root: Path,
                         train_end_year: int
                         ) -> Tuple[np.ndarray, List[Dict]]:
    sums = np.zeros((len(markets), len(markets)), dtype=float)
    counts = np.zeros_like(sums)
    sources: List[Dict] = []
    pattern = re.compile(r'^NTC_(.+)_to_(.+)_(\d{4})$')
    for path in sorted(raw_root.glob('NTC_*.csv')):
        match = pattern.match(path.stem)
        if not match:
            continue
        source, target, year_text = match.groups()
        year = int(year_text)
        if year > train_end_year or source not in markets or target not in markets:
            continue
        frame = pd.read_csv(path)
        numeric = frame.select_dtypes(include=[np.number])
        if numeric.empty:
            continue
        values = numeric.iloc[:, 0].to_numpy(dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        if not len(values):
            continue
        mean_mw = float(values.mean())
        i, j = markets.index(source), markets.index(target)
        sums[i, j] += mean_mw
        counts[i, j] += 1
        sources.append({
            'file': str(path.as_posix()),
            'source_market': source,
            'target_market': target,
            'year': year,
            'mean_mw': mean_mw,
            'n_observations': int(len(values)),
        })
    matrix = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    maximum = matrix.max()
    if maximum > 0:
        matrix /= maximum
    np.fill_diagonal(matrix, 1.0)
    return matrix.astype(np.float32), sources


def build_coupling_prior(markets: List[str],
                         mode: str = 'topology',
                         raw_root: Path = Path('data/raw/entsoe/ntc'),
                         train_end_year: int = 2022,
                         export_dir: Path = None,
                         seed: int = 42) -> Tuple[object, Dict]:
    """Build and optionally export a coupling prior with full provenance.

    Supported modes are ``topology``, ``ntc_train_only``, ``uniform`` and
    ``shuffled_topology``. No mode reads realized test-period physical flows.
    """
    sources: List[Dict] = []
    if mode == 'topology':
        matrix = _topology_matrix(markets)
        description = 'Binary direct-interconnection topology with self links.'
    elif mode == 'ntc_train_only':
        matrix, sources = _training_ntc_matrix(markets, Path(raw_root),
                                                train_end_year)
        if not sources:
            raise RuntimeError('No positive training-period NTC observations found')
        description = 'Directional NTC averages using training years only.'
    elif mode == 'uniform':
        matrix = np.ones((len(markets), len(markets)), dtype=np.float32)
        description = 'Uniform control prior.'
    elif mode == 'shuffled_topology':
        matrix = _topology_matrix(markets)
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(len(markets))
        matrix = matrix[permutation][:, permutation]
        description = 'Seeded market-label permutation of topology prior.'
    else:
        raise ValueError(f'unknown coupling prior mode: {mode}')

    metadata = {
        'mode': mode,
        'markets': markets,
        'train_end_year': int(train_end_year),
        'description': description,
        'uses_test_period_data': False,
        'uses_realized_physical_flow': False,
        'n_source_files': len(sources),
        'seed': int(seed),
    }
    if export_dir is not None:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(matrix, index=markets, columns=markets).to_csv(
            export_dir / 'coupling_prior_matrix.csv')
        pd.DataFrame(sources).to_csv(export_dir / 'coupling_prior_sources.csv',
                                     index=False)
        with open(export_dir / 'coupling_prior_metadata.json', 'w',
                  encoding='utf-8') as handle:
            json.dump(metadata, handle, indent=2)
    prior = torch.from_numpy(matrix) if HAS_TORCH else matrix
    return prior, metadata
