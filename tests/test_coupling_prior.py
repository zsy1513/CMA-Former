import numpy as np
import pandas as pd

from src.coupling_prior import build_coupling_prior


MARKETS = ['DE_LU', 'FR', 'NL', 'BE']


def test_topology_prior_has_only_documented_direct_links(tmp_path):
    prior, metadata = build_coupling_prior(
        MARKETS, mode='topology', export_dir=tmp_path)
    matrix = prior.numpy() if hasattr(prior, 'numpy') else np.asarray(prior)
    assert np.all(np.diag(matrix) == 1)
    assert matrix[0, 1] == matrix[1, 0] == 1
    assert matrix[0, 3] == matrix[3, 0] == 1
    assert matrix[1, 2] == matrix[2, 1] == 0
    assert metadata['uses_test_period_data'] is False
    assert (tmp_path / 'coupling_prior_matrix.csv').exists()


def test_ntc_prior_uses_training_years_only(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    for year, value in [(2021, 1000.0), (2022, 2000.0), (2024, 9000.0)]:
        pd.DataFrame({'timestamp': [f'{year}-01-01'], 'value': [value]}).to_csv(
            raw / f'NTC_DE_LU_to_NL_{year}.csv', index=False)
    prior, metadata = build_coupling_prior(
        MARKETS, mode='ntc_train_only', raw_root=raw,
        train_end_year=2022, export_dir=tmp_path / 'audit')
    assert float(prior[0, 2]) == 1.0
    assert metadata['n_source_files'] == 2
