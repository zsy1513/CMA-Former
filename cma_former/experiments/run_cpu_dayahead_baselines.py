"""CPU baselines for the operational day-ahead protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Lasso
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.day_ahead_protocol import (
    DayAheadMarketDataset, DayAheadSpec, apply_publication_lags,
    compute_training_stats)
from src.utils.metrics import compute_all_metrics, daily_loss_panel


def _numpy(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _extract(dataset):
    features, targets, known_day, metadata = [], [], [], []
    selected_lags = [1, 2, 24, 25, 48, 49, 168, 169, 336]
    for index in range(len(dataset)):
        item = dataset[index]
        price = _numpy(item['x_price'])
        hist_exog = _numpy(item['x_exog'])
        hist_global = _numpy(item['x_glob'])
        future_exog = _numpy(item['x_future_exog'])
        future_global = _numpy(item['x_future_glob'])
        parts = []
        for lag in selected_lags:
            parts.append(price[:, -lag])
        for lag in (1, 24, 168):
            parts.append(hist_exog[:, -lag, :].reshape(-1))
        parts.extend([
            hist_global[-1].reshape(-1),
            future_exog.reshape(-1),
            future_global.reshape(-1),
        ])
        features.append(np.concatenate(parts))
        targets.append(_numpy(item['y_price_raw']).reshape(-1))
        known_day.append(future_exog[..., 0])
        metadata.append(int(_numpy(item['delivery_date_ordinal'])))
    return (np.asarray(features, dtype=np.float32),
            np.asarray(targets, dtype=np.float32),
            np.asarray(known_day, dtype=np.float32),
            np.asarray(metadata, dtype=np.int64))


def _metrics(y_true, y_pred, markets):
    days = len(y_true)
    truth = y_true.reshape(days, len(markets), 24)
    prediction = y_pred.reshape(days, len(markets), 24)
    per_market = {}
    for index, market in enumerate(markets):
        per_market[market] = compute_all_metrics(
            truth[:, index, :].reshape(-1),
            prediction[:, index, :].reshape(-1))
    macro = {
        key: float(np.mean([per_market[m][key] for m in markets]))
        for key in per_market[markets[0]]
    }
    return truth, prediction, {'per_market': per_market, 'macro': macro}


def _save_result(output, name, truth, prediction, dates, metrics):
    run = output / name
    run.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run / 'predictions.npz', y_true_raw=truth, y_pred_raw=prediction,
        delivery_date_ordinal=dates)
    with open(run / 'metrics.json', 'w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)
    daily = daily_loss_panel(truth, prediction)
    frame = pd.DataFrame(daily, columns=metrics['per_market'].keys())
    frame['delivery_date_ordinal'] = dates
    frame.to_csv(run / 'daily_mae.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/applied_energy.yaml')
    parser.add_argument('--out', default='results/applied_energy_cpu')
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    data = config['data']
    markets = data['markets']
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    master = pd.read_parquet(data['processed_path'])
    spec = DayAheadSpec(
        lookback=data['lookback'], horizon=data['horizon'],
        market_timezone=data['timezone'],
        gate_closure_hour=data['gate_closure_hour_local'],
        flow_publication_lag_hours=data['flow_publication_lag_hours'],
        financial_publication_lag_days=data['financial_publication_lag_days'])
    safe = apply_publication_lags(master, markets, spec)
    stats = compute_training_stats(safe, data['train_end'], data['timezone'])
    common = dict(master=safe, markets=markets, spec=spec, stats=stats)
    train = DayAheadMarketDataset(
        delivery_start=data['train_start'], delivery_end=data['train_end'],
        **common)
    validation = DayAheadMarketDataset(
        delivery_start=data['val_start'], delivery_end=data['val_end'], **common)
    test = DayAheadMarketDataset(
        delivery_start=data['test_start'], delivery_end=data['test_end'], **common)
    x_train, y_train, _, _ = _extract(train)
    x_val, y_val, _, _ = _extract(validation)
    x_test, y_test, known_test, dates_test = _extract(test)

    x_scaler = StandardScaler().fit(x_train)
    x_train_scaled = x_scaler.transform(x_train)
    x_val_scaled = x_scaler.transform(x_val)
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True)
    y_std[y_std < 1e-8] = 1.0
    y_train_scaled = (y_train - y_mean) / y_std

    candidate_alphas = [0.0001, 0.001, 0.01, 0.1]
    validation_rows = []
    for alpha in candidate_alphas:
        estimator = MultiOutputRegressor(
            Lasso(alpha=alpha, max_iter=20000, tol=5e-4,
                  selection='cyclic'), n_jobs=4)
        estimator.fit(x_train_scaled, y_train_scaled)
        prediction = estimator.predict(x_val_scaled) * y_std + y_mean
        validation_rows.append({
            'alpha': alpha,
            'validation_mae': float(np.mean(np.abs(y_val - prediction))),
        })
    validation_frame = pd.DataFrame(validation_rows)
    validation_frame.to_csv(output / 'lear_alpha_selection.csv', index=False)
    best_alpha = float(validation_frame.loc[
        validation_frame['validation_mae'].idxmin(), 'alpha'])

    x_fit = np.concatenate([x_train, x_val])
    y_fit = np.concatenate([y_train, y_val])
    final_scaler = StandardScaler().fit(x_fit)
    y_fit_mean = y_fit.mean(axis=0, keepdims=True)
    y_fit_std = y_fit.std(axis=0, keepdims=True)
    y_fit_std[y_fit_std < 1e-8] = 1.0
    lear = MultiOutputRegressor(
        Lasso(alpha=best_alpha, max_iter=100000, tol=5e-5,
              selection='cyclic'), n_jobs=4)
    lear.fit(final_scaler.transform(x_fit),
             (y_fit - y_fit_mean) / y_fit_std)
    lear_prediction = (lear.predict(final_scaler.transform(x_test))
                       * y_fit_std + y_fit_mean)
    truth, prediction, metrics = _metrics(y_test, lear_prediction, markets)
    metrics['selected_alpha'] = best_alpha
    _save_result(output, 'lear', truth, prediction, dates_test, metrics)

    # Previous delivery day's published prices are the first future-exog
    # channel. Convert them from train-set z-scores to EUR/MWh.
    naive = np.empty_like(known_test)
    for market_index, market in enumerate(markets):
        stat = stats[f'price_{market}']
        naive[:, market_index, :] = (
            known_test[:, market_index, :] * stat['std'] + stat['mean'])
    truth, prediction, metrics = _metrics(
        y_test, naive.reshape(len(test), -1), markets)
    _save_result(output, 'naive_previous_day', truth, prediction,
                 dates_test, metrics)
    print(json.dumps({
        'selected_alpha': best_alpha,
        'lear_macro': json.load(open(output / 'lear' / 'metrics.json'))['macro'],
        'naive_macro': json.load(open(
            output / 'naive_previous_day' / 'metrics.json'))['macro'],
    }, indent=2))


if __name__ == '__main__':
    main()
