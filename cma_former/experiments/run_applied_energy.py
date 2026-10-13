"""Unified multi-seed experiment runner for the day-ahead forecasting protocol."""

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

import torch
import torch.nn as nn

from src.baselines.applied_energy import (
    ConcatenatedTransformerBaseline, DLinearBaseline,
    IndependentMarketWrapper, NBEATSxBaseline, TemporalGATBaseline,
    topology_edge_index)
from src.baselines.deep_learning import TFTSimplified
from src.baselines.statistical import NaiveLastDay, NaiveSeasonal
from src.baselines.transformers import PatchTSTBaseline, iTransformerBaseline
from src.coupling_prior import build_coupling_prior
from src.day_ahead_protocol import DayAheadSpec, make_day_ahead_dataloaders
from src.models.cma_former import CMAFormer, CMAFormerConfig
from src.models.future_adapter import FutureCovariateAdapter
from src.utils.losses import QuantileLoss
from src.utils.metrics import (
    compute_all_metrics, daily_loss_panel, negative_metrics)
from src.utils.trainer import Trainer, TrainerConfig, seed_everything


class QuantileOnlyLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.loss = QuantileLoss(quantiles)

    def forward(self, predictions, target, gate, regime, gating_matrix):
        value = self.loss(predictions, target)
        zero = torch.zeros((), device=predictions.device)
        return {'total': value, 'quantile': value.detach(),
                'regime': zero, 'sparse': zero}


def build_model(name, config, dimensions, prior):
    model_cfg = config['model']
    data_cfg = config['data']
    markets = data_cfg['markets']
    quantiles = model_cfg['quantiles']
    common = dict(
        lookback=data_cfg['lookback'], horizon=data_cfg['horizon'],
        n_markets=len(markets), n_quantiles=len(quantiles))

    cma_names = {
        'cma_former', 'cma_no_cross', 'cma_no_prior',
        'cma_topology_prior', 'cma_fixed_prior', 'cma_shuffled_prior',
        'cma_late_future'}
    if name in cma_names:
        use_prior = name in {
            'cma_topology_prior', 'cma_fixed_prior', 'cma_shuffled_prior'}
        cross_layers = 0 if name == 'cma_no_cross' else model_cfg[
            'n_layers_cross']
        cma = CMAFormerConfig(
            n_markets=len(markets), lookback=data_cfg['lookback'],
            horizon=data_cfg['horizon'], d_exog=dimensions['d_exog'],
            d_glob=dimensions['d_glob'],
            d_future_exog=dimensions['d_future_exog'],
            d_future_glob=dimensions['d_future_glob'],
            d_model=model_cfg['d_model'], patch_size=model_cfg['patch_size'],
            patch_stride=model_cfg['patch_stride'],
            n_heads=model_cfg['n_heads'], d_ff=model_cfg['d_ff'],
            n_layers_intra=model_cfg['n_layers_intra'],
            n_layers_cross=cross_layers,
            dropout=model_cfg['dropout'], quantiles=quantiles,
            use_moe=model_cfg['use_moe'], use_revin=model_cfg['use_revin'],
            use_coupling_prior=use_prior,
            fixed_coupling_prior=name == 'cma_fixed_prior',
            use_future_covariates=(model_cfg['use_future_covariates']
                                   and name != 'cma_late_future'))
        base = CMAFormer(cma, coupling_prior=prior)
    elif name == 'patchtst':
        base = PatchTSTBaseline(
            **common, patch_size=model_cfg['patch_size'],
            patch_stride=model_cfg['patch_stride'],
            d_model=model_cfg['d_model'], n_heads=model_cfg['n_heads'],
            d_ff=model_cfg['d_ff'], n_layers=model_cfg['n_layers_intra'],
            dropout=model_cfg['dropout'], d_exog=dimensions['d_exog'],
            d_glob=dimensions['d_glob'])
    elif name == 'itransformer':
        base = iTransformerBaseline(
            **common, d_model=model_cfg['d_model'],
            n_heads=model_cfg['n_heads'], d_ff=model_cfg['d_ff'],
            n_layers=model_cfg['n_layers_intra'], dropout=model_cfg['dropout'],
            d_exog=dimensions['d_exog'], d_glob=dimensions['d_glob'])
    elif name == 'tft':
        base = TFTSimplified(
            d_exog=dimensions['d_exog'], d_glob=dimensions['d_glob'],
            horizon=data_cfg['horizon'], lookback=data_cfg['lookback'],
            d_hidden=64, n_heads=4, dropout=model_cfg['dropout'],
            n_quantiles=len(quantiles), n_markets=len(markets))
    elif name == 'dlinear':
        base = DLinearBaseline(**common)
    elif name == 'nbeatsx':
        base = NBEATSxBaseline(
            **common, d_exog=dimensions['d_exog'],
            d_glob=dimensions['d_glob'],
            d_future_exog=dimensions['d_future_exog'],
            d_future_glob=dimensions['d_future_glob'],
            hidden=256, n_blocks=4, n_layers=3,
            dropout=model_cfg['dropout'])
    elif name == 'temporal_gat':
        base = TemporalGATBaseline(
            n_markets=len(markets), d_exog=dimensions['d_exog'],
            d_glob=dimensions['d_glob'], horizon=data_cfg['horizon'],
            n_quantiles=len(quantiles), edge_index=topology_edge_index(markets),
            hidden=model_cfg['d_model'], heads=4,
            dropout=model_cfg['dropout'])
    elif name == 'single_market_patchtst':
        models = [PatchTSTBaseline(
            lookback=data_cfg['lookback'], horizon=data_cfg['horizon'],
            patch_size=model_cfg['patch_size'],
            patch_stride=model_cfg['patch_stride'],
            d_model=model_cfg['d_model'], n_heads=model_cfg['n_heads'],
            d_ff=model_cfg['d_ff'], n_layers=model_cfg['n_layers_intra'],
            dropout=model_cfg['dropout'], n_quantiles=len(quantiles),
            n_markets=1, d_exog=dimensions['d_exog'],
            d_glob=dimensions['d_glob']) for _ in markets]
        base = IndependentMarketWrapper(models)
    elif name == 'concatenated_transformer':
        base = ConcatenatedTransformerBaseline(
            n_markets=len(markets), d_exog=dimensions['d_exog'],
            d_glob=dimensions['d_glob'], horizon=data_cfg['horizon'],
            n_quantiles=len(quantiles), d_model=model_cfg['d_model'],
            n_heads=model_cfg['n_heads'], d_ff=model_cfg['d_ff'],
            n_layers=model_cfg['n_layers_intra'],
            dropout=model_cfg['dropout'])
    elif name == 'naive_last':
        return NaiveLastDay(
            n_quantiles=len(quantiles), horizon=data_cfg['horizon'],
            n_markets=len(markets))
    elif name == 'naive_seasonal':
        return NaiveSeasonal(
            horizon=data_cfg['horizon'], n_quantiles=len(quantiles))
    else:
        raise ValueError(f'unsupported model in unified runner: {name}')

    if model_cfg.get('future_covariate_integration') == 'shared_adapter':
        base = FutureCovariateAdapter(
            base, n_quantiles=len(quantiles),
            d_future_exog=dimensions['d_future_exog'],
            d_future_glob=dimensions['d_future_glob'])
    return base


def export_model_diagnostics(model, path):
    """Export trainable size and learned cross-market controls."""
    core = getattr(model, 'base_model', model)
    payload = {
        'trainable_parameters': int(sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad)),
        'cross_residual_scales': [],
        'coupling_prior_coefficients': [],
        'coupling_gates': [],
    }
    for layer in getattr(core, 'cross_layers', []):
        payload['cross_residual_scales'].append(
            float(layer.cross_scale.detach().cpu()))
        payload['coupling_prior_coefficients'].append(
            float(layer.cross.gate_beta.detach().cpu()))
        payload['coupling_gates'].append(
            layer.cross.gating_matrix().detach().cpu().tolist())
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def evaluate(prediction, truth, quantile_prediction, markets, quantiles):
    per_market = {}
    for market_index, market in enumerate(markets):
        y = truth[:, market_index, :].reshape(-1)
        median = prediction[:, market_index, :].reshape(-1)
        q = quantile_prediction[:, market_index, :, :].reshape(-1, len(quantiles))
        per_market[market] = compute_all_metrics(
            y, median, y_pred_quantiles=q, quantiles=quantiles)
    numeric_keys = sorted(set.intersection(*[
        set(metrics) for metrics in per_market.values()]))
    macro = {key: float(np.mean([per_market[m][key] for m in markets]))
             for key in numeric_keys}
    micro_negative = negative_metrics(truth.reshape(-1), prediction.reshape(-1))
    return {'per_market': per_market, 'macro': macro,
            'micro_negative': micro_negative}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/applied_energy.yaml')
    parser.add_argument('--out', default='results/applied_energy')
    parser.add_argument('--models', nargs='+', default=[
        'cma_former', 'patchtst', 'dlinear', 'nbeatsx', 'tft',
        'itransformer', 'temporal_gat'])
    parser.add_argument('--seeds', nargs='+', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    data_cfg = config['data']
    training_cfg = config['training']
    if args.epochs is not None:
        training_cfg['epochs'] = args.epochs
    seeds = args.seeds or training_cfg['seeds']
    output_root = Path(args.out)
    output_root.mkdir(parents=True, exist_ok=True)

    master = pd.read_parquet(data_cfg['processed_path'])
    spec = DayAheadSpec(
        lookback=data_cfg['lookback'], horizon=data_cfg['horizon'],
        market_timezone=data_cfg['timezone'],
        gate_closure_hour=data_cfg['gate_closure_hour_local'],
        flow_publication_lag_hours=data_cfg['flow_publication_lag_hours'],
        financial_publication_lag_days=data_cfg[
            'financial_publication_lag_days'],
        skip_dst_transition_days=data_cfg['skip_dst_transition_days'])
    train_dl, val_dl, test_dl, stats = make_day_ahead_dataloaders(
        master, data_cfg['markets'], spec,
        data_cfg['train_start'], data_cfg['train_end'],
        data_cfg['val_start'], data_cfg['val_end'],
        data_cfg['test_start'], data_cfg['test_end'],
        batch_size=training_cfg['batch_size'], num_workers=0)
    first = train_dl.dataset[0]
    dimensions = {
        'd_exog': first['x_exog'].shape[-1],
        'd_glob': first['x_glob'].shape[-1],
        'd_future_exog': first['x_future_exog'].shape[-1],
        'd_future_glob': first['x_future_glob'].shape[-1],
    }
    prior, _ = build_coupling_prior(
        data_cfg['markets'], mode=config['coupling_prior']['sensitivity_mode'],
        train_end_year=config['coupling_prior']['train_end_year'],
        export_dir=output_root / 'coupling_prior')

    rows = []
    for model_name in args.models:
        for seed in seeds:
            run_dir = output_root / model_name / f'seed_{seed}'
            run_dir.mkdir(parents=True, exist_ok=True)
            if (args.resume and (run_dir / 'predictions.npz').exists()
                    and (run_dir / 'metrics.json').exists()):
                print(f'Skipping completed run: {model_name} seed={seed}')
                continue
            # Seed before model construction so the directory seed controls
            # parameter initialization as well as shuffling and optimization.
            seed_everything(seed)
            model_prior = prior
            if model_name == 'cma_shuffled_prior':
                model_prior, _ = build_coupling_prior(
                    data_cfg['markets'], mode='shuffled_topology',
                    train_end_year=config['coupling_prior']['train_end_year'],
                    seed=seed)
            model = build_model(model_name, config, dimensions, model_prior)
            trainer_cfg = TrainerConfig(
                epochs=training_cfg['epochs'],
                batch_size=training_cfg['batch_size'],
                lr=training_cfg['learning_rate'],
                weight_decay=training_cfg['weight_decay'],
                warmup_epochs=training_cfg['warmup_epochs'],
                grad_clip=training_cfg['grad_clip'],
                early_stopping_patience=training_cfg['early_stopping_patience'],
                device=training_cfg['device'], seed=seed,
                save_dir=str(run_dir / 'checkpoints'),
                log_dir=str(run_dir / 'logs'),
                experiment_name='run')
            trainer = Trainer(
                model, QuantileOnlyLoss(config['model']['quantiles']),
                trainer_cfg, stats=stats)
            if trainer.has_trainable_params:
                fit_result = trainer.fit(train_dl, val_dl)
                best = trainer.save_dir / 'best.pt'
                if best.exists():
                    trainer.load_checkpoint(str(best))
                with open(run_dir / 'history.json', 'w', encoding='utf-8') as handle:
                    json.dump(fit_result['history'], handle, indent=2)
            export_model_diagnostics(model, run_dir / 'model_diagnostics.json')

            # One representative full-model seed is sufficient for the
            # attention diagnostics. All runs still save complete predictions.
            collect_attention = model_name == 'cma_former' and seed == 42
            result = trainer.predict(
                test_dl, collect_attn=collect_attention,
                markets=data_cfg['markets'])
            np.savez_compressed(
                run_dir / 'predictions.npz',
                **{key: value for key, value in result.items()
                   if key != 'attn_dump'})
            if 'attn_dump' in result:
                np.savez_compressed(run_dir / 'attention.npz',
                                    **result['attn_dump'])
            metrics = evaluate(
                result['y_pred_raw'], result['y_true_raw'],
                result['y_pred_quantiles_raw'], data_cfg['markets'],
                config['model']['quantiles'])
            with open(run_dir / 'metrics.json', 'w', encoding='utf-8') as handle:
                json.dump(metrics, handle, indent=2)

            daily = daily_loss_panel(result['y_true_raw'], result['y_pred_raw'])
            daily_frame = pd.DataFrame(daily, columns=data_cfg['markets'])
            daily_frame['delivery_date_ordinal'] = result[
                'delivery_date_ordinal']
            daily_frame.to_csv(run_dir / 'daily_mae.csv', index=False)
            rows.append({'model': model_name, 'seed': seed,
                         **metrics['macro']})

    pd.DataFrame(rows).to_csv(output_root / 'seed_metrics.csv', index=False)


if __name__ == '__main__':
    main()
