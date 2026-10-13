"""
Training loop for CMA-Former and baseline models.

Features:
  - AdamW optimizer with cosine schedule + linear warmup
  - Gradient clipping
  - Early stopping on validation loss
  - Checkpoint of best model
  - TensorBoard / file logging
  - Reproducible seeding
  - Walk-forward (re-training periodically) optional

Author: CMA-Former Project
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger('cma_former.trainer')


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------

@dataclass
class TrainerConfig:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    early_stopping_patience: int = 15
    device: str = 'cuda'
    log_every: int = 50
    save_dir: str = './results/checkpoints'
    log_dir: str = './results/logs'
    experiment_name: str = 'cma_former'
    seed: int = 42


# ---------------------------------------------------------------------
# Helper: cosine LR with warmup
# ---------------------------------------------------------------------

def cosine_warmup_schedule(epoch: int, warmup: int, total: int,
                           lr_max: float, lr_min: float = 1e-6) -> float:
    if epoch < warmup:
        return lr_max * (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------

class Trainer:
    """
    Generic trainer compatible with CMA-Former and baseline models.

    The model is expected to:
      - accept inputs (x_price, x_exog, x_glob)
      - return (preds, gate_probs, experts) for CMA-Former, or
               (preds,) / (preds, ...) for baselines (gate_probs=None ok)

    The loss callable is given full freedom: Trainer just calls
        out = loss_fn(preds, target, gate_probs, regime_labels, gating_matrix)
    and expects a dict with key 'total'.
    """

    def __init__(self,
                 model: 'nn.Module',
                 loss_fn: Callable,
                 cfg: TrainerConfig,
                 quantile_idx_median: int = 1,
                 stats: Dict = None):
        if not HAS_TORCH:
            raise RuntimeError('PyTorch required to instantiate Trainer.')
        self.model = model
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.quantile_idx_median = quantile_idx_median
        self.stats = stats                                 # for de-normalization

        self.device = torch.device(cfg.device if torch.cuda.is_available()
                                   else 'cpu')
        self.model.to(self.device)
        # Move loss_fn buffers to device too (loss may be an nn.Module with
        # registered buffers like quantile vectors).
        if isinstance(self.loss_fn, nn.Module):
            self.loss_fn.to(self.device)

        # Optimizer -- skip construction if the model has no trainable
        # parameters (e.g. naive baselines, zero-shot foundation models).
        # The `fit()` method will detect this and short-circuit training.
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.has_trainable_params = len(trainable) > 0
        if self.has_trainable_params:
            self.optimizer = torch.optim.AdamW(
                trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
        else:
            self.optimizer = None
            logger.info('Model has no trainable parameters; '
                        'training will be skipped.')

        # Output dirs
        self.save_dir = Path(cfg.save_dir) / cfg.experiment_name
        self.log_dir = Path(cfg.log_dir) / cfg.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # File log
        fh = logging.FileHandler(self.log_dir / 'train.log', mode='w')
        fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)

        # State
        self.best_val = float('inf')
        self.epochs_no_improve = 0
        self.history: List[Dict] = []

    # ----- iteration -----

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _model_forward(self, batch):
        kwargs = {}
        forward_vars = self.model.forward.__code__.co_varnames
        if 'x_future_exog' in forward_vars and 'x_future_exog' in batch:
            kwargs['x_future_exog'] = batch['x_future_exog']
        if 'x_future_glob' in forward_vars and 'x_future_glob' in batch:
            kwargs['x_future_glob'] = batch['x_future_glob']
        out = self.model(batch['x_price'], batch['x_exog'], batch['x_glob'],
                         **kwargs)
        if isinstance(out, tuple) and len(out) == 3:
            preds, gate, _ = out
        elif isinstance(out, tuple) and len(out) == 2:
            preds, gate = out
        else:
            preds, gate = out, None
        return preds, gate

    def _compute_loss(self, preds, gate, batch):
        # If the model uses RevIN, the predictions are in the per-window
        # normalised space. We must normalise y_price_raw with the SAME
        # window stats so that the loss is comparable.
        # Otherwise we use the dataset-normalised y_price (z-score with
        # train-set stats).
        model_does_revin = bool(getattr(self.model, 'use_revin', False))
        if model_does_revin:
            y_raw = batch['y_price_raw']                  # (B, M, H) raw EUR/MWh
            target = self.model.normalize_target(y_raw)   # (B, M, H) per-window normed
        else:
            target = batch['y_price']                     # (B, M, H) train-norm
        regime = batch['y_regime']                        # (B, M, H)

        gating_matrix = None
        if hasattr(self.model, 'get_coupling_gates') and gate is not None:
            gating_matrix = self.model.get_coupling_gates()

        out = self.loss_fn(preds, target, gate, regime, gating_matrix)
        return out

    # ----- one epoch -----

    def train_epoch(self, dl: 'DataLoader', epoch: int) -> Dict[str, float]:
        self.model.train()
        meters = {'total': 0.0, 'quantile': 0.0, 'regime': 0.0, 'sparse': 0.0}
        n_batches = 0
        t0 = time.time()

        # Set LR
        lr = cosine_warmup_schedule(epoch, self.cfg.warmup_epochs,
                                    self.cfg.epochs, self.cfg.lr)
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

        for i, batch in enumerate(dl):
            batch = self._move_batch(batch)
            preds, gate = self._model_forward(batch)
            losses = self._compute_loss(preds, gate, batch)

            self.optimizer.zero_grad(set_to_none=True)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.cfg.grad_clip)
            self.optimizer.step()

            for k in meters:
                if k in losses:
                    v = losses[k]
                    meters[k] += float(v.detach().cpu()) if torch.is_tensor(v) else float(v)
            n_batches += 1

            if (i + 1) % self.cfg.log_every == 0:
                logger.info(f'  epoch {epoch} batch {i+1}/{len(dl)} '
                            f'loss={meters["total"]/n_batches:.4f} lr={lr:.2e}')

        for k in meters:
            meters[k] /= max(1, n_batches)
        meters['lr'] = lr
        meters['time_s'] = time.time() - t0
        return meters

    # ----- validation -----

    @torch.no_grad()
    def validate(self, dl: 'DataLoader') -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        n = 0
        for batch in dl:
            batch = self._move_batch(batch)
            preds, gate = self._model_forward(batch)
            losses = self._compute_loss(preds, gate, batch)
            total_loss += float(losses['total'].detach().cpu())
            n += 1
        return {'val_loss': total_loss / max(1, n)}

    # ----- inference: collect raw predictions -----

    @torch.no_grad()
    def predict(self, dl: 'DataLoader',
                collect_attn: bool = False,
                markets: List[str] = None
                ) -> Dict[str, np.ndarray]:
        """
        Returns dict with:
          y_pred_norm  : (N, M, H, Q)  normalized quantile predictions
          y_pred_quantiles_raw : (N, M, H, Q) raw quantile predictions
          y_pred_raw   : (N, M, H)     median predictions in original scale
          y_true_raw   : (N, M, H)     true prices in original scale
          gate_probs   : (N, M, 3)     regime gate probabilities
          attn_dump    : dict of attention tensors (optional, only last batch
                         to limit memory; full collection is in run_msim_analysis)
        """
        self.model.eval()
        all_pred, all_pred_raw, all_pred_qraw = [], [], []
        all_true_raw, all_gate = [], []
        metadata = {key: [] for key in (
            'issue_time_utc_ns', 'target_start_utc_ns',
            'delivery_date_ordinal')}
        attn_dump = {} if collect_attn else None

        # If model accepts attn_dump kwarg, pass it
        accepts_dump = hasattr(self.model, 'forward') and \
                       'attn_dump' in self.model.forward.__code__.co_varnames

        last_dump = None
        for i, batch in enumerate(dl):
            batch = self._move_batch(batch)
            single_dump = {} if collect_attn else None

            kwargs = {}
            forward_vars = self.model.forward.__code__.co_varnames
            if 'x_future_exog' in forward_vars and 'x_future_exog' in batch:
                kwargs['x_future_exog'] = batch['x_future_exog']
            if 'x_future_glob' in forward_vars and 'x_future_glob' in batch:
                kwargs['x_future_glob'] = batch['x_future_glob']

            if accepts_dump and collect_attn:
                preds, gate, _ = self.model(batch['x_price'], batch['x_exog'],
                                            batch['x_glob'],
                                            **kwargs,
                                            attn_dump=single_dump)
            else:
                out = self.model(batch['x_price'], batch['x_exog'],
                                 batch['x_glob'], **kwargs)
                if isinstance(out, tuple) and len(out) == 3:
                    preds, gate, _ = out
                elif isinstance(out, tuple) and len(out) == 2:
                    preds, gate = out
                else:
                    preds, gate = out, None

            preds_np = preds.detach().cpu().numpy()                       # (B, M, H, Q)
            all_pred.append(preds_np)

            # De-normalize median predictions back to raw EUR/MWh.
            # If the model uses RevIN, the per-window stats are stored on
            # the model from the forward() pass. Otherwise we use the
            # dataset-level z-score stats.
            median_idx = self.quantile_idx_median
            preds_norm_median = preds[..., median_idx]                     # tensor (B, M, H)
            model_does_revin = bool(getattr(self.model, 'use_revin', False))
            if model_does_revin:
                preds_raw_t = self.model.denormalize_predictions(
                    preds_norm_median)
                preds_raw = preds_raw_t.detach().cpu().numpy()
                quantiles_raw = self.model.denormalize_predictions(
                    preds).detach().cpu().numpy()
            else:
                preds_raw = self._denormalize(
                    preds_norm_median.detach().cpu().numpy(), markets)
                quantiles_raw = self._denormalize_quantiles(preds_np, markets)
            all_pred_raw.append(preds_raw)
            all_pred_qraw.append(quantiles_raw)

            all_true_raw.append(batch['y_price_raw'].detach().cpu().numpy())
            if gate is not None:
                all_gate.append(gate.detach().cpu().numpy())
            for key in metadata:
                if key in batch:
                    metadata[key].append(batch[key].detach().cpu().numpy())

            # Aggregate attn dump across batches: store mean per key
            if collect_attn and single_dump:
                for k, v in single_dump.items():
                    # MSIM uses only cross-market attention and coupling
                    # gates. Intra-market self-attention is much larger and
                    # is not retained in the attention artifacts.
                    if not (k.startswith('cross_') and
                            (k.endswith('cross') or k.endswith('gate'))):
                        continue
                    if k not in attn_dump:
                        attn_dump[k] = []
                    attn_dump[k].append(v.cpu().numpy()
                                        if torch.is_tensor(v) else v)

        out = {
            'y_pred_norm': np.concatenate(all_pred, axis=0),
            'y_pred_raw':  np.concatenate(all_pred_raw, axis=0),
            'y_pred_quantiles_raw': np.concatenate(all_pred_qraw, axis=0),
            'y_true_raw':  np.concatenate(all_true_raw, axis=0),
        }
        if all_gate:
            out['gate_probs'] = np.concatenate(all_gate, axis=0)
        if collect_attn and attn_dump:
            # Only keep mean to save memory; the MSIM module already aggregates
            for k in list(attn_dump.keys()):
                arrs = attn_dump[k]
                if arrs[0].ndim >= 4:                       # cross attn (B,N,H,M,M)
                    attn_dump[k] = np.concatenate(arrs, axis=0)
                else:                                       # gate (M,M) per batch
                    attn_dump[k] = np.stack(arrs, axis=0).mean(axis=0)
            out['attn_dump'] = attn_dump
        for key, chunks in metadata.items():
            if chunks:
                out[key] = np.concatenate(chunks, axis=0)
        return out

    def _denormalize(self, x_norm: np.ndarray,
                     markets: Optional[List[str]] = None) -> np.ndarray:
        """De-normalize prediction tensor with stats."""
        if self.stats is None or markets is None:
            return x_norm
        out = np.zeros_like(x_norm)
        for i, m in enumerate(markets):
            col = f'price_{m}'
            mu = self.stats[col]['mean']
            sd = self.stats[col]['std']
            out[..., i, :] = x_norm[..., i, :] * sd + mu     # (B, M, H)
        return out

    def _denormalize_quantiles(self, x_norm: np.ndarray,
                               markets: Optional[List[str]] = None
                               ) -> np.ndarray:
        """De-normalize a ``(B, M, H, Q)`` prediction tensor."""
        if self.stats is None or markets is None:
            return x_norm
        out = np.zeros_like(x_norm)
        for i, market in enumerate(markets):
            stat = self.stats[f'price_{market}']
            out[:, i, ...] = x_norm[:, i, ...] * stat['std'] + stat['mean']
        return out

    # ----- main fit loop -----

    def fit(self,
            train_dl: 'DataLoader',
            val_dl: 'DataLoader') -> Dict:
        seed_everything(self.cfg.seed)
        logger.info(f'Training {self.cfg.experiment_name} '
                    f'on {self.device}, '
                    f'params={sum(p.numel() for p in self.model.parameters()):,}')

        # Short-circuit: untrainable model (naive baselines, zero-shot foundation
        # models). We just record a single "epoch -1" with no improvement and
        # skip the loop entirely. The caller will use predict() directly.
        if not self.has_trainable_params:
            logger.info(f'  no trainable parameters -- skipping training loop.')
            self.history = [{'epoch': -1, 'train_total': 0.0,
                             'val_loss': 0.0,
                             'note': 'untrainable model, training skipped'}]
            with open(self.log_dir / 'history.json', 'w') as f:
                json.dump(self.history, f, indent=2)
            return {'history': self.history, 'best_val': 0.0}

        for epoch in range(self.cfg.epochs):
            tr_metrics = self.train_epoch(train_dl, epoch)
            val_metrics = self.validate(val_dl)

            row = {'epoch': epoch, **{f'train_{k}': v for k, v in tr_metrics.items()},
                   **val_metrics}
            self.history.append(row)
            logger.info(f'epoch {epoch:3d} '
                        f'train_loss={tr_metrics["total"]:.4f} '
                        f'val_loss={val_metrics["val_loss"]:.4f} '
                        f'lr={tr_metrics["lr"]:.2e} '
                        f't={tr_metrics["time_s"]:.1f}s')

            # Check for improvement
            if val_metrics['val_loss'] < self.best_val - 1e-6:
                self.best_val = val_metrics['val_loss']
                self.epochs_no_improve = 0
                self._save_checkpoint('best.pt', epoch)
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.cfg.early_stopping_patience:
                    logger.info(f'Early stopping at epoch {epoch} '
                                f'(no improvement for {self.epochs_no_improve} epochs)')
                    break

        # Save history
        with open(self.log_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        return {'history': self.history, 'best_val': self.best_val}

    # ----- checkpoint -----

    def _save_checkpoint(self, name: str, epoch: int):
        path = self.save_dir / name
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val': self.best_val,
            'cfg': asdict(self.cfg),
        }, path)
        logger.info(f'  -> saved checkpoint: {path}')

    def load_checkpoint(self, path: str):
        ck = torch.load(path, map_location=self.device)
        state_dict = ck['model_state_dict']

        # PatchEmbedding lazily allocates `embed.pos_emb` on first forward.
        # If the checkpoint contains pos_emb but our model hasn't been
        # forwarded yet, the parameter doesn't exist and strict load fails.
        # Solution: pre-allocate pos_emb at the right shape if present in ckpt.
        if 'embed.pos_emb' in state_dict:
            import torch as _torch
            from torch import nn as _nn
            target_shape = state_dict['embed.pos_emb'].shape
            embed = self.model.embed
            if embed.pos_emb is None:
                embed.pos_emb = _nn.Parameter(
                    _torch.zeros(*target_shape, device=self.device))

        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            logger.warning(f'Strict checkpoint load failed: {e}')
            logger.warning('Retrying with strict=False; some keys may be skipped.')
            result = self.model.load_state_dict(state_dict, strict=False)
            if result.missing_keys:
                logger.warning(f'  Missing keys: {result.missing_keys}')
            if result.unexpected_keys:
                logger.warning(f'  Unexpected keys (ignored): '
                               f'{result.unexpected_keys}')
        # Optimizer load is best-effort
        if self.optimizer is not None and 'optimizer_state_dict' in ck:
            try:
                self.optimizer.load_state_dict(ck['optimizer_state_dict'])
            except (ValueError, KeyError) as e:
                logger.warning(f'Optimizer state load failed: {e}; '
                              f'using fresh optimizer.')
        self.best_val = ck.get('best_val', float('inf'))
        logger.info(f'Loaded checkpoint from {path} '
                   f'(epoch={ck.get("epoch", "?")})')
