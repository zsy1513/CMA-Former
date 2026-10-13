"""
Transformer family baselines (simplified, faithful to the originals).

Implements:
  - InformerLite      : ProbSparse self-attention + distilling, simplified
  - AutoformerLite    : Auto-correlation + series decomposition (key ideas)
  - PatchTST          : the canonical patch-based TST (Nie et al. 2023)
  - iTransformer      : inverted transformer (Liu et al. 2024)

All keep the I/O contract: forward(x_price, x_exog, x_glob) -> (preds, None, None)
preds shape (B, M, H, Q).

Channel-independent design: each market is processed independently through
shared weights, like PatchTST.

Author: CMA-Former Project
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Helpers shared across this file
# =====================================================================

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)


class _FFN(nn.Module):
    def __init__(self, d, h, drop=0.1):
        super().__init__()
        self.l1 = nn.Linear(d, h)
        self.l2 = nn.Linear(h, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.l2(self.drop(F.gelu(self.l1(x))))


# =====================================================================
# PatchTST (the most relevant comparator)
# =====================================================================

class PatchTSTBaseline(nn.Module):
    def __init__(self,
                 lookback: int,
                 horizon: int,
                 patch_size: int = 16,
                 patch_stride: int = 8,
                 d_model: int = 128,
                 n_heads: int = 8,
                 d_ff: int = 256,
                 n_layers: int = 3,
                 dropout: float = 0.1,
                 n_quantiles: int = 3,
                 n_markets: int = 4,
                 d_exog: int = 3,
                 d_glob: int = 9):
        super().__init__()
        self.M = n_markets
        self.L = lookback
        self.H = horizon
        self.P = patch_size
        self.S = patch_stride
        self.Q = n_quantiles

        in_dim = patch_size + d_exog + d_glob
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = _PositionalEncoding(d_model, max_len=512)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu',
            norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, horizon * n_quantiles)

    def forward(self, x_price, x_exog, x_glob):
        B, M, L = x_price.shape
        P, S = self.P, self.S
        N = (L - P) // S + 1

        patches = x_price.unfold(2, P, S)                           # (B,M,N,P)
        # x_exog (B,M,L,d_x).unfold(2,P,S) -> (B,M,N,d_x,P); mean over patch dim
        exog_unf = x_exog.unfold(2, P, S).mean(dim=-1)               # (B,M,N,d_x)
        # x_glob (B,L,d_z).unfold(1,P,S) -> (B,N,d_z,P); mean over patch dim
        glob_unf = x_glob.unfold(1, P, S).mean(dim=-1)               # (B,N,d_z)
        glob_e = glob_unf.unsqueeze(1).expand(-1, M, -1, -1)              # (B,M,N,d_z)

        feat = torch.cat([patches, exog_unf, glob_e], dim=-1)
        h = self.proj(feat).reshape(B * M, N, -1)
        h = self.pos(h)
        h = self.encoder(h)                                          # (B*M,N,D)
        # Use last patch token
        pooled = h[:, -1, :]
        out = self.head(pooled).view(B, M, self.H, self.Q)
        return out, None, None


# =====================================================================
# iTransformer (Liu et al. 2024) - "inverted" transformer
# =====================================================================

class iTransformerBaseline(nn.Module):
    """
    Treats each variable's whole time series as a token, attends over the
    variable axis. We implement variable axis = M markets + d_exog channels
    (per market collapsed). For consistency we just attend across the M
    markets, with each market embedded from its full lookback series.
    """

    def __init__(self,
                 lookback: int,
                 horizon: int,
                 n_markets: int,
                 d_model: int = 128,
                 n_heads: int = 8,
                 d_ff: int = 256,
                 n_layers: int = 3,
                 dropout: float = 0.1,
                 n_quantiles: int = 3,
                 d_exog: int = 3,
                 d_glob: int = 9):
        super().__init__()
        self.M = n_markets
        self.L = lookback
        self.H = horizon
        self.Q = n_quantiles

        in_dim = lookback + lookback * d_exog + lookback * d_glob
        self.proj = nn.Linear(in_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu',
            norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, horizon * n_quantiles)

    def forward(self, x_price, x_exog, x_glob):
        B, M, L = x_price.shape
        d_x = x_exog.shape[-1]
        d_z = x_glob.shape[-1]
        glob_exp = x_glob.unsqueeze(1).expand(-1, M, -1, -1)        # (B,M,L,d_z)

        # Per-market token: concat(price_L, exog_flat, global_flat)
        feat = torch.cat([x_price,
                          x_exog.reshape(B, M, -1),
                          glob_exp.reshape(B, M, -1)], dim=-1)       # (B, M, in_dim)
        tokens = self.proj(feat)                                     # (B, M, D)
        tokens = self.encoder(tokens)                                # attend across markets
        out = self.head(tokens).view(B, M, self.H, self.Q)
        return out, None, None


# =====================================================================
# InformerLite (ProbSparse simplified)
# =====================================================================

class _ProbSparseAttention(nn.Module):
    """
    Simplified ProbSparse attention: compute "important" queries via a sparsity
    score based on max - mean of scores. Top-c queries get full attention,
    others get average context.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 factor: int = 5):
        super().__init__()
        self.d_h = d_model // n_heads
        self.H = n_heads
        self.factor = factor
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.H, self.d_h).transpose(1, 3)
        q, k, v = qkv.unbind(dim=2)                                # (B, H, L, d_h)
        # Sparsity score: max - mean of attention logits (sample subset of K)
        c = max(1, int(self.factor * math.log(L) + 1))
        sample_k = torch.randint(0, L, (c,), device=x.device)
        k_sample = k[:, :, sample_k, :]                            # (B,H,c,dh)
        scores_sample = q @ k_sample.transpose(-2, -1) / math.sqrt(self.d_h)
        importance = scores_sample.max(-1).values - scores_sample.mean(-1)  # (B,H,L)
        u = max(1, int(self.factor * math.log(L) + 1))
        topu = importance.topk(u, dim=-1).indices                  # (B,H,u)

        # Full attention only for top-u queries
        # Default context: mean(V) per head
        v_mean = v.mean(dim=2, keepdim=True).expand(-1, -1, L, -1)
        ctx = v_mean.clone()

        # Gather top-u queries -> compute exact attention
        # Build a flat indexing trick
        gather_idx = topu.unsqueeze(-1).expand(-1, -1, -1, self.d_h)
        q_top = q.gather(2, gather_idx)                            # (B,H,u,dh)
        scores_full = q_top @ k.transpose(-2, -1) / math.sqrt(self.d_h)
        attn = F.softmax(scores_full, dim=-1)
        ctx_top = attn @ v                                         # (B,H,u,dh)
        # Scatter back
        ctx.scatter_(2, gather_idx, ctx_top)

        out = ctx.transpose(1, 2).contiguous().view(B, L, D)
        return self.out(self.drop(out))


class InformerLite(nn.Module):
    def __init__(self,
                 lookback: int,
                 horizon: int,
                 n_markets: int,
                 d_exog: int,
                 d_glob: int,
                 d_model: int = 128,
                 n_heads: int = 8,
                 d_ff: int = 256,
                 n_layers: int = 3,
                 dropout: float = 0.1,
                 n_quantiles: int = 3):
        super().__init__()
        self.M = n_markets
        self.H = horizon
        self.Q = n_quantiles

        in_size = 1 + d_exog + d_glob
        self.embed = nn.Linear(in_size, d_model)
        self.pos = _PositionalEncoding(d_model, max_len=lookback + 16)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(d_model),
                'attn': _ProbSparseAttention(d_model, n_heads, dropout),
                'norm2': nn.LayerNorm(d_model),
                'ffn': _FFN(d_model, d_ff, dropout),
            }) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, horizon * n_quantiles)

    def forward(self, x_price, x_exog, x_glob):
        B, M, L = x_price.shape
        glob_exp = x_glob.unsqueeze(1).expand(-1, M, -1, -1)
        x = torch.cat([x_price.unsqueeze(-1), x_exog, glob_exp], dim=-1)
        x = x.reshape(B * M, L, -1)
        h = self.pos(self.embed(x))
        for layer in self.layers:
            h = h + layer['attn'](layer['norm1'](h))
            h = h + layer['ffn'](layer['norm2'](h))
        out = self.head(h.mean(dim=1)).view(B, M, self.H, self.Q)
        return out, None, None


# =====================================================================
# AutoformerLite - Auto-correlation + series decomposition
# =====================================================================

class _SeriesDecomp(nn.Module):
    """Moving-average decomposition into trend + seasonal components."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.k = kernel_size
        self.avg = nn.AvgPool1d(kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: (B, L, D)
        # Pad with edge values
        front = x[:, 0:1, :].repeat(1, self.k // 2, 1)
        end = x[:, -1:, :].repeat(1, self.k // 2, 1)
        padded = torch.cat([front, x, end], dim=1)
        trend = self.avg(padded.transpose(1, 2)).transpose(1, 2)
        if trend.size(1) > x.size(1):
            trend = trend[:, :x.size(1), :]
        elif trend.size(1) < x.size(1):
            pad = x[:, :x.size(1) - trend.size(1), :] * 0
            trend = torch.cat([trend, pad], dim=1)
        seasonal = x - trend
        return seasonal, trend


class _AutoCorrelation(nn.Module):
    """Auto-correlation as a substitute for self-attention (frequency domain)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 factor: int = 1):
        super().__init__()
        self.d_h = d_model // n_heads
        self.H = n_heads
        self.factor = factor
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.H, self.d_h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                            # (B,H,L,dh)

        # FFT-based auto-correlation: corr[t] = real(IFFT(Q * conj(K)))
        Q_f = torch.fft.rfft(q, dim=-2)
        K_f = torch.fft.rfft(k, dim=-2)
        corr = torch.fft.irfft(Q_f * torch.conj(K_f), n=L, dim=-2)  # (B,H,L,dh)
        # Sum across feature dim to get a (B,H,L) periodicity score
        score = corr.mean(dim=-1)
        # Top-k delays
        top = max(1, int(self.factor * math.log(L)))
        delays = score.topk(top, dim=-1).indices                    # (B,H,top)

        # Aggregate: weighted sum of v shifted by each delay
        ctx = torch.zeros_like(v)
        weights = F.softmax(score.gather(-1, delays), dim=-1)       # (B,H,top)
        for i in range(top):
            d = delays[..., i]                                      # (B,H)
            # Roll v by -d along time axis; for simplicity, use mean delay
            # (vectorized rolling per-batch is expensive; we use a single
            # global mean delay to keep this simple and runnable)
            mean_d = int(d.float().mean().item()) % L
            v_rolled = torch.roll(v, shifts=mean_d, dims=-2)
            ctx = ctx + v_rolled * weights[..., i:i+1].unsqueeze(-1)

        out = ctx.transpose(1, 2).contiguous().view(B, L, D)
        return self.out(self.drop(out))


class AutoformerLite(nn.Module):
    def __init__(self,
                 lookback: int,
                 horizon: int,
                 n_markets: int,
                 d_exog: int,
                 d_glob: int,
                 d_model: int = 128,
                 n_heads: int = 8,
                 d_ff: int = 256,
                 n_layers: int = 3,
                 dropout: float = 0.1,
                 n_quantiles: int = 3,
                 kernel_size: int = 25):
        super().__init__()
        self.M = n_markets
        self.H = horizon
        self.Q = n_quantiles

        in_size = 1 + d_exog + d_glob
        self.embed = nn.Linear(in_size, d_model)

        self.decomp = _SeriesDecomp(kernel_size)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': _AutoCorrelation(d_model, n_heads, dropout),
                'decomp1': _SeriesDecomp(kernel_size),
                'ffn': _FFN(d_model, d_ff, dropout),
                'decomp2': _SeriesDecomp(kernel_size),
            }) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, horizon * n_quantiles)

    def forward(self, x_price, x_exog, x_glob):
        B, M, L = x_price.shape
        glob_exp = x_glob.unsqueeze(1).expand(-1, M, -1, -1)
        x = torch.cat([x_price.unsqueeze(-1), x_exog, glob_exp], dim=-1)
        x = x.reshape(B * M, L, -1)
        h = self.embed(x)

        # Initial decomposition
        seasonal, trend = self.decomp(h)
        h = seasonal
        for layer in self.layers:
            attn_out = layer['attn'](h)
            s1, _ = layer['decomp1'](h + attn_out)
            ffn_out = layer['ffn'](s1)
            s2, _ = layer['decomp2'](s1 + ffn_out)
            h = s2
        # Combine
        h_pool = h.mean(dim=1)
        out = self.head(h_pool).view(B, M, self.H, self.Q)
        return out, None, None
