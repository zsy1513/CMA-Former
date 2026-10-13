"""
RNN-based deep-learning baselines.

Implements:
  - LSTMForecaster      : per-market BiLSTM encoder + linear decoder
  - CNNLSTMForecaster   : Conv1D feature extractor + LSTM + Attention
  - TFTSimplified       : a faithful but minimal Temporal Fusion Transformer

All keep the same interface as CMA-Former:
  forward(x_price, x_exog, x_glob) -> (preds, None, None)

For TFT we keep its key components (variable selection networks, gated
residual networks, multi-head attention) but simplify by removing static
covariates and the specific 'static encoder'.

Author: CMA-Former Project
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------

class LSTMForecaster(nn.Module):
    def __init__(self,
                 n_markets: int,
                 d_exog: int,
                 d_glob: int,
                 horizon: int,
                 hidden: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 n_quantiles: int = 3,
                 bidirectional: bool = True):
        super().__init__()
        self.M = n_markets
        self.H = horizon
        self.Q = n_quantiles
        in_size = 1 + d_exog + d_glob                           # price + exog + global
        self.lstm = nn.LSTM(input_size=in_size,
                            hidden_size=hidden,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=bidirectional)
        d_out = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(d_out, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon * n_quantiles),
        )

    def forward(self, x_price, x_exog, x_glob):
        # Run per market, weights shared
        B, M, L = x_price.shape
        # Build per-(B, M) sequence: (B*M, L, 1+d_exog+d_glob)
        glob_exp = x_glob.unsqueeze(1).expand(-1, M, -1, -1)         # (B,M,L,d_z)
        x = torch.cat([x_price.unsqueeze(-1),
                       x_exog,
                       glob_exp], dim=-1)                            # (B,M,L,F)
        x = x.reshape(B * M, L, -1)
        out, _ = self.lstm(x)
        # Use last time step
        last = out[:, -1, :]                                         # (B*M, d_out)
        h = self.head(last).view(B, M, self.H, self.Q)
        return h, None, None


# ---------------------------------------------------------------------
# CNN-LSTM-Attention
# ---------------------------------------------------------------------

class CNNLSTMForecaster(nn.Module):
    """Conv1D feature extraction + LSTM + Bahdanau-style additive attention."""

    def __init__(self,
                 n_markets: int,
                 d_exog: int,
                 d_glob: int,
                 horizon: int,
                 cnn_channels: int = 64,
                 lstm_hidden: int = 128,
                 dropout: float = 0.1,
                 n_quantiles: int = 3):
        super().__init__()
        self.M = n_markets
        self.H = horizon
        self.Q = n_quantiles
        in_size = 1 + d_exog + d_glob
        # Conv1d expects (B, C, L); we transpose
        self.conv = nn.Sequential(
            nn.Conv1d(in_size, cnn_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(cnn_channels, lstm_hidden,
                            num_layers=2, batch_first=True, dropout=dropout)
        # Additive attention
        self.attn_W = nn.Linear(lstm_hidden, lstm_hidden)
        self.attn_v = nn.Linear(lstm_hidden, 1)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, horizon * n_quantiles),
        )

    def forward(self, x_price, x_exog, x_glob):
        B, M, L = x_price.shape
        glob_exp = x_glob.unsqueeze(1).expand(-1, M, -1, -1)
        x = torch.cat([x_price.unsqueeze(-1), x_exog, glob_exp], dim=-1)  # (B,M,L,F)
        x = x.reshape(B * M, L, -1).transpose(1, 2)                       # (B*M, F, L)
        c = self.conv(x).transpose(1, 2)                                   # (B*M, L, C)
        h, _ = self.lstm(c)                                                # (B*M, L, H)
        # Attention pooling over time
        u = torch.tanh(self.attn_W(h))                                     # (B*M, L, H)
        a = F.softmax(self.attn_v(u).squeeze(-1), dim=-1)                  # (B*M, L)
        ctx = (a.unsqueeze(-1) * h).sum(dim=1)                             # (B*M, H_dim)
        out = self.head(ctx).view(B, M, self.H, self.Q)
        return out, None, None


# ---------------------------------------------------------------------
# Simplified TFT (variable selection + GRN + interpretable attention)
# ---------------------------------------------------------------------

class GRN(nn.Module):
    """Gated Residual Network."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int = None,
                 dropout: float = 0.1):
        super().__init__()
        d_out = d_out or d_hidden
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.gate = nn.Linear(d_in, d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = self.drop(self.fc2(h))
        g = torch.sigmoid(self.gate(x))
        return self.norm(self.skip(x) + g * h)


class VariableSelection(nn.Module):
    """Variable Selection Network (per-step soft selection of K features)."""

    def __init__(self, n_vars: int, d_per_var: int, d_hidden: int,
                 dropout: float = 0.1):
        super().__init__()
        self.n_vars = n_vars
        self.d_per_var = d_per_var
        self.var_grns = nn.ModuleList([GRN(d_per_var, d_hidden, d_hidden, dropout)
                                       for _ in range(n_vars)])
        self.weights_grn = GRN(n_vars * d_per_var, d_hidden, n_vars, dropout)

    def forward(self, x):
        # x: (B, T, n_vars, d_per_var)
        B, T, V, D = x.shape
        flat = x.reshape(B, T, V * D)
        w = F.softmax(self.weights_grn(flat), dim=-1)             # (B, T, V)
        zs = []
        for i in range(V):
            zs.append(self.var_grns[i](x[:, :, i, :]))
        z = torch.stack(zs, dim=-2)                               # (B, T, V, H)
        out = (w.unsqueeze(-1) * z).sum(dim=-2)                   # (B, T, H)
        return out, w


class TFTSimplified(nn.Module):
    """
    Simplified TFT: per-market temporal fusion. Variables = price (1) +
    per-market exog (3) + global (d_glob), each treated as a 1-D feature.

    Uses VS for past + future inputs (for future we only have known global +
    no price), then a multi-head self-attention over the horizon.
    """

    def __init__(self,
                 d_exog: int,
                 d_glob: int,
                 horizon: int,
                 lookback: int,
                 d_hidden: int = 64,
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 n_quantiles: int = 3,
                 n_markets: int = 4):
        super().__init__()
        self.M = n_markets
        self.H = horizon
        self.L = lookback
        self.Q = n_quantiles
        self.d_h = d_hidden

        # Each variable is a scalar -> embed to d_h via linear
        n_past = 1 + d_exog + d_glob
        n_fut = d_glob                                 # only global known in future
        self.past_embed = nn.ModuleList(
            [nn.Linear(1, d_hidden) for _ in range(n_past)])
        self.fut_embed = nn.ModuleList(
            [nn.Linear(1, d_hidden) for _ in range(n_fut)])

        self.vs_past = VariableSelection(n_past, d_hidden, d_hidden, dropout)
        self.vs_fut = VariableSelection(n_fut, d_hidden, d_hidden, dropout)

        self.lstm_enc = nn.LSTM(d_hidden, d_hidden, batch_first=True)
        self.lstm_dec = nn.LSTM(d_hidden, d_hidden, batch_first=True)

        self.attn = nn.MultiheadAttention(d_hidden, n_heads,
                                          dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_hidden)
        self.grn_post = GRN(d_hidden, d_hidden, d_hidden, dropout)

        self.head = nn.Linear(d_hidden, n_quantiles)

    def _embed_past(self, x_price, x_exog, x_glob):
        # x_price (B,M,L), x_exog (B,M,L,d_x), x_glob (B,L,d_z)
        B, M, L = x_price.shape
        d_x = x_exog.shape[-1]
        d_z = x_glob.shape[-1]

        # Build a per-(B,M) variable-axis tensor: (B*M, L, n_vars, 1)
        p = x_price.unsqueeze(-1)                                  # (B,M,L,1)
        glob_exp = x_glob.unsqueeze(1).expand(-1, M, -1, -1)       # (B,M,L,d_z)
        full = [p] + [x_exog[..., k:k+1] for k in range(d_x)] + \
               [glob_exp[..., k:k+1] for k in range(d_z)]
        full_t = torch.cat([t.reshape(B*M, L, 1, 1) for t in full], dim=2)
        # Embed each variable
        embs = []
        for i, layer in enumerate(self.past_embed):
            embs.append(layer(full_t[:, :, i, :]))
        emb = torch.stack(embs, dim=2)                              # (B*M,L,V,H)
        return emb

    def _embed_fut(self, x_glob_fut):
        # x_glob_fut: (B, H, d_z)  -> for future we only know global features.
        # In our setting we need to fabricate this from training data; for
        # simplicity we tile the last lookback global.
        B, H, d_z = x_glob_fut.shape
        embs = []
        for i, layer in enumerate(self.fut_embed):
            embs.append(layer(x_glob_fut[..., i:i+1]))
        emb = torch.stack(embs, dim=2)                              # (B,H,V,H)
        return emb

    def forward(self, x_price, x_exog, x_glob):
        B, M, L = x_price.shape
        H = self.H

        # ------ encode past
        past = self._embed_past(x_price, x_exog, x_glob)            # (B*M,L,V,h)
        past_z, past_w = self.vs_past(past)                          # (B*M,L,h)
        enc_out, (hn, cn) = self.lstm_enc(past_z)

        # ------ encode future (use last lookback global as proxy)
        # In production you'd plug in actual day-ahead known features. Here
        # we tile the last available global vector across H steps.
        x_glob_fut = x_glob[:, -H:, :]                               # (B,H,d_z)
        x_glob_fut_bm = x_glob_fut.unsqueeze(1).expand(-1, M, -1, -1
                                                       ).reshape(B*M, H, -1)
        fut_emb = self._embed_fut(x_glob_fut_bm)                     # (B*M,H,V,h)
        fut_z, fut_w = self.vs_fut(fut_emb)                          # (B*M,H,h)
        dec_out, _ = self.lstm_dec(fut_z, (hn, cn))

        # ------ attention: queries from decoder, keys/values from encoder
        attn_out, _ = self.attn(dec_out, enc_out, enc_out)
        h_attn = self.attn_norm(dec_out + attn_out)
        h_post = self.grn_post(h_attn)                               # (B*M, H, h)

        out = self.head(h_post).view(B, M, H, self.Q)
        return out, None, None
