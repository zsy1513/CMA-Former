"""
CMA-Former core components.

Implements all reusable building blocks of the model:
  - PatchEmbedding              (Module 1)
  - PositionwiseFFN             (helper)
  - IntraMarketBlock            (Module 2: shared-weight, per-market temporal)
  - CrossMarketAttention        (Module 3: with Coupling Prior Gating)
  - RegimeAwareMoEHead          (Module 4)

Design notes
------------
* All modules accept an "attn_dump" dict (optional). When provided, the module
  stores its attention weights into the dict, keyed by a string identifier.
  This is what powers the MSIM analysis in `src/msim.py` -- we re-run the
  trained model on the test set with attn_dump enabled and aggregate weights.

* The cross-market attention takes shapes (B, M, N, D) and produces an
  asymmetric attention matrix per token: alpha[b, m, m', n] in [0, 1].

Author: CMA-Former Project
"""

from __future__ import annotations

import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Module 1: Patch Embedding
# =====================================================================

class PatchEmbedding(nn.Module):
    """
    Splits each market's 1-D price series (and per-market exog mean over each
    patch) into patches and projects to d_model.

    Input :
      price : (B, M, L)           past prices, normalized
      exog  : (B, M, L, d_x)      per-market exogenous (load, RES, flow_net)
      glob  : (B, L, d_z)         global exogenous (gas, carbon, calendar)
    Output:
      tokens: (B, M, N, d_model)  N = (L - P) // S + 1
    """

    def __init__(self,
                 d_model: int,
                 patch_size: int,
                 patch_stride: int,
                 d_exog: int,
                 d_glob: int):
        super().__init__()
        self.P = patch_size
        self.S = patch_stride
        # Per-token feature is concat(price_patch, exog_mean, glob_mean)
        in_dim = patch_size + d_exog + d_glob
        self.proj = nn.Linear(in_dim, d_model)
        # Learnable positional encoding allocated lazily on first forward
        self.register_buffer('_pos_init', torch.zeros(1), persistent=False)
        self.pos_emb: Optional[nn.Parameter] = None
        self.d_model = d_model

    def _make_pos(self, n_patches: int, device: torch.device):
        if self.pos_emb is None or self.pos_emb.shape[0] != n_patches:
            self.pos_emb = nn.Parameter(
                torch.zeros(n_patches, self.d_model, device=device))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self,
                price: torch.Tensor,    # (B, M, L)
                exog:  torch.Tensor,    # (B, M, L, d_x)
                glob:  torch.Tensor     # (B, L, d_z)
                ) -> torch.Tensor:      # (B, M, N, d_model)
        B, M, L = price.shape
        P, S = self.P, self.S
        N = (L - P) // S + 1

        # PyTorch tensor.unfold(dim, size, step) replaces shape[dim] with N
        # and APPENDS a new trailing dim of length `size` (the window).
        # So the output puts the patch dim at the END, not at index `dim`.

        # price (B, M, L) -> (B, M, N, P)
        patches = price.unfold(dimension=2, size=P, step=S)            # (B, M, N, P)

        # exog (B, M, L, d_x) unfold along time (dim=2) -> (B, M, N, d_x, P)
        # Mean over the last (patch) dim to get (B, M, N, d_x).
        exog_unf = exog.unfold(dimension=2, size=P, step=S)            # (B, M, N, d_x, P)
        exog_patch = exog_unf.mean(dim=-1)                             # (B, M, N, d_x)

        # glob (B, L, d_z) unfold along time (dim=1) -> (B, N, d_z, P)
        # Mean over patch dim, then broadcast across markets.
        glob_unf = glob.unfold(dimension=1, size=P, step=S)            # (B, N, d_z, P)
        glob_patch = glob_unf.mean(dim=-1)                             # (B, N, d_z)
        glob_patch = glob_patch.unsqueeze(1).expand(-1, M, -1, -1)     # (B, M, N, d_z)

        feat = torch.cat([patches, exog_patch, glob_patch], dim=-1)    # (B, M, N, P+d_x+d_z)

        tokens = self.proj(feat)                                        # (B, M, N, d_model)

        # Positional encoding (lazy init in case L varies between train/eval)
        self._make_pos(N, tokens.device)
        tokens = tokens + self.pos_emb.view(1, 1, N, self.d_model)

        return tokens


# =====================================================================
# Helpers
# =====================================================================

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))


class _MHA(nn.Module):
    """Standard multi-head self-attention with optional weight capture."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                attn_dump: Optional[Dict[str, torch.Tensor]] = None,
                dump_key: Optional[str] = None
                ) -> torch.Tensor:
        # x: (..., N, d_model)
        *batch, N, D = x.shape
        qkv = self.qkv(x).view(*batch, N, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=-3)                          # (..., N, H, dh)

        # Rearrange to (..., H, N, dh)
        q = q.transpose(-3, -2)
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)
        if attn_dump is not None and dump_key is not None:
            attn_dump[dump_key] = attn.detach()

        attn = self.drop(attn)
        ctx = torch.matmul(attn, v)                            # (..., H, N, dh)
        ctx = ctx.transpose(-3, -2).contiguous()               # (..., N, H, dh)
        ctx = ctx.view(*batch, N, self.d_model)
        return self.out(ctx)


# =====================================================================
# Module 2: Intra-market block (shared-weight Transformer encoder)
# =====================================================================

class IntraMarketBlock(nn.Module):
    """One Transformer encoder layer applied per-market, weights shared."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = _MHA(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                attn_dump: Optional[Dict] = None,
                dump_prefix: str = ''
                ) -> torch.Tensor:
        # x: (B, M, N, D)  -- treat batch and market as one big batch
        B, M, N, D = x.shape
        h = x.view(B * M, N, D)
        # Pre-LN
        h_norm = self.norm1(h)
        attn_out = self.attn(h_norm, attn_dump, dump_prefix + 'self')
        h = h + self.drop(attn_out)
        h = h + self.drop(self.ffn(self.norm2(h)))
        return h.view(B, M, N, D)


# =====================================================================
# Module 3: Cross-market Attention (with Coupling Prior Gating)
# =====================================================================

class CrossMarketAttention(nn.Module):
    """
    For each market m and each patch position n, attend over the SAME patch
    position across ALL markets (attention across markets at aligned
    patch positions).

    Prior-gating: each pair (m -> m') has a learnable scalar a_{m,m'}; the
    NTC-based prior c_{m,m'} (passed in via `coupling_prior`, M x M) acts as
    additive log-bias. The effective gate is:
        g_{m,m'} = sigmoid(a_{m,m'} + beta * c_{m,m'})

    Implementation: we run multi-head attention with M as the sequence axis,
    one such "mini-attention" per (b, n, h) triple. To keep this efficient
    we batch over (B*N) and run one MHA over the M-length sequence.

    Captured weights (when attn_dump is given):
        attn_dump['cross']: (B, N, H, M, M)  -- alpha[m -> m'] per token/head
        attn_dump['gate']:                 (M, M)  -- effective coupling gates
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float,
                 n_markets: int, use_prior: bool = True,
                 fixed_prior: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.M = n_markets
        self.use_prior = use_prior
        self.fixed_prior = fixed_prior

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        # Coupling prior gating params: a_{m,m'} learnable
        self.gate_logits = nn.Parameter(torch.zeros(n_markets, n_markets))
        # Start from a data-driven gate and let optimization decide whether
        # the frozen topology prior is useful. This avoids imposing an
        # unvalidated topology effect at initialization.
        self.gate_beta = nn.Parameter(torch.tensor(0.0))

        # Coupling prior buffer (set by `set_prior`)
        self.register_buffer('coupling_prior',
                             torch.zeros(n_markets, n_markets))

    def set_prior(self, prior: torch.Tensor):
        """prior: (M, M) -- normalized in [0, 1] by the caller."""
        assert prior.shape == (self.M, self.M)
        with torch.no_grad():
            self.coupling_prior.copy_(prior)

    def gating_matrix(self) -> torch.Tensor:
        """Returns the effective (M, M) gating values in [0, 1]."""
        if self.fixed_prior:
            return self.coupling_prior.clamp(min=1e-4, max=1.0)
        if self.use_prior:
            return torch.sigmoid(self.gate_logits +
                                 self.gate_beta * self.coupling_prior)
        return torch.sigmoid(self.gate_logits)

    def forward(self, x: torch.Tensor,
                attn_dump: Optional[Dict] = None,
                dump_prefix: str = '') -> torch.Tensor:
        # x: (B, M, N, D)
        B, M, N, D = x.shape
        assert M == self.M, f'Expected M={self.M}, got {M}'

        # Rearrange to put M on the sequence axis: (B*N, M, D)
        h = x.permute(0, 2, 1, 3).contiguous().view(B * N, M, D)

        q = self.q_proj(h).view(B * N, M, self.n_heads, self.d_head)
        k = self.k_proj(h).view(B * N, M, self.n_heads, self.d_head)
        v = self.v_proj(h).view(B * N, M, self.n_heads, self.d_head)

        # (B*N, H, M, dh)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Logits: (B*N, H, M, M) -- m queries m'
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Add log-gate bias: log(g + eps). g shape (M, M)
        g = self.gating_matrix()
        log_g = torch.log(g + 1e-6)
        # Broadcast to (1, 1, M, M)
        logits = logits + log_g.view(1, 1, M, M)

        attn = F.softmax(logits, dim=-1)                       # (B*N, H, M, M)
        if attn_dump is not None:
            # Reshape to (B, N, H, M, M) for downstream MSIM aggregation
            attn_dump[dump_prefix + 'cross'] = \
                attn.detach().view(B, N, self.n_heads, M, M)
            attn_dump[dump_prefix + 'gate'] = g.detach()
        attn = self.drop(attn)

        ctx = torch.matmul(attn, v)                            # (B*N, H, M, dh)
        ctx = ctx.transpose(1, 2).contiguous().view(B * N, M, D)
        ctx = self.out_proj(ctx)

        # Back to (B, M, N, D)
        ctx = ctx.view(B, N, M, D).permute(0, 2, 1, 3).contiguous()
        return ctx


class CrossMarketBlock(nn.Module):
    """Pre-LN block: LayerNorm -> CrossMarketAttention -> residual -> FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float,
                 n_markets: int, use_prior: bool = True,
                 fixed_prior: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross = CrossMarketAttention(d_model, n_heads, dropout,
                                          n_markets, use_prior, fixed_prior)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)
        # LayerScale keeps early optimization close to the market-wise
        # temporal encoder while allowing cross-market exchange to grow.
        self.cross_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor,
                attn_dump: Optional[Dict] = None,
                dump_prefix: str = ''):
        h_norm = self.norm1(x)
        cross_out = self.cross(h_norm, attn_dump, dump_prefix)
        x = x + self.cross_scale * self.drop(cross_out)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# =====================================================================
# Module 4: Regime-Aware Mixture-of-Experts head
# =====================================================================

class RegimeAwareMoEHead(nn.Module):
    """
    Three experts (Normal, Spike, Negative) plus a soft gating network.

    Each expert outputs Q quantile predictions for H future steps.
    Gate output is a 3-way softmax. Final prediction = sum_k g_k * e_k.

    Inputs:
      h : (B, M, D_pool)        pooled per-market representation
    Outputs:
      preds  : (B, M, H, Q)     final quantile predictions (normalized scale)
      gate   : (B, M, 3)        regime probabilities (for aux loss + analysis)
      experts: (B, M, 3, H, Q)  individual expert outputs (for diagnostics)
    """

    def __init__(self, d_model: int, horizon: int, n_quantiles: int = 3,
                 hidden: int = 128, dropout: float = 0.1,
                 init_gate_bias_normal: float = 2.0):
        """
        init_gate_bias_normal: extra logit added to the 'normal' gate at init,
        so the model starts with ~softmax([2.0, 0, 0]) ~ (0.79, 0.10, 0.10)
        instead of uniform (0.33, 0.33, 0.33). This anchors training to the
        majority regime; the gate then opens up to rare regimes only when
        the regime loss provides clear signal.
        """
        super().__init__()
        self.H = horizon
        self.Q = n_quantiles
        self.K = 3                                 # 3 regimes

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, horizon * n_quantiles),
            )
            for _ in range(self.K)
        ])
        self.gate = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.K),
        )
        # Bias 'normal' at init so untrained experts don't dominate
        with torch.no_grad():
            self.gate[-1].bias.zero_()
            self.gate[-1].bias[0] = init_gate_bias_normal

        # ============ CRITICAL INIT SCALING ============
        # The default PyTorch init for the FINAL Linear of each expert
        # produces outputs with std ~ 0.16 (abs_mean ~ 0.13), but the
        # normalised target has abs_mean ~ 0.69. The model would then
        # have to spend many epochs learning a 5x amplification on the
        # output, severely slowing convergence and causing the val loss
        # to plateau early.
        #
        # We rescale the LAST Linear of each expert by a factor that
        # brings the initial output magnitude up to match the target.
        # This is a well-known fix for MoE / multi-stage MLP heads
        # (see e.g. PatchTST's choice of single-layer head).
        #
        # Empirically, scale = 4.0 brings expert output abs_mean from
        # ~0.13 to ~0.55, comparable to PatchTST's single-layer head.
        with torch.no_grad():
            for expert in self.experts:
                last_linear = expert[-1]                          # final Linear
                last_linear.weight.mul_(4.0)
                last_linear.bias.zero_()

    def forward(self, h: torch.Tensor):
        # h: (B, M, D)
        B, M, D = h.shape
        H, Q, K = self.H, self.Q, self.K

        gate_logits = self.gate(h)                         # (B, M, K)
        gate_probs = F.softmax(gate_logits, dim=-1)         # (B, M, K)

        expert_outs = []
        for k in range(K):
            o = self.experts[k](h).view(B, M, H, Q)
            expert_outs.append(o)
        experts = torch.stack(expert_outs, dim=2)          # (B, M, K, H, Q)

        # Mixture: sum_k g_k * e_k
        # gate_probs (B, M, K) -> (B, M, K, 1, 1)
        preds = (gate_probs.view(B, M, K, 1, 1) * experts).sum(dim=2)  # (B,M,H,Q)

        return preds, gate_probs, experts
