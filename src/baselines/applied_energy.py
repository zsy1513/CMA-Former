"""Task-adapted baseline models for the day-ahead forecasting protocol."""

from __future__ import annotations

import torch
import torch.nn as nn


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, values):
        pad = (self.kernel_size - 1) // 2
        front = values[:, :, :1].expand(-1, -1, pad)
        end = values[:, :, -1:].expand(-1, -1, pad)
        padded = torch.cat([front, values, end], dim=-1)
        return self.pool(padded)


class DLinearBaseline(nn.Module):
    """Channel-independent DLinear with separate seasonal/trend heads."""

    def __init__(self, lookback: int, horizon: int, n_markets: int,
                 n_quantiles: int = 3, moving_average: int = 25):
        super().__init__()
        self.horizon = horizon
        self.n_quantiles = n_quantiles
        self.decomposition = MovingAverage(moving_average)
        self.seasonal = nn.ModuleList([
            nn.Linear(lookback, horizon * n_quantiles)
            for _ in range(n_markets)
        ])
        self.trend = nn.ModuleList([
            nn.Linear(lookback, horizon * n_quantiles)
            for _ in range(n_markets)
        ])

    def forward(self, x_price, x_exog, x_glob):
        trend = self.decomposition(x_price)
        seasonal = x_price - trend
        forecasts = []
        for market, (seasonal_head, trend_head) in enumerate(
                zip(self.seasonal, self.trend)):
            forecast = seasonal_head(seasonal[:, market, :])
            forecast = forecast + trend_head(trend[:, market, :])
            forecasts.append(forecast.view(
                x_price.shape[0], self.horizon, self.n_quantiles))
        return torch.stack(forecasts, dim=1), None, None


class _NBEATSxBlock(nn.Module):
    def __init__(self, lookback: int, context_dim: int, horizon: int,
                 n_quantiles: int, hidden: int, n_layers: int, dropout: float):
        super().__init__()
        layers = []
        input_dim = lookback + context_dim
        for layer in range(n_layers):
            layers.extend([
                nn.Linear(input_dim if layer == 0 else hidden, hidden),
                nn.ReLU(), nn.Dropout(dropout),
            ])
        self.mlp = nn.Sequential(*layers)
        self.backcast = nn.Linear(hidden, lookback)
        self.forecast = nn.Linear(hidden, horizon * n_quantiles)

    def forward(self, residual, context):
        hidden = self.mlp(torch.cat([residual, context], dim=-1))
        return self.backcast(hidden), self.forecast(hidden)


class NBEATSxBaseline(nn.Module):
    """N-BEATS residual stacks conditioned on historical and future exogenous data."""

    def __init__(self, lookback: int, horizon: int, n_markets: int,
                 d_exog: int, d_glob: int, d_future_exog: int,
                 d_future_glob: int, n_quantiles: int = 3,
                 context_lookback: int = 168, hidden: int = 256,
                 n_blocks: int = 4, n_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.n_markets = n_markets
        self.n_quantiles = n_quantiles
        self.context_lookback = min(context_lookback, lookback)
        context_dim = (
            self.context_lookback * (d_exog + d_glob)
            + horizon * (d_future_exog + d_future_glob))
        self.blocks = nn.ModuleList([
            _NBEATSxBlock(lookback, context_dim, horizon, n_quantiles,
                          hidden, n_layers, dropout)
            for _ in range(n_blocks)
        ])

    def forward(self, x_price, x_exog, x_glob,
                x_future_exog, x_future_glob):
        batch, markets, _ = x_price.shape
        context_length = self.context_lookback
        historical_global = x_glob[:, -context_length:, :].unsqueeze(1).expand(
            -1, markets, -1, -1)
        future_global = x_future_glob.unsqueeze(1).expand(-1, markets, -1, -1)
        context = torch.cat([
            x_exog[:, :, -context_length:, :].reshape(batch, markets, -1),
            historical_global.reshape(batch, markets, -1),
            x_future_exog.reshape(batch, markets, -1),
            future_global.reshape(batch, markets, -1),
        ], dim=-1).reshape(batch * markets, -1)
        residual = x_price.reshape(batch * markets, self.lookback)
        forecast = torch.zeros(
            batch * markets, self.horizon * self.n_quantiles,
            device=x_price.device, dtype=x_price.dtype)
        for block in self.blocks:
            backcast, block_forecast = block(residual, context)
            residual = residual - backcast
            forecast = forecast + block_forecast
        forecast = forecast.view(
            batch, markets, self.horizon, self.n_quantiles)
        return forecast, None, None


class TemporalGATBaseline(nn.Module):
    """Temporal encoder followed by a PyG graph-attention layer."""

    def __init__(self, n_markets: int, d_exog: int, d_glob: int,
                 horizon: int, n_quantiles: int, edge_index,
                 hidden: int = 128, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        try:
            from torch_geometric.nn import GATConv
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                'TemporalGATBaseline requires torch-geometric') from exc
        self.n_markets = n_markets
        self.horizon = horizon
        self.n_quantiles = n_quantiles
        self.temporal = nn.Sequential(
            nn.Conv1d(1 + d_exog + d_glob, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.gat = GATConv(hidden, hidden // heads, heads=heads,
                           dropout=dropout, add_self_loops=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, horizon * n_quantiles),
        )
        self.register_buffer('edge_index', edge_index.long())

    def _batched_edges(self, batch_size: int, device):
        edges = []
        for batch in range(batch_size):
            edges.append(self.edge_index + batch * self.n_markets)
        return torch.cat(edges, dim=1).to(device)

    def forward(self, x_price, x_exog, x_glob):
        batch, markets, length = x_price.shape
        global_features = x_glob.unsqueeze(1).expand(-1, markets, -1, -1)
        sequence = torch.cat(
            [x_price.unsqueeze(-1), x_exog, global_features], dim=-1)
        sequence = sequence.reshape(batch * markets, length, -1).transpose(1, 2)
        node_features = self.temporal(sequence).squeeze(-1)
        graph_features = self.gat(
            node_features, self._batched_edges(batch, node_features.device))
        prediction = self.head(graph_features).view(
            batch, markets, self.horizon, self.n_quantiles)
        return prediction, None, None


def topology_edge_index(markets):
    pairs = [('DE_LU', 'FR'), ('DE_LU', 'NL'),
             ('FR', 'BE'), ('NL', 'BE')]
    edges = []
    for left, right in pairs:
        if left in markets and right in markets:
            i, j = markets.index(left), markets.index(right)
            edges.extend([(i, j), (j, i)])
    return torch.tensor(edges, dtype=torch.long,
                        device='cpu').t().contiguous()


class IndependentMarketWrapper(nn.Module):
    """Train one separate forecaster per market with no shared parameters."""

    def __init__(self, market_models):
        super().__init__()
        self.market_models = nn.ModuleList(market_models)

    def forward(self, x_price, x_exog, x_glob):
        predictions = []
        for market, model in enumerate(self.market_models):
            result = model(x_price[:, market:market + 1],
                           x_exog[:, market:market + 1], x_glob)
            prediction = result[0] if isinstance(result, tuple) else result
            predictions.append(prediction)
        return torch.cat(predictions, dim=1), None, None


class ConcatenatedTransformerBaseline(nn.Module):
    """Temporal Transformer after direct concatenation of all market channels."""

    def __init__(self, n_markets: int, d_exog: int, d_glob: int,
                 horizon: int, n_quantiles: int, d_model: int = 128,
                 n_heads: int = 8, d_ff: int = 256, n_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.n_markets = n_markets
        self.horizon = horizon
        self.n_quantiles = n_quantiles
        input_dim = n_markets * (1 + d_exog) + d_glob
        self.input_projection = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(
            d_model, n_markets * horizon * n_quantiles)

    def forward(self, x_price, x_exog, x_glob):
        batch, markets, length = x_price.shape
        market_features = torch.cat(
            [x_price.unsqueeze(-1), x_exog], dim=-1)
        market_features = market_features.permute(0, 2, 1, 3).reshape(
            batch, length, -1)
        sequence = torch.cat([market_features, x_glob], dim=-1)
        encoded = self.encoder(self.input_projection(sequence))
        prediction = self.head(encoded[:, -1]).view(
            batch, markets, self.horizon, self.n_quantiles)
        return prediction, None, None
