import pytest

torch = pytest.importorskip('torch')
nn = torch.nn

from src.models.future_adapter import FutureCovariateAdapter


class _ZeroForecaster(nn.Module):
    def forward(self, x_price, x_exog, x_glob):
        batch, markets, _ = x_price.shape
        return torch.zeros(batch, markets, 24, 3)


def test_future_adapter_uses_previous_day_curve_as_anchor():
    model = FutureCovariateAdapter(
        _ZeroForecaster(), n_quantiles=3,
        d_future_exog=3, d_future_glob=6)
    for parameter in model.future_head.parameters():
        nn.init.zeros_(parameter)

    x_price = torch.zeros(2, 4, 336)
    x_exog = torch.zeros(2, 4, 336, 3)
    x_glob = torch.zeros(2, 336, 8)
    x_future_exog = torch.zeros(2, 4, 24, 3)
    x_future_exog[..., 0] = 1.25
    x_future_glob = torch.zeros(2, 24, 6)

    prediction, _, _ = model(
        x_price, x_exog, x_glob, x_future_exog, x_future_glob)
    assert prediction.shape == (2, 4, 24, 3)
    assert torch.allclose(prediction, torch.full_like(prediction, 1.25))
