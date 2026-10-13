import pytest

torch = pytest.importorskip('torch')

from src.utils.trainer import seed_everything


def test_seed_before_construction_reproduces_initial_parameters():
    seed_everything(77)
    first = torch.nn.Linear(7, 3)
    first_state = {key: value.detach().clone()
                   for key, value in first.state_dict().items()}

    seed_everything(77)
    second = torch.nn.Linear(7, 3)
    for key, value in second.state_dict().items():
        assert torch.equal(first_state[key], value)
