"""Tests for reproducibility utilities."""
import numpy as np
import torch

from src.utils import count_params, set_seed


def test_set_seed_is_reproducible():
    set_seed(123)
    a = torch.randn(5)
    n1 = np.random.rand(3)
    set_seed(123)
    b = torch.randn(5)
    n2 = np.random.rand(3)
    assert torch.allclose(a, b)
    assert np.allclose(n1, n2)


def test_count_params_positive():
    from src.models import AutoencoderAD
    model = AutoencoderAD(device="cpu")
    assert count_params(model.net) > 0
