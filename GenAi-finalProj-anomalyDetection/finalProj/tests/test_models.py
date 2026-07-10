"""Interface + shape tests for all registered models (synthetic, fast)."""
import numpy as np
import torch

from src.models import REGISTRY
from src.models.base import AnomalyModel
from src.models.blocks import FEATURE_DIM


def test_registry_models_subclass_interface():
    for key, cls in REGISTRY.items():
        assert issubclass(cls, AnomalyModel), f"{key} must subclass AnomalyModel"


def test_anomaly_score_shape_and_nonneg():
    x = torch.rand(8, 1, 28, 28)
    for key, cls in REGISTRY.items():
        model = cls(device="cpu")
        scores = model.anomaly_score(x)
        assert scores.shape == (8,), f"{key} returned wrong score shape"
        assert np.all(scores >= 0), f"{key} produced negative scores"


def test_feature_dim_constant():
    assert FEATURE_DIM == 64 * 7 * 7


def test_autoencoder_learns_on_synthetic():
    """One batch, a few steps: AE loss should drop (pipeline sanity)."""
    from torch.utils.data import DataLoader, TensorDataset
    from src.models import AutoencoderAD
    x = torch.rand(64, 1, 28, 28)
    loader = DataLoader(TensorDataset(x, torch.zeros(64, dtype=torch.long)), batch_size=32)
    model = AutoencoderAD(device="cpu")
    hist = model.fit(loader, epochs=3)
    assert hist["loss"][-1] < hist["loss"][0]
