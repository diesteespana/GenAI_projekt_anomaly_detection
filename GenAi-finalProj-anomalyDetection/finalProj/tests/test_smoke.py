"""Integration smoke test: real data, tiny training, full pipeline.

Marked 'slow' because it downloads Fashion-MNIST and trains for one epoch. Skip
it in fast runs with:  pytest -m "not slow"
"""
import os

import pytest

from src.data import get_fashion_mnist_oneclass
from src.evaluation import evaluate
from src.models import AutoencoderAD
from src.utils import set_seed


@pytest.mark.slow
def test_pipeline_runs(tmp_path):
    set_seed(0)
    train_loader, test_loader = get_fashion_mnist_oneclass(
        normal_class=0, batch_size=64, anomaly_test_per_class=20, seed=0
    )
    model = AutoencoderAD(device="cpu")
    history = model.fit(train_loader, epochs=1)
    assert len(history["loss"]) == 1 and history["loss"][0] > 0

    metrics = evaluate(model, test_loader, results_dir=str(tmp_path), history=history)
    assert 0.0 <= metrics["auroc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert os.path.exists(os.path.join(tmp_path, "autoencoder_metrics.json"))
    assert os.path.exists(os.path.join(tmp_path, "autoencoder_loss.png"))
