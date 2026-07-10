"""Unit tests for the pure metrics module (no dataset, no training)."""
import numpy as np
import pytest

from src.metrics import compute_metrics, f1_at_threshold


def test_perfect_separation_gives_auroc_one():
    scores = np.array([0.1, 0.2, 0.3, 0.9, 1.0, 1.1])
    labels = np.array([0, 0, 0, 1, 1, 1])
    m = compute_metrics(scores, labels)
    assert m["auroc"] == pytest.approx(1.0)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["best_f1"] == pytest.approx(1.0)


def test_direction_matters():
    # higher score must mean more anomalous; flipped labels -> auroc 0
    scores = np.array([0.1, 0.2, 0.9, 1.0])
    labels = np.array([1, 1, 0, 0])
    assert compute_metrics(scores, labels)["auroc"] == pytest.approx(0.0)


def test_counts_and_shapes():
    scores = np.random.rand(50)
    labels = (np.arange(50) % 2)
    m = compute_metrics(scores, labels)
    assert m["n_normal"] == 25 and m["n_anomaly"] == 25
    assert 0.0 <= m["auroc"] <= 1.0


def test_requires_both_classes():
    with pytest.raises(ValueError):
        compute_metrics(np.array([0.1, 0.2]), np.array([0, 0]))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_metrics(np.array([0.1, 0.2, 0.3]), np.array([0, 1]))


def test_f1_at_threshold():
    scores = np.array([0.0, 0.4, 0.6, 1.0])
    labels = np.array([0, 0, 1, 1])
    assert f1_at_threshold(scores, labels, 0.5) == pytest.approx(1.0)
