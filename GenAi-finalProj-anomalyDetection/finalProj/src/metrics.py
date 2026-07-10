"""Metric computation, separated from plotting/IO so it can be unit-tested.

`compute_metrics` is a pure function: given scores and labels it returns a dict.
The evaluation harness (src/evaluation.py) handles files and figures and calls
this for the numbers.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Compute anomaly-detection metrics.

    Args:
        scores: 1-D array, higher = more anomalous.
        labels: 1-D array in {0 normal, 1 anomaly}.
    Returns:
        dict with auroc, pr_auc, best_f1, best_threshold, n_normal, n_anomaly.
    """
    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(int)
    if scores.shape != labels.shape:
        raise ValueError(f"scores and labels must match: {scores.shape} vs {labels.shape}")
    if len(np.unique(labels)) < 2:
        raise ValueError("labels must contain both classes (0 and 1) to score AUROC/PR-AUC")

    auroc = float(roc_auc_score(labels, scores))
    pr_auc = float(average_precision_score(labels, scores))
    best_f1, best_thr = _best_f1(scores, labels)
    return {
        "auroc": auroc,
        "pr_auc": pr_auc,
        "best_f1": best_f1,
        "best_threshold": best_thr,
        "n_normal": int((labels == 0).sum()),
        "n_anomaly": int((labels == 1).sum()),
    }


def _best_f1(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Best achievable F1 over all thresholds, plus the threshold that gives it.

    Reported so the paper can quote a threshold-dependent operating point, not
    only threshold-free AUROC/PR-AUC.
    """
    prec, rec, thr = precision_recall_curve(labels, scores)
    # precision_recall_curve returns len(thr) = len(prec) - 1
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    if len(thr) == 0:
        return 0.0, 0.0
    best_idx = int(np.argmax(f1[:-1])) if len(f1) > 1 else 0
    return float(f1[best_idx]), float(thr[best_idx])


def f1_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """F1 when flagging samples with score >= threshold as anomalies."""
    preds = (np.asarray(scores).reshape(-1) >= threshold).astype(int)
    return float(f1_score(np.asarray(labels).reshape(-1).astype(int), preds, zero_division=0))
