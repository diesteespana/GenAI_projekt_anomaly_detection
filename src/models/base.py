"""Common interface for all anomaly-detection models.

AE, VAE and AAE all subclass this and implement fit() + anomaly_score().
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import DataLoader


class AnomalyModel(ABC):
    """Abstract base class for a one-class anomaly-detection model.

    Convention used throughout the project:
      * Models are trained on NORMAL data only.
      * ``anomaly_score`` returns a 1-D array where a HIGHER value means MORE
        anomalous. The evaluation harness assumes this direction.
    """

    name: str = "base"

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)

    @abstractmethod
    def fit(self, train_loader: DataLoader, epochs: int) -> dict:
        """Train the model on normal data.

        Returns a dict of training history (e.g. {"loss": [...]}) for plotting.
        """
        raise NotImplementedError

    @abstractmethod
    def anomaly_score(self, x: torch.Tensor) -> np.ndarray:
        """Return per-sample anomaly scores for a batch ``x``.

        Args:
            x: tensor of shape (B, C, H, W) on any device.
        Returns:
            np.ndarray of shape (B,), higher = more anomalous.
        """
        raise NotImplementedError

    def score_loader(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Run ``anomaly_score`` over a labeled loader.

        Returns (scores, labels) as numpy arrays, where label 0 = normal and
        1 = anomaly. This helper is shared so nobody re-implements the loop.
        """
        all_scores: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        for x, y in loader:
            x = x.to(self.device)
            s = self.anomaly_score(x)
            all_scores.append(np.asarray(s).reshape(-1))
            all_labels.append(np.asarray(y).reshape(-1))
        return np.concatenate(all_scores), np.concatenate(all_labels)
