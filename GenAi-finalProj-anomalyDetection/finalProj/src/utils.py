"""Shared utilities: seeding, device selection, JSON/plot helpers."""

from __future__ import annotations

import json
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int = 0, deterministic: bool = True) -> None:
    """Seed python, numpy and torch, and (optionally) force deterministic algs.

    deterministic=True makes runs bit-reproducible on the same machine at a small
    speed cost; set False for large GPU runs where throughput matters.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> str:
    """Return 'cuda' if available and preferred, else 'cpu'."""
    return "cuda" if (prefer_cuda and torch.cuda.is_available()) else "cpu"


def count_params(module: torch.nn.Module) -> int:
    """Number of trainable parameters — handy for the report's model table."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_history_plot(history: dict, path: str, title: str = "Training curves") -> None:
    """Plot every numeric series in a training-history dict on one axis.

    Models return histories like {"loss": [...]} or {"loss_g": [...],
    "loss_d": [...]}. This turns them into a figure for the report/appendix.
    """
    if not history:
        return
    plt.figure(figsize=(5, 3.6))
    for name, series in history.items():
        if isinstance(series, (list, tuple)) and len(series) and isinstance(series[0], (int, float)):
            plt.plot(range(1, len(series) + 1), series, marker="o", markersize=3, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, dpi=130)
    plt.close()
