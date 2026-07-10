"""Visualization utilities (Person 1, shared).

  * reconstruction_grid: input / reconstruction / error heatmap, normal vs
    anomalous samples.
  * overlay_roc: every model's ROC curve on one axis.

Both just need the AnomalyModel interface, so they work for any model without
changes.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from .constants import FASHION_MNIST_CLASSES
from .models.base import AnomalyModel


@torch.no_grad()
def reconstruction_grid(model, test_loader, out_path: str, n: int = 6):
    """Save a grid: top rows = normal samples, bottom rows = anomalies.

    Each sample shows three panels: input, reconstruction, |input-recon| heatmap.
    Only works for models that expose a reconstruction (AE/VAE). For models
    without one (e.g. a purely latent-based scorer) this is skipped by the caller.
    """
    # collect a few normal and a few anomalous images
    normals, anomalies = [], []
    for x, y in test_loader:
        for img, lab in zip(x, y):
            if lab.item() == 0 and len(normals) < n:
                normals.append(img)
            elif lab.item() == 1 and len(anomalies) < n:
                anomalies.append(img)
        if len(normals) >= n and len(anomalies) >= n:
            break

    samples = torch.stack(normals + anomalies).to(model.device)
    recon = _reconstruct(model, samples)
    if recon is None:
        return False
    err = (samples - recon).abs()

    rows = 2 * n
    fig, axes = plt.subplots(rows, 3, figsize=(4.2, 1.4 * rows))
    col_titles = ["input", "reconstruction", "error"]
    for r in range(rows):
        imgs = [samples[r, 0].cpu(), recon[r, 0].cpu(), err[r, 0].cpu()]
        cmaps = ["gray", "gray", "inferno"]
        for c in range(3):
            ax = axes[r, c]
            ax.imshow(imgs[c], cmap=cmaps[c])
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=9)
        tag = "normal" if r < n else "ANOMALY"
        axes[r, 0].set_ylabel(tag, fontsize=8, rotation=90)
    fig.suptitle(f"Reconstructions — {model.name}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def _reconstruct(model, x):
    """Get a reconstruction tensor from AE/VAE-style nets, else None."""
    if hasattr(model, "net") and hasattr(model.net, "forward"):
        out = model.net(x)
        if isinstance(out, tuple):  # VAE returns (recon, mu, logvar)
            return out[0]
        return out  # AE returns recon
    return None


def overlay_roc(curves: list[tuple[np.ndarray, np.ndarray, str]], out_path: str, title: str = "ROC comparison"):
    """curves: list of (labels, scores, model_name). Draws all ROCs on one axis."""
    plt.figure(figsize=(5, 4.5))
    for labels, scores, name in curves:
        fpr, tpr, _ = roc_curve(labels, scores)
        auroc = roc_auc_score(labels, scores)
        plt.plot(fpr, tpr, label=f"{name} (AUROC={auroc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(title); plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def auroc_by_class_bar(class_aurocs: dict[int, float], out_path: str, model_name: str = "autoencoder"):
    """Bar chart of AUROC for each Fashion-MNIST normal class."""
    classes = sorted(class_aurocs)
    vals = [class_aurocs[c] for c in classes]
    names = FASHION_MNIST_CLASSES
    plt.figure(figsize=(7, 3.8))
    bars = plt.bar([names[c] for c in classes], vals, color="#4C78A8")
    plt.axhline(np.mean(vals), color="crimson", linestyle="--", linewidth=1,
                label=f"mean = {np.mean(vals):.3f}")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}",
                 ha="center", fontsize=7)
    plt.ylim(0, 1.05); plt.ylabel("AUROC")
    plt.title(f"{model_name}: AUROC per normal class (Fashion-MNIST)")
    plt.xticks(rotation=40, ha="right"); plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
