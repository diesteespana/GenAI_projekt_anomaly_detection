"""Latent-space visualization

Extracts each model's latent code for the test set and projects it to 2D with
PCA and t-SNE, coloured by normal vs. anomaly.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


@torch.no_grad()
def get_latents(model, loader, max_points: int = 2000):
    """Return (Z, y): latent codes and labels for up to max_points test samples.

    Handles all three model types uniformly: a VAE exposes encode()->(mu, logvar)
    (we take mu); the AAE exposes encode()->z; the AE exposes .encoder.
    """
    model.net.eval()
    Zs, ys, n = [], [], 0
    for x, y in loader:
        x = x.to(model.device)
        net = model.net
        if hasattr(net, "encode"):
            out = net.encode(x)
            z = out[0] if isinstance(out, tuple) else out
        else:
            z = net.encoder(x)
        Zs.append(z.detach().cpu().numpy())
        ys.append(np.asarray(y).reshape(-1))
        n += len(z)
        if n >= max_points:
            break
    Z = np.concatenate(Zs)[:max_points]
    yv = np.concatenate(ys)[:max_points]
    return Z, yv


def _scatter(ax, P, y, title):
    ax.scatter(P[y == 0, 0], P[y == 0, 1], s=6, alpha=0.5, label="normal", color="#4C78A8")
    ax.scatter(P[y == 1, 0], P[y == 1, 1], s=6, alpha=0.5, label="anomaly", color="#E4572E")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])


def latent_panel(models_latents: dict, out_path: str, seed: int = 0):
    """Build a 2-row (PCA, t-SNE) x N-column (one per model) panel.

    models_latents: {model_name: (Z, y)}.
    """
    names = list(models_latents)
    ncol = len(names)
    fig, axes = plt.subplots(2, ncol, figsize=(3.4 * ncol, 6.6))
    if ncol == 1:
        axes = axes.reshape(2, 1)
    for j, name in enumerate(names):
        Z, y = models_latents[name]
        pca = PCA(n_components=2, random_state=seed).fit_transform(Z)
        _scatter(axes[0, j], pca, y, f"{name} — PCA")
        # t-SNE (perplexity scaled to sample size)
        perp = max(5, min(30, len(Z) // 50))
        tsne = TSNE(n_components=2, random_state=seed, perplexity=perp, init="pca").fit_transform(Z)
        _scatter(axes[1, j], tsne, y, f"{name} — t-SNE")
    axes[0, 0].legend(loc="upper right", fontsize=8, markerscale=1.5)
    fig.suptitle("Latent-space geometry across models (normal vs. anomaly)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
