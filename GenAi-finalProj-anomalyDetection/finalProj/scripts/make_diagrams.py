"""Generate the report's architecture diagrams (reproducible, no manual drawing).

Outputs:
  report/figures/aae_architecture.png  — the AAE pipeline
  report/figures/latent_spectrum.png   — AE -> VAE -> AAE regularization spectrum

Run:  python scripts/make_diagrams.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "report", "figures")

TEAL = "#1D9E75"
TEAL_BG = "#E1F5EE"
AMBER = "#BA7517"
AMBER_BG = "#FAEEDA"
GRAY = "#888780"
GRAY_BG = "#F1EFE8"
INK = "#2C2C2A"


def _box(ax, x, y, w, h, text, edge, face, fontsize=11, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.5, edgecolor=edge, facecolor=face))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=INK, weight=weight, wrap=True)


def _arrow(ax, x1, y1, x2, y2, color=INK, style="-|>", lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=14, linewidth=lw, color=color))


def aae_architecture():
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off")

    y = 3.6; h = 0.9
    _box(ax, 0.2, y, 1.5, h, "Input\n$x$", GRAY, GRAY_BG)
    _box(ax, 2.2, y, 1.6, h, "Encoder\n$q(z|x)$", TEAL, TEAL_BG, weight="bold")
    _box(ax, 4.3, y, 1.4, h, "Latent\n$z$", TEAL, "#FFFFFF", weight="bold")
    _box(ax, 6.2, y, 1.6, h, "Decoder\n$p(x|z)$", TEAL, TEAL_BG, weight="bold")
    _box(ax, 8.3, y, 1.5, h, "Recon.\n$\\hat{x}$", GRAY, GRAY_BG)
    for x1, x2 in [(1.7, 2.2), (3.8, 4.3), (5.7, 6.2), (7.8, 8.3)]:
        _arrow(ax, x1, y + h / 2, x2, y + h / 2)

    # reconstruction loss bracket
    ax.annotate("", xy=(0.95, y - 0.15), xytext=(9.05, y - 0.15),
                arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1.2))
    ax.text(5.0, y - 0.5, "reconstruction loss  $\\|x-\\hat{x}\\|^2$",
            ha="center", va="center", fontsize=10, color=GRAY, style="italic")

    # adversarial branch
    _box(ax, 4.0, 1.2, 2.0, 0.9, "Latent\ndiscriminator", AMBER, AMBER_BG, weight="bold")
    _box(ax, 7.0, 1.2, 2.4, 0.9, "Prior  $p(z)=\\mathcal{N}(0,I)$", AMBER, "#FFFFFF")
    _arrow(ax, 5.0, y, 5.0, 2.1, color=AMBER)               # latent -> discriminator
    ax.text(5.15, 2.75, "$q(z)$", ha="left", va="center", fontsize=10, color=AMBER)
    _arrow(ax, 7.0, 1.65, 6.0, 1.65, color=AMBER)            # prior -> discriminator
    ax.text(2.0, 1.65, "encoder trained\nto fool $D$", ha="center", va="center",
            fontsize=10, color=AMBER, style="italic")
    _arrow(ax, 4.0, 1.65, 2.9, 1.65, color=AMBER, style="-|>")
    ax.text(6.5, 0.75, "adversarial loss  $\\Rightarrow$  match $q(z)$ to prior",
            ha="center", va="center", fontsize=10, color=AMBER, style="italic")

    ax.set_title("Adversarial Autoencoder (AAE)", fontsize=14, weight="bold", color=INK, pad=10)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "aae_architecture.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def latent_spectrum():
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.6); ax.axis("off")

    cols = [
        ("AE", "no latent constraint", "$-$", GRAY, GRAY_BG),
        ("VAE", "KL divergence to\nGaussian prior", "$\\mathrm{KL}(q\\,\\|\\,p)$", TEAL, TEAL_BG),
        ("AAE", "adversarial match\nto prior", "discriminator", AMBER, AMBER_BG),
    ]
    xs = [0.4, 3.7, 7.0]; w = 2.6; h = 2.0; y = 1.1
    for (name, mech, tag, edge, face), x in zip(cols, xs):
        _box(ax, x, y, w, h, "", edge, face)
        ax.text(x + w / 2, y + h - 0.4, name, ha="center", va="center",
                fontsize=15, weight="bold", color=INK)
        ax.text(x + w / 2, y + h / 2 + 0.05, mech, ha="center", va="center",
                fontsize=11, color=INK)
        ax.text(x + w / 2, y + 0.35, tag, ha="center", va="center",
                fontsize=11, color=edge, style="italic")
    for x1, x2 in [(3.0, 3.7), (6.3, 7.0)]:
        _arrow(ax, x1, y + h / 2, x2, y + h / 2)

    ax.annotate("", xy=(0.4, 0.5), xytext=(9.6, 0.5),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6))
    ax.text(5.0, 0.2, "increasing latent-space regularization",
            ha="center", va="center", fontsize=11, color=INK, style="italic")

    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "latent_spectrum.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    aae_architecture()
    latent_spectrum()
    print(f"diagrams written to {os.path.abspath(OUT)}")
