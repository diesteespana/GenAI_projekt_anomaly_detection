"""Project extensions: β-VAE study, latent visualization, and cost analysis.

Runs the three additions in one place and writes artefacts to results/studies/
and report/figures/. Designed to finish quickly on CPU.

    python scripts/run_studies.py --epochs 8

Written as shared tooling (Person 1) so the beta-study, latent-dim sweep, latent
visualization, and cost analysis all use the same harness.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import get_fashion_mnist_oneclass  # noqa: E402
from src.latent_viz import get_latents, latent_panel  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.models import AAEAD, AutoencoderAD, VAEAD  # noqa: E402
from src.utils import count_params, get_device, save_json, set_seed  # noqa: E402

OUT = "results/studies"
FIGS = "report/figures"


def _score(model, test_loader):
    scores, labels = model.score_loader(test_loader)
    return compute_metrics(scores, labels)


def beta_study(train_loader, test_loader, device, epochs, betas):
    print("\n=== β-VAE study ===")
    rows = []
    for b in betas:
        set_seed(0)
        m = VAEAD(beta=b, device=device)
        m.fit(train_loader, epochs=epochs)
        met = _score(m, test_loader)
        met["beta"] = b
        rows.append(met)
        print(f"  β={b:<4}  AUROC={met['auroc']:.4f}  PR-AUC={met['pr_auc']:.4f}")
    # table
    tbl = "| β | AUROC | PR-AUC | best-F1 |\n|---|---|---|---|\n"
    for r in rows:
        tbl += f"| {r['beta']} | {r['auroc']:.4f} | {r['pr_auc']:.4f} | {r['best_f1']:.4f} |\n"
    open(os.path.join(OUT, "beta_study.md"), "w").write("# β-VAE study (Fashion-MNIST, class 0)\n\n" + tbl)
    save_json(rows, os.path.join(OUT, "beta_study.json"))
    # plot
    plt.figure(figsize=(5, 3.6))
    bs = [r["beta"] for r in rows]
    plt.plot(bs, [r["auroc"] for r in rows], "o-", label="AUROC")
    plt.plot(bs, [r["pr_auc"] for r in rows], "s--", label="PR-AUC")
    plt.xscale("log", base=2)
    plt.xlabel("β  (KL weight, log scale)")
    plt.ylabel("score")
    plt.title("β-VAE: effect of latent-regularization strength")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "beta_study.png"), dpi=140)
    plt.close()
    print(tbl)
    return rows


def latent_dim_study(train_loader, test_loader, device, epochs, dims):
    print("\n=== latent-dimension study (AE) ===")
    rows = []
    for d in dims:
        set_seed(0)
        m = AutoencoderAD(latent_dim=d, device=device)
        m.fit(train_loader, epochs=epochs)
        met = _score(m, test_loader)
        met["latent_dim"] = d
        rows.append(met)
        print(f"  latent_dim={d:<3}  AUROC={met['auroc']:.4f}")
    tbl = "| latent dim | AUROC | PR-AUC |\n|---|---|---|\n"
    for r in rows:
        tbl += f"| {r['latent_dim']} | {r['auroc']:.4f} | {r['pr_auc']:.4f} |\n"
    open(os.path.join(OUT, "latent_dim_study.md"), "w").write("# Latent-dimension study (AE, class 0)\n\n" + tbl)
    save_json(rows, os.path.join(OUT, "latent_dim_study.json"))
    print(tbl)
    return rows


def latent_visualization(train_loader, test_loader, device, epochs):
    print("\n=== latent-space visualization ===")
    models = {}
    for name, ctor in [("AE", AutoencoderAD), ("VAE", VAEAD), ("AAE", AAEAD)]:
        set_seed(0)
        m = ctor(device=device)
        m.fit(train_loader, epochs=epochs)
        models[name] = m
    latents = {name: get_latents(m, test_loader, max_points=1800) for name, m in models.items()}
    latent_panel(latents, os.path.join(FIGS, "latent_panel.png"))
    print(f"  wrote {FIGS}/latent_panel.png")
    return models  # reuse for cost timing


def cost_analysis(models, train_loader, test_loader, device, epochs=1):
    print("\n=== computational cost ===")
    rows = []
    for name, m in models.items():
        # parameter count (include discriminator for AAE)
        params = count_params(m.net)
        if hasattr(m, "disc"):
            params += count_params(m.disc)
        # train time / epoch
        t0 = time.perf_counter(); m.fit(train_loader, epochs=1); t_train = time.perf_counter() - t0
        # inference time / image
        t0 = time.perf_counter(); m.score_loader(test_loader); t_inf = time.perf_counter() - t0
        n_test = len(test_loader.dataset)
        rows.append({"model": name, "params": params,
                     "train_s_per_epoch": round(t_train, 2),
                     "infer_ms_per_img": round(1000 * t_inf / n_test, 3)})
        print(f"  {name}: {params:,} params, {t_train:.2f}s/epoch, {1000*t_inf/n_test:.3f} ms/img")
    tbl = "| Model | Parameters | Train (s/epoch) | Inference (ms/img) |\n|---|---|---|---|\n"
    for r in rows:
        tbl += f"| {r['model']} | {r['params']:,} | {r['train_s_per_epoch']} | {r['infer_ms_per_img']} |\n"
    open(os.path.join(OUT, "cost_analysis.md"), "w").write("# Computational cost (CPU)\n\n" + tbl)
    save_json(rows, os.path.join(OUT, "cost_analysis.json"))
    print(tbl)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--normal-class", type=int, default=0)
    p.add_argument("--anomaly-per-class", type=int, default=200)
    args = p.parse_args()

    os.makedirs(OUT, exist_ok=True); os.makedirs(FIGS, exist_ok=True)
    device = get_device()
    set_seed(0)
    train_loader, test_loader = get_fashion_mnist_oneclass(
        normal_class=args.normal_class, anomaly_test_per_class=args.anomaly_per_class, seed=0)

    beta_study(train_loader, test_loader, device, args.epochs, betas=[0.5, 1, 2, 4, 8])
    latent_dim_study(train_loader, test_loader, device, args.epochs, dims=[8, 16, 32, 64])
    models = latent_visualization(train_loader, test_loader, device, args.epochs)
    cost_analysis(models, train_loader, test_loader, device)
    print("\nAll studies complete. Artefacts in results/studies/ and report/figures/.")


if __name__ == "__main__":
    main()
