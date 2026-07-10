"""Baseline experiment: AE across every Fashion-MNIST normal class.

Trains the autoencoder ten times, once per choice of "normal" class, and
aggregates AUROC / PR-AUC into a table + bar chart.

Usage:
    python scripts/run_baseline_all_classes.py --epochs 15
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.constants import FASHION_MNIST_CLASSES as CLASS_NAMES  # noqa: E402
from src.data import get_fashion_mnist_oneclass  # noqa: E402
from src.evaluation import evaluate  # noqa: E402
from src.models import AutoencoderAD  # noqa: E402
from src.utils import get_device, set_seed  # noqa: E402
from src.visualize import auroc_by_class_bar, reconstruction_grid  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--anomaly-per-class", type=int, default=200)
    p.add_argument("--results-dir", default="results/baseline_all_classes")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"device={device}")

    rows = []
    class_aurocs = {}
    for c in range(10):
        print(f"\n=== normal_class={c} ({CLASS_NAMES[c]}) ===")
        train_loader, test_loader = get_fashion_mnist_oneclass(
            normal_class=c, batch_size=args.batch_size,
            anomaly_test_per_class=args.anomaly_per_class, seed=args.seed,
        )
        model = AutoencoderAD(device=device)
        hist = model.fit(train_loader, epochs=args.epochs)
        m = evaluate(model, test_loader, results_dir=args.results_dir, tag=f"ae_class{c}", history=hist)
        m["class_name"] = CLASS_NAMES[c]
        rows.append(m)
        class_aurocs[c] = m["auroc"]
        print(f"class {c} ({CLASS_NAMES[c]}): AUROC={m['auroc']:.4f}  PR-AUC={m['pr_auc']:.4f}")
        # save one reconstruction figure (class 0 is enough for the paper)
        if c == 0:
            reconstruction_grid(model, test_loader,
                                os.path.join(args.results_dir, "ae_reconstructions_class0.png"))

    # aggregate
    aurocs = [r["auroc"] for r in rows]
    praucs = [r["pr_auc"] for r in rows]
    summary = {
        "per_class": rows,
        "mean_auroc": float(np.mean(aurocs)),
        "std_auroc": float(np.std(aurocs)),
        "mean_pr_auc": float(np.mean(praucs)),
        "std_pr_auc": float(np.std(praucs)),
    }
    with open(os.path.join(args.results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # markdown table
    table = "| Class | AUROC | PR-AUC |\n|---|---|---|\n"
    for r in rows:
        table += f"| {r['class_name']} | {r['auroc']:.4f} | {r['pr_auc']:.4f} |\n"
    table += f"| **mean ± std** | **{np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}** | **{np.mean(praucs):.4f} ± {np.std(praucs):.4f}** |\n"
    with open(os.path.join(args.results_dir, "baseline_table.md"), "w") as f:
        f.write("# AE baseline across normal classes\n\n" + table)

    auroc_by_class_bar(class_aurocs, os.path.join(args.results_dir, "auroc_by_class.png"))

    print("\n" + table)
    print(f"mean AUROC = {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")


if __name__ == "__main__":
    main()
