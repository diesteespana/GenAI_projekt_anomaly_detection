"""Model comparison / integration runner.

Trains every registered model that is implemented, on one shared data split, and
produces the integrated comparison artefacts the report needs:
  * per-model metrics JSON + diagnostic plots (via evaluate)
  * results/comparison/overlay_roc.png     — all ROC curves on one axis
  * results/comparison/comparison_bars.png — AUROC/PR-AUC/best-F1 bar chart
  * results/comparison/comparison.md       — the summary table

Example:
    python scripts/run_comparison.py --epochs 20 --normal-class 0
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import get_fashion_mnist_oneclass  # noqa: E402
from src.evaluation import compare, evaluate  # noqa: E402
from src.models import REGISTRY  # noqa: E402
from src.utils import get_device, save_json, set_seed  # noqa: E402
from src.visualize import overlay_roc  # noqa: E402


def _comparison_bars(metrics_list, path):
    models = [m["model"] for m in metrics_list]
    keys = ["auroc", "pr_auc", "best_f1"]
    labels = ["AUROC", "PR-AUC", "best-F1"]
    x = np.arange(len(models))
    w = 0.25
    plt.figure(figsize=(1.6 * len(models) + 3, 4))
    for i, (k, lab) in enumerate(zip(keys, labels)):
        vals = [m.get(k, 0.0) for m in metrics_list]
        bars = plt.bar(x + (i - 1) * w, vals, w, label=lab)
        for b, v in zip(bars, vals):
            plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                     ha="center", fontsize=7)
    plt.xticks(x, models)
    plt.ylim(0, 1.08)
    plt.ylabel("Score")
    plt.title("Model comparison — one-class Fashion-MNIST")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--normal-class", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--anomaly-per-class", type=int, default=200)
    p.add_argument("--results-dir", default="results/comparison")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"device={device}  normal_class={args.normal_class}")

    train_loader, test_loader = get_fashion_mnist_oneclass(
        normal_class=args.normal_class, batch_size=args.batch_size,
        anomaly_test_per_class=args.anomaly_per_class, seed=args.seed,
    )

    metrics_list, roc_curves = [], []
    for key in REGISTRY:
        print(f"\n=== {key} ===")
        model = REGISTRY[key](device=device)
        try:
            hist = model.fit(train_loader, epochs=args.epochs)
        except NotImplementedError as e:
            print(f"skipping {key}: not implemented yet ({e})")
            continue
        m = evaluate(model, test_loader, results_dir=args.results_dir, tag=key, history=hist)
        metrics_list.append(m)
        scores, labels = model.score_loader(test_loader)
        roc_curves.append((labels, scores, key))
        print(f"{key}: AUROC={m['auroc']:.4f}  PR-AUC={m['pr_auc']:.4f}  best-F1={m['best_f1']:.4f}")

    if not metrics_list:
        print("\nNo models were runnable. Implement at least one fit().")
        return

    save_json({"normal_class": args.normal_class, "epochs": args.epochs,
               "seed": args.seed, "models": metrics_list},
              os.path.join(args.results_dir, "comparison_summary.json"))
    table = compare(metrics_list, args.results_dir)
    overlay_roc(roc_curves, os.path.join(args.results_dir, "overlay_roc.png"),
                title=f"ROC comparison (normal={args.normal_class})")
    _comparison_bars(metrics_list, os.path.join(args.results_dir, "comparison_bars.png"))
    print("\n" + table)
    print(f"Artefacts written to {args.results_dir}/")


if __name__ == "__main__":
    main()
