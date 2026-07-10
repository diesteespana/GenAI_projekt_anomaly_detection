"""Shared evaluation harness.

Owned by Person 1 (integration). Every model is scored through ``evaluate`` so
the numbers are directly comparable. Person 2 and Person 3: you do not need to
touch this file — just make your model implement ``AnomalyModel`` and pass it in.

Metrics (computed in src/metrics.py):
  * AUROC   — threshold-free; main headline number.
  * PR-AUC  — average precision; more informative when anomalies are rare.
  * best-F1 — best F1 over all thresholds, with the threshold that achieves it,
              so the paper can quote a concrete operating point.

Outputs go to a results dir: a metrics JSON, three diagnostic plots (ROC,
precision-recall, score histogram), and — if a training history is passed — a
loss-curve plot.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

from .metrics import compute_metrics
from .models.base import AnomalyModel
from .utils import save_history_plot, save_json


def evaluate(model: AnomalyModel, test_loader, results_dir: str,
             tag: str | None = None, history: dict | None = None) -> dict:
    """Score a trained model on a labeled test loader and write metrics + plots.

    Args:
        model: a trained AnomalyModel.
        test_loader: yields (x, y), y in {0 normal, 1 anomaly}.
        results_dir: directory to write <tag>_metrics.json and <tag>_*.png.
        tag: filename prefix; defaults to model.name.
        history: optional training-history dict; if given, a loss curve is saved.
    Returns:
        dict with model, tag, auroc, pr_auc, best_f1, best_threshold, counts.
    """
    tag = tag or model.name
    os.makedirs(results_dir, exist_ok=True)

    scores, labels = model.score_loader(test_loader)
    metrics = {"model": model.name, "tag": tag, **compute_metrics(scores, labels)}
    save_json(metrics, os.path.join(results_dir, f"{tag}_metrics.json"))

    _plot_roc(labels, scores, metrics["auroc"], os.path.join(results_dir, f"{tag}_roc.png"), tag)
    _plot_pr(labels, scores, metrics["pr_auc"], os.path.join(results_dir, f"{tag}_pr.png"), tag)
    _plot_hist(labels, scores, metrics["best_threshold"],
               os.path.join(results_dir, f"{tag}_score_hist.png"), tag)
    if history:
        save_history_plot(history, os.path.join(results_dir, f"{tag}_loss.png"),
                          title=f"Training curves — {tag}")
    return metrics


def compare(metrics_list: list[dict], results_dir: str) -> str:
    """Write a comparison table (markdown) across models. Returns the table str."""
    os.makedirs(results_dir, exist_ok=True)
    header = "| Model | AUROC | PR-AUC | best-F1 |\n|---|---|---|---|\n"
    rows = "".join(
        f"| {m['model']} | {m['auroc']:.4f} | {m['pr_auc']:.4f} | {m.get('best_f1', float('nan')):.4f} |\n"
        for m in sorted(metrics_list, key=lambda m: m["auroc"], reverse=True)
    )
    table = header + rows
    with open(os.path.join(results_dir, "comparison.md"), "w") as f:
        f.write("# Model comparison\n\n" + table)
    return table


def _plot_roc(labels, scores, auroc, path, tag):
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(4.5, 4))
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC — {tag}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def _plot_pr(labels, scores, pr_auc, path, tag):
    prec, rec, _ = precision_recall_curve(labels, scores)
    plt.figure(figsize=(4.5, 4))
    plt.plot(rec, prec, label=f"PR-AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall — {tag}")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def _plot_hist(labels, scores, threshold, path, tag):
    plt.figure(figsize=(4.5, 4))
    plt.hist(scores[labels == 0], bins=50, alpha=0.6, label="normal", density=True)
    plt.hist(scores[labels == 1], bins=50, alpha=0.6, label="anomaly", density=True)
    if threshold is not None:
        plt.axvline(threshold, color="black", linestyle="--", linewidth=1,
                    label=f"best-F1 thr = {threshold:.3g}")
    plt.xlabel("Anomaly score")
    plt.ylabel("Density")
    plt.title(f"Score distribution — {tag}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
