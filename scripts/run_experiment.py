"""Unified experiment runner (single model, or all).

One command trains and evaluates any of the three models on the same data split,
writing metrics + plots into results/. This is the entry point everyone uses, so
everyone is scored the same way. For the full side-by-side comparison with an
overlay ROC, use scripts/run_comparison.py instead.

Examples:
    python scripts/run_experiment.py --model autoencoder --epochs 20
    python scripts/run_experiment.py --model vae --normal-class 0 --epochs 20
    python scripts/run_experiment.py --model all --epochs 20      # train all, compare
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import get_fashion_mnist_oneclass  # noqa: E402
from src.evaluation import compare, evaluate  # noqa: E402
from src.models import REGISTRY  # noqa: E402
from src.utils import count_params, get_device, set_seed  # noqa: E402


def run_one(model_key, train_loader, test_loader, epochs, device, results_dir, args):
    print(f"\n=== {model_key} ===")
    if model_key == "vae":
        model = REGISTRY[model_key](
            device=device,
            beta=args.beta,
            score_method=args.vae_score_method,
            mc_samples=args.mc_samples,
        )
        tag = f"vae_beta{str(args.beta).replace('.', 'p')}_{args.vae_score_method}"
    else:
        model = REGISTRY[model_key](device=device)
        tag = model_key
    hist = model.fit(train_loader, epochs=epochs)
    metrics = evaluate(model, test_loader, results_dir=results_dir, tag=tag, history=hist)
    print(f"{model_key}: AUROC={metrics['auroc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  "
          f"best-F1={metrics['best_f1']:.4f}")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="autoencoder", choices=list(REGISTRY) + ["all"])
    p.add_argument("--normal-class", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--anomaly-per-class", type=int, default=200,
                   help="cap anomaly test images per class (keeps it fast)")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--beta", type=float, default=1.0, help="VAE beta-KL weight")
    p.add_argument("--vae-score-method", default="reconstruction_probability",
                   choices=["mse", "mc_mse", "reconstruction_probability"],
                   help="VAE anomaly score used when --model vae")
    p.add_argument("--mc-samples", type=int, default=10,
                   help="Monte-Carlo samples for VAE scoring")
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"device={device}  normal_class={args.normal_class}")

    train_loader, test_loader = get_fashion_mnist_oneclass(
        normal_class=args.normal_class,
        batch_size=args.batch_size,
        anomaly_test_per_class=args.anomaly_per_class,
        seed=args.seed,
    )

    keys = list(REGISTRY) if args.model == "all" else [args.model]
    results = []
    for k in keys:
        try:
            results.append(run_one(k, train_loader, test_loader, args.epochs, device, args.results_dir, args))
        except NotImplementedError as e:
            print(f"skipping {k}: {e}")

    if len(results) > 1:
        print("\n" + compare(results, args.results_dir))


if __name__ == "__main__":
    main()
