"""Download Fashion-MNIST (cross-platform, no shell snippets).

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --data-root ./data

This simply triggers the loader once, which fetches the data (with the automatic
GitHub-mirror fallback) and caches it under <data-root>/FashionMNIST/raw. Run it
before offline work, or just let the first experiment download on demand.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.constants import DEFAULT_DATA_ROOT  # noqa: E402
from src.data import get_fashion_mnist_oneclass  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    args = p.parse_args()
    print(f"Fetching Fashion-MNIST into {args.data_root} ...")
    tr, te = get_fashion_mnist_oneclass(normal_class=0, data_root=args.data_root,
                                        anomaly_test_per_class=10)
    n_train = len(tr.dataset)
    n_test = len(te.dataset)
    print(f"OK. Cached under {os.path.join(args.data_root, 'FashionMNIST', 'raw')}")
    print(f"(sanity: {n_train} normal training images, {n_test} test images for class 0)")


if __name__ == "__main__":
    main()
