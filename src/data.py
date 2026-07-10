"""Data loaders for one-class anomaly detection.

Every model trains and is evaluated through
these loaders, so the data split is identical for everyone.

Two datasets:
  * Fashion-MNIST one-class  -> ``get_fashion_mnist_oneclass`` (light, always
    reproducible; use this for development and as an ablation).
  * MVTec AD                 -> ``get_mvtec`` (the headline benchmark; needs a
    manual download, see data_cards/mvtec_ad.md).

Label convention everywhere: 0 = normal, 1 = anomaly.
"""

from __future__ import annotations

import os
import urllib.request

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from .constants import (
    DEFAULT_DATA_ROOT,
    FASHION_MNIST_FILES,
    FASHION_MNIST_MIRROR,
    LABEL_ANOMALY,
    LABEL_NORMAL,
)


def _download_fashion_mnist_mirror(raw_dir: str) -> None:
    """Fetch the four raw .gz files from the GitHub mirror (cross-platform)."""
    os.makedirs(raw_dir, exist_ok=True)
    for fname in FASHION_MNIST_FILES:
        dest = os.path.join(raw_dir, fname)
        if os.path.exists(dest):
            continue
        url = f"{FASHION_MNIST_MIRROR}/{fname}"
        urllib.request.urlretrieve(url, dest)


def _ensure_fashion_mnist(data_root: str, download: bool):
    """Return (train, test) FashionMNIST datasets, using a mirror fallback.

    Tries torchvision's normal path first; if the download fails (blocked
    mirror), fetches from the GitHub mirror and retries with download disabled.
    """
    tfm = transforms.ToTensor()
    try:
        train = datasets.FashionMNIST(data_root, train=True, download=download, transform=tfm)
        test = datasets.FashionMNIST(data_root, train=False, download=download, transform=tfm)
        return train, test
    except Exception:
        if not download:
            raise
        raw_dir = os.path.join(data_root, "FashionMNIST", "raw")
        _download_fashion_mnist_mirror(raw_dir)
        train = datasets.FashionMNIST(data_root, train=True, download=False, transform=tfm)
        test = datasets.FashionMNIST(data_root, train=False, download=False, transform=tfm)
        return train, test


def get_fashion_mnist_oneclass(
    normal_class: int = 0,
    data_root: str = DEFAULT_DATA_ROOT,
    batch_size: int = 128,
    anomaly_test_per_class: int | None = None,
    download: bool = True,
    seed: int = 0,
):
    """Build a one-class anomaly-detection split from Fashion-MNIST.

    Training set  : only images of ``normal_class`` (labelled 0).
    Test set      : all ``normal_class`` test images (label 0) plus images from
                    every other class (label 1).

    Args:
        normal_class: which Fashion-MNIST class is treated as "normal" (0-9).
        anomaly_test_per_class: cap on anomaly images sampled per other class
            (keeps the test set balanced-ish and fast). None = use all.
    Returns:
        (train_loader, test_loader). Tensors are (B, 1, 28, 28), float in [0, 1].
    """
    g = torch.Generator().manual_seed(seed)
    train_full, test_full = _ensure_fashion_mnist(data_root, download)

    # --- training: normal class only ---
    train_x = train_full.data[train_full.targets == normal_class].float() / 255.0
    train_x = train_x.unsqueeze(1)  # (N,1,28,28)
    train_y = torch.full((len(train_x),), LABEL_NORMAL, dtype=torch.long)
    train_ds = TensorDataset(train_x, train_y)

    # --- test: normal (label 0) + others (label 1) ---
    tx, ty = test_full.data.float() / 255.0, test_full.targets
    tx = tx.unsqueeze(1)
    normal_mask = ty == normal_class
    test_x = [tx[normal_mask]]
    test_y = [torch.full((int(normal_mask.sum()),), LABEL_NORMAL, dtype=torch.long)]
    for c in range(10):
        if c == normal_class:
            continue
        idx = torch.where(ty == c)[0]
        if anomaly_test_per_class is not None:
            perm = torch.randperm(len(idx), generator=g)[:anomaly_test_per_class]
            idx = idx[perm]
        test_x.append(tx[idx])
        test_y.append(torch.full((len(idx),), LABEL_ANOMALY, dtype=torch.long))
    test_ds = TensorDataset(torch.cat(test_x), torch.cat(test_y))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_mvtec(category: str, data_root: str = "./data/mvtec_anomaly_detection",
              batch_size: int = 32, image_size: int = 256):
    """Loader for one MVTec AD category (e.g. 'bottle', 'cable', 'screw').

    MVTec is not auto-downloadable (license click-through). See
    data_cards/mvtec_ad.md for the download link and expected folder layout::

        data/mvtec_anomaly_detection/<category>/train/good/*.png   # normal only
        data/mvtec_anomaly_detection/<category>/test/good/*.png    # normal
        data/mvtec_anomaly_detection/<category>/test/<defect>/*.png # anomalies

    Returns (train_loader, test_loader) with the same 0/1 label convention.
    """
    from torchvision.datasets import ImageFolder  # local import: optional path

    root = os.path.join(data_root, category)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"MVTec category not found at {root!r}. Download it first — see "
            "data_cards/mvtec_ad.md. For development use get_fashion_mnist_oneclass()."
        )

    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_ds = ImageFolder(os.path.join(root, "train"), transform=tfm)  # only 'good'
    test_raw = ImageFolder(os.path.join(root, "test"), transform=tfm)
    good_idx = test_raw.class_to_idx.get("good")

    # Remap test labels: 'good' -> 0, every defect class -> 1.
    test_x, test_y = [], []
    loader = DataLoader(test_raw, batch_size=batch_size, shuffle=False)
    for x, y in loader:
        test_x.append(x)
        test_y.append((y != good_idx).long())
    test_ds = TensorDataset(torch.cat(test_x), torch.cat(test_y))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
