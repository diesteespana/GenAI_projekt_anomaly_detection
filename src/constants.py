"""Project-wide constants."""

from __future__ import annotations

# Fashion-MNIST class index -> human-readable name.
FASHION_MNIST_CLASSES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Boot",
]

# Label convention used everywhere in the project.
LABEL_NORMAL = 0
LABEL_ANOMALY = 1

# Default filesystem locations.
DEFAULT_DATA_ROOT = "./data"
DEFAULT_RESULTS_DIR = "results"

# GitHub mirror used as an automatic fallback when the torchvision download
# mirror is unreachable (see src/data.py).
FASHION_MNIST_MIRROR = (
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion"
)
FASHION_MNIST_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]
