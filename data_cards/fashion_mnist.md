# Data Card — Fashion-MNIST (one-class anomaly setup)

## Source
Fashion-MNIST, Zalando Research. https://github.com/zalandoresearch/fashion-mnist
License: MIT.

## Original dataset
- 70,000 grayscale images, 28×28 pixels (60k train / 10k test).
- 10 balanced classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal,
  Shirt, Sneaker, Bag, Ankle boot.
- Drop-in replacement for MNIST; harder and more representative of real images.

## How we use it (one-class reformulation)
We turn classification into anomaly detection:
- **Train**: only images of a chosen `normal_class` (e.g. class 0). Label 0.
- **Test**: all test images of `normal_class` (label 0 = normal) plus images
  from the other nine classes (label 1 = anomaly). `--anomaly-per-class` caps how
  many anomalies are sampled per other class to keep runs fast.

## Preprocessing
- Pixel values scaled to [0, 1] (`ToTensor`).
- Shape (1, 28, 28); no augmentation (keeps reconstruction targets clean).

## Splits & determinism
- Train/test come from the official Fashion-MNIST split.
- Anomaly sampling uses a seeded RNG (`seed`, default 0) for reproducibility.

## Known limitations / ethical notes
- Clothing product images; no personal or sensitive data.
- "Anomaly" here is synthetic (other clothing classes), not a real defect —
  this is why we also report on MVTec AD, where anomalies are genuine defects.
- Class difficulty varies (e.g. Shirt vs T-shirt overlap), so AUROC depends on
  the chosen `normal_class`; we report across several to avoid cherry-picking.
