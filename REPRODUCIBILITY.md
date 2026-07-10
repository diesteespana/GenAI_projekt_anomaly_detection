# Reproducibility Guide

This guide describes how to reproduce the experiments reported in the project
from a clean machine.

## 1. Environment

- Python 3.10–3.12.
- CPU is sufficient for the reported Fashion-MNIST experiments.
- CUDA is detected automatically when available.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Minimum compatible dependency versions are listed in `requirements.txt`.

Reproducibility is handled centrally by `src/utils.py:set_seed`, which seeds
Python, NumPy, and PyTorch and configures deterministic CuDNN behaviour.
The experiment scripts use `--seed 0` by default. Small numerical differences
may still occur across different hardware or library versions.

## 2. Data

### Fashion-MNIST

Fashion-MNIST is downloaded automatically into `./data/` on the first run.
If the default torchvision source is unavailable, the loader attempts to use
the configured fallback mirror.

The dataset can also be downloaded in advance with:

```bash
python scripts/download_data.py
```

The reported experiments use a one-class setup:

- Training data: only the selected normal class.
- Test data: normal samples from that class and anomalous samples from the
  remaining classes.
- Label `0`: normal.
- Label `1`: anomaly.

### MVTec AD

A preliminary MVTec AD data loader and dataset documentation are included in
the repository. However, the current architecture and the reported experiments
were validated on Fashion-MNIST. Full adaptation and evaluation on MVTec AD
remain future work.

See `data_cards/mvtec_ad.md` for the expected folder structure.

## 3. Reported configuration

The main comparison uses:

- Normal class: `0` (T-shirt/top).
- Latent dimension: `32`.
- Batch size: `128`.
- Epochs: `20`.
- Random seed: `0`.
- Anomaly samples per non-normal test class: `200`.
- VAE beta: `1.0`.
- VAE anomaly score: deterministic reconstruction MSE.
- AAE adversarial weight: `1e-3`.

The AE all-class baseline uses 15 epochs.

Removing `--anomaly-per-class 200` uses the complete test set and may produce
slightly different results.

## 4. Main model comparison

The three reported models were executed independently.

### Autoencoder

```bash
python scripts/run_experiment.py \
  --model autoencoder \
  --normal-class 0 \
  --epochs 20 \
  --batch-size 128 \
  --anomaly-per-class 200 \
  --seed 0 \
  --results-dir results/final_comparison
```

### Variational Autoencoder

```bash
python scripts/run_experiment.py \
  --model vae \
  --normal-class 0 \
  --epochs 20 \
  --batch-size 128 \
  --anomaly-per-class 200 \
  --seed 0 \
  --beta 1.0 \
  --vae-score-method mse \
  --results-dir results/final_comparison
```

### Adversarial Autoencoder

```bash
python scripts/run_experiment.py \
  --model aae \
  --normal-class 0 \
  --epochs 20 \
  --batch-size 128 \
  --anomaly-per-class 200 \
  --seed 0 \
  --results-dir results/final_comparison
```

The integrated comparison, including an overlay ROC curve and comparison table,
can also be generated with:

```bash
python scripts/run_comparison.py \
  --epochs 20 \
  --normal-class 0 \
  --batch-size 128 \
  --anomaly-per-class 200 \
  --seed 0 \
  --results-dir results/comparison
```

## 5. AE baseline across all Fashion-MNIST classes

```bash
python scripts/run_baseline_all_classes.py \
  --epochs 15 \
  --batch-size 128 \
  --anomaly-per-class 200 \
  --seed 0 \
  --results-dir results/baseline_all_classes
```

The reported baseline result is approximately:

- Mean AUROC: `0.902 ± 0.061`.
- Mean PR-AUC: `0.939 ± 0.041`.

The exact values are stored in:

```text
results/baseline_all_classes/summary.json
```

## 6. VAE beta ablation

The VAE beta ablation can be reproduced with:

```bash
python scripts/run_vae_ablation.py \
  --epochs 20 \
  --normal-class 0 \
  --seed 0 \
  --results-dir results/vae_ablation
```

This experiment compares different values of the KL-divergence weight while
keeping the remaining settings fixed.

Use the arguments shown by the following command if the script exposes
additional beta-selection options:

```bash
python scripts/run_vae_ablation.py --help
```

## 7. AAE latent-dimension ablation

The AAE latent-dimension ablation was performed through three independent
executions with latent dimensions `16`, `32`, and `64`.

Before each execution, the default `latent_dim` value in the `AAEAD`
constructor in `src/models/aae.py` was changed to the corresponding value.
All remaining settings were kept fixed.

For `latent_dim = 16`:

```bash
python scripts/run_experiment.py \
  --model aae \
  --normal-class 0 \
  --epochs 20 \
  --batch-size 128 \
  --anomaly-per-class 200 \
  --seed 0 \
  --results-dir results/aae_latent16
```

For `latent_dim = 32`:

```bash
python scripts/run_experiment.py \
  --model aae \
  --normal-class 0 \
  --epochs 20 \
  --batch-size 128 \
  --anomaly-per-class 200 \
  --seed 0 \
  --results-dir results/aae_latent32
```

For `latent_dim = 64`:

```bash
python scripts/run_experiment.py \
  --model aae \
  --normal-class 0 \
  --epochs 20 \
  --batch-size 128 \
  --anomaly-per-class 200 \
  --seed 0 \
  --results-dir results/aae_latent64
```

After the ablation, the default latent dimension was restored to `32`.

## 8. Generated outputs

Each execution writes:

- `<tag>_metrics.json`: AUROC, PR-AUC, best-F1, best threshold, and sample counts.
- `<tag>_roc.png`: ROC curve.
- `<tag>_pr.png`: precision-recall curve.
- `<tag>_score_hist.png`: anomaly-score distributions.
- `<tag>_loss.png`: training-loss curves.

The integrated comparison additionally generates:

- `comparison.md`.
- `overlay_roc.png`.
- `comparison_bars.png`.

The `best-F1` threshold is selected using the labelled test set. Therefore,
best-F1 should be interpreted as an oracle descriptive operating point.
AUROC and PR-AUC are the main comparison metrics.

## 9. Tests

Run the fast unit tests with:

```bash
make test
```

or:

```bash
pytest -q -m "not slow"
```

Run all tests, including the integration smoke test, with:

```bash
make test-all
```

The smoke test may download Fashion-MNIST and train a model for one epoch.
