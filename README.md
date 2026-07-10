# A Comparative Study of Autoencoder-Based Generative Models for One-Class Anomaly Detection

GenAI seminar final project (University of Oldenburg). We compare three families
of generative models on the same one-class anomaly-detection task, scored with
identical metrics: a **Convolutional Autoencoder** (baseline), a **Variational
Autoencoder**, and an **Adversarial Autoencoder** (AAE).

These three form a spectrum of increasingly explicit latent-space
regularization: the AE imposes no constraint on its latent codes, the VAE adds a
KL term toward a Gaussian prior, and the AAE matches the latent to that prior with
an adversarial game. Each model trains on *normal* data only and flags test
samples that reconstruct poorly (higher score = more anomalous). Because every
model implements the same interface and is scored by reconstruction error, one
shared evaluation harness scores them identically, so the comparison isolates the
latent-regularization strategy.

## Datasets
- **Fashion-MNIST one-class** — light, fully reproducible; used for development
  and as an ablation. Auto-downloaded (with a GitHub-mirror fallback).
- **MVTec AD** — a preliminary loader is included. The current
  architecture and reported experiments are validated on Fashion-MNIST;
  full MVTec adaptation wan't done at the end.

Label convention everywhere: `0 = normal`, `1 = anomaly`.

## Repository layout
```
src/
  constants.py       # single source of truth (class names, label convention, paths)
  utils.py           # set_seed, get_device, JSON/plot helpers            [P1]
  data.py            # one-class loaders + auto mirror-fallback download   [P1]
  metrics.py         # pure metric functions (AUROC, PR-AUC, best-F1)      [P1]
  evaluation.py      # scoring harness: metrics + plots + loss curves      [P2]
  visualize.py       # reconstruction grids, overlay ROC, per-class bars   [P3]
  models/
    base.py          # AnomalyModel interface every model implements       [P1]
    blocks.py        # shared conv encoder/decoder (no duplication)        [P1]
    autoencoder.py   # conv AE baseline                                    [P1]
    vae.py # VAE — KL-regularized probabilistic latent model               [P2]
    aae.py           # AAE — adversarially-regularized latent (working)     [P3]
scripts/
  download_data.py         # fetch Fashion-MNIST (cross-platform)
  run_experiment.py        # train + eval one model (or --model all)
  run_comparison.py        # train all runnable models + overlay ROC + table
  run_baseline_all_classes.py  # AE across all 10 classes -> baseline table
data_cards/          # dataset documentation (graded)
report/main.tex      # LaTeX paper skeleton (compile on Overleaf)
docs/                # onboarding + per-teammate guides
tests/               # fast unit tests + slow integration smoke test
results/             # metrics JSON + figures land here
```

## Quick start
```bash
make setup      # pip install -r requirements.txt   (or run the pip line directly)
make data       # download Fashion-MNIST (optional; happens on first run anyway)
make test       # fast unit tests (no data, no training)
python scripts/run_experiment.py --model autoencoder --epochs 20
make compare    # train all runnable models + overlay ROC -> results/comparison/
```
No `make`? The Makefile targets are one-liners; see it for the exact commands.

Sanity check: the AE baseline reaches **AUROC ≈ 0.90** across all Fashion-MNIST
classes (mean of 10, 15 epochs each; see `results/baseline_all_classes/`).

## Testing & CI
- `make test` runs fast, synthetic unit tests (metrics, model interface, seeding).
- `make test-all` also runs the slow integration smoke test (downloads data,
  trains one epoch).

## Team split
| Person | Owns |
|---|---|
| **Person 1 (Nicolás)** | shared framework (data, metrics, evaluation, interface, blocks, visualization), AE baseline, reproducibility + data cards, integration, paper assembly (Intro / Related Work / Results & Discussion) |
| **Person 2 (Claudio)** | VAE: theory (ELBO, reparameterization, KL), reconstruction-probability scoring, β-ablation, paper sections |
| **Person 3 (Andrés)** | AAE: theory (latent discriminator, adversarial latent regularization, prior matching), experiments/ablations, paper sections |

## Adding another model
1. Subclass `AnomalyModel` in your file under `src/models/`.
2. Implement `fit(train_loader, epochs)` and `anomaly_score(x) -> np.ndarray`
   (higher = more anomalous). Reuse the shared `blocks.py` backbone.
3. It's already in the registry — run `python scripts/run_experiment.py --model aae`.

See `docs/` for a beginner-friendly walkthrough and a step-by-step guide per model.
