# A Comparative Study of Autoencoder-Based Generative Models for One-Class Anomaly Detection

Final project for the Generative Artificial Intelligence seminar at the University of Oldenburg.

This repository implements and compares three autoencoder-based models for one-class image anomaly detection:

- **Convolutional Autoencoder (AE)** — baseline without explicit latent-space regularization.
- **Variational Autoencoder (VAE)** — regularizes the latent representation toward a Gaussian prior using KL divergence.
- **Adversarial Autoencoder (AAE)** — matches the encoded latent distribution to a Gaussian prior through a latent discriminator.

The main research question is:

> Does regularizing the latent space improve reconstruction-based one-class anomaly detection, and does the regularization mechanism matter?

All models are trained only on normal samples. At test time, inputs with larger reconstruction errors receive higher anomaly scores.

---

## Methodology

The comparison is designed to isolate the effect of latent-space regularization.

All models use:

- The same Fashion-MNIST one-class data split.
- The same label convention.
- Comparable convolutional encoder-decoder components.
- The same evaluation pipeline.
- Mean squared reconstruction error as the main anomaly score.
- AUROC and PR-AUC as the main evaluation metrics.
- Best-F1 as an additional descriptive metric.

Label convention:

```text
0 = normal
1 = anomaly
```

A larger anomaly score means that the sample is considered more anomalous.

For the main comparison, Fashion-MNIST class `0` (`T-shirt/top`) is treated as the normal class. The models are trained only on T-shirt images, while the remaining Fashion-MNIST classes are treated as anomalies during testing.

---

## Models

### Autoencoder

The Autoencoder learns a deterministic mapping:

```text
input → encoder → latent representation → decoder → reconstruction
```

It is trained by minimizing reconstruction error on normal images.

At test time, the per-image mean squared reconstruction error is used as the anomaly score.

### Variational Autoencoder

The VAE maps each input to a probabilistic latent representation defined by a mean and a log-variance.

It is trained using:

- A reconstruction term.
- A KL-divergence regularization term.
- The reparameterization trick.

For the main comparison, the VAE anomaly score is the deterministic reconstruction MSE obtained by decoding the posterior mean. Additional scoring methods are also implemented:

- `mse`
- `mc_mse`
- `reconstruction_probability`

### Adversarial Autoencoder

The AAE combines an autoencoder with a discriminator operating in latent space.

The encoder-decoder is optimized using:

```text
reconstruction loss + weighted adversarial encoder loss
```

The discriminator is trained to distinguish:

- Samples drawn from the Gaussian prior as real.
- Encoded latent vectors as fake.

The AAE uses mean squared reconstruction error as its anomaly score.

---

## Datasets

### Fashion-MNIST one-class

Fashion-MNIST is the dataset used for the reported experiments.

The loader:

- Downloads the dataset automatically.
- Uses a fallback mirror when necessary.
- Trains only on the selected normal class.
- Creates a mixed test set containing normal and anomalous samples.

The main comparison uses:

```text
normal class = 0
batch size = 128
epochs = 20
seed = 0
anomaly samples per non-normal class = 200
latent dimension = 32
```

### MVTec AD

A preliminary MVTec AD data loader and dataset documentation are included.

However, the current architecture and the reported experiments were validated on Fashion-MNIST. Full architectural adaptation and experimental evaluation on MVTec AD remain future work.

---

## Repository structure

```text
.
├── README.md
├── REPRODUCIBILITY.md
├── requirements.txt
├── LICENSE
├── Makefile
│
├── src/
│   ├── constants.py
│   ├── data.py
│   ├── evaluation.py
│   ├── metrics.py
│   ├── utils.py
│   ├── visualize.py
│   └── models/
│       ├── __init__.py
│       ├── base.py
│       ├── blocks.py
│       ├── autoencoder.py
│       ├── vae.py
│       └── aae.py
│
├── scripts/
│   ├── download_data.py
│   ├── run_experiment.py
│   ├── run_comparison.py
│   ├── run_baseline_all_classes.py
│   ├── run_vae_ablation.py
│   └── make_diagrams.py
│
├── tests/
│   ├── conftest.py
│   ├── test_metrics.py
│   ├── test_models.py
│   ├── test_smoke.py
│   └── test_utils.py
│
├── docs/
│   └── 01_how_the_code_works.md
│
└── data_cards/
    ├── fashion_mnist.md
    └── mvtec_ad.md
```

The `results/` directory is created automatically when experiments are executed.

---

## Installation

Python 3.10–3.12 is recommended.

### Linux or macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Fashion-MNIST can be downloaded in advance with:

```bash
python scripts/download_data.py
```

It will also be downloaded automatically when an experiment is executed for the first time.

---

## Running experiments

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

### Run all registered models

```bash
python scripts/run_experiment.py \
  --model all \
  --normal-class 0 \
  --epochs 20 \
  --batch-size 128 \
  --anomaly-per-class 200 \
  --seed 0 \
  --results-dir results/comparison
```

### Integrated comparison

This command trains all registered models and generates an overlay ROC curve and a comparison table:

```bash
python scripts/run_comparison.py \
  --normal-class 0 \
  --epochs 20 \
  --batch-size 128 \
  --anomaly-per-class 200 \
  --seed 0 \
  --results-dir results/comparison
```

### Autoencoder baseline across all classes

```bash
python scripts/run_baseline_all_classes.py \
  --epochs 15 \
  --batch-size 128 \
  --anomaly-per-class 200 \
  --seed 0 \
  --results-dir results/baseline_all_classes
```

The AE baseline obtains a mean AUROC of approximately `0.90` across the ten Fashion-MNIST normal-class configurations.

### VAE beta ablation

```bash
python scripts/run_vae_ablation.py \
  --epochs 20 \
  --normal-class 0 \
  --seed 0 \
  --results-dir results/vae_ablation
```

The AAE latent-dimension ablation was performed using three independent executions with latent dimensions `16`, `32`, and `64`, while keeping the remaining settings fixed.

See `REPRODUCIBILITY.md` for the complete experimental procedure.

---

## Generated outputs

Each experiment generates files such as:

```text
<tag>_metrics.json
<tag>_roc.png
<tag>_pr.png
<tag>_score_hist.png
<tag>_loss.png
```

The metrics JSON contains:

- AUROC
- PR-AUC
- Best-F1
- Best threshold
- Number of normal samples
- Number of anomalous samples

The integrated comparison additionally generates:

```text
comparison.md
comparison_summary.json
overlay_roc.png
comparison_bars.png
```

Best-F1 is obtained by selecting the threshold that maximizes F1 on the labelled test set. It should therefore be interpreted as an oracle descriptive operating point. AUROC and PR-AUC are the main comparison metrics.

---

## Testing

Run the fast tests with:

```bash
pytest -q -m "not slow"
```

or:

```bash
make test
```

Run all tests, including the integration smoke test, with:

```bash
make test-all
```

The smoke test may download Fashion-MNIST and train a model for one epoch.

The test suite checks:

- Metric calculations.
- Model interfaces.
- Output shapes.
- Reproducible seeding.
- End-to-end experiment execution.

---

## Adding another model

1. Create a new model file under `src/models/`.
2. Subclass `AnomalyModel`.
3. Implement:

```python
fit(train_loader, epochs)
```

and:

```python
anomaly_score(x) -> np.ndarray
```

Higher anomaly scores must indicate more anomalous samples.

4. Reuse the shared components from `src/models/blocks.py` when appropriate.
5. Import the new model and add it to `REGISTRY` in `src/models/__init__.py`.
6. Run it with:

```bash
python scripts/run_experiment.py --model <model_name>
```

The existing AAE can be executed with:

```bash
python scripts/run_experiment.py --model aae
```

See `docs/01_how_the_code_works.md` for a walkthrough of the project architecture.

---

## Team contributions

| Team member | Main contributions |
|---|---|
| **Nicolás Dieste España** | Shared framework, data loading, evaluation pipeline, model interface, shared convolutional blocks, Autoencoder baseline, reproducibility, integration, and baseline experiments |
| **Claudio Esteban Arámbula Delgado** | Variational Autoencoder theory and implementation, ELBO and KL-divergence analysis, VAE anomaly-scoring methods, beta ablation, and interpretation of VAE results |
| **Andrés Campoy Sebastián** | Adversarial Autoencoder theory and implementation, latent discriminator and prior-matching analysis, AAE experiments, latent-dimension ablation, and interpretation of AAE results |

All group members contributed to the final comparison, discussion, report review, and presentation preparation.

---

## Main conclusion

The three models achieve strong anomaly-detection performance on the Fashion-MNIST one-class task.

The experiments show that stronger latent-space regularization does not automatically improve anomaly detection when all models are evaluated mainly through pixel-level reconstruction error.

The simpler Autoencoder remains a strong baseline, while the VAE and AAE provide more structured latent representations and competitive anomaly-detection performance.

Future work includes:

- Multi-seed experiments with uncertainty estimates.
- Latent-aware anomaly scores.
- Pixel-level anomaly localization.
- Full adaptation to MVTec AD.
- Evaluation on more realistic industrial defects.

---

## License

The project code is distributed under the license included in the `LICENSE` file.
