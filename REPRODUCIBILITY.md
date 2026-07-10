# Reproducibility Guide

This guide lets a grader reproduce every reported number from a clean machine.

## 1. Environment
- Python 3.10–3.12.
- CPU is sufficient for Fashion-MNIST. A GPU helps for MVTec / AAE runs (the code
  auto-detects CUDA).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or: make setup
```

Exact versions used for the reported results are documented in `requirements.txt`.
Reproducibility is handled centrally by `src/utils.py:set_seed`, which seeds
Python, NumPy and PyTorch and enables deterministic algorithms. Every entry point
calls it with `--seed 0` by default, so splits and initialization are
deterministic across runs on the same machine.

## 2. Data

### Fashion-MNIST (automatic, with fallback)
The loader downloads Fashion-MNIST on first run into `./data/`. If the default
torchvision mirror is blocked, the loader **automatically** falls back to the
GitHub mirror — no manual step needed. To pre-fetch explicitly:

```bash
python scripts/download_data.py          # or: make data
```

(If you ever need to place the files by hand, drop the four `*-ubyte.gz` files in
`./data/FashionMNIST/raw/`; the loader detects them and skips downloading.)

### MVTec AD (manual)
See `data_cards/mvtec_ad.md` for the link and the exact folder layout expected by
`get_mvtec()`.

## 3. Reproduce the results
```bash
# AE baseline across all 10 normal classes -> results/baseline_all_classes/
python scripts/run_baseline_all_classes.py --epochs 15        # or: make baseline

# all runnable models + overlay ROC + comparison table -> results/comparison/
python scripts/run_comparison.py --epochs 20 --normal-class 0 # or: make compare

# a single model
python scripts/run_experiment.py --model autoencoder --epochs 20 --normal-class 0
```
Outputs written per model:
- `<tag>_metrics.json` — AUROC, PR-AUC, best-F1, best threshold, sample counts.
- `<tag>_roc.png`, `<tag>_pr.png`, `<tag>_score_hist.png`, `<tag>_loss.png`.
- `comparison.md`, `overlay_roc.png`, `comparison_bars.png` (comparison run).

## 4. Reported configuration
- normal_class = 0, batch_size = 128, epochs = 15 (baseline) / 20 (comparison),
  seed = 0.
- `--anomaly-per-class 200` caps anomaly test images per class for speed; remove
  it to use the full test set (numbers shift slightly).
- Baseline headline: mean AUROC 0.902 ± 0.061, mean PR-AUC 0.939 ± 0.041
  (`results/baseline_all_classes/summary.json`).

## 5. Tests
```bash
make test        # fast unit tests: metrics, model interface, seeding (no data)
make test-all    # also the slow integration smoke test (downloads data, 1 epoch)
```
CI (`.github/workflows/ci.yml`) runs the fast tests on every push.
