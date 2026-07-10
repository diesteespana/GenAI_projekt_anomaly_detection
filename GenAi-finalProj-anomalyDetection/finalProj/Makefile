# Convenience targets. Run `make help` to list them.
PY ?= python

.PHONY: help setup data test test-all baseline compare clean

help:
	@echo "Targets:"
	@echo "  setup     Install dependencies (pip install -r requirements.txt)"
	@echo "  data      Download Fashion-MNIST (with mirror fallback)"
	@echo "  test      Fast unit tests (no data, no training)"
	@echo "  test-all  Full test suite incl. slow integration smoke test"
	@echo "  baseline  AE baseline across all 10 classes -> results/baseline_all_classes/"
	@echo "  compare   Train all runnable models + overlay ROC -> results/comparison/"
	@echo "  clean     Remove caches and generated __pycache__"

setup:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) scripts/download_data.py

test:
	$(PY) -m pytest -m "not slow"

test-all:
	$(PY) -m pytest

baseline:
	$(PY) scripts/run_baseline_all_classes.py --epochs 15

compare:
	$(PY) scripts/run_comparison.py --epochs 20

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
