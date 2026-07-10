# 01 — How the Code Works (the mental model)

Before touching your model, understand how the pieces fit. The whole project is
built around **one simple agreement**: every model promises to do two things.
If your model keeps that promise, all the shared machinery (data, scoring, plots)
works for you automatically. You never rewrite that machinery.

## The agreement (the interface)
Open `src/models/base.py`. Every model is a class with two methods:

1. `fit(train_loader, epochs)` — **learn from normal data.** You're given the
   training pictures in batches; you update the model so it gets good at them.
2. `anomaly_score(x)` — **judge a batch of pictures.** Return one number per
   picture. **Higher = more abnormal.** That's the only rule about the number.

That's it. The autoencoder (`src/models/autoencoder.py`) is the worked example —
read it once, slowly. It's short, and yours will look similar in shape.

## The data (you don't change this)
`src/data.py` hands you two "loaders":
- `train_loader` — batches of **normal** pictures only (e.g. only T-shirts).
- `test_loader` — a **mix** of normal and abnormal pictures, each tagged with a
  label: `0 = normal`, `1 = anomaly`.

A "loader" is just something you loop over to get batches:
```python
for x, y in train_loader:
    # x = a batch of pictures, shape (batch, 1, 28, 28)
    # y = their labels (all 0 during training, since training is normal-only)
    ...
```
Pictures are grayscale `28x28`, with pixel values between 0 and 1.

## The scoring (you don't change this either)
`src/evaluation.py` takes your trained model and the `test_loader`, calls your
`anomaly_score` on everything, and computes:
- **AUROC** — "if I pick a random anomaly and a random normal, how often does the
  model score the anomaly higher?" 1.0 is perfect, 0.5 is random guessing.
- **PR-AUC** — similar idea, better when anomalies are rare.

It also saves plots automatically. So your job is *only* to make a good model;
the grading numbers come out for free.

## The data flow, end to end
```
   normal pictures ──► YOUR model.fit() ──► a trained model
                                                  │
   mixed test pictures ──► model.anomaly_score() ─┘──► scores (1 per picture)
                                                  │
                          evaluation.py compares scores to true labels
                                                  │
                                  AUROC, PR-AUC, and plots in results/
```

## How you run things
One command runs train + score + plots for any model:
```bash
python scripts/run_experiment.py --model vae --epochs 20      # Person 2
python scripts/run_experiment.py --model aae --epochs 20      # Person 3
python scripts/run_experiment.py --model all --epochs 20      # everyone, + comparison
```
Your model is already registered (`src/models/__init__.py`), so as soon as you
fill it in, these commands just work.

## What "training" actually does (intuition, no heavy math)
A model has thousands of internal numbers ("weights"), random at first. Training
shows it normal pictures over and over. Each time, it measures how wrong it was
(the "loss"), and nudges the weights to be a little less wrong next time. One
full pass over the data is an **epoch**. After enough epochs the loss stops
dropping much — the model has learned what normal looks like. You'll literally
watch the loss number shrink in the terminal; that's the model learning.

## Where you work
- **Person 2:** only `src/models/vae.py`. Guide: `guide_person2_vae.md`.
- **Person 3:** only `src/models/aae.py`. Guide: `guide_person3_aae.md`.

You will *not* edit `data.py`, `evaluation.py`, or `base.py`. If you think you
need to, ask first — usually there's an easier way that keeps the comparison fair.
