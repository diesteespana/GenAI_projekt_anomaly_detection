# Guide for Person 3 — the Adversarial Autoencoder (AAE)

Your file is `src/models/aae.py`. Good news: like the VAE, it already runs. Your
job is to *understand* it, run a couple of experiments, and write the AAE parts
of the report. This guide moves in small steps you can run and see. You do **not**
need heavy math to follow it.

## Where the AAE sits in our project
Our whole project is one story: three autoencoders that differ only in how they
tidy up their "latent space" (the short list of numbers the encoder squeezes each
image into).

```
AE   → no tidying           (baseline)
VAE  → tidy with a KL term   (Person 2)
AAE  → tidy with a game      (you)
```

"Tidying" means forcing those latent numbers to follow a simple bell-curve shape
(a Gaussian). A messy latent space lets the model accidentally reconstruct weird
inputs too well; a tidy one makes anomalies stand out. The VAE tidies with a
formula (the KL term). The **AAE tidies with a game** instead — and that game is
your topic.

## What an AAE is, in plain language
The AAE has three parts:
- an **encoder** (image → latent code), same as the AE,
- a **decoder** (latent code → rebuilt image), same as the AE,
- a **latent discriminator** — a small referee network.

The referee is shown two kinds of latent codes: fake-looking ones the encoder
produced, and "ideal" ones drawn from a perfect bell curve. Its job is to tell
them apart. The encoder's *extra* job is to fool the referee — to produce codes
so bell-curve-like that the referee can't tell. When the encoder wins, its latent
space has been tidied into the bell-curve shape, without any KL formula. That's
the whole trick: **an adversarial game replaces the VAE's KL term.**

Anomaly score: exactly like the AE and VAE — reconstruction error. Keeping the
score identical is what makes the three-way comparison fair.

## Checkpoint 0 — Run it as-is
```bash
python scripts/run_experiment.py --model aae --epochs 20
```
You'll see three numbers per epoch: `recon` (should fall), `d_loss` (the referee),
and `g_loss` (the encoder fooling the referee). Then an AUROC. It already works —
now understand and improve it.

## Checkpoint 1 — Read and label the code
Open `aae.py` and find these; write a one-line comment in your own words above
each (this is how you learn it):
- `_AAENet` → the encoder + decoder (the reconstruction path).
- `_LatentDiscriminator` → the referee (a small MLP on the latent vector).
- In `fit`, the **three phases**: (1) reconstruction, (2) discriminator update,
  (3) encoder-fools-discriminator update. Each has its own optimizer
  (`opt_ae`, `opt_d`, `opt_g`).
- `_sample_prior` → draws the "ideal" bell-curve codes.
Re-run Checkpoint 0 to confirm nothing broke.

## Checkpoint 2 — Understand the three losses (write about these)
For each batch the code does three small updates:
1. **Reconstruction:** encoder+decoder rebuild the image; minimize `|x - x̂|²`.
2. **Discriminator:** show the referee real prior codes (label 1) and encoder
   codes (label 0); it learns to tell them apart.
3. **Encoder as generator:** run the encoder codes through the referee again, but
   now push them toward "looks real" (label 1) — this is what tidies the latent
   space toward the prior.
In the report, explain why step 3 is the adversarial version of the VAE's KL
term (both pull `q(z)` toward the prior — one by a formula, one by a game).

## Checkpoint 3 — One ablation (your small experiment)
Pick one knob, run a few values, record AUROC, and make a tiny table. Options:
- **Prior scale** `AAEAD(prior_std=...)`: try 0.5, 1.0, 2.0.
- **Latent size** `AAEAD(latent_dim=...)`: try 16, 32, 64.
- **Score type** (optional, more advanced): compare reconstruction error vs a
  latent-based score (there's a `TODO` marker in `anomaly_score` pointing at this).

A quick way to run a value without editing files:
```bash
python scripts/run_experiment.py --model aae --epochs 20   # default
# then edit the default in aae.py or ask Nicolás to add a CLI flag
```

## What to expect (so you don't panic)
Adversarial training wobbles a little — `d_loss` and `g_loss` push against each
other rather than dropping smoothly. That's normal and *much gentler* than an
image-space GAN, because here the game is only over the small latent vector, not
whole images. If `recon` keeps falling, the model is fine. If AUROC is near 0.5,
train longer or lower the adversarial learning rate (`adv_lr`).

## What you hand back
1. Finished understanding of `aae.py` (your comments) + your ablation results.
2. Your AUROC/PR-AUC and the loss curve (`results/.../aae_loss.png`).
3. The report sections marked `[P3]` in `report/main.tex`: the AAE method
   subsection (3.4, referencing the architecture figure), the AAE results
   subsection (5.3), one Related-Work sentence, and one Discussion sentence.
   Keep notation consistent with Section 3.1.

## If you get stuck
- AUROC near 0.5 → check `recon` actually fell; lower `adv_lr`; train longer.
- Loss becomes NaN → learning rate too high; lower `lr`/`adv_lr`.
- Shapes error → print `x.shape` (should be `(batch, 1, 28, 28)`) and the latent
  shape after `self.net.encode(x)`.
- Always re-run Checkpoint 0's command after a change to catch breakage early.

Recommended skim (optional, for the write-up): Makhzani et al. 2015 (Adversarial
Autoencoders); Goodfellow et al. 2014 (the adversarial idea, intro only).
