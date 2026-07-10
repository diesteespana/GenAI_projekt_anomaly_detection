# Guide for Person 2 — the VAE

Your file is `src/models/vae.py`. Good news: the structure is already there and
it already runs. Your job is to *understand* it, then add the one piece that
makes a VAE special for anomaly detection, plus an experiment. Work through the
checkpoints in order — each one is something you can run and see.

## What a VAE is, in plain language
A normal autoencoder squeezes a picture into a short list of numbers (the
"latent code"), then rebuilds it. A **Variational** Autoencoder does the same
but with a twist: instead of producing one fixed code, it produces a small
*cloud* of possible codes (a mean and a spread). It then picks a random point in
that cloud and rebuilds from it. Training pushes those clouds to (a) rebuild the
picture well and (b) stay close to a simple, tidy shape (a standard bell curve).

Why this helps anomaly detection: a VAE learns a smooth, well-organized "map" of
what normal looks like. Abnormal inputs land in odd parts of the map and rebuild
poorly, *and* we can ask "how probable is this picture?" — a richer signal than
plain error.

Two ideas you'll write about in the report (don't worry, they're short):
- **ELBO** — the training objective. It's just two parts added together:
  "rebuild it accurately" + "keep the cloud tidy." The tidiness part is the
  *KL term*. Both are already coded in `_elbo_loss`.
- **Reparameterization trick** — a small algebra trick that lets us pick a
  random point from the cloud while still being able to train. It's the line
  `z = mu + eps * std`. Already coded in `reparameterize`.

> Note: the encoder/decoder now come from `src/models/blocks.py` (shared with the
> AE and AAE so all three have the same backbone). You don't need to touch
> `blocks.py`; the methods you care about — `encode`, `reparameterize`, `decode`,
> `_elbo_loss` — are still right here in `vae.py`.

## Checkpoint 0 — Run it as-is
```bash
python scripts/run_experiment.py --model vae --epochs 10
```
You should see `loss`, `recon`, and `kld` numbers, all trending down, then an
AUROC. It already works using simple reconstruction error. Now you'll improve and
understand it.

## Checkpoint 1 — Read and label the code
Open `vae.py`. Find these and write a one-line comment in your own words above
each (this is how you learn it):
- `encode` → produces `mu` and `logvar` (the cloud's center and spread).
- `reparameterize` → picks a random point in the cloud.
- `decode` → rebuilds the picture from that point.
- `_elbo_loss` → reconstruction term + KL term.
Run Checkpoint 0 again to confirm you didn't break anything (comments can't, but
it builds the habit).

## Checkpoint 2 — Your main contribution: reconstruction-probability score
Right now `anomaly_score` uses plain reconstruction error (copied from the AE).
The proper VAE score is the **reconstruction probability**: instead of rebuilding
once, sample the cloud several times, rebuild each time, and average how poorly
the picture is reconstructed across samples. Averaging over samples uses the
VAE's uncertainty, which is the whole point of a VAE.

Here is the shape of what to write (fill in the loop):
```python
@torch.no_grad()
def anomaly_score(self, x, n_samples: int = 10):
    self.net.eval()
    x = x.to(self.device)
    mu, logvar = self.net.encode(x)
    errors = []
    for _ in range(n_samples):
        z = self.net.reparameterize(mu, logvar)   # a fresh random point each time
        recon = self.net.decode(z)
        err = ((recon - x) ** 2).flatten(1).mean(dim=1)   # per-picture error
        errors.append(err)
    return torch.stack(errors).mean(dim=0).cpu().numpy()   # average over samples
```
Replace the current body of `anomaly_score` with this. Then:
```bash
python scripts/run_experiment.py --model vae --epochs 20
```
Compare the AUROC to Checkpoint 0. Write down both numbers — that comparison
(plain error vs reconstruction probability) is a result for the paper.

## Checkpoint 3 — One ablation (a small experiment)
The model has a `beta` knob (in `VAEAD(beta=...)`) that controls how strongly we
enforce "keep the cloud tidy." Run the model with a few values, e.g.:
```bash
# edit the registry call or add a quick script; ask Nicolás if unsure how
# try beta = 0.5, 1.0, 2.0 and record AUROC for each
```
Make a tiny table of beta vs AUROC. This is your "ablation study" — graders love
these because they show you understood the knob, not just ran defaults.

## What you hand back
1. Finished `vae.py` (reconstruction-probability score working).
2. Three numbers: plain-error AUROC, reconstruction-probability AUROC, and your
   beta table.
3. The report sections marked `[P2]` in `report/main.tex`: the VAE method
   subsection (3.3), the VAE results subsection (5.2), one Related-Work sentence,
   and one Discussion sentence. Keep the math notation matching Section 3.1.

## If you get stuck
- AUROC near 0.5 → the score direction may be flipped or training didn't run;
  re-check the loss went down.
- An error mentioning shapes/sizes → print `x.shape` at the top of the method;
  it should be `(batch, 1, 28, 28)`.
- Always re-run Checkpoint 0's command after a change to catch breakage early.

Recommended skim (optional, for the write-up): Kingma & Welling 2014 (the VAE
paper, intro only); An & Cho 2015 (reconstruction probability).
