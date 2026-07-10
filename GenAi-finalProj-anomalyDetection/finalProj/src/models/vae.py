"""Variational Autoencoder for anomaly detection — PERSON 2's file.

This file implements the VAE contribution used in the project:
  * ELBO objective: reconstruction term + beta-scaled KL divergence.
  * Reparameterization trick: z = mu + sigma * eps.
  * VAE-specific anomaly scoring with Monte-Carlo reconstruction probability.

The encoder/decoder use the shared blocks in ``blocks.py`` so the VAE has the
same backbone as the AE and AAE. The only differences are the probabilistic
latent variables and the KL regularization term.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AnomalyModel
from .blocks import FEATURE_DIM, ConvBody, ConvDecoder


class _VAENet(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.body = ConvBody(in_ch=1)                 # shared backbone -> (B, FEATURE_DIM)
        self.fc_mu = nn.Linear(FEATURE_DIM, latent_dim)
        self.fc_logvar = nn.Linear(FEATURE_DIM, latent_dim)
        self.dec = ConvDecoder(latent_dim)            # shared decoder, outputs probabilities in [0, 1]

    def encode(self, x):
        h = self.body(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """Sample z while keeping the operation differentiable.

        Instead of sampling directly from N(mu, sigma^2), we sample eps from a
        standard normal and transform it: z = mu + sigma * eps. This is the
        reparameterization trick, which allows gradients to flow through mu and
        logvar during training.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAEAD(AnomalyModel):
    name = "vae"

    def __init__(
        self,
        latent_dim: int = 32,
        lr: float = 1e-3,
        beta: float = 1.0,
        score_method: str = "mse",
        mc_samples: int = 10,
        device="cpu",
    ):
        super().__init__(device)
        self.net = _VAENet(latent_dim).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.beta = beta
        self.score_method = score_method
        self.mc_samples = mc_samples
        self._valid_score_methods = {"mse", "mc_mse", "reconstruction_probability"}
        if self.score_method not in self._valid_score_methods:
            raise ValueError(
                f"Unknown VAE score_method={score_method!r}. "
                f"Choose from {sorted(self._valid_score_methods)}."
            )

    def _elbo_loss(self, recon, x, mu, logvar):
        """Negative ELBO used for training.

        Loss = reconstruction NLL + beta * KL(q_phi(z|x) || p(z)).
        The reconstruction term is Bernoulli negative log-likelihood implemented
        with binary cross-entropy because the decoder outputs pixel probabilities.
        """
        recon_loss = F.binary_cross_entropy(recon, x, reduction="sum") / x.size(0)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + self.beta * kld, recon_loss, kld

    def fit(self, train_loader, epochs: int = 20) -> dict:
        self.net.train()
        history = {"loss": [], "recon": [], "kld": []}
        for ep in range(epochs):
            tot = rec = kl = n = 0.0
            for x, _ in train_loader:
                x = x.to(self.device)
                self.opt.zero_grad()
                recon, mu, logvar = self.net(x)
                loss, r, k = self._elbo_loss(recon, x, mu, logvar)
                loss.backward()
                self.opt.step()
                bs = x.size(0)
                tot += loss.item() * bs
                rec += r.item() * bs
                kl += k.item() * bs
                n += bs
            history["loss"].append(tot / n)
            history["recon"].append(rec / n)
            history["kld"].append(kl / n)
            print(
                f"[vae beta={self.beta:g} score={self.score_method}] "
                f"epoch {ep + 1}/{epochs}  loss={tot / n:.4f}  "
                f"recon={rec / n:.4f}  kld={kl / n:.4f}"
            )
        return history

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor, n_samples: int | None = None) -> np.ndarray:
        """Return VAE anomaly scores; higher means more anomalous.

        Supported scoring methods:
          * ``mse``: deterministic reconstruction error using the posterior mean.
          * ``mc_mse``: Monte-Carlo average of reconstruction errors.
          * ``reconstruction_probability``: negative Monte-Carlo log reconstruction
            probability under a Bernoulli decoder. This is the VAE-specific score
            used for Person 2's comparison.
        """
        self.net.eval()
        x = x.to(self.device)
        n_samples = int(n_samples or self.mc_samples)

        mu, logvar = self.net.encode(x)

        if self.score_method == "mse":
            # Deterministic baseline: decode the posterior mean to avoid random scores.
            recon = self.net.decode(mu)
            err = ((recon - x) ** 2).flatten(1).mean(dim=1)
            return err.cpu().numpy()

        scores = []
        log_probs = []
        eps = 1e-6
        num_pixels = x[0].numel()

        for _ in range(n_samples):
            z = self.net.reparameterize(mu, logvar)
            recon = self.net.decode(z)

            if self.score_method == "mc_mse":
                err = ((recon - x) ** 2).flatten(1).mean(dim=1)
                scores.append(err)
            elif self.score_method == "reconstruction_probability":
                # Bernoulli log p_theta(x|z), summed over pixels.
                recon = recon.clamp(eps, 1.0 - eps)
                log_px_z = (x * torch.log(recon) + (1.0 - x) * torch.log(1.0 - recon)).flatten(1).sum(dim=1)
                log_probs.append(log_px_z)
            else:  # defensive guard; __init__ already validates.
                raise ValueError(f"Unknown VAE score method: {self.score_method}")

        if self.score_method == "mc_mse":
            return torch.stack(scores, dim=0).mean(dim=0).cpu().numpy()

        # log mean_l p(x|z_l) = logsumexp_l(log p(x|z_l)) - log(L)
        log_reconstruction_prob = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(n_samples)
        # Negative log-probability per pixel so larger = more anomalous.
        return (-log_reconstruction_prob / num_pixels).cpu().numpy()
