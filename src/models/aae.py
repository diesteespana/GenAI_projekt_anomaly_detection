"""Adversarial Autoencoder anomaly detection

Idea:
  * The autoencoder learns to reconstruct normal images.
  * A latent discriminator regularizes encoded vectors z so they look like
    samples from a simple prior N(0, I).
  * At test time, anomalies should reconstruct worse, so the anomaly score is
    the per-sample reconstruction error. Higher score = more anomalous.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .base import AnomalyModel
from .blocks import ConvDecoder, ConvEncoder


class _AAENet(nn.Module):
    """28x28 grayscale adversarial autoencoder backbone."""

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


class _LatentDiscriminator(nn.Module):
    """Discriminator that separates prior samples from encoded latent vectors."""

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class AAEAD(AnomalyModel):
    """Adversarial Autoencoder for one-class anomaly detection."""

    name = "aae"

    def __init__(
        self,
        latent_dim: int = 32,
        lr: float = 1e-3,
        adv_weight: float = 1e-3,
        device="cpu",
    ):
        super().__init__(device)

        self.net = _AAENet(latent_dim).to(self.device)
        self.D = _LatentDiscriminator(latent_dim).to(self.device)

        self.opt_ae = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=lr)

        self.recon_loss = nn.MSELoss()
        self.adv_loss = nn.BCEWithLogitsLoss()
        self.adv_weight = adv_weight

    def fit(self, train_loader, epochs: int = 20) -> dict:
        history = {"loss": [], "recon": [], "adv": [], "disc": []}

        for ep in range(epochs):
            self.net.train()
            self.D.train()

            total = 0.0
            recon_total = 0.0
            adv_total = 0.0
            disc_total = 0.0
            n = 0

            for x, _ in train_loader:
                x = x.to(self.device)
                bs = x.size(0)

                # 1) Train the autoencoder and encoder adversarially.
                recon, z_fake = self.net(x)

                pred_fake_for_encoder = self.D(z_fake)
                loss_recon = self.recon_loss(recon, x)
                loss_adv = self.adv_loss(
                    pred_fake_for_encoder,
                    torch.ones_like(pred_fake_for_encoder),
                )

                loss_ae = loss_recon + self.adv_weight * loss_adv

                self.opt_ae.zero_grad()
                loss_ae.backward()
                self.opt_ae.step()

                # 2) Train the latent discriminator.
                with torch.no_grad():
                    z_fake_detached = self.net.encode(x)

                z_real = torch.randn_like(z_fake_detached)

                pred_real = self.D(z_real)
                pred_fake = self.D(z_fake_detached)

                loss_d = 0.5 * (
                    self.adv_loss(pred_real, torch.ones_like(pred_real))
                    + self.adv_loss(pred_fake, torch.zeros_like(pred_fake))
                )

                self.opt_d.zero_grad()
                loss_d.backward()
                self.opt_d.step()

                total += loss_ae.item() * bs
                recon_total += loss_recon.item() * bs
                adv_total += loss_adv.item() * bs
                disc_total += loss_d.item() * bs
                n += bs

            history["loss"].append(total / n)
            history["recon"].append(recon_total / n)
            history["adv"].append(adv_total / n)
            history["disc"].append(disc_total / n)

            print(
                f"[aae] epoch {ep + 1}/{epochs}  "
                f"loss={total / n:.5f}  "
                f"recon={recon_total / n:.5f}  "
                f"adv={adv_total / n:.5f}  "
                f"disc={disc_total / n:.5f}"
            )

        return history

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> np.ndarray:
        self.net.eval()

        x = x.to(self.device)
        recon, _ = self.net(x)

        err = ((recon - x) ** 2).flatten(1).mean(dim=1)
        return err.cpu().numpy()


