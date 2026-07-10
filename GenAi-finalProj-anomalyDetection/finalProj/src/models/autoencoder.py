"""Convolutional Autoencoder baseline (Person 1).

Standard encoder-decoder trained to reconstruct normal images. Anomalies
weren't seen in training so they reconstruct badly -> high error -> flagged.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .base import AnomalyModel
from .blocks import ConvDecoder, ConvEncoder


class _ConvAE(nn.Module):
    """28x28 grayscale conv autoencoder with a small bottleneck."""

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoencoderAD(AnomalyModel):
    name = "autoencoder"

    def __init__(self, latent_dim: int = 32, lr: float = 1e-3, device="cpu"):
        super().__init__(device)
        self.net = _ConvAE(latent_dim).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def fit(self, train_loader, epochs: int = 20) -> dict:
        self.net.train()
        history = {"loss": []}
        for ep in range(epochs):
            running, n = 0.0, 0
            for x, _ in train_loader:
                x = x.to(self.device)
                self.opt.zero_grad()
                recon = self.net(x)
                loss = self.criterion(recon, x)
                loss.backward()
                self.opt.step()
                running += loss.item() * x.size(0)
                n += x.size(0)
            epoch_loss = running / n
            history["loss"].append(epoch_loss)
            print(f"[autoencoder] epoch {ep + 1}/{epochs}  loss={epoch_loss:.5f}")
        return history

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> np.ndarray:
        self.net.eval()
        x = x.to(self.device)
        recon = self.net(x)
        # per-sample mean squared reconstruction error
        err = ((recon - x) ** 2).flatten(1).mean(dim=1)
        return err.cpu().numpy()
