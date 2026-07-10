"""Shared conv encoder/decoder for the AE, VAE and AAE (28x28 grayscale).

Same backbone for all three models so we're only comparing the latent
regularization, not different architectures.

28 -> 14 -> 7 spatial, 64 channels -> flat size 64*7*7 = 3136.
"""

from __future__ import annotations

import torch
import torch.nn as nn

BOTTLENECK_C = 64
BOTTLENECK_HW = 7
FEATURE_DIM = BOTTLENECK_C * BOTTLENECK_HW * BOTTLENECK_HW  # 3136


class ConvBody(nn.Module):
    """Two stride-2 conv layers + flatten. Output: (B, FEATURE_DIM)."""

    def __init__(self, in_ch: int = 1, act: type[nn.Module] = nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),   # 28 -> 14
            act(inplace=True) if act is not nn.LeakyReLU else nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),      # 14 -> 7
            act(inplace=True) if act is not nn.LeakyReLU else nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


class DeconvHead(nn.Module):
    """Linear -> reshape -> two transposed convs. Maps latent to a 28x28 image."""

    def __init__(self, latent_dim: int, out_ch: int = 1):
        super().__init__()
        self.fc = nn.Linear(latent_dim, FEATURE_DIM)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),   # 7 -> 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_ch, 3, stride=2, padding=1, output_padding=1),  # 14 -> 28
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, BOTTLENECK_C, BOTTLENECK_HW, BOTTLENECK_HW)
        return self.deconv(h)


class ConvEncoder(nn.Module):
    """ConvBody + a linear projection to a latent vector. Output: (B, latent_dim)."""

    def __init__(self, latent_dim: int, in_ch: int = 1):
        super().__init__()
        self.body = ConvBody(in_ch)
        self.fc = nn.Linear(FEATURE_DIM, latent_dim)

    def forward(self, x):
        return self.fc(self.body(x))


class ConvDecoder(DeconvHead):
    """Alias for DeconvHead, named symmetrically with ConvEncoder."""
    pass
