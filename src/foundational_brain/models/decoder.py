"""Spatial decoder: maps a latent vector z_t back to an fMRI frame (ROI vector)."""

from __future__ import annotations

import torch
from torch import nn


class Decoder(nn.Module):
    """MLP decoder mirroring the encoder.

    Input:  (..., latent_dim)
    Output: (..., n_regions)
    """

    def __init__(
        self,
        latent_dim: int,
        n_regions: int,
        hidden: list[int] | None = None,
    ) -> None:
        super().__init__()
        hidden = hidden or [512, 1024]

        layers: list[nn.Module] = []
        prev = latent_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.LayerNorm(h)]
            prev = h
        layers.append(nn.Linear(prev, n_regions))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
