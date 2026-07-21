"""Spatial encoder: maps one fMRI frame (ROI vector) to a latent vector z_t."""

from __future__ import annotations

import torch
from torch import nn


class Encoder(nn.Module):
    """MLP encoder over parcellated ROI activations.

    Input:  (..., n_regions)
    Output: (..., latent_dim)   [+ (mu, logvar) if variational]
    """

    def __init__(
        self,
        n_regions: int,
        latent_dim: int,
        hidden: list[int] | None = None,
        variational: bool = False,
    ) -> None:
        super().__init__()
        hidden = hidden or [1024, 512]
        self.variational = variational

        layers: list[nn.Module] = []
        prev = n_regions
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.LayerNorm(h)]
            prev = h
        self.backbone = nn.Sequential(*layers)

        out_dim = latent_dim * 2 if variational else latent_dim
        self.head = nn.Linear(prev, out_dim)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor):
        h = self.head(self.backbone(x))
        if not self.variational:
            return h
        mu, logvar = h.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar
