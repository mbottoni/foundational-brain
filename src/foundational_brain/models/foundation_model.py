"""Full model: Encoder -> LatentRNN (dynamics) -> Decoder.

Self-supervised objectives are computed by the training loop (see
``foundational_brain.training``); this module wires the components together and
exposes the tensors those objectives need.
"""

from __future__ import annotations

import torch
from torch import nn

from .decoder import Decoder
from .encoder import Encoder
from .latent_rnn import LatentRNN


class FoundationModel(nn.Module):
    def __init__(
        self,
        n_regions: int,
        latent_dim: int = 256,
        encoder_hidden: list[int] | None = None,
        decoder_hidden: list[int] | None = None,
        variational: bool = False,
        rnn_hidden: int = 512,
        rnn_layers: int = 2,
        rnn_type: str = "gru",
        rnn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.variational = variational
        self.latent_dim = latent_dim
        self.encoder = Encoder(n_regions, latent_dim, encoder_hidden, variational)
        self.decoder = Decoder(latent_dim, n_regions, decoder_hidden)
        self.latent_rnn = LatentRNN(
            latent_dim, rnn_hidden, rnn_layers, rnn_type, rnn_dropout
        )
        # Learned stand-in for a masked timepoint. A zero vector would be
        # ambiguous with a genuinely near-zero latent, so the RNN could not
        # tell "masked" from "quiet".
        self.mask_token = nn.Parameter(torch.zeros(latent_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def encode(self, x: torch.Tensor):
        out = self.encoder(x)
        return out[0] if self.variational else out

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
        """x: (B, T, n_regions); mask: (B, T) bool, True where masked.

        Returns the tensors the SSL losses need. ``z`` is always the *clean*
        encoding of every frame — it is the target for masked modelling, so it
        must not itself be corrupted; only the RNN's input is masked.
        """
        enc = self.encoder(x)
        if self.variational:
            z, mu, logvar = enc
        else:
            z, mu, logvar = enc, None, None

        z_in = z
        if mask is not None:
            m = mask.unsqueeze(-1).to(z.dtype)
            z_in = z * (1.0 - m) + self.mask_token.expand_as(z) * m

        z_next_pred, _, feats = self.latent_rnn(z_in)  # predicted next latent per step
        x_recon = self.decoder(z)                      # reconstruct current frame
        x_next_pred = self.decoder(z_next_pred)        # forecast next frame

        return {
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "feats": feats,
            "z_next_pred": z_next_pred,
            "x_recon": x_recon,
            "x_next_pred": x_next_pred,
        }
