"""The foundation component: an RNN that models temporal dynamics in latent space.

Given a sequence of latents z_{1:T}, predict the next latent at each step. The RNN
hidden state is the transferable "brain dynamics" representation.
"""

from __future__ import annotations

import torch
from torch import nn


class LatentRNN(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        rnn_type: str = "gru",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[rnn_type.lower()]
        self.rnn = rnn_cls(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.readout = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z: torch.Tensor, hidden=None):
        """z: (B, T, latent_dim) -> (z_next_pred: (B, T, latent_dim), hidden, feats)."""
        feats, hidden = self.rnn(z, hidden)
        z_next = self.readout(feats)
        return z_next, hidden, feats
