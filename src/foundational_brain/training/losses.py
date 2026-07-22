"""The self-supervised objectives.

Three losses, and one failure mode worth stating explicitly.

**Reconstruction** ties the latent to real signal: ``||D(E(x_t)) - x_t||^2``.

**Forecasting** is the dynamics objective. It is computed in *signal space* —
decode the RNN's predicted next latent and compare to the true next frame —
with the latent-space term as a secondary, stop-gradiented anchor. The reason
is collapse: a latent-only forecasting loss shares its target with the thing
being trained, so the encoder can drive the loss to zero by making ``z``
constant. Grounding the objective in ``x`` removes that shortcut, and detaching
the latent target removes what remains of it.

**Masked latent modelling** hides a fraction of timepoints from the RNN and
asks it to fill them in from context. Because the RNN is causal, "context"
means the past only — this is next-token prediction with gaps, not BERT.

All losses are computed on per-region z-scored input, so a value of 1.0 is
exactly the loss of predicting the mean.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def reconstruction_loss(x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Per-frame autoencoding error."""
    return F.mse_loss(x_recon, x)


def forecast_loss(
    x_next_pred: torch.Tensor,
    z_next_pred: torch.Tensor,
    x: torch.Tensor,
    z: torch.Tensor,
    latent_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Next-frame prediction, in signal space with a latent-space anchor.

    Step ``t`` of the RNN output predicts step ``t+1``, so the last step has no
    target and is dropped.
    """
    pred_x, true_x = x_next_pred[:, :-1], x[:, 1:]
    signal = F.mse_loss(pred_x, true_x)

    # detached target: the latent term must not be minimizable by collapsing z
    pred_z, true_z = z_next_pred[:, :-1], z[:, 1:].detach()
    latent = F.mse_loss(pred_z, true_z)

    total = signal + latent_weight * latent
    return total, {
        "forecast_signal": signal.detach().item(),
        "forecast_latent": latent.detach().item(),
    }


def masked_loss(
    x_next_pred: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Prediction error restricted to timepoints the RNN could not see.

    A masked step ``t`` is predicted by the RNN output at ``t-1``, which read
    only unmasked history. Returns 0 when nothing is masked so the term can be
    switched off without special-casing the caller.
    """
    pred, true = x_next_pred[:, :-1], x[:, 1:]
    target_mask = mask[:, 1:]
    n = target_mask.sum()
    if n == 0:
        return x.new_zeros(())
    err = ((pred - true) ** 2).mean(dim=-1)
    return (err * target_mask.to(err.dtype)).sum() / n


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Standard VAE KL to a unit Gaussian, per element."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def sample_mask(
    batch: int, seq_len: int, ratio: float, device=None, generator=None
) -> torch.Tensor:
    """Bernoulli timepoint mask, with the first step always visible.

    The RNN has no history at ``t=0``; masking it would ask the model to
    predict a frame from nothing and add pure noise to the gradient.
    """
    mask = torch.rand(batch, seq_len, device=device, generator=generator) < ratio
    mask[:, 0] = False
    return mask


class SSLObjective(nn.Module):
    """Weighted sum of the self-supervised losses.

    Returns ``(total, parts)`` where ``parts`` is a plain float dict for
    logging — keeping tensors out of the log path avoids accidentally holding
    the graph alive across steps.
    """

    def __init__(
        self,
        w_reconstruction: float = 1.0,
        w_forecast: float = 1.0,
        w_masked: float = 1.0,
        w_kl: float = 0.0,
        latent_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.w_reconstruction = w_reconstruction
        self.w_forecast = w_forecast
        self.w_masked = w_masked
        self.w_kl = w_kl
        self.latent_weight = latent_weight

    def forward(
        self, out: dict, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        recon = reconstruction_loss(out["x_recon"], x)
        fcast, parts = forecast_loss(
            out["x_next_pred"], out["z_next_pred"], x, out["z"], self.latent_weight
        )

        total = self.w_reconstruction * recon + self.w_forecast * fcast
        parts["reconstruction"] = recon.detach().item()
        parts["forecast"] = fcast.detach().item()

        if mask is not None and self.w_masked:
            masked = masked_loss(out["x_next_pred"], x, mask)
            total = total + self.w_masked * masked
            parts["masked"] = masked.detach().item()

        if self.w_kl and out.get("logvar") is not None:
            kl = kl_loss(out["mu"], out["logvar"])
            total = total + self.w_kl * kl
            parts["kl"] = kl.detach().item()

        parts["total"] = total.detach().item()
        return total, parts
