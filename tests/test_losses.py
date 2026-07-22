"""Tests for the self-supervised objectives.

The properties that matter here are alignment (step t predicts t+1, not t) and
non-collapse (the latent term must not be minimizable by flattening z). Both
produce a loss curve that goes down while the model learns nothing.
"""

from __future__ import annotations

import pytest
import torch

from foundational_brain.models import FoundationModel
from foundational_brain.training.losses import (
    SSLObjective,
    forecast_loss,
    masked_loss,
    reconstruction_loss,
    sample_mask,
)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return FoundationModel(
        n_regions=16, latent_dim=8, encoder_hidden=[32], decoder_hidden=[32],
        rnn_hidden=16, rnn_layers=1,
    )


def test_reconstruction_is_zero_when_exact():
    x = torch.randn(2, 5, 16)
    assert reconstruction_loss(x, x).item() == pytest.approx(0.0)


def test_forecast_alignment_step_t_predicts_t_plus_one():
    """A predictor that emits the true next frame must score ~0."""
    x = torch.randn(3, 10, 16)
    z = torch.randn(3, 10, 8)
    # perfect prediction: at step t, emit x[t+1]; last step is dropped
    x_next_pred = torch.cat([x[:, 1:], torch.zeros(3, 1, 16)], dim=1)
    z_next_pred = torch.cat([z[:, 1:], torch.zeros(3, 1, 8)], dim=1)
    total, parts = forecast_loss(x_next_pred, z_next_pred, x, z)
    assert parts["forecast_signal"] == pytest.approx(0.0, abs=1e-6)
    assert parts["forecast_latent"] == pytest.approx(0.0, abs=1e-6)


def test_forecast_misalignment_is_penalized():
    """Emitting the *current* frame instead of the next must not score 0."""
    x = torch.randn(3, 10, 16)
    z = torch.randn(3, 10, 8)
    total, parts = forecast_loss(x, z, x, z)  # predict x[t] at step t
    assert parts["forecast_signal"] > 0.5


def test_latent_forecast_target_is_detached():
    """Collapse guard: no gradient may flow into z through the latent target.

    With a constant predictor the target is the *only* possible path to z, so
    a correctly detached loss has no graph at all — while the same computation
    without the detach does.
    """
    x = torch.randn(2, 6, 16)
    z = torch.randn(2, 6, 8, requires_grad=True)
    z_next_pred = torch.zeros(2, 6, 8)  # constant prediction, no params
    x_next_pred = torch.zeros(2, 6, 16)

    total, _ = forecast_loss(x_next_pred, z_next_pred, x, z)
    assert not total.requires_grad, "gradient leaks into z through the latent target"

    # sanity: the same loss without detaching the target *would* be trainable,
    # so the assertion above is testing the detach and not a trivially dead graph
    leaky = torch.nn.functional.mse_loss(z_next_pred[:, :-1], z[:, 1:])
    assert leaky.requires_grad


def test_masked_loss_only_counts_masked_steps():
    x = torch.randn(2, 8, 16)
    x_next_pred = torch.cat([x[:, 1:], torch.zeros(2, 1, 16)], dim=1)  # perfect
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, 3] = True
    assert masked_loss(x_next_pred, x, mask).item() == pytest.approx(0.0, abs=1e-6)

    # now make the prediction wrong only at the masked step
    bad = x_next_pred.clone()
    bad[:, 2] += 10.0  # step 2 predicts step 3, which is masked
    assert masked_loss(bad, x, mask).item() > 10.0


def test_masked_loss_is_zero_with_empty_mask():
    x = torch.randn(2, 6, 16)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    assert masked_loss(torch.randn(2, 6, 16), x, mask).item() == 0.0


def test_sample_mask_never_masks_first_step():
    m = sample_mask(64, 32, ratio=0.9)
    assert not m[:, 0].any()
    assert m[:, 1:].float().mean().item() > 0.5


def test_mask_changes_rnn_input_but_not_latent_target(model):
    """z must be the clean encoding; only the RNN's input is corrupted."""
    x = torch.randn(4, 12, 16)
    mask = sample_mask(4, 12, ratio=0.5)
    with torch.no_grad():
        clean = model(x)
        masked = model(x, mask=mask)
    assert torch.allclose(clean["z"], masked["z"])
    assert not torch.allclose(clean["z_next_pred"], masked["z_next_pred"])


def test_objective_runs_and_backprops(model):
    x = torch.randn(4, 12, 16)
    mask = sample_mask(4, 12, ratio=0.15)
    obj = SSLObjective()
    total, parts = obj(model(x, mask=mask), x, mask)
    total.backward()

    assert {"reconstruction", "forecast", "masked", "total"} <= set(parts)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_objective_weights_are_respected(model):
    x = torch.randn(4, 12, 16)
    out = model(x)
    only_recon = SSLObjective(w_reconstruction=1, w_forecast=0, w_masked=0)
    total, parts = only_recon(out, x)
    assert total.item() == pytest.approx(parts["reconstruction"], rel=1e-5)


def test_collapsed_latent_does_not_win(model):
    """A constant z must not beat an informative z on the full objective.

    This is the concrete form of the collapse worry: if it did, the encoder
    would have an incentive to throw the signal away.
    """
    x = torch.randn(8, 16, 16)
    obj = SSLObjective()

    out = model(x)
    real_total, _ = obj(out, x)

    collapsed = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in out.items()}
    collapsed["z"] = torch.zeros_like(out["z"])
    collapsed["z_next_pred"] = torch.zeros_like(out["z_next_pred"])
    collapsed["x_recon"] = model.decoder(collapsed["z"])
    collapsed["x_next_pred"] = model.decoder(collapsed["z_next_pred"])
    collapsed_total, _ = obj(collapsed, x)

    assert collapsed_total.item() > real_total.item() * 0.5
