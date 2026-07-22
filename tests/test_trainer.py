"""Tests for the training loop, on synthetic data with a known answer.

The central check is that the loop can actually learn AR(1) dynamics well
enough to approach the analytic optimum. A loop that runs without error but
does not learn looks identical from the outside.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from foundational_brain.data.dataset import WindowedFMRIDataset
from foundational_brain.models import FoundationModel
from foundational_brain.training.trainer import TrainConfig, evaluate, train
from foundational_brain.training.losses import SSLObjective


def ar1_series(n_subjects, T, n_regions, rho, seed=0, rank=None):
    """Regions generated as a linear mixture of ``rank`` AR(1) factors.

    Low-rank by default, because that is both what fMRI looks like (effective
    rank ~86 of 200 regions in ABIDE) and what makes the analytic optimum
    reachable: if the data's rank exceeds ``latent_dim``, the bottleneck
    discards signal and reconstruction error puts a floor under forecast
    error, so the model cannot reach 1-rho^2 no matter how well it trains.

    A linear mixture of AR(1) factors sharing rho is itself AR(1) with the
    same rho, so the per-region optimum stays 1-rho^2 after z-scoring.
    """
    rng = np.random.default_rng(seed)
    rank = rank if rank is not None else max(n_regions // 2, 1)
    mixing = np.random.default_rng(1234).standard_normal((rank, n_regions))
    out = []
    for _ in range(n_subjects):
        f = np.zeros((T, rank), dtype=np.float32)
        noise = rng.standard_normal((T, rank)) * np.sqrt(1 - rho**2)
        for t in range(1, T):
            f[t] = rho * f[t - 1] + noise[t]
        out.append((f @ mixing).astype(np.float32))
    return out


@pytest.fixture(scope="module")
def loaders():
    tr = ar1_series(24, 200, 12, rho=0.8, seed=0, rank=6)
    va = ar1_series(8, 200, 12, rho=0.8, seed=99, rank=6)
    ds_tr = WindowedFMRIDataset(tr, seq_len=32, stride=16)
    ds_va = WindowedFMRIDataset(va, seq_len=32, stride=32)
    return (
        DataLoader(ds_tr, batch_size=32, shuffle=True),
        DataLoader(ds_va, batch_size=32),
    )


def small_model(n_regions=12, latent_dim=8):
    torch.manual_seed(0)
    return FoundationModel(
        n_regions=n_regions, latent_dim=latent_dim,
        encoder_hidden=[64, 32], decoder_hidden=[32, 64],
        rnn_hidden=64, rnn_layers=1, rnn_dropout=0.0,
    )


def test_train_reduces_validation_loss(loaders):
    train_loader, val_loader = loaders
    model, hist = train(
        small_model(), train_loader, val_loader,
        TrainConfig(epochs=8, lr=3e-3, patience=8, device="cpu"), verbose=False,
    )
    assert hist.val[-1]["total"] < hist.val[0]["total"]


def test_train_learns_ar1_dynamics_near_optimum(loaders):
    """On AR(1) data with rho=0.8 the optimal one-step MSE is 1-rho^2 = 0.36.

    The model must land near it — not merely improve — or the loop is
    optimizing something other than what we think.
    """
    train_loader, val_loader = loaders
    model, hist = train(
        small_model(), train_loader, val_loader,
        TrainConfig(epochs=30, lr=3e-3, patience=30, w_masked=0.0, device="cpu"),
        verbose=False,
    )
    best = hist.val[hist.best_epoch]["forecast_mse_1tr"]
    assert best < 0.50, f"forecast MSE {best:.3f} far above the 0.36 optimum"


def test_latent_bottleneck_below_data_rank_floors_forecast_error():
    """A latent narrower than the data's rank cannot be trained around.

    This is the synthetic analogue of the ABIDE finding that frame space has
    effective rank ~86: choosing latent_dim below it discards signal that no
    amount of training recovers, which is why latent_dim is capped by the
    eigenspectrum rather than picked for capacity.
    """
    full_rank_tr = ar1_series(24, 200, 12, rho=0.8, seed=0, rank=12)
    full_rank_va = ar1_series(8, 200, 12, rho=0.8, seed=99, rank=12)
    tr_loader = DataLoader(
        WindowedFMRIDataset(full_rank_tr, seq_len=32, stride=16), batch_size=32,
        shuffle=True,
    )
    va_loader = DataLoader(
        WindowedFMRIDataset(full_rank_va, seq_len=32, stride=32), batch_size=32
    )
    _, hist = train(
        small_model(latent_dim=4), tr_loader, va_loader,
        TrainConfig(epochs=30, lr=3e-3, patience=30, w_masked=0.0, device="cpu"),
        verbose=False,
    )
    # 4 latent dims for rank-12 data: most of the variance is unrepresentable
    assert hist.val[hist.best_epoch]["forecast_mse_1tr"] > 0.60


def test_returned_model_carries_best_not_last_weights(loaders):
    """Early stopping must return the weights whose score it reported."""
    train_loader, val_loader = loaders
    model, hist = train(
        small_model(), train_loader, val_loader,
        TrainConfig(epochs=6, lr=3e-3, patience=6, device="cpu"), verbose=False,
    )
    rescored = evaluate(model, val_loader, SSLObjective(), torch.device("cpu"))
    assert rescored["total"] == pytest.approx(hist.best_val, rel=0.02)


def test_training_is_deterministic_given_seed(loaders):
    train_loader, val_loader = loaders
    cfg = TrainConfig(epochs=3, lr=3e-3, device="cpu", seed=7)
    _, h1 = train(small_model(), train_loader, val_loader, cfg, verbose=False)
    _, h2 = train(small_model(), train_loader, val_loader, cfg, verbose=False)
    assert h1.val[-1]["reconstruction"] == pytest.approx(
        h2.val[-1]["reconstruction"], rel=1e-4
    )


def test_early_stopping_triggers(loaders):
    train_loader, val_loader = loaders
    # lr=0 means the model can never improve, so patience must fire
    _, hist = train(
        small_model(), train_loader, val_loader,
        TrainConfig(epochs=50, lr=0.0, patience=2, device="cpu"), verbose=False,
    )
    assert len(hist.val) < 50


def test_evaluate_forecast_mse_matches_manual_computation(loaders):
    _, val_loader = loaders
    model = small_model()
    device = torch.device("cpu")
    res = evaluate(model, val_loader, SSLObjective(), device)

    err, count = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"]
            out = model(x)
            err += float(((out["x_next_pred"][:, :-1] - x[:, 1:]) ** 2).sum())
            count += x[:, 1:].numel()
    assert res["forecast_mse_1tr"] == pytest.approx(err / count, rel=1e-5)
