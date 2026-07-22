"""Training loop for the foundation model.

Deliberately small: an epoch loop, an eval pass, early stopping on validation
total loss, and best-checkpoint tracking. The interesting decisions are not in
the loop, they are in what gets reported — every eval pass records the model's
one-step forecasting MSE in the *same units as the baselines*
(``foundational_brain.eval.baselines``), so "is this better than AR(1)" is
answerable at every epoch rather than at the end.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .losses import SSLObjective, sample_mask


def pick_device(prefer: str | None = None) -> torch.device:
    """Best available device: explicit choice, else MPS/CUDA, else CPU."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4
    mask_ratio: float = 0.15
    grad_clip: float = 1.0
    patience: int = 8
    seed: int = 42
    w_reconstruction: float = 1.0
    w_forecast: float = 1.0
    w_masked: float = 1.0
    w_kl: float = 0.0
    latent_weight: float = 0.1
    device: str | None = None
    log_every: int = 0  # 0 = only per-epoch logging


@dataclass
class History:
    train: list[dict] = field(default_factory=list)
    val: list[dict] = field(default_factory=list)
    best_epoch: int = -1
    best_val: float = float("inf")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    objective: SSLObjective,
    device: torch.device,
    mask_ratio: float = 0.0,
) -> dict[str, float]:
    """Validation pass.

    ``forecast_mse_1tr`` is the number to compare against the AR(1) baseline:
    plain MSE of the predicted next frame against the true next frame, with no
    masking and no loss weighting, so the units match exactly.
    """
    model.eval()
    sums: dict[str, float] = {}
    n_batches = 0
    fc_err, fc_count = 0.0, 0

    for batch in loader:
        x = batch["x"].to(device)
        mask = (
            sample_mask(x.shape[0], x.shape[1], mask_ratio, device=device)
            if mask_ratio
            else None
        )
        out = model(x, mask=mask)
        _, parts = objective(out, x, mask)
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + v
        n_batches += 1

        pred, true = out["x_next_pred"][:, :-1], x[:, 1:]
        fc_err += float(((pred - true) ** 2).sum())
        fc_count += true.numel()

    res = {k: v / max(n_batches, 1) for k, v in sums.items()}
    res["forecast_mse_1tr"] = fc_err / max(fc_count, 1)
    return res


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig | None = None,
    verbose: bool = True,
) -> tuple[nn.Module, History]:
    """Train and return ``(model_with_best_weights, history)``.

    The returned model carries the *best* validation weights, not the last
    epoch's — otherwise early stopping would report a score the returned model
    does not have.
    """
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = pick_device(cfg.device)
    model = model.to(device)
    objective = SSLObjective(
        cfg.w_reconstruction, cfg.w_forecast, cfg.w_masked, cfg.w_kl, cfg.latent_weight
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))

    history = History()
    best_state = copy.deepcopy(model.state_dict())
    since_improved = 0

    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        sums: dict[str, float] = {}
        n_batches = 0

        for step, batch in enumerate(train_loader):
            x = batch["x"].to(device)
            mask = (
                sample_mask(x.shape[0], x.shape[1], cfg.mask_ratio, device=device)
                if cfg.mask_ratio
                else None
            )
            out = model(x, mask=mask)
            loss, parts = objective(out, x, mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            n_batches += 1
            if cfg.log_every and step % cfg.log_every == 0 and verbose:
                print(f"  epoch {epoch} step {step} loss {parts['total']:.4f}")

        sched.step()
        tr = {k: v / max(n_batches, 1) for k, v in sums.items()}
        va = evaluate(model, val_loader, objective, device, mask_ratio=0.0)
        history.train.append(tr)
        history.val.append(va)

        improved = va["total"] < history.best_val - 1e-5
        if improved:
            history.best_val = va["total"]
            history.best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            since_improved = 0
        else:
            since_improved += 1

        if verbose:
            print(
                f"epoch {epoch:>3}  train {tr['total']:.4f}  val {va['total']:.4f}  "
                f"recon {va['reconstruction']:.4f}  "
                f"fcast_1tr {va['forecast_mse_1tr']:.4f}  "
                f"{time.time() - t0:.1f}s{'  *' if improved else ''}"
            )

        if since_improved >= cfg.patience:
            if verbose:
                print(f"early stop at epoch {epoch} (best {history.best_epoch})")
            break

    model.load_state_dict(best_state)
    return model, history
