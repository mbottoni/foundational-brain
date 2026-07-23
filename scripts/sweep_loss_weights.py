#!/usr/bin/env python
"""Does down-weighting reconstruction free the latent to forecast better?

Open problem 3. In the full model, adding the forecasting objective doubled the
reconstruction error (0.0859 -> 0.1816): the two losses compete for one latent.
The latent-width ablation then showed reconstruction is essentially a linear
problem PCA already solves — so spending latent capacity on it may be wasted
when the goal is forecasting.

This sweeps the reconstruction weight (forecast and masked fixed at 1.0) and
measures validation 1-TR forecast MSE — the metric that matters — plus
reconstruction MSE, so the trade is visible. If forecasting improves as
reconstruction weight falls, the default 1:1:1 was leaving performance on the
table; if it degrades, reconstruction is a useful regulariser and the tension
is worth keeping.

    python scripts/sweep_loss_weights.py --recon-weights 0.0 0.1 0.3 1.0 --epochs 40

Writes ``docs/loss_weight_sweep_report.md``, ``docs/loss_weight_sweep.json``,
``figures/loss_weight_sweep.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from foundational_brain.data.pipeline import prepare_pretrain_data  # noqa: E402
from foundational_brain.eval.baselines import evaluate_baselines  # noqa: E402
from foundational_brain.models import FoundationModel  # noqa: E402
from foundational_brain.training.losses import SSLObjective  # noqa: E402
from foundational_brain.training.trainer import (  # noqa: E402
    TrainConfig,
    evaluate,
    pick_device,
    train,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recon-weights", type=float, nargs="+",
                   default=[0.0, 0.1, 0.3, 1.0])
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs/default.yaml")
    p.add_argument("--n-subjects", type=int, default=0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs")
    p.add_argument("--fig-dir", type=Path, default=REPO_ROOT / "figures")
    return p.parse_args()


@torch.no_grad()
def recon_mse(model, loader, device) -> float:
    model.eval()
    err, count = 0.0, 0
    for batch in loader:
        x = batch["x"].to(device)
        err += float(((model(x)["x_recon"] - x) ** 2).sum())
        count += x.numel()
    return err / max(count, 1)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    m = cfg["model"]
    seq_len = cfg["data"]["seq_len"]
    device = pick_device(args.device)
    print(f"device: {device}; reconstruction weights {args.recon_weights}")

    n = args.n_subjects if args.n_subjects > 0 else None
    data = prepare_pretrain_data(n_subjects=n, seq_len=seq_len, seed=args.seed)
    loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=(name == "train"))
        for name, ds in data.datasets.items()
    }

    base = evaluate_baselines(
        data.split_series["train"], data.split_series["val"],
        pca_components=m["latent_dim"],
    )
    ar1_bar = base["forecast"]["h1"]["ar1"]
    print(f"AR(1) bar: {ar1_bar:.4f}")

    obj = SSLObjective()  # unweighted, for reporting comparable component losses
    rows = []
    for w in args.recon_weights:
        model = FoundationModel(
            n_regions=data.n_regions, latent_dim=m["latent_dim"],
            encoder_hidden=m["encoder_hidden"], decoder_hidden=m["decoder_hidden"],
            rnn_hidden=m["rnn"]["hidden_dim"], rnn_layers=m["rnn"]["num_layers"],
            rnn_type=m["rnn"]["type"], rnn_dropout=m["rnn"]["dropout"],
        )
        cfg_w = TrainConfig(
            epochs=args.epochs, lr=args.lr,
            weight_decay=cfg["train"]["weight_decay"],
            mask_ratio=cfg["train"]["mask_ratio"],
            w_reconstruction=w, w_forecast=1.0, w_masked=1.0,
            seed=args.seed, device=str(device),
        )
        model, hist = train(model, loaders["train"], loaders["val"], cfg_w,
                            verbose=False)
        val = evaluate(model, loaders["val"], obj, device)
        rows.append({
            "w_reconstruction": w,
            "val_forecast_mse_1tr": val["forecast_mse_1tr"],
            "val_reconstruction_mse": recon_mse(model, loaders["val"], device),
            "beats_ar1": val["forecast_mse_1tr"] < ar1_bar,
            "epochs_run": len(hist.val),
        })
        r = rows[-1]
        print(
            f"w_recon {w:>4}: forecast {r['val_forecast_mse_1tr']:.4f}  "
            f"recon {r['val_reconstruction_mse']:.4f}  "
            f"{'beats' if r['beats_ar1'] else 'below'} AR(1)"
        )

    write_outputs(rows, ar1_bar, args)


def write_outputs(rows, ar1_bar, args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "loss_weight_sweep.json").write_text(
        json.dumps({"ar1_bar": ar1_bar, "results": rows}, indent=2)
    )

    best = min(rows, key=lambda r: r["val_forecast_mse_1tr"])
    default = next((r for r in rows if r["w_reconstruction"] == 1.0), None)

    lines = [
        "# Loss-weight sweep: reconstruction weight vs forecasting quality",
        "",
        f"Forecast and masked weights fixed at 1.0. AR(1) bar: **{ar1_bar:.4f}**. "
        "Lower forecast MSE is better.",
        "",
        "| w_reconstruction | val forecast MSE | val reconstruction MSE | vs AR(1) |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['w_reconstruction']} | {r['val_forecast_mse_1tr']:.4f} | "
            f"{r['val_reconstruction_mse']:.4f} | "
            f"{'beats' if r['beats_ar1'] else 'below'} |"
        )
    lines += ["", "## Reading", ""]
    lines.append(
        f"- Best forecasting at **w_reconstruction = {best['w_reconstruction']}** "
        f"({best['val_forecast_mse_1tr']:.4f})."
    )
    if default is not None:
        delta = (default["val_forecast_mse_1tr"] - best["val_forecast_mse_1tr"])
        if best["w_reconstruction"] != 1.0 and delta > 1e-3:
            lines.append(
                f"- That beats the default 1:1:1 ({default['val_forecast_mse_1tr']:.4f}) "
                f"by {delta:.4f} — the reconstruction term was competing with "
                "forecasting, and down-weighting it helps. Worth updating the config."
            )
        else:
            lines.append(
                "- The default 1:1:1 is at or near the best; reconstruction is not "
                "meaningfully hurting forecasting, so keeping it as a regulariser "
                "costs nothing."
            )
    if any(r["w_reconstruction"] == 0.0 for r in rows):
        z = next(r for r in rows if r["w_reconstruction"] == 0.0)
        lines.append(
            f"- With reconstruction fully off (w=0), forecast MSE is "
            f"{z['val_forecast_mse_1tr']:.4f} and reconstruction MSE is "
            f"{z['val_reconstruction_mse']:.4f} — a check on whether the decoder "
            "is needed at all for the forecasting objective."
        )
    (args.out_dir / "loss_weight_sweep_report.md").write_text("\n".join(lines))
    print(f"wrote {args.out_dir / 'loss_weight_sweep_report.md'}")

    try:
        make_figure(rows, ar1_bar, args.fig_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"(figure skipped: {exc})")


def make_figure(rows, ar1_bar, fig_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    w = [r["w_reconstruction"] for r in rows]
    fc = [r["val_forecast_mse_1tr"] for r in rows]
    rc = [r["val_reconstruction_mse"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ax1.plot(w, fc, "o-", color="#C44E52", label="forecast MSE (↓ better)")
    ax1.axhline(ar1_bar, ls="--", color="gray", lw=1, label=f"AR(1) {ar1_bar:.3f}")
    ax1.set_xlabel("reconstruction loss weight")
    ax1.set_ylabel("validation forecast MSE", color="#C44E52")
    ax1.tick_params(axis="y", labelcolor="#C44E52")

    ax2 = ax1.twinx()
    ax2.plot(w, rc, "s--", color="#4C72B0", label="reconstruction MSE")
    ax2.set_ylabel("validation reconstruction MSE", color="#4C72B0")
    ax2.tick_params(axis="y", labelcolor="#4C72B0")

    ax1.set_title("Reconstruction weight vs forecasting quality")
    fig.legend(loc="upper center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 0.99))
    fig.tight_layout()
    fig.savefig(fig_dir / "loss_weight_sweep.png", dpi=130)
    plt.close(fig)
    print(f"wrote {fig_dir / 'loss_weight_sweep.png'}")


if __name__ == "__main__":
    main()
