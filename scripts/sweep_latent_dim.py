#!/usr/bin/env python
"""Latent-width ablation: does the encoder's nonlinearity ever beat PCA?

Open problem 1 from the pretraining report. At latent_dim=128 the nonlinear
autoencoder lost to a 128-component PCA on reconstruction. That has two very
different explanations:

* the nonlinearity buys nothing, so the whole encoder/decoder is decorative; or
* 128 dims is simply past the point where anything nonlinear is left to model,
  because ~95% of frame variance is already linear by then.

These are distinguishable. Sweep latent_dim and, at each width, train the
autoencoder and fit a PCA of the same width on the same training subjects. If
the autoencoder wins at *narrow* widths and only loses at wide ones, the
nonlinearity is real and 128 was the wrong place to look. If it loses
everywhere, the encoder is not earning its parameters.

    python scripts/sweep_latent_dim.py --widths 8 16 32 64 128 --epochs 30

Writes ``docs/latent_sweep_report.md``, ``docs/latent_sweep.json`` and
``figures/latent_sweep.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from foundational_brain.data.pipeline import prepare_pretrain_data  # noqa: E402
from foundational_brain.eval.baselines import fit_pca, reconstruction_mse  # noqa: E402
from foundational_brain.models import FoundationModel  # noqa: E402
from foundational_brain.training.trainer import (  # noqa: E402
    TrainConfig,
    pick_device,
    train,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--widths", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    p.add_argument("--n-subjects", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs")
    p.add_argument("--fig-dir", type=Path, default=REPO_ROOT / "figures")
    return p.parse_args()


@torch.no_grad()
def ae_reconstruction_mse(model, loader, device) -> float:
    model.eval()
    err, count = 0.0, 0
    for batch in loader:
        x = batch["x"].to(device)
        err += float(((model(x)["x_recon"] - x) ** 2).sum())
        count += x.numel()
    return err / max(count, 1)


def train_autoencoder(n_regions, width, loaders, cfg: TrainConfig, device):
    """A width-`width` autoencoder: encoder + decoder, no RNN objective.

    Hidden widths scale with the latent so a narrow bottleneck is not starved
    by a wide MLP and a wide one is not bottlenecked by a narrow MLP — the
    comparison to PCA is about the latent width, and the surrounding capacity
    should not confound it.
    """
    enc_hidden = [max(width * 4, 128), max(width * 2, 64)]
    model = FoundationModel(
        n_regions=n_regions, latent_dim=width,
        encoder_hidden=enc_hidden, decoder_hidden=list(reversed(enc_hidden)),
        rnn_hidden=64, rnn_layers=1, rnn_dropout=0.0,
    )
    model, hist = train(model, loaders["train"], loaders["val"], cfg, verbose=False)
    return model, hist


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    print(f"device: {device}; widths {args.widths}")

    n = args.n_subjects if args.n_subjects > 0 else None
    data = prepare_pretrain_data(n_subjects=n, seq_len=args.seq_len, seed=args.seed)
    loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=(name == "train"))
        for name, ds in data.datasets.items()
    }
    train_series = data.split_series["train"]
    val_series = data.split_series["val"]

    rows = []
    for width in args.widths:
        # PCA bar at this width, fitted on the same training subjects
        comps, mean = fit_pca(train_series, n_components=width)
        pca_mse = reconstruction_mse(val_series, comps, mean)

        cfg = TrainConfig(
            epochs=args.epochs, lr=args.lr, mask_ratio=0.0,
            w_reconstruction=1.0, w_forecast=0.0, w_masked=0.0,
            seed=args.seed, device=str(device),
        )
        model, hist = train_autoencoder(data.n_regions, width, loaders, cfg, device)
        ae_mse = ae_reconstruction_mse(model, loaders["val"], device)

        gap = (ae_mse - pca_mse) / pca_mse
        rows.append({
            "latent_dim": width,
            "ae_val_mse": ae_mse,
            "pca_val_mse": pca_mse,
            "ae_minus_pca_rel": gap,
            "ae_wins": ae_mse < pca_mse,
            "epochs_run": len(hist.val),
        })
        print(
            f"width {width:>4}: AE {ae_mse:.4f}  PCA {pca_mse:.4f}  "
            f"gap {gap:+.1%}  {'AE wins' if ae_mse < pca_mse else 'PCA wins'}"
        )

    write_outputs(rows, data.n_regions, args)


def write_outputs(rows, n_regions, args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "latent_sweep.json").write_text(
        json.dumps({"n_regions": n_regions, "results": rows}, indent=2)
    )

    crossover = next((r["latent_dim"] for r in rows if not r["ae_wins"]), None)
    lines = [
        "# Latent-width ablation: autoencoder vs PCA",
        "",
        f"Reconstruction MSE on the validation split, {n_regions} regions. "
        "PCA is fitted at each width on the same training subjects the "
        "autoencoder trains on.",
        "",
        "| latent_dim | autoencoder | PCA | AE − PCA | winner |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['latent_dim']} | {r['ae_val_mse']:.4f} | "
            f"{r['pca_val_mse']:.4f} | {r['ae_minus_pca_rel']:+.1%} | "
            f"{'**AE**' if r['ae_wins'] else 'PCA'} |"
        )
    lines += ["", "## Reading"]
    if all(r["ae_wins"] for r in rows):
        lines.append(
            "- The autoencoder beats PCA at **every** width tested: the "
            "nonlinearity is doing real work, and the earlier loss at 128 was "
            "specific to that width."
        )
    elif not any(r["ae_wins"] for r in rows):
        lines.append(
            "- The autoencoder loses to PCA at **every** width: for pure "
            "reconstruction the nonlinear encoder is not earning its "
            "parameters. Its value, if any, has to come from the dynamics "
            "objective, not reconstruction."
        )
    else:
        lines.append(
            f"- The autoencoder wins below latent_dim ≈ {crossover} and loses "
            f"at or above it. So the nonlinearity is real at the widths where "
            "reconstruction is genuinely hard, and the loss at 128 reflects "
            "there being little nonlinear signal left once most variance is "
            "already linearly captured — not a broken encoder."
        )
    (args.out_dir / "latent_sweep_report.md").write_text("\n".join(lines))
    print(f"wrote {args.out_dir / 'latent_sweep_report.md'}")

    try:
        make_figure(rows, args.fig_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"(figure skipped: {exc})")


def make_figure(rows, fig_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    widths = [r["latent_dim"] for r in rows]
    ae = [r["ae_val_mse"] for r in rows]
    pca = [r["pca_val_mse"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(widths, ae, "o-", color="#C44E52", label="autoencoder")
    ax.plot(widths, pca, "s--", color="#4C72B0", label="PCA")
    ax.set_xscale("log", base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels(widths)
    ax.set_xlabel("latent_dim")
    ax.set_ylabel("validation reconstruction MSE")
    ax.set_title("Nonlinear autoencoder vs PCA at matched width")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "latent_sweep.png", dpi=130)
    plt.close(fig)
    print(f"wrote {fig_dir / 'latent_sweep.png'}")


if __name__ == "__main__":
    main()
