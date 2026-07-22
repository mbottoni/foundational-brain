#!/usr/bin/env python
"""Pretrain the foundation model on ABIDE and score it against the baselines.

    python scripts/pretrain.py --epochs 40

Runs the full chain end to end:

1. load ABIDE ROI series, restrict to the TR-homogeneous pretraining group
2. site-stratified subject-level train/val/test split
3. fit baselines (AR(1), persistence, PCA) on train, score on val/test
4. **Phase 2** — train the autoencoder alone and check it beats PCA at the
   same width
5. **Phase 3** — train the full model with all three SSL objectives and check
   it beats AR(1) at a 1-TR horizon
6. evaluate on the held-out sites, which have a different TR than anything
   seen in training

Writes ``docs/pretrain_report.md`` and ``docs/pretrain_results.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from foundational_brain.data import load_abide_series  # noqa: E402
from foundational_brain.data.dataset import (  # noqa: E402
    WindowedFMRIDataset,
    build_datasets,
    normalize_subject,
)
from foundational_brain.data.sites import split_by_tr  # noqa: E402
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
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs/default.yaml")
    p.add_argument("--n-subjects", type=int, default=0, help="0 = all")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--ae-epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs")
    p.add_argument("--ckpt-dir", type=Path, default=REPO_ROOT / "checkpoints")
    return p.parse_args()


def make_loaders(datasets, batch_size):
    return {
        name: DataLoader(
            ds, batch_size=batch_size, shuffle=(name == "train"), drop_last=False
        )
        for name, ds in datasets.items()
    }


def build_model(cfg, n_regions: int) -> FoundationModel:
    m = cfg["model"]
    return FoundationModel(
        n_regions=n_regions,
        latent_dim=m["latent_dim"],
        encoder_hidden=m["encoder_hidden"],
        decoder_hidden=m["decoder_hidden"],
        variational=m.get("variational", False),
        rnn_hidden=m["rnn"]["hidden_dim"],
        rnn_layers=m["rnn"]["num_layers"],
        rnn_type=m["rnn"]["type"],
        rnn_dropout=m["rnn"]["dropout"],
    )


@torch.no_grad()
def reconstruction_mse_model(model, loader, device) -> float:
    """Autoencoding MSE, directly comparable to the PCA reconstruction bar."""
    model.eval()
    err, count = 0.0, 0
    for batch in loader:
        x = batch["x"].to(device)
        out = model(x)
        err += float(((out["x_recon"] - x) ** 2).sum())
        count += x.numel()
    return err / max(count, 1)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    seq_len = cfg["data"]["seq_len"]
    batch_size = args.batch_size or cfg["data"]["batch_size"]
    seed = args.seed if args.seed is not None else cfg["train"]["seed"]
    epochs = args.epochs or cfg["train"]["epochs"]
    ae_epochs = args.ae_epochs or max(epochs // 2, 1)
    lr = args.lr or cfg["train"]["lr"]
    device = pick_device(args.device)
    print(f"device: {device}")

    # ---------------------------------------------------------------- data
    n = args.n_subjects if args.n_subjects > 0 else None
    series, pheno = load_abide_series(
        n_subjects=n, derivative=cfg["data"]["derivative"]
    )
    sites = [str(s) for s in pheno["SITE_ID"]]

    in_group, held_out = split_by_tr(sites)
    print(
        f"loaded {len(series)} subjects; {len(in_group)} in the TR="
        f"{cfg.get('data', {}).get('pretrain_tr', 2.0)}s pretraining group, "
        f"{len(held_out)} held out for cross-TR evaluation"
    )

    group_series = [series[i] for i in in_group]
    group_sites = [sites[i] for i in in_group]

    datasets, splits, keep = build_datasets(
        group_series, group_sites, seq_len=seq_len, seed=seed
    )
    loaders = make_loaders(datasets, batch_size)
    n_regions = int(keep.sum())
    print(
        f"regions kept: {n_regions}; windows "
        + ", ".join(f"{k}={len(v)}" for k, v in datasets.items())
    )

    # normalized per-subject series, matching exactly what the model sees
    norm = [normalize_subject(s) for s in group_series]
    split_series = {k: [norm[i] for i in idx] for k, idx in splits.items()}
    split_sites = {k: [group_sites[i] for i in idx] for k, idx in splits.items()}

    # ----------------------------------------------------------- baselines
    print("\nfitting baselines on train, scoring on val ...")
    base = evaluate_baselines(
        split_series["train"],
        split_series["val"],
        eval_sites=split_sites["val"],
        horizons=(1, 2),
        pca_components=cfg["model"]["latent_dim"],
    )
    ar1_bar = base["forecast"]["h1"]["ar1"]
    pca_bar = base["reconstruction"][f"pca_{cfg['model']['latent_dim']}"]
    print(f"  AR(1) 1-TR forecast MSE : {ar1_bar:.4f}   <- the bar")
    print(f"  persistence             : {base['forecast']['h1']['persistence']:.4f}")
    print(f"  PCA-{cfg['model']['latent_dim']} reconstruction  : {pca_bar:.4f}")

    # ------------------------------------------- Phase 2: autoencoder only
    print("\n=== Phase 2: autoencoder (reconstruction only) ===")
    ae = build_model(cfg, n_regions)
    ae_cfg = TrainConfig(
        epochs=ae_epochs, lr=lr, weight_decay=cfg["train"]["weight_decay"],
        mask_ratio=0.0, w_reconstruction=1.0, w_forecast=0.0, w_masked=0.0,
        seed=seed, device=str(device),
    )
    ae, ae_hist = train(ae, loaders["train"], loaders["val"], ae_cfg)
    ae_recon = reconstruction_mse_model(ae, loaders["val"], device)
    print(f"autoencoder val reconstruction MSE {ae_recon:.4f} vs PCA {pca_bar:.4f}")

    # -------------------------------------------- Phase 3: full SSL model
    print("\n=== Phase 3: full model (reconstruction + forecast + masked) ===")
    model = build_model(cfg, n_regions)
    full_cfg = TrainConfig(
        epochs=epochs, lr=lr, weight_decay=cfg["train"]["weight_decay"],
        mask_ratio=cfg["train"]["mask_ratio"],
        w_reconstruction=cfg["train"]["loss_weights"]["reconstruction"],
        w_forecast=cfg["train"]["loss_weights"]["forecast"],
        w_masked=cfg["train"]["loss_weights"]["masked"],
        w_kl=cfg["train"]["loss_weights"]["kl"],
        seed=seed, device=str(device),
    )
    model, hist = train(model, loaders["train"], loaders["val"], full_cfg)

    obj = SSLObjective()
    val_res = evaluate(model, loaders["val"], obj, device)
    test_res = evaluate(model, loaders["test"], obj, device)
    model_recon = reconstruction_mse_model(model, loaders["val"], device)

    print(f"\nval  forecast MSE {val_res['forecast_mse_1tr']:.4f}  (AR1 {ar1_bar:.4f})")
    print(f"test forecast MSE {test_res['forecast_mse_1tr']:.4f}")

    # ------------------------------------------ cross-TR generalization
    cross = None
    if held_out:
        # Apply the *training* region mask, not a freshly computed one: the
        # model's input width is fixed by training, and recomputing the mask on
        # held-out subjects yields a different set of regions in a different
        # order, which would silently feed the model mismatched channels.
        ho_series = [series[i][:, keep] for i in held_out]
        ho_sites = [sites[i] for i in held_out]

        ho_ds = WindowedFMRIDataset(ho_series, seq_len=seq_len, stride=seq_len)
        ho_loader = DataLoader(ho_ds, batch_size=batch_size)
        ho_res = evaluate(model, ho_loader, obj, device)
        ho_base = evaluate_baselines(
            split_series["train"],
            [normalize_subject(s) for s in ho_series],
            eval_sites=ho_sites,
            pca_components=cfg["model"]["latent_dim"],
        )
        cross = {
            "n_subjects": len(ho_series),
            "n_windows": len(ho_ds),
            "model_forecast_mse_1tr": ho_res["forecast_mse_1tr"],
            "ar1_forecast_mse_1tr": ho_base["forecast"]["h1"]["ar1"],
            "per_site": ho_base.get("per_site", {}),
        }
        print(
            f"held-out sites forecast MSE {cross['model_forecast_mse_1tr']:.4f} "
            f"(AR1 {cross['ar1_forecast_mse_1tr']:.4f})"
        )

    # ------------------------------------------------------------ outputs
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "config": cfg, "n_regions": n_regions},
        args.ckpt_dir / "foundation_model.pt",
    )

    results = {
        "n_subjects_total": len(series),
        "n_subjects_pretrain_group": len(in_group),
        "n_regions": n_regions,
        "seq_len": seq_len,
        "latent_dim": cfg["model"]["latent_dim"],
        "splits": {k: len(v) for k, v in splits.items()},
        "windows": {k: len(v) for k, v in datasets.items()},
        "baselines": base,
        "autoencoder": {
            "epochs_run": len(ae_hist.val),
            "best_epoch": ae_hist.best_epoch,
            "val_reconstruction_mse": ae_recon,
        },
        "full_model": {
            "epochs_run": len(hist.val),
            "best_epoch": hist.best_epoch,
            "val": val_res,
            "test": test_res,
            "val_reconstruction_mse": model_recon,
        },
        "cross_tr": cross,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pretrain_results.json").write_text(json.dumps(results, indent=2))
    (args.out_dir / "pretrain_report.md").write_text(fmt_report(results))
    print(f"\nwrote {args.out_dir / 'pretrain_report.md'}")


def fmt_report(r: dict) -> str:
    ar1 = r["baselines"]["forecast"]["h1"]["ar1"]
    pers = r["baselines"]["forecast"]["h1"]["persistence"]
    pca = r["baselines"]["reconstruction"][f"pca_{r['latent_dim']}"]
    ae = r["autoencoder"]["val_reconstruction_mse"]
    fm = r["full_model"]["val"]["forecast_mse_1tr"]
    fm_test = r["full_model"]["test"]["forecast_mse_1tr"]

    def verdict(model_v: float, bar: float) -> str:
        gain = (bar - model_v) / bar * 100
        return (
            f"**{'beats' if model_v < bar else 'does NOT beat'}** the bar "
            f"({gain:+.1f}% vs {bar:.4f})"
        )

    lines = [
        "# Pretraining report",
        "",
        f"- {r['n_subjects_pretrain_group']} subjects in the TR-homogeneous "
        f"pretraining group (of {r['n_subjects_total']} total)",
        f"- {r['n_regions']} regions, seq_len {r['seq_len']}, "
        f"latent_dim {r['latent_dim']}",
        f"- split subjects: " + ", ".join(f"{k}={v}" for k, v in r["splits"].items()),
        f"- windows: " + ", ".join(f"{k}={v}" for k, v in r["windows"].items()),
        "",
        "## Baselines (fitted on train, scored on val)",
        "",
        "| predictor | 1-TR forecast MSE |",
        "|---|---|",
        f"| predict the mean | {r['baselines']['forecast']['h1']['mean']:.4f} |",
        f"| persistence | {pers:.4f} |",
        f"| AR(1) | **{ar1:.4f}** |",
        "",
        f"Reconstruction bar: PCA-{r['latent_dim']} at **{pca:.4f}**.",
        "",
        "## Phase 2 — autoencoder",
        "",
        f"Validation reconstruction MSE **{ae:.4f}**, {verdict(ae, pca)}.",
        "",
        "## Phase 3 — full model",
        "",
        f"Validation 1-TR forecast MSE **{fm:.4f}**, {verdict(fm, ar1)}.",
        f"Test 1-TR forecast MSE **{fm_test:.4f}**.",
        f"Validation reconstruction MSE "
        f"{r['full_model']['val_reconstruction_mse']:.4f}.",
        "",
    ]

    if r.get("cross_tr"):
        c = r["cross_tr"]
        lines += [
            "## Cross-TR generalization (held-out sites)",
            "",
            f"{c['n_subjects']} subjects from sites whose TR was never seen in "
            "training.",
            "",
            f"- model 1-TR forecast MSE **{c['model_forecast_mse_1tr']:.4f}**",
            f"- AR(1) on the same subjects: {c['ar1_forecast_mse_1tr']:.4f}",
            "",
        ]

    return "\n".join(lines)


if __name__ == "__main__":
    main()
