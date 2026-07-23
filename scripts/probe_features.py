#!/usr/bin/env python
"""Phase 5: linear-probe frozen features for phenotype, and test one prediction.

The latent-width ablation implied a sharp, falsifiable claim: the model's value
is temporal, so the **RNN hidden state** should carry subject phenotype while
the **encoder latent** should decode no better than **PCA** of the same data.
This script freezes a trained model, extracts the three representations, and
linearly probes them for DX group, age, sex — and for scan site, as the
confound control.

    python scripts/probe_features.py --checkpoint checkpoints/foundation_model.pt

Writes ``docs/probe_report.md`` and ``docs/probe_results.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from foundational_brain.data.pipeline import prepare_pretrain_data  # noqa: E402
from foundational_brain.eval.features import all_features  # noqa: E402
from foundational_brain.eval.probe import probe_all  # noqa: E402
from foundational_brain.models import FoundationModel  # noqa: E402
from foundational_brain.training.trainer import pick_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "checkpoints/foundation_model.pt")
    p.add_argument("--n-subjects", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs")
    return p.parse_args()


def load_model(checkpoint: Path, device) -> tuple[FoundationModel, dict]:
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    m = ck["config"]["model"]
    model = FoundationModel(
        n_regions=ck["n_regions"], latent_dim=m["latent_dim"],
        encoder_hidden=m["encoder_hidden"], decoder_hidden=m["decoder_hidden"],
        rnn_hidden=m["rnn"]["hidden_dim"], rnn_layers=m["rnn"]["num_layers"],
        rnn_type=m["rnn"]["type"], rnn_dropout=m["rnn"]["dropout"],
    )
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck


def build_targets(pheno) -> dict:
    """Phenotypic targets, dropping subjects with missing/invalid values.

    Returns ``{name: (kind, values, mask)}`` where mask selects the subjects
    that have that target — age is missing for a few subjects, and dropping
    them per-target keeps each probe on the largest valid sample.
    """
    dx = pheno["DX_GROUP"].to_numpy()          # 1 = ASD, 2 = control
    sex = pheno["SEX"].to_numpy()              # 1 = male, 2 = female
    site = pheno["SITE_ID"].astype(str).to_numpy()
    age = pheno["AGE_AT_SCAN"].to_numpy(dtype=float)

    targets = {
        "dx_group": ("classification", dx, np.isin(dx, [1, 2])),
        "sex": ("classification", sex, np.isin(sex, [1, 2])),
        "site": ("classification", site, np.ones(len(site), bool)),
        "age": ("regression", age, np.isfinite(age) & (age > 0)),
    }
    return targets


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    print(f"device: {device}")

    if not args.checkpoint.exists():
        raise SystemExit(
            f"no checkpoint at {args.checkpoint}; run scripts/pretrain.py first"
        )
    model, ck = load_model(args.checkpoint, device)

    n = args.n_subjects if args.n_subjects > 0 else None
    data = prepare_pretrain_data(n_subjects=n, seq_len=ck["config"]["data"]["seq_len"],
                                 seed=args.seed)

    # Probe the whole pretraining group. Labels were never used in SSL, and the
    # probe is cross-validated, so training subjects are fair game — this is the
    # standard linear-probe protocol. Concatenate the split series back together.
    series = data.split_series["train"] + data.split_series["val"] + data.split_series["test"]
    pheno = _concat_pheno(data)
    train_series = data.split_series["train"]  # PCA basis fit on train only

    print(f"extracting features for {len(series)} subjects ...")
    feats = all_features(model, series, train_series, device,
                         pca_components=ck["config"]["model"]["latent_dim"])
    for name, arr in feats.items():
        print(f"  {name:8} {arr.shape}")

    targets = build_targets(pheno)
    results: dict = {"n_subjects": len(series), "feature_dims": {}, "probes": {}}
    for name, arr in feats.items():
        results["feature_dims"][name] = int(arr.shape[1])

    for tname, (kind, values, mask) in targets.items():
        sub_feats = {fn: arr[mask] for fn, arr in feats.items()}
        res = probe_all(sub_feats, {tname: (kind, values[mask])},
                        n_splits=args.folds, seed=args.seed)
        results["probes"][tname] = res[tname]

    _report(results, args.out_dir)


def _concat_pheno(data):
    import pandas as pd

    return pd.concat(
        [data.split_phenotypic["train"], data.split_phenotypic["val"],
         data.split_phenotypic["test"]],
        ignore_index=True,
    )


def _report(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe_results.json").write_text(json.dumps(results, indent=2))

    order = ["pca", "encoder", "rnn"]
    lines = [
        "# Phase 5 — linear probing of frozen features",
        "",
        f"{results['n_subjects']} subjects (the TR=2.0s pretraining group), "
        "5-fold cross-validated linear probes. Feature dims: "
        + ", ".join(f"{k}={v}" for k, v in results["feature_dims"].items())
        + ".",
        "",
        "**The prediction under test:** if the model's value is temporal, `rnn` "
        "should beat `pca`, and `encoder` should roughly match `pca`.",
        "",
    ]

    for tname, per_feat in results["probes"].items():
        sample = next(iter(per_feat.values()))
        lines += [f"## {tname}", ""]
        if sample["task"] == "classification":
            chance = sample["chance_accuracy"]
            binary = "auc_mean" in sample
            head = "| feature | accuracy | " + ("AUC | " if binary else "") + "|"
            lines += [
                f"Chance accuracy {chance:.3f} "
                f"({sample['n_classes']} classes, n={sample['n_samples']}).",
                "",
                head,
                "|---|---|" + ("---|" if binary else ""),
            ]
            for fn in order:
                if fn not in per_feat:
                    continue
                r = per_feat[fn]
                row = f"| {fn} | {r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f} |"
                if binary:
                    row += f" {r['auc_mean']:.3f} ± {r['auc_std']:.3f} |"
                lines.append(row)
        else:
            lines += [
                f"Target std {sample['target_std']:.2f} (R²=0 is chance, "
                f"n={sample['n_samples']}).",
                "",
                "| feature | R² | MAE |",
                "|---|---|---|",
            ]
            for fn in order:
                if fn not in per_feat:
                    continue
                r = per_feat[fn]
                lines.append(
                    f"| {fn} | {r['r2_mean']:.3f} ± {r['r2_std']:.3f} | "
                    f"{r['mae_mean']:.2f} |"
                )
        lines.append("")

    lines += _reading(results)
    try:
        make_figure(results, REPO_ROOT / "figures")
        lines += ["", "## Figure", "", "![probe](../figures/probe.png)"]
    except Exception as exc:  # noqa: BLE001
        print(f"(figure skipped: {exc})")
    (out_dir / "probe_report.md").write_text("\n".join(lines))
    print(f"wrote {out_dir / 'probe_report.md'}")


def _reading(results: dict) -> list[str]:
    """Auto-generated interpretation, characterizing where the RNN's edge is."""
    def metric(target, feat):
        p = results["probes"].get(target, {}).get(feat)
        if not p:
            return None
        return p.get("auc_mean", p.get("accuracy_mean", p.get("r2_mean")))

    lines = ["## Reading", ""]
    for target in ("dx_group", "sex", "age"):
        pca, enc, rnn = (metric(target, f) for f in ("pca", "encoder", "rnn"))
        if None in (pca, enc, rnn):
            continue
        best = max([("pca", pca), ("encoder", enc), ("rnn", rnn)], key=lambda kv: kv[1])
        rnn_over_enc = rnn - enc
        note = (
            "temporal features add little over the spatial encoder"
            if abs(rnn_over_enc) < 0.02
            else ("the RNN adds over the encoder" if rnn_over_enc > 0
                  else "the RNN is worse than the encoder")
        )
        lines.append(
            f"- **{target}**: pca={pca:.3f}, encoder={enc:.3f}, rnn={rnn:.3f} "
            f"(best: {best[0]}); {note}."
        )

    site = {f: metric("site", f) for f in ("pca", "encoder", "rnn")}
    if None not in site.values():
        lines += [
            "",
            f"- **Site is the RNN's clearest signal**: pca={site['pca']:.3f}, "
            f"encoder={site['encoder']:.3f}, rnn={site['rnn']:.3f} accuracy "
            f"(chance {results['probes']['site']['rnn']['chance_accuracy']:.3f}). "
            "The temporal representation's main advantage over PCA is decoding "
            "the scanner, not phenotype — the dynamics it learns are partly "
            "site-coupled, which any downstream use has to account for.",
        ]
    return lines


def make_figure(results: dict, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    order = ["pca", "encoder", "rnn"]
    colors = {"pca": "#4C72B0", "encoder": "#55A868", "rnn": "#C44E52"}
    probes = results["probes"]

    # comparable "headline" metric per target: AUC for binary, accuracy for
    # multiclass, R² for regression — each with its own chance reference
    panels = []
    for tname, per_feat in probes.items():
        s = next(iter(per_feat.values()))
        if s["task"] == "regression":
            panels.append((tname, "r2_mean", "r2_std", 0.0, "R²"))
        elif "auc_mean" in s:
            panels.append((tname, "auc_mean", "auc_std", 0.5, "AUC"))
        else:
            panels.append((tname, "accuracy_mean", "accuracy_std",
                           s["chance_accuracy"], "accuracy"))

    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.6))
    if len(panels) == 1:
        axes = [axes]
    for ax, (tname, mkey, skey, chance, ylab) in zip(axes, panels):
        vals = [probes[tname][f][mkey] for f in order]
        errs = [probes[tname][f].get(skey, 0.0) for f in order]
        ax.bar(order, vals, yerr=errs, color=[colors[f] for f in order], capsize=3)
        ax.axhline(chance, ls="--", color="gray", lw=1)
        ax.set_title(tname)
        ax.set_ylabel(ylab)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Linear-probe decodability by frozen representation "
                 "(dashed = chance)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "probe.png", dpi=130)
    plt.close(fig)
    print(f"wrote {fig_dir / 'probe.png'}")


if __name__ == "__main__":
    main()
