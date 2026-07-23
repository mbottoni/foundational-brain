#!/usr/bin/env python
"""Is phenotype decodable under connectivity pooling? (Phase 5 follow-up.)

Phase 5 found phenotype only weakly decodable, but used mean+std pooling —
which discards the region-region covariance where resting-state phenotype signal
classically lives. This re-runs the probes with **functional-connectivity**
features for each representation (raw ROI, encoder latent, RNN hidden), so the
question "is the weak result a pooling artifact?" gets an answer.

Two references anchor it:

* ``raw_fc`` — connectivity of the raw ROI series. The classic ABIDE phenotype
  feature; if the *model* representations do not beat this, pretraining adds
  nothing a standard connectivity pipeline lacks.
* the mean+std ``pca`` number from ``docs/probe_report.md`` — the pooling this
  is meant to improve on.

    python scripts/probe_connectivity.py --checkpoint checkpoints/foundation_model.pt

Writes ``docs/probe_connectivity_report.md`` and JSON.
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
from foundational_brain.eval.features import connectivity_features  # noqa: E402
from foundational_brain.eval.probe import probe_all  # noqa: E402
from foundational_brain.models import FoundationModel  # noqa: E402
from foundational_brain.training.trainer import pick_device  # noqa: E402

# reuse the loader/target helpers from the mean+std probe script
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from probe_features import build_targets, load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "checkpoints/foundation_model.pt")
    p.add_argument("--n-subjects", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fc-components", type=int, default=100)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    print(f"device: {device}")

    if not args.checkpoint.exists():
        raise SystemExit(f"no checkpoint at {args.checkpoint}; run pretrain.py first")
    model, ck = load_model(args.checkpoint, device)

    n = args.n_subjects if args.n_subjects > 0 else None
    data = prepare_pretrain_data(
        n_subjects=n, seq_len=ck["config"]["data"]["seq_len"], seed=args.seed
    )

    # same subject ordering as probe_features.py: train + val + test
    series = (data.split_series["train"] + data.split_series["val"]
              + data.split_series["test"])
    import pandas as pd
    pheno = pd.concat(
        [data.split_phenotypic[k] for k in ("train", "val", "test")],
        ignore_index=True,
    )
    n_train = len(data.split_series["train"])
    train_idx = np.arange(n_train)  # first block is the training subjects

    print(f"computing connectivity features for {len(series)} subjects "
          f"(reduce to {args.fc_components} PCs, fit on {n_train} train) ...")
    feats = connectivity_features(model, series, train_idx, device,
                                  n_components=args.fc_components)
    for name, arr in feats.items():
        print(f"  {name:12} {arr.shape}")

    targets = build_targets(pheno)
    results = {"n_subjects": len(series), "fc_components": args.fc_components,
               "probes": {}}
    for tname, (kind, values, mask) in targets.items():
        sub_feats = {fn: arr[mask] for fn, arr in feats.items()}
        res = probe_all(sub_feats, {tname: (kind, values[mask])},
                        n_splits=args.folds, seed=args.seed)
        results["probes"][tname] = res[tname]

    _report(results, args.out_dir)


def _report(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe_connectivity_results.json").write_text(
        json.dumps(results, indent=2)
    )

    order = ["raw_fc", "encoder_fc", "rnn_fc"]

    def head(target, feat):
        p = results["probes"].get(target, {}).get(feat)
        if not p:
            return None
        return p.get("auc_mean", p.get("accuracy_mean", p.get("r2_mean")))

    lines = [
        "# Phenotype under connectivity pooling",
        "",
        f"{results['n_subjects']} subjects, functional connectivity reduced to "
        f"{results['fc_components']} PCs (fit on train). Compare against the "
        "mean+std numbers in `docs/probe_report.md`.",
        "",
    ]
    for tname, per_feat in results["probes"].items():
        s = next(iter(per_feat.values()))
        lines += [f"## {tname}", ""]
        if s["task"] == "classification":
            binary = "auc_mean" in s
            lines += [
                f"Chance accuracy {s['chance_accuracy']:.3f}.",
                "",
                "| feature | accuracy |" + (" AUC |" if binary else ""),
                "|---|---|" + ("---|" if binary else ""),
            ]
            for fn in order:
                r = per_feat[fn]
                row = f"| {fn} | {r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f} |"
                if binary:
                    row += f" {r['auc_mean']:.3f} ± {r['auc_std']:.3f} |"
                lines.append(row)
        else:
            lines += [
                f"Target std {s['target_std']:.2f} (R²=0 is chance).",
                "",
                "| feature | R² | MAE |",
                "|---|---|---|",
            ]
            for fn in order:
                r = per_feat[fn]
                lines.append(
                    f"| {fn} | {r['r2_mean']:.3f} ± {r['r2_std']:.3f} | "
                    f"{r['mae_mean']:.2f} |"
                )
        lines.append("")

    # verdict on the pooling-artifact question, anchored on DX group
    dx_raw = head("dx_group", "raw_fc")
    dx_best_model = max(
        (head("dx_group", f) for f in ("encoder_fc", "rnn_fc")), default=None
    )
    lines += ["## Reading", ""]
    if dx_raw is not None:
        lines.append(
            f"- DX-group AUC under connectivity: raw_fc={dx_raw:.3f}, best model "
            f"representation={dx_best_model:.3f}. Compare to ~0.63 under mean+std."
        )
    (out_dir / "probe_connectivity_report.md").write_text("\n".join(lines))
    print(f"wrote {out_dir / 'probe_connectivity_report.md'}")


if __name__ == "__main__":
    main()
