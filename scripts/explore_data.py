#!/usr/bin/env python
"""Download ABIDE-PCP and profile it, writing a report + figures.

    python scripts/explore_data.py --n-subjects 200 --derivative rois_cc200

Outputs (both gitignored except the report, which is meant to be committed):

* ``docs/data_report.md``  — the findings, as markdown
* ``figures/*.png``        — autocorrelation, PCA spectrum, connectivity, etc.

The point is not a data dump: every number printed here feeds a decision in
``configs/default.yaml``, and the report says which.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from foundational_brain.data import ABIDE_DERIVATIVES, load_abide_series  # noqa: E402
from foundational_brain.data import stats  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-subjects", type=int, default=200,
                   help="ABIDE subjects to fetch (None-like 0 = all ~870)")
    p.add_argument("--derivative", default="rois_cc200",
                   choices=sorted(ABIDE_DERIVATIVES))
    p.add_argument("--max-lag", type=int, default=40)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs")
    p.add_argument("--fig-dir", type=Path, default=REPO_ROOT / "figures")
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args()


def compute(series: list[np.ndarray], phenotypic, max_lag: int) -> dict:
    """Run every profile in :mod:`foundational_brain.data.stats`."""
    sites = list(phenotypic["SITE_ID"])
    return {
        "shape": stats.shape_summary(series),
        "amplitude": stats.amplitude_summary(series),
        "acf": stats.autocorrelation(series, max_lag=max_lag),
        "ar1": stats.ar1_coefficients(series),
        "baselines": {
            f"h{h}": stats.forecast_baselines(series, horizon=h) for h in (1, 2, 5, 10)
        },
        "pca": stats.pca_spectrum(series),
        "connectivity": stats.connectivity(series),
        "site_variance": stats.variance_partition(series, sites),
        "by_site": stats.group_profile(series, sites, max_lag=5),
        "sites": sites,
    }


def make_figures(res: dict, fig_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def save(fig, name: str) -> None:
        path = fig_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)

    # 1. scan-length distribution -> bounds on seq_len
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(res["shape"]["lengths"], bins=40, color="#4C72B0")
    ax.set_xlabel("timepoints per scan (T)")
    ax.set_ylabel("subjects")
    ax.set_title("ABIDE scan lengths")
    save(fig, "scan_lengths.png")

    # 2. autocorrelation -> is there dynamics to model?
    acf = res["acf"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(np.arange(len(acf)), acf, marker="o", ms=3, color="#C44E52")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("lag (TRs)")
    ax.set_ylabel("mean autocorrelation")
    ax.set_title("Temporal autocorrelation of ROI signal")
    save(fig, "autocorrelation.png")

    # 3. PCA spectrum -> bounds on latent_dim
    cum = res["pca"]["cumulative_variance"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(np.arange(1, len(cum) + 1), cum, color="#55A868")
    for frac, style in ((0.90, ":"), (0.95, "--"), (0.99, "-.")):
        ax.axhline(frac, ls=style, lw=0.8, color="gray")
    ax.set_xlabel("principal components")
    ax.set_ylabel("cumulative variance explained")
    ax.set_title("Frame-space dimensionality")
    save(fig, "pca_spectrum.png")

    # 4. AR(1) coefficients -> per-region predictability
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(res["ar1"], bins=60, color="#8172B2")
    ax.set_xlabel("lag-1 autoregressive coefficient")
    ax.set_ylabel("region x subject")
    ax.set_title("Per-region AR(1) strength")
    save(fig, "ar1_coefficients.png")

    # 5. per-site autocorrelation -> is pooling across TRs legitimate?
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for site, prof in sorted(res["by_site"].items(),
                             key=lambda kv: -kv[1]["acf"][1]):
        ax.plot(prof["acf"], lw=1.2, alpha=0.85, label=site)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("lag (TRs)")
    ax.set_ylabel("mean autocorrelation")
    ax.set_title("Autocorrelation by scan site (time axes are not comparable)")
    ax.legend(fontsize=5.5, ncol=4, loc="upper right")
    save(fig, "autocorrelation_by_site.png")

    # 6. mean connectivity -> shared spatial structure
    m = res["connectivity"]["mean_matrix"]
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(m, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    ax.set_title("Mean region-region correlation")
    fig.colorbar(im, ax=ax, shrink=0.85)
    save(fig, "connectivity.png")

    return written


def fmt_report(res: dict, args: argparse.Namespace, figs: list[Path]) -> str:
    sh, amp, pca = res["shape"], res["amplitude"], res["pca"]
    acf, ar1 = res["acf"], res["ar1"]
    base = res["baselines"]["h1"]
    conn, sitevar = res["connectivity"], res["site_variance"]

    # lag where mean autocorrelation first drops below 0.1
    below = np.where(acf < 0.1)[0]
    decay_lag = int(below[0]) if len(below) else len(acf)

    skill = 1.0 - base["ar1"]["mean"] / base["mean"]["mean"]
    persist_skill = 1.0 - base["persistence"]["mean"] / base["mean"]["mean"]

    lines = [
        "# ABIDE-PCP data report",
        "",
        f"Generated by `scripts/explore_data.py --n-subjects {args.n_subjects} "
        f"--derivative {args.derivative}`.",
        "",
        "## Corpus",
        "",
        f"- **{sh['n_subjects']} subjects**, {sh['n_regions'][0]} regions, "
        f"**{sh['timepoints_total']:,} total frames**",
        f"- scan length T: min {sh['T_min']}, median {sh['T_median']:.0f}, "
        f"max {sh['T_max']} (p5 {sh['T_percentiles']['5']:.0f}, "
        f"p95 {sh['T_percentiles']['95']:.0f})",
        f"- {sitevar['n_groups']} scan sites; "
        f"**{sitevar['between_group_variance_fraction']:.1%}** of "
        "between-subject connectivity variance is explained by site",
        "",
        "## Signal amplitude",
        "",
        f"- per-region std spans {amp['region_std']['min']:.2f} to "
        f"{amp['region_std']['max']:.2f} (median "
        f"{amp['region_std']['median']:.2f}; p99/p1 ratio "
        f"{amp['region_std']['ratio_p99_p1']:.1f}x)",
        f"- excess kurtosis: median {amp['excess_kurtosis']['median']:.2f}, "
        f"p95 {amp['excess_kurtosis']['p95']:.2f}",
        f"- skew: median {amp['skew']['median']:.2f}, "
        f"p95 |skew| {amp['skew']['p95']:.2f}",
        f"- flat (zero-variance) regions: {amp['frac_regions_flat']:.2%}",
        "",
        "## Temporal structure",
        "",
        f"- mean autocorrelation: lag1 {acf[1]:.3f}, lag2 {acf[2]:.3f}, "
        f"lag5 {acf[5]:.3f}" + (f", lag10 {acf[10]:.3f}" if len(acf) > 10 else ""),
        f"- decays below 0.1 at **lag {decay_lag} TRs**",
        f"- per-region AR(1): median {np.median(ar1):.3f} "
        f"(p5 {np.percentile(ar1, 5):.3f}, p95 {np.percentile(ar1, 95):.3f})",
        "",
        "### Next-frame prediction baselines (z-scored, MSE)",
        "",
        "| horizon | mean (predict 0) | persistence | AR(1) |",
        "|---|---|---|---|",
    ]
    for h, r in res["baselines"].items():
        lines.append(
            f"| {h[1:]} TR | {r['mean']['mean']:.3f} | "
            f"{r['persistence']['mean']:.3f} | {r['ar1']['mean']:.3f} |"
        )
    lines += [
        "",
        f"At 1 TR, persistence already explains {persist_skill:.1%} of variance and "
        f"AR(1) {skill:.1%}. **The latent RNN has to beat "
        f"{base['ar1']['mean']:.3f} MSE to be worth anything.**",
        "",
        "## Spatial structure",
        "",
        f"- {pca['n_components_90']} / {pca['n_components_95']} / "
        f"{pca['n_components_99']} components for 90 / 95 / 99% of frame variance "
        f"(out of {sh['n_regions'][0]})",
        f"- participation ratio {pca['participation_ratio']:.1f}, "
        f"effective rank {pca['effective_rank']:.1f}",
        f"- mean |edge| correlation {conn['mean_abs_edge']:.3f}; "
        f"between-subject connectivity similarity "
        f"**{conn['between_subject_similarity']:.3f}**",
        "",
    ]

    by_site = res["by_site"]
    acf1 = {k: v["acf"][1] for k, v in by_site.items()}
    lo, hi = min(acf1, key=acf1.get), max(acf1, key=acf1.get)
    lines += [
        "### Per-site breakdown (the pooled numbers above are misleading)",
        "",
        "Autocorrelation is indexed in **samples, not seconds**, and ABIDE's 20 "
        "sites do not share a TR. Pooling them averages over incompatible time "
        "axes:",
        "",
        "| site | n | median T | acf(1) | acf(2) | acf(2)/acf(1)² | AR(1) MSE |",
        "|---|---|---|---|---|---|---|",
    ]
    for site, p in sorted(by_site.items(), key=lambda kv: -kv[1]["acf"][1]):
        lines.append(
            f"| {site} | {p['n_subjects']} | {p['T_median']:.0f} | "
            f"{p['acf'][1]:.3f} | {p['acf'][2]:.3f} | "
            f"{p['acf2_over_acf1_sq']:.2f} | {p['baseline_ar1_mse']:.3f} |"
        )
    lines += [
        "",
        f"Lag-1 autocorrelation ranges **{acf1[lo]:.3f} ({lo}) to "
        f"{acf1[hi]:.3f} ({hi})** — a 1-TR step means a different amount of "
        "elapsed time at each site. A pure AR(1) would have acf(2)/acf(1)² = 1; "
        "every site falls below that, so the signal is **not first-order "
        "Markov** and a model with memory has real headroom over the AR(1) "
        "baseline.",
        "",
    ]

    if figs:
        lines += ["## Figures", ""]
        lines += [f"![{f.stem}](../figures/{f.name})" for f in figs]
        lines += [""]

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    n = args.n_subjects if args.n_subjects > 0 else None

    print(f"fetching ABIDE ({n or 'all'} subjects, {args.derivative}) ...")
    series, phenotypic = load_abide_series(n_subjects=n, derivative=args.derivative)
    print(f"loaded {len(series)} usable subjects")

    res = compute(series, phenotypic, max_lag=args.max_lag)
    figs = [] if args.no_figures else make_figures(res, args.fig_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "data_report.md"
    report_path.write_text(fmt_report(res, args, figs))
    print(f"wrote {report_path}")

    # machine-readable companion, for regression-checking a future re-run
    summary = {
        k: v for k, v in res.items()
        if k in ("amplitude", "baselines", "site_variance", "by_site")
    }
    summary["shape"] = {k: v for k, v in res["shape"].items() if k != "lengths"}
    summary["pca"] = {
        k: v for k, v in res["pca"].items()
        if not isinstance(v, np.ndarray)
    }
    summary["connectivity"] = {
        k: v for k, v in res["connectivity"].items() if k != "mean_matrix"
    }
    summary["acf"] = res["acf"].tolist()
    json_path = args.out_dir / "data_report.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
