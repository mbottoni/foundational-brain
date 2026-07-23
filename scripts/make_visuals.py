#!/usr/bin/env python
"""Generate data-driven animations that explain the project to a non-expert.

Everything here is rendered from *real* ABIDE data and the *trained* model — no
mock-ups. Three GIFs, each carrying one idea:

* ``brain_activity.gif``    — what fMRI is: brain-region activity changing over
  time (a schematic grid, one cell per region, not an anatomical map).
* ``forecast.gif``          — what the model does: predict the next moment of
  brain activity from the recent past, tracked against the truth.
* ``latent_trajectory.gif`` — how it represents a brain: as a path through a
  compressed "state space", the thing the model actually learns to continue.

It also writes ``viz_data.json`` — small real arrays the web explainer animates
live in the browser, so the Artifact needs no heavy embedded video.

    python scripts/make_visuals.py --checkpoint checkpoints/foundation_model.pt
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from foundational_brain.data.pipeline import prepare_pretrain_data  # noqa: E402
from foundational_brain.training.trainer import pick_device  # noqa: E402
from probe_features import load_model  # noqa: E402

# consistent, colourblind-friendly accents used across all three animations
C_TRUE = "#4C72B0"
C_PRED = "#C44E52"
C_TRAIL = "#55A868"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "checkpoints/foundation_model.pt")
    p.add_argument("--n-subjects", type=int, default=0,
                   help="0 = all; needed because ABIDE is ordered by site, so a "
                        "small count selects zero TR=2.0s subjects")
    p.add_argument("--max-frames", type=int, default=150)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs" / "assets")
    return p.parse_args()


def pick_subject(data, max_frames: int):
    """Longest available subject from the training split, capped to max_frames."""
    series = data.split_series["train"]
    pheno = data.split_phenotypic["train"]
    idx = int(np.argmax([s.shape[0] for s in series]))
    s = series[idx]
    T = min(s.shape[0], max_frames)
    return s[:T], pheno.iloc[idx]


@torch.no_grad()
def model_traces(model, s: np.ndarray, device):
    """Latent path, one-step forecasts, and the reconstruction for one subject."""
    x = torch.from_numpy(np.ascontiguousarray(s)).unsqueeze(0).to(device)
    z = model.encode(x)                       # (1, T, latent)
    _, _, feats = model.latent_rnn(z)         # (1, T, rnn_hidden)
    out = model(x)
    return {
        "latent": z[0].cpu().numpy(),
        "rnn": feats[0].cpu().numpy(),
        "x_recon": out["x_recon"][0].cpu().numpy(),
        "x_next_pred": out["x_next_pred"][0].cpu().numpy(),
    }


def pca_2d(x: np.ndarray) -> np.ndarray:
    """Project a (T, d) sequence to its top-2 principal components."""
    xc = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    return xc @ vt[:2].T


# ---------------------------------------------------------------------------
# animations
# ---------------------------------------------------------------------------


def save_gif(fig, update, frames: int, out: Path, fps: int, step: int = 1) -> None:
    """Render each frame with savefig and assemble the GIF with Pillow.

    Deliberately not matplotlib's PillowWriter: that path grabs the raw RGBA
    canvas buffer, and when the figure's pixel width is odd it misreads the row
    stride and shears every frame diagonally. savefig always rasterizes
    correctly, so this is slower but cannot shear.
    """
    import io

    from PIL import Image

    imgs = []
    for f in range(0, frames, step):
        update(f)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        buf.seek(0)
        imgs.append(Image.open(buf).convert("RGB"))
    imgs[0].save(
        out, save_all=True, append_images=imgs[1:],
        duration=int(1000 / fps), loop=0, optimize=True,
    )


def anim_brain_activity(s: np.ndarray, out: Path, fps: int) -> None:
    import matplotlib.pyplot as plt
    T = s.shape[0]
    grid = s.reshape(T, 14, 13)
    traces = s[:, [20, 90, 150]]  # three example regions to scroll

    fig, (axg, axt) = plt.subplots(1, 2, figsize=(8.2, 3.8),
                                   gridspec_kw={"width_ratios": [1, 1.3]})
    im = axg.imshow(grid[0], cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    axg.set_title("Brain state right now", fontsize=11)
    axg.set_xticks([]); axg.set_yticks([])
    axg.text(0.5, -0.09, "each cell = one brain region (schematic)",
             transform=axg.transAxes, ha="center", fontsize=8, color="gray")

    t = np.arange(T)
    lines = [axt.plot([], [], lw=1.6, color=c)[0]
             for c in (C_TRUE, C_TRAIL, C_PRED)]
    axt.set_xlim(0, T); axt.set_ylim(traces.min() - 0.5, traces.max() + 0.5)
    axt.set_title("Three regions over time", fontsize=11)
    axt.set_xlabel("time (scan frames)")
    axt.set_ylabel("activity (z-scored)")
    cursor = axt.axvline(0, color="k", lw=0.8, alpha=0.4)

    def update(f):
        im.set_data(grid[f])
        for i, ln in enumerate(lines):
            ln.set_data(t[: f + 1], traces[: f + 1, i])
        cursor.set_xdata([f, f])
        return [im, cursor, *lines]

    fig.suptitle("fMRI: watching a brain change, moment by moment", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_gif(fig, update, T, out, fps)
    plt.close(fig)
    print(f"wrote {out}")


def anim_forecast(s: np.ndarray, x_next_pred: np.ndarray, out: Path, fps: int) -> None:
    import matplotlib.pyplot as plt
    T = s.shape[0]
    region = 90  # one clear region
    true = s[:, region]
    # step t predicts t+1; align the prediction to the frame it forecasts
    pred = np.empty(T)
    pred[:] = np.nan
    pred[1:] = x_next_pred[:-1, region]

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    (ln_true,) = ax.plot([], [], lw=2.0, color=C_TRUE, label="what the brain did")
    (ln_pred,) = ax.plot([], [], lw=1.4, color=C_PRED, ls="--",
                         label="what the model predicted")
    dot = ax.scatter([], [], s=45, color=C_PRED, zorder=5)
    ax.set_xlim(0, T); ax.set_ylim(true.min() - 0.6, true.max() + 0.6)
    ax.set_xlabel("time (scan frames)"); ax.set_ylabel("activity (z-scored)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Predicting the next moment of brain activity", fontsize=12)

    tt = np.arange(T)

    def update(f):
        ln_true.set_data(tt[: f + 1], true[: f + 1])
        ln_pred.set_data(tt[: f + 1], pred[: f + 1])
        if f >= 1 and np.isfinite(pred[f]):
            dot.set_offsets([[f, pred[f]]])
        return [ln_true, ln_pred, dot]

    fig.tight_layout()
    save_gif(fig, update, T, out, fps)
    plt.close(fig)
    print(f"wrote {out}")


def anim_latent_trajectory(rnn: np.ndarray, out: Path, fps: int) -> np.ndarray:
    import matplotlib.pyplot as plt
    xy = pca_2d(rnn)
    T = xy.shape[0]

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot(xy[:, 0], xy[:, 1], lw=0.7, color="gray", alpha=0.25)  # full path, faint
    (trail,) = ax.plot([], [], lw=2.0, color=C_TRAIL)
    head = ax.scatter([], [], s=70, color=C_PRED, zorder=5)
    pad = 0.5
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("The brain's path through the model's 'state space'", fontsize=11)
    ax.text(0.5, -0.06, "each point = one whole-brain state, compressed to 2-D",
            transform=ax.transAxes, ha="center", fontsize=8, color="gray")

    tail = 25

    def update(f):
        lo = max(0, f - tail)
        trail.set_data(xy[lo : f + 1, 0], xy[lo : f + 1, 1])
        head.set_offsets([xy[f]])
        return [trail, head]

    fig.tight_layout()
    save_gif(fig, update, T, out, fps)
    plt.close(fig)
    print(f"wrote {out}")
    return xy


def main() -> None:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    device = pick_device(args.device)

    if not args.checkpoint.exists():
        raise SystemExit(f"no checkpoint at {args.checkpoint}; run pretrain.py first")
    model, ck = load_model(args.checkpoint, device)

    data = prepare_pretrain_data(
        n_subjects=args.n_subjects or None,
        seq_len=ck["config"]["data"]["seq_len"], seed=args.seed, verbose=False,
    )
    s, pheno = pick_subject(data, args.max_frames)
    print(f"subject: site {pheno['SITE_ID']}, age {pheno['AGE_AT_SCAN']:.0f}, "
          f"{s.shape[0]} frames, {s.shape[1]} regions")

    tr = model_traces(model, s, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    anim_brain_activity(s, args.out_dir / "brain_activity.gif", args.fps)
    anim_forecast(s, tr["x_next_pred"], args.out_dir / "forecast.gif", args.fps)
    xy = anim_latent_trajectory(tr["rnn"], args.out_dir / "latent_trajectory.gif",
                                args.fps)

    # small real arrays for the browser explainer to animate live
    region = 90
    pred = np.concatenate([[np.nan], tr["x_next_pred"][:-1, region]])
    viz = {
        "n_frames": int(s.shape[0]),
        "grid_shape": [14, 13],
        "brain_grid": np.round(s, 3).tolist(),                 # (T, 182)
        "latent_xy": np.round(xy, 3).tolist(),                 # (T, 2)
        "forecast_true": np.round(s[:, region], 3).tolist(),   # (T,)
        "forecast_pred": [None if np.isnan(v) else round(float(v), 3) for v in pred],
        "example_regions": np.round(s[:, [20, 90, 150]], 3).T.tolist(),  # (3, T)
    }
    (args.out_dir / "viz_data.json").write_text(json.dumps(viz))
    print(f"wrote {args.out_dir / 'viz_data.json'} "
          f"({(args.out_dir / 'viz_data.json').stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
