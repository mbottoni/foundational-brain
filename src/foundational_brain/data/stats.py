"""Descriptive statistics for parcellated fMRI, aimed at model design choices.

Each function here exists to answer a question that sets a hyperparameter or
tests an assumption baked into the architecture:

===========================  =================================================
question                     decides
===========================  =================================================
how long are the scans?      ``seq_len`` (can't window longer than the scans)
how heavy-tailed is signal?  normalization scheme; MSE vs robust loss
how autocorrelated is it?    whether an RNN over latents has anything to learn
how low-rank is a frame?     ``latent_dim`` (don't exceed the real rank)
how strong is persistence?   the baseline the forecasting loss must beat
how much variance is site?   whether pretraining learns dynamics or scanners
===========================  =================================================

Everything takes a list of ``(T, n_regions)`` arrays and returns plain dicts /
arrays, so the reporting layer stays free of statistics.
"""

from __future__ import annotations

import numpy as np


def zscore(x: np.ndarray, axis: int = 0, eps: float = 1e-8) -> np.ndarray:
    """Z-score along ``axis`` (default: per-region across time)."""
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    return (x - mu) / (sd + eps)


# --------------------------------------------------------------------------
# shape / amplitude
# --------------------------------------------------------------------------


def shape_summary(series: list[np.ndarray]) -> dict:
    """Scan-length and region-count distribution across subjects."""
    lengths = np.array([s.shape[0] for s in series])
    regions = {int(s.shape[1]) for s in series}
    return {
        "n_subjects": len(series),
        "n_regions": sorted(regions),
        "timepoints_total": int(lengths.sum()),
        "T_min": int(lengths.min()),
        "T_max": int(lengths.max()),
        "T_mean": float(lengths.mean()),
        "T_median": float(np.median(lengths)),
        "T_percentiles": {
            str(p): float(np.percentile(lengths, p)) for p in (5, 25, 50, 75, 95)
        },
        "lengths": lengths,
    }


def amplitude_summary(series: list[np.ndarray]) -> dict:
    """Signal distribution before normalization: scale, skew, tails.

    ABIDE derivatives are not z-scored on delivery, so per-region scale varies
    by orders of magnitude — this is what motivates per-region z-scoring rather
    than a global scale factor.
    """
    per_region_mean, per_region_std, kurt, skew = [], [], [], []
    for s in series:
        per_region_mean.append(s.mean(axis=0))
        per_region_std.append(s.std(axis=0))
        z = zscore(s)
        kurt.append((z**4).mean(axis=0) - 3.0)
        skew.append((z**3).mean(axis=0))

    mean = np.concatenate(per_region_mean)
    std = np.concatenate(per_region_std)
    kurt = np.concatenate(kurt)
    skew = np.concatenate(skew)
    return {
        "region_mean": {"min": float(mean.min()), "max": float(mean.max()),
                        "median": float(np.median(mean))},
        "region_std": {"min": float(std.min()), "max": float(std.max()),
                       "median": float(np.median(std)),
                       "ratio_p99_p1": float(
                           np.percentile(std, 99) / max(np.percentile(std, 1), 1e-12)
                       )},
        "excess_kurtosis": {"median": float(np.median(kurt)),
                            "p95": float(np.percentile(kurt, 95))},
        "skew": {"median": float(np.median(skew)),
                 "p95": float(np.percentile(np.abs(skew), 95))},
        "frac_regions_flat": float((std < 1e-6).mean()),
    }


# --------------------------------------------------------------------------
# temporal structure
# --------------------------------------------------------------------------


def autocorrelation(series: list[np.ndarray], max_lag: int = 30) -> np.ndarray:
    """Mean autocorrelation curve over lags ``1..max_lag``.

    Averaged over regions and subjects on per-region z-scored series. The lag
    at which this decays to ~0 is the horizon over which an autoregressive
    model can predict at all, and therefore a floor for a useful ``seq_len``.
    """
    acc = np.zeros(max_lag + 1)
    n = 0
    for s in series:
        z = zscore(s)
        T = z.shape[0]
        if T <= max_lag + 1:
            continue
        for lag in range(max_lag + 1):
            a = z[: T - lag]
            b = z[lag:]
            acc[lag] += float((a * b).mean())
        n += 1
    return acc / max(n, 1)


def ar1_coefficients(series: list[np.ndarray]) -> np.ndarray:
    """Per-region lag-1 autoregressive coefficient, pooled over subjects."""
    out = []
    for s in series:
        z = zscore(s)
        num = (z[:-1] * z[1:]).sum(axis=0)
        den = (z[:-1] ** 2).sum(axis=0)
        out.append(num / np.maximum(den, 1e-12))
    return np.concatenate(out)


def forecast_baselines(series: list[np.ndarray], horizon: int = 1) -> dict:
    """MSE of trivial next-frame predictors — the bar the RNN must clear.

    On per-region z-scored series (unit variance), these are directly
    comparable across subjects and interpretable as fraction of variance:

    * ``mean``       — predict 0 (the per-region mean). MSE = 1.0 by construction.
    * ``persistence``— predict ``x_t``. Beats ``mean`` iff the signal is
      autocorrelated at this horizon.
    * ``ar1``        — predict ``rho * x_t`` with per-region ``rho`` fitted on
      the same subject. The strongest linear-in-time, region-independent model.

    A latent RNN only earns its keep by beating ``ar1``, since anything less
    is reproducible with 200 scalars and no training.
    """
    mse = {"mean": [], "persistence": [], "ar1": []}
    for s in series:
        z = zscore(s)
        if z.shape[0] <= horizon:
            continue
        x, y = z[:-horizon], z[horizon:]
        rho = (z[:-1] * z[1:]).sum(axis=0) / np.maximum((z[:-1] ** 2).sum(axis=0), 1e-12)
        mse["mean"].append(float((y**2).mean()))
        mse["persistence"].append(float(((y - x) ** 2).mean()))
        mse["ar1"].append(float(((y - (rho**horizon) * x) ** 2).mean()))
    return {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for k, v in mse.items()
        if v
    }


# --------------------------------------------------------------------------
# spatial structure / dimensionality
# --------------------------------------------------------------------------


def pca_spectrum(series: list[np.ndarray], max_frames: int = 20000,
                 seed: int = 0) -> dict:
    """Eigenspectrum of the frame covariance — how low-rank is a brain state?

    Frames are pooled across subjects (per-region z-scored first, so no single
    high-variance region dominates) and subsampled to ``max_frames``. The
    number of components needed for 90/95/99% variance is the empirical ceiling
    on a useful ``latent_dim``: past it, the encoder is modelling noise.
    """
    rng = np.random.default_rng(seed)
    frames = np.concatenate([zscore(s) for s in series], axis=0)
    if frames.shape[0] > max_frames:
        frames = frames[rng.choice(frames.shape[0], max_frames, replace=False)]

    frames = frames - frames.mean(axis=0, keepdims=True)
    cov = (frames.T @ frames) / max(frames.shape[0] - 1, 1)
    evals = np.linalg.eigvalsh(cov)[::-1]
    evals = np.clip(evals, 0.0, None)
    cum = np.cumsum(evals) / max(evals.sum(), 1e-12)

    def n_for(frac: float) -> int:
        return int(np.searchsorted(cum, frac) + 1)

    part = evals / max(evals.sum(), 1e-12)
    entropy = float(-(part[part > 0] * np.log(part[part > 0])).sum())
    return {
        "n_frames_used": int(frames.shape[0]),
        "n_components_90": n_for(0.90),
        "n_components_95": n_for(0.95),
        "n_components_99": n_for(0.99),
        "participation_ratio": float(evals.sum() ** 2 / max((evals**2).sum(), 1e-12)),
        "effective_rank": float(np.exp(entropy)),
        "explained_variance_ratio": part,
        "cumulative_variance": cum,
    }


def connectivity(series: list[np.ndarray]) -> dict:
    """Region-region correlation structure, and how stable it is across subjects.

    ``between_subject_similarity`` is the mean correlation between subjects'
    vectorized connectivity matrices. High values mean the spatial structure is
    shared, i.e. a single encoder can serve all subjects — the premise of
    pretraining across a heterogeneous corpus.
    """
    mats = [np.corrcoef(zscore(s).T) for s in series]
    mats = [np.nan_to_num(m) for m in mats]
    stacked = np.stack(mats)
    iu = np.triu_indices(stacked.shape[1], k=1)
    vecs = stacked[:, iu[0], iu[1]]

    sim = np.corrcoef(vecs)
    off = sim[np.triu_indices(len(sim), k=1)] if len(sim) > 1 else np.array([np.nan])
    return {
        "mean_matrix": stacked.mean(axis=0),
        "mean_abs_edge": float(np.abs(vecs).mean()),
        "between_subject_similarity": float(np.nanmean(off)),
        "edge_std_across_subjects": float(vecs.std(axis=0).mean()),
    }


def variance_partition(series: list[np.ndarray], groups: list) -> dict:
    """Fraction of frame variance explained by group identity (e.g. scan site).

    A one-way decomposition on per-subject mean connectivity: if site explains
    a large share, the pretrained representation risks encoding the scanner
    rather than the brain, and site-stratified splits become mandatory.
    """
    mats = [np.nan_to_num(np.corrcoef(zscore(s).T)) for s in series]
    stacked = np.stack(mats)
    iu = np.triu_indices(stacked.shape[1], k=1)
    vecs = stacked[:, iu[0], iu[1]]

    groups = np.asarray(groups)
    grand = vecs.mean(axis=0, keepdims=True)
    total = ((vecs - grand) ** 2).sum()

    between = 0.0
    for g in np.unique(groups):
        m = groups == g
        between += m.sum() * ((vecs[m].mean(axis=0) - grand[0]) ** 2).sum()

    return {
        "n_groups": int(len(np.unique(groups))),
        "between_group_variance_fraction": float(between / max(total, 1e-12)),
    }
