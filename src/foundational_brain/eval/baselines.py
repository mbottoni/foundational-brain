"""Reference predictors the foundation model has to beat.

The corpus-wide numbers in ``docs/data_report.md`` were computed over all
subjects. They are the right thing for planning and the wrong thing for
judging a trained model: the model is scored on a particular validation split,
so the baseline must be too, or the comparison silently measures which subjects
landed where.

All baselines operate on per-region z-scored series, so MSE is in units of
signal variance and 1.0 means "no better than predicting the mean".

* ``mean``        - predict 0.
* ``persistence`` - predict x_t. Free, no fitting.
* ``ar1``         - predict rho_r * x_t with per-region rho fitted **on the
  training split only** and applied unchanged to val/test. Fitting rho on the
  evaluation data would give the baseline an advantage the model does not get.
* ``pca_k``       - reconstruct x_t from its top-k principal components, with
  the basis fitted on training data. This is the reconstruction bar: if the
  autoencoder cannot beat a linear projection of the same width, its
  nonlinearity is decorative.
"""

from __future__ import annotations

import numpy as np


def fit_ar1(series: list[np.ndarray]) -> np.ndarray:
    """Per-region lag-1 coefficient pooled over the given (training) subjects."""
    num = None
    den = None
    for s in series:
        x = np.asarray(s, dtype=np.float64)
        n = (x[:-1] * x[1:]).sum(axis=0)
        d = (x[:-1] ** 2).sum(axis=0)
        num = n if num is None else num + n
        den = d if den is None else den + d
    return (num / np.maximum(den, 1e-12)).astype(np.float32)


def fit_pca(series: list[np.ndarray], n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit a PCA basis on pooled training frames -> ``(components, mean)``."""
    frames = np.concatenate([np.asarray(s, np.float64) for s in series], axis=0)
    mean = frames.mean(axis=0)
    centered = frames - mean
    cov = (centered.T @ centered) / max(len(centered) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1][:n_components]
    return evecs[:, order].astype(np.float32), mean.astype(np.float32)


def forecast_mse(
    series: list[np.ndarray],
    ar1_rho: np.ndarray | None = None,
    horizon: int = 1,
) -> dict[str, float]:
    """One-step forecasting MSE for each baseline, averaged over frames.

    Averaged over *frames*, not subjects, so it matches how a training loop
    aggregates its own loss.
    """
    tot = {"mean": 0.0, "persistence": 0.0, "ar1": 0.0}
    count = 0
    for s in series:
        x = np.asarray(s, dtype=np.float64)
        if x.shape[0] <= horizon:
            continue
        src, tgt = x[:-horizon], x[horizon:]
        n = tgt.size
        tot["mean"] += float((tgt**2).sum())
        tot["persistence"] += float(((tgt - src) ** 2).sum())
        if ar1_rho is not None:
            pred = (ar1_rho.astype(np.float64) ** horizon) * src
            tot["ar1"] += float(((tgt - pred) ** 2).sum())
        count += n
    if count == 0:
        return {}
    out = {k: v / count for k, v in tot.items()}
    if ar1_rho is None:
        out.pop("ar1")
    return out


def reconstruction_mse(
    series: list[np.ndarray], components: np.ndarray, mean: np.ndarray
) -> float:
    """MSE of reconstructing each frame from a fixed PCA basis."""
    tot, count = 0.0, 0
    for s in series:
        x = np.asarray(s, dtype=np.float32)
        centered = x - mean
        recon = (centered @ components) @ components.T + mean
        tot += float(((x - recon) ** 2).sum())
        count += x.size
    return tot / max(count, 1)


def evaluate_baselines(
    train_series: list[np.ndarray],
    eval_series: list[np.ndarray],
    eval_sites: list[str] | None = None,
    horizons: tuple[int, ...] = (1,),
    pca_components: int = 128,
) -> dict:
    """Fit every baseline on train, score it on eval, overall and per site."""
    rho = fit_ar1(train_series)
    comps, mean = fit_pca(train_series, n_components=pca_components)

    out: dict = {
        "n_train_subjects": len(train_series),
        "n_eval_subjects": len(eval_series),
        "ar1_rho_median": float(np.median(rho)),
        "forecast": {
            f"h{h}": forecast_mse(eval_series, rho, horizon=h) for h in horizons
        },
        "reconstruction": {
            f"pca_{pca_components}": reconstruction_mse(eval_series, comps, mean)
        },
    }

    if eval_sites is not None:
        sites = np.asarray(eval_sites)
        per_site = {}
        for site in np.unique(sites):
            idx = np.where(sites == site)[0]
            sub = [eval_series[i] for i in idx]
            per_site[str(site)] = {
                "n_subjects": int(len(sub)),
                "forecast_h1": forecast_mse(sub, rho, horizon=1),
                f"reconstruction_pca_{pca_components}": reconstruction_mse(
                    sub, comps, mean
                ),
            }
        out["per_site"] = per_site

    return out
