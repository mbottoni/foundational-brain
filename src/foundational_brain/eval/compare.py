"""Per-subject paired comparison between the model and a baseline.

A single averaged MSE cannot distinguish "better on every subject" from
"much better on a handful and worse on the rest", and it carries no
uncertainty. Since both predictors are evaluated on exactly the same subjects,
the comparison is naturally paired, which is both more powerful and more honest
than comparing two pooled means.

Reports three things:

* **win rate** — the fraction of subjects where the model has lower error. A
  large mean improvement with a win rate near 50% means a few subjects are
  carrying the result.
* **bootstrap CI** on the mean paired difference — does the interval exclude
  zero?
* **Wilcoxon signed-rank p-value** — a distribution-free paired test, since
  per-subject MSE differences are not remotely Gaussian.
"""

from __future__ import annotations

import numpy as np


def paired_difference(
    model_mse: np.ndarray, baseline_mse: np.ndarray
) -> dict[str, float]:
    """Summarize per-subject ``baseline - model`` (positive = model better)."""
    model_mse = np.asarray(model_mse, dtype=np.float64)
    baseline_mse = np.asarray(baseline_mse, dtype=np.float64)
    if model_mse.shape != baseline_mse.shape:
        raise ValueError(
            f"paired inputs must align: {model_mse.shape} vs {baseline_mse.shape}"
        )
    diff = baseline_mse - model_mse
    return {
        "n_subjects": int(len(diff)),
        "mean_model_mse": float(model_mse.mean()),
        "mean_baseline_mse": float(baseline_mse.mean()),
        "mean_improvement": float(diff.mean()),
        "median_improvement": float(np.median(diff)),
        "relative_improvement": float(diff.mean() / max(baseline_mse.mean(), 1e-12)),
        "win_rate": float((diff > 0).mean()),
    }


def bootstrap_ci(
    model_mse: np.ndarray,
    baseline_mse: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean paired difference.

    Resamples *subjects*, not windows — windows within a subject are highly
    correlated, so bootstrapping them would understate the interval badly.
    """
    diff = np.asarray(baseline_mse, np.float64) - np.asarray(model_mse, np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return (
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


def wilcoxon_p(model_mse: np.ndarray, baseline_mse: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value for the paired difference."""
    from scipy.stats import wilcoxon

    diff = np.asarray(baseline_mse, np.float64) - np.asarray(model_mse, np.float64)
    if np.allclose(diff, 0):
        return 1.0
    return float(wilcoxon(diff).pvalue)


def compare(
    model_mse: np.ndarray,
    baseline_mse: np.ndarray,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict:
    """Full paired comparison: effect size, win rate, CI and p-value."""
    res = paired_difference(model_mse, baseline_mse)
    lo, hi = bootstrap_ci(model_mse, baseline_mse, n_boot=n_boot, seed=seed)
    res["ci95_low"] = lo
    res["ci95_high"] = hi
    res["ci_excludes_zero"] = bool(lo > 0 or hi < 0)
    res["wilcoxon_p"] = wilcoxon_p(model_mse, baseline_mse)
    return res
