"""Tests for the reference predictors.

If a baseline is wrong the model's headline claim is wrong, in whichever
direction the error happens to point. These check the analytic cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from foundational_brain.eval.baselines import (
    evaluate_baselines,
    fit_ar1,
    fit_pca,
    forecast_mse,
    reconstruction_mse,
)


def ar1_series(T, n_regions, rho, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((T, n_regions), dtype=np.float32)
    noise = rng.standard_normal((T, n_regions)) * np.sqrt(1 - rho**2)
    for t in range(1, T):
        x[t] = rho * x[t - 1] + noise[t]
    return x


@pytest.mark.parametrize("rho", [0.2, 0.5, 0.85])
def test_fit_ar1_recovers_coefficient(rho):
    series = [ar1_series(3000, 12, rho, seed=s) for s in range(4)]
    assert np.median(fit_ar1(series)) == pytest.approx(rho, abs=0.03)


def test_forecast_mse_mean_baseline_is_unit_variance():
    series = [ar1_series(2000, 10, 0.6, seed=s) for s in range(3)]
    m = forecast_mse(series, horizon=1)
    assert m["mean"] == pytest.approx(1.0, abs=0.05)
    assert "ar1" not in m  # not reported when no rho is supplied


def test_ar1_beats_persistence_beats_mean_on_ar1_data():
    train = [ar1_series(3000, 12, 0.7, seed=s) for s in range(3)]
    ev = [ar1_series(3000, 12, 0.7, seed=10 + s) for s in range(2)]
    m = forecast_mse(ev, fit_ar1(train), horizon=1)
    assert m["ar1"] < m["persistence"] < m["mean"]
    assert m["ar1"] == pytest.approx(1 - 0.7**2, abs=0.05)


def test_persistence_is_twice_variance_on_white_noise():
    rng = np.random.default_rng(1)
    series = [rng.standard_normal((3000, 10)).astype(np.float32)]
    m = forecast_mse(series, horizon=1)
    assert m["persistence"] == pytest.approx(2.0, abs=0.1)


def test_ar1_fitted_on_train_does_not_peek_at_eval():
    """rho from train must be used verbatim; a refit would score better."""
    train = [ar1_series(2000, 8, 0.8, seed=0)]
    ev = [ar1_series(2000, 8, 0.2, seed=1)]  # deliberately different dynamics
    honest = forecast_mse(ev, fit_ar1(train), horizon=1)["ar1"]
    cheating = forecast_mse(ev, fit_ar1(ev), horizon=1)["ar1"]
    assert honest > cheating


def test_pca_reconstruction_is_exact_at_full_rank():
    rng = np.random.default_rng(2)
    series = [rng.standard_normal((500, 10)).astype(np.float32) for _ in range(3)]
    comps, mean = fit_pca(series, n_components=10)
    assert reconstruction_mse(series, comps, mean) < 1e-8


def test_pca_reconstruction_recovers_low_rank_signal():
    rng = np.random.default_rng(3)
    latents = rng.standard_normal((1500, 4))
    mixing = rng.standard_normal((4, 20))
    series = [(latents @ mixing).astype(np.float32)]
    comps, mean = fit_pca(series, n_components=4)
    assert reconstruction_mse(series, comps, mean) < 1e-6


def test_pca_error_decreases_with_more_components():
    rng = np.random.default_rng(4)
    series = [rng.standard_normal((800, 16)).astype(np.float32) for _ in range(2)]
    errs = [
        reconstruction_mse(series, *fit_pca(series, n_components=k))
        for k in (2, 4, 8, 16)
    ]
    assert all(a > b for a, b in zip(errs, errs[1:]))


def test_evaluate_baselines_reports_per_site():
    train = [ar1_series(600, 8, 0.7, seed=s) for s in range(6)]
    ev = [ar1_series(600, 8, 0.7, seed=100 + s) for s in range(4)]
    res = evaluate_baselines(
        train, ev, eval_sites=["X", "X", "Y", "Y"], pca_components=4
    )
    assert res["n_eval_subjects"] == 4
    assert set(res["per_site"]) == {"X", "Y"}
    assert res["per_site"]["X"]["n_subjects"] == 2
    assert "ar1" in res["forecast"]["h1"]
