"""Tests for the exploratory statistics.

These statistics decide ``latent_dim``, ``seq_len``, and the baseline the
forecasting loss is judged against, so a silent error here would propagate into
the architecture. Each test drives a function with synthetic data whose answer
is known analytically.
"""

from __future__ import annotations

import numpy as np
import pytest

from foundational_brain.data import stats


def ar1_series(T: int, n_regions: int, rho: float, seed: int = 0) -> np.ndarray:
    """AR(1) process with known coefficient: x_t = rho * x_{t-1} + noise."""
    rng = np.random.default_rng(seed)
    x = np.zeros((T, n_regions), dtype=np.float32)
    noise = rng.standard_normal((T, n_regions)) * np.sqrt(1 - rho**2)
    for t in range(1, T):
        x[t] = rho * x[t - 1] + noise[t]
    return x


def test_zscore_is_unit_variance():
    x = ar1_series(500, 8, 0.7) * 5.0 + 3.0
    z = stats.zscore(x)
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(z.std(axis=0), 1.0, atol=1e-3)


def test_shape_summary_counts_frames():
    series = [np.zeros((100, 10)), np.zeros((50, 10)), np.zeros((150, 10))]
    s = stats.shape_summary(series)
    assert s["n_subjects"] == 3
    assert s["n_regions"] == [10]
    assert s["timepoints_total"] == 300
    assert (s["T_min"], s["T_max"]) == (50, 150)
    assert s["T_median"] == 100.0


def test_amplitude_flags_flat_regions():
    x = ar1_series(200, 4, 0.5)
    x[:, 2] = 0.0  # a dead region, as ABIDE occasionally contains
    amp = stats.amplitude_summary([x])
    assert amp["frac_regions_flat"] == pytest.approx(0.25)


@pytest.mark.parametrize("rho", [0.3, 0.6, 0.9])
def test_ar1_coefficients_recovered(rho):
    series = [ar1_series(4000, 20, rho, seed=s) for s in range(3)]
    est = np.median(stats.ar1_coefficients(series))
    assert est == pytest.approx(rho, abs=0.03)


def test_autocorrelation_decays_geometrically():
    rho = 0.8
    acf = stats.autocorrelation([ar1_series(6000, 20, rho, seed=1)], max_lag=5)
    assert acf[0] == pytest.approx(1.0, abs=1e-3)
    # an AR(1) process has autocorrelation rho^lag
    for lag in range(1, 6):
        assert acf[lag] == pytest.approx(rho**lag, abs=0.05)


def test_white_noise_has_no_autocorrelation():
    rng = np.random.default_rng(0)
    acf = stats.autocorrelation([rng.standard_normal((4000, 20))], max_lag=5)
    assert np.abs(acf[1:]).max() < 0.05


def test_forecast_baselines_ordering():
    """AR(1) must beat persistence, which must beat the mean predictor."""
    series = [ar1_series(3000, 20, 0.6, seed=s) for s in range(3)]
    b = stats.forecast_baselines(series, horizon=1)
    assert b["mean"]["mean"] == pytest.approx(1.0, abs=0.02)
    assert b["ar1"]["mean"] < b["persistence"]["mean"] < b["mean"]["mean"]
    # for a true AR(1), the optimal one-step MSE is 1 - rho^2
    assert b["ar1"]["mean"] == pytest.approx(1 - 0.6**2, abs=0.05)


def test_forecast_baselines_on_white_noise_beat_nothing():
    rng = np.random.default_rng(2)
    series = [rng.standard_normal((3000, 20)).astype(np.float32)]
    b = stats.forecast_baselines(series, horizon=1)
    assert b["persistence"]["mean"] == pytest.approx(2.0, abs=0.1)  # 2x worse
    assert b["ar1"]["mean"] == pytest.approx(1.0, abs=0.05)


def test_pca_spectrum_recovers_low_rank_structure():
    """Frames built from 5 latent factors should need ~5 components."""
    rng = np.random.default_rng(3)
    latents = rng.standard_normal((2000, 5))
    mixing = rng.standard_normal((5, 60))
    frames = (latents @ mixing).astype(np.float32)
    p = stats.pca_spectrum([frames])
    assert p["n_components_99"] <= 6
    assert p["participation_ratio"] < 10


def test_pca_spectrum_on_isotropic_noise_is_full_rank():
    rng = np.random.default_rng(4)
    frames = rng.standard_normal((5000, 40)).astype(np.float32)
    p = stats.pca_spectrum([frames])
    assert p["n_components_90"] > 30
    assert p["participation_ratio"] > 30


def test_connectivity_similarity_high_for_shared_structure():
    """Subjects sharing a mixing matrix have similar connectivity."""
    rng = np.random.default_rng(5)
    mixing = rng.standard_normal((6, 30))
    shared = [
        (rng.standard_normal((400, 6)) @ mixing).astype(np.float32) for _ in range(6)
    ]
    independent = [
        (rng.standard_normal((400, 6)) @ rng.standard_normal((6, 30))).astype(np.float32)
        for _ in range(6)
    ]
    assert stats.connectivity(shared)["between_subject_similarity"] > 0.8
    assert (
        stats.connectivity(independent)["between_subject_similarity"]
        < stats.connectivity(shared)["between_subject_similarity"]
    )


def test_variance_partition_detects_group_structure():
    rng = np.random.default_rng(6)
    mix_a, mix_b = rng.standard_normal((6, 30)), rng.standard_normal((6, 30))
    series, groups = [], []
    for mix, name in ((mix_a, "siteA"), (mix_b, "siteB")):
        for _ in range(5):
            series.append((rng.standard_normal((400, 6)) @ mix).astype(np.float32))
            groups.append(name)

    strong = stats.variance_partition(series, groups)
    assert strong["n_groups"] == 2
    assert strong["between_group_variance_fraction"] > 0.5

    # shuffled labels carry no real structure
    shuffled = ["siteA", "siteB"] * 5
    weak = stats.variance_partition(series, shuffled)
    assert weak["between_group_variance_fraction"] < 0.3
