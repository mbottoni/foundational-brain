"""Tests for the paired comparison.

This is the machinery that decides whether the project's headline claim is
real, so it needs to say "no" when the difference is noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from foundational_brain.eval.compare import (
    bootstrap_ci,
    compare,
    paired_difference,
    wilcoxon_p,
)


def test_identical_predictors_show_no_difference():
    rng = np.random.default_rng(0)
    m = rng.uniform(0.2, 0.8, 40)
    res = compare(m, m.copy())
    assert res["mean_improvement"] == pytest.approx(0.0)
    assert res["win_rate"] == 0.0
    assert not res["ci_excludes_zero"]
    assert res["wilcoxon_p"] == 1.0


def test_uniformly_better_model_is_detected():
    rng = np.random.default_rng(1)
    base = rng.uniform(0.4, 0.6, 50)
    model = base - 0.1  # better on every subject
    res = compare(model, base)
    assert res["win_rate"] == 1.0
    assert res["mean_improvement"] == pytest.approx(0.1, abs=1e-9)
    assert res["ci_excludes_zero"]
    assert res["wilcoxon_p"] < 0.01


def test_pure_noise_difference_is_not_significant():
    """The check that matters: no false positive on a coin flip."""
    rng = np.random.default_rng(2)
    base = rng.uniform(0.4, 0.6, 60)
    model = base + rng.normal(0, 0.05, 60)  # same on average, noisy
    res = compare(model, base)
    assert not res["ci_excludes_zero"]
    assert res["wilcoxon_p"] > 0.05


def test_win_rate_exposes_a_result_carried_by_outliers():
    """A big mean improvement with a poor win rate must be visible as such."""
    base = np.full(40, 0.5)
    model = np.full(40, 0.52)  # slightly worse on 37 subjects: 37 * -0.02
    model[:3] = 0.01           # hugely better on 3:            3 * +0.49
    res = paired_difference(model, base)
    assert res["mean_improvement"] > 0, "the mean should look like a win"
    assert res["win_rate"] == pytest.approx(3 / 40), "but only 3/40 subjects improved"


def test_bootstrap_resamples_subjects_and_narrows_with_n():
    rng = np.random.default_rng(3)
    small_b = rng.uniform(0.4, 0.6, 10)
    big_b = rng.uniform(0.4, 0.6, 400)
    w_small = bootstrap_ci(small_b - 0.05, small_b, n_boot=2000)
    w_big = bootstrap_ci(big_b - 0.05, big_b, n_boot=2000)
    assert (w_big[1] - w_big[0]) < (w_small[1] - w_small[0])


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        paired_difference(np.zeros(5), np.zeros(6))


def test_relative_improvement_matches_report_convention():
    base = np.full(20, 0.4)
    model = np.full(20, 0.3)
    res = paired_difference(model, base)
    assert res["relative_improvement"] == pytest.approx(0.25)


def test_wilcoxon_direction_is_symmetric():
    rng = np.random.default_rng(4)
    base = rng.uniform(0.4, 0.6, 30)
    model = base - 0.08
    assert wilcoxon_p(model, base) == pytest.approx(wilcoxon_p(base, model))
