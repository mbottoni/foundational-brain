"""Tests for the linear probes.

This machinery decides whether "the RNN encodes phenotype" is true, so the
cases that matter are the null ones: random features must land at chance, and
an imbalanced target must not look decodable just because a probe predicts the
majority class.
"""

from __future__ import annotations

import numpy as np
import pytest

from foundational_brain.eval.features import pool_mean_std
from foundational_brain.eval.probe import (
    classify_probe,
    probe_all,
    regress_probe,
)


def test_random_features_classify_at_chance():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 32))
    y = rng.integers(0, 2, 200)
    res = classify_probe(x, y, seed=0)
    assert res["accuracy_mean"] == pytest.approx(0.5, abs=0.12)
    assert res["auc_mean"] == pytest.approx(0.5, abs=0.12)


def test_separable_classes_are_decoded():
    rng = np.random.default_rng(1)
    y = np.array([0] * 100 + [1] * 100)
    x = rng.standard_normal((200, 16))
    x[y == 1, 0] += 3.0  # a single informative dimension
    res = classify_probe(x, y, seed=1)
    assert res["accuracy_mean"] > 0.9
    assert res["auc_mean"] > 0.95


def test_chance_accuracy_reflects_imbalance():
    rng = np.random.default_rng(2)
    y = np.array([0] * 180 + [1] * 20)  # 90% majority
    x = rng.standard_normal((200, 8))
    res = classify_probe(x, y, seed=2)
    assert res["chance_accuracy"] == pytest.approx(0.9, abs=1e-6)
    # balanced logistic regression on noise should not beat the majority rate
    assert res["accuracy_mean"] < res["chance_accuracy"] + 0.1


def test_imbalanced_noise_auc_stays_near_half():
    """AUC is the honest metric under imbalance; it must not inflate."""
    rng = np.random.default_rng(3)
    y = np.array([0] * 170 + [1] * 30)
    x = rng.standard_normal((200, 8))
    res = classify_probe(x, y, seed=3)
    assert res["auc_mean"] == pytest.approx(0.5, abs=0.15)


def test_random_features_regress_at_zero_r2():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((200, 16))
    y = rng.standard_normal(200)
    res = regress_probe(x, y, seed=4)
    assert res["r2_mean"] < 0.1


def test_linear_target_is_recovered():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((300, 10))
    w = rng.standard_normal(10)
    y = x @ w + rng.standard_normal(300) * 0.1
    res = regress_probe(x, y, seed=5)
    assert res["r2_mean"] > 0.9


def test_probe_all_covers_every_pair():
    rng = np.random.default_rng(6)
    feats = {"a": rng.standard_normal((120, 8)), "b": rng.standard_normal((120, 8))}
    targets = {
        "dx": ("classification", rng.integers(0, 2, 120)),
        "age": ("regression", rng.standard_normal(120)),
    }
    res = probe_all(feats, targets, seed=6)
    assert set(res) == {"dx", "age"}
    assert set(res["dx"]) == {"a", "b"}
    assert res["dx"]["a"]["task"] == "classification"
    assert res["age"]["b"]["task"] == "regression"


def test_pool_mean_std_shape_and_values():
    x = np.stack([np.arange(5.0), np.arange(5.0) * 2], axis=1)  # (5, 2)
    pooled = pool_mean_std(x)
    assert pooled.shape == (4,)
    assert pooled[0] == pytest.approx(2.0)   # mean of 0..4
    assert pooled[1] == pytest.approx(4.0)   # mean of 0..8


def test_constant_feature_does_not_blow_up():
    """A zero-variance feature must be handled by the sd clamp, not NaN out."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal((80, 4))
    x[:, 2] = 5.0  # constant column
    y = rng.integers(0, 2, 80)
    res = classify_probe(x, y, seed=0)
    assert np.isfinite(res["accuracy_mean"])
