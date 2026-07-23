"""Tests for feature pooling.

The connectivity path has two failure modes that would silently corrupt the
probe: NaNs from constant channels, and a PCA reduction that peeks at the probe
set. Both are checked here.
"""

from __future__ import annotations

import numpy as np
import pytest

from foundational_brain.eval.features import (
    _pca_reduce,
    fc_vector,
    pool_mean_std,
)


def test_pool_mean_std_dimension():
    x = np.random.default_rng(0).standard_normal((50, 7))
    assert pool_mean_std(x).shape == (14,)


def test_fc_vector_length_and_symmetry_free():
    x = np.random.default_rng(1).standard_normal((100, 6))
    v = fc_vector(x)
    assert v.shape == (6 * 5 // 2,)  # upper triangle only
    assert np.isfinite(v).all()


def test_fc_vector_handles_constant_channel():
    """A constant region has undefined correlation; it must map to 0, not NaN."""
    x = np.random.default_rng(2).standard_normal((80, 5))
    x[:, 2] = 3.0
    v = fc_vector(x)
    assert np.isfinite(v).all()


def test_fc_vector_recovers_known_correlation():
    rng = np.random.default_rng(3)
    a = rng.standard_normal(2000)
    x = np.stack([a, a * 0.5 + rng.standard_normal(2000) * 0.01], axis=1)
    v = fc_vector(x)  # two near-perfectly correlated channels
    assert v[0] == pytest.approx(1.0, abs=0.02)


def test_pca_reduce_shape_and_train_only_fit():
    rng = np.random.default_rng(4)
    train = rng.standard_normal((60, 200))
    allx = rng.standard_normal((100, 200))
    out = _pca_reduce(train, allx, n_components=20)
    assert out.shape == (100, 20)


def test_pca_reduce_caps_components_at_rank():
    rng = np.random.default_rng(5)
    train = rng.standard_normal((10, 200))  # rank <= 10
    allx = rng.standard_normal((30, 200))
    out = _pca_reduce(train, allx, n_components=100)
    assert out.shape[1] <= 10


def test_pca_reduce_preserves_variance_ordering():
    """The first component should capture the dominant direction."""
    rng = np.random.default_rng(6)
    direction = rng.standard_normal(50)
    scores = rng.standard_normal((200, 1)) * 10
    data = scores @ direction[None, :] + rng.standard_normal((200, 50)) * 0.1
    out = _pca_reduce(data, data, n_components=3)
    assert out[:, 0].std() > out[:, 1].std()
