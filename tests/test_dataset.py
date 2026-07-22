"""Tests for windowing and splitting.

The bugs this guards against are the silent kind: a window spanning two
subjects, or a subject appearing in both train and val, both produce a model
that trains fine and reports a validation number that means nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from foundational_brain.data.dataset import (
    WindowedFMRIDataset,
    build_datasets,
    drop_flat_regions,
    normalize_subject,
    split_subjects,
)


def make_series(n_subjects=12, T=100, n_regions=8, seed=0):
    rng = np.random.default_rng(seed)
    # each subject gets a distinct constant offset so windows are traceable
    return [
        (rng.standard_normal((T, n_regions)) + i * 100).astype(np.float32)
        for i in range(n_subjects)
    ]


def test_normalize_is_per_region_within_subject():
    x = make_series(1, T=200, n_regions=5)[0] * np.array([1, 10, 100, 1000, 5])
    z = normalize_subject(x)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-4)
    assert np.allclose(z.std(axis=0), 1, atol=1e-3)


def test_drop_flat_regions_removes_region_flat_in_any_subject():
    series = make_series(4, T=50, n_regions=6)
    series[2][:, 3] = 7.0  # constant in one subject only
    kept, mask = drop_flat_regions(series)
    assert mask.sum() == 5
    assert not mask[3]
    assert all(s.shape[1] == 5 for s in kept)


def test_windows_never_cross_subject_boundary():
    """Every window must come from exactly one subject's series."""
    series = make_series(6, T=100, n_regions=4)
    ds = WindowedFMRIDataset(series, seq_len=32, stride=16, normalize=False)
    for k in range(len(ds)):
        w = ds[k]["x"].numpy()
        # the constant offset identifies the source subject; a spliced window
        # would show two different offsets within one sample
        offsets = np.round(w.mean(axis=1) / 100).astype(int)
        assert len(np.unique(offsets)) == 1


def test_window_count_matches_stride_arithmetic():
    series = [np.zeros((100, 4), np.float32), np.zeros((64, 4), np.float32)]
    ds = WindowedFMRIDataset(series, seq_len=32, stride=32, normalize=False)
    # subject 1: starts 0,32,64 -> 3 windows;  subject 2: 0,32 -> 2 windows
    assert len(ds) == 5


def test_scans_shorter_than_window_are_skipped_not_padded():
    series = [np.zeros((100, 4), np.float32), np.zeros((10, 4), np.float32)]
    ds = WindowedFMRIDataset(series, seq_len=64, stride=64, normalize=False)
    assert ds.n_subjects_used == 1
    assert all(ds[k]["x"].shape == (64, 4) for k in range(len(ds)))


def test_seq_len_one_rejected():
    with pytest.raises(ValueError):
        WindowedFMRIDataset(make_series(2), seq_len=1)


def test_splits_are_disjoint_and_complete():
    sites = ["A"] * 30 + ["B"] * 20 + ["C"] * 10
    sp = split_subjects(sites, seed=1)
    all_idx = np.concatenate([sp["train"], sp["val"], sp["test"]])
    assert len(np.unique(all_idx)) == len(all_idx) == 60
    assert set(all_idx) == set(range(60))


def test_splits_are_stratified_by_site():
    sites = ["A"] * 50 + ["B"] * 30 + ["C"] * 20
    sp = split_subjects(sites, fractions=(0.6, 0.2, 0.2), seed=3)
    arr = np.asarray(sites)
    for name in ("train", "val", "test"):
        present = set(arr[sp[name]])
        assert present == {"A", "B", "C"}, f"{name} is missing a site"


def test_tiny_site_still_contributes_to_training():
    sites = ["A"] * 40 + ["B"]  # site B has a single subject
    sp = split_subjects(sites, seed=5)
    assert 40 in sp["train"]


def test_split_is_deterministic_given_seed():
    sites = ["A"] * 25 + ["B"] * 25
    a = split_subjects(sites, seed=7)
    b = split_subjects(sites, seed=7)
    c = split_subjects(sites, seed=8)
    assert np.array_equal(a["train"], b["train"])
    assert not np.array_equal(a["train"], c["train"])


def test_build_datasets_has_no_subject_overlap():
    series = make_series(30, T=120, n_regions=6)
    sites = ["A"] * 15 + ["B"] * 15
    ds, splits, _ = build_datasets(series, sites, seq_len=32, seed=2)

    subs = {name: set(d.subject_ids) for name, d in ds.items()}
    assert subs["train"] & subs["val"] == set()
    assert subs["train"] & subs["test"] == set()
    assert subs["val"] & subs["test"] == set()


def test_eval_splits_use_non_overlapping_windows():
    """Overlapping val windows would over-weight the middle of each scan."""
    series = make_series(20, T=128, n_regions=4)
    sites = ["A"] * 20
    ds, _, _ = build_datasets(series, sites, seq_len=32, stride=8, seed=4)
    assert ds["train"].stride == 8
    assert ds["val"].stride == 32
    assert ds["test"].stride == 32


def test_dataset_items_are_normalized():
    series = make_series(8, T=100, n_regions=5)
    ds = WindowedFMRIDataset(series, seq_len=64, normalize=True)
    x = ds[0]["x"].numpy()
    # windows are slices of a subject-normalized series, so they are close to
    # zero-mean but not exactly so - the check is that the 100x offset is gone
    assert abs(x.mean()) < 1.0


def test_subject_ids_are_preserved_through_build():
    series = make_series(20, T=100, n_regions=4)
    sites = ["A"] * 10 + ["B"] * 10
    ds, splits, _ = build_datasets(series, sites, seq_len=32, seed=6)
    for name, d in ds.items():
        assert set(d.subject_ids) <= set(splits[name].tolist())
        for k in range(min(len(d), 5)):
            assert d[k]["subject"] in splits[name].tolist()
