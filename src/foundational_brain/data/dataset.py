"""Windowed sequence dataset over parcellated fMRI, with leak-free splits.

Three invariants this module exists to enforce:

1. **Windows never cross a scan boundary.** Each window is indexed into exactly
   one subject's series, so no training sample splices two brains together.
2. **Normalization is fit per subject.** The corpus profile found per-region std
   spanning 0 to 237; z-scoring within subject-and-region removes both the
   scanner gain and the arbitrary per-region scale. There are no statistics
   shared across the split boundary, so this cannot leak.
3. **Splits are by subject, stratified by site.** A subject's windows are highly
   correlated with one another — splitting on windows would put near-duplicates
   on both sides and report a meaningless validation number.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .stats import zscore


def drop_flat_regions(
    series: list[np.ndarray], threshold: float = 1e-6
) -> tuple[list[np.ndarray], np.ndarray]:
    """Remove regions that are constant in *any* subject.

    0.03% of ABIDE regions are exactly constant (dead parcels). A constant
    region z-scores to a divide-by-zero and contributes a degenerate target, so
    the region is dropped corpus-wide rather than per subject — the model needs
    a fixed input width.
    """
    n_regions = series[0].shape[1]
    keep = np.ones(n_regions, dtype=bool)
    for s in series:
        keep &= s.std(axis=0) > threshold
    return [s[:, keep] for s in series], keep


def normalize_subject(x: np.ndarray) -> np.ndarray:
    """Per-region z-scoring within one subject's scan."""
    return zscore(np.asarray(x, dtype=np.float32), axis=0).astype(np.float32)


def split_subjects(
    sites: list[str],
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Split subject indices into train/val/test, stratified by site.

    Stratifying matters even inside a TR-homogeneous group: sites still differ
    in scanner, sequence and population, and NYU alone is a third of the
    corpus, so an unstratified draw can hand validation a site the model never
    trained on and call the resulting gap overfitting.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {fractions}")

    rng = np.random.default_rng(seed)
    sites_arr = np.asarray(sites)
    out: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    for site in np.unique(sites_arr):
        idx = np.where(sites_arr == site)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(fractions[0] * n))
        n_val = int(round(fractions[1] * n))
        # a site with very few subjects should still contribute to training
        n_train = max(n_train, 1) if n else 0
        n_val = min(n_val, max(n - n_train, 0))
        out["train"] += idx[:n_train].tolist()
        out["val"] += idx[n_train : n_train + n_val].tolist()
        out["test"] += idx[n_train + n_val :].tolist()

    return {k: np.array(sorted(v), dtype=int) for k, v in out.items()}


class WindowedFMRIDataset(Dataset):
    """Fixed-length windows over per-subject-normalized ROI series.

    Each item is a ``(seq_len, n_regions)`` float32 tensor plus the subject
    index it came from, so evaluation can aggregate per subject and per site
    rather than per window.
    """

    def __init__(
        self,
        series: list[np.ndarray],
        seq_len: int = 64,
        stride: int | None = None,
        normalize: bool = True,
        subject_ids: list | None = None,
    ) -> None:
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2 to have a next frame")

        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len // 2
        self.series = [normalize_subject(s) if normalize else np.asarray(s, np.float32)
                       for s in series]
        self.subject_ids = (
            list(subject_ids) if subject_ids is not None else list(range(len(series)))
        )
        if len(self.subject_ids) != len(self.series):
            raise ValueError("subject_ids must be one per series")

        # (subject_index, start_offset) for every window; built once so
        # __getitem__ is O(1) and windows never straddle two subjects
        self.index: list[tuple[int, int]] = []
        for i, s in enumerate(self.series):
            T = s.shape[0]
            if T < seq_len:
                continue  # scan too short for even one window
            for start in range(0, T - seq_len + 1, self.stride):
                self.index.append((i, start))

        self.n_regions = self.series[0].shape[1] if self.series else 0

    @property
    def n_subjects_used(self) -> int:
        return len({i for i, _ in self.index})

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int) -> dict:
        i, start = self.index[k]
        window = self.series[i][start : start + self.seq_len]
        return {
            "x": torch.from_numpy(np.ascontiguousarray(window)),
            "subject": self.subject_ids[i],
        }


def build_datasets(
    series: list[np.ndarray],
    sites: list[str],
    seq_len: int = 64,
    stride: int | None = None,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    drop_flat: bool = True,
) -> tuple[dict[str, WindowedFMRIDataset], dict[str, np.ndarray], np.ndarray]:
    """Full path from raw series to train/val/test datasets.

    Returns ``(datasets, split_indices, kept_region_mask)``. Flat-region
    dropping is applied across the whole corpus *before* splitting so every
    split shares one input width; it uses no per-split statistics, so it does
    not leak.
    """
    keep = np.ones(series[0].shape[1], dtype=bool)
    if drop_flat:
        series, keep = drop_flat_regions(series)

    splits = split_subjects(sites, fractions=fractions, seed=seed)
    datasets = {
        name: WindowedFMRIDataset(
            [series[i] for i in idx],
            seq_len=seq_len,
            # no window overlap at eval time: overlapping val windows would
            # weight the middle of each scan more heavily than its edges
            stride=stride if name == "train" else seq_len,
            subject_ids=[int(i) for i in idx],
        )
        for name, idx in splits.items()
    }
    return datasets, splits, keep
