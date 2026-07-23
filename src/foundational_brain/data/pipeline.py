"""One canonical path from raw ABIDE to model-ready splits.

Every experiment (pretraining, ablation sweeps, probing) needs the same
sequence: load ABIDE, restrict to the TR-homogeneous group, drop flat regions,
window, split by subject, and produce per-subject-normalized series for the
baselines. Doing that in each script invites drift — and one of the steps,
"mask flat regions *before* z-scoring", is a correctness requirement, not a
convenience: normalizing the full region set scores baselines on regions the
model never receives, some of them constant and therefore z-scoring to zero,
which silently lowers the baseline. This module is the single place that step
lives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import WindowedFMRIDataset, build_datasets, normalize_subject
from .download import load_abide_series
from .sites import split_by_tr


@dataclass
class PretrainData:
    """Everything an experiment needs, with the split boundaries already drawn."""

    datasets: dict[str, WindowedFMRIDataset]      # windowed, for the model
    split_series: dict[str, list[np.ndarray]]     # normalized per-subject, for baselines
    split_sites: dict[str, list[str]]             # site label per subject, per split
    splits: dict[str, np.ndarray]                 # subject indices into the TR group
    keep: np.ndarray                              # boolean region mask from training
    n_regions: int
    held_out_series: list[np.ndarray]             # cross-TR subjects (masked, raw scale)
    held_out_sites: list[str]

    def held_out_normalized(self) -> list[np.ndarray]:
        return [normalize_subject(s) for s in self.held_out_series]


def prepare_pretrain_data(
    n_subjects: int | None = None,
    derivative: str = "rois_cc200",
    seq_len: int = 64,
    stride: int | None = None,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    verbose: bool = True,
) -> PretrainData:
    """Load ABIDE and produce the pretraining-group splits + held-out set.

    The held-out series are masked with the *training* region mask (not a
    freshly computed one) so the model receives the same channels in the same
    order it was trained on.
    """
    series, pheno = load_abide_series(n_subjects=n_subjects, derivative=derivative)
    sites = [str(s) for s in pheno["SITE_ID"]]

    in_group, held_out = split_by_tr(sites)
    if verbose:
        print(
            f"loaded {len(series)} subjects; {len(in_group)} in the TR=2.0s "
            f"pretraining group, {len(held_out)} held out for cross-TR eval"
        )

    group_series = [series[i] for i in in_group]
    group_sites = [sites[i] for i in in_group]

    datasets, splits, keep = build_datasets(
        group_series, group_sites, seq_len=seq_len, stride=stride,
        fractions=fractions, seed=seed,
    )
    n_regions = int(keep.sum())

    # mask THEN normalize — see the module docstring
    norm = [normalize_subject(s[:, keep]) for s in group_series]
    split_series = {k: [norm[i] for i in idx] for k, idx in splits.items()}
    split_sites = {k: [group_sites[i] for i in idx] for k, idx in splits.items()}

    held_out_series = [series[i][:, keep] for i in held_out]
    held_out_sites = [sites[i] for i in held_out]

    if verbose:
        print(
            f"regions kept: {n_regions}; windows "
            + ", ".join(f"{k}={len(v)}" for k, v in datasets.items())
        )

    return PretrainData(
        datasets=datasets,
        split_series=split_series,
        split_sites=split_sites,
        splits=splits,
        keep=keep,
        n_regions=n_regions,
        held_out_series=held_out_series,
        held_out_sites=held_out_sites,
    )
