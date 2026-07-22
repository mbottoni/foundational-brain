"""Fetchers for public fMRI datasets.

Everything here downloads open data with no access application and no PHI, via
``nilearn.datasets``. Two datasets, chosen for complementary reasons:

* **ABIDE-PCP** — resting-state, ~1000 subjects across 17 sites, distributed as
  *already parcellated* ROI time series (T x N_regions). This is exactly the
  representation the foundation model consumes, so it is the primary corpus.
* **development_fmri** — 155 subjects watching a movie, distributed as
  preprocessed 4D NIfTI. Used to exercise the raw-volume -> atlas -> ROI matrix
  path in :mod:`foundational_brain.data.parcellate`.

Downloads are cached under the repo's (gitignored) ``data/`` directory so a
checkout is self-contained rather than leaking into ``~/nilearn_data``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# repo_root/src/foundational_brain/data/download.py -> repo_root
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = REPO_ROOT / "data"

# ABIDE-PCP is served pre-parcellated under several atlases; the number of
# regions is fixed by the derivative name.
ABIDE_DERIVATIVES = {
    "rois_cc200": 200,
    "rois_cc400": 392,  # CC400 ships 392 usable regions, not 400
    "rois_ho": 111,
    "rois_aal": 116,
    "rois_dosenbach160": 161,
}


def data_dir(path: str | Path | None = None) -> Path:
    """Resolve (and create) the download cache directory."""
    d = Path(path) if path is not None else DEFAULT_DATA_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch_abide(
    n_subjects: int | None = 100,
    derivative: str = "rois_cc200",
    pipeline: str = "cpac",
    band_pass_filtering: bool = True,
    global_signal_regression: bool = True,
    path: str | Path | None = None,
):
    """Download ABIDE-PCP ROI time series.

    Returns the raw nilearn ``Bunch``; ``bunch[derivative]`` is a list of
    ``(T, n_regions)`` arrays and ``bunch.phenotypic`` is the subject table.

    The default preprocessing (CPAC, band-pass filtered, global signal
    regressed) is the standard "filt_global" strand — the most aggressively
    denoised variant, which is what we want for a first look at dynamics.
    """
    if derivative not in ABIDE_DERIVATIVES:
        raise ValueError(
            f"unknown derivative {derivative!r}; expected one of "
            f"{sorted(ABIDE_DERIVATIVES)}"
        )

    from nilearn.datasets import fetch_abide_pcp

    return fetch_abide_pcp(
        data_dir=str(data_dir(path)),
        n_subjects=n_subjects,
        pipeline=pipeline,
        band_pass_filtering=band_pass_filtering,
        global_signal_regression=global_signal_regression,
        derivatives=[derivative],
        quality_checked=True,
        verbose=1,
    )


def load_abide_series(
    n_subjects: int | None = 100,
    derivative: str = "rois_cc200",
    path: str | Path | None = None,
    **kwargs,
) -> tuple[list[np.ndarray], "pd.DataFrame"]:
    """ABIDE as a list of ``(T, n_regions)`` float32 arrays + phenotypic table.

    Subjects whose series are empty or whose region count disagrees with the
    atlas are dropped, and the phenotypic table is filtered to match — ABIDE
    has a handful of truncated files.
    """
    bunch = fetch_abide(
        n_subjects=n_subjects, derivative=derivative, path=path, **kwargs
    )
    raw = bunch[derivative]
    expected = ABIDE_DERIVATIVES[derivative]

    series: list[np.ndarray] = []
    keep: list[int] = []
    for i, ts in enumerate(raw):
        arr = np.asarray(ts, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != expected:
            continue
        series.append(arr)
        keep.append(i)

    phenotypic = bunch.phenotypic.iloc[keep].reset_index(drop=True)
    return series, phenotypic


def fetch_development(n_subjects: int = 10, path: str | Path | None = None):
    """Download the development_fmri movie-watching set (4D NIfTI + confounds)."""
    from nilearn.datasets import fetch_development_fmri

    return fetch_development_fmri(
        n_subjects=n_subjects, data_dir=str(data_dir(path)), verbose=1
    )
