"""Project 4D fMRI volumes onto a brain atlas -> ``(T, n_regions)`` matrices.

ABIDE-PCP already ships parcellated series, so this module is not on the
critical path for the first experiments. It exists so the pipeline can ingest
any raw NIfTI dataset (development_fmri, OpenNeuro, HCP) under the same
representation, and so the choice of atlas becomes a config knob rather than a
property of whichever derivative someone happened to download.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .download import data_dir

#: Atlases we can parcellate with. ``kind`` selects the masker: deterministic
#: integer-label atlases use NiftiLabelsMasker, probabilistic map atlases use
#: NiftiMapsMasker.
ATLASES = {
    "schaefer_100": {"kind": "labels", "n_regions": 100},
    "schaefer_200": {"kind": "labels", "n_regions": 200},
    "schaefer_400": {"kind": "labels", "n_regions": 400},
    "aal": {"kind": "labels", "n_regions": 116},
    "harvard_oxford": {"kind": "labels", "n_regions": 48},
    "msdl": {"kind": "maps", "n_regions": 39},
}


def load_atlas(name: str = "schaefer_400", path: str | Path | None = None):
    """Fetch an atlas and return ``(image_or_filename, labels, kind)``."""
    if name not in ATLASES:
        raise ValueError(f"unknown atlas {name!r}; expected one of {sorted(ATLASES)}")

    from nilearn import datasets as nds

    cache = str(data_dir(path))
    kind = ATLASES[name]["kind"]

    if name.startswith("schaefer_"):
        n_rois = int(name.split("_")[1])
        atlas = nds.fetch_atlas_schaefer_2018(n_rois=n_rois, data_dir=cache)
        return atlas.maps, list(atlas.labels), kind
    if name == "aal":
        atlas = nds.fetch_atlas_aal(data_dir=cache)
        return atlas.maps, list(atlas.labels), kind
    if name == "harvard_oxford":
        atlas = nds.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm", data_dir=cache)
        return atlas.maps, list(atlas.labels), kind
    # msdl
    atlas = nds.fetch_atlas_msdl(data_dir=cache)
    return atlas.maps, list(atlas.labels), kind


def make_masker(
    atlas: str = "schaefer_400",
    t_r: float | None = None,
    standardize: str | bool = "zscore_sample",
    detrend: bool = True,
    low_pass: float | None = 0.1,
    high_pass: float | None = 0.01,
    path: str | Path | None = None,
):
    """Build a nilearn masker that turns a 4D image into ``(T, n_regions)``.

    The default band-pass (0.01-0.1 Hz) is the resting-state convention and
    matches ABIDE's ``filt_global`` strand, so series produced here are
    comparable to the primary corpus. Pass ``low_pass=None`` for task/movie
    data, where higher-frequency signal is not noise.
    """
    from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker

    maps, _labels, kind = load_atlas(atlas, path=path)
    common = dict(
        standardize=standardize,
        detrend=detrend,
        low_pass=low_pass,
        high_pass=high_pass,
        t_r=t_r,
        memory=str(data_dir(path) / "nilearn_cache"),
        memory_level=1,
        verbose=0,
    )
    if kind == "labels":
        return NiftiLabelsMasker(labels_img=maps, **common)
    return NiftiMapsMasker(maps_img=maps, **common)


def parcellate(
    func_img,
    atlas: str = "schaefer_400",
    confounds=None,
    t_r: float | None = None,
    **kwargs,
) -> np.ndarray:
    """Parcellate one 4D image into a ``(T, n_regions)`` float32 array."""
    masker = make_masker(atlas=atlas, t_r=t_r, **kwargs)
    return masker.fit_transform(func_img, confounds=confounds).astype(np.float32)


def parcellate_many(
    func_imgs,
    atlas: str = "schaefer_400",
    confounds=None,
    t_r: float | None = None,
    **kwargs,
) -> list[np.ndarray]:
    """Parcellate a list of images, reusing one fitted masker across subjects."""
    masker = make_masker(atlas=atlas, t_r=t_r, **kwargs)
    if confounds is None:
        confounds = [None] * len(func_imgs)
    return [
        masker.fit_transform(img, confounds=conf).astype(np.float32)
        for img, conf in zip(func_imgs, confounds)
    ]
