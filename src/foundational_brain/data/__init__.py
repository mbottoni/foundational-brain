"""data subpackage: fetching public fMRI, parcellation, and exploratory stats.

Note that :mod:`~foundational_brain.data.stats` is deliberately not re-exported
here — importing it should not be an implicit dependency of loading data.
"""

from .download import (
    ABIDE_DERIVATIVES,
    fetch_abide,
    fetch_development,
    load_abide_series,
)
from .parcellate import ATLASES, load_atlas, make_masker, parcellate, parcellate_many

__all__ = [
    "ABIDE_DERIVATIVES",
    "ATLASES",
    "fetch_abide",
    "fetch_development",
    "load_abide_series",
    "load_atlas",
    "make_masker",
    "parcellate",
    "parcellate_many",
]
