"""Scan-site metadata, and the TR-homogeneous grouping used for pretraining.

The corpus profile (``docs/data_report.md``) found that lag-1 autocorrelation
ranges 0.53-0.87 across ABIDE sites. Reading TR out of the NIfTI headers
(``scripts/fetch_site_tr.py``) showed why: TR spans 1.5s to 3.0s, so a
"one timestep" window covers twice as much real time at one site as another.

Pooling sites naively therefore asks the latent RNN to model dynamics on an
inconsistent time axis. The first-pass resolution is to pretrain on the largest
TR-homogeneous group (TR = 2.0 s, 10 sites, ~522 subjects, ~60% of the corpus)
and hold the remaining sites out as a cross-TR generalization test — cheap,
and it turns the confound into an evaluation.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SITE_TR_PATH = REPO_ROOT / "docs" / "site_tr.json"

#: TR the pretraining group is defined by, in seconds.
PRETRAIN_TR = 2.0

#: Tolerance when matching a site's TR to a group. Leuven 1/2 differ by 15 ms
#: (1.667 vs 1.652) purely as a header rounding artifact, so exact equality
#: would split what is really one protocol.
TR_TOLERANCE = 0.05


@lru_cache(maxsize=1)
def site_metadata() -> dict[str, dict]:
    """Measured per-site scan parameters, keyed by ABIDE ``SITE_ID``."""
    if not SITE_TR_PATH.exists():
        raise FileNotFoundError(
            f"{SITE_TR_PATH} missing — run scripts/fetch_site_tr.py first"
        )
    return json.loads(SITE_TR_PATH.read_text())


def site_tr(site: str) -> float:
    """TR in seconds for one site."""
    meta = site_metadata()
    if site not in meta:
        raise KeyError(f"no TR recorded for site {site!r}")
    return float(meta[site]["tr_seconds"])


def tr_groups(tolerance: float = TR_TOLERANCE) -> dict[float, list[str]]:
    """Cluster sites into groups of (near-)identical TR.

    Returns ``{representative_tr: [site, ...]}``, ordered by descending group
    size so the first entry is the natural pretraining group.
    """
    meta = site_metadata()
    groups: dict[float, list[str]] = {}
    for site in sorted(meta, key=lambda s: float(meta[s]["tr_seconds"])):
        tr = float(meta[site]["tr_seconds"])
        for key in groups:
            if abs(key - tr) <= tolerance:
                groups[key].append(site)
                break
        else:
            groups[tr] = [site]
    return dict(sorted(groups.items(), key=lambda kv: -len(kv[1])))


def sites_with_tr(tr: float = PRETRAIN_TR, tolerance: float = TR_TOLERANCE) -> list[str]:
    """Sites whose TR matches ``tr`` within ``tolerance``."""
    meta = site_metadata()
    return sorted(
        s for s in meta if abs(float(meta[s]["tr_seconds"]) - tr) <= tolerance
    )


def split_by_tr(
    sites: list[str], tr: float = PRETRAIN_TR, tolerance: float = TR_TOLERANCE
) -> tuple[list[int], list[int]]:
    """Partition subject indices into (in-group, held-out) by their site's TR.

    ``sites`` is the per-subject site label, so the returned indices address
    subjects, not sites. Subjects from a site with no recorded TR are held out
    rather than silently dropped — an unknown time axis is exactly the thing
    this split exists to keep out of pretraining.
    """
    meta = site_metadata()
    in_group, held_out = [], []
    for i, s in enumerate(sites):
        s = str(s)
        known = s in meta
        if known and abs(float(meta[s]["tr_seconds"]) - tr) <= tolerance:
            in_group.append(i)
        else:
            held_out.append(i)
    return in_group, held_out
