#!/usr/bin/env python
"""Read each ABIDE site's repetition time (TR) from the NIfTI headers.

ABIDE's phenotypic table does not carry TR, and the project website links it
only inside per-site PDFs. But TR is recorded in ``pixdim[4]`` of every NIfTI
header, and a NIfTI header is the first 348 bytes of the file — so a single
ranged HTTP request per site (64 KB, enough to cover the first gzip block) is
enough to read it from the authoritative source rather than transcribing it.

    python scripts/fetch_site_tr.py

Writes ``docs/site_tr.json``. Total transfer is ~1.3 MB for all 20 sites,
versus ~2 GB to download the volumes themselves.
"""

from __future__ import annotations

import json
import struct
import sys
import urllib.request
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from foundational_brain.data import fetch_abide  # noqa: E402

S3 = (
    "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs"
    "/cpac/filt_global/func_preproc/{file_id}_func_preproc.nii.gz"
)
HEADER_BYTES = 65536  # one gzip block; the 348-byte header lands well inside


def read_nifti_header(url: str, timeout: int = 90) -> dict:
    """Fetch just enough of a gzipped NIfTI to parse its header."""
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{HEADER_BYTES - 1}"})
    raw = urllib.request.urlopen(req, timeout=timeout).read()
    hdr = zlib.decompressobj(zlib.MAX_WBITS | 16).decompress(raw, 400)

    sizeof_hdr = struct.unpack("<i", hdr[0:4])[0]
    if sizeof_hdr != 348:
        # big-endian file, or not NIfTI-1 at all
        sizeof_hdr = struct.unpack(">i", hdr[0:4])[0]
        if sizeof_hdr != 348:
            raise ValueError(f"not a NIfTI-1 header (sizeof_hdr={sizeof_hdr})")
        endian = ">"
    else:
        endian = "<"

    dim = struct.unpack(f"{endian}8h", hdr[40:56])
    pixdim = struct.unpack(f"{endian}8f", hdr[76:108])
    # xyzt_units packs spatial units in bits 0-2 and temporal in bits 3-5;
    # 8=sec, 16=msec, 24=usec. Many ABIDE headers leave it unset (0).
    time_code = hdr[123] & 0x38
    return {
        "tr_raw": float(pixdim[4]),
        "time_units_code": int(time_code),
        "n_timepoints": int(dim[4]),
        "voxel_size_mm": [round(float(p), 3) for p in pixdim[1:4]],
    }


def normalize_tr(tr_raw: float, time_code: int) -> float:
    """Convert a raw pixdim[4] to seconds.

    When ``xyzt_units`` is unset (common in ABIDE), fall back on magnitude:
    fMRI TRs live in roughly 0.3-5 s, so a value in the hundreds or thousands
    is milliseconds.
    """
    if time_code == 16:  # msec
        return tr_raw / 1000.0
    if time_code == 24:  # usec
        return tr_raw / 1e6
    if time_code == 8:  # sec
        return tr_raw
    return tr_raw / 1000.0 if tr_raw > 20 else tr_raw


def main() -> None:
    bunch = fetch_abide(n_subjects=None, derivative="rois_cc200")
    pheno = bunch.phenotypic

    # one representative subject per site — TR is a scanner protocol constant
    reps = pheno.groupby("SITE_ID")["FILE_ID"].first()

    out: dict[str, dict] = {}
    for site, file_id in reps.items():
        if not isinstance(file_id, str) or file_id == "no_filename":
            print(f"{site:<12} skipped (no file id)")
            continue
        try:
            hdr = read_nifti_header(S3.format(file_id=file_id))
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"{site:<12} FAILED: {type(exc).__name__}: {exc}")
            continue
        tr = normalize_tr(hdr["tr_raw"], hdr["time_units_code"])
        out[str(site)] = {
            "tr_seconds": round(tr, 4),
            "n_timepoints": hdr["n_timepoints"],
            "voxel_size_mm": hdr["voxel_size_mm"],
            "source_subject": file_id,
        }
        print(
            f"{site:<12} TR={tr:>5.3f}s  T={hdr['n_timepoints']:>4}  "
            f"duration={tr * hdr['n_timepoints'] / 60:>5.2f} min  ({file_id})"
        )

    path = REPO_ROOT / "docs" / "site_tr.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nwrote {path} ({len(out)} sites)")


if __name__ == "__main__":
    main()
