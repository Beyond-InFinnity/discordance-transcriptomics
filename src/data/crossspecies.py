"""Macaque → human cortical surface transfer.

Motivation
----------
No human capillary-density atlas exists. The closest available measurement is
ferumoxytol-weighted laminar MRI in macaque (Autio et al., eLife 2025), which
resolves the microvasculature and reports primary sensory cortex carrying 2-3x
the vascular volume of association cortex. Human blood-volume maps — ours and
independent PET — do **not** reproduce that gradient, almost certainly because
they measure total blood volume dominated by large vessels rather than the
capillary bed. So a macaque map, transferred to human cortex, is currently the
only route to testing the capillary explanation directly.

How the transfer works
----------------------
Macaque Yerkes19 and human fs_LR share a mesh **topology** (32,492 or 10,242
vertices) but not an anatomy: fitting the best uniform scale between the two
midthickness surfaces leaves a 51% residual. Reading macaque vertex *i* as
human vertex *i* would therefore produce a plausible-looking but meaningless
map. A registration is required.

Xu et al. (2020) published one, built with Multimodal Surface Matching on
myelin maps and anatomical landmarks, as ``sphere.reg`` surfaces at 10k and 32k
fs_LR. Those are public on GitHub. Connectome Workbench applies them.

⚠️ Accuracy is strongly non-uniform — measure it, do not assume it
--------------------------------------------------------------------
Warping the published macaque landmark set into human space and comparing
against the published human landmarks gives a **median centroid offset of
9.4 mm**, but the spread is what matters:

===============================  ========  ===========
landmark                          Dice      offset
===============================  ========  ===========
V1                                0.80      2.4 mm
somatosensory area 3              0.73      8.3 mm
motor area 4                      0.56      6.7 mm
LIP                               0.38      3.5 mm
---------------------------------------------------
MT                                0.00      16.6 mm
area 46 (dlPFC)                   0.02      22.2 mm
FEF / 8Av                         0.00      26.6 mm
PFm / PF (inf. parietal)          0.00      34.0 mm
===============================  ========  ===========

The registration is accurate in primary sensory and motor cortex and poor in
prefrontal and inferior parietal association cortex — errors there exceed the
width of a Schaefer-200 parcel. **This is the region the discordance hypothesis
is about**, so any macaque-derived result must be reported with per-region
confidence rather than as a single uniform map.

The round-trip check is separate and passes cleanly: macaque → human → macaque
recovers the input at r = 0.998, so the resampling itself is near-lossless. The
error above is anatomical correspondence, not interpolation.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np

from ..utils.config import REPO_ROOT
from ..utils.workbench import ensure_workbench

logger = logging.getLogger(__name__)

__all__ = [
    "ALIGNMENT_DIR",
    "REQUIRED_FILES",
    "landmark_accuracy",
    "macaque_to_human",
    "roundtrip_error",
]

ALIGNMENT_DIR = REPO_ROOT / "data" / "external" / "macaque_human_alignment"

# Public, from github.com/TingsterX/alignment_macaque-human (Xu et al. 2020).
REQUIRED_FILES = (
    "L.macaque-to-human.sphere.reg.10k_fs_LR.surf.gii",
    "L.human-to-macaque.sphere.reg.10k_fs_LR.surf.gii",
    "MacaqueYerkes19.L.sphere.10k_fs_LR.surf.gii",
    "MacaqueYerkes19.L.midthickness.10k_fs_LR.surf.gii",
    "S1200.L.sphere.10k_fs_LR.surf.gii",
    "S1200.L.midthickness_MSMAll.10k_fs_LR.surf.gii",
)


def _check_files() -> None:
    missing = [f for f in REQUIRED_FILES if not (ALIGNMENT_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing {len(missing)} alignment file(s) in {ALIGNMENT_DIR}:\n  "
            + "\n  ".join(missing)
            + "\n\nFetch from https://github.com/TingsterX/alignment_macaque-human"
        )


def macaque_to_human(
    metric: str | Path,
    out: str | Path,
    method: str = "ADAP_BARY_AREA",
    is_label: bool = False,
) -> Path:
    """Resample a macaque Yerkes19 10k fs_LR map onto the human 10k fs_LR mesh.

    Parameters
    ----------
    metric : path
        Macaque-space ``.func.gii`` / ``.shape.gii`` (or ``.label.gii`` with
        ``is_label``).
    out : path
        Destination.
    method : str
        ``ADAP_BARY_AREA`` preserves areal quantities; ``BARYCENTRIC`` is plain
        interpolation. They differ negligibly here (r = 0.9995).
    is_label : bool
        Use label resampling, which never invents intermediate values.

    Returns
    -------
    Path
    """
    _check_files()
    ensure_workbench()
    d, out = ALIGNMENT_DIR, Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "wb_command",
        "-label-resample" if is_label else "-metric-resample",
        str(metric),
        str(d / "L.macaque-to-human.sphere.reg.10k_fs_LR.surf.gii"),
        str(d / "S1200.L.sphere.10k_fs_LR.surf.gii"),
        method,
        str(out),
        "-area-surfs",
        str(d / "MacaqueYerkes19.L.midthickness.10k_fs_LR.surf.gii"),
        str(d / "S1200.L.midthickness_MSMAll.10k_fs_LR.surf.gii"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"wb_command failed:\n{res.stderr}")
    logger.info("macaque -> human: %s", out.name)
    return out


def human_to_macaque(
    metric: str | Path, out: str | Path, method: str = "ADAP_BARY_AREA"
) -> Path:
    """The reverse direction, used for the round-trip check."""
    _check_files()
    ensure_workbench()
    d, out = ALIGNMENT_DIR, Path(out)
    cmd = [
        "wb_command", "-metric-resample", str(metric),
        str(d / "L.human-to-macaque.sphere.reg.10k_fs_LR.surf.gii"),
        str(d / "MacaqueYerkes19.L.sphere.10k_fs_LR.surf.gii"),
        method, str(out), "-area-surfs",
        str(d / "S1200.L.midthickness_MSMAll.10k_fs_LR.surf.gii"),
        str(d / "MacaqueYerkes19.L.midthickness.10k_fs_LR.surf.gii"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"wb_command failed:\n{res.stderr}")
    return out


def roundtrip_error(metric: str | Path, tmpdir: str | Path) -> float:
    """Correlation between a macaque map and itself after a round trip.

    Tests the interpolation, not the anatomy. Values near 1 mean the resampling
    is lossless; they say nothing about whether the anatomical correspondence
    is correct — for that see :func:`landmark_accuracy`.
    """
    import nibabel as nib
    from scipy.stats import pearsonr

    tmp = Path(tmpdir)
    a = macaque_to_human(metric, tmp / "_rt_fwd.func.gii")
    b = human_to_macaque(a, tmp / "_rt_back.func.gii")
    x = np.asarray(nib.load(str(metric)).agg_data()).ravel()
    y = np.asarray(nib.load(str(b)).agg_data()).ravel()
    ok = np.isfinite(x) & np.isfinite(y) & (x != 0) & (y != 0)
    return float(pearsonr(x[ok], y[ok]).statistic)


def landmark_accuracy(tmpdir: str | Path) -> "list[dict]":
    """Per-landmark registration accuracy, in human space.

    Warps the published macaque landmark set into human space and compares it
    against the published human landmarks. Reports Dice overlap and the
    distance between region centroids in millimetres.

    Dice is harsh on these landmarks — several span only tens of vertices, so a
    small shift zeroes it. The centroid offset is the more interpretable number.

    Returns
    -------
    list of dict
        One entry per landmark with ``name``, ``dice``, ``offset_mm``.
    """
    import nibabel as nib

    tmp = Path(tmpdir)
    mac = ALIGNMENT_DIR / "Macaque.L_LANDMARK_ROI.10k_fs_LR.label.gii"
    hum = ALIGNMENT_DIR / "Human.L_LANDMARK_ROI.10k_fs_LR.label.gii"
    if not mac.exists() or not hum.exists():
        raise FileNotFoundError(f"landmark labels not found in {ALIGNMENT_DIR}")

    warped = macaque_to_human(mac, tmp / "_lm.label.gii", is_label=True)
    w = np.asarray(nib.load(str(warped)).agg_data()).astype(int)
    hg = nib.load(str(hum))
    h = np.asarray(hg.agg_data()).astype(int)
    names = hg.labeltable.get_labels_as_dict()
    coords = nib.load(
        str(ALIGNMENT_DIR / "S1200.L.midthickness_MSMAll.10k_fs_LR.surf.gii")
    ).agg_data()[0]

    out = []
    for lab in sorted(set(np.unique(h)) - {0}):
        a, b = (w == lab), (h == lab)
        if not a.any() or not b.any():
            continue
        out.append(
            {
                "name": str(names.get(lab, lab)),
                "dice": float(2 * (a & b).sum() / (a.sum() + b.sum())),
                "offset_mm": float(
                    np.linalg.norm(coords[a].mean(0) - coords[b].mean(0))
                ),
            }
        )
    return out
