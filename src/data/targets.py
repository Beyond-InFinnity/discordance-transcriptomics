"""Target map construction (CLAUDE.md §7.3) and the Phase 0 loading layer.

:data:`DERIVATIVE_PATTERNS` was filled in from a real inspection of ds004873
snapshot 2.0.7 (`python -m src.data.targets inspect`), not guessed. One entry —
``discordance_freq`` — remains ``None`` deliberately; see below.

What Phase 1 established, and which is baked into the choices here:

* The design has **four co-equal conditions** (rest / control / mem / calc for
  subjects p019-p055; control / calc only for p058-p068), confirmed by the
  condition branching in their ``B_Fig1.ipynb``. ``control`` is a task in its
  own right, not the control regressor of the calculation task.
* Only ``calc`` and ``control`` are published in MNI152 at group scale, so a
  discordance-frequency map over four conditions is **not constructible from
  the release** even though it is well defined in their design. Hence
  ``discordance_freq: None`` — a Stop-and-Ask under §13.5.
* Group-level baselines should come from the authors' published masked maps
  via :func:`load_authors_group_map`, not from reconstruction. See that
  function's docstring for the evidence.
"""

from __future__ import annotations

import logging
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.config import REPO_ROOT, BaseConfig
from .parcellate import (
    ParcelResult,
    get_parcellation,
    project_to_parcels,
    surface_from_mni152,
)

logger = logging.getLogger(__name__)

__all__ = [
    "TargetMeta",
    "DERIVATIVE_PATTERNS",
    "inspect_derivatives",
    "coupling_ratio_angle",
    "coupling_ratio_signed_log",
    "load_target_map",
    "load_subject_target_matrix",
    "load_authors_group_map",
    "load_coupling_components",
    "discordance_fraction",
    "load_dropout_proxy",
]


# ---------------------------------------------------------------------------
# Dataset layout — FILL IN AFTER RUNNING inspect_derivatives()
# ---------------------------------------------------------------------------
# Map each quantity the protocol needs to a glob relative to the ds004873 root.
# `None` means "not yet identified in the dataset". Any None encountered at load
# time raises with instructions rather than guessing.
DERIVATIVE_PATTERNS: dict[str, str | None] = {
    # Baseline quantities come from the `control` condition, which Phase 1
    # confirmed is a co-equal task condition (one of rest/control/mem/calc),
    # not the control regressor of the calculation task.
    "baseline_oef": "derivatives/sub-*/qmri/sub-*_task-control_space-MNI152_oef.nii.gz",
    "baseline_cbv": "derivatives/sub-*/qmri/sub-*_task-control_space-MNI152_cbv.nii.gz",
    "baseline_cbf": "derivatives/sub-*/qmri/sub-*_task-control_space-MNI152_cbf.nii.gz",
    # Coupling ratio n = %ΔCBF / %ΔCMRO2 for calc relative to control.
    "calc_cbf": "derivatives/sub-*/qmri/sub-*_task-calc_space-MNI152_cbf.nii.gz",
    "calc_cmro2": "derivatives/sub-*/qmri/sub-*_task-calc_space-MNI152_*cmro2.nii.gz",
    "control_cbf": "derivatives/sub-*/qmri/sub-*_task-control_space-MNI152_cbf.nii.gz",
    "control_cmro2": "derivatives/sub-*/qmri/sub-*_task-control_space-MNI152_*cmro2.nii.gz",
    # Dropout proxies (Phase 0b).
    "snr_mask": "derivatives/task-all_space-MNI152_res-2_SNR_YEO_group_mask.nii.gz",
    "t2star": "derivatives/sub-*/qmri/sub-*_task-control_space-MNI152_T2Smap.nii.gz",
    # NOT WIRED — requires the Phase 1 contrast-structure answer.
    "discordance_freq": None,
}

# Loose sanity bounds only. These are NOT a substitute for the authors' mask.
#
# An earlier version capped OEF at 1.0 on the reasoning that OEF > 1 is
# physiologically impossible. That was wrong: Phase 1 established that the
# published maps ALREADY carry the authors' own cap, applied in
# A_preprocessing.ipynb as
#
#     rOEFmax = max(5 * nanmedian(rOEF[brain_mask]), 1.5)
#     rOEF[rOEF > rOEFmax] = rOEFmax          # clipped, NOT excluded
#
# Verified empirically: per-subject map maxima equal 5x that subject's median
# to within interpolation error. So values above 1.0 are legitimate *clipped*
# data covering 8-18% of voxels, and excluding them biased parcel means
# downward in exactly the high-OEF regions the hypothesis concerns.
#
# The bound below (5.0) sits above any observed cap and exists only to catch
# corrupt files, not to do physiological screening.
VALID_RANGES: dict[str, tuple[float, float]] = {
    "oef": (0.0, 5.0),
    "cbf": (0.0, 500.0),  # mL/100g/min; note the /0.75 upscale below
    "cbv": (0.0, 20.0),  # %
    "cmro2": (0.0, 1000.0),  # umol/100g/min
    "t2star": (0.0, 200.0),  # ms
}

# The authors upscale CBF by 25% before any group aggregation
# ("upscale CSF by 25% because of scanner", A_preprocessing / B_Fig1).
# Reproducing their published group map requires it (verified r = 1.000000).
CBF_UPSCALE = 1 / 0.75

_MISSING_MSG = (
    "DERIVATIVE_PATTERNS[{key!r}] is not set.\n\n"
    "The protocol assumes ds004873 contains this quantity, but the mapping from\n"
    "protocol variable -> file pattern has not been established yet. Do NOT guess.\n\n"
    "  1. Fetch ds004873 (see data/MANIFEST.yaml)\n"
    "  2. python -m src.data.targets inspect --root data/raw/ds004873\n"
    "  3. Fill in DERIVATIVE_PATTERNS in src/data/targets.py\n\n"
    "If the quantity turns out NOT to exist in the derivatives, that is a\n"
    "Stop-and-Ask item (CLAUDE.md §13.5) — surface it, do not substitute a proxy."
)


@dataclass
class TargetMeta:
    """Provenance for a loaded map, fed straight into the manifest (R10)."""

    name: str
    parcellation: str
    inputs: list[str] = field(default_factory=list)
    parcellation_files: Any = None
    n_parcels: int | None = None
    coverage: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dataset inspection — runnable the moment ds004873 lands
# ---------------------------------------------------------------------------
def inspect_derivatives(root: str | Path, max_examples: int = 3) -> dict[str, Any]:
    """Survey a BIDS derivatives tree and report what quantities it contains.

    Groups files by their BIDS suffix (the ``_<suffix>.nii.gz`` token) so the
    available quantities are visible at a glance, and reports which
    protocol-required variables appear to be present or missing.

    Parameters
    ----------
    root : path
        Dataset root, e.g. ``data/raw/ds004873``.
    max_examples : int
        Example paths to show per suffix.

    Returns
    -------
    dict
        ``{'suffixes': {suffix: {...}}, 'subjects': [...], 'tasks': [...],
        'protocol_check': {...}}``
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} does not exist. Fetch ds004873 first (see data/MANIFEST.yaml)."
        )

    nii = list(root.rglob("*.nii*"))
    if not nii:
        logger.warning("no NIfTI files under %s", root)

    by_suffix: dict[str, list[Path]] = defaultdict(list)
    subjects: set[str] = set()
    tasks: set[str] = set()

    for p in nii:
        stem = p.name.split(".")[0]
        suffix = stem.rsplit("_", 1)[-1] if "_" in stem else stem
        by_suffix[suffix].append(p)
        if m := re.search(r"sub-([A-Za-z0-9]+)", p.as_posix()):
            subjects.add(m.group(1))
        if m := re.search(r"task-([A-Za-z0-9]+)", p.as_posix()):
            tasks.add(m.group(1))

    suffixes = {
        suf: {
            "count": len(paths),
            "examples": [
                p.relative_to(root).as_posix() for p in sorted(paths)[:max_examples]
            ],
        }
        for suf, paths in sorted(by_suffix.items(), key=lambda kv: -len(kv[1]))
    }

    # Heuristic check against what the protocol needs. Substring match only —
    # this is a hint for a human, never a basis for auto-wiring.
    # (tokens, search_whole_filename). Quantities are matched on the BIDS suffix
    # alone: searching whole names would make `GMR2pCBVmasked_cbf` register as a
    # CBV map. The dropout proxy is the exception — ds004873's SNR map is named
    # task-all_space-MNI152_res-2_SNR_YEO_group_mask, whose suffix is just
    # 'mask', so it is only findable from the full name.
    wanted: dict[str, tuple[tuple[str, ...], bool]] = {
        "baseline_oef": (("oef",), False),
        "baseline_cbv": (("cbv",), False),
        "baseline_cbf": (("cbf", "perfusion"), False),
        "task_cmro2": (("cmro2", "cmro"), False),
        "dropout_proxy": (
            ("snr", "sbref", "bold", "fieldmap", "magnitude", "phasediff"),
            True,
        ),
    }
    names_by_suffix = {
        suf: " ".join(p.name for p in paths).lower() for suf, paths in by_suffix.items()
    }
    protocol_check = {
        key: sorted(
            suf
            for suf in by_suffix
            if any(
                t in suf.lower() or (whole and t in names_by_suffix[suf]) for t in toks
            )
        )
        for key, (toks, whole) in wanted.items()
    }

    report = {
        "root": str(root),
        "n_nifti": len(nii),
        "n_subjects": len(subjects),
        "subjects": sorted(subjects),
        "tasks": sorted(tasks),
        "suffixes": suffixes,
        "protocol_check": protocol_check,
        "top_level": sorted(d.name for d in root.iterdir() if d.is_dir()),
    }

    logger.info(
        "inspected %s: %d NIfTIs, %d subjects, tasks=%s",
        root,
        len(nii),
        len(subjects),
        sorted(tasks) or "none found",
    )
    return report


def _format_report(rep: dict[str, Any]) -> str:
    lines = [
        "=" * 72,
        f"ds004873 INSPECTION — {rep['root']}",
        "=" * 72,
        f"  NIfTI files : {rep['n_nifti']}",
        f"  subjects    : {rep['n_subjects']}",
        f"  tasks       : {', '.join(rep['tasks']) or '(none detected)'}",
        f"  top level   : {', '.join(rep['top_level'])}",
        "",
        "SUFFIXES FOUND (most common first)",
        "-" * 72,
    ]
    for suf, info in rep["suffixes"].items():
        lines.append(
            f"  {suf:<20} {info['count']:>6}   e.g. {info['examples'][0] if info['examples'] else ''}"
        )
    lines += ["", "PROTOCOL REQUIREMENTS (§7.3) — candidate matches", "-" * 72]
    for key, matches in rep["protocol_check"].items():
        status = ", ".join(matches) if matches else "*** NO CANDIDATE FOUND ***"
        lines.append(f"  {key:<20} {status}")
    lines += [
        "",
        "Any '*** NO CANDIDATE FOUND ***' above for baseline OEF or per-task",
        "CBF/CMRO2 is a Stop-and-Ask item (CLAUDE.md §13.5).",
        "=" * 72,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Coupling ratio n — §7.3
# ---------------------------------------------------------------------------
def coupling_ratio_angle(d_cbf: np.ndarray, d_cmro2: np.ndarray) -> np.ndarray:
    """Angle in the (ΔCMRO2, ΔCBF) plane, in radians.

    The primary handling of the coupling ratio. ``n = %ΔCBF / %ΔCMRO2`` blows up
    as the denominator approaches zero; the angle ``atan2(ΔCBF, ΔCMRO2)`` is
    bounded, continuous through the origin, and preserves the sign information
    that distinguishes concordant from discordant responses.

    Parameters
    ----------
    d_cbf, d_cmro2 : ndarray
        Percent-change maps, same shape.

    Returns
    -------
    ndarray
        Angle in ``(-pi, pi]``. Concordant responses sit near the first and
        third quadrants; discordant responses near the second and fourth.
    """
    d_cbf = np.asarray(d_cbf, dtype=float)
    d_cmro2 = np.asarray(d_cmro2, dtype=float)
    if d_cbf.shape != d_cmro2.shape:
        raise ValueError(f"shape mismatch: {d_cbf.shape} vs {d_cmro2.shape}")
    return np.arctan2(d_cbf, d_cmro2)


def coupling_ratio_signed_log(
    d_cbf: np.ndarray, d_cmro2: np.ndarray, eps: float = 1e-3
) -> np.ndarray:
    """Signed log of the coupling ratio — the §7.3 sensitivity check.

    Parameters
    ----------
    d_cbf, d_cmro2 : ndarray
        Percent-change maps.
    eps : float
        Denominator floor guarding the division. Values of ``|ΔCMRO2|`` below
        this are set to NaN rather than clipped, because clipping would invent
        a finite ratio where the data cannot support one.

    Returns
    -------
    ndarray
        ``sign(n) * log1p(|n|)``, NaN where the denominator is unusable.
    """
    d_cbf = np.asarray(d_cbf, dtype=float)
    d_cmro2 = np.asarray(d_cmro2, dtype=float)
    if d_cbf.shape != d_cmro2.shape:
        raise ValueError(f"shape mismatch: {d_cbf.shape} vs {d_cmro2.shape}")

    denom = np.where(np.abs(d_cmro2) < eps, np.nan, d_cmro2)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = d_cbf / denom
    return np.sign(n) * np.log1p(np.abs(n))


# ---------------------------------------------------------------------------
# Loading layer — blocked on the dataset
# ---------------------------------------------------------------------------
DATA_ROOT = REPO_ROOT / "data" / "raw" / "ds004873"

def _require_pattern(key: str) -> str:
    pattern = DERIVATIVE_PATTERNS.get(key)
    if pattern is None:
        raise NotImplementedError(_MISSING_MSG.format(key=key))
    return pattern


def _parc_kwargs(parcellation: str, density: str = "10k") -> dict[str, Any]:
    """Left-hemisphere projection kwargs (R3). Validated by resolving the atlas."""
    get_parcellation(parcellation, density, "L")  # raises on an unknown name
    return {"parcellation": parcellation, "density": density, "hemi": "L"}


def _subject_of(path: Path) -> str:
    m = re.search(r"sub-([A-Za-z0-9]+)", path.name)
    if not m:
        raise ValueError(f"cannot parse subject from {path.name}")
    return m.group(1)


def _find(key: str, root: Path | None = None) -> dict[str, Path]:
    """Resolve a pattern to ``{subject: path}`` (or ``{'group': path}``)."""
    root = root or DATA_ROOT
    pattern = _require_pattern(key)
    hits = sorted(root.glob(pattern))
    if not hits:
        raise FileNotFoundError(
            f"no files match {pattern!r} under {root}. Fetch ds004873 with "
            "src/data/fetch.py (snapshot 2.0.7 — the S3 mirror has no derivatives)."
        )
    if "sub-*" not in pattern:
        return {"group": hits[0]}
    return {_subject_of(p): p for p in hits}


def _quantity_of(key: str) -> str:
    for q in VALID_RANGES:
        if key.endswith(q) or key.startswith(q):
            return q
    return ""


def _snr_mask_surface(density: str = "10k") -> np.ndarray:
    """The authors' SNR mask projected to the surface, as a boolean vertex mask."""
    path = _find("snr_mask")["group"]
    lh, _ = surface_from_mni152(path, density=density, method="nearest")
    return lh > 0.5


def _parcellate_one(
    path: Path,
    key: str,
    parcellation: str,
    masked: bool,
    density: str = "10k",
) -> ParcelResult:
    q = _quantity_of(key)
    return project_to_parcels(
        path,
        mask=_snr_mask_surface(density) if masked else None,
        valid_range=VALID_RANGES.get(q) if masked else None,
        **_parc_kwargs(parcellation, density),
    )


def _stack_subjects(
    key: str, parcellation: str, masked: bool, subjects: list[str] | None = None
) -> tuple[np.ndarray, list[str], list[Path]]:
    """Parcellate one quantity for every subject -> (n_sub, n_parcels)."""
    found = _find(key)
    subs = sorted(found) if subjects is None else [s for s in subjects if s in found]
    rows, paths = [], []
    for s in subs:
        rows.append(_parcellate_one(found[s], key, parcellation, masked).values)
        paths.append(found[s])
    logger.info("parcellated %s for %d subjects (masked=%s)", key, len(subs), masked)
    return np.vstack(rows), subs, paths


def load_coupling_components(
    parcellation: str, masked: bool = True
) -> tuple[np.ndarray, np.ndarray, list[str], list[Path]]:
    """Per-subject percent-change maps for the calc-vs-control contrast.

    Restricted to subjects having all four required maps, so every parcel value
    comes from a single subject rather than a mix.

    Returns
    -------
    d_cbf, d_cmro2 : ndarray, shape (n_subjects, n_parcels)
        Percent change from control to calc.
    subjects : list of str
    paths : list of Path
    """
    keys = ["calc_cbf", "calc_cmro2", "control_cbf", "control_cmro2"]
    common = sorted(set.intersection(*(set(_find(k)) for k in keys)))
    logger.info("coupling contrast: %d subjects have all four maps", len(common))

    mats, paths = {}, []
    for k in keys:
        m, _, p = _stack_subjects(k, parcellation, masked, subjects=common)
        mats[k] = m
        paths += p

    with np.errstate(divide="ignore", invalid="ignore"):
        d_cbf = 100 * (mats["calc_cbf"] - mats["control_cbf"]) / mats["control_cbf"]
        d_cmro2 = (
            100 * (mats["calc_cmro2"] - mats["control_cmro2"]) / mats["control_cmro2"]
        )
    return d_cbf, d_cmro2, common, paths


def discordance_fraction(
    d_cbf: np.ndarray, d_cmro2: np.ndarray, min_pct: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Fraction of subjects showing a discordant BOLD/CMRO₂ response per parcel.

    Discordance in the Epp et al. sense is BOLD and CMRO₂ moving in *opposite*
    directions. BOLD tracks deoxyhaemoglobin, which to first order falls when
    CBF outpaces oxygen consumption, so

        sign(ΔBOLD) ≈ sign(ΔCBF − ΔCMRO₂)

    and a parcel is discordant for a subject when that differs from
    ``sign(ΔCMRO₂)``. Working the two cases through, both reduce to the same
    condition — **discordant ⇔ coupling ratio n < 1** — which is why the n = 1
    isocline (θ = π/4 in the angle representation) is the natural boundary
    rather than a tuned threshold:

    * ΔCMRO₂ > 0: discordant ⇔ ΔCBF < ΔCMRO₂ ⇔ n < 1 (BOLD falls, CMRO₂ rises)
    * ΔCMRO₂ < 0: discordant ⇔ ΔCBF > ΔCMRO₂ ⇔ n < 1 (BOLD rises, CMRO₂ falls)

    .. note::
       An earlier version counted only **opposite-sign** ΔCBF/ΔCMRO₂ as
       discordant, on the grounds that this needs no model at all. Measured on
       ds004873 that criterion is not merely conservative, it is close to
       vacuous: 76 of 100 parcels have both quantities rising, so the dominant
       discordance mode in this data — CMRO₂ rising faster than CBF — was
       invisible to it, and the resulting column carried essentially no signal
       (Spearman −0.04 against the coupling angle). The first-order BOLD sign
       assumption above is standard and buys the actual phenomenon.

    Parameters
    ----------
    d_cbf, d_cmro2 : ndarray, shape (n_subjects, n_parcels)
        Percent-change maps.
    min_pct : float
        Ignore responses smaller than this magnitude in ΔCMRO₂, where the sign
        is noise rather than signal. 0 disables.

    Returns
    -------
    fraction : ndarray, shape (n_parcels,)
        Discordant subjects / subjects with usable data.
    n_used : ndarray, shape (n_parcels,)
        Denominator per parcel.
    """
    d_cbf = np.asarray(d_cbf, dtype=float)
    d_cmro2 = np.asarray(d_cmro2, dtype=float)
    if d_cbf.shape != d_cmro2.shape:
        raise ValueError(f"shape mismatch: {d_cbf.shape} vs {d_cmro2.shape}")

    usable = np.isfinite(d_cbf) & np.isfinite(d_cmro2)
    if min_pct > 0:
        usable &= np.abs(d_cmro2) >= min_pct

    bold_sign = np.sign(d_cbf - d_cmro2)
    discordant = (bold_sign != np.sign(d_cmro2)) & usable

    n_used = usable.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(n_used > 0, discordant.sum(axis=0) / n_used, np.nan)
    return fraction, n_used


def load_subject_target_matrix(
    cfg: BaseConfig,
    target: str,
    parcellation: str,
    masked: bool = True,
) -> tuple[np.ndarray, TargetMeta]:
    """Load the ``(n_subjects, n_parcels)`` matrix needed by Phase 0a.

    Parameters
    ----------
    cfg : BaseConfig
        Loaded configuration.
    target : {'baseline_oef', 'coupling_n'}
        Which target map. ``discordance_freq`` is deliberately unavailable —
        it needs the Phase 1 contrast-structure answer.
    parcellation : str
        Parcellation name from the config.
    masked : bool
        Apply the authors' SNR mask and physiological range limits.

    Returns
    -------
    data : ndarray, shape (n_subjects, n_parcels)
    meta : TargetMeta
    """
    if target == "baseline_oef":
        data, subs, paths = _stack_subjects("baseline_oef", parcellation, masked)

    elif target == "coupling_n":
        d_cbf, d_cmro2, common, paths = load_coupling_components(
            parcellation, masked=masked
        )
        transform = (
            coupling_ratio_angle
            if cfg.targets.coupling_n_transform == "angle"
            else coupling_ratio_signed_log
        )
        data, subs = transform(d_cbf, d_cmro2), common

    else:
        raise NotImplementedError(
            f"target {target!r} is not wired. Available: 'baseline_oef', "
            "'coupling_n'. 'discordance_freq' requires the Phase 1 "
            "contrast-structure answer — see CLAUDE.md §13.5."
        )

    meta = TargetMeta(
        name=target,
        parcellation=parcellation,
        inputs=[str(p) for p in paths],
        n_parcels=data.shape[1],
        coverage={"n_subjects": len(subs), "subjects": subs, "masked": masked},
    )
    return data, meta


def load_target_map(
    cfg: BaseConfig,
    target: str,
    parcellation: str,
    masked: bool = True,
) -> tuple[np.ndarray, TargetMeta]:
    """Group-level parcel vector for one target map.

    Aggregates across subjects with the **median**, matching the authors
    (``np.nanmedian(par_map, axis=3)`` in B_Fig1.ipynb). An earlier version
    used the mean; the median is both their choice and the more robust one
    given the clipped upper tail in the OEF maps.
    """
    data, meta = load_subject_target_matrix(cfg, target, parcellation, masked=masked)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
        group = np.nanmedian(data, axis=0)
    meta.coverage["aggregation"] = "median"
    return group, meta


def load_authors_group_map(
    parcellation: str,
    quantity: str = "oef",
    density: str = "10k",
) -> tuple[np.ndarray, TargetMeta]:
    """Parcellate the authors' own published group map.

    This is the **authoritative** source for group-level baseline quantities.
    Phase 1 established that we cannot rebuild their per-subject masking — the
    ``_qBmasked`` files and per-subject GM/R2'/CBV masks are released for a
    single subject only — and that rebuilding from the unmasked per-subject
    maps yields a materially different topography (Spearman 0.66 against their
    masked map, on their own voxels).

    So for anything group-level we use their map rather than our reconstruction.

    Parameters
    ----------
    parcellation : str
        Parcellation name from the config.
    quantity : {'oef', 'cbf', 'cmro2'}
        Which published group map to use. Control condition, n=40.

    Returns
    -------
    values : ndarray
    meta : TargetMeta
    """
    fname = (
        f"N40_cond-control_space-MNI152_median_GMR2pCBVmasked_{quantity}.nii.gz"
    )
    path = DATA_ROOT / "derivatives" / fname
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; fetch ds004873 derivatives")

    # NEAREST, not linear. Their masked map is a thin grey-matter ribbon
    # (46,594 voxels, 19% of the unmasked extent). Trilinear interpolation
    # blends each in-mask voxel with out-of-mask zeros, which biases every
    # parcel mean downward: measured on the group OEF map, linear gives a
    # parcel median of 0.248 against a within-mask volumetric median of 0.394,
    # whereas nearest-neighbour gives 0.379. The per-subject maps are dense and
    # unmasked, so linear remains correct for those.
    res = project_to_parcels(
        path,
        valid_range=VALID_RANGES.get(quantity),
        method="nearest",
        **_parc_kwargs(parcellation, density),
    )
    meta = TargetMeta(
        name=f"authors_group_{quantity}",
        parcellation=parcellation,
        inputs=[str(path)],
        n_parcels=res.values.shape[0],
        coverage={
            "source": "authors published group map (GMR2pCBVmasked)",
            "n_subjects": 40,
            "aggregation": "median",
            "n_empty_parcels": res.n_empty,
            "frac_vertices_kept": res.frac_vertices_kept,
        },
    )
    return res.values, meta


def load_dropout_proxy(
    cfg: BaseConfig,
    proxy: str,
    parcellation: str,
    masked: bool = False,
) -> tuple[np.ndarray, TargetMeta]:
    """Load a Phase 0b dropout proxy as a parcel vector.

    Parameters
    ----------
    proxy : {'snr_coverage', 't2star'}
        ``snr_coverage`` is the fraction of each parcel's cortical vertices
        surviving the authors' own SNR criterion — a continuous severity
        measure derived from a binary mask. ``t2star`` is the group-mean T2*,
        the physical quantity that macroscopic B0 inhomogeneity destroys and
        that mqBOLD's OEF estimate depends on.
    masked : bool
        Whether to apply the SNR mask to the proxy itself. Meaningless for
        ``snr_coverage`` (it *is* the mask) and forced to False there.

    Returns
    -------
    values : ndarray, shape (n_parcels,)
    meta : TargetMeta
    """
    kw = _parc_kwargs(parcellation)

    if proxy == "snr_coverage":
        path = _find("snr_mask")["group"]
        res = project_to_parcels(path, mask=None, drop_zero=False, method="nearest", **kw)
        # project_to_parcels averages a 0/1 volume -> the mean IS the retained
        # fraction. Parcels wholly outside the mask read 0, not NaN.
        values = np.nan_to_num(res.values, nan=0.0)
        paths, cov = [path], {"n_parcels_zero_coverage": int((values == 0).sum())}

    elif proxy == "t2star":
        data, subs, paths = _stack_subjects("t2star", parcellation, masked)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
            values = np.nanmean(data, axis=0)
        cov = {"n_subjects": len(subs), "masked": masked}

    else:
        raise ValueError(
            f"unknown dropout proxy {proxy!r}; use 'snr_coverage' or 't2star'"
        )

    meta = TargetMeta(
        name=f"dropout_{proxy}",
        parcellation=parcellation,
        inputs=[str(p) for p in paths],
        n_parcels=values.shape[0],
        coverage=cov,
    )
    return values, meta


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Inspect ds004873 derivatives.")
    ap.add_argument("command", choices=["inspect"])
    ap.add_argument("--root", default="data/raw/ds004873")
    ap.add_argument("--json", action="store_true", help="emit raw JSON")
    a = ap.parse_args()

    logging.basicConfig(level="INFO")
    rep = inspect_derivatives(a.root)
    print(json.dumps(rep, indent=2) if a.json else _format_report(rep))
