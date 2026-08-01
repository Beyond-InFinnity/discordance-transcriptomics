"""Hierarchy control — the Phase 5 confound machinery (CLAUDE.md §9).

⛔ **DECISIVE.** Association cortex differs from sensory cortex on essentially
everything — myelin, gene expression, receptor density, metabolism,
evolutionary expansion. Any map varying along the unimodal→transmodal axis will
correlate with any other map varying along it. §2 names this as the single most
likely way this project produces a false positive.

So the question is never "does the target correlate with X" but "does it
correlate with X **over and above** the principal gradient".

If nothing survives partialling for the gradient, we do not have a molecular
finding, we have a hierarchy finding. That is still publishable, but it must be
reported as such (§9).

Reference maps come from neuromaps and arrive in fsLR; they are transformed to
fsaverage once via ``neuromaps.transforms`` (R4) and parcellated with the same
atlas as the targets.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import stats as sps

logger = logging.getLogger(__name__)

__all__ = [
    "REFERENCE_MAPS",
    "PartialResult",
    "fetch_reference_parcels",
    "partial_corr_with_null",
]

# neuromaps (source, desc) keyed by the name we use downstream.
# The first two are the hierarchy covariates proper; the rest are the §9
# comparison targets, which are informative but not confounds to partial out.
REFERENCE_MAPS: dict[str, tuple[str, str]] = {
    # The confound to beat.
    "margulies_gradient1": ("margulies2016", "fcgradient01"),
    # Registered for the extended (sensitivity) hierarchy specification. Our
    # maps track these MORE than gradient 1: the coupling angle sits at +0.04
    # against gradient 1 but +0.46 and +0.49 against 2 and 3.
    "margulies_gradient2": ("margulies2016", "fcgradient02"),
    "margulies_gradient3": ("margulies2016", "fcgradient03"),
    "t1w_t2w_myelin": ("hcps1200", "myelinmap"),
    # §9 comparison maps.
    "raichle_cmrglc": ("raichle", "cmrglc"),
    "raichle_cmro2": ("raichle", "cmr02"),
    "raichle_cbf": ("raichle", "cbf"),
    "raichle_cbv": ("raichle", "cbv"),
    "evolutionary_expansion": ("hill2010", "evoexp"),
    "abagen_genepc1": ("abagen", "genepc1"),
}

# Covariates entered in the hierarchical step, per config.covariates.hierarchy.
# The pre-registered set. CLAUDE.md §9 says "Margulies principal functional
# gradient + T1w/T2w myelin (+ dropout proxy)" — singular, the first gradient.
# This is the confirmatory specification and does not change.
HIERARCHY_COVARIATES = ("margulies_gradient1", "t1w_t2w_myelin")

# A disclosed SENSITIVITY specification, added 2026-07-31, rationale recorded
# before any expanded result was computed.
#
# The protocol assumed the first connectivity gradient is the hierarchy
# confound. For these particular maps it is not the main one. Measured against
# each of the first five gradients:
#
#                      grad1   grad2   grad3   grad4   grad5
#   baseline OEF       +0.15   +0.34   -0.27   -0.02   -0.05
#   coupling angle     +0.04   +0.46   +0.49   -0.20   -0.03
#   extraction mode    -0.40   -0.31   -0.11   +0.25   -0.21
#
# The coupling angle is essentially orthogonal to gradient 1 (+0.04) — the
# number repeatedly cited as evidence that the hierarchy confound does not apply
# here — while correlating at +0.46 and +0.49 with gradients 2 and 3. Those are
# not noise dimensions: gradient 2 separates visual from somatomotor/auditory
# systems, gradient 3 task-positive from task-negative networks.
#
# Both directions carry risk. Under-controlling risks reporting a hierarchy
# artifact as molecular. Over-controlling risks partialling away real signal,
# since gradients 2 and 3 are themselves partly metabolic and vascular in
# origin. Neither is obviously right, so BOTH are run and BOTH are reported;
# the difference between them is itself the result.
#
# This was found by asking how the hierarchy is operationalised, not by
# disliking an outcome — no expanded result had been computed when the decision
# was made. The pre-registered specification above remains primary.
HIERARCHY_COVARIATES_EXTENDED = (
    "margulies_gradient1",
    "margulies_gradient2",
    "margulies_gradient3",
    "t1w_t2w_myelin",
)


@dataclass(frozen=True)
class PartialResult:
    """A correlation before and after partialling covariates."""

    name: str
    rho_raw: float
    p_spin_raw: float
    rho_partial: float
    p_spin_partial: float
    covariates: list[str]
    n_valid: int
    n_perm: int
    attenuation: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def survives(self) -> bool:
        """Whether the association is still significant after partialling."""
        return self.p_spin_partial < 0.05


def fetch_reference_parcels(
    name: str,
    parcellation: str,
    density: str = "10k",
) -> np.ndarray:
    """Fetch a neuromaps reference map as a left-hemisphere parcel vector.

    Parameters
    ----------
    name : str
        Key in :data:`REFERENCE_MAPS`.
    parcellation : str
        Parcellation name, matching the targets.
    density : str
        fsaverage density.

    Returns
    -------
    ndarray, shape (n_parcels,)
    """
    import nibabel as nib
    from neuromaps import datasets, transforms

    from ..data.parcellate import get_parcellation
    from ..utils.workbench import ensure_workbench

    # neuromaps shells out to wb_command for fsLR transforms; the subprocess
    # inherits this process's PATH, not the interactive shell's.
    ensure_workbench()

    if name not in REFERENCE_MAPS:
        raise KeyError(f"unknown reference map {name!r}; have {sorted(REFERENCE_MAPS)}")
    source, desc = REFERENCE_MAPS[name]

    # fetch_annotation returns a bare list of paths, so the source space has to
    # come from the registry rather than the return value.
    matches = [
        a for a in datasets.available_annotations() if a[0] == source and a[1] == desc
    ]
    if not matches:
        raise KeyError(f"neuromaps has no annotation {source}/{desc}")
    space, src_density = matches[0][2], matches[0][3]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        paths = datasets.fetch_annotation(source=source, desc=desc, verbose=0)

        # Some annotations ship one hemisphere only (AHBA-derived maps in
        # particular, since AHBA is left-dominant). The surface transforms
        # infer density from the file pair and raise if handed a single file
        # without being told which hemisphere it is.
        if isinstance(paths, str):
            paths = [paths]
        hemi = "L" if len(paths) == 1 else None
        if hemi:
            logger.info("%s ships a single hemisphere; assuming L", name)

        if space == "fsLR":
            gii = transforms.fslr_to_fsaverage(
                paths[0] if hemi else paths, target_density=density, hemi=hemi
            )
        elif space == "fsaverage":
            gii = (
                paths
                if src_density == density
                else transforms.fsaverage_to_fsaverage(
                    paths[0] if hemi else paths, target_density=density, hemi=hemi
                )
            )
        elif space == "civet":
            gii = transforms.civet_to_fsaverage(
                paths[0] if hemi else paths, target_density=density, hemi=hemi
            )
        elif space == "MNI152":
            gii = transforms.mni152_to_fsaverage(paths, fsavg_density=density)
        else:
            raise ValueError(f"cannot route {space!r} to fsaverage for {name!r}")

    lh = gii[0] if isinstance(gii, list | tuple) else gii
    data = np.asarray(nib.load(lh).agg_data() if isinstance(lh, str) else lh.agg_data())

    labels, _, n = get_parcellation(parcellation, density, "L")
    if labels.shape != data.shape:
        raise ValueError(f"map has {data.shape[0]} vertices, atlas has {labels.shape[0]}")

    out = np.full(n, np.nan)
    for i in range(1, n + 1):
        sel = (labels == i) & np.isfinite(data) & (data != 0)
        if sel.any():
            out[i - 1] = float(data[sel].mean())
    logger.info("%s @ %s: %d/%d parcels", name, parcellation, np.isfinite(out).sum(), n)
    return out


def _residualise(y: np.ndarray, covars: np.ndarray) -> np.ndarray:
    """Residuals of y after least-squares regression on covars (intercept added)."""
    design = np.column_stack([np.ones(len(y)), covars])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def partial_corr_with_null(
    target: np.ndarray,
    other: np.ndarray,
    covariates: np.ndarray,
    nulls: np.ndarray,
    covariate_names: list[str],
    name: str = "",
    method: str = "spearman",
) -> PartialResult:
    """Correlate two maps before and after partialling covariates, both with
    spatial nulls (R1).

    The null is applied to the *residualised* target, so the partial p-value
    still respects spatial autocorrelation. Residualising and then using a
    parametric p-value would reintroduce exactly the inflation R1 exists to
    prevent.

    Parameters
    ----------
    target : ndarray, shape (n_parcels,)
        The map the surrogates were built from.
    other : ndarray, shape (n_parcels,)
        The map being tested against it.
    covariates : ndarray, shape (n_parcels, n_covars)
        Covariates to partial out of both.
    nulls : ndarray, shape (n_parcels, n_perm)
        Spatial surrogates of ``target``.
    covariate_names : list of str
        Recorded in the result.
    name : str
        Label for the comparison.
    method : {'spearman', 'pearson'}

    Returns
    -------
    PartialResult
    """
    from .spatial import corr_with_null

    target = np.asarray(target, float).ravel()
    other = np.asarray(other, float).ravel()
    covariates = np.atleast_2d(np.asarray(covariates, float))
    if covariates.shape[0] != target.shape[0]:
        covariates = covariates.T

    ok = np.isfinite(target) & np.isfinite(other) & np.isfinite(covariates).all(axis=1)
    if ok.sum() < 10:
        raise ValueError(f"only {ok.sum()} parcels valid across target/other/covariates")

    raw = corr_with_null(
        target[ok], other[ok], nulls=nulls[ok, :], method=method, null_method="spin"
    )

    # Rank-transform before residualising so the partial matches the Spearman
    # convention: partial Spearman is Pearson on residualised ranks.
    if method == "spearman":
        t = sps.rankdata(target[ok])
        o = sps.rankdata(other[ok])
        c = np.column_stack(
            [sps.rankdata(covariates[ok, j]) for j in range(covariates.shape[1])]
        )
    else:
        t, o, c = target[ok], other[ok], covariates[ok, :]

    t_res = _residualise(t, c)
    o_res = _residualise(o, c)

    # Residualise each surrogate the same way, so the null distribution is of
    # partial correlations rather than raw ones.
    nulls_ok = nulls[ok, :]
    null_res = np.full(nulls_ok.shape[1], np.nan)
    for i in range(nulls_ok.shape[1]):
        col = nulls_ok[:, i]
        good = np.isfinite(col)
        if good.sum() < 10:
            continue
        s = sps.rankdata(col[good]) if method == "spearman" else col[good]
        s_res = _residualise(s, c[good, :])
        null_res[i] = sps.pearsonr(s_res, o_res[good]).statistic

    rho_partial = float(sps.pearsonr(t_res, o_res).statistic)
    finite = np.isfinite(null_res)
    n_extreme = int(np.sum(np.abs(null_res[finite]) >= abs(rho_partial)))
    p_partial = (n_extreme + 1) / (int(finite.sum()) + 1)

    return PartialResult(
        name=name,
        rho_raw=raw.rho,
        p_spin_raw=raw.p_spin,
        rho_partial=rho_partial,
        p_spin_partial=float(p_partial),
        covariates=list(covariate_names),
        n_valid=int(ok.sum()),
        n_perm=int(finite.sum()),
        attenuation=float(abs(raw.rho) - abs(rho_partial)),
    )
