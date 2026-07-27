"""Phase 0a — reliability of the target map (CLAUDE.md §9).

⛔ BLOCKING GATE. If the target map is not reliable, nothing downstream means
anything: an unreliable map cannot correlate with gene expression except by
chance, and a null result would be uninterpretable.

Gate (config ``gates.p0_reliability``):

===========================  ==========================================
median Spearman-Brown r       action
===========================  ==========================================
>= 0.50                       proceed
0.30 - 0.50                   proceed with prominent caveats, drop to DK-68
< 0.30                        **STOP** and report (§13.1)
===========================  ==========================================

Per §9, the number is reported regardless of outcome — nobody else has
computed it, and it belongs in the paper either way.

Everything here operates on a plain ``(n_subjects, n_parcels)`` array, so it is
fully testable without the imaging data.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from scipy import stats as sps

logger = logging.getLogger(__name__)

__all__ = [
    "ReliabilityResult",
    "evaluate_gate",
    "icc21",
    "spearman_brown",
    "split_half_reliability",
]

GateVerdict = Literal["pass", "caveat", "stop"]


def spearman_brown(r: float, factor: float = 2.0) -> float:
    """Spearman-Brown prophecy correction.

    A split-half correlation is computed on half the sample, so it
    underestimates full-sample reliability. This projects it up.

    Parameters
    ----------
    r : float
        Observed split-half correlation.
    factor : float
        Length ratio; 2.0 to go from half-sample to full-sample.

    Returns
    -------
    float
        Corrected reliability. Negative ``r`` is returned unchanged — the
        formula is not meaningful there and inflating a negative correlation
        would be misleading.
    """
    if not np.isfinite(r) or r <= 0:
        return float(r)
    return float(factor * r / (1 + (factor - 1) * r))


@dataclass(frozen=True)
class ReliabilityResult:
    """Outcome of the Phase 0a gate."""

    median_r_raw: float
    median_r_corrected: float
    ci_lo: float
    ci_hi: float
    n_splits: int
    n_subjects: int
    n_parcels: int
    verdict: GateVerdict
    icc_median: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def passed(self) -> bool:
        return self.verdict != "stop"


def split_half_reliability(
    data: np.ndarray,
    n_splits: int = 1000,
    seed: int = 42,
    method: Literal["spearman", "pearson"] = "spearman",
) -> tuple[np.ndarray, np.ndarray]:
    """Repeated random split-half correlation of a parcel-level map.

    On each split, subjects are partitioned into two random halves, the
    parcel-wise mean map is computed in each, and the two maps are correlated
    across parcels.

    Parameters
    ----------
    data : ndarray, shape (n_subjects, n_parcels)
        Per-subject parcel-level target map.
    n_splits : int
        Number of random splits.
    seed : int
        RNG seed (R7).
    method : {'spearman', 'pearson'}
        Correlation metric across parcels.

    Returns
    -------
    raw : ndarray, shape (n_splits,)
        Raw split-half correlations.
    corrected : ndarray, shape (n_splits,)
        Spearman-Brown-corrected correlations.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"expected (n_subjects, n_parcels), got {data.shape}")
    n_sub, n_par = data.shape
    if n_sub < 4:
        raise ValueError(f"need >=4 subjects to split, got {n_sub}")
    if n_par < 3:
        raise ValueError(f"need >=3 parcels to correlate, got {n_par}")

    rng = np.random.default_rng(seed)
    corrfunc = sps.spearmanr if method == "spearman" else sps.pearsonr
    half = n_sub // 2

    raw = np.full(n_splits, np.nan)
    # A parcel with zero coverage is all-NaN in every subject, so nanmean over
    # it is legitimately empty. Expected, not an anomaly — the `ok` mask below
    # drops those parcels.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        for i in range(n_splits):
            perm = rng.permutation(n_sub)
            a = np.nanmean(data[perm[:half]], axis=0)
            b = np.nanmean(data[perm[half : 2 * half]], axis=0)
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() >= 3:
                raw[i] = corrfunc(a[ok], b[ok]).statistic

    finite = np.isfinite(raw)
    if not finite.any():
        raise ValueError("every split produced a non-finite correlation")
    if finite.sum() < n_splits:
        logger.warning("%d/%d splits non-finite", n_splits - finite.sum(), n_splits)

    corrected = np.array([spearman_brown(r) for r in raw])
    return raw, corrected


@dataclass(frozen=True)
class VarianceDecomposition:
    """How much of a map's between-parcel spread is real signal.

    A group map's parcel means differ for two reasons: parcels genuinely differ
    (signal), and each mean carries sampling error from a finite subject sample
    (noise). Splitting them says how much spatial structure is actually there
    and, more usefully, what true effect size an external correlation could
    have detected.

    Attributes
    ----------
    var_observed : float
        Variance across parcel means of the group map — what you see.
    var_error : float
        Sampling variance of a parcel mean, ``within-subject variance / n``.
    var_true : float
        ``var_observed - var_error``. The spatial signal.
    signal_fraction : float
        ``var_true / var_observed``. A variance-based analogue of reliability,
        so it should land near the split-half estimate — a useful cross-check
        of both, since they are computed by different routes.
    attenuation_ceiling : float
        ``sqrt(signal_fraction)``. Measurement noise caps the correlation this
        map can show against even a perfect external map at this value.
    detectable_true_rho : float
        The **true** correlation needed to clear the spin test, after
        attenuation. This is the number that says whether a null result is
        informative or merely underpowered.
    cv : float
        Coefficient of variation of the group map — relative spread, for maps
        on a ratio scale.
    """

    name: str
    var_observed: float
    var_error: float
    var_true: float
    signal_fraction: float
    attenuation_ceiling: float
    detectable_true_rho: float
    cv: float
    n_subjects: int
    n_parcels: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def variance_decomposition(
    data: np.ndarray, name: str = "", spin_threshold: float = 0.245
) -> VarianceDecomposition:
    """Split a map's between-parcel variance into signal and sampling noise.

    Parameters
    ----------
    data : ndarray, shape (n_subjects, n_parcels)
        Per-subject parcel values.
    name : str
        Label carried into the result.
    spin_threshold : float
        The |rho| a spin test requires at this parcel count, measured from the
        null distribution rather than assumed. Used to convert the attenuation
        ceiling into a detectable true effect size.

    Returns
    -------
    VarianceDecomposition
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"expected (n_subjects, n_parcels), got {data.shape}")
    n_sub, n_par = data.shape

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        warnings.filterwarnings("ignore", "Degrees of freedom", RuntimeWarning)
        parcel_means = np.nanmean(data, axis=0)
        # Between-subject variance within each parcel, averaged over parcels.
        within = np.nanmean(np.nanvar(data, axis=0, ddof=1))

    var_obs = float(np.nanvar(parcel_means, ddof=1))
    var_err = float(within / n_sub)
    var_true = max(var_obs - var_err, 0.0)
    frac = var_true / var_obs if var_obs > 0 else 0.0
    ceiling = float(np.sqrt(frac))
    detectable = float(spin_threshold / ceiling) if ceiling > 0 else np.inf

    grand = float(np.nanmean(parcel_means))
    cv = float(np.sqrt(var_obs) / abs(grand)) if grand != 0 else np.nan

    return VarianceDecomposition(
        name=name,
        var_observed=var_obs,
        var_error=var_err,
        var_true=var_true,
        signal_fraction=float(frac),
        attenuation_ceiling=ceiling,
        detectable_true_rho=detectable,
        cv=cv,
        n_subjects=int(n_sub),
        n_parcels=int(n_par),
    )


def icc21(data: np.ndarray) -> np.ndarray:
    """ICC(2,1) per parcel — two-way random effects, single rater, absolute agreement.

    Treats subjects as targets and parcels as the measured units: for each
    parcel we ask how consistent that parcel's value is across subjects
    relative to between-subject variance.

    Parameters
    ----------
    data : ndarray, shape (n_subjects, n_parcels)

    Returns
    -------
    ndarray, shape (n_parcels,)
        ICC(2,1) per parcel. NaN where variance is degenerate.

    Notes
    -----
    Computed with the Shrout & Fleiss (1979) mean-squares formulation:
    ``(MSR - MSE) / (MSR + (k-1)*MSE + k*(MSC - MSE)/n)``.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"expected (n_subjects, n_parcels), got {data.shape}")
    n, k = data.shape
    if n < 2 or k < 2:
        raise ValueError("ICC needs >=2 subjects and >=2 parcels")

    grand = np.nanmean(data)
    row_means = np.nanmean(data, axis=1, keepdims=True)  # per subject
    col_means = np.nanmean(data, axis=0, keepdims=True)  # per parcel

    ss_rows = k * np.nansum((row_means - grand) ** 2)
    ss_cols = n * np.nansum((col_means - grand) ** 2)
    ss_total = np.nansum((data - grand) ** 2)
    ss_err = ss_total - ss_rows - ss_cols

    df_rows, df_cols = n - 1, k - 1
    df_err = df_rows * df_cols
    if df_err <= 0:
        return np.full(k, np.nan)

    msr, msc, mse = ss_rows / df_rows, ss_cols / df_cols, ss_err / df_err
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    icc = np.nan if abs(denom) < 1e-12 else (msr - mse) / denom
    # Shrout-Fleiss ICC(2,1) is a single scalar for the whole matrix; broadcast
    # so callers can align it with a per-parcel vector.
    return np.full(k, icc, dtype=float)


def evaluate_gate(
    corrected: np.ndarray,
    pass_threshold: float = 0.5,
    caveat_threshold: float = 0.3,
) -> GateVerdict:
    """Apply the Phase 0a gate thresholds to the corrected split-half values.

    Returns
    -------
    {'pass', 'caveat', 'stop'}
        ``stop`` means STOP and report — do not proceed with a workaround (R9).
    """
    median = float(np.nanmedian(corrected))
    if median >= pass_threshold:
        return "pass"
    if median >= caveat_threshold:
        return "caveat"
    return "stop"


def run_reliability(
    data: np.ndarray,
    n_splits: int = 1000,
    seed: int = 42,
    pass_threshold: float = 0.5,
    caveat_threshold: float = 0.3,
    method: Literal["spearman", "pearson"] = "spearman",
) -> ReliabilityResult:
    """Full Phase 0a computation and gate evaluation.

    Parameters
    ----------
    data : ndarray, shape (n_subjects, n_parcels)
        Per-subject parcel-level target map.
    n_splits, seed, method
        Passed to :func:`split_half_reliability`.
    pass_threshold, caveat_threshold : float
        Gate thresholds from ``config.gates.p0_reliability``.

    Returns
    -------
    ReliabilityResult
    """
    raw, corrected = split_half_reliability(
        data, n_splits=n_splits, seed=seed, method=method
    )
    lo, hi = np.nanpercentile(corrected, [2.5, 97.5])
    icc = icc21(data)

    result = ReliabilityResult(
        median_r_raw=float(np.nanmedian(raw)),
        median_r_corrected=float(np.nanmedian(corrected)),
        ci_lo=float(lo),
        ci_hi=float(hi),
        n_splits=int(n_splits),
        n_subjects=int(data.shape[0]),
        n_parcels=int(data.shape[1]),
        verdict=evaluate_gate(corrected, pass_threshold, caveat_threshold),
        icc_median=float(np.nanmedian(icc)),
    )
    logger.info(
        "Phase 0a: median SB-corrected r = %.3f [%.3f, %.3f] -> %s",
        result.median_r_corrected,
        result.ci_lo,
        result.ci_hi,
        result.verdict.upper(),
    )
    return result
