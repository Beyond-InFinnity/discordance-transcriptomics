"""Parcel-level mediation with spatial-null inference — CLAUDE.md §9 Phase 6.

The pre-specified model (H2) is::

    vascular/metabolic gene expression  ->  baseline OEF  ->  discordance
              X                                M                  Y

Three things about doing this on brain maps rather than on subjects, each of
which this module handles explicitly:

**Every path needs a spatial null, not just the total effect.** R1 applies to
each coefficient separately. A bootstrap over parcels — which is what the
mediation literature reaches for by default, and what §9 asks for alongside —
treats parcels as exchangeable and so ignores spatial autocorrelation entirely.
It is reported here as a *precision* interval, and the spin p-value is the
inferential test. The two answer different questions and can disagree; when they
do, believe the spin.

**Different paths need different maps rotated.** Rotating the exposure X tests
whether X's spatial pattern relates to M and Y at all, which is the right null
for ``a``, ``c`` and the indirect effect. The two *adjusted* coefficients, ``b``
and ``c'``, need the **outcome** rotated instead.

*Why the outcome is rotated for b.* Both adjusted coefficients are divided by
``1 - r_xm^2``, so both are variance-inflated whenever exposure and mediator
overlap. Rotating the outcome leaves ``r_xm`` untouched, so the observed
coefficient and every null draw carry the same inflation. Rotating the mediator
— the intuitive choice, and what this module did first — destroys it: a
surrogate mediator is uncorrelated with X, so its nulls carry no inflation while
the observed value carries all of it. At ``r_xm = 0.93`` that is a 7.5x mismatch,
and a ``b`` path that is null by construction reports p = 0.002. Using one
rotation set for all four paths has the same failure mode.

**Path coefficients here are rank-based.** Brain maps are rarely bivariate
normal (§11), so variables are rank-transformed and z-scored before the model is
fit. Standardised two-predictor OLS then reduces *exactly* to the three pairwise
correlations::

    a  = r_xm
    c  = r_xy
    b  = (r_my - r_xy * r_xm) / (1 - r_xm^2)
    c' = (r_xy - r_my * r_xm) / (1 - r_xm^2)

with the identity ``c = c' + a*b`` holding to machine precision. That algebra is
not a shortcut with a cost — it is what makes 10,000 rotations of a full path
model take milliseconds instead of hours, and it is why this phase can afford a
proper null on every path rather than on the headline only.

Interpreting a null result: an indirect effect can be real while the total
effect is not significant, so a null ``c`` is not grounds for skipping the
model. But an indirect effect cannot exceed what its weakest link allows — if
``b`` is indistinguishable from zero, ``a*b`` has nowhere to come from, and the
honest report is which link failed rather than that "mediation was not found".
:attr:`MediationResult.limiting_path` names it.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from scipy import stats as sps

logger = logging.getLogger(__name__)

__all__ = ["MediationResult", "mediation", "path_coefficients"]

CorrMethod = Literal["spearman", "pearson"]

# Below this, ``1 - r_xm**2`` is small enough that b and c' are numerically
# unstable: exposure and mediator are nearly the same map and the model cannot
# separate their contributions.
_COLLINEAR_R = 0.99


@dataclass(frozen=True)
class MediationResult:
    """One fitted path model, with a spatial null p-value on every path.

    Attributes
    ----------
    a, b, c, c_prime : float
        Standardised path coefficients. ``a`` is X->M, ``b`` is M->Y adjusting
        for X, ``c`` is the total X->Y, ``c_prime`` the direct X->Y adjusting
        for M.
    a_p, b_p, c_p, c_prime_p : float
        Spatial-null p-values. ``b_p`` and ``c_prime_p`` come from rotating the
        outcome; ``a_p``, ``c_p`` and ``indirect_p`` from rotating the exposure
        (see the module docstring for why they differ).
    indirect : float
        ``a * b``.
    indirect_p : float
        Headline test for the indirect effect: ``max(a_p, b_p)``, i.e. joint
        significance of both links. Conservative by construction — see the
        comment at its computation for why the product's own null is not used.
    indirect_p_product : float
        The product-of-coefficients p-value under exposure rotation. Retained
        for comparison and for the manifest. **Do not report alone** — it is
        dominated by whichever path is stronger and will call mediation on the
        strength of a single real link.
    indirect_ci_lo, indirect_ci_hi : float
        Percentile bootstrap interval over parcels. **A precision interval, not
        a test** — the bootstrap does not preserve spatial autocorrelation.
    proportion_mediated : float
        ``a * b / c``. NaN when ``c`` is near zero, where the ratio is
        meaningless rather than large.
    limiting_path : str
        Which link constrains the indirect effect — ``'a'``, ``'b'``, or
        ``'none'`` when both clear ``alpha``.
    n_valid : int
        Parcels contributing after dropping non-finite values.
    n_perm_exposure, n_perm_outcome, n_boot : int
    method : str
        Correlation metric underlying the path algebra.
    n_covariates : int
        Number of maps X, M and Y were residualised on before fitting.
    """

    a: float
    a_p: float
    b: float
    b_p: float
    c: float
    c_p: float
    c_prime: float
    c_prime_p: float
    indirect: float
    indirect_p: float
    indirect_p_product: float
    indirect_ci_lo: float
    indirect_ci_hi: float
    proportion_mediated: float
    limiting_path: str
    n_valid: int
    n_perm_exposure: int
    n_perm_outcome: int
    n_boot: int
    method: str
    n_covariates: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _standardise(v: np.ndarray) -> np.ndarray:
    """Centre and scale to unit variance, column-wise."""
    v = np.asarray(v, dtype=float)
    out = v - v.mean(axis=0)
    sd = out.std(axis=0)
    return np.divide(out, sd, out=np.zeros_like(out), where=sd > 0)


_std = _standardise


def _std_rows(v: np.ndarray) -> np.ndarray:
    """Centre and scale each *row* to unit variance (one bootstrap draw per row)."""
    out = v - v.mean(axis=1, keepdims=True)
    sd = out.std(axis=1, keepdims=True)
    return np.divide(out, sd, out=np.zeros_like(out), where=sd > 0)


def _ranks(v: np.ndarray) -> np.ndarray:
    """Rank-transform and z-score, so OLS slopes are standardised."""
    r = sps.rankdata(v, axis=0).astype(float)
    r -= r.mean(axis=0)
    sd = r.std(axis=0)
    return np.divide(r, sd, out=np.zeros_like(r), where=sd > 0)


def _rank_cols(a: np.ndarray, method: CorrMethod) -> np.ndarray:
    """Rank each column independently over its finite entries, keeping NaN as NaN.

    Surrogate maps carry missing values wherever a rotation pulled in a parcel
    that was never observed. Ranking has to happen within each column's own
    observed subset, or a draw with two missing parcels is ranked on a different
    scale from a draw with none.
    """
    a = np.asarray(a, dtype=float)
    if method != "spearman":
        return a
    ok = np.isfinite(a)
    # rankdata puts +inf last, so substituting it keeps observed ranks correct;
    # the placeholder ranks are then discarded.
    r = sps.rankdata(np.where(ok, a, np.inf), axis=0).astype(float)
    return np.where(ok, r, np.nan)


def _nan_corr_cols(a: np.ndarray, v: np.ndarray, min_n: int = 10) -> np.ndarray:
    """Per-column correlation of ``a`` against fixed ``v``, pairwise-deleting NaN.

    Both are re-centred within each column's observed subset, so a draw missing
    a few parcels is still compared on its own terms.

    Returns
    -------
    ndarray, shape (n_cols,)
        NaN for columns with fewer than ``min_n`` observed parcels or no
        variance; those draws are dropped from the null when the p-value is
        formed.
    """
    a = np.asarray(a, dtype=float)
    ok = np.isfinite(a)
    A = np.where(ok, a, 0.0)
    B = np.where(ok, v[:, None], 0.0)
    n = ok.sum(axis=0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma, mb = A.sum(axis=0) / n, B.sum(axis=0) / n
        cov = (A * B).sum(axis=0) / n - ma * mb
        va = (A * A).sum(axis=0) / n - ma**2
        vb = (B * B).sum(axis=0) / n - mb**2
        r = cov / np.sqrt(va * vb)
    return np.where((n >= min_n) & (va > 0) & (vb > 0), r, np.nan)


def _residualise_cols(a: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Residualise each column on ``cov`` using only that column's observed rows."""
    a = np.asarray(a, dtype=float)
    out = np.full_like(a, np.nan)
    ok = np.isfinite(a)
    # Columns sharing a missingness pattern share a projection, and with 10,000
    # draws over a handful of missing parcels there are very few distinct
    # patterns — so solve once per pattern rather than once per column.
    patterns, inverse = np.unique(ok, axis=1, return_inverse=True)
    for j in range(patterns.shape[1]):
        rows = patterns[:, j]
        if rows.sum() < cov.shape[1] + 2:
            continue
        cols = np.flatnonzero(inverse == j)
        block = _residualise(a[np.ix_(rows, cols)], cov[rows, :])
        out[np.ix_(rows, cols)] = _std(block)
    return out


def _residualise(v: np.ndarray, cov: np.ndarray | None) -> np.ndarray:
    """Regress ``v`` on ``cov`` (with intercept) and return residuals."""
    if cov is None or cov.size == 0:
        return v
    design = np.column_stack([np.ones(len(cov)), cov])
    beta, *_ = np.linalg.lstsq(design, v, rcond=None)
    return v - design @ beta


def path_coefficients(
    r_xm: np.ndarray | float, r_xy: np.ndarray | float, r_my: float
) -> tuple[Any, Any, Any, Any]:
    """Standardised path coefficients from the three pairwise correlations.

    Vectorised over ``r_xm`` and ``r_xy`` so a whole rotation set is one call.

    Parameters
    ----------
    r_xm, r_xy : array_like or float
        Exposure-mediator and exposure-outcome correlations. Arrays when
        evaluating a null distribution.
    r_my : float
        Mediator-outcome correlation. Fixed under exposure rotation.

    Returns
    -------
    a, b, c, c_prime : ndarray or float
        NaN where exposure and mediator are collinear beyond
        ``_COLLINEAR_R``, since ``b`` and ``c_prime`` are not identified there.
    """
    r_xm = np.asarray(r_xm, dtype=float)
    r_xy = np.asarray(r_xy, dtype=float)
    denom = 1.0 - r_xm**2
    bad = np.abs(r_xm) >= _COLLINEAR_R
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(bad, np.nan, (r_my - r_xy * r_xm) / denom)
        c_prime = np.where(bad, np.nan, (r_xy - r_my * r_xm) / denom)
    return r_xm, b, r_xy, c_prime


def _pairwise(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation of two already-ranked, z-scored vectors."""
    return float(a @ b / len(a))


def _two_tailed_p(observed: float, null: np.ndarray) -> tuple[float, int]:
    """Permutation p-value with the +1 correction (never exactly zero)."""
    finite = null[np.isfinite(null)]
    if finite.size == 0:
        return float("nan"), 0
    n_extreme = int(np.sum(np.abs(finite) >= abs(observed)))
    return (n_extreme + 1) / (finite.size + 1), int(finite.size)


def mediation(
    x: np.ndarray,
    m: np.ndarray,
    y: np.ndarray,
    x_nulls: np.ndarray,
    y_nulls: np.ndarray,
    covariates: np.ndarray | None = None,
    n_boot: int = 10_000,
    seed: int = 42,
    method: CorrMethod = "spearman",
    alpha: float = 0.05,
) -> MediationResult:
    """Fit X -> M -> Y at parcel level with a spatial null on every path.

    Parameters
    ----------
    x, m, y : ndarray, shape (n_parcels,)
        Exposure (gene-set expression score), mediator (e.g. baseline OEF), and
        outcome (e.g. discordance).
    x_nulls : ndarray, shape (n_parcels, n_perm)
        Surrogates of ``x``. Used for ``a``, ``c`` and the indirect effect.
        Required — R1 admits no default.
    y_nulls : ndarray, shape (n_parcels, n_perm)
        Surrogates of ``y``, used for ``b`` and ``c'``. Required for the same
        reason, and it must be the *outcome* rather than the mediator — see
        "Why the outcome is rotated for b" in the module docstring.
    covariates : ndarray, shape (n_parcels, n_cov), optional
        Maps to residualise X, M and Y on before fitting — the Phase 0b dropout
        proxy, and the gradient/myelin controls from Phase 5. Surrogates are
        residualised identically, so the null is like-for-like.
    n_boot : int
        Parcel bootstrap draws for the indirect-effect interval. 0 to skip.
    seed : int
        R7.
    method : {'spearman', 'pearson'}
        ``spearman`` rank-transforms first; ``pearson`` uses values as given.
    alpha : float
        Level used only to decide :attr:`MediationResult.limiting_path`.

    Returns
    -------
    MediationResult

    Raises
    ------
    ValueError
        If either null set is missing, empty, or the wrong shape.
    """
    x = np.asarray(x, dtype=float).ravel()
    m = np.asarray(m, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if x_nulls is None or y_nulls is None:
        raise ValueError(
            "mediation() requires surrogates for both the exposure and the "
            "outcome (CLAUDE.md R1). The outcome set is not optional: without "
            "it the b path has no valid null, and b is the link an indirect "
            "effect depends on most."
        )
    x_nulls = np.asarray(x_nulls, dtype=float)
    y_nulls = np.asarray(y_nulls, dtype=float)
    for name, arr, ref in (("x_nulls", x_nulls, x), ("y_nulls", y_nulls, y)):
        if arr.ndim != 2:
            raise ValueError(f"{name} must be 2D (n_parcels, n_perm), got {arr.shape}")
        if arr.shape[0] != ref.shape[0]:
            raise ValueError(
                f"{name} has {arr.shape[0]} parcels but its map has {ref.shape[0]}"
            )
        if arr.shape[1] == 0:
            raise ValueError(f"{name} is empty; cannot compute a spatial p-value")
    if not (x.shape == m.shape == y.shape):
        raise ValueError(f"x/m/y length mismatch: {x.shape} {m.shape} {y.shape}")

    cov = None if covariates is None else np.asarray(covariates, dtype=float)
    if cov is not None and cov.ndim == 1:
        cov = cov[:, None]

    # One mask across every *observed* map, so all four paths are estimated on
    # exactly the same parcels — otherwise c != c' + a*b and the decomposition
    # stops meaning anything.
    #
    # The surrogates are deliberately not part of this mask. A rotation can move
    # an unobserved parcel into the analysis window, so with a handful of missing
    # parcels almost every one of 10,000 draws contains at least one NaN.
    # Demanding that every draw be complete therefore empties the analysis
    # rather than protecting it. Missing entries are handled per draw by pairwise
    # deletion below, which is what the surrounding code already does for a
    # plain correlation.
    valid = np.isfinite(x) & np.isfinite(m) & np.isfinite(y)
    if cov is not None:
        valid &= np.isfinite(cov).all(axis=1)
    n_valid = int(valid.sum())
    if n_valid < 10:
        raise ValueError(f"only {n_valid} valid parcels; a path model needs more")
    if n_valid < x.shape[0]:
        logger.info("mediation on %d/%d parcels", n_valid, x.shape[0])

    tr = _ranks if method == "spearman" else _standardise
    xv, mv, yv = tr(x[valid]), tr(m[valid]), tr(y[valid])
    xn, yn = _rank_cols(x_nulls[valid, :], method), _rank_cols(y_nulls[valid, :], method)
    cv = None if cov is None else tr(cov[valid, :])

    if cv is not None:
        # Residualise, then re-standardise so coefficients stay comparable.
        xv, mv, yv = (_std(_residualise(v, cv)) for v in (xv, mv, yv))
        xn = _residualise_cols(xn, cv)
        yn = _residualise_cols(yn, cv)

    r_xm, r_xy, r_my = _pairwise(xv, mv), _pairwise(xv, yv), _pairwise(mv, yv)
    a, b, c, c_prime = path_coefficients(r_xm, r_xy, r_my)
    indirect = float(a * b)

    if abs(r_xm) >= _COLLINEAR_R:
        logger.warning(
            "exposure and mediator correlate at %.3f; b and c' are not "
            "identified and are returned as NaN",
            r_xm,
        )

    # --- null distributions -------------------------------------------------
    # Exposure rotations for a, c and the indirect effect. r_my is a property of
    # M and Y and is untouched, which is exactly the null "X's topography is
    # unrelated to the M-Y system".
    n = len(xv)
    r_xm_null = _nan_corr_cols(xn, mv)
    r_xy_null = _nan_corr_cols(xn, yv)
    a_n, b_n, c_n, _ = path_coefficients(r_xm_null, r_xy_null, r_my)
    ind_n = a_n * b_n

    # Outcome rotations for b and c'. Both are adjusted coefficients carrying
    # the variance-inflation factor 1/(1 - r_xm^2), and rotating the *outcome*
    # leaves r_xm untouched — so the observed coefficient and its null are
    # inflated by exactly the same amount.
    #
    # Rotating the mediator instead, which is the intuitive choice, breaks this.
    # A surrogate mediator is uncorrelated with X, so its null coefficients
    # carry no inflation at all while the observed one carries the full factor.
    # With X and M correlated at 0.93 that is a 7.5x mismatch, and a b path that
    # is null by construction comes out at p = 0.002. The test
    # ``test_b_null_is_calibrated_under_collinearity`` is what caught it.
    # Note on the n-mismatch: _nan_corr_cols scores each draw on its own valid
    # parcels while the observed coefficient uses the full valid set, so a draw
    # that lost a parcel is compared against an observed value that kept it.
    # Here that costs at most one parcel in 96, and only in the third of cells
    # using missing=None (the other strategies fill every parcel). The gene
    # screen in src/expression/datadriven.py faces the same issue at a scale
    # where it matters — up to 17 parcels — and recomputes the observed value per
    # draw to remove it entirely. If a target with substantial missingness is
    # ever used as a mediator or outcome here, adopt that approach.
    r_my_yn = _nan_corr_cols(yn, mv)
    r_xy_yn = _nan_corr_cols(yn, xv)
    _, b_from_y, _, cp_from_y = path_coefficients(r_xm, r_xy_yn, 0.0)
    # path_coefficients holds r_my fixed, so feed the varying r_my explicitly.
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = 1.0 - r_xm**2
        if abs(r_xm) >= _COLLINEAR_R:
            b_from_y = np.full(yn.shape[1], np.nan)
            cp_from_y = np.full(yn.shape[1], np.nan)
        else:
            b_from_y = (r_my_yn - r_xy_yn * r_xm) / denom
            cp_from_y = (r_xy_yn - r_my_yn * r_xm) / denom

    a_p, n_px = _two_tailed_p(a, a_n)
    c_p, _ = _two_tailed_p(c, c_n)
    b_p, n_py = _two_tailed_p(b, b_from_y)
    cp_p, _ = _two_tailed_p(c_prime, cp_from_y)
    ind_prod_p, _ = _two_tailed_p(indirect, ind_n)

    # The headline test for the indirect effect is joint significance — both
    # links must clear alpha — rather than the product's own null.
    #
    # The product null is built by rotating the exposure, which makes it a test
    # of "X is unrelated to the M-Y system", and that is *not* the mediation
    # null. It is dominated by whichever link is strong: with a = 0.93 and b
    # pure noise, the product still reports p = 0.002, because rotating X kills
    # a and the product collapses regardless of what b was doing. Reading it
    # alone manufactures mediation out of a single real path.
    #
    # max(a_p, b_p) has no such failure mode: an indirect effect cannot survive
    # a link that does not exist. It is conservative — under a complete null the
    # rejection rate is roughly alpha^2, not alpha — and that is the right
    # direction to err for a pre-specified mechanistic claim.
    ind_p = max(a_p, b_p)

    # --- bootstrap interval on the indirect effect ---------------------------
    # Vectorised across draws. Phase 6 fits thousands of models across the
    # multiverse, and a per-draw Python loop puts the sweep out of reach; done
    # this way the bootstrap is a handful of array operations per model.
    lo = hi = float("nan")
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        s = rng.integers(0, n, size=(n_boot, n))
        # Re-standardise each draw: a resample's ranks are no longer centred,
        # and without this the correlations are wrong by a draw-specific amount.
        bx, bm, by = (_std_rows(v[s]) for v in (xv, mv, yv))
        r_xm_b = np.einsum("ij,ij->i", bx, bm) / n
        r_xy_b = np.einsum("ij,ij->i", bx, by) / n
        r_my_b = np.einsum("ij,ij->i", bm, by) / n
        with np.errstate(invalid="ignore", divide="ignore"):
            den_b = 1.0 - r_xm_b**2
            b_b = np.where(
                np.abs(r_xm_b) >= _COLLINEAR_R,
                np.nan,
                (r_my_b - r_xy_b * r_xm_b) / den_b,
            )
        boot = r_xm_b * b_b
        ok = boot[np.isfinite(boot)]
        if ok.size:
            lo, hi = (float(v) for v in np.percentile(ok, [2.5, 97.5]))

    prop = float(indirect / c) if abs(c) > 0.05 else float("nan")

    if a_p > alpha and b_p > alpha:
        limiting = "a" if a_p >= b_p else "b"
    elif a_p > alpha:
        limiting = "a"
    elif b_p > alpha:
        limiting = "b"
    else:
        limiting = "none"

    return MediationResult(
        a=float(a),
        a_p=float(a_p),
        b=float(b),
        b_p=float(b_p),
        c=float(c),
        c_p=float(c_p),
        c_prime=float(c_prime),
        c_prime_p=float(cp_p),
        indirect=indirect,
        indirect_p=float(ind_p),
        indirect_p_product=float(ind_prod_p),
        indirect_ci_lo=lo,
        indirect_ci_hi=hi,
        proportion_mediated=prop,
        limiting_path=limiting,
        n_valid=n_valid,
        n_perm_exposure=n_px,
        n_perm_outcome=n_py,
        n_boot=int(n_boot),
        method=method,
        n_covariates=0 if cv is None else int(cv.shape[1]),
    )
