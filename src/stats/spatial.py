"""Spatial-autocorrelation-preserving correlation — CLAUDE.md R1.

Brain maps are smooth. Two arbitrary smooth maps correlate at r ~ 0.4 by chance,
so a p-value from a naive ``scipy.stats.pearsonr`` on two brain maps is
meaningless. **This module is the only sanctioned way to correlate two brain
maps in this repo.**

R1 is enforced structurally: :func:`corr_with_null` has no code path that
returns a p-value without a null distribution. Passing ``nulls=None`` raises.

Typical use::

    rot = make_nulls(target, cfg, parcellation=schaefer_lh)   # cached, reusable
    res = corr_with_null(target, gene_map, nulls=rot, cfg=cfg)
    print(res.rho, res.p_spin)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy import stats as sps

logger = logging.getLogger(__name__)

__all__ = ["SpatialCorrResult", "corr_with_null", "fdr_bh", "make_nulls"]

CorrMethod = Literal["spearman", "pearson"]


@dataclass(frozen=True)
class SpatialCorrResult:
    """Result of one spatially-informed correlation.

    Attributes
    ----------
    rho : float
        Observed effect size (Spearman by default — brain maps are rarely
        bivariate normal, CLAUDE.md §11).
    p_spin : float
        Two-tailed p-value against the spatial null distribution.
    p_naive : float
        Parametric p-value. **Recorded for comparison only — never report this
        as evidence.** It is retained so the inflation from spatial
        autocorrelation is visible in the manifest.
    n_perm : int
        Number of null maps actually used.
    null_method : str
        Which null model generated the surrogates.
    n_valid : int
        Number of parcels contributing after NaN removal.
    method : str
        Correlation metric used.
    """

    rho: float
    p_spin: float
    p_naive: float
    n_perm: int
    null_method: str
    n_valid: int
    method: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _corrfunc(method: CorrMethod):
    return sps.spearmanr if method == "spearman" else sps.pearsonr


def _corr(x: np.ndarray, y: np.ndarray, method: CorrMethod) -> float:
    """Correlation coefficient only, NaN-safe on the pair."""
    fn = _corrfunc(method)
    return float(fn(x, y).statistic)


def corr_with_null(
    x: np.ndarray,
    y: np.ndarray,
    nulls: np.ndarray,
    method: CorrMethod = "spearman",
    null_method: str = "unspecified",
) -> SpatialCorrResult:
    """Correlate two parcel-level brain maps against a spatial null.

    The null distribution is built by correlating ``y`` against each surrogate
    of ``x``, which preserves the spatial autocorrelation structure of ``x``.

    Parameters
    ----------
    x : ndarray, shape (n_parcels,)
        First map. Must be the map the surrogates in ``nulls`` were built from.
    y : ndarray, shape (n_parcels,)
        Second map.
    nulls : ndarray, shape (n_parcels, n_perm)
        Surrogate maps of ``x``, from :func:`make_nulls`. Required — there is
        deliberately no default (R1).
    method : {'spearman', 'pearson'}
        Correlation metric. Spearman is the repo default.
    null_method : str
        Name of the null model, recorded in the result for the manifest.

    Returns
    -------
    SpatialCorrResult

    Raises
    ------
    ValueError
        If ``nulls`` is None/empty or shapes are inconsistent. Refusing to
        proceed without surrogates is the point of this function.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if nulls is None:
        raise ValueError(
            "corr_with_null() requires a spatial null distribution (CLAUDE.md R1). "
            "Generate one with make_nulls() — a bare correlation p-value on brain "
            "maps is not interpretable."
        )
    nulls = np.asarray(nulls, dtype=float)
    if nulls.ndim != 2:
        raise ValueError(f"nulls must be 2D (n_parcels, n_perm), got {nulls.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x and y length mismatch: {x.shape[0]} vs {y.shape[0]}")
    if nulls.shape[0] != x.shape[0]:
        raise ValueError(
            f"nulls has {nulls.shape[0]} parcels but x has {x.shape[0]}. "
            "The surrogates must be built from x."
        )
    if nulls.shape[1] == 0:
        raise ValueError("nulls is empty; cannot compute a spatial p-value")

    # Drop parcels missing in either observed map (e.g. zero AHBA coverage).
    # The same mask is applied to every surrogate so the comparison is like-for-like.
    valid = np.isfinite(x) & np.isfinite(y)
    n_valid = int(valid.sum())
    if n_valid < 3:
        raise ValueError(f"only {n_valid} valid parcels; cannot correlate")
    if n_valid < x.shape[0]:
        logger.info(
            "dropping %d/%d parcels with non-finite values",
            x.shape[0] - n_valid,
            x.shape[0],
        )

    xv, yv, nv = x[valid], y[valid], nulls[valid, :]

    rho = _corr(xv, yv, method)
    p_naive = float(_corrfunc(method)(xv, yv).pvalue)

    # Surrogates may carry their own NaNs (e.g. medial wall rotated into a parcel).
    null_rhos = np.full(nv.shape[1], np.nan)
    for i in range(nv.shape[1]):
        col = nv[:, i]
        ok = np.isfinite(col)
        if ok.sum() >= 3:
            null_rhos[i] = _corr(col[ok], yv[ok], method)

    finite = np.isfinite(null_rhos)
    n_perm_used = int(finite.sum())
    if n_perm_used == 0:
        raise ValueError("every surrogate produced a non-finite correlation")
    if n_perm_used < nv.shape[1]:
        logger.warning(
            "%d/%d surrogates dropped (non-finite correlation)",
            nv.shape[1] - n_perm_used,
            nv.shape[1],
        )

    # Two-tailed, +1 correction: a permutation p-value is never exactly zero,
    # and reporting p=0 from a finite permutation set overstates the evidence.
    n_extreme = int(np.sum(np.abs(null_rhos[finite]) >= abs(rho)))
    p_spin = (n_extreme + 1) / (n_perm_used + 1)

    return SpatialCorrResult(
        rho=rho,
        p_spin=float(p_spin),
        p_naive=p_naive,
        n_perm=n_perm_used,
        null_method=null_method,
        n_valid=n_valid,
        method=method,
    )


def make_nulls(
    data: np.ndarray,
    atlas: str = "fsaverage",
    density: str = "10k",
    parcellation: Any = None,
    n_perm: int = 10_000,
    seed: int = 42,
    method: str = "alexander_bloch",
    cache_path: str | Path | None = None,
) -> np.ndarray:
    """Generate (or load) spatial surrogate maps.

    Surrogates are expensive and reusable across every test against the same
    target map (§7.4), so results are cached to ``cache_path`` when given.

    Parameters
    ----------
    data : ndarray
        Parcel-level map to build surrogates from.
    atlas, density : str
        Surface space, passed to neuromaps.
    parcellation : tuple of str or None
        Parcellation files (e.g. LH/RH annot paths) for parcellated data.
    n_perm : int
        Number of surrogates.
    seed : int
        RNG seed (R7).
    method : str
        ``alexander_bloch`` (surface, default) or ``burt2020`` (volumetric).
    cache_path : path, optional
        ``.npy`` file to read from / write to.

    Returns
    -------
    ndarray, shape (n_parcels, n_perm)
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            cached = np.load(cache_path)
            if cached.shape[1] >= n_perm:
                logger.info("loaded cached nulls from %s %s", cache_path, cached.shape)
                return cached[:, :n_perm]
            logger.info(
                "cached nulls have %d < %d perms; regenerating",
                cached.shape[1],
                n_perm,
            )

    from neuromaps import nulls as nm_nulls  # imported lazily; pulls in heavy deps

    try:
        fn = getattr(nm_nulls, method)
    except AttributeError as exc:
        raise ValueError(f"unknown null method {method!r}") from exc

    logger.info("generating %d %s surrogates (seed=%d)", n_perm, method, seed)
    out = fn(
        data,
        atlas=atlas,
        density=density,
        parcellation=parcellation,
        n_perm=n_perm,
        seed=seed,
    )
    out = np.asarray(out)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, out)
        logger.info("cached nulls to %s", cache_path)
    return out


def fdr_bh(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (§11).

    Parameters
    ----------
    pvals : array_like
        Raw p-values across a family of tests.
    alpha : float
        Unused in the adjustment itself; kept so callers document their level.

    Returns
    -------
    ndarray
        BH-adjusted p-values, same order as input.
    """
    p = np.asarray(pvals, dtype=float).ravel()
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    # Scale each p by n / (its 1-based rank in ascending order), then enforce
    # monotonicity by taking a running minimum from the largest p downward.
    scaled = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.clip(adj, 0, 1)
    return out
