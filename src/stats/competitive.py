"""Competitive gene-set nulls — matched on size *and* differential stability.

Why a spatial null is not enough
--------------------------------
The spin test asks "could this correlation arise from two smooth maps by
chance?" It does not ask "could *any* gene set of this size do as well?" Those
are different questions, and the second one bites hard in expression data.

Two properties inflate a gene set's apparent association independently of its
biology:

**Size.** A 200-gene set averages away noise that a 6-gene set carries, so its
score is smoother and correlates more reliably with any smooth map.

**Differential stability.** Genes whose regional pattern replicates across
donors carry real anatomical structure; genes that do not are close to noise. A
set rich in stable genes will out-correlate a set of unstable ones regardless of
what either does biologically.

So the null must draw random sets matched on both. A set only counts as
associated if it beats gene sets *like it*, not gene sets in general.

Differential stability here is the mean pairwise Spearman correlation of a
gene's regional profile across donors — the standard definition (Hawrylycz et
al. 2015), and the same quantity abagen filters on.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

logger = logging.getLogger(__name__)

__all__ = [
    "CompetitiveResult",
    "competitive_null",
    "differential_stability",
    "matched_random_sets",
]


@dataclass(frozen=True)
class CompetitiveResult:
    """A gene set tested against matched random sets."""

    name: str
    n_genes: int
    rho: float
    p_competitive: float
    null_mean: float
    null_sd: float
    z_competitive: float
    n_draws: int
    mean_stability: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def differential_stability(
    per_donor: list[pd.DataFrame], method: str = "spearman"
) -> pd.Series:
    """Mean pairwise cross-donor correlation of each gene's regional profile.

    Parameters
    ----------
    per_donor : list of DataFrame
        One expression matrix per donor, all with the same parcels (rows) and
        genes (columns).
    method : {'spearman', 'pearson'}

    Returns
    -------
    Series
        Differential stability per gene, indexed by gene symbol. Genes present
        in fewer than two donors get NaN.
    """
    if len(per_donor) < 2:
        raise ValueError("differential stability needs at least two donors")

    genes = per_donor[0].columns
    for d in per_donor[1:]:
        genes = genes.intersection(d.columns)
    logger.info(
        "differential stability over %d genes, %d donors", len(genes), len(per_donor)
    )

    mats = [d[genes].to_numpy() for d in per_donor]
    if method == "spearman":
        mats = [sps.rankdata(m, axis=0, nan_policy="omit") for m in mats]

    # Correlate each donor pair column-wise, then average over pairs.
    acc = np.zeros(len(genes))
    n_pairs = 0
    for i in range(len(mats)):
        for j in range(i + 1, len(mats)):
            a, b = mats[i], mats[j]
            ok = np.isfinite(a) & np.isfinite(b)
            with np.errstate(invalid="ignore"):
                az = (a - np.nanmean(a, axis=0)) / np.nanstd(a, axis=0)
                bz = (b - np.nanmean(b, axis=0)) / np.nanstd(b, axis=0)
                r = np.nanmean(np.where(ok, az * bz, np.nan), axis=0)
            acc += np.nan_to_num(r)
            n_pairs += 1
    return pd.Series(acc / n_pairs, index=genes, name="differential_stability")


def matched_random_sets(
    target_genes: list[str],
    all_genes: pd.Index,
    stability: pd.Series,
    n_draws: int = 10_000,
    n_bins: int = 10,
    seed: int = 42,
) -> list[list[str]]:
    """Draw random gene sets matched to a target on size and stability.

    The target's genes are binned by differential stability; each random draw
    takes the same number of genes from each bin. That preserves both the size
    and the stability composition of the real set.

    Parameters
    ----------
    target_genes : list of str
        The real gene set.
    all_genes : Index
        Every gene available to sample from.
    stability : Series
        Differential stability per gene.
    n_draws : int
        Number of random sets.
    n_bins : int
        Stability bins.
    seed : int
        RNG seed (determinism is required — see the repo's reproducibility rule).

    Returns
    -------
    list of list of str
    """
    present = [g for g in target_genes if g in stability.index]
    if not present:
        raise ValueError("none of the target genes have stability values")

    pool = stability.reindex(all_genes).dropna()
    # Rank-based bins so each holds a comparable number of genes.
    bins = pd.qcut(pool.rank(method="first"), n_bins, labels=False)
    bin_of = pd.Series(bins, index=pool.index)

    need = bin_of.reindex(present).dropna().astype(int).value_counts().to_dict()
    members = {b: pool.index[bin_of == b].to_numpy() for b in range(n_bins)}

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_draws):
        pick: list[str] = []
        for b, k in need.items():
            avail = members[b]
            if len(avail) >= k:
                pick.extend(rng.choice(avail, size=k, replace=False))
        draws.append(pick)
    return draws


def competitive_null(
    expression: pd.DataFrame,
    target_map: np.ndarray,
    genes: list[str],
    stability: pd.Series,
    name: str = "",
    n_draws: int = 10_000,
    seed: int = 42,
    score_fn: Callable[[pd.DataFrame, list[str]], np.ndarray] | None = None,
) -> CompetitiveResult:
    """Test a gene set against size- and stability-matched random sets.

    Parameters
    ----------
    expression : DataFrame
        Parcels x genes.
    target_map : ndarray
        Parcel values to correlate against.
    genes : list of str
        The gene set.
    stability : Series
        Differential stability per gene.
    name : str
    n_draws : int
    seed : int
    score_fn : callable, optional
        How to reduce a gene set to one value per parcel. Defaults to the mean
        of z-scored expression.

    Returns
    -------
    CompetitiveResult
    """
    if score_fn is None:

        def score_fn(exp: pd.DataFrame, gs: list[str]) -> np.ndarray:
            sub = exp[[g for g in gs if g in exp.columns]]
            z = (sub - sub.mean()) / sub.std()
            return z.mean(axis=1).to_numpy()

    present = [g for g in genes if g in expression.columns]
    if not present:
        raise ValueError(f"gene set {name!r}: no genes present in the expression matrix")

    obs_score = score_fn(expression, present)
    ok = np.isfinite(obs_score) & np.isfinite(target_map)
    rho = float(sps.spearmanr(obs_score[ok], target_map[ok]).statistic)

    draws = matched_random_sets(
        present, expression.columns, stability, n_draws=n_draws, seed=seed
    )
    null = np.full(len(draws), np.nan)
    for i, gs in enumerate(draws):
        if not gs:
            continue
        s = score_fn(expression, gs)
        m = np.isfinite(s) & np.isfinite(target_map)
        if m.sum() >= 3:
            null[i] = sps.spearmanr(s[m], target_map[m]).statistic

    null = null[np.isfinite(null)]
    # Two-tailed, +1 corrected: a permutation p is never exactly zero.
    n_extreme = int(np.sum(np.abs(null) >= abs(rho)))
    p = (n_extreme + 1) / (len(null) + 1)
    sd = float(np.std(null))

    return CompetitiveResult(
        name=name,
        n_genes=len(present),
        rho=rho,
        p_competitive=float(p),
        null_mean=float(np.mean(null)),
        null_sd=sd,
        z_competitive=float((rho - np.mean(null)) / sd) if sd > 0 else np.nan,
        n_draws=len(null),
        mean_stability=float(stability.reindex(present).mean()),
    )
