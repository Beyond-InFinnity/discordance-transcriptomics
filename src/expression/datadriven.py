"""Data-driven arm of Phase 4 — CLAUDE.md §8.2.

The hypothesis-driven arm asks whether eleven pre-specified gene sets predict the
target maps. This arm asks the complementary question: does *any* gene programme,
including ones nobody thought to freeze in advance, predict them? Three analyses,
all with spatial-null inference:

1. :func:`gene_screen` — every stable gene correlated against the target, each
   with its own spin p-value, then BH-FDR across the transcriptome.
2. :func:`tail_enrichment` — are the extremes of that ranking enriched for known
   biology? Tested against a null that resamples genes *matched on differential
   stability*, because stable genes have smoother maps and correlate better with
   everything, so an unmatched null finds "enrichment" everywhere.
3. :func:`pls_with_spin` — partial least squares from the full expression matrix
   to the target, with the variance explained by each component tested against
   rotations of the target.

**Why this is not p-hacking.** It is pre-registered in §8.2, it runs in parallel
with the frozen arm rather than replacing it, and nothing it finds is licensed to
make a confirmatory claim (R5). A gene set discovered here is a hypothesis for
someone else's dataset. What it *can* do is bound the negative: if 15,000 genes
tested properly turn up nothing above chance, "no molecular signal" is a much
stronger statement than "our eleven sets missed."

**Why the whole transcriptome is affordable now.** 15,000 genes against 10,000
rotations is 150 million correlations. As scipy calls that is days; as two matrix
products on ranked data it is seconds, because a Spearman correlation of
z-scored ranks is just a dot product. The memory cost is the (n_genes, n_perm)
null block, which is chunked over genes.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

logger = logging.getLogger(__name__)

__all__ = [
    "EnrichmentResult",
    "PLSResult",
    "gene_screen",
    "pls_with_spin",
    "screen_summary",
    "tail_enrichment",
]


def _zrank(a: np.ndarray) -> np.ndarray:
    """Rank along axis 0, then centre and scale to unit norm.

    After this, a Spearman correlation between two columns is their dot product,
    which is what makes the whole-transcriptome screen a matrix product.
    """
    r = sps.rankdata(a, axis=0).astype(float)
    r -= r.mean(axis=0)
    n = np.linalg.norm(r, axis=0)
    return np.divide(r, n, out=np.zeros_like(r), where=n > 0)


def gene_screen(
    expression: pd.DataFrame,
    target: np.ndarray,
    target_nulls: np.ndarray,
    chunk: int = 2000,
) -> pd.DataFrame:
    """Correlate every gene against the target, with a spin p-value each.

    Parameters
    ----------
    expression : DataFrame, shape (n_parcels, n_genes)
        Parcel-by-gene expression for one multiverse cell.
    target : ndarray, shape (n_parcels,)
        The map being explained.
    target_nulls : ndarray, shape (n_parcels, n_perm)
        Surrogates of ``target``. Required (R1).
    chunk : int
        Genes per block, bounding peak memory at roughly
        ``chunk * n_perm * 8`` bytes.

    Returns
    -------
    DataFrame indexed by gene, columns ``rho``, ``p_spin``, ``p_fdr``,
    ``p_maxt``, sorted by ``rho`` descending. Genes constant across parcels are
    dropped. Read ``p_maxt`` for per-gene inference — see the note at its
    computation for why ``p_fdr`` cannot resolve at transcriptome scale.

    Raises
    ------
    ValueError
        If nulls are missing or shapes disagree.
    """
    if target_nulls is None:
        raise ValueError(
            "gene_screen() requires spatial surrogates of the target (R1). A "
            "transcriptome-wide screen without them is 15,000 uninterpretable "
            "p-values, not a result."
        )
    target = np.asarray(target, dtype=float).ravel()
    target_nulls = np.asarray(target_nulls, dtype=float)
    if target_nulls.ndim != 2:
        raise ValueError(f"target_nulls must be 2D, got {target_nulls.shape}")
    if target_nulls.shape[0] != target.shape[0]:
        raise ValueError(
            f"target_nulls has {target_nulls.shape[0]} parcels, target has "
            f"{target.shape[0]}"
        )
    if expression.shape[0] != target.shape[0]:
        raise ValueError(
            f"expression has {expression.shape[0]} parcels, target has {target.shape[0]}"
        )

    # Parcels usable in the observed comparison. Surrogates are handled by
    # dropping non-finite null columns rather than parcels, since a rotation can
    # pull an unobserved parcel in and demanding complete draws empties the set.
    valid = np.isfinite(target) & np.isfinite(expression.to_numpy()).all(axis=1)
    if valid.sum() < 10:
        # Fall back to per-gene masking only if the all-gene mask is too strict.
        valid = np.isfinite(target)
    n_valid = int(valid.sum())
    if n_valid < 10:
        raise ValueError(f"only {n_valid} valid parcels")

    expr = expression[valid]
    tgt = target[valid]
    nul = target_nulls[valid, :]
    good_cols = np.isfinite(nul).all(axis=0)
    if not good_cols.any():
        raise ValueError("every surrogate contains a missing parcel")
    nul = nul[:, good_cols]
    n_perm = nul.shape[1]
    if good_cols.sum() < target_nulls.shape[1]:
        logger.info(
            "using %d/%d surrogates (rest touched an unobserved parcel)",
            n_perm,
            target_nulls.shape[1],
        )

    G = expr.to_numpy(dtype=float)
    keep = np.isfinite(G).all(axis=0) & (np.nanstd(G, axis=0) > 0)
    genes = expr.columns[keep]
    G = _zrank(G[:, keep])
    t = _zrank(tgt[:, None]).ravel()
    N = _zrank(nul)

    rho = G.T @ t
    n_genes = len(genes)
    logger.info("screening %d genes against %d surrogates", n_genes, n_perm)

    # The hit threshold has to come from the NULL, not from the observed
    # correlations. Taking the observed 95th percentile makes the test vacuous:
    # 5% of observed genes exceed it by definition, the rotations also average
    # 5%, and the comparison is 5% against 5% whatever the biology. Estimated
    # from a subsample of rotations across all genes, which is unbiased for the
    # pooled null and costs one extra small matrix product.
    n_probe = min(200, n_perm)
    probe = np.abs(G.T @ N[:, :n_probe])
    thresh = float(np.quantile(probe, 0.95))
    del probe

    # Two counts from the same blocks.
    #
    # Per gene: how many surrogates reach its own |rho|. That is the gene's spin
    # p-value, resolution-limited at 1/(n_perm+1).
    #
    # Per surrogate: how many genes reach the null threshold. That gives a null
    # distribution for the *number* of hits, which is what the
    # transcriptome-level test needs — and unlike a per-gene correction it
    # respects co-expression, because every gene is scored against the same
    # rotated map within a draw. See :func:`screen_summary`.
    n_extreme = np.zeros(n_genes, dtype=np.int64)
    hits_per_perm = np.zeros(n_perm, dtype=np.int64)
    # Running maximum |rho| over genes, per rotation — the Westfall-Young
    # max-statistic. See the p_maxt note below.
    max_per_perm = np.zeros(n_perm, dtype=float)
    for lo in range(0, n_genes, chunk):
        hi = min(lo + chunk, n_genes)
        block = np.abs(G[:, lo:hi].T @ N)  # (chunk, n_perm)
        n_extreme[lo:hi] = (block >= np.abs(rho[lo:hi])[:, None]).sum(axis=1)
        hits_per_perm += (block >= thresh).sum(axis=0)
        np.maximum(max_per_perm, block.max(axis=0), out=max_per_perm)

    p_spin = (n_extreme + 1) / (n_perm + 1)
    out = pd.DataFrame({"rho": rho, "p_spin": p_spin}, index=genes)
    out.index.name = "gene"

    from src.stats.spatial import fdr_bh

    # Two multiple-comparison corrections, because the obvious one does not work
    # at this scale.
    #
    # p_fdr: BH across the transcriptome. Cannot resolve below
    # n_genes/(n_perm+1) — with 15,562 genes and 10,000 rotations the smallest
    # achievable adjusted p is 1.56, so *every* gene returns 1.0 however strong
    # it is. Kept because it is the right correction once the permutation budget
    # can support it, with the floor attached so it cannot be misread as
    # "nothing was significant".
    #
    # p_maxt: Westfall-Young. Compare each gene's |rho| against the distribution
    # of the *largest* |rho| anywhere in the transcriptome under each rotation.
    # This is family-wise error control, it resolves to 1/(n_perm+1) regardless
    # of gene count, and it inherits the co-expression structure for free because
    # the maximum is taken across genes within a single rotated map. It is the
    # right per-gene answer here, and it is strictly stronger than BH: a gene
    # clearing p_maxt < 0.05 has beaten every other gene's best shot at chance.
    out["p_fdr"] = fdr_bh(out.p_spin.to_numpy())
    out["p_maxt"] = [
        (int((max_per_perm >= abs(r)).sum()) + 1) / (n_perm + 1) for r in rho
    ]
    floor = n_genes / (n_perm + 1)
    out.attrs["fdr_floor"] = floor
    out.attrs["hits_per_perm"] = hits_per_perm
    out.attrs["max_per_perm"] = max_per_perm
    out.attrs["threshold"] = thresh
    out.attrs["n_perm"] = n_perm
    if floor > 0.05:
        logger.info(
            "per-gene FDR is floored at %.2f (%d genes, %d rotations); read "
            "p_maxt for per-gene inference and screen_summary() for the "
            "transcriptome-level test",
            floor,
            n_genes,
            n_perm,
        )
    return out.sort_values("rho", ascending=False)


def screen_summary(screen: pd.DataFrame) -> dict[str, Any]:
    """Transcriptome-level test: are there more strong genes than rotation allows?

    Per-gene inference is resolution-limited, and per-gene FDR is floored at
    ``n_genes / (n_perm + 1)`` — with 15,000 genes and 10,000 rotations no gene
    can ever clear 0.05, however real it is. Asking the question at the level of
    the whole transcriptome sidesteps that entirely: count the genes whose
    correlation exceeds a threshold, and compare against how many exceed it when
    the target map is rotated.

    Because all genes are scored against the *same* rotated map within a draw,
    the null carries the co-expression structure of the transcriptome. A
    per-gene correction assuming independence would not, and would be
    anticonservative by a wide margin — genes are heavily correlated with each
    other.

    Parameters
    ----------
    screen : DataFrame
        Output of :func:`gene_screen`, with its ``attrs`` intact. Note that
        ``attrs`` do not survive most pandas round-trips, so call this on the
        object returned by the screen rather than on a reloaded CSV.

    Returns
    -------
    dict
        ``n_observed``, ``null_mean``, ``null_sd``, ``z``, ``p``, plus the
        threshold and the resolution limits, ready for a manifest.
    """
    if "hits_per_perm" not in screen.attrs:
        raise ValueError(
            "screen is missing its attrs; pass the DataFrame returned by "
            "gene_screen() rather than one reloaded from disk"
        )
    hits = np.asarray(screen.attrs["hits_per_perm"], dtype=float)
    thresh = float(screen.attrs["threshold"])
    n_obs = int((screen.rho.abs() >= thresh).sum())
    mu, sd = float(hits.mean()), float(hits.std())
    n_ext = int((hits >= n_obs).sum())
    return {
        "n_genes": len(screen),
        "n_perm": int(screen.attrs["n_perm"]),
        "threshold_abs_rho": thresh,
        "n_observed": n_obs,
        "null_mean": mu,
        "null_sd": sd,
        "z": float((n_obs - mu) / sd) if sd > 0 else float("nan"),
        "p": (n_ext + 1) / (len(hits) + 1),
        "per_gene_fdr_floor": float(screen.attrs["fdr_floor"]),
        "frac_p_spin_below_05": float((screen.p_spin < 0.05).mean()),
        "n_genes_maxt_below_05": int((screen.p_maxt < 0.05).sum()),
        "min_p_maxt": float(screen.p_maxt.min()),
    }


@dataclass(frozen=True)
class EnrichmentResult:
    """One gene set tested against one tail of the screen ranking."""

    name: str
    tail: str
    n_in_set: int
    n_overlap: int
    observed: float
    null_mean: float
    null_sd: float
    z: float
    p: float
    n_draws: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def tail_enrichment(
    screen: pd.DataFrame,
    gene_sets: dict[str, list[str]],
    stability: pd.Series,
    tail_frac: float = 0.05,
    n_draws: int = 10_000,
    seed: int = 42,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Test whether each gene set concentrates in a tail of the screen ranking.

    The statistic is the fraction of the set's genes falling in the tail. The
    null resamples same-size gene sets **matched on differential stability**,
    binned by rank. Stability matching is the point: a stable gene has a smoother,
    more reproducible spatial map and so correlates better with any smooth map,
    which means an unmatched null reports enrichment for every stable set
    regardless of biology. This is the same principle as the competitive null in
    the hypothesis-driven arm (R2), applied to a ranking rather than a mean.

    Parameters
    ----------
    screen : DataFrame
        Output of :func:`gene_screen`, sorted by ``rho``.
    gene_sets : dict
        Set name to gene symbols.
    stability : Series
        Differential stability per gene.
    tail_frac : float
        Fraction of the ranking treated as each tail.
    n_draws : int
        Matched random sets per test.
    seed : int
        R7.
    n_bins : int
        Stability bins used for matching.

    Returns
    -------
    DataFrame, one row per (gene set, tail).
    """
    ranked = screen.sort_values("rho", ascending=False)
    genes = ranked.index.to_numpy()
    n = len(genes)
    k = max(1, round(n * tail_frac))
    tails = {"positive": set(genes[:k]), "negative": set(genes[-k:])}

    # Bin genes by stability rank so a matched draw has the same stability
    # profile as the real set. Genes absent from the stability table are given
    # the median rather than dropped, so a set is never silently shrunk.
    stab = stability.reindex(genes)
    if stab.isna().all():
        stab = pd.Series(np.zeros(n), index=genes)
    stab = stab.fillna(stab.median())
    # Equal-count bins on the rank, then to a plain int array: qcut returns a
    # categorical whose NaN handling and dtype vary with pandas version, and
    # indexing a dict by those labels is how this broke.
    bins = np.asarray(
        pd.qcut(
            stab.rank(method="first"), min(n_bins, n), labels=False, duplicates="drop"
        ),
        dtype=float,
    )
    bins = np.nan_to_num(bins, nan=-1.0).astype(int)
    by_bin: dict[int, np.ndarray] = {
        int(b): np.flatnonzero(bins == b) for b in np.unique(bins)
    }
    pos = {g: i for i, g in enumerate(genes)}
    rng = np.random.default_rng(seed)

    rows = []
    for name, members in gene_sets.items():
        idx = np.array([pos[g] for g in members if g in pos], dtype=int)
        if len(idx) < 3:
            logger.info("%s: %d genes present, skipping", name, len(idx))
            continue
        # Bin composition of the real set, matched draw for draw.
        set_bins = bins[idx]
        counts = {int(b): int((set_bins == b).sum()) for b in np.unique(set_bins)}
        draws = np.empty((n_draws, len(idx)), dtype=int)
        col = 0
        for b, cnt in counts.items():
            pool = by_bin[b]
            if len(pool) <= cnt:
                # Bin too small to draw distinctly; take it whole, padding by
                # repetition rather than borrowing from a different stability
                # band, which would break the matching this null exists for.
                draws[:, col : col + cnt] = np.resize(pool, cnt)[None, :]
            else:
                # Without replacement *within* each draw, independently across
                # draws. rng.choice with a 2D size and replace=False instead
                # demands n_draws*cnt distinct items from the pool, which is a
                # different (and usually impossible) request. argpartition on
                # random keys gives cnt distinct positions per row in O(len(pool)).
                keys = rng.random((n_draws, len(pool)))
                pick = np.argpartition(keys, cnt - 1, axis=1)[:, :cnt]
                draws[:, col : col + cnt] = pool[pick]
            col += cnt

        for tail, members_in_tail in tails.items():
            in_tail = np.isin(genes, list(members_in_tail))
            obs = float(in_tail[idx].mean())
            null = in_tail[draws].mean(axis=1)
            mu, sd = float(null.mean()), float(null.std())
            n_ext = int((null >= obs).sum())
            rows.append(
                EnrichmentResult(
                    name=name,
                    tail=tail,
                    n_in_set=len(members),
                    n_overlap=len(idx),
                    observed=obs,
                    null_mean=mu,
                    null_sd=sd,
                    z=float((obs - mu) / sd) if sd > 0 else float("nan"),
                    p=(n_ext + 1) / (n_draws + 1),
                    n_draws=n_draws,
                ).as_dict()
            )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class PLSResult:
    """Partial least squares from expression to the target, with a spin null."""

    component: int
    r2_target: float
    p_spin: float
    n_perm: int
    n_genes: int
    n_parcels: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def pls_with_spin(
    expression: pd.DataFrame,
    target: np.ndarray,
    target_nulls: np.ndarray,
    n_components: int = 3,
    max_perm: int = 1000,
) -> pd.DataFrame:
    """PLS regression with each component's fit tested against target rotations.

    The quantity tested is the share of target variance the component explains.
    Refitting PLS for every rotation is the expensive part, so ``max_perm``
    caps the null at a level that still resolves p to about 0.001 — the screen in
    :func:`gene_screen` carries the full 10,000.

    Parameters
    ----------
    expression : DataFrame, shape (n_parcels, n_genes)
    target : ndarray, shape (n_parcels,)
    target_nulls : ndarray, shape (n_parcels, n_perm)
        Required (R1).
    n_components : int
        Components to extract and test.
    max_perm : int
        Rotations used.

    Returns
    -------
    DataFrame, one row per component.
    """
    if target_nulls is None:
        raise ValueError("pls_with_spin() requires spatial surrogates (R1)")
    from sklearn.cross_decomposition import PLSRegression

    target = np.asarray(target, dtype=float).ravel()
    valid = np.isfinite(target) & np.isfinite(expression.to_numpy()).all(axis=1)
    if valid.sum() < 10:
        valid = np.isfinite(target)
    X = expression[valid].to_numpy(dtype=float)
    X = X[:, np.isfinite(X).all(axis=0) & (X.std(axis=0) > 0)]
    y = target[valid]
    nul = target_nulls[valid, :]
    nul = nul[:, np.isfinite(nul).all(axis=0)][:, :max_perm]
    n_perm = nul.shape[1]

    def fit_r2(yv: np.ndarray) -> np.ndarray:
        """Cumulative variance in ``yv`` explained, per component."""
        out = np.empty(n_components)
        yc = (yv - yv.mean()) / (yv.std() or 1.0)
        for k in range(1, n_components + 1):
            p = PLSRegression(n_components=k, scale=True)
            p.fit(X, yv)
            pred = p.predict(X).ravel()
            pred = (pred - pred.mean()) / (pred.std() or 1.0)
            out[k - 1] = float(np.corrcoef(pred, yc)[0, 1] ** 2)
        return out

    logger.info(
        "PLS: %d genes, %d parcels, %d components, %d rotations",
        X.shape[1],
        X.shape[0],
        n_components,
        n_perm,
    )
    obs = fit_r2(y)
    null = np.empty((n_perm, n_components))
    for i in range(n_perm):
        null[i] = fit_r2(nul[:, i])

    rows = []
    for k in range(n_components):
        n_ext = int((null[:, k] >= obs[k]).sum())
        rows.append(
            PLSResult(
                component=k + 1,
                r2_target=float(obs[k]),
                p_spin=(n_ext + 1) / (n_perm + 1),
                n_perm=n_perm,
                n_genes=int(X.shape[1]),
                n_parcels=int(X.shape[0]),
            ).as_dict()
        )
    return pd.DataFrame(rows)
