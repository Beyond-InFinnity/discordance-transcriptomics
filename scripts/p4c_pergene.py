#!/usr/bin/env python
"""Phase 4c — per-gene spatial association, aggregated at the set level.

Phase 4 scores a gene set by averaging its members' expression into one map and
correlating that. Phase 0c measures what that costs: averaging *k* genes improves
reliability only when their true spatial patterns resemble each other more than
their measurement errors do, and AHBA measures every gene from the same tissue
samples, so error is shared by construction while signal is not. For large
pathway sets the average cancels signal faster than noise, and the resulting map
can be less reliable than any single gene inside it.

This runs the same frozen sets against the same targets under the same two null
models, changing only *where the aggregation happens*: every gene is tested
individually, and the set-level question — is this set enriched for strong
spatial associations relative to size- and stability-matched random sets? — is
asked of the resulting statistics rather than of an averaged map.

Reported alongside Phase 4, never instead of it. The pre-registration froze which
genes (§8.1) and which nulls (§7.4); it did not specify how genes are combined
into a score, and the unweighted mean was an inherited convention rather than a
registered decision. See Appendix A of the draft.

WHY THIS IS AFFORDABLE
----------------------
Testing ~15,500 genes against 10,000 rotations sounds prohibitive and is not,
because the rotations do not depend on the gene. With the surrogate block ranked
and centred once per target, all genes and all rotations collapse into a single
matrix product — (10,000 x 100) @ (100 x n_genes) — about 31 GFLOP per
cell x target, roughly 0.1 s at the float64 CPU throughput measured in
CLAUDE.md §4.2. The whole multiverse is minutes, not days.

It stays on CPU in float64 deliberately. §4.2's precision contract is "identical
decisions with the discrepancy measured": float64 on the consumer cards here runs
at 0.31 TFLOPS against the CPU's 0.28, so the only GPU win is float32, which
flips threshold comparisons at ~5e-7 and moves permutation p-values by 1/n_perm.
That is a bad trade on a job this short, and §4.2 names the spin test explicitly
as work a GPU does not justify.

Usage
-----
    python scripts/p4c_pergene.py
    python scripts/p4c_pergene.py --max-cells 4 --n-draws 200   # smoke test
"""

# Greek rho and the true minus sign are correct typography in prose that
# humans read; ruff flags them as ambiguous with latin p.
# ruff: noqa: RUF002
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from p4_genesets import (
    cell_path,
    load_genesets,
    multiverse_dir,
    targets_for,
)

from src.data.parcellate import gifti_for_nulls
from src.stats.spatial import apply_spin, prepare_nulls, spin_indices
from src.utils.config import load_config
from src.utils.manifest import manifest

logger = logging.getLogger("p4c_pergene")

N_PARCELS_LH = 100


def rank_centre(a: np.ndarray) -> np.ndarray:
    """Column-wise ranks, centred. Matches prepare_nulls() so ρ is Spearman."""
    r = sps.rankdata(a, axis=0)
    return r - r.mean(axis=0)


def pergene_spin(
    expr: np.ndarray,
    target: np.ndarray,
    nulls_prepared: np.ndarray,
    chunk: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Spearman ρ and two-tailed spin p for every gene against one target.

    Parameters
    ----------
    expr : (n_parcels, n_genes)
    target : (n_parcels,)
    nulls_prepared : (n_parcels, n_perm), already ranked and centred.

    Returns
    -------
    rho, p_spin : each (n_genes,)

    The genes are chunked only to bound memory: the full (n_genes x n_perm)
    correlation block is 1.2 GB in float64 and is never needed at once, since
    only the count of |null| >= |observed| survives it.
    """
    _n_par, n_genes = expr.shape
    n_perm = nulls_prepared.shape[1]

    tc = rank_centre(target[:, None])[:, 0]
    t_norm = np.linalg.norm(tc)
    n_norm = np.linalg.norm(nulls_prepared, axis=0)  # (n_perm,)

    rho = np.full(n_genes, np.nan)
    p = np.full(n_genes, np.nan)
    for lo in range(0, n_genes, chunk):
        hi = min(lo + chunk, n_genes)
        gc = rank_centre(expr[:, lo:hi])  # (n_par, m)
        g_norm = np.linalg.norm(gc, axis=0)  # (m,)
        with np.errstate(invalid="ignore", divide="ignore"):
            obs = (gc * tc[:, None]).sum(axis=0) / (g_norm * t_norm)
            # (m, n_perm): every gene against every rotation, one product.
            nullr = (gc.T @ nulls_prepared) / np.outer(g_norm, n_norm)
        # Two-tailed with the +1 correction, matching corr_with_null().
        n_extreme = (np.abs(nullr) >= np.abs(obs)[:, None]).sum(axis=1)
        rho[lo:hi] = obs
        p[lo:hi] = (n_extreme + 1) / (n_perm + 1)
    return rho, p


def run_cell(
    cell_file: Path,
    targets: dict[str, np.ndarray],
    raw_nulls: dict[str, np.ndarray],
    chunk: int,
    prepared: dict[tuple[str, bytes], np.ndarray | None],
    ragged: set[str],
) -> pd.DataFrame:
    """Per-gene ρ and spin p for one multiverse cell, all targets.

    Two things here are easy to get wrong and were:

    **The parcel mask is per cell, not global.** Multiverse cells with
    ``missing=None`` leave parcels with no AHBA coverage entirely empty — cell
    ``01e98e73`` has parcel 27 blank for all 15,562 genes. Requiring every gene
    to be finite across all 100 parcels therefore selects nothing at all, which
    is how the first version of this returned zero rows for every cell.

    **Surrogates must be ranked after masking, not before.** ``corr_with_null``
    subsets ``nulls[valid, :]`` and *then* ranks and centres. Preparing the full
    100-parcel block once and slicing rows afterwards leaves ranks computed over
    parcels that are no longer in the comparison and a centre that no longer
    sums to zero, so every null correlation is subtly wrong. The prepared block
    is therefore cached per (target, mask) — the same trick ``p4_genesets`` uses,
    since most cells share a mask and re-ranking each time is 86% of the cost.
    """
    exp = pd.read_parquet(cell_file).iloc[:N_PARCELS_LH]
    have_expr = exp.notna().any(axis=1).to_numpy()
    rows = []
    for tname, y in targets.items():
        if tname in ragged:
            continue
        mask = np.isfinite(y) & have_expr
        if mask.sum() < 10:
            continue
        rowsel = mask.nonzero()[0]
        keep = exp.iloc[rowsel].notna().all()
        sub = exp.loc[:, keep]
        if sub.shape[1] == 0:
            continue

        key = (tname, mask.tobytes())
        nprep = prepared.get(key)
        if nprep is None:
            nprep = prepare_nulls(raw_nulls[tname][mask, :])
            prepared[key] = nprep

        rho, p = pergene_spin(
            sub.to_numpy(float)[mask], np.asarray(y, float)[mask], nprep, chunk
        )
        rows.append(
            pd.DataFrame(
                {
                    "gene": sub.columns,
                    "target": tname,
                    "rho": rho,
                    "p_spin": p,
                    "n_parcels": int(mask.sum()),
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def competitive_pergene(
    stat: pd.Series,
    genes: list[str],
    stability: pd.Series,
    n_draws: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Is this set's aggregate per-gene statistic beyond matched random sets?

    Matching is on size and on differential stability, as in
    ``src/stats/competitive.py`` — the difference is that the statistic being
    aggregated is a per-gene spatial association rather than a correlation of an
    averaged map. Draws are stratified by stability decile so a set of unusually
    stable genes is not compared against the atlas at large.
    """
    # The pool is genes that have BOTH a statistic and a stability value, and
    # membership must be judged against it rather than against stat alone.
    # Matching on stability is impossible for a gene with no stability, and
    # selecting on stat.index put such genes in `present` and not in the decile
    # table, which raised a KeyError naming six of them.
    pool = stat.index.intersection(stability.index)
    present = [g for g in genes if g in pool]
    if len(present) < 3:
        return float("nan"), float("nan"), float("nan")
    obs = float(stat.loc[present].mean())

    s = stability.loc[pool]
    deciles = pd.qcut(s.rank(method="first"), 10, labels=False)
    want = pd.Series(deciles.loc[present]).value_counts()
    by_dec = {d: np.asarray(deciles.index[deciles == d]) for d in range(10)}

    draws = np.empty(n_draws)
    for i in range(n_draws):
        picked = [
            rng.choice(by_dec[d], size=min(k, len(by_dec[d])), replace=False)
            for d, k in want.items()
            if len(by_dec[d])
        ]
        draws[i] = stat.loc[np.concatenate(picked)].mean()
    z = (obs - draws.mean()) / draws.std() if draws.std() > 0 else np.nan
    p = (int((np.abs(draws - draws.mean()) >= abs(obs - draws.mean())).sum()) + 1) / (
        n_draws + 1
    )
    return obs, float(z), float(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--n-draws", type=int, default=10_000)
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--chunk-genes", type=int, default=2000)
    ap.add_argument("--parcellation", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.logging.level, format=cfg.logging.format)
    parc = args.parcellation or cfg.parcellation.primary.name
    density = cfg.parcellation.primary.density

    mv_dir = multiverse_dir(cfg, parc)
    idx = pd.read_csv(mv_dir / "multiverse_index.csv")
    idx = idx[idx.status.isin(["ok", "cached"])]
    if args.max_cells:
        idx = idx.head(args.max_cells)
    logger.info("cells: %d", len(idx))

    gsets = load_genesets()
    targets = targets_for(cfg, parc)
    sidx = spin_indices(
        len(next(iter(targets.values()))),
        atlas=cfg.parcellation.primary.space,
        density=density,
        parcellation=gifti_for_nulls(parc, density, "L"),
        n_perm=cfg.nulls.n_perm,
        seed=cfg.seed,
        method=cfg.nulls.surface_method,
        cache_path=cfg.path("nulls") / f"spin_indices_{parc}_{density}.npy",
    )
    # Raw surrogates; prepare_nulls() runs per (target, mask) inside run_cell,
    # because ranking before masking would rank parcels that are then dropped.
    raw_nulls = {k: apply_spin(v, sidx) for k, v in targets.items()}
    prepared: dict[tuple[str, bytes], np.ndarray | None] = {}

    # Which targets this arm cannot score, decided ONCE from the target's own
    # finite mask -- not discovered inside the cell loop, which is skipped
    # entirely on a resumed run and would then leave the exclusion list empty
    # while the excluded rows came back as NaN.
    #
    # A rotation can pull a parcel the target never observed into the analysis
    # window, so a map with missing parcels yields surrogates with NaNs.
    # prepare_nulls() deliberately returns such a block untouched -- ranking it
    # would be meaningless -- and consuming it as if prepared feeds raw uncentred
    # values into a correlation, driving every p toward zero. spatial.py
    # documents this exact failure ("that is how the cross-species control came
    # back at p = 1/3"), and the first run of this script reproduced it:
    # macaque_vascular_CONTROL returned 100% significant for all 11 gene sets,
    # from 83/100 finite parcels and 2 complete draws of 10,000.
    # corr_with_null() has a paired-observation path for the ragged case; this
    # vectorised sweep does not, so such targets are named and excluded.
    ragged: set[str] = set()
    for tname, y in targets.items():
        m = np.isfinite(np.asarray(y, float))
        blk = raw_nulls[tname][m, :]
        if not bool(np.isfinite(blk).all()):
            n_complete = int(np.isfinite(blk).all(axis=0).sum())
            logger.warning(
                "%s: ragged surrogates (%d/%d complete draws, %d/%d finite parcels)"
                " — excluded from the per-gene arm; Phase 4 scores it through"
                " corr_with_null's paired path",
                tname,
                n_complete,
                blk.shape[1],
                int(m.sum()),
                m.size,
            )
            ragged.add(tname)
    logger.info("targets: %s", ", ".join(targets))

    stability = pd.read_csv(mv_dir / "differential_stability.csv", index_col=0).iloc[:, 0]

    # Checkpoint per cell. A cell killed by a resource ceiling then costs one
    # cell, and a rerun skips whatever is already on disk -- the pattern
    # p3_multiverse.py uses for the same reason.
    ck = cfg.path("expression") / f"p4c_cells_{parc}"
    ck.mkdir(parents=True, exist_ok=True)

    for n, (_, cell) in enumerate(idx.iterrows(), 1):
        dest = ck / f"{cell.hash}.parquet"
        if dest.exists():
            continue
        df = run_cell(
            cell_path(mv_dir, cell),
            targets,
            raw_nulls,
            args.chunk_genes,
            prepared,
            ragged,
        )
        if len(df):
            df.to_parquet(dest)
        logger.info("cell %d/%d (%s): %d rows", n, len(idx), cell.hash[:8], len(df))

    have = sorted(ck.glob("*.parquet"))
    logger.info("aggregating %d cells", len(have))
    per_cell = pd.concat([pd.read_parquet(p) for p in have], ignore_index=True)

    # Across the multiverse, per gene x target (R6).
    agg = (
        per_cell.groupby(["gene", "target"])
        .agg(
            rho_median=("rho", "median"),
            rho_q1=("rho", lambda s: s.quantile(0.25)),
            rho_q3=("rho", lambda s: s.quantile(0.75)),
            pct_sign=("rho", lambda s: 100 * max((s > 0).mean(), (s < 0).mean())),
            pct_spin_sig=("p_spin", lambda s: 100 * (s < 0.05).mean()),
            n_cells=("rho", "size"),
        )
        .reset_index()
    )

    out = Path("results")
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for tname in targets:
        if tname in ragged:
            continue
        a = agg[agg.target == tname].set_index("gene")
        # Aggregate statistic per gene: how often its association survives the
        # spatial null across the multiverse. Sign-free, so a set whose genes
        # split in direction is not rewarded for cancelling.
        stat = a.pct_spin_sig
        for gname, spec in gsets.items():
            obs, z, p = competitive_pergene(
                stat, spec["genes"], stability, args.n_draws, rng
            )
            present = [g for g in spec["genes"] if g in a.index]
            rows.append(
                {
                    "gene_set": gname,
                    "target": tname,
                    "n_genes_present": len(present),
                    "mean_pct_spin_sig": obs,
                    "z_competitive": z,
                    "p_competitive": p,
                    "median_abs_rho": float(a.loc[present, "rho_median"].abs().median())
                    if present
                    else np.nan,
                }
            )
    summary = pd.DataFrame(rows)
    summary = summary[~summary.target.isin(ragged)].sort_values("p_competitive")

    # Genome-wide calibration, per target. A spin test controls the rotated
    # map's autocorrelation, so how conservative it is depends on the relative
    # smoothness of the two maps being compared -- and that differs by target
    # rather than being a property of any gene set. Measured here: baseline OEF,
    # the smoothest target, returns ~0.8% of genes at p < 0.05 while the noisier
    # discordance maps return ~8-12%, against the 5% a uniform null would give.
    #
    # This is why the set-level question is asked through a competitive null
    # (R2) rather than by counting significant genes: matched random sets are
    # scored through the same non-uniformity, so it cancels. The numbers are
    # recorded because a reader needs them to interpret mean_pct_spin_sig, which
    # is not comparable across targets on its own.
    calib = (
        per_cell.groupby("target")
        .agg(
            n_tests=("p_spin", "size"),
            pct_p_lt_05=("p_spin", lambda s: 100 * (s < 0.05).mean()),
            pct_p_lt_01=("p_spin", lambda s: 100 * (s < 0.01).mean()),
            mean_p=("p_spin", "mean"),
            median_abs_rho=("rho", lambda s: s.abs().median()),
        )
        .reset_index()
    )

    with manifest(f"p4c_pergene_{parc}", cfg, results_dir=out) as man:
        out.mkdir(parents=True, exist_ok=True)
        agg.to_csv(out / f"p4c_pergene_genes_{parc}.csv", index=False)
        calib.to_csv(out / f"p4c_pergene_calibration_{parc}.csv", index=False)
        summary.to_csv(out / f"p4c_pergene_summary_{parc}.csv", index=False)
        man.record(
            # Every file written above must be listed. The calibration table was
            # written and not declared, so audit_provenance's third check found a
            # CSV in results/ that no manifest claimed and failed the run 5/6.
            # An artifact without provenance is exactly what that gate exists to
            # catch, and it caught it.
            outputs=[
                str(out / f"p4c_pergene_genes_{parc}.csv"),
                str(out / f"p4c_pergene_summary_{parc}.csv"),
                str(out / f"p4c_pergene_calibration_{parc}.csv"),
            ],
            n_cells=len(have),
            n_genes=int(agg.gene.nunique()),
            n_targets=len(targets),
            n_tests=len(agg),
            n_draws=args.n_draws,
            n_perm=cfg.nulls.n_perm,
            seed=cfg.seed,
            targets_excluded_ragged=sorted(ragged),
            pct_genes_spin_sig_by_target={
                r.target: round(float(r.pct_p_lt_05), 3) for _, r in calib.iterrows()
            },
        )
        man.note(
            "Per-gene spatial association with set-level competitive aggregation. "
            "Same frozen sets, same targets, same two nulls as Phase 4; only the "
            "level at which genes are combined differs. Reported alongside Phase "
            "4's averaged-score result, not as a replacement — the construction "
            "was not pre-registered either way (Appendix A)."
        )

    print(f"\n{'=' * 78}\nPER-GENE ARM — {parc}\n{'=' * 78}")
    print(f"  {len(have)} cells x {agg.gene.nunique():,} genes x {len(targets)} targets")
    print(f"\n  {'gene set':34}{'target':>24}{'mean %sig':>11}{'z':>8}{'p':>9}")
    for _, r in summary.iterrows():
        print(
            f"  {r.gene_set[:33]:34}{r.target:>24}{r.mean_pct_spin_sig:>11.2f}"
            f"{r.z_competitive:>8.2f}{r.p_competitive:>9.4f}"
        )
    print("\n  spin-test calibration (genome-wide, per target):")
    print(f"  {'target':30}{'n tests':>10}{'% p<0.05':>10}{'mean p':>9}")
    for _, r in calib.iterrows():
        print(
            f"  {r.target:30}{int(r.n_tests):>10,}{r.pct_p_lt_05:>10.2f}{r.mean_p:>9.3f}"
        )
    if ragged:
        print(f"\n  excluded (ragged surrogates): {sorted(ragged)}")
    print(f"\n  -> {out}/p4c_pergene_summary_{parc}.csv\n{'=' * 78}")
    return 0


if __name__ == "__main__":
    # One BLAS thread per process: the inner matmul is already the whole job, and
    # oversubscribing it against the outer loop costs more than it gains.
    os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
    raise SystemExit(main())
