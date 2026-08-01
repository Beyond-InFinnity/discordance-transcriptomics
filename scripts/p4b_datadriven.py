#!/usr/bin/env python
"""Phase 4b — the data-driven arm (CLAUDE.md §8.2).

The frozen arm asked whether eleven pre-specified gene sets explain the target
maps. They do not, for discordance. This arm asks the complementary question:
does *any* gene programme, including ones nobody thought to freeze, explain them?

Three analyses per target, each with spatial-null inference:

1. **Transcriptome screen.** Every stable gene correlated against the target,
   with its own spin p-value, plus a transcriptome-level test asking whether
   there are more strong genes than rotation of the target allows. The
   transcriptome-level test is the one to read: per-gene BH-FDR across ~15,000
   genes is floored at ``n_genes / (n_perm + 1)``, which at 10,000 rotations is
   above 1.0, so no gene can clear 0.05 however real it is.
2. **Tail enrichment.** Are the extremes of the ranking enriched for the frozen
   sets, against a null of random sets matched on differential stability?
3. **PLS.** Does a multivariate expression component predict the target better
   than it predicts rotations of it? PLS on 15,000 genes and 100 parcels
   overfits enormously, so the spin test is not a refinement here — it is the
   only thing that makes the number interpretable at all.

Run on the primary pipeline plus a spread across the multiverse, since the screen
is cheap but not free and R6 asks for a distribution rather than a point estimate.

Usage
-----
    python scripts/p4b_datadriven.py
    python scripts/p4b_datadriven.py --max-cells 4 --no-pls
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.parcellate import gifti_for_nulls
from src.expression.datadriven import (
    gene_screen,
    pls_with_spin,
    screen_summary,
    tail_enrichment,
)
from src.expression.multiverse import cell_path, multiverse_dir
from src.stats.spatial import apply_spin, spin_indices
from src.utils.config import load_config
from src.utils.manifest import manifest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4_genesets import load_genesets, targets_for

logger = logging.getLogger("p4b_datadriven")


# The primary pipeline (§7.5). Reported as the headline; the rest form the
# multiverse distribution.
PRIMARY = {
    "probe_selection": "diff_stability",
    "lr_mirror": "bidirectional",
    "missing": "centroids",
    # All five, or "primary" matches the four cells that leave tolerance and
    # norm_matched free and every headline number is reported four times.
    "tolerance": "2",
    "norm_matched": "True",
}


def select_cells(idx: pd.DataFrame, n_cells: int | None) -> pd.DataFrame:
    """Primary pipeline first, then an even spread over the rest."""
    is_primary = np.ones(len(idx), dtype=bool)
    for k, v in PRIMARY.items():
        is_primary &= idx[k].astype(str) == str(v)
    primary, rest = idx[is_primary], idx[~is_primary]
    if n_cells is None or n_cells >= len(idx):
        return pd.concat([primary, rest])
    take = max(0, n_cells - len(primary))
    step = max(1, len(rest) // take) if take else 1
    return pd.concat([primary, rest.iloc[::step].head(take)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--max-cells", type=int, default=12)
    ap.add_argument("--n-draws", type=int, default=10_000)
    ap.add_argument("--pls-components", type=int, default=3)
    ap.add_argument("--pls-perm", type=int, default=1000)
    ap.add_argument("--no-pls", action="store_true", help="skip the PLS stage")
    ap.add_argument(
        "--stability-threshold",
        type=float,
        default=0.1,
        help="keep genes at or above this differential stability (§7.5)",
    )
    ap.add_argument(
        "--parcellation",
        default=None,
        help="override the primary parcellation (§7.1/§11 sensitivity analyses)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.logging.level, format=cfg.logging.format)
    parc = args.parcellation or cfg.parcellation.primary.name
    density = cfg.parcellation.primary.density

    mv_dir = multiverse_dir(cfg, parc)
    idx = pd.read_csv(mv_dir / "multiverse_index.csv")
    idx = idx[idx.status.isin(["ok", "cached"])]
    cells = select_cells(idx, args.max_cells)
    logger.info("cells: %d of %d available", len(cells), len(idx))

    stability = pd.read_csv(mv_dir / "differential_stability.csv", index_col=0).iloc[:, 0]
    keep_genes = set(stability[stability >= args.stability_threshold].index)
    logger.info(
        "stability >= %.2f keeps %d of %d genes",
        args.stability_threshold,
        len(keep_genes),
        len(stability),
    )

    targets = targets_for(cfg, parc)
    gsets = {k: v["genes"] for k, v in load_genesets().items()}
    n_parcels = len(next(iter(targets.values())))

    sidx = spin_indices(
        n_parcels,
        atlas=cfg.parcellation.primary.space,
        density=density,
        parcellation=gifti_for_nulls(parc, density, "L"),
        n_perm=cfg.nulls.n_perm,
        seed=cfg.seed,
        method=cfg.nulls.surface_method,
        cache_path=cfg.path("nulls") / f"spin_indices_{parc}_{density}.npy",
    )
    nulls = {k: apply_spin(v, sidx) for k, v in targets.items()}

    out = Path("results")
    # Suffix non-primary parcellations so a sensitivity run never overwrites the
    # headline result. §11 requires reporting whether each effect holds at DK-68
    # and Schaefer-400, and that is impossible if they share filenames.
    tag = "" if parc == cfg.parcellation.primary.name else f"_{parc}"
    out.mkdir(exist_ok=True)
    summaries: list[dict] = []
    enrich_rows: list[pd.DataFrame] = []
    pls_rows: list[pd.DataFrame] = []
    top_genes: dict[str, list] = {}

    for n_cell, (_, cell) in enumerate(cells.iterrows(), 1):
        exp = pd.read_parquet(cell_path(mv_dir, cell)).iloc[:n_parcels]
        exp = exp[[c for c in exp.columns if c in keep_genes]]
        is_primary = all(str(cell[k]) == str(v) for k, v in PRIMARY.items())

        for tname, y in targets.items():
            screen = gene_screen(exp, y, nulls[tname])
            s = screen_summary(screen)
            summaries.append(
                {
                    "cell": cell["hash"],
                    "primary": is_primary,
                    "target": tname,
                    "probe_selection": cell["probe_selection"],
                    "lr_mirror": cell["lr_mirror"],
                    "missing": cell["missing"],
                    **s,
                }
            )

            en = tail_enrichment(
                screen, gsets, stability, n_draws=args.n_draws, seed=cfg.seed
            )
            if len(en):
                en["cell"] = cell["hash"]
                en["target"] = tname
                en["primary"] = is_primary
                enrich_rows.append(en)

            if is_primary:
                # The full ranking is only kept for the headline pipeline —
                # 15,000 rows x every cell x every target is not a useful artifact.
                screen.to_csv(out / f"p4b_screen_{tname}.csv")
                top_genes[tname] = {
                    "positive": screen.head(50).index.tolist(),
                    "negative": screen.tail(50).index.tolist()[::-1],
                }
                if not args.no_pls:
                    p = pls_with_spin(
                        exp,
                        y,
                        nulls[tname],
                        n_components=args.pls_components,
                        max_perm=args.pls_perm,
                    )
                    p["target"] = tname
                    p["cell"] = cell["hash"]
                    pls_rows.append(p)

        logger.info("  %d/%d cells", n_cell, len(cells))

    summ = pd.DataFrame(summaries)
    enrich = pd.concat(enrich_rows, ignore_index=True) if enrich_rows else pd.DataFrame()
    pls = pd.concat(pls_rows, ignore_index=True) if pls_rows else pd.DataFrame()

    with manifest(f"p4b_datadriven{tag}", cfg) as man:
        summ.to_csv(out / f"p4b_screen_summary{tag}.csv", index=False)
        if len(enrich):
            enrich.to_csv(out / f"p4b_tail_enrichment{tag}.csv", index=False)
        if len(pls):
            pls.to_csv(out / f"p4b_pls{tag}.csv", index=False)
        (out / f"p4b_top_genes{tag}.json").write_text(json.dumps(top_genes, indent=2))

        sig = summ[summ.p < cfg.stats.alpha]
        man.record(
            outputs=[str(p) for p in sorted(out.glob("p4b_*"))],
            n_cells=len(cells),
            n_cells_available=len(idx),
            stability_threshold=args.stability_threshold,
            n_genes_screened=int(summ.n_genes.iloc[0]) if len(summ) else 0,
            n_perm=int(sidx.shape[1]),
            n_competitive_draws=args.n_draws,
            targets=list(targets),
            n_target_cell_pairs=len(summ),
            n_with_transcriptome_excess=len(sig),
            targets_with_excess=sorted(sig.target.unique().tolist()),
            per_gene_fdr_floor=(
                float(summ.per_gene_fdr_floor.iloc[0]) if len(summ) else None
            ),
        )
        floor_txt = f"{summ.per_gene_fdr_floor.iloc[0]:.2f}" if len(summ) else "unknown"
        man.note(
            "Per-gene BH-FDR is reported but is floored at n_genes/(n_perm+1) = "
            f"{floor_txt}, so no single gene can reach 0.05 at this permutation "
            "budget. The transcriptome-level test in p4b_screen_summary.csv is "
            "the inferential result: it counts genes above a null-derived "
            "threshold and compares against rotations of the target, which "
            "preserves co-expression because all genes are scored against the "
            "same rotated map within a draw."
        )
        man.note(
            "Exploratory by construction (R5). Nothing here licenses a "
            "confirmatory claim; a programme found here is a hypothesis for "
            "another dataset. Its job is to bound the frozen arm's negative."
        )

    print(f"\n{'=' * 78}\nPHASE 4b — DATA-DRIVEN ARM\n{'=' * 78}")
    if len(summ):
        print(
            f"  {int(summ.n_genes.iloc[0])} genes x {int(sidx.shape[1])} rotations "
            f"x {len(cells)} pipelines"
        )
        print(
            f"  per-gene FDR floor: {summ.per_gene_fdr_floor.iloc[0]:.2f} "
            f"(> 1 means no gene can ever clear 0.05)\n"
        )
        print("  transcriptome-level excess, primary pipeline:")
        print(f"    {'target':<26}{'hits':>7}{'null':>9}{'z':>8}{'p':>8}")
        for _, r in summ[summ.primary].iterrows():
            print(
                f"    {r.target[:25]:<26}{r.n_observed:>7}{r.null_mean:>9.1f}"
                f"{r.z:>8.2f}{r.p:>8.4f}"
            )
        agg = (
            summ.groupby("target")
            .agg(pct_sig=("p", lambda x: (x < cfg.stats.alpha).mean()), n=("p", "size"))
            .reset_index()
        )
        print("\n  across pipelines (fraction showing an excess):")
        for _, r in agg.iterrows():
            print(f"    {r.target[:25]:<26}{r.pct_sig:>7.0%}  of {int(r.n)}")
    if len(enrich):
        e = enrich[enrich.primary & (enrich["p"] < cfg.stats.alpha)]
        print(f"\n  tail enrichment, primary pipeline, p < {cfg.stats.alpha}: {len(e)}")
        for _, r in e.sort_values("p").head(10).iterrows():
            print(
                f"    {r['name'][:34]:<36}{r['target'][:22]:<24}"
                f"{r['tail']:<10}z={r['z']:>6.2f} p={r['p']:.4f}"
            )
    if len(pls):
        print("\n  PLS, primary pipeline:")
        for _, r in pls.iterrows():
            print(
                f"    {r.target[:25]:<26}comp {int(r.component)}  "
                f"r2={r.r2_target:.3f}  p_spin={r.p_spin:.4f}"
            )
    print(f"\n  -> results/p4b_*\n{'=' * 78}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
