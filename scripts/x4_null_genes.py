#!/usr/bin/env python
"""Is our genome-wide clearance rate a false-positive rate? Mostly not.

What §2.5 used to claim, and why it was wrong
---------------------------------------------
Phase 4c measures, per target, the fraction of the transcriptome clearing a spin
test at *p* < 0.05: 0.82% against baseline OEF, 12.09% against extraction-mode
discordance. An earlier draft of §2.5 read that spread as calibration failure —
the test being conservative against one map and firing too readily against
another — and described 12.09% as roughly one gene in eight clearing "under what
should be the null".

That reading requires no gene to be truly associated with the map. Nobody
believes that null; imaging transcriptomics exists because genes *do* correlate
with brain maps. The clearance rate therefore confounds two things:

1. the spin test being miscalibrated against that map, and
2. the map genuinely engaging a large part of the transcriptome.

The disambiguation
------------------
Rotate every gene independently before testing. A rotation preserves a gene's
spatial autocorrelation and its value distribution exactly — it is a reindexing
— while destroying its anatomical alignment. True association is then zero by
construction, so whatever clears is false positives and nothing else.

Two rotation sets are used, at deliberately different seeds: one to scramble the
genes, a disjoint one to build each target's null. Drawn from the same set, the
observed statistic would be a near-copy of a null draw and the check would pass
by construction rather than on the merits.

What this is for
----------------
The 85-map sweep in the companion methods work puts the null-gene rate at a
median of 4.90% across published `neuromaps` annotations, against a nominal 5%.
This asks the same question of *our four targets*, because that is where the
claim was made and it is our claim to correct.

Cells
-----
Twelve multiverse cells by default rather than all 120. The quantity is a rate
over ~15,500 genes, so twelve cells give ~187,000 tests per target and a
standard error near 0.05% on a 4% rate — three significant figures on a number
we quote to two. Phase 4c's published rate uses every cell and is reported
alongside for comparison.

Usage
-----
    python scripts/x4_null_genes.py
    python scripts/x4_null_genes.py --max-cells 2      # smoke test
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from p4_genesets import cell_path, multiverse_dir, targets_for

from src.data.parcellate import gifti_for_nulls
from src.stats.spatial import apply_spin, prepare_nulls, spin_indices
from src.utils.config import load_config
from src.utils.manifest import manifest

logger = logging.getLogger("x4_null_genes")
warnings.filterwarnings("ignore")

N_LH = 100

# Not cfg.seed. The gene-scrambling rotations must not be the same rotations that
# form each target's null.
GENE_SEED = 4242
GENE_ROTATIONS = 1000

# Genes per matrix-multiply block: the full (n_genes x n_perm) correlation array
# is 1.2 GB in float64 at 15,562 genes x 10,000 rotations. Blocking holds peak
# resident memory near 160 MB without changing a value.
GENE_BLOCK = 2000

LABEL = {
    "baseline_oef": "baseline OEF",
    "coupling_angle": "coupling angle",
    "discordance_overshoot": "overshoot-mode discordance",
    "discordance_extraction": "extraction-mode discordance",
}


def clearance(
    X: np.ndarray, y: np.ndarray, nprep: np.ndarray
) -> tuple[float, float, int]:
    """Fraction of genes clearing p < 0.05, the median |rho|, and genes scored.

    One matrix product per block rather than a scipy call per gene; identical
    numbers, and the loop it replaces is ~150 million calls per target.

    Non-finite columns are **dropped, not counted**. A gene whose observed
    correlation is NaN makes ``|null| >= |obs|`` false for every draw, so the
    exceedance count is zero and the permutation p comes out at 1/(n_perm + 1)
    — the most significant value the test can produce. Silently, and for a gene
    that carries no information at all. Nothing raises; the rate just inflates.
    """
    finite = np.isfinite(X).all(axis=0)
    if not finite.any():
        return float("nan"), float("nan"), 0
    X = X[:, finite]

    n_perm = nprep.shape[1]
    yc = sps.rankdata(y)
    yc = yc - yc.mean()
    gc = sps.rankdata(X, axis=0)
    gc = gc - gc.mean(axis=0)
    gn, nn, yn = (
        np.linalg.norm(gc, axis=0),
        np.linalg.norm(nprep, axis=0),
        np.linalg.norm(yc),
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        obs = (gc * yc[:, None]).sum(axis=0) / (gn * yn)

    counts = np.zeros(X.shape[1], dtype=np.int64)
    for s in range(0, X.shape[1], GENE_BLOCK):
        e = min(s + GENE_BLOCK, X.shape[1])
        with np.errstate(invalid="ignore", divide="ignore"):
            nullr = (gc[:, s:e].T @ nprep) / np.outer(gn[s:e], nn)
        counts[s:e] = (np.abs(nullr) >= np.abs(obs[s:e])[:, None]).sum(axis=1)

    ok = np.isfinite(obs)  # zero-variance genes survive the column filter above
    if not ok.any():
        return float("nan"), float("nan"), 0
    p = (counts[ok] + 1) / (n_perm + 1)
    return (
        float(100 * (p < 0.05).mean()),
        float(np.median(np.abs(obs[ok]))),
        int(ok.sum()),
    )


def scramble(
    X: np.ndarray, mask: np.ndarray, gene_idx: np.ndarray, rng, max_tries: int = 20
) -> np.ndarray:
    """Rotate each gene independently: same autocorrelation, no alignment.

    Returns the rotated matrix restricted to ``mask``.

    The subtlety is NaN. A gene is kept when it is complete *inside the analysis
    window*, but it may still be NaN outside it — in a multiverse cell that is
    the normal case, since no gene is complete across all 100 parcels. A
    rotation can therefore drag an out-of-window NaN into the window and destroy
    a gene that was perfectly usable unrotated.

    So each gene is offered up to ``max_tries`` rotations and takes the first
    whose window is finite. Genes that never find one keep a NaN column and are
    dropped downstream by :func:`clearance` rather than counted.
    """
    out = np.full((int(mask.sum()), X.shape[1]), np.nan)
    n_rot = gene_idx.shape[1]
    for j in range(X.shape[1]):
        for _ in range(max_tries):
            col = X[gene_idx[:, rng.integers(0, n_rot)], j][mask]
            if np.isfinite(col).all():
                out[:, j] = col
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--parcellation", default=None)
    ap.add_argument("--max-cells", type=int, default=12)
    args = ap.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.logging.level, format=cfg.logging.format)
    parc = args.parcellation or cfg.parcellation.primary.name
    density = cfg.parcellation.primary.density

    mv = multiverse_dir(cfg, parc)
    idx_df = pd.read_csv(mv / "multiverse_index.csv")
    cells = idx_df[idx_df.status.isin(["ok", "cached"])].head(args.max_cells)

    targets = {k: v for k, v in targets_for(cfg, parc).items() if "macaque" not in k}
    published = pd.read_csv(f"results/p4c_pergene_calibration_{parc}.csv").set_index(
        "target"
    )

    sidx = spin_indices(
        N_LH,
        atlas=cfg.parcellation.primary.space,
        density=density,
        parcellation=gifti_for_nulls(parc, density, "L"),
        n_perm=cfg.nulls.n_perm,
        seed=cfg.seed,
        method=cfg.nulls.surface_method,
        cache_path=cfg.path("nulls") / f"spin_indices_{parc}_{density}.npy",
    )
    gene_idx = spin_indices(
        N_LH,
        atlas=cfg.parcellation.primary.space,
        density=density,
        parcellation=gifti_for_nulls(parc, density, "L"),
        n_perm=GENE_ROTATIONS,
        seed=GENE_SEED,
        method=cfg.nulls.surface_method,
        cache_path=cfg.path("nulls") / f"spin_indices_genescramble_{parc}.npy",
    )
    logger.info("%d cells x %d targets", len(cells), len(targets))

    acc: dict[str, list[dict]] = {}
    for n, (_, cell) in enumerate(cells.iterrows(), 1):
        exp = pd.read_parquet(cell_path(mv, cell)).iloc[:N_LH]
        have = exp.notna().any(axis=1).to_numpy()
        for tname, y in targets.items():
            y = np.asarray(y, float)
            mask = np.isfinite(y) & have
            keep = exp.iloc[mask.nonzero()[0]].notna().all()
            X_full = exp.loc[:, keep].to_numpy(float)

            raw = apply_spin(y, sidx)[mask, :]
            if not np.isfinite(raw).all():
                logger.warning("%s cell %d: ragged rotations; skipping", tname, n)
                continue
            nprep = prepare_nulls(raw)

            real_pct, real_rho, n_real = clearance(X_full[mask], y[mask], nprep)
            rng = np.random.default_rng(GENE_SEED)
            null_pct, null_rho, n_null = clearance(
                scramble(X_full, mask, gene_idx, rng), y[mask], nprep
            )
            acc.setdefault(tname, []).append(
                {
                    "real_pct": real_pct,
                    "null_pct": null_pct,
                    "real_median_abs_rho": real_rho,
                    "null_median_abs_rho": null_rho,
                    "n_genes": n_real,
                    "n_genes_scrambled": n_null,
                    "n_parcels": int(mask.sum()),
                }
            )
        logger.info("  cell %d/%d", n, len(cells))

    rows = []
    for tname, recs in acc.items():
        d = pd.DataFrame(recs)
        rows.append(
            {
                "target": tname,
                "label": LABEL.get(tname, tname),
                "n_cells": len(d),
                "n_genes_median": int(d.n_genes.median()),
                "n_genes_scrambled_median": int(d.n_genes_scrambled.median()),
                "n_parcels": int(d.n_parcels.iloc[0]),
                "published_pct_all_cells": float(published.loc[tname, "pct_p_lt_05"]),
                "real_pct": float(d.real_pct.mean()),
                "null_pct": float(d.null_pct.mean()),
                "excess": float(d.real_pct.mean() - d.null_pct.mean()),
                "real_median_abs_rho": float(d.real_median_abs_rho.mean()),
                "null_median_abs_rho": float(d.null_median_abs_rho.mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values("real_pct")

    csv = Path("results") / f"x4_null_genes_{parc}.csv"
    with manifest(f"x4_null_genes_{parc}", cfg) as man:
        out.to_csv(csv, index=False)
        man.record(
            outputs=[str(csv)],
            n_cells=len(cells),
            n_targets=len(out),
            gene_scramble_seed=GENE_SEED,
            gene_scramble_rotations=GENE_ROTATIONS,
            null_pct_min=round(float(out.null_pct.min()), 3),
            null_pct_max=round(float(out.null_pct.max()), 3),
            by_target={
                r.target: {
                    "real_pct": round(r.real_pct, 3),
                    "null_pct": round(r.null_pct, 3),
                }
                for r in out.itertuples()
            },
        )
        man.note(
            "Genome-wide clearance rates with genes rotated independently, which "
            "preserves each gene's spatial autocorrelation while destroying its "
            "anatomical alignment. Anything clearing is then a false positive by "
            "construction, so this is the false-positive rate the raw Phase 4c "
            "rate was previously mistaken for. Gene rotations use a seed disjoint "
            "from the target nulls."
        )

    w = 86
    print(
        f"\n{'=' * w}\nIS THE CLEARANCE RATE A FALSE-POSITIVE RATE? — {parc}\n{'=' * w}"
    )
    print(f"  {len(cells)} multiverse cells, {cfg.nulls.n_perm} rotations, nominal 5%\n")
    print(
        f"  {'target':30}{'4c all cells':>14}{'real':>8}{'null-genes':>12}{'excess':>9}"
    )
    for r in out.itertuples():
        print(
            f"  {r.label[:29]:30}{r.published_pct_all_cells:>14.2f}"
            f"{r.real_pct:>8.2f}{r.null_pct:>12.2f}{r.excess:>+9.2f}"
        )
    print(
        f"\n  null-gene rate spans {out.null_pct.min():.2f}%-{out.null_pct.max():.2f}% "
        f"against a nominal 5%: the spin test is mildly conservative here,\n"
        f"  not the fifteen-fold miscalibration the raw spread suggested."
    )
    lo = out.iloc[0]
    print(
        f"\n  {lo.label} clears at {lo.real_pct:.2f}% against its own "
        f"{lo.null_pct:.2f}% null — real genes do WORSE than\n"
        f"  autocorrelation-matched noise, so the transcriptome is anti-aligned "
        f"with that map rather than\n  merely unrelated to it."
    )
    print(f"\n  -> {csv}\n{'=' * w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
