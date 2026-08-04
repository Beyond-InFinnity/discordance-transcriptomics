#!/usr/bin/env python
"""x3 — is a competitive-null result spatial autocorrelation?

The competitive null (§7.4) matches random gene sets on size and differential
stability. It does not match on spatial autocorrelation, and a spin test's
conservativeness depends on the smoothness of *both* maps compared: a smooth
target produces smooth rotations that a smooth gene can correlate with by chance,
widening its null and making its test conservative. A set of unusually smooth
genes can therefore look depleted — or an unusually rough one enriched — for
reasons unconnected to biology. Fulcher et al. (2021) document the same mechanism
inflating false positives in gene-category enrichment.

Phase 4c reports HALLMARK_OXPHOS as depleted against three targets. This asks
whether that is autocorrelation, and deliberately applies the same test to
pericyte/mural, the study's headline result. Both arms of the paper use a null
blind to autocorrelation, so their agreement cannot rescue a finding from that
blind spot. A control that can only confirm what one hopes is not a control.

  1. Are the set's genes unusual in autocorrelation? Moran's I on the parcel
     graph, and the SD of each gene's own spin-null — the quantity that
     mechanically sets how conservative its test is.
  2. Does autocorrelation predict per-gene spin significance genome-wide? A
     matching variable that does not predict the statistic cannot explain a
     shift in it.
  3. Does the result survive draws stratified on autocorrelation as well as
     stability?

Reads Phase 4c's per-gene statistics, so it must follow it. Autocorrelation is a
property of the expression map rather than of the preprocessing choice, so step 1
runs over a subset of pipelines while step 3 uses Phase 4c's full 120-cell
statistic.

Usage
-----
    python scripts/x3_autocorr_matched.py
    python scripts/x3_autocorr_matched.py --n-cells 8 --n-draws 500   # quick
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from p4_genesets import cell_path, load_genesets, multiverse_dir, targets_for

from src.data.parcellate import get_parcellation, gifti_for_nulls
from src.stats.spatial import apply_spin, prepare_nulls, spin_indices
from src.utils.config import load_config
from src.utils.manifest import manifest

logger = logging.getLogger("x3_autocorr")

N_LH = 100
KNN = 8
FOCUS = ["HALLMARK_OXIDATIVE_PHOSPHORYLATION", "pericyte_mural"]


def parcel_weights(parc: str, density: str) -> np.ndarray:
    """Row-standardised k-nearest-neighbour weights over parcel centroids.

    Centroids come from the inflated fsaverage5 surface, so distance is euclidean
    rather than geodesic — adequate for identifying neighbours, which is all a
    kNN weight matrix uses.
    """
    from nilearn import datasets, surface

    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    coords, _ = surface.load_surf_mesh(fs["infl_left"])
    labels, _g, _n = get_parcellation(parc, density, "L")
    cen = np.full((N_LH, 3), np.nan)
    for lab in range(1, N_LH + 1):
        m = labels == lab
        if m.any():
            cen[lab - 1] = coords[m].mean(axis=0)
    d = np.linalg.norm(cen[:, None, :] - cen[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    W = np.zeros_like(d)
    for i in range(N_LH):
        W[i, np.argsort(d[i])[:KNN]] = 1.0
    return W / W.sum(axis=1, keepdims=True)


def morans_i(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Moran's I for every column of X (parcels x genes), row-standardised W."""
    Z = X - X.mean(axis=0)
    num = np.einsum("ig,ij,jg->g", Z, W, Z)
    den = (Z**2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def competitive(
    stat: pd.Series,
    genes: list[str],
    strata: pd.DataFrame,
    keys: list[str],
    n_draws: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Competitive null with draws matched on an arbitrary set of strata.

    ``keys=["ds"]`` reproduces the published null; ``["ds", "moran"]`` adds
    autocorrelation. Matching is on the joint cell, so a set that is both
    unusually stable and unusually smooth is compared against random sets that
    are both.
    """
    pool = stat.index.intersection(strata.index)
    present = [g for g in genes if g in pool]
    if len(present) < 3:
        return np.nan, np.nan, np.nan
    obs = float(stat.loc[present].mean())
    cell = strata.loc[pool, keys].astype(str).agg("|".join, axis=1)
    want = cell.loc[present].value_counts()
    by_cell = {c: np.asarray(cell.index[cell == c]) for c in cell.unique()}
    draws = np.empty(n_draws)
    for i in range(n_draws):
        picked = [
            rng.choice(by_cell[c], size=min(k, len(by_cell[c])), replace=False)
            for c, k in want.items()
            if c in by_cell and len(by_cell[c])
        ]
        draws[i] = stat.loc[np.concatenate(picked)].mean()
    sd = draws.std()
    z = (obs - draws.mean()) / sd if sd > 0 else np.nan
    p = (int((np.abs(draws - draws.mean()) >= abs(obs - draws.mean())).sum()) + 1) / (
        n_draws + 1
    )
    return obs, float(z), float(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--n-cells", type=int, default=16)
    ap.add_argument("--n-draws", type=int, default=10_000)
    ap.add_argument("--parcellation", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.logging.level, format=cfg.logging.format)
    parc = args.parcellation or cfg.parcellation.primary.name
    density = cfg.parcellation.primary.density
    out = Path("results")

    mv = multiverse_dir(cfg, parc)
    idx = pd.read_csv(mv / "multiverse_index.csv")
    idx = idx[idx.status.isin(["ok", "cached"])].head(args.n_cells)
    targets = {k: v for k, v in targets_for(cfg, parc).items() if "macaque" not in k}
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
    raw = {k: apply_spin(v, sidx) for k, v in targets.items()}
    W = parcel_weights(parc, density)

    # --- steps 1-2: per-gene autocorrelation, and whether it predicts --------
    acc: dict[str, list] = {}
    moran_acc = []
    for n, (_, cell) in enumerate(idx.iterrows(), 1):
        exp = pd.read_parquet(cell_path(mv, cell)).iloc[:N_LH]
        have = exp.notna().any(axis=1).to_numpy()
        for tname, y in targets.items():
            mask = np.isfinite(np.asarray(y, float)) & have
            keep = exp.iloc[mask.nonzero()[0]].notna().all()
            sub = exp.loc[:, keep]
            X = sub.to_numpy(float)[mask]
            nprep = prepare_nulls(raw[tname][mask, :])
            tc = sps.rankdata(np.asarray(y, float)[mask])
            tc = tc - tc.mean()
            gc = sps.rankdata(X, axis=0)
            gc = gc - gc.mean(axis=0)
            gn, tn = np.linalg.norm(gc, axis=0), np.linalg.norm(tc)
            nn = np.linalg.norm(nprep, axis=0)
            obs = (gc * tc[:, None]).sum(0) / (gn * tn)
            nullr = (gc.T @ nprep) / np.outer(gn, nn)
            p = ((np.abs(nullr) >= np.abs(obs)[:, None]).sum(1) + 1) / (
                nullr.shape[1] + 1
            )
            acc.setdefault(tname, []).append(
                pd.DataFrame(
                    {
                        "gene": sub.columns,
                        "sig": (p < 0.05).astype(float),
                        "null_sd": nullr.std(axis=1),
                    }
                )
            )
        sw = W[np.ix_(have, have)]
        moran_acc.append(
            pd.Series(
                morans_i(exp.to_numpy(float)[have], sw / sw.sum(1, keepdims=True)),
                index=exp.columns,
            )
        )
        logger.info("cell %d/%d", n, len(idx))

    moran = pd.concat(moran_acc, axis=1).mean(axis=1).rename("moran_i")
    sets = load_genesets()
    diag_rows = []
    per_target = {}
    for tname in targets:
        d = pd.concat(acc[tname]).groupby("gene").mean().join(moran, how="inner")
        per_target[tname] = d
        for sname in FOCUS:
            inset = d.index.isin(sets[sname]["genes"])
            diag_rows.append(
                {
                    "gene_set": sname,
                    "target": tname,
                    "moran_set": float(d.moran_i[inset].mean()),
                    "moran_other": float(d.moran_i[~inset].mean()),
                    "moran_mwu_p": float(
                        sps.mannwhitneyu(d.moran_i[inset], d.moran_i[~inset]).pvalue
                    ),
                    "nullsd_set": float(d.null_sd[inset].mean()),
                    "nullsd_other": float(d.null_sd[~inset].mean()),
                    "nullsd_mwu_p": float(
                        sps.mannwhitneyu(d.null_sd[inset], d.null_sd[~inset]).pvalue
                    ),
                    "moran_predicts_sig": float(
                        sps.spearmanr(d.moran_i, d.sig).statistic
                    ),
                }
            )
    diag = pd.DataFrame(diag_rows)

    # --- step 3: matched null, on Phase 4c's full-multiverse statistic -------
    ds = pd.read_csv(mv / "differential_stability.csv", index_col=0).iloc[:, 0]
    agg = pd.read_csv(out / f"p4c_pergene_genes_{parc}.csv")
    rows = []
    for tname, d in per_target.items():
        stat = agg[agg.target == tname].set_index("gene").pct_spin_sig
        common = stat.index.intersection(d.index).intersection(ds.index)
        strata = pd.DataFrame(
            {
                "ds": pd.qcut(ds.loc[common].rank(method="first"), 10, labels=False),
                "moran": pd.qcut(
                    d.loc[common, "moran_i"].rank(method="first"), 5, labels=False
                ),
            },
            index=common,
        )
        st = stat.loc[common]
        for name, spec in sets.items():
            rng = np.random.default_rng(cfg.seed)
            o1, z1, p1 = competitive(st, spec["genes"], strata, ["ds"], args.n_draws, rng)
            rng = np.random.default_rng(cfg.seed)
            _o2, z2, p2 = competitive(
                st, spec["genes"], strata, ["ds", "moran"], args.n_draws, rng
            )
            rows.append(
                {
                    "gene_set": name,
                    "target": tname,
                    "obs": o1,
                    "z_ds": z1,
                    "p_ds": p1,
                    "z_ds_moran": z2,
                    "p_ds_moran": p2,
                }
            )
    matched = pd.DataFrame(rows)

    with manifest(f"x3_autocorr_matched_{parc}", cfg, results_dir=out) as man:
        out.mkdir(parents=True, exist_ok=True)
        diag.to_csv(out / f"x3_autocorr_diagnostics_{parc}.csv", index=False)
        matched.to_csv(out / f"x3_autocorr_matched_{parc}.csv", index=False)
        man.record(
            outputs=[
                str(out / f"x3_autocorr_diagnostics_{parc}.csv"),
                str(out / f"x3_autocorr_matched_{parc}.csv"),
            ],
            n_cells_autocorr=len(idx),
            n_draws=args.n_draws,
            knn=KNN,
            n_clear_ds_only=int((matched.p_ds < 0.05).sum()),
            n_clear_ds_moran=int((matched.p_ds_moran < 0.05).sum()),
        )
        man.note(
            "Adds spatial autocorrelation to the competitive null's matching. The "
            "published null matches size and differential stability only, and a "
            "spin test's conservativeness depends on the smoothness of both maps, "
            "so a set of unusually smooth genes can appear depleted for reasons "
            "unconnected to biology. pericyte_mural is tested alongside the "
            "depletions because both arms of the paper share this blind spot and "
            "their agreement cannot resolve it."
        )

    print(f"\n{'=' * 92}\nAUTOCORRELATION-MATCHED NULL — {parc}\n{'=' * 92}")
    print(f"\n  {'gene set':36}{'target':>24}{'Moran set':>11}{'others':>9}{'MWU p':>11}")
    for _, r in diag.iterrows():
        print(
            f"  {r.gene_set[:35]:36}{r.target:>24}{r.moran_set:>11.4f}"
            f"{r.moran_other:>9.4f}{r.moran_mwu_p:>11.3g}"
        )
    print(
        f"\n  {'gene set':36}{'target':>24}{'z(ds)':>9}{'p(ds)':>9}"
        f"{'z(+mor)':>10}{'p(+mor)':>10}"
    )
    key = matched[(matched.p_ds < 0.05) | (matched.gene_set.isin(FOCUS))]
    for _, r in key.sort_values("p_ds").iterrows():
        print(
            f"  {r.gene_set[:35]:36}{r.target:>24}{r.z_ds:>9.2f}{r.p_ds:>9.4f}"
            f"{r.z_ds_moran:>10.2f}{r.p_ds_moran:>10.4f}"
        )
    print(
        f"\n  clearing p<0.05:  stability-only {int((matched.p_ds < 0.05).sum())}"
        f"   +autocorrelation {int((matched.p_ds_moran < 0.05).sum())}"
        f"   (of {len(matched)})"
    )
    print(f"\n  -> {out}/x3_autocorr_matched_{parc}.csv\n{'=' * 92}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
