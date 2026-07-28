#!/usr/bin/env python
"""Phase 4 — gene-set association across the multiverse, with both nulls.

Every reported effect carries four numbers, because each answers a different
objection:

**Spatial null (spin test).** Two smooth brain maps correlate by chance. This
asks whether the association beats rotated versions of the target map.

**Competitive null.** A large, stable gene set out-correlates a small unstable
one regardless of biology. This asks whether the set beats *random gene sets
matched on size and differential stability*. A result that clears the spin test
but not this one is a statement about gene-set size, not about the genes.

**BH-FDR across the family.** Many sets are tested at once.

**Multiverse distribution.** Markello et al. showed AHBA processing choices can
move a correlation by up to rho = 1.0. A point estimate from one pipeline is not
evidence, so every effect is reported as median, interquartile range, and the
share of pipelines agreeing on sign. Under 80% consistent sign counts as
unstable.

Gene sets are the frozen list, committed before any result was seen. Adding a
set now would be the exact failure the freeze exists to prevent, so anything not
on the list belongs in a separately labelled exploratory arm.

Usage
-----
    python scripts/p4_genesets.py --n-draws 10000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.targets import (
    discordance_modes,
    load_coupling_components,
    load_target_map,
)
from src.stats.competitive import competitive_null, differential_stability
from src.stats.spatial import corr_with_null, fdr_bh
from src.utils.config import REPO_ROOT, load_config
from src.utils.manifest import manifest

logger = logging.getLogger("p4_genesets")

MSIGDB = REPO_ROOT / "data/raw/genesets/msigdb_sets.json"
MACAQUE = REPO_ROOT / "data/derived/macaque/macaque_vascular_parcels.npy"


def load_genesets() -> dict[str, dict]:
    """The frozen sets: curated small ones plus the pinned MSigDB collections."""
    cfg_sets = yaml.safe_load((REPO_ROOT / "config/genesets.yaml").read_text())
    out: dict[str, dict] = {}
    for name, spec in cfg_sets["curated"].items():
        if spec.get("genes"):
            out[name] = {"genes": spec["genes"], "direction": spec.get("direction_h1")}

    if MSIGDB.exists():
        want = {s["name"]: s["direction_h1"] for s in cfg_sets["msigdb"]}
        raw = json.loads(MSIGDB.read_text())
        for key, genes in raw.items():
            # Map the fetched library key back onto the frozen set names.
            k = key.upper().replace(" ", "_")
            match = next(
                (w for w in want if w.replace("HALLMARK_", "").replace("GOBP_", "") in k),
                None,
            )
            if match:
                out[match] = {"genes": genes, "direction": want[match]}
    else:
        logger.warning("MSigDB sets not found at %s — curated sets only", MSIGDB)
    return out


def targets_for(cfg, parc: str) -> dict[str, np.ndarray]:
    d_cbf, d_cmro2, _s, _p = load_coupling_components(parc, masked=True)
    modes = discordance_modes(d_cbf, d_cmro2)
    oef, _ = load_target_map(cfg, "baseline_oef", parc, masked=True)
    ang, _ = load_target_map(cfg, "coupling_n", parc, masked=True)
    out = {
        "discordance_extraction": modes.extraction,
        "discordance_overshoot": modes.overshoot,
        "coupling_angle": ang,
        "baseline_oef": oef,
    }
    if MACAQUE.exists():
        mac = np.load(MACAQUE)
        out["macaque_vascular_CONTROL"] = mac[mac.shape[0] // 2]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--n-draws", type=int, default=10_000)
    ap.add_argument("--max-cells", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.logging.level, format=cfg.logging.format)
    parc = cfg.parcellation.primary.name

    mv_dir = cfg.path("expression") / "multiverse"
    idx_path = mv_dir / "multiverse_index.csv"
    if not idx_path.exists():
        raise FileNotFoundError(
            f"{idx_path} missing — run scripts/p3_multiverse.py first"
        )
    idx = pd.read_csv(idx_path)
    idx = idx[idx.status.isin(["ok", "cached"])]
    if args.max_cells:
        idx = idx.head(args.max_cells)
    logger.info("multiverse cells available: %d", len(idx))

    gsets = load_genesets()
    logger.info("frozen gene sets: %d — %s", len(gsets), ", ".join(gsets))
    targets = targets_for(cfg, parc)
    nulls = np.load(cfg.path("nulls") / f"baseline_oef_{parc}_masked_nulls.npy")

    # Differential stability from the primary pipeline's per-donor matrices.
    # Computed once: it is a property of the atlas and donors, not of the cell.
    stab_path = mv_dir / "differential_stability.csv"
    if stab_path.exists():
        stability = pd.read_csv(stab_path, index_col=0).iloc[:, 0]
    else:
        import abagen

        from src.data.parcellate import gifti_atlas_paths

        atlas = gifti_atlas_paths(parc, cfg.parcellation.primary.density)
        per_donor = [
            abagen.get_expression_data(
                atlas, donors=[d], probe_selection="max_intensity", verbose=0
            )
            for d in ["9861", "10021", "12876", "14380", "15697"]
        ]
        stability = differential_stability(per_donor)
        stability.to_frame().to_csv(stab_path)
    logger.info("differential stability for %d genes", len(stability))

    rows = []
    for n_cell, (_, cell) in enumerate(idx.iterrows(), 1):
        exp_all = pd.read_parquet(cell["path"])
        exp = exp_all.iloc[:100]  # left hemisphere
        for thr in (0.0, 0.1, 0.2):
            keep = stability[stability >= thr].index if thr > 0 else exp.columns
            sub = exp[[c for c in exp.columns if c in set(keep)]]
            if sub.shape[1] < 100:
                continue
            for gname, gspec in gsets.items():
                present = [g for g in gspec["genes"] if g in sub.columns]
                if len(present) < 3:
                    continue
                z = (sub[present] - sub[present].mean()) / sub[present].std()
                score = z.mean(axis=1).to_numpy()
                for tname, y in targets.items():
                    ok = np.isfinite(score) & np.isfinite(y)
                    sp = corr_with_null(
                        score[ok], y[ok], nulls=nulls[ok, :], method=cfg.stats.correlation
                    )
                    rows.append(
                        {
                            "cell": cell["hash"],
                            "stability_threshold": thr,
                            "gene_set": gname,
                            "n_genes": len(present),
                            "target": tname,
                            "rho": sp.rho,
                            "p_spin": sp.p_spin,
                            "direction_h1": gspec["direction"],
                            "probe_selection": cell["probe_selection"],
                            "lr_mirror": cell["lr_mirror"],
                            "missing": cell["missing"],
                        }
                    )
        if n_cell % 10 == 0:
            logger.info("  %d/%d cells", n_cell, len(idx))

    df = pd.DataFrame(rows)
    df["p_fdr"] = np.nan
    for _, g in df.groupby(["cell", "stability_threshold", "target"]).groups.items():
        df.loc[g, "p_fdr"] = fdr_bh(df.loc[g, "p_spin"].to_numpy())

    # Competitive null on the PRIMARY pipeline only — 10,000 matched draws per
    # set is expensive, and the multiverse already covers processing variance.
    primary = idx[
        (idx.probe_selection == "diff_stability")
        & (idx.lr_mirror == "bidirectional")
        & (idx.missing == "centroids")
    ]
    comp_rows = []
    if len(primary):
        exp = pd.read_parquet(primary.iloc[0]["path"]).iloc[:100]
        for gname, gspec in gsets.items():
            for tname, y in targets.items():
                try:
                    c = competitive_null(
                        exp,
                        y,
                        gspec["genes"],
                        stability,
                        name=gname,
                        n_draws=args.n_draws,
                        seed=cfg.seed,
                    )
                except ValueError as exc:
                    logger.warning("%s vs %s: %s", gname, tname, exc)
                    continue
                comp_rows.append({**c.as_dict(), "target": tname})
    comp = pd.DataFrame(comp_rows)

    # Multiverse summary per (gene set, target).
    summ = (
        df.groupby(["gene_set", "target"])
        .agg(
            rho_median=("rho", "median"),
            rho_q1=("rho", lambda x: x.quantile(0.25)),
            rho_q3=("rho", lambda x: x.quantile(0.75)),
            pct_consistent_sign=("rho", lambda x: max((x > 0).mean(), (x < 0).mean())),
            pct_spin_sig=("p_spin", lambda x: (x < cfg.stats.alpha).mean()),
            n_cells=("rho", "size"),
        )
        .reset_index()
    )
    summ["stable_sign"] = summ.pct_consistent_sign >= 0.80
    if len(comp):
        summ = summ.merge(
            comp[["name", "target", "p_competitive", "z_competitive"]],
            left_on=["gene_set", "target"],
            right_on=["name", "target"],
            how="left",
        ).drop(columns=["name"])

    out = Path("results")
    with manifest("p4_genesets", cfg) as man:
        df.to_csv(out / "p4_genesets_full.csv", index=False)
        summ.to_csv(out / "p4_genesets_summary.csv", index=False)
        if len(comp):
            comp.to_csv(out / "p4_competitive_nulls.csv", index=False)
        man.record(
            n_cells=len(idx),
            n_gene_sets=len(gsets),
            n_tests=len(df),
            n_competitive_draws=args.n_draws,
            n_survive_both=int(
                (
                    (summ.get("p_competitive", 1) < cfg.stats.alpha)
                    & (summ.pct_spin_sig > 0.5)
                    & summ.stable_sign
                ).sum()
            ),
        )
        man.note(
            "Frozen gene sets only. An effect counts only if it clears the "
            "spatial null in most pipelines, clears the competitive null, and "
            "holds its sign in at least 80% of the multiverse."
        )

    pd.set_option("display.width", 240)
    print(
        f"\n{'=' * 108}\nPHASE 4 — {len(gsets)} frozen sets x {len(idx)} pipelines\n{'=' * 108}"
    )
    cols = [
        "gene_set",
        "target",
        "rho_median",
        "rho_q1",
        "rho_q3",
        "pct_consistent_sign",
        "pct_spin_sig",
        "n_cells",
    ]
    if "p_competitive" in summ:
        cols.append("p_competitive")
    key = summ[summ.target != "macaque_vascular_CONTROL"].sort_values(
        "rho_median", key=abs, ascending=False
    )
    print(key[cols].head(20).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if "macaque_vascular_CONTROL" in summ.target.values:
        print("\nPOSITIVE CONTROL (macaque vascular density):")
        print(
            summ[summ.target == "macaque_vascular_CONTROL"][cols].to_string(
                index=False, float_format=lambda x: f"{x:.3f}"
            )
        )
    print(f"{'=' * 108}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
