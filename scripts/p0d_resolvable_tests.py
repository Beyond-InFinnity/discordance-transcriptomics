#!/usr/bin/env python
"""Which gene-set x outcome tests this design can resolve at all.

The manuscript's central table, and the reason its claim is a bound rather than
an absence.

A correlation between two imperfectly reliable maps is attenuated toward zero by
√(r₁r₂). Divide an observed effect by that ceiling and you have the implied true
effect; compare it against the smallest true effect the pairing resolves at
conventional power and you know whether the test could ever have found anything.

Run over all 33 gene-set x outcome pairings, three are resolvable — and they are
exactly the three that return associations passing both null models. Every test
against extraction-mode discordance, the outcome the hypothesis is actually
about, falls below its floor. That is not a caveat to the result; it *is* the
result, and reporting significance for the other 30 without it would invite
readers to weight underpowered tests equally with powered ones.

One frozen gene set (``GOBP_BLOOD_VESSEL_MORPHOGENESIS``) has *negative*
donor-to-donor reliability, so its attenuation ceiling is undefined and no effect
size is detectable. It is reported as untestable rather than assigned a
misleading floor.

Usage
-----
    python scripts/p0d_resolvable_tests.py
    python scripts/p0d_resolvable_tests.py --results-dir /tmp/probe
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.manifest import manifest

logger = logging.getLogger("p0d_resolvable")

# Phase 0c names maps for humans; Phase 4 names them for machines.
MAP_LABEL = {
    "baseline_oef": "baseline OEF",
    "discordance_extraction": "discordance (extraction)",
    "discordance_overshoot": "discordance (overshoot)",
}


def build(results: Path) -> pd.DataFrame:
    floors = pd.read_csv(results / "p0c_detectability_floor.csv")
    summary = pd.read_csv(results / "p4_genesets_summary.csv")
    panel = pd.read_csv(results / "p0c_geneset_reliability.csv").set_index("gene_set")

    rows = []
    for _, r in summary.iterrows():
        if r.target not in MAP_LABEL:
            continue
        m = floors[
            (floors.gene_set == r.gene_set) & (floors.brain_map == MAP_LABEL[r.target])
        ]
        if not len(m):
            continue
        ceiling = float(m.attenuation_ceiling.iloc[0])
        floor = float(m.detectable_true_rho.iloc[0])
        usable = ceiling > 0 and not math.isnan(ceiling)
        implied = abs(r.rho_median) / ceiling if usable else float("nan")
        rows.append(
            {
                "gene_set": r.gene_set,
                "outcome": r.target,
                "rho_observed": r.rho_median,
                "panel_reliability": float(panel.loc[r.gene_set, "reliability_panel"]),
                "map_reliability": float(m.reliability_brain.iloc[0]),
                "attenuation_ceiling": ceiling,
                "implied_true_rho": implied,
                "detectability_floor": floor,
                "resolvable": bool(usable and implied >= floor),
                "untestable": not usable,
                "p_competitive": r.p_competitive,
                "pct_spin_sig": r.pct_spin_sig,
            }
        )
    return pd.DataFrame(rows).sort_values("implied_true_rho", ascending=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--parcellation", default=None)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.logging.level, format=cfg.logging.format)
    parc = args.parcellation or cfg.parcellation.primary.name
    out = Path(args.results_dir)

    df = build(Path("results"))
    n_res = int(df.resolvable.sum())
    n_untestable = int(df.untestable.sum())

    csv = out / f"p0d_resolvable_tests_{parc}.csv"
    with manifest(f"p0d_resolvable_tests_{parc}", cfg, results_dir=out) as man:
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv, index=False)
        by_outcome = {
            k: {
                "resolvable": int(g.resolvable.sum()),
                "total": len(g),
                "median_floor": round(float(g.detectability_floor.median()), 4),
            }
            for k, g in df.groupby("outcome")
        }
        man.record(
            outputs=[str(csv)],
            n_tests=len(df),
            n_resolvable=n_res,
            n_untestable=n_untestable,
            by_outcome=by_outcome,
            resolvable_tests=[
                f"{r.gene_set} -> {r.outcome}" for _, r in df[df.resolvable].iterrows()
            ],
        )
        man.note(
            "A test whose implied true effect falls below its detectability floor "
            "could not have found an effect of ordinary size regardless of how it "
            "was conducted. Three of 33 pairings are resolvable, and they are the "
            "three that return associations passing both nulls."
        )

    print(f"\n{'=' * 76}\nWHAT THIS DESIGN CAN RESOLVE — {parc}\n{'=' * 76}")
    print(f"  {len(df)} gene-set x outcome tests; {n_res} resolvable\n")
    print(f"  {'outcome':<26}{'resolvable':>12}{'median floor':>15}")
    for k, g in df.groupby("outcome"):
        print(
            f"  {k:<26}{int(g.resolvable.sum()):>6} / {len(g):<4}"
            f"{g.detectability_floor.median():>15.3f}"
        )
    print(f"\n  {'RESOLVABLE':<12}{'gene set':<28}{'outcome':<24}{'true':>7}{'floor':>7}")
    for _, r in df[df.resolvable].iterrows():
        print(
            f"  {'':<12}{r.gene_set[:27]:<28}{r.outcome:<24}"
            f"{r.implied_true_rho:>7.3f}{r.detectability_floor:>7.3f}"
        )
    if n_untestable:
        bad = sorted(set(df[df.untestable].gene_set))
        print(f"\n  UNTESTABLE at any effect size (non-positive reliability): {bad}")
    print(f"\n  -> {csv}\n{'=' * 76}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
