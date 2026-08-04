#!/usr/bin/env python
"""What effect size each gene-set x outcome test could have detected.

The manuscript's central table, and the reason its negative is a bound rather
than an absence.

A correlation between two imperfectly reliable maps is attenuated toward zero by
√(r₁r₂). Divide the spin-test threshold by that ceiling and you have the
smallest *true* effect the pairing could ever resolve — the **detectability
floor**. A test whose floor sits above any biologically plausible effect could
not have found one regardless of how carefully it was run.

WHAT THIS FILE USED TO DO, AND WHY IT WAS WRONG
-----------------------------------------------
An earlier version reported a boolean ``resolvable``, defined as

    implied_true_rho >= detectability_floor
    where  implied_true_rho = |rho_observed| / ceiling
    and    detectability_floor = spin_threshold / ceiling

The ceiling appears in both and cancels exactly. The criterion therefore reduced
to ``|rho_observed| >= spin_threshold`` — a restatement of the significance test,
wearing the costume of a power analysis. It was verified numerically: across all
30 testable pairings ``ceiling x floor`` was 0.244889 to six decimals, invariant.

That made the headline claim — "three tests are resolvable, and they are exactly
the three that pass both null models" — a tautology. It could not have come out
any other way, and the reliability correction did no work at all.

The fix is to report the floor **on its own**. The floor depends only on the two
reliabilities and the parcellation; it does not depend on what was observed, so
it is a genuine statement about what the design could detect. Observed effects
are still carried in the table for reference, but nothing is derived from them.

WHAT IS NEW
-----------
1. Each test is attributed to its **binding side** — is the ceiling limited by the
   brain map or by the gene map? These have completely different remedies (more
   subjects vs. a different gene-set construct) and were previously conflated.
2. Gene-set size is reported, because it predicts gene-map reliability: averaging
   k genes into one score cancels signal whenever the genes' spatial patterns
   resemble each other less than their measurement noise does, which is the usual
   case above k ~ 50.
3. Sets that were **declared frozen but never actually ran** are recorded. The
   loader in ``p4_genesets.py`` silently skips any curated entry without an
   explicit ``genes:`` list, so a pre-registered set can vanish without warning.

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
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from p4_genesets import load_genesets

from src.utils.config import REPO_ROOT, load_config
from src.utils.manifest import manifest

logger = logging.getLogger("p0d_resolvable")

# Phase 0c names maps for humans; Phase 4 names them for machines.
#
# The coupling angle belongs here even though earlier versions of this table
# omitted it. §7.3 of the protocol names the coupling ratio the PRIMARY outcome
# — "continuous is statistically stronger than binary" — and the angle is the
# reparameterisation that does not blow up as the denominator approaches zero.
# It is already a target in Phase 4 and Phase 4c and already has floors computed
# in Phase 0c; its absence from the central table left the pre-registered primary
# outcome as the one outcome the design's own detectability analysis never
# covered. It is also the most reliable of the three coupling-derived maps
# (0.711 against 0.579 and 0.595), so excluding it understated what the design
# could reach.
MAP_LABEL = {
    "baseline_oef": "baseline OEF",
    "coupling_angle": "coupling angle",
    "discordance_extraction": "discordance (extraction)",
    "discordance_overshoot": "discordance (overshoot)",
}

# Reading aid only. These are labels for a continuous number, NOT a test, and
# nothing downstream branches on them. Spatial correlations between independent
# modalities above ~0.5 are rare and above ~0.7 essentially unobserved, which is
# where the upper cuts come from.
BANDS = [
    (0.30, "resolves modest effects"),
    (0.50, "moderate-to-large only"),
    (0.70, "large effects only"),
    (math.inf, "implausible — effectively untestable"),
]


def band(floor: float) -> str:
    if not math.isfinite(floor):
        return "UNTESTABLE at any effect size"
    for edge, label in BANDS:
        if floor < edge:
            return label
    return BANDS[-1][1]


def declared_but_never_ran() -> list[str]:
    """Curated sets present in the frozen config but absent from every analysis.

    ``load_genesets`` keeps only entries with an explicit ``genes:`` list. A set
    specified another way — ``mitochondrial_density_proxy`` is declared by HGNC
    family prefixes — is dropped with no error, so it never appears in a result
    and never appears in a failure either.
    """
    cfg = yaml.safe_load((REPO_ROOT / "config/genesets.yaml").read_text())
    declared = set(cfg.get("curated", {}) or {})
    return sorted(declared - set(load_genesets()))


def build(results: Path) -> tuple[pd.DataFrame, float]:
    floors = pd.read_csv(results / "p0c_detectability_floor.csv")
    summary = pd.read_csv(results / "p4_genesets_summary.csv")
    panel = pd.read_csv(results / "p0c_geneset_reliability.csv").set_index("gene_set")
    sizes = {k: len(v["genes"]) for k, v in load_genesets().items()}

    # The spin threshold is implicit in Phase 0c's output as ceiling x floor.
    # Recover it once and assert it is constant: if it is not, the two files
    # disagree about the parcellation and nothing below is comparable.
    usable = floors[floors.attenuation_ceiling > 0]
    implied = (usable.attenuation_ceiling * usable.detectable_true_rho).round(6)
    if implied.nunique() != 1:
        raise ValueError(
            f"spin threshold is not constant across Phase 0c rows: {sorted(implied.unique())[:5]}"
        )
    spin = float(implied.iloc[0])
    logger.info("spin-test threshold recovered from Phase 0c: |rho| >= %.4f", spin)

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
        rel_brain = float(m.reliability_brain.iloc[0])
        rel_genes = float(panel.loc[r.gene_set, "reliability_panel"])
        ok = ceiling > 0 and not math.isnan(ceiling)

        floor = spin / ceiling if ok else math.inf
        # Which side is the bottleneck? Floor each side would give on its own,
        # with the other assumed noiseless.
        floor_genes_perfect = spin / math.sqrt(rel_brain) if rel_brain > 0 else math.inf
        floor_brain_perfect = spin / math.sqrt(rel_genes) if rel_genes > 0 else math.inf
        binding = "genes" if floor_brain_perfect >= floor_genes_perfect else "brain map"

        # The same genes and the same brain map under the construction that
        # measures them best -- whole-set averaging, chunks, or per-gene. Chosen
        # in Phase 0c on reliability alone, never on any outcome. The gap
        # between the two floors separates blindness the data imposed from
        # blindness the analysis chose.
        ceil_best = float(m.get("attenuation_ceiling_best", pd.Series([ceiling])).iloc[0])
        floor_best = spin / ceil_best if ceil_best > 0 else math.inf
        best_name = str(m.get("best_construction", pd.Series(["whole_set"])).iloc[0])

        rows.append(
            {
                "gene_set": r.gene_set,
                "n_genes": sizes.get(r.gene_set, pd.NA),
                "outcome": r.target,
                "reliability_genes": rel_genes,
                "reliability_brain": rel_brain,
                "attenuation_ceiling": ceiling if ok else 0.0,
                "detectability_floor": floor,
                "best_construction": best_name,
                "detectability_floor_best": floor_best,
                "floor_recovered": (floor - floor_best)
                if math.isfinite(floor)
                else math.inf,
                "floor_if_genes_were_perfect": floor_genes_perfect,
                "floor_if_brain_were_perfect": floor_brain_perfect,
                "binding_side": binding if ok else "genes",
                "verdict": band(floor),
                "verdict_best": band(floor_best),
                # Carried for reference ONLY. Nothing above is derived from these;
                # that conflation is the defect this rewrite exists to remove.
                "rho_observed": r.rho_median,
                "p_competitive": r.get("p_competitive", float("nan")),
                "pct_spin_sig": r.get("pct_spin_sig", float("nan")),
            }
        )
    df = pd.DataFrame(rows).sort_values("detectability_floor")
    return df, spin


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

    df, spin = build(Path("results"))
    absent = declared_but_never_ran()
    n_untestable = int((~df.detectability_floor.apply(math.isfinite)).sum())
    finite = df[df.detectability_floor.apply(math.isfinite)]

    csv = out / f"p0d_resolvable_tests_{parc}.csv"
    with manifest(f"p0d_resolvable_tests_{parc}", cfg, results_dir=out) as man:
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv, index=False)
        man.record(
            outputs=[str(csv)],
            n_tests=len(df),
            spin_threshold=round(spin, 6),
            n_untestable=n_untestable,
            median_floor=round(float(finite.detectability_floor.median()), 4),
            n_binding_genes=int((df.binding_side == "genes").sum()),
            n_binding_brain=int((df.binding_side == "brain map").sum()),
            by_verdict={k: int(v) for k, v in df.verdict.value_counts().items()},
            by_outcome={
                k: {
                    "n": len(g),
                    "median_floor": round(float(g.detectability_floor.median()), 4)
                    if g.detectability_floor.apply(math.isfinite).any()
                    else None,
                }
                for k, g in finite.groupby("outcome")
            },
            declared_but_never_ran=absent,
        )
        man.note(
            "Reports the detectability floor -- the smallest TRUE effect each "
            "pairing could resolve -- which depends only on the two reliabilities "
            "and the parcellation. It does not depend on what was observed. The "
            "previous `resolvable` flag did, and cancelled algebraically to "
            "|rho| >= spin threshold, making it a restatement of the significance "
            "test rather than a power analysis."
        )

    print(f"\n{'=' * 84}\nWHAT THIS DESIGN COULD DETECT — {parc}\n{'=' * 84}")
    print(f"  spin-test threshold |rho| >= {spin:.3f}; floor = threshold / ceiling\n")
    print(f"  {'outcome':26}{'n':>4}{'median floor':>14}{'binding side':>28}")
    for k, g in finite.groupby("outcome"):
        binds = g.binding_side.value_counts()
        desc = ", ".join(f"{v} {kk}" for kk, v in binds.items())
        print(f"  {k:26}{len(g):>4}{g.detectability_floor.median():>14.3f}{desc:>28}")

    print(f"\n  {'gene set':36}{'k':>5}{'outcome':>24}{'floor':>8}  verdict")
    for _, r in df.iterrows():
        f = (
            "  inf"
            if not math.isfinite(r.detectability_floor)
            else f"{r.detectability_floor:5.3f}"
        )
        print(f"  {r.gene_set[:35]:36}{r.n_genes!s:>5}{r.outcome:>24}{f:>8}  {r.verdict}")

    print(f"\n  {n_untestable} of {len(df)} pairings are untestable at any effect size.")
    print(
        f"  Bottleneck: {int((df.binding_side == 'genes').sum())} gene-side, "
        f"{int((df.binding_side == 'brain map').sum())} brain-map-side."
    )

    # How much of the blindness was self-inflicted?
    fb = df[df.detectability_floor_best.apply(math.isfinite)]
    improved = df[df.floor_recovered > 0.01]
    print(
        f"\n  Under the best-measuring construction per set (Phase 0c, chosen on\n"
        f"  reliability alone): median floor {fb.detectability_floor_best.median():.3f} "
        f"vs {finite.detectability_floor.median():.3f} as run."
    )
    print(
        f"  {len(improved)} of {len(df)} pairings improve; "
        f"{int((~df.detectability_floor_best.apply(math.isfinite)).sum())} remain untestable."
    )
    for name, g in df[df.floor_recovered > 0.05].groupby("gene_set"):
        print(
            f"    {name[:38]:40} {g.best_construction.iloc[0]:>12}  "
            f"floor {g.detectability_floor.median():.2f} -> {g.detectability_floor_best.median():.2f}"
        )
    if absent:
        print(f"\n  DECLARED FROZEN BUT NEVER RAN: {absent}")
        print("    p4_genesets.load_genesets() keeps only curated entries with an")
        print("    explicit `genes:` list and skips the rest silently.")
    print(f"\n  -> {csv}\n{'=' * 84}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
