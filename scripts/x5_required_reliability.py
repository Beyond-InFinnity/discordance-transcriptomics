#!/usr/bin/env python
"""What would a study need to measure, for this question to be answerable?

§4 claims this study's most useful output is not the null but the specification
that comes with it: "A null accompanied by the reliability at which the test
becomes possible is a specification for the next experiment." That sentence has
been standing without the number behind it. This computes it.

The arithmetic
--------------
A correlation between two imperfectly measured maps is attenuated by
sqrt(r_brain * r_genes), so the smallest resolvable true effect is

    floor = spin_threshold / sqrt(r_brain * r_genes)

Setting floor equal to the effect you want to detect and solving for one side:

    r_needed = spin_threshold^2 / (rho_true^2 * r_other)

Values above 1 mean the effect is unreachable by improving that side alone,
however good the measurement gets — the other side has to move too. That case is
reported rather than clipped, because "impossible from here" is the more useful
answer.

What this is not
----------------
Not a power analysis in the sampling sense. It says nothing about how many
subjects produce a given reliability; that mapping depends on the measurement
and is out of scope. It answers the prior question — what reliability would have
to be reached — which is the one a reader can act on without re-deriving our
attenuation algebra.

The two anchors
---------------
Reported for the study's actual gene panels and for a hypothetical set at 0.55,
which is roughly the best-measured panel here (pericyte/mural, 0.557). The
brain-side anchor is baseline OEF at 0.978, already near ceiling — which is why
the gene side is where the requirement bites.

Usage
-----
    python scripts/x5_required_reliability.py
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

logger = logging.getLogger("x5_required_reliability")

# Effects worth designing for. 0.3 is an ordinary cross-modal spatial
# correlation; above ~0.5 is rare between independent modalities, so a design
# that needs 0.7 to work is not a design.
TARGET_EFFECTS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.50)


def required(threshold: float, rho_true: float, r_other: float) -> float:
    """Reliability one side needs for ``rho_true`` to become resolvable."""
    if rho_true <= 0 or r_other <= 0:
        return math.inf
    return threshold**2 / (rho_true**2 * r_other)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--parcellation", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.logging.level, format=cfg.logging.format)
    parc = args.parcellation or cfg.parcellation.primary.name

    floors = pd.read_csv("results/p0c_detectability_floor.csv")
    panel = pd.read_csv("results/p0c_geneset_reliability.csv")
    dyn = pd.read_csv(f"results/p0_dynamic_range_{parc}.csv").set_index("name")

    # Recover the spin threshold as ceiling x floor and assert it is constant,
    # the same check Phase 0d makes. If it is not, two upstream files disagree
    # about the parcellation and nothing here is comparable.
    ok = floors[floors.attenuation_ceiling > 0]
    implied = (ok.attenuation_ceiling * ok.detectable_true_rho).round(6)
    if implied.nunique() != 1:
        raise ValueError(f"spin threshold not constant: {sorted(implied.unique())[:5]}")
    spin = float(implied.iloc[0])
    logger.info("spin threshold %.6f", spin)

    r_oef = float(dyn.loc["baseline OEF", "split_half_reliability"])
    r_disc = float(dyn.loc["discordance (extraction)", "split_half_reliability"])

    # --- the curve: gene-side requirement against a near-ceiling brain map ---
    rows = []
    for e in TARGET_EFFECTS:
        for bname, rb in (
            ("baseline OEF", r_oef),
            ("discordance (extraction)", r_disc),
        ):
            need = required(spin, e, rb)
            rows.append(
                {
                    "target_effect": e,
                    "brain_map": bname,
                    "reliability_brain": round(rb, 4),
                    "gene_reliability_needed": round(need, 4),
                    "reachable": need <= 1.0,
                }
            )
    curve = pd.DataFrame(rows)

    # --- what each real panel would need on the brain side, and vice versa ---
    per_set = []
    for r in panel.itertuples():
        rg = float(r.reliability_panel)
        best = float(r.reliability_panel_best)
        per_set.append(
            {
                "gene_set": r.gene_set,
                "reliability_panel": round(rg, 4),
                "reliability_panel_best": round(best, 4),
                "floor_now": round(spin / math.sqrt(r_oef * rg), 4)
                if rg > 0
                else math.inf,
                "gene_reliability_needed_for_0.30": round(required(spin, 0.30, r_oef), 4),
                "shortfall": round(required(spin, 0.30, r_oef) - rg, 4),
                "closed_by_best_construction": best >= required(spin, 0.30, r_oef),
            }
        )
    sets = pd.DataFrame(per_set).sort_values("shortfall")

    out = Path("results")
    c_csv = out / f"x5_required_reliability_curve_{parc}.csv"
    s_csv = out / f"x5_required_reliability_by_set_{parc}.csv"
    need30 = required(spin, 0.30, r_oef)

    with manifest(f"x5_required_reliability_{parc}", cfg) as man:
        curve.to_csv(c_csv, index=False)
        sets.to_csv(s_csv, index=False)
        man.record(
            outputs=[str(c_csv), str(s_csv)],
            spin_threshold=round(spin, 6),
            reliability_baseline_oef=round(r_oef, 4),
            gene_reliability_needed_for_0_30=round(need30, 4),
            n_sets_closed_by_best_construction=int(
                sets.closed_by_best_construction.sum()
            ),
            n_sets=len(sets),
        )
        man.note(
            "Inverts the detectability floor: the reliability a side would have "
            "to reach for a given true effect to become resolvable. Says nothing "
            "about sample size, which depends on the measurement; it answers the "
            "prior question a reader can act on."
        )

    w = 78
    print(f"\n{'=' * w}\nWHAT WOULD THE NEXT STUDY NEED? — {parc}\n{'=' * w}")
    print(f"  spin threshold {spin:.3f}; brain side held at its measured value\n")
    print(f"  {'true |rho|':>12}{'vs baseline OEF':>20}{'vs extraction':>18}")
    for e in TARGET_EFFECTS:
        a = curve[(curve.target_effect == e) & (curve.brain_map == "baseline OEF")]
        b = curve[(curve.target_effect == e) & (curve.brain_map != "baseline OEF")]
        fa, fb = a.gene_reliability_needed.iloc[0], b.gene_reliability_needed.iloc[0]
        sa = f"{fa:.2f}" + ("" if fa <= 1 else "  X")
        sb = f"{fb:.2f}" + ("" if fb <= 1 else "  X")
        print(f"  {e:>12.2f}{sa:>20}{sb:>18}")
    print("\n  X = unreachable by improving the gene side alone")
    print(
        f"\n  To resolve a true rho of 0.30 against baseline OEF, a gene panel needs\n"
        f"  reliability {need30:.2f}. The best panel measured here is "
        f"{sets.reliability_panel.max():.3f}."
    )
    closed = sets[sets.closed_by_best_construction]
    print(
        f"  Re-scoring under the best-measuring construction closes the gap for "
        f"{len(closed)} of {len(sets)} sets."
    )
    if len(closed):
        for r in closed.itertuples():
            print(
                f"    {r.gene_set[:44]:46} {r.reliability_panel:.3f} -> "
                f"{r.reliability_panel_best:.3f}"
            )
    print(f"\n  -> {c_csv}\n  -> {s_csv}\n{'=' * w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
