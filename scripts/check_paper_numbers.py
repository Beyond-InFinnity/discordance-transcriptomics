#!/usr/bin/env python
"""Verify every number in the paper against the artifact it came from. ⛔ GATE.

This project's recurring failure is not bad statistics, it is **drift**: a
number gets into prose, the pipeline is rerun, the number changes, and the prose
does not. An independent review already caught a mislabelled gene set, a stale
README, and a summary contradicting its own manifest. Each was a hand-maintained
statement that no longer matched its artifact.

``audit_provenance.py`` proves ``results/`` came from one code state. It cannot
tell whether the paper describes those results. This closes that gap: each claim
below names the artifact it comes from and how to recompute it, and the check
fails if the recomputed value is not present in the draft.

That makes the draft a *derived* document. If a rerun moves a number, this says
so, names the claim, and prints the value the paper should have instead.

Usage
-----
    python scripts/check_paper_numbers.py
    python scripts/check_paper_numbers.py --draft paper/draft.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"

# U+2212, the typographic minus the draft uses for signed effects.
MINUS = "\u2212"

# Phrases that must NOT appear in the draft, because they assert a claim the
# analysis no longer supports.
#
# Presence checks are substring searches, which are fine for a distinctive
# number and useless for a bare word: a claim of "3 untestable tests" expecting
# "three" matches a sentence about "three resolvable tests" and reports ok. A
# positive check cannot catch a framing that is wrong; only a negative one can.
#
# The `resolvable` criterion was |rho|/ceiling >= spin/ceiling, which cancels to
# |rho| >= spin -- the significance test wearing the costume of a power analysis.
# Its headline, "the resolvable tests are exactly the ones passing both nulls",
# was therefore a tautology and could not have come out otherwise. See
# scripts/p0d_resolvable_tests.py.
FORBIDDEN = [
    "are exactly the three that return associations",
    "those three are exactly the three returning associations",
    "Three of 33 tests are resolvable",
    "call a test *resolvable*",
    "resolvable tests |",
    "| implied true |",
    # Withdrawn in favour of x4: reading the genome-wide clearance rate as a
    # false-positive rate assumes no gene is truly associated with the target.
    # The rate against independently rotated genes is 3.8-4.3%, so the spread in
    # the raw rates is association, not error.
    "under what should be the null",
    "making the test conservative; a noisy target produces rotations",
    "roughly **one gene in eight**",
]


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(R / name)


def _man(name: str) -> dict:
    with (R / f"{name}.manifest.json").open() as fh:
        return json.load(fh)["results"]


def _p4(gene_set: str, target: str, col: str) -> float:
    d = _csv("p4_genesets_summary.csv")
    return float(d[(d.gene_set == gene_set) & (d.target == target)][col].iloc[0])


def _p5(reference: str, target: str, col: str, tag: str = "") -> float:
    d = _csv(f"p5_hierarchy_schaefer200x7{tag}.csv")
    g = d[
        (d.step == "geneset_partial") & (d.reference == reference) & (d.target == target)
    ]
    return float(g[col].median())


def _reliability(name: str) -> float:
    d = _csv("p0_dynamic_range_schaefer200x7.csv")
    return float(d[d.name == name].split_half_reliability.iloc[0])


def _controls(source: str, a: str, b: str) -> float:
    d = _csv("p5_positive_controls_schaefer200x7.csv")
    m = d[(d.source == source) & (d.a.str.endswith(a)) & (d.b.str.endswith(b))]
    return float(m.rho.iloc[0])


def _calib(target: str, col: str = "pct_p_lt_05") -> float:
    """Genome-wide spin-test calibration, per target (Phase 4c).

    Quoted in §2.5. It is the number a reader needs to interpret any per-gene
    spatial p-value, so it is checked like any other claim rather than left as
    prose.
    """
    d = _csv("p4c_pergene_calibration_schaefer200x7.csv")
    return float(d[d.target == target][col].iloc[0])


def _x3(gene_set: str, target: str, col: str) -> float:
    """Autocorrelation-matched competitive null (x3, §3.4.1).

    Raises FileNotFoundError until x3 has run inside a regeneration; the caller
    below turns that into a "pending" notice rather than a dead gate, so claims
    for a newly-added analysis can be written before its artifact exists and
    start verifying automatically once it does.
    """
    d = _csv("x3_autocorr_matched_schaefer200x7.csv")
    m = d[(d.gene_set == gene_set) & (d.target == target)]
    return float(m[col].iloc[0])


def _x4(target: str, col: str) -> float:
    """Null-gene calibration (x4, §2.5).

    Raises FileNotFoundError until x4 has run inside a regeneration; the caller
    turns that into a "pending" notice rather than a dead gate.
    """
    d = _csv("x4_null_genes_schaefer200x7.csv").set_index("target")
    return float(d.loc[target, col])


def _p4c(gene_set: str, target: str, col: str) -> float:
    """Per-gene arm, set-level competitive result (§3.4.1)."""
    d = _csv("p4c_pergene_summary_schaefer200x7.csv")
    m = d[(d.gene_set == gene_set) & (d.target == target)]
    return float(m[col].iloc[0])


def _macaque(target: str) -> float:
    d = _csv("x1_macaque_vascular.csv")
    return float(d[d.target == target].rho.iloc[0])


# (label, computed value, format spec). The format is how the number must appear
# in the draft; a claim whose formatted value is absent is a drift failure.


def _resolvability() -> dict:
    """Recompute §3.2: which gene-set x outcome tests the design can resolve.

    The paper's central table: a detectability floor per pairing, plus which side
    of the attenuation product binds it. Read from Phase 0d rather than
    recomputed -- see the comment below for why recomputation was the problem.
    """
    import math

    # Read Phase 0d rather than recomputing. This function used to derive
    # `resolvable` as |rho|/ceiling >= spin/ceiling -- an identity that cancels
    # to |rho| >= spin, i.e. the significance test. It was the third independent
    # copy of that formula in the repo, which is why fixing p0d alone would not
    # have fixed the paper's numbers.
    d = _csv(sorted(R.glob("p0d_resolvable_tests_*.csv"))[0].name)
    finite = d[d.detectability_floor.apply(math.isfinite)]
    return {
        "total": len(d),
        "untestable": int((~d.detectability_floor.apply(math.isfinite)).sum()),
        "min_floor": float(finite.detectability_floor.min()),
        "median_floor": float(finite.detectability_floor.median()),
        "n_binding_genes": int((d.binding_side == "genes").sum()),
        "per_outcome": {
            k: round(float(g.detectability_floor.median()), 4)
            for k, g in finite.groupby("outcome")
        },
    }


def build_claims() -> list[tuple[str, float, str]]:
    p6 = _man("p6_mediation")
    p6s = _csv("p6_mediation_summary.csv")
    peri = p6s[
        (p6s.gene_set == "pericyte_mural")
        & (p6s.mediator == "baseline_oef")
        & (p6s.outcome == "discordance_extraction")
        & (p6s.adjusted)
    ].iloc[0]
    chain = _csv("p0b_full_dropout_audit_schaefer200x7.csv")
    p4full = _csv("p4_genesets_full.csv")
    peri4 = p4full[
        (p4full.gene_set == "pericyte_mural") & (p4full.target == "baseline_oef")
    ]

    res = _resolvability()
    rel = _csv("p0c_geneset_reliability.csv").set_index("gene_set")

    # Claims whose artifact may not exist yet, because the analysis was added
    # after the last regeneration. Reported as PENDING rather than failing the
    # gate or, worse, killing it -- build_claims() raising would take every
    # other claim down with it.
    pending: list[tuple[str, float, str]] = []
    try:
        pending = [
            (
                "x3: OXPHOS -> OEF, autocorr-matched p",
                _x3("HALLMARK_OXIDATIVE_PHOSPHORYLATION", "baseline_oef", "p_ds_moran"),
                "0.052",
            ),
            (
                "x3: pericyte -> OEF, autocorr-matched p",
                _x3("pericyte_mural", "baseline_oef", "p_ds_moran"),
                "0.013",
            ),
        ]
    except (FileNotFoundError, IndexError, KeyError):
        print("  PENDING  x3 autocorrelation-matched null: artifact not in results/")
        print("           (analysis added after the last regeneration; its claims")
        print("            in §3.4.1 verify once a regeneration includes x3)\n")

    try:
        pending += [
            ("x4: rotated genes vs baseline OEF", _x4("baseline_oef", "null_pct"), "4.1"),
            (
                "x4: rotated genes vs coupling angle",
                _x4("coupling_angle", "null_pct"),
                "3.9",
            ),
            (
                "x4: rotated genes vs overshoot",
                _x4("discordance_overshoot", "null_pct"),
                "4.1",
            ),
            (
                "x4: rotated genes vs extraction",
                _x4("discordance_extraction", "null_pct"),
                "4.1",
            ),
            (
                "x4: real rate, 12 cells, baseline OEF",
                _x4("baseline_oef", "real_pct"),
                "0.68",
            ),
            (
                "x4: real rate, 12 cells, extraction",
                _x4("discordance_extraction", "real_pct"),
                "12.67",
            ),
            (
                "x4: rotated-gene median |rho|, baseline OEF",
                _x4("baseline_oef", "null_median_abs_rho"),
                "0.112",
            ),
            (
                "x4: real median |rho|, baseline OEF",
                _x4("baseline_oef", "real_median_abs_rho"),
                "0.074",
            ),
        ]
    except (FileNotFoundError, IndexError, KeyError):
        print("  PENDING  x4 null-gene calibration: artifact not in results/")
        print("           (analysis added after the last regeneration; its claims")
        print("            in §2.5 verify once a regeneration includes x4)\n")

    return [
        *pending,
        # --- what the design can detect (§3.2, the paper's central table) ---
        # Floors, not a resolvability count: the count was circular. See
        # p0d_resolvable_tests.py.
        ("total gene-set x outcome tests", float(res["total"]), "44"),
        ("tests untestable at any effect size", float(res["untestable"]), "four"),
        ("smallest detectability floor", res["min_floor"], "0.30"),
        ("median floor vs baseline OEF", res["per_outcome"]["baseline_oef"], "0.387"),
        ("median floor vs coupling angle", res["per_outcome"]["coupling_angle"], "0.473"),
        (
            "median floor vs overshoot",
            res["per_outcome"]["discordance_overshoot"],
            "0.49",
        ),
        (
            "median floor vs extraction",
            res["per_outcome"]["discordance_extraction"],
            "0.52",
        ),
        ("tests limited by the gene side", float(res["n_binding_genes"]), "38"),
        # --- per-gene arm (§3.4.1) -------------------------------------------
        (
            "per-gene: pericyte -> OEF competitive p",
            _p4c("pericyte_mural", "baseline_oef", "p_competitive"),
            "0.006",
        ),
        (
            "per-gene: pericyte mean %spin-sig",
            _p4c("pericyte_mural", "baseline_oef", "mean_pct_spin_sig"),
            "18.54",
        ),
        (
            "per-gene: OXPHOS -> coupling angle z",
            _p4c("HALLMARK_OXIDATIVE_PHOSPHORYLATION", "coupling_angle", "z_competitive"),
            "2.94",
        ),
        (
            "per-gene: glycolytic enzymes -> extraction p",
            _p4c("glycolytic_enzymes", "discordance_extraction", "p_competitive"),
            "0.044",
        ),
        # --- spin-test calibration (§2.5) ------------------------------------
        ("calibration: baseline OEF", _calib("baseline_oef"), "0.82"),
        ("calibration: coupling angle", _calib("coupling_angle"), "2.06"),
        ("calibration: overshoot", _calib("discordance_overshoot"), "7.85"),
        ("calibration: extraction", _calib("discordance_extraction"), "12.09"),
        ("calibration: mean p, baseline OEF", _calib("baseline_oef", "mean_p"), "0.632"),
        (
            "calibration: mean p, extraction",
            _calib("discordance_extraction", "mean_p"),
            "0.412",
        ),
        (
            "GOBP set has negative reliability",
            float(rel.loc["GOBP_BLOOD_VESSEL_MORPHOGENESIS", "reliability_panel"]),
            "-0.011",
        ),
        (
            "pericyte panel reliability",
            float(rel.loc["pericyte_mural", "reliability_panel"]),
            "0.557",
        ),
        # --- gates ---------------------------------------------------------
        ("coupling map reliability", _reliability("coupling angle"), "0.711"),
        ("baseline OEF reliability", _reliability("baseline OEF"), "0.978"),
        ("baseline CMRO2 reliability", _reliability("baseline CMRO2"), "0.984"),
        ("discordance (total) reliability", _reliability("discordance (total)"), "0.491"),
        (
            "discordance (extraction) reliability",
            _reliability("discordance (extraction)"),
            "0.579",
        ),
        (
            "discordance (overshoot) reliability",
            _reliability("discordance (overshoot)"),
            "0.595",
        ),
        ("whole-chain worst link", float(chain.abs_rho.max()), "0.315"),
        ("whole-chain links tested", float(len(chain)), "12"),
        # --- H1 ------------------------------------------------------------
        (
            "pericyte->OEF rho",
            _p4("pericyte_mural", "baseline_oef", "rho_median"),
            "0.391",
        ),
        (
            "pericyte->OEF spin-significant fraction",
            _p4("pericyte_mural", "baseline_oef", "pct_spin_sig") * 100,
            "86",
        ),
        (
            "pericyte->OEF competitive p",
            _p4("pericyte_mural", "baseline_oef", "p_competitive"),
            "0.0004",
        ),
        ("pericyte->OEF min FDR", float(peri4.p_fdr.min()), "0.130"),
        (
            "angiogenesis->OEF rho",
            _p4("HALLMARK_ANGIOGENESIS", "baseline_oef", "rho_median"),
            "0.355",
        ),
        (
            "astrocyte->overshoot rho",
            _p4("astrocyte", "discordance_overshoot", "rho_median"),
            "0.256",
        ),
        # --- positive controls ---------------------------------------------
        (
            "endothelial->macaque rho",
            _p4("endothelial", "macaque_vascular_CONTROL", "rho_median"),
            "0.46",
        ),
        ("our OEF<->CMRO2", _controls("our_reconstruction", "oef", "cmro2"), "0.78"),
        ("our OEF<->CBF", _controls("our_reconstruction", "oef", "cbf"), "0.36"),
        ("our CBF<->CMRO2", _controls("our_reconstruction", "cbf", "cmro2"), "0.16"),
        (
            "our CBF vs PET",
            _controls("our_reconstruction", "cbf", "raichle_cbf"),
            "0.39",
        ),
        (
            "our CMRO2 vs PET",
            _controls("our_reconstruction", "cmro2", "raichle_cmro2"),
            "0.138",
        ),
        (
            "authors' CMRO2 vs PET",
            _controls("authors_published", "cmro2", "raichle_cmro2"),
            "0.090",
        ),
        # --- hierarchy -------------------------------------------------------
        (
            "p5 pericyte partial rho",
            _p5("pericyte_mural", "baseline_oef", "rho"),
            "0.419",
        ),
        (
            "p5 pericyte raw rho",
            _p5("pericyte_mural", "baseline_oef", "rho_before"),
            "0.386",
        ),
        (
            "p5 extended pericyte rho",
            _p5("pericyte_mural", "baseline_oef", "rho", "_extended"),
            "0.284",
        ),
        # --- H2 ---------------------------------------------------------------
        ("mediation models", float(p6["n_models"]), "15,840"),
        ("mediation supported", float(p6["n_supported_adjusted"]), "0"),
        ("path a estimate", float(peri.a_median), "0.408"),
        ("path a significant fraction", float(peri.pct_a_sig) * 100, "88"),
        # --- the independent vascular test -------------------------------------
        ("macaque vs extraction", _macaque("discordance_extraction"), "0.079"),
        ("macaque vs overshoot", _macaque("discordance_overshoot"), "0.094"),
        ("macaque vs coupling angle", _macaque("coupling_angle"), "0.040"),
        ("macaque vs baseline OEF", _macaque("baseline_oef"), "0.081"),
        (
            "pericyte vs macaque vascular",
            _p4("pericyte_mural", "macaque_vascular_CONTROL", "rho_median"),
            "0.05",
        ),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draft", default="paper/draft.md")
    args = ap.parse_args()

    draft_path = ROOT / args.draft
    if not draft_path.exists():
        print(f"no draft at {draft_path}")
        return 1
    # Normalise the typographic minus the draft uses for readability.
    # U+2212 MINUS SIGN: the draft uses it for readability, results/ does not.
    text = draft_path.read_text().replace("\u2212", "-")

    try:
        claims = build_claims()
    except (FileNotFoundError, IndexError, KeyError) as exc:
        print(f"cannot read results/: {type(exc).__name__}: {exc}")
        print("Run scripts/regenerate_all.sh first.")
        return 1

    print(f"\n{'=' * 74}\nPAPER NUMBERS vs results/\n{'=' * 74}")
    print(f"  draft: {args.draft}   claims: {len(claims)}\n")
    missing = []
    # Two questions, and the second one is the one that matters.
    #
    # Presence alone -- "does this string appear in the draft" -- passed every
    # claim the night the detectability table went from 33 tests to 44: it found
    # "33" in the prose, never compared it to the computed 44, and reported ok.
    # A gate that verifies a number is *mentioned* rather than *correct* would
    # sign off on a paper contradicting its own results, which is exactly the
    # drift this file exists to prevent.
    #
    # So: the expected string must appear in the draft AND must equal the
    # recomputed value, to the precision the draft states it at. Comparison is on
    # magnitude because the draft writes signed effects with a typographic minus
    # and the artifacts do not.
    for label, value, want in claims:
        present = want in text
        agrees: bool | None = None
        try:
            target = float(want.replace(",", "").replace(MINUS, "-"))
        except ValueError:
            target = None  # a word, e.g. "three" -- presence is all we can check
        if target is not None:
            dp = len(want.split(".")[1]) if "." in want else 0
            tol = 0.5 * 10**-dp + 1e-9
            agrees = abs(abs(target) - abs(float(value))) <= tol

        if not present:
            mark = "MISS"
        elif agrees is False:
            mark = "DRIFT"
        elif agrees is None:
            mark = "ok? "  # unverifiable: the claim is a word, not a number
        else:
            mark = "ok  "
        print(f"  {mark}  {label:<40} {want:>10}   (computed {value:.4g})")
        if not present or agrees is False:
            missing.append((label, want, value))

    stale = [p for p in FORBIDDEN if p in text]
    if stale:
        print(f"\n  {'-' * 70}")
        print("  STALE FRAMING — the draft still asserts the circular criterion:")
        for p in stale:
            print(f"    · {p!r}")
        print("    p0d no longer computes `resolvable`; it reports detectability")
        print("    floors. These passages describe an analysis that no longer exists.")

    print(f"\n{'=' * 74}")
    if stale:
        print(f"VERDICT: {len(stale)} stale passage(s) in the draft. FIX THESE FIRST —")
        print("  positive number checks below cannot detect a wrong framing.\n")
    if missing:
        print(
            f"VERDICT: {len(missing)} claim(s) missing from or contradicted by the draft.\n"
        )
        for label, want, value in missing:
            print(f"  {label}: results/ says {value:.4g}, draft says {want!r}.")
            print("    Either the draft is stale or the formatting changed --")
            print("    check which, do not just edit the number to match.")
    elif not stale:
        print("VERDICT: every checked number in the draft matches results/.")
    print(f"{'=' * 74}\n")
    return 1 if (missing or stale) else 0


if __name__ == "__main__":
    raise SystemExit(main())
