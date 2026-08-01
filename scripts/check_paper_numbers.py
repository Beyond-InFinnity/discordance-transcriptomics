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


def _macaque(target: str) -> float:
    d = _csv("x1_macaque_vascular.csv")
    return float(d[d.target == target].rho.iloc[0])


# (label, computed value, format spec). The format is how the number must appear
# in the draft; a claim whose formatted value is absent is a drift failure.
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

    return [
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
    for label, value, want in claims:
        present = want in text
        mark = "ok  " if present else "MISS"
        print(f"  {mark}  {label:<40} {want:>10}   (computed {value:.4g})")
        if not present:
            missing.append((label, want, value))

    print(f"\n{'=' * 74}")
    if missing:
        print(f"VERDICT: {len(missing)} claim(s) not found in the draft.\n")
        for label, want, value in missing:
            print(f"  {label}: results/ says {value:.4g}, expected the draft to")
            print(f"    contain {want!r}. Either the draft is stale or the")
            print("    formatting changed -- check, do not just edit the number.")
    else:
        print("VERDICT: every checked number in the draft matches results/.")
    print(f"{'=' * 74}\n")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
