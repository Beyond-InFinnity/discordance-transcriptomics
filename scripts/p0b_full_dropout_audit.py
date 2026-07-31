#!/usr/bin/env python
"""Phase 0b, extended — dropout against every intermediate quantity. ⛔ GATE.

CLAUDE.md §9 makes Phase 0b blocking: *"if |r| >= 0.5, the confound is severe.
STOP and report — this becomes the finding, and it is a more important one than
the original hypothesis."*

That gate was run against the **final** maps only, and it passed. This script
exists because passing it turned out not to mean what it appeared to.

The CMRO2 audit (``x2_cmro2_audit.py``) found that our baseline CMRO2 disagrees
with the PET reference at -0.14 while being internally reliable at 0.984, and
that the disagreement tracks the dropout proxy at -0.28 and the extraction-mode
discordance map at -0.31, with a network profile — high in limbic and default,
low in visual and somatomotor — matching where T2* is corrupted by sinus-adjacent
field inhomogeneity. T2* is precisely what mqBOLD converts into oxygen
extraction.

So the confound may be entering through an intermediate term rather than through
the final map, which the original gate could not see. mqBOLD is a chain:

    T2, T2*  ->  R2'  ->  OEF  ->  CMRO2  ->  dCMRO2  ->  discordance

A gate applied only to the last link cannot detect corruption at the first. This
tests **every** link, and applies the §9 threshold to each.

If any intermediate quantity breaches |r| >= 0.5 against dropout, the honest
reading is that the gate should have failed, and the dropout finding supersedes
the molecular one.

Usage
-----
    python scripts/p0b_full_dropout_audit.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.parcellate import gifti_for_nulls
from src.data.targets import (
    discordance_modes,
    load_coupling_components,
    load_dropout_proxy,
    load_target_map,
)
from src.stats.spatial import apply_spin, corr_with_null, fdr_bh, spin_indices
from src.utils.config import load_config
from src.utils.manifest import manifest

logger = logging.getLogger("p0b_full_audit")

GATE = 0.5  # §9


def build_chain(cfg, parc: str) -> dict[str, np.ndarray]:
    """Every quantity in the mqBOLD chain, from raw signal to final map."""
    out: dict[str, np.ndarray] = {}

    for name, key in (
        ("1_t2star", "t2star"),
        ("2_baseline_cbv", "baseline_cbv"),
        ("3_baseline_oef", "baseline_oef"),
        ("4_baseline_cbf", "baseline_cbf"),
        ("5_baseline_cmro2", "baseline_cmro2"),
    ):
        try:
            out[name], _ = load_target_map(cfg, key, parc, masked=True)
        except Exception as exc:
            logger.warning("%s unavailable: %s", name, exc)

    d_cbf, d_cmro2, _s, _p = load_coupling_components(parc, masked=True)
    out["6_delta_cbf"] = np.nanmedian(d_cbf, axis=0)
    out["7_delta_cmro2"] = np.nanmedian(d_cmro2, axis=0)

    modes = discordance_modes(d_cbf, d_cmro2)
    out["8_discordance_extraction"] = modes.extraction
    out["9_discordance_overshoot"] = modes.overshoot
    try:
        out["10_coupling_angle"], _ = load_target_map(
            cfg, "coupling_n", parc, masked=True
        )
    except Exception as exc:
        logger.warning("coupling angle unavailable: %s", exc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--parcellation", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.logging.level, format=cfg.logging.format)
    parc = args.parcellation or cfg.parcellation.primary.name
    density = cfg.parcellation.primary.density

    dropout, _ = load_dropout_proxy(cfg, "snr_coverage", parc)
    chain = build_chain(cfg, parc)
    logger.info("testing %d links in the mqBOLD chain", len(chain))

    sidx = spin_indices(
        len(dropout),
        atlas=cfg.parcellation.primary.space,
        density=density,
        parcellation=gifti_for_nulls(parc, density, "L"),
        n_perm=cfg.nulls.n_perm,
        seed=cfg.seed,
        method=cfg.nulls.surface_method,
        cache_path=cfg.path("nulls") / f"spin_indices_{parc}_{density}.npy",
    )
    dnulls = apply_spin(dropout, sidx)

    rows = []
    for name, q in chain.items():
        ok = np.isfinite(dropout) & np.isfinite(q)
        if ok.sum() < 10:
            logger.warning("%s: only %d usable parcels, skipped", name, ok.sum())
            continue
        res = corr_with_null(
            dropout[ok], q[ok], nulls=dnulls[ok, :], method=cfg.stats.correlation
        )
        rows.append(
            {
                "link": name,
                "rho_vs_dropout": res.rho,
                "abs_rho": abs(res.rho),
                "p_spin": res.p_spin,
                "n_valid": res.n_valid,
                "breaches_gate": abs(res.rho) >= GATE,
            }
        )

    df = pd.DataFrame(rows).sort_values("abs_rho", ascending=False)
    df["p_fdr"] = fdr_bh(df.p_spin.to_numpy())
    breached = df[df.breaches_gate]
    verdict = "FAIL" if len(breached) else "PASS"

    out = Path("results")
    with manifest(f"p0b_full_dropout_audit_{parc}", cfg) as man:
        df.to_csv(out / f"p0b_full_dropout_audit_{parc}.csv", index=False)
        man.record(
            gate_threshold=GATE,
            verdict=verdict,
            n_links_tested=len(df),
            n_breaching=len(breached),
            breaching_links=breached.link.tolist(),
            max_abs_rho=float(df.abs_rho.max()),
            worst_link=str(df.link.iloc[0]),
        )
        man.note(
            "The original Phase 0b tested the final maps only. mqBOLD is a chain "
            "(T2/T2* -> R2' -> OEF -> CMRO2 -> dCMRO2 -> discordance) and a gate "
            "on the last link cannot see corruption at the first. This applies "
            "the §9 threshold to every link."
        )

    print(
        f"\n{'=' * 74}\nPHASE 0b EXTENDED — DROPOUT vs THE WHOLE mqBOLD CHAIN\n{'=' * 74}"
    )
    print(f"  §9 gate: |rho| >= {GATE} against the dropout proxy is a STOP\n")
    print(f"  {'link':<28}{'rho':>9}{'|rho|':>8}{'p_spin':>10}{'':>4}")
    for _, r in df.iterrows():
        flag = "  <-- BREACH" if r.breaches_gate else ""
        print(
            f"  {r.link:<28}{r.rho_vs_dropout:>+9.3f}{r.abs_rho:>8.3f}"
            f"{r.p_spin:>10.4f}{flag}"
        )
    print(f"\n  VERDICT: {verdict}")
    if verdict == "FAIL":
        print(f"  {len(breached)} link(s) breach the gate: {', '.join(breached.link)}")
        print("  Per §9 this supersedes the molecular hypothesis and must be")
        print("  reported as the primary finding.")
    else:
        print(f"  worst link: {df.link.iloc[0]} at |rho| = {df.abs_rho.iloc[0]:.3f}")
        print("  Dropout remains a mandatory covariate in every downstream model.")
    print(f"\n  -> results/p0b_full_dropout_audit_{parc}.csv\n{'=' * 74}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
