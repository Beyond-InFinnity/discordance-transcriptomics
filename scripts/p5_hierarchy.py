#!/usr/bin/env python
"""Phase 5 — hierarchy control. ⛔ DECISIVE (CLAUDE.md §9).

Association cortex differs from sensory cortex on essentially everything, so
any map varying along the unimodal→transmodal axis correlates with any other
map varying along it. §2 names this the single most likely route to a false
positive in this project.

This script asks, for each target map:

1. How strongly does it track the **Margulies principal gradient** and
   **T1w/T2w myelin**, with spin-test inference (R1)?
2. Do the §9 comparison maps (Raichle CMRGlu/CMRO2/CBF/CBV, evolutionary
   expansion, AHBA gene PC1) still relate to the target **after partialling
   the hierarchy covariates and the mandatory dropout proxy**?

**Interpretation.** If the targets are essentially the principal gradient, we
do not have a molecular finding, we have a hierarchy finding — publishable, but
it must be reported as such (§9) and it changes whether the Phase 3 expression
multiverse is worth running.

This is a scoped early run of the Phase 5 machinery, executed before Phase 3 so
that a null result can save the multiverse compute. The full Phase 5 adds
gene-set terms once Phase 4 exists.

Usage
-----
    python scripts/p5_hierarchy.py
    python scripts/p5_hierarchy.py --parcellation dk68
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.parcellate import schaefer_gifti_for_nulls
from src.data.targets import load_dropout_proxy, load_target_map
from src.stats.hierarchy import (
    HIERARCHY_COVARIATES,
    REFERENCE_MAPS,
    fetch_reference_parcels,
    partial_corr_with_null,
)
from src.stats.spatial import corr_with_null, fdr_bh, make_nulls
from src.utils.config import load_config
from src.utils.manifest import manifest

logger = logging.getLogger("p5_hierarchy")

TARGETS = ["coupling_n", "baseline_oef"]
_PARC_SPEC = {"schaefer200x7": (200, 7), "schaefer400x7": (400, 7)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--parcellation", default=None)
    ap.add_argument("--targets", nargs="*", default=TARGETS)
    args = ap.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.logging.level, format=cfg.logging.format)
    parc = args.parcellation or cfg.parcellation.primary.name
    density = cfg.parcellation.primary.density

    logger.info("fetching %d reference maps @ %s", len(REFERENCE_MAPS), parc)
    refs = {k: fetch_reference_parcels(k, parc, density) for k in REFERENCE_MAPS}

    # Dropout proxy is a mandatory covariate everywhere downstream (Phase 0b).
    dropout, _ = load_dropout_proxy(cfg, "snr_coverage", parc)
    cov_names = [*HIERARCHY_COVARIATES, "dropout_snr_coverage"]

    rows: list[dict] = []
    n_spec = _PARC_SPEC.get(parc, (200, 7))

    for target_name in args.targets:
        target, _tmeta = load_target_map(cfg, target_name, parc, masked=True)
        nulls = make_nulls(
            target,
            atlas=cfg.parcellation.primary.space,
            density=density,
            parcellation=schaefer_gifti_for_nulls(n_spec[0], n_spec[1], density, "L"),
            n_perm=cfg.nulls.n_perm,
            seed=cfg.seed,
            method=cfg.nulls.surface_method,
            cache_path=cfg.path("nulls") / f"{target_name}_{parc}_masked_nulls.npy",
        )

        # --- step 1: raw association with each reference map ---------------
        for ref_name, ref in refs.items():
            ok = np.isfinite(target) & np.isfinite(ref)
            res = corr_with_null(
                target[ok],
                ref[ok],
                nulls=nulls[ok, :],
                method=cfg.stats.correlation,
                null_method=cfg.nulls.surface_method,
            )
            rows.append(
                {
                    "target": target_name,
                    "reference": ref_name,
                    "step": "raw",
                    "rho": res.rho,
                    "p_spin": res.p_spin,
                    "p_naive": res.p_naive,
                    "n_valid": res.n_valid,
                }
            )

        # --- step 2: partial out hierarchy + dropout -----------------------
        covars = np.column_stack([refs[c] for c in HIERARCHY_COVARIATES] + [dropout])
        for ref_name, ref in refs.items():
            if ref_name in HIERARCHY_COVARIATES:
                continue  # partialling a covariate against itself is vacuous
            pr = partial_corr_with_null(
                target,
                ref,
                covars,
                nulls,
                covariate_names=cov_names,
                name=f"{target_name}~{ref_name}",
                method=cfg.stats.correlation,
            )
            rows.append(
                {
                    "target": target_name,
                    "reference": ref_name,
                    "step": "partial",
                    "rho": pr.rho_partial,
                    "p_spin": pr.p_spin_partial,
                    "p_naive": np.nan,
                    "n_valid": pr.n_valid,
                    "rho_before": pr.rho_raw,
                    "attenuation": pr.attenuation,
                }
            )

    # --- positive controls -------------------------------------------------
    # A hierarchy result that is null across the board is only interpretable if
    # the pipeline can be shown to detect relationships that must exist. Two
    # checks: our own quantities against each other (the mqBOLD identity
    # OEF = CMRO2 / (CBF x CaO2) forces specific signs), and our maps against
    # the Raichle PET maps of the same physiology.
    from scipy.stats import spearmanr

    from src.data.targets import load_authors_group_map

    ours = {
        q: load_authors_group_map(parc, q, density)[0] for q in ("oef", "cbf", "cmro2")
    }

    def _rho(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
        m = np.isfinite(a) & np.isfinite(b)
        return float(spearmanr(a[m], b[m]).statistic), int(m.sum())

    controls = []
    for a, b, expect in [
        ("oef", "cmro2", "positive (OEF rises with CMRO2 at fixed CBF)"),
        ("oef", "cbf", "negative (OEF falls as CBF rises at fixed CMRO2)"),
        ("cbf", "cmro2", "positive (flow-metabolism coupling)"),
    ]:
        r, n = _rho(ours[a], ours[b])
        controls.append(
            {"kind": "internal", "a": a, "b": b, "rho": r, "n": n, "expected": expect}
        )
    for q, ref_name in [("cbf", "raichle_cbf"), ("cmro2", "raichle_cmro2")]:
        r, n = _rho(ours[q], refs[ref_name])
        controls.append(
            {
                "kind": "cross_modality",
                "a": q,
                "b": ref_name,
                "rho": r,
                "n": n,
                "expected": "positive (same physiology, different method)",
            }
        )
    controls_df = pd.DataFrame(controls)

    df = pd.DataFrame(rows)
    # BH-FDR within each (target, step) family (§11).
    df["p_fdr"] = np.nan
    for _, idx in df.groupby(["target", "step"]).groups.items():
        df.loc[idx, "p_fdr"] = fdr_bh(df.loc[idx, "p_spin"].to_numpy())

    out_dir = cfg.path("results") if "results" in cfg.paths else Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"p5_hierarchy_{parc}.csv"

    with manifest(f"p5_hierarchy_{parc}", cfg) as man:
        df.to_csv(csv, index=False)
        controls_df.to_csv(out_dir / f"p5_positive_controls_{parc}.csv", index=False)
        grad = df[(df.step == "raw") & (df.reference == "margulies_gradient1")].set_index(
            "target"
        )["rho"]
        man.record(
            parcellation=parc,
            n_tests=len(df),
            covariates=cov_names,
            gradient_rho={k: float(v) for k, v in grad.items()},
            n_survive_partial=int(
                ((df.step == "partial") & (df.p_fdr < cfg.stats.alpha)).sum()
            ),
            output=str(csv),
            positive_controls={
                f"{r.a}~{r.b}": round(float(r.rho), 3) for r in controls_df.itertuples()
            },
        )
        man.note(
            "Scoped early run of Phase 5, executed before Phase 3 so a null "
            "result can save the multiverse compute. Gene-set terms are added "
            "once Phase 4 exists."
        )

    # ---------------------------------------------------------------- report
    pd.set_option("display.width", 200)
    print(
        f"\n{'=' * 78}\nPOSITIVE CONTROLS (does the pipeline detect what must be there?)\n{'=' * 78}"
    )
    print(controls_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    for target_name in args.targets:
        d = df[df.target == target_name]
        print(f"\n{'=' * 78}\nPHASE 5 (early) — {target_name} @ {parc}\n{'=' * 78}")
        print("\nRAW association with reference maps:")
        raw = d[d.step == "raw"][["reference", "rho", "p_spin", "p_naive", "p_fdr"]]
        print(raw.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\nPARTIAL, controlling {', '.join(cov_names)}:")
        par = d[d.step == "partial"][
            ["reference", "rho_before", "rho", "attenuation", "p_spin", "p_fdr"]
        ]
        print(par.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        g = d[(d.step == "raw") & (d.reference == "margulies_gradient1")]
        if len(g):
            r, p = float(g.rho.iloc[0]), float(g.p_spin.iloc[0])
            verdict = (
                "DOMINATED BY HIERARCHY"
                if abs(r) >= 0.5 and p < 0.05
                else "moderate hierarchy loading"
                if abs(r) >= 0.3 and p < 0.05
                else "largely independent of the hierarchy"
            )
            print(f"\n  principal gradient: rho={r:+.3f} (spin p={p:.4f}) -> {verdict}")
    print(f"\n{'=' * 78}\nwrote {csv}\n{'=' * 78}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
