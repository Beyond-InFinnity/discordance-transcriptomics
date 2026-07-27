#!/usr/bin/env python
"""Phase 0b — the dropout confound. ⛔ BLOCKING GATE (CLAUDE.md §9).

⚠️ This is the strongest attack on the entire project. Run it before investing
anything else.

mqBOLD derives OEF partly from T2*, and T2* is corrupted by macroscopic B0
inhomogeneity — worst near the sphenoid and frontal sinuses, i.e. directly under
vmPFC, a DMN node. If the discordance map tracks signal dropout, the finding is
an artifact of where the scanner loses signal, not neurobiology.

Gate: |rho| >= 0.5 -> STOP and report. The confound becomes the finding, and
CLAUDE.md is explicit that it is a MORE important one than the original
hypothesis. Below 0.5, the dropout proxy is carried as a mandatory covariate in
every downstream model (config ``covariates.mandatory``).

Inference is via spin test (R1) — a naive correlation between two smooth
volumetric maps is meaningless.

Usage
-----
    python scripts/p0_dropout.py --config config/base.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.parcellate import schaefer_gifti_for_nulls
from src.data.targets import load_dropout_proxy, load_target_map
from src.stats.spatial import corr_with_null, make_nulls
from src.utils.config import load_config
from src.utils.manifest import manifest

logger = logging.getLogger("p0_dropout")

# Parcellation name -> (n_parcels, networks), for building the null parcellation.
_PARC_SPEC_FOR_NULLS = {"schaefer200x7": (200, 7), "schaefer400x7": (400, 7)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--target", default="coupling_n")
    ap.add_argument(
        "--proxy",
        default="snr_coverage",
        choices=["snr_coverage", "t2star"],
        help="dropout proxy to build from the dataset",
    )
    ap.add_argument("--parcellation", default=None)
    ap.add_argument(
        "--unmasked",
        action="store_true",
        help=(
            "evaluate on unmasked data. The gate criterion is defined on MASKED "
            "data (what downstream analysis consumes); the unmasked run is a "
            "reportable diagnostic, not the gate."
        ),
    )
    ap.add_argument(
        "--allow-circular",
        action="store_true",
        help="override the circularity guard below (for diagnostics only)",
    )
    args = ap.parse_args()
    masked = not args.unmasked

    # CIRCULARITY GUARD (CLAUDE.md §13.6).
    # mqBOLD defines OEF = R2' / (x * CBV) with R2' = 1/T2* - 1/T2, so OEF is a
    # deterministic function of T2*. Correlating them measures the formula, not
    # a dropout confound: empirically rho(OEF, R2'/CBV) = +0.90 and
    # rho(R2', T2*) = -0.82 at the parcel level. Running this combination gives
    # rho = -0.75 and "fails" the gate for a reason that is arithmetic.
    # Baseline CBV and CBF are inputs to the same chain and are equally circular.
    _CIRCULAR = {"baseline_oef", "baseline_cbv", "baseline_cbf"}
    if args.proxy == "t2star" and args.target in _CIRCULAR and not args.allow_circular:
        ap.error(
            f"--proxy t2star is circular for --target {args.target}: mqBOLD derives "
            "it from T2* via R2'. Use --proxy snr_coverage, which comes from BOLD "
            "tSNR and is independent of the quantitative chain. Pass "
            "--allow-circular to run it anyway as a diagnostic."
        )

    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.logging.level, format=cfg.logging.format)

    gate = cfg.gates.p0_dropout
    parc = args.parcellation or cfg.parcellation.primary.name
    thresh = gate.severe_threshold

    logger.info(
        "Phase 0b | target=%s proxy=%s parcellation=%s masked=%s",
        args.target,
        args.proxy,
        parc,
        masked,
    )

    target, t_meta = load_target_map(
        cfg, target=args.target, parcellation=parc, masked=masked
    )
    proxy, p_meta = load_dropout_proxy(cfg, proxy=args.proxy, parcellation=parc)

    inputs = list(t_meta.inputs) + list(p_meta.inputs)
    suffix = "masked" if masked else "unmasked"
    name = f"p0_dropout_{args.target}_{args.proxy}_{parc}_{suffix}"

    n_spec = _PARC_SPEC_FOR_NULLS.get(parc, (200, 7))

    with manifest(name, cfg, inputs=inputs) as man:
        # Surrogates are built from the target and cached — they are reused by
        # every downstream test against this same map (§7.4).
        nulls = make_nulls(
            target,
            atlas=cfg.parcellation.primary.space,
            density=cfg.parcellation.primary.density,
            parcellation=schaefer_gifti_for_nulls(
                n_spec[0], n_spec[1], cfg.parcellation.primary.density, "L"
            ),
            n_perm=cfg.nulls.n_perm,
            seed=cfg.seed,
            method=cfg.nulls.surface_method,
            cache_path=cfg.path("nulls") / f"{args.target}_{parc}_{suffix}_nulls.npy",
        )

        res = corr_with_null(
            target,
            proxy,
            nulls=nulls,
            method=cfg.stats.correlation,
            null_method=cfg.nulls.surface_method,
        )
        severe = abs(res.rho) >= thresh
        man.record(
            target=args.target,
            proxy=args.proxy,
            parcellation=parc,
            severe=severe,
            threshold=thresh,
            **res.as_dict(),
        )

        if severe:
            man.note(
                f"GATE FAIL: |rho|={abs(res.rho):.3f} >= {thresh}. The dropout "
                "confound is severe. STOP (R9). Per §9 this becomes the finding, "
                "and it is more important than the original hypothesis."
            )
        else:
            man.note(
                f"GATE PASS: |rho|={abs(res.rho):.3f} < {thresh}. Carry '{args.proxy}' "
                "as a MANDATORY covariate in every downstream model "
                "(config covariates.mandatory)."
            )

    print(
        f"\n{'=' * 68}\n"
        f"PHASE 0b GATE — dropout confound ({args.target} vs {args.proxy})\n"
        f"{'=' * 68}\n"
        f"  Spearman rho                {res.rho:+.3f}\n"
        f"  spin-test p                 {res.p_spin:.4f}  ({res.n_perm} perms, "
        f"{res.null_method})\n"
        f"  naive p (NOT evidence)      {res.p_naive:.4g}\n"
        f"  parcels used                {res.n_valid}\n"
        f"  severity threshold          |rho| >= {thresh}\n"
        f"  VERDICT                     {'FAIL — STOP' if severe else 'PASS'}\n"
        f"{'=' * 68}"
    )

    if severe:
        logger.error(
            "GATE FAILED — dropout confound severe. STOP per R9 and report. "
            "Do not proceed with a workaround."
        )
        return 1

    logger.info("Dropout proxy must now be a mandatory covariate downstream.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
