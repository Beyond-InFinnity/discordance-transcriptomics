#!/usr/bin/env python
"""Phase 0a — reliability of the target map. ⛔ BLOCKING GATE (CLAUDE.md §9).

Splits subjects into random halves 1,000 times, computes the parcel-level
discordance map in each half, correlates them, and applies the Spearman-Brown
correction to estimate full-sample reliability.

Gate: median corrected r >= 0.5 proceed | 0.3-0.5 caveats + DK-68 | < 0.3 STOP.

The number is written to the manifest regardless of outcome (§9) — it belongs
in the paper either way.

Usage
-----
    python scripts/p0_reliability.py --config config/base.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.targets import load_subject_target_matrix
from src.stats.reliability import run_reliability
from src.utils.config import load_config
from src.utils.manifest import manifest

logger = logging.getLogger("p0_reliability")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument(
        "--target",
        default="coupling_n",
        help="which target map to test (see config targets.maps)",
    )
    ap.add_argument(
        "--parcellation",
        default=None,
        help="override parcellation name; defaults to config primary",
    )
    ap.add_argument(
        "--unmasked",
        action="store_true",
        help="skip the SNR mask and physiological range limits (diagnostic only)",
    )
    args = ap.parse_args()
    masked = not args.unmasked

    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.logging.level, format=cfg.logging.format)

    gate = cfg.gates.p0_reliability
    parc = args.parcellation or cfg.parcellation.primary.name

    logger.info(
        "Phase 0a | target=%s parcellation=%s n_splits=%d seed=%d",
        args.target,
        parc,
        gate.n_splits,
        cfg.seed,
    )

    # (n_subjects, n_parcels), left hemisphere only (R3).
    data, meta = load_subject_target_matrix(
        cfg, target=args.target, parcellation=parc, masked=masked
    )

    suffix = "masked" if masked else "unmasked"
    name = f"p0_reliability_{args.target}_{parc}_{suffix}"
    with manifest(name, cfg, inputs=meta.inputs) as man:
        res = run_reliability(
            data,
            n_splits=gate.n_splits,
            seed=cfg.seed,
            pass_threshold=gate.pass_threshold,
            caveat_threshold=gate.caveat_threshold,
            method=cfg.stats.correlation,
        )
        man.record(
            target=args.target,
            parcellation=parc,
            masked=masked,
            n_subjects_available=meta.coverage.get("n_subjects"),
            **res.as_dict(),
        )

        if res.verdict == "pass":
            man.note(f"GATE PASS: median SB-corrected r={res.median_r_corrected:.3f}")
        elif res.verdict == "caveat":
            man.note(
                f"GATE CAVEAT: median SB-corrected r={res.median_r_corrected:.3f} "
                f"in [{gate.caveat_threshold}, {gate.pass_threshold}). Proceed only "
                "with prominent caveats and reduced parcel resolution (DK-68). "
                "This is a Stop-and-Ask item (§13.1)."
            )
        else:
            man.note(
                f"GATE FAIL: median SB-corrected r={res.median_r_corrected:.3f} "
                f"< {gate.caveat_threshold}. STOP (R9). Do not proceed with a "
                "workaround."
            )

    print(
        f"\n{'=' * 68}\n"
        f"PHASE 0a GATE — {args.target} @ {parc}\n"
        f"{'=' * 68}\n"
        f"  median split-half r (raw)        {res.median_r_raw:.3f}\n"
        f"  median Spearman-Brown corrected  {res.median_r_corrected:.3f}\n"
        f"  95% interval across splits       [{res.ci_lo:.3f}, {res.ci_hi:.3f}]\n"
        f"  ICC(2,1) median                  {res.icc_median:.3f}\n"
        f"  subjects / parcels               {res.n_subjects} / {res.n_parcels}\n"
        f"  VERDICT                          {res.verdict.upper()}\n"
        f"{'=' * 68}"
    )

    if res.verdict == "stop":
        logger.error("GATE FAILED — stopping per R9. Report this to the user.")
        return 1
    if res.verdict == "caveat":
        logger.warning("GATE IN GREY ZONE — Stop-and-Ask item §13.1.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
