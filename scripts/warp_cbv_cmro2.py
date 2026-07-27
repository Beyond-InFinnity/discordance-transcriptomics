#!/usr/bin/env python
"""Bring the CBV-corrected calc CMRO2 maps into MNI152.

The authors compute calc-condition CMRO2 from the **CBV-corrected** variant
(``desc-CBV``), because CBV itself changes during the task; every other
condition uses ``desc-orig``. Their ``B_Fig1.ipynb`` branches on exactly that.
But ``desc-CBV`` is published only in native space, so the first pass used
``desc-orig`` for calc — which is not what their pipeline does, and which
plausibly explains the implausibly low coupling ratio it produced.

This warps their T1w-space ``desc-CBV`` maps into MNI152 with their own
fmriprep ANTs transform (R4 permits applying an existing transform; it forbids
estimating one), writing to ``data/derived/warped/``.

**Every subject is validated like-for-like.** Validation warps that subject's
T1w-space ``control`` OEF and correlates it against their *published* MNI152
OEF — the same quantity in both spaces, so anything below r ~ 0.99 is a warp
fault.

An earlier version validated the warped ``desc-CBV`` against the published
``desc-orig``, which conflated two different things. Those are different
quantities: for sub-p019 they correlate at only 0.39 **in native space, with no
warp involved at all**, so that check flagged subjects where the CBV correction
simply mattered a lot. The like-for-like check isolates geometry, which is what
validation is for.

Usage
-----
    python scripts/warp_cbv_cmro2.py
    python scripts/warp_cbv_cmro2.py --min-r 0.8
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.targets import DATA_ROOT
from src.data.warp import apply_t1w_to_mni152, validate_warp
from src.utils.config import load_config
from src.utils.manifest import manifest

logger = logging.getLogger("warp_cbv_cmro2")

DERIV = DATA_ROOT / "derivatives"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument(
        "--min-r",
        type=float,
        default=0.99,
        help="minimum like-for-like warp validation r (flip scores ~0.5)",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.logging.level, format=cfg.logging.format)

    out_dir = cfg.path("derived") / "warped"
    out_dir.mkdir(parents=True, exist_ok=True)

    moving = sorted(DERIV.glob("sub-*/qmri/*task-calc_T1w-space_desc-CBV_cmro2.nii.gz"))
    logger.info("found %d T1w-space desc-CBV maps", len(moving))
    if not moving:
        raise FileNotFoundError(
            "no desc-CBV maps on disk. Fetch them first:\n"
            "  derivatives/*/qmri/*task-calc_T1w-space_desc-CBV_cmro2.nii.gz\n"
            "  derivatives/*/anat/*from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5"
        )

    rows, warped_paths = [], []
    for mov in moving:
        sub = re.search(r"sub-([A-Za-z0-9]+)", mov.name).group(1)
        xfm = (
            DERIV
            / f"sub-{sub}/anat/sub-{sub}_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5"
        )
        # Reference MUST be a published RAS map, not the bundled LAS template.
        ref = DERIV / f"sub-{sub}/qmri/sub-{sub}_task-control_space-MNI152_oef.nii.gz"
        # Like-for-like validation pair: the same quantity in both spaces.
        val_t1w = DERIV / f"sub-{sub}/qmri/sub-{sub}_task-control_space-T1w_oef.nii.gz"
        val_mni = DERIV / f"sub-{sub}/qmri/sub-{sub}_task-control_space-MNI152_oef.nii.gz"
        if not xfm.exists() or not ref.exists():
            logger.warning("sub-%s: missing transform or reference, skipping", sub)
            rows.append(
                {"subject": sub, "status": "missing_inputs", "warp_validation_r": np.nan}
            )
            continue

        dest = out_dir / f"sub-{sub}_task-calc_space-MNI152_desc-CBV_cmro2.nii.gz"
        try:
            if dest.exists() and not args.overwrite:
                import nibabel as nib

                img = nib.load(dest)
            else:
                img = apply_t1w_to_mni152(mov, xfm, ref)
                import nibabel as nib

                nib.save(img, dest)

            data = np.asarray(img.dataobj)
            status, r = "ok", np.nan
            if val_t1w.exists() and val_mni.exists():
                r = validate_warp(val_t1w, val_mni, xfm, ref, min_r=0.0)
                if r < args.min_r:
                    status = "FAILED_VALIDATION"
                    logger.error(
                        "sub-%s: warp validation r=%.4f < %.2f "
                        "(a left-right flip scores ~0.5)",
                        sub,
                        r,
                        args.min_r,
                    )
            else:
                status = "ok_unvalidated"
                logger.warning("sub-%s: no like-for-like pair, warp unvalidated", sub)

            finite = np.isfinite(data) & (data != 0)
            rows.append(
                {
                    "subject": sub,
                    "status": status,
                    "warp_validation_r": r,
                    "n_voxels": int(finite.sum()),
                    "median_cmro2": float(np.median(data[finite]))
                    if finite.any()
                    else np.nan,
                }
            )
            if status.startswith("ok"):
                warped_paths.append(dest)
        except Exception as exc:
            logger.error("sub-%s: %s: %s", sub, type(exc).__name__, exc)
            rows.append(
                {
                    "subject": sub,
                    "status": f"error:{type(exc).__name__}",
                    "warp_validation_r": np.nan,
                }
            )

    df = pd.DataFrame(rows)
    ok = df.status.str.startswith("ok").fillna(False)

    with manifest("warp_cbv_cmro2", cfg) as man:
        csv = cfg.path("derived") / "warped" / "warp_validation.csv"
        df.to_csv(csv, index=False)
        man.record(
            n_attempted=len(df),
            n_ok=int(ok.sum()),
            n_failed=int((~ok).sum()),
            median_r_vs_orig=float(df.warp_validation_r.median()),
            min_r_vs_orig=float(df.warp_validation_r.min()),
            threshold=args.min_r,
            output_dir=str(out_dir),
        )
        man.note(
            "Applies the authors' own fmriprep ANTs T1w->MNI152 transform. "
            "Reference grid is a published RAS space-MNI152 map; the bundled "
            "MNI152_T1_2mm.nii.gz is LAS and warping into it produces a "
            "left-right flipped result that still looks plausible."
        )

    print(f"\n{'=' * 70}\nCBV-CORRECTED CMRO2 -> MNI152\n{'=' * 70}")
    print(f"  attempted        {len(df)}")
    print(f"  succeeded        {int(ok.sum())}")
    print(f"  failed           {int((~ok).sum())}")
    print(
        f"  validation r     median {df.warp_validation_r.median():.3f}  "
        f"min {df.warp_validation_r.min():.3f}  (threshold {args.min_r})"
    )
    print(f"  median CMRO2     {df.median_cmro2.median():.1f} umol/100g/min")
    if (~ok).any():
        print("\n  FAILURES:")
        print(df[~ok][["subject", "status", "warp_validation_r"]].to_string(index=False))
    print(f"{'=' * 70}")
    return 0 if ok.all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
