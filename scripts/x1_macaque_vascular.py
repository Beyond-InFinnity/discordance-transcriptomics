#!/usr/bin/env python
"""EXPLORATORY — test discordance against macaque microvascular density.

This is the first *direct* test of the capillary explanation for discordance.
Everything before it used human blood volume as a proxy, and that proxy turned
out to measure the wrong compartment: human dynamic-susceptibility-contrast and
PET blood volume are dominated by large vessels and show no sensory-to-
association gradient (0.97x and 1.04x), whereas ferumoxytol laminar MRI in
macaque resolves the microvasculature and reports 2-3x.

Pipeline
--------
1. macaque Yerkes19 32k fs_LR  --Xu 2020 registration-->  human 32k fs_LR
2. human fs_LR 32k             --neuromaps-->             fsaverage 10k
3. parcellate with the same atlas as every other map here
4. spatial-null correlation against the discordance measures

Status: EXPLORATORY. The frozen hypothesis list was fixed before any results
were seen, and this map is not on it. Nothing here carries confirmatory weight;
it is reported as a labelled exploratory arm.

Caveats that bound the interpretation
-------------------------------------
* **The registration is weakest where the hypothesis lives.** Landmark centroid
  offsets are 6.7 mm median in sensory/motor cortex but 18.2 mm in association
  cortex, exceeding a Schaefer-200 parcel width. Macaque values in default-mode
  parcels are the least trustworthy ones in the map.
* **Coverage is incomplete.** Human cortex expanded relative to macaque, so
  after warping only ~83 of 100 left-hemisphere parcels receive values.
* **Four macaques.**
* **ΔR2\\* is not calibrated CBV.** It carries an additive baseline, so ratios
  compress — the sensory/association ratio reads 1.07x here against the
  paper's 2-3x. Rank order is preserved, which is what Spearman uses, but
  absolute magnitudes should not be quoted from this pipeline.

Usage
-----
    python scripts/x1_macaque_vascular.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.crossspecies import ALIGNMENT_DIR
from src.data.parcellate import get_parcellation
from src.data.targets import (
    discordance_modes,
    load_coupling_components,
    load_target_map,
)
from src.stats.hierarchy import fetch_reference_parcels
from src.stats.spatial import corr_with_null, fdr_bh
from src.utils.config import load_config
from src.utils.manifest import manifest
from src.utils.workbench import ensure_workbench

logger = logging.getLogger("x1_macaque_vascular")

BALSA = ALIGNMENT_DIR / "balsa_1vjnV" / "Autio_eLife2025_km_1vjnV" / "Autio_CBV_files"
CBV_FILE = (
    BALSA / "L.Average.PostMION_R2s_MinusPreMION_Layers12_All_B0CurvCorrected_label"
    ".native.32k_fs_LR.func.gii"
)


def macaque_depths_to_parcels(parcellation: str, tmp: Path) -> np.ndarray:
    """Every cortical depth of the macaque vascular map, on human parcels.

    Returns
    -------
    ndarray, shape (n_depths, n_parcels)
        Depth 0 is the pial surface, the last is the white-matter boundary.
    """
    import nibabel as nib
    from neuromaps import transforms

    from src.data.crossspecies import macaque_to_human

    ensure_workbench()
    tmp.mkdir(parents=True, exist_ok=True)

    warped = tmp / "mac_cbv_human32k.func.gii"
    if not warped.exists():
        # 32k registration, matching the data's native resolution.
        import subprocess

        d = ALIGNMENT_DIR
        subprocess.run(
            [
                "wb_command",
                "-metric-resample",
                str(CBV_FILE),
                str(d / "L.macaque-to-human.sphere.reg.32k_fs_LR.surf.gii"),
                str(d / "S1200.L.sphere.32k_fs_LR.surf.gii"),
                "ADAP_BARY_AREA",
                str(warped),
                "-area-surfs",
                str(d / "MacaqueYerkes19.L.midthickness.32k_fs_LR.surf.gii"),
                str(d / "S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii"),
            ],
            check=True,
            capture_output=True,
        )
    _ = macaque_to_human  # imported for API parity; 32k path used directly above

    arr = np.asarray(nib.load(str(warped)).agg_data())
    if arr.shape[0] != 14:  # agg_data orientation varies by file
        arr = arr.T

    labels, _, n_par = get_parcellation(parcellation, "10k", "L")
    out = np.full((arr.shape[0], n_par), np.nan)

    for k in range(arr.shape[0]):
        col = tmp / f"_depth{k}.func.gii"
        nib.save(
            nib.gifti.GiftiImage(
                darrays=[nib.gifti.GiftiDataArray(arr[k].astype(np.float32))]
            ),
            col,
        )
        gii = transforms.fslr_to_fsaverage(str(col), target_density="10k", hemi="L")
        d = np.asarray(
            gii[0].agg_data() if isinstance(gii, list | tuple) else gii.agg_data()
        )
        for i in range(1, n_par + 1):
            sel = (labels == i) & np.isfinite(d) & (d != 0)
            if sel.any():
                out[k, i - 1] = d[sel].mean()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.logging.level, format=cfg.logging.format)
    parc = cfg.parcellation.primary.name

    if not CBV_FILE.exists():
        raise FileNotFoundError(
            f"macaque vascular map not found at {CBV_FILE}\n"
            "Download BALSA study 1vjnV (free account) into "
            f"{ALIGNMENT_DIR}"
        )

    tmp = cfg.path("derived") / "macaque"
    depths = macaque_depths_to_parcels(parc, tmp)
    logger.info(
        "macaque map: %d depths, %d/%d parcels covered",
        depths.shape[0],
        int(np.isfinite(depths[0]).sum()),
        depths.shape[1],
    )

    d_cbf, d_cmro2, _subs, _ = load_coupling_components(parc, masked=True)
    modes = discordance_modes(d_cbf, d_cmro2)
    oef, _ = load_target_map(cfg, "baseline_oef", parc, masked=True)
    ang, _ = load_target_map(cfg, "coupling_n", parc, masked=True)
    our_cbv, _ = load_target_map(cfg, "baseline_cbv", parc, masked=True)
    grad = fetch_reference_parcels("margulies_gradient1", parc)
    nulls = np.load(cfg.path("nulls") / f"baseline_oef_{parc}_masked_nulls.npy")

    targets = {
        "discordance_extraction": modes.extraction,
        "discordance_overshoot": modes.overshoot,
        "coupling_angle": ang,
        "baseline_oef": oef,
        "our_human_cbv": our_cbv,
        "principal_gradient": grad,
    }

    # Mid-cortical depth: least contaminated by pial surface vessels, closest
    # to the layer where capillary density peaks.
    mid = depths[depths.shape[0] // 2]

    rows = []
    for name, y in targets.items():
        ok = np.isfinite(mid) & np.isfinite(y)
        r = corr_with_null(
            mid[ok],
            y[ok],
            nulls=nulls[ok, :],
            method=cfg.stats.correlation,
            null_method=cfg.nulls.surface_method,
        )
        rows.append(
            {"target": name, "rho": r.rho, "p_spin": r.p_spin, "n_parcels": r.n_valid}
        )
    df = pd.DataFrame(rows)
    df["p_fdr"] = fdr_bh(df.p_spin.to_numpy())

    # Which cortical depth, if any, best tracks the extraction mode? Mouse work
    # puts peak capillary density in the input layer rather than uniformly.
    prof = []
    for k in range(depths.shape[0]):
        ok = np.isfinite(depths[k]) & np.isfinite(modes.extraction)
        r = corr_with_null(
            depths[k][ok],
            modes.extraction[ok],
            nulls=nulls[ok, :],
            method=cfg.stats.correlation,
        )
        prof.append({"depth": k, "rho_vs_extraction": r.rho, "p_spin": r.p_spin})
    prof_df = pd.DataFrame(prof)

    out = Path("results")
    with manifest("x1_macaque_vascular", cfg) as man:
        df.to_csv(out / "x1_macaque_vascular.csv", index=False)
        prof_df.to_csv(out / "x1_macaque_depth_profile.csv", index=False)
        np.save(tmp / "macaque_vascular_parcels.npy", depths)
        man.record(
            parcellation=parc,
            n_depths=int(depths.shape[0]),
            parcels_covered=int(np.isfinite(mid).sum()),
            results={r["target"]: round(r["rho"], 3) for r in rows},
            best_depth=int(prof_df.rho_vs_extraction.abs().idxmax()),
        )
        man.note(
            "EXPLORATORY, not on the frozen hypothesis list. Registration "
            "accuracy is 6.7 mm in sensory cortex but 18.2 mm in association "
            "cortex, so macaque values are least reliable in exactly the "
            "regions the hypothesis concerns."
        )

    print(
        f"\n{'=' * 76}\nMACAQUE MICROVASCULAR DENSITY vs DISCORDANCE — {parc}\n{'=' * 76}"
    )
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(
        "\n  The capillary explanation predicts a NEGATIVE correlation with the"
        "\n  extraction mode: fewer vessels, supply fails, extraction rises.\n"
    )
    print("Depth profile against the extraction mode (0 = pial, last = white matter):")
    print(prof_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
