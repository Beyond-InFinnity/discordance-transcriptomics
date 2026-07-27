#!/usr/bin/env python
"""Phase 2 — build the target maps at every parcellation (CLAUDE.md §9).

Writes one tidy table per parcellation to ``data/derived/target_maps/`` plus a
manifest per artifact (R10).

What gets built, and from where — this distinction matters and is carried into
the output as a ``source`` column:

**Group-level baseline quantities** (OEF, CBF, CMRO2) come from the authors'
own published ``GMR2pCBVmasked`` group maps, not from our reconstruction.
Phase 1 established that their per-subject masking cannot be reproduced — the
``_qBmasked`` files and per-subject GM/R2'/CBV masks are released for a single
subject only — and that rebuilding from the unmasked per-subject maps yields a
materially different topography (Spearman 0.66 against their map, on their own
voxels). Using their map keeps the released numbers consistent with the paper.

**Subject-varying quantities** (the coupling ratio, and everything feeding the
reliability estimates) must come from the per-subject maps, which are published
unmasked. These are flagged ``source='reconstructed_unmasked'`` and carry the
caveat.

**Not built:** the discordance-frequency map. Phase 1 established the design has
four co-equal conditions (rest/control/mem/calc for subjects p019-p055), but
only ``calc`` and ``control`` are published in MNI152. Building a frequency map
from two of four conditions would misrepresent it. See §13.5.

Usage
-----
    python scripts/p2_build_targets.py
    python scripts/p2_build_targets.py --parcellation dk68
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.parcellate import get_parcellation, project_to_parcels
from src.data.targets import (
    DATA_ROOT,
    load_authors_group_map,
    load_dropout_proxy,
    load_subject_target_matrix,
)
from src.stats.reliability import run_reliability
from src.utils.config import load_config
from src.utils.manifest import manifest

logger = logging.getLogger("p2_build_targets")

PARCELLATIONS = ["schaefer200x7", "schaefer400x7", "dk68"]


def _parcel_names(parcellation: str, density: str) -> list[str]:
    """Human-readable parcel labels, so the released table is interpretable."""
    import nibabel as nib

    from src.data.parcellate import get_schaefer_annot

    if parcellation == "dk68":
        import abagen

        gii = nib.load(abagen.fetch_desikan_killiany(surface=True)["image"][0])
        lut = gii.labeltable.get_labels_as_dict()
        n = int(np.asarray(gii.agg_data()).max())
        return [str(lut.get(i, f"parcel_{i}")) for i in range(1, n + 1)]

    import re

    m = re.fullmatch(r"schaefer(\d+)x(\d+)", parcellation)
    annot = get_schaefer_annot(int(m.group(1)), int(m.group(2)), density, "L")
    _, _, names = nib.freesurfer.read_annot(str(annot))
    return [(n.decode() if isinstance(n, bytes) else str(n)) for n in names[1:]]


def build(cfg, parcellation: str, density: str) -> pd.DataFrame:
    """Assemble the per-parcel table for one parcellation."""
    _, _, n_parcels = get_parcellation(parcellation, density, "L")
    df = pd.DataFrame(
        {
            "parcel_index": np.arange(1, n_parcels + 1),
            "parcel_name": _parcel_names(parcellation, density),
            "hemisphere": "L",
            "parcellation": parcellation,
        }
    )

    # --- authoritative group baselines (authors' masked maps) ---------------
    for qty in ("oef", "cbf", "cmro2"):
        vals, meta = load_authors_group_map(parcellation, qty, density)
        df[f"baseline_{qty}"] = vals
        logger.info(
            "%s baseline_%s: %d/%d parcels empty",
            parcellation,
            qty,
            meta.coverage["n_empty_parcels"],
            n_parcels,
        )

    # --- coupling ratio, reconstructed from per-subject maps ---------------
    mat, cmeta = load_subject_target_matrix(cfg, "coupling_n", parcellation, masked=True)
    df["coupling_n_angle"] = np.nanmedian(mat, axis=0)
    df["coupling_n_n_subjects"] = int(np.isfinite(mat).sum(axis=0).min())

    # Per-parcel reliability of the coupling ratio: how much can each parcel's
    # value be trusted? Released alongside the value itself, which is unusual
    # and is one of the more useful things in this table.
    rel = run_reliability(
        mat,
        n_splits=cfg.gates.p0_reliability.n_splits,
        seed=cfg.seed,
        method=cfg.stats.correlation,
    )
    df["coupling_n_map_reliability"] = rel.median_r_corrected

    # --- mandatory covariates ----------------------------------------------
    snr, _ = load_dropout_proxy(cfg, "snr_coverage", parcellation)
    df["dropout_snr_coverage"] = snr

    ven = project_to_parcels(
        DATA_ROOT / "derivatives" / "VENAT_PartialVolume.nii.gz",
        parcellation=parcellation,
        density=density,
        hemi="L",
        drop_zero=False,
    )
    df["venous_partial_volume"] = ven.values

    # --- provenance columns -------------------------------------------------
    df["baseline_source"] = "authors_published_GMR2pCBVmasked_n40"
    df["coupling_source"] = f"reconstructed_unmasked_n{cmeta.coverage['n_subjects']}"
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--parcellation", default=None, choices=PARCELLATIONS)
    args = ap.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.logging.level, format=cfg.logging.format)

    density = cfg.parcellation.primary.density
    out_dir = cfg.path("target_maps")
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [args.parcellation] if args.parcellation else PARCELLATIONS

    for parc in targets:
        logger.info("building target maps @ %s", parc)
        with manifest(f"p2_target_maps_{parc}", cfg) as man:
            df = build(cfg, parc, density)
            csv = out_dir / f"target_maps_{parc}.csv"
            pq = out_dir / f"target_maps_{parc}.parquet"
            df.to_csv(csv, index=False)
            df.to_parquet(pq, index=False)

            numeric = df.select_dtypes(include=[np.number])
            man.record(
                parcellation=parc,
                n_parcels=len(df),
                columns=list(df.columns),
                n_missing={c: int(df[c].isna().sum()) for c in numeric.columns},
                coupling_n_map_reliability=float(df["coupling_n_map_reliability"][0]),
                outputs=[str(csv), str(pq)],
            )
            man.note(
                "Baselines are the authors' published GMR2pCBVmasked group maps "
                "(n=40). The coupling ratio is reconstructed from unmasked "
                "per-subject maps — their per-subject masking is not "
                "reproducible from the release (see results/"
                "p1_reproduction_notes.md)."
            )
            man.note(
                "discordance_freq NOT built: only 2 of 4 conditions are "
                "published in MNI152 (CLAUDE.md §13.5)."
            )
        logger.info("wrote %s (%d parcels)", csv, len(df))

    print(f"\n{'=' * 66}\nPHASE 2 — target maps written to {out_dir}\n{'=' * 66}")
    for parc in targets:
        d = pd.read_csv(out_dir / f"target_maps_{parc}.csv")
        miss = d["baseline_oef"].isna().sum()
        print(
            f"  {parc:<16} {len(d):>4} parcels   "
            f"OEF missing {miss:>2}   "
            f"coupling reliability {d['coupling_n_map_reliability'][0]:.3f}"
        )
    print(f"{'=' * 66}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
