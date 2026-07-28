#!/usr/bin/env python
"""Fetch every external dataset this project needs.

One command to bootstrap a fresh machine. Everything here is public and
scriptable; the two things that are not are reported at the end rather than
failing silently.

Each step is idempotent — already-present files are verified and skipped, so
re-running after an interruption costs only what is missing.

Usage
-----
    python scripts/fetch_all.py                    # everything
    python scripts/fetch_all.py --only ds004873 ahba
    python scripts/fetch_all.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import REPO_ROOT, load_config

logger = logging.getLogger("fetch_all")

# ds004873 patterns actually consumed by the pipeline. The fmriprep BOLD
# derivatives are ~1 GB per subject and are deliberately excluded.
DS004873_PATTERNS = [
    "derivatives/*.nii.gz",  # group maps, atlases
    "derivatives/*/qmri/*space-MNI152*",  # per-subject quantitative
    "derivatives/*/qmri/*task-calc_T1w-space_desc-CBV_cmro2.nii.gz",
    "derivatives/*/qmri/*task-control_space-T1w_oef.nii.gz",  # warp validation
    "derivatives/*/anat/*from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5",
]

ALIGNMENT_BASE = (
    "https://raw.githubusercontent.com/TingsterX/alignment_macaque-human/master"
)
ALIGNMENT_FILES = [
    "deformation_macaque-human/L.macaque-to-human.sphere.reg.10k_fs_LR.surf.gii",
    "deformation_macaque-human/L.human-to-macaque.sphere.reg.10k_fs_LR.surf.gii",
    "deformation_macaque-human/L.macaque-to-human.sphere.reg.32k_fs_LR.surf.gii",
    "surfaces/Macaque/10k_fs_LR/MacaqueYerkes19.L.sphere.10k_fs_LR.surf.gii",
    "surfaces/Macaque/10k_fs_LR/MacaqueYerkes19.L.midthickness.10k_fs_LR.surf.gii",
    "surfaces/Macaque/32k_fs_LR/MacaqueYerkes19.L.sphere.32k_fs_LR.surf.gii",
    "surfaces/Macaque/32k_fs_LR/MacaqueYerkes19.L.midthickness.32k_fs_LR.surf.gii",
    "surfaces/Human/10k_fs_LR/S1200.L.sphere.10k_fs_LR.surf.gii",
    "surfaces/Human/10k_fs_LR/S1200.L.midthickness_MSMAll.10k_fs_LR.surf.gii",
    "surfaces/Human/32k_fs_LR/S1200.L.sphere.32k_fs_LR.surf.gii",
    "surfaces/Human/32k_fs_LR/S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii",
    "landmarks/Macaque.L_LANDMARK_ROI.10k_fs_LR.label.gii",
    "landmarks/Human.L_LANDMARK_ROI.10k_fs_LR.label.gii",
    "area_expansion/L.macaque-human.RelativeAreaExpansion.monkey.10k_fs_LR.shape.gii",
    "area_expansion/L.macaque-human.RelativeAreaExpansion.human.10k_fs_LR.shape.gii",
]

AHBA_DONORS = ["9861", "10021", "12876", "14380", "15697"]  # 15496 is 404 upstream


def fetch_ds004873(cfg, dry: bool) -> str:
    from src.data.fetch import DEFAULT_SNAPSHOT, fetch_files

    out = cfg.path("raw") / "ds004873"
    s = fetch_files(
        patterns=DS004873_PATTERNS, out_root=out, tag=DEFAULT_SNAPSHOT, dry_run=dry
    )
    return f"{s['n_matched']} files, {s['total_bytes'] / 1e9:.2f} GB" + (
        "" if dry else f", {len(s['failures'])} failures"
    )


def fetch_ahba(cfg, dry: bool) -> str:
    if dry:
        return f"{len(AHBA_DONORS)} donors (~3.6 GB)"
    import abagen

    got = []
    for d in AHBA_DONORS:
        try:
            abagen.fetch_microarray(donors=[d], verbose=0)
            got.append(d)
        except Exception as exc:
            logger.warning("donor %s unavailable: %s", d, exc)
    return f"{len(got)}/6 donors ({', '.join(got)}) — 15496 is 404 upstream"


def fetch_alignment(cfg, dry: bool) -> str:
    out = REPO_ROOT / "data" / "external" / "macaque_human_alignment"
    if dry:
        return f"{len(ALIGNMENT_FILES)} files (~10 MB)"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for rel in ALIGNMENT_FILES:
        dest = out / Path(rel).name
        if dest.exists():
            n += 1
            continue
        try:
            urllib.request.urlretrieve(f"{ALIGNMENT_BASE}/{rel}", dest)
            n += 1
        except Exception as exc:
            logger.warning("%s: %s", Path(rel).name, exc)
    return f"{n}/{len(ALIGNMENT_FILES)} files"


def fetch_genesets(cfg, dry: bool) -> str:
    out = cfg.path("raw") / "genesets" / "msigdb_sets.json"
    if dry:
        return "MSigDB hallmark + GO blood vessel morphogenesis"
    if out.exists():
        return f"cached ({len(json.loads(out.read_text()))} sets)"
    import gseapy

    want = ["Glycolysis", "Oxidative Phosphorylation", "Angiogenesis", "Hypoxia"]
    h = gseapy.get_library("MSigDB_Hallmark_2020")
    g = gseapy.get_library("GO_Biological_Process_2023")
    sel = {k: h[k] for w in want for k in h if w.lower() in k.lower()}
    sel.update({k: g[k] for k in g if "blood vessel morphogenesis" in k.lower()})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sel))
    return f"{len(sel)} sets"


def fetch_reference_maps(cfg, dry: bool) -> str:
    """neuromaps annotations and the Schaefer/fsaverage surfaces."""
    if dry:
        return "neuromaps annotations + Schaefer atlases (~160 MB)"
    from src.data.parcellate import get_parcellation, gifti_atlas_paths
    from src.stats.hierarchy import REFERENCE_MAPS, fetch_reference_parcels

    for parc in ("schaefer200x7", "schaefer400x7", "dk68"):
        get_parcellation(parc, "10k", "L")
    gifti_atlas_paths("schaefer200x7", "10k")
    n = 0
    for name in REFERENCE_MAPS:
        try:
            fetch_reference_parcels(name, "schaefer200x7")
            n += 1
        except Exception as exc:
            logger.warning("%s: %s", name, exc)
    return f"3 parcellations, {n}/{len(REFERENCE_MAPS)} reference maps"


STEPS = {
    "ds004873": fetch_ds004873,
    "ahba": fetch_ahba,
    "alignment": fetch_alignment,
    "genesets": fetch_genesets,
    "reference": fetch_reference_maps,
}

MANUAL = """
NOT FETCHABLE BY SCRIPT — needs a human:

  BALSA study 1vjnV  (macaque ferumoxytol vascular maps, ~57 MB)
    https://balsa.wustl.edu/study/1vjnV
    Requires a free account; the download endpoint redirects to a login.
    Take the scene 'Laminar R2 and CBV Maps in the Macaque Cortex' and unzip
    it into data/external/macaque_human_alignment/.
    Without it, scripts/x1_macaque_vascular.py cannot run.

  AHBA donor 15496
    The Allen Institute file returns HTTP 404 and their catalogue no longer
    lists it. Every analysis runs on five of six donors. Recovering it means
    contacting them, not scripting.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--only", nargs="*", choices=list(STEPS), default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.logging.level, format=cfg.logging.format)
    steps = args.only or list(STEPS)

    print(f"\n{'=' * 68}\nFETCHING{' (dry run)' if args.dry_run else ''}\n{'=' * 68}")
    results = {}
    for name in steps:
        logger.info("--- %s ---", name)
        try:
            results[name] = STEPS[name](cfg, args.dry_run)
        except Exception as exc:
            results[name] = f"ERROR: {type(exc).__name__}: {exc}"
            logger.error("%s failed: %s", name, exc)

    for name, msg in results.items():
        mark = "!" if str(msg).startswith("ERROR") else "+"
        print(f"  {mark} {name:<12} {msg}")
    print(MANUAL)
    return 0 if not any(str(v).startswith("ERROR") for v in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
