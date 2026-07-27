#!/usr/bin/env python
"""Build the released parcel-level annotation table (CLAUDE.md deliverable 2).

Writes ``discordance_annotation.csv`` + a JSON Schema + a data dictionary to
``data/derived/annotation/``. This is the reusable public artifact, so the bar
here is different from an internal analysis file: every column is documented,
every value carries its provenance, and quantities that cannot be computed
honestly are **absent rather than approximated**.

Deliberately absent in v1
-------------------------
``discordance_frequency`` (ordinal 0-4 across tasks). Phase 1 confirmed the
design has four co-equal conditions, but only ``calc`` and ``control`` are
published in MNI152. A 0-2 ordinal over two conditions is not the same
quantity and would be misread as one. See ``results/p1_reproduction_notes.md``.

In its place, ``discordance_risk`` measures the fraction of *subjects* showing
a discordant response in the calc-vs-control contrast — a different and
computable quantity, with its assumptions documented in the data dictionary.

Usage
-----
    python scripts/build_annotation.py
    python scripts/build_annotation.py --ahba     # add AHBA sample coverage
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.targets import discordance_fraction, load_coupling_components
from src.utils.config import load_config
from src.utils.manifest import manifest

logger = logging.getLogger("build_annotation")

PARCELLATIONS = ["schaefer200x7", "schaefer400x7", "dk68"]
VERSION = "1.0.0"

# column -> (dtype, unit, description). Drives both the schema and the dictionary.
COLUMNS: dict[str, tuple[str, str, str]] = {
    "parcellation": ("string", "", "Parcellation this row belongs to."),
    "parcel_index": ("integer", "", "1-based parcel label within the hemisphere."),
    "parcel_name": ("string", "", "Atlas label name."),
    "hemisphere": (
        "string",
        "",
        "Always 'L': only 2 of 6 AHBA donors have right-hemisphere tissue (CLAUDE.md R3).",
    ),
    "baseline_oef": (
        "number",
        "fraction",
        "Baseline oxygen extraction fraction, control condition. Authors' published group median (n=40, GM/R2'/CBV masked).",
    ),
    "baseline_cbf": (
        "number",
        "mL/100g/min",
        "Baseline cerebral blood flow, control condition, same source. Includes the authors' 1/0.75 scanner upscaling.",
    ),
    "baseline_cmro2": (
        "number",
        "umol/100g/min",
        "Baseline cerebral metabolic rate of oxygen, control condition, same source.",
    ),
    "coupling_n_angle": (
        "number",
        "radians",
        "Coupling ratio as the angle atan2(%dCBF, %dCMRO2) in the calc-vs-control contrast, median across subjects. Bounded and continuous through the origin, unlike the raw ratio n.",
    ),
    "discordance_risk": (
        "number",
        "fraction",
        "Fraction of subjects in whom BOLD and CMRO2 move in opposite directions, i.e. coupling ratio n < 1. Uses the first-order approximation sign(dBOLD) = sign(dCBF - dCMRO2). See the data dictionary before using.",
    ),
    "discordance_risk_n": (
        "integer",
        "",
        "Number of subjects contributing to discordance_risk.",
    ),
    "dropout_snr_coverage": (
        "number",
        "fraction",
        "Fraction of the parcel's cortical vertices surviving the authors' SNR criterion. Low values indicate signal dropout; carry as a covariate.",
    ),
    "venous_partial_volume": (
        "number",
        "fraction",
        "Mean venous partial volume (VENAT atlas). The brain-vs-vein confound.",
    ),
    "map_reliability_coupling": (
        "number",
        "correlation",
        "Split-half Spearman-Brown reliability of the WHOLE coupling map at this parcellation (a map-level constant, not per-parcel).",
    ),
    "n_subjects_coupling": (
        "integer",
        "",
        "Subjects with all four maps needed for the coupling contrast.",
    ),
    "baseline_source": ("string", "", "Provenance of the baseline columns."),
    "coupling_source": (
        "string",
        "",
        "Provenance of the coupling and discordance columns.",
    ),
    "ahba_n_samples": (
        "number",
        "",
        "AHBA microarray samples falling in this parcel (left hemisphere). Null when not computed. Parcels with 0 cannot support transcriptomic analysis.",
    ),
}


def available_donors() -> list[str]:
    """Donor IDs actually present in the local abagen cache.

    Donor 15496 is unavailable upstream as of 2026-07-26 (the Allen file ID
    404s; see ``data/MANIFEST.yaml``). Rather than let ``donors='all'`` fail or
    silently vary, we enumerate what is on disk and record it, so the released
    coverage counts always state the denominator they were computed from.
    """
    from abagen.datasets.utils import _get_dataset_dir

    # Checked on disk, NOT via fetch_microarray: calling the fetcher for a
    # missing donor would attempt a download and hang on the 404.
    root = Path(_get_dataset_dir("microarray", verbose=0))
    required = ("MicroarrayExpression.csv", "SampleAnnot.csv", "Probes.csv")

    found = []
    for donor in ("9861", "10021", "12876", "14380", "15496", "15697"):
        d = root / f"normalized_microarray_donor{donor}"
        if d.is_dir() and all((d / f).exists() for f in required):
            found.append(donor)
        else:
            logger.warning("AHBA donor %s not present locally", donor)
    return found


def _ahba_counts(parcellation: str, donors: list[str]) -> dict[int, int] | None:
    """AHBA samples per parcel, or None if the data is unavailable."""
    try:
        import abagen

        from src.data.parcellate import get_parcellation, gifti_atlas_paths

        atlas = gifti_atlas_paths(parcellation, "10k")
        _, counts = abagen.get_expression_data(
            atlas, donors=donors, return_counts=True, verbose=0
        )
        # counts is indexed by parcel label; left-hemisphere labels come first.
        n_lh = get_parcellation(parcellation, "10k", "L")[2]
        total = counts.sum(axis=1)
        return {
            int(label): int(total.loc[label])
            for label in range(1, n_lh + 1)
            if label in total.index
        }
    except Exception as exc:
        logger.warning("AHBA coverage unavailable (%s: %s)", type(exc).__name__, exc)
        return None


def build(cfg, with_ahba: bool) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    tm_dir = cfg.path("target_maps")
    donors = available_donors() if with_ahba else []
    if with_ahba:
        logger.info("AHBA donors available: %d/6 -> %s", len(donors), donors)

    for parc in PARCELLATIONS:
        src = tm_dir / f"target_maps_{parc}.csv"
        if not src.exists():
            raise FileNotFoundError(
                f"{src} missing — run scripts/p2_build_targets.py first."
            )
        t = pd.read_csv(src)

        d_cbf, d_cmro2, subs, _ = load_coupling_components(parc, masked=True)
        frac, n_used = discordance_fraction(d_cbf, d_cmro2)

        df = pd.DataFrame(
            {
                "parcellation": parc,
                "parcel_index": t["parcel_index"],
                "parcel_name": t["parcel_name"],
                "hemisphere": "L",
                "baseline_oef": t["baseline_oef"],
                "baseline_cbf": t["baseline_cbf"],
                "baseline_cmro2": t["baseline_cmro2"],
                "coupling_n_angle": t["coupling_n_angle"],
                "discordance_risk": frac,
                "discordance_risk_n": n_used,
                "dropout_snr_coverage": t["dropout_snr_coverage"],
                "venous_partial_volume": t["venous_partial_volume"],
                "map_reliability_coupling": t["coupling_n_map_reliability"],
                "n_subjects_coupling": len(subs),
                "baseline_source": t["baseline_source"],
                "coupling_source": t["coupling_source"],
            }
        )

        counts = _ahba_counts(parc, donors) if with_ahba else None
        df["ahba_n_samples"] = df["parcel_index"].map(counts) if counts else np.nan
        frames.append(df)
        logger.info("%s: %d parcels", parc, len(df))

    out = pd.concat(frames, ignore_index=True)
    return out[list(COLUMNS)], donors


def write_schema(path: Path, df: pd.DataFrame) -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "discordance_annotation",
        "version": VERSION,
        "description": (
            "Parcel-level annotation of BOLD/CMRO2 discordance and baseline "
            "oxygen metabolism in human left cortex, derived from ds004873 "
            "(Epp et al. 2025, Nat Neurosci)."
        ),
        "type": "array",
        "items": {
            "type": "object",
            "required": ["parcellation", "parcel_index", "parcel_name", "hemisphere"],
            "additionalProperties": False,
            "properties": {
                col: {
                    "type": [dtype, "null"],
                    "unit": unit,
                    "description": desc,
                }
                for col, (dtype, unit, desc) in COLUMNS.items()
            },
        },
    }
    path.write_text(json.dumps(schema, indent=2) + "\n")


def write_dictionary(path: Path, df: pd.DataFrame, cfg, donors: list[str]) -> None:
    lines = [
        f"# discordance_annotation v{VERSION} — data dictionary",
        "",
        "Parcel-level annotation of BOLD/CMRO2 discordance and baseline oxygen",
        "metabolism in human **left** cortex.",
        "",
        "Derived from ds004873 (Epp SM, Castrillón G, Yuan B, Andrews-Hanna J,",
        "Preibisch C, Riedl V, 2025, *Nature Neuroscience*,",
        "doi:10.1038/s41593-025-02132-9), OpenNeuro snapshot 2.0.7, CC0.",
        "",
        f"{len(df)} rows across {df['parcellation'].nunique()} parcellations.",
        "",
        "## Columns",
        "",
        "| column | unit | description |",
        "|---|---|---|",
    ]
    for col, (_dtype, unit, desc) in COLUMNS.items():
        lines.append(f"| `{col}` | {unit or '—'} | {desc} |")

    lines += [
        "",
        "## Read this before using `discordance_risk`",
        "",
        "It is the fraction of subjects, in the calc-vs-control contrast, for whom",
        "BOLD and CMRO₂ move in opposite directions. BOLD tracks deoxyhaemoglobin,",
        "which to first order falls when CBF outpaces oxygen consumption, so",
        "`sign(dBOLD) = sign(dCBF - dCMRO2)`. Both sign cases reduce to the same",
        "condition, **coupling ratio n < 1**, so the n = 1 isocline is the natural",
        "boundary rather than a tuned threshold.",
        "",
        "Two things to hold in mind:",
        "",
        "1. **It rests on that first-order BOLD approximation.** It is not a",
        "   measured BOLD sign. Per-subject BOLD percent-change maps are published",
        "   for only one participant, so a directly measured version is not",
        "   possible from this release.",
        "2. **It is not directly comparable to the ~40% of voxels reported by Epp",
        "   et al.** This counts *subjects* per parcel, not voxels, on a single",
        "   two-condition contrast. The values here run higher.",
        "",
        "Sanity check that it behaves: the default mode network carries the",
        "highest mean discordance of the seven Yeo networks at Schaefer-200,",
        "which is the concentration the source paper reports.",
        "",
        "## A deviation forced by the release",
        "",
        "The authors' own analysis uses the CBV-corrected CMRO₂ variant",
        "(`desc-CBV_cmro2`) for the calc condition and `desc-orig_cmro2` for the",
        "others. **Only `desc-orig_cmro2` is published in MNI152**; the CBV-corrected",
        "variant exists in native T2 space only. The coupling and discordance",
        "columns here therefore use `desc-orig` for both conditions, which is not",
        "what their pipeline used for calc.",
        "",
        "This likely biases the coupling ratio: the median `coupling_n_angle` of",
        "0.26 rad implies n ≈ 0.27, lower than the n ~ 2-4 typical of task",
        "activation. Treat the *relative* ordering across parcels as more",
        "trustworthy than the absolute values, and treat both as provisional",
        "pending either the CBV-corrected maps in MNI152 or a native-space",
        "reconstruction.",
        "",
        "## Limitations",
        "",
        "- **Left hemisphere only.** Only 2 of 6 AHBA donors have right-hemisphere",
        "  tissue, so the annotation is built to pair with transcriptomics.",
        "- **Group-level only.** No individual-level inference is licensed. The",
        "  between-subject consistency of these maps is low even where the group",
        "  map is highly reproducible.",
        "- **Reliability varies by parcellation.** `map_reliability_coupling` is",
        "  the split-half Spearman-Brown reliability of the whole coupling map.",
        "  At DK-68 its confidence interval extends below 0.3; treat coupling",
        "  values at that resolution with caution.",
        "- **Baselines and coupling have different provenance.** Baselines are the",
        "  authors' own masked group maps. The coupling columns are reconstructed",
        "  from the per-subject maps, which are published *unmasked* — the",
        "  per-subject masking in their pipeline is not reproducible from the",
        "  release (only one subject's masked files are included). See",
        "  `results/p1_reproduction_notes.md`.",
        "- **Volumetric→surface projection discards subcortex and cerebellum.**",
        (
            f"- **`ahba_n_samples` was computed from {len(donors)} of the 6 AHBA "
            f"donors** ({', '.join(donors) if donors else 'none'})."
            if donors
            else "- **`ahba_n_samples` is not populated in this build.**"
        ),
        (
            "  Donor 15496 is unavailable upstream as of 2026-07-26 — the Allen"
            if "15496" not in donors
            else ""
        ),
        (
            "  file ID returns HTTP 404 and no replacement is indexed by their API."
            if "15496" not in donors
            else ""
        ),
        (
            "  It is a left-hemisphere donor, so coverage counts here understate a"
            if "15496" not in donors
            else ""
        ),
        (
            "  complete-AHBA run. See `data/MANIFEST.yaml`."
            if "15496" not in donors
            else ""
        ),
        "- **n=40 for baselines, n=30 for coupling**, one scanner, one site.",
        "  Generalisation is untested.",
        "- **`discordance_frequency` is absent by design** — see the header of",
        "  `scripts/build_annotation.py`.",
        "",
        "## Citation",
        "",
        "Cite ds004873 (Epp et al. 2025) alongside this table. The underlying",
        "data is theirs; this artifact is a derived annotation.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument(
        "--ahba",
        action="store_true",
        help="compute AHBA sample coverage per parcel (slow; needs the microarray data)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.logging.level, format=cfg.logging.format)

    out_dir = cfg.path("derived") / "annotation"
    out_dir.mkdir(parents=True, exist_ok=True)

    with manifest("annotation_table", cfg) as man:
        df, donors = build(cfg, with_ahba=args.ahba)
        csv = out_dir / "discordance_annotation.csv"
        pq = out_dir / "discordance_annotation.parquet"
        df.to_csv(csv, index=False)
        df.to_parquet(pq, index=False)
        write_schema(out_dir / "discordance_annotation.schema.json", df)
        write_dictionary(out_dir / "README.md", df, cfg, donors)

        man.record(
            version=VERSION,
            n_rows=len(df),
            parcellations=sorted(df["parcellation"].unique()),
            columns=list(df.columns),
            ahba_included=bool(df["ahba_n_samples"].notna().any()),
            ahba_donors=donors,
            ahba_n_donors=len(donors),
            n_missing={
                c: int(df[c].isna().sum())
                for c in df.select_dtypes(include=[np.number]).columns
            },
        )
        man.note(
            "discordance_frequency omitted: only 2 of 4 conditions published in "
            "MNI152 (§13.5). discordance_risk is a model-free lower bound."
        )

    print(f"\n{'=' * 70}\nANNOTATION TABLE v{VERSION}\n{'=' * 70}")
    print(f"  {len(df)} rows -> {csv}")
    for parc, g in df.groupby("parcellation", sort=False):
        print(
            f"  {parc:<16} {len(g):>4} parcels   "
            f"OEF missing {g['baseline_oef'].isna().sum():>2}   "
            f"discordance_risk median {g['discordance_risk'].median():.3f}"
        )
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
