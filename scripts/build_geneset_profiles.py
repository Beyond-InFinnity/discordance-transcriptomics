#!/usr/bin/env python
"""Per-parcel gene-set expression profiles — the app's molecular layer.

The Streamlit app shows, for a chosen parcel, how each frozen gene set is
expressed relative to the cortical average. That needs a small table, because
the expression multiverse it derives from is 4 GB and 120 parquets.

The interesting constraint is agreement. If this file scored gene sets its own
way, the app would quietly report a third number that matches neither Phase 4
nor Phase 5, and nobody would notice until someone compared them. So it reuses
the analysis's own pieces rather than reimplementing them:

* ``load_genesets`` from Phase 4 — so MSigDB sets resolve through ``source_key``
  exactly as the analysis resolves them. A previous ad-hoc version walked the
  YAML for inline ``genes`` lists and silently found 6 of the 11 sets.
* ``select_cells`` from Phase 4b — the same multiverse cells Phase 5 reports on.
* The Phase 4 scoring: z-score each gene across parcels, then average across the
  set.

Median and inter-quartile range across cells, never a single pipeline (R6). A
gene set's parcel profile moves with abagen preprocessing choices, and a single
number would hide that.

Provenance is written **beside the data**, not to ``results/``: this is an app
input rather than an analysis result, and ``regenerate_all.sh`` wipes
``results/`` — a manifest kept there would be destroyed by the very run that
depends on it.

Usage
-----
    python scripts/build_geneset_profiles.py
    python scripts/build_geneset_profiles.py --max-cells 24
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from p4_genesets import load_genesets
from p4b_datadriven import select_cells

from src.expression.multiverse import cell_path, load_index, multiverse_dir
from src.utils.config import load_config
from src.utils.manifest import manifest

logger = logging.getLogger("build_geneset_profiles")

MIN_GENES = 3  # matches p4_genesets: fewer than this is not a set


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--parcellation", default=None)
    ap.add_argument("--max-cells", type=int, default=12)
    args = ap.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.logging.level, format=cfg.logging.format)
    parc = args.parcellation or cfg.parcellation.primary.name

    mv_dir = multiverse_dir(cfg, parc)
    idx = load_index(mv_dir)
    cells = select_cells(idx, args.max_cells)
    gsets = {k: v["genes"] for k, v in load_genesets().items()}
    logger.info("%d gene sets across %d multiverse cells", len(gsets), len(cells))

    n_lh = 100 if parc == "schaefer200x7" else None
    per_cell: dict[str, list[np.ndarray]] = {}
    present_n: dict[str, int] = {}
    for _, cell in cells.iterrows():
        exp = pd.read_parquet(cell_path(mv_dir, cell))
        if n_lh:
            exp = exp.iloc[:n_lh]
        for name, genes in gsets.items():
            present = [g for g in genes if g in exp.columns]
            if len(present) < MIN_GENES:
                logger.debug("%s: only %d genes present, skipped", name, len(present))
                continue
            block = exp[present]
            z = (block - block.mean()) / block.std()
            per_cell.setdefault(name, []).append(z.mean(axis=1).to_numpy())
            present_n[name] = len(present)

    rows = []
    for name, mats in per_cell.items():
        m = np.vstack(mats)
        for i in range(m.shape[1]):
            col = m[:, i]
            rows.append(
                {
                    "parcellation": parc,
                    "parcel_index": i + 1,
                    "gene_set": name,
                    "score_median": float(np.nanmedian(col)),
                    "score_q1": float(np.nanpercentile(col, 25)),
                    "score_q3": float(np.nanpercentile(col, 75)),
                    "n_cells": int(m.shape[0]),
                    "n_genes_present": present_n[name],
                    "n_genes_frozen": len(gsets[name]),
                }
            )
    out = pd.DataFrame(rows)

    dest_dir = cfg.path("derived") / "annotation"
    dest_dir.mkdir(parents=True, exist_ok=True)
    csv = dest_dir / "geneset_profiles.csv"
    with manifest("geneset_profiles", cfg, results_dir=dest_dir) as man:
        out.to_csv(csv, index=False)
        man.record(
            outputs=[str(csv)],
            parcellation=parc,
            n_gene_sets=int(out.gene_set.nunique()),
            n_parcels=int(out.parcel_index.nunique()),
            n_cells=args.max_cells,
            cells=[str(c) for c in cells["hash"]],
            genes_found={k: int(v) for k, v in sorted(present_n.items())},
            genes_frozen={k: len(v) for k, v in sorted(gsets.items())},
        )
        man.note(
            "Scored with the Phase 4 method and the Phase 4b cell selection, so "
            "the app cannot drift from the analysis it presents. Median and IQR "
            "across cells rather than one pipeline (R6)."
        )

    missing = sorted(set(gsets) - set(per_cell))
    print(f"\n{'=' * 70}\nGENE-SET PROFILES — {parc}\n{'=' * 70}")
    print(f"  {out.gene_set.nunique()} sets x {out.parcel_index.nunique()} parcels")
    print(f"  across {args.max_cells} multiverse cells\n")
    print(f"  {'gene set':<38}{'found':>7}{'frozen':>8}")
    for name in sorted(present_n):
        flag = "" if present_n[name] == len(gsets[name]) else "  <- partial"
        print(f"  {name[:37]:<38}{present_n[name]:>7}{len(gsets[name]):>8}{flag}")
    if missing:
        print(f"\n  NOT PROFILED (under {MIN_GENES} genes present): {', '.join(missing)}")
    print(f"\n  -> {csv}\n{'=' * 70}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
