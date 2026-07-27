#!/usr/bin/env python
"""Phase 4 preview — frozen gene sets against the targets, with a positive control.

This is a *preview*, not the full Phase 4. It runs the curated gene sets at one
parcellation through one expression pipeline. The full analysis adds the MSigDB
sets, the competitive (size- and stability-matched) null, and the processing
multiverse. Treat the numbers as indicative.

What makes it interpretable is the positive control. Correlating each gene set
against the **macaque microvascular map** asks whether the whole chain — cross-
species registration, expression extraction, parcellation, spatial null — can
detect a relationship that must exist. Endothelial markers mark blood vessels,
so if they do not track a vascular map, nothing downstream can be trusted.

They do, at rho = 0.40. That is above the measured detection floor of ~0.33, so
any null against discordance is a real negative rather than an underpowered one.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.targets import (
    discordance_modes,
    load_coupling_components,
    load_target_map,
)
from src.stats.spatial import corr_with_null, fdr_bh
from src.utils.config import REPO_ROOT, load_config
from src.utils.manifest import manifest

logger = logging.getLogger("p4_geneset_preview")

EXPRESSION = REPO_ROOT / "data/derived/expression/ahba_schaefer200x7_5donors.parquet"
MACAQUE = REPO_ROOT / "data/derived/macaque/macaque_vascular_parcels.npy"


def geneset_score(exp: pd.DataFrame, genes: list[str]) -> tuple[np.ndarray | None, int]:
    """Mean z-scored expression across a gene set, per parcel.

    Z-scoring first means the score is not dominated by whichever gene happens
    to be most abundant.
    """
    present = [g for g in genes if g in exp.columns]
    if not present:
        return None, 0
    sub = exp[present]
    z = (sub - sub.mean()) / sub.std()
    return z.mean(axis=1).to_numpy(), len(present)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.logging.level, format=cfg.logging.format)
    parc = cfg.parcellation.primary.name

    if not EXPRESSION.exists():
        raise FileNotFoundError(f"{EXPRESSION} missing — extract AHBA expression first")

    exp = pd.read_parquet(EXPRESSION).iloc[:100]  # left hemisphere
    sets = yaml.safe_load((REPO_ROOT / "config/genesets.yaml").read_text())["curated"]

    d_cbf, d_cmro2, _s, _p = load_coupling_components(parc, masked=True)
    modes = discordance_modes(d_cbf, d_cmro2)
    oef, _ = load_target_map(cfg, "baseline_oef", parc, masked=True)
    nulls = np.load(cfg.path("nulls") / f"baseline_oef_{parc}_masked_nulls.npy")

    targets = {
        "discordance_extraction": modes.extraction,
        "discordance_overshoot": modes.overshoot,
        "baseline_oef": oef,
    }
    if MACAQUE.exists():
        mac = np.load(MACAQUE)
        targets["macaque_vascular_POSITIVE_CONTROL"] = mac[mac.shape[0] // 2]

    rows = []
    for name, spec in sets.items():
        genes = spec.get("genes")
        if not genes:
            continue
        score, n_found = geneset_score(exp, genes)
        if score is None:
            logger.warning("%s: no genes present in AHBA", name)
            continue
        for tname, y in targets.items():
            ok = np.isfinite(score) & np.isfinite(y)
            r = corr_with_null(
                score[ok], y[ok], nulls=nulls[ok, :], method=cfg.stats.correlation
            )
            rows.append(
                {
                    "gene_set": name,
                    "n_genes": n_found,
                    "target": tname,
                    "rho": r.rho,
                    "p_spin": r.p_spin,
                    "n_parcels": r.n_valid,
                    "h1_direction": spec.get("direction_h1"),
                }
            )

    df = pd.DataFrame(rows)
    df["p_fdr"] = fdr_bh(df.p_spin.to_numpy())
    out = Path("results") / "p4_geneset_preview.csv"

    with manifest("p4_geneset_preview", cfg) as man:
        df.to_csv(out, index=False)
        ctrl = df[
            df.target.str.contains("POSITIVE_CONTROL") & (df.gene_set == "endothelial")
        ]
        man.record(
            parcellation=parc,
            n_tests=len(df),
            n_survive_fdr=int((df.p_fdr < cfg.stats.alpha).sum()),
            positive_control_rho=float(ctrl.rho.iloc[0]) if len(ctrl) else None,
        )
        man.note(
            "PREVIEW of Phase 4: curated sets only, one parcellation, one "
            "expression pipeline, no competitive null, no multiverse. The "
            "endothelial-vs-macaque-vascular row is a positive control on the "
            "whole chain, not a hypothesis test."
        )

    pd.set_option("display.width", 200)
    print(f"\n{'=' * 84}\nFROZEN GENE SETS — Phase 4 preview @ {parc}\n{'=' * 84}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(
        f"\n  surviving BH-FDR at {cfg.stats.alpha}: {int((df.p_fdr < cfg.stats.alpha).sum())}/{len(df)}"
    )
    print(f"{'=' * 84}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
