#!/usr/bin/env python
"""Quantify how much spatial signal each target map actually carries.

Motivation
----------
Looking at the surface renders, the discordance maps appear far flatter than
baseline OEF. That observation on its own does **not** imply they cannot
correlate with anything: Spearman correlation is rank-based and scale-free, so
a narrow absolute range with a reproducible ordering correlates perfectly well.
Visual flatness is largely a colour-scale property.

What does matter is the split between real spatial variance and sampling noise.
A parcel mean carries error ``within-subject variance / n``, and that error
attenuates any correlation the map can show against an external map. So the
useful quantity is not "is the map flat" but:

    what TRUE effect size could we have detected, given this map's noise?

That converts every null result in Phase 5 from "we found nothing" into
"we could have found anything above X, and did not" — or, if X is large,
into "we were never in a position to find anything."

The variance route and the split-half route estimate the same thing by
different means, so agreement between ``signal_fraction`` here and the
Spearman-Brown reliability is a cross-check on both.

Usage
-----
    python scripts/p0_dynamic_range.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.targets import (
    load_coupling_components,
    load_subject_target_matrix,
)
from src.stats.reliability import split_half_reliability, variance_decomposition
from src.utils.config import load_config
from src.utils.manifest import manifest

logger = logging.getLogger("p0_dynamic_range")


def _spin_threshold(cfg, parc: str) -> float:
    """|rho| the spin test requires here, measured from the cached null set."""
    from scipy.stats import spearmanr

    from src.data.targets import load_target_map
    from src.stats.hierarchy import fetch_reference_parcels

    path = cfg.path("nulls") / f"coupling_n_{parc}_masked_nulls.npy"
    if not path.exists():
        logger.warning("no cached nulls; falling back to the measured 0.245")
        return 0.245
    nulls = np.load(path)
    tgt, _ = load_target_map(cfg, "coupling_n", parc, masked=True)
    ref = fetch_reference_parcels("margulies_gradient1", parc)
    ok = np.isfinite(tgt) & np.isfinite(ref)
    nv, rv = nulls[ok, :], ref[ok]
    r = np.array([spearmanr(nv[:, i], rv).statistic for i in range(nv.shape[1])])
    return float(np.percentile(np.abs(r[np.isfinite(r)]), 95))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--parcellation", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.logging.level, format=cfg.logging.format)
    parc = args.parcellation or cfg.parcellation.primary.name

    thr = _spin_threshold(cfg, parc)
    logger.info("spin-test detection threshold at %s: |rho| >= %.3f", parc, thr)

    # Per-subject matrices for every target we can decompose.
    mats: dict[str, np.ndarray] = {}
    oef, _ = load_subject_target_matrix(cfg, "baseline_oef", parc, masked=True)
    mats["baseline OEF"] = oef
    coup, _ = load_subject_target_matrix(cfg, "coupling_n", parc, masked=True)
    mats["coupling angle"] = coup

    d_cbf, d_cmro2, _subs, _ = load_coupling_components(parc, masked=True)
    # Mode fractions are per-subject indicators, so a "subject map" is that
    # subject's 0/1 discordance pattern across parcels.
    bold = np.sign(d_cbf - d_cmro2)
    disc = (bold != np.sign(d_cmro2)).astype(float)
    disc[~(np.isfinite(d_cbf) & np.isfinite(d_cmro2))] = np.nan
    mats["discordance (total)"] = disc
    mats["discordance (extraction)"] = np.where(d_cmro2 > 0, disc, 0.0)
    mats["discordance (overshoot)"] = np.where(d_cmro2 < 0, disc, 0.0)

    rows = []
    for name, mat in mats.items():
        vd = variance_decomposition(mat, name=name, spin_threshold=thr)
        _, corrected = split_half_reliability(
            mat, n_splits=cfg.gates.p0_reliability.n_splits, seed=cfg.seed
        )
        rows.append(
            {
                **vd.as_dict(),
                "split_half_reliability": float(np.nanmedian(corrected)),
            }
        )

    df = pd.DataFrame(rows)
    out = Path("results") / f"p0_dynamic_range_{parc}.csv"
    with manifest(f"p0_dynamic_range_{parc}", cfg) as man:
        df.to_csv(out, index=False)
        man.record(
            parcellation=parc,
            spin_threshold=thr,
            detectable_true_rho={
                r["name"]: round(r["detectable_true_rho"], 3) for r in rows
            },
            signal_fraction={r["name"]: round(r["signal_fraction"], 3) for r in rows},
        )
        man.note(
            "signal_fraction and split_half_reliability estimate the same "
            "quantity by different routes; agreement cross-checks both."
        )

    pd.set_option("display.width", 200)
    print(f"\n{'=' * 92}\nSPATIAL SIGNAL vs SAMPLING NOISE — {parc}\n{'=' * 92}")
    show = df[
        [
            "name",
            "cv",
            "signal_fraction",
            "split_half_reliability",
            "attenuation_ceiling",
            "detectable_true_rho",
        ]
    ].rename(
        columns={
            "name": "map",
            "cv": "coef. var.",
            "signal_fraction": "signal frac",
            "split_half_reliability": "split-half",
            "attenuation_ceiling": "max obs. ρ",
            "detectable_true_rho": "min true ρ",
        }
    )
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(
        f"\n  'min true ρ' is the TRUE correlation needed to clear the spin test "
        f"(|ρ| ≥ {thr:.3f}) once\n  measurement noise has attenuated it. Above ~0.5 the map "
        "could not have shown a\n  realistic biological effect even if one existed.\n"
        f"{'=' * 92}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
