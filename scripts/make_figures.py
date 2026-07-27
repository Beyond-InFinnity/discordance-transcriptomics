#!/usr/bin/env python
"""Build the figure set into ``results/figures/``.

All computation happens here or in ``src/``; ``src/viz`` only draws
(CLAUDE.md §5). Every figure is reproducible from the committed pipeline rather
than from notebook state.

Usage
-----
    python scripts/make_figures.py
    python scripts/make_figures.py --only F2 F3
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.parcellate import get_parcellation, get_schaefer_annot
from src.data.targets import (
    load_authors_group_map,
    load_coupling_components,
    load_subject_target_matrix,
)
from src.stats.reliability import split_half_reliability
from src.utils.config import load_config
from src.utils.manifest import manifest
from src.viz import figures as figs
from src.viz.palette import apply_style

logger = logging.getLogger("make_figures")
PARC = "schaefer200x7"


def _networks() -> pd.Series:
    import nibabel as nib

    _, _, names = nib.freesurfer.read_annot(str(get_schaefer_annot(200, 7, "10k", "L")))
    lbl = [n.decode() if isinstance(n, bytes) else str(n) for n in names[1:]]
    return pd.Series(lbl).str.extract(r"LH_(\w+?)_")[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.logging.level, format=cfg.logging.format)
    warnings.filterwarnings("ignore")
    apply_style()

    out = Path("results/figures")
    want = lambda k: args.only is None or k in args.only  # noqa: E731

    ann = pd.read_csv(cfg.path("derived") / "annotation" / "discordance_annotation.csv")
    nets = _networks()
    written: list[str] = []

    with manifest("figures", cfg) as man:
        # -- F1 surface panel -------------------------------------------------
        if want("F1"):
            labels, _, _ = get_parcellation(PARC, "10k", "L")
            d = ann[ann.parcellation == PARC]
            maps = {
                "Baseline oxygen extraction fraction": d.baseline_oef.to_numpy(),
                "Coupling ratio (angle, radians)": d.coupling_n_angle.to_numpy(),
                "Discordance — extraction mode": d.discordance_risk_extraction.to_numpy(),
                "Discordance — overshoot mode": d.discordance_risk_overshoot.to_numpy(),
            }
            written.append(str(figs.fig_surface_panel(maps, labels, out)))

        # -- F2 network modes -------------------------------------------------
        if want("F2"):
            written.append(str(figs.fig_network_modes(ann, out)))

        # -- F3 coupling plane ------------------------------------------------
        d_cbf = d_cmro2 = None
        if want("F3") or want("F5"):
            d_cbf, d_cmro2, subs, _ = load_coupling_components(PARC, masked=True)
            logger.info("coupling components: %d subjects", len(subs))
        if want("F3"):
            written.append(str(figs.fig_coupling_plane(d_cbf, d_cmro2, nets, out)))

        # -- F4 correlation matrix --------------------------------------------
        if want("F4"):
            written.append(str(figs.fig_correlation_matrix(ann, out)))

        # -- F5 reliability ----------------------------------------------------
        if want("F5"):
            splits = {}
            for parc, label in (
                ("dk68", "DK-68"),
                ("schaefer200x7", "Schaefer-200"),
                ("schaefer400x7", "Schaefer-400"),
            ):
                mat, _ = load_subject_target_matrix(cfg, "coupling_n", parc, masked=True)
                _, corrected = split_half_reliability(
                    mat, n_splits=cfg.gates.p0_reliability.n_splits, seed=cfg.seed
                )
                splits[label] = corrected
            written.append(
                str(
                    figs.fig_reliability(
                        splits, cfg.gates.p0_reliability.pass_threshold, out
                    )
                )
            )

        # -- F6 spin null ------------------------------------------------------
        if want("F6"):
            from scipy.stats import spearmanr

            from src.data.targets import load_target_map
            from src.stats.hierarchy import fetch_reference_parcels
            from src.stats.spatial import corr_with_null

            tgt, _ = load_target_map(cfg, "coupling_n", PARC, masked=True)
            ref = fetch_reference_parcels("margulies_gradient1", PARC)
            nulls = np.load(cfg.path("nulls") / f"coupling_n_{PARC}_masked_nulls.npy")
            ok = np.isfinite(tgt) & np.isfinite(ref)
            res = corr_with_null(tgt[ok], ref[ok], nulls=nulls[ok, :], method="spearman")
            nv, rv = nulls[ok, :], ref[ok]
            null_r = np.array(
                [spearmanr(nv[:, i], rv).statistic for i in range(nv.shape[1])]
            )
            written.append(
                str(
                    figs.fig_spin_null(
                        null_r,
                        res.rho,
                        res.p_spin,
                        res.p_naive,
                        "Coupling ratio vs the principal gradient — the confound that had to be ruled out",
                        out,
                    )
                )
            )

        # -- F7 mqBOLD vs PET --------------------------------------------------
        if want("F7"):
            from scipy.stats import spearmanr

            from src.stats.hierarchy import fetch_reference_parcels

            pairs = {}
            for q, ref_name, label in (
                ("cbf", "raichle_cbf", "Cerebral blood flow"),
                ("cmro2", "raichle_cmro2", "Oxygen metabolism"),
            ):
                ours, _ = load_authors_group_map(PARC, q)
                theirs = fetch_reference_parcels(ref_name, PARC)
                m = np.isfinite(ours) & np.isfinite(theirs)
                pairs[label] = (
                    ours,
                    theirs,
                    float(spearmanr(ours[m], theirs[m]).statistic),
                )
            written.append(str(figs.fig_mqbold_vs_pet(pairs, out)))

        # -- F8 AHBA coverage --------------------------------------------------
        if want("F8"):
            written.append(str(figs.fig_ahba_coverage(ann, out)))

        man.record(n_figures=len(written), outputs=written, parcellation=PARC)
        man.note(
            "Figures draw values computed in src/ and passed in; src/viz performs "
            "no analysis (§5). Palette validated for colour-vision separation "
            "rather than chosen by eye."
        )

    print(f"\n{'=' * 62}\nFIGURES -> {out}\n{'=' * 62}")
    for w in written:
        print(f"  {Path(w).name}")
    print(f"{'=' * 62}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
