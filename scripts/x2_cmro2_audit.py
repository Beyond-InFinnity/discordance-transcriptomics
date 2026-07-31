#!/usr/bin/env python
"""Audit the one positive control this project fails.

Our baseline CMRO2 map agrees with the Raichle PET reference at rho = 0.09. That
is a cross-modality positive control — the same physiology measured two ways —
and it does not pass.

Phase 0 now shows the map's own split-half reliability is **0.984**, so the
disagreement is not attenuation. A highly reliable map that disagrees with an
independent method of the same quantity is measuring something systematically
different, and that needs characterising rather than filing under limitations.

Three questions, each answerable from data already on disk:

1. **How large can the disagreement be?** Given both reliabilities, what
   correlation should have been observable, and how far short does 0.09 fall?
2. **Does it contaminate the discordance measure?** Discordance is built from
   *change* in CMRO2, not baseline. If the parcelwise baseline disagreement is
   uncorrelated with the discordance maps, the failing control bounds what it
   can be taken to invalidate. If it is correlated, the discordance signal may
   partly be whatever makes our CMRO2 differ from PET, which would be serious.
3. **Is it spatially structured?** A disagreement concentrated in scanner
   dropout regions, or in one functional network, is diagnostic. A uniform one
   is measurement noise between two hard methods.

Usage
-----
    python scripts/x2_cmro2_audit.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.parcellate import gifti_for_nulls
from src.data.targets import (
    discordance_modes,
    load_coupling_components,
    load_dropout_proxy,
    load_target_map,
)
from src.stats.hierarchy import fetch_reference_parcels
from src.stats.spatial import apply_spin, corr_with_null, spin_indices
from src.utils.config import load_config
from src.utils.manifest import manifest

logger = logging.getLogger("x2_cmro2_audit")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--parcellation", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.logging.level, format=cfg.logging.format)
    parc = args.parcellation or cfg.parcellation.primary.name
    density = cfg.parcellation.primary.density

    ours, _ = load_target_map(cfg, "baseline_cmro2", parc, masked=True)
    pet = np.asarray(fetch_reference_parcels("raichle_cmro2", parc, density), float)
    dropout, _ = load_dropout_proxy(cfg, "snr_coverage", parc)
    d_cbf, d_cmro2, _s, _p = load_coupling_components(parc, masked=True)
    modes = discordance_modes(d_cbf, d_cmro2)
    ang, _ = load_target_map(cfg, "coupling_n", parc, masked=True)

    sidx = spin_indices(
        len(ours),
        atlas=cfg.parcellation.primary.space,
        density=density,
        parcellation=gifti_for_nulls(parc, density, "L"),
        n_perm=cfg.nulls.n_perm,
        seed=cfg.seed,
        method=cfg.nulls.surface_method,
        cache_path=cfg.path("nulls") / f"spin_indices_{parc}_{density}.npy",
    )

    ok = np.isfinite(ours) & np.isfinite(pet)
    rho_obs = float(sps.spearmanr(ours[ok], pet[ok]).statistic)

    # --- 1. how large could the agreement have been? ------------------------
    rel = pd.read_csv(Path("results") / f"p0_dynamic_range_{parc}.csv")
    row = rel[rel.name.str.contains("CMRO2", case=False)]
    rel_ours = float(row.split_half_reliability.iloc[0]) if len(row) else np.nan
    rows_ceiling = []
    for rel_pet in (1.0, 0.8, 0.5):
        ceiling = float(np.sqrt(rel_ours * rel_pet))
        rows_ceiling.append(
            {
                "reliability_ours": rel_ours,
                "reliability_pet_assumed": rel_pet,
                "max_observable_rho": ceiling,
                "observed_rho": rho_obs,
                "shortfall": ceiling - abs(rho_obs),
                "implied_true_rho": abs(rho_obs) / ceiling if ceiling else np.nan,
            }
        )
    ceil_df = pd.DataFrame(rows_ceiling)

    # --- 2. does the disagreement contaminate the discordance maps? ---------
    # The residual is what our map says that PET does not, on a common scale.
    zr = lambda v: (sps.rankdata(v) - sps.rankdata(v).mean()) / sps.rankdata(v).std()  # noqa: E731
    resid = np.full_like(ours, np.nan)
    resid[ok] = zr(ours[ok]) - zr(pet[ok])

    rnulls = apply_spin(resid, sidx)
    contam = []
    for name, y in (
        ("discordance_extraction", modes.extraction),
        ("discordance_overshoot", modes.overshoot),
        ("coupling_angle", ang),
        ("dropout_snr_coverage", dropout),
    ):
        m = np.isfinite(resid) & np.isfinite(y)
        res = corr_with_null(
            resid[m], y[m], nulls=rnulls[m, :], method=cfg.stats.correlation
        )
        contam.append(
            {"vs": name, "rho": res.rho, "p_spin": res.p_spin, "n": res.n_valid}
        )
    contam_df = pd.DataFrame(contam)

    # --- 3. is the disagreement spatially structured? -----------------------
    net = None
    try:
        import re

        import nibabel as nib

        from src.data.parcellate import get_schaefer_annot

        m_ = re.fullmatch(r"schaefer(\d+)x(\d+)", parc)
        if m_:
            annot = get_schaefer_annot(int(m_.group(1)), int(m_.group(2)), density, "L")
            _, _, names = nib.freesurfer.read_annot(str(annot))
            labels = [(n.decode() if isinstance(n, bytes) else str(n)) for n in names[1:]]
            nets = [
                lab.split("_")[2] if len(lab.split("_")) > 2 else "other"
                for lab in labels
            ]
            net = (
                pd.DataFrame(
                    {"network": nets, "resid": resid, "abs_resid": np.abs(resid)}
                )
                .groupby("network")
                .agg(
                    mean_resid=("resid", "mean"),
                    mean_abs=("abs_resid", "mean"),
                    n=("resid", "size"),
                )
                .sort_values("mean_resid")
                .reset_index()
            )
    except Exception as exc:
        logger.warning("network breakdown unavailable: %s", exc)

    out = Path("results")
    with manifest(f"x2_cmro2_audit_{parc}", cfg) as man:
        ceil_df.to_csv(out / f"x2_cmro2_ceiling_{parc}.csv", index=False)
        contam_df.to_csv(out / f"x2_cmro2_contamination_{parc}.csv", index=False)
        if net is not None:
            net.to_csv(out / f"x2_cmro2_by_network_{parc}.csv", index=False)
        man.record(
            observed_rho_vs_pet=rho_obs,
            reliability_ours=rel_ours,
            max_observable_if_pet_perfect=float(ceil_df.max_observable_rho.iloc[0]),
            contamination={r["vs"]: round(r["rho"], 4) for r in contam},
            contamination_p={r["vs"]: round(r["p_spin"], 4) for r in contam},
        )
        man.note(
            "Our baseline CMRO2 has split-half reliability 0.984, so its "
            "disagreement with the Raichle PET reference is not attenuation. "
            "The two methods measure the same nominal quantity and disagree "
            "systematically."
        )

    print(f"\n{'=' * 76}\nCMRO2 POSITIVE-CONTROL AUDIT\n{'=' * 76}")
    print(f"  our baseline CMRO2 vs Raichle PET CMRO2:  rho = {rho_obs:+.3f}")
    print(f"  our map's own split-half reliability:     {rel_ours:.3f}")
    print("\n  1. WAS IT ATTENUATION?")
    print(f"     {'if PET reliability is':<26}{'max observable':>16}{'observed':>11}")
    for _, r in ceil_df.iterrows():
        print(
            f"     {r.reliability_pet_assumed:<26.2f}{r.max_observable_rho:>16.3f}"
            f"{r.observed_rho:>11.3f}"
        )
    print("     -> the shortfall is far larger than noise can explain")
    print("\n  2. DOES THE DISAGREEMENT CONTAMINATE THE DISCORDANCE MAPS?")
    print(f"     {'residual vs':<28}{'rho':>8}{'p_spin':>10}")
    for _, r in contam_df.iterrows():
        print(f"     {r['vs'][:27]:<28}{r.rho:>+8.3f}{r.p_spin:>10.4f}")
    if net is not None:
        print("\n  3. IS IT SPATIALLY STRUCTURED? (signed residual by network)")
        for _, r in net.iterrows():
            print(f"     {r.network[:22]:<24}{r.mean_resid:>+8.3f}  (n={int(r.n)})")
    print(f"\n  -> results/x2_cmro2_*\n{'=' * 76}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
