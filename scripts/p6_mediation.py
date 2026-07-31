#!/usr/bin/env python
"""Phase 6 — the mediation model (CLAUDE.md §9).

Fits the pre-specified H2 chain at parcel level::

    gene-set expression  ->  baseline OEF (or CBV)  ->  discordance
           X                        M                       Y

with a spatial null on every path (R1), the indirect effect tested by joint
significance of both links, a bootstrap interval alongside, and the whole thing
repeated across the expression multiverse (R6).

This is the test that decides whether the project supplies a *mechanism* for the
Epp et al. phenomenon rather than a correlation. The thesis conjecture is that
association cortex has sparse capillaries, that this shows up as altered baseline
oxygen extraction, and that the extraction difference is what produces discordant
responses. Written as a path model, that is exactly a -> b.

**Read Phase 5 and Phase 4 before interpreting the output.** They already
establish where this is going: the ``a`` path is real and robust (vascular gene
sets predict baseline OEF at rho ~ -0.39, competitive p = 0.000), while baseline
OEF does not rank-correlate with discordance across parcels (rho = -0.13,
p_spin = 0.36). An indirect effect cannot exceed what its weakest link allows, so
the expected result here is a null indirect effect whose ``limiting_path`` is
``b``. Running it anyway is the point — a pre-specified model reported as fitted,
with the failing link named, is a result. Quietly not running it is not.

Both mediators from §7.3 are fitted, and every model is fitted twice: raw, and
adjusted for the mandatory Phase 0b dropout covariate plus the Phase 5 hierarchy
controls. If an effect only exists before adjustment it is not reported as
mechanism.

Usage
-----
    python scripts/p6_mediation.py --n-boot 10000
    python scripts/p6_mediation.py --max-cells 4     # smoke test
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.parcellate import gifti_for_nulls
from src.data.targets import (
    discordance_modes,
    load_coupling_components,
    load_dropout_proxy,
    load_target_map,
)
from src.expression.multiverse import cell_path, multiverse_dir
from src.stats.hierarchy import fetch_reference_parcels
from src.stats.mediation import mediation
from src.stats.spatial import apply_spin, fdr_bh, spin_indices
from src.utils.config import load_config
from src.utils.manifest import manifest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4_genesets import load_genesets  # shares the frozen gene-set loader

logger = logging.getLogger("p6_mediation")

# Mediators from §7.3. baseline_oef is the pre-specified one; CBV is the
# alternative the protocol asks for alongside.
MEDIATORS = ["baseline_oef", "baseline_cbv"]

# Outcomes. The extraction mode is the one the capillary-density mechanism
# actually concerns — demand rises, flow fails to keep pace, extraction goes up.
# The overshoot mode and the continuous coupling angle are carried for contrast.
OUTCOMES = ["discordance_extraction", "discordance_overshoot", "coupling_angle"]

# Phase 5 controls plus the mandatory Phase 0b dropout proxy.
HIERARCHY_REFS = ["margulies_gradient1", "t1w_t2w_myelin"]


def build_maps(cfg, parc: str) -> tuple[dict, dict, np.ndarray]:
    """Mediators, outcomes, and the covariate block."""
    d_cbf, d_cmro2, _s, _p = load_coupling_components(parc, masked=True)
    modes = discordance_modes(d_cbf, d_cmro2)
    ang, _ = load_target_map(cfg, "coupling_n", parc, masked=True)

    mediators = {}
    for name in MEDIATORS:
        try:
            mediators[name] = load_target_map(cfg, name, parc, masked=True)[0]
        except (FileNotFoundError, KeyError) as exc:
            logger.warning("mediator %s unavailable: %s", name, exc)

    available = {
        "discordance_extraction": modes.extraction,
        "discordance_overshoot": modes.overshoot,
        "coupling_angle": ang,
    }
    outcomes = {k: available[k] for k in OUTCOMES}

    cov_cols = [load_dropout_proxy(cfg, "snr_coverage", parc)[0]]
    cov_names = ["dropout_snr_coverage"]
    for ref in HIERARCHY_REFS:
        try:
            cov_cols.append(np.asarray(fetch_reference_parcels(ref, parc), dtype=float))
            cov_names.append(ref)
        except Exception as exc:
            logger.warning("covariate %s unavailable: %s", ref, exc)
    logger.info("covariates: %s", ", ".join(cov_names))
    return mediators, outcomes, np.column_stack(cov_cols)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument(
        "--parcellation",
        default=None,
        help="override the primary parcellation (§7.1/§11 sensitivity analyses)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.logging.level, format=cfg.logging.format)
    parc = args.parcellation or cfg.parcellation.primary.name
    density = cfg.parcellation.primary.density

    mv_dir = multiverse_dir(cfg, parc)
    idx_path = mv_dir / "multiverse_index.csv"
    if not idx_path.exists():
        raise FileNotFoundError(f"{idx_path} missing — run scripts/p3_multiverse.py")
    idx = pd.read_csv(idx_path)
    idx = idx[idx.status.isin(["ok", "cached"])]
    if args.max_cells:
        idx = idx.head(args.max_cells)
    logger.info("multiverse cells: %d", len(idx))

    gsets = load_genesets()
    logger.info("frozen gene sets: %d", len(gsets))
    mediators, outcomes, covariates = build_maps(cfg, parc)
    n_parcels = len(covariates)

    # One geometry, reused for every exposure and outcome map. Generating
    # surrogates per map would repeat identical spherical rotations thousands of
    # times; see src/stats/spatial.py::spin_indices.
    sidx = spin_indices(
        n_parcels,
        atlas=cfg.parcellation.primary.space,
        density=density,
        parcellation=gifti_for_nulls(parc, density, "L"),
        n_perm=cfg.nulls.n_perm,
        seed=cfg.seed,
        method=cfg.nulls.surface_method,
        cache_path=cfg.path("nulls") / f"spin_indices_{parc}_{density}.npy",
    )
    y_nulls = {k: apply_spin(v, sidx) for k, v in outcomes.items()}

    rows: list[dict] = []
    for n_cell, (_, cell) in enumerate(idx.iterrows(), 1):
        exp = pd.read_parquet(cell_path(mv_dir, cell)).iloc[:n_parcels]
        for gname, gspec in gsets.items():
            present = [g for g in gspec["genes"] if g in exp.columns]
            if len(present) < 3:
                continue
            z = (exp[present] - exp[present].mean()) / exp[present].std()
            score = z.mean(axis=1).to_numpy()
            x_nulls = apply_spin(score, sidx)

            for mname, m in mediators.items():
                for oname, y in outcomes.items():
                    for adjusted in (False, True):
                        try:
                            res = mediation(
                                score,
                                m,
                                y,
                                x_nulls=x_nulls,
                                y_nulls=y_nulls[oname],
                                covariates=covariates if adjusted else None,
                                n_boot=args.n_boot,
                                seed=cfg.seed,
                                method=cfg.stats.correlation,
                                alpha=cfg.stats.alpha,
                            )
                        except ValueError as exc:
                            logger.warning("%s / %s / %s: %s", gname, mname, oname, exc)
                            continue
                        rows.append(
                            {
                                "cell": cell["hash"],
                                "gene_set": gname,
                                "n_genes": len(present),
                                "mediator": mname,
                                "outcome": oname,
                                "adjusted": adjusted,
                                "probe_selection": cell["probe_selection"],
                                "lr_mirror": cell["lr_mirror"],
                                "missing": cell["missing"],
                                **res.as_dict(),
                            }
                        )
        if n_cell % 10 == 0:
            logger.info("  %d/%d cells", n_cell, len(idx))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("no models fitted — check gene-set overlap with expression")

    # FDR across the gene-set family, within each model configuration.
    df["indirect_p_fdr"] = np.nan
    for _, g in df.groupby(
        ["cell", "mediator", "outcome", "adjusted"], sort=False
    ).groups.items():
        df.loc[g, "indirect_p_fdr"] = fdr_bh(df.loc[g, "indirect_p"].to_numpy())

    # Multiverse distribution per model, per R6.
    summ = (
        df.groupby(["gene_set", "mediator", "outcome", "adjusted"], sort=False)
        .agg(
            a_median=("a", "median"),
            b_median=("b", "median"),
            indirect_median=("indirect", "median"),
            indirect_q1=("indirect", lambda x: x.quantile(0.25)),
            indirect_q3=("indirect", lambda x: x.quantile(0.75)),
            pct_consistent_sign=(
                "indirect",
                lambda x: max((x > 0).mean(), (x < 0).mean()),
            ),
            pct_indirect_sig=("indirect_p", lambda x: (x < cfg.stats.alpha).mean()),
            pct_a_sig=("a_p", lambda x: (x < cfg.stats.alpha).mean()),
            pct_b_sig=("b_p", lambda x: (x < cfg.stats.alpha).mean()),
            limiting_path=("limiting_path", lambda x: x.mode().iloc[0]),
            n_cells=("indirect", "size"),
        )
        .reset_index()
    )
    summ["stable_sign"] = summ.pct_consistent_sign >= 0.80
    # A mediation claim needs the effect to hold in most of the multiverse AND
    # to survive the covariates. Both, not either.
    summ["supported"] = (summ.pct_indirect_sig >= 0.80) & summ.stable_sign

    out = Path("results")
    # Suffix non-primary parcellations so a sensitivity run never overwrites the
    # headline result. §11 requires reporting whether each effect holds at DK-68
    # and Schaefer-400, and that is impossible if they share filenames.
    tag = "" if parc == cfg.parcellation.primary.name else f"_{parc}"
    with manifest(f"p6_mediation{tag}", cfg) as man:
        df.to_csv(out / f"p6_mediation_full{tag}.csv", index=False)
        summ.to_csv(out / f"p6_mediation_summary{tag}.csv", index=False)
        adj = summ[summ.adjusted]
        man.record(
            n_cells=len(idx),
            n_gene_sets=len(gsets),
            n_models=len(df),
            n_boot=args.n_boot,
            n_perm=int(sidx.shape[1]),
            mediators=list(mediators),
            outcomes=list(outcomes),
            n_covariates=int(covariates.shape[1]),
            n_supported_adjusted=int(adj.supported.sum()),
            limiting_path_counts=(
                adj.limiting_path.value_counts().to_dict() if len(adj) else {}
            ),
            outputs=[
                str(out / f"p6_mediation_full{tag}.csv"),
                str(out / f"p6_mediation_summary{tag}.csv"),
            ],
        )
        man.note(
            "The indirect effect is tested by joint significance of both links, "
            "max(a_p, b_p), not by the product's own null. The product null is "
            "built by rotating the exposure and is therefore dominated by "
            "whichever link is strong: it reports mediation whenever a is real, "
            "regardless of b. Both are recorded; indirect_p is the joint test."
        )
        man.note(
            "Every model is fitted twice. adjusted=True residualises X, M and Y "
            "on the Phase 0b dropout proxy and the Phase 5 hierarchy controls, "
            "and is the version any mechanistic claim must rest on."
        )

    print(f"\n{'=' * 74}\nPHASE 6 — MEDIATION\n{'=' * 74}")
    print(f"  {len(df)} models across {len(idx)} pipelines, {args.n_boot} bootstrap")
    adj = summ[summ.adjusted]
    print(
        f"\n  supported (adjusted, >=80% of pipelines, stable sign): "
        f"{int(adj.supported.sum())}/{len(adj)}"
    )
    if len(adj):
        print("\n  which link limits the indirect effect:")
        for k, v in adj.limiting_path.value_counts().items():
            print(f"    {k:<6} {v:>4} models")
    show = (
        summ[summ.adjusted & (summ.outcome == "discordance_extraction")]
        .sort_values("indirect_median", key=abs, ascending=False)
        .head(8)
    )
    if len(show):
        print("\n  strongest indirect effects on the extraction mode (adjusted):")
        print(f"    {'gene set':<34}{'med':<14}{'a':>7}{'b':>7}{'a*b':>8}{'lim':>6}")
        for _, r in show.iterrows():
            print(
                f"    {r.gene_set[:33]:<34}{r.mediator.replace('baseline_', ''):<14}"
                f"{r.a_median:>+7.3f}{r.b_median:>+7.3f}{r.indirect_median:>+8.3f}"
                f"{r.limiting_path:>6}"
            )
    print(f"\n  -> results/p6_mediation_summary.csv\n{'=' * 74}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
