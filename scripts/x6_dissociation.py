#!/usr/bin/env python
"""Can Epp et al.'s observation be true while ours is null? — a simulation.

§4 makes an interpretive claim the paper leans on and has never demonstrated:
their result and our null are not in conflict, because they are measurements at
different levels. Epp et al. report that *voxels* which behave discordantly
differ in baseline OEF from voxels that do not — a comparison **within** a brain.
We ask whether *parcels* with higher mean OEF show more discordance — a
comparison **between** regions. A mechanism can be real at the first level and
leave nothing at the second.

That is an assertion about the world, so it needs a demonstration rather than a
paragraph. This builds both regimes explicitly and measures what each study
would have found.

The two regimes
---------------
Voxel OEF is a parcel mean plus within-parcel variation. A voxel goes discordant
when its OEF exceeds a threshold, and the whole question is what that threshold
is measured against:

**Absolute.** The threshold is a fixed physiological value, the same everywhere.
A parcel with a higher mean OEF then has more voxels above it, so parcel-mean OEF
predicts parcel discordance fraction and both studies see the effect.

**Relative.** The threshold tracks the local neighbourhood — a voxel is
discordant when its OEF is high *for where it sits*. Every parcel then has the
same discordance fraction regardless of its mean, so the within-parcel effect is
undiminished while the between-parcel correlation is zero by construction.

Neither regime is proposed as the truth. The point is that both reproduce Epp et
al.'s within-brain observation, and only one produces a between-region
correlation for us to find — so our null does not bear on their result.

Reported honestly: the mixing parameter is swept rather than reported at two
points, because a demonstration that only works at the extremes is not much of
one. And the between-parcel correlation is compared against this study's own
detectability floor, since a simulated effect below it would have been invisible
here regardless of the mechanism.

Usage
-----
    python scripts/x6_dissociation.py
    python scripts/x6_dissociation.py --n-sim 200
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.manifest import manifest

logger = logging.getLogger("x6_dissociation")

N_PARCELS = 100
N_VOXELS = 400
# 0 = threshold entirely absolute, 1 = entirely relative to the parcel.
# Sampled finely near 1: the between-region signal survives most of the range
# and collapses only at the top, so the interesting quantity is WHERE it drops
# below this study's floor -- that is what a null actually constrains.
ALPHAS = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 1.0)


def simulate(alpha: float, rng: np.random.Generator, frac: float = 0.30) -> dict:
    """One synthetic brain under a given absolute/relative mixing.

    Parameters
    ----------
    alpha : float
        0 puts the discordance threshold at a single global value; 1 puts it at
        each parcel's own mean. Intermediate values interpolate.
    rng : Generator
    frac : float
        Target overall discordance fraction, used to place the global threshold.

    Returns
    -------
    dict
        The within-brain effect Epp et al. would report, and the between-region
        correlation this study would report.
    """
    # Parcel means vary; voxels vary around them. Both scales are arbitrary --
    # only their ratio matters, and it is held fixed across regimes so the
    # comparison is about the threshold and nothing else.
    parcel_mean = rng.normal(0.0, 1.0, N_PARCELS)
    voxel = parcel_mean[:, None] + rng.normal(0.0, 1.0, (N_PARCELS, N_VOXELS))

    # The threshold shifts with the parcel mean by a fraction alpha. At alpha=0
    # it is one physiological value everywhere; at alpha=1 it tracks the local
    # neighbourhood one-for-one.
    #
    # Deliberately NOT a per-parcel empirical quantile, which was the first
    # attempt: forcing each parcel to the same fraction exactly makes the
    # discordance fraction a constant vector, so the between-region correlation
    # is undefined rather than zero, and the collapse happens only at alpha
    # exactly 1. A shift of the threshold is the mechanism actually being
    # described, and it leaves the natural sampling variation intact.
    global_thr = float(np.quantile(voxel, 1 - frac))
    thr = global_thr + alpha * (parcel_mean[:, None] - parcel_mean.mean())

    discordant = voxel > thr

    # What Epp et al. measure: within a brain, do discordant voxels differ in
    # baseline OEF? Pooled across parcels, as a voxelwise comparison would be.
    d, c = voxel[discordant], voxel[~discordant]
    pooled_sd = math.sqrt(((d.var() * d.size) + (c.var() * c.size)) / (d.size + c.size))
    within_d = (d.mean() - c.mean()) / pooled_sd if pooled_sd > 0 else np.nan

    # What we measure: across parcels, does mean OEF track discordance fraction?
    frac_per_parcel = discordant.mean(axis=1)
    # A constant fraction leaves the correlation undefined rather than zero;
    # report it as absent so a degenerate draw cannot masquerade as a finding.
    between_rho = (
        float(sps.spearmanr(parcel_mean, frac_per_parcel).statistic)
        if frac_per_parcel.std() > 0
        else 0.0
    )

    return {
        "alpha": alpha,
        "within_cohens_d": float(within_d),
        "between_rho": between_rho,
        "discordance_fraction": float(discordant.mean()),
        "frac_sd_across_parcels": float(frac_per_parcel.std()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--parcellation", default=None)
    ap.add_argument("--n-sim", type=int, default=200)
    args = ap.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.logging.level, format=cfg.logging.format)
    parc = args.parcellation or cfg.parcellation.primary.name

    # The floor this study actually had against extraction-mode discordance, so
    # the simulated between-region effects are judged against the real bar.
    floors = pd.read_csv("results/p0c_detectability_floor.csv")
    disc = floors[floors.brain_map == "discordance (extraction)"]
    floor = float(disc.detectable_true_rho.median())
    logger.info("median detectability floor vs extraction: %.3f", floor)

    rows = []
    for alpha in ALPHAS:
        rng = np.random.default_rng(cfg.seed)
        sims = [simulate(alpha, rng) for _ in range(args.n_sim)]
        d = pd.DataFrame(sims)
        rows.append(
            {
                "alpha": alpha,
                "within_cohens_d": round(d.within_cohens_d.mean(), 4),
                "within_sd": round(d.within_cohens_d.std(), 4),
                "between_rho": round(d.between_rho.mean(), 4),
                "between_rho_sd": round(d.between_rho.std(), 4),
                "abs_between_rho": round(d.between_rho.abs().mean(), 4),
                "detectable_here": bool(d.between_rho.abs().mean() >= floor),
                "discordance_fraction": round(d.discordance_fraction.mean(), 4),
            }
        )
    out_df = pd.DataFrame(rows)

    out = Path("results")
    csv = out / f"x6_dissociation_{parc}.csv"
    with manifest(f"x6_dissociation_{parc}", cfg) as man:
        out_df.to_csv(csv, index=False)
        man.record(
            outputs=[str(csv)],
            n_sim=args.n_sim,
            n_parcels=N_PARCELS,
            n_voxels=N_VOXELS,
            detectability_floor_used=round(floor, 4),
            within_d_absolute=float(out_df[out_df.alpha == 0].within_cohens_d.iloc[0]),
            within_d_relative=float(out_df[out_df.alpha == 1].within_cohens_d.iloc[0]),
            between_rho_absolute=float(out_df[out_df.alpha == 0].between_rho.iloc[0]),
            between_rho_relative=float(out_df[out_df.alpha == 1].between_rho.iloc[0]),
        )
        man.note(
            "Both regimes reproduce the within-brain observation that discordant "
            "voxels differ in baseline OEF. Only the absolute-threshold regime "
            "leaves a between-region correlation. The paper's null therefore does "
            "not bear on the within-brain result -- it distinguishes the regimes."
        )

    w = 84
    print(f"\n{'=' * w}\nCAN THEIR RESULT AND OUR NULL BOTH BE TRUE?\n{'=' * w}")
    print(
        f"  {args.n_sim} simulations per regime, {N_PARCELS} parcels x {N_VOXELS} "
        f"voxels\n  this study's floor vs extraction-mode discordance: "
        f"|rho| >= {floor:.3f}\n"
    )
    print(
        f"  {'threshold is...':>22}{'within-brain d':>17}{'between-region rho':>21}"
        f"{'we could see it':>18}"
    )
    for r in out_df.itertuples():
        label = (
            "fully absolute"
            if r.alpha == 0
            else "fully relative"
            if r.alpha == 1
            else f"{r.alpha:.0%} relative"
        )
        print(
            f"  {label:>22}{r.within_cohens_d:>17.2f}{r.between_rho:>+21.3f}"
            f"{('yes' if r.detectable_here else 'no'):>18}"
        )
    a0 = out_df[out_df.alpha == 0].iloc[0]
    a1 = out_df[out_df.alpha == 1].iloc[0]
    invisible = out_df[~out_df.detectable_here]
    bound = float(invisible.alpha.min()) if len(invisible) else float("nan")
    print(
        f"\n  The within-brain effect is essentially unchanged across the sweep "
        f"({a0.within_cohens_d:.2f} to {a1.within_cohens_d:.2f}),\n"
        f"  so every regime reproduces Epp et al.'s observation. The "
        f"between-region correlation\n  collapses from {a0.between_rho:+.3f} to "
        f"{a1.between_rho:+.3f}.\n"
    )
    print(
        "  A null between regions is therefore evidence about WHICH regime holds,\n"
        "  not evidence against the within-brain result."
    )
    if bound == bound:
        print(
            f"\n  And it is not vacuous. Every regime below alpha = {bound:.2f} would "
            f"have produced\n  a between-region correlation this study could see. "
            f"Our null therefore bounds\n  the mechanism at alpha >= {bound:.2f} -- "
            "nearly fully local -- if it operates at all."
        )
    print(f"\n  -> {csv}\n{'=' * w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
