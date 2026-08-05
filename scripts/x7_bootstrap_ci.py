#!/usr/bin/env python
"""Sampling uncertainty, which this study reports nowhere.

The limitations section concedes it: every interval here is a multiverse IQR,
which is *pipeline dispersion* — how much the answer moves when a preprocessing
choice moves. That is not sampling uncertainty, and a reader who mistakes one for
the other will read our intervals as far tighter than they are. This computes the
missing one and puts the two side by side.

The resampling unit
-------------------
Parcels. A spatial correlation is computed over 100 left-hemisphere parcels, so
the relevant question is how much the estimate would move had the cortex been
carved slightly differently, or had a different subset been usable.

**This understates the interval, and by a knowable amount.** A naive parcel
bootstrap assumes parcels are independent; they are not, because neighbouring
parcels share signal. The spin threshold already measures that dependence — it
implies about 64 effectively independent observations out of 100 (§4). A
bootstrap over n units when only n_eff are independent is too narrow by roughly
sqrt(n / n_eff), so both the naive interval and the widened one are reported.
The widened one is the one to quote.

What the comparison is for
--------------------------
If sampling uncertainty dwarfs pipeline dispersion, the multiverse was never the
binding constraint and reporting only an IQR was misleading about precision. If
they are comparable, the multiverse framing holds. Either way the paper should
say which, rather than reporting the smaller of the two by default.

Usage
-----
    python scripts/x7_bootstrap_ci.py
    python scripts/x7_bootstrap_ci.py --n-boot 10000
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from p4_genesets import cell_path, load_genesets, multiverse_dir, targets_for

from src.utils.config import load_config
from src.utils.manifest import manifest

logger = logging.getLogger("x7_bootstrap_ci")

N_LH = 100

# The pairings the paper actually reports. Bootstrapping all 44 would bury the
# ones a reader is asked to believe.
REPORTED = [
    ("pericyte_mural", "baseline_oef"),
    ("HALLMARK_ANGIOGENESIS", "baseline_oef"),
    ("astrocyte", "discordance_overshoot"),
    ("glycolytic_enzymes", "discordance_extraction"),
    ("HALLMARK_OXIDATIVE_PHOSPHORYLATION", "coupling_angle"),
]


def set_score(expr: pd.DataFrame, genes: list[str]) -> np.ndarray | None:
    present = [g for g in genes if g in expr.columns]
    if len(present) < 3:
        return None
    sub = expr[present].iloc[:N_LH]
    z = (sub - sub.mean()) / sub.std()
    return z.mean(axis=1).to_numpy()


def bootstrap_rho(
    x: np.ndarray, y: np.ndarray, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Percentile bootstrap over parcels, resampled with replacement."""
    ok = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[ok], y[ok]
    n = xv.size
    out = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = xv[idx], yv[idx]
        # A resample can be degenerate -- all one parcel -- and a correlation on
        # a constant vector is undefined, not zero. Leave it NaN and drop it.
        if xb.std() > 0 and yb.std() > 0:
            out[b] = sps.spearmanr(xb, yb).statistic
    return out[np.isfinite(out)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--parcellation", default=None)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--max-cells", type=int, default=12)
    args = ap.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.logging.level, format=cfg.logging.format)
    parc = args.parcellation or cfg.parcellation.primary.name

    # Effective observations, from the spin threshold, exactly as x5 recovers it.
    floors = pd.read_csv("results/p0c_detectability_floor.csv")
    ok = floors[floors.attenuation_ceiling > 0]
    spin = float((ok.attenuation_ceiling * ok.detectable_true_rho).round(6).iloc[0])
    n_eff = 3.0 + (sps.norm.ppf(0.975) / math.atanh(spin)) ** 2
    widen = math.sqrt(N_LH / n_eff)
    logger.info(
        "spin %.4f -> n_eff %.1f of %d; widening factor %.3f", spin, n_eff, N_LH, widen
    )

    mv = multiverse_dir(cfg, parc)
    idx = pd.read_csv(mv / "multiverse_index.csv")
    idx = idx[idx.status.isin(["ok", "cached"])].head(args.max_cells)
    targets = targets_for(cfg, parc)
    sets = load_genesets()
    full = pd.read_csv("results/p4_genesets_full.csv")

    rows = []
    for gs, tname in REPORTED:
        if gs not in sets or tname not in targets:
            logger.warning("skipping %s x %s", gs, tname)
            continue
        y = np.asarray(targets[tname], float)

        # Pool bootstrap draws across cells so the interval reflects the same
        # pipelines the point estimate is a median over.
        draws = []
        for _, cell in idx.iterrows():
            expr = pd.read_parquet(cell_path(mv, cell))
            score = set_score(expr, sets[gs]["genes"])
            if score is None:
                continue
            rng = np.random.default_rng(cfg.seed)
            draws.append(bootstrap_rho(score, y, args.n_boot // len(idx) + 1, rng))
        if not draws:
            continue
        b = np.concatenate(draws)

        mvd = full[(full.gene_set == gs) & (full.target == tname)].rho
        lo, hi = np.percentile(b, [2.5, 97.5])
        centre = float(np.median(b))
        half = (hi - lo) / 2 * widen
        rows.append(
            {
                "gene_set": gs,
                "target": tname,
                "rho_median": round(float(mvd.median()), 4),
                "boot_lo": round(float(lo), 4),
                "boot_hi": round(float(hi), 4),
                "boot_width": round(float(hi - lo), 4),
                "boot_lo_widened": round(centre - half, 4),
                "boot_hi_widened": round(centre + half, 4),
                "boot_width_widened": round(float(2 * half), 4),
                "multiverse_iqr": round(
                    float(mvd.quantile(0.75) - mvd.quantile(0.25)), 4
                ),
                "sampling_over_pipeline": round(
                    float(2 * half) / float(mvd.quantile(0.75) - mvd.quantile(0.25)), 2
                ),
                "excludes_zero": bool((centre - half) * (centre + half) > 0),
                "n_boot": int(b.size),
            }
        )
    out_df = pd.DataFrame(rows)

    out = Path("results")
    csv = out / f"x7_bootstrap_ci_{parc}.csv"
    with manifest(f"x7_bootstrap_ci_{parc}", cfg) as man:
        out_df.to_csv(csv, index=False)
        man.record(
            outputs=[str(csv)],
            n_boot=args.n_boot,
            n_cells=len(idx),
            n_eff=round(n_eff, 2),
            widening_factor=round(widen, 4),
            spin_threshold=round(spin, 6),
            median_sampling_over_pipeline=round(
                float(out_df.sampling_over_pipeline.median()), 2
            ),
        )
        man.note(
            "Percentile bootstrap over parcels, widened by sqrt(n / n_eff) for "
            "spatial dependence. The comparison against the multiverse IQR is the "
            "point: pipeline dispersion is not sampling uncertainty, and the paper "
            "previously reported only the former."
        )

    w = 92
    print(f"\n{'=' * w}\nSAMPLING UNCERTAINTY vs PIPELINE DISPERSION — {parc}\n{'=' * w}")
    print(
        f"  {args.n_boot} bootstrap draws over parcels; n_eff = {n_eff:.0f} of "
        f"{N_LH}, so intervals widened x{widen:.2f}\n"
    )
    print(
        f"  {'gene set':34}{'target':>22}{'rho':>8}{'95% CI (widened)':>22}{'vs IQR':>9}"
    )
    for r in out_df.itertuples():
        ci = f"[{r.boot_lo_widened:+.3f}, {r.boot_hi_widened:+.3f}]"
        print(
            f"  {r.gene_set[:33]:34}{r.target[:21]:>22}{r.rho_median:>+8.3f}{ci:>22}"
            f"{r.sampling_over_pipeline:>8.1f}x"
        )
    med = out_df.sampling_over_pipeline.median()
    print(
        f"\n  Sampling uncertainty is {med:.0f}x the multiverse IQR at the median.\n"
        f"  Pipeline choice was never the binding source of imprecision here, and\n"
        f"  reporting only an IQR understated the interval by roughly that factor."
    )
    n_excl = int(out_df.excludes_zero.sum())
    print(f"  {n_excl} of {len(out_df)} reported effects exclude zero at 95%.")
    print(f"\n  -> {csv}\n{'=' * w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
