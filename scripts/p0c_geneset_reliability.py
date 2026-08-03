#!/usr/bin/env python
"""How reproducible is a gene-set score map? — and what that does to the negative.

Every correlation this project reports is attenuated by measurement noise on
*both* sides. The brain-map side was quantified in Phase 0a. The gene side never
was, and that gap is load-bearing: the detectability floor quoted for the
headline negative assumed the gene maps were noise-free.

    detectable true rho  =  spin threshold / sqrt(reliability_brain * reliability_genes)

With ``reliability_genes = 1`` the floor for the extraction map comes out at
0.33. Individual genes have cross-donor reproducibility around 0.15, and if that
were the set-score reliability the floor would be past 0.6 — the difference
between "we can exclude moderate effects" and "we can only exclude large ones".
Averaging genes into a set score lands somewhere between those, and *where* is an
empirical question nobody had answered here.

Method, deliberately the same shape as the differential stability already used
for single genes (§7.4): extract expression one donor at a time, build each
frozen set's score map per donor, and take the mean pairwise correlation of that
score map across donors. Spearman-Brown then projects a single-donor-pair
agreement up to the full donor panel, exactly as Phase 0a projects a split-half
correlation up to the full subject sample.

What this buys: the floor stops being a range and becomes a number, per gene set,
so a negative can be stated as "an effect above X is excluded" rather than
"nothing reached significance".

Usage
-----
    python scripts/p0c_geneset_reliability.py
    python scripts/p0c_geneset_reliability.py --probe-selection diff_stability
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.parcellate import gifti_atlas_paths
from src.stats.reliability import spearman_brown
from src.utils.config import REPO_ROOT, load_config
from src.utils.manifest import manifest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4_genesets import load_genesets

logger = logging.getLogger("p0c_geneset_reliability")

DONORS = ["9861", "10021", "12876", "14380", "15697"]  # 15496 is 404 upstream

# Each donor extracted in its own interpreter. One donor is far cheaper than the
# full panel, but the pattern is kept from p3: an out-of-memory kill costs one
# donor rather than the run.
_WORKER = """
import sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, {repo!r})
import abagen
from src.data.parcellate import gifti_atlas_paths
donor, dest, probe = sys.argv[1], sys.argv[2], sys.argv[3]
atlas = gifti_atlas_paths({parc!r}, {density!r})
exp = abagen.get_expression_data(
    atlas, donors=[donor], probe_selection=probe, missing="centroids", verbose=0
)
exp.to_parquet(dest)
print("SHAPE", *exp.shape)
"""


def extract_donor(donor: str, dest: Path, parc: str, density: str, probe: str) -> bool:
    """Expression for one donor, cached to parquet."""
    if dest.exists():
        logger.info("donor %s cached", donor)
        return True
    src = _WORKER.format(repo=str(REPO_ROOT), parc=parc, density=density)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        worker = fh.name
    try:
        res = subprocess.run(
            [sys.executable, worker, donor, str(dest), probe],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if res.returncode == 0 and dest.exists():
            logger.info("donor %s extracted", donor)
            return True
        tail = (res.stderr or "").strip().splitlines()
        logger.error("donor %s failed: %s", donor, tail[-1] if tail else "unknown")
        return False
    finally:
        Path(worker).unlink(missing_ok=True)


def set_score(
    expr: pd.DataFrame, genes: list[str], n_parcels: int, min_genes: int = 3
) -> np.ndarray | None:
    """Mean z-scored expression of a set's genes, per parcel.

    ``min_genes`` is 3 for the set score actually used by Phase 4, and 1 when
    scoring a single gene or a small chunk for the construction sweep below.
    """
    present = [g for g in genes if g in expr.columns]
    if len(present) < min_genes:
        return None
    sub = expr[present].iloc[:n_parcels]
    z = (sub - sub.mean()) / sub.std()
    return z.mean(axis=1).to_numpy()


def construction_reliability(
    mats: list[pd.DataFrame],
    genes: list[str],
    n_parcels: int,
    chunk: int,
    seed: int,
    n_reps: int = 10,
) -> float:
    """Panel reliability if the set were scored in chunks of ``chunk`` genes.

    Averaging k genes into one map only *helps* when their true spatial patterns
    resemble each other more than their measurement noise does. In AHBA every
    gene is measured from the same tissue samples, so the noise is shared while
    the signal is shared only where genes genuinely co-localise. For large
    pathway sets that condition fails and the average cancels signal faster than
    noise — so scoring the set in smaller pieces recovers reliability, with
    per-gene (``chunk=1``) as the limit.

    This measures how much. Chunk membership is randomised and averaged over
    ``n_reps`` partitions so one lucky split cannot decide the answer; a partial
    trailing chunk is dropped to keep the chunk size constant, since reliability
    depends on it.
    """
    present = [g for g in genes if g in mats[0].columns]
    if len(present) < max(chunk, 1):
        return float("nan")
    rng = np.random.default_rng(seed)
    reps = 1 if chunk >= len(present) else n_reps
    vals = []
    for _ in range(reps):
        perm = list(rng.permutation(present))
        for i in range(0, len(perm), chunk):
            piece = perm[i : i + chunk]
            if len(piece) < chunk:
                continue
            scores = [set_score(m, piece, n_parcels, min_genes=1) for m in mats]
            scores = [s for s in scores if s is not None]
            if len(scores) < 2:
                continue
            r_pair, _ = mean_pairwise(scores)
            if np.isfinite(r_pair):
                vals.append(spearman_brown(r_pair, factor=len(scores)))
    return float(np.mean(vals)) if vals else float("nan")


def mean_pairwise(scores: list[np.ndarray]) -> tuple[float, int]:
    """Mean Spearman correlation over all donor pairs, and the pair count."""
    from scipy import stats as sps

    vals = []
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            ok = np.isfinite(scores[i]) & np.isfinite(scores[j])
            if ok.sum() >= 10:
                r = sps.spearmanr(scores[i][ok], scores[j][ok]).statistic
                if np.isfinite(r):
                    vals.append(float(r))
    return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/base.yaml")
    # max_intensity, not diff_stability. Two reasons, and the second is the
    # important one: abagen refuses diff_stability on a single donor because the
    # method *is* a cross-donor comparison; and even if it ran, choosing probes
    # to maximise cross-donor stability and then measuring cross-donor stability
    # would be circular. max_intensity picks probes per donor without reference
    # to any other, which is what this measurement needs. Phase 4 uses it for
    # single-gene differential stability for the same reason.
    ap.add_argument("--probe-selection", default="max_intensity")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.logging.level, format=cfg.logging.format)
    parc = cfg.parcellation.primary.name
    density = cfg.parcellation.primary.density

    out_dir = cfg.path("expression") / "per_donor"
    out_dir.mkdir(parents=True, exist_ok=True)
    gifti_atlas_paths(parc, density)  # ensure the atlas is materialised

    mats: dict[str, pd.DataFrame] = {}
    for d in DONORS:
        dest = out_dir / f"donor_{d}_{args.probe_selection}_{parc}.parquet"
        if extract_donor(d, dest, parc, density, args.probe_selection):
            mats[d] = pd.read_parquet(dest)
    if len(mats) < 2:
        raise RuntimeError(f"only {len(mats)} donors extracted; need at least 2")
    logger.info("donors available: %s", ", ".join(mats))

    n_parcels = 100 if parc == "schaefer200x7" else 200
    gsets = load_genesets()

    # Brain-side reliability, so the two halves of the attenuation product can be
    # combined into an actual detectability floor.
    dyn_path = Path("results") / f"p0_dynamic_range_{parc}.csv"
    dyn = pd.read_csv(dyn_path) if dyn_path.exists() else None
    spin_threshold = None
    if dyn is not None and "detectable_true_rho" in dyn:
        # Recover the threshold the Phase 0a run measured from its own null.
        spin_threshold = float(
            (dyn.detectable_true_rho * dyn.attenuation_ceiling).median()
        )
        logger.info("spin threshold from Phase 0a: %.4f", spin_threshold)

    rows = []
    for name, spec in gsets.items():
        scores = []
        for m in mats.values():
            s = set_score(m, spec["genes"], n_parcels)
            if s is not None:
                scores.append(s)
        if len(scores) < 2:
            logger.info("%s: fewer than 2 donors with the set, skipping", name)
            continue
        r_pair, n_pairs = mean_pairwise(scores)
        # Spearman-Brown from one donor pair up to the full panel, matching the
        # projection Phase 0a applies to split-half subject correlations.
        # r_pair is the agreement between two *single* donors, i.e. the
        # reliability of a one-donor map. The panel averages k donors, so the
        # length ratio is k.
        r_panel = spearman_brown(r_pair, factor=len(scores))

        # What the same genes would support under a less destructive
        # construction. reliability_panel is the number the pre-registered
        # analysis actually had; these are what it could have had.
        ms = list(mats.values())
        r_chunk5 = construction_reliability(ms, spec["genes"], n_parcels, 5, cfg.seed)
        r_pergene = construction_reliability(ms, spec["genes"], n_parcels, 1, cfg.seed)

        # Which construction this set should have used. Chosen on reliability
        # alone -- a property of the expression data -- and never on any
        # outcome, so it is a measurement decision rather than a forking path.
        options = {
            "whole_set": r_panel,
            "chunks_of_5": r_chunk5,
            "per_gene": r_pergene,
        }
        best = max(
            (k for k, v in options.items() if v is not None and np.isfinite(v)),
            key=lambda k: options[k],
            default="whole_set",
        )
        rows.append(
            {
                "gene_set": name,
                "n_genes": len(spec["genes"]),
                "n_donors": len(scores),
                "n_pairs": n_pairs,
                "reliability_one_pair": r_pair,
                "reliability_panel": r_panel,
                "reliability_panel_chunk5": r_chunk5,
                "reliability_panel_pergene": r_pergene,
                "best_construction": best,
                "reliability_panel_best": options[best],
                "gain_best_over_whole": options[best] - r_panel,
            }
        )

    rel = pd.DataFrame(rows).sort_values("reliability_panel", ascending=False)

    # Attenuation, per gene set x brain map.
    floors = []
    if dyn is not None and spin_threshold:
        for _, g in rel.iterrows():
            for _, b in dyn.iterrows():
                rg = max(g.reliability_panel, 0.0)
                ceiling = float(np.sqrt(b.signal_fraction * rg))
                # The same pairing under a per-gene construction: same genes,
                # same brain map, same nulls — only the aggregation changes.
                # The gap between the two floors is the part of the design's
                # blindness that was self-inflicted rather than imposed by data.
                rg_best = max(float(g.reliability_panel_best or 0.0), 0.0)
                ceil_best = float(np.sqrt(b.signal_fraction * rg_best))
                floors.append(
                    {
                        "gene_set": g.gene_set,
                        "brain_map": b["name"],
                        "reliability_genes": rg,
                        "reliability_brain": float(b.signal_fraction),
                        "attenuation_ceiling": ceiling,
                        "detectable_true_rho": (
                            float(spin_threshold / ceiling) if ceiling > 0 else np.inf
                        ),
                        "best_construction": g.best_construction,
                        "reliability_genes_best": rg_best,
                        "attenuation_ceiling_best": ceil_best,
                        "detectable_true_rho_best": (
                            float(spin_threshold / ceil_best) if ceil_best > 0 else np.inf
                        ),
                    }
                )
    floor = pd.DataFrame(floors)

    out = Path("results")
    with manifest("p0c_geneset_reliability", cfg) as man:
        rel.to_csv(out / "p0c_geneset_reliability.csv", index=False)
        if len(floor):
            floor.to_csv(out / "p0c_detectability_floor.csv", index=False)
        man.record(
            outputs=[str(p) for p in sorted(out.glob("p0c_*.csv"))],
            probe_selection=args.probe_selection,
            donors=list(mats),
            n_gene_sets=len(rel),
            spin_threshold=spin_threshold,
            median_set_reliability=float(rel.reliability_panel.median()),
            min_set_reliability=float(rel.reliability_panel.min()),
            max_set_reliability=float(rel.reliability_panel.max()),
        )
        man.note(
            "Gene-set score reliability is the mean pairwise cross-donor "
            "correlation of the set's score map, Spearman-Brown projected to the "
            "full donor panel — the same construction as differential stability "
            "for single genes, applied to the averaged score actually used in "
            "the analysis."
        )
        man.note(
            "The detectability floor previously quoted (0.33 for the extraction "
            "map) assumed the gene side was noise-free. These numbers replace "
            "that assumption with a measurement, so a negative can be stated as "
            "an exclusion bound rather than an absence of significance."
        )

    print(f"\n{'=' * 76}\nGENE-SET SCORE RELIABILITY\n{'=' * 76}")
    print(f"  {len(mats)} donors, probe selection = {args.probe_selection}\n")
    print(f"  {'gene set':<38}{'genes':>6}{'1 pair':>9}{'panel':>9}")
    for _, r in rel.iterrows():
        print(
            f"  {r.gene_set[:37]:<38}{r.n_genes:>6}"
            f"{r.reliability_one_pair:>9.3f}{r.reliability_panel:>9.3f}"
        )
    if len(floor):
        print(f"\n  DETECTABILITY FLOOR (spin threshold {spin_threshold:.3f})")
        key = floor[floor.brain_map.str.contains("extraction", case=False)]
        if len(key):
            print("  against the extraction-mode discordance map:\n")
            print(f"    {'gene set':<38}{'ceiling':>9}{'min true rho':>14}")
            for _, r in key.sort_values("detectable_true_rho").iterrows():
                print(
                    f"    {r.gene_set[:37]:<38}{r.attenuation_ceiling:>9.3f}"
                    f"{r.detectable_true_rho:>14.3f}"
                )
    print(f"\n  -> results/p0c_*\n{'=' * 76}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
