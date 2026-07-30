#!/usr/bin/env python
"""Phase 3 — the expression multiverse.

Markello et al. (2021) showed that defensible choices in AHBA processing can
move an imaging-expression correlation by as much as rho = 1.0 — enough to
reverse a finding. A single-pipeline result is therefore not evidence. Every
association this project reports has to come with the distribution of that
association across the grid of reasonable choices.

The grid varies the five parameters that require re-extracting expression:

    probe_selection   diff_stability, rnaseq, max_intensity, max_variance,
                      corr_variance
    lr_mirror         None, bidirectional
    missing           None, centroids, interpolate
    tolerance         1, 2
    norm_matched      True, False

That is 5 x 2 x 3 x 2 x 2 = 120 cells. The stability threshold (0.0, 0.1, 0.2)
is applied *after* extraction, so it multiplies the analysis without
multiplying the expensive part.

Each cell is cached to parquet keyed by a hash of its parameters, so a rerun
costs nothing and an interrupted run resumes.

Usage
-----
    python scripts/p3_multiverse.py --n-jobs 6
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.parcellate import gifti_atlas_paths
from src.utils.config import REPO_ROOT, config_hash, load_config
from src.utils.manifest import manifest

logger = logging.getLogger("p3_multiverse")

DONORS = ["9861", "10021", "12876", "14380", "15697"]  # 15496 unavailable upstream

GRID = {
    "probe_selection": [
        "diff_stability",
        "rnaseq",
        "max_intensity",
        "max_variance",
        "corr_variance",
    ],
    "lr_mirror": [None, "bidirectional"],
    "missing": [None, "centroids", "interpolate"],
    "tolerance": [1, 2],
    "norm_matched": [True, False],
}


def cells() -> list[dict]:
    """Every combination in the grid, each with its cache key."""
    keys = list(GRID)
    out = []
    for combo in itertools.product(*(GRID[k] for k in keys)):
        params = dict(zip(keys, combo))
        out.append({**params, "hash": config_hash(params)})
    return out


# Each cell runs in its own interpreter. The expensive probe-selection methods
# peak near 7 GB on a machine with roughly 6 GB free, and when the kernel kills
# one it kills the whole process — an in-process loop loses every remaining
# cell. Isolating them means a kill costs exactly one cell, and the parent
# records it as a failure and carries on.
_WORKER = """
import sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, {repo!r})
import abagen
from src.data.parcellate import gifti_atlas_paths
params = json.loads(sys.argv[1])
dest = sys.argv[2]
atlas = gifti_atlas_paths({parc!r}, {density!r})
exp = abagen.get_expression_data(atlas, donors={donors!r}, verbose=0, **params)
exp.to_parquet(dest)
print("SHAPE", *exp.shape)
"""


def run_cell(params: dict, atlas, out_dir: Path, parc: str, density: str) -> dict:
    """Extract one expression matrix in an isolated interpreter, or reuse cache."""
    import json as _json
    import subprocess
    import tempfile

    dest = out_dir / f"expr_{params['hash']}.parquet"
    # Relative to the repo root: an absolute path here does not survive the
    # index being copied to another machine, and this grid is routinely
    # computed on one box and analysed on another.
    try:
        rel = dest.relative_to(REPO_ROOT)
    except ValueError:
        rel = dest
    rec = {**params, "path": str(rel)}
    if dest.exists():
        rec["status"] = "cached"
        return rec

    kwargs = {k: v for k, v in params.items() if k != "hash"}
    src = _WORKER.format(repo=str(REPO_ROOT), parc=parc, density=density, donors=DONORS)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        worker = fh.name

    try:
        res = subprocess.run(
            [sys.executable, worker, _json.dumps(kwargs), str(dest)],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if res.returncode == 0 and dest.exists():
            rec["status"] = "ok"
            shape = [ln for ln in res.stdout.splitlines() if ln.startswith("SHAPE")]
            if shape:
                rec["shape"] = [int(x) for x in shape[0].split()[1:]]
        elif res.returncode < 0:
            rec["status"] = f"killed: signal {-res.returncode} (likely out of memory)"
            logger.warning("cell %s killed by signal %d", params["hash"], -res.returncode)
        else:
            tail = (res.stderr or "").strip().splitlines()
            rec["status"] = f"failed: {tail[-1] if tail else 'unknown'}"[:200]
            logger.warning("cell %s failed", params["hash"])
    except subprocess.TimeoutExpired:
        rec["status"] = "failed: timeout after 1800s"
    finally:
        Path(worker).unlink(missing_ok=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="run only the first N cells")
    ap.add_argument(
        "--parcellation",
        default=None,
        help="override the primary parcellation (§7.1 sensitivity analyses)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.logging.level, format=cfg.logging.format)

    parc = args.parcellation or cfg.parcellation.primary.name
    density = cfg.parcellation.primary.density
    # One directory per parcellation: the cell hash covers abagen parameters
    # only, so two parcellations would otherwise collide on the same filenames
    # and silently serve each other's cached matrices.
    sub = "multiverse" if parc == cfg.parcellation.primary.name else f"multiverse_{parc}"
    out_dir = cfg.path("expression") / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    atlas = gifti_atlas_paths(parc, density)

    grid = cells()[: args.limit]
    logger.info("multiverse @ %s: %d cells, %d jobs", parc, len(grid), args.n_jobs)

    from joblib import Parallel, delayed

    recs = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(run_cell)(p, atlas, out_dir, parc, density) for p in grid
    )

    idx = pd.DataFrame(recs)
    idx_path = out_dir / "multiverse_index.csv"
    idx.to_csv(idx_path, index=False)

    ok = idx.status.isin(["ok", "cached"])
    with manifest(f"p3_multiverse_{parc}", cfg) as man:
        man.record(
            parcellation=parc,
            n_cells=len(idx),
            n_ok=int(ok.sum()),
            n_failed=int((~ok).sum()),
            grid={k: [str(x) for x in v] for k, v in GRID.items()},
            donors=DONORS,
            index=str(idx_path),
        )
        man.note(
            "Five donors, not six: AHBA donor 15496 is unavailable upstream. "
            "Failed cells are recorded rather than dropped — which parameter "
            "combinations abagen refuses is itself part of the multiverse."
        )

    print(f"\n{'=' * 64}\nMULTIVERSE\n{'=' * 64}")
    print(f"  cells      {len(idx)}")
    print(f"  succeeded  {int(ok.sum())}")
    print(f"  failed     {int((~ok).sum())}")
    if (~ok).any():
        print("\n  failures by reason:")
        print(idx[~ok].status.str.slice(0, 60).value_counts().to_string())
    print(f"\n  index -> {idx_path}\n{'=' * 64}")
    with (out_dir / "grid.json").open("w") as fh:
        json.dump(GRID, fh, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
