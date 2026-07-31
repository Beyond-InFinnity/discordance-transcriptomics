"""Reading the expression multiverse — shared by every phase that consumes it.

The multiverse index records one row per processing cell. Its ``path`` column
holds whatever absolute path the machine that ran Phase 3 happened to use, which
does not survive the index being copied between hosts — and this project
routinely computes the grid on one machine and analyses it on another.

Phase 3 now writes a repo-relative path, but indexes generated before that
change still carry absolute ones, and three separate scripts had each grown
their own resolution logic (two identical, one missing entirely, which is how a
regeneration run died partway through). The hash is the canonical key and the
filename is a pure function of it, so resolution belongs in one place.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["cell_path", "load_index", "multiverse_dir", "select_cells"]


def multiverse_dir(cfg, parcellation: str | None = None) -> Path:
    """Directory holding a parcellation's multiverse cells.

    The primary parcellation lives in ``multiverse/``; sensitivity
    parcellations get their own suffixed directory, because the cell hash covers
    abagen parameters only and two parcellations would otherwise collide on
    identical filenames and silently serve each other's matrices.
    """
    primary = cfg.parcellation.primary.name
    parc = parcellation or primary
    sub = "multiverse" if parc == primary else f"multiverse_{parc}"
    return cfg.path("expression") / sub


def cell_path(mv_dir: Path, cell) -> Path:
    """Locate one cell's parquet, preferring the hash over any stored path."""
    dest = Path(mv_dir) / f"expr_{cell['hash']}.parquet"
    if dest.exists():
        return dest
    stored = Path(str(cell.get("path", "")))
    if stored.exists():
        return stored
    raise FileNotFoundError(
        f"no parquet for cell {cell['hash']} under {mv_dir}. If the index was "
        "generated on another machine its stored paths will not resolve here; "
        "the file should be named expr_<hash>.parquet."
    )


def load_index(mv_dir: Path) -> pd.DataFrame:
    """Successful cells from a multiverse index, or a clear error."""
    idx_path = Path(mv_dir) / "multiverse_index.csv"
    if not idx_path.exists():
        raise FileNotFoundError(
            f"{idx_path} missing — run scripts/p3_multiverse.py for this parcellation"
        )
    idx = pd.read_csv(idx_path)
    return idx[idx.status.isin(["ok", "cached"])]


def select_cells(idx: pd.DataFrame, n_cells: int | None, primary: dict) -> pd.DataFrame:
    """Primary pipeline first, then an even spread across the rest.

    Analyses that cannot afford all 120 cells should still report the headline
    pipeline plus a representative sample rather than an arbitrary head slice,
    which would take every cell from one corner of the grid.
    """
    import numpy as np

    is_primary = np.ones(len(idx), dtype=bool)
    for k, v in primary.items():
        is_primary &= idx[k].astype(str) == str(v)
    head, rest = idx[is_primary], idx[~is_primary]
    if n_cells is None or n_cells >= len(idx):
        return pd.concat([head, rest])
    take = max(0, n_cells - len(head))
    step = max(1, len(rest) // take) if take else 1
    return pd.concat([head, rest.iloc[::step].head(take)])
