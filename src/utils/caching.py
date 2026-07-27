"""Shared ``joblib.Memory`` cache — CLAUDE.md §10.

abagen calls, null-model generation, and multiverse cells are all expensive and
deterministic given their inputs, so they are cached by config hash. The cache
lives at ``data/.cache/`` and is gitignored.

Determinism matters here (R7): a cache hit must be indistinguishable from a
recomputation. Only cache functions whose output depends solely on their
arguments — anything reading a mutable file or an unseeded RNG will poison the
cache with a stale result that looks legitimate.
"""

from __future__ import annotations

import logging
from pathlib import Path

from joblib import Memory

from .config import REPO_ROOT

logger = logging.getLogger(__name__)

__all__ = ["clear_cache", "get_memory"]

_DEFAULT_CACHE = REPO_ROOT / "data" / ".cache"
_memory: Memory | None = None


def get_memory(location: str | Path | None = None, verbose: int = 0) -> Memory:
    """Return the process-wide joblib cache, creating it on first use.

    Parameters
    ----------
    location : path, optional
        Cache directory. Defaults to ``data/.cache``.
    verbose : int
        joblib verbosity.

    Returns
    -------
    joblib.Memory
    """
    global _memory
    if _memory is None or location is not None:
        loc = Path(location) if location is not None else _DEFAULT_CACHE
        loc.mkdir(parents=True, exist_ok=True)
        _memory = Memory(location=str(loc), verbose=verbose)
        logger.debug("joblib cache at %s", loc)
    return _memory


def clear_cache(location: str | Path | None = None) -> None:
    """Delete all cached results.

    Use after changing a cached function's implementation — joblib keys on the
    function's source, but not on code it calls, so a change one level down can
    otherwise leave stale entries in place.
    """
    mem = get_memory(location)
    mem.clear(warn=False)
    logger.info("cleared joblib cache")
