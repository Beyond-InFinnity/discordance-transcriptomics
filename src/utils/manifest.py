"""Result manifests — CLAUDE.md R10.

Every artifact written to ``results/`` is accompanied by ``<name>.manifest.json``
recording git SHA, config hash, package versions, seed, wall-clock time, and
input file checksums. Without this a result is not reproducible and, per R10,
not reportable.

Typical use::

    with manifest("p0_dropout", cfg, inputs=[epi_path]) as man:
        r, p = run_analysis()
        man.record(spearman_r=r, spatial_null_p=p)
    # writes results/p0_dropout.manifest.json on clean exit
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, BaseConfig, config_hash

logger = logging.getLogger(__name__)

__all__ = ["Manifest", "git_sha", "manifest", "sha256_file"]

# Packages whose versions materially affect numerical results.
_TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "nibabel",
    "nilearn",
    "abagen",
    "neuromaps",
    "netneurotools",
    "brainspace",
    "statsmodels",
    "scikit-learn",
    "pingouin",
)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA256 of a file, streamed so large NIfTIs don't blow up memory."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def git_sha(short: bool = False) -> str | None:
    """Current git commit SHA, or None if not in a git repo.

    Returns None rather than raising: a manifest with ``git_sha: null`` is an
    honest record of an uncommitted working tree, which is better than a
    crash mid-analysis.
    """
    cmd = ["git", "rev-parse", "--short" if short else "HEAD"]
    try:
        out = subprocess.run(
            cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("not a git repository — manifest will record git_sha=null")
        return None
    return out.stdout.strip()


def _git_dirty() -> bool | None:
    """True if the working tree has uncommitted changes."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return bool(out.stdout.strip())


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    return versions


@dataclass
class Manifest:
    """Accumulates provenance for one result artifact."""

    name: str
    seed: int
    config_hash: str | None = None
    inputs: dict[str, str] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    _start: float = field(default_factory=time.perf_counter, repr=False)

    def record(self, **kwargs: Any) -> None:
        """Attach result values to the manifest."""
        self.results.update(kwargs)

    def note(self, message: str) -> None:
        """Attach a free-text note (gate outcomes, caveats, deviations)."""
        self.notes.append(message)

    def add_input(self, path: str | Path) -> None:
        """Checksum an input file and record it."""
        p = Path(path)
        self.inputs[str(p)] = sha256_file(p) if p.is_file() else "MISSING"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_sha": git_sha(),
            "git_dirty": _git_dirty(),
            "config_hash": self.config_hash,
            "seed": self.seed,
            "wall_clock_sec": round(time.perf_counter() - self._start, 3),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
            "inputs": self.inputs,
            "results": self.results,
            "notes": self.notes,
        }

    def write(self, results_dir: str | Path | None = None) -> Path:
        """Write ``<name>.manifest.json`` and return its path."""
        out_dir = Path(results_dir) if results_dir else REPO_ROOT / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self.name}.manifest.json"
        with out_path.open("w") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
        logger.info("wrote manifest %s", out_path)
        return out_path


@contextmanager
def manifest(
    name: str,
    cfg: BaseConfig,
    inputs: Sequence[str | Path] = (),
    results_dir: str | Path | None = None,
) -> Iterator[Manifest]:
    """Context manager that writes a manifest on clean exit.

    On an exception the manifest is still written, with the traceback recorded,
    so a failed run leaves evidence rather than nothing.

    Parameters
    ----------
    name : str
        Artifact name; the manifest lands at ``results/<name>.manifest.json``.
    cfg : BaseConfig
        Config, hashed and stamped into the manifest.
    inputs : sequence of path
        Input files to checksum.
    results_dir : path, optional
        Override the output directory.
    """
    man = Manifest(name=name, seed=cfg.seed, config_hash=config_hash(cfg))
    for path in inputs:
        man.add_input(path)
    try:
        yield man
    except Exception as exc:
        man.note(f"FAILED: {type(exc).__name__}: {exc}")
        man.write(results_dir)
        raise
    man.write(results_dir)
