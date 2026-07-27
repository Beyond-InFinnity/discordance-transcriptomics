"""Locate Connectome Workbench and put it on PATH — CLAUDE.md §4.1.

``neuromaps`` shells out to ``wb_command`` for every fsLR transform, via
``subprocess`` with ``/bin/sh``. That subprocess inherits the *Python process's*
environment, not your interactive shell's — so adding Workbench to
``.bashrc``/``config.fish`` makes it work at a terminal prompt and still fail
from inside a script. The failure surfaces as::

    SubprocessError: Command failed with non-zero exit status 127.
    Error traceback: "/bin/sh: 1: wb_command: not found"

which does not obviously point at PATH. §4.1 calls Workbench the most common
setup failure; this is the specific trap.

:func:`ensure_workbench` searches PATH, then ``$WORKBENCH_DIR``, then the usual
install locations, and prepends whatever it finds to ``os.environ['PATH']`` so
child processes inherit it.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["WorkbenchNotFoundError", "ensure_workbench", "workbench_version"]

# Checked in order. $WORKBENCH_DIR wins, then conventional install locations.
_CANDIDATE_DIRS = (
    "~/opt/workbench/bin_linux64",
    "~/workbench/bin_linux64",
    "/opt/workbench/bin_linux64",
    "/usr/local/workbench/bin_linux64",
    "/Applications/workbench/bin_macosx64",
)


class WorkbenchNotFoundError(RuntimeError):
    """Raised when wb_command cannot be located anywhere."""


def ensure_workbench(raise_if_missing: bool = True) -> str | None:
    """Ensure ``wb_command`` is callable from subprocesses.

    Idempotent: returns immediately if it is already resolvable.

    Parameters
    ----------
    raise_if_missing : bool
        Raise :class:`WorkbenchNotFoundError` rather than returning None.

    Returns
    -------
    str or None
        Path to the ``wb_command`` executable.

    Raises
    ------
    WorkbenchNotFoundError
        With install instructions, when not found and ``raise_if_missing``.
    """
    found = shutil.which("wb_command")
    if found:
        return found

    candidates = []
    if env_dir := os.environ.get("WORKBENCH_DIR"):
        candidates.append(env_dir)
    candidates.extend(_CANDIDATE_DIRS)

    for raw in candidates:
        d = Path(raw).expanduser()
        exe = d / "wb_command"
        if exe.is_file() and os.access(exe, os.X_OK):
            os.environ["PATH"] = f"{d}{os.pathsep}{os.environ.get('PATH', '')}"
            logger.info("added Connectome Workbench to PATH: %s", d)
            return str(exe)

    if raise_if_missing:
        raise WorkbenchNotFoundError(
            "wb_command not found. neuromaps needs Connectome Workbench for every\n"
            "fsLR transform, and it must be on PATH *for this Python process* —\n"
            "adding it to .bashrc or config.fish only covers interactive shells.\n\n"
            "Install (Linux):\n"
            "  wget https://humanconnectome.org/storage/app/media/workbench/"
            "workbench-linux64-v1.5.0.zip\n"
            "  unzip workbench-linux64-v1.5.0.zip -d ~/opt/\n\n"
            "Then either export it before running:\n"
            '  export PATH="$HOME/opt/workbench/bin_linux64:$PATH"\n'
            "or point this helper at it:\n"
            '  export WORKBENCH_DIR="$HOME/opt/workbench/bin_linux64"\n'
        )
    logger.warning("Connectome Workbench not found; fsLR transforms will fail")
    return None


def workbench_version() -> str | None:
    """Version string of the resolved wb_command, or None if unavailable."""
    exe = ensure_workbench(raise_if_missing=False)
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "-version"], capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    for line in out.stdout.splitlines():
        if line.strip().startswith("Version:"):
            return line.split(":", 1)[1].strip()
    return None
