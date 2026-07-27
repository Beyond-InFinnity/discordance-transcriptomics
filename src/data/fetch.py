"""Dataset acquisition with checksum verification — CLAUDE.md R8.

OpenNeuro exposes each snapshot's file tree over GraphQL and serves the blobs
over plain HTTP, so we can enumerate and fetch selectively without datalad or
git-annex. Two things make this worth doing directly:

1. **Selectivity.** ds004873's fmriprep BOLD derivatives are ~1 GB *per subject*
   (~50 GB total) and Phase 0 does not need them. The MNI152 qmri maps that
   Phase 0 *does* need are ~800 kB each. A pattern-filtered fetch is ~250 MB
   instead of ~50 GB.

2. **Free checksums.** OpenNeuro returns git-annex keys of the form
   ``SHA256E-s<size>--<sha256>.nii.gz``. The expected SHA256 is embedded in the
   key, so every download can be verified against the server's own record
   rather than trusting the transfer.

⚠️ **Snapshot version matters.** The S3 mirror at ``s3://openneuro.org/ds004873``
serves snapshot 1.0.4, which contains *no derivatives at all* — only MESE, BOLD
and T1w. The mqBOLD derivatives this project depends on appear in 2.0.x. Always
pin the tag explicitly; see :data:`DEFAULT_SNAPSHOT`.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from ..utils.manifest import sha256_file

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_SNAPSHOT",
    "GRAPHQL_URL",
    "RemoteFile",
    "download_file",
    "fetch_files",
    "list_snapshot_files",
    "list_snapshots",
]

GRAPHQL_URL = "https://openneuro.org/crn/graphql"
FILE_URL = "https://openneuro.org/crn/datasets/{ds}/snapshots/{tag}/files/{path}"

# Pinned deliberately. 1.0.4 (what the S3 mirror serves) has no derivatives.
DEFAULT_SNAPSHOT = "2.0.7"

# Annex keys look like: SHA256E-s822572--<64 hex>.nii.gz
_ANNEX_RE = re.compile(r"SHA256E-s(\d+)--([0-9a-f]{64})")


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Compile a path-aware glob.

    ``fnmatch`` is unusable here: its ``*`` happily matches ``/``, so
    ``derivatives/*.nii.gz`` silently matches every NIfTI at any depth (which
    on ds004873 is 60 GB rather than the intended handful). This treats ``*``
    as "anything except a separator" and ``**`` as "anything".
    """
    out, i = [], 0
    while i < len(pattern):
        c = pattern[i]
        if c == "*":
            if pattern[i : i + 2] == "**":
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(c))
        i += 1
    return re.compile("".join(out) + r"\Z")


def path_matches(path: str, patterns: list[str]) -> bool:
    """True if ``path`` matches any glob in ``patterns`` (separator-aware)."""
    return any(_glob_to_regex(p).match(path) for p in patterns)


@dataclass(frozen=True)
class RemoteFile:
    """One file in an OpenNeuro snapshot."""

    path: str
    size: int
    annex_key: str

    @property
    def expected_sha256(self) -> str | None:
        """SHA256 parsed out of the git-annex key, if it is a SHA256E key."""
        m = _ANNEX_RE.search(self.annex_key or "")
        return m.group(2) if m else None

    @property
    def url_path(self) -> str:
        """OpenNeuro's file endpoint uses ':' as the path separator."""
        return self.path.replace("/", ":")


def _graphql(query: str, timeout: int = 60) -> dict[str, Any]:
    req = urllib.request.Request(
        GRAPHQL_URL,
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.load(resp)
    if "errors" in payload and payload.get("data") is None:
        raise RuntimeError(f"GraphQL error: {payload['errors']}")
    return payload["data"]


def list_snapshots(dataset: str = "ds004873") -> list[str]:
    """Return available snapshot tags, oldest first."""
    data = _graphql(f'{{dataset(id:"{dataset}"){{snapshots{{tag}}}}}}')
    return [s["tag"] for s in data["dataset"]["snapshots"]]


def list_snapshot_files(
    dataset: str = "ds004873",
    tag: str = DEFAULT_SNAPSHOT,
    subtree: str = "",
) -> Iterator[RemoteFile]:
    """Recursively walk a snapshot's file tree.

    Parameters
    ----------
    dataset : str
        OpenNeuro accession, e.g. ``ds004873``.
    tag : str
        Snapshot tag. Pin this — see the module docstring.
    subtree : str
        Restrict the walk to a top-level directory (e.g. ``derivatives``) to
        avoid enumerating the whole dataset.

    Yields
    ------
    RemoteFile
    """

    def _walk(tree_id: str | None, prefix: str) -> Iterator[RemoteFile]:
        arg = f',tree:"{tree_id}"' if tree_id else ""
        q = (
            f'{{snapshot(datasetId:"{dataset}",tag:"{tag}")'
            f"{{files(NONE){{filename directory size id}}}}}}"
        ).replace("(NONE)", f"({arg.lstrip(',')})" if arg else "")
        data = _graphql(q)
        for entry in data["snapshot"]["files"] or []:
            name = entry["filename"]
            full = f"{prefix}{name}"
            if entry["directory"]:
                yield from _walk(entry["id"], f"{full}/")
            else:
                yield RemoteFile(
                    path=full, size=entry["size"] or 0, annex_key=entry["id"] or ""
                )

    if subtree:
        root = _graphql(
            f'{{snapshot(datasetId:"{dataset}",tag:"{tag}"){{files{{filename directory id}}}}}}'
        )
        match = [
            e
            for e in root["snapshot"]["files"]
            if e["filename"] == subtree and e["directory"]
        ]
        if not match:
            raise FileNotFoundError(f"{subtree!r} not found at root of {dataset}@{tag}")
        yield from _walk(match[0]["id"], f"{subtree}/")
    else:
        yield from _walk(None, "")


def download_file(
    rf: RemoteFile,
    dest: Path,
    dataset: str = "ds004873",
    tag: str = DEFAULT_SNAPSHOT,
    verify: bool = True,
    timeout: int = 300,
) -> bool:
    """Download one file, verifying its checksum against the annex key.

    Skips the download when a local file already matches the expected SHA256,
    so re-running is cheap and idempotent.

    Returns
    -------
    bool
        True if the file is present and verified after the call.
    """
    expected = rf.expected_sha256
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and expected and sha256_file(dest) == expected:
        logger.debug("already verified, skipping %s", rf.path)
        return True

    url = FILE_URL.format(ds=dataset, tag=tag, path=rf.url_path)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as resp, tmp.open("wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)

    if verify and expected:
        got = sha256_file(tmp)
        if got != expected:
            tmp.unlink(missing_ok=True)
            raise ValueError(
                f"checksum mismatch for {rf.path}\n  expected {expected}\n  got      {got}"
            )
    tmp.replace(dest)
    logger.info("fetched %s (%.1f MB)", rf.path, dest.stat().st_size / 1e6)
    return True


def fetch_files(
    patterns: list[str],
    out_root: Path,
    dataset: str = "ds004873",
    tag: str = DEFAULT_SNAPSHOT,
    subtree: str = "derivatives",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Fetch every file in a snapshot whose path matches any glob in ``patterns``.

    Parameters
    ----------
    patterns : list of str
        fnmatch globs against the snapshot-relative path, e.g.
        ``["derivatives/*/qmri/*space-MNI152*"]``.
    out_root : Path
        Local destination root; snapshot paths are recreated beneath it.
    dataset, tag, subtree
        See :func:`list_snapshot_files`.
    dry_run : bool
        Enumerate and report totals without downloading.

    Returns
    -------
    dict
        Summary with matched count, total bytes, and any failures.
    """
    matched = [
        rf
        for rf in list_snapshot_files(dataset, tag, subtree)
        if path_matches(rf.path, patterns)
    ]
    total = sum(rf.size for rf in matched)
    logger.info(
        "%d files match (%.2f GB)%s",
        len(matched),
        total / 1e9,
        " [dry run]" if dry_run else "",
    )

    summary: dict[str, Any] = {
        "dataset": dataset,
        "snapshot": tag,
        "patterns": patterns,
        "n_matched": len(matched),
        "total_bytes": total,
        "failures": [],
        "files": [rf.path for rf in matched],
    }
    if dry_run:
        return summary

    for i, rf in enumerate(matched, 1):
        try:
            download_file(rf, out_root / rf.path, dataset, tag)
        except Exception as exc:  # noqa: BLE001 — collected and reported
            logger.error("FAILED %s: %s", rf.path, exc)
            summary["failures"].append({"path": rf.path, "error": str(exc)})
        if i % 25 == 0:
            logger.info("progress %d/%d", i, len(matched))

    summary["n_ok"] = len(matched) - len(summary["failures"])
    return summary
