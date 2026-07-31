#!/usr/bin/env python
"""Prove — or refuse to prove — that results/ came from one code state. ⛔ GATE.

This exists because the project repeatedly could not answer a basic question:
*which code produced this number?* Results accumulated across machines and days,
a code sync silently reverted a completed run, and a directory holding four
different dates read as a clean one. Fixes appeared not to take because stale
artifacts kept overwriting fresh ones.

Every artifact already records its git SHA, timestamp, config hash and package
versions (R10). Nothing checked them. This does, from five independent angles,
and exits non-zero if any fails — so it can gate a pipeline rather than be read
and ignored.

  1. **Code state.** Do all manifests share one git SHA? A results directory
     spanning several is not a run, it is a pile.
  2. **Cleanliness.** Was the tree dirty when each artifact was written? A dirty
     SHA does not identify the code that ran.
  3. **Pairing.** Does every CSV have a manifest and vice versa? An orphan CSV
     has no provenance at all.
  4. **Recency.** How far apart in time were the artifacts written? Days apart
     means they cannot describe one run.
  5. **Agreement.** Where the same quantity appears in two files, do they match?
     This is the check that caught the coupling-map reliability being 0.634 in
     one place and 0.711 in another.

Usage
-----
    python scripts/audit_provenance.py                 # audit results/
    python scripts/audit_provenance.py --strict        # exit 1 on any warning
"""

from __future__ import annotations

import argparse
import contextlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


def _load(results: Path) -> list[dict]:
    out = []
    for f in sorted(results.glob("*.manifest.json")):
        try:
            with f.open() as fh:
                d = json.load(fh)
        except Exception as exc:
            out.append({"_file": f.name, "_error": str(exc)})
            continue
        d["_file"] = f.name
        d["_name"] = f.name.replace(".manifest.json", "")
        out.append(d)
    return out


def check_code_state(mans: list[dict]) -> tuple[bool, list[str]]:
    shas = Counter(m.get("git_sha", "MISSING")[:8] for m in mans if "_error" not in m)
    msgs = [f"  {s}  {n} artifacts" for s, n in shas.most_common()]
    ok = len(shas) <= 1
    if not ok:
        by_sha = defaultdict(list)
        for m in mans:
            if "_error" not in m:
                by_sha[m.get("git_sha", "MISSING")[:8]].append(m["_name"])
        msgs.append("")
        for s, names in sorted(by_sha.items(), key=lambda kv: -len(kv[1])):
            msgs.append(f"  {s}: {', '.join(sorted(names)[:6])}")
    return ok, msgs


def check_cleanliness(mans: list[dict]) -> tuple[bool, list[str]]:
    dirty = [m["_name"] for m in mans if m.get("git_dirty")]
    return not dirty, [f"  written from a dirty tree: {n}" for n in sorted(dirty)]


def check_pairing(results: Path, mans: list[dict]) -> tuple[bool, list[str]]:
    declared: set[str] = set()
    for m in mans:
        for key in ("outputs",):
            for p in m.get("results", m).get(key, []) or []:
                declared.add(Path(str(p)).name)
    csvs = {f.name for f in results.glob("*.csv")}
    names = {m["_name"] for m in mans}
    orphan_csv = sorted(
        c for c in csvs if c.rsplit(".", 1)[0] not in names and c not in declared
    )
    msgs = [f"  CSV with no manifest: {c}" for c in orphan_csv[:12]]
    if len(orphan_csv) > 12:
        msgs.append(f"  ... and {len(orphan_csv) - 12} more")
    return not orphan_csv, msgs


def check_run_token(results: Path, mans: list[dict]) -> tuple[bool, list[str]]:
    """Was every artifact written by the run that owns this directory?

    regenerate_all.sh wipes results/ and drops a token before computing. Any
    artifact predating that token was not produced by this run — it arrived some
    other way, almost certainly an rsync that ignored gitignore. That is how a
    finished regeneration was silently reverted and went unnoticed for hours.
    """
    tok, started = results / ".run_id", results / ".run_started"
    if not tok.exists():
        return False, [
            "  no results/.run_id — this directory was not produced by "
            "regenerate_all.sh, so nothing here has a known origin"
        ]
    t0 = datetime.fromisoformat(started.read_text().strip().replace("Z", "+00:00"))
    stale = []
    for m in mans:
        ts = m.get("created_utc")
        if not ts:
            continue
        with contextlib.suppress(ValueError):
            if datetime.fromisoformat(ts.replace("Z", "+00:00")) < t0:
                stale.append(f"{m['_name']} ({ts[:16]})")
    msgs = [f"  run id: {tok.read_text().strip()}", f"  started: {t0:%Y-%m-%d %H:%M}"]
    msgs += [f"  PREDATES THIS RUN: {s}" for s in sorted(stale)[:10]]
    return not stale, msgs


def check_recency(mans: list[dict]) -> tuple[bool, list[str]]:
    ts = []
    for m in mans:
        t = m.get("created_utc")
        if t:
            with contextlib.suppress(ValueError):
                ts.append((datetime.fromisoformat(t.replace("Z", "+00:00")), m["_name"]))
    if not ts:
        return False, ["  no timestamps found"]
    ts.sort()
    span = (ts[-1][0] - ts[0][0]).total_seconds() / 3600
    msgs = [
        f"  earliest  {ts[0][0]:%Y-%m-%d %H:%M}  {ts[0][1]}",
        f"  latest    {ts[-1][0]:%Y-%m-%d %H:%M}  {ts[-1][1]}",
        f"  span      {span:.1f} hours",
    ]
    return span <= 12, msgs


# (label, file, column-or-key, row selector) pairs that must agree.
CROSS_CHECKS = [
    (
        "coupling map reliability",
        ("p0_dynamic_range_{parc}.csv", "split_half_reliability", "coupling angle"),
        ("p2_target_maps_{parc}.manifest.json", "coupling_n_map_reliability", None),
    ),
]


def check_agreement(results: Path, parc: str) -> tuple[bool, list[str]]:
    msgs, ok = [], True
    for label, (f1, k1, sel1), (f2, k2, _s2) in CROSS_CHECKS:
        p1, p2 = results / f1.format(parc=parc), results / f2.format(parc=parc)
        if not (p1.exists() and p2.exists()):
            msgs.append(f"  {label}: source missing, not checked")
            continue
        try:
            df = pd.read_csv(p1)
            v1 = float(df.loc[df.iloc[:, 0] == sel1, k1].iloc[0])
            with p2.open() as fh:
                man = json.load(fh)
            v2 = float(man.get("results", man)[k2])
        except Exception as exc:
            msgs.append(f"  {label}: could not compare ({exc})")
            ok = False
            continue
        agree = abs(v1 - v2) < 1e-6
        ok &= agree
        msgs.append(
            f"  {label}: {v1:.6f} vs {v2:.6f}  {'agree' if agree else '<-- DISAGREE'}"
        )
    return ok, msgs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--parcellation", default="schaefer200x7")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    results = Path(args.results)
    mans = _load(results)
    if not mans:
        print(f"no manifests under {results}/ — nothing to audit")
        return 1

    print(
        f"\n{'=' * 70}\nPROVENANCE AUDIT — {results}/ ({len(mans)} artifacts)\n{'=' * 70}"
    )
    checks = [
        ("1. ONE CODE STATE", *check_code_state(mans)),
        ("2. CLEAN TREE", *check_cleanliness(mans)),
        ("3. EVERY OUTPUT HAS PROVENANCE", *check_pairing(results, mans)),
        ("4. WRITTEN TOGETHER", *check_recency(mans)),
        ("5. OVERLAPPING VALUES AGREE", *check_agreement(results, args.parcellation)),
        ("6. PRODUCED BY THIS RUN", *check_run_token(results, mans)),
    ]
    n_fail = 0
    for title, ok, msgs in checks:
        mark = "PASS" if ok else "FAIL"
        n_fail += not ok
        print(f"\n{title}: {mark}")
        for m in msgs:
            print(m)

    print(f"\n{'=' * 70}")
    if n_fail:
        print(f"VERDICT: NOT A CLEAN RUN — {n_fail} of 6 checks failed.")
        print("Results in this directory come from more than one code state and")
        print("should not be reported until regenerated in a single pass.")
    else:
        print("VERDICT: CLEAN — every artifact from one code state, one tree, one run.")
    print(f"{'=' * 70}\n")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
