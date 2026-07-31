#!/bin/bash
# Pull results FROM a compute host, and verify they came from one code state.
#
# A results directory spanning several git SHAs is the drift this project keeps
# reintroducing. This refuses to stay quiet about it.
set -euo pipefail
HOST="${1:?usage: pull_results.sh <host>}"
cd "$(dirname "$0")/.."
rsync -a "$HOST:~/discordance-transcriptomics/results/" results/
.venv/bin/python - <<'PY'
import json, pathlib
from collections import Counter
shas, dates = Counter(), Counter()
for f in sorted(pathlib.Path("results").glob("*.manifest.json")):
    d = json.load(open(f))
    shas[d.get("git_sha", "?")[:8]] += 1
    dates[d.get("created_utc", "?")[:10]] += 1
print("git SHAs present:")
for s, n in shas.most_common():
    print(f"  {s}  {n} artifacts")
print("dates present:")
for s, n in sorted(dates.items()):
    print(f"  {s}  {n} artifacts")
if len(shas) > 1:
    print("\nWARNING: results span multiple code states. Not a clean run.")
PY
