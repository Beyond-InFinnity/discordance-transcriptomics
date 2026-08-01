#!/bin/bash
# Pull results FROM a compute host, and verify they came from one code state.
#
# A results directory spanning several git SHAs is the drift this project keeps
# reintroducing. This refuses to stay quiet about it.
set -euo pipefail
HOST="${1:?usage: pull_results.sh <host>}"
cd "$(dirname "$0")/.."

# Refuse to pull from a run that is still going.
#
# This mirrors with --delete, so pulling mid-run replaces the laptop's complete
# set with whatever subset the compute node has written so far, and deletes the
# rest. The laptop is MASTER (§4.1a); this script is the one path by which its
# canonical copy is replaced, and it would happily overwrite three hours of good
# artifacts with a half-finished run and report success.
#
# regenerate_all.sh writes regen.done only after the provenance audit, so a
# regen.log with no regen.done beside it means "in flight".
remote_state=$(ssh -o ConnectTimeout=15 "$HOST" '
  cd ~/discordance-transcriptomics 2>/dev/null || exit 0
  running=$(pgrep -fc regenerate_all || true)
  if [ -f regen.log ] && [ ! -f regen.done ]; then echo "INFLIGHT";
  elif [ "${running:-0}" -gt 1 ]; then echo "INFLIGHT";
  else echo "IDLE"; fi' 2>/dev/null || echo "UNREACHABLE")
if [ "$remote_state" = "INFLIGHT" ]; then
  echo "REFUSING: a regeneration is still running on $HOST."
  echo "  This mirrors with --delete, so pulling now would replace the local"
  echo "  results/ with a partial run and delete the rest. Wait for regen.done."
  exit 1
fi
# --delete, so this MIRRORS rather than merges.
#
# Without it, rsync leaves local files that the remote does not have, so a fresh
# run's 14 artifacts land beside whatever the laptop was already holding and the
# directory becomes a blend of runs — the exact pollution this script exists to
# prevent. The audit caught it, but a tool that needs its own audit to catch its
# own bug is not doing its job.
rsync -a --delete "$HOST:~/discordance-transcriptomics/results/" results/

# The annotation table is deliverable #2 -- the reusable public artifact -- but
# it is written under data/derived/ because that is where it is built from, so
# it was never pulled. It lived only on the compute node while the laptop, which
# CLAUDE.md §4.1a designates MASTER, held a copy three days stale. Comparing
# that stale copy against a freshly pulled manifest reads as a corrupted
# artifact: the manifest recorded ahba_included=true while the local CSV had
# that column entirely empty.
#
# Mirrored separately rather than folded into the line above, because --delete
# on data/derived/ would destroy the expression multiverse and the warped maps.
rsync -a --delete "$HOST:~/discordance-transcriptomics/data/derived/annotation/" \
  data/derived/annotation/
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
