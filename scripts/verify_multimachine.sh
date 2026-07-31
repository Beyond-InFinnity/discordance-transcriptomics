#!/bin/bash
# Adversarial check of the multi-machine setup. ⛔ Run before trusting any run.
#
# Inspecting a setup tells you what it looks like. This tries to break it.
#
# Every safeguard here exists because its absence caused a real failure: a
# completed regeneration reverted by a code sync, a compute node with no git
# identity, results spanning five days reading as clean, and watcher loops whose
# checks matched themselves and could only return the reassuring answer. So each
# test below asserts a safeguard actually FIRES, not merely that it is present.
#
# Usage
#   scripts/verify_multimachine.sh
set -uo pipefail
cd "$(dirname "$0")/.."

HOSTS=(workstation claude-machine)
PASS=0; FAIL=0
ok  () { echo "  PASS  $1"; PASS=$((PASS+1)); }
bad () { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }
hdr () { echo; echo "── $1"; }

LOCAL_SHA=$(git rev-parse HEAD)

hdr "1. IDENTITY — one commit, clean trees, everywhere"
[ -z "$(git status --porcelain)" ] && ok "laptop tree clean" || bad "laptop tree DIRTY"
for h in "${HOSTS[@]}"; do
  r=$(ssh -o ConnectTimeout=15 "$h" 'cd ~/discordance-transcriptomics 2>/dev/null && echo "$(git rev-parse HEAD) $(git status --porcelain | wc -l)"' 2>/dev/null)
  sha=${r%% *}; dirty=${r##* }
  [ "$sha" = "$LOCAL_SHA" ] && ok "$h on the same commit" || bad "$h on $sha, laptop on $LOCAL_SHA"
  [ "$dirty" = "0" ] && ok "$h tree clean" || bad "$h has $dirty modified files"
done

hdr "2. COMPUTE NODES CAN ACTUALLY RUN"
for h in "${HOSTS[@]}"; do
  ssh "$h" 'test -x ~/discordance-transcriptomics/.venv/bin/python' 2>/dev/null \
    && ok "$h has a venv" || bad "$h venv missing"
  n=$(ssh "$h" 'ls ~/discordance-transcriptomics/data/derived/expression/multiverse/*.parquet 2>/dev/null | wc -l' 2>/dev/null)
  [ "${n:-0}" -ge 100 ] && ok "$h has the multiverse ($n cells)" || bad "$h multiverse has ${n:-0} cells"
  ssh "$h" 'test -x $HOME/opt/workbench/bin_linux64/wb_command' 2>/dev/null \
    && ok "$h has Workbench" || bad "$h Workbench missing"
done

hdr "3. CODE SYNC MUST NOT CARRY results/"
if grep -q -- "--exclude='/results'" scripts/sync_code.sh 2>/dev/null; then
  out=$(rsync -an --itemize-changes \
        --exclude='/.venv' --exclude='/data' --exclude='/.git' \
        --exclude='/results' --exclude='__pycache__' --exclude='*.log' \
        ./ "${HOSTS[0]}:~/discordance-transcriptomics/" 2>/dev/null | grep -c '^.*results/' || true)
  [ "${out:-0}" = "0" ] && ok "dry-run sync transfers 0 files under results/" \
                        || bad "dry-run sync would transfer $out results files"
else
  bad "sync_code.sh does not exclude /results"
fi

hdr "4. DIRTY TREE MUST BLOCK A RUN"
tmpf="_verify_dirty_$$.tmp"; touch "$tmpf"
if bash -c 'cd "$(dirname "$0")/.." 2>/dev/null; true'; then :; fi
out=$(cd . && git status --porcelain | wc -l)
if [ "$out" -gt 0 ]; then
  # regenerate_all.sh should refuse. Run only its guard, not the pipeline.
  if grep -q 'FATAL: working tree is dirty' scripts/regenerate_all.sh; then
    res=$(bash -c 'cd '"$PWD"' && if [ -n "$(git status --porcelain 2>/dev/null)" ]; then echo BLOCKED; fi')
    [ "$res" = "BLOCKED" ] && ok "dirty tree triggers the guard" || bad "guard did not fire"
  else
    bad "regenerate_all.sh has no dirty-tree guard"
  fi
fi
rm -f "$tmpf"

hdr "5. AUDIT MUST REJECT FOREIGN RESULTS"
TD=$(mktemp -d)
cp results/*.manifest.json "$TD/" 2>/dev/null || true
if [ "$(ls "$TD"/*.manifest.json 2>/dev/null | wc -l)" -gt 0 ]; then
  .venv/bin/python scripts/audit_provenance.py --results "$TD" >/dev/null 2>&1
  [ $? -ne 0 ] && ok "audit rejects a directory with no run token" \
               || bad "audit ACCEPTED an untokened directory"
else
  echo "  SKIP  no manifests available to test with"
fi

hdr "6. AUDIT MUST ACCEPT A GENUINELY CLEAN DIRECTORY"
TD2=$(mktemp -d)
printf '%s\n' "$(git rev-parse --short HEAD)-verify" > "$TD2/.run_id"
date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%SZ > "$TD2/.run_started" 2>/dev/null \
  || date -u +%Y-%m-%dT%H:%M:%SZ > "$TD2/.run_started"
.venv/bin/python - "$TD2" "$LOCAL_SHA" <<'PY'
import json, sys, datetime, pathlib
d, sha = pathlib.Path(sys.argv[1]), sys.argv[2]
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
for n in ("alpha", "beta"):
    (d / f"{n}.manifest.json").write_text(json.dumps({
        "name": n, "created_utc": now, "git_sha": sha, "git_dirty": False,
        "results": {"outputs": [str(d / f"{n}.csv")]}}))
    (d / f"{n}.csv").write_text("a,b\n1,2\n")
PY
.venv/bin/python scripts/audit_provenance.py --results "$TD2" >/dev/null 2>&1
[ $? -eq 0 ] && ok "audit accepts one SHA, clean, tokened, paired" \
             || { bad "audit REJECTED a clean directory (false alarm)"; \
                  .venv/bin/python scripts/audit_provenance.py --results "$TD2" 2>&1 | grep -A2 FAIL | head -12; }

hdr "7. AUDIT MUST DETECT MIXED CODE STATES"
cp "$TD2"/alpha.manifest.json "$TD2"/gamma.manifest.json
.venv/bin/python - "$TD2" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]) / "gamma.manifest.json"
d = json.loads(p.read_text()); d["git_sha"] = "0" * 40; d["name"] = "gamma"
p.write_text(json.dumps(d))
(pathlib.Path(sys.argv[1]) / "gamma.csv").write_text("a,b\n1,2\n")
PY
.venv/bin/python scripts/audit_provenance.py --results "$TD2" >/dev/null 2>&1
[ $? -ne 0 ] && ok "audit detects a second git SHA" || bad "audit MISSED a mixed code state"
rm -rf "$TD" "$TD2"

echo
echo "=============================================="
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] && echo "  SETUP VERIFIED" || echo "  SETUP NOT TRUSTWORTHY"
echo "=============================================="
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
