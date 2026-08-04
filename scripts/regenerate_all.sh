#!/bin/bash
# Regenerate every analysis artifact from one code state, in dependency order.
#
# Why this exists. Results in this project accumulated across days and code
# states, and an independent review found the reporting layer had drifted from
# the files it described — a mislabelled gene set, a stale README, a summary
# contradicting its own manifest. Each was a hand-maintained statement that no
# longer matched its artifact.
#
# The structural fix is to make a full regeneration cheap enough to be routine,
# so `results/` is never a mix of eras. On a 62 GB / 16-thread host this is
# about 3.5 hours. It deliberately does NOT rebuild the expression multiverse:
# that is 4 GB of pure abagen output, is untouched by any analysis-layer defect,
# and costs hours to recompute for no change.
#
# Usage
#   scripts/regenerate_all.sh              # everything
#   scripts/regenerate_all.sh --quick      # reduced permutation counts, for a
#                                          # smoke test of the whole chain
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
NDRAWS=10000
NBOOT=10000
CELLS=12
P4_CELLS=""          # empty = all 120
if [[ "${1:-}" == "--quick" ]]; then
  NDRAWS=200; NBOOT=200; CELLS=2
  # Cap the CELL COUNT too, not just the permutation counts. Phase 4 and 6 loop
  # every cell in the multiverse and that loop, not the null draws, is what
  # dominates their runtime — a first attempt at this reduced only the draws and
  # still took over an hour to reach cell 20 of 120, proving nothing about the
  # steps after it.
  P4_CELLS="--max-cells 3"
  echo "QUICK MODE — reduced permutations AND cells; results are not publishable"
fi

export WORKBENCH_DIR="${WORKBENCH_DIR:-$HOME/opt/workbench/bin_linux64}"

# A dirty tree makes every manifest's git SHA a lie: it names a commit that is
# not what ran. Ten artifacts were written that way before this check existed.
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
  echo "FATAL: working tree is dirty. Every artifact would record a git SHA that"
  echo "does not identify the code that produced it. Commit or stash first:"
  git status --short
  exit 1
fi
echo "code state: $(git rev-parse --short HEAD) (clean)"

# Own results/ outright, and stamp the run.
#
# Being gitignored stops git from carrying results between machines; it does
# nothing about rsync, which ignores gitignore entirely. A stray `rsync ./ host:`
# would still deposit the laptop's copies here, and that is precisely how a
# completed regeneration was silently reverted. So rather than trusting the
# operator to use sync_code.sh, this wipes results/ and writes a token naming
# the run. audit_provenance.py then verifies every artifact was written after
# that token, which makes foreign files detectable rather than invisible.
RUN_ID="$(git rev-parse --short HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"
# Own the completion marker rather than trusting the caller to append it.
# pull_results.sh treats regen.done as "this run finished and may be pulled",
# and a wrapper of the form `regenerate_all.sh; touch regen.done` -- semicolon,
# not && -- writes it even when the run aborts. That happened: a crash at step
# 17 of 18 was marked complete. set -e means anything below this line failing
# skips the touch at the end, so the marker now means what it says.
rm -f regen.done
rm -rf results && mkdir -p results
printf '%s\n' "$RUN_ID" > results/.run_id
date -u +%Y-%m-%dT%H:%M:%SZ > results/.run_started
echo "run id: $RUN_ID (results/ wiped; any pre-existing files discarded)"

# The dirty-tree check at the top of this script runs ONCE, before computing.
# That is not enough: a tree can be dirtied *during* a run, and the manifests
# written afterwards record git_dirty=true while every earlier one records
# false. The provenance audit catches it -- but only at the end, so the cost of
# noticing is the whole run. It happened here: two scratch scripts copied into
# the repo root mid-run left the last three artifacts dirty and forced a full
# 3.7-hour regeneration.
#
# Re-checking before each step turns that into a few seconds. `git status` on
# this repo takes ~30 ms against steps measured in minutes, so the check is free.
step () {
  if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    echo ""
    echo "FATAL: the working tree became dirty mid-run, before: $1"
    git status --porcelain | sed 's/^/  /'
    echo ""
    echo "Every manifest written from here on would record git_dirty=true and"
    echo "fail the provenance audit. Stopping now rather than at the end."
    exit 1
  fi
  echo ""
  echo "=============================================================="
  echo "  $1"
  echo "  started $(date +%H:%M:%S)"
  echo "=============================================================="
  shift
  "$@"
}

# Prerequisites that are NOT regenerated: the expression multiverse and the
# warped CBV-corrected CMRO2 maps. Both are upstream data preparation, cost
# hours, and are untouched by any analysis-layer defect. Fail early and loudly
# rather than half-running.
#
# Their manifests live beside their data rather than in results/, precisely
# because this script wipes results/ — provenance kept there would be destroyed
# by the regeneration that depends on it. Echo the code state that produced
# them, so a mismatch with the analysis SHA is visible rather than assumed away.
for d in data/derived/expression/multiverse; do
  n=$(ls "$d"/*.parquet 2>/dev/null | wc -l)
  [ "$n" -ge 100 ] || { echo "FATAL: $d has $n parquets; run p3_multiverse.py first"; exit 1; }
done
n=$(ls data/derived/warped/*.nii.gz 2>/dev/null | wc -l)
[ "$n" -ge 30 ] || { echo "FATAL: data/derived/warped has $n maps; run warp_cbv_cmro2.py first"; exit 1; }
for m in data/derived/expression/multiverse/p3_multiverse_*.manifest.json \
         data/derived/warped/warp_cbv_cmro2.manifest.json; do
  [ -f "$m" ] && echo "  upstream $(basename "$m" .manifest.json): $($PY -c \
    "import json,sys;print(json.load(open(sys.argv[1]))['git_sha'][:8])" "$m" 2>/dev/null || echo unknown)"
done

# --- targets first: everything downstream reads them ---------------------
step "Phase 2 — target maps (3 parcellations)"        $PY scripts/p2_build_targets.py

# --- gates ----------------------------------------------------------------
step "Phase 0a — reliability gate"                    $PY scripts/p0_reliability.py
step "Phase 0b — dropout gate"                        $PY scripts/p0_dropout.py
# The §9 gate was applied to the final maps only, and passed. mqBOLD is a chain
# (T2/T2' -> R2' -> OEF -> CMRO2 -> dCMRO2 -> discordance) and a gate on the last
# link cannot see corruption at the first, so the threshold is applied to every
# link. This is a gate, not an extra: it must run on every regeneration or the
# gate that actually constrains the project is missing from the record.
step "Phase 0b — dropout vs the whole mqBOLD chain (GATE)" $PY scripts/p0b_full_dropout_audit.py
step "Phase 0  — dynamic range / detectability"       $PY scripts/p0_dynamic_range.py
step "Phase 0c — gene-set map reliability"            $PY scripts/p0c_geneset_reliability.py

# --- positive controls ------------------------------------------------------
step "x1 — macaque vascular control"                  $PY scripts/x1_macaque_vascular.py
# Reads p0_dynamic_range's reliability, so it must follow it. Characterises the
# one positive control the project fails: our baseline CMRO2 against the Raichle
# PET reference. Omitting it from the regeneration would leave the failure
# documented only in prose.
step "x2 — CMRO2 positive-control audit"              $PY scripts/x2_cmro2_audit.py

# --- the analysis proper --------------------------------------------------
step "Phase 4  — frozen gene sets, both nulls"        $PY scripts/p4_genesets.py --n-draws $NDRAWS $P4_CELLS
# Which gene-set x outcome tests the design can resolve at all. Reads Phase 0c's
# reliabilities and Phase 4's effects, so it must follow both. This is the
# manuscript's central table: the detectability floor per test -- the smallest
# TRUE effect each could resolve. Reports floors, not a resolvability count:
# that count cancelled algebraically to the significance test (see p0d).
step "Phase 0d — resolvable tests (manuscript Table 1)" $PY scripts/p0d_resolvable_tests.py
step "Phase 4b — data-driven arm"                     $PY scripts/p4b_datadriven.py --max-cells $CELLS --n-draws $NDRAWS
# The same frozen sets and nulls as Phase 4, aggregating per gene instead of
# averaging expression first. Phase 0c shows the averaged score is the less
# reliable measurement for every large set, and this is what the same data
# supports without it. Runs on the FULL multiverse: the rotations do not depend
# on the gene, so all ~15,500 genes against 10,000 rotations collapse into one
# matrix product per cell x target -- minutes, not days.
step "Phase 4c — per-gene arm, full multiverse"       $PY scripts/p4c_pergene.py --n-draws $NDRAWS $P4_CELLS
# Whether a competitive-null result is spatial autocorrelation. The published
# null matches size and differential stability but not smoothness, and a spin
# test's conservativeness depends on the smoothness of both maps -- so a set of
# unusually smooth genes can look depleted for reasons unconnected to biology.
# Reads Phase 4c's per-gene statistics, so it must follow it.
step "x3 — autocorrelation-matched null"              $PY scripts/x3_autocorr_matched.py --n-draws $NDRAWS
# Whether Phase 4c's genome-wide clearance rate is a FALSE-POSITIVE rate. It is
# not: reading it as one assumes no gene is truly associated with the target,
# which is not a null this field believes. Rotating each gene independently
# preserves its autocorrelation while destroying its alignment, so the rate
# against rotated genes is the error rate with nothing else in it. Reads Phase
# 4c's published table for the side-by-side, so it must follow it.
step "x4 — null-gene calibration"                     $PY scripts/x4_null_genes.py --max-cells $CELLS
step "Phase 5  — hierarchy, pre-registered (DECISIVE)" $PY scripts/p5_hierarchy.py --max-cells $CELLS --covariates principal
# Disclosed sensitivity analysis, run every time rather than on request. The
# pre-registered specification removes only the FIRST connectivity gradient, but
# our maps track gradients 2 and 3 more strongly — the coupling angle sits at
# +0.04 against gradient 1 and +0.46/+0.49 against 2 and 3. Reporting only the
# weaker control would overstate what survives; reporting only the stronger one
# would abandon the pre-registration. Both, always, so neither can be chosen
# after the fact.
step "Phase 5b — hierarchy, extended (sensitivity)"   $PY scripts/p5_hierarchy.py --max-cells $CELLS --covariates extended
step "Phase 6  — mediation"                           $PY scripts/p6_mediation.py --n-boot $NBOOT $P4_CELLS

# --- released artifacts ---------------------------------------------------
step "Annotation table"                               $PY scripts/build_annotation.py --ahba
# The app's molecular layer. Writes beside the annotation table rather than into
# results/, because it is an app input rather than an analysis result -- and
# because this script wipes results/, so a manifest kept there would be
# destroyed by the run that produced it.
step "Gene-set profiles (app input)"                  $PY scripts/build_geneset_profiles.py
step "Figures"                                        $PY scripts/make_figures.py
# The five display items the manuscript argues with. make_figures.py predates the
# rewrite and depicts none of the resolvability analysis, which is now Figure 1.
step "Manuscript figures"                             $PY scripts/make_manuscript_figures.py

# Refuse to call it complete unless the provenance gate passes.
# Reported, not gated. A regeneration should not fail because prose is stale --
# but drift between results/ and the paper is this project's most persistent
# defect, and the moment a number moves is the moment to notice.
step "Paper numbers vs results/ (report only)" bash -c \
  "$PY scripts/check_paper_numbers.py || echo '  ^^ paper/draft.md is STALE against these results'"

step "Provenance audit (GATE)"                        $PY scripts/audit_provenance.py

touch regen.done
echo ""
echo "=============================================================="
echo "  REGENERATION COMPLETE — $(date +%H:%M:%S)"
echo "  every artifact in results/ now comes from one code state"
echo "=============================================================="
