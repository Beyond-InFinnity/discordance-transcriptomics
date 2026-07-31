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

step () {
  echo ""
  echo "=============================================================="
  echo "  $1"
  echo "  started $(date +%H:%M:%S)"
  echo "=============================================================="
  shift
  "$@"
}

# Prerequisites that are NOT regenerated: the multiverse parquets and the
# per-donor matrices. Fail early and loudly rather than half-running.
for d in data/derived/expression/multiverse; do
  n=$(ls "$d"/*.parquet 2>/dev/null | wc -l)
  [ "$n" -ge 100 ] || { echo "FATAL: $d has $n parquets; run p3_multiverse.py first"; exit 1; }
done

# --- targets first: everything downstream reads them ---------------------
step "Phase 2 — target maps (3 parcellations)"        $PY scripts/p2_build_targets.py

# --- gates ----------------------------------------------------------------
step "Phase 0a — reliability gate"                    $PY scripts/p0_reliability.py
step "Phase 0b — dropout gate"                        $PY scripts/p0_dropout.py
step "Phase 0  — dynamic range / detectability"       $PY scripts/p0_dynamic_range.py
step "Phase 0c — gene-set map reliability"            $PY scripts/p0c_geneset_reliability.py

# --- cross-species positive control ---------------------------------------
step "x1 — macaque vascular control"                  $PY scripts/x1_macaque_vascular.py

# --- the analysis proper --------------------------------------------------
step "Phase 4  — frozen gene sets, both nulls"        $PY scripts/p4_genesets.py --n-draws $NDRAWS $P4_CELLS
step "Phase 4b — data-driven arm"                     $PY scripts/p4b_datadriven.py --max-cells $CELLS --n-draws $NDRAWS
step "Phase 5  — hierarchy control (DECISIVE)"        $PY scripts/p5_hierarchy.py --max-cells $CELLS
step "Phase 6  — mediation"                           $PY scripts/p6_mediation.py --n-boot $NBOOT $P4_CELLS

# --- released artifacts ---------------------------------------------------
step "Annotation table"                               $PY scripts/build_annotation.py --ahba
step "Figures"                                        $PY scripts/make_figures.py

echo ""
echo "=============================================================="
echo "  REGENERATION COMPLETE — $(date +%H:%M:%S)"
echo "  every artifact in results/ now comes from one code state"
echo "=============================================================="
