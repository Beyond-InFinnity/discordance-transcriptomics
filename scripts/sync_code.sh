#!/bin/bash
# Push CODE to a compute host. Never results.
#
# results/ is git-tracked, so a plain `rsync ./ host:` carries it — and
# overwrites whatever that host just computed with the laptop's committed
# copies. That happened repeatedly on 2026-07-31: a full regeneration completed
# at 17:01 and was silently reverted by the next code sync, leaving results/ a
# mix of four different dates that looked like a clean run.
#
# Results move one way only: compute host -> laptop, via scripts/pull_results.sh.
set -euo pipefail
HOST="${1:?usage: sync_code.sh <host>}"
cd "$(dirname "$0")/.."
rsync -a --delete-excluded \
  --exclude='/.venv' --exclude='/data' --exclude='/.git' \
  --exclude='/results' --exclude='__pycache__' --exclude='*.log' \
  ./ "$HOST:~/discordance-transcriptomics/"
echo "code -> $HOST (results/ deliberately excluded)"
