#!/bin/bash
# Parcellation sensitivity multiverses (CLAUDE.md §7.1).
#
# Sequential rather than parallel, deliberately: a single abagen cell peaks near
# 7 GB, and two parcellations running at once would exceed the 31 GB host. DK-68
# goes first because it is much cheaper (34 left-hemisphere parcels against 200)
# and so surfaces any configuration error in minutes rather than hours.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== DK-68 (34 LH parcels) ==="
.venv/bin/python scripts/p3_multiverse.py --parcellation dk68 --n-jobs 3

echo "=== Schaefer-400 (200 LH parcels) ==="
.venv/bin/python scripts/p3_multiverse.py --parcellation schaefer400x7 --n-jobs 2

echo "=== TIER 1 COMPLETE ==="
