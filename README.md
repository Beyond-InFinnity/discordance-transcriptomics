# discordance-transcriptomics

Testing whether the spatial topography of **BOLD/CMRO₂ discordance** in human
cortex is explained by **molecular vascular and metabolic architecture**, using
post-mortem transcriptomics (AHBA) as the explanatory layer.

Epp et al. (2025, *Nat Neurosci*, doi:10.1038/s41593-025-02132-9) showed that
~40% of voxels with significant task-evoked BOLD changes show oxygen metabolism
moving in the *opposite* direction, concentrated in the default mode network.
The first author's thesis speculates that association cortex has lower capillary
density and that this could produce weakened or reversed responses. **That
speculation has never been tested against molecular vascular architecture.**

- **H1** — discordance propensity tracks glycolytic and vascular-sparsity gene
  programs (negatively with oxidative phosphorylation), *over and above* the
  unimodal→transmodal cortical hierarchy.
- **H2** — `vascular/metabolic expression → baseline OEF/CBV → discordance`.

**`CLAUDE.md` is the specification.** Read it before writing any code; §3 (Hard
Rules) and §13 (Stop-and-Ask) are binding.

---

## Status

| Phase | State |
|---|---|
| Environment | ✅ done |
| Repo scaffold, config, core stats | ✅ done — 90 tests passing |
| ds004873 selective fetch | ✅ done (checksum-verified) |
| **Phase 0a** reliability gate | ⛔ blocked — see `docs/PHASE0_HANDOFF.md` |
| **Phase 0b** dropout gate | ⛔ blocked — same |
| Phase 1+ | not started |

**Read `docs/PHASE0_HANDOFF.md` first.** It records an open §13.5 decision:
ds004873's published derivatives contain 2 usable conditions, not the 4 the
protocol assumes.

---

## Setup

Requires Python 3.11 and Connectome Workbench.

```bash
# Connectome Workbench — needed by neuromaps for fsLR transforms.
# Install this FIRST; it is the most common setup failure.
wget https://humanconnectome.org/storage/app/media/workbench/workbench-linux64-v1.5.0.zip
unzip workbench-linux64-v1.5.0.zip -d ~/opt/
export PATH="$HOME/opt/workbench/bin_linux64:$PATH"
wb_command -version    # must succeed before proceeding

# Python
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

Two deviations from CLAUDE.md §4, both documented inline in `requirements.txt`:
`abagen` installs from its GitHub tag (0.1.4 was never published to PyPI), and
`setuptools<81` is pinned because abagen still imports `pkg_resources`.

---

## Layout

```
config/        base.yaml, multiverse.yaml, genesets.yaml (FROZEN — R5)
src/           importable, tested logic
  data/        fetch.py, targets.py, parcellate.py
  stats/       spatial.py (R1), reliability.py, competitive.py, ...
  utils/       config.py, manifest.py (R10), caching.py
scripts/       thin CLI wrappers, one per phase step
tests/         pytest
results/       each output + its .manifest.json
data/          GITIGNORED — provenance in data/MANIFEST.yaml
docs/          PHASE0_HANDOFF.md
```

`src/` holds logic, `scripts/` holds argparse wrappers, `notebooks/` are figures
only. If you are tempted to put a computation in a notebook, it belongs in `src/`.

---

## The three statistical guardrails

Imaging-transcriptomics has a bad reputation for specific reasons, and the repo
is built to prevent each structurally:

1. **Spatial autocorrelation.** Two arbitrary smooth brain maps correlate at
   r ≈ 0.4 by chance. `src/stats/spatial.py::corr_with_null()` is the only
   sanctioned way to correlate two maps, and it *cannot* return a p-value
   without a spatial null — passing `nulls=None` raises.
2. **Pipeline dependence.** Markello et al. 2021 showed AHBA processing choices
   can shift correlations by ρ ≥ 1.0. Every effect is reported with its
   multiverse distribution, not just a point estimate.
3. **The hierarchy confound.** Association cortex differs from sensory cortex on
   everything. Phase 5 partials the Margulies principal gradient and T1w/T2w
   myelin; if nothing survives, the finding is a hierarchy finding and must be
   reported as one.

---

## Commands

```bash
pytest -q && ruff check src/ scripts/ tests/

# Survey the fetched dataset
python -m src.data.targets inspect --root data/raw/ds004873

# Phase 0 — BLOCKING GATES
python scripts/p0_reliability.py --config config/base.yaml
python scripts/p0_dropout.py    --config config/base.yaml
```

---

## Data

`data/` is gitignored (R8). Provenance — URLs, snapshot tags, checksums, fetch
dates — lives in `data/MANIFEST.yaml`.

⚠️ **Fetch ds004873 via `src/data/fetch.py`, not `aws s3 sync`.** The S3 mirror
serves snapshot 1.0.4, which contains no derivatives at all. The mqBOLD maps
exist only in the 2.0.x snapshots. `fetch.py` pins 2.0.7 and verifies every
download against the SHA256 embedded in its git-annex key.

---

## Licence and citation

Analysis code: MIT. ds004873 is CC0; cite Epp et al. 2025 and the OpenNeuro DOI
when using it.
