# discordance-transcriptomics

Does the spatial topography of **BOLD/CMRO₂ discordance** in human cortex reflect
**molecular vascular and metabolic architecture**? Post-mortem transcriptomics
(Allen Human Brain Atlas) as the explanatory layer, tested against
spatial-autocorrelation-preserving nulls across the full space of defensible
processing choices.

Epp et al. (2025, *Nat Neurosci*, [doi:10.1038/s41593-025-02132-9](https://doi.org/10.1038/s41593-025-02132-9))
showed that ~40% of voxels with significant task-evoked BOLD changes have oxygen
metabolism moving in the *opposite* direction, concentrated in the default mode
network. The first author's thesis speculates that association cortex has lower
capillary density, and that sparse supply could produce weakened or reversed
responses. That speculation had never been tested against molecular vascular
architecture. This repository tests it.

Two hypotheses, both pre-specified before any result was seen:

- **H1** — discordance propensity tracks glycolytic and vascular gene programs
  (negatively with oxidative phosphorylation), *over and above* the
  unimodal→transmodal cortical hierarchy.
- **H2** — `vascular/metabolic expression → baseline OEF/CBV → discordance`.

`CLAUDE.md` is the full specification. §3 (Hard Rules) and §13 (Stop-and-Ask)
are binding on all work here.

---

## What was found

**Both pre-specified hypotheses failed, and one unlooked-for result held.**
Reported here in the same order and detail as if they had succeeded.

**Vascular gene expression predicts baseline oxygen extraction.** Pericyte/mural
genes at ρ = −0.39 and angiogenesis genes at ρ = −0.36, each holding sign in
**100% of 120 processing pipelines**, each surviving a competitive null of random
gene sets matched on both size and cross-donor reproducibility (p = 0.0004,
0.0002). Correcting both sides for measurement noise puts the underlying effect
near −0.53. The signal runs through **blood flow** (ρ = +0.45), not blood volume
(ρ = −0.03) — which matters, because blood volume enters the derivation of the
extraction estimate and would otherwise have made the result circular.

**Discordance itself is not predicted by any frozen gene set.** Nor, in a
whole-transcriptome screen of 15,562 genes with Westfall–Young family-wise
correction, by any individual gene.

**H2 breaks at its second link.** Gene expression predicts baseline extraction
(strong), but baseline extraction does not predict where discordance occurs
(ρ = −0.13, p = 0.36). A path model fitted across 120 pipelines returns a null
indirect effect and names `b` as the limiting link, rather than reporting "no
mediation found".

**Discordance is two phenomena, not one.** An extraction mode (demand rises, flow
lags, extraction increases, signal falls) and an overshoot mode (demand falls,
flow lags, signal rises). They are spatially anticorrelated at −0.56, and only
the extraction mode is what a capillary-density account predicts. It peaks in
**somatomotor cortex** — among the best-perfused tissue in the brain — with the
default mode network fourth.

### Why these negatives are interpretable

Three things separate a bounded negative from an underpowered one:

- **A working positive control.** The same machinery, same parcels, same
  statistics detects endothelial genes tracking directly-measured macaque
  microvascular density at ρ = +0.46, spin-significant in 99% of the 120
  pipelines, and a transcriptome-wide excess at z = 5.9 in all 12 pipelines the
  whole-transcriptome arm was run over. The same test returns nothing for any
  discordance measure, in 0 of 12.
- **Measured detectability bounds.** Gene-set map reproducibility was measured
  per donor rather than assumed, giving a real exclusion threshold per set —
  from 0.41 for the best-measured to 0.94 for the worst. For the vascular sets
  the hypothesis concerns, true effects above ~0.44 are excluded. **For the three
  large HALLMARK collections it is not an exclusion at all**, and those nulls are
  reported as uninformative rather than as evidence.
- **The dominant confound does not apply.** Correlation with the Margulies
  principal gradient is 0.04. Most negatives in this literature cannot rule out
  that they are hierarchy artifacts; this one can.

### A methodological result

Human cerebral blood volume gives a sensory-to-association ratio of **0.97**,
against **2–3** in directly measured macaque cortical tissue. Both available
human measurements are effectively flat. The standard human measurement therefore
cannot resolve the microvasculature, and cannot test the capillary hypothesis at
all — which is a useful thing to have established.

---

## Status

| Phase | State | Result |
|---|---|---|
| **0a** reliability gate | ✅ pass | baseline OEF 0.98, coupling map 0.71 (split-half, Spearman-Brown) |
| **0b** dropout gate | ✅ pass | carried as a mandatory covariate downstream |
| **1** reproduction (scoped) | ✅ pass | authors' group CBF map regenerated: r = 1.000000, median abs difference 4.5 × 10⁻¹³ over 867,944 voxels |
| **2** target maps | ✅ done | 3 parcellations, per-column reliability released |
| **3** expression multiverse | ✅ done | 120/120 cells × 3 parcellations, no gaps |
| **4** gene sets, both nulls | ✅ done | 11 frozen sets × 120 pipelines × 3 stability thresholds |
| **4b** data-driven arm | ✅ done | 15,562-gene screen over 12 pipelines, max-T family-wise correction, PLS |
| **5** hierarchy control | ✅ done | gradient + myelin + dropout partialled |
| **6** mediation | ✅ done | 15,840 path models, spatial null on every path |
| **7** artifacts | ◐ partial | annotation table released; app and preprint outstanding |

251 tests. `docs/WHERE_WE_ARE.md` is the plain-language running summary and the
best entry point; `CLAUDE.md` is the specification.

---

## The released artifact

`data/derived/annotation/discordance_annotation.csv` — per cortical parcel, at
three parcellations (Schaefer-200, Schaefer-400, Desikan-Killiany), with a JSON
Schema and data dictionary:

- baseline oxygen extraction fraction, cerebral blood flow, CMRO₂
- coupling ratio as a bounded angle rather than a ratio that diverges at zero
- discordance frequency, **separated into extraction and overshoot modes**
- the scanner-dropout and venous partial-volume covariates
- Allen atlas sample coverage per parcel

**Every measurable column publishes its own split-half reliability**, so users
can attenuation-correct rather than assume perfect measurement. Stability labels
are derived from those measurements at build time, not hand-maintained — a column
cannot be advertised as reliable while its measured reliability says otherwise.

---

## The three statistical guardrails

Imaging-transcriptomics has a poor reputation for specific, well-documented
reasons. This repository is built to prevent each structurally rather than by
discipline:

1. **Spatial autocorrelation.** Two arbitrary smooth brain maps correlate at
   r ≈ 0.4 by chance. `src/stats/spatial.py::corr_with_null()` is the only
   sanctioned way to correlate two maps here, and it *cannot* return a p-value
   without a spatial null — passing `nulls=None` raises.
2. **Pipeline dependence.** Markello et al. (2021) showed AHBA processing choices
   can shift a correlation by ρ ≥ 1.0 — enough to reverse a published finding.
   Every effect is reported as a distribution over 120 pipelines with the share
   agreeing on sign, never as a point estimate.
3. **The hierarchy confound.** Association cortex differs from sensory cortex on
   nearly everything. Phase 5 partials the principal functional gradient, T1w/T2w
   myelin and the dropout proxy. If nothing survives, the result is a hierarchy
   finding and is reported as one.

Every result artifact carries a `.manifest.json` recording git SHA, config hash,
package versions, seed, wall-clock time and input checksums.

---

## Setup

Python 3.11 and Connectome Workbench.

```bash
# Workbench first — neuromaps needs it for every surface transform,
# and it is the most common setup failure.
wget https://humanconnectome.org/storage/app/media/workbench/workbench-linux64-v1.5.0.zip
unzip workbench-linux64-v1.5.0.zip -d ~/opt/
export WORKBENCH_DIR="$HOME/opt/workbench/bin_linux64"
wb_command -version

uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
nbstripout --install --attributes .gitattributes   # per checkout; see note below

python scripts/fetch_all.py       # all public datasets, checksum-verified
pytest -q && ruff check src/ scripts/ tests/
```

Three pinned deviations from a plain install, each documented inline in
`requirements.txt`: `abagen` is pinned to a commit SHA rather than a release (the
PyPI build calls `DataFrame.append`, removed in pandas 2.0, in its core
aggregation path); `pandas` is capped `<3` (3.0 removed `groupby(axis=)`, which
abagen still uses); `setuptools<81` because abagen imports `pkg_resources`.

`nbstripout` must be registered per checkout — `.gitattributes` is committed but
the filter lives in `.git/config` and does not travel with a clone.

---

## Running it

```bash
python scripts/p0_reliability.py            # blocking gate
python scripts/p0_dropout.py                # blocking gate
python scripts/p0_dynamic_range.py          # detectability bounds
python scripts/p0c_geneset_reliability.py   # gene-set map reproducibility
python scripts/p2_build_targets.py
python scripts/p3_multiverse.py --n-jobs 6  # 120 cells; --parcellation for sensitivity
python scripts/p4_genesets.py --n-draws 10000
python scripts/p4b_datadriven.py            # whole-transcriptome arm
python scripts/p5_hierarchy.py
python scripts/p6_mediation.py --n-boot 10000
python scripts/build_annotation.py --ahba
python scripts/x1_macaque_vascular.py       # cross-species positive control
```

Phase 1 is a **scoped** reproduction — rather than regenerating every published
figure, it reproduces the authors' group-level parameter maps exactly and
extracts the two pipeline conventions Phase 2 depends on. It was run against
`two_modes_of_hemodynamics` @ `1b22c2cb`; the working is in
`results/p1_reproduction_notes.md`, including the discrepancies found.

Everything is cached by content hash and idempotent, so an interrupted run
resumes from where it stopped. Peak memory per abagen process is ~7 GB; see
`docs/MIGRATING_MACHINES.md` for running across hosts.

---

## Layout

```
config/       base.yaml, multiverse.yaml, genesets.yaml (frozen before Phase 4)
src/          importable, tested logic
  data/       fetch, target maps, parcellation, cross-species transfer, warping
  stats/      spatial.py (the null enforcement point), reliability, competitive,
              hierarchy, mediation
  expression/ abagen multiverse runner, gene sets, data-driven arm
  utils/      config, manifest, caching, compute backend
scripts/      thin CLI wrappers, one per phase step
tests/        pytest — 251 tests
results/      every output plus its .manifest.json
data/         gitignored; provenance in data/MANIFEST.yaml
docs/         WHERE_WE_ARE.md, MIGRATING_MACHINES.md, NV_DATA_SURVEY.md
```

`src/` holds logic, `scripts/` holds argparse wrappers, `notebooks/` are figures
only and never contain analysis.

---

## Data

`data/` is gitignored. Provenance — URLs, snapshot tags, SHA256, fetch dates —
lives in `data/MANIFEST.yaml`.

⚠️ **Fetch ds004873 through `src/data/fetch.py`, not `aws s3 sync`.** The S3
mirror serves snapshot 1.0.4, which contains no derivatives at all; the mqBOLD
maps exist only in the 2.0.x snapshots. `fetch.py` pins 2.0.7 and verifies every
file against the SHA256 embedded in its git-annex key.

Two items cannot be scripted, and are reported rather than failed silently: the
BALSA macaque vascular download requires an account, and Allen donor 15496
returns HTTP 404 upstream, so every analysis here runs on five of six donors.

---

## Known limitations

Written here rather than discovered in review:

- The Allen atlas is six adult post-mortem donors, bulk microarray, predominantly
  left hemisphere. It is a *modal* brain. No individual-level inference is
  licensed.
- Spatial correlation is not mechanism. The mediation model is suggestive at best.
- The source dataset is ~40 subjects, one scanner, one site.
- Only 2 of 4 experimental conditions are published in standard space, so the
  contrast is task-versus-task rather than task-versus-rest. This is the largest
  open uncertainty in the discordance maps.
- Volumetric→surface projection discards subcortex and cerebellum.
- Cross-species registration error is ~6.7 mm in sensory cortex and ~18.2 mm in
  association cortex — worst precisely where the question lives.
- mqBOLD carries its own assumptions, which propagate into every extraction and
  metabolism estimate here.

---

## Licence and citation

Analysis code: MIT. ds004873 is CC0 — cite Epp et al. (2025) and the OpenNeuro
DOI when using it. The macaque vascular maps are from Autio et al. and should be
cited directly.
