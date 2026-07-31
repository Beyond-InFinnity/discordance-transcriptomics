# CLAUDE.md — `discordance-transcriptomics`

> **Read this file completely before writing any code.**
> If a task appears to conflict with the **Hard Rules** (§3) or the **Stop-and-Ask List** (§13), stop and surface the conflict rather than resolving it yourself.
>
> *Note for future maintenance:* if this file starts eating too much context, split §5–§9 into `docs/DATA.md`, `docs/PROTOCOL.md`, and `docs/PHASES.md` and leave pointers here. Keep §1–§4 and §13 in CLAUDE.md permanently.

---

## 1. What this project is

We are testing whether the **spatial topography of BOLD/CMRO₂ discordance** in the human cortex is explained by **molecular vascular and metabolic architecture**, using post-mortem transcriptomics as the explanatory layer.

**Background in one paragraph.** Epp, Castrillón, Yuan, Andrews-Hanna, Preibisch & Riedl (2025, *Nature Neuroscience*, doi:10.1038/s41593-025-02132-9) showed that ~40% of voxels with significant task-evoked BOLD changes exhibit *oxygen metabolism moving in the opposite direction*, concentrated in the default mode network. Their mechanistic observation: "discordant" voxels differ in **baseline oxygen extraction fraction (OEF)** and regulate oxygen demand via **OEF changes**, while "concordant" voxels are driven mainly by **CBF changes**. The first author's thesis speculates that association cortex has **lower capillary density** than primary sensory cortex, and that this could produce weakened or reversed responses. **That speculation has never been tested against molecular vascular architecture. That is the gap this project fills.**

**Primary hypothesis (H1, pre-specified).**
Discordance propensity is positively associated with regional expression of glycolytic and vascular-sparsity-related gene programs, and negatively with oxidative-phosphorylation programs — **over and above** the unimodal→transmodal cortical hierarchy.

**Primary mediation model (H2, pre-specified).**

```
vascular/metabolic gene expression  →  baseline OEF (and/or baseline CBV)  →  discordance propensity
```

**Deliverables.**
1. A reproducible analysis repo.
2. A released **parcel-level annotation table** (`discordance_annotation.csv` + JSON schema) giving, per cortical parcel: coupling ratio *n*, discordance frequency, baseline OEF, and a discordance-risk score. This is the reusable public artifact.
3. A Streamlit app that accepts a region name or an uploaded NeuroVault statistical map and returns per-region discordance risk plus molecular profile.
4. A preprint.

**Non-goals.** No claims about individual subjects. No causal claims from spatial correlation. No consciousness/philosophy content anywhere in the code, docstrings, or paper drafts — that framing is deliberately excluded from this artifact.

---

## 2. Intellectual guardrails (why the statistics are strict here)

Imaging-transcriptomics has a bad reputation for a reason. Three specific pathologies, which this repo must structurally prevent:

1. **Spatial autocorrelation inflation.** Brain maps are smooth. Two arbitrary smooth maps correlate at r ≈ 0.4 by chance. A p-value from a naive Pearson test is meaningless.
2. **Pipeline-dependent results.** Markello et al. (2021, *eLife* 10:e72129) showed AHBA processing choices can shift imaging–expression correlations by as much as ρ ≥ 1.0 — i.e. a finding can be reversed by a defensible parameter change. Single-pipeline results are not evidence.
3. **The hierarchy confound.** Association cortex differs from sensory cortex on *everything* — myelin, gene expression, receptor density, metabolism, evolutionary expansion. Any map that varies along the sensory→association axis will correlate with any gene set that varies along it. **This is the single most likely way this project produces a false positive.**

Every design decision below exists to neutralise one of these three.

---

## 3. Hard Rules

These are non-negotiable. Violating one invalidates the result.

- **R1. No spatial correlation is ever reported without a spatial-autocorrelation-preserving null model.** No bare `scipy.stats.pearsonr` / `spearmanr` p-values on brain maps. Ever. If you find yourself writing one, wrap it in `src/stats/spatial.py::corr_with_null()` instead.
- **R2. No gene-set result is reported without a competitive null** (size- and differential-stability-matched random gene sets), in addition to R1.
- **R3. Left hemisphere only** for primary analyses. Only 2 of 6 AHBA donors have right-hemisphere tissue. Right-hemisphere analysis is a labelled sensitivity check, never primary.
- **R4. Never hand-roll a coordinate-space transform.** All MNI152 ↔ fsaverage ↔ fsLR conversions go through `neuromaps.transforms`. Successive interpolation is a silent, unrecoverable error source.
- **R5. The hypothesis-driven gene sets are frozen** (§8.1). They are committed before Phase 4 results are viewed. Adding a gene set after seeing results is p-hacking. New sets go in a clearly labelled `exploratory/` analysis with no confirmatory claims attached.
- **R6. Every reported effect must be accompanied by its multiverse distribution**, not just its primary-pipeline point estimate.
- **R7. Fixed seeds everywhere.** `SEED = 42` in `config/base.yaml`, threaded through every stochastic call. Any script must produce byte-identical output on re-run.
- **R8. Never commit data.** `data/` is gitignored. Provenance lives in `data/MANIFEST.yaml` (URLs, versions, checksums, fetch date).
- **R9. Gates are gates.** Phase 0 and Phase 1 have pass/fail criteria (§9). If a gate fails, STOP and report. Do not proceed with a workaround.
- **R10. Every result artifact carries a manifest** (`*.manifest.json`) recording: git SHA, config hash, package versions, seed, wall-clock time, input file checksums.

---

## 4. Environment

Python **3.11** (3.12 has known friction with parts of the neuroimaging stack).

```bash
# Recommended: uv
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Or conda
conda create -n disctrans python=3.11 -y && conda activate disctrans
pip install -r requirements.txt
```

`requirements.txt`:

```
numpy>=1.26,<2.0
scipy>=1.11
pandas>=2.1
pyarrow                 # parquet for cached matrices
nibabel>=5.2
nilearn>=0.10.3
abagen>=0.1.4
neuromaps>=0.0.5
netneurotools>=0.2.3
brainspace>=0.1.10      # gradients, alternative surrogate maps
statsmodels>=0.14
scikit-learn>=1.4
pingouin>=0.5.4         # partial correlations, mediation
gseapy>=1.1             # MSigDB access
joblib>=1.3
pyyaml
pydantic>=2.6
tqdm
matplotlib>=3.8
seaborn>=0.13
surfplot>=0.2           # surface figures
datalad                 # OpenNeuro fetch
streamlit>=1.32
pytest>=8.0
ruff
```

### 4.1 Non-Python dependency — Connectome Workbench

`neuromaps` requires `wb_command` for fsLR transforms. **This is the most common setup failure.** Install before anything else:

```bash
# Linux
wget https://humanconnectome.org/storage/app/media/workbench/workbench-linux64-v1.5.0.zip
unzip workbench-linux64-v1.5.0.zip -d ~/opt/
echo 'export PATH="$HOME/opt/workbench/bin_linux64:$PATH"' >> ~/.bashrc
source ~/.bashrc
wb_command -version   # must succeed before proceeding
```

### 4.1a Machine hierarchy — READ BEFORE RUNNING ANYTHING REMOTE

Undefined roles caused the worst failure in this project's history: a completed
regeneration was silently reverted by a routine code sync, `results/` came to
hold four different dates while reading as a clean run, and repeated fixes
appeared not to take because stale artifacts kept overwriting fresh ones. It was
not detected for hours, and then only by an audit.

| machine | role | rule |
|---|---|---|
| **laptop** | **MASTER** | Only machine that commits, pushes, and holds canonical `results/`. |
| `workstation` (ssh) | compute | i9, 62 GB. Git **clone**. Two cards but only **one usable**: the RTX 3070 (8 GB). The RTX 5050 is Blackwell (sm_120) and the pinned PyTorch 2.5.1+cu121 supports only up to sm_90, so it is visible to `torch` and unusable. Effective GPU capacity is 8 GB, not 16. |
| `claude-machine` (ssh) | compute | i5, 31 GB. Git **clone**. |

**Code moves by `git pull`, never by rsync.** A compute node that is an rsync
target has no version identity — the workstation had no `.git` at all, so
nothing could establish which code produced its output. Both are now clones and
`git rev-parse HEAD` is the answer.

**Results move one direction only: compute → laptop.** Use
`scripts/pull_results.sh <host>`. Never rsync the repo root to a compute node:
`results/` is git-tracked, so a plain `rsync ./ host:` carries the laptop's
committed copies and destroys whatever that host just computed. Use
`scripts/sync_code.sh <host>`, which excludes it.

**Nothing in `results/` is trustworthy until `scripts/audit_provenance.py`
passes.** It is a gate (exit 1 on failure) and checks five things: one git SHA
across all artifacts, a clean tree at write time, every output paired with a
manifest, artifacts written hours rather than days apart, and agreement between
values appearing in more than one file. A run that fails it is not a run.

**Never compute on a dirty tree.** Ten artifacts were written from one, and a
dirty SHA identifies nothing. `regenerate_all.sh` refuses to start on one.

### 4.2 Disk and compute

- AHBA microarray download: **~4 GB** (abagen caches to `~/.abagen/` or `$ABAGEN_DATA`).
- ds004873: check size before `datalad get`; fetch only the derivatives you need.
- neuromaps reference maps: ~1 GB.

**Available hardware** (as of 2026-07-30):

| host | cores | RAM | GPU |
|---|---|---|---|
| laptop | 16 | 15 GB | RTX 3070 Laptop, 8 GB |
| `claude-machine` (ssh) | 4 | 31 GB | GTX 1080 Ti, 11 GB |

**The binding constraint has been memory, not cores and not FLOPs.** A single
abagen extraction peaks near 7 GB; on the laptop that meant 24 of 120 multiverse
cells were killed by the kernel, all in the same corner of the grid. Moving to
the 31 GB host completed all 120 with zero kills. Plan work around peak RSS per
process first, parallelism second.

**GPU work is now permitted, but only where the shape of the problem justifies
it.** The earlier blanket ban was right about the work that existed then and
wrong as a permanent rule. The test is whether the step is dense linear algebra
over arrays large enough to amortise the transfer:

- **Justified.** Vertex-level analysis (10,242 vertices/hemisphere rather than
  100 parcels), permutation budgets in the hundreds of thousands, PLS or
  regression over the full gene x vertex matrix. These are matrix products with
  both dimensions in the thousands.
- **Not justified.** abagen extraction (file I/O and pandas), anything shelling
  out to `wb_command`, the competitive null (10,000 small resamples bound by
  Python overhead), the parcel-level spin test (a 100 x 10,000 product that is
  already microseconds — the transfer costs more than the compute).

The old warning still applies in spirit and is worth keeping in mind: **do not
invent GPU work to justify the hardware.** If a step is fast enough on CPU, leave
it there.

**Precision rule, revised after measurement.** An earlier draft of this section
required GPU results to be numerically identical to the CPU path. Benchmarking
showed that rule cannot be satisfied at any speed advantage on consumer cards:

| | TFLOPS |
|---|---:|
| RTX 3070, float32 | 6.50 |
| RTX 3070, float64 | 0.26 |
| 16-core CPU, float64 | 0.26 |

Double precision runs at 1/32 rate on consumer silicon, landing exactly on top of
the CPU — a bit-identical GPU path is achievable and worthless. The whole
advantage is in float32, which is not bit-identical.

So the standard is **identical decisions with the discrepancy measured**, not
identical bits. On screen-shaped data, float32 against float64 gives a maximum
absolute difference of 2.7e-7 and flips 19 of 40,000,000 threshold comparisons
(4.75e-7). Each flip moves one gene's permutation p-value by 1/n_perm — the
fourth decimal — and cannot move it across 0.05 unless it already sat on 0.05 to
four digits.

Concretely: `src/utils/compute.py::validate_backend` performs that measurement,
the test suite asserts the flip rate stays below 1e-5, and `dtype='float64'`
remains the default for anything whose exact reproduction matters more than its
runtime.

---

## 5. Repository layout

```
discordance-transcriptomics/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── config/
│   ├── base.yaml              # seed, paths, parcellation, n_perm
│   ├── multiverse.yaml        # abagen parameter grid
│   └── genesets.yaml          # FROZEN gene set definitions (R5)
├── data/                      # GITIGNORED
│   ├── MANIFEST.yaml          # provenance: URL, version, checksum, date
│   ├── raw/
│   │   ├── ds004873/          # Epp et al. mqBOLD derivatives
│   │   ├── ds004513/          # Castrillón et al. energetic costs
│   │   └── genesets/          # MSigDB GMT files
│   ├── external/
│   │   ├── two_modes_of_hemodynamics/   # cloned repo
│   │   └── control_costs/               # cloned repo
│   └── derived/
│       ├── target_maps/       # n, discordance_freq, baseline_OEF, baseline_CBV
│       ├── expression/        # abagen outputs, one parquet per multiverse cell
│       └── nulls/             # cached permutation sets
├── src/
│   ├── data/
│   │   ├── fetch.py           # dataset acquisition + checksum verification
│   │   ├── targets.py         # build the four target maps
│   │   └── parcellate.py      # volumetric → surface → parcel
│   ├── expression/
│   │   ├── abagen_runner.py   # single-cell multiverse execution
│   │   └── genesets.py        # GMT loading, stability-matched resampling
│   ├── stats/
│   │   ├── spatial.py         # corr_with_null(), spin/variogram wrappers
│   │   ├── competitive.py     # gene-set competitive nulls
│   │   ├── hierarchy.py       # gradient/myelin partialling
│   │   └── mediation.py       # path model with spatial nulls
│   ├── viz/
│   └── utils/
│       ├── manifest.py        # R10 implementation
│       └── caching.py         # joblib.Memory setup
├── scripts/                   # thin CLI wrappers, one per phase step
│   ├── p0_reliability.py
│   ├── p0_dropout.py
│   ├── p1_reproduce.py
│   ├── p2_build_targets.py
│   ├── p3_multiverse.py
│   ├── p4_genesets.py
│   ├── p5_hierarchy.py
│   └── p6_mediation.py
├── notebooks/                 # FIGURES ONLY — no pipeline logic
├── results/                   # each output + its .manifest.json
├── app/                       # Streamlit
├── tests/
└── paper/
```

**Convention:** `src/` holds importable, tested logic. `scripts/` holds thin argparse/hydra wrappers that read config and call `src/`. **Notebooks never contain analysis logic** — they import from `src/` and plot. If you are tempted to put a computation in a notebook, it belongs in `src/`.

---

## 6. Data sources and provenance

| ID | What | How to get it | Notes |
|---|---|---|---|
| `ds004873` | Epp et al. raw + mqBOLD-processed NIfTIs | `datalad install ///openneuro/ds004873` then selective `datalad get`; or `aws s3 sync --no-sign-request s3://openneuro.org/ds004873 .` | **The primary input.** Their GitHub scripts take the mqBOLD-processed NIfTIs as input, not raw. |
| `two_modes` | Their analysis code | `git clone https://github.com/NeuroenergeticsLab/two_modes_of_hemodynamics` | Phase 1 reproduces this. |
| `ds004513` | Castrillón/Riedl energetic costs (*Sci Adv*) | OpenNeuro | Source of CMRGlu / neuromodulator maps for comparison. |
| `control_costs` | Ceballos, Luppi, Castrillón, Saggar, Misic & Riedl (*Network Neuroscience* 2025;9:77–99) | `git clone https://github.com/NeuroenergeticsLab/control_costs`; data on OSF | Time-averaged control energy map — a comparison target, and evidence Riedl's group already works with Misic-lab tooling. |
| `AHBA` | Allen Human Brain Atlas microarray | `abagen.fetch_microarray(donors='all')` | 6 donors, >20,000 genes, 3,702 samples in MNI space. |
| `neuromaps` | Reference maps + null models | `neuromaps.datasets.fetch_annotation()` | Source of the Margulies principal gradient, T1w/T2w myelin, CBF/CMRGlu, Hansen receptor atlas. |
| `MSigDB` | HALLMARK + GO gene sets | `gseapy.get_library()` or download GMT | See §8.1. Pin the version in MANIFEST. |

**Every fetch writes to `data/MANIFEST.yaml`:** source URL, version/tag/DOI, SHA256, fetch timestamp.

---

## 7. Analysis protocol — core decisions

### 7.1 Parcellation

- **Primary:** Schaefer-2018, 200 parcels, 7-network, **left hemisphere only (100 parcels)**, `fsaverage` space.
- **Sensitivity:** Desikan-Killiany 68 (34 LH); Schaefer-400 (200 LH).
- Rationale: functionally defined and DMN-appropriate; enough parcels for statistics; not so many that AHBA sample coverage collapses.
- **Track and report AHBA coverage per parcel.** Parcels with zero samples are filled per the `missing` strategy — record how many, and confirm findings hold when they are dropped entirely.

### 7.2 Volumetric → surface pipeline

The Epp maps are volumetric (MNI152). Spin tests require spherical coordinates. Therefore:

```
MNI152 volume
  → neuromaps.transforms.mni152_to_fsaverage(density='10k')
  → parcellate with Schaefer fsaverage annot
  → parcel-level vector (100 values, LH)
```

**Known limitation to state in the paper:** this discards subcortex and cerebellum. Acceptable — the hypothesis concerns cortical DMN nodes. Do not silently drop it; write it in the limitations section.

### 7.3 The four target maps (Phase 2)

Do **not** analyse "discordance" as a single variable.

| Map | Type | Role |
|---|---|---|
| Coupling ratio *n* = %ΔCBF / %ΔCMRO₂ | continuous, signed | **Primary outcome** — continuous is statistically stronger than binary |
| Discordance frequency across the 4 tasks | ordinal 0–4 | Robust phenomenological measure |
| Baseline OEF | continuous | **The mechanistic discriminator** (mediator in H2) |
| Baseline CBV / CBF | continuous | Alternative mediator |

Handle *n* carefully: it is a ratio and blows up when the denominator approaches zero. Use a signed log transform or work with the angle in the (ΔCBF, ΔCMRO₂) plane. Document the choice and include the alternative as a sensitivity check.

### 7.4 Null models — both are required

**Spatial null (R1).** Alexander-Bloch spherical rotation on parcellated surface data:

```python
from neuromaps import nulls, stats
rot = nulls.alexander_bloch(target, atlas='fsaverage', density='10k',
                            n_perm=10_000, seed=42, parcellation=schaefer_lh)
r, p = stats.compare_images(target, gene_map, nulls=rot)
```
Cache the rotation set — it is reusable across every test against the same target map.
For any volumetric-only analysis use `nulls.burt2020` (variogram-matched surrogates) instead.

**Competitive gene-set null (R2).** Resample random gene sets matched on **both** set size and differential-stability distribution, ≥10,000 draws. Implemented in `src/stats/competitive.py`.

**Reporting.** Every effect gets: point estimate, spatial-null p, competitive-null p, and BH-FDR across the gene-set family.

### 7.5 The multiverse (Phase 3)

Primary pipeline (Arnatkeviciute 2019 / Markello 2021 recommended defaults):

```python
expr = abagen.get_expression_data(
    atlas, probe_selection='diff_stability', lr_mirror='bidirectional',
    missing='centroids', norm_matched=True, sample_norm='srs',
    gene_norm='srs', donors='all',
)
stable = abagen.keep_stable_genes(expr, threshold=0.1)
```

Multiverse grid (`config/multiverse.yaml`), ~96–120 cells:

| Parameter | Values |
|---|---|
| `probe_selection` | diff_stability, rnaseq, max_intensity, max_variance, corr_variance |
| `lr_mirror` | None, 'bidirectional' |
| `missing` | None, 'centroids', 'interpolate' |
| `tolerance` | 1, 2 |
| `norm_matched` | True, False |
| stability threshold | 0.0, 0.1, 0.2 |

Full factorial is ~360 cells; take the **full grid over `probe_selection` × `lr_mirror` × `missing` (30 cells)** plus a **seeded random sample of 70 from the remainder**. Each cell writes a parquet to `data/derived/expression/` keyed by a config hash. Parallelise with joblib; cache aggressively — abagen is slow.

**Report the primary pipeline as the headline and the multiverse as a distribution.** A finding that survives <80% of the multiverse with consistent sign is reported as unstable.

---

## 8. Gene sets — FROZEN (R5)

### 8.1 Hypothesis-driven sets

Committed to `config/genesets.yaml` **before** Phase 4 results are viewed.

**Sourced from MSigDB (pin version):**
- `HALLMARK_GLYCOLYSIS`
- `HALLMARK_OXIDATIVE_PHOSPHORYLATION`
- `HALLMARK_ANGIOGENESIS`
- `HALLMARK_HYPOXIA`
- `GOBP_BLOOD_VESSEL_MORPHOGENESIS`

**Curated small sets (cite the source paper for each in the YAML):**
- Endothelial: `PECAM1, CLDN5, VWF, FLT1, ESAM, TIE1`
- Pericyte/mural: `PDGFRB, RGS5, ANPEP, KCNJ8, ACTA2`
- Astrocyte: `AQP4, GJA1, SLC1A3, ALDH1L1`
- Glucose/lactate transport: `SLC2A1, SLC2A3, SLC16A1, SLC16A3, SLC16A7`
- Glycolytic enzymes: `HK1, HK2, PFKFB3, LDHA, LDHB, PKM`
- Interneuron subclass: `PVALB, SST, VIP, LAMP5`
- Mitochondrial density proxy: nuclear-encoded OXPHOS complex subunits (NDUFx, SDHx, UQCRx, COXx, ATP5x families)

**Cell-type deconvolution:** use a published set (Seidlitz et al. 2020 / Lake et al. 2018-derived) rather than hand-assembling. Cite it.

### 8.2 Data-driven arm (run in parallel, not instead)

Rank all stable genes by correlation with the target map; run enrichment on both tails; and run PLS regression (expression → target) with spin-test inference on component scores.

**Convergence between the hypothesis-driven and data-driven arms is the strongest available evidence.** Divergence is a finding too — report it.

---

## 9. Phases, with gates

### Phase 0 — Sanity gates (3–4 days) ⛔ **BLOCKING**

**0a. Reliability of the target map.**
- Split subjects into random halves 1,000 times; compute the parcel-level discordance map in each half; Spearman correlate.
- Apply Spearman-Brown correction to estimate full-sample reliability.
- Compute ICC(2,1) per parcel across subjects.
- **GATE:** median Spearman-Brown-corrected r ≥ 0.5 → proceed. 0.3–0.5 → proceed with prominent caveats and reduced parcel resolution (try DK-68). **< 0.3 → STOP. Report to the user. Do not proceed.**
- Output this number regardless of outcome — it belongs in the paper and nobody else has computed it.

**0b. The dropout confound.** ⚠️ *This is the strongest attack on the entire project. Run it before investing anything else.*
- mqBOLD derives OEF partly from T2*, and T2* is corrupted by macroscopic B0 inhomogeneity — worst near the sphenoid and frontal sinuses, i.e. directly under vmPFC, a DMN node.
- Build a dropout proxy from the dataset: mean EPI signal intensity map, tSNR map, or a B0 field map if available.
- Correlate discordance against the dropout proxy (with spin-test inference).
- **GATE:** if |r| ≥ 0.5, the confound is severe. STOP and report — this becomes the finding, and it is a more important one than the original hypothesis. If |r| < 0.5, carry the dropout proxy as a mandatory covariate in every downstream model.

### Phase 1 — Exact reproduction (1–2 weeks) ⛔ **BLOCKING**

Run the `two_modes_of_hemodynamics` notebooks end-to-end on the ds004873 mqBOLD derivatives until their published figures regenerate.

- **GATE:** figures match published versions within visual/numeric tolerance. Any discrepancy is documented in `results/p1_reproduction_notes.md`.
- Unglamorous and non-negotiable: it validates the pipeline, teaches their conventions, and converts the eventual email to Riedl from "I read your paper" into "I reproduced your analysis end-to-end and here is what I found next."

### Phase 2 — Target maps (1 week)
Build the four maps (§7.3) at all three parcellations. Write `data/derived/target_maps/` + manifests.

### Phase 3 — Expression multiverse (2 weeks)
Run §7.5. Output one parquet per cell plus a `multiverse_index.csv`.

### Phase 4 — Gene-set association (1–2 weeks)
Hypothesis-driven and data-driven arms (§8), with both nulls (§7.4), across the multiverse.

### Phase 5 — Hierarchy control (1 week) ⛔ **DECISIVE**
Hierarchical regression:
1. Enter Margulies principal functional gradient + T1w/T2w myelin (+ dropout proxy from 0b).
2. Ask whether gene sets explain **additional** variance in the target.
3. Report partial correlations with spin-test inference.

Also correlate the target against: Vaishnavi aerobic glycolysis, CBF, CMRGlu, Hansen receptor/transporter atlas, evolutionary cortical expansion, `control_costs` TCE map.

**If nothing survives partialling for the principal gradient, we do not have a molecular finding — we have a hierarchy finding.** That is still publishable, but it must be reported honestly as such. Do not bury it.

### Phase 6 — Mediation (2 weeks)
Fit `expression → baseline OEF/CBV → discordance` as a parcel-level path model with spin-test-based inference on **each path**. Bootstrap the indirect effect. Report the direct effect too.

This directly tests the capillary-density conjecture in the Epp thesis. If the mediation holds, we have supplied a mechanism for their phenomenon.

### Phase 7 — Artifacts and writing (3 weeks)
Annotation table, Streamlit app, preprint, repo polish.

**Realistic total: ~3.5 months at serious part-time.**

---

## 10. Coding conventions

- **Config-driven.** No magic numbers in code. Everything in `config/*.yaml`, loaded via pydantic models in `src/utils/config.py`.
- **Determinism.** `SEED` from config threaded to every RNG. Re-running a script produces byte-identical output.
- **Caching.** `joblib.Memory` at `data/.cache/`. abagen calls, null-model generation, and multiverse cells are all cached by config hash.
- **Manifests.** Every write to `results/` emits `<name>.manifest.json` per R10.
- **Style.** `ruff` for lint+format. Type hints on all public functions. NumPy-style docstrings.
- **Testing.** `pytest`. Mandatory tests for: coordinate-space transforms (round-trip a known map), parcellation alignment (parcel label ↔ vertex mapping), null-model shape/reproducibility, and gene-set loading.
- **Logging.** `logging` module, not `print`. INFO for progress, DEBUG for shapes and hashes.
- **Commits.** Conventional commits. Never commit `data/`, `results/*.nii.gz`, or `.cache/`.

---

## 11. Statistical reporting checklist

Every reported association must state:

- [ ] Effect size (Spearman ρ preferred; brain maps are rarely bivariate normal)
- [ ] Spatial-null p (n_perm, null type, seed)
- [ ] Competitive-null p where a gene set is involved
- [ ] BH-FDR-adjusted p across the family
- [ ] Multiverse distribution: median, IQR, % of cells with consistent sign
- [ ] Partial effect after gradient + myelin + dropout covariates
- [ ] Parcellation sensitivity (does it hold at DK-68 and Schaefer-400?)
- [ ] AHBA coverage for the parcels involved

---

## 12. Glossary

### 12.1 Physiology and signal

| Term | Meaning |
|---|---|
| **BOLD** | Blood-oxygenation-level-dependent signal. Sensitive to deoxyhaemoglobin, not neurons. To first order `sign(ΔBOLD) ≈ sign(ΔCBF − ΔCMRO₂)`. |
| **CMRO₂** | Cerebral metabolic rate of oxygen. Direct measure of oxidative metabolism. Cortical grey matter ≈ 130–160 µmol/100g/min. |
| **CBF / CBV** | Cerebral blood flow / volume. Cortical grey-matter CBF ≈ 45–60 mL/100g/min. |
| **OEF** | Oxygen extraction fraction = CMRO₂ / (CBF × arterial O₂ content). Cortex ≈ 0.3–0.4. The mechanistic discriminator in Epp et al. |
| **CaO₂** | Arterial oxygen content. In ds004873, `0.334 × Hct × 55.6 × O₂sat/100` — **subject-specific via haematocrit**, so CMRO₂ is not a pure imaging quantity. |
| **R2′** | `1/T2* − 1/T2`. The reversible transverse relaxation rate; what mqBOLD converts into OEF. |
| **Coupling ratio *n*** | %ΔCBF / %ΔCMRO₂. Determines BOLD sign and amplitude. **n = 1 is the BOLD null line**: below it, BOLD opposes CMRO₂. Typical task activation is n ≈ 2–4. |
| **Concordant / discordant** | BOLD and CMRO₂ moving in the same / opposite direction. Equivalent to n > 1 / n < 1 under the first-order BOLD approximation. |
| **DMN** | Default mode network. Where discordance concentrates — reproduced in our own data (highest of the 7 Yeo networks). |

### 12.2 Method and acquisition

| Term | Meaning |
|---|---|
| **mqBOLD** | Multiparametric quantitative BOLD (Preibisch). Derives OEF from T2, T2*, CBV. **The method in ds004873 — NOT hypercapnia-calibrated fMRI.** Sensitivity analyses must target mqBOLD's assumptions, not the Davis model's. |
| **MESE / MEGRE** | Multi-echo spin-echo (→ T2) / multi-echo gradient-echo (→ T2*). The two acquisitions R2′ is built from. |
| **DSC / ASL** | Dynamic susceptibility contrast (→ CBV) / arterial spin labelling (→ CBF). |
| **OEF cap** | The authors clip OEF at `max(5 × subject median, 1.5)` and **retain** clipped voxels. Already applied in the published maps — do not re-threshold at 1.0. |
| **qBmasked** | Their per-subject maps after GM ∩ R2′ ∩ CBV masking. **Released for one subject only**, so their per-subject masking is not reproducible from the public data. |
| **GMR2pCBVmasked** | The same masking applied to the published *group* maps. These are the authoritative baseline source. |
| **desc-orig vs desc-CBV** | Two CMRO₂ variants. Their calc-condition analysis uses `desc-CBV`; **only `desc-orig` is published in MNI152**, which is why our coupling columns are provisional. |

### 12.3 Statistics

| Term | Meaning |
|---|---|
| **Spin test** | Spatial-autocorrelation-preserving permutation via spherical rotation (Alexander-Bloch 2018). Required by R1. |
| **Variogram surrogate** | `burt2020` volumetric alternative to the spin test, for analyses that never reach the surface. |
| **Competitive null** | Gene-set null resampling random sets matched on size *and* differential stability. Required by R2 — a spatial null alone does not control for set size. |
| **Principal gradient** | Margulies 2016 unimodal→transmodal cortical axis. **The confound to beat.** |
| **Differential stability** | Consistency of a gene's regional expression pattern across donors. Standard filter. |
| **Spearman-Brown** | Correction projecting a split-half correlation up to full-sample reliability. The Phase 0a metric. |
| **Multiverse** | Running the analysis across the grid of defensible AHBA preprocessing choices and reporting the distribution, not one point estimate (R6). |

### 12.4 Data and artifacts

| Term | Meaning |
|---|---|
| **AHBA** | Allen Human Brain Atlas. 6 post-mortem donors, microarray, mostly left hemisphere. **Donor 15496 is currently unavailable upstream (HTTP 404); analyses run on 5.** |
| **ds004873** | Epp et al. OpenNeuro dataset. **Pin snapshot 2.0.x** — the S3 mirror serves 1.0.4, which contains no derivatives at all. |
| **Schaefer / DK-68** | Primary parcellation (200×7, LH = 100 parcels) / Desikan-Killiany sensitivity parcellation (LH = 34). |
| **fsaverage5** | The FreeSurfer subject matching neuromaps density `10k` (10,242 vertices/hemisphere). What all surface work here uses. |
| **`discordance_risk`** | Our released measure: fraction of subjects with n < 1 in a parcel. **Not** the Epp voxel percentage, and not directly comparable to it. |
| **`dropout_snr_coverage`** | Fraction of a parcel's vertices surviving the authors' SNR criterion. The mandatory Phase 0b covariate. |
| **VENAT** | Venous atlas shipped with ds004873. Source of the venous partial-volume covariate — the "brain vs vein" confound. |

---

## 13. Stop-and-Ask List

Do **not** decide these unilaterally. Surface them and wait:

1. Any Phase 0 or Phase 1 gate failing or landing in a grey zone.
2. Changing the primary parcellation, target map definition, or null model.
3. Adding, removing, or modifying a gene set in `config/genesets.yaml` after Phase 4 has been run.
4. Any decision that would let an analysis proceed despite a failed gate.
5. Discovering that the mqBOLD derivatives in ds004873 don't contain a variable the protocol assumes (especially baseline OEF or per-task CBF/CMRO₂ maps).
6. Any result that looks *too* clean — ρ > 0.7 against a gene set is more likely a bug or a confound than a discovery. Investigate before reporting.
7. Anything requiring data not listed in §6.

---

## 14. Known limitations (write these into the paper, don't discover them in review)

- AHBA is 6 adult post-mortem donors, bulk microarray, predominantly left hemisphere. It is a **modal** brain, not a matched sample. **No individual-level inference is licensed.**
- Spatial correlation is not mechanism. The mediation model is suggestive, not causal.
- ds004873 is ~40 subjects, 4 tasks, one scanner, one site. Generalisation is untested.
- Volumetric→surface projection discards subcortex and cerebellum.
- mqBOLD carries its own assumptions (vessel geometry, blood volume estimation, T2' modelling) that propagate into the OEF and CMRO₂ estimates.
- The dropout confound (Phase 0b) is mitigated, not eliminated.

---

## 15. Quick command reference

```bash
# Setup
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
wb_command -version                      # must pass

# Data
python scripts/fetch_all.py --datasets ds004873 ahba neuromaps
python -c "import abagen; abagen.fetch_microarray(donors='all')"

# Phase 0 — GATES
python scripts/p0_reliability.py --config config/base.yaml
python scripts/p0_dropout.py    --config config/base.yaml

# Phase 1 — reproduction
python scripts/p1_reproduce.py --repo data/external/two_modes_of_hemodynamics

# Phases 2–6
python scripts/p2_build_targets.py
python scripts/p3_multiverse.py --n-jobs -1
python scripts/p4_genesets.py --n-perm 10000
python scripts/p5_hierarchy.py
python scripts/p6_mediation.py --n-boot 10000

# Quality
pytest -v && ruff check src/ scripts/ && ruff format src/ scripts/

# App
streamlit run app/main.py
```

---

## 16. Suggested first actions for a fresh session

1. Confirm `wb_command -version` succeeds.
2. Read `data/MANIFEST.yaml` to see what has already been fetched.
3. Read `results/` for existing manifests — find the furthest completed phase.
4. Check whether Phase 0 gates have passed (`results/p0_*.manifest.json`). **If not, that is the only work available.**
5. Re-read §3 (Hard Rules) and §13 (Stop-and-Ask) before writing code.
