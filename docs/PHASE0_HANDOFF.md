# Phase 0 handoff — what is done, what is blocked, what needs you

Status as of 2026-07-26. Written for Connor; assumes CLAUDE.md is the spec.

> **UPDATE — both Phase 0 gates now PASS.** See `results/p0_summary.md`.
> Following Opus 5's review, the §1 decision below was **deferred rather than
> taken**: Phase 1 is reordered ahead of the Phase 2 target design, and Phase 0
> ran on the two targets that are unambiguous under any reading of the contrast
> structure (baseline OEF and the calc-vs-control coupling ratio). The
> frequency-map question stays open and stays unwired.
>
> The n=31 problem in §1 is **resolved in principle**: ds004873 ships the
> authors' own `from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5` warps for 40
> subjects. Applying an existing warp is not estimating one, so R4 permits it,
> and it lifts the coupling ratio from n=31 to **n=41**. Not yet applied — the
> gates ran at n=30.

---

## 1. Deferred (not decided): ds004873 has 2 usable conditions, not 4

**This is CLAUDE.md §13.5 and it needs your decision before Phase 2 target-map
design is final. It does not block Phase 0a/0b, but it changes one of the four
target maps.**

The protocol (§7.3) specifies a **discordance frequency map, ordinal 0–4,
across the 4 tasks**, and `config/base.yaml` carries `targets.n_tasks: 4`.

I enumerated every quantitative map in snapshot 2.0.7. Coverage by condition,
counting subjects with each quantity present:

| condition | space | OEF | CBF | CBV | CMRO₂ |
|---|---|---:|---:|---:|---:|
| **calc**    | MNI152 | 41 | 41 | 40 | **31** |
| **calc**    | T1w    | 40 | 40 | 40 | 40 |
| **control** | MNI152 | 40 | 41 | 40 | 41 |
| **control** | T1w    | 40 | 40 | 40 | 40 |
| **mem**     | T2 only | — | — | — | 30 |
| **rest**    | — | — | — | — | — |

Read that as: **`calc` and `control` are fully available. `mem` has CMRO₂ only,
in T2 space, and no CBF — so no coupling ratio can be formed for it. `rest` has
essentially nothing.**

This is consistent with the dataset README, which states four conditions for
subjects p019–p055 but only two (CALC and CTRL) for p058–p068 — except the
quantitative maps are thinner still than that implies.

Note the authors' own notebooks in `two_modes_of_hemodynamics` reference all
four conditions heavily (calc 285, control 161, mem 117, rest 66 mentions), so
they clearly had richer data than what is published. The published derivatives
are a subset.

### Why this is now deferred rather than decided

The framing above offered three options. Opus 5 identified a fourth that
dissolves the question, and it is right:

**Nobody has established the contrast structure yet.** `control` is very likely
the control condition *of the calculation task*, not a co-equal second task. If
so, there is not "2 conditions" — there is **one contrast**, and a
discordance-frequency map is not degraded from 0–4 to 0–2, it is **undefined**.
If instead every condition was contrasted against `rest`, then losing `rest`
kills the contrast structure entirely and we would be reconstructing something
different from what they did.

Which is true is exactly what Phase 1 reproduction answers, and **Phase 1 is not
blocked by this decision**. So the order is: run Phase 1, then design the
Phase 2 targets.

Phase 0 did not need the frequency map. It needed *a* well-defined target, and
baseline OEF and the calc-vs-control coupling ratio are unambiguous under either
reading. Both gates ran on those and both passed.

**When you do decide, the lean is option 2 — drop the frequency map** (not
option 1). A 0/1/2 ordinal over two non-independent conditions has poor
measurement properties: low variance, heavily tied, and a spatial pattern
near-collinear with the continuous *n* map. It would add a family member to the
FDR correction while contributing almost no independent information. §7.3
already says continuous is stronger; this is that principle cashing out.

**Take option 3 too, but after Phase 1 and not as a blocker.** "I reproduced
your analysis end to end; I notice the published derivatives omit the mem/rest
quantitative maps in MNI152 — are those releasable?" is a far better first
contact with Riedl's group than a bare data request, and it arrives with proof
of work.

### The n=31 question — resolved

ds004873 **does** ship transform files: `sub-*_from-T1w_to-MNI152NLin6Asym_
mode-image_xfm.h5` for 40 subjects, plus `*_desc-fmriprep_T2_to_T1w.mat`.
Applying the authors' own existing warp is not *estimating* a transform, so R4
permits it. Coverage with the warps applied:

| quantity | MNI152 as published | + authors' warp |
|---|---:|---:|
| coupling ratio (calc+control × CBF+CMRO₂) | 31 | **41** |

Not yet applied — the Phase 0a gate ran at n=30. Doing so needs ANTs
(`antsApplyTransforms`) or `nitransforms` for `.h5` composite transforms, which
is a dependency decision worth making deliberately rather than in passing.

---

## 2. What I could not do at all

### 2a. Nothing, for Phase 0a/0b setup — but these need a human eventually

| Item | Why it needs you |
|---|---|
| **The §13.5 decision above** | Protocol change. §13 forbids me deciding it. |
| **Emailing Riedl / Epp** | Correspondence is yours. |
| **Phase 1 conda environment** | `two_modes_of_hemodynamics/environment.yml` is a fully-pinned Linux conda export (`_libgcc_mutex`, MKL blas, etc). It will likely need loosening to solve on your machine. I did not create it because guessing at replacements changes their numerical stack, which is exactly what Phase 1 exists to validate. See §4 below. |
| **mqBOLD reprocessing** | Requires MATLAB and the Preibisch group's code (`qBOLD_BIDS_Hct_April21.zip`, or gitlab.lrz.de/nmrm_lab). Only relevant if you decide to regenerate maps rather than use the published ones. Almost certainly not worth it. |

### 2b. Things I *could* do but deliberately did not

- **Fill in `DERIVATIVE_PATTERNS`** in `src/data/targets.py`. I now know the
  real filenames, but wiring the loader is a Phase 2 task and it should follow
  your answer to §1 above. The loaders raise a clear instruction message until
  then rather than guessing.
- **Fetch AHBA** (~4 GB via `abagen.fetch_microarray`). Not needed until Phase
  3, and it is a long download. Command in §5.
- **`--force` replace your `~/.local/bin/python3.11` symlink.** It points into
  the *Loom* project's vendored runtime. I left it alone and built this
  project's venv on a separate uv-managed interpreter instead. See §3.

---

## 3. Environment — done, with two fixes worth knowing

Working and verified:

- **Connectome Workbench 1.5.0** installed to `~/opt/workbench`, added to PATH
  in `~/.config/fish/config.fish`. `wb_command -version` succeeds. This is the
  setup step CLAUDE.md §4.1 calls the most common failure.
- **Python 3.11.15 venv** at `.venv`, all dependencies installed and importing.

Two deviations from CLAUDE.md §4 that you should know about, both recorded in
`requirements.txt` comments:

1. **`abagen` is pinned to `==0.1.3` from PyPI**, below the `>=0.1.4` in §4.
   0.1.4 exists only as a GitHub tag, and tags can be force-moved whereas PyPI
   releases cannot — so installing from the tag is *less* reproducible, not
   more. Checked what we would lose: 0.1.4 differs from 0.1.3 by **three
   commits, all documentation**, the only code change being a citation string
   in `reporting.py`, which we do not call. The PyPI artifact is the stronger
   provenance. (Credit: Opus 5 caught this; the original reasoning was backwards.)
2. **`setuptools<81` is now pinned.** abagen 0.1.4 imports `pkg_resources`,
   which setuptools removed from the default install in v81. Without the pin,
   `import abagen` fails outright.

And one hazard I fixed: `uv` initially built the venv on
`~/.local/bin/python3.11`, which symlinks into
`~/Documents/projects/Loom/apps/desktop/src-tauri/resources/python/runtime/`.
A venv rooted in another project's vendored runtime breaks if Loom moves or
updates, which would silently violate R7. I installed a standalone uv-managed
3.11 and rebuilt on that. Your Loom symlink is untouched.

---

## 4. Phase 1 conda environment — the one real setup task left

`two_modes_of_hemodynamics/environment.yml` is a frozen full-export from the
authors' machine. Try it as-is first:

```fish
conda env create -f data/external/two_modes_of_hemodynamics/environment.yml
conda activate pr_postPLS_hist
```

If it fails to solve (likely — it pins `ca-certificates=2023.01.10` and similar),
**do not hand-edit pins one at a time.** Use their Dockerfile instead, which is
the environment they actually shipped:

```fish
docker build -t two_modes data/external/two_modes_of_hemodynamics/
```

Record whichever route worked in `results/p1_reproduction_notes.md` — Phase 1's
gate is "figures match published versions", and which environment produced them
is part of that record.

---

## 5. Commands to run, in order

```fish
# 0. Confirm the environment (should already pass)
wb_command -version
.venv/bin/python -m pytest -q          # 90 tests
.venv/bin/ruff check src/ scripts/ tests/

# 1. Survey what actually landed, once the fetch finishes
.venv/bin/python -m src.data.targets inspect --root data/raw/ds004873

# 2. AHBA — 4 GB, needed from Phase 3 onward. Start it whenever.
.venv/bin/python -c "import abagen; abagen.fetch_microarray(donors='all')"

# 3. Phase 0 gates — BLOCKED until DERIVATIVE_PATTERNS is filled in,
#    which is blocked on your answer to §1.
.venv/bin/python scripts/p0_reliability.py --config config/base.yaml
.venv/bin/python scripts/p0_dropout.py    --config config/base.yaml
```

---

## 6. Good news for Phase 0b

The dropout gate has a better proxy available than I expected. Snapshot 2.0.7
ships `derivatives/task-all_space-MNI152_res-2_SNR_YEO_group_mask.nii.gz`
(91×109×91, 2 mm MNI152) — a group SNR mask computed by the authors themselves,
already in the right space. That is a stronger and less arguable dropout proxy
than a tSNR map I would have derived myself, and using theirs means the gate
cannot be dismissed as my own construction.

Also present and useful later: Yeo 2011 7-network masks in MNI152 (DMN, FPN,
DAN, VAN, SMN, VIS), `VENAT_PartialVolume.nii.gz` (a venous atlas — directly
relevant to the mqBOLD vessel-geometry caveat in §14), and group median
`N40_cond-control_*_{oef,cbf,cmro2}` maps.

---

## 6b. Masking is not optional — a note for Phase 2

Sanity-checking the fetched maps turned up something that will matter:

| map | p5 | median | p95 |
|---|---:|---:|---:|
| `sub-p019_task-control_space-MNI152_oef` (raw) | 0.055 | 0.418 | **1.229** |
| `N40_cond-control_..._GMR2pCBVmasked_oef` (group) | 0.249 | 0.394 | 0.610 |

The group median OEF of **0.394** is textbook for human cortex, so the data are
sound. But the **per-subject maps are unmasked**, and an OEF above 1.0 is
physiologically impossible — the p95 of 1.23 is noise from voxels where the
R2′/CBV denominator is small.

The authors' own group map carries the suffix `GMR2pCBVmasked`, i.e. grey
matter ∩ R2′ ∩ CBV constrained. **Phase 2 must apply an equivalent mask before
parcellating**, or a handful of garbage voxels will drag whole parcel means.
Deriving the exact mask from their notebooks is a Phase 1 deliverable — one more
reason the reproduction step is worth doing rather than skipping.

---

## 7. What is built and tested

90 tests pass, ruff clean. The pieces that matter:

- `src/stats/spatial.py` — **R1 enforcement.** `corr_with_null()` has no code
  path to a p-value without surrogates; passing `nulls=None` raises. Tests
  include a calibration check that the null p-value is uniform under the null,
  and a check that it is more conservative than the naive p on autocorrelated
  maps. A bug in my first BH-FDR implementation (reversed rank divisor) was
  caught by a test and fixed; it now matches statsmodels exactly.
- `src/stats/reliability.py` — Phase 0a: split-half, Spearman-Brown, ICC(2,1),
  and the three-way gate verdict. Fully tested against synthetic data with
  known reliability.
- `src/data/fetch.py` — checksum-verified OpenNeuro fetch. Verifies every file
  against the SHA256 embedded in its git-annex key.
- `src/utils/manifest.py` — R10. Writes a manifest even when a run fails.
- `config/genesets.yaml` — **frozen per R5**, with `exploratory: []` empty and a
  test asserting it stays that way.

⚠️ One trap worth recording: the **S3 mirror at `s3://openneuro.org/ds004873`
serves snapshot 1.0.4, which contains no derivatives at all** — only MESE, BOLD
and T1w, 3.0 GB. The mqBOLD maps this project depends on only exist in the
2.0.x snapshots. The `aws s3 sync` command in CLAUDE.md §6 would therefore
fetch a dataset with none of the needed data and give no error. `src/data/fetch.py`
pins snapshot 2.0.7 explicitly and documents this.
