# Phase 1 — reproduction notes ⛔ GATE: **PASS** (exact)

Run 2026-07-26 against `two_modes_of_hemodynamics` @ `1b22c2cb` and ds004873
snapshot 2.0.7. Required by CLAUDE.md §9.

**Scope note.** This is a *scoped* reproduction: rather than regenerating every
published figure, it reproduces their group-level parameter maps exactly and
extracts the two pipeline conventions Phase 2 depends on. The quantitative gate
(`config: gates.p1_reproduction`, Spearman ≥ 0.8) is met with room to spare.

---

## Gate result

Reproducing their published group median CBF map from the per-subject MNI152
derivatives, using their recipe (upscale CBF by 1/0.75, then `np.nanmedian`
across 40 subjects):

| comparison | value |
|---|---:|
| Pearson r | **1.000000** |
| Spearman ρ | **1.000000** |
| max abs difference | 2.5 × 10⁻⁶ |
| median abs difference | 4.5 × 10⁻¹³ |
| voxels compared | 867,944 |

This is bit-for-bit agreement to floating-point precision. It validates the
input file selection, the CBF upscaling, and the aggregation rule.

---

## What the notebooks established

### 1. The design has four co-equal conditions

From `B_Fig1.ipynb`:

```python
if ID < 56:  conds = ['rest', 'control', 'mem', 'calc']
if ID > 55:  conds = ['control', 'calc']
```

**`control` is a task condition in its own right, not the control regressor of
the calculation task.** This settles the open §13.5 question in the direction
that makes a discordance-frequency map *well defined in their design*.

It does not, however, make it constructible. Only `calc` and `control` are
published in MNI152 at group scale; `mem` has CMRO₂ only (30 subjects, T2
space, no CBF, so no coupling ratio) and `rest` has essentially nothing. A
frequency map built from two of four conditions would misrepresent the
quantity. `DERIVATIVE_PATTERNS["discordance_freq"]` stays `None`.

### 2. Their OEF cap — and why our mask was wrong

From `A_preprocessing.ipynb`:

```python
mask   = BrMsk_CSF_arr > 0.5
rOEF   = (R2prime / (C * rCBV + eps)) * mask
rOEFmax = 5 * np.nanmedian(rOEF[mask])
if rOEFmax < 1.5: rOEFmax = 1.5
rOEF[rOEF > rOEFmax] = rOEFmax        # CLIPPED, not excluded
```

with `C = 4/3 · 267.61918 · π · 0.264 · (Hct/100) · 0.85 · 3` (subject-specific
via haematocrit) and

```python
CaO2  = 0.334 * Hct * 55.6 * O2sat/100
CMRO2 = (CBF/0.75) * OEF * CaO2 / 100
```

**The published MNI152 maps already carry this cap.** Verified empirically —
each subject's map maximum equals 5 × that subject's median to within
interpolation error:

| subject | max | 5 × median | frac > 1.0 |
|---|---:|---:|---:|
| p019 | 2.211 | 2.088 | 8.2% |
| p023 | 2.934 | 2.729 | 18.2% |
| p027 | 2.550 | 2.319 | 13.3% |

Our Phase 0 pipeline had applied `valid_range=(0, 1)` on the reasoning that
OEF > 1 is physiologically impossible. That was **wrong**: those values are
legitimate *clipped* data covering 8–18% of voxels, and discarding them biased
parcel means downward precisely in the high-OEF regions the hypothesis
concerns. Corrected in `src/data/targets.py::VALID_RANGES`.

Re-running Phase 0a after the correction changed the reliability estimates
negligibly (baseline OEF 0.978 → 0.978; coupling ratio 0.638 → 0.634), because
split-half reliability measures a *spatial pattern* and the error was close to
monotone. Absolute parcel values, however, were materially affected — which is
exactly why this mattered for a released dataset and not for the gate verdicts.

### 3. Group aggregation is the median, not the mean

`par_map_median = np.nanmedian(par_map, axis=3)`. We had been using the mean.
Corrected in `load_target_map`.

---

## ⚠️ A reproducibility gap in ds004873

Their analysis notebooks read `_qBmasked` files — the per-subject maps after
the grey-matter ∩ R2′ ∩ CBV masking that the group maps' `GMR2pCBVmasked`
suffix refers to:

```python
par_nii = os.path.join(dir_deriv,'qmri', sub + '_task-'+cond+'_space-T2_'+par+'_qBmasked.nii.gz')
```

**Those files are released for exactly one subject.** The full snapshot contains
24 `_qBmasked` files, all in T2 space, all from a single participant. The
per-subject masks (`space-T2_{GM,R2prime,cbf,oef}_mask.nii.gz`) are likewise
single-subject.

Consequence: **their per-subject masking cannot be reproduced from the public
release.** We can reproduce the *unmasked* group maps exactly (above), but not
the masked ones.

How much this matters, measured directly — comparing our unmasked
reconstruction against their published masked group map, on their own voxels:

| quantity | their mask keeps | Spearman ρ (unmasked vs masked) |
|---|---:|---:|
| OEF | 46,594 / 246,289 voxels (18.9%) | **0.663** |
| CMRO₂ | 46,594 / 238,773 voxels (19.5%) | **0.653** |

Value *distributions* stay close (OEF median 0.401 vs 0.394; CMRO₂ 134.4 vs
131.9), but the *spatial topography* diverges substantially. Per-subject
masking is not cosmetic.

**Decision for Phase 2:** group-level baseline quantities are taken from the
authors' published masked maps (`load_authors_group_map`), not reconstructed.
Subject-varying quantities — the coupling ratio, everything feeding reliability
— necessarily come from the unmasked per-subject maps and are flagged
`coupling_source = reconstructed_unmasked_n30` in the released table.

**This is worth raising with the authors.** Releasing the `_qBmasked`
derivatives for all 40 subjects would make their analysis fully reproducible
and would materially improve any downstream use of this dataset.

---

## A projection artifact this surfaced

Not from their code, but found while validating against it. Projecting their
masked group map to the surface with **trilinear** interpolation blends each
in-mask voxel with out-of-mask zeros, because the mask is a thin ribbon:

| interpolation | surface vertices | parcel median OEF |
|---|---:|---:|
| volumetric reference (their mask) | 46,594 voxels | **0.394** |
| linear | 7,458 | 0.248 ✗ |
| nearest | 4,922 | **0.379** ✓ |

Linear interpolation depressed every baseline: CBF read 27.7 instead of 45.0.
`load_authors_group_map` now uses nearest-neighbour. The per-subject maps are
dense and unmasked, so linear remains correct for those.

---

## Not done

- Figures 2–5 were not regenerated. The PLS, native-space and network analyses
  are not prerequisites for Phase 2 and were out of scope for a scoped run.
- Their conda environment was never built; the reproduction was done by reading
  their code and re-implementing the recipe in this repo's stack, which is a
  stronger test of understanding than re-executing their notebook but a weaker
  test of their environment.
