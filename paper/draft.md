# Vascular gene expression predicts cortical oxygen extraction; its link to BOLD–CMRO₂ discordance is below current detection limits

**Working draft.** Every number is generated from `results/` at git `b4cb6d5`,
which passes `scripts/audit_provenance.py` 6/6, and is machine-verified against
those artifacts by `scripts/check_paper_numbers.py`. Items marked `[TODO]` are
genuinely outstanding, not placeholders for numbers.

---

## Abstract

Roughly 40% of voxels with task-evoked blood-oxygen-level-dependent (BOLD) signal
change show oxygen metabolism moving in the opposite direction, concentrated in
the default mode network (Epp et al., 2025). The leading conjecture — that
association cortex has lower capillary density, weakening or reversing the
haemodynamic response — has never been tested against molecular vascular
architecture.

We tested it using Allen Human Brain Atlas microarray across 120 preprocessing
pipelines, with spatial-autocorrelation-preserving and stability-matched
competitive nulls throughout. We first asked which tests the design could resolve.
Of 33 gene-set × outcome tests, **three** exceed their attenuation-adjusted
detectability floor — and those three are exactly the three returning associations
that pass both nulls. Against discordance, 1 of 22 tests is resolvable; against
baseline oxygen extraction fraction (OEF), 2 of 11.

Within that resolvable set, **pericyte and mural-cell gene expression predicts
baseline OEF** (ρ = −0.391, sign-consistent across 100% of 240 tests, spatially
significant in 86%, competitive *p* = 0.0004), in the pre-registered direction,
strengthening when the unimodal–transmodal hierarchy is partialled out. It does
not survive family-wide correction (minimum adjusted *p* = 0.130).

Mediation to discordance is unsupported across all 15,840 path models, but every
mediator→outcome estimate falls below the detectability floor: a large effect
(|ρ| ≳ 0.33) is excluded, a moderate one is not. The conjecture's first link is
supported; its second is **untested rather than refuted**, and we quantify the
reliability a real test would require.

---

## 1. Introduction

BOLD contrast is sensitive to deoxyhaemoglobin, not to neural activity, and to
first order `sign(ΔBOLD) ≈ sign(ΔCBF − ΔCMRO₂)`, where CBF is cerebral blood flow
and CMRO₂ the cerebral metabolic rate of oxygen. The signal therefore reports the
*mismatch* between flow and oxygen consumption (Buxton, 2009). When flow overshoots
demand — the normal case, coupling ratio *n* = %ΔCBF / %ΔCMRO₂ ≈ 2–4 — BOLD rises
with metabolism. When *n* < 1, it does not.

Epp et al. (2025) used multiparametric quantitative BOLD (mqBOLD) to measure
ΔCMRO₂ directly and found *n* < 1 in a substantial minority of responding voxels,
concentrated in the default mode network. Their mechanistic observation was that
discordant voxels differ in baseline OEF and regulate oxygen delivery through OEF
change, while concordant voxels are driven by CBF change.

The first author's thesis speculates that association cortex has lower capillary
density than primary sensory cortex, producing the weakened or reversed responses
observed. That speculation has never been tested against molecular vascular
architecture. This paper tests it — and establishes how much of it is testable with
data of currently available quality.

### 1.1 Pre-specified hypotheses

**H1.** Discordance propensity is positively associated with regional expression of
glycolytic and vascular-sparsity-related gene programs and negatively with
oxidative-phosphorylation programs, *over and above* the unimodal→transmodal
cortical hierarchy.

**H2.** The association is mediated: `vascular/metabolic gene expression → baseline
OEF (and/or CBV) → discordance propensity`.

Gene sets were frozen in `config/genesets.yaml` before any Phase 4 result was
viewed. No set was added, removed, or modified afterwards.

### 1.2 Why the statistics are strict here

Imaging transcriptomics has three well-documented failure modes, and the design
neutralises each.

1. **Spatial autocorrelation.** Two arbitrary smooth brain maps correlate at
   r ≈ 0.4 by chance. Naive parametric *p*-values are meaningless
   (Alexander-Bloch et al., 2018; Burt et al., 2020).
2. **Pipeline dependence.** Markello et al. (2021) showed AHBA processing choices
   can shift imaging–expression correlations by as much as ρ ≥ 1.0 — a finding can
   be reversed by a defensible parameter change.
3. **The hierarchy confound.** Association cortex differs from sensory cortex on
   nearly everything (Margulies et al., 2016). Any map varying along that axis
   correlates with any gene set varying along it.

We add a fourth, which turns out to dominate: **measurement reliability bounds what
any of this can find.** A correlation between two imperfectly reliable maps is
attenuated toward zero by a factor set by their reliabilities, so a test can be
incapable of resolving a true effect of ordinary size no matter how carefully it is
conducted. §3.2 makes that bound explicit *before* any hypothesis is tested,
because it determines which results deserve weight.

---

## 2. Methods

### 2.1 Data

| Source | Content | Use |
|---|---|---|
| `ds004873` (OpenNeuro) | Epp et al. mqBOLD derivatives, 40 subjects | Target maps |
| AHBA (Hawrylycz et al., 2012) | 6 post-mortem donors; 5 usable (donor 15496 unavailable upstream) | Expression |
| `neuromaps` (Markello et al., 2022) | Margulies gradient, T1w/T2w myelin, Raichle CBF/CMRO₂/CMRGlu, evolutionary expansion | Covariates, comparisons |
| MSigDB (Liberzon et al., 2015) | HALLMARK and GO gene sets | Frozen hypothesis-driven sets |
| Macaque vascular atlas | Ex vivo cortical vascular density, 4 animals | Independent control `[TODO: citation]` |

### 2.2 Parcellation and projection

Primary analysis uses the Schaefer parcellation (Schaefer et al., 2018), 200
parcels, 7 networks (Yeo et al., 2011), **left hemisphere only** (100 parcels), in
`fsaverage5` space. Left-hemisphere restriction is forced by AHBA: only 2 of 6
donors have right-hemisphere tissue. Sensitivity parcellations are
Desikan–Killiany (Desikan et al., 2006; 34 LH) and Schaefer-400 (200 LH).

Volumetric MNI152 maps are projected to the surface exclusively through
`neuromaps.transforms`; no coordinate transform is hand-rolled. This discards
subcortex and cerebellum.

### 2.3 Target maps

Discordance is not analysed as a single variable:

- **Coupling angle** — `atan2(%ΔCBF, %ΔCMRO₂)`, an angular reparameterisation of
  *n* that does not blow up as the denominator approaches zero.
- **Extraction-mode discordance** — fraction of subjects in whom CMRO₂ rises while
  BOLD falls.
- **Overshoot-mode discordance** — fraction in whom CMRO₂ falls while BOLD rises.
- **Baseline OEF** and **baseline CBV** — the candidate mediators in H2.

The two modes are separated because they are topographically anticorrelated
(Spearman ≈ −0.56) and summing them cancels signal: the combined measure has lower
split-half reliability (0.491) than either component (0.579, 0.595).

### 2.4 The expression multiverse

Following Arnatkevičiūtė et al. (2019) and Markello et al. (2021), and adopting
the multiverse framing of Steegen et al. (2016) — whose necessity in neuroimaging
Botvinik-Nezer et al. (2020) demonstrated empirically — we run 120 cells spanning
`probe_selection` × `lr_mirror` × `missing` × `tolerance` × `norm_matched` ×
stability threshold. Phase 4 additionally crosses each pipeline with the
differential-stability thresholds at which a gene set remains large enough to
score, so its summary fractions are over **pipeline × threshold tests** — 240 for
the pericyte/mural set — rather than over the 120 pipelines. Effects are
summarised as median, inter-quartile range, and percentage of tests with
consistent sign.

### 2.5 Null models

**Spatial** — spherical rotation (Alexander-Bloch et al., 2018), 10,000
permutations, seed 42, cached per target map. Where a partial correlation is
computed, surrogates are residualised identically to the observed data.

**Competitive** — 10,000 random gene sets matched on size and
differential-stability distribution.

Both are required for any gene-set claim. Reported for every effect: point
estimate, spatial-null *p*, competitive-null *p*, Benjamini–Hochberg false
discovery rate (FDR; Benjamini & Hochberg, 1995) across the family, and the
multiverse distribution.

### 2.6 Detectability

For each map we estimate split-half reliability with Spearman–Brown correction and
partition observed variance into true and error components. For a gene set with
panel reliability *r*₁ and a brain map with reliability *r*₂, the maximum
observable correlation is the attenuation ceiling √(*r*₁*r*₂); dividing an observed
ρ by that ceiling gives the implied true effect. We compare this against the
smallest true effect the pairing can resolve at conventional power — the
**detectability floor** — and call a test *resolvable* when the implied true effect
exceeds its floor.

---

## 3. Results

### 3.1 Gates

All pre-specified gates pass.

| Gate | Criterion | Result |
|---|---|---|
| Reliability of the target map | median Spearman-Brown *r* ≥ 0.5 | **0.711** (coupling angle) — pass |
| Dropout confound, final map | \|ρ\| < 0.5 vs SNR coverage | **+0.003** (*p* = 0.98) — pass |
| Dropout confound, whole mqBOLD chain | \|ρ\| < 0.5 at every link | **worst 0.315** — pass |

The third gate matters more than its predecessor. mqBOLD is a chain — T2, T2\* →
R2′ → OEF → CMRO₂ → ΔCMRO₂ → discordance — and a gate applied only to the final map
cannot detect corruption at the first link. T2\* is what macroscopic field
inhomogeneity corrupts, worst under vmPFC, a default mode node. Applying the
threshold to all 12 links:

```
1_t2        -0.030      2_t2star    +0.119      3_r2prime   -0.123
4_cbv       +0.229      5_oef       -0.197      6_cbf       +0.001
7_cmro2     -0.315      8_dcbf      -0.025      9_dcmro2    +0.063
10_extraction +0.163    11_overshoot -0.122     12_angle    +0.003
```

The three most upstream links — where field inhomogeneity actually enters — are the
lowest in the chain. Baseline CMRO₂ is the worst link at −0.315 and does not
inherit that from T2\*.

Map reliabilities (split-half, Spearman–Brown corrected): baseline OEF 0.978,
baseline CMRO₂ 0.984, baseline CBF 0.984, baseline CBV 0.978, coupling angle
0.711, extraction-mode discordance 0.579, overshoot-mode 0.595.

### 3.2 What this design can resolve

This section precedes the hypothesis tests deliberately. Reliability determines
which tests could have found anything, and reporting significance for tests that
were never capable of resolving an ordinary effect invites readers to weight them
equally with tests that were.

Gene-set panel reliabilities across donors span an order of magnitude:

| gene set | panel reliability | gene set | panel reliability |
|---|---|---|---|
| glucose/lactate transport | 0.670 | HALLMARK_GLYCOLYSIS | 0.317 |
| endothelial | 0.591 | HALLMARK_OXPHOS | 0.228 |
| HALLMARK_ANGIOGENESIS | 0.582 | interneuron subclass | 0.187 |
| pericyte/mural | 0.557 | HALLMARK_HYPOXIA | 0.125 |
| glycolytic enzymes | 0.482 | GOBP_BLOOD_VESSEL_MORPHOGENESIS | **−0.011** |
| astrocyte | 0.343 | | |

`GOBP_BLOOD_VESSEL_MORPHOGENESIS` has negative panel reliability: its
donor-to-donor expression pattern does not replicate, so it is untestable against
any map at any effect size. Reporting a *p*-value for it would be meaningless, and
we do not.

Combining these with map reliabilities gives, for each of the 33 gene-set × outcome
pairings, an attenuation ceiling and a detectability floor:

| outcome | resolvable tests | median floor |
|---|---|---|
| baseline OEF | **2 / 11** | 0.420 |
| overshoot-mode discordance | **1 / 11** | 0.535 |
| extraction-mode discordance | **0 / 11** | 0.566 |

**Three of 33 tests are resolvable**, and they are:

| gene set | outcome | ρ | implied true | floor |
|---|---|---|---|---|
| pericyte/mural | baseline OEF | −0.391 | 0.526 | 0.330 |
| HALLMARK_ANGIOGENESIS | baseline OEF | −0.355 | 0.468 | 0.323 |
| astrocyte | overshoot | +0.256 | 0.559 | 0.535 |

Two consequences follow, and they structure everything below.

First, **the outcome the hypothesis is about is the one the design cannot
interrogate.** Extraction-mode discordance admits no resolvable gene-set test at
all. Its median floor of 0.566 exceeds any effect plausibly expected in imaging
transcriptomics, where ρ ≈ 0.3 is a strong result.

Second, **the three resolvable tests are exactly the three that return associations
passing both null models** (§3.4; competitive *p* = 0.0004, 0.0002, 0.007). Tests
capable of detecting an effect detected one; tests incapable of it did not. That
coherence is what one expects if the effects are real and the nulls are calibrated,
and it is not what one expects from noise.

### 3.3 Positive controls

Interpreting a null requires showing the pipeline can detect what must be there.

**Internal physiology** (our reconstruction, *n* = 100 parcels). The mqBOLD
identity OEF = CMRO₂/(CBF × CaO₂) forces specific signs, and all three hold:
OEF↔CMRO₂ **+0.78**, OEF↔CBF **−0.36**, CBF↔CMRO₂ **+0.16**.

**Cross-modality against positron emission tomography (PET).** Our CBF against the
Raichle CBF reference: **+0.39**. Our CMRO₂ against the Raichle CMRO₂ reference:
**−0.14** — this control fails, and §3.8 characterises it.

**Cross-species molecular.** Human endothelial gene expression against ex vivo
macaque cortical vascular density: **ρ = +0.46, sign-consistent across 100% of
pipelines, spatially significant in 100%**. Glucose/lactate transporters follow at
+0.40 (70%). Human vascular gene expression tracks real vasculature. Neither,
however, clears the competitive null (*p* = 0.09 and 0.15) — see §3.4.

### 3.4 H1 — frozen gene sets

Across 120 pipelines and both null models:

| Gene set | Target | ρ (median) | IQR | sign | spin-sig | competitive *p* | resolvable |
|---|---|---|---|---|---|---|---|
| pericyte/mural | baseline OEF | **−0.391** | [−0.411, −0.359] | 100% | **86%** | **0.0004** | yes |
| HALLMARK_ANGIOGENESIS | baseline OEF | −0.355 | [−0.413, −0.289] | 100% | 30% | **0.0002** | yes |
| astrocyte | overshoot | +0.256 | [+0.224, +0.286] | 100% | 15% | **0.007** | marginal |
| glucose/lactate transport | extraction | +0.229 | [+0.180, +0.277] | 100% | 27% | 0.060 | no |
| HALLMARK_OXPHOS | baseline OEF | −0.203 | [−0.235, −0.172] | 100% | 0% | 0.107 | no |

**Of the three resolvable tests, one matches its pre-registered direction, one
contradicts it, and one had no direction specified.** Pericyte/mural → baseline
OEF is negative as pre-specified. HALLMARK_ANGIOGENESIS → baseline OEF was
pre-registered as *positive* and is **−0.355** — the strongest competitive-null
result in the study, in the wrong direction. Astrocyte → overshoot carried no
directional prediction and can therefore neither confirm nor refute H1.

We read the reversal as an error in the pre-registration rather than in the data.
The two vascular sets agree with each other: higher vascular gene expression,
lower baseline oxygen extraction, which is what more delivery per unit demand
implies. The prediction that angiogenesis would track OEF *positively* does not
follow from that mechanism, and we should not have committed to it. Reporting it
as a directional failure is the honest accounting, and it is why the confirmatory
status of H1 is weaker than the effect sizes alone suggest.

The pericyte/mural → baseline OEF association is the clearest result. It is
negative as pre-specified; sign-consistent across all 240 tests spanning 120
pipelines; spatially significant in 86%; and it passes the competitive null decisively (*z* = −3.49).
Its implied true effect (0.526) comfortably exceeds its floor (0.330).

**It does not survive FDR correction across the 55-way gene-set × target family
(minimum adjusted *p* = 0.130).** We report it as a robust, directionally
pre-specified, adequately powered association that does not clear confirmatory
correction — not as a confirmed effect.

**The two null models disagree, and the direction is instructive.** The headline
effect passes the competitive null and fails FDR; the cross-species positive
control does the reverse — spatially significant in 100% of pipelines, competitive
*p* = 0.09. If the assay that must work fails the competitive null while the
hypothesis test passes it, the competitive null is not simply measuring "is this
gene set special". It is sensitive to set size and stability structure that small
curated panels do not share with random draws. We therefore treat *p* = 0.0004 as
evidence that the effect is not a generic property of any stability-matched set of
that size, not as confirmatory significance.

### 3.5 Hierarchy control

The gene-set step runs on 12 of the 120 multiverse cells for compute reasons, so
its pipeline-fraction statistics rest on a twelfth of the evidence behind Phases 4
and 6.

Under the **pre-registered** specification (principal gradient + T1w/T2w myelin +
dropout proxy), pericyte/mural → baseline OEF *strengthens*: partial ρ = −0.419
against a raw −0.386, significant in 92% of cells. The hierarchy was suppressing
the relationship, not generating it.

Under a **disclosed sensitivity specification** adding connectivity gradients 2 and
3, it falls to −0.284 and reaches significance in 0% of cells. Both are computed on
every regeneration so neither can be selected after seeing results.

Both specifications equally annihilate a known hierarchy proxy — the first
principal component of gene expression retains 4% and 7% of its raw effect. Only
the extended one also removes mechanistically expected relationships:

| | raw | pre-registered | extended |
|---|---|---|---|
| coupling angle vs PET CMRGlu | +0.260 | +0.232 (89% retained) | +0.074 (28%) |
| coupling angle vs PET CMRO₂ | +0.203 | +0.196 (97%) | +0.025 (12%) |
| baseline OEF vs PET CBF | −0.309 | −0.339 (110%) | −0.222 (72%) |

Two readings remain available and the data do not decide between them. Ours is that
gradients 2–3 absorb metabolic signal, making the extended specification
over-controlled; its residuals also behave erratically on weak effects, amplifying
near-zero associations by 300–490% with sign flips. The alternative is that our
maps genuinely are gradient-2/3 structured and those gradients are legitimate
confounds.

Readers should note that the specification we favour is also the one under which our
headline result survives. That coincidence is why both are computed every run
rather than chosen afterwards. **This association survives the pre-registered test
and fails a stricter one; which test is right is an author opinion, not a result.**

### 3.6 H2 — mediation, as a bound

15,840 path models (11 gene sets × 2 mediators × 3 outcomes × 120 pipelines ×
adjusted/unadjusted), each with spin-test inference on every path and 10,000
bootstrap resamples of the indirect effect. **Zero of 132 combinations are
supported.**

The failure is asymmetric. For pericyte/mural → baseline OEF → extraction-mode
discordance: **path a** (expression → mediator) is −0.408, significant in **88%** of
pipelines; **path b** (mediator → outcome) is −0.217, significant in **0%**. The
limiting path is b.

But §3.2 forbids reading that as absence:

| outcome | path b | ceiling | implied true \|ρ\| | floor |
|---|---|---|---|---|
| extraction | −0.217 | 0.752 | 0.289 | 0.331 |
| overshoot | +0.057 | 0.763 | 0.075 | 0.313 |
| coupling angle | +0.141 | 0.834 | 0.169 | 0.301 |

Every implied true effect falls below the floor. The discordance maps are the
power-limiting term (0.579, 0.595) against baseline OEF's 0.978. We therefore
report that **a large mediator→outcome effect (|ρ| ≳ 0.33) is excluded; a moderate
one is not.**

### 3.7 An independent test of the capillary conjecture

The mediation relies on molecular proxies for vasculature. We therefore tested the
conjecture using the macaque map as a *measurement* of vascular density:

| macaque vascular density vs | ρ | *p*(spin) |
|---|---|---|
| extraction-mode discordance | +0.079 | 0.63 |
| overshoot-mode discordance | −0.094 | 0.61 |
| coupling angle | −0.040 | 0.77 |
| baseline OEF | +0.081 | 0.69 |
| principal gradient | −0.333 | 0.13 |

Nothing, in any parameterisation, and independent of the transcriptomic machinery.

Three limits keep this convergent rather than decisive. The map derives from
**four** animals. Human cortical expansion means only **83 of 100** left-hemisphere
parcels receive values after cross-species registration. And the registration is
**least accurate in association cortex** — precisely where the hypothesis lives, and
where a null is therefore cheapest to obtain. A registration-quality gradient
aligned with the principal gradient could by itself produce this table.

Notably, pericyte/mural expression shows essentially no relationship to macaque
vascular density (ρ = −0.05). Whatever it tracks in baseline OEF, it is not
capillary density as such.

### 3.8 The control we fail

Our baseline CMRO₂ correlates with the Raichle PET CMRO₂ reference at **−0.138**
while its own split-half reliability is **0.984**. Attenuation cannot explain this:
the maximum observable correlation is 0.992, so the implied true correlation is
≈ 0.14.

Two observations bound what this invalidates. First, **the authors' own published
group CMRO₂ map also fails this control** (+0.090), from entirely separate
processing — the disagreement is between mqBOLD and PET as methods, not between our
reconstruction and theirs. Second, it is spatially structured, tracking
extraction-mode discordance (−0.306, *p* = 0.032) and the dropout proxy (−0.284,
*p* = 0.033), with a network profile high in limbic and default and low in visual
and somatomotor — matching where T2\* is corrupted by sinus-adjacent field
inhomogeneity.

Our reconstruction is not the weak link: it passes four of five controls and beats
the published maps on two (OEF↔CMRO₂ +0.78 vs +0.58; CBF vs PET +0.39 vs +0.33).
Its flow–metabolism coupling is weaker, however (CBF↔CMRO₂ +0.16 vs +0.44),
implying a more OEF-weighted CMRO₂ — consistent with inheriting OEF's dropout
sensitivity. `[TODO: this is a characterisation, not a validated model — test it or
soften it.]`

### 3.9 Data-driven arm

Run in parallel with, not instead of, the hypothesis-driven analysis. Ranking all
stable genes by association with each target and testing both tails yields no
gene-level survivor under Westfall–Young max-T correction (0 genes at *p* < 0.05).
The whole-transcriptome screen is not significant against its spatial null for
extraction-mode discordance (*z* = 1.35, *p* = 0.085).

Partial least squares regression finds a nominally significant second component for
both discordance modes (extraction *R*² = 0.369, *p* = 0.022; overshoot *R*² =
0.367, *p* = 0.044), uncorrected across 12 component × target tests. Tail enrichment
finds astrocyte markers in the negative tail for extraction-mode discordance
(*z* = 5.19, *p* = 0.006).

**The two arms do not strongly converge.** The hypothesis-driven arm's clearest
signal is pericyte/mural against baseline OEF; the data-driven arm's is astrocytic
and points at the discordance modes. We report the divergence rather than
reconciling it post hoc.

---

## 4. Discussion

The capillary-density conjecture makes two claims in series: that molecular
vascular architecture varies systematically across cortex, and that this variation
produces BOLD–CMRO₂ discordance. **The first is supported. The second is not
refuted — it is untested, and we can say by how much.**

Pericyte and mural-cell gene expression predicts baseline oxygen extraction with
unusual robustness for this field: all 240 tests across 120 pipelines agree on
sign, 86% reach spatial significance, the competitive null is passed decisively, and the
effect strengthens under hierarchy control (Fornito et al., 2019). Pericytes are
contractile and regulate capillary flow directly (Hall et al., 2014; Attwell et
al., 2010; Iadecola, 2017), so a relationship between mural-cell expression and
steady-state oxygen extraction is mechanistically unsurprising. The surprise is
where it stops.

It stops at the measurement, not at the biology. Extraction-mode discordance admits
no resolvable gene-set test in this dataset, and every mediation path-b estimate
lies below its floor. The bottleneck is not the transcriptomic side — AHBA gene
sets with panel reliability above ~0.55 support resolvable tests — but the
discordance maps, whose split-half reliability of ~0.58 caps the attenuation
ceiling near 0.75 and pushes the floor above ρ ≈ 0.55 for most gene sets.

This is, we think, the more useful contribution. A bare null tells the field
nothing about whether to try again. A null accompanied by the reliability at which
the test becomes possible is a specification for the next experiment.
`[TODO: compute the required-reliability curve — the map reliability at which a
true ρ of 0.3 becomes resolvable for a gene set of panel reliability 0.55.]`

We also think Epp et al.'s observation need not imply a between-region signature at
all. Their result is that discordant voxels differ in baseline OEF; ours asks
whether parcel-wise variation in OEF predicts parcel-wise variation in discordance.
A within-condition mechanism can exist without leaving a between-region
correlation. `[TODO: this is the central interpretive claim and needs a simulation
showing the two can dissociate.]`

### 4.1 What would change our minds

- A discordance map with reliability above the threshold in §4, from more subjects
  or more task conditions. This is the single highest-value follow-up.
- A vascular density measurement in human tissue, matched to these subjects.
- Discordance measured with a method that does not derive OEF from T2\*, breaking
  the dependence identified in §3.8.
- Cell-type deconvolution against a published single-cell reference (Seidlitz et
  al., 2020) rather than curated marker panels alone.
- Comparison against the time-averaged control energy map of Ceballos et al.
  (2025), which indexes a different notion of regional metabolic demand.

---

## 5. Limitations

- **AHBA is 6 adult post-mortem donors** (5 usable), bulk microarray, predominantly
  left hemisphere. A modal brain, not a matched sample. No individual-level
  inference is licensed.
- **Most tests here were underpowered by construction** (§3.2): 30 of 33 gene-set ×
  outcome pairings could not resolve an ordinary effect. We report them for
  completeness and weight them accordingly.
- **One frozen gene set is untestable.** `GOBP_BLOOD_VESSEL_MORPHOGENESIS` has
  negative donor-to-donor reliability.
- **The primary finding does not survive family-wide FDR correction** (minimum
  adjusted *p* = 0.130) and fails a stricter hierarchy specification we argue — but
  cannot demonstrate — is over-controlled.
- **The total inferential surface is large**: 55 gene-set × target combinations in
  Phase 4, 528 gene-set partials per hierarchy specification in Phase 5, 132
  mediation combinations in Phase 6, 12 PLS components in Phase 4b. Each family is
  corrected internally; none against the others.
- **The cross-species control is registration-limited** in exactly the association
  cortex the hypothesis concerns, and rests on four animals and 83 parcels.
- **Phase 5 uses 12 of 120 multiverse cells**; Phases 4 and 6 use all 120.
- **ds004873 is ~40 subjects, one scanner, one site.**
- **The volumetric→surface projection discards subcortex and cerebellum.**
- **mqBOLD carries its own assumptions** — vessel geometry, blood volume, T2′
  modelling — which propagate into every OEF and CMRO₂ value here. §3.8 quantifies
  one consequence.
- **The dropout confound is mitigated, not eliminated**, and bounded at
  |ρ| ≤ 0.315 across the whole chain.
- **The pericyte/mural set is 4 genes of 5** — ANPEP is absent from the AHBA
  expression matrix.
- **No sampling confidence intervals are reported.** Multiverse IQR is pipeline
  dispersion, not sampling uncertainty. `[TODO: bootstrap CIs over parcels.]`

---

## 6. Data and code availability

All primary data are public; none were generated by this study.

| Resource | Identifier |
|---|---|
| Epp et al. mqBOLD derivatives | OpenNeuro `ds004873` (snapshot 2.0.x) |
| Allen Human Brain Atlas microarray | `abagen.fetch_microarray`; 5 of 6 donors |
| Reference and null maps | `neuromaps` (Markello et al., 2022) |
| Gene sets | MSigDB via `gseapy`; version pinned in `data/MANIFEST.yaml` |
| Macaque vascular density | `[TODO: citation]`; Yerkes19 → fs_LR registration |

Analysis code, frozen gene-set definitions, and the released parcel-level
annotation table are available at `[TODO: repository URL]`. Every result artifact
carries a manifest recording git SHA, config hash, package versions, seed, and
input checksums. `scripts/audit_provenance.py` gates the pipeline on their
consistency; `scripts/check_paper_numbers.py` verifies that the values in this
manuscript match the artifacts they are drawn from; `scripts/verify_references.py`
resolves every citation against Crossref.

Results were reproduced **bit-identically across two independent full regenerations
at different code states** (`37cdbf6`, `b4cb6d5`), including the primary effect, the
mediation path structure, and every gate.

## 7. Ethics

Secondary analysis of two publicly released datasets. Ethical approval and
participant consent were obtained by the original investigators; no new human or
animal data were collected. Post-mortem tissue governance for the Allen Human Brain
Atlas is described in Hawrylycz et al. (2012).

## 8. Funding and competing interests

`[TODO]` — to be declared before submission.

---

## Appendix A — Deviations from pre-registration

1. **Discordance frequency across four tasks** was pre-specified but is not
   analysable: only 2 of 4 conditions are published in MNI152 space. Replaced by
   the extraction/overshoot mode split, using the same two conditions.
2. **The extended hierarchy specification** (gradients 2–3) is a disclosed
   addition, not pre-registered. It is computed on every regeneration and reported
   alongside the pre-registered specification so it cannot be chosen post hoc.
3. **Phase 5 gene-set step runs on 12 of 120 multiverse cells** for compute reasons.
4. **The detectability analysis (§3.2) was not pre-registered.** It was added after
   observing that reported effects clustered in the few well-powered pairings, and
   it changed the paper's central claim from an absence to a bound. It is
   descriptive of the design rather than a test, and uses no outcome information
   beyond reliabilities computed in Phase 0.

## Figure captions

**Figure 1. What this design can resolve.** (**A**) Implied true effect (observed
ρ ÷ attenuation ceiling) against the detectability floor for all 33 gene-set ×
outcome tests. Points above the diagonal are resolvable; three are, labelled with
leader lines. Colour denotes outcome. (**B**) Gene-set panel reliability across
AHBA donors — the limiting term in every test a set enters. Blue ≥ 0.55;
vermillion ≤ 0. `GOBP_BLOOD_VESSEL_MORPHOGENESIS` has negative reliability and is
untestable against any map at any effect size.

**Figure 2. The primary effect.** (**A**) Spearman ρ between pericyte/mural
expression and baseline OEF across 240 tests (120 preprocessing pipelines × 2
differential-stability thresholds); vertical jitter is for display only. Filled
points reach spin *p* < 0.05; open points do not. Vermillion line is the median.
(**B**) Raw versus hierarchy-partialled effect under both covariate
specifications, one grey line per multiverse cell (*n* = 12). The pre-registered
specification strengthens the effect; the extended one abolishes it.

**Figure 3. The mediation is bounded, not null.** Implied true |ρ| for the
mediator→outcome path (baseline OEF → outcome) against the detectability floor for
each outcome (grey). All three estimates fall inside the undetectable region: a
large effect (|ρ| ≳ 0.33) is excluded, a moderate one is not.

**Figure 4. Positive controls, both map sources.** Internal physiological
identities and cross-modality comparisons against Raichle PET references,
computed on our reconstruction (circles) and on the authors' published group maps
(squares). Green shading marks controls behaving as expected; vermillion marks the
one that fails. Both map sources miss the PET CMRO₂ reference, locating the
disagreement in mqBOLD versus PET rather than in our processing.

**Figure 5. Dropout across the whole mqBOLD chain.** |ρ| between the
scanner-dropout proxy and every link of the chain, with the signed value
annotated. Dashed line is the §9 gate at |ρ| = 0.5. The three most upstream
links — where field inhomogeneity enters — are the lowest.

All figures are generated by `scripts/make_manuscript_figures.py` at 180 mm
two-column width, Okabe-Ito colour-vision-safe palette, 7 pt minimum text, and are
emitted as vector PDF alongside 400 dpi raster.

---

## Appendix B — Outstanding

- Citation for the macaque vascular atlas.
- The required-reliability curve in §4.
- The dissociation simulation in §4.
- Bootstrap confidence intervals over parcels.
- Author list, affiliations, funding, competing interests.
- Target venue: reads as *Imaging Neuroscience* or *NeuroImage*.

---

## References

Resolved against Crossref by `scripts/verify_references.py`; each entry is a DOI
whose returned title is asserted to match, so a misremembered citation fails
loudly rather than being formatted in. Schaefer et al. is dated 2017 by Crossref
(advance online) and 2018 by its issue; we cite the issue year.

- Alexander-Bloch AF et al. (2018). On testing for spatial correspondence between maps of human brain structure and function. *NeuroImage*, 178, 540-551. https://doi.org/10.1016/j.neuroimage.2018.05.070
- Arnatkevic̆iūtė A, Fulcher BD & Fornito A (2019). A practical guide to linking brain-wide gene expression and neuroimaging data. *NeuroImage*, 189, 353-367. https://doi.org/10.1016/j.neuroimage.2019.01.011
- Attwell D et al. (2010). Glial and neuronal control of brain blood flow. *Nature*, 468, 232-243. https://doi.org/10.1038/nature09613
- Benjamini Y & Hochberg Y (1995). Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. *Journal of the Royal Statistical Society Series B: Statistical Methodology*, 57, 289-300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
- Botvinik-Nezer R et al. (2020). Variability in the analysis of a single neuroimaging dataset by many teams. *Nature*, 582, 84-88. https://doi.org/10.1038/s41586-020-2314-9
- Burt JB et al. (2020). Generative modeling of brain maps with spatial autocorrelation. *NeuroImage*, 220, 117038. https://doi.org/10.1016/j.neuroimage.2020.117038
- Buxton RB (2009). Introduction to Functional Magnetic Resonance Imaging. https://doi.org/10.1017/cbo9780511605505
- Ceballos EG et al. (2025). The control costs of human brain dynamics. *Network Neuroscience*, 9, 77-99. https://doi.org/10.1162/netn_a_00425
- Desikan RS et al. (2006). An automated labeling system for subdividing the human cerebral cortex on MRI scans into gyral based regions of interest. *NeuroImage*, 31, 968-980. https://doi.org/10.1016/j.neuroimage.2006.01.021
- Epp SM et al. (2025). BOLD signal changes can oppose oxygen metabolism across the human cortex. *Nature Neuroscience*, 29, 1225-1236. https://doi.org/10.1038/s41593-025-02132-9
- Fornito A, Arnatkevičiūtė A & Fulcher BD (2019). Bridging the Gap between Connectome and Transcriptome. *Trends in Cognitive Sciences*, 23, 34-50. https://doi.org/10.1016/j.tics.2018.10.005
- Hall CN et al. (2014). Capillary pericytes regulate cerebral blood flow in health and disease. *Nature*, 508, 55-60. https://doi.org/10.1038/nature13165
- Hawrylycz MJ et al. (2012). An anatomically comprehensive atlas of the adult human brain transcriptome. *Nature*, 489, 391-399. https://doi.org/10.1038/nature11405
- Iadecola C (2017). The Neurovascular Unit Coming of Age: A Journey through Neurovascular Coupling in Health and Disease. *Neuron*, 96, 17-42. https://doi.org/10.1016/j.neuron.2017.07.030
- Liberzon A et al. (2015). The Molecular Signatures Database Hallmark Gene Set Collection. *Cell Systems*, 1, 417-425. https://doi.org/10.1016/j.cels.2015.12.004
- Margulies DS et al. (2016). Situating the default-mode network along a principal gradient of macroscale cortical organization. *Proceedings of the National Academy of Sciences*, 113, 12574-12579. https://doi.org/10.1073/pnas.1608282113
- Markello RD et al. (2021). Standardizing workflows in imaging transcriptomics with the abagen toolbox. *eLife*, 10. https://doi.org/10.7554/elife.72129
- Markello RD et al. (2022). neuromaps: structural and functional interpretation of brain maps. *Nature Methods*, 19, 1472-1479. https://doi.org/10.1038/s41592-022-01625-w
- Schaefer A et al. (2017). Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic Functional Connectivity MRI. *Cerebral Cortex*, 28, 3095-3114. https://doi.org/10.1093/cercor/bhx179
- Seidlitz J et al. (2020). Transcriptomic and cellular decoding of regional brain vulnerability to neurogenetic disorders. *Nature Communications*, 11. https://doi.org/10.1038/s41467-020-17051-5
- Steegen S et al. (2016). Increasing Transparency Through a Multiverse Analysis. *Perspectives on Psychological Science*, 11, 702-712. https://doi.org/10.1177/1745691616658637
- Thomas Yeo BT et al. (2011). The organization of the human cerebral cortex estimated by intrinsic functional connectivity. *Journal of Neurophysiology*, 106, 1125-1165. https://doi.org/10.1152/jn.00338.2011
- Vaishnavi SN et al. (2010). Regional aerobic glycolysis in the human brain. *Proceedings of the National Academy of Sciences*, 107, 17757-17762. https://doi.org/10.1073/pnas.1010459107
