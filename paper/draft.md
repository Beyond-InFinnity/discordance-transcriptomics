# Vascular gene expression predicts baseline oxygen extraction but not BOLD–CMRO₂ discordance

**Working draft.** All numbers are generated from `results/` at git `b4cb6d5`, which
passes `scripts/audit_provenance.py` 6/6. Every value in this document should be
traceable to a named artifact; where one is not, it is marked `[TODO]`.

---

## Abstract

Roughly 40% of voxels showing task-evoked blood-oxygen-level-dependent (BOLD)
signal change exhibit oxygen metabolism moving in the opposite direction,
concentrated in the default mode network (Epp et al., 2025). The mechanism is
unknown. The leading conjecture — that association cortex has lower capillary
density than primary sensory cortex, weakening or reversing the haemodynamic
response — has never been tested against molecular vascular architecture.

Using Allen Human Brain Atlas microarray across 120 preprocessing pipelines,
with spatial-autocorrelation-preserving and stability-matched competitive nulls
throughout, we find that **pericyte and
mural-cell gene expression predicts baseline oxygen extraction fraction (OEF)**
(ρ = −0.39, sign-consistent across 100% of pipelines, spatially significant in
86%, competitive *p* = 0.0004), in the pre-registered direction. The association
**strengthens** when the unimodal–transmodal cortical hierarchy is partialled
out.

The conjecture nonetheless fails, and we locate where. Across 15,840 path models, the exposure→mediator path is supported in 88% of pipelines and the
mediator→outcome path in **0%**. Every mediator→outcome estimate falls below this
design's detectability floor, so we report a bound rather than an absence: a large
effect (|ρ| ≳ 0.33) is excluded, a moderate one is not. Independently, an ex vivo
macaque vascular density map is unrelated to discordance (|ρ| ≤ 0.09, all
*p* > 0.6), though its cross-species registration is weakest in
association cortex.

The molecular half of the capillary-density conjecture holds; the vascular half
is bounded well below what the conjecture requires. We report a structured negative result with the
positive controls that make it interpretable and the one control we fail.

---

## 1. Introduction

BOLD contrast is sensitive to deoxyhaemoglobin, not to neural activity, and to
first order `sign(ΔBOLD) ≈ sign(ΔCBF − ΔCMRO₂)`, where CBF is cerebral blood flow
and CMRO₂ the cerebral metabolic rate of oxygen. The signal therefore reports the
*mismatch* between flow and oxygen consumption (Buxton, 2009). When flow overshoots demand — the
normal case, coupling ratio *n* = %ΔCBF / %ΔCMRO₂ ≈ 2–4 — BOLD rises with
metabolism. When *n* < 1, it does not.

Epp et al. (2025, *Nature Neuroscience*, doi:10.1038/s41593-025-02132-9) used
multiparametric quantitative BOLD (mqBOLD) to measure ΔCMRO₂ directly and found
*n* < 1 in a substantial minority of responding voxels, concentrated in the default
mode network. Their mechanistic observation was that discordant voxels differ in
baseline OEF and regulate oxygen delivery through OEF change, while concordant
voxels are driven by CBF change.

The first author's thesis speculates that association cortex has lower capillary
density than primary sensory cortex, and that this could produce the weakened or
reversed responses observed. **That speculation has never been tested against
molecular vascular architecture.** This paper tests it.

### 1.1 Pre-specified hypotheses

**H1.** Discordance propensity is positively associated with regional expression of
glycolytic and vascular-sparsity-related gene programs and negatively with
oxidative-phosphorylation programs, *over and above* the unimodal→transmodal
cortical hierarchy.

**H2.** The association is mediated:
`vascular/metabolic gene expression → baseline OEF (and/or CBV) → discordance
propensity`.

Gene sets were frozen in `config/genesets.yaml` before any Phase 4 result was
viewed. No set was added, removed, or modified afterwards.

### 1.2 Why the statistics are strict here

Imaging transcriptomics has three well-documented failure modes, and the design
below exists to neutralise each.

1. **Spatial autocorrelation.** Two arbitrary smooth brain maps correlate at
   r ≈ 0.4 by chance. Naive parametric *p*-values are meaningless
   (Alexander-Bloch et al., 2018; Burt et al., 2020).
2. **Pipeline dependence.** Markello et al. (2021) showed AHBA
   processing choices can shift imaging–expression correlations by as much as
   ρ ≥ 1.0 — a finding can be reversed by a defensible parameter change.
3. **The hierarchy confound.** Association cortex differs from sensory cortex on
   nearly everything (Margulies et al., 2016). Any map varying along that axis
   correlates with any gene set varying along it.

Accordingly: every spatial correlation carries a spatial-autocorrelation-preserving
null; every gene-set result additionally carries a competitive null matched on set
size *and* differential stability; every effect is reported as a distribution
across 120 preprocessing pipelines rather than a point estimate; and the decisive
analysis partials the cortical hierarchy.

---

## 2. Methods

### 2.1 Data

| Source | Content | Use |
|---|---|---|
| `ds004873` (OpenNeuro) | Epp et al. mqBOLD derivatives, 40 subjects | Target maps |
| AHBA (Hawrylycz et al., 2012) | 6 post-mortem donors (5 usable; donor 15496 is unavailable upstream, HTTP 404) | Expression |
| `neuromaps` (Markello et al., 2022) | Margulies principal gradient, T1w/T2w myelin, Raichle CBF/CMRO₂/CMRGlu, evolutionary expansion | Covariates and comparison maps |
| MSigDB (Liberzon et al., 2015) | HALLMARK and GO gene sets | Frozen hypothesis-driven sets |
| Macaque vascular atlas | Ex vivo cortical vascular density | Independent positive control `[TODO: full citation]` |

### 2.2 Parcellation and projection

Primary analysis uses the Schaefer parcellation (Schaefer et al., 2018), 200
parcels, 7 networks (Yeo et al., 2011), **left hemisphere
only** (100 parcels), in `fsaverage5` space. Left-hemisphere restriction is forced
by AHBA: only 2 of 6 donors have right-hemisphere tissue. Sensitivity parcellations
are Desikan–Killiany (Desikan et al., 2006; 34 LH) and Schaefer-400 (200 LH).

Volumetric MNI152 maps are projected to the surface exclusively through
`neuromaps.transforms`; no coordinate transform is hand-rolled. This discards
subcortex and cerebellum, which we state as a limitation rather than a footnote.

### 2.3 Target maps

Discordance is not analysed as a single variable. We use:

- **Coupling angle** — `atan2(%ΔCBF, %ΔCMRO₂)`, an angular reparameterisation of
  *n* that does not blow up as the denominator approaches zero.
- **Extraction-mode discordance** — fraction of subjects in whom CMRO₂ rises while
  BOLD falls (demand up, flow lags).
- **Overshoot-mode discordance** — fraction in whom CMRO₂ falls while BOLD rises
  (flow delivered in excess of a reduced demand).
- **Baseline OEF** and **baseline CBV** — the candidate mediators in H2.

The two discordance modes are treated separately because they are topographically
anticorrelated (Spearman ≈ −0.56) and summing them cancels signal: the combined
measure is *less* reliable (split-half 0.491) than either component (0.579, 0.595).

### 2.4 The expression multiverse

Following Arnatkevičiūtė et al. (2019) and Markello et al. (2021), and adopting
the multiverse framing of Steegen et al. (2016) — whose necessity in neuroimaging
Botvinik-Nezer et al. (2020) demonstrated empirically — we run
120 cells spanning `probe_selection` × `lr_mirror` × `missing` × `tolerance` ×
`norm_matched` × stability threshold. Every reported effect is summarised as
median, inter-quartile range, and percentage of cells with consistent sign.

### 2.5 Null models

**Spatial** — spherical rotation (Alexander-Bloch et al., 2018), 10,000
permutations, seed 42,
cached per target map. Where a partial correlation is computed, surrogates are
residualised identically to the observed data.

**Competitive** — 10,000 random gene sets matched on size and differential-stability
distribution.

Both are required for any gene-set claim. Reported for every effect: point
estimate, spatial-null *p*, competitive-null *p*, Benjamini–Hochberg false
discovery rate (FDR; Benjamini & Hochberg, 1995) across the family, and the multiverse distribution.

---

## 3. Results

### 3.1 Gates

All pre-specified gates pass.

| Gate | Criterion | Result |
|---|---|---|
| Reliability of the target map | median Spearman-Brown *r* ≥ 0.5 | **0.711** (coupling angle) — pass |
| Dropout confound, final map | \|ρ\| < 0.5 vs SNR coverage | **+0.003** (*p* = 0.98) — pass |
| Dropout confound, whole mqBOLD chain | \|ρ\| < 0.5 at every link | **worst 0.315** — pass |

The third gate deserves comment. mqBOLD is a chain — T2, T2\* → R2′ → OEF → CMRO₂
→ ΔCMRO₂ → discordance — and a gate applied only to the final map cannot detect
corruption at the first link. T2\* is precisely what macroscopic field
inhomogeneity corrupts, worst under vmPFC, a default mode node. We therefore
applied the threshold to all 12 links:

```
1_t2        -0.030      2_t2star    +0.119      3_r2prime   -0.123
4_cbv       +0.229      5_oef       -0.197      6_cbf       +0.001
7_cmro2     -0.315      8_dcbf      -0.025      9_dcmro2    +0.063
10_extraction +0.163    11_overshoot -0.122     12_angle    +0.003
```

The three most upstream links — the ones where field inhomogeneity actually enters
— are the *lowest* in the chain. The confound does not propagate from the top.
Baseline CMRO₂ is the worst link at −0.315, and it does not inherit that from T2\*.

Map reliabilities (split-half, Spearman-Brown corrected): baseline OEF 0.978,
baseline CMRO₂ 0.984, baseline CBF 0.984, baseline CBV 0.978, coupling angle 0.711.

### 3.2 Positive controls

Interpreting a null requires showing the pipeline can detect what must be there.

**Internal physiology** (our reconstruction, *n* = 100 parcels). The mqBOLD
identity OEF = CMRO₂/(CBF × CaO₂) forces specific signs, and all three hold:
OEF↔CMRO₂ **+0.78**, OEF↔CBF **−0.36**, CBF↔CMRO₂ **+0.16**.

**Cross-modality against positron emission tomography (PET).** Our CBF against the
Raichle CBF reference: **+0.39**. Our CMRO₂ against the Raichle CMRO₂ reference:
**−0.14** — *this control fails* (§3.6).

**Cross-species molecular.** Human endothelial gene expression against an ex vivo
macaque cortical vascular density map: **ρ = +0.46, sign consistent in 100% of
pipelines, spatially significant in 100%**. Glucose/lactate transporters follow at
+0.40 (70%). Human vascular gene expression does track real vasculature. Both,
however, fall short of the competitive null (*p* = 0.09 and 0.15) — they are not
more vascular-predictive than random size- and stability-matched sets, which we
note rather than obscure.

### 3.3 H1 — frozen gene sets

Across 120 pipelines and both null models, the strongest associations are:

| Gene set | Target | ρ (median) | IQR | sign | spin-sig | competitive *p* |
|---|---|---|---|---|---|---|
| pericyte/mural | baseline OEF | **−0.391** | [−0.411, −0.359] | 100% | **86%** | **0.0004** |
| HALLMARK_ANGIOGENESIS | baseline OEF | −0.355 | [−0.413, −0.289] | 100% | 30% | **0.0002** |
| astrocyte | overshoot | +0.256 | [+0.224, +0.286] | 100% | 15% | **0.007** |
| glucose/lactate transport | extraction | +0.229 | [+0.180, +0.277] | 100% | 27% | 0.060 |
| HALLMARK_OXPHOS | baseline OEF | −0.203 | [−0.235, −0.172] | 100% | 0% | 0.107 |

The pericyte/mural → baseline OEF association is the clearest result in the study.
It is negative, as pre-specified; sign-consistent across every one of 120
pipelines; spatially significant in 86% of them; and it passes the competitive null
decisively (*z* = −3.49). Its effect size sits above the measured detectability
floor for that pairing (0.330) and well below the attenuation ceiling (0.742), so
it is neither unmeasurable nor suspiciously clean.

**It does not survive FDR correction across the 55-way gene-set × target family
(minimum adjusted *p* = 0.130).** We report it as a robust, directionally
pre-specified association that does not clear family-wide correction, and we do not
present it as confirmed.

**The two null models disagree, and the direction is instructive.** The headline
effect passes the competitive null decisively (*p* = 0.0004) but fails FDR. The
cross-species positive control does the reverse: spatially significant in 100% of
pipelines, yet competitive *p* = 0.09. If the assay that must work fails the
competitive null while the hypothesis test passes it, the competitive null is not
simply measuring "is this gene set special". It is sensitive to set size and
stability structure in ways that small curated sets (pericyte/mural, 4 genes)
and broader ones (endothelial, 6 genes) do not share. We therefore treat
*p* = 0.0004 as evidence that the effect is not a generic property of any
stability-matched set of that size — not as confirmatory significance.

### 3.4 Hierarchy control

The gene-set step here runs on 12 of the 120 multiverse cells for compute reasons,
so its pipeline-fraction statistics rest on a twelfth of the evidence behind
Phases 4 and 6. Reference-map results below use the full parcel set.

Under the **pre-registered** specification (principal gradient + T1w/T2w myelin +
dropout proxy), pericyte/mural → baseline OEF *strengthens*: partial ρ = −0.419
against a raw −0.386, significant in 92% of cells. The hierarchy was suppressing
the relationship, not generating it.

Under a **disclosed sensitivity specification** adding connectivity gradients 2 and
3, it falls to −0.284 and reaches significance in 0% of cells. Both specifications
are run on every regeneration so neither can be selected after seeing results.

We judge the extended specification over-controlled, on the evidence of the
controls rather than on preference. Both specifications equally annihilate a known
hierarchy proxy (the first principal component of gene expression: 4% and 7% of the
raw effect retained). But only the extended one destroys relationships that are
mechanistically expected:

| | raw | pre-registered | extended |
|---|---|---|---|
| coupling angle vs PET CMRGlu | +0.260 | +0.232 (89% retained) | +0.074 (28%) |
| coupling angle vs PET CMRO₂ | +0.203 | +0.196 (97%) | +0.025 (12%) |
| baseline OEF vs PET CBF | −0.309 | −0.339 (110%) | −0.222 (72%) |

These are metabolic maps against metabolic quantities. A control that removes them
is removing physiology, not confound. The extended residuals additionally behave
erratically on weak effects, amplifying near-zero associations by 300–490% with
sign flips — the signature of over-fitting a five-covariate, spatially smooth
design on 100 parcels.

Two readings remain available and the data do not decide between them. Ours is
that gradients 2–3 absorb metabolic signal, making the extended specification
over-controlled. The alternative is that our maps genuinely are gradient-2/3
structured and those gradients are legitimate confounds, in which case the effect
should not survive.

Readers should note that the specification we favour is also the one under which
our headline result survives. That is a coincidence worth being suspicious of, and
it is why both specifications are computed on every regeneration rather than
selected afterwards. **This association survives the pre-registered test and fails
a stricter one; which test is right is an author opinion, not a result.**

### 3.5 H2 — mediation fails at the second link

15,840 path models (11 gene sets × 2 mediators × 3 outcomes × 120 pipelines ×
adjusted/unadjusted), each with spin-test inference on every path and 10,000
bootstrap resamples of the indirect effect.

**Zero of 132 gene-set × mediator × outcome combinations are supported.**

The failure is asymmetric and informative. For pericyte/mural → baseline OEF →
extraction-mode discordance:

- **path a** (expression → mediator): −0.408, significant in **88%** of pipelines
- **path b** (mediator → outcome): −0.217, significant in **0%**
- limiting path: **b**

Baseline OEF does not predict discordance *detectably* in these data, so nothing
can be mediated through it regardless of how well gene expression predicts OEF.

**How large an effect can we exclude?** This matters more than the null itself,
and the reliabilities in §3.1 answer it. The discordance maps are the
power-limiting term (split-half 0.579 and 0.595) against baseline OEF's 0.978:

| outcome | path b | ceiling √(r₁r₂) | implied true \|ρ\| | detectability floor |
|---|---|---|---|---|
| extraction | −0.217 | 0.752 | 0.289 | 0.331 |
| overshoot | +0.057 | 0.763 | 0.075 | 0.313 |
| coupling angle | +0.141 | 0.834 | 0.169 | 0.301 |

Every implied true effect falls **below** the floor this design can resolve. We
therefore report a bound rather than an absence: **a large mediator→outcome
effect (|ρ| ≳ 0.33) is excluded; a moderate one is not.** Claiming that baseline
OEF plays no causal role would overstate what a 100-parcel design with a
reliability-0.58 outcome can establish.

### 3.6 An independent test of the capillary conjecture

The mediation result relies on molecular proxies for vasculature. We therefore
tested the conjecture directly, using the ex vivo macaque vascular density map as a
*measurement* of vascular density rather than a proxy for it:

| macaque vascular density vs | ρ | *p*(spin) |
|---|---|---|
| extraction-mode discordance | +0.079 | 0.63 |
| overshoot-mode discordance | −0.094 | 0.61 |
| coupling angle | −0.040 | 0.77 |
| baseline OEF | +0.081 | 0.69 |
| principal gradient | −0.333 | 0.13 |

Nothing. An independent, direct measurement of cortical vascular density shows no
relationship to discordance in any parameterisation, and it does not depend on the
transcriptomic machinery at all.

Three limits keep this convergent rather than decisive. The map derives from
**four** macaques. Human cortical expansion means that after cross-species surface
registration only **83 of 100** left-hemisphere parcels receive values. And the
registration is **least accurate in association cortex** — which is precisely
where the hypothesis lives, and where a null is therefore cheapest to obtain. A
registration-quality gradient aligned with the principal gradient could by itself
produce the result in this table, and we cannot exclude that.

Consistently, pericyte/mural expression itself shows essentially no relationship to
macaque vascular density (ρ = −0.05). Whatever pericyte gene expression is tracking
in baseline OEF, it is **not** simply capillary density.

### 3.7 The control we fail

Our baseline CMRO₂ correlates with the Raichle PET CMRO₂ reference at only
**−0.138**, while its own split-half reliability is **0.984**. Attenuation cannot
explain this: with that reliability the maximum observable correlation is 0.992, so
the implied true correlation is ≈ 0.14.

Two observations bound what this invalidates.

First, **the authors' own published group CMRO₂ map also fails this control**
(+0.090), from entirely separate processing. The disagreement is between mqBOLD and
PET as methods, not between our reconstruction and theirs.

Second, the disagreement is spatially structured, tracking extraction-mode
discordance (−0.306, *p* = 0.032) and the dropout proxy (−0.284, *p* = 0.033), with
a network profile high in limbic and default and low in visual and somatomotor —
matching where T2\* is corrupted by sinus-adjacent field inhomogeneity.

Our reconstruction is not the weak link: it passes four of five controls and beats
the published maps on two (OEF↔CMRO₂ +0.78 vs +0.58; CBF vs PET +0.39 vs +0.33).
But its flow–metabolism coupling is weaker (CBF↔CMRO₂ +0.16 vs +0.44), implying our
CMRO₂ is more OEF-weighted, which is consistent with it inheriting OEF's
dropout sensitivity. `[TODO: this is a characterisation, not a validated model —
either test it or soften the claim.]`

### 3.8 Data-driven arm

Run in parallel with, not instead of, the hypothesis-driven analysis. Ranking all
stable genes by association with each target and testing both tails yields no
gene-level survivor under Westfall–Young max-T correction (0 genes at *p* < 0.05).
The whole-transcriptome screen is not significant against its spatial null for
extraction-mode discordance (*z* = 1.35, *p* = 0.085).

Partial least squares regression finds a nominally significant second component for
both discordance modes (extraction *R*² = 0.369, *p* = 0.022; overshoot *R*² =
0.367, *p* = 0.044), uncorrected across 12 component × target tests. Tail
enrichment finds astrocyte markers in the negative tail for extraction-mode
discordance (*z* = 5.19, *p* = 0.006).

**The two arms do not strongly converge.** The hypothesis-driven arm's clearest
signal is pericyte/mural against baseline OEF; the data-driven arm's is astrocytic
and points at the discordance modes. We report the divergence rather than
reconciling it post hoc.

---

## 4. Discussion

The capillary-density conjecture makes two claims in series: that molecular
vascular architecture varies systematically across cortex, and that this variation
produces BOLD–CMRO₂ discordance. **Our data support the first and reject the
second.**

Pericyte and mural-cell gene expression predicts baseline oxygen extraction with
unusual robustness for this field — every one of 120 defensible pipelines agrees on
sign, 86% reach spatial significance, the competitive null is passed decisively,
and the effect strengthens under hierarchy control. This is a real feature of
cortical organisation, and it is the kind of molecular–physiological coupling the
transcriptomic approach was supposed to find (Fornito et al., 2019).

Pericytes are contractile and regulate capillary flow directly (Hall et al., 2014;
Attwell et al., 2010; Iadecola, 2017), so a relationship between mural-cell gene
expression and steady-state oxygen extraction is mechanistically unsurprising. The
surprise is where it stops.

It does not reach discordance. The mediation model fails at the mediator→outcome
path in 100% of pipelines, and an independent ex vivo measurement of vascular
density is unrelated to discordance in any form. The two failures are
methodologically independent — one transcriptomic, one anatomical — and they agree.

The more interesting reading is that **any between-region contribution of
baseline OEF to discordance is at most moderate** (§3.5), despite Epp et al.'s
observation that discordant voxels differ in baseline OEF. A group-level topographic association between OEF and
discordance is what their result implies; our path model asks whether parcel-wise
variation in OEF predicts parcel-wise variation in discordance, and it does not.
These are compatible: a between-condition mechanism need not leave a between-region
signature. `[TODO: this is the paper's central interpretive claim and needs to be
argued more carefully, ideally with a simulation showing the two can dissociate.]`

### 4.1 What would change our minds

- A vascular density measurement in human tissue, matched to these subjects.
- Cell-type deconvolution against a published single-cell reference
  (Seidlitz et al., 2020) rather than curated marker panels alone.
- Comparison against the time-averaged control energy map of Ceballos et al.
  (2025), which indexes a different notion of regional metabolic demand.
- Discordance measured with a method that does not derive OEF from T2\*, breaking
  the dependence identified in §3.7.
- Subject-level rather than parcel-level mediation, which the released data cannot
  support (per-subject masking is not reproducible from the public release).

---

## 5. Limitations

Stated here rather than discovered in review.

- **AHBA is 6 adult post-mortem donors** (5 usable), bulk microarray, predominantly
  left hemisphere. It is a modal brain, not a matched sample. No individual-level
  inference is licensed.
- **Spatial correlation is not mechanism.** The mediation model is suggestive at
  best, and it is null in any case.
- **ds004873 is ~40 subjects, one scanner, one site.** Generalisation is untested.
- **The volumetric→surface projection discards subcortex and cerebellum.**
- **mqBOLD carries its own assumptions** — vessel geometry, blood volume
  estimation, T2′ modelling — and these propagate into every OEF and CMRO₂ value
  used here. §3.7 quantifies one consequence.
- **The dropout confound is mitigated, not eliminated.** It is carried as a
  mandatory covariate throughout, and the whole-chain gate bounds it at |ρ| ≤ 0.32.
- **The primary finding does not survive family-wide FDR correction** (minimum
  adjusted *p* = 0.130), and does not survive a stricter hierarchy specification
  that we argue — but cannot demonstrate — is over-controlled.
- **The total inferential surface is large**: 55 gene-set × target combinations in
  Phase 4, 528 gene-set partials per hierarchy specification in Phase 5, 132
  mediation combinations in Phase 6, and 12 PLS components in Phase 4b. Each family
  is corrected internally; none is corrected against the others. We report a robust
  *association*, not a confirmed effect.
- **The mediation is bounded, not null.** Every path-b estimate lies below this
  design's detectability floor (§3.5), so a moderate mediator→outcome effect
  remains possible.
- **The cross-species control is registration-limited** in exactly the association
  cortex the hypothesis concerns, and rests on four animals and 83 parcels.
- **Phase 5 runs on 12 multiverse cells**, not 120, for compute reasons; Phases 4
  and 6 use all 120.
- **The pericyte/mural set is 4 genes of 5** — ANPEP is absent from the AHBA
  expression matrix. The frozen set is unchanged; one gene is not measurable.

---

## 6. Reproducibility

Every artifact in `results/` carries a manifest recording git SHA, config hash,
package versions, seed, wall-clock time, and input checksums.
`scripts/audit_provenance.py` gates the pipeline on six checks: one code state
across all artifacts, a clean working tree at write time, every output paired with
provenance, artifacts written hours rather than days apart, agreement between
values appearing in more than one file, and production by the run that owns the
directory.

The numbers in this draft were reproduced **bit-identically across two independent
full regenerations at different code states** (`37cdbf6` and `b4cb6d5`), including
the primary effect, the mediation path structure, and every gate.

`scripts/regenerate_all.sh` rebuilds everything from the expression multiverse in
~3.7 hours on a 16-thread host.

---

## 7. Data and code availability

All primary data are public and none were generated by this study.

| Resource | Identifier |
|---|---|
| Epp et al. mqBOLD derivatives | OpenNeuro `ds004873` (snapshot 2.0.x) |
| Allen Human Brain Atlas microarray | `abagen.fetch_microarray`; 5 of 6 donors (15496 unavailable upstream) |
| Reference and null maps | `neuromaps` (Markello et al., 2022) |
| Gene sets | MSigDB via `gseapy`; version pinned in `data/MANIFEST.yaml` |
| Macaque vascular density | see `[TODO: citation]`; Yerkes19 → fs_LR registration |

Analysis code, the frozen gene-set definitions, and the released parcel-level
annotation table are available at `[TODO: repository URL]`. Every result artifact
carries a manifest recording git SHA, config hash, package versions, seed, and
input checksums; `scripts/audit_provenance.py` gates the pipeline on their
consistency and `scripts/check_paper_numbers.py` verifies that the values in this
manuscript match the artifacts they are drawn from.

## 8. Ethics

This is a secondary analysis of two publicly released datasets. Ethical approval
and participant consent were obtained by the original investigators; no new human
or animal data were collected. Post-mortem tissue governance for the Allen Human
Brain Atlas is described in Hawrylycz et al. (2012).

## 9. Funding and competing interests

`[TODO]` — Funding sources and competing interests to be declared before
submission.

---

## Appendix A — Deviations from pre-registration

1. **Discordance frequency across four tasks** was pre-specified but is not
   analysable: only 2 of 4 conditions are published in MNI152 space. Replaced by
   the extraction/overshoot mode split, which uses the same two conditions.
2. **The extended hierarchy specification** (gradients 2–3) is a disclosed addition,
   not pre-registered. It is run on every regeneration and reported alongside the
   pre-registered specification precisely so it cannot be chosen post hoc. §3.4
   argues it is over-controlled; readers may disagree, which is why both are shown.
3. **Phase 5 gene-set step runs on 12 of 120 multiverse cells** for compute reasons.

## Appendix B — `[TODO]`

- Full citation for the macaque vascular atlas.
- Figures: currently F1–F8 are generated by `scripts/make_figures.py`; captions and
  selection for the manuscript are not yet written.
- The §4 dissociation argument needs a simulation.
- Author list, affiliations, funding, competing interests.
- Decide target venue; this reads as *Imaging Neuroscience* or *NeuroImage* rather
  than a high-impact venue, which is appropriate for a structured negative result.

---

## References

Resolved against Crossref by `scripts/verify_references.py`; each entry is a
DOI whose returned title is asserted to match, so a misremembered citation
fails loudly rather than being formatted in. Schaefer et al. is dated 2017 by
Crossref (advance online) and 2018 by its issue; we cite the issue year.

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
