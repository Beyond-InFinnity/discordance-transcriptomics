# Vascular gene expression predicts cortical oxygen extraction; its link to BOLD–CMRO₂ discordance is below current detection limits

**Working draft.** Every number is generated from `results/` at git `e25f745`,
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

We tested it using Allen Human Brain Atlas (AHBA) microarray across 120
preprocessing pipelines, with spatial-autocorrelation-preserving and
stability-matched competitive nulls throughout. We first asked what effect size
each test could detect at all. For every gene-set × outcome pairing we computed a
**detectability floor** — the smallest true correlation resolvable given the
measured reliability of both maps. **No test in the pre-registered design could
resolve a true effect below |ρ| = 0.30**, four were untestable at any effect
size, and 38 of 44 were limited by the gene side rather than the imaging.

That limit is substantially **self-imposed**. The standard construct — averaging a
gene set's members into one spatial score — cancels signal whenever the genes'
spatial patterns resemble each other less than their shared measurement noise
does, which holds for every large pathway set we examined. Scoring the same genes
in smaller pieces recovers reliability: `HALLMARK_HYPOXIA` moves from a panel
reliability of 0.126 to 0.384 and a floor of 0.88 to 0.51, and
`GOBP_BLOOD_VESSEL_MORPHOGENESIS` from untestable to a floor of 0.51. Choosing the
best-measuring construction per set — on reliability alone, never on any outcome —
improves 24 of 44 pairings and leaves none untestable.

**Pericyte and mural-cell gene expression predicts baseline oxygen extraction
fraction (OEF)** (ρ = −0.391, sign-consistent across 100% of 240 tests, spatially
significant in 86%, competitive *p* = 0.0004) — and survives when the same genes
are tested individually and aggregated statistically instead of averaged into one
map (competitive *p* = 0.006), so it does not depend on the construction §3.2.1
shows to be the weaker measurement. It is in the pre-registered direction,
strengthening when the unimodal–transmodal hierarchy is partialled out. It does
not survive family-wide correction (minimum adjusted *p* = 0.130). Its floor is
0.330 and it is a small, spatially coherent set, so it is among the few tests the
design could have resolved.

Mediation to discordance is unsupported across all 15,840 path models, but every
mediator→outcome estimate falls below its detectability floor: a large effect
(|ρ| ≳ 0.33) is excluded, a moderate one is not. The conjecture's first link is
supported; its second is **untested rather than refuted**. We quantify the
reliability a real test would require, and show that a substantial part of the
shortfall is a choice of estimator rather than a limit of the data.

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
| Macaque vascular atlas (Autio et al., 2025) | Ferumoxytol-weighted laminar MRI, cortical vascular density, 4 animals | Independent control |

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

**How often the spatial null fires, and why.** We measured, per target, the
fraction of the transcriptome clearing a spin test — all 15,563 genes across the
multiverse, ≈1.8 million gene × pipeline tests each. The rates span fifteen-fold:

| target | split-half reliability | real genes, % at spin *p* < 0.05 | rotated genes, % |
|---|---|---|---|
| baseline OEF | 0.978 | **0.82** | 4.1 |
| coupling angle | 0.711 | 2.06 | 3.9 |
| overshoot-mode discordance | 0.595 | 7.85 | 4.1 |
| extraction-mode discordance | 0.579 | **12.09** | 4.1 |

It is tempting to read that spread as calibration failure — the test being
conservative against baseline OEF and firing too readily against extraction-mode
discordance. **That reading is wrong, and the second column is why.** It requires
assuming no gene is truly associated with the target, which is not a null this
field believes: the raw rate confounds the test's error rate with how much of the
transcriptome a map genuinely engages.

So we measured the error rate directly. Rotating a gene's parcel vector preserves
its spatial autocorrelation and its value distribution exactly — a rotation is a
reindexing — while destroying its anatomical alignment. A set of independently
rotated genes therefore has zero true association *by construction*, and anything
clearing is a false positive and nothing else. Gene rotations are drawn from a
seed disjoint from the one generating each target's null; drawn from the same
set, the observed statistic would be a near-copy of a null draw and the check
would pass by construction rather than on its merits.

**The spin test is well behaved.** Against rotated genes it fires at 3.9–4.1%
across all four targets, against a nominal 5% — mildly conservative, by about a
fifth, and essentially flat across maps whose real rates differ fifteen-fold.
The spread is association, not error.

The mean *p* over all genes says the same thing in a form that does not depend on
a threshold. A neutral value is 0.5; baseline OEF gives 0.632 and extraction-mode
discordance 0.412. Genes sit systematically *further* from baseline OEF than its
own rotations do, and systematically closer to extraction-mode discordance.

Two consequences follow, and they are not the ones this comparison was expected
to produce. A single gene's spin *p* against extraction-mode discordance is still
not comparable with one against baseline OEF — but because the *background rate
of genuine association* differs by an order of magnitude, not because the test
misbehaves. And the competitive null still handles it either way, since matched
random sets are scored through the same background; that is why no set-level
claim here rests on a spin *p* alone.

The third consequence is specific to baseline OEF and cuts in our favour. Real
genes clear against it at 0.82% while their own rotated counterparts clear at
4.1%, and their median |ρ| is *lower* than the rotated genes' (0.074 against
0.112). Real genes do worse than autocorrelation-matched noise built from
themselves. Gene expression is not isotropic — most genes carry a component of
the dominant unimodal–transmodal axis — and baseline OEF lies close to orthogonal
to it, so many of its rotations align with the transcriptome better than the map
itself does. Every negative result against baseline OEF is therefore stronger
than a null rather than weaker, and any set that does track it stands against a
0.82% background rather than a 5% one (§3.4.1).

Companion methodological work applying the identical measurement to 85 published
`neuromaps` annotations puts the rotated-gene rate at a median of 4.90% (range
2.81–6.63%) while real-gene rates span 2.02–47.23%, so this is a general property
of spin-tested transcriptomics rather than of our maps.

The rotated-gene column uses 12 multiverse cells (≈187,000 tests per target,
standard error ≈0.05% on a 4% rate) against the real column's 120. Real rates
recomputed on the same 12 cells are 0.68, 1.93, 8.20 and 12.67, so the
comparison does not turn on the cell count.

### 2.6 Detectability

For each map we estimate split-half reliability with Spearman–Brown correction and
partition observed variance into true and error components. For a gene set with
panel reliability *r*₁ and a brain map with reliability *r*₂, the maximum
observable correlation is the **attenuation ceiling** √(*r*₁*r*₂) — the correlation
the two would show if the underlying relationship were perfect. Dividing the
spin-test threshold by that ceiling gives the **detectability floor**: the smallest
*true* effect the pairing can resolve.

The floor depends only on the two reliabilities and the parcellation. It does not
use the observed effect, and nothing we report is derived by comparing the two.
An earlier version of this analysis did make that comparison, calling a test
*resolvable* when |ρ|/ceiling exceeded threshold/ceiling. The ceiling cancels, so
that criterion reduced to |ρ| ≥ threshold — the significance test restated. It is
withdrawn; see Appendix A.

**Construction.** A gene set can be scored as one averaged map, in chunks, or gene
by gene. These are not equivalent measurements. Averaging *k* genes reduces noise
variance by a factor [1 + (*k*−1)ρ̄ₑ]/*k* and signal variance by
[1 + (*k*−1)ρ̄ₛ]/*k*, where ρ̄ₛ and ρ̄ₑ are the mean pairwise correlations among the
genes' true patterns and among their measurement errors. Averaging therefore
improves reliability only when ρ̄ₛ > ρ̄ₑ. In AHBA every gene is measured from the
same tissue samples, so error is shared across genes while signal is shared only
where genes genuinely co-localise — a condition pathway membership does not
guarantee. We report reliability under each construction and select per set on
reliability alone, never on any outcome.

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

`GOBP_BLOOD_VESSEL_MORPHOGENESIS` has negative panel reliability *as a set score*:
averaged into one map, its donor-to-donor pattern does not replicate at all. Its
individual genes do (§3.2.1), so this is a property of the construct, not of the
genes.

Combining these with map reliabilities gives a detectability floor for each of the
44 pairings. The coupling angle is included because §7.3 names it the *primary*
outcome — continuous rather than binarised — and it is the most reliable of the
three coupling-derived maps:

| outcome | n | median floor | worst |
|---|---|---|---|
| baseline OEF | 11 | 0.387 | untestable |
| coupling angle | 11 | 0.473 | untestable |
| overshoot-mode discordance | 11 | 0.493 | untestable |
| extraction-mode discordance | 11 | 0.522 | untestable |

**No test in the design resolves a true effect below |ρ| = 0.30.** Distributed
across plausibility bands: 22 pairings could detect only moderate-to-large effects
(floor 0.30–0.50), 13 only large ones (0.50–0.70), 5 sit above 0.70 where no
spatial correlation between independent modalities is plausibly expected, and 4
are untestable at any effect size.

Two consequences follow, and they structure everything below.

First, **the outcome the hypothesis is about is the one the design interrogates
worst.** Extraction-mode discordance has the highest median floor, 0.522, against
a field in which ρ ≈ 0.3 is a strong result.

Second, **the limit is overwhelmingly on the gene side: 38 of 44 pairings are
bound by gene-map reliability rather than by the imaging.** Given brain maps of
0.98 (baseline OEF) against gene-set scores of 0.13–0.67, more subjects would not
have helped. This is the opposite of where effort in this literature usually goes.

### 3.2.1 How much of the limit was self-imposed

The gene-side bottleneck is partly a property of AHBA and partly a property of how
gene sets are conventionally scored. Averaging *k* genes into one map improves
reliability only when the genes' true spatial patterns are more alike than their
measurement errors (§2.6), and because AHBA measures every gene from the same
tissue samples, the error is shared by construction while the signal is not.

Scoring the same genes in smaller pieces separates the two:

| gene set | *k* | as one score | in chunks of 5 | per gene |
|---|---|---|---|---|
| glucose/lactate transport | 5 | **0.670** | 0.670 | 0.389 |
| endothelial | 6 | **0.591** | 0.591 | 0.465 |
| pericyte/mural | 5 | **0.557** | — | 0.450 |
| HALLMARK_ANGIOGENESIS | 36 | **0.582** | 0.440 | 0.399 |
| HALLMARK_GLYCOLYSIS | 200 | 0.316 | 0.341 | **0.356** |
| HALLMARK_OXPHOS | 200 | 0.229 | 0.304 | **0.309** |
| HALLMARK_HYPOXIA | 200 | 0.126 | 0.357 | **0.384** |
| GOBP_BLOOD_VESSEL_MORPH. | 53 | **−0.011** | 0.344 | **0.379** |
| interneuron subclass | 4 | 0.187 | — | **0.678** |

The pattern follows the variance argument rather than set size alone. Small,
spatially coherent sets — cell-type markers that genuinely co-localise — are best
averaged. Large database sets are not: pathway co-membership does not imply
co-localisation, and averaging cancels their signal faster than their noise.
`interneuron_subclass` is the instructive exception at *k* = 4: PVALB, SST, VIP
and LAMP5 mark four distinct interneuron populations with genuinely different
cortical distributions, and averaging them destroys what each measures.

Selecting the best-measuring construction per set — on reliability alone, with no
reference to any outcome — improves 24 of 44 pairings, lowers the median floor
from 0.448 to 0.420, and **leaves none untestable**. `HALLMARK_HYPOXIA` moves from
a floor of 0.88 to 0.51 and `GOBP_BLOOD_VESSEL_MORPHOGENESIS` from infinite to
0.51.

The pre-registration froze *which genes* (§8.1 of the protocol) and *which null
models* (§7.4). It did not specify how to combine genes into a score; the
unweighted average was an implementation choice inherited from convention. We
report the pre-registered construction as primary throughout and the
reliability-selected alternative alongside it.

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

| Gene set | Target | ρ (median) | IQR | sign | spin-sig | competitive *p* | floor |
|---|---|---|---|---|---|---|---|
| pericyte/mural | baseline OEF | **−0.391** | [−0.411, −0.359] | 100% | **86%** | **0.0004** | 0.330 |
| HALLMARK_ANGIOGENESIS | baseline OEF | −0.355 | [−0.413, −0.289] | 100% | 30% | **0.0002** | 0.323 |
| astrocyte | overshoot | +0.256 | [+0.224, +0.286] | 100% | 15% | **0.007** | 0.535 |
| glucose/lactate transport | extraction | +0.229 | [+0.180, +0.277] | 100% | 27% | 0.060 | 0.405 |
| HALLMARK_OXPHOS | baseline OEF | −0.203 | [−0.235, −0.172] | 100% | 0% | 0.107 | 0.516 |

**Of the three effects clearing the spin threshold, one matches its
pre-registered direction, one contradicts it, and one had no direction
specified.** Pericyte/mural → baseline
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
Its detectability floor is 0.330, among the lowest in the design: pericyte/mural
is a small, spatially coherent set, so the averaged construction measures it well.

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

### 3.4.1 The same sets, aggregated per gene

§3.2.1 shows the averaged score is the less reliable measurement for every large
set. This re-runs H1 with the aggregation moved: each gene tested individually
against each target, and the set-level question — are this set's genes
collectively significant more often than size- and stability-matched random sets?
— asked of the resulting statistics rather than of an averaged map. Same frozen
sets, same targets, same two nulls. 15,563 genes × 120 pipelines × 4 targets,
7.3 million spin tests.

The construction was not pre-registered either way (Appendix A), so this is
reported *beside* §3.4, not in place of it. Five of 44 set × target tests clear
the competitive null, against four of 55 in the averaged arm:

| gene set | target | *k* | mean % spin-sig | *z* | *p* |
|---|---|---|---|---|---|
| **pericyte/mural** | baseline OEF | 4 | 18.54 | +7.54 | **0.006** |
| glycolytic enzymes | extraction | 6 | 31.32 | +1.98 | 0.044 |
| HALLMARK_OXPHOS | coupling angle | 199 | 0.33 | −2.94 | **0.004** |
| HALLMARK_OXPHOS | baseline OEF | 199 | 0.01 | −2.13 | 0.032 |
| HALLMARK_OXPHOS | overshoot | 199 | 4.98 | −2.09 | 0.038 |

**Pericyte/mural → baseline OEF survives the change of construction.** It was
found by averaging (§3.4, ρ = −0.391, competitive *p* = 0.0004) and is found again
without averaging. Its four genes reach spin significance 18.5% of the time
against a target whose genome-wide rate is 0.82% (§2.5) — roughly 23-fold
enrichment, and against a background that is low because the transcriptome is
anti-aligned with baseline OEF rather than because the test is strict: rotated
genes clear the same threshold at 4.1%. The two arms share only the genes,
the target and the nulls; they
combine evidence in ways that fail differently, so agreement between them is
stronger evidence than either alone.

The *z* of +7.54 should be ignored. With four genes the null distribution of a
matched-set mean is lumpy and heavy-tailed, not Gaussian: a true *z* of 7.54 would
imply *p* ≈ 5 × 10⁻¹⁴, whereas 60 of 10,000 draws were in fact that extreme. The
permutation *p* is the honest quantity, which is why the competitive null is
empirical rather than parametric.

**The three oxidative-phosphorylation entries are depletions, not associations.**
Negative *z*: those 199 genes reach significance *less* often than matched random
sets — against baseline OEF, 0.01% versus 0.82% genome-wide. The statistic is
deliberately sign-free, so it says these genes are unusually *unrelated* to the
maps in either direction, whereas H1 predicts a negative *association*. Depletion
therefore neither confirms nor refutes the hypothesis whatever its cause.

Its cause is partly measurable. The competitive null matches on set size and
differential stability but **not on spatial autocorrelation**, and a spin test's
behaviour depends on the smoothness of both maps — modestly, on the evidence of
§2.5's rotated-gene rates, but not negligibly — so a set of unusually smooth
genes can appear depleted for reasons unconnected to oxygen.
Fulcher et al. (2021) document the mirror image, categories with high spatial
autocorrelation acquiring inflated false-positive rates. We tested it directly
(x3), applying the same test to pericyte/mural, because both arms of this study
use a null blind to autocorrelation and their agreement cannot resolve a shared
blind spot.

The premise holds: oxidative-phosphorylation genes are smoother than the genome
(Moran's I 0.316 against 0.268, *p* = 5 × 10⁻⁷), and against baseline OEF their
spin nulls are correspondingly wider (0.178 against 0.163, *p* = 4 × 10⁻¹²).
Smoothness predicts per-gene significance against the smooth targets — ρ = −0.13
for baseline OEF and the coupling angle — and not against the noisy ones
(ρ = +0.03 for extraction), which is the mechanism rather than a coincidence: a
gene's own smoothness only matters when the target's rotations are themselves
smooth enough to correlate with it.

Adding autocorrelation to the matching then attenuates every depletion, by an
amount that tracks how strongly smoothness predicted significance for that target:

| test | *z* (stability) | *p* | *z* (+ autocorrelation) | *p* |
|---|---|---|---|---|
| OXPHOS → coupling angle | −2.98 | 0.005 | −2.60 | 0.011 |
| OXPHOS → baseline OEF | −2.09 | 0.035 | −1.82 | **0.054** |
| OXPHOS → overshoot | −2.06 | 0.039 | −2.00 | 0.044 |
| **pericyte/mural → baseline OEF** | +7.53 | 0.007 | **+5.62** | **0.013** |

One depletion is fully accounted for — baseline OEF crosses to *p* = 0.054, and it
is the test where the mechanism was strongest. Two survive. Autocorrelation is
therefore a demonstrated contributor and not a sufficient explanation, and we
still draw no biological inference from a sign-free depletion.

The control matters more than the result it was built to check. **Pericyte/mural
survives the matched null** (*p* = 0.013), clearing a null the published one does
not impose. Its *z* falls further than the depletions' do, from +7.53 to +5.62,
and not because those genes are smooth: their Moran's I is 0.273 against 0.268
for the genome (*p* = 0.94), indistinguishable. The attenuation is mechanical.
Stratifying on a second variable splits the draw pool into up to fifty joint
cells, and for a four-gene set that makes the null lumpier and wider — the same
small-set behaviour that made its *z* untrustworthy above, which is why the
permutation *p*, moving only 0.007 to 0.013, is the quantity to read.

**Glycolytic enzymes → extraction-mode discordance (*p* = 0.044) is what chance
looks like.** It sits on H1's glycolytic arm, against the outcome the hypothesis
concerns, and the averaged score found nothing there (ρ = −0.024) — so the
construction change turned a null into a nominal hit, which is the mechanism of
§3.2.1 working as described. It is also entirely expected under the null: across
44 tests at α = 0.05 one expects ≈2.2, and this is one. It does not survive
family-wide correction and is not evidence for H1.

`macaque_vascular_CONTROL` is excluded from this arm. Its map has 83 of 100
parcels, so rotations pull unobserved parcels into the analysis window and only 2
of 10,000 surrogate draws are complete. The vectorised sweep has no paired-
observation path for ragged surrogates; §3.3 scores that control through
`corr_with_null`, which does.

**What this arm changes.** The mediator arm of H1 now rests on two analyses that
combine evidence differently and agree, and on a null that matches the nuisance
both of them were blind to. The outcome arm gains one nominal hit
consistent with chance, on a target where 12% of all genes clear the spin
threshold under the null. The conclusion of §3.4 is unaltered; what changes is
that it no longer depends on a construction §3.2.1 shows to be the weaker
measurement.

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

| outcome | path b | ceiling | \|path b\| ÷ ceiling | floor |
|---|---|---|---|---|
| extraction | −0.217 | 0.752 | 0.289 | 0.331 |
| overshoot | +0.057 | 0.763 | 0.075 | 0.313 |
| coupling angle | +0.141 | 0.834 | 0.169 | 0.301 |

Every path-b estimate, divided by its ceiling, falls below the floor — that is,
none reaches the size the test could have resolved. The discordance maps are the
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
Its flow–metabolism coupling is weaker, however (CBF↔CMRO₂ +0.16 vs +0.44). One
reading is that our CMRO₂ is more OEF-weighted and so inherits more of OEF's
dropout sensitivity, which would fit the network profile above. We state that as
a possibility rather than a result: distinguishing it from the alternatives —
differences in the CBF estimate, or in how the two reconstructions propagate
haematocrit — needs a variance decomposition across the mqBOLD chain, and the
components' reliabilities (§3.1) do not support one. The failure is established;
its attribution within the chain is not.

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

It stops at the measurement, not at the biology. Extraction-mode discordance has
the highest median floor of the three outcomes (0.522), and every mediation
path-b estimate lies below its floor.

The bottleneck is mostly the **gene** side, not the imaging: 38 of 44 pairings are
bound by gene-map reliability, against brain maps measured at 0.96–0.99. More
subjects would not have rescued them. And a substantial part of that gene-side
limit is the estimator rather than the atlas — averaging a large set into one map
cancels its signal (§3.2.1), and scoring the same genes in pieces removes every
untestable pairing in the design.

This is, we think, the more useful contribution. A bare null tells the field
nothing about whether to try again. A null accompanied by the reliability at which
the test becomes possible is a specification for the next experiment, and the
attenuation identity inverts to give it directly: a side needs reliability
*t*² / (ρ² · *r*_other) for a true effect ρ to become resolvable.

| true \|ρ\| | gene reliability needed, vs baseline OEF | vs extraction-mode discordance |
|---|---|---|
| 0.20 | 1.53 — unreachable | 2.59 — unreachable |
| 0.25 | 0.98 | 1.66 — unreachable |
| **0.30** | **0.68** | 1.15 — unreachable |
| 0.35 | 0.50 | 0.85 |
| 0.40 | 0.38 | 0.65 |

Read against the brain maps as measured. Two things follow, and they point in
opposite directions.

**Against baseline OEF the gap is small and specific.** A gene panel needs
reliability **0.68** to resolve a true ρ of 0.30. The best panel measured here
reaches **0.670** — short by a hundredth. This is not a call for an order of
magnitude more data; it is a call for a modestly better-measured gene score, and
the estimator work in §3.2.1 moves in exactly that direction without quite
arriving (it closes the gap for none of the eleven sets, though it narrows it for
most).

**Against the discordance maps it is not a gap at all.** Resolving ρ = 0.30
against extraction-mode discordance would need a gene panel of reliability 1.15,
which does not exist and cannot: no improvement to the molecular side alone can
rescue a test whose imaging side sits at 0.579. That side has to move first, and
§3.1 says how — more subjects, or more task conditions, since discordance is the
one target here whose reliability is limited by sampling rather than by
measurement. A study that improves only the gene side and tests against
discordance is unfalsifiable by construction, whatever it reports.

We also think Epp et al.'s observation need not imply a between-region signature at
all, and this is the interpretive claim the paper leans on hardest, so we
demonstrate it rather than assert it. Their result is that discordant *voxels*
differ in baseline OEF — a comparison within a brain. Ours asks whether *parcels*
with higher mean OEF show more discordance — a comparison between regions. The two
can come apart, and what separates them is what the discordance threshold is
measured against.

Simulating voxels within parcels, a voxel turns discordant when its OEF exceeds a
threshold. If that threshold is one physiological value everywhere (**absolute**),
a parcel with higher mean OEF has more voxels above it and both studies see the
effect. If it tracks the local neighbourhood — discordant means *high for where
you sit* (**relative**) — the within-brain effect is undiminished while the
between-region correlation vanishes. Sweeping between the two:

| threshold | within-brain *d* | between-region ρ | visible to this design |
|---|---|---|---|
| fully absolute | 2.55 | +0.995 | yes |
| 50% relative | 2.25 | +0.987 | yes |
| 90% relative | 1.60 | +0.803 | yes |
| 95% relative | 1.51 | +0.557 | **no** |
| fully relative | 1.42 | −0.007 | no |

**Every regime reproduces their observation.** The within-brain effect stays
large throughout (*d* = 2.55 to 1.42); only the between-region correlation moves,
and it moves from +0.995 to zero. A null between regions is therefore evidence
about which regime holds, not evidence against the within-brain result.

It is also not vacuous. Every regime below 95% relative would have produced a
between-region correlation this design could resolve, so **our null bounds the
mechanism at ≳95% local, if it operates at all.** That is a constraint rather
than an absence — and it is the optimistic version: the simulation carries no
measurement noise beyond the sampling, and any additional noise attenuates the
between-region correlation, moves the crossing point down, and weakens what the
null constrains.

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
- **Most tests here were underpowered by construction** (§3.2): 40 of 44 gene-set ×
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
- **Sampling uncertainty is roughly seven times the multiverse IQR**, so every
  interval elsewhere in this paper understates precision by about that factor.
  Bootstrapping over parcels and widening by √(n / n_eff) = 1.25 for spatial
  dependence, the primary effect is ρ = −0.391, 95% CI **[−0.62, −0.16]** — it
  excludes zero, but spans from a large effect to a small one. Three of five
  reported effects exclude zero at 95%. Pipeline choice was never the binding
  source of imprecision here, and reporting only a multiverse IQR implied a
  tightness the data do not support.

---

## 6. Data and code availability

All primary data are public; none were generated by this study.

| Resource | Identifier |
|---|---|
| Epp et al. mqBOLD derivatives | OpenNeuro `ds004873` (snapshot 2.0.x) |
| Allen Human Brain Atlas microarray | `abagen.fetch_microarray`; 5 of 6 donors |
| Reference and null maps | `neuromaps` (Markello et al., 2022) |
| Gene sets | MSigDB via `gseapy`; version pinned in `data/MANIFEST.yaml` |
| Macaque vascular density | Autio et al. (2025), BALSA study `1vjnV`; Yerkes19 → fs_LR registration via Xu et al. (2020) |

Analysis code, frozen gene-set definitions, and the released parcel-level
annotation table are available at `[TODO: repository URL]`. Every result artifact
carries a manifest recording git SHA, config hash, package versions, seed, and
input checksums. `scripts/audit_provenance.py` gates the pipeline on their
consistency; `scripts/check_paper_numbers.py` verifies that the values in this
manuscript match the artifacts they are drawn from; `scripts/verify_references.py`
resolves every citation against Crossref.

**Reproducibility, stated precisely.** Effect estimates reproduce across
independent full regenerations at different code states: every *z*, ρ and
reliability in this manuscript is stable to the digits reported. Permutation
*p*-values are Monte Carlo estimates and are not bit-identical. The one case we
measured directly moved from 0.0515 to 0.0537 between regenerations — 0.99
standard errors for a 10,000-draw permutation estimate near *p* = 0.05 — while
its *z* held at −1.82. The cause is benign and worth naming: the competitive
null stratifies on a gene pool derived upstream, so a handful of genes entering
or leaving that pool shifts the quantile bin edges and therefore every draw. We
report *p* to three decimals because that is what the design supports; the third
decimal should not be read as stable.

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
5. **A `resolvable` criterion is withdrawn.** An earlier version of §3.2 called a
   test resolvable when |ρ| ÷ ceiling exceeded threshold ÷ ceiling. The ceiling
   cancels, so the criterion was identical to |ρ| ≥ threshold — the significance
   test relabelled — and its headline, that the resolvable tests were exactly
   those passing both nulls, was a tautology that could not have come out
   otherwise. We now report detectability floors alone, which use no outcome
   information. The affected claims are removed rather than restated, and
   `scripts/check_paper_numbers.py` fails if the withdrawn phrasing reappears.
6. **Construction selection (§3.2.1) is disclosed, not pre-registered.** The
   protocol froze which genes (§8.1) and which null models (§7.4) but did not
   specify how genes are combined into a score; the unweighted average was an
   inherited convention. Reliability under alternative constructions is computed
   from expression data alone and the selection never consults an outcome, so it
   cannot be a forking path — but it was chosen after the primary results were
   seen, and the pre-registered construction remains primary throughout.
7. **The autocorrelation-matched null (x3) is a disclosed addition**, prompted by
   a specific alternative explanation for the oxidative-phosphorylation depletion
   rather than by its result. It was applied to pericyte/mural at the same time
   and for the same reason, so it could not be a test run only where a negative
   was wanted.
8. **One frozen gene set never ran.** `mitochondrial_density_proxy` is declared in
   `config/genesets.yaml` by HGNC family prefixes rather than an explicit gene
   list, and the loader silently keeps only sets with an explicit list. It
   therefore appears in no result and in no error. It was the only small curated
   set addressing H1's oxidative-phosphorylation arm, which was consequently
   tested only through the 200-gene HALLMARK set — the construction §3.2.1 shows
   to be least reliable. Phase 0d now reports declared-but-absent sets.
9. **A miscalibration claim is withdrawn.** An earlier version of §2.5 read the
   fifteen-fold spread in genome-wide clearance rates as the spin test being
   conservative against baseline OEF and firing too readily against
   extraction-mode discordance, and read the latter's 12.09% as a false-positive
   rate. That inference assumes no gene
   is truly associated with the target, which is not a null this field believes,
   and it is refuted by measurement: against independently rotated genes — same
   autocorrelation, no anatomical alignment, zero true association by
   construction — the test fires at 3.9–4.1% against a nominal 5%, flat across
   all four targets (x4). The spread is association, not error. The correction
   strengthens rather than weakens the study's negatives, since a background rate
   of 0.82% against baseline OEF now reflects genuine anti-alignment of the
   transcriptome with that map. `scripts/check_paper_numbers.py` fails if the
   withdrawn phrasing reappears.

## Figure captions

**Figure 1. What this design can detect.** (**A**) Detectability floor — the
smallest *true* |ρ| a test could resolve — against the effect actually observed,
for all 44 gene-set × outcome tests. The two axes are independent: the floor is a
property of the design and does not use the observation. Dashed line, spin-test
threshold; shading, plausibility bands; labelled points cleared the threshold.
Colour denotes outcome. (**B**) Gene-set panel reliability across AHBA donors, the
limiting term in 38 of 44 tests. Blue ≥ 0.55; vermillion ≤ 0.
`GOBP_BLOOD_VESSEL_MORPHOGENESIS` has negative reliability *as an averaged score*;
its individual genes replicate (§3.2.1).

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
- Autio JA et al. (2025). Mapping vascular network architecture in primate brain using ferumoxytol-weighted laminar MRI. *eLife*, 13. https://doi.org/10.7554/elife.99940
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
- Xu T et al. (2020). Cross-species functional alignment reveals evolutionary hierarchy within the connectome. *NeuroImage*, 223, 117346. https://doi.org/10.1016/j.neuroimage.2020.117346
- Vaishnavi SN et al. (2010). Regional aerobic glycolysis in the human brain. *Proceedings of the National Academy of Sciences*, 107, 17757-17762. https://doi.org/10.1073/pnas.1010459107
