# Internal peer review — `paper/draft.md`

Reviewed against `results/` at git `b4cb6d5` (provenance audit 6/6) using the
7-stage framework in `peer-review-methodology`. Written adversarially and on
purpose: the author of the draft is the wrong person to judge whether its central
negative claim is supported, so this review tries to break it.

**Recommendation: major revision.** The analysis is unusually well controlled and
the reproducibility infrastructure exceeds field norms. But the paper's central
claim — that the mediator→outcome link is *absent* — is not supported by the
design, and one supporting result is described as stronger than the code that
produced it says it is.

---

## Summary

The manuscript tests whether the capillary-density conjecture explains BOLD–CMRO₂
discordance, using AHBA transcriptomics across a 120-cell preprocessing multiverse
with spatial and competitive nulls throughout. It reports that vascular gene
expression predicts baseline OEF robustly, that mediation to discordance fails,
and that an independent macaque vascular map is unrelated to discordance.

**Strengths.**

1. **The null-model discipline is genuinely rigorous.** Every spatial claim
   carries an autocorrelation-preserving null; every gene-set claim additionally
   carries a size- and stability-matched competitive null. This is stricter than
   most published imaging-transcriptomics work.
2. **Positive controls are present, reported, and one of them fails openly.**
   §3.2 and §3.7 are the sections that make the negative result interpretable, and
   the observation that the *original authors'* published CMRO₂ map also misses
   the PET reference is a real contribution independent of the hypothesis.
3. **Reproducibility is demonstrated rather than asserted** — bit-identical
   results across two full regenerations at different code states, under a gating
   provenance audit.

**Weaknesses.** Detailed below: the central negative claim is underpowered rather
than demonstrated (Major 1); the adjudication between hierarchy specifications is
self-serving (Major 2); and the macaque control is presented as the paper's
strongest evidence while its own generating script documents a registration
weakness in exactly the cortex the hypothesis concerns (Major 3).

---

## Major comments

**1. The central negative claim is an absence of evidence, not evidence of
absence, and the paper has the numbers to say so precisely.**

§3.5 states "Baseline OEF does not predict discordance in these data." §4 builds
on it: "baseline OEF is not on the causal path to discordance."

But the manuscript already reports the reliabilities needed to check whether such
an effect *could* have been detected, and it could not. Using the paper's own
attenuation framework:

| outcome | path b | ceiling √(r₁r₂) | implied true \|ρ\| | detectability floor |
|---|---|---|---|---|
| extraction | −0.217 | 0.752 | 0.289 | 0.331 |
| overshoot | +0.057 | 0.763 | 0.075 | 0.313 |
| coupling angle | +0.141 | 0.834 | 0.169 | 0.301 |

**All three implied true effects sit below the floor.** The discordance maps have
split-half reliability near 0.58–0.71 against baseline OEF's 0.978; the pairing is
power-limited by the outcome, not by the mediator.

This does not make the result uninteresting — it makes it *bounded*. The paper can
legitimately exclude a **large** mediator→outcome effect (|ρ| ≳ 0.33) while being
unable to exclude a moderate one. That is a publishable and honest statement, and
it is stronger than the vague "does not predict."

*Required*: reframe §3.5 and §4 as an exclusion bound with the floor stated
explicitly; move the claim "not on the causal path" to something the design
supports.

**2. The adjudication between hierarchy specifications reads as motivated.**

§3.4 argues the extended specification is over-controlled because it destroys
metabolic-versus-metabolic relationships. The evidence offered is real and the
gene-PC1 comparison is a good control. But the competing explanation — that these
maps genuinely are gradient-2/3 structured, and that gradients 2–3 are legitimate
confounds — is dismissed in one sentence.

The paper cannot resolve this from the data, and should say so more evenly. As
written, the pre-registered specification is favoured and it is also the
specification under which the headline result survives. That coincidence needs to
be visible to the reader rather than argued past.

*Suggested*: state the two readings with equal weight, give the reader the numbers
for both (already present), and drop "we judge the extended specification
over-controlled" to a clearly labelled author opinion.

**3. The macaque control is described as the study's strongest evidence while its
own script documents a disqualifying weakness for that role.**

§3.6 calls it "the strongest statement the study makes, because it does not depend
on the transcriptomic machinery at all." But `scripts/x1_macaque_vascular.py`
states in its own header: *"The registration is weakest where the hypothesis
lives"*, that human cortical expansion leaves only ~83 of 100 left-hemisphere
parcels with values, and that the map derives from **four** macaques.

A cross-species surface registration that is least trustworthy in association
cortex cannot carry the paper's strongest claim about association cortex. None of
these caveats appear in the manuscript.

*Required*: carry all three caveats into §3.6 and §5, and downgrade the framing
from "strongest statement" to convergent-but-limited evidence.

**4. No sampling uncertainty is reported for any effect.**

Effects are reported as median with inter-quartile range across multiverse cells.
That is pipeline dispersion, not sampling uncertainty — two different things that
the notation invites readers to conflate. A ρ of −0.391 with IQR [−0.411, −0.359]
looks far more precise than an *n* = 100 parcel correlation is.

*Required*: bootstrap confidence intervals over parcels for the primary effect,
reported separately from the multiverse IQR.

**5. Multiple comparisons are controlled within families but never across the
paper.**

Phase 4 tests 55 gene-set × target combinations; Phase 5 adds 528 gene-set
partials across two specifications; Phase 6 adds 132; Phase 4b adds 12 PLS
components. Each is corrected internally. The manuscript never states the total
inferential surface, and the one effect it headlines fails even its own
within-family FDR at 0.130.

*Suggested*: state the total number of tests, and be explicit that the primary
finding is a robust *association* that does not clear confirmatory correction.

**6. Reporting-standard compliance is not addressed.**

This is a secondary analysis of observational human imaging plus post-mortem
microarray. STROBE applies to the former, MIAME-style reporting to the latter
(platform, normalization, accession). Neither is mentioned.

*Required*: add data availability with accessions (OpenNeuro `ds004873`, AHBA,
MSigDB version), ethics inheritance from the original studies, funding, and
competing interests. The draft currently has none of these.

**7. The two null models disagree in an undiscussed way.**

The headline effect passes the competitive null decisively (*p* = 0.0004) but
fails FDR. The positive control (endothelial → macaque vascular) does the reverse:
spatially significant in 100% of pipelines but competitive *p* = 0.09.

If the assay that *should* work fails the competitive null while the hypothesis
test passes it, the competitive null is measuring something other than "is this
gene set special." This asymmetry needs a paragraph — it bears directly on how
much weight *p* = 0.0004 can carry.

---

## Minor comments

1. **No figures exist.** Eight are generated by `scripts/make_figures.py` but none
   are selected, captioned, or referenced. At ~4,000 words the manuscript should
   carry roughly four display items.
2. **Sample sizes are inconsistently reported.** ρ values appear without *n*;
   parcel counts vary (100, 97, 83) across analyses without comment.
3. **Phase 5 uses 12 of 120 multiverse cells** while Phases 4 and 6 use all 120.
   Disclosed in §5, but the asymmetry deserves a sentence where the Phase 5
   results are presented, not only in limitations.
4. **Abbreviation load is high** (BOLD, CMRO₂, CBF, CBV, OEF, AHBA, mqBOLD, FDR,
   PET, SNR, IQR). Acceptable for a specialist venue; would need trimming for a
   general one.
5. **§3.7's OEF-weighting explanation is speculative** and correctly flagged
   `[TODO]`. Either test it — the components are on disk — or reduce it to one
   sentence.
6. **Schaefer et al. year** is 2017 by Crossref (advance online) and 2018 by
   issue. The draft notes this; ensure in-text use is consistent.

## Questions for the authors

1. What is the parcel-level power to detect path b at the observed reliabilities?
   (Major 1 suggests it is low; a formal curve would settle it.)
2. Were the discordance-mode maps' reliabilities known before the mediation was
   specified? If so, was the mediation adequately powered by design?
3. How was the macaque→human registration validated, and does its quality vary
   systematically with the principal gradient? If it does, the null in §3.6 is
   partly a registration artifact.
4. Does the primary effect survive dropping parcels with zero AHBA samples?
