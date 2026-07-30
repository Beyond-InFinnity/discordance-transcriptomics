# Data survey: can we synthesise a human cortical microvascular map?

Status: **survey only**. Nothing below has been downloaded, verified or analysed.
Availability claims come from publications and repository listings and each needs
checking before anything is built on it.

---

## The goal, stated precisely

Produce a per-region estimate of **microvascular density in human cortex**, of
the kind that would let people test claims about vasculature and brain function
that are currently untestable. No such map exists. Human imaging measures total
blood volume, dominated by large vessels: this project measured a
sensory-to-association ratio of **0.97** in human cerebral blood volume against
**2–3** in macaque tissue, which is the gap the whole idea addresses.

The approach: learn a mapping from **gene expression → measured vessel density**
in species where both exist, then apply it to human gene expression, which we
have.

---

## Architecture: parallel, not chained

The obvious design is a chain — mouse to macaque to human — and it is the wrong
one. Every hop applies an imperfect cross-species transform, and errors compound
multiplicatively. Worse, a chain has no internal check: if the output is wrong
there is nothing to compare it against.

Train one model per species, independently, on that species' own paired data.
Apply each to human expression separately. Then:

- **Agreement between independently-trained models is evidence.** Their errors
  are independent, so convergence is informative in a way a chain's output never
  is.
- **Disagreement is also informative** — it localises which part of the
  gene-to-vessel relationship is conserved and which is primate-specific.
- Ensembling is optional and secondary. The comparison is the point.

```
   mouse expression  ──►  mouse model  ──┐
  (Allen ISH, CCF)      (VesSAP density) │
                                         ├──►  applied to human AHBA  ──►  compare
 macaque expression ──►  macaque model ──┘
  (97 regions, D99)     (Autio CBV)
```

---

## A. Paired training data (vasculature + transcriptome, same species)

### A1. Mouse — the best-registered pair

| | source | notes |
|---|---|---|
| vasculature | **VesSAP**, Todorov et al. 2020 *Nat Methods* | whole brain, capillary level, light-sheet, **registered to the Allen CCF**. Code on GitHub and Code Ocean. |
| vasculature (alt) | **Kirst et al. 2020** *Cell* | whole-brain vasculature reconstruction, independent method |
| transcriptome | **Allen Mouse Brain Atlas ISH** | ~20,000 genes, voxel resolution, **also in the Allen CCF** |

**Both halves live in the same coordinate frame.** That removes the registration
problem entirely for this species, and it is a much stronger position than the
mouse resource dismissed earlier in this project (~27 mostly-subcortical regions
behind a ~200 TB request). That earlier judgement was about the wrong dataset.

**The serious caveat, and it may be disqualifying.** Mouse cortex is
lissencephalic and has essentially no association cortex in the primate sense.
The gradient this project cares about — sensory versus association vascular
supply — may not exist in mouse to be learned from. So mouse is probably not
viable as the primary training species for *this* question. Its real role is a
different and still valuable one: **testing whether the gene-to-vessel
relationship is conserved at all.** If mouse-trained weights transfer sensibly to
primate, that is a strong statement about conservation. If they do not, that is
worth knowing before trusting any cross-species map.

### A2. Macaque — the right training species, harder registration

| | source | notes |
|---|---|---|
| vasculature | **Autio et al. 2025**, ferumoxytol-weighted laminar MRI | already held in this repo, already transferred to the human surface |
| transcriptome | **Bo et al. 2023** *Nat Commun* s41467-023-37246-w | 9 adult *M. fascicularis*, **100 cortical areas / 757 samples**, D99 atlas, MRI-space. Published matrix: **97 cortical regions x 23,613 genes**. SRA `PRJNA905082`. |

This is the pair that matters, because macaque *has* the sensory-to-association
distinction and has the vascular gradient measured.

**Open question, and the first thing to check:** is the 97 x 23,613 matrix
available as a processed artifact (supplementary data / CNGB), or only as raw
reads in SRA? Processed, this is tractable. Raw, it is 819 RNA-seq samples to
align and quantify before any science begins — an order of magnitude more work.

### A3. Marmoset — possible third point

Brain/MINDS publishes a marmoset gene expression atlas. Whether comparable
vascular measurements exist is unchecked. A third independent species would
strengthen the parallel design considerably, so this is worth ten minutes.

---

## B. Human — target and validation

| | source | role |
|---|---|---|
| **AHBA** | already held | the input the model is applied to |
| **Human vascular single-cell atlas** | Nature 2024, 606,380 cells; also Yang et al. 2022 *Nature* (`abi7377`), 181,388 cells across 7 regions | **principled marker genes.** Replaces the six hand-picked endothelial markers currently used with data-derived vascular cell-type signatures. Worth doing regardless of whether the rest of this proceeds. |
| **Stereology / histology** | e.g. Cortical numerical vessel density 1311 ± 326 mm⁻³, length density 255 ± 119 mm⁻²; white matter far lower (222 ± 147 mm⁻³) | **validation, not training.** A handful of regions, not a cortex-wide map — which is exactly what a held-out test set should be. |
| **Human CBV / CBF** | already held | weak validation only. Known to miss the microvasculature, which is the premise of this whole exercise. |
| **VENAT** | already held | venous partial-volume confound, already a covariate here |

---

## C. Validation plan, in order of strength

1. **V1 as a hard gate.** Primary visual cortex is the most densely vascularised
   region of primate cortex, established independently of anything here. If a
   synthetic human map does not rank V1 near the top, the model is wrong and
   gets discarded rather than reported.
2. **Human histology.** Sparse regional measurements as a held-out set. Small n,
   but it is *human ground truth*, which nothing else on this list is.
3. **Spatially-blocked cross-validation within the training species.** Not
   random splits — cortical regions are spatially autocorrelated, so random
   held-out sets leak and report inflated accuracy. Hold out contiguous
   territory.
4. **Cross-species agreement.** Mouse-trained and macaque-trained predictions
   compared on human cortex.
5. **Weak positive correlation with human CBV.** Expected but not diagnostic.

---

## D. Known difficulties, honestly

**Sample size is the binding constraint.** 97 macaque regions against ~23,000
genes is severely underdetermined, and spatial autocorrelation means the
*effective* n is well below 97. This forces heavy regularisation, dimension
reduction, or restriction to gene sets rather than individual genes — and makes
honest cross-validation harder than usual.

**Orthology.** Macaque-to-human is mostly 1:1 for protein-coding genes.
Mouse-to-human is considerably messier. Needs a real orthology table, not symbol
matching.

**Transform chains.** D99 is volumetric; reaching the Autio surface map means
D99 → macaque surface → sampled per region, and R4 forbids hand-rolling any step.

**The cross-species registration error is already quantified here** and it is not
uniform: roughly 6.7 mm in sensory cortex against 18.2 mm in association cortex.
Accuracy is worst exactly where the scientific question lives.

**For this project's own hypothesis, this adds sharpness but not information.**
A synthetic map is a weighted combination of human gene expression, and the
relationship between human gene expression and discordance has already been
tested exhaustively. What changes is that the weighting would be supervised by
measured vasculature instead of chosen by hand — so a negative afterwards is a
considerably stronger statement. The map's main value is as a standalone
artifact, not as a rescue of H1.

---

## E. Prior art

A search for existing attempts to predict cortical vascular density from
transcriptome, cross-species or otherwise, returned nothing doing this directly.
Related work connects regional expression to cortical thickness, connectivity and
morphometry, but not to vasculature. That gap looks real, and it is the reason
this is worth attempting.

---

## Sources

- Todorov et al. 2020, *Machine learning analysis of whole mouse brain vasculature*, Nat Methods — https://www.nature.com/articles/s41592-020-0792-1
- Kirst et al., whole-brain vasculature reconstruction — https://www.biorxiv.org/content/10.1101/2020.10.19.344903v1.full
- Bo et al. 2023, macaque brain-wide transcriptome + morphology, Nat Commun — https://www.nature.com/articles/s41467-023-37246-w (PMC10023667)
- Autio et al., ferumoxytol-weighted laminar MRI in primate brain — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11118324/
- Yang et al. 2022, single-cell atlas of normal and malformed human brain vasculature, Science — https://www.science.org/doi/10.1126/science.abi7377
- Single-cell atlas of the human brain vasculature across development, adulthood and disease, Nature 2024 — https://www.nature.com/articles/s41586-024-07493-y
- Human microvessel numerical and length densities — https://sciencedirect.com/science/article/pii/S0891061817302119
- Regional variation in brain capillary density — https://pubmed.ncbi.nlm.nih.gov/11489257/
- Transcriptomic and open chromatin atlas of rhesus macaque brain regions, Nat Commun — https://www.nature.com/articles/s41467-020-14368-z
