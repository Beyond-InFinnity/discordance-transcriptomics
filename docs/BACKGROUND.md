# Background

Plain-language explanation of what this project is about, for a reader who is
not a neuroimager.

**It deliberately contains no results.** Findings live in the README, and the
full argument lives in `paper/draft.md`, which is machine-checked against
`results/` by `scripts/check_paper_numbers.py`. This file replaced an earlier
"running summary" that mixed explanation with numbers, went stale within days,
and told readers to trust it over everything else. Explanation does not rot.
Numbers do, so they are kept where a gate can watch them.

---

## The question

Functional MRI does not measure brain activity. It measures how much
oxygen-depleted blood sits in a piece of tissue.

Usually that works as a proxy. Neurons work harder, the body over-delivers blood,
depleted blood gets flushed out, and the scanner signal rises. The signal goes up
when the region is busy, so we read it as activity.

Epp et al. (2025) showed that a large fraction of cortex does not behave that
way. In roughly 40% of responsive tissue, oxygen consumption moves *opposite* to
the scanner signal. The region is working harder and the signal is going down, or
the reverse.

Their proposed explanation: association cortex, the parts of the brain doing
abstract and self-directed work, may have sparser capillaries. Thin supply cannot
surge on demand, so the usual over-delivery fails and the signal inverts.

That is a claim about blood vessels. Nobody had checked it against a measurement
of blood vessels. Gene expression is the closest available proxy, because the
genes that build and maintain capillaries can be measured across the cortex in
post-mortem tissue. This project makes that comparison.

---

## What discordance actually is

Two different things go wrong, and they are different biology. Early work
reported them as one number.

**Extraction mode.** Oxygen demand rises. Blood flow fails to keep pace. The
tissue compensates by stripping a larger share of oxygen from the blood it
already has. More depleted blood accumulates, so the scanner signal *falls*, at
the moment the region is working hardest.

This is the mode a sparse-capillary explanation predicts.

**Overshoot mode.** Oxygen demand falls. Blood flow does not fall as fast.
Surplus oxygen is left over, so the signal *rises*. Same arithmetic, different
story, no obvious connection to how many vessels there are.

They are separate columns in the released dataset. Use the extraction one for any
question about blood vessels. Treating them as one number averages two opposite
phenomena.

---

## Why this is hard to test

Three failure modes are well documented in this field, and a fourth turned out to
dominate.

**Smooth maps correlate by chance.** Brain maps vary gradually across the cortex.
Two unrelated ones still reach a correlation around 0.4. Any test has to compare
against maps that share that smoothness but nothing else.

**Processing choices move the answer.** Turning gene expression into a brain map
takes a dozen defensible decisions. Published work shows those choices can shift
a correlation enough to reverse its sign. One pipeline is one opinion.

**Everything varies along the same axis.** Sensory cortex and association cortex
differ on nearly every measurable property. Any map that varies along that axis
will correlate with any gene set that varies along it, for reasons that have
nothing to do with the hypothesis.

**Measurement noise sets a hard ceiling.** This is the one that dominated here.
If a brain map and a gene map are each measured imperfectly, the correlation
between them is dragged toward zero no matter what the truth is. Past a certain
point, a real effect of ordinary size *cannot* be detected, however carefully the
analysis is done.

Most of this project's tests were on the wrong side of that line. Knowing which
ones, and by how much, is the part most likely to be useful to someone else.

---

## What can and cannot be concluded

**Cannot: anything about an individual.** The gene data is six post-mortem adult
brains, five usable. That is a template, not a sample. No statement about any
living person follows from it.

**Cannot: mechanism.** Two maps correlating across the cortex does not establish
that one causes the other. The mediation model tested here is suggestive at best,
and it came out null.

**Cannot: generalisation.** One imaging dataset, forty subjects, one scanner, one
site.

**Can: bounds.** Where a test could resolve an effect and did not, that excludes
effects above a stated size. Where it could not have resolved one either way,
that is reported as uninformative rather than as evidence of absence. The
difference between those two is most of what this project is for.

---

## Terms

| Term | Meaning |
|---|---|
| Oxygen extraction fraction (OEF) | Share of oxygen the tissue pulls from passing blood. Normally 30–40%. |
| Cerebral metabolic rate of oxygen (CMRO₂) | How much oxygen the tissue actually burns. |
| Cerebral blood flow / volume (CBF, CBV) | How fast blood moves through, and how much sits there. |
| Coupling ratio (*n*) | Flow change divided by oxygen-use change. Below 1, the scanner signal opposes real oxygen use. That is discordance. |
| Discordance | Scanner signal and oxygen consumption moving in opposite directions. |
| Default mode network | Regions most active during rest and internal thought. |
| Spin test | Asking whether two brain maps really correlate, given that any two smooth maps correlate somewhat by chance. |
| Parcel | One of the ~100 cortical regions the analysis divides the hemisphere into. |
| Split-half reliability | Build the map twice from separate halves of the sample, then see how well the two agree. |
| Detectability floor | The smallest true correlation a test could resolve, given how noisily both sides were measured. |
| Allen Human Brain Atlas (AHBA) | Gene expression measured across six donated post-mortem brains. The molecular side of this project. |

---

## Where to go next

| You want | Read |
|---|---|
| What was found | `README.md` |
| The full argument and every number | `paper/draft.md` |
| The data itself | `data/derived/annotation/` |
| How the project is specified and constrained | `CLAUDE.md` |
