# Where we are

A plain-language running summary. Updated as work proceeds. If you read one
document in this repository, read this one.

Last updated: 2026-07-27.

---

## The question

Functional magnetic resonance imaging (fMRI) does not measure brain activity.
It measures how much oxygen-depleted blood sits in a piece of tissue. Normally
that works as a proxy: neurons work harder, the body over-delivers blood, the
depleted blood is flushed out, the scanner signal rises.

Epp and colleagues (2025) showed that a large fraction of cortex does not behave
that way. In roughly 40% of responsive tissue, oxygen consumption moves in the
*opposite* direction to the scanner signal. They called this **discordance**.

Nobody knows why. The lead author's thesis speculated that association cortex —
the abstract-thinking parts — has fewer capillaries than sensory cortex, and
that this sparse plumbing produces the reversed responses. **That speculation
has never been tested against molecular data. Testing it is why this project
exists.**

---

## What discordance actually is

Two things can go wrong, and they are different biology. Both were being
reported as one number until recently.

**Extraction mode (53% of cases).** Oxygen demand rises. Blood flow fails to
keep pace. The tissue compensates by stripping a larger share of oxygen out of
the blood it already has. More depleted blood accumulates, so the scanner signal
*falls* — at the moment the region is working hardest. **This is the mode a
sparse-capillary explanation predicts.**

**Overshoot mode (47%).** Oxygen demand falls. Blood flow does not fall as fast.
Surplus oxygen is left over, so the signal *rises*. Same arithmetic, different
story, no obvious link to how many vessels there are.

The two are now separate columns in the released dataset. Use the extraction one
for any question about blood vessels.

---

## What we have done

| Stage | Status | Outcome |
|---|---|---|
| Check the measurement is stable | done | passed |
| Check scanner dropout isn't causing it | done | passed |
| Reproduce the original paper exactly | done | passed, exactly |
| Build the region-by-region dataset | done | released, version 0.9.2 |
| Rule out the main statistical trap | done | trap does not apply |
| Gene expression analysis | **not started** | — |
| Mediation model | **not started** | — |

---

## What we have found

**The measurements are trustworthy.** Split the 40 subjects into two random
halves a thousand times, build the map separately in each half, and the halves
agree. Baseline oxygen extraction agrees at 0.98, blood volume at 0.98, the
task-response map at 0.71.

**We reproduced the original paper exactly.** Rebuilding their published group
map from their raw data returns their numbers to six decimal places. Our
pipeline is doing what theirs did.

**The biggest statistical trap does not apply.** Brains vary along one dominant
axis, from sensory to abstract regions, and almost everything correlates with
almost everything else along it. Any finding that rides that axis is suspect.
Our maps do not — the correlation is 0.04. Whatever this is, it is not that.

**But nothing correlates with anything.** Not glucose metabolism, oxygen
metabolism, blood flow, evolutionary expansion, or the dominant pattern of gene
expression. We checked whether that is a power problem: it is not. Every map
could have detected a true correlation of 0.34 or larger, and the best of them
anything above 0.25. Nothing came close. These are real negatives.

**Our oxygen-metabolism map disagrees with the PET gold standard.** They agree
at 0.09. Blood flow does better at 0.33, and blood volume best at 0.46. Since
oxygen metabolism is calculated *from* the other quantities, errors compound —
which is a plausible explanation, and also a warning about how much weight that
map can carry.

---

## What this means for the original hypothesis

One line points away from the hypothesis. A second line that appeared to is
retracted below.

**First, the wrong part of the brain.** If sparse capillaries cause discordance,
the extraction mode should concentrate in association cortex. It does not. It
peaks in **somatomotor cortex** (0.46), which is among the best-perfused tissue
in the brain, while the default mode network sits fourth (0.34). Somatomotor
cortex shows oxygen demand rising 3.8% against a blood flow response of only
0.6% — real demand, almost no delivery response.

**Second — RETRACTED.** We tested blood volume against discordance and found
nothing (+0.215 raw, +0.060 once large veins were removed), and I reported that
as evidence against the capillary explanation. **That test was invalid**, for a
reason found afterwards.

A 2025 study mapped vascular volume in macaque cortex using a contrast agent
(ferumoxytol) at layer-by-layer resolution, and found primary sensory cortex has
**2–3× higher** vascular volume than association cortex. That is exactly the
gradient the capillary hypothesis assumes — so in primates, the hypothesis's
premise is *supported*.

Our human blood-volume maps do not reproduce that gradient at all. Sensory
divided by association gives **0.97×** in our data and **1.04×** in an
independent human PET dataset, against the macaque's 2–3×. Both human
measurements are essentially flat.

The likely reason is that both human measurements capture *total* blood volume,
dominated by larger vessels, while the macaque method resolves the
microvasculature. So our blood volume is not a valid stand-in for capillary
density, and our test of the hypothesis was measuring the wrong thing. The
capillary explanation is **untested**, not refuted.

**Important distinction.** This does *not* contradict the published paper. Their
mechanistic claim is that discordant regions regulate oxygen supply through
extraction rather than through flow, and our data supports that. What looks
wrong is the *thesis speculation* about why — the capillary-density explanation.

**A possible reframe** is about regulation rather than plumbing size — though
with the blood-volume evidence withdrawn, this is now speculation rather than
something the data supports. Somatomotor cortex having real demand with almost
no flow response is a control failure rather than a capacity failure, and the
default mode network failing in the opposite direction fits the same picture:
vasculature that tracks demand poorly in both directions. Testing it needs a
real capillary measurement, which we do not yet have.

---

## What we cannot say

- We never measured capillary density. Blood volume is a proxy, and it includes
  arteries and veins.
- Our comparison is one task against another task, not task against rest. The
  original analysis used four conditions including rest, and those maps are not
  published. Which regions look discordant may depend on the contrast.
- Everything rests on 40 people, one scanner, one site.
- The extraction/overshoot split depends on the *sign* of the oxygen-metabolism
  change — our least trustworthy measurement.
- Nothing here says anything about any individual person. Individual subjects
  vary enormously even where the group average is stable.

---

## What comes next

**The gene expression analysis is still worth running.** Typical effects in this
field land around 0.3 to 0.5, and our detection floor is 0.25 to 0.34. A strong
effect would show; a weak one would not. Run it on the frozen gene list as
originally committed, and report whatever comes out.

**The vessel-tone idea goes in a separate exploratory section.** The gene list
was fixed in advance specifically so that nobody could look at the results and
then pick genes that fit. Swapping in vessel-tone genes now would break that.
Reporting them separately, clearly labelled, with no confirmatory claim, is the
honest route — and if the density genes miss while the tone genes hit, that is a
real result you will have earned.

**Get a real capillary-density measurement.** This is now the priority. The
macaque ferumoxytol maps are public (BALSA study 1vjnV, in a standard macaque
surface space), and published macaque-to-human surface correspondences exist —
two evolutionary-expansion maps in our toolkit were built using them. Bringing
that map into human space would give the first valid test of the capillary
explanation, rather than the invalid proxy we used.

**Ask the original authors for the resting-condition maps.** That single
addition would tell you whether the somatomotor finding is real or an artifact
of comparing two active tasks. You are in a strong position to ask, having
reproduced their analysis exactly.

**One donor is missing.** Of six post-mortem brains in the gene atlas, one has
vanished from the Allen Institute's servers — the file returns an error and
their catalogue no longer lists it. Analyses run on five. Only a human asking
them will fix it.

---

## Terms that keep coming up

| Term | What it means |
|---|---|
| **Oxygen extraction fraction (OEF)** | The share of oxygen the tissue pulls out of passing blood. Normally 30–40%. |
| **Cerebral metabolic rate of oxygen (CMRO₂)** | How much oxygen the tissue actually burns. |
| **Cerebral blood flow (CBF) / blood volume (CBV)** | How fast blood moves through / how much sits there. |
| **Coupling ratio (n)** | Flow change divided by oxygen-use change. Below 1 means the scanner signal opposes real oxygen use — that is discordance. |
| **Discordance** | Scanner signal and oxygen consumption moving in opposite directions. |
| **Default mode network (DMN)** | The regions most active during rest and internal thought. |
| **Spin test** | A way of asking whether two brain maps really correlate, given that any two smooth maps correlate somewhat by chance. |
| **Parcel** | One of the ~100 regions the cortex is divided into for analysis. |
| **Split-half reliability** | Build the map twice from separate halves of the sample; how well do they agree? |
| **Allen Human Brain Atlas (AHBA)** | Gene expression measured across six donated post-mortem brains. The molecular side of this project. |
