# Resume here

State as of the pause on 2026-07-27. Everything below is safe to interrupt.

---

## Nothing is lost by stopping

**All code is committed and pushed** to `github.com/Beyond-InFinnity/discordance-transcriptomics`.

**The expression multiverse resumes automatically.** Each of the 120 cells
writes a parquet named by a hash of its parameters, and the runner checks for
that file before doing any work. Re-running skips everything already finished
and continues from the first gap. Interrupting mid-cell loses only that cell,
which costs about a minute.

The parquet files live in `data/derived/expression/multiverse/` and are
gitignored, so they stay on this machine — roughly 35 MB per cell, about 4.2 GB
for the full grid.

---

## To resume

```bash
cd ~/Documents/projects/discordance-transcriptomics

# 1. Continue the multiverse. Safe to run repeatedly.
#    MUST be sequential: each call peaks near 4.6 GB and this machine has
#    about 5 GB free. Six parallel workers were killed by the OOM killer.
.venv/bin/python scripts/p3_multiverse.py --n-jobs 1

# 2. How far along is it?
ls data/derived/expression/multiverse/*.parquet | wc -l    # target: 120

# 3. Once complete, run the analysis.
.venv/bin/python scripts/p4_genesets.py --n-draws 10000
```

If the machine slept rather than shut down, the background job may still be
alive — check with `pgrep -af p3_multiverse` before starting another.

---

## Where the science stands

Both hypothesis tests have now failed, and they failed in a way that is
believable rather than merely inconclusive.

**The capillary explanation.** Discordance was supposed to arise because
association cortex has sparse capillaries. Two independent tests say otherwise.
The mode of discordance that a sparse-supply account predicts peaks in
somatomotor cortex, not the default mode network. And macaque microvascular
density — the only real capillary-scale measurement available, transferred onto
human cortex — does not predict it (0.079, and the wrong sign).

**The gene hypothesis, preliminary.** None of the frozen gene sets predict
discordance. Both directional predictions fail: glycolytic genes are flat where
the hypothesis says positive, endothelial genes carry the wrong sign where it
says negative.

**Why these are informative rather than underpowered.** The same pipeline, the
same parcels, the same statistics detect endothelial genes tracking the macaque
vascular map at 0.403. The machinery finds vascular biology when vascular
biology is there. It does not find any in discordance.

**What still stands.** The measurements are reliable, scanner dropout does not
explain them, and the dominant confound in this field — the sensory-to-
association axis — does not apply. Discordance is real and well characterised.
Nothing yet explains it.

---

## What the full run will add

The preview used the curated gene sets, one parcellation, one processing
pipeline. The full run adds:

- the five MSigDB sets, now fetched and pinned to disk
- the competitive null, which asks whether a set beats random sets matched on
  size *and* differential stability rather than whether it beats rotated maps
- 120 processing pipelines instead of one, so each effect is reported as a
  distribution with the share of pipelines agreeing on sign

It would take a lot to move a correlation of +0.031 somewhere interesting, so
the expectation is confirmation rather than reversal. The value is that the
negative becomes defensible instead of provisional.

---

## Open items needing a human

**AHBA donor 15496** is unavailable upstream — the Allen Institute file returns
an error and their catalogue no longer lists it. Everything runs on five of six
donors. Only asking them will fix it.

**The resting-condition maps** from the source dataset would settle whether the
somatomotor result is real or an artifact of comparing two active task
conditions. Worth asking for in the same message as the two findings they would
want to know: their masked per-subject derivatives are released for only one
subject, and the CBV-corrected oxygen-metabolism variant is missing from the
standard-space release.

**`discordance_risk` should be relabelled.** Its split-half reliability is
0.491, below this project's own 0.5 threshold, and below both of the two mode
columns it sums. It is currently marked stable in the released table. The two
mode columns are the ones to use.
