# Phase 0 results — both gates PASS

Run 2026-07-26. Schaefer-200 7-network, left hemisphere (100 parcels), fsaverage
10k, seed 42. Every number below has a `.manifest.json` beside it recording
package versions, config hash, and input checksums (R10).

Targets run: **baseline OEF** and the **calc-vs-control coupling ratio *n***.
The discordance-frequency map is deliberately not run — see §4.

---

## Phase 0a — reliability ⛔ GATE: **PASS**

1,000 random split-halves, Spearman correlation across parcels,
Spearman-Brown corrected. Gate: ≥0.5 pass, 0.3–0.5 caveat, <0.3 STOP.

| target | median r (raw) | **SB-corrected** | 95% across splits | n | verdict |
|---|---:|---:|---|---:|---|
| baseline OEF | 0.956 | **0.978** | [0.965, 0.986] | 40 | **PASS** |
| coupling ratio *n* | 0.468 | **0.638** | [0.486, 0.746] | 30 | **PASS** |

Both clear the gate. Two honest caveats:

- The coupling ratio's 95% interval **dips to 0.486**, just below the 0.5
  threshold. The median passes comfortably but the map is meaningfully noisier
  than baseline OEF, which is expected — it is a contrast of four noisy
  quantitative maps rather than one.
- **ICC(2,1) is 0.174 (OEF) and 0.199 (*n*).** The *group* spatial map is highly
  reproducible; *individual subjects* vary a great deal. This is consistent with
  §14 — no individual-level inference is licensed — but it should be stated
  plainly rather than hidden behind the split-half number.

Nobody else has published these numbers. Per §9 they belong in the paper
regardless of outcome.

---

## Phase 0b — the dropout confound ⛔ GATE: **PASS**

This was the strongest attack on the project. It does not land.

Proxy: **`snr_coverage`** — the per-parcel fraction of cortical vertices
surviving the authors' own SNR criterion
(`task-all_space-MNI152_res-2_SNR_YEO_group_mask.nii.gz`). It ranges 0.18–0.92
across parcels (median 0.65; 14 parcels below 0.5), so it has real dynamic range.

Gate: |ρ| ≥ 0.5 → STOP. Inference by Alexander-Bloch spin test, 10,000
rotations. Criterion evaluated on **masked** data — what downstream analysis
actually consumes.

| target | mask | ρ | spin p | naive p | verdict |
|---|---|---:|---:|---:|---|
| coupling ratio *n* | masked | −0.042 | 0.735 | 0.677 | **PASS** |
| coupling ratio *n* | unmasked | −0.002 | 0.986 | 0.985 | pass (diagnostic) |
| baseline OEF | masked | −0.185 | 0.189 | 0.065 | **PASS** |
| baseline OEF | unmasked | −0.259 | 0.063 | **0.009** | pass (diagnostic) |

The primary outcome is essentially **uncorrelated** with signal dropout
(ρ = −0.04). Baseline OEF shows a weak negative trend that does not approach
the severity threshold.

Two things worth noting:

1. **R1 earns its keep in the last row.** Unmasked baseline OEF has a naive
   p of 0.009 — nominally significant — against a spin p of 0.063. Without a
   spatial-autocorrelation-preserving null we would have reported a dropout
   association that the null model does not support. This is exactly the
   pathology §2 exists to prevent, visible in our own data.
2. **Masking moves the estimate toward null** (−0.259 → −0.185 for OEF), which
   is the expected direction: masking removes the low-SNR voxels that drive the
   association. Evaluating the gate on unmasked data would have been
   pessimistic for a reason masking already fixes.

Per the gate, `snr_coverage` is now a **mandatory covariate** in every
downstream model (`config/base.yaml: covariates.mandatory`).

---

## A rejected proxy — and why it matters

An earlier run used group-mean **T2\*** as the dropout proxy and returned
**ρ = −0.749, spin p = 0.0002** against baseline OEF — a decisive gate failure.

It is not a confound. It is arithmetic. mqBOLD defines

```
R2' = 1/T2* − 1/T2          OEF = R2' / (x · CBV)
```

so OEF is a deterministic function of T2\*. Verified at the parcel level in one
subject:

| relationship | ρ |
|---|---:|
| OEF vs R2′/CBV — *reconstructing the defining formula* | **+0.901** |
| R2′ vs T2\* — *R2′ is defined as 1/T2\* − 1/T2* | **−0.815** |
| OEF vs T2\* | −0.555 |

Correlating OEF with T2\* measures the formula, not the artifact. CLAUDE.md
§13.6 — "any result that looks *too* clean is more likely a bug or a confound
than a discovery" — is what caught this.

`scripts/p0_dropout.py` now **refuses** `--proxy t2star` against baseline OEF,
CBV or CBF, with `--allow-circular` as an explicit diagnostic override.

T2\* remains valid against the coupling ratio, because the baseline T2\*
dependence largely cancels in a calc-vs-control percent-change contrast — and
it passes there (ρ = −0.146, spin p = 0.283).

---

## What was NOT run, and why

The **discordance-frequency map** is not wired and was not run. Determining
whether `control` is a co-equal second condition or the control condition *of
the calculation task* is a Phase 1 deliverable. If it is the latter — which the
naming strongly suggests — a "frequency across conditions" map is not degraded
from 0–4 to 0–2, it is **undefined**. Building it now would mean inventing a
variable. `DERIVATIVE_PATTERNS["discordance_freq"]` is `None` and a test asserts
it stays that way.

---

## Coverage and caveats for the paper

- **~55–60% of cortical vertices survive** masking (SNR mask ∩ physiological
  range). Parcel means therefore rest on roughly half the ribbon. 1–2 parcels
  per subject have no usable vertices.
- **Coupling ratio runs at n=30**, limited by `calc` CMRO₂ in MNI152 (31
  subjects). The dataset ships the authors' own
  `from-T1w_to-MNI152NLin6Asym_..._xfm.h5` warps for 40 subjects; applying them
  to the T1w-space `calc` CMRO₂ maps would raise this to **n=41**. Applying an
  existing warp is not hand-rolling a transform and does not violate R4. Not
  yet done.
- OEF was range-limited to (0, 1) before averaging. The unmasked per-subject
  maps reach p95 ≈ 1.23, which is physiologically impossible.
