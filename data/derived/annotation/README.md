# discordance_annotation v0.9.2 — data dictionary

Parcel-level annotation of BOLD/CMRO2 discordance and baseline oxygen
metabolism in human **left** cortex.

Derived from ds004873 (Epp SM, Castrillón G, Yuan B, Andrews-Hanna J,
Preibisch C, Riedl V, 2025, *Nature Neuroscience*,
doi:10.1038/s41593-025-02132-9), OpenNeuro snapshot 2.0.7, CC0.

334 rows across 3 parcellations.

## Columns

| column | unit | stability | reliability | description |
|---|---|---|---|---|
| `parcellation` | — | — | — | Parcellation this row belongs to. |
| `parcel_index` | — | — | — | 1-based parcel label within the hemisphere. |
| `parcel_name` | — | — | — | Atlas label name. |
| `hemisphere` | — | — | — | Always 'L': only 2 of 6 AHBA donors have right-hemisphere tissue (CLAUDE.md R3). |
| `baseline_oef` | fraction | **stable** | 0.98 | Baseline oxygen extraction fraction, control condition. Authors' published group median (n=40, GM/R2'/CBV masked). |
| `baseline_cbf` | mL/100g/min | **stable** | — | Baseline cerebral blood flow, control condition, same source. Includes the authors' 1/0.75 scanner upscaling. |
| `baseline_cmro2` | umol/100g/min | **stable** | — | Baseline cerebral metabolic rate of oxygen, control condition, same source. |
| `coupling_n_angle` | radians | **stable** | 0.71 | Coupling ratio as the angle atan2(%dCBF, %dCMRO2) in the calc-vs-control contrast, median across subjects. Bounded and continuous through the origin, unlike the raw ratio n. |
| `discordance_risk` | fraction | **low_reliability** | 0.49 | Fraction of subjects in whom BOLD and CMRO2 move in opposite directions, i.e. coupling ratio n < 1. Uses the first-order approximation sign(dBOLD) = sign(dCBF - dCMRO2). LOW RELIABILITY (split-half 0.49, below this project's 0.5 floor): it sums two topographically distinct modes, which cancels signal and makes the total less reliable than either part. Prefer discordance_risk_extraction (0.58) or discordance_risk_overshoot (0.60). |
| `discordance_risk_extraction` | fraction | **stable** | 0.58 | Fraction of subjects in the EXTRACTION mode: CMRO2 rises while BOLD falls, i.e. oxygen demand goes up and flow fails to keep pace, so the tissue raises its extraction fraction. This is the mode the Epp et al. mechanism concerns and the one a capillary-density hypothesis predicts. Use THIS column, not discordance_risk, for vascular hypotheses. |
| `discordance_risk_overshoot` | fraction | **stable** | 0.60 | Fraction of subjects in the OVERSHOOT mode: CMRO2 falls while BOLD rises, i.e. flow is delivered in excess of falling demand. Same arithmetic as extraction, different physiology. |
| `discordance_risk_n` | — | **stable** | — | Number of subjects contributing to the discordance columns. |
| `dropout_snr_coverage` | fraction | **stable** | — | Fraction of the parcel's cortical vertices surviving the authors' SNR criterion. Low values indicate signal dropout; carry as a covariate. |
| `venous_partial_volume` | fraction | **stable** | — | Mean venous partial volume (VENAT atlas). The brain-vs-vein confound. |
| `map_reliability_coupling` | correlation | **stable** | — | Split-half Spearman-Brown reliability of the WHOLE coupling map at this parcellation (a map-level constant, not per-parcel). |
| `n_subjects_coupling` | — | — | — | Subjects with all four maps needed for the coupling contrast. |
| `baseline_source` | — | — | — | Provenance of the baseline columns. |
| `coupling_source` | — | — | — | Provenance of the coupling and discordance columns. |
| `ahba_n_samples` | — | **provisional** | — | AHBA microarray samples falling in this parcel (left hemisphere). Null when not computed. Parcels with 0 cannot support transcriptomic analysis. |

### What the stability labels mean

- **stable** — Values are not expected to change. Sourced directly from the authors' published group maps, or computed from a fixed atlas.
- **provisional** — Values may change. Computed from inputs known to differ from what the source analysis used, or from an incomplete donor set.
- **low_reliability** — The values are what they are — they are not expected to change — but the map does not measure a stable individual difference well enough to rank parcels confidently. Split-half reliability falls below the 0.5 threshold this project set for itself (CLAUDE.md §9, Phase 0a). Prefer a higher-reliability column measuring the same thing.

### The reliability column

Split-half reliability, Spearman-Brown corrected, from Phase 0a: split
the 40 subjects in half 1,000 times, rebuild the parcel map in each
half, and correlate. It answers *how much of the parcel-to-parcel
variation in this column is signal rather than sampling noise*, and it
sets a ceiling on how strongly the column can correlate with anything
else. A correlation against a column with reliability 0.5 is attenuated
by roughly sqrt(0.5) = 0.71. Columns below 0.5 are
labelled `low_reliability`.

**`discordance_risk` is one of them (0.49).** It is the sum of the two
mode columns, and it is *less* reliable than either of them, because the
two modes are topographically distinct and adding them cancels signal.
Use `discordance_risk_extraction` (0.58) or
`discordance_risk_overshoot` (0.60) — and for any vascular or
capillary-density hypothesis, the extraction column is the one the
mechanism actually concerns.

`baseline_*` are the columns to build on. The coupling and discordance
columns are usable but expected to be revised; see the deviation note
below before treating their absolute values as final.

## Read this before using `discordance_risk`

It is the fraction of subjects, in the calc-vs-control contrast, for whom
BOLD and CMRO₂ move in opposite directions. BOLD tracks deoxyhaemoglobin,
which to first order falls when CBF outpaces oxygen consumption, so
`sign(dBOLD) = sign(dCBF - dCMRO2)`. Both sign cases reduce to the same
condition, **coupling ratio n < 1**, so the n = 1 isocline is the natural
boundary rather than a tuned threshold.

Two things to hold in mind:

1. **It rests on that first-order BOLD approximation.** It is not a
   measured BOLD sign. Per-subject BOLD percent-change maps are published
   for only one participant, so a directly measured version is not
   possible from this release.
2. **It is not directly comparable to the ~40% of voxels reported by Epp
   et al.** This counts *subjects* per parcel, not voxels, on a single
   two-condition contrast. The values here run higher.

Sanity check that it behaves: the default mode network carries the
highest mean discordance of the seven Yeo networks at Schaefer-200,
which is the concentration the source paper reports.

### Use the mode columns, not the total

`discordance_risk` sums two mechanistically different things, and the
split on this data is close to even (53% / 47%):

- **`discordance_risk_extraction`** — CMRO2 rises while BOLD falls.
  Demand goes up, flow fails to keep pace, and the tissue compensates by
  extracting a larger fraction of the oxygen already present. This is
  what the source paper's mechanism is about (discordant regions regulate
  supply through OEF rather than CBF) and the mode a capillary-density
  hypothesis predicts. Within it, flow is still rising but insufficiently
  in ~74% of cases and genuinely falling in ~26%.
- **`discordance_risk_overshoot`** — CMRO2 falls while BOLD rises. Flow
  delivered in excess of falling demand. Same arithmetic, different
  physiology, no obvious vascular-capacity interpretation.

For any vascular or metabolic hypothesis, test against
`discordance_risk_extraction`. Using the total halves your effective
signal by averaging it with an unrelated phenomenon.

## A deviation forced by the release

The authors' own analysis uses the CBV-corrected CMRO₂ variant
(`desc-CBV_cmro2`) for the calc condition and `desc-orig_cmro2` for the
others. **Only `desc-orig_cmro2` is published in MNI152**; the CBV-corrected
variant exists in native T2 space only. The coupling and discordance
columns here therefore use `desc-orig` for both conditions, which is not
what their pipeline used for calc.

This likely biases the coupling ratio: the median `coupling_n_angle` of
0.26 rad implies n ≈ 0.27, lower than the n ~ 2-4 typical of task
activation. Treat the *relative* ordering across parcels as more
trustworthy than the absolute values, and treat both as provisional
pending either the CBV-corrected maps in MNI152 or a native-space
reconstruction.

## Limitations

- **Left hemisphere only.** Only 2 of 6 AHBA donors have right-hemisphere
  tissue, so the annotation is built to pair with transcriptomics.
- **Group-level only.** No individual-level inference is licensed. The
  between-subject consistency of these maps is low even where the group
  map is highly reproducible.
- **Reliability varies by parcellation.** `map_reliability_coupling` is
  the split-half Spearman-Brown reliability of the whole coupling map.
  At DK-68 its confidence interval extends below 0.3; treat coupling
  values at that resolution with caution.
- **Baselines and coupling have different provenance.** Baselines are the
  authors' own masked group maps. The coupling columns are reconstructed
  from the per-subject maps, which are published *unmasked* — the
  per-subject masking in their pipeline is not reproducible from the
  release (only one subject's masked files are included). See
  `results/p1_reproduction_notes.md`.
- **Volumetric→surface projection discards subcortex and cerebellum.**
- **`ahba_n_samples` was computed from 5 of the 6 AHBA donors** (9861, 10021, 12876, 14380, 15697).
  Donor 15496 is unavailable upstream as of 2026-07-26 — the Allen
  file ID returns HTTP 404 and no replacement is indexed by their API.
  It is a left-hemisphere donor, so coverage counts here understate a
  complete-AHBA run. See `data/MANIFEST.yaml`.
- **n=40 for baselines, n=30 for coupling**, one scanner, one site.
  Generalisation is untested.
- **`discordance_frequency` is absent by design** — see the header of
  `scripts/build_annotation.py`.

## Citation

Cite ds004873 (Epp et al. 2025) alongside this table. The underlying
data is theirs; this artifact is a derived annotation.
