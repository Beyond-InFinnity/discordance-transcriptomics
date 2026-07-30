# Data request: macaque regional expression matrix

## Who to write to

**Primary: Guang-Zhong Wang** — `guangzhong.wang@picb.ac.cn`
Shanghai Institute of Nutrition and Health (SINH), CAS. He is the computational
and transcriptomics lead, and the analysis code is published under his lab's
GitHub organisation (`WangLab-SINH`), so the processed matrix is almost certainly
his to send.

**CC: Zheng Wang** — `zheng.wang@pku.edu.cn` (Peking University)
Co-corresponding author, the imaging side of the paper. Worth copying because the
request concerns the *link* between expression and imaging, which is his half.

**Also CC if findable: Jie Li**, co-first author, in the same department as
Guang-Zhong Wang. The contributions statement says "T.B. and J.L. analyzed all or
parts of the data", so Jie Li is the person most likely to have the file on disk.
Their email is not in the paper; if a current address turns up, adding them makes
a reply faster.

Third corresponding author, **Meiyun Wang** (`mywang@ha.edu.cn`, Henan Provincial
People's Hospital), is on the clinical imaging side — not necessary for this.

**Paper:** Bo, Li, Hu, Zhang, Wang, Lv, Zhao, Ma, Qin, Yao, Wang, Wang & Wang
(2023), *Brain-wide and cell-specific transcriptomic insights into MRI-derived
cortical morphology in macaque monkeys*, **Nature Communications** 14:1499.
doi:10.1038/s41467-023-37246-w

---

## What to ask for — one specific thing

The **processed region-by-gene expression matrix**: 97 cortical regions ×
23,613 genes, as described in the Methods. Normalised values are what matter
(TPM/FPKM/CPM); raw counts plus the region key would also work.

Be specific that you have already checked the obvious places, or the likely reply
is a pointer back to them:

- SRA `PRJNA905082` — raw reads only
- Zenodo `10.5281/zenodo.7641873` — 19 kB, source code only
- GitHub `WangLab-SINH/Macaque_Brain_Transcriptome_MRI` — `Code/` directory only
- Source Data (33.8 MB XLSX) — checked all 39 sheets; the only regional
  expression table is a 408-gene neurotransmitter panel across 105 regions
- Supplementary Data 1–15 — gene lists and enrichment tables

Also worth asking for, in the same message since it costs nothing:
**the region key** mapping their region labels to D99 atlas identifiers. Without
it the matrix is much harder to use, and it is a small file they will have.

---

## How much to disclose

**Be straightforward.** Vagueness reads as either confusion or as someone
planning to scoop, and both lower the reply rate. The realistic risk here is not
being scooped — it is being ignored. Three sentences of honest context is the
single biggest lever on whether a busy PI replies.

There is also little to protect. The idea needs *their* data, which they already
have; if they wanted to do this themselves, nothing is stopping them. What you
actually want is for them to find it interesting enough to help.

Do say:
- What you want to build (a cross-species model predicting human cortical
  microvascular density from gene expression)
- Why their dataset specifically (macaque has both a regional transcriptome and
  measured vascular density; human has only the transcriptome)
- That you will cite them and are happy to acknowledge or involve them

Do not:
- Over-explain the discordance project. It is a different question and adds
  length without helping.
- Oversell. "I think this might work" travels better than "this will be a major
  advance."

**On affiliation:** independent researchers get lower reply rates than
institutional ones. Do not hide it — just lead with competence rather than
credentials. A public repository with reproducible code does more work than any
title, so linking it is worth more than explaining your position.

**Offer co-authorship.** If their data enables a released artifact, that offer is
genuinely appropriate, not a bribe, and it substantially raises reply rates.

---

## Draft

> **Subject:** Request for processed regional expression matrix — Bo et al. 2023 macaque brain transcriptome
>
> Dear Dr Wang,
>
> I am writing to ask whether you would be willing to share the processed
> region-by-gene expression matrix underlying Bo et al. (2023), *Nat Commun*
> 14:1499 — the 97 cortical regions × 23,613 genes described in your Methods.
>
> I have checked the public sources and believe only raw reads are deposited:
> SRA PRJNA905082 holds the sequencing data, and the Zenodo and GitHub releases
> contain the analysis code. The Source Data file includes a 408-gene
> neurotransmitter panel across regions, but not the full matrix. If a processed
> version exists, it would save considerable reprocessing.
>
> The reason I ask: I am building a cross-species model to estimate cortical
> microvascular density in the human brain from gene expression. Macaque is the
> only species where both a regional transcriptome and directly measured cortical
> vascular density (Autio et al., ferumoxytol-weighted laminar MRI) are available,
> so your dataset paired with theirs is the training set the approach depends on.
> Human has the transcriptome but no comparable vascular map, which is the gap the
> work aims to fill.
>
> If it is straightforward, the region key mapping your labels to D99 identifiers
> would also be very helpful.
>
> Any resulting work would of course cite the paper, and I would be glad to
> acknowledge your contribution or discuss involvement if the direction is of
> interest to you. The analysis code is public at
> https://github.com/Beyond-InFinnity/discordance-transcriptomics.
>
> With thanks for your time,
>
> Connor Finnerty

---

## Practical notes

- Send in the morning, China Standard Time (UTC+8) — Shanghai and Beijing.
- If no reply in two weeks, send one short follow-up to the same thread. One
  only.
- If that fails, Jie Li or Tingting Bo directly (co-first authors) are the next
  step — early-career authors often respond faster than PIs.
- Keep it to the length above. Longer emails do worse.
