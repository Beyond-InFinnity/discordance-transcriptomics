# Data request: macaque regional expression matrix

## Who to write to

**Primary: Guang-Zhong Wang** — `guangzhong.wang@sinh.ac.cn`
Shanghai Institute of Nutrition and Health (SINH), CAS. He is the computational
and transcriptomics lead, and the analysis code is published under his lab's
GitHub organisation (`WangLab-SINH`), so the processed matrix is almost certainly
his to send.

> **The address printed in the paper is stale.** It gives
> `guangzhong.wang@picb.ac.cn`, but PICB (the CAS-MPG Partner Institute for
> Computational Biology) merged into SINH around 2020. The SINH staff page lists
> `guangzhong.wang@sinh.ac.cn` — same local part, current domain. Send to the
> sinh.ac.cn address; adding the picb one as a second recipient costs nothing if
> it still forwards.

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

## Draft — send as plain text

**To:** guangzhong.wang@sinh.ac.cn
**Cc:** zheng.wang@pku.edu.cn, guangzhong.wang@picb.ac.cn

**Subject:** Data request - processed region-by-gene matrix from Bo et al. 2023 (Nat Commun 14:1499)

```
Dear Dr Guang-Zhong Wang and Dr Zheng Wang,

I am writing to ask whether you would be willing to share the processed
region-by-gene expression matrix underlying Bo et al. 2023, Nat Commun
14:1499 - the 97 cortical regions by 23,613 genes described in your Methods.

I have looked through the public sources and believe only raw reads are
deposited: SRA PRJNA905082 holds the sequencing data, and the Zenodo and
GitHub releases contain the analysis code. The Source Data file includes a
408-gene neurotransmitter panel across regions, but not the full matrix.

The reason I ask is that I am working on a cross-species model to estimate
cortical microvascular density in the human brain from gene expression.
Macaque appears to be the only species where both a regional transcriptome
and directly measured cortical vascular density are available - yours paired
with the ferumoxytol laminar MRI maps from Autio and colleagues. Human has
the transcriptome but no comparable vascular measurement, which is the gap
I am trying to address.

If it is easy to include, the key mapping your region labels to D99
identifiers would also be very helpful.

If sharing the matrix is not straightforward, I am equally happy to
reprocess from SRA - in that case any note on the alignment and
quantification settings you used would be valuable, so that my results stay
comparable to yours.

Any resulting work would of course cite the paper, and I would be glad to
acknowledge your contribution or to discuss involvement if the direction is
of interest. My analysis code is public at
github.com/Beyond-InFinnity/discordance-transcriptomics

With thanks for your time,

Connor Finnerty
infinnity12@gmail.com
```

**Why it is shaped this way**

- The ask is the first sentence. A busy PI should know what is wanted before
  deciding whether to keep reading.
- The paragraph on what you already checked prevents the most likely reply,
  which is a pointer back to SRA.
- The offer to reprocess from SRA gives them a near-zero-cost way to help even
  if the matrix is awkward to dig out, and it signals you are not asking to be
  carried.
- No affiliation is claimed and none is explained. The missing institution line
  is itself the honest signal; drawing attention to it only invites a filter.
  Answer directly if asked.
- No greeting flourishes, no adjectives about significance. Around 250 words.

---

## Will Gmail even reach them?

Yes. The Great Firewall blocks traffic *from inside China to Google*, which is
why people there cannot open Gmail without a VPN. That is the opposite direction
from this. Google's servers connect outbound to the Chinese mail servers, the
recipients read their mail on their own institutional systems inside China, and
replies come back the same way. Nothing in the path is a blocked service.

**The real risk is spam filtering, and it fails silently** — no bounce, no
delivery, indistinguishable from being ignored. Design around it:

- Plain text, no attachments on first contact.
- Minimise links. The GitHub URL in the draft is a mild spam signal; either drop
  it or write it bare, without the protocol prefix.
- Send to all three addresses at once — different servers, different filters.
- Keep the subject specific and dull.

Backup channels, in order: **ResearchGate** (works from China, academics check
it); then **Jie Li or Tingting Bo** directly, since early-career authors reply
faster and tend to have less aggressive filters. Not LinkedIn — it discontinued
its China service in 2023.

## Practical notes

- Send in the morning, China Standard Time (UTC+8) — Shanghai and Beijing.
- If no reply in two weeks, send one short follow-up to the same thread. One
  only.
- If that fails, Jie Li or Tingting Bo directly (co-first authors) are the next
  step — early-career authors often respond faster than PIs.
- Keep it to the length above. Longer emails do worse.
