# discordance-transcriptomics

About 40% of cortical voxels that respond to a task show oxygen metabolism moving
the *opposite* way to the BOLD signal (Epp et al., 2025). The leading explanation
is that association cortex has sparser capillaries, so blood supply cannot keep
up. Nobody had checked that against molecular data.

This repository checks it, using post-mortem gene expression from the Allen Human
Brain Atlas. It also measures something the field usually skips: whether the test
could have worked at all.

**Short answer.** Vascular gene expression does predict baseline oxygen
extraction. It does not predict discordance. And most of the tests could never
have found an ordinary effect in the first place. That last part turned out to be
the useful finding.

---

## What it found

**Pericyte and mural-cell genes predict baseline oxygen extraction fraction.**
ρ = −0.391, same sign in all 240 tests, competitive *p* = 0.0004. It survives
four independent ways of computing it, and gets stronger when the
sensory-to-association hierarchy is partialled out.

It does not survive correction for the family of tests it belongs to (adjusted
*p* = 0.130), and the 95% confidence interval is wide: [−0.62, −0.16]. Treat it
as suggestive.

**Nothing predicts discordance.** Not the frozen gene sets, not any of 15,562
individual genes, and not baseline extraction. The mediation model that would
have supported the capillary explanation returns nothing across 15,840 fits.

**Most of the design could not have worked.** For each of the 44 planned tests we
computed the smallest true effect it could resolve, given how reliably both maps
were measured:

| | |
|---|---|
| tests that could resolve an ordinary effect (ρ ≈ 0.3) | 0 of 44 |
| tests untestable at any effect size | 4 |
| limited by the gene data rather than the imaging | 38 |
| smallest resolvable effect anywhere in the design | 0.30 |

More subjects would not have helped. The bottleneck is the gene side.

A large part of that is self-inflicted. Averaging a gene set into a single map
destroys the signal whenever the genes resemble each other less than their shared
noise does. That is the usual case in a microarray atlas, because every gene is
measured from the same tissue punches. One Gene Ontology set here has *negative*
reliability as an average, while its individual genes replicate fine.

**What the next study would need.** To resolve ρ = 0.30 against baseline
extraction, a gene panel needs reliability 0.68. The best one here reaches 0.670.
Against discordance the requirement is 1.15, which is impossible: no improvement
to the gene side alone can rescue it.

---

## The data

The released artifact is `data/derived/annotation/`. It is 334 cortical parcels
across three parcellations, with:

- baseline oxygen extraction, blood flow, and metabolic rate
- flow-metabolism coupling angle, and discordance split by mode
- scanner dropout coverage and venous partial-volume, as covariates
- a JSON schema, and a reliability label on every column

```python
import pandas as pd
df = pd.read_csv("data/derived/annotation/discordance_annotation.csv")
```

Read the reliability labels before using a column. Several are too noisy to
support the analysis you may want.

`geneset_profiles.csv` holds the gene-set results behind the findings above.

---

## Reproducing it

Setup needs Python 3.11 and Connectome Workbench. Install Workbench first;
`neuromaps` shells out to it for every surface transform and a missing install is
the most common failure here.

```bash
wget https://humanconnectome.org/storage/app/media/workbench/workbench-linux64-v1.5.0.zip
unzip workbench-linux64-v1.5.0.zip -d ~/opt/
export WORKBENCH_DIR="$HOME/opt/workbench/bin_linux64"
wb_command -version

uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
nbstripout --install --attributes .gitattributes

python scripts/fetch_all.py
pytest -q && ruff check src/ scripts/ tests/
```

Then:

```bash
scripts/regenerate_all.sh          # everything, about 4 hours
scripts/regenerate_all.sh --quick  # reduced permutations, smoke test only
```

`results/` is not tracked. It rebuilds from the command above, and every file
carries a manifest recording the git commit, config hash, package versions and
seed that produced it. Three gates check the output:

| script | what it checks |
|---|---|
| `audit_provenance.py` | every artifact came from one commit, one clean tree, one run |
| `check_paper_numbers.py` | every number in the manuscript matches its artifact |
| `verify_references.py` | every citation resolves to the paper it claims |

Three pinned dependencies deviate from a plain install, each explained inline in
`requirements.txt`. `abagen` is pinned to a commit rather than a release because
the PyPI build calls a pandas method removed in 2.0. `pandas` is capped below 3.0
for the same reason. `setuptools` is capped below 81 because abagen imports
`pkg_resources`.

`nbstripout` has to be registered per checkout. `.gitattributes` is committed but
the filter itself lives in `.git/config` and does not survive a clone.

---

## How the statistics are guarded

Imaging transcriptomics has three well-known ways to produce a false positive.
Each is handled structurally rather than by convention.

**Smooth maps correlate by chance.** Two arbitrary brain maps reach ρ ≈ 0.4 with
no relationship at all. Every correlation here is tested against
spatial-autocorrelation-preserving nulls, and `src/stats/spatial.py` has no code
path that returns a p-value without one.

**Preprocessing choices move results.** Published work shows AHBA parameter
choices can shift a correlation by enough to flip its sign. Every effect is
reported across 120 pipelines, not one.

**Association cortex differs from sensory cortex on everything.** Any map varying
along that axis correlates with any gene set varying along it. Every result is
reported before and after partialling the principal gradient and cortical myelin.

---

## Layout

```
config/     seed, parcellation, and the frozen gene-set definitions
src/        importable, tested analysis code
scripts/    one thin wrapper per pipeline step
tests/      307 tests
paper/      manuscript, verified reference list
app/        Streamlit browser for the annotation table
data/       gitignored, except the released annotation table
results/    gitignored; rebuilds from scripts/regenerate_all.sh
```

`CLAUDE.md` is the full specification. Its §3 (hard rules) and §13
(stop-and-ask) are binding on any work here.

---

## Limitations

- The Allen atlas is six post-mortem adults, five usable, mostly left
  hemisphere. It is a modal brain, not a sample. No individual-level claim
  follows from it.
- Spatial correlation is not mechanism. The mediation model is suggestive at
  best, and here it is null.
- One imaging dataset, 40 subjects, one scanner, one site.
- Projecting volumes to the surface discards subcortex and cerebellum.
- The mqBOLD method carries its own assumptions about vessel geometry and blood
  volume, and those propagate into every extraction and metabolism value.
- One positive control fails: our metabolic rate map disagrees with the PET
  reference. So does the authors' published map, which locates the disagreement
  in the method rather than in this reconstruction.

---

## Licence and citation

Analysis code is MIT. ds004873 is CC0; cite Epp et al. (2025) and the OpenNeuro
DOI when using it. The macaque vascular maps are from Autio et al. (2025) and
should be cited directly.
