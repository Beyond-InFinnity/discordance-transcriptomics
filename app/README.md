# Discordance annotation browser

A Streamlit front-end for the released annotation table (deliverable #2). It
answers two questions:

1. **I care about region X — is BOLD trustworthy there?** Per-parcel discordance
   risk, split into its two physiological modes, with the molecular profile and
   the data-quality checks that qualify it.
2. **I have a statistical map — does it land in discordant territory?** Upload an
   MNI152 volume; it is projected to `fsaverage5` and parcellated, and scored
   against per-parcel discordance risk weighted by your own activation.

```bash
streamlit run app/main.py
```

## What it reads

| file | built by |
|---|---|
| `data/derived/annotation/discordance_annotation.csv` | `scripts/build_annotation.py --ahba` |
| `data/derived/annotation/discordance_annotation.schema.json` | same |
| `data/derived/annotation/geneset_profiles.csv` | `scripts/build_geneset_profiles.py` |

All three are gitignored under `data/`, so a fresh clone has an empty app until
the pipeline has run. `scripts/regenerate_all.sh` builds all of them.

Column tooltips and the definitions table are read **from the JSON schema**
rather than written into the app, so they cannot drift from the released
artifact.

## Design rules

**No analysis lives here.** The app reads precomputed artifacts. The one
exception is coordinate transforms for uploaded maps, which go through
`src.data.parcellate` because R4 forbids hand-rolled ones. Anything you are
tempted to compute in this directory belongs in `src/`.

**Every number arrives with what qualifies it.** A parcel's discordance risk is
not meaningful without its scanner-dropout coverage, its map reliability, and
how many AHBA tissue samples sit inside it. The table carries all of that and
the app shows it, including warnings when a parcel is mostly dropout or has zero
donor samples. The molecular profile is a median with an inter-quartile range
across multiverse pipelines, never a single number (R6).

**The gene-set scoring reuses the analysis's own functions** — `load_genesets`
from Phase 4 and `select_cells` from Phase 4b — so the app cannot report a
number that disagrees with the paper. An earlier ad-hoc version walked the YAML
itself and silently found 6 of the 11 frozen sets, because the MSigDB sets
resolve through `source_key` rather than an inline gene list.

## Upload requirements

Map scoring needs `neuromaps` and Connectome Workbench (`wb_command`) on `PATH`
— see CLAUDE.md §4.1. Without them the region browser still works and the upload
tab reports the real error rather than failing silently.

Uploaded maps must be **MNI152 volumes**. Scoring runs at `schaefer200x7`, the
primary parcellation, because that is where the expression multiverse is built.

## Status

The annotation table is versioned **provisional**. Columns may change before
1.0. The app displays the table's version and status in the sidebar, so a
screenshot always carries its own provenance.
