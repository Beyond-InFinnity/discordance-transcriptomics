# Moving this project to another machine

The repository is 2 MB. Everything else is either re-fetchable by script or
needs one deliberate copy. This note says which is which.

---

## The short version

```bash
git clone https://github.com/Beyond-InFinnity/discordance-transcriptomics
cd discordance-transcriptomics
uv venv --python 3.11 .venv && uv pip install -r requirements.txt
nbstripout --install --attributes .gitattributes

# Connectome Workbench — needed by neuromaps for every surface transform
wget https://humanconnectome.org/storage/app/media/workbench/workbench-linux64-v1.5.0.zip
unzip workbench-linux64-v1.5.0.zip -d ~/opt/
export WORKBENCH_DIR="$HOME/opt/workbench/bin_linux64"

python scripts/fetch_all.py           # or src/data/fetch.py, see below
```

Then start a Claude Code session in the directory. `CLAUDE.md` is the
specification, `docs/BACKGROUND.md` is the plain-language explainer, and
`docs/RESUME_HERE.md` lists the open items.

---

## What travels in git (2 MB)

All code, all configs including the frozen gene list, every result manifest,
the result CSVs, and the documentation. That is the entire intellectual state
of the project — nothing about *what was found* lives outside the repository.

## What has to be re-fetched or copied

| what | size | verdict |
|---|---:|---|
| `data/raw/ds004873` | 3.9 GB | **re-fetch** — `src/data/fetch.py` is automated and verifies every file against the checksum embedded in its git-annex key |
| AHBA microarray (`~/abagen-data`) | 3.6 GB | **re-fetch**, but see the donor note below |
| `~/neuromaps-data`, `~/nnt-data` | 160 MB | **re-fetch** — automatic on first use |
| `data/external/macaque_human_alignment` | 151 MB | **mostly re-fetch** from the public GitHub repo — but see the BALSA note |
| `data/derived/**` | 3.3 GB | **regenerate** — all of it is reproducible, and on a larger machine it regenerates *better* than the copy |

### Copy this one thing: the BALSA download

`data/external/macaque_human_alignment/Autio_eLife2025_km_1vjnV.zip` (57 MB)
came from BALSA, which requires a login and cannot be fetched by script. Either
copy that single file across or download it again through the browser. Losing
it costs the macaque vascular analysis.

### Donor 15496 will still be missing

Re-fetching AHBA gets five of six donors. The sixth returns an error from the
Allen Institute and their catalogue no longer lists it, so a fresh download on
a new machine reproduces exactly the same five. Copying `~/abagen-data` gains
nothing over re-fetching — we never had it.

If you want to try recovering it, that is a conversation with the Allen
Institute, not a scripting problem.

### One rsync trap that fails silently

`rsync --exclude=data` excludes **every** path component named `data`, not just
the top-level directory — so it also drops `src/data/`, and the copy looks fine
until an import fails. Anchor the pattern:

```bash
rsync -a --exclude=/.venv --exclude=/data --exclude=/.git --exclude='__pycache__' \
  ./ host:~/discordance-transcriptomics/
```

The leading slash pins the pattern to the transfer root. Verify with
`find src -name "__init__.py" | wc -l` — it should return 6.

### Machine-local memory

Notes I keep under
`~/.claude/projects/-home-connor-Documents-projects-discordance-transcriptomics/memory/`
do not travel with the repository. Copy that directory if you want the new
session to start with the same accumulated preferences and context.

---

## What actually improves with 64 GB

**The multiverse completes.** On the 15 GB machine, 24 of 120 processing
pipelines were killed by the kernel — every one of them the same corner, where
an expensive probe-selection method met hemisphere mirroring and peaked near
7 GB against roughly 6 GB free. With 64 GB all 120 run, and the gap in the grid
closes.

**It runs in minutes rather than hours.** The grid had to be sequential because
one cell nearly exhausted memory. At 64 GB, eight parallel workers fit
comfortably:

```bash
python scripts/p3_multiverse.py --n-jobs 8     # ~15 min instead of ~2 h
python scripts/p4_genesets.py --n-draws 10000
```

**Bigger parcellations become practical.** Schaefer-400 doubles the parcel count
and roughly doubles peak memory; it was not attemptable before.

Regenerating rather than copying `data/derived/` is therefore the better move —
you get the complete 120-cell grid instead of the 96-cell one.

---

## The GPUs will not help, and that is fine

`CLAUDE.md` §4.2 says it directly: *"This project is CPU-bound and laptop-scale.
Do not write GPU code. If a task seems to need a GPU, you have misunderstood the
task."* That remains true, and the two cards are close to irrelevant here.

Nothing in the pipeline is a dense-linear-algebra problem:

- expression extraction is pandas and file I/O
- surface transforms shell out to Connectome Workbench, which is CPU
- the spin test is now a single matrix product over ~100 parcels — already
  microseconds, and the transfer to a card would cost more than the compute
- the competitive null is 10,000 small resamples, bound by Python overhead

The honest bottleneck this whole project has faced is **memory**, and secondly
single-core speed. Both are addressed by the 64 GB; neither is addressed by a
GPU.

Where a card could eventually earn its place is outside the current protocol:
the data-driven arm's partial-least-squares work if it were pushed to
vertex-level rather than parcel-level, or any future model fit over the full
voxel grid rather than 100 regions. Neither is on the critical path, and
inventing GPU work to justify the hardware would be the wrong instinct.

---

## First commands on the new machine

```bash
# 1. Confirm the environment
wb_command -version
python -m pytest -q                    # 132 tests
ruff check src/ scripts/ tests/

# 2. Fetch the data (see fetch patterns in data/MANIFEST.yaml)
python -c "import abagen; abagen.fetch_microarray(donors='all')"

# 3. Rebuild the derived products, now completely
python scripts/p3_multiverse.py --n-jobs 8
python scripts/p4_genesets.py --n-draws 10000
python scripts/make_figures.py
```

Everything is idempotent and cached by content hash, so re-running is safe and
an interruption resumes.
