#!/usr/bin/env python
"""Resolve every reference the paper cites against Crossref. ⛔ Do not skip.

A reference list written from memory is a list of plausible-looking claims about
other people's work. Author order, year, journal and volume are exactly the
details that degrade in recall, and a wrong citation is not a typo — it
misattributes work, and reviewers check.

So the reference list is *derived*, not authored. Each entry below is a search
key; this script asks Crossref for the canonical record and prints what came
back, including the match score, so a wrong hit is visible rather than silently
formatted into the bibliography.

Entries that do not resolve are reported as unresolved rather than guessed.
Books are expected to fail — Crossref indexes them unevenly — and are marked as
manual.

Usage
-----
    python scripts/verify_references.py                  # verify and report
    python scripts/verify_references.py --emit-markdown  # formatted list
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "references.json"

# Crossref asks for a contact address so it can reach you about heavy use, and
# routes identified traffic through a faster pool.
UA = "discordance-transcriptomics/0.1 (mailto:infinnity12@gmail.com)"

# (key, DOI, distinctive title fragment the record MUST contain).
#
# Every entry is a DOI plus an assertion. Fuzzy title search was tried first and
# resolved 6 of 23 to the wrong record; hardening the junk filter still left
# three wrong, including a conference abstract standing in for a journal article
# and an unrelated review titled "MR Imaging of the Brain". Bibliographic search
# is not precise enough to cite from.
#
# A DOI recalled from memory is equally fallible -- so the fragment check makes
# that fallibility *visible*: a wrong DOI resolves to a real paper with the
# wrong title, and the script reports MISMATCH rather than formatting it in.
REFERENCES: list[tuple[str, str, str]] = [
    ("epp2025", "10.1038/s41593-025-02132-9", "oppose oxygen metabolism"),
    ("markello2021", "10.7554/eLife.72129", "abagen toolbox"),
    (
        "alexanderbloch2018",
        "10.1016/j.neuroimage.2018.05.070",
        "spatial correspondence between maps",
    ),
    ("burt2020", "10.1016/j.neuroimage.2020.117038", "Generative modeling of brain maps"),
    (
        "margulies2016",
        "10.1073/pnas.1608282113",
        "principal gradient of macroscale cortical organization",
    ),
    (
        "arnatkeviciute2019",
        "10.1016/j.neuroimage.2019.01.011",
        "practical guide to linking brain-wide gene expression",
    ),
    ("schaefer2018", "10.1093/cercor/bhx179", "Local-Global Parcellation"),
    ("hawrylycz2012", "10.1038/nature11405", "anatomically comprehensive atlas"),
    ("markello2022", "10.1038/s41592-022-01625-w", "neuromaps"),
    ("vaishnavi2010", "10.1073/pnas.1010459107", "aerobic glycolysis in the human brain"),
    (
        "hall2014",
        "10.1038/nature13165",
        "Capillary pericytes regulate cerebral blood flow",
    ),
    (
        "attwell2010",
        "10.1038/nature09613",
        "Glial and neuronal control of brain blood flow",
    ),
    ("iadecola2017", "10.1016/j.neuron.2017.07.030", "Neurovascular Unit Coming of Age"),
    (
        "botviniknezer2020",
        "10.1038/s41586-020-2314-9",
        "single neuroimaging dataset by many teams",
    ),
    ("yeo2011", "10.1152/jn.00338.2011", "organization of the human cerebral cortex"),
    ("desikan2006", "10.1016/j.neuroimage.2006.01.021", "automated labeling system"),
    (
        "benjamini1995",
        "10.1111/j.2517-6161.1995.tb02031.x",
        "Controlling the False Discovery Rate",
    ),
    ("liberzon2015", "10.1016/j.cels.2015.12.004", "Hallmark Gene Set Collection"),
    ("fornito2019", "10.1016/j.tics.2018.10.005", "Bridging the Gap"),
    ("steegen2016", "10.1177/1745691616658637", "Multiverse Analysis"),
    ("seidlitz2020", "10.1038/s41467-020-17051-5", "regional brain vulnerability"),
    ("ceballos2025", "10.1162/netn_a_00425", "control costs of human brain dynamics"),
    ("buxton2009", "10.1017/CBO9780511605505", "Functional Magnetic Resonance Imaging"),
]


def _get(url: str) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as fh:
            return json.load(fh)
    except Exception:
        return None


# Crossref indexes a great deal that is *about* a paper alongside the paper.
# Taking the top hit resolved 6 of 23 references to the wrong record: three
# Faculty Opinions (F1000) recommendations, a commentary "comment on ..." rather
# than the article it comments on, a table component lifted from some other
# manuscript, and -- worst, because it looks right -- a different paper on brain
# aerobic glycolysis in ageing standing in for the one on its regional
# distribution. Every one would have read as a plausible citation.
_JUNK_PREFIXES = (
    "faculty opinions recommendation",
    "faculty of 1000",
    "table ",
    "figure ",
    "supplementary",
    "correction",
    "erratum",
    "retraction",
)
_JUNK_SUBSTRINGS = ("comment on", "reply to", "response to")
# Types that are commentary or fragments rather than the work itself.
_JUNK_TYPES = {"component", "peer-review", "grant", "dataset"}


def _is_junk(item: dict) -> bool:
    title = ((item.get("title") or [""])[0] or "").lower()
    if any(title.startswith(p) for p in _JUNK_PREFIXES):
        return True
    if any(sub in title for sub in _JUNK_SUBSTRINGS):
        return True
    return item.get("type", "") in _JUNK_TYPES


def resolve(doi: str) -> dict | None:
    d = _get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
    return d["message"] if d else None


def fmt(m: dict) -> dict:
    authors = m.get("author", []) or []
    names = [
        f"{a.get('family', '?')} {''.join(p[0] for p in a.get('given', '').split() if p)}"
        for a in authors
    ]
    date = m.get("issued", {}).get("date-parts", [[None]])[0]
    return {
        "authors": names,
        "year": date[0] if date else None,
        "title": (m.get("title") or ["?"])[0],
        "journal": (m.get("container-title") or [""])[0],
        "volume": m.get("volume", ""),
        "pages": m.get("page", ""),
        "doi": m.get("DOI", ""),
        "score": round(m.get("score", 0.0), 1),
    }


def citation(r: dict) -> str:
    a = r["authors"]
    if not a:
        who = "?"
    elif len(a) == 1:
        who = a[0]
    elif len(a) <= 3:
        who = ", ".join(a[:-1]) + " & " + a[-1]
    else:
        who = f"{a[0]} et al."
    bits = [f"{who} ({r['year']}). {r['title']}."]
    if r["journal"]:
        j = f" *{r['journal']}*"
        if r["volume"]:
            j += f", {r['volume']}"
        if r["pages"]:
            j += f", {r['pages']}"
        bits.append(j + ".")
    if r["doi"]:
        bits.append(f" https://doi.org/{r['doi']}")
    return "".join(bits)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emit-markdown", action="store_true")
    args = ap.parse_args()

    resolved: dict[str, dict] = {}
    unresolved: list[str] = []
    print(f"\n{'=' * 78}\nREFERENCE RESOLUTION (Crossref)\n{'=' * 78}\n", file=sys.stderr)
    mismatched: list[str] = []
    for key, doi, must_contain in REFERENCES:
        m = resolve(doi)
        time.sleep(0.3)  # polite pool
        if not m:
            unresolved.append(key)
            print(f"  UNRESOLVED  {key}  ({doi})", file=sys.stderr)
            continue
        r = fmt(m)
        if must_contain.lower() not in r["title"].lower():
            mismatched.append(key)
            print(
                f"  MISMATCH    {key}\n"
                f"      DOI {doi} resolves to: {r['title'][:80]}\n"
                f"      expected the title to contain: {must_contain!r}",
                file=sys.stderr,
            )
            continue
        resolved[key] = r
        print(f"  ok  {key:<20} {r['year']}  {r['title'][:66]}", file=sys.stderr)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        json.dump(
            {"resolved": resolved, "unresolved": unresolved, "mismatched": mismatched},
            fh,
            indent=1,
        )

    print(f"\n{'=' * 78}", file=sys.stderr)
    print(
        f"resolved {len(resolved)}/{len(REFERENCES)}; unresolved: {unresolved or 'none'}",
        file=sys.stderr,
    )
    print(f"-> {OUT}\n", file=sys.stderr)

    if args.emit_markdown:
        for key in sorted(resolved, key=lambda k: (resolved[k]["authors"] or [""])[0]):
            print(f"- {citation(resolved[key])}")
    return 1 if (unresolved or mismatched) else 0


if __name__ == "__main__":
    raise SystemExit(main())
