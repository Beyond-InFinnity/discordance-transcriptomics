#!/usr/bin/env python
"""The five display items the manuscript actually argues with.

``make_figures.py`` predates the rewrite and depicts the analysis as it was
originally framed: eight panels of maps, networks and diagnostics. None of them
shows which tests the design could resolve, which is now the paper's central
result. A figure set that does not carry the argument is decoration.

These five do:

1. **What this design can resolve.** Implied true effect against detectability
   floor for all 33 gene-set x outcome tests. Everything above the diagonal is
   resolvable; three points are. This is Figure 1 because §3.2 precedes the
   hypothesis tests.
2. **The primary effect across the multiverse**, and what the hierarchy control
   does to it under both specifications.
3. **The mediation as a bound**, with the undetectable region shaded — the
   distinction between "no effect" and "no power" made visual.
4. **Positive controls**, including the one that fails, on both map sources.
5. **The whole-chain dropout gate**, all twelve links against the §9 threshold.

Design follows the target venue's two-column width (180 mm), the Okabe-Ito
colour-vision-safe palette, and a 7 pt floor on text at final size. Everything is
drawn from ``results/``; nothing is hand-entered.

Usage
-----
    python scripts/make_manuscript_figures.py
    python scripts/make_manuscript_figures.py --out /tmp/figs   # preview
"""

# Axis labels are read by humans, not parsed: Greek rho, the true minus sign and
# the division sign are the correct typography for a figure and deliberate here.
# ruff: noqa: RUF001
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.manifest import manifest

logger = logging.getLogger("manuscript_figures")

R = Path("results")

# Okabe-Ito: distinguishable under all common colour-vision deficiencies and in
# greyscale. Rainbow and red/green encodings fail both.
OI = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#999999",
}

MM = 1 / 25.4
W2 = 180 * MM  # two-column width for the target venue
W1 = 85 * MM  # single column

STYLE = {
    "font.family": "sans-serif",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.0,
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # embed as TrueType so text stays editable
}

OUTCOME_COLOUR = {
    "baseline_oef": OI["blue"],
    "discordance_extraction": OI["vermillion"],
    "discordance_overshoot": OI["orange"],
}
OUTCOME_LABEL = {
    "baseline_oef": "baseline OEF",
    "discordance_extraction": "discordance (extraction)",
    "discordance_overshoot": "discordance (overshoot)",
}


def panel(ax, letter: str) -> None:
    ax.text(
        -0.16,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def save(fig, out: Path, name: str) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(str(p))
    plt.close(fig)
    logger.info("wrote %s", name)
    return paths


# ---------------------------------------------------------------- data ------
def resolvability() -> pd.DataFrame:
    floors = pd.read_csv(R / "p0c_detectability_floor.csv")
    summary = pd.read_csv(R / "p4_genesets_summary.csv")
    panel_rel = pd.read_csv(R / "p0c_geneset_reliability.csv").set_index("gene_set")
    label = {v: k for k, v in OUTCOME_LABEL.items()}
    rows = []
    for _, r in summary.iterrows():
        if r.target not in OUTCOME_LABEL:
            continue
        m = floors[
            (floors.gene_set == r.gene_set)
            & (floors.brain_map == OUTCOME_LABEL[r.target])
        ]
        if not len(m):
            continue
        ceil = float(m.attenuation_ceiling.iloc[0])
        floor = float(m.detectable_true_rho.iloc[0])
        ok = ceil > 0 and not math.isnan(ceil)
        rows.append(
            {
                "gene_set": r.gene_set,
                "outcome": r.target,
                "implied_true": abs(r.rho_median) / ceil if ok else np.nan,
                "floor": floor,
                "panel_rel": float(panel_rel.loc[r.gene_set, "reliability_panel"]),
                "resolvable": bool(ok and abs(r.rho_median) / ceil >= floor),
                "untestable": not ok,
            }
        )
    _ = label
    return pd.DataFrame(rows)


# ------------------------------------------------------------- figures ------
def figure1(out: Path) -> list[str]:
    """What this design can resolve."""
    d = resolvability()
    rel = pd.read_csv(R / "p0c_geneset_reliability.csv").sort_values("reliability_panel")

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(W2, 82 * MM), gridspec_kw={"width_ratios": [1.05, 1]}
    )

    # -- A: implied true effect vs the floor it must clear ------------------
    # Resolvable means implied true effect >= floor, i.e. ABOVE the diagonal. A
    # first version shaded and labelled this the wrong way round, putting all
    # three resolvable tests inside a region captioned "not resolvable".
    lim = 1.0
    ax.fill_between(
        [0, lim], [0, 0], [0, lim], color=OI["grey"], alpha=0.16, lw=0, zorder=0
    )
    ax.plot([0, lim], [0, lim], color=OI["black"], lw=0.8, zorder=3)
    ax.text(
        0.62,
        0.955,
        "resolvable",
        transform=ax.transAxes,
        fontsize=7,
        color="#333333",
        ha="center",
        style="italic",
        zorder=2,
    )
    ax.text(
        0.955,
        0.055,
        "not resolvable",
        transform=ax.transAxes,
        fontsize=7,
        color="#333333",
        ha="right",
        style="italic",
        zorder=2,
    )
    for oc, g in d.groupby("outcome"):
        ax.scatter(
            g.floor,
            g.implied_true,
            s=30,
            alpha=0.9,
            facecolor=OUTCOME_COLOUR[oc],
            edgecolor="white",
            linewidth=0.5,
            label=OUTCOME_LABEL[oc],
            zorder=4,
        )
    # Leader lines. The resolvable points sit close together near the diagonal,
    # so floating labels cannot be matched to them unambiguously.
    #
    # Anchor positions are derived, not hard-coded. A first version kept a dict
    # keyed by the three gene sets resolvable in the full run; under any other
    # settings a different set qualifies and the figure died on a KeyError. A
    # figure must not encode its own results.
    res = d[d.resolvable].sort_values("floor").reset_index(drop=True)
    for i, r in res.iterrows():
        left = i % 2 == 0
        ax.annotate(
            r.gene_set.replace("HALLMARK_", "").replace("_", " ").lower(),
            xy=(r.floor, r.implied_true),
            xytext=(-14 if left else 14, 16 if left else -18),
            textcoords="offset points",
            fontsize=6.5,
            ha="right" if left else "left",
            va="center",
            zorder=6,
            arrowprops=dict(
                arrowstyle="-", lw=0.5, color="#666666", shrinkA=1, shrinkB=3
            ),
        )
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("detectability floor (smallest resolvable true |ρ|)")
    ax.set_ylabel("implied true |ρ|  (observed ÷ attenuation ceiling)")
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.88),
        handletextpad=0.25,
        borderpad=0.0,
        labelspacing=0.35,
    )
    ax.set_title(
        f"{int(d.resolvable.sum())} of {len(d)} tests resolvable",
        fontsize=8,
        pad=6,
    )
    panel(ax, "A")

    # -- B: gene-set panel reliability, the limiting term -------------------
    y = np.arange(len(rel))
    colours = [
        OI["vermillion"] if v <= 0 else (OI["blue"] if v >= 0.55 else OI["grey"])
        for v in rel.reliability_panel
    ]
    bx.hlines(y, 0, rel.reliability_panel, color=colours, lw=1.5)
    bx.scatter(rel.reliability_panel, y, s=24, color=colours, zorder=3)
    bx.axvline(0, color=OI["black"], lw=0.6)
    bx.set_yticks(y)
    bx.set_yticklabels(
        [g.replace("HALLMARK_", "").replace("_", " ").lower() for g in rel.gene_set]
    )
    # Right-hand ticks: on the left they ran into panel A's plot area.
    bx.yaxis.tick_right()
    bx.spines["left"].set_visible(False)
    bx.spines["right"].set_visible(True)
    bx.set_xlabel("gene-set panel reliability across donors")
    bx.set_xlim(-0.09, 0.78)
    bx.annotate(
        "negative: does not replicate\nacross donors, untestable",
        xy=(-0.011, 0),
        xytext=(0.20, 2.0),
        textcoords="data",
        fontsize=6,
        color=OI["vermillion"],
        va="center",
        arrowprops=dict(arrowstyle="-", lw=0.5, color=OI["vermillion"]),
    )
    bx.set_title("the limiting term in every test it enters", fontsize=8, pad=6)
    bx.margins(y=0.05)
    panel(bx, "B")

    fig.subplots_adjust(wspace=0.06)
    return save(fig, out, "MF1_resolvable_tests")


def figure2(out: Path) -> list[str]:
    """The primary effect across the multiverse, and under hierarchy control."""
    full = pd.read_csv(R / "p4_genesets_full.csv")
    peri = full[(full.gene_set == "pericyte_mural") & (full.target == "baseline_oef")]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W2, 68 * MM))

    # -- A: every pipeline -------------------------------------------------
    rng = np.random.default_rng(42)
    jit = rng.uniform(-0.16, 0.16, len(peri))
    sig = peri.p_spin < 0.05
    ax.axvline(0, color=OI["black"], lw=0.7)
    ax.scatter(
        peri.rho[~sig],
        jit[~sig],
        s=13,
        facecolor="white",
        edgecolor=OI["grey"],
        linewidth=0.6,
        label="spin p ≥ 0.05",
        zorder=3,
    )
    ax.scatter(
        peri.rho[sig],
        jit[sig],
        s=13,
        color=OI["blue"],
        alpha=0.75,
        label="spin p < 0.05",
        zorder=4,
    )
    med = float(peri.rho.median())
    ax.plot([med, med], [-0.30, 0.30], color=OI["vermillion"], lw=1.4, zorder=5)
    ax.annotate(
        f"median {med:.3f}",
        (med, 0.30),
        color=OI["vermillion"],
        fontsize=6.5,
        ha="center",
        va="bottom",
    )
    ax.set_ylim(-0.45, 0.45)
    ax.set_yticks([])
    ax.set_xlabel("Spearman ρ, pericyte/mural vs baseline OEF")
    ax.set_title(
        f"{len(peri)} tests  ·  {peri.cell.nunique()} pipelines × "
        f"{peri.stability_threshold.nunique()} thresholds  ·  "
        f"{(peri.rho < 0).mean():.0%} negative",
        fontsize=7.5,
        pad=8,
    )
    ax.legend(frameon=False, loc="center right", handletextpad=0.3, borderpad=0.1)
    panel(ax, "A")

    # -- B: raw vs partial, both specifications ----------------------------
    specs = [
        ("", "pre-registered\n(gradient 1 + myelin)"),
        ("_extended", "extended\n(gradients 1-3 + myelin)"),
    ]
    for i, (tag, lab) in enumerate(specs):
        p5 = pd.read_csv(R / f"p5_hierarchy_schaefer200x7{tag}.csv")
        g = p5[
            (p5.step == "geneset_partial")
            & (p5.reference == "pericyte_mural")
            & (p5.target == "baseline_oef")
        ]
        x = [i - 0.17, i + 0.17]
        for _, r in g.iterrows():
            bx.plot(
                x, [r.rho_before, r.rho], color=OI["grey"], lw=0.5, alpha=0.55, zorder=2
            )
        bx.scatter([x[0]] * len(g), g.rho_before, s=16, color=OI["grey"], zorder=3)
        bx.scatter(
            [x[1]] * len(g),
            g.rho,
            s=16,
            color=OI["blue"] if not tag else OI["orange"],
            zorder=4,
        )
        frac = (g.p_spin < 0.05).mean()
        bx.annotate(
            f"{frac:.0%} sig",
            (x[1], g.rho.median()),
            xytext=(9, -2),
            textcoords="offset points",
            fontsize=6.5,
            color=OI["blue"] if not tag else OI["orange"],
        )
        _ = lab
    bx.axhline(0, color=OI["black"], lw=0.7)
    bx.set_xticks([-0.17, 0.17, 0.83, 1.17])
    bx.set_xticklabels(["raw", "partial", "raw", "partial"])
    bx.set_xlim(-0.45, 1.45)
    bx.set_ylabel("Spearman ρ")
    bx.set_title("hierarchy control, both specifications", fontsize=8, pad=22)
    for i, (_, lab) in enumerate(specs):
        bx.annotate(
            lab,
            (i, 1.012),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    panel(bx, "B")

    fig.subplots_adjust(wspace=0.30, top=0.80)
    return save(fig, out, "MF2_primary_effect")


def figure3(out: Path) -> list[str]:
    """Mediation as a bound: the undetectable region, shaded."""
    dyn = pd.read_csv(R / "p0_dynamic_range_schaefer200x7.csv").set_index("name")
    p6 = pd.read_csv(R / "p6_mediation_summary.csv")
    rel_oef = float(dyn.loc["baseline OEF", "split_half_reliability"])
    pairs = [
        ("discordance_extraction", "discordance (extraction)", "extraction"),
        ("discordance_overshoot", "discordance (overshoot)", "overshoot"),
        ("coupling_angle", "coupling angle", "coupling angle"),
    ]
    fig, ax = plt.subplots(figsize=(W1 * 1.7, 62 * MM))

    xs, floors, trues, labels = [], [], [], []
    for i, (key, relname, lab) in enumerate(pairs):
        rel_y = float(dyn.loc[relname, "split_half_reliability"])
        ceiling = math.sqrt(rel_oef * rel_y)
        floor = float(dyn.loc[relname, "detectable_true_rho"])
        row = p6[
            (p6.gene_set == "pericyte_mural")
            & (p6.mediator == "baseline_oef")
            & (p6.outcome == key)
            & (p6.adjusted)
        ].iloc[0]
        xs.append(i)
        floors.append(floor)
        trues.append(abs(float(row.b_median)) / ceiling)
        labels.append(lab)

    ax.bar(
        xs,
        floors,
        width=0.62,
        color=OI["grey"],
        alpha=0.28,
        label="undetectable region (below floor)",
        zorder=1,
    )
    ax.scatter(
        xs, trues, s=46, color=OI["blue"], zorder=4, label="implied true |ρ| of path b"
    )
    for x, t, f in zip(xs, trues, floors):
        ax.annotate(
            f"{t:.3f}",
            (x, t),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=6.5,
            color=OI["blue"],
        )
        ax.annotate(
            f"floor {f:.3f}",
            (x, f),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=6,
            color="#555555",
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("|ρ|,  baseline OEF → outcome")
    ax.set_ylim(0, max(floors) * 1.35)
    ax.set_title(
        "every mediator→outcome estimate falls below what the design resolves",
        fontsize=8,
        pad=6,
    )
    ax.legend(frameon=False, loc="upper right", handletextpad=0.4)
    return save(fig, out, "MF3_mediation_bound")


def figure4(out: Path) -> list[str]:
    """Positive controls, including the one that fails."""
    c = pd.read_csv(R / "p5_positive_controls_schaefer200x7.csv")
    c = c.assign(
        short=lambda d: (
            d.a.str.split(":").str[-1]
            + " ↔ "
            + d.b.str.replace("our_reconstruction:|authors_published:", "", regex=True)
        ),
        expect_pos=lambda d: d.expected.str.startswith("positive"),
    )
    order = c.drop_duplicates("short").short.tolist()
    fig, ax = plt.subplots(figsize=(W2 * 0.72, 66 * MM))

    y = np.arange(len(order))
    for src, off, colour, mark in (
        ("our_reconstruction", -0.16, OI["blue"], "o"),
        ("authors_published", 0.16, OI["orange"], "s"),
    ):
        g = c[c.source == src].set_index("short").reindex(order)
        ax.scatter(
            g.rho,
            y + off,
            s=30,
            color=colour,
            marker=mark,
            zorder=4,
            label=src.replace("_", " "),
        )
    ax.axvline(0, color=OI["black"], lw=0.7)
    for i, s in enumerate(order):
        row = c[c.short == s].iloc[0]
        ok = c[c.short == s]
        failed = any((r.rho > 0) != r.expect_pos for _, r in ok.iterrows())
        ax.axhspan(
            i - 0.42,
            i + 0.42,
            color=OI["vermillion"] if failed else OI["green"],
            alpha=0.07,
            lw=0,
            zorder=1,
        )
        ax.annotate(
            "expected " + ("+" if row.expect_pos else "−"),
            (1.02, i),
            xycoords=("axes fraction", "data"),
            fontsize=6,
            va="center",
            color="#555555",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("Spearman ρ")
    ax.set_xlim(-0.55, 0.95)
    ax.set_title("positive controls, both map sources", fontsize=8, pad=6)
    ax.legend(frameon=False, loc="upper left", handletextpad=0.3, borderpad=0.2)
    ax.invert_yaxis()
    return save(fig, out, "MF4_positive_controls")


def figure5(out: Path) -> list[str]:
    """The whole mqBOLD chain against the dropout proxy."""
    d = pd.read_csv(R / "p0b_full_dropout_audit_schaefer200x7.csv")
    d = d.assign(order=d.link.str.split("_").str[0].astype(int)).sort_values("order")
    fig, ax = plt.subplots(figsize=(W2 * 0.78, 62 * MM))
    x = np.arange(len(d))
    colours = [OI["vermillion"] if b else OI["sky"] for b in d.breaches_gate]
    ax.bar(x, d.abs_rho, width=0.62, color=colours, zorder=3)
    ax.axhline(0.5, color=OI["black"], lw=0.9, ls="--", zorder=4)
    ax.annotate(
        "§9 gate,  |ρ| = 0.5",
        (len(d) - 0.4, 0.5),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        fontsize=6.5,
    )
    for xi, (_, r) in zip(x, d.iterrows()):
        ax.annotate(
            f"{r.rho_vs_dropout:+.3f}",
            (xi, r.abs_rho),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=5.6,
            rotation=90,
            va="bottom",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [ln.split("_", 1)[1].replace("_", " ") for ln in d.link],
        rotation=38,
        ha="right",
    )
    ax.set_ylim(0, 0.62)
    ax.set_ylabel("|ρ| vs scanner-dropout proxy")
    ax.set_title(
        "every link of the mqBOLD chain, not only the final map", fontsize=8, pad=6
    )
    ax.legend(
        handles=[Line2D([], [], color=OI["sky"], lw=5, label="within gate")],
        frameon=False,
        loc="upper left",
    )
    return save(fig, out, "MF5_dropout_chain")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.logging.level, format=cfg.logging.format)
    plt.rcParams.update(STYLE)
    out = Path(args.out)

    written: list[str] = []
    for fn in (figure1, figure2, figure3, figure4, figure5):
        written += fn(out)

    if args.out == "results/figures":
        with manifest("manuscript_figures", cfg) as man:
            man.record(outputs=written, n_figures=5, width_mm=180, palette="Okabe-Ito")
            man.note(
                "Five display items carrying the manuscript's argument. Figure 1 "
                "is the resolvability analysis, which the previous figure set did "
                "not depict at all."
            )

    print(f"\n{'=' * 62}\nMANUSCRIPT FIGURES\n{'=' * 62}")
    for p in written:
        if p.endswith(".png"):
            print(f"  {p}")
    print(f"{'=' * 62}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
