"""Figure builders — plotting only, no analysis (CLAUDE.md §5).

Every function here takes already-computed values and draws them. Nothing in
this module derives a statistic; if a number appears in a figure it was
computed in ``src/`` and passed in. That keeps the figures reproducible from
the manifested pipeline rather than from ad-hoc notebook state.

Colour decisions live in :mod:`src.viz.palette` and were validated rather than
chosen by eye — see that module's docstring for the constraints.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from .palette import (
    AQUA,
    BLUE,
    DIVERGING,
    GREY,
    INK,
    INK_MUTED,
    ORANGE,
    SEQUENTIAL,
)

logger = logging.getLogger(__name__)

__all__ = [
    "fig_ahba_coverage",
    "fig_correlation_matrix",
    "fig_coupling_plane",
    "fig_mqbold_vs_pet",
    "fig_network_modes",
    "fig_reliability",
    "fig_spin_null",
    "fig_surface_panel",
]

# Yeo network display order: sensory -> association, so the x-axis carries the
# hierarchy the analysis is trying to rule out.
NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
NETWORK_LABEL = {
    "Vis": "Visual",
    "SomMot": "Somatomotor",
    "DorsAttn": "Dorsal attention",
    "SalVentAttn": "Salience/vent. att.",
    "Limbic": "Limbic",
    "Cont": "Control",
    "Default": "Default mode",
}


def _save(fig: plt.Figure, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


# ---------------------------------------------------------------------------
# F1 — the maps themselves
# ---------------------------------------------------------------------------
def fig_surface_panel(
    maps: dict[str, np.ndarray],
    labels: np.ndarray,
    out: Path,
    name: str = "F1_surface_panel",
) -> Path:
    """Parcel maps painted back onto the cortical surface.

    The first question about any parcel-level result is whether it looks like
    structured biology or like noise, and no correlation coefficient answers
    that. Each map is projected from parcels back to vertices and rendered on
    the inflated left hemisphere, lateral and medial.

    Parameters
    ----------
    maps : dict
        ``{title: parcel_vector}``. Vectors are 1-based parcel order.
    labels : ndarray
        Per-vertex parcel labels for the same parcellation.
    out : Path
        Output directory.
    """
    import tempfile

    from neuromaps.datasets import fetch_fsaverage
    from surfplot import Plot

    surfaces = fetch_fsaverage(density="10k")
    lh_inflated = str(surfaces["inflated"][0])
    lh_sulc = str(surfaces["sulc"][0])

    n = len(maps)
    fig, axes = plt.subplots(n, 1, figsize=(7.6, 1.62 * n))
    fig.subplots_adjust(hspace=0.02)
    axes = np.atleast_1d(axes)

    with tempfile.TemporaryDirectory() as tmp:
        for k, (ax, (title, vec)) in enumerate(zip(axes, maps.items())):
            vertex = np.full(labels.shape, np.nan)
            for i, v in enumerate(vec, start=1):
                vertex[labels == i] = v
            finite = vertex[np.isfinite(vertex)]

            # Coupling angle is the only diverging quantity: it has a real
            # midpoint at pi/4, where dCBF equals dCMRO2 and BOLD is null.
            # The others are magnitudes and take a single-hue ramp.
            diverging = title.lower().startswith("coupling")
            cmap = DIVERGING if diverging else SEQUENTIAL
            if diverging:
                # Symmetric about the midpoint, but clipped to the 2nd/98th
                # percentile of |deviation|. Scaling to the extremes lets a
                # couple of outlier parcels wash the whole map to the midpoint
                # colour, which reads as "no signal anywhere".
                centre = float(np.pi / 4)
                half = float(np.nanpercentile(np.abs(finite - centre), 98))
                crange = (centre - half, centre + half)
            else:
                crange = tuple(np.nanpercentile(finite, [2, 98]))

            p = Plot(
                surf_lh=lh_inflated,
                views=["lateral", "medial"],
                size=(1000, 320),
                zoom=1.2,
                brightness=0.75,
            )
            p.add_layer({"left": lh_sulc}, cmap="Greys", cbar=False)
            p.add_layer(
                {"left": vertex},
                cmap=cmap,
                color_range=crange,
                cbar=True,
                cbar_label=None,
            )
            sub = p.build(
                colorbar=True,
                cbar_kws={
                    "location": "right",
                    "n_ticks": 3,
                    "decimals": 2,
                    "fontsize": 9,
                    "shrink": 0.5,
                    "aspect": 12,
                },
            )
            png = f"{tmp}/panel{k}.png"
            sub.savefig(png, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
            plt.close(sub)

            ax.imshow(plt.imread(png))
            ax.axis("off")
            # The midpoint note belongs in the title rather than as free text:
            # placed anywhere in the axes it collided with the neighbouring panel.
            suffix = "   ·   midpoint π/4 (n = 1), blue = discordant" if diverging else ""
            ax.set_title(title + suffix, loc="left", pad=4, fontsize=10.5)

    fig.suptitle(
        "Left hemisphere, inflated · lateral and medial views",
        x=0.01,
        ha="left",
        fontsize=9,
        color=INK_MUTED,
        y=1.005,
    )
    return _save(fig, out, name)


# ---------------------------------------------------------------------------
# F2 — the two discordance modes by network
# ---------------------------------------------------------------------------
def fig_network_modes(
    df: pd.DataFrame, out: Path, name: str = "F2_network_modes"
) -> Path:
    """Extraction vs overshoot discordance per Yeo network.

    These two modes were collapsed into a single column until recently, which
    hid the fact that they peak in different networks. Two series, so two
    categorical slots and a legend; values are direct-labelled so identity
    never rests on colour alone.
    """
    d = df[df.parcellation == "schaefer200x7"].copy()
    d["net"] = d.parcel_name.str.extract(r"LH_(\w+?)_")[0]
    g = (
        d.groupby("net")[["discordance_risk_extraction", "discordance_risk_overshoot"]]
        .mean()
        .sort_values("discordance_risk_extraction")
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    y = np.arange(len(g))
    h = 0.36
    # 2px surface gap between adjacent fills.
    ax.barh(
        y + h / 2 + 0.01,
        g.discordance_risk_extraction,
        height=h,
        color=BLUE,
        label="Extraction  (CMRO₂ ↑, BOLD ↓)",
    )
    ax.barh(
        y - h / 2 - 0.01,
        g.discordance_risk_overshoot,
        height=h,
        color=ORANGE,
        label="Overshoot  (CMRO₂ ↓, BOLD ↑)",
    )

    for i, (e, o) in enumerate(
        zip(g.discordance_risk_extraction, g.discordance_risk_overshoot)
    ):
        ax.text(
            e + 0.006,
            i + h / 2 + 0.01,
            f"{e:.2f}",
            va="center",
            fontsize=8,
            color=INK_MUTED,
        )
        ax.text(
            o + 0.006,
            i - h / 2 - 0.01,
            f"{o:.2f}",
            va="center",
            fontsize=8,
            color=INK_MUTED,
        )

    ax.set_yticks(y, [NETWORK_LABEL[n] for n in g.index])
    ax.set_xlabel("fraction of subjects")
    ax.set_xlim(0, max(g.max()) * 1.18)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_title(
        "Discordance splits into two modes that peak in different networks",
        loc="left",
        pad=26,
    )
    # Legend above the plot area: inside it, it collided with the longest bar.
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncols=2)
    fig.text(
        0.01,
        -0.10,
        "Extraction is the mode a capillary-density hypothesis predicts. It peaks in somatomotor "
        "cortex,\nnot the default mode network — which instead leads on overshoot.",
        fontsize=8,
        color=INK_MUTED,
        ha="left",
    )
    return _save(fig, out, name)


# ---------------------------------------------------------------------------
# F3 — the coupling plane
# ---------------------------------------------------------------------------
def fig_coupling_plane(
    d_cbf: np.ndarray,
    d_cmro2: np.ndarray,
    networks: pd.Series,
    out: Path,
    name: str = "F3_coupling_plane",
) -> Path:
    """Every parcel in the (ΔCMRO₂, ΔCBF) plane, with the BOLD null line.

    This is the figure that makes "discordant" mean something concrete: the
    diagonal is where ΔCBF equals ΔCMRO₂ and BOLD is null, so everything below
    it has BOLD opposing CMRO₂.

    Only two networks are coloured. Seven categorical hues cannot clear the
    all-pairs colour-vision floor, and the story is carried by the two extremes
    anyway — so the rest are grey context rather than competing colours.
    """
    mc = np.nanmedian(d_cbf, axis=0)
    mm = np.nanmedian(d_cmro2, axis=0)

    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    lim = float(np.nanmax(np.abs(np.concatenate([mc, mm])))) * 1.12

    # The discordant region is NOT a single half-plane. Discordance is
    # sign(dCBF - dCMRO2) != sign(dCMRO2), which resolves differently on each
    # side of the vertical axis:
    #   dCMRO2 > 0 : discordant where dCBF < dCMRO2  (below the diagonal)
    #   dCMRO2 < 0 : discordant where dCBF > dCMRO2  (ABOVE the diagonal)
    # so it is a bowtie pivoting at the origin, not the lower half.
    ax.fill_between([0, lim], [0, lim], [-lim, -lim], color=BLUE, alpha=0.05, lw=0)
    ax.fill_between([-lim, 0], [lim, lim], [-lim, 0], color=BLUE, alpha=0.05, lw=0)
    ax.plot([-lim, lim], [-lim, lim], color=INK_MUTED, lw=1.2, ls="--", zorder=2)
    ax.axhline(0, color="#d8d7d2", lw=0.8, zorder=1)
    ax.axvline(0, color="#d8d7d2", lw=0.8, zorder=1)

    other = ~networks.isin(["SomMot", "Default"]).to_numpy()
    ax.scatter(mm[other], mc[other], s=26, color=GREY, alpha=0.75, lw=0, zorder=3)
    for net, colour in (("SomMot", BLUE), ("Default", ORANGE)):
        sel = (networks == net).to_numpy()
        # 2px surface ring on overlapping marks.
        ax.scatter(
            mm[sel],
            mc[sel],
            s=52,
            color=colour,
            lw=1.4,
            edgecolor="#fcfcfb",
            zorder=4,
            label=NETWORK_LABEL[net],
        )

    ax.text(
        lim * 0.95,
        -lim * 0.90,
        "discordant\n(n < 1)",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=INK_MUTED,
    )
    ax.text(
        -lim * 0.95,
        lim * 0.90,
        "discordant\n(n < 1)",
        ha="left",
        va="top",
        fontsize=8.5,
        color=INK_MUTED,
    )
    ax.text(
        lim * 0.95,
        lim * 0.90,
        "concordant",
        ha="right",
        va="top",
        fontsize=8.5,
        color=INK_MUTED,
    )
    ax.text(
        -lim * 0.95,
        -lim * 0.90,
        "concordant",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=INK_MUTED,
    )

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Δ CMRO₂  (%, calc vs control)")
    ax.set_ylabel("Δ CBF  (%, calc vs control)")
    ax.set_title(
        "Where each parcel sits relative to the BOLD null line", loc="left", pad=26
    )
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncols=2)
    ax.grid(True)
    ax.set_axisbelow(True)
    return _save(fig, out, name)


# ---------------------------------------------------------------------------
# F4 — what is redundant with what
# ---------------------------------------------------------------------------
def fig_correlation_matrix(
    df: pd.DataFrame, out: Path, name: str = "F4_correlation_matrix"
) -> Path:
    """Spearman correlations among released columns.

    Diverging scale with a grey midpoint: zero has to read as absence, not as a
    third category. Values are printed in every cell, so the colour is a fast
    path rather than the only channel.
    """
    cols = [
        "baseline_oef",
        "baseline_cbf",
        "baseline_cmro2",
        "coupling_n_angle",
        "discordance_risk_extraction",
        "discordance_risk_overshoot",
        "dropout_snr_coverage",
        "venous_partial_volume",
        "ahba_n_samples",
    ]
    short = [
        "OEF",
        "CBF",
        "CMRO₂",
        "coupling n",
        "extraction",
        "overshoot",
        "SNR coverage",
        "venous PV",
        "AHBA n",
    ]
    d = df[df.parcellation == "schaefer200x7"][cols]
    c = d.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(c.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(short)), short, rotation=45, ha="right")
    ax.set_yticks(range(len(short)), short)
    for i in range(len(short)):
        for j in range(len(short)):
            v = c.iat[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="#ffffff" if abs(v) > 0.55 else INK,
            )
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, length=2, colors=INK_MUTED)
    cb.set_label("Spearman ρ", fontsize=8.5, color=INK_MUTED)
    ax.set_title("Released columns against each other", loc="left")
    fig.text(
        0.01,
        -0.09,
        "Correlations only — no spatial null applied here, so these are descriptive, "
        "not inferential.",
        fontsize=8,
        color=INK_MUTED,
        ha="left",
    )
    return _save(fig, out, name)


# ---------------------------------------------------------------------------
# F5 — reliability
# ---------------------------------------------------------------------------
def fig_reliability(
    splits: dict[str, np.ndarray], gate: float, out: Path, name: str = "F5_reliability"
) -> Path:
    """Split-half reliability distributions, with the gate drawn.

    Three series, which clears the all-pairs floor, each direct-labelled.
    """
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    colours = (BLUE, ORANGE, AQUA)
    for (label, vals), colour in zip(splits.items(), colours):
        v = vals[np.isfinite(vals)]
        ax.hist(v, bins=45, histtype="step", lw=2.0, color=colour, label=label)
        med = float(np.median(v))
        ax.axvline(med, color=colour, lw=1.0, ls=":", alpha=0.9)
        ax.text(
            med,
            ax.get_ylim()[1] * 0.94,
            f" {label}  {med:.3f}",
            fontsize=8,
            color=colour,
            rotation=90,
            va="top",
        )

    ax.axvline(gate, color=INK, lw=1.4)
    ax.text(
        gate - 0.008,
        ax.get_ylim()[1] * 0.5,
        f"gate {gate:g} ",
        rotation=90,
        ha="right",
        va="center",
        fontsize=8.5,
        color=INK,
    )
    ax.set_xlabel("Spearman-Brown corrected split-half reliability")
    ax.set_ylabel("splits")
    ax.set_title("Coupling-map reliability, 1000 random split-halves", loc="left")
    ax.grid(True)
    ax.set_axisbelow(True)
    return _save(fig, out, name)


# ---------------------------------------------------------------------------
# F6 — why the spatial null matters
# ---------------------------------------------------------------------------
def fig_spin_null(
    null_rhos: np.ndarray,
    observed: float,
    p_spin: float,
    p_naive: float,
    title: str,
    out: Path,
    name: str = "F6_spin_null",
) -> Path:
    """The spin-test null distribution against the observed correlation.

    Two smooth brain maps correlate substantially by chance; this shows how
    much. The naive p-value is the one you get by pretending parcels are
    independent, and the gap between the two is the entire reason R1 exists.
    """
    r = null_rhos[np.isfinite(null_rhos)]
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.hist(r, bins=70, color=BLUE, alpha=0.30, lw=0)
    ax.hist(r, bins=70, histtype="step", lw=1.6, color=BLUE)

    thr = float(np.percentile(np.abs(r), 95))
    for s in (-1, 1):
        ax.axvline(s * thr, color=INK_MUTED, lw=1.0, ls="--")
    ax.text(
        thr,
        ax.get_ylim()[1] * 0.97,
        f"  |ρ| ≥ {thr:.2f} needed for p < 0.05",
        fontsize=8.5,
        color=INK_MUTED,
        va="top",
    )

    ax.axvline(observed, color=ORANGE, lw=2.4)
    ax.text(
        observed,
        ax.get_ylim()[1] * 0.62,
        f"  observed ρ = {observed:+.3f}",
        fontsize=9,
        color=ORANGE,
        va="top",
        fontweight="semibold",
    )

    ax.set_xlabel("Spearman ρ against the rotated (null) maps")
    ax.set_ylabel("rotations")
    ax.set_title(title, loc="left")
    fig.text(
        0.01,
        -0.16,
        f"spin p = {p_spin:.3f}   ·   naive p = {p_naive:.2g}   — the naive test treats "
        "parcels as independent,\nwhich they are not; the null here is what smooth maps do "
        "by chance.",
        fontsize=8,
        color=INK_MUTED,
        ha="left",
    )
    ax.grid(True)
    ax.set_axisbelow(True)
    return _save(fig, out, name)


# ---------------------------------------------------------------------------
# F7 — the disagreement with PET
# ---------------------------------------------------------------------------
def fig_mqbold_vs_pet(
    pairs: dict[str, tuple[np.ndarray, np.ndarray, float]],
    out: Path,
    name: str = "F7_mqbold_vs_pet",
) -> Path:
    """mqBOLD maps against the PET maps of the same physiology.

    Small multiples rather than one panel with several colours: the comparison
    is within each pair, not across them.
    """
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.5 * len(pairs), 3.5))
    axes = np.atleast_1d(axes)
    for ax, (label, (x, y, rho)) in zip(axes, pairs.items()):
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=30, color=BLUE, alpha=0.65, lw=0)
        if ok.sum() > 2:
            b = np.polyfit(x[ok], y[ok], 1)
            xs = np.linspace(np.nanmin(x[ok]), np.nanmax(x[ok]), 50)
            ax.plot(xs, np.polyval(b, xs), color=ORANGE, lw=2.0)
        ax.set_title(f"{label}   ρ = {rho:+.2f}", loc="left", fontsize=10)
        ax.set_xlabel("mqBOLD (this dataset)")
        ax.set_ylabel("PET reference")
        ax.grid(True)
        ax.set_axisbelow(True)
    fig.suptitle(
        "The same physiology, measured two ways",
        x=0.005,
        ha="left",
        fontsize=11,
        fontweight="semibold",
        y=1.04,
    )
    fig.text(
        0.005,
        -0.17,
        "Parcel-level agreement between MRI-derived and PET-derived maps. Weak agreement is "
        "either a real\nmethodological divergence or a sign the MRI maps are noisier than "
        "their reliability implies.",
        fontsize=8,
        color=INK_MUTED,
        ha="left",
    )
    return _save(fig, out, name)


# ---------------------------------------------------------------------------
# F8 — where the gene analysis can actually run
# ---------------------------------------------------------------------------
def fig_ahba_coverage(
    df: pd.DataFrame, out: Path, name: str = "F8_ahba_coverage"
) -> Path:
    """AHBA sample counts per parcel, per parcellation.

    Directly gates Phase 3: a parcel with no donor samples cannot contribute to
    any expression analysis, whatever its imaging values look like.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2), sharey=False)
    order = ["dk68", "schaefer200x7", "schaefer400x7"]
    nice = {
        "dk68": "DK-68",
        "schaefer200x7": "Schaefer-200",
        "schaefer400x7": "Schaefer-400",
    }

    for ax, parc in zip(axes, order):
        g = df[df.parcellation == parc]["ahba_n_samples"].dropna()
        zero = int((g == 0).sum())
        ax.hist(g, bins=np.arange(0, g.max() + 3) - 0.5, color=BLUE, alpha=0.85, lw=0)
        if zero:
            ax.axvspan(-0.5, 0.5, color=ORANGE, alpha=0.35, lw=0)
            ax.text(
                0.6,
                ax.get_ylim()[1] * 0.9,
                f"{zero} parcels\nwith no samples",
                fontsize=8.5,
                color=ORANGE,
                va="top",
                fontweight="semibold",
            )
        ax.set_title(f"{nice[parc]}  ·  median {g.median():.0f}", loc="left", fontsize=10)
        ax.set_xlabel("AHBA samples in parcel")
        ax.set_ylabel("parcels")
        ax.grid(True)
        ax.set_axisbelow(True)

    fig.suptitle(
        "Transcriptomic coverage limits which parcellations are usable",
        x=0.005,
        ha="left",
        fontsize=11,
        fontweight="semibold",
        y=1.05,
    )
    fig.text(
        0.005,
        -0.21,
        "Computed from 5 of 6 donors — 15496 is unavailable upstream. Parcels with zero "
        "samples cannot\ncontribute to Phase 3 regardless of their imaging values.",
        fontsize=8,
        color=INK_MUTED,
        ha="left",
    )
    return _save(fig, out, name)


def legend_handles(labels_colours: list[tuple[str, str]]) -> list[Line2D]:
    """Proxy handles so a legend can be built without re-plotting."""
    return [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=c, markersize=7, label=lbl
        )
        for lbl, c in labels_colours
    ]
