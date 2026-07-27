"""Figure palette — validated, not eyeballed.

Every colour here comes from a palette that was run through a CVD/contrast
validator rather than chosen by taste. The rules that shaped it:

* **Categorical hues are assigned in fixed order, never cycled.** Colour follows
  the entity, so a figure that drops a series must not repaint the survivors.
* **At most three categorical hues in scatter-like forms.** The three-slot set
  clears the all-pairs colour-vision-deficiency floor (worst ΔE 9.2 deutan,
  24.0 normal); a fourth puts yellow beside orange and fails it. Seven Yeo
  networks in one scatter is therefore not an option — highlight two and grey
  the rest, or facet.
* **Sequential is one hue, light→dark. Diverging is two hues with a neutral
  grey midpoint.** Never a rainbow, and never a hue at the diverging midpoint —
  the middle has to read as "nothing".
* ``AQUA`` sits at 2.74:1 against the surface, below the 3:1 bar, so anything
  drawn in it carries a visible direct label rather than relying on colour.

Figures are light-mode only. That is a deliberate single-look commitment: these
are publication figures, and a dark variant would need its own validated steps
rather than an inverted flip.
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

__all__ = [
    "AQUA",
    "BLUE",
    "CATEGORICAL",
    "DIVERGING",
    "GREY",
    "INK",
    "INK_MUTED",
    "ORANGE",
    "SEQUENTIAL",
    "SEQUENTIAL_WARM",
    "SURFACE",
    "diverging_norm",
]

# --- categorical: fixed order, first three only for scatter-like forms -------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
CATEGORICAL = (BLUE, ORANGE, AQUA)

# --- ink and surface --------------------------------------------------------
# Text wears text tokens, never the series colour.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GREY = "#b8b7b2"  # de-emphasised marks

# --- sequential: one hue, light -> dark ------------------------------------
_BLUE_RAMP = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
SEQUENTIAL = LinearSegmentedColormap.from_list("seq_blue", _BLUE_RAMP)

# Second sequential context takes the next categorical hue, as its own ramp.
_ORANGE_RAMP = ["#fbe0d3", "#f6bfa4", "#f09a71", "#eb6834", "#c14f24", "#8f3a1a"]
SEQUENTIAL_WARM = LinearSegmentedColormap.from_list("seq_orange", _ORANGE_RAMP)

# --- diverging: two poles, neutral grey midpoint ---------------------------
# Blue<->red. The midpoint is grey, not a hue, so "no difference" reads as
# absence rather than as a third category.
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red",
    ["#104281", "#2a78d6", "#86b6ef", "#f0efec", "#f0a3a2", "#e34948", "#a32725"],
)


def diverging_norm(vmin: float, vcenter: float, vmax: float) -> TwoSlopeNorm:
    """Diverging normalisation anchored on a meaningful midpoint.

    The midpoint must be the value that means "nothing" — for the coupling
    angle that is π/4, where ΔCBF equals ΔCMRO₂ and BOLD is null. Centring on
    the data mean instead would invent a boundary.
    """
    if not vmin < vcenter < vmax:
        raise ValueError(f"need vmin < vcenter < vmax, got {vmin}, {vcenter}, {vmax}")
    return TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)


def apply_style() -> None:
    """Recessive grid and axes, thin marks, text in ink tokens."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": "#d8d7d2",
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK,
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#e8e7e3",
            "grid.linewidth": 0.7,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "font.size": 9.5,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.0,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )
