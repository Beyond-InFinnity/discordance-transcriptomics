"""Cortical oxygen atlas — where the BOLD signal can be trusted.

The released annotation table, on a brain you can turn.

Two questions this answers:

1. *I care about region X. Is the signal trustworthy there, and why?*
2. *I have a statistical map. Does it land in territory where BOLD and oxygen
   metabolism disagree?*

**Design decision that carries the science.** Every parcel is drawn with opacity
proportional to how much it should be trusted — scanner-dropout coverage, the
split-half reliability of *the column being displayed*, and whether any donor
tissue was sampled inside it. Regions the data cannot support fade into the
background rather than presenting a confident colour. The project's central
finding is that most of what was measured could not be resolved; a viewer that
hid that behind uniform saturation would be lying by omission. The fade is
switchable, and the legend says exactly what it encodes.

The reliability term follows the metric because it is a property of the map, not
of the parcel. An earlier version read ``map_reliability_coupling`` off the row —
a single constant, 0.7105, the coupling-angle map's reliability — and applied it
to whatever was on screen. That flattered ``discordance_risk`` (0.49, flagged
``low_reliability`` in the schema) and penalised ``baseline_oef`` (0.98): wrong in
both directions, for the two columns that matter most. It also capped every
parcel at 0.887 so none could read as fully supported, and was the binding term
for 29 of 100 parcels — a third of the map rated by a number with no per-parcel
content.

Colour is derived from what is actually measured: the ratio of oxygenated to
deoxygenated haemoglobin. Arterial scarlet through fixed-tissue neutral to venous
indigo.

**Why the brain is hovered, not clicked.** Plotly's WebGL 3D scenes do not emit
``plotly_click`` at all — verified against plotly.js 3.6.0 as bundled by Streamlit
1.59.1, on a fully exposed marker, for a plain click, a zero-movement down/up, a
double click, and with ``scene.dragmode`` disabled. ``plotly_hover`` fires
correctly and carries ``customdata``. Since ``st.plotly_chart(on_select=...)``
listens for click and lasso events, click-to-select on a surface is unreachable
from this stack, and an earlier version of this file advertised it anyway: the
caption said "click a region" while the anchors it relied on sat *inside* the
mesh, depth-occluded and inert. Selection lives in the Region control, which
always worked; the surface reads out under the cursor.

Nothing here computes a result. The app reads artifacts built by the pipeline and
uses ``src.data.parcellate`` for parcellation geometry only (R4 forbids
hand-rolled coordinate work). Analysis belongs in ``src/``.

Usage
-----
    streamlit run app/main.py
"""

# Prose and axis labels are read by humans, not parsed: the true minus sign
# and Greek letters are correct typography in an interface.
# ruff: noqa: RUF001
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ANNOT_DIR = ROOT / "data" / "derived" / "annotation"
PRIMARY = "schaefer200x7"  # the only parcellation with surface geometry wired

st.set_page_config(
    page_title="Cortical oxygen atlas",
    page_icon="🩸",
    layout="wide",
)

# --- palette: oximetry, because that is literally what is being measured -----
OXY = "#C41E3A"  # arterial scarlet — oxygenated haemoglobin
DEOXY = "#3B2E6E"  # venous indigo — deoxygenated haemoglobin
TISSUE = "#C9C2B6"  # fixed tissue, the neutral midpoint
FIELD = "#E9EDF2"  # cool pale slate — the background the fade dissolves into
INK = "#16191F"
INSTRUMENT = "#0F766E"  # interface only; never encodes data

# Millimetres to lift a hover anchor off the surface, toward its camera. Large
# enough to clear the mesh at every parcel, small enough that the readout still
# lands on the tissue it describes.
ANCHOR_LIFT = 2.5

# Reliability ramp: below RELIABILITY_FLOOR a map carries no usable regional
# signal, at RELIABILITY_FULL it is taken at face value. The floor sits just
# under the project's own 0.5 gate (CLAUDE.md §9, Phase 0a), so a column the
# schema flags `low_reliability` lands below the "unresolved" band rather than
# being quietly rounded up into usability.
RELIABILITY_FLOOR = 0.40
RELIABILITY_FULL = 0.75

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;600&family=IBM+Plex+Mono:wght@400;600&display=swap');
    :root {{
      --oxy:{OXY}; --deoxy:{DEOXY}; --tissue:{TISSUE};
      --field:{FIELD}; --ink:{INK}; --instrument:{INSTRUMENT};
    }}
    .stApp {{ background:{FIELD}; }}
    html, body, [class*="css"] {{
      font-family:'Inter',-apple-system,'Segoe UI',sans-serif; color:{INK};
    }}
    h1,h2,h3 {{ font-family:'Space Grotesk','Inter',sans-serif; letter-spacing:-.02em; }}
    .eyebrow {{
      font-family:'IBM Plex Mono',ui-monospace,monospace; font-size:.68rem;
      letter-spacing:.18em; text-transform:uppercase; color:{INSTRUMENT};
    }}
    .lede {{ font-size:1.02rem; line-height:1.5; max-width:60ch; opacity:.85; }}
    .metric-big {{
      font-family:'Space Grotesk',sans-serif; font-size:2.6rem; font-weight:700;
      line-height:1; letter-spacing:-.03em;
    }}
    .metric-label {{
      font-family:'IBM Plex Mono',monospace; font-size:.66rem; letter-spacing:.12em;
      text-transform:uppercase; opacity:.6;
    }}
    .chip {{
      font-family:'IBM Plex Mono',monospace; font-size:.66rem; padding:.16rem .5rem;
      border-radius:2px; letter-spacing:.06em;
    }}
    .rule {{ border:0; border-top:1px solid rgba(22,25,31,.13); margin:1.1rem 0; }}
    [data-testid="stSidebar"] {{ background:#DFE5EC; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ================================================================ data ======
@st.cache_data(show_spinner=False)
def load_tables() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    df = pd.read_csv(ANNOT_DIR / "discordance_annotation.csv")
    with (ANNOT_DIR / "discordance_annotation.schema.json").open() as fh:
        schema = json.load(fh)
    prof = pd.read_csv(ANNOT_DIR / "geneset_profiles.csv")
    return df, schema, prof


@st.cache_resource(show_spinner="Loading cortical surface…")
def load_surface() -> dict:
    """fsaverage5 inflated left hemisphere, with Schaefer parcel labels.

    Vertex count is asserted rather than assumed: a silent mismatch between mesh
    and labels would paint every parcel in the wrong place, and the result would
    look entirely plausible.
    """
    from nilearn import datasets, surface

    from src.data.parcellate import get_parcellation

    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    coords, faces = surface.load_surf_mesh(fs["infl_left"])
    sulc = surface.load_surf_data(fs["sulc_left"])
    labels, _gii, n_parcels = get_parcellation(PRIMARY, "10k", "L")

    assert coords.shape[0] == labels.shape[0] == sulc.shape[0], (
        f"geometry mismatch: {coords.shape[0]} mesh vertices, "
        f"{labels.shape[0]} labels, {sulc.shape[0]} sulcal values"
    )
    centroids = np.full((n_parcels + 1, 3), np.nan)
    for lab in range(1, n_parcels + 1):
        m = labels == lab
        if m.any():
            centroids[lab] = coords[m].mean(axis=0)

    # Hover anchors, one set per camera — and they cannot be the centroids.
    #
    # A centroid is the mean of a parcel's vertices, so it lies *inside* the
    # surface: 99 of 100 parcels have their own tissue between the camera and
    # their centroid. Plotly depth-tests markers against Mesh3d, so anchors put
    # there are occluded and never hover, which is why the brain was inert.
    # Anchor instead to the vertex nearest each camera and lift it clear, so the
    # marker sits just in front of the tissue it labels. A parcel facing away
    # from a camera keeps its anchor behind the mesh and stays correctly
    # unreachable in that view — you should not be able to read out the medial
    # wall from the lateral picture.
    anchors = {}
    for scene, sign in (("lateral", -1.0), ("medial", 1.0)):
        pts = np.full((n_parcels + 1, 3), np.nan)
        for lab in range(1, n_parcels + 1):
            m = labels == lab
            if not m.any():
                continue
            v = coords[m]
            k = np.argmin(v[:, 0]) if sign < 0 else np.argmax(v[:, 0])
            pts[lab] = v[k] + np.array([sign * ANCHOR_LIFT, 0.0, 0.0])
        anchors[scene] = pts

    return {
        "coords": coords,
        "faces": faces,
        "sulc": sulc,
        "labels": labels,
        "n_parcels": n_parcels,
        "centroids": centroids,
        "anchors": anchors,
    }


def metric_reliability(schema: dict, metric: str) -> float | None:
    """Split-half reliability of one column, or None where none was established.

    Read per metric rather than taken from ``map_reliability_coupling``, which is
    a single number — the coupling-angle map's reliability — repeated on every
    row. Using it for whatever happened to be on screen flattered
    ``discordance_risk`` (0.49, flagged low) and penalised ``baseline_oef``
    (0.98), i.e. it was wrong in both directions for the two columns that matter
    most.
    """
    prop = schema.get("items", {}).get("properties", {}).get(metric, {})
    rel = prop.get("split_half_reliability")
    return float(rel) if rel is not None else None


def hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], float)


def diverging(t: np.ndarray) -> np.ndarray:
    """Oximetry scale: 0 → venous indigo, 0.5 → tissue, 1 → arterial scarlet."""
    lo, mid, hi = hex_to_rgb(DEOXY), hex_to_rgb(TISSUE), hex_to_rgb(OXY)
    t = np.clip(t, 0, 1)[:, None]
    return np.where(
        t < 0.5, lo + (mid - lo) * (t / 0.5), mid + (hi - mid) * ((t - 0.5) / 0.5)
    )


def confidence(row: pd.Series, reliability: float | None) -> float:
    """How much of this parcel's colour the data actually supports, in [0, 1].

    Three independent things can undermine a parcel: the scanner lost most of its
    vertices to signal dropout, the map on screen is unreliable, or no donor
    tissue was sampled inside it. The weakest governs, since any one alone is
    disqualifying.

    ``reliability`` belongs to the metric being displayed and is passed in rather
    than read off the row: only the first and third terms are properties of the
    parcel. Where a column has no established split-half reliability the term is
    dropped rather than assumed — an unmeasured map is not a perfect one, and
    pretending otherwise is the error this argument exists to prevent.
    """
    cov = float(row.dropout_snr_coverage)  # fraction of vertices surviving SNR
    ahba = row.get("ahba_n_samples", np.nan)
    terms = [np.clip((cov - 0.30) / 0.45, 0, 1)]  # 0.30 unusable → 0.75 usable
    if reliability is not None:
        terms.append(
            np.clip(
                (reliability - RELIABILITY_FLOOR)
                / (RELIABILITY_FULL - RELIABILITY_FLOOR),
                0,
                1,
            )
        )
    if pd.notna(ahba):
        terms.append(np.clip(float(ahba) / 3.0, 0, 1))  # 0 samples → interpolated
    return float(min(terms))


def table(frame: pd.DataFrame, widths: tuple[str, ...] | None = None) -> None:
    """Render a small table as HTML, deliberately not through ``st.dataframe``.

    ``st.dataframe`` serialises via ``pyarrow.Table.from_pandas``, which
    zero-copies the pandas buffers. Streamlit cancels an in-flight script run the
    moment a new interaction arrives, so a rerun killed inside that call leaves
    Arrow reading freed memory. This app segfaulted three times that way during a
    single review session — ``libarrow.so.2500``, null dereference at ``0x18``,
    the same instruction pointer each time, taking the whole server down.

    Every table here is a handful of rows of text. Rendering them directly costs
    nothing and removes Arrow from the per-rerun path entirely, which is a
    structural fix rather than a narrowed race window. ``pyarrow`` stays in
    requirements.txt — the pipeline uses it for parquet, off the interactive path.
    """
    from html import escape

    cols = list(frame.columns)
    w = widths or ()
    head = "".join(
        f'<th style="text-align:left;padding:.45rem .6rem;font-weight:600;'
        f"font-family:'IBM Plex Mono',monospace;font-size:.62rem;"
        f"letter-spacing:.1em;text-transform:uppercase;opacity:.55;"
        f"border-bottom:1px solid rgba(22,25,31,.13);"
        f'{f"width:{w[i]};" if i < len(w) else ""}">{escape(str(c))}</th>'
        for i, c in enumerate(cols)
    )
    body = "".join(
        "<tr>"
        + "".join(
            f'<td style="padding:.45rem .6rem;vertical-align:top;font-size:.82rem;'
            f'border-bottom:1px solid rgba(22,25,31,.07);">{escape(str(v))}</td>'
            for v in r
        )
        + "</tr>"
        for r in frame.itertuples(index=False)
    )
    st.markdown(
        f'<table style="width:100%;border-collapse:collapse;">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>",
        unsafe_allow_html=True,
    )


# ============================================================== figure ======
METRICS = {
    "discordance_risk": "Discordance risk",
    "discordance_risk_extraction": "Extraction mode",
    "discordance_risk_overshoot": "Overshoot mode",
    "baseline_oef": "Baseline oxygen extraction",
    "baseline_cbf": "Baseline blood flow",
    "dropout_snr_coverage": "Scanner coverage",
}


def brain_figure(
    surf: dict,
    sub: pd.DataFrame,
    metric: str,
    reliability: float | None,
    fade: bool,
    selected: int | None,
    neighbours: list[int],
) -> go.Figure:
    labels = surf["labels"]
    vals = np.full(surf["n_parcels"] + 1, np.nan)
    conf = np.ones(surf["n_parcels"] + 1)
    for _, r in sub.iterrows():
        vals[int(r.parcel_index)] = r[metric]
        conf[int(r.parcel_index)] = confidence(r, reliability)

    finite = vals[np.isfinite(vals)]
    lo, hi = (finite.min(), finite.max()) if finite.size else (0.0, 1.0)
    span = hi - lo if hi > lo else 1.0

    # Sulcal shading gives the surface its anatomy; without it an inflated brain
    # reads as a featureless blob and the reader cannot locate anything.
    s = surf["sulc"]
    shade = 0.78 + 0.22 * (s - s.min()) / (s.ptp() or 1)
    base = np.repeat(hex_to_rgb("#9AA3AE")[None, :], len(labels), axis=0) * shade[:, None]

    v = vals[labels]
    c = conf[labels]
    has = np.isfinite(v) & (labels > 0)
    rgb = base.copy()
    rgb[has] = diverging((v[has] - lo) / span) * (0.86 + 0.14 * shade[has])[:, None]

    if fade:
        # The signature: dissolve unsupported parcels toward the background
        # rather than colouring them as confidently as well-sampled ones.
        a = (0.45 + 0.55 * c[has])[:, None]
        rgb[has] = rgb[has] * a + hex_to_rgb(FIELD) * (1 - a)

    if selected is not None:
        rgb[labels == selected] = hex_to_rgb(INK)

    x, y, z = surf["coords"].T
    i, j, k = surf["faces"].T
    vcol = np.clip(rgb, 0, 255).astype(np.uint8)
    fig = go.Figure()
    for scene in ("scene", "scene2"):
        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                vertexcolor=vcol,
                flatshading=False,
                hoverinfo="skip",
                scene=scene,
                lighting=dict(ambient=0.62, diffuse=0.68, specular=0.06, roughness=0.92),
                showscale=False,
            )
        )
    # Readout anchors, per scene. Invisible markers that carry the tooltip; see
    # load_surface() for why they sit on the surface rather than at the centroid,
    # and the module docstring for why this is hover and not click.
    names = {int(r.parcel_index): r.parcel_name for _, r in sub.iterrows()}
    for scene, key in (("scene", "lateral"), ("scene2", "medial")):
        anc = surf["anchors"][key]
        idx = [int(p) for p in sub.parcel_index if np.isfinite(anc[int(p)][0])]
        fig.add_trace(
            go.Scatter3d(
                x=anc[idx, 0],
                y=anc[idx, 1],
                z=anc[idx, 2],
                mode="markers",
                marker=dict(size=9, color="rgba(0,0,0,0.001)"),
                customdata=idx,
                name="",
                scene=scene,
                showlegend=False,
                hovertemplate=[
                    f"<b>{names[p]}</b><br>{METRICS.get(metric, metric).lower()}: "
                    + ("no data" if not np.isfinite(vals[p]) else f"{vals[p]:.3f}")
                    + f"<br>confidence {conf[p]:.0%}<extra></extra>"
                    for p in idx
                ],
            )
        )

    if selected is not None and neighbours:
        # Arcs to the most molecularly similar parcels. Similarity of expression
        # profile — NOT anatomical or functional connectivity.
        #
        # Drawn in BOTH scenes. Pinning them to the lateral one left arcs from a
        # medially-sited parcel hanging over a surface that does not contain
        # either endpoint.
        cen = surf["centroids"]
        a = cen[selected]
        for scene, sign in (("scene", -1.0), ("scene2", 1.0)):
            for nb in neighbours:
                b = cen[nb]
                # Lift the arc clear of the surface this camera sees. Scaling the
                # midpoint outward from the origin fails for parcels on opposite
                # sides: their midpoint sits near zero, so the "lifted" arc is
                # drawn inside the mesh and is invisible. Push it out along the
                # view axis instead.
                mid = (a + b) / 2
                mid[0] = (
                    (min(a[0], b[0]) - 34.0) if sign < 0 else (max(a[0], b[0]) + 34.0)
                )
                mid[1:] *= 1.08
                t = np.linspace(0, 1, 28)[:, None]
                curve = (1 - t) ** 2 * a + 2 * (1 - t) * t * mid + t**2 * b
                fig.add_trace(
                    go.Scatter3d(
                        x=curve[:, 0],
                        y=curve[:, 1],
                        z=curve[:, 2],
                        mode="lines",
                        line=dict(color=INSTRUMENT, width=3.5),
                        opacity=0.85,
                        hoverinfo="skip",
                        showlegend=False,
                        scene=scene,
                    )
                )
    axoff = dict(visible=False)
    common = dict(
        xaxis=axoff,
        yaxis=axoff,
        zaxis=axoff,
        aspectmode="data",
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        scene=dict(
            **common,
            domain=dict(x=[0.0, 0.5]),
            camera=dict(eye=dict(x=-2.0, y=0, z=0), up=dict(x=0, y=0, z=1)),
        ),
        scene2=dict(
            **common,
            domain=dict(x=[0.5, 1.0]),
            camera=dict(eye=dict(x=2.0, y=0, z=0), up=dict(x=0, y=0, z=1)),
        ),
        annotations=[
            dict(
                text=t,
                x=xx,
                y=0.02,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=10, color="rgba(22,25,31,.55)", family="IBM Plex Mono"),
            )
            for t, xx in (("lateral", 0.25), ("medial", 0.75))
        ],
        margin=dict(l=0, r=0, t=0, b=0),
        height=430,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# =============================================================== layout =====
df, schema, prof = load_tables()
sub = df[df.parcellation == PRIMARY].reset_index(drop=True)

st.markdown('<div class="eyebrow">Cortical oxygen atlas</div>', unsafe_allow_html=True)
st.markdown("# Where the BOLD signal can be trusted")
st.markdown(
    '<p class="lede">In roughly 40% of responding cortex, oxygen metabolism moves '
    "the opposite way from what the imaging signal implies. This atlas maps where, "
    "and how much of the map the underlying data actually supports.</p>",
    unsafe_allow_html=True,
)
st.markdown('<hr class="rule">', unsafe_allow_html=True)

left, right = st.columns([1.45, 1], gap="large")

# The Region control is declared before the surface that depends on it. Streamlit
# places output by container, not by execution order, so `right` still renders to
# the right — this only removes the extra rerun the old code needed to get the
# selection back to the figure.
with right:
    names = sub.parcel_name.tolist()
    name = st.selectbox("Region", names, key="region")
    row = sub[sub.parcel_name == name].iloc[0]
    sel = int(row.parcel_index)

with left:
    c1, c2 = st.columns([2, 1.5])
    metric = c1.selectbox(
        "Colour the surface by", list(METRICS), format_func=lambda k: METRICS[k]
    )
    fade = c2.toggle(
        "Fade by confidence",
        value=True,
        help="Dissolve parcels the data cannot support toward the "
        "background. Encodes coverage, this metric's split-half "
        "reliability, and donor sampling — not the metric's value.",
    )

    reliability = metric_reliability(schema, metric)
    surf = load_surface()

    # Molecular neighbours of the selected parcel, from gene-set profiles.
    neighbours: list[int] = []
    w = prof.pivot(index="parcel_index", columns="gene_set", values="score_median")
    if sel in w.index:
        c = np.corrcoef(w.to_numpy())
        order = np.argsort(c[w.index.get_loc(sel)])[::-1]
        neighbours = [int(w.index[o]) for o in order if int(w.index[o]) != sel][:6]

    st.plotly_chart(
        brain_figure(surf, sub, metric, reliability, fade, sel, neighbours),
        width="stretch",
        key="brain",
    )

    lo_lab = f"{sub[metric].min():.2f}"
    hi_lab = f"{sub[metric].max():.2f}"
    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:.6rem;
        font-family:'IBM Plex Mono',monospace;font-size:.68rem;opacity:.75;">
        <span>{lo_lab}</span>
        <span style="flex:1;height:9px;border-radius:1px;
        background:linear-gradient(90deg,{DEOXY},{TISSUE},{OXY});"></span>
        <span>{hi_lab}</span>
        <span style="margin-left:.8rem;opacity:.7;">
        {"faded = low confidence" if fade else "fade off"}</span></div>
        <div style="font-size:.72rem;opacity:.55;margin-top:.45rem;">
        Left hemisphere, inflated. Drag to turn, hover to read a region out,
        and pick one with <b>Region</b> to see its molecular profile.</div>""",
        unsafe_allow_html=True,
    )

    if reliability is not None and reliability < 0.5:
        st.warning(
            f"**{METRICS[metric]}** has a split-half reliability of "
            f"{reliability:.2f}, below this project's 0.5 floor, so every parcel "
            "reads as unresolved no matter how well the scanner covered it. "
            "That is the honest picture for this column — it sums two "
            "topographically distinct modes. Use **Extraction mode** "
            "(0.58) or **Overshoot mode** (0.60) instead."
        )

with right:
    conf = confidence(row, reliability)
    network = name.split("_")[2] if len(name.split("_")) > 2 else "—"
    st.markdown(
        f'<div class="metric-label">{network} network · parcel {int(row.parcel_index)}'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="metric-big" style="color:{OXY};">{row.discordance_risk:.1%}</div>'
        '<div class="metric-label">of subjects show opposed signals here</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Extraction",
        f"{row.discordance_risk_extraction:.1%}",
        help="Oxygen demand rises, flow lags, signal falls.",
    )
    m2.metric(
        "Overshoot",
        f"{row.discordance_risk_overshoot:.1%}",
        help="Demand falls, flow arrives anyway, signal rises.",
    )
    m3.metric(
        "Baseline OEF", "—" if pd.isna(row.baseline_oef) else f"{row.baseline_oef:.3f}"
    )

    band = OXY if conf < 0.34 else (TISSUE if conf < 0.67 else INSTRUMENT)
    verdict = (
        "treat as unresolved"
        if conf < 0.34
        else "usable with caution"
        if conf < 0.67
        else "well supported"
    )
    st.markdown(
        f"""<div style="margin:.9rem 0 .3rem;">
        <div class="metric-label">confidence in this parcel</div>
        <div style="height:10px;background:rgba(22,25,31,.10);border-radius:1px;
        overflow:hidden;margin:.35rem 0;">
        <div style="height:100%;width:{conf * 100:.0f}%;background:{band};"></div></div>
        <span class="chip" style="background:{band}22;color:{band};">{verdict}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    # The reliability shown is the DISPLAYED metric's, not the coupling map's.
    # Naming the metric in the row label is the point: the same parcel is well
    # supported for baseline OEF (0.98) and unresolved for discordance risk
    # (0.49), and a row that just said "map reliability" hid that entirely.
    q = pd.DataFrame(
        {
            "check": [
                "Scanner coverage",
                "Venous partial volume",
                f"Reliability — {METRICS[metric].lower()}",
                "AHBA samples",
            ],
            "value": [
                f"{row.dropout_snr_coverage:.0%}",
                f"{row.venous_partial_volume:.3f}",
                "not established" if reliability is None else f"{reliability:.3f}",
                "—"
                if pd.isna(row.get("ahba_n_samples"))
                else f"{int(row.ahba_n_samples)}",
            ],
        }
    )
    table(q, widths=("62%", "38%"))

    if row.dropout_snr_coverage < 0.35:
        st.warning(
            "Most of this parcel is lost to scanner dropout. mqBOLD derives "
            "oxygen extraction from T2\\*, which is exactly what field "
            "inhomogeneity corrupts. Read its values as unresolved, not low."
        )
    if pd.notna(row.get("ahba_n_samples")) and row.ahba_n_samples == 0:
        st.warning(
            "No donor tissue was sampled here. The molecular profile is "
            "interpolated from neighbours, not measured."
        )

st.markdown('<hr class="rule">', unsafe_allow_html=True)
tab_mol, tab_map, tab_about = st.tabs(
    ["Molecular profile", "Score a statistical map", "What these numbers mean"]
)

with tab_mol:
    pr = prof[prof.parcel_index == int(row.parcel_index)].sort_values("score_median")
    if not len(pr):
        st.info("Gene-set profiles are built at schaefer200x7 only.")
    else:
        st.markdown(f"**{name}** — expression relative to the cortical average")
        fig = go.Figure(
            go.Bar(
                x=pr.score_median,
                y=[
                    g.replace("HALLMARK_", "").replace("_", " ").lower()
                    for g in pr.gene_set
                ],
                orientation="h",
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=pr.score_q3 - pr.score_median,
                    arrayminus=pr.score_median - pr.score_q1,
                    thickness=1.1,
                ),
                marker_color=[OXY if v > 0 else DEOXY for v in pr.score_median],
            )
        )
        fig.update_layout(
            height=330,
            margin=dict(l=0, r=0, t=6, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="median z-score across 12 pipelines (bars: IQR)",
        )
        st.plotly_chart(fig, width="stretch")
        if neighbours:
            nb = ", ".join(
                sub.loc[sub.parcel_index == n, "parcel_name"].iloc[0]
                for n in neighbours[:4]
            )
            st.caption(
                f"Arcs on the brain link this parcel to its most molecularly "
                f"similar regions — {nb}. This is similarity of expression "
                f"profile, **not** anatomical or functional connectivity."
            )

with tab_map:
    st.markdown(
        "Upload an MNI152 volume. It is projected to `fsaverage5` through "
        "`neuromaps.transforms`, parcellated, and weighted against per-parcel "
        "discordance risk."
    )
    up = st.file_uploader("MNI152 NIfTI", type=["nii", "gz"])
    if up is None:
        st.info(
            "No file loaded. This answers: *of the cortex my contrast actually "
            "implicates, how much is territory where BOLD and oxygen metabolism "
            "move in opposite directions?*"
        )
    else:
        import tempfile

        import nibabel as nib

        from src.data.parcellate import surface_from_mni152

        with tempfile.NamedTemporaryFile(
            suffix=".nii.gz" if up.name.endswith(".gz") else ".nii", delete=False
        ) as fh:
            fh.write(up.getbuffer())
            tmp = fh.name
        with st.spinner("Projecting to surface…"):
            lh, _rh = surface_from_mni152(nib.load(tmp), density="10k", method="linear")
            labels = surf["labels"]
            vals = np.array(
                [
                    np.nanmean(lh[labels == i]) if (labels == i).any() else np.nan
                    for i in range(1, surf["n_parcels"] + 1)
                ]
            )
        merged = sub.copy()
        merged["activation"] = vals[merged.parcel_index.to_numpy() - 1]
        ok = merged.activation.notna() & merged.discordance_risk.notna()
        w = merged.loc[ok, "activation"].abs()
        if not w.sum():
            st.error("Map is empty over left cortex after projection.")
        else:
            weighted = float((w * merged.loc[ok, "discordance_risk"]).sum() / w.sum())
            baseline = float(merged.loc[ok, "discordance_risk"].mean())
            a, b = st.columns(2)
            a.metric("Activation-weighted discordance risk", f"{weighted:.1%}")
            b.metric(
                "Cortex-wide average",
                f"{baseline:.1%}",
                delta=f"{(weighted - baseline) * 100:+.1f} pts",
            )
            if weighted > baseline + 0.03:
                st.warning(
                    "This contrast is weighted toward higher-discordance cortex "
                    "than average. A BOLD increase here is less safely read as "
                    "an increase in oxygen metabolism."
                )
            top = (
                merged.loc[ok]
                .assign(abs_a=lambda d: d.activation.abs())
                .nlargest(12, "abs_a")[
                    [
                        "parcel_name",
                        "activation",
                        "discordance_risk",
                        "dropout_snr_coverage",
                    ]
                ]
            )
            table(
                pd.DataFrame(
                    {
                        "region": top.parcel_name,
                        "activation": top.activation.map("{:.3f}".format),
                        "discordance risk": top.discordance_risk.map("{:.1%}".format),
                        "coverage": top.dropout_snr_coverage.map("{:.0%}".format),
                    }
                ),
                widths=("46%", "18%", "20%", "16%"),
            )

with tab_about:
    st.markdown(
        """
**Discordance** is BOLD and cerebral oxygen metabolism moving in *opposite*
directions in the same voxel during the same task. Epp et al. (2025) found it in
roughly 40% of significantly-responding voxels, concentrated in the default mode
network.

`discordance_risk` here is **our** measure — the fraction of subjects whose
coupling ratio falls below 1 — not the voxel percentage those authors report.

The two modes are physiologically distinct and spatially anticorrelated
(Spearman ≈ −0.56), so the combined column is the *least* reliable of the three.
Prefer the modes.

**What the fade encodes.** Scanner-dropout coverage, the split-half reliability
*of the column currently on screen*, and whether any donor tissue was sampled in
the parcel — whichever is worst. It is not the metric's value. Switch it off to
see raw colour, but a confident colour on a parcel with 40% coverage is a
confident colour about nothing.

Because the reliability term follows the displayed column, the same parcel can be
well supported for baseline oxygen extraction (0.98) and unresolved for
discordance risk (0.49). That is not an inconsistency — it is the two columns
being known to different degrees in the same piece of cortex.

**Why you hover the brain instead of clicking it.** Plotly's WebGL 3D scenes emit
no click event; only hover. Selection therefore lives in the **Region** control.
Everything the surface knows is in the tooltip.
"""
    )
    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    helps = {
        k: v.get("description", "")
        for k, v in schema.get("items", {}).get("properties", {}).items()
    }
    st.markdown("**Column definitions** — straight from the released schema")
    table(
        pd.DataFrame({"column": list(helps), "description": list(helps.values())}),
        widths=("26%", "74%"),
    )

st.sidebar.markdown(
    '<div class="eyebrow">Read before using a number</div>', unsafe_allow_html=True
)
st.sidebar.markdown(
    f"""
- **Left hemisphere only.** Only 2 of 6 Allen Human Brain Atlas donors have
  right-hemisphere tissue.
- **No individual inference.** Group medians over
  {int(sub.n_subjects_coupling.max())} subjects, one scanner, one site.
- **Spatial correlation is not mechanism.**
- Table **v{schema.get("version", "?")}**, status
  *{schema.get("status", "?")}*. Columns may change before 1.0.
"""
)
