"""Discordance annotation browser — the released artifact, made usable.

Two questions, which are the ones the annotation table exists to answer:

1. *I care about region X. Is BOLD trustworthy there?*
2. *I have a statistical map. Does it land in territory where BOLD and oxygen
   metabolism disagree?*

Design constraint specific to this project: every number shown must arrive with
what qualifies it. A parcel's discordance risk is meaningless without its
reliability, its scanner-dropout coverage, and the number of donors behind its
expression profile. The table carries all of that, so the app shows it rather
than presenting a clean-looking number the data does not support.

Nothing here computes a result. The app reads artifacts built by the pipeline
and uses ``src.data.parcellate`` for coordinate transforms only (R4 forbids
hand-rolled ones). If you find yourself adding analysis here, it belongs in
``src/``.

Usage
-----
    streamlit run app/main.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ANNOT_DIR = ROOT / "data" / "derived" / "annotation"

st.set_page_config(
    page_title="Discordance annotation browser",
    page_icon="🧠",
    layout="wide",
)


# ---------------------------------------------------------------- data ------
@st.cache_data(show_spinner=False)
def load_annotation() -> tuple[pd.DataFrame, dict, pd.DataFrame | None]:
    """Annotation table, its JSON schema, and per-parcel gene-set profiles."""
    df = pd.read_csv(ANNOT_DIR / "discordance_annotation.csv")
    with (ANNOT_DIR / "discordance_annotation.schema.json").open() as fh:
        schema = json.load(fh)
    prof_path = ANNOT_DIR / "geneset_profiles.csv"
    prof = pd.read_csv(prof_path) if prof_path.exists() else None
    return df, schema, prof


def column_help(schema: dict) -> dict[str, str]:
    """Per-column descriptions, so tooltips come from the schema not from me."""
    props = schema.get("items", {}).get("properties", {})
    return {k: v.get("description", "") for k, v in props.items()}


def network_of(parcel_name: str) -> str:
    """Yeo network from a Schaefer label; DK names have no network field."""
    parts = str(parcel_name).split("_")
    return parts[2] if len(parts) > 2 else "—"


def pct_rank(series: pd.Series, value: float) -> float:
    s = series.dropna()
    if not len(s) or not np.isfinite(value):
        return float("nan")
    return float((s < value).mean() * 100)


# ------------------------------------------------------------- sidebar ------
df, schema, profiles = load_annotation()
helps = column_help(schema)

st.sidebar.title("Discordance annotation")
st.sidebar.caption(
    f"table v{schema.get('version', '?')} · status **{schema.get('status', '?')}**"
)

parcellation = st.sidebar.selectbox(
    "Parcellation",
    sorted(df.parcellation.unique(), key=lambda s: (s != "schaefer200x7", s)),
    help="schaefer200x7 is the primary analysis parcellation (§7.1). "
    "The others are sensitivity checks.",
)
sub = df[df.parcellation == parcellation].reset_index(drop=True)

st.sidebar.divider()
st.sidebar.markdown(
    f"""
**Read this before using a number.**

- **Left hemisphere only.** Only 2 of 6 AHBA donors have right-hemisphere
  tissue, so the whole project is left-lateralised by construction.
- **No individual-level inference.** These are group medians over
  {int(sub.n_subjects_coupling.max())} subjects from one scanner and one site.
  They describe a modal brain, not any particular one.
- **Spatial correlation is not mechanism.**
- The combined `discordance_risk` is **less reliable than either of its parts**
  — the two modes are topographically anticorrelated, so summing them cancels
  signal. Prefer the extraction and overshoot columns.
"""
)

# --------------------------------------------------------------- tabs -------
tab_region, tab_map, tab_about = st.tabs(
    ["Look up a region", "Score a statistical map", "What these numbers mean"]
)

# ============================================================ REGION =========
with tab_region:
    st.subheader("Per-parcel discordance risk and molecular profile")

    name = st.selectbox(
        "Region",
        sub.parcel_name.tolist(),
        help="Type to search. Schaefer labels are 7Networks_LH_<network>_<n>.",
    )
    row = sub[sub.parcel_name == name].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Discordance risk",
        f"{row.discordance_risk:.0%}",
        help=helps.get("discordance_risk", ""),
    )
    c2.metric(
        "Extraction mode",
        f"{row.discordance_risk_extraction:.0%}",
        help=helps.get("discordance_risk_extraction", ""),
    )
    c3.metric(
        "Overshoot mode",
        f"{row.discordance_risk_overshoot:.0%}",
        help=helps.get("discordance_risk_overshoot", ""),
    )
    c4.metric(
        "Baseline OEF",
        "—" if pd.isna(row.baseline_oef) else f"{row.baseline_oef:.3f}",
        help=helps.get("baseline_oef", ""),
    )

    p = pct_rank(sub.discordance_risk, row.discordance_risk)
    st.caption(
        f"**{name}** · network *{network_of(name)}* · "
        f"discordance risk is higher than **{p:.0f}%** of parcels in this "
        f"parcellation (median {sub.discordance_risk.median():.0%})."
    )

    left, right = st.columns([3, 2])

    with left:
        st.markdown("**Molecular profile** — expression relative to cortical mean")
        if profiles is None:
            st.info("Gene-set profiles not built for this parcellation.")
        else:
            pr = profiles[
                (profiles.parcellation == parcellation)
                & (profiles.parcel_index == row.parcel_index)
            ]
            if not len(pr):
                st.info(
                    "Gene-set profiles exist for schaefer200x7 only — the "
                    "expression multiverse is built at the primary parcellation."
                )
            else:
                pr = pr.sort_values("score_median")
                st.bar_chart(
                    pr.set_index("gene_set")["score_median"],
                    height=340,
                    color="#4c78a8",
                )
                st.caption(
                    "Median z-score across "
                    f"{int(pr.n_cells.iloc[0])} multiverse pipelines. Positive = "
                    "higher expression than the average cortical parcel. Bars "
                    "without their spread are misleading — see the table."
                )
                show = pr[
                    [
                        "gene_set",
                        "score_median",
                        "score_q1",
                        "score_q3",
                        "n_genes_present",
                        "n_genes_frozen",
                    ]
                ].rename(
                    columns={
                        "score_median": "median",
                        "score_q1": "Q1",
                        "score_q3": "Q3",
                        "n_genes_present": "genes found",
                        "n_genes_frozen": "genes frozen",
                    }
                )
                st.dataframe(
                    show,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        c: st.column_config.NumberColumn(format="%.3f")
                        for c in ("median", "Q1", "Q3")
                    },
                )

    with right:
        st.markdown("**Does this parcel support the numbers on the left?**")
        checks = []

        cov = row.dropout_snr_coverage
        checks.append(
            {
                "check": "Scanner dropout coverage",
                "value": f"{cov:.0%}",
                "verdict": "ok" if cov >= 0.6 else ("poor" if cov < 0.35 else "marginal"),
            }
        )
        ven = row.venous_partial_volume
        checks.append(
            {
                "check": "Venous partial volume",
                "value": f"{ven:.3f}",
                "verdict": "ok" if ven < 0.05 else "elevated",
            }
        )
        rel = row.map_reliability_coupling
        checks.append(
            {
                "check": "Coupling map reliability",
                "value": f"{rel:.3f}",
                "verdict": "ok" if rel >= 0.5 else "low",
            }
        )
        ahba = row.get("ahba_n_samples", np.nan)
        checks.append(
            {
                "check": "AHBA samples in parcel",
                "value": "—" if pd.isna(ahba) else f"{int(ahba)}",
                "verdict": (
                    "unknown"
                    if pd.isna(ahba)
                    else ("none" if ahba == 0 else ("ok" if ahba >= 3 else "sparse"))
                ),
            }
        )
        st.dataframe(pd.DataFrame(checks), hide_index=True, width="stretch")

        if cov < 0.35:
            st.warning(
                "This parcel loses most of its vertices to the scanner SNR "
                "criterion. mqBOLD derives oxygen extraction from T2\\*, which "
                "is exactly what field inhomogeneity corrupts. Treat its "
                "values as unresolved rather than low."
            )
        if not pd.isna(ahba) and ahba == 0:
            st.warning(
                "No AHBA tissue samples fall in this parcel. Its expression "
                "profile is interpolated from neighbours, not measured."
            )

        st.markdown("**Provenance**")
        st.caption(
            f"baseline: `{row.baseline_source}`  \ncoupling: `{row.coupling_source}`"
        )

# ============================================================== MAP ==========
with tab_map:
    st.subheader("Score an uploaded statistical map")
    st.markdown(
        "Upload an MNI152 volume — a NeuroVault contrast, a group *t*-map, "
        "anything volumetric. It is projected to `fsaverage5` through "
        "`neuromaps.transforms` and parcellated, then weighted against "
        "per-parcel discordance risk."
    )

    up = st.file_uploader("MNI152 NIfTI", type=["nii", "gz"])
    if up is None:
        st.info(
            "No file loaded. This answers: *of the cortex my contrast "
            "actually implicates, how much of it is territory where BOLD and "
            "oxygen metabolism move in opposite directions?*"
        )
    elif parcellation != "schaefer200x7":
        st.warning("Map scoring runs at schaefer200x7. Switch parcellation.")
    else:
        try:
            import tempfile

            import nibabel as nib

            from src.data.parcellate import get_parcellation, surface_from_mni152

            with tempfile.NamedTemporaryFile(
                suffix=".nii.gz" if up.name.endswith(".gz") else ".nii", delete=False
            ) as fh:
                fh.write(up.getbuffer())
                tmp = fh.name

            with st.spinner("Projecting to surface and parcellating…"):
                img = nib.load(tmp)
                lh, _rh = surface_from_mni152(img, density="10k", method="linear")
                labels, _gii, n_parc = get_parcellation("schaefer200x7", "10k", "L")
                vals = np.full(n_parc, np.nan)
                for i in range(1, n_parc + 1):
                    m = labels == i
                    if m.any():
                        v = lh[m]
                        v = v[np.isfinite(v)]
                        if v.size:
                            vals[i - 1] = float(v.mean())

            merged = sub.copy()
            merged["activation"] = vals[merged.parcel_index.to_numpy() - 1]
            ok = merged.activation.notna() & merged.discordance_risk.notna()
            w = merged.loc[ok, "activation"].abs()
            risk = merged.loc[ok, "discordance_risk"]

            if not w.sum():
                st.error("Map is empty over left cortex after projection.")
            else:
                weighted = float((w * risk).sum() / w.sum())
                baseline = float(risk.mean())
                m1, m2, m3 = st.columns(3)
                m1.metric("Activation-weighted discordance risk", f"{weighted:.1%}")
                m2.metric(
                    "Cortex-wide average",
                    f"{baseline:.1%}",
                    delta=f"{(weighted - baseline) * 100:+.1f} pts",
                )
                cov_w = float(
                    (w * merged.loc[ok, "dropout_snr_coverage"]).sum() / w.sum()
                )
                m3.metric("Weighted dropout coverage", f"{cov_w:.0%}")

                if weighted > baseline + 0.03:
                    st.warning(
                        "This contrast is weighted toward higher-discordance "
                        "cortex than average. A BOLD increase here is less "
                        "safely read as an increase in oxygen metabolism."
                    )
                if cov_w < 0.5:
                    st.warning(
                        "The implicated cortex sits in low-SNR territory. Some "
                        "of the apparent discordance may be measurement, not "
                        "physiology — Phase 0b keeps dropout as a mandatory "
                        "covariate for exactly this reason."
                    )

                top = (
                    merged.loc[ok]
                    .assign(abs_activation=lambda d: d.activation.abs())
                    .nlargest(12, "abs_activation")[
                        [
                            "parcel_name",
                            "activation",
                            "discordance_risk",
                            "discordance_risk_extraction",
                            "discordance_risk_overshoot",
                            "dropout_snr_coverage",
                        ]
                    ]
                )
                st.markdown("**Most-implicated parcels**")
                st.dataframe(
                    top,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "activation": st.column_config.NumberColumn(format="%.3f"),
                        "discordance_risk": st.column_config.ProgressColumn(
                            min_value=0.0, max_value=1.0, format="%.2f"
                        ),
                    },
                )
        except Exception as exc:
            st.error(f"Could not process the map: {type(exc).__name__}: {exc}")
            st.caption(
                "Surface projection needs `neuromaps` and Connectome Workbench "
                "(`wb_command`) on PATH — see CLAUDE.md §4.1."
            )

# ============================================================ ABOUT ==========
with tab_about:
    st.subheader("What these numbers mean")
    st.markdown(
        """
**Discordance** is BOLD and cerebral oxygen metabolism moving in *opposite*
directions in the same voxel during the same task. Epp et al. (2025,
*Nature Neuroscience*) found it in roughly 40% of significantly-responding
voxels, concentrated in the default mode network.

`discordance_risk` here is **our** measure — the fraction of subjects in whom a
parcel's coupling ratio falls below 1 — not the voxel percentage those authors
report. The two are not directly comparable.

The two modes are physiologically distinct and worth keeping apart:

- **Extraction** — oxygen demand rises, flow lags, so the signal falls.
- **Overshoot** — demand falls, flow arrives anyway, so the signal rises.

They are spatially anticorrelated (Spearman ~ -0.56), which is why the combined
column is the *least* reliable of the three.
"""
    )
    st.divider()
    st.markdown("**Column definitions** — straight from the released schema")
    st.dataframe(
        pd.DataFrame({"column": list(helps), "description": [helps[k] for k in helps]}),
        hide_index=True,
        width="stretch",
    )
    st.divider()
    st.markdown(
        """
**Known limitations**, stated here rather than discovered later:

- AHBA is 6 post-mortem adult donors (5 usable — donor 15496 is 404 upstream),
  bulk microarray, predominantly left hemisphere. A modal brain, not a sample.
- The volumetric→surface projection discards subcortex and cerebellum.
- mqBOLD carries its own assumptions — vessel geometry, blood volume, T2'
  modelling — and they propagate into every OEF and CMRO₂ value here.
- The scanner-dropout confound is mitigated by covariate adjustment, not
  eliminated.
- Table status is **provisional**. Columns can change until it is versioned 1.0.
"""
    )
