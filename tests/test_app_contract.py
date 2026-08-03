"""The app reads released artifacts; this pins the contract between them.

The Streamlit app displays the annotation table and its gene-set profiles. Both
are rebuilt by the pipeline, so a column rename in ``build_annotation.py`` would
break the app silently — the failure would surface as an empty widget in a demo
rather than as a test failure.

These tests skip when the artifacts are absent, because they live under
``data/`` and a fresh clone has not built them yet.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
ANNOT = ROOT / "data" / "derived" / "annotation"
APP = ROOT / "app" / "main.py"

# Every column app/main.py addresses by name.
APP_NEEDS = {
    "parcellation",
    "parcel_index",
    "parcel_name",
    "baseline_oef",
    "discordance_risk",
    "discordance_risk_extraction",
    "discordance_risk_overshoot",
    "dropout_snr_coverage",
    "venous_partial_volume",
    "map_reliability_coupling",
    "n_subjects_coupling",
    "baseline_source",
    "coupling_source",
    "ahba_n_samples",
}

PROFILE_NEEDS = {
    "parcellation",
    "parcel_index",
    "gene_set",
    "score_median",
    "score_q1",
    "score_q3",
    "n_cells",
    "n_genes_present",
    "n_genes_frozen",
}


@pytest.fixture(scope="module")
def annotation() -> pd.DataFrame:
    p = ANNOT / "discordance_annotation.csv"
    if not p.exists():
        pytest.skip("annotation table not built (run scripts/build_annotation.py)")
    return pd.read_csv(p)


@pytest.fixture(scope="module")
def schema() -> dict:
    p = ANNOT / "discordance_annotation.schema.json"
    if not p.exists():
        pytest.skip("annotation schema not built")
    with p.open() as fh:
        return json.load(fh)


class TestAnnotationContract:
    def test_app_columns_present(self, annotation):
        missing = APP_NEEDS - set(annotation.columns)
        assert not missing, f"app/main.py reads columns the table lacks: {missing}"

    def test_schema_documents_every_column(self, annotation, schema):
        documented = set(schema["items"]["properties"])
        undocumented = set(annotation.columns) - documented
        assert not undocumented, (
            "columns released with no schema description: "
            f"{sorted(undocumented)}. The app renders its definitions table "
            "from the schema, so an undocumented column is invisible to users."
        )

    def test_schema_describes_nothing_absent(self, annotation, schema):
        documented = set(schema["items"]["properties"])
        phantom = documented - set(annotation.columns)
        assert not phantom, f"schema documents columns not in the table: {phantom}"

    def test_primary_parcellation_is_complete(self, annotation):
        """100 left-hemisphere parcels, no duplicates — the app indexes by these."""
        s = annotation[annotation.parcellation == "schaefer200x7"]
        assert len(s) == 100
        assert s.parcel_index.is_unique
        assert set(s.parcel_index) == set(range(1, 101))

    def test_risk_columns_are_fractions(self, annotation):
        """The app formats these as percentages; out-of-range would mislead."""
        for col in (
            "discordance_risk",
            "discordance_risk_extraction",
            "discordance_risk_overshoot",
            "dropout_snr_coverage",
        ):
            v = annotation[col].dropna()
            assert ((v >= 0) & (v <= 1)).all(), f"{col} outside [0, 1]"

    def test_modes_partition_the_total_exactly(self, annotation):
        """Extraction and overshoot are a partition, not merely a subset.

        The app displays all three to one decimal place precisely so a reader
        can see them close. At whole percents 0.475 and 0.275 render as 48% and
        28% against a total of 75%, and a correct decomposition reads as broken
        arithmetic. If these ever stop summing, the display is wrong too.
        """
        a = annotation
        total = a.discordance_risk_extraction + a.discordance_risk_overshoot
        assert (total - a.discordance_risk).abs().max() < 1e-9


class TestConfidenceContract:
    """The fade is the app's central claim; these pin what it is allowed to say."""

    @staticmethod
    @pytest.fixture(scope="class")
    def app_metrics() -> dict[str, str]:
        """The METRICS dict, read from source rather than imported.

        Importing app/main.py would execute its Streamlit UI at collection time.
        """
        tree = ast.parse(APP.read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "METRICS" for t in node.targets
            ):
                return ast.literal_eval(node.value)
        pytest.fail("app/main.py no longer defines METRICS")

    def test_every_selectable_metric_is_a_real_column(self, app_metrics, annotation):
        missing = set(app_metrics) - set(annotation.columns)
        assert not missing, f"app offers metrics the table lacks: {sorted(missing)}"

    def test_faded_metrics_carry_a_reliability(self, app_metrics, schema):
        """Reliability drives the fade, so the measured columns must declare one.

        Only the QC/exposure columns are allowed to omit it — for those the fade
        legitimately falls back to coverage and donor sampling alone.
        """
        props = schema["items"]["properties"]
        qc_only = {"dropout_snr_coverage", "baseline_cbf"}
        for m in app_metrics:
            if m in qc_only:
                continue
            assert props[m].get("split_half_reliability") is not None, (
                f"{m} is displayed and faded but declares no split_half_reliability, "
                "so the fade would silently drop its reliability term and overstate "
                "how well the column is known"
            )

    def test_reliabilities_match_the_prose_the_app_shows(self, schema):
        """The warning banner quotes these to two decimals; drift makes it a lie.

        This project's most persistent defect is prose that outlives the number it
        describes. The app tells a reader discordance_risk is 0.49 and points them
        at 0.58 and 0.60 instead — pinned here so a rebuild cannot move one
        without failing the suite.

        The tolerance is half a display unit, inclusive, because "quoted to two
        decimals" means exactly that. Overshoot sits on the boundary: 0.595 is
        0.60 under half-up rounding and 0.59 under both banker's rounding and
        float formatting, and the released schema writes it 0.60. An exclusive
        bound would fail that agreement rather than check it.
        """
        props = schema["items"]["properties"]
        for col, quoted in (
            ("discordance_risk", 0.49),
            ("discordance_risk_extraction", 0.58),
            ("discordance_risk_overshoot", 0.60),
        ):
            actual = props[col]["split_half_reliability"]
            assert abs(actual - quoted) <= 5e-3 + 1e-9, (
                f"{col} is now {actual:.3f}; app/main.py still says {quoted:.2f}"
            )

    def test_combined_column_stays_below_the_floor(self, schema):
        """The banner's whole premise is that this column fails the 0.5 gate."""
        rel = schema["items"]["properties"]["discordance_risk"]["split_half_reliability"]
        assert rel < 0.5, (
            f"discordance_risk reliability rose to {rel:.3f}; it now passes the "
            "project's 0.5 floor and the app's 'treat as unresolved' banner is "
            "no longer warranted"
        )

    def test_reliability_is_not_taken_from_the_constant_column(self):
        """map_reliability_coupling is one number repeated on every row.

        Using it to fade whatever metric happened to be on screen flattered
        discordance_risk and penalised baseline_oef, and capped every parcel at
        0.887 so none could read as fully supported. The fade must read the
        displayed column's own reliability instead.
        """
        src = APP.read_text()
        start = src.index("def confidence(")
        end = src.index("\ndef ", start + 1)
        assert "map_reliability_coupling" not in src[start:end], (
            "confidence() reads map_reliability_coupling again — that column is a "
            "constant and cannot describe the metric being displayed"
        )

    def test_the_constant_column_is_still_released(self, annotation):
        """Dropped from the fade, but it remains a documented published column."""
        assert annotation.map_reliability_coupling.notna().all()


class TestNoArrowOnTheRerunPath:
    """Guards a crash that is real, intermittent, and hard to reproduce.

    ``st.dataframe`` serialises through ``pyarrow.Table.from_pandas``, which
    zero-copies the pandas buffers. Streamlit cancels an in-flight script run as
    soon as a new interaction arrives, so a rerun killed inside that call leaves
    Arrow reading freed memory. The app segfaulted three times in one review
    session that way — ``libarrow.so.2500``, null dereference at ``0x18``, same
    instruction pointer each time, taking the whole server down with it.

    Every table the app draws is a handful of rows of text, so ``app/main.py``
    renders them itself. This test exists because the failure is invisible in
    review: ``st.dataframe`` is the idiomatic call and looks completely correct.
    """

    def test_app_does_not_call_st_dataframe(self):
        tree = ast.parse(APP.read_text())
        offenders = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"dataframe", "table"}
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "st"
        ]
        assert not offenders, (
            f"app/main.py calls st.dataframe/st.table at line(s) {offenders}. "
            "Both serialise via pyarrow, which has segfaulted this app when a "
            "rerun was cancelled mid-serialisation. Use the local table() helper."
        )

    def test_local_table_helper_still_exists(self):
        tree = ast.parse(APP.read_text())
        names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert "table" in names, "the Arrow-free table() renderer was removed"


class TestProfileContract:
    @staticmethod
    @pytest.fixture(scope="class")
    def profiles() -> pd.DataFrame:
        p = ANNOT / "geneset_profiles.csv"
        if not p.exists():
            pytest.skip("profiles not built (scripts/build_geneset_profiles.py)")
        return pd.read_csv(p)

    def test_columns(self, profiles):
        missing = PROFILE_NEEDS - set(profiles.columns)
        assert not missing, f"app reads profile columns that are absent: {missing}"

    def test_quartiles_bracket_the_median(self, profiles):
        assert (profiles.score_q1 <= profiles.score_median + 1e-9).all()
        assert (profiles.score_median <= profiles.score_q3 + 1e-9).all()

    def test_every_profiled_parcel_exists_in_the_table(self, profiles, annotation):
        for parc, grp in profiles.groupby("parcellation"):
            known = set(annotation[annotation.parcellation == parc].parcel_index)
            orphan = set(grp.parcel_index) - known
            assert not orphan, f"{parc}: profiles for unknown parcels {sorted(orphan)}"

    def test_gene_counts_are_sane(self, profiles):
        assert (profiles.n_genes_present >= 3).all(), "a set below the 3-gene floor"
        assert (profiles.n_genes_present <= profiles.n_genes_frozen).all(), (
            "more genes found than the frozen set contains — the set was not "
            "resolved from config/genesets.yaml (R5)"
        )

    def test_multiverse_not_collapsed_to_one_cell(self, profiles):
        """R6: a single-pipeline profile hides the spread it exists to show."""
        assert (profiles.n_cells > 1).all()
