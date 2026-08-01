"""The app reads released artifacts; this pins the contract between them.

The Streamlit app displays the annotation table and its gene-set profiles. Both
are rebuilt by the pipeline, so a column rename in ``build_annotation.py`` would
break the app silently — the failure would surface as an empty widget in a demo
rather than as a test failure.

These tests skip when the artifacts are absent, because they live under
``data/`` and a fresh clone has not built them yet.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
ANNOT = ROOT / "data" / "derived" / "annotation"

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
