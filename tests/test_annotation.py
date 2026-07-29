"""Tests for the released annotation table's stability labelling.

The table is the public deliverable, so a column labelled ``stable`` is a claim
other people will rely on. This got it wrong once: ``discordance_risk`` was
labelled stable while its split-half reliability was 0.49, below the 0.5 floor
the project sets for itself in CLAUDE.md §9. The label was a hand-maintained
string that had drifted from the measurement.

These tests pin the fix: labels for measurable columns are *derived* from the
Phase 0a reliability, so drift is impossible rather than merely unlikely.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_annotation import (
    COLUMNS,
    RELIABILITY_FLOOR,
    RELIABILITY_SOURCE,
    STABILITY,
    column_stability,
    map_reliabilities,
)


class TestLabelDerivation:
    def test_below_floor_is_low_reliability(self):
        labels = column_stability({"discordance_risk": RELIABILITY_FLOOR - 0.01})
        assert labels["discordance_risk"] == "low_reliability"

    def test_at_floor_is_stable(self):
        """The floor is inclusive, matching evaluate_gate in the Phase 0a code."""
        labels = column_stability({"baseline_oef": RELIABILITY_FLOOR})
        assert labels["baseline_oef"] == "stable"

    def test_measurement_overrides_any_asserted_default(self):
        """No hand-written default may survive a contradicting measurement."""
        for col in RELIABILITY_SOURCE:
            assert column_stability({col: 0.1})[col] == "low_reliability"

    def test_missing_phase0_leaves_columns_unlabelled(self):
        """Absent evidence must not silently become a 'stable' claim."""
        labels = column_stability({})
        for col in RELIABILITY_SOURCE:
            assert col not in labels

    def test_every_label_used_is_documented(self):
        labels = column_stability(map_reliabilities())
        assert set(labels.values()) <= set(STABILITY)

    def test_labelled_columns_all_exist_in_the_table(self):
        assert set(column_stability(map_reliabilities())) <= set(COLUMNS)


class TestAgainstRealPhase0Output:
    """These run against the committed Phase 0a results."""

    @pytest.fixture
    def reliab(self):
        r = map_reliabilities()
        if not r:
            pytest.skip("Phase 0a output not present")
        return r

    def test_all_measurable_columns_resolved(self, reliab):
        assert set(reliab) == set(RELIABILITY_SOURCE)

    def test_reliabilities_are_correlations(self, reliab):
        assert all(-1.0 <= v <= 1.0 for v in reliab.values())

    def test_the_regression_that_prompted_this(self, reliab):
        """discordance_risk must never again be advertised as stable."""
        assert reliab["discordance_risk"] < RELIABILITY_FLOOR
        assert column_stability(reliab)["discordance_risk"] == "low_reliability"

    def test_total_is_less_reliable_than_its_parts(self, reliab):
        """The reason the total is worse: the two modes are topographically
        distinct, so summing them cancels signal. If this ever inverts, the
        advice in the data dictionary needs rewriting."""
        assert reliab["discordance_risk"] < reliab["discordance_risk_extraction"]
        assert reliab["discordance_risk"] < reliab["discordance_risk_overshoot"]

    def test_mode_columns_clear_the_floor(self, reliab):
        labels = column_stability(reliab)
        assert labels["discordance_risk_extraction"] == "stable"
        assert labels["discordance_risk_overshoot"] == "stable"

    def test_baselines_are_the_most_reliable(self, reliab):
        """The dictionary tells users to build on baseline_*; that should be true."""
        assert reliab["baseline_oef"] == max(reliab.values())
