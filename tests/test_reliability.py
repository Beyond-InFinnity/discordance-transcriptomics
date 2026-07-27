"""Tests for the Phase 0a reliability gate.

The gate decides whether the whole project proceeds, so its arithmetic and its
threshold logic both need to be right.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.stats.reliability import (
    evaluate_gate,
    icc21,
    run_reliability,
    spearman_brown,
    split_half_reliability,
)


def make_data(n_sub=40, n_par=100, signal=1.0, noise=1.0, seed=0):
    """Subjects sharing a common parcel map plus per-subject noise.

    ``signal``/``noise`` controls how reliable the group map is.
    """
    rng = np.random.default_rng(seed)
    common = rng.normal(size=n_par) * signal
    return common[None, :] + rng.normal(size=(n_sub, n_par)) * noise


class TestSpearmanBrown:
    def test_known_value(self):
        # r=0.5 half-length -> 2*0.5/(1+0.5) = 0.667
        assert spearman_brown(0.5) == pytest.approx(2 / 3)

    def test_identity_at_one(self):
        assert spearman_brown(1.0) == pytest.approx(1.0)

    def test_zero_stays_zero(self):
        assert spearman_brown(0.0) == pytest.approx(0.0)

    def test_always_increases_positive_r(self):
        for r in (0.1, 0.3, 0.45, 0.7, 0.9):
            assert spearman_brown(r) > r

    def test_negative_r_passed_through(self):
        """Inflating a negative correlation would misrepresent it."""
        assert spearman_brown(-0.3) == pytest.approx(-0.3)


class TestSplitHalf:
    def test_shapes(self):
        raw, corr = split_half_reliability(make_data(), n_splits=50, seed=42)
        assert raw.shape == corr.shape == (50,)

    def test_high_signal_gives_high_reliability(self):
        _, corr = split_half_reliability(
            make_data(signal=3.0, noise=0.5), n_splits=100, seed=42
        )
        assert np.nanmedian(corr) > 0.9

    def test_pure_noise_gives_near_zero(self):
        _, corr = split_half_reliability(
            make_data(signal=0.0, noise=1.0), n_splits=200, seed=42
        )
        assert abs(np.nanmedian(corr)) < 0.25

    def test_more_noise_lowers_reliability(self):
        _, clean = split_half_reliability(
            make_data(signal=2.0, noise=0.5, seed=1), n_splits=100, seed=42
        )
        _, dirty = split_half_reliability(
            make_data(signal=2.0, noise=4.0, seed=1), n_splits=100, seed=42
        )
        assert np.nanmedian(clean) > np.nanmedian(dirty)

    def test_corrected_exceeds_raw_when_positive(self):
        raw, corr = split_half_reliability(
            make_data(signal=2.0, noise=1.0), n_splits=100, seed=42
        )
        pos = raw > 0
        assert np.all(corr[pos] >= raw[pos] - 1e-12)

    def test_deterministic_under_seed(self):
        """R7: same seed, byte-identical output."""
        d = make_data()
        a, _ = split_half_reliability(d, n_splits=50, seed=42)
        b, _ = split_half_reliability(d, n_splits=50, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        d = make_data()
        a, _ = split_half_reliability(d, n_splits=50, seed=42)
        b, _ = split_half_reliability(d, n_splits=50, seed=43)
        assert not np.array_equal(a, b)

    def test_nan_parcels_tolerated(self):
        d = make_data()
        d[:, :5] = np.nan  # parcels with zero AHBA/EPI coverage
        _, corr = split_half_reliability(d, n_splits=50, seed=42)
        assert np.isfinite(corr).all()

    @pytest.mark.parametrize(
        "shape,match",
        [((3, 100), "4 subjects"), ((40, 2), "3 parcels")],
    )
    def test_rejects_degenerate_input(self, shape, match):
        with pytest.raises(ValueError, match=match):
            split_half_reliability(np.random.default_rng(0).normal(size=shape))

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="n_subjects"):
            split_half_reliability(np.arange(100.0))


class TestGateLogic:
    """The thresholds that decide whether the project proceeds (§9)."""

    @pytest.mark.parametrize(
        "median,expected",
        [
            (0.90, "pass"),
            (0.51, "pass"),
            (0.50, "pass"),  # boundary is inclusive
            (0.49, "caveat"),
            (0.31, "caveat"),
            (0.30, "caveat"),  # boundary is inclusive
            (0.29, "stop"),
            (0.00, "stop"),
            (-0.20, "stop"),
        ],
    )
    def test_thresholds(self, median, expected):
        assert evaluate_gate(np.full(100, median)) == expected

    def test_uses_median_not_mean(self):
        """A few extreme splits must not drag the verdict across a boundary."""
        vals = np.concatenate([np.full(90, 0.6), np.full(10, -5.0)])
        assert np.mean(vals) < 0.3  # mean would say STOP
        assert evaluate_gate(vals) == "pass"  # median says PASS


class TestICC:
    def test_shape(self):
        assert icc21(make_data(n_par=100)).shape == (100,)

    def test_bounded_above_by_one(self):
        assert np.nanmax(icc21(make_data(signal=3.0, noise=0.1))) <= 1.0 + 1e-9

    def test_rejects_degenerate(self):
        with pytest.raises(ValueError, match=">=2"):
            icc21(np.ones((1, 10)))


class TestRunReliability:
    def test_end_to_end_pass(self):
        res = run_reliability(make_data(signal=3.0, noise=0.5), n_splits=100)
        assert res.verdict == "pass"
        assert res.passed
        assert res.ci_lo <= res.median_r_corrected <= res.ci_hi
        assert res.n_subjects == 40 and res.n_parcels == 100

    def test_end_to_end_stop(self):
        res = run_reliability(make_data(signal=0.0, noise=1.0), n_splits=200)
        assert res.verdict == "stop"
        assert not res.passed

    def test_result_is_serialisable(self):
        """Must round-trip into a manifest (R10)."""
        import json

        res = run_reliability(make_data(), n_splits=20)
        assert json.loads(json.dumps(res.as_dict()))["verdict"] in {
            "pass",
            "caveat",
            "stop",
        }
