"""Tests for the spatial null machinery — the enforcement point for R1.

These are the tests that matter most in the repo: if ``corr_with_null`` can be
coaxed into returning a p-value without a null distribution, or if its null
p-value is miscalibrated, every downstream result is invalid.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.stats.spatial import SpatialCorrResult, corr_with_null, fdr_bh


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def maps(rng):
    """A target map, a correlated map, and surrogates of the target."""
    n_parcels, n_perm = 100, 500
    x = rng.normal(size=n_parcels)
    y = x * 0.6 + rng.normal(size=n_parcels) * 0.8
    # Surrogates: permutations of x. Not spatially structured, but the right
    # shape and exchangeability for testing the p-value machinery.
    nulls = np.column_stack([rng.permutation(x) for _ in range(n_perm)])
    return x, y, nulls


class TestR1Enforcement:
    """R1: there must be no path to a p-value without a null distribution."""

    def test_none_nulls_raises(self, maps):
        x, y, _ = maps
        with pytest.raises(ValueError, match="R1"):
            corr_with_null(x, y, nulls=None)

    def test_empty_nulls_raises(self, maps):
        x, y, _ = maps
        with pytest.raises(ValueError, match="empty"):
            corr_with_null(x, y, nulls=np.empty((100, 0)))

    def test_nulls_must_be_2d(self, maps):
        x, y, _ = maps
        with pytest.raises(ValueError, match="2D"):
            corr_with_null(x, y, nulls=x.copy())

    def test_parcel_count_mismatch_raises(self, maps):
        x, y, nulls = maps
        with pytest.raises(ValueError, match="parcels"):
            corr_with_null(x, y, nulls=nulls[:50, :])

    def test_xy_length_mismatch_raises(self, maps):
        x, y, nulls = maps
        with pytest.raises(ValueError, match="length mismatch"):
            corr_with_null(x, y[:50], nulls=nulls)


class TestPValueBehaviour:
    def test_returns_result_dataclass(self, maps):
        x, y, nulls = maps
        res = corr_with_null(x, y, nulls=nulls)
        assert isinstance(res, SpatialCorrResult)
        assert 0 < res.p_spin <= 1
        assert res.n_valid == 100

    def test_p_never_exactly_zero(self, maps):
        """A finite permutation set cannot license p=0 (the +1 correction)."""
        x, _, nulls = maps
        res = corr_with_null(x, x, nulls=nulls)  # perfect correlation
        assert res.rho == pytest.approx(1.0)
        assert res.p_spin > 0
        assert res.p_spin == pytest.approx(1 / (nulls.shape[1] + 1))

    def test_uncorrelated_maps_give_high_p(self, rng):
        n, n_perm = 100, 1000
        x = rng.normal(size=n)
        y = rng.normal(size=n)
        nulls = np.column_stack([rng.permutation(x) for _ in range(n_perm)])
        res = corr_with_null(x, y, nulls=nulls)
        assert res.p_spin > 0.05

    def test_null_p_is_calibrated_under_the_null(self, rng):
        """Uniformity check: with exchangeable surrogates and an unrelated y,
        p_spin should be roughly uniform, so ~5% of trials fall below 0.05."""
        n, n_perm, n_trials = 60, 200, 300
        below = 0
        for _ in range(n_trials):
            x = rng.normal(size=n)
            y = rng.normal(size=n)
            nulls = np.column_stack([rng.permutation(x) for _ in range(n_perm)])
            if corr_with_null(x, y, nulls=nulls).p_spin < 0.05:
                below += 1
        rate = below / n_trials
        # Generous band: this catches gross miscalibration, not fine bias.
        assert 0.01 < rate < 0.12, f"false-positive rate {rate:.3f} off nominal 0.05"

    def test_spatial_p_exceeds_naive_p_for_smooth_maps(self, rng):
        """The whole reason R1 exists: on autocorrelated maps the naive p-value
        is anticonservative relative to a structure-preserving null."""
        n, n_perm = 100, 1000
        # Build smooth maps by convolving noise — a crude 1D stand-in for
        # cortical spatial autocorrelation.
        kernel = np.exp(-(np.linspace(-3, 3, 21) ** 2))
        kernel /= kernel.sum()
        smooth = lambda v: np.convolve(v, kernel, mode="same")  # noqa: E731
        x = smooth(rng.normal(size=n))
        y = smooth(rng.normal(size=n))
        # Surrogates preserve smoothness by circularly shifting x.
        nulls = np.column_stack([np.roll(x, s) for s in range(1, n_perm % n + n)])
        res = corr_with_null(x, y, nulls=nulls)
        assert res.p_spin >= res.p_naive


class TestNaNHandling:
    def test_nan_parcels_are_dropped(self, maps):
        x, y, nulls = maps
        x_nan = x.copy()
        x_nan[:10] = np.nan
        res = corr_with_null(x_nan, y, nulls=nulls)
        assert res.n_valid == 90

    def test_too_few_valid_parcels_raises(self, maps):
        x, y, nulls = maps
        x_nan = x.copy()
        x_nan[2:] = np.nan
        with pytest.raises(ValueError, match="valid parcels"):
            corr_with_null(x_nan, y, nulls=nulls)


class TestDeterminism:
    """R7: identical inputs must give byte-identical output."""

    def test_repeated_calls_identical(self, maps):
        x, y, nulls = maps
        a = corr_with_null(x, y, nulls=nulls)
        b = corr_with_null(x, y, nulls=nulls)
        assert a == b

    def test_seeded_surrogates_reproduce(self):
        gen = lambda: np.column_stack(  # noqa: E731
            [np.random.default_rng(42).permutation(np.arange(50.0)) for _ in range(10)]
        )
        np.testing.assert_array_equal(gen(), gen())


class TestFDR:
    def test_monotone_and_bounded(self, rng):
        p = rng.uniform(size=50)
        adj = fdr_bh(p)
        assert np.all(adj >= 0) and np.all(adj <= 1)
        # Adjusted p-values preserve the ordering of raw p-values.
        assert np.all(np.diff(adj[np.argsort(p)]) >= -1e-12)

    def test_adjusted_never_below_raw(self, rng):
        p = rng.uniform(size=50)
        assert np.all(fdr_bh(p) >= p - 1e-12)

    def test_known_values(self):
        # Classic BH worked example.
        p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        expected = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
        np.testing.assert_allclose(fdr_bh(p), expected, rtol=1e-9)
