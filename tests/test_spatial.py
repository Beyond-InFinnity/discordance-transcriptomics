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


class TestSpinIndices:
    """``spin_indices`` claims its output is data-independent and that applying
    it is *exact*, not approximate. If that claim is false, every mediation
    p-value built on it is wrong, so it is pinned here rather than trusted."""

    def test_rejects_non_reindexing_nulls(self):
        """Variogram surrogates synthesise values and cannot be reused."""
        from src.stats.spatial import spin_indices

        with pytest.raises(ValueError, match="cannot"):
            spin_indices(100, method="burt2020")

    def test_apply_spin_shape_and_dtype(self):
        from src.stats.spatial import apply_spin

        x = np.arange(10.0)
        idx = np.tile(np.arange(10)[:, None], (1, 5))
        out = apply_spin(x, idx)
        assert out.shape == (10, 5)
        np.testing.assert_array_equal(out[:, 0], x)

    def test_apply_spin_is_a_gather(self):
        from src.stats.spatial import apply_spin

        x = np.array([10.0, 20.0, 30.0])
        idx = np.array([[2], [0], [1]])
        np.testing.assert_array_equal(apply_spin(x, idx).ravel(), [30.0, 10.0, 20.0])

    def test_apply_spin_parcel_mismatch_raises(self):
        from src.stats.spatial import apply_spin

        with pytest.raises(ValueError, match="parcels"):
            apply_spin(np.arange(5.0), np.zeros((7, 3), dtype=int))

    def test_apply_spin_requires_2d_idx(self):
        from src.stats.spatial import apply_spin

        with pytest.raises(ValueError, match="2D"):
            apply_spin(np.arange(5.0), np.arange(5))

    def test_apply_spin_propagates_nan(self):
        """A parcel that is NaN in the observed map stays NaN in every surrogate,
        so corr_with_null's masking behaves the same either way."""
        from src.stats.spatial import apply_spin

        x = np.array([1.0, np.nan, 3.0])
        out = apply_spin(x, np.array([[1], [2], [0]]))
        assert np.isnan(out[0, 0]) and out[1, 0] == 3.0


class TestSpinIndicesAgainstRealGeometry:
    """The exactness claim, checked against neuromaps itself.

    Slow: generates surrogates twice on the real Schaefer geometry. It earns the
    cost — the entire mediation phase reuses one index array across a dozen
    exposure maps, and that is only legitimate if reuse is bit-for-bit identical
    to generating each map's surrogates directly.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def geometry():
        pytest.importorskip("neuromaps")
        from src.data.parcellate import schaefer_gifti_for_nulls
        from src.utils.workbench import ensure_workbench

        try:
            ensure_workbench()
            parc = schaefer_gifti_for_nulls(200, 7, "10k", "L")
        except Exception as exc:
            pytest.skip(f"surface geometry unavailable: {exc}")
        return parc

    def test_reuse_is_bit_identical_to_direct_generation(self, geometry):
        from src.stats.spatial import apply_spin, make_nulls, spin_indices

        n, n_perm = 100, 20
        idx = spin_indices(
            n, parcellation=geometry, n_perm=n_perm, seed=42, density="10k"
        )
        assert idx.shape == (n, n_perm)

        x = np.random.default_rng(0).normal(size=n)
        direct = np.asarray(
            make_nulls(x, parcellation=geometry, n_perm=n_perm, seed=42, density="10k")
        )
        np.testing.assert_array_equal(apply_spin(x, idx), direct)

    def test_indices_are_data_independent(self, geometry):
        """Two unrelated maps must yield the same index array."""
        from src.stats.spatial import make_nulls

        n, n_perm = 100, 20
        kw = dict(parcellation=geometry, n_perm=n_perm, seed=42, density="10k")
        a = np.rint(np.asarray(make_nulls(np.arange(float(n)), **kw))).astype(int)
        rng = np.random.default_rng(7)
        shuffled = rng.permutation(np.arange(float(n)))
        b = np.asarray(make_nulls(shuffled, **kw))
        # b holds shuffled values; mapping them back through the sentinel index
        # must reproduce b exactly.
        np.testing.assert_array_equal(shuffled[a], b)

    def test_indices_are_a_resampling_not_a_permutation(self, geometry):
        """Documents the real behaviour: rotated parcels can share a source, so
        values repeat. A test asserting 'permutation' would be wrong."""
        from src.stats.spatial import spin_indices

        idx = spin_indices(100, parcellation=geometry, n_perm=20, seed=42, density="10k")
        counts = [len(np.unique(idx[:, i])) for i in range(idx.shape[1])]
        assert max(counts) <= 100
        assert min(counts) < 100, "expected at least one repeated source parcel"

    def test_seed_changes_the_indices(self, geometry):
        from src.stats.spatial import spin_indices

        kw = dict(parcellation=geometry, n_perm=20, density="10k")
        a = spin_indices(100, seed=42, **kw)
        b = spin_indices(100, seed=43, **kw)
        assert not np.array_equal(a, b)

    def test_cache_roundtrip_and_parcel_guard(self, geometry, tmp_path):
        from src.stats.spatial import spin_indices

        p = tmp_path / "idx.npy"
        kw = dict(parcellation=geometry, n_perm=20, seed=42, density="10k")
        a = spin_indices(100, cache_path=p, **kw)
        b = spin_indices(100, cache_path=p, **kw)  # served from cache
        np.testing.assert_array_equal(a, b)
        with pytest.raises(ValueError, match="parcels"):
            spin_indices(50, cache_path=p, **kw)


class TestNullGeometryResolution:
    """Null geometry must come from the named atlas, never from a default.

    Four scripts shared this::

        _PARC_SPEC = {"schaefer200x7": (200, 7), "schaefer400x7": (400, 7)}
        n_spec = _PARC_SPEC.get(parc, (200, 7))

    There is no dk68 entry, so `--parcellation dk68` silently rotated 100
    Schaefer parcels to build a null for a 34-parcel Desikan-Killiany map and
    reported a p-value without complaint. p5_hierarchy documented that flag in
    its usage string.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def geom():
        pytest.importorskip("neuromaps")
        from src.utils.workbench import ensure_workbench

        try:
            ensure_workbench()
        except Exception as exc:
            pytest.skip(f"workbench unavailable: {exc}")

    @pytest.mark.parametrize(
        "name,expected",
        [("schaefer200x7", 100), ("schaefer400x7", 200), ("dk68", 34)],
    )
    def test_each_atlas_gets_its_own_geometry(self, geom, name, expected):
        import numpy as np

        from src.data.parcellate import gifti_for_nulls

        (gii,) = gifti_for_nulls(name, "10k", "L")
        n = int(np.asarray(gii.agg_data()).max())
        assert n == expected, f"{name} resolved to {n} parcels, expected {expected}"

    def test_unknown_atlas_raises_rather_than_defaulting(self, geom):
        from src.data.parcellate import gifti_for_nulls

        with pytest.raises(ValueError, match="unknown parcellation"):
            gifti_for_nulls("not_an_atlas", "10k", "L")

    def test_dk68_does_not_silently_become_schaefer(self, geom):
        """The specific wrong answer the old default produced."""
        import numpy as np

        from src.data.parcellate import gifti_for_nulls

        (dk,) = gifti_for_nulls("dk68", "10k", "L")
        (sch,) = gifti_for_nulls("schaefer200x7", "10k", "L")
        assert int(np.asarray(dk.agg_data()).max()) != int(
            np.asarray(sch.agg_data()).max()
        )
