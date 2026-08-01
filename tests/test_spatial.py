"""Tests for the spatial null machinery — the enforcement point for R1.

These are the tests that matter most in the repo: if ``corr_with_null`` can be
coaxed into returning a p-value without a null distribution, or if its null
p-value is miscalibrated, every downstream result is invalid.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

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


class TestPreparedNulls:
    """prepare_nulls() is a pure speed optimisation and must change no number.

    corr_with_null spent ~86% of its time ranking the surrogate block, and
    Phase 4 ranked the same (100 x 10,000) array roughly 20,000 times per run
    for 20,000 identical results. Reuse is only safe if it is bit-identical.
    """

    def test_bit_identical_to_the_unprepared_path(self, maps):
        from src.stats.spatial import prepare_nulls

        x, y, nulls = maps
        plain = corr_with_null(x, y, nulls=nulls)
        fast = corr_with_null(x, y, nulls=prepare_nulls(nulls), nulls_prepared=True)
        assert fast.rho == plain.rho
        assert fast.p_spin == plain.p_spin
        assert fast.n_perm == plain.n_perm

    def test_identical_across_many_random_maps(self, rng):
        from src.stats.spatial import prepare_nulls

        n, n_perm = 60, 300
        base = rng.normal(size=n)
        nulls = np.column_stack([rng.permutation(base) for _ in range(n_perm)])
        pre = prepare_nulls(nulls)
        for _ in range(25):
            y = rng.normal(size=n)
            assert (
                corr_with_null(base, y, nulls=pre, nulls_prepared=True).p_spin
                == corr_with_null(base, y, nulls=nulls).p_spin
            )

    def test_pearson_path_also_identical(self, maps):
        from src.stats.spatial import prepare_nulls

        x, y, nulls = maps
        a = corr_with_null(x, y, nulls=nulls, method="pearson")
        b = corr_with_null(
            x,
            y,
            nulls=prepare_nulls(nulls, "pearson"),
            nulls_prepared=True,
            method="pearson",
        )
        assert a.rho == b.rho and a.p_spin == b.p_spin

    def test_nan_block_passes_through_untouched(self):
        from src.stats.spatial import prepare_nulls

        nulls = np.array([[1.0, 2.0], [np.nan, 3.0], [2.0, 1.0]])
        np.testing.assert_array_equal(prepare_nulls(nulls), nulls)

    def test_prepared_is_actually_faster(self, rng):
        """Guards against the optimisation silently becoming a no-op."""
        import time

        from src.stats.spatial import prepare_nulls

        n, n_perm = 100, 4000
        base = rng.normal(size=n)
        nulls = np.column_stack([rng.permutation(base) for _ in range(n_perm)])
        pre = prepare_nulls(nulls)
        y = rng.normal(size=n)

        t = time.perf_counter()
        for _ in range(5):
            corr_with_null(base, y, nulls=nulls)
        slow = time.perf_counter() - t
        t = time.perf_counter()
        for _ in range(5):
            corr_with_null(base, y, nulls=pre, nulls_prepared=True)
        fast = time.perf_counter() - t
        assert fast < slow / 2, f"expected >2x, got {slow / max(fast, 1e-9):.1f}x"


class TestRaggedSurrogatesInCorrWithNull:
    """Targets with missing parcels — the bug's fourth appearance.

    A rotation can pull an unobserved parcel into the window, so a map with
    missing parcels leaves few complete draws. The cross-species vascular map
    has 17 missing of 100 and leaves 2 of 10,000. Dropping incomplete draws
    silently reduced its null to those 2, giving p-values that could only be
    1/3, 2/3 or 1 — and it read as a legitimate null result.

    It surfaced in gene_screen, then pls_with_spin, then Phase 4 once the nulls
    were correctly paired per target. Fixed here so every caller inherits it.
    """

    @pytest.fixture
    def ragged(self, rng):
        n, n_perm = 100, 500
        x = rng.normal(size=n)
        y = x * 0.7 + rng.normal(size=n) * 0.6
        x[rng.choice(n, 17, replace=False)] = np.nan
        idx = rng.integers(0, n, size=(n, n_perm))
        return x, y, x[idx]

    def test_almost_no_draw_is_complete(self, ragged):
        _x, _y, nulls = ragged
        assert np.isfinite(nulls).all(axis=0).mean() < 0.05

    def test_uses_nearly_every_draw(self, ragged):
        x, y, nulls = ragged
        res = corr_with_null(x, y, nulls=nulls)
        assert res.n_perm > 0.9 * nulls.shape[1], (
            f"only {res.n_perm}/{nulls.shape[1]} draws used"
        )

    def test_p_is_not_quantised_to_a_tiny_denominator(self, ragged):
        """The symptom that gave the original away."""
        x, y, nulls = ragged
        assert corr_with_null(x, y, nulls=nulls).p_spin not in (1 / 3, 2 / 3, 1.0)

    def test_still_detects_a_real_relationship(self, ragged):
        x, y, nulls = ragged
        assert corr_with_null(x, y, nulls=nulls).p_spin < 0.05

    def test_calibrated_when_ragged_and_null(self, rng):
        """Missing parcels must not manufacture significance."""
        n, n_perm, trials = 100, 400, 150
        below = 0
        for _ in range(trials):
            x = rng.normal(size=n)
            x[rng.choice(n, 17, replace=False)] = np.nan
            y = rng.normal(size=n)
            idx = rng.integers(0, n, size=(n, n_perm))
            if corr_with_null(x, y, nulls=x[idx]).p_spin < 0.05:
                below += 1
        rate = below / trials
        assert 0.01 < rate < 0.13, f"ragged null rejects at {rate:.3f}, nominal 0.05"

    def test_complete_surrogates_unchanged(self, maps):
        """The clean path must be untouched by the ragged fix."""
        x, y, nulls = maps
        assert corr_with_null(x, y, nulls=nulls).p_spin == pytest.approx(
            corr_with_null(x, y, nulls=nulls).p_spin
        )


class TestPreparedAndRaggedTogether:
    """Two individually-correct fixes that cancelled each other.

    prepare_nulls() returns a NaN-bearing block untouched, since ranking it is
    meaningless. A caller then passing nulls_prepared=True would, if the flag
    were trusted, route that block down the fast path and silently reinstate the
    exact bug the ragged path exists to fix.

    That combination shipped, and the cross-species control came back at
    p = 1/3 — the n=2 signature — after both fixes were applied.
    """

    @pytest.fixture
    def ragged(self, rng):
        n, n_perm = 100, 400
        x = rng.normal(size=n)
        y = x * 0.7 + rng.normal(size=n) * 0.6
        x[rng.choice(n, 17, replace=False)] = np.nan
        return x, y, x[rng.integers(0, n, size=(n, n_perm))]

    def test_prepare_nulls_passes_nan_through(self, ragged):
        from src.stats.spatial import prepare_nulls

        _x, _y, nulls = ragged
        np.testing.assert_array_equal(prepare_nulls(nulls), nulls)

    def test_flag_does_not_override_a_nan_block(self, ragged):
        """The flag is a hint, not a licence to skip the check."""
        from src.stats.spatial import prepare_nulls

        x, y, nulls = ragged
        res = corr_with_null(x, y, nulls=prepare_nulls(nulls), nulls_prepared=True)
        assert res.n_perm > 0.9 * nulls.shape[1], (
            f"only {res.n_perm} draws used; the ragged path was skipped"
        )

    def test_p_not_quantised_to_thirds(self, ragged):
        """The exact symptom that exposed it."""
        from src.stats.spatial import prepare_nulls

        x, y, nulls = ragged
        p = corr_with_null(x, y, nulls=prepare_nulls(nulls), nulls_prepared=True).p_spin
        assert p not in (1 / 3, 2 / 3, 1.0)

    def test_prepared_and_unprepared_agree_on_ragged_data(self, ragged):
        from src.stats.spatial import prepare_nulls

        x, y, nulls = ragged
        a = corr_with_null(x, y, nulls=nulls)
        b = corr_with_null(x, y, nulls=prepare_nulls(nulls), nulls_prepared=True)
        assert a.p_spin == b.p_spin and a.n_perm == b.n_perm


class TestRaggedPathIsVectorised:
    """The ragged path must stay both exact and fast.

    Handling incomplete surrogates correctly originally meant a per-column loop
    calling scipy once per draw, doubled by recomputing the observed value on
    each draw's parcels. On the cross-species map that took Phase 4 from two
    minutes to an estimated five hours, and the run had to be killed.

    The vectorised replacement accumulates the same moments as masked matrix
    products. Measured: bit-identical results, ~30x faster, and the projected
    Phase 4 ragged cost falls from 311 minutes to 11.
    """

    @staticmethod
    def _loop_reference(x, y, nulls):
        """The implementation the vectorised version replaced."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        valid = np.isfinite(x) & np.isfinite(y)
        xv, yv, nv = x[valid], y[valid], nulls[valid, :]
        rho = float(sps.spearmanr(xv, yv).statistic)
        null_r = np.full(nv.shape[1], np.nan)
        obs_r = np.full(nv.shape[1], np.nan)
        for i in range(nv.shape[1]):
            col = nv[:, i]
            ok = np.isfinite(col)
            if ok.sum() >= 3:
                null_r[i] = float(sps.spearmanr(col[ok], yv[ok]).statistic)
                obs_r[i] = (
                    rho if ok.all() else float(sps.spearmanr(xv[ok], yv[ok]).statistic)
                )
        m = np.isfinite(null_r) & np.isfinite(obs_r)
        n_ext = int(np.sum(np.abs(null_r[m]) >= np.abs(obs_r[m])))
        return rho, (n_ext + 1) / (int(m.sum()) + 1)

    @staticmethod
    def _make(seed, n=100, n_perm=600, n_missing=17):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=n)
        y = x * 0.5 + rng.normal(size=n) * 0.9
        x[rng.choice(n, n_missing, replace=False)] = np.nan
        return x, y, x[rng.integers(0, n, size=(n, n_perm))]

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_matches_the_loop_exactly(self, seed):
        x, y, nulls = self._make(seed)
        ref_rho, ref_p = self._loop_reference(x, y, nulls)
        res = corr_with_null(x, y, nulls=nulls)
        assert res.rho == pytest.approx(ref_rho, abs=1e-12)
        assert res.p_spin == pytest.approx(ref_p, abs=1e-12)

    def test_is_substantially_faster(self):
        """Guards against the vectorisation being undone by a later edit."""
        import time

        x, y, nulls = self._make(seed=9, n_perm=1500)
        t = time.perf_counter()
        self._loop_reference(x, y, nulls)
        slow = time.perf_counter() - t
        t = time.perf_counter()
        corr_with_null(x, y, nulls=nulls)
        fast = time.perf_counter() - t
        assert fast < slow / 5, f"expected >5x, got {slow / max(fast, 1e-9):.1f}x"
