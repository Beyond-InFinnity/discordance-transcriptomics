"""Tests for the Phase 6 path model.

The module replaces an explicit two-predictor regression with closed-form
algebra on three correlations. That is only safe if the algebra is right, so it
is checked three ways here: against the exact decomposition identity, against a
direct least-squares fit, and against ``pingouin``'s partial correlation.

The inference is also checked for the specific mistake the module was written to
avoid — using one rotation set for every path, which mistests ``b``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.stats.mediation import MediationResult, mediation, path_coefficients


@pytest.fixture
def chain():
    """A genuine X -> M -> Y chain, plus exchangeable surrogates for each map."""
    rng = np.random.default_rng(42)
    n, n_perm = 100, 500
    x = rng.normal(size=n)
    m = 0.7 * x + rng.normal(size=n) * 0.7
    y = 0.6 * m + rng.normal(size=n) * 0.7
    xn = np.column_stack([rng.permutation(x) for _ in range(n_perm)])
    yn = np.column_stack([rng.permutation(y) for _ in range(n_perm)])
    return x, m, y, xn, yn


@pytest.fixture
def broken_b():
    """X predicts M strongly, but M carries no information about Y.

    This is the shape of the real Phase 6 data, and the case where naive
    reporting ("no mediation found") hides which link failed.
    """
    rng = np.random.default_rng(7)
    n, n_perm = 100, 500
    x = rng.normal(size=n)
    m = 0.9 * x + rng.normal(size=n) * 0.3
    y = rng.normal(size=n)
    xn = np.column_stack([rng.permutation(x) for _ in range(n_perm)])
    yn = np.column_stack([rng.permutation(y) for _ in range(n_perm)])
    return x, m, y, xn, yn


class TestPathAlgebra:
    def test_decomposition_identity(self):
        """c = c' + a*b must hold exactly, or the model is not a decomposition."""
        rng = np.random.default_rng(0)
        for _ in range(200):
            r_xm, r_xy, r_my = rng.uniform(-0.9, 0.9, 3)
            a, b, c, cp = path_coefficients(r_xm, r_xy, r_my)
            assert c == pytest.approx(cp + a * b, abs=1e-12)

    def test_matches_least_squares(self):
        """b and c' must equal the coefficients of a standardised 2-predictor OLS."""
        rng = np.random.default_rng(1)
        n = 200
        x = rng.normal(size=n)
        m = 0.5 * x + rng.normal(size=n)
        y = 0.4 * m + 0.3 * x + rng.normal(size=n)
        z = lambda v: (v - v.mean()) / v.std()  # noqa: E731
        x, m, y = z(x), z(m), z(y)

        design = np.column_stack([np.ones(n), x, m])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        _, b, _, cp = path_coefficients(
            float(x @ m / n), float(x @ y / n), float(m @ y / n)
        )
        assert cp == pytest.approx(beta[1], abs=1e-10)
        assert b == pytest.approx(beta[2], abs=1e-10)

    def test_matches_pingouin_partial_correlation(self):
        """b has the sign and significance of partial(M, Y | X)."""
        pg = pytest.importorskip("pingouin")
        import pandas as pd

        rng = np.random.default_rng(2)
        n = 300
        x = rng.normal(size=n)
        m = 0.6 * x + rng.normal(size=n)
        y = 0.5 * m + rng.normal(size=n)
        df = pd.DataFrame({"x": x, "m": m, "y": y})
        pr = pg.partial_corr(data=df, x="m", y="y", covar="x", method="pearson")
        z = lambda v: (v - v.mean()) / v.std()  # noqa: E731
        xs, ms, ys = z(x), z(m), z(y)
        _, b, _, _ = path_coefficients(
            float(xs @ ms / n), float(xs @ ys / n), float(ms @ ys / n)
        )
        # A standardised regression coefficient is not a partial correlation,
        # but they share sign and vanish together.
        assert np.sign(b) == np.sign(pr["r"].iloc[0])
        assert abs(b) > 0.2 and float(pr.filter(like="p").iloc[0, 0]) < 0.001

    def test_collinear_exposure_and_mediator_give_nan(self):
        """When X and M are the same map, b and c' are not identified."""
        _, b, _, cp = path_coefficients(0.999, 0.5, 0.5)
        assert np.isnan(b) and np.isnan(cp)

    def test_vectorises_over_a_rotation_set(self):
        r_xm = np.linspace(-0.8, 0.8, 50)
        r_xy = np.full(50, 0.3)
        a, b, c, cp = path_coefficients(r_xm, r_xy, 0.4)
        assert a.shape == b.shape == c.shape == cp.shape == (50,)
        np.testing.assert_allclose(c, cp + a * b, atol=1e-12)


class TestR1Enforcement:
    def test_missing_outcome_nulls_raises(self, chain):
        x, m, y, xn, _ = chain
        with pytest.raises(ValueError, match="R1"):
            mediation(x, m, y, x_nulls=xn, y_nulls=None)

    def test_missing_exposure_nulls_raises(self, chain):
        x, m, y, _, yn = chain
        with pytest.raises(ValueError, match="R1"):
            mediation(x, m, y, x_nulls=None, y_nulls=yn)

    def test_empty_nulls_raises(self, chain):
        x, m, y, _xn, yn = chain
        with pytest.raises(ValueError, match="empty"):
            mediation(x, m, y, x_nulls=np.empty((100, 0)), y_nulls=yn)

    def test_wrong_parcel_count_raises(self, chain):
        x, m, y, xn, yn = chain
        with pytest.raises(ValueError, match="parcels"):
            mediation(x, m, y, x_nulls=xn[:50], y_nulls=yn)

    def test_nulls_must_be_2d(self, chain):
        x, m, y, xn, yn = chain
        with pytest.raises(ValueError, match="2D"):
            mediation(x, m, y, x_nulls=xn[:, 0], y_nulls=yn)


class TestRecoversAKnownChain:
    def test_all_paths_detected(self, chain):
        res = mediation(*chain, n_boot=500)
        assert isinstance(res, MediationResult)
        assert res.a > 0.5 and res.a_p < 0.01
        assert res.b > 0.3 and res.b_p < 0.01
        assert res.indirect > 0.15 and res.indirect_p < 0.01
        assert res.limiting_path == "none"

    def test_bootstrap_interval_excludes_zero(self, chain):
        res = mediation(*chain, n_boot=2000)
        assert res.indirect_ci_lo > 0
        assert res.indirect_ci_lo < res.indirect < res.indirect_ci_hi

    def test_decomposition_holds_on_real_fit(self, chain):
        res = mediation(*chain, n_boot=0)
        assert res.c == pytest.approx(res.c_prime + res.indirect, abs=1e-10)

    def test_proportion_mediated_is_a_fraction_here(self, chain):
        res = mediation(*chain, n_boot=0)
        assert 0 < res.proportion_mediated <= 1.2


class TestIdentifiesTheFailingLink:
    """The behaviour that matters for reporting the real Phase 6 result."""

    def test_names_b_as_the_limiting_path(self, broken_b):
        res = mediation(*broken_b, n_boot=500)
        assert res.a_p < 0.01, "a should be strong by construction"
        assert res.b_p > 0.05, "b should be null by construction"
        assert res.limiting_path == "b"

    def test_indirect_effect_is_null(self, broken_b):
        res = mediation(*broken_b, n_boot=2000)
        assert res.indirect_p > 0.05
        assert res.indirect_ci_lo < 0 < res.indirect_ci_hi

    def test_product_null_alone_would_have_claimed_mediation(self, broken_b):
        """Why indirect_p is joint significance and not the product's own null.

        Here a is real and b is noise, so there is no mediation. The product
        null still rejects, because rotating the exposure destroys a and the
        product collapses whatever b was doing. This pins that the headline
        p-value does not inherit that behaviour.
        """
        res = mediation(*broken_b, n_boot=0)
        assert res.indirect_p_product < 0.05, "documents the failure mode"
        assert res.indirect_p > 0.05, "headline must not inherit it"

    def test_b_null_is_calibrated_under_collinearity(self):
        """The regression that forced rotating the outcome rather than the mediator.

        ``b`` is divided by ``1 - r_xm**2``, so it is variance-inflated whenever
        exposure and mediator overlap. The null has to carry the same inflation.
        Rotating the mediator does not — a surrogate mediator is uncorrelated
        with X, so its nulls are computed with a denominator near 1 while the
        observed value used 0.13. That made a b path which is null by
        construction report p ~ 0.002.

        Here X and M correlate at ~0.93 and Y is independent of both, so ``b``
        is null by construction. A calibrated test rejects at roughly the
        nominal rate; the mediator-rotation version rejected essentially always.
        """
        rng = np.random.default_rng(11)
        n, n_perm, n_trials = 100, 500, 200
        below = 0
        for _ in range(n_trials):
            x = rng.normal(size=n)
            m = 0.9 * x + rng.normal(size=n) * 0.3
            y = rng.normal(size=n)
            xn = np.column_stack([rng.permutation(x) for _ in range(n_perm)])
            yn = np.column_stack([rng.permutation(y) for _ in range(n_perm)])
            if mediation(x, m, y, xn, yn, n_boot=0).b_p < 0.05:
                below += 1
        rate = below / n_trials
        assert rate < 0.15, (
            f"b_p rejects at {rate:.3f} under a true null with r_xm~0.93; "
            "the null distribution is not matched to the observed coefficient"
        )

    def test_indirect_null_is_conservative(self):
        """Under a complete null the headline test must not over-reject.

        Because it is joint significance of two independent tests, the expected
        rejection rate is about alpha^2 = 0.0025 rather than alpha, so seeing
        zero rejections in a few hundred trials is the correct behaviour and not
        a sign the test is inert. That it *can* reject is covered by
        ``TestRecoversAKnownChain::test_all_paths_detected``.
        """
        rng = np.random.default_rng(12)
        n, n_perm, n_trials = 80, 500, 200
        below = 0
        for _ in range(n_trials):
            x = rng.normal(size=n)
            m = rng.normal(size=n)
            y = rng.normal(size=n)
            xn = np.column_stack([rng.permutation(x) for _ in range(n_perm)])
            yn = np.column_stack([rng.permutation(y) for _ in range(n_perm)])
            if mediation(x, m, y, xn, yn, n_boot=0).indirect_p < 0.05:
                below += 1
        rate = below / n_trials
        assert rate < 0.05, f"indirect_p rejects at {rate:.3f} under a total null"


class TestCovariates:
    def test_covariate_absorbs_a_confound(self):
        """X and Y share a driver; adjusting for it should remove the total effect."""
        rng = np.random.default_rng(3)
        n, n_perm = 120, 500
        conf = rng.normal(size=n)
        x = conf + rng.normal(size=n) * 0.3
        m = rng.normal(size=n)
        y = conf + rng.normal(size=n) * 0.3
        xn = np.column_stack([rng.permutation(x) for _ in range(n_perm)])
        yn = np.column_stack([rng.permutation(y) for _ in range(n_perm)])

        raw = mediation(x, m, y, xn, yn, n_boot=0)
        adj = mediation(x, m, y, xn, yn, covariates=conf, n_boot=0)
        assert abs(raw.c) > 0.6
        assert abs(adj.c) < 0.3
        assert adj.n_covariates == 1

    def test_accepts_2d_covariates(self, chain):
        x, m, y, xn, yn = chain
        cov = np.random.default_rng(4).normal(size=(100, 3))
        res = mediation(x, m, y, xn, yn, covariates=cov, n_boot=0)
        assert res.n_covariates == 3


class TestNaNAndDeterminism:
    def test_nan_parcels_dropped_consistently(self, chain):
        x, m, y, xn, yn = chain
        x = x.copy()
        x[:7] = np.nan
        res = mediation(x, m, y, xn, yn, n_boot=0)
        assert res.n_valid == 93
        # The decomposition must survive masking — it only holds if every path
        # is fit on the same parcels.
        assert res.c == pytest.approx(res.c_prime + res.indirect, abs=1e-10)

    def test_too_few_parcels_raises(self, chain):
        x, m, y, xn, yn = chain
        x = x.copy()
        x[5:] = np.nan
        with pytest.raises(ValueError, match="valid parcels"):
            mediation(x, m, y, xn, yn, n_boot=0)

    def test_seeded_bootstrap_reproduces(self, chain):
        """R7."""
        a = mediation(*chain, n_boot=500, seed=42)
        b = mediation(*chain, n_boot=500, seed=42)
        assert a == b

    def test_different_seed_changes_only_the_interval(self, chain):
        a = mediation(*chain, n_boot=500, seed=42)
        b = mediation(*chain, n_boot=500, seed=43)
        assert a.indirect_ci_lo != b.indirect_ci_lo
        assert a.indirect == pytest.approx(b.indirect)
        assert a.indirect_p == pytest.approx(b.indirect_p)

    def test_p_values_never_exactly_zero(self, chain):
        res = mediation(*chain, n_boot=0)
        for p in (res.a_p, res.b_p, res.c_p, res.c_prime_p, res.indirect_p):
            assert p > 0

    def test_serialisable_for_a_manifest(self, chain):
        import json

        res = mediation(*chain, n_boot=100)
        assert json.loads(json.dumps(res.as_dict()))["limiting_path"] in {
            "a",
            "b",
            "none",
        }


class TestRaggedSurrogates:
    """Surrogates with missing values — the case that broke Phase 6 on real data.

    A rotation can pull an unobserved parcel into the analysis window, so with a
    handful of missing parcels almost every one of 10,000 draws contains at least
    one NaN. An earlier version required every draw to be complete, which left
    zero usable parcels and fitted no models at all.
    """

    @pytest.fixture
    def ragged(self):
        rng = np.random.default_rng(21)
        n, n_perm = 100, 400
        x = rng.normal(size=n)
        m = 0.7 * x + rng.normal(size=n) * 0.7
        y = 0.6 * m + rng.normal(size=n) * 0.7
        # Three parcels unobserved in the exposure, as in a `missing=None` cell.
        x[[4, 17, 63]] = np.nan
        xn = np.column_stack([rng.permutation(x) for _ in range(n_perm)])
        yn = np.column_stack([rng.permutation(y) for _ in range(n_perm)])
        return x, m, y, xn, yn

    def test_fits_despite_nan_in_most_draws(self, ragged):
        _x, _m, _y, xn, _yn = ragged
        assert np.isfinite(xn).all(axis=0).mean() < 0.5, "fixture should be ragged"
        res = mediation(*ragged, n_boot=200)
        assert res.n_valid == 97
        assert np.isfinite(res.a) and np.isfinite(res.b)
        assert np.isfinite(res.indirect_p)

    def test_uses_most_of_the_draws(self, ragged):
        """Pairwise deletion should retain draws, not discard them."""
        res = mediation(*ragged, n_boot=0)
        assert res.n_perm_exposure > 0.9 * 400

    def test_still_recovers_the_chain(self, ragged):
        res = mediation(*ragged, n_boot=500)
        assert res.a > 0.4 and res.a_p < 0.01
        assert res.b > 0.3 and res.b_p < 0.01
        assert res.limiting_path == "none"

    def test_decomposition_still_exact(self, ragged):
        res = mediation(*ragged, n_boot=0)
        assert res.c == pytest.approx(res.c_prime + res.indirect, abs=1e-10)

    def test_ragged_with_covariates(self, ragged):
        x, m, y, xn, yn = ragged
        cov = np.random.default_rng(5).normal(size=(100, 2))
        res = mediation(x, m, y, xn, yn, covariates=cov, n_boot=0)
        assert res.n_covariates == 2
        assert np.isfinite(res.a) and np.isfinite(res.b_p)
        assert res.n_perm_exposure > 0.9 * 400

    def test_calibrated_when_ragged(self):
        """Missingness must not manufacture significance."""
        rng = np.random.default_rng(22)
        n, n_perm, n_trials = 100, 400, 150
        below = 0
        for _ in range(n_trials):
            x = rng.normal(size=n)
            x[rng.choice(n, 3, replace=False)] = np.nan
            m = rng.normal(size=n)
            y = rng.normal(size=n)
            xn = np.column_stack([rng.permutation(x) for _ in range(n_perm)])
            yn = np.column_stack([rng.permutation(y) for _ in range(n_perm)])
            if mediation(x, m, y, xn, yn, n_boot=0).a_p < 0.05:
                below += 1
        rate = below / n_trials
        assert 0.005 < rate < 0.15, f"a_p rejects at {rate:.3f} under a ragged null"
