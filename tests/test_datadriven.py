"""Tests for the data-driven arm (CLAUDE.md §8.2).

The screen replaces 150 million scipy correlations with two matrix products, so
the first thing checked is that it agrees with scipy exactly. The second is
calibration: a transcriptome-wide screen that over-rejects turns 15,000 genes
into 15,000 chances to find nothing, and the whole value of this arm is that its
negative can be trusted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from src.expression.datadriven import (
    gene_screen,
    pls_with_spin,
    screen_summary,
    tail_enrichment,
)


@pytest.fixture
def data():
    """Expression, a target one gene genuinely drives, and target surrogates."""
    rng = np.random.default_rng(42)
    n_parcels, n_genes, n_perm = 60, 200, 400
    expr = pd.DataFrame(
        rng.normal(size=(n_parcels, n_genes)),
        columns=[f"G{i:03d}" for i in range(n_genes)],
    )
    # G000 drives the target; everything else is noise.
    target = expr["G000"].to_numpy() * 1.5 + rng.normal(size=n_parcels) * 0.5
    nulls = np.column_stack([rng.permutation(target) for _ in range(n_perm)])
    return expr, target, nulls


@pytest.fixture
def programme():
    """A block of 40 genes sharing a spatial programme with the target."""
    rng = np.random.default_rng(11)
    n_parcels, n_genes, n_perm = 60, 200, 400
    latent = rng.normal(size=n_parcels)
    G = rng.normal(size=(n_parcels, n_genes))
    G[:, :40] += latent[:, None] * 1.2
    expr = pd.DataFrame(G, columns=[f"G{i:03d}" for i in range(n_genes)])
    target = latent * 1.5 + rng.normal(size=n_parcels) * 0.5
    nulls = np.column_stack([rng.permutation(target) for _ in range(n_perm)])
    return expr, target, nulls


@pytest.fixture
def stability(data):
    expr, _, _ = data
    rng = np.random.default_rng(1)
    return pd.Series(rng.uniform(0, 0.6, expr.shape[1]), index=expr.columns)


class TestScreenCorrectness:
    def test_rho_matches_scipy_exactly(self, data):
        """The matrix-product shortcut must be exact, not approximate."""
        expr, target, nulls = data
        res = gene_screen(expr, target, nulls)
        for g in ["G000", "G007", "G199"]:
            expected = sps.spearmanr(expr[g].to_numpy(), target).statistic
            assert res.loc[g, "rho"] == pytest.approx(expected, abs=1e-12)

    def test_all_genes_returned_and_sorted(self, data):
        expr, target, nulls = data
        res = gene_screen(expr, target, nulls)
        assert len(res) == expr.shape[1]
        assert res.rho.is_monotonic_decreasing

    def test_chunking_does_not_change_the_answer(self, data):
        expr, target, nulls = data
        a = gene_screen(expr, target, nulls, chunk=17)
        b = gene_screen(expr, target, nulls, chunk=100_000)
        pd.testing.assert_frame_equal(a, b)

    def test_finds_the_planted_gene(self, data):
        expr, target, nulls = data
        res = gene_screen(expr, target, nulls)
        assert res.index[0] == "G000"
        # p_spin, not p_fdr: BH across the transcriptome cannot resolve below
        # n_genes/(n_perm+1), so the adjusted p is floored well above 0.05 here
        # even for a gene that drives the target outright. That is the
        # limitation test_per_gene_fdr_floor_is_reported exists to make visible.
        assert res.loc["G000", "p_spin"] == pytest.approx(1 / (nulls.shape[1] + 1))

    def test_constant_genes_dropped(self, data):
        expr, target, nulls = data
        expr = expr.copy()
        expr["FLAT"] = 1.0
        res = gene_screen(expr, target, nulls)
        assert "FLAT" not in res.index

    def test_p_never_exactly_zero(self, data):
        expr, target, nulls = data
        assert (gene_screen(expr, target, nulls).p_spin > 0).all()

    def test_deterministic(self, data):
        expr, target, nulls = data
        pd.testing.assert_frame_equal(
            gene_screen(expr, target, nulls), gene_screen(expr, target, nulls)
        )


class TestScreenR1Enforcement:
    def test_missing_nulls_raises(self, data):
        expr, target, _ = data
        with pytest.raises(ValueError, match="R1"):
            gene_screen(expr, target, None)

    def test_nulls_must_be_2d(self, data):
        expr, target, _ = data
        with pytest.raises(ValueError, match="2D"):
            gene_screen(expr, target, target.copy())

    def test_parcel_mismatch_raises(self, data):
        expr, target, nulls = data
        with pytest.raises(ValueError, match="parcels"):
            gene_screen(expr, target, nulls[:20])

    def test_expression_parcel_mismatch_raises(self, data):
        expr, target, nulls = data
        with pytest.raises(ValueError, match="parcels"):
            gene_screen(expr.iloc[:30], target, nulls)


class TestScreenCalibration:
    def test_false_positive_rate_near_nominal(self):
        """With no gene related to the target, ~5% of genes should reach p<0.05.

        This is the property that makes the arm's negative meaningful. Over the
        real transcriptome the observed rate is 3.9% against a 5% expectation.
        """
        rng = np.random.default_rng(7)
        n_parcels, n_genes, n_perm = 60, 3000, 500
        expr = pd.DataFrame(
            rng.normal(size=(n_parcels, n_genes)),
            columns=[f"G{i:04d}" for i in range(n_genes)],
        )
        target = rng.normal(size=n_parcels)
        nulls = np.column_stack([rng.permutation(target) for _ in range(n_perm)])
        res = gene_screen(expr, target, nulls)
        rate = float((res.p_spin < 0.05).mean())
        assert 0.02 < rate < 0.09, f"screen rejects at {rate:.3f}, nominal 0.05"

    def test_fdr_controls_when_nothing_is_real(self):
        rng = np.random.default_rng(8)
        n_parcels, n_genes, n_perm = 60, 2000, 500
        expr = pd.DataFrame(
            rng.normal(size=(n_parcels, n_genes)),
            columns=[f"G{i:04d}" for i in range(n_genes)],
        )
        target = rng.normal(size=n_parcels)
        nulls = np.column_stack([rng.permutation(target) for _ in range(n_perm)])
        res = gene_screen(expr, target, nulls)
        assert int((res.p_fdr < 0.05).sum()) <= 2


class TestTailEnrichment:
    def test_detects_a_set_planted_in_the_positive_tail(self, data, stability):
        expr, target, nulls = data
        screen = gene_screen(expr, target, nulls)
        top = list(screen.index[:10])
        out = tail_enrichment(
            screen, {"planted": top}, stability, n_draws=2000, tail_frac=0.05
        )
        pos = out[(out["name"] == "planted") & (out["tail"] == "positive")].iloc[0]
        assert pos.observed > pos.null_mean
        assert pos.p < 0.01

    def test_random_set_is_not_enriched(self, data, stability):
        expr, target, nulls = data
        screen = gene_screen(expr, target, nulls)
        rng = np.random.default_rng(3)
        rand = list(rng.choice(screen.index, 30, replace=False))
        out = tail_enrichment(screen, {"rand": rand}, stability, n_draws=2000)
        assert (out["p"] > 0.05).all()

    def test_both_tails_reported(self, data, stability):
        expr, target, nulls = data
        screen = gene_screen(expr, target, nulls)
        out = tail_enrichment(
            screen, {"s": list(screen.index[:20])}, stability, n_draws=500
        )
        assert set(out["tail"]) == {"positive", "negative"}

    def test_tiny_sets_skipped(self, data, stability):
        expr, target, nulls = data
        screen = gene_screen(expr, target, nulls)
        out = tail_enrichment(
            screen, {"tiny": list(screen.index[:2])}, stability, n_draws=100
        )
        assert out.empty

    def test_absent_genes_ignored(self, data, stability):
        expr, target, nulls = data
        screen = gene_screen(expr, target, nulls)
        members = [*list(screen.index[:8]), "NOT_A_GENE", "ALSO_NOT"]
        out = tail_enrichment(screen, {"s": members}, stability, n_draws=500)
        assert out.iloc[0].n_overlap == 8
        assert out.iloc[0].n_in_set == 10

    def test_deterministic(self, data, stability):
        expr, target, nulls = data
        screen = gene_screen(expr, target, nulls)
        kw = dict(gene_sets={"s": list(screen.index[:15])}, stability=stability)
        a = tail_enrichment(screen, n_draws=500, seed=42, **kw)
        b = tail_enrichment(screen, n_draws=500, seed=42, **kw)
        pd.testing.assert_frame_equal(a, b)

    def test_matching_uses_stability(self, data):
        """A set of only high-stability genes must be compared against
        high-stability draws, so the null mean tracks the set rather than the
        transcriptome average."""
        expr, target, nulls = data
        screen = gene_screen(expr, target, nulls)
        # Stability perfectly aligned with the screen ranking: the top of the
        # ranking is also the most stable. An unmatched null would call the top
        # genes wildly enriched; a matched one should not.
        stab = pd.Series(np.linspace(1.0, 0.0, len(screen)), index=screen.index).reindex(
            expr.columns
        )
        top = list(screen.index[:12])
        out = tail_enrichment(screen, {"top": top}, stab, n_draws=3000, n_bins=10)
        pos = out[out["tail"] == "positive"].iloc[0]
        # Matched draws come from the same stability band, i.e. also near the top.
        assert pos.null_mean > 0.1, (
            "matched null should itself land in the tail; if it does not, "
            "stability matching is not being applied"
        )


class TestPLS:
    def test_recovers_a_real_component(self, data):
        expr, target, nulls = data
        out = pls_with_spin(expr, target, nulls, n_components=2, max_perm=100)
        assert len(out) == 2
        assert out.iloc[0].r2_target > 0.5
        assert out.iloc[0].p_spin < 0.05

    def test_requires_nulls(self, data):
        expr, target, _ = data
        with pytest.raises(ValueError, match="R1"):
            pls_with_spin(expr, target, None)

    def test_null_target_is_not_significant(self, data):
        """PLS on 200 genes and 60 parcels overfits badly — r2 will be high even
        for an unrelated target. The spin test is what makes it interpretable,
        and this pins that it does its job."""
        expr, _, _ = data
        rng = np.random.default_rng(99)
        y = rng.normal(size=expr.shape[0])
        nulls = np.column_stack([rng.permutation(y) for _ in range(200)])
        out = pls_with_spin(expr, y, nulls, n_components=1, max_perm=200)
        assert out.iloc[0].r2_target > 0.3, "expected overfitting"
        assert out.iloc[0].p_spin > 0.05, "spin test should not be fooled by it"

    def test_records_shape(self, data):
        expr, target, nulls = data
        out = pls_with_spin(expr, target, nulls, n_components=1, max_perm=50)
        assert out.iloc[0].n_genes == expr.shape[1]
        assert out.iloc[0].n_parcels == expr.shape[0]
        assert out.iloc[0].n_perm == 50


class TestScreenSummary:
    """The transcriptome-level test, which exists because per-gene FDR cannot
    resolve at any affordable permutation budget."""

    def test_per_gene_fdr_floor_is_reported(self, data):
        expr, target, nulls = data
        res = gene_screen(expr, target, nulls)
        floor = res.attrs["fdr_floor"]
        assert floor == pytest.approx(expr.shape[1] / (nulls.shape[1] + 1))
        # Every adjusted p must sit at or above the floor, which is the whole
        # reason the column is not usable on its own.
        assert res.p_fdr.min() >= min(1.0, floor) - 1e-9

    def test_detects_a_real_programme(self, programme):
        """Sensitivity is to a *programme*, not to one gene.

        A single real gene among 200 raises the hit count by one against a null
        of about ten, which is correctly undetectable — that is what a
        transcriptome-level test is for. Here 40 genes share signal with the
        target and the excess is unmistakable.
        """
        expr, target, nulls = programme
        s = screen_summary(gene_screen(expr, target, nulls))
        assert s["n_observed"] > 2 * s["null_mean"]
        assert s["p"] < 0.05

    def test_single_gene_does_not_move_it(self, data):
        """The complement of the above, stated as a limitation rather than a bug."""
        expr, target, nulls = data
        s = screen_summary(gene_screen(expr, target, nulls))
        assert s["p"] > 0.05

    def test_no_excess_when_nothing_is_real(self):
        rng = np.random.default_rng(31)
        n_parcels, n_genes, n_perm = 60, 1500, 600
        expr = pd.DataFrame(
            rng.normal(size=(n_parcels, n_genes)),
            columns=[f"G{i:04d}" for i in range(n_genes)],
        )
        target = rng.normal(size=n_parcels)
        nulls = np.column_stack([rng.permutation(target) for _ in range(n_perm)])
        s = screen_summary(gene_screen(expr, target, nulls))
        assert s["p"] > 0.05, f"claimed an excess where none exists (p={s['p']})"

    def test_requires_attrs(self, data):
        expr, target, nulls = data
        res = gene_screen(expr, target, nulls)
        stripped = pd.DataFrame(res.to_dict())  # drops attrs, as a CSV round-trip does
        with pytest.raises(ValueError, match="attrs"):
            screen_summary(stripped)

    def test_serialisable_for_a_manifest(self, data):
        import json

        expr, target, nulls = data
        s = screen_summary(gene_screen(expr, target, nulls))
        assert json.loads(json.dumps(s))["n_genes"] == expr.shape[1]


class TestMaxTCorrection:
    """Westfall-Young max-T: the per-gene correction that actually resolves.

    BH across 15,000 genes at 10,000 rotations is floored above 1.0, so it can
    never flag anything. max-T compares each gene against the distribution of the
    strongest correlation anywhere in the transcriptome per rotation, which
    resolves to 1/(n_perm+1) whatever the gene count.
    """

    def test_resolves_below_the_fdr_floor(self, data):
        expr, target, nulls = data
        res = gene_screen(expr, target, nulls)
        assert res.attrs["fdr_floor"] > 0.05, "fixture should have a blocking floor"
        assert res.p_maxt.min() < 0.05, "max-T should still flag the planted gene"
        assert res.p_fdr.min() > 0.05, "BH should be unable to"

    def test_flags_the_planted_gene(self, data):
        expr, target, nulls = data
        res = gene_screen(expr, target, nulls)
        assert res.loc["G000", "p_maxt"] < 0.05

    def test_is_conservative_relative_to_the_uncorrected_p(self, data):
        expr, target, nulls = data
        res = gene_screen(expr, target, nulls)
        assert (res.p_maxt >= res.p_spin - 1e-12).all()

    def test_controls_fwer_when_nothing_is_real(self):
        """Across independent trials, the chance of *any* gene being flagged
        should sit near alpha — that is what family-wise control means."""
        rng = np.random.default_rng(41)
        n_parcels, n_genes, n_perm, n_trials = 50, 400, 400, 60
        any_flagged = 0
        for _ in range(n_trials):
            expr = pd.DataFrame(
                rng.normal(size=(n_parcels, n_genes)),
                columns=[f"G{i:03d}" for i in range(n_genes)],
            )
            target = rng.normal(size=n_parcels)
            nulls = np.column_stack([rng.permutation(target) for _ in range(n_perm)])
            if (gene_screen(expr, target, nulls).p_maxt < 0.05).any():
                any_flagged += 1
        rate = any_flagged / n_trials
        assert rate < 0.20, f"family-wise error {rate:.3f} at nominal 0.05"

    def test_monotone_in_effect_size(self, data):
        expr, target, nulls = data
        res = gene_screen(expr, target, nulls).sort_values("rho", key=abs)
        assert res.p_maxt.is_monotonic_decreasing

    def test_reported_in_summary(self, data):
        expr, target, nulls = data
        s = screen_summary(gene_screen(expr, target, nulls))
        assert s["n_genes_maxt_below_05"] >= 1
        assert 0 < s["min_p_maxt"] <= 1
