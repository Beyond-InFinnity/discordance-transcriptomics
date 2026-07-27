"""Tests for target-map construction and the dataset inspector."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.targets import (
    DERIVATIVE_PATTERNS,
    VALID_RANGES,
    coupling_ratio_angle,
    coupling_ratio_signed_log,
    discordance_fraction,
    inspect_derivatives,
    load_dropout_proxy,
    load_subject_target_matrix,
)


class TestCouplingAngle:
    """§7.3 — the angle handling of n = %ΔCBF / %ΔCMRO2."""

    def test_concordant_positive_is_first_quadrant(self):
        # Both increase: classic concordant activation.
        a = coupling_ratio_angle(np.array([2.0]), np.array([1.0]))
        assert 0 < a[0] < np.pi / 2

    def test_concordant_negative_is_third_quadrant(self):
        a = coupling_ratio_angle(np.array([-2.0]), np.array([-1.0]))
        assert -np.pi < a[0] < -np.pi / 2

    def test_discordant_is_second_or_fourth_quadrant(self):
        # CBF up, CMRO2 down — the Epp et al. discordant case.
        up_down = coupling_ratio_angle(np.array([2.0]), np.array([-1.0]))
        assert np.pi / 2 < up_down[0] < np.pi
        down_up = coupling_ratio_angle(np.array([-2.0]), np.array([1.0]))
        assert -np.pi / 2 < down_up[0] < 0

    def test_bounded_even_at_zero_denominator(self):
        """The whole point: the angle survives ΔCMRO2 -> 0, the ratio does not."""
        a = coupling_ratio_angle(np.array([1.0, 1.0]), np.array([0.0, 1e-12]))
        assert np.all(np.isfinite(a))
        assert np.allclose(a, np.pi / 2)

    def test_is_continuous_through_the_origin(self):
        eps = np.array([1e-9, -1e-9])
        a = coupling_ratio_angle(np.array([1.0, 1.0]), eps)
        assert abs(a[0] - a[1]) < 1e-6

    def test_range(self):
        rng = np.random.default_rng(0)
        a = coupling_ratio_angle(rng.normal(size=500), rng.normal(size=500))
        assert np.all(a > -np.pi - 1e-12) and np.all(a <= np.pi + 1e-12)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            coupling_ratio_angle(np.zeros(5), np.zeros(6))


class TestCouplingSignedLog:
    def test_matches_ratio_sign(self):
        v = coupling_ratio_signed_log(np.array([2.0, -2.0]), np.array([1.0, 1.0]))
        assert v[0] > 0 and v[1] < 0

    def test_compresses_large_ratios(self):
        small = coupling_ratio_signed_log(np.array([2.0]), np.array([1.0]))
        large = coupling_ratio_signed_log(np.array([200.0]), np.array([1.0]))
        assert large[0] > small[0]
        assert large[0] < 10  # compressed, not exploded

    def test_small_denominator_becomes_nan_not_clipped(self):
        """Clipping would invent a finite ratio the data cannot support."""
        v = coupling_ratio_signed_log(np.array([1.0]), np.array([1e-9]), eps=1e-3)
        assert np.isnan(v[0])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            coupling_ratio_signed_log(np.zeros(5), np.zeros(6))

    def test_agrees_with_angle_on_ordering(self):
        """Sensitivity check should rank parcels similarly to the primary."""
        rng = np.random.default_rng(1)
        cbf = rng.normal(size=300)
        cmro2 = rng.normal(size=300) + 2.0  # keep denominator away from zero
        from scipy.stats import spearmanr

        ang = coupling_ratio_angle(cbf, cmro2)
        slog = coupling_ratio_signed_log(cbf, cmro2)
        ok = np.isfinite(ang) & np.isfinite(slog)
        assert spearmanr(ang[ok], slog[ok]).statistic > 0.9


class TestWiringState:
    """§13.5 — what is wired, and what must stay unwired until Phase 1."""

    def test_phase0_targets_are_wired(self):
        for key in (
            "baseline_oef",
            "calc_cbf",
            "calc_cmro2",
            "control_cbf",
            "control_cmro2",
            "snr_mask",
            "t2star",
        ):
            assert DERIVATIVE_PATTERNS[key] is not None, f"{key} should be wired"

    def test_discordance_freq_stays_unwired(self):
        """It needs the Phase 1 contrast-structure answer. `control` is probably
        the control condition of the calc task, not a co-equal second task — in
        which case a frequency map is undefined, not merely coarser."""
        assert DERIVATIVE_PATTERNS["discordance_freq"] is None

    def test_loading_discordance_freq_refuses(self):
        with pytest.raises(NotImplementedError, match=r"13\.5"):
            load_subject_target_matrix(
                None, target="discordance_freq", parcellation="schaefer200x7"
            )

    def test_unknown_target_refuses(self):
        with pytest.raises(NotImplementedError, match="not wired"):
            load_subject_target_matrix(
                None, target="made_up", parcellation="schaefer200x7"
            )

    def test_unknown_dropout_proxy_refuses(self):
        with pytest.raises(ValueError, match="unknown dropout proxy"):
            load_dropout_proxy(None, proxy="tsnr", parcellation="schaefer200x7")

    def test_oef_range_does_not_clip_the_authors_cap(self):
        """Regression: OEF must NOT be limited to (0, 1).

        Phase 1 established the published maps already carry the authors' cap
        (5 x subject median, floored at 1.5, clipped rather than excluded), so
        values above 1.0 are legitimate data covering 8-18% of voxels. An
        earlier (0, 1) bound discarded them and biased parcel means downward in
        exactly the high-OEF regions H1 concerns.
        """
        lo, hi = VALID_RANGES["oef"]
        assert lo == 0.0
        assert hi >= 3.0, "must sit above the largest observed 5x-median cap"


class TestDiscordanceFraction:
    """Discordance = coupling ratio n < 1, via sign(dBOLD) = sign(dCBF - dCMRO2)."""

    def test_cmro2_up_cbf_down_is_discordant(self):
        # BOLD falls (CBF cannot supply), CMRO2 rises -> opposite.
        f, n = discordance_fraction(np.array([[-1.0]]), np.array([[5.0]]))
        assert f[0] == 1.0 and n[0] == 1

    def test_cmro2_down_cbf_up_is_discordant(self):
        f, _ = discordance_fraction(np.array([[5.0]]), np.array([[-1.0]]))
        assert f[0] == 1.0

    def test_same_sign_with_n_below_one_is_discordant(self):
        """The mode the earlier opposite-sign definition missed entirely:
        both rise, but CMRO2 outpaces CBF, so BOLD falls while CMRO2 rises."""
        f, _ = discordance_fraction(np.array([[2.0]]), np.array([[5.0]]))
        assert f[0] == 1.0

    def test_same_sign_with_n_above_one_is_concordant(self):
        # Classic activation: CBF rises far more than CMRO2 -> BOLD up, CMRO2 up.
        f, _ = discordance_fraction(np.array([[10.0]]), np.array([[3.0]]))
        assert f[0] == 0.0

    def test_both_down_with_n_above_one_is_concordant(self):
        f, _ = discordance_fraction(np.array([[-10.0]]), np.array([[-3.0]]))
        assert f[0] == 0.0

    def test_fraction_across_subjects(self):
        d_cbf = np.array([[10.0], [2.0], [10.0], [2.0]])
        d_cmro2 = np.array([[3.0], [5.0], [3.0], [5.0]])
        f, n = discordance_fraction(d_cbf, d_cmro2)
        assert f[0] == 0.5 and n[0] == 4

    def test_nan_subjects_excluded_from_denominator(self):
        d_cbf = np.array([[2.0], [np.nan], [2.0]])
        d_cmro2 = np.array([[5.0], [5.0], [5.0]])
        f, n = discordance_fraction(d_cbf, d_cmro2)
        assert n[0] == 2 and f[0] == 1.0

    def test_min_pct_screens_small_responses(self):
        d_cbf = np.array([[0.1], [2.0]])
        d_cmro2 = np.array([[0.2], [5.0]])
        _, n = discordance_fraction(d_cbf, d_cmro2, min_pct=1.0)
        assert n[0] == 1

    def test_all_nan_parcel_is_nan_not_zero(self):
        f, n = discordance_fraction(np.array([[np.nan]]), np.array([[np.nan]]))
        assert np.isnan(f[0]) and n[0] == 0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            discordance_fraction(np.zeros((2, 3)), np.zeros((2, 4)))


class TestInspector:
    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="MANIFEST"):
            inspect_derivatives(tmp_path / "nope")

    def test_groups_by_suffix_and_finds_entities(self, tmp_path):
        for rel in [
            "derivatives/sub-01/anat/sub-01_OEF.nii.gz",
            "derivatives/sub-02/anat/sub-02_OEF.nii.gz",
            "derivatives/sub-01/func/sub-01_task-rest_bold.nii.gz",
            "derivatives/sub-01/func/sub-01_task-nback_CMRO2.nii.gz",
        ]:
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"")

        rep = inspect_derivatives(tmp_path)
        assert rep["n_nifti"] == 4
        assert rep["n_subjects"] == 2
        assert rep["tasks"] == ["nback", "rest"]
        assert rep["suffixes"]["OEF"]["count"] == 2
        assert "OEF" in rep["protocol_check"]["baseline_oef"]
        assert "CMRO2" in rep["protocol_check"]["task_cmro2"]

    def test_finds_snr_mask_as_dropout_proxy(self, tmp_path):
        """ds004873's SNR map has BIDS suffix 'mask'; only the full name says SNR."""
        p = tmp_path / "derivatives/task-all_space-MNI152_res-2_SNR_YEO_group_mask.nii.gz"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"")
        assert "mask" in inspect_derivatives(tmp_path)["protocol_check"]["dropout_proxy"]

    def test_cbv_substring_does_not_false_match(self, tmp_path):
        """Regression: 'GMR2pCBVmasked_cbf' must not register as a CBV map."""
        p = tmp_path / "derivatives/N40_cond-control_median_GMR2pCBVmasked_cbf.nii.gz"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"")
        check = inspect_derivatives(tmp_path)["protocol_check"]
        assert check["baseline_cbf"] == ["cbf"]
        assert check["baseline_cbv"] == []

    def test_reports_missing_requirements(self, tmp_path):
        p = tmp_path / "sub-01/anat/sub-01_T1w.nii.gz"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"")
        rep = inspect_derivatives(tmp_path)
        # No OEF anywhere -> empty candidate list, which the report flags.
        assert rep["protocol_check"]["baseline_oef"] == []
