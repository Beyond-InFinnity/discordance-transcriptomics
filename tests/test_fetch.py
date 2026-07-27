"""Tests for OpenNeuro fetching (no network — pure logic only)."""

from __future__ import annotations

import pytest

from src.data.fetch import DEFAULT_SNAPSHOT, RemoteFile, path_matches


class TestGlobMatching:
    """Regression tests for a real bug.

    The first implementation used ``fnmatch``, whose ``*`` matches ``/``. That
    made ``derivatives/*.nii.gz`` match every NIfTI at any depth — 19,075 files
    and 60 GB instead of the intended 14 files.
    """

    def test_star_does_not_cross_separator(self):
        assert path_matches("derivatives/a.nii.gz", ["derivatives/*.nii.gz"])
        assert not path_matches(
            "derivatives/sub-p019/qmri/a.nii.gz", ["derivatives/*.nii.gz"]
        )

    def test_doublestar_does_cross_separator(self):
        assert path_matches("derivatives/a/b/c/d.nii.gz", ["derivatives/**/d.nii.gz"])

    def test_mid_segment_wildcards(self):
        p = "derivatives/sub-p019/qmri/sub-p019_task-calc_space-MNI152_oef.nii.gz"
        assert path_matches(p, ["derivatives/*/qmri/*space-MNI152*"])
        assert not path_matches(p, ["derivatives/*/func/*space-MNI152*"])

    def test_excludes_the_huge_bold_files(self):
        """The 1 GB/subject fmriprep output must not be swept in by accident."""
        bold = "derivatives/sub-p019/func/sub-p019task-all_space-MNI152_res-2_desc-preproc_bold.nii.gz"
        assert not path_matches(bold, ["derivatives/*/qmri/*space-MNI152*"])
        assert not path_matches(bold, ["derivatives/*.nii.gz"])

    def test_question_mark_single_char(self):
        assert path_matches("a/b1.txt", ["a/b?.txt"])
        assert not path_matches("a/b12.txt", ["a/b?.txt"])

    def test_anchored_at_both_ends(self):
        assert not path_matches("x/derivatives/a.nii.gz", ["derivatives/*.nii.gz"])
        assert not path_matches("derivatives/a.nii.gz.bak", ["derivatives/*.nii.gz"])

    def test_empty_patterns_match_nothing(self):
        assert not path_matches("anything", [])

    def test_multiple_patterns_are_or(self):
        pats = ["derivatives/*.nii.gz", "derivatives/*/qmri/*oef*"]
        assert path_matches("derivatives/x.nii.gz", pats)
        assert path_matches("derivatives/sub-p019/qmri/y_oef.nii.gz", pats)
        assert not path_matches("derivatives/sub-p019/func/z.nii.gz", pats)


class TestRemoteFile:
    def test_parses_sha256_from_annex_key(self):
        sha = "a" * 64
        rf = RemoteFile(path="x.nii.gz", size=10, annex_key=f"SHA256E-s10--{sha}.nii.gz")
        assert rf.expected_sha256 == sha

    def test_real_annex_key(self):
        key = (
            "SHA256E-s822572--"
            "5139cfe1bfc8428c50ecd392fd677f6cc0677c52079ab8e7ad9f6a8ce4df737b.nii.gz"
        )
        rf = RemoteFile(path="x", size=822572, annex_key=key)
        assert rf.expected_sha256 == (
            "5139cfe1bfc8428c50ecd392fd677f6cc0677c52079ab8e7ad9f6a8ce4df737b"
        )

    def test_non_annex_key_yields_none(self):
        """Plain git blob SHAs are 40 hex chars and carry no content checksum."""
        rf = RemoteFile(path="x", size=1, annex_key="007f1e90e9d72e6569c145e3a02bd68e")
        assert rf.expected_sha256 is None

    def test_empty_key_yields_none(self):
        assert RemoteFile(path="x", size=1, annex_key="").expected_sha256 is None

    def test_url_path_uses_colon_separator(self):
        rf = RemoteFile(path="derivatives/sub-p019/qmri/a.nii.gz", size=1, annex_key="")
        assert rf.url_path == "derivatives:sub-p019:qmri:a.nii.gz"


class TestSnapshotPin:
    def test_snapshot_is_pinned_to_a_2x_release(self):
        """1.0.x snapshots contain no derivatives at all — see module docstring."""
        assert DEFAULT_SNAPSHOT.startswith("2."), (
            "ds004873 1.0.x has no derivatives; the pin must stay on a 2.x snapshot"
        )


@pytest.mark.parametrize(
    "path,expected",
    [
        ("derivatives/task-all_space-MNI152_res-2_SNR_YEO_group_mask.nii.gz", True),
        ("derivatives/N40_cond-control_space-MNI152_median_cbf.nii.gz", True),
        ("derivatives/sub-p019/qmri/sub-p019_task-calc_space-MNI152_oef.nii.gz", True),
        ("derivatives/sub-p019/qmri/sub-p019_task-calc_space-T1w_oef.nii.gz", False),
        ("derivatives/sub-p019/perf/sub-p019_task-control_acq-perf_dsc.nii.gz", False),
    ],
)
def test_phase0_pattern_set(path, expected):
    """The exact patterns used for the Phase 0 fetch."""
    patterns = ["derivatives/*.nii.gz", "derivatives/*/qmri/*space-MNI152*"]
    assert path_matches(path, patterns) is expected
