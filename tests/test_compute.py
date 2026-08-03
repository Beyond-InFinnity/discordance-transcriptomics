"""Tests for the GPU backend dispatch (CLAUDE.md §4.2).

The rule this file enforces is not "the GPU gives the same bits" — measurement
showed that standard costs the entire speed advantage, because float64 on a
consumer card runs at CPU speed. The rule is that float64 *is* exact when asked
for, and that float32 changes decisions at a rate small enough to be irrelevant
to any p-value, verified rather than asserted.

Everything here skips cleanly when torch or a GPU is absent, so the suite still
passes on a machine with neither.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.compute import available, matmul_abs, resolve, validate_backend

HAS_TORCH = "torch-cpu" in available()
HAS_CUDA = "torch-cuda" in available()


@pytest.fixture
def pair():
    rng = np.random.default_rng(42)
    a = rng.normal(size=(200, 60))
    b = rng.normal(size=(60, 300))
    return a, b


class TestDispatch:
    def test_numpy_always_available(self):
        assert "numpy" in available()

    def test_auto_resolves_to_something_available(self):
        assert resolve("auto") in available()

    def test_unavailable_backend_falls_back(self):
        """A missing GPU should slow a run down, not stop it."""
        assert resolve("torch-rocm-imaginary") == "numpy"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("DISCORDANCE_BACKEND", "numpy")
        assert resolve("torch-cuda") == "numpy"

    def test_rejects_bad_dtype(self, pair):
        a, b = pair
        with pytest.raises(ValueError, match="float64 or float32"):
            matmul_abs(a, b, dtype="float16")

    def test_rejects_shape_mismatch(self, pair):
        a, _ = pair
        with pytest.raises(ValueError, match="shape mismatch"):
            matmul_abs(a, a)

    def test_rejects_non_2d(self, pair):
        a, b = pair
        with pytest.raises(ValueError, match="2D"):
            matmul_abs(a[0], b)


class TestCorrectness:
    def test_numpy_matches_plain_expression(self, pair):
        a, b = pair
        np.testing.assert_allclose(matmul_abs(a, b, backend="numpy"), np.abs(a @ b))

    def test_chunking_is_numerically_transparent_but_not_bitwise(self, pair):
        """Blocking changes bits, and that is worth pinning rather than hiding.

        BLAS picks different kernels and accumulation orders by shape, so a
        blocked product differs from a whole one at ~1e-14 on about 1% of
        entries. Numerically irrelevant, but it means chunk size is part of the
        computation — see the R7 note in the module docstring.
        """
        a, b = pair
        small = matmul_abs(a, b, backend="numpy", chunk=7)
        whole = matmul_abs(a, b, backend="numpy", chunk=10_000)
        np.testing.assert_allclose(small, whole, rtol=1e-12, atol=1e-12)
        assert not np.array_equal(small, whole), (
            "if this ever passes bitwise the docstring's R7 caveat is obsolete"
        )

    def test_same_chunk_is_byte_identical(self, pair):
        """R7 in the form that actually matters: fixed inputs, fixed chunk."""
        a, b = pair
        np.testing.assert_array_equal(
            matmul_abs(a, b, backend="numpy", chunk=7),
            matmul_abs(a, b, backend="numpy", chunk=7),
        )

    def test_default_chunk_is_deterministic(self, pair):
        """The default is derived from shape and dtype, so it cannot drift."""
        a, b = pair
        np.testing.assert_array_equal(
            matmul_abs(a, b, backend="numpy"), matmul_abs(a, b, backend="numpy")
        )

    def test_reducer_sees_every_row_exactly_once(self, pair):
        a, b = pair
        seen = np.zeros(a.shape[0], dtype=int)
        acc = {}

        def reducer(block, lo, hi):
            seen[lo:hi] += 1
            acc[lo] = block.copy()

        assert matmul_abs(a, b, backend="numpy", chunk=13, reducer=reducer) is None
        assert (seen == 1).all()
        rebuilt = np.vstack([acc[k] for k in sorted(acc)])
        np.testing.assert_allclose(rebuilt, np.abs(a @ b))

    def test_reducer_returns_nothing(self, pair):
        a, b = pair
        assert matmul_abs(a, b, backend="numpy", reducer=lambda *_: None) is None


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestTorchAgreesWithNumpy:
    def test_float64_cpu_agrees_to_rounding(self, pair):
        """float64 on CPU must agree with numpy to rounding, not bit-for-bit.

        This asserted ``assert_array_equal`` — bit-identity — and passed on the
        laptop while failing on the workstation, which is the machine that
        actually runs the pipeline: 1,677 of 60,000 elements differed, by at most
        3.7e-13 relative. Nothing is wrong. torch and numpy dispatch to different
        BLAS builds, which block and thread a matmul differently, and float64
        addition is not associative, so the summation order decides the last few
        ulps. Two independent implementations cannot be held to bit-identity on
        any machine, let alone the same result across machines.

        CLAUDE.md §4.2 already withdrew that requirement — "the standard is
        identical decisions with the discrepancy measured, not identical bits" —
        after benchmarking showed a bit-identical GPU path was achievable only at
        no speed advantage. This test was left behind from the earlier draft, and
        it is the same reordering argument the CUDA test below already makes.

        The tolerance is far tighter than anything that could change a decision:
        the flip-rate assertion elsewhere in this file is what guards that.
        """
        a, b = pair
        np.testing.assert_allclose(
            matmul_abs(a, b, backend="torch-cpu", dtype="float64"),
            matmul_abs(a, b, backend="numpy", dtype="float64"),
            rtol=1e-11,
            atol=1e-11,
        )

    @pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device")
    def test_float64_cuda_is_near_exact(self, pair):
        """A GPU may reorder a reduction, so allow the last couple of ulps."""
        a, b = pair
        np.testing.assert_allclose(
            matmul_abs(a, b, backend="torch-cuda", dtype="float64"),
            matmul_abs(a, b, backend="numpy", dtype="float64"),
            rtol=1e-12,
            atol=1e-12,
        )

    @pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device")
    def test_float32_absolute_error_is_tiny(self, pair):
        a, b = pair
        ref = matmul_abs(a, b, backend="numpy", dtype="float64")
        got = matmul_abs(a, b, backend="torch-cuda", dtype="float32")
        assert np.max(np.abs(ref - got)) < 1e-4

    @pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device")
    def test_float32_decision_flip_rate_is_negligible(self):
        """The claim CLAUDE.md §4.2 rests on, checked at realistic scale.

        A flipped comparison moves one gene's permutation p-value by 1/n_perm.
        Below 1e-5 of comparisons, that cannot move any p-value across a
        threshold unless it was already sitting on it to four digits.
        """
        v = validate_backend("torch-cuda", shape=(2000, 100, 5000))
        assert v["frac_decisions_flipped"] < 1e-5, v
        assert v["max_abs_diff"] < 1e-5, v

    @pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device")
    def test_float32_reducer_path_matches_full_path(self, pair):
        a, b = pair
        full = matmul_abs(a, b, backend="torch-cuda", dtype="float32")
        parts = {}
        matmul_abs(
            a,
            b,
            backend="torch-cuda",
            dtype="float32",
            chunk=11,
            reducer=lambda blk, lo, hi: parts.__setitem__(lo, blk.copy()),
        )
        np.testing.assert_allclose(
            np.vstack([parts[k] for k in sorted(parts)]), full, rtol=1e-5, atol=1e-5
        )


class TestValidateBackend:
    def test_reports_the_expected_fields(self):
        v = validate_backend("numpy", shape=(100, 30, 100))
        for k in (
            "max_abs_diff",
            "max_rel_diff",
            "n_decisions",
            "n_decisions_flipped",
            "frac_decisions_flipped",
        ):
            assert k in v

    def test_numpy_against_itself_is_not_bit_identical_only_because_of_dtype(self):
        """float32 differs from float64 even on CPU — this is about precision,
        not about GPUs, and the validator should show that."""
        v = validate_backend("numpy", shape=(200, 50, 200))
        assert v["max_abs_diff"] > 0
        assert v["frac_decisions_flipped"] < 1e-3

    def test_deterministic(self):
        assert validate_backend("numpy", shape=(100, 30, 100), seed=7) == (
            validate_backend("numpy", shape=(100, 30, 100), seed=7)
        )
