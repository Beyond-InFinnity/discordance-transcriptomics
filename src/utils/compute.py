"""Backend dispatch for the handful of operations large enough to justify a GPU.

CLAUDE.md §4.2 permits GPU work where the problem is dense linear algebra over
large arrays. In this project exactly one operation qualifies at present: the
gene-by-surrogate correlation block in the transcriptome screen, which at vertex
level is a (15,000 x 9,000) @ (9,000 x 10,000) product — about 2.7 TFLOP per
target map.

**The precision problem, measured rather than assumed.** §4.2 originally required
GPU results to be numerically identical to the CPU path. Benchmarking showed that
rule cannot be met at any speed advantage on consumer hardware:

    RTX 3070 Laptop, float32   6.50 TFLOPS
    RTX 3070 Laptop, float64   0.26 TFLOPS
    16-core CPU,     float64   0.26 TFLOPS

Double precision on a consumer card runs at 1/32 rate, which lands it exactly on
top of the CPU. So a bit-identical GPU path is possible and pointless: it buys
nothing. The entire advantage lives in float32, which is not bit-identical.

What float32 costs, concretely. The screen's inference is a comparison,
``|null rho| >= |observed rho|``. Correlations lie in [-1, 1] and float32 carries
about 1e-7 of relative error, so a comparison flips only when the two values sit
within ~1e-7 of each other. Across a full screen that is a handful of flipped
comparisons out of hundreds of millions, each shifting one gene's p-value by
1/n_perm — the fourth decimal place. It cannot move a p-value across 0.05 unless
the p-value was already sitting on 0.05 to four digits.

So the standard this module holds to is **identical decisions, with the
discrepancy measured**, not identical bits. :func:`validate_backend` performs
that measurement on real-shaped data and is exercised by the test suite. Use
``dtype='float64'`` when a result must be reproducible bit-for-bit and the cost
does not matter; that path is available and is the default.

**Blocking is not bit-neutral, and that touches R7.** A matrix product computed
in blocks does not agree bit-for-bit with the same product computed whole, on any
backend: BLAS selects different kernels and different accumulation orders by
shape. Measured here at 1e-14 on float64, affecting about 1% of entries — far
below anything that matters numerically, but it means *chunk size is part of the
computation*, not merely a memory knob.

Determinism therefore holds in the form R7 requires — same inputs and same chunk
give byte-identical output, verified in the tests — but a run that changes
``chunk`` is not reproducing an earlier one exactly. ``chunk`` is derived
deterministically from the array shape and dtype when not supplied, so the
default path is stable; anything that sets it explicitly should record it in the
manifest.

Nothing else in the pipeline belongs here. abagen is file I/O and pandas, surface
transforms shell out to Connectome Workbench, and the parcel-level spin test is a
(100 x 10,000) product already measured in microseconds — moving any of them to a
GPU would cost more in transfer than it saves in compute.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["available", "matmul_abs", "resolve", "validate_backend"]

Backend = Literal["numpy", "torch-cpu", "torch-cuda"]

# Env override, so a run can be pinned without touching call sites.
_ENV = "DISCORDANCE_BACKEND"


def _torch():
    try:
        import torch

        return torch
    except ImportError:
        return None


def available() -> list[str]:
    """Backends usable in this process, cheapest-to-set-up first."""
    out = ["numpy"]
    t = _torch()
    if t is not None:
        out.append("torch-cpu")
        if t.cuda.is_available():
            out.append("torch-cuda")
    return out


def resolve(prefer: str = "auto") -> str:
    """Pick a backend.

    ``auto`` takes the fastest available, unless ``DISCORDANCE_BACKEND`` is set.
    An explicit request for something unavailable falls back to numpy with a
    warning rather than failing — a missing GPU should slow a run down, not stop
    it.
    """
    env = os.environ.get(_ENV)
    if env:
        prefer = env
    have = available()
    if prefer == "auto":
        for b in ("torch-cuda", "numpy"):
            if b in have:
                return b
        return "numpy"
    if prefer not in have:
        logger.warning("backend %r unavailable (have %s); using numpy", prefer, have)
        return "numpy"
    return prefer


def matmul_abs(
    a: np.ndarray,
    b: np.ndarray,
    backend: str = "auto",
    dtype: str = "float64",
    chunk: int | None = None,
    reducer=None,
) -> Any:
    """``|a @ b|``, chunked over the rows of ``a``.

    Parameters
    ----------
    a : ndarray, shape (m, k)
    b : ndarray, shape (k, n)
    backend : str
        ``auto``, ``numpy``, ``torch-cpu`` or ``torch-cuda``.
    dtype : {'float64', 'float32'}
        ``float64`` reproduces the CPU result bit-for-bit and, on consumer
        cards, runs at CPU speed. ``float32`` is roughly 25x faster on a GPU and
        differs in the seventh decimal — see the module docstring for what that
        does and does not affect.
    chunk : int, optional
        Rows of ``a`` per block. Defaults to something that keeps a block near
        256 MB, which matters most on an 8 GB card.
    reducer : callable, optional
        Called as ``reducer(block, lo, hi)`` per block instead of accumulating
        the full result. Use this when the full (m, n) product would not fit —
        the screen only needs counts and maxima, never the product itself.

    Returns
    -------
    ndarray or None
        The full ``|a @ b|`` when ``reducer`` is None, otherwise None.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"expected 2D arrays, got {a.shape} and {b.shape}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"shape mismatch for a @ b: {a.shape} and {b.shape}")
    if dtype not in ("float64", "float32"):
        raise ValueError(f"dtype must be float64 or float32, got {dtype!r}")

    be = resolve(backend)
    m, n = a.shape[0], b.shape[1]
    if chunk is None:
        itemsize = 8 if dtype == "float64" else 4
        chunk = max(1, min(m, int(256e6 // max(1, n * itemsize))))

    out = None if reducer is not None else np.empty((m, n), dtype=dtype)

    if be.startswith("torch"):
        torch = _torch()
        dev = "cuda" if be == "torch-cuda" else "cpu"
        td = torch.float64 if dtype == "float64" else torch.float32
        bt = torch.from_numpy(np.ascontiguousarray(b)).to(dev, td)
        for lo in range(0, m, chunk):
            hi = min(lo + chunk, m)
            at = torch.from_numpy(np.ascontiguousarray(a[lo:hi])).to(dev, td)
            block = (at @ bt).abs().cpu().numpy()
            if reducer is not None:
                reducer(block, lo, hi)
            else:
                out[lo:hi] = block
            del at, block
        del bt
        if dev == "cuda":
            torch.cuda.empty_cache()
    else:
        an = a.astype(dtype, copy=False)
        bn = b.astype(dtype, copy=False)
        for lo in range(0, m, chunk):
            hi = min(lo + chunk, m)
            block = np.abs(an[lo:hi] @ bn)
            if reducer is not None:
                reducer(block, lo, hi)
            else:
                out[lo:hi] = block
    return out


def validate_backend(
    backend: str = "auto",
    shape: tuple[int, int, int] = (2000, 100, 2000),
    seed: int = 42,
) -> dict[str, float]:
    """Measure how far a backend/dtype combination departs from numpy float64.

    Reports the quantity that actually matters — whether the sign of a
    ``>=`` comparison changes — alongside the raw numerical difference, so the
    float32 decision is made against evidence rather than a rule of thumb.

    Returns
    -------
    dict
        ``max_abs_diff``, ``max_rel_diff``, ``n_decisions``,
        ``n_decisions_flipped``, ``frac_decisions_flipped``.
    """
    m, k, n = shape
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(m, k))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = rng.normal(size=(k, n))
    b /= np.linalg.norm(b, axis=0, keepdims=True)

    ref = matmul_abs(a, b, backend="numpy", dtype="float64")
    got = matmul_abs(a, b, backend=backend, dtype="float32")

    # The decision the screen actually makes: does each entry reach the row's
    # own observed value? Use the row median as a stand-in for that threshold.
    thr = np.median(ref, axis=1, keepdims=True)
    flipped = int(((ref >= thr) != (got >= thr)).sum())
    denom = np.maximum(np.abs(ref), 1e-12)
    return {
        "backend": resolve(backend),
        "max_abs_diff": float(np.max(np.abs(ref - got))),
        "max_rel_diff": float(np.max(np.abs(ref - got) / denom)),
        "n_decisions": int(ref.size),
        "n_decisions_flipped": flipped,
        "frac_decisions_flipped": flipped / ref.size,
    }
