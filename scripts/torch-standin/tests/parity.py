"""Helpers that run the same code on the stand-in and on real torch and compare the results."""

import numpy as np
import pytest
import torch

RNG = np.random.default_rng(0)
A = RNG.standard_normal((3, 4)).astype(np.float32)
B = RNG.standard_normal((3, 4)).astype(np.float32)
V = RNG.standard_normal(4).astype(np.float32)
POS = np.abs(A) + 0.5
INT = RNG.integers(-5, 6, (3, 4))
CUBE = RNG.standard_normal((2, 3, 4)).astype(np.float32)
SQUARE = RNG.standard_normal((4, 4)).astype(np.float32)
MASK = A > 0
# Repeated values, so ties in max, sort, mode and unique are exercised.
TIES = np.array([[1.0, 5.0, 5.0, 2.0], [3.0, 3.0, 0.0, 3.0]], dtype=np.float32)


def same(got, want):
    """`got` (stand-in) equals `want` (real torch): same kind, shape, dtype and values."""
    if isinstance(want, torch.Tensor):
        assert type(got).__name__ in ("Tensor", "Parameter"), f"expected a tensor, got {type(got)}"
        assert tuple(got.shape) == tuple(want.shape)
        assert str(got.dtype) == str(want.dtype)
        np.testing.assert_allclose(got.numpy(), want.numpy(), rtol=1e-5, atol=1e-6)
    elif isinstance(want, (tuple, list)):
        assert isinstance(got, (tuple, list)), f"expected a sequence, got {type(got)}"
        assert isinstance(got, tuple) == isinstance(want, tuple)
        assert len(got) == len(want)
        for g, w in zip(got, want):
            same(g, w)
    elif isinstance(want, float):
        assert type(got) is float
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    else:
        assert type(got) is type(want), f"{type(got)} is not {type(want)}"
        assert got == want


def check(st, real, fn, *arrays):
    """Call fn(torch, *tensors) on both libraries, tensors built from the same arrays."""
    got = fn(st, *[st.tensor(a) for a in arrays])
    want = fn(real, *[real.tensor(a) for a in arrays])
    same(got, want)


def check_raises(st, real, fn, *arrays):
    """fn must fail on both libraries with exactly the same exception type."""
    with pytest.raises(Exception) as want:
        fn(real, *[real.tensor(a) for a in arrays])
    with pytest.raises(Exception) as got:
        fn(st, *[st.tensor(a) for a in arrays])
    # Exact, because the not-available NotImplementedError is itself a RuntimeError.
    assert type(got.value) is type(want.value), f"{got.value!r} is not a {type(want.value).__name__}"


def agree(run, st, real):
    """run(torch) gives the same result on both libraries, or fails the same way on both."""
    try:
        want = run(real)
    except Exception as err:
        with pytest.raises(Exception) as got:
            run(st)
        assert type(got.value) is type(err), f"{got.value!r} is not a {type(err).__name__}"
    else:
        same(run(st), want)


def cases(*items):
    """Parametrize over (name, fn, *arrays) tuples, using the name as the test id."""
    return pytest.mark.parametrize("case", items, ids=[item[0] for item in items])
