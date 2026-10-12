"""Comparisons and tests on values. None of these carries a gradient, so they build plain
tensors: a bool mask that can index, be summed, or be cast with .float()."""

import numpy as np

from . import _dtype
from ._ops import _operands, data_of, numeric
from ._tensor import Tensor


def _comparison(function):
    def compare(input, other):
        return Tensor(function(*_operands(input, other)))

    compare.__name__ = compare.__qualname__ = function.__name__
    return numeric(compare)


eq, ne, lt, le, gt, ge = map(
    _comparison, (np.equal, np.not_equal, np.less, np.less_equal, np.greater, np.greater_equal)
)
# On bool tensors these are the logical operations; on integer tensors they act bit by bit.
bitwise_and, bitwise_or, bitwise_xor = map(_comparison, (np.bitwise_and, np.bitwise_or, np.bitwise_xor))


def bitwise_not(input):
    x = data_of(input)
    if x.dtype.kind == "f":
        raise TypeError("~ (operator.invert) is only implemented on integer and Boolean-type tensors")
    return Tensor(np.invert(x))


def isnan(input):
    return Tensor(np.isnan(data_of(input)))


def isfinite(input):
    return Tensor(np.isfinite(data_of(input)))


def allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    """True when |input - other| <= atol + rtol * |other| everywhere. A Python bool."""
    x, y = data_of(input), data_of(other)
    if x.dtype != y.dtype:
        raise RuntimeError(f"{_dtype.of(x.dtype).legacy} did not match {_dtype.of(y.dtype).legacy}")
    return bool(np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan))


def isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Elementwise |input - other| <= atol + rtol * |other|, as a bool tensor. Both must be
    tensors of the same dtype, as in torch; an exact match (inf with inf) is always close."""
    if not isinstance(input, Tensor) or not isinstance(other, Tensor):
        raise TypeError("isclose(): argument 'other' (position 2) must be Tensor")
    x, y = data_of(input), data_of(other)
    if x.dtype != y.dtype:
        raise RuntimeError(f"{_dtype.of(x.dtype).legacy} did not match {_dtype.of(y.dtype).legacy}")
    if x.dtype.kind == "b":
        return Tensor(np.equal(x, y))
    return Tensor(np.isclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan))


def equal(input, other):
    """True when the two tensors have the same shape and the same values. A Python bool."""
    return bool(np.array_equal(data_of(input), data_of(other)))
