"""In-place updates (the methods whose names end in an underscore, and +=, -=, *=, /=).

Each one computes its answer with the ordinary operation and then stores it through
_ops.write, so views of the tensor see the change and the tensor keeps its identity. The
tensor takes the history of that answer (x.mul_(y) gets a MulBackward0). An operation whose
gradient reads its input (mul_, div_) reads _autograd.original(x): a copy of x's values from
before the change when the step is recorded, as torch saves them.
"""

import numpy as np

from . import _autograd, _dtype, _index, _ops
from ._tensor import Tensor


def _store(target, value):
    """Finish target.op_(...): `value` is the out-of-place answer. Like torch, refuse an
    answer of a higher kind than the tensor (a float into an int tensor) or of another shape."""
    have, want = value._data.dtype, target._data.dtype
    if have != want and not np.can_cast(have, want, "same_kind"):
        raise RuntimeError(
            f"result type {_dtype.of(have).legacy} can't be cast to the desired output type {_dtype.of(want).legacy}"
        )
    if value._data.shape != target._data.shape:
        raise RuntimeError(
            f"output with shape {list(target._data.shape)} doesn't match the broadcast shape {list(value._data.shape)}"
        )
    return _ops.write(target, ..., value, adopt=True)


def add_(input, other, *, alpha=1):
    return _store(input, _ops.add(input, other, alpha=alpha))


def sub_(input, other):
    return _store(input, _ops.sub(input, other))


def mul_(input, other):
    return _store(input, _ops.mul(_autograd.original(input, other), other))


def div_(input, other):
    return _store(input, _ops.div(_autograd.original(input, other), other))


def zero_(input):
    return _ops.write(input, ..., 0, name="ZeroBackward0")


def fill_(input, value):
    """Set every value to a number or a 0-d tensor (which then gets the gradient of them all)."""
    if isinstance(value, Tensor):
        if value._data.ndim:
            raise RuntimeError(
                f"fill_ only supports 0-dimension value tensor but got tensor with {value._data.ndim} dimensions."
            )
        return _ops.write(input, ..., value, name="FillBackward3")
    return _ops.write(input, ..., value, name="FillBackward2")


def copy_(input, src):
    """Copy the values of `src`, broadcast to this tensor's shape and cast to its dtype."""
    return _ops.write(input, ..., src, name="CopyBackwards")


def scatter_add_(input, dim, index, src):
    return _store(input, _index.scatter_add(input, dim, index, src))


def fill_diagonal_(input, fill_value):
    """Set the main diagonal; a matrix that is not square is filled up to its shorter side."""
    steps = np.arange(min(input._data.shape))
    return _ops.write(input, (steps,) * input._data.ndim, fill_value)
