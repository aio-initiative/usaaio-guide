"""Ordering: sort, argsort, topk, median, mode, quantile, unique and bincount.

The values these return are gathered from the input (see _reduce.pick), so gradients reach
the elements they came from; the indices, counts and unique values carry no gradient.
quantile's gradient goes to the two sorted neighbours it interpolates between.
"""

import numpy as np

from . import _dtype, return_types
from ._index import put_flat
from ._ops import add, constant, data_of, mul, numeric, result
from ._dims import axis_of
from ._reduce import as_vector, pick, reduction
from ._shape import permute, reshape
from ._tensor import Tensor


def _order(x, axis, descending):
    """Indices that sort x along axis. Equal values keep their order either way, as in
    torch's sort, and nan counts as the largest value."""
    if not descending:
        return np.argsort(x, axis=axis, kind="stable")
    # Sorting the reversed array and reversing the answer puts ties back in their first order.
    flipped = np.argsort(np.flip(x, axis), axis=axis, kind="stable")
    return x.shape[axis] - 1 - np.flip(flipped, axis)


@numeric
def sort(input, dim=-1, descending=False, stable=False):
    axis = axis_of(input, dim)
    index = _order(as_vector(data_of(input)), axis, descending)
    return return_types.sort(*pick(input, axis, index, keepdim=True))


@numeric
def argsort(input, dim=-1, descending=False, stable=False):
    x = data_of(input)
    index = _order(as_vector(x), axis_of(input, dim), descending)
    return Tensor(index.reshape(x.shape).astype(np.int64))


@numeric
def topk(input, k, dim=-1, largest=True, sorted=True):
    """The k largest (or smallest) values along dim, largest first, and their indices."""
    x, axis = as_vector(data_of(input)), axis_of(input, dim)
    if not 0 <= k <= x.shape[axis]:
        raise RuntimeError("selected index k out of range")
    index = np.take(_order(x, axis, largest), np.arange(k), axis=axis)
    return return_types.topk(*pick(input, axis, index, keepdim=True))


@reduction
def median(input, dim=None, keepdim=False):
    """The lower of the two middle values (no averaging), as in torch; nan if there is a nan.

    Over all values it returns the value alone; along dim, a (values, indices) pair.
    """
    x = data_of(input)
    if dim is None:
        if x.size == 0:
            shape = x.shape
            return result(
                np.full((), np.nan, x.dtype), input, name="MedianBackward0",
                grads=(lambda g: constant(np.zeros(shape, g._data.dtype)),),
            )
        return median(reshape(input, -1), 0).values
    axis = axis_of(input, dim)
    x = as_vector(x)
    middle = (x.shape[axis] - 1) // 2
    index = np.take(np.argsort(x, axis=axis, kind="stable"), [middle], axis=axis)
    if x.dtype.kind == "f":
        nan = np.isnan(x)
        index = np.where(nan.any(axis=axis, keepdims=True), np.argmax(nan, axis=axis, keepdims=True), index)
    return return_types.median(*pick(input, axis, index, keepdim))


@reduction
def mode(input, dim=-1, keepdim=False):
    """The most frequent value along dim (the smallest of equally frequent ones) and the index
    of its last occurrence, as torch reports it."""
    axis = axis_of(input, dim)
    rows = np.moveaxis(as_vector(data_of(input)), axis, -1)
    flat = rows.reshape(-1, rows.shape[-1])
    last = np.empty(len(flat), dtype=np.int64)
    for r, row in enumerate(flat):
        values, counts = np.unique(row, return_counts=True)
        top = values[np.argmax(counts)]
        last[r] = len(row) - 1 - np.argmax(row[::-1] == top)
    index = np.expand_dims(last.reshape(rows.shape[:-1]), axis)
    return return_types.mode(*pick(input, axis, index, keepdim))


QUANTILE_METHODS = ("linear", "lower", "higher", "midpoint", "nearest")


@reduction
def quantile(input, q, dim=None, keepdim=False, *, interpolation="linear"):
    """The q-th quantile(s) over all values or along dim; a tensor q adds a leading dimension.

    Computed as torch does, in the input's dtype: the rank q * (n - 1) into the sorted values,
    rounded for "lower", "higher" and "nearest" (half to even), and interpolated otherwise.
    """
    x = data_of(input)
    if x.dtype.kind != "f":
        raise RuntimeError("quantile() input tensor must be either float or double dtype")
    if x.size == 0:
        raise RuntimeError("quantile() input tensor must be non-empty")
    if isinstance(q, Tensor) and q._data.dtype != x.dtype:
        raise RuntimeError("quantile() q tensor must be same dtype as the input tensor")
    qs = np.asarray(q._data if isinstance(q, Tensor) else q, dtype=x.dtype)
    if np.any((qs < 0) | (qs > 1)):
        raise RuntimeError(f"quantile() q must be in the range [0, 1] but got {q}")
    if interpolation not in QUANTILE_METHODS:
        raise RuntimeError(f"quantile() interpolation must be one of {', '.join(QUANTILE_METHODS)}")
    axis = None if dim is None else axis_of(input, dim)

    def as_rows(array):  # the values to take quantiles of, one row each, along the last axis
        return array.reshape(-1) if axis is None else np.moveaxis(as_vector(array), axis, -1)

    order = np.argsort(as_rows(x), axis=-1, kind="stable")
    rows = np.take_along_axis(as_rows(x), order, axis=-1)
    # Where each sorted value sits in the input (row-major), for the gradient.
    sources = np.take_along_axis(as_rows(np.arange(x.size).reshape(x.shape)), order, axis=-1)
    ranks = qs * (rows.shape[-1] - 1)
    if interpolation != "linear" and interpolation != "midpoint":
        ranks = {"lower": np.floor, "higher": np.ceil, "nearest": np.round}[interpolation](ranks)
    low, high = np.floor(ranks).astype(np.intp), np.ceil(ranks).astype(np.intp)
    weight = np.full_like(ranks, 0.5) if interpolation == "midpoint" else ranks - low
    below, above = rows[..., low], rows[..., high]
    # torch's lerp, which is exact at both ends.
    out = np.where(weight < 0.5, below + weight * (above - below), above - (above - below) * (1 - weight))
    nan = np.isnan(rows).any(axis=-1)
    out = np.where(nan.reshape(nan.shape + (1,) * qs.ndim), np.nan, out).astype(x.dtype, copy=False)
    plain_shape = out.shape  # the rows' shape, then the quantiles'
    if keepdim:
        out = np.expand_dims(out, tuple(range(x.ndim)) if axis is None else axis)
    if qs.ndim:  # the quantiles come first, as in torch
        out = np.moveaxis(out, -1, 0)
    # Copies: a view would keep the whole `sources` array alive.
    lower, upper = np.array(sources[..., low]), np.array(sources[..., high])
    weight = weight.astype(x.dtype)

    def gradient(g):
        if qs.ndim:
            g = permute(g, *range(1, g._data.ndim), 0)
        g = reshape(g, *plain_shape)
        if interpolation in ("lower", "higher", "nearest"):  # torch picks one value, no lerp
            return put_flat(x.shape, lower, g)
        below = put_flat(x.shape, lower, mul(g, constant(1 - weight)))
        return add(below, put_flat(x.shape, upper, mul(g, constant(weight))))

    return result(out, input, name="QuantileBackward0", grads=(gradient,))


def unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
    """The distinct values in order (whole slices along dim if given), and optionally where
    each input value went and how often each distinct value occurs."""
    x = data_of(input)
    values, inverse, counts = np.unique(x, return_inverse=True, return_counts=True, axis=dim)
    inverse = inverse.reshape(x.shape if dim is None else -1)
    out = [Tensor(values)]
    if return_inverse:
        out.append(Tensor(inverse.astype(np.int64)))
    if return_counts:
        out.append(Tensor(counts.astype(np.int64)))
    return out[0] if len(out) == 1 else tuple(out)


def bincount(input, weights=None, minlength=0):
    """How often each of 0, 1, 2, ... occurs in a 1-d tensor of non-negative integers."""
    x = data_of(input)
    if x.dtype.kind not in "iu":
        raise NotImplementedError(f"\"bincount_cpu\" not implemented for '{_dtype.of(x.dtype).legacy}'")
    if x.ndim != 1 or (x.size and x.min() < 0):
        raise RuntimeError("bincount only supports 1-d non-negative integral inputs.")
    x = x.astype(np.intp, copy=False)  # np.bincount refuses int64 where pointers are 32-bit
    if weights is None:
        return Tensor(np.bincount(x, minlength=minlength).astype(np.int64))
    w = data_of(weights)
    dtype = w.dtype if w.dtype.kind == "f" else np.float64
    return Tensor(np.bincount(x, weights=w, minlength=minlength).astype(dtype))
