"""Operations that rearrange, join or split tensors without changing their values.

Each gradient puts the values back where they came from: reshape back, permute by the
inverse order, flip and roll back, slice a join apart. expand's gradient is the incoming one
as it is, because autograd sums it back over the expanded dimensions.
"""

import math

import numpy as np

from . import _dtype
from ._dims import dim_index
from ._index import getitem, put, select
from ._ops import data_of, result, sub
from ._tensor import as_size


def reshape(input, *shape):
    """The same values in a new shape (one size may be -1). Shares memory when it can."""
    return _reshaped(input, as_size(shape), "ViewBackward0")


def _reshaped(input, shape, name):
    x = data_of(input)
    try:
        out = x.reshape(shape)
    except ValueError:
        raise RuntimeError(f"shape '{list(shape)}' is invalid for input of size {x.size}") from None
    old = x.shape
    return result(out, input, name=name, grads=(lambda g: reshape(g, *old),))


def view(input, *shape):
    """reshape that refuses to copy: when the values are not laid out in order (after a
    transpose, an expand or a column slice) torch raises and asks for reshape or contiguous."""
    out = reshape(input, *shape)
    if out._data.size and not np.may_share_memory(out._data, data_of(input)):
        raise RuntimeError(
            "view size is not compatible with input tensor's size and stride (at least one dimension "
            "spans across two contiguous subspaces). Use .reshape(...) instead."
        )
    return out


def is_contiguous(input):
    """True when the values sit in memory in row-major order with no gaps."""
    return bool(data_of(input).flags.c_contiguous)


def contiguous(input):
    """The tensor itself when it is contiguous, otherwise a row-major copy."""
    x = data_of(input)
    if x.flags.c_contiguous:
        return input
    return result(np.ascontiguousarray(x), input, name="CloneBackward0", grads=(lambda g: g,))


def flatten(input, start_dim=0, end_dim=-1):
    shape = data_of(input).shape
    start, end = dim_index(start_dim, len(shape)), dim_index(end_dim, len(shape))
    if start > end:
        raise RuntimeError("flatten() has invalid args: start_dim cannot come after end_dim")
    if not shape:
        return reshape(input, 1)
    return reshape(input, *shape[:start], -1, *shape[end + 1 :])


def ravel(input):
    return reshape(input, -1)


def squeeze(input, dim=None):
    """Drop dimensions of size 1: all of them, or only those in `dim` (an int or a tuple);
    a listed dimension that is longer than 1 is left alone."""
    shape = data_of(input).shape
    if dim is None:
        dims = range(len(shape))
    else:
        dims = [dim_index(d, len(shape)) for d in (dim if isinstance(dim, (tuple, list)) else (dim,))]
    return _reshaped(input, tuple(n for d, n in enumerate(shape) if n != 1 or d not in dims), "SqueezeBackward0")


def unsqueeze(input, dim):
    """Insert a dimension of size 1 at `dim`, which may be one past the last dimension."""
    shape = list(data_of(input).shape)
    shape.insert(dim_index(dim, len(shape), extra=1), 1)
    return _reshaped(input, tuple(shape), "UnsqueezeBackward0")


def permute(input, *dims):
    x = data_of(input)
    order = [dim_index(d, x.ndim) for d in as_size(dims)]
    if len(order) != x.ndim:
        raise RuntimeError(
            "permute(sparse_coo): number of dimensions in the tensor input does not match the length of "
            f"the desired ordering of dimensions i.e. input.dim() = {x.ndim} is not equal to len(dims) = {len(order)}"
        )
    if len(set(order)) != len(order):
        raise RuntimeError("permute(): duplicate dims are not allowed.")
    back = [int(d) for d in np.argsort(order)]
    return result(x.transpose(order), input, name="PermuteBackward0", grads=(lambda g: permute(g, *back),))


def transpose(input, dim0, dim1):
    ndim = data_of(input).ndim
    dim0, dim1 = dim_index(dim0, ndim), dim_index(dim1, ndim)
    if ndim == 0:
        return permute(input)
    order = list(range(ndim))
    order[dim0], order[dim1] = order[dim1], order[dim0]
    return permute(input, *order)


def expand(input, *sizes):
    """Repeat dimensions of size 1 without copying; -1 keeps a dimension as it is."""
    x, sizes = data_of(input), as_size(sizes)
    old = (1,) * (len(sizes) - x.ndim) + x.shape
    shape = tuple(have if want == -1 else want for want, have in zip(sizes, old))
    try:
        out = np.broadcast_to(x, shape)
    except ValueError:
        raise RuntimeError(
            f"The expanded size of the tensor {list(sizes)} must match the existing size {list(x.shape)}"
        ) from None
    return result(out, input, name="ExpandBackward0", grads=(lambda g: g,))


def flip(input, *args, dims=None):
    """Reverse the order along the given dimensions: flip(0, 1), flip([0, 1]) or flip(dims=...)."""
    x = data_of(input)
    dims = tuple(dim_index(d, x.ndim) for d in as_size(args if dims is None else (dims,)))
    return result(np.flip(x, dims).copy(), input, name="FlipBackward0", grads=(lambda g: flip(g, dims),))


def roll(input, shifts, dims=None):
    """Shift the values along `dims`, wrapping round; with no dims the tensor is rolled flat."""
    x = data_of(input)
    if dims is not None:
        dims = tuple(dim_index(d, x.ndim) for d in as_size((dims,)))
    back = tuple(-s for s in shifts) if isinstance(shifts, (tuple, list)) else -shifts
    return result(np.roll(x, shifts, dims), input, name="RollBackward0", grads=(lambda g: roll(g, back, dims),))


def triu(input, diagonal=0):
    return result(
        np.triu(data_of(input), diagonal), input, name="TriuBackward0", grads=(lambda g: triu(g, diagonal),)
    )


def tril(input, diagonal=0):
    return result(
        np.tril(data_of(input), diagonal), input, name="TrilBackward0", grads=(lambda g: tril(g, diagonal),)
    )


def diag(input, diagonal=0):
    """The diagonal of a matrix as a vector, or a vector laid out as a diagonal matrix."""
    x = data_of(input)
    out = np.diag(x, diagonal).copy()
    if x.ndim == 1:
        return result(out, input, name="DiagEmbedBackward0", grads=(lambda g: diag(g, diagonal),))
    steps = np.arange(out.shape[0])
    rows, cols = (steps, steps + diagonal) if diagonal >= 0 else (steps - diagonal, steps)
    shape = x.shape
    return result(
        out, input, name="DiagonalBackward0", grads=(lambda g: put(shape, None, (rows, cols, Ellipsis), g),)
    )


def _span(input, dim, start, stop):
    key = [slice(None)] * data_of(input).ndim
    key[dim_index(dim, len(key))] = slice(start, stop)
    return getitem(input, tuple(key))


def diff(input, n=1, dim=-1):
    """Differences of neighbours along `dim`, taken n times."""
    for _ in range(n):
        input = sub(_span(input, dim, 1, None), _span(input, dim, None, -1))
    return input


def _joined(join, tensors, dim, extra):
    """Join along `dim`, checked against the first tensor (`extra` is 1 for a new dimension)."""
    arrays = [data_of(tensor) for tensor in tensors]
    dim = dim_index(dim, arrays[0].ndim, extra) if arrays else dim
    dtype = arrays[0].dtype
    for array in arrays[1:]:
        dtype = _dtype.promote(dtype, array.dtype)
    try:
        out = join([a.astype(dtype, copy=False) for a in arrays], axis=dim)
    except ValueError as err:
        raise RuntimeError(f"Sizes of tensors must match except in dimension {dim}: {err}") from None
    # Each input's gradient is its own part of the output's: a slice along dim for cat,
    # one position along the new dim for stack.
    keys, start = [], 0
    for array in arrays:
        place = [slice(None)] * out.ndim
        if join is np.stack:
            place[dim] = len(keys)
        else:
            place[dim] = slice(start, start + array.shape[dim])
            start += array.shape[dim]
        keys.append(tuple(place) + (Ellipsis,))
    grads = tuple(lambda g, key=key: select(g, None, key) for key in keys)
    name = "StackBackward0" if join is np.stack else "CatBackward0"
    return result(out, *tensors, name=name, grads=grads)


def cat(tensors, dim=0):
    """Join along an existing dimension. Empty 1-d tensors are skipped, as in torch."""
    kept = [tensor for tensor in tensors if data_of(tensor).shape != (0,)]
    return _joined(np.concatenate, kept or list(tensors), dim, 0)


def stack(tensors, dim=0):
    """Join along a new dimension; every tensor must have the shape of the first."""
    tensors = list(tensors)
    if not tensors:
        raise RuntimeError("stack expects a non-empty TensorList")
    first = data_of(tensors[0]).shape
    dim_index(dim, len(first), 1)  # torch checks the dim before the shapes
    for entry, tensor in enumerate(tensors):
        shape = data_of(tensor).shape
        if shape != first:
            raise RuntimeError(
                f"stack expects each tensor to be equal size, but got {list(first)} at entry 0 "
                f"and {list(shape)} at entry {entry}"
            )
    return _joined(np.stack, tensors, dim, 1)


def split(input, split_size_or_sections, dim=0):
    """Pieces of the given size (the last may be shorter), or of the listed sizes."""
    dim = dim_index(dim, data_of(input).ndim)
    length = data_of(input).shape[dim]
    sizes = split_size_or_sections
    if isinstance(sizes, int):
        sizes = [min(sizes, length - start) for start in range(0, length, sizes)]
    elif sum(sizes) != length:
        raise RuntimeError(
            f"split_with_sizes expects split_sizes to sum exactly to {length} "
            f"(input tensor's size at dimension {dim}), but got split_sizes={list(sizes)}"
        )
    ends = np.cumsum(sizes).tolist()
    return tuple(_span(input, dim, end - size, end) for size, end in zip(sizes, ends))


def chunk(input, chunks, dim=0):
    """At most `chunks` pieces of equal size (the last may be shorter)."""
    dim = dim_index(dim, data_of(input).ndim)
    return split(input, math.ceil(data_of(input).shape[dim] / chunks), dim)


def unbind(input, dim=0):
    """The slices along `dim`, as a tuple."""
    key = [slice(None)] * data_of(input).ndim
    dim = dim_index(dim, len(key))
    pieces = []
    for i in range(input._data.shape[dim]):
        key[dim] = i
        pieces.append(getitem(input, tuple(key)))
    return tuple(pieces)
