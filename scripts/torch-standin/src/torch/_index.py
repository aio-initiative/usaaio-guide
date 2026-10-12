"""Reading and writing by position: x[index], x[index] = value, gather and scatter."""

import numpy as np

from ._dims import dim_index
from ._ops import data_of, result, write
from ._tensor import Tensor


def _raw(index):
    """One entry of a torch index as NumPy takes it: tensors and lists become arrays."""
    if isinstance(index, Tensor):
        index = index._data
    elif isinstance(index, list):
        index = np.asarray([_raw(item) for item in index])
    if isinstance(index, np.integer) or (
        isinstance(index, np.ndarray) and index.ndim == 0 and index.dtype.kind in "iu"
    ):
        return int(index)  # a 0-d integer tensor selects like a plain int
    return index


def _keys(index):
    """(ints, key): x[index] is x._data[key], or x._data[ints][key] when `ints` is not None.

    Two differences from NumPy are settled here. A key that selects single elements gets an
    Ellipsis appended, so the answer is a 0-d view instead of a detached NumPy scalar. And
    when a key mixes ints with index arrays (x[0, :, [1, 2]]), the ints are applied first, as
    torch does; NumPy alone would move the indexed dimension to the front.
    """
    key = tuple(_raw(item) for item in index) if isinstance(index, tuple) else (_raw(index),)
    ints = None
    if any(type(k) is int for k in key) and any(isinstance(k, np.ndarray) for k in key):
        ints, rest = [], []
        for k in key:
            if type(k) is int:
                ints.append(k)
            elif k is Ellipsis:
                ints.append(k)
                rest.append(k)
            elif k is None:
                rest.append(k)
            else:
                spans = k.ndim if isinstance(k, np.ndarray) and k.dtype.kind == "b" else 1
                ints.extend([slice(None)] * spans)
                rest.append(k)
        ints, key = tuple(ints), tuple(rest)
    if not any(k is Ellipsis for k in key):
        key += (Ellipsis,)
    return ints, key


def _advanced(key):
    return any(isinstance(k, np.ndarray) for k in key)


def select(input, ints, key):
    """input._data[ints][key] (see _keys) as an operation gradients flow through."""
    x = data_of(input)
    data = x if ints is None else x[ints]
    if _advanced(key):
        name = "IndexBackward0"
    else:
        name = "SelectBackward0" if any(type(k) is int for k in key) else "SliceBackward0"
    shape = x.shape
    return result(data[key], input, name=name, grads=(lambda g: put(shape, ints, key, g),))


def put(shape, ints, key, values):
    """Zeros of `shape` with `values` added at [ints][key], where repeated positions add up:
    the gradient of select, and itself differentiable (its gradient is select again)."""
    out = np.zeros(shape, dtype=values._data.dtype)
    target = out if ints is None else out[ints]
    if _advanced(key):
        np.add.at(target, key, values._data)
    else:
        target[key] += values._data
    return result(out, values, name="IndexPutBackward0", grads=(lambda g: select(g, ints, key),))


def take_flat(input, positions):
    """The values at the given row-major positions; the result has the shape of `positions`."""
    shape = data_of(input).shape
    data = input._data.reshape(-1)[positions]
    return result(data, input, name="TakeBackward0", grads=(lambda g: put_flat(shape, positions, g),))


def put_flat(shape, positions, values):
    """Zeros of `shape` with `values` added at the row-major `positions` (same shape as values)."""
    out = np.zeros(int(np.prod(shape)), dtype=values._data.dtype)
    np.add.at(out, positions.reshape(-1), values._data.reshape(-1))
    return result(
        out.reshape(shape), values, name="PutBackward0", grads=(lambda g: take_flat(g, positions),)
    )


def getitem(input, index):
    ints, key = _keys(index)
    return select(input, ints, key)


def setitem(input, index, value):
    ints, key = _keys(index)
    name = "IndexPutBackward0" if _advanced(key) else "CopySlices"
    write(input, key, value, ints=ints, indexed=True, name=name)


def _along(input, index, dim):
    """The NumPy key that reads x[..., index[position], ...] along `dim` at every position."""
    ndim = data_of(input).ndim
    dim = dim_index(dim, ndim)
    if index.ndim != ndim:
        raise RuntimeError("Index tensor must have the same number of dimensions as input tensor")
    key = list(np.indices(index.shape, sparse=True))
    key[dim] = index
    return tuple(key)


def gather(input, dim, index):
    """out[i][j] = input[index[i][j]][j] for dim 0, input[i][index[i][j]] for dim 1, and so on."""
    shape = data_of(input).shape
    return result(
        data_of(input)[_along(input, data_of(index), dim)], input, name="GatherBackward0", saved=(index,),
        grads=(lambda g, i: scatter_add(Tensor(np.zeros(shape, dtype=g._data.dtype)), dim, i, g),),
    )


def _source(src, index):
    """The values scatter writes: a number, or the corner of `src` that `index` covers."""
    if not isinstance(src, Tensor):
        return src
    return src._data[tuple(slice(0, n) for n in index.shape)]


def scatter(input, dim, index, src=None, *, value=None):
    """A copy of `input` with out[index[i][j]][j] = src[i][j] (dim 0); `src` may be a number."""
    src = value if src is None else src
    out = data_of(input).copy()
    out[_along(input, data_of(index), dim)] = _source(src, index._data)
    return result(
        out, input, src, name="ScatterBackward0", saved=(index,),
        grads=(lambda g, i: scatter(g, dim, i, 0), lambda g, i: gather(g, dim, i)),
    )


def scatter_add(input, dim, index, src):
    """Like scatter, but values that land on the same place are added together."""
    out = data_of(input).copy()
    np.add.at(out, _along(input, data_of(index), dim), _source(src, index._data))
    return result(
        out, input, src, name="ScatterAddBackward0", saved=(index,),
        grads=(lambda g, i: g, lambda g, i: gather(g, dim, i)),
    )
