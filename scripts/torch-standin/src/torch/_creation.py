"""Functions that build a tensor from Python data, a NumPy array or a shape."""

import numpy as np

from . import _dtype
from ._ops import data_of
from ._tensor import Tensor, as_size, require_cpu


def _is_plain(data):
    """True for Python numbers and (nested) lists of them, with no NumPy or tensor values."""
    if isinstance(data, (list, tuple, range)):
        return all(_is_plain(item) for item in data)
    return type(data) in (bool, int, float)


def _unwrapped(data):
    if isinstance(data, Tensor):
        return data._data
    if isinstance(data, (list, tuple)):
        return [_unwrapped(item) for item in data]
    return data


def _inferred(data, dtype=None):
    """`data` as an array of the dtype torch infers. Shares memory with a tensor or an array.

    Python floats become float32 and Python ints int64 (set explicitly: NumPy's default
    integer is 32-bit in the browser); NumPy values keep their own dtype, so float64 stays.
    With an explicit `dtype` the Python floats stay float64 (as torch converts each one
    straight to that dtype), so tensor([0.3], dtype=float64) holds the double 0.3, not the float32 nearest to it.
    """
    if isinstance(data, Tensor):
        return data._data
    if isinstance(data, (np.ndarray, np.generic)):
        array = np.asarray(data)
    elif _is_plain(data):
        array = np.array(data)
        if array.dtype.kind == "i":
            array = array.astype(_dtype.INT64)
        elif array.dtype.kind == "f" and dtype is None:
            array = array.astype(_dtype.FLOAT32)
    else:
        array = np.array(_unwrapped(data))
    _dtype.of(array.dtype)  # refuses strings, objects and the dtypes torch does not have here
    return array


def shape_from(args, size=None):
    """The shape in f(2, 3), f((2, 3)) or f(size=(2, 3)); a negative size is refused."""
    shape = as_size(args if size is None else (size,))
    if any(n < 0 for n in shape):
        raise RuntimeError(f"Trying to create tensor with negative dimension {min(shape)}: {list(shape)}")
    return shape


def _cast(array, dtype):
    return array if dtype is None else array.astype(_dtype.as_numpy(dtype), copy=False)


def tensor(data, dtype=None, device=None, requires_grad=False):
    """A new tensor holding a copy of `data`."""
    require_cpu(device)
    array = _inferred(data, dtype)
    return Tensor(np.array(_cast(array, dtype)), requires_grad=requires_grad)


def as_tensor(data, dtype=None, device=None):
    """`data` as a tensor, without copying when it already is a tensor or an array of that dtype."""
    require_cpu(device)
    if isinstance(data, Tensor) and (dtype is None or _dtype.as_numpy(dtype) == data._data.dtype):
        return data
    return Tensor(_cast(_inferred(data, dtype), dtype))


def from_numpy(ndarray):
    """A tensor that shares memory with the array."""
    if not isinstance(ndarray, np.ndarray):
        raise TypeError(f"expected np.ndarray (got {type(ndarray).__name__})")
    _dtype.of(ndarray.dtype)
    return Tensor(ndarray)


def is_tensor(obj):
    return isinstance(obj, Tensor)


def _fill_dtype(fill_value, dtype):
    """The dtype of full(size, fill_value): the fill value's own, as torch.tensor would infer it."""
    return _inferred(fill_value).dtype if dtype is None else _dtype.as_numpy(dtype)


def full(*args, size=None, fill_value=None, dtype=None, device=None, requires_grad=False):
    require_cpu(device)
    if fill_value is None:
        *args, fill_value = args
    if isinstance(fill_value, Tensor):
        fill_value = fill_value.item()
    array = np.full(shape_from(args, size), fill_value, dtype=_fill_dtype(fill_value, dtype))
    return Tensor(array, requires_grad=requires_grad)


def zeros(*args, size=None, dtype=None, device=None, requires_grad=False):
    return full(*args, size=size, fill_value=0, dtype=dtype or _dtype.float32, device=device, requires_grad=requires_grad)


def ones(*args, size=None, dtype=None, device=None, requires_grad=False):
    return full(*args, size=size, fill_value=1, dtype=dtype or _dtype.float32, device=device, requires_grad=requires_grad)


# Uninitialised memory has no use in a notebook, so empty tensors are simply zeros.
empty = zeros


def full_like(input, fill_value, dtype=None, device=None, requires_grad=False):
    x = data_of(input)
    return full(x.shape, fill_value, dtype=x.dtype if dtype is None else dtype, device=device, requires_grad=requires_grad)


def zeros_like(input, dtype=None, device=None, requires_grad=False):
    return full_like(input, 0, dtype=dtype, device=device, requires_grad=requires_grad)


def ones_like(input, dtype=None, device=None, requires_grad=False):
    return full_like(input, 1, dtype=dtype, device=device, requires_grad=requires_grad)


empty_like = zeros_like


def _number(value):
    return value.item() if isinstance(value, (Tensor, np.generic)) else value


def arange(*args, start=0, end=None, step=1, dtype=None, device=None, requires_grad=False):
    """arange(end), arange(start, end) or arange(start, end, step): int64 unless a float is given."""
    require_cpu(device)
    if len(args) == 1:
        end = args[0]
    elif args:
        start, end, step = (*args, step)[:3]
    start, end, step = _number(start), _number(end), _number(step)
    if step == 0:
        raise RuntimeError("step must be nonzero")
    whole = all(isinstance(value, int) for value in (start, end, step))
    array = np.arange(start, end, step, dtype=np.int64 if whole else np.float64)
    array = array.astype(_dtype.as_numpy(dtype or (_dtype.int64 if whole else _dtype.float32)))
    return Tensor(array, requires_grad=requires_grad)


def linspace(start, end, steps, dtype=None, device=None, requires_grad=False):
    require_cpu(device)
    array = np.linspace(_number(start), _number(end), steps)
    return Tensor(array.astype(_dtype.as_numpy(dtype or _dtype.float32)), requires_grad=requires_grad)


def eye(n, m=None, dtype=None, device=None, requires_grad=False):
    require_cpu(device)
    return Tensor(np.eye(n, m, dtype=_dtype.as_numpy(dtype or _dtype.float32)), requires_grad=requires_grad)
