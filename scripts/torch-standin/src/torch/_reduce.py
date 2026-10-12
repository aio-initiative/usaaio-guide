"""Reductions over all dimensions or some: sum, mean, max, std, norm, logsumexp, softmax.

Each takes dim (an int, a tuple or list of ints, or None for all of them) and keepdim, and
also NumPy's names axis and keepdims, as torch does. sum, mean, amax, amin, cumsum, cumprod
and norm are computed here and return result(...) with their gradients; the others are
written with those and the elementwise operations (var is a mean, a subtraction and a sum),
so their gradients follow. std and norm guard their gradient where the result is 0, where
torch gives 0 rather than the inf or nan the plain formula would.
"""

import builtins
import functools
import math
import operator
import warnings

import numpy as np

from . import _dtype, return_types
from ._dims import axis_of, dims_of
from ._index import gather
from ._missing import not_available
from ._autograd import OUTPUT
from ._ops import (
    abs,
    add,
    cast,
    constant,
    data_of,
    div,
    exp,
    log,
    masked,
    maximum,
    minimum,
    mul,
    numeric,
    pow,
    result,
    shared,
    sub,
    where,
)
from ._shape import diag, expand, flip, reshape, squeeze
from ._tensor import Tensor


def reduction(function):
    """Accept axis and keepdims for dim and keepdim, and fail the way torch does."""

    @functools.wraps(function)
    def run(*args, axis=None, keepdims=None, **kwargs):
        if axis is not None:
            kwargs["dim"] = axis
        if keepdims is not None:
            kwargs["keepdim"] = keepdims
        return function(*args, **kwargs)

    return numeric(run)


def nonempty(input, axis, name):
    """Refuse to reduce along an empty dimension, as torch does for max, argmax and the rest."""
    if data_of(input).ndim and data_of(input).shape[axis] == 0:
        raise IndexError(f"{name}(): Expected reduction dim {axis} to have non-zero size.")


def as_vector(x):
    """A 0-d array as a 1-element vector, so operations along one axis work on it too."""
    return x if x.ndim else x.reshape(1)


def pick(input, axis, index, keepdim):
    """(values, indices) at positions `index` along `axis`; `index` keeps that axis.

    The values are gathered from the input, so gradients reach the picked elements.
    """
    index = index.astype(np.int64, copy=False)  # NumPy's indices are 32-bit in the browser
    if data_of(input).ndim == 0:
        return reshape(gather(reshape(input, 1), 0, Tensor(index)), ()), Tensor(index.reshape(()))
    values, indices = gather(input, axis, Tensor(index)), Tensor(index)
    if not keepdim:
        values, indices = squeeze(values, axis), squeeze(indices, axis)
    return values, indices


def _require_float(input, message):
    x = data_of(input)
    if x.dtype.kind != "f":
        raise RuntimeError(message.format(_dtype.of(x.dtype).legacy))


def _count(input, axes):
    shape = data_of(input).shape
    return math.prod(shape if axes is None else [shape[a] for a in axes])


def _kept(shape, axes):
    """`shape` with the reduced axes kept as 1s (keepdim=True)."""
    return tuple(1 if axes is None or a in axes else n for a, n in enumerate(shape))


def _spread(g, shape, axes):
    """The gradient of a reduction over `axes`, copied back over the values it reduced."""
    return expand(reshape(g, *_kept(shape, axes)), *shape)


def _safe_div(numerator, denominator):
    """numerator / denominator, and 0 where the denominator is 0. The 0s are replaced before
    dividing, so no inf appears, not even in a second derivative."""
    zero = denominator._data == 0
    return where(constant(zero), 0, div(numerator, where(constant(zero), 1, denominator)))


@reduction
def sum(input, dim=None, keepdim=False, *, dtype=None):
    """Integers and bools add up as int64, as in torch."""
    if dtype is not None:
        input = cast(input, _dtype.as_numpy(dtype))
    elif data_of(input).dtype.kind in "biu":
        input = cast(input, _dtype.INT64)
    axes, shape = dims_of(input, dim), input._data.shape
    return result(
        np.sum(input._data, axis=axes, keepdims=keepdim), input, name="SumBackward1" if dim is not None else "SumBackward0",
        grads=(lambda g: _spread(g, shape, axes),),
    )


@reduction
def mean(input, dim=None, keepdim=False, *, dtype=None):
    if dtype is not None:
        input = cast(input, _dtype.as_numpy(dtype))
    _require_float(
        input,
        "mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: {}",
    )
    axes, shape = dims_of(input, dim), input._data.shape
    count = _count(input, axes)
    # A sum and a division: np.mean warns about an empty slice, where torch quietly gives nan.
    return result(
        np.sum(input._data, axis=axes, keepdims=keepdim) / count, input, name="MeanBackward0",
        grads=(lambda g: div(_spread(g, shape, axes), count),),
    )


def _extreme_value(function, public, name):
    def extreme(input, dim=(), keepdim=False):
        """The gradient is shared evenly between the values that tie for the extreme, as in
        torch, which multiplies by the share (so an inf gradient gives nan off the ties)."""
        x, axes = data_of(input), dims_of(input, dim)
        peak = function(x, axis=axes, keepdims=True)
        hits = x == peak
        share = constant((hits / np.sum(hits, axis=axes, keepdims=True)).astype(x.dtype, copy=False))
        out = peak if keepdim else function(x, axis=axes)
        return result(out, input, name=name, grads=(lambda g: mul(_spread(g, x.shape, axes), share),))

    extreme.__name__ = extreme.__qualname__ = public
    return reduction(extreme)


amax = _extreme_value(np.amax, "amax", "AmaxBackward0")
amin = _extreme_value(np.amin, "amin", "AminBackward0")


def _whole_extreme(function, name):
    def extreme(input):
        """max() or min() of every value: the gradient is shared evenly between the values
        that tie (the nans, if the answer is nan) and is exactly 0 elsewhere, as in torch."""
        x = data_of(input)
        out = function(x)
        hits = np.isnan(x) if x.dtype.kind == "f" and np.isnan(out) else x == out
        share = hits / np.sum(hits)
        return result(out, input, name=name, grads=(lambda g: shared(share, expand(g, *x.shape)),))

    return extreme


def _extreme(name, whole, elementwise, arg):
    def extreme(input, dim=None, keepdim=False):
        if isinstance(dim, Tensor):  # torch.max(a, b) is the elementwise maximum
            return elementwise(input, dim)
        if dim is None:
            if data_of(input).size == 0:
                raise RuntimeError(
                    f"{name}(): Expected reduction dim to be specified for input.numel() == 0. "
                    "Specify the reduction dim with the 'dim' argument."
                )
            return whole(input)
        axis = axis_of(input, dim)
        nonempty(input, axis, name)
        index = arg(as_vector(input._data), axis=axis, keepdims=True)
        return getattr(return_types, name)(*pick(input, axis, index, keepdim))

    extreme.__name__ = extreme.__qualname__ = name
    extreme.__doc__ = f"The {name} of all values, or a ({name} values, their indices) pair along dim."
    return reduction(extreme)


max = _extreme("max", _whole_extreme(np.amax, "MaxBackward1"), maximum, np.argmax)
min = _extreme("min", _whole_extreme(np.amin, "MinBackward1"), minimum, np.argmin)


def _arg(function):
    def arg(input, dim=None, keepdim=False):
        """The index of the first extreme value; over all values it indexes the flattened tensor."""
        x = data_of(input)
        if dim is None:
            index = np.asarray(function(x))
            return Tensor(index.reshape((1,) * x.ndim if keepdim else ()).astype(np.int64))
        axis = axis_of(input, dim)
        nonempty(input, axis, function.__name__)
        index = function(as_vector(x), axis=axis, keepdims=keepdim)
        return Tensor((index.reshape(()) if x.ndim == 0 else index).astype(np.int64))

    arg.__name__ = arg.__qualname__ = function.__name__
    return reduction(arg)


argmax, argmin = _arg(np.argmax), _arg(np.argmin)


def _logical(function):
    def test(input, dim=None, keepdim=False):
        x = data_of(input)
        out = function(x, axis=dims_of(input, dim), keepdims=keepdim)
        return Tensor(np.asarray(out).astype(np.uint8 if x.dtype == np.uint8 else np.bool_))

    test.__name__ = test.__qualname__ = function.__name__
    return reduction(test)


all, any = _logical(np.all), _logical(np.any)


@reduction
def var(input, dim=None, unbiased=True, keepdim=False, *, correction=None):
    """The variance, divided by n - 1 unless unbiased=False or another correction is given."""
    if isinstance(dim, bool):  # var(False) is var(unbiased=False)
        dim, unbiased = None, dim
    _require_float(input, "std and var only support floating point and complex dtypes")
    axes = dims_of(input, dim)
    if correction is None:
        correction = 1 if unbiased else 0
    freedom = _count(input, axes) - correction
    if freedom <= 0:
        warnings.warn(
            "var(): degrees of freedom is <= 0. Correction should be strictly less than the "
            "reduction factor (input numel divided by output numel).",
            stacklevel=3,
        )
    x = input._data
    centred = x - np.sum(x, axis=axes, keepdims=True) / _count(input, axes)
    out = np.sum(centred * centred, axis=axes, keepdims=keepdim) / builtins.max(0, freedom)
    factor = 2.0 / freedom if freedom else math.inf
    shape = x.shape
    # torch's formula, 2 / (n - correction) * g * (x - mean): the mean's own gradient is 0,
    # so it is left out rather than computed (it would turn an inf gradient into nan).
    return result(
        out, input, name="VarBackward0", saved=(input,),
        grads=(lambda g, x: mul(mul(_spread(g, shape, axes), factor), sub(x, mean(x, axes, True))),),
    )


@reduction
def std(input, dim=None, unbiased=True, keepdim=False, *, correction=None):
    """The square root of var, with torch's gradient of 0 where the deviation is 0."""
    deviation = var(input, dim, unbiased, keepdim, correction=correction)
    return result(
        np.sqrt(deviation._data), deviation, name="StdBackward0", saved=(OUTPUT,),
        grads=(lambda g, out: _safe_div(g, mul(out, 2)),),
    )


def _p_norm(input, p, axes, keepdim, distance=False):
    """(sum |x|^p)^(1/p), whose gradient is g * sign(x) |x|^(p-1) / norm^(p-1), 0 where the
    norm is 0. `distance` is cdist's form of it, which divides g first for p=2 as well."""
    x = input._data
    if p == 2:
        out = np.sqrt(np.sum(x * x, axis=axes, keepdims=True))
    else:
        out = np.sum(np.abs(x) ** p, axis=axes, keepdims=True) ** (1 / p)

    def gradient(g, x, out):
        spread, peak = _spread(g, x._data.shape, axes), reshape(out, *_kept(x._data.shape, axes))
        # As torch writes them: for p=2 the gradient multiplies x / norm (0 where the norm is
        # 0); otherwise it multiplies g / norm^(p-1), 0 where the norm is 0.
        if p == 2 and not distance:
            return mul(spread, _safe_div(x, peak))
        if p == 2:
            return mul(x, _safe_div(spread, peak))
        slope = mul(constant(np.sign(x._data)), pow(abs(x), p - 1))
        return mul(slope, _safe_div(spread, pow(peak, p - 1)))

    if not keepdim:
        out = out.reshape(()) if axes is None else np.squeeze(out, axis=axes)
    return result(
        out, input, name="LinalgVectorNormBackward0", saved=(input, OUTPUT), grads=(gradient,)
    )


def distance(differences, p):
    """cdist's p-norm of the differences along the last dim, with torch's cdist gradient."""
    if p == 2 or (p not in (0, 1) and math.isfinite(p)):
        return _p_norm(differences, p, (differences._data.ndim - 1,), False, distance=True)
    return norm(differences, p, dim=-1)


@reduction
def norm(input, p="fro", dim=None, keepdim=False, dtype=None):
    """The p-norm of the values as one vector (or of each slice along dim)."""
    if dtype is not None:
        input = cast(input, _dtype.as_numpy(dtype))
    _require_float(input, "linalg.vector_norm: Expected a floating point or complex tensor as input. Got {}")
    axes = dims_of(input, dim)
    if p in ("fro", 2):
        return _p_norm(input, 2, axes, keepdim)
    if p == 1:
        return sum(abs(input), axes, keepdim)
    if p == math.inf:
        return amax(abs(input), axes, keepdim)
    if p == -math.inf:
        return amin(abs(input), axes, keepdim)
    if p == 0:
        x = data_of(input)
        return Tensor(np.sum(x != 0, axis=axes, keepdims=keepdim).astype(x.dtype))
    if isinstance(p, str):
        raise not_available(f"torch.norm(p={p!r})")
    return _p_norm(input, p, axes, keepdim)


def trace(input):
    """The sum of the main diagonal of a matrix."""
    x = data_of(input)
    if x.ndim != 2:
        raise RuntimeError(f"trace: expected a matrix, but got tensor with dim {x.ndim}")
    if x.dtype.kind == "b":
        raise NotImplementedError("\"trace\" not implemented for 'Bool'")
    return sum(diag(input))


def _reverse_cumsum(g, axis):
    """sum over j >= i of g[j], along axis: what each position contributes to a running sum."""
    return flip(cumsum(flip(g, axis), axis), axis)


def _cumprod_grad(axis):
    def gradient(g, x, out):
        """d out[j] / d x[i] is out[j] / x[i] for j >= i, which needs care at zeros. Along
        each line, with k the first zero: before k only the outputs before k depend on x[i];
        at k it is the product up to j with x[k] left out; after k every product holds
        x[k] = 0, so the gradient there is exactly 0 (as in torch, even for an inf gradient)."""
        if x._data.ndim == 0:
            return g
        zeros = np.cumsum(x._data == 0, axis=axis)
        before = zeros == 0
        if before.all():
            return div(_reverse_cumsum(mul(g, out), axis), x)
        first = (x._data == 0) & (zeros == 1)
        early = _reverse_cumsum(masked(before, mul(g, out)), axis)
        early = div(early, where(constant(before), x, 1))
        products = cumprod(where(constant(first), 1, x), axis)  # up to the next zero, if any
        at_first = sum(masked(zeros == 1, mul(g, products)), axis, keepdim=True)
        return masked(before | first, where(constant(before), early, at_first))

    return gradient


def _running(function, name, gradient, saves):
    def running(input, dim, *, dtype=None):
        if dtype is not None:
            input = cast(input, _dtype.as_numpy(dtype))
        elif data_of(input).dtype.kind in "biu":
            input = cast(input, _dtype.INT64)
        x, axis = input._data, axis_of(input, dim)
        out = function(as_vector(x), axis=axis).reshape(x.shape)
        saved = (input, OUTPUT) if saves else ()
        return result(out, input, name=name, saved=saved, grads=(gradient(axis),))

    running.__name__ = running.__qualname__ = function.__name__
    running.__doc__ = f"The running {function.__name__[3:]} along dim."
    return numeric(running)


def _cumsum_grad(axis):
    return lambda g: g if g._data.ndim == 0 else _reverse_cumsum(g, axis)


cumsum = _running(np.cumsum, "CumsumBackward0", _cumsum_grad, saves=False)
cumprod = _running(np.cumprod, "CumprodBackward0", _cumprod_grad, saves=True)


def _shift(x, axes):
    """The largest value along `axes` (kept), or 0 where it is infinite: subtracting it
    before exp keeps exp from overflowing, and leaves rows of -inf or inf to give torch's answer."""
    peak = np.amax(x, axis=axes, keepdims=True, initial=-np.inf)
    return np.where(np.isfinite(peak), peak, 0).astype(x.dtype, copy=False)


def _float_tensor(input):
    return input if data_of(input).dtype.kind == "f" else cast(input, _dtype.FLOAT32)


@reduction
def logsumexp(input, dim, keepdim=False):
    """log(sum(exp(x))) along dim, computed without overflow."""
    input = _float_tensor(input)
    axes = dims_of(input, dim)
    peak = _shift(input._data, axes)
    total = log(sum(exp(sub(input, Tensor(peak))), axes, keepdim))
    if not keepdim:
        peak = peak.reshape(()) if axes is None else np.squeeze(peak, axis=axes)
    return add(total, Tensor(peak))


def _softmax_input(input, dim, dtype, kernel):
    if dtype is not None:
        input = cast(input, _dtype.as_numpy(dtype))
    x = data_of(input)
    if x.dtype.kind != "f":
        raise NotImplementedError(f"\"{kernel}\" not implemented for '{_dtype.of(x.dtype).legacy}'")
    axes = dims_of(input, operator.index(dim))
    return sub(input, Tensor(_shift(x, axes))), axes


@numeric
def softmax(input, dim, dtype=None):
    """exp(x) / sum(exp(x)) along dim, after subtracting the largest value."""
    shifted, axes = _softmax_input(input, dim, dtype, "softmax_lastdim_kernel_impl")
    e = exp(shifted)
    return div(e, sum(e, axes, True))


@numeric
def log_softmax(input, dim, dtype=None):
    shifted, axes = _softmax_input(input, dim, dtype, "log_softmax_lastdim_kernel_impl")
    return sub(shifted, log(sum(exp(shifted), axes, True)))
