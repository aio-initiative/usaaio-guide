# Partly ported from PyTorch (torch/nn/functional.py: argument checks and error messages), BSD-3-Clause.
# See THIRD_PARTY_NOTICES.md.
"""torch.nn.functional: what the layers compute, as functions.

Most functions are written with the package's differentiable operations, so their gradients
(and second derivatives) follow. A few are primitives with their own backward, written with
tensor operations as everywhere else:

- unfold and fold (im2col and col2im), each the other's gradient. Convolution is unfold, one
  matmul and a reshape; transposed convolution is a matmul and fold; average pooling is
  unfold, a sum and a division. So none of them loops over output pixels (fold loops over
  the kernel's offsets), and their gradients come from matmul's and fold's.
- constant padding, whose gradient is a slice. The other padding modes, max pooling,
  embedding and nearest interpolation read values by index, so their gradients add up.
- the activations and losses whose plain formulas would overflow, or give nan for an inf
  gradient where torch does not (elu, gelu, silu, softplus, logsigmoid, binary cross entropy
  with or without logits). These use torch's own backward formulas.

Shapes follow torch: images are (N, C, H, W), and the image functions also take one
unbatched image (C, H, W).
"""

import math
import operator as _operator
import warnings

import numpy as np

# The package's own operations are imported under private names, so that torch.nn.functional
# offers only torch's names: F.sum or F.mul gives the not-available message, as torch's
# absence would.
from .. import _autograd, _dtype
from .._index import gather as _gather
from .._index import getitem as _getitem
from .._linalg import matmul as _matmul
from .._missing import module_getattr as _module_getattr
from .._missing import not_available as _not_available
from .._ops import abs as _abs
from .._ops import add as _add
from .._ops import clamp as _clamp
from .._ops import constant as _constant
from .._ops import data_of as _data_of
from .._ops import div as _div
from .._ops import exp as _exp
from .._ops import log as _log
from .._ops import mul as _mul
from .._ops import neg as _neg
from .._ops import numeric as _numeric
from .._ops import pow as _pow
from .._ops import relu as _relu
from .._ops import result as _result
from .._ops import sigmoid, tanh
from .._ops import sqrt as _sqrt
from .._ops import sub as _sub
from .._ops import where as _where
from .._ops import write as _write
from .._random import _stream
from .._reduce import log_softmax as _log_softmax
from .._reduce import mean as _mean
from .._reduce import softmax as _softmax
from .._reduce import sum as _sum
from .._reduce import var as _var
from .._shape import reshape as _reshape
from .._shape import squeeze as _squeeze
from .._shape import transpose as _transpose
from .._shape import unsqueeze as _unsqueeze
from .._tensor import Tensor


def _pair(value):
    """An int or a pair of ints as a pair, as torch's _pair."""
    if isinstance(value, (tuple, list)):
        return tuple(_operator.index(v) for v in (value * 2 if len(value) == 1 else value))
    return (_operator.index(value),) * 2


def _legacy(array):
    return _dtype.of(array.dtype).legacy


def _unbatched(function, input, *args, **kwargs):
    """function on a (C, H, W) image, given as a batch of one."""
    out = function(_unsqueeze(input, 0), *args, **kwargs)
    if isinstance(out, tuple):
        return tuple(_squeeze(o, 0) for o in out)
    return _squeeze(out, 0)


def _maybe_in_place(function, input, inplace):
    """function(input), or with inplace=True the same values written into `input`, which takes
    the result's history. The function reads a copy of the old values when the step is
    recorded, so a gradient that needs them still has them, as in torch."""
    if not inplace:
        return function(input)
    _write(input, ..., function(_autograd.original(input)), adopt=True)
    return input


# Activations. Those whose plain formula would overflow, lose digits or (with an inf
# gradient) give nan where torch does not are primitives with torch's own backward formula.


def relu(input, inplace=False):
    return _maybe_in_place(_relu, input, inplace)


def leaky_relu(input, negative_slope=0.01, inplace=False):
    """x where x > 0, else negative_slope * x (and that slope's gradient at 0, as torch)."""

    def leaky(x):
        return _where(_constant(x._data > 0), x, _mul(x, negative_slope))

    return _maybe_in_place(leaky, input, inplace)


def _float_array(input):
    x = _data_of(input)
    if x.dtype.kind != "f":
        raise NotImplementedError(f"\"elementwise_kernel\" not implemented for '{_legacy(x)}'")
    return x


def _activation(name, forward, derivative):
    """A primitive activation: forward(array) gives the values, and the input's gradient is
    g * derivative(x), written with tensor operations (so second derivatives work)."""

    @_numeric
    def apply(input):
        x = _float_array(input)
        out = forward(x).astype(x.dtype, copy=False)
        return _result(out, input, name=name, saved=(input,), grads=(lambda g, x: _mul(g, derivative(x)),))

    return apply


def _elu_derivative(alpha):
    # exp of clamp(x, max=0), not of x, so a large x cannot overflow into the unused branch.
    return lambda x: _where(_constant(x._data > 0), 1.0, _mul(_exp(_clamp(x, max=0)), alpha))


def elu(input, alpha=1.0, inplace=False):
    function = _activation(
        "EluBackward0", lambda x: np.where(x > 0, x, alpha * np.expm1(np.minimum(x, 0))), _elu_derivative(alpha)
    )
    return _maybe_in_place(function, input, inplace)


# erf, which NumPy lacks: W. J. Cody's rational approximations (SPECFUN's CALERF), evaluated
# in double precision. Each tuple holds a polynomial's coefficients, highest power first.
_ERF_NEAR = (
    (1.85777706184603153e-1, 3.16112374387056560e00, 1.13864154151050156e02, 3.77485237685302021e02,
     3.20937758913846947e03),
    (1.0, 2.36012909523441209e01, 2.44024637934444173e02, 1.28261652607737228e03, 2.84423683343917062e03),
)
_ERFC_MIDDLE = (
    (2.15311535474403846e-8, 5.64188496988670089e-1, 8.88314979438837594e00, 6.61191906371416295e01,
     2.98635138197400131e02, 8.81952221241769090e02, 1.71204761263407058e03, 2.05107837782607147e03,
     1.23033935479799725e03),
    (1.0, 1.57449261107098347e01, 1.17693950891312499e02, 5.37181101862009858e02, 1.62138957456669019e03,
     3.29079923573345963e03, 4.36261909014324716e03, 3.43936767414372164e03, 1.23033935480374942e03),
)
_ERFC_FAR = (
    (1.63153871373020978e-2, 3.05326634961232344e-1, 3.60344899949804439e-1, 1.25781726111229246e-1,
     1.60837851487422766e-2, 6.58749161529837803e-4),
    (1.0, 2.56852019228982242e00, 1.87295284992346725e00, 5.27905102951428412e-1, 6.05183413124413191e-2,
     2.33520497626869185e-3),
)


def _horner(y, coefficients):
    out = np.full_like(y, coefficients[0])
    for c in coefficients[1:]:
        out *= y
        out += c
    return out


def _ratio(y, pair):
    return _horner(y, pair[0]) / _horner(y, pair[1])


def _erf_values(x):
    """erf of every element, in float64, to within a few units in the last place: a rational
    function up to |x| = 0.46875, and 1 - erfc beyond it."""
    y = np.abs(np.asarray(x, dtype=np.float64))
    out = np.empty_like(y)
    near = y <= 0.46875
    out[near] = y[near] * _ratio(y[near] ** 2, _ERF_NEAR)
    rest = np.minimum(y[~near], 27.0)  # erfc(27) is below the smallest double; nan stays nan
    middle = rest <= 4.0
    erfc = np.empty_like(rest)
    erfc[middle] = _ratio(rest[middle], _ERFC_MIDDLE)
    far = rest[~middle]
    inverse = 1 / (far * far)
    erfc[~middle] = (1 / math.sqrt(math.pi) - inverse * _ratio(inverse, _ERFC_FAR)) / far
    # Cody splits exp(-y * y) in two to keep erfc's last digits; erf = 1 - erfc does not need
    # them (erfc is below 2e-8 wherever the rounding of y * y matters), so one exp will do.
    erfc *= np.exp(-rest * rest)
    out[~near] = (0.5 - erfc) + 0.5
    return np.copysign(out, x)


@_numeric
def _erf(input):
    """The error function, as an operation (rounded to the input's dtype)."""
    x = input._data
    return _result(
        _erf_values(x).astype(x.dtype), input, name="ErfBackward0", saved=(input,),
        grads=(lambda g, x: _mul(g, _mul(_exp(_neg(_mul(x, x))), 2 / math.sqrt(math.pi))),),
    )


def _gelu_grad(g, x, erf):
    """torch's GELU backward, g * (cdf(x) + x * pdf(x)). The cdf reuses the forward's erf
    values, unless the backward pass is itself recorded (create_graph), which needs it as a
    function of x."""
    erf = _erf(_mul(x, math.sqrt(0.5))) if _autograd.is_grad_enabled() else _constant(erf)
    cdf = _mul(_add(erf, 1), 0.5)
    pdf = _mul(_exp(_mul(_mul(x, x), -0.5)), 1 / math.sqrt(2 * math.pi))
    return _mul(g, _add(cdf, _mul(x, pdf)))


@_numeric
def _gelu(input):
    """x * 0.5 * (1 + erf(x / sqrt(2))), in torch's order and in the input's dtype."""
    x = _float_array(input)
    erf = _erf_values(x * math.sqrt(0.5)).astype(x.dtype)
    out = x * 0.5 * (1 + erf)
    return _result(out, input, name="GeluBackward0", saved=(input, erf), grads=(_gelu_grad,))


_KAPPA, _BETA = 0.044715, math.sqrt(2 / math.pi)


def _gelu_tanh_derivative(x):
    inner = tanh(_mul(_add(x, _mul(_mul(_mul(x, x), x), _KAPPA)), _BETA))
    left = _mul(_add(inner, 1), 0.5)
    right = _mul(_mul(_mul(x, 0.5), _sub(1, _mul(inner, inner))), _mul(_add(_mul(_mul(x, x), 3 * _KAPPA), 1), _BETA))
    return _add(left, right)


def _gelu_tanh_values(x):
    return 0.5 * x * (1 + np.tanh(_BETA * (x + _KAPPA * (x * x * x))))


_gelu_tanh = _activation("GeluBackward0", _gelu_tanh_values, _gelu_tanh_derivative)


def gelu(input, approximate="none"):
    """x * Phi(x), with Phi the normal CDF; approximate="tanh" uses torch's tanh formula."""
    if approximate == "none":
        return _gelu(input)
    if approximate == "tanh":
        return _gelu_tanh(input)
    raise RuntimeError("approximate argument must be either none or tanh.")


def _sigmoid_values(x):
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + e), e / (1 + e))


def _silu_derivative(x):
    s = sigmoid(x)
    return _mul(s, _add(_mul(x, _sub(1, s)), 1))


_silu = _activation("SiluBackward0", lambda x: x * _sigmoid_values(x), _silu_derivative)


def silu(input, inplace=False):
    return _maybe_in_place(_silu, input, inplace)


@_numeric
def softplus(input, beta=1.0, threshold=20.0):
    """log(1 + exp(beta * x)) / beta, and x itself where beta * x > threshold."""
    x = _float_array(input)
    linear = x * beta > threshold
    out = np.where(linear, x, np.log1p(np.exp(np.minimum(x * beta, threshold))) / beta).astype(x.dtype, copy=False)
    return _result(
        out, input, name="SoftplusBackward0", saved=(input,),
        grads=(lambda g, x: _where(_constant(linear), g, _mul(g, sigmoid(_mul(x, beta)))),),
    )


logsigmoid = _activation(
    "LogSigmoidBackward0", lambda x: np.minimum(x, 0) - np.log1p(np.exp(-np.abs(x))), lambda x: sigmoid(_neg(x))
)
logsigmoid.__doc__ = "log(sigmoid(x)), computed as min(x, 0) - log1p(exp(-|x|)) so it never overflows."


def _softmax_dim(input, dim, name):
    if dim is not None:
        return dim
    warnings.warn(
        f"Implicit dimension choice for {name} has been deprecated. Change the call to include dim=X as an argument.",
        stacklevel=3,
    )
    return 0 if _data_of(input).ndim in (0, 1, 3) else 1


def softmax(input, dim=None, _stacklevel=3, dtype=None):
    return _softmax(input, _softmax_dim(input, dim, "softmax"), dtype=dtype)


def log_softmax(input, dim=None, _stacklevel=3, dtype=None):
    return _log_softmax(input, _softmax_dim(input, dim, "log_softmax"), dtype=dtype)


# Linear layers.


def linear(input, weight, bias=None):
    """input @ weight.T + bias."""
    w = weight if _data_of(weight).ndim < 2 else _transpose(weight, 0, 1)
    out = _matmul(input, w)
    return out if bias is None else _add(out, bias)


# Convolution and pooling. A geometry is (kernel, dilation, padding, stride), each a pair.


def _geometry(kernel_size, dilation, padding, stride):
    return tuple(_pair(v) for v in (kernel_size, dilation, padding, stride))


def _span(geometry):
    """The height and width a dilated kernel covers."""
    (kh, kw), (dh, dw), _, _ = geometry
    return dh * (kh - 1) + 1, dw * (kw - 1) + 1


def _blocks(size, geometry):
    """How many window positions fit along the height and the width."""
    span, (_, _), (ph, pw), (sh, sw) = _span(geometry), geometry[0], geometry[2], geometry[3]
    return (size[0] + 2 * ph - span[0]) // sh + 1, (size[1] + 2 * pw - span[1]) // sw + 1


def _windows(x, geometry, blocks, fill=0):
    """A (N, C, oh, ow, kh, kw) view of every window of the padded image array x."""
    _, (dh, dw), (ph, pw), (sh, sw) = geometry
    if ph or pw:
        x = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), constant_values=fill)
    view = np.lib.stride_tricks.sliding_window_view(x, _span(geometry), axis=(2, 3))
    return view[:, :, ::sh, ::sw, ::dh, ::dw][:, :, : blocks[0], : blocks[1]]


@_numeric
def _unfold(input, geometry, blocks=None):
    """im2col: (N, C, H, W) to (N, C * kh * kw, oh * ow), each column one window."""
    x = input._data
    n, c, h, w = x.shape
    (kh, kw) = geometry[0]
    blocks = blocks or _blocks((h, w), geometry)
    cols = _windows(x, geometry, blocks).transpose(0, 1, 4, 5, 2, 3).reshape(n, c * kh * kw, blocks[0] * blocks[1])
    return _result(cols, input, name="Im2ColBackward0", grads=(lambda g: _fold(g, (h, w), geometry, blocks),))


@_numeric
def _fold(input, size, geometry, blocks=None):
    """col2im, unfold's adjoint: every column's values added back where its window lies.
    It loops over the kernel's offsets (kh * kw strided additions), not over pixels."""
    cols = input._data
    n = cols.shape[0]
    (kh, kw), (dh, dw), (ph, pw), (sh, sw) = geometry
    (h, w), blocks = size, blocks or _blocks(size, geometry)
    oh, ow = blocks
    pieces = cols.reshape(n, -1, kh, kw, oh, ow)
    out = np.zeros((n, pieces.shape[1], h + 2 * ph, w + 2 * pw), dtype=cols.dtype)
    for i in range(kh):
        for j in range(kw):
            out[:, :, i * dh : i * dh + sh * (oh - 1) + 1 : sh, j * dw : j * dw + sw * (ow - 1) + 1 : sw] += pieces[:, :, i, j]
    if ph or pw:
        out = out[:, :, ph : ph + h, pw : pw + w].copy()
    return _result(out, input, name="Col2ImBackward0", grads=(lambda g: _unfold(g, geometry, blocks),))


def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    """Every kernel-sized window as a column: (N, C, H, W) to (N, C * kh * kw, L)."""
    x = _data_of(input)
    if x.ndim == 3:
        return _unbatched(unfold, input, kernel_size, dilation, padding, stride)
    if x.ndim != 4:
        raise NotImplementedError(f"Input Error: Only 3D or 4D input Tensors are supported (got {x.ndim}D)")
    geometry = _geometry(kernel_size, dilation, padding, stride)
    _check_fits(x.shape[2:], geometry)
    return _unfold(input, geometry)


def fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    """unfold's adjoint: (N, C * kh * kw, L) columns summed back into (N, C, H, W)."""
    x = _data_of(input)
    if x.ndim == 2:
        return _unbatched(fold, input, output_size, kernel_size, dilation, padding, stride)
    geometry = _geometry(kernel_size, dilation, padding, stride)
    size, (kh, kw) = _pair(output_size), geometry[0]
    oh, ow = _blocks(size, geometry)
    if x.ndim != 3 or x.shape[1] % (kh * kw) or x.shape[2] != oh * ow:
        raise RuntimeError(
            f"Given output_size={size}, kernel_size={geometry[0]}, dilation={geometry[1]}, padding={geometry[2]}, "
            f"stride={geometry[3]}, expected size of input's dimension 2 to match the calculated number of sliding "
            f"blocks {oh} * {ow} = {oh * ow}, but got input.size(2)={x.shape[-1]}."
        )
    return _fold(input, size, geometry)


def _check_fits(size, geometry):
    (ph, pw), span = geometry[2], _span(geometry)
    if size[0] + 2 * ph < span[0] or size[1] + 2 * pw < span[1]:
        raise RuntimeError(
            f"Calculated padded input size per channel: ({size[0] + 2 * ph} x {size[1] + 2 * pw}). "
            f"Kernel size: ({span[0]} x {span[1]}). Kernel size can't be greater than actual input size"
        )


def _check_conv(name, x, w, bias):
    if x.ndim != 4:
        raise RuntimeError(f"Expected 3D (unbatched) or 4D (batched) input to {name}, but got input of size: {list(x.shape)}")
    if x.dtype != w.dtype:
        raise RuntimeError(f"expected scalar type {_legacy(x)} but found {_legacy(w)}")
    if bias is not None and _data_of(bias).dtype != x.dtype:
        raise RuntimeError(
            f"Input type ({_dtype.of(x.dtype).np}) and bias type ({_dtype.of(_data_of(bias).dtype).np}) should be the same"
            .replace("float64", "double").replace("float32", "float")
        )
    if x.dtype.kind != "f":
        raise RuntimeError(f"\"slow_conv2d_cpu\" not implemented for '{_legacy(x)}'")


def _same_padding(input, kernel, stride, dilation):
    """padding="same": the input padded so the output keeps its size. An even kernel needs
    one more row or column on the bottom or right, as in torch."""
    if stride != (1, 1):
        raise RuntimeError("padding='same' is not supported for strided convolutions")
    totals = [d * (k - 1) for k, d in zip(kernel, dilation)]
    if all(total % 2 == 0 for total in totals):
        return input, tuple(total // 2 for total in totals)
    (th, tw) = totals
    return pad(input, (tw // 2, tw - tw // 2, th // 2, th - th // 2)), (0, 0)


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """2D cross-correlation: unfold the windows, one matmul with the flattened kernels."""
    x, w = _data_of(input), _data_of(weight)
    if x.ndim == 3:
        return _unbatched(conv2d, input, weight, bias, stride, padding, dilation, groups)
    if groups != 1:
        raise _not_available(f"torch.nn.functional.conv2d with groups={groups}")
    _check_conv("conv2d", x, w, bias)
    c_out, c_in, kh, kw = w.shape
    if x.shape[1] != c_in:
        raise RuntimeError(
            f"Given groups=1, weight of size {list(w.shape)}, expected input{list(x.shape)} to have {c_in} channels, "
            f"but got {x.shape[1]} channels instead"
        )
    stride, dilation = _pair(stride), _pair(dilation)
    if padding == "same":
        input, padding = _same_padding(input, (kh, kw), stride, dilation)
    elif padding == "valid":
        padding = 0
    elif isinstance(padding, str):
        raise RuntimeError(f"Invalid padding string: '{padding}'")
    geometry = ((kh, kw), dilation, _pair(padding), stride)
    size = input._data.shape[2:]
    _check_fits(size, geometry)
    oh, ow = _blocks(size, geometry)
    out = _reshape(_matmul(_reshape(weight, c_out, -1), _unfold(input, geometry)), x.shape[0], c_out, oh, ow)
    return out if bias is None else _add(out, _reshape(bias, c_out, 1, 1))


def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    """The adjoint of conv2d: one matmul spreads every input pixel over a kernel-sized patch,
    and fold adds the overlapping patches up."""
    x, w = _data_of(input), _data_of(weight)
    if x.ndim == 3:
        return _unbatched(conv_transpose2d, input, weight, bias, stride, padding, output_padding, groups, dilation)
    if groups != 1:
        raise _not_available(f"torch.nn.functional.conv_transpose2d with groups={groups}")
    _check_conv("conv_transpose2d", x, w, bias)
    c_in, c_out, kh, kw = w.shape
    if x.shape[1] != c_in:
        raise RuntimeError(
            f"Given transposed=1, weight of size {list(w.shape)}, expected input{list(x.shape)} to have {c_in} "
            f"channels, but got {x.shape[1]} channels instead"
        )
    stride, padding, extra, dilation = (_pair(v) for v in (stride, padding, output_padding, dilation))
    if any(e >= s and e >= d for e, s, d in zip(extra, stride, dilation)):
        raise RuntimeError(
            "output padding must be smaller than either stride or dilation, but got "
            f"output_padding_height: {extra[0]} output_padding_width: {extra[1]} stride_height: {stride[0]} "
            f"stride_width: {stride[1]} dilation_height: {dilation[0]} dilation_width: {dilation[1]}"
        )
    n, _, h, wd = x.shape
    geometry = ((kh, kw), dilation, padding, stride)
    size = tuple((i - 1) * s - 2 * p + d * (k - 1) + e + 1 for i, s, p, d, k, e in zip((h, wd), stride, padding, dilation, (kh, kw), extra))
    if min(size) < 1:
        raise RuntimeError(
            f"Given input size per channel: ({h} x {wd}). Calculated output size per channel: ({size[0]} x {size[1]}). "
            "Output size is too small"
        )
    cols = _matmul(_transpose(_reshape(weight, c_in, -1), 0, 1), _reshape(input, n, c_in, h * wd))
    out = _fold(cols, size, geometry, (h, wd))
    return out if bias is None else _add(out, _reshape(bias, c_out, 1, 1))


def _pool_geometry(name, input, kernel_size, stride, padding, dilation, ceil_mode):
    if ceil_mode:
        raise _not_available(f"torch.nn.functional.{name} with ceil_mode=True")
    x = _data_of(input)
    if x.ndim != 4 or x.size == 0:
        raise RuntimeError("non-empty 3D or 4D (batch mode) tensor expected for input")
    kernel = _pair(kernel_size)
    stride = kernel if stride is None or (isinstance(stride, (tuple, list)) and not stride) else _pair(stride)
    geometry = (kernel, _pair(dilation), _pair(padding), stride)
    for k, d, p in zip(kernel, geometry[1], geometry[2]):
        if p > ((k - 1) * d + 1) // 2:
            raise RuntimeError(f"pad should be at most half of effective kernel size, but got pad={p}, kernel_size={k} and dilation={d}")
    n, c, h, w = x.shape
    oh, ow = _blocks((h, w), geometry)
    if oh < 1 or ow < 1:
        raise RuntimeError(f"Given input size: ({c}x{h}x{w}). Calculated output size: ({c}x{oh}x{ow}). Output size is too small")
    return geometry, (oh, ow)


def _first_inside(blocks, stride, pad, dilation):
    """For each window along one axis, the first kernel step that lands inside the input."""
    start = np.arange(blocks) * stride - pad
    return np.maximum(0, (dilation - 1 - start) // dilation)


def _max_positions(windows, geometry, fill):
    """The kernel position torch's max pooling picks in each window of `windows` (..., kh * kw):
    it starts at the first position inside the input and moves on a strictly larger value or a
    nan, so a tie goes to the first maximum, a nan to the last nan, and a window whose values are
    all the fill (-inf) stays at its first position inside the input, never on the padding."""
    (kh, kw), (dh, dw), (ph, pw), (sh, sw) = geometry
    best = windows.argmax(-1)
    if windows.dtype.kind == "f":
        nan = np.isnan(windows)
        best = np.where(nan.any(-1), kh * kw - 1 - nan[..., ::-1].argmax(-1), best)
    oh, ow = windows.shape[-3:-1]
    first = _first_inside(oh, sh, ph, dh)[:, None] * kw + _first_inside(ow, sw, pw, dw)
    unbeaten = np.take_along_axis(windows, best[..., None], -1)[..., 0] == fill
    return np.where(unbeaten, first, best)


def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    """The largest value in each window, chosen as torch chooses it (see _max_positions). The
    values are gathered from the input, so the gradient goes to each window's chosen entry."""
    x = _data_of(input)
    if x.ndim == 3:
        return _unbatched(max_pool2d, input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)
    geometry, (oh, ow) = _pool_geometry("max_pool2d", input, kernel_size, stride, padding, dilation, ceil_mode)
    (kh, kw), (dh, dw), (ph, pw), (sh, sw) = geometry
    n, c, h, w = x.shape
    fill = -np.inf if x.dtype.kind == "f" else np.iinfo(x.dtype).min
    best = _max_positions(_windows(x, geometry, (oh, ow), fill).reshape(n, c, oh, ow, kh * kw), geometry, fill)
    row, col = np.divmod(best, kw)
    index = ((np.arange(oh)[:, None] * sh + row * dh - ph) * w + np.arange(ow) * sw + col * dw - pw).astype(np.int64)
    out = _gather(_reshape(input, n, c, h * w), 2, Tensor(index.reshape(n, c, oh * ow)))
    out = _reshape(out, n, c, oh, ow)
    return (out, Tensor(index)) if return_indices else out


def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    """The mean of each window: unfold, then a sum and a division."""
    x = _data_of(input)
    if x.ndim == 3:
        return _unbatched(avg_pool2d, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
    geometry, (oh, ow) = _pool_geometry("avg_pool2d", input, kernel_size, stride, padding, 1, ceil_mode)
    (kh, kw), _, (ph, pw), (sh, sw) = geometry
    n, c, h, w = x.shape
    total = _sum(_reshape(_unfold(input, geometry), n, c, kh * kw, oh * ow), 2)
    if divisor_override:
        divisor = divisor_override
    elif count_include_pad:
        divisor = kh * kw
    else:  # only the window's positions inside the image count

        def inside(blocks, stride, pad, kernel, size):
            start = np.arange(blocks) * stride - pad
            return np.minimum(start + kernel, size) - np.maximum(start, 0)

        counts = np.outer(inside(oh, sh, ph, kh, h), inside(ow, sw, pw, kw, w)).reshape(-1)
        divisor = _constant(counts.astype(x.dtype))
    return _reshape(_div(total, divisor), n, c, oh, ow)


def _adaptive_matrix(size, out, dtype):
    """The (out, size) matrix that averages each of torch's adaptive bins."""
    matrix = np.zeros((out, size), dtype=dtype)
    for i in range(out):
        start, end = (i * size) // out, -(-((i + 1) * size) // out)
        matrix[i, start:end] = 1 / (end - start)
    return matrix


def adaptive_avg_pool2d(input, output_size):
    """Average pooling to a fixed output size. When the size divides the input it is
    avg_pool2d; otherwise a product with averaging matrices on both sides."""
    x = _data_of(input)
    if x.ndim == 3:
        return _unbatched(adaptive_avg_pool2d, input, output_size)
    sizes = output_size if isinstance(output_size, (tuple, list)) else (output_size, output_size)
    if len(sizes) != 2:
        raise RuntimeError("adaptive_avg_pool2d: output_size must be 2")
    h, w = x.shape[-2:]
    oh, ow = (h if sizes[0] is None else sizes[0]), (w if sizes[1] is None else sizes[1])
    if h % oh == 0 and w % ow == 0:
        return avg_pool2d(input, (h // oh, w // ow))
    rows, cols = _adaptive_matrix(h, oh, x.dtype), _adaptive_matrix(w, ow, x.dtype)
    return _matmul(_matmul(_constant(rows), input), _constant(np.ascontiguousarray(cols.T)))


# Padding.


@_numeric
def _constant_pad(input, widths, value):
    crop = tuple(slice(max(-before, 0), n - max(-after, 0)) for (before, after), n in zip(widths, input._data.shape))
    if any(before < 0 or after < 0 for before, after in widths):
        input = _getitem(input, crop)
    grow = [(max(before, 0), max(after, 0)) for before, after in widths]
    x = input._data
    keep = tuple(slice(before, before + n) for (before, _), n in zip(grow, x.shape))
    out = np.pad(x, grow, constant_values=value)
    return _result(out, input, name="ConstantPadNdBackward0", grads=(lambda g: _getitem(g, keep),))


def _pad_positions(size, before, after, mode, axis, shape):
    """Which input position each output position copies, along one axis."""
    positions = np.arange(-before, size + after)
    if mode == "replicate":
        return np.clip(positions, 0, size - 1)
    if mode == "reflect":
        if before >= size or after >= size:
            raise RuntimeError(
                f"Argument #4: Padding size should be less than the corresponding input dimension, but got: "
                f"padding ({before}, {after}) at dimension {axis} of input {list(shape)}"
            )
        positions = np.abs(positions)
        return np.where(positions >= size, 2 * (size - 1) - positions, positions)
    if before > size or after > size:
        raise RuntimeError("Padding value causes wrapping around more than once.")
    return positions % size


def pad(input, pad, mode="constant", value=None):
    """Pad the last dimensions: pad = (left, right) for the last, then (top, bottom) for the
    one before, and so on. Negative amounts crop. The modes reflect, replicate and circular
    copy values from the input by index, so their gradients add up where values repeat."""
    x = _data_of(input)
    pad = [_operator.index(p) for p in pad]
    if len(pad) % 2:
        raise RuntimeError("Padding length must be divisible by 2")
    if len(pad) // 2 > x.ndim:
        raise RuntimeError(
            "Padding length should be less than or equal to two times the input dimension but got padding length "
            f"{len(pad)} and input of dimension {x.ndim}"
        )
    widths = [(0, 0)] * x.ndim
    for i in range(len(pad) // 2):
        widths[x.ndim - 1 - i] = (pad[2 * i], pad[2 * i + 1])
    if mode == "constant":
        return _constant_pad(input, widths, 0 if value is None else value)
    if mode not in ("reflect", "replicate", "circular"):
        raise NotImplementedError(f"Unrecognised padding mode {mode}")
    if value is not None and value != 0:
        raise RuntimeError(f'Padding mode "{mode}" doesn\'t take in value argument')
    if x.ndim - len(pad) // 2 not in (1, 2):
        raise NotImplementedError(f"Padding size {len(pad)} is not supported for {x.ndim}D input tensor.")
    out = input
    for axis, (before, after) in enumerate(widths):
        if before or after:
            positions = _pad_positions(x.shape[axis], before, after, mode, axis, x.shape)
            out = _getitem(out, (slice(None),) * axis + (Tensor(positions.astype(np.int64)),))
    return out


# Normalization.


def _update_running(statistic, value, momentum):
    """statistic = momentum * value + (1 - momentum) * statistic, in place. Like torch's
    kernel, this writes the memory directly: unrecorded, and the version stays the same."""
    old = statistic._data
    old[...] = momentum * value.reshape(old.shape) + (1 - momentum) * old


def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    """Normalize each channel (dimension 1) by the batch's mean and biased variance when
    training, updating the running statistics given (with the unbiased variance, as torch);
    by the running statistics otherwise. Then scale by weight and shift by bias."""
    x = _data_of(input)
    channels = x.shape[1] if x.ndim > 1 else 1
    for name, value in (("running_mean", running_mean), ("running_var", running_var), ("weight", weight), ("bias", bias)):
        if value is not None and _data_of(value).size != channels:
            raise RuntimeError(f"{name} should contain {channels} elements not {_data_of(value).size}")
    shape = (1, channels) + (1,) * (x.ndim - 2)
    if training:
        count = x.size // channels
        if count <= 1:
            raise ValueError(f"Expected more than 1 value per channel when training, got input size {input.shape}")
        axes = (0,) + tuple(range(2, x.ndim))
        batch_mean, batch_var = _mean(input, axes, keepdim=True), _var(input, axes, keepdim=True, correction=0)
        if running_mean is not None:
            _update_running(running_mean, batch_mean._data, momentum)
        if running_var is not None:
            _update_running(running_var, batch_var._data * (count / (count - 1)), momentum)
    else:
        if running_mean is None or running_var is None:
            raise RuntimeError("running_mean must be defined in evaluation mode")
        batch_mean, batch_var = _reshape(running_mean, *shape), _reshape(running_var, *shape)
    out = _div(_sub(input, batch_mean), _sqrt(_add(batch_var, eps)))
    if weight is not None:
        out = _mul(out, _reshape(weight, *shape))
    return out if bias is None else _add(out, _reshape(bias, *shape))


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    """Normalize over the last dimensions (normalized_shape) of every sample, then scale and shift."""
    x = _data_of(input)
    shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    if len(shape) > x.ndim or x.shape[x.ndim - len(shape) :] != shape:
        raise RuntimeError(
            f"Given normalized_shape={list(shape)}, expected input with shape [*, {', '.join(map(str, shape))}], "
            f"but got input of size{list(x.shape)}"
        )
    axes = tuple(range(x.ndim - len(shape), x.ndim))
    out = _div(_sub(input, _mean(input, axes, keepdim=True)), _sqrt(_add(_var(input, axes, keepdim=True, correction=0), eps)))
    if weight is not None:
        out = _mul(out, weight)
    return out if bias is None else _add(out, bias)


def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    """Split the channels (dimension 1) into num_groups groups, normalize each sample's groups
    by their mean and biased variance over the group's channels and positions, then scale and
    shift every channel by weight and bias."""
    x = _data_of(input)
    if x.ndim < 2:
        raise RuntimeError(f"Expected at least 2 dimensions for input tensor but received {x.ndim}")
    n, channels = x.shape[:2]
    num_groups = _operator.index(num_groups)
    if channels % num_groups != 0:
        raise RuntimeError(
            "Expected number of channels in input to be divisible by num_groups, "
            f"but got input of shape {list(x.shape)} and num_groups={num_groups}"
        )
    for name, value in (("weight", weight), ("bias", bias)):
        if value is not None and _data_of(value).shape != (channels,):
            raise RuntimeError(
                f"Expected {name} to be a vector of size equal to the number of channels in input, "
                f"but got {name} of shape {list(_data_of(value).shape)} and input of shape {list(x.shape)}"
            )
    positions = math.prod(x.shape[2:])
    if positions == 0:
        raise RuntimeError("Expected HxW to be greater than 0, got 0")
    if x.dtype.kind != "f":
        raise NotImplementedError(f'"GroupNormKernelImpl" not implemented for \'{_legacy(x)}\'')
    if any(v is not None and _data_of(v).dtype != x.dtype for v in (weight, bias)):
        raise RuntimeError("mixed dtype (CPU): all inputs must share same datatype.")
    groups = _reshape(input, n, num_groups, channels // num_groups * positions)
    centred = _sub(groups, _mean(groups, (2,), keepdim=True))
    out = _reshape(_div(centred, _sqrt(_add(_var(groups, (2,), keepdim=True, correction=0), eps))), *x.shape)
    shape = (1, channels) + (1,) * (x.ndim - 2)
    if weight is not None:
        out = _mul(out, _reshape(weight, *shape))
    return out if bias is None else _add(out, _reshape(bias, *shape))


# Embedding and dropout.


def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    """The rows of `weight` at the indices in `input`; the row padding_idx gets no gradient."""
    idx, rows = _data_of(input), _data_of(weight).shape[0]
    if idx.dtype not in (np.int64, np.int32):
        raise RuntimeError(
            "Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; "
            f"but got torch.{_legacy(idx)}Tensor instead (while checking arguments for embedding)"
        )
    for name, value in (("max_norm", max_norm), ("scale_grad_by_freq", scale_grad_by_freq or None), ("sparse", sparse or None)):
        if value is not None:
            raise _not_available(f"torch.nn.functional.embedding with {name}")
    if idx.size and (idx.min() < 0 or idx.max() >= rows):
        raise IndexError("index out of range in self")
    out = _getitem(weight, input)
    if padding_idx is None:
        return out
    padding_idx = padding_idx + rows if padding_idx < 0 else padding_idx
    return _where(_constant((idx != padding_idx)[..., None]), out, out.detach())


def dropout(input, p=0.5, training=True, inplace=False):
    """In training, each value is zeroed with probability p and the rest are scaled by
    1 / (1 - p): the input times a fixed random mask, as torch does on the CPU. Otherwise
    the input itself."""
    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
    x = _data_of(input)
    if not training or p == 0 or x.size == 0:
        return input
    if p == 1:
        noise = np.zeros(x.shape, dtype=x.dtype)
    else:
        noise = ((_stream(None).random(x.shape) < 1 - p) / (1 - p)).astype(x.dtype)
    return _maybe_in_place(lambda t: _mul(t, _constant(noise)), input, inplace)


# Losses.

_REDUCTIONS = ("none", "mean", "sum")


def _reduction(size_average, reduce, reduction):
    """The reduction, from the old size_average and reduce arguments when either is given."""
    if size_average is not None or reduce is not None:
        size_average = True if size_average is None else size_average
        reduce = True if reduce is None else reduce
        reduction = "mean" if size_average and reduce else ("sum" if reduce else "none")
        warnings.warn(
            f"size_average and reduce args will be deprecated, please use reduction='{reduction}' instead.", stacklevel=3
        )
    if reduction not in _REDUCTIONS:
        raise ValueError(f"{reduction} is not a valid value for reduction")
    return reduction


def _reduce(loss, reduction):
    return loss if reduction == "none" else (_sum(loss) if reduction == "sum" else _mean(loss))


def _check_sizes(input, target):
    if input.shape != target.shape:
        warnings.warn(
            f"Using a target size ({target.shape}) that is different to the input size ({input.shape}). "
            "This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.",
            stacklevel=3,
        )


def _weighted(loss, weight, input, reduction, mean_by_weight):
    """An elementwise loss times `weight`; its mean is over the weights (mse, l1) or the count (huber)."""
    if weight.shape != input.shape:
        raise ValueError("Weights and input must have the same size.")
    loss = _mul(loss, weight)
    if reduction == "mean" and mean_by_weight:
        return _div(_sum(loss), _sum(weight))
    return _reduce(loss, reduction)


def mse_loss(input, target, size_average=None, reduce=None, reduction="mean", weight=None):
    _check_sizes(input, target)
    reduction = _reduction(size_average, reduce, reduction)
    loss = _pow(_sub(input, target), 2)
    return _reduce(loss, reduction) if weight is None else _weighted(loss, weight, input, reduction, True)


def l1_loss(input, target, size_average=None, reduce=None, reduction="mean", weight=None):
    _check_sizes(input, target)
    reduction = _reduction(size_average, reduce, reduction)
    loss = _abs(_sub(input, target))
    return _reduce(loss, reduction) if weight is None else _weighted(loss, weight, input, reduction, True)


def huber_loss(input, target, reduction="mean", delta=1.0, weight=None):
    """0.5 * d**2 where |d| < delta, else delta * (|d| - 0.5 * delta), with d = input - target."""
    _check_sizes(input, target)
    reduction = _reduction(None, None, reduction)
    d = _sub(input, target)
    size = _abs(d)
    loss = _where(_constant(size._data < delta), _mul(_mul(d, d), 0.5), _mul(_sub(size, 0.5 * delta), delta))
    return _reduce(loss, reduction) if weight is None else _weighted(loss, weight, input, reduction, False)


def _class_targets(input, target, weight):
    """Check class-index targets as torch does; returns (class dim, number of classes)."""
    x, t = _data_of(input), _data_of(target)
    if t.dtype not in (np.int64, np.uint8):
        raise RuntimeError(f"expected target dtype to be Long or Byte, but got {_legacy(t)}")
    class_dim = 0 if x.ndim == 1 else 1
    if x.ndim > 1 and t.ndim and t.shape[0] != x.shape[0]:
        raise ValueError(f"Expected input batch_size ({x.shape[0]}) to match target batch_size ({t.shape[0]}).")
    if x.ndim > 2 and t.shape != x.shape[:1] + x.shape[2:]:
        raise RuntimeError(f"Expected target size {[x.shape[0]] + list(x.shape[2:])}, got {list(t.shape)}")
    classes = x.shape[class_dim]
    if weight is not None:
        w = _data_of(weight)
        if w.size != classes:
            raise RuntimeError(
                f"weight tensor should be defined either for all {classes} classes or no classes but got weight "
                f"tensor of shape: {list(w.shape)}"
            )
        if w.dtype != x.dtype:
            raise RuntimeError(f"expected scalar type {_legacy(x)} but found {_legacy(w)}")
    return class_dim, classes


def nll_loss(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean"):
    """-input[n, target[n]] * weight[target[n]], with targets equal to ignore_index left out.
    The mean divides by the total weight of the targets kept."""
    reduction = _reduction(size_average, reduce, reduction)
    class_dim, classes = _class_targets(input, target, weight)
    t = _data_of(target).astype(np.int64)
    ignored = t == ignore_index
    picked = np.where(ignored, 0, t)
    bad = picked[(picked < 0) | (picked >= classes)]
    if bad.size:
        raise IndexError(f"Target {bad.reshape(-1)[0]} is out of bounds.")
    x = _data_of(input)
    w = np.ones(classes, dtype=x.dtype) if weight is None else _data_of(weight)
    target_weight = np.where(ignored, 0, w[picked]).astype(x.dtype)
    values = _squeeze(_gather(input, class_dim, Tensor(np.expand_dims(picked, class_dim))), class_dim)
    loss = _where(_constant(ignored), 0.0, _neg(_mul(values, _constant(target_weight))))
    if reduction == "none":
        return loss
    total = _sum(loss)
    return total if reduction == "sum" else _div(total, _constant(np.asarray(target_weight.sum(), dtype=x.dtype)))


def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean",
                  label_smoothing=0.0):
    """nll_loss of log_softmax over the classes (dimension 1, or 0 for one unbatched sample).
    Targets are class indices, or class probabilities of the input's shape."""
    reduction = _reduction(size_average, reduce, reduction)
    if not 0.0 <= label_smoothing <= 1.0:
        raise RuntimeError(f"label_smoothing must be between 0.0 and 1.0. Got: {label_smoothing}")
    x, t = _data_of(input), _data_of(target)
    class_dim = 0 if x.ndim == 1 else 1
    log_p = _log_softmax(input, class_dim)
    classes = x.shape[class_dim]
    shape = [1] * x.ndim
    shape[class_dim] = classes
    if t.shape == x.shape:
        return _soft_cross_entropy(log_p, target, weight, reduction, label_smoothing, class_dim, shape)
    loss = nll_loss(log_p, target, weight, None, ignore_index, None, reduction)
    if label_smoothing == 0:
        return loss
    # torch's smoothing term: the (weighted) mean of -log p over the classes, kept targets only.
    ignored = _constant(np.expand_dims(t == ignore_index, class_dim) if x.ndim > 1 else t == ignore_index)
    terms = log_p if weight is None else _mul(log_p, _reshape(weight, *shape))
    smooth = _neg(_sum(_where(ignored, 0.0, terms), class_dim))
    if reduction == "mean":
        kept = t != ignore_index
        total = _data_of(weight)[t[kept].astype(np.int64)].sum() if weight is not None else kept.sum()
        smooth = _div(_sum(smooth), _constant(np.asarray(total, dtype=x.dtype)))
    elif reduction == "sum":
        smooth = _sum(smooth)
    return _add(_mul(loss, 1 - label_smoothing), _mul(smooth, label_smoothing / classes))


def _soft_cross_entropy(log_p, target, weight, reduction, label_smoothing, class_dim, shape):
    """Cross entropy against class probabilities, as torch computes it."""
    t = _data_of(target)
    if t.dtype.kind != "f":
        raise RuntimeError(f"Expected floating point type for target with class probabilities, got {_legacy(t)}")
    classes = shape[class_dim]
    if weight is not None and (_data_of(weight).ndim != 1 or _data_of(weight).size != classes):
        raise RuntimeError(
            f"cross_entropy: weight tensor should be defined either for all {classes} classes or no classes but got "
            f"weight tensor of shape: {list(_data_of(weight).shape)}"
        )
    if label_smoothing > 0:
        target = _add(_mul(target, 1 - label_smoothing), label_smoothing / classes)
    terms = _mul(log_p, target)
    if weight is not None:
        terms = _mul(terms, _reshape(weight, *shape))
    if reduction == "none":
        return _neg(_sum(terms, class_dim))
    total = _neg(_sum(terms))
    return total if reduction == "sum" else _div(total, log_p._data.size // classes)


def _bce_input_grad(g, x, t, w):
    grad = _div(_mul(g, _sub(x, t)), _clamp(_mul(_sub(1, x), x), min=1e-12))
    return grad if w is None else _mul(grad, w)


def _bce_target_grad(g, x, t, w):
    grad = _neg(_mul(g, _log(_div(x, _sub(1, x)))))
    return grad if w is None else _mul(grad, w)


@_numeric
def _binary_cross_entropy(input, target, weight):
    x, t = input._data, target._data
    out = (t - 1) * np.maximum(np.log1p(-x), -100) - t * np.maximum(np.log(x), -100)
    if weight is not None:
        out = out * weight._data
    return _result(
        out.astype(x.dtype, copy=False), input, target, name="BinaryCrossEntropyBackward0", saved=(input, target, weight),
        grads=(_bce_input_grad, _bce_target_grad),
    )


def binary_cross_entropy(input, target, weight=None, size_average=None, reduce=None, reduction="mean"):
    """-(t * log(p) + (1 - t) * log(1 - p)), each log clamped at -100 as in torch, so p = 0
    or 1 gives a finite loss."""
    reduction = _reduction(size_average, reduce, reduction)
    x, t = _data_of(input), _data_of(target)
    if t.shape != x.shape:
        raise ValueError(
            f"Using a target size ({target.shape}) that is different to the input size ({input.shape}) is deprecated. "
            "Please ensure they have the same size."
        )
    if t.dtype != x.dtype:
        raise RuntimeError(f"Found dtype {_legacy(t)} but expected {_legacy(x)}")
    if np.any((x < 0) | (x > 1)):
        raise RuntimeError("all elements of input should be between 0 and 1")
    if weight is not None:
        weight = _constant(np.broadcast_to(_data_of(weight), x.shape))
    return _reduce(_binary_cross_entropy(input, target, weight), reduction)


def _bce_logits_grads(weight, pos_weight):
    """torch's gradients of binary_cross_entropy_with_logits, written per element: for the
    input (sigmoid(x) - t) * g, or with pos_weight ((pw * t + 1 - t) * sigmoid(x) - pw * t) * g;
    for the target -x * g, or (logsigmoid(-x) - pw * logsigmoid(x)) * g."""

    def weighted(g):
        return g if weight is None else _mul(g, weight)

    def input_grad(g, x, t):
        if pos_weight is None:
            return _mul(_sub(sigmoid(x), t), weighted(g))
        pt = _mul(pos_weight, t)
        return _mul(_sub(_mul(_sub(_add(pt, 1), t), sigmoid(x)), pt), weighted(g))

    def target_grad(g, x, t):
        if pos_weight is None:
            return _mul(_neg(x), weighted(g))
        return _mul(_sub(logsigmoid(_neg(x)), _mul(logsigmoid(x), pos_weight)), weighted(g))

    return input_grad, target_grad


@_numeric
def _bce_with_logits(input, target, weight, pos_weight):
    x, t = input._data, target._data
    log_sigmoid = np.minimum(x, 0) - np.log1p(np.exp(-np.abs(x)))
    log_weight = 1 if pos_weight is None else (pos_weight._data - 1) * t + 1
    out = (1 - t) * x - log_weight * log_sigmoid
    if weight is not None:
        out = out * weight._data
    dtype = np.result_type(x.dtype, t.dtype)
    return _result(
        out.astype(dtype, copy=False), input, target, name="BinaryCrossEntropyWithLogitsBackward0",
        saved=(input, target), grads=_bce_logits_grads(weight, pos_weight),
    )


def binary_cross_entropy_with_logits(input, target, weight=None, size_average=None, reduce=None, reduction="mean",
                                     pos_weight=None):
    """binary_cross_entropy(sigmoid(x), t) computed from the logits without overflow, as
    torch does: (1 - t) * x - (1 + (pos_weight - 1) * t) * logsigmoid(x), times weight."""
    reduction = _reduction(size_average, reduce, reduction)
    if _data_of(target).shape != _data_of(input).shape:
        raise ValueError(f"Target size ({target.shape}) must be the same as input size ({input.shape})")
    weight, pos_weight = (None if w is None else _constant(_data_of(w)) for w in (weight, pos_weight))
    return _reduce(_bce_with_logits(input, target, weight, pos_weight), reduction)


def one_hot(tensor, num_classes=-1):
    """An int64 tensor with one more dimension, of size num_classes (max + 1 by default)."""
    t = _data_of(tensor)
    if t.dtype != np.int64:
        raise RuntimeError("one_hot is only applicable to index tensor of type LongTensor.")
    if t.size and t.min() < 0:
        raise RuntimeError("Class values must be non-negative.")
    if num_classes == -1:
        if not t.size:
            raise RuntimeError("Can not infer total number of classes from empty tensor.")
        num_classes = int(t.max()) + 1
    elif t.size and t.max() >= num_classes:
        raise RuntimeError("Class values must be smaller than num_classes.")
    return Tensor((t[..., None] == np.arange(num_classes)).astype(np.int64))


# Interpolation.


def _scale(size, out, scale, dtype):
    """torch's compute_scales_value: 1 / scale_factor when one was given, else size / out."""
    return dtype(1.0 / scale) if scale else dtype(size) / dtype(out)


def _resample(input, axis, out, scale, mode, align_corners):
    """Resize one dimension: nearest copies a value, linear blends the two neighbours with
    torch's source positions and weights (computed in float32 for float32 inputs, as torch)."""
    size, dtype = input._data.shape[axis], input._data.dtype.type
    take = lambda positions: _getitem(input, (slice(None),) * axis + (Tensor(positions),))
    steps = np.arange(out)
    if mode == "nearest":
        if out == size:
            return take(steps)
        if out == 2 * size:
            return take(steps >> 1)
        source = np.floor(steps.astype(np.float32) * _scale(size, out, scale, np.float32))
        return take(np.minimum(source.astype(np.int64), size - 1))
    if align_corners:
        source = steps.astype(dtype) * (dtype(size - 1) / dtype(out - 1) if out > 1 else dtype(0))
    else:
        source = np.maximum(_scale(size, out, scale, dtype) * (steps.astype(dtype) + dtype(0.5)) - dtype(0.5), 0)
    low = source.astype(np.int64)
    high = low + (low < size - 1)
    shape = [1] * input._data.ndim
    shape[axis] = out
    weight = (source - low).astype(dtype).reshape(shape)
    return _add(_mul(take(low), _constant(1 - weight)), _mul(take(high), _constant(weight)))


_LINEAR_DIMS = {"linear": 3, "bilinear": 4, "trilinear": 5}


def _output_sizes(x, size, scale_factor, recompute_scale_factor):
    """(output sizes, scale factors or Nones) for the spatial dimensions, checked as torch."""
    dims = x.ndim - 2
    if size is not None and scale_factor is not None:
        raise ValueError("only one of size or scale_factor should be defined")
    if size is not None:
        if recompute_scale_factor:
            raise ValueError("recompute_scale_factor is not meaningful with an explicit size.")
        if isinstance(size, (list, tuple)):
            if len(size) != dims:
                raise ValueError(
                    "Input and output must have the same number of spatial dimensions, but got input with spatial "
                    f"dimensions of {list(x.shape[2:])} and output size of {size}. Please provide input tensor in "
                    "(N, C, d1, d2, ...,dK) format and output size in (o1, o2, ...,oK) format."
                )
            return [_operator.index(n) for n in size], [None] * dims
        return [_operator.index(size)] * dims, [None] * dims
    if scale_factor is None:
        raise ValueError("either size or scale_factor should be defined")
    factors = list(scale_factor) if isinstance(scale_factor, (list, tuple)) else [scale_factor] * dims
    if len(factors) != dims:
        raise ValueError(
            "Input and scale_factor must have the same number of spatial dimensions, but got input with spatial "
            f"dimensions of {list(x.shape[2:])} and scale_factor of shape {scale_factor}. Please provide input "
            "tensor in (N, C, d1, d2, ...,dK) format and scale_factor in (s1, s2, ...,sK) format."
        )
    sizes = [int(n * f) for n, f in zip(x.shape[2:], factors)]
    return sizes, ([None] * dims if recompute_scale_factor else factors)


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None, recompute_scale_factor=None,
                antialias=False):
    """Resize the spatial dimensions (everything after N and C) with mode "nearest", "linear"
    (3D input), "bilinear" (4D) or "trilinear" (5D), one dimension at a time."""
    x = _data_of(input)
    if mode in ("nearest", "area", "nearest-exact"):
        if align_corners is not None:
            raise ValueError(
                "align_corners option can only be set with the interpolating modes: linear | bilinear | bicubic | trilinear"
            )
    elif align_corners is None:
        align_corners = False
    sizes, scales = _output_sizes(x, size, scale_factor, recompute_scale_factor)
    if mode in _LINEAR_DIMS and x.ndim != _LINEAR_DIMS[mode]:
        raise NotImplementedError(f"Got {x.ndim}D input, but {mode} mode needs {_LINEAR_DIMS[mode]}D input")
    if (mode != "nearest" and mode not in _LINEAR_DIMS) or antialias:
        raise _not_available(f"torch.nn.functional.interpolate with {'antialias=True' if antialias else f'mode={mode!r}'}")
    if x.ndim not in (3, 4, 5):
        raise NotImplementedError(f"Input Error: Only 3D, 4D and 5D input Tensors supported (got {x.ndim}D) for the modes: nearest | linear | bilinear | bicubic | trilinear | area | nearest-exact (got {mode})")
    if min(sizes) <= 0:
        raise RuntimeError(
            f"Input and output sizes should be greater than 0, but got input {tuple(x.shape[2:])} output {tuple(sizes)}"
        )
    out = input
    for axis in reversed(range(len(sizes))):  # the last dimension first, as torch's kernels weigh
        out = _resample(out, 2 + axis, sizes[axis], scales[axis], "nearest" if mode == "nearest" else "linear", align_corners)
    return out


__getattr__ = _module_getattr(__name__, "torch.nn.functional")
