"""Elementwise operations, and the two functions every operation's output passes through.

result(data, *parents, name=..., grads=..., saved=...) returns the output of a
differentiable operation. It is the single place where autograd attaches a backward
function, so an operation that gradients flow through must return result(...), never
Tensor(...), and give its derivative: `grads` holds one formula per parent (None for a
parent with no gradient), each called as formula(grad, *saved) with tensor operations.
Operations with no gradient (comparisons, indices, counts) build a Tensor directly.
Operations written in terms of other operations (masked_fill, the in-place forms) need
nothing: they inherit from what they call.

write(target, index, value) is the single place where a tensor's values change in place.
"""

import functools

import numpy as np

from . import _autograd, _dtype
from ._autograd import OUTPUT
from ._tensor import Tensor, require_cpu


def numeric(function):
    """Make a NumPy computation fail and warn the way torch does.

    torch computes inf and nan silently, so NumPy's floating point warnings are off while
    `function` runs (and only then: the caller's own NumPy is untouched). And shapes that do
    not broadcast are a RuntimeError in torch where NumPy raises ValueError.
    """

    @functools.wraps(function)
    def run(*args, **kwargs):
        try:
            with np.errstate(all="ignore"):
                return function(*args, **kwargs)
        except ValueError as err:
            raise RuntimeError(str(err)) from None

    return run


def result(data, *parents, name=None, grads=(), saved=()):
    """The output tensor of a differentiable operation whose inputs were `parents`.

    `parents` are the operands as the caller gave them, so they can include plain numbers.
    `name` is its grad_fn's name, `grads` a formula per parent and `saved` the values the
    formulas read (tensors are checked for in-place changes; OUTPUT stands for the result).
    """
    out = Tensor(data)
    _autograd.record(out, parents, name, grads, saved)
    return out


@numeric
def write(target, index, value, **how):
    """Set target[index] = value in place, cast to target's dtype, and return target.
    `index` is a NumPy key; `how` passes ints, name, indexed and adopt to _autograd.write."""
    key = index if isinstance(index, tuple) else (index,)
    return _autograd.write(target, key, value, **how)


def for_grad_of(value, other):
    """`value` when `other` will need a gradient, else None. Torch saves an operand only for
    the gradient that reads it (a product saves y for x's gradient), so only then does
    changing it in place make the backward pass fail."""
    if _autograd.is_grad_enabled() and isinstance(other, Tensor) and _autograd.requires_grad(other):
        return value
    return None


def constant(array):
    """A tensor no gradient flows into: a mask or a factor computed from the forward values."""
    return Tensor(np.asarray(array))


def masked(mask, grad):
    """`grad` where `mask` holds and 0 elsewhere. A where, not a product with the mask, so an
    inf in `grad` gives 0 where the mask is off (inf * 0 would be nan), as torch's do."""
    return where(constant(mask), grad, 0)


def shared(share, grad):
    """grad * share for a share of 0, 1/2 or 1 per element, and exactly 0 where it is 0."""
    share = np.asarray(share)
    if np.all((share == 0) | (share == 1)):
        return masked(share != 0, grad)
    return masked(share != 0, mul(grad, constant(share.astype(grad._data.dtype))))


def zeros_for(grad):
    """A zero gradient of grad's shape and dtype (not grad * 0, which turns inf into nan)."""
    return constant(np.zeros(grad._data.shape, grad._data.dtype))


def data_of(input):
    """The array of an argument that has to be a tensor, or torch's TypeError.

    The rule across the package: a public function reads each tensor argument with
    data_of(...) the first time, which checks it is a tensor; after that, and for tensors the
    package made itself (results, saved values, gradients), it reads `._data` directly.
    """
    if not isinstance(input, Tensor):
        raise TypeError(f"expected a Tensor, not {type(input).__name__}")
    return input._data


def _kind(value):
    """(NumPy dtype, rank) of an operand; rank 0 is a tensor with dimensions, 1 a 0-d tensor
    and 2 a number. NumPy scalars count as numbers and NumPy arrays as tensors, as in torch."""
    if isinstance(value, (Tensor, np.ndarray)):
        data = value._data if isinstance(value, Tensor) else value
        return data.dtype, (0 if data.ndim else 1)
    if isinstance(value, (bool, np.bool_)):
        return _dtype.BOOL, 2
    if isinstance(value, (int, np.integer)):
        return _dtype.INT64, 2
    if isinstance(value, (float, np.floating)):
        return _dtype.FLOAT32, 2
    raise TypeError(f"unsupported operand type for a tensor operation: {type(value).__name__}")


def result_dtype(*values):
    """The NumPy dtype torch gives an operation on these tensors and numbers."""
    groups = [None, None, None]
    for value in values:
        dtype, rank = _kind(value)
        groups[rank] = dtype if groups[rank] is None else _dtype.promote(groups[rank], dtype)
    return _dtype.result_type(*groups)


def as_array(value, dtype):
    """An operand as an array of `dtype`. A Python int that does not fit wraps, as in torch."""
    if isinstance(value, Tensor):
        value = value._data
    return np.asarray(value).astype(dtype, copy=False)


def _operands(a, b, floating=False):
    """Both operands as arrays of the result dtype (a float one if `floating`)."""
    if isinstance(a, Tensor) and isinstance(b, Tensor) and a._data.dtype == b._data.dtype:
        if not floating or a._data.dtype.kind == "f":
            return a._data, b._data
    dtype = result_dtype(a, b)
    if floating and dtype.kind != "f":
        dtype = _dtype.FLOAT32
    return as_array(a, dtype), as_array(b, dtype)


@numeric
def add(input, other, *, alpha=1):
    x, y = _operands(input, other)
    scaled = (lambda g: g) if alpha == 1 else (lambda g: mul(g, alpha))
    return result(np.add(x, y if alpha == 1 else y * alpha), input, other, name="AddBackward0", grads=(lambda g: g, scaled))


def _is_bool(value):
    return isinstance(value, (bool, np.bool_)) or (isinstance(value, Tensor) and value._data.dtype.kind == "b")


@numeric
def sub(input, other):
    if _is_bool(input) or _is_bool(other):
        raise NotImplementedError(
            "Subtraction, the `-` operator, with a bool tensor is not supported. "
            "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead."
        )
    x, y = _operands(input, other)
    return result(np.subtract(x, y), input, other, name="SubBackward0", grads=(lambda g: g, neg))


@numeric
def mul(input, other):
    x, y = _operands(input, other)
    return result(
        np.multiply(x, y), input, other, name="MulBackward0",
        saved=(for_grad_of(input, other), for_grad_of(other, input)),
        grads=(lambda g, a, b: mul(g, b), lambda g, a, b: mul(g, a)),
    )


@numeric
def div(input, other):
    """True division: integers and bools divide to float32, so tensor([3]) / 2 is 1.5."""
    x, y = _operands(input, other, floating=True)
    return result(
        np.true_divide(x, y), input, other, name="DivBackward0", saved=(for_grad_of(input, other), other),
        grads=(lambda g, a, b: div(g, b), lambda g, a, b: neg(div(mul(g, a), mul(b, b)))),
    )


def _not_differentiable(name):
    def fail(g, *saved):
        raise RuntimeError(f"derivative for aten::{name} is not implemented")

    return fail


def _no_integer_zero(y):
    if y.dtype.kind in "iu" and not np.all(y):
        raise RuntimeError("ZeroDivisionError")


@numeric
def floor_divide(input, other):
    x, y = _operands(input, other)
    _no_integer_zero(y)
    fail = _not_differentiable("floor_divide")
    return result(np.floor_divide(x, y), input, other, name="FloorDivideBackward0", grads=(fail, fail))


@numeric
def remainder(input, other):
    """Python's %: the result takes the sign of the divisor."""
    x, y = _operands(input, other)
    _no_integer_zero(y)
    quotient = constant(np.floor_divide(x, y))
    # torch differentiates tensor % tensor in both, but number % tensor not at all.
    divisor = _not_differentiable("remainder") if not isinstance(input, Tensor) else lambda g: neg(mul(g, quotient))
    return result(
        np.remainder(x, y), input, other, name="RemainderBackward0" if isinstance(input, Tensor) else "RemainderBackward1",
        grads=(lambda g: g, divisor),
    )


def _power_rule(g, base, exponent):
    """g * exponent * base ** (exponent - 1), and 0 where the exponent is 0, as in torch."""
    if not isinstance(exponent, Tensor):
        if exponent == 0:
            return zeros_for(g)
        return mul(g, mul(pow(base, exponent - 1), exponent))
    return masked(exponent._data != 0, mul(g, mul(exponent, pow(base, sub(exponent, 1)))))


def _exponent_rule(g, base, exponent, out):
    """g * base ** exponent * log(base), and 0 where the base is 0 and the exponent is not
    negative, as in torch."""
    if isinstance(base, Tensor):
        zero = (base._data == 0) & (np.asarray(exponent._data) >= 0)
        return mul(g, masked(~zero, mul(out, log(base))))  # torch's order: nan for an inf g at 0
    if base == 0:
        return masked(exponent._data < 0, mul(g, mul(out, float("-inf"))))
    return mul(g, mul(out, float(np.log(base))))


@numeric
def pow(input, exponent):
    if type(exponent) in (int, float) and isinstance(input, Tensor) and input._data.dtype.kind == "f":
        # NumPy is several times faster with a plain number here (x ** 2 is one multiplication).
        return result(
            input._data**exponent, input, exponent, name="PowBackward0", saved=(input,),
            grads=(lambda g, a: _power_rule(g, a, exponent), None),
        )
    x, y = _operands(input, exponent)
    if x.dtype.kind in "iu":
        if isinstance(exponent, (int, np.integer)) and exponent < 0:
            raise RuntimeError("Integers to negative integer powers are not allowed.")
        if y.dtype.kind == "i" and np.any(y < 0):
            # A tensor of exponents may be negative: the integer result truncates, 2 ** -1 is 0.
            return result(np.power(x.astype(np.float64), y).astype(x.dtype), input, exponent)
    return result(
        np.power(x, y), input, exponent, name="PowBackward1", saved=(input, exponent, OUTPUT),
        grads=(lambda g, a, e, out: _power_rule(g, a, e), lambda g, a, e, out: _exponent_rule(g, a, e, out)),
    )


def neg(input):
    x = data_of(input)
    if x.dtype.kind == "b":
        raise NotImplementedError(
            "Negation, the `-` operator, on a bool tensor is not supported. "
            "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead."
        )
    return result(np.negative(x), input, name="NegBackward0", grads=(neg,))


def abs(input):
    x = data_of(input)
    return result(
        np.abs(x), input, name="AbsBackward0", saved=(input,), grads=(lambda g, a: mul(g, constant(np.sign(a._data))),)
    )


def _float_array(input):
    """The array of a tensor, as float32 unless it is already a float."""
    x = data_of(input)
    return x if x.dtype.kind == "f" else x.astype(_dtype.FLOAT32)


def _unary(function, name, saves, formula):
    """A torch function that applies a NumPy float function to every element. Its gradient
    is formula(grad, saved), where the saved value is the input or (OUTPUT) the result."""

    def apply(input):
        saved = OUTPUT if saves is OUTPUT else input
        return result(function(_float_array(input)), input, name=name, saved=(saved,), grads=(formula,))

    apply.__name__ = apply.__qualname__ = function.__name__
    return numeric(apply)


exp = _unary(np.exp, "ExpBackward0", OUTPUT, lambda g, out: mul(g, out))
log = _unary(np.log, "LogBackward0", "input", lambda g, x: div(g, x))
sqrt = _unary(np.sqrt, "SqrtBackward0", OUTPUT, lambda g, out: div(g, mul(out, 2)))
sin = _unary(np.sin, "SinBackward0", "input", lambda g, x: mul(g, cos(x)))
cos = _unary(np.cos, "CosBackward0", "input", lambda g, x: neg(mul(g, sin(x))))
tanh = _unary(np.tanh, "TanhBackward0", OUTPUT, lambda g, out: mul(g, sub(1, mul(out, out))))


@numeric
def sigmoid(input):
    x = _float_array(input)
    # exp of a non-positive number cannot overflow, so both branches are safe for any x.
    e = np.exp(-np.abs(x))
    return result(
        np.where(x >= 0, 1 / (1 + e), e / (1 + e)).astype(x.dtype, copy=False), input,
        name="SigmoidBackward0", saved=(OUTPUT,), grads=(lambda g, out: mul(g, mul(out, sub(1, out))),),
    )


def round(input, *, decimals=0):
    """Halves go to the even neighbour, as in torch and NumPy. Its gradient is 0."""
    return result(np.round(data_of(input), decimals), input, name="RoundBackward0", grads=(zeros_for,))


@numeric
def clamp(input, min=None, max=None):
    if min is None and max is None:
        raise RuntimeError("torch.clamp: At least one of 'min' or 'max' must not be None")
    bounds = [bound for bound in (min, max) if bound is not None]
    dtype = result_dtype(input, *bounds)
    low, high = (None if bound is None else as_array(bound, dtype) for bound in (min, max))
    x = data_of(input).astype(dtype, copy=False)
    out = np.clip(x, low, high)
    # As in torch: the input's gradient passes strictly inside the bounds, and a tensor bound
    # gets it where the input is beyond it. At a tie with a tensor bound the two share it,
    # unless both bounds are tensors and equal there, when the input keeps all of it.
    tensors = [isinstance(bound, Tensor) for bound in (min, max)]
    pinched = all(tensors) and low is not None and high is not None
    pinched = (low == high) if pinched else np.zeros((), dtype=bool)
    share_x = np.ones(np.broadcast_shapes(x.shape, out.shape))
    shares = []
    for bound, is_tensor, outside in ((low, tensors[0], np.less), (high, tensors[1], np.greater)):
        if bound is None:
            continue
        tie = (x == bound) & ~pinched
        share_x = share_x * ~(outside(x, bound) | (x == bound)) + (0.5 * tie if is_tensor else 0)
        if is_tensor:
            shares.append(outside(x, bound) + 0.5 * tie)
    share_x = np.where(pinched & (x == low), 1.0, share_x)
    grads = [lambda g, x, share=share: shared(share, g) for share in shares]
    return result(
        out, input, *bounds, name="ClampBackward1", saved=(input,), grads=(lambda g, x: shared(share_x, g), *grads)
    )


clip = clamp


@numeric
def maximum(input, other):
    """The larger value at every position (torch.max(a, b)); a nan wins, as in torch."""
    x, y = _operands(input, other)
    return _extreme_of_two(np.maximum(x, y), input, other, x > y, y > x, "MaximumBackward0")


@numeric
def minimum(input, other):
    x, y = _operands(input, other)
    return _extreme_of_two(np.minimum(x, y), input, other, x < y, y < x, "MinimumBackward0")


def _extreme_of_two(out, input, other, x_wins, y_wins, name):
    """The gradient goes to the operand that was chosen, half to each where they tie."""
    tie = ~x_wins & ~y_wins
    share_x, share_y = x_wins + 0.5 * tie, y_wins + 0.5 * tie
    return result(
        out, input, other, name=name, saved=(input, other),
        grads=(lambda g, a, b: shared(share_x, g), lambda g, a, b: shared(share_y, g)),
    )


def clamp_min(input, min):
    return clamp(input, min=min)


@numeric
def relu(input):
    """max(x, 0). Its gradient is 0 at x == 0, as in torch (clamp passes it there)."""
    x = data_of(input)
    if x.dtype.kind != "f":
        return clamp(input, min=0)
    out = np.maximum(x, 0).astype(x.dtype, copy=False)
    positive = out > 0
    return result(out, input, name="ReluBackward0", saved=(OUTPUT,), grads=(lambda g, out: masked(positive, g),))


@numeric
def where(condition, input=None, other=None):
    """where(condition, a, b) picks a where the condition holds, else b.
    where(condition) alone returns the indices where it holds, one tensor per dimension."""
    mask = data_of(condition)
    if input is None and other is None:
        return tuple(Tensor(index.astype(np.int64, copy=False)) for index in np.nonzero(mask))
    dtype = result_dtype(input, other)
    chosen = mask.astype(bool).copy()
    return result(
        np.where(mask, as_array(input, dtype), as_array(other, dtype)), input, other, name="WhereBackward0",
        grads=(lambda g: masked(chosen, g), lambda g: masked(~chosen, g)),
    )


def masked_fill(input, mask, value):
    """A copy with `value` wherever the mask is true; the dtype stays that of `input`."""
    return cast(where(mask, value, input), data_of(input).dtype)


@numeric
def cast(input, dtype):
    """`input` as the NumPy dtype `dtype`; the same tensor when it already has that dtype."""
    x = data_of(input)
    return input if x.dtype == dtype else result(x.astype(dtype), input, name="ToCopyBackward0", grads=(lambda g: g,))


def clone(input):
    return result(data_of(input).copy(), input, name="CloneBackward0", grads=(lambda g: g,))


def to(input, *args, dtype=None, device=None, non_blocking=False, copy=False):
    """tensor.to(dtype), .to(device), .to(device, dtype) or .to(other_tensor)."""
    for arg in args:
        if isinstance(arg, Tensor):
            dtype = arg.dtype
        elif isinstance(arg, (_dtype.dtype, type)):
            dtype = arg
        elif not isinstance(arg, bool):  # a positional bool is non_blocking
            device = arg
    require_cpu(device)
    out = input if dtype is None else cast(input, _dtype.as_numpy(dtype))
    return clone(out) if copy and out is input else out
