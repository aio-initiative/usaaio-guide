"""Unary maths, clamp, where, casts and in-place updates against real torch."""

import numpy as np
import pytest

from parity import A, B, INT, MASK, POS, V, cases, check, check_raises

ROUNDS = np.array([0.5, 1.5, 2.5, -0.5, -1.4, 3.14159], dtype=np.float32)
WIDE = np.array([-800.0, -20.0, -1.0, 0.0, 1.0, 20.0, 800.0], dtype=np.float32)
UNARY = ["exp", "sin", "cos", "tanh", "sigmoid", "abs", "relu", "neg"]


@pytest.mark.parametrize("name", UNARY)
def test_unary_function_and_method(st, real, name):
    for array in (A, INT, A.astype(np.float64), WIDE):
        check(st, real, lambda t, x: getattr(x, name)(), array)
        check(st, real, lambda t, x: getattr(t, name)(x), array)


@pytest.mark.parametrize("name", ["log", "sqrt"])
def test_unary_on_positive_values(st, real, name):
    check(st, real, lambda t, x: getattr(t, name)(x), POS)
    check(st, real, lambda t, x: getattr(x, name)(), np.abs(INT) + 1)
    # log(0) is -inf and sqrt(-1) is nan, silently, as in torch.
    check(st, real, lambda t, x: getattr(t, name)(x), np.array([0.0, -1.0, 4.0], dtype=np.float32))


@cases(
    ("exp of bool", lambda t, x: x.exp(), MASK),
    ("sigmoid of int", lambda t, x: t.sigmoid(x), INT),
    ("round", lambda t, x: x.round(), ROUNDS),
    ("round int", lambda t, x: x.round(), INT),
    ("round decimals", lambda t, x: x.round(decimals=2), A),
    ("torch.round", lambda t, x: t.round(x), ROUNDS),
    ("clamp both", lambda t, x: x.clamp(-0.5, 0.5), A),
    ("clamp min", lambda t, x: x.clamp(min=0), A),
    ("clamp max", lambda t, x: x.clamp(max=0.25), A),
    ("clamp int", lambda t, x: x.clamp(0), INT),
    ("clamp int by float", lambda t, x: x.clamp(0.5), INT),
    ("clamp int max float", lambda t, x: x.clamp(max=2.5), INT),
    ("clamp by tensors", lambda t, x, lo: x.clamp(lo, lo + 1), A, B),
    ("torch.clamp", lambda t, x: t.clamp(x, min=-1, max=1), A),
    ("clamp_min", lambda t, x: x.clamp_min(0.1), A),
    ("clip", lambda t, x: x.clip(-0.5, 0.5), A),
    ("torch.clip", lambda t, x: t.clip(x, max=0), A),
    ("where", lambda t, x, y: t.where(x > 0, x, y), A, B),
    ("where numbers", lambda t, x: t.where(x > 0, 1.0, 0.0), A),
    ("where ints", lambda t, x: t.where(x > 0, 1, 0), A),
    ("where tensor and number", lambda t, x: t.where(x > 0, x, 0), A),
    ("where number and tensor", lambda t, x: t.where(x > 0, 0.5, x), INT),
    ("where int and float tensors", lambda t, x, y: t.where(y > 0, x, y), INT, A),
    ("where broadcast", lambda t, x, v: t.where(v > 0, x, v), A, V),
    ("where bools", lambda t, x: t.where(x, x, ~x), MASK),
    ("where one argument", lambda t, x: t.where(x > 0), A),
    ("where one argument 1-d", lambda t, x: t.where(x > 0)[0], V),
    ("method where", lambda t, x: x.where(x > 0, 0.0), A),
    ("masked_fill", lambda t, x: x.masked_fill(x > 0, -1.0), A),
    ("masked_fill inf", lambda t, x: x.masked_fill(x < 0, float("-inf")), A),
    ("masked_fill int", lambda t, x: x.masked_fill(x > 0, 9), INT),
    ("masked_fill broadcast", lambda t, x, v: x.masked_fill(v > 0, 0.0), A, V),
)
def test_maths(st, real, case):
    check(st, real, *case[1:])


@cases(
    ("float", lambda t, x: x.float(), INT),
    ("double", lambda t, x: x.double(), A),
    ("long truncates", lambda t, x: (x * 3).long(), A),
    ("int", lambda t, x: (x * 3).int(), A),
    ("bool", lambda t, x: x.bool(), INT),
    ("bool to float", lambda t, x: x.float(), MASK),
    ("to dtype", lambda t, x: x.to(t.float64), A),
    ("to dtype keyword", lambda t, x: x.to(dtype=t.long), A),
    ("to device string", lambda t, x: x.to("cpu"), A),
    ("to device object", lambda t, x: x.to(t.device("cpu")), A),
    ("to device and dtype", lambda t, x: x.to("cpu", t.float64), A),
    ("to device keyword", lambda t, x: x.to(device="cpu", dtype=t.float32, non_blocking=True), INT),
    ("to other tensor", lambda t, x, y: x.to(y), INT, A),
    ("same dtype is the same tensor", lambda t, x: (x.float() is x, x.to(t.float32) is x, x.to("cpu") is x), A),
    ("clone", lambda t, x: x.clone(), A),
    ("torch.clone", lambda t, x: t.clone(x), INT),
)
def test_casts(st, real, case):
    check(st, real, *case[1:])


def test_clone_does_not_share_memory(st):
    x = st.tensor([1.0, 2.0])
    y = x.clone()
    y.zero_()
    assert x.tolist() == [1.0, 2.0] and y.tolist() == [0.0, 0.0]


@cases(
    ("add_", lambda t, x, y: x.add_(y), A, B),
    ("add_ number", lambda t, x: x.add_(1), INT),
    ("add_ alpha", lambda t, x, y: x.add_(y, alpha=-0.5), A, B),
    ("sub_", lambda t, x, y: x.sub_(y), A, B),
    ("mul_", lambda t, x: x.mul_(0.5), A),
    ("div_", lambda t, x: x.div_(4), A),
    ("float32 stays float32", lambda t, x: x.add_(x.double()), A),
    ("uint8 wraps", lambda t, x: x.to(t.uint8).add_(250), np.abs(INT)),
    ("zero_", lambda t, x: x.zero_(), A),
    ("fill_", lambda t, x: x.fill_(2.5), A),
    ("fill_ truncates", lambda t, x: x.fill_(2.7), INT),
    ("copy_", lambda t, x, y: x.copy_(y), A, B),
    ("copy_ casts", lambda t, x, y: x.copy_(y * 3), INT, A),
    ("copy_ broadcasts", lambda t, x, v: x.copy_(v), A, V),
    ("iadd", lambda t, x, y: x.__iadd__(y), A, B),
    ("isub number", lambda t, x: x.__isub__(2), INT),
    ("imul", lambda t, x, v: x.__imul__(v), A, V),
    ("itruediv", lambda t, x: x.__itruediv__(2.0), A),
    ("returns self", lambda t, x: (x.add_(1) is x, x.zero_() is x, x.fill_(1) is x, x.copy_(x) is x), A),
)
def test_in_place(st, real, case):
    check(st, real, *case[1:])


def test_in_place_updates_reach_views_and_keep_identity(st):
    x = st.tensor([[1.0, 2.0], [3.0, 4.0]])
    array, alias = x.numpy(), x
    x += 1
    x *= 2
    x -= st.tensor([1.0, 1.0])
    x /= 2
    assert x is alias and x.numpy() is array and x.tolist() == [[1.5, 2.5], [3.5, 4.5]]
    row = x[0]
    row.zero_()
    assert x.tolist() == [[0.0, 0.0], [3.5, 4.5]]


@cases(
    ("float into int", lambda t, x: x.add_(1.5), INT),
    ("iadd float into int", lambda t, x: x.__iadd__(1.5), INT),
    ("int into bool", lambda t, x: x.add_(1), MASK),
    ("true division of ints", lambda t, x: x.div_(2), INT),
    ("shape grows", lambda t, x, v: v.add_(x), A, V),
    ("softmax without dim", lambda t, x: t.softmax(x), A),
    ("clamp without bounds", lambda t, x: x.clamp(), A),
    ("to an unknown device", lambda t, x: x.to("gpu"), A),
)
def test_maths_errors(st, real, case):
    check_raises(st, real, *case[1:])
