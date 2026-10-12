"""Elementwise maths, comparisons, casts and in-place updates against real torch."""

import itertools
import operator

import numpy as np
import pytest

from parity import A, B, INT, MASK, POS, V, agree, cases, check, check_raises, same

# Real torch itself trips a NumPy deprecation when a tensor meets an array; not under test.
pytestmark = pytest.mark.filterwarnings("ignore:__array_wrap__ must accept context")

DTYPES = ["bool", "uint8", "int8", "int16", "int32", "int64", "float32", "float64"]
NANS = np.array([1.0, float("nan"), float("inf"), -float("inf"), 0.0], dtype=np.float32)


def test_review_focus_integer_division_is_float(st, real):
    got = st.tensor([3]) / 2
    assert got.dtype is st.float32 and got.tolist() == [1.5]
    same(got, real.tensor([3]) / 2)
    same(st.tensor([3, -7]) // 2, real.tensor([3, -7]) // 2)
    same(st.tensor([7, 8]) / st.tensor([2, 4]), real.tensor([7, 8]) / real.tensor([2, 4]))


def test_review_focus_comparisons_give_bool_masks(st, real):
    x, rx = st.tensor(A), real.tensor(A)
    mask = x > 0
    assert mask.dtype is st.bool
    same(mask, rx > 0)
    same(mask.sum(), (rx > 0).sum())
    same(x[mask], rx[rx > 0])
    same(mask.float().mean(), (rx > 0).float().mean())
    same((x.argmax(dim=1) == st.tensor([0, 1, 2])).sum().item(), (rx.argmax(dim=1) == real.tensor([0, 1, 2])).sum().item())


BINARY = [operator.add, operator.sub, operator.mul, operator.truediv, operator.floordiv, operator.mod, operator.pow]


@pytest.mark.parametrize("op", BINARY, ids=lambda op: op.__name__)
@pytest.mark.parametrize("left,right", list(itertools.product(DTYPES[1:], DTYPES[1:])))
def test_binary_result_dtype_and_values(st, real, op, left, right):
    def run(t):
        a = t.tensor([[7, 2, 9], [4, 5, 6]], dtype=getattr(t, left))
        b = t.tensor([2, 1, 3], dtype=getattr(t, right))
        return op(a, b)

    agree(run, st, real)


SCALARS = [2, 2.5, True, -3]


@pytest.mark.parametrize("op", BINARY, ids=lambda op: op.__name__)
@pytest.mark.parametrize("dtype", ["int64", "int32", "float32", "float64", "uint8"])
@pytest.mark.parametrize("scalar", SCALARS)
def test_python_scalars_do_not_widen(st, real, op, dtype, scalar):
    def run(t, flip):
        a = t.tensor([[7, 2, 9], [4, 5, 6]], dtype=getattr(t, dtype))
        return op(scalar, a) if flip else op(a, scalar)

    # Where torch refuses (an integer to a negative power, a bool in a subtraction), so do we.
    agree(lambda t: run(t, False), st, real)
    agree(lambda t: run(t, True), st, real)


@cases(
    ("0-d float64 does not widen float32", lambda t, x: x * t.tensor(2.0, dtype=t.float64), A),
    ("0-d float32 lifts ints", lambda t, x: x * t.tensor(2.5), INT),
    ("0-d float64 lifts ints to float64", lambda t, x: x * t.tensor(2.5, dtype=t.float64), INT),
    ("0-d int64 keeps int32", lambda t, x: x.int() + t.tensor(5), INT),
    ("numpy scalar is a Python number", lambda t, x: x * np.float64(2.0), A),
    ("numpy scalar on the left", lambda t, x: np.float64(2.0) * x, A),
    ("numpy array is a tensor", lambda t, x: x * np.ones(4), A),
    ("broadcast", lambda t, x, v: x + v, A, V),
    ("broadcast column", lambda t, x: x - x[:, :1], A),
    ("neg", lambda t, x: -x, A),
    ("neg int", lambda t, x: -x, INT),
    ("pos", lambda t, x: +x, A),
    ("abs operator", lambda t, x: abs(x), A),
    ("invert bool", lambda t, x: ~x, MASK),
    ("invert int", lambda t, x: ~x, INT),
    ("and", lambda t, x, y: (x > 0) & (y > 0), A, B),
    ("or", lambda t, x, y: (x > 0) | (y > 0), A, B),
    ("xor", lambda t, x, y: (x > 0) ^ (y > 0), A, B),
    ("int and", lambda t, x: x & 3, INT),
    ("bool plus bool", lambda t, x: x + x, MASK),
    ("bool times bool", lambda t, x: x * x, MASK),
    ("bool plus int", lambda t, x: x + 1, MASK),
    ("bool divide", lambda t, x: x / 2, MASK),
    ("divide by zero", lambda t, x: x / 0, INT),
    ("float floor divide", lambda t, x: x // 0.4, A),
    ("negative remainder", lambda t, x: x % -3, INT),
    ("float remainder", lambda t, x: x % 0.7, A),
    ("int to float power", lambda t, x: x.abs() ** 0.5, INT),
    ("float base", lambda t, x: 2.0 ** x, INT),
    ("int base", lambda t, x: 2 ** x.abs(), INT),
    ("negative float power", lambda t, x: x ** -1.0, POS),
    ("square", lambda t, x: x**2, A),
    ("functions", lambda t, x, y: (t.add(x, y), t.sub(x, y), t.mul(x, y), t.div(x, y)), A, B),
    ("methods", lambda t, x, y: (x.add(y), x.sub(y), x.mul(y), x.div(y), x.pow(2), x.neg()), A, B),
)
def test_arithmetic(st, real, case):
    check(st, real, *case[1:])


COMPARE = [operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge]


@pytest.mark.parametrize("op", COMPARE, ids=lambda op: op.__name__)
def test_comparisons(st, real, op):
    check(st, real, lambda t, x, y: op(x, y), A, B)
    check(st, real, lambda t, x: op(x, 0), INT)
    check(st, real, lambda t, x: op(0.5, x), A)
    check(st, real, lambda t, x, y: op(x, y), INT, A)
    check(st, real, lambda t, x: op(x, x[0]), INT)


@cases(
    ("eq None", lambda t, x: x == None, A),  # noqa: E711
    ("ne None", lambda t, x: x != None, A),  # noqa: E711
    ("eq string", lambda t, x: x == "a", A),
    ("in list", lambda t, x: x in [x], A),
    ("functions", lambda t, x, y: (t.eq(x, y), t.ne(x, y), t.lt(x, y), t.gt(x, 0), x.le(y), x.ge(0)), A, B),
    ("isnan", lambda t, x: t.isnan(x), NANS),
    ("isnan int", lambda t, x: t.isnan(x), INT),
    ("isfinite", lambda t, x: t.isfinite(x), NANS),
    ("allclose", lambda t, x: t.allclose(x, x + 1e-9), A),
    ("allclose far", lambda t, x: t.allclose(x, x + 0.1), A),
    ("allclose tolerances", lambda t, x: t.allclose(x, x + 0.1, atol=0.2, rtol=0), A),
    ("allclose nan", lambda t, x: (t.allclose(x, x), t.allclose(x, x, equal_nan=True)), NANS),
    ("allclose broadcast", lambda t, x: t.allclose(x, x[:1].expand(3, 4)), A),
    ("method allclose", lambda t, x: x.allclose(x), A),
    ("equal", lambda t, x: (t.equal(x, x), t.equal(x, x[:1]), t.equal(x, x + 1), x.equal(x)), A),
    ("equal across dtypes", lambda t, x: t.equal(x, x.float()), INT),
)
def test_comparison_helpers(st, real, case):
    check(st, real, *case[1:])


@cases(
    ("bool minus bool", lambda t, x: x - x, MASK),
    ("neg bool", lambda t, x: -x, MASK),
    ("invert float", lambda t, x: ~x, A),
    ("int to a negative power", lambda t, x: x**-1, INT),
    ("int floor divide by zero", lambda t, x: x // 0, INT),
    ("tensor plus list", lambda t, x: x + [1.0, 2.0, 3.0, 4.0], A),
    ("shapes do not broadcast", lambda t, x: x + x[:, :3], A),
    ("allclose with a number", lambda t, x: t.allclose(x, 1.5), A),
    ("allclose across dtypes", lambda t, x: t.allclose(x, x.double()), A),
)
def test_arithmetic_errors(st, real, case):
    check_raises(st, real, *case[1:])
