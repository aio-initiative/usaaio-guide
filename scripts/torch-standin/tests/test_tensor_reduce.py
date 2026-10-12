"""Reductions against real torch: values, dtypes, dim and keepdim, and the named results."""

import numpy as np
import pytest

from parity import A, B, CUBE, INT, MASK, POS, SQUARE, TIES, V, cases, check, check_raises, same

# Both libraries warn that the variance of one value divides by zero; that is not under test.
pytestmark = pytest.mark.filterwarnings("ignore:.*degrees of freedom is <= 0")

REDUCERS = ["sum", "mean", "amax", "amin", "std", "var", "max", "min", "argmax", "argmin", "all", "any", "median", "mode"]
ARGUMENTS = [
    ("whole", lambda f: f()),
    ("dim", lambda f: f(1)),
    ("dim keyword", lambda f: f(dim=0)),
    ("negative dim", lambda f: f(-1)),
    ("keepdim", lambda f: f(1, keepdim=True)),
    ("positional keepdim", lambda f: f(0, True)),
    ("numpy names", lambda f: f(axis=1, keepdims=True)),
]


@pytest.mark.parametrize("name", REDUCERS)
@pytest.mark.parametrize("call", ARGUMENTS, ids=[label for label, _ in ARGUMENTS])
def test_reduction_method_and_function(st, real, name, call):
    for array in (A, CUBE, TIES):
        check(st, real, lambda t, x: call[1](getattr(x, name)), array)
        check(st, real, lambda t, x: call[1](lambda *a, **k: getattr(t, name)(x, *a, **k)), array)


@cases(
    ("sum of ints", lambda t, x: (x.sum(), x.sum(0)), INT),
    ("sum of bools counts", lambda t, x: (x.sum(), x.sum(dim=1)), MASK),
    ("sum of uint8 is int64", lambda t, x: x.to(t.uint8).sum(), np.abs(INT)),
    ("sum of int32 is int64", lambda t, x: x.int().sum(1), INT),
    ("sum over two dims", lambda t, x: x.sum((0, 2)), CUBE),
    ("sum over a list of dims", lambda t, x: x.sum(dim=[0, 1], keepdim=True), CUBE),
    ("sum dtype", lambda t, x: x.sum(dtype=t.float64), A),
    ("mean over two dims", lambda t, x: t.mean(x, dim=(1, 2)), CUBE),
    ("mean float64", lambda t, x: x.double().mean(0), A),
    ("amax over two dims", lambda t, x: x.amax(dim=(1, 2)), CUBE),
    ("amax of ints", lambda t, x: x.amax(1), INT),
    ("max of ints", lambda t, x: (x.max(), x.max(1)), INT),
    ("min of bools", lambda t, x: x.min(0), MASK),
    ("argmax of ints", lambda t, x: x.argmax(), INT),
    ("any of floats", lambda t, x: (x.any(), x.all(1)), A),
    ("all of ints", lambda t, x: t.all(x, dim=0), INT),
    ("std biased", lambda t, x: x.std(dim=1, unbiased=False), A),
    ("std correction", lambda t, x: x.std(dim=1, correction=0), A),
    ("var of one value", lambda t, x: x[:1, 0].var(), A),
    ("norm", lambda t, x: (x.norm(), t.norm(x), x.norm(dim=1), x.norm(dim=0, keepdim=True)), A),
    ("norm of a vector", lambda t, x: x.norm(), V),
    ("norm 1", lambda t, x: x.norm(p=1), A),
    ("norm 1 dim", lambda t, x: x.norm(1, dim=0), A),
    ("norm inf", lambda t, x: x.norm(float("inf")), A),
    ("norm 3", lambda t, x: x.norm(p=3, dim=1, keepdim=True), A),
    ("norm fro", lambda t, x: x.norm("fro"), A),
    ("norm two dims", lambda t, x: x.norm(dim=(0, 1)), A),
    ("norm float64", lambda t, x: x.double().norm(dim=1), A),
    ("trace", lambda t, x: x.trace(), SQUARE),
    ("trace wide", lambda t, x: t.trace(x), A),
    ("trace of ints", lambda t, x: x.trace(), INT),
    ("cumsum", lambda t, x: x.cumsum(1), A),
    ("cumsum dim keyword", lambda t, x: t.cumsum(x, dim=0), A),
    ("cumsum of ints", lambda t, x: x.cumsum(0), INT),
    ("cumsum of bools", lambda t, x: x.cumsum(-1), MASK),
    ("cumprod", lambda t, x: t.cumprod(x, dim=1), A),
    ("cumprod of ints", lambda t, x: t.cumprod(x, 0), INT),
    ("logsumexp", lambda t, x: t.logsumexp(x, 1), A),
    ("logsumexp keepdim", lambda t, x: x.logsumexp(dim=0, keepdim=True), A),
    ("logsumexp two dims", lambda t, x: t.logsumexp(x, (0, 1)), A),
    ("logsumexp large", lambda t, x: t.logsumexp(x * 1000, 1), A),
    ("logsumexp of ints", lambda t, x: x.logsumexp(0), INT),
    ("logsumexp all -inf", lambda t, x: t.logsumexp(x * float("-inf"), 0), POS),
    ("logsumexp with inf", lambda t, x: t.logsumexp(x * float("inf"), 0), POS),
    ("softmax", lambda t, x: t.softmax(x, dim=1), A),
    ("softmax method", lambda t, x: x.softmax(-1), CUBE),
    ("softmax dim 0", lambda t, x: x.softmax(dim=0), A),
    ("softmax large", lambda t, x: t.softmax(x * 1000, 1), A),
    ("softmax float64", lambda t, x: x.double().softmax(0), A),
    ("log_softmax", lambda t, x: t.log_softmax(x, dim=1), A),
    ("log_softmax method", lambda t, x: x.log_softmax(0), A),
    ("log_softmax large", lambda t, x: t.log_softmax(x * 1000, -1), A),
    ("0-d along dim 0", lambda t, x: (tuple(x[0, 0].max(0)), x[0, 0].argmax(0), x[0, 0].cumsum(0), x[0, 0].softmax(0)), A),
    ("empty sums and means", lambda t, x: (x[:0].sum(0), x[:0].mean(0), x[:0].amax(1)), A),
    ("max of two tensors", lambda t, x, y: (t.max(x, y), t.min(x, y), t.maximum(x, y), t.minimum(x, y)), A, B),
)
def test_reductions(st, real, case):
    check(st, real, *case[1:])


def test_review_focus_named_results_unpack_and_have_fields(st, real):
    x, rx = st.tensor(TIES), real.tensor(TIES)
    for name in ("max", "min", "sort", "topk", "median", "mode"):
        args = (2,) if name == "topk" else ()
        got, want = getattr(x, name)(*args, dim=1), getattr(rx, name)(*args, dim=1)
        same(got.values, want.values)
        if name != "topk":  # which of several equal values topk reports is not defined
            same(got.indices, want.indices)
        values, indices = got
        assert isinstance(got, tuple) and len(got) == 2 and values is got[0] and indices is got[1]
        assert type(got).__name__ == type(want).__name__ == name
    same(st.max(x, dim=1)[1], real.max(rx, dim=1)[1])
    same(x.max(dim=1, keepdim=True).values, rx.max(dim=1, keepdim=True).values)


@cases(
    ("mean of ints", lambda t, x: x.mean(), INT),
    ("std of ints", lambda t, x: x.std(), INT),
    ("norm of ints", lambda t, x: x.norm(), INT),
    ("softmax of ints", lambda t, x: x.softmax(0), INT),
    ("quantile of ints", lambda t, x: x.quantile(0.5), INT),
    ("max of nothing", lambda t, x: x[x > 100].max(), A),
    ("cumsum without dim", lambda t, x: x.cumsum(), A),
    ("logsumexp without dim", lambda t, x: x.logsumexp(), A),
    ("dim out of range", lambda t, x: x.sum(2), A),
    ("max along an empty dim", lambda t, x: x[:0].max(0), A),
    ("argmax along an empty dim", lambda t, x: x[:0].argmax(dim=0), A),
    ("bincount of floats", lambda t, x: t.bincount(x), A),
)
def test_reduction_errors(st, real, case):
    check_raises(st, real, *case[1:])
