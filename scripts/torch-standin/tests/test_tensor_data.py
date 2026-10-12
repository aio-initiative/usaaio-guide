"""Tensor.data against real torch: a tensor over the same memory that autograd does not see."""

import numpy as np
import pytest

from parity import agree

RNG = np.random.default_rng(5)
M = RNG.standard_normal((3, 4))
VEC = RNG.standard_normal(4)


def leaf(t, array):
    return t.tensor(array, requires_grad=True)


def test_data_is_a_detached_tensor_over_the_same_memory(st, real):
    def run(t):
        x = leaf(t, VEC)
        y = x * 2
        d = x.data
        flags = [type(d).__name__, d.requires_grad, d.grad_fn is None, d.is_leaf, d is x.data]
        flags += [y.data.requires_grad, y.data.grad_fn is None, x[1:].data._base is None]
        d[0] = 10.0  # x is a leaf that requires grad, yet this is not an in-place error
        return flags, x.detach(), d, x.requires_grad, x.is_leaf

    agree(run, st, real)


@pytest.mark.parametrize(
    "change",
    [
        lambda t, p: p.data.zero_(),
        lambda t, p: p.data.copy_(t.tensor(VEC[::-1].copy())),
        lambda t, p: p.data.fill_(0.01),
        lambda t, p: p.data.add_(-0.1 * p.data),
        lambda t, p: p.data.mul_(3),
        lambda t, p: p.data.__isub__(1.5),
        lambda t, p: p.data.__setitem__(slice(1, 3), 7.0),
        lambda t, p: p.data.clamp(-0.5, 0.5),
    ],
    ids=["zero_", "copy_", "fill_", "add_", "mul_", "-=", "setitem", "out of place"],
)
def test_changes_through_data_reach_the_tensor_unrecorded(st, real, change):
    def run(t):
        p = leaf(t, VEC)
        change(t, p)
        return p.detach(), p.requires_grad, p.is_leaf, p.grad_fn is None, p._version

    agree(run, st, real)


def test_data_clone_and_numpy(st, real):
    def run(t):
        p = leaf(t, VEC)
        copy = p.data.clone()
        copy.add_(1)
        array = p.data.numpy()
        array[0] = 5.0  # shares memory with p
        return copy, p.detach(), copy.requires_grad, type(array).__name__

    agree(run, st, real)


def test_assigning_data(st, real):
    def run(t):
        w = leaf(t, VEC)
        (w * w).sum().backward()
        w.data = t.tensor(M)  # another shape and values; .grad and requires_grad stay
        flags = [w.requires_grad, w.is_leaf, tuple(w.shape), w.grad is not None]
        (w * 2).sum().backward()  # the old .grad has the old shape, as in torch: refused
        return flags

    with pytest.raises(RuntimeError):
        run(real)
    with pytest.raises(RuntimeError):
        run(st)

    def fresh(t):
        w = leaf(t, VEC)
        w.data = t.tensor(M)
        (w * 2).sum().backward()
        n = t.zeros(2)
        n.data = t.tensor([1, 2])  # a tensor that does not require grad may change dtype
        return w.detach(), w.grad, w.requires_grad, n

    agree(fresh, st, real)


@pytest.mark.parametrize(
    "value, kind",
    [
        (lambda t: np.zeros(4), TypeError),
        (lambda t: [1.0, 2.0], TypeError),
        (lambda t: t.zeros(4, dtype=t.int64), RuntimeError),
    ],
    ids=["ndarray", "list", "int into a tensor that requires grad"],
)
def test_assigning_data_errors(st, real, value, kind):
    for t in (real, st):
        with pytest.raises(kind) as err:
            leaf(t, VEC).data = value(t)
        if kind is TypeError:
            assert "Variable data has to be a tensor" in str(err.value)


def test_changing_data_does_not_trip_the_saved_tensor_check(st, real):
    def run(t):
        x = leaf(t, VEC)
        y = x * 1
        z = (x * x).sum() + y.exp().sum()  # mul saves x, exp saves its result
        x.data.mul_(2)
        y.data.add_(1)
        z.backward()  # reads the changed values, as torch does
        return x.grad, x._version, y._version

    agree(run, st, real)


def test_manual_sgd_through_data_matches_a_no_grad_step(st, real):
    x = RNG.standard_normal((16, 4))
    target = x @ np.array([0.5, -1.0, 2.0, 0.25]) - 0.3

    def train(t, through_data):
        w = t.zeros(4, dtype=t.float64, requires_grad=True)
        b = t.zeros(1, dtype=t.float64, requires_grad=True)
        X, Y = t.tensor(x), t.tensor(target)
        for _ in range(30):
            loss = ((X @ w + b - Y) ** 2).mean()
            loss.backward()
            for p in (w, b):
                if through_data:
                    p.data.add_(-0.05 * p.grad.data)
                    p.grad.data.zero_()
                else:
                    with t.no_grad():
                        p -= 0.05 * p.grad
                    p.grad.zero_()
        return w.detach(), b.detach(), loss.detach()

    agree(lambda t: train(t, True), st, real)
    agree(lambda t: train(t, False), st, real)
    via_data, via_no_grad = train(st, True), train(st, False)
    for a, b in zip(via_data, via_no_grad):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-12)
