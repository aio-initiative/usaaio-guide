"""Autograd's rules against real torch: .grad and its accumulation, the gradient modes, the
errors, in-place changes and views (the Review focus), autograd.grad and second derivatives,
and autograd.functional."""

import copy
import inspect
import pickle
import warnings

import numpy as np
import pytest

from parity import agree, same

RNG = np.random.default_rng(3)
M = RNG.standard_normal((3, 4))
W = RNG.standard_normal((4, 2))
VEC = RNG.standard_normal(4)
BLOCK = RNG.standard_normal((4, 3))


def leaf(t, array, dtype=None):
    return t.tensor(array, dtype=dtype, requires_grad=True)


def raises_like_torch(st, real, run, match):
    """run(torch) fails on both libraries with the same exception type and a message with `match`."""
    with pytest.raises(Exception) as want:
        run(real)
    with pytest.raises(Exception) as got:
        run(st)
    assert type(got.value) is type(want.value), f"{got.value!r} is not a {type(want.value).__name__}"
    assert match in str(want.value), str(want.value)
    assert match in str(got.value), str(got.value)


# .grad, its accumulation and the flags.


def test_gradients_accumulate_over_backward_calls(st, real):
    def run(t):
        x = leaf(t, VEC)
        (x * x).sum().backward()
        first = x.grad.clone()
        (x * 3).sum().backward()
        return first, x.grad

    agree(run, st, real)


def test_zeroing_and_clearing_grad(st, real):
    def run(t):
        x = leaf(t, VEC)
        (x * x).sum().backward()
        x.grad.zero_()
        x.exp().sum().backward()
        zeroed = x.grad.clone()
        x.grad = None
        (x * 2).sum().backward()
        return zeroed, x.grad

    agree(run, st, real)


def test_the_grad_is_not_shared_between_leaves(st):
    a, b = st.ones(3, requires_grad=True), st.ones(3, requires_grad=True)
    (a + b).sum().backward()
    a.grad.zero_()
    assert b.grad.tolist() == [1.0, 1.0, 1.0]


def test_detach(st, real):
    def run(t):
        x = leaf(t, VEC)
        d = x.detach()
        d.add_(1)  # shares memory and the version counter with x
        return d.requires_grad, d.grad_fn is None, d.is_leaf, x.detach(), x._version, d._version

    agree(run, st, real)


def test_requires_grad_flags(st, real):
    def run(t):
        x = t.tensor(VEC)
        flags = [x.requires_grad, x.is_leaf]
        assert x.requires_grad_() is x
        y = x * 2
        flags += [x.requires_grad, y.requires_grad, y.is_leaf, x.grad_fn is None, y.grad_fn is not None]
        flags += [(x > 0).requires_grad, x.argmax().requires_grad, x.long().requires_grad, x.long().grad_fn is None]
        flags += [x.max(0).indices.requires_grad, x.sort().indices.requires_grad, x.detach().requires_grad]
        z = t.tensor(VEC).requires_grad_(False)
        return flags + [z.requires_grad]

    agree(run, st, real)


@pytest.mark.parametrize(
    "run",
    [
        lambda t: t.tensor([1, 2], requires_grad=True),
        lambda t: t.arange(3).requires_grad_(),
        lambda t: t.zeros(2, dtype=t.int64, requires_grad=True),
        lambda t: t.tensor([True]).requires_grad_(),
    ],
    ids=["tensor", "requires_grad_", "zeros", "bool"],
)
def test_integer_and_bool_tensors_cannot_require_grad(st, real, run):
    raises_like_torch(st, real, run, "can require gradients")


def test_retain_grad(st, real):
    def run(t):
        x = leaf(t, VEC)
        y = x * 2
        y.retain_grad()
        z = (y * y).sum()
        z.backward(retain_graph=True)
        first = y.grad.clone()
        z.backward()
        return first, y.grad, y.retains_grad, x.retains_grad

    agree(run, st, real)


def test_grad_of_a_non_leaf_is_none_with_a_warning(st, real):
    for t in (st, real):
        y = leaf(t, VEC) * 2
        y.sum().backward()
        with pytest.warns(UserWarning, match="not a leaf Tensor"):
            assert y.grad is None


def test_backward_of_a_leaf_and_with_a_gradient(st, real):
    def run(t):
        s = t.tensor(2.0, requires_grad=True)
        s.backward()
        x = leaf(t, M)
        (x * 3).backward(t.tensor(M))
        return s.grad, x.grad

    agree(run, st, real)


@pytest.mark.parametrize(
    "run, match",
    [
        (lambda t: (leaf(t, VEC) * 2).backward(), "grad can be implicitly created only for scalar outputs"),
        (lambda t: t.tensor(VEC).sum().backward(), "does not require grad and does not have a grad_fn"),
        (lambda t: (leaf(t, VEC) * 2).backward(t.ones(3)), "Mismatch in shape"),
        (lambda t: setattr(leaf(t, VEC) * 2, "requires_grad", False), "only change requires_grad flags of leaf"),
        (lambda t: setattr(leaf(t, VEC), "grad", t.ones(3)), "attempting to assign a gradient"),
        (lambda t: leaf(t, VEC).numpy(), "Can't call numpy() on Tensor that requires grad"),
        (lambda t: np.asarray(leaf(t, VEC) * 2), "Can't call numpy() on Tensor that requires grad"),
        (lambda t: copy.deepcopy(leaf(t, VEC) * 2), "graph leaves"),
    ],
    ids=["non-scalar", "no grad", "shape", "non-leaf flag", "grad size", "numpy", "asarray", "deepcopy"],
)
def test_errors(st, real, run, match):
    raises_like_torch(st, real, run, match)


def test_a_second_backward_needs_retain_graph(st, real):
    def twice(t, make, retain):
        x = leaf(t, VEC)
        y = make(x)
        y.backward(retain_graph=retain)
        y.backward()
        return x.grad

    second_time = "Trying to backward through the graph a second time"
    raises_like_torch(st, real, lambda t: twice(t, lambda x: (x * x).sum(), False), second_time)
    raises_like_torch(st, real, lambda t: twice(t, lambda x: (x * 2).sum(), False), second_time)
    raises_like_torch(st, real, lambda t: twice(t, lambda x: x.exp().sum(), False), second_time)
    for make in (lambda x: (x * x).sum(), lambda x: x.exp().mean()):
        agree(lambda t: twice(t, make, True), st, real)
    # A graph that saved nothing can be run again.
    agree(lambda t: twice(t, lambda x: (x + 1).sum(), False), st, real)


def test_grad_fn_names_and_next_functions(st, real):
    def run(t):
        x, a = leaf(t, M), leaf(t, W)
        outputs = [
            x * 2, x + 1, x.sum(), x.sum(0), x.mean(), x.exp(), x @ a, x[0], x[1:], x.permute(1, 0),
            x.view(-1), x.unsqueeze(0), x.expand(2, 3, 4), x.relu(), x.sigmoid(), x.tanh(), x.abs(),
            x.clone(), x / 2, -x, x.amax(1), x.gather(1, t.tensor([[0], [1], [2]])),
        ]
        names = [type(y.grad_fn).__name__ for y in outputs]
        y = x * t.tensor(M)
        kinds = [type(fn).__name__ if fn is not None else None for fn, _ in y.grad_fn.next_functions]
        return names, kinds, y.grad_fn.name(), y.grad_fn.next_functions[0][0].variable is x

    agree(run, st, real)


def test_repr_shows_the_history(st, real):
    def run(t):
        x = leaf(t, [1.0, 2.0])
        return repr(x), repr(x * 2), repr(x.detach()), repr(x.sum())

    agree(run, st, real)


def test_pickle_keeps_requires_grad_but_not_the_history(st, real):
    def run(t):
        x = leaf(t, VEC)
        x.grad = t.ones(4, dtype=x.dtype)
        y = pickle.loads(pickle.dumps(x * 2))
        z = pickle.loads(pickle.dumps(x))
        return y.detach(), y.requires_grad, y.is_leaf, z.grad, z.requires_grad

    agree(run, st, real)


def test_deepcopy_of_a_leaf(st, real):
    def run(t):
        x = leaf(t, VEC)
        (x * x).sum().backward()
        c = copy.deepcopy(x)
        c.data[0] = 100.0
        return x.detach(), c.detach(), c.grad, c.requires_grad, c.is_leaf

    agree(run, st, real)


# Gradient modes.


def test_no_grad_and_enable_grad(st, real):
    def run(t):
        x = leaf(t, VEC)
        flags = []
        with t.no_grad():
            flags += [t.is_grad_enabled(), (x * 2).requires_grad]
            with t.enable_grad():
                flags += [t.is_grad_enabled(), (x * 2).requires_grad]
            flags.append(t.is_grad_enabled())
        flags.append(t.is_grad_enabled())

        @t.no_grad()
        def called(v):
            return v * 2

        @t.no_grad
        def bare(v):
            return v * 2

        flags += [called(x).requires_grad, bare(x).requires_grad, t.is_grad_enabled()]
        return flags

    agree(run, st, real)


def test_set_grad_enabled(st, real):
    def run(t):
        x = leaf(t, VEC)
        flags = []
        with t.set_grad_enabled(False):
            flags += [t.is_grad_enabled(), (x * 2).requires_grad]
        flags.append(t.is_grad_enabled())
        t.set_grad_enabled(False)
        flags += [t.is_grad_enabled(), (x * 2).requires_grad]
        t.set_grad_enabled(True)

        @t.set_grad_enabled(False)
        def off(v):
            return v * 2

        flags += [t.is_grad_enabled(), off(x).requires_grad, t.is_grad_enabled()]
        return flags

    agree(run, st, real)


def test_no_grad_restores_the_mode_after_an_error(st):
    with pytest.raises(ValueError):
        with st.no_grad():
            raise ValueError
    assert st.is_grad_enabled()


# In-place changes and views (the Review focus).


@pytest.mark.parametrize(
    "change",
    [
        lambda t, w: w.sub_(1),
        lambda t, w: w.__isub__(0.1 * t.ones(4, dtype=w.dtype)),
        lambda t, w: w.mul_(2),
        lambda t, w: w.zero_(),
        lambda t, w: w.fill_(1),
        lambda t, w: w.copy_(t.ones(4)),
        lambda t, w: w.add_(1, alpha=2),
    ],
    ids=["sub_", "-=", "mul_", "zero_", "fill_", "copy_", "add_"],
)
def test_in_place_on_a_leaf_that_requires_grad_raises(st, real, change):
    raises_like_torch(st, real, lambda t: change(t, leaf(t, VEC)), "a leaf Variable that requires grad")


@pytest.mark.parametrize(
    "change",
    [
        lambda t, w: w.__setitem__(0, 1.0),
        lambda t, w: w[1:].zero_(),
        lambda t, w: w.view(2, 2).mul_(2),
    ],
    ids=["setitem", "slice zero_", "view mul_"],
)
def test_in_place_on_a_view_of_a_leaf_raises(st, real, change):
    raises_like_torch(st, real, lambda t: change(t, leaf(t, VEC)), "a view of a leaf Variable that requires grad")


def test_review_focus_sgd_step_under_no_grad(st, real):
    """w -= lr * w.grad works inside no_grad, and a few steps of linear regression match torch."""
    x = RNG.standard_normal((8, 4))
    y = x @ np.array([1.0, -2.0, 0.5, 3.0]) + 0.7

    def run(t):
        w = t.zeros(4, dtype=t.float64, requires_grad=True)
        b = t.zeros(1, dtype=t.float64, requires_grad=True)
        X, Y = t.tensor(x), t.tensor(y)
        for _ in range(25):
            loss = ((X @ w + b - Y) ** 2).mean()
            loss.backward()
            with t.no_grad():
                w -= 0.1 * w.grad
                b -= 0.1 * b.grad
            w.grad.zero_()
            b.grad = None
        return w.detach(), b.detach(), loss.detach(), w.is_leaf, w.requires_grad, w._version

    agree(run, st, real)


def test_review_focus_bias_gradient_is_summed_over_the_batch(st, real):
    def run(t):
        x = t.tensor(M)
        weight, bias = leaf(t, W), leaf(t, [0.5, -0.5])
        out = x @ weight + bias  # (3, 2) + (2,)
        (out * t.tensor(M[:, :2])).sum().backward()
        return bias.grad, weight.grad, bias.grad.shape

    agree(run, st, real)


@pytest.mark.parametrize(
    "make",
    [
        lambda t, x: (lambda y: (y * y, y))(x * 2),  # mul saves both operands
        lambda t, x: (lambda y: (y, y))(x.exp()),  # exp saves its result
        lambda t, x: (lambda y: (y, y))(x.sigmoid()),
        lambda t, x: (lambda s: (s, s))((x * x + 1).sqrt()),  # sqrt saves its result
        lambda t, x: (lambda y: (t.ones(2, 3, dtype=y.dtype, requires_grad=True) @ y, y))(x * 1),
        lambda t, x: (lambda y: (y[0] * y[1], y))(x * 1),  # a view of y is saved
        lambda t, x: (lambda y: (y.clamp(-1, 1), y))(x * 1),
    ],
    ids=["mul", "exp", "sigmoid", "sqrt", "matmul", "view", "clamp"],
)
def test_review_focus_changing_a_saved_tensor_raises_at_backward(st, real, make):
    def run(t):
        out, saved = make(t, leaf(t, M))
        saved.add_(1)
        out.sum().backward()

    raises_like_torch(st, real, run, "modified by an inplace operation")


def test_changing_a_tensor_that_was_not_saved_is_fine(st, real):
    def run(t):
        x = leaf(t, M)
        y = x * 2
        z = y + 1 + y.sqrt().sum()  # add saves nothing, sqrt saves its result
        z = z + y @ t.ones(4, 2, dtype=y.dtype)  # y is saved only if the other operand needs a gradient
        y.add_(1)
        q = x * 3
        with t.no_grad():
            q.mul_(5)  # values change, the history does not
        (z.sum() + q.sum()).backward()
        return x.grad, y.detach(), y._version

    agree(run, st, real)


@pytest.mark.parametrize(
    "change",
    [
        lambda t, y, x: y.add_(x),
        lambda t, y, x: y.mul_(x),
        lambda t, y, x: y.div_(x * x + 1),
        lambda t, y, x: y.sub_(3),
        lambda t, y, x: y.__imul__(x),
        lambda t, y, x: y.__itruediv__(2.0),
        lambda t, y, x: y.zero_(),
        lambda t, y, x: y.fill_(2.0),
        lambda t, y, x: y.copy_(x.exp()),
        lambda t, y, x: y.copy_(x[0]),
        lambda t, y, x: y.scatter_add_(0, t.tensor([[0, 1, 2, 0]]), x[:1]),
        lambda t, y, x: y.copy_(y.clamp(-0.5, 0.5)),
    ],
    ids=["add_", "mul_", "div_", "sub_", "*=", "/=", "zero_", "fill_", "copy_", "copy_ broadcast", "scatter_add_", "copy_ of self"],
)
def test_in_place_changes_carry_gradients(st, real, change):
    def run(t):
        x = leaf(t, M)
        y = x * 2
        change(t, y, x)
        (y * t.tensor(M)).sum().backward()
        return y.detach(), x.grad, y.is_leaf, y._version

    agree(run, st, real)


def test_in_place_grad_fn_names(st, real):
    def run(t):
        x = leaf(t, M)
        names = []
        for change in (
            lambda y: y.add_(1),
            lambda y: y.mul_(x),
            lambda y: y.zero_(),
            lambda y: y.copy_(x),
            lambda y: y.__setitem__(0, 1.0),
            lambda y: y[1:].mul_(2),
        ):
            y = x * 2
            change(y)
            names.append(type(y.grad_fn).__name__)
        return names

    agree(run, st, real)


@pytest.mark.parametrize(
    "write",
    [
        lambda t, y, v: y.__setitem__(1, v[0]),
        lambda t, y, v: y.__setitem__((slice(None), 2), v[:, 1]),
        lambda t, y, v: y.__setitem__((slice(0, 2), slice(1, 3)), v[:2, :2]),
        lambda t, y, v: y.__setitem__([0, 2], v[:2]),
        lambda t, y, v: y.__setitem__(y > 0, v[0, 0]),
        lambda t, y, v: y.__setitem__((slice(None), 1), v[0, 0]),
        lambda t, y, v: y.__setitem__(..., v[0]),
        lambda t, y, v: y.__setitem__(t.tensor([2, 0]), v[1:]),
    ],
    ids=["row", "column", "block", "rows by list", "mask with a number tensor", "broadcast", "ellipsis", "index tensor"],
)
def test_setitem_carries_gradients_to_the_value_and_the_rest(st, real, write):
    def run(t):
        x, v = leaf(t, M), leaf(t, -np.abs(M))
        y = x * 1
        write(t, y, v)
        (y * t.tensor(M + 2)).sum().backward()
        name = type(y.grad_fn).__name__
        return y.detach(), x.grad, v.grad, name if name != "CopyBackwards" else "CopySlices"

    agree(run, st, real)


def test_setitem_into_a_tensor_without_history(st, real):
    def run(t):
        v = leaf(t, VEC)
        y = t.zeros(3, 4, dtype=v.dtype)
        y[1] = v * 2
        flags = [y.requires_grad, y.is_leaf]
        (y * t.tensor(M)).sum().backward()
        return flags, v.grad

    agree(run, st, real)


def test_setitem_with_ints_and_a_list_reaches_the_value(st, real):
    """x[0, :, [1, 2]] = v: the ints are applied first, as torch does, and v gets its gradient."""
    cube = RNG.standard_normal((2, 3, 4))

    def run(t):
        x, v = leaf(t, cube), leaf(t, M[:, :2].copy())
        y = x * 1
        y[0, :, [1, 2]] = v
        y[1, 2, [0, 3]] = v[0] * 2
        (y * t.tensor(cube + 1)).sum().backward()
        return y.detach(), x.grad, v.grad

    agree(run, st, real)


def test_in_place_through_a_view_reaches_the_base(st, real):
    def run(t):
        x, w = leaf(t, M), leaf(t, M[1:])
        y = x * 1
        v = y[1:]
        v.mul_(w)
        names = [type(y.grad_fn).__name__, type(v.grad_fn).__name__, v._base is y]
        (y * t.tensor(M + 1)).sum().backward()
        return names, y.detach(), x.grad, w.grad

    agree(run, st, real)


@pytest.mark.parametrize(
    "change",
    [
        lambda t, y, w: y.view(-1)[2:5].zero_(),
        lambda t, y, w: y.T[0].add_(w[0, :3]),
        lambda t, y, w: y[:, 1:3][1].copy_(w[0, :2]),
        lambda t, y, w: y.reshape(2, 6)[1].mul_(w[0, 0]),
        lambda t, y, w: y.unsqueeze(0)[0, 2].sub_(w[1]),
        lambda t, y, w: y[1].__setitem__(slice(0, 2), w[0, :2]),
    ],
    ids=["view of a view", "transpose", "slice of a slice", "reshape", "unsqueeze", "setitem on a view"],
)
def test_in_place_through_other_views(st, real, change):
    def run(t):
        x, w = leaf(t, M), leaf(t, M + 1)
        y = x * 1
        change(t, y, w)
        (y * t.tensor(M - 1)).sum().backward()
        return y.detach(), x.grad, w.grad

    agree(run, st, real)


def test_views_taken_before_an_in_place_change_see_the_new_values(st, real):
    def run(t):
        x = leaf(t, M)
        y = x * 1
        row, col = y[0], y[:, 2]
        y.mul_(x)
        y[1, 1] = 0.0
        (row * t.tensor(VEC) + col.sum()).sum().backward()
        return row.detach(), col.detach(), x.grad

    agree(run, st, real)


def test_writing_into_an_expanded_view_fails(st, real):
    def run(t):
        x = t.tensor(VEC)
        x.expand(3, 4).add_(1)

    with pytest.raises(RuntimeError):
        run(real)
    with pytest.raises(RuntimeError):
        run(st)


def test_version_counters_are_shared_with_views(st, real):
    def run(t):
        x = t.zeros(3, 4)
        v, d = x[1], x.detach()
        x.add_(1)
        v.zero_()
        d[0, 0] = 5.0
        return x._version, v._version, d._version, x

    agree(run, st, real)


# autograd.grad and second derivatives.


def test_second_derivative_of_x_cubed(st):
    x = st.tensor(2.0, requires_grad=True)
    (first,) = st.autograd.grad(x**3, x, create_graph=True)
    (second,) = st.autograd.grad(first, x)
    assert first.item() == 12.0 and second.item() == 12.0
    assert first.requires_grad and not second.requires_grad
    assert x.grad is None


def test_autograd_grad(st, real):
    def run(t):
        x, y = leaf(t, VEC), leaf(t, VEC[::-1].copy())
        out = x * y + x.exp()
        gx, gy = t.autograd.grad(out, [x, y], grad_outputs=t.tensor(VEC), retain_graph=True)
        (only,) = t.autograd.grad(out.sum(), y)
        unused = t.autograd.grad(x.sum(), [x, y], allow_unused=True)
        tuple_input = t.autograd.grad((x * 2).sum(), (x,))
        middle = x * 2
        (through,) = t.autograd.grad((middle * middle).sum(), middle)
        return gx, gy, only, unused[0], unused[1] is None, tuple_input, through, x.grad is None

    agree(run, st, real)


def test_autograd_grad_takes_any_iterable_of_tensors(st, real):
    """A generator works wherever a list does, as in real torch, so a reader can hand
    autograd.grad a network's parameters() straight away instead of wrapping it in list().

    Compared with same() rather than agree(): agree() also passes when both libraries raise the
    same error, and this test is only worth anything if neither does.
    """

    def run(t):
        # The same weights and a float32 batch in both libraries, so the gradients can be
        # compared value by value (a float64 batch would stop both at the first matmul).
        model = t.nn.Linear(3, 2)
        with t.no_grad():
            model.weight.copy_(t.tensor(M[:2, :3], dtype=t.float32))
            model.bias.copy_(t.tensor(VEC[:2], dtype=t.float32))
        loss = t.nn.functional.mse_loss(model(t.tensor(BLOCK, dtype=t.float32)), t.zeros(4, 2))
        assert inspect.isgenerator(model.parameters()), "parameters() should be a generator"
        grads = t.autograd.grad(loss, model.parameters(), retain_graph=True)
        as_list = t.autograd.grad(loss, list(model.parameters()), retain_graph=True)
        from_outputs = t.autograd.grad((out for out in [loss]), model.parameters(), retain_graph=True)
        seeded = t.autograd.grad(
            loss,
            model.parameters(),
            grad_outputs=(g for g in [t.ones_like(loss)]),
            retain_graph=True,
        )
        model.zero_grad()
        t.autograd.backward((out for out in [loss]))
        return (*grads, *as_list, *from_outputs, *seeded, model.weight.grad, model.bias.grad)

    same(run(st), run(real))


def test_autograd_grad_errors(st, real):
    raises_like_torch(
        st, real, lambda t: t.autograd.grad(leaf(t, VEC).sum(), [leaf(t, VEC)]), "appears to not have been used"
    )
    raises_like_torch(
        st, real, lambda t: t.autograd.grad(leaf(t, VEC).sum(), t.tensor(VEC)), "does not require grad"
    )


def test_autograd_backward_with_inputs(st, real):
    def run(t):
        x, y = leaf(t, VEC), leaf(t, VEC + 1)
        t.autograd.backward((x * y).sum(), inputs=[x])
        (x * y).sum().backward(inputs=[y])
        middle = x * y
        (middle * middle).sum().backward(inputs=[middle, x])
        return x.grad, y.grad, middle.grad, middle.retains_grad

    agree(run, st, real)


SECOND_ORDER = (
    ("polynomial", lambda t, x: x**4 - 3 * x**3 + 2 * x),
    ("exp and log", lambda t, x: (x * x + 1).log() * x.exp()),
    ("softmax", lambda t, x: x.reshape(2, 2).softmax(1)),
    ("log_softmax", lambda t, x: t.log_softmax(x * 2, 0)),
    ("matmul", lambda t, x: (x.reshape(2, 2) @ x.reshape(2, 2).T) @ x),
    ("sigmoid and tanh", lambda t, x: x.sigmoid() * x.tanh()),
    ("division", lambda t, x: x / (x * x + 2)),
    ("sqrt and pow", lambda t, x: (x * x + 1).sqrt() ** 1.5),
    ("tensor power", lambda t, x: (x.abs() + 1) ** x),
    ("trig", lambda t, x: x.sin() * x.cos() ** 2),
    ("norm", lambda t, x: x.norm() * x),
    ("std and var", lambda t, x: x.std() * x.var() + x.mean()),
    ("logsumexp", lambda t, x: x.logsumexp(0) * x),
    ("indexing and cumsum", lambda t, x: x[[0, 2, 2]].cumsum(0) * x[1:]),
    ("max and where", lambda t, x: t.where(x > 0, x * x, -x) * x.max()),
    ("einsum", lambda t, x: t.einsum("i,j->ij", x, x * x).sum(1)),
    ("cdist", lambda t, x: t.cdist(x.reshape(2, 2), (x * x).reshape(2, 2))),
    ("relu and clamp", lambda t, x: x.relu() * x + x.clamp(-0.5, 0.5) ** 2),
)


@pytest.mark.parametrize("case", SECOND_ORDER, ids=[c[0] for c in SECOND_ORDER])
def test_second_derivatives(st, real, case):
    fn = case[1]
    point = np.array([0.7, -1.3, 0.4, 2.1])

    def run(t):
        x = leaf(t, point)
        out = fn(t, x)
        weight = t.tensor(np.linspace(0.5, 1.5, out.numel()).reshape(tuple(out.shape)))
        (g,) = t.autograd.grad((out * weight).sum(), x, create_graph=True)
        (h,) = t.autograd.grad((g * t.tensor([1.0, -2.0, 0.5, 3.0])).sum(), x)
        return g.detach(), h

    agree(run, st, real)


# Real torch warns that .grad holding a graph makes a reference cycle; that is not under test.
@pytest.mark.filterwarnings("ignore:Using backward\\(\\) with create_graph=True")
def test_backward_with_create_graph_gives_a_differentiable_grad(st, real):
    def run(t):
        x = leaf(t, VEC)
        (x**3).sum().backward(create_graph=True)
        first = x.grad
        flags = [first.requires_grad]
        (g2,) = t.autograd.grad(first.sum(), x)
        return flags, first.detach(), g2

    agree(run, st, real)


def test_third_derivative(st):
    x = st.tensor(2.0, requires_grad=True)
    y = x**4
    for expected in (32.0, 48.0, 48.0):
        (y,) = st.autograd.grad(y, x, create_graph=True)
        assert y.item() == expected


# autograd.functional.


def test_hessian_of_a_quadratic_form(st, real):
    a = RNG.standard_normal((3, 3))
    for t in (st, real):
        h = t.autograd.functional.hessian(lambda v: v @ t.tensor(a) @ v, t.tensor(VEC[:3]))
        np.testing.assert_allclose(h.numpy(), a + a.T, rtol=1e-12)
    agree(lambda t: t.autograd.functional.hessian(lambda v: v @ t.tensor(a) @ v, t.tensor(VEC[:3])), st, real)


@pytest.mark.parametrize(
    "run",
    [
        lambda t: t.autograd.functional.jacobian(lambda v: v.exp() * v.sum(), t.tensor(VEC)),
        lambda t: t.autograd.functional.jacobian(lambda m: m @ t.tensor(W), t.tensor(M)),
        lambda t: t.autograd.functional.jacobian(lambda a, b: (a * b).sum(0), (t.tensor(M), t.tensor(M + 1))),
        lambda t: t.autograd.functional.jacobian(lambda v: (v * 2, v.sum()), t.tensor(VEC)),
        lambda t: t.autograd.functional.jacobian(lambda v: t.tensor(VEC), t.tensor(VEC)),
        lambda t: t.autograd.functional.hessian(lambda v: (v**3).sum() * v[0], t.tensor(VEC)),
        lambda t: t.autograd.functional.hessian(lambda a, b: (a * a * b).sum(), (t.tensor(VEC), t.tensor(VEC + 1))),
    ],
    ids=["vector", "matrix", "two inputs", "two outputs", "constant", "hessian", "hessian of two inputs"],
)
def test_jacobian_and_hessian(st, real, run):
    agree(run, st, real)


def test_jacobian_with_create_graph(st, real):
    def run(t):
        x = leaf(t, VEC)
        j = t.autograd.functional.jacobian(lambda v: v**3, x, create_graph=True)
        (g,) = t.autograd.grad(j.sum(), x)
        return j.detach(), g, j.requires_grad

    agree(run, st, real)


def test_hessian_needs_a_single_value(st, real):
    raises_like_torch(
        st, real, lambda t: t.autograd.functional.hessian(lambda v: v * 2, t.tensor(VEC)), "single element"
    )


def test_warnings_are_not_raised_by_plain_training(st):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        w = st.zeros(4, requires_grad=True)
        loss = (st.tensor(M, dtype=st.float32) @ w).pow(2).mean()
        loss.backward()
        assert w.grad.shape == (4,)


# Fix round 1: inf gradients through masks, fill_ with a tensor, fresh .grad copies,
# generators under no_grad, remainder, and what a write keeps for the backward pass.


@pytest.mark.parametrize(
    "run",
    [
        lambda t, x: (lambda y: (y.__setitem__(0, 0.0), y.sqrt()))(x * 1)[1],
        lambda t, x: (lambda y: (y[1:].zero_(), y.sqrt()))(x * 1)[1],
        lambda t, x: (lambda y: (y.view(-1)[:1].copy_(t.zeros(1, dtype=y.dtype)), y.log()))(x * 1)[1],
        lambda t, x: (lambda y: (y.zero_(), y.sqrt()))(x * 1)[1],
        lambda t, x: (lambda y: (y.fill_(0.0), y.sqrt()))(x * 1)[1],
        lambda t, x: (lambda y: (y.__setitem__(y < 1, 0.0), y.sqrt()))(x * 1)[1],
    ],
    ids=["setitem", "zero_ of a slice", "copy_ through a view", "zero_", "fill_", "mask setitem"],
)
def test_an_overwritten_region_gets_0_not_nan_from_an_inf_gradient(st, real, run):
    """sqrt and log of an overwritten 0 send inf back; the overwritten values get 0, as in torch."""
    agree(lambda t: (lambda x: (run(t, x).sum().backward(), x.grad)[1])(leaf(t, [4.0, 0.25, 9.0])), st, real)


def test_review_probe_setitem_then_sqrt(st):
    x = st.tensor([4.0, 0.25], requires_grad=True)
    y = x * 1
    y[0] = 0
    y.sqrt().sum().backward()
    assert x.grad.tolist() == [0.0, 1.0]


def test_zeroing_a_distance_matrix_diagonal_then_sqrt(st, real):
    points = RNG.standard_normal((4, 2))

    def run(t):
        x = leaf(t, points)
        d = ((x.unsqueeze(1) - x.unsqueeze(0)) ** 2).sum(-1)
        d = d + 1.0
        steps = t.arange(4)
        d[steps, steps] = 0.0
        loss = d.sqrt().sum() + (d + t.eye(4, dtype=d.dtype)).log().sum()
        loss.backward()
        return x.grad

    agree(run, st, real)


def test_fill_with_a_tensor_carries_its_gradient(st, real):
    def run(t):
        x, v = leaf(t, VEC), t.tensor(2.0, dtype=t.float64, requires_grad=True)
        y = x * 1
        y.fill_(v * 3)
        name = type(y.grad_fn).__name__
        (y * t.tensor(VEC)).sum().backward()
        z = x * 1
        z.fill_(0.5)
        return name, type(z.grad_fn).__name__, v.grad, x.grad, y.detach()

    agree(run, st, real)
    raises_like_torch(st, real, lambda t: (leaf(t, VEC) * 1).fill_(t.ones(2)), "0-dimension value tensor")


def test_grad_is_a_fresh_writable_tensor(st, real):
    def run(t):
        x, y = leaf(t, VEC), leaf(t, VEC + 1)
        start = t.ones(4, dtype=x.dtype)
        (x + y).backward(start, create_graph=True)
        flags = [x.grad is y.grad, x.grad is start]
        with t.no_grad():
            x.grad.mul_(10)
        a, b = leaf(t, VEC), leaf(t, M)
        a.sum().backward()
        a.grad.zero_()  # a sum's gradient is a broadcast; .grad must still be writable
        b.mean().backward(create_graph=True)
        with t.no_grad():
            b.grad.add_(1)
        return flags, x.grad.detach(), y.grad.detach(), start, a.grad, b.grad.detach()

    agree(run, st, real)


def test_autograd_grad_of_a_sum_is_a_broadcast_view_in_both(st, real):
    raises_like_torch(
        st,
        real,
        lambda t: (lambda x: t.autograd.grad(x.sum(), x)[0].add_(1))(leaf(t, VEC)),
        "",
    )


def test_no_grad_on_a_generator_covers_every_step(st, real):
    def run(t):
        x = leaf(t, VEC)

        @t.no_grad()
        def steps():
            received = yield t.is_grad_enabled(), (x * 2).requires_grad
            yield received, t.is_grad_enabled(), (x * 2).requires_grad
            return "done"

        @t.enable_grad()
        def inside():
            yield t.is_grad_enabled()

        g = steps()
        first = next(g)
        between = t.is_grad_enabled()
        second = g.send("sent")
        with pytest.raises(StopIteration) as stop:
            next(g)
        with t.no_grad():
            turned_on = list(inside())
        closed = steps()
        next(closed)
        closed.close()
        return first, between, second, stop.value.value, turned_on, t.is_grad_enabled()

    agree(run, st, real)


def test_remainder_of_a_number_by_a_tensor_has_no_derivative(st, real):
    raises_like_torch(
        st, real, lambda t: (5.0 % leaf(t, VEC + 3)).sum().backward(), "derivative for aten::remainder is not implemented"
    )


def arrays_kept_by(formulas):
    """Every NumPy array the formulas keep alive, found through their closures (tensors give their array)."""
    found, seen, todo = [], set(), list(formulas)
    while todo:
        item = todo.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if hasattr(item, "_data") and isinstance(item._data, np.ndarray):
            item = item._data
        if isinstance(item, np.ndarray):
            found.append(item)
        elif isinstance(item, (tuple, list)):
            todo.extend(item)
        elif callable(item):
            todo.extend(cell.cell_contents for cell in getattr(item, "__closure__", None) or ())
    return found


def owner_of(array):
    while array.base is not None:  # a view keeps the array it views alive, at its full size
        array = array.base
    return array


def memory_kept_by(formulas):
    """The bytes of every array the formulas keep alive: whole owners, as a view keeps its .base."""
    return sum({id(owner_of(a)): owner_of(a).nbytes for a in arrays_kept_by(formulas)}.values())


def test_memory_kept_by_counts_the_array_a_view_views():
    """The measuring helper sees through .base (a 0-d view of a big array is a big record)."""
    big = np.arange(1000, dtype=np.intp)
    element, piece, copied = big[3:4].reshape(()), big[3:5], np.array(big[3:5])
    assert memory_kept_by([lambda: element]) == 8000
    assert memory_kept_by([lambda: piece]) == 8000
    assert memory_kept_by([lambda: copied]) == 16
    assert memory_kept_by([(lambda: element), (lambda: piece)]) == 8000  # one owner is counted once


def small_write_cases():
    """(the shape of the tensor, a function that makes one write at step i)."""
    return {
        "index": ((20_000,), lambda y, i, v: y.__setitem__(i, 1.0)),
        "slice": ((20_000,), lambda y, i, v: y.__setitem__(slice(i, i + 2), 1.0)),
        "tuple": ((100, 200), lambda y, i, v: y.__setitem__((i % 100, i), 1.0)),
        "row piece": ((100, 200), lambda y, i, v: y.__setitem__((i % 100, slice(5, 7)), 1.0)),
        "list": ((20_000,), lambda y, i, v: y.__setitem__([i, i + 1], 1.0)),
        "tensor value": ((20_000,), lambda y, i, v: y.__setitem__(i, v)),
        "mixed int and list": ((20, 100, 10), lambda y, i, v: y.__setitem__((i % 20, i % 100, [1, 2]), 1.0)),
        "through a view": ((20_000,), lambda y, i, v: y[100:].__setitem__(i, 1.0)),
        "through a transpose": ((100, 200), lambda y, i, v: y.T.__setitem__((i, i % 100), 1.0)),
        "fill_ of a slice": ((20_000,), lambda y, i, v: y[i : i + 2].fill_(1.0)),
    }


@pytest.mark.parametrize("case", list(small_write_cases()))
def test_a_loop_of_small_writes_keeps_small_records(st, case):
    """Each write keeps the positions it wrote, not a mask or an index array the size of the tensor.

    Counted through .base: a 0-d view of an arange keeps the whole arange alive."""
    shape, write = small_write_cases()[case]
    x = st.zeros(*shape, requires_grad=True)
    y = x * 1
    value = st.tensor(1.0, requires_grad=True)
    nodes = []
    for i in range(20):
        write(y, i, value)
        nodes.append(y.grad_fn)
    size = int(np.prod(shape))
    assert memory_kept_by([f for node in nodes for f in node._formulas]) < 20 * 64
    assert size * 8 > 20 * 64 * 10  # a kept root-sized array would be well above the bound
    y.sum().backward()


def test_two_hundred_small_writes_retain_little_memory(st):
    """Measured by tracemalloc, which sees every array a record keeps alive."""
    import tracemalloc

    x = st.zeros(20_000, requires_grad=True)
    y = x * 1
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    for i in range(200):
        y[i] = 1.0
    after = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    assert after - before < 1_000_000  # the leak was 160 KB a write, 32 MB here
    y.sum().backward()
    assert x.grad[:200].tolist() == [0.0] * 200 and x.grad[200:].sum().item() == 19800.0


@pytest.mark.parametrize("interpolation", ["linear", "lower", "higher", "nearest", "midpoint"])
@pytest.mark.parametrize("q", [0.3, [0.25, 0.75]])
def test_quantile_keeps_no_view_of_a_larger_array(st, interpolation, q):
    """The positions its gradient needs are copies, not views of the whole sorted-order array."""
    x = st.randn(5000, dtype=st.float64, requires_grad=True)
    out = x.quantile(st.tensor(q, dtype=st.float64), interpolation=interpolation)
    for array in arrays_kept_by(out.grad_fn._formulas):
        if not np.may_share_memory(array, x._data):
            assert owner_of(array).nbytes == array.nbytes
    out.sum().backward()
