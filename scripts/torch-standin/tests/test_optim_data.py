"""Optimisers, learning rate schedulers and data loading, each compared with real torch.

Optimisers are tried on plain leaf tensors that require grad, which is all one needs. The
random order of a shuffled loader is not torch's (random numbers differ), so those tests
check that the stand-in repeats itself and covers the data, not that it matches torch.
"""

import io
import warnings
from collections import OrderedDict, namedtuple

import numpy as np
import pytest

from parity import agree, same

# ---------------------------------------------------------------- optimisers: the problems

CURVATURE = np.array([[3.0, 0.5, 0.0], [0.5, 1.0, 0.2], [0.0, 0.2, 0.1]], dtype=np.float32)


def quadratic_params(t):
    return [
        t.tensor([1.0, -2.0, 0.5], requires_grad=True),
        t.tensor([[0.5, -1.0], [2.0, 0.25]], requires_grad=True),
    ]


def quadratic_loss(t, params):
    w, m = params
    return 0.5 * (w @ (t.tensor(CURVATURE) @ w)) + (m**2).sum() - m[0, 1]


def rosenbrock_params(t):
    return [t.tensor(-1.2, requires_grad=True), t.tensor(1.0, requires_grad=True)]


def rosenbrock_loss(t, params):
    x, y = params
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


PROBLEMS = {
    "quadratic": (quadratic_params, quadratic_loss, {"SGD": 1.0, "RMSprop": 1.0, "Adagrad": 1.0, "Adam": 1.0, "AdamW": 1.0}),
    # The gradient at the start is about 200, so SGD needs a small step.
    "rosenbrock": (
        rosenbrock_params,
        rosenbrock_loss,
        {"SGD": 0.01, "RMSprop": 0.2, "Adagrad": 0.2, "Adam": 0.2, "AdamW": 0.2},
    ),
}

SETTINGS = [
    ("SGD", {"lr": 0.1}),
    ("SGD", {"lr": 0.1, "momentum": 0.9}),
    ("SGD", {"lr": 0.05, "momentum": 0.9, "dampening": 0.5}),
    ("SGD", {"lr": 0.05, "momentum": 0.9, "nesterov": True}),
    ("SGD", {"lr": 0.05, "momentum": 0.8, "weight_decay": 0.1}),
    ("SGD", {"lr": 0.05, "momentum": 0.8, "weight_decay": 0.1, "nesterov": True}),
    ("SGD", {"lr": 0.05, "weight_decay": 0.3}),
    ("SGD", {"lr": 0.02, "momentum": 0.9, "maximize": True}),
    ("SGD", {}),
    ("RMSprop", {}),
    ("RMSprop", {"lr": 0.05, "alpha": 0.9}),
    ("RMSprop", {"lr": 0.01, "momentum": 0.9}),
    ("RMSprop", {"lr": 0.01, "centered": True}),
    ("RMSprop", {"lr": 0.01, "centered": True, "momentum": 0.5, "alpha": 0.3}),
    ("RMSprop", {"lr": 0.01, "centered": True, "eps": 0.1}),
    ("RMSprop", {"lr": 0.02, "weight_decay": 0.1, "eps": 1e-3}),
    ("RMSprop", {"lr": 0.02, "maximize": True}),
    ("Adagrad", {}),
    ("Adagrad", {"lr": 0.1}),
    ("Adagrad", {"lr": 0.1, "lr_decay": 0.1}),
    ("Adagrad", {"lr": 0.1, "weight_decay": 0.1}),
    ("Adagrad", {"lr": 0.1, "initial_accumulator_value": 0.5, "eps": 1e-3}),
    ("Adagrad", {"lr": 0.1, "maximize": True}),
    ("Adam", {}),
    ("Adam", {"lr": 0.1}),
    ("Adam", {"lr": 0.1, "betas": (0.8, 0.9), "eps": 1e-3}),
    ("Adam", {"lr": 0.05, "weight_decay": 0.1}),
    ("Adam", {"lr": 0.05, "amsgrad": True}),
    ("Adam", {"lr": 0.05, "maximize": True}),
    ("AdamW", {}),
    ("AdamW", {"lr": 0.1, "weight_decay": 0.5}),
    ("AdamW", {"lr": 0.05, "betas": (0.5, 0.7), "amsgrad": True}),
    ("AdamW", {"lr": 0.05, "weight_decay": 0.0}),
]
SETTING_IDS = [f"{name}-{'-'.join(f'{k}={v}' for k, v in kwargs.items()) or 'defaults'}" for name, kwargs in SETTINGS]


def scaled(name, kwargs, problem):
    """The settings with the learning rate scaled to what the problem can take."""
    kwargs = dict(kwargs)
    if "lr" in kwargs:
        kwargs["lr"] *= PROBLEMS[problem][2][name]
    return kwargs


def descend(t, problem, name, kwargs, steps=10):
    """The parameters after each of `steps` optimiser steps on the problem."""
    make, loss, _ = PROBLEMS[problem]
    params = make(t)
    optimiser = getattr(t.optim, name)(params, **kwargs)
    path = []
    for _ in range(steps):
        optimiser.zero_grad()
        loss(t, params).backward()
        optimiser.step()
        path.append([p.detach().numpy().copy() for p in params])
    return path


def same_path(got, want):
    assert len(got) == len(want)
    for got_step, want_step in zip(got, want):
        for g, w in zip(got_step, want_step):
            assert g.dtype == w.dtype and g.shape == w.shape
            np.testing.assert_allclose(g, w, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("problem", sorted(PROBLEMS))
@pytest.mark.parametrize("name, kwargs", SETTINGS, ids=SETTING_IDS)
def test_ten_steps_match_torch(st, real, problem, name, kwargs):
    kwargs = scaled(name, kwargs, problem)
    same_path(descend(st, problem, name, kwargs), descend(real, problem, name, kwargs))


@pytest.mark.parametrize(
    "name, kwargs",
    [
        ("SGD", {"momentum": 0.9}),
        ("RMSprop", {"centered": True, "momentum": 0.5}),
        ("Adagrad", {}),
        ("Adam", {}),
        ("AdamW", {"amsgrad": True}),
    ],
)
def test_float64_parameters_stay_float64(st, real, name, kwargs):
    def run(t):
        w = t.tensor(np.array([1.0, -2.0, 0.5]), dtype=t.float64, requires_grad=True)
        optimiser = getattr(t.optim, name)([w], lr=0.1, **kwargs)
        for _ in range(5):
            optimiser.zero_grad()
            (w**3).sum().backward()
            optimiser.step()
        return w.detach()

    agree(run, st, real)


def test_parameter_groups_keep_their_own_options(st, real):
    def run(t):
        w, m = quadratic_params(t)
        groups = [{"params": [w], "lr": 0.1}, {"params": [m], "lr": 0.01, "momentum": 0.5}]
        optimiser = t.optim.SGD(groups, lr=0.05, momentum=0.9)
        for _ in range(6):
            optimiser.zero_grad()
            quadratic_loss(t, [w, m]).backward()
            optimiser.step()
        options = [(g["lr"], g["momentum"]) for g in optimiser.param_groups]
        return w.detach(), m.detach(), options

    agree(run, st, real)


def test_a_group_dict_may_come_from_a_generator(st, real):
    def run(t):
        w, m = quadratic_params(t)
        optimiser = t.optim.Adam(({"params": [p]} for p in (w, m)), lr=0.1)
        for _ in range(3):
            optimiser.zero_grad()
            quadratic_loss(t, [w, m]).backward()
            optimiser.step()
        return w.detach(), m.detach(), len(optimiser.param_groups)

    agree(run, st, real)


def test_named_parameters_are_accepted(st, real):
    def run(t):
        w, m = quadratic_params(t)
        optimiser = t.optim.SGD([("w", w), ("m", m)], lr=0.1)
        optimiser.zero_grad()
        quadratic_loss(t, [w, m]).backward()
        optimiser.step()
        return w.detach(), m.detach(), optimiser.param_groups[0]["param_names"]

    agree(run, st, real)


def test_a_parameter_without_a_gradient_is_skipped(st, real):
    def run(t):
        a = t.tensor([1.0, 2.0], requires_grad=True)
        b = t.tensor([3.0, 4.0], requires_grad=True)
        optimiser = t.optim.Adam([a, b], lr=0.1)
        (a**2).sum().backward()
        optimiser.step()
        return a.detach(), b.detach(), len(optimiser.state), b in optimiser.state

    agree(run, st, real)


def test_state_is_made_on_the_first_step_only_when_needed(st, real):
    def run(t):
        w = t.tensor([1.0, 2.0], requires_grad=True)
        plain = t.optim.SGD([w], lr=0.1)
        momentum = t.optim.SGD([w], lr=0.1, momentum=0.9)
        w.grad = t.tensor([1.0, 1.0])
        before = (len(plain.state), len(momentum.state))
        plain.step()
        momentum.step()
        return before, len(plain.state), sorted(momentum.state[w]), momentum.state[w]["momentum_buffer"]

    agree(run, st, real)


def test_adam_state_has_torchs_names_and_shapes(st, real):
    def run(t):
        w = t.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        optimiser = t.optim.Adam([w], amsgrad=True)
        for _ in range(3):
            optimiser.zero_grad()
            (w**2).sum().backward()
            optimiser.step()
        state = optimiser.state[w]
        return list(state), state["step"], state["exp_avg"], state["exp_avg_sq"], state["max_exp_avg_sq"]

    agree(run, st, real)


def test_zero_grad_sets_to_none_by_default(st, real):
    def run(t):
        w = t.tensor([1.0, 2.0], requires_grad=True)
        optimiser = t.optim.SGD([w], lr=0.1)
        (w * 3).sum().backward()
        had = w.grad is not None
        optimiser.zero_grad()
        return had, w.grad is None

    agree(run, st, real)


def test_zero_grad_can_zero_in_place(st, real):
    def run(t):
        w = t.tensor([1.0, 2.0], requires_grad=True)
        optimiser = t.optim.SGD([w], lr=0.1)
        (w * 3).sum().backward()
        held = w.grad
        optimiser.zero_grad(set_to_none=False)
        return w.grad, w.grad is held

    agree(run, st, real)


def test_step_leaves_the_gradient_and_the_leaf_alone(st, real):
    def run(t):
        w = t.tensor([1.0, 2.0], requires_grad=True)
        optimiser = t.optim.SGD([w], lr=0.1, momentum=0.9, weight_decay=0.1, maximize=True)
        (w**2).sum().backward()
        grad_before = w.grad.clone()
        optimiser.step()
        return w.grad, grad_before, w.requires_grad, w.is_leaf, w.grad_fn is None

    agree(run, st, real)


def test_a_manual_change_to_lr_takes_effect(st, real):
    def run(t):
        w = t.tensor([1.0, 2.0], requires_grad=True)
        optimiser = t.optim.Adam([w], lr=0.1)
        out = []
        for lr in (0.1, 0.01, 0.5):
            optimiser.param_groups[0]["lr"] = lr
            optimiser.zero_grad()
            (w**2).sum().backward()
            optimiser.step()
            out.append(w.detach().clone())
        return out

    agree(run, st, real)


def test_closure_is_called_and_its_loss_returned(st, real):
    def run(t):
        w, m = quadratic_params(t)
        optimiser = t.optim.SGD([w, m], lr=0.1, momentum=0.9)
        calls = []

        def closure():
            calls.append(1)
            optimiser.zero_grad()
            loss = quadratic_loss(t, [w, m])
            loss.backward()
            return loss

        losses = [optimiser.step(closure).detach() for _ in range(4)]
        return losses, len(calls), w.detach(), m.detach()

    agree(run, st, real)


def test_closure_runs_with_gradients_on_even_inside_no_grad(st, real):
    def run(t):
        w = t.tensor([1.0, 2.0], requires_grad=True)
        optimiser = t.optim.SGD([w], lr=0.1)

        def closure():
            loss = (w**2).sum()
            loss.backward()
            return loss

        with t.no_grad():
            loss = optimiser.step(closure)
        return loss.detach(), loss.requires_grad, w.detach()

    agree(run, st, real)


def test_step_without_a_closure_returns_none(st, real):
    def run(t):
        w = t.tensor([1.0], requires_grad=True)
        w.grad = t.tensor([1.0])
        return t.optim.SGD([w], lr=0.1).step(), t.optim.Adam([w]).step()

    agree(run, st, real)


def test_a_step_counts_as_a_change_to_the_parameter(st, real):
    # The backward pass saved w for y, and the step changed it in place, in torch and here.
    def run(t):
        w = t.tensor([1.0, 2.0], requires_grad=True)
        y = (w * w).sum()
        w.grad = t.tensor([1.0, 1.0])
        t.optim.SGD([w], lr=0.1).step()
        y.backward()

    agree(run, st, real)


@pytest.mark.parametrize(
    "name, kwargs",
    [
        ("SGD", {"foreach": True}),
        ("SGD", {"foreach": False, "fused": False, "differentiable": False}),
        ("Adam", {"foreach": False, "capturable": False, "fused": None}),
        ("AdamW", {"foreach": True}),
        ("RMSprop", {"foreach": False, "capturable": False, "differentiable": False}),
        ("Adagrad", {"foreach": False, "fused": None, "differentiable": False}),
    ],
)
def test_implementation_choices_are_accepted_and_change_nothing(st, real, name, kwargs):
    same_path(descend(st, "quadratic", name, {"lr": 0.05, **kwargs}), descend(real, "quadratic", name, {"lr": 0.05, **kwargs}))


def test_add_param_group(st, real):
    def run(t):
        w, m = quadratic_params(t)
        optimiser = t.optim.SGD([w], lr=0.1, momentum=0.9)
        for step in range(6):
            if step == 3:
                optimiser.add_param_group({"params": [m], "lr": 0.05})
            optimiser.zero_grad()
            quadratic_loss(t, [w, m]).backward()
            optimiser.step()
        return w.detach(), m.detach(), [g["lr"] for g in optimiser.param_groups], list(optimiser.param_groups[1])

    agree(run, st, real)


def test_add_param_group_refuses_a_parameter_twice(st, real):
    def run(t):
        w = t.tensor([1.0], requires_grad=True)
        optimiser = t.optim.SGD([w], lr=0.1)
        optimiser.add_param_group({"params": [w]})

    agree(run, st, real)


@pytest.mark.parametrize(
    "build",
    [
        lambda t: t.optim.SGD([], lr=0.1),
        lambda t: t.optim.SGD(t.tensor([1.0], requires_grad=True), lr=0.1),
        lambda t: t.optim.SGD([t.tensor([1.0], requires_grad=True) * 2], lr=0.1),
        lambda t: t.optim.SGD([1.0, 2.0], lr=0.1),
        lambda t: t.optim.SGD([{"params": {t.tensor([1.0], requires_grad=True)}}], lr=0.1),
        lambda t: t.optim.SGD([t.tensor([1.0], requires_grad=True)], lr=-0.1),
        lambda t: t.optim.SGD([t.tensor([1.0], requires_grad=True)], lr=0.1, momentum=-1.0),
        lambda t: t.optim.SGD([t.tensor([1.0], requires_grad=True)], lr=0.1, weight_decay=-1.0),
        lambda t: t.optim.SGD([t.tensor([1.0], requires_grad=True)], lr=0.1, nesterov=True),
        lambda t: t.optim.SGD([t.tensor([1.0], requires_grad=True)], lr=0.1, momentum=0.9, dampening=0.1, nesterov=True),
        lambda t: t.optim.Adam([t.tensor([1.0], requires_grad=True)], lr=-1.0),
        lambda t: t.optim.Adam([t.tensor([1.0], requires_grad=True)], eps=-1.0),
        lambda t: t.optim.Adam([t.tensor([1.0], requires_grad=True)], betas=(1.0, 0.9)),
        lambda t: t.optim.Adam([t.tensor([1.0], requires_grad=True)], betas=(0.9, -0.1)),
        lambda t: t.optim.Adam([t.tensor([1.0], requires_grad=True)], weight_decay=-0.1),
        lambda t: t.optim.AdamW([t.tensor([1.0], requires_grad=True)], lr=-1.0),
        lambda t: t.optim.AdamW([{"lr": 0.1}]),
        lambda t: t.optim.RMSprop([t.tensor([1.0], requires_grad=True)], lr=-1.0),
        lambda t: t.optim.RMSprop([t.tensor([1.0], requires_grad=True)], eps=-1.0),
        lambda t: t.optim.RMSprop([t.tensor([1.0], requires_grad=True)], momentum=-1.0),
        lambda t: t.optim.RMSprop([t.tensor([1.0], requires_grad=True)], weight_decay=-1.0),
        lambda t: t.optim.RMSprop([t.tensor([1.0], requires_grad=True)], alpha=-1.0),
        lambda t: t.optim.Adagrad([t.tensor([1.0], requires_grad=True)], lr=-1.0),
        lambda t: t.optim.Adagrad([t.tensor([1.0], requires_grad=True)], lr_decay=-1.0),
        lambda t: t.optim.Adagrad([t.tensor([1.0], requires_grad=True)], weight_decay=-1.0),
        lambda t: t.optim.Adagrad([t.tensor([1.0], requires_grad=True)], initial_accumulator_value=-1.0),
        lambda t: t.optim.Adagrad([t.tensor([1.0], requires_grad=True)], eps=-1.0),
        lambda t: t.optim.Adagrad([]),
    ],
    ids=[
        "empty",
        "bare tensor",
        "non-leaf",
        "numbers",
        "group params as a set",
        "negative lr",
        "negative momentum",
        "negative weight decay",
        "nesterov without momentum",
        "nesterov with dampening",
        "adam negative lr",
        "adam negative eps",
        "adam beta 1",
        "adam negative beta",
        "adam negative weight decay",
        "adamw negative lr",
        "group without params",
        "rmsprop negative lr",
        "rmsprop negative eps",
        "rmsprop negative momentum",
        "rmsprop negative weight decay",
        "rmsprop negative alpha",
        "adagrad negative lr",
        "adagrad negative lr_decay",
        "adagrad negative weight decay",
        "adagrad negative initial accumulator",
        "adagrad negative eps",
        "adagrad empty",
    ],
)
def test_bad_arguments_fail_as_in_torch(st, real, build):
    try:
        build(real)
    except Exception as want:
        with pytest.raises(type(want)) as got:
            build(st)
        assert type(got.value) is type(want)
    else:
        pytest.fail("real torch accepted these arguments")


def test_the_main_error_messages_are_torchs(st, real):
    with pytest.raises(ValueError, match="optimizer got an empty parameter list"):
        st.optim.SGD([], lr=0.1)
    with pytest.raises(ValueError, match="Invalid learning rate: -0.1"):
        st.optim.SGD([st.tensor([1.0], requires_grad=True)], lr=-0.1)
    with pytest.raises(ValueError, match="can't optimize a non-leaf Tensor"):
        st.optim.SGD([st.tensor([1.0], requires_grad=True) * 2], lr=0.1)
    with pytest.raises(TypeError, match="should be an iterable of Tensors or dicts, but got torch.FloatTensor"):
        st.optim.SGD(st.tensor([1.0], requires_grad=True), lr=0.1)
    with pytest.raises(ValueError, match="Nesterov momentum requires a momentum and zero dampening"):
        st.optim.SGD([st.tensor([1.0], requires_grad=True)], lr=0.1, nesterov=True)
    with pytest.raises(ValueError, match="some parameters appear in more than one parameter group"):
        w = st.tensor([1.0], requires_grad=True)
        st.optim.SGD([w], lr=0.1).add_param_group({"params": [w]})


@pytest.mark.parametrize(
    "name, kwargs",
    [
        ("SGD", {"lr": 0.1}),
        ("SGD", {"lr": 0.1, "momentum": 0.9, "nesterov": True, "weight_decay": 0.01}),
        ("Adam", {}),
        ("Adam", {"lr": 0.01, "betas": (0.8, 0.9), "amsgrad": True}),
        ("AdamW", {}),
        ("AdamW", {"weight_decay": 0.1}),
        ("RMSprop", {}),
        ("RMSprop", {"lr": 0.05, "alpha": 0.9, "momentum": 0.5, "centered": True}),
        ("Adagrad", {}),
        ("Adagrad", {"lr": 0.1, "lr_decay": 0.01, "initial_accumulator_value": 0.1}),
    ],
)
def test_repr_and_param_groups_match_torch(st, real, name, kwargs):
    def build(t):
        w, m = quadratic_params(t)
        return getattr(t.optim, name)([{"params": [w]}, {"params": [m], "lr": 0.5}], **kwargs)

    got, want = build(st), build(real)
    assert repr(got) == repr(want)
    assert str(got) == str(want)
    assert got.defaults == want.defaults
    for g, w in zip(got.param_groups, want.param_groups):
        assert list(g) == list(w)
        assert {k: v for k, v in g.items() if k != "params"} == {k: v for k, v in w.items() if k != "params"}


DIVERGING = [
    ("SGD", {"lr": 1e4, "momentum": 0.9}),
    ("SGD", {"lr": 1e4, "weight_decay": 0.1, "nesterov": True, "momentum": 0.9}),
    ("RMSprop", {"lr": 1e30, "eps": 0.0, "centered": True, "momentum": 0.9}),
    ("RMSprop", {"lr": 1e4, "weight_decay": 1e30}),
    ("Adagrad", {"lr": 1e30, "eps": 0.0}),
    ("Adagrad", {"lr": 1e4, "lr_decay": 1e30, "weight_decay": 1e30}),
    ("Adam", {"lr": 1e30, "eps": 0.0}),
    ("AdamW", {"lr": 1e30, "eps": 0.0, "amsgrad": True, "weight_decay": 1e30}),
]


@pytest.mark.parametrize("name, kwargs", DIVERGING, ids=[f"{n}-{'-'.join(map(str, k))}" for n, k in DIVERGING])
def test_a_diverging_run_is_silent_as_in_torch(st, real, name, kwargs):
    def diverge(t):
        w = t.tensor([1.0, -2.0, 0.5], requires_grad=True)
        optimiser = getattr(t.optim, name)([w], **kwargs)
        for _ in range(60):
            optimiser.zero_grad()
            (w**4).sum().backward()
            optimiser.step()
        return w.detach().numpy().copy()

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # NumPy's floating point warnings are RuntimeWarnings
        got, want = diverge(st), diverge(real)
    np.testing.assert_array_equal(np.isfinite(got), np.isfinite(want))  # inf and nan show up in the same places


def test_loading_a_state_too_big_for_the_dtype_is_silent(st, real):
    def run(t):
        double = t.tensor([1.0, 2.0], dtype=t.float64, requires_grad=True)
        source = t.optim.Adam([double], lr=0.1)
        double.grad = t.tensor([1e200, 1e200], dtype=t.float64)
        source.step()
        single = t.tensor([1.0, 2.0], requires_grad=True)
        target = t.optim.Adam([single], lr=0.1)
        target.load_state_dict(source.state_dict())
        return target.state[single]["exp_avg"], target.state[single]["exp_avg_sq"]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        agree(run, st, real)


# ---------------------------------------------------------------- a gradient that goes missing and comes back

# Steps in which b's gradient is present (a's always is): b starts without one, gets one, loses
# it again and comes back, so a skipped parameter is seen both before and after it has state.
B_PRESENT = [False, False, True, True, False, True, False, True]

GAPPY = [
    ("SGD", {"lr": 0.05, "momentum": 0.9}),
    ("SGD", {"lr": 0.05, "momentum": 0.9, "nesterov": True, "weight_decay": 0.1}),
    ("SGD", {"lr": 0.05}),
    ("RMSprop", {"lr": 0.01, "momentum": 0.5, "centered": True}),
    ("Adagrad", {"lr": 0.1, "lr_decay": 0.1}),
    ("Adam", {"lr": 0.05, "amsgrad": True}),
    ("AdamW", {"lr": 0.05}),
]


def gappy_run(t, name, kwargs):
    """Eight steps in which b gets no gradient in some of them; a snapshot of the state after each."""
    a = t.tensor([1.0, -2.0], requires_grad=True)
    b = t.tensor([0.5, 2.0], requires_grad=True)
    optimiser = getattr(t.optim, name)([a, b], **kwargs)
    snapshots = [(len(optimiser.state), a in optimiser.state, b in optimiser.state)]
    for present in B_PRESENT:
        optimiser.zero_grad()
        loss = (a**2).sum() + ((b**2).sum() if present else 0.0)
        loss.backward()
        optimiser.step()
        snapshots.append(
            (
                len(optimiser.state),
                b in optimiser.state,
                sorted(optimiser.state[b]) if b in optimiser.state else None,
                a.detach().clone(),
                b.detach().clone(),
                b.grad is None,
            )
        )
    return snapshots, optimiser.state_dict()


@pytest.mark.parametrize("name, kwargs", GAPPY, ids=[f"{n}-{'-'.join(f'{k}={v}' for k, v in kw.items())}" for n, kw in GAPPY])
def test_a_gradient_that_goes_missing_and_returns_matches_torch(st, real, name, kwargs):
    got_steps, got_state = gappy_run(st, name, kwargs)
    want_steps, want_state = gappy_run(real, name, kwargs)
    for got, want in zip(got_steps, want_steps):
        same(got, want)
    same_nested(got_state, want_state)


def test_a_skipped_parameter_gets_no_state_entry_until_it_has_a_gradient(st):
    a = st.tensor([1.0, -2.0], requires_grad=True)
    b = st.tensor([0.5, 2.0], requires_grad=True)
    for name, kwargs in [("SGD", {"momentum": 0.9}), ("RMSprop", {}), ("Adam", {})]:
        optimiser = getattr(st.optim, name)([a, b], lr=0.01, **kwargs)
        a.grad = st.tensor([1.0, 1.0])
        b.grad = None
        optimiser.step()
        assert len(optimiser.state) == 1 and b not in optimiser.state
        b.grad = st.tensor([1.0, 1.0])
        optimiser.step()
        assert len(optimiser.state) == 2 and len(optimiser.state[b]) > 0


def test_adagrad_makes_its_state_when_built_and_for_a_group_added_later(st, real):
    def run(t):
        a = t.tensor([1.0, -2.0], requires_grad=True)
        b = t.tensor([0.5, 2.0], requires_grad=True)
        c = t.tensor([3.0], requires_grad=True)
        optimiser = t.optim.Adagrad([a, b], lr=0.1, initial_accumulator_value=0.25)
        built = (len(optimiser.state), sorted(optimiser.state[b]), optimiser.state[b]["sum"], optimiser.state[b]["step"])
        optimiser.add_param_group({"params": [c], "lr": 0.2})
        a.grad = t.tensor([1.0, 1.0])
        c.grad = t.tensor([2.0])
        optimiser.step()
        return built, len(optimiser.state), c in optimiser.state, optimiser.state[c]["sum"], a.detach(), c.detach()

    agree(run, st, real)


# ---------------------------------------------------------------- state_dict and load_state_dict


def same_nested(got, want):
    """A state_dict equals torch's: same keys in the same order, same tensors and numbers."""
    if isinstance(want, dict):
        assert type(got) is type(want)
        assert list(got) == list(want)
        for key in want:
            same_nested(got[key], want[key])
    elif isinstance(want, (list, tuple)):
        assert type(got) is type(want) and len(got) == len(want)
        for g, w in zip(got, want):
            same_nested(g, w)
    elif hasattr(want, "dtype") and hasattr(want, "shape"):
        same(got, want)
    else:
        assert type(got) is type(want) and got == want


def train(t, optimiser, params, steps):
    for _ in range(steps):
        optimiser.zero_grad()
        quadratic_loss(t, params).backward()
        optimiser.step()


STATEFUL = [
    ("SGD", {"lr": 0.05, "momentum": 0.9}),
    ("SGD", {"lr": 0.05}),
    ("RMSprop", {"lr": 0.01}),
    ("RMSprop", {"lr": 0.01, "momentum": 0.9, "centered": True}),
    ("Adagrad", {"lr": 0.1, "lr_decay": 0.05, "initial_accumulator_value": 0.1}),
    ("Adam", {"lr": 0.05, "amsgrad": True}),
    ("AdamW", {"lr": 0.05}),
]


@pytest.mark.parametrize("name, kwargs", STATEFUL)
def test_state_dict_has_torchs_structure(st, real, name, kwargs):
    def build(t):
        params = quadratic_params(t)
        optimiser = getattr(t.optim, name)(params, **kwargs)
        train(t, optimiser, params, 3)
        return optimiser.state_dict()

    same_nested(build(st), build(real))


@pytest.mark.parametrize("through_a_file", [False, True], ids=["in memory", "through torch.save"])
@pytest.mark.parametrize("name, kwargs", STATEFUL)
def test_a_loaded_state_dict_continues_the_run(st, real, name, kwargs, through_a_file):
    def straight(t):
        params = quadratic_params(t)
        train(t, getattr(t.optim, name)(params, **kwargs), params, 8)
        return [p.detach().numpy() for p in params]

    params = quadratic_params(st)
    optimiser = getattr(st.optim, name)(params, **kwargs)
    train(st, optimiser, params, 4)
    saved = optimiser.state_dict()
    if through_a_file:
        buffer = io.BytesIO()
        st.save(saved, buffer)
        buffer.seek(0)
        saved = st.load(buffer)
    # A new optimiser on copies of the parameters as they are at this point in the run.
    fresh = [p.detach().clone().requires_grad_() for p in params]
    second = getattr(st.optim, name)(fresh, **kwargs)
    second.load_state_dict(saved)
    train(st, second, fresh, 4)
    for got, want in zip([p.detach().numpy() for p in fresh], straight(real)):
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_load_state_dict_takes_the_saved_hyperparameters(st, real):
    def run(t):
        w = t.tensor([1.0, 2.0], requires_grad=True)
        saved = t.optim.SGD([w], lr=0.3, momentum=0.7).state_dict()
        other = t.optim.SGD([w], lr=0.1)
        other.load_state_dict(saved)
        return other.param_groups[0]["lr"], other.param_groups[0]["momentum"], other.param_groups[0]["params"][0] is w

    agree(run, st, real)


def test_loaded_state_is_the_dicts_own_tensors_as_in_torch(st, real):
    # torch does not copy same-dtype state on load, so the two optimisers share a buffer.
    def run(t):
        w = t.tensor([1.0, 2.0], requires_grad=True)
        first = t.optim.SGD([w], lr=0.1, momentum=0.9)
        w.grad = t.tensor([1.0, 1.0])
        first.step()
        saved = first.state_dict()
        second = t.optim.SGD([w], lr=0.1, momentum=0.9)
        second.load_state_dict(saved)
        first.step()
        return second.state[w]["momentum_buffer"], second.state[w]["momentum_buffer"] is first.state[w]["momentum_buffer"]

    agree(run, st, real)


def test_load_state_dict_checks_the_groups(st, real):
    def different_count(t):
        a = t.tensor([1.0], requires_grad=True)
        b = t.tensor([1.0], requires_grad=True)
        t.optim.SGD([a], lr=0.1).load_state_dict(t.optim.SGD([{"params": [a]}, {"params": [b]}], lr=0.1).state_dict())

    def different_size(t):
        a = t.tensor([1.0], requires_grad=True)
        b = t.tensor([1.0], requires_grad=True)
        t.optim.SGD([a], lr=0.1).load_state_dict(t.optim.SGD([a, b], lr=0.1).state_dict())

    agree(different_count, st, real)
    agree(different_size, st, real)
    with pytest.raises(ValueError, match="different number of parameter groups"):
        different_count(st)
    with pytest.raises(ValueError, match="doesn't match the size of optimizer's group"):
        different_size(st)


def test_state_is_cast_to_the_parameters_dtype(st, real):
    def run(t):
        w = t.tensor([1.0, 2.0], requires_grad=True)
        source = t.optim.Adam([w], lr=0.1)
        w.grad = t.tensor([1.0, 1.0])
        source.step()
        double = t.tensor([1.0, 2.0], dtype=t.float64, requires_grad=True)
        target = t.optim.Adam([double], lr=0.1)
        target.load_state_dict(source.state_dict())
        state = target.state[double]
        return state["exp_avg"], state["exp_avg_sq"], state["step"]

    agree(run, st, real)


# ---------------------------------------------------------------- what is not here


@pytest.mark.parametrize("name", ["LBFGS", "SparseAdam", "Adadelta"])
def test_other_optimisers_say_they_are_not_available(st, name):
    with pytest.raises(NotImplementedError, match=f"torch.optim.{name} is not available in the browser version"):
        getattr(st.optim, name)


# ---------------------------------------------------------------- learning rate schedulers


def warmup(epoch):
    return min(1.0, (epoch + 1) / 5)


def decay(epoch):
    return 0.95**epoch


SCHEDULERS = [
    ("StepLR", lambda t, o: t.optim.lr_scheduler.StepLR(o, step_size=3, gamma=0.5)),
    ("StepLR defaults", lambda t, o: t.optim.lr_scheduler.StepLR(o, 4)),
    ("ExponentialLR", lambda t, o: t.optim.lr_scheduler.ExponentialLR(o, gamma=0.9)),
    ("CosineAnnealingLR", lambda t, o: t.optim.lr_scheduler.CosineAnnealingLR(o, T_max=7)),
    ("CosineAnnealingLR eta_min", lambda t, o: t.optim.lr_scheduler.CosineAnnealingLR(o, T_max=5, eta_min=0.001)),
    ("CosineAnnealingLR one half", lambda t, o: t.optim.lr_scheduler.CosineAnnealingLR(o, T_max=20)),
    ("LambdaLR decay", lambda t, o: t.optim.lr_scheduler.LambdaLR(o, lr_lambda=decay)),
    ("LambdaLR warmup", lambda t, o: t.optim.lr_scheduler.LambdaLR(o, warmup)),
    ("LambdaLR per group", lambda t, o: t.optim.lr_scheduler.LambdaLR(o, [warmup, decay])),
    ("MultiStepLR", lambda t, o: t.optim.lr_scheduler.MultiStepLR(o, milestones=[3, 8, 15], gamma=0.5)),
    ("MultiStepLR repeated milestone", lambda t, o: t.optim.lr_scheduler.MultiStepLR(o, [2, 5, 5, 5, 9, 0])),
    ("MultiStepLR unsorted", lambda t, o: t.optim.lr_scheduler.MultiStepLR(o, (12, 4, 7), 0.8)),
    ("LinearLR defaults", lambda t, o: t.optim.lr_scheduler.LinearLR(o)),
    ("LinearLR warmup", lambda t, o: t.optim.lr_scheduler.LinearLR(o, start_factor=0.1, total_iters=6)),
    ("LinearLR down", lambda t, o: t.optim.lr_scheduler.LinearLR(o, start_factor=1.0, end_factor=0.2, total_iters=10)),
    ("LinearLR both ends", lambda t, o: t.optim.lr_scheduler.LinearLR(o, 0.5, 0.9, 7)),
    ("OneCycleLR", lambda t, o: t.optim.lr_scheduler.OneCycleLR(o, max_lr=0.5, total_steps=20)),
    ("OneCycleLR per group", lambda t, o: t.optim.lr_scheduler.OneCycleLR(o, [0.5, 0.2], epochs=4, steps_per_epoch=5)),
    ("OneCycleLR linear", lambda t, o: t.optim.lr_scheduler.OneCycleLR(o, 0.3, total_steps=20, anneal_strategy="linear")),
    ("OneCycleLR three phases", lambda t, o: t.optim.lr_scheduler.OneCycleLR(o, 0.3, total_steps=20, three_phase=True)),
    (
        "OneCycleLR three phases linear",
        lambda t, o: t.optim.lr_scheduler.OneCycleLR(
            o, 0.3, total_steps=21, pct_start=0.25, anneal_strategy="linear", three_phase=True
        ),
    ),
    ("OneCycleLR pct_start", lambda t, o: t.optim.lr_scheduler.OneCycleLR(o, 0.3, total_steps=20, pct_start=0.5)),
    (
        "OneCycleLR momentum range",
        lambda t, o: t.optim.lr_scheduler.OneCycleLR(
            o, 0.3, total_steps=20, base_momentum=[0.7, 0.8], max_momentum=0.9, div_factor=10.0, final_div_factor=100.0
        ),
    ),
    ("OneCycleLR without momentum", lambda t, o: t.optim.lr_scheduler.OneCycleLR(o, 0.3, total_steps=20, cycle_momentum=False)),
]
SCHEDULER_IDS = [name for name, _ in SCHEDULERS]


def two_group_optimiser(t):
    w, m = quadratic_params(t)
    return t.optim.SGD([{"params": [w], "lr": 0.1}, {"params": [m], "lr": 0.02}], lr=0.05, momentum=0.9)


def momentum_of(group):
    """The momentum a group holds, or Adam's beta1: what OneCycleLR cycles."""
    return group["betas"][0] if "betas" in group else group.get("momentum")


def lr_history(t, build, epochs=20, make_optimiser=two_group_optimiser):
    """Each group's lr (from get_last_lr and from the optimiser) and momentum (or beta1), before
    training and after each of `epochs` steps."""
    optimiser = make_optimiser(t)
    scheduler = build(t, optimiser)

    def now():
        return scheduler.get_last_lr(), [g["lr"] for g in optimiser.param_groups], [momentum_of(g) for g in optimiser.param_groups]

    history = [now()]
    for _ in range(epochs):
        optimiser.step()
        scheduler.step()
        history.append(now())
    return history, scheduler.last_epoch, scheduler.base_lrs, [g["initial_lr"] for g in optimiser.param_groups]


@pytest.mark.parametrize("name, build", SCHEDULERS, ids=SCHEDULER_IDS)
def test_lr_over_twenty_epochs_equals_torchs(st, real, name, build):
    got, want = lr_history(st, build), lr_history(real, build)
    assert len(got[0]) == 21
    assert got[0] == want[0]  # the same float operations, so exactly the same floats
    assert got[1:] == want[1:]


@pytest.mark.parametrize("name, build", SCHEDULERS, ids=SCHEDULER_IDS)
def test_a_scheduler_drives_the_optimisers_steps(st, real, name, build):
    def run(t):
        params = quadratic_params(t)
        groups = [{"params": [params[0]], "lr": 0.1}, {"params": [params[1]], "lr": 0.02}]
        optimiser = t.optim.SGD(groups, lr=0.05, momentum=0.5)
        scheduler = build(t, optimiser)
        for _ in range(12):
            optimiser.zero_grad()
            quadratic_loss(t, params).backward()
            optimiser.step()
            scheduler.step()
        return [p.detach().numpy().copy() for p in params], scheduler.get_last_lr()

    got, want = run(st), run(real)
    for g, w in zip(got[0], want[0]):
        np.testing.assert_allclose(g, w, rtol=1e-5, atol=1e-5)
    assert got[1] == want[1]


def test_get_last_lr_is_a_list_of_floats_with_a_value_per_group(st, real):
    def run(t):
        optimiser = two_group_optimiser(t)
        scheduler = t.optim.lr_scheduler.StepLR(optimiser, step_size=2)
        optimiser.step()
        scheduler.step()
        scheduler.step()
        last = scheduler.get_last_lr()
        return type(last).__name__, [type(v).__name__ for v in last], len(last), scheduler.last_epoch

    agree(run, st, real)


def test_a_scheduler_records_the_starting_lr_in_the_groups(st, real):
    def run(t):
        optimiser = two_group_optimiser(t)
        t.optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.5)
        optimiser.step()
        return [g["initial_lr"] for g in optimiser.param_groups], list(optimiser.param_groups[0])

    agree(run, st, real)


@pytest.mark.parametrize("name, build", SCHEDULERS, ids=SCHEDULER_IDS)
def test_a_scheduler_resumes_from_its_state_dict(st, real, name, build):
    def run(t):
        optimiser = two_group_optimiser(t)
        scheduler = build(t, optimiser)
        for _ in range(7):
            optimiser.step()
            scheduler.step()
        saved = scheduler.state_dict()
        saved_optimiser = optimiser.state_dict()
        # A new run with new objects, loaded in torch's order: scheduler first, then the optimiser.
        resumed = two_group_optimiser(t)
        again = build(t, resumed)
        again.load_state_dict(saved)
        resumed.load_state_dict(saved_optimiser)
        rates = []
        for _ in range(8):
            resumed.step()
            again.step()
            rates.append(again.get_last_lr())
        return rates, again.last_epoch

    got, want = run(st), run(real)
    assert got == want


@pytest.mark.parametrize(
    "build",
    [
        lambda t: t.optim.lr_scheduler.StepLR("not an optimiser", 3),
        lambda t: t.optim.lr_scheduler.LambdaLR(two_group_optimiser(t), [warmup, decay, decay]),
        lambda t: t.optim.lr_scheduler.StepLR(two_group_optimiser(t), 3, last_epoch=4),
    ],
    ids=["not an optimiser", "wrong number of lambdas", "resume without initial_lr"],
)
def test_scheduler_errors_match_torch(st, real, build):
    def run(t):
        build(t)

    agree(run, st, real)


def test_a_scheduler_names_what_it_refuses(st):
    with pytest.raises(TypeError, match="str is not an Optimizer"):
        st.optim.lr_scheduler.StepLR("not an optimiser", 3)
    with pytest.raises(ValueError, match="Expected 2 lr_lambdas, but got 3"):
        st.optim.lr_scheduler.LambdaLR(two_group_optimiser(st), [warmup, decay, decay])
    with pytest.raises(KeyError, match="initial_lr"):
        st.optim.lr_scheduler.StepLR(two_group_optimiser(st), 3, last_epoch=4)


@pytest.mark.parametrize("name", ["CyclicLR", "PolynomialLR", "SequentialLR", "ChainedScheduler"])
def test_other_schedulers_say_they_are_not_available(st, name):
    with pytest.raises(NotImplementedError, match=f"torch.optim.lr_scheduler.{name} is not available in the browser"):
        getattr(st.optim.lr_scheduler, name)


def two_group_adam(t):
    w, m = quadratic_params(t)
    return t.optim.Adam([{"params": [w], "lr": 0.1}, {"params": [m], "lr": 0.02, "betas": (0.8, 0.99)}], lr=0.05)


def two_group_rmsprop(t):
    w, m = quadratic_params(t)
    return t.optim.RMSprop([{"params": [w], "lr": 0.1}, {"params": [m], "lr": 0.02, "momentum": 0.3}], lr=0.05, momentum=0.9)


@pytest.mark.parametrize("make", [two_group_adam, two_group_rmsprop], ids=["Adam beta1", "RMSprop momentum"])
@pytest.mark.parametrize("name, build", SCHEDULERS[-8:], ids=SCHEDULER_IDS[-8:])
def test_one_cycle_moves_adams_beta1_and_other_optimisers_momentum(st, real, name, build, make):
    got = lr_history(st, build, make_optimiser=make)
    want = lr_history(real, build, make_optimiser=make)
    assert got == want


def test_one_cycle_leaves_the_second_beta_alone_and_trains_like_torch(st, real):
    def run(t):
        params = quadratic_params(t)
        optimiser = t.optim.Adam(params, lr=0.1)
        scheduler = t.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=0.2, total_steps=15)
        for _ in range(15):
            optimiser.zero_grad()
            quadratic_loss(t, params).backward()
            optimiser.step()
            scheduler.step()
        return [p.detach().numpy().copy() for p in params], optimiser.param_groups[0]["betas"]

    got, want = run(st), run(real)
    for g, w in zip(got[0], want[0]):
        np.testing.assert_allclose(g, w, rtol=1e-5, atol=1e-5)
    assert got[1] == want[1] and got[1][1] == 0.999


def test_one_cycle_writes_its_values_into_the_groups_as_torch_does(st, real):
    def run(t):
        optimiser = two_group_optimiser(t)
        t.optim.lr_scheduler.OneCycleLR(optimiser, [0.5, 0.2], total_steps=10, base_momentum=0.8, max_momentum=0.9)
        return [{k: v for k, v in g.items() if k != "params"} for g in optimiser.param_groups]

    assert run(st) == run(real)


def test_one_cycle_refuses_to_run_past_its_steps(st, real):
    def run(t):
        optimiser = two_group_optimiser(t)
        scheduler = t.optim.lr_scheduler.OneCycleLR(optimiser, 0.3, total_steps=4)
        for _ in range(6):
            optimiser.step()
            scheduler.step()

    agree(run, st, real)
    with pytest.raises(ValueError, match="Tried to step 5 times. The specified number of total steps is 4"):
        run(st)


@pytest.mark.parametrize(
    "build",
    [
        lambda t, o: t.optim.lr_scheduler.OneCycleLR(o, 0.3),
        lambda t, o: t.optim.lr_scheduler.OneCycleLR(o, 0.3, epochs=3),
        lambda t, o: t.optim.lr_scheduler.OneCycleLR(o, 0.3, total_steps=0),
        lambda t, o: t.optim.lr_scheduler.OneCycleLR(o, 0.3, total_steps=2.5),
        lambda t, o: t.optim.lr_scheduler.OneCycleLR(o, 0.3, epochs=0, steps_per_epoch=3),
        lambda t, o: t.optim.lr_scheduler.OneCycleLR(o, 0.3, epochs=2, steps_per_epoch=-1),
        lambda t, o: t.optim.lr_scheduler.OneCycleLR(o, 0.3, total_steps=10, pct_start=1.5),
        lambda t, o: t.optim.lr_scheduler.OneCycleLR(o, 0.3, total_steps=10, pct_start=1),
        lambda t, o: t.optim.lr_scheduler.OneCycleLR(o, 0.3, total_steps=10, anneal_strategy="exp"),
        lambda t, o: t.optim.lr_scheduler.OneCycleLR(o, [0.3, 0.2, 0.1], total_steps=10),
        lambda t, o: t.optim.lr_scheduler.OneCycleLR(o, 0.3, total_steps=10, max_momentum=[0.9]),
        lambda t, o: t.optim.lr_scheduler.OneCycleLR("optimiser", 0.3, total_steps=10),
        lambda t, o: t.optim.lr_scheduler.OneCycleLR(t.optim.Adagrad(o.param_groups[0]["params"]), 0.3, total_steps=10),
        lambda t, o: t.optim.lr_scheduler.LinearLR(o, start_factor=0.0),
        lambda t, o: t.optim.lr_scheduler.LinearLR(o, start_factor=1.5),
        lambda t, o: t.optim.lr_scheduler.LinearLR(o, end_factor=-0.1),
        lambda t, o: t.optim.lr_scheduler.LinearLR(o, end_factor=1.5),
    ],
    ids=[
        "no steps",
        "epochs only",
        "zero steps",
        "float steps",
        "zero epochs",
        "negative steps per epoch",
        "pct_start over 1",
        "pct_start an int",
        "unknown anneal",
        "wrong number of max_lr",
        "wrong number of momenta",
        "not an optimiser",
        "optimiser without momentum",
        "linear start 0",
        "linear start over 1",
        "linear negative end",
        "linear end over 1",
    ],
)
def test_more_scheduler_errors_match_torch(st, real, build):
    def run(t):
        build(t, two_group_optimiser(t))

    agree(run, st, real)


def test_one_cycle_without_cycling_accepts_an_optimiser_without_momentum(st, real):
    def run(t):
        w, m = quadratic_params(t)
        optimiser = t.optim.Adagrad([w, m], lr=0.1)
        scheduler = t.optim.lr_scheduler.OneCycleLR(optimiser, 0.3, total_steps=8, cycle_momentum=False)
        rates = []
        for _ in range(8):
            optimiser.step()
            scheduler.step()
            rates.append(scheduler.get_last_lr())
        return rates

    assert run(st) == run(real)


# ReduceLROnPlateau is stepped with a metric, not by epoch.
LOSSES = [1.0, 0.8, 0.79999, 0.85, 0.9, 0.7, 0.71, 0.72, 0.73, 0.74, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
LOSSES += [1.2] * 8  # a long enough plateau for even the default patience of 10

PLATEAUS = [
    ("defaults", {}),
    ("patience 2", {"patience": 2}),
    ("patience 0 factor", {"patience": 0, "factor": 0.5}),
    ("max mode", {"mode": "max", "patience": 2}),
    ("abs threshold", {"patience": 1, "threshold": 0.05, "threshold_mode": "abs"}),
    ("rel threshold", {"patience": 1, "threshold": 0.1}),
    ("max abs", {"mode": "max", "patience": 1, "threshold": 0.05, "threshold_mode": "abs"}),
    ("cooldown", {"patience": 1, "cooldown": 3}),
    ("min_lr", {"patience": 1, "min_lr": 0.004, "factor": 0.3}),
    ("min_lr per group", {"patience": 1, "min_lr": [0.01, 0.0005]}),
    ("eps", {"patience": 1, "eps": 0.5}),
]


# Falls that a relative threshold of 0.1 counts (0.715 after 0.8) and an absolute one of 0.1 does not.
SMALL_GAINS = [1.0, 0.8, 0.715, 0.7, 0.64, 0.64, 0.64, 0.6, 0.57, 0.5, 0.49, 0.48, 0.47, 0.46]


def plateau_history(t, options, metrics=LOSSES, as_tensor=False):
    optimiser = two_group_optimiser(t)
    scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimiser, **options)
    history = [scheduler.get_last_lr()]
    for metric in metrics:
        scheduler.step(t.tensor(metric) if as_tensor else metric)
        history.append(
            (
                scheduler.get_last_lr(),
                [g["lr"] for g in optimiser.param_groups],
                scheduler.best,
                scheduler.num_bad_epochs,
                scheduler.cooldown_counter,
                scheduler.last_epoch,
                scheduler.in_cooldown,
            )
        )
    return history


@pytest.mark.parametrize("name, options", PLATEAUS, ids=[name for name, _ in PLATEAUS])
def test_reduce_lr_on_plateau_follows_torch(st, real, name, options):
    got = plateau_history(st, options)
    assert got == plateau_history(real, options)
    assert len({tuple(h[1]) for h in got[1:]}) > 1 or name in ("min_lr per group", "eps")  # the lr did move


@pytest.mark.parametrize("mode", ["min", "max"])
@pytest.mark.parametrize("threshold_mode", ["rel", "abs"])
@pytest.mark.parametrize("metrics", [SMALL_GAINS, [-m for m in SMALL_GAINS]], ids=["falling", "rising"])
def test_plateau_thresholds_count_small_gains_as_torch_does(st, real, mode, threshold_mode, metrics):
    options = {"mode": mode, "threshold_mode": threshold_mode, "threshold": 0.1, "patience": 1, "factor": 0.5}
    assert plateau_history(st, options, metrics) == plateau_history(real, options, metrics)


def test_reduce_lr_on_plateau_takes_a_tensor_metric(st, real):
    assert plateau_history(st, {"patience": 2}, as_tensor=True) == plateau_history(real, {"patience": 2}, as_tensor=True)


def test_reduce_lr_on_plateau_drives_a_training_run(st, real):
    def run(t):
        params = quadratic_params(t)
        optimiser = t.optim.SGD(params, lr=0.2, momentum=0.5)
        scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=1, threshold=1.0, threshold_mode="abs")
        rates = []
        for _ in range(14):
            optimiser.zero_grad()
            loss = quadratic_loss(t, params)
            loss.backward()
            optimiser.step()
            scheduler.step(loss.item())
            rates.append(optimiser.param_groups[0]["lr"])
        return [p.detach().numpy().copy() for p in params], rates

    got, want = run(st), run(real)
    for g, w in zip(got[0], want[0]):
        np.testing.assert_allclose(g, w, rtol=1e-5, atol=1e-5)
    assert got[1] == want[1] and got[1][0] > got[1][-1]


def test_reduce_lr_on_plateau_resumes_from_its_state_dict(st, real):
    def run(t):
        optimiser = two_group_optimiser(t)
        scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="max", patience=1, cooldown=2)
        for metric in LOSSES[:9]:
            scheduler.step(metric)
        saved = scheduler.state_dict()
        again = t.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="max", patience=1, cooldown=2)
        again.load_state_dict(saved)
        for metric in LOSSES[9:]:
            again.step(metric)
        return again.get_last_lr(), again.best, again.num_bad_epochs, sorted(saved)

    assert run(st) == run(real)


@pytest.mark.parametrize(
    "build",
    [
        lambda t, o: t.optim.lr_scheduler.ReduceLROnPlateau(o, factor=1.0),
        lambda t, o: t.optim.lr_scheduler.ReduceLROnPlateau(o, factor=2.0),
        lambda t, o: t.optim.lr_scheduler.ReduceLROnPlateau(o, mode="smallest"),
        lambda t, o: t.optim.lr_scheduler.ReduceLROnPlateau(o, threshold_mode="relative"),
        lambda t, o: t.optim.lr_scheduler.ReduceLROnPlateau(o, min_lr=[0.1]),
        lambda t, o: t.optim.lr_scheduler.ReduceLROnPlateau("optimiser"),
        lambda t, o: t.optim.lr_scheduler.ReduceLROnPlateau(o).step(),
        lambda t, o: t.optim.lr_scheduler.ReduceLROnPlateau(o).step(t.tensor([1.0, 2.0])),
    ],
    ids=["factor 1", "factor 2", "unknown mode", "unknown threshold mode", "min_lr count", "not an optimiser", "no metric", "tensor metric of two values"],
)
def test_plateau_errors_match_torch(st, real, build):
    def run(t):
        build(t, two_group_optimiser(t))

    agree(run, st, real)


def test_a_group_added_after_a_plateau_scheduler_is_an_error_unless_min_lr_is_a_number(st, real):
    def run(t, min_lr):
        optimiser = two_group_optimiser(t)
        scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=0, min_lr=min_lr)
        optimiser.add_param_group({"params": [t.tensor([1.0], requires_grad=True)], "lr": 0.5})
        for _ in range(3):
            scheduler.step(1.0)
        return scheduler.get_last_lr()

    assert run(st, 0.001) == run(real, 0.001)
    agree(lambda t: run(t, [0.1, 0.1]), st, real)


# ---------------------------------------------------------------- data: collation

Point = namedtuple("Point", ["x", "y"])


def same_batch(got, want):
    """A collated batch equals torch's: the same containers (and their types), tensor dtypes,
    shapes and values, and plain values."""
    if isinstance(want, dict):
        assert type(got) is type(want) and list(got) == list(want)
        for key in want:
            same_batch(got[key], want[key])
    elif isinstance(want, (list, tuple)):
        assert type(got) is type(want) and len(got) == len(want)
        for g, w in zip(got, want):
            same_batch(g, w)
    elif hasattr(want, "dtype") and hasattr(want, "shape"):
        same(got, want)
    else:
        assert type(got) is type(want) and got == want


COLLATE_CASES = [
    ("tensors", lambda t: [t.tensor([1.0, 2.0]), t.tensor([3.0, 4.0])]),
    ("0-d tensors", lambda t: [t.tensor(1.0), t.tensor(2.0)]),
    ("int tensors", lambda t: [t.tensor([1, 2]), t.tensor([3, 4])]),
    ("float64 tensors", lambda t: [t.tensor([1.0], dtype=t.float64), t.tensor([2.0], dtype=t.float64)]),
    ("ints", lambda t: [1, 2, 3]),
    ("floats", lambda t: [0.5, 1.5, 2.5]),
    ("bools", lambda t: [True, False]),
    ("ints and a float", lambda t: [1, 2.5]),
    ("numpy float32 scalars", lambda t: [np.float32(1.5), np.float32(2.5)]),
    ("numpy int64 scalars", lambda t: [np.int64(1), np.int64(2)]),
    ("numpy bool scalars", lambda t: [np.bool_(True), np.bool_(False)]),
    ("numpy float64 arrays", lambda t: [np.ones((2, 3)), np.zeros((2, 3))]),
    ("numpy int32 arrays", lambda t: [np.arange(4, dtype=np.int32), np.arange(4, dtype=np.int32) * 2]),
    ("numpy uint8 arrays", lambda t: [np.full((2, 2), 7, dtype=np.uint8), np.full((2, 2), 9, dtype=np.uint8)]),
    ("0-d numpy arrays", lambda t: [np.array(1.5, dtype=np.float32), np.array(2.5, dtype=np.float32)]),
    ("strings", lambda t: ["a", "bc"]),
    ("bytes", lambda t: [b"a", b"bc"]),
    ("tuples of a tensor and an int", lambda t: [(t.tensor([1.0, 2.0]), 0), (t.tensor([3.0, 4.0]), 1)]),
    ("tuples of two tensors", lambda t: [(t.tensor([1.0]), t.tensor(0)), (t.tensor([2.0]), t.tensor(1))]),
    ("lists", lambda t: [[1, 2.0], [3, 4.0]]),
    ("tuple of a list and a tuple", lambda t: [([1, 2], (3.0, 4.0)), ([5, 6], (7.0, 8.0))]),
    ("dicts", lambda t: [{"x": t.tensor([1.0]), "n": 1, "w": 0.5, "name": "a"}, {"x": t.tensor([2.0]), "n": 2, "w": 1.5, "name": "b"}]),
    ("ordered dicts", lambda t: [OrderedDict(a=1, b=t.tensor(2.0)), OrderedDict(a=3, b=t.tensor(4.0))]),
    ("nested", lambda t: [{"pair": (t.tensor([1.0]), 1), "k": [1, 2]}, {"pair": (t.tensor([2.0]), 2), "k": [3, 4]}]),
    ("namedtuples", lambda t: [Point(t.tensor([1.0]), 1), Point(t.tensor([2.0]), 2)]),
    (
        "a mixed tuple",
        lambda t: [(t.tensor([1.0]), np.array([1, 2]), 0.25, "a", 3), (t.tensor([2.0]), np.array([3, 4]), 0.75, "b", 4)],
    ),
    ("a batch of one", lambda t: [(t.tensor([1.0, 2.0]), 5)]),
]
COLLATE_IDS = [name for name, _ in COLLATE_CASES]


@pytest.mark.parametrize("name, make", COLLATE_CASES, ids=COLLATE_IDS)
def test_default_collate_matches_torch(st, real, name, make):
    same_batch(st.utils.data.default_collate(make(st)), real.utils.data.default_collate(make(real)))


def test_default_collate_gives_torchs_dtypes(st):
    collate = st.utils.data.default_collate
    assert str(collate([1, 2]).dtype) == "torch.int64"
    assert str(collate([1.0, 2.0]).dtype) == "torch.float64"
    assert str(collate([st.tensor([1.0]), st.tensor([2.0])]).dtype) == "torch.float32"
    assert str(collate([True, False]).dtype) == "torch.bool"
    features, label = collate([(st.tensor([1.0, 2.0]), 0), (st.tensor([3.0, 4.0]), 1)])
    assert type(collate([(1, 2)])) is list  # a batch of tuples comes back as a list
    assert tuple(features.shape) == (2, 2) and str(label.dtype) == "torch.int64"


@pytest.mark.parametrize(
    "name, make",
    [
        ("ragged lists", lambda t: [[1, 2], [3]]),
        ("tensors of different shapes", lambda t: [t.tensor([1.0, 2.0]), t.tensor([1.0])]),
        ("None", lambda t: [None, None]),
        ("an object", lambda t: [object(), object()]),
        ("a set", lambda t: [{1}, {2}]),
        ("numpy strings", lambda t: [np.array(["a"]), np.array(["b"])]),
        ("an empty batch", lambda t: []),
        ("dicts with different keys", lambda t: [{"a": 1}, {"b": 2}]),
    ],
)
def test_default_collate_refuses_what_torch_refuses(st, real, name, make):
    def run(t):
        t.utils.data.default_collate(make(t))

    agree(run, st, real)


def test_default_collate_names_what_it_cannot_batch(st):
    with pytest.raises(TypeError, match="default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'NoneType'>"):
        st.utils.data.default_collate([None, None])
    with pytest.raises(RuntimeError, match="each element in list of batch should be of equal size"):
        st.utils.data.default_collate([[1, 2], [3]])


@pytest.mark.parametrize(
    "make",
    [
        lambda t: [t.tensor([1.0, 2.0]), t.tensor([1.0])],
        lambda t: [t.zeros(2, 3), t.zeros(2, 3), t.zeros(3, 2)],
        lambda t: [(t.tensor([1.0, 2.0]), 0), (t.tensor([1.0]), 1)],
        lambda t: [{"tokens": t.arange(3)}, {"tokens": t.arange(5)}],
        lambda t: [np.zeros(3), np.zeros(4)],
    ],
    ids=["vectors", "third entry", "inside a tuple", "inside a dict", "numpy arrays"],
)
def test_collating_samples_of_different_sizes_gives_torchs_message(st, real, make):
    with pytest.raises(RuntimeError) as want:
        real.utils.data.default_collate(make(real))
    with pytest.raises(RuntimeError) as got:
        st.utils.data.default_collate(make(st))
    assert str(got.value) == str(want.value)
    assert str(got.value).startswith("stack expects each tensor to be equal size, but got [")


def test_a_loader_over_variable_length_samples_fails_with_that_message(st, real):
    def run(t):
        class Sentences(t.utils.data.Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, i):
                return t.arange(i + 1)

        return list(t.utils.data.DataLoader(Sentences(), batch_size=2))

    agree(run, st, real)
    with pytest.raises(RuntimeError, match=r"stack expects each tensor to be equal size, but got \[1\] at entry 0 and \[2\] at entry 1"):
        run(st)


@pytest.mark.parametrize(
    "make",
    [
        lambda t: np.arange(6, dtype=np.float32).reshape(2, 3),
        lambda t: np.float32(2.5),
        lambda t: (np.arange(3), 5, "text"),
        lambda t: {"a": np.arange(2), "b": [np.ones(2), 1.5]},
        lambda t: Point(np.arange(2), 3),
        lambda t: 7,
        lambda t: "text",
        lambda t: None,
        lambda t: t.tensor([1.0, 2.0]),
    ],
    ids=["array", "numpy scalar", "tuple", "dict", "namedtuple", "int", "str", "None", "tensor"],
)
def test_default_convert_matches_torch(st, real, make):
    same_batch(st.utils.data.default_convert(make(st)), real.utils.data.default_convert(make(real)))


# ---------------------------------------------------------------- data: datasets and loaders


def table(t, n=10):
    """n rows: features [2i, 2i + 1] and label i, so a shuffle that splits rows is visible."""
    x = t.arange(2 * n, dtype=t.float32).reshape(n, 2)
    return t.utils.data.TensorDataset(x, t.arange(n))


def labels(batches):
    return [b[1].tolist() for b in batches]


def order(loader):
    return [i for batch in labels(loader) for i in batch]


def test_tensor_dataset_indexes_each_tensor_along_the_first_dimension(st, real):
    def run(t):
        data = table(t, 6)
        return (
            len(data),
            data[2],
            data[-1],
            data[1:3],
            data[[0, 4]],
            data[t.tensor(3)],
            len(data.tensors),
            type(data[0]).__name__,
        )

    agree(run, st, real)


def test_tensor_dataset_refuses_tensors_of_different_lengths(st, real):
    def run(t):
        t.utils.data.TensorDataset(t.zeros(3, 2), t.zeros(4))

    agree(run, st, real)
    with pytest.raises(AssertionError, match="Size mismatch between tensors"):
        run(st)


def test_a_dataset_must_define_getitem(st, real):
    def run(t):
        class Empty(t.utils.data.Dataset):
            pass

        Empty()[0]

    agree(run, st, real)
    with pytest.raises(NotImplementedError, match="Subclasses of Dataset should implement __getitem__."):
        run(st)


def test_subset(st, real):
    def run(t):
        data = table(t, 8)
        part = t.utils.data.Subset(data, [6, 1, 3])
        again = t.utils.data.Subset(part, [2, 0])
        return len(part), part[0], part[-1], part[[0, 2]], len(again), again[0], part.indices, part.dataset is data

    agree(run, st, real)


def test_subset_batches_through_getitems_when_the_dataset_has_it(st, real):
    def run(t):
        class Counting(t.utils.data.Dataset):
            def __init__(self):
                self.batch_calls = 0

            def __getitem__(self, i):
                return t.tensor(float(i))

            def __getitems__(self, indices):
                self.batch_calls += 1
                return [self[i] for i in indices]

            def __len__(self):
                return 6

        data = Counting()
        batches = list(t.utils.data.DataLoader(t.utils.data.Subset(data, [5, 4, 3, 2]), batch_size=2))
        return batches, data.batch_calls

    agree(run, st, real)


BATCHINGS = [(10, 3, False), (10, 3, True), (10, 5, False), (10, 5, True), (10, 10, False), (10, 20, False), (10, 20, True), (10, 1, False), (7, 2, False)]


@pytest.mark.parametrize("n, batch_size, drop_last", BATCHINGS)
def test_sequential_batches_and_length_match_torch(st, real, n, batch_size, drop_last):
    def run(t):
        loader = t.utils.data.DataLoader(table(t, n), batch_size=batch_size, drop_last=drop_last)
        return list(loader), len(loader)

    got, want = run(st), run(real)
    same_batch(got[0], want[0])
    assert got[1] == want[1]


def test_the_last_short_batch_is_kept_unless_drop_last(st):
    data = table(st, 10)
    g = st.Generator().manual_seed(0)
    kept = st.utils.data.DataLoader(data, batch_size=4, shuffle=True, generator=g)
    dropped = st.utils.data.DataLoader(data, batch_size=4, shuffle=True, drop_last=True, generator=g)
    assert [len(b[0]) for b in kept] == [4, 4, 2] and len(kept) == 3
    assert [len(b[0]) for b in dropped] == [4, 4] and len(dropped) == 2
    assert [len(b[0]) for b in st.utils.data.DataLoader(data, batch_size=4)] == [4, 4, 2]


def test_a_batch_size_larger_than_the_data_gives_one_batch_or_none_when_dropping(st):
    data = table(st, 3)
    assert len(list(st.utils.data.DataLoader(data, batch_size=8))) == 1
    assert list(st.utils.data.DataLoader(data, batch_size=8, drop_last=True)) == []


def test_shuffle_with_a_generator_of_the_same_seed_gives_the_same_batches(st):
    data = table(st, 20)

    def batches(seed, **options):
        loader = st.utils.data.DataLoader(data, batch_size=6, shuffle=True, generator=st.Generator().manual_seed(seed), **options)
        return labels(loader)

    assert batches(7) == batches(7)
    assert batches(7) != batches(8)
    assert batches(7, drop_last=True) == batches(7)[:-1]  # the same order, minus the short batch
    assert [len(b) for b in batches(7)] == [6, 6, 6, 2]


def test_one_generator_reseeded_repeats_an_epoch_and_otherwise_moves_on(st):
    g = st.Generator().manual_seed(11)
    loader = st.utils.data.DataLoader(table(st, 20), batch_size=5, shuffle=True, generator=g)
    first, second = order(loader), order(loader)
    assert first != second  # the second epoch is a new shuffle, as in torch
    g.manual_seed(11)
    assert order(loader) == first


def test_shuffle_without_a_generator_follows_the_global_seed(st):
    loader = st.utils.data.DataLoader(table(st, 20), batch_size=5, shuffle=True)
    st.manual_seed(5)
    a = order(loader)
    st.manual_seed(5)
    b = order(loader)
    st.manual_seed(6)
    c = order(loader)
    assert a == b and a != c


def test_each_shuffled_epoch_covers_the_data_once_with_rows_kept_together(st):
    loader = st.utils.data.DataLoader(table(st, 23), batch_size=5, shuffle=True, generator=st.Generator().manual_seed(1))
    for _ in range(3):
        seen = []
        for features, label in loader:
            np.testing.assert_array_equal(features[:, 0].numpy(), 2 * label.numpy())
            seen += label.tolist()
        assert sorted(seen) == list(range(23))
    assert seen != list(range(23))


def test_shuffled_batches_have_torchs_types_and_shapes(st, real):
    def run(t):
        g = t.Generator().manual_seed(0)
        loader = t.utils.data.DataLoader(table(t, 10), batch_size=4, shuffle=True, generator=g)
        return [(type(b).__name__, tuple(b[0].shape), str(b[0].dtype), tuple(b[1].shape), str(b[1].dtype)) for b in loader]

    assert run(st) == run(real)


def test_a_loader_can_be_iterated_again_and_again(st):
    loader = st.utils.data.DataLoader(table(st, 7), batch_size=3)
    first = labels(loader)
    assert labels(loader) == first == [[0, 1, 2], [3, 4, 5], [6]]
    it = iter(loader)
    assert len(it) == 3
    assert next(it)[1].tolist() == [0, 1, 2]
    assert list(it)[-1][1].tolist() == [6]
    with pytest.raises(StopIteration):
        next(it)


def test_workers_and_pinning_are_accepted_and_change_nothing(st, real):
    plain = labels(st.utils.data.DataLoader(table(st), batch_size=4))
    options = st.utils.data.DataLoader(
        table(st), batch_size=4, num_workers=2, pin_memory=True, prefetch_factor=4, persistent_workers=True, timeout=0
    )
    assert labels(options) == plain == labels(real.utils.data.DataLoader(table(real), batch_size=4))
    assert options.num_workers == 2 and options.pin_memory is True


def test_batch_size_none_gives_the_samples_converted_one_by_one(st, real):
    class Samples:
        def __init__(self, t):
            self.t = t

        def __len__(self):
            return 3

        def __getitem__(self, i):
            return {"x": np.arange(2, dtype=np.float32) + i, "id": i}

    def run(t):
        return list(t.utils.data.DataLoader(Samples(t), batch_size=None))

    same_batch(run(st), run(real))


def test_a_custom_dataset_with_dict_samples(st, real):
    def run(t):
        class Pairs(t.utils.data.Dataset):
            def __len__(self):
                return 5

            def __getitem__(self, i):
                return {"x": t.tensor([float(i), 1.0]), "y": i % 2, "name": f"row{i}"}

        return list(t.utils.data.DataLoader(Pairs(), batch_size=2))

    same_batch(run(st), run(real))


@pytest.mark.parametrize(
    "make",
    [
        lambda t: t.arange(7),
        lambda t: t.arange(14, dtype=t.float32).reshape(7, 2),
        lambda t: list(range(7)),
        lambda t: [float(i) for i in range(7)],
        lambda t: [(i, float(i) / 2) for i in range(7)],
        lambda t: [(t.tensor([float(i)]), i) for i in range(7)],
        lambda t: np.arange(7),
        lambda t: np.arange(14, dtype=np.float32).reshape(7, 2),
        lambda t: range(7),
    ],
    ids=["tensor", "tensor rows", "ints", "floats", "tuples", "tensor tuples", "numpy", "numpy rows", "range"],
)
def test_any_sequence_can_be_the_dataset(st, real, make):
    def run(t):
        return list(t.utils.data.DataLoader(make(t), batch_size=3))

    same_batch(run(st), run(real))


def test_a_custom_collate_fn_gets_the_list_of_samples(st, real):
    def run(t):
        seen = []

        def collate(samples):
            seen.append(len(samples))
            return t.stack([x for x, _ in samples]).sum(0), [int(y) for _, y in samples]

        out = list(t.utils.data.DataLoader(table(t, 5), batch_size=2, collate_fn=collate))
        return out, seen

    same_batch(run(st), run(real))


def test_a_sampler_or_batch_sampler_can_be_given(st, real):
    def run(t):
        by_sampler = t.utils.data.DataLoader(table(t), batch_size=2, sampler=[7, 2, 9, 0, 5])
        by_batches = t.utils.data.DataLoader(table(t), batch_sampler=[[1, 2, 3], [9], [4, 5]])
        sequential = t.utils.data.DataLoader(table(t), batch_size=4, sampler=t.utils.data.SequentialSampler(range(10)))
        return list(by_sampler), len(by_sampler), list(by_batches), len(by_batches), list(sequential)

    same_batch(run(st), run(real))


@pytest.mark.parametrize(
    "options",
    [
        {"shuffle": True, "sampler": [0, 1]},
        {"batch_sampler": [[0, 1]], "batch_size": 2},
        {"batch_sampler": [[0, 1]], "shuffle": True},
        {"batch_sampler": [[0, 1]], "drop_last": True},
        {"batch_size": None, "drop_last": True},
        {"batch_size": 0},
        {"batch_size": -1},
        {"batch_size": 2.5},
        {"num_workers": -1},
        {"timeout": -1},
        {"prefetch_factor": 2},
        {"persistent_workers": True},
        {"drop_last": "yes"},
    ],
    ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()),
)
def test_incompatible_options_fail_as_in_torch(st, real, options):
    def run(t):
        t.utils.data.DataLoader(table(t), **options)

    agree(run, st, real)


def test_loader_attributes(st):
    data = table(st)
    loader = st.utils.data.DataLoader(data, batch_size=4, shuffle=True, drop_last=True)
    assert loader.dataset is data and loader.batch_size == 4 and loader.drop_last is True
    assert type(loader.sampler).__name__ == "RandomSampler"
    assert type(st.utils.data.DataLoader(data).sampler).__name__ == "SequentialSampler"
    assert loader.batch_sampler.batch_size == 4 and loader.collate_fn is st.utils.data.default_collate
    assert len(loader.dataset) == 10


def test_samplers_match_torch(st, real):
    def run(t):
        data = t.utils.data
        batches = data.BatchSampler(data.SequentialSampler(range(7)), batch_size=3, drop_last=False)
        dropped = data.BatchSampler(range(7), 3, True)
        return list(data.SequentialSampler(range(4))), len(data.SequentialSampler(range(4))), list(batches), len(batches), list(dropped), len(dropped)

    agree(run, st, real)


def test_random_sampler(st, real):
    def kinds(t):
        data = t.utils.data
        return len(data.RandomSampler(range(10))), len(data.RandomSampler(range(10), num_samples=25)), len(data.RandomSampler(range(10), replacement=True, num_samples=7))

    assert kinds(st) == kinds(real)
    data = st.utils.data
    assert sorted(data.RandomSampler(range(10), generator=st.Generator().manual_seed(1))) == list(range(10))
    longer = list(data.RandomSampler(range(10), num_samples=25, generator=st.Generator().manual_seed(1)))
    assert len(longer) == 25 and sorted(longer[:10]) == list(range(10)) and sorted(longer[10:20]) == list(range(10))
    drawn = list(data.RandomSampler(range(5), replacement=True, num_samples=40, generator=st.Generator().manual_seed(2)))
    assert len(drawn) == 40 and set(drawn) <= set(range(5)) and len(set(drawn)) < 40
    same_seed = [list(data.RandomSampler(range(30), generator=st.Generator().manual_seed(4))) for _ in range(2)]
    assert same_seed[0] == same_seed[1]
    with pytest.raises(ValueError, match="num_samples should be a positive integer value, but got num_samples=0"):
        data.RandomSampler(range(5), num_samples=0)
    with pytest.raises(TypeError, match="replacement should be a boolean value"):
        data.RandomSampler(range(5), replacement=1)


# ---------------------------------------------------------------- data: random_split


@pytest.mark.parametrize(
    "n, lengths",
    [
        (10, [7, 3]),
        (10, [10]),
        (10, [5, 5, 0]),
        (10, [0.7, 0.3]),
        (10, [0.5, 0.25, 0.25]),
        (10, [0.35, 0.35, 0.3]),
        (7, [0.5, 0.5]),
        (11, [0.2, 0.2, 0.2, 0.2, 0.2]),
        (3, [0.1, 0.9]),
        (100, (0.8, 0.1, 0.1)),
        (10, [1.0]),
        (10, (6, 4)),
    ],
)
def test_random_split_sizes_match_torch(st, real, n, lengths):
    def sizes(t):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parts = t.utils.data.random_split(table(t, n), lengths)
        return [len(p) for p in parts], [type(p).__name__ for p in parts]

    assert sizes(st) == sizes(real)


@pytest.mark.parametrize("lengths", [[40, 25, 15], [0.5, 0.3, 0.2], [100]])
def test_random_split_parts_are_disjoint_and_cover_the_data(st, lengths):
    data = table(st, 80 if lengths != [100] else 100)
    n = len(data)
    lengths = [int(x) if x > 1 else x for x in lengths]
    parts = st.utils.data.random_split(data, lengths, generator=st.Generator().manual_seed(3))
    seen = [i for part in parts for i in part.indices]
    assert sorted(seen) == list(range(n))  # every row once: disjoint, and nothing missing
    assert all(part.dataset is data for part in parts)
    for part in parts:
        features, label = part[0]  # the subset reads the underlying row
        assert features[0].item() == 2 * label.item()


def test_random_split_with_a_generator_repeats_and_without_one_follows_the_global_seed(st):
    data = table(st, 30)
    split = lambda g: [p.indices for p in st.utils.data.random_split(data, [20, 10], generator=g)]
    assert split(st.Generator().manual_seed(9)) == split(st.Generator().manual_seed(9))
    assert split(st.Generator().manual_seed(9)) != split(st.Generator().manual_seed(10))
    st.manual_seed(3)
    a = [p.indices for p in st.utils.data.random_split(data, [20, 10])]
    st.manual_seed(3)
    assert [p.indices for p in st.utils.data.random_split(data, [20, 10])] == a
    assert sorted(a[0] + a[1]) == list(range(30))
    assert a[0] != sorted(a[0])  # shuffled, not the first 20 in order


@pytest.mark.parametrize(
    "lengths",
    [[6, 3], [6, 5], [0.5, 0.6], [1.5, -0.5], [0.4, 0.4], [-1, 11], [5.5, 4.5]],
    ids=["too few", "too many", "fractions over 1", "fraction outside 0..1", "fractions under 1", "negative", "float counts"],
)
def test_random_split_refuses_bad_lengths_as_torch_does(st, real, lengths):
    def run(t):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t.utils.data.random_split(table(t, 10), lengths)

    agree(run, st, real)


def test_random_split_names_the_mismatch_and_warns_about_an_empty_part(st):
    with pytest.raises(ValueError, match="Sum of input lengths does not equal the length of the input dataset!"):
        st.utils.data.random_split(table(st, 10), [6, 3])
    with pytest.raises(ValueError, match="Fraction at index 0 is not between 0 and 1"):
        st.utils.data.random_split(table(st, 10), [1.5, -0.5])
    with pytest.warns(UserWarning, match="Length of split at index 1 is 0"):
        parts = st.utils.data.random_split(table(st, 3), [1.0, 0.0])
    assert [len(p) for p in parts] == [3, 0]


def test_a_split_feeds_loaders(st):
    train, test = st.utils.data.random_split(table(st, 25), [0.8, 0.2], generator=st.Generator().manual_seed(0))
    loader = st.utils.data.DataLoader(train, batch_size=8, shuffle=True, generator=st.Generator().manual_seed(0))
    assert [len(b[0]) for b in loader] == [8, 8, 4] and len(test) == 5
    assert sorted(order(loader)) == sorted(train.indices)


def test_other_data_names_say_they_are_not_available(st):
    for name in ("WeightedRandomSampler", "IterableDataset", "ConcatDataset"):
        with pytest.raises(NotImplementedError, match=f"torch.utils.data.{name} is not available in the browser"):
            getattr(st.utils.data, name)
    with pytest.raises(NotImplementedError, match="torch.utils.hub is not available in the browser"):
        st.utils.hub


# ---------------------------------------------------------------- end to end

REGRESSION_RNG = np.random.default_rng(0)
FEATURES = REGRESSION_RNG.standard_normal((50, 3)).astype(np.float32)
TRUE_WEIGHTS = np.array([1.5, -2.0, 0.5], dtype=np.float32)
TARGETS = (FEATURES @ TRUE_WEIGHTS + 0.3 + 0.01 * REGRESSION_RNG.standard_normal(50)).astype(np.float32)
START_WEIGHTS = np.array([0.1, -0.1, 0.2], dtype=np.float32)


def fit_linear_regression(t, epochs, make_optimiser, **loader_options):
    """Mini-batch linear regression written out by hand, since torch.nn is another module."""
    data = t.utils.data.TensorDataset(t.tensor(FEATURES), t.tensor(TARGETS))
    loader = t.utils.data.DataLoader(data, batch_size=8, **loader_options)  # 50 rows: batches of 8, the last of 2
    w = t.tensor(START_WEIGHTS, requires_grad=True)
    b = t.zeros(1, requires_grad=True)
    optimiser = make_optimiser(t, [w, b])
    losses = []
    for _ in range(epochs):
        for x, y in loader:
            optimiser.zero_grad()
            loss = ((x @ w + b - y) ** 2).mean()
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
    return w.detach().numpy(), b.detach().numpy(), losses


def test_linear_regression_with_sgd_momentum_matches_torch_after_three_epochs(st, real):
    def make(t, params):
        return t.optim.SGD(params, lr=0.05, momentum=0.9)

    got = fit_linear_regression(st, 3, make, shuffle=False)
    want = fit_linear_regression(real, 3, make, shuffle=False)
    assert len(got[2]) == len(want[2]) == 21
    np.testing.assert_allclose(got[0], want[0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(got[1], want[1], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(got[2], want[2], rtol=1e-5, atol=1e-5)
    assert got[2][-1] < got[2][0] / 10  # and it is learning, not just agreeing


def test_shuffled_adam_training_finds_the_weights(st):
    def make(t, params):
        return t.optim.Adam(params, lr=0.05)

    w, b, losses = fit_linear_regression(st, 40, make, shuffle=True, generator=st.Generator().manual_seed(0))
    np.testing.assert_allclose(w, TRUE_WEIGHTS, atol=0.05)
    np.testing.assert_allclose(b, [0.3], atol=0.05)
    assert losses[-1] < 1e-2
