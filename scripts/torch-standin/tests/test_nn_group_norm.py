"""GroupNorm and F.group_norm against real torch: with the real layer's weights copied in, the
outputs, the input gradients and the weight and bias gradients agree, for several group counts
and for inputs of two to five dimensions."""

import numpy as np
import pytest

from nn_parity import function_agrees, grads_same, run, to_standin
from parity import agree, same

RNG = np.random.default_rng(23)


@pytest.fixture(autouse=True)
def fresh_numbers():
    """Every test draws the same numbers, whichever tests ran before it."""
    RNG.bit_generator.state = np.random.default_rng(23).bit_generator.state

# (name, channels, input shape): a matrix, a sequence, an image, a volume.
INPUTS = (
    ("matrix", 6, (5, 6)),
    ("sequence", 6, (3, 6, 4)),
    ("image", 6, (2, 6, 5, 3)),
    ("volume", 4, (2, 4, 3, 2, 2)),
    ("one sample", 6, (1, 6, 4, 4)),
)
GROUPS = (1, 2, 3, 6)


def divisors(channels):
    return [g for g in GROUPS if channels % g == 0]


# A group of one value has no variance, so what torch leaves of it in float32 is rounding noise
# (a gradient of 1e-5 where the stand-in has exactly 0); those groupings run in float64 only.
CASES = [
    pytest.param(name, g, shape, dtype, id=f"{name}-{g}groups-{dtype}")
    for name, channels, shape in INPUTS
    for g in divisors(channels)
    for dtype in ("float64", "float32")
    if dtype == "float64" or channels // g * int(np.prod(shape[2:])) > 1
]


def twin_group_norm(st, real, groups, channels, dtype, **kwargs):
    """A real GroupNorm with random weight and bias, and a stand-in one holding the same state."""
    r = real.nn.GroupNorm(groups, channels, **kwargs).to(dtype)
    with real.no_grad():
        for p in r.parameters():
            p.copy_(real.tensor(RNG.standard_normal(p.shape) + 0.5).to(dtype))
    s = st.nn.GroupNorm(groups, channels, **kwargs)
    if dtype == real.float64:
        s.double()
    assert repr(s.load_state_dict(to_standin(st, r.state_dict()))) == "<All keys matched successfully>"
    return s, r


def close(got, want, dtype):
    """Equal to float64's tolerance, or in float32 to the precision of its sums: a group of two
    values has a variance that is a difference of nearly equal squares, so its gradient has only
    four or five good digits in either library."""
    if dtype == "float64":
        return same(got, want)
    for g, w in zip(got, want):
        assert tuple(g.shape) == tuple(w.shape) and str(g.dtype) == str(w.dtype)
        scale = max(1.0, float(np.abs(w.numpy()).max()))
        np.testing.assert_allclose(g.numpy(), w.numpy(), rtol=1e-4, atol=1e-5 * scale)


@pytest.mark.parametrize("name,groups,shape,dtype", CASES)
def test_group_norm_layer_matches_torch(st, real, name, groups, shape, dtype):
    x = (RNG.standard_normal(shape) * 3 + 1).astype(dtype)
    s, r = twin_group_norm(st, real, groups, shape[1], getattr(real, dtype), eps=1e-3)
    for train in (True, False):
        s.train(train), r.train(train)
        (got_out,), got_grads = run(st, lambda t: s(t), [x])
        (want_out,), want_grads = run(real, lambda t: r(t), [x])
        close([got_out, *got_grads], [want_out, *want_grads], dtype)
        got_params = [p.grad for p in s.parameters()]
        want_params = [p.grad for p in r.parameters()]
        close(got_params, want_params, dtype)
        s.zero_grad(), r.zero_grad()


@pytest.mark.parametrize("kwargs", [{"affine": False}, {"bias": False}, {"eps": 0.1}, {"affine": False, "bias": False}])
def test_group_norm_without_affine_or_bias_matches_torch(st, real, kwargs):
    x = RNG.standard_normal((3, 6, 4, 2))
    s, r = twin_group_norm(st, real, 2, 6, real.float64, **kwargs)
    got = run(st, lambda t: s(t), [x])
    want = run(real, lambda t: r(t), [x])
    same(got, want)
    grads_same({n: p.grad for n, p in s.named_parameters()}, {n: p.grad for n, p in r.named_parameters()})
    assert [n for n, _ in s.named_parameters()] == [n for n, _ in r.named_parameters()]


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("value", [2.5, -0.3])
def test_group_norm_of_a_constant_input_is_the_bias_whatever_the_value(st, real, value, dtype):
    """With no variance every value of a group equals its mean, so the output is the bias and
    the scale gets no gradient. The input's gradient is not zero (eps is all that is left to
    divide by), but it still sums to zero over each group, as mean subtraction makes it. Each
    group holds 24 values here, so the sum that makes the mean has a few roundings to survive."""
    x = np.full((2, 6, 4, 3), value, dtype=dtype)
    s, r = twin_group_norm(st, real, 3, 6, getattr(real, dtype), eps=0.1)
    (got_out,), got_grads = run(st, lambda t: s(t), [x])
    (want_out,), want_grads = run(real, lambda t: r(t), [x])
    got_params = [p.grad for p in s.parameters()]
    want_params = [p.grad for p in r.parameters()]
    close([got_out, *got_grads, *got_params], [want_out, *want_grads, *want_params], dtype)
    bias = s.bias.detach().numpy().reshape(1, 6, 1, 1)
    np.testing.assert_allclose(got_out.numpy(), np.broadcast_to(bias, x.shape), atol=1e-5)
    np.testing.assert_allclose(got_grads[0].numpy().reshape(2, 3, 24).sum(axis=2), 0, atol=1e-4)
    assert np.abs(got_params[0].numpy()).max() < 1e-4, "the scale's gradient is a sum of upstream * 0"


@pytest.mark.parametrize(
    "make",
    [
        lambda nn: nn.GroupNorm(2, 4),
        lambda nn: nn.GroupNorm(1, 6, eps=1e-3),
        lambda nn: nn.GroupNorm(3, 6, affine=False),
        lambda nn: nn.GroupNorm(2, 4, bias=False),
        lambda nn: nn.GroupNorm(2, 4, affine=False, bias=False),
    ],
)
def test_group_norm_repr_and_state_dict_match_torch(st, real, make):
    s, r = make(st.nn), make(real.nn)
    assert repr(s) == repr(r)
    assert [(k, tuple(v.shape), str(v.dtype)) for k, v in s.state_dict().items()] == [
        (k, tuple(v.shape), str(v.dtype)) for k, v in r.state_dict().items()
    ]
    same(list(s.state_dict().values()), list(r.state_dict().values()))


def test_group_norm_inside_a_sequential_has_torchs_repr_and_keys(st, real):
    def make(nn):
        return nn.Sequential(nn.Conv2d(3, 4, 3), nn.GroupNorm(2, 4), nn.ReLU())

    s, r = make(st.nn), make(real.nn)
    assert repr(s) == repr(r)
    assert [(k, tuple(v.shape)) for k, v in s.state_dict().items()] == [
        (k, tuple(v.shape)) for k, v in r.state_dict().items()
    ]


def test_group_norm_starts_as_the_identity_scale_and_shift(st, real):
    layer = st.nn.GroupNorm(2, 4)
    assert layer.weight.tolist() == [1.0] * 4 and layer.bias.tolist() == [0.0] * 4
    assert (layer.num_groups, layer.num_channels, layer.eps, layer.affine) == (2, 4, 1e-5, True)


def test_group_norm_keeps_the_parameters_a_state_dict_round_trip_needs(st, real):
    s, r = twin_group_norm(st, real, 3, 6, real.float32)
    x = RNG.standard_normal((2, 6, 3)).astype(np.float32)
    again = st.nn.GroupNorm(3, 6)
    again.load_state_dict(s.state_dict())
    same(again(st.tensor(x)).detach(), r(real.tensor(x)).detach())


FUNCTIONS = (
    ("plain", lambda F, x: F.group_norm(x, 3)),
    ("one group", lambda F, x: F.group_norm(x, 1, eps=1e-2)),
    ("per channel", lambda F, x: F.group_norm(x, 6)),
)


@pytest.mark.parametrize("name,fn", FUNCTIONS, ids=[f[0] for f in FUNCTIONS])
def test_group_norm_function_matches_torch(st, real, name, fn):
    x = RNG.standard_normal((2, 6, 5, 3))
    function_agrees(st, real, fn, x)
    function_agrees(st, real, fn, x.astype(np.float32))


@pytest.mark.parametrize("which", ["weight", "bias", "both"])
def test_group_norm_function_with_a_weight_or_a_bias_matches_torch(st, real, which):
    x, w, b = RNG.standard_normal((2, 6, 4)), RNG.standard_normal(6), RNG.standard_normal(6)
    if which == "weight":
        function_agrees(st, real, lambda F, x, w: F.group_norm(x, 2, w), x, w)
    elif which == "bias":
        function_agrees(st, real, lambda F, x, b: F.group_norm(x, 2, None, b), x, b)
    else:
        function_agrees(st, real, lambda F, x, w, b: F.group_norm(x, 2, w, b, 1e-3), x, w, b)


def test_group_norm_gives_each_group_zero_mean_and_unit_variance(st):
    x = st.tensor(RNG.standard_normal((4, 6, 5)) * 7 + 3)
    out = st.nn.GroupNorm(3, 6, eps=0.0).double()(x).detach().numpy().reshape(4, 3, 10)
    np.testing.assert_allclose(out.mean(axis=2), 0, atol=1e-9)
    np.testing.assert_allclose(out.var(axis=2), 1, atol=1e-9)


def test_group_norm_does_not_depend_on_the_batch(st):
    x = st.tensor(RNG.standard_normal((4, 6, 5)).astype(np.float32))
    layer = st.nn.GroupNorm(2, 6)
    np.testing.assert_allclose(layer(x)[1:2].detach().numpy(), layer(x[1:2]).detach().numpy(), rtol=1e-6, atol=1e-6)


def test_group_norm_second_derivative_matches_torch(st, real):
    x = RNG.standard_normal((2, 4, 3))

    def run_twice(t):
        v = t.tensor(x, requires_grad=True)
        (g,) = t.autograd.grad((t.nn.functional.group_norm(v, 2) ** 3).sum(), v, create_graph=True)
        (g * g).sum().backward()
        return g.detach(), v.grad

    agree(run_twice, st, real)


@pytest.mark.parametrize(
    "call",
    [
        lambda t: t.nn.GroupNorm(3, 4),
        lambda t: t.nn.GroupNorm(0, 4),
        lambda t: t.nn.GroupNorm(2, 4)(t.ones(2, 6, 3)),
        lambda t: t.nn.GroupNorm(2, 4)(t.ones(4)),
        lambda t: t.nn.functional.group_norm(t.ones(2, 5, 3), 2),
        lambda t: t.nn.functional.group_norm(t.ones(4), 2),
        lambda t: t.nn.functional.group_norm(t.ones(2, 4, 3), 0),
        lambda t: t.nn.functional.group_norm(t.ones(2, 4, 3), 2.0),
        lambda t: t.nn.functional.group_norm(t.ones(2, 4, 3), 2, t.ones(3)),
        lambda t: t.nn.functional.group_norm(t.ones(2, 4, 3), 2, None, t.ones(2, 4)),
        lambda t: t.nn.functional.group_norm(t.ones(2, 4, 0), 2),
        lambda t: t.nn.functional.group_norm(t.ones(2, 4, 3, dtype=t.int64), 2),
        lambda t: t.nn.functional.group_norm(t.ones(2, 4, 3), 2, t.ones(4, dtype=t.float64)),
        lambda t: t.nn.functional.group_norm(t.ones(2, 4, 3, dtype=t.float64), 2, t.ones(4)),
    ],
)
def test_group_norm_misuse_fails_like_torch(st, real, call):
    agree(call, st, real)


def test_group_norm_messages_match_torch(st, real):
    def message(t, call):
        with pytest.raises(Exception) as err:
            call(t)
        return str(err.value)

    for call in (
        lambda t: t.nn.GroupNorm(3, 4),
        lambda t: t.nn.GroupNorm(2, 4)(t.ones(2, 6, 3)),
        lambda t: t.nn.functional.group_norm(t.ones(2, 5, 3), 2),
        lambda t: t.nn.functional.group_norm(t.ones(4), 2),
        lambda t: t.nn.functional.group_norm(t.ones(2, 4, 3), 2, None, t.ones(3)),
        lambda t: t.nn.functional.group_norm(t.ones(2, 4, 0), 2),
        lambda t: t.nn.functional.group_norm(t.ones(2, 4, 3, dtype=t.int64), 2),
    ):
        assert message(st, call) == message(real, call)


def test_group_norm_of_an_empty_batch_is_empty(st, real):
    agree(lambda t: t.nn.functional.group_norm(t.ones(0, 4, 3), 2), st, real)
