"""The layers and their functions against real torch: with the real layer's weights copied
in, the outputs, the input gradients and the weight gradients agree, over a range of shapes,
strides and paddings."""

import time

import numpy as np
import pytest

from nn_parity import function_agrees, layer_agrees, twin
from parity import agree, same

# Real torch's own note about padding="same" with an even kernel.
pytestmark = pytest.mark.filterwarnings("ignore:Using padding='same' with even kernel")

RNG = np.random.default_rng(11)
M = RNG.standard_normal((4, 6))
WIDE = np.concatenate([RNG.standard_normal(6) * 30, [0.0, -0.0, 88.0, -88.0, 100.0, -100.0]]).reshape(2, 6)
IMAGES = RNG.standard_normal((2, 3, 7, 6))
CUBE = RNG.standard_normal((2, 3, 5))


def cases(*items):
    return pytest.mark.parametrize("case", items, ids=[item[0] for item in items])


ACTIVATIONS = (
    ("relu", lambda F, x: F.relu(x), WIDE),
    ("leaky_relu", lambda F, x: F.leaky_relu(x, 0.2), WIDE),
    ("leaky_relu default", lambda F, x: F.leaky_relu(x), M),
    ("elu", lambda F, x: F.elu(x), WIDE),
    ("elu alpha", lambda F, x: F.elu(x, alpha=0.3), M),
    ("gelu", lambda F, x: F.gelu(x), WIDE),
    ("gelu tanh", lambda F, x: F.gelu(x, approximate="tanh"), WIDE),
    ("silu", lambda F, x: F.silu(x), WIDE),
    ("softplus", lambda F, x: F.softplus(x), WIDE),
    ("softplus beta threshold", lambda F, x: F.softplus(x, beta=2, threshold=5), WIDE),
    ("logsigmoid", lambda F, x: F.logsigmoid(x), WIDE),
    ("logsigmoid negated", lambda F, x: F.logsigmoid(-x), WIDE),
    ("sigmoid", lambda F, x: F.sigmoid(x), WIDE),
    ("tanh", lambda F, x: F.tanh(x), WIDE),
    ("softmax", lambda F, x: F.softmax(x, dim=1), WIDE),
    ("softmax dim 0", lambda F, x: F.softmax(x, 0), M),
    ("log_softmax", lambda F, x: F.log_softmax(x, dim=-1), WIDE),
    ("log_softmax dtype", lambda F, x: F.log_softmax(x, 1, dtype=None), M),
    ("linear", lambda F, x, w, b: F.linear(x, w, b), M, RNG.standard_normal((3, 6)), RNG.standard_normal(3)),
    ("linear no bias", lambda F, x, w: F.linear(x, w), CUBE, RNG.standard_normal((2, 5))),
)


@cases(*ACTIVATIONS)
def test_functions_match_torch(st, real, case):
    _, fn, *arrays = case
    function_agrees(st, real, fn, *arrays)
    function_agrees(st, real, fn, *[a.astype(np.float32) for a in arrays])


@cases(*ACTIVATIONS)
def test_gradients_with_an_infinite_upstream_gradient(st, real, case):
    # A gradient that multiplies by a 0 mask would turn inf into nan where torch gives 0.
    _, fn, *arrays = case
    function_agrees(st, real, fn, *arrays, infinite=True)


@pytest.mark.parametrize(
    "make",
    [
        lambda nn: nn.ReLU(),
        lambda nn: nn.LeakyReLU(0.2),
        lambda nn: nn.ELU(alpha=0.5),
        lambda nn: nn.GELU(),
        lambda nn: nn.GELU(approximate="tanh"),
        lambda nn: nn.SiLU(),
        lambda nn: nn.Tanh(),
        lambda nn: nn.Sigmoid(),
        lambda nn: nn.Softmax(dim=-1),
    ],
)
def test_activation_layers_match_torch(st, real, make):
    layer_agrees(st, real, make, WIDE)


def test_softmax_without_dim_warns_and_picks_torchs_dim(st, real):
    def run(t):
        with pytest.warns(UserWarning, match="Implicit dimension choice for softmax has been deprecated"):
            two = t.nn.functional.softmax(t.tensor(M))
        with pytest.warns(UserWarning):
            three = t.nn.Softmax()(t.tensor(CUBE))
        return two, three

    agree(run, st, real)


@pytest.mark.parametrize(
    "name, call",
    [
        ("relu", lambda F, x: F.relu(x, inplace=True)),
        ("leaky_relu", lambda F, x: F.leaky_relu(x, 0.1, inplace=True)),
        ("elu", lambda F, x: F.elu(x, inplace=True)),
        ("silu", lambda F, x: F.silu(x, inplace=True)),
        ("ReLU layer", lambda F, x: F.relu(x, True)),
    ],
)
def test_inplace_activations_change_their_input(st, real, name, call):
    def run(t):
        leaf = t.tensor(M, requires_grad=True)
        y = leaf * 2
        out = call(t.nn.functional, y)
        (out * t.tensor(M)).sum().backward()
        return out is y, y.detach(), leaf.grad

    agree(run, st, real)
    # On a leaf that requires grad it fails, as in torch.
    agree(lambda t: call(t.nn.functional, t.tensor(M, requires_grad=True)), st, real)


def test_gelu_rejects_an_unknown_approximation(st, real):
    agree(lambda t: t.nn.functional.gelu(t.tensor(M), approximate="fast"), st, real)


def test_second_derivatives_of_activations(st, real):
    def run(t):
        F = t.nn.functional
        x = t.tensor(M, requires_grad=True)
        y = F.gelu(x) + F.softplus(x) + F.logsigmoid(x) + F.elu(x) + F.silu(x) + F.leaky_relu(x, 0.3)
        (g,) = t.autograd.grad(y.sum(), x, create_graph=True)
        (h,) = t.autograd.grad((g * g).sum(), x)
        return g.detach(), h

    agree(run, st, real)


# Convolution, pooling and padding.

KERNEL = RNG.standard_normal((4, 3, 3, 3))
WIDE_KERNEL = RNG.standard_normal((2, 3, 2, 4))
TRANSPOSED = RNG.standard_normal((3, 2, 3, 3))
BIAS4, BIAS2 = RNG.standard_normal(4), RNG.standard_normal(2)
TIED = np.round(RNG.standard_normal((2, 2, 6, 5)))  # whole numbers: many ties inside windows
SMALL = RNG.standard_normal((1, 2, 5, 5))

CONV = (
    ("plain", lambda F, x, w, b: F.conv2d(x, w, b), IMAGES, KERNEL, BIAS4),
    ("padding 1", lambda F, x, w, b: F.conv2d(x, w, b, padding=1), IMAGES, KERNEL, BIAS4),
    ("stride 2", lambda F, x, w, b: F.conv2d(x, w, b, stride=2), IMAGES, KERNEL, BIAS4),
    ("stride 2 padding 1", lambda F, x, w, b: F.conv2d(x, w, b, stride=2, padding=1), IMAGES, KERNEL, BIAS4),
    ("pairs", lambda F, x, w, b: F.conv2d(x, w, b, stride=(2, 1), padding=(1, 2)), IMAGES, KERNEL, BIAS4),
    ("dilation 2", lambda F, x, w, b: F.conv2d(x, w, b, dilation=2, padding=1), IMAGES, KERNEL, BIAS4),
    ("dilation pair", lambda F, x, w, b: F.conv2d(x, w, b, dilation=(1, 2), stride=(1, 2)), IMAGES, KERNEL, BIAS4),
    ("same", lambda F, x, w, b: F.conv2d(x, w, b, padding="same"), IMAGES, KERNEL, BIAS4),
    ("same even kernel", lambda F, x, w, b: F.conv2d(x, w, b, padding="same"), IMAGES, WIDE_KERNEL, BIAS2),
    ("same dilated", lambda F, x, w, b: F.conv2d(x, w, b, padding="same", dilation=2), IMAGES, WIDE_KERNEL, BIAS2),
    ("valid", lambda F, x, w, b: F.conv2d(x, w, b, padding="valid"), IMAGES, WIDE_KERNEL, BIAS2),
    ("no bias", lambda F, x, w: F.conv2d(x, w, None, 1, 1), IMAGES, KERNEL),
    ("1x1", lambda F, x, w: F.conv2d(x, w), IMAGES, KERNEL[:, :, :1, :1].copy()),
    ("wide kernel stride 3", lambda F, x, w, b: F.conv2d(x, w, b, 3, 2), IMAGES, WIDE_KERNEL, BIAS2),
    ("unbatched", lambda F, x, w, b: F.conv2d(x, w, b, padding=1), IMAGES[0], KERNEL, BIAS4),
    ("transposed 2 2", lambda F, x, w, b: F.conv_transpose2d(x, w, b, stride=2), IMAGES, TRANSPOSED[:, :, :2, :2].copy(), BIAS2),
    ("transposed 3 2 1 1", lambda F, x, w, b: F.conv_transpose2d(x, w, b, 2, 1, 1), IMAGES, TRANSPOSED, BIAS2),
    ("transposed same size", lambda F, x, w, b: F.conv_transpose2d(x, w, b, padding=1), IMAGES, TRANSPOSED, BIAS2),
    ("transposed pairs", lambda F, x, w: F.conv_transpose2d(x, w, stride=(2, 1), padding=(0, 1)), IMAGES, TRANSPOSED),
    ("transposed dilated", lambda F, x, w: F.conv_transpose2d(x, w, dilation=2, stride=2), IMAGES, TRANSPOSED),
    ("transposed padding past stride", lambda F, x, w: F.conv_transpose2d(x, w, output_padding=1, dilation=2), IMAGES, TRANSPOSED),
    ("transposed unbatched", lambda F, x, w: F.conv_transpose2d(x, w, stride=2), IMAGES[1], TRANSPOSED),
    ("unfold", lambda F, x: F.unfold(x, 3, padding=1, stride=2), IMAGES),
    ("unfold dilated", lambda F, x: F.unfold(x, (2, 3), dilation=(2, 1)), IMAGES),
    ("fold", lambda F, x: F.fold(x, (7, 6), 3, padding=1, stride=2), RNG.standard_normal((2, 27, 12))),
    ("max_pool 2", lambda F, x: F.max_pool2d(x, 2), IMAGES),
    ("max_pool resnet", lambda F, x: F.max_pool2d(x, kernel_size=3, stride=2, padding=1), IMAGES),
    ("max_pool pairs", lambda F, x: F.max_pool2d(x, (2, 3), (1, 2)), IMAGES),
    ("max_pool dilated", lambda F, x: F.max_pool2d(x, 2, 1, 1, 2), IMAGES),
    ("max_pool ties", lambda F, x: F.max_pool2d(x, 3, 1), TIED),
    ("max_pool ties padded", lambda F, x: F.max_pool2d(x, 2, padding=1), TIED),
    ("max_pool unbatched", lambda F, x: F.max_pool2d(x, 2), IMAGES[0]),
    ("max_pool indices", lambda F, x: F.max_pool2d(x, 3, 2, 1, return_indices=True), TIED),
    ("avg_pool 2", lambda F, x: F.avg_pool2d(x, 2), IMAGES),
    ("avg_pool padded", lambda F, x: F.avg_pool2d(x, 3, 2, 1), IMAGES),
    ("avg_pool padding not counted", lambda F, x: F.avg_pool2d(x, 3, 2, 1, count_include_pad=False), IMAGES),
    ("avg_pool divisor", lambda F, x: F.avg_pool2d(x, (2, 3), divisor_override=4), IMAGES),
    ("avg_pool unbatched", lambda F, x: F.avg_pool2d(x, 2, 1), IMAGES[0]),
    ("adaptive 1", lambda F, x: F.adaptive_avg_pool2d(x, 1), IMAGES),
    ("adaptive 1x1", lambda F, x: F.adaptive_avg_pool2d(x, (1, 1)), IMAGES),
    ("adaptive uneven", lambda F, x: F.adaptive_avg_pool2d(x, (3, 4)), IMAGES),
    ("adaptive larger", lambda F, x: F.adaptive_avg_pool2d(x, (9, 8)), IMAGES),
    ("adaptive None", lambda F, x: F.adaptive_avg_pool2d(x, (None, 2)), IMAGES),
    ("adaptive unbatched", lambda F, x: F.adaptive_avg_pool2d(x, 2), IMAGES[0]),
    ("pad", lambda F, x: F.pad(x, (1, 1)), IMAGES),
    ("pad four", lambda F, x: F.pad(x, (1, 2, 3, 0), value=2.5), IMAGES),
    ("pad negative", lambda F, x: F.pad(x, (-1, 2, 0, -2)), IMAGES),
    ("pad every dim", lambda F, x: F.pad(x, (1, 0, 0, 1, 2, 2)), CUBE),
    ("pad matrix", lambda F, x: F.pad(x, [0, 3]), M),
    ("pad reflect", lambda F, x: F.pad(x, (2, 1, 1, 2), mode="reflect"), IMAGES),
    ("pad replicate", lambda F, x: F.pad(x, (3, 1, 0, 2), mode="replicate"), IMAGES),
    ("pad circular", lambda F, x: F.pad(x, (2, 1, 1, 3), mode="circular"), IMAGES),
    ("pad reflect 1d", lambda F, x: F.pad(x, (2, 3), "reflect"), CUBE),
)


@cases(*CONV)
def test_convolution_pooling_and_padding_match_torch(st, real, case):
    _, fn, *arrays = case
    function_agrees(st, real, fn, *arrays)
    function_agrees(st, real, fn, *[a.astype(np.float32) for a in arrays])


@cases(*[c for c in CONV if "pool" in c[0] or "pad" in c[0]])
def test_pooling_and_padding_with_an_infinite_upstream_gradient(st, real, case):
    _, fn, *arrays = case
    function_agrees(st, real, fn, *arrays, infinite=True)


def test_second_derivatives_through_convolution_and_pooling(st, real):
    def run(t):
        F = t.nn.functional
        x = t.tensor(IMAGES, requires_grad=True)
        w = t.tensor(KERNEL, requires_grad=True)
        y = F.max_pool2d(F.conv2d(x, w, padding=1).tanh(), 2) + F.avg_pool2d(F.conv2d(x, w), 2, 1).sum()
        gx, gw = t.autograd.grad((y * y).sum(), (x, w), create_graph=True)
        hx, hw = t.autograd.grad((gx * gx).sum() + (gw**2).sum(), (x, w))
        return gx.detach(), gw.detach(), hx, hw

    agree(run, st, real)


@pytest.mark.parametrize(
    "call",
    [
        lambda t, F: F.conv2d(t.ones(1, 2, 5, 5), t.ones(3, 1, 3, 3)),
        lambda t, F: F.conv2d(t.ones(1, 1, 2, 2), t.ones(2, 1, 3, 3)),
        lambda t, F: F.conv2d(t.ones(1, 1, 5, 5, dtype=t.float64), t.ones(2, 1, 3, 3)),
        lambda t, F: F.conv2d(t.ones(1, 1, 5, 5, dtype=t.float64), t.ones(2, 1, 3, 3, dtype=t.float64), t.ones(2)),
        lambda t, F: F.conv2d(t.ones(1, 1, 5, 5), t.ones(2, 1, 3, 3), stride=2, padding="same"),
        lambda t, F: F.conv_transpose2d(t.ones(1, 2, 3, 3), t.ones(3, 1, 2, 2)),
        lambda t, F: F.conv_transpose2d(t.ones(1, 1, 3, 3), t.ones(1, 1, 2, 2), output_padding=1),
        lambda t, F: F.max_pool2d(t.ones(1, 1, 4, 4), 2, padding=2),
        lambda t, F: F.max_pool2d(t.ones(1, 1, 1, 1), 2),
        lambda t, F: F.avg_pool2d(t.ones(1, 1, 4, 4), 2, padding=2),
        lambda t, F: F.pad(t.ones(2, 3), (1,)),
        lambda t, F: F.pad(t.ones(1, 2, 3), (3, 0), mode="reflect"),
        lambda t, F: F.pad(t.ones(1, 2, 3), (1, 1), mode="nearest"),
    ],
)
def test_bad_arguments_fail_like_torch(st, real, call):
    agree(lambda t: call(t, t.nn.functional), st, real)


CONV_LAYERS = (
    ("conv", lambda nn: nn.Conv2d(3, 4, 3), IMAGES),
    ("conv padding", lambda nn: nn.Conv2d(3, 4, 3, padding=1), IMAGES),
    ("conv stride", lambda nn: nn.Conv2d(3, 2, kernel_size=5, stride=2, padding=2), IMAGES),
    ("conv no bias", lambda nn: nn.Conv2d(3, 4, (3, 2), 1, (1, 0), bias=False), IMAGES),
    ("conv same", lambda nn: nn.Conv2d(3, 2, 4, padding="same"), IMAGES),
    ("conv reflect", lambda nn: nn.Conv2d(3, 2, 3, padding=1, padding_mode="reflect"), IMAGES),
    ("conv replicate", lambda nn: nn.Conv2d(3, 2, 3, padding=(2, 1), padding_mode="replicate"), IMAGES),
    ("conv circular", lambda nn: nn.Conv2d(3, 2, 3, stride=2, padding=1, padding_mode="circular"), IMAGES),
    ("conv same reflect", lambda nn: nn.Conv2d(3, 2, 2, padding="same", padding_mode="reflect"), IMAGES),
    ("conv unbatched", lambda nn: nn.Conv2d(3, 2, 3, padding=1), IMAGES[0]),
    ("transposed", lambda nn: nn.ConvTranspose2d(3, 2, kernel_size=2, stride=2), IMAGES),
    ("transposed padded", lambda nn: nn.ConvTranspose2d(3, 2, 3, stride=2, padding=1, output_padding=1), IMAGES),
    ("transposed no bias", lambda nn: nn.ConvTranspose2d(3, 4, (2, 3), bias=False, dilation=(1, 2)), IMAGES),
    ("max pool", lambda nn: nn.MaxPool2d(2), IMAGES),
    ("max pool resnet", lambda nn: nn.MaxPool2d(kernel_size=3, stride=2, padding=1), IMAGES),
    ("avg pool", lambda nn: nn.AvgPool2d(2), IMAGES),
    ("avg pool stride", lambda nn: nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), IMAGES),
    ("adaptive", lambda nn: nn.AdaptiveAvgPool2d((1, 1)), IMAGES),
    ("adaptive uneven", lambda nn: nn.AdaptiveAvgPool2d((2, 4)), IMAGES),
)


@cases(*CONV_LAYERS)
def test_convolution_and_pooling_layers_match_torch(st, real, case):
    _, make, x = case
    layer_agrees(st, real, make, x)
    layer_agrees(st, real, make, x.astype(np.float32), double=False)


@cases(*CONV_LAYERS)
def test_convolution_and_pooling_reprs_and_parameters_match_torch(st, real, case):
    _, make, _ = case
    s, r = make(st.nn), make(real.nn)
    assert repr(s) == repr(r)
    assert [(n, tuple(p.shape)) for n, p in s.named_parameters()] == [(n, tuple(p.shape)) for n, p in r.named_parameters()]
    names = ("in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "output_padding")
    for name in names + ("padding_mode", "ceil_mode", "return_indices", "output_size", "count_include_pad"):
        if hasattr(r, name):
            assert getattr(s, name) == getattr(r, name), name


def test_more_layer_reprs(st, real):
    for make in (
        lambda nn: nn.Conv2d(2, 4, (3, 5), stride=(2, 1), dilation=2, padding_mode="reflect"),
        lambda nn: nn.Conv2d(3, 64, 7, 2, 3, bias=False),
        lambda nn: nn.Conv2d(2, 4, 3, padding="valid"),
        lambda nn: nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding=1),
        lambda nn: nn.MaxPool2d(3, 1, 1, ceil_mode=False),
        lambda nn: nn.MaxPool2d((2, 2), return_indices=True),
        lambda nn: nn.AvgPool2d(3, 2, 1, divisor_override=2),
        lambda nn: nn.AdaptiveAvgPool2d(1),
    ):
        assert repr(make(st.nn)) == repr(make(real.nn))


def test_transposed_convolution_with_an_output_size(st, real):
    def run(t):
        t.manual_seed(0)
        layer = t.nn.ConvTranspose2d(3, 2, 3, stride=2, padding=1)
        return layer(t.ones(1, 3, 4, 4), output_size=[8, 8]).shape, layer(t.ones(1, 3, 4, 4)).shape

    agree(run, st, real)


@pytest.mark.parametrize(
    "make",
    [
        lambda nn: nn.Conv2d(4, 4, 3, groups=0),
        lambda nn: nn.Conv2d(3, 4, 3, groups=2),
        lambda nn: nn.Conv2d(3, 4, 3, padding="full"),
        lambda nn: nn.Conv2d(3, 4, 3, stride=2, padding="same"),
        lambda nn: nn.Conv2d(3, 4, 3, padding_mode="mirror"),
        lambda nn: nn.ConvTranspose2d(3, 4, 3, padding_mode="reflect"),
    ],
)
def test_bad_layer_settings_fail_like_torch(st, real, make):
    agree(lambda t: make(t.nn), st, real)


def test_what_the_stand_in_leaves_out_says_so(st):
    message = " is not available in the browser version of PyTorch. Download the notebook or open it in Colab to use it."
    nn, F = st.nn, st.nn.functional
    for call, name in (
        (lambda: nn.Conv2d(4, 4, 3, groups=2), "torch.nn.Conv2d with groups=2"),
        (lambda: F.conv2d(st.ones(1, 4, 5, 5), st.ones(4, 2, 3, 3), groups=2), "torch.nn.functional.conv2d with groups=2"),
        (lambda: F.max_pool2d(st.ones(1, 1, 5, 5), 2, ceil_mode=True), "torch.nn.functional.max_pool2d with ceil_mode=True"),
        (lambda: F.avg_pool2d(st.ones(1, 1, 5, 5), 2, ceil_mode=True), "torch.nn.functional.avg_pool2d with ceil_mode=True"),
    ):
        with pytest.raises(NotImplementedError) as err:
            call()
        assert str(err.value) == name + message


def lenet(nn):
    return nn.Sequential(
        nn.Conv2d(1, 6, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.ReLU(), nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10),
    )


def test_a_lenet_step_on_a_batch_of_64_mnist_images_is_fast(st):
    st.manual_seed(0)
    model = lenet(st.nn)
    x = st.randn(64, 1, 28, 28)
    y = st.randint(0, 10, (64,))
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        loss = st.nn.functional.cross_entropy(model(x), y)
        model.zero_grad()
        loss.backward()
        best = min(best, time.perf_counter() - start)
    assert all(p.grad is not None for p in model.parameters())
    assert best < 0.5, f"forward and backward took {best:.3f} s"


# Normalization, embedding and dropout.

ROWS = RNG.standard_normal((6, 4)) * 3 + 1
SEQUENCE = RNG.standard_normal((5, 4, 3)) * 2 - 1


NORM_LAYERS = (
    ("batch norm 1d", lambda nn: nn.BatchNorm1d(4), ROWS),
    ("batch norm 1d sequence", lambda nn: nn.BatchNorm1d(4, momentum=0.3), SEQUENCE),
    ("batch norm 2d", lambda nn: nn.BatchNorm2d(3), IMAGES),
    ("batch norm cumulative", lambda nn: nn.BatchNorm2d(3, momentum=None), IMAGES),
    ("batch norm not affine", lambda nn: nn.BatchNorm2d(3, affine=False), IMAGES),
    ("batch norm no bias", lambda nn: nn.BatchNorm1d(4, bias=False, eps=0.1), ROWS),
    ("batch norm untracked", lambda nn: nn.BatchNorm2d(3, track_running_stats=False), IMAGES),
    ("layer norm", lambda nn: nn.LayerNorm(6), IMAGES),
    ("layer norm two dims", lambda nn: nn.LayerNorm((7, 6), eps=1e-3), IMAGES),
    ("layer norm plain", lambda nn: nn.LayerNorm([3], elementwise_affine=False), SEQUENCE),
    ("layer norm no bias", lambda nn: nn.LayerNorm(4, bias=False), ROWS),
)


@cases(*NORM_LAYERS)
def test_normalization_layers_match_torch(st, real, case):
    _, make, x = case
    for train in (True, False):
        layer_agrees(st, real, make, x, train=train)
        layer_agrees(st, real, make, x.astype(np.float32), double=False, train=train)


@cases(*NORM_LAYERS)
def test_normalization_reprs_and_state_match_torch(st, real, case):
    _, make, _ = case
    s, r = make(st.nn), make(real.nn)
    assert repr(s) == repr(r)
    assert [(k, tuple(v.shape), str(v.dtype)) for k, v in s.state_dict().items()] == [
        (k, tuple(v.shape), str(v.dtype)) for k, v in r.state_dict().items()
    ]
    same(list(s.state_dict().values()), list(r.state_dict().values()))


def test_batch_norm_running_statistics_after_three_training_steps(st, real):
    """Three steps of training (forward, backward, SGD) update the running statistics as torch
    does; eval mode then normalizes with them."""
    def model(nn):
        return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 7 * 6, 2))

    s, r = twin(st, real, model, double=False)
    batches = [RNG.standard_normal((4, 3, 7, 6)).astype(np.float32) * (i + 1) + i for i in range(3)]
    for net, t in ((s, st), (r, real)):
        net.train()
        for batch in batches:
            loss = (net(t.tensor(batch)) ** 2).mean()
            net.zero_grad()
            loss.backward()
            with t.no_grad():
                for p in net.parameters():
                    p -= 0.1 * p.grad
    for name in ("running_mean", "running_var", "num_batches_tracked"):
        same(getattr(s[1], name), getattr(r[1], name))
    assert s[1].num_batches_tracked.item() == 3
    s.eval(), r.eval()
    x = RNG.standard_normal((2, 3, 7, 6)).astype(np.float32)
    same(s(st.tensor(x)).detach(), r(real.tensor(x)).detach())
    before = s[1].running_mean.clone()
    s(st.tensor(x))
    assert st.equal(before, s[1].running_mean), "eval mode must not touch the running statistics"


@pytest.mark.parametrize(
    "fn",
    [
        lambda t, F, x: F.batch_norm(x, None, None, training=True),
        lambda t, F, x: F.batch_norm(x, t.zeros(3, dtype=x.dtype), t.ones(3, dtype=x.dtype), training=False),
        lambda t, F, x: F.batch_norm(x, None, None, t.full((3,), 2.0, dtype=x.dtype), t.ones(3, dtype=x.dtype), True, 0.1, 1e-3),
        lambda t, F, x: F.layer_norm(x, (6,)),
        lambda t, F, x: F.layer_norm(x, [7, 6], t.ones(7, 6, dtype=x.dtype) * 3, None, 1e-2),
    ],
)
def test_normalization_functions_match_torch(st, real, fn):
    function_agrees(st, real, lambda F, x: fn(st if F is st.nn.functional else real, F, x), IMAGES)


def test_batch_norm_updates_the_running_statistics_it_is_given(st, real):
    def run(t):
        mean, var = t.zeros(3, dtype=t.float64), t.ones(3, dtype=t.float64)
        t.nn.functional.batch_norm(t.tensor(IMAGES), mean, var, training=True, momentum=0.25)
        return mean, var, mean._version

    agree(run, st, real)


@pytest.mark.parametrize(
    "call",
    [
        lambda t: t.nn.BatchNorm1d(3)(t.ones(1, 3)),
        lambda t: t.nn.BatchNorm1d(3)(t.ones(2, 3, 4, 5)),
        lambda t: t.nn.BatchNorm2d(3)(t.ones(2, 3)),
        lambda t: t.nn.LayerNorm(4)(t.ones(2, 3)),
        lambda t: t.nn.LayerNorm((2, 3))(t.ones(2, 3, 2)),
        lambda t: t.nn.functional.batch_norm(t.ones(2, 3), None, None),
    ],
)
def test_normalization_misuse_fails_like_torch(st, real, call):
    agree(call, st, real)


EMBEDDINGS = (
    ("plain", lambda nn: nn.Embedding(10, 4), np.array([[1, 2, 2], [9, 0, 1]])),
    ("vector", lambda nn: nn.Embedding(5, 3), np.array([4, 4, 4, 0])),
    ("scalar", lambda nn: nn.Embedding(5, 3), np.array(2)),
    ("padding", lambda nn: nn.Embedding(6, 2, padding_idx=0), np.array([[0, 1, 0], [5, 0, 2]])),
    ("negative padding", lambda nn: nn.Embedding(6, 2, padding_idx=-1), np.array([5, 1, 5, 3])),
)


@cases(*EMBEDDINGS)
def test_embedding_matches_torch(st, real, case):
    _, make, idx = case
    layer_agrees(st, real, make, idx)
    layer_agrees(st, real, make, idx, double=False)
    s, r = make(st.nn), make(real.nn)
    assert repr(s) == repr(r) and s.padding_idx == r.padding_idx
    if r.padding_idx is not None:
        assert float(s.weight[r.padding_idx].abs().sum()) == 0.0  # the padding row starts at zero


def test_embedding_from_pretrained_and_the_function(st, real):
    table = RNG.standard_normal((5, 3))

    def run(t):
        frozen = t.nn.Embedding.from_pretrained(t.tensor(table))
        free = t.nn.Embedding.from_pretrained(t.tensor(table), freeze=False, padding_idx=1)
        idx = t.tensor([[0, 1], [4, 1]])
        out = free(idx)
        (out * t.tensor(table[:2, :2].reshape(2, 2, 1).repeat(3, 2))).sum().backward()
        return frozen(idx), frozen.weight.requires_grad, out.detach(), free.weight.grad, repr(free), t.nn.functional.embedding(idx, t.tensor(table))

    agree(run, st, real)


@pytest.mark.parametrize(
    "call",
    [
        lambda t: t.nn.Embedding(4, 2)(t.tensor([4])),
        lambda t: t.nn.Embedding(4, 2)(t.tensor([-1])),
        lambda t: t.nn.Embedding(4, 2)(t.tensor([1.0])),
        lambda t: t.nn.Embedding(4, 2, padding_idx=4),
    ],
)
def test_embedding_misuse_fails_like_torch(st, real, call):
    agree(call, st, real)


def test_dropout_in_eval_is_the_identity(st, real):
    def run(t):
        x = t.tensor(M)
        layer = t.nn.Dropout(0.5).eval()
        return layer(x) is x, t.nn.functional.dropout(x, 0.3, training=False) is x, t.nn.functional.dropout(x, 0.0) is x

    agree(run, st, real)


@pytest.mark.parametrize("p", [0.1, 0.5, 0.8])
def test_dropout_in_training_zeros_a_fraction_and_scales_the_rest(st, p):
    st.manual_seed(0)
    x = st.ones(200, 500, requires_grad=True)
    layer = st.nn.Dropout(p)
    out = layer(x)
    values = out.detach().numpy()
    kept = values != 0
    assert abs(1 - kept.mean() - p) < 0.01
    np.testing.assert_allclose(values[kept], 1 / (1 - p), rtol=1e-6)
    out.sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), np.where(kept, 1 / (1 - p), 0), rtol=1e-6)
    assert type(out.grad_fn).__name__ == "MulBackward0"
    st.manual_seed(0)
    assert st.equal(st.nn.Dropout(p)(st.ones(200, 500)), out.detach()), "manual_seed repeats the mask"


def test_dropout_edges_and_misuse(st, real):
    def run(t):
        F = t.nn.functional
        x = t.tensor(M, requires_grad=True)
        y = x * 1
        F.dropout(y, 0.5, inplace=True)
        dropped, doubled = y.detach() == 0, (y.detach() - 2 * x.detach()).abs() < 1e-12
        flags = [bool((dropped | doubled).all()), type(y.grad_fn).__name__]
        return F.dropout(x, 1.0).detach(), flags, repr(t.nn.Dropout(0.25)), repr(t.nn.Dropout(inplace=True))

    agree(run, st, real)
    agree(lambda t: t.nn.Dropout(1.5), st, real)
    agree(lambda t: t.nn.functional.dropout(t.ones(2), -0.1), st, real)


# Max pooling picks the entry torch picks: the first maximum, the last nan, and for a window
# whose values are all -inf its first position inside the input, never the padding.

INF, NAN = np.inf, np.nan
CORNER = np.array([[[[-INF, -INF, 1.0], [-INF, -INF, 2.0], [3.0, 4.0, 5.0]]]])
NANS = np.array([[[[1.0, NAN, 3.0], [NAN, 2.0, 0.0], [4.0, 5.0, NAN]]]])
LONE = np.full((1, 1, 4, 4), -INF)
LONE[0, 0, 3, 3] = 1.0
SPECIALS = RNG.standard_normal((2, 3, 6, 7))
SPECIALS[RNG.random(SPECIALS.shape) < 0.3] = -INF
SPECIALS[RNG.random(SPECIALS.shape) < 0.1] = NAN
SPECIALS[RNG.random(SPECIALS.shape) < 0.05] = INF

MAX_POOL_SPECIALS = (
    ("all -inf corner with padding", (2, 1, 1), {}, CORNER),
    ("nan windows", (2, 1), {}, NANS),
    ("nan windows padded", (2, 1, 1), {}, NANS),
    ("one finite value", (3, 1, 1), {}, LONE),
    ("one finite value dilated", (2, 1, 1), dict(dilation=2), LONE),
    ("mixed specials", (3, 2, 1), {}, SPECIALS),
    ("mixed specials dilated", ((2, 3), (1, 2), (1, 1), (2, 1)), {}, SPECIALS),
    ("mixed specials unbatched", (), dict(kernel_size=3, stride=2, padding=1), SPECIALS[0]),
)


@cases(*MAX_POOL_SPECIALS)
def test_max_pool_with_infinities_and_nans_matches_torch(st, real, case):
    _, args, kwargs, x = case
    fn = lambda F, x: F.max_pool2d(x, *args, **kwargs)
    function_agrees(st, real, fn, x)
    function_agrees(st, real, fn, x.astype(np.float32))
    function_agrees(st, real, fn, x, infinite=True)


@cases(*MAX_POOL_SPECIALS)
def test_max_pool_indices_match_torch(st, real, case):
    _, args, kwargs, x = case
    agree(lambda t: t.nn.functional.max_pool2d(t.tensor(x), *args, return_indices=True, **kwargs), st, real)


def test_the_reviewers_corner_probe(st, real):
    def run(t):
        x = t.tensor(CORNER, requires_grad=True)
        out, index = t.nn.functional.max_pool2d(x, 2, 1, 1, return_indices=True)
        out.sum().backward()
        return out.detach(), index, x.grad

    agree(run, st, real)


@pytest.mark.parametrize("dtype", ["int64", "uint8", "int32"])
def test_max_pool_of_integers_matches_torch(st, real, dtype):
    x = np.array([[[[-3, -1, 5], [-2, -4, 0], [1, 2, 3]]]]).astype(dtype)
    agree(lambda t: t.nn.functional.max_pool2d(t.tensor(x), 2, 1, 1, return_indices=True), st, real)


# GELU's erf: a vectorised double precision erf, checked against math.erf and real torch.

ERF_INPUTS = np.concatenate(
    [RNG.standard_normal(2000) * scale for scale in (1e-300, 1e-30, 1e-8, 0.1, 0.5, 1, 2, 3, 5, 10, 40)]
    + [np.array([0.0, -0.0, 0.46875, -0.46875, 0.6629, 4.0, -4.0, 5.657, 26.5, 37.0, 1e300, -1e300, INF, -INF, NAN, 5e-324])]
)


def test_erf_is_accurate_to_double_precision(st):
    import math

    got = st.nn.functional._erf_values(ERF_INPUTS)
    want = np.array([math.erf(v) for v in ERF_INPUTS])
    np.testing.assert_allclose(got, want, rtol=2e-15, atol=0)
    assert np.signbit(got[ERF_INPUTS == 0]).tolist() == np.signbit(ERF_INPUTS[ERF_INPUTS == 0]).tolist()


@pytest.mark.parametrize("dtype, tolerance", [(np.float64, 1e-12), (np.float32, 1e-6)])
def test_gelu_matches_torch_over_tiny_and_large_inputs(st, real, dtype, tolerance):
    with np.errstate(over="ignore"):
        values = ERF_INPUTS.astype(dtype)  # 1e300 becomes inf in float32
    if dtype == np.float32:
        # torch's vectorised float32 kernel gives nan at +inf (one element alone gives inf),
        # an artefact the stand-in does not copy; the float64 run keeps the infinities.
        values = values[~np.isinf(values)]

    def run(t):
        x = t.tensor(values, requires_grad=True)
        out = t.nn.functional.gelu(x)
        (g,) = t.autograd.grad(out.sum(), x)
        return out.detach().numpy(), g.numpy()

    for got, want in zip(run(st), run(real)):
        np.testing.assert_allclose(got, want, rtol=tolerance, atol=tolerance)


def test_gelu_on_a_million_values_is_fast(st):
    x = st.randn(1_000_000, requires_grad=True)
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        st.nn.functional.gelu(x).sum().backward()
        best = min(best, time.perf_counter() - start)
    assert best < 0.15, f"forward and backward took {best:.3f} s"
