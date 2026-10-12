"""The losses against real torch: values and gradients for every reduction, class weights,
ignore_index, label smoothing and probability targets, and the same errors for misuse."""

import warnings

import numpy as np
import pytest

from nn_parity import function_agrees, layer_agrees
from parity import agree

RNG = np.random.default_rng(21)
PRED = RNG.standard_normal((5, 3))
TARGET = RNG.standard_normal((5, 3))
COLUMN = RNG.standard_normal((5, 1))
WEIGHTS = RNG.random((5, 3)) + 0.1
LOGITS = RNG.standard_normal((6, 4)) * 2
LABELS = np.array([0, 3, 1, 1, 2, 3])
IGNORED = np.array([0, -100, 1, 3, -100, 2])
CLASS_WEIGHT = np.array([0.5, 2.0, 1.0, 3.0])
SOFT = RNG.dirichlet(np.ones(4), 6)
SPATIAL = RNG.standard_normal((2, 4, 3, 2))
SPATIAL_LABELS = RNG.integers(0, 4, (2, 3, 2))
PROBS = np.array([[0.0, 0.2, 0.5], [0.9, 1.0, 1e-50], [0.3, 0.7, 1 - 1e-12]])
BINARY = np.array([[0.0, 1.0, 1.0], [1.0, 1.0, 0.0], [0.25, 0.0, 1.0]])
WIDE_LOGITS = np.array([[-100.0, -5.0, 0.0], [3.0, 40.0, 100.0], [-0.5, 1e-3, -30.0]])
POS_WEIGHT = np.array([0.5, 2.0, 4.0])
BCE_WEIGHT = WEIGHTS[:3].copy()
# Inputs torch does not differentiate (class weights, BCE weights): they never require grad.
NOT_DIFFERENTIABLE = (CLASS_WEIGHT, POS_WEIGHT, BCE_WEIGHT)
REDUCTIONS = ("none", "mean", "sum")


def cases(*items):
    return pytest.mark.parametrize("case", items, ids=[item[0] for item in items])


def F(t):
    return t.nn.functional


LOSSES = (
    ("mse", lambda t, r, x, y: F(t).mse_loss(x, y, reduction=r), PRED, TARGET),
    ("mse weighted", lambda t, r, x, y, w: F(t).mse_loss(x, y, reduction=r, weight=w), PRED, TARGET, WEIGHTS),
    ("l1", lambda t, r, x, y: F(t).l1_loss(x, y, reduction=r), PRED, np.where(TARGET > 0.5, PRED, TARGET)),
    ("l1 weighted", lambda t, r, x, y, w: F(t).l1_loss(x, y, reduction=r, weight=w), PRED, TARGET, WEIGHTS),
    ("huber", lambda t, r, x, y: F(t).huber_loss(x, y, reduction=r), PRED * 2, TARGET),
    ("huber delta", lambda t, r, x, y: F(t).huber_loss(x, y, reduction=r, delta=0.5), PRED, TARGET),
    ("huber weighted", lambda t, r, x, y, w: F(t).huber_loss(x, y, r, 0.3, w), PRED, TARGET, WEIGHTS),
    ("cross entropy", lambda t, r, x: F(t).cross_entropy(x, t.tensor(LABELS), reduction=r), LOGITS),
    ("cross entropy weighted", lambda t, r, x, w: F(t).cross_entropy(x, t.tensor(LABELS), w, reduction=r), LOGITS, CLASS_WEIGHT),
    ("cross entropy ignored", lambda t, r, x: F(t).cross_entropy(x, t.tensor(IGNORED), reduction=r), LOGITS),
    ("cross entropy ignore 2", lambda t, r, x, w: F(t).cross_entropy(x, t.tensor(LABELS), w, ignore_index=3, reduction=r), LOGITS, CLASS_WEIGHT),
    ("cross entropy smoothing", lambda t, r, x: F(t).cross_entropy(x, t.tensor(LABELS), reduction=r, label_smoothing=0.1), LOGITS),
    (
        "cross entropy smoothing weighted ignored",
        lambda t, r, x, w: F(t).cross_entropy(x, t.tensor(IGNORED), w, reduction=r, label_smoothing=0.3),
        LOGITS, CLASS_WEIGHT,
    ),
    ("cross entropy soft", lambda t, r, x, p: F(t).cross_entropy(x, p, reduction=r), LOGITS, SOFT),
    ("cross entropy soft weighted", lambda t, r, x, p, w: F(t).cross_entropy(x, p, w, reduction=r), LOGITS, SOFT, CLASS_WEIGHT),
    ("cross entropy soft smoothing", lambda t, r, x, p: F(t).cross_entropy(x, p, reduction=r, label_smoothing=0.2), LOGITS, SOFT),
    ("cross entropy spatial", lambda t, r, x: F(t).cross_entropy(x, t.tensor(SPATIAL_LABELS), reduction=r), SPATIAL),
    (
        "cross entropy spatial smoothing",
        lambda t, r, x, w: F(t).cross_entropy(x, t.tensor(SPATIAL_LABELS), w, reduction=r, label_smoothing=0.1),
        SPATIAL, CLASS_WEIGHT,
    ),
    ("cross entropy unbatched", lambda t, r, x: F(t).cross_entropy(x, t.tensor(np.array(2)), reduction=r), LOGITS[0]),
    ("cross entropy unbatched soft", lambda t, r, x, p: F(t).cross_entropy(x, p, reduction=r), LOGITS[0], SOFT[0]),
    ("cross entropy byte labels", lambda t, r, x: F(t).cross_entropy(x, t.tensor(LABELS.astype(np.uint8)), reduction=r), LOGITS),
    ("nll", lambda t, r, x: F(t).nll_loss(F(t).log_softmax(x, 1), t.tensor(LABELS), reduction=r), LOGITS),
    ("nll weighted ignored", lambda t, r, x, w: F(t).nll_loss(x, t.tensor(IGNORED), w, reduction=r), LOGITS, CLASS_WEIGHT),
    ("nll spatial", lambda t, r, x: F(t).nll_loss(x, t.tensor(SPATIAL_LABELS), reduction=r, ignore_index=1), SPATIAL),
    ("bce", lambda t, r, p, y: F(t).binary_cross_entropy(p, y, reduction=r), PROBS, BINARY),
    ("bce weighted", lambda t, r, p, y, w: F(t).binary_cross_entropy(p, y, w, reduction=r), PROBS, BINARY, POS_WEIGHT),
    ("bce of a sigmoid", lambda t, r, x, y: F(t).binary_cross_entropy(F(t).sigmoid(x), y, reduction=r), WIDE_LOGITS / 10, BINARY),
    ("bce logits", lambda t, r, x, y: F(t).binary_cross_entropy_with_logits(x, y, reduction=r), WIDE_LOGITS, BINARY),
    (
        "bce logits weighted",
        lambda t, r, x, y, w, pw: F(t).binary_cross_entropy_with_logits(x, y, w, reduction=r, pos_weight=pw),
        WIDE_LOGITS, BINARY, BCE_WEIGHT, POS_WEIGHT,
    ),
)


def _flags(arrays):
    return [a.dtype.kind == "f" and not any(a is w for w in NOT_DIFFERENTIABLE) for a in arrays]


@cases(*LOSSES)
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_losses_match_torch(st, real, case, reduction):
    _, fn, *arrays = case
    run, flags = (lambda t, *xs: fn(t, reduction, *xs)), _flags(arrays)
    function_agrees(st, real, run, *arrays, needs_grad=flags, torch_module=True)
    arrays = [a.astype(np.float32) if a.dtype.kind == "f" else a for a in arrays]
    function_agrees(st, real, run, *arrays, needs_grad=flags, torch_module=True)


@cases(*LOSSES)
def test_unreduced_losses_with_an_infinite_upstream_gradient(st, real, case):
    _, fn, *arrays = case
    run = lambda t, *xs: fn(t, "none", *xs)
    function_agrees(st, real, run, *arrays, infinite=True, needs_grad=_flags(arrays), torch_module=True)


def test_every_label_ignored_gives_nan_like_torch(st, real):
    agree(lambda t: t.nn.functional.cross_entropy(t.tensor(LOGITS), t.full((6,), -100)), st, real)


def test_second_derivatives_of_losses(st, real):
    def run(t):
        F = t.nn.functional
        x = t.tensor(LOGITS, requires_grad=True)
        loss = F(t).cross_entropy(x, t.tensor(LABELS), label_smoothing=0.1)
        loss = loss + F(t).binary_cross_entropy_with_logits(x, t.tensor(SOFT)) + F(t).huber_loss(x, t.tensor(SOFT))
        (g,) = t.autograd.grad(loss, x, create_graph=True)
        (h,) = t.autograd.grad((g * g).sum(), x)
        return g.detach(), h

    agree(run, st, real)


LOSS_LAYERS = (
    ("mse", lambda nn: nn.MSELoss(), PRED, TARGET),
    ("mse sum", lambda nn: nn.MSELoss(reduction="sum"), PRED, TARGET),
    ("l1", lambda nn: nn.L1Loss(reduction="none"), PRED, TARGET),
    ("huber", lambda nn: nn.HuberLoss(delta=0.7), PRED, TARGET),
    ("cross entropy", lambda nn: nn.CrossEntropyLoss(), LOGITS, LABELS),
    ("cross entropy options", lambda nn: nn.CrossEntropyLoss(ignore_index=3, label_smoothing=0.05), LOGITS, LABELS),
    ("nll", lambda nn: nn.NLLLoss(reduction="sum"), LOGITS, LABELS),
    ("bce", lambda nn: nn.BCELoss(), PROBS, BINARY),
    ("bce logits", lambda nn: nn.BCEWithLogitsLoss(), WIDE_LOGITS, BINARY),
)


@cases(*LOSS_LAYERS)
def test_loss_layers_match_torch(st, real, case):
    _, make, *arrays = case
    layer_agrees(st, real, make, *arrays)
    assert repr(make(st.nn)) == repr(make(real.nn))


def test_loss_layers_keep_their_weights_as_buffers(st, real):
    def run(t):
        w = t.tensor(CLASS_WEIGHT)
        layers = [
            t.nn.CrossEntropyLoss(w),
            t.nn.CrossEntropyLoss(weight=w, reduction="sum"),
            t.nn.NLLLoss(w),
            t.nn.BCELoss(weight=t.tensor(POS_WEIGHT)),
            t.nn.BCEWithLogitsLoss(pos_weight=t.tensor(POS_WEIGHT)),
            t.nn.MSELoss(),
        ]
        x = t.tensor(LOGITS, requires_grad=True)
        values = [layers[0](x, t.tensor(LABELS)), layers[1](x, t.tensor(LABELS)), layers[2](x, t.tensor(LABELS))]
        values.append(layers[4](t.tensor(WIDE_LOGITS), t.tensor(BINARY)))
        values.append(layers[3](t.tensor(PROBS), t.tensor(BINARY)))
        return [list(layer.state_dict()) for layer in layers], [v.detach() for v in values], [repr(layer) for layer in layers]

    agree(run, st, real)


def test_the_old_size_average_and_reduce_arguments(st, real):
    def run(t):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            layers = [t.nn.MSELoss(size_average=False), t.nn.L1Loss(reduce=False), t.nn.CrossEntropyLoss(None, True, -100, True)]
        x, y = t.tensor(PRED), t.tensor(TARGET)
        values = [layers[0](x, y), layers[1](x, y), layers[2](t.tensor(LOGITS), t.tensor(LABELS))]
        return [layer.reduction for layer in layers], values, sorted({str(w.message) for w in caught})

    agree(run, st, real)


def test_mse_warns_when_the_shapes_broadcast(st, real):
    def run(t):
        with pytest.warns(UserWarning) as caught:
            loss = t.nn.functional.mse_loss(t.tensor(COLUMN), t.tensor(COLUMN[:, 0]))
        return loss, str(caught[0].message)

    agree(run, st, real)


@pytest.mark.parametrize(
    "call",
    [
        lambda t, F: F(t).mse_loss(t.zeros(3), t.zeros(3), reduction="avg"),
        lambda t, F: F(t).cross_entropy(t.zeros(3, 4), t.tensor([0, 1])),
        lambda t, F: F(t).cross_entropy(t.zeros(3, 4), t.tensor([0, 4, 1])),
        lambda t, F: F(t).cross_entropy(t.zeros(3, 4), t.tensor([0, -2, 1])),
        lambda t, F: F(t).cross_entropy(t.zeros(3, 4), t.tensor([0.0, 2.0, 1.0])),
        lambda t, F: F(t).cross_entropy(t.zeros(3, 4), t.tensor([0, 2, 1], dtype=t.int32)),
        lambda t, F: F(t).cross_entropy(t.zeros(3, 4), t.ones(3, 4, dtype=t.long)),
        lambda t, F: F(t).cross_entropy(t.zeros(3, 4), t.tensor([0, 2, 1]), label_smoothing=1.5),
        lambda t, F: F(t).cross_entropy(t.zeros(3, 4), t.tensor([0, 2, 1]), weight=t.ones(3)),
        lambda t, F: F(t).cross_entropy(t.zeros(3, 4), t.tensor([0, 2, 1]), weight=t.ones(4, dtype=t.float64)),
        lambda t, F: F(t).nll_loss(t.zeros(3, 4, 5), t.zeros(3, 4, dtype=t.long)),
        lambda t, F: F(t).binary_cross_entropy(t.tensor([0.5, 1.5]), t.tensor([0.0, 1.0])),
        lambda t, F: F(t).binary_cross_entropy(t.tensor([[0.5], [0.5]]), t.tensor([0.0, 1.0])),
        lambda t, F: F(t).binary_cross_entropy(t.tensor([0.5, 0.5]), t.tensor([0.0, 1.0], dtype=t.float64)),
        lambda t, F: F(t).binary_cross_entropy_with_logits(t.tensor([[0.5], [0.5]]), t.tensor([0.0, 1.0])),
        lambda t, F: F(t).mse_loss(t.zeros(3), t.zeros(3), weight=t.ones(2)),
    ],
)
def test_loss_misuse_fails_like_torch(st, real, call):
    agree(lambda t: call(t, t.nn.functional), st, real)


@pytest.mark.parametrize(
    "call",
    [
        lambda t, F: F.one_hot(t.tensor([0, 2, 1])),
        lambda t, F: F.one_hot(t.tensor([[0, 2], [1, 1]]), 4),
        lambda t, F: F.one_hot(t.tensor(3), num_classes=5),
        lambda t, F: F.one_hot(t.tensor([0, 3]), 3),
        lambda t, F: F.one_hot(t.tensor([0, -1])),
        lambda t, F: F.one_hot(t.tensor([0.0, 1.0])),
        lambda t, F: F.one_hot(t.tensor([], dtype=t.long), 3),
    ],
)
def test_one_hot_matches_torch(st, real, call):
    agree(lambda t: call(t, t.nn.functional), st, real)
