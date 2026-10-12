"""torch.nn.Module, Parameter, the containers and nn.init against real torch: registration,
walking a model, state_dict and load_state_dict, train and eval, dtypes and devices, hooks,
repr, and the not-available message for what the stand-in lacks."""

import copy
import io
import math
import warnings

import numpy as np
import pytest

from nn_parity import to_standin
from parity import agree, same

NOT_AVAILABLE = " is not available in the browser version of PyTorch. Download the notebook or open it in Colab to use it."


def make_net(t):
    """A small model with a buffer, a nested Sequential and a parameter of its own."""
    nn = t.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 4))
            self.head = nn.Linear(4, 2, bias=False)
            self.scale = nn.Parameter(t.ones(1))
            self.register_buffer("steps", t.zeros(1))
            self.note = "plain attribute"

        def forward(self, x):
            return self.head(self.body(x)) * self.scale

    return Net()


def structure(model):
    return (
        [(n, tuple(p.shape), p.requires_grad) for n, p in model.named_parameters()],
        [(n, tuple(b.shape)) for n, b in model.named_buffers()],
        [n for n, _ in model.named_children()],
        [(n, type(m).__name__) for n, m in model.named_modules()],
        [(k, tuple(v.shape)) for k, v in model.state_dict().items()],
        sum(p.numel() for p in model.parameters()),
        [tuple(p.shape) for p in model.parameters(recurse=False)],
        repr(model).replace("standin_torch", "torch"),
    )


def test_a_model_registers_and_walks_like_torch(st, real):
    assert structure(make_net(st)) == structure(make_net(real))


def test_attribute_access_and_deletion(st, real):
    def run(t):
        net = make_net(t)
        found = [type(net.body).__name__, type(net.scale).__name__, net.note, type(net.steps).__name__]
        del net.head
        del net.steps
        net.extra = t.nn.ReLU()
        return found, [n for n, _ in net.named_modules()], list(net.state_dict()), "head" in dir(net)

    agree(run, st, real)


@pytest.mark.parametrize(
    "bad",
    [
        lambda t, net: setattr(net, "scale", t.zeros(1)),  # a plain tensor where a parameter is
        lambda t, net: setattr(net, "body", 3),
        lambda t, net: setattr(net, "steps", [1.0]),
        lambda t, net: net.register_parameter("w", t.zeros(1)),
        lambda t, net: net.register_parameter("a.b", None),
        lambda t, net: net.register_parameter("note", None),
        lambda t, net: net.register_buffer("", t.zeros(1)),
        lambda t, net: net.register_buffer("b", 3),
        lambda t, net: net.add_module("x", 3),
        lambda t, net: net.register_parameter("w", t.nn.Parameter(t.zeros(1)) * 2),
        lambda t, net: net.missing,
        lambda t, net: net.train(1),
        lambda t, net: net.to(t.int64),
        lambda t, net: t.nn.Module()(t.zeros(1)),
        lambda t, net: t.nn.ModuleList([t.nn.ReLU()])(t.zeros(1)),
        lambda t, net: t.nn.Sequential(t.nn.ReLU())[1],
    ],
)
def test_misuse_fails_like_torch(st, real, bad):
    def run(t):
        try:
            bad(t, make_net(t))
        except Exception as err:
            return type(err).__name__, str(err).replace("standin_torch", "torch")
        return "no error"

    assert run(st) == run(real)


def test_assigning_before_init_fails_like_torch(st, real):
    def run(t):
        class Forgot(t.nn.Module):
            def __init__(self):
                self.fc = t.nn.Linear(2, 2)

        with pytest.raises(AttributeError) as err:
            Forgot()
        return str(err.value)

    assert run(st) == run(real)


def test_parameter_is_a_leaf_sharing_memory(st, real):
    def run(t):
        base = t.tensor([1.0, 2.0, 3.0])
        p = t.nn.Parameter(base)
        frozen = t.nn.Parameter(t.tensor([1, 2]), requires_grad=False)
        base[0] = 10.0
        (p * p).sum().backward()
        flags = [type(p).__name__, p.is_leaf, p.requires_grad, isinstance(p, t.Tensor), type(p.detach()).__name__]
        flags += [type(p * 2).__name__, frozen.requires_grad, isinstance(p, t.nn.Parameter), t.nn.Parameter().shape]
        return flags, p.detach(), p.grad, repr(p), repr(frozen)

    agree(run, st, real)


def test_parameter_refuses_integers_and_non_tensors(st, real):
    agree(lambda t: t.nn.Parameter(t.tensor([1, 2])), st, real)
    agree(lambda t: t.nn.Parameter([1.0, 2.0]), st, real)


def test_parameter_survives_deepcopy_and_save(st, tmp_path):
    p = st.nn.Parameter(st.tensor([1.0, 2.0]))
    p.grad = st.tensor([3.0, 4.0])
    clone = copy.deepcopy(p)
    assert type(clone).__name__ == "Parameter" and clone.requires_grad and clone.grad is None
    clone.data[0] = 5.0
    assert p[0].item() == 1.0
    st.save(p, tmp_path / "p.pt")
    loaded = st.load(tmp_path / "p.pt")
    assert type(loaded).__name__ == "Parameter" and loaded.requires_grad and loaded.tolist() == [1.0, 2.0]


def test_state_dict_shares_memory_and_deepcopy_snapshots(st, real):
    def run(t):
        t.manual_seed(0)
        net = make_net(t)
        by_reference, by_copy = net.state_dict(), copy.deepcopy(net.state_dict())
        with t.no_grad():
            net.body[0].bias += 1.0
        return (
            t.equal(by_reference["body.0.bias"], net.body[0].bias),
            t.equal(by_copy["body.0.bias"], net.body[0].bias),
            by_reference["body.0.bias"].requires_grad,
            type(net.state_dict(keep_vars=True)["scale"]).__name__,
        )

    agree(run, st, real)


def test_load_state_dict_copies_the_values(st, real):
    real.manual_seed(0)
    source = make_net(real)
    net = make_net(st)
    loaded = net.load_state_dict(to_standin(st, source.state_dict()))
    assert repr(loaded) == "<All keys matched successfully>" and str(loaded) == repr(loaded)
    x = np.random.default_rng(0).standard_normal((5, 3)).astype(np.float32)
    same(net(st.tensor(x)).detach(), source(real.tensor(x)).detach())
    # A real state_dict saved from the stand-in and read back loads too.
    buffer = io.BytesIO()
    st.save(net.state_dict(), buffer)
    buffer.seek(0)
    again = make_net(st)
    again.load_state_dict(st.load(buffer))
    same(again(st.tensor(x)).detach(), source(real.tensor(x)).detach())


def _edit(state, drop=(), add=(), reshape=()):
    state = dict(state)
    for key in drop:
        del state[key]
    for key in add:
        state[key] = state["head.weight"]
    for key in reshape:
        state[key] = state[key][:1]
    return state


@pytest.mark.parametrize(
    "edit",
    [
        dict(drop=["head.weight", "steps"]),
        dict(add=["head.bias", "extra.weight", "body.9.weight"]),
        dict(reshape=["body.0.weight"]),
        dict(drop=["scale"], add=["head.bias"], reshape=["body.2.bias"]),
    ],
    ids=["missing", "unexpected", "size mismatch", "all three"],
)
def test_load_state_dict_errors_match_torch(st, real, edit):
    def run(t, strict):
        net = make_net(t)
        state = _edit(net.state_dict(), **edit)
        try:
            return repr(net.load_state_dict(state, strict=strict))
        except RuntimeError as err:
            return "RuntimeError: " + str(err)

    for strict in (True, False):
        assert run(st, strict) == run(real, strict)


def test_train_eval_requires_grad_and_zero_grad(st, real):
    def run(t):
        net = make_net(t)
        modes = [m.training for m in net.eval().modules()]
        net.body.train()
        modes += [m.training for m in net.modules()]
        net(t.ones(2, 3)).sum().backward()
        had = [p.grad is not None for p in net.parameters()]
        net.zero_grad(set_to_none=False)
        zeros = [float(p.grad.abs().sum()) for p in net.parameters()]
        net.zero_grad()
        cleared = [p.grad is None for p in net.parameters()]
        net.body.requires_grad_(False)
        flags = [p.requires_grad for p in net.parameters()]
        return modes, had, zeros, cleared, flags, net.train() is net

    agree(run, st, real)


def test_apply_visits_children_first(st, real):
    def run(t):
        order = []
        make_net(t).apply(lambda m: order.append(type(m).__name__))
        return order

    agree(run, st, real)


def test_shared_modules_and_parameters_are_listed_once(st, real):
    def run(t):
        layer = t.nn.Linear(2, 2)
        model = t.nn.Sequential(layer, t.nn.ReLU(), layer)
        return (
            [n for n, _ in model.named_parameters()],
            [n for n, _ in model.named_modules()],
            [n for n, _ in model.named_parameters(remove_duplicate=False)],
            len(list(model.children())),
            list(model.state_dict()),
        )

    agree(run, st, real)


def test_dtype_and_device_moves(st, real):
    def run(t):
        net = make_net(t)
        weight = net.head.weight
        net(t.ones(2, 3)).sum().backward()
        net.double()
        kinds = [str(p.dtype) for p in net.parameters()] + [str(b.dtype) for b in net.buffers()]
        kinds.append(str(weight.grad.dtype))
        same_object = net.head.weight is weight
        net.to(t.float32)
        kinds += [str(p.dtype) for p in net.parameters()]
        net.to("cpu").to(t.device("cpu")).to("cpu", t.float64).cpu()
        kinds += [str(p.dtype) for p in net.parameters()]
        net.float()
        return kinds, same_object, str(net.to(t.ones(1, dtype=t.float64)).head.weight.dtype)

    agree(run, st, real)


def test_cuda_is_refused_with_the_browser_message(st):
    net = make_net(st)
    message = "CUDA is not available in the browser. Run this notebook in Colab for a GPU."
    for move in (lambda: net.to("cuda"), lambda: net.to(st.device("cuda:0")), net.cuda, lambda: net.to("cuda", st.float64)):
        with pytest.raises(RuntimeError) as err:
            move()
        assert str(err.value) == message
    assert net.to("cpu") is net and net.cpu() is net


def test_buffers_persistent_and_not(st, real):
    def run(t):
        m = t.nn.Module()
        m.register_buffer("kept", t.ones(2))
        m.register_buffer("scratch", t.zeros(3), persistent=False)
        m.register_buffer("unset", None)
        m.kept = t.full((2,), 5.0)
        return list(m.state_dict()), [n for n, _ in m.named_buffers()], m.kept

    agree(run, st, real)


def test_forward_hooks_and_their_handles(st, real):
    def run(t):
        layer = t.nn.Linear(2, 2)
        seen = []
        hook = layer.register_forward_hook(lambda m, args, out: seen.append(tuple(out.shape)))
        doubled = layer.register_forward_hook(lambda m, args, out: out * 0)
        pre = layer.register_forward_pre_hook(lambda m, args: (args[0] * 0,))
        first = layer(t.ones(3, 2))
        doubled.remove()
        pre.remove()
        second = layer(t.ones(1, 2))
        hook.remove()
        layer(t.ones(4, 2))
        return seen, float(first.detach().abs().sum()), second.shape

    agree(run, st, real)


def test_missing_module_methods_give_the_not_available_message(st, real):
    stand_in = st.nn.Linear(2, 2)
    names = [n for n in dir(real.nn.Module) if not n.startswith("_") and n != "T_destination"]
    for name in names:
        try:
            getattr(stand_in, name)
        except NotImplementedError as err:
            assert str(err) == f"torch.nn.Module.{name}" + NOT_AVAILABLE
            assert isinstance(err, AttributeError) and not hasattr(stand_in, name)
    with pytest.raises(AttributeError) as err:
        stand_in.wieght
    assert str(err.value) == "'Linear' object has no attribute 'wieght'"
    assert not isinstance(err.value, NotImplementedError)


@pytest.mark.parametrize("name", ["nn.Conv3d", "nn.functional.ctc_loss", "nn.init.orthogonal_", "nn.utils"])
def test_missing_nn_names_give_the_not_available_message(st, name):
    obj = st
    *parents, last = name.split(".")
    for part in parents:
        obj = getattr(obj, part)
    with pytest.raises(NotImplementedError) as err:
        getattr(obj, last)
    assert str(err.value) == f"torch.{name}" + NOT_AVAILABLE


# Containers.


def test_sequential_indexing_and_editing(st, real):
    def run(t):
        nn = t.nn
        seq = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
        out = [type(seq[-1]).__name__, len(seq), [type(m).__name__ for m in seq[1:]], type(seq[:2]).__name__]
        seq.append(nn.Sigmoid())
        seq.insert(1, nn.Tanh())
        seq[0] = nn.Linear(2, 3, bias=False)
        del seq[2]
        out.append(repr(seq))
        out.append(type(seq.pop(0)).__name__)
        seq.extend([nn.Identity()])
        out.append([n for n, _ in seq.named_children()])
        named = nn.Sequential(__import__("collections").OrderedDict(fc=nn.Linear(1, 1), act=nn.ReLU()))
        out.append([n for n, _ in named.named_children()])
        out.append(repr(named))
        return out

    agree(run, st, real)


def test_module_list_and_dict(st, real):
    def run(t):
        nn = t.nn
        layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])
        layers.append(nn.ReLU())
        layers.insert(0, nn.Tanh())
        layers += [nn.Linear(2, 2)]
        out = [len(layers), type(layers[-2]).__name__, type(layers[1:3]).__name__, repr(layers)]
        del layers[0]
        out.append([n for n, _ in layers.named_children()])
        out.append(repr(layers + nn.ModuleList([nn.ReLU()])))
        heads = nn.ModuleDict({"a": nn.Linear(2, 1), "b": nn.ReLU()})
        heads["c"] = nn.Tanh()
        heads.update([("d", nn.Identity())])
        out += [list(heads), "a" in heads, len(heads), [type(v).__name__ for v in heads.values()], repr(heads)]
        out.append(type(heads.pop("b")).__name__)
        out.append([n for n, _ in heads.named_parameters()])
        out.append(repr(nn.ModuleList()) + repr(nn.Sequential()))
        return out

    agree(run, st, real)


# Linear and the simple layers.


@pytest.mark.parametrize(
    "make, shape",
    [
        (lambda nn: nn.Linear(4, 5), (3, 4)),
        (lambda nn: nn.Linear(4, 5), (2, 3, 4)),
        (lambda nn: nn.Linear(4, 1, bias=False), (6, 4)),
        (lambda nn: nn.Linear(4, 2), (4,)),
        (lambda nn: nn.Identity(54, unused=1), (2, 3)),
        (lambda nn: nn.Flatten(), (2, 3, 4)),
        (lambda nn: nn.Flatten(0), (2, 3)),
        (lambda nn: nn.Flatten(1, 2), (2, 3, 4, 5)),
    ],
)
def test_linear_and_shape_layers_match_torch(st, real, make, shape):
    from nn_parity import layer_agrees

    x = np.random.default_rng(1).standard_normal(shape)
    layer_agrees(st, real, make, x)
    layer_agrees(st, real, make, x.astype(np.float32), double=False)


def test_linear_with_a_dtype(st, real):
    def run(t):
        layer = t.nn.Linear(3, 2, dtype=t.float64)
        return str(layer.weight.dtype), str(layer.bias.dtype), str(layer(t.ones(1, 3, dtype=t.float64)).dtype)

    agree(run, st, real)


def test_linear_with_wrong_input_size_fails_like_torch(st, real):
    agree(lambda t: t.nn.Linear(3, 2)(t.ones(4, 5)), st, real)


@pytest.mark.parametrize(
    "make",
    [
        lambda nn: nn.Linear(3, 4),
        lambda nn: nn.Linear(1, 1, bias=False),
        lambda nn: nn.Identity(),
        lambda nn: nn.Flatten(0, 2),
        lambda nn: nn.ReLU(inplace=True),
        lambda nn: nn.LeakyReLU(0.2),
        lambda nn: nn.LeakyReLU(0.1, inplace=True),
        lambda nn: nn.ELU(),
        lambda nn: nn.ELU(0.5, inplace=True),
        lambda nn: nn.GELU(),
        lambda nn: nn.GELU("tanh"),
        lambda nn: nn.SiLU(),
        lambda nn: nn.SiLU(inplace=True),
        lambda nn: nn.Tanh(),
        lambda nn: nn.Sigmoid(),
        lambda nn: nn.Softmax(dim=1),
        lambda nn: nn.Softmax(),
    ],
)
def test_reprs_match_torch(st, real, make):
    assert repr(make(st.nn)) == repr(make(real.nn))


# nn.init.


def test_default_linear_init_has_torch_scale(st):
    st.manual_seed(0)
    layer = st.nn.Linear(400, 300)
    bound = 1 / math.sqrt(400)
    w, b = layer.weight.detach().numpy(), layer.bias.detach().numpy()
    assert np.abs(w).max() <= bound and np.abs(b).max() <= bound
    assert abs(w.std() - bound / math.sqrt(3)) < 0.01 * bound  # uniform on [-bound, bound]
    st.manual_seed(0)
    assert st.equal(st.nn.Linear(400, 300).weight, layer.weight)  # manual_seed repeats the draw


@pytest.mark.parametrize(
    "init, std",
    [
        (lambda init, w: init.kaiming_normal_(w, nonlinearity="relu"), math.sqrt(2 / 200)),
        (lambda init, w: init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu"), math.sqrt(2 / 300)),
        (lambda init, w: init.kaiming_uniform_(w, nonlinearity="relu"), math.sqrt(2 / 200)),
        (lambda init, w: init.xavier_uniform_(w), math.sqrt(2 / 500)),
        (lambda init, w: init.xavier_normal_(w, gain=2.0), 2 * math.sqrt(2 / 500)),
        (lambda init, w: init.normal_(w, 1.0, 0.5), 0.5),
        (lambda init, w: init.uniform_(w, -3, 3), math.sqrt(3)),
    ],
)
def test_initialisers_draw_with_torch_scale(st, real, init, std):
    for t in (st, real):
        t.manual_seed(0)
        w = t.nn.Parameter(t.empty(300, 200))
        out = init(t.nn.init, w)
        assert out is w and w.requires_grad and w.is_leaf and w.grad_fn is None
        assert abs(float(w.detach().numpy().std()) - std) < 0.03 * std, t.__name__


def test_constant_initialisers(st, real):
    def run(t):
        w = t.nn.Parameter(t.randn(2, 3))
        t.nn.init.zeros_(w)
        a = w.detach().clone()
        t.nn.init.ones_(w)
        b = w.detach().clone()
        t.nn.init.constant_(w, 0.25)
        return a, b, w.detach(), w.requires_grad, w.is_leaf, w._version

    agree(run, st, real)


@pytest.mark.parametrize(
    "call",
    [
        lambda t: t.nn.init.calculate_gain("tanh"),
        lambda t: t.nn.init.calculate_gain("relu"),
        lambda t: t.nn.init.calculate_gain("leaky_relu", 0.2),
        lambda t: t.nn.init.calculate_gain("conv2d"),
        lambda t: t.nn.init.calculate_gain("selu"),
        lambda t: t.nn.init.calculate_gain("swish"),
        lambda t: t.nn.init._calculate_fan_in_and_fan_out(t.zeros(3)),
        lambda t: t.nn.init._calculate_fan_in_and_fan_out(t.zeros(8, 3, 5, 5)),
        lambda t: t.nn.init.kaiming_normal_(t.zeros(3, 3), mode="fan_avg"),
    ],
)
def test_gains_and_fans(st, real, call):
    agree(call, st, real)


def test_init_of_an_empty_tensor_warns(st):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        st.nn.init.kaiming_uniform_(st.empty(0, 3))
    assert "Initializing zero-element tensors is a no-op" in str(caught[0].message)


@pytest.mark.parametrize(
    "make, name",
    [
        (lambda nn: nn.Linear(300, 2000), "weight"),
        (lambda nn: nn.Linear(300, 2000), "bias"),
        (lambda nn: nn.Conv2d(16, 1000, 3), "weight"),
        (lambda nn: nn.Conv2d(16, 1000, 3), "bias"),
        (lambda nn: nn.ConvTranspose2d(16, 1000, 3), "weight"),
        (lambda nn: nn.ConvTranspose2d(16, 1000, 3), "bias"),
        (lambda nn: nn.Embedding(500, 64), "weight"),
        (lambda nn: nn.MultiheadAttention(64, 4), "in_proj_weight"),
        (lambda nn: nn.MultiheadAttention(64, 4), "out_proj.weight"),
        (lambda nn: nn.BatchNorm2d(8), "weight"),
        (lambda nn: nn.LayerNorm(8), "bias"),
    ],
)
def test_default_initialisation_has_torch_scale(st, real, make, name):
    """The draws differ from torch's, but their spread and bound are the same."""

    def stats(t):
        t.manual_seed(0)
        w = dict(make(t.nn).named_parameters())[name].detach().numpy()
        return np.abs(w).max(), w.std(), w.mean()

    (got_max, got_std, got_mean), (want_max, want_std, want_mean) = stats(st), stats(real)
    assert abs(got_std - want_std) <= 0.1 * want_std
    if "Embedding" not in repr(make(st.nn)):  # normal draws have no bound to compare
        assert abs(got_max - want_max) <= 0.1 * want_max
    assert abs(got_mean - want_mean) <= 0.1 * max(want_std, 1e-12) + 1e-6


def _bn_net(t):
    return t.nn.Sequential(t.nn.Linear(2, 3), t.nn.BatchNorm1d(3))


def test_state_dict_metadata_records_versions(st, real):
    def run(t):
        state = _bn_net(t).state_dict()
        snapshot = copy.deepcopy(state)
        buffer = io.BytesIO()
        t.save(state, buffer)
        buffer.seek(0)
        loaded = t.load(buffer)
        return dict(state._metadata), dict(snapshot._metadata), dict(loaded._metadata)

    assert run(st) == run(real)


@pytest.mark.parametrize(
    "edit",
    [
        lambda state: {k: v for k, v in state.items() if "num_batches_tracked" not in k},  # a plain dict
        lambda state: [state.pop("1.num_batches_tracked"), state][1],  # torch's own dict, version 2
        lambda state: {k: v for k, v in state.items() if k != "1.running_mean"},
    ],
    ids=["plain dict without the count", "metadata says version 2", "missing running mean"],
)
def test_batch_norm_loads_a_state_without_num_batches_tracked_like_torch(st, real, edit):
    def run(t):
        source, target = _bn_net(t), _bn_net(t)
        with t.no_grad():
            source[1].num_batches_tracked.fill_(5)
            target[1].num_batches_tracked.fill_(7)
        state = edit(source.state_dict())
        keys = list(state)
        try:
            result = repr(target.load_state_dict(state))
        except RuntimeError as err:
            result = str(err)
        return result, target[1].num_batches_tracked.item(), list(state) == keys

    assert run(st) == run(real)


def test_batch_norm_without_running_statistics_loads_weights_only(st, real):
    def run(t):
        layer = t.nn.BatchNorm1d(3, track_running_stats=False)
        return repr(layer.load_state_dict({"weight": t.ones(3) * 2, "bias": t.zeros(3)})), layer.weight.detach()

    agree(run, st, real)


@pytest.mark.parametrize("path", ["nn", "nn.functional", "nn.init"])
def test_the_nn_modules_offer_only_torchs_names(st, real, path):
    """The package's own operations are not reachable through nn: F.sum is torch's absence."""
    ours, theirs = st, real
    for part in path.split("."):
        ours, theirs = getattr(ours, part), getattr(theirs, part)
    extra = [name for name in dir(ours) if not name.startswith("_") and not hasattr(theirs, name)]
    assert extra == []


@pytest.mark.parametrize("name", ["sum", "pow", "write", "result", "data_of", "constant", "mul", "reshape", "module_getattr"])
def test_internal_helpers_are_not_functional_names(st, name):
    with pytest.raises(NotImplementedError) as err:
        getattr(st.nn.functional, name)
    assert str(err.value) == f"torch.nn.functional.{name}" + NOT_AVAILABLE
