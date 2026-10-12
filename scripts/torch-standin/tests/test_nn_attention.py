"""MultiheadAttention and interpolation (Upsample) against real torch."""

import sys

import numpy as np
import pytest

from nn_parity import function_agrees, layer_agrees
from parity import agree, same

RNG = np.random.default_rng(31)
B, L, S, E, H = 2, 5, 3, 8, 2
X = RNG.standard_normal((B, L, E))
KEYS = RNG.standard_normal((B, S, E))
VALUES = RNG.standard_normal((B, S, E))
CAUSAL = np.triu(np.ones((L, L), dtype=bool), 1)
FLOAT_MASK = RNG.standard_normal((L, S))
PADDING = np.array([[False, False, True], [False, True, True]])
FLOAT_PADDING = np.where(PADDING, -1e9, 0.5)
MASK_3D = RNG.standard_normal((B * H, L, S))


def cases(*items):
    return pytest.mark.parametrize("case", items, ids=[item[0] for item in items])


def _call(**options):
    """Call an attention layer on self-attention (one input) or on (query, key, value)."""

    def call(layer, *inputs):
        t = layer._torch

        def tensor(v):  # a float mask in the inputs' dtype, as attention requires
            return t.tensor(v, dtype=inputs[0].dtype) if v.dtype.kind == "f" else t.tensor(v)

        masks = {k: (tensor(v) if isinstance(v, np.ndarray) else v) for k, v in options.items()}
        if len(inputs) == 1:
            return layer(inputs[0], inputs[0], inputs[0], **masks)
        return layer(*inputs, **masks)

    return call


ATTENTION = (
    ("self", dict(batch_first=True), {}, X),
    ("self sequence first", {}, {}, X.transpose(1, 0, 2).copy()),
    ("causal bool mask, per head", dict(batch_first=True), dict(attn_mask=CAUSAL, average_attn_weights=False), X),
    ("causal hint", dict(batch_first=True), dict(attn_mask=CAUSAL, is_causal=True), X),
    ("causal hint without weights", dict(batch_first=True), dict(attn_mask=CAUSAL, is_causal=True, need_weights=False), X),
    ("no weights", dict(batch_first=True), dict(need_weights=False), X),
    ("cross", dict(batch_first=True), {}, X, KEYS, VALUES),
    ("cross float mask", dict(batch_first=True), dict(attn_mask=FLOAT_MASK), X, KEYS, VALUES),
    ("cross 3d mask", dict(batch_first=True), dict(attn_mask=MASK_3D, average_attn_weights=False), X, KEYS, VALUES),
    ("cross padding", dict(batch_first=True), dict(key_padding_mask=PADDING), X, KEYS, VALUES),
    ("cross float padding", {}, dict(key_padding_mask=FLOAT_PADDING), X.transpose(1, 0, 2).copy(), KEYS.transpose(1, 0, 2).copy(), VALUES.transpose(1, 0, 2).copy()),
    ("both masks", dict(batch_first=True), dict(key_padding_mask=FLOAT_PADDING, attn_mask=FLOAT_MASK, need_weights=False), X, KEYS, VALUES),
    ("no bias", dict(batch_first=True, bias=False), {}, X, KEYS, VALUES),
    ("unbatched", {}, dict(attn_mask=CAUSAL), X[0]),
    ("unbatched cross with padding", {}, dict(key_padding_mask=PADDING[1]), X[0], KEYS[0], VALUES[0]),
    ("dropout in eval", dict(batch_first=True, dropout=0.5), {}, X),
)


@cases(*ATTENTION)
def test_multihead_attention_matches_torch(st, real, case):
    _, settings, options, *arrays = case

    def make(nn):
        layer = nn.MultiheadAttention(E, H, **settings)
        layer._torch = st if nn is st.nn else real
        return layer

    train = "dropout" not in case[0]
    layer_agrees(st, real, make, *arrays, call=_call(**options), train=train)
    layer_agrees(st, real, make, *[a.astype(np.float32) for a in arrays], double=False, call=_call(**options), train=train)


def test_multihead_attention_parameters_state_and_repr_match_torch(st, real):
    for settings in (dict(), dict(bias=False), dict(batch_first=True, dropout=0.1)):
        s, r = st.nn.MultiheadAttention(E, H, **settings), real.nn.MultiheadAttention(E, H, **settings)
        assert repr(s) == repr(r)
        assert [(n, tuple(p.shape)) for n, p in s.named_parameters()] == [(n, tuple(p.shape)) for n, p in r.named_parameters()]
        assert [(k, tuple(v.shape)) for k, v in s.state_dict().items()] == [(k, tuple(v.shape)) for k, v in r.state_dict().items()]
        for name in ("embed_dim", "num_heads", "head_dim", "dropout", "batch_first", "kdim", "vdim", "_qkv_same_embed_dim"):
            assert getattr(s, name) == getattr(r, name), name
        if s.in_proj_bias is not None:
            assert float(s.in_proj_bias.abs().sum()) == 0 and float(s.out_proj.bias.abs().sum()) == 0


def test_the_transformer_notebook_check_passes(st, real):
    """The notebook compares a hand-written attention with nn.MultiheadAttention: copy_ the
    weights in under no_grad, then compare outputs and per-head weights with a causal mask."""

    def run(t):
        rng = np.random.default_rng(3)
        nn = t.nn
        qkv, out = nn.Linear(E, 3 * E), nn.Linear(E, E)
        with t.no_grad():
            for layer in (qkv, out):
                layer.weight.copy_(t.tensor(rng.standard_normal(tuple(layer.weight.shape)), dtype=t.float32))
                layer.bias.copy_(t.tensor(rng.standard_normal(tuple(layer.bias.shape)), dtype=t.float32))
        ref = nn.MultiheadAttention(E, H, batch_first=True)
        with t.no_grad():
            ref.in_proj_weight.copy_(qkv.weight)
            ref.in_proj_bias.copy_(qkv.bias)
            ref.out_proj.weight.copy_(out.weight)
            ref.out_proj.bias.copy_(out.bias)
        x = t.tensor(X, dtype=t.float32)
        mask = t.triu(t.ones(L, L, dtype=t.bool), diagonal=1)
        return tuple(o.detach() for o in ref(x, x, x, attn_mask=mask, average_attn_weights=False))

    agree(run, st, real)


@pytest.mark.parametrize(
    "call",
    [
        lambda t: t.nn.MultiheadAttention(E, 3),
        lambda t: t.nn.MultiheadAttention(E, H)(t.ones(L, B, E), t.ones(S, B, E), t.ones(S, B, E), is_causal=True),
        lambda t: t.nn.MultiheadAttention(E, H)(t.ones(L, B, E), t.ones(S, B, E), t.ones(S, B, E), attn_mask=t.ones(L, L)),
        lambda t: t.nn.MultiheadAttention(E, H)(t.ones(L, B, E), t.ones(S, B, E), t.ones(S, B, E), attn_mask=t.ones(L, S, dtype=t.long)),
    ],
)
def test_attention_misuse_fails_like_torch(st, real, call):
    agree(call, st, real)


def test_attention_options_the_stand_in_leaves_out_say_so(st):
    message = " is not available in the browser version of PyTorch. Download the notebook or open it in Colab to use it."
    for kwargs, name in ((dict(kdim=4), "kdim or vdim"), (dict(add_bias_kv=True), "add_bias_kv"), (dict(add_zero_attn=True), "add_zero_attn")):
        with pytest.raises(NotImplementedError) as err:
            st.nn.MultiheadAttention(E, H, **kwargs)
        assert str(err.value) == f"torch.nn.MultiheadAttention with {name}" + message


# Interpolation.

IMAGE = RNG.standard_normal((2, 3, 5, 4))
STRIP = RNG.standard_normal((2, 3, 6))
ROW_IMAGE = RNG.standard_normal((4, 1, 1, 6))

INTERPOLATE = (
    ("nearest x2", lambda F, x: F.interpolate(x, scale_factor=2), IMAGE),
    ("nearest size", lambda F, x: F.interpolate(x, size=(7, 9)), IMAGE),
    ("nearest int size", lambda F, x: F.interpolate(x, 3), IMAGE),
    ("nearest down", lambda F, x: F.interpolate(x, size=(2, 3)), IMAGE),
    ("nearest scale pair", lambda F, x: F.interpolate(x, scale_factor=(1.5, 0.5)), IMAGE),
    ("nearest recomputed", lambda F, x: F.interpolate(x, scale_factor=2.5, recompute_scale_factor=True), IMAGE),
    ("nearest 1d", lambda F, x: F.interpolate(x, scale_factor=1.7), STRIP),
    ("linear 1d", lambda F, x: F.interpolate(x, size=10, mode="linear"), STRIP),
    ("linear 1d corners", lambda F, x: F.interpolate(x, size=4, mode="linear", align_corners=True), STRIP),
    ("bilinear size", lambda F, x: F.interpolate(x, size=(9, 7), mode="bilinear"), IMAGE),
    ("bilinear down", lambda F, x: F.interpolate(x, size=(3, 2), mode="bilinear", align_corners=False), IMAGE),
    ("bilinear x2", lambda F, x: F.interpolate(x, scale_factor=2, mode="bilinear"), IMAGE),
    ("bilinear scale 1.7", lambda F, x: F.interpolate(x, scale_factor=1.7, mode="bilinear"), IMAGE),
    ("bilinear corners", lambda F, x: F.interpolate(x, size=(9, 7), mode="bilinear", align_corners=True), IMAGE),
    ("bilinear corners to one", lambda F, x: F.interpolate(x, size=(1, 7), mode="bilinear", align_corners=True), IMAGE),
    ("bilinear notebook strip", lambda F, x: F.interpolate(x, size=(1, 16), mode="bilinear", align_corners=False), ROW_IMAGE),
)


@cases(*INTERPOLATE)
def test_interpolate_matches_torch(st, real, case):
    _, fn, x = case
    function_agrees(st, real, fn, x)
    function_agrees(st, real, fn, x.astype(np.float32))
    function_agrees(st, real, fn, x, infinite=True)


UPSAMPLE = (
    ("nearest", lambda nn: nn.Upsample(scale_factor=2)),
    ("bilinear size", lambda nn: nn.Upsample(size=(6, 6), mode="bilinear", align_corners=False)),
    ("bilinear corners", lambda nn: nn.Upsample(size=4, mode="bilinear", align_corners=True)),
    ("scale pair", lambda nn: nn.Upsample(scale_factor=(2, 1.5), mode="nearest")),
)


@cases(*UPSAMPLE)
def test_upsample_matches_torch(st, real, case):
    _, make = case
    layer_agrees(st, real, make, IMAGE)
    assert repr(make(st.nn)) == repr(make(real.nn))


@pytest.mark.parametrize(
    "call",
    [
        lambda F, x: F.interpolate(x, size=4, scale_factor=2),
        lambda F, x: F.interpolate(x),
        lambda F, x: F.interpolate(x, size=(2, 2, 2)),
        lambda F, x: F.interpolate(x, size=4, align_corners=True),
        lambda F, x: F.interpolate(x[0], size=4, mode="bilinear"),
        lambda F, x: F.interpolate(x, size=4, recompute_scale_factor=True),
        lambda F, x: F.interpolate(x, size=(0, 3)),
    ],
)
def test_interpolate_misuse_fails_like_torch(st, real, call):
    agree(lambda t: call(t.nn.functional, t.tensor(IMAGE)), st, real)


def test_interpolate_modes_left_out_say_so(st):
    with pytest.raises(NotImplementedError) as err:
        st.nn.functional.interpolate(st.tensor(IMAGE), size=8, mode="bicubic")
    assert str(err.value).startswith("torch.nn.functional.interpolate with mode='bicubic' is not available")


# A query row that may attend to nothing (left padding under a causal mask): without weights,
# torch's scaled_dot_product_attention gives it zeros, so the output there is the projection's
# bias and the next layer stays finite; with weights torch gives nan, and so does the stand-in.

LEFT_PADDING = np.array([[True, True, False, False, False], [False, False, False, False, False]])


def stacked(need_weights):
    def make(nn):
        class Stack(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = nn.MultiheadAttention(E, H, batch_first=True)
                self.second = nn.MultiheadAttention(E, H, batch_first=True)

            def forward(self, x):
                t = self._torch
                padding = t.tensor(LEFT_PADDING)
                causal = t.triu(t.ones(L, L, dtype=t.bool), diagonal=1)
                options = dict(key_padding_mask=padding, attn_mask=causal, need_weights=need_weights)
                h, first_weights = self.first(x, x, x, **options)
                out, second_weights = self.second(h, h, h, **options)
                return (out,) if not need_weights else (out, first_weights, second_weights)

        stack = Stack()
        stack._torch = sys.modules[nn.__name__.rpartition(".")[0]]  # the torch this nn belongs to
        return stack

    return make


def test_fully_masked_rows_without_weights_match_torch(st, real):
    layer_agrees(st, real, stacked(False), X)
    layer_agrees(st, real, stacked(False), X.astype(np.float32), double=False)

    def run(t):
        layer = t.nn.MultiheadAttention(E, H, batch_first=True)
        with t.no_grad():
            layer.out_proj.bias.copy_(t.arange(E, dtype=t.float32))
        x = t.tensor(X, dtype=t.float32)
        causal = t.triu(t.ones(L, L, dtype=t.bool), diagonal=1)
        out, _ = layer(x, x, x, key_padding_mask=t.tensor(LEFT_PADDING), attn_mask=causal, need_weights=False)
        return out[0, :2].detach(), bool(out.isfinite().all())

    agree(run, st, real)


def test_fully_masked_rows_with_weights_are_nan_like_torch(st, real):
    from nn_parity import twin

    s, r = twin(st, real, stacked(True), double=True)
    got = s(st.tensor(X))
    want = r(real.tensor(X))
    same(tuple(o.detach() for o in got), tuple(o.detach() for o in want))
    assert bool(got[0][0].isnan().all()) and not bool(got[0][1].isnan().any())
