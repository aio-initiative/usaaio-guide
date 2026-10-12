"""Gradients of every differentiable operation against real torch, including broadcasting,
keepdim and negative dims, ties and zeros, and second derivatives with create_graph."""

import numpy as np
import pytest

from parity import same

RNG = np.random.default_rng(1)
M = RNG.standard_normal((3, 4))
N = RNG.standard_normal((3, 4))
ROW = RNG.standard_normal(4)
COL = RNG.standard_normal((3, 1))
POSM = np.abs(M) + 0.5
CUBE = RNG.standard_normal((2, 3, 4))
SQ = RNG.standard_normal((4, 4))
VEC = RNG.standard_normal(4)
ZEROS = np.array([[0.0, 1.5, -2.0, 0.0], [3.0, 0.0, 1.0, -1.0], [2.0, 2.0, -0.5, 0.25]])
TIES = np.array([[1.0, 5.0, 5.0, 2.0], [3.0, 3.0, 0.0, 3.0], [-1.0, 4.0, 4.0, 4.0]])
INDEX = np.array([[0, 2, 1, 0], [2, 2, 0, 1]])
MASK = M > 0


def _outputs(out):
    """The float tensors in what an operation returned (a tensor, a tuple, a named pair)."""
    items = out if isinstance(out, (tuple, list)) else (out,)
    return [o for o in items if o.dtype.is_floating_point]


def _weights(rng, shape, infinite):
    """Random weights for an output; with `infinite`, every third one is inf."""
    w = rng.standard_normal(shape)
    if infinite:
        w.reshape(-1)[::3] = np.inf
    return w


def _run(t, fn, arrays, infinite=False):
    leaves = [t.tensor(a, requires_grad=a.dtype.kind == "f") for a in arrays]
    outs = _outputs(fn(t, *leaves))
    weights = np.random.default_rng(7)
    total = 0
    for out in outs:
        w = t.tensor(_weights(weights, tuple(out.shape), infinite), dtype=out.dtype)
        total = total + (out * w).sum()
    total.backward()
    values = tuple(out.detach() for out in outs)
    grads = tuple(leaf.grad for leaf in leaves if leaf.requires_grad)
    return values, grads


def grads_agree(st, real, fn, *arrays, infinite=False):
    """fn gives the same values and the same gradients on both libraries. The outputs are
    weighted by random numbers before summing, so every gradient entry is tested. With
    `infinite` some weights are inf, so a gradient that multiplies by a 0 mask shows up as
    nan where torch, which masks with where, gives 0."""
    got, want = _run(st, fn, arrays, infinite), _run(real, fn, arrays, infinite)
    same(got[0], want[0])
    assert len(got[1]) == len(want[1])
    for g, w in zip(got[1], want[1]):
        if w is None:
            assert g is None
        else:
            same(g, w)


def cases(*items):
    return pytest.mark.parametrize("case", items, ids=[item[0] for item in items])


ELEMENTWISE = (
    ("add broadcast row", lambda t, x, y: x + y, M, ROW),
    ("add broadcast column", lambda t, x, y: t.add(x, y), M, COL),
    ("add alpha", lambda t, x, y: t.add(x, y, alpha=-2.5), M, N),
    ("add number", lambda t, x: 3 + x, M),
    ("sub broadcast", lambda t, x, y: x - y, COL, ROW),
    ("rsub", lambda t, x: 1 - x, M),
    ("mul broadcast", lambda t, x, y: x * y, M, ROW),
    ("mul self", lambda t, x: x * x, M),
    ("mul number", lambda t, x: x * 2.5, M),
    ("div", lambda t, x, y: x / y, M, POSM),
    ("div broadcast", lambda t, x, y: x / y, M, COL),
    ("div number", lambda t, x: x / 4, M),
    ("rdiv", lambda t, x: 2 / x, POSM),
    ("pow number", lambda t, x: x**3, M),
    ("pow half", lambda t, x: x**0.5, POSM),
    ("pow zero", lambda t, x: x**0, M),
    ("pow negative", lambda t, x: t.pow(x, -2), POSM),
    ("pow tensor", lambda t, x, y: x**y, POSM, N),
    ("pow zero base", lambda t, x, y: x**y, np.array([0.0, 0.0, 2.0]), np.array([0.0, 2.0, 3.0])),
    ("rpow", lambda t, x: 2**x, M),
    ("neg", lambda t, x: -x, M),
    ("abs with zeros", lambda t, x: x.abs(), ZEROS),
    ("exp", lambda t, x: x.exp(), M),
    ("log", lambda t, x: t.log(x), POSM),
    ("sqrt", lambda t, x: x.sqrt(), POSM),
    ("sin cos", lambda t, x: x.sin() * x.cos(), M),
    ("tanh", lambda t, x: x.tanh(), M),
    ("sigmoid", lambda t, x: t.sigmoid(x * 10), M),
    ("relu at zero", lambda t, x: x.relu(), ZEROS),
    ("torch relu", lambda t, x: t.relu(x), M),
    ("clamp", lambda t, x: x.clamp(-0.5, 0.5), M),
    ("clamp at the bound", lambda t, x: x.clamp(min=0), ZEROS),
    ("clamp max", lambda t, x: t.clamp(x, max=1.0), ZEROS),
    ("clamp tensor bounds", lambda t, x, lo, hi: x.clamp(lo, hi), M, N - 1, N + 1),
    (
        "clamp tensor bounds with ties",
        lambda t, x, lo, hi: x.clamp(lo, hi),
        np.array([0.0, 1.0, 2.0, 3.0, -1.0]),
        np.array([0.0, 0.0, 2.5, 3.0, 0.0]),
        np.array([1.0, 1.0, 3.0, 3.0, 2.0]),
    ),
    ("clamp tensor min only", lambda t, x, lo: t.clamp(x, min=lo), TIES, np.flip(TIES, 1).copy()),
    ("clip", lambda t, x: x.clip(-1, 1), M),
    ("clamp_min", lambda t, x: x.clamp_min(0.2), M),
    ("maximum with ties", lambda t, x, y: t.maximum(x, y), TIES, np.flip(TIES, 1).copy()),
    ("minimum broadcast", lambda t, x, y: t.minimum(x, y), M, ROW),
    ("max of two", lambda t, x, y: t.max(x, y), M, N),
    ("where", lambda t, x, y: t.where(x > 0, x, y), M, N),
    ("where number", lambda t, x: t.where(x > 0, x, 0.0), M),
    ("where method", lambda t, x, y: x.where(x < 0, y), M, ROW),
    ("masked_fill", lambda t, x: x.masked_fill(x > 0.5, -1.0), M),
    ("round", lambda t, x: x.round() + x, M),
    ("remainder", lambda t, x, y: x % y, M * 5, POSM),
    ("float of double", lambda t, x: x.float() * 2, M),
    ("to float64", lambda t, x: x.float().to(t.float64), M),
    ("clone", lambda t, x: x.clone() * x, M),
    ("float32 plus float64", lambda t, x, y: x.float() + y, M, N),
)


@cases(*ELEMENTWISE)
def test_elementwise_gradients(st, real, case):
    grads_agree(st, real, case[1], *case[2:])


INDEXING = (
    ("slice", lambda t, x: x[1:, ::2], M),
    ("int", lambda t, x: x[1], M),
    ("negative int", lambda t, x: x[-1, -2], M),
    ("list with repeats", lambda t, x: x[[0, 2, 0]], M),
    ("tensor index", lambda t, x: x[t.tensor([[0, 1], [1, 1]])], M),
    ("bool mask", lambda t, x: x[x > 0], M),
    ("mask of rows", lambda t, x: x[t.tensor([True, False, True])], M),
    ("mixed int and list", lambda t, x: x[0, :, [1, 3, 1]], CUBE),
    ("two lists", lambda t, x: x[[0, 2], [1, 3]], M),
    ("none and ellipsis", lambda t, x: x[None, ..., 1], CUBE),
    ("iterate", lambda t, x: t.stack([row * i for i, row in enumerate(x)]), M),
    ("gather", lambda t, x: x.gather(1, t.tensor([[0, 0], [3, 1], [2, 2]])), M),
    ("gather dim 0", lambda t, x: t.gather(x, 0, t.tensor(INDEX)), M),
    ("scatter src", lambda t, x, s: x.scatter(0, t.tensor(INDEX[:, :3]), s), M, N[:2, :3]),
    ("scatter value", lambda t, x: x.scatter(1, t.tensor([[0], [2], [1]]), 9.0), M),
    ("scatter_add", lambda t, x, s: x.scatter_add(0, t.tensor(INDEX), s), M, N[:2]),
    ("torch scatter_add", lambda t, x, s: t.scatter_add(x, 1, t.tensor([[0, 0], [1, 1], [3, 3]]), s), M, N[:, :2]),
)


@cases(*INDEXING)
def test_indexing_gradients(st, real, case):
    grads_agree(st, real, case[1], *case[2:])


SHAPES = (
    ("reshape", lambda t, x: x.reshape(4, 3) * t.arange(3.0, dtype=x.dtype), M),
    ("view", lambda t, x: x.view(-1), M),
    ("flatten", lambda t, x: x.flatten(1), CUBE),
    ("ravel", lambda t, x: x.T.ravel(), M),
    ("squeeze", lambda t, x: x.squeeze(1), COL),
    ("unsqueeze", lambda t, x: x.unsqueeze(-2) * x.unsqueeze(-1), VEC),
    ("permute", lambda t, x: x.permute(2, 0, 1), CUBE),
    ("transpose", lambda t, x: x.transpose(0, -1), CUBE),
    ("T then matmul", lambda t, x: x.T @ x, M),
    ("expand", lambda t, x: x.expand(2, 3, 4), ROW),
    ("expand -1", lambda t, x: x.expand(-1, 4), COL),
    ("flip", lambda t, x: x.flip(0, 1) * x, M),
    ("roll", lambda t, x: x.roll(1, 1), M),
    ("roll two dims", lambda t, x: t.roll(x, (1, -2), (0, 1)), M),
    ("roll flat", lambda t, x: x.roll(5), M),
    ("triu", lambda t, x: x.triu(1), SQ),
    ("tril", lambda t, x: t.tril(x, -1), SQ),
    ("diag of matrix", lambda t, x: x.diag(), SQ),
    ("diag offset", lambda t, x: t.diag(x, -1), M),
    ("diag of vector", lambda t, x: t.diag(x, 1), VEC),
    ("diff", lambda t, x: x.diff(dim=0), M),
    ("diff twice", lambda t, x: t.diff(x, n=2), M),
    ("split", lambda t, x: x.split(3, 1), M),
    ("split sizes", lambda t, x: x.split([1, 2], 0), M),
    ("chunk", lambda t, x: x.chunk(2, -1), M),
    ("unbind", lambda t, x: x.unbind(1), M),
    ("cat", lambda t, x, y: t.cat([x, y, x], dim=1), M, N),
    ("cat promotes", lambda t, x, y: t.cat([x.float(), y]), M, N),
    ("stack", lambda t, x, y: t.stack([x, y], -1), M, N),
    ("contiguous", lambda t, x: x.T.contiguous() * 2, M),
)


@cases(*SHAPES)
def test_shape_gradients(st, real, case):
    grads_agree(st, real, case[1], *case[2:])


REDUCTIONS = (
    ("sum", lambda t, x: x.sum(), M),
    ("sum dim", lambda t, x: x.sum(1), CUBE),
    ("sum negative dim keepdim", lambda t, x: x.sum(-1, keepdim=True), CUBE),
    ("sum two dims", lambda t, x: t.sum(x, (0, 2)), CUBE),
    ("sum numpy names", lambda t, x: x.sum(axis=0, keepdims=True), M),
    ("mean", lambda t, x: x.mean(), M),
    ("mean dims keepdim", lambda t, x: x.mean((-1, 0), keepdim=True), CUBE),
    ("amax ties", lambda t, x: x.amax(1), TIES),
    ("amin whole", lambda t, x: x.amin(), TIES),
    ("max whole ties", lambda t, x: x.max(), TIES),
    ("max dim", lambda t, x: x.max(1), TIES),
    ("min dim keepdim", lambda t, x: x.min(dim=0, keepdim=True), M),
    ("max dim values", lambda t, x: t.max(x, -1).values, CUBE),
    ("var", lambda t, x: x.var(), M),
    ("var dim biased", lambda t, x: x.var(1, unbiased=False), M),
    ("std", lambda t, x: x.std(dim=0), M),
    ("std correction keepdim", lambda t, x: t.std(x, -1, keepdim=True, correction=0), CUBE),
    ("std of equal values", lambda t, x: x.std(1), np.array([[2.0, 2.0, 2.0], [1.0, 2.0, 4.0]])),
    ("norm", lambda t, x: x.norm(), M),
    ("norm dim", lambda t, x: x.norm(dim=1, keepdim=True), M),
    ("norm 1", lambda t, x: x.norm(p=1, dim=0), ZEROS),
    ("norm inf with ties", lambda t, x: x.norm(float("inf"), dim=1), TIES),
    ("norm 3", lambda t, x: x.norm(p=3), ZEROS),
    ("norm 1.5 with zeros", lambda t, x: t.norm(x, 1.5, dim=0), ZEROS),
    ("norm of zero rows", lambda t, x: x.norm(dim=1), np.array([[0.0, 0.0], [3.0, 4.0]])),
    ("norm 3 of zero rows", lambda t, x: x.norm(3, dim=1), np.array([[0.0, 0.0], [3.0, 4.0]])),
    ("trace", lambda t, x: x.trace(), SQ),
    ("cumsum", lambda t, x: x.cumsum(1), M),
    ("cumsum dim 0", lambda t, x: t.cumsum(x, dim=-3), CUBE),
    ("cumprod", lambda t, x: x.cumprod(1), POSM),
    ("cumprod with zeros", lambda t, x: x.cumprod(1), ZEROS),
    ("cumprod with zeros dim 0", lambda t, x: t.cumprod(x, 0), ZEROS),
    ("logsumexp", lambda t, x: x.logsumexp(1), M),
    ("logsumexp keepdim", lambda t, x: t.logsumexp(x * 50, dim=(0, 1), keepdim=True), M),
    ("softmax", lambda t, x: x.softmax(-1), M),
    ("softmax dim 0", lambda t, x: t.softmax(x * 30, dim=0), M),
    ("log_softmax", lambda t, x: x.log_softmax(1), CUBE),
)


@cases(*REDUCTIONS)
def test_reduction_gradients(st, real, case):
    grads_agree(st, real, case[1], *case[2:])


ORDERING = (
    ("sort", lambda t, x: x.sort(1), M),
    ("sort descending", lambda t, x: t.sort(x, 0, descending=True).values, M),
    ("topk", lambda t, x: x.topk(2), M),
    ("median", lambda t, x: x.median(), M),
    ("median dim", lambda t, x: x.median(1), TIES),
    ("mode", lambda t, x: x.mode(1), TIES),
    ("quantile", lambda t, x: x.quantile(0.3), M),
    ("quantile dim", lambda t, x: x.quantile(0.5, dim=1), M),
    ("quantile tensor q keepdim", lambda t, x: x.quantile(t.tensor([0.1, 0.75], dtype=x.dtype), dim=0, keepdim=True), M),
    ("quantile lower", lambda t, x: t.quantile(x, 0.4, dim=-1, interpolation="lower"), CUBE),
    ("quantile midpoint", lambda t, x: t.quantile(x, 0.4, interpolation="midpoint"), M),
)


@cases(*ORDERING)
def test_ordering_gradients(st, real, case):
    grads_agree(st, real, case[1], *case[2:])


LINALG = (
    ("mm", lambda t, x, y: x @ y, M, M.T.copy()),
    ("matmul broadcast batch", lambda t, x, y: t.matmul(x, y), CUBE, SQ),
    ("matmul batch of one", lambda t, x, y: x @ y, CUBE[:1], CUBE.transpose(0, 2, 1).copy()),
    ("matrix vector", lambda t, x, v: x @ v, M, VEC),
    ("vector matrix", lambda t, v, x: v @ x, VEC[:3], M),
    ("dot", lambda t, u, v: u @ v, VEC, ROW),
    ("vector times batch", lambda t, v, x: v @ x, VEC[:3], CUBE),
    ("batch times vector", lambda t, x, v: x @ v, CUBE, VEC),
    ("mm function", lambda t, x, y: t.mm(x, y), M, SQ),
    ("bmm", lambda t, x, y: t.bmm(x, y), CUBE, CUBE.transpose(0, 2, 1).copy()),
    ("outer", lambda t, u, v: t.outer(u, v), VEC, ROW),
    ("einsum matrix product", lambda t, x, y: t.einsum("ij,jk->ik", x, y), M, SQ),
    ("einsum batch", lambda t, x, y: t.einsum("bij,bkj->bik", x, y), CUBE, CUBE),
    ("einsum sum away", lambda t, x: t.einsum("ij->j", x), M),
    ("einsum total", lambda t, x: t.einsum("ij->", x), M),
    ("einsum diagonal", lambda t, x: t.einsum("ii->i", x), SQ),
    ("einsum trace", lambda t, x: t.einsum("ii", x), SQ),
    ("einsum same operand twice", lambda t, x: t.einsum("i,i->", x, x), VEC),
    ("einsum outer", lambda t, u, v: t.einsum("i,j->ij", u, v), VEC, ROW),
    ("einsum implicit", lambda t, x, y: t.einsum("ij,jk", x, y), M, SQ),
    ("einsum ellipsis", lambda t, x, y: t.einsum("...ij,jk->...ik", x, y), CUBE, SQ),
    ("einsum list", lambda t, x, y: t.einsum("ij,ij->i", [x, y]), M, N),
    ("cdist", lambda t, x, y: t.cdist(x, y), M, N),
    ("cdist with equal rows", lambda t, x, y: t.cdist(x, y), M, np.concatenate([M[:1], N[1:]])),
    ("cdist p=1", lambda t, x, y: t.cdist(x, y, p=1.0), M, N),
    ("meshgrid", lambda t, u, v: t.meshgrid(u, v, indexing="ij"), VEC, ROW[:3]),
    ("meshgrid xy", lambda t, u, v: t.meshgrid(u, v, indexing="xy"), VEC, ROW[:3]),
)


@cases(*LINALG)
def test_linalg_gradients(st, real, case):
    grads_agree(st, real, case[1], *case[2:])


def _normal(t, loc, scale):
    n = t.distributions.Normal(loc, scale)
    value = t.tensor(np.linspace(-1, 1, 4), dtype=loc.dtype)
    other = t.distributions.Normal(loc * 0.5, scale + 1)
    return n.log_prob(value), n.entropy(), t.distributions.kl_divergence(n, other), n.variance


@cases(("normal", _normal, ROW, np.abs(VEC) + 0.3))
def test_distribution_gradients(st, real, case):
    grads_agree(st, real, case[1], *case[2:])


def test_normal_rsample_carries_gradient(st):
    loc = st.zeros(3, requires_grad=True)
    scale = st.ones(3, requires_grad=True)
    st.manual_seed(0)
    sample = st.distributions.Normal(loc, scale).rsample()
    sample.sum().backward()
    np.testing.assert_allclose(loc.grad.numpy(), np.ones(3))
    np.testing.assert_allclose(scale.grad.numpy(), (sample - loc).detach().numpy(), rtol=1e-6)
    assert not st.distributions.Normal(loc, scale).sample().requires_grad


ALL_CASES = ELEMENTWISE + INDEXING + SHAPES + REDUCTIONS + ORDERING + LINALG


@cases(*ALL_CASES)
def test_gradients_with_an_infinite_upstream_gradient(st, real, case):
    grads_agree(st, real, case[1], *case[2:], infinite=True)
