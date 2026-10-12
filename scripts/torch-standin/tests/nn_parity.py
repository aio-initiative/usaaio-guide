"""Helpers for torch.nn parity: the same layer on both libraries, with the real layer's weights
and buffers copied into the stand-in, run forward and backward and compared."""

import numpy as np

from parity import same


def to_standin(st, state):
    """A real state_dict as stand-in tensors with the same names, values and dtypes."""
    return {name: st.tensor(value.detach().numpy().copy()) for name, value in state.items()}


def twin(st, real, make, double=True):
    """make(nn) on both libraries; the stand-in then loads the real layer's state_dict."""
    r, s = make(real.nn), make(st.nn)
    if double:
        r.double(), s.double()
    loaded = s.load_state_dict(to_standin(st, r.state_dict()))
    assert repr(loaded) == "<All keys matched successfully>"
    return s, r


def run(t, fn, arrays, seed=7, infinite=False, needs_grad=None):
    """fn(*tensors) with every float input requiring grad (or those flagged in needs_grad);
    each output entry is weighted by a random number (every third one inf, with `infinite`),
    the sum is backpropagated. Returns (outputs, input gradients)."""
    needs_grad = needs_grad or [a.dtype.kind == "f" for a in arrays]
    inputs = [t.tensor(a, requires_grad=bool(g)) for a, g in zip(arrays, needs_grad)]
    out = fn(*inputs)
    outs = [o for o in (out if isinstance(out, tuple) else (out,)) if o is not None and o.dtype.is_floating_point]
    rng = np.random.default_rng(seed)
    total = 0
    for o in outs:
        w = rng.standard_normal(tuple(o.shape))
        if infinite:
            w.reshape(-1)[::3] = np.inf
        total = total + (o * t.tensor(w, dtype=o.dtype)).sum()
    if isinstance(total, t.Tensor) and total.requires_grad:
        total.backward()
    return [o.detach() for o in outs], [x.grad for x in inputs if x.requires_grad]


def grads_same(got, want):
    """Two {name: gradient or None} dicts agree."""
    assert got.keys() == want.keys()
    for name, value in want.items():
        if value is None:
            assert got[name] is None, name
        else:
            same(got[name], value)


def layer_agrees(st, real, make, *arrays, double=True, train=True, call=None):
    """The layer make(nn) gives the same outputs, input gradients and parameter gradients on
    both libraries, and leaves the same state behind (BatchNorm's running statistics)."""
    call = call or (lambda layer, *inputs: layer(*inputs))
    s, r = twin(st, real, make, double)
    s.train(train), r.train(train)
    got, want = run(st, lambda *x: call(s, *x), arrays), run(real, lambda *x: call(r, *x), arrays)
    same(got, want)
    grads_same({n: p.grad for n, p in s.named_parameters()}, {n: p.grad for n, p in r.named_parameters()})
    after_s, after_r = s.state_dict(), r.state_dict()
    assert list(after_s) == list(after_r)
    for name in after_r:
        same(after_s[name], after_r[name])
    return s, r


def function_agrees(st, real, fn, *arrays, infinite=False, needs_grad=None, torch_module=False):
    """fn(F, *tensors) (or fn(torch, *tensors) with torch_module) gives the same values and
    input gradients on both libraries."""
    pick = (lambda t: t) if torch_module else (lambda t: t.nn.functional)
    got = run(st, lambda *x: fn(pick(st), *x), arrays, infinite=infinite, needs_grad=needs_grad)
    want = run(real, lambda *x: fn(pick(real), *x), arrays, infinite=infinite, needs_grad=needs_grad)
    same(got, want)
