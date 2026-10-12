"""Random numbers: the draws differ from real torch's, so these compare shapes, dtypes, ranges
and errors with torch, and check that a seed makes the stand-in repeat itself."""

import numpy as np
import pytest

from parity import A, INT, cases, check_raises


def _like(got, want):
    assert type(got).__name__ == "Tensor"
    assert tuple(got.shape) == tuple(want.shape) and str(got.dtype) == str(want.dtype)
    assert got.requires_grad == want.requires_grad


@cases(
    ("rand", lambda t: t.rand(2, 3)),
    ("rand tuple", lambda t: t.rand((2, 3))),
    ("rand size keyword", lambda t: t.rand(size=(4,))),
    ("rand scalar", lambda t: t.rand(())),
    ("rand float64", lambda t: t.rand(2, dtype=t.float64)),
    ("rand requires_grad", lambda t: t.rand(2, requires_grad=True)),
    ("randn", lambda t: t.randn(3, 1)),
    ("randn float64", lambda t: t.randn(2, 2, dtype=t.float64)),
    ("randn device", lambda t: t.randn(2, device="cpu")),
    ("randint high", lambda t: t.randint(10, (3,))),
    ("randint low high", lambda t: t.randint(3, 10, size=(2, 2))),
    ("randint keywords", lambda t: t.randint(low=0, high=5, size=(4,))),
    ("randint float dtype", lambda t: t.randint(0, 2, (3,), dtype=t.float32)),
    ("randperm", lambda t: t.randperm(5)),
    ("randperm dtype", lambda t: t.randperm(3, dtype=t.float32)),
    ("randn_like", lambda t: t.randn_like(t.zeros(2, 3, dtype=t.float64))),
    ("rand_like", lambda t: t.rand_like(t.zeros(4))),
    ("generator keyword", lambda t: t.randn(2, generator=t.Generator().manual_seed(1))),
)
def test_shapes_and_dtypes_match_torch(st, real, case):
    _like(case[1](st), case[1](real))


@cases(
    ("rand of ints", lambda t: t.rand(2, dtype=t.long)),
    ("randn of ints", lambda t: t.randn(2, dtype=t.int64)),
    ("randn_like of ints", lambda t: t.randn_like(t.tensor([1, 2]))),
    ("negative size", lambda t: t.rand(-1)),
    ("randint without a size", lambda t: t.randint(10)),
    ("randint float bound", lambda t: t.randint(0, 10.5, (2,))),
    ("randint empty range", lambda t: t.randint(5, 5, (2,))),
)
def test_errors_match_torch(st, real, case):
    check_raises(st, real, case[1])


def test_cuda_is_refused(st):
    with pytest.raises(RuntimeError, match="CUDA is not available in the browser"):
        st.randn(2, device="cuda")


def _draws(t, generator=None):
    return [
        t.rand(3, generator=generator).tolist(),
        t.randn(2, 2, generator=generator).tolist(),
        t.randint(0, 100, (5,), generator=generator).tolist(),
        t.randperm(6, generator=generator).tolist(),
        t.randn_like(t.zeros(3)).tolist() if generator is None else None,
    ]


def test_manual_seed_repeats_the_draws(st):
    st.manual_seed(0)
    first = _draws(st)
    st.manual_seed(0)
    assert _draws(st) == first
    st.manual_seed(1)
    assert _draws(st) != first


def test_manual_seed_returns_the_default_generator(st, real):
    assert type(st.manual_seed(3)).__name__ == type(real.manual_seed(3)).__name__ == "Generator"
    assert st.manual_seed(5).initial_seed() == 5


def test_a_generator_has_its_own_stream(st):
    g = st.Generator()
    assert g.manual_seed(7) is g and g.initial_seed() == 7
    first = _draws(st, g)
    st.manual_seed(0)
    st.rand(10)  # the default generator moving on does not touch g
    g.manual_seed(7)
    assert _draws(st, g) == first
    assert st.Generator(device="cpu").device == st.device("cpu")


def test_values_fall_in_range(st):
    st.manual_seed(0)
    u = st.rand(10_000).numpy()
    assert u.dtype == np.float32 and u.min() >= 0.0 and u.max() < 1.0
    z = st.randn(10_000).numpy()
    assert abs(z.mean()) < 0.05 and abs(z.std() - 1.0) < 0.05
    k = st.randint(-3, 4, (10_000,)).numpy()
    assert k.dtype == np.int64 and set(k.tolist()) == set(range(-3, 4))
    assert sorted(st.randperm(50).tolist()) == list(range(50))


def test_like_functions_follow_the_input(st):
    x = st.tensor(A).double()
    assert st.randn_like(x).shape == x.shape and st.randn_like(x).dtype is st.float64
    assert st.rand_like(x, dtype=st.float32).dtype is st.float32
    with pytest.raises(NotImplementedError):
        st.rand_like(st.tensor(INT))
