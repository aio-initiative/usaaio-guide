"""torch.distributions.Normal and kl_divergence against real torch."""

import numpy as np
import pytest

from parity import A, B, POS, V, cases, check, check_raises, same

SCALE = POS[0]


def _normal(t, loc, scale):
    return t.distributions.Normal(loc, scale)


@cases(
    ("loc and scale", lambda t, m, s: (_normal(t, m, s).loc, _normal(t, m, s).scale), V, SCALE),
    ("numbers", lambda t: (_normal(t, 0.0, 2.0).loc, _normal(t, 0.0, 2.0).scale)),
    ("number and tensor", lambda t, m: _normal(t, m, 1.5).scale, V),
    ("float64", lambda t, m: _normal(t, m.double(), 2.0).scale, V),
    ("broadcast", lambda t, m, s: _normal(t, m, s[:1]).scale, A, SCALE),
    ("moments", lambda t, m, s: (_normal(t, m, s).mean, _normal(t, m, s).stddev, _normal(t, m, s).variance), V, SCALE),
    ("mode", lambda t, m, s: _normal(t, m, s).mode, V, SCALE),
    ("batch_shape", lambda t, m, s: (tuple(_normal(t, m, s).batch_shape), tuple(_normal(t, m, s).event_shape)), A, SCALE),
    ("log_prob", lambda t, m, s, x: _normal(t, m, s).log_prob(x), V, SCALE, B[0]),
    ("log_prob broadcast", lambda t, m, s, x: _normal(t, m, s).log_prob(x), V, SCALE, A),
    ("entropy", lambda t, m, s: _normal(t, m, s).entropy(), V, SCALE),
    ("kl", lambda t, m, s, n: t.distributions.kl_divergence(_normal(t, m, s), _normal(t, n, s * 2)), V, SCALE, B[0]),
    ("kl to a standard normal", lambda t, m, s: t.distributions.kl_divergence(_normal(t, m, s), _normal(t, 0.0, 1.0)), A, SCALE),
    ("kl to itself", lambda t, m, s: t.distributions.kl_divergence(_normal(t, m, s), _normal(t, m, s)), V, SCALE),
    ("repr scalar", lambda t: repr(_normal(t, 0.0, 1.0))),
    ("repr batch", lambda t, m: repr(_normal(t, m, 1.0)), V),
)
def test_normal(st, real, case):
    check(st, real, *case[1:])


@cases(
    ("sample", lambda t, m, s: _normal(t, m, s).sample(), V, SCALE),
    ("sample shape", lambda t, m, s: _normal(t, m, s).sample((5, 2)), V, SCALE),
    ("rsample shape", lambda t, m, s: _normal(t, m, s).rsample((3,)), A, SCALE),
    ("sample float64", lambda t: _normal(t, t.zeros(2, dtype=t.float64), 1.0).sample((2,))),
)
def test_samples_have_torch_shapes(st, real, case):
    got, want = case[1](st, *map(st.tensor, case[2:])), case[1](real, *map(real.tensor, case[2:]))
    assert tuple(got.shape) == tuple(want.shape) and str(got.dtype) == str(want.dtype)


def test_samples_follow_the_distribution(st):
    st.manual_seed(0)
    draws = st.distributions.Normal(st.tensor([1.0, -2.0]), st.tensor([0.5, 3.0])).sample((20_000,)).numpy()
    np.testing.assert_allclose(draws.mean(0), [1.0, -2.0], atol=0.06)
    np.testing.assert_allclose(draws.std(0), [0.5, 3.0], rtol=0.03)


@cases(
    ("negative scale", lambda t: _normal(t, 0.0, -1.0)),
    ("zero scale", lambda t, m: _normal(t, m, 0.0), V),
    ("nan loc", lambda t: _normal(t, float("nan"), 1.0)),
    ("log_prob of a number", lambda t: _normal(t, 0.0, 1.0).log_prob(0.5)),
    ("sample shape as an int", lambda t: _normal(t, 0.0, 1.0).sample(5)),
)
def test_normal_errors(st, real, case):
    check_raises(st, real, *case[1:])


def test_kl_of_unknown_pairs_is_not_implemented(st):
    with pytest.raises(NotImplementedError):
        st.distributions.kl_divergence(st.distributions.Normal(0.0, 1.0), object())


def test_unknown_distributions_say_they_are_not_available(st):
    with pytest.raises(NotImplementedError, match="torch.distributions.Categorical is not available"):
        st.distributions.Categorical
