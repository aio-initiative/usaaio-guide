"""torch.save and torch.load round trips, the weights_only guard, and the thread count calls."""

import collections
import io
import pickle

import numpy as np
import pytest

from parity import A, INT, same


class Opaque:
    """Not a tensor, a container or a number, so weights_only loading must refuse it."""


def _round_trip(t, obj, **load_args):
    buffer = io.BytesIO()
    t.save(obj, buffer)
    buffer.seek(0)
    return t.load(buffer, **load_args)


def test_a_state_dict_survives_a_file(st, tmp_path):
    state = collections.OrderedDict(weight=st.tensor(A), steps=st.tensor(INT), mask=st.tensor(A) > 0)
    path = tmp_path / "model.pt"
    st.save({"model": state, "epoch": 3, "name": "mlp"}, path)
    loaded = st.load(path, weights_only=True)
    assert loaded["epoch"] == 3 and loaded["name"] == "mlp"
    assert isinstance(loaded["model"], collections.OrderedDict) and list(loaded["model"]) == list(state)
    for key, value in state.items():
        got = loaded["model"][key]
        assert isinstance(got, st.Tensor) and got.dtype is value.dtype
        np.testing.assert_array_equal(got.numpy(), value.numpy())


def test_a_string_path_and_map_location(st, tmp_path):
    path = str(tmp_path / "x.pt")
    st.save(st.tensor(A), path)
    loaded = st.load(path, map_location="cpu", weights_only=True)
    np.testing.assert_array_equal(loaded.numpy(), A)


def test_loaded_values_match_real_torch(st, real):
    got, want = (_round_trip(t, {"w": t.tensor(A).double(), "shape": t.zeros(2, 3).shape}) for t in (st, real))
    same(got["w"], want["w"])
    assert type(got["shape"]) is st.Size and tuple(got["shape"]) == tuple(want["shape"]) == (2, 3)


def test_requires_grad_is_kept(st):
    loaded = _round_trip(st, st.tensor([1.0], requires_grad=True))
    assert loaded.requires_grad is True


@pytest.mark.parametrize("weights_only", [True, None])
def test_weights_only_refuses_other_objects(st, real, weights_only):
    for t in (real, st):
        with pytest.raises(pickle.UnpicklingError):
            _round_trip(t, {"thing": Opaque()}, weights_only=weights_only)


def test_weights_only_false_loads_anything(st, real):
    for t in (real, st):
        assert isinstance(_round_trip(t, Opaque(), weights_only=False), Opaque)


def test_thread_counts(st):
    assert isinstance(st.get_num_threads(), int) and st.get_num_threads() >= 1
    assert st.set_num_threads(2) is None
