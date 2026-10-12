"""The Tensor class itself: dtypes, shape, device, conversion to Python and NumPy, printing."""

import numpy as np
import pytest

from parity import A, INT, MASK, V, cases, check, check_raises, same

DTYPES = ["bool", "uint8", "int8", "int16", "int32", "int64", "float32", "float64"]


def test_tensor_holds_an_array_and_the_autograd_fields(st):
    array = np.ones((2, 3), dtype=np.float32)
    x = st.Tensor(array)
    assert x.numpy() is array and x.grad is None and x.grad_fn is None and x.requires_grad is False
    assert st.Tensor(array, requires_grad=True).requires_grad is True
    assert isinstance(x.shape, st.Size) and isinstance(x.shape, tuple) and x.shape == (2, 3)
    assert x.dtype is st.float32 and x.ndim == 2 and x.T.shape == (3, 2)
    x.data = st.Tensor(np.zeros(2, dtype=np.float32))
    assert type(x.data) is st.Tensor and x.shape == (2,)


@pytest.mark.parametrize("name", ["grad", "data", "requires_grad", "device", "dtype", "shape", "grad_fn", "T", "ndim"])
def test_attributes_exist_on_the_class(st, name):
    getattr(st.Tensor, name)


def test_dtype_names_and_aliases(st, real):
    for name in DTYPES:
        mine, theirs = getattr(st, name), getattr(real, name)
        assert repr(mine) == repr(theirs) and str(mine) == str(theirs)
        assert mine.is_floating_point == theirs.is_floating_point
        assert mine.itemsize == theirs.itemsize
    assert st.float is st.float32 and st.double is st.float64
    assert st.long is st.int64 and st.int is st.int32


@pytest.mark.parametrize("left", DTYPES)
@pytest.mark.parametrize("right", DTYPES)
def test_promote_types(st, real, left, right):
    got = st.promote_types(getattr(st, left), getattr(st, right))
    assert str(got) == str(real.promote_types(getattr(real, left), getattr(real, right)))


def test_size(st, real):
    mine, theirs = st.zeros(2, 3).shape, real.zeros(2, 3).shape
    assert repr(mine) == repr(theirs) == "torch.Size([2, 3])"
    assert mine == (2, 3) and mine.numel() == 6 and len(mine) == 2
    assert type(mine[1:]) is st.Size and type(mine + (4,)) is st.Size and mine[0] == 2
    x = st.zeros(2, 3)
    assert x.size() == (2, 3) and x.size(1) == 3 and x.size(-1) == 3 and x.size(dim=0) == 2


def test_device(st, real):
    mine, theirs = st.device("cpu"), real.device("cpu")
    assert repr(mine) == repr(theirs) and str(mine) == str(theirs) == "cpu"
    assert mine.type == "cpu" and mine.index is None and mine == st.device("cpu")
    assert repr(st.zeros(1).device) == repr(real.zeros(1).device)
    assert st.device(mine) == mine and str(st.device("cuda:0")) == "cuda:0"
    assert st.device("cuda" if st.cuda.is_available() else "cpu") == mine
    assert st.cuda.is_available() is False
    with pytest.raises(RuntimeError):
        st.device("gpu")


@pytest.mark.parametrize(
    "move",
    [
        lambda t, x: x.to("cuda"),
        lambda t, x: x.to(t.device("cuda")),
        lambda t, x: x.to(device="cuda:0"),
        lambda t, x: x.cuda(),
        lambda t, x: t.zeros(2, device="cuda"),
        lambda t, x: t.tensor([1.0], device=t.device("cuda")),
    ],
)
def test_cuda_is_refused_with_the_browser_message(st, move):
    with pytest.raises(RuntimeError) as err:
        move(st, st.zeros(2))
    assert str(err.value) == "CUDA is not available in the browser. Run this notebook in Colab for a GPU."


@cases(
    ("item float", lambda t, x: x[0, 0].item(), A),
    ("item int", lambda t, x: x[0, 0].item(), INT),
    ("item bool", lambda t, x: x[0, 0].item(), MASK),
    ("item of one element", lambda t, x: x[:1, :1].item(), A),
    ("tolist", lambda t, x: x.tolist(), A),
    ("tolist int", lambda t, x: x.tolist(), INT),
    ("tolist scalar", lambda t, x: x[0, 0].tolist(), A),
    ("float()", lambda t, x: float(x[0, 0]), A),
    ("int()", lambda t, x: int(x[0, 0]), A),
    ("bool()", lambda t, x: bool(x[0, 0]), MASK),
    ("index", lambda t, x: [10, 20, 30, 40, 50, 60][abs(x[0, 0])], INT),
    ("len", lambda t, x: len(x), A),
    ("iter", lambda t, x: list(x), A),
    ("numel", lambda t, x: (x.numel(), x.nelement(), x.dim(), x.ndim, x.element_size()), A),
    ("element_size int", lambda t, x: x.element_size(), INT),
    ("is_tensor", lambda t, x: (t.is_tensor(x), t.is_tensor(1.0), isinstance(x, t.Tensor)), A),
    ("is_floating_point", lambda t, x: x.is_floating_point(), A),
    ("is_floating_point int", lambda t, x: x.is_floating_point(), INT),
    ("format scalar", lambda t, x: f"{x[0, 0]:.3f} {x[0, 1]:8.2e} {x[0, 0]}", A),
    ("format int", lambda t, x: f"{x[0, 0]:d} {x[0, 0]}", INT),
    ("cpu", lambda t, x: x.cpu(), A),
    ("T", lambda t, x: x.T, A),
)
def test_python_conversions(st, real, case):
    check(st, real, *case[1:])


@cases(
    ("item of many", lambda t, x: x.item(), A),
    ("bool of many", lambda t, x: bool(x), A),
    ("float of many", lambda t, x: float(x), A),
    ("len of scalar", lambda t, x: len(x[0, 0]), A),
    ("iter of scalar", lambda t, x: list(x[0, 0]), A),
    ("format vector with spec", lambda t, x: f"{x:.2f}", A),
)
def test_python_conversion_errors(st, real, case):
    check_raises(st, real, *case[1:])


def test_numpy_shares_memory(st):
    x = st.tensor([1.0, 2.0])
    assert x.numpy().dtype == np.float32
    x.numpy()[0] = 9.0
    assert x.tolist() == [9.0, 2.0]
    assert np.asarray(x).dtype == np.float32 and np.asarray(x, dtype=np.float64).dtype == np.float64
    assert np.allclose(np.array([x, x]), [[9.0, 2.0], [9.0, 2.0]])


def test_tensors_hash_by_identity(st):
    x = st.tensor([1.0, 2.0])
    assert len({x, x, st.tensor([1.0, 2.0])}) == 2


REPRS = [
    [1.5, 2.5],
    [3, -7],
    [True, False],
    1.5,
    3,
    [],
    [[0.0, 0.0], [0.0, 1.0]],
    [1e-5, 1.0],
    [1.0, 1e6],
    [1.0, float("nan"), float("inf")],
    [[-1.25, 10.5], [100.125, 0.0]],
    V,
    A,
    INT,
    MASK,
    A.astype(np.float64),
    INT.astype(np.int32),
    np.arange(3, dtype=np.uint8),
    np.zeros((2, 0), dtype=np.float32),
    np.arange(2000, dtype=np.float32),
    np.arange(2000).reshape(40, 50),
]


@pytest.mark.parametrize("value", REPRS, ids=range(len(REPRS)))
def test_repr_matches_torch(st, real, value):
    assert repr(st.tensor(value)) == repr(real.tensor(value))
    assert str(st.tensor(value)) == str(real.tensor(value))


def test_repr_shows_requires_grad(st, real):
    assert repr(st.tensor([1.0, 2.0], requires_grad=True)) == repr(real.tensor([1.0, 2.0], requires_grad=True))


def test_same_helper_rejects_a_wrong_dtype(st, real):
    with pytest.raises(AssertionError):
        same(st.tensor([1.0]), real.tensor([1.0], dtype=real.float64))


@cases(
    ("sizes", lambda t: (t.Tensor(2, 3).shape, str(t.Tensor(2, 3).dtype), t.Tensor(2, 3).requires_grad)),
    ("one size", lambda t: t.Tensor(3).shape),
    ("a Size", lambda t: t.Tensor(t.Size([2, 3])).shape),
    ("nothing", lambda t: t.Tensor()),
    ("a list", lambda t: t.Tensor([1, 2])),
    ("a nested list", lambda t: t.Tensor([[1], [2]])),
    ("a tuple is data", lambda t: t.Tensor((2, 3))),
    ("an int tensor", lambda t: t.Tensor(t.tensor([1, 2]))),
    ("a float64 tensor", lambda t: t.Tensor(t.tensor([1.5, 2.5], dtype=t.float64))),
)
def test_the_torch_tensor_constructor(st, real, case):
    check(st, real, case[1])


@cases(
    ("a negative size", lambda t: t.Tensor(-1)),
    ("a float", lambda t: t.Tensor(3.5)),
)
def test_the_torch_tensor_constructor_errors(st, real, case):
    check_raises(st, real, case[1])


def test_tensor_of_a_tensor_shares_its_values_as_in_torch(st, real):
    for t in (st, real):
        x = t.tensor([1, 2])
        y = t.Tensor(x)
        y[0] = 9
        assert y is not x and x.tolist() == [9, 2]


def test_a_missing_method_says_it_is_not_available(st):
    x = st.zeros(2)
    with pytest.raises(NotImplementedError) as err:
        x.not_a_real_method()
    assert str(err.value) == (
        "torch.Tensor.not_a_real_method is not available in the browser version of PyTorch. "
        "Download the notebook or open it in Colab to use it."
    )
    # It is also an AttributeError, so the usual ways of asking still answer.
    assert not hasattr(x, "not_a_real_method") and getattr(x, "not_a_real_method", 7) == 7
    assert hasattr(x, "shape") and hasattr(x, "data") and hasattr(x, "grad")


def test_private_names_and_failing_properties_keep_their_attribute_errors(st):
    x = st.zeros(2)
    with pytest.raises(AttributeError) as err:
        x._not_there
    assert not isinstance(err.value, NotImplementedError)

    class Broken(st.Tensor):
        @property
        def broken(self):
            raise AttributeError("the property's own message")

    with pytest.raises(AttributeError, match="the property's own message"):
        Broken(np.zeros(2, dtype=np.float32)).broken
