"""Creation functions: values, shapes and inferred dtypes equal real torch's."""

import numpy as np
import pytest

from parity import A, INT, MASK, cases, check, check_raises, same

# Real torch warns that a list of arrays is slow to convert; that is not what is under test.
pytestmark = pytest.mark.filterwarnings("ignore:Creating a tensor from a list of numpy.ndarrays")


@cases(
    ("floats", lambda t: t.tensor([1.5, 2.5])),
    ("ints", lambda t: t.tensor([1, 2])),
    ("bools", lambda t: t.tensor([True, False])),
    ("int and float", lambda t: t.tensor([1, 2.5])),
    ("bool and int", lambda t: t.tensor([True, 2])),
    ("nested", lambda t: t.tensor([[1, 2], [3, 4]])),
    ("tuple", lambda t: t.tensor((1.0, 2.0))),
    ("range", lambda t: t.tensor(range(3))),
    ("empty", lambda t: t.tensor([])),
    ("empty rows", lambda t: t.tensor([[], []])),
    ("scalar float", lambda t: t.tensor(3.5)),
    ("scalar int", lambda t: t.tensor(3)),
    ("scalar bool", lambda t: t.tensor(True)),
    ("numpy float64 array", lambda t: t.tensor(np.ones(2))),
    ("numpy float32 array", lambda t: t.tensor(np.ones(2, dtype=np.float32))),
    ("numpy int32 array", lambda t: t.tensor(np.ones(2, dtype=np.int32))),
    ("numpy uint8 array", lambda t: t.tensor(np.ones(2, dtype=np.uint8))),
    ("numpy bool array", lambda t: t.tensor(MASK)),
    ("numpy float64 scalar", lambda t: t.tensor(np.float64(1.5))),
    ("list of numpy float64", lambda t: t.tensor([np.float64(1.5), np.float64(2.5)])),
    ("list of numpy int32", lambda t: t.tensor([np.int32(1)])),
    ("list of arrays", lambda t: t.tensor([np.ones(2), np.zeros(2)])),
    ("list of scalar tensors", lambda t: t.tensor([t.tensor(1.0), t.tensor(2.0)])),
    ("dtype float32", lambda t: t.tensor([1, 2], dtype=t.float32)),
    ("dtype float64", lambda t: t.tensor([1.5, 2.5], dtype=t.float64)),
    ("dtype long", lambda t: t.tensor([1.7, -1.7], dtype=t.long)),
    ("dtype bool", lambda t: t.tensor([0, 2], dtype=t.bool)),
    ("dtype python float", lambda t: t.tensor([1, 2], dtype=float)),
    ("dtype python int", lambda t: t.zeros(2, dtype=int)),
    ("dtype python bool", lambda t: t.zeros(2, dtype=bool)),
    ("as_tensor list", lambda t: t.as_tensor([1.0, 2.0])),
    ("as_tensor array", lambda t: t.as_tensor(np.ones(2))),
    ("as_tensor dtype", lambda t: t.as_tensor(np.ones(2), dtype=t.float32)),
    ("as_tensor tensor", lambda t: t.as_tensor(t.tensor([1, 2]))),
    ("from_numpy", lambda t: t.from_numpy(INT.copy())),
    ("zeros ints", lambda t: t.zeros(2, 3)),
    ("zeros tuple", lambda t: t.zeros((2, 3))),
    ("zeros list", lambda t: t.zeros([2, 3])),
    ("zeros size keyword", lambda t: t.zeros(size=(2, 3))),
    ("zeros of a Size", lambda t: t.zeros(t.zeros(2, 3).shape)),
    ("zeros numpy int", lambda t: t.zeros(np.int64(2))),
    ("zeros scalar", lambda t: t.zeros(())),
    ("zeros long", lambda t: t.zeros(2, dtype=t.long)),
    ("ones", lambda t: t.ones(2, 3)),
    ("ones bool", lambda t: t.ones(2, 2, dtype=t.bool)),
    ("full int", lambda t: t.full((2,), 3)),
    ("full float", lambda t: t.full((2, 2), 3.5)),
    ("full bool", lambda t: t.full((2,), True)),
    ("full dtype", lambda t: t.full((2,), 3, dtype=t.float32)),
    ("empty shape", lambda t: t.empty(2, 3).shape),
    ("empty dtype", lambda t: t.empty(2, dtype=t.int64).dtype is t.int64),
    ("arange end", lambda t: t.arange(5)),
    ("arange start end", lambda t: t.arange(2, 5)),
    ("arange step", lambda t: t.arange(0, 10, 3)),
    ("arange float step", lambda t: t.arange(1, 2.5, 0.5)),
    ("arange float end", lambda t: t.arange(5.0)),
    ("arange tenths", lambda t: t.arange(0, 1, 0.1)),
    ("arange negative step", lambda t: t.arange(5, 0, -2)),
    ("arange dtype", lambda t: t.arange(5, dtype=t.float32)),
    ("arange tensor end", lambda t: t.arange(t.tensor(3))),
    ("arange keywords", lambda t: t.arange(start=1, end=4)),
    ("linspace", lambda t: t.linspace(0, 1, 5)),
    ("linspace steps keyword", lambda t: t.linspace(-2.0, 2.0, steps=7)),
    ("linspace one step", lambda t: t.linspace(3, 4, 1)),
    ("linspace float64", lambda t: t.linspace(0, 1, 4, dtype=t.float64)),
    ("eye", lambda t: t.eye(3)),
    ("eye rectangular", lambda t: t.eye(2, 3)),
    ("eye dtype", lambda t: t.eye(2, dtype=t.int64)),
    ("zeros_like", lambda t: t.zeros_like(t.tensor(INT))),
    ("ones_like", lambda t: t.ones_like(t.tensor(A))),
    ("ones_like dtype", lambda t: t.ones_like(t.tensor(A), dtype=t.long)),
    ("full_like", lambda t: t.full_like(t.tensor(A), 2.5)),
    ("full_like keeps the int dtype", lambda t: t.full_like(t.tensor(INT), 2.7)),
    ("empty_like shape", lambda t: t.empty_like(t.tensor(A)).shape),
    ("requires_grad", lambda t: t.zeros(2, requires_grad=True).requires_grad),
    ("tensor requires_grad", lambda t: t.tensor([1.0], requires_grad=True).requires_grad),
    ("device keyword", lambda t: t.ones(2, device="cpu")),
    ("device object", lambda t: t.arange(3, device=t.device("cpu"))),
)
def test_creation(st, real, case):
    check(st, real, case[1])


# Digits are compared exactly (parity.same allows 1e-5, which is the size of the float32 rounding
# this section is about). A Python float given an explicit dtype goes straight to that dtype.
EXACT = (
    ("tensor list float64", lambda t: t.tensor([0.3], dtype=t.float64)),
    ("tensor scalar float64", lambda t: t.tensor(0.3, dtype=t.float64)),
    ("tensor several float64", lambda t: t.tensor([0.1, 1 / 3, 2 / 3, 0.7], dtype=t.float64)),
    ("tensor nested float64", lambda t: t.tensor([[0.3, 0.1], [0.2, 0.7]], dtype=t.float64)),
    ("tensor tuple float64", lambda t: t.tensor((0.3, 0.1), dtype=t.float64)),
    ("tensor python float dtype", lambda t: t.tensor([0.3, 0.1], dtype=float)),
    ("tensor int and float float64", lambda t: t.tensor([1, 0.3], dtype=t.float64)),
    ("tensor large float float64", lambda t: t.tensor([16777217.0, 1e-50, 1e300], dtype=t.float64)),
    ("tensor large int float64", lambda t: t.tensor([2**53 + 1], dtype=t.float64)),
    ("tensor bools float64", lambda t: t.tensor([True, False], dtype=t.float64)),
    ("tensor of scalar tensors float64", lambda t: t.tensor([t.tensor(0.3, dtype=t.float64)], dtype=t.float64)),
    ("tensor float64 requires_grad", lambda t: t.tensor([0.3], dtype=t.float64, requires_grad=True).detach()),
    ("tensor float32 explicit", lambda t: t.tensor([0.3, 1 / 3], dtype=t.float32)),
    ("tensor float32 default", lambda t: t.tensor([0.3, 1 / 3])),
    ("tensor float32 scalar default", lambda t: t.tensor(0.3)),
    ("tensor large float default", lambda t: t.tensor([16777217.0, 1e-50])),
    ("tensor floats to long", lambda t: t.tensor([0.7, -0.7, 2.9], dtype=t.long)),
    ("tensor floats to int8", lambda t: t.tensor([1.9, -1.9], dtype=t.int8)),
    ("tensor floats to bool", lambda t: t.tensor([0.3, 0.0], dtype=t.bool)),
    ("tensor numpy float64", lambda t: t.tensor(np.array([0.3, 0.1]))),
    ("tensor numpy float64 scalar", lambda t: t.tensor(np.float64(0.3))),
    ("tensor numpy float32 to float64", lambda t: t.tensor(np.array([0.3], dtype=np.float32), dtype=t.float64)),
    ("as_tensor list float64", lambda t: t.as_tensor([0.3], dtype=t.float64)),
    ("as_tensor scalar float64", lambda t: t.as_tensor(0.3, dtype=t.float64)),
    ("as_tensor list default", lambda t: t.as_tensor([0.3])),
    ("as_tensor ints float64", lambda t: t.as_tensor([1, 2], dtype=t.float64)),
    ("as_tensor float32 tensor to float64", lambda t: t.as_tensor(t.tensor([0.3]), dtype=t.float64)),
    ("full float64", lambda t: t.full((2,), 0.3, dtype=t.float64)),
    ("full default", lambda t: t.full((2,), 0.3)),
    ("full_like float64", lambda t: t.full_like(t.zeros(2, dtype=t.float64), 0.3)),
    ("float32 tensor cast up keeps its float32 digits", lambda t: t.tensor([0.3]).to(t.float64)),
    ("legacy Tensor constructor is float32", lambda t: t.Tensor([0.3])),
    ("float64 tensor times a Python float", lambda t: t.tensor([0.3], dtype=t.float64) * 0.3),
    ("float64 tensor minus a Python float", lambda t: t.tensor([0.3, 0.1], dtype=t.float64) - t.tensor([0.3, 0.1], dtype=t.float64)),
    ("float64 tensor equals a Python float", lambda t: t.tensor([0.3, 0.1], dtype=t.float64) == 0.3),
    ("float64 tensor item", lambda t: t.tensor(t.tensor(0.3, dtype=t.float64).item(), dtype=t.float64)),
)


@cases(*EXACT)
def test_creation_keeps_full_precision_and_matches_torch_exactly(st, real, case):
    got, want = case[1](st), case[1](real)
    assert str(got.dtype) == str(want.dtype)
    assert got.tolist() == want.tolist()


def test_tolist_of_a_float64_tensor_made_from_python_floats(st):
    """The reported symptom: 0.3 came back as 0.30000001192092896."""
    assert st.tensor([0.3], dtype=st.float64).tolist() == [0.3]
    assert st.tensor(0.3, dtype=st.float64).item() == 0.3
    assert st.as_tensor([0.3], dtype=st.float64).tolist() == [0.3]
    assert st.tensor([0.3]).tolist() == [0.30000001192092896]  # no dtype: float32, as in torch


def test_a_float64_gradient_check_sees_the_digits_it_was_given(st, real):
    """A central difference at 1e-6 is only as good as the inputs: 0.3 must be 0.3."""

    def run(t):
        x = t.tensor([0.3, 1 / 3], dtype=t.float64, requires_grad=True)
        (x * x * x).sum().backward()
        return x.grad.tolist(), (x.detach() - 0.3).tolist()

    assert run(st) == run(real)


@cases(
    ("ragged list", lambda t: t.tensor([[1, 2], [3]])),
    ("string", lambda t: t.tensor("abc")),
    ("from_numpy of a list", lambda t: t.from_numpy([1, 2])),
    ("arange zero step", lambda t: t.arange(0, 3, 0)),
    ("negative size", lambda t: t.zeros(-1)),
)
def test_creation_errors(st, real, case):
    check_raises(st, real, case[1])


def test_tensor_copies_and_from_numpy_shares(st):
    array = np.zeros(3, dtype=np.float32)
    copied, shared, viewed = st.tensor(array), st.from_numpy(array), st.as_tensor(array)
    array[0] = 7.0
    assert copied[0].item() == 0.0 and shared[0].item() == 7.0 and viewed[0].item() == 7.0
    x = st.tensor([1.0])
    assert st.as_tensor(x) is x and st.tensor(x) is not x


def test_unsupported_numpy_dtype_is_refused(st):
    with pytest.raises(TypeError, match="float16"):
        st.tensor(np.ones(2, dtype=np.float16))
    with pytest.raises(TypeError):
        st.from_numpy(np.array(["a", "b"]))


def test_like_functions_take_the_shape_of_the_input(st, real):
    same(st.zeros_like(st.zeros(2, 5)), real.zeros_like(real.zeros(2, 5)))
