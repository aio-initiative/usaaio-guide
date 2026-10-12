"""The Tensor class: a NumPy array plus the fields autograd needs.

This file holds only what a tensor is (its array, shape, dtype, device, and conversion to
Python and NumPy). Its maths methods and operators are the functions in _ops, _shape, _reduce
and _linalg, attached to the class by _methods.

The array is `_data`, and the package reads it there. The public `.data` is torch's: another
tensor over the same memory that autograd does not see.
"""

import math
import operator

import numpy as np

from . import _dtype
from ._dims import dim_index
from ._missing import MissingTensorMethod
from ._repr import tensor_repr

NO_CUDA = "CUDA is not available in the browser. Run this notebook in Colab for a GPU."


class Size(tuple):
    """A tensor's shape: a tuple that prints as torch.Size([2, 3])."""

    def __repr__(self):
        return f"torch.Size({list(self)})"

    def numel(self):
        return math.prod(self)

    def __getitem__(self, key):
        item = tuple.__getitem__(self, key)
        return Size(item) if isinstance(key, slice) else item

    def __add__(self, other):
        return Size(tuple.__add__(self, other))

    def __radd__(self, other):
        return Size(other + tuple(self))


def as_size(args):
    """The sizes in f(2, 3), f((2, 3)), f([2, 3]) or f(torch.Size([2, 3])) as a tuple of ints."""
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        args = args[0]
    return tuple(operator.index(n) for n in args)


class device:
    """Where a tensor lives. Only the CPU exists here; a CUDA device can be named but not used."""

    def __init__(self, type, index=None):
        if isinstance(type, device):
            type, index = type.type, type.index
        elif ":" in type:
            type, _, number = type.partition(":")
            index = int(number)
        if type not in ("cpu", "cuda"):
            raise RuntimeError(f"Expected one of cpu, cuda device type at start of device string: {type}")
        self.type, self.index = type, index

    def __repr__(self):
        index = "" if self.index is None else f", index={self.index}"
        return f"device(type='{self.type}'{index})"

    def __str__(self):
        return self.type if self.index is None else f"{self.type}:{self.index}"

    def __eq__(self, other):
        return isinstance(other, device) and (self.type, self.index) == (other.type, other.index)

    def __hash__(self):
        return hash((self.type, self.index))


CPU = device("cpu")


def require_cpu(where):
    """Accept device=None, 'cpu' or device('cpu'); anything else is a GPU we do not have."""
    if where is not None and device(where).type != "cpu":
        raise RuntimeError(NO_CUDA)


def _new_float32(args):
    """The array torch.Tensor(*args) holds: sizes give zeros of that shape, a list a float32
    copy, and a tensor its own array."""
    if not args:
        return np.zeros(0, dtype=np.float32)
    if len(args) == 1 and isinstance(args[0], Size):
        args = tuple(args[0])
    if all(isinstance(a, int) and not isinstance(a, bool) for a in args):
        if any(n < 0 for n in args):
            raise RuntimeError(f"Trying to create tensor with negative dimension {min(args)}: {list(args)}")
        return np.zeros(args, dtype=np.float32)
    if len(args) == 1 and isinstance(args[0], Tensor):
        return args[0]._data  # torch.Tensor(t) shares t's memory and keeps its dtype
    if len(args) != 1 or not isinstance(args[0], (list, tuple)):
        raise TypeError(f"new(): data must be a sequence (got {type(args[0]).__name__})")
    data = [item._data if isinstance(item, Tensor) else item for item in args[0]]
    return np.array(data, dtype=np.float32)


def _type_name(value):
    kind = type(value)
    return kind.__name__ if kind.__module__ == "builtins" else f"{kind.__module__}.{kind.__name__}"


class Tensor:
    # Defaults on the class, so a new tensor only stores what differs. These are autograd's
    # fields; _autograd reads and writes them, and _methods attaches the public properties
    # (grad, grad_fn, requires_grad, is_leaf) that wrap them.
    _grad = None
    _grad_fn = None
    _requires_grad = False  # set on leaves; a tensor with a grad_fn requires grad through it
    _vc = None  # the version counter, a one-item list shared with views and detached copies
    _base = None  # for a view, the tensor whose memory it shares
    _view_version = None  # the version the view's grad_fn was made at (None: never re-derive)
    _accumulator = None  # a leaf's AccumulateGrad node
    _retains_grad = False
    # Makes ndarray + tensor call Tensor.__radd__ instead of treating the tensor as an array.
    __array_priority__ = 1000

    def __init__(self, *args, requires_grad=False):
        """Tensor(array) wraps a NumPy array without copying it; the package builds every
        tensor this way. A notebook's torch.Tensor(...) behaves as in torch: Tensor(2, 3) or
        Tensor(torch.Size([2, 3])) is a float32 tensor of that shape (zeros here, where torch
        leaves the memory uninitialised), Tensor([1, 2]) a float32 copy, and Tensor(t) a new
        tensor sharing t's values and dtype. It never requires grad."""
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            data = args[0]
        elif len(args) == 1 and isinstance(args[0], np.generic):  # what NumPy reductions return
            data = np.asarray(args[0])
        else:
            data = _new_float32(args)
        self._data = data
        if requires_grad:
            if data.dtype.kind != "f":
                raise RuntimeError("Only Tensors of floating point and complex dtype can require gradients")
            self._requires_grad = True

    def __getattr__(self, name):
        """Called only for a name the tensor does not have: a torch method the stand-in lacks
        gets the not-available message, as an error that is also an AttributeError."""
        if name.startswith("_") or any(name in klass.__dict__ for klass in type(self).__mro__):
            # A private name, or a property that raised AttributeError itself: run it again so
            # its own error shows.
            return object.__getattribute__(self, name)
        raise MissingTensorMethod(name)

    @property
    def data(self):
        """The values as a new tensor over the same memory, outside autograd, as in torch: it
        has no history and does not require grad, and changing it in place changes this
        tensor without being recorded or counted as a change to a saved tensor."""
        return Tensor(self._data)

    @data.setter
    def data(self, value):
        """Make this tensor hold another tensor's values (its memory, shape and dtype)."""
        if not isinstance(value, Tensor):
            raise TypeError(f"Variable data has to be a tensor, but got {_type_name(value)}")
        if self.requires_grad and value._data.dtype.kind != "f":
            raise RuntimeError("data set to a tensor that requires gradients must be floating point or complex dtype")
        self._data = value._data

    @property
    def shape(self):
        return Size(self._data.shape)

    @property
    def dtype(self):
        return _dtype.of(self._data.dtype)

    @property
    def device(self):
        return CPU

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def T(self):
        return self.permute(*reversed(range(self._data.ndim)))

    def size(self, dim=None):
        return self.shape if dim is None else self._data.shape[dim_index(dim, self._data.ndim)]

    def dim(self):
        return self._data.ndim

    def numel(self):
        return self._data.size

    nelement = numel

    def element_size(self):
        return self._data.itemsize

    def is_floating_point(self):
        return self._data.dtype.kind == "f"

    def cpu(self):
        return self

    def cuda(self, device=None):
        raise RuntimeError(NO_CUDA)

    def item(self):
        if self._data.size != 1:
            raise RuntimeError(f"a Tensor with {self._data.size} elements cannot be converted to Scalar")
        return self._data.item()

    def tolist(self):
        return self._data.tolist()

    def numpy(self):
        if self.requires_grad:
            raise RuntimeError("Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.")
        return self._data

    def __array__(self, dtype=None, copy=None):
        data = self.numpy()
        return data if dtype is None else data.astype(dtype)

    def _scalar(self):
        if self._data.size != 1:
            raise ValueError("only one element tensors can be converted to Python scalars")
        return self._data.item()

    def __float__(self):
        return float(self._scalar())

    def __int__(self):
        return int(self._scalar())

    def __index__(self):
        if self._data.size != 1 or self._data.dtype.kind not in "iu":
            raise TypeError("only integer tensors of a single element can be converted to an index")
        return self._data.item()

    def __bool__(self):
        if self._data.size != 1:
            many = "more than one value" if self._data.size else "no values"
            raise RuntimeError(f"Boolean value of Tensor with {many} is ambiguous")
        return bool(self._data.item())

    def __len__(self):
        if self._data.ndim == 0:
            raise TypeError("len() of a 0-d tensor")
        return self._data.shape[0]

    def __iter__(self):
        if self._data.ndim == 0:
            raise TypeError("iteration over a 0-d tensor")
        return (self[i] for i in range(self._data.shape[0]))

    def __format__(self, spec):
        if self._data.ndim == 0:
            return format(self._data.item(), spec)
        return object.__format__(self, spec)

    def __repr__(self):
        return tensor_repr(self)

    def __hash__(self):
        return id(self)  # by identity, because == compares elementwise
