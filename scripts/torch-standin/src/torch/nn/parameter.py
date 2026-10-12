"""torch.nn.Parameter: a tensor that a module registers as one of its weights."""

import numpy as np

from .. import _autograd
from .._tensor import Tensor


class Parameter(Tensor):
    """A leaf tensor that requires grad by default. Like torch's, it shares the memory (and
    the version counter) of the tensor it wraps, and assigning one to a module attribute
    registers it with the module."""

    def __init__(self, data=None, requires_grad=True):
        if data is None:
            data = Tensor(np.empty(0, dtype=np.float32))
        if not isinstance(data, Tensor):
            raise AttributeError(f"'{type(data).__name__}' object has no attribute 'detach'")
        if requires_grad and data._data.dtype.kind != "f":
            raise RuntimeError("Only Tensors of floating point and complex dtype can require gradients")
        self._data = data._data
        self._vc = _autograd.counter(data)
        self._requires_grad = bool(requires_grad)

    def __repr__(self):
        return "Parameter containing:\n" + super().__repr__()

    def __deepcopy__(self, memo):
        """A new Parameter with a copy of the values, and no .grad, as torch's deepcopy gives."""
        if id(self) not in memo:
            memo[id(self)] = type(self)(Tensor(self._data.copy()), self._requires_grad)
        return memo[id(self)]
