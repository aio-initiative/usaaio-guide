"""Gives Tensor its methods and operators.

Almost every tensor method is the torch function of the same name with the tensor as the
first argument (x.exp() is torch.exp(x)), so the functions are attached to the class here
instead of being written out twice.
"""

import numpy as np

from . import _autograd, _compare, _dtype, _index, _inplace, _linalg, _ops, _reduce, _shape, _sort
from ._tensor import Tensor


def _bind(module, names):
    for name in names.split():
        setattr(Tensor, name, getattr(module, name))


_bind(_ops, "add sub mul div pow neg abs exp log sqrt sin cos tanh sigmoid relu round")
_bind(_ops, "clamp clip clamp_min masked_fill clone to")
_bind(_compare, "eq ne lt le gt ge isnan isfinite allclose isclose equal")
_bind(_inplace, "add_ sub_ mul_ div_ zero_ fill_ copy_ scatter_add_ fill_diagonal_")
_bind(_index, "gather scatter scatter_add")
_bind(_shape, "reshape view flatten ravel squeeze unsqueeze permute transpose expand flip roll")
_bind(_shape, "triu tril diag diff split chunk unbind contiguous is_contiguous")
_bind(_reduce, "sum mean amax amin max min argmax argmin all any var std norm trace")
_bind(_reduce, "cumsum cumprod logsumexp softmax log_softmax")
_bind(_sort, "sort argsort topk median mode quantile unique bincount")
_bind(_linalg, "matmul mm bmm outer")
Tensor.backward = _autograd.tensor_backward
Tensor.detach = _autograd.detach
Tensor.requires_grad_ = _autograd.requires_grad_
Tensor.retain_grad = _autograd.retain_grad
Tensor.grad = property(_autograd.get_grad, _autograd.set_grad)
Tensor.grad_fn = property(_autograd.grad_fn)
Tensor.requires_grad = property(_autograd.requires_grad, _autograd.set_requires_grad)
Tensor.is_leaf = property(lambda self: _autograd.grad_fn(self) is None)
Tensor.retains_grad = property(lambda self: self._retains_grad and _autograd.grad_fn(self) is not None)
Tensor._version = property(lambda self: 0 if self._vc is None else self._vc[0])
Tensor.__getstate__ = _autograd.getstate
Tensor.__deepcopy__ = _autograd.deepcopy
Tensor.__getitem__ = _index.getitem
Tensor.__setitem__ = _index.setitem


def _where(self, condition, other):
    return _ops.where(condition, self, other)


Tensor.where = _where


def _caster(dtype):
    def convert(self):
        return _ops.cast(self, dtype.np)

    return convert


for _name in ("float", "double", "long", "int", "bool"):
    setattr(Tensor, _name, _caster(getattr(_dtype, _name)))


def _reflected(function):
    def reflected(self, other):
        return function(other, self)

    return reflected


_OPERATORS = {
    "add": _ops.add,
    "sub": _ops.sub,
    "mul": _ops.mul,
    "truediv": _ops.div,
    "floordiv": _ops.floor_divide,
    "mod": _ops.remainder,
    "pow": _ops.pow,
    "and": _compare.bitwise_and,
    "or": _compare.bitwise_or,
    "xor": _compare.bitwise_xor,
}
for _name, _function in _OPERATORS.items():
    setattr(Tensor, f"__{_name}__", _function)
    setattr(Tensor, f"__r{_name}__", _reflected(_function))

for _name in ("lt", "le", "gt", "ge"):
    setattr(Tensor, f"__{_name}__", getattr(_compare, _name))

_OPERANDS = (Tensor, np.ndarray, np.generic, bool, int, float)


def _eq(self, other):
    # tensor == None and tensor == "text" are plain False, as in torch, not an error.
    return _compare.eq(self, other) if isinstance(other, _OPERANDS) else NotImplemented


def _ne(self, other):
    return _compare.ne(self, other) if isinstance(other, _OPERANDS) else NotImplemented


Tensor.__eq__ = _eq
Tensor.__ne__ = _ne
Tensor.__matmul__ = _linalg.matmul
Tensor.__rmatmul__ = _reflected(_linalg.matmul)
Tensor.__neg__ = _ops.neg
Tensor.__abs__ = _ops.abs
Tensor.__invert__ = _compare.bitwise_not
Tensor.__pos__ = lambda self: self
Tensor.__iadd__ = _inplace.add_
Tensor.__isub__ = _inplace.sub_
Tensor.__imul__ = _inplace.mul_
Tensor.__itruediv__ = _inplace.div_
