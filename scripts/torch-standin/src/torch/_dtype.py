"""Tensor dtypes and torch's type promotion, which differs from NumPy's."""

import builtins

import numpy as np


class dtype:
    """A torch dtype. `np` is the NumPy dtype that stores it."""

    def __init__(self, name, numpy_type, legacy):
        self.name = name
        self.np = np.dtype(numpy_type)
        self.legacy = legacy  # the name in torch's error messages and in Tensor.type()
        self.is_floating_point = self.np.kind == "f"
        self.itemsize = self.np.itemsize

    def __repr__(self):
        return f"torch.{self.name}"

    def __reduce__(self):
        return self.name  # pickles as the module-level name, so identity survives


bool = dtype("bool", np.bool_, "Bool")
uint8 = dtype("uint8", np.uint8, "Byte")
int8 = dtype("int8", np.int8, "Char")
int16 = short = dtype("int16", np.int16, "Short")
int32 = int = dtype("int32", np.int32, "Int")
int64 = long = dtype("int64", np.int64, "Long")
float32 = float = dtype("float32", np.float32, "Float")
float64 = double = dtype("float64", np.float64, "Double")

ALL = (bool, uint8, int8, int16, int32, int64, float32, float64)
_BY_NUMPY = {d.np: d for d in ALL}
# dtype=float, dtype=int and dtype=bool mean what they mean in torch.
_BY_PYTHON = {builtins.float: float64, builtins.int: int64, builtins.bool: bool}

BOOL, INT64, FLOAT32 = bool.np, int64.np, float32.np


def of(numpy_dtype):
    """The torch dtype stored as this NumPy dtype."""
    try:
        return _BY_NUMPY[numpy_dtype]
    except KeyError:
        raise TypeError(
            f"can't convert a NumPy array of type {numpy_dtype}. The supported types are: "
            + ", ".join(d.name for d in ALL)
        ) from None


def as_numpy(value):
    """The NumPy dtype for a torch dtype, a Python type (float, int, bool) or a NumPy dtype."""
    if isinstance(value, dtype):
        return value.np
    if value in _BY_PYTHON:
        return _BY_PYTHON[value].np
    return of(np.dtype(value)).np


def promote(a, b):
    """torch.promote_types on NumPy dtypes: a float wins over any integer, whatever its width."""
    if a == b:
        return a
    if a.kind == "f" or b.kind == "f":
        if a.kind == b.kind:
            return a if a.itemsize >= b.itemsize else b
        return a if a.kind == "f" else b
    return np.promote_types(a, b)


def promote_types(type1, type2):
    return of(promote(type1.np, type2.np))


def _combine(higher, lower):
    if higher is None or lower is None:
        return lower if higher is None else higher
    if higher.kind == "b" or (lower.kind == "f" and higher.kind != "f"):
        return promote(higher, lower)
    return higher


def result_type(dimensioned, zero_dim, scalar):
    """The dtype of an operation's result, from the promoted dtype of each kind of operand.

    Tensors with dimensions decide; a 0-d tensor or a Python number only matters when it is of
    a higher category (bool < integer < float), so int_tensor * 2.5 is float32 and
    float32_tensor * 2 or float32_tensor * float64_scalar_tensor stays float32. Each argument
    is a NumPy dtype or None.
    """
    return _combine(dimensioned, _combine(zero_dim, scalar))
