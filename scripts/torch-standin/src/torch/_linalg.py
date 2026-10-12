"""Matrix products (matmul, mm, bmm, outer, einsum), distances (cdist) and meshgrid.

matmul and einsum are computed here, with their gradients; outer, cdist and meshgrid are
written with other operations, so their gradients follow from those.
"""

import warnings

import numpy as np

from . import _dtype
from ._index import put
from ._ops import data_of, for_grad_of, mul, numeric, result, sub
from ._reduce import distance
from ._shape import expand, reshape, squeeze, transpose, unsqueeze


def _legacy(x):
    return _dtype.of(x.dtype).legacy


@numeric
def matmul(input, other):
    """The matrix product, with NumPy's rules for vectors and batches (which are torch's)."""
    x, y = data_of(input), data_of(other)
    if x.ndim == 0 or y.ndim == 0:
        raise RuntimeError(f"both arguments to matmul need to be at least 1D, but they are {x.ndim}D and {y.ndim}D")
    if x.dtype != y.dtype:
        raise RuntimeError(f"expected m1 and m2 to have the same dtype, but got: {_legacy(x)} != {_legacy(y)}")
    if x.dtype.kind == "b":
        raise NotImplementedError("\"addmm_impl_cpu_\" not implemented for 'Bool'")
    try:
        out = np.matmul(x, y)
    except ValueError:
        raise RuntimeError(
            f"mat1 and mat2 shapes cannot be multiplied ({'x'.join(map(str, x.shape))} and "
            f"{'x'.join(map(str, y.shape))})"
        ) from None
    vector_x, vector_y = x.ndim == 1, y.ndim == 1
    name = {(1, 1): "DotBackward0", (2, 1): "MvBackward0", (2, 2): "MmBackward0"}.get((x.ndim, y.ndim), "MatmulBackward0")

    def as_matrices(g, a, b):
        """The product as one of matrices: a vector operand is a 1-row or 1-column matrix.
        An operand that was not saved (its partner needs no gradient) stays None."""
        if vector_x:
            g = unsqueeze(g, -2) if not vector_y else unsqueeze(g, 0)
            a = None if a is None else unsqueeze(a, 0)
        if vector_y:
            g = unsqueeze(g, -1)
            b = None if b is None else unsqueeze(b, 1)
        return g, a, b

    def grad_x(g, a, b):
        g, a, b = as_matrices(g, a, b)
        out = matmul(g, transpose(b, -1, -2))
        return squeeze(out, -2) if vector_x else out

    def grad_y(g, a, b):
        g, a, b = as_matrices(g, a, b)
        out = matmul(transpose(a, -1, -2), g)
        return squeeze(out, -1) if vector_y else out

    saved = (for_grad_of(input, other), for_grad_of(other, input))
    return result(out, input, other, name=name, saved=saved, grads=(grad_x, grad_y))


def mm(input, mat2):
    if data_of(input).ndim != 2 or data_of(mat2).ndim != 2:
        raise RuntimeError("self must be a matrix")
    return matmul(input, mat2)


def bmm(input, mat2):
    """Matrix products of two batches of matrices, of shapes (b, n, m) and (b, m, p)."""
    if data_of(input).ndim != 3 or data_of(mat2).ndim != 3:
        raise RuntimeError("batch1 must be a 3D tensor")
    return matmul(input, mat2)


def outer(input, vec2):
    """out[i, j] = input[i] * vec2[j]."""
    if data_of(input).ndim != 1 or data_of(vec2).ndim != 1:
        raise RuntimeError(f"outer: Expected 1-D argument self, but got {data_of(input).ndim}-D")
    return mul(unsqueeze(input, 1), vec2)


@numeric
def einsum(equation, *operands):
    """einsum("ij,jk->ik", a, b), or with the operands in one list as torch also accepts."""
    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        operands = tuple(operands[0])
    arrays = [data_of(operand) for operand in operands]
    for array in arrays[1:]:
        if array.dtype != arrays[0].dtype:
            raise RuntimeError(f"expected scalar type {_legacy(arrays[0])} but found {_legacy(array)}")
    equation = equation.replace(" ", "")
    # optimize lets NumPy hand products to BLAS instead of looping in C, much faster for big operands.
    out = np.einsum(equation, *arrays, optimize=True)
    inputs, output = _subscripts(equation, [a.ndim for a in arrays])
    sizes = {letter: n for letters, a in zip(inputs, arrays) for letter, n in zip(letters, a.shape)}
    grads = tuple(
        lambda g, *ops, k=k: _einsum_grad(g, ops, k, inputs, output, sizes) for k in range(len(operands))
    )
    return result(out, *operands, name="EinsumBackward0", saved=operands, grads=grads)


def _subscripts(equation, ndims):
    """The letters of each operand and of the output, with any ... written out as letters."""
    left, arrow, right = equation.partition("->")
    inputs = left.split(",")
    spare = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in equation]
    width = max([n - len(s.replace("...", "")) for s, n in zip(inputs, ndims) if "..." in s] or [0])
    ellipsis = "".join(spare[:width])
    inputs = [s.replace("...", ellipsis[width - (n - len(s.replace("...", ""))):]) for s, n in zip(inputs, ndims)]
    if arrow:
        output = right.replace("...", ellipsis)
    else:
        letters = "".join(inputs)
        output = ellipsis + "".join(sorted(c for c in set(letters) if letters.count(c) == 1 and c not in ellipsis))
    return inputs, output


def _einsum_grad(g, operands, k, inputs, output, sizes):
    """The gradient of operand k: an einsum of the output's gradient with the other operands.
    A letter only operand k has was summed away, so its gradient is spread back along it; a
    repeated letter (a diagonal) puts the gradient back on that diagonal."""
    mine = inputs[k]
    others = [s for i, s in enumerate(inputs) if i != k]
    unique = "".join(dict.fromkeys(mine))
    found = "".join(c for c in unique if c in output or any(c in s for s in others))
    terms = ",".join([output, *others]) + "->" + found
    rest = [op for i, op in enumerate(operands) if i != k]
    part = einsum(terms, g, *rest)
    if found != unique:  # letters summed away within this operand alone
        part = reshape(part, *[sizes[c] if c in found else 1 for c in unique])
        part = expand(part, *[sizes[c] for c in unique])
    if unique == mine:
        return part
    shape = tuple(sizes[c] for c in mine)
    key = tuple(np.arange(sizes[c]).reshape([-1 if u == c else 1 for u in unique]) for c in mine)
    return put(shape, None, key + (Ellipsis,), part)


def cdist(x1, x2, p=2.0):
    """The p-norm distance between each row of x1 and each row of x2 (batches allowed)."""
    a, b = data_of(x1), data_of(x2)
    for name, x in (("X1", a), ("X2", b)):
        if x.dtype.kind != "f":
            raise RuntimeError(f"cdist only supports floating-point dtypes, {name} got: {_legacy(x)}")
        if x.ndim < 2:
            raise RuntimeError(f"cdist only supports at least 2D tensors, {name} got: {x.ndim}D")
    if a.shape[-1] != b.shape[-1]:
        raise RuntimeError(
            f"X1 and X2 must have the same number of columns. X1: {a.shape[-1]} X2: {b.shape[-1]}"
        )
    return distance(sub(unsqueeze(x1, -2), unsqueeze(x2, -3)), p)


def meshgrid(*tensors, indexing=None):
    """Grids from 1-d tensors: with indexing="ij" the i-th output varies along dimension i;
    with "xy" the first two swap, as in NumPy. The outputs are views of the inputs."""
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tuple(tensors[0])
    if indexing is None:
        warnings.warn(
            "torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.",
            stacklevel=2,
        )
        indexing = "ij"
    if indexing not in ("ij", "xy"):
        raise RuntimeError(f"torch.meshgrid: indexing must be one of \"xy\" or \"ij\", but received: {indexing}")
    if len({data_of(t).dtype for t in tensors}) > 1:
        raise RuntimeError("meshgrid expects all tensors to have the same dtype")
    axes = list(range(len(tensors)))
    if indexing == "xy" and len(tensors) > 1:
        axes[0], axes[1] = 1, 0
    shape = [0] * len(tensors)
    for axis, t in zip(axes, tensors):
        shape[axis] = data_of(t).size
    grids = []
    for axis, t in zip(axes, tensors):
        column = [1] * len(tensors)
        column[axis] = -1
        grids.append(expand(reshape(t, *column), *shape))
    return tuple(grids)
