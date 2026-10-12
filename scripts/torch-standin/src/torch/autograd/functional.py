"""torch.autograd.functional.jacobian and hessian, built on autograd.grad one output value
at a time."""

import numpy as np

from .. import _autograd
from .._missing import module_getattr
from .._shape import reshape, stack
from .._tensor import Tensor

__getattr__ = module_getattr(__name__, "torch.autograd.functional")


def _prepared(inputs, create_graph):
    """Each input as a tensor that requires grad, so the function's outputs can be
    differentiated with respect to it (a fresh leaf, or a view that keeps its graph)."""
    prepared = []
    for value in inputs:
        if not isinstance(value, Tensor):
            raise TypeError(f"The inputs given to jacobian must be either a Tensor or a tuple of Tensors but got {type(value).__name__}")
        if create_graph and _autograd.requires_grad(value):
            prepared.append(reshape(value, *value.shape))
        else:
            prepared.append(_autograd.detach(value).requires_grad_())
    return tuple(prepared)


def jacobian(func, inputs, create_graph=False, strict=False, vectorize=False, strategy="reverse-mode"):
    """d func(inputs) / d inputs. For one input and one output tensor the result has shape
    output.shape + input.shape; tuples of inputs or outputs give tuples of those blocks."""
    one_input = not isinstance(inputs, tuple)
    with _autograd.enable_grad():
        inputs = _prepared((inputs,) if one_input else inputs, create_graph)
        outputs = func(*inputs)
    one_output = not isinstance(outputs, tuple)
    outputs = (outputs,) if one_output else outputs
    blocks = []
    for out in outputs:
        if not isinstance(out, Tensor):
            raise TypeError(f"The outputs of the user-provided function given to jacobian must be either a Tensor or a tuple of Tensors but got {type(out).__name__}")
        rows = [[] for _ in inputs]
        flat = reshape(out, -1)
        for k in range(out._data.size):
            if _autograd.requires_grad(out):
                with _autograd.enable_grad():
                    grads = _autograd.grad(flat[k], inputs, retain_graph=True, create_graph=create_graph, allow_unused=True)
            else:
                grads = (None,) * len(inputs)
            for row, value, g in zip(rows, inputs, grads):
                row.append(Tensor(np.zeros_like(value._data)) if g is None else g)
        block = []
        for row, value in zip(rows, inputs):
            if row:
                jac = stack(row)
            else:
                jac = Tensor(np.zeros((0,) + value._data.shape, dtype=value._data.dtype))
            jac = reshape(jac, *(tuple(out.shape) + tuple(value.shape)))
            block.append(jac if create_graph else _autograd.detach(jac))
        blocks.append(block[0] if one_input else tuple(block))
    return blocks[0] if one_output else tuple(blocks)


def hessian(func, inputs, create_graph=False, strict=False, vectorize=False, outer_jacobian_strategy="reverse-mode"):
    """The second derivatives of a function with a single output value: shape
    input.shape + input.shape, or a tuple of tuples of blocks for a tuple of inputs."""
    one_input = not isinstance(inputs, tuple)

    def scalar(*values):
        out = func(*values)
        if not isinstance(out, Tensor) or out._data.size != 1:
            raise RuntimeError("The Tensor returned by the function given to hessian should contain a single element")
        return reshape(out, ())

    def gradient(*values):
        return jacobian(scalar, values[0] if one_input else values, create_graph=True)

    return jacobian(gradient, inputs, create_graph=create_graph)
