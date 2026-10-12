# Ported from PyTorch (torch/nn/init.py), BSD-3-Clause. See THIRD_PARTY_NOTICES.md.
"""torch.nn.init: fill a weight tensor in place, outside autograd.

The formulas (gains, fans, bounds) are torch's, so weights get torch's scales; the draws come
from the stand-in's own generator (torch.manual_seed), so they differ from torch's values.
"""

import math
import warnings

from .. import _autograd
from .._missing import module_getattr as _module_getattr
from .._random import _stream
from .._tensor import Tensor


def _fill(tensor, draw):
    """Write draw(shape, numpy dtype) into `tensor` under no_grad and return the tensor."""
    array = tensor._data
    with _autograd.no_grad():
        tensor.copy_(Tensor(draw(array.shape, array.dtype)))
    return tensor


def _empty(tensor):
    if tensor._data.size == 0:
        warnings.warn("Initializing zero-element tensors is a no-op", stacklevel=3)
        return True
    return False


def calculate_gain(nonlinearity, param=None):
    """The factor that keeps the variance of activations steady through `nonlinearity`."""
    linear = ("linear", "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d")
    if nonlinearity in linear or nonlinearity == "sigmoid":
        return 1
    if nonlinearity == "tanh":
        return 5.0 / 3
    if nonlinearity == "relu":
        return math.sqrt(2.0)
    if nonlinearity == "leaky_relu":
        if param is None:
            slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, (int, float)):
            slope = param
        else:
            raise ValueError(f"negative_slope {param} not a valid number")
        return math.sqrt(2.0 / (1 + slope**2))
    if nonlinearity == "selu":
        return 3.0 / 4
    raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


def _calculate_fan_in_and_fan_out(tensor):
    """(fan_in, fan_out): dimension 1 and dimension 0 times the receptive field (the rest)."""
    shape = tensor._data.shape
    if len(shape) < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    receptive = math.prod(shape[2:])
    return shape[1] * receptive, shape[0] * receptive


def _fan(tensor, mode):
    if mode not in ("fan_in", "fan_out"):
        raise ValueError(f"Mode {mode} not supported, please use one of fan_in, fan_out")
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out


def uniform_(tensor, a=0.0, b=1.0, generator=None):
    return _fill(tensor, lambda shape, dtype: _stream(generator).uniform(a, b, shape).astype(dtype))


def normal_(tensor, mean=0.0, std=1.0, generator=None):
    return _fill(tensor, lambda shape, dtype: _stream(generator).normal(mean, std, shape).astype(dtype))


def constant_(tensor, val):
    with _autograd.no_grad():
        return tensor.fill_(val)


def ones_(tensor):
    return constant_(tensor, 1.0)


def zeros_(tensor):
    with _autograd.no_grad():
        return tensor.zero_()


def xavier_uniform_(tensor, gain=1.0, generator=None):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    bound = math.sqrt(3.0) * gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return uniform_(tensor, -bound, bound, generator)


def xavier_normal_(tensor, gain=1.0, generator=None):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return normal_(tensor, 0.0, gain * math.sqrt(2.0 / float(fan_in + fan_out)), generator)


def kaiming_uniform_(tensor, a=0, mode="fan_in", nonlinearity="leaky_relu", generator=None):
    """Uniform on [-bound, bound] with bound = gain * sqrt(3 / fan). Linear and Conv2d start
    with a = sqrt(5), which makes the bound 1 / sqrt(fan_in)."""
    if _empty(tensor):
        return tensor
    bound = math.sqrt(3.0) * calculate_gain(nonlinearity, a) / math.sqrt(_fan(tensor, mode))
    return uniform_(tensor, -bound, bound, generator)


def kaiming_normal_(tensor, a=0, mode="fan_in", nonlinearity="leaky_relu", generator=None):
    """Normal with std = gain / sqrt(fan) (He initialisation)."""
    if _empty(tensor):
        return tensor
    return normal_(tensor, 0.0, calculate_gain(nonlinearity, a) / math.sqrt(_fan(tensor, mode)), generator)


__getattr__ = _module_getattr(__name__, "torch.nn.init")
