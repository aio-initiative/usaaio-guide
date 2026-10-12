# Ported from PyTorch (torch/nn/modules/linear.py), BSD-3-Clause. See THIRD_PARTY_NOTICES.md.
"""Linear and Identity."""

import math

from ... import _creation
from .. import functional as F
from .. import init
from ..parameter import Parameter
from .module import Module


class Identity(Module):
    """Returns its input; any arguments are accepted and ignored."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input


class Linear(Module):
    """y = x @ weight.T + bias, with weight of shape (out_features, in_features)."""

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = Parameter(_creation.empty((out_features, in_features), dtype=dtype, device=device))
        if bias:
            self.bias = Parameter(_creation.empty(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        """torch's start: kaiming_uniform_ with a = sqrt(5) (a bound of 1 / sqrt(in_features))
        for the weight, and the same bound for the bias."""
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class NonDynamicallyQuantizableLinear(Linear):
    """The Linear that MultiheadAttention uses for its output projection (the name shows in its repr)."""
