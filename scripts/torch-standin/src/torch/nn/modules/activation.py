# Ported from PyTorch (torch/nn/modules/activation.py), BSD-3-Clause. See THIRD_PARTY_NOTICES.md.
"""Activation layers: each calls its function in torch.nn.functional."""

from .. import functional as F
from .module import Module


def _inplace_text(inplace):
    return "inplace=True" if inplace else ""


class ReLU(Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.relu(input, inplace=self.inplace)

    def extra_repr(self):
        return _inplace_text(self.inplace)


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01, inplace=False):
        super().__init__()
        self.negative_slope, self.inplace = negative_slope, inplace

    def forward(self, input):
        return F.leaky_relu(input, self.negative_slope, self.inplace)

    def extra_repr(self):
        return f"negative_slope={self.negative_slope}" + (", inplace=True" if self.inplace else "")


class ELU(Module):
    def __init__(self, alpha=1.0, inplace=False):
        super().__init__()
        self.alpha, self.inplace = alpha, inplace

    def forward(self, input):
        return F.elu(input, self.alpha, self.inplace)

    def extra_repr(self):
        return f"alpha={self.alpha}" + (", inplace=True" if self.inplace else "")


class GELU(Module):
    def __init__(self, approximate="none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, input):
        return F.gelu(input, approximate=self.approximate)

    def extra_repr(self):
        return f"approximate={self.approximate!r}"


class SiLU(Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.silu(input, inplace=self.inplace)

    def extra_repr(self):
        return _inplace_text(self.inplace)


class Tanh(Module):
    def forward(self, input):
        return F.tanh(input)


class Sigmoid(Module):
    def forward(self, input):
        return F.sigmoid(input)


class Softmax(Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.softmax(input, self.dim, _stacklevel=5)

    def extra_repr(self):
        return f"dim={self.dim}"
