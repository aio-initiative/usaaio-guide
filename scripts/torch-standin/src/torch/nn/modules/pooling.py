# Ported from PyTorch (torch/nn/modules/pooling.py), BSD-3-Clause. See THIRD_PARTY_NOTICES.md.
"""MaxPool2d, AvgPool2d and AdaptiveAvgPool2d."""

from .. import functional as F
from .module import Module


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride if stride is not None else kernel_size
        self.padding, self.dilation = padding, dilation
        self.return_indices, self.ceil_mode = return_indices, ceil_mode

    def forward(self, input):
        return F.max_pool2d(
            input, self.kernel_size, self.stride, self.padding, self.dilation,
            ceil_mode=self.ceil_mode, return_indices=self.return_indices,
        )

    def extra_repr(self):
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
            f"dilation={self.dilation}, ceil_mode={self.ceil_mode}"
        )


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride if stride is not None else kernel_size
        self.padding, self.ceil_mode = padding, ceil_mode
        self.count_include_pad, self.divisor_override = count_include_pad, divisor_override

    def forward(self, input):
        return F.avg_pool2d(
            input, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override
        )

    def extra_repr(self):
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, input):
        return F.adaptive_avg_pool2d(input, self.output_size)

    def extra_repr(self):
        return f"output_size={self.output_size}"
