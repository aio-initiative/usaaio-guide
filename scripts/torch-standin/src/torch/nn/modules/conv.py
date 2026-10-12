# Ported from PyTorch (torch/nn/modules/conv.py), BSD-3-Clause. See THIRD_PARTY_NOTICES.md.
"""Conv2d and ConvTranspose2d."""

import math

from ... import _creation
from ..._missing import not_available
from .. import functional as F
from .. import init
from ..functional import _pair
from ..parameter import Parameter
from .module import Module

_PADDING_MODES = {"zeros", "reflect", "replicate", "circular"}


class _ConvNd(Module):
    """What Conv2d and ConvTranspose2d share: their settings, weight, bias, start and repr."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed,
                 output_padding, groups, bias, padding_mode, device, dtype):
        super().__init__()
        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        if isinstance(padding, str):
            if padding not in ("same", "valid"):
                raise ValueError(f"Invalid padding string {padding!r}, should be one of {{'same', 'valid'}}")
            if padding == "same" and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")
        if padding_mode not in _PADDING_MODES:
            raise ValueError(
                f"padding_mode must be one of {{'zeros', 'reflect', 'replicate', 'circular'}}, but got padding_mode='{padding_mode}'"
            )
        if groups != 1:
            raise not_available(f"torch.nn.{type(self).__name__} with groups={groups}")
        self.in_channels, self.out_channels, self.kernel_size = in_channels, out_channels, kernel_size
        self.stride, self.padding, self.dilation, self.transposed = stride, padding, dilation, transposed
        self.output_padding, self.groups, self.padding_mode = output_padding, groups, padding_mode
        if isinstance(padding, str):
            # For padding="same" with a padding_mode: total padding d * (k - 1), the extra one
            # (for an even kernel) on the right, in F.pad's last-dimension-first order.
            totals = [d * (k - 1) for d, k in zip(dilation, kernel_size)]
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == "same":
                for i, total in zip(range(len(kernel_size) - 1, -1, -1), totals):
                    self._reversed_padding_repeated_twice[2 * i] = total // 2
                    self._reversed_padding_repeated_twice[2 * i + 1] = total - total // 2
        else:
            self._reversed_padding_repeated_twice = [p for p in reversed(padding) for _ in range(2)]
        shape = (in_channels, out_channels // groups) if transposed else (out_channels, in_channels // groups)
        self.weight = Parameter(_creation.empty((*shape, *kernel_size), dtype=dtype, device=device))
        if bias:
            self.bias = Parameter(_creation.empty(out_channels, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        """torch's start: kaiming_uniform_ with a = sqrt(5) for the weight, and a bias
        uniform on +-1 / sqrt(fan_in)."""
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}"
        if self.padding != (0,) * len(self.padding):
            s += f", padding={self.padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += f", dilation={self.dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += f", output_padding={self.output_padding}"
        if self.groups != 1:
            s += f", groups={self.groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += f", padding_mode={self.padding_mode}"
        return s


class Conv2d(_ConvNd):
    """2D convolution over (N, C, H, W) images, weight of shape (out, in, kh, kw)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode="zeros", device=None, dtype=None):
        padding = padding if isinstance(padding, str) else _pair(padding)
        super().__init__(in_channels, out_channels, _pair(kernel_size), _pair(stride), padding, _pair(dilation),
                         False, (0, 0), groups, bias, padding_mode, device, dtype)

    def forward(self, input):
        if self.padding_mode != "zeros":
            padded = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            return F.conv2d(padded, self.weight, self.bias, self.stride, (0, 0), self.dilation, self.groups)
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ConvTranspose2d(_ConvNd):
    """Transposed convolution (upsampling), weight of shape (in, out, kh, kw)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, padding_mode="zeros", device=None, dtype=None):
        if padding_mode != "zeros":
            raise ValueError(f'Only "zeros" padding mode is supported for {type(self).__name__}')
        super().__init__(in_channels, out_channels, _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation),
                         True, _pair(output_padding), groups, bias, padding_mode, device, dtype)

    def _output_padding(self, input, output_size):
        """The output_padding that makes the output exactly output_size (its last two)."""
        if output_size is None:
            return self.output_padding
        size = list(output_size)[-2:]
        extra = []
        for d, want in enumerate(size):
            have = input.shape[d - 2]
            smallest = (have - 1) * self.stride[d] - 2 * self.padding[d] + self.dilation[d] * (self.kernel_size[d] - 1) + 1
            if not smallest <= want < smallest + self.stride[d]:
                raise ValueError(
                    f"requested an output size of {size}, but valid sizes range from {smallest} to "
                    f"{smallest + self.stride[d] - 1} (for an input of {list(input.shape[-2:])})"
                )
            extra.append(want - smallest)
        return tuple(extra)

    def forward(self, input, output_size=None):
        extra = self._output_padding(input, output_size)
        return F.conv_transpose2d(input, self.weight, self.bias, self.stride, self.padding, extra, self.groups, self.dilation)
