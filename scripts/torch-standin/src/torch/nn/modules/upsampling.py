# Ported from PyTorch (torch/nn/modules/upsampling.py), BSD-3-Clause. See THIRD_PARTY_NOTICES.md.
"""Upsample: interpolate as a layer."""

from .. import functional as F
from .module import Module


class Upsample(Module):
    def __init__(self, size=None, scale_factor=None, mode="nearest", align_corners=None, recompute_scale_factor=None):
        super().__init__()
        self.name = type(self).__name__
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode, self.align_corners, self.recompute_scale_factor = mode, align_corners, recompute_scale_factor

    def forward(self, input):
        return F.interpolate(
            input, self.size, self.scale_factor, self.mode, self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
        )

    def extra_repr(self):
        info = f"scale_factor={self.scale_factor!r}" if self.scale_factor is not None else f"size={self.size!r}"
        return info + f", mode={self.mode!r}"
