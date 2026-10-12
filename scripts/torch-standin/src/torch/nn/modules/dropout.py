# Ported from PyTorch (torch/nn/modules/dropout.py), BSD-3-Clause. See THIRD_PARTY_NOTICES.md.
"""Dropout: zeros random values while training, and does nothing in eval mode."""

from .. import functional as F
from .module import Module


class Dropout(Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p, self.inplace = p, inplace

    def forward(self, input):
        return F.dropout(input, self.p, self.training, self.inplace)

    def extra_repr(self):
        return f"p={self.p}, inplace={self.inplace}"
