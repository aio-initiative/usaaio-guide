# Ported from PyTorch (torch/nn/modules/flatten.py), BSD-3-Clause. See THIRD_PARTY_NOTICES.md.
"""Flatten: merges the dimensions start_dim to end_dim into one."""

from .module import Module


class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim, self.end_dim = start_dim, end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)

    def extra_repr(self):
        return f"start_dim={self.start_dim}, end_dim={self.end_dim}"
