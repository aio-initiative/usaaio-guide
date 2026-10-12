# Ported from PyTorch (torch/nn/modules/sparse.py), BSD-3-Clause. See THIRD_PARTY_NOTICES.md.
"""Embedding: a lookup table of vectors, one row per index."""

from ... import _autograd, _creation
from .. import functional as F
from .. import init
from ..parameter import Parameter
from .module import Module


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0,
                 scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None):
        super().__init__()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < num_embeddings, "Padding_idx must be within num_embeddings"
            elif padding_idx < 0:
                assert padding_idx >= -num_embeddings, "Padding_idx must be within num_embeddings"
                padding_idx = num_embeddings + padding_idx
        self.padding_idx, self.max_norm, self.norm_type = padding_idx, max_norm, norm_type
        self.scale_grad_by_freq, self.sparse = scale_grad_by_freq, sparse
        if _weight is None:
            self.weight = Parameter(_creation.empty((num_embeddings, embedding_dim), dtype=dtype, device=device))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], "Shape of weight does not match num_embeddings and embedding_dim"
            self.weight = Parameter(_weight, requires_grad=not _freeze)

    def reset_parameters(self):
        """Standard normal rows, and a zero row at padding_idx."""
        init.normal_(self.weight)
        if self.padding_idx is not None:
            with _autograd.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input):
        return F.embedding(input, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

    def extra_repr(self):
        s = f"{self.num_embeddings}, {self.embedding_dim}"
        if self.padding_idx is not None:
            s += f", padding_idx={self.padding_idx}"
        if self.max_norm is not None:
            s += f", max_norm={self.max_norm}"
        if self.norm_type != 2:
            s += f", norm_type={self.norm_type}"
        if self.scale_grad_by_freq is not False:
            s += f", scale_grad_by_freq={self.scale_grad_by_freq}"
        if self.sparse is not False:
            s += ", sparse=True"
        return s

    @classmethod
    def from_pretrained(cls, embeddings, freeze=True, padding_idx=None, max_norm=None, norm_type=2.0,
                        scale_grad_by_freq=False, sparse=False):
        """An Embedding whose rows are `embeddings` (frozen unless freeze=False)."""
        assert embeddings.dim() == 2, "Embeddings parameter is expected to be 2-dimensional"
        rows, cols = embeddings.shape
        return cls(rows, cols, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, _weight=embeddings, _freeze=freeze)
