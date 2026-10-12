# Partly ported from PyTorch (torch/nn/modules/activation.py: MultiheadAttention; torch/nn/functional.py:
# multi_head_attention_forward, _canonical_mask), BSD-3-Clause. See THIRD_PARTY_NOTICES.md.
"""MultiheadAttention, with torch's parameter names and shapes (in_proj_weight, in_proj_bias,
out_proj), so a real layer's state_dict loads into it, and torch's order of operations."""

import math

import numpy as np

from ... import _creation
from ..._missing import not_available
from ..._ops import add, constant, data_of, mul, where
from ..._reduce import mean
from ..._linalg import matmul
from ..._shape import expand, reshape, squeeze, transpose, unsqueeze
from ..._tensor import Tensor
from .. import functional as F
from .. import init
from ..parameter import Parameter
from .linear import NonDynamicallyQuantizableLinear
from .module import Module


def _canonical_mask(mask, name, dtype):
    """A bool mask (True: may not attend) as a float mask of 0 and -inf; a float mask as it is."""
    if mask is None:
        return None
    m = data_of(mask)
    if m.dtype == np.bool_:
        return Tensor(np.where(m, -np.inf, 0).astype(dtype.np))
    if m.dtype.kind != "f":
        raise AssertionError(f"only bool and floating types of {name} are supported")
    return mask


def _safe_softmax(scores):
    """softmax over the last dimension, with zeros (and a zero gradient) for a row that is all
    -inf, as torch's scaled_dot_product_attention gives."""
    masked = np.all(scores._data == -np.inf, axis=-1, keepdims=True)
    if not masked.any():
        return F.softmax(scores, dim=-1)
    rows = constant(masked)
    return where(rows, 0.0, F.softmax(where(rows, 0.0, scores), dim=-1))


def _projections(query, key, value, weight, bias, embed_dim):
    """q, k, v: one projection for self-attention, three when query, key and value differ."""
    if query is key and key is value:
        packed = F.linear(query, weight, bias)
        return tuple(packed[..., i * embed_dim : (i + 1) * embed_dim] for i in range(3))
    weights = [weight[i * embed_dim : (i + 1) * embed_dim] for i in range(3)]
    biases = [None] * 3 if bias is None else [bias[i * embed_dim : (i + 1) * embed_dim] for i in range(3)]
    return tuple(F.linear(x, w, b) for x, w, b in zip((query, key, value), weights, biases))


class MultiheadAttention(Module):
    """Attention with num_heads heads of size embed_dim // num_heads. Inputs are (L, N, E), or
    (N, L, E) with batch_first, or one unbatched (L, E) sequence. forward returns the output
    and the attention weights (averaged over the heads unless average_attn_weights=False)."""

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None):
        super().__init__()
        if (kdim is not None and kdim != embed_dim) or (vdim is not None and vdim != embed_dim):
            raise not_available("torch.nn.MultiheadAttention with kdim or vdim")
        for name, value in (("add_bias_kv", add_bias_kv), ("add_zero_attn", add_zero_attn)):
            if value:
                raise not_available(f"torch.nn.MultiheadAttention with {name}")
        self.embed_dim, self.kdim, self.vdim, self._qkv_same_embed_dim = embed_dim, embed_dim, embed_dim, True
        self.num_heads, self.dropout, self.batch_first = num_heads, dropout, batch_first
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise AssertionError("embed_dim must be divisible by num_heads")
        factory = {"dtype": dtype, "device": device}
        self.in_proj_weight = Parameter(_creation.empty((3 * embed_dim, embed_dim), **factory))
        for name in ("q_proj_weight", "k_proj_weight", "v_proj_weight"):
            self.register_parameter(name, None)
        if bias:
            self.in_proj_bias = Parameter(_creation.empty(3 * embed_dim, **factory))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory)
        self.bias_k = self.bias_v = None
        self.add_zero_attn = add_zero_attn
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            init.constant_(self.in_proj_bias, 0.0)
            init.constant_(self.out_proj.bias, 0.0)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None,
                average_attn_weights=True, is_causal=False):
        batched = query.dim() == 3
        key_padding_mask = _canonical_mask(key_padding_mask, "key_padding_mask", query.dtype)
        attn_mask = _canonical_mask(attn_mask, "attn_mask", query.dtype)
        if self.batch_first and batched:
            if key is value:
                if query is key:
                    query = key = value = transpose(query, 1, 0)
                else:
                    query, key = transpose(query, 1, 0), transpose(key, 1, 0)
                    value = key
            else:
                query, key, value = (transpose(x, 1, 0) for x in (query, key, value))
        out, weights = self._attend(
            query, key, value, batched, key_padding_mask, need_weights, attn_mask, average_attn_weights, is_causal
        )
        if self.batch_first and batched:
            out = transpose(out, 1, 0)
        return out, weights

    def _attend(self, query, key, value, batched, key_padding_mask, need_weights, attn_mask, average, is_causal):
        """torch's multi_head_attention_forward on (L, N, E) inputs."""
        if not batched:
            query, key, value = unsqueeze(query, 1), unsqueeze(key, 1), unsqueeze(value, 1)
            if key_padding_mask is not None:
                key_padding_mask = unsqueeze(key_padding_mask, 0)
        tgt_len, bsz, embed_dim = query.shape
        src_len = key.shape[0]
        heads, head_dim = self.num_heads, self.head_dim
        if is_causal and attn_mask is None:
            raise RuntimeError(
                "Need attn_mask if specifying the is_causal hint. You may use the Transformer module method "
                "`generate_square_subsequent_mask` to create this mask."
            )
        if is_causal and key_padding_mask is None and not need_weights:
            # As torch: the hint alone decides, so the mask is the causal one.
            causal = np.triu(np.ones((tgt_len, src_len), dtype=bool), 1)
            attn_mask = Tensor(np.where(causal, -np.inf, 0).astype(query.dtype.np))
        q, k, v = _projections(query, key, value, self.in_proj_weight, self.in_proj_bias, embed_dim)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                if tuple(attn_mask.shape) != (tgt_len, src_len):
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {(tgt_len, src_len)}."
                    )
                attn_mask = unsqueeze(attn_mask, 0)
            elif attn_mask.dim() == 3:
                if tuple(attn_mask.shape) != (bsz * heads, tgt_len, src_len):
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {(bsz * heads, tgt_len, src_len)}."
                    )
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")
        q = transpose(reshape(q, tgt_len, bsz * heads, head_dim), 0, 1)
        k = transpose(reshape(k, src_len, bsz * heads, head_dim), 0, 1)
        v = transpose(reshape(v, src_len, bsz * heads, head_dim), 0, 1)
        if key_padding_mask is not None:
            if tuple(key_padding_mask.shape) != (bsz, src_len):
                raise AssertionError(f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}")
            padding = reshape(expand(reshape(key_padding_mask, bsz, 1, 1, src_len), -1, heads, -1, -1), bsz * heads, 1, src_len)
            attn_mask = padding if attn_mask is None else add(attn_mask, padding)
        scores = matmul(mul(q, math.sqrt(1.0 / float(head_dim))), transpose(k, -2, -1))
        if attn_mask is not None:
            scores = add(attn_mask, scores)
        # Without weights torch runs scaled_dot_product_attention, whose softmax gives a fully
        # masked row zeros (so its output is just the projection's bias); with weights, nan.
        weights = F.softmax(scores, dim=-1) if need_weights else _safe_softmax(scores)
        if self.training and self.dropout > 0.0:
            weights = F.dropout(weights, p=self.dropout)
        out = reshape(transpose(matmul(weights, v), 0, 1), tgt_len * bsz, embed_dim)
        out = reshape(F.linear(out, self.out_proj.weight, self.out_proj.bias), tgt_len, bsz, embed_dim)
        if not need_weights:
            return (out if batched else squeeze(out, 1)), None
        weights = reshape(weights, bsz, heads, tgt_len, src_len)
        if average:
            weights = mean(weights, 1)
        if not batched:
            out, weights = squeeze(out, 1), squeeze(weights, 0)
        return out, weights
