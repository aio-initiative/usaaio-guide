# Ported from PyTorch (torch/nn/modules/batchnorm.py, normalization.py), BSD-3-Clause. See THIRD_PARTY_NOTICES.md.
"""BatchNorm1d, BatchNorm2d, LayerNorm and GroupNorm."""

from ... import _creation, _dtype
from .. import functional as F
from .. import init
from ..parameter import Parameter
from .module import Module


class _BatchNorm(Module):
    """Normalizes each channel by the batch's statistics in training (and keeps running
    averages of them), by the running averages in eval mode, as torch's BatchNorm does."""

    _dims = ()
    _version = 2  # the version that added num_batches_tracked

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 device=None, dtype=None, *, bias=True):
        super().__init__()
        self.num_features, self.eps, self.momentum = num_features, eps, momentum
        self.affine, self.track_running_stats = affine, track_running_stats
        factory = {"dtype": dtype, "device": device}
        if affine:
            self.weight = Parameter(_creation.empty(num_features, **factory))
            if bias:
                self.bias = Parameter(_creation.empty(num_features, **factory))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if track_running_stats:
            self.register_buffer("running_mean", _creation.zeros(num_features, **factory))
            self.register_buffer("running_var", _creation.ones(num_features, **factory))
            self.register_buffer("num_batches_tracked", _creation.tensor(0, dtype=_dtype.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, missing, unexpected, errors):
        """As torch: a state saved without version 2's metadata (a plain dict, or an old file)
        may lack num_batches_tracked; the layer then keeps its own count."""
        version = local_metadata.get("version")
        key = prefix + "num_batches_tracked"
        if (version is None or version < 2) and self.track_running_stats and key not in state_dict:
            state_dict[key] = self.num_batches_tracked
        super()._load_from_state_dict(state_dict, prefix, local_metadata, missing, unexpected, errors)

    def _check_input_dim(self, input):
        if input.dim() not in self._dims:
            raise ValueError(f"expected {self._dims_text} input (got {input.dim()}D input)")

    def forward(self, input):
        self._check_input_dim(input)
        factor = 0.0 if self.momentum is None else self.momentum
        if self.training and self.track_running_stats and self.num_batches_tracked is not None:
            self.num_batches_tracked.add_(1)
            factor = 1.0 / float(self.num_batches_tracked) if self.momentum is None else self.momentum
        use_batch = self.training or (self.running_mean is None and self.running_var is None)
        track = not self.training or self.track_running_stats
        return F.batch_norm(
            input, self.running_mean if track else None, self.running_var if track else None,
            self.weight, self.bias, use_batch, factor, self.eps,
        )

    def extra_repr(self):
        return (
            f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}, "
            f"bias={self.bias is not None}, track_running_stats={self.track_running_stats}"
        )


class BatchNorm1d(_BatchNorm):
    """For (N, C) or (N, C, L) inputs."""

    _dims, _dims_text = (2, 3), "2D or 3D"


class BatchNorm2d(_BatchNorm):
    """For (N, C, H, W) inputs."""

    _dims, _dims_text = (4,), "4D"


class LayerNorm(Module):
    """Normalizes each sample over its last dimensions (normalized_shape)."""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True, device=None, dtype=None):
        super().__init__()
        shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.normalized_shape, self.eps, self.elementwise_affine = shape, eps, elementwise_affine
        factory = {"dtype": dtype, "device": device}
        if elementwise_affine:
            self.weight = Parameter(_creation.empty(shape, **factory))
            if bias:
                self.bias = Parameter(_creation.empty(shape, **factory))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def forward(self, input):
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self):
        return (
            f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}, "
            f"bias={self.bias is not None}"
        )


class GroupNorm(Module):
    """Normalizes each sample over groups of channels (and all positions), so it does not
    depend on the batch."""

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, device=None, dtype=None, *, bias=True):
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError(f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})")
        self.num_groups, self.num_channels, self.eps, self.affine = num_groups, num_channels, eps, affine
        factory = {"dtype": dtype, "device": device}
        if affine:
            self.weight = Parameter(_creation.empty(num_channels, **factory))
            if bias:
                self.bias = Parameter(_creation.empty(num_channels, **factory))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def forward(self, input):
        return F.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)

    def extra_repr(self):
        return f"{self.num_groups}, {self.num_channels}, eps={self.eps}, affine={self.affine}, bias={self.bias is not None}"
