"""torch.distributions.Normal, written with tensor operations so that rsample, log_prob and
entropy carry gradients to loc and scale."""

import math

import numpy as np

from .. import _dtype
from .._creation import full
from .._random import randn
from .._shape import expand
from .._tensor import Size, Tensor


def broadcast_all(*values):
    """Tensors of one shape, as views. Numbers take the dtype of the first tensor given, or
    float32 when there is none."""
    tensors = [v for v in values if isinstance(v, Tensor)]
    dtype = tensors[0].dtype if tensors else _dtype.float32
    values = [v if isinstance(v, Tensor) else full((), v, dtype=dtype) for v in values]
    shape = np.broadcast_shapes(*(v.shape for v in values))
    return [expand(v, *shape) for v in values]


class Normal:
    """The normal distribution with mean `loc` and standard deviation `scale`."""

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        self._validate = validate_args is not False
        if self._validate:
            self._check("loc", self.loc, self.loc._data == self.loc._data, "Real()")
            self._check("scale", self.scale, self.scale._data > 0, "GreaterThan(lower_bound=0.0)")

    def _check(self, name, value, valid, constraint):
        if not np.all(valid):
            raise ValueError(
                f"Expected parameter {name} (Tensor of shape {tuple(value.shape)}) of distribution {self!r} "
                f"to satisfy the constraint {constraint}, but found invalid values:\n{value!r}"
            )

    @property
    def batch_shape(self):
        return Size(self.loc.shape)

    @property
    def event_shape(self):
        return Size(())

    @property
    def mean(self):
        return self.loc

    @property
    def mode(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self.scale**2

    def _noise(self, sample_shape):
        return randn(Size(sample_shape) + self.batch_shape, dtype=self.loc.dtype)

    def rsample(self, sample_shape=()):
        """loc + noise * scale: a draw that gradients flow through (the reparameterisation trick)."""
        return self.loc + self._noise(sample_shape) * self.scale

    def sample(self, sample_shape=()):
        """A draw with no gradient. sample_shape is a tuple, as in torch."""
        return Tensor(self.loc._data + self._noise(sample_shape)._data * self.scale._data)

    def log_prob(self, value):
        if self._validate and not isinstance(value, Tensor):
            raise ValueError("The value argument to log_prob must be a Tensor")
        return -((value - self.loc) ** 2) / (2 * self.variance) - self.scale.log() - math.log(math.sqrt(2 * math.pi))

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + self.scale.log()

    def __repr__(self):
        parts = [
            f"{name}: {value if value.numel() == 1 else value.size()}"
            for name, value in (("loc", self.loc), ("scale", self.scale))
        ]
        return f"Normal({', '.join(parts)})"
