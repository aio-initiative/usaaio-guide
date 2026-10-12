"""Random numbers: torch.manual_seed, torch.Generator and the functions that draw from them.

The draws come from NumPy's PCG64, so they do not match real torch's; a seed only makes the
stand-in repeat itself. Every function takes generator=, and uses the default generator
(the one manual_seed seeds) without it.
"""

import operator

import numpy as np

from . import _dtype
from ._creation import shape_from
from ._tensor import CPU, Tensor, require_cpu

DEFAULT_SEED = 67280421310721  # what torch's default generator starts from


class Generator:
    """A random number stream. manual_seed(n) restarts it from seed n and returns it."""

    def __init__(self, device="cpu"):
        require_cpu(device)
        self.device = CPU
        self.manual_seed(DEFAULT_SEED)

    def manual_seed(self, seed):
        self._seed = operator.index(seed)
        self._numpy = np.random.default_rng(self._seed % 2**64)  # torch accepts negative seeds
        return self

    def initial_seed(self):
        return self._seed

    def seed(self):
        """Reseed from the operating system's entropy and return the new seed."""
        seed = int(np.random.SeedSequence().entropy) % 2**64
        self.manual_seed(seed)
        return seed


default_generator = Generator()


def manual_seed(seed):
    """Seed the default generator, which every random function uses unless given another."""
    return default_generator.manual_seed(seed)


def _stream(generator):
    return (generator or default_generator)._numpy


def _float_dtype(dtype, kernel):
    """The NumPy float dtype to draw in; torch refuses integer dtypes for these draws."""
    want = _dtype.as_numpy(dtype or _dtype.float32)
    if want.kind != "f":
        raise NotImplementedError(f"\"{kernel}\" not implemented for '{_dtype.of(want).legacy}'")
    return want


def rand(*args, size=None, generator=None, dtype=None, device=None, requires_grad=False):
    """Uniform values in [0, 1)."""
    require_cpu(device)
    want = _float_dtype(dtype, "check_uniform_bounds")
    return Tensor(_stream(generator).random(shape_from(args, size), dtype=want), requires_grad=requires_grad)


def randn(*args, size=None, generator=None, dtype=None, device=None, requires_grad=False):
    """Values from the standard normal distribution."""
    require_cpu(device)
    want = _float_dtype(dtype, "normal_kernel_cpu")
    return Tensor(_stream(generator).standard_normal(shape_from(args, size), dtype=want), requires_grad=requires_grad)


def rand_like(input, dtype=None, device=None, requires_grad=False):
    return rand(input.shape, dtype=dtype or input.dtype, device=device, requires_grad=requires_grad)


def randn_like(input, dtype=None, device=None, requires_grad=False):
    return randn(input.shape, dtype=dtype or input.dtype, device=device, requires_grad=requires_grad)


def randint(*args, low=0, high=None, size=None, generator=None, dtype=None, device=None, requires_grad=False):
    """Integers in [low, high): randint(high, size) or randint(low, high, size)."""
    require_cpu(device)
    if size is None and args:
        *args, size = args
    if len(args) == 2:
        low, high = args
    elif len(args) == 1:
        high = args[0]
    if high is None or size is None:
        raise TypeError("randint() missing required arguments: give high and size, as randint(high, size)")
    low, high = operator.index(low), operator.index(high)
    if low >= high:
        raise RuntimeError(f"random_ expects 'from' to be less than 'to', but got from={low} >= to={high}")
    values = _stream(generator).integers(low, high, size=shape_from((size,)), dtype=np.int64)
    return Tensor(values.astype(_dtype.as_numpy(dtype or _dtype.int64), copy=False), requires_grad=requires_grad)


def randperm(n, *, generator=None, dtype=None, device=None, requires_grad=False):
    """The numbers 0 to n - 1 in a random order."""
    require_cpu(device)
    values = _stream(generator).permutation(operator.index(n)).astype(_dtype.as_numpy(dtype or _dtype.int64))
    return Tensor(values, requires_grad=requires_grad)
