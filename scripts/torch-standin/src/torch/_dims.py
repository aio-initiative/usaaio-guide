"""Turning a dim argument into an axis, with torch's range check.

Every operation that takes a dim goes through here, so a dim out of range raises torch's
IndexError instead of wrapping round or picking the wrong axis. This module imports nothing
from the package, so even _tensor can use it.
"""

import operator


def dim_index(dim, ndim, extra=0):
    """`dim` as a non-negative axis of a tensor with `ndim` dimensions.

    `extra` widens the range for operations that add dimensions (unsqueeze and stack accept
    one more). A 0-d tensor counts as having one dimension, as in torch, so it accepts 0 and -1.
    """
    rank = max(ndim + extra, 1)
    d = operator.index(dim)
    if not -rank <= d < rank:
        raise IndexError(f"Dimension out of range (expected to be in range of [{-rank}, {rank - 1}], but got {d})")
    return d % rank


def dims_of(input, dim):
    """`dim` (an int, a tuple or a list) as a tuple of non-negative axes, or None for every
    dimension. dim=None and dim=() mean every dimension, as in torch's reductions."""
    if dim is None:
        return None
    many = tuple(dim) if isinstance(dim, (tuple, list)) else (dim,)
    ndim = input._data.ndim
    axes = tuple(dim_index(d, ndim) for d in many)
    return axes if axes and ndim else None


def axis_of(input, dim):
    """One dim as a non-negative axis; 0 for a 0-d tensor, which counts as one long."""
    return dim_index(dim, input._data.ndim)
