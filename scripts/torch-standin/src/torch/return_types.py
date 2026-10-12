"""The named pairs that max, min, sort, topk, median and mode return, as in torch.return_types.

Each is a 2-tuple with .values and .indices, so `values, indices = x.max(dim=1)` works too.
"""

import collections


def _pair(name):
    pair = collections.namedtuple(name, ["values", "indices"], module=__name__)
    pair.__repr__ = lambda self: f"torch.return_types.{name}(\nvalues={self.values!r},\nindices={self.indices!r})"
    return pair


max, min, sort, topk, median, mode = map(_pair, ["max", "min", "sort", "topk", "median", "mode"])
