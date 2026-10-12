"""torch.save and torch.load.

The files are plain pickles rather than torch's zip archives, so they load back into the
stand-in but not into real PyTorch. Loading with weights_only (the default, as in torch)
accepts only tensors, NumPy arrays, containers and numbers, so a file cannot run code.
"""

import os
import pickle

from . import _dtype
from ._tensor import device, require_cpu

# The globals a pickle of tensors, state dicts and plain values refers to.
_SAFE = {
    ("collections", "OrderedDict"),
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("builtins", "slice"),
    ("builtins", "complex"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    (f"{__package__}._tensor", "Tensor"),
    (f"{__package__}._tensor", "Size"),
    (f"{__package__}._tensor", "device"),
    (f"{__package__}.nn.parameter", "Parameter"),
}
_SAFE |= {(f"{__package__}._dtype", d.name) for d in _dtype.ALL}
_SAFE |= {(f"{__package__}.return_types", name) for name in ("max", "min", "sort", "topk", "median", "mode")}


class _WeightsUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _SAFE:
            raise pickle.UnpicklingError(
                f"Weights only load failed: the file contains {module}.{name}, which is not a tensor, "
                "a container or a number. Load it with weights_only=False only if you trust where it came from."
            )
        return super().find_class(module, name)


def save(obj, f, pickle_protocol=pickle.DEFAULT_PROTOCOL):
    """Write `obj` (a tensor, a state_dict, a dict of them...) to a path or a binary file."""
    if isinstance(f, (str, os.PathLike)):
        with open(f, "wb") as file:
            pickle.dump(obj, file, protocol=pickle_protocol)
    else:
        pickle.dump(obj, f, protocol=pickle_protocol)


def load(f, map_location=None, *, weights_only=None, **pickle_load_args):
    """Read what save wrote. map_location may name the CPU; there is no GPU to map to."""
    if isinstance(map_location, (str, device)):
        require_cpu(map_location)
    unpickler = pickle.Unpickler if weights_only is False else _WeightsUnpickler
    if isinstance(f, (str, os.PathLike)):
        with open(f, "rb") as file:
            return unpickler(file, **pickle_load_args).load()
    return unpickler(f, **pickle_load_args).load()
