"""torch.utils: torch.utils.data, the datasets and loaders."""

from .._missing import module_getattr
from . import data

__getattr__ = module_getattr(__name__, "torch.utils")
