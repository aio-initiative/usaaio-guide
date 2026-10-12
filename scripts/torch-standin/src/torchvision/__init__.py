"""A NumPy-backed stand-in for torchvision that runs in the browser (Pyodide).

It covers the datasets, transforms and image reading that our notebooks use. Anything else
says it is not available in the browser, the way torch does for the names it does not have.
"""

from torch._missing import module_getattr

from . import datasets, io, models, transforms

# The release of torchvision this package is tested against.
__version__ = "0.29.1+standin"

__getattr__ = module_getattr(__name__, "torchvision")
