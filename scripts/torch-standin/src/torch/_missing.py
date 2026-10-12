"""The error for a PyTorch name the stand-in does not have, and the module __getattr__ that
raises it, shared by torch and its subpackages."""

import importlib.util
import sys


MESSAGE = (
    "{} is not available in the browser version of PyTorch. "
    "Download the notebook or open it in Colab to use it."
)


def not_available(name):
    """The error for `name`, written as the notebook would write it (torch.x, torch.linalg.y)."""
    return NotImplementedError(MESSAGE.format(name))


class MissingTensorMethod(NotImplementedError, AttributeError):
    """A torch.Tensor method the stand-in does not have. Also an AttributeError, so hasattr()
    and getattr() with a default answer as they would for any missing attribute."""

    def __init__(self, name):
        super().__init__(MESSAGE.format(f"torch.Tensor.{name}"))


def module_getattr(module_name, public_name):
    """A module __getattr__ for the module `module_name`, which notebooks call `public_name`."""

    def __getattr__(name):
        # Python probes private names (__path__, __wrapped__, ...) and "from . import nn" asks
        # hasattr(package, "nn") before importing it; both need a real AttributeError.
        module = sys.modules[module_name]
        if name.startswith("_") or (
            hasattr(module, "__path__") and importlib.util.find_spec(f"{module_name}.{name}") is not None
        ):
            raise AttributeError(f"module {public_name!r} has no attribute {name!r}")
        raise not_available(f"{public_name}.{name}")

    return __getattr__
