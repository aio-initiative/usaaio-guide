"""Fixtures: the stand-in loaded as standin_torch, and the real torch beside it."""

import importlib.util
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1]
SRC = TOOL / "src"

# So a test can `import build` and use the tool beside src/. The folder is not src/, so the
# real torch in site-packages is still what `import torch` finds.
sys.path.insert(0, str(TOOL))


@pytest.fixture(scope="session")
def st():
    """src/torch imported under the name standin_torch, so real torch can stay `torch`."""
    package = SRC / "torch"
    spec = importlib.util.spec_from_file_location(
        "standin_torch", package / "__init__.py", submodule_search_locations=[str(package)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["standin_torch"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def real():
    """The real torch from the venv. It must not be the stand-in, so src stays off sys.path."""
    import torch

    assert SRC not in Path(torch.__file__).parents, "src is on sys.path, so this is the stand-in"
    return torch
