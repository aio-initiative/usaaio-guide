"""Every name in SURFACE.txt must resolve on the stand-in; the rest raise a clear error."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import SRC, TOOL

NAMES = [line for line in (TOOL / "SURFACE.txt").read_text().splitlines() if line]
TORCH_NAMES = [n for n in NAMES if not n.startswith("torchvision.")]
VISION_NAMES = [n for n in NAMES if n.startswith("torchvision.")]


def resolve(st, name):
    """Follow a SURFACE.txt name on the stand-in; Tensor.x is looked up on the Tensor class."""
    root, *parts = name.split(".")
    assert root in ("torch", "Tensor"), name
    obj = st.Tensor if root == "Tensor" else st
    for part in parts:
        obj = getattr(obj, part)
    return obj


def test_real_torch_sits_beside_the_stand_in(st, real):
    assert real.__name__ == "torch" and st.__name__ == "standin_torch"
    assert real is not st and hasattr(real, "Tensor")


def test_unknown_names_say_they_are_not_available(st):
    with pytest.raises(NotImplementedError) as err:
        st.not_a_real_function
    assert str(err.value) == (
        "torch.not_a_real_function is not available in the browser version of PyTorch. "
        "Download the notebook or open it in Colab to use it."
    )


def test_private_and_dunder_names_are_plain_attribute_errors(st):
    # Import machinery, pickle and inspect probe these; they must not see NotImplementedError.
    for name in ("__wrapped__", "__not_there__", "_hidden"):
        with pytest.raises(AttributeError):
            getattr(st, name)
    assert not hasattr(st, "__wrapped__")


def test_the_getattr_does_not_block_importing_submodules(tmp_path, monkeypatch):
    # "from . import nn" asks hasattr(package, "nn") first; NotImplementedError there would
    # abort the import, so submodule names must give AttributeError and let the import go on.
    package = tmp_path / "standin_probe"
    package.mkdir()
    # The stand-in's own __init__ imports its whole package, so the probe gets the same
    # __getattr__ from the same helper instead of a copy of that file.
    (package / "_missing.py").write_text((SRC / "torch" / "_missing.py").read_text())
    (package / "__init__.py").write_text(
        "from ._missing import module_getattr\n__getattr__ = module_getattr(__name__, 'torch')\n"
    )
    (package / "nn.py").write_text("VALUE = 1\n")
    (package / "_core.py").write_text("VALUE = 2\n")
    (package / "user.py").write_text("from . import nn, _core\nTOTAL = nn.VALUE + _core.VALUE\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        assert importlib.import_module("standin_probe.user").TOTAL == 3
    finally:
        for name in [m for m in sys.modules if m.startswith("standin_probe")]:
            del sys.modules[name]


@pytest.mark.parametrize("name", TORCH_NAMES)
def test_torch_name_resolves(st, name):
    try:
        resolve(st, name)
    except (AttributeError, NotImplementedError) as err:
        pytest.fail(f"{name} does not resolve on the stand-in: {err}", pytrace=False)


@pytest.fixture(scope="session")
def vision_report():
    """Resolve every torchvision name in one subprocess where `import torch` is the stand-in."""
    if not (SRC / "torchvision" / "__init__.py").exists():
        return {name: "src/torchvision is not written yet" for name in VISION_NAMES}
    code = (
        "import importlib, json, sys\n"
        "report = {}\n"
        "for name in json.load(sys.stdin):\n"
        "    root, *parts = name.split('.')\n"
        "    try:\n"
        "        obj = importlib.import_module(root)\n"
        f"        assert obj.__file__.startswith({str(SRC)!r}), 'not the stand-in: ' + obj.__file__\n"
        "        for part in parts:\n"
        "            obj = getattr(obj, part)\n"
        "        report[name] = None\n"
        "    except Exception as err:\n"
        "        report[name] = f'{type(err).__name__}: {err}'\n"
        "print(json.dumps(report))\n"
    )
    env = dict(os.environ, PYTHONPATH=str(SRC))
    done = subprocess.run(
        [sys.executable, "-c", code],
        input=json.dumps(VISION_NAMES),
        capture_output=True,
        text=True,
        env=env,
        cwd=TOOL,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@pytest.mark.parametrize("name", VISION_NAMES)
def test_torchvision_name_resolves(vision_report, name):
    if vision_report[name] is not None:
        pytest.fail(f"{name} does not resolve on the stand-in: {vision_report[name]}", pytrace=False)
