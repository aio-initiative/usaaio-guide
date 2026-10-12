"""The stand-in is derived in part from PyTorch and torchvision, which are BSD-3-Clause: the
notices file carries their licences unchanged, and every ported file names its origin there."""

import re
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path

import pytest

from conftest import SRC, TOOL

NOTICES = (TOOL / "THIRD_PARTY_NOTICES.md").read_text()
HEADER = re.compile(r"^# (?:Partly p|P)orted from (PyTorch|torchvision) \(([^)]*)\)", re.MULTILINE)


def ported_files():
    """{path relative to the tool folder: project} for every source file whose first lines say
    it was ported."""
    found = {}
    for path in sorted(SRC.rglob("*.py")):
        match = HEADER.match("".join(path.read_text().splitlines(keepends=True)[:2]))
        if match:
            found[path.relative_to(TOOL).as_posix()] = match.group(1)
    return found


def listed_files():
    """The src/ paths the notices file lists in its tables."""
    return set(re.findall(r"^\| `(src/[^`]+)`", NOTICES, re.MULTILINE))


def test_every_ported_file_is_listed_in_the_notices_and_every_listed_file_is_ported():
    ported = ported_files()
    assert ported, "no file says it was ported"
    assert set(ported) == listed_files()


def test_the_notices_name_a_file_of_each_project_it_lists():
    assert {"PyTorch", "torchvision"} == set(ported_files().values())
    models = NOTICES.split("From torchvision (paths inside the `torchvision` package):")[1]
    assert "`src/torchvision/models.py` | `torchvision/models/resnet.py`" in models


def test_a_ported_file_names_the_file_it_follows_and_the_licence():
    for path, project in ported_files().items():
        head = "".join((TOOL / path).read_text().splitlines(keepends=True)[:2])
        assert "BSD-3-Clause" in head and "THIRD_PARTY_NOTICES.md" in head, path


def named_sources(path):
    """The upstream files a ported file's header names, each as (as written, path inside its
    package). `torch/utils/data/dataloader.py, sampler.py` names two files in one folder, and a
    `;` starts another file, with what is taken from it after a colon."""
    head = "".join((TOOL / path).read_text().splitlines(keepends=True)[:2])
    inside = HEADER.match(head).group(2).replace("\n# ", " ")
    named = []
    for part in inside.split(";"):
        first, *rest = (name.strip() for name in part.split(":")[0].split(","))
        named.append((first, first))
        folder = first.rsplit("/", 1)[0]
        named.extend((name, f"{folder}/{name}") for name in rest)
    return named


def test_a_header_names_files_that_exist_upstream_and_the_notices_row_names_them_too():
    rows = dict(re.findall(r"^\| `(src/[^`]+)`[^|]*\|(.*)\|$", NOTICES, re.MULTILINE))
    for path, project in ported_files().items():
        package = "torch" if project == "PyTorch" else "torchvision"
        installed = Path(find_spec(package).origin).parents[1]
        for written, upstream in named_sources(path):
            assert upstream.startswith(f"{package}/"), f"{path} names {upstream}, outside {package}"
            assert (installed / upstream).is_file(), f"{path} names {upstream}, which {package} does not have"
            if written == upstream:
                assert f"`{upstream}`" in rows[path], f"the notices row for {path} does not name {upstream}"


@pytest.mark.parametrize(
    "distribution,candidates",
    [("torch", ["licenses/LICENSE", "LICENSE"]), ("torchvision", ["LICENSE", "licenses/LICENSE"])],
)
def test_the_licences_are_the_installed_ones_unchanged(distribution, candidates):
    for name in candidates:
        text = metadata.distribution(distribution).read_text(name)
        if text:
            break
    else:
        pytest.fail(f"no LICENSE file in the installed {distribution}")
    assert text.rstrip("\n") in NOTICES, f"the {distribution} licence in the notices differs from the installed one"
