"""The zip the browser downloads holds the two packages and nothing else, and is reproducible.

The notebook editor unpacks public/pyodide/torch-standin.zip into Pyodide's site-packages, so
what is in it, and what is left out, is what the browser can import. A stale test file or a
__pycache__ in the zip is dead weight on every page load, and a zip whose bytes change on every
build makes a diff of the site's build output unreadable.
"""

import io
import zipfile

import pytest

import build
from conftest import SRC, TOOL

NOTICES = "THIRD_PARTY_NOTICES.md"
# What the browser must find after unpacking: the two packages, and the licences they run under.
ALLOWED_TOPS = {"torch", "torchvision", NOTICES}


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    """A zip built into a temp folder, so the test never writes the one the site serves."""
    path = tmp_path_factory.mktemp("zip") / "torch-standin.zip"
    build.build(path)
    with zipfile.ZipFile(path) as opened:
        return path, opened.namelist()


def test_the_zip_has_the_two_packages_and_the_notices_and_nothing_else(archive):
    _, members = archive
    assert len(members) > 50, "the whole stand-in, not a handful of files"
    assert sorted({name.split("/")[0] for name in members}) == sorted(ALLOWED_TOPS)
    assert members.count(NOTICES) == 1


def test_every_module_of_both_packages_is_there(archive):
    _, members = archive
    on_disk = {
        path.relative_to(SRC).as_posix()
        for package in build.PACKAGES
        for path in (SRC / package).rglob("*.py")
        if "__pycache__" not in path.parts
    }
    assert sorted(set(members) - {NOTICES}) == sorted(on_disk)


def test_no_tests_caches_or_bytecode_go_in_the_zip(archive):
    _, members = archive
    for name in members:
        assert not name.startswith("/") and ".." not in name.split("/"), name
        assert name.endswith(".py") or name == NOTICES, name
        assert "__pycache__" not in name and not name.endswith(".pyc"), name
        assert "tests" not in name.split("/"), name


def test_the_zip_is_reproducible_from_the_sources():
    """The same sources give the same bytes, so the served file changes only when they do."""
    assert build.zip_bytes() == build.zip_bytes()


def test_the_zip_the_repository_serves_is_up_to_date():
    """What the site serves is what the sources make, so build.py has been run for a change."""
    served = build.OUTPUT
    if not served.is_file():
        pytest.skip(f"{served} has not been built yet")
    assert served.read_bytes() == build.zip_bytes()


def test_the_notices_in_the_zip_are_the_ones_in_the_tool():
    with zipfile.ZipFile(io.BytesIO(build.zip_bytes())) as archive:
        assert archive.read(NOTICES) == (TOOL / NOTICES).read_bytes()
