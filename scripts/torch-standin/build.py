"""Pack the stand-in into the zip the browser downloads: public/pyodide/torch-standin.zip.

The notebook editor fetches this file, unpacks it into Pyodide's site-packages, and imports
torch and torchvision from there. So the zip holds exactly the two packages and the licence
notices: the source under src/, no tests, no __pycache__ and no other build output.

The zip is written so that building it twice from the same sources gives the same bytes: the
members are sorted, every timestamp is fixed and the permissions are the same on every machine.
The file the site serves then changes only when the stand-in does, and `--check` can tell a
stale zip from a current one.

Run it from scripts/torch-standin with the venv's Python (any Python 3 works, the venv is only
needed to be the one the tests use):

    .venv/bin/python build.py            # write public/pyodide/torch-standin.zip
    .venv/bin/python build.py --check    # fail if the zip is out of date or missing
    .venv/bin/python build.py --out /tmp/copy.zip
"""

import argparse
import io
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
NOTICES = HERE / "THIRD_PARTY_NOTICES.md"
REPO = HERE.parents[1]
OUTPUT = REPO / "public" / "pyodide" / "torch-standin.zip"

# The two packages the notebook imports under their own names, as they sit in src/.
PACKAGES = ("torch", "torchvision")
# The notices travel with them: the code is derived from PyTorch and torchvision, so the browser
# has to be able to read the licences it runs under.
NOTICES_NAME = NOTICES.name

# Every member gets this timestamp (1 January 1980, the earliest a zip can hold) so the bytes
# depend on the sources alone. Zip stores local time in its fields and carries no zone, so the
# time of day is the same whatever the machine's clock says.
FIXED_TIME = (1980, 1, 1, 0, 0, 0)
# 0o100644 (a regular file, readable by everyone) shifted into the high half of the external
# attributes, and Unix as the creating system, which Python would otherwise take from the
# machine doing the build and so would differ between Windows and elsewhere.
FILE_ATTRIBUTES = (0o100644) << 16
UNIX_SYSTEM = 3
# How hard to squeeze: the default for ZIP_DEFLATED, named so the size does not drift with the
# Python version's default.
COMPRESS_LEVEL = 6


def sources():
    """(name in the zip, path to read) for every source file of the two packages, sorted.

    Only `.py` files are taken, and only from the two package folders, so a test, a notebook or
    a stray cache cannot reach the zip. The name is the path under src/, which is the name the
    zip needs: `torch/nn/modules/conv.py` and no top-level folder of its own.
    """
    found = []
    for package in PACKAGES:
        for path in (SRC / package).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            found.append((path.relative_to(SRC).as_posix(), path))
    return sorted(found)


def write_to(archive):
    """Add every source file and then the notices to an open archive, in that order.

    The level goes to each writestr: with a ZipInfo of its own, a member does not take the
    level the archive was opened with.
    """
    for name, path in sources():
        archive.writestr(entry_for(name), path.read_bytes(), compresslevel=COMPRESS_LEVEL)
    archive.writestr(
        entry_for(NOTICES_NAME), NOTICES.read_bytes(), compresslevel=COMPRESS_LEVEL
    )


def entry_for(name):
    """The zip header for a member: fixed, so two builds of the same source match byte for byte."""
    info = zipfile.ZipInfo(name, date_time=FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = UNIX_SYSTEM
    info.external_attr = FILE_ATTRIBUTES
    return info


def zip_bytes():
    """The zip the current sources make, as bytes, without writing anything."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=COMPRESS_LEVEL) as archive:
        write_to(archive)
    return buffer.getvalue()


def build(output=OUTPUT):
    """Write the zip and return its members, in the order they were written."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(zip_bytes())
    return [name for name, _ in sources()] + [NOTICES_NAME]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=OUTPUT, help=f"where to write (default {OUTPUT})")
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; fail when the zip is missing or differs from the sources",
    )
    args = parser.parse_args()

    wanted = zip_bytes()
    if args.check:
        if not args.out.is_file():
            print(f"{args.out} does not exist; run build.py to build it", file=sys.stderr)
            return 1
        if args.out.read_bytes() != wanted:
            print(f"{args.out} is out of date; run build.py to rebuild it", file=sys.stderr)
            return 1
        print(f"{args.out} is up to date ({len(wanted) / 1e3:.0f} kB)")
        return 0

    members = build(args.out)
    print(f"wrote {args.out}: {len(members)} members, {len(wanted) / 1e3:.0f} kB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
