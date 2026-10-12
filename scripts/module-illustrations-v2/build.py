#!/usr/bin/env python3
"""Redraw every module picture: run each <section>/<id>.py, which writes
assets/module-illustrations/<section>/<id>.svg through common.write."""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

scripts = sorted(path for path in HERE.glob("*/*.py"))
failed = [path for path in scripts if subprocess.run([sys.executable, str(path)], cwd=ROOT).returncode]
print(f"{len(scripts) - len(failed)} of {len(scripts)} pictures drawn")
for path in failed:
    print(f"failed: {path.relative_to(ROOT)}")
sys.exit(1 if failed else 0)
