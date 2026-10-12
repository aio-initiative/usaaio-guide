#!/usr/bin/env python3
"""Gate the notebooks: run each one's worked solution and its blank template.

Usage:
  scripts/check_notebooks.py [--timeout SECONDS] [--python INTERPRETER] <notebook>...

For each A-convention notebook, (a) builds the solved version (each task cell
replaced by that part's solution cells, the solution section dropped) and runs
it top to bottom, checking that every self-check cell passes; (b) runs the
blank template and checks that the self-checks fail or the task cells raise.

One line per notebook on stdout:
  PASS   <notebook> (solved: <n>/<n> checks ok in <s>s; blank: <n> checks fail as they should)
  FAIL   <notebook> <which run, cell and reason>
  TIMEOUT <notebook> <which run, after <n>s>
  NEEDS  <notebook> <gpu|download|package>: <reason>
The exit code is 1 when any notebook FAILs and 0 otherwise. TIMEOUT and NEEDS
are reported, not counted as failures.

Each notebook runs in its own throwaway folder through a worker subprocess of
`~/miniforge3/envs/Main/bin/python` (override with --python), so one notebook
that crashes a library cannot kill the run. MNIST comes from the MNIST_ROOT
environment variable: torchvision is pointed at it with download=False, so
nothing is ever downloaded and nothing is written into the repo. A notebook
that cannot find its data says NEEDS download rather than downloading it.

What counts as a self-check: a code cell whose last statement is a print whose
text says "passed" (site B's convention, `print("Part 3 passed")`), or any code
cell that prints a line containing "pass". The cell passes when the last such
line it printed is not a "NOT PASSED yet" line, which also reads site A's older
`print("PASSED ..." if ok else "NOT PASSED ...")` convention.
"""

import argparse
import ast
import contextlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

DEFAULT_PYTHON = os.path.expanduser("~/miniforge3/envs/Main/bin/python")
DEFAULT_TIMEOUT = 600

PART = re.compile(r"^#{2,3}\s*Part\s+(\d+)", re.M)
FENCE = re.compile(r"```python\n(.*?)```", re.S)
SAYS_PASSED = re.compile(r"passed", re.I)
SAYS_PASS = re.compile(r"pass", re.I)
NOT_PASS = re.compile(r"^\W*not\s+pass", re.I)
MAGIC = re.compile(r"^\s*(%|!)", re.M)


def src(cell):
    text = cell.get("source")
    if isinstance(text, list):
        return "".join(text)
    return text or ""


def is_solution(cell):
    return bool((cell.get("metadata") or {}).get("usaaio_solution"))


def is_todo(text):
    return (
        "TODO" in text
        or "raise NotImplementedError" in text
        or "YOUR SOLUTION GOES HERE" in text
        or re.search(r"=\s*\.\.\.\s*($|#)", text, re.M) is not None
    )


def code_cell(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": text,
        "outputs": [],
        "execution_count": None,
    }


def build_solved(cells):
    """The worked notebook: task cells replaced by their solution cells.

    The solution section is read first, cell by cell: a `### Part N` heading in
    it sets the part the following solution cells belong to, and a fenced
    ```python block inside a markdown cell is site A's older convention, where
    the whole solution is one block after `## Solution`. Then the exercise is
    walked and each task cell is replaced by its part's solution cells, once.
    """
    by_part = {}
    part = None
    for cell in cells:
        if not is_solution(cell):
            continue
        text = src(cell)
        if cell["cell_type"] == "markdown":
            match = PART.search(text)
            if match:
                part = int(match.group(1))
                continue
            block = FENCE.search(text)
            if block:
                by_part.setdefault(None, []).append(code_cell(block.group(1)))
        elif cell["cell_type"] == "code":
            by_part.setdefault(part, []).append(code_cell(text))
    worked = []
    part = None
    used = set()
    for cell in cells:
        if is_solution(cell):
            continue
        if cell["cell_type"] != "code":
            if cell["cell_type"] == "markdown":
                match = PART.search(src(cell))
                if match:
                    part = int(match.group(1))
            continue
        text = src(cell)
        key = part if part in by_part else (None if None in by_part else part)
        if is_todo(text) and key in by_part and key not in used:
            worked.extend(by_part[key])
            used.add(key)
        elif is_todo(text) and key in used:
            continue
        else:
            worked.append(code_cell(text))
    return worked, used


def blank_cells(cells):
    """The exercise as the student gets it: everything but the solution section."""
    out = []
    for cell in cells:
        if is_solution(cell):
            continue
        if cell["cell_type"] == "code":
            out.append(code_cell(src(cell)))
    return out


def _strings(node):
    """The string constants a print argument carries, including the two arms of
    site A's older `print("PASSED ..." if ok else "NOT PASSED ...")` form."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node.value
    if isinstance(node, ast.IfExp):
        yield from _strings(node.body)
        yield from _strings(node.orelse)


def is_self_check(text):
    """A code cell that prints its own verdict. Site B's convention is the last
    statement, `print("Part 3 passed")`; site A's older notebooks print it from
    inside an if/else, `print("PASSED, nice work!" if _ok else "NOT PASSED ...")`.
    Both say the word "passed"."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "print":
            if any(SAYS_PASSED.search(word)
                   for arg in node.args for word in _strings(arg)):
                return True
    return False


def check_lines(stdout):
    """The verdict lines a cell printed: 'Part 3 passed', 'PASSED, nice work!',
    'NOT PASSED yet', in order."""
    return [line.strip() for line in stdout.splitlines() if SAYS_PASS.search(line)]


def verdict(stdout):
    """True when the last verdict line the cell printed is a passing one."""
    lines = check_lines(stdout)
    if not lines:
        return None
    return not NOT_PASS.match(lines[-1])


def demagic(text):
    """IPython magics and shell lines are not Python; the runner skips them."""
    return "\n".join(
        (line[: len(line) - len(line.lstrip())] + "pass")
        if MAGIC.match(line)
        else line
        for line in text.split("\n")
    )


# --------------------------------------------------------------------------- worker


def mnist_patch():
    """Point torchvision at MNIST_ROOT with download=False, whatever the
    notebook asks for. Nothing is downloaded; nothing is written to the repo."""
    root = os.environ.get("MNIST_ROOT")
    source = """
import os
import torchvision.datasets as _D
_OrigMNIST = _D.MNIST


def _mnist(root=".", train=True, download=False, transform=None,
           target_transform=None):
    here = os.environ.get("MNIST_ROOT")
    if not here:
        raise RuntimeError(
            "MNIST_ROOT is not set, so the notebook cannot read MNIST without "
            "downloading it")
    return _OrigMNIST(here, train=train, download=False, transform=transform,
                      target_transform=target_transform)


_D.MNIST = _mnist
"""
    g = {"__name__": "__main__"}
    exec(compile(source, "<mnist patch>", "exec"), g)
    return root


def exec_cells(cells, label):
    """Run one notebook's code cells in one shared namespace, top to bottom."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.show = lambda *a, **k: None
    g = {"__name__": "__main__"}
    report = []
    total = time.time()
    for cell in cells:
        text = src(cell)
        buf = io.StringIO()
        started = time.time()
        error = None
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                exec(compile(demagic(text), f"<{label} cell>", "exec"), g)
        except BaseException as caught:  # SystemExit and KeyboardInterrupt too
            error = f"{type(caught).__name__}: {str(caught).splitlines()[0][:200] if str(caught) else ''}"
        out = buf.getvalue()
        report.append(
            {
                "source": text,
                "head": text.strip().splitlines()[0][:70] if text.strip() else "",
                "check": is_self_check(text),
                "error": error,
                "stdout": out[-4000:],
                "seconds": round(time.time() - started, 2),
            }
        )
        print(
            f"  [{'ok  ' if error is None else 'FAIL'}] "
            f"{report[-1]['seconds']:7.2f}s  {report[-1]['head']}",
            file=sys.stderr,
        )
        if error:
            print(f"        {error}", file=sys.stderr)
        for line in [line for line in out.strip().splitlines() if line.strip()][-3:]:
            print(f"        | {line[:150]}", file=sys.stderr)
        plt.close("all")
    print(f"  total {time.time() - total:.1f}s", file=sys.stderr)
    return report


def worker(notebook, mode, scratch):
    os.chdir(scratch)
    os.environ["MPLBACKEND"] = "Agg"
    mnist_patch()
    with open(notebook, encoding="utf-8") as handle:
        cells = json.load(handle)["cells"]
    if mode == "solved":
        run, _used = build_solved(cells)
    elif mode == "blank":
        run = blank_cells(cells)
    else:
        raise SystemExit(f"unknown mode {mode!r}")
    report = exec_cells(run, mode)
    print(json.dumps({"report": report}))


# --------------------------------------------------------------------------- parent


def needs(errors):
    """What a notebook is missing, from the errors it raised, or None."""
    for error in errors:
        match = re.search(r"No module named '([^']+)'", error)
        if match or "ModuleNotFoundError" in error or "ImportError" in error:
            return f"package: {match.group(1) if match else error}"
    for error in errors:
        if re.search(r"cuda|gpu", error, re.I) and re.search(
            r"not available|no gpu|device|assert", error, re.I
        ):
            return f"gpu: {error}"
    for error in errors:
        if "MNIST_ROOT" in error or re.search(r"download", error, re.I):
            if "urlopen" in error or "HTTP" in error or "MNIST_ROOT" in error:
                return f"download: {error}"
    return None


def run_one(notebook, mode, timeout, python, scratch):
    """Run one notebook in one mode in a worker subprocess. Returns
    (report, wall_seconds) or (None, reason) when the worker itself died."""
    command = [python, os.path.abspath(__file__), "--worker",
               os.path.abspath(notebook), "--mode", mode, "--dir", scratch]
    started = time.time()
    try:
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True,
        )
    except OSError as caught:
        return None, f"cannot start {python}: {caught}"
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        out, err = proc.communicate()
        if err:
            sys.stderr.write(err)
        return None, "timeout"
    wall = time.time() - started
    if err:
        sys.stderr.write(err)
    if proc.returncode != 0:
        tail = (err or out or "").strip().splitlines()
        return None, "worker died: " + (tail[-1][:200] if tail else f"exit {proc.returncode}")
    try:
        return json.loads(out)["report"], wall
    except (ValueError, KeyError) as caught:
        return None, f"no report from the worker ({caught})"


def judge(name, notebook, timeout, python):
    """Both runs for one notebook. Returns (verdict, detail, seconds, record)."""
    record = {"notebook": notebook, "name": name}
    for mode in ("solved", "blank"):
        scratch = tempfile.mkdtemp(prefix=f"nb-{mode}-")
        try:
            report, problem = run_one(notebook, mode, timeout, python, scratch)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        record[mode] = {
            "cells": [
                {
                    "source": cell["source"],
                    "head": cell["head"],
                    "check": cell["check"],
                    "ok": cell["error"] is None,
                    "error": cell["error"],
                    "verdict": verdict(cell["stdout"]) if cell["check"] else None,
                    "lines": check_lines(cell["stdout"]) if cell["check"] else [],
                    "seconds": cell["seconds"],
                }
                for cell in (report or [])
            ],
            "problem": problem,
            "wall": None,
        }
        if report is None:
            if problem == "timeout":
                return ("TIMEOUT", f"{mode} run, after {timeout}s", record)
            short = needs([problem]) or problem
            if short.startswith("download") or short.startswith("package"):
                return ("NEEDS", short, record)
            return ("FAIL", f"{mode} run: {problem}", record)
        record[mode]["wall"] = round(sum(cell["seconds"] for cell in report), 1)
        errors = [cell["error"] for cell in report if cell["error"]]
        missing = needs(errors)
        if missing:
            return ("NEEDS", missing, record)

    for mode, what in (("solved", "the worked solution"), ("blank", "the blank template")):
        cells = record[mode]["cells"]
        checks = [index for index, cell in enumerate(cells) if cell["check"]]
        if not checks:
            return ("FAIL", f"{mode} run has no self-check cell, so nothing "
                    "proves the solution works", record)
        for index in checks:
            cell = cells[index]
            if cell["error"] is not None:
                if mode == "solved":
                    return ("FAIL", f"{mode} check cell {index + 1} raised "
                            f"{cell['error']}", record)
                continue  # the blank template's check cell dies on the unimplemented
                          # task cell above it, which is what the gate wants to see
            if cell["verdict"] is not True:
                said = (cell["lines"] or ["no verdict printed"])[-1]
                return ("FAIL", f"{mode} check cell {index + 1} printed "
                        f"{said!r}", record)
            if mode == "blank":
                return ("FAIL", f"{mode} check cell {index + 1} passes before "
                        "the student does the work", record)
        if mode == "solved":
            for index, cell in enumerate(cells):
                if not cell["ok"] and not cell["check"]:
                    return ("FAIL", f"{mode} cell {index + 1} ({cell['head']}) "
                            f"raised {cell['error']}", record)
        else:
            for index, cell in enumerate(cells):
                if not cell["ok"] and not cell["check"] and not is_todo(cell["source"]):
                    return ("FAIL", f"{mode} cell {index + 1} ({cell['head']}) "
                            "raises before the student touches it: "
                            f"{cell['error']}", record)
    solved_checks = sum(1 for cell in record["solved"]["cells"] if cell["check"])
    blank_failed = sum(
        1
        for cell in record["blank"]["cells"]
        if cell["check"] and (cell["error"] is not None or cell["verdict"] is not True)
    )
    return (
        "PASS",
        f"solved: {solved_checks}/{solved_checks} checks ok in "
        f"{record['solved']['wall']}s; blank: {blank_failed} check"
        f"{'s' if blank_failed != 1 else ''} fail as they should",
        record,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("notebooks", nargs="*", help="A-convention notebooks")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"seconds per run, solved and blank each "
                             f"(default {DEFAULT_TIMEOUT})")
    parser.add_argument("--python", default=DEFAULT_PYTHON,
                        help="the interpreter for the worker (default "
                             f"{DEFAULT_PYTHON})")
    parser.add_argument("--json", help="append one JSON record per notebook here")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--mode", help=argparse.SUPPRESS)
    parser.add_argument("--dir", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        worker(args.worker, args.mode, args.dir)
        return

    if not args.notebooks:
        parser.error("no notebooks given")

    python = args.python
    if not os.path.isfile(python):
        parser.error(f"no such interpreter: {python}")
    if not os.environ.get("MNIST_ROOT"):
        print("note: MNIST_ROOT is not set, so a notebook that reads MNIST "
              "will report NEEDS download", file=sys.stderr)

    records = []
    worst = 0
    for notebook in args.notebooks:
        name = os.path.splitext(os.path.basename(notebook))[0]
        started = time.time()
        verdict, detail, record = judge(name, notebook, args.timeout, python)
        record["verdict"] = verdict
        record["detail"] = detail
        record["wall"] = round(time.time() - started, 1)
        records.append(record)
        print(f"{verdict:<8} {name} ({detail})")
        if verdict == "FAIL":
            worst = 1
    if args.json:
        with open(args.json, "a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return worst


if __name__ == "__main__":
    sys.exit(main())
