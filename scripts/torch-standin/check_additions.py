"""Check the names added to the stand-in in this repository against real PyTorch.

    ~/miniforge3/envs/Main/bin/python scripts/torch-standin/check_additions.py

The stand-in came from the USAAIO Guide with its own parity tests (tests/, run with pytest in a
venv that has real torch). The MLP notebooks needed two names it did not have, so they were
added here: torch.isclose (and Tensor.isclose) and torch.set_printoptions. This script runs
the same cases twice, once in a child process on real torch and once on the stand-in in src/,
and compares what comes back: bool results exactly, numbers within a tolerance, printed
tensors character for character, and errors by their type. The inputs are the ones the
notebooks use (an MNIST digit, a 784 to 16 sigmoid layer) plus edge cases around them.

Run it with a Python that has real torch and NumPy. It exits 1 when any case differs.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
MNIST = HERE.parents[1] / "public" / "pyodide" / "datasets" / "mnist.npz"

# Each case is Python that leaves its answer in `result`. It runs with numpy as np, torch as
# torch (real or stand-in), `rng` (a NumPy generator seeded the same way in both runs) and
# `digit` (the first MNIST test image, as uint8). Printing state is reset before each case.
CASES = {
    # Part 2 of the MNIST notebook: row 3 of the layer equals neuron 3 computed on its own.
    "isclose: layer row against one neuron": """
x = torch.tensor(digit.reshape(-1).astype(np.float32) / 255)
W = torch.tensor(rng.normal(0, 0.05, (16, 784)).astype(np.float32))
b = torch.tensor(rng.normal(0, 0.05, 16).astype(np.float32))
layer = torch.sigmoid(W @ x + b)
one = torch.sigmoid((W[3] * x).sum() + b[3])
result = {"close": bool(torch.isclose(layer[3], one)), "values": [float(layer[3]), float(one)]}
""",
    "isclose: tolerances, nan and inf": """
a = torch.tensor([1.0, 1.0001, 1.1, float("nan"), float("inf"), float("-inf"), 0.0, 1e-9])
c = torch.tensor([1.0, 1.0, 1.0, float("nan"), float("inf"), float("inf"), 1e-9, 0.0])
result = {
    "default": torch.isclose(a, c).tolist(),
    "rtol": torch.isclose(a, c, rtol=1e-3).tolist(),
    "atol": torch.isclose(a, c, atol=0.2).tolist(),
    "nan": torch.isclose(a, c, equal_nan=True).tolist(),
    "method": a.isclose(c, rtol=0, atol=1e-8).tolist(),
    "dtype": str(torch.isclose(a, c).dtype),
}
""",
    "isclose: integers, bools and broadcasting": """
i = torch.tensor([1, 2, 3, 100])
j = torch.tensor([1, 2, 4, 101])
p = torch.tensor(rng.normal(size=(3, 1)).astype(np.float32))
q = p + torch.tensor(np.array([[0.0, 1e-7, 1e-3, 1.0]], dtype=np.float32))
result = {
    "ints": torch.isclose(i, j).tolist(),
    "ints_atol": torch.isclose(i, j, atol=1).tolist(),
    "bools": torch.isclose(torch.tensor([True, False]), torch.tensor([True, True])).tolist(),
    "broadcast": torch.isclose(p, q).tolist(),
    "shape": list(torch.isclose(p, q).shape),
}
""",
    "isclose: mismatched dtypes": """
try:
    torch.isclose(torch.tensor([1.0]), torch.tensor([1]))
    result = "no error"
except Exception as error:
    result = type(error).__name__
""",
    # Part 1 of the MNIST notebook: a corner of the digit, printed 120 characters wide.
    "set_printoptions: the digit's corner at linewidth 120": """
image = torch.tensor(digit)
before = repr(image[7:15, 14:22])
torch.set_printoptions(linewidth=120)
result = {"before": before, "after": repr(image[7:15, 14:22]), "wide row": repr(image[12])}
""",
    "set_printoptions: precision and sci_mode": """
t = torch.tensor(rng.normal(0, 1, (3, 5)).astype(np.float32))
m = torch.tensor([1e-5, 0.5, 3.0, 2e4])
torch.set_printoptions(precision=2)
p2 = repr(t)
torch.set_printoptions(precision=6)
p6 = repr(t)
torch.set_printoptions(sci_mode=False)
fixed = repr(m)
torch.set_printoptions(sci_mode=True)
sci = repr(t[0])
torch.set_printoptions(sci_mode=None)
auto = repr(m)
result = {"p2": p2, "p6": p6, "fixed": fixed, "sci": sci, "auto": auto}
""",
    "set_printoptions: profiles, threshold and edgeitems": """
long = torch.tensor(np.arange(1100, dtype=np.float32))
grid = torch.tensor(np.arange(100).reshape(10, 10))
torch.set_printoptions(profile="full")
full = repr(long)
torch.set_printoptions(profile="default")
default = repr(long)
torch.set_printoptions(profile="short")
short = repr(torch.tensor([0.123456, 2.5]))
torch.set_printoptions(profile="default", threshold=5, edgeitems=1)
small = repr(grid)
result = {"full_has_dots": "..." in full, "full_len": len(full), "default": default, "short": short, "small": small}
""",
}

CHILD = r"""
import json, sys
import numpy as np
mode, mnist, cases = sys.argv[1], sys.argv[2], json.loads(sys.stdin.read())
if mode == "standin":
    sys.path.insert(0, sys.argv[3])
import torch
assert ("scripts/torch-standin/src" in torch.__file__) == (mode == "standin"), torch.__file__
with np.load(mnist) as data:
    digit = np.asarray(data["x_test"][0], dtype=np.uint8)
out = {}
for name, code in cases.items():
    torch.set_printoptions(profile="default", sci_mode=None)
    scope = {"np": np, "torch": torch, "rng": np.random.default_rng(7), "digit": digit}
    try:
        exec(code, scope)
        out[name] = {"ok": scope["result"]}
    except Exception as error:
        out[name] = {"error": f"{type(error).__name__}: {error}"}
print(json.dumps(out))
"""


def run(mode):
    args = [sys.executable, "-c", CHILD, mode, str(MNIST), str(SRC)]
    done = subprocess.run(args, input=json.dumps(CASES), capture_output=True, text=True, timeout=600)
    if done.returncode:
        sys.exit(f"{mode} run failed:\n{done.stderr[-2000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def same(real, standin, path=""):
    """Differences between two answers: floats within 1e-6 relative, everything else equal."""
    if isinstance(real, float) and isinstance(standin, float):
        if math.isnan(real) and math.isnan(standin):
            return []
        return [] if math.isclose(real, standin, rel_tol=1e-6, abs_tol=1e-7) else [f"{path}: {real!r} != {standin!r}"]
    if isinstance(real, dict) and isinstance(standin, dict) and real.keys() == standin.keys():
        return [d for key in real for d in same(real[key], standin[key], f"{path}.{key}")]
    if isinstance(real, list) and isinstance(standin, list) and len(real) == len(standin):
        return [d for n, (a, b) in enumerate(zip(real, standin)) for d in same(a, b, f"{path}[{n}]")]
    if real == standin:
        return []
    if isinstance(real, str) and isinstance(standin, str) and "\n" in real + standin:
        return [f"{path}: printed differently\n  real:\n{real}\n  stand-in:\n{standin}"]
    return [f"{path}: {real!r} != {standin!r}"]


def main():
    real, standin = run("real"), run("standin")
    failed = 0
    for name in CASES:
        problems = same(real[name], standin[name], "result")
        failed += bool(problems)
        print(("ok    " if not problems else "DIFF  ") + name)
        for problem in problems:
            print("      " + problem.replace("\n", "\n      "))
    print(f"{len(CASES) - failed} of {len(CASES)} cases match real torch.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
