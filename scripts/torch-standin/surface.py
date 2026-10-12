"""List every torch name the notebooks use, so the stand-in covers a measured surface.

    python surface.py [--extra DIR ...] [--other DIR]

Reads the .ipynb files under notebooks/ (exercises and solutions) and writes SURFACE.txt
next to this script: one dotted name per line, sorted.

* Every name reached through torch or torchvision, with import aliases resolved to the full
  path: nn.Linear and F.relu become torch.nn.Linear and torch.nn.functional.relu.
* Every attribute of a torch.Tensor used on some other object, written Tensor.name: methods
  that are called (x.sum()) and properties (x.shape, x.grad, x.T). A receiver that comes
  from a non-torch import (np.sum(x), expit(z).sum()), a literal or self (self.diff) is not
  a tensor, so it does not count. The check is syntactic, so a pandas or NumPy variable
  still adds a name, which is harmless when real torch has that method too.

Only notebooks that import torch or torchvision are read. A name the real torch lacks (a
variable that shadows an alias, say) is reported on stderr and left out. Run it inside the
test venv, because the real torch decides what is a Tensor attribute.

--extra DIR  another notebooks folder, merged over notebooks/ (repeatable). A file with the
             same relative path replaces the one in notebooks/.
--other DIR  a flat folder of further notebooks, read the same way.
"""

import argparse
import ast
import json
import sys
import types
from pathlib import Path

import torch
import torchvision

HERE = Path(__file__).resolve().parent
NOTEBOOKS = HERE.parents[1] / "notebooks"
OUT = HERE / "SURFACE.txt"

ROOTS = {"torch": torch, "torchvision": torchvision}
# Cell magics that wrap ordinary code. Any other %% magic makes the cell non-Python.
CODE_MAGICS = {"%%time", "%%timeit", "%%capture"}
PROPERTY_TYPES = (property, types.GetSetDescriptorType, types.MemberDescriptorType)
# Tensor attributes with no meaning on a dense tensor (sparse-only values, the named-tensor
# legacy name, a list-style index) that every notebook reaches through dict.values(), pandas
# or its own classes. They would otherwise ask the stand-in for something torch itself rejects.
NOT_ON_DENSE = {"Tensor.index", "Tensor.name", "Tensor.values"}


def tensor_attributes():
    """Public attributes of the real torch.Tensor, split into (methods, properties)."""
    methods, properties = set(), set()
    for name in dir(torch.Tensor):
        if name.startswith("_"):
            continue
        attr = getattr(torch.Tensor, name)
        if isinstance(attr, PROPERTY_TYPES):
            properties.add(name)
        elif callable(attr):
            methods.add(name)
    return methods, properties


TENSOR_METHODS, TENSOR_PROPERTIES = tensor_attributes()


def find_notebooks(folder):
    """Map each notebook's path relative to folder to its absolute path."""
    return {
        str(path.relative_to(folder)): path
        for path in sorted(folder.rglob("*.ipynb"))
        if ".ipynb_checkpoints" not in path.parts
    }


def code_cells(path):
    """The Python source of each code cell, with IPython magics turned into no-ops."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        lines = "".join(cell["source"]).split("\n")
        if lines[0].startswith("%%"):
            if lines[0].split()[0] not in CODE_MAGICS:
                continue
            lines = lines[1:]
        cleaned = []
        for line in lines:
            body = line.lstrip()
            if body.startswith(("%", "!")):
                line = line[: len(line) - len(body)] + "pass"
            cleaned.append(line)
        cells.append((index, "\n".join(cleaned)))
    return cells


def parse_cells(path):
    """(syntax trees, problems): a cell that does not parse is skipped and described."""
    trees, problems = [], []
    for index, source in code_cells(path):
        try:
            trees.append(ast.parse(source))
        except SyntaxError as err:
            problems.append(f"warning: {path.name} cell {index} does not parse ({err.msg})")
    return trees, problems


class Imports(ast.NodeVisitor):
    """What the import statements of one notebook bind.

    aliases: local name -> dotted torch path (nn -> torch.nn)
    foreign: local names bound by non-torch imports (np, plt, special, expit)
    names:   dotted torch paths the imports themselves mention (torch.nn.functional)
    """

    def __init__(self):
        self.aliases, self.foreign, self.names = {}, set(), set()

    def visit_Import(self, node):
        for item in node.names:
            root = item.name.split(".")[0]
            if root not in ROOTS:
                self.foreign.add(item.asname or root)
                continue
            self.names.add(item.name)
            if item.asname:
                self.aliases[item.asname] = item.name
            else:
                self.aliases[root] = root

    def visit_ImportFrom(self, node):
        if node.level or node.module is None:
            return
        if node.module.split(".")[0] not in ROOTS:
            self.foreign.update(item.asname or item.name for item in node.names)
            return
        for item in node.names:
            if item.name == "*":
                print(f"warning: from {node.module} import * hides names", file=sys.stderr)
                continue
            path = f"{node.module}.{item.name}"
            self.names.add(path)
            self.aliases[item.asname or item.name] = path


class Uses(ast.NodeVisitor):
    """The torch names and Tensor attributes one notebook uses."""

    def __init__(self, imports):
        self.imports = imports
        self.torch_names = set()
        self.tensor_names = set()

    def torch_path(self, node):
        """The full dotted path if node is an alias followed only by attributes, else None."""
        attrs = []
        while isinstance(node, ast.Attribute):
            attrs.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name) and node.id in self.imports.aliases:
            return ".".join([self.imports.aliases[node.id], *reversed(attrs)])
        return None

    def receiver_is_foreign(self, node):
        """True when node is not a tensor: a non-torch import, a literal, or self itself."""
        if isinstance(node, ast.Name) and node.id == "self":
            return True  # self.diff is the notebook's own attribute; self.w.grad is not
        while isinstance(node, (ast.Attribute, ast.Call, ast.Subscript)):
            node = node.func if isinstance(node, ast.Call) else node.value
        if isinstance(node, (ast.Constant, ast.JoinedStr)):
            return True
        return isinstance(node, ast.Name) and node.id in self.imports.foreign

    def visit_Name(self, node):
        path = self.imports.aliases.get(node.id)
        # Covers names imported directly ("from torch.nn import Linear", then Linear(...)).
        if path and isinstance(node.ctx, ast.Load):
            self.torch_names.add(path)

    def visit_Attribute(self, node):
        path = self.torch_path(node)
        if path:
            self.torch_names.add(path)
            return
        if node.attr in TENSOR_PROPERTIES and not self.receiver_is_foreign(node.value):
            self.tensor_names.add(f"Tensor.{node.attr}")
        self.generic_visit(node)

    def visit_Call(self, node):
        func = node.func
        if isinstance(func, ast.Attribute) and not self.torch_path(func):
            if func.attr in TENSOR_METHODS and not self.receiver_is_foreign(func.value):
                self.tensor_names.add(f"Tensor.{func.attr}")
            self.visit(func.value)
            for child in [*node.args, *node.keywords]:
                self.visit(child)
        else:
            self.generic_visit(node)


def resolves_on_real_torch(dotted):
    root, *parts = dotted.split(".")
    obj = ROOTS[root]
    try:
        for part in parts:
            obj = getattr(obj, part)
    except AttributeError:
        return False
    return True


def scan(path):
    """(torch names, Tensor names) for one notebook, or None if it does not import torch."""
    trees, problems = parse_cells(path)
    imports = Imports()
    for tree in trees:
        imports.visit(tree)
    if not imports.aliases:
        return None
    for problem in problems:
        print(problem, file=sys.stderr)
    uses = Uses(imports)
    for tree in trees:
        uses.visit(tree)
    return imports.names | uses.torch_names, uses.tensor_names


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--extra", action="append", default=[], type=Path, metavar="DIR")
    parser.add_argument("--other", type=Path, metavar="DIR")
    args = parser.parse_args()

    notebooks = find_notebooks(NOTEBOOKS)
    for folder in args.extra:
        notebooks.update(find_notebooks(folder))
    if args.other:
        notebooks.update({f"other/{rel}": p for rel, p in find_notebooks(args.other).items()})

    torch_names, tensor_names, scanned = set(), set(), 0
    for rel, path in notebooks.items():
        found = scan(path)
        if found is None:
            continue
        scanned += 1
        torch_names |= found[0]
        tensor_names |= found[1]

    # An import of the package itself ("torch") is not a name to implement.
    torch_names = {name for name in torch_names if "." in name}
    missing = sorted(name for name in torch_names if not resolves_on_real_torch(name))
    for name in missing:
        print(f"warning: {name} is not in the real torch, left out", file=sys.stderr)
    names = sorted((torch_names - set(missing)) | (tensor_names - NOT_ON_DENSE))

    OUT.write_text("".join(f"{name}\n" for name in names), encoding="utf-8")
    print(
        f"{scanned} of {len(notebooks)} notebooks import torch; wrote {len(names)} names "
        f"({len(tensor_names - NOT_ON_DENSE)} Tensor attributes) to {OUT.name}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
