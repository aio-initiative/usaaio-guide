# Ported from PyTorch (torch/nn/modules/container.py), BSD-3-Clause. See THIRD_PARTY_NOTICES.md.
"""Sequential, ModuleList and ModuleDict: modules that hold other modules."""

import itertools
import operator
from collections import OrderedDict
from collections.abc import Iterable, Mapping

from .module import Module, _addindent


class Sequential(Module):
    """Runs its modules in order, each on the previous one's output. Built from modules
    (named "0", "1", ...) or from an OrderedDict of names and modules."""

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for index, module in enumerate(args):
                self.add_module(str(index), module)

    def _key_at(self, index):
        size = len(self)
        index = operator.index(index)
        if not -size <= index < size:
            raise IndexError(f"index {index} is out of range")
        return next(itertools.islice(self._modules, index % size, None))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return type(self)(OrderedDict(list(self._modules.items())[index]))
        return self._modules[self._key_at(index)]

    def __setitem__(self, index, module):
        setattr(self, self._key_at(index), module)

    def __delitem__(self, index):
        keys = list(self._modules)[index] if isinstance(index, slice) else [self._key_at(index)]
        for key in keys:
            delattr(self, key)
        # Number the rest again, as torch does.
        self._modules = {str(i): module for i, module in enumerate(self._modules.values())}

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def append(self, module):
        self.add_module(str(len(self)), module)
        return self

    def insert(self, index, module):
        if not isinstance(module, Module):
            raise AssertionError(f"module should be of type: {Module}")
        size = len(self)
        if not -size <= index <= size:
            raise IndexError(f"Index out of range: {index}")
        index = index + size if index < 0 else index
        for i in range(size, index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module
        return self

    def extend(self, sequential):
        for layer in sequential:
            self.append(layer)
        return self

    def pop(self, key):
        module = self[key]
        del self[key]
        return module

    def forward(self, input):
        for module in self:
            input = module(input)
        return input


class ModuleList(Module):
    """A list of modules whose parameters belong to the model. It has no forward()."""

    def __init__(self, modules=None):
        super().__init__()
        if modules is not None:
            self += modules

    def _key_at(self, index):
        size = len(self)
        index = operator.index(index)
        if not -size <= index < size:
            raise IndexError(f"index {index} is out of range")
        return str(index % size)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return type(self)(list(self._modules.values())[index])
        return self._modules[self._key_at(index)]

    def __setitem__(self, index, module):
        setattr(self, self._key_at(index), module)

    def __delitem__(self, index):
        keys = list(self._modules)[index] if isinstance(index, slice) else [self._key_at(index)]
        for key in keys:
            del self._modules[key]
        self._modules = {str(i): module for i, module in enumerate(self._modules.values())}

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __iadd__(self, modules):
        return self.extend(modules)

    def __add__(self, other):
        combined = ModuleList()
        for index, module in enumerate(itertools.chain(self, other)):
            combined.add_module(str(index), module)
        return combined

    def insert(self, index, module):
        for i in range(len(self._modules), index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module

    def append(self, module):
        self.add_module(str(len(self)), module)
        return self

    def pop(self, key):
        module = self[key]
        del self[key]
        return module

    def extend(self, modules):
        if not isinstance(modules, Iterable):
            raise TypeError("ModuleList.extend should be called with an iterable, but got " + type(modules).__name__)
        offset = len(self)
        for index, module in enumerate(modules):
            self.add_module(str(offset + index), module)
        return self

    def __repr__(self):
        """Runs of identical modules are shown once, as "(0-2): 3 x Linear(...)", as in torch."""
        reprs = [repr(module) for module in self]
        if not reprs:
            return self._get_name() + "()"
        runs = []  # [first index, last index, repr]
        for index, text in enumerate(reprs):
            if runs and runs[-1][2] == text:
                runs[-1][1] = index
            else:
                runs.append([index, index, text])
        lines = []
        for first, last, text in runs:
            label = f"({first}): {text}" if first == last else f"({first}-{last}): {last - first + 1} x {text}"
            lines.append(_addindent(label, 2))
        return self._get_name() + "(\n  " + "\n  ".join(lines) + "\n)"


class ModuleDict(Module):
    """A dict of modules whose parameters belong to the model, in insertion order."""

    def __init__(self, modules=None):
        super().__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key):
        return self._modules[key]

    def __setitem__(self, key, module):
        self.add_module(key, module)

    def __delitem__(self, key):
        del self._modules[key]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules)

    def __contains__(self, key):
        return key in self._modules

    def clear(self):
        self._modules.clear()

    def pop(self, key):
        module = self[key]
        del self[key]
        return module

    def keys(self):
        return self._modules.keys()

    def items(self):
        return self._modules.items()

    def values(self):
        return self._modules.values()

    def update(self, modules):
        if not isinstance(modules, Iterable):
            raise TypeError("ModuleDict.update should be called with an iterable of key/value pairs, but got " + type(modules).__name__)
        pairs = modules.items() if isinstance(modules, (Mapping, ModuleDict)) else modules
        for key, module in pairs:
            self[key] = module
