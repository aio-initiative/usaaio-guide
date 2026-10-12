# Ported from PyTorch (torch/optim/optimizer.py), BSD-3-Clause. See THIRD_PARTY_NOTICES.md.
"""torch.optim.Optimizer: the parameter groups, per-parameter state and bookkeeping that SGD,
Adam and AdamW share, and that a notebook's own optimiser subclass builds on.

An optimiser changes its parameters through their NumPy arrays, outside autograd (the same
thing torch's own step does under no_grad), and then counts the change in the parameter's
version counter, so a backward pass that saved the old values refuses them as in torch.
Each step runs with NumPy's floating point warnings off: a run that diverges to inf and nan
is silent in torch, so it must not print RuntimeWarnings here.
"""

import warnings
from collections import defaultdict

import numpy as np

from .. import _autograd
from .._tensor import Tensor


class _RequiredParameter:
    def __repr__(self):
        return "<required parameter>"


# What a custom optimiser puts in its defaults for a hyperparameter the caller must give.
required = _RequiredParameter()


def typename(obj):
    """torch.typename: how torch's error messages name a tensor's type or any other object."""
    if isinstance(obj, Tensor):
        return f"torch.{obj.dtype.legacy}Tensor"
    kind = type(obj)
    return kind.__name__ if kind.__module__ == "builtins" else f"{kind.__module__}.{kind.__name__}"


def updated(param):
    """Record that the optimiser changed `param`'s array in place."""
    _autograd.counter(param)[0] += 1


def lerp_(start, end, weight):
    """start += weight * (end - start) in place, taken from the nearer end as torch's lerp is."""
    if weight < 0.5:
        start += weight * (end - start)
    else:
        start[...] = end - (end - start) * (1 - weight)


def quiet():
    """The context an optimiser's arithmetic runs in: overflow, invalid values and division by
    zero give inf and nan without a warning, as in torch."""
    return np.errstate(all="ignore")


class Optimizer:
    def __init__(self, params, defaults):
        self.defaults = defaults
        if isinstance(params, Tensor):
            raise TypeError(
                "params argument given to the optimizer should be an iterable of Tensors or dicts, "
                f"but got {typename(params)}"
            )
        self.state = defaultdict(dict)
        self.param_groups = []
        groups = list(params)
        if len(groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(groups[0], dict):
            groups = [{"params": groups}]
        for group in groups:
            self.add_param_group(group)

    def __repr__(self):
        text = f"{type(self).__name__} ("
        for number, group in enumerate(self.param_groups):
            text += f"\nParameter Group {number}\n"
            for key in sorted(group):
                if key != "params":
                    text += f"    {key}: {group[key]}\n"
        return text + ")"

    def step(self, closure=None):
        raise NotImplementedError

    @staticmethod
    def _evaluate(closure):
        """The loss from `closure`, run with gradients on whatever the caller's mode is."""
        if closure is None:
            return None
        with _autograd.enable_grad():
            return closure()

    def zero_grad(self, set_to_none=True):
        """Forget every parameter's gradient, or (set_to_none=False) fill it with zeros."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if set_to_none:
                    p.grad = None
                    continue
                if p.grad.grad_fn is not None:  # a gradient made with create_graph=True
                    p.grad = p.grad.detach()
                p.grad.requires_grad_(False)
                p.grad.zero_()

    def add_param_group(self, param_group):
        if not isinstance(param_group, dict):
            raise TypeError(f"param_group must be a dict, but got {type(param_group)}")
        params = param_group["params"]
        if isinstance(params, Tensor):
            params = [params]
        elif isinstance(params, set):
            raise TypeError(
                "optimizer parameters need to be organized in ordered collections, but the ordering of "
                "tensors in sets will change between runs. Please use a list instead."
            )
        else:
            params = list(params)
        # model.named_parameters() gives (name, tensor) pairs; the names ride along in the group.
        names = [p[0] for p in params if isinstance(p, tuple)]
        params = [p[1] if isinstance(p, tuple) else p for p in params]
        if names:
            if len(names) != len(params):
                raise ValueError("all optimizer params should be with/without names. Some param names are missing")
            param_group["param_names"] = names
        param_group["params"] = params
        for p in params:
            if not isinstance(p, Tensor):
                raise TypeError(f"optimizer can only optimize Tensors, but one of the params is {typename(p)}")
            if not (p.is_leaf or p.retains_grad):
                raise ValueError("can't optimize a non-leaf Tensor")
        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError(f"parameter group didn't specify a value of required optimization parameter {name}")
            param_group.setdefault(name, default)
        if len({id(p) for p in params}) != len(params):
            warnings.warn(
                "optimizer contains a parameter group with duplicate parameters; in future, this will cause an "
                "error; see github.com/pytorch/pytorch/issues/40967 for more information",
                stacklevel=3,
            )
        held = {id(p) for group in self.param_groups for p in group["params"]}
        for group in self.param_groups:
            if ("param_names" in param_group) != ("param_names" in group):
                kind = "with names" if "param_names" in param_group else "without names"
                raise ValueError(
                    f"all optimizer param groups should be with/without names. cannot add param group {kind} "
                    "to the optimizer"
                )
        if held & {id(p) for p in params}:
            raise ValueError("some parameters appear in more than one parameter group")
        self.param_groups.append(param_group)

    def state_dict(self):
        """The state and the hyperparameters, with each parameter named by its position."""
        ids = {}
        groups = []
        for group in self.param_groups:
            for p in group["params"]:
                ids.setdefault(id(p), len(ids))
            packed = {k: v for k, v in group.items() if k != "params"}
            packed["params"] = [ids[id(p)] for p in group["params"]]
            groups.append(packed)
        state = {(ids[id(k)] if isinstance(k, Tensor) else k): v for k, v in self.state.items()}
        return {"state": state, "param_groups": groups}

    def load_state_dict(self, state_dict):
        """Take the state and hyperparameters of an earlier state_dict, in the same order of
        parameters. As in torch, state of the right dtype is used as it is, not copied."""
        groups = self.param_groups
        saved = [dict(g) for g in state_dict["param_groups"]]
        if len(groups) != len(saved):
            raise ValueError("loaded state dict has a different number of parameter groups")
        if any(len(g["params"]) != len(s["params"]) for g, s in zip(groups, saved)):
            raise ValueError("loaded state dict contains a parameter group that doesn't match the size of optimizer's group")
        by_id = {i: p for g, s in zip(groups, saved) for i, p in zip(s["params"], g["params"])}
        state = defaultdict(dict)
        with quiet():
            for key, value in state_dict["state"].items():
                state[by_id[key]] = _cast(by_id[key], value) if key in by_id else value
        for group, new in zip(groups, saved):
            new["params"] = group["params"]
            if "param_names" in group and "param_names" not in new:
                new["param_names"] = group["param_names"]
        self.state = state
        self.param_groups = saved


def _cast(param, value, key=None):
    """`value` (a tensor or the dicts and lists holding them) in the parameter's dtype. The
    step counter keeps its own."""
    if isinstance(value, Tensor):
        if key == "step" or not param.is_floating_point() or value.dtype == param.dtype:
            return value
        return Tensor(value._data.astype(param._data.dtype))
    if isinstance(value, dict):
        return {k: _cast(param, v, k) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cast(param, v) for v in value)
    return value
