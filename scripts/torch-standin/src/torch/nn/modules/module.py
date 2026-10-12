# Ported from PyTorch (torch/nn/modules/module.py), BSD-3-Clause. See THIRD_PARTY_NOTICES.md.
"""torch.nn.Module, the base class of every layer and model.

A module keeps its parameters, buffers and submodules in three dicts (_parameters, _buffers,
_modules). Assigning an attribute fills them (self.fc = nn.Linear(...) registers a submodule)
and __getattr__ reads them back, as in torch. Everything that walks a model (parameters(),
state_dict(), train(), to(), apply(), repr) walks those dicts, so it sees what torch would.
"""

import itertools
from collections import OrderedDict, namedtuple
from collections.abc import Mapping

import numpy as np

from ... import _autograd, _dtype
from ..._missing import MESSAGE, not_available
from ..._tensor import NO_CUDA, Tensor, device, require_cpu
from ..parameter import Parameter

# The public torch.nn.Module methods the stand-in does not have. Asking for one gives the
# not-available message; any other missing name is torch's plain AttributeError.
_NOT_AVAILABLE = frozenset(
    """
    bfloat16 compile get_buffer get_extra_state get_parameter get_submodule half ipu mtia
    register_backward_hook register_full_backward_hook register_full_backward_pre_hook
    register_load_state_dict_post_hook register_load_state_dict_pre_hook
    register_state_dict_post_hook register_state_dict_pre_hook set_extra_state set_submodule
    share_memory to_empty type xpu
    """.split()
)


class _MissingModuleMethod(NotImplementedError, AttributeError):
    """A torch.nn.Module method the stand-in lacks; also an AttributeError, so hasattr() is False."""

    def __init__(self, name):
        super().__init__(MESSAGE.format(f"torch.nn.Module.{name}"))


class _IncompatibleKeys(namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])):
    """What load_state_dict returns."""

    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return "<All keys matched successfully>"
        return super().__repr__()

    __str__ = __repr__


class RemovableHandle:
    """What register_forward_hook returns: handle.remove() takes the hook off again."""

    _ids = itertools.count()

    def __init__(self, hooks):
        self._hooks = hooks
        self.id = next(RemovableHandle._ids)

    def remove(self):
        self._hooks.pop(self.id, None)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


def _typename(value):
    """The type name torch's messages use: torch.FloatTensor for a tensor."""
    if isinstance(value, Tensor):
        return f"torch.{_dtype.of(value._data.dtype).legacy}Tensor"
    kind = type(value)
    return kind.__qualname__ if kind.__module__ == "builtins" else f"{kind.__module__}.{kind.__qualname__}"


def _addindent(text, spaces):
    """Indent every line of `text` but the first, as nested reprs are."""
    first, *rest = text.split("\n")
    return "\n".join([first] + [" " * spaces + line for line in rest])


def _add_hook(hooks, hook, prepend):
    handle = RemovableHandle(hooks)
    hooks[handle.id] = hook
    if prepend:
        others = [(key, value) for key, value in hooks.items() if key != handle.id]
        hooks.clear()
        hooks[handle.id] = hook
        hooks.update(others)
    return handle


def _parse_to(args, kwargs):
    """(device, dtype) from the arguments of module.to(...)."""
    target, dtype = kwargs.get("device"), kwargs.get("dtype")
    for arg in args:
        if isinstance(arg, Tensor):
            target, dtype = arg.device, arg.dtype
        elif isinstance(arg, _dtype.dtype):
            dtype = arg
        elif isinstance(arg, (str, device)):
            target = arg
    return target, dtype


def _float_cast(np_dtype):
    """A conversion that changes floating point arrays to `np_dtype` and leaves others alone."""
    return lambda array: array.astype(np_dtype, copy=False) if array.dtype.kind == "f" else array


class Module:
    """Base class for all neural network modules: subclass it, build the layers in
    __init__ (after super().__init__()) and write forward()."""

    call_super_init = False
    _version = 1  # recorded in state_dict()._metadata; a layer whose saved state changed raises it
    dump_patches = False

    def __init__(self, *args, **kwargs):
        if kwargs:
            raise TypeError(f"{type(self).__name__}.__init__() got an unexpected keyword argument '{next(iter(kwargs))}'")
        if args:
            raise TypeError(f"{type(self).__name__}.__init__() takes 1 positional argument but {len(args) + 1} were given")
        for name, value in (
            ("training", True),
            ("_parameters", {}),
            ("_buffers", {}),
            ("_non_persistent_buffers_set", set()),
            ("_modules", {}),
            ("_forward_pre_hooks", {}),
            ("_forward_hooks", {}),
        ):
            object.__setattr__(self, name, value)

    def forward(self, *input):
        raise NotImplementedError(f'Module [{type(self).__name__}] is missing the required "forward" function')

    def __call__(self, *args, **kwargs):
        for hook in tuple(self._forward_pre_hooks.values()):
            changed = hook(self, args)
            if changed is not None:
                args = changed if isinstance(changed, tuple) else (changed,)
        out = self.forward(*args, **kwargs)
        for hook in tuple(self._forward_hooks.values()):
            changed = hook(self, args, out)
            if changed is not None:
                out = changed
        return out

    # Attributes: parameters, buffers and submodules live in their dicts.

    def __getattr__(self, name):
        state = self.__dict__
        for kind in ("_parameters", "_buffers", "_modules"):
            if kind in state and name in state[kind]:
                return state[kind][name]
        if name in _NOT_AVAILABLE:
            raise _MissingModuleMethod(name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        state = self.__dict__

        def forget(*places):
            for place in places:
                if name in place:
                    place.discard(name) if isinstance(place, set) else place.pop(name)

        params, modules, buffers = state.get("_parameters"), state.get("_modules"), state.get("_buffers")
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError("cannot assign parameters before Module.__init__() call")
            forget(state, buffers, modules, state["_non_persistent_buffers_set"])
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError(
                    f"cannot assign '{_typename(value)}' as parameter '{name}' (torch.nn.Parameter or None expected)"
                )
            self.register_parameter(name, value)
        elif isinstance(value, Module):
            if modules is None:
                raise AttributeError("cannot assign module before Module.__init__() call")
            forget(state, params, buffers, state["_non_persistent_buffers_set"])
            modules[name] = value
        elif modules is not None and name in modules:
            if value is not None:
                raise TypeError(
                    f"cannot assign '{_typename(value)}' as child module '{name}' (torch.nn.Module or None expected)"
                )
            modules[name] = value
        elif buffers is not None and name in buffers:
            if value is not None and not isinstance(value, Tensor):
                raise TypeError(f"cannot assign '{_typename(value)}' as buffer '{name}' (torch.nn.Buffer, torch.Tensor or None expected)")
            buffers[name] = value
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
            self._non_persistent_buffers_set.discard(name)
        elif name in self._modules:
            del self._modules[name]
        else:
            object.__delattr__(self, name)

    def __dir__(self):
        names = list(dir(type(self))) + list(self.__dict__)
        names += list(self._parameters) + list(self._modules) + list(self._buffers)
        return sorted({name for name in names if not name[0].isdigit()})

    def _check_new_name(self, name, kind, registered):
        if not isinstance(name, str):
            raise TypeError(f"{kind} name should be a string. Got {_typename(name)}")
        if "." in name:
            raise KeyError(f'{kind} name can\'t contain "."')
        if name == "":
            raise KeyError(f'{kind} name can\'t be empty string ""')
        if hasattr(self, name) and name not in registered:
            raise KeyError(f"attribute '{name}' already exists")

    def register_parameter(self, name, param):
        if "_parameters" not in self.__dict__:
            raise AttributeError("cannot assign parameter before Module.__init__() call")
        self._check_new_name(name, "parameter", self._parameters)
        if param is not None and not isinstance(param, Parameter):
            raise TypeError(
                f"cannot assign '{_typename(param)}' object to parameter '{name}' (torch.nn.Parameter or None required)"
            )
        if param is not None and param.grad_fn is not None:
            raise ValueError(
                f"Cannot assign non-leaf Tensor to parameter '{name}'. Model parameters must be created explicitly. "
                f"To express '{name}' as a function of another Tensor, compute the value in the forward() method."
            )
        self._parameters[name] = param

    def register_buffer(self, name, tensor, persistent=True):
        """A tensor that belongs to the module's state (it is in state_dict() unless not
        persistent, and moves with to()) but is not a parameter."""
        if "_buffers" not in self.__dict__:
            raise AttributeError("cannot assign buffer before Module.__init__() call")
        self._check_new_name(name, "buffer", self._buffers)
        if tensor is not None and not isinstance(tensor, Tensor):
            raise TypeError(f"cannot assign '{_typename(tensor)}' object to buffer '{name}' (torch Tensor or None required)")
        self._buffers[name] = tensor
        if persistent:
            self._non_persistent_buffers_set.discard(name)
        else:
            self._non_persistent_buffers_set.add(name)

    def add_module(self, name, module):
        if module is not None and not isinstance(module, Module):
            raise TypeError(f"{_typename(module)} is not a Module subclass")
        if "_modules" not in self.__dict__:
            raise AttributeError("cannot assign module before Module.__init__() call")
        self._check_new_name(name, "module", self._modules)
        self._modules[name] = module

    register_module = add_module

    def register_forward_pre_hook(self, hook, *, prepend=False, with_kwargs=False):
        """hook(module, args) runs before forward; a value it returns replaces the arguments."""
        if with_kwargs:
            raise not_available("torch.nn.Module.register_forward_pre_hook with with_kwargs=True")
        return _add_hook(self._forward_pre_hooks, hook, prepend)

    def register_forward_hook(self, hook, *, prepend=False, with_kwargs=False, always_call=False):
        """hook(module, args, output) runs after forward; a value it returns replaces the output."""
        if with_kwargs:
            raise not_available("torch.nn.Module.register_forward_hook with with_kwargs=True")
        return _add_hook(self._forward_hooks, hook, prepend)

    # Walking the model.

    def named_modules(self, memo=None, prefix="", remove_duplicate=True):
        if memo is None:
            memo = set()
        if self in memo:
            return
        if remove_duplicate:
            memo.add(self)
        yield prefix, self
        for name, module in self._modules.items():
            if module is not None:
                yield from module.named_modules(memo, prefix + ("." if prefix else "") + name, remove_duplicate)

    def modules(self):
        for _, module in self.named_modules():
            yield module

    def named_children(self):
        seen = set()
        for name, module in self._modules.items():
            if module is not None and module not in seen:
                seen.add(module)
                yield name, module

    def children(self):
        for _, module in self.named_children():
            yield module

    def _named_members(self, members, prefix, recurse, remove_duplicate):
        seen = set()
        modules = self.named_modules(prefix=prefix, remove_duplicate=remove_duplicate) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            for name, value in members(module):
                if value is None or value in seen:
                    continue
                if remove_duplicate:
                    seen.add(value)
                yield module_prefix + ("." if module_prefix else "") + name, value

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        return self._named_members(lambda m: m._parameters.items(), prefix, recurse, remove_duplicate)

    def parameters(self, recurse=True):
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def named_buffers(self, prefix="", recurse=True, remove_duplicate=True):
        return self._named_members(lambda m: m._buffers.items(), prefix, recurse, remove_duplicate)

    def buffers(self, recurse=True):
        for _, buffer in self.named_buffers(recurse=recurse):
            yield buffer

    def apply(self, fn):
        """fn(module) for every submodule, children first, then this module."""
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    # Modes and gradients.

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def requires_grad_(self, requires_grad=True):
        for param in self.parameters():
            param.requires_grad_(requires_grad)
        return self

    def zero_grad(self, set_to_none=True):
        for param in self.parameters():
            if param.grad is None:
                continue
            if set_to_none:
                param.grad = None
            else:
                if param.grad.grad_fn is not None:
                    param.grad = param.grad.detach()
                else:
                    param.grad.requires_grad_(False)
                param.grad.zero_()

    # Devices and dtypes.

    def _cast(self, convert):
        """Give every parameter (and its .grad) and buffer the values convert(array). Parameters
        stay the same objects, as torch's do; buffers are replaced when their dtype changes."""
        for module in self.modules():
            for param in module._parameters.values():
                if param is not None:
                    param._data = convert(param._data)
                    if param._grad is not None:
                        param._grad._data = convert(param._grad._data)
            for name, buffer in module._buffers.items():
                if buffer is not None:
                    array = convert(buffer._data)
                    if array is not buffer._data:
                        module._buffers[name] = Tensor(array)
        return self

    def to(self, *args, **kwargs):
        """module.to(device), .to(dtype), .to(device, dtype) or .to(tensor). Only the CPU
        exists here, and a dtype must be a floating point one."""
        target, dtype = _parse_to(args, kwargs)
        require_cpu(target)
        if dtype is None:
            return self
        if not dtype.is_floating_point:
            raise TypeError(f"nn.Module.to only accepts floating point or complex dtypes, but got desired dtype={dtype}")
        return self._cast(_float_cast(dtype.np))

    def cpu(self):
        return self

    def cuda(self, device=None):
        raise RuntimeError(NO_CUDA)

    def float(self):
        return self._cast(_float_cast(np.float32))

    def double(self):
        return self._cast(_float_cast(np.float64))

    # Saving and loading.

    def state_dict(self, *, destination=None, prefix="", keep_vars=False):
        """Every parameter and persistent buffer by its dotted name. The values share memory
        with the model's, as in torch, so copy.deepcopy them to keep a snapshot. As in torch,
        the dict's _metadata records each module's _version, which load_state_dict reads."""
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = {"version": self._version}
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param if keep_vars else param.detach()
        for name, buffer in self._buffers.items():
            if buffer is not None and name not in self._non_persistent_buffers_set:
                destination[prefix + name] = buffer if keep_vars else buffer.detach()
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination=destination, prefix=prefix + name + ".", keep_vars=keep_vars)
        return destination

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Copy the values in `state_dict` into the parameters and buffers. With strict, a
        missing or unexpected name is an error; the result lists them either way."""
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"Expected state_dict to be dict-like, got {type(state_dict)}.")
        if assign:
            raise not_available("torch.nn.Module.load_state_dict with assign=True")
        missing, unexpected, errors = [], [], []
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = OrderedDict(state_dict)  # a module may fill in a key; the caller's dict stays as it was

        def load(module, prefix):
            local = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local, missing, unexpected, errors)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(self, "")
        if strict:
            if unexpected:
                errors.insert(0, "Unexpected key(s) in state_dict: {}. ".format(", ".join(f'"{k}"' for k in unexpected)))
            if missing:
                errors.insert(0, "Missing key(s) in state_dict: {}. ".format(", ".join(f'"{k}"' for k in missing)))
        if errors:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(type(self).__name__, "\n\t".join(errors))
            )
        return _IncompatibleKeys(missing, unexpected)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, missing, unexpected, errors):
        persistent = ((k, v) for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set)
        local = {k: v for k, v in itertools.chain(self._parameters.items(), persistent) if v is not None}
        for name, param in local.items():
            key = prefix + name
            if key not in state_dict:
                missing.append(key)
                continue
            value = state_dict[key]
            if not isinstance(value, Tensor):
                errors.append(
                    f'While copying the parameter named "{key}", expected torch.Tensor or Tensor-like object from '
                    f"checkpoint but received {type(value)}"
                )
            elif value.shape != param.shape:
                errors.append(
                    f"size mismatch for {key}: copying a param with shape {value.shape} from checkpoint, "
                    f"the shape in current model is {param.shape}."
                )
            else:
                with _autograd.no_grad():
                    param.copy_(value)
        # As in torch, unexpected names are collected even when not strict: a dotted name must
        # lead to a submodule, a plain one to a parameter or buffer of this module.
        for key in state_dict:
            if key.startswith(prefix):
                parts = key[len(prefix) :].split(".", 1)
                if parts[0] not in (self._modules if len(parts) > 1 else local):
                    unexpected.append(key)

    # Printing.

    def extra_repr(self):
        """The layer's own settings, shown inside its repr: Linear(in_features=2, ...)."""
        return ""

    def _get_name(self):
        return type(self).__name__

    def __repr__(self):
        extra = self.extra_repr()
        extra_lines = extra.split("\n") if extra else []
        child_lines = [f"({key}): " + _addindent(repr(module), 2) for key, module in self._modules.items()]
        lines = extra_lines + child_lines
        text = self._get_name() + "("
        if lines:
            if len(extra_lines) == 1 and not child_lines:
                text += extra_lines[0]
            else:
                text += "\n  " + "\n  ".join(lines) + "\n"
        return text + ")"
