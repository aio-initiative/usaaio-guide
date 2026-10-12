"""Autograd: the graph operations record, and the pass that runs it backwards.

Every differentiable operation returns _ops.result(data, *parents, name=..., grads=...,
saved=...), which calls record() here. When gradients are on and a parent requires grad,
record() gives the output a Node (its grad_fn): an edge to each parent's own node (the
parent's grad_fn, or the AccumulateGrad of a leaf), one formula per parent, and the values
the formulas read. A formula takes the output's gradient and the saved values and returns
the parent's gradient, written with tensor operations; so running the formulas with
gradients on (create_graph=True) records the backward pass too, which gives second
derivatives. The engine sums a gradient back over broadcast dimensions and casts it to the
parent's dtype, so formulas need not.

In-place changes go through write(), which keeps torch's rules:
- a version counter per block of memory, shared by a tensor, its views and its detached
  copies, so a saved tensor that changed is caught when the backward pass reads it;
- a leaf that requires grad, or a view of one, cannot change in place while gradients are on;
- a tensor written in place gets a new history (CopySlices). A write through a view gives the
  view's base the new history, and every view of that base re-derives its grad_fn from the
  base the next time it is read (AsStridedBackward0), as torch does.
"""

import functools
import inspect
import weakref

import numpy as np

from . import _dtype
from ._tensor import Tensor

_enabled = True


def is_grad_enabled():
    return _enabled


def _set_enabled(mode):
    global _enabled
    _enabled = bool(mode)


class _GradMode:
    """A context manager that also works as a decorator, setting gradient mode to `mode`."""

    mode = True

    def __enter__(self):
        self._previous = _enabled
        _set_enabled(self.mode)

    def __exit__(self, *exc):
        _set_enabled(self._previous)
        return False

    def __call__(self, function):
        mode = self.mode
        if inspect.isgeneratorfunction(function):
            return _generator_in_mode(function, mode)

        @functools.wraps(function)
        def run(*args, **kwargs):
            previous = _enabled
            _set_enabled(mode)
            try:
                return function(*args, **kwargs)
            finally:
                _set_enabled(previous)

        return run


def _generator_in_mode(function, mode):
    """A generator function whose every step (not only its creation) runs in `mode`, as torch's
    decorators do; what the caller sends or throws in is passed on."""

    @functools.wraps(function)
    def generator(*args, **kwargs):
        previous = _enabled
        _set_enabled(mode)
        try:
            steps = function(*args, **kwargs)
        finally:
            _set_enabled(previous)
        resume, value = steps.send, None
        while True:
            previous = _enabled
            _set_enabled(mode)
            try:
                item = resume(value)
            except StopIteration as stop:
                return stop.value
            finally:
                _set_enabled(previous)
            try:
                resume, value = steps.send, (yield item)
            except GeneratorExit:
                steps.close()
                raise
            except BaseException as err:
                resume, value = steps.throw, err

    return generator


class no_grad(_GradMode):
    """Operations inside record no graph: with torch.no_grad(): ..., or @torch.no_grad()."""

    mode = False

    def __new__(cls, orig_func=None):
        self = super().__new__(cls)
        return self(orig_func) if callable(orig_func) else self

    def __init__(self, orig_func=None):
        pass


class enable_grad(no_grad):
    """Turns gradients back on inside a no_grad block."""

    mode = True


class set_grad_enabled(_GradMode):
    """set_grad_enabled(False) switches gradients off at once; as a context manager or a
    decorator it puts the previous mode back afterwards."""

    def __init__(self, mode):
        self.mode = bool(mode)
        self._previous = _enabled
        _set_enabled(mode)

    def __enter__(self):
        pass

    def __call__(self, function):
        _set_enabled(self._previous)  # used as a decorator: only the decorated call changes mode
        return super().__call__(function)


# The graph.


class Node:
    """A step of the backward pass: the grad_fn of the tensor one operation produced."""

    __slots__ = ("_edges", "_formulas", "_meta", "_saved", "_retained", "__weakref__")

    def name(self):
        return type(self).__name__

    @property
    def next_functions(self):
        return tuple((edge, 0) for edge in self._edges)

    def __repr__(self):
        return f"<{type(self).__name__} object at {id(self):#x}>"


class AccumulateGrad(Node):
    """Where a leaf's gradient arrives: it is added to the leaf's .grad."""

    __slots__ = ("_leaf",)

    def __init__(self, leaf):
        self._leaf = weakref.ref(leaf)
        self._edges, self._formulas, self._meta, self._saved, self._retained = (), (), (), (), ()

    @property
    def variable(self):
        return self._leaf()


_NODE_TYPES = {}


def _node_type(name):
    """One Node subclass per operation name, so type(y.grad_fn).__name__ is "MulBackward0"."""
    kind = _NODE_TYPES.get(name)
    if kind is None:
        kind = _NODE_TYPES[name] = type(name, (Node,), {"__slots__": ()})
    return kind


def counter(tensor):
    """The tensor's version counter, a one-item list shared with its views and detached copies."""
    if tensor._vc is None:
        tensor._vc = [0]
    return tensor._vc


def requires_grad(tensor):
    return tensor._requires_grad or grad_fn(tensor) is not None


def grad_fn(tensor):
    """The tensor's grad_fn; a view whose base changed in place first re-derives it."""
    if tensor._base is not None and tensor._view_version is not None and tensor._view_version != tensor._vc[0]:
        _rederive(tensor)
    return tensor._grad_fn


def edge(tensor):
    """Where gradient for `tensor` goes: its grad_fn, a leaf's AccumulateGrad, or None."""
    fn = grad_fn(tensor)
    if fn is not None:
        return fn
    if tensor._requires_grad:
        if tensor._accumulator is None:
            tensor._accumulator = AccumulateGrad(tensor)
        return tensor._accumulator
    return None


OUTPUT = object()  # in saved=(...), stands for the operation's own output
_TENSOR, _OUTPUT, _PLAIN = range(3)
_FREED = "freed"


def _save(value, out):
    if value is OUTPUT:
        vc = counter(out)
        return (_OUTPUT, out._data, vc, vc[0])
    if isinstance(value, Tensor):
        vc = counter(value)
        return (_TENSOR, value, vc, vc[0])
    return (_PLAIN, value, None, None)


def _make_node(name, edges, formulas, parents, saved, out):
    node = _node_type(name)()
    node._edges = edges
    node._formulas = formulas
    node._meta = tuple((p._data.shape, p._data.dtype) if isinstance(p, Tensor) else None for p in parents)
    node._saved = tuple(_save(value, out) for value in saved)
    node._retained = ()
    return node


def record(out, parents, name, grads, saved):
    """Link `out`, just made from `parents`, into the graph (see the module docstring)."""
    data = out._data
    if data.base is not None:
        for parent in parents:
            if isinstance(parent, Tensor) and np.may_share_memory(data, parent._data):
                root = parent if parent._base is None else parent._base
                out._base, out._vc = root, counter(root)
                out._view_version = out._vc[0] if _enabled else None
                break
    if not _enabled or not grads or data.dtype.kind != "f":
        return
    edges = tuple(
        edge(p) if formula is not None and isinstance(p, Tensor) else None for p, formula in zip(parents, grads)
    )
    if any(e is not None for e in edges):
        out._grad_fn = _make_node(name, edges, grads, parents, saved, out)


def _unpack(node):
    """The saved values, checked: none may have changed in place since it was saved."""
    if node._saved is _FREED:
        raise RuntimeError(
            "Trying to backward through the graph a second time (or directly access saved tensors after "
            "they have already been freed). Saved intermediate values of the graph are freed when you call "
            ".backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the "
            "graph a second time or if you need to access saved tensors after calling backward."
        )
    values = []
    for kind, value, vc, version in node._saved:
        if kind == _PLAIN:
            values.append(value)
            continue
        if vc[0] != version:
            _modified(value if kind == _TENSOR else Tensor(value), vc[0], version, node)
        if kind == _TENSOR:
            values.append(value)
        else:  # the output, rebuilt with this node as its history (as torch does)
            out = Tensor(value)
            out._grad_fn = node
            values.append(out)
    return values


def _modified(tensor, now, then, node):
    data = tensor._data
    owner = tensor._grad_fn if tensor._grad_fn is not None else (node if tensor._base is None else None)
    which = "" if owner is None else f", which is output 0 of {type(owner).__name__}"
    raise RuntimeError(
        "one of the variables needed for gradient computation has been modified by an inplace operation: "
        f"[torch.{_dtype.of(data.dtype).legacy}Tensor {list(data.shape)}]{which}, is at version {now}; "
        f"expected version {then} instead. Hint: enable anomaly detection to find the operation that failed "
        "to compute its gradient, with torch.autograd.set_detect_anomaly(True)."
    )


# Views and in-place writes.


def _addresses(array):
    """The memory address of every element of `array`."""
    start = array.__array_interface__["data"][0]
    out = np.full(array.shape, start, dtype=np.int64)
    for axis, (n, step) in enumerate(zip(array.shape, array.strides)):
        out += (np.arange(n, dtype=np.int64) * step).reshape((n,) + (1,) * (array.ndim - axis - 1))
    return out


def positions_in(view, base):
    """The row-major position in `base` of every element of `view`, an array that shares its memory."""
    if base.flags.c_contiguous:
        start = base.__array_interface__["data"][0]
        return ((_addresses(view) - start) // base.itemsize).astype(np.intp)
    flat = _addresses(base).reshape(-1)
    order = np.argsort(flat, kind="stable")
    return order[np.searchsorted(flat[order], _addresses(view))].astype(np.intp)


def _retain_on(node, tensor):
    node._retained = (*node._retained, weakref.ref(tensor))


def _rederive(view):
    """Give a view whose base changed in place a grad_fn that reads it out of the base."""
    from ._index import put_flat

    view._view_version = view._vc[0]
    base = view._base
    base_edge = edge(base)
    if base_edge is None:
        return
    shape, positions = base._data.shape, positions_in(view._data, base._data)
    node = _make_node(
        "AsStridedBackward0", (base_edge,), (lambda g: put_flat(shape, positions, g),), (base,), (), view
    )
    view._grad_fn = node
    if view._retains_grad:
        _retain_on(node, view)


def _leaf_error(view):
    if view:
        return RuntimeError("a view of a leaf Variable that requires grad is being used in an in-place operation.")
    return RuntimeError("a leaf Variable that requires grad is being used in an in-place operation.")


def check_inplace(target, indexed=False):
    """Refuse, as torch does, to change a leaf that requires grad (or a view of one) in place
    while gradients are on. Writing by index (x[i] = v) counts as writing through a view."""
    if not _enabled:
        return
    if target._base is None:
        if target._grad_fn is None and target._requires_grad:
            raise _leaf_error(indexed)
    elif target._base._grad_fn is None and target._base._requires_grad:
        raise _leaf_error(True)


def original(target, *others):
    """The tensor an in-place operation should read and save: `target` itself, or, when the
    operation will be recorded, a copy with the same history, because `target` is about to
    change and torch saves the value from before the change."""
    check_inplace(target)
    if not _enabled or not any(isinstance(t, Tensor) and requires_grad(t) for t in (target, *others)):
        return target
    copy = Tensor(target._data.copy())
    copy._grad_fn = grad_fn(target)
    return copy


def _is_whole(ints, key):
    return ints is None and len(key) == 1 and key[0] is Ellipsis


def write(target, key, value, ints=None, name="CopySlices", indexed=False, adopt=False):
    """target._data[ints][key] = value, with the version counter and autograd kept right.

    `adopt` is for the in-place forms of operations (x.add_(y)): `value` is the out-of-place
    answer, freshly made, and a whole tensor that is not a view takes its history.
    """
    check_inplace(target, indexed)
    root = target if target._base is None else target._base
    root_edge = edge(root) if _enabled else None
    value_tracked = _enabled and isinstance(value, Tensor) and requires_grad(value)
    track = (root_edge is not None or value_tracked) and root._data.dtype.kind == "f"
    whole = target is root and _is_whole(ints, key)
    if track and not whole:
        # The written positions in the root, kept for the backward pass (its size is the
        # region's, so a loop of small writes stays small).
        if target is root:
            region = np.arange(root._data.size, dtype=np.intp).reshape(root._data.shape)
        else:
            region = positions_in(target._data, root._data)
        if ints is not None:
            region = region[ints]
        region = np.array(region[key])  # a copy: a view would keep the whole root-sized array alive
    destination = target._data if ints is None else target._data[ints]
    destination[key] = value._data if isinstance(value, Tensor) else value
    counter(root)[0] += 1
    if not track:
        return target
    if adopt and whole and value._grad_fn is not None:
        node = value._grad_fn
    else:
        from ._index import take_flat
        from ._ops import masked, zeros_for

        shape = root._data.shape
        if whole:  # every value replaced: none of the old ones gets a gradient
            formulas = (zeros_for, lambda g: g)
        else:

            def rest(g):
                """g where the old values were kept, 0 (not g * 0) where they were written."""
                kept = np.ones(root_size, dtype=bool)
                kept[region.reshape(-1)] = False
                return masked(kept.reshape(shape), g)

            root_size = root._data.size
            formulas = (rest, lambda g: take_flat(g, region))
        parents = (root, value)
        edges = (root_edge, edge(value) if value_tracked else None)
        node = _make_node(name, edges, formulas, parents, (), root)
    root._grad_fn = node
    if root._retains_grad:
        _retain_on(node, root)
    return target


# The backward pass.


def _fit(grad, meta):
    """`grad` summed back over broadcast dimensions to the parent's shape, in its dtype."""
    from ._ops import cast
    from ._reduce import sum
    from ._shape import reshape

    shape, dtype = meta
    if grad._data.shape != shape:
        lead = grad._data.ndim - len(shape)
        axes = tuple(range(lead)) + tuple(
            lead + i for i, n in enumerate(shape) if n == 1 and grad._data.shape[lead + i] != 1
        )
        if axes:
            grad = sum(grad, axes, keepdim=True)
        grad = reshape(grad, *shape)
    if grad._data.dtype != dtype:
        grad = cast(grad, dtype)
    return grad


def _store_grad(tensor, grad, create_graph):
    """Add `grad` into tensor.grad: in place normally, recorded when create_graph is on."""
    from ._ops import add

    if tensor._grad is None:
        # Always a fresh, writable copy: never the caller's gradient tensor, another leaf's
        # .grad or a read-only broadcast view (a sum's gradient is one).
        if create_graph:
            from ._ops import clone

            tensor._grad = clone(grad)
        else:
            tensor._grad = Tensor(np.array(grad._data))
    elif create_graph or requires_grad(tensor._grad):
        tensor._grad = add(tensor._grad, grad)
    else:
        old = tensor._grad._data
        if old.shape != grad._data.shape:  # .grad kept its shape after x.data = (another shape)
            raise RuntimeError(
                f"output with shape {list(old.shape)} doesn't match the broadcast shape {list(grad._data.shape)}"
            )
        np.add(old, grad._data, out=old, casting="unsafe")


def _order(starts):
    """Every node reachable from `starts`, each after all the nodes it leads to."""
    order, seen = [], set()
    for start in starts:
        if start in seen:
            continue
        seen.add(start)
        stack = [(start, iter(start._edges))]
        while stack:
            node, edges = stack[-1]
            for e in edges:
                if e is not None and e not in seen:
                    seen.add(e)
                    stack.append((e, iter(e._edges)))
                    break
            else:
                stack.pop()
                order.append(node)
    return order


def run_backward(outputs, grads, targets=None, retain_graph=False, create_graph=False, accumulate=True):
    """Run the graph backwards from `outputs`, whose gradients are `grads`.

    With `targets` (a list of nodes) only the paths to them run, and the gradient that reaches
    each is returned in a dict; .grad is filled only for target leaves, and only if
    `accumulate`. Without targets every leaf on the way gets its .grad, and so does every
    tensor that called retain_grad().
    """
    from ._ops import add

    starts = []
    for i, (out, grad) in enumerate(zip(outputs, grads)):
        e = edge(out)
        if e is None:
            raise RuntimeError(f"element {i} of tensors does not require grad and does not have a grad_fn")
        starts.append((e, grad))
    order = _order([e for e, _ in starts])
    wanted = None if targets is None else set(targets)
    if wanted is None:
        needed = set(order)
    else:
        needed = set()
        for node in order:  # a node is needed when it is a target or leads to a needed node
            if node in wanted or any(e in needed for e in node._edges):
                needed.add(node)
    buffer, captured, ran = {}, {}, []
    previous = _enabled
    _set_enabled(create_graph)
    try:
        for e, grad in starts:
            buffer[e] = grad if e not in buffer else add(buffer[e], grad)
        for node in reversed(order):
            if node not in needed:
                continue
            grad = buffer.pop(node, None)
            if grad is None:
                continue
            for ref in node._retained:
                tensor = ref()
                if tensor is not None and tensor._grad_fn is node and wanted is None:
                    _store_grad(tensor, grad, create_graph)
            if wanted is not None and node in wanted:
                captured[node] = grad
            if isinstance(node, AccumulateGrad):
                leaf = node.variable
                if leaf is not None and accumulate and (wanted is None or node in wanted):
                    _store_grad(leaf, grad, create_graph)
                continue
            values = _unpack(node)
            ran.append(node)
            for e, formula, meta in zip(node._edges, node._formulas, node._meta):
                if e is None or e not in needed:
                    continue
                part = formula(grad, *values)
                if part is None:
                    continue
                part = _fit(part, meta)
                buffer[e] = part if e not in buffer else add(buffer[e], part)
    finally:
        _set_enabled(previous)
    if not retain_graph:
        for node in ran:
            if node._saved:
                node._saved = _FREED
    return captured


def _as_tuple(value):
    """One tensor as a one-tuple, any other iterable of tensors as a tuple.

    A Tensor is iterable too (a 0 by one tensor yields nothing), so it is checked first: real
    torch's autograd.grad takes a single output or a sequence of them, and a single tensor has to
    stay single. Everything else is taken as a sequence, which is why `model.parameters()` (a
    generator) works where a list of parameters would.
    """
    if value is None:
        return None
    if isinstance(value, Tensor):
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(value)
    if hasattr(value, "__iter__"):
        return tuple(value)
    return (value,)


def make_grads(outputs, grads):
    """The gradient to start each output from: ones for a single value, else the one given."""
    if grads is None:
        grads = (None,) * len(outputs)
    if len(grads) != len(outputs):
        raise RuntimeError(f"grad_outputs has {len(grads)} tensors but outputs has {len(outputs)}")
    made = []
    for i, (out, grad) in enumerate(zip(outputs, grads)):
        if grad is None:
            if requires_grad(out) and out._data.size != 1:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
            grad = Tensor(np.ones(out._data.shape, dtype=out._data.dtype))
        else:
            if not isinstance(grad, Tensor):
                raise TypeError(f"gradients can be either Tensors or None, but got {type(grad).__name__}")
            if grad._data.shape != out._data.shape:
                raise RuntimeError(
                    f"Mismatch in shape: grad_output[{i}] has a shape of torch.Size({list(grad._data.shape)}) "
                    f"and output[{i}] has a shape of torch.Size({list(out._data.shape)})."
                )
        made.append(grad)
    return made


def backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False, inputs=None):
    """torch.autograd.backward: add the gradients of `tensors` into the .grad of the leaves
    (or of `inputs` only, when given)."""
    outputs = _as_tuple(tensors)
    grads = make_grads(outputs, _as_tuple(grad_tensors))
    if retain_graph is None:
        retain_graph = create_graph
    if inputs is None:
        run_backward(outputs, grads, None, retain_graph, create_graph)
        return
    inputs = _as_tuple(inputs)
    for tensor in inputs:
        if not requires_grad(tensor):
            raise RuntimeError("One of the differentiated Tensors does not require grad")
    targets = [edge(tensor) for tensor in inputs]
    captured = run_backward(outputs, grads, targets, retain_graph, create_graph)
    for tensor, node in zip(inputs, targets):  # leaves were filled in on the way
        if not isinstance(node, AccumulateGrad):
            if not tensor._retains_grad:  # as in torch, a non-leaf input now keeps its .grad
                tensor._retains_grad = True
                _retain_on(node, tensor)
            if node in captured:
                _store_grad(tensor, captured[node], create_graph)


def grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, allow_unused=None,
         materialize_grads=False):
    """torch.autograd.grad: the gradients of `outputs` with respect to `inputs`, as a tuple,
    without touching any .grad."""
    outputs, inputs = _as_tuple(outputs), _as_tuple(inputs)
    grads = make_grads(outputs, _as_tuple(grad_outputs))
    if retain_graph is None:
        retain_graph = create_graph
    for tensor in inputs:
        if not requires_grad(tensor):
            raise RuntimeError("One of the differentiated Tensors does not require grad")
    targets = [edge(tensor) for tensor in inputs]
    captured = run_backward(outputs, grads, targets, retain_graph, create_graph, accumulate=False)
    found = []
    for i, (tensor, node) in enumerate(zip(inputs, targets)):
        g = captured.get(node)
        if g is None:
            if materialize_grads:
                g = Tensor(np.zeros_like(tensor._data))
            elif not allow_unused:
                raise RuntimeError(
                    f"The differentiated Tensor at index {i} appears to not have been used in the graph. "
                    "Set allow_unused=True if this is the desired behavior."
                )
        found.append(g)
    return tuple(found)


# Tensor methods and properties (attached to Tensor by _methods).


def tensor_backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None):
    backward(self, gradient, retain_graph, create_graph, inputs)


def detach(self):
    """The same values with no history; shares memory and the version counter with `self`."""
    out = Tensor(self._data)
    out._vc = counter(self)
    return out


def requires_grad_(self, requires_grad=True):
    set_requires_grad(self, requires_grad)
    return self


def retain_grad(self):
    """Keep .grad for a non-leaf tensor too."""
    if not requires_grad(self):
        raise RuntimeError("can't retain_grad on Tensor that has requires_grad=False")
    fn = grad_fn(self)
    if fn is not None and not self._retains_grad:
        self._retains_grad = True
        _retain_on(fn, self)


def _float_only(tensor, message):
    if tensor._data.dtype.kind != "f":
        raise RuntimeError(message)


def set_requires_grad(self, value):
    if grad_fn(self) is not None:
        hint = "" if value else (
            " If you want to use a computed variable in a subgraph that doesn't require differentiation use "
            "var_no_grad = var.detach()."
        )
        raise RuntimeError("you can only change requires_grad flags of leaf variables." + hint)
    if value:
        _float_only(self, "only Tensors of floating point dtype can require gradients")
    self._requires_grad = bool(value)


def get_grad(self):
    if self._grad is None and grad_fn(self) is not None and not self._retains_grad:
        import warnings

        warnings.warn(
            "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute "
            "won't be populated during autograd.backward(). If you indeed want the .grad field to be populated "
            "for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf "
            "Tensor by mistake, make sure you access the leaf Tensor instead.",
            UserWarning,
            stacklevel=2,
        )
    return self._grad


def set_grad(self, value):
    if value is not None:
        if not isinstance(value, Tensor):
            raise TypeError(f"assigned grad expected to be a Tensor or None but got grad of type {type(value).__name__}")
        if value._data.shape != self._data.shape:
            raise RuntimeError(
                f"attempting to assign a gradient of size '{list(value._data.shape)}' to a tensor of size "
                f"'{list(self._data.shape)}'. Please ensure that the gradient and the tensor are the same size"
            )
        if value._data.dtype != self._data.dtype:
            raise RuntimeError(
                f"attempting to assign a gradient with dtype '{_dtype.of(value._data.dtype)}' to a tensor with "
                f"dtype '{_dtype.of(self._data.dtype)}'."
            )
    self._grad = value


# Fields that belong to one tensor's place in a graph, which copies and pickles do not keep.
GRAPH_FIELDS = ("_grad_fn", "_vc", "_base", "_view_version", "_accumulator", "_retains_grad")


def getstate(self):
    """What a pickle keeps: the values and requires_grad; never .grad or the history, as in torch."""
    state = {k: v for k, v in self.__dict__.items() if k not in GRAPH_FIELDS and k != "_grad"}
    if requires_grad(self):
        state["_requires_grad"] = True
    return state


def deepcopy(self, memo):
    if grad_fn(self) is not None:
        raise RuntimeError(
            "Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment."
        )
    import copy

    out = object.__new__(type(self))
    memo[id(self)] = out
    for key, value in self.__dict__.items():
        if key == "_data":
            out._data = value.copy()
        elif key not in GRAPH_FIELDS:
            setattr(out, key, copy.deepcopy(value, memo))
    return out
