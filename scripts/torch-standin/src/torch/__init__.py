"""A NumPy-backed stand-in for PyTorch that runs in the browser (Pyodide).

Every import of this package's own modules is relative (from . import x), so the tests can
load it as standin_torch next to the real torch.
"""

from . import _methods, autograd, cuda, distributions, linalg, return_types
from ._autograd import enable_grad, is_grad_enabled, no_grad, set_grad_enabled
from ._compare import allclose, eq, equal, ge, gt, isclose, isfinite, isnan, le, lt, ne
from ._repr import set_printoptions
from ._creation import (
    arange,
    as_tensor,
    empty,
    empty_like,
    eye,
    from_numpy,
    full,
    full_like,
    is_tensor,
    linspace,
    ones,
    ones_like,
    tensor,
    zeros,
    zeros_like,
)
from ._dtype import (
    bool,
    double,
    dtype,
    float,
    float32,
    float64,
    int,
    int8,
    int16,
    int32,
    int64,
    long,
    promote_types,
    short,
    uint8,
)
from ._ops import (
    abs,
    add,
    clamp,
    clip,
    clone,
    cos,
    div,
    exp,
    log,
    maximum,
    minimum,
    mul,
    neg,
    pow,
    relu,
    round,
    sigmoid,
    sin,
    sqrt,
    sub,
    tanh,
    where,
)
from ._index import gather, scatter, scatter_add
from ._linalg import bmm, cdist, einsum, matmul, meshgrid, mm, outer
from ._missing import module_getattr
from ._random import (
    Generator,
    default_generator,
    manual_seed,
    rand,
    rand_like,
    randint,
    randn,
    randn_like,
    randperm,
)
from ._reduce import (
    all,
    amax,
    amin,
    any,
    argmax,
    argmin,
    cumprod,
    cumsum,
    log_softmax,
    logsumexp,
    max,
    mean,
    min,
    norm,
    softmax,
    std,
    sum,
    trace,
    var,
)
from ._serialization import load, save
from ._shape import (
    cat,
    chunk,
    diag,
    diff,
    expand,
    flatten,
    flip,
    permute,
    ravel,
    reshape,
    roll,
    split,
    squeeze,
    stack,
    transpose,
    tril,
    triu,
    unbind,
    unsqueeze,
)
from ._sort import argsort, bincount, median, mode, quantile, sort, topk, unique
from ._tensor import Size, Tensor, device
from . import nn  # after the rest: the layers are written with everything above
from . import optim, utils

# The release of PyTorch this package is tested against.
__version__ = "2.14.1+standin"


def get_num_threads():
    """Python in the browser runs on one thread."""
    return 1


def set_num_threads(n):
    """Accepted and ignored: there is only one thread."""


__getattr__ = module_getattr(__name__, "torch")
