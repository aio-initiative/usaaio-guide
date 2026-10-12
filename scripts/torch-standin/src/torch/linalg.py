"""torch.linalg: the parts our notebooks use."""

import numpy as np

from ._missing import module_getattr
from ._ops import data_of
from ._tensor import Tensor


def matrix_rank(A, tol=None, hermitian=False, *, atol=None, rtol=None):
    """The number of singular values above max(atol, rtol * largest singular value).

    With neither tolerance given, rtol is the dtype's machine epsilon times the larger side
    of the matrix, as in torch. Batches of matrices give one rank each.
    """
    x = data_of(A)
    if x.dtype.kind != "f":
        raise RuntimeError("This function doesn't handle types other than float and double")
    if tol is not None:  # the old name for atol
        atol = tol
    if rtol is None:
        rtol = np.finfo(x.dtype).eps * max(x.shape[-2:]) if atol is None or atol <= 0 else 0.0
    singular = np.abs(np.linalg.eigvalsh(x)) if hermitian else np.linalg.svd(x, compute_uv=False)
    largest = singular.max(axis=-1, keepdims=True, initial=0.0)
    cutoff = np.maximum(atol or 0.0, rtol * largest)
    return Tensor(np.asarray(np.sum(singular > cutoff, axis=-1), dtype=np.int64))


__getattr__ = module_getattr(__name__, "torch.linalg")
