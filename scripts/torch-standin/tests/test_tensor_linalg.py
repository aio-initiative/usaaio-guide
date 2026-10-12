"""Matrix products, einsum, distances, rank and meshgrid against real torch."""

import numpy as np
import pytest

from parity import A, B, CUBE, INT, SQUARE, V, cases, check, check_raises

# Real torch warns when meshgrid is called without indexing=; that is not under test.
pytestmark = pytest.mark.filterwarnings("ignore:torch.meshgrid")

BATCH = np.random.default_rng(1).standard_normal((2, 4, 3)).astype(np.float32)
LOW_RANK = np.outer(np.arange(1, 5), np.arange(1, 4)).astype(np.float32)


@cases(
    ("matmul", lambda t, x, y: x @ y.T, A, B),
    ("torch.matmul", lambda t, x, y: t.matmul(x, y.T), A, B),
    ("method matmul", lambda t, x, y: x.matmul(y.T), A, B),
    ("matrix vector", lambda t, x, v: x @ v, A, V),
    ("vector matrix", lambda t, x, v: v @ x.T, A, V),
    ("vector vector", lambda t, v: v @ v, V),
    ("batched", lambda t, x, y: x @ y, CUBE, BATCH),
    ("batch times matrix", lambda t, x, y: x @ y, CUBE, SQUARE),
    ("ints", lambda t, x: x @ x.T, INT),
    ("float64", lambda t, x: x.double() @ x.double().T, A),
    ("mm", lambda t, x, y: t.mm(x, y.T), A, B),
    ("method mm", lambda t, x: x.mm(x.T), A),
    ("bmm", lambda t, x, y: t.bmm(x, y), CUBE, BATCH),
    ("method bmm", lambda t, x, y: x.bmm(y), CUBE, BATCH),
    ("outer", lambda t, v: t.outer(v, v[:3]), V),
    ("outer promotes", lambda t, v: t.outer(t.arange(3), v), V),
    ("method outer", lambda t, v: v.outer(v), V),
    ("einsum matmul", lambda t, x, y: t.einsum("ij,kj->ik", x, y), A, B),
    ("einsum list", lambda t, x, y: t.einsum("ij,kj->ik", [x, y]), A, B),
    ("einsum spaces", lambda t, x, y: t.einsum("i j , k j -> i k", x, y), A, B),
    ("einsum trace", lambda t, x: t.einsum("ii", x), SQUARE),
    ("einsum batch", lambda t, x, y: t.einsum("bij,bjk->bik", x, y), CUBE, BATCH),
    ("einsum three", lambda t, x, y, v: t.einsum("ij,kj,j->ik", x, y, v), A, B, V),
    ("einsum ellipsis", lambda t, x: t.einsum("...ij->...ji", x), CUBE),
    ("einsum ints", lambda t, x: t.einsum("ij,ij->", x, x), INT),
    ("cdist", lambda t, x, y: t.cdist(x, y), A, B),
    ("cdist self", lambda t, x: t.cdist(x, x), A),
    ("cdist p 1", lambda t, x, y: t.cdist(x, y[:2], p=1), A, B),
    ("cdist p inf", lambda t, x, y: t.cdist(x, y, p=float("inf")), A, B),
    ("cdist batched", lambda t, x, y: t.cdist(x, y), CUBE, CUBE),
    ("matrix_rank", lambda t, x: t.linalg.matrix_rank(x), LOW_RANK),
    ("matrix_rank full", lambda t, x: t.linalg.matrix_rank(x), SQUARE),
    ("matrix_rank batch", lambda t, x: t.linalg.matrix_rank(x), CUBE),
    ("matrix_rank atol", lambda t, x: t.linalg.matrix_rank(x, atol=1.0), SQUARE),
    ("meshgrid ij", lambda t, x, y: t.meshgrid(x[0], y[1, :3], indexing="ij"), A, B),
    ("meshgrid xy", lambda t, x, y: t.meshgrid(x[0], y[1, :3], indexing="xy"), A, B),
    ("meshgrid default", lambda t, x: t.meshgrid(x[0], x[1]), A),
    ("meshgrid list", lambda t, x: t.meshgrid([x[0], x[1, :2]], indexing="ij"), A),
    ("meshgrid three xy", lambda t, x: t.meshgrid(x[0], x[1, :2], x[2, :3], indexing="xy"), A),
    ("meshgrid ints", lambda t: t.meshgrid(t.arange(2), t.arange(3), indexing="xy")),
)
def test_linear_algebra(st, real, case):
    check(st, real, *case[1:])


def test_meshgrid_gives_views_of_its_inputs(st):
    rows, columns = st.arange(2.0), st.arange(3.0)
    x, y = st.meshgrid(rows, columns, indexing="ij")
    assert np.shares_memory(x.numpy(), rows.numpy()) and np.shares_memory(y.numpy(), columns.numpy())


@cases(
    ("matmul across dtypes", lambda t, x, y: x @ y.T, INT, A),
    ("matmul float32 and float64", lambda t, x, y: x @ y.double().T, A, B),
    ("matmul shapes", lambda t, x, y: x @ y, A, B),
    ("array @ tensor", lambda t, x: np.ones((2, 3), dtype=np.float32) @ x, A),
    ("matmul of a 0-d tensor", lambda t, x: x[0, 0] @ x[0], A),
    ("matmul of bools", lambda t, x: (x > 0) @ (x > 0).T, A),
    ("mm of a batch", lambda t, x: t.mm(x, x[0]), CUBE),
    ("bmm of matrices", lambda t, x: t.bmm(x, x.T), A),
    ("outer of a matrix", lambda t, x: t.outer(x, x[0]), A),
    ("einsum across dtypes", lambda t, x, y: t.einsum("ij,ij->", x, y), INT, A),
    ("cdist of ints", lambda t, x: t.cdist(x, x), INT),
    ("cdist of vectors", lambda t, v: t.cdist(v, v), V),
    ("cdist columns", lambda t, x, y: t.cdist(x, y[:, :3]), A, B),
    ("matrix_rank of ints", lambda t, x: t.linalg.matrix_rank(x), INT),
    ("meshgrid across dtypes", lambda t, x: t.meshgrid(t.arange(2), x[0], indexing="ij"), A),
)
def test_linear_algebra_errors(st, real, case):
    check_raises(st, real, *case[1:])
