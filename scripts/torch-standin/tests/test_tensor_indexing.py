"""Indexing and assignment against real torch."""

import numpy as np

from parity import A, B, CUBE, INT, V, cases, check, check_raises

ROWS = np.array([2, 0, 2])
COLS = np.array([1, 3, 0])


@cases(
    ("int", lambda t, x: x[1], A),
    ("negative int", lambda t, x: x[-1], A),
    ("two ints", lambda t, x: x[1, 2], A),
    ("slice", lambda t, x: x[1:], A),
    ("slice step", lambda t, x: x[:, ::2], A),
    ("slice both", lambda t, x: x[:2, 1:3], A),
    ("ellipsis", lambda t, x: x[..., 0], CUBE),
    ("ellipsis first", lambda t, x: x[0, ...], CUBE),
    ("none", lambda t, x: x[None], A),
    ("none inside", lambda t, x: x[:, None, :], A),
    ("none at the end", lambda t, x: x[..., None], A),
    ("list", lambda t, x: x[[0, 2]], A),
    ("list of bools", lambda t, x: x[[True, False, True]], A),
    ("index tensor", lambda t, x, i: x[i], A, ROWS),
    ("index tensor columns", lambda t, x, i: x[:, i], A, COLS),
    ("two index tensors", lambda t, x, i, j: x[i, j], A, ROWS, COLS),
    ("arange and labels", lambda t, x, j: x[t.arange(3), j], A, COLS),
    ("index tensor 2-d", lambda t, x, i: x[i.reshape(3, 1)], A, ROWS),
    ("scalar index tensor", lambda t, x, i: x[i[0]], A, ROWS),
    ("numpy index", lambda t, x: x[np.array([0, 2])], A),
    ("mask", lambda t, x: x[x > 0], A),
    ("row mask", lambda t, x: x[x[:, 0] > 0], A),
    ("mask and slice", lambda t, x: x[x[:, 0] > 0, 1:], A),
    ("mask of a vector", lambda t, x: x[x < 0], V),
    ("empty selection", lambda t, x: x[x > 100], A),
    ("int and index tensor", lambda t, x, i: x[0, i], A, COLS),
    ("chained", lambda t, x: x[1][2], A),
    ("int, slice, index tensor", lambda t, x, i: x[0, :, i], CUBE, COLS),
    ("index tensor, slice, int", lambda t, x, i: x[i[:2] % 2, :, 1], CUBE, COLS),
    ("numpy int", lambda t, x: x[np.int64(1)], A),
    ("int tensor data", lambda t, x: x[1:, 0], INT),
)
def test_getitem(st, real, case):
    check(st, real, *case[1:])


def _set(index_of, value_of):
    """A case function that assigns into a copy and returns it."""

    def run(t, x, *rest):
        x = x.clone()
        x[index_of(t, x, *rest)] = value_of(t, x, *rest)
        return x

    return run


@cases(
    ("number at an int", _set(lambda t, x: 1, lambda t, x: 5.0), A),
    ("number at a cell", _set(lambda t, x: (1, 2), lambda t, x: 5.0), A),
    ("number in a slice", _set(lambda t, x: (slice(None), 0), lambda t, x: 0), A),
    ("row tensor", _set(lambda t, x, y: 0, lambda t, x, y: y[1]), A, B),
    ("broadcast a row", _set(lambda t, x, v: slice(1, None), lambda t, x, v: v), A, V),
    ("mask with a number", _set(lambda t, x: x < 0, lambda t, x: 0.0), A),
    ("mask with values", _set(lambda t, x, y: x < 0, lambda t, x, y: y[x < 0]), A, B),
    ("index tensors", _set(lambda t, x, i, j: (i[:2], j[:2]), lambda t, x, i, j: -1.0), A, ROWS, COLS),
    ("arange and labels", _set(lambda t, x, j: (t.arange(3), j), lambda t, x, j: 1.0), A, COLS),
    ("ellipsis", _set(lambda t, x: (Ellipsis, 0), lambda t, x: 2.0), CUBE),
    ("int, slice, index tensor", _set(lambda t, x, i: (0, slice(None), i), lambda t, x, i: 3.0), CUBE, COLS),
    ("float into ints truncates", _set(lambda t, x: 0, lambda t, x: 2.7), INT),
    ("list value", _set(lambda t, x: 0, lambda t, x: t.tensor([1.0, 2.0, 3.0, 4.0])), A),
    ("0-d tensor value", _set(lambda t, x: (0, 0), lambda t, x: x[1, 1]), A),
    ("float64 into float32", _set(lambda t, x: 0, lambda t, x: x[1].double()), A),
)
def test_setitem(st, real, case):
    check(st, real, *case[1:])


def _augmented(t, x):
    x = x.clone()
    x[:, 0] += 1.5
    x[1] *= 2
    x[x < 0] -= 1
    return x


@cases(
    ("augmented assignment through an index", _augmented, A),
    ("assignment reaches the base through a view", lambda t, x: (x, x[0].fill_(0.0), x[:, 1].add_(1))[0], A),
)
def test_in_place_through_indexing(st, real, case):
    check(st, real, *case[1:])


def test_basic_indexing_is_a_view_and_fancy_indexing_a_copy(st):
    x = st.zeros(3, 4)
    x[1].fill_(1.0)
    x[0, 0].fill_(5.0)
    x[..., 3].fill_(2.0)
    assert x.tolist() == [[5.0, 0.0, 0.0, 2.0], [1.0, 1.0, 1.0, 2.0], [0.0, 0.0, 0.0, 2.0]]
    x[[0, 1]].fill_(9.0)
    x[x > 1].zero_()
    assert x[0, 0].item() == 5.0


@cases(
    ("index out of range", lambda t, x: x[3], A),
    ("too many indices", lambda t, x: x[0, 0, 0], A),
    ("float index", lambda t, x: x[1.0], A),
    ("mask of the wrong shape", lambda t, x: x[x[:2] > 0], A),
    ("value does not fit", lambda t, x, v: x.__setitem__(0, v[:3]), A, V),
)
def test_indexing_errors(st, real, case):
    check_raises(st, real, *case[1:])
