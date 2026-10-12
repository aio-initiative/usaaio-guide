"""Shape operations, joining and splitting against real torch."""

import numpy as np
import pytest

from parity import A, B, CUBE, INT, MASK, SQUARE, V, agree, cases, check, check_raises

ROW = np.arange(6, dtype=np.float32)
INDEX = np.array([[0, 2, 1, 3], [3, 3, 0, 1], [1, 0, 2, 2]])


@cases(
    ("reshape", lambda t, x: x.reshape(4, 3), A),
    ("reshape tuple", lambda t, x: x.reshape((2, 6)), A),
    ("reshape infer", lambda t, x: x.reshape(-1, 2), A),
    ("reshape scalar", lambda t, x: x[:1, :1].reshape(()), A),
    ("torch.reshape", lambda t, x: t.reshape(x, (6, 2)), A),
    ("view", lambda t, x: x.view(2, 6), A),
    ("view infer", lambda t, x: x.view(-1), A),
    ("view list", lambda t, x: x.view([12, 1]), A),
    ("flatten", lambda t, x: x.flatten(), CUBE),
    ("flatten from", lambda t, x: x.flatten(1), CUBE),
    ("flatten range", lambda t, x: x.flatten(start_dim=0, end_dim=1), CUBE),
    ("flatten scalar", lambda t, x: x[0, 0].flatten(), A),
    ("torch.flatten", lambda t, x: t.flatten(x, 1), CUBE),
    ("ravel", lambda t, x: x.ravel(), A),
    ("squeeze", lambda t, x: x[:1, :, None].squeeze(), A),
    ("squeeze dim", lambda t, x: x[:1, :, None].squeeze(0), A),
    ("squeeze negative dim", lambda t, x: x[:1, :, None].squeeze(-1), A),
    ("squeeze a full dim", lambda t, x: x.squeeze(1), A),
    ("unsqueeze", lambda t, x: x.unsqueeze(1), A),
    ("unsqueeze end", lambda t, x: x.unsqueeze(-1), A),
    ("unsqueeze keyword", lambda t, x: t.unsqueeze(x, dim=0), A),
    ("transpose", lambda t, x: x.transpose(0, 1), A),
    ("transpose negative", lambda t, x: x.transpose(-1, -2), CUBE),
    ("torch.transpose", lambda t, x: t.transpose(x, 0, 2), CUBE),
    ("permute", lambda t, x: x.permute(2, 0, 1), CUBE),
    ("permute tuple", lambda t, x: x.permute((1, 2, 0)), CUBE),
    ("expand", lambda t, x: x.expand(3, 4), V),
    ("expand keep", lambda t, x: x[:, :1].expand(-1, 5), A),
    ("expand new dims", lambda t, x: x.expand(2, 3, 4), A),
    ("expand tuple", lambda t, x: x.expand((2, 4)), V),
    ("expand bool", lambda t, x: x[:1].expand(3, 4), MASK),
    ("flip", lambda t, x: x.flip(0), A),
    ("flip two", lambda t, x: x.flip(0, 1), A),
    ("flip list", lambda t, x: x.flip([1]), A),
    ("flip dims keyword", lambda t, x: t.flip(x, dims=(0, 2)), CUBE),
    ("roll", lambda t, x: t.roll(x, 1), A),
    ("roll dim", lambda t, x: t.roll(x, 1, 1), A),
    ("roll negative", lambda t, x: t.roll(x, -2, dims=0), A),
    ("roll several", lambda t, x: t.roll(x, shifts=(1, 2), dims=(0, 1)), A),
    ("roll past the length", lambda t, x: x.roll(7, 1), A),
    ("triu", lambda t, x: t.triu(x), SQUARE),
    ("triu above", lambda t, x: t.triu(x, diagonal=1), SQUARE),
    ("triu wide", lambda t, x: t.triu(x, -1), A),
    ("triu int", lambda t, x: x.triu(), INT),
    ("triu bool", lambda t, x: t.triu(t.ones(3, 3, dtype=t.bool), 1), A),
    ("triu batch", lambda t, x: t.triu(x), CUBE),
    ("tril", lambda t, x: t.tril(x), SQUARE),
    ("tril below", lambda t, x: x.tril(-1), SQUARE),
    ("diag of a matrix", lambda t, x: x.diag(), A),
    ("diag above", lambda t, x: x.diag(1), A),
    ("diag below", lambda t, x: t.diag(x, -1), A),
    ("diag of a vector", lambda t, x: x.diag(), V),
    ("diag of a vector offset", lambda t, x: x.diag(diagonal=2), V),
    ("diff", lambda t, x: x.diff(), A),
    ("diff dim", lambda t, x: x.diff(dim=0), A),
    ("diff twice", lambda t, x: x.diff(n=2), A),
    ("diff int", lambda t, x: t.diff(x), INT),
    ("gather", lambda t, x, i: x.gather(1, i), A, INDEX),
    ("gather rows", lambda t, x, i: t.gather(x, 0, i[:, :2] % 3), A, INDEX),
    ("scatter", lambda t, x, i, y: x.scatter(1, i, y), A, INDEX, B),
    ("scatter value", lambda t, x, i: x.scatter(1, i[:, :2], 9.0), A, INDEX),
    ("scatter one hot", lambda t, i: t.zeros(3, 4).scatter(1, i[:, :1], 1.0), INDEX),
    ("scatter_add_", lambda t, x, i, y: x.scatter_add_(1, i, y), A, INDEX, B),
    ("scatter_add_ repeats", lambda t, i: t.zeros(3, 4).scatter_add_(1, i * 0, t.ones(3, 4)), INDEX),
    ("fill_diagonal_", lambda t, x: x.fill_diagonal_(7.0), SQUARE),
    ("fill_diagonal_ wide", lambda t, x: x.fill_diagonal_(0), A),
)
def test_shape_operations(st, real, case):
    check(st, real, *case[1:])


@cases(
    ("cat", lambda t, x, y: t.cat([x, y]), A, B),
    ("cat dim", lambda t, x, y: t.cat((x, y), dim=1), A, B),
    ("cat negative dim", lambda t, x, y: t.cat([x, y, x], -1), A, B),
    ("cat vectors", lambda t, x: t.cat([x, x[:2]]), V),
    ("cat promotes", lambda t, x, y: t.cat([x, y]), INT, A),
    ("cat bool and int", lambda t, x, y: t.cat([x, y]), MASK, INT),
    ("cat onto an empty tensor", lambda t, x: t.cat([t.tensor([]), x[0], x[1]]), A),
    ("stack", lambda t, x, y: t.stack([x, y]), A, B),
    ("stack dim", lambda t, x, y: t.stack((x, y), dim=1), A, B),
    ("stack last", lambda t, x, y: t.stack([x, y], dim=-1), A, B),
    ("stack scalars", lambda t, x: t.stack([x[0], x[1]]), V),
    ("stack rows", lambda t, x: t.stack(list(x)), A),
    ("chunk", lambda t, x: x.chunk(2), ROW),
    ("chunk uneven", lambda t, x: x.chunk(4), ROW),
    ("chunk dim", lambda t, x: x.chunk(3, dim=1), A),
    ("torch.chunk", lambda t, x: t.chunk(x, 2, 1), A),
    ("split size", lambda t, x: x.split(4), ROW),
    ("split sizes", lambda t, x: x.split([1, 2, 3]), ROW),
    ("split dim", lambda t, x: x.split(3, dim=1), A),
    ("torch.split", lambda t, x: t.split(x, [1, 3], dim=1), A),
    ("unbind", lambda t, x: x.unbind(), A),
    ("unbind dim", lambda t, x: x.unbind(1), A),
    ("unbind keyword", lambda t, x: t.unbind(x, dim=-1), CUBE),
)
def test_joining_and_splitting(st, real, case):
    check(st, real, *case[1:])


@cases(
    ("reshape wrong size", lambda t, x: x.reshape(5, 5), A),
    ("cat mismatched", lambda t, x, v: t.cat([x, v[None, :3]]), A, V),
    ("stack mismatched", lambda t, x, v: t.stack([x, v]), A, V),
    ("expand shrinks", lambda t, x: x.expand(3, 2), A),
    ("transpose out of range", lambda t, x: x.transpose(0, 2), A),
    ("split sizes do not add up", lambda t, x: x.split([1, 2]), ROW),
)
def test_shape_errors(st, real, case):
    check_raises(st, real, *case[1:])


def test_views_share_memory(st):
    x = st.arange(6.0)
    for view in (x.view(2, 3), x.reshape(3, 2), x.unsqueeze(0), x.view(2, 3).transpose(0, 1), x.view(2, 3).T):
        view.zero_()
        assert x.sum().item() == 0.0
        x.copy_(st.arange(6.0))


GATHER_INDEX = np.zeros((3, 4), dtype=np.int64)

# Every operation that takes a dim, with a dim one past either end of the valid range.
# unsqueeze and stack accept one more dimension than the input has.
DIM_OPERATIONS = [
    ("flatten", lambda t, x, d: x.flatten(d)),
    ("flatten end", lambda t, x, d: x.flatten(0, d)),
    ("squeeze", lambda t, x, d: x[:, :1].squeeze(d)),
    ("unsqueeze", lambda t, x, d: x.unsqueeze(d + (1 if d >= 0 else -1))),
    ("transpose", lambda t, x, d: x.transpose(0, d)),
    ("permute", lambda t, x, d: x.permute(d, 0)),
    ("split", lambda t, x, d: x.split(1, dim=d)),
    ("chunk", lambda t, x, d: x.chunk(2, d)),
    ("unbind", lambda t, x, d: x.unbind(d)),
    ("diff", lambda t, x, d: x.diff(dim=d)),
    ("flip", lambda t, x, d: x.flip(d)),
    ("roll", lambda t, x, d: x.roll(1, d)),
    ("cat", lambda t, x, d: t.cat([x, x], d)),
    ("stack", lambda t, x, d: t.stack([x, x], d + (1 if d >= 0 else -1))),
    ("gather", lambda t, x, d: x.gather(d, t.tensor(GATHER_INDEX))),
    ("scatter", lambda t, x, d: x.scatter(d, t.tensor(GATHER_INDEX), 1.0)),
    ("scatter_add_", lambda t, x, d: x.scatter_add_(d, t.tensor(GATHER_INDEX), x * 2)),
    ("size", lambda t, x, d: x.size(d)),
    ("sum", lambda t, x, d: x.sum(d)),
    ("softmax", lambda t, x, d: x.softmax(d)),
    ("sort", lambda t, x, d: x.sort(d)),
]


@pytest.mark.parametrize("dim", [2, -3, -1, -2], ids=["one past the end", "one before the start", "-1", "-2"])
@pytest.mark.parametrize("operation", DIM_OPERATIONS, ids=[name for name, _ in DIM_OPERATIONS])
def test_dims_are_range_checked(st, real, operation, dim):
    # The two out-of-range dims raise torch's IndexError; the negative ones in range work.
    agree(lambda t: operation[1](t, t.tensor(A), dim), st, real)


@cases(
    ("flatten backwards", lambda t, x: x.flatten(1, 0), CUBE),
    ("permute too few dims", lambda t, x: x.permute(1, 0), CUBE),
    ("permute a repeated dim", lambda t, x: x.permute(0, 0, 1), CUBE),
    ("gather with fewer index dims", lambda t, x: x.gather(1, t.tensor([0, 1])), A),
    ("view across a transpose", lambda t, x: x.T.view(-1), A),
    ("view of an expand", lambda t, x: x.expand(3, 4).view(-1), V),
    ("view of a column slice", lambda t, x: x[:, :2].view(-1), A),
)
def test_more_shape_errors(st, real, case):
    check_raises(st, real, *case[1:])


@cases(
    ("flatten of a 0-d tensor", lambda t, x: x[0, 0].flatten(0), A),
    ("squeeze a tuple of dims", lambda t, x: x[:, None, :1].squeeze((1, 2)), A),
    ("transpose of a 0-d tensor", lambda t, x: x[0, 0].transpose(0, -1), A),
    ("unsqueeze before the first", lambda t, x: x.unsqueeze(-3), A),
    ("view that keeps the layout", lambda t, x: x.T.view(4, 3), A),
    ("view of a row slice", lambda t, x: x[:2].view(-1), A),
    ("reshape across a transpose copies", lambda t, x: x.T.reshape(-1), A),
    ("contiguous", lambda t, x: (x.T.is_contiguous(), x.is_contiguous(), x.T.contiguous().is_contiguous()), A),
    ("contiguous then view", lambda t, x: x.T.contiguous().view(-1), A),
)
def test_views_and_layout(st, real, case):
    check(st, real, *case[1:])


def test_contiguous_returns_the_same_tensor_or_a_copy(st):
    x = st.tensor(A)
    assert x.contiguous() is x
    copy = x.T.contiguous()
    copy.zero_()
    assert x.sum().item() != 0.0


@pytest.mark.parametrize(
    "build",
    [
        lambda t: [t.tensor([1.0, 2.0]), t.tensor([1.0])],
        lambda t: [t.zeros(2, 3), t.zeros(2, 3), t.zeros(3, 2)],
        lambda t: [t.zeros(2, 3), t.zeros(3)],
        lambda t: [t.tensor(1.0), t.tensor([1.0])],
        lambda t: [t.zeros(2), t.zeros(2), t.zeros(2), t.zeros(1)],
        lambda t: [],
    ],
    ids=["vectors", "third entry", "different ranks", "scalar and vector", "last entry", "empty list"],
)
def test_stack_names_the_entries_that_differ_as_torch_does(st, real, build):
    with pytest.raises(RuntimeError) as want:
        real.stack(build(real))
    with pytest.raises(RuntimeError) as got:
        st.stack(build(st))
    assert str(got.value) == str(want.value)


def test_stack_checks_the_dim_before_the_shapes(st, real):
    # An out-of-range dim is an IndexError even when the shapes also differ, as in torch.
    for dim in (3, -4):
        with pytest.raises(IndexError) as want:
            real.stack([real.zeros(2), real.zeros(3)], dim=dim)
        with pytest.raises(IndexError) as got:
            st.stack([st.zeros(2), st.zeros(3)], dim=dim)
        assert str(got.value) == str(want.value)
