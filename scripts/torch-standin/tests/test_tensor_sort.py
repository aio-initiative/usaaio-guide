"""Sorting and counting against real torch: sort, argsort, topk, median, mode, quantile,
unique and bincount."""

import numpy as np

from parity import A, CUBE, INT, MASK, TIES, V, cases, check, check_raises

NANS = np.array([[1.0, np.nan, 0.0], [2.0, -1.0, 5.0]], dtype=np.float32)
LABELS = np.array([1, 3, 1, 0, 4, 1])


@cases(
    ("sort", lambda t, x: x.sort(), A),
    ("sort ties keep their order", lambda t, x: tuple(x.sort(dim=1)), TIES),
    ("sort descending ties", lambda t, x: tuple(t.sort(x, 1, descending=True)), TIES),
    ("sort dim 0", lambda t, x: tuple(x.sort(0)), CUBE),
    ("sort ints", lambda t, x: tuple(x.sort(descending=True)), INT),
    ("sort bools", lambda t, x: tuple(x.sort()), MASK),
    ("sort nan last", lambda t, x: tuple(x.sort()), NANS),
    ("sort nan first descending", lambda t, x: tuple(x.sort(descending=True)), NANS),
    ("argsort", lambda t, x: x.argsort(), TIES),
    ("argsort descending", lambda t, x: t.argsort(x, dim=1, descending=True), TIES),
    ("argsort dim 0", lambda t, x: x.argsort(0), A),
    ("topk", lambda t, x: x.topk(2).values, A),
    ("topk indices", lambda t, x: tuple(t.topk(x, 2, dim=0)), CUBE),
    ("topk smallest", lambda t, x: tuple(x.topk(2, largest=False)), A),
    ("topk of a vector", lambda t, x: tuple(x.topk(1)), V),
    ("median", lambda t, x: x.median(), A),
    ("median of an even count is the lower", lambda t, x: x.median(), TIES),
    ("median dim", lambda t, x: tuple(x.median(1)), TIES),
    ("median keepdim", lambda t, x: tuple(t.median(x, 0, keepdim=True)), CUBE),
    ("median ints", lambda t, x: (x.median(), tuple(x.median(dim=0))), INT),
    ("median with nan", lambda t, x: (x.median(), tuple(x.median(1))), NANS),
    ("mode", lambda t, x: tuple(x.mode()), TIES),
    ("mode dim 0", lambda t, x: tuple(t.mode(x, 0)), TIES),
    ("mode ints", lambda t, x: tuple(x.mode(1, keepdim=True)), INT),
    ("quantile", lambda t, x: x.quantile(0.3), A),
    ("quantile dim", lambda t, x: t.quantile(x, 0.75, dim=1), A),
    ("quantile keepdim", lambda t, x: x.quantile(0.5, dim=0, keepdim=True), A),
    ("quantile tensor q", lambda t, x: x.quantile(t.tensor([0.1, 0.5, 0.9]), dim=1), A),
    ("quantile methods", lambda t, x: [x.quantile(0.3, interpolation=m) for m in ("lower", "higher", "midpoint")], A),
    ("quantile float64", lambda t, x: x.double().quantile(0.4), A),
    ("quantile keepdim over all", lambda t, x: x.quantile(t.tensor([0.3, 0.6]), keepdim=True), CUBE),
    ("quantile keepdim along a dim", lambda t, x: x.quantile(t.tensor([0.3, 0.6]), dim=1, keepdim=True), CUBE),
    ("quantile nearest", lambda t, x: x.quantile(t.tensor([0.1, 0.5, 0.9]), dim=1, interpolation="nearest"), CUBE),
    ("quantile with nan", lambda t, x: x.quantile(t.tensor([0.1, 0.5]), dim=1), NANS),
    ("0-d sort", lambda t, x: (tuple(x[0, 0].sort()), tuple(x[0, 0].median(0)), tuple(x[0, 0].mode())), A),
    ("unique", lambda t, x: t.unique(x), INT),
    ("unique method", lambda t, x: x.unique(), TIES),
    ("unique counts", lambda t, x: t.unique(x, return_counts=True), INT),
    ("unique inverse", lambda t, x: t.unique(x, return_inverse=True), INT),
    ("unique both", lambda t, x: x.unique(return_inverse=True, return_counts=True), INT),
    ("unique rows", lambda t, x: t.unique(t.cat([x, x[:1]]), dim=0), INT),
    ("unique bools", lambda t, x: t.unique(x), MASK),
    ("bincount", lambda t, x: t.bincount(x), LABELS),
    ("bincount minlength", lambda t, x: t.bincount(x, minlength=7), LABELS),
    ("bincount weights", lambda t, x, w: t.bincount(x, weights=w), LABELS, A[0, :3].repeat(2)),
    ("bincount float64 weights", lambda t, x: t.bincount(x, x.double() / 2), LABELS),
    ("bincount nothing", lambda t, x: t.bincount(x[:0]), LABELS),
    ("bincount int32", lambda t, x: t.bincount(x.int()), LABELS),
)
def test_ordering(st, real, case):
    check(st, real, *case[1:])


@cases(
    ("topk too many", lambda t, x: x.topk(5), A),
    ("quantile of nothing", lambda t, x: x[:0].quantile(0.5), A),
    ("quantile out of range", lambda t, x: x.quantile(1.5), A),
    ("quantile q of another dtype", lambda t, x: x.quantile(t.tensor([0.5], dtype=t.float64)), A),
    ("bincount negative", lambda t, x: t.bincount(x), INT),
    ("bincount 2-d", lambda t, x: t.bincount(x.abs()), INT),
    ("bincount of bools", lambda t, x: t.bincount(x), MASK),
    ("sort dim out of range", lambda t, x: x.sort(2), A),
)
def test_ordering_errors(st, real, case):
    check_raises(st, real, *case[1:])
