# Ported from PyTorch (torch/utils/data/dataloader.py, sampler.py, dataset.py, _utils/collate.py), BSD-3-Clause.
# See THIRD_PARTY_NOTICES.md.
"""torch.utils.data: datasets, samplers, collation and DataLoader.

Everything runs in the main process (there is only one in the browser), so num_workers,
pin_memory and the other multiprocessing options are accepted and ignored. A shuffled
loader draws a new order from its generator at the start of each epoch, or from the global
generator (torch.manual_seed) when it has none; the order is not torch's, but it repeats
from the same seed.
"""

import collections.abc
import copy
import itertools
import math
import re
import warnings

import numpy as np

from .._creation import as_tensor, tensor
from .._dtype import float64
from .._missing import module_getattr
from .._random import default_generator, randint, randperm
from .._shape import stack
from .._tensor import Tensor

# ---------------------------------------------------------------- datasets


class Dataset:
    """Subclass it with __getitem__ (and __len__) to give a DataLoader something to batch."""

    def __getitem__(self, index):
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")


class TensorDataset(Dataset):
    """Rows of tensors that share their first dimension: dataset[i] is the tuple of row i."""

    def __init__(self, *tensors):
        if any(tensors[0].size(0) != each.size(0) for each in tensors):
            raise AssertionError("Size mismatch between tensors")
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(each[index] for each in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)


class Subset(Dataset):
    """The rows of `dataset` at `indices`."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        if isinstance(index, list):
            return self.dataset[[self.indices[i] for i in index]]
        return self.dataset[self.indices[index]]

    def __getitems__(self, indices):
        # Batched reads reach the parent dataset's own __getitems__ when it has one.
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([self.indices[i] for i in indices])
        return [self.dataset[self.indices[i]] for i in indices]

    def __len__(self):
        return len(self.indices)


def random_split(dataset, lengths, generator=default_generator):
    """Split into non-overlapping random Subsets of the given lengths, or of the given
    fractions (which sum to 1) with the remainder dealt out one by one from the first."""
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        counts = []
        for i, fraction in enumerate(lengths):
            if fraction < 0 or fraction > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            counts.append(math.floor(len(dataset) * fraction))
        for i in range(len(dataset) - sum(counts)):
            counts[i % len(counts)] += 1
        lengths = counts
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. This might result in an empty dataset.", stacklevel=2)
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    indices = randperm(sum(lengths), generator=generator).tolist()
    return [
        Subset(dataset, indices[end - length : end]) for end, length in zip(itertools.accumulate(lengths), lengths)
    ]


# ---------------------------------------------------------------- samplers


class Sampler:
    """Gives a DataLoader the order of the rows: iterating it yields indices."""

    def __iter__(self):
        raise NotImplementedError


class SequentialSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class RandomSampler(Sampler):
    """A new random order each time it is iterated, drawn from `generator` or, without one, the
    global generator. With replacement, or with num_samples, it may repeat rows."""

    def __init__(self, data_source, replacement=False, num_samples=None, generator=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        if not isinstance(replacement, bool):
            raise TypeError(f"replacement should be a boolean value, but got replacement={replacement}")
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={self.num_samples}")

    @property
    def num_samples(self):
        return len(self.data_source) if self._num_samples is None else self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(randint(0, n, (self.num_samples,), generator=self.generator).tolist())
        passes, rest = divmod(self.num_samples, n)
        indices = []
        for _ in range(passes):
            indices += randperm(n, generator=self.generator).tolist()
        if rest:
            indices += randperm(n, generator=self.generator).tolist()[:rest]
        return iter(indices)

    def __len__(self):
        return self.num_samples


class BatchSampler(Sampler):
    """Groups a sampler's indices into lists of batch_size, the last one short unless drop_last."""

    def __init__(self, sampler, batch_size, drop_last):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last should be a boolean value, but got drop_last={drop_last}")
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        indices = iter(self.sampler)
        while True:
            batch = list(itertools.islice(indices, self.batch_size))
            if not batch or (self.drop_last and len(batch) < self.batch_size):
                return
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size


# ---------------------------------------------------------------- collation

_NOT_BATCHABLE = "default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {}"
_TEXT_ARRAY = re.compile(r"[SaUO]")  # NumPy dtype codes for strings and objects


def default_convert(data):
    """Turn NumPy arrays and scalars inside `data` into tensors, leaving the rest as it is. What a
    DataLoader does to each sample when batch_size=None."""
    kind = type(data)
    if isinstance(data, Tensor):
        return data
    if kind.__module__ == "numpy" and kind.__name__ not in ("str_", "string_"):
        if kind.__name__ == "ndarray" and _TEXT_ARRAY.search(data.dtype.str):
            return data
        return as_tensor(data)
    if isinstance(data, collections.abc.Mapping):
        try:
            if isinstance(data, collections.abc.MutableMapping):
                clone = copy.copy(data)
                clone.update({key: default_convert(data[key]) for key in data})
                return clone
            return kind({key: default_convert(data[key]) for key in data})
        except TypeError:
            return {key: default_convert(data[key]) for key in data}
    if isinstance(data, tuple) and hasattr(data, "_fields"):  # a namedtuple
        return kind(*(default_convert(item) for item in data))
    if isinstance(data, tuple):
        return [default_convert(item) for item in data]
    if isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes)):
        try:
            if isinstance(data, collections.abc.MutableSequence):
                clone = copy.copy(data)
                for i, item in enumerate(data):
                    clone[i] = default_convert(item)
                return clone
            return kind([default_convert(item) for item in data])
        except TypeError:
            return [default_convert(item) for item in data]
    return data


def default_collate(batch):
    """Stack a list of samples into one batch: tensors and arrays along a new first dimension,
    ints as int64 and floats as float64 tensors, and dicts, lists and tuples of these field by field
    (a tuple of fields comes back as a list). Strings stay a list."""
    elem = batch[0]
    kind = type(elem)
    if isinstance(elem, Tensor):
        return stack(batch, 0)
    if isinstance(elem, np.ndarray):
        if _TEXT_ARRAY.search(elem.dtype.str):
            raise TypeError(_NOT_BATCHABLE.format(elem.dtype))
        return default_collate([as_tensor(item) for item in batch])
    if isinstance(elem, (np.bool_, np.number, np.object_)):
        return as_tensor(batch)
    if isinstance(elem, float):
        return tensor(batch, dtype=float64)
    if isinstance(elem, int):
        return tensor(batch)
    if isinstance(elem, (str, bytes)):
        return batch
    if isinstance(elem, collections.abc.Mapping):
        try:
            if isinstance(elem, collections.abc.MutableMapping):
                clone = copy.copy(elem)
                clone.update({key: default_collate([item[key] for item in batch]) for key in elem})
                return clone
            return kind({key: default_collate([item[key] for item in batch]) for key in elem})
        except TypeError:
            return {key: default_collate([item[key] for item in batch]) for key in elem}
    if isinstance(elem, tuple) and hasattr(elem, "_fields"):  # a namedtuple
        return kind(*(default_collate(fields) for fields in zip(*batch)))
    if isinstance(elem, collections.abc.Sequence):
        size = len(elem)
        if not all(len(item) == size for item in batch):
            raise RuntimeError("each element in list of batch should be of equal size")
        fields = list(zip(*batch))
        if isinstance(elem, tuple):
            return [default_collate(samples) for samples in fields]
        try:
            if isinstance(elem, collections.abc.MutableSequence):
                clone = copy.copy(elem)
                for i, samples in enumerate(fields):
                    clone[i] = default_collate(samples)
                return clone
            return kind([default_collate(samples) for samples in fields])
        except TypeError:
            return [default_collate(samples) for samples in fields]
    raise TypeError(_NOT_BATCHABLE.format(kind))


# ---------------------------------------------------------------- the loader


class DataLoader:
    """Batches a dataset: iterating it gives one collated batch per list of indices from the
    sampler. Every option about worker processes and pinned memory is accepted and ignored."""

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=None,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor=None,
        persistent_workers=False,
        pin_memory_device="",
        in_order=True,
    ):
        if num_workers < 0:
            raise ValueError("num_workers option should be non-negative; use num_workers=0 to disable multiprocessing.")
        if timeout < 0:
            raise ValueError("timeout option should be non-negative")
        if num_workers == 0 and prefetch_factor is not None:
            raise ValueError(
                "prefetch_factor option could only be specified in multiprocessing."
                "let num_workers > 0 to enable multiprocessing, otherwise set prefetch_factor to None."
            )
        if num_workers > 0 and prefetch_factor is None:
            prefetch_factor = 2
        elif prefetch_factor is not None and prefetch_factor < 0:
            raise ValueError("prefetch_factor option should be non-negative")
        if persistent_workers and num_workers == 0:
            raise ValueError("persistent_workers option needs num_workers > 0")
        self.dataset = dataset
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.pin_memory_device = pin_memory_device
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context
        self.persistent_workers = persistent_workers
        self.in_order = in_order
        shuffle = bool(shuffle)
        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with shuffle")
        if batch_sampler is not None:
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError(
                    "batch_sampler option is mutually exclusive with batch_size, shuffle, sampler, and drop_last"
                )
            batch_size = None
            drop_last = False
        elif batch_size is None and drop_last:
            raise ValueError("batch_size=None option disables auto-batching and is mutually exclusive with drop_last")
        if sampler is None:
            sampler = RandomSampler(dataset, generator=generator) if shuffle else SequentialSampler(dataset)
        if batch_size is not None and batch_sampler is None:
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.generator = generator
        if collate_fn is None:
            collate_fn = default_collate if batch_sampler is not None else default_convert
        self.collate_fn = collate_fn

    @property
    def _index_sampler(self):
        """What the loader walks through: batches of indices, or single indices without batching."""
        return self.batch_sampler if self.batch_sampler is not None else self.sampler

    def __iter__(self):
        return _Epoch(self)

    def __len__(self):
        return len(self._index_sampler)


class _Epoch:
    """One pass over a DataLoader; its order is drawn when it starts."""

    def __init__(self, loader):
        self._loader = loader
        self._indices = iter(loader._index_sampler)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._loader._index_sampler)

    def __next__(self):
        index = next(self._indices)
        loader = self._loader
        dataset = loader.dataset
        if loader.batch_sampler is None:
            return loader.collate_fn(dataset[index])
        getitems = getattr(dataset, "__getitems__", None)
        samples = getitems(index) if getitems else [dataset[i] for i in index]
        return loader.collate_fn(samples)


__getattr__ = module_getattr(__name__, "torch.utils.data")
