# Partly ported from torchvision (torchvision/datasets/mnist.py), BSD-3-Clause.
# See THIRD_PARTY_NOTICES.md.
"""MNIST, read from the npz subset that ships with the page. PIL is not used.

The browser cannot download MNIST, so `make_mnist.py` writes the first 10,000 train and 2,000
test images of real torchvision's MNIST to public/pyodide/datasets/mnist.npz, and that file is
what an item is read from here. The notebook's own `root` and `download=True` are accepted and
ignored, so the same notebook runs in the page and in Colab.
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch._missing import module_getattr

# Where the notebook editor's plan puts the file inside the browser.
BROWSER_FILE = Path("/home/pyodide/.standin/mnist.npz")
# Where it sits in this repository, and in a deployed copy of the stand-in.
DATA_DIRECTORY = Path("public") / "pyodide" / "datasets"
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"

__getattr__ = module_getattr(__name__, "torchvision.datasets")


class Image:
    """The image an item hands to a transform: PIL's small interface over a uint8 array.

    PIL is not in the browser, so this stands in for it. `size` is PIL's (width, height),
    `mode` names the colour space, and np.asarray gives the pixels as they are stored: (H, W)
    for one channel, (H, W, C) for more.
    """

    MODES = {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}

    def __init__(self, array):
        if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] not in self.MODES):
            raise ValueError(
                "an image is (H, W), (H, W, 2), (H, W, 3) or (H, W, 4), not "
                f"{tuple(array.shape)}"
            )
        self._array = array

    @property
    def array(self):
        """The pixels, as the uint8 array they are stored in."""
        return self._array

    @property
    def size(self):
        """(width, height), as PIL gives it."""
        return (self._array.shape[1], self._array.shape[0])

    @property
    def mode(self):
        return self.MODES[self.channels]

    @property
    def channels(self):
        return 1 if self._array.ndim == 2 else self._array.shape[2]

    def __array__(self, dtype=None, copy=None):
        """The pixels as an array.

        A copy, as PIL hands one, unless the caller asks for no copy with copy=False: a notebook
        that writes to np.array(image) must not be writing into the dataset.
        """
        array = np.asarray(self._array, dtype=dtype)
        return array if copy is False else array.copy()

    def __repr__(self):
        width, height = self.size
        return f"<Image {self.mode} image of size {width}x{height}>"


def data_paths():
    """Where the npz may be: the browser's copy, then $STANDIN_DATA_DIR, then the repository."""
    candidates = [BROWSER_FILE]
    folder = os.environ.get("STANDIN_DATA_DIR")
    if folder:
        candidates.append(Path(folder) / "mnist.npz")
    # In this repository the file is public/pyodide/datasets/ under the repository root; in a
    # deployed copy of the stand-in it is under whatever folder the page runs from.
    for start in (Path(__file__).resolve(), Path.cwd()):
        for parent in start.parents:
            candidates.append(parent / DATA_DIRECTORY / "mnist.npz")
    return candidates


def find_data_file():
    """The first of `data_paths` that is there, or an error saying where it looked."""
    candidates = data_paths()
    for path in candidates:
        if path.is_file():
            return path
    searched = ", ".join(str(path) for path in candidates[:5])
    raise FileNotFoundError(
        f"there is no MNIST subset to read; looked in {searched}. Run make_mnist.py to build "
        "it, or set STANDIN_DATA_DIR to the folder holding mnist.npz."
    )


class MNIST:
    """The MNIST subset, with the constructor, the attributes and the items torchvision has.

    `root` and `download` are accepted and ignored: the images come from the npz file, not
    from a folder a notebook can write to.
    """

    classes = [str(digit) for digit in range(10)]

    def __init__(
        self, root=".", train=True, download=False, transform=None, target_transform=None
    ):
        self.root = root
        self.download = download
        self.train = bool(train)
        self.transform = transform
        self.target_transform = target_transform
        split = TRAIN_SPLIT if self.train else TEST_SPLIT
        with np.load(find_data_file()) as data:
            self._pixels = np.ascontiguousarray(data[f"x_{split}"])
            self._labels = np.ascontiguousarray(data[f"y_{split}"])
        self.data = torch.from_numpy(self._pixels)
        self.targets = torch.from_numpy(self._labels)

    def __len__(self):
        return self._pixels.shape[0]

    def __getitem__(self, index):
        """(image, target), where the target is a Python int as torchvision returns."""
        image = Image(self._pixels[index])
        target = int(self._labels[index])
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __repr__(self):
        split = TRAIN_SPLIT if self.train else TEST_SPLIT
        return f"MNIST(split={split}, n={len(self)})"
