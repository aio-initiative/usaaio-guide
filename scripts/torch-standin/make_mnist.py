"""Build the MNIST subset the browser loads: public/pyodide/datasets/mnist.npz.

The browser cannot download MNIST, so the first 10,000 train and 2,000 test images are kept
in one npz file that ships with the page. Real torchvision downloads MNIST into a temporary
folder here and takes the images in its own order, so the subset is the same every time and
matches what real torchvision hands a notebook.

Run it from scripts/torch-standin with the venv's Python (it needs real torchvision):

    .venv/bin/python make_mnist.py
"""

import tempfile
from pathlib import Path

import numpy as np

TRAIN_IMAGES = 10000
TEST_IMAGES = 2000
OUTPUT = Path(__file__).resolve().parents[2] / "public" / "pyodide" / "datasets" / "mnist.npz"


def build(output=OUTPUT):
    """Write the npz, and return the four arrays it holds."""
    from torchvision.datasets import MNIST

    with tempfile.TemporaryDirectory() as folder:
        train = MNIST(folder, train=True, download=True)
        test = MNIST(folder, train=False, download=True)
        x_train = train.data[:TRAIN_IMAGES].numpy()
        y_train = train.targets[:TRAIN_IMAGES].numpy()
        x_test = test.data[:TEST_IMAGES].numpy()
        y_test = test.targets[:TEST_IMAGES].numpy()

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    return x_train, y_train, x_test, y_test


def report(name, labels, count):
    """How many of each digit the split holds, so the file can be checked at a glance."""
    counts = np.bincount(labels, minlength=10)
    print(f"{name}: {count} images, {labels.shape[0]} labels")
    print("  per class: " + ", ".join(f"{digit}: {n}" for digit, n in enumerate(counts)))
    print(f"  smallest pixel {labels.min()}, largest {labels.max()}")


def main():
    x_train, y_train, x_test, y_test = build()
    for name, labels, images in (("x_train", y_train, x_train), ("x_test", y_test, x_test)):
        assert images.dtype == np.uint8 and images.shape[1:] == (28, 28), images.shape
        assert labels.shape == (images.shape[0],), labels.shape
        report(name, labels, images.shape[0])
    size = OUTPUT.stat().st_size
    print(f"wrote {OUTPUT} ({size / 1e6:.1f} MB for {x_train.size + x_test.size} bytes of pixels)")


if __name__ == "__main__":
    main()
