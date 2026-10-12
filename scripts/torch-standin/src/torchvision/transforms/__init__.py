# Partly ported from torchvision (torchvision/transforms/transforms.py), BSD-3-Clause.
# See THIRD_PARTY_NOTICES.md.
"""The transforms our notebooks use. Images are uint8 arrays here, not PIL images.

A transform takes whatever the dataset hands it, an Image, and gives the next transform
something it can take too; ToTensor is where an image becomes a tensor.
"""

import numpy as np

from torch._missing import module_getattr

from ..datasets import Image
from . import functional
from .functional import InterpolationMode, normalize, to_tensor


class Compose:
    """Apply transforms in order, as torchvision does."""

    def __init__(self, transforms):
        self.transforms = list(transforms) if isinstance(transforms, (list, tuple)) else [transforms]

    def __call__(self, img):
        for transform in self.transforms:
            img = transform(img)
        return img

    def __repr__(self):
        body = ", ".join(repr(transform) for transform in self.transforms)
        return f"Compose([{body}])"


class ToTensor:
    """A uint8 image as a float32 (C, H, W) tensor scaled to [0, 1]."""

    def __call__(self, pic):
        return to_tensor(pic)

    def __repr__(self):
        return "ToTensor()"


class Normalize:
    """(image - mean) / std, with a mean and a standard deviation per channel."""

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        return normalize(tensor, self.mean, self.std, self.inplace)

    def __repr__(self):
        return f"Normalize(mean={self.mean}, std={self.std})"


class Lambda:
    """Apply the given function to the image."""

    def __init__(self, lambd):
        if not callable(lambd):
            raise TypeError(f"Argument lambd should be callable, got {type(lambd).__name__}")
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return f"Lambda({self.lambd})"


class Resize:
    """The image at a new size: `size` as an int for the smaller edge, or [height, width].

    An Image stays an Image and a tensor stays a tensor, but a resized Image goes through the
    same bilinear kernel rather than through PIL, which the browser does not have.
    """

    def __init__(
        self,
        size,
        interpolation=InterpolationMode.BILINEAR,
        max_size=None,
        antialias=True,
    ):
        self.size = size
        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias

    def __call__(self, img):
        if not isinstance(img, Image):
            return functional.resize(
                img, self.size, self.interpolation, self.max_size, self.antialias
            )
        array = np.asarray(img)
        resized = functional.resize(
            to_tensor(array), self.size, self.interpolation, self.max_size, self.antialias
        )
        planes = np.rint(resized.detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        return Image(planes[:, :, 0] if planes.shape[2] == 1 else planes)

    def __repr__(self):
        return f"Resize(size={self.size})"


__getattr__ = module_getattr(__name__, "torchvision.transforms")
