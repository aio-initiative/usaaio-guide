# Partly ported from torchvision (torchvision/transforms/functional.py), BSD-3-Clause.
# See THIRD_PARTY_NOTICES.md.
"""The tensor maths behind the transforms, with torchvision's values.

`resize` is the one that has to agree numerically: it is bilinear as torchvision is
(align_corners=False), computed the way ATen computes it, a separable pass along the width then
a pass along the height, with the same triangle filter and the same weights for antialiasing. A
resized uint8 image therefore matches real torchvision exactly for most sizes and within one
level of a uint8 pixel for the rest, and a resized float one matches to float32's precision.
"""

import math
from enum import Enum

import numpy as np
import torch

from torch._missing import not_available

from ..datasets import Image


class InterpolationMode(Enum):
    """The modes torchvision names. Only BILINEAR is here, the only one we resize with."""

    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"


def to_tensor(pic):
    """An image as a (C, H, W) tensor: byte levels scaled to [0, 1], anything else left alone.

    Takes an Image or a (H, W) or (H, W, C) array, as real torchvision does. A tensor gets
    torchvision's TypeError: it may well be (C, H, W) already, and reading it as (H, W, C) would
    scramble it without a word. Only uint8 is scaled and cast to float32; any other dtype keeps
    its own values, as it does in real torchvision.
    """
    if not isinstance(pic, (Image, np.ndarray)):
        # Real torch's Tensor prints as torch.Tensor, the stand-in's lives in torch._tensor.
        kind = "<class 'torch.Tensor'>" if type(pic) is torch.Tensor else type(pic)
        raise TypeError(f"pic should be PIL Image or ndarray. Got {kind}")
    array = np.asarray(pic)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError(f"pic should be 2/3 dimensional. Got {array.ndim} dimensions.")
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
    if tensor.dtype is torch.uint8:
        return tensor.to(torch.float32).div_(255)
    return tensor


def normalize(tensor, mean, std, inplace=False):
    """(image - mean) / std, with the mean and the standard deviation per channel."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"img should be a Tensor. Got {type(tensor).__name__}.")
    if not tensor.is_floating_point():
        raise TypeError(f"Input tensor should be a float tensor. Got {tensor.dtype}.")
    if tensor.ndim < 3:
        raise ValueError(
            "Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = "
            f"{list(tensor.shape)}"
        )
    if not inplace:
        tensor = tensor.clone()
    dtype = tensor.numpy().dtype
    mean, std = _per_channel(mean, dtype), _per_channel(std, dtype)
    if np.any(std == 0):
        raise ValueError(
            f"std evaluated to zero after conversion to {dtype}, leading to division by zero."
        )
    return tensor.sub_(mean).div_(std)


def _per_channel(value, dtype):
    """mean or std as an array that broadcasts over (C, H, W), as torchvision broadcasts it."""
    array = np.asarray(value, dtype=dtype)
    return array.reshape(-1, 1, 1) if array.ndim == 1 else array


def resize(img, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True):
    """Resize a (C, H, W) or (N, C, H, W) tensor, as torchvision's functional.resize does.

    `size` is an int, for the smaller edge, or [height, width]. Only bilinear is here, and the
    result is a new tensor with no autograd history, which is what an image needs and weights
    inside a model do not.
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"img should be a Tensor. Got {type(img).__name__}.")
    if interpolation != InterpolationMode.BILINEAR:
        raise not_available(
            f"torchvision.transforms.functional.resize with {interpolation.value} interpolation"
        )
    if img.ndim not in (3, 4):
        raise ValueError(f"Input tensor must be 3D or 4D, got {img.ndim}D.")
    if isinstance(size, int):
        size = [size]
    elif len(size) not in (1, 2):
        raise ValueError(
            f"Size must be an int or a 1 or 2 element list, not a {len(size)} element list."
        )
    height, width = img.shape[-2], img.shape[-1]
    new_height, new_width = _output_size((height, width), list(size), max_size)
    if (new_height, new_width) == (height, width):
        return img
    data = img.detach().numpy()
    # Both passes in float, with the one rounding at the end. Rounding the uint8 pass in between
    # as well puts the answer a tenth of a level further from ATen's on random images, so the
    # pass that feeds the height keeps its fractions.
    values = _resample(data.astype(np.float64), -1, new_width, antialias)
    values = _resample(values, -2, new_height, antialias)
    return torch.from_numpy(_as_dtype(values, data.dtype))


def _output_size(image_size, size, max_size=None):
    """(new_height, new_width) for an image of `image_size` and a requested `size`."""
    height, width = image_size
    if len(size) == 1:
        short, long = (width, height) if width <= height else (height, width)
        new_short, new_long = size[0], int(size[0] * long / short)
        if max_size is not None:
            if max_size <= new_short:
                raise ValueError(
                    f"max_size = {max_size} must be strictly greater than the requested size "
                    f"for the smaller edge size = {new_short}"
                )
            if new_long > max_size:
                new_short, new_long = int(max_size * new_short / new_long), max_size
        new_width, new_height = (new_short, new_long) if width <= height else (new_long, new_short)
    else:
        new_height, new_width = size[0], size[1]
    if new_height <= 0 or new_width <= 0:
        raise ValueError(f"the requested size {size} gives an empty image.")
    return new_height, new_width


def _resample(array, axis, size, antialias):
    """One axis of a separable bilinear resize, keeping the axis in place.

    Both passes read the input row by row (along the last axis) or column by column, which is
    what ATen's separable kernel does: the width first, then the height.
    """
    indices, weights = _weights(array.shape[axis], size, antialias)
    if axis == -1:
        return (array[..., indices] * weights).sum(axis=-1)
    # np.take leaves the index axis where it took the values from, one axis along.
    picked = np.take(array, indices, axis=-2)
    return (picked * weights[:, :, None]).sum(axis=-2)


def _weights(input_size, output_size, antialias):
    """The input indices and weights of every output pixel along one axis, as ATen has them.

    With antialiasing, a triangle filter of half-width max(1, scale) is sampled and normalized,
    which is what keeps a downsampled image from aliasing. Without it, each output pixel takes
    its two nearest input pixels, the second one repeated at the right edge.
    """
    scale = input_size / output_size
    support = scale if antialias and scale >= 1.0 else 1.0
    count = math.ceil(support) * 2 + 1 if antialias else 2
    indices = np.zeros((output_size, count), dtype=np.int64)
    weights = np.zeros((output_size, count), dtype=np.float64)
    for out_index in range(output_size):
        if antialias:
            first, row = _antialiased_row(scale, support, count, input_size, out_index)
        else:
            first, row = _bilinear_row(scale, input_size, out_index)
        indices[out_index, : len(row)] = first + np.arange(len(row))
        weights[out_index, : len(row)] = row
    return indices, weights


def _antialiased_row(scale, support, count, input_size, out_index):
    """The input offset and weights of one output pixel, with the triangle filter."""
    centre = scale * (out_index + 0.5)
    # Upscaling samples the filter at full width, downscaling at its own width.
    inverse = 1.0 / scale if scale >= 1.0 else 1.0
    first = max(int(centre - support + 0.5), 0)
    size = min(int(centre + support + 0.5), input_size) - first
    size = min(max(size, 0), count)
    row = [_triangle((offset + first - centre + 0.5) * inverse) for offset in range(size)]
    total = sum(row)
    return first, [value / total for value in row] if total else row


def _bilinear_row(scale, input_size, out_index):
    """The input offset and weights of one output pixel, from its two nearest pixels."""
    source = max(scale * (out_index + 0.5) - 0.5, 0.0)
    first = min(math.floor(source), input_size - 1)
    distance = min(max(source - first, 0.0), 1.0)
    size = min(first + 2, input_size) - first
    # At the last pixel only one input pixel is left, so both weights land on it.
    row = [0.0] * size
    row[min(0, size - 1)] += 1.0 - distance
    row[min(1, size - 1)] += distance
    return first, row


def _triangle(x):
    """The filter ATen resamples with: 1 - |x| inside the unit interval, and 0 outside."""
    return 1.0 - abs(x) if abs(x) < 1.0 else 0.0


def _as_dtype(values, dtype):
    """uint8 output is rounded half up and clipped, as torch's uint8 kernel is."""
    if dtype == np.uint8:
        return np.clip(np.floor(values + 0.5), 0, 255).astype(np.uint8)
    return values.astype(dtype)
