# Partly ported from torchvision (torchvision/io/image.py), BSD-3-Clause.
# See THIRD_PARTY_NOTICES.md.
"""Reading image files, decoded here with zlib and NumPy so the browser needs no decoder.

Only 8-bit non-interlaced PNGs are read (colour types 0, 2, 4 and 6), which is what our
notebooks open. The format of a file comes from its signature and never from its name, so a
corrupt PNG is reported as a damaged PNG and a text file with a .png name is reported as not an
image. A format with no decoder here says which one it is and that it is not available in the
browser. An argument that is not a path at all, an Image or an array, is a TypeError naming
read_image, before any of that reading starts.
"""

import binascii
import os
import struct
import zlib
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from torch._missing import module_getattr, not_available

SIGNATURE = b"\x89PNG\r\n\x1a\n"
# The channels each PNG colour type decodes to: gray, RGB, gray+alpha, RGBA.
CHANNELS = {0: 1, 2: 3, 4: 2, 6: 4}
# The formats a file's first bytes give away, so the message can name the one it is.
FORMATS = (
    (b"\xff\xd8\xff", "JPEG"),
    (b"GIF87a", "GIF"),
    (b"GIF89a", "GIF"),
    (b"BM", "BMP"),
    (b"II*\x00", "TIFF"),
    (b"MM\x00*", "TIFF"),
)
# libpng's rgb-to-gray weights at libpng's 14 bits of precision, which is where torchvision's
# png_set_rgb_to_gray(0.2989, 0.587) lands: 0.29889, 0.58697, 0.11413, added up to 1 and cut off.
GRAY_WEIGHTS = (4897, 9617, 1870)
GRAY_SHIFT = 14

__getattr__ = module_getattr(__name__, "torchvision.io")


class ImageReadMode(Enum):
    """Which channels read_image gives back. The values are torchvision's."""

    UNCHANGED = 0
    GRAY = 1
    GRAY_ALPHA = 2
    RGB = 3
    RGB_ALPHA = 4
    RGBA = RGB_ALPHA  # an alias, as in torchvision


def read_image(path, mode=ImageReadMode.UNCHANGED):
    """Read an image file into a uint8 tensor of shape (channels, height, width).

    `mode` is an ImageReadMode or its name, as torchvision accepts.
    """
    return _tensor(_convert(_read(_path(path)), mode))


def _path(path):
    """`path` as a Path, or a TypeError that names torchvision.io.read_image.

    Without this the error is pathlib's own "argument should be a str or an os.PathLike
    object ... not 'ndarray'", which says nothing about which call was wrong. An Image or a
    NumPy array is the mistake a reader makes: it looks like something to read, and real
    torchvision turns it into a "No such file or directory" on the object's own repr.
    """
    if isinstance(path, (str, os.PathLike)):
        return Path(path)
    raise TypeError(
        f"torchvision.io.read_image: path should be a str or an os.PathLike object naming an "
        f"image file. Got {type(path).__name__}. To turn an image you already have into a "
        f"tensor, use torchvision.transforms.ToTensor()."
    )


def _not_available(what):
    return not_available(f"torchvision.io.read_image of {what}")


class Damaged(Exception):
    """A PNG that is not intact, however it is broken. Read turns it into a RuntimeError."""


def _read(path):
    """The file's pixels as a (C, H, W) uint8 array, or an error that says what is wrong.

    The format comes from the file's signature, never from its name, so a corrupt PNG gets an
    error about the PNG rather than being sent to Colab as if it were an unsupported format.
    """
    data = path.read_bytes()
    if not data:
        # Real torchvision says "Expected a non empty file" here.
        raise RuntimeError(f"torchvision.io.read_image: the file is empty: {path}")
    if data.startswith(SIGNATURE):
        try:
            return _decode_png(data)
        except (Damaged, struct.error, zlib.error) as err:
            # Real torchvision says "Out of bound read in decode_png. Probably, the input image
            # is corrupted" for a truncated file and crashes on a corrupt one.
            raise RuntimeError(
                f"torchvision.io.read_image: the PNG file is truncated or damaged: {path}"
            ) from err
    name = _format_of(data)
    if name is None:
        raise RuntimeError(f"torchvision.io.read_image: not an image file: {path}")
    raise _not_available(f"{name} files")


def _format_of(data):
    """The image format a file's first bytes give away, or None when they give nothing away."""
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    if data[4:8] == b"ftyp":
        return "HEIF"
    return next((name for head, name in FORMATS if data.startswith(head)), None)


def _chunks(data):
    """(the IHDR fields, the joined IDAT bytes), every chunk checked on the way past."""
    fields, compressed = None, []
    at = len(SIGNATURE)
    while at + 8 <= len(data):
        (length,) = struct.unpack(">I", data[at : at + 4])
        kind, payload = data[at + 4 : at + 8], data[at + 8 : at + 8 + length]
        stored = struct.unpack(">I", data[at + 8 + length : at + 12 + length])[0]
        if binascii.crc32(kind + payload) != stored:
            raise Damaged(f"the PNG chunk {kind.decode('ascii', 'replace')} has a bad CRC")
        if kind == b"IHDR":
            fields = struct.unpack(">IIBBBBB", payload)
        elif kind == b"IDAT":
            compressed.append(payload)
        elif kind == b"IEND":
            break
        at += 12 + length
    if fields is None:
        raise Damaged("the PNG has no IHDR chunk, so it has no header to read")
    return fields, b"".join(compressed)


def _decode_png(data):
    """A PNG as a (C, H, W) uint8 array: the header, then zlib, then the row filters."""
    header, compressed = _chunks(data)
    width, height, depth, kind, compression, method, interlace = header
    if compression != 0 or method != 0:
        raise _not_available(f"PNG files compressed in an unknown way ({width}x{height})")
    if depth != 8 or interlace != 0 or kind not in CHANNELS:
        raise _not_available(f"{_png_kind(depth, kind, interlace)} files")
    channels = CHANNELS[kind]
    stride = width * channels
    raw = zlib.decompress(compressed)
    if len(raw) != height * (stride + 1):
        raise Damaged(
            f"the PNG holds {len(raw)} bytes of pixels, not the {height * (stride + 1)} its "
            f"header ({width}x{height}, {channels} channels) needs"
        )
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, stride + 1)
    pixels = _unfilter(rows, width, channels)
    return np.ascontiguousarray(pixels.transpose(2, 0, 1))


def _png_kind(depth, kind, interlace):
    """How to name a PNG this decoder does not read."""
    if kind == 3:
        return "palette PNG"
    if depth != 8:
        return f"{depth}-bit PNG"
    if interlace:
        return "interlaced PNG"
    return f"PNG of colour type {kind}"


def _unfilter(rows, width, channels):
    """(H, W, C) pixels, with each row's filter taken off.

    Filters 0 (none), 1 (sub), 2 (up) and 3 (average) are undone a whole row at a time; the
    average and the Paeth predictor (4) need the pixel to the left, so those walk the row.
    """
    height = rows.shape[0]
    pixels = np.zeros((height, width, channels), dtype=np.uint8)
    previous = np.zeros((width, channels), dtype=np.int64)
    for y in range(height):
        filter_type, line = int(rows[y, 0]), rows[y, 1:].reshape(width, channels)
        line = line.astype(np.int64)
        if filter_type == 0:
            row = line
        elif filter_type == 1:
            # Each byte adds the one `channels` back, which is a running sum per channel.
            row = np.cumsum(line, axis=0) % 256
        elif filter_type == 2:
            row = (line + previous) % 256
        elif filter_type == 3:
            row = line.copy()
            for x in range(width):
                left = row[x - 1] if x else 0
                row[x] = (line[x] + (left + previous[x]) // 2) % 256
        elif filter_type == 4:
            row = line.copy()
            for x in range(width):
                left = row[x - 1] if x else 0
                upleft = previous[x - 1] if x else 0
                row[x] = (line[x] + _paeth(left, previous[x], upleft)) % 256
        else:
            raise Damaged(f"the PNG row {y} has filter type {filter_type}, which is not one")
        pixels[y] = row
        previous = row
    return pixels


def _paeth(left, up, upleft):
    """The PNG Paeth predictor: whichever of left, up and upleft is closest to their sum."""
    distance = left + up - upleft
    to_left, to_up, to_upleft = (np.abs(distance - x) for x in (left, up, upleft))
    return np.where(
        (to_left <= to_up) & (to_left <= to_upleft), left, np.where(to_up <= to_upleft, up, upleft)
    )


def _gray(rgb):
    """(3, H, W) RGB as one (1, H, W) plane of gray, with libpng's own weights."""
    red, green, blue = (plane.astype(np.int64) for plane in rgb)
    wr, wg, wb = GRAY_WEIGHTS
    return ((wr * red + wg * green + wb * blue) >> GRAY_SHIFT).astype(np.uint8)[None]


def _split(chw):
    """An image as its colours (one or three channels) and its alpha, which may be None."""
    channels = chw.shape[0]
    if channels == 1:
        return chw, None
    if channels == 2:
        return chw[:1], chw[1:]
    return chw[:3], chw[3:] if channels == 4 else None


def _rgb(color):
    """The colours as three channels, which is how libpng widens a gray image."""
    return np.repeat(color, 3, axis=0) if color.shape[0] == 1 else color


def _gray_of(color):
    """The colours as one channel of gray."""
    return color if color.shape[0] == 1 else _gray(color)


def _convert(chw, mode):
    """The channels of an image in this mode: repeated, dropped, or filled with 255.

    This follows what libpng does for each mode in torchvision's decode_png, so the values
    match torchvision's for every mode and every number of channels a PNG can have.
    """
    if isinstance(mode, str):
        try:
            mode = ImageReadMode[mode.upper()]
        except KeyError:
            raise ValueError(
                f"torchvision.io.read_image: {mode!r} is not a mode. The modes are "
                + ", ".join(member.name for member in ImageReadMode)
            ) from None
    if mode is ImageReadMode.UNCHANGED:
        return chw
    color, alpha = _split(chw)
    if mode is ImageReadMode.GRAY:
        return _gray_of(color)
    if mode is ImageReadMode.GRAY_ALPHA:
        return np.concatenate([_gray_of(color), _alpha(alpha, color)])
    if mode is ImageReadMode.RGB:
        return _rgb(color)
    return np.concatenate([_rgb(color), _alpha(alpha, color)])


def _alpha(alpha, color):
    """The image's own alpha channel, or an opaque one where it had none."""
    return alpha if alpha is not None else np.full_like(color[:1], 255)


def _tensor(array):
    return torch.from_numpy(np.ascontiguousarray(array))
