"""The stand-in torchvision, compared with real torchvision.

`import torchvision` has to see the stand-in `torch`, so every check of the stand-in runs in
its own process with PYTHONPATH=src. The real torchvision (which sits beside the real torch
of the venv) runs here in the parent process, and the two are compared value by value.
"""

import binascii
import json
import os
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from conftest import SRC, TOOL

NPZ = TOOL.parent.parent / "public" / "pyodide" / "datasets" / "mnist.npz"
MODES = ["UNCHANGED", "GRAY", "GRAY_ALPHA", "RGB", "RGB_ALPHA"]
# The PNG colour types, by the number of channels a decoder sees them as.
COLOR_TYPES = {1: 0, 2: 4, 3: 2, 4: 6}
# Down, up, both at once, and a size that needs no change at all.
SIZES = [7, 13, 14, 21, 28, 56, 96, [13, 29], [9, 40], [40, 9]]
# The resize tolerance: ATen multiplies int16 weights and shifts, where the stand-in rounds a
# float sum half up, so a uint8 result lands within a level of a uint8 pixel of ATen's and a
# float one matches to float32's precision. Widths of a single pixel are left out of the parity
# checks: ATen answers those with one row repeated, which
# test_a_one_pixel_wide_resize_is_where_aten_and_the_stand_in_disagree records instead.

PREAMBLE = """import json
import os

STEADIN_SRC = os.environ["STEADIN_SRC"]

# Checked before the import, so a missing stand-in reads as itself rather than as an import
# error from the venv's real torchvision, which cannot load beside this torch.
assert os.path.isfile(os.path.join(STEADIN_SRC, "torchvision", "__init__.py")), (
    "src/torchvision is not written yet"
)

import numpy as np
import torch
import torchvision
from torchvision import datasets, io, transforms

assert torch.__file__.startswith(STEADIN_SRC), "torch is not the stand-in: " + torch.__file__
assert torchvision.__file__.startswith(STEADIN_SRC), "torchvision is not the stand-in: " + torchvision.__file__


def shape_and_dtype(tensor):
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}


def report(value):
    # JSON-safe: a tensor or an array becomes its dtype, shape and values.
    if isinstance(value, torch.Tensor):
        return {
            "tensor": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "values": value.detach().numpy().tolist(),
        }
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "values": value.tolist()}
    if isinstance(value, dict):
        return {key: report(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [report(item) for item in value]
    return value


def result(value):
    print(json.dumps(report(value)))
"""


def run(body, env=None):
    """Run body in a process where torch and torchvision are the stand-in."""
    return subprocess.run(
        [sys.executable, "-c", PREAMBLE + body],
        capture_output=True,
        text=True,
        cwd=TOOL,
        env=dict(os.environ, PYTHONPATH=str(SRC), STEADIN_SRC=str(SRC), **(env or {})),
    )


def standin(body, env=None):
    """What body gives on the stand-in, as plain JSON, read back from result()."""
    done = run(body, env)
    assert done.returncode == 0, f"the stand-in failed:\n{done.stderr}"
    return json.loads(done.stdout)


def standin_error(body):
    """The name and the message of the error the stand-in raises for body."""
    done = run(body)
    assert done.returncode != 0, f"the stand-in did not fail:\n{done.stdout}"
    name, _, message = done.stderr.strip().splitlines()[-1].partition(": ")
    return name, message


def values_of(reported):
    """The values a reported tensor or array holds, as a NumPy array."""
    dtype = reported["dtype"].replace("torch.", "")
    return np.array(reported["values"], dtype=np.dtype(dtype)).reshape(reported["shape"])


def close_to(reported, want, atol=0):
    """The reported values are want's, to within atol levels."""
    got = values_of(reported)
    assert got.shape == want.shape, f"{got.shape} is not {want.shape}"
    assert got.dtype == want.dtype, f"{got.dtype} is not {want.dtype}"
    if got.size == 0:
        return
    worst = np.abs(got.astype(np.float64) - want.astype(np.float64)).max()
    assert worst <= atol, f"the values differ by {worst}, more than {atol}"


def pixels(chw):
    """`chw` as a Python list literal, to hand the stand-in the same image."""
    return repr(chw.tolist())


def literal(value):
    """Any NumPy value or dict of them as a Python literal, for the stand-in to read back."""
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key!r}: {literal(item)}" for key, item in value.items()) + "}"
    if isinstance(value, np.ndarray):
        return literal(value.tolist())
    if isinstance(value, list):
        return "[" + ", ".join(literal(item) for item in value) + "]"
    return repr(value)


def chunk(kind, data):
    """One PNG chunk: its length, its kind, its data and its CRC."""
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", binascii.crc32(body))


def png_bytes(chw, filter_type=0):
    """An 8-bit non-interlaced PNG of `chw` with the given row filter. Real torchvision's
    write_png only takes 1 or 3 channels and always writes filter 0, so this is here to cover
    gray+alpha, RGBA and every filter."""
    channels, height, width = chw.shape
    rows = chw.transpose(1, 2, 0).reshape(height, width * channels).astype(np.int64)
    raw = bytearray()
    previous = np.zeros(width * channels, dtype=np.int64)
    for row in rows:
        if filter_type == 0:
            prediction = np.zeros_like(row)
        elif filter_type == 1:
            prediction = np.concatenate([np.zeros(channels, dtype=np.int64), row[:-channels]])
        elif filter_type == 2:
            prediction = previous
        else:
            left = np.concatenate([np.zeros(channels, dtype=np.int64), row[:-channels]])
            upleft = np.concatenate([np.zeros(channels, dtype=np.int64), previous[:-channels]])
            if filter_type == 3:
                prediction = (left + previous) // 2
            else:
                distance = left + previous - upleft
                da, db, dc = (np.abs(distance - x) for x in (left, previous, upleft))
                prediction = np.where(
                    (da <= db) & (da <= dc), left, np.where(db <= dc, previous, upleft)
                )
        raw.append(filter_type)
        raw.extend(((row - prediction) % 256).astype(np.uint8).tobytes())
        previous = row
    header = struct.pack(">IIBBBBB", width, height, 8, COLOR_TYPES[channels], 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )


def sample(channels, height=9, width=11):
    """A deterministic uint8 image with sharp edges and gradients, so a resize has to work."""
    rows = np.arange(height * width).reshape(height, width)
    columns = np.broadcast_to(np.arange(width), (height, width))
    rng = np.random.default_rng(height * 100 + width * 10 + channels)
    planes = [
        (rows * 7) % 256,
        rng.integers(0, 256, (height, width)),
        (columns * 13 + 5) % 256,
        (rows * 3 + columns * 11 + 255) % 256,
    ]
    return np.stack(planes[:channels]).astype(np.uint8)


@pytest.fixture(scope="session")
def torch():
    """The real torch from the venv, which real torchvision also uses."""
    import torch as real

    assert SRC not in Path(real.__file__).parents, "src is on sys.path, so this is the stand-in"
    return real


@pytest.fixture(scope="session")
def tv(torch):
    """The real torchvision from the venv."""
    import torchvision

    assert SRC not in Path(torchvision.__file__).parents
    return torchvision


@pytest.fixture(scope="session")
def mnist_root():
    """Where real MNIST is downloaded, once for the whole session."""
    return Path(tempfile.gettempdir()) / "torch-standin-real-mnist"


@pytest.fixture(scope="session")
def real_mnist(tv, mnist_root):
    """Real torchvision's MNIST, both splits. Skipped where there is no network, since the
    subset we ship has to be its first images in its order."""
    try:
        return {
            train: tv.datasets.MNIST(str(mnist_root), train=train, download=True)
            for train in (True, False)
        }
    except Exception as err:  # noqa: BLE001  (any failure here means we cannot check parity)
        pytest.skip(f"real MNIST is not available here: {err}")


def test_the_dataset_has_ten_thousand_train_and_two_thousand_test_images():
    got = standin(
        """
train = datasets.MNIST("data", train=True, download=True)
test = datasets.MNIST("data", train=False, download=True)
result({"train": len(train), "test": len(test), "classes": test.classes})
"""
    )
    assert got["train"] == 10000
    assert got["test"] == 2000
    assert got["classes"] == ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


def test_data_and_targets_are_the_tensors_torchvision_returns():
    got = standin(
        """
train = datasets.MNIST("data")
image, target = train[0]
result({
    "data": shape_and_dtype(train.data),
    "targets": shape_and_dtype(train.targets),
    "pixels": shape_and_dtype(np.asarray(image)),
    "target": target,
    "target_type": type(target).__name__,
    "size": list(image.size),
    "mode": image.mode,
    "train": train.train,
    "index_9999": train[9999][1],
})
"""
    )
    assert got["data"] == {"shape": [10000, 28, 28], "dtype": "torch.uint8"}
    assert got["targets"] == {"shape": [10000], "dtype": "torch.int64"}
    assert got["pixels"] == {"shape": [28, 28], "dtype": "uint8"}
    assert got["target_type"] == "int"
    assert 0 <= got["target"] <= 9
    assert got["size"] == [28, 28]  # PIL gives (width, height)
    assert got["mode"] == "L"
    assert got["train"] is True
    assert 0 <= got["index_9999"] <= 9


def test_the_test_split_is_a_dataset_of_its_own():
    got = standin(
        """
test = datasets.MNIST("data", train=False)
result({"train": test.train, "targets": shape_and_dtype(test.targets), "n": len(test)})
"""
    )
    assert got == {"train": False, "targets": {"shape": [2000], "dtype": "torch.int64"}, "n": 2000}


def test_the_first_item_after_to_tensor_is_real_torchvisions_first_item(tv, real_mnist, mnist_root):
    got = standin(
        """
train = datasets.MNIST("data", transform=transforms.ToTensor())
image, target = train[0]
result({
    "image": image,
    "target": target,
    "min": float(image.min()),
    "max": float(image.max()),
})
"""
    )
    assert got["image"]["shape"] == [1, 28, 28]
    assert got["image"]["dtype"] == "torch.float32"
    assert got["min"] >= 0.0 and got["max"] <= 1.0
    # Real torchvision's own first train item, through its own ToTensor and its own dataset.
    real = tv.datasets.MNIST(
        str(mnist_root), train=True, download=True, transform=tv.transforms.ToTensor()
    )
    want_image, want_target = real[0]
    close_to(got["image"], want_image.numpy())
    assert got["target"] == want_target == real_mnist[True].targets[0].item()


def test_to_tensor_takes_a_gray_array_or_an_hwc_array():
    gray = np.array([[0, 128], [255, 64]], dtype=np.uint8)
    rgb = np.stack([gray, 255 - gray, gray // 2], axis=-1).astype(np.uint8)
    got = standin(
        f"""
gray = np.array({pixels(gray)}, dtype=np.uint8)
rgb = np.array({pixels(rgb)}, dtype=np.uint8)
result({{"gray": transforms.ToTensor()(gray), "rgb": transforms.ToTensor()(rgb)}})
"""
    )
    close_to(got["gray"], np.array([[[0.0, 128 / 255], [1.0, 64 / 255]]], dtype=np.float32))
    close_to(got["rgb"], np.moveaxis(rgb, -1, 0).astype(np.float32) / 255)


def test_to_tensor_leaves_an_array_that_is_not_bytes_alone(tv):
    """Only byte levels are scaled: anything else keeps its values and its dtype."""
    arrays = {
        "float32_gray": np.arange(6, dtype=np.float32).reshape(2, 3),
        "float32_hwc": np.linspace(0, 1, 12, dtype=np.float32).reshape(2, 2, 3),
        "int64_hwc": np.arange(12, dtype=np.int64).reshape(2, 2, 3),
        "uint8_gray": np.arange(6, dtype=np.uint8).reshape(2, 3),
    }
    body = literal({name: (array.dtype.name, array.tolist()) for name, array in arrays.items()})
    got = standin(
        f"""
arrays = {{name: np.array(values, dtype=dtype) for name, (dtype, values) in {body}.items()}}
result({{name: transforms.ToTensor()(array) for name, array in arrays.items()}})
"""
    )
    for name, array in arrays.items():
        want = tv.transforms.functional.to_tensor(array)
        assert got[name]["dtype"] == str(want.dtype)
        close_to(got[name], want.numpy())


def test_an_array_of_an_image_cannot_be_written_through_to_the_dataset():
    """np.array(image) is PIL's fresh copy, not a window on the dataset's own pixels."""
    got = standin(
        """
dataset = datasets.MNIST("data")

def write(array, value):
    array[14, 14] = value
    return int(array[14, 14])

before = int(dataset.data[0, 14, 14])
result({
    "before": before,
    "array": write(np.array(dataset[0][0]), 84),
    "asarray": write(np.asarray(dataset[0][0]), 7),
    "copy_true": write(np.array(dataset[0][0], copy=True), 9),
    "after": int(dataset.data[0, 14, 14]),
})
"""
    )
    assert got["before"] == 240
    assert got["array"] == 84 and got["asarray"] == 7 and got["copy_true"] == 9
    assert got["after"] == got["before"], "writing to an array of an image reached m.data"


def test_an_images_size_is_width_then_height_on_an_image_that_is_not_square():
    got = standin(
        """
wide = datasets.Image(np.zeros((3, 7), dtype=np.uint8))
tall = datasets.Image(np.zeros((9, 4), dtype=np.uint8))
coloured = datasets.Image(np.zeros((5, 6, 3), dtype=np.uint8))
result({
    "wide": {"size": list(wide.size), "mode": wide.mode},
    "tall": {"size": list(tall.size), "mode": tall.mode},
    "coloured": {"size": list(coloured.size), "mode": coloured.mode},
})
"""
    )
    assert got["wide"] == {"size": [7, 3], "mode": "L"}
    assert got["tall"] == {"size": [4, 9], "mode": "L"}
    assert got["coloured"] == {"size": [6, 5], "mode": "RGB"}


def test_a_target_transform_is_applied_to_the_target():
    got = standin(
        """
train = datasets.MNIST("data", target_transform=lambda target: target + 100)
test = datasets.MNIST("data", train=False, target_transform=str)
both = datasets.MNIST(
    "data", transform=transforms.ToTensor(), target_transform=lambda t: t * 2)
result({
    "plain": datasets.MNIST("data")[0][1],
    "plain_type": type(datasets.MNIST("data")[0][1]).__name__,
    "shifted": train[0][1],
    "shifted_type": type(train[0][1]).__name__,
    "string": test[0][1],
    "string_target": datasets.MNIST("data", train=False)[0][1],
    "double": both[0][1],
    "double_shape": list(both[0][0].shape),
})
"""
    )
    assert got["plain_type"] == "int"
    assert got["shifted"] == got["plain"] + 100
    assert got["shifted_type"] == "int", "a target transform returning an int stays an int"
    assert got["string"] == str(got["string_target"])
    assert got["double"] == 2 * got["plain"]
    assert got["double_shape"] == [1, 28, 28]


def test_to_tensor_refuses_something_that_is_not_an_image():
    kind, message = standin_error("transforms.ToTensor()([1, 2, 3])")
    assert kind == "TypeError"
    # The wording real torchvision's to_tensor uses.
    assert "pic should be PIL Image or ndarray" in message


@pytest.mark.parametrize(
    "value",
    ["torch.zeros(1, 4, 4)", "torch.zeros(4, 4, 3, dtype=torch.uint8)", "np.float32(0.5)"],
)
def test_to_tensor_refuses_a_tensor_or_a_scalar_as_torchvision_does(tv, value):
    """A tensor is already a tensor. Real torchvision raises rather than read a (C, H, W) image
    as (H, W, C), which would hand back a (W, C, H) tensor without a word."""
    kind, message = standin_error(f"transforms.ToTensor()({value})")
    with pytest.raises(TypeError) as real:
        tv.transforms.ToTensor()(eval(value, {"torch": tv.torch, "np": np}))
    assert kind == "TypeError"
    assert message == str(real.value)


def test_compose_normalize_and_lambda_match_real_torchvision(tv, real_mnist):
    got = standin(
        """
mean, std = (0.1307,), (0.3081,)
image = datasets.MNIST("data")[3][0]
composed = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std),
     transforms.Lambda(lambda tensor: tensor.flatten())]
)
one_step = composed.transforms[1](composed.transforms[0](image))
result({
    "composed": composed(image),
    "normalized_first_five": one_step.flatten()[:5],
    "input_untouched": composed.transforms[0](image),
})
"""
    )
    image = real_mnist[True][3][0]
    tensor = tv.transforms.ToTensor()(image)
    normalized = tv.transforms.Normalize((0.1307,), (0.3081,))(tensor)
    close_to(got["composed"], normalized.flatten().numpy(), 1e-6)
    close_to(got["normalized_first_five"], normalized.flatten()[:5].numpy(), 1e-6)
    close_to(got["input_untouched"], tensor.numpy())


def test_normalize_takes_a_channel_wise_mean_and_std(tv, real_mnist):
    got = standin(
        """
image = datasets.MNIST("data")[0][0]
one = transforms.ToTensor()(image)
tensor = torch.cat([one, one, one], dim=0)
result({
    "normalized": transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.25, 0.5])(tensor),
    "before": tensor,
})
"""
    )
    want_before = tv.transforms.ToTensor()(real_mnist[True][0][0]).repeat(3, 1, 1)
    want = tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.25, 0.5])(want_before)
    close_to(got["normalized"], want.numpy(), 1e-6)
    close_to(got["before"], want_before.numpy())


def test_resize_takes_an_int_for_the_smaller_edge_or_a_height_and_a_width(tv, torch):
    image = sample(3, height=20, width=10)
    got = standin(
        f"""
image = torch.from_numpy(np.array({pixels(image)}, dtype=np.uint8))
result({{
    "small_edge": transforms.functional.resize(image, 14),
    "both": transforms.functional.resize(image, [7, 33]),
}})
"""
    )
    close_to(
        got["small_edge"],
        tv.transforms.functional.resize(torch.from_numpy(image), 14).numpy(),
        1,
    )
    close_to(
        got["both"],
        tv.transforms.functional.resize(torch.from_numpy(image), [7, 33]).numpy(),
        1,
    )


def test_the_resize_transform_on_an_mnist_image_matches_real_torchvision(tv, real_mnist):
    """Resize on an Image is real torchvision's Resize, within the same level of tolerance.

    Real torchvision resizes a PIL image with PIL and a tensor with ATen, and those two kernels
    differ by a level here and there, so this is the same 1-level bound the tensor resize is
    held to rather than an exact match.
    """
    got = standin(
        """
dataset = datasets.MNIST("data")
smaller_edge = transforms.Resize(14)(dataset[0][0])
both = transforms.Resize([20, 10])(dataset[1][0])
up = transforms.Resize(42)(dataset[2][0])
result({
    "smaller_edge": np.asarray(smaller_edge),
    "both": np.asarray(both),
    "up": np.asarray(up),
    "kind": type(smaller_edge).__name__,
    "mode": smaller_edge.mode,
})
"""
    )
    assert got["kind"] == "Image", "Resize must give an Image an Image back, not a tensor"
    assert got["mode"] == "L"
    for key, index, size in (
        ("smaller_edge", 0, 14),
        ("both", 1, [20, 10]),
        ("up", 2, 42),
    ):
        # The PIL image real torchvision's MNIST dataset builds from the same raw pixels.
        pil = PILImage.fromarray(real_mnist[True].data[index].numpy(), mode="L")
        want = tv.transforms.functional.resize(pil, size)
        close_to(got[key], np.asarray(want), 1)


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("filter_type", [0, 1, 2, 3, 4])
def test_read_image_matches_real_torchvision_in_every_mode(
    tmp_path, tv, torch, channels, filter_type
):
    image = sample(channels)
    path = tmp_path / "image.png"
    path.write_bytes(png_bytes(image, filter_type))
    reads = ", ".join(
        f"'{mode}': io.read_image({str(path)!r}, mode=io.ImageReadMode.{mode})" for mode in MODES
    )
    got = standin(f"result({{{reads}}})")

    for mode in MODES:
        want = tv.io.read_image(str(path), mode=getattr(tv.io.ImageReadMode, mode))
        assert want.dtype == torch.uint8
        close_to(got[mode], want.numpy())
    # UNCHANGED is the file's own channels, whatever the other modes did to theirs.
    assert got["UNCHANGED"]["shape"] == [channels, 9, 11]


def test_read_image_reads_what_real_torchvision_wrote(tmp_path, tv, torch):
    path = str(tmp_path / "gray.png")
    tv.io.write_png(torch.from_numpy(sample(1)), path)
    got = standin(f"result({{'rgb': io.read_image({path!r}, mode='RGB')}})")
    close_to(got["rgb"], tv.io.read_image(path, mode=tv.io.ImageReadMode.RGB).numpy())

    path = str(tmp_path / "rgb.png")
    image = sample(3)
    tv.io.write_png(torch.from_numpy(image), path)
    got = standin(f"result({{'rgb': io.read_image({path!r})}})")
    close_to(got["rgb"], image)


def test_read_image_decodes_random_noise_for_every_filter(tmp_path, tv, torch):
    """Random bytes, which is what puts Paeth ties in a file.

    Paeth's predictor has to break a tie between the pixel above and the one above-left. With
    noise, plenty of pixels are exactly on that tie, so this is where the rule that picks the
    pixel above is checked against libpng, through real torchvision's read.
    """
    rng = np.random.default_rng(5)
    for filter_type in range(5):
        image = rng.integers(0, 256, (3, 41, 37), dtype=np.uint8)
        assert paeth_ties(image) > 0, "no Paeth ties in this noise, the test would prove nothing"
        path = tmp_path / f"noise{filter_type}.png"
        path.write_bytes(png_bytes(image, filter_type))
        got = standin(f"result(io.read_image({str(path)!r}))")
        want = tv.io.read_image(str(path))
        assert want.dtype == torch.uint8
        close_to(got, want.numpy())


def paeth_ties(image):
    """How many pixels of `image` have the Paeth predictor's tie between up and up-left."""
    rows = image[0].astype(np.int64)
    padded = np.zeros((rows.shape[0] + 1, rows.shape[1] + 1), dtype=np.int64)
    padded[1:, 1:] = rows
    left, up, upleft = padded[1:, :-1], padded[:-1, 1:], padded[:-1, :-1]
    to_up = np.abs(left - upleft)
    to_upleft = np.abs(left + up - 2 * upleft)
    to_left = np.abs(up - upleft)
    return int(((to_up == to_upleft) & (to_left > to_up)).sum())


def test_read_image_says_which_image_format_it_cannot_read(tmp_path):
    """A real signature for a format the browser has no decoder for names that format."""
    formats = {
        "photo.jpg": (b"\xff\xd8\xff\xe0\x00\x10JFIF not really a jpeg", "JPEG"),
        "anim.gif": (b"GIF89a" + b"\x00" * 40, "GIF"),
        "picture.webp": (b"RIFF\x20\x00\x00\x00WEBPVP8 " + b"\x00" * 20, "WEBP"),
        "art.bmp": (b"BM" + b"\x00" * 40, "BMP"),
        "scan.tiff": (b"II*\x00" + b"\x00" * 40, "TIFF"),
    }
    for name, (data, expected) in formats.items():
        path = tmp_path / name
        path.write_bytes(data)
        kind, message = standin_error(f"io.read_image({str(path)!r})")
        assert kind == "NotImplementedError", f"{name} gave {kind}"
        assert expected in message, message
        assert "not available in the browser version of PyTorch" in message
        assert "torchvision.io.read_image" in message


def test_a_file_that_is_not_an_image_at_all_says_so(tmp_path):
    """The format comes from the signature, so a text file named .png is not sent to Colab."""
    for name, data in (
        ("mystery.png", b"\x01\x02\x03\x04" + b"\x00" * 40),
        ("notes.txt", b"this is not an image at all\n"),
        ("empty.png", b""),
    ):
        path = tmp_path / name
        path.write_bytes(data)
        kind, message = standin_error(f"io.read_image({str(path)!r})")
        assert kind == "RuntimeError", f"{name} gave {kind}"
        assert "not an image file" in message or "the file is empty" in message, message
        assert "not available in the browser version" not in message, (
            "a file with no image signature must not be sent to Colab"
        )


@pytest.mark.parametrize(
    "value,expected",
    [
        ("np.zeros((4, 4), dtype=np.uint8)", "ndarray"),
        ("datasets.MNIST('data')[0][0]", "Image"),
        ("torch.zeros(1, 4, 4)", "Tensor"),
        ("42", "int"),
        ("None", "NoneType"),
        ("['image.png']", "list"),
        ("b'image.png'", "bytes"),
    ],
)
def test_read_image_says_its_argument_is_not_a_path(value, expected):
    """read_image takes a path. Anything else is a TypeError naming read_image, not pathlib's
    own "argument should be a str or an os.PathLike object", which names no call at all. An
    Image or an array is the mistake a reader makes: real torchvision turns it into a "No such
    file or directory" on the object's own repr."""
    kind, message = standin_error(f"io.read_image({value})")
    assert kind == "TypeError", f"{value} gave {kind}: {message}"
    assert "torchvision.io.read_image" in message, message
    assert expected in message, message
    assert "os.PathLike" in message, message
    assert "ToTensor" in message, "the message should say what to use on an image you have"


def test_read_image_still_takes_a_str_a_path_and_anything_path_like(tmp_path, tv):
    """The check is on the type, not on which paths are allowed: a str and a Path both work and
    give the same pixels, and the errors for a directory, a missing file and an empty file are
    unchanged."""
    path = tmp_path / "image.png"
    path.write_bytes(png_bytes(sample(3), 0))
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    for argument in (repr(str(path)), f"pathlib.Path({str(path)!r})"):
        got = standin(f"import pathlib\nresult(io.read_image({argument}))")
        close_to(got, tv.io.read_image(str(path)).numpy())

    for argument, kind_wanted, message_wanted in (
        (repr(str(tmp_path)), "IsADirectoryError", "Is a directory"),
        (repr(str(tmp_path / "missing.png")), "FileNotFoundError", "No such file"),
        (repr(str(empty)), "RuntimeError", "the file is empty"),
    ):
        kind, message = standin_error(f"io.read_image({argument})")
        assert kind == kind_wanted, f"{argument} gave {kind}: {message}"
        assert message_wanted in message, f"{argument} said {message}"
        assert "should be a str or an os.PathLike" not in message, (
            "a path of the wrong shape is not a wrongly-typed path"
        )


def test_a_damaged_png_says_it_is_damaged(tmp_path):
    """Every way a PNG can be broken gives one clear error, not a struct or zlib failure."""
    good = png_bytes(sample(3), 0)
    broken = {
        "truncated": good[: len(good) // 2],
        "header_only": good[:8],
        "signature_then_rubbish": good[:8] + b"nonsense rubbish bytes here",
        "bad_crc": good[:-13] + b"\x00\x00\x00\x00",
        "corrupt_idat": corrupt_idat(good),
        "no_pixels": good[:8]
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 9, 11, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00" + b"\x01" * 9))
        + chunk(b"IEND", b""),
    }
    for name, data in broken.items():
        path = tmp_path / f"{name}.png"
        path.write_bytes(data)
        kind, message = standin_error(f"io.read_image({str(path)!r})")
        assert kind == "RuntimeError", f"{name} gave {kind}: {message}"
        assert "truncated or damaged" in message, f"{name} said {message}"
        assert "not available in the browser version" not in message


def corrupt_idat(good):
    """A PNG whose compressed pixels are corrupted, with the chunk's CRC put right again."""
    data = bytearray(good)
    at = data.find(b"IDAT") + 4
    start = data.find(b"IDAT") - 4
    length = int.from_bytes(data[start : start + 4], "big")
    data[at + 5] ^= 0xFF
    struct.pack_into(">I", data, at + length, binascii.crc32(b"IDAT" + bytes(data[at : at + length])))
    return bytes(data)


def test_an_unknown_mode_lists_the_modes_there_are(tmp_path, tv):
    """Real torchvision raises KeyError('BOGUS'), which does not say what to type instead."""
    path = tmp_path / "image.png"
    path.write_bytes(png_bytes(sample(3), 0))
    kind, message = standin_error(f"io.read_image({str(path)!r}, mode='bogus')")
    assert kind == "ValueError"
    assert "is not a mode" in message
    for mode in MODES:
        assert mode in message
    # The modes that are named still work, and a lower case name works as it does there.
    got = standin(f"result(io.read_image({str(path)!r}, mode='gray'))")
    close_to(got, tv.io.read_image(str(path), mode=tv.io.ImageReadMode.GRAY).numpy())


def test_a_png_the_stand_in_cannot_decode_says_so(tmp_path):
    # A palette image, which real torchvision reads and the stand-in does not.
    path = tmp_path / "palette.png"
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 3, 0, 0, 0))
        + chunk(b"PLTE", b"\x00\x00\x00\xff\xff\xff")
        + chunk(b"IDAT", zlib.compress(b"\x00\x00\x01\x00\x00\x00\x01\x00\x00"))
        + chunk(b"IEND", b"")
    )
    kind, message = standin_error(f"io.read_image({str(path)!r})")
    assert kind == "NotImplementedError"
    assert "torchvision.io.read_image" in message
    assert "palette PNG" in message


@pytest.mark.parametrize("antialias", [True, False])
def test_resize_matches_real_torchvision_up_and_down(tv, torch, antialias):
    image = sample(3)
    batch = np.stack([image, 255 - image])
    for size in SIZES:
        got = standin(
            f"""
image = torch.from_numpy(np.array({pixels(image)}, dtype=np.uint8))
batch = torch.from_numpy(np.array({pixels(batch)}, dtype=np.uint8))
float_image = image.to(torch.float32) / 255
size = {size!r}
result({{
    "uint8": transforms.functional.resize(image, size, antialias={antialias}),
    "float32": transforms.functional.resize(float_image, size, antialias={antialias}),
    "batch": transforms.functional.resize(batch, size, antialias={antialias}),
}})
"""
        )
        want_uint8 = tv.transforms.functional.resize(
            torch.from_numpy(image), size, antialias=antialias
        )
        assert got["uint8"]["shape"] == list(want_uint8.shape)
        assert got["uint8"]["dtype"] == "torch.uint8"
        # ATen's uint8 kernel multiplies int16 weights and shifts, where the stand-in rounds a
        # float sum half up, so the two land on neighbouring levels now and then.
        close_to(got["uint8"], want_uint8.numpy(), 1)
        close_to(
            got["float32"],
            tv.transforms.functional.resize(
                torch.from_numpy(image).to(torch.float32) / 255, size, antialias=antialias
            ).numpy(),
            1e-4,
        )
        close_to(
            got["batch"],
            tv.transforms.functional.resize(
                torch.from_numpy(batch), size, antialias=antialias
            ).numpy(),
            1,
        )


def test_resize_matches_real_torchvision_across_a_sweep_of_sizes(tv, torch):
    """Many sizes, both antialias settings, uint8 and float, one image and a batch."""
    image = sample(3, height=17, width=11)
    batch = np.stack([image, 255 - image])
    sizes = list(range(2, 30)) + [[rows, 8] for rows in range(2, 20)]
    for antialias in (True, False):
        got = standin(
            f"""
image = torch.from_numpy(np.array({pixels(image)}, dtype=np.uint8))
batch = torch.from_numpy(np.array({pixels(batch)}, dtype=np.uint8))
float_image = image.to(torch.float32) / 255
out = {{}}
for size in {sizes!r}:
    out[repr(size)] = [
        transforms.functional.resize(image, size, antialias={antialias}),
        transforms.functional.resize(batch, size, antialias={antialias}),
        transforms.functional.resize(float_image, size, antialias={antialias}),
    ]
result(out)
"""
        )
        for size in sizes:
            reference = torch.from_numpy(image)
            batched = torch.from_numpy(batch)
            float_image = reference.to(torch.float32) / 255
            levels, batched_levels, floats = got[repr(size)]
            close_to(levels, tv.transforms.functional.resize(
                reference, size, antialias=antialias).numpy(), 1)
            close_to(batched_levels, tv.transforms.functional.resize(
                batched, size, antialias=antialias).numpy(), 1)
            close_to(floats, tv.transforms.functional.resize(
                float_image, size, antialias=antialias).numpy(), 1e-4)


def test_a_one_pixel_wide_resize_is_where_aten_and_the_stand_in_disagree(tv, torch):
    """A width of one pixel is the one size where the two libraries may part company.

    ATen 2.14.1 hands back the same value for every row of such an output, whichever rows the
    image started with. The stand-in weights each output row over the rows that belong to it, so
    its rows differ from each other. Nothing here asserts that ATen still does this: a torch that
    fixed it would simply agree with the stand-in, and the checks below hold either way. No
    notebook resizes to one pixel, so the stand-in keeps the answer that is actually right.
    """
    image = sample(1, height=20, width=10)
    got = standin(
        f"""
image = torch.from_numpy(np.array({pixels(image)}, dtype=np.uint8))
result({{
    "uint8": transforms.functional.resize(image, [4, 1]),
    "float32": transforms.functional.resize(image.to(torch.float32) / 255, [4, 1]),
    "rows_differ": int(
        transforms.functional.resize(image, [4, 1])[:, 1:]
        .sub(transforms.functional.resize(image, [4, 1])[:, :1]).abs().sum() > 0),
}})
"""
    )
    reference = torch.from_numpy(image)
    for key, tolerance in (("uint8", 1), ("float32", 1e-4)):
        source = reference if key == "uint8" else reference.to(torch.float32) / 255
        want = tv.transforms.functional.resize(source, [4, 1]).numpy()
        mine = values_of(got[key])
        assert mine.shape == want.shape
        # The stand-in's own rows differ, and its first row is ATen's, bug or no bug.
        assert np.abs(mine[:, 1:] - mine[:, :1]).max() > 0
        assert np.abs(mine[:, :1].astype(np.float64) - want[:, :1]).max() <= tolerance
    assert got["rows_differ"] == 1


def test_resize_of_real_mnist_images_matches_real_torchvision(tv, real_mnist):
    got = standin(
        """
images = datasets.MNIST("data").data[:4]
result({
    "down": transforms.functional.resize(images, 14),
    "up": transforms.functional.resize(images, 42),
    "odd": transforms.functional.resize(images, [21, 9]),
})
"""
    )
    images = real_mnist[True].data[:4]
    close_to(got["down"], tv.transforms.functional.resize(images, 14).numpy(), 1)
    close_to(got["up"], tv.transforms.functional.resize(images, 42).numpy(), 1)
    close_to(got["odd"], tv.transforms.functional.resize(images, [21, 9]).numpy(), 1)


def test_the_uint8_resize_is_within_a_few_hundredths_of_a_level_on_average(tv, torch):
    """A max of 1 level hides a bias, so the average has a bound of its own.

    ATen scales its weights to int16 and shifts where the stand-in rounds a float64 sum, and the
    two land on neighbouring levels for the pixels that sit near a boundary: a hundredth of a
    level per pixel, and exactly equal for the sizes where no pixel does. Rounding the wrong
    way, rounding the width pass again before the height pass, or shifting every value by a
    bias all move that average by more, which a max of 1 does not see.
    """
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 256, (3, 29, 37), dtype=np.uint8) for _ in range(4)]
    sizes = [7, 13, 14, 21, 28, 56, [9, 40], [40, 9]]
    for antialias in (True, False):
        for image in images:
            got = standin(
                f"""
image = torch.from_numpy(np.array({pixels(image)}, dtype=np.uint8))
out = {{}}
for size in {sizes!r}:
    out[repr(size)] = transforms.functional.resize(image, size, antialias={antialias})
result(out)
"""
            )
            for size in sizes:
                want = tv.transforms.functional.resize(
                    torch.from_numpy(image), size, antialias=antialias
                ).numpy()
                difference = np.abs(
                    values_of(got[repr(size)]).astype(np.int64) - want.astype(np.int64)
                )
                assert difference.max() <= 1, f"size {size} is {difference.max()} levels off"
                assert difference.mean() <= 0.05, (
                    f"size {size} averages {difference.mean():.3f} levels off, "
                    f"antialias={antialias}"
                )


def test_resize_leaves_a_tensor_alone_when_the_size_does_not_change():
    got = standin(
        """
image = torch.from_numpy(np.arange(2 * 8 * 8, dtype=np.uint8).reshape(2, 8, 8))
result({"resized": transforms.functional.resize(image, 8), "same": transforms.functional.resize(image, 8) is image})
"""
    )
    assert got["same"] is True
    assert got["resized"]["shape"] == [2, 8, 8]


def test_a_loop_over_sixty_four_items_matches_real_torchvision(tv, real_mnist):
    got = standin(
        """
train = datasets.MNIST("data", transform=transforms.ToTensor())
batch = torch.stack([train[i][0] for i in range(64)])
result({
    "batch": shape_and_dtype(batch),
    "first": batch[0],
    "second_last": batch[62],
    "mean": float(batch.mean()),
    "std": float(batch.std()),
    "min": float(batch.min()),
    "max": float(batch.max()),
    "sum": float(batch.sum()),
    "targets": [train[i][1] for i in range(64)],
})
"""
    )
    assert got["batch"] == {"shape": [64, 1, 28, 28], "dtype": "torch.float32"}
    to_tensor = tv.transforms.ToTensor()
    want = tv.torch.stack([to_tensor(real_mnist[True][i][0]) for i in range(64)])
    close_to(got["first"], want[0].numpy())
    close_to(got["second_last"], want[62].numpy())
    assert got["mean"] == pytest.approx(float(want.mean()), rel=1e-6)
    assert got["std"] == pytest.approx(float(want.std()), rel=1e-6)
    assert got["min"] == pytest.approx(float(want.min()))
    assert got["max"] == pytest.approx(float(want.max()))
    assert got["sum"] == pytest.approx(float(want.sum()), rel=1e-6)
    assert got["targets"] == [int(t) for t in real_mnist[True].targets[:64]]


def test_the_data_directory_environment_variable_is_used(tmp_path):
    np.savez_compressed(
        tmp_path / "mnist.npz",
        x_train=np.zeros((10000, 28, 28), dtype=np.uint8),
        y_train=np.full(10000, 7, dtype=np.int64),
        x_test=np.zeros((2000, 28, 28), dtype=np.uint8),
        y_test=np.full(2000, 3, dtype=np.int64),
    )
    got = standin(
        """
train = datasets.MNIST("ignored", train=True)
test = datasets.MNIST("ignored", train=False)
result({"len": len(train), "target": train[0][1], "test_len": len(test), "test_target": test[5][1]})
""",
        # The path the environment names, not the repository's copy.
        env=dict(STANDIN_DATA_DIR=str(tmp_path)),
    )
    assert got == {"len": 10000, "target": 7, "test_len": 2000, "test_target": 3}


def test_the_browsers_own_copy_of_the_data_wins_over_everything_else(tmp_path):
    """/home/pyodide/.standin/mnist.npz is where the page puts it, so it is read first."""
    browser = tmp_path / "pyodide-standin"
    browser.mkdir()
    write_fake_mnist(browser / "mnist.npz", target=4)
    other = tmp_path / "elsewhere"
    other.mkdir()
    write_fake_mnist(other / "mnist.npz", target=9)
    got = standin(
        f"""
# Stand in for the file the page writes, at the path the stand-in looks there first.
datasets.BROWSER_FILE = __import__("pathlib").Path({str(browser / "mnist.npz")!r})
result({{
    "target": datasets.MNIST("ignored")[0][1],
    "first": str(datasets.find_data_file()),
    "candidates": [str(path) for path in datasets.data_paths()[:3]],
}})
""",
        # Even with a data directory pointing somewhere else, the browser's copy comes first.
        env=dict(STANDIN_DATA_DIR=str(other)),
    )
    assert got["target"] == 4, "the browser's copy did not win"
    assert got["first"] == str(browser / "mnist.npz")
    assert got["candidates"][0] == str(browser / "mnist.npz")
    assert got["candidates"][1] == str(other / "mnist.npz")


def write_fake_mnist(path, target):
    """An npz shaped like the real subset, with a target a notebook can spot."""
    np.savez_compressed(
        path,
        x_train=np.zeros((10000, 28, 28), dtype=np.uint8),
        y_train=np.full(10000, target, dtype=np.int64),
        x_test=np.zeros((2000, 28, 28), dtype=np.uint8),
        y_test=np.full(2000, target, dtype=np.int64),
    )


def test_a_data_directory_without_the_file_falls_back_to_the_bundled_one(tmp_path):
    got = standin(
        """
result({"len": len(datasets.MNIST("data")), "mode": datasets.MNIST("data")[0][0].mode})
""",
        env=dict(STANDIN_DATA_DIR=str(tmp_path)),
    )
    assert got == {"len": 10000, "mode": "L"}


def test_the_missing_names_say_why_they_are_missing():
    kind, message = standin_error("result(torchvision.models.resnet50(weights=None))")
    assert kind == "NotImplementedError"
    assert message.startswith(
        "torchvision.models.resnet50 is not available in the browser version of PyTorch."
    )

    kind, message = standin_error("result(transforms.RandomRotation(10))")
    assert kind == "NotImplementedError"
    assert "torchvision.transforms.RandomRotation" in message

    kind, message = standin_error("result(io.decode_jpeg(b''))")
    assert kind == "NotImplementedError"
    assert "torchvision.io.decode_jpeg" in message


def test_the_surface_names_resolve():
    got = standin(
        """
result({
    "version": torchvision.__version__,
    "dataset": torchvision.datasets.MNIST.__name__,
    "mode_gray": int(torchvision.io.ImageReadMode.GRAY.value),
    "resize": torchvision.transforms.functional.resize.__name__,
    "functional": torchvision.transforms.functional.__name__,
})
"""
    )
    assert got["version"] == "0.29.1+standin"
    assert got["dataset"] == "MNIST"
    assert got["mode_gray"] == 1
    assert got["resize"] == "resize"
    assert got["functional"] == "torchvision.transforms.functional"


def test_the_bundled_npz_holds_the_first_images_of_real_mnist(real_mnist):
    if not NPZ.exists():
        pytest.skip(f"{NPZ} is not built yet; run make_mnist.py")
    with np.load(NPZ) as data:
        assert sorted(data.files) == ["x_test", "x_train", "y_test", "y_train"]
        assert data["x_train"].shape == (10000, 28, 28)
        assert data["x_test"].shape == (2000, 28, 28)
        assert data["x_train"].dtype == np.uint8
        assert data["x_test"].dtype == np.uint8
        assert data["y_train"].shape == (10000,)
        assert data["y_test"].shape == (2000,)
        np.testing.assert_array_equal(data["x_train"], real_mnist[True].data[:10000].numpy())
        np.testing.assert_array_equal(data["y_train"], real_mnist[True].targets[:10000].numpy())
        np.testing.assert_array_equal(data["x_test"], real_mnist[False].data[:2000].numpy())
        np.testing.assert_array_equal(data["y_test"], real_mnist[False].targets[:2000].numpy())
