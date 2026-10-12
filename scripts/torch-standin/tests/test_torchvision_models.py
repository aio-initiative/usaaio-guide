"""torchvision.models.resnet18 on the stand-in, compared with real torchvision's.

The stand-in's torchvision needs the stand-in's torch, so it runs in its own process with
PYTHONPATH=src. The real torchvision runs here. They meet in files: real weights go in as an
npz of the state_dict, and what the stand-in's forward and backward give comes back as one.
"""

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import SRC, TOOL


def inventory(model):
    """What the model holds, by name: parameters, buffers and the state_dict, with shapes and dtypes."""
    return {
        "parameters": [[n, list(p.shape), str(p.dtype)] for n, p in model.named_parameters()],
        "buffers": [[n, list(b.shape), str(b.dtype)] for n, b in model.named_buffers()],
        "state": [[n, list(v.shape), str(v.dtype)] for n, v in model.state_dict().items()],
        "count": sum(p.numel() for p in model.parameters()),
        "repr": repr(model),
    }


# The code that runs in the stand-in's process: its checks and inventory(), written once above.
PREAMBLE = """import json
import os
import sys

STEADIN_SRC = os.environ["STEADIN_SRC"]
import numpy as np
import torch
import torchvision
from torchvision import models

assert torch.__file__.startswith(STEADIN_SRC), "torch is not the stand-in: " + torch.__file__
assert torchvision.__file__.startswith(STEADIN_SRC), "torchvision is not the stand-in: " + torchvision.__file__


{inventory}


def result(value):
    print(json.dumps(value))
""".replace("{inventory}", inspect.getsource(inventory))


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
    """What body gives on the stand-in, from result()."""
    done = run(body, env)
    assert done.returncode == 0, f"the stand-in failed:\n{done.stderr}"
    return json.loads(done.stdout.strip().splitlines()[-1])


def standin_error(body):
    """The name and the message of the error the stand-in raises for body."""
    done = run(body)
    assert done.returncode != 0, f"the stand-in did not fail:\n{done.stdout}"
    name, _, message = done.stderr.strip().splitlines()[-1].partition(": ")
    return name, message


@pytest.fixture(scope="session")
def tv():
    """The real torchvision from the venv, which sits beside the real torch."""
    import torch
    import torchvision

    assert SRC not in Path(torch.__file__).parents, "src is on sys.path"
    return torchvision


@pytest.fixture(scope="session")
def torch(tv):
    import torch as real

    return real


# The structure.

# (builder, its arguments, its parameter count)
BUILDS = {
    "resnet18": ("resnet18", {}, 11_689_512),
    "resnet18 ten classes": ("resnet18", {"num_classes": 10}, 11_181_642),
    "resnet18 zero-init residual": ("resnet18", {"zero_init_residual": True}, 11_689_512),
    "resnet34": ("resnet34", {}, 21_797_672),
}


@pytest.mark.parametrize("name", BUILDS)
def test_the_models_have_torchvisions_parameters_buffers_and_layers(tv, name):
    builder, arguments, count = BUILDS[name]
    want = inventory(getattr(tv.models, builder)(**arguments))
    assert want["count"] == count
    got = standin(f"result(inventory(models.{builder}(**{arguments!r})))")
    assert got["count"] == count
    assert got["parameters"] == want["parameters"]
    assert got["buffers"] == want["buffers"]
    assert got["state"] == want["state"]
    assert got["repr"] == want["repr"]


def test_the_number_of_classes_changes_only_the_last_layer():
    default, ten = (standin(f"result(inventory(models.resnet18({a})))") for a in ("", "num_classes=10"))
    changed = [(a, b) for a, b in zip(default["state"], ten["state"]) if a != b]
    assert changed == [
        (["fc.weight", [1000, 512], "torch.float32"], ["fc.weight", [10, 512], "torch.float32"]),
        (["fc.bias", [1000], "torch.float32"], ["fc.bias", [10], "torch.float32"]),
    ]
    assert len(default["state"]) == len(ten["state"])
    assert default["count"] - ten["count"] == (1000 - 10) * 513


# The initialisation.

INITIAL = """
torch.manual_seed(0)
model = models.resnet18(**KWARGS)
convs = {n: [float(m.weight.std()), float(m.weight.mean()), list(m.weight.shape)]
         for n, m in model.named_modules() if isinstance(m, torch.nn.Conv2d)}
norms = {n: [m.weight.tolist(), m.bias.tolist()]
         for n, m in model.named_modules() if isinstance(m, torch.nn.BatchNorm2d)}
fc = [float(model.fc.weight.abs().max()), float(model.fc.bias.abs().max())]
result({"convs": convs, "norms": norms, "fc": fc})
"""


def test_convolutions_start_with_he_normal_weights_in_fan_out_mode():
    start = standin(INITIAL.replace("**KWARGS", ""))
    assert len(start["convs"]) == 20  # the stem, sixteen in the blocks, three shortcuts
    for name, (std, mean, (out, _, kh, kw)) in start["convs"].items():
        want = (2.0 / (out * kh * kw)) ** 0.5
        assert std == pytest.approx(want, rel=0.05), name
        assert abs(mean) < 0.1 * want, name
    # The linear layer keeps torch's own start: uniform on +-1/sqrt(512).
    assert 0.8 / 512**0.5 < start["fc"][0] <= 1 / 512**0.5
    assert start["fc"][1] <= 1 / 512**0.5


def test_batch_norm_starts_with_scale_one_and_shift_zero():
    start = standin(INITIAL.replace("**KWARGS", ""))
    assert len(start["norms"]) == 20
    for name, (weight, bias) in start["norms"].items():
        assert set(weight) == {1.0} and set(bias) == {0.0}, name


def test_zero_init_residual_zeroes_the_last_norm_scale_of_each_block():
    start = standin(INITIAL.replace("**KWARGS", "zero_init_residual=True"))
    for name, (weight, bias) in start["norms"].items():
        last = name.endswith(".bn2")
        assert set(weight) == ({0.0} if last else {1.0}), name
        assert set(bias) == {0.0}, name


# Pretrained weights are not in the browser.

@pytest.mark.parametrize(
    "arguments",
    ['weights="DEFAULT"', 'weights="IMAGENET1K_V1"', "weights=True", "pretrained=True", "weights=object()"],
)
def test_pretrained_weights_say_they_are_not_available_in_the_browser(arguments):
    name, message = standin_error(f"models.resnet18({arguments})")
    assert name == "NotImplementedError"
    assert message == (
        "torchvision.models.resnet18 with pretrained weights is not available in the browser version of "
        "PyTorch. Download the notebook or open it in Colab to use it."
    )


@pytest.mark.parametrize("arguments", ["", "None", "weights=None", "weights=False", "pretrained=False", "False"])
def test_no_weights_and_the_old_untrained_spellings_build_the_plain_model(arguments):
    assert standin(f"result(inventory(models.resnet18({arguments}))['count'])") == 11_689_512


def test_the_old_positional_spelling_of_pretrained_is_refused_too():
    name, message = standin_error("models.resnet18(True)")
    assert name == "NotImplementedError" and "with pretrained weights is not available" in message


@pytest.mark.parametrize("name", ["ResNet18_Weights.DEFAULT", "resnet50", "vgg16", "resnet.BasicBlock"])
def test_the_other_model_names_say_they_are_not_available(name):
    kind, message = standin_error(f"models.{name}")
    assert kind == "NotImplementedError"
    assert message.startswith(f"torchvision.models.{name.split('.')[0]} is not available in the browser version")



# The same weights give the same answers.

STEPS = """
import numpy as np
data = np.load(os.environ["IN_NPZ"])
model = MODEL
model.load_state_dict({n[len("w:"):]: torch.tensor(data[n]) for n in data.files if n.startswith("w:")})
x = torch.tensor(data["x"])
out = {}

model.eval()
with torch.no_grad():
    out["eval"] = model(x).numpy()

model.train()
y = model(x)
(y * torch.tensor(data["upstream"])).sum().backward()
out["train"] = y.detach().numpy()
for name in NAMES:
    out["grad:" + name] = dict(model.named_parameters())[name].grad.numpy()
for name, value in model.state_dict().items():
    if "running" in name or "num_batches" in name:
        out["state:" + name] = value.numpy()
np.savez(os.environ["OUT_NPZ"], **out)
result("done")
"""

def relatively_close(got, want, what, tolerance=1e-4):
    """The largest difference is at most `tolerance` of the largest value."""
    assert got.shape == want.shape and got.dtype == want.dtype, what
    worst = float(np.abs(got.astype(np.float64) - want.astype(np.float64)).max())
    scale = max(float(np.abs(want).max()), 1e-12)
    assert worst <= tolerance * scale, f"{what}: differs by {worst:.3g} on values up to {scale:.3g}"


def meet(tv, torch, tmp_path, build, standin_model):
    """The real model built by `build(tv.models)` with its norm layers' numbers randomised,
    its state_dict handed to the stand-in's `standin_model`, and the same forward and
    backward run on both. Returns (real outputs, stand-in outputs)."""
    torch.manual_seed(0)
    model = build(tv.models)
    rng = np.random.default_rng(5)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                m.weight.copy_(torch.tensor(rng.uniform(0.6, 1.4, m.weight.shape), dtype=torch.float32))
                m.bias.copy_(torch.tensor(rng.normal(0, 0.1, m.bias.shape), dtype=torch.float32))
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.copy_(torch.tensor(rng.normal(0, 0.1, m.running_mean.shape), dtype=torch.float32))
                m.running_var.copy_(torch.tensor(rng.uniform(0.5, 1.5, m.running_var.shape), dtype=torch.float32))
    x = rng.standard_normal((2, 3, 64, 64)).astype(np.float32)
    classes = model.fc.out_features
    upstream = rng.standard_normal((2, classes)).astype(np.float32)
    state = {"w:" + n: v.numpy() for n, v in model.state_dict().items()}
    np.savez(tmp_path / "in.npz", x=x, upstream=upstream, **state)

    want = {}
    model.eval()
    with torch.no_grad():
        want["eval"] = model(torch.tensor(x)).numpy()
    model.train()
    y = model(torch.tensor(x))
    (y * torch.tensor(upstream)).sum().backward()
    want["train"] = y.detach().numpy()
    parameters = dict(model.named_parameters())
    for name, parameter in parameters.items():
        want["grad:" + name] = parameter.grad.numpy()
    for name, value in model.state_dict().items():
        if "running" in name or "num_batches" in name:
            want["state:" + name] = value.numpy()

    body = STEPS.replace("MODEL", standin_model).replace("NAMES", repr(list(parameters)))
    standin(body, env={"IN_NPZ": str(tmp_path / "in.npz"), "OUT_NPZ": str(tmp_path / "out.npz")})
    got = dict(np.load(tmp_path / "out.npz"))
    assert got.keys() == want.keys()
    return want, got


def test_resnet18_with_torchvisions_weights_gives_torchvisions_outputs_and_gradients(tv, torch, tmp_path):
    want, got = meet(tv, torch, tmp_path, lambda m: m.resnet18(), "models.resnet18()")
    assert np.abs(want["eval"]).max() > 0.1, "an all-zero output would prove nothing"
    relatively_close(got["eval"], want["eval"], "eval output")
    relatively_close(got["train"], want["train"], "training output")
    assert len([k for k in want if k.startswith("grad:")]) == 62  # every parameter of resnet18
    for name in want:
        if name.startswith("grad:"):
            assert np.abs(want[name]).max() > 0, name
            relatively_close(got[name], want[name], name)
        elif name.startswith("state:"):
            if want[name].dtype.kind == "f":
                relatively_close(got[name], want[name], name)
            else:
                assert got[name].tolist() == want[name].tolist() == 1, name


def test_resnet18_with_ten_classes_gives_torchvisions_outputs(tv, torch, tmp_path):
    want, got = meet(tv, torch, tmp_path, lambda m: m.resnet18(num_classes=10), "models.resnet18(num_classes=10)")
    assert want["eval"].shape == (2, 10)
    relatively_close(got["eval"], want["eval"], "eval output")
    relatively_close(got["grad:fc.weight"], want["grad:fc.weight"], "fc.weight")
    relatively_close(got["grad:conv1.weight"], want["grad:conv1.weight"], "conv1.weight")


def test_resnet18_with_group_norm_gives_torchvisions_outputs_and_gradients(tv, torch, tmp_path):
    norm = "lambda channels: torch.nn.GroupNorm(8, channels)"
    want, got = meet(
        tv, torch, tmp_path,
        lambda m: m.resnet18(norm_layer=lambda channels: torch.nn.GroupNorm(8, channels)),
        f"models.resnet18(norm_layer={norm})",
    )
    assert not [k for k in want if k.startswith("state:")], "GroupNorm keeps no running statistics"
    relatively_close(got["eval"], want["eval"], "eval output")
    relatively_close(got["train"], want["train"], "training output")
    for name in want:
        relatively_close(got[name], want[name], name)


def test_the_group_norm_model_has_torchvisions_names_and_shapes(tv, torch):
    want = inventory(tv.models.resnet18(norm_layer=lambda c: torch.nn.GroupNorm(8, c)))
    got = standin("result(inventory(models.resnet18(norm_layer=lambda c: torch.nn.GroupNorm(8, c))))")
    assert got["state"] == want["state"] and got["buffers"] == want["buffers"] == []
    assert got["repr"] == want["repr"]


def test_the_group_norm_model_starts_with_scale_one_and_shift_zero():
    start = standin(
        "model = models.resnet18(norm_layer=lambda c: torch.nn.GroupNorm(8, c), zero_init_residual=True)\n"
        "result({n: [m.weight.tolist(), m.bias.tolist()] for n, m in model.named_modules()"
        " if isinstance(m, torch.nn.GroupNorm)})"
    )
    assert len(start) == 20
    for name, (weight, bias) in start.items():
        assert set(weight) == ({0.0} if name.endswith(".bn2") else {1.0}), name
        assert set(bias) == {0.0}, name


# What the block rejects, as torchvision's does.

@pytest.mark.parametrize(
    "arguments",
    [{"groups": 2}, {"width_per_group": 32}, {"replace_stride_with_dilation": [False, True, False]},
     {"replace_stride_with_dilation": [True]}],
)
def test_basic_blocks_reject_what_torchvisions_reject(tv, arguments):
    with pytest.raises(Exception) as want:
        tv.models.resnet18(**arguments)
    name, message = standin_error(f"models.resnet18(**{arguments!r})")
    assert name == type(want.value).__name__ and message == str(want.value)


# What a notebook does with the model: count parameters per stage, hook the stages, freeze the
# backbone, swap the head and train it.

TRANSFER = """
torch.manual_seed(0)
model = models.resnet18()
counts = {n: sum(p.numel() for p in m.parameters()) for n, m in model.named_children()}
shapes = {}
handles = [getattr(model, n).register_forward_hook(lambda m, i, o, n=n: shapes.update({n: list(o.shape)}))
           for n in ("layer1", "layer2", "layer3", "layer4")]
x = torch.randn(2, 3, 64, 64)
model.eval()
with torch.no_grad():
    model(x)
for handle in handles:
    handle.remove()

for p in model.parameters():
    p.requires_grad_(False)
model.fc = torch.nn.Linear(512, 10)
before = {n: p.detach().clone() for n, p in model.named_parameters()}
buffers = {n: b.clone() for n, b in model.named_buffers()}
optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([3, 7]))
optimizer.zero_grad()
loss.backward()
optimizer.step()
moved = [n for n, p in model.named_parameters() if not torch.equal(p, before[n])]
kept = all(torch.equal(b, buffers[n]) for n, b in model.named_buffers())
result({"counts": counts, "shapes": shapes, "moved": moved, "kept": kept,
        "trainable": [n for n, p in model.named_parameters() if p.requires_grad]})
"""


def test_a_notebook_can_audit_hook_freeze_and_retrain_the_head():
    out = standin(TRANSFER)
    assert out["counts"] == {
        "conv1": 9_408, "bn1": 128, "relu": 0, "maxpool": 0, "layer1": 147_968, "layer2": 525_568,
        "layer3": 2_099_712, "layer4": 8_393_728, "avgpool": 0, "fc": 513_000,
    }
    assert out["shapes"] == {
        "layer1": [2, 64, 16, 16], "layer2": [2, 128, 8, 8], "layer3": [2, 256, 4, 4], "layer4": [2, 512, 2, 2],
    }
    assert out["trainable"] == ["fc.weight", "fc.bias"]
    assert out["moved"] == ["fc.weight", "fc.bias"]
    assert out["kept"]


# How long a step takes.

TIMED = """
import time
torch.manual_seed(0)
model = models.resnet18()
x = torch.randn(8, 3, 64, 64)
y = torch.randint(0, 1000, (8,))
best_forward = best_step = float("inf")
for _ in range(3):
    start = time.perf_counter()
    loss = torch.nn.functional.cross_entropy(model(x), y)
    forward = time.perf_counter() - start
    model.zero_grad()
    loss.backward()
    best_forward = min(best_forward, forward)
    best_step = min(best_step, time.perf_counter() - start)
assert all(p.grad is not None for p in model.parameters())
result([best_forward, best_step])
"""


def test_a_training_step_on_eight_64_pixel_images_takes_seconds_not_minutes():
    forward, step = standin(TIMED)
    assert step < 20, f"forward {forward:.1f} s, forward and backward {step:.1f} s"
