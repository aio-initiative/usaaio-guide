"""Whole models against real torch: a small CNN trained for 20 steps of SGD from the same
weights gives the same loss curve, and a ResNet basic block (in-place ReLU after BatchNorm,
a residual +=) gives the same outputs and gradients."""

import numpy as np

from nn_parity import layer_agrees, twin

RNG = np.random.default_rng(41)


def small_cnn(nn):
    return nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(4 * 4 * 4, 3)
    )


def test_a_small_cnn_trains_like_torch(st, real):
    images = RNG.standard_normal((16, 1, 8, 8)).astype(np.float32)
    labels = RNG.integers(0, 3, 16)
    s, r = twin(st, real, small_cnn, double=False)
    curves, weights = [], []
    for net, t in ((s, st), (r, real)):
        x, y = t.tensor(images), t.tensor(labels)
        losses = []
        for _ in range(20):
            loss = t.nn.functional.cross_entropy(net(x), y)
            net.zero_grad()
            loss.backward()
            with t.no_grad():
                for p in net.parameters():
                    p -= 0.2 * p.grad
            losses.append(loss.item())
        curves.append(losses)
        weights.append([p.detach().numpy().copy() for p in net.parameters()])
    np.testing.assert_allclose(curves[0], curves[1], rtol=0, atol=1e-4)
    assert curves[0][-1] < 0.5 * curves[0][0], "the loss should fall"
    for got, want in zip(*weights):
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


def make_block(nn):
    """torchvision's BasicBlock with a downsampling shortcut, as resnet18 builds it."""

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))

        def forward(self, x):
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.downsample(identity)
            return self.relu(out)

    return nn.Sequential(
        nn.Conv2d(3, 4, 7, 2, 3, bias=False), nn.BatchNorm2d(4), nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1), BasicBlock(4, 8, stride=2),
        nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(8, 5),
    )


def test_a_resnet_basic_block_matches_torch(st, real):
    x = RNG.standard_normal((2, 3, 20, 18))
    for train in (True, False):
        layer_agrees(st, real, make_block, x, train=train)
    s, r = make_block(st.nn), make_block(real.nn)
    assert repr(s) == repr(r)
    assert sum(p.numel() for p in s.parameters()) == sum(p.numel() for p in r.parameters())
