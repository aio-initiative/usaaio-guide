"""Redrawn figure for the training-loop module (training-loop-in-pytorch).

Copy of site B's `training-curves` panel from `tools/diagrams/
gen_training_figures.py`, drawn through A's theme, with the curves re-measured
for the network this lesson actually trains.

B generated its curve data with `tools/diagrams/training_curves_data.py`, which
trains `nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))`. The
lesson's only model class is 784-128-10, though, and its task paragraph sets
this overfitting demo up as a 784-128-10 network trained on 2,000 images, so
the shipped figure disagreed with the code a reader is handed. The arrays below
are the training and validation loss of the lesson's own network under the
lesson's own settings: MNIST flattened and scaled to [0, 1],
`torch.manual_seed(0)` then `perm = torch.randperm(60000)` with `perm[:2000]`
for training and `perm[50000:]` for validation, `MLP` with `nn.Linear(784, 128)`
ReLU `nn.Linear(128, 10)` built after a second `torch.manual_seed(0)`,
`torch.optim.Adam(model.parameters(), lr=1e-3)`, batches of 64, 40 epochs. The
validation loss reaches its minimum, 0.3355, at epoch 13, so the marker, the
annotation and the lesson's text all say 13.

Nothing else about the panel moved: same two series, same dash ladder, same
marker at the best epoch with its label, same axis labels, same legend, same
canvas. The y axis runs to 1.0 instead of B's 0.85, because this network starts
at a validation loss of 0.978 and B's limit would clip the first epoch.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from b_theme import P, clean, label, render, run_all  # noqa: E402

# Measured with the script and settings described in the module docstring.
TRAIN_LOSS = [
    0.9358, 0.5236, 0.3913, 0.3245, 0.2806, 0.2469, 0.2172, 0.2057, 0.1753, 0.1703,
    0.1430, 0.1236, 0.1098, 0.1022, 0.0860, 0.0799, 0.0688, 0.0615, 0.0574, 0.0513,
    0.0494, 0.0395, 0.0364, 0.0333, 0.0309, 0.0279, 0.0246, 0.0225, 0.0216, 0.0190,
    0.0177, 0.0162, 0.0148, 0.0139, 0.0130, 0.0121, 0.0113, 0.0105, 0.0099, 0.0092,
]
VAL_LOSS = [
    0.9781, 0.5767, 0.4663, 0.4182, 0.3934, 0.3828, 0.3635, 0.3603, 0.3527, 0.3723,
    0.3474, 0.3414, 0.3355, 0.3471, 0.3363, 0.3423, 0.3387, 0.3431, 0.3398, 0.3399,
    0.3522, 0.3476, 0.3453, 0.3491, 0.3508, 0.3518, 0.3526, 0.3594, 0.3584, 0.3610,
    0.3616, 0.3623, 0.3667, 0.3706, 0.3757, 0.3706, 0.3740, 0.3790, 0.3799, 0.3817,
]


@render("training-curves", size="wide")
def training_curves(ax):
    tr, va = np.array(TRAIN_LOSS), np.array(VAL_LOSS)
    epochs = np.arange(1, len(tr) + 1)
    best = int(np.argmin(va)) + 1
    ax.plot(epochs, tr, **P.trace(1, label="training loss"))
    ax.plot(epochs, va, **P.trace(0, label="validation loss"))
    ax.scatter([best], [va[best - 1]], s=42, color=P.signal, zorder=5)
    label(ax, best + 0.8, va[best - 1] - 0.03, f"lowest at epoch {best}", rank="tick",
          ha="left", va="top")
    ax.set_xlim(0, len(tr) + 1)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    clean(ax, spines="lb")
    leg = ax.legend(frameon=False, loc="upper right")
    for t in leg.get_texts():
        t.set_color(P.body)


if __name__ == "__main__":
    run_all(globals())
