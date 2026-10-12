# Known differences between the stand-in and real PyTorch in our notebooks

Every `Part N passed` line a solution notebook prints under real PyTorch should also print under
the stand-in, and the exceptions are listed here. `run_notebooks.py --real` and
`run_notebooks.py --standin` produce the two runs; each entry below names the notebook, the part,
why it differs, and what the check should test instead.

None of these is a bug in `src/`. They fall into two groups:

- **The browser's MNIST is a subset.** The page ships the first 10,000 training and 2,000 test
  images (`public/pyodide/datasets/mnist.npz`), because a reader cannot download the real thing.
  Nine of the twelve failing checks below (in four notebooks) fail because of it: they hold a
  number measured on the full 60,000 training images, or the notebook hard-codes the full set's
  sizes, so they fail on the subset whether the torch is real or not. The evidence for each is
  given as real PyTorch's own answer on the smaller set.
- **The draw differs.** `torch.manual_seed` makes the stand-in reproducible, but its numbers are
  not real PyTorch's. Where a check's outcome turns on one particular draw, the stand-in can land
  on the other side of it. The remaining three checks (in two notebooks) say which draw matters
  and how often it goes each way.

Counted by check cell, there are twelve: nine from the subset and three from the draw, over six
notebooks (mlp-activations is in the comparison twice, once from this worktree and once from
`wip/notebook-organization-by-topic`, which is why 18 of the 25 runs pass and 7 do not). Part 1
of training-loop-pytorch is the only check there that fails in its own right; its Parts 2 to 4
never run, because the subset makes Part 2 raise.

## Re-checked on 2026-10-05

Every entry was checked again against both libraries, and nothing here moved. The two tests were
run for each notebook:

- **Real PyTorch on the subset.** The same notebook was run on real torch with
  `datasets.MNIST` replaced by the first 10,000 training and 2,000 test images, which is exactly
  what `make_mnist.py` ships. Every sample-size check below fails there too, with real PyTorch's
  own numbers, so the constant in the notebook is right for Colab and the sample size is what
  breaks it.
- **Real PyTorch on the stand-in's draw.** The stand-in's epoch orders, its seed-0 weights, or
  its noise were loaded into real PyTorch and the notebook's own cells re-run. Each of the three
  draw checks fails there as well, and passes on real PyTorch's own draw.

`scripts/validate-notebooks.py` and the notebook pages do not read this file; it is for whoever
rewrites these lessons (the lesson rewrite in piece 4 of the plan). Until a check is rewritten,
the notebooks that name a number here are the ones to be careful about in the browser.

## The browser's MNIST is 10,000 images, not 60,000

### multilayer-perceptron/mlp-activations

| Part | Check | Stand-in | What happens |
|---|---|---|---|
| 2 | `all(h[-1] > 0.9 for h in results.values())`, "all three should pass 0.9 by epoch 5" | sigmoid 0.6485, tanh 0.8775, relu 0.8475 after 5 epochs | Real torch on 10,000 images gives the same answer: 0.6965, 0.8865, 0.8935. On 60,000 it gives 0.9256, 0.9450, 0.9515. |
| 4 | `deep["tanh"][0][-1] > 0.85 and deep["relu"][0][-1] > 0.85`, "the deep tanh and ReLU networks should train" | tanh 0.4575, relu 0.1170 (relu never leaves 0.1170) | Real torch on 10,000 images gives tanh 0.6960 and relu 0.2075. On 60,000, 0.9244 and 0.9247. |

Part 3 passes on the subset, and so does the sigmoid half of Part 4 (the eight-layer sigmoid
network stays stuck near chance either way, which is the point of the part).

The deep ReLU network's 0.1170 is the draw on top of the subset, not the stand-in's arithmetic.
With the seed-0 weights and the five epoch orders swapped between the libraries, each reproduces
the other's five accuracies exactly: real torch on the stand-in's draw stays at 0.1170, and the
stand-in on real torch's draw reaches 0.2075. (Both directions were re-measured on 2026-10-05:
the Part 2 accuracies and the whole deep ReLU series come out the same to four places in each
direction. The deep *tanh* series does not, in either direction, because it is on a plateau
where float32 rounding decides the epoch; it is a draw too, and it fails the check either way.)

Note that the weights have to be swapped as well as the orders. The two libraries draw from
different generators, so `torch.manual_seed(0)` gives different weights in each, and swapping
the epoch orders alone leaves the two runs training different networks.

**What the check should test instead.** The lesson is about the *shape* of the curves, not the
accuracy they reach: five epochs is enough to see sigmoid trailing, and to see relu and tanh
climbing, but not enough to name a number. Assert the ordering and the shape:

```python
assert results["sigmoid"][-1] < min(results["tanh"][-1], results["relu"][-1]), "..."
assert all(h[-1] > h[0] for h in results.values()), "every activation improves over five epochs"
```

Not the order of tanh and ReLU: real torch puts ReLU ahead on the subset (0.8935 against 0.8865)
and the stand-in puts tanh ahead (0.8775 against 0.8475). And not the deep ReLU network against
the deep sigmoid one after five epochs on the subset: with the notebook's default
initialisation it has not yet left the plateau at 0.1170 on the stand-in, so the two can tie.

Better still: train for as many epochs as the subset needs to reach the bar (Part 4 says He
initialisation trains much faster in the first epoch, so the comparison to make is
`he_history[0] > deep["relu"][0][0]`, which holds on 10,000 images too). And note in the prose
that the page trains on 10,000 of the 60,000 images, so the accuracies are a little below the
ones in Colab.

### deep-learning/mlp-mnist

| Part | Check | Stand-in | What happens |
|---|---|---|---|
| 2 | `history[-1][2] > 0.93`, "expected about 0.95 after 10 epochs" | 0.8955 | Real torch on 10,000 images reaches 0.8905 in 10 epochs, and 0.9476 on 60,000. |
| 3 | `M.sum().item() == 10000` inside "the diagonal holds the correct answers" | 2,000 | The confusion matrix has one row entry per test image, and the browser's test set is 2,000 images. |
| 4 | `sweep[0.5][0] > 0.9`, "0.5 should pass 0.9 within 3 epochs" | 0.858 | Real torch on 10,000 images gives 0.8440 after 3 epochs, and 0.9330 on 60,000. The lr 0.001 and lr 20 halves of the same check pass either way. |

**What the check should test instead.** Part 2: the point of the part is that a 10-epoch run at
lr 0.5 learns and the loss falls, so compare against what the same run reaches on the subset
rather than a fixed number, or train for the number of epochs the subset needs. Part 3: drop the
hard-coded total, which is the only thing that breaks, and keep
`M.trace().item() == (pred == y_test).sum().item()`, which is the actual claim. Part 4: compare
the three learning rates against each other (lr 0.5 much better than lr 0.001, which stays near
0.1) instead of against 0.9.

### deep-learning/training-loop-pytorch

| Part | Check | Stand-in | What happens |
|---|---|---|---|
| 1 | `len(full_train) == 60000 and len(test_set) == 10000`, "__len__ is the number of examples" | 10,000 and 2,000 | The notebook asks for the full MNIST and splits it into 50,000 train and 10,000 validation. |
| 2, 3, 4 | none of their own: they never get to run | not reached | The subset is a sixth of the full set, so `split_indices(60000, 10000, seed=0)` returns indices past the end of the array and Part 2 raises `IndexError: index 23840 is out of bounds for axis 0 with size 10000`. Parts 3 and 4 then fail on names the earlier cell never set. |

Everything else in this notebook works on the subset: `Dataset`, `Subset`, `DataLoader`, the
batch shapes, the split and the checkpointing all behave, at a sixth of the scale. The
diagnostic that ran the notebook's own split scaled to the subset reported 8,334 train, 1,666
validation, 131 batches of (64, 784) int64 labels, with the two splits disjoint.

**What the check should test instead.** Nothing about the stand-in. This notebook needs the full
MNIST, so it should read its sizes from the dataset rather than hard-code them, or the page
should say "Colab recommended" for it. If it reads its sizes from the dataset, every part below
Part 1 works on the browser's subset as they stand.

### modern-ai/fine-tune-linear-probe-pytorch

| Part | Check | Stand-in | What happens |
|---|---|---|---|
| 1 | `src_acc > 0.97`, "one epoch should give well over 0.97 on digits 0-4" | 0.9648 on 5,131 pre-training images | Real torch gives 0.9910 on 30,596 pre-training images and about 0.96 on the 5,131 the subset has (0.9650 and 0.9610 in two runs). |
| 2 | `table[("conv", 5)] > table[("pixels", 5)] + 0.05`, "pre-trained conv features should clearly beat pixels at 5 labels per class" | 0.752 against 0.703, a gap of 0.049 | The conv features are one epoch of training on a sixth of the data, so they are a little weaker than in Colab. The gap is 0.049 rather than 0.05. Real torch on the subset clears this one (0.692 against 0.634, a gap of 0.058) and fails the *next* assertion in the same cell instead, so read the row with the next one. |
| 3 | `ft_table[("fine-tune", 5)] > ft_table[("scratch", 5)] + 0.02`, "pre-trained weights should beat a random start" | 0.743 against 0.761 at k = 5 | Same cause: the starting point is weaker, so fine-tuning starts no better than a random one at five labels per class. At k = 20 the two tie on the subset (0.857 against 0.858), where Colab gives 0.918 against 0.885. |

Parts 2 and 3 are not independent of Part 1: both compare pre-trained features against something
else, and the pre-training is what the smaller dataset weakens.

Part 2 is worth reading carefully, because the stand-in and real torch on the subset stop at
*different* assertions in that cell, and a reader looking for one cause will not find it. The
stand-in's conv features beat the pixels by 0.049 and lose that assertion; real torch's beat the
pixels by 0.058, pass it, and then lose the next one, `both("conv") > both("random conv")`
(1.542 against 1.575). All three assertions in the cell pass on the full 60,000, where real torch
gives conv 0.838 and 0.926 against random conv 0.803 and 0.899. So the cell fails on the subset in
both libraries, and which assertion gives way is decided by the draw on top of the weaker
pre-training.

**What the check should test instead.** Pretrain for as many epochs as the subset needs to clear
the Part 1 bar (two epochs on 5,131 images give 0.9838 on real torch), then Parts 2 and 3
follow. Or check the gaps relative to the part's own control rather than by a fixed margin: the
claim is that pre-trained features beat pixels and beat random features, not by exactly 0.05.

## The draw differs

### deep-learning/optimizer-race-digits, Part 4

The check is `w[0] < c[0] / 1.5`, "warmup should take the spike out of the first steps", where
`w[0]` and `c[0]` are the largest mini-batch loss in the first 30 steps of the warmup and the
constant run. Both runs train SGD with momentum at learning rate 1.0, which is at the edge of
stability, so how violently the constant run diverges depends on the draw.

The arithmetic is not the problem. Given the same weights and the same gradients, one step of
`torch.optim.SGD` with momentum 0.9 at lr 1.0 agrees between the two libraries to the last bit
(`0.00e+00` over 1, 5 and 20 steps), as does Adam's first step at lr 0.01. What differs is the
batch order: the notebook shuffles with `torch.randperm(len(Xtr), generator=g)`, and the
stand-in's permutation for seed 1 starts `[755, 731, 1083, 693, ...]` where real torch's is
`[1101, 492, 881, 699, ...]`. Running real torch's own code on the stand-in's permutation gives
a ratio of 0.870, which fails the check the same way; on its own permutation it gives 0.640,
which passes.

Re-measured on 2026-10-05 over all 25 epoch orders rather than the first, the two ratios are 0.822
and 0.513. Same verdict either side of the 0.667 the check asks for. This notebook uses sklearn's
digits, not MNIST, so the subset is nowhere near it: it passes all four parts on real torch with
MNIST cut to the browser's sizes.

**What the check should test instead.** The part is about what the schedule does, so compare the
schedules on what the notebook already has that is stable: that the decayed run ends at a lower
training loss than the constant one (that check passes), that the constant run's *final* loss is
worse, and that the learning rate curve from `lr_at` matches the worked example (that check
passes). If the early-loss comparison is worth keeping, average the worst loss over several seeds
rather than asserting it for one.

### modern-ai/tiny-gan-pytorch, Part 3

The check is `trace[0][2] / 3 < trace[0][1] < 3 * trace[0][2]`, "at the start, with D near 0.5,
the two gradients should be about the same size". It holds exactly when the discriminator starts
near 0.5 on the generator's fakes, because that is where the saturating and non-saturating losses
have equal gradient magnitude. The stand-in's `torch.manual_seed(0)` gives a discriminator that
opens at 0.751 on the fakes instead of 0.528, and the non-saturating gradient comes out at 0.325
of the saturating one. Real torch given the stand-in's seed-0 weights and noise gives the same
0.751 and the same two gradients to within 1.2e-07, and fails the check the same way.

Re-measured on 2026-10-05, serving both libraries from one fixed pool of noise: the stand-in gives
D = 0.751, gradients 1.1072 and 0.3628, ratio 0.328; real torch on that same pool gives D = 0.751,
gradients 1.1072 and 0.3628, ratio 0.328, identical to four places, and fails. Real torch on its
own pool gives D = 0.527, gradients 0.0816 and 0.0758, ratio 0.930, and passes. The three other
Part 3 checks pass in all three runs.

This check was expected to differ. The stand-in's draw is not a bad one: over
seeds 0 to 7 the ratio is within the factor of 3 for seven of the eight, and only seed 0 (the
one the notebook uses) falls outside it. Real torch's seeds 0 to 7 all pass.

**What the check should test instead.** The reason the two gradients match is a fact about the
losses, not about the draw: per logit, the saturating gradient is `-D` and the non-saturating one
`-(1 - D)`, so they are equal exactly where D is 0.5. Test that with the notebook's own
`logit_gradients` from Part 1, and print D's opening value as an observation rather than
asserting on it:

```python
g_sat, g_ns = logit_gradients(torch.zeros(BATCH))
assert torch.allclose(g_sat, g_ns), "at D = 0.5 the two losses push the generator equally hard"
```

The rest of Part 3 (the discriminator learning to reject the fakes, the saturating gradient
shrinking, the non-saturating one becoming hundreds of times larger) passes on the stand-in.

### modern-ai/tiny-gan-pytorch, Part 4

The check is `plain_left < 0.1 or plain_left > 0.9` for seeds 0, 1 and 2, "the plain GAN should
collapse onto one mode". On the stand-in, seeds 0 and 1 collapse (share 0.00 and 1.00) and seed
2 lands at 0.59, covering both modes.

The check is a statement about a draw, not about the mechanism. Sweeping seeds 0 to 11 with the
notebook's own `train_two_modes(seed, False)`, 5 of 12 collapse on the stand-in and 10 of 12 on
real torch; real torch fails the same assertion on seeds 6 and 7. Which mode it collapses onto is
the draw too: real torch's seeds 0 and 3 collapse to +2 and seeds 1, 2, 4 and 5 to -2.

Re-measured on 2026-10-05 by loading the stand-in's seed-0 to seed-2 weights into real torch and
running the notebook's own `train_two_modes`: the stand-in's shares are 0.00, 1.00 and 0.56, and
real torch's on those same weights are 0.54, 0.60 and 0.00. Both fail, at different seeds, and
both are one short of three. Real torch on its own weights gives 1.00, 0.00 and 0.00, which
passes. The shares are not the same numbers on the same weights, which is itself the point: near
the decision boundary a seed decides the outcome, and that is what the check is asserting on.

**What the check should test instead.** The part's claim is that the plain discriminator does not
stop the generator from collapsing while the minibatch-standard-deviation one does, so count how
many of several seeds collapse under each kind and compare the two counts, rather than requiring
all three plain runs to collapse. The `minibatch std` half passes on the stand-in as it stands
(0.50, 0.51, 0.52 against the 0.3 to 0.7 the check wants, and the $W_1$ ratios hold).

## What was checked and is not a difference

- `multilayer-perceptron/mlp-activations` Part 1 (the activation slopes against autograd) and Part 3
  (the gradient profiles) pass exactly on the stand-in: the largest slope, the ratio of first to
  last layer and the vanishing thresholds all hold.
- `modern-ai/tiny-gan-pytorch` Parts 1 and 2 pass, including the saturating and non-saturating
  sign table to four decimals and the learned mean and standard deviation of a one-dimensional
  GAN.
- The optimisers themselves agree. The optimizer-race notebook's Part 1 check, which compares its
  own `MySGD` and `MyAdam` against `torch.optim` over 20 steps, passes on the stand-in with
  differences of 0.0 to 2.2e-08.
- `deep-learning/training-loop-pytorch` Part 1's other four assertions (item shape, label
  provenance, split disjointness, batch shapes) pass on the subset; only the two hard-coded sizes
  fail.
- `torchvision.transforms.functional.to_tensor` agrees with real torchvision on the type of `pic`,
  value for value. Every one of nineteen inputs was run through both: a list, tuple, str, int,
  None, bytes, a tensor, a `uint8` (C, H, W) tensor, a NumPy scalar, a 0-d, 1-d, 2-d, 3-d and 4-d
  array, a float64 array, a masked array, a list of lists and an empty array gives the same
  exception kind and the same message in both libraries, except that the stand-in prints its
  tensor class as `torch.Tensor`, which is what real torchvision prints.
- `torchvision.io.read_image` gives the damaged-PNG and not-an-image errors added in `be5e182f`,
  and now also checks its argument: anything that is not a `str` or an `os.PathLike` is a
  `TypeError` naming `torchvision.io.read_image` and pointing at `ToTensor`, instead of pathlib's
  own `argument should be a str or an os.PathLike object ... not 'ndarray'`. The check is on the
  type only, so a directory, a missing file and an empty file still give the errors they gave.

## How long each notebook takes on the stand-in

Seconds for the whole notebook under `run_notebooks.py --standin` in CPython with
`OMP_NUM_THREADS=2` (2026-10-05). The browser runs the same NumPy work several times slower, so a
notebook over 30 seconds here is one to mark "Colab recommended". The MNIST notebooks train on the
10,000-image subset, so they are quicker here than on real PyTorch with the full set, and
training-loop stops at its first error.

| Notebook | Seconds | Note |
|---|---|---|
| modern-ai/fine-tune-linear-probe-pytorch | 105.4 | over 30 s: Colab recommended |
| modern-ai/tiny-unet-pytorch | 65.9 | over 30 s: Colab recommended |
| modern-ai/tiny-gan-pytorch | 34.4 | over 30 s: Colab recommended (26.0 in an earlier run) |
| modern-ai/transformer-block-pytorch | 15.6 | |
| multilayer-perceptron/mlp-activations | 11.7 | |
| modern-ai/tiny-diffusion-pytorch | 9.1 | |
| convolutional-neural-networks/cnn-train-pytorch | 5.6 | |
| deep-learning/mlp-mnist | 4.4 | |
| multilayer-perceptron/mlp-train-pytorch | 4.2 | |
| deep-learning/optimizer-race-digits | 3.9 | |
| modern-ai/tiny-vae-pytorch | 3.2 | |
| deep-learning/tensor-autograd-drills | 2.0 | |
| modern-ai/resnet-numel-audit | 1.1 | |
| modern-ai/linear-attention-lab | 0.7 | |
| practice/mock-mlp-backprop-pytorch | 0.5 | |
| deep-learning/training-loop-pytorch | 0.3 | stops at Part 1 on the subset |
| practice/finetune-linear-head | 0.1 | |
