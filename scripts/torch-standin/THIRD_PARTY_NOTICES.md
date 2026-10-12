# Third-party notices

This folder holds a NumPy-backed stand-in for PyTorch and torchvision that runs in the browser.
It does not contain PyTorch or torchvision themselves, but a number of its files were written by
following the Python source of those two projects, line by line in places, and so are derived from
it. PyTorch and torchvision are released under the BSD 3-Clause licence; the copyright notices and
licence texts follow, copied unchanged from the packages the tests compare against (torch
2.14.1 and torchvision 0.29.1). This file ships in the browser zip next to `torch/` and
`torchvision/`.

Each ported file starts with a comment that names its origin and points here. The code was
changed to run on NumPy and on the stand-in's own tensor, so it is not a copy of the originals,
and the stand-in is not endorsed by either project.

## Ported files

From PyTorch (paths inside the `torch` package):

| Stand-in file | Follows |
|---|---|
| `src/torch/nn/modules/module.py` | `torch/nn/modules/module.py` |
| `src/torch/nn/modules/container.py` | `torch/nn/modules/container.py` |
| `src/torch/nn/modules/normalization.py` | `torch/nn/modules/batchnorm.py`, `torch/nn/modules/normalization.py` |
| `src/torch/nn/modules/conv.py` | `torch/nn/modules/conv.py` |
| `src/torch/nn/modules/loss.py` | `torch/nn/modules/loss.py` |
| `src/torch/nn/modules/linear.py` | `torch/nn/modules/linear.py` |
| `src/torch/nn/modules/sparse.py` | `torch/nn/modules/sparse.py` |
| `src/torch/nn/modules/activation.py` | `torch/nn/modules/activation.py` |
| `src/torch/nn/modules/attention.py` (in part: `MultiheadAttention`'s constructor, parameters and initialisation, and the order of operations of its forward pass) | `torch/nn/modules/activation.py`, `torch/nn/functional.py` (`multi_head_attention_forward`, `_canonical_mask`) |
| `src/torch/nn/modules/pooling.py` | `torch/nn/modules/pooling.py` |
| `src/torch/nn/modules/upsampling.py` | `torch/nn/modules/upsampling.py` |
| `src/torch/nn/modules/dropout.py` | `torch/nn/modules/dropout.py` |
| `src/torch/nn/modules/flatten.py` | `torch/nn/modules/flatten.py` |
| `src/torch/nn/init.py` | `torch/nn/init.py` |
| `src/torch/nn/functional.py` (in part: the argument checks and error messages of `interpolate`, the loss reductions, `dropout` and the like) | `torch/nn/functional.py` |
| `src/torch/optim/optimizer.py` | `torch/optim/optimizer.py` |
| `src/torch/optim/sgd.py` | `torch/optim/sgd.py` |
| `src/torch/optim/adam.py` | `torch/optim/adam.py` |
| `src/torch/optim/rmsprop.py` | `torch/optim/rmsprop.py` |
| `src/torch/optim/adagrad.py` | `torch/optim/adagrad.py` |
| `src/torch/optim/lr_scheduler.py` | `torch/optim/lr_scheduler.py` |
| `src/torch/utils/data.py` | `torch/utils/data/dataloader.py`, `sampler.py`, `dataset.py`, `_utils/collate.py` |
| `src/torch/_repr.py` (in part: the print options and their defaults, the rules that choose precision or scientific notation, and the summarisation of a large tensor) | `torch/_tensor_str.py` |

From torchvision (paths inside the `torchvision` package):

| Stand-in file | Follows |
|---|---|
| `src/torchvision/models.py` | `torchvision/models/resnet.py` |
| `src/torchvision/transforms/__init__.py` | `torchvision/transforms/transforms.py` (the transform classes, their constructors and their `__repr__`) |
| `src/torchvision/transforms/functional.py` | `torchvision/transforms/functional.py` (the argument checks and error messages of `to_tensor`, `normalize` and `resize`, and the output-size rule) |
| `src/torchvision/datasets.py` | `torchvision/datasets/mnist.py` (the MNIST class: its constructor, its attributes and what `__getitem__` returns) |
| `src/torchvision/io.py` | `torchvision/io/image.py` (`ImageReadMode` and its values, `read_image`, and the channel conversion each mode makes) |

Beyond these files, the stand-in reproduces PyTorch's and torchvision's documented behaviour, and
many of its error messages and docstring phrases are theirs word for word, so that a notebook
sees the same text it would see in PyTorch. Those fragments are covered by the same licences.
Where the follow is of interfaces and values rather than of source, the tables above say which:
resizing follows the kernels of ATen, PyTorch's tensor library, and PNG decoding follows what
libpng does for each colour type, as torchvision's `decode_png` asks of it.

## Other sources

- The MNIST subset that ships with the page is the digits themselves, not code: Yann LeCun,
  Corinna Cortes and Christopher J. C. Burges, "MNIST handwritten digit database", 1998, from
  http://yann.lecun.com/exdb/mnist/. `make_mnist.py` takes the first 10,000 training and 2,000
  test images of real torchvision's copy of it and writes them to
  `public/pyodide/datasets/mnist.npz`, because a reader in a browser cannot download the
  original. The images are redistributed here unmodified, in the order torchvision reads them.
  No licence is stated for the dataset here: the files that carry the images do not state one, so
  this notice names where they came from and leaves the terms to the authors to publish.
- The error function in `src/torch/nn/functional.py` uses the rational approximations and
  coefficients of W. J. Cody, "Rational Chebyshev approximations for the error function",
  Mathematics of Computation 23 (1969), as published in his SPECFUN routine CALERF.

## PyTorch licence

```
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions by Tri Dao:
Copyright (c) 2024 Tri Dao.
All rights reserved.

All contributions by Arm:
Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

## torchvision licence

```
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016, 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
