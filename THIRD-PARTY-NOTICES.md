# Third-party notices

This project ships or loads third-party code, fonts, datasets and images. Each is listed below
with its licence, its source, and the attribution its licence requires. Only facts verified against
the files in this repository and against the sources named here are stated; where a licence could
not be pinned down, that is said plainly rather than guessed.

Verbatim licence texts are reproduced where a licence requires it. Otherwise the licence is named
and linked, which the permissive licences permit.

## What the browser loads at runtime

### KaTeX 0.16.21

Loaded from cdn.jsdelivr.net on four pages: the editor, the glossary, the module reader and the
theory pages. Version-exact.

- CSS: `https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.css`
- JS: `https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.js`
- Licence: MIT. Source: https://github.com/KaTeX/KaTeX , releases tagged `v0.16.21`.
- Required attribution: the above copyright notice and permission notice, which the licence's
  text requires in "all copies or substantial portions of the Software". KaTeX's own README carries
  them; see https://github.com/KaTeX/KaTeX/blob/main/LICENSE .
- Font files resolve under `https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/fonts/` and are
  served under the same MIT terms.

```
KaTeX is licensed under the MIT license.

Copyright (c) 2013-2020 Khan Academy and the KaTeX contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Pyodide 0.29.4

Loaded inside the notebook Web Worker from cdn.jsdelivr.net. Version-exact.

- Index URL: `https://cdn.jsdelivr.net/pyodide/v0.29.4/full/`
- Licence: Apache-2.0. Source: https://github.com/pyodide/pyodide , releases tagged `0.29.4`.
- Terms: https://www.apache.org/licenses/LICENSE-2.0 . Apache-2.0 requires preserving the
  copyright and licence notices and the NOTICE file; Pyodide's are at
  https://github.com/pyodide/pyodide/blob/main/LICENSE and
  https://github.com/pyodide/pyodide/blob/main/NOTICE .

### NumPy

Bundled inside the Pyodide distribution above, not loaded separately.

- Licence: BSD 3-Clause. Source: https://github.com/numpy/numpy .
- Terms: https://github.com/numpy/numpy/blob/main/LICENSE.txt .

### Supabase JavaScript client, v2

Loaded from cdn.jsdelivr.net and injected at runtime by the client code. The URL pins only the
major version (`@supabase/supabase-js@2`), which this notice records as an open supply-chain item;
pinning to an exact version is listed as a platform fix in PRODUCT-AUDIT.md.

- URL: `https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2`
- Licence: MIT. Source: https://github.com/supabase/supabase , package `@supabase/supabase-js`.
- Terms: https://github.com/supabase/supabase/blob/master/packages/supabase-js/LICENSE .

### Google Fonts: Inter, Outfit, JetBrains Mono

Loaded from Google's CSS on every page. Three families, requested with the `css2` API.

- CSS: `https://fonts.googleapis.com/css2?family=Inter:wght@400..700&family=Outfit:wght@100..900&family=JetBrains+Mono:wght@400;600&display=swap`
- Preconnects: `https://fonts.googleapis.com` and `https://fonts.gstatic.com`.
- Inter, by Rasmus Andersson. Licence: SIL Open Font License 1.1.
  Source: https://github.com/rsms/inter .
- Outfit, by Rodrigo Fuenzalida and Pablo Impallari. Licence: SIL Open Font License 1.1.
  Source: https://github.com/impallari/Outfit .
- JetBrains Mono, by Philipp Nurullin. Licence: SIL Open Font License 1.1.
  Source: https://www.jetbrains.com/lp/mono/ .
- Terms of the OFL 1.1: https://scripts.sil.org/OFL . The OFL requires the licence text to travel
  with the fonts; it is linked from here and should also be linked from any self-hosted copy.
- Google's own terms for the Fonts API are at https://developers.google.com/fonts/faq and
  https://policies.google.com/privacy . Loading the CSS sends the visitor's IP address and page URL
  to Google; this is disclosed in PRODUCT-NOTES.md.

### jsDelivr

The CDN serving KaTeX, Pyodide and the Supabase client above.

- Terms: https://cdn.jsdelivr.com/terms and https://cdn.jsdelivr.com/privacy-policy .

## What ships in this repository

### The NumPy-backed PyTorch and torchvision stand-in

`public/pyodide/torch-standin.zip` ships a stand-in that runs in the browser. It is not PyTorch or
torchvision, but a number of its files were written by following the Python source of those two
projects, line by line in places, and so are derived from them. The stand-in's own notices file at
`scripts/torch-standin/THIRD_PARTY_NOTICES.md` carries the full provenance: the per-file tables
mapping each stand-in file to the PyTorch or torchvision file it follows, the verbatim BSD 3-Clause
texts with their copyright blocks, and the entries below. That file is the authoritative record;
this one points at it.

- Licence: BSD 3-Clause, for the PyTorch and torchvision material.
  Terms: https://github.com/pytorch/pytorch/blob/main/LICENSE and
  https://github.com/pytorch/vision/blob/main/LICENSE .
- Attribution: the copyright notices and the licence text, which BSD 3-Clause requires in binary
  distributions, are reproduced in full in `scripts/torch-standin/THIRD_PARTY_NOTICES.md`.
- Provenance discrepancy to resolve: that file states it was compared against torch 2.14.1 and
  torchvision 0.29.1, while the real Python environment used to validate this project's notebooks
  runs torch 2.10. The ported interfaces and values are compared against one version and validated
  against another; the platform lane should reconcile the two so the notice states the version the
  shipped code was actually checked against.
- The error function in the stand-in's `functional.py` uses the rational approximations and
  coefficients of W. J. Cody, "Rational Chebyshev approximations for the error function",
  Mathematics of Computation 23 (1969), as published in his SPECFUN routine CALERF.
- The stand-in is not endorsed by the PyTorch or torchvision projects.

### MNIST subset

`public/pyodide/datasets/mnist.npz` (1,970,032 bytes) ships a subset of MNIST to the browser,
because a reader cannot download the original.

- Source: Yann LeCun, Corinna Cortes and Christopher J. C. Burges, "MNIST handwritten digit
  database", 1998, http://yann.lecun.com/exdb/mnist/ .
- How the subset was made: `scripts/torch-standin/make_mnist.py` takes the first 10,000 training and
  2,000 test images of torchvision's copy of the dataset and writes them to the npz file, in the
  order torchvision reads them, unmodified.
- Licence: no licence is stated for the dataset. The files that carry the images state none, so this
  notice names where they came from and leaves the terms to the authors to publish. The Hugging Face
  dataset card at https://huggingface.co/datasets/ylecun/mnist shows "License: mit", but that
  designation is the uploader's, not a statement from the authors. What licence applies, and whether
  redistribution under any stated terms is permitted, is an open question for the owner; see
  PRODUCT-AUDIT.md part 1.3.
- The lesson figure `assets/module-media/mnist-digits.png` displays digits from this subset.

### Wikimedia Commons images

Eight images in `assets/module-media/web/`, credited in `content/media-credits.json` and attributed
inline in the lesson that embeds each one. All eight are Creative Commons; the licence, its source
link and the author are recorded per image in that file and in the lessons. Creative Commons
licences require attribution with a link to the licence and an indication of whether the material
was changed (https://creativecommons.org/licenses/by/4.0/ and the ShareAlike equivalents), which the
inline credit lines carry.

| Image file | Author | Licence | Commons page | Module |
|---|---|---|---|---|
| cv4-residual.png | Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola | CC BY-SA 4.0 | https://commons.wikimedia.org/wiki/File:ResNet_block.svg | CV4 |
| tf2-families.png | dvgodoy | CC BY 4.0 | https://commons.wikimedia.org/wiki/File:Encoder_self-attention,_block_diagram.png | TF2 |
| tf3-families.png | dvgodoy | CC BY 4.0 | https://commons.wikimedia.org/wiki/File:Decoder_self-attention_with_causal_masking,_detailed_diagram.png | TF3 |
| tf4-families.png | Yuening Jia | CC BY-SA 3.0 | https://commons.wikimedia.org/wiki/File:The-Transformer-model-architecture.png | TF4 |
| rp1-latent.png | Chervinskii | CC BY-SA 4.0 | https://commons.wikimedia.org/wiki/File:Autoencoder_structure.png | RP1 |
| rp3-latent.png | EugenioTL | CC BY-SA 4.0 | https://commons.wikimedia.org/wiki/File:Reparameterized_Variational_Autoencoder.png | RP3 |
| nl2-bottleneck.png | Guillaume Chevalier | CC BY 4.0 | https://commons.wikimedia.org/wiki/File:The_LSTM_cell.png | NL2 |
| cv6-patches.png | Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola | CC BY-SA 4.0 | https://commons.wikimedia.org/wiki/File:Vision_Transformer.svg | CV6 |

Two of these (cv4-residual, cv6-patches) are PNGs in this repository where Commons holds SVGs, so
they were rasterised. Whether anything beyond rasterisation was changed was not checked, so the
credit lines should state the truth per image; that is a content-lane fix in PRODUCT-AUDIT.md part
3. The tf4-families image is a redrawing of the figure from Vaswani et al., "Attention Is All You
Need"; the Commons page credits only Yuening Jia under CC BY-SA 3.0.

### Roboto

Three TTF files in `assets/slide-fonts/`: `Roboto-Regular.ttf`, `Roboto-Bold.ttf` and
`RobotoMono-Regular.ttf`. No page or script in the site loads them (verified by grep), so they ship
in the repository but do not reach a visitor's browser.

- Licence: Apache-2.0. Source: https://github.com/googlefonts/roboto-3-classic and
  https://fonts.google.com/specimen/Roboto , https://fonts.google.com/specimen/Roboto+Mono .
- Terms: https://www.apache.org/licenses/LICENSE-2.0 .

### Supabase backend

The site's backend is a Supabase project at `https://tjrkhyvfrqeuasmgaerg.supabase.co`. Terms and
privacy policy:

- https://supabase.com/terms and https://supabase.com/privacy .

## What this project's own licence is

Not settled. `package.json` declares `"license": "MIT"`, but this repository contains material
ported from a second project whose code is GNU AGPL-3.0 and whose content is
CC BY-NC-SA 4.0, and there is no LICENSE file. That combination needs an owner decision; see
PRODUCT-AUDIT.md part 1.1 and 1.2. Until it is settled, no licence statement here would be
accurate, so none is made beyond recording the facts.
