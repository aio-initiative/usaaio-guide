# PyTorch stand-in

A NumPy-backed `torch` and `torchvision` for the browser's Python. Real PyTorch is installed here only to test against.

    python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt   # once, from this folder
    .venv/bin/pytest -q                                                      # tests/test_surface.py fails for each name not written yet
    .venv/bin/python surface.py --other <folder of more notebooks>        # rewrite SURFACE.txt from the notebooks

`SURFACE.txt` lists every torch name and `Tensor` attribute the notebooks use; the stand-in must resolve all of them.

The browser build and the checks on it:

    .venv/bin/python build.py                    # write public/pyodide/torch-standin.zip (--check: is it current?)
    .venv/bin/python run_notebooks.py --real     # run the torch solution notebooks on real PyTorch
    .venv/bin/python run_notebooks.py --standin  # the same on the stand-in; KNOWN_DIFFERENCES.md lists every gap

Rebuild the zip and commit it with any change under `src/`; a test fails while it is stale.

## In this repository

The stand-in and these tools were copied from the USAAIO Guide (its `tools/torch-standin/`), with
the paths moved to `public/pyodide/`. The browser loads `public/pyodide/torch-standin.zip` and
`public/pyodide/datasets/mnist.npz` (see `src/notebook-worker.js`). Names added here, each checked
against real PyTorch by `check_additions.py`:

- `torch.isclose` and `Tensor.isclose` (src/torch/_compare.py)
- `torch.set_printoptions` with precision, threshold, edgeitems, linewidth, profile and sci_mode
  (src/torch/_repr.py)

After a change under `src/`:

    ~/miniforge3/envs/Main/bin/python scripts/torch-standin/check_additions.py   # parity of the additions
    ~/miniforge3/envs/Main/bin/python scripts/torch-standin/build.py             # rebuild the zip
    ~/miniforge3/envs/Main/bin/python scripts/torch-standin/run_notebooks.py --standin <solution notebook>

The editor's Node smoke test was left out: it read the USAAIO Guide's TypeScript.
