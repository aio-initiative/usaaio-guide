/*
 * The Python behind one notebook, in a Web Worker (loaded by src/editor.js).
 *
 * One Pyodide interpreter and one global namespace for the worker's whole
 * life, so names defined in one cell are visible in the next, as in Jupyter.
 * Python runs on this thread, so a long cell never freezes the page. Nothing
 * can interrupt it from outside, so Stop ends the worker and the next run
 * starts a new one.
 *
 * Page to worker:  { type: "run", runId, code, assetBase }
 * Worker to page:  { type: "status", status: "loading" | "ready" | "failed", message? }
 *                  { type: "stdout", runId, text }
 *                  { type: "figure", runId, png }       (base64 PNG)
 *                  { type: "done", runId, error, seconds }
 *
 * Ported from the USAAIO Guide's notebook worker. PyTorch does not exist for
 * browser Python, so a cell that imports torch or torchvision first installs a
 * small NumPy-backed stand-in (public/pyodide/torch-standin.zip) and, for
 * torchvision, an MNIST subset (public/pyodide/datasets/mnist.npz).
 */
/* global importScripts, loadPyodide */
"use strict";

const PYODIDE_VERSION = "0.29.4";
const PYODIDE_INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const MNIST_DIR = "/home/pyodide/.standin";

const STANDIN_MISSING = "The PyTorch stand-in is not available yet.";
const STANDIN_OFFLINE = "The PyTorch stand-in could not be downloaded. Check your connection and try again.";

// Printed text is posted at once for the first few lines, then at most once
// per STDOUT_REFILL_MS, so a loop that prints a great deal cannot flood the
// page. Python holds this thread while it runs, so the burst allowance is what
// lets a few prints appear before a long computation rather than after it.
const STDOUT_BURST = 20;
const STDOUT_REFILL_MS = 10;

// Run once per interpreter. matplotlib draws off-screen (Agg), so figures can
// be saved as PNGs after each cell, and plt.show() is harmless. pandas warns
// on every import that PyArrow is missing, which cannot be installed here.
const PRELUDE = `
__name__ = "__main__"
import os as _os
_os.environ["MPLBACKEND"] = "Agg"
del _os
import warnings as _warnings
_warnings.filterwarnings("ignore", message=".*non-interactive.*")
_warnings.filterwarnings("ignore", message="(?s).*[Pp]yarrow", category=DeprecationWarning)
del _warnings

def _usaaio_figures():
    import base64, io, json, sys
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None:
        return "[]"
    images = []
    for num in plt.get_fignums():
        buffer = io.BytesIO()
        plt.figure(num).savefig(buffer, format="png", bbox_inches="tight", dpi=110)
        images.append(base64.b64encode(buffer.getvalue()).decode())
    plt.close("all")
    return json.dumps(images)
`;

const CLOSE_FIGURES = `
import sys
plt = sys.modules.get("matplotlib.pyplot")
if plt is not None:
    plt.close("all")
`;

function post(message) {
  self.postMessage(message);
}

function messageOf(error) {
  return String((error && error.message) || error);
}

// Start loading as soon as the worker starts; runs wait for it.
const loaded = (async () => {
  post({ type: "status", status: "loading" });
  importScripts(`${PYODIDE_INDEX_URL}pyodide.js`);
  if (typeof loadPyodide !== "function") throw new Error("Pyodide loaded but did not expose loadPyodide().");
  const pyodide = await loadPyodide({ indexURL: PYODIDE_INDEX_URL });
  await pyodide.runPythonAsync(PRELUDE);
  post({ type: "status", status: "ready" });
  return pyodide;
})();
loaded.catch((error) => post({ type: "status", status: "failed", message: `Python could not be loaded: ${messageOf(error)}` }));

/* ------------------------------ helper Python ----------------------------- */

// Python for the runner's own bookkeeping, in a throwaway namespace so it
// leaves no names behind in the reader's.
function runInternal(pyodide, code) {
  const globals = pyodide.toPy({});
  try {
    return pyodide.runPython(code, { globals });
  } finally {
    globals.destroy();
  }
}

// The figures the last cell drew, as PNGs, then closed. When they cannot be
// saved they are closed anyway, so the next cell is not stuck with them.
function saveFigures(pyodide) {
  try {
    const images = JSON.parse(String(pyodide.runPython("_usaaio_figures()")));
    return Array.isArray(images) ? images : [];
  } catch (error) {
    try {
      runInternal(pyodide, CLOSE_FIGURES);
    } catch (ignored) {
      /* nothing more can be done */
    }
    return [];
  }
}

/* --------------------------- which stand-in to load ----------------------- */

// The code with comments removed and every string literal collapsed to a
// blank, so `# import torch` or a docstring that mentions it does not count.
function withoutStringsAndComments(code) {
  let out = "";
  let i = 0;
  while (i < code.length) {
    const ch = code[i];
    if (ch === "#") {
      while (i < code.length && code[i] !== "\n") i += 1;
    } else if (ch === '"' || ch === "'") {
      const quote = code.startsWith(ch.repeat(3), i) ? ch.repeat(3) : ch;
      i += quote.length;
      while (i < code.length) {
        if (code.startsWith(quote, i)) {
          i += quote.length;
          break;
        }
        if (code[i] === "\\") {
          i += 2;
          continue;
        }
        if (quote.length === 1 && code[i] === "\n") break;
        if (code[i] === "\n") out += "\n";
        i += 1;
      }
      out += " ";
    } else {
      out += ch;
      i += 1;
    }
  }
  return out;
}

function importedPackages(code) {
  const packages = new Set();
  for (const statement of withoutStringsAndComments(code).split(/[\n;:]/)) {
    const text = statement.trim();
    const plain = text.match(/^import\s+(.+)$/);
    if (plain) {
      for (const item of plain[1].split(",")) {
        const name = item.trim().split(/\s+/)[0];
        if (name) packages.add(name.split(".")[0]);
      }
      continue;
    }
    const from = text.match(/^from\s+([\w.]+)\s+import\b/);
    if (from && !from[1].startsWith(".")) packages.add(from[1].split(".")[0]);
  }
  return packages;
}

function standinNeeded(code) {
  if (!code.includes("torch")) return { torch: false, torchvision: false };
  const packages = importedPackages(code);
  return { torch: packages.has("torch"), torchvision: packages.has("torchvision") };
}

/* ------------------------------- installing ------------------------------- */

async function download(url) {
  let response;
  try {
    response = await fetch(url);
  } catch (error) {
    throw new Error(STANDIN_OFFLINE);
  }
  if (!response.ok) throw new Error(STANDIN_MISSING);
  return response.arrayBuffer();
}

function assetUrl(assetBase, file) {
  return `${assetBase.endsWith("/") ? assetBase : `${assetBase}/`}${file}`;
}

async function installStandin(pyodide, assetBase) {
  const zip = await download(assetUrl(assetBase, "torch-standin.zip"));
  // The stand-in is written on NumPy, which nothing else would load for it.
  await pyodide.loadPackage("numpy");
  const sitePackages = runInternal(pyodide, 'import sys\nnext(p for p in sys.path if p.endswith("site-packages"))');
  pyodide.unpackArchive(zip, "zip", { extractDir: sitePackages });
  runInternal(pyodide, "import importlib\nimportlib.invalidate_caches()");
}

async function installMnist(pyodide, assetBase) {
  const data = await download(assetUrl(assetBase, "datasets/mnist.npz"));
  pyodide.FS.mkdirTree(MNIST_DIR);
  pyodide.FS.writeFile(`${MNIST_DIR}/mnist.npz`, new Uint8Array(data));
}

// Runs an installer once per worker; if it fails, the next call tries again.
function oncePerWorker(install) {
  let installing = null;
  return (pyodide, assetBase) => {
    if (!installing) {
      installing = install(pyodide, assetBase);
      installing.catch(() => {
        installing = null;
      });
    }
    return installing;
  };
}
const ensureStandin = oncePerWorker(installStandin);
const ensureMnist = oncePerWorker(installMnist);

async function prepareCell(pyodide, code, assetBase) {
  const need = standinNeeded(code);
  // One zip holds torch and torchvision, so either import installs both.
  if (need.torch || need.torchvision) await ensureStandin(pyodide, assetBase);
  if (need.torchvision) await ensureMnist(pyodide, assetBase);
  try {
    await pyodide.loadPackagesFromImports(code);
  } catch (error) {
    throw new Error(`Failed to load Python packages: ${messageOf(error)}`);
  }
}

/* --------------------------------- running -------------------------------- */

// A traceback without Pyodide's own frames. The reader's code shows as <exec>.
function cleanTraceback(raw) {
  const lines = raw.replace(/\s+$/, "").split("\n");
  const out = [];
  let skipping = false;
  for (const line of lines) {
    const frame = line.match(/^ {2}File "([^"]+)"/);
    if (frame) {
      skipping = frame[1].includes("/_pyodide/") || frame[1].includes("/pyodide/");
      if (skipping) continue;
    } else if (skipping && /^ {4}/.test(line)) {
      continue;
    } else {
      skipping = false;
    }
    out.push(line);
  }
  return out.join("\n");
}

function stdoutBatcher(runId) {
  let pending = "";
  let tokens = STDOUT_BURST;
  let refilledAt = performance.now();
  let timer = null;
  const flush = () => {
    if (timer !== null) clearTimeout(timer);
    timer = null;
    if (!pending) return;
    post({ type: "stdout", runId, text: pending });
    pending = "";
  };
  return {
    push(line) {
      pending += `${line}\n`;
      const now = performance.now();
      tokens = Math.min(STDOUT_BURST, tokens + (now - refilledAt) / STDOUT_REFILL_MS);
      refilledAt = now;
      if (tokens >= 1) {
        tokens -= 1;
        flush();
      } else if (timer === null) {
        timer = setTimeout(flush, STDOUT_REFILL_MS);
      }
    },
    flush,
  };
}

async function runCell(runId, code, assetBase) {
  let error = null;
  let seconds = 0;
  try {
    const pyodide = await loaded;
    await prepareCell(pyodide, code, assetBase);
    const stdout = stdoutBatcher(runId);
    pyodide.setStdout({ batched: stdout.push });
    pyodide.setStderr({ batched: stdout.push });
    const started = performance.now();
    try {
      const value = await pyodide.runPythonAsync(code);
      // The last expression's value is not shown; free it if Python owns it.
      if (value && typeof value.destroy === "function") value.destroy();
    } catch (raised) {
      error = cleanTraceback(messageOf(raised));
    } finally {
      seconds = (performance.now() - started) / 1000;
      try {
        // print(..., end="") leaves text buffered until Python flushes it.
        runInternal(pyodide, "import sys\nsys.stdout.flush()\nsys.stderr.flush()");
      } catch (ignored) {
        /* Python's streams are gone */
      }
      stdout.flush();
      pyodide.setStdout({});
      pyodide.setStderr({});
    }
    // Figures are sent even when the cell failed: it may have drawn first.
    for (const png of saveFigures(pyodide)) post({ type: "figure", runId, png });
  } catch (failure) {
    error = messageOf(failure);
  }
  post({ type: "done", runId, error, seconds });
}

// One run at a time: a cell can await, and two runs must not share stdout.
let queue = Promise.resolve();
self.onmessage = (event) => {
  const message = event.data || {};
  if (message.type !== "run") return;
  queue = queue.then(() => runCell(message.runId, String(message.code || ""), String(message.assetBase || "")));
};
