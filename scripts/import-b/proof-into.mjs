#!/usr/bin/env node
/* Prove scripts/import-b/problems.py --into writes the per-module file
 * correctly, so wave 2 lanes can merge B's entries into a module file that may
 * already hold A's kept entries.
 *
 * 1. reproduce: merging into the module file scripts/split-content.mjs writes
 *    for the MLP module must leave it byte-identical. That module's file holds
 *    9 theory cards this importer reproduces exactly (proof-mlp.mjs), 9 theory
 *    entries, and 4 notebook cards from the notebook converter that the
 *    importer never emits and must not touch.
 * 2. idempotent: running it again must leave the file byte-identical.
 * 3. create: with no file there, --into writes {"module","stage","index",
 *    "theory"} and running it again is byte-identical.
 * 4. append: entries the file already holds are replaced in place; ids the file
 *    lacks are appended after its own entries, in B's order.
 * 5. guard: a file holding another module is refused, not written.
 *
 * The importer is run under the memory cap, per the lane rule.
 *
 * Usage: node scripts/import-b/proof-into.mjs */
import { execFileSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import url from "node:url";

const root = path.resolve(path.dirname(url.fileURLToPath(import.meta.url)), "..", "..");
const here = path.dirname(url.fileURLToPath(import.meta.url));
const scratch = "/tmp/opencode/proof-into";
fs.rmSync(scratch, { recursive: true, force: true });
fs.mkdirSync(scratch, { recursive: true });

const CAPPED = "/home/jackwu/Desktop/hp/merge/capped.sh";
const B_MODULE = "multilayer-perceptron";
const A_MODULE = "neural-networks-perceptrons-mlps-activations";
const MODULE_FILE = path.join(scratch, "split", `${A_MODULE}.json`);

const runImporter = (into) =>
  execFileSync(CAPPED, ["3", "python3", path.join(here, "problems.py"), B_MODULE, "--into", into],
    { cwd: root, encoding: "utf8", maxBuffer: 1 << 28 });
const sha = (file) => crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const canon = (value) =>
  JSON.stringify(value, (key, val) => (val && typeof val === "object" && !Array.isArray(val)
    ? Object.keys(val).sort().reduce((acc, k) => ({ ...acc, [k]: val[k] }), {})
    : val));

let failures = 0;
const check = (name, ok, detail) => {
  if (!ok) failures += 1;
  console.log(`${ok ? "ok  " : "FAIL"} ${name}`);
  if (!ok && detail) console.log(`  ${detail}`);
};

/* The module files the lanes merge into come from scripts/split-content.mjs. */
execFileSync("node", [path.join(root, "scripts", "split-content.mjs"), "--out", path.join(scratch, "split")],
  { cwd: root, encoding: "utf8", stdio: ["ignore", "ignore", "pipe"] });
const split = JSON.parse(fs.readFileSync(MODULE_FILE, "utf8"));
const notebookCards = split.index.filter((e) => e.url.startsWith("/editor/?nb="));
console.log(`split wrote the MLP module file: ${split.index.length} index entries ` +
  `(${notebookCards.length} notebook cards), ${split.theory.length} theory entries`);

/* 1. reproduce: the merged file is byte-identical to the split file. */
const into = path.join(scratch, "mlp.json");
fs.copyFileSync(MODULE_FILE, into);
runImporter(into);
const after = fs.readFileSync(into, "utf8");
check("1. merging into the MLP module file reproduces it byte for byte",
  after === fs.readFileSync(MODULE_FILE, "utf8"));

/* 2. idempotent. */
runImporter(into);
check("2. running the import again leaves the file byte-identical",
  sha(into) === sha(MODULE_FILE));

/* 3. create: no file there. The new file holds the four keys in the lane's
 * order and nothing else, and the entries are the importer's. */
const created = path.join(scratch, "created.json");
fs.rmSync(created, { force: true });
runImporter(created);
const first = JSON.parse(fs.readFileSync(created, "utf8"));
check("3. a new file holds module, stage, index, theory and nothing else",
  Object.keys(first).join(",") === "module,stage,index,theory",
  Object.keys(first).join(","));
check("3. a new file carries the module's 9 index and 9 theory entries",
  first.index.length === 9 && first.theory.length === 9,
  `${first.index.length} index, ${first.theory.length} theory`);
check("3. a new file's entries equal the ones merged into an existing file",
  canon(first.index) === canon(JSON.parse(fs.readFileSync(into, "utf8")).index.filter((e) => !e.url.startsWith("/editor/?nb="))) &&
  canon(first.theory) === canon(JSON.parse(fs.readFileSync(into, "utf8")).theory));
const createdSha = sha(created);
runImporter(created);
check("3. creating then re-running is byte-identical", sha(created) === createdSha);

/* 4. append: entries the file already has are replaced in place; ids it lacks
 * are appended after its own entries, in B's order. Drop two theory cards and
 * one theory entry, keeping a notebook card between them. */
const partial = path.join(scratch, "partial.json");
const droppedIndex = ["mlp-max-by-hand", "mlp-square-by-sawtooth"];
const droppedTheory = ["mlp-square-by-sawtooth"];
const base = JSON.parse(fs.readFileSync(MODULE_FILE, "utf8"));
base.index = base.index.filter((e) => !droppedIndex.includes(e.id));
base.theory = base.theory.filter((e) => !droppedTheory.includes(e.id));
fs.writeFileSync(partial, JSON.stringify(base, null, 2) + "\n");
runImporter(partial);
const merged = JSON.parse(fs.readFileSync(partial, "utf8"));
const keptIndexIds = base.index.map((e) => e.id);
const keptTheoryIds = base.theory.map((e) => e.id);
check("4. the file's own entries stay where they were",
  merged.index.slice(0, keptIndexIds.length).map((e) => e.id).join(",") === keptIndexIds.join(",") &&
  merged.theory.slice(0, keptTheoryIds.length).map((e) => e.id).join(",") === keptTheoryIds.join(","));
check("4. the ids the file lacked are appended after its own entries",
  merged.index.slice(keptIndexIds.length).map((e) => e.id).join(",") === droppedIndex.join(",") &&
  merged.theory.slice(keptTheoryIds.length).map((e) => e.id).join(",") === droppedTheory.join(","));
check("4. the notebook cards between them are untouched",
  canon(merged.index.filter((e) => e.url.startsWith("/editor/?nb="))) === canon(notebookCards));
/* Order is not asserted here: the appended ids land after the file's own
 * entries, which is the point of the check above. */
const byId = (a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);
check("4. every entry the file holds now equals the reproduced module file's",
  canon([...merged.index].sort(byId)) === canon([...split.index].sort(byId)) &&
  canon([...merged.theory].sort(byId)) === canon([...split.theory].sort(byId)));
const partialSha = sha(partial);
runImporter(partial);
check("4. the partial merge is idempotent too", sha(partial) === partialSha);

/* 5. guard: a file holding another module is refused and left alone. */
const foreign = path.join(scratch, "foreign.json");
const before = JSON.stringify({ module: "some-other-module", stage: "Round 1", index: [], theory: [] }, null, 2) + "\n";
fs.writeFileSync(foreign, before);
let refused = false;
try {
  runImporter(foreign);
} catch (error) {
  refused = /holds module/.test(String(error.stderr || error));
}
check("5. a file holding another module is refused", refused);
check("5. the refused file is untouched", fs.readFileSync(foreign, "utf8") === before);

console.log(failures === 0 ? "PROOF PASSED" : `PROOF FAILED (${failures})`);
process.exit(failures === 0 ? 0 : 1);
