#!/usr/bin/env node
/* Prove scripts/import-b/problems.py reproduces the MLP module's committed
 * entries exactly: run the importer on B's multilayer-perceptron folder and
 * compare every entry, keyed by id, against content/theory.js and
 * content/problems.js. Deep equality, key order irrelevant.
 *
 * Usage: node scripts/import-b/proof-mlp.mjs [--out <scratch file>] */
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import url from "node:url";

const root = path.resolve(path.dirname(url.fileURLToPath(import.meta.url)), "..", "..");
const here = path.dirname(url.fileURLToPath(import.meta.url));

const loadGlobal = (file, name) => {
  const sandbox = { window: {} };
  vm.runInNewContext(fs.readFileSync(path.join(root, file), "utf8"), sandbox, { filename: file });
  return sandbox.window[name];
};

const theory = loadGlobal("content/theory.js", "AIO_THEORY").problems;
const index = loadGlobal("content/problems.js", "AIO_PROBLEMS").problems;
const A_MODULE = "neural-networks-perceptrons-mlps-activations";

const out = execFileSync("python3", [path.join(here, "problems.py"), "multilayer-perceptron"], {
  cwd: root,
  encoding: "utf8",
  maxBuffer: 1 << 28,
});
const imported = JSON.parse(out);

const problems = {
  theory: theory.filter((p) => p.moduleId === A_MODULE),
  index: index.filter((p) => p.moduleId === A_MODULE && p.url.startsWith("/theory/")),
};

/* JSON.stringify with every object's keys sorted, so the comparison ignores
 * key order (the committed files were hand-indented). */
const canon = (value) =>
  JSON.stringify(value, (key, val) => (val && typeof val === "object" && !Array.isArray(val)
    ? Object.keys(val).sort().reduce((acc, k) => ({ ...acc, [k]: val[k] }), {})
    : val));

let failures = 0;
const report = (name, actual, expected) => {
  const ok = canon(actual) === canon(expected);
  if (!ok) failures += 1;
  console.log(`${ok ? "ok  " : "FAIL"} ${name}`);
  if (!ok) {
    for (const id of new Set([...Object.keys(actual), ...Object.keys(expected)])) {
      if (canon(actual[id]) !== canon(expected[id])) {
        console.log(`  ${id}:
    importer:  ${canon(actual[id])}
    committed: ${canon(expected[id])}`);
      }
    }
  }
  return ok;
};

for (const field of ["theory", "index"]) {
  const byId = (list) => Object.fromEntries(list.map((entry) => [entry.id, entry]));
  const importedIds = imported[field].map((entry) => entry.id);
  const committedIds = problems[field].map((entry) => entry.id);
  const sameSet = JSON.stringify([...importedIds].sort()) === JSON.stringify([...committedIds].sort());
  if (!sameSet) {
    failures += 1;
    console.log(`FAIL ${field}: id sets differ`);
    console.log(`  importer:   ${importedIds.join(", ")}`);
    console.log(`  committed:  ${committedIds.join(", ")}`);
  } else {
    console.log(`ok   ${field}: ${importedIds.length} ids match`);
  }
  report(`${field}: every entry deep-equal`, byId(imported[field]), byId(problems[field]));
}

/* The committed problems.js also interleaves the module's four notebook cards
 * (url /editor/?nb=...), which this importer does not produce: those come from
 * the notebook converter. List them so the proof is explicit about it. */
const notebookCards = index.filter((p) => p.moduleId === A_MODULE && p.url.startsWith("/editor/?nb="));
console.log(`note: problems.js also holds ${notebookCards.length} notebook card(s) for this module, from the notebook converter: ${notebookCards.map((p) => p.id).join(", ")}`);

const outFlag = process.argv.indexOf("--out");
if (outFlag !== -1 && process.argv[outFlag + 1]) {
  fs.writeFileSync(process.argv[outFlag + 1], out);
  console.log(`wrote ${process.argv[outFlag + 1]}`);
}
console.log(failures === 0 ? "PROOF PASSED" : `PROOF FAILED (${failures})`);
process.exit(failures === 0 ? 0 : 1);
