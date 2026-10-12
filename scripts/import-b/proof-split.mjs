/**
 * Proof that the per-module split loses nothing: loads today's content/problems.js
 * and content/theory.js, splits them into scratch, reassembles, and compares
 * after JSON.parse.
 *
 *   node scripts/import-b/proof-split.mjs
 *
 * What must hold:
 *  - every module's index and theory array is deep-equal to the original
 *    subsequence for that module (same entries, same order),
 *  - globally the assembled entries are the same multiset as the original,
 *  - contests are equal,
 *  - the module order is content/chapters.json order (stage key order, then
 *    chapter, then item),
 *  - the .js files the assembler writes, loaded back the way the site loads
 *    them, give the same data.
 *
 * What cannot hold, and is not claimed: byte-identical global order. Today's
 * problems.js interleaves modules in 133 runs over 55 modules, and a per-module
 * split has one run per module. Order inside a module is exact.
 */
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

import { assembleProblems, assembleTheory, chaptersOrder, moduleIds, writeContentFiles } from "../lib/assemble-problems.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const scratch = "/home/jackwu/Desktop/hp/merge/scratch/practice/split-roundtrip";
// The split writes into scratch, whose parent has no chapters.json, so the
// chapters file is the repo's own.
const chaptersPath = path.join(repoRoot, "content", "chapters.json");

let failures = 0;
const check = (ok, message) => {
  console.log(`${ok ? "ok  " : "FAIL"} ${message}`);
  if (!ok) failures++;
};

function loadGlobal(file, name) {
  const sandbox = { window: {} };
  vm.runInNewContext(fs.readFileSync(file, "utf8"), sandbox, { filename: file });
  return sandbox.window[name];
}

function sorted(value) {
  if (Array.isArray(value)) return value.map(sorted);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, sorted(value[key])]));
  }
  return value;
}

const deepEqual = (a, b) => JSON.stringify(sorted(a)) === JSON.stringify(sorted(b));

const originalProblems = loadGlobal(path.join(repoRoot, "content/problems.js"), "AIO_PROBLEMS");
const originalTheory = loadGlobal(path.join(repoRoot, "content/theory.js"), "AIO_THEORY");

fs.rmSync(scratch, { recursive: true, force: true });
execFileSync(process.execPath, [path.join(repoRoot, "scripts/split-content.mjs"), "--out", scratch], {
  cwd: repoRoot,
  stdio: "inherit",
});

const assembledProblems = assembleProblems(scratch, chaptersPath);
const assembledTheory = assembleTheory(scratch, chaptersPath);

// Per-module: the assembled file must contain, for each module, exactly the
// original entries in the original order.
function byModule(entries) {
  const map = new Map();
  for (const entry of entries) {
    if (!map.has(entry.moduleId)) map.set(entry.moduleId, []);
    map.get(entry.moduleId).push(entry);
  }
  return map;
}

const originalIndexByModule = byModule(originalProblems.problems);
const originalTheoryByModule = byModule(originalTheory.problems);
const assembledIndexByModule = byModule(assembledProblems.problems);
const assembledTheoryByModule = byModule(assembledTheory.problems);

check(
  [...originalIndexByModule.keys()].sort().join(",") === [...assembledIndexByModule.keys()].sort().join(","),
  `index: same ${assembledIndexByModule.size} module(s)`,
);
let moduleFailures = 0;
for (const [moduleId, original] of originalIndexByModule) {
  if (!deepEqual(original, assembledIndexByModule.get(moduleId))) moduleFailures++;
}
check(moduleFailures === 0, `index: every module's entries deep-equal in original order`);

let theoryModuleFailures = 0;
for (const [moduleId, original] of originalTheoryByModule) {
  if (!deepEqual(original, assembledTheoryByModule.get(moduleId))) theoryModuleFailures++;
}
for (const moduleId of assembledTheoryByModule.keys()) {
  if (!originalTheoryByModule.has(moduleId)) theoryModuleFailures++;
}
check(theoryModuleFailures === 0, `theory: every module's entries deep-equal in original order`);

// Global multiset: no entry lost, gained or altered.
const multiset = (entries) =>
  entries.map((entry) => JSON.stringify(sorted(entry))).sort().join("\n");
check(
  multiset(originalProblems.problems) === multiset(assembledProblems.problems),
  `index: same multiset of ${originalProblems.problems.length} entries`,
);
check(
  multiset(originalTheory.problems) === multiset(assembledTheory.problems),
  `theory: same multiset of ${originalTheory.problems.length} entries`,
);
check(deepEqual(originalProblems.contests, assembledProblems.contests), `contests: equal (${originalProblems.contests.length})`);

// Module order follows content/chapters.json, not first appearance.
const expectedOrder = chaptersOrder(chaptersPath).filter((moduleId) => originalIndexByModule.has(moduleId));
check(
  JSON.stringify(moduleIds(scratch, chaptersPath)) === JSON.stringify(expectedOrder),
  "index: module order is content/chapters.json order",
);

// The emitted .js files must load the way the site loads them.
const rebuiltRoot = path.join(scratch, "rebuilt");
fs.mkdirSync(path.join(rebuiltRoot, "content"), { recursive: true });
const written = writeContentFiles(scratch, rebuiltRoot, chaptersPath);
const reloadedProblems = loadGlobal(written.problemsFile, "AIO_PROBLEMS");
const reloadedTheory = loadGlobal(written.theoryFile, "AIO_THEORY");
check(
  multiset(originalProblems.problems) === multiset(reloadedProblems.problems),
  "rebuilt problems.js loads to the same entries",
);
check(
  multiset(originalTheory.problems) === multiset(reloadedTheory.problems),
  "rebuilt theory.js loads to the same entries",
);
check(deepEqual(originalProblems.contests, reloadedProblems.contests), "rebuilt problems.js keeps the contests");

const note = [
  `problems.js holds ${originalProblems.problems.length} entries over ${originalIndexByModule.size} modules,`,
  `theory.js ${originalTheory.problems.length} over ${originalTheoryByModule.size}. problems.js interleaves its`,
  "modules, so the reassembled file is not byte-identical to it: within a module the entries are in the",
  "original order, and the modules are in content/chapters.json order.",
].join(" ");

if (failures) {
  console.log(`\nPROOF FAILED (${failures} check(s))`);
  process.exit(1);
}
console.log(`\nPROOF PASSED\n${note}`);
