/**
 * Splits content/problems.js and content/theory.js into per-module files under
 * content/problems/<module id>.json, plus content/problems/contests.json.
 *
 * One-off for the move to per-module files (see merge/CONTRACT-CHANGES.md). The
 * two .js files stay as they are; the split writes wherever --out points, which
 * is scratch until the platform lane applies the layout.
 *
 *   node scripts/split-content.mjs --out /tmp/opencode/split
 *
 * Nothing is renamed, normalised or reordered inside a module: each file holds
 * the subsequence of the original bank exactly as it appeared. Module files
 * carry no ordering field; assembly order comes from content/chapters.json.
 */
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const STAGE_BY_KEY = { round1: "Round 1", round2: "Round 2", camp: "Camp", advanced: "Advanced" };

/**
 * A module's stage comes from content/chapters.json, not from the entries' own
 * tags: a handful of entries carry the tag of the module they were written for
 * (two modules on this branch), and the split must not touch an entry, so the
 * stage is a property of where the module sits in the curriculum.
 */
function stageOfModule(moduleId, chapters) {
  for (const key of Object.keys(chapters)) {
    for (const chapter of chapters[key]) {
      if (chapter.items.includes(moduleId)) return STAGE_BY_KEY[key];
    }
  }
  throw new Error(`${moduleId}: not in content/chapters.json`);
}

function loadGlobal(file, name) {
  const sandbox = { window: {} };
  vm.runInNewContext(fs.readFileSync(path.join(repoRoot, file), "utf8"), sandbox, { filename: file });
  const value = sandbox.window[name];
  if (!value) throw new Error(`${file} does not set window.${name}`);
  return value;
}

function fileNameFor(moduleId) {
  const safe = moduleId.replace(/[^A-Za-z0-9._-]/g, "-");
  if (!safe) throw new Error(`module id ${moduleId} has no usable file name`);
  return `${safe}.json`;
}

function main() {
  const outDir = process.argv.slice(2).includes("--out")
    ? path.resolve(process.argv[process.argv.indexOf("--out") + 1])
    : path.join(repoRoot, "content", "problems");
  fs.mkdirSync(outDir, { recursive: true });
  // Clear the destination first: a re-run must not leave a stale file behind
  // when a module is renamed or its entries move to another module.
  for (const entry of fs.readdirSync(outDir, { withFileTypes: true })) {
    if (entry.isFile() && entry.name.endsWith(".json")) fs.unlinkSync(path.join(outDir, entry.name));
  }

  const problems = loadGlobal("content/problems.js", "AIO_PROBLEMS");
  const theory = loadGlobal("content/theory.js", "AIO_THEORY");
  const chapters = JSON.parse(fs.readFileSync(path.join(repoRoot, "content", "chapters.json"), "utf8"));

  const indexByModule = new Map();
  for (const entry of problems.problems) {
    if (!indexByModule.has(entry.moduleId)) indexByModule.set(entry.moduleId, []);
    indexByModule.get(entry.moduleId).push(entry);
  }
  const theoryByModule = new Map();
  for (const entry of theory.problems) {
    if (!theoryByModule.has(entry.moduleId)) theoryByModule.set(entry.moduleId, []);
    theoryByModule.get(entry.moduleId).push(entry);
  }

  const modules = [...indexByModule.keys()].sort();
  const orphan = [...theoryByModule.keys()].filter((m) => !indexByModule.has(m));
  if (orphan.length) throw new Error(`theory modules with no problems.js entries: ${orphan.join(", ")}`);

  let withTheory = 0;
  for (const moduleId of modules) {
    const index = indexByModule.get(moduleId);
    const moduleTheory = theoryByModule.get(moduleId) || [];
    if (moduleTheory.length) withTheory++;
    const file = {
      module: moduleId,
      stage: stageOfModule(moduleId, chapters),
      index,
      theory: moduleTheory,
    };
    fs.writeFileSync(path.join(outDir, fileNameFor(moduleId)), JSON.stringify(file, null, 2) + "\n");
  }

  fs.writeFileSync(
    path.join(outDir, "contests.json"),
    JSON.stringify({ contests: problems.contests }, null, 2) + "\n",
  );

  console.log(
    `split ${problems.problems.length} index entries and ${theory.problems.length} theory entries into ` +
      `${modules.length} module file(s) (${withTheory} with theory) plus contests.json in ${outDir}`,
  );
}

main();
