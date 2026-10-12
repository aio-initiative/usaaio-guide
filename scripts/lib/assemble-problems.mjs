/**
 * Reassembles content/problems.js and content/theory.js from the per-module
 * files under content/problems/*.json. See merge/CONTRACT-CHANGES.md for the
 * layout; scripts/split-content.mjs wrote today's banks into it.
 *
 * Each module file is {module, stage, index: [...], theory: [...]};
 * contests.json is {contests: [...]}. Entries are copied verbatim, so a module
 * file is the source of truth for its own cards once the split is applied.
 *
 * Modules assemble in their content/chapters.json position (stage key order,
 * then chapter, then item), so the problems index opens on the curriculum's
 * first module rather than on whichever module happened to be written into
 * problems.js first.
 *
 *   import { assembleProblems, assembleTheory, writeContentFiles } from
 *     "./scripts/lib/assemble-problems.mjs";
 *
 * Errors are thrown rather than warned: the build calls this, and a half-built
 * bank is worse than no bank.
 */
import fs from "node:fs";
import path from "node:path";

const CONTESTS_FILE = "contests.json";

/** Reads and validates one module file. */
function readModuleFile(file) {
  let parsed;
  try {
    parsed = JSON.parse(fs.readFileSync(file, "utf8"));
  } catch (error) {
    throw new Error(`${file}: ${error.message}`);
  }
  const where = `content/problems/${path.basename(file)}`;
  if (typeof parsed !== "object" || parsed === null) throw new Error(`${where}: not a JSON object`);
  for (const key of ["module", "stage", "index", "theory"]) {
    if (!(key in parsed)) throw new Error(`${where}: missing "${key}"`);
  }
  if (typeof parsed.module !== "string" || !parsed.module) throw new Error(`${where}: "module" is not a non-empty string`);
  for (const key of ["index", "theory"]) {
    if (!Array.isArray(parsed[key])) throw new Error(`${where}: "${key}" is not an array`);
  }
  if (parsed.module !== path.basename(file, ".json")) {
    throw new Error(`${where}: file name does not match "module" (${parsed.module})`);
  }
  for (const entry of [...parsed.index, ...parsed.theory]) {
    if (entry.moduleId !== parsed.module) {
      throw new Error(`${where}: entry ${entry.id} has moduleId ${entry.moduleId}`);
    }
  }
  return parsed;
}

/**
 * The module ids in content/chapters.json order: stage key order (round1,
 * round2, camp, advanced), then chapter, then item. Problems index pages list
 * in array order, so this order is what the reader sees.
 */
export function chaptersOrder(chaptersPath) {
  const chapters = JSON.parse(fs.readFileSync(chaptersPath, "utf8"));
  const order = [];
  const seen = new Set();
  for (const stageId of Object.keys(chapters)) {
    for (const chapter of chapters[stageId]) {
      for (const id of chapter.items) {
        if (!seen.has(id)) {
          seen.add(id);
          order.push(id);
        }
      }
    }
  }
  return order;
}

/**
 * Module files in assembly order: the module's position in content/chapters.json,
 * then file name. A module id that is not in the chapters file is a build error
 * naming content/problems/<id>.json, because its entries belong in the file of
 * the module the chapters file lists instead (a redirected old id, for example).
 */
function moduleFiles(dir, chaptersPath) {
  const position = new Map();
  chaptersOrder(chaptersPath).forEach((id, index) => position.set(id, index));
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const files = entries
    .filter((entry) => entry.isFile() && entry.name.endsWith(".json") && entry.name !== CONTESTS_FILE)
    .map((entry) => path.join(dir, entry.name));
  const withPosition = files.map((file) => {
    const module = readModuleFile(file);
    if (!position.has(module.module)) {
      throw new Error(`content/problems/${path.basename(file)}: module ${module.module} is not in content/chapters.json`);
    }
    return { file, module, position: position.get(module.module) };
  });
  return withPosition.sort((a, b) => a.position - b.position || a.file.localeCompare(b.file));
}

function readContests(dir) {
  const file = path.join(dir, CONTESTS_FILE);
  if (!fs.existsSync(file)) return [];
  let parsed;
  try {
    parsed = JSON.parse(fs.readFileSync(file, "utf8"));
  } catch (error) {
    throw new Error(`${file}: ${error.message}`);
  }
  if (!parsed || !Array.isArray(parsed.contests)) throw new Error(`${file}: "contests" is not an array`);
  return parsed.contests;
}

/** {problems, contests} for content/problems.js, from a content/problems dir. */
export function assembleProblems(dir, chaptersPath = path.join(dir, "..", "chapters.json")) {
  const modules = moduleFiles(dir, chaptersPath);
  const problems = modules.flatMap((entry) => entry.module.index);
  const ids = new Set();
  for (const problem of problems) {
    if (ids.has(problem.id)) throw new Error(`problem ${problem.id} appears in two module files`);
    ids.add(problem.id);
  }
  return { problems, contests: readContests(dir) };
}

/** {problems} for content/theory.js, from a content/problems dir. */
export function assembleTheory(dir, chaptersPath = path.join(dir, "..", "chapters.json")) {
  const problems = moduleFiles(dir, chaptersPath).flatMap((entry) => entry.module.theory);
  const ids = new Set();
  for (const problem of problems) {
    if (ids.has(problem.id)) throw new Error(`theory problem ${problem.id} appears in two module files`);
    ids.add(problem.id);
  }
  return { problems };
}

/** The module ids in assembly order, for anything that lists modules. */
export function moduleIds(dir, chaptersPath = path.join(dir, "..", "chapters.json")) {
  return moduleFiles(dir, chaptersPath).map((entry) => entry.module.module);
}

/**
 * Writes content/problems.js and content/theory.js from a content/problems dir
 * into repoRoot, in the IIFE wrappers the browser globals expect. Returns the
 * two paths.
 */
export function writeContentFiles(dir, repoRoot, chaptersPath = path.join(dir, "..", "chapters.json")) {
  const problems = assembleProblems(dir, chaptersPath);
  const theory = assembleTheory(dir, chaptersPath);
  const problemsFile = path.join(repoRoot, "content", "problems.js");
  const theoryFile = path.join(repoRoot, "content", "theory.js");
  fs.writeFileSync(problemsFile, toProblemsJs(problems));
  fs.writeFileSync(theoryFile, toTheoryJs(theory));
  return { problemsFile, theoryFile };
}

const INDENT = "  ";

function toJs(value, depth) {
  const pad = INDENT.repeat(depth);
  const inner = INDENT.repeat(depth + 1);
  if (Array.isArray(value)) {
    if (value.length === 0) return "[]";
    return (
      "[\n" +
      value.map((item) => `${inner}${toJs(item, depth + 1)},`).join("\n") +
      `\n${pad}]`
    );
  }
  if (value && typeof value === "object") {
    const keys = Object.keys(value);
    if (keys.length === 0) return "{}";
    return (
      "{\n" +
      keys.map((key) => `${inner}${JSON.stringify(key)}: ${toJs(value[key], depth + 1)},`).join("\n") +
      `\n${pad}}`
    );
  }
  return JSON.stringify(value);
}

function toProblemsJs(data) {
  return (
    "(function () {\n" +
    `  const problems = ${toJs(data.problems, 1)};\n` +
    `  const contests = ${toJs(data.contests, 1)};\n` +
    "\n" +
    "  window.AIO_PROBLEMS = { problems, contests };\n" +
    "})();\n"
  );
}

function toTheoryJs(data) {
  return (
    "(function () {\n" +
    `  const problems = ${toJs(data.problems, 1)};\n` +
    "\n" +
    "  window.AIO_THEORY = { problems };\n" +
    "})();\n"
  );
}
