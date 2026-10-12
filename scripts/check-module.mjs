#!/usr/bin/env node
// scripts/check-module.mjs
//
// One command a module manager runs before saying READY. Usage:
//   npm run check:module -- <module id> [--no-browser] [--base http://127.0.0.1:4173/]
//
// One line per check, and a non-zero exit if any failed. The site must already
// be served for the render checks (npm run dev, or the merge preview); --base
// or AIO_BASE_URL says where, and --no-browser skips them.

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import vm from "node:vm";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const args = process.argv.slice(2);
const moduleId = args.find((arg) => !arg.startsWith("--"));
const noBrowser = args.includes("--no-browser");
const baseFlagIndex = args.indexOf("--base");
const base =
  (baseFlagIndex !== -1 && args[baseFlagIndex + 1]) ||
  process.env.AIO_BASE_URL ||
  "http://127.0.0.1:4173/";

if (!moduleId) {
  console.error("usage: npm run check:module -- <module id> [--no-browser] [--base <url>]");
  process.exit(2);
}

function loadBrowserGlobal(filePath, globalName) {
  const source = fs.readFileSync(path.join(repoRoot, filePath), "utf8");
  const sandbox = { window: {} };
  vm.runInNewContext(source, sandbox, { filename: filePath });
  return sandbox.window[globalName];
}

const results = [];

function report(name, ok, detail) {
  results.push({ name, ok, detail });
}

// ----------------------------------- build ----------------------------------

const build = spawnSync(process.execPath, ["scripts/build-content.mjs"], { cwd: repoRoot, encoding: "utf8" });
function buildFailureDetail(run) {
  if (run.status === 0) return "";
  // The last line is a Node banner ("Node.js v26.10.0"); the thrown message is
  // the first line that names an Error.
  const lines = (run.stdout + run.stderr).split("\n");
  const message = lines.find((line) => line.includes("Error:")) || lines[0] || "";
  return message.trim().slice(0, 160);
}
report("build", build.status === 0, buildFailureDetail(build));

// --------------------------------- validate ---------------------------------

const validate = spawnSync(process.execPath, ["scripts/validate-content.mjs"], { cwd: repoRoot, encoding: "utf8" });
report(
  "validate",
  validate.status === 0,
  validate.status === 0 ? "" : (validate.stdout + validate.stderr).match(/^- .*$/gm)?.slice(0, 3).join("\n") || "",
);

// ------------------------------ the module file ------------------------------

const content = loadBrowserGlobal("content/catalog.js", "AIO_CONTENT");
const module = content.modules.find((entry) => entry.id === moduleId);
if (!module) {
  console.error(`unknown module id "${moduleId}"`);
  process.exit(2);
}
const lessonPath = path.join(repoRoot, module.file);
if (!fs.existsSync(lessonPath)) {
  console.error(`no lesson file for ${moduleId} (expected ${module.file})`);
  process.exit(2);
}
const source = fs.readFileSync(lessonPath, "utf8");
const lines = source.split("\n");

// --------------------- figures, notebooks, problem cards ---------------------

// These read the generated `content/lessons.js` and `content/problems.js`, so a
// failed build leaves them testing the previous build and reporting ok. When the
// build failed, report that instead of a stale answer.
const buildOk = results.find((entry) => entry.name === "build").ok;
const notChecked = "not checked: build failed";

// The generated files may be missing or stale when the build failed; empty data
// makes the checks below report their failure rather than crash.
function safeLoad(filePath, globalName) {
  try {
    return loadBrowserGlobal(filePath, globalName);
  } catch (error) {
    return null;
  }
}
const loadedLessons = safeLoad("content/lessons.js", "AIO_LESSONS");
const loadedProblems = safeLoad("content/problems.js", "AIO_PROBLEMS");
const lessons = (buildOk && loadedLessons ? loadedLessons.lessons : null) || {};
const problemData = (buildOk && loadedProblems ? loadedProblems : null) || { problems: [] };
const problemIds = new Set(problemData.problems.map((problem) => problem.id));

// Every lesson file on disk, by basename; a link may cross stage directories.
const lessonFiles = new Set();
for (const stageDir of fs.readdirSync(path.join(repoRoot, "content"), { withFileTypes: true })) {
  if (!stageDir.isDirectory()) continue;
  for (const entry of fs.readdirSync(path.join(repoRoot, "content", stageDir.name))) {
    if (entry.endsWith(".md")) lessonFiles.add(entry);
  }
}

function walk(blocks, kind, found) {
  (blocks || []).forEach((block) => {
    if (!block) return;
    if (block[kind]) found.push(block[kind]);
    if (block.optional) walk(block.optional.blocks, kind, found);
    if (block.callout) walk(block.callout.blocks, kind, found);
  });
  return found;
}

const lesson = lessons[moduleId] || { sections: [] };
const allBlocks = lesson.sections.flatMap((section) => section.blocks);

const figures = walk(allBlocks, "figure", []);
const figureMissing = figures.filter((figure) => !fs.existsSync(path.join(repoRoot, figure.src.replace(/^\//, ""))));
report("figures", buildOk && !figureMissing.length, !buildOk ? notChecked : figures.length ? `${figures.length} named` : "none named");
if (figureMissing.length) results.find((r) => r.name === "figures").detail = figureMissing.map((f) => `missing ${f.src}`).join("; ");

const notebooks = walk(allBlocks, "notebookCard", []);
const notebookMissing = notebooks.filter((card) => !fs.existsSync(path.join(repoRoot, "notebooks", `${card.id}.ipynb`)));
report("notebooks", buildOk && !notebookMissing.length, !buildOk ? notChecked : notebooks.length ? `${notebooks.length} named` : "none named");
if (notebookMissing.length) results.find((r) => r.name === "notebooks").detail = notebookMissing.map((c) => `missing notebooks/${c.id}.ipynb`).join("; ");

const cards = walk(allBlocks, "problemCard", []);
const cardMissing = cards.filter((card) => !problemIds.has(card.id));
report("problems", buildOk && !cardMissing.length, !buildOk ? notChecked : cards.length ? `${cards.length} named` : "none named");
if (cardMissing.length) results.find((r) => r.name === "problems").detail = cardMissing.map((c) => `no problem ${c.id}`).join("; ");

// ------------------------------ cross-module links --------------------------

// The reader only links module ids with a leading slash; anything else is
// stripped to plain text, so those are the only links that have to resolve.
const linkIds = new Set();
for (const match of source.matchAll(/\/module\/\?id=([A-Za-z0-9_.-]+)/g)) linkIds.add(match[1]);

const redirectsPath = path.join(repoRoot, "content/redirects.json");
const redirects = fs.existsSync(redirectsPath) ? JSON.parse(fs.readFileSync(redirectsPath, "utf8")) : {};
const known = new Set([...content.modules.map((entry) => entry.id), ...Object.keys(redirects)]);

const brokenLinks = [...linkIds].filter((id) => !known.has(id));
report("links", buildOk && !brokenLinks.length, !buildOk ? notChecked : linkIds.size ? `${linkIds.size} to other modules` : "none");
if (brokenLinks.length) results.find((r) => r.name === "links").detail = `unresolved: ${brokenLinks.join(", ")}`;

// ------------------------------- file links --------------------------------

// Lessons also link each other as files ("[Vectors](1.7-MA1-....md)"). The build
// turns one into a /module/?id= link when the file is on disk, and into plain
// text when it is not, so a link to a file nobody has written yet is invisible
// unless the checker looks for it. A link is good when the file is on disk, or
// when the merge map has already assigned it to a module whose lesson is not on
// disk yet (wave 2, still pending).
const MERGE_MAP = "/home/jackwu/Desktop/hp/merge/MERGE-MAP.json";
const pendingFiles = new Set();
if (fs.existsSync(MERGE_MAP)) {
  const map = JSON.parse(fs.readFileSync(MERGE_MAP, "utf8"));
  (map.part1_modules || []).forEach((entry) => {
    if (entry.file && !fs.existsSync(path.join(repoRoot, entry.file))) pendingFiles.add(path.basename(entry.file));
  });
}

// Code fences and inline code are not links, and the parser does not treat them
// as text, so strip them before scanning.
const prose = source
  .replace(/```[^\n]*\n[\s\S]*?```/g, "")
  .replace(/`[^`\n]*`/g, "");

const fileLinks = new Map();
for (const match of prose.matchAll(/\[([^\]]+)\]\(([^)]+)\)/g)) {
  const target = match[2].split("#")[0];
  if (!target.endsWith(".md")) continue;
  const name = path.basename(target);
  if (!fileLinks.has(name)) fileLinks.set(name, []);
  fileLinks.get(name).push(match[1]);
}
const unknownFiles = [];
const pendingNames = [];
[...fileLinks.keys()].forEach((name) => {
  if (lessonFiles.has(name)) return;
  if (pendingFiles.has(name)) {
    pendingNames.push(name);
    return;
  }
  unknownFiles.push(name);
});
const fileLinkDetail = [
  ...(unknownFiles.length ? [`unknown file ${unknownFiles.join(", ")}`] : []),
  ...(pendingNames.length ? pendingNames.map((name) => `pending ${name}`) : []),
].join("; ");
report("file-links", !unknownFiles.length, fileLinks.size ? fileLinkDetail || `${fileLinks.size} to lesson files` : "none");
if (unknownFiles.length) results.find((r) => r.name === "file-links").detail = fileLinkDetail;

// ------------------------------- b-notebooks -------------------------------

// Site B ships notebooks per module; the merge has to land each of them in A,
// reachable from the lesson or from a problem, or the reader loses them
// silently. B's `-solution` notebooks are answers and are not carried over.
// A B lesson also names notebooks it did not ship in its own folder: shared
// folders (round-2, classical-ml, deep-learning, modern-ai) reached by a
// <NotebookCard notebookPath="..." />, and those count for this module too.
const B_SITE = "/home/jackwu/Desktop/Repos/USAAIO-Guide";
const unreachable = [];
let bNotebookCount = 0;
const mapEntry = fs.existsSync(MERGE_MAP)
  ? JSON.parse(fs.readFileSync(MERGE_MAP, "utf8")).part1_modules?.find((entry) => entry.id === moduleId)
  : null;
if (mapEntry && mapEntry.b_id) {
  // Ids (basename without .ipynb) of the notebooks B ships for this module,
  // plus the ones its lesson names by path. The path cards and the folder scan
  // are deduplicated here.
  const bNotebooks = new Set();
  const bDir = path.join(B_SITE, "notebooks", mapEntry.b_id);
  if (fs.existsSync(bDir)) {
    for (const name of fs.readdirSync(bDir)) {
      if (name.endsWith(".ipynb") && !name.endsWith("-solution.ipynb")) bNotebooks.add(name.slice(0, -".ipynb".length));
    }
  }
  if (mapEntry.b_path) {
    const bLessonPath = path.join(B_SITE, mapEntry.b_path);
    if (fs.existsSync(bLessonPath)) {
      const bLesson = fs.readFileSync(bLessonPath, "utf8");
      for (const match of bLesson.matchAll(/notebookPath="([^"]+)"/g)) {
        const name = path.basename(match[1]);
        if (!name.endsWith(".ipynb") || name.endsWith("-solution.ipynb")) continue;
        bNotebooks.add(name.slice(0, -".ipynb".length));
      }
    }
  }

  if (buildOk && bNotebooks.size) {
    // The reachability test needs the problem data the build generates, so a
    // failed build would flag every notebook as unreferenced; the report below
    // says "not checked: build failed" instead, like the other generated-file
    // checks.
    const referencedNbs = new Set(
      problemData.problems.filter((problem) => problem.url).map((problem) => (problem.url.split("nb=")[1] || "").trim()),
    );
    [...bNotebooks]
      .sort()
      .forEach((id) => {
        bNotebookCount += 1;
        const exists = fs.existsSync(path.join(repoRoot, "notebooks", `${id}.ipynb`));
        const inLesson = new RegExp(`<NotebookCard id="${id}"`).test(source);
        const inProblems = referencedNbs.has(id);
        if (!exists || (!inLesson && !inProblems)) {
          unreachable.push(`${id} (${exists ? "no reference" : "missing in A"})`);
        }
      });
  }
}
report(
  "b-notebooks",
  buildOk && !unreachable.length,
  !buildOk ? notChecked : !mapEntry || !mapEntry.b_id ? "no B module" : !bNotebookCount ? "none in B" : `${bNotebookCount} reachable`,
);
if (unreachable.length) results.find((r) => r.name === "b-notebooks").detail = `unreachable: ${unreachable.join(", ")}`;

// ----------------------------------- em dash --------------------------------

// Two lines keep their em dash: the H1, whose title format uses one, and the
// "> **Stage:**" metadata line. The "> **Goal:**" line does not, because the
// site shows it as the module summary. Everywhere else an em dash is a style
// error.
const emDashLines = lines
  .map((line, index) => ({ line, index }))
  .filter(({ line, index }) => line.includes("\u2014") && index !== 0 && !line.startsWith("> **Stage:**"));
report("em-dashes", !emDashLines.length, emDashLines.length ? `${emDashLines.length} line(s)` : "");
if (emDashLines.length) {
  results.find((r) => r.name === "em-dashes").detail = emDashLines
    .slice(0, 5)
    .map(({ index }) => `line ${index + 1}`)
    .join(", ");
}

// ------------------------------ render in Chrome ----------------------------

if (!noBrowser) {
  // Nothing serves the default base in the merge, and the browser pass would
  // take one of the two machine-wide heavy.sh slots just to fail on a dead
  // page. Ask the server for the module page first; a local static server
  // answers well inside 3 s.
  const moduleUrl = `${base.replace(/\/$/, "")}/module/?id=${moduleId}`;
  let served = true;
  try {
    const response = await fetch(moduleUrl, { signal: AbortSignal.timeout(3000) });
    if (!response.ok) served = false;
  } catch (error) {
    served = false;
  }
  if (!served) {
    const why = `no server at ${base}: use the shared server, AIO_BASE_URL=http://127.0.0.1:4200/<worktree folder name>/ (MODULE-PIPELINE.md step 7); never start one`;
    for (const width of [1440, 375]) report(`render ${width}`, false, why);
  } else {
    const heavy = process.env.HEAVY_SH || "/home/jackwu/Desktop/hp/merge/heavy.sh";
    const useHeavy = fs.existsSync(heavy) && fs.accessSync(heavy, fs.constants.X_OK) === undefined;
    const command = useHeavy ? heavy : process.execPath;
    const commandArgs = useHeavy ? ["python3", "scripts/check-module-browser.py", base, moduleId] : ["scripts/check-module-browser.py", base, moduleId];

    for (const width of [1440, 375]) {
      const env = { ...process.env, AIO_CHECK_WIDTH: String(width) };
      const run = spawnSync(command, [...commandArgs, String(width)], { cwd: repoRoot, encoding: "utf8", env });
      const name = `render ${width}`;
      if (run.status !== 0) {
        report(name, false, (run.stderr || run.stdout || "").trim().split("\n").pop());
        continue;
      }
      let parsed;
      try {
        parsed = JSON.parse(run.stdout.trim().split("\n").pop());
      } catch (error) {
        report(name, false, `could not read the browser report: ${error.message}`);
        continue;
      }
      if (!parsed.ok) {
        report(name, false, parsed.error);
        continue;
      }
      const problems = [];
      if (parsed.console.length) problems.push(`${parsed.console.length} console error(s)`);
      if (parsed.katex) problems.push(`${parsed.katex} KaTeX error(s)`);
      if (parsed.scrollWider || parsed.wide.length) problems.push(`${parsed.wide.length} element(s) wider than the viewport`);
      if (!parsed.loaded) problems.push("the reader never rendered");
      report(name, !problems.length, problems.join(", ") || `rendered "${parsed.title}"`);
    }
  }
}

// ----------------------------------- report ---------------------------------

const width = Math.max(...results.map((entry) => entry.name.length));
let failed = 0;
results.forEach((entry) => {
  if (entry.ok) {
    console.log(`${entry.name.padEnd(width)}  ok${entry.detail ? ` (${entry.detail})` : ""}`);
  } else {
    failed += 1;
    console.log(`${entry.name.padEnd(width)}  FAIL${entry.detail ? `  ${entry.detail}` : ""}`);
  }
});

process.exit(failed ? 1 : 0);
