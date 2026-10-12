import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

import { assembleProblems, assembleTheory } from "./lib/assemble-problems.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function loadBrowserGlobal(filePath, globalName) {
  const source = fs.readFileSync(path.join(repoRoot, filePath), "utf8");
  const sandbox = { window: {} };
  vm.runInNewContext(source, sandbox, { filename: filePath });
  return sandbox.window[globalName];
}

// Module Markdown files open with "# <n> · <CODE> — <title>" followed by a
// "> **Goal:**" line; there is no YAML frontmatter.
function readModuleHeader(filePath) {
  const source = fs.readFileSync(path.join(repoRoot, filePath), "utf8");
  const lines = source.split("\n");
  const titleLine = lines.find((line) => line.startsWith("# ")) || "";
  const match = titleLine.slice(2).match(/^([\d.]+)\s+·\s+([A-Z]{2}\d{1,2})\s+(?:·|\u2014)\s+(.*)$/);
  return {
    title: match ? match[3] : "",
    code: match ? match[2] : "",
    goal: (lines.find((line) => line.startsWith("> **Goal:**")) || "").replace("> **Goal:**", "").trim(),
    headings: lines.filter((line) => line.startsWith("## ")).map((line) => line.slice(3).trim()),
  };
}

// A module is a linear lesson, so it names its own topic headings. Only the two
// metadata sections the catalog reads are required; everything after them is
// free-form and becomes one reader section per "## " heading.
const REQUIRED_MODULE_SECTIONS = [];   // Prerequisites is optional: a lesson that declares none has no such section

// Leftovers from the old fixed-section schema. A module still carrying one of
// these is a conversion that was never finished, not a deliberate topic name.
const RETIRED_MODULE_SECTIONS = new Set([
  "Learning Outcomes",
  "Theory",
  "From-Scratch Implementation",
  "Library Usage",
  "Further Practice",
  "Practice Problems",
]);

const LESSON_BLOCK_KEYS = [
  "p", "tex", "heading", "list", "olist", "table", "quote", "code", "details", "image",
  "figure", "optional", "callout", "pyEditor", "problemCard", "notebookCard", "resources", "output",
];

// Lesson component blocks point at files and at the two problem banks, so they
// are checked against those here. `lessonCards` collects every card for the
// cross-checks further down, once the banks are loaded.
const lessonCards = [];

function validateBlocks(blocks, where, moduleId) {
  if (!Array.isArray(blocks) || !blocks.length) {
    errors.push(`${where}: expected a non-empty blocks array`);
    return;
  }
  blocks.forEach((block, index) => {
    const at = `${where} block ${index}`;
    const keys = LESSON_BLOCK_KEYS.filter((key) => block && block[key] !== undefined);
    if (keys.length !== 1) {
      errors.push(`${at}: must carry exactly one of ${LESSON_BLOCK_KEYS.join("/")}`);
      return;
    }
    if (block.table && !Array.isArray(block.table.rows)) {
      errors.push(`${at}: table is missing its rows`);
    }
    if (block.code && !Array.isArray(block.code.lines)) {
      errors.push(`${at}: code is missing its lines`);
    }
    if (block.output && !Array.isArray(block.output.lines)) {
      errors.push(`${at}: output is missing its lines`);
    }
    // An output fence is the print of a code block in the same section. A
    // sentence or two between them is fine ("...the two ways of computing it
    // agree:" then the output), but a heading, a second code block or a figure
    // in between means the output no longer belongs to the code above it.
    if (block.output) {
      let scan = index;
      while (scan > 0 && blocks[scan - 1].p) scan -= 1;
      if (scan === 0 || !blocks[scan - 1].code) {
        errors.push(`${at}: an output block must follow the code block it prints`);
      }
    }
    if (block.details) validateBlocks(block.details.blocks, `${at} (details)`, moduleId);
    if (block.optional) {
      if (!block.optional.title) errors.push(`${at}: <Optional> needs a title`);
      validateBlocks(block.optional.blocks, `${at} (optional)`, moduleId);
    }
    if (block.callout) {
      const { variant, label, blocks } = block.callout;
      if (variant !== "warning" && variant !== "info") {
        errors.push(`${at}: callout variant must be warning or info, found "${variant}"`);
      }
      if (!label) errors.push(`${at}: a <Warning> or <Info> callout needs its label`);
      validateBlocks(blocks, `${at} (callout)`, moduleId);
    }
    if (block.pyEditor) {
      const { code, packages } = block.pyEditor;
      if (typeof code !== "string" || !code.trim()) errors.push(`${at}: <EmbeddedPyEditor> needs its code`);
      if (packages !== undefined && !Array.isArray(packages)) {
        errors.push(`${at}: <EmbeddedPyEditor> packages must be a list`);
      }
    }
    if (block.figure) {
      const { src, alt } = block.figure;
      if (!src || /^(https?:)?\/\//.test(src)) {
        errors.push(`${at}: <Figure> src must be a file in this repo, found "${src}"`);
      } else if (!fs.existsSync(path.join(repoRoot, src.replace(/^\//, "")))) {
        errors.push(`${at}: <Figure> file is missing: ${src}`);
      }
      if (!alt || alt.trim().length < 10) errors.push(`${at}: <Figure src="${src}"> needs real alt text`);
    }
    if (block.resources) {
      if (!Array.isArray(block.resources)) {
        errors.push(`${at}: resources must be an array of { title, url }`);
      } else {
        block.resources.forEach((resource, n) => {
          if (!resource.title || !resource.url) {
            errors.push(`${at}: resource ${n + 1} needs a title and a url`);
          } else if (!/^(https?:\/\/|\/|#)/.test(resource.url)) {
            errors.push(`${at}: resource ${n + 1} needs an http(s), root or anchor url`);
          }
        });
      }
    }
    if (block.problemCard || block.notebookCard) lessonCards.push({ at, moduleId, block });
  });
}

const content = loadBrowserGlobal("content/catalog.js", "AIO_CONTENT");
const ordering = loadBrowserGlobal("content/ordering.js", "AIO_ORDERING");
const lessons = loadBrowserGlobal("content/lessons.js", "AIO_LESSONS");
const problemData = loadBrowserGlobal("content/problems.js", "AIO_PROBLEMS");
const theoryData = loadBrowserGlobal("content/theory.js", "AIO_THEORY");

// The two banks are generated from content/problems/<module id>.json, so a
// shipped .js file that is not what the assembler builds from those files is
// stale: a lane edited a module file and did not rebuild, and the deployed
// page would disagree with the source of truth. Object key order is irrelevant
// (both sides are serialised with sorted keys), array order is not: the
// problems index lists in array order, and a stale file would show a module's
// problems in the wrong place or not at all.
function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])]));
  }
  return value;
}

const banksMatch = (committed, assembled) =>
  JSON.stringify(canonical(committed)) === JSON.stringify(canonical(assembled));

const errors = [];
const unconverted = [];
const presentation = [];
const moduleIds = new Set();
const expectedPages = [
  "index.html",
  "dashboard/index.html",
  "roadmap/index.html",
  "round1/index.html",
  "round2/index.html",
  "camp/index.html",
  "advanced/index.html",
  "module/index.html",
  "problems/index.html",
  "theory/index.html",
  "mock/index.html",
  "mock/run/index.html",
  "contests/index.html",
  "resources/index.html",
  "glossary/index.html",
  "search/index.html",
  "settings/index.html",
  "login/index.html",
  "contact/index.html",
];

expectedPages.forEach((filePath) => {
  if (!fs.existsSync(path.join(repoRoot, filePath))) {
    errors.push(`Missing route page: ${filePath}`);
  }
});

const problemsDir = path.join(repoRoot, "content", "problems");
const chaptersPath = path.join(repoRoot, "content", "chapters.json");
// Assembling can fail on a malformed module file or one whose module id is not
// in content/chapters.json; the per-entry checks below then have nothing to
// compare against, so the failure is one error rather than a crash.
let assembledProblems = null;
let assembledTheory = null;
try {
  assembledProblems = assembleProblems(problemsDir, chaptersPath);
  assembledTheory = assembleTheory(problemsDir, chaptersPath);
} catch (error) {
  errors.push(`content/problems/: ${error.message}`);
}
if (assembledProblems && !banksMatch(problemData, assembledProblems)) {
  errors.push("content/problems.js is out of date: run npm run build:content");
}
if (assembledTheory && !banksMatch(theoryData, assembledTheory)) {
  errors.push("content/theory.js is out of date: run npm run build:content");
}

content.modules.forEach((module) => {
  if (moduleIds.has(module.id)) errors.push(`Duplicate module id: ${module.id}`);
  moduleIds.add(module.id);

  if (!ordering.sectionOrdering[module.section]) {
    errors.push(`Module ${module.id} uses unknown section ${module.section}`);
  }

  const modulePath = path.join(repoRoot, module.file);
  if (!fs.existsSync(modulePath)) {
    errors.push(`Missing module file for ${module.id}: ${module.file}`);
    return;
  }

  const header = readModuleHeader(module.file);
  if (!header.title) {
    errors.push(`${module.file}: H1 must read "# <number> · <CODE> · <title>"`);
  }
  if (!header.goal) {
    errors.push(`${module.file}: missing the "> **Goal:**" line the summary is built from`);
  }
  if (module.code !== header.code) {
    errors.push(`${module.file}: catalog code ${module.code} does not match the H1 code ${header.code}`);
  }
  REQUIRED_MODULE_SECTIONS.forEach((heading) => {
    if (!header.headings.includes(heading)) {
      errors.push(`${module.file}: missing "## ${heading}" section`);
    }
  });
  // Warn, don't fail: modules are converted to the linear format in batches, so
  // the ones still on the old schema have to keep building meanwhile.
  const retired = header.headings.filter((heading) => RETIRED_MODULE_SECTIONS.has(heading));
  if (retired.length) {
    unconverted.push(`${module.file}: still on the fixed-section schema (${retired.join(", ")})`);
  }

  module.prerequisites.forEach((prerequisiteId) => {
    if (!content.modules.some((entry) => entry.id === prerequisiteId)) {
      errors.push(`Module ${module.id} lists an unknown prerequisite: ${prerequisiteId}`);
    }
  });
});

// The chapter structure lives in its own data file now, so check it directly:
// every module that has a file must be listed exactly once, and under the stage
// its file is actually in.
const chapters = JSON.parse(fs.readFileSync(path.join(repoRoot, "content/chapters.json"), "utf8"));
const listingStage = new Map();
ordering.sections.forEach((sectionId) => {
  const list = chapters[sectionId];
  if (!Array.isArray(list)) {
    errors.push(`content/chapters.json has no chapter list for stage ${sectionId}`);
    return;
  }
  list.forEach((chapter) => {
    if (typeof chapter.name !== "string") {
      errors.push(`content/chapters.json: a chapter in ${sectionId} has no name`);
      return;
    }
    if (!Array.isArray(chapter.items)) {
      errors.push(`content/chapters.json: chapter ${chapter.name} in ${sectionId} has no items list`);
      return;
    }
    chapter.items.forEach((moduleId) => {
      if (listingStage.has(moduleId)) {
        errors.push(`content/chapters.json lists ${moduleId} in two chapters`);
        return;
      }
      listingStage.set(moduleId, sectionId);
    });
  });
});
content.modules.forEach((module) => {
  const listed = listingStage.get(module.id);
  if (!listed) {
    errors.push(`content/chapters.json does not list module ${module.id}`);
  } else if (listed !== module.section) {
    errors.push(
      `content/chapters.json lists ${module.id} under ${listed}, but its file is in the ${module.section} stage`,
    );
  }
});

// Retired module ids redirect to their replacements. The table must be flat (no
// chains) and every target must be a module the structure knows about.
const redirects = JSON.parse(fs.readFileSync(path.join(repoRoot, "content/redirects.json"), "utf8"));
const listedIds = new Set([...moduleIds, ...listingStage.keys()]);
Object.entries(redirects).forEach(([source, target]) => {
  if (target === source) {
    errors.push(`content/redirects.json redirects ${source} to itself`);
    return;
  }
  if (Object.prototype.hasOwnProperty.call(redirects, target)) {
    errors.push(`content/redirects.json chains: ${source} -> ${target} -> ${redirects[target]}`);
    return;
  }
  if (!listedIds.has(target)) {
    errors.push(`content/redirects.json points ${source} at unknown module ${target}`);
  }
});

const orderedIds = new Set();
ordering.sections.forEach((sectionId) => {
  const section = ordering.sectionOrdering[sectionId];
  if (!section) {
    errors.push(`Missing section object for ${sectionId}`);
    return;
  }

  section.chapters.forEach((chapter) => {
    chapter.items.forEach((moduleId) => {
      orderedIds.add(moduleId);
      if (!moduleIds.has(moduleId)) {
        errors.push(`Ordering references missing module id: ${moduleId}`);
      }
    });
  });
});

moduleIds.forEach((moduleId) => {
  if (!orderedIds.has(moduleId)) {
    errors.push(`Catalog module is not listed in ordering: ${moduleId}`);
  }
  if (!lessons.lessons[moduleId]) {
    errors.push(`Missing structured lesson data for module: ${moduleId}`);
  }
});

Object.entries(lessons.lessons).forEach(([moduleId, lesson]) => {
  if (!moduleIds.has(moduleId)) {
    errors.push(`Lesson data references unknown module: ${moduleId}`);
  }
  // Section count is free-form; the Prerequisites box is optional, but every module needs at
  // least one topic of its own.
  if (!Array.isArray(lesson.sections) || !lesson.sections.some((section) => section.id !== "prerequisites")) {
    errors.push(`Lesson ${moduleId} needs at least one topic section`);
    return;
  }
  const seenIds = new Set();
  lesson.sections.forEach((section) => {
    // The "intro" lead-in is deliberately untitled; every other section needs a heading.
    if (!section.id) errors.push(`Lesson ${moduleId} has a section without an id`);
    if (!section.title && section.id !== "intro") {
      errors.push(`Lesson ${moduleId} section ${section.id} has no title`);
    }
    // Section ids become page anchors, so a collision would break the TOC.
    if (seenIds.has(section.id)) errors.push(`Lesson ${moduleId} has duplicate section id ${section.id}`);
    seenIds.add(section.id);
    validateBlocks(section.blocks, `Lesson ${moduleId} / ${section.id}`, moduleId);
  });
  if (!Array.isArray(lesson.anchors) || lesson.anchors.length !== lesson.sections.length) {
    errors.push(`Lesson ${moduleId} anchors do not match its sections`);
  }
});

// A stage tag is the round the problem belongs to, and the module's position
// in content/chapters.json is the authority for that: an import that tags a
// module for the wrong round (wave 2 modules tagged Round 1 on import) would
// file the reader's problems under the wrong filter on the problems index.
const STAGE_TAGS = new Set(["Round 1", "Round 2", "Camp", "Advanced"]);
const STAGE_TAG_BY_SECTION = {
  round1: "Round 1",
  round2: "Round 2",
  camp: "Camp",
  advanced: "Advanced",
};

const problemIds = new Set();
problemData.problems.forEach((problem) => {
  if (problemIds.has(problem.id)) errors.push(`Duplicate problem id: ${problem.id}`);
  problemIds.add(problem.id);
  if (!moduleIds.has(problem.moduleId)) {
    errors.push(`Problem ${problem.id} references unknown module ${problem.moduleId}`);
  }
  ["title", "source", "difficulty", "prompt", "answer"].forEach((field) => {
    if (!problem[field]) errors.push(`Problem ${problem.id} is missing ${field}`);
  });
  const expectedStage = STAGE_TAG_BY_SECTION[listingStage.get(problem.moduleId)];
  if (expectedStage) {
    const entryStage = (problem.tags || []).filter((tag) => STAGE_TAGS.has(tag));
    if (entryStage.length !== 1) {
      errors.push(
        `content/problems/${problem.moduleId}.json: entry ${problem.id} has no single stage tag ` +
          `(found ${entryStage.length === 0 ? "none" : entryStage.join(" and ")}); its module's stage in ` +
          `content/chapters.json is ${expectedStage}`,
      );
    } else if (entryStage[0] !== expectedStage) {
      errors.push(
        `content/problems/${problem.moduleId}.json: entry ${problem.id} is tagged ${entryStage[0]}, but its ` +
          `module's stage in content/chapters.json is ${expectedStage}`,
      );
    }
  }
});

// Every problem must declare exactly one type tag, since the problems page
// filter and the card badge both assume it.
problemData.problems.forEach((problem) => {
  const types = ["Coding", "Theory"].filter((tag) => problem.tags.includes(tag));
  if (types.length !== 1) {
    errors.push(`Problem ${problem.id} must carry exactly one of the Coding/Theory tags (found ${types.length})`);
  }
});

// Theory problems: structure, and a registered problem-bank entry pointing at them.
const theoryIds = new Set();
theoryData.problems.forEach((problem) => {
  if (theoryIds.has(problem.id)) errors.push(`Duplicate theory problem id: ${problem.id}`);
  theoryIds.add(problem.id);

  if (!moduleIds.has(problem.moduleId)) {
    errors.push(`Theory problem ${problem.id} references unknown module ${problem.moduleId}`);
  }
  ["title", "difficulty"].forEach((field) => {
    if (!problem[field]) errors.push(`Theory problem ${problem.id} is missing ${field}`);
  });
  // A multiple-choice problem is one question: exactly one "mcq" part, whose
  // prompt is the whole problem, so it needs no overview or grading line. A
  // written problem has one part or more, at most 15.
  const registeredEntry = problemData.problems.find((entry) => entry.id === problem.id);
  const multipleChoice = Boolean(registeredEntry && registeredEntry.tags.includes("Multiple choice"));
  if (!multipleChoice && (!Array.isArray(problem.overview) || !problem.overview.length)) {
    errors.push(`Theory problem ${problem.id} is missing an overview`);
  }
  if (!Array.isArray(problem.parts) || !problem.parts.length) {
    errors.push(`Theory problem ${problem.id} has no parts`);
    return;
  }
  if (multipleChoice && (problem.parts.length !== 1 || problem.parts[0].kind !== "mcq")) {
    errors.push(`Theory problem ${problem.id} is tagged Multiple choice, so it needs exactly one part of kind "mcq"`);
  }
  if (problem.parts.length > 15) {
    errors.push(`Theory problem ${problem.id} has more than 15 parts`);
  }

  const partIds = new Set();
  problem.parts.forEach((part) => {
    const where = `Theory problem ${problem.id} part ${part.id}`;
    if (!part.id) errors.push(`${where}: missing id`);
    if (partIds.has(part.id)) errors.push(`${where}: duplicate part id`);
    partIds.add(part.id);
    if (!part.requirement && !multipleChoice) errors.push(`${where}: missing grading instruction`);
    ["prompt", "answer"].forEach((field) => {
      if (!Array.isArray(part[field]) || !part[field].length) errors.push(`${where}: missing ${field} blocks`);
    });
    // An mcq part is only answerable if it actually lists choices.
    if (part.kind === "mcq" && !part.prompt.some((block) => Array.isArray(block.choices))) {
      errors.push(`${where}: kind is "mcq" but no choices block is present`);
    }
    // Choices are strings (A's papers, letters written in) or objects judged
    // in the page: { text, correct, explain }, exactly one correct, each with
    // its own explanation. An answer block can hold choices too, and the page
    // judges those the same way, so both are scanned.
    const choiceBlocks = [...(part.prompt || []), ...(part.answer || [])].filter((block) => Array.isArray(block.choices));
    choiceBlocks.forEach((block) => {
      const objects = block.choices.filter((choice) => choice && typeof choice === "object");
      if (objects.length && objects.length !== block.choices.length) {
        errors.push(`${where}: mixes string and object choices`);
        return;
      }
      if (!objects.length) {
        if (multipleChoice) errors.push(`${where}: a Multiple choice problem needs object choices { text, correct, explain }`);
        return;
      }
      if (objects.length < 2) errors.push(`${where}: needs at least two choices`);
      const correct = objects.filter((choice) => choice.correct === true).length;
      if (correct !== 1) errors.push(`${where}: needs exactly one choice with "correct": true (found ${correct})`);
      objects.forEach((choice, n) => {
        if (typeof choice.text !== "string" || !choice.text.trim()) errors.push(`${where}: choice ${n + 1} has no text`);
        if (typeof choice.explain !== "string" || !choice.explain.trim()) {
          errors.push(`${where}: choice ${n + 1} has no explain`);
        }
        if (/^[A-J][.)]\s/.test(choice.text || "")) {
          errors.push(`${where}: choice ${n + 1} starts with its letter; the page adds the letters`);
        }
        Object.keys(choice).forEach((field) => {
          if (!["text", "correct", "explain"].includes(field)) errors.push(`${where}: choice ${n + 1} has an unknown field "${field}"`);
        });
      });
    });
  });

  const registered = problemData.problems.find((entry) => entry.id === problem.id);
  if (!registered) {
    errors.push(`Theory problem ${problem.id} has no entry in content/problems.js`);
  } else if (registered.url !== `/theory/?id=${problem.id}`) {
    errors.push(`Theory problem ${problem.id} is registered with the wrong url: ${registered.url}`);
  }
});

problemData.problems
  .filter((problem) => problem.tags.includes("Theory"))
  .forEach((problem) => {
    if (!theoryIds.has(problem.id)) {
      errors.push(`Problem ${problem.id} is tagged Theory but has no data in content/theory.js`);
    }
  });

problemData.problems
  .filter((problem) => problem.tags.includes("Multiple choice") && !problem.tags.includes("Theory"))
  .forEach((problem) => errors.push(`Problem ${problem.id} is tagged Multiple choice but not Theory`));

/* -------------------------------- notebooks -------------------------------- */

const DIFFICULTIES = new Set(["Easy", "Medium", "Hard", "Very hard", "Very Hard"]);
const notebookDir = path.join(repoRoot, "notebooks");

function readNotebook(id) {
  const file = path.join(notebookDir, `${id}.ipynb`);
  if (!fs.existsSync(file)) return null;
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch (error) {
    errors.push(`notebooks/${id}.ipynb is not valid JSON: ${error.message}`);
    return null;
  }
}

const cellText = (cell) => (Array.isArray(cell.source) ? cell.source.join("") : String(cell.source || ""));

// Coding problems open notebooks/<id>.ipynb in the editor.
problemData.problems
  .filter((problem) => problem.tags.includes("Coding") && /^\/editor\/\?nb=/.test(problem.url || ""))
  .forEach((problem) => {
    const id = problem.url.replace(/^\/editor\/\?nb=/, "");
    if (!fs.existsSync(path.join(notebookDir, `${id}.ipynb`))) {
      errors.push(`Problem ${problem.id} opens notebooks/${id}.ipynb, which does not exist`);
    }
  });

if (fs.existsSync(notebookDir)) {
  fs.readdirSync(notebookDir)
    .filter((file) => file.endsWith(".ipynb"))
    .forEach((file) => {
      const id = file.replace(/\.ipynb$/, "");
      const notebook = readNotebook(id);
      if (!notebook) return;
    const cells = Array.isArray(notebook.cells) ? notebook.cells : [];
    // The worked solution sits at the end: a "## Solution" markdown cell and
    // only solution cells after it, each marked {"usaaio_solution": true}, so
    // the editor can fold all of it and ?embed=1 can strip all of it. The
    // boundary is found the same way build-content.mjs finds it: either signal
    // opens it, and the opener must carry both.
    const isMarked = (cell) => cell && cell.metadata && cell.metadata.usaaio_solution;
    const isSolutionStart = (cell) => cell && cell.cell_type === "markdown" && /^##\s+Solution\b/.test(cellText(cell).trim());
    const first = cells.findIndex((cell) => isMarked(cell) || isSolutionStart(cell));
    if (first >= 0) {
      const opener = cells[first];
      if (!isMarked(opener) || !isSolutionStart(opener)) {
        errors.push(`notebooks/${file}: the solution must open with a markdown cell "## Solution" marked usaaio_solution`);
      }
      cells.slice(first).forEach((cell, n) => {
        if (!isMarked(cell)) {
          errors.push(`notebooks/${file}: cell ${first + n} follows the solution but is not marked usaaio_solution`);
        }
      });
    }
    // Notebooks written to the merged convention carry full metadata.
    const meta = (notebook.metadata && notebook.metadata.usaaio) || {};
    if (meta.id === undefined) return;
    if (meta.id !== id) errors.push(`notebooks/${file}: metadata.usaaio.id is "${meta.id}", expected "${id}"`);
    ["title", "description"].forEach((field) => {
      if (typeof meta[field] !== "string" || !meta[field].trim()) errors.push(`notebooks/${file}: metadata.usaaio.${field} is missing`);
    });
    if (!moduleIds.has(meta.module)) errors.push(`notebooks/${file}: metadata.usaaio.module "${meta.module}" is not a module id`);
    if (meta.difficulty !== undefined && !DIFFICULTIES.has(meta.difficulty)) {
      errors.push(`notebooks/${file}: metadata.usaaio.difficulty "${meta.difficulty}" is not one of ${[...DIFFICULTIES].join(", ")}`);
    }
    if (meta.minutes !== undefined && !(Number.isInteger(meta.minutes) && meta.minutes > 0)) {
      errors.push(`notebooks/${file}: metadata.usaaio.minutes must be a whole number of minutes`);
    }
    if (meta.colab !== undefined && (typeof meta.colab !== "string" || !meta.colab.trim())) {
      errors.push(`notebooks/${file}: metadata.usaaio.colab must be a sentence saying why`);
    }
  });
} else {
  errors.push("notebooks/ does not exist, so no notebook was checked");
}

/* ----------------------------- lesson card checks --------------------------- */

const theoryById = new Map(theoryData.problems.map((problem) => [problem.id, problem]));
const problemById = new Map(problemData.problems.map((problem) => [problem.id, problem]));
const seenProblemCards = new Set();
const STALE = "content/lessons.js is out of date: run npm run build:content";

lessonCards.forEach(({ at, moduleId, block }) => {
  if (block.problemCard) {
    const { id, mcq } = block.problemCard;
    // Two cards for one problem share an anchor and a question key, so the
    // second one would be unreachable and its stored answer would conflict.
    const seenKey = `${moduleId}:${id}`;
    if (seenProblemCards.has(seenKey)) {
      errors.push(`${at}: <ProblemCard id="${id}"> appears twice in this lesson`);
    } else {
      seenProblemCards.add(seenKey);
    }
    const problem = problemById.get(id);
    if (!problem) {
      errors.push(`${at}: <ProblemCard id="${id}"> names no problem in content/problems.js`);
      return;
    }
    if (problem.moduleId !== moduleId) {
      errors.push(`${at}: <ProblemCard id="${id}"> belongs to module ${problem.moduleId}, not this one`);
    }
    const theory = theoryById.get(id);
    const asked = problem.tags.includes("Multiple choice") && theory && theory.parts.length === 1 && theory.parts[0].kind === "mcq";
    if (asked && JSON.stringify(mcq && mcq.prompt) !== JSON.stringify(theory.parts[0].prompt)) errors.push(`${at}: ${STALE}`);
    if (!asked && mcq) errors.push(`${at}: ${STALE}`);
  }
  if (block.notebookCard) {
    const card = block.notebookCard;
    const notebook = readNotebook(card.id);
    if (!notebook) {
      errors.push(`${at}: <NotebookCard id="${card.id}"> has no notebooks/${card.id}.ipynb`);
      return;
    }
    const meta = (notebook.metadata && notebook.metadata.usaaio) || {};
    if (!meta.title || !meta.description) {
      errors.push(`${at}: notebooks/${card.id}.ipynb needs metadata.usaaio.title and description for its card`);
    }
    if (meta.module && meta.module !== moduleId) {
      errors.push(`${at}: <NotebookCard id="${card.id}"> belongs to module ${meta.module}, not this one`);
    }
    if (card.title !== (meta.title || card.id) || (card.description || undefined) !== (meta.description || undefined)) {
      errors.push(`${at}: ${STALE}`);
    }
  }
});

problemData.contests.forEach((contest) => {
  if (!contest.problemIds || !contest.problemIds.length) {
    errors.push(`Contest ${contest.id} has no problems`);
  }
  contest.problemIds.forEach((problemId) => {
    if (!problemIds.has(problemId)) {
      errors.push(`Contest ${contest.id} references unknown problem ${problemId}`);
    }
  });
});

const problemBankDir = path.join(repoRoot, "content/problemBank");
fs.readdirSync(problemBankDir)
  .filter((file) => file.endsWith(".problems.json"))
  .forEach((file) => {
    const data = JSON.parse(fs.readFileSync(path.join(problemBankDir, file), "utf8"));
    if (!moduleIds.has(data.MODULE_ID)) {
      errors.push(`Problem bank ${file} references unknown MODULE_ID ${data.MODULE_ID}`);
    }
    if (!Array.isArray(data.practice)) {
      errors.push(`Problem bank ${file} is missing practice array`);
    }
  });

/* ------------------------ rendered-output presentation --------------------- */

// The readers support one small set of inline markup: `code`, [links](...),
// **bold**, *italic* and $math$. Anything else reaches the page as literal
// characters — the "*hello*" bug — so render every lesson block the way the
// module reader does and look for markers that survived.
const renderers = loadBrowserGlobal("src/components/renderers.js", "AIO_RENDERERS");

const LEFTOVER_MARKUP = [
  [/\*/, "literal * (emphasis did not render)"],
  [/`/, "literal backtick"],
  [/~~/, "literal ~~ (strikethrough is not supported)"],
  [/!\[/, "literal ![ (an image must sit alone on its line)"],
  [/\[[^\]]*\]\([^)\s]*\)/, "literal [text](url)"],
  [/(^|\s)_[^_\s][^_]*_(\s|$|[.,;:)])/, "literal _..._ (write *italic* instead)"],
  [/\|[^|]*\|/, "pipes in running text (the table did not parse)"],
  [/&[a-zA-Z]{2,10};/, "raw HTML entity"],
  [/<\/?(br|sub|sup|em|strong|b|i|u|span|div|img|a|p|h[1-6])\b/i, "raw HTML tag"],
  // Prose like "keep x <A" or "the loss stays <N" would match a bare "<Tag";
  // a real leftover component tag carries attributes or is closed.
  [/<\/?[A-Z][A-Za-z]*(\s[^>]*?)?\/?>/, "a component tag the lesson parser does not know"],
];
// Positions of the two patterns glossary prose shows literally (see checkGlossary).
const STAR_INDEX = 0;
const PIPE_INDEX = 6;

// Code, pre and math spans legitimately hold these characters; everything else
// is what a reader actually sees.
function visibleText(html) {
  return String(html)
    .replace(/<pre[\s\S]*?<\/pre>/g, " ")
    .replace(/<code[\s\S]*?<\/code>/g, " ")
    // A <EmbeddedPyEditor> cell puts its code in a textarea: code, not prose.
    .replace(/<textarea[\s\S]*?<\/textarea>/g, " ")
    .replace(/<span class="reader-math[\s\S]*?<\/span>/g, " ")
    .replace(/<[^>]+>/g, " ")
    .replaceAll("&amp;", "&")
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&quot;", '"')
    .replaceAll("&#039;", "'")
    .replace(/\s+/g, " ");
}

Object.entries(lessons.lessons).forEach(([moduleId, lesson]) => {
  // A problem card looks its problem up, so the sweep has to pass the same
  // options the page does, or the card renders empty and is never checked.
  const options = { problemLookup: problemById, appState: {} };
  (lesson.sections || []).forEach((section) => {
    (section.blocks || []).forEach((block, index) => {
      const text = visibleText(renderers.renderBlocks([block], "", options));
      LEFTOVER_MARKUP.forEach(([pattern, label]) => {
        const match = text.match(pattern);
        if (!match) return;
        const at = Math.max(0, text.indexOf(match[0]) - 40);
        presentation.push(`${moduleId} › ${section.title} block ${index}: ${label}: "…${text.slice(at, at + 110).trim()}…"`);
      });
    });
  });
});

// KaTeX gives up on a bare "_" or "^" inside \text{...} and paints the source
// red on the page, so catch those before they ship.
function checkTex(where, tex) {
  (String(tex).match(/\\text\{[^{}]*\}/g) || []).forEach((group) => {
    if (/(^|[^\\])[_^]/.test(group.slice(6, -1))) {
      presentation.push(`${where}: \\text{...} holds a bare _ or ^ that KaTeX cannot parse — "${group.slice(0, 70)}"`);
    }
  });
}

function collectStrings(block, sink) {
  if (!block || typeof block !== "object") return;
  if (block.p) sink.push(block.p);
  if (block.heading) sink.push(block.heading);
  if (block.tex) sink.push(`$${block.tex}$`);
  ["list", "olist", "quote"].forEach((key) => (block[key] || []).forEach((item) => sink.push(item)));
  if (block.table) {
    (block.table.head || []).forEach((cell) => sink.push(cell));
    (block.table.rows || []).forEach((row) => row.forEach((cell) => sink.push(cell)));
  }
  if (block.details) (block.details.blocks || []).forEach((inner) => collectStrings(inner, sink));
  if (block.optional) {
    if (block.optional.title) sink.push(block.optional.title);
    (block.optional.blocks || []).forEach((inner) => collectStrings(inner, sink));
  }
  if (block.figure && block.figure.caption) sink.push(block.figure.caption);
}

Object.entries(lessons.lessons).forEach(([moduleId, lesson]) => {
  (lesson.sections || []).forEach((section) => {
    const strings = [];
    (section.blocks || []).forEach((block) => collectStrings(block, strings));
    strings.forEach((value) => {
      const where = `${moduleId} › ${section.title}`;
      (String(value).match(/\$[^$]+\$/g) || []).forEach((math) => checkTex(where, math.slice(1, -1)));
    });
  });
});

// Summaries, problem titles and problem prompts are printed with escapeHtml and
// nothing else, so any markup written into them shows up as punctuation.
const PLAIN_TEXT_MARKUP = [
  [/\*\*[^*]+\*\*/, "**bold**"],
  [/(^|[^*\w])\*[^*\s][^*]*\*/, "*italic*"],
  [/`[^`]+`/, "`code`"],
  [/\[[^\]]+\]\([^)]+\)/, "[text](url)"],
];

function checkPlainText(where, value) {
  if (typeof value !== "string" || !value) return;
  PLAIN_TEXT_MARKUP.forEach(([pattern, label]) => {
    if (pattern.test(value.replace(/\$[^$]*\$/g, " "))) {
      presentation.push(`${where}: ${label} is shown literally — this field is rendered as plain text`);
    }
  });
}

content.modules.forEach((module) => checkPlainText(`catalog ${module.id} summary`, module.summary));
problemData.problems.forEach((problem) => {
  ["title", "prompt", "answer", "source"].forEach((field) => checkPlainText(`problem ${problem.id} ${field}`, problem[field]));
});

/* --------------------------------- glossary -------------------------------- */

// content/glossary.js sets window.AIO_GLOSSARY = [{ id, term, aka, definition,
// moduleId, tags }]. moduleId is an A module id, or null when A has none yet.
function checkGlossary() {
  if (!fs.existsSync(path.join(repoRoot, "content/glossary.js"))) return;
  const glossary = loadBrowserGlobal("content/glossary.js", "AIO_GLOSSARY");
  if (!Array.isArray(glossary)) {
    errors.push("content/glossary.js must set window.AIO_GLOSSARY to an array");
  } else {
    const seen = new Set();
    const terms = new Set();
    glossary.forEach((entry, n) => {
      const where = `content/glossary.js entry ${n + 1}${entry && entry.term ? ` (${entry.term})` : ""}`;
      if (!entry || typeof entry !== "object") {
        errors.push(`${where}: not an object`);
        return;
      }
      if (typeof entry.id !== "string" || !/^[a-z0-9][a-z0-9-]*$/.test(entry.id)) errors.push(`${where}: id must be a lower-case slug`);
      else if (seen.has(entry.id)) errors.push(`${where}: duplicate id ${entry.id}`);
      seen.add(entry.id);
      if (typeof entry.term !== "string" || !entry.term.trim()) errors.push(`${where}: missing term`);
      else if (terms.has(entry.term.trim().toLowerCase())) errors.push(`${where}: the term appears twice`);
      else terms.add(entry.term.trim().toLowerCase());
      if (typeof entry.definition !== "string" || !entry.definition.trim()) errors.push(`${where}: missing definition`);
      if (typeof entry.aka === "string" && entry.aka.trim().toLowerCase() === String(entry.term).trim().toLowerCase()) {
        errors.push(`${where}: aka repeats the term`);
      }
      if (entry.aka !== undefined && entry.aka !== null && (typeof entry.aka !== "string" || !entry.aka.trim())) {
        errors.push(`${where}: aka must be a string or null`);
      }
      if (entry.moduleId !== null && !moduleIds.has(entry.moduleId)) {
        errors.push(`${where}: moduleId "${entry.moduleId}" is not a module id (use null when there is none)`);
      }
      if (!Array.isArray(entry.tags)) {
        errors.push(`${where}: tags must be an array`);
      } else {
        entry.tags.forEach((tag) => {
          if (typeof tag !== "string" || !tag.trim()) errors.push(`${where}: tags must be non-empty strings`);
        });
      }
      ["term", "aka", "definition"].forEach((field) => {
        if (typeof entry[field] !== "string") return;
        const source = entry[field];
        const text = visibleText(renderers.inline(source, ""));
        // Glossary prose is plain text, where P(A|B) and a lone * (as in
        // theta*) are meant literally, so those two checks do not apply. The
        // exemptions are picked by position, not by label text, so rewording a
        // label cannot silently bring a check back.
        const exempt = new Set([STAR_INDEX, PIPE_INDEX]);
        LEFTOVER_MARKUP.forEach(([pattern, label], index) => {
          if (exempt.has(index)) return;
          if (pattern.test(text)) errors.push(`${where}: ${field} shows ${label}`);
        });
        // Math is expected in a definition, so it is parse-checked too.
        (source.match(/\$[^$]+\$/g) || []).forEach((math) => checkTex(where, math.slice(1, -1)));
      });
    });
  }
}

checkGlossary();

presentation.forEach((issue) => errors.push(`raw markup reaches the reader — ${issue}`));

if (errors.length) {
  console.error("Content validation failed:");
  errors.forEach((error) => console.error(`- ${error}`));
  process.exit(1);
}

if (unconverted.length) {
  console.warn(`${unconverted.length} module(s) not yet converted to the linear format:`);
  unconverted.forEach((warning) => console.warn(`- ${warning}`));
}

console.log(
  `Content validation passed for ${content.modules.length} modules, ${problemData.problems.length} problems (${theoryData.problems.length} theory, ${theoryData.problems.reduce((sum, p) => sum + p.parts.length, 0)} parts), and ${problemData.contests.length} contests.`,
);
