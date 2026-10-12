/**
 * Builds content/catalog.js, content/ordering.js and content/lessons.js from the
 * Markdown module set under content/{1_Round_1,2_Round_2,3_Camp,4_Advanced}, and
 * content/problems.js and content/theory.js from the per-module files under
 * content/problems/.
 *
 * The Markdown files are the source of truth. Every module follows the same
 * ten-section shape (see content/README.md), so the parser can be strict: it
 * throws rather than guessing when a file does not match.
 *
 * Run with `npm run build:content`.
 */
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";
import { assembleProblems, assembleTheory, writeContentFiles } from "./lib/assemble-problems.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

/* ------------------------------- stage setup ------------------------------ */

const STAGES = [
  {
    "id": "round1",
    "label": "Round 1",
    "dir": "content/1_Round_1",
    "route": "round1/",
    "level": "Core",
    "seoTitle": "Round 1 for USA AI Olympiad",
    "description": "Python, mathematics, probability, machine learning, and computer vision."
  },
  {
    "id": "round2",
    "label": "Round 2",
    "dir": "content/2_Round_2",
    "route": "round2/",
    "level": "Core",
    "seoTitle": "Round 2 for AIO",
    "description": "Backpropagation, model evaluation, computer vision, and transformers."
  },
  {
    "id": "camp",
    "label": "Camp",
    "dir": "content/3_Camp",
    "route": "camp/",
    "level": "Advanced",
    "seoTitle": "AIO Camp Preparation",
    "description": "Information theory, vision, NLP, generative models, and representation learning."
  },
  {
    "id": "advanced",
    "label": "Advanced",
    "dir": "content/4_Advanced",
    "route": "advanced/",
    "level": "Advanced",
    "seoTitle": "Advanced AIO Topics",
    "description": "Long-context attention, multimodal models, reinforcement learning, and graph networks."
  }
];

// Chapters group the modules of a stage by topic rather than by file number, so
// a stage page reads as "Math Foundations, Classical Machine Learning, ..."
// instead of a flat list. The structure lives in content/chapters.json, which
// scripts/import-b/chapters-from-map.mjs generates from MERGE-MAP.json; that
// keeps the whole final structure in one committed file so a module branch that
// lands a lesson file never has to edit it. Every on-disk module must appear in
// exactly one chapter of its own stage (checked below); a module listed before
// its lesson file exists is skipped with a note instead of failing the build.
const CHAPTERS = loadChapters();

// How often each module's topic has come up on past papers, 0 to 4, in
// usaco.guide's sense ("Has Not Appeared" up to "Very Frequent"). The levels
// are set by hand from the past papers in content/frequency.json, keyed by
// module id. A module missing from the file carries no label.
const FREQUENCY = loadFrequency();

function loadFrequency() {
  const file = path.join(repoRoot, "content/frequency.json");
  if (!fs.existsSync(file)) return {};
  const levels = JSON.parse(fs.readFileSync(file, "utf8"));
  const bad = Object.entries(levels).filter(([, level]) => !Number.isInteger(level) || level < 0 || level > 4);
  if (bad.length) throw new Error(`content/frequency.json: levels must be 0 to 4 (${bad.map(([id]) => id).join(", ")})`);
  return levels;
}

function loadChapters() {
  const file = path.join(repoRoot, "content/chapters.json");
  if (!fs.existsSync(file)) {
    throw new Error(
      "content/chapters.json is missing; generate it with scripts/import-b/chapters-from-map.mjs",
    );
  }
  const chapters = JSON.parse(fs.readFileSync(file, "utf8"));
  const problems = [];
  STAGES.forEach((stage) => {
    const list = chapters[stage.id];
    if (!Array.isArray(list)) {
      problems.push(`stage "${stage.id}" has no chapter list`);
      return;
    }
    list.forEach((chapter) => {
      if (typeof chapter.name !== "string") problems.push(`a chapter in ${stage.id} has no name`);
      if (!Array.isArray(chapter.items)) problems.push(`chapter ${chapter.name} in ${stage.id} has no items list`);
    });
  });
  if (problems.length) throw new Error(`Malformed content/chapters.json: ${problems.join("; ")}`);
  return chapters;
}

// Filenames encode "<number>-<subject code>-<slug>". The slug is the module id
// except where a longer name reads better in the sidebar.
const ID_OVERRIDES = {};

/* ------------------------------ lesson shape ------------------------------ */

// A module is a linear lesson: after the two metadata sections below, every
// "## " heading is a topic that becomes one reader section, in document order.
// There is no fixed section vocabulary — a module names its own topics.
//
// These headings are structural rather than lesson content:
//   Learning Outcomes  -> lifted into the catalog, never a reader section
//   Prerequisites      -> a reader section, but hoisted by the reader into a
//                         box under the title instead of appearing in the flow
//   Practice Problems  -> dropped; the module page ends with that module's
//                         slice of content/problems.js instead
const PREREQ_HEADING = "Prerequisites";
const PREREQ_ID = "prerequisites";
const DROPPED_HEADINGS = new Set(["Practice Problems", "Learning Outcomes"]);

// The lead-in. Markdown needs a heading to delimit the section, but the reader
// prints this one without a title so the module opens straight into prose the
// way usaco.guide does.
const LEAD_HEADING = "Overview";
const LEAD_ID = "intro";

function slugify(text) {
  return text
    .toLowerCase()
    .replace(/\$[^$]*\$/g, " ") // inline math never reads well in an anchor
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

// The slug the lesson reader gives a heading id (slugify in
// src/components/renderers.js), kept in step so built anchors match the page.
function readerSlug(value) {
  return (
    String(value ?? "")
      .toLowerCase()
      .replace(/&/g, " and ")
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "") || "section"
  );
}

/* --------------------------------- parsing -------------------------------- */

function splitByHeading(lines, level) {
  // Fence-aware: a "# comment" inside a Python block is not a heading.
  const marker = `${"#".repeat(level)} `;
  const groups = [];
  let current = null;
  let fenced = false;

  // An <Optional> box is part of the section it sits in, so a "## " line
  // inside one would cut the box in two. Say so rather than fail later.
  let optional = 0;
  lines.forEach((line) => {
    if (line.startsWith("```")) fenced = !fenced;
    if (!fenced && /^<Optional(?=[\s>]|$)/.test(line.trim())) optional += 1;
    if (!fenced && line.trim().startsWith("</Optional>")) optional = Math.max(0, optional - 1);
    if (!fenced && optional && line.startsWith(marker)) {
      throw new Error(`"${line.trim()}" is inside an <Optional>; use ### for headings inside it`);
    }
    if (!fenced && line.startsWith(marker)) {
      current = { heading: line.slice(marker.length).trim(), lines: [] };
      groups.push(current);
      return;
    }
    if (current) current.lines.push(line);
    else groups.push((current = { heading: null, lines: [line] }));
  });

  return groups;
}

function isTableRow(line) {
  return line.startsWith("|") && line.endsWith("|");
}

function isTableDivider(line) {
  return isTableRow(line) && /^\|[\s:|-]+\|$/.test(line);
}

function splitTableRow(line) {
  // A "|" inside inline math is an absolute value, a norm or a conditional bar
  // ($|y-\hat y|$, $\frac{1}{|A||B|}$, $P(a \mid b)$), not a cell separator.
  const body = line.slice(1, -1);
  const cells = [];
  let current = "";
  let inMath = false;
  for (let i = 0; i < body.length; i += 1) {
    const char = body[i];
    if (char === "\\") {
      current += char + (body[i + 1] || "");
      i += 1;
      continue;
    }
    if (char === "$") inMath = !inMath;
    if (char === "|" && !inMath) {
      cells.push(current);
      current = "";
      continue;
    }
    current += char;
  }
  cells.push(current);
  return cells.map((cell) => cell.trim());
}

/* --------------------------- lesson components ---------------------------- */

// The lessons use a few component tags, each on its own lines with a blank
// line around it:
//   <Figure src="..." alt="...">caption</Figure>
//   <Optional title="...">markdown</Optional>
//   <Warning title="...">markdown</Warning>
//   <Info title="...">markdown</Info>
//   <EmbeddedPyEditor packages="numpy,matplotlib">python code</EmbeddedPyEditor>
//   <ProblemCard id="..." />
//   <NotebookCard id="..." />
//   <Resources><Resource source="..." title="..." url="..." starred /></Resources>
// Each becomes one block of its own type in lessons.js. A tag that does not
// parse throws, like the rest of this file, rather than leaking into the page.
const COMPONENT_TAG = /^<(Figure|Optional|Warning|Info|EmbeddedPyEditor|ProblemCard|NotebookCard|Resources)(?=[\s/>]|$)/;

// Reads one opening tag starting at lines[index], which may run over several
// lines. Quote-aware, so a ">" inside alt text does not end the tag.
function readTag(lines, index, where) {
  let text = "";
  let cursor = index;
  while (cursor < lines.length) {
    text += (text ? " " : "") + lines[cursor].trim();
    let quoted = false;
    for (let i = 0; i < text.length; i += 1) {
      if (text[i] === '"') quoted = !quoted;
      if (text[i] === ">" && !quoted) {
        const inner = text.slice(1, i);
        const selfClosing = inner.endsWith("/");
        const name = inner.match(/^[A-Za-z]+/)[0];
        return {
          name,
          attrs: parseAttributes(inner.slice(name.length, selfClosing ? -1 : undefined), where),
          selfClosing,
          rest: text.slice(i + 1).trim(),
          next: cursor + 1,
        };
      }
    }
    cursor += 1;
  }
  throw new Error(`${where}: a component tag is never closed with ">"`);
}

function parseAttributes(source, where) {
  if (/=\s*'/.test(source)) throw new Error(`${where}: write attribute values in double quotes`);
  const attrs = {};
  const pattern = /\s*([A-Za-z][\w-]*)(?:\s*=\s*"([^"]*)")?/y;
  let cursor = 0;
  const body = source.trim();
  while (cursor < body.length) {
    pattern.lastIndex = cursor;
    const match = pattern.exec(body);
    if (!match || !match[0].trim()) throw new Error(`${where}: cannot read the attributes in "${body}"`);
    attrs[match[1]] = match[2] === undefined ? true : match[2];
    cursor = pattern.lastIndex;
  }
  return attrs;
}

// The body of a block component up to its closing tag, fence-aware so a tag
// written inside a code example does not close it. Returns the inner lines and
// the index after the closing line. Text before the closing tag on its line is
// kept; text after it is an error.
function readUntilClosing(lines, start, name, firstRest, where) {
  const closing = `</${name}>`;
  const opening = new RegExp(`^<${name}(?=[\\s>]|$)`);
  const body = [];
  if (firstRest) {
    if (firstRest.includes(closing)) {
      const [before, after] = firstRest.split(closing);
      if (after.trim()) throw new Error(`${where}: text after ${closing}`);
      return { body: before.trim() ? [before] : [], next: start };
    }
    body.push(firstRest);
  }
  let depth = 0;
  let fenced = false;
  for (let cursor = start; cursor < lines.length; cursor += 1) {
    const line = lines[cursor];
    const trimmed = line.trim();
    if (trimmed.startsWith("```")) fenced = !fenced;
    // A line indented by four spaces is a code block in Markdown, so a tag
    // written there is an example, not a nested opening or a closer.
    const code = line.startsWith("    ");
    if (!fenced && !code && opening.test(trimmed) && !trimmed.endsWith("/>")) depth += 1;
    if (!fenced && !code && trimmed.includes(closing)) {
      if (depth === 0) {
        const [before, after] = lines[cursor].split(closing);
        if (after.trim()) throw new Error(`${where}: text after ${closing}`);
        if (before.trim()) body.push(before);
        return { body, next: cursor + 1 };
      }
      depth -= 1;
    }
    body.push(lines[cursor]);
  }
  throw new Error(`${where}: <${name}> has no ${closing}`);
}

// Display size of a figure in CSS pixels, so the page reserves its space
// before it loads and never scales a figure up past the size it was drawn
// at. SVGs carry it in width/height (or the viewBox); PNGs are drawn at 2x.
function figureSize(src) {
  const file = path.join(repoRoot, src.replace(/^\//, ""));
  if (!fs.existsSync(file)) return null;
  if (/\.svg$/i.test(file)) {
    const head = fs.readFileSync(file, "utf8").slice(0, 4000);
    const tag = (head.match(/<svg\b[^>]*>/i) || [""])[0];
    const number = (name) => {
      const match = tag.match(new RegExp(`\\s${name}="([\\d.]+)(px)?"`));
      return match ? Number(match[1]) : null;
    };
    let width = number("width");
    let height = number("height");
    const viewBox = tag.match(/viewBox="([^"]+)"/);
    if ((!width || !height) && viewBox) {
      const [, , vw, vh] = viewBox[1].trim().split(/[\s,]+/).map(Number);
      if (!width && !height) [width, height] = [vw, vh];
      else if (!height) height = (width * vh) / vw;
      else width = (height * vw) / vh;
    }
    return width && height ? { width: Math.round(width), height: Math.round(height) } : null;
  }
  if (/\.png$/i.test(file)) {
    const header = fs.readFileSync(file).subarray(0, 24);
    if (header.toString("latin1", 12, 16) !== "IHDR") return null;
    return { width: Math.round(header.readUInt32BE(16) / 2), height: Math.round(header.readUInt32BE(20) / 2) };
  }
  return null;
}

// The facts a <NotebookCard> shows, read from the notebook's own metadata so
// the card and the editor never disagree. Parts are the guided notebook's
// "## Part N" headings before the solution.
function notebookFacts(id, where) {
  const file = path.join(repoRoot, "notebooks", `${id}.ipynb`);
  if (!fs.existsSync(file)) throw new Error(`${where}: notebooks/${id}.ipynb does not exist`);
  let notebook;
  try {
    notebook = JSON.parse(fs.readFileSync(file, "utf8"));
  } catch (error) {
    throw new Error(`notebooks/${id}.ipynb is not valid JSON: ${error.message}`);
  }
  const meta = (notebook.metadata && notebook.metadata.usaaio) || {};
  let parts = 0;
  for (const cell of notebook.cells || []) {
    const source = Array.isArray(cell.source) ? cell.source.join("") : String(cell.source || "");
    if (cell.metadata && cell.metadata.usaaio_solution) break;
    if (cell.cell_type === "markdown" && /^##\s+Solution\b/.test(source.trim())) break;
    if (cell.cell_type === "markdown") parts += (source.match(/^##\s+Part\s+\d+/gm) || []).length;
  }
  const facts = { id, title: meta.title || id };
  if (meta.description) facts.description = meta.description;
  if (parts) facts.parts = parts;
  if (meta.minutes) facts.minutes = meta.minutes;
  if (meta.difficulty) facts.difficulty = meta.difficulty;
  return facts;
}

// A multiple-choice problem is asked inside the lesson, so its one part (the
// question and its options) is copied into the block. content/theory.js is
// 300 KB and loads only on the theory page; the module page needs one part.
function problemCardFacts(id, context, where) {
  const problem = context.problems.get(id);
  if (!problem) throw new Error(`${where}: <ProblemCard id="${id}"> is not in content/problems.js`);
  const theory = context.theory.get(id);
  const card = { id };
  if (
    problem &&
    theory &&
    (problem.tags || []).includes("Multiple choice") &&
    Array.isArray(theory.parts) &&
    theory.parts.length === 1 &&
    theory.parts[0].kind === "mcq"
  ) {
    card.mcq = { prompt: theory.parts[0].prompt };
  }
  return card;
}

function parseComponent(lines, index, context) {
  const where = `${context.filename || "lesson"} line ~${index + 1}`;
  const tag = readTag(lines, index, where);
  const { name, attrs } = tag;

  if (name === "ProblemCard" || name === "NotebookCard") {
    if (!tag.selfClosing || tag.rest) throw new Error(`${where}: <${name} id="..." /> must close itself on its own line`);
    if (typeof attrs.id !== "string" || !attrs.id) throw new Error(`${where}: <${name}> needs an id`);
    const block = name === "ProblemCard"
      ? { problemCard: problemCardFacts(attrs.id, context, where) }
      : { notebookCard: notebookFacts(attrs.id, where) };
    return { block, next: tag.next };
  }

  if (name === "Figure") {
    if (typeof attrs.src !== "string" || !attrs.src) throw new Error(`${where}: <Figure> needs a src`);
    let caption = "";
    let next = tag.next;
    if (!tag.selfClosing) {
      const body = readUntilClosing(lines, tag.next, "Figure", tag.rest, where);
      caption = body.body.map((line) => line.trim()).filter(Boolean).join(" ");
      next = body.next;
    }
    const figure = { src: attrs.src, alt: typeof attrs.alt === "string" ? attrs.alt.trim() : "" };
    if (caption) figure.caption = rewriteLinks(caption, context);
    const size = figureSize(attrs.src);
    if (size) Object.assign(figure, size);
    return { block: { figure }, next };
  }

  if (name === "Optional") {
    if (tag.selfClosing) throw new Error(`${where}: <Optional> needs a body`);
    const body = readUntilClosing(lines, tag.next, "Optional", tag.rest, where);
    const blocks = parseBlocks(body.body, context);
    if (!blocks.length) throw new Error(`${where}: <Optional> is empty`);
    return {
      block: { optional: { title: typeof attrs.title === "string" ? attrs.title.trim() : "", blocks } },
      next: body.next,
    };
  }

  // <Warning> and <Info>: a short callout with a bold label and a markdown body.
  // Warning's title is optional (the label is then just "Warning"); Info's is
  // required, as in B's lessons.
  if (name === "Warning" || name === "Info") {
    if (tag.selfClosing) throw new Error(`${where}: <${name}> needs a body`);
    const title = typeof attrs.title === "string" ? attrs.title.trim() : "";
    if (name === "Info" && !title) throw new Error(`${where}: <Info> needs a title`);
    const body = readUntilClosing(lines, tag.next, name, tag.rest, where);
    const blocks = parseBlocks(body.body, context);
    if (!blocks.length) throw new Error(`${where}: <${name}> is empty`);
    const label = name === "Warning" ? (title ? `Warning: ${title}` : "Warning") : title;
    return {
      block: { callout: { variant: name === "Warning" ? "warning" : "info", label, blocks } },
      next: body.next,
    };
  }

  // <EmbeddedPyEditor>: a Python cell the reader can edit and run, using the
  // same worker as the notebook editor. The code is the tag body, verbatim;
  // B's initialCode template literal becomes plain lines. `packages` is a
  // comma-separated list; B's `check` assertion has no equivalent here and is
  // reported rather than silently dropped.
  if (name === "EmbeddedPyEditor") {
    if (tag.selfClosing) throw new Error(`${where}: <EmbeddedPyEditor> needs a code body`);
    const known = new Set(["packages"]);
    Object.keys(attrs).forEach((attr) => {
      if (!known.has(attr)) {
        console.log(`Note: ${where}: <EmbeddedPyEditor> attribute "${attr}" is not supported and was dropped.`);
      }
    });
    const body = readUntilClosing(lines, tag.next, "EmbeddedPyEditor", tag.rest, where);
    const code = body.body.join("\n").trimEnd();
    if (!code.trim()) throw new Error(`${where}: <EmbeddedPyEditor> has no code`);
    const packages = typeof attrs.packages === "string"
      ? attrs.packages.split(",").map((part) => part.trim()).filter(Boolean)
      : [];
    const block = { pyEditor: { code } };
    if (packages.length) block.pyEditor.packages = packages;
    return { block, next: body.next };
  }


  if (tag.selfClosing) throw new Error(`${where}: <Resources> needs <Resource /> entries`);
  const body = readUntilClosing(lines, tag.next, "Resources", tag.rest, where);
  const resources = [];
  const inner = body.body.map((line) => line.trim()).filter(Boolean);
  let cursor = 0;
  while (cursor < inner.length) {
    if (!/^<Resource(?=[\s/>]|$)/.test(inner[cursor])) {
      throw new Error(`${where}: only <Resource /> may sit inside <Resources>, found "${inner[cursor].slice(0, 60)}"`);
    }
    const entry = readTag(inner, cursor, where);
    if (!entry.selfClosing || entry.rest) throw new Error(`${where}: write each resource as <Resource ... />`);
    const { source, title, url, starred } = entry.attrs;
    if (typeof url !== "string" || !/^(https?:\/\/|\/|#)/.test(url)) {
      throw new Error(`${where}: <Resource> "${title || "?"}" has no valid url`);
    }
    if (typeof title !== "string" || !title.trim()) throw new Error(`${where}: <Resource> ${url} has no title`);
    const resource = { title: title.trim(), url };
    if (typeof source === "string" && source.trim()) resource.source = source.trim();
    if (starred) resource.starred = true;
    resources.push(resource);
    cursor = entry.next;
  }
  if (!resources.length) throw new Error(`${where}: <Resources> is empty`);
  return { block: { resources }, next: body.next };
}

/**
 * Turns the body of one "## " section into an ordered block list. Blocks are
 * the units the reader renders: paragraph, display math, list, table, code,
 * printed output, sub-heading, quote, a collapsible answer, or one of the
 * lesson components above.
 */
function parseBlocks(rawLines, context) {
  const lines = rawLines.map((line) => line.replace(/\s+$/, ""));
  const blocks = [];
  let index = 0;

  const flushParagraph = (buffer) => {
    if (!buffer.length) return;
    blocks.push({ p: rewriteLinks(buffer.join(" ").trim(), context) });
  };

  let paragraph = [];

  while (index < lines.length) {
    const line = lines[index];
    const trimmed = line.trim();

    // Blank line closes the running paragraph.
    if (!trimmed) {
      flushParagraph(paragraph);
      paragraph = [];
      index += 1;
      continue;
    }

    // Lesson components.
    if (COMPONENT_TAG.test(trimmed)) {
      flushParagraph(paragraph);
      paragraph = [];
      const parsed = parseComponent(lines, index, context);
      blocks.push(parsed.block);
      index = parsed.next;
      continue;
    }

    // A capitalised tag on its own line is meant to be a component. Anything
    // else there is a typo or an unconverted import, and would otherwise leak
    // into the page as text.
    if (/^<[A-Z][A-Za-z]*($|\s|>|\/)/.test(trimmed)) {
      throw new Error(`${context.filename || "lesson"} line ~${index + 1}: unknown component tag "${trimmed.split(">")[0]}>"`);
    }

    // HTML comments are authoring markers (the figure pipeline writes one
    // before each generated image) and never reach the reader.
    if (trimmed.startsWith("<!--")) {
      flushParagraph(paragraph);
      paragraph = [];
      while (index < lines.length && !lines[index].includes("-->")) index += 1;
      index += 1;
      continue;
    }

    // Horizontal rules only separate the "## " sections, so they carry no content.
    if (/^-{3,}$/.test(trimmed)) {
      flushParagraph(paragraph);
      paragraph = [];
      index += 1;
      continue;
    }

    // Sub-headings.
    const headingMatch = trimmed.match(/^(#{3,6})\s+(.*)$/);
    if (headingMatch) {
      flushParagraph(paragraph);
      paragraph = [];
      blocks.push({
        heading: rewriteLinks(headingMatch[2].trim(), context),
        level: Math.min(headingMatch[1].length, 4),
      });
      index += 1;
      continue;
    }

    // Standalone images.
    const imageMatch = trimmed.match(/^!\[([^\]]*)\]\(([^)]+)\)$/);
    if (imageMatch) {
      flushParagraph(paragraph);
      paragraph = [];
      blocks.push({ image: { alt: imageMatch[1].trim(), src: imageMatch[2].trim() } });
      index += 1;
      continue;
    }

    // Fenced code.
    if (trimmed.startsWith("```")) {
      flushParagraph(paragraph);
      paragraph = [];
      const language = trimmed.slice(3).trim() || "text";
      const body = [];
      index += 1;
      while (index < lines.length && !lines[index].trim().startsWith("```")) {
        body.push(lines[index]);
        index += 1;
      }
      index += 1; // closing fence
      while (body.length && !body[body.length - 1].trim()) body.pop();
      // A fence marked "output" is what the code block above it prints.
      if (language === "output") {
        if (body.length) blocks.push({ output: { lines: body } });
      }
      else blocks.push({ code: { language, lines: body } });
      continue;
    }

    // Display math, either "$$ ... $$" on one line or fenced across several.
    if (trimmed.startsWith("$$")) {
      flushParagraph(paragraph);
      paragraph = [];
      if (trimmed.length > 4 && trimmed.endsWith("$$")) {
        blocks.push({ tex: trimmed.slice(2, -2).trim() });
        index += 1;
        continue;
      }
      const body = [trimmed.slice(2)];
      index += 1;
      while (index < lines.length && !lines[index].trim().endsWith("$$")) {
        body.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) {
        body.push(lines[index].trim().replace(/\$\$$/, ""));
        index += 1;
      }
      blocks.push({ tex: body.join("\n").trim() });
      continue;
    }

    // Collapsible answer (Practice Problems use <details><summary>Answer</summary>).
    if (trimmed.startsWith("<details>")) {
      flushParagraph(paragraph);
      paragraph = [];
      const body = [];
      let summary = "Answer";
      index += 1;
      while (index < lines.length && !lines[index].trim().startsWith("</details>")) {
        const inner = lines[index].trim();
        const summaryMatch = inner.match(/^<summary>(.*)<\/summary>$/);
        if (summaryMatch) summary = summaryMatch[1].trim();
        else body.push(lines[index]);
        index += 1;
      }
      index += 1; // </details>
      blocks.push({ details: { summary, blocks: parseBlocks(body, context) } });
      continue;
    }

    // Tables.
    if (isTableRow(trimmed) && index + 1 < lines.length && isTableDivider(lines[index + 1].trim())) {
      flushParagraph(paragraph);
      paragraph = [];
      const head = splitTableRow(trimmed).map((cell) => rewriteLinks(cell, context));
      index += 2;
      const rows = [];
      while (index < lines.length && isTableRow(lines[index].trim())) {
        rows.push(splitTableRow(lines[index].trim()).map((cell) => rewriteLinks(cell, context)));
        index += 1;
      }
      blocks.push({ table: { head, rows } });
      continue;
    }

    // Blockquote.
    if (trimmed.startsWith(">")) {
      flushParagraph(paragraph);
      paragraph = [];
      const body = [];
      while (index < lines.length && lines[index].trim().startsWith(">")) {
        body.push(lines[index].trim().replace(/^>\s?/, ""));
        index += 1;
      }
      blocks.push({ quote: body.filter(Boolean).map((entry) => rewriteLinks(entry, context)) });
      continue;
    }

    // Lists. Continuation lines are indented or bare, and a blank line inside a
    // list ends it — which matches how the source files are written.
    const bulletMatch = trimmed.match(/^[-*]\s+(.*)$/);
    const orderedMatch = trimmed.match(/^\d+\.\s+(.*)$/);
    if (bulletMatch || orderedMatch) {
      flushParagraph(paragraph);
      paragraph = [];
      const ordered = Boolean(orderedMatch);
      const items = [];
      while (index < lines.length) {
        const candidate = lines[index].trim();
        if (!candidate) break;
        const nextBullet = candidate.match(/^[-*]\s+(.*)$/);
        const nextOrdered = candidate.match(/^\d+\.\s+(.*)$/);
        if (nextBullet || nextOrdered) {
          if (Boolean(nextOrdered) !== ordered) break;
          items.push((nextBullet || nextOrdered)[1]);
        } else if (items.length) {
          items[items.length - 1] += ` ${candidate}`;
        } else {
          break;
        }
        index += 1;
      }
      blocks.push({
        [ordered ? "olist" : "list"]: items.map((item) => rewriteLinks(item, context)),
      });
      continue;
    }

    paragraph.push(trimmed);
    index += 1;
  }

  flushParagraph(paragraph);
  return blocks;
}

/**
 * Cross-module Markdown links ("[2.7 RP1](2.7-RP1-representation-learning.md)")
 * become site routes so the reader can follow them. Anything else is left alone.
 */
function rewriteLinks(text, context) {
  return text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, label, href) => {
    if (/^(https?:|mailto:|#|\/)/.test(href)) return match;
    const moduleId = context.idByFilename.get(path.basename(href.split("#")[0]));
    if (!moduleId) return label;
    return `[${label}](/module/?id=${moduleId})`;
  });
}

/* --------------------------------- module --------------------------------- */

function parseModule(stage, filename, sharedContext) {
  const context = { ...sharedContext, filename };
  const source = fs.readFileSync(path.join(repoRoot, stage.dir, filename), "utf8");
  const lines = source.split("\n");

  const titleLine = lines.find((line) => line.startsWith("# "));
  if (!titleLine) throw new Error(`${filename}: no H1 title`);
  const titleMatch = titleLine.slice(2).match(/^([\d.]+)\s+·\s+([A-Z]{2}\d{1,2})\s+(?:·|\u2014)\s+(.*)$/);
  if (!titleMatch) throw new Error(`${filename}: H1 is not "<n> · <CODE> · <title>"`);
  const [, number, code, title] = titleMatch;

  const goalLine = lines.find((line) => line.startsWith("> **Goal:**"));
  if (!goalLine) throw new Error(`${filename}: no "> **Goal:**" line`);
  const summaryRaw = goalLine.replace("> **Goal:**", "").trim().replace(/\s+/g, " ");
  const summary = summaryRaw.charAt(0).toUpperCase() + summaryRaw.slice(1);

  const groups = splitByHeading(lines, 2);
  const byHeading = new Map(groups.filter((group) => group.heading).map((g) => [g.heading, g.lines]));

  const prerequisites = [];
  (byHeading.get("Prerequisites") || []).forEach((line) => {
    const match = line.trim().match(/^[-*]\s+\[[^\]]+\]\(([^)]+)\)/);
    if (!match) return;
    const moduleId = context.idByFilename.get(path.basename(match[1].split("#")[0]));
    if (moduleId && !prerequisites.includes(moduleId)) prerequisites.push(moduleId);
  });

  // Topics in document order. Duplicate or math-only headings would collide on
  // their slug, so ids are de-duplicated with a numeric suffix.
  const usedIds = new Set();
  const sections = [];
  groups.forEach((group, groupIndex) => {
    if (!group.heading) return; // preamble above the first "## "
    if (DROPPED_HEADINGS.has(group.heading)) return;

    // The files close with a "**Previous:** ... **Next:** ..." line for reading
    // them directly on disk. The reader builds its own prev/next cards from the
    // chapter ordering, which is not the file-number ordering, so drop it.
    const blocks = parseBlocks(group.lines, context).filter(
      (block) => !(block.p && /^\*\*(Previous|Next):\*\*/.test(block.p)),
    );
    if (!blocks.length) throw new Error(`${filename}: "## ${group.heading}" is empty`);

    const isLead = group.heading === LEAD_HEADING;
    let id = isLead ? LEAD_ID : group.heading === PREREQ_HEADING ? PREREQ_ID : slugify(group.heading);
    if (!id) id = `section-${groupIndex}`;
    if (usedIds.has(id)) {
      let suffix = 2;
      while (usedIds.has(`${id}-${suffix}`)) suffix += 1;
      id = `${id}-${suffix}`;
    }
    usedIds.add(id);

    // An empty title tells the reader to print the body with no heading.
    sections.push({ id, title: isLead ? "" : group.heading, blocks });
  });

  // Prerequisites is optional (a lesson that declares none has no such section), but a module
  // needs at least one real topic besides it.
  if (!sections.some((section) => section.id !== PREREQ_ID)) throw new Error(`${filename}: no lesson topics`);

  const words = source.split(/\s+/).length;
  const time = words > 6500 ? "2 weeks" : words > 4500 ? "1-2 weeks" : "1 week";

  // Where each problem card sits in the lesson, as [anchor, title]: the
  // subheading above the card, or its section when none comes first. A problem
  // page links here to the part of the lesson that teaches it, without loading
  // lessons.js.
  const problemSections = {};
  sections.forEach((section) => {
    let place = [`lesson-${readerSlug(section.id)}`, section.title || "Overview"];
    section.blocks.forEach((block, index) => {
      if (block.heading) place = [`lesson-${readerSlug(section.id)}-${index}-${readerSlug(block.heading)}`, block.heading];
      if (block.problemCard && !problemSections[block.problemCard.id]) problemSections[block.problemCard.id] = place;
      // A card folded inside an Optional box points at the heading above the box.
      else if (!block.problemCard) {
        for (const id of JSON.stringify(block).matchAll(/"problemCard":\{"id":"([^"]+)"/g)) {
          if (!problemSections[id[1]]) problemSections[id[1]] = place;
        }
      }
    });
  });

  return {
    module: {
      id: context.idByFilename.get(filename),
      title,
      section: stage.id,
      level: stage.level,
      time,
      summary,
      prerequisites,
      author: "USAAIO Guide Team",
      file: `${stage.dir}/${filename}`,
      code,
      number,
      ...(Number.isInteger(FREQUENCY[context.idByFilename.get(filename)])
        ? { frequency: FREQUENCY[context.idByFilename.get(filename)] }
        : {}),
    },
    lesson: { anchors: sections.map((section) => section.id), sections },
    problemSections,
  };
}

/* ---------------------------------- build --------------------------------- */

function moduleIdFor(filename) {
  const match = filename.replace(/\.md$/, "").match(/^[\d.]+-[A-Z]{2}\d{1,2}-(.*)$/);
  if (!match) throw new Error(`Cannot derive a module id from ${filename}`);
  return ID_OVERRIDES[match[1]] || match[1];
}

function banner(body) {
  return `(function () {\n${body}\n})();\n`;
}

/* --------------------------------- sitemap -------------------------------- */

// The canonical origin. The apex redirects here, so every URL emitted anywhere
// -- sitemap, robots, canonical tags -- must carry the www.
const ORIGIN = "https://www.aioinitiative.com";

// Routes worth indexing, most important first. Utility routes (dashboard,
// planner, editor, search, mock, login) are deliberately absent: they carry a
// noindex tag and render nothing without a session or a query string. The
// redirect stubs (contests, dl, foundation, ml, practice, roadmap, settings)
// are absent too, since they are duplicates of the pages they point at.
const SITEMAP_ROUTES = [
  ["/", "1.0"],
  ["/round1/", "0.9"],
  ["/round2/", "0.9"],
  ["/camp/", "0.9"],
  ["/advanced/", "0.8"],
  ["/problems/", "0.8"],
  ["/resources/", "0.6"],
  ["/glossary/", "0.6"],
  ["/official/", "0.6"],
  ["/about/", "0.5"],
  ["/contact/", "0.4"],
];

function writeSitemap(modules, theory) {
  const lastmod = new Date().toISOString().slice(0, 10);
  const entries = SITEMAP_ROUTES.map(([route, priority]) => ({
    loc: `${ORIGIN}${route}`,
    priority,
  }));

  modules.forEach((module) => {
    entries.push({ loc: `${ORIGIN}/module/?id=${encodeURIComponent(module.id)}`, priority: "0.7" });
  });

  // Theory problems live in the assembled bank, so take the ids from it rather
  // than duplicating the list here.
  theory.forEach((problem) => {
    entries.push({ loc: `${ORIGIN}/theory/?id=${encodeURIComponent(problem.id)}`, priority: "0.5" });
  });

  const body = entries
    .map(
      (entry) =>
        `  <url>\n    <loc>${entry.loc}</loc>\n` +
        `    <lastmod>${lastmod}</lastmod>\n    <priority>${entry.priority}</priority>\n  </url>`,
    )
    .join("\n");

  fs.writeFileSync(
    path.join(repoRoot, "sitemap.xml"),
    `<?xml version="1.0" encoding="UTF-8"?>\n` +
      `<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${body}\n</urlset>\n`,
  );

  fs.writeFileSync(
    path.join(repoRoot, "robots.txt"),
    [
      "User-agent: *",
      "Allow: /",
      "",
      "# The dashboard, planner, editor, search and mock pages carry a noindex",
      "# meta tag rather than a Disallow. Blocking them here would stop crawlers",
      "# fetching the page, so they would never see the noindex and could still",
      "# list the bare URL.",
      "",
      `Sitemap: ${ORIGIN}/sitemap.xml`,
      "",
    ].join("\n"),
  );

  console.log(`Wrote sitemap.xml (${entries.length} URLs) and robots.txt for ${ORIGIN}.`);
}

function build() {
  const idByFilename = new Map();
  const filesByStage = new Map();

  STAGES.forEach((stage) => {
    const files = fs
      .readdirSync(path.join(repoRoot, stage.dir))
      .filter((file) => file.endsWith(".md"))
      .sort((a, b) => {
        const rank = (name) => parseFloat(name.split("-")[0].split(".")[1]);
        return rank(a) - rank(b);
      });
    filesByStage.set(stage.id, files);
    files.forEach((file) => idByFilename.set(file, moduleIdFor(file)));
  });

  // Problem cards look their problem up in the two banks, which are assembled
  // from the per-module files under content/problems/ (the source of truth; the
  // two .js files are generated output the site ships as-is). The arrays are
  // kept: the sitemap reads the theory ids from the bank, and the banks are
  // written back out below from the same module files.
  const problemsDir = path.join(repoRoot, "content", "problems");
  const assembledProblems = assembleProblems(problemsDir);
  const assembledTheory = assembleTheory(problemsDir);
  const problems = new Map(assembledProblems.problems.map((p) => [p.id, p]));
  const theory = new Map(assembledTheory.problems.map((p) => [p.id, p]));

  const context = { idByFilename, problems, theory };
  const modules = [];
  const lessons = {};
  const problemSections = {};

  STAGES.forEach((stage) => {
    filesByStage.get(stage.id).forEach((file) => {
      const parsed = parseModule(stage, file, context);
      modules.push(parsed.module);
      lessons[parsed.module.id] = parsed.lesson;
      Object.entries(parsed.problemSections).forEach(([id, place]) => {
        (problemSections[id] = problemSections[id] || []).push([parsed.module.id, ...place]);
      });
    });
  });

  // Prerequisite ids are slugs; the UI shows titles, so resolve them once here
  // rather than making every renderer carry the whole catalog.
  const titleById = new Map(modules.map((module) => [module.id, module.title]));
  modules.forEach((module) => {
    module.prerequisiteTitles = module.prerequisites.map((id) => titleById.get(id) || id);
  });

  // Every on-disk module must be placed in exactly one chapter. The chapters
  // file also lists modules whose lesson files have not landed yet; those are
  // skipped with a note so the whole final structure can be committed once.
  const placed = new Set();
  STAGES.forEach((stage) => {
    CHAPTERS[stage.id].forEach((chapter) => {
      chapter.items.forEach((id) => {
        if (placed.has(id)) throw new Error(`Module ${id} appears in two chapters`);
        placed.add(id);
      });
    });
  });
  modules.forEach((module) => {
    if (!placed.has(module.id)) throw new Error(`Module ${module.id} is not in any chapter`);
    const stageChapters = CHAPTERS[module.section];
    if (!stageChapters.some((chapter) => chapter.items.includes(module.id))) {
      throw new Error(`Module ${module.id} is filed under the wrong stage's chapters`);
    }
  });
  const onDiskIds = new Set(modules.map((module) => module.id));
  const placedById = new Map();
  STAGES.forEach((stage) => {
    CHAPTERS[stage.id].forEach((chapter) => {
      chapter.items.forEach((id) => placedById.set(id, `${stage.id}/${chapter.name}`));
    });
  });
  placed.forEach((id) => {
    if (!onDiskIds.has(id)) {
      console.log(`Note: ${id} is listed in content/chapters.json under ${placedById.get(id)} but has no lesson file yet; skipped.`);
    }
  });

  const sectionOrdering = {};
  STAGES.forEach((stage) => {
    sectionOrdering[stage.id] = {
      label: stage.label,
      seoTitle: stage.seoTitle,
      description: stage.description,
      path: stage.dir,
      route: stage.route,
      // Chapters whose modules have not landed yet still appear, empty, so the
      // structure is visible while the files catch up.
      chapters: CHAPTERS[stage.id].map((chapter) => ({
        name: chapter.name,
        description: "",
        items: chapter.items.filter((id) => onDiskIds.has(id)),
      })),
    };
  });

  fs.writeFileSync(
    path.join(repoRoot, "content/ordering.js"),
    banner(`  const SECTION_ORDERING = ${JSON.stringify(sectionOrdering, null, 2).replace(/^/gm, "  ").trim()};

  const sectionIds = Object.keys(SECTION_ORDERING);
  const moduleIdToSectionMap = {};
  const moduleIdToURLMap = {};

  sectionIds.forEach((sectionId) => {
    SECTION_ORDERING[sectionId].chapters.forEach((chapter) => {
      chapter.items.forEach((moduleId) => {
        moduleIdToSectionMap[moduleId] = sectionId;
        moduleIdToURLMap[moduleId] = \`/module/?id=\${moduleId}\`;
      });
    });
  });

  window.AIO_ORDERING = {
    sections: sectionIds,
    sectionOrdering: SECTION_ORDERING,
    moduleIdToSectionMap,
    moduleIdToURLMap,
  };`),
  );

  // Order the catalog the way the stage pages read it: chapter by chapter.
  // Modules listed ahead of their files are left out; the note above reports
  // them, and the catalog must never carry a null.
  const moduleById = new Map(modules.map((module) => [module.id, module]));
  const orderedModules = STAGES.flatMap((stage) =>
    CHAPTERS[stage.id].flatMap((chapter) =>
      chapter.items.filter((id) => onDiskIds.has(id)).map((id) => moduleById.get(id)),
    ),
  );

  fs.writeFileSync(
    path.join(repoRoot, "content/catalog.js"),
    banner(`  const modules = ${JSON.stringify(orderedModules, null, 2).replace(/^/gm, "  ").trim()};

  window.AIO_CONTENT = { modules };`),
  );

  const orderedLessons = {};
  orderedModules.forEach((module) => {
    orderedLessons[module.id] = lessons[module.id];
  });

  fs.writeFileSync(
    path.join(repoRoot, "content/lessons.js"),
    banner(`  const lessons = ${JSON.stringify(orderedLessons, null, 2).replace(/^/gm, "  ").trim()};

  window.AIO_LESSONS = { lessons };`),
  );

  // Problem id -> every [module id, anchor, heading] of a lesson part that
  // asks it (a card can sit in another module's lesson too). Only the
  // theory page loads this, so the catalog every page reads stays small.
  fs.writeFileSync(
    path.join(repoRoot, "content/problem-sections.js"),
    banner(`  window.AIO_PROBLEM_SECTIONS = ${JSON.stringify(problemSections)};`),
  );

  // The two banks are generated from the per-module files, like lessons.js, and
  // committed because the site deploys straight from the repo with no build
  // step. A lane that edits content/problems/<module id>.json rebuilds, so the
  // shipped file can never fall behind its module files.
  const writtenBanks = writeContentFiles(problemsDir, repoRoot);
  console.log(
    `Wrote ${path.relative(repoRoot, writtenBanks.problemsFile)} (${assembledProblems.problems.length} problems, ` +
      `${assembledProblems.contests.length} contests) and ${path.relative(repoRoot, writtenBanks.theoryFile)} ` +
      `(${assembledTheory.problems.length} problems).`,
  );

  writeSitemap(modules, assembledTheory.problems);

  // Retired module ids point at their replacements; the module page reads this
  // to keep old links working. See scripts/import-b/redirects-from-map.mjs.
  const redirectsPath = path.join(repoRoot, "content/redirects.json");
  if (!fs.existsSync(redirectsPath)) {
    throw new Error("content/redirects.json is missing; run scripts/import-b/redirects-from-map.mjs");
  }
  const redirects = JSON.parse(fs.readFileSync(redirectsPath, "utf8"));
  fs.writeFileSync(
    path.join(repoRoot, "content/redirects.js"),
    banner(`  const redirects = ${JSON.stringify(redirects, null, 2).replace(/^/gm, "  ").trim()};

  window.AIO_REDIRECTS = redirects;`),
  );

  const blockCount = Object.values(lessons).reduce(
    (sum, lesson) => sum + lesson.sections.reduce((inner, section) => inner + section.blocks.length, 0),
    0,
  );
  console.log(
    `Built ${modules.length} modules across ${STAGES.length} stages ` +
      `(${Object.values(CHAPTERS).flat().length} chapters, ${blockCount} lesson blocks).`,
  );
}

build();
