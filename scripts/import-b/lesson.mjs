#!/usr/bin/env node
/*
 * scripts/import-b/lesson.mjs <B mdx file> <A md file>
 *
 * Turns one B lesson (MDX with B's frontmatter and components) into an A lesson
 * file with the contract-1 tags, changing no sentence. Prose, math, code, table
 * and caption text is carried through verbatim; only the container syntax is
 * converted. Everything the importer cannot map is printed as a warning on
 * stderr, and the list of image paths the lesson needs is printed at the end.
 *
 * See docs/merge-tooling.md for the full contract and the known conversions.
 */

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "../..");

// The merge workspace lives outside this repo; both paths can be overridden.
const B_REPO = process.env.B_REPO || "/home/jackwu/Desktop/Repos/USAAIO-Guide";
const MAP_FILE = process.env.MERGE_MAP || "/home/jackwu/Desktop/hp/merge/MERGE-MAP.json";

/* ------------------------------ remap tables ------------------------------ */

const STAGE_LABEL = {
  "round-1": "Round 1, Foundations.",
  "round-2": "Round 2, Core contest depth.",
  camp: "Camp, Advanced preparation.",
  advanced: "Advanced, Extension topics.",
};

// MERGE-MAP.json: b_id -> { file: "content/<stage>/<file>.md", b_path, title, ... }.
// Entries without b_id are new modules with no B source and are skipped.
function loadMap(mapPath) {
  const raw = JSON.parse(fs.readFileSync(mapPath, "utf8"));
  const byBId = new Map();
  for (const entry of raw.part1_modules || []) {
    if (entry.b_id && entry.file) byBId.set(entry.b_id, entry);
  }
  return byBId;
}

function loadLinks() {
  const raw = JSON.parse(fs.readFileSync(path.join(here, "links.json"), "utf8"));
  const pick = (section) => {
    const out = {};
    for (const [key, value] of Object.entries(raw[section] || {})) {
      if (!key.startsWith("_")) out[key] = value;
    }
    return out;
  };
  return {
    modules: pick("modules"),
    notebooks: pick("notebooks"),
    problems: pick("problems"),
  };
}

/* ------------------------------ small pieces ------------------------------ */

function warning(note) {
  warnings.push(note);
}

const warnings = [];
const images = [];

// A heading line like "# 1.18 · DL1 · MLP" gives the title. The site writes the
// same H1 shape with a middle dot between the code and the title; older files
// use an em dash there, so the pattern accepts both.
function titleFromH1(line) {
  const match = line.match(/^#\s+[\d.]+\s+·\s+[A-Z]{2}\d{1,2}\s+(?:\u2014|·)\s+(.*)$/);
  return match ? match[1].trim() : null;
}

// Rewrites [label](href) outside math spans. Internal B routes are remapped
// through the module table; unmapped ones lose the link but keep B's label.
function rewriteLinks(line, context) {
  const out = [];
  let math = false;
  let cursor = 0;
  while (cursor < line.length) {
    const dollar = line.indexOf("$", cursor);
    const linkStart = line.indexOf("[", cursor);
    if (dollar === -1 && linkStart === -1) {
      out.push(line.slice(cursor));
      break;
    }
    const next = dollar === -1 ? linkStart : linkStart === -1 ? dollar : Math.min(dollar, linkStart);
    out.push(line.slice(cursor, next));
    if (next === dollar) {
      // A "$" toggles a math span unless it is escaped. "$$" on one line
      // toggles twice through the same path and lands back in prose.
      if (next > 0 && line[next - 1] === "\\") {
        out.push("$");
        cursor = next + 1;
      } else {
        const closing = line.indexOf("$", next + 1);
        if (closing === -1) {
          math = !math;
          out.push("$");
          cursor = next + 1;
        } else {
          out.push(line.slice(next, closing + 1));
          cursor = closing + 1;
        }
      }
      continue;
    }
    // A "[" outside math: convert if it is a link with a parenthesised target.
    const close = line.indexOf("]", next);
    if (close === -1 || line[close + 1] !== "(") {
      out.push("[");
      cursor = next + 1;
      continue;
    }
    const end = line.indexOf(")", close + 1);
    if (end === -1) {
      out.push("[");
      cursor = next + 1;
      continue;
    }
    const label = line.slice(next + 1, close);
    const href = line.slice(close + 2, end);
    out.push(convertLink(label, href, context));
    cursor = end + 1;
  }
  return out.join("");
}

function convertLink(label, href, context) {
  if (/^(https?:|mailto:)/.test(href)) return `[${label}](${href})`;
  if (href.startsWith("#")) return `[${label}](${href})`;
  if (!href.startsWith("/")) {
    warning(`${context.where}: relative link [${label}](${href}) is not a B route; left alone`);
    return `[${label}](${href})`;
  }
  const [route, anchor] = href.slice(1).split("#");
  const segments = route.split("/");
  if (!["round-1", "round-2", "camp", "advanced"].includes(segments[0])) {
    warning(`${context.where}: link [${label}](${href}) is not a module route; left alone`);
    return `[${label}](${href})`;
  }
  const bId = segments[segments.length - 1];
  const target = context.links.modules[bId];
  if (!target) {
    warning(`${context.where}: link [${label}](${href}) has no A module in links.json yet; the label is kept as plain text`);
    return label;
  }
  // A's lesson links are bare filenames resolved by basename, so the target is
  // written as a normal link whether or not the file exists yet; the reader
  // degrades it to plain text until the target module lands.
  const hrefName = path.basename(target);
  if (!fs.existsSync(context.root ? path.join(context.root, target) : path.join(context.dir || ".", target))) {
    warning(`${context.where}: link [${label}](${href}) -> ${target} is not merged yet; the link is written to ${hrefName} and reads as plain text until that module lands`);
  }
  if (anchor) {
    warning(`${context.where}: link [${label}](${href}) carries the anchor #${anchor}; A's reader drops anchors, so it needs a manual check`);
  }
  return `[${label}](${hrefName}${anchor ? `#${anchor}` : ""})`;
}

/* --------------------------- frontmatter (YAML) --------------------------- */

function parseFrontmatter(source) {
  const lines = source.split("\n");
  if (lines[0] !== "---") throw new Error("the file does not start with a YAML frontmatter block");
  let index = 1;
  const attrs = {};
  let listKey = null;
  while (index < lines.length && lines[index] !== "---") {
    const line = lines[index];
    if (!line.trim()) {
      index += 1;
      continue;
    }
    const item = line.match(/^\s+-\s+(.*)$/);
    if (item) {
      if (!listKey) throw new Error(`line ${index + 1}: a list item with no key`);
      attrs[listKey].push(item[1].trim());
      index += 1;
      continue;
    }
    const entry = line.match(/^([A-Za-z][\w-]*):\s*(.*)$/);
    if (!entry) throw new Error(`line ${index + 1}: cannot read the frontmatter line "${line}"`);
    const [, key, value] = entry;
    listKey = null;
    if (value === "") {
      attrs[key] = [];
      listKey = key;
    } else {
      attrs[key] = value.replace(/^"(.*)"$/, "$1");
    }
    index += 1;
  }
  if (lines[index] !== "---") throw new Error("the frontmatter block is never closed");
  return { attrs, bodyStart: index + 1, lines };
}

/* ----------------------------- component tags ----------------------------- */

// Reads an opening tag that may run over several lines, quote-aware so a ">"
// inside an attribute value does not end the tag. Returns the tag text up to and
// including its closing ">", the line index after it, and any text that followed
// the ">" on the last line.
function readOpeningTag(lines, start) {
  let text = "";
  let index = start;
  while (index < lines.length) {
    const line = lines[index].trim();
    text += (text ? " " : "") + line;
    let quoted = false;
    for (let i = 0; i < text.length; i += 1) {
      if (text[i] === '"') quoted = !quoted;
      if (text[i] === ">" && !quoted) {
        return { tag: text.slice(0, i + 1), rest: text.slice(i + 1).trim(), next: index + 1 };
      }
    }
    index += 1;
  }
  throw new Error(`line ${start + 1}: a component tag is never closed with ">"`);
}

// Takes a full opening tag (including its angle brackets) and returns the
// attribute source plus whether the tag closed itself.
function tagInner(tag, name, where) {
  const inner = tag.slice(1, -1).trim();
  const selfClosing = inner.endsWith("/");
  const body = selfClosing ? inner.slice(0, -1).trim() : inner;
  if (!body.startsWith(name)) {
    throw new Error(`${where}: the tag does not start with <${name}`);
  }
  return { attrsBody: body.slice(name.length).trim(), selfClosing };
}

function parseAttributes(source, where) {
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

// The body of a paired component up to its closing tag, fence-aware so a tag
// written inside a code example does not close it.
function readUntilClosing(lines, start, name, where) {
  const closing = `</${name}>`;
  const opening = new RegExp(`^<${name}(?=[\\s>]|$)`);
  const body = [];
  let index = start;
  let fenced = false;
  while (index < lines.length) {
    const line = lines[index];
    const trimmed = line.trim();
    if (trimmed.startsWith("```")) {
      fenced = !fenced;
      body.push(line);
      index += 1;
      continue;
    }
    if (!fenced && trimmed === closing) return { body, next: index + 1 };
    if (!fenced && opening.test(trimmed)) {
      warning(`${where}: a nested <${name}> inside a <${name}> is carried through as text`);
    }
    body.push(line);
    index += 1;
  }
  throw new Error(`${where}: the <${name}> block is never closed`);
}

// <FigureBox image="..." imageDark="..." alt="...">caption</FigureBox>
function convertFigureBox(lines, start, context) {
  const where = `${context.where}: line ${start + 1} <FigureBox>`;
  const { tag, next } = readOpeningTag(lines, start);
  const { attrsBody, selfClosing } = tagInner(tag, "FigureBox", where);
  const attrs = parseAttributes(attrsBody, where);
  const src = attrs.image;
  if (!src) {
    warning(`${where}: has no image attribute; written as an <Figure> with an empty src`);
  }
  const aSrc = src ? `assets/module-media/${path.basename(src)}` : "";
  if (aSrc) images.push(aSrc);
  if (attrs.imageDark) {
    // A has no dark variants; the dark image is dropped and not needed.
  }
  const alt = attrs.alt || "";
  for (const value of Object.values(attrs)) {
    if (typeof value === "string" && value.includes('"')) {
      warning(`${where}: an attribute value contains a double quote, which A's tag parser cannot read; check the output`);
    }
  }
  const out = [`<Figure src="${aSrc}" alt="${alt}">`];
  if (!selfClosing) {
    const block = readUntilClosing(lines, next, "FigureBox", where);
    const caption = block.body.map((line) => line.trim()).filter(Boolean).join(" ");
    out.push(caption);
    out.push("</Figure>");
    return { lines: out, next: block.next };
  }
  out.push("</Figure>");
  return { lines: out, next };
}

// <Resources> with one <Resource ... /> per line: the indentation is dropped to
// match how A's files are written, and nothing else changes.
function convertResources(lines, start, context) {
  const where = `${context.where}: line ${start + 1} <Resources>`;
  const { next } = readOpeningTag(lines, start);
  const block = readUntilClosing(lines, next, "Resources", where);
  const out = ["<Resources>"];
  for (const line of block.body) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    if (trimmed.startsWith("<Resource")) {
      out.push(convertResourceTag(trimmed, where));
      continue;
    }
    warning(`${where}: a line inside <Resources> is not a <Resource>: "${trimmed}"`);
    out.push(trimmed);
  }
  out.push("</Resources>");
  return { lines: out, next: block.next, resources: true };
}

function convertResourceTag(text, where) {
  if (!/\/>\s*$/.test(text)) {
    warning(`${where}: a <Resource> does not close itself on one line; check the output`);
    return text;
  }
  return text;
}

// <EmbeddedPyEditor packages={['numpy']} initialCode={`...code...`} />: B writes
// the tag as JSX over several lines, with the Python code inside the initialCode
// template literal. A's parser wants the tag's own shape instead: an opening tag
// whose packages attribute is a comma separated list, the code as plain lines,
// then a closing tag. The code is copied byte for byte; only leading and
// trailing blank lines are trimmed.
function convertEmbeddedPyEditor(lines, index, context) {
  const where = `${context.where} line ${index + 1}`;
  const attrs = [];
  let codeLine = -1;
  let head = "";
  for (let scan = index; scan < lines.length; scan += 1) {
    const match = /^(\s*)initialCode=\{`(.*)$/.exec(lines[scan]);
    if (match) {
      head = match[2];
      codeLine = scan;
      break;
    }
    if (scan !== index) attrs.push(lines[scan].trim());
  }
  if (codeLine === -1) throw new Error(`${where}: <EmbeddedPyEditor> has no initialCode attribute`);

  const packages = [];
  for (const attr of attrs) {
    const attrName = attr.split("=")[0].trim();
    if (attrName === "packages") {
      const names = [...attr.matchAll(/'([^']*)'|"([^"]*)"/g)].map((m) => m[1] || m[2]);
      if (!names.length) {
        warning(`${where}: <EmbeddedPyEditor> has a packages attribute the importer could not read (${attr}); the attribute was dropped`);
      } else {
        packages.push(...names);
      }
    } else {
      warning(`${where}: <EmbeddedPyEditor> attribute "${attrName}" is not supported; it was dropped`);
    }
  }

  // The template literal ends at the first backtick, which may close the code
  // at the end of a line or sit on a line of its own.
  const code = head === "" ? [] : [head];
  let end = codeLine + 1;
  let tick = -1;
  while (end < lines.length) {
    tick = lines[end].indexOf("`");
    if (tick !== -1) break;
    code.push(lines[end]);
    end += 1;
  }
  if (end >= lines.length) throw new Error(`${where}: <EmbeddedPyEditor> initialCode is never closed with a backtick`);
  const last = lines[end].slice(0, tick);
  if (last !== "") code.push(last);

  // After the literal comes "}", then the tag closes itself, either on the same
  // line or on the next one.
  const tail = lines[end].slice(tick + 1).trim();
  if (!tail.startsWith("}")) throw new Error(`${where}: <EmbeddedPyEditor> initialCode backtick is not followed by "}"`);
  let next = end + 1;
  if (!tail.slice(1).includes("/")) {
    while (next < lines.length && lines[next].trim() === "") next += 1;
    if (next >= lines.length || !lines[next].includes("/>")) {
      throw new Error(`${where}: <EmbeddedPyEditor> is never closed with a "/>"`);
    }
    next += 1;
  }

  while (code.length && code[0].trim() === "") code.shift();
  while (code.length && code[code.length - 1].trim() === "") code.pop();

  const out = [];
  out.push(packages.length ? `<EmbeddedPyEditor packages="${packages.join(",")}">` : "<EmbeddedPyEditor>");
  out.push(...code);
  out.push("</EmbeddedPyEditor>");
  return { lines: out, next };
}

function loadProblemIds(contentDir) {
  if (!contentDir) return null;
  const file = path.join(path.dirname(path.resolve(contentDir)), "problems.js");
  if (!fs.existsSync(file)) return null;
  const source = fs.readFileSync(file, "utf8");
  const ids = new Set();
  for (const match of source.matchAll(/^ *"id": "([^"]+)" *$/gm)) ids.add(match[1]);
  return ids;
}

/* A's files write one paragraph per line, so B's ~80-column wrapping is joined
 * back together. That touches whitespace only; no sentence changes. Everything
 * that is not prose (fences, math blocks, tables, tags, headings) is pushed
 * verbatim and closes the running paragraph. */
function convertBody(lines, context, state) {
  const out = [];
  const flush = () => {
    if (state.pending.length) {
      out.push(state.pending.map((entry) => entry.trim()).filter(Boolean).join(" "));
      state.pending = [];
      state.pendingKind = null;
    }
  };
  let index = state.index || 0;

  while (index < lines.length) {
    const line = lines[index];
    const trimmed = line.trim();

    if (state.fenced !== null) {
      out.push(line);
      if (trimmed.startsWith("```")) state.fenced = null;
      index += 1;
      continue;
    }

    // Fenced code.
    if (trimmed.startsWith("```")) {
      flush();
      let info = trimmed.slice(3).trim();
      // A latex or markdown fence is code to show, not math to render: B uses
      // ```latex for "type this" examples next to the math they produce, so the
      // fence is kept as a fenced code block with its language. Only ```py is
      // renamed, and only a fence left unlabelled after a python block becomes
      // output.
      if (info === "py") info = "python";
      if (info === "" && state.lastOpened === "python") {
        info = "output";
        const block = [];
        let scan = index + 1;
        while (scan < lines.length && !lines[scan].trim().startsWith("```")) {
          block.push(lines[scan]);
          scan += 1;
        }
        if (block.some((entry) => /^\s*(import |from |def |class |for \(|while |return )/.test(entry))) {
          warning(`${context.where}: line ${index + 1}: the fence after a python block looks like code, not output; check it`);
        }
      }
      state.lastOpened = info || "text";
      out.push("```" + info);
      state.fenced = info || "";
      index += 1;
      continue;
    }

    // Fenced display math: a lone $$ line opens or closes it.
    if (trimmed === "$$") {
      flush();
      state.inDisplay = !state.inDisplay;
      out.push(line);
      index += 1;
      continue;
    }
    if (state.inDisplay) {
      flush();
      out.push(line);
      index += 1;
      continue;
    }

    // Blank line: close the running paragraph and keep the blank line itself,
    // so the output keeps B's paragraph spacing.
    if (!trimmed) {
      flush();
      out.push("");
      index += 1;
      continue;
    }

    // Component tags.
    if (/^<[A-Z]/.test(trimmed)) {
      const name = trimmed.match(/^<([A-Za-z]+)/)[1];
      const where = `${context.where} line ${index + 1}`;
      if (name === "ProblemCard") {
        flush();
        const { tag, next } = readOpeningTag(lines, index);
        const attrs2 = parseAttributes(tagInner(tag, name, where).attrsBody, where);
        const id = attrs2.id;
        if (!id) throw new Error(`${where}: <ProblemCard> has no id`);
        const mapped = context.links.problems[id] || id;
        out.push(`<ProblemCard id="${mapped}" />`);
        if (!context.links.problems[id]) {
          context.problemsSeen = context.problemsSeen || new Set();
          if (!context.problemsSeen.has(id)) {
            context.problemsSeen.add(id);
            if (context.problemIds && !context.problemIds.has(mapped)) {
              warning(`${where}: <ProblemCard id="${id}"> has no entry in links.json and is not in A's content/problems.js; the merge map must say which A problem it becomes`);
            }
          }
        }
        index = next;
        continue;
      }
      if (name === "NotebookCard") {
        flush();
        const { tag, next } = readOpeningTag(lines, index);
        const attrs2 = parseAttributes(tagInner(tag, name, where).attrsBody, where);
        const notebookPath = attrs2.notebookPath;
        if (!notebookPath) throw new Error(`${where}: <NotebookCard> has no notebookPath`);
        const id = context.links.notebooks[notebookPath] || path.basename(notebookPath, ".ipynb");
        if (!context.links.notebooks[notebookPath]) {
          warning(`${where}: <NotebookCard notebookPath="${notebookPath}"> is mapped to id "${id}" by filename stem; confirm the notebook exists in A`);
        }
        out.push(`<NotebookCard id="${id}" />`);
        index = next;
        continue;
      }
      if (name === "FigureBox") {
        flush();
        const block = convertFigureBox(lines, index, context);
        out.push(...block.lines);
        index = block.next;
        continue;
      }
      if (name === "Resources") {
        flush();
        const block = convertResources(lines, index, context);
        state.resources = block.lines;
        index = block.next;
        continue;
      }
      if (name === "Optional") {
        flush();
        const { tag, next } = readOpeningTag(lines, index);
        const attrs2 = parseAttributes(tagInner(tag, name, where).attrsBody, where);
        const titleAttr = attrs2.title || "";
        out.push(`<Optional title="${titleAttr}">`);
        const block = readUntilClosing(lines, next, name, where);
        const inner = convertBody(block.body, context, { fenced: null, lastOpened: state.lastOpened, inDisplay: false, pending: [] });
        out.push(...inner.out);
        state.lastOpened = inner.state.lastOpened;
        out.push("</Optional>");
        index = block.next;
        continue;
      }
      if (name === "Warning" || name === "Info") {
        // A's build parses these two callouts as B writes them, so they pass
        // through untouched under their own names.
        flush();
        const { tag, next } = readOpeningTag(lines, index);
        const attrs2 = parseAttributes(tagInner(tag, name, where).attrsBody, where);
        const titleAttr = attrs2.title ? ` title="${attrs2.title}"` : "";
        out.push(`<${name}${titleAttr}>`);
        const block = readUntilClosing(lines, next, name, where);
        out.push(...block.body);
        out.push(`</${name}>`);
        index = block.next;
        continue;
      }
      if (name === "EmbeddedPyEditor") {
        flush();
        const block = convertEmbeddedPyEditor(lines, index, context);
        out.push(...block.lines);
        index = block.next;
        continue;
      }
      throw new Error(`${where}: unknown component <${name}>`);
    }

    if (trimmed === "<center>" || trimmed === "</center>") {
      flush();
      warning(`${context.where} line ${index + 1}: a <center> wrapper was dropped (A has no centering); the table it wrapped is unchanged`);
      index += 1;
      continue;
    }

    // Headings.
    if (trimmed.startsWith("#")) {
      flush();
      if (trimmed === "## Resources") {
        // Drop B's top resources heading; the section is re-emitted at the end.
        index += 1;
        continue;
      }
      out.push(line);
      index += 1;
      continue;
    }

    // Tables: one row per line.
    if (trimmed.startsWith("|")) {
      flush();
      out.push(line);
      index += 1;
      continue;
    }

    // A list item starts a new line only at a block boundary or when the buffer
    // already holds a list. A wrapped prose line that happens to begin with
    // digits and a full stop ("which is / 297.") is a continuation, not a new
    // item. The buffer's kind is tracked explicitly because an item that wraps
    // onto a second line leaves plain text, not a marker, as the last entry.
    const isItemList = /^[-*] /.test(trimmed) || /^\d+\. /.test(trimmed);
    if (isItemList && (state.pending.length === 0 || state.pendingKind === "list")) {
      flush();
    }

    // Prose, list items and their wrapped continuation lines.
    state.pending.push(rewriteLinks(line, { ...context, where: `${context.where} line ${index + 1}` }));
    if (state.pending.length === 1) state.pendingKind = isItemList ? "list" : "prose";
    index += 1;
  }
  flush();
  if (state.inDisplay) warning(`${context.where}: a fenced $$ block is never closed`);
  return { out, state };
}

// Reads the "## Prerequisites" bullets of an existing A file, so a B lesson that
// declares none can inherit them instead of getting an empty section (A's build
// rejects an empty one). Verbatim: they are already A links.
function readExistingPrerequisites(file) {
  if (!fs.existsSync(file)) return [];
  const lines = fs.readFileSync(file, "utf8").split("\n");
  const start = lines.findIndex((line) => /^## Prerequisites/.test(line));
  if (start === -1) return [];
  const kept = [];
  for (let index = start + 1; index < lines.length && !/^## /.test(lines[index]); index += 1) {
    kept.push(lines[index]);
  }
  let from = 0;
  while (from < kept.length && !kept[from].trim()) from += 1;
  let to = kept.length;
  while (to > from && !kept[to - 1].trim()) to -= 1;
  return kept.slice(from, to);
}

/* -------------------------------- the file -------------------------------- */

function convert(source, outputPath, links, contentDir, map) {
  const { attrs, bodyStart, lines } = parseFrontmatter(source);
  const outName = path.basename(outputPath);
  const nameMatch = outName.match(/^([\d.]+)-([A-Z]{2}\d{1,2})-/);
  if (!nameMatch) throw new Error(`the output filename "${outName}" is not "<number>-<CODE>-<slug>.md"`);
  const [, number, code] = nameMatch;
  const title = attrs.title;
  if (!title) throw new Error("the frontmatter has no title");
  const tier = attrs.tier;
  if (!STAGE_LABEL[tier]) throw new Error(`the frontmatter tier "${tier}" has no stage label`);
  const root = repoRoot;
  const context = { links, where: outName, problemIds: loadProblemIds(contentDir), root, map };

  const header = [
    `# ${number} · ${code} · ${title}`,
    "",
    `> **Goal:** ${attrs.description || ""}`,
    `> **Stage:** ${STAGE_LABEL[tier]}`,
    "",
  ];

  const dir = contentDir || path.dirname(path.resolve(outputPath));
  const declared = Array.isArray(attrs.prerequisites) ? attrs.prerequisites : [];
  if (declared.length) header.push("## Prerequisites", "");
  for (const id of declared) {
    const target = links.modules[id];
    if (!target) {
      warning(`${outName}: prerequisite ${id} has no A module in links.json yet; written as a plain bullet`);
      header.push(`- ${id}`);
      continue;
    }
    const targetPath = target.includes("/")
      ? path.join(root, target)
      : path.join(dir, target);
    let label = id;
    if (fs.existsSync(targetPath)) {
      const h1 = fs.readFileSync(targetPath, "utf8").split("\n").find((line) => line.startsWith("# "));
      label = (h1 && titleFromH1(h1)) || label;
    } else {
      const entry = map ? map.get(id) : null;
      label = (entry && entry.title) || label;
      warning(`${outName}: prerequisite ${id} -> ${target} is not merged yet; the link is written and reads as plain text until that module lands`);
    }
    header.push(`- [${label}](${path.basename(target)})`);
  }

  if (!declared.length) {
    // B declares no prerequisites. The section is optional on the site, so
    // nothing is invented: an existing A file keeps its own list, and a new
    // file simply has no section.
    const kept = readExistingPrerequisites(outputPath);
    if (kept.length) {
      header.push("## Prerequisites", "", ...kept);
      warning(`${outName}: B declares no prerequisites; kept A's existing ${kept.length} prerequisite bullet(s) from ${outName} verbatim`);
    } else {
      warning(`${outName}: no prerequisites in B; section omitted`);
    }
  }

  if (attrs.redirects) {
    warning(`${outName}: the frontmatter has redirects, which A does not use: ${JSON.stringify(attrs.redirects)}`);
  }

  // Body.
  const { out: body, state } = convertBody(lines.slice(bodyStart), context, {
    fenced: null,
    lastOpened: null,
    inDisplay: false,
    pending: [],
    pendingKind: null,
    resources: null,
  });
  const resources = state.resources;

  if (!resources) warning(`${outName}: the lesson has no <Resources> block`);

  const out = [...header, "", ...body, "", "## Resources", "", ...(resources || []), ""];
  return { text: `${out.join("\n")}\n`, images, warnings };
}

/* ---------------------------------- main ---------------------------------- */

function main() {
  const args = process.argv.slice(2);
  let bFile = null;
  let aFile = null;
  let contentDir = null;
  let mapPath = MAP_FILE;
  let map = null;

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--map") {
      mapPath = args[index + 1];
      index += 1;
    } else if (arg === "--from-map") {
      if (!map) {
        try {
          map = loadMap(mapPath);
        } catch (error) {
          throw new Error(`cannot read the merge map ${mapPath}: ${error.message}`);
        }
      }
      const bId = args[index + 1];
      if (!bId) {
        console.error("usage: node scripts/import-b/lesson.mjs --from-map <b_id> [--map <map.json>] [A content dir]");
        process.exit(2);
      }
      const entry = map.get(bId);
      if (!entry) throw new Error(`${bId} is not in ${mapPath} (or has no b_id and file)`);
      bFile = path.join(B_REPO, entry.b_path);
      aFile = path.join(repoRoot, entry.file);
      index += 1;
    } else if (!bFile) {
      bFile = arg;
    } else if (!aFile) {
      aFile = arg;
    } else {
      contentDir = arg;
    }
  }

  if (!bFile || !aFile) {
    console.error("usage: node scripts/import-b/lesson.mjs <B mdx file> <A md file> [A content dir]");
    console.error("       node scripts/import-b/lesson.mjs --from-map <b_id> [--map <map.json>] [A content dir]");
    console.error("  <A content dir> is where sibling A lessons live, for reading prerequisite");
    console.error("  titles. Defaults to the output file's own directory.");
    process.exit(2);
  }
  if (!map) {
    try {
      map = loadMap(mapPath);
    } catch (error) {
      warning(`the merge map ${mapPath} could not be read (${error.message}); prerequisite titles come from A files only`);
      map = new Map();
    }
  }
  const source = fs.readFileSync(bFile, "utf8");
  const links = loadLinks();
  const { text } = convert(source, aFile, links, contentDir, map);
  if (aFile === "-") {
    process.stdout.write(text);
  } else {
    fs.writeFileSync(aFile, text);
    console.error(`wrote ${aFile}`);
  }
  for (const note of warnings) console.error(`warning: ${note}`);
  console.error(`images needed by ${path.basename(aFile)}:`);
  for (const image of images) console.error(`  ${image}`);
  if (warnings.length) console.error(`${warnings.length} warning(s)`);
}

main();
