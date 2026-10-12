#!/usr/bin/env node
/* Prove scripts/import-b/problems.py reproduces the Colab Notebooks module's
 * committed entries exactly. m-colab ran an older fork of the importer and
 * fixed its four bugs by hand; those hand-fixed entries are the gold this
 * tool must match with no manual step.
 *
 * The four bugs, all already fixed in this tool, each get a check:
 *  (a) blocks() spun forever on a numbered list. Checked by running the
 *      importer with a spec whose `fixes` were removed, so the numbered list
 *      reaches the block parser itself: the run must terminate and emit
 *      {"olist": [...]} with B's items, words unchanged.
 *  (b) a ```fence was absorbed into prose. Checked by the deep-equal pass:
 *      every Colab problem has a code fence, and the gold holds
 *      {"code": {"language", "lines"}}.
 *  (c) <center> tags survived as paragraphs. Checked by asserting no emitted
 *      block is a paragraph whose text is a center tag.
 *  (d) the index entry's field order differed from every other entry. Checked
 *      against the gold's key order, not just deep equality.
 *
 * Usage: node scripts/import-b/proof-colab.mjs */
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import url from "node:url";

const root = path.resolve(path.dirname(url.fileURLToPath(import.meta.url)), "..", "..");
const here = path.dirname(url.fileURLToPath(import.meta.url));
const scratch = "/tmp/opencode/proof-colab";
fs.mkdirSync(scratch, { recursive: true });

const run = (file, args) => execFileSync(file, args, { cwd: root, encoding: "utf8", maxBuffer: 1 << 28 });
const sh = (what) => run("git", ["show", what]);
const A_MODULE = "colab-markdown-notebooks";
const B_MODULE = "colab-notebooks";
const BRANCH = "merge/m-colab";

/* The gold: the entries m-colab committed, and its spec (the module's
 * per-problem metadata plus the two confirmed `fixes`). */
fs.writeFileSync(path.join(scratch, "spec.json"), sh(`${BRANCH}:scripts/import-b/modules/${B_MODULE}.json`));
for (const [file, name] of [["content/problems.js", "problems.js"], ["content/theory.js", "theory.js"]]) {
  fs.writeFileSync(path.join(scratch, name), sh(`${BRANCH}:${file}`));
}
const load = (file, name) => {
  const sandbox = { window: {} };
  vm.runInNewContext(fs.readFileSync(path.join(scratch, file), "utf8"), sandbox);
  return sandbox.window[name].problems.filter((e) => e.moduleId === A_MODULE);
};
/* Notebook cards (url /editor/?nb=...) come from the notebook converter, not
 * this importer, so they are excluded from the comparison and listed instead. */
const loadTheoryCards = (file, name) => load(file, name).filter((e) => e.url.startsWith("/theory/"));
const gold = {
  index: loadTheoryCards("problems.js", "AIO_PROBLEMS"),
  theory: load("theory.js", "AIO_THEORY"),
};

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

const importModule = (specPath) =>
  JSON.parse(run("python3", [path.join(here, "problems.py"), B_MODULE, "--spec", specPath]));

/* (b)(c)(d) and the overall shape: with m-colab's spec, the importer's output
 * must equal the hand-fixed gold, entry for entry and key for key. */
const imported = importModule(path.join(scratch, "spec.json"));
for (const field of ["theory", "index"]) {
  const mine = imported[field];
  const want = gold[field];
  check(`${field}: ${mine.length} ids match gold's ${want.length}`,
    canon(mine.map((e) => e.id)) === canon(want.map((e) => e.id)),
    `mine ${mine.map((e) => e.id).join(",")}`);
  const byId = (list) => Object.fromEntries(list.map((e) => [e.id, e]));
  check(`${field}: every entry deep-equal gold`,
    canon(byId(mine)) === canon(byId(want)));
  const keyOrder = (list) => list.map((e) => Object.keys(e).join(","));
  check(`${field}: every entry's field order matches gold`,
    canon(keyOrder(mine)) === canon(keyOrder(want)),
    `mine ${keyOrder(mine)[0]}\n  gold ${keyOrder(want)[0]}`);
}

/* (c) standalone: no block may be a leftover center tag, in either run. */
for (const [label, data] of [["spec run", imported]]) {
  const tags = [];
  for (const t of data.theory) {
    for (const p of t.parts) {
      for (const b of [...p.prompt, ...p.answer]) {
        if (b.p && b.p.trim().startsWith("<center")) tags.push(`${t.id} ${b.p.trim()}`);
      }
    }
  }
  check(`(c) ${label}: no <center> tag survives as a paragraph`, tags.length === 0, tags.join(", "));
}

/* (b) standalone: every problem carries its code verbatim. The fence may sit
 * in the stem (-> setup) or inside a part (-> prompt/answer). */
const code = imported.theory.flatMap((t) => {
  const blocks = [...(t.setup || []), ...t.parts.flatMap((p) => [...p.prompt, ...p.answer])];
  return blocks.filter((b) => b.code).map((b) => `${t.id}:${b.code.language}(${b.code.lines.length} lines)`);
});
check(`(b) all 5 problems carry a verbatim code block`, code.length === 5, code.join(", "));

/* (a) standalone: with the spec's `fixes` removed, B's numbered list reaches
 * the block parser. The fork looped forever here; this run must terminate and
 * convert the list to {"olist": [...]} with the words unchanged. */
const noFixes = JSON.parse(fs.readFileSync(path.join(scratch, "spec.json"), "utf8"));
delete noFixes.fixes;
const noFixPath = path.join(scratch, "spec-no-fixes.json");
fs.writeFileSync(noFixPath, JSON.stringify(noFixes, null, 2));
const source = fs.readFileSync(path.join("/home/jackwu/Desktop/Repos/USAAIO-Guide/content/problems",
  B_MODULE, "cn-audit-before-you-submit.mdx"), "utf8");
const want = source.split("\n").filter((l) => /^\d+\. /.test(l)).map((l) => l.replace(/^\d+\. /, "").trim());
const raw = importModule(noFixPath);
const audit = raw.theory.find((t) => t.id === "cn-audit-before-you-submit");
const got = audit.parts.flatMap((p) => [...p.prompt, ...p.answer]).filter((b) => b.olist);
check(`(a) numbered list converted to olist, ${want.length} items, words unchanged`,
  got.length === 1 && canon(got[0].olist) === canon(want),
  got.length ? `got ${canon(got[0].olist)}` : "no olist block emitted");
check(`(a) without the spec's fixes only the two fixed entries move`,
  canon(raw.theory.map((t) => t.id).filter((id) =>
    canon(raw.theory.find((t) => t.id === id)) !== canon(gold.theory.find((t) => t.id === id))))
    === canon(["cn-audit-before-you-submit", "cn-read-the-counts-backward"]));
check(`(a) without the spec's fixes the index entries still match gold`,
  canon(raw.index) === canon(gold.index));

console.log(failures === 0 ? "PROOF PASSED" : `PROOF FAILED (${failures})`);
if (failures === 0) {
  const cards = load("problems.js", "AIO_PROBLEMS").filter((e) => e.url.startsWith("/editor/"));
  console.log(`note: gold also holds ${cards.length} notebook card(s) for this module, from the notebook converter: ${cards.map((e) => e.id).join(", ")}`);
}
process.exit(failures === 0 ? 0 : 1);
