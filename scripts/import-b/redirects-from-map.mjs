#!/usr/bin/env node
// scripts/import-b/redirects-from-map.mjs
//
// Generates content/redirects.json from MERGE-MAP.json. When a B module takes
// over from an A module, the map carries the A id in `redirect_from`; this
// script collects those pairs into one table the site honours, so an old link
// or bookmark to /module/?id=<old> lands on the module that replaced it.
//
// The table maps old id -> new id. It is deliberately flat: a redirect never
// points at another redirect, and the validator checks that. A source may have
// been retired before its target's file lands; the target is trusted because it
// is a module the map places, and the validator confirms it is listed in
// content/chapters.json.
//
// Usage:
//   node scripts/import-b/redirects-from-map.mjs [--map <path>] [--out <path>]
//                                                [--chapters <path>] [--dry-run]
//
//   --map <path>       MERGE-MAP.json (default: $MERGE_MAP, then the merge path)
//   --out <path>       redirects.json to write (default: content/redirects.json)
//   --chapters <path>  content/chapters.json, used only to sanity-check targets
//   --dry-run          print the notes and the summary, do not write the file

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const DEFAULT_MAP = "/home/jackwu/Desktop/hp/merge/MERGE-MAP.json";

const args = process.argv.slice(2);
const flag = (name) => {
  const index = args.indexOf(name);
  return index === -1 ? null : args[index + 1];
};

const mapPath = flag("--map") || process.env.MERGE_MAP || DEFAULT_MAP;
const outPath = path.resolve(repoRoot, flag("--out") || "content/redirects.json");
const chaptersPath = path.resolve(repoRoot, flag("--chapters") || "content/chapters.json");
const dryRun = args.includes("--dry-run");

const map = JSON.parse(fs.readFileSync(mapPath, "utf8"));

const pairs = [];
(map.part1_modules || []).forEach((entry) => {
  if (!entry.redirect_from) return;
  const sources = Array.isArray(entry.redirect_from) ? entry.redirect_from : [entry.redirect_from];
  sources.forEach((source) => {
    if (!source) return;
    if (source === entry.id) {
      console.log(`Note: ${entry.id} redirects from itself; skipped.`);
      return;
    }
    pairs.push([source, entry.id]);
  });
});

// One old id cannot point at two replacements; the first entry wins and the
// rest are reported, because two branches naming the same retired module is a
// decision an owner has to make.
const table = {};
const notes = [];
pairs.forEach(([source, target]) => {
  if (Object.prototype.hasOwnProperty.call(table, source)) {
    notes.push(`${source} is redirected to ${table[source]} and again to ${target}; kept the first`);
    return;
  }
  table[source] = target;
});

// Targets should be modules the structure knows about. A target that is in
// neither the structure nor on disk yet is reported rather than dropped: the map
// is the authority, but a typo'd id would silently send every old link nowhere.
if (fs.existsSync(chaptersPath)) {
  const chapters = JSON.parse(fs.readFileSync(chaptersPath, "utf8"));
  const listed = new Set();
  Object.values(chapters).forEach((chaptersList) => {
    (chaptersList || []).forEach((chapter) => {
      (chapter.items || []).forEach((id) => listed.add(id));
    });
  });
  Object.entries(table).forEach(([source, target]) => {
    if (!listed.has(target)) {
      notes.push(`${target}, the replacement for ${source}, is not listed in ${path.relative(repoRoot, chaptersPath)}`);
    }
  });
}

const sorted = {};
Object.keys(table)
  .sort()
  .forEach((source) => {
    sorted[source] = table[source];
  });

notes.forEach((note) => console.log(`Note: ${note}.`));
console.log(
  `${dryRun ? "Would write" : "Wrote"} ${Object.keys(sorted).length} redirects to ${path.relative(repoRoot, outPath)}.`,
);

if (dryRun) process.exit(0);

fs.writeFileSync(outPath, `${JSON.stringify(sorted, null, 2)}\n`);
