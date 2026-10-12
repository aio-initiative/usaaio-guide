#!/usr/bin/env node
/*
 * scripts/import-b/fill-links.mjs [--map <MERGE-MAP.json>] [--links <links.json>]
 *
 * Fills scripts/import-b/links.json's "modules" table from the merge map: every
 * part1_modules entry with a b_id and a file maps that B module id to its A
 * file, so the importer can remap B's /round-1/<b_id> links and prerequisites.
 *
 * The "notebooks" and "problems" tables are not in the map (notebooks come with
 * the notebook lane, problem ids with the practice lane), so they are left
 * alone. Rerun this whenever the map gains entries; it overwrites "modules".
 *
 * The map's default location is outside this repo and can be overridden with
 * the MERGE_MAP environment variable or --map.
 */

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "../..");
const defaultMap = process.env.MERGE_MAP || "/home/jackwu/Desktop/hp/merge/MERGE-MAP.json";

const args = process.argv.slice(2);
let mapPath = defaultMap;
let linksPath = path.join(here, "links.json");
for (let index = 0; index < args.length; index += 1) {
  if (args[index] === "--map") {
    mapPath = args[index + 1];
    index += 1;
  } else if (args[index] === "--links") {
    linksPath = args[index + 1];
    index += 1;
  } else {
    console.error(`usage: node ${path.relative(process.cwd(), fileURLToPath(import.meta.url))} [--map <map.json>] [--links <links.json>]`);
    process.exit(2);
  }
}

const raw = JSON.parse(fs.readFileSync(mapPath, "utf8"));
const links = JSON.parse(fs.readFileSync(linksPath, "utf8"));

const modules = {
  _comment:
    "B frontmatter id / B route slug -> the merged module's A file, as the map has it (content/<stage>/<file>.md). The importer writes the link's href as the file's basename, which is how A's own lessons cross-reference modules, and warns when a target is not merged yet. Filled from MERGE-MAP.json by scripts/import-b/fill-links.mjs.",
};

let count = 0;
const ordered = [];
for (const entry of raw.part1_modules || []) {
  if (!entry.b_id || !entry.file) continue;
  ordered.push(entry);
}
ordered.sort((a, b) => a.b_id.localeCompare(b.b_id));
for (const entry of ordered) {
  if (modules[entry.b_id] !== undefined) {
    console.error(`fill-links: ${entry.b_id} appears twice in the map; keeping the first`);
    continue;
  }
  modules[entry.b_id] = entry.file;
  count += 1;
}

const out = { ...links, modules };
fs.writeFileSync(linksPath, `${JSON.stringify(out, null, 2)}\n`);
console.error(`fill-links: wrote ${count} module links from ${mapPath} into ${path.relative(repoRoot, linksPath)}`);
