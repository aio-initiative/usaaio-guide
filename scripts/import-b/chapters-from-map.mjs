#!/usr/bin/env node
// scripts/import-b/chapters-from-map.mjs
//
// Generates content/chapters.json from MERGE-MAP.json. The map is the agreed
// final structure of the guide: every part 1 module with its stage, chapter and
// order, plus the part 2 modules with their position anchors. The build reads
// content/chapters.json instead of a hard-coded chapter list, so the whole final
// structure can be committed once and module branches never have to touch it.
//
// Placement rule. A module whose lesson file is already on disk keeps the
// placement it has now: the build derives a module's stage from the directory
// its file lives in, so listing it under another stage would fail the build, and
// listing it under another chapter would move it on the site before its file
// has moved. Those moves stay deferred and are printed as notes. A module whose
// file is not on disk yet is listed exactly where the map puts it, so the entry
// is already in place the day its file lands and no branch has to edit the file.
//
// Usage:
//   node scripts/import-b/chapters-from-map.mjs [--map <path>] [--out <path>]
//                                              [--only-on-disk] [--dry-run]
//
//   --map <path>      MERGE-MAP.json (default: $MERGE_MAP, then the merge path)
//   --out <path>      chapters.json to write (default: content/chapters.json)
//   --only-on-disk    emit only the modules whose lesson files exist; chapters
//                     that hold none are still emitted (today none are empty)
//   --dry-run         print the notes and the summary, do not write the file

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");

// Stage labels come from the map; ids and directories come from the build, which
// keys every other file (ordering.js, the catalog, lessons) on these ids.
const STAGES = [
  { id: "round1", label: "Round 1", dir: "content/1_Round_1" },
  { id: "round2", label: "Round 2", dir: "content/2_Round_2" },
  { id: "camp", label: "Camp", dir: "content/3_Camp" },
  { id: "advanced", label: "Advanced", dir: "content/4_Advanced" },
];
const LABEL_TO_ID = Object.fromEntries(STAGES.map((stage) => [stage.label, stage.id]));
// Lesson files are named "<number>-<code>-<slug>.md"; the slug is the module id.
const ID_RE = /^[\d.]+-[A-Z]{2}\d{1,2}-(.*)$/;
const DEFAULT_MAP = "/home/jackwu/Desktop/hp/merge/MERGE-MAP.json";

const args = process.argv.slice(2);
const flag = (name) => {
  const index = args.indexOf(name);
  return index === -1 ? null : args[index + 1];
};

const mapPath = flag("--map") || process.env.MERGE_MAP || DEFAULT_MAP;
// The file being regenerated is both the current placement (what on-disk modules
// keep) and, by default, the write target. --current and --out split those roles
// apart so a variant can be generated without disturbing the committed file.
const currentPath = path.resolve(repoRoot, flag("--current") || "content/chapters.json");
const outPath = path.resolve(repoRoot, flag("--out") || "content/chapters.json");
const onlyOnDisk = args.includes("--only-on-disk");
const dryRun = args.includes("--dry-run");

const notes = [];
const note = (message) => notes.push(message);

/* -------------------------------- inputs --------------------------------- */

const map = JSON.parse(fs.readFileSync(mapPath, "utf8"));
const part1 = map.part1_modules || [];
const part2 = map.part2_a_only || [];

let current = null;
if (fs.existsSync(currentPath)) {
  current = JSON.parse(fs.readFileSync(currentPath, "utf8"));
} else {
  note(`no existing ${path.relative(repoRoot, currentPath)}; placing every module per the map`);
}

// Which lesson files are on disk today, and in which stage directory.
const onDiskStage = new Map();
STAGES.forEach((stage) => {
  fs.readdirSync(path.join(repoRoot, stage.dir))
    .filter((file) => file.endsWith(".md"))
    .forEach((file) => {
      const match = file.replace(/\.md$/, "").match(ID_RE);
      if (match) onDiskStage.set(match[1], stage.id);
    });
});

// The placement the current chapters.json gives each module: id -> placement.
const currentAt = new Map();
if (current) {
  STAGES.forEach((stage) => {
    (current[stage.id] || []).forEach((chapter, chapterIndex) => {
      chapter.items.forEach((id, index) => {
        currentAt.set(id, { stage: stage.id, chapter: chapter.name, chapterIndex, index });
      });
    });
  });
}

/* --------------------------- the map's structure -------------------------- */

// (stage label, chapter) -> list of { id, key }, in the order the map states it:
// part 1 modules sorted by `order` with part 2 modules at their anchors.
const mapChapters = new Map();
const mapChapterOrder = new Map(); // stage id -> chapter names in first-seen order
const mapAt = new Map(); // id -> { stage, chapter, key }

const chapterKey = (stageLabel, chapter) => `${stageLabel}\u0000${chapter}`;

function pushMapChapter(stageLabel, chapter, entry) {
  const stageId = LABEL_TO_ID[stageLabel];
  if (!stageId) throw new Error(`MERGE-MAP.json places ${entry.id} in an unknown stage: ${stageLabel}`);
  const key = chapterKey(stageLabel, chapter);
  if (!mapChapters.has(key)) {
    mapChapters.set(key, []);
    const order = mapChapterOrder.get(stageId) || [];
    if (!order.includes(chapter)) order.push(chapter);
    mapChapterOrder.set(stageId, order);
  }
  mapChapters.get(key).push(entry);
}

const part1Seen = new Set();
part1.forEach((entry) => {
  if (part1Seen.has(entry.id)) {
    note(`duplicate part 1 entry for ${entry.id}; ignored`);
    return;
  }
  part1Seen.add(entry.id);
  pushMapChapter(entry.stage, entry.chapter, { id: entry.id, key: entry.order ?? 0 });
  mapAt.set(entry.id, {
    stage: LABEL_TO_ID[entry.stage],
    chapter: entry.chapter,
    key: entry.order ?? 0,
  });
});

part2.forEach((entry) => {
  if (part1Seen.has(entry.id)) {
    note(`part 2 entry for ${entry.id} is already placed via part 1; its position "${entry.position}" is ignored`);
    return;
  }
  const items = mapChapters.get(chapterKey(entry.stage, entry.chapter)) || [];
  const position = (entry.position || "").trim();
  const after = position.match(/^after\s+([A-Za-z0-9][A-Za-z0-9-]*)/);
  const anchor = after && items.some((item) => item.id === after[1]) ? items.find((item) => item.id === after[1]) : null;
  const key = anchor ? anchor.key + 0.5 : (items.length ? Math.max(...items.map((item) => item.key)) + 1 : 1);
  if (!anchor) {
    note(`${entry.id}: anchor "${position || "(none)"}" not found in ${entry.stage}/${entry.chapter}; appended at end`);
  }
  pushMapChapter(entry.stage, entry.chapter, { id: entry.id, key });
  mapAt.set(entry.id, { stage: LABEL_TO_ID[entry.stage], chapter: entry.chapter, key });
});

/* --------------------------- emitted placement ---------------------------- */

// Where each map module is listed in the file we write. See the header comment
// for why a module on disk keeps the placement it already has.
const emitAt = new Map();
mapAt.forEach((where, id) => {
  const diskStage = onDiskStage.get(id);
  const currentPlacement = currentAt.get(id);
  if (diskStage !== undefined) {
    if (currentPlacement) {
      emitAt.set(id, { stage: currentPlacement.stage, chapter: currentPlacement.chapter });
      if (currentPlacement.stage !== where.stage || currentPlacement.chapter !== where.chapter) {
        const moves =
          currentPlacement.stage !== where.stage
            ? `maps to ${where.stage}/${where.chapter} but its file is on disk under ${diskStage}`
            : `maps to ${where.chapter} but is currently in ${currentPlacement.chapter}`;
        note(`deferred: ${id} ${moves}; kept in ${currentPlacement.stage}/${currentPlacement.chapter} until its file moves`);
      }
    } else {
      // A file landed with no chapter entry yet. File it in its own stage, in
      // the map's chapter; if that chapter is new to the stage it is added.
      const names = stageChapterNames(diskStage);
      const chapter = where.chapter;
      emitAt.set(id, { stage: diskStage, chapter });
      if (!names.includes(chapter)) {
        note(`${id}: its file is on disk under ${diskStage} and the map's chapter ${chapter} is new there; added as a new chapter`);
      } else {
        note(`${id}: its file is on disk under ${diskStage} with no chapter entry; filed in ${diskStage}/${chapter}`);
      }
    }
  } else if (currentPlacement && (currentPlacement.stage !== where.stage || currentPlacement.chapter !== where.chapter)) {
    note(`${id}: its lesson file is no longer on disk; moved to the map's placement ${where.stage}/${where.chapter}`);
    emitAt.set(id, { stage: where.stage, chapter: where.chapter });
  } else {
    emitAt.set(id, { stage: where.stage, chapter: where.chapter });
  }
});

// Chapters of a stage, current order first, then chapters the map adds.
function stageChapterNames(stageId) {
  const names = [];
  const seen = new Set();
  const currentList = current?.[stageId] || [];
  currentList.forEach((chapter) => {
    if (!seen.has(chapter.name)) {
      seen.add(chapter.name);
      names.push(chapter.name);
    }
  });
  (mapChapterOrder.get(stageId) || []).forEach((name) => {
    if (!seen.has(name)) {
      seen.add(name);
      names.push(name);
    }
  });
  return names;
}

/* ------------------------------ assembly --------------------------------- */

const out = {};
STAGES.forEach((stage) => {
  const chapters = [];
  stageChapterNames(stage.id).forEach((name) => {
    const currentChapter = (current?.[stage.id] || []).find((chapter) => chapter.name === name);
    // Kept: currently filed here and still on disk. Their relative order is the
    // site's order, so it never changes; everything else is inserted around it.
    const kept = currentChapter ? currentChapter.items.filter((id) => onDiskStage.has(id)) : [];
    const keptSet = new Set(kept);
    if (currentChapter) {
      currentChapter.items.forEach((id) => {
        if (!keptSet.has(id) && !mapAt.has(id)) {
          note(`dropped: ${id} is filed in ${stage.id}/${name}, its lesson file is gone, and the map does not list it`);
        }
      });
      kept.forEach((id) => {
        if (!mapAt.has(id)) {
          note(`kept: ${id} is not listed in MERGE-MAP.json; carried forward in ${stage.id}/${name}`);
        }
      });
    }

    // Keys let a not-yet-on-disk module be inserted at its map position relative
    // to the on-disk map members of this chapter.
    const keyOf = new Map();
    (mapChapters.get(chapterKey(stage.label, name)) || []).forEach((item) => {
      if (keptSet.has(item.id)) keyOf.set(item.id, item.key);
    });

    const incoming = (mapChapters.get(chapterKey(stage.label, name)) || [])
      .filter((item) => !keptSet.has(item.id))
      .filter((item) => {
        const emitted = emitAt.get(item.id);
        return emitted && emitted.stage === stage.id && emitted.chapter === name;
      })
      .sort((a, b) => a.key - b.key);

    const items = [...kept];
    incoming.forEach((item) => {
      let index = -1;
      for (let i = 0; i < items.length; i++) {
        if (keyOf.has(items[i]) && keyOf.get(items[i]) < item.key) index = i + 1;
      }
      if (index === -1) index = keyOf.size === 0 ? items.length : 0;
      items.splice(index, 0, item.id);
      // Later arrivals must sort relative to this one too, or a run of new
      // modules at the same gap ends up inserted in reverse.
      keyOf.set(item.id, item.key);
    });

    chapters.push({ name, items });
  });
  out[stage.id] = chapters;
});

if (onlyOnDisk) {
  STAGES.forEach((stage) => {
    out[stage.id].forEach((chapter) => {
      chapter.items = chapter.items.filter((id) => onDiskStage.has(id));
    });
  });
}

/* -------------------------------- output --------------------------------- */

const total = STAGES.reduce(
  (sum, stage) => sum + out[stage.id].reduce((inner, chapter) => inner + chapter.items.length, 0),
  0,
);
const onDiskTotal = STAGES.reduce(
  (sum, stage) => sum + out[stage.id].reduce((inner, chapter) => inner + chapter.items.filter((id) => onDiskStage.has(id)).length, 0),
  0,
);

const mode = onlyOnDisk ? " (on-disk modules only)" : "";
const summary = [
  `content/chapters.json${mode}: ${total} module entries across ${STAGES.reduce(
    (sum, stage) => sum + out[stage.id].length,
    0,
  )} chapters.`,
  `${onDiskTotal} have a lesson file on disk; ${total - onDiskTotal} are listed ahead of their files (the build skips those with a note).`,
].join("\n");

if (dryRun) {
  notes.forEach((line) => console.log(line));
  console.log("");
  console.log(summary);
  process.exit(0);
}

fs.writeFileSync(outPath, `${JSON.stringify(out, null, 2)}\n`);
notes.forEach((line) => console.log(line));
if (notes.length) console.log("");
console.log(`Wrote ${path.relative(repoRoot, outPath)}.`);
console.log(summary);
