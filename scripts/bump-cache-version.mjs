import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const [from, to] = process.argv.slice(2);
if (!from || !to || from === to) {
  throw new Error("Usage: node scripts/bump-cache-version.mjs <old> <new>");
}

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
let changed = 0;
// ui.css imports the per-area stylesheets with the same version string, so they refresh with the pages.
const uiStyles = path.join(root, "src", "styles", "ui.css");

function visit(directory) {
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    if (entry.name === ".git" || entry.name === "node_modules") continue;
    const target = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      visit(target);
    } else if (entry.name.endsWith(".html") || target === uiStyles) {
      const source = fs.readFileSync(target, "utf8");
      const updated = source.replaceAll(from, to);
      if (updated !== source) {
        fs.writeFileSync(target, updated);
        changed += 1;
      }
    }
  }
}

visit(root);
if (!changed) throw new Error(`No HTML or ui.css files contained ${from}`);
console.log(`Updated ${changed} HTML and CSS files from ${from} to ${to}.`);
