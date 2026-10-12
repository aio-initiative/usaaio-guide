import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const scope = { window: {} };
for (const path of ['content/definitions.js', 'content/catalog.js', 'content/lessons.js']) {
  vm.runInNewContext(fs.readFileSync(new URL('../' + path, import.meta.url), 'utf8'), scope);
}
const entries = scope.window.AIO_DEFINITIONS;
const modules = new Set(scope.window.AIO_CONTENT.modules.map((m) => m.id));
const ids = new Set(), aliases = new Set();
for (const entry of entries) {
  assert(!ids.has(entry.id), `Duplicate definition: ${entry.id}`);
  ids.add(entry.id);
  assert(entry.title && entry.brief && entry.body.length, `Incomplete definition: ${entry.id}`);
  assert(modules.has(entry.moduleId), `Unknown module for ${entry.id}`);
  assert(scope.window.AIO_LESSONS.lessons[entry.moduleId].sections.some((s) => !['intro', 'prerequisites'].includes(s.id)), `Missing destination for ${entry.id}`);
  for (const alias of entry.aliases) {
    assert(!aliases.has(alias.toLowerCase()), `Duplicate alias: ${alias}`);
    aliases.add(alias.toLowerCase());
  }
  for (const block of [...entry.body, ...entry.proof]) {
    const text = block.p || block.tex;
    assert(text && !/[\x00-\x08\x0b\x0c\x0e-\x1f]/.test(text), `Broken string escape in ${entry.id}`);
    if (block.p) assert.equal((text.match(/\$/g) || []).length % 2, 0, `Unpaired math delimiter in ${entry.id}`);
  }
}
console.log(`Validated ${entries.length} definitions, their aliases, math delimiters, and module destinations.`);
