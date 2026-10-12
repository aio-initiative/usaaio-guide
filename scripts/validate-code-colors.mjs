import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const source = fs.readFileSync(new URL('../src/components/renderers.js', import.meta.url), 'utf8');
const tokenizer = source.slice(source.indexOf('  const PY_KEYWORDS'), source.indexOf('  function renderCodeBlock'));
const context = { escapeHtml: text => String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;') };
vm.createContext(context);
vm.runInContext(tokenizer + '\nthis.highlight = highlightPython;', context);
for (const code of ['import numpy as np\nprint("<script>&") # comment', 'x = """first\nsecond"""\nreturn x', 'def f(x):\n    return x + 2', '']) {
  const html = context.highlight(code);
  assert.equal(html.replace(/<[^>]+>/g, '').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&'), code);
  for (const line of html.split('\n')) assert.equal((line.match(/<span/g) || []).length, (line.match(/<\/span>/g) || []).length);
  assert(!html.includes('<script>'));
}
assert(context.highlight('return 2').includes('tok-kw'));
console.log('Syntax coloring preserves text, escapes HTML, and balances multiline tokens.');
