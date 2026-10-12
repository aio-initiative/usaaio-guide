"""Generates slides/index.html — a local browser for the built decks.

The deck list is baked in rather than fetched so the page works over
file:// as well as through `npm run dev`.
"""
import csv
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "slides" / "index.html"
CSV = ROOT / "slides" / "manifest.csv"
STAGES = ["Round 1", "Round 2", "Camp", "Advanced"]

PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Module decks</title>
<style>
:root{
  --purple:#5b21b6; --blue:#2563eb; --panel:#dbeafe;
  --bg:#f6f5f9; --card:#fff; --edge:#e4e1ec; --ink:#2b2733; --muted:#6f6981;
}
@media (prefers-color-scheme:dark){
  :root{--purple:#a78bfa; --blue:#8ab0f7; --panel:#241f36;
        --bg:#15131b; --card:#1c1926; --edge:#312c40; --ink:#e2dfea; --muted:#8f88a3;}
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;display:flex;background:var(--bg);color:var(--ink);
  font:400 15px/1.5 "Roboto",system-ui,-apple-system,sans-serif;overflow:hidden}

/* sidebar */
aside{width:320px;flex:none;display:flex;flex-direction:column;
  background:var(--card);border-right:1px solid var(--edge)}
.head{padding:16px 18px 14px;border-bottom:1px solid var(--edge)}
.head b{display:block;font-size:15px;letter-spacing:-.01em}
.head small{display:block;margin-top:3px;font:400 11.5px/1 ui-monospace,monospace;
  letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
#q{width:100%;margin-top:12px;padding:8px 11px;border:1px solid var(--edge);border-radius:6px;
  background:var(--bg);color:var(--ink);font:inherit;font-size:14px}
#q:focus{outline:2px solid var(--blue);outline-offset:-1px;border-color:transparent}
nav{flex:1;overflow-y:auto;padding-bottom:20px}
.stage{position:sticky;top:0;background:var(--panel);color:var(--purple);z-index:1;
  padding:7px 18px;font:700 11.5px/1.4 ui-monospace,monospace;letter-spacing:.1em;
  text-transform:uppercase}
a.deck{display:flex;gap:10px;align-items:baseline;padding:8px 18px;text-decoration:none;
  color:inherit;border-left:3px solid transparent}
a.deck:hover{background:var(--bg)}
a.deck[aria-current=true]{background:var(--bg);border-left-color:var(--purple);font-weight:700}
a.deck .c{font:400 11.5px/1.4 ui-monospace,monospace;color:var(--muted);width:30px;flex:none}
a.deck .t{flex:1;font-size:14px}
a.deck .n{font:400 11.5px/1.4 ui-monospace,monospace;color:var(--muted);font-variant-numeric:tabular-nums}
.empty{padding:20px 18px;color:var(--muted);font-size:14px}

/* main */
main{flex:1;display:flex;flex-direction:column;min-width:0}
.bar{display:flex;align-items:center;gap:14px;padding:12px 20px;
  background:var(--card);border-bottom:1px solid var(--edge)}
.bar h1{margin:0;font-size:17px;letter-spacing:-.01em;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis}
.bar .meta{font:400 12px/1 ui-monospace,monospace;color:var(--muted);white-space:nowrap}
.bar .sp{flex:1}
.bar a{font:400 12px/1 ui-monospace,monospace;letter-spacing:.06em;text-transform:uppercase;
  color:var(--blue);text-decoration:none;border:1px solid var(--edge);border-radius:5px;
  padding:7px 11px;white-space:nowrap}
.bar a:hover{border-color:var(--blue)}
iframe{flex:1;width:100%;border:0;background:var(--bg)}
kbd{font:400 11px/1 ui-monospace,monospace;border:1px solid var(--edge);border-bottom-width:2px;
  border-radius:3px;padding:2px 5px;color:var(--muted)}
@media (max-width:760px){aside{width:230px}.bar .meta,.bar .hint{display:none}}
</style>

<aside>
  <div class="head">
    <b>Module decks</b>
    <small>__N__ decks · __S__ slides</small>
    <input id="q" type="search" placeholder="Filter by name or code" autocomplete="off">
  </div>
  <nav id="nav"></nav>
</aside>

<main>
  <div class="bar">
    <h1 id="title">—</h1>
    <span class="meta" id="meta"></span>
    <span class="sp"></span>
    <span class="meta hint"><kbd>↑</kbd><kbd>↓</kbd> switch · <kbd>/</kbd> filter</span>
    <a id="pdf" href="#" target="_blank" rel="noopener">Open PDF</a>
    <a id="mod" href="#" target="_blank" rel="noopener">Lesson</a>
  </div>
  <iframe id="view" title="Slide deck"></iframe>
</main>

<script>
const DECKS = __DECKS__;
const STAGES = __STAGES__;
const nav = document.getElementById("nav"), q = document.getElementById("q");
let current = null;

function draw(filter) {
  const f = filter.trim().toLowerCase();
  const hits = DECKS.filter(d =>
    !f || d.title.toLowerCase().includes(f) || d.code.toLowerCase().includes(f));
  nav.innerHTML = "";
  if (!hits.length) {
    nav.innerHTML = '<p class="empty">No deck matches that.</p>';
    return;
  }
  STAGES.forEach(stage => {
    const inStage = hits.filter(d => d.stage === stage);
    if (!inStage.length) return;
    const h = document.createElement("div");
    h.className = "stage";
    h.textContent = stage;
    nav.appendChild(h);
    inStage.forEach(d => {
      const a = document.createElement("a");
      a.className = "deck";
      a.href = "#" + d.id;
      a.dataset.id = d.id;
      if (d.id === current) a.setAttribute("aria-current", "true");
      a.innerHTML = '<span class="c"></span><span class="t"></span><span class="n"></span>';
      a.children[0].textContent = d.code;
      a.children[1].textContent = d.title;
      a.children[2].textContent = d.slides;
      nav.appendChild(a);
    });
  });
}

function show(id) {
  const d = DECKS.find(x => x.id === id) || DECKS[0];
  current = d.id;
  document.getElementById("title").textContent = d.title;
  document.getElementById("meta").textContent = d.code + " · " + d.slides + " slides";
  document.getElementById("view").src = d.pdf + "#view=FitH";
  document.getElementById("pdf").href = d.pdf;
  document.getElementById("mod").href = d.page;
  document.title = d.title + " — Module decks";
  draw(q.value);
  const el = nav.querySelector('[data-id="' + d.id + '"]');
  if (el) el.scrollIntoView({ block: "nearest" });
}

function step(delta) {
  const shown = [...nav.querySelectorAll("a.deck")].map(a => a.dataset.id);
  const i = shown.indexOf(current);
  const next = shown[Math.min(shown.length - 1, Math.max(0, (i < 0 ? 0 : i + delta)))];
  if (next) location.hash = next;
}

q.addEventListener("input", () => draw(q.value));
addEventListener("hashchange", () => show(location.hash.slice(1)));
addEventListener("keydown", e => {
  if (e.key === "/" && document.activeElement !== q) { e.preventDefault(); q.focus(); q.select(); }
  else if (e.key === "Escape" && document.activeElement === q) { q.value = ""; draw(""); q.blur(); }
  else if ((e.key === "ArrowDown" || e.key === "ArrowUp") && document.activeElement !== q) {
    e.preventDefault(); step(e.key === "ArrowDown" ? 1 : -1);
  }
});

show(location.hash.slice(1));
</script>
"""


def main():
    rows = list(csv.DictReader(CSV.open()))
    decks = [{
        "id": r["PPTX file"].replace(".pptx", ""),
        "stage": r["Stage"],
        "code": r["Code"],
        "title": r["Module"],
        "slides": int(r["Slides"]),
        "pdf": r["PDF file"],
        "page": r["Module page"],
    } for r in rows if r["PDF file"]]

    page = (PAGE
            .replace("__DECKS__", json.dumps(decks, indent=1))
            .replace("__STAGES__", json.dumps(STAGES))
            .replace("__N__", str(len(decks)))
            .replace("__S__", f"{sum(d['slides'] for d in decks):,}"))
    OUT.write_text(page)
    missing = len(rows) - len(decks)
    note = f"  ({missing} module(s) have no PDF yet — run npm run build:slides)" if missing else ""
    print(f"  {len(decks)} decks -> {OUT.relative_to(ROOT)}{note}")


if __name__ == "__main__":
    main()
