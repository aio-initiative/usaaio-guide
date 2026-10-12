#!/usr/bin/env python3
"""Headless Chrome checks for one module page. scripts/check-module.mjs runs this
through heavy.sh, twice (1440 px and 375 px), and reports one line per check.

Prints one JSON line on stdout:
  {"ok": true, "title": "...", "katex": 0, "wide": [], "console": []}
  {"ok": false, "error": "..."}          -- the page could not be checked at all
The browser is expected to exist (google-chrome) and the site to be served
already; the caller says where with --base.

Requires the websocket-client package (as scripts/figures/figures.py requires
numpy). Nothing else outside the standard library.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import websocket


def main():
    base = sys.argv[1].rstrip("/")
    module_id = sys.argv[2]
    width = int(sys.argv[3]) if len(sys.argv) > 3 else 1440
    height = 900 if width < 500 else 1000
    mobile = width < 500

    url = f"{base}/module/?id={module_id}"
    profile = tempfile.mkdtemp(prefix="aio-check-")
    port = 9400 + os.getpid() % 400
    proc = None
    try:
        proc = subprocess.Popen(
            [
                "google-chrome", "--headless=new", "--disable-gpu", "--hide-scrollbars",
                "--no-first-run", "--no-default-browser-check",
                f"--user-data-dir={profile}", f"--remote-debugging-port={port}",
                f"--window-size={width},{height}", "about:blank",
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        target = None
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=2))
                pages = [t for t in tabs if t.get("type") == "page"]
                if pages:
                    target = pages[0]
                    break
            except Exception:
                pass
            time.sleep(0.1)
        if target is None:
            print(json.dumps({"ok": False, "error": "Chrome's debug endpoint never answered"}))
            return

        ws = websocket.create_connection(target["webSocketDebuggerUrl"], timeout=120, suppress_origin=True)
        events = []
        msg_id = 0

        def call(method, **params):
            nonlocal msg_id
            msg_id += 1
            ws.send(json.dumps({"id": msg_id, "method": method, "params": params}))
            while True:
                msg = json.loads(ws.recv())
                if msg.get("id") == msg_id:
                    if "error" in msg:
                        raise RuntimeError(f"{method}: {msg['error']}")
                    return msg.get("result", {})
                events.append(msg)

        def js(expression):
            result = call("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
            if "exceptionDetails" in result:
                raise RuntimeError(str(result["exceptionDetails"].get("exception", {}).get("description", result["exceptionDetails"])))
            return result.get("result", {}).get("value")

        call("Page.enable")
        call("Runtime.enable")
        call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=mobile)
        call("Page.navigate", url=url)

        # Give the page's own scripts (KaTeX, the reader render) time to settle.
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                if js("document.readyState === 'complete'"):
                    break
            except RuntimeError:
                pass
            time.sleep(0.25)
        time.sleep(2.5)

        summary = js(
            """(() => {
              const vw = window.innerWidth;
              // An element is only a real overflow if it pushes the page wider. Decorative
              // SVG in the hero canvas and anything an ancestor already clips are not.
              const clipped = (el) => {
                for (let cur = el; cur && cur !== document.documentElement; cur = cur.parentElement) {
                  const cs = getComputedStyle(cur);
                  if (cs.overflowX !== 'visible' || cs.overflowY !== 'visible' || cs.position === 'fixed') return true;
                }
                return false;
              };
              const wide = [];
              for (const el of document.querySelectorAll('body *')) {
                if (el.namespaceURI && el.namespaceURI.indexOf('svg') !== -1) continue;
                const rect = el.getBoundingClientRect();
                if (rect.width > vw + 1 && !clipped(el)) {
                  const cls = (el.className && el.className.baseVal !== undefined) ? el.className.baseVal : el.className;
                  wide.push(`${el.tagName.toLowerCase()}${cls ? '.' : ''}${String(cls).split(' ')[0]}`.slice(0, 60));
                }
              }
              return JSON.stringify({
                title: (document.querySelector('h1') || {}).textContent ? document.querySelector('h1').textContent.trim() : '',
                loaded: !!document.querySelector('.reader-article, .module-reader'),
                katex: document.querySelectorAll('.katex-error').length,
                scrollWider: document.documentElement.scrollWidth > vw + 1,
                wide: wide.slice(0, 10),
                vw,
              });
            })()"""
        )
        parsed = json.loads(summary) if summary else {}

        console = []
        for event in events:
            if event.get("method") == "Runtime.exceptionThrown":
                details = event["params"]["exceptionDetails"]
                console.append((details.get("exception", {}).get("description") or details.get("text") or "exception")[:200])
            elif event.get("method") == "Runtime.consoleAPICalled" and event["params"]["type"] == "error":
                console.append(" ".join(str(arg.get("value", arg.get("description", ""))) for arg in event["params"]["args"])[:200])

        ws.close()
        parsed.update({"ok": True, "console": console})
        print(json.dumps(parsed))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(5)
            except Exception:
                proc.kill()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
