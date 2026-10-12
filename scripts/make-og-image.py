"""Renders public/assets/og-card.png, the 1200x630 image social platforms show.

Social crawlers do not render SVG, so the site's logo.svg cannot be used as an
og:image. The card is authored as HTML (scripts/og-card.html) and shot with
headless Chrome so it uses the real Outfit display face and the real --brand
tokens, matching the homepage splash. Re-run after changing the wordmark,
the hero copy, or the brand colours:

    python3 scripts/make-og-image.py

Bump the ?v= on the og:image URLs afterwards, or Discord and Slack will keep
serving the previously scraped card from their own caches.
"""
import pathlib
import shutil
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ROOT / "scripts" / "og-card.html"
TARGET = ROOT / "public" / "assets" / "og-card.png"
WIDTH, HEIGHT = 1200, 630

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "google-chrome",
    "chromium",
]


def find_chrome() -> str:
    for candidate in CHROME_CANDIDATES:
        if candidate.startswith("/"):
            if pathlib.Path(candidate).exists():
                return candidate
        else:
            found = shutil.which(candidate)
            if found:
                return found
    sys.exit(
        "No Chrome/Chromium found. Install Google Chrome, or render "
        f"{SOURCE.relative_to(ROOT)} at {WIDTH}x{HEIGHT} by hand."
    )


def main() -> None:
    if not SOURCE.exists():
        sys.exit(f"missing {SOURCE.relative_to(ROOT)}")

    before = TARGET.stat().st_mtime if TARGET.exists() else 0
    chrome = find_chrome()
    proc = subprocess.Popen(
        [
            chrome,
            "--headless",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
            f"--window-size={WIDTH},{HEIGHT}",
            # Long enough for the webfont to arrive and the canvas to paint.
            "--virtual-time-budget=8000",
            f"--screenshot={TARGET}",
            SOURCE.as_uri(),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Headless Chrome writes the screenshot and then sometimes lingers instead of
    # exiting, so treat a fresh file as success rather than waiting on the process.
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    if not TARGET.exists() or TARGET.stat().st_mtime <= before:
        sys.exit("Chrome produced no screenshot")

    print(f"wrote {TARGET.relative_to(ROOT)} ({TARGET.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
