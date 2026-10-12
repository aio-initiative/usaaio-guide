#!/bin/sh
# Regenerates the deck viewer, serves the repo, opens it in a browser.
#
# The viewer is served rather than opened over file:// because Chrome refuses
# to load a local PDF into an iframe from a file:// page.
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PORT=${PORT:-4173}
URL="http://127.0.0.1:$PORT/slides/"

if [ ! -f "$ROOT/slides/manifest.csv" ]; then
  echo "No decks built yet. Run:  npm run slides:setup && npm run build:slides" >&2
  exit 1
fi

sh "$ROOT/scripts/slides/run.sh" "$ROOT/scripts/slides/make_viewer.py"

echo "  $URL"
echo "  Ctrl-C to stop."
( sleep 1; (command -v open >/dev/null && open "$URL") || true ) &
exec python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$ROOT"
