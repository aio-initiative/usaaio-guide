#!/bin/sh
# Runs a slides script under the repo virtualenv.
#
# The deck builders need pymupdf, python-pptx and matplotlib, which are rarely
# installed system-wide — hence .venv rather than whatever `python3` resolves to.
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY="$ROOT/.venv/bin/python"

if [ ! -x "$PY" ]; then
  PY=$(command -v python3 || true)
  [ -n "$PY" ] || { echo "No python3 on PATH." >&2; exit 1; }
fi

if ! "$PY" -c "import pptx, pymupdf, matplotlib" 2>/dev/null; then
  echo "Missing the slide-building packages." >&2
  echo "Install them once with:  npm run slides:setup" >&2
  exit 1
fi

exec "$PY" "$@"
