#!/usr/bin/env sh
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)" || exit 1
PYTHON="./.venv-puffer/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi
exec "$PYTHON" scripts/theseus_cli.py "$@"
