#!/usr/bin/env sh
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
if [ -x ".venv-puffer/bin/python" ]; then
  PYTHON=".venv-puffer/bin/python"
fi
exec "$PYTHON" scripts/theseus_setup_wizard.py --open
