#!/bin/bash
set -euo pipefail

cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

if [ -d ".venv-demo" ]; then
  # shellcheck disable=SC1091
  source ".venv-demo/bin/activate"
fi

python3 scripts/travel_demo_preflight.py --mode parents_demo --target apple_mlx

echo
echo "Demo preflight written to reports/travel_demo_preflight.md"
echo "Travel demo runbook: docs/THESEUS_TRAVEL_PARENT_DEMO.md"
echo
if command -v open >/dev/null 2>&1; then
  open "reports/travel_demo_preflight.md" || true
  open "docs/THESEUS_TRAVEL_PARENT_DEMO.md" || true
fi
