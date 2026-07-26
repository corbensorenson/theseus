from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import resource_aware_execution_policy as policy  # noqa: E402


def test_canonical_qualified_mlx_runtime_is_probed_before_ambient_python() -> None:
    candidates = policy.mlx_python_candidates()
    canonical = (
        ROOT / "runtime" / "venvs" / "mlx-0.32.0-py312" / "bin" / "python"
    )

    assert canonical in candidates
    assert candidates.index(canonical) <= candidates.index(Path(sys.executable))
