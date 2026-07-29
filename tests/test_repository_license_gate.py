from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import repository_license_gate  # noqa: E402


def test_repository_and_cargo_license_metadata_are_aligned() -> None:
    report = repository_license_gate.audit(ROOT)

    assert report["trigger_state"] == "GREEN"
    assert report["expected_spdx"] == "Apache-2.0"
    assert report["hard_gaps"] == []
