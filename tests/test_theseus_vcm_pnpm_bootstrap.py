from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_pnpm_bootstrap as owner  # noqa: E402


def test_preflight_seals_exact_pnpm_without_dependency_or_repository_authority() -> None:
    path = ROOT / "configs" / "theseus_vcm_pnpm_bootstrap.json"
    report = owner.preflight(json.loads(path.read_text()), path)
    assert report["trigger_state"] == "PAUSED"
    assert report["network_requests"] == 0
    assert report["dependency_installations"] == 0
    assert report["repository_executions"] == 0
    assert report["candidate_or_control_calls"] == 0
