from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_trusted_build_canaries as owner  # noqa: E402


def test_preflight_binds_tools_and_denies_repository_authority() -> None:
    path = ROOT / "configs" / "theseus_vcm_trusted_build_canaries.json"
    report = owner.preflight(json.loads(path.read_text()), path)
    assert report["trigger_state"] == "PAUSED"
    assert report["repository_execution_authorized"] is False
    assert report["dependency_prefetch_authorized"] is False
    assert report["candidate_or_control_calls"] == 0


def test_sandbox_profile_denies_network_and_outside_writes() -> None:
    profile = owner.sandbox_profile(Path("/private/tmp/theseus-vcm-build-test"))
    assert "(deny network*)" in profile
    assert "deny file-write*" in profile
    assert "/private/tmp/theseus-vcm-build-test" in profile
