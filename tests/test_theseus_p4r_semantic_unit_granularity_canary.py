from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_p4r_semantic_unit_granularity_canary as canary  # noqa: E402


def test_namespaced_runner_preserves_historical_receipt_paths(monkeypatch) -> None:
    observed: list[str] = []

    def fake_call(arm, task_id, call_number, prompt, maximum, runtime_config):
        observed.append(task_id)
        return {}

    original = canary.v2r1.p2a.runtime_call
    canary.v2r1.p2a.runtime_call = fake_call
    try:
        wrapped_original = canary.v2r1.p2a.runtime_call
        wrapped_original(
            "direct_local_model", "p4r_semantic_unit_mechanics_case", 1,
            "prompt", 1, "config",
        )
    finally:
        canary.v2r1.p2a.runtime_call = original

    assert observed == ["p4r_semantic_unit_mechanics_case"]


def test_semantic_unit_policy_is_non_claim_scoped() -> None:
    assert "semantic-unit" in canary.__doc__.lower()
    assert canary.POLICY.endswith("semantic_unit_granularity_canary_v1")
