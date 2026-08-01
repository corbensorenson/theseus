from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_p4_cognitive_compilation_repaired as p4r  # noqa: E402


def test_repaired_instrument_is_green_and_has_no_quality_token_cap() -> None:
    report = p4r.audit_instrument(
        ROOT / "configs" / "theseus_p4_cognitive_compilation_repaired_instrument_r1.json"
    )

    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    frozen = report["completion_model_contract"]["model_card"]
    assert frozen["maximum_action_tokens"] == 262144

    instrument = p4r.p2a.read_json(
        ROOT / "configs" / "theseus_p4_cognitive_compilation_repaired_instrument_r1.json"
    )
    assert instrument["generation_budget"]["project_selected_quality_token_cap"] is None
    assert instrument["budgets"]["maximum_generation_tokens_per_call"] == 262144
    assert (
        instrument["budgets"]["maximum_generation_tokens_per_call"]
        == instrument["generation_budget"]["model_declared_context_window_tokens"]
    )
    assert instrument["mechanics_repair"]["candidate_or_control_calls_before_repair"] == 0
