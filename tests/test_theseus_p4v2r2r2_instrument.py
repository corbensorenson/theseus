from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4v2r2r2_cognitive_compilation as runner  # noqa: E402


INSTRUMENT = ROOT / "configs" / "theseus_p4v2r2r2_cognitive_compilation_instrument.json"


def test_fresh_recovery_instrument_is_green_and_uncapped() -> None:
    report = runner.audit_instrument(INSTRUMENT)
    value = p2a.read_json(INSTRUMENT)

    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["runtime_attempt_namespace"] == "p4v2r2r2_attempt1"
    assert value["generation_budget"]["project_selected_quality_token_cap"] is None
    assert value["boundaries"]["user_gate"] == "none"
    assert value["single_mechanism_change"]["causal_implementation_changed_from_r1"] is False


def test_fresh_recovery_instrument_excludes_every_predecessor_task() -> None:
    value = json.loads(INSTRUMENT.read_text(encoding="utf-8"))
    predecessor = value["recovery_predecessor"]
    fresh = value["fresh_task_pool_contract"]

    assert predecessor["complete_tasks_excluded"] == 6
    assert predecessor["partial_tasks_excluded"] == 1
    assert predecessor["candidate_unseen_tasks_excluded"] == 3
    assert predecessor["same_denominator_resume_authorized"] is False
    assert fresh["all_predecessor_tasks_excluded"] is True
    assert fresh["task_replacement_after_any_candidate_call"] is False
    assert fresh["user_task_label_or_approval_dependency"] is False
