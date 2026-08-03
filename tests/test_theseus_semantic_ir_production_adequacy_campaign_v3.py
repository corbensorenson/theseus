from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_semantic_ir_production_adequacy_campaign_v3 as campaign  # noqa: E402
import theseus_semantic_ir_production_adequacy_scorer_v3 as scorer  # noqa: E402


def test_prospective_v3_audit_is_green_and_call_free() -> None:
    report = campaign.audit_config(campaign.DEFAULT_CONFIG)
    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["sealed_task_count"] == 18
    assert report["preserved_candidate_count"] == 3
    assert report["resume_generation_indices"] == list(range(4, 19))
    assert report["watchdog_classifier_precedence"] == "backend_telemetry_before_route_consequence"
    assert report["candidate_generation_opened"] is False
    assert report["hidden_evaluation_opened"] is False
    assert all(value == 0 for value in report["counters"].values())


def test_config_has_no_quality_cap_and_authorizes_only_30_local_calls() -> None:
    config = p2a.read_json(campaign.DEFAULT_CONFIG)
    completion = p2a.mapping(config["generation_completion"])
    assert completion["project_selected_quality_token_cap"] is None
    assert completion["host_watchdog_seconds"] == 600
    assert completion["backend_telemetry_precedes_route_consequence_classification"] is True
    assert config["authority"]["new_local_model_calls_authorized_after_green_audit"] == 30
    assert config["authority"]["external_inference_authorized"] is False


def test_watchdog_classification_precedes_downstream_route_hold() -> None:
    telemetry = [{
        "termination_reason": "host_safety_wall_time",
        "host_safety_wall_time_hit": True,
        "safety_ceiling_hit": True,
    }]
    route = {"ready": False, "release_allowed": False}
    assert campaign.completion_fault(4, 2, telemetry, route) == (
        "host_watchdog_infrastructure_invalid:task_04:call_2"
    )


def fake_call(output: str, prompts: list[tuple[str, int, str]]):
    def call(
        arm: str,
        task_id: str,
        call_number: int,
        prompt: str,
        maximum: int,
        runtime_config: str,
    ) -> dict:
        prompts.append((task_id, call_number, prompt))
        return {
            "assistant_text": output,
            "runtime_report": {"route_integrity": {"ready": True, "release_allowed": True}},
            "receipt": {"call_number": call_number, "candidate_output_sha256": p2a.sha256_text(output)},
        }
    return call


def test_replacement_task_04_uses_fresh_namespace_and_two_normal_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pool = p2a.read_json(ROOT / "reports/theseus_semantic_ir_production_adequacy_task_pool_v3.json")
    prompts: list[tuple[str, int, str]] = []
    monkeypatch.setattr(p2a, "runtime_call", fake_call("not a semantic artifact", prompts))
    monkeypatch.setattr(
        campaign.p4r,
        "termination_telemetry",
        lambda _attempts: [{"termination_reason": "model_eos", "safety_ceiling_hit": False}],
    )
    result = campaign.run_task(
        pool["rows"][3],
        "configs/theseus_assistant_runtime.json",
        tmp_path / "journal.json",
        campaign.DEFAULT_CONFIG,
        [1, 2, 3],
    )
    assert [task_id for task_id, _, _ in prompts] == [
        "semantic_ir_adequacy_04r1",
        "semantic_ir_adequacy_04r1",
    ]
    assert result["index"] == 4
    assert result["hidden_evaluator_executions"] == 0


def test_independent_scorer_fails_closed_on_unsealed_run(tmp_path: Path) -> None:
    run = tmp_path / "unsealed.json"
    p2a.write_json(run, {
        "policy": campaign.POLICY,
        "state": "RUNNING_CANDIDATE_GENERATION",
        "trigger_state": "YELLOW",
        "completed_task_count": 3,
        "preserved_candidate_count": 3,
        "new_candidate_count": 0,
        "hidden_evaluation_opened": False,
        "rows": [],
        "counters": campaign.zero_counters(),
    })
    report = scorer.score(run)
    assert report["trigger_state"] == "RED"
    assert report["scientific_status"] == "INVALID_INFORMATION_FLOW"
    assert report["counters"]["hidden_evaluator_executions"] == 0
    assert "candidate_run_not_sealed_for_hidden_evaluation" in report["faults"]


def test_two_replacement_source_denominator_preserves_six_strata() -> None:
    rows, strata = scorer.source_denominator()
    assert rows[2]["repository"] == "Universal-Commerce-Protocol/conformance"
    assert rows[4]["repository"] == "scikit-bio/scikit-bio"
    assert rows[4]["selected_source_paths"] == ["skbio/alignment/_pair.py"]
    assert len(set(strata.values())) == 6
    assert all(list(strata.values()).count(value) == 3 for value in set(strata.values()))
