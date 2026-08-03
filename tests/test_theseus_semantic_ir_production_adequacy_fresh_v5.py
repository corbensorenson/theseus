from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_semantic_ir_production_adequacy_campaign_v5 as campaign  # noqa: E402
import theseus_semantic_ir_production_adequacy_fresh_v5_acquisition as acquisition  # noqa: E402
import theseus_semantic_ir_production_adequacy_fresh_v5_evaluator as evaluator  # noqa: E402
import theseus_semantic_ir_production_adequacy_scorer_v5 as scorer  # noqa: E402


def test_v5_source_preflight_binds_the_consumed_v4_wall() -> None:
    report = acquisition.preflight()
    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["source_pairs_admitted"] is False
    assert all(value == 0 for value in report["counters"].values())


def test_v5_source_is_post_snapshot_licensed_and_repository_disjoint() -> None:
    report = p2a.read_json(acquisition.DEFAULT_OUT)
    prior = p2a.read_json(
        ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v4_task_pool.json"
    )
    prior_repositories = {row["repository"] for row in prior["source_denominator"]}
    row = report["rows"][0]
    assert report["trigger_state"] == "GREEN"
    assert report["source_pairs_admitted"] is True
    assert row["repository"] == "dknowles2/pytboss"
    assert row["repository"] not in prior_repositories
    assert row["license_spdx"] == "Apache-2.0"
    assert row["stratum"] == "single_expression_replacement"


def test_v5_hidden_evaluator_is_qualified_before_packet_creation() -> None:
    report = evaluator.qualify()
    assert report["trigger_state"] == "GREEN"
    assert report["green_task_count"] == 1
    assert report["candidate_packet_materialized"] is False
    assert report["candidate_or_model_exposure_authorized"] is False
    assert all(report["rows"][0]["checks"].values())


def test_v5_pool_replaces_only_task_one_and_reduces_its_prompt() -> None:
    v4 = p2a.read_json(
        ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v4_task_pool.json"
    )
    v5 = p2a.read_json(
        ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v5_task_pool.json"
    )
    assert v5["trigger_state"] == "GREEN"
    assert v5["task_count"] == 18
    assert v5["repository_count"] == 18
    assert v5["replacement_indices"] == [1]
    assert v5["rebound_unexposed_indices"] == list(range(2, 19))
    assert all(count == 3 for count in v5["stratum_counts"].values())
    assert v5["rows"][0]["exact_prompt_tokens"] == 9165
    assert v5["rows"][0]["exact_prompt_tokens"] < v4["rows"][0]["exact_prompt_tokens"] / 10
    assert v5["rows"][0]["serialized_prompt_sha256"] != v4["rows"][0]["serialized_prompt_sha256"]
    for index in range(1, 18):
        assert v5["rows"][index]["serialized_prompt_sha256"] == v4["rows"][index]["serialized_prompt_sha256"]


def test_v5_campaign_audit_is_green_and_call_free() -> None:
    report = campaign.audit_config(campaign.DEFAULT_CONFIG)
    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["sealed_task_count"] == 18
    assert report["generation_indices"] == list(range(1, 19))
    assert report["candidate_generation_opened"] is False
    assert report["hidden_evaluation_opened"] is False
    assert all(value == 0 for value in report["counters"].values())


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
            "receipt": {
                "call_number": call_number,
                "candidate_output_sha256": p2a.sha256_text(output),
            },
        }

    return call


def test_v5_task_uses_a_new_namespace_without_hidden_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pool = p2a.read_json(
        ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v5_task_pool.json"
    )
    prompts: list[tuple[str, int, str]] = []
    monkeypatch.setattr(p2a, "runtime_call", fake_call("not a semantic artifact", prompts))
    monkeypatch.setattr(
        campaign.p4r,
        "termination_telemetry",
        lambda attempts: [
            {"termination_reason": "model_eos", "safety_ceiling_hit": False}
            for _ in attempts[0]["runtime_calls"]
        ],
    )
    result = campaign.run_task(
        pool["rows"][0],
        "configs/theseus_assistant_runtime.json",
        tmp_path / "journal.json",
        campaign.DEFAULT_CONFIG,
        [],
    )
    assert [task_id for task_id, _, _ in prompts] == [
        "semantic_ir_adequacy_fresh_v5_01",
        "semantic_ir_adequacy_fresh_v5_01",
    ]
    assert result["index"] == 1
    assert result["hidden_evaluator_executions"] == 0


def test_v5_scorer_fails_closed_before_complete_candidate_seal(tmp_path: Path) -> None:
    run = tmp_path / "unsealed.json"
    p2a.write_json(
        run,
        {
            "policy": campaign.POLICY,
            "state": "RUNNING_CANDIDATE_GENERATION",
            "trigger_state": "YELLOW",
            "completed_task_count": 0,
            "preserved_candidate_count": 0,
            "new_candidate_count": 0,
            "hidden_evaluation_opened": False,
            "rows": [],
            "counters": campaign.zero_counters(),
        },
    )
    report = scorer.score(run)
    assert report["trigger_state"] == "RED"
    assert report["scientific_status"] == "INVALID_INFORMATION_FLOW_OR_EVALUATOR"
    assert report["counters"]["hidden_evaluator_executions"] == 0
    assert "candidate_run_not_sealed_for_hidden_evaluation" in report["faults"]


def test_watchdog_remains_infrastructure_not_capability() -> None:
    telemetry = [{
        "termination_reason": "host_safety_wall_time",
        "host_safety_wall_time_hit": True,
        "safety_ceiling_hit": True,
    }]
    assert campaign.completion_fault(1, 1, telemetry, {"ready": False}) == (
        "host_watchdog_infrastructure_invalid:task_01:call_1"
    )
