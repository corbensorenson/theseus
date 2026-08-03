from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_semantic_ir_production_adequacy_campaign_v4 as campaign  # noqa: E402
import theseus_semantic_ir_production_adequacy_fresh_v4_acquisition as acquisition  # noqa: E402
import theseus_semantic_ir_production_adequacy_fresh_v4_evaluator as evaluator  # noqa: E402
import theseus_semantic_ir_production_adequacy_scorer_v4 as scorer  # noqa: E402


def test_fresh_source_preflight_is_green_and_call_free() -> None:
    report = acquisition.preflight()
    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["source_pairs_admitted"] is False
    assert all(value == 0 for value in report["counters"].values())


def test_fresh_sources_are_post_snapshot_and_repository_disjoint() -> None:
    report = p2a.read_json(acquisition.DEFAULT_OUT)
    materialization = p2a.read_json(
        ROOT / "reports" / "theseus_semantic_ir_production_adequacy_materialization_v4.json"
    )
    prior = {row["repository"] for row in materialization["rows"]}
    repositories = [row["repository"] for row in report["rows"]]
    assert report["trigger_state"] == "GREEN"
    assert report["source_pairs_admitted"] is True
    assert len(set(repositories)) == 4
    assert not prior.intersection(repositories)
    assert [row["stratum"] for row in report["rows"]] == [
        "single_expression_replacement",
        "single_expression_replacement",
        "single_expression_replacement",
        "branch_or_predicate_replacement",
    ]


def test_four_fresh_hidden_evaluators_are_qualified_before_packets() -> None:
    report = evaluator.qualify()
    assert report["trigger_state"] == "GREEN"
    assert report["green_task_count"] == 4
    assert report["candidate_packet_materialized"] is False
    assert report["candidate_or_model_exposure_authorized"] is False
    for row in report["rows"]:
        assert all(row["checks"].values())


def test_statement_granular_pool_uses_exact_frozen_token_counts() -> None:
    pool = p2a.read_json(
        ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v4_task_pool.json"
    )
    assert pool["trigger_state"] == "GREEN"
    assert pool["task_count"] == 18
    assert pool["repository_count"] == 18
    assert all(count == 3 for count in pool["stratum_counts"].values())
    assert max(row["exact_prompt_tokens"] for row in pool["rows"]) == 124138
    assert min(row["exact_context_residual_tokens"] for row in pool["rows"]) == 138006
    assert pool["rows"][0]["utf8_byte_upper_bound_exceeds_context"] is True
    assert pool["rows"][0]["exact_context_residual_tokens"] > 0


def test_prospective_campaign_v4_audit_is_green_and_call_free() -> None:
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


def test_fresh_task_uses_statement_runtime_and_fresh_namespace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pool = p2a.read_json(
        ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v4_task_pool.json"
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
        pool["rows"][2],
        "configs/theseus_assistant_runtime.json",
        tmp_path / "journal.json",
        campaign.DEFAULT_CONFIG,
        [1, 2],
    )
    assert [task_id for task_id, _, _ in prompts] == [
        "semantic_ir_adequacy_fresh_v4_03",
        "semantic_ir_adequacy_fresh_v4_03",
    ]
    assert result["index"] == 3
    assert result["hidden_evaluator_executions"] == 0


def test_independent_scorer_fails_closed_on_unsealed_run(tmp_path: Path) -> None:
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


def test_watchdog_is_infrastructure_not_capability() -> None:
    telemetry = [{
        "termination_reason": "host_safety_wall_time",
        "host_safety_wall_time_hit": True,
        "safety_ceiling_hit": True,
    }]
    assert campaign.completion_fault(8, 1, telemetry, {"ready": False}) == (
        "host_watchdog_infrastructure_invalid:task_08:call_1"
    )
