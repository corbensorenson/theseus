from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_semantic_ir_production_adequacy_campaign_v6 as campaign  # noqa: E402
import theseus_semantic_ir_production_adequacy_disposition_v6 as disposition  # noqa: E402
import theseus_semantic_ir_production_adequacy_fresh_v6_acquisition as acquisition  # noqa: E402
import theseus_semantic_ir_production_adequacy_fresh_v6_evaluator as evaluator  # noqa: E402
import theseus_semantic_ir_production_adequacy_fresh_v6_task_pool as task_pool  # noqa: E402
import theseus_semantic_ir_production_adequacy_scorer_v6 as scorer  # noqa: E402


def test_v6_source_preflight_is_green_and_call_free() -> None:
    report = acquisition.preflight()
    assert report["trigger_state"] == "GREEN"
    assert report["source_pairs_admitted"] is False
    assert report["candidate_packet_materialized"] is False
    assert report["faults"] == []
    assert all(value == 0 for value in report["counters"].values())


def test_v6_sources_are_frozen_licensed_and_disjoint() -> None:
    report = acquisition.materialize.read_json(acquisition.DEFAULT_OUT)
    assert report["trigger_state"] == "GREEN"
    assert report["source_pairs_admitted"] is True
    assert all(row["trigger_state"] == "GREEN" for row in report["rows"])
    assert [row["repository"] for row in report["rows"]] == [
        "ModelTC/LightLLM",
        "WeblateOrg/translation-finder",
        "durandtibo/feu",
        "statsmodels/statsmodels",
    ]
    assert all(row["pr_base_head_authoritative"] is True for row in report["rows"])
    assert all(row["merge_parent_is_lineage_only"] is True for row in report["rows"])


def test_v6_hidden_evaluators_qualify_before_packet_creation() -> None:
    report = evaluator.qualify()
    assert report["trigger_state"] == "GREEN"
    assert report["green_task_count"] == 4
    assert report["candidate_packet_materialized"] is False
    assert report["candidate_or_model_exposure_authorized"] is False
    assert report["faults"] == []
    assert all(all(row["checks"].values()) for row in report["rows"])


def test_v6_pool_is_uniform_fresh_balanced_and_call_free() -> None:
    report = task_pool.p2a.read_json(task_pool.DEFAULT_OUT)
    assert report["trigger_state"] == "GREEN"
    assert report["task_count"] == 18
    assert report["sealed_packet_count"] == 18
    assert report["repository_count"] == 18
    assert report["replacement_indices"] == [1, 2, 3, 4]
    assert report["uniformly_rebound_unexposed_indices"] == list(range(5, 19))
    assert all(count == 3 for count in report["stratum_counts"].values())
    assert report["information_flow"]["uniform_compact_protocol_for_all_tasks"] is True
    assert report["consumed_v5_prompt_or_candidate_reused"] is False
    assert report["faults"] == []
    assert all(value == 0 for value in report["counters"].values())
    assert all(row["compact_integrity_abi"]["handle_bits"] == 128 for row in report["rows"])


def test_v6_campaign_audit_is_green_and_call_free() -> None:
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


def test_v6_task_uses_fresh_receipt_namespace_without_hidden_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pool = p2a.read_json(task_pool.DEFAULT_OUT)
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
        "semantic_ir_adequacy_fresh_v6_01",
        "semantic_ir_adequacy_fresh_v6_01",
    ]
    assert result["index"] == 1
    assert result["hidden_evaluator_executions"] == 0


def test_v6_scorer_fails_closed_before_complete_candidate_seal(tmp_path: Path) -> None:
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


def test_v6_watchdog_remains_infrastructure_not_capability() -> None:
    telemetry = [{
        "termination_reason": "host_safety_wall_time",
        "host_safety_wall_time_hit": True,
        "safety_ceiling_hit": True,
    }]
    assert campaign.completion_fault(1, 1, telemetry, {"ready": False}) == (
        "host_watchdog_infrastructure_invalid:task_01:call_1"
    )


def test_v6_terminal_disposition_freezes_only_the_exact_implementation() -> None:
    report = disposition.dispose()
    assert report["trigger_state"] == "GREEN"
    assert report["scientific_status"] == "INCONCLUSIVE_EXPERIMENT"
    assert report["implementation_disposition"] == "FROZEN_FOR_CURRENT_TMAX_HOST_BLOCK"
    assert report["claim_effect_decision_authorized"] is False
    assert report["book_support_effect"] == "none"
    assert report["preserved_candidate_indices"] == [1]
    assert report["consumed_unsealed_indices"] == [2]
    assert report["hidden_evaluation_opened"] is False
    assert report["observation"]["exact_prompt_tokens"] == 45_113
    assert report["observation"]["generated_tokens"] == 0
    assert report["portfolio_transition"]["next_claim_id"] == "virtual-context-abi.core"
    assert report["portfolio_transition"]["semantic_ir_fresh_reseal_authorized_in_current_block"] is False
    assert report["portfolio_transition"]["next_stage_model_calls_authorized"] == 0
    assert report["faults"] == []
