from __future__ import annotations

import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_semantic_ir_production_adequacy_campaign as campaign  # noqa: E402
import theseus_semantic_ir_production_adequacy_scorer as scorer  # noqa: E402


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_prospective_campaign_audit_is_green_and_call_free() -> None:
    report = campaign.audit_config(campaign.DEFAULT_CONFIG)

    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["sealed_task_count"] == 18
    assert report["candidate_generation_opened"] is False
    assert report["frozen_model_contract"]["ready"] is True
    assert all(value == 0 for value in report["counters"].values())


def test_campaign_has_no_hidden_evaluator_and_no_quality_token_cap() -> None:
    runner_source = Path(campaign.__file__).read_text(encoding="utf-8")
    config = read(campaign.DEFAULT_CONFIG)

    assert "adequacy_evaluator" not in runner_source
    assert config["generation_completion"]["project_selected_quality_token_cap"] is None
    assert config["generation_completion"]["normal_completion"] == [
        "parser_complete",
        "model_eos",
    ]
    assert config["authority"]["external_inference_authorized"] is False
    assert config["authority"]["user_or_operator_gate"] is False


def test_campaign_report_seals_only_complete_denominator() -> None:
    audit = campaign.audit_config(campaign.DEFAULT_CONFIG)
    rows = [{"runtime_calls": [{}, {}]} for _ in range(18)]
    report = campaign.campaign_report(
        campaign.DEFAULT_CONFIG,
        audit,
        rows,
        [],
        time.perf_counter(),
        terminal=True,
    )

    assert report["trigger_state"] == "GREEN"
    assert report["state"] == "CANDIDATES_SEALED_BEFORE_HIDDEN_EVALUATION"
    assert report["completed_task_count"] == 18
    assert report["hidden_evaluation_opened"] is False
    assert report["counters"]["local_model_calls"] == 36
    assert report["counters"]["external_inference_calls"] == 0


def test_run_task_uses_exact_packet_then_non_answer_bearing_repair(
    monkeypatch,
) -> None:
    pool = read(
        ROOT / "reports" / "theseus_semantic_ir_production_adequacy_task_pool.json"
    )
    pool_row = pool["rows"][0]
    packet = read(ROOT / pool_row["candidate_packet"])
    prompts: list[str] = []

    def fake_runtime_call(
        arm: str,
        task_id: str,
        call_number: int,
        prompt: str,
        maximum: int,
        runtime_config: str,
    ) -> dict:
        prompts.append(prompt)
        output = "not a semantic artifact"
        return {
            "assistant_text": output,
            "runtime_report": {
                "route_integrity": {"ready": True, "release_allowed": True}
            },
            "receipt": {
                "call_number": call_number,
                "candidate_output_sha256": p2a.sha256_text(output),
            },
        }

    monkeypatch.setattr(p2a, "runtime_call", fake_runtime_call)
    monkeypatch.setattr(
        campaign.p4r,
        "termination_telemetry",
        lambda attempts: [
            {
                "termination_reason": "model_eos",
                "safety_ceiling_hit": False,
            },
            {
                "termination_reason": "model_eos",
                "safety_ceiling_hit": False,
            },
        ],
    )

    row = campaign.run_task(pool_row, "configs/theseus_assistant_runtime.json")

    assert len(prompts) == 2
    assert prompts[0] == packet["serialized_prompt"]
    assert "[COMPLETE_PROVISIONAL_ARTIFACT]" in prompts[1]
    assert "[COMPLETE_VISIBLE_FEEDBACK]" in prompts[1]
    assert row["candidate_seal"]["sealed_before_hidden_evaluation"] is True
    assert row["hidden_evaluator_executions"] == 0


def test_independent_scorer_fails_closed_before_evaluator_on_unsealed_run(
    tmp_path: Path,
) -> None:
    run_path = tmp_path / "unsealed.json"
    p2a.write_json(run_path, {
        "policy": campaign.POLICY,
        "state": "RUNNING_CANDIDATE_GENERATION",
        "trigger_state": "YELLOW",
        "completed_task_count": 0,
        "hidden_evaluation_opened": False,
        "rows": [],
        "counters": campaign.zero_counters(),
    })

    report = scorer.score(run_path)

    assert report["trigger_state"] == "RED"
    assert report["scientific_status"] == "INVALID_INFORMATION_FLOW"
    assert report["task_count"] == 0
    assert report["counters"]["hidden_evaluator_executions"] == 0
    assert "candidate_run_not_sealed_for_hidden_evaluation" in report["faults"]


def test_independent_scorer_uses_six_bound_strata() -> None:
    materialization = read(scorer.MATERIALIZATION)
    strata = {str(row["stratum"]) for row in materialization["rows"]}

    assert len(strata) == 6
    assert all(
        sum(row["stratum"] == stratum for row in materialization["rows"]) == 3
        for stratum in strata
    )
