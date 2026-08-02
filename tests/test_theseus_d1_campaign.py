from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_d1_campaign as campaign  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "theseus_d1_campaign.json"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def survivor(plan_useful: int = 2, direct_useful: int = 1) -> dict:
    return {
        "policy": "project_theseus_p4v2r2r3_terminal_disposition_v1",
        "trigger_state": "GREEN",
        "scientific_status": "P4V2R2R2_DEVELOPMENT_SURVIVOR_D1_ELIGIBLE",
        "arm_totals": {
            "typed_semantic_ir_treatment": {"useful_candidates": 4},
            "direct_target_generation": {"useful_candidates": direct_useful},
            "natural_language_plan_control": {"useful_candidates": plan_useful},
        },
    }


def pool(tmp_path: Path) -> dict:
    rows = []
    for index in range(1, 45):
        task = tmp_path / f"task-{index}.json"
        evaluator = tmp_path / f"evaluator-{index}.json"
        task.write_text("{}\n", encoding="utf-8")
        evaluator.write_text("{}\n", encoding="utf-8")
        rows.append(
            {
                "campaign_index": index,
                "repository": f"owner/repo-{index}",
                "selection_digest": f"{index:064x}",
                "task": str(task),
                "task_sha256": p2a.sha256_file(task),
                "evaluator": str(evaluator),
                "evaluator_sha256": p2a.sha256_file(evaluator),
            }
        )
    return {
        "policy": "project_theseus_d1_sealed_task_pool_v1",
        "state": "SEALED_BEFORE_CANDIDATE_OR_CONTROL_GENERATION",
        "task_count": 44,
        "distinct_repository_count": 44,
        "tasks": rows,
        "candidate_or_control_calls": 0,
        "post_candidate_task_replacement_allowed": False,
        "project_selected_quality_token_cap": None,
    }


def test_preterminal_p4_keeps_D1_closed_without_calls(tmp_path: Path) -> None:
    report = campaign.audit_campaign(
        config(),
        config_path=CONFIG_PATH,
        disposition_override={},
        pool_override=pool(tmp_path),
        consumption_override=[],
        lease_exists_override=False,
    )
    assert report["trigger_state"] == "PAUSED"
    assert report["execution_authorized"] is False
    assert report["candidate_or_control_calls_before_final_pool_seal"] == 0


def test_survivor_and_pool_open_exactly_once_campaign_with_P4_selected_control(
    tmp_path: Path,
) -> None:
    surface = pool(tmp_path)
    report = campaign.audit_campaign(
        config(),
        config_path=CONFIG_PATH,
        disposition_override=survivor(plan_useful=3, direct_useful=2),
        pool_override=surface,
        consumption_override=[],
        lease_exists_override=False,
    )
    assert report["trigger_state"] == "GREEN"
    assert report["execution_authorized"] is True
    assert report["primary_control"] == "natural_language_plan_control"
    assert report["primary_control_selected_before_D1_outcomes"] is True
    assert report["project_selected_quality_token_cap"] is None


def test_control_tie_break_is_direct_and_never_uses_D1_outcomes() -> None:
    value = config()
    assert campaign.select_primary_control(survivor(2, 2), value) == (
        "direct_target_generation"
    )
    assert value["primary_arm_policy"]["D1_outcomes_may_change_selected_control"] is False


def test_consumed_pool_cannot_rerun(tmp_path: Path) -> None:
    surface = pool(tmp_path)
    pool_sha = p2a.stable_hash(surface)
    report = campaign.audit_campaign(
        config(),
        config_path=CONFIG_PATH,
        disposition_override=survivor(),
        pool_override=surface,
        consumption_override=[{"task_pool_sha256": pool_sha}],
        lease_exists_override=False,
    )
    assert report["activation_state"] == "D1_TASK_POOL_ALREADY_CONSUMED_RERUN_FORBIDDEN"
    assert report["execution_authorized"] is False


def test_config_has_no_user_gate_external_authority_or_quality_cap() -> None:
    value = config()
    assert campaign.validate_config(value) == []
    assert value["authority"]["user_or_operator_approval_required"] is False
    assert value["authority"]["external_inference_calls"] == 0
    assert value["matching"]["project_selected_quality_token_cap"] is None
    assert value["matching"]["project_selected_first_artifact_character_cap"] is None
    assert value["matching"]["project_selected_first_artifact_token_cap"] is None
    assert value["matching"]["complete_visible_verifier_feedback_visible_to_second_call"] is True
    assert value["matching"]["project_selected_verifier_feedback_character_cap"] is None
