from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_d1_evaluator as evaluator  # noqa: E402
import theseus_d1_terminal_disposition as disposition  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "theseus_d1_terminal_disposition.json"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def pool(tmp_path: Path) -> dict:
    rows = []
    for index in range(1, 45):
        task = tmp_path / f"task-{index}.json"
        manifest = tmp_path / f"evaluator-{index}.json"
        task.write_text("{}\n", encoding="utf-8")
        manifest.write_text("{}\n", encoding="utf-8")
        rows.append(
            {
                "campaign_index": index,
                "repository": f"owner/repo-{index}",
                "selection_digest": f"{index:064x}",
                "task": str(task),
                "task_sha256": p2a.sha256_file(task),
                "evaluator": str(manifest),
                "evaluator_sha256": p2a.sha256_file(manifest),
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


def result(arm: str, useful: int) -> dict:
    return {
        "arm_id": arm,
        "useful": useful,
        "unsafe": 0,
        "rollback_verified": True,
        "boundary_hit": False,
        "integrity_faults": [],
        "sandbox_receipt": {"duration_ms": 1.0, "boundary_hit": False},
    }


def evaluation(index: int) -> dict:
    treatment_useful = int(index <= 30)
    control_useful = int(31 <= index <= 40)
    return {
        "policy": evaluator.POLICY,
        "trigger_state": "GREEN",
        "evaluation_blinding": {
            "arm_labels_passed_to_scoring": False,
            "arm_labels_attached_after_scoring": True,
            "candidate_emitted_integrity_flags_trusted": False,
            "target_or_test_source_visible_to_generation": False,
        },
        "results": [
            result("typed_semantic_ir_treatment", treatment_useful),
            result("direct_target_generation", control_useful),
            result("natural_language_plan_control", 0),
            result("deterministic_request_compiler_baseline", 0),
        ],
        "oracle_ceiling": {"useful": 1},
    }


def run() -> dict:
    def call(runtime_ms: float, prompt_tokens: int, generated_tokens: int) -> dict:
        return {
            "runtime_ms": runtime_ms,
            "metrics": {
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "model_context_window_tokens": 262144,
                "effective_maximum_tokens": 262144 - prompt_tokens,
                "project_selected_quality_token_cap": None,
                "physical_context_boundary_hit": False,
            },
        }

    return {
        "attempts": [
            {
                "arm_id": "typed_semantic_ir_treatment",
                "runtime_calls": [call(2.0, 100, 20), call(2.0, 130, 25)],
            },
            {
                "arm_id": "direct_target_generation",
                "runtime_calls": [call(1.0, 90, 20), call(1.0, 120, 25)],
            },
            {
                "arm_id": "natural_language_plan_control",
                "runtime_calls": [call(1.0, 95, 20), call(1.0, 125, 25)],
            },
        ],
        "deterministic_compiler_control": {"runtime_calls": []},
    }


def test_preterminal_report_makes_no_D1_inference() -> None:
    report = disposition.build_report(
        config(), progress_override={}, pool_override={}, consumption_override=[]
    )
    assert report["trigger_state"] == "RED"
    assert report["scientific_status"] == "D1_REVIEW_REQUIRED"
    assert report["faults"] == ["campaign_not_complete"]


def test_complete_adequate_effect_qualifies_only_exact_implementation(
    tmp_path: Path,
) -> None:
    surface = pool(tmp_path)
    progress = {
        "policy": "project_theseus_d1_blind_matched_campaign_v1",
        "complete_tasks": 44,
        "primary_control": "direct_target_generation",
        "model_calls_retained": 264,
        "physical_context_boundary_hits": 0,
        "tasks": [{"campaign_index": index} for index in range(1, 45)],
    }
    evaluations = {index: evaluation(index) for index in range(1, 45)}
    runs = {index: run() for index in range(1, 45)}
    pool_sha = p2a.stable_hash(surface)
    report = disposition.build_report(
        config(),
        progress_override=progress,
        pool_override=surface,
        consumption_override=[{"task_pool_sha256": pool_sha}],
        evaluation_overrides=evaluations,
        run_overrides=runs,
    )
    assert report["trigger_state"] == "GREEN"
    assert report["scientific_status"] == "D1_EXACT_IMPLEMENTATION_QUALIFIED"
    assert report["primary_test"]["discordant_pairs"] == 40
    assert report["primary_test"]["treatment_wins"] == 30
    assert report["primary_test"]["qualification_rule_passed"] is True
    assert report["denominators"]["project_selected_quality_token_cap"] is None
    assert report["automatic_book_support_promotion"] is False
    cost = report["primary_test"]["joined_cost"]
    assert cost["typed_semantic_ir_treatment"]["prompt_tokens"] == 44 * 230
    assert cost["typed_semantic_ir_treatment"]["generated_tokens"] == 44 * 45
    assert report["adequacy"]["cost_custody_passed"] is True


def test_status_precedence_is_fail_closed_and_scoped() -> None:
    common = {
        "source_custody_green": True,
        "information_flow_green": True,
        "boundary_hits": 0,
        "rollback_failures": 0,
        "experiment_floor": True,
        "effect_rule": True,
    }
    assert disposition.classify_status(
        **{**common, "source_custody_green": False}
    ) == "INVALID_SOURCE_OR_CONSUMPTION_CUSTODY"
    assert disposition.classify_status(
        **{**common, "information_flow_green": False}
    ) == "INVALID_INFORMATION_FLOW"
    assert disposition.classify_status(
        **{**common, "boundary_hits": 1}
    ) == "INVALID_OBSERVATION_CONTEXT_OR_HOST_BOUNDARY"
    assert disposition.classify_status(
        **{**common, "rollback_failures": 1}
    ) == "INCONCLUSIVE_IMPLEMENTATION"
    assert disposition.classify_status(
        **{**common, "experiment_floor": False}
    ) == "INCONCLUSIVE_EXPERIMENT"
    assert disposition.classify_status(
        **{**common, "effect_rule": False}
    ) == "D1_EXACT_IMPLEMENTATION_NOT_QUALIFIED"


def test_exact_sign_probability_matches_predeclared_tail() -> None:
    assert round(disposition.binomial_upper_tail(18, 13, 0.5), 10) == 0.0481262207


def test_config_forbids_automatic_support_user_gate_and_quality_cap() -> None:
    value = config()
    assert disposition.validate_config(value) == []
    assert value["authority"]["automatic_book_support_promotion"] is False
    assert value["authority"]["user_or_operator_approval_required"] is False
    assert value["adequacy"]["project_selected_quality_token_cap"] is None
