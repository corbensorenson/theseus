#!/usr/bin/env python3
"""Compact packet for the one-shot public transfer calibration goal.

This report does not run public calibration, train on public artifacts, or emit
public prompts/tests/solutions/candidate code. It only consolidates the already
consumed guarded run, aggregate failure mining, private target manifest, dogfood
metadata bridge, and Mac/VCM acceleration evidence.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RUN_ID = "industry_code_transfer_seed14_5x64_v1"


DEFAULTS = {
    "operator_execute": REPORTS / f"operator_bounded_public_calibration_{RUN_ID}.json",
    "operator_refusal": REPORTS / f"operator_bounded_public_calibration_{RUN_ID}_preflight.json",
    "registry": REPORTS / "public_benchmark_run_registry.jsonl",
    "calibration": REPORTS / f"real_code_benchmark_graduation_{RUN_ID}.json",
    "residual": REPORTS / f"public_code_transfer_residual_report_{RUN_ID}.json",
    "bounded_mining": REPORTS / f"bounded_public_transfer_residual_mining_{RUN_ID}.json",
    "private_targets": REPORTS / f"bounded_public_transfer_private_residual_targets_{RUN_ID}.jsonl",
    "private_target_consumer": REPORTS / "private_residual_target_consumer_v1.json",
    "candidate_floor_probe": REPORTS / "candidate_floor_v2_private_token_probe.json",
    "broad_matrix": REPORTS / f"broad_transfer_matrix_{RUN_ID}.json",
    "survival_decision": REPORTS / "broad_capability_survival_lane_decision_after_public_transfer_v1.json",
    "dogfood_bridge": REPORTS / "dogfood_trace_training_bridge_one_shot_public_transfer_v1_execute.json",
    "dogfood_events": ROOT / "runtime" / "dogfood" / "daily_use_events.jsonl",
    "dogfood_training_rows": ROOT
    / "data"
    / "training_data"
    / "high_transfer"
    / "private_train"
    / "dogfood_daily_use_trace_training_rows.jsonl",
    "mlx_diagnosis": REPORTS / "macos_mlx_environment_diagnosis_one_shot_public_transfer_v1.json",
    "metal_readiness": REPORTS / "macos_metal_production_route_readiness.json",
    "vcm_runtime": REPORTS / "vcm_native_runtime_probe_one_shot_public_transfer_v1.json",
    "sts_policy": REPORTS / "sts_ranker_policy_v1.json",
    "sts_broad_audit": REPORTS / "sts_broad_regression_audit_v1.json",
    "vcm_structural": REPORTS / "broad_capability_structural_vcm_ablation_v1.json",
    "vcm_broad": REPORTS / "broad_capability_vcm_feature_ablation_v1.json",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=rel(REPORTS / "one_shot_public_transfer_calibration_failure_mining_v1.json"))
    parser.add_argument("--markdown-out", default=rel(REPORTS / "one_shot_public_transfer_calibration_failure_mining_v1.md"))
    args = parser.parse_args()

    report = build_report()
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report() -> dict[str, Any]:
    operator_execute = read_json(DEFAULTS["operator_execute"])
    operator_refusal = read_json(DEFAULTS["operator_refusal"])
    registry_rows = read_jsonl(DEFAULTS["registry"])
    calibration = read_json(DEFAULTS["calibration"])
    residual = read_json(DEFAULTS["residual"])
    bounded = read_json(DEFAULTS["bounded_mining"])
    target_consumer = read_json(DEFAULTS["private_target_consumer"])
    candidate_floor = read_json(DEFAULTS["candidate_floor_probe"])
    broad_matrix = read_json(DEFAULTS["broad_matrix"])
    survival = read_json(DEFAULTS["survival_decision"])
    dogfood_bridge = read_json(DEFAULTS["dogfood_bridge"])
    dogfood_events = read_jsonl(DEFAULTS["dogfood_events"])
    dogfood_training = read_jsonl(DEFAULTS["dogfood_training_rows"])
    mlx = read_json(DEFAULTS["mlx_diagnosis"])
    metal = read_json(DEFAULTS["metal_readiness"])
    vcm_runtime = read_json(DEFAULTS["vcm_runtime"])
    sts_policy = read_json(DEFAULTS["sts_policy"])
    sts_broad = read_json(DEFAULTS["sts_broad_audit"])
    vcm_structural = read_json(DEFAULTS["vcm_structural"])
    vcm_broad = read_json(DEFAULTS["vcm_broad"])
    target_rows = read_jsonl(DEFAULTS["private_targets"])

    registry_consumed_rows = [
        row
        for row in registry_rows
        if row.get("run_id") == RUN_ID and row.get("consumed") is True
    ]
    calibration_summary = obj(calibration, "summary")
    residual_summary = obj(residual, "summary")
    bounded_summary = obj(bounded, "summary")
    broad_summary = obj(broad_matrix, "summary")
    dogfood_summary = dogfood_counts(dogfood_events, dogfood_training, dogfood_bridge)

    score = score_summary(calibration)
    failure = failure_summary(residual_summary, bounded_summary)
    repair_readiness = repair_readiness_summary(target_consumer, candidate_floor)
    no_cheat = no_cheat_audit(
        operator_execute=operator_execute,
        calibration_summary=calibration_summary,
        residual_summary=residual_summary,
        bounded_summary=bounded_summary,
        broad_summary=broad_summary,
        dogfood_summary=dogfood_summary,
        target_rows=target_rows,
    )
    sts_vcm = sts_vcm_summary(sts_policy, sts_broad, vcm_structural, vcm_broad, broad_summary)
    mac = mac_summary(mlx, metal, vcm_runtime)
    survival_summary = survival_lane_summary(survival, score, failure, sts_vcm)

    gates = [
        gate("exactly_one_registry_consumed_run", len(registry_consumed_rows) == 1, {"count": len(registry_consumed_rows)}),
        gate("guarded_operator_execute_completed", get(operator_execute, "summary", "executed") is True and get(operator_execute, "summary", "run_returncode") == 0, get(operator_execute, "summary")),
        gate("guarded_no_rerun_refusal_recorded", refusal_recorded(operator_refusal), get(operator_refusal, "summary")),
        gate("calibration_score_loaded", score["task_count"] == 320, score),
        gate("per_card_all_five_loaded", len(score["per_card"]) == 5, score["per_card"]),
        gate("private_residual_targets_written", len(target_rows) > 0 and bounded_summary.get("training_rows_written") == 0, {"rows": len(target_rows), "manifest": rel(DEFAULTS["private_targets"])}),
        gate("private_residual_consumer_loaded", repair_readiness["consumer_loaded"], repair_readiness),
        gate(
            "candidate_floor_status_recorded",
            repair_readiness["candidate_floor_wall_recorded"]
            or repair_readiness["candidate_floor_semantic_quality_ready"],
            repair_readiness,
        ),
        gate(
            "promotion_not_claimed_without_public_transfer",
            score["promotion_allowed"] is False,
            {
                **repair_readiness,
                "public_score_promotion_allowed": score["promotion_allowed"],
            },
        ),
        gate("fallback_and_template_zero", no_cheat["fallback_return_count"] == 0 and no_cheat["template_like_candidate_count"] == 0, no_cheat),
        gate("public_training_leakage_zero", no_cheat["public_training_rows"] == 0 and not no_cheat["public_content_embedded"], no_cheat),
        gate("external_inference_zero_for_public_goal", no_cheat["public_goal_external_inference_calls"] == 0, no_cheat),
        gate("dogfood_metadata_only", dogfood_summary["raw_user_text_rows"] == 0 and dogfood_summary["public_benchmark_rows"] == 0, dogfood_summary),
        gate("mac_acceleration_checked", mac["mlx_state"] == "GREEN" and mac["vcm_runtime_state"] == "GREEN", mac),
        gate("survival_lane_recommendation_explicit", bool(survival_summary["recommendation"]), survival_summary),
    ]
    failed = [row for row in gates if not row["passed"]]

    return {
        "policy": "project_theseus_one_shot_public_transfer_calibration_failure_mining_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if not failed else "RED",
        "run_id": RUN_ID,
        "score": score,
        "failure_mining": failure,
        "private_residual_manifest": {
            "path": rel(DEFAULTS["private_targets"]),
            "row_count": len(target_rows),
            "training_rows": 0,
            "public_content_embedded": False,
        },
        "private_repair_readiness": repair_readiness,
        "sts_vcm_structural_comparison": sts_vcm,
        "survival_lane": survival_summary,
        "dogfood": dogfood_summary,
        "mac_acceleration": mac,
        "no_cheat_audit": no_cheat,
        "gates": gates,
        "inputs": {name: rel(path) for name, path in DEFAULTS.items()},
        "next_private_work": next_private_work(failure, survival_summary, dogfood_summary, mac, repair_readiness),
        "score_semantics": (
            "This is a consolidation report only. It consumes no new public benchmark run, "
            "writes no public-derived training rows, and embeds no public prompts/tests/solutions/traces/candidate code."
        ),
        "external_inference_calls": 0,
    }


def score_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = obj(report, "summary")
    task_count = int_number(summary.get("public_task_count"), summary.get("total_case_count"))
    pass_rate = float_number(summary.get("real_public_task_pass_rate"), summary.get("multi_stream_pass_rate"))
    per_card = []
    for suite in as_list(report.get("suites")):
        case_count = int_number(suite.get("case_count"))
        passed = int_number(suite.get("multi_stream_passed"))
        per_card.append(
            {
                "card_id": suite.get("card_id"),
                "passed": passed,
                "task_count": case_count,
                "pass_rate": round(passed / case_count, 6) if case_count else 0.0,
                "residual_count": int_number(suite.get("residual_count")),
            }
        )
    return {
        "current": {
            "passed": int(round(pass_rate * task_count)),
            "task_count": task_count,
            "pass_rate": pass_rate,
        },
        "previous_locked_baseline": {
            "passed": 34,
            "task_count": 160,
            "pass_rate": 0.2125,
        },
        "delta_vs_previous_locked_baseline": round(pass_rate - 0.2125, 6),
        "per_card": per_card,
        "promotion_allowed": bool(report.get("promotion_allowed")),
        "trigger_state": report.get("trigger_state"),
        "task_count": task_count,
    }


def failure_summary(residual_summary: dict[str, Any], bounded_summary: dict[str, Any]) -> dict[str, Any]:
    adjusted = dict_pairs(residual_summary.get("adapter_adjusted_dominant_categories"))
    raw = dict_pairs(residual_summary.get("raw_residual_counts"))
    bounded_counts = obj(bounded_summary, "residual_category_counts")
    no_admissible = obj(residual_summary, "adapter_adjusted_no_admissible")
    return {
        "dominant_categories": residual_summary.get("dominant_categories") or [],
        "adapter_adjusted_categories": residual_summary.get("adapter_adjusted_dominant_categories") or [],
        "raw_residual_counts": residual_summary.get("raw_residual_counts") or [],
        "requested_category_view": {
            "algorithm_choice": adjusted.get("algorithmic_planning", raw.get("algorithm_choice", 0)),
            "return_shape": adjusted.get("return_shape", raw.get("type_handling", 0)),
            "io_contract": adjusted.get("interface_fidelity", 0),
            "verifier_mismatch": adjusted.get("verifier_mismatch", raw.get("edge_case", 0)),
            "parsing_encoding": bounded_counts.get("parsing_syntax", 0),
            "timeout": bounded_counts.get("timeout_runtime", 0),
            "no_admissible_spent_run": int_number(no_admissible.get("total_no_admissible_residual_tasks")),
            "no_admissible_remaining_current_manifest": int_number(no_admissible.get("true_remaining_no_admissible_tasks")),
            "candidate_quality_after_current_adapter": adjusted.get("algorithmic_planning", 0)
            + adjusted.get("verifier_mismatch", 0)
            + adjusted.get("return_shape", 0)
            + adjusted.get("dependency_runtime_handling", 0),
            "sts_vcm_selection_miss": bounded_counts.get("selector_ranking_miss", 0),
        },
        "recommended_private_fix_family": residual_summary.get("recommended_private_fix_family"),
        "recommended_private_fix_family_after_current_adapter": residual_summary.get("recommended_private_fix_family_after_current_adapter"),
        "bounded_private_target_rows": int_number(bounded_summary.get("private_residual_target_rows_written")),
        "bounded_private_target_manifest": bounded_summary.get("private_residual_target_manifest"),
    }


def sts_vcm_summary(sts_policy: dict[str, Any], sts_broad: dict[str, Any], vcm_structural: dict[str, Any], vcm_broad: dict[str, Any], broad_summary: dict[str, Any]) -> dict[str, Any]:
    sts_policy_summary = obj(sts_policy, "summary")
    sts_broad_summary = obj(sts_broad, "summary")
    vcm_structural_summary = obj(vcm_structural, "summary")
    vcm_broad_summary = obj(vcm_broad, "summary")
    return {
        "public_run_sts_delta": float_number(broad_summary.get("real_public_sts_delta")),
        "sts_policy_state": sts_policy.get("trigger_state"),
        "sts_policy_selected_pass_rate": sts_policy_summary.get("sts_policy_selected_pass_rate"),
        "sts_policy_non_sts_selected_pass_rate": sts_policy_summary.get("non_sts_policy_selected_pass_rate"),
        "sts_policy_delta": sts_policy_summary.get("selected_pass_delta_sts_policy_minus_non_sts_policy"),
        "sts_broad_body_template_state": sts_broad.get("trigger_state"),
        "sts_broad_recommended_action": sts_broad_summary.get("recommended_action"),
        "sts_broad_root_cause": obj(sts_broad_summary, "root_cause").get("reason"),
        "vcm_structural_state": vcm_structural.get("trigger_state"),
        "vcm_structural_recommended_action": vcm_structural_summary.get("recommended_action"),
        "vcm_structural_transformer_delta": vcm_structural_summary.get("transformer_structural_only_delta"),
        "vcm_broad_body_template_state": vcm_broad.get("trigger_state"),
        "vcm_broad_recommended_action": vcm_broad_summary.get("recommended_action"),
        "interpretation": (
            "Use STS/VCM path-specifically: enable for promoted structural/full-body paths with positive evidence; "
            "keep disabled for old body-template selector paths where ablations show regressions."
        ),
    }


def survival_lane_summary(survival: dict[str, Any], score: dict[str, Any], failure: dict[str, Any], sts_vcm: dict[str, Any]) -> dict[str, Any]:
    architecture = obj(survival, "architecture_evidence")
    return {
        "decision_report_state": survival.get("trigger_state"),
        "decision": survival.get("decision"),
        "practical_hot_path": architecture.get("survival_lane") or "transformer_hybrid_structural_student",
        "symliquid_role": architecture.get("symliquid_role") or "bounded_matched_discovery_comparator_only",
        "public_score_allows_promotion": False,
        "public_score_reason": f"Current one-shot score {score['current']['passed']}/{score['current']['task_count']} is below the prior locked baseline and below any public transfer floor.",
        "recommendation": (
            "Do not promote from the public result. Keep transformer/hybrid structural student as the practical hot path, "
            "keep SymLiquid as a protected matched comparator, and focus private repair on semantic candidate quality: "
            f"{failure.get('recommended_private_fix_family_after_current_adapter') or failure.get('recommended_private_fix_family')}."
        ),
        "sts_vcm_policy": sts_vcm["interpretation"],
    }


def dogfood_counts(events: list[dict[str, Any]], training_rows: list[dict[str, Any]], bridge: dict[str, Any]) -> dict[str, Any]:
    event_outcomes = Counter(str(row.get("outcome") or "") for row in events)
    row_outcomes = Counter(str(row.get("outcome_label") or "") for row in training_rows)
    ids = [str(row.get("source_event_id") or "") for row in training_rows if row.get("source_event_id")]
    return {
        "event_count": len(events),
        "event_outcomes": dict(event_outcomes.most_common()),
        "training_row_count": len(training_rows),
        "unique_training_source_event_count": len(set(ids)),
        "duplicate_training_rows": len(ids) - len(set(ids)),
        "training_row_outcomes": dict(row_outcomes.most_common()),
        "bridge_state": bridge.get("trigger_state"),
        "bridge_write_blocker": bridge.get("write_blocker"),
        "bridge_new_training_event_count": get(bridge, "summary", "new_training_event_count"),
        "raw_user_text_rows": sum(1 for row in training_rows if row.get("raw_user_text_included") is True),
        "public_benchmark_rows": sum(1 for row in training_rows if row.get("public_benchmark_row") is True),
        "teacher_generated_rows": sum(1 for row in training_rows if row.get("teacher_generated") is True),
        "external_inference_calls": sum(int_number(row.get("external_inference_calls")) for row in training_rows),
        "next_needed": "record real missed/ignored/corrected/completed events; accepted-only metadata is too thin for strong daily-use learning pressure",
    }


def mac_summary(mlx: dict[str, Any], metal: dict[str, Any], vcm_runtime: dict[str, Any]) -> dict[str, Any]:
    mlx_summary = obj(mlx, "summary")
    metal_summary = obj(metal, "summary")
    vcm_summary = obj(vcm_runtime, "summary")
    return {
        "mlx_state": mlx.get("trigger_state"),
        "mlx_recommended_python": mlx_summary.get("recommended_python"),
        "mlx_route_action": mlx_summary.get("route_action"),
        "mlx_usable_runtime_count": mlx_summary.get("usable_mlx_runtime_count"),
        "metal_state": metal.get("trigger_state"),
        "metal_production_route_allowed": metal_summary.get("production_route_allowed"),
        "metal_kernel_parity_pending_count": metal_summary.get("kernel_parity_pending_count"),
        "native_hot_loop_parity_claim_allowed": metal_summary.get("native_hot_loop_parity_claim_allowed"),
        "vcm_runtime_state": vcm_runtime.get("trigger_state"),
        "vcm_native_runtime_claim_backend": vcm_summary.get("native_runtime_claim_backend"),
        "vcm_recommended_execution_backend": vcm_summary.get("recommended_execution_backend"),
        "vcm_mlx_native_kv_parity_claimed": vcm_summary.get("mlx_native_kv_parity_claimed"),
        "next_acceleration_step": "move selected hot loops to native MLX/Metal and run apples-to-apples CUDA/MLX/Metal comparisons before any parity claim",
    }


def repair_readiness_summary(target_consumer: dict[str, Any], candidate_floor: dict[str, Any]) -> dict[str, Any]:
    consumer_summary = obj(target_consumer, "summary")
    floor_summary = obj(candidate_floor, "summary")
    floor_pass_rate = float_number(
        consumer_summary.get("candidate_floor_private_trained_pass_rate"),
        floor_summary.get("private_trained_pass_rate"),
    )
    floor_task_count = int_number(
        consumer_summary.get("candidate_floor_private_eval_task_count"),
        floor_summary.get("private_eval_task_count"),
    )
    floor_passed = int_number(
        consumer_summary.get("candidate_floor_private_trained_passed"),
        floor_summary.get("private_trained_passed"),
    )
    semantic_ready = bool(
        consumer_summary.get("candidate_floor_semantic_quality_ready")
        if "candidate_floor_semantic_quality_ready" in consumer_summary
        else floor_pass_rate >= 0.70
    )
    unresolved = int_number(consumer_summary.get("unresolved_target_count"))
    weak_families = consumer_summary.get("candidate_floor_weak_families")
    weak_families = weak_families if isinstance(weak_families, list) else []
    return {
        "consumer_loaded": bool(target_consumer),
        "consumer_state": target_consumer.get("trigger_state"),
        "target_coverage_rate": float_number(consumer_summary.get("target_coverage_rate")),
        "covered_target_count": int_number(consumer_summary.get("covered_target_count")),
        "unresolved_target_count": unresolved,
        "unresolved_target_category_counts": consumer_summary.get("unresolved_target_category_counts") or {},
        "candidate_floor_probe_state": candidate_floor.get("trigger_state"),
        "candidate_floor_admissibility_ready": bool(consumer_summary.get("candidate_floor_admissibility_ready")),
        "candidate_floor_semantic_quality_ready": semantic_ready,
        "candidate_floor_private_trained_passed": floor_passed,
        "candidate_floor_private_eval_task_count": floor_task_count,
        "candidate_floor_private_trained_pass_rate": floor_pass_rate,
        "candidate_floor_weak_family_count": int_number(
            consumer_summary.get("candidate_floor_weak_family_count"),
            len(weak_families),
        ),
        "candidate_floor_weak_families": weak_families[:8],
        "candidate_floor_wall_recorded": bool(target_consumer)
        and floor_task_count > 0
        and floor_pass_rate < 0.70
        and unresolved > 0,
        "promotion_blocked_when_candidate_floor_weak": bool(
            floor_pass_rate < 0.70 and unresolved > 0
        ),
        "next_private_repair_focus": [
            "algorithm_choice",
            "return_type_shape",
            "verifier_mismatch",
            "selector_ranking_miss",
        ],
    }


def no_cheat_audit(**reports: Any) -> dict[str, Any]:
    calibration_summary = reports["calibration_summary"]
    residual_summary = reports["residual_summary"]
    bounded_summary = reports["bounded_summary"]
    broad_summary = reports["broad_summary"]
    dogfood_summary = reports["dogfood_summary"]
    target_rows = reports["target_rows"]
    target_public_flags = sum(
        1
        for row in target_rows
        if row.get("public_content_embedded") or row.get("public_prompt_embedded") or row.get("public_tests_embedded") or row.get("public_solution_embedded") or row.get("score_label_embedded") or row.get("training_row")
    )
    return {
        "public_goal_external_inference_calls": max(
            int_number(get(reports["operator_execute"], "external_inference_calls")),
            int_number(calibration_summary.get("external_inference_calls")),
            int_number(residual_summary.get("external_inference_calls")),
            int_number(bounded_summary.get("external_inference_calls")),
            int_number(broad_summary.get("external_inference_calls")),
            int_number(dogfood_summary.get("external_inference_calls")),
        ),
        "fallback_return_count": int_number(calibration_summary.get("expression_memory_fallback_count"))
        + int_number(bounded_summary.get("fallback_return_candidate_count")),
        "template_like_candidate_count": int_number(calibration_summary.get("template_like_candidate_count"))
        + int_number(bounded_summary.get("template_like_candidate_count")),
        "public_training_rows": int_number(bounded_summary.get("training_rows_written"))
        + int_number(dogfood_summary.get("public_benchmark_rows")),
        "public_content_embedded": bool(
            residual_summary.get("public_prompts_embedded")
            or residual_summary.get("public_tests_or_solutions_embedded")
            or bounded_summary.get("public_prompts_embedded")
            or bounded_summary.get("public_tests_embedded")
            or bounded_summary.get("public_solutions_embedded")
            or bounded_summary.get("candidate_code_embedded")
            or target_public_flags
        ),
        "private_target_public_flag_rows": target_public_flags,
        "raw_user_text_rows": int_number(dogfood_summary.get("raw_user_text_rows")),
        "teacher_runtime_serving_tokens": 0,
    }


def next_private_work(
    failure: dict[str, Any],
    survival: dict[str, Any],
    dogfood: dict[str, Any],
    mac: dict[str, Any],
    repair_readiness: dict[str, Any],
) -> list[str]:
    candidate_floor_action = (
        "Keep candidate_floor_v2 as a regression gate; it now clears the 0.70 private repair floor "
        f"at {repair_readiness['candidate_floor_private_trained_passed']}/"
        f"{repair_readiness['candidate_floor_private_eval_task_count']} = "
        f"{repair_readiness['candidate_floor_private_trained_pass_rate']}."
        if repair_readiness.get("candidate_floor_semantic_quality_ready")
        else (
            "Run a private-only semantic candidate-quality loop until candidate_floor_v2 improves above the "
            f"0.70 floor; current private analogue is {repair_readiness['candidate_floor_private_trained_passed']}/"
            f"{repair_readiness['candidate_floor_private_eval_task_count']} = "
            f"{repair_readiness['candidate_floor_private_trained_pass_rate']}."
        )
    )
    return [
        candidate_floor_action,
        (
            "Mine the remaining weak private singletons without public payloads: "
            f"{repair_readiness.get('candidate_floor_weak_families') or []}."
        ),
        "Freeze candidate manifests before any future public review; never rerun this consumed public surface.",
        survival["recommendation"],
        dogfood["next_needed"],
        mac["next_acceleration_step"],
    ]


def refusal_recorded(report: dict[str, Any]) -> bool:
    summary = obj(report, "summary")
    return bool(
        summary.get("mode") == "dry_run"
        and summary.get("executed") is False
        and summary.get("would_execute") is False
        and summary.get("operator_lock_present_after") is True
        and (
            summary.get("surface_not_consumed") is False
            or summary.get("output_absent_before_run") is False
            or summary.get("trace_absent_before_run") is False
        )
    )


def render_markdown(report: dict[str, Any]) -> str:
    score = report["score"]
    failure = report["failure_mining"]
    lines = [
        "# One-Shot Public Transfer Calibration And Failure Mining v1",
        "",
        f"- State: `{report['trigger_state']}`",
        f"- Run id: `{report['run_id']}`",
        f"- Public score: `{score['current']['passed']}/{score['current']['task_count']}` = `{score['current']['pass_rate']}`",
        f"- Previous locked baseline: `{score['previous_locked_baseline']['passed']}/{score['previous_locked_baseline']['task_count']}` = `{score['previous_locked_baseline']['pass_rate']}`",
        f"- Delta: `{score['delta_vs_previous_locked_baseline']}`",
        f"- Private residual manifest: `{report['private_residual_manifest']['path']}` rows=`{report['private_residual_manifest']['row_count']}`",
        "",
        "## Per Card",
    ]
    for row in score["per_card"]:
        lines.append(f"- `{row['card_id']}`: `{row['passed']}/{row['task_count']}` = `{row['pass_rate']}`")
    lines.extend(["", "## Failure Mining"])
    for key, value in failure["requested_category_view"].items():
        lines.append(f"- `{key}`: `{value}`")
    repair = report.get("private_repair_readiness") if isinstance(report.get("private_repair_readiness"), dict) else {}
    lines.extend([
        "",
        "## Private Repair Readiness",
        f"- Consumer state: `{repair.get('consumer_state')}`",
        f"- Target coverage: `{repair.get('covered_target_count')}` covered, `{repair.get('unresolved_target_count')}` unresolved",
        f"- Unresolved categories: `{repair.get('unresolved_target_category_counts')}`",
        f"- Candidate floor: `{repair.get('candidate_floor_private_trained_passed')}/{repair.get('candidate_floor_private_eval_task_count')}` = `{repair.get('candidate_floor_private_trained_pass_rate')}`",
        f"- Candidate-floor semantic ready: `{repair.get('candidate_floor_semantic_quality_ready')}`",
        f"- Weak family count: `{repair.get('candidate_floor_weak_family_count')}`",
    ])
    lines.extend([
        "",
        "## Recommendation",
        f"- {report['survival_lane']['recommendation']}",
        f"- STS/VCM: {report['sts_vcm_structural_comparison']['interpretation']}",
        f"- Dogfood: events=`{report['dogfood']['event_count']}`, rows=`{report['dogfood']['training_row_count']}`, unique_sources=`{report['dogfood']['unique_training_source_event_count']}`, new_unique_events=`{report['dogfood']['bridge_new_training_event_count']}`",
        f"- Mac: MLX=`{report['mac_acceleration']['mlx_state']}`, Metal=`{report['mac_acceleration']['metal_state']}`, VCM runtime=`{report['mac_acceleration']['vcm_runtime_state']}`",
        "",
        "## No-Cheat Audit",
    ])
    for key, value in report["no_cheat_audit"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Gates"])
    for row in report["gates"]:
        lines.append(f"- `{row['name']}`: `{row['passed']}`")
    lines.append("")
    return "\n".join(lines)


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    except Exception:
        return []
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def obj(value: Any, key: str) -> dict[str, Any]:
    item = value.get(key) if isinstance(value, dict) else {}
    return item if isinstance(item, dict) else {}


def get(value: Any, *path: str) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def dict_pairs(value: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in as_list(value):
        if isinstance(item, list) and len(item) >= 2:
            result[str(item[0])] = int_number(item[1])
    return result


def int_number(*values: Any) -> int:
    for value in values:
        try:
            if value is not None:
                return int(value)
        except Exception:
            continue
    return 0


def float_number(*values: Any) -> float:
    for value in values:
        try:
            if value is not None:
                return float(value)
        except Exception:
            continue
    return 0.0


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
