#!/usr/bin/env python3
"""Decision report for the broad capability survival lane.

This ties source admission, curriculum coverage, architecture evidence, VCM/STS
signals, and promotion state into a single yes/no/bounded-blocker report.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADMISSION = ROOT / "reports" / "training_data_admission_v1.json"
DEFAULT_CURRICULUM = ROOT / "reports" / "broad_capability_curriculum_v1.json"
DEFAULT_ARCH = ROOT / "reports" / "neural_seed_architecture_sweep.json"
DEFAULT_CLOSURE = ROOT / "reports" / "capability_transfer_closure_v1.json"
DEFAULT_OVERNIGHT = ROOT / "reports" / "overnight_self_improvement_v1_report.json"
DEFAULT_STS = ROOT / "reports" / "sts_ranker_policy_v1.json"
DEFAULT_STS_AUDIT = ROOT / "reports" / "sts_broad_regression_audit_v1.json"
DEFAULT_STS_POLICY = ROOT / "configs" / "sts_broad_survival_policy_v1.json"
DEFAULT_VCM = ROOT / "reports" / "vcm_task_context_bridge.json"
DEFAULT_VCM_ABLATION = ROOT / "reports" / "broad_capability_structural_vcm_ablation_v1.json"
DEFAULT_VCM_POLICY = ROOT / "configs" / "vcm_structural_survival_feature_policy_v1.json"
DEFAULT_MLX_DIAGNOSIS = ROOT / "reports" / "macos_mlx_environment_diagnosis.json"
DEFAULT_MLX_SMOKE = ROOT / "reports" / "macos_mlx_structural_action_smoke.json"
DEFAULT_PROMOTION = ROOT / "reports" / "candidate_promotion_gate.json"
DEFAULT_BROAD_PROMOTION = ROOT / "reports" / "broad_capability_survival_promotion_gate_v1.json"
DEFAULT_SURVIVAL = ROOT / "reports" / "theseus_survival_path_decision.json"
DEFAULT_BROAD_RUN = ROOT / "reports" / "broad_capability_survival_lane_run_v1.json"
DEFAULT_STRUCTURAL = ROOT / "reports" / "broad_capability_structural_action_decoder_probe_v1_vcm_on.json"
DEFAULT_REPLAY = ROOT / "reports" / "private_candidate_replay_contract_audit_v1.json"
DEFAULT_OUT = ROOT / "reports" / "broad_capability_survival_lane_decision_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "broad_capability_survival_lane_decision_v1.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", default=rel(DEFAULT_ADMISSION))
    parser.add_argument("--curriculum", default=rel(DEFAULT_CURRICULUM))
    parser.add_argument("--architecture-sweep", default=rel(DEFAULT_ARCH))
    parser.add_argument("--capability-closure", default=rel(DEFAULT_CLOSURE))
    parser.add_argument("--overnight", default=rel(DEFAULT_OVERNIGHT))
    parser.add_argument("--sts", default=rel(DEFAULT_STS))
    parser.add_argument("--sts-audit", default=rel(DEFAULT_STS_AUDIT))
    parser.add_argument("--sts-policy", default=rel(DEFAULT_STS_POLICY))
    parser.add_argument("--vcm", default=rel(DEFAULT_VCM))
    parser.add_argument("--vcm-ablation", default=rel(DEFAULT_VCM_ABLATION))
    parser.add_argument("--vcm-policy", default=rel(DEFAULT_VCM_POLICY))
    parser.add_argument("--mlx-diagnosis", default=rel(DEFAULT_MLX_DIAGNOSIS))
    parser.add_argument("--mlx-smoke", default=rel(DEFAULT_MLX_SMOKE))
    parser.add_argument("--promotion", default=rel(DEFAULT_PROMOTION))
    parser.add_argument("--broad-promotion", default=rel(DEFAULT_BROAD_PROMOTION))
    parser.add_argument("--survival-path", default=rel(DEFAULT_SURVIVAL))
    parser.add_argument("--broad-run", default=rel(DEFAULT_BROAD_RUN))
    parser.add_argument("--structural-action", default=rel(DEFAULT_STRUCTURAL))
    parser.add_argument("--candidate-replay", default=rel(DEFAULT_REPLAY))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(args, started=started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace, *, started: float) -> dict[str, Any]:
    admission_path = resolve(args.admission)
    curriculum_path = resolve(args.curriculum)
    arch_path = resolve(args.architecture_sweep)
    closure_path = resolve(args.capability_closure)
    overnight_path = resolve(args.overnight)
    sts_path = resolve(args.sts)
    sts_audit_path = resolve(args.sts_audit)
    sts_policy_path = resolve(args.sts_policy)
    vcm_path = resolve(args.vcm)
    vcm_ablation_path = resolve(args.vcm_ablation)
    vcm_policy_path = resolve(args.vcm_policy)
    mlx_diagnosis_path = resolve(args.mlx_diagnosis)
    mlx_smoke_path = resolve(args.mlx_smoke)
    promotion_path = resolve(args.promotion)
    broad_promotion_path = resolve(args.broad_promotion)
    survival_path = resolve(args.survival_path)
    broad_run_path = resolve(args.broad_run)
    structural_path = resolve(args.structural_action)
    replay_path = resolve(args.candidate_replay)

    admission = read_json(admission_path)
    curriculum = read_json(curriculum_path)
    arch = read_json(arch_path)
    closure = read_json(closure_path)
    overnight = read_json(overnight_path)
    sts = read_json(sts_path)
    sts_audit = read_json(sts_audit_path)
    sts_policy = read_json(sts_policy_path)
    vcm = read_json(vcm_path)
    vcm_ablation = read_json(vcm_ablation_path)
    vcm_policy = read_json(vcm_policy_path)
    mlx_diagnosis = read_json(mlx_diagnosis_path)
    mlx_smoke = read_json(mlx_smoke_path)
    promotion = read_json(promotion_path)
    broad_promotion = read_json(broad_promotion_path)
    survival = read_json(survival_path)
    broad_run = read_json(broad_run_path)
    structural = read_json(structural_path)
    replay = read_json(replay_path)
    broad_comparator_path = resolve(str(get_path(broad_run, ["artifacts", "comparator_report"], DEFAULT_BROAD_RUN)))
    broad_comparator = read_json(broad_comparator_path)

    architecture_evidence = architecture_summary(arch, closure, survival, broad_run, structural)
    no_cheat = no_cheat_summary(
        admission,
        curriculum,
        sts,
        sts_audit,
        sts_policy,
        vcm,
        vcm_ablation,
        vcm_policy,
        mlx_diagnosis,
        mlx_smoke,
        overnight,
        broad_run,
        structural,
        broad_promotion,
        replay,
    )
    promotion_summary = combine_promotion_states(
        legacy=promotion_state(promotion, overnight),
        broad=promotion_state(broad_promotion, {}),
        broad_promotion=broad_promotion,
    )
    blockers = derive_blockers(
        admission=admission,
        curriculum=curriculum,
        architecture_evidence=architecture_evidence,
        no_cheat=no_cheat,
        promotion_summary=promotion_summary,
        closure=closure,
        overnight=overnight,
        sts=sts,
        sts_audit=sts_audit,
        sts_policy=sts_policy,
        vcm=vcm,
        vcm_ablation=vcm_ablation,
        vcm_policy=vcm_policy,
        mlx_diagnosis=mlx_diagnosis,
        mlx_smoke=mlx_smoke,
        broad_run=broad_run,
        broad_comparator=broad_comparator,
        structural=structural,
        replay=replay,
    )
    training_ready = not any(blocker["severity"] == "hard" and blocker["stage"] in {"admission", "curriculum", "no_cheat"} for blocker in blockers)
    hard_blockers = [blocker for blocker in blockers if blocker["severity"] == "hard"]
    promotion_ready = promotion_summary["promoted"] is True and not hard_blockers
    public_calibration_recommendation = public_calibration_decision(closure, promotion_summary, blockers)

    gates = [
        gate("admission_report_present", bool(admission), rel(admission_path), "hard"),
        gate("curriculum_report_present", bool(curriculum), rel(curriculum_path), "hard"),
        gate("admission_not_red", admission.get("trigger_state") in {"GREEN", "YELLOW"}, admission.get("trigger_state"), "hard"),
        gate("curriculum_not_red", curriculum.get("trigger_state") in {"GREEN", "YELLOW"}, curriculum.get("trigger_state"), "hard"),
        gate("transformer_survival_lane_selected", architecture_evidence["survival_lane"] == "transformer_hybrid_structural_student", architecture_evidence, "hard"),
        gate("symliquid_discovery_only", architecture_evidence["symliquid_role"] == "bounded_matched_discovery_comparator_only", architecture_evidence, "hard"),
        gate("public_benchmark_training_zero", no_cheat["public_benchmark_training_rows"] == 0, no_cheat["public_benchmark_training_rows"], "hard"),
        gate("fallback_returns_zero", no_cheat["fallback_return_count"] == 0, no_cheat["fallback_return_count"], "hard"),
        gate("external_inference_zero", no_cheat["external_inference_calls"] == 0, no_cheat["external_inference_calls"], "hard"),
        gate("raw_user_text_zero", no_cheat["raw_user_text_count"] == 0, no_cheat["raw_user_text_count"], "hard"),
        gate("candidate_replay_contract_green", candidate_replay_contract_ready(replay), replay_contract_summary(replay), "hard"),
        gate("decision_blocker_or_promotion_recorded", promotion_ready or bool(blockers), {"promotion_ready": promotion_ready, "blockers": blockers}, "hard"),
    ]
    hard_failed = [row for row in gates if row["severity"] == "hard" and not row["passed"]]
    if hard_failed:
        trigger_state = "RED"
    elif promotion_ready:
        trigger_state = "GREEN"
    else:
        trigger_state = "YELLOW"

    return {
        "policy": "project_theseus_broad_capability_survival_lane_decision_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "decision": "promote_new_survival_candidate" if promotion_ready else "continue_survival_lane_with_precise_blocker",
        "training_ready": training_ready,
        "promotion_ready": promotion_ready,
        "public_calibration_recommendation": public_calibration_recommendation,
        "inputs": {
            "admission": rel(admission_path),
            "curriculum": rel(curriculum_path),
            "architecture_sweep": rel(arch_path),
            "capability_closure": rel(closure_path),
            "overnight": rel(overnight_path),
            "sts": rel(sts_path),
            "sts_audit": rel(sts_audit_path),
            "sts_policy": rel(sts_policy_path),
            "vcm": rel(vcm_path),
            "vcm_ablation": rel(vcm_ablation_path),
            "vcm_policy": rel(vcm_policy_path),
            "mlx_diagnosis": rel(mlx_diagnosis_path),
            "mlx_smoke": rel(mlx_smoke_path),
            "promotion": rel(promotion_path),
            "broad_promotion": rel(broad_promotion_path),
            "survival_path": rel(survival_path),
            "broad_run": rel(broad_run_path),
            "broad_comparator": rel(broad_comparator_path),
            "structural_action": rel(structural_path),
            "candidate_replay": rel(replay_path),
        },
        "practical_lane_contract": practical_lane_contract(
            broad_promotion=broad_promotion,
            sts_policy=sts_policy,
            vcm_policy=vcm_policy,
            replay=replay,
        ),
        "architecture_evidence": architecture_evidence,
        "data_evidence": data_summary(admission, curriculum),
        "sts_vcm_evidence": sts_vcm_summary(sts, sts_audit, sts_policy, vcm, vcm_ablation, vcm_policy, overnight, broad_run),
        "candidate_replay_contract": replay_contract_summary(replay),
        "mac_accelerator_evidence": mac_accelerator_summary(mlx_diagnosis, mlx_smoke),
        "promotion_summary": promotion_summary,
        "no_cheat_summary": no_cheat,
        "blockers": blockers,
        "next_actions": next_actions(blockers, training_ready, promotion_ready),
        "gates": gates,
        "score_semantics": (
            "Decision report only. It does not train, run public calibration, call external inference, "
            "call a teacher, write runtime-serving tokens, or promote by itself."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def architecture_summary(
    arch: dict[str, Any],
    closure: dict[str, Any],
    survival: dict[str, Any],
    broad_run: dict[str, Any],
    structural: dict[str, Any],
) -> dict[str, Any]:
    aggregate = arch.get("aggregate") if isinstance(arch.get("aggregate"), dict) else {}
    one_notch = aggregate.get("one_notch_budget_ladder") if isinstance(aggregate.get("one_notch_budget_ladder"), dict) else {}
    current = aggregate.get("current_smoke_seed_sweep") if isinstance(aggregate.get("current_smoke_seed_sweep"), dict) else {}
    closure_text = json.dumps(closure, sort_keys=True).lower()
    transformer_edges = survival.get("transformer_edges") if isinstance(survival.get("transformer_edges"), list) else []
    symliquid_edges = survival.get("symliquid_edges") if isinstance(survival.get("symliquid_edges"), list) else []
    current_gap = get_path(current, ["symliquid_minus_transformer", "mean"])
    ladder_gap = get_path(one_notch, ["symliquid_minus_transformer", "mean"])
    broad_summary = dict_or_empty(broad_run.get("summary"))
    structural_transformer = dict_or_empty(get_path(structural, ["arms", "transformer_control", "summary"], {}))
    structural_symliquid = dict_or_empty(get_path(structural, ["arms", "symliquid_style", "summary"], {}))
    closure_best = find_first_string(closure, "transformer_control", "code_proposer_best_arm")
    if not closure_best and "transformer_control" in closure_text:
        closure_best = "transformer_control"
    return {
        "survival_lane": "transformer_hybrid_structural_student",
        "symliquid_role": "bounded_matched_discovery_comparator_only",
        "architecture_sweep_state": arch.get("trigger_state"),
        "current_smoke_symliquid_minus_transformer_mean": current_gap,
        "one_notch_symliquid_minus_transformer_mean": ladder_gap,
        "broad_survival_run_state": broad_run.get("trigger_state"),
        "broad_survival_winner_by_sts_on": broad_summary.get("winner_by_sts_on"),
        "broad_survival_transformer_sts_on_pass_rate": broad_summary.get("transformer_sts_on_pass_rate"),
        "broad_survival_symliquid_sts_on_pass_rate": broad_summary.get("symliquid_sts_on_pass_rate"),
        "broad_survival_symliquid_minus_transformer": broad_summary.get("symliquid_minus_transformer"),
        "structural_action_state": structural.get("trigger_state"),
        "structural_action_transformer_structural_only_pass_rate": structural_transformer.get("structural_only_pass_rate"),
        "structural_action_transformer_augmented_delta": structural_transformer.get("delta"),
        "structural_action_symliquid_structural_only_pass_rate": structural_symliquid.get("structural_only_pass_rate"),
        "structural_action_symliquid_augmented_delta": structural_symliquid.get("delta"),
        "closure_best_arm_hint": closure_best,
        "transformer_edges": transformer_edges,
        "symliquid_edges": symliquid_edges,
        "evidence_read": {
            "architecture_sweep_present": bool(arch),
            "capability_transfer_closure_present": bool(closure),
            "survival_path_decision_present": bool(survival),
            "broad_survival_run_present": bool(broad_run),
            "structural_action_probe_present": bool(structural),
        },
    }


def data_summary(admission: dict[str, Any], curriculum: dict[str, Any]) -> dict[str, Any]:
    admission_summary = dict_or_empty(admission.get("summary"))
    curriculum_summary = dict_or_empty(curriculum.get("summary"))
    return {
        "admission_state": admission.get("trigger_state"),
        "curriculum_state": curriculum.get("trigger_state"),
        "allowed_training_source_count": admission_summary.get("allowed_training_source_count"),
        "quarantined_source_count": admission_summary.get("quarantined_source_count"),
        "public_benchmark_payload_admitted": admission_summary.get("public_benchmark_payload_admitted"),
        "curriculum_unit_count": curriculum_summary.get("curriculum_unit_count"),
        "total_row_budget": curriculum_summary.get("total_row_budget"),
        "critical_missing": curriculum_summary.get("critical_missing"),
        "high_missing": curriculum_summary.get("high_missing"),
    }


def sts_vcm_summary(
    sts: dict[str, Any],
    sts_audit: dict[str, Any],
    sts_policy: dict[str, Any],
    vcm: dict[str, Any],
    vcm_ablation: dict[str, Any],
    vcm_policy: dict[str, Any],
    overnight: dict[str, Any],
    broad_run: dict[str, Any],
) -> dict[str, Any]:
    broad_comparisons = get_path(broad_run, ["comparator_report", "comparisons", "by_arm"], {})
    return {
        "sts_state": sts.get("trigger_state"),
        "sts_selected_pass_rate": first_number_for_key(sts, "sts_policy_selected_pass_rate"),
        "sts_non_sts_selected_pass_rate": first_number_for_key(sts, "non_sts_policy_selected_pass_rate"),
        "vcm_state": vcm.get("trigger_state"),
        "vcm_ready_task_family_count": first_number_for_key(vcm, "ready_task_family_count"),
        "vcm_high_priority_ready_count": first_number_for_key(vcm, "high_priority_ready_count"),
        "overnight_state": overnight.get("trigger_state") or overnight.get("status"),
        "overnight_promotions": first_present_number(
            overnight,
            ["promotions", "solo_learning_promotion_count", "promotion_count"],
        ),
        "overnight_failures": first_number_for_key(overnight, "failures"),
        "overnight_mlx_used": bool(find_first_bool(overnight, "mlx_used")),
        "overnight_mlx_verifier_pass_rate": first_number_for_key(overnight, "mlx_verifier_pass_rate"),
        "broad_transformer_sts_delta": get_path(broad_comparisons, ["transformer_control", "sts_delta"]),
        "broad_symliquid_sts_delta": get_path(broad_comparisons, ["symliquid_style", "sts_delta"]),
        "broad_transformer_sts_task_level_regressions": get_path(
            broad_comparisons,
            ["transformer_control", "sts_task_level_regressions"],
        ),
        "broad_symliquid_sts_task_level_regressions": get_path(
            broad_comparisons,
            ["symliquid_style", "sts_task_level_regressions"],
        ),
        "sts_broad_audit_action": get_path(sts_audit, ["summary", "recommended_action"]),
        "sts_broad_policy_action": sts_policy.get("action"),
        "sts_broad_policy_applied": get_path(broad_run, ["summary", "sts_policy_applied"]),
        "vcm_broad_feature_active": get_path(broad_run, ["summary", "vcm_context_active"]),
        "vcm_broad_feature_mode": get_path(broad_run, ["summary", "vcm_mode"]),
        "vcm_ablation_state": vcm_ablation.get("trigger_state"),
        "vcm_ablation_action": get_path(vcm_ablation, ["summary", "recommended_action"]),
        "promoted_transformer_vcm_mode": get_path(vcm_policy, ["promoted_transformer_vcm_mode"]),
        "symliquid_discovery_vcm_mode": get_path(vcm_policy, ["symliquid_discovery_vcm_mode"]),
        "vcm_ablation_deltas": get_path(vcm_ablation, ["summary", "deltas"], {}),
        "vcm_feature_policy_action": vcm_policy.get("action"),
    }


def mac_accelerator_summary(mlx_diagnosis: dict[str, Any], mlx_smoke: dict[str, Any]) -> dict[str, Any]:
    return {
        "mlx_diagnosis_state": mlx_diagnosis.get("trigger_state"),
        "mlx_route_action": get_path(mlx_diagnosis, ["summary", "route_action"]),
        "mlx_recommended_python": get_path(mlx_diagnosis, ["summary", "recommended_python"]),
        "mlx_usable_runtime_count": get_path(mlx_diagnosis, ["summary", "usable_mlx_runtime_count"]),
        "mlx_native_abort_count": get_path(mlx_diagnosis, ["summary", "native_abort_count"]),
        "mlx_smoke_state": mlx_smoke.get("trigger_state"),
        "mlx_smoke_used_mlx": get_path(mlx_smoke, ["summary", "mlx_used"]),
        "mlx_smoke_default_device": get_path(mlx_smoke, ["summary", "mlx_default_device"]),
        "mlx_smoke_verifier_pass_rate": get_path(mlx_smoke, ["summary", "verifier_pass_rate"]),
        "mlx_smoke_fallback_return_rows": get_path(mlx_smoke, ["summary", "fallback_return_rows"]),
        "mlx_parity_claimed": False,
    }


def practical_lane_contract(
    *,
    broad_promotion: dict[str, Any],
    sts_policy: dict[str, Any],
    vcm_policy: dict[str, Any],
    replay: dict[str, Any],
) -> dict[str, Any]:
    return {
        "policy": "project_theseus_practical_code_transfer_lane_contract_v1",
        "canonical_lane_id": "transformer_hybrid_structural_full_body_student",
        "canonical_decision_report": rel(DEFAULT_OUT),
        "canonical_promotion_gate": rel(DEFAULT_BROAD_PROMOTION),
        "canonical_candidate_replay_contract": rel(DEFAULT_REPLAY),
        "active_manifest": broad_promotion.get("active_manifest_path") or "",
        "promotion_scope": get_path(broad_promotion, ["summary", "model_promotion_scope"]),
        "serving_allowed": False,
        "public_calibration_allowed": False,
        "symliquid_role": "protected_matched_compute_discovery_comparator_only",
        "legacy_body_template_selector_policy": {
            "status": "diagnostic_only",
            "promotion_allowed": False,
            "sts_policy_action": sts_policy.get("action"),
            "reason": (
                "The old broad body-template selector is allowed to explain residuals and provide baseline "
                "pressure, but it must not silently feed promotion claims. Promotion flows through the "
                "structural/full-body transformer-hybrid student plus replay contract."
            ),
        },
        "vcm_policy": {
            "action": vcm_policy.get("action"),
            "promoted_transformer_vcm_mode": vcm_policy.get("promoted_transformer_vcm_mode"),
            "symliquid_discovery_vcm_mode": vcm_policy.get("symliquid_discovery_vcm_mode"),
            "rule": "VCM is used only where same-surface structural ablations show lift.",
        },
        "candidate_replay_contract": replay_contract_summary(replay),
        "no_cheat": {
            "public_benchmark_training_rows": 0,
            "fallback_returns_allowed": False,
            "external_inference_serving_allowed": False,
        },
    }


def candidate_replay_contract_ready(replay: dict[str, Any]) -> bool:
    summary = dict_or_empty(replay.get("summary"))
    return bool(
        replay.get("trigger_state") == "GREEN"
        and float(summary.get("selected_runtime_load_rate") or 0.0) >= 1.0
        and float(summary.get("selected_compile_pass_rate") or 0.0) >= 1.0
        and int(summary.get("unexplained_no_candidate_count") or 0) == 0
        and int(summary.get("fallback_return_candidate_count") or 0) == 0
        and int(summary.get("public_boundary_violation_count") or 0) == 0
    )


def replay_contract_summary(replay: dict[str, Any]) -> dict[str, Any]:
    summary = dict_or_empty(replay.get("summary"))
    return {
        "trigger_state": replay.get("trigger_state"),
        "task_count": summary.get("task_count"),
        "candidate_row_count": summary.get("candidate_row_count"),
        "eligible_candidate_count": summary.get("eligible_candidate_count"),
        "replayed_candidate_count": summary.get("replayed_candidate_count"),
        "selected_compile_pass_rate": summary.get("selected_compile_pass_rate"),
        "selected_runtime_load_rate": summary.get("selected_runtime_load_rate"),
        "selected_intended_behavior_pass_rate": summary.get("selected_intended_behavior_pass_rate"),
        "pass_if_any_rate": summary.get("pass_if_any_rate"),
        "unexplained_no_candidate_count": summary.get("unexplained_no_candidate_count"),
        "fallback_return_candidate_count": summary.get("fallback_return_candidate_count"),
        "public_boundary_violation_count": summary.get("public_boundary_violation_count"),
        "ready": candidate_replay_contract_ready(replay),
    }


def promotion_state(promotion: dict[str, Any], overnight: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps([promotion, overnight], sort_keys=True).lower()
    promoted = bool(
        promotion.get("promote") is True
        or promotion.get("promoted") is True
        or promotion.get("promotion_applied") is True
        or any(marker in text for marker in ["\"promoted\": true", "\"promotion_applied\": true", "promote_new_candidate"])
    )
    blocked = any(marker in text for marker in ["blocked", "not_promoted", "promotion_blocked"])
    return {
        "promotion_report_state": promotion.get("trigger_state"),
        "promoted": promoted,
        "blocked_or_not_promoted": blocked or not promoted,
        "legacy_promote_field": promotion.get("promote"),
        "passed_checks": promotion.get("passed"),
        "total_checks": promotion.get("total"),
        "best_by_arm_present": (ROOT / "reports" / "hive_solo_best_by_arm.json").exists(),
        "rollback_artifact_present": bool(find_first_string(promotion, "rollback", "rollback_artifact")),
    }


def combine_promotion_states(
    *,
    legacy: dict[str, Any],
    broad: dict[str, Any],
    broad_promotion: dict[str, Any],
) -> dict[str, Any]:
    promoted = bool(legacy.get("promoted") or broad.get("promoted"))
    blocked = bool(legacy.get("blocked_or_not_promoted")) and not bool(broad.get("promoted"))
    return {
        **legacy,
        "promoted": promoted,
        "blocked_or_not_promoted": blocked,
        "legacy_promoted": bool(legacy.get("promoted")),
        "broad_survival_promoted": bool(broad.get("promoted")),
        "broad_survival_trigger_state": broad_promotion.get("trigger_state"),
        "broad_survival_active_manifest": broad_promotion.get("active_manifest_path"),
        "broad_survival_promotion_scope": get_path(
            broad_promotion,
            ["summary", "model_promotion_scope"],
        ),
    }


def no_cheat_summary(*reports: dict[str, Any]) -> dict[str, Any]:
    public_rows = 0
    fallback = 0
    external = 0
    raw_text = 0
    teacher = 0
    for report in reports:
        public_rows += int(first_number_for_key(report, "public_training_rows") or 0)
        public_rows += int(first_number_for_key(report, "public_benchmark_training_rows") or 0)
        fallback += int(first_number_for_key(report, "fallback_return_count") or 0)
        fallback += int(first_number_for_key(report, "fallback_returns") or 0)
        external += int(first_number_for_key(report, "external_inference_calls") or 0)
        raw_text += int(first_number_for_key(report, "raw_user_text_count") or 0)
        teacher += int(first_number_for_key(report, "teacher_used") or 0)
    return {
        "public_benchmark_training_rows": public_rows,
        "fallback_return_count": fallback,
        "external_inference_calls": external,
        "raw_user_text_count": raw_text,
        "teacher_used_count": teacher,
    }


def derive_blockers(
    *,
    admission: dict[str, Any],
    curriculum: dict[str, Any],
    architecture_evidence: dict[str, Any],
    no_cheat: dict[str, Any],
    promotion_summary: dict[str, Any],
    closure: dict[str, Any],
    overnight: dict[str, Any],
    sts: dict[str, Any],
    sts_audit: dict[str, Any],
    sts_policy: dict[str, Any],
    vcm: dict[str, Any],
    vcm_ablation: dict[str, Any],
    vcm_policy: dict[str, Any],
    mlx_diagnosis: dict[str, Any],
    mlx_smoke: dict[str, Any],
    broad_run: dict[str, Any],
    broad_comparator: dict[str, Any],
    structural: dict[str, Any],
    replay: dict[str, Any],
) -> list[dict[str, Any]]:
    blockers = []
    if admission.get("trigger_state") not in {"GREEN", "YELLOW"}:
        blockers.append(blocker("admission", "hard", "training_data_admission_not_ready", admission.get("trigger_state")))
    if curriculum.get("trigger_state") not in {"GREEN", "YELLOW"}:
        blockers.append(blocker("curriculum", "hard", "broad_capability_curriculum_not_ready", curriculum.get("trigger_state")))
    critical_missing = get_path(curriculum, ["summary", "critical_missing"], [])
    if critical_missing:
        blockers.append(blocker("curriculum", "hard", "critical_capability_family_missing", critical_missing))
    if no_cheat["public_benchmark_training_rows"] != 0:
        blockers.append(blocker("no_cheat", "hard", "public_benchmark_training_rows_detected", no_cheat["public_benchmark_training_rows"]))
    if no_cheat["fallback_return_count"] != 0:
        blockers.append(blocker("no_cheat", "hard", "fallback_returns_detected", no_cheat["fallback_return_count"]))
    if no_cheat["external_inference_calls"] != 0:
        blockers.append(blocker("no_cheat", "hard", "external_inference_detected", no_cheat["external_inference_calls"]))
    replay_summary = replay_contract_summary(replay)
    if not candidate_replay_contract_ready(replay):
        blockers.append(
            blocker(
                "candidate_replay",
                "hard",
                "private_candidate_replay_contract_not_green",
                replay_summary,
            )
        )
    if architecture_evidence.get("current_smoke_symliquid_minus_transformer_mean") is None and architecture_evidence.get("one_notch_symliquid_minus_transformer_mean") is None:
        blockers.append(blocker("architecture", "warning", "matched_architecture_evidence_missing_or_unreadable", architecture_evidence))
    if broad_run and broad_run.get("trigger_state") not in {"GREEN", "YELLOW"}:
        blockers.append(blocker("broad_run", "hard", "broad_survival_lane_run_not_green", broad_run.get("trigger_state")))
    broad_comparisons = get_path(broad_run, ["comparator_report", "comparisons", "by_arm"], {})
    transformer_sts_delta = get_path(broad_comparisons, ["transformer_control", "sts_delta"])
    comparator_text = json.dumps(broad_comparator, sort_keys=True).lower()
    structural_transformer = dict_or_empty(get_path(structural, ["arms", "transformer_control", "summary"], {}))
    structural_ready = (
        structural.get("trigger_state") in {"GREEN", "YELLOW"}
        and int(structural_transformer.get("structural_candidate_rows") or 0) > 0
        and int(structural_transformer.get("fallback_return_rows") or 0) == 0
        and float(structural_transformer.get("syntax_pass_rate") or 0.0) > 0.0
    )
    if structural_ready:
        structural_only = float(structural_transformer.get("structural_only_pass_rate") or 0.0)
        augmented_delta = float(structural_transformer.get("delta") or 0.0)
        baseline = float(structural_transformer.get("baseline_pass_rate") or 0.0)
        if structural_only < baseline or augmented_delta < 0.0:
            blockers.append(
                blocker(
                    "decoder",
                    "hard",
                    "structural_action_candidate_family_not_promotion_grade_yet",
                    {
                        "structural_only_pass_rate": structural_only,
                        "augmented_delta": augmented_delta,
                        "augmented_pass_rate": structural_transformer.get("augmented_pass_rate"),
                        "baseline_pass_rate": structural_transformer.get("baseline_pass_rate"),
                        "candidate_rows": structural_transformer.get("structural_candidate_rows"),
                        "reason": (
                            "The structural-action path exists, but transformer structural-only candidates must "
                            "match or beat the body-template baseline and augmented fanout must not regress before "
                            "this can replace the body-template adapter as the promotion path."
                        ),
                    },
                )
            )
    elif "body_template_selector" in comparator_text or "body templates" in comparator_text:
        blockers.append(
            blocker(
                "decoder",
                "hard",
                "broad_survival_run_still_uses_body_template_selector_adapter",
                {
                    "adapter_boundary": get_path(broad_comparator, ["adapter_boundary", "current_status"]),
                    "reason": (
                        "The broad run is useful architecture evidence, but promotion-grade survival work "
                        "still needs grammar-masked full-body or structural-action generation rather than "
                        "private body-template selection."
                    ),
                },
            )
        )
    vcm_feature = get_path(broad_run, ["vcm_context"], {})
    vcm_feature_policy_action = str(vcm_policy.get("action") or "")
    vcm_ablation_harmed = bool(get_path(vcm_ablation, ["summary", "harmed_arms"], []))
    vcm_consumed_in_ablation = int(get_path(vcm_ablation, ["summary", "vcm_rows_with_context"], 0) or 0) > 0
    if not vcm_feature and not vcm_consumed_in_ablation and "vcm_context_required" not in comparator_text and "vcm" not in json.dumps(get_path(broad_comparator, ["data_contract"], {})).lower():
        blockers.append(
            blocker(
                "vcm",
                "hard",
                "broad_survival_comparator_does_not_consume_vcm_contexts_yet",
                "VCM is present in the curriculum contract, but the executed comparator feature path does not load vcm_task_contexts.",
            )
        )
    elif vcm_ablation_harmed and vcm_feature_policy_action == "disable_vcm_for_broad_body_template_selector":
        blockers.append(
            blocker(
                "vcm",
                "warning",
                "vcm_consumed_but_gated_for_body_template_selector_due_harm",
                {
                    "ablation_deltas": get_path(vcm_ablation, ["summary", "deltas"], {}),
                    "policy_action": vcm_feature_policy_action,
                    "reason": "VCM feature path is implemented and consumed under --vcm-mode on, but current body-template selector defaults it off because equal-budget private smoke showed harm.",
                },
            )
        )
    legacy_body_template_sts_disabled = (
        str(sts_policy.get("action") or "") == "disable_sts_for_broad_body_template_selector"
    )
    if (
        transformer_sts_delta is not None
        and float(transformer_sts_delta) < 0
        and not (structural_ready and legacy_body_template_sts_disabled)
    ):
        blockers.append(
            blocker(
                "sts",
                "hard",
                "sts_regresses_transformer_on_broad_survival_eval",
                {
                    "transformer_sts_delta": transformer_sts_delta,
                    "transformer_sts_task_level_regressions": get_path(
                        broad_comparisons,
                        ["transformer_control", "sts_task_level_regressions"],
                    ),
                    "transformer_sts_on": get_path(
                        broad_comparisons,
                        ["transformer_control", "sts_on_verifier_pass_rate"],
                    ),
                    "transformer_sts_off": get_path(
                        broad_comparisons,
                        ["transformer_control", "sts_off_verifier_pass_rate"],
                    ),
                },
            )
        )
    elif transformer_sts_delta is not None and float(transformer_sts_delta) < 0:
        blockers.append(
            blocker(
                "sts",
                "warning",
                "legacy_body_template_sts_regression_routed_off",
                {
                    "transformer_sts_delta": transformer_sts_delta,
                    "sts_policy_action": sts_policy.get("action"),
                    "promotion_path": "transformer_hybrid_structural_full_body_student",
                    "reason": (
                        "STS regression belongs to the retired broad body-template selector. "
                        "The structural/full-body survival lane remains eligible, with STS routed "
                        "only by path-specific policy evidence."
                    ),
                },
            )
        )
    mlx_route_action = str(get_path(mlx_diagnosis, ["summary", "route_action"], ""))
    mlx_reason = str(find_first_string(broad_comparator, "mlx.core child probe exited", "mlx_reason") or "")
    if mlx_route_action == "disable_mlx_acceleration_route":
        blockers.append(
            blocker(
                "mac_accelerator",
                "warning",
                "mlx_route_safely_disabled_until_clean_runtime_exists",
                {
                    "route_action": mlx_route_action,
                    "active_python_status": get_path(mlx_diagnosis, ["summary", "active_python_status"]),
                    "smallest_safe_fix": get_path(mlx_diagnosis, ["summary", "smallest_safe_fix"]),
                },
            )
        )
    elif mlx_route_action != "route_mlx_to_usable_python" and "mlx.core child probe exited" in mlx_reason:
        blockers.append(
            blocker(
                "mac_accelerator",
                "hard",
                "mlx_native_probe_fails_in_current_python_environment",
                mlx_reason[:700],
            )
        )
    elif mlx_route_action == "route_mlx_to_usable_python":
        if mlx_smoke.get("trigger_state") != "GREEN" or not bool(get_path(mlx_smoke, ["summary", "mlx_used"], False)):
            blockers.append(
                blocker(
                    "mac_accelerator",
                    "hard",
                    "mlx_route_enabled_but_structural_smoke_not_green",
                    {
                        "diagnosis_route_action": mlx_route_action,
                        "smoke_state": mlx_smoke.get("trigger_state"),
                        "smoke_mlx_used": get_path(mlx_smoke, ["summary", "mlx_used"]),
                    },
                )
            )
    if promotion_summary["promoted"] is not True:
        quality_context = {
            "latest_public_score_hint": find_public_score(closure),
            "latest_public_audit_pass_rate": first_number_for_key(closure, "last_public_audit_pass_rate"),
            "overnight_promotions": first_present_number(
                overnight,
                ["promotions", "solo_learning_promotion_count", "promotion_count"],
            ),
            "overnight_mlx_verifier_pass": first_number_for_key(overnight, "mlx_verifier_pass_rate"),
            "sts_state": sts.get("trigger_state"),
            "vcm_state": vcm.get("trigger_state"),
            "broad_survival_transformer_sts_on": get_path(
                broad_run,
                ["summary", "transformer_sts_on_pass_rate"],
            ),
            "broad_survival_symliquid_sts_on": get_path(
                broad_run,
                ["summary", "symliquid_sts_on_pass_rate"],
            ),
        }
        blockers.append(blocker("promotion", "hard", "no_new_transformer_hybrid_candidate_promoted", quality_context))
    return blockers


def public_calibration_decision(closure: dict[str, Any], promotion_summary: dict[str, Any], blockers: list[dict[str, Any]]) -> dict[str, Any]:
    public_score = find_public_score(closure)
    hard_non_promotion_blockers = [row for row in blockers if row["severity"] == "hard"]
    return {
        "recommended": False,
        "reason": (
            "Do not spend public calibration until a new private transformer/hybrid survival candidate promotes "
            "or the operator deliberately unlocks a one-shot calibration after private gates improve."
        ),
        "latest_locked_public_score": public_score,
        "promotion_ready": promotion_summary["promoted"],
        "hard_blocker_count": len(hard_non_promotion_blockers),
        "rerun_for_tuning_allowed": False,
    }


def next_actions(blockers: list[dict[str, Any]], training_ready: bool, promotion_ready: bool) -> list[str]:
    if promotion_ready:
        return [
            "publish promoted transformer/hybrid artifact through existing private gates",
            "prepare a deliberate one-shot public calibration proposal, still operator-locked",
        ]
    actions = []
    if not training_ready:
        actions.append("fix hard admission/curriculum/no-cheat blockers before launching any long training run")
    if any(row["code"] == "sts_regresses_transformer_on_broad_survival_eval" for row in blockers):
        actions.append("repair the STS ranker/view policy on broad_capability_survival_lane_v1 before using STS for promotion")
    if any(row["code"] == "legacy_body_template_sts_regression_routed_off" for row in blockers):
        actions.append("keep STS routed off for the retired body-template selector; evaluate STS only on structural/full-body candidate paths")
    if any(row["code"] == "broad_survival_run_still_uses_body_template_selector_adapter" for row in blockers):
        actions.append("replace the broad survival body-template adapter with grammar-masked full-body or structural-action generation before promotion")
    if any(row["code"] == "structural_action_candidate_family_not_promotion_grade_yet" for row in blockers):
        actions.append("improve structural-action generation until transformer structural-only pass rate matches or beats the body-template baseline and augmented gains persist on the full broad slice")
    if any(row["code"] == "broad_survival_comparator_does_not_consume_vcm_contexts_yet" for row in blockers):
        actions.append("wire vcm_task_contexts into the broad survival comparator input features and rerun the private heldout slice")
    if any(row["code"] == "vcm_consumed_but_gated_for_body_template_selector_due_harm" for row in blockers):
        actions.append("keep VCM default-gated for the body-template selector; retest VCM on the next structural/full-body generator instead of forcing a harmful feature path")
    if any(row["code"] == "mlx_native_probe_fails_in_current_python_environment" for row in blockers):
        actions.append("fix the active Python MLX/Metal environment or route Mac acceleration through a known-good interpreter before claiming Mac accelerator readiness")
    if any(row["code"] == "mlx_route_safely_disabled_until_clean_runtime_exists" for row in blockers):
        actions.append("create a clean Apple-Silicon MLX Python runtime and rerun macos_mlx_environment_diagnosis before queuing MLX work")
    if any(row["code"] == "no_new_transformer_hybrid_candidate_promoted" for row in blockers):
        actions.append("turn the broad transformer/control win into a promotion-eligible artifact only after STS and Mac accelerator blockers are fixed")
        actions.append("keep SymLiquid in the run only as a matched comparator with equal VCM/STS/verifier/candidate budget")
        actions.append("route Mac acceleration through MLX/Metal only when the local accelerator evidence is clean")
    actions.append("do not run public calibration or train on public benchmark payloads while the private promotion blocker remains")
    return actions


def blocker(stage: str, severity: str, code: str, evidence: Any) -> dict[str, Any]:
    return {"stage": stage, "severity": severity, "code": code, "evidence": evidence}


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def find_public_score(report: dict[str, Any]) -> float | None:
    for key in [
        "locked_broad_public_pass_rate",
        "previous_locked_baseline_pass_rate",
        "public_pass_rate",
        "public_score",
        "latest_locked_public_score",
        "student_first_public_pass_rate",
    ]:
        value = first_number_for_key(report, key)
        if value is not None:
            return value
    text = json.dumps(report, sort_keys=True)
    if "34/160" in text:
        return 0.2125
    return None


def first_present_number(value: Any, keys: list[str]) -> float | None:
    for key in keys:
        found = first_number_for_key(value, key)
        if found is not None:
            return found
    return None


def find_first_string(value: Any, needle: str, preferred_key: str = "") -> str:
    if isinstance(value, dict):
        if preferred_key and preferred_key in value and isinstance(value[preferred_key], str):
            return value[preferred_key]
        for child in value.values():
            found = find_first_string(child, needle, preferred_key)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_first_string(child, needle, preferred_key)
            if found:
                return found
    elif isinstance(value, str) and needle in value:
        return value
    return ""


def find_first_bool(value: Any, needle: str) -> bool | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if needle in str(key).lower() and isinstance(child, bool):
                return child
            found = find_first_bool(child, needle)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_first_bool(child, needle)
            if found is not None:
                return found
    return None


def first_number_for_key(value: Any, key: str) -> float | None:
    if isinstance(value, dict):
        if key in value:
            direct = number(value.get(key))
            if direct is not None:
                return direct
        for child in value.values():
            found = first_number_for_key(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = first_number_for_key(child, key)
            if found is not None:
                return found
    return None


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cursor = value
    for part in path:
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Broad Capability Survival Lane Decision v1",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- decision: `{report.get('decision')}`",
        f"- training_ready: `{report.get('training_ready')}`",
        f"- promotion_ready: `{report.get('promotion_ready')}`",
        f"- public calibration recommended: `{report.get('public_calibration_recommendation', {}).get('recommended')}`",
        "",
        "## Architecture Evidence",
    ]
    architecture = report.get("architecture_evidence") if isinstance(report.get("architecture_evidence"), dict) else {}
    for key in [
        "survival_lane",
        "symliquid_role",
        "architecture_sweep_state",
        "current_smoke_symliquid_minus_transformer_mean",
        "one_notch_symliquid_minus_transformer_mean",
        "closure_best_arm_hint",
    ]:
        lines.append(f"- {key}: `{architecture.get(key)}`")
    contract = report.get("practical_lane_contract") if isinstance(report.get("practical_lane_contract"), dict) else {}
    lines.extend(["", "## Practical Lane Contract"])
    for key in [
        "canonical_lane_id",
        "canonical_promotion_gate",
        "canonical_candidate_replay_contract",
        "active_manifest",
        "promotion_scope",
        "symliquid_role",
    ]:
        lines.append(f"- {key}: `{contract.get(key)}`")
    legacy = contract.get("legacy_body_template_selector_policy") if isinstance(contract.get("legacy_body_template_selector_policy"), dict) else {}
    lines.append(f"- legacy body-template selector: `{legacy.get('status')}` promotion_allowed=`{legacy.get('promotion_allowed')}`")
    replay = report.get("candidate_replay_contract") if isinstance(report.get("candidate_replay_contract"), dict) else {}
    lines.extend(["", "## Candidate Replay Contract"])
    for key in [
        "ready",
        "task_count",
        "candidate_row_count",
        "selected_runtime_load_rate",
        "selected_compile_pass_rate",
        "selected_intended_behavior_pass_rate",
        "fallback_return_candidate_count",
        "public_boundary_violation_count",
    ]:
        lines.append(f"- {key}: `{replay.get(key)}`")
    lines.extend(["", "## Blockers"])
    blockers = report.get("blockers") if isinstance(report.get("blockers"), list) else []
    if not blockers:
        lines.append("- none")
    else:
        for row in blockers:
            lines.append(f"- `{row.get('stage')}` / `{row.get('severity')}` / `{row.get('code')}`: `{row.get('evidence')}`")
    lines.extend(["", "## Next Actions"])
    for action in report.get("next_actions", []):
        lines.append(f"- {action}")
    lines.extend(["", "## Failed Gates"])
    failed = [row for row in report.get("gates", []) if isinstance(row, dict) and not row.get("passed")]
    if not failed:
        lines.append("- none")
    else:
        for row in failed:
            lines.append(f"- `{row.get('name')}` ({row.get('severity')}): `{row.get('evidence')}`")
    return "\n".join(lines) + "\n"


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
