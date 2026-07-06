#!/usr/bin/env python3
"""Prepare the remaining-gate closure packet without approving any gate."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "theseus_gate_closure_packet.json"
DEFAULT_MD = REPORTS / "theseus_gate_closure_packet.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    parser.add_argument("--write-templates", action="store_true")
    args = parser.parse_args()

    public = run_json([sys.executable, "scripts/operator_bounded_public_calibration.py"])
    metal = run_json([sys.executable, "scripts/macos_metal_production_route_readiness.py"])
    parity = run_json([sys.executable, "scripts/accelerator_parity_claim_readiness.py"])
    dogfood = run_json([sys.executable, "scripts/dogfood_trace_consent.py"])
    dogfood_bridge = read_json(REPORTS / "dogfood_trace_training_bridge.json", {})
    teacher_gate = run_json([sys.executable, "scripts/teacher_distillation_gate.py"])
    teacher_manifest = read_json(REPORTS / "teacher_distillation_manifest_audit.json", {})
    training_admission = read_json(REPORTS / "training_data_admission_v1.json", {})
    capability = read_json(REPORTS / "capability_transfer_closure_v1.json", {})
    public_transfer_readiness = read_json(REPORTS / "public_transfer_readiness_refresh_v1.json", {})
    maturity = read_json(REPORTS / "maturity_integrity_audit.json", {})
    private_v3_gate = read_json(REPORTS / "private_residual_repair_v3_gate.json", {})
    private_ratchet = read_json(REPORTS / "private_residual_self_improvement_ratchet_v1.json", {})
    vcm = read_json(REPORTS / "vcm_release_conformance_audit.json", {})
    mlx = read_json(REPORTS / "macos_mlx_structural_action_smoke.json", {})
    resource = read_json(REPORTS / "resource_governor.json", {})
    autonomy = read_json(REPORTS / "autonomy_launch_readiness.json", {})
    network = refresh_network_doctor()
    cuda_env = cuda_environment()
    proposed_slug = str(get_path(public, ["summary", "proposed_slug"], "industry_code_transfer_seed14_5x64_v1"))

    templates = {
        "public_calibration": {
            "path": f"reports/public_calibration_operator_approval_{proposed_slug}.template.json",
            "payload": public.get("approval_template") or {},
        },
        "macos_metal_production_route": {
            "path": "configs/macos_metal_production_route_approval.template.json",
            "payload": metal.get("approval_template") or {},
        },
        "accelerator_parity_claim": {
            "path": "configs/accelerator_parity_claim_approval.template.json",
            "payload": parity.get("approval_template") or {},
        },
    }
    written_templates: list[str] = []
    if args.write_templates:
        for item in templates.values():
            payload = item["payload"] if isinstance(item["payload"], dict) else {}
            if payload:
                payload = {**payload, "approved": False}
                write_json(resolve(item["path"]), payload)
                written_templates.append(item["path"])

    report = {
        "policy": "project_theseus_remaining_gate_closure_packet_v0",
        "created_utc": now(),
        "trigger_state": "YELLOW",
        "summary": {
            "dogfood_capture_enabled": get_path(dogfood, ["summary", "capture_enabled"], False),
            "dogfood_training_enabled": get_path(dogfood, ["summary", "training_enabled"], False),
            "dogfood_training_rows_written": get_path(dogfood_bridge, ["training_rows_written"], 0),
            "dogfood_trainable_event_count": get_path(dogfood_bridge, ["summary", "trainable_event_count"], 0),
            "teacher_distillation_allowed": get_path(teacher_gate, ["summary", "distillation_allowed"], False),
            "teacher_share_metric_ready": get_path(teacher_gate, ["summary", "teacher_share_metric_ready"], False),
            "teacher_share_ledger_present": get_path(teacher_gate, ["summary", "teacher_share_ledger_present"], False),
            "teacher_share_ledger_row_count": get_path(teacher_gate, ["summary", "teacher_share_ledger_row_count"], 0),
            "teacher_accepted_row_share": get_path(teacher_gate, ["summary", "teacher_accepted_row_share"], 0.0),
            "teacher_distillation_fail_closed": get_path(teacher_gate, ["summary", "teacher_distillation_fail_closed"], False),
            "teacher_manifest_ready_for_distillation": get_path(teacher_manifest, ["summary", "manifest_ready_for_distillation"], False),
            "teacher_manifest_admission_safety_checks_clean": get_path(
                teacher_manifest, ["summary", "admission_safety_checks_clean"], False
            ),
            "teacher_manifest_verifier_pass_rate_applicable": get_path(
                teacher_manifest, ["summary", "verifier_pass_rate_applicable"], False
            ),
            "teacher_rows_admitted": get_path(teacher_manifest, ["summary", "teacher_rows_admitted"], 0),
            "teacher_proposal_rows_retained_not_training": get_path(teacher_manifest, ["summary", "proposal_rows_retained_not_training"], 0),
            "teacher_rejected_ledger_row_count": get_path(teacher_manifest, ["summary", "rejected_ledger_row_count"], 0),
            "teacher_manifest_ledger_row_count": get_path(teacher_manifest, ["summary", "ledger_row_count"], 0),
            "teacher_training_admission_manifest_rows": get_path(training_admission, ["summary", "teacher_distillation_manifest_rows"], 0),
            "teacher_training_admission_ledger_rows": get_path(training_admission, ["summary", "teacher_distillation_ledger_rows"], 0),
            "teacher_training_admission_proposal_ledger_rows": get_path(
                training_admission, ["summary", "teacher_distillation_proposal_ledger_rows"], 0
            ),
            "teacher_training_admission_safety_clean": get_path(
                training_admission, ["summary", "teacher_distillation_admission_safety_clean"], False
            ),
            "teacher_training_admission_public_overlap_hits": get_path(training_admission, ["summary", "teacher_distillation_public_overlap_hits"], 0),
            "private_semantic_ready": get_path(capability, ["summary", "private_semantic_ready"], False),
            "private_selected_pass_rate": get_path(capability, ["summary", "private_selected_pass_rate"], 0.0),
            "private_pass_if_any_rate": get_path(capability, ["summary", "private_pass_if_any_rate"], 0.0),
            "public_candidate_coverage_ready": get_path(capability, ["summary", "public_candidate_coverage_ready"], False),
            "public_promotion_ready": get_path(capability, ["summary", "public_promotion_ready"], False),
            "public_transfer_readiness_state": public_transfer_readiness.get("trigger_state"),
            "public_transfer_readiness_hard_failed_gate_count": get_path(
                public_transfer_readiness, ["summary", "hard_failed_gate_count"], 0
            ),
            "public_transfer_readiness_evidence_failed_gate_count": get_path(
                public_transfer_readiness, ["summary", "evidence_failed_gate_count"], 0
            ),
            "public_transfer_readiness_transfer_failed_gate_count": get_path(
                public_transfer_readiness, ["summary", "transfer_failed_gate_count"], 0
            ),
            "public_transfer_contract_task_count": get_path(
                public_transfer_readiness, ["summary", "contract_task_count"], 0
            ),
            "public_transfer_full_body_selected_pass_rate": get_path(
                public_transfer_readiness, ["summary", "full_body_selected_pass_rate"], 0.0
            ),
            "public_transfer_full_body_pass_if_any_rate": get_path(
                public_transfer_readiness, ["summary", "full_body_pass_if_any_rate"], 0.0
            ),
            "public_transfer_full_body_benchmark_promotion_eligible_candidate_count": get_path(
                public_transfer_readiness, ["summary", "full_body_benchmark_promotion_eligible_candidate_count"], 0
            ),
            "public_transfer_full_body_default_semantic_dead": get_path(
                public_transfer_readiness, ["summary", "full_body_post_v4_default_semantic_dead"], True
            ),
            "private_residual_target_rows_written": get_path(
                capability,
                [
                    "summary",
                    "public_transfer_wall",
                    "current_full_body_residual_mining",
                    "private_residual_target_rows_written",
                ],
                0,
            ),
            "private_residual_target_manifest": get_path(
                capability,
                [
                    "summary",
                    "public_transfer_wall",
                    "current_full_body_residual_mining",
                    "private_residual_target_manifest",
                ],
                "",
            ),
            "private_residual_target_consumer_state": get_path(
                capability,
                ["summary", "public_transfer_wall", "private_residual_target_consumer", "trigger_state"],
                "",
            ),
            "private_residual_repair_queue_rows": get_path(
                capability,
                ["summary", "public_transfer_wall", "private_residual_target_consumer", "repair_queue_rows"],
                0,
            ),
            "private_residual_repair_queue_path": get_path(
                capability,
                ["summary", "public_transfer_wall", "private_residual_target_consumer", "queue_path"],
                "",
            ),
            "private_residual_repair_unresolved_target_count": get_path(
                capability,
                ["summary", "public_transfer_wall", "private_residual_target_consumer", "unresolved_target_count"],
                0,
            ),
            "private_residual_repair_unresolved_target_state_counts": get_path(
                capability,
                ["summary", "public_transfer_wall", "private_residual_target_consumer", "unresolved_target_state_counts"],
                {},
            ),
            "private_residual_repair_unresolved_target_category_counts": get_path(
                capability,
                ["summary", "public_transfer_wall", "private_residual_target_consumer", "unresolved_target_category_counts"],
                {},
            ),
            "private_residual_consumer_training_rows_written": get_path(
                capability,
                ["summary", "public_transfer_wall", "private_residual_target_consumer", "training_rows_written"],
                0,
            ),
            "private_residual_v3_gate_state": private_v3_gate.get("trigger_state"),
            "private_residual_v3_learned_candidate_pass_rate": get_path(
                private_v3_gate, ["summary", "private_learned_candidate_pass_rate"], 0.0
            ),
            "private_residual_v3_diagnostic_adapter_pass_rate": get_path(
                private_v3_gate, ["summary", "private_diagnostic_adapter_pass_rate"], 0.0
            ),
            "private_residual_v3_adapter_off_scoring": get_path(
                private_v3_gate, ["summary", "private_heldout_adapter_off_scoring"], False
            ),
            "private_residual_v3_pending_gate_count": get_path(private_v3_gate, ["summary", "pending_gate_count"], 0),
            "private_residual_v3_failed_gate_count": get_path(private_v3_gate, ["summary", "failed_gate_count"], 0),
            "private_residual_ratchet_completed_queue_item_count": get_path(
                private_ratchet, ["summary", "completed_queue_item_count"], 0
            ),
            "private_residual_ratchet_pending_queue_item_count": get_path(
                private_ratchet, ["summary", "pending_queue_item_count"], 0
            ),
            "locked_broad_public_pass_rate": get_path(capability, ["summary", "public_transfer_wall", "locked_broad_public_pass_rate"], 0.0),
            "locked_broad_public_pass_count": get_path(capability, ["summary", "public_transfer_wall", "locked_broad_public_pass_count"], 0),
            "locked_broad_public_task_count": get_path(capability, ["summary", "public_transfer_wall", "locked_broad_public_task_count"], 0),
            "maturity_blocker_count": get_path(maturity, ["summary", "maturity_blocker_count"], 0),
            "maturity_blockers": maturity.get("maturity_blockers") if isinstance(maturity.get("maturity_blockers"), list) else [],
            "vcm_release_state": vcm.get("trigger_state"),
            "vcm_runtime_profile_claimed": get_path(vcm, ["summary", "runtime_profile_claimed"], False),
            "vcm_native_runtime_claimable": get_path(vcm, ["summary", "native_runtime_claimable"], False),
            "vcm_native_runtime_claim_scope": get_path(vcm, ["summary", "native_runtime_claim_scope"], ""),
            "vcm_native_runtime_claim_backend": get_path(vcm, ["summary", "native_runtime_claim_backend"], ""),
            "vcm_native_runtime_recommended_backend": get_path(vcm, ["summary", "native_runtime_recommended_execution_backend"], ""),
            "vcm_native_runtime_backend_matches_recommended": get_path(
                vcm, ["summary", "native_runtime_claim_backend_matches_recommended_execution_backend"], False
            ),
            "vcm_scheduler_native_kv_route_allowed_for_recommended_backend": get_path(
                vcm, ["summary", "scheduler_native_kv_route_allowed_for_recommended_backend"], False
            ),
            "vcm_mlx_tensor_descriptor_lifecycle_test_passed": get_path(
                vcm, ["summary", "mlx_tensor_descriptor_lifecycle_test_passed"], False
            ),
            "vcm_recommended_backend_runtime_descriptor_lifecycle_claimable": get_path(
                vcm, ["summary", "recommended_backend_runtime_descriptor_lifecycle_claimable"], False
            ),
            "vcm_scheduler_descriptor_route_allowed_for_recommended_backend": get_path(
                vcm, ["summary", "scheduler_vcm_descriptor_route_allowed_for_recommended_backend"], False
            ),
            "vcm_scheduler_native_kv_route_fail_closed": get_path(vcm, ["summary", "scheduler_native_kv_route_fail_closed"], True),
            "vcm_runtime_cache_lifecycle_state": get_path(vcm, ["summary", "runtime_cache_lifecycle_state"], ""),
            "vcm_accelerator_kv_parity_claimed": get_path(vcm, ["summary", "accelerator_kv_parity_claimed"], False),
            "mlx_smoke_state": mlx.get("trigger_state"),
            "mlx_available": get_path(mlx, ["summary", "mlx_available"], False),
            "mlx_used": get_path(mlx, ["summary", "mlx_used"], False),
            "mlx_verifier_pass_rate": get_path(mlx, ["summary", "verifier_pass_rate"], 0.0),
            "mlx_fallback_return_rows": get_path(mlx, ["summary", "fallback_return_rows"], 0),
            "resource_governor_state": resource.get("trigger_state"),
            "resource_execution_owner": get_path(resource, ["summary", "execution_owner"], ""),
            "resource_can_run_requested_profile": get_path(resource, ["summary", "can_run_requested_profile"], False),
            "autonomy_launch_state": autonomy.get("trigger_state"),
            "ready_for_autonomous_training": get_path(autonomy, ["summary", "ready_for_autonomous_training"], False),
            "public_packet_ready": get_path(public, ["summary", "packet_ready"], False),
            "public_approval_valid": get_path(public, ["summary", "approval_valid"], False),
            "metal_route_guarded_evidence_ok_count": get_path(metal, ["summary", "guarded_evidence_ok_count"], 0),
            "metal_route_allowed": get_path(metal, ["summary", "production_route_allowed"], False),
            "metal_route_approval_valid": get_path(metal, ["summary", "operator_approval_valid"], False),
            "parity_no_cheat_ok": get_path(parity, ["summary", "no_cheat_ok"], False),
            "parity_claim_allowed": get_path(parity, ["summary", "parity_claim_allowed"], False),
            "parity_ready_surface_count": get_path(parity, ["summary", "parity_ready_surface_count"], 0),
            "templates_written": written_templates,
            "external_inference_calls": 0,
            "public_training_rows": 0,
            "fallback_returns": 0,
        },
        "templates": templates,
        "cuda_environment": cuda_env,
        "network_doctor": network,
        "cuda_reference_commands": cuda_reference_commands(),
        "gate_reports": {
            "dogfood_consent": "reports/dogfood_trace_consent.json",
            "teacher_distillation_gate": "reports/teacher_distillation_gate.json",
            "teacher_distillation_manifest": "reports/teacher_distillation_manifest_audit.json",
            "training_data_admission": "reports/training_data_admission_v1.json",
            "capability_transfer_closure": "reports/capability_transfer_closure_v1.json",
            "public_transfer_readiness": "reports/public_transfer_readiness_refresh_v1.json",
            "private_residual_v3_gate": "reports/private_residual_repair_v3_gate.json",
            "private_residual_ratchet": "reports/private_residual_self_improvement_ratchet_v1.json",
            "maturity_integrity": "reports/maturity_integrity_audit.json",
            "vcm_release_conformance": "reports/vcm_release_conformance_audit.json",
            "macos_mlx_structural_action_smoke": "reports/macos_mlx_structural_action_smoke.json",
            "resource_governor": "reports/resource_governor.json",
            "autonomy_launch_readiness": "reports/autonomy_launch_readiness.json",
            "public_calibration": "reports/operator_bounded_public_calibration_dry_run.json",
            "macos_metal_route": "reports/macos_metal_production_route_readiness.json",
            "accelerator_parity_claim": "reports/accelerator_parity_claim_readiness.json",
        },
        "supporting_audits": supporting_audits(),
        "requirement_matrix": requirement_matrix(
            public,
            metal,
            parity,
            dogfood,
            dogfood_bridge,
            teacher_gate,
            teacher_manifest,
            training_admission,
            capability,
            public_transfer_readiness,
            maturity,
            private_v3_gate,
            private_ratchet,
            vcm,
            mlx,
            resource,
            autonomy,
            cuda_env,
            network,
        ),
        "remaining_blockers": remaining_blockers(
            public,
            metal,
            parity,
            dogfood,
            dogfood_bridge,
            teacher_gate,
            teacher_manifest,
            training_admission,
            capability,
            public_transfer_readiness,
            maturity,
            private_v3_gate,
            private_ratchet,
            vcm,
            mlx,
            resource,
            autonomy,
            cuda_env,
            network,
        ),
        "rules": {
            "templates_are_not_approval": True,
            "operator_must_set_approved_true_deliberately": True,
            "public_benchmark_training_forbidden": True,
            "raw_text_capture_default_forbidden": True,
            "fallback_returns_forbidden": True,
            "external_inference_serving_forbidden": True,
        },
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cuda_reference_commands() -> list[dict[str, str]]:
    return [
        {
            "surface": "standalone_readout_cli",
            "required_report": "reports/symliquid_standalone_cuda_train_report.json",
            "command": "cargo run --release -p symliquid-cli --features cuda -- train-standalone-cuda --cases-per-task 4 --epochs 2 --samples-per-launch 4 --hv-dim 64 --lr 0.03 --out reports/symliquid_standalone_cuda_train_report.json",
        },
        {
            "surface": "rollout_cli",
            "required_report": "reports/symliquid_rollout_cuda_train_report.json",
            "command": "cargo run --release -p symliquid-cli --features cuda -- train-rollout-cuda --cases-per-task 4 --epochs 2 --state-epochs 0 --samples-per-launch 4 --rollout-batch 4 --obs-dim 8 --hidden-dim 8 --reservoir-dim 8 --hv-dim 8 --seq-len 8 --lr 0.03 --out reports/symliquid_rollout_cuda_train_report.json",
        },
        {
            "surface": "rollout_sweep_cli",
            "required_report": "reports/symliquid_rollout_cuda_sweep.json",
            "command": "cargo run --release -p symliquid-cli --features cuda -- train-rollout-cuda-sweep --out reports/symliquid_rollout_cuda_sweep.json",
        },
        {
            "surface": "token_superposition_cli",
            "required_report": "reports/token_superposition_cuda_training.json",
            "command": "cargo run --release -p symliquid-cli --features cuda -- train-token-superposition-cuda --out reports/token_superposition_cuda_training.json",
        },
    ]


def remaining_blockers(
    public: dict[str, Any],
    metal: dict[str, Any],
    parity: dict[str, Any],
    dogfood: dict[str, Any],
    dogfood_bridge: dict[str, Any],
    teacher_gate: dict[str, Any],
    teacher_manifest: dict[str, Any],
    training_admission: dict[str, Any],
    capability: dict[str, Any],
    public_transfer_readiness: dict[str, Any],
    maturity: dict[str, Any],
    private_v3_gate: dict[str, Any],
    private_ratchet: dict[str, Any],
    vcm: dict[str, Any],
    mlx: dict[str, Any],
    resource: dict[str, Any],
    autonomy: dict[str, Any],
    cuda_env: dict[str, Any],
    network: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if not get_path(dogfood, ["summary", "capture_enabled"], False):
        blockers.append("dogfood_capture_consent_missing")
    if not get_path(dogfood, ["summary", "training_enabled"], False):
        blockers.append("dogfood_training_consent_missing")
    if int(get_path(dogfood_bridge, ["training_rows_written"], 0) or 0) <= 0:
        blockers.append("dogfood_real_training_rows_missing")
    if not get_path(teacher_gate, ["summary", "teacher_share_metric_ready"], False):
        blockers.append("teacher_share_ledger_metric_missing")
    if int(get_path(teacher_manifest, ["summary", "teacher_rows_admitted"], 0) or 0) > 0 and not get_path(
        teacher_gate, ["summary", "distillation_allowed"], False
    ):
        blockers.append("teacher_rows_admitted_while_gate_locked")
    if int(get_path(training_admission, ["summary", "teacher_distillation_manifest_rows"], 0) or 0) > 0 and not get_path(
        training_admission, ["summary", "teacher_distillation_gate_allowed"], False
    ):
        blockers.append("teacher_training_rows_admitted_outside_gate")
    if int(get_path(training_admission, ["summary", "teacher_distillation_public_overlap_hits"], 0) or 0) > 0:
        blockers.append("teacher_distillation_public_overlap_detected")
    if not get_path(capability, ["summary", "no_cheat_clean"], False):
        blockers.append("capability_transfer_no_cheat_not_clean")
    if not get_path(capability, ["summary", "private_semantic_ready"], False):
        blockers.append("private_semantic_transfer_not_ready")
    if not get_path(capability, ["summary", "public_candidate_coverage_ready"], False):
        blockers.append("public_candidate_coverage_not_ready")
    if int(get_path(public_transfer_readiness, ["summary", "hard_failed_gate_count"], 0) or 0) > 0:
        blockers.append("public_transfer_readiness_hard_gates_failed")
    if int(get_path(public_transfer_readiness, ["summary", "evidence_failed_gate_count"], 0) or 0) > 0:
        blockers.append("public_transfer_readiness_evidence_gates_failed")
    if float_number(get_path(public_transfer_readiness, ["summary", "full_body_selected_pass_rate"], 0.0)) <= 0.0:
        blockers.append("public_transfer_readiness_full_body_selected_zero")
    if float_number(get_path(public_transfer_readiness, ["summary", "full_body_pass_if_any_rate"], 0.0)) <= 0.0:
        blockers.append("public_transfer_readiness_full_body_oracle_zero")
    if get_path(public_transfer_readiness, ["summary", "full_body_post_v4_default_semantic_dead"], True) is True:
        blockers.append("public_transfer_readiness_default_surface_semantic_dead")
    if int(
        get_path(
            capability,
            ["summary", "public_transfer_wall", "current_full_body_residual_mining", "private_residual_target_rows_written"],
            0,
        )
        or 0
    ) <= 0:
        blockers.append("private_public_transfer_residual_targets_missing")
    residual_consumer_state = get_path(
        capability,
        ["summary", "public_transfer_wall", "private_residual_target_consumer", "trigger_state"],
        "",
    )
    residual_repair_queue_rows = int(
        get_path(
            capability,
            ["summary", "public_transfer_wall", "private_residual_target_consumer", "repair_queue_rows"],
            0,
        )
        or 0
    )
    residual_unresolved_targets = int(
        get_path(
            capability,
            ["summary", "public_transfer_wall", "private_residual_target_consumer", "unresolved_target_count"],
            0,
        )
        or 0
    )
    if residual_consumer_state not in {"GREEN", "YELLOW"} or residual_repair_queue_rows <= 0:
        blockers.append("private_public_transfer_residual_targets_not_consumed")
    if residual_unresolved_targets > 0:
        blockers.append("private_public_transfer_residual_targets_unclosed")
    if int(
        get_path(
            capability,
            ["summary", "public_transfer_wall", "private_residual_target_consumer", "training_rows_written"],
            0,
        )
        or 0
    ) > 0:
        blockers.append("private_residual_target_consumer_wrote_training_rows")
    maturity_blockers = maturity.get("maturity_blockers") if isinstance(maturity.get("maturity_blockers"), list) else []
    if "public_transfer_floor_cleared" in maturity_blockers:
        blockers.append("public_transfer_floor_not_cleared")
    elif not maturity:
        blockers.append("maturity_integrity_audit_missing")
    elif maturity.get("trigger_state") != "GREEN":
        blockers.append("maturity_integrity_not_green")
    if maturity.get("trigger_state") not in {"GREEN", "YELLOW"}:
        blockers.append("maturity_integrity_audit_missing_or_red")
    private_v3_summary = private_v3_gate.get("summary") if isinstance(private_v3_gate.get("summary"), dict) else {}
    if private_v3_gate.get("trigger_state") != "GREEN":
        blockers.append("private_residual_v3_learned_gate_not_green")
    if private_v3_summary.get("private_heldout_adapter_off_scoring") is not True:
        blockers.append("private_residual_v3_adapter_off_scoring_missing")
    if float_number(private_v3_summary.get("private_learned_candidate_pass_rate")) < 0.70:
        blockers.append("private_residual_v3_learned_candidate_floor_not_met")
    if float_number(private_v3_summary.get("private_diagnostic_adapter_pass_rate")) > 0.0:
        blockers.append("private_residual_v3_diagnostic_adapter_counted")
    if int(get_path(private_ratchet, ["summary", "pending_queue_item_count"], 0) or 0) > 0:
        blockers.append("private_residual_ratchet_queue_not_drained")
    if vcm.get("trigger_state") != "GREEN":
        blockers.append("vcm_release_conformance_not_green")
    if not get_path(vcm, ["summary", "runtime_profile_claimed"], False):
        blockers.append("vcm_runtime_profile_not_claimed")
    if not get_path(vcm, ["summary", "native_runtime_claimable"], False):
        blockers.append("vcm_native_runtime_not_claimable")
    if mlx.get("trigger_state") != "GREEN":
        blockers.append("macos_mlx_structural_smoke_not_green")
    if not get_path(mlx, ["summary", "mlx_used"], False):
        blockers.append("macos_mlx_not_used")
    if int(get_path(mlx, ["summary", "fallback_return_rows"], 0) or 0) > 0:
        blockers.append("macos_mlx_fallback_returns_detected")
    if resource.get("trigger_state") != "GREEN":
        blockers.append("resource_governor_not_green")
    if not get_path(resource, ["summary", "can_run_requested_profile"], False):
        blockers.append("resource_governor_cannot_run_requested_profile")
    if autonomy.get("trigger_state") != "GREEN":
        blockers.append("autonomy_launch_readiness_not_green")
    if not get_path(autonomy, ["summary", "ready_for_autonomous_training"], False):
        blockers.append("autonomy_training_not_ready")
    if not get_path(public, ["summary", "approval_valid"], False):
        blockers.append("public_calibration_operator_approval_missing")
    if not get_path(metal, ["summary", "operator_approval_valid"], False):
        blockers.append("macos_metal_route_approval_missing")
    if not get_path(metal, ["summary", "production_route_allowed"], False):
        blockers.append("macos_metal_production_route_not_allowed")
    if not get_path(parity, ["summary", "parity_claim_allowed"], False):
        blockers.append("accelerator_parity_claim_not_allowed")
    if not cuda_env["cuda_tools_available"]:
        blockers.append("cuda_hardware_or_toolchain_unavailable_on_this_mac")
    if network.get("state") == "RED" and "coordinator_unreachable" in (network.get("red_finding_codes") or []):
        blockers.append("cuda_windows_node_unreachable_from_this_mac")
    blockers.extend(str(item) for item in parity.get("blockers", []) if isinstance(item, str) and item.startswith("cuda_"))
    return sorted(set(blockers))


def requirement_matrix(
    public: dict[str, Any],
    metal: dict[str, Any],
    parity: dict[str, Any],
    dogfood: dict[str, Any],
    dogfood_bridge: dict[str, Any],
    teacher_gate: dict[str, Any],
    teacher_manifest: dict[str, Any],
    training_admission: dict[str, Any],
    capability: dict[str, Any],
    public_transfer_readiness: dict[str, Any],
    maturity: dict[str, Any],
    private_v3_gate: dict[str, Any],
    private_ratchet: dict[str, Any],
    vcm: dict[str, Any],
    mlx: dict[str, Any],
    resource: dict[str, Any],
    autonomy: dict[str, Any],
    cuda_env: dict[str, Any],
    network: dict[str, Any],
) -> list[dict[str, Any]]:
    dogfood_capture_enabled = bool(get_path(dogfood, ["summary", "capture_enabled"], False))
    dogfood_training_enabled = bool(get_path(dogfood, ["summary", "training_enabled"], False))
    dogfood_training_consent_separate = bool(
        get_path(dogfood, ["summary", "training_consent_separate_from_capture"], False)
    )
    dogfood_rows_written = int(get_path(dogfood_bridge, ["training_rows_written"], 0) or 0)
    dogfood_trainable_events = int(get_path(dogfood_bridge, ["summary", "trainable_event_count"], 0) or 0)
    dogfood_bridge_clean = (
        dogfood_bridge.get("trigger_state") == "GREEN"
        and dogfood_rows_written > 0
        and dogfood_trainable_events > 0
        and get_path(dogfood_bridge, ["summary", "raw_text_capture_enabled"], True) is False
        and get_path(dogfood_bridge, ["summary", "teacher_used"], True) is False
        and int(get_path(dogfood_bridge, ["summary", "public_training_rows"], 1) or 0) == 0
        and int(get_path(dogfood_bridge, ["summary", "fallback_returns"], 1) or 0) == 0
    )
    teacher_metric_ready = bool(get_path(teacher_gate, ["summary", "teacher_share_metric_ready"], False))
    teacher_rows_admitted = int(get_path(teacher_manifest, ["summary", "teacher_rows_admitted"], 0) or 0)
    teacher_manifest_ledger_rows = int(get_path(teacher_manifest, ["summary", "ledger_row_count"], 0) or 0)
    teacher_training_manifest_rows = int(get_path(training_admission, ["summary", "teacher_distillation_manifest_rows"], 0) or 0)
    teacher_training_public_overlap_hits = int(
        get_path(training_admission, ["summary", "teacher_distillation_public_overlap_hits"], 0) or 0
    )
    teacher_gate_allowed = bool(get_path(teacher_gate, ["summary", "distillation_allowed"], False))
    teacher_training_gate_allowed = bool(
        get_path(training_admission, ["summary", "teacher_distillation_gate_allowed"], False)
    )
    capability_summary = capability.get("summary") if isinstance(capability.get("summary"), dict) else {}
    public_readiness_summary = (
        public_transfer_readiness.get("summary") if isinstance(public_transfer_readiness.get("summary"), dict) else {}
    )
    no_cheat = capability_summary.get("no_cheat") if isinstance(capability_summary.get("no_cheat"), dict) else {}
    vcm_summary = vcm.get("summary") if isinstance(vcm.get("summary"), dict) else {}
    mlx_summary = mlx.get("summary") if isinstance(mlx.get("summary"), dict) else {}
    resource_summary = resource.get("summary") if isinstance(resource.get("summary"), dict) else {}
    autonomy_summary = autonomy.get("summary") if isinstance(autonomy.get("summary"), dict) else {}
    private_v3_summary = private_v3_gate.get("summary") if isinstance(private_v3_gate.get("summary"), dict) else {}
    private_ratchet_summary = private_ratchet.get("summary") if isinstance(private_ratchet.get("summary"), dict) else {}
    maturity_blockers = maturity.get("maturity_blockers") if isinstance(maturity.get("maturity_blockers"), list) else []
    public_transfer_floor_cleared = maturity.get("trigger_state") == "GREEN" and not maturity_blockers
    return [
        requirement(
            "dogfood_local_consent_artifact_visible",
            "PROVEN",
            "reports/dogfood_trace_consent.json",
            {
                "config": dogfood.get("config"),
                "current_policy": get_path(dogfood, ["current", "policy"]),
                "raw_text_capture_enabled": get_path(dogfood, ["summary", "raw_text_capture_enabled"], False),
            },
        ),
        requirement(
            "dogfood_capture_consent_enabled",
            "PROVEN" if dogfood_capture_enabled else "BLOCKED",
            "reports/dogfood_trace_consent.json",
            {"capture_enabled": dogfood_capture_enabled},
            "operator must explicitly enable metadata-only capture consent",
        ),
        requirement(
            "dogfood_training_consent_enabled",
            "PROVEN" if dogfood_training_enabled and dogfood_training_consent_separate else "BLOCKED",
            "reports/dogfood_trace_consent.json",
            {
                "training_enabled": dogfood_training_enabled,
                "training_consent_separate_from_capture": dogfood_training_consent_separate,
            },
            "operator must explicitly enable separate private training export consent",
        ),
        requirement(
            "dogfood_real_events_and_training_rows",
            "PROVEN" if dogfood_bridge_clean else "BLOCKED",
            "reports/dogfood_trace_training_bridge.json",
            {
                "capture_enabled": dogfood_capture_enabled,
                "training_enabled": dogfood_training_enabled,
                "trainable_event_count": dogfood_trainable_events,
                "training_rows_written": dogfood_rows_written,
                "raw_text_capture_enabled": get_path(dogfood_bridge, ["summary", "raw_text_capture_enabled"], None),
                "teacher_used": get_path(dogfood_bridge, ["summary", "teacher_used"], None),
                "public_training_rows": get_path(dogfood_bridge, ["summary", "public_training_rows"], None),
                "fallback_returns": get_path(dogfood_bridge, ["summary", "fallback_returns"], None),
            },
            "real accepted/missed/ignored events cannot be written or exported before consent",
        ),
        requirement(
            "teacher_share_ledger_metric_ready",
            "PROVEN" if teacher_metric_ready else "BLOCKED",
            "reports/teacher_distillation_gate.json",
            {
                "distillation_allowed": teacher_gate_allowed,
                "teacher_share_metric_ready": teacher_metric_ready,
                "teacher_share_ledger_present": get_path(teacher_gate, ["summary", "teacher_share_ledger_present"], False),
                "teacher_share_ledger_row_count": get_path(teacher_gate, ["summary", "teacher_share_ledger_row_count"], 0),
                "teacher_accepted_row_share": get_path(teacher_gate, ["summary", "teacher_accepted_row_share"], 0.0),
                "teacher_share_cap": get_path(teacher_gate, ["summary", "teacher_share_cap"], None),
            },
            "teacher_share_of_accepted_training_rows must be ledger-backed before any teacher row admission",
        ),
        requirement(
            "teacher_proposals_retained_not_training",
            "PROVEN" if teacher_rows_admitted == 0 and teacher_manifest_ledger_rows >= 0 else "BLOCKED",
            "reports/teacher_distillation_manifest_audit.json",
            {
                "manifest_ready_for_distillation": get_path(teacher_manifest, ["summary", "manifest_ready_for_distillation"], False),
                "proposal_rows_retained_not_training": get_path(
                    teacher_manifest, ["summary", "proposal_rows_retained_not_training"], 0
                ),
                "admission_safety_checks_clean": get_path(
                    teacher_manifest, ["summary", "admission_safety_checks_clean"], None
                ),
                "verifier_pass_rate_applicable": get_path(
                    teacher_manifest, ["summary", "verifier_pass_rate_applicable"], None
                ),
                "teacher_rows_admitted": teacher_rows_admitted,
                "manifest_ledger_row_count": teacher_manifest_ledger_rows,
                "public_training_rows_written": get_path(teacher_manifest, ["summary", "public_training_rows_written"], 0),
            },
            "proposal-mode teacher outputs must stay retained evidence, not training rows",
        ),
        requirement(
            "teacher_training_admission_gate_enforced",
            "PROVEN"
            if teacher_training_manifest_rows == 0
            or (teacher_training_manifest_rows > 0 and teacher_training_gate_allowed and teacher_gate_allowed)
            else "BLOCKED",
            "reports/training_data_admission_v1.json",
            {
                "teacher_distillation_gate_allowed": teacher_training_gate_allowed,
                "teacher_distillation_manifest_rows": teacher_training_manifest_rows,
                "teacher_distillation_ledger_rows": get_path(training_admission, ["summary", "teacher_distillation_ledger_rows"], 0),
                "teacher_distillation_proposal_ledger_rows": get_path(
                    training_admission, ["summary", "teacher_distillation_proposal_ledger_rows"], 0
                ),
                "teacher_distillation_admission_safety_clean": get_path(
                    training_admission, ["summary", "teacher_distillation_admission_safety_clean"], None
                ),
                "public_benchmark_training_allowed": get_path(training_admission, ["summary", "public_benchmark_training_allowed"], None),
                "public_benchmark_payload_admitted": get_path(training_admission, ["summary", "public_benchmark_payload_admitted"], None),
            },
            "teacher rows must be admitted only through the governed teacher distillation gate",
        ),
        requirement(
            "teacher_public_and_runtime_boundary_clean",
            "PROVEN"
            if teacher_training_public_overlap_hits == 0
            and int(get_path(teacher_manifest, ["summary", "public_training_rows_written"], 0) or 0) == 0
            and int(get_path(teacher_gate, ["summary", "hard_blocker_count"], 1) or 0) == 0
            else "BLOCKED",
            "reports/teacher_distillation_gate.json",
            {
                "hard_blocker_count": get_path(teacher_gate, ["summary", "hard_blocker_count"], None),
                "public_overlap_hits": teacher_training_public_overlap_hits,
                "public_training_rows_written": get_path(teacher_manifest, ["summary", "public_training_rows_written"], 0),
                "runtime_external_tokens_forbidden": get_path(teacher_gate, ["summary", "runtime_external_tokens_forbidden"], None),
            },
            "teacher data must never train on public benchmark content or serve external tokens at runtime",
        ),
        requirement(
            "private_semantic_transfer_ready",
            "PROVEN" if bool(capability_summary.get("private_semantic_ready")) else "BLOCKED",
            "reports/capability_transfer_closure_v1.json",
            {
                "private_selected_pass_rate": capability_summary.get("private_selected_pass_rate"),
                "private_pass_if_any_rate": capability_summary.get("private_pass_if_any_rate"),
                "private_no_admissible_task_rate": capability_summary.get("private_no_admissible_task_rate"),
                "private_semantic_eligible_candidate_count": capability_summary.get("private_semantic_eligible_candidate_count"),
            },
            "private strict learned candidates must stay semantically strong before any public spend",
        ),
        requirement(
            "public_candidate_coverage_ready",
            "PROVEN" if bool(capability_summary.get("public_candidate_coverage_ready")) else "BLOCKED",
            "reports/capability_transfer_closure_v1.json",
            {
                "public_candidate_coverage_ready": capability_summary.get("public_candidate_coverage_ready"),
                "strict_public_promotion_candidate_count": capability_summary.get("strict_public_promotion_candidate_count"),
                "public_candidate_coverage": compact_public_candidate_coverage(capability_summary),
            },
            "public-like candidate coverage must be full-body learned, admissible, and no-cheat before calibration",
        ),
        requirement(
            "public_transfer_readiness_private_evidence_current",
            "PROVEN"
            if public_transfer_readiness.get("trigger_state") in {"GREEN", "YELLOW"}
            and int(public_readiness_summary.get("hard_failed_gate_count") or 0) == 0
            and int(public_readiness_summary.get("evidence_failed_gate_count") or 0) == 0
            and float_number(public_readiness_summary.get("full_body_selected_pass_rate")) > 0.0
            and float_number(public_readiness_summary.get("full_body_pass_if_any_rate")) > 0.0
            and int(public_readiness_summary.get("full_body_fallback_return_candidate_count") or 0) == 0
            and int(public_readiness_summary.get("full_body_public_leakage_count") or 0) == 0
            and public_readiness_summary.get("full_body_post_v4_default_semantic_dead") is False
            and public_transfer_readiness.get("public_calibration_allowed") is False
            else "BLOCKED",
            "reports/public_transfer_readiness_refresh_v1.json",
            {
                "trigger_state": public_transfer_readiness.get("trigger_state"),
                "contract_task_count": public_readiness_summary.get("contract_task_count"),
                "hard_failed_gate_count": public_readiness_summary.get("hard_failed_gate_count"),
                "evidence_failed_gate_count": public_readiness_summary.get("evidence_failed_gate_count"),
                "transfer_failed_gate_count": public_readiness_summary.get("transfer_failed_gate_count"),
                "full_body_selected_pass_rate": public_readiness_summary.get("full_body_selected_pass_rate"),
                "full_body_pass_if_any_rate": public_readiness_summary.get("full_body_pass_if_any_rate"),
                "full_body_benchmark_promotion_eligible_candidate_count": public_readiness_summary.get(
                    "full_body_benchmark_promotion_eligible_candidate_count"
                ),
                "full_body_default_semantic_dead": public_readiness_summary.get(
                    "full_body_post_v4_default_semantic_dead"
                ),
                "public_calibration_allowed": public_transfer_readiness.get("public_calibration_allowed"),
            },
            "public-transfer readiness must show fresh private evidence while keeping public calibration locked",
        ),
        requirement(
            "private_public_transfer_residual_targets_prepared",
            "PROVEN"
            if int(
                get_path(
                    capability,
                    [
                        "summary",
                        "public_transfer_wall",
                        "current_full_body_residual_mining",
                        "private_residual_target_rows_written",
                    ],
                    0,
                )
                or 0
            )
            > 0
            else "BLOCKED",
            "reports/bounded_public_transfer_residual_mining_current_private_full_body_v1.json",
            {
                "target_rows": get_path(
                    capability,
                    [
                        "summary",
                        "public_transfer_wall",
                        "current_full_body_residual_mining",
                        "private_residual_target_rows_written",
                    ],
                    0,
                ),
                "target_manifest": get_path(
                    capability,
                    [
                        "summary",
                        "public_transfer_wall",
                        "current_full_body_residual_mining",
                        "private_residual_target_manifest",
                    ],
                    "",
                ),
                "training_rows_written": get_path(
                    capability,
                    [
                        "summary",
                        "public_transfer_wall",
                        "current_full_body_residual_mining",
                        "training_rows_written",
                    ],
                    0,
                ),
            },
            "public-transfer failures must become private-only residual targets without public training rows",
        ),
        requirement(
            "private_public_transfer_residual_targets_consumed",
            "PROVEN"
            if get_path(
                capability,
                ["summary", "public_transfer_wall", "private_residual_target_consumer", "trigger_state"],
                "",
            )
            in {"GREEN", "YELLOW"}
            and int(
                get_path(
                    capability,
                    ["summary", "public_transfer_wall", "private_residual_target_consumer", "repair_queue_rows"],
                    0,
                )
                or 0
            )
            > 0
            and int(
                get_path(
                    capability,
                    ["summary", "public_transfer_wall", "private_residual_target_consumer", "training_rows_written"],
                    0,
                )
                or 0
            )
            == 0
            else "BLOCKED",
            "reports/private_residual_target_consumer_v1.json",
            {
                "trigger_state": get_path(
                    capability,
                    ["summary", "public_transfer_wall", "private_residual_target_consumer", "trigger_state"],
                    "",
                ),
                "repair_queue_rows": get_path(
                    capability,
                    ["summary", "public_transfer_wall", "private_residual_target_consumer", "repair_queue_rows"],
                    0,
                ),
                "queue_path": get_path(
                    capability,
                    ["summary", "public_transfer_wall", "private_residual_target_consumer", "queue_path"],
                    "",
                ),
                "training_rows_written": get_path(
                    capability,
                    ["summary", "public_transfer_wall", "private_residual_target_consumer", "training_rows_written"],
                    0,
                ),
                "public_training_rows_written": get_path(
                    capability,
                    ["summary", "public_transfer_wall", "private_residual_target_consumer", "public_training_rows_written"],
                    0,
                ),
                "unresolved_target_count": get_path(
                    capability,
                    ["summary", "public_transfer_wall", "private_residual_target_consumer", "unresolved_target_count"],
                    0,
                ),
            },
            "private residual targets must be consumed into a private repair queue without becoming training rows",
        ),
        requirement(
            "private_public_transfer_residual_targets_closed",
            "PROVEN"
            if get_path(
                capability,
                ["summary", "public_transfer_wall", "private_residual_target_consumer", "trigger_state"],
                "",
            )
            == "GREEN"
            and int(
                get_path(
                    capability,
                    ["summary", "public_transfer_wall", "private_residual_target_consumer", "unresolved_target_count"],
                    0,
                )
                or 0
            )
            == 0
            else "BLOCKED",
            "reports/private_residual_target_consumer_v1.json",
            {
                "trigger_state": get_path(
                    capability,
                    ["summary", "public_transfer_wall", "private_residual_target_consumer", "trigger_state"],
                    "",
                ),
                "unresolved_target_count": get_path(
                    capability,
                    ["summary", "public_transfer_wall", "private_residual_target_consumer", "unresolved_target_count"],
                    0,
                ),
                "unresolved_target_state_counts": get_path(
                    capability,
                    ["summary", "public_transfer_wall", "private_residual_target_consumer", "unresolved_target_state_counts"],
                    {},
                ),
                "unresolved_target_category_counts": get_path(
                    capability,
                    ["summary", "public_transfer_wall", "private_residual_target_consumer", "unresolved_target_category_counts"],
                    {},
                ),
            },
            "private residual targets must be closed by current private evidence or remain an explicit blocker",
        ),
        requirement(
            "private_residual_v3_learned_path_green",
            "PROVEN"
            if private_v3_gate.get("trigger_state") == "GREEN"
            and private_v3_summary.get("private_heldout_adapter_off_scoring") is True
            and float_number(private_v3_summary.get("private_learned_candidate_pass_rate")) >= 0.70
            and float_number(private_v3_summary.get("private_diagnostic_adapter_pass_rate")) == 0.0
            and int(private_v3_summary.get("pending_gate_count") or 0) == 0
            and int(private_v3_summary.get("failed_gate_count") or 0) == 0
            and int(private_ratchet_summary.get("pending_queue_item_count") or 0) == 0
            else "BLOCKED",
            "reports/private_residual_repair_v3_gate.json",
            {
                "trigger_state": private_v3_gate.get("trigger_state"),
                "adapter_off_scoring": private_v3_summary.get("private_heldout_adapter_off_scoring"),
                "learned_candidate_pass_rate": private_v3_summary.get("private_learned_candidate_pass_rate"),
                "diagnostic_adapter_pass_rate": private_v3_summary.get("private_diagnostic_adapter_pass_rate"),
                "pending_gate_count": private_v3_summary.get("pending_gate_count"),
                "failed_gate_count": private_v3_summary.get("failed_gate_count"),
                "ratchet_pending_queue_item_count": private_ratchet_summary.get("pending_queue_item_count"),
                "ratchet_completed_queue_item_count": private_ratchet_summary.get("completed_queue_item_count"),
            },
            "private residual v3 evidence must come from learned structural candidates with diagnostic adapters excluded from pass credit",
        ),
        requirement(
            "public_transfer_floor_cleared",
            "PROVEN" if public_transfer_floor_cleared else "BLOCKED",
            "reports/maturity_integrity_audit.json",
            {
                "locked_broad_public_pass_rate": get_path(capability, ["summary", "public_transfer_wall", "locked_broad_public_pass_rate"], None),
                "locked_broad_public_pass_count": get_path(capability, ["summary", "public_transfer_wall", "locked_broad_public_pass_count"], None),
                "locked_broad_public_task_count": get_path(capability, ["summary", "public_transfer_wall", "locked_broad_public_task_count"], None),
                "maturity_blockers": maturity_blockers,
                "public_calibration_allowed": get_path(maturity, ["summary", "public_calibration_allowed"], None),
            },
            "locked public transfer floor remains below promotion threshold until one governed calibration proves lift",
        ),
        requirement(
            "transfer_no_cheat_invariants",
            "PROVEN" if bool(capability_summary.get("no_cheat_clean")) else "BLOCKED",
            "reports/capability_transfer_closure_v1.json",
            {
                "no_cheat_clean": capability_summary.get("no_cheat_clean"),
                "external_inference_calls": no_cheat.get("external_inference_calls"),
                "public_training_rows": no_cheat.get("public_training_rows"),
                "fallback_return_count": no_cheat.get("fallback_return_count"),
                "template_like_count": no_cheat.get("template_like_count"),
                "teacher_used_count": no_cheat.get("teacher_used_count"),
            },
            "transfer evidence cannot include fallback returns, templates, public leakage, or external inference",
        ),
        requirement(
            "vcm_native_runtime_scope_real",
            "PROVEN"
            if vcm.get("trigger_state") == "GREEN"
            and bool(vcm_summary.get("runtime_profile_claimed"))
            and bool(vcm_summary.get("native_runtime_claimable"))
            and vcm_summary.get("runtime_cache_lifecycle_state") == "GREEN"
            else "BLOCKED",
            "reports/vcm_release_conformance_audit.json",
            {
                "profile_states": vcm_summary.get("profile_states"),
                "native_runtime_claim_scope": vcm_summary.get("native_runtime_claim_scope"),
                "native_runtime_claim_backend": vcm_summary.get("native_runtime_claim_backend"),
                "native_runtime_recommended_backend": vcm_summary.get("native_runtime_recommended_execution_backend"),
                "native_runtime_backend_matches_recommended": vcm_summary.get(
                    "native_runtime_claim_backend_matches_recommended_execution_backend"
                ),
                "scheduler_native_kv_route_allowed_for_recommended_backend": vcm_summary.get(
                    "scheduler_native_kv_route_allowed_for_recommended_backend"
                ),
                "mlx_tensor_descriptor_lifecycle_test_passed": vcm_summary.get(
                    "mlx_tensor_descriptor_lifecycle_test_passed"
                ),
                "recommended_backend_runtime_descriptor_lifecycle_claimable": vcm_summary.get(
                    "recommended_backend_runtime_descriptor_lifecycle_claimable"
                ),
                "scheduler_vcm_descriptor_route_allowed_for_recommended_backend": vcm_summary.get(
                    "scheduler_vcm_descriptor_route_allowed_for_recommended_backend"
                ),
                "scheduler_native_kv_route_fail_closed": vcm_summary.get("scheduler_native_kv_route_fail_closed"),
                "runtime_cache_lifecycle_state": vcm_summary.get("runtime_cache_lifecycle_state"),
                "runtime_cache_reuse_hit_rate": vcm_summary.get("runtime_cache_reuse_hit_rate"),
                "accelerator_kv_parity_claimed": vcm_summary.get("accelerator_kv_parity_claimed"),
                "mlx_native_kv_parity_claimed": vcm_summary.get("mlx_native_kv_parity_claimed"),
                "metal_native_kv_parity_claimed": vcm_summary.get("metal_native_kv_parity_claimed"),
                "cuda_native_kv_parity_claimed": vcm_summary.get("cuda_native_kv_parity_claimed"),
            },
            "VCM-Runtime must be a real scoped runtime/cache lifecycle claim, not an accelerator parity claim",
        ),
        requirement(
            "macos_mlx_execution_green",
            "PROVEN"
            if mlx.get("trigger_state") == "GREEN"
            and bool(mlx_summary.get("mlx_used"))
            and int(mlx_summary.get("fallback_return_rows") or 0) == 0
            and int(mlx_summary.get("public_training_rows") or 0) == 0
            else "BLOCKED",
            "reports/macos_mlx_structural_action_smoke.json",
            {
                "mlx_available": mlx_summary.get("mlx_available"),
                "mlx_used": mlx_summary.get("mlx_used"),
                "mlx_default_device": mlx_summary.get("mlx_default_device"),
                "verifier_pass_rate": mlx_summary.get("verifier_pass_rate"),
                "fallback_return_rows": mlx_summary.get("fallback_return_rows"),
                "public_training_rows": mlx_summary.get("public_training_rows"),
            },
            "Apple Silicon lane must use MLX with no fallback returns or public training rows",
        ),
        requirement(
            "mac_resource_and_autonomy_ready",
            "PROVEN"
            if resource.get("trigger_state") == "GREEN"
            and autonomy.get("trigger_state") == "GREEN"
            and bool(resource_summary.get("can_run_requested_profile"))
            and bool(autonomy_summary.get("ready_for_autonomous_training"))
            else "BLOCKED",
            "reports/autonomy_launch_readiness.json",
            {
                "resource_governor_state": resource.get("trigger_state"),
                "execution_owner": resource_summary.get("execution_owner"),
                "can_run_requested_profile": resource_summary.get("can_run_requested_profile"),
                "autonomy_launch_state": autonomy.get("trigger_state"),
                "ready_for_autonomous_training": autonomy_summary.get("ready_for_autonomous_training"),
                "ready_for_teacher_enabled_run": autonomy_summary.get("ready_for_teacher_enabled_run"),
            },
            "Mac resource policy and autonomy profile must be internally consistent for unattended work",
        ),
        requirement(
            "public_calibration_packet_ready",
            "PROVEN" if get_path(public, ["summary", "packet_ready"], False) else "MISSING",
            "reports/operator_bounded_public_calibration_dry_run.json",
            {"packet_ready": get_path(public, ["summary", "packet_ready"], False)},
        ),
        requirement(
            "public_calibration_approval_and_single_execution",
            "BLOCKED",
            "reports/operator_bounded_public_calibration_dry_run.json",
            {
                "approval_valid": get_path(public, ["summary", "approval_valid"], False),
                "executed": get_path(public, ["summary", "executed"], False),
                "would_execute": get_path(public, ["summary", "would_execute"], False),
            },
            "operator approval artifact is required before exactly one public calibration run",
        ),
        requirement(
            "macos_metal_guarded_evidence",
            "PROVEN" if int(get_path(metal, ["summary", "guarded_evidence_ok_count"], 0)) >= 4 else "MISSING",
            "reports/macos_metal_production_route_readiness.json",
            {
                "guarded_evidence_ok_count": get_path(metal, ["summary", "guarded_evidence_ok_count"], 0),
                "hard_failure_count": get_path(metal, ["summary", "hard_failure_count"], 0),
            },
        ),
        requirement(
            "macos_metal_production_route_approval",
            "PROVEN"
            if get_path(metal, ["summary", "operator_approval_valid"], False)
            and get_path(metal, ["summary", "production_route_allowed"], False)
            else "BLOCKED",
            "reports/macos_metal_production_route_readiness.json",
            {
                "operator_approval_valid": get_path(metal, ["summary", "operator_approval_valid"], False),
                "production_route_allowed": get_path(metal, ["summary", "production_route_allowed"], False),
            },
            "operator approval artifact is required before bounded production route enablement",
        ),
        requirement(
            "cuda_reference_reports_present",
            "BLOCKED",
            "reports/accelerator_parity_claim_readiness.json",
            {
                "local_cuda_tools_available": cuda_env.get("cuda_tools_available"),
                "windows_coordinator_reachable": network.get("coordinator_reachable"),
                "parity_ready_surface_count": get_path(parity, ["summary", "parity_ready_surface_count"], 0),
            },
            "CUDA reference reports must be produced on a CUDA-capable reachable node",
        ),
        requirement(
            "accelerator_parity_claim_approval",
            "BLOCKED",
            "reports/accelerator_parity_claim_readiness.json",
            {
                "parity_claim_allowed": get_path(parity, ["summary", "parity_claim_allowed"], False),
                "approval_valid": get_path(parity, ["approval", "valid"], False),
            },
            "parity approval is separate from production routing and requires CUDA-vs-Metal evidence",
        ),
        requirement(
            "no_cheat_invariants",
            "PROVEN",
            "reports/theseus_gate_closure_packet.json",
            {
                "external_inference_calls": 0,
                "public_training_rows": 0,
                "fallback_returns": 0,
                "teacher_calls": 0,
                "serving_external_inference": False,
            },
        ),
    ]


def requirement(
    name: str,
    state: str,
    evidence: str,
    facts: dict[str, Any],
    blocker: str = "",
) -> dict[str, Any]:
    row = {"name": name, "state": state, "evidence": evidence, "facts": facts}
    if blocker:
        row["blocker"] = blocker
    return row


def compact_public_candidate_coverage(summary: dict[str, Any]) -> dict[str, Any]:
    coverage = summary.get("public_candidate_coverage") if isinstance(summary.get("public_candidate_coverage"), dict) else {}
    keys = [
        "task_count",
        "candidate_count",
        "full_body_token_candidate_count",
        "grammar_masked_learned_token_candidate_count",
        "semantic_structure_multi_statement_candidate_rate",
        "return_contract_expected_shape_coverage_rate",
        "template_like_candidate_count",
        "expression_memory_fallback_count",
        "external_inference_calls",
        "public_tests_visible_to_generator",
        "canonical_solution_seen_by_solver",
        "ready",
    ]
    return {key: coverage.get(key) for key in keys if key in coverage}


def supporting_audits() -> list[dict[str, Any]]:
    specs = [
        ("dogfood_readiness", "reports/dogfood_trace_readiness.json"),
        ("dogfood_event_dry_run", "reports/dogfood_trace_event_dry_run.json"),
        ("dogfood_training_bridge", "reports/dogfood_trace_training_bridge.json"),
        ("teacher_distillation_gate", "reports/teacher_distillation_gate.json"),
        ("teacher_distillation_manifest", "reports/teacher_distillation_manifest_audit.json"),
        ("training_data_admission", "reports/training_data_admission_v1.json"),
        ("capability_transfer_closure", "reports/capability_transfer_closure_v1.json"),
        ("public_transfer_readiness", "reports/public_transfer_readiness_refresh_v1.json"),
        ("private_residual_target_consumer", "reports/private_residual_target_consumer_v1.json"),
        ("private_residual_v3_gate", "reports/private_residual_repair_v3_gate.json"),
        ("private_residual_ratchet", "reports/private_residual_self_improvement_ratchet_v1.json"),
        ("vcm_release_conformance", "reports/vcm_release_conformance_audit.json"),
        ("macos_mlx_structural_action_smoke", "reports/macos_mlx_structural_action_smoke.json"),
        ("resource_governor", "reports/resource_governor.json"),
        ("autonomy_launch_readiness", "reports/autonomy_launch_readiness.json"),
        ("public_calibration_dry_run", "reports/operator_bounded_public_calibration_dry_run.json"),
        ("macos_metal_route_readiness", "reports/macos_metal_production_route_readiness.json"),
        ("accelerator_manifest", "reports/accelerator_parity_manifest.json"),
        ("accelerator_parity_claim", "reports/accelerator_parity_claim_readiness.json"),
        ("external_inference", "reports/external_inference_audit.json"),
        ("maturity_integrity", "reports/maturity_integrity_audit.json"),
        ("candidate_promotion", "reports/candidate_promotion_gate.json"),
        ("overnight_training", "reports/hive_overnight_training_report.json"),
    ]
    rows: list[dict[str, Any]] = []
    for name, path_text in specs:
        data = read_json(resolve(path_text), {})
        summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
        rows.append(
            {
                "name": name,
                "path": path_text,
                "exists": bool(data),
                "ok": data.get("ok"),
                "trigger_state": data.get("trigger_state"),
                "promote": data.get("promote"),
                "passed": data.get("passed"),
                "created_utc": data.get("created_utc"),
                "summary": compact_summary(summary),
            }
        )
    return rows


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "capture_enabled",
        "training_enabled",
        "raw_text_capture_enabled",
        "training_rows_written",
        "event_count",
        "distillation_allowed",
        "teacher_share_metric_ready",
        "teacher_share_ledger_present",
        "teacher_share_ledger_row_count",
        "teacher_accepted_row_share",
        "manifest_ready_for_distillation",
        "proposal_rows_retained_not_training",
        "teacher_rows_admitted",
        "ledger_row_count",
        "teacher_distillation_gate_allowed",
        "teacher_distillation_manifest_rows",
        "teacher_distillation_ledger_rows",
        "teacher_distillation_public_overlap_hits",
        "public_benchmark_payload_admitted",
        "public_benchmark_training_allowed",
        "private_semantic_ready",
        "private_selected_pass_rate",
        "private_pass_if_any_rate",
        "private_no_admissible_task_rate",
        "public_candidate_coverage_ready",
        "public_promotion_ready",
        "contract_task_count",
        "hard_failed_gate_count",
        "evidence_failed_gate_count",
        "transfer_failed_gate_count",
        "full_body_selected_pass_rate",
        "full_body_pass_if_any_rate",
        "full_body_benchmark_promotion_eligible_candidate_count",
        "full_body_post_v4_default_semantic_dead",
        "private_residual_target_rows_written",
        "private_residual_target_manifest",
        "private_residual_target_consumer_state",
        "private_residual_repair_queue_rows",
        "private_residual_repair_queue_path",
        "private_residual_consumer_training_rows_written",
        "private_residual_v3_gate_state",
        "private_residual_v3_learned_candidate_pass_rate",
        "private_residual_v3_diagnostic_adapter_pass_rate",
        "private_residual_v3_adapter_off_scoring",
        "private_residual_v3_pending_gate_count",
        "private_residual_v3_failed_gate_count",
        "private_residual_ratchet_completed_queue_item_count",
        "private_residual_ratchet_pending_queue_item_count",
        "private_learned_candidate_pass_rate",
        "private_diagnostic_adapter_pass_rate",
        "private_heldout_adapter_off_scoring",
        "pending_gate_count",
        "failed_gate_count",
        "completed_queue_item_count",
        "pending_queue_item_count",
        "locked_broad_public_pass_rate",
        "locked_broad_public_pass_count",
        "locked_broad_public_task_count",
        "maturity_blocker_count",
        "runtime_profile_claimed",
        "native_runtime_claimable",
        "native_runtime_claim_scope",
        "runtime_cache_lifecycle_state",
        "accelerator_kv_parity_claimed",
        "mlx_available",
        "mlx_used",
        "mlx_default_device",
        "verifier_pass_rate",
        "fallback_return_rows",
        "can_run_requested_profile",
        "execution_owner",
        "ready_for_autonomous_training",
        "ready_for_teacher_enabled_run",
        "packet_ready",
        "approval_valid",
        "executed",
        "guarded_evidence_ok_count",
        "production_route_allowed",
        "surface_ok_count",
        "hard_failure_count",
        "parity_claim_allowed",
        "parity_ready_surface_count",
        "total_violations",
        "hard_blocker_count",
        "maturity_blocker_count",
        "candidate_promotion_allowed",
        "worker_report_count",
        "failed_count",
        "stale_lease_count",
        "promotion_count",
    ]
    return {key: summary.get(key) for key in keys if key in summary}


def cuda_environment() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "nvcc_path": shutil.which("nvcc") or "",
        "nvidia_smi_path": shutil.which("nvidia-smi") or "",
        "cuda_tools_available": bool(shutil.which("nvcc") and shutil.which("nvidia-smi")),
        "interpretation": "CUDA reference reports must be generated on a CUDA-capable node; this local Mac cannot produce them."
        if not (shutil.which("nvcc") and shutil.which("nvidia-smi"))
        else "Local CUDA tools are present; CUDA reference commands may be runnable here.",
    }


def network_doctor_summary() -> dict[str, Any]:
    report = read_json(REPORTS / "hive_network_doctor.json", {})
    return summarize_network_doctor(report)


def refresh_network_doctor() -> dict[str, Any]:
    report = run_json(
        [
            sys.executable,
            "scripts/hive_network_doctor.py",
            "--timeout",
            "2",
            "--retries",
            "1",
            "--out",
            "reports/hive_network_doctor.json",
            "--markdown-out",
            "reports/hive_network_doctor.md",
        ],
        timeout=45,
    )
    if report:
        return summarize_network_doctor(report)
    return network_doctor_summary()


def summarize_network_doctor(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return {
        "report": "reports/hive_network_doctor.json",
        "state": report.get("state"),
        "created_utc": report.get("created_utc"),
        "coordinator_reachable": summary.get("coordinator_reachable"),
        "remote_peer_reachable_count": summary.get("remote_peer_reachable_count"),
        "stale_peer_count": summary.get("stale_peer_count"),
        "red_finding_codes": summary.get("red_finding_codes") if isinstance(summary.get("red_finding_codes"), list) else [],
        "yellow_finding_codes": summary.get("yellow_finding_codes") if isinstance(summary.get("yellow_finding_codes"), list) else [],
    }


def run_json(command: list[str], *, timeout: int = 120) -> dict[str, Any]:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        payload = {}
    if isinstance(payload, dict):
        payload.setdefault("_returncode", result.returncode)
        if result.stderr:
            payload.setdefault("_stderr_tail", result.stderr[-2000:])
        return payload
    return {"_returncode": result.returncode}


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def float_number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_json(path: Path, default: Any) -> Any:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    return data if isinstance(data, dict) else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# Theseus Remaining Gate Closure Packet",
        "",
        f"- State: `{report.get('trigger_state')}`",
        f"- Dogfood capture enabled: `{summary.get('dogfood_capture_enabled')}`",
        f"- Dogfood training enabled: `{summary.get('dogfood_training_enabled')}`",
        f"- Teacher share metric ready: `{summary.get('teacher_share_metric_ready')}`",
        f"- Teacher rows admitted: `{summary.get('teacher_rows_admitted')}`",
        f"- Teacher proposal rows retained: `{summary.get('teacher_proposal_rows_retained_not_training')}`",
        f"- Private semantic ready: `{summary.get('private_semantic_ready')}`",
        f"- Public candidate coverage ready: `{summary.get('public_candidate_coverage_ready')}`",
        f"- Private residual target rows: `{summary.get('private_residual_target_rows_written')}`",
        f"- Private residual repair queue rows: `{summary.get('private_residual_repair_queue_rows')}`",
        f"- Private residual unresolved targets: `{summary.get('private_residual_repair_unresolved_target_count')}`",
        f"- Private residual v3 learned pass: `{summary.get('private_residual_v3_learned_candidate_pass_rate')}`",
        f"- Private residual v3 diagnostic adapter pass: `{summary.get('private_residual_v3_diagnostic_adapter_pass_rate')}`",
        f"- Private residual v3 adapter-off scoring: `{summary.get('private_residual_v3_adapter_off_scoring')}`",
        f"- Public-transfer readiness: `{summary.get('public_transfer_readiness_state')}`; full-body selected/pass-if-any `{summary.get('public_transfer_full_body_selected_pass_rate')}` / `{summary.get('public_transfer_full_body_pass_if_any_rate')}`",
        f"- Locked broad public score: `{summary.get('locked_broad_public_pass_count')}/{summary.get('locked_broad_public_task_count')}` = `{summary.get('locked_broad_public_pass_rate')}`",
        f"- VCM runtime scope: `{summary.get('vcm_native_runtime_claim_scope')}`",
        f"- VCM claim/recommended backend: `{summary.get('vcm_native_runtime_claim_backend')}` / `{summary.get('vcm_native_runtime_recommended_backend')}`; scheduler native KV route allowed `{summary.get('vcm_scheduler_native_kv_route_allowed_for_recommended_backend')}`",
        f"- VCM MLX descriptor lifecycle: `{summary.get('vcm_mlx_tensor_descriptor_lifecycle_test_passed')}`; scheduler descriptor route allowed `{summary.get('vcm_scheduler_descriptor_route_allowed_for_recommended_backend')}`",
        f"- MLX used: `{summary.get('mlx_used')}`",
        f"- Resource owner: `{summary.get('resource_execution_owner')}`",
        f"- Autonomous training ready: `{summary.get('ready_for_autonomous_training')}`",
        f"- Public approval valid: `{summary.get('public_approval_valid')}`",
        f"- Metal route allowed: `{summary.get('metal_route_allowed')}`",
        f"- Parity claim allowed: `{summary.get('parity_claim_allowed')}`",
        f"- Templates written: `{len(summary.get('templates_written') or [])}`",
        "",
        "## Remaining Blockers",
    ]
    for blocker in report.get("remaining_blockers", []):
        lines.append(f"- `{blocker}`")
    lines.extend(["", "## Requirement Matrix"])
    for row in report.get("requirement_matrix", []):
        lines.append(f"- `{row.get('state')}` `{row.get('name')}` via `{row.get('evidence')}`")
        if row.get("blocker"):
            lines.append(f"  blocker: {row.get('blocker')}")
    lines.extend(["", "## Supporting Audits"])
    for row in report.get("supporting_audits", []):
        status = row.get("trigger_state")
        if status is None:
            status = row.get("ok")
        if status is None:
            status = row.get("promote")
        lines.append(f"- `{row.get('name')}` status=`{status}` path=`{row.get('path')}`")
    cuda_env = report.get("cuda_environment") if isinstance(report.get("cuda_environment"), dict) else {}
    network = report.get("network_doctor") if isinstance(report.get("network_doctor"), dict) else {}
    lines.extend(
        [
            "",
            "## CUDA/Network Evidence",
            "",
            f"- Local CUDA tools available: `{cuda_env.get('cuda_tools_available')}`",
            f"- Local system: `{cuda_env.get('system')} {cuda_env.get('machine')}`",
            f"- Network doctor state: `{network.get('state')}`",
            f"- Coordinator reachable: `{network.get('coordinator_reachable')}`",
        ]
    )
    lines.extend(["", "## CUDA Reference Commands"])
    for row in report.get("cuda_reference_commands", []):
        lines.append(f"- `{row.get('surface')}` -> `{row.get('required_report')}`")
        lines.append(f"  `{row.get('command')}`")
    lines.extend(["", "Templates in this packet keep `approved=false`; copying one is not approval until the operator deliberately edits it."])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
