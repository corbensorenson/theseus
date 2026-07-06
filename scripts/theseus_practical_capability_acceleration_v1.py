#!/usr/bin/env python3
"""Run the practical capability acceleration loop.

This orchestrator ties together the evidence-producing pieces that already
exist in Theseus: private residual targets, dogfood metadata, governed teacher
distillation, admitted training data, VCM context routing, the broad survival
lane comparator, Mac acceleration probes, and the promotion/growth gates.

It intentionally does not run public calibration, train on public benchmark
payloads, call a live teacher, serve external tokens, or count fallback returns.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RUN_ROOT = REPORTS / "practical_capability_acceleration_v1"
DEFAULT_TARGETS = REPORTS / "bounded_public_transfer_calibration_review_v1_private_residual_targets.jsonl"
DEFAULT_OUT = REPORTS / "theseus_practical_capability_acceleration_v1.json"
DEFAULT_MD = REPORTS / "theseus_practical_capability_acceleration_v1.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--seed", type=int, default=313)
    parser.add_argument("--max-train-rows", type=int, default=1536)
    parser.add_argument("--max-eval-rows", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--targets", default=rel(DEFAULT_TARGETS))
    parser.add_argument("--resume-run-dir", default="")
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    args = parser.parse_args()

    started = time.perf_counter()
    if args.execute and args.resume_run_dir:
        run_dir = resolve(args.resume_run_dir)
        report = summarize_existing_run(args, run_dir, started=started)
    elif args.execute:
        report = run_acceleration(args, started=started)
    else:
        report = planned_report(args, started=started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(compact_report(report), indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") in {"GREEN", "YELLOW"} else 2


def planned_report(args: argparse.Namespace, *, started: float) -> dict[str, Any]:
    return {
        "policy": "project_theseus_practical_capability_acceleration_v1",
        "created_utc": now(),
        "trigger_state": "YELLOW",
        "execute": False,
        "next_command": (
            "python3 scripts/theseus_practical_capability_acceleration_v1.py --execute "
            f"--seed {int(args.seed)} --max-train-rows {int(args.max_train_rows)} "
            f"--max-eval-rows {int(args.max_eval_rows)} --epochs {int(args.epochs)}"
        ),
        "contract": acceleration_contract(args),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def run_acceleration(args: argparse.Namespace, *, started: float) -> dict[str, Any]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    steps: list[dict[str, Any]] = []

    steps.append(run_step("external_inference_audit_initial", [
        sys.executable, "scripts/external_inference_audit.py", "--no-scan-reports",
        "--out", rel(run_dir / "external_inference_audit_initial.json"),
    ], run_dir, timeout_seconds=240))

    steps.append(run_step("dogfood_trace_event", [
        sys.executable, "scripts/dogfood_trace_event.py", "--execute",
        "--surface", "codex_goal",
        "--assistant-lane", "long_horizon_planning",
        "--outcome", "accepted",
        "--intent-summary-redacted", "practical_capability_acceleration_loop_metadata_only",
        "--artifact-ref", rel(run_dir),
        "--artifact-ref", rel(resolve(args.targets)),
        "--duration-ms", "1",
        "--out", rel(run_dir / "dogfood_trace_event.json"),
        "--markdown-out", rel(run_dir / "dogfood_trace_event.md"),
    ], run_dir, timeout_seconds=120, allowed_returncodes={0, 2}))

    steps.append(run_step("dogfood_trace_training_bridge", [
        sys.executable, "scripts/dogfood_trace_training_bridge.py", "--execute",
        "--out", rel(run_dir / "dogfood_trace_training_bridge.json"),
        "--markdown-out", rel(run_dir / "dogfood_trace_training_bridge.md"),
    ], run_dir, timeout_seconds=240, allowed_returncodes={0, 2}))

    steps.append(run_step("dogfood_trace_bootstrap", [
        sys.executable, "scripts/dogfood_trace_bootstrap.py",
        "--out", rel(run_dir / "dogfood_trace_bootstrap.json"),
        "--markdown-out", rel(run_dir / "dogfood_trace_bootstrap.md"),
    ], run_dir, timeout_seconds=120))

    steps.append(run_step("teacher_distillation_manifest_builder", [
        sys.executable, "scripts/teacher_distillation_manifest_builder.py",
        "--manifest-out", rel(run_dir / "teacher_distillation_manifest.json"),
        "--ledger-out", rel(run_dir / "teacher_distillation_ledger.jsonl"),
        "--audit-out", rel(run_dir / "teacher_distillation_manifest_audit.json"),
        "--markdown-out", rel(run_dir / "teacher_distillation_manifest_audit.md"),
    ], run_dir, timeout_seconds=240, allowed_returncodes={0, 2}))

    steps.append(run_step("teacher_distillation_gate", [
        sys.executable, "scripts/teacher_distillation_gate.py",
        "--out", rel(run_dir / "teacher_distillation_gate.json"),
        "--markdown-out", rel(run_dir / "teacher_distillation_gate.md"),
    ], run_dir, timeout_seconds=180, allowed_returncodes={0, 2}))

    steps.append(run_step("training_data_admission", [
        sys.executable, "scripts/training_data_admission_v1.py",
        "--out", rel(run_dir / "training_data_admission_v1.json"),
        "--markdown-out", rel(run_dir / "training_data_admission_v1.md"),
        "--manifest-out", rel(run_dir / "training_data_admission_manifest_v1.jsonl"),
    ], run_dir, timeout_seconds=1200))

    steps.append(run_step("broad_capability_curriculum", [
        sys.executable, "scripts/broad_capability_curriculum_v1.py",
        "--admission", rel(run_dir / "training_data_admission_v1.json"),
        "--out", rel(run_dir / "broad_capability_curriculum_v1.json"),
        "--markdown-out", rel(run_dir / "broad_capability_curriculum_v1.md"),
        "--index-out", rel(run_dir / "broad_capability_curriculum_v1_index.jsonl"),
        "--training-sources-out", rel(run_dir / "broad_capability_curriculum_v1_training_sources.json"),
        "--max-rows-per-source", "512",
    ], run_dir, timeout_seconds=300))

    steps.append(run_step("vcm_task_context_bridge", [
        sys.executable, "scripts/vcm_task_context_bridge.py",
        "--out", rel(run_dir / "vcm_task_context_bridge.json"),
        "--markdown-out", rel(run_dir / "vcm_task_context_bridge.md"),
        "--contexts-out", rel(run_dir / "vcm_task_contexts.json"),
    ], run_dir, timeout_seconds=240))

    steps.append(run_step("private_residual_target_consumer", [
        sys.executable, "scripts/private_residual_target_consumer_v1.py",
        "--targets", rel(resolve(args.targets)),
        "--out", rel(run_dir / "private_residual_target_consumer_v1.json"),
        "--markdown-out", rel(run_dir / "private_residual_target_consumer_v1.md"),
        "--queue-out", rel(run_dir / "private_residual_repair_queue_v1.jsonl"),
    ], run_dir, timeout_seconds=240))

    common_broad_args = [
        "--admission", rel(run_dir / "training_data_admission_v1.json"),
        "--curriculum", rel(run_dir / "broad_capability_curriculum_v1.json"),
        "--vcm-contexts", rel(run_dir / "vcm_task_contexts.json"),
        "--max-train-rows", str(max(1, int(args.max_train_rows))),
        "--max-eval-rows", str(max(1, int(args.max_eval_rows))),
        "--epochs", str(max(1, int(args.epochs))),
        "--seed", str(int(args.seed)),
        "--execute",
    ]
    for mode in ["off", "on"]:
        prefix = f"broad_survival_vcm_{mode}"
        steps.append(run_step(prefix, [
            sys.executable, "scripts/broad_capability_survival_lane_run_v1.py",
            *common_broad_args,
            "--vcm-mode", mode,
            "--out", rel(run_dir / f"{prefix}.json"),
            "--markdown-out", rel(run_dir / f"{prefix}.md"),
            "--train-out", rel(run_dir / f"{prefix}_train.jsonl"),
            "--eval-out", rel(run_dir / f"{prefix}_eval.jsonl"),
            "--config-out", rel(run_dir / f"{prefix}_config.json"),
            "--candidate-manifest-out", rel(run_dir / f"{prefix}_candidates.jsonl"),
            "--comparator-out", rel(run_dir / f"{prefix}_comparator.json"),
            "--comparator-markdown-out", rel(run_dir / f"{prefix}_comparator.md"),
        ], run_dir, timeout_seconds=5400))

    steps.append(run_step("broad_vcm_feature_ablation", [
        sys.executable, "scripts/broad_capability_vcm_feature_ablation_v1.py",
        "--vcm-on", rel(run_dir / "broad_survival_vcm_on.json"),
        "--vcm-off", rel(run_dir / "broad_survival_vcm_off.json"),
        "--out", rel(run_dir / "broad_capability_vcm_feature_ablation_v1.json"),
        "--markdown-out", rel(run_dir / "broad_capability_vcm_feature_ablation_v1.md"),
        "--policy-out", rel(run_dir / "vcm_broad_survival_feature_policy_v1.json"),
    ], run_dir, timeout_seconds=180, allowed_returncodes={0, 2}))

    chosen_broad = choose_broad_run(run_dir)

    steps.append(run_step("broad_survival_decision", [
        sys.executable, "scripts/broad_capability_survival_lane_decision_v1.py",
        "--admission", rel(run_dir / "training_data_admission_v1.json"),
        "--curriculum", rel(run_dir / "broad_capability_curriculum_v1.json"),
        "--vcm", rel(run_dir / "vcm_task_context_bridge.json"),
        "--vcm-ablation", rel(run_dir / "broad_capability_vcm_feature_ablation_v1.json"),
        "--vcm-policy", rel(run_dir / "vcm_broad_survival_feature_policy_v1.json"),
        "--broad-run", rel(chosen_broad),
        "--out", rel(run_dir / "broad_capability_survival_lane_decision_v1.json"),
        "--markdown-out", rel(run_dir / "broad_capability_survival_lane_decision_v1.md"),
    ], run_dir, timeout_seconds=240, allowed_returncodes={0, 2}))

    if platform.system() == "Darwin":
        steps.append(run_step("macos_mlx_environment_diagnosis", [
            sys.executable, "scripts/macos_mlx_environment_diagnosis.py",
            "--out", rel(run_dir / "macos_mlx_environment_diagnosis.json"),
            "--markdown-out", rel(run_dir / "macos_mlx_environment_diagnosis.md"),
        ], run_dir, timeout_seconds=240, allowed_returncodes={0, 1, 2}))
        steps.append(run_step("macos_metal_production_route_readiness", [
            sys.executable, "scripts/macos_metal_production_route_readiness.py",
            "--out", rel(run_dir / "macos_metal_production_route_readiness.json"),
            "--markdown-out", rel(run_dir / "macos_metal_production_route_readiness.md"),
        ], run_dir, timeout_seconds=240, allowed_returncodes={0, 1, 2}))

    steps.append(run_step("vcm_native_runtime_probe", [
        sys.executable, "scripts/vcm_native_runtime_probe.py",
        "--out", rel(run_dir / "vcm_native_runtime_probe.json"),
        "--markdown-out", rel(run_dir / "vcm_native_runtime_probe.md"),
        "--descriptors-out", rel(run_dir / "vcm_native_runtime_descriptors.jsonl"),
    ], run_dir, timeout_seconds=300, allowed_returncodes={0, 1, 2}))

    steps.append(run_step("maturity_integrity_audit", [
        sys.executable, "scripts/maturity_integrity_audit.py",
        "--student-candidate-manifest", rel(chosen_candidate_manifest(chosen_broad)),
        "--sts-ranker-policy", "reports/sts_ranker_policy_v1.json",
        "--out", rel(run_dir / "maturity_integrity_audit.json"),
        "--markdown-out", rel(run_dir / "maturity_integrity_audit.md"),
    ], run_dir, timeout_seconds=300, allowed_returncodes={0, 1, 2}))

    steps.append(run_step("public_transfer_readiness_refresh", [
        sys.executable, "scripts/public_transfer_readiness_refresh_v1.py",
        "--maturity-audit", rel(run_dir / "maturity_integrity_audit.json"),
        "--full-body-semantic-ablation", "reports/composition_contract_blind_repair_v1_sts_vcm_ablation.json",
        "--public-readiness-packet", "reports/bounded_public_transfer_calibration_review_v1_readiness_packet.json",
        "--operator-dry-run", "reports/bounded_public_transfer_calibration_review_v1_guarded_dry_run.json",
        "--operator-lock", "reports/public_calibration_operator_lock.flag",
        "--out", rel(run_dir / "public_transfer_readiness_refresh_v1.json"),
        "--markdown-out", rel(run_dir / "public_transfer_readiness_refresh_v1.md"),
    ], run_dir, timeout_seconds=300, allowed_returncodes={0, 1, 2}))

    steps.append(run_step("candidate_promotion_gate", [
        sys.executable, "scripts/candidate_promotion_gate.py",
        "--maturity-integrity-audit", rel(run_dir / "maturity_integrity_audit.json"),
        "--real-code-graduation", "reports/real_code_benchmark_graduation_industry_code_transfer_seed14_5x64_v1.json",
        "--broad-transfer-matrix", "reports/broad_transfer_matrix_industry_code_transfer_seed14_5x64_v1.json",
        "--out", rel(run_dir / "candidate_promotion_gate.json"),
    ], run_dir, timeout_seconds=300, allowed_returncodes={0, 1, 2}))

    steps.append(run_step("model_growth_gate", [
        sys.executable, "scripts/model_growth_gate.py",
        "--out", rel(run_dir / "model_growth_gate.json"),
    ], run_dir, timeout_seconds=240, allowed_returncodes={0, 1, 2}))

    steps.append(run_step("artifact_retention_dry_run", [
        sys.executable, "scripts/theseus_artifact_retention.py",
        "--max-files", "40",
        "--min-bytes", "10485760",
        "--include-jsonl",
        "--include-vcm-payloads",
        "--out", rel(run_dir / "artifact_retention.json"),
        "--markdown-out", rel(run_dir / "artifact_retention.md"),
        "--manifest-out", rel(run_dir / "artifact_retention_manifest.jsonl"),
    ], run_dir, timeout_seconds=240, allowed_returncodes={0, 1, 2}))

    report = summarize_run(args, run_dir, chosen_broad, steps, started)
    write_json(run_dir / "final_report.json", report)
    write_text(run_dir / "final_report.md", render_markdown(report))
    return report


def choose_broad_run(run_dir: Path) -> Path:
    on = read_json(run_dir / "broad_survival_vcm_on.json")
    off = read_json(run_dir / "broad_survival_vcm_off.json")
    on_rate = number(get_path(on, ["summary", "transformer_sts_on_pass_rate"], 0.0))
    off_rate = number(get_path(off, ["summary", "transformer_sts_on_pass_rate"], 0.0))
    return run_dir / ("broad_survival_vcm_on.json" if on_rate >= off_rate else "broad_survival_vcm_off.json")


def chosen_candidate_manifest(broad_run_path: Path) -> Path:
    report = read_json(broad_run_path)
    path = str(get_path(report, ["summary", "candidate_manifest"], "") or "")
    return resolve(path) if path else broad_run_path.parent / "broad_survival_vcm_off_candidates.jsonl"


def summarize_run(
    args: argparse.Namespace,
    run_dir: Path,
    chosen_broad: Path,
    steps: list[dict[str, Any]],
    started: float,
) -> dict[str, Any]:
    residual = read_json(run_dir / "private_residual_target_consumer_v1.json")
    dogfood_bootstrap = read_json(run_dir / "dogfood_trace_bootstrap.json")
    dogfood_bridge = read_json(run_dir / "dogfood_trace_training_bridge.json")
    teacher_gate = read_json(run_dir / "teacher_distillation_gate.json")
    admission = read_json(run_dir / "training_data_admission_v1.json")
    curriculum = read_json(run_dir / "broad_capability_curriculum_v1.json")
    vcm_bridge = read_json(run_dir / "vcm_task_context_bridge.json")
    broad_on = read_json(run_dir / "broad_survival_vcm_on.json")
    broad_off = read_json(run_dir / "broad_survival_vcm_off.json")
    broad_ablation = read_json(run_dir / "broad_capability_vcm_feature_ablation_v1.json")
    broad = read_json(chosen_broad)
    decision = read_json(run_dir / "broad_capability_survival_lane_decision_v1.json")
    mlx = read_json(run_dir / "macos_mlx_environment_diagnosis.json")
    metal = read_json(run_dir / "macos_metal_production_route_readiness.json")
    vcm_runtime = read_json(run_dir / "vcm_native_runtime_probe.json")
    maturity = read_json(run_dir / "maturity_integrity_audit.json")
    transfer = read_json(run_dir / "public_transfer_readiness_refresh_v1.json")
    public_packet = read_json(REPORTS / "public_calibration_readiness_packet.json")
    promotion = read_json(run_dir / "candidate_promotion_gate.json")
    growth = read_json(run_dir / "model_growth_gate.json")
    retention = read_json(run_dir / "artifact_retention.json")

    failed_steps = [step for step in steps if not step.get("ok")]
    no_cheat = no_cheat_summary(
        residual,
        dogfood_bootstrap,
        dogfood_bridge,
        admission,
        vcm_bridge,
        broad_on,
        broad_off,
        broad,
        vcm_runtime,
    )
    hard_violation = {key: value for key, value in no_cheat.items() if key.endswith("_violation") and value}
    summary = {
        "run_dir": rel(run_dir),
        "residual_target_state": residual.get("trigger_state"),
        "residual_targets": get_path(residual, ["summary", "target_rows"], 0),
        "residual_targets_covered": get_path(residual, ["summary", "covered_target_count"], 0),
        "residual_targets_unresolved": get_path(residual, ["summary", "unresolved_target_count"], 0),
        "private_selected_pass_rate": get_path(residual, ["summary", "private_selected_pass_rate"], None),
        "private_pass_if_any_rate": get_path(residual, ["summary", "private_pass_if_any_rate"], None),
        "private_no_admissible_task_rate": get_path(residual, ["summary", "private_no_admissible_task_rate"], None),
        "dogfood_event_count": get_path(dogfood_bootstrap, ["summary", "event_count"], 0),
        "dogfood_training_row_count": get_path(dogfood_bootstrap, ["summary", "training_row_count"], 0),
        "dogfood_bridge_rows_written": dogfood_bridge.get("training_rows_written", 0),
        "teacher_distillation_allowed": get_path(teacher_gate, ["summary", "distillation_allowed"], False),
        "teacher_accepted_rows": get_path(teacher_gate, ["summary", "teacher_accepted_rows"], 0),
        "teacher_accepted_row_share": get_path(teacher_gate, ["summary", "teacher_accepted_row_share"], 0.0),
        "admitted_training_source_count": get_path(admission, ["summary", "allowed_training_source_count"], 0),
        "admitted_open_public_row_count": get_path(admission, ["summary", "admitted_open_public_row_count"], 0),
        "public_benchmark_payload_admitted": get_path(admission, ["summary", "public_benchmark_payload_admitted"], False),
        "curriculum_unit_count": get_path(curriculum, ["summary", "curriculum_unit_count"], 0),
        "vcm_bridge_state": vcm_bridge.get("trigger_state"),
        "vcm_ready_task_family_count": get_path(vcm_bridge, ["summary", "ready_task_family_count"], 0),
        "vcm_on_transformer_rate": get_path(broad_on, ["summary", "transformer_sts_on_pass_rate"], None),
        "vcm_off_transformer_rate": get_path(broad_off, ["summary", "transformer_sts_on_pass_rate"], None),
        "vcm_feature_action": get_path(broad_ablation, ["summary", "recommended_action"], ""),
        "chosen_broad_run": rel(chosen_broad),
        "architecture_winner": get_path(broad, ["summary", "winner_by_sts_on"], ""),
        "transformer_sts_on_pass_rate": get_path(broad, ["summary", "transformer_sts_on_pass_rate"], None),
        "symliquid_sts_on_pass_rate": get_path(broad, ["summary", "symliquid_sts_on_pass_rate"], None),
        "symliquid_minus_transformer": get_path(broad, ["summary", "symliquid_minus_transformer"], None),
        "survival_lane_decision": decision.get("decision"),
        "broad_survival_promotion_ready": decision.get("promotion_ready"),
        "broad_survival_promoted": get_path(decision, ["promotion_summary", "broad_survival_promoted"], False),
        "broad_survival_promotion_scope": get_path(
            decision,
            ["promotion_summary", "broad_survival_promotion_scope"],
            "",
        ),
        "mlx_state": mlx.get("trigger_state"),
        "mlx_recommended_python": get_path(mlx, ["summary", "recommended_python"], ""),
        "metal_state": metal.get("trigger_state"),
        "metal_production_route_allowed": get_path(metal, ["summary", "production_route_allowed"], False),
        "vcm_runtime_state": vcm_runtime.get("trigger_state"),
        "vcm_native_runtime_claimable": get_path(vcm_runtime, ["summary", "native_runtime_claimable"], False),
        "maturity_state": maturity.get("trigger_state"),
        "maturity_blockers": get_path(maturity, ["summary", "maturity_blockers"], []),
        "public_transfer_state": transfer.get("trigger_state"),
        "public_transfer_ready_for_one_new_public_calibration": transfer.get("ready_for_one_new_public_calibration"),
        "public_transfer_private_fix_review_ready": get_path(transfer, ["summary", "private_fix_review_ready"], False),
        "public_transfer_calibration_review_ready_after_private_fixes": get_path(
            transfer,
            ["summary", "calibration_review_ready_after_private_fixes"],
            False,
        ),
        "public_transfer_contract_task_count": get_path(transfer, ["summary", "contract_task_count"], 0),
        "public_transfer_latest_public_pass_rate": get_path(transfer, ["summary", "latest_public_pass_rate"], None),
        "public_transfer_latest_public_task_count": get_path(transfer, ["summary", "latest_public_task_count"], None),
        "public_transfer_failed_gate_count": get_path(transfer, ["summary", "transfer_failed_gate_count"], 0),
        "public_calibration_packet_state": public_packet.get("trigger_state"),
        "public_calibration_packet_technical_ready": public_packet.get(
            "technical_ready_for_one_bounded_public_calibration"
        ),
        "public_calibration_packet_operator_lock_active": public_packet.get("operator_lock_active"),
        "public_calibration_packet_frozen_integrity_sha256": get_path(
            public_packet,
            ["summary", "frozen_integrity_sha256"],
            "",
        ),
        "legacy_candidate_promotion_allowed": promotion.get("promote"),
        "candidate_promotion_allowed": bool(
            promotion.get("promote") is True or decision.get("promotion_ready") is True
        ),
        "model_growth_allowed": growth.get("model_growth_allowed"),
        "artifact_retention_dry_run_gib": get_path(retention, ["summary", "dry_run_candidate_gib"], 0.0),
        "no_cheat": no_cheat,
    }
    trigger_state = "RED" if failed_steps or hard_violation else "GREEN"
    if trigger_state == "GREEN" and (
        summary["candidate_promotion_allowed"] is not True
        or summary["model_growth_allowed"] is not True
        or summary["public_transfer_state"] != "GREEN"
    ):
        trigger_state = "YELLOW"
    return {
        "policy": "project_theseus_practical_capability_acceleration_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "execute": True,
        "contract": acceleration_contract(args),
        "run_dir": rel(run_dir),
        "summary": summary,
        "architecture_recommendation": architecture_recommendation(summary),
        "remaining_blockers": remaining_blockers(summary),
        "failed_steps": [
            {
                "label": step.get("label"),
                "returncode": step.get("returncode"),
                "stdout_tail": step.get("stdout_tail"),
                "stderr_tail": step.get("stderr_tail"),
            }
            for step in failed_steps
        ],
        "hard_violation": hard_violation or None,
        "steps": steps,
        "artifacts": {
            "run_dir": rel(run_dir),
            "private_residual_target_consumer": rel(run_dir / "private_residual_target_consumer_v1.json"),
            "dogfood_bootstrap": rel(run_dir / "dogfood_trace_bootstrap.json"),
            "dogfood_training_bridge": rel(run_dir / "dogfood_trace_training_bridge.json"),
            "teacher_gate": rel(run_dir / "teacher_distillation_gate.json"),
            "training_admission": rel(run_dir / "training_data_admission_v1.json"),
            "vcm_bridge": rel(run_dir / "vcm_task_context_bridge.json"),
            "broad_vcm_on": rel(run_dir / "broad_survival_vcm_on.json"),
            "broad_vcm_off": rel(run_dir / "broad_survival_vcm_off.json"),
            "broad_vcm_ablation": rel(run_dir / "broad_capability_vcm_feature_ablation_v1.json"),
            "survival_decision": rel(run_dir / "broad_capability_survival_lane_decision_v1.json"),
            "maturity": rel(run_dir / "maturity_integrity_audit.json"),
            "public_transfer_readiness": rel(run_dir / "public_transfer_readiness_refresh_v1.json"),
            "final_report": rel(run_dir / "final_report.json"),
        },
        "score_semantics": (
            "Practical local capability acceleration loop. Public benchmark payloads remain calibration-only; "
            "this run does not execute public calibration, call live teacher inference, serve external tokens, "
            "or count fallback returns."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def summarize_existing_run(args: argparse.Namespace, run_dir: Path, *, started: float) -> dict[str, Any]:
    chosen_broad = choose_broad_run(run_dir)
    report = summarize_run(args, run_dir, chosen_broad, [], started)
    report["resumed_existing_run"] = True
    report["runtime_ms"] = int((time.perf_counter() - started) * 1000)
    write_json(run_dir / "final_report.json", report)
    write_text(run_dir / "final_report.md", render_markdown(report))
    return report


def no_cheat_summary(*reports: dict[str, Any]) -> dict[str, Any]:
    public_training = sum(summary_number(report, "public_training_rows") for report in reports)
    public_training += sum(summary_number(report, "public_training_rows_written") for report in reports)
    public_training += sum(summary_number(report, "public_benchmark_training_rows") for report in reports)
    external = sum(summary_number(report, "external_inference_calls") for report in reports)
    fallback = sum(summary_number(report, "fallback_return_count") for report in reports)
    fallback += sum(summary_number(report, "fallback_returns") for report in reports)
    fallback += sum(summary_number(report, "fallback_return_rows") for report in reports)
    text = json.dumps(reports, sort_keys=True).lower()
    teacher_runtime = '"raw_teacher_outputs_to_user": "allowed"' in text or '"runtime_serving_forbidden": false' in text
    raw_user_text = '"raw_user_text_included": true' in text or '"raw_text_capture_enabled": true' in text
    return {
        "public_training_rows": public_training,
        "external_inference_calls": external,
        "fallback_return_count": fallback,
        "teacher_runtime_serving_violation": teacher_runtime,
        "raw_user_text_violation": raw_user_text,
        "public_training_violation": public_training != 0,
        "external_inference_violation": external != 0,
        "fallback_return_violation": fallback != 0,
    }


def architecture_recommendation(summary: dict[str, Any]) -> dict[str, Any]:
    winner = str(summary.get("architecture_winner") or "")
    delta = number(summary.get("symliquid_minus_transformer"))
    if summary.get("broad_survival_promotion_ready") is True:
        if winner == "symliquid_style" and delta > 0:
            return {
                "survival_lane": "promoted_transformer_structural_student",
                "symliquid_role": "repeat_matched_full_body_candidate_before_hot_path",
                "reason": (
                    "A transformer structural-action survival artifact is already privately promoted; "
                    "SymLiquid won this body-template comparator slice, so it should get repeat/full-body "
                    "evidence before changing the hot path."
                ),
            }
        return {
            "survival_lane": "promoted_transformer_structural_student",
            "symliquid_role": "bounded_discovery_comparator",
            "reason": "The broad structural survival promotion gate is green; keep the promoted transformer structural artifact as the practical hot path.",
        }
    if winner == "transformer_control" and delta < 0:
        return {
            "survival_lane": "transformer_hybrid_structural_student",
            "symliquid_role": "bounded_discovery_comparator_only",
            "reason": "Transformer control wins the chosen broad private matched comparator; SymLiquid remains useful only where matched evidence shows complementarity.",
        }
    if winner == "symliquid_style" and delta > 0:
        return {
            "survival_lane": "symliquid_candidate_pending_repeat",
            "symliquid_role": "candidate_survival_lane_after_repeat_and_gate",
            "reason": "SymLiquid wins this matched slice; repeat and gate before making it the hot path.",
        }
    return {
        "survival_lane": "hybrid_pending_more_evidence",
        "symliquid_role": "bounded_discovery_comparator",
        "reason": "Current comparison is tied or inconclusive; keep practical route on the most stable verified path.",
    }


def remaining_blockers(summary: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if summary.get("candidate_promotion_allowed") is not True:
        blockers.append("candidate_promotion_still_blocked")
    if summary.get("model_growth_allowed") is not True:
        blockers.append("model_growth_still_blocked")
    if summary.get("public_transfer_state") != "GREEN":
        if (
            summary.get("public_calibration_packet_technical_ready") is True
            and summary.get("public_calibration_packet_operator_lock_active") is True
        ):
            blockers.append("public_calibration_locked_after_private_readiness_green")
        else:
            blockers.append("public_transfer_floor_or_lock_still_blocks_public_claims")
    if summary.get("public_benchmark_payload_admitted"):
        blockers.append("public_payload_admission_violation")
    if number(summary.get("artifact_retention_dry_run_gib")) > 0.0:
        blockers.append("report_artifact_retention_execute_pending")
    if summary.get("vcm_native_runtime_claimable") is not True:
        blockers.append("vcm_native_runtime_not_claimable")
    return blockers


def acceleration_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "private_residual_targets": rel(resolve(args.targets)),
        "public_calibration": "never executed by this loop",
        "public_benchmark_training": "forbidden",
        "teacher": "manifest/gate only; no live teacher call from this loop",
        "dogfood": "redacted metadata only; raw text disabled",
        "architecture": "survival lane follows matched evidence; SymLiquid kept as discovery comparator if behind",
        "vcm": "default context substrate, ablated where consumed",
        "mac": "MLX/Metal readiness probed, no parity claim without exact evidence",
        "fallback_returns": "forbidden",
    }


def run_step(
    label: str,
    command: list[str],
    run_dir: Path,
    *,
    timeout_seconds: int,
    allowed_returncodes: set[int] | None = None,
) -> dict[str, Any]:
    allowed = allowed_returncodes or {0}
    started = time.perf_counter()
    stdout_path = run_dir / f"{label}.stdout.txt"
    stderr_path = run_dir / f"{label}.stderr.txt"
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=max(1, int(timeout_seconds)),
        )
        stdout_path.write_text((result.stdout or "")[-20000:], encoding="utf-8")
        stderr_path.write_text((result.stderr or "")[-20000:], encoding="utf-8")
        return {
            "label": label,
            "command": command,
            "ok": result.returncode in allowed,
            "returncode": result.returncode,
            "allowed_returncodes": sorted(allowed),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_log": rel(stdout_path),
            "stderr_log": rel(stderr_path),
            "stdout_tail": (result.stdout or "")[-1200:],
            "stderr_tail": (result.stderr or "")[-1200:],
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else str(exc)
        stdout_path.write_text(stdout[-20000:], encoding="utf-8")
        stderr_path.write_text(stderr[-20000:], encoding="utf-8")
        return {
            "label": label,
            "command": command,
            "ok": False,
            "returncode": 124,
            "allowed_returncodes": sorted(allowed),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_log": rel(stdout_path),
            "stderr_log": rel(stderr_path),
            "stdout_tail": stdout[-1200:],
            "stderr_tail": stderr[-1200:],
        }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# Theseus Practical Capability Acceleration v1",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- execute: `{report.get('execute')}`",
        f"- run_dir: `{report.get('run_dir')}`",
        f"- residual targets covered/unresolved: `{summary.get('residual_targets_covered')}` / `{summary.get('residual_targets_unresolved')}`",
        f"- private selected/pass-if-any/no-admissible: `{summary.get('private_selected_pass_rate')}` / `{summary.get('private_pass_if_any_rate')}` / `{summary.get('private_no_admissible_task_rate')}`",
        f"- dogfood events/training rows: `{summary.get('dogfood_event_count')}` / `{summary.get('dogfood_training_row_count')}`",
        f"- teacher accepted rows/share: `{summary.get('teacher_accepted_rows')}` / `{summary.get('teacher_accepted_row_share')}`",
        f"- architecture winner: `{summary.get('architecture_winner')}`",
        f"- transformer vs SymLiquid: `{summary.get('transformer_sts_on_pass_rate')}` / `{summary.get('symliquid_sts_on_pass_rate')}`",
        f"- VCM on/off transformer: `{summary.get('vcm_on_transformer_rate')}` / `{summary.get('vcm_off_transformer_rate')}`",
        f"- VCM action: `{summary.get('vcm_feature_action')}`",
        f"- MLX/Metal: `{summary.get('mlx_state')}` / `{summary.get('metal_state')}`",
        f"- promotion/growth: `{summary.get('candidate_promotion_allowed')}` / `{summary.get('model_growth_allowed')}`",
        f"- public transfer: `{summary.get('public_transfer_state')}`",
        f"- public review packet: `{summary.get('public_calibration_packet_state')}` technical_ready=`{summary.get('public_calibration_packet_technical_ready')}` lock_active=`{summary.get('public_calibration_packet_operator_lock_active')}`",
        f"- next public contract tasks: `{summary.get('public_transfer_contract_task_count')}` latest locked public: `{summary.get('public_transfer_latest_public_pass_rate')}` / `{summary.get('public_transfer_latest_public_task_count')}`",
        f"- no-cheat: `{summary.get('no_cheat')}`",
        "",
        "## Architecture Recommendation",
        "",
        json.dumps(report.get("architecture_recommendation", {}), indent=2, sort_keys=True),
        "",
        "## Remaining Blockers",
        "",
    ]
    blockers = report.get("remaining_blockers") if isinstance(report.get("remaining_blockers"), list) else []
    if blockers:
        lines.extend(f"- `{item}`" for item in blockers)
    else:
        lines.append("- none")
    if report.get("failed_steps"):
        lines.extend(["", "## Failed Steps", ""])
        for step in report["failed_steps"]:
            lines.append(f"- `{step.get('label')}` returncode=`{step.get('returncode')}`")
    return "\n".join(lines).rstrip() + "\n"


def compact_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": report.get("policy"),
        "trigger_state": report.get("trigger_state"),
        "execute": report.get("execute"),
        "run_dir": report.get("run_dir"),
        "summary": report.get("summary"),
        "architecture_recommendation": report.get("architecture_recommendation"),
        "remaining_blockers": report.get("remaining_blockers"),
        "failed_steps": report.get("failed_steps"),
        "hard_violation": report.get("hard_violation"),
        "external_inference_calls": report.get("external_inference_calls", 0),
    }


def summary_number(report: dict[str, Any], key: str) -> int:
    if not isinstance(report, dict):
        return 0
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return int(number(summary.get(key)))


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: str | Path) -> str:
    value = Path(path)
    try:
        return str(value.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(value).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
