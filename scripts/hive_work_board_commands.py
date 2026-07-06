"""Command routing and safety guards for Hive work board tasks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hive_work_board_command_guards import code_contract_preflight_command, code_contract_preflight_guard, code_lm_training_command_gate, decoder_relevant_source_mtime, execution_shape_no_template_smoke_command, execution_shape_no_template_smoke_guard, private_pressure_private_closure_needed, private_public_calibration_guard
from progress_integrity_policy import (
    non_promotable_diagnostic_reason,
    promotion_safe_replacement_concept,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FOUR_CARD_RECEIVER_SLUG = "source_mbpp_source_evalplus_source_bigcodebench_source_livecodebench_seed14_32"
CODE_CONTRACT_PREFLIGHT_REPORT = REPORTS / "code_lm_closure_public_contract_preflight_seed23_32.json"
EXECUTION_SHAPE_NO_TEMPLATE_SMOKE_REPORT = REPORTS / "execution_shape_private_ablation_smoke.json"
PRIVATE_PRESSURE_TRAIN_ONCE_CURRENT_SLUG = "frontier_private_transfer_private_only_train_once_v1"
PRIVATE_PRESSURE_TRAIN_ONCE_LEGACY_SLUG = "private_pressure_private_recovery_train_once_fanout_v1"
DECODER_RELEVANT_SOURCES = (
    ROOT / "crates" / "symliquid-cli" / "src" / "code_lm_closure.rs",
    ROOT / "scripts" / "code_lm_closure.py",
    ROOT / "scripts" / "code_residual_curriculum.py",
    ROOT / "scripts" / "type_contract_diagnostic.py",
)
CODE_HIGH_TRANSFER_CONCEPTS = {
    "type_contract_diagnostic",
    "type_contract_four_card_calibration",
    "edge_exec_repair_four_card_calibration",
    "execution_shaped_four_card_calibration",
    "execution_shape_private_ablation",
    "type_and_return_shape",
    "typed_interface_skeleton",
    "typed_interface_private_closure",
    "edge_contract_4card",
    "edge_contract_private_closure",
    "edge_contract_balanced_4card_private_curriculum_v2",
    "edge_contract_balanced_private_closure_v2",
    "edge_case_full_body_private_curriculum_v1",
    "edge_case_full_body_private_closure_v1",
    "edge_contract_v2_private_residual_curriculum",
    "edge_contract_v2_private_closure",
    "candidate_floor_v2_private_residual_curriculum",
    "candidate_floor_v2_private_closure",
    "residual_targeted_private_edge_case_contract_v1",
    "decoder_v2_private_ablation_gate",
    "private_type_shape_receiver_veto_ablation",
    "private_pressure_private_closure",
    "admissibility_and_interface",
    "edge_conditions",
    "algorithmic_planning",
    "execution_shaped_programs",
    "private_pressure_four_card_recalibration",
}


def safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._-")
    return text or "item"


def get_path(data: Any, path: list[Any], default: Any = None) -> Any:
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def private_pressure_private_closure_state_for_task(task: dict[str, Any]) -> dict[str, Any]:
    state = get_path(task, ["evidence", "private_pressure_private_closure_state"], {})
    if isinstance(state, dict) and state.get("reason"):
        return state
    scheduler = read_json(REPORTS / "high_transfer_curriculum_scheduler.json", {})
    concepts = scheduler.get("concepts") if isinstance(scheduler.get("concepts"), list) else []
    for row in concepts:
        if not isinstance(row, dict):
            continue
        if str(row.get("concept") or "").lower() != "private_pressure_private_closure":
            continue
        state = get_path(row, ["evidence", "private_pressure_private_closure_state"], {})
        if isinstance(state, dict):
            return state
    return {}


def private_pressure_edge_obligation_gate_needed(task: dict[str, Any]) -> bool:
    state = private_pressure_private_closure_state_for_task(task)
    if not isinstance(state, dict):
        return False
    reason = str(state.get("reason") or "")
    if reason not in {
        "edge_obligation_decode_gate_required_for_private_pressure_closure",
        "edge_obligation_decode_gate_stale_for_private_pressure_closure",
    }:
        return False
    report = REPORTS / "edge_obligation_decode_gate_v1_private_pressure_private.json"
    payload = read_json(report, {})
    report_mtime = report.stat().st_mtime if report.exists() else 0.0
    closure_mtime = float(state.get("closure_report_mtime") or 0.0)
    ready = bool(
        payload.get("policy") == "project_theseus_edge_obligation_decode_gate_v1"
        and payload.get("ready_for_public_calibration")
        and payload.get("trigger_state") == "GREEN"
    )
    current = bool(report.exists() and report_mtime >= closure_mtime)
    return not current


def private_pressure_edge_obligation_gate_failed_current(task: dict[str, Any]) -> dict[str, Any]:
    state = private_pressure_private_closure_state_for_task(task)
    reason = str(state.get("reason") or "") if isinstance(state, dict) else ""
    if reason not in {
        "edge_obligation_decode_gate_required_for_private_pressure_closure",
        "edge_obligation_decode_gate_stale_for_private_pressure_closure",
    }:
        return {"blocked": False}
    report = REPORTS / "edge_obligation_decode_gate_v1_private_pressure_private.json"
    payload = read_json(report, {})
    report_mtime = report.stat().st_mtime if report.exists() else 0.0
    closure_mtime = float(state.get("closure_report_mtime") or 0.0) if isinstance(state, dict) else 0.0
    current = bool(report.exists() and report_mtime >= closure_mtime)
    ready = bool(
        payload.get("policy") == "project_theseus_edge_obligation_decode_gate_v1"
        and payload.get("ready_for_public_calibration")
        and payload.get("trigger_state") == "GREEN"
    )
    if not current or ready:
        return {"blocked": False}
    return {
        "blocked": True,
        "reason": "edge_obligation_private_gate_failed_current_requires_architecture_patch",
        "report": rel(report),
        "trigger_state": payload.get("trigger_state"),
        "ready_for_public_calibration": payload.get("ready_for_public_calibration"),
        "summary": payload.get("summary") if isinstance(payload.get("summary"), dict) else {},
        "policy": "do_not_repeat_same_private_gate_or_full_closure_after_fresh_edge_obligation_failure",
    }


def private_pressure_edge_obligation_gate_command(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "allowed": True,
        "reason": "private_pressure_edge_obligation_gate_before_retraining",
        "evidence_paths": [
            "reports/edge_obligation_decode_gate_v1_private_pressure_private.json",
            "reports/edge_obligation_decode_gate_v1_private_pressure_private.md",
            f"data/private_code_curriculum/code_lm_closure_{PRIVATE_PRESSURE_TRAIN_ONCE_CURRENT_SLUG}.jsonl",
            f"reports/code_lm_private_candidates_{PRIVATE_PRESSURE_TRAIN_ONCE_CURRENT_SLUG}.jsonl",
            f"reports/code_lm_closure_{PRIVATE_PRESSURE_TRAIN_ONCE_CURRENT_SLUG}.json",
        ],
        "command": [
            sys.executable,
            "scripts/edge_obligation_decode_gate_v1.py",
            "--private-curriculum",
            f"data/private_code_curriculum/code_lm_closure_{PRIVATE_PRESSURE_TRAIN_ONCE_CURRENT_SLUG}.jsonl",
            "--private-candidates",
            f"reports/code_lm_private_candidates_{PRIVATE_PRESSURE_TRAIN_ONCE_CURRENT_SLUG}.jsonl",
            "--closure-report",
            f"reports/code_lm_closure_{PRIVATE_PRESSURE_TRAIN_ONCE_CURRENT_SLUG}.json",
            "--out",
            "reports/edge_obligation_decode_gate_v1_private_pressure_private.json",
            "--markdown-out",
            "reports/edge_obligation_decode_gate_v1_private_pressure_private.md",
            "--min-heldout-tasks",
            "64",
            "--max-exec-tasks",
            "128",
            "--exec-timeout-seconds",
            str(max(1, min(int(args.timeout_seconds), 6))),
        ],
        "hook_target": "training_launch",
    }














def cross_domain_capsule_command() -> list[str]:
    return [
        sys.executable,
        "scripts/cross_domain_sts_capsules.py",
        "--out",
        "reports/cross_domain_sts_capsules.json",
        "--markdown-out",
        "reports/cross_domain_sts_capsules.md",
        "--capsules-out",
        "data/training_sources/cross_domain_sts_capsules.jsonl",
        "--sts-out",
        "D:/ProjectTheseus/training_data/cross_domain_sts/cross_domain_sts_streams.jsonl",
    ]

def command_for_task(task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    source = str(task.get("source") or "")
    kind = str(task.get("kind") or "")
    task_id = str(task.get("task_id") or "")
    if source == "feedback_action_queue" and task_id.startswith("viea_action_"):
        evidence_paths = ["reports/viea_action_executor.json"]
        card_id = str(get_path(task, ["evidence", "card_id"], "") or "")
        if kind == "expand_public_adapter_clean_slice" and card_id:
            evidence_paths.append(f"reports/broad_transfer_closure_runner_{safe_name(card_id)}.json")
            evidence_paths.append("reports/broad_transfer_matrix.json")
        elif kind == "run_same_seed_sts_repair_ablation":
            evidence_paths.append("reports/sts_repair_ablation.json")
        command = [
            sys.executable,
            "scripts/viea_action_executor.py",
            "--execute",
            "--resume",
            "--only-action-id",
            task_id,
            "--max-actions",
            "1",
            "--max-steps",
            str(max(1, int(args.max_steps))),
            "--timeout-seconds",
            str(max(60, int(args.timeout_seconds))),
            "--out",
            "reports/viea_action_executor.json",
            "--markdown-out",
            "reports/viea_action_executor.md",
        ]
        if args.allow_teacher:
            command.append("--allow-teacher")
        return {
            "allowed": True,
            "reason": "viea_action_executor_exact_action",
            "evidence_paths": evidence_paths,
            "command": command,
            "hook_target": "viea_action_executor",
        }
    if source == "live_command_channel" and kind == "background_broad_transfer_status":
        return {
            "allowed": True,
            "reason": "broad_transfer_status_report",
            "command": [
                sys.executable,
                "scripts/broad_transfer_matrix.py",
                "--min-public-tasks",
                "32",
                "--out",
                "reports/broad_transfer_matrix.json",
                "--markdown-out",
                "reports/broad_transfer_matrix.md",
            ],
            "hook_target": "background_task",
        }
    if source == "live_command_channel" and kind == "background_operator_status":
        return {
            "allowed": True,
            "reason": "operator_status_report",
            "command": [
                sys.executable,
                "scripts/hive_operator_os.py",
                "--config",
                "configs/hive_operator_os.json",
                "--db",
                "reports/hive_work_board.sqlite",
                "--out",
                "reports/hive_operator_os.json",
                "--markdown-out",
                "reports/hive_operator_os.md",
            ],
            "hook_target": "background_task",
        }
    if source == "live_command_channel" and kind == "background_vacation_status":
        return {
            "allowed": True,
            "reason": "vacation_status_report",
            "command": [
                sys.executable,
                "scripts/vacation_mode_supervisor.py",
                "--cycles",
                "1",
                "--max-actions-per-cycle",
                "0",
                "--out",
                "reports/vacation_mode_supervisor.json",
                "--markdown-out",
                "reports/vacation_mode_supervisor.md",
            ],
            "hook_target": "background_task",
        }
    if source == "high_transfer_curriculum_scheduler":
        concept = str(get_path(task, ["evidence", "concept"], "") or "")
        diagnostic_reason = non_promotable_diagnostic_reason(concept)
        if diagnostic_reason:
            return {
                "allowed": False,
                "reason": "non_promotable_concept_diagnostic_only",
                "concept": concept,
                "diagnostic_reason": diagnostic_reason,
                "replacement_concept": promotion_safe_replacement_concept(concept) or None,
                "command": [],
            }
        if concept == "type_contract_diagnostic":
            return {
                "allowed": True,
                "reason": "direct_high_transfer_type_contract_diagnostic",
                "concept": concept,
                "evidence_paths": [
                    "reports/type_contract_diagnostic.json",
                    "D:/ProjectTheseus/training_data/high_transfer/private_train/type_contract_decoder_feedback.jsonl",
                ],
                "command": [
                    sys.executable,
                    "scripts/type_contract_diagnostic.py",
                    "--max-rows",
                    "960",
                    "--feedback-out",
                    "D:/ProjectTheseus/training_data/high_transfer/private_train/type_contract_decoder_feedback.jsonl",
                    "--out",
                    "reports/type_contract_diagnostic.json",
                    "--markdown-out",
                    "reports/type_contract_diagnostic.md",
                ],
                "hook_target": "training_launch",
            }
        if concept == "type_contract_four_card_calibration":
            guard = private_public_calibration_guard()
            if not guard["allowed"]:
                return {
                    "allowed": False,
                    "reason": "public_receiver_calibration_blocked_by_private_gate",
                    "concept": concept,
                    "guard": guard,
                    "command": [],
                }
            slug = "source_mbpp_source_evalplus_source_bigcodebench_source_livecodebench_seed14_32"
            return {
                "allowed": True,
                "reason": "direct_high_transfer_type_contract_four_card_calibration",
                "concept": concept,
                "guard": guard,
                "evidence_paths": [
                    f"reports/broad_transfer_closure_runner_{slug}.json",
                    f"reports/real_code_benchmark_graduation_{slug}.json",
                    f"reports/code_lm_closure_{slug}.json",
                    "reports/broad_transfer_matrix.json",
                ],
                "command": [
                    sys.executable,
                    "scripts/broad_transfer_closure_runner.py",
                    "--execute",
                    "--cards",
                    "source_mbpp,source_evalplus,source_bigcodebench,source_livecodebench",
                    "--seed",
                    "14",
                    "--max-public-cases-per-card",
                    "32",
                    "--typed-edge-exec-receiver-v1",
                    "--private-type-shape-receiver-veto-v1",
                    "--score-existing-public-candidates",
                    "--student-candidate-manifest",
                    "reports/student_code_candidates_private_pressure_private.jsonl",
                    "--max-high-transfer-private-train",
                    "14400",
                    "--max-rust-work-steps",
                    "12000000",
                    "--out",
                    f"reports/broad_transfer_closure_runner_{slug}.json",
                    "--markdown-out",
                    f"reports/broad_transfer_closure_runner_{slug}.md",
                ],
                "hook_target": "training_launch",
            }
        if concept == "execution_shaped_four_card_calibration":
            guard = private_public_calibration_guard()
            if not guard["allowed"]:
                return {
                    "allowed": False,
                    "reason": "public_receiver_calibration_blocked_by_private_gate",
                    "concept": concept,
                    "guard": guard,
                    "command": [],
                }
            slug = "source_mbpp_source_evalplus_source_bigcodebench_source_livecodebench_seed14_32"
            return {
                "allowed": True,
                "reason": "direct_high_transfer_execution_shaped_four_card_calibration",
                "concept": concept,
                "guard": guard,
                "evidence_paths": [
                    f"reports/broad_transfer_closure_runner_{slug}.json",
                    f"reports/real_code_benchmark_graduation_{slug}.json",
                    f"reports/code_lm_closure_{slug}.json",
                    f"reports/code_lm_closure_rust_{slug}.json",
                    "reports/high_transfer_execution_shaped_programs_code_residual_curriculum.json",
                    "reports/type_contract_diagnostic.json",
                    "reports/broad_transfer_matrix.json",
                    "reports/transfer_generalization_audit.json",
                ],
                "command": [
                    sys.executable,
                    "scripts/broad_transfer_closure_runner.py",
                    "--execute",
                    "--cards",
                    "source_mbpp,source_evalplus,source_bigcodebench,source_livecodebench",
                    "--seed",
                    "14",
                    "--max-public-cases-per-card",
                    "32",
                    "--score-existing-public-candidates",
                    "--student-candidate-manifest",
                    "reports/student_code_candidates_private_pressure_private.jsonl",
                    "--out",
                    f"reports/broad_transfer_closure_runner_{slug}.json",
                    "--markdown-out",
                    f"reports/broad_transfer_closure_runner_{slug}.md",
                ],
                "hook_target": "training_launch",
            }
        if concept == "execution_shape_private_ablation":
            smoke_guard = execution_shape_no_template_smoke_guard()
            if smoke_guard.get("reason") in {
                "execution_shape_no_template_smoke_missing",
                "execution_shape_no_template_smoke_stale_after_decoder_source_change",
            }:
                return {
                    "allowed": True,
                    "reason": "direct_no_template_execution_shape_smoke_before_full_ablation",
                    "concept": concept,
                    "guard": smoke_guard,
                    "evidence_paths": [
                        "reports/execution_shape_private_ablation_smoke.json",
                        "reports/execution_shape_private_ablation_smoke_rust.json",
                        "reports/execution_shape_private_ablation_smoke_candidates.jsonl",
                    ],
                    "command": execution_shape_no_template_smoke_command(),
                    "env": {
                        "THESEUS_TEMPLATE_FREE_STUDENT_CANDIDATES": "1",
                        "THESEUS_ALLOW_DIAGNOSTIC_TEMPLATE_CANDIDATES": "0",
                    },
                    "hook_target": "training_launch",
                }
            if not smoke_guard.get("allowed"):
                return {
                    "allowed": False,
                    "reason": "execution_shape_full_ablation_blocked_by_no_template_smoke_gate",
                    "concept": concept,
                    "guard": smoke_guard,
                    "command": [],
                }
            return {
                "allowed": True,
                "reason": "direct_high_transfer_execution_shape_private_ablation",
                "concept": concept,
                "guard": smoke_guard,
                "evidence_paths": [
                    "reports/execution_shape_private_ablation.json",
                    "reports/execution_shape_private_ablation_rust.json",
                    "reports/high_transfer_execution_shaped_programs_code_residual_curriculum.json",
                    "D:/ProjectTheseus/training_data/high_transfer/private_train/execution_shaped_programs_residual_code_lm_tasks.jsonl",
                ],
                "command": [
                    sys.executable,
                    "scripts/execution_shape_private_ablation.py",
                    "--seed",
                    "14",
                    "--train-rows",
                    "320",
                    "--eval-rows",
                    "64",
                    "--out",
                    "reports/execution_shape_private_ablation.json",
                    "--markdown-out",
                    "reports/execution_shape_private_ablation.md",
                ],
                "hook_target": "training_launch",
            }
        if concept == "edge_exec_repair_four_card_calibration":
            guard = private_public_calibration_guard()
            if not guard["allowed"]:
                return {
                    "allowed": False,
                    "reason": "public_receiver_calibration_blocked_by_private_gate",
                    "concept": concept,
                    "guard": guard,
                    "command": [],
                }
            slug = "source_mbpp_source_evalplus_source_bigcodebench_source_livecodebench_seed14_32"
            return {
                "allowed": True,
                "reason": "direct_high_transfer_edge_exec_repair_four_card_calibration",
                "concept": concept,
                "guard": guard,
                "evidence_paths": [
                    f"reports/broad_transfer_closure_runner_{slug}.json",
                    f"reports/real_code_benchmark_graduation_{slug}.json",
                    f"reports/code_lm_closure_{slug}.json",
                    f"reports/code_lm_closure_rust_{slug}.json",
                    "reports/broad_transfer_matrix.json",
                    "reports/transfer_generalization_audit.json",
                ],
                "command": [
                    sys.executable,
                    "scripts/broad_transfer_closure_runner.py",
                    "--execute",
                    "--cards",
                    "source_mbpp,source_evalplus,source_bigcodebench,source_livecodebench",
                    "--seed",
                    "14",
                    "--max-public-cases-per-card",
                    "32",
                    "--score-existing-public-candidates",
                    "--student-candidate-manifest",
                    "reports/student_code_candidates_private_pressure_private.jsonl",
                    "--out",
                    f"reports/broad_transfer_closure_runner_{slug}.json",
                    "--markdown-out",
                    f"reports/broad_transfer_closure_runner_{slug}.md",
                ],
                "hook_target": "training_launch",
            }
        if concept == "private_pressure_four_card_recalibration":
            guard = private_public_calibration_guard()
            if not guard["allowed"]:
                return {
                    "allowed": False,
                    "reason": "public_receiver_calibration_blocked_by_private_gate",
                    "concept": concept,
                    "guard": guard,
                    "command": [],
                }
            slug = "source_mbpp_source_evalplus_source_bigcodebench_source_livecodebench_seed14_32"
            return {
                "allowed": True,
                "reason": "direct_high_transfer_private_pressure_four_card_recalibration_after_private_gate",
                "concept": concept,
                "guard": guard,
                "evidence_paths": [
                    f"reports/broad_transfer_closure_runner_{slug}.json",
                    f"reports/real_code_benchmark_graduation_{slug}.json",
                    f"reports/code_lm_closure_{slug}.json",
                    f"reports/code_lm_closure_rust_{slug}.json",
                    "reports/broad_transfer_matrix.json",
                    "reports/transfer_generalization_audit.json",
                ],
                "command": [
                    sys.executable,
                    "scripts/broad_transfer_closure_runner.py",
                    "--execute",
                    "--cards",
                    "source_mbpp,source_evalplus,source_bigcodebench,source_livecodebench",
                    "--seed",
                    "14",
                    "--max-public-cases-per-card",
                    "32",
                    "--typed-edge-exec-receiver-v1",
                    "--private-type-shape-receiver-veto-v1",
                    "--score-existing-public-candidates",
                    "--student-candidate-manifest",
                    "reports/student_code_candidates_private_pressure_private.jsonl",
                    "--max-high-transfer-private-train",
                    "14400",
                    "--max-rust-work-steps",
                    "12000000",
                    "--out",
                    f"reports/broad_transfer_closure_runner_{slug}.json",
                    "--markdown-out",
                    f"reports/broad_transfer_closure_runner_{slug}.md",
                ],
                "hook_target": "training_launch",
            }
        if concept == "typed_interface_private_closure":
            high_transfer_rows = ";".join(
                [
                    "D:/ProjectTheseus/training_data/high_transfer/private_train/type_and_return_shape_residual_code_lm_tasks.jsonl",
                    "D:/ProjectTheseus/training_data/high_transfer/private_train/type_contract_decoder_feedback.jsonl",
                    "D:/ProjectTheseus/training_data/high_transfer/private_train/typed_interface_skeleton_residual_code_lm_tasks.jsonl",
                    "D:/ProjectTheseus/training_data/high_transfer/private_train/admissibility_and_interface_residual_code_lm_tasks.jsonl",
                    "D:/ProjectTheseus/training_data/high_transfer/private_train/edge_conditions_residual_code_lm_tasks.jsonl",
                    "D:/ProjectTheseus/training_data/high_transfer/private_train/edge_contract_4card_residual_code_lm_tasks.jsonl",
                    "D:/ProjectTheseus/training_data/high_transfer/private_train/edge_contract_balanced_4card_private_curriculum_v2_residual_code_lm_tasks.jsonl",
                    "D:/ProjectTheseus/training_data/high_transfer/private_train/edge_case_full_body_private_curriculum_v1_residual_code_lm_tasks.jsonl",
                    "D:/ProjectTheseus/training_data/high_transfer/private_train/algorithmic_planning_residual_code_lm_tasks.jsonl",
                    "D:/ProjectTheseus/training_data/high_transfer/private_train/execution_shaped_programs_residual_code_lm_tasks.jsonl",
                ]
            )
            return {
                "allowed": True,
                "reason": "direct_high_transfer_typed_interface_private_closure",
                "concept": concept,
                "evidence_paths": [
                    "reports/code_lm_closure_typed_interface_private.json",
                    "reports/code_lm_closure_rust_typed_interface_private.json",
                    "reports/high_transfer_typed_interface_skeleton_code_residual_curriculum.json",
                    "D:/ProjectTheseus/training_data/high_transfer/private_train/typed_interface_skeleton_residual_code_lm_tasks.jsonl",
                ],
                "command": [
                    sys.executable,
                    "scripts/code_lm_closure.py",
                    "--skip-public-calibration",
                    "--public-cards",
                    "source_mbpp,source_evalplus,source_bigcodebench,source_livecodebench",
                    "--seed",
                    "19",
                    "--max-public-cases-per-card",
                    "32",
                    "--private-count",
                    "960",
                    "--epochs",
                    "8",
                    "--candidates-per-task",
                    "12",
                    "--max-extra-private-train",
                    "2000",
                    "--max-residual-private-train",
                    "1200",
                    "--max-repo-repair-private-train",
                    "1200",
                    "--high-transfer-private-train-jsonl",
                    high_transfer_rows,
                    "--max-high-transfer-private-train",
                    "14400",
                    "--max-rust-work-steps",
                    "12000000",
                    "--rust-timeout-seconds",
                    str(max(60, int(args.timeout_seconds))),
                    "--sts-timeout-seconds",
                    str(max(60, min(int(args.timeout_seconds), 7200))),
                    "--private-curriculum-out",
                    "data/private_code_curriculum/code_lm_closure_typed_interface_private.jsonl",
                    "--public-task-manifest-out",
                    "reports/code_lm_public_tasks_typed_interface_private.jsonl",
                    "--checkpoint-out",
                    "reports/student_code_lm_checkpoint_typed_interface_private.json",
                    "--private-candidate-out",
                    "reports/code_lm_private_candidates_typed_interface_private.jsonl",
                    "--public-candidate-out",
                    "reports/student_code_candidates_typed_interface_private.jsonl",
                    "--rust-report-out",
                    "reports/code_lm_closure_rust_typed_interface_private.json",
                    "--public-report-out",
                    "reports/real_code_benchmark_graduation_typed_interface_private_skipped.json",
                    "--public-trace-out",
                    "reports/real_code_benchmark_traces_typed_interface_private_skipped.jsonl",
                    "--out",
                    "reports/code_lm_closure_typed_interface_private.json",
                    "--sts-conditioning-input-out",
                    "reports/code_lm_sts_conditioning_input_typed_interface_private.jsonl",
                    "--sts-generation-out",
                    "reports/code_lm_sts_public_generations_typed_interface_private.jsonl",
                    "--sts-conditioning-checkpoint-out",
                    "reports/code_lm_sts_conditioning_checkpoint_typed_interface_private.json",
                    "--sts-conditioning-report-out",
                    "reports/code_lm_sts_conditioning_report_typed_interface_private.json",
                    "--lock-path",
                    "reports/code_lm_closure_typed_interface_private.lock",
                    "--typed-edge-exec-receiver-v1",
                    "--edge-obligation-decode-gate-v1",
                    "--private-type-shape-receiver-veto-v1",
                    "--edge-obligation-report-out",
                    "reports/edge_obligation_decode_gate_v1_typed_interface_private.json",
                    "--edge-obligation-markdown-out",
                    "reports/edge_obligation_decode_gate_v1_typed_interface_private.md",
                ],
                "hook_target": "training_launch",
            }
        if concept == "edge_contract_private_closure":
            return {
                "allowed": True,
                "reason": "direct_high_transfer_edge_contract_private_closure",
                "concept": concept,
                "evidence_paths": [
                    "reports/code_lm_closure_edge_contract_4card_private.json",
                    "reports/code_lm_closure_rust_edge_contract_4card_private.json",
                    "reports/high_transfer_edge_contract_4card_code_residual_curriculum.json",
                    "D:/ProjectTheseus/training_data/high_transfer/private_train/edge_contract_4card_residual_code_lm_tasks.jsonl",
                ],
                "command": [
                    sys.executable,
                    "scripts/code_lm_closure.py",
                    "--skip-public-calibration",
                    "--public-cards",
                    "source_mbpp,source_evalplus,source_bigcodebench,source_livecodebench",
                    "--seed",
                    "31",
                    "--max-public-cases-per-card",
                    "32",
                    "--private-count",
                    "960",
                    "--epochs",
                    "8",
                    "--candidates-per-task",
                    "12",
                    "--disable-extra-private-train",
                    "--disable-residual-private-train",
                    "--disable-repo-repair-private-train",
                    "--high-transfer-private-train-jsonl",
                    "D:/ProjectTheseus/training_data/high_transfer/private_train/edge_contract_4card_residual_code_lm_tasks.jsonl",
                    "--max-high-transfer-private-train",
                    "960",
                    "--max-rust-work-steps",
                    "12000000",
                    "--rust-timeout-seconds",
                    str(max(60, int(args.timeout_seconds))),
                    "--sts-timeout-seconds",
                    str(max(60, min(int(args.timeout_seconds), 7200))),
                    "--private-curriculum-out",
                    "data/private_code_curriculum/code_lm_closure_edge_contract_4card_private.jsonl",
                    "--public-task-manifest-out",
                    "reports/code_lm_public_tasks_edge_contract_4card_private.jsonl",
                    "--checkpoint-out",
                    "reports/student_code_lm_checkpoint_edge_contract_4card_private.json",
                    "--private-candidate-out",
                    "reports/code_lm_private_candidates_edge_contract_4card_private.jsonl",
                    "--public-candidate-out",
                    "reports/student_code_candidates_edge_contract_4card_private.jsonl",
                    "--rust-report-out",
                    "reports/code_lm_closure_rust_edge_contract_4card_private.json",
                    "--public-report-out",
                    "reports/real_code_benchmark_graduation_edge_contract_4card_private_skipped.json",
                    "--public-trace-out",
                    "reports/real_code_benchmark_traces_edge_contract_4card_private_skipped.jsonl",
                    "--out",
                    "reports/code_lm_closure_edge_contract_4card_private.json",
                    "--sts-conditioning-input-out",
                    "reports/code_lm_sts_conditioning_input_edge_contract_4card_private.jsonl",
                    "--sts-generation-out",
                    "reports/code_lm_sts_public_generations_edge_contract_4card_private.jsonl",
                    "--sts-conditioning-checkpoint-out",
                    "reports/code_lm_sts_conditioning_checkpoint_edge_contract_4card_private.json",
                    "--sts-conditioning-report-out",
                    "reports/code_lm_sts_conditioning_report_edge_contract_4card_private.json",
                    "--lock-path",
                    "reports/code_lm_closure_edge_contract_4card_private.lock",
                    "--typed-edge-exec-receiver-v1",
                    "--edge-obligation-decode-gate-v1",
                    "--private-type-shape-receiver-veto-v1",
                    "--edge-obligation-report-out",
                    "reports/edge_obligation_decode_gate_v1_edge_contract_4card_private.json",
                    "--edge-obligation-markdown-out",
                    "reports/edge_obligation_decode_gate_v1_edge_contract_4card_private.md",
                ],
                "hook_target": "training_launch",
            }
        if concept == "edge_contract_balanced_private_closure_v2":
            high_transfer_rows = "D:/ProjectTheseus/training_data/high_transfer/private_train/edge_contract_balanced_4card_private_curriculum_v2_residual_code_lm_tasks.jsonl"
            return {
                "allowed": True,
                "reason": "direct_high_transfer_edge_contract_balanced_private_closure_v2",
                "concept": concept,
                "evidence_paths": [
                    "reports/code_lm_closure_edge_contract_balanced_4card_private_v2.json",
                    "reports/code_lm_closure_rust_edge_contract_balanced_4card_private_v2.json",
                    "reports/high_transfer_edge_contract_balanced_4card_private_curriculum_v2_code_residual_curriculum.json",
                    high_transfer_rows,
                ],
                "command": [
                    sys.executable,
                    "scripts/code_lm_closure.py",
                    "--skip-public-calibration",
                    "--public-cards",
                    "source_mbpp,source_evalplus,source_bigcodebench,source_livecodebench",
                    "--seed",
                    "41",
                    "--max-public-cases-per-card",
                    "32",
                    "--private-count",
                    "960",
                    "--epochs",
                    "8",
                    "--candidates-per-task",
                    "12",
                    "--disable-extra-private-train",
                    "--disable-residual-private-train",
                    "--disable-repo-repair-private-train",
                    "--high-transfer-private-train-jsonl",
                    high_transfer_rows,
                    "--max-high-transfer-private-train",
                    "960",
                    "--max-rust-work-steps",
                    "12000000",
                    "--rust-timeout-seconds",
                    str(max(60, int(args.timeout_seconds))),
                    "--sts-timeout-seconds",
                    str(max(60, min(int(args.timeout_seconds), 7200))),
                    "--private-curriculum-out",
                    "data/private_code_curriculum/code_lm_closure_edge_contract_balanced_4card_private_v2.jsonl",
                    "--public-task-manifest-out",
                    "reports/code_lm_public_tasks_edge_contract_balanced_4card_private_v2.jsonl",
                    "--checkpoint-out",
                    "reports/student_code_lm_checkpoint_edge_contract_balanced_4card_private_v2.json",
                    "--private-candidate-out",
                    "reports/code_lm_private_candidates_edge_contract_balanced_4card_private_v2.jsonl",
                    "--public-candidate-out",
                    "reports/student_code_candidates_edge_contract_balanced_4card_private_v2.jsonl",
                    "--rust-report-out",
                    "reports/code_lm_closure_rust_edge_contract_balanced_4card_private_v2.json",
                    "--public-report-out",
                    "reports/real_code_benchmark_graduation_edge_contract_balanced_4card_private_v2_skipped.json",
                    "--public-trace-out",
                    "reports/real_code_benchmark_traces_edge_contract_balanced_4card_private_v2_skipped.jsonl",
                    "--out",
                    "reports/code_lm_closure_edge_contract_balanced_4card_private_v2.json",
                    "--sts-conditioning-input-out",
                    "reports/code_lm_sts_conditioning_input_edge_contract_balanced_4card_private_v2.jsonl",
                    "--sts-generation-out",
                    "reports/code_lm_sts_public_generations_edge_contract_balanced_4card_private_v2.jsonl",
                    "--sts-conditioning-checkpoint-out",
                    "reports/code_lm_sts_conditioning_checkpoint_edge_contract_balanced_4card_private_v2.json",
                    "--sts-conditioning-report-out",
                    "reports/code_lm_sts_conditioning_report_edge_contract_balanced_4card_private_v2.json",
                    "--lock-path",
                    "reports/code_lm_closure_edge_contract_balanced_4card_private_v2.lock",
                    "--typed-edge-exec-receiver-v1",
                    "--edge-obligation-decode-gate-v1",
                    "--private-type-shape-receiver-veto-v1",
                    "--edge-obligation-report-out",
                    "reports/edge_obligation_decode_gate_v1_edge_contract_balanced_4card_private_v2.json",
                    "--edge-obligation-markdown-out",
                    "reports/edge_obligation_decode_gate_v1_edge_contract_balanced_4card_private_v2.md",
                ],
                "hook_target": "training_launch",
            }
        if concept == "edge_case_full_body_private_closure_v1":
            high_transfer_rows = "D:/ProjectTheseus/training_data/high_transfer/private_train/edge_case_full_body_private_curriculum_v1_residual_code_lm_tasks.jsonl"
            return {
                "allowed": True,
                "reason": "direct_high_transfer_edge_case_full_body_private_closure_v1",
                "concept": concept,
                "evidence_paths": [
                    "reports/code_lm_closure_edge_case_full_body_private_v1.json",
                    "reports/code_lm_closure_rust_edge_case_full_body_private_v1.json",
                    "reports/high_transfer_edge_case_full_body_private_curriculum_v1_code_residual_curriculum.json",
                    high_transfer_rows,
                ],
                "command": [
                    sys.executable,
                    "scripts/code_lm_closure.py",
                    "--skip-public-calibration",
                    "--public-cards",
                    "source_mbpp,source_evalplus,source_bigcodebench,source_livecodebench",
                    "--seed",
                    "43",
                    "--max-public-cases-per-card",
                    "32",
                    "--private-count",
                    "960",
                    "--epochs",
                    "8",
                    "--candidates-per-task",
                    "12",
                    "--disable-extra-private-train",
                    "--disable-residual-private-train",
                    "--disable-repo-repair-private-train",
                    "--high-transfer-private-train-jsonl",
                    high_transfer_rows,
                    "--max-high-transfer-private-train",
                    "960",
                    "--max-rust-work-steps",
                    "12000000",
                    "--rust-timeout-seconds",
                    str(max(60, int(args.timeout_seconds))),
                    "--sts-timeout-seconds",
                    str(max(60, min(int(args.timeout_seconds), 7200))),
                    "--private-curriculum-out",
                    "data/private_code_curriculum/code_lm_closure_edge_case_full_body_private_v1.jsonl",
                    "--public-task-manifest-out",
                    "reports/code_lm_public_tasks_edge_case_full_body_private_v1.jsonl",
                    "--checkpoint-out",
                    "reports/student_code_lm_checkpoint_edge_case_full_body_private_v1.json",
                    "--private-candidate-out",
                    "reports/code_lm_private_candidates_edge_case_full_body_private_v1.jsonl",
                    "--public-candidate-out",
                    "reports/student_code_candidates_edge_case_full_body_private_v1.jsonl",
                    "--rust-report-out",
                    "reports/code_lm_closure_rust_edge_case_full_body_private_v1.json",
                    "--public-report-out",
                    "reports/real_code_benchmark_graduation_edge_case_full_body_private_v1_skipped.json",
                    "--public-trace-out",
                    "reports/real_code_benchmark_traces_edge_case_full_body_private_v1_skipped.jsonl",
                    "--out",
                    "reports/code_lm_closure_edge_case_full_body_private_v1.json",
                    "--sts-conditioning-input-out",
                    "reports/code_lm_sts_conditioning_input_edge_case_full_body_private_v1.jsonl",
                    "--sts-generation-out",
                    "reports/code_lm_sts_public_generations_edge_case_full_body_private_v1.jsonl",
                    "--sts-conditioning-checkpoint-out",
                    "reports/code_lm_sts_conditioning_checkpoint_edge_case_full_body_private_v1.json",
                    "--sts-conditioning-report-out",
                    "reports/code_lm_sts_conditioning_report_edge_case_full_body_private_v1.json",
                    "--lock-path",
                    "reports/code_lm_closure_edge_case_full_body_private_v1.lock",
                    "--typed-edge-exec-receiver-v1",
                    "--edge-obligation-decode-gate-v1",
                    "--private-type-shape-receiver-veto-v1",
                    "--edge-obligation-report-out",
                    "reports/edge_obligation_decode_gate_v1_edge_case_full_body_private_v1.json",
                    "--edge-obligation-markdown-out",
                    "reports/edge_obligation_decode_gate_v1_edge_case_full_body_private_v1.md",
                ],
                "hook_target": "training_launch",
            }
        if concept == "edge_contract_v2_private_closure":
            high_transfer_rows = "D:/ProjectTheseus/training_data/high_transfer/private_train/edge_contract_v2_private_residual_curriculum_residual_code_lm_tasks.jsonl"
            return {
                "allowed": True,
                "reason": "direct_high_transfer_edge_contract_v2_private_closure",
                "concept": concept,
                "evidence_paths": [
                    "reports/code_lm_closure_edge_contract_v2_private.json",
                    "reports/code_lm_closure_rust_edge_contract_v2_private.json",
                    "reports/edge_contract_v2_private_verifier.json",
                    "reports/edge_contract_v2_private_closure_runner.json",
                    "reports/high_transfer_edge_contract_v2_private_residual_curriculum_code_residual_curriculum.json",
                    high_transfer_rows,
                ],
                "command": [
                    sys.executable,
                    "scripts/edge_contract_v2_private_closure_runner.py",
                    "--timeout-seconds",
                    str(max(60, int(args.timeout_seconds))),
                    "--rust-timeout-seconds",
                    str(max(60, int(args.timeout_seconds))),
                    "--sts-timeout-seconds",
                    str(max(60, min(int(args.timeout_seconds), 7200))),
                ],
                "hook_target": "training_launch",
            }
        if concept == "candidate_floor_v2_private_closure":
            high_transfer_rows = "D:/ProjectTheseus/training_data/high_transfer/private_train/candidate_floor_v2_private_residual_curriculum_residual_code_lm_tasks.jsonl"
            return {
                "allowed": True,
                "reason": "direct_high_transfer_candidate_floor_v2_private_closure",
                "concept": concept,
                "evidence_paths": [
                    "reports/code_lm_closure_candidate_floor_v2_private.json",
                    "reports/code_lm_closure_rust_candidate_floor_v2_private.json",
                    "reports/high_transfer_candidate_floor_v2_private_residual_curriculum_code_residual_curriculum.json",
                    high_transfer_rows,
                ],
                "command": [
                    sys.executable,
                    "scripts/code_lm_closure.py",
                    "--skip-public-calibration",
                    "--public-cards",
                    "source_mbpp,source_evalplus,source_bigcodebench,source_livecodebench",
                    "--seed",
                    "23",
                    "--max-public-cases-per-card",
                    "32",
                    "--private-count",
                    "960",
                    "--epochs",
                    "8",
                    "--candidates-per-task",
                    "12",
                    "--disable-extra-private-train",
                    "--disable-residual-private-train",
                    "--disable-repo-repair-private-train",
                    "--high-transfer-private-train-jsonl",
                    high_transfer_rows,
                    "--max-high-transfer-private-train",
                    "14400",
                    "--max-rust-work-steps",
                    "12000000",
                    "--rust-timeout-seconds",
                    str(max(60, int(args.timeout_seconds))),
                    "--sts-timeout-seconds",
                    str(max(60, min(int(args.timeout_seconds), 7200))),
                    "--private-curriculum-out",
                    "data/private_code_curriculum/code_lm_closure_candidate_floor_v2_private.jsonl",
                    "--public-task-manifest-out",
                    "reports/code_lm_public_tasks_candidate_floor_v2_private.jsonl",
                    "--checkpoint-out",
                    "reports/student_code_lm_checkpoint_candidate_floor_v2_private.json",
                    "--private-candidate-out",
                    "reports/code_lm_private_candidates_candidate_floor_v2_private.jsonl",
                    "--public-candidate-out",
                    "reports/student_code_candidates_candidate_floor_v2_private.jsonl",
                    "--rust-report-out",
                    "reports/code_lm_closure_rust_candidate_floor_v2_private.json",
                    "--public-report-out",
                    "reports/real_code_benchmark_graduation_candidate_floor_v2_private_skipped.json",
                    "--public-trace-out",
                    "reports/real_code_benchmark_traces_candidate_floor_v2_private_skipped.jsonl",
                    "--out",
                    "reports/code_lm_closure_candidate_floor_v2_private.json",
                    "--sts-conditioning-input-out",
                    "reports/code_lm_sts_conditioning_input_candidate_floor_v2_private.jsonl",
                    "--sts-generation-out",
                    "reports/code_lm_sts_public_generations_candidate_floor_v2_private.jsonl",
                    "--sts-conditioning-checkpoint-out",
                    "reports/code_lm_sts_conditioning_checkpoint_candidate_floor_v2_private.json",
                    "--sts-conditioning-report-out",
                    "reports/code_lm_sts_conditioning_report_candidate_floor_v2_private.json",
                    "--lock-path",
                    "reports/code_lm_closure_candidate_floor_v2_private.lock",
                    "--typed-edge-exec-receiver-v1",
                    "--edge-obligation-decode-gate-v1",
                    "--private-type-shape-receiver-veto-v1",
                    "--edge-obligation-report-out",
                    "reports/edge_obligation_decode_gate_v1_candidate_floor_v2_private.json",
                    "--edge-obligation-markdown-out",
                    "reports/edge_obligation_decode_gate_v1_candidate_floor_v2_private.md",
                ],
                "env": {
                    "THESEUS_STRATIFIED_WORK_BUDGET_ADMISSION": "1",
                    "THESEUS_TARGET_FAMILY_STARVATION_RESCUE": "1",
                    "THESEUS_TARGET_FAMILY_STARVATION_RESCUE_MIN_ROWS": "48",
                    "THESEUS_TYPED_EDGE_EXEC_RECEIVER_V1": "1",
                    "THESEUS_PRIVATE_TYPE_SHAPE_RECEIVER_VETO_V1": "1",
                },
                "hook_target": "training_launch",
            }
        if concept == "private_pressure_private_closure":
            preflight_guard = code_contract_preflight_guard()
            if not preflight_guard.get("allowed"):
                return {
                    "allowed": True,
                    "reason": "direct_decoder_contract_preflight_before_private_pressure_closure",
                    "concept": concept,
                    "guard": preflight_guard,
                    "evidence_paths": [
                        "reports/code_lm_closure_public_contract_preflight_seed23_32.json",
                        "reports/code_lm_preflight_private_curriculum_seed23_32.jsonl",
                        "reports/code_lm_public_tasks_preflight_seed23_32.jsonl",
                    ],
                    "command": code_contract_preflight_command(),
                    "env": {
                        "THESEUS_TEMPLATE_FREE_STUDENT_CANDIDATES": "1",
                        "THESEUS_ALLOW_DIAGNOSTIC_TEMPLATE_CANDIDATES": "0",
                    },
                    "hook_target": "training_launch",
                }
            if private_pressure_edge_obligation_gate_needed(task):
                command_spec = private_pressure_edge_obligation_gate_command(args)
                command_spec["concept"] = concept
                command_spec["guard"] = private_pressure_private_closure_state_for_task(task)
                return command_spec
            edge_gate_failure = private_pressure_edge_obligation_gate_failed_current(task)
            if edge_gate_failure.get("blocked"):
                return {
                    "allowed": False,
                    "reason": edge_gate_failure["reason"],
                    "concept": concept,
                    "guard": edge_gate_failure,
                    "command": [],
                }
            no_template_smoke_guard = execution_shape_no_template_smoke_guard()
            if not no_template_smoke_guard.get("allowed"):
                return {
                    "allowed": False,
                    "reason": "private_pressure_closure_blocked_by_no_template_execution_shape_gate",
                    "concept": concept,
                    "guard": no_template_smoke_guard,
                    "command": [],
                }
            return {
                "allowed": True,
                "reason": "resource_aware_train_once_private_pressure_fanout",
                "concept": concept,
                "sts_default_policy": {
                    "default": "native_sts_conditioning_on",
                    "disable_flag": "--disable-native-sts-conditioning",
                    "sts_off_role": "control_ablation_only",
                },
                "evidence_paths": [
                    "reports/code_lm_train_once_fanout.json",
                    "reports/code_lm_train_once_fanout.md",
                    f"reports/code_lm_closure_{PRIVATE_PRESSURE_TRAIN_ONCE_CURRENT_SLUG}.json",
                    f"reports/code_lm_closure_rust_{PRIVATE_PRESSURE_TRAIN_ONCE_CURRENT_SLUG}_fanout.json",
                    "reports/resource_aware_execution_policy.json",
                    f"reports/code_lm_train_once_fanout_{PRIVATE_PRESSURE_TRAIN_ONCE_CURRENT_SLUG}_phase_ledger.jsonl",
                    f"reports/code_lm_private_candidates_{PRIVATE_PRESSURE_TRAIN_ONCE_CURRENT_SLUG}.jsonl",
                    f"reports/student_code_candidates_{PRIVATE_PRESSURE_TRAIN_ONCE_CURRENT_SLUG}.jsonl",
                    "reports/decoder_v2_private_ablation_gate.json",
                    "reports/private_public_transfer_proof.json",
                ],
                "command": [
                    sys.executable,
                    "scripts/code_lm_train_once_fanout.py",
                    "--private-only",
                    "--execute",
                    "--slug",
                    PRIVATE_PRESSURE_TRAIN_ONCE_CURRENT_SLUG,
                    "--private-count",
                    "480",
                    "--epochs",
                    "4",
                    "--candidates-per-task",
                    "8",
                    "--max-high-transfer-private-train",
                    "6400",
                    "--max-rust-work-steps",
                    "4000000",
                    "--rust-timeout-seconds",
                    str(max(60, min(int(args.timeout_seconds), 7200))),
                    "--fanout-timeout-seconds",
                    str(max(60, int(args.timeout_seconds))),
                    "--out",
                    "reports/code_lm_train_once_fanout.json",
                    "--markdown-out",
                    "reports/code_lm_train_once_fanout.md",
                ],
                "env": {
                    "THESEUS_STRATIFIED_WORK_BUDGET_ADMISSION": "1",
                    "THESEUS_TARGET_FAMILY_STARVATION_RESCUE": "1",
                    "THESEUS_TARGET_FAMILY_STARVATION_RESCUE_MIN_ROWS": "32",
                    "THESEUS_TYPED_EDGE_EXEC_RECEIVER_V1": "1",
                    "THESEUS_PRIVATE_TYPE_SHAPE_RECEIVER_VETO_V1": "1",
                    "THESEUS_TEMPLATE_FREE_STUDENT_CANDIDATES": "1",
                    "THESEUS_ALLOW_DIAGNOSTIC_TEMPLATE_CANDIDATES": "0",
                },
                "hook_target": "training_launch",
            }
        if concept == "private_pressure_four_card_recalibration":
            guard = private_public_calibration_guard()
            if not guard["allowed"]:
                return {
                    "allowed": False,
                    "reason": "public_receiver_calibration_blocked_by_private_gate",
                    "concept": concept,
                    "guard": guard,
                    "command": [],
                }
            slug = FOUR_CARD_RECEIVER_SLUG
            return {
                "allowed": True,
                "reason": "direct_high_transfer_private_pressure_four_card_recalibration_after_private_gate",
                "concept": concept,
                "guard": guard,
                "evidence_paths": [
                    f"reports/broad_transfer_closure_runner_{slug}.json",
                    f"reports/real_code_benchmark_graduation_{slug}.json",
                    f"reports/code_lm_closure_{slug}.json",
                    f"reports/code_lm_closure_rust_{slug}.json",
                    "reports/broad_transfer_matrix.json",
                    "reports/transfer_generalization_audit.json",
                    "reports/decoder_v2_private_ablation_gate.json",
                ],
                "command": [
                    sys.executable,
                    "scripts/broad_transfer_closure_runner.py",
                    "--execute",
                    "--cards",
                    "source_mbpp,source_evalplus,source_bigcodebench,source_livecodebench",
                    "--seed",
                    "14",
                    "--max-public-cases-per-card",
                    "32",
                    "--typed-edge-exec-receiver-v1",
                    "--private-type-shape-receiver-veto-v1",
                    "--score-existing-public-candidates",
                    "--student-candidate-manifest",
                    "reports/student_code_candidates_private_pressure_private.jsonl",
                    "--max-high-transfer-private-train",
                    "14400",
                    "--max-rust-work-steps",
                    "12000000",
                    "--out",
                    f"reports/broad_transfer_closure_runner_{slug}.json",
                    "--markdown-out",
                    f"reports/broad_transfer_closure_runner_{slug}.md",
                ],
                "hook_target": "training_launch",
            }
        if concept in CODE_HIGH_TRANSFER_CONCEPTS:
            safe = concept.replace("-", "_")
            seed_by_concept = {
                "type_and_return_shape": "14",
                "typed_interface_skeleton": "19",
                "admissibility_and_interface": "16",
                "edge_conditions": "15",
                "edge_contract_4card": "31",
                "edge_contract_balanced_4card_private_curriculum_v2": "41",
                "edge_case_full_body_private_curriculum_v1": "43",
                "edge_contract_v2_private_residual_curriculum": "47",
                "candidate_floor_v2_private_residual_curriculum": "53",
                "residual_targeted_private_edge_case_contract_v1": "59",
                "algorithmic_planning": "17",
                "execution_shaped_programs": "18",
            }
            trace_path = "reports/real_code_benchmark_traces_source_mbpp_source_evalplus_source_bigcodebench_source_livecodebench_seed14_32.jsonl"
            report_path = "reports/real_code_benchmark_graduation_source_mbpp_source_evalplus_source_bigcodebench_source_livecodebench_seed14_32.json"
            if concept == "edge_case_full_body_private_curriculum_v1":
                trace_path = "reports/real_code_benchmark_traces_edge_contract_4card_private_4card_seed31_32.jsonl"
                report_path = "reports/real_code_benchmark_graduation_edge_contract_4card_private_4card_seed31_32.json"
            return {
                "allowed": True,
                "reason": f"direct_high_transfer_code_curriculum:{concept}",
                "concept": concept,
                "evidence_paths": [
                    f"reports/high_transfer_{safe}_code_residual_curriculum.json",
                    "reports/architecture_guidance_loop_semantic_decoder_v2_4card.json",
                    "reports/transfer_generalization_audit.json",
                    f"D:/ProjectTheseus/training_data/high_transfer/private_train/{safe}_residual_code_lm_tasks.jsonl",
                ],
                "command": [
                    sys.executable,
                    "scripts/code_residual_curriculum.py",
                    "--trace-in",
                    trace_path,
                    "--real-code-report",
                    report_path,
                    "--max-rows",
                    "960",
                    "--seed",
                    seed_by_concept.get(concept, "14"),
                    "--concept-focus",
                    concept,
                    "--private-out",
                    f"D:/ProjectTheseus/training_data/high_transfer/private_train/{safe}_residual_code_lm_tasks.jsonl",
                    "--out",
                    f"reports/high_transfer_{safe}_code_residual_curriculum.json",
                    "--markdown-out",
                    f"reports/high_transfer_{safe}_code_residual_curriculum.md",
                ],
                "hook_target": "training_launch",
            }
        if concept == "open_conversation_pantry":
            return {
                "allowed": True,
                "reason": "direct_high_transfer_open_conversation_pantry",
                "concept": concept,
                "evidence_paths": [
                    "reports/open_conversation_training_pantry.json",
                    "D:/ProjectTheseus/training_data/open_conversation_pantry/private_train/conversation_sft_pressure.jsonl",
                    "D:/ProjectTheseus/training_data/open_conversation_pantry/sts_streams/conversation_sts_streams.jsonl",
                ],
                "command": [
                    sys.executable,
                    "scripts/open_conversation_training_pantry.py",
                    "--config",
                    "configs/open_conversation_training_pantry.json",
                    "--root",
                    "D:/ProjectTheseus/training_data/open_conversation_pantry",
                    "--allow-network-fetch",
                    "--refresh",
                    "--out",
                    "reports/open_conversation_training_pantry.json",
                    "--markdown-out",
                    "reports/open_conversation_training_pantry.md",
                ],
                "hook_target": "conversation_training",
            }
        if concept == "multi_turn_conversation":
            return {
                "allowed": True,
                "reason": "direct_high_transfer_multi_turn_conversation",
                "concept": concept,
                "evidence_paths": ["reports/high_transfer_multi_turn_conversation.json"],
                "command": [
                    sys.executable,
                    "scripts/multi_turn_conversation_benchmark.py",
                    "--suite-mode",
                    "large",
                    "--case-limit",
                    "72",
                    "--out",
                    "reports/high_transfer_multi_turn_conversation.json",
                    "--markdown-out",
                    "reports/high_transfer_multi_turn_conversation.md",
                    "--session-prefix",
                    "overnight_conversation_lane",
                ],
                "hook_target": "conversation_training",
            }
        if concept == "multi_turn_conversation_hard":
            return {
                "allowed": True,
                "reason": "direct_high_transfer_multi_turn_conversation_hard",
                "concept": concept,
                "evidence_paths": [
                    "reports/high_transfer_multi_turn_conversation_hard.json",
                    "reports/cross_domain_sts_capsules.json",
                ],
                "command": [
                    sys.executable,
                    "scripts/multi_turn_conversation_benchmark.py",
                    "--suite-mode",
                    "hard",
                    "--case-limit",
                    "96",
                    "--out",
                    "reports/high_transfer_multi_turn_conversation_hard.json",
                    "--markdown-out",
                    "reports/high_transfer_multi_turn_conversation_hard.md",
                    "--session-prefix",
                    "hard_conversation_regression",
                ],
                "postprocess_commands": [cross_domain_capsule_command()],
                "hook_target": "conversation_training",
            }
        if concept == "multi_turn_conversation_hard_v2":
            return {
                "allowed": True,
                "reason": "direct_high_transfer_multi_turn_conversation_hard_v2",
                "concept": concept,
                "evidence_paths": [
                    "reports/high_transfer_multi_turn_conversation_hard_v2.json",
                    "reports/cross_domain_sts_capsules.json",
                ],
                "command": [
                    sys.executable,
                    "scripts/multi_turn_conversation_benchmark.py",
                    "--suite-mode",
                    "hard_v2",
                    "--case-limit",
                    "128",
                    "--out",
                    "reports/high_transfer_multi_turn_conversation_hard_v2.json",
                    "--markdown-out",
                    "reports/high_transfer_multi_turn_conversation_hard_v2.md",
                    "--session-prefix",
                    "hard_conversation_v2_frontier",
                ],
                "postprocess_commands": [cross_domain_capsule_command()],
                "hook_target": "conversation_training",
            }
        if concept == "multi_turn_conversation_hard_v3":
            return {
                "allowed": True,
                "reason": "direct_high_transfer_multi_turn_conversation_hard_v3",
                "concept": concept,
                "evidence_paths": [
                    "reports/high_transfer_multi_turn_conversation_hard_v3.json",
                    "reports/cross_domain_sts_capsules.json",
                ],
                "command": [
                    sys.executable,
                    "scripts/multi_turn_conversation_benchmark.py",
                    "--suite-mode",
                    "hard_v3",
                    "--case-limit",
                    "256",
                    "--out",
                    "reports/high_transfer_multi_turn_conversation_hard_v3.json",
                    "--markdown-out",
                    "reports/high_transfer_multi_turn_conversation_hard_v3.md",
                    "--session-prefix",
                    "hard_conversation_v3_product_frontier",
                ],
                "postprocess_commands": [cross_domain_capsule_command()],
                "hook_target": "conversation_training",
            }
        if concept == "multi_turn_conversation_hard_v4":
            return {
                "allowed": True,
                "reason": "direct_high_transfer_multi_turn_conversation_hard_v4",
                "concept": concept,
                "evidence_paths": [
                    "reports/high_transfer_multi_turn_conversation_hard_v4.json",
                    "reports/cross_domain_sts_capsules.json",
                ],
                "command": [
                    sys.executable,
                    "scripts/multi_turn_conversation_benchmark.py",
                    "--suite-mode",
                    "hard_v4",
                    "--case-limit",
                    "384",
                    "--out",
                    "reports/high_transfer_multi_turn_conversation_hard_v4.json",
                    "--markdown-out",
                    "reports/high_transfer_multi_turn_conversation_hard_v4.md",
                    "--session-prefix",
                    "hard_conversation_v4_a_plus_frontier",
                ],
                "postprocess_commands": [cross_domain_capsule_command()],
                "hook_target": "conversation_training",
            }
        if concept == "private_type_shape_receiver_veto_ablation":
            return {
                "allowed": True,
                "reason": "direct_private_type_shape_receiver_veto_ablation",
                "concept": concept,
                "evidence_paths": [
                    "reports/private_type_shape_receiver_ablation.json",
                    "reports/private_type_shape_receiver_ablation.md",
                    "reports/code_lm_private_candidates.jsonl",
                    "reports/teacher_public_transfer_residual_last.json",
                ],
                "command": [
                    sys.executable,
                    "scripts/private_type_shape_receiver_ablation.py",
                    "--max-tasks",
                    "192",
                    "--out",
                    "reports/private_type_shape_receiver_ablation.json",
                    "--markdown-out",
                    "reports/private_type_shape_receiver_ablation.md",
                ],
                "hook_target": "training_launch",
            }
        if concept == "decoder_v2_private_ablation_gate":
            if private_pressure_private_closure_needed():
                return {
                    "allowed": False,
                    "reason": "decoder_v2_private_ablation_gate_blocked_until_fresh_private_pressure_closure",
                    "concept": concept,
                    "command": [],
                    "evidence_paths": [
                        "reports/code_lm_closure_private_pressure_private.json",
                        "reports/high_transfer_curriculum_scheduler.json",
                    ],
                }
            return {
                "allowed": True,
                "reason": "direct_decoder_v2_private_ablation_gate",
                "concept": concept,
                "evidence_paths": [
                    "reports/decoder_v2_private_ablation_gate.json",
                    "reports/decoder_v2_private_ablation_gate.md",
                ],
                "command": [
                    sys.executable,
                    "scripts/decoder_v2_private_ablation_gate.py",
                    "--out",
                    "reports/decoder_v2_private_ablation_gate.json",
                    "--markdown-out",
                    "reports/decoder_v2_private_ablation_gate.md",
                ],
                "hook_target": "training_launch",
            }
        if concept == "repo_repair":
            return {
                "allowed": True,
                "reason": "direct_high_transfer_repo_repair",
                "concept": concept,
                "evidence_paths": [
                    "reports/high_transfer_repo_repair_learner.json",
                    "reports/cross_domain_sts_capsules.json",
                ],
                "command": [
                    sys.executable,
                    "scripts/viea_repo_repair_learner.py",
                    "--max-tasks",
                    "160",
                    "--out",
                    "reports/high_transfer_repo_repair_learner.json",
                    "--markdown-out",
                    "reports/high_transfer_repo_repair_learner.md",
                    "--trace-out",
                    "reports/high_transfer_repo_repair_training_traces.jsonl",
                    "--checkpoint-out",
                    "reports/high_transfer_repo_repair_trace_checkpoint.json",
                ],
                "postprocess_commands": [cross_domain_capsule_command()],
                "hook_target": "repo_repair_training",
            }
        if concept == "long_horizon_tool_use":
            return {
                "allowed": True,
                "reason": "direct_high_transfer_long_horizon_tool_use",
                "concept": concept,
                "evidence_paths": [
                    "reports/high_transfer_long_horizon_tool_use.json",
                    "reports/cross_domain_sts_capsules.json",
                ],
                "command": [
                    sys.executable,
                    "scripts/long_horizon_tool_use_benchmark.py",
                    "--out",
                    "reports/high_transfer_long_horizon_tool_use.json",
                    "--markdown-out",
                    "reports/high_transfer_long_horizon_tool_use.md",
                    "--trace-out",
                    "D:/ProjectTheseus/training_data/tool_use/private_train/long_horizon_tool_use_traces.jsonl",
                    "--sts-out",
                    "D:/ProjectTheseus/training_data/tool_use/sts/long_horizon_tool_use_sts.jsonl",
                ],
                "postprocess_commands": [cross_domain_capsule_command()],
                "hook_target": "long_horizon_tool_use",
            }
        if concept == "board_game_rl":
            return {
                "allowed": True,
                "reason": "direct_high_transfer_board_game_rl",
                "concept": concept,
                "evidence_paths": [
                    "reports/board_game_rl_benchmark.json",
                    "reports/board_game_elo_ratings.json",
                    "reports/rl_benchmark_registry.json",
                    "reports/cross_domain_sts_capsules.json",
                    "D:/ProjectTheseus/training_data/board_game_rl/private_train/board_game_policy_rows.jsonl",
                ],
                "command": [
                    sys.executable,
                    "scripts/board_game_rl_benchmark.py",
                    "--games",
                    "chess,go",
                    "--seed",
                    "14",
                    "--chess-games",
                    "24",
                    "--go-games",
                    "24",
                    "--go-board-size",
                    "5",
                    "--out",
                    "reports/board_game_rl_benchmark.json",
                    "--markdown-out",
                    "reports/board_game_rl_benchmark.md",
                    "--ratings-out",
                    "reports/board_game_elo_ratings.json",
                    "--trace-out",
                    "reports/board_game_rl_traces.jsonl",
                    "--learned-policy-out",
                    "reports/board_game_learned_policy.json",
                    "--policy-train-out",
                    "D:/ProjectTheseus/training_data/board_game_rl/private_train/board_game_policy_rows.jsonl",
                ],
                "postprocess_commands": [cross_domain_capsule_command()],
                "hook_target": "training_launch",
            }
        if concept == "pufferlib4_rl":
            return {
                "allowed": True,
                "reason": "direct_high_transfer_pufferlib4_rl_lane",
                "concept": concept,
                "evidence_paths": [
                    "reports/pufferlib4_capability_probe.json",
                    "reports/pufferlib4_rl_lane.json",
                    "reports/cross_domain_sts_capsules.json",
                    "D:/ProjectTheseus/training_data/rl/pufferlib4/pufferlib4_rl_sts_capsules.jsonl",
                ],
                "command": [
                    sys.executable,
                    "scripts/pufferlib4_rl_lane.py",
                    "--probe",
                    "--out",
                    "reports/pufferlib4_rl_lane.json",
                    "--markdown-out",
                    "reports/pufferlib4_rl_lane.md",
                    "--capsules-out",
                    "D:/ProjectTheseus/training_data/rl/pufferlib4/pufferlib4_rl_sts_capsules.jsonl",
                ],
                "postprocess_commands": [cross_domain_capsule_command()],
                "hook_target": "training_launch",
            }
        if concept == "cross_domain_sts_capsules":
            return {
                "allowed": True,
                "reason": "direct_high_transfer_cross_domain_sts_capsules",
                "concept": concept,
                "evidence_paths": [
                    "reports/cross_domain_sts_capsules.json",
                    "data/training_sources/cross_domain_sts_capsules.jsonl",
                    "D:/ProjectTheseus/training_data/cross_domain_sts/cross_domain_sts_streams.jsonl",
                ],
                "command": cross_domain_capsule_command(),
                "hook_target": "training_launch",
            }
        return {
            "allowed": True,
            "reason": "refresh_high_transfer_scheduler",
            "command": [
                sys.executable,
                "scripts/high_transfer_curriculum_scheduler.py",
                "--out",
                "reports/high_transfer_curriculum_scheduler.json",
                "--markdown-out",
                "reports/high_transfer_curriculum_scheduler.md",
                "--tasks-out",
                "reports/high_transfer_curriculum_tasks.jsonl",
            ],
            "hook_target": "training_launch",
        }
    if source == "auto_teacher_escalation" or kind == "teacher_architecture_escalation":
        command = [
            sys.executable,
            "scripts/teacher_architect_experiment_runner.py",
            "--execute",
            "--max-experiments",
            "1",
            "--max-steps",
            "2",
            "--timeout-seconds",
            str(max(60, int(args.timeout_seconds))),
            "--out",
            "reports/hive_teacher_auto_escalation.json",
            "--markdown-out",
            "reports/hive_teacher_auto_escalation.md",
        ]
        if args.allow_teacher:
            command.append("--allow-teacher")
        return {
            "allowed": True,
            "reason": "teacher_architecture_auto_escalation",
            "evidence_paths": ["reports/hive_teacher_auto_escalation.json"],
            "command": command,
            "hook_target": "teacher_call",
        }
    return {"allowed": False, "reason": f"unsupported_board_task:{source}:{kind}", "command": []}
