#!/usr/bin/env python3
"""Private residual self-improvement ratchet for the current transfer wall.

This script turns the already-spent public residual summary into a private-only
action queue. It is deliberately not a public-calibration runner and not a
teacher-apply path. Its job is to keep the self-improvement loop pointed at the
real wall: broad transfer remains below floor, so the next safe action is more
private residual work, not another public run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

POLICY = "project_theseus_private_residual_self_improvement_ratchet_v1"
PUBLIC_FLOOR = 0.70

DEFAULT_OUT = REPORTS / "private_residual_self_improvement_ratchet_v1.json"
DEFAULT_MARKDOWN = REPORTS / "private_residual_self_improvement_ratchet_v1.md"
DEFAULT_QUEUE = REPORTS / "private_residual_self_improvement_ratchet_v1_queue.jsonl"

PUBLIC_LOCK = REPORTS / "public_calibration_operator_lock.flag"
PUBLIC_RESIDUAL = REPORTS / "public_code_transfer_residual_report_wide_public_seed23_5x32_interface_floor_v1.json"
GOVERNOR = REPORTS / "theseus_generalization_governor_v1.json"
TEACHER_PREFLIGHT = REPORTS / "full_training_teacher_preflight.json"
READINESS_PACKET = REPORTS / "public_calibration_readiness_packet.json"
V5_REFRESH = REPORTS / "private_ecology_generalization_v5_refresh.json"
POST_V4_AUTOPILOT = REPORTS / "post_v4_generalization_autopilot_v1.json"
OPERATOR_EXECUTE = REPORTS / "operator_bounded_public_calibration_execute.json"
OPERATOR_APPROVAL = REPORTS / "public_calibration_operator_approval_post_v4_seed23_5x32.json"
TARGETED_V3 = REPORTS / "targeted_private_residual_curriculum_v3.json"
TRAIN_ONCE_V3 = REPORTS / "code_lm_train_once_fanout_private_residual_repair_v3_post_v4.json"
HELDOUT_FANOUT_V3 = REPORTS / "code_lm_closure_rust_private_residual_repair_v3_post_v4_heldout_targeted_fanout.json"
HELDOUT_CANDIDATES_V3 = REPORTS / "code_lm_private_candidates_private_residual_repair_v3_post_v4_heldout_targeted.jsonl"
HELDOUT_SCORE_V3 = REPORTS / "private_residual_repair_v3_heldout_score.json"
PRIVATE_RESIDUAL_V3_GATE = REPORTS / "private_residual_repair_v3_gate.json"
PRIVATE_RESIDUAL_V3_HELDOUT = ROOT / "data" / "training_data" / "high_transfer" / "private_eval" / "private_residual_repair_v3_heldout_code_lm_tasks.jsonl"
PRIVATE_ONLY_EMPTY_PUBLIC = REPORTS / "code_lm_public_tasks_private_residual_repair_v3_post_v4_private_only_empty.jsonl"
PRIVATE_RESIDUAL_V3_CHECKPOINT = REPORTS / "student_code_lm_checkpoint_private_residual_repair_v3_post_v4.json"
PRIVATE_RESIDUAL_V3_STS_STREAMS = REPORTS / "code_lm_sts_public_generations_private_residual_repair_v3_post_v4.jsonl"
HELDOUT_EMPTY_PUBLIC_CANDIDATES_V3 = REPORTS / "student_code_candidates_private_residual_repair_v3_post_v4_heldout_targeted_private_only_empty.jsonl"
STUDENT_REPAIR_CANDIDATES_V3 = REPORTS / "code_lm_private_candidates_private_residual_repair_v3_student_repair.jsonl"
STUDENT_REPAIR_CONTROL_CANDIDATES_V3 = REPORTS / "code_lm_private_candidates_private_residual_repair_v3_student_repair_sts_off_control.jsonl"
ARTIFACT_DERIVED_ACTION_KINDS = {
    "materialize_private_residual_v3_candidate_floor",
    "run_private_residual_shadow_autopilot",
    "train_private_residual_v3_candidate_floor",
    "fanout_private_residual_v3_heldout_targeted",
    "score_private_residual_v3_heldout_targeted",
    "refresh_private_residual_v3_gate",
}

FORBIDDEN_POST_V4_PUBLIC_ARTIFACTS = [
    REPORTS / "real_code_benchmark_graduation_post_v4_seed23_5x32.json",
    REPORTS / "real_code_benchmark_traces_post_v4_seed23_5x32.jsonl",
    REPORTS / "student_code_candidates_post_v4_seed23_5x32.jsonl",
    REPORTS / "operator_bounded_public_calibration_post_v4_seed23_5x32.json",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-actions", type=int, default=1)
    parser.add_argument("--public-residual", default=rel(PUBLIC_RESIDUAL))
    parser.add_argument("--governor", default=rel(GOVERNOR))
    parser.add_argument("--teacher-preflight", default=rel(TEACHER_PREFLIGHT))
    parser.add_argument("--readiness-packet", default=rel(READINESS_PACKET))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--queue-out", default=rel(DEFAULT_QUEUE))
    args = parser.parse_args()

    report = build_report(args)
    if args.execute and report["decision"]["kind"] == "retry_private":
        report["execution"] = execute_queue(report["queue"], max_actions=max(0, int(args.max_actions)))
        report["queue"] = apply_execution_results(report["queue"], report["execution"])
        refresh_queue_summary(report)
        if any(row.get("returncode") not in {0, None} for row in report["execution"]["actions"]):
            report["trigger_state"] = "YELLOW"

    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    write_jsonl(resolve(args.queue_out), report["queue"])
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report["trigger_state"] == "RED" else 0


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    residual_path = resolve(args.public_residual)
    governor_path = resolve(args.governor)
    teacher_path = resolve(args.teacher_preflight)
    readiness_path = resolve(args.readiness_packet)

    residual = read_json(residual_path, {})
    governor = read_json(governor_path, {})
    teacher = read_json(teacher_path, {})
    readiness = read_json(readiness_path, {})

    residual_summary = object_field(residual, "summary")
    governor_summary = object_field(governor, "summary")
    teacher_summary = object_field(teacher, "summary")
    readiness_summary = object_field(readiness, "summary")

    public_pass_rate = number(
        governor_summary.get("public_pass_rate"),
        residual_summary.get("real_public_task_pass_rate"),
        residual_summary.get("multi_stream_pass_rate"),
        0.0,
    )
    public_task_count = int(number(governor_summary.get("public_task_count"), residual_summary.get("public_task_count"), 0))
    cards_below_floor = as_list(governor_summary.get("cards_below_floor") or readiness_summary.get("cards_below_floor"))
    dominant = normalize_residual_categories(
        residual_summary.get("adapter_adjusted_dominant_categories")
        or residual_summary.get("dominant_categories")
        or []
    )
    raw_residuals = normalize_residual_categories(residual_summary.get("raw_residual_counts") or [])
    post_v4_state = post_v4_public_artifact_state()
    queue = build_private_queue(
        dominant,
        residual_report=rel(residual_path),
        real_code_report=str(residual.get("calibration_report") or "reports/real_code_benchmark_graduation_post_v4_seed23_5x32.json"),
        governor_summary=governor_summary,
        teacher_summary=teacher_summary,
    )
    completed = completed_action_kinds(read_json(resolve(args.out), {})) | artifact_completed_action_kinds()
    queue = apply_completed_kinds(queue, completed)
    pending_queue = [row for row in queue if row.get("status") == "pending"]
    gates = [
        gate("public_calibration_operator_lock_active", PUBLIC_LOCK.exists(), rel(PUBLIC_LOCK), "hard"),
        gate("post_v4_public_artifacts_approved_or_absent", post_v4_state["allowed"], post_v4_state, "hard"),
        gate("public_residual_report_green", residual.get("trigger_state") == "GREEN", residual.get("trigger_state"), "hard"),
        gate(
            "public_residual_report_is_aggregate_only",
            residual_summary.get("public_tests_or_solutions_embedded") is False
            and residual_summary.get("public_prompts_embedded") is False,
            {
                "public_tests_or_solutions_embedded": residual_summary.get("public_tests_or_solutions_embedded"),
                "public_prompts_embedded": residual_summary.get("public_prompts_embedded"),
            },
            "hard",
        ),
        gate(
            "public_calibration_stays_disallowed",
            governor_summary.get("public_calibration_allowed") is False
            and bool(readiness.get("public_calibration_allowed")) is False,
            {
                "governor_public_calibration_allowed": governor_summary.get("public_calibration_allowed"),
                "readiness_public_calibration_allowed": readiness.get("public_calibration_allowed"),
            },
            "hard",
        ),
        gate(
            "teacher_remains_proposal_only",
            teacher.get("ok") is True
            and int(number(teacher_summary.get("blocker_count"), 999)) == 0
            and teacher_summary.get("apply_mode_blocked") is True
            and teacher_summary.get("worker_teacher_invariant") is True,
            {
                "teacher_trigger_state": teacher.get("trigger_state"),
                "teacher_live_status": teacher_summary.get("teacher_live_status"),
                "apply_mode_blocked": teacher_summary.get("apply_mode_blocked"),
                "worker_teacher_invariant": teacher_summary.get("worker_teacher_invariant"),
            },
            "warning",
        ),
        gate("spent_public_floor_not_cleared", public_pass_rate < PUBLIC_FLOOR or bool(cards_below_floor), {
            "public_pass_rate": public_pass_rate,
            "floor": PUBLIC_FLOOR,
            "cards_below_floor": cards_below_floor,
        }, "warning"),
        gate("private_queue_safe", all(row["safe_to_execute_without_operator_public_approval"] for row in queue), len(queue), "hard"),
        gate("external_inference_zero", external_call_total(residual, governor, teacher, readiness) == 0, external_call_total(residual, governor, teacher, readiness), "hard"),
    ]
    hard_failed = [row for row in gates if row["severity"] == "hard" and not row["passed"]]
    warning_failed = [row for row in gates if row["severity"] == "warning" and not row["passed"]]
    decision = choose_decision(hard_failed, public_pass_rate, cards_below_floor)
    trigger_state = "RED" if hard_failed else "YELLOW" if warning_failed or decision["kind"] == "retry_private" else "GREEN"
    return {
        "policy": POLICY,
        "created_utc": now(),
        "trigger_state": trigger_state,
        "decision": decision,
        "summary": {
            "decision": decision["kind"],
            "decision_reason": decision["reason"],
            "public_pass_rate": public_pass_rate,
            "public_floor": PUBLIC_FLOOR,
            "public_task_count": public_task_count,
            "cards_below_floor": cards_below_floor,
            "dominant_residual_categories": dominant,
            "raw_residual_counts": raw_residuals,
            "top_private_action": pending_queue[0]["kind"] if pending_queue else "",
            "queue_item_count": len(queue),
            "pending_queue_item_count": len(pending_queue),
            "completed_queue_item_count": sum(1 for row in queue if row.get("status") == "completed"),
            "safe_queue_item_count": sum(1 for row in queue if row["safe_to_execute_without_operator_public_approval"]),
            "operator_lock_active": PUBLIC_LOCK.exists(),
            "public_calibration_allowed": False,
            "teacher_preflight_ok": teacher.get("ok") is True and int(number(teacher_summary.get("blocker_count"), 999)) == 0,
            "teacher_live_status": teacher_summary.get("teacher_live_status"),
            "public_residual_report": rel(residual_path),
            "governor_report": rel(governor_path),
            "score_semantics": "private residual ratchet only; not promotion evidence and not public calibration",
            "external_inference_calls": 0,
        },
        "inputs": {
            "public_residual": rel(residual_path),
            "governor": rel(governor_path),
            "teacher_preflight": rel(teacher_path),
            "readiness_packet": rel(readiness_path),
            "public_tests_used": False,
            "public_solutions_used": False,
            "public_prompts_used": False,
            "public_traces_used_for_training": False,
        },
        "queue": queue,
        "gates": gates,
        "rules": {
            "public_calibration": "Never executed by this ratchet; the public lock must remain active.",
            "training": "Use aggregate residual categories only to choose private curricula; public prompts/tests/solutions/traces are not rows.",
            "teacher": "Teacher remains proposal-only and is not called unless a separate explicit live-teacher command is run.",
            "promotion": "Private queue success is not promotion; it only earns refreshed private evidence and possible operator review.",
        },
        "external_inference_calls": 0,
    }


def choose_decision(hard_failed: list[dict[str, Any]], public_pass_rate: float, cards_below_floor: list[Any]) -> dict[str, Any]:
    if hard_failed:
        return {
            "kind": "stop_blocker",
            "reason": "A hard safety or evidence boundary failed; do not train or calibrate until it is fixed.",
            "public_calibration_allowed": False,
        }
    if public_pass_rate >= PUBLIC_FLOOR and not cards_below_floor:
        return {
            "kind": "operator_review_public_calibration",
            "reason": "Public floor appears clear; only an explicit operator public-unlock path may spend calibration.",
            "public_calibration_allowed": False,
        }
    return {
        "kind": "retry_private",
        "reason": "The spent public score remains below floor, so the next safe step is private residual work.",
        "public_calibration_allowed": False,
    }


def build_private_queue(
    dominant: list[list[Any]],
    *,
    residual_report: str,
    real_code_report: str,
    governor_summary: dict[str, Any],
    teacher_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    categories = [str(row[0]) for row in dominant]
    rows: list[dict[str, Any]] = []
    if "no_admissible_candidate_regression" in categories:
        rows.append(
            queue_item(
                "materialize_private_residual_v3_candidate_floor",
                "Generate the canonical private residual v3 rows from aggregate no-admissible public residual categories.",
                [
                    sys.executable,
                    "scripts/targeted_private_residual_curriculum_v3.py",
                    "--residual-report",
                    residual_report,
                    "--real-code-report",
                    real_code_report,
                    "--rows-per-family",
                    "192",
                    "--heldout-rows-per-family",
                    "48",
                    "--out",
                    "reports/targeted_private_residual_curriculum_v3.json",
                    "--markdown-out",
                    "reports/targeted_private_residual_curriculum_v3.md",
                ],
                priority=5,
            )
        )
        rows.append(
            queue_item(
                "train_private_residual_v3_candidate_floor",
                "Run private-only train-once/fanout against the refreshed v3 rows; public calibration stays locked.",
                [
                    sys.executable,
                    "scripts/code_lm_train_once_fanout.py",
                    "--execute",
                    "--slug",
                    "private_residual_repair_v3_post_v4",
                    "--private-only",
                    "--private-count",
                    "960",
                    "--epochs",
                    "8",
                    "--candidates-per-task",
                    "4",
                    "--max-high-transfer-private-train",
                    "14400",
                    "--max-rust-work-steps",
                    "12000000",
                    "--rust-timeout-seconds",
                    "7200",
                    "--sts-timeout-seconds",
                    "7200",
                    "--fanout-timeout-seconds",
                    "7200",
                    "--refresh-private-eval-limit",
                    "240",
                    "--out",
                    "reports/code_lm_train_once_fanout_private_residual_repair_v3_post_v4.json",
                    "--markdown-out",
                    "reports/code_lm_train_once_fanout_private_residual_repair_v3_post_v4.md",
                ],
                priority=10,
            )
        )
        rows.append(
            queue_item(
                "refresh_private_residual_v3_gate",
                "Refresh the v3 gate after private-only training and decoder/STS evidence; do not unlock public calibration.",
                [
                    sys.executable,
                    "scripts/private_residual_repair_v3_gate.py",
                    "--curriculum",
                    "reports/targeted_private_residual_curriculum_v3.json",
                    "--private-heldout",
                    "reports/private_residual_repair_v3_heldout_score.json",
                    "--decoder-gate",
                    "reports/code_lm_train_once_fanout_private_residual_repair_v3_post_v4.json",
                    "--sts-causal",
                    "reports/private_residual_repair_v3_heldout_score.json",
                    "--out",
                    "reports/private_residual_repair_v3_gate.json",
                    "--markdown-out",
                    "reports/private_residual_repair_v3_gate.md",
                ],
                priority=15,
            )
        )
        rows.append(
            queue_item(
                "fanout_private_residual_v3_heldout_targeted",
                "Generate candidates from the v3 checkpoint against the actual private v3 heldout rows; public manifest is empty.",
                [
                    release_binary_command(),
                    "generate-code-lm-closure-fanout",
                    "--private-curriculum",
                    rel(PRIVATE_RESIDUAL_V3_HELDOUT),
                    "--public-task-manifest",
                    rel(PRIVATE_ONLY_EMPTY_PUBLIC),
                    "--checkpoint-in",
                    rel(PRIVATE_RESIDUAL_V3_CHECKPOINT),
                    "--seed",
                    "14",
                    "--candidates-per-task",
                    "4",
                    "--private-candidate-out",
                    rel(HELDOUT_CANDIDATES_V3),
                    "--public-candidate-out",
                    rel(HELDOUT_EMPTY_PUBLIC_CANDIDATES_V3),
                    "--report-out",
                    rel(HELDOUT_FANOUT_V3),
                    "--sts-streams",
                    rel(PRIVATE_RESIDUAL_V3_STS_STREAMS),
                    "--private-eval-limit",
                    "240",
                    "--public-task-limit",
                    "0",
                ],
                priority=12,
            )
        )
        rows.append(
            queue_item(
                "score_private_residual_v3_heldout_targeted",
                "Score the learned structural private v3 candidates against private synthetic heldout tests with diagnostic adapters excluded from pass credit.",
                [
                    sys.executable,
                    "scripts/private_residual_repair_v3_heldout_score.py",
                    "--heldout",
                    rel(PRIVATE_RESIDUAL_V3_HELDOUT),
                    "--candidates",
                    rel(STUDENT_REPAIR_CANDIDATES_V3),
                    "--control-candidates",
                    rel(STUDENT_REPAIR_CONTROL_CANDIDATES_V3),
                    "--adapter-off",
                    "--out",
                    rel(HELDOUT_SCORE_V3),
                    "--markdown-out",
                    "reports/private_residual_repair_v3_heldout_score.md",
                ],
                priority=13,
            )
        )
    if "verifier_mismatch" in categories:
        rows.append(
            queue_item(
                "run_private_residual_shadow_autopilot",
                "Regenerate and score private shadow residual tasks against the verifier-mismatch-heavy public residual summary.",
                [
                    sys.executable,
                    "scripts/post_v4_generalization_autopilot_v1.py",
                    "--execute",
                    "--train-rows",
                    "12000",
                    "--heldout-rows",
                    "2400",
                    "--max-hours",
                    "6",
                ],
                priority=10,
            )
        )
    if "return_shape" in categories or "no_admissible_candidate_regression" in categories:
        rows.append(
            queue_item(
                "refresh_semantic_alias_transfer_gate",
                "Rerun the full private semantic-alias gate so exact family lookup is not mistaken for residual transfer.",
                [
                    sys.executable,
                    "scripts/broad_private_semantic_alias_gate_v1.py",
                    "--execute",
                    "--task-limit",
                    "0",
                    "--min-alias-rows",
                    "1008",
                ],
                priority=20,
            )
        )
    if "algorithmic_planning" in categories or "verifier_mismatch" in categories:
        rows.append(
            queue_item(
                "refresh_novel_composition_transfer_gate",
                "Rerun the full private novel-composition gate to prove reusable two-step token composition.",
                [
                    sys.executable,
                    "scripts/broad_private_novel_composition_gate_v1.py",
                    "--execute",
                    "--rows",
                    "1008",
                    "--min-rows",
                    "1008",
                ],
                priority=30,
            )
        )
    rows.append(
        queue_item(
            "refresh_private_ecology_v5",
            "Run the full private v5 ecology refresh so Hive/project workflow pressure has current fanout, score, and learned-only evidence.",
            [
                sys.executable,
                "scripts/private_ecology_generalization_v5_refresh.py",
                "--execute",
                "--train-rows",
                "1200",
                "--heldout-rows",
                "480",
                "--private-eval-limit",
                "480",
                "--max-hours",
                "6",
            ],
            priority=40,
        )
    )
    if teacher_summary.get("teacher_live_status") != "completed":
        rows.append(
            queue_item(
                "teacher_preflight_proposal_only_no_live",
                "Refresh teacher governance without a live teacher call; teacher remains proposal-only.",
                [
                    sys.executable,
                    "scripts/full_training_teacher_preflight.py",
                    "--profile",
                    "smoke",
                    "--skip-autonomy-readiness",
                    "--out",
                    "reports/full_training_teacher_preflight.json",
                    "--markdown-out",
                    "reports/full_training_teacher_preflight.md",
                ],
                priority=50,
            )
        )
    rows.append(
        queue_item(
            "refresh_generalization_governor",
            "Refresh the governor after private residual work; public calibration must remain locked.",
            [
                sys.executable,
                "scripts/theseus_generalization_governor_v1.py",
                "--out",
                "reports/theseus_generalization_governor_v1.json",
                "--markdown-out",
                "reports/theseus_generalization_governor_v1.md",
                "--queue-out",
                "reports/theseus_generalization_governor_v1_queue.jsonl",
            ],
            priority=90,
        )
    )
    return sorted(dedupe_queue(rows), key=lambda row: (int(row["priority"]), str(row["kind"])))


def execute_queue(queue: list[dict[str, Any]], *, max_actions: int) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    for item in [row for row in queue if row.get("status") == "pending"][:max_actions]:
        if not item.get("safe_to_execute_without_operator_public_approval"):
            actions.append({"kind": item.get("kind"), "returncode": None, "status": "skipped_unsafe"})
            continue
        command = [str(part) for part in item.get("command") or []]
        if not command:
            actions.append({"kind": item.get("kind"), "returncode": None, "status": "skipped_no_command"})
            continue
        started = time.time()
        proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=6 * 3600)
        actions.append(
            {
                "kind": item.get("kind"),
                "command": command,
                "returncode": proc.returncode,
                "runtime_seconds": round(time.time() - started, 3),
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-4000:],
                "status": "completed" if proc.returncode == 0 else "failed",
            }
        )
    return {
        "policy": "project_theseus_private_residual_self_improvement_ratchet_v1_execution",
        "created_utc": now(),
        "max_actions": max_actions,
        "actions": actions,
    }


def completed_action_kinds(previous: dict[str, Any]) -> set[str]:
    kinds: set[str] = set()
    for action in object_field(previous, "execution").get("actions", []) if isinstance(object_field(previous, "execution").get("actions"), list) else []:
        if isinstance(action, dict) and action.get("status") == "completed" and int(number(action.get("returncode"), 1)) == 0:
            kind = str(action.get("kind") or "")
            if kind not in ARTIFACT_DERIVED_ACTION_KINDS:
                kinds.add(kind)
    for row in previous.get("queue", []) if isinstance(previous.get("queue"), list) else []:
        if isinstance(row, dict) and row.get("status") == "completed":
            kind = str(row.get("kind") or "")
            if kind not in ARTIFACT_DERIVED_ACTION_KINDS:
                kinds.add(kind)
    kinds.discard("")
    return kinds


def artifact_completed_action_kinds() -> set[str]:
    kinds: set[str] = set()
    post_v4_autopilot = read_json(POST_V4_AUTOPILOT, {})
    post_v4_summary = object_field(post_v4_autopilot, "summary")
    post_v4_gate_status = {
        str(row.get("gate") or ""): row.get("passed") is True
        for row in as_list(post_v4_autopilot.get("gates"))
        if isinstance(row, dict)
    }
    if (
        post_v4_autopilot.get("trigger_state") == "GREEN"
        and post_v4_summary.get("completion_evidence_status") == "post_v4_private_learned_shadow_ready"
        and int(number(post_v4_summary.get("private_heldout_row_count"), 0)) >= 2400
        and float(number(post_v4_summary.get("learned_only_pass_rate"), 0.0)) >= 0.95
        and int(number(post_v4_summary.get("prototype_pass_count"), 999)) == 0
        and int(number(post_v4_summary.get("external_inference_calls"), 0)) == 0
        and object_field(post_v4_summary, "public_score_unchanged").get("operator_lock_active") is True
        and post_v4_gate_status.get("fanout_sts_on_private_only") is True
        and post_v4_gate_status.get("fanout_sts_off_control_private_only") is True
        and post_v4_gate_status.get("external_inference_zero") is True
    ):
        kinds.add("run_private_residual_shadow_autopilot")
    targeted_v3 = read_json(TARGETED_V3, {})
    targeted_summary = object_field(targeted_v3, "summary")
    if (
        targeted_v3.get("trigger_state") == "GREEN"
        and int(number(targeted_summary.get("private_train_row_count"), 0)) >= 960
        and int(number(targeted_summary.get("private_heldout_row_count"), 0)) >= 240
        and int(number(targeted_summary.get("private_train_solution_failures"), 1)) == 0
        and int(number(targeted_summary.get("private_heldout_solution_failures"), 1)) == 0
        and int(number(targeted_summary.get("external_inference_calls"), 0)) == 0
    ):
        kinds.add("materialize_private_residual_v3_candidate_floor")
    train_once_v3 = read_json(TRAIN_ONCE_V3, {})
    train_once_summary = object_field(train_once_v3, "summary")
    private_manifest = object_field(train_once_summary, "private_candidate_manifest_diagnostics")
    public_manifest = object_field(train_once_summary, "public_candidate_manifest_diagnostics")
    if (
        train_once_v3.get("trigger_state") == "GREEN"
        and str(train_once_v3.get("run_status") or train_once_summary.get("run_status") or "").lower() == "completed"
        and (train_once_v3.get("private_only") is True or train_once_summary.get("private_only") is True)
        and float(number(private_manifest.get("task_coverage"), 0.0)) >= 0.97
        and int(number(private_manifest.get("full_body_candidate_count"), 0)) > 0
        and int(number(public_manifest.get("row_count"), 0)) == 0
        and int(number(train_once_v3.get("external_inference_calls"), train_once_summary.get("external_inference_calls"), 0)) == 0
    ):
        kinds.add("train_private_residual_v3_candidate_floor")
    heldout_fanout = read_json(HELDOUT_FANOUT_V3, {})
    heldout_summary = object_field(heldout_fanout, "summary")
    if (
        heldout_fanout.get("trigger_state") in {"GREEN", "YELLOW"}
        and str(heldout_fanout.get("run_status") or "").lower() == "completed"
        and int(number(heldout_summary.get("private_eval_task_count"), 0)) >= 240
        and int(number(heldout_summary.get("public_task_count"), 0)) == 0
        and count_jsonl_rows(HELDOUT_CANDIDATES_V3) > 0
        and count_jsonl_rows(HELDOUT_EMPTY_PUBLIC_CANDIDATES_V3) == 0
        and int(number(heldout_fanout.get("external_inference_calls"), heldout_summary.get("external_inference_calls"), 0)) == 0
    ):
        kinds.add("fanout_private_residual_v3_heldout_targeted")
    heldout_score = read_json(HELDOUT_SCORE_V3, {})
    heldout_score_summary = object_field(heldout_score, "summary")
    heldout_score_inputs = object_field(heldout_score, "inputs")
    if (
        heldout_score.get("trigger_state") == "GREEN"
        and heldout_score_inputs.get("candidates") == rel(STUDENT_REPAIR_CANDIDATES_V3)
        and heldout_score_inputs.get("control_candidates") == rel(STUDENT_REPAIR_CONTROL_CANDIDATES_V3)
        and int(number(heldout_score_inputs.get("task_limit"), heldout_score_summary.get("private_residual_v3_heldout_task_limit"), 0)) == 0
        and int(number(heldout_score_summary.get("private_residual_v3_heldout_task_count"), 0)) >= 240
        and int(number(heldout_score_summary.get("candidate_row_count"), 0)) > 0
        and heldout_score_summary.get("adapter_off_scoring") is True
        and float(number(heldout_score_summary.get("learned_candidate_task_pass_rate"), 0.0)) >= 0.70
        and int(number(heldout_score_summary.get("diagnostic_adapter_task_passes"), 0)) == 0
        and heldout_score_summary.get("public_tests_used") is False
        and heldout_score_summary.get("public_solutions_used") is False
        and int(number(heldout_score.get("external_inference_calls"), heldout_score_summary.get("external_inference_calls"), 0)) == 0
    ):
        kinds.add("score_private_residual_v3_heldout_targeted")
    v3_gate = read_json(PRIVATE_RESIDUAL_V3_GATE, {})
    v3_gate_summary = object_field(v3_gate, "summary")
    v3_gate_inputs = object_field(v3_gate, "inputs")
    if (
        v3_gate.get("trigger_state") == "GREEN"
        and v3_gate_summary.get("full_heldout_score_present") is True
        and int(number(v3_gate_summary.get("failed_gate_count"), 1)) == 0
        and int(number(v3_gate_summary.get("pending_gate_count"), 1)) == 0
        and v3_gate_summary.get("private_heldout_adapter_off_scoring") is True
        and float(number(v3_gate_summary.get("private_learned_candidate_pass_rate"), 0.0)) >= 0.70
        and v3_gate_inputs.get("private_heldout") == rel(HELDOUT_SCORE_V3)
        and v3_gate_inputs.get("decoder_gate") == rel(TRAIN_ONCE_V3)
    ):
        kinds.add("refresh_private_residual_v3_gate")
    v5_refresh = read_json(V5_REFRESH, {})
    v5_summary = object_field(v5_refresh, "summary")
    if (
        v5_refresh.get("trigger_state") == "GREEN"
        and v5_summary.get("completion_evidence_status") == "private_ecology_v5_learned_refresh_ready"
        and object_field(v5_summary, "freshness").get("fresh") is True
        and int(number(v5_summary.get("prototype_pass_count"), 999)) == 0
        and int(number(v5_summary.get("learned_token_pass_count"), 0)) >= 480
        and int(number(v5_summary.get("external_inference_calls"), 0)) == 0
    ):
        kinds.add("refresh_private_ecology_v5")
    return kinds


def apply_completed_kinds(queue: list[dict[str, Any]], completed: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in queue:
        item = dict(row)
        if str(item.get("kind") or "") in completed:
            item["status"] = "completed"
        out.append(item)
    return out


def apply_execution_results(queue: list[dict[str, Any]], execution: dict[str, Any]) -> list[dict[str, Any]]:
    status_by_kind: dict[str, str] = {}
    for action in execution.get("actions", []) if isinstance(execution.get("actions"), list) else []:
        if not isinstance(action, dict):
            continue
        kind = str(action.get("kind") or "")
        if not kind:
            continue
        status_by_kind[kind] = "completed" if action.get("returncode") == 0 and action.get("status") == "completed" else "failed"
    out: list[dict[str, Any]] = []
    for row in queue:
        item = dict(row)
        if str(item.get("kind") or "") in status_by_kind:
            item["status"] = status_by_kind[str(item.get("kind") or "")]
        out.append(item)
    return out


def refresh_queue_summary(report: dict[str, Any]) -> None:
    summary = object_field(report, "summary")
    queue = report.get("queue") if isinstance(report.get("queue"), list) else []
    pending = [row for row in queue if isinstance(row, dict) and row.get("status") == "pending"]
    summary["executed_action_count"] = len(object_field(report, "execution").get("actions", []) or [])
    summary["queue_item_count"] = len(queue)
    summary["pending_queue_item_count"] = len(pending)
    summary["completed_queue_item_count"] = sum(1 for row in queue if isinstance(row, dict) and row.get("status") == "completed")
    summary["safe_queue_item_count"] = sum(1 for row in queue if isinstance(row, dict) and row.get("safe_to_execute_without_operator_public_approval"))
    summary["top_private_action"] = str(pending[0].get("kind") or "") if pending else ""
    report["summary"] = summary


def queue_item(kind: str, title: str, command: list[str], *, priority: int) -> dict[str, Any]:
    return {
        "policy": "project_theseus_private_residual_self_improvement_queue_item_v1",
        "queue": "private_residual_self_improvement_ratchet_v1",
        "kind": kind,
        "title": title,
        "priority": int(priority),
        "status": "pending",
        "command": command,
        "safe_to_execute_without_operator_public_approval": True,
        "requires_operator_public_unlock": False,
        "public_calibration_allowed": False,
    }


def release_binary_command() -> str:
    name = "symliquid-cli.exe" if sys.platform.startswith("win") else "symliquid-cli"
    return str(Path("target") / "release" / name)


def count_jsonl_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def dedupe_queue(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("kind") or "")
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def render_markdown(report: dict[str, Any]) -> str:
    summary = object_field(report, "summary")
    lines = [
        "# Private Residual Self-Improvement Ratchet v1",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- decision: `{summary.get('decision')}`",
        f"- decision_reason: {summary.get('decision_reason')}",
        f"- public_pass_rate: `{summary.get('public_pass_rate')}` floor=`{summary.get('public_floor')}`",
        f"- public_task_count: `{summary.get('public_task_count')}`",
        f"- cards_below_floor: `{summary.get('cards_below_floor')}`",
        f"- dominant_residual_categories: `{summary.get('dominant_residual_categories')}`",
        f"- operator_lock_active: `{summary.get('operator_lock_active')}`",
        f"- public_calibration_allowed: `{summary.get('public_calibration_allowed')}`",
        f"- teacher_preflight_ok: `{summary.get('teacher_preflight_ok')}`",
        f"- queue_item_count: `{summary.get('queue_item_count')}`",
        "",
        "## Queue",
        "",
    ]
    for row in report.get("queue") or []:
        command = " ".join(str(part) for part in row.get("command") or [])
        lines.append(f"- `{row.get('kind')}` priority=`{row.get('priority')}` safe=`{row.get('safe_to_execute_without_operator_public_approval')}`")
        lines.append(f"  Command: `{command}`")
    lines.extend(["", "## Gates", ""])
    for row in report.get("gates") or []:
        lines.append(f"- `{row.get('gate')}`: `{row.get('passed')}` ({row.get('severity')})")
    lines.append("")
    return "\n".join(lines)


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def normalize_residual_categories(value: Any) -> list[list[Any]]:
    rows: list[list[Any]] = []
    if isinstance(value, dict):
        value = list(value.items())
    if not isinstance(value, list):
        return rows
    for row in value:
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            try:
                rows.append([str(row[0]), int(float(row[1]))])
            except (TypeError, ValueError):
                rows.append([str(row[0]), 0])
        elif isinstance(row, dict):
            key = str(row.get("category") or row.get("residual") or row.get("name") or "")
            if not key:
                continue
            rows.append([key, int(number(row.get("count"), 0))])
    return rows


def external_call_total(*reports: dict[str, Any]) -> int:
    total = 0
    for report in reports:
        total += int(number(report.get("external_inference_calls"), 0))
        total += int(number(object_field(report, "summary").get("external_inference_calls"), 0))
    return total


def post_v4_public_artifact_state() -> dict[str, Any]:
    present = [rel(path) for path in FORBIDDEN_POST_V4_PUBLIC_ARTIFACTS if path.exists()]
    if not present:
        return {
            "allowed": True,
            "mode": "absent",
            "present_artifacts": [],
            "approval_valid": False,
            "execute_report_valid": False,
            "operator_lock_active": PUBLIC_LOCK.exists(),
        }
    approval = read_json(OPERATOR_APPROVAL, {})
    execute = read_json(OPERATOR_EXECUTE, {})
    summary = object_field(execute, "summary")
    approval_valid = (
        approval.get("policy") == "project_theseus_public_calibration_operator_approval_v1"
        and approval.get("approved") is True
        and approval.get("proposed_slug") == "post_v4_seed23_5x32"
        and int(number(approval.get("max_runs"), 0)) == 1
    )
    execute_valid = (
        execute.get("policy") == "project_theseus_operator_bounded_public_calibration_v1"
        and execute.get("trigger_state") == "GREEN"
        and summary.get("executed") is True
        and summary.get("proposed_slug") == "post_v4_seed23_5x32"
        and summary.get("output_exists_after") is True
        and summary.get("operator_lock_present_after") is True
        and int(number(summary.get("run_returncode"), -1)) == 0
    )
    required_outputs_present = all(path.exists() for path in FORBIDDEN_POST_V4_PUBLIC_ARTIFACTS[:3])
    allowed = approval_valid and execute_valid and required_outputs_present and PUBLIC_LOCK.exists()
    return {
        "allowed": allowed,
        "mode": "approved_spent_one_shot" if allowed else "unapproved_or_incomplete",
        "present_artifacts": present,
        "approval_valid": approval_valid,
        "execute_report_valid": execute_valid,
        "required_outputs_present": required_outputs_present,
        "operator_lock_active": PUBLIC_LOCK.exists(),
        "rules": "post-v4 public artifacts may exist only after the approved one-shot calibration completed and relocked",
    }


def number(*values: Any) -> float:
    for value in values:
        try:
            if value is None or value == "":
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def object_field(value: Any, key: str) -> dict[str, Any]:
    if isinstance(value, dict) and isinstance(value.get(key), dict):
        return value[key]
    return {}


def read_json(path: Path, default: Any) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    return value if isinstance(value, dict) else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


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
