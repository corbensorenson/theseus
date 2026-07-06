#!/usr/bin/env python3
"""Pre-public private generalization readiness audit.

This audit is the handoff point between unattended private learning and an
operator-approved public calibration. It consolidates the current private
decoder, agent, teacher, ratchet, and safety evidence into one compact report.

It never runs public calibration, never unlocks the operator lock, and never
uses public prompts, tests, solutions, traces, or score labels as training
rows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PUBLIC_FLOOR = 0.70
MIN_LEARNED_TOKEN_PASSES = 4000
MIN_AGENT_TOOL_CASES = 64

DEFAULT_OUT = REPORTS / "pre_public_generalization_readiness_audit.json"
DEFAULT_MARKDOWN = REPORTS / "pre_public_generalization_readiness_audit.md"
DEFAULT_QUEUE = REPORTS / "pre_public_generalization_readiness_audit_queue.jsonl"

PUBLIC_LOCK = REPORTS / "public_calibration_operator_lock.flag"
GOVERNOR = REPORTS / "theseus_generalization_governor_v1.json"
GOVERNOR_QUEUE = REPORTS / "theseus_generalization_governor_v1_queue.jsonl"
RATCHET = REPORTS / "private_residual_self_improvement_ratchet_v1.json"
RATCHET_QUEUE = REPORTS / "private_residual_self_improvement_ratchet_v1_queue.jsonl"
AGENT_LANE = REPORTS / "agent_lane_transfer_gate.json"
AGENT_REFRESH = REPORTS / "agent_lane_private_refresh.json"
TEACHER_PREFLIGHT = REPORTS / "full_training_teacher_preflight.json"
READINESS_PACKET = REPORTS / "public_calibration_readiness_packet.json"
PUBLIC_CALIBRATION = REPORTS / "real_code_benchmark_graduation_wide_public_seed23_5x32_interface_floor_v1.json"
PUBLIC_MATRIX = REPORTS / "broad_transfer_matrix_wide_public_seed23_5x32_interface_floor_v1.json"
FRONTIER_EXPANDER = REPORTS / "private_generalization_frontier_expander_v1.json"

FORBIDDEN_POST_V4_PUBLIC_ARTIFACTS = [
    REPORTS / "real_code_benchmark_graduation_post_v4_seed23_5x32.json",
    REPORTS / "real_code_benchmark_traces_post_v4_seed23_5x32.jsonl",
    REPORTS / "student_code_candidates_post_v4_seed23_5x32.jsonl",
    REPORTS / "operator_bounded_public_calibration_post_v4_seed23_5x32.json",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--queue-out", default=rel(DEFAULT_QUEUE))
    parser.add_argument("--stale-seconds", type=int, default=72 * 3600)
    args = parser.parse_args()

    report = build_report(stale_seconds=max(3600, int(args.stale_seconds)), queue_out=resolve(args.queue_out))
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    write_jsonl(resolve(args.queue_out), report["queue"])
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report["trigger_state"] == "RED" else 0


def build_report(*, stale_seconds: int, queue_out: Path) -> dict[str, Any]:
    state = observe()
    evidence = summarize(state, stale_seconds=stale_seconds)
    gates = build_gates(evidence)
    hard_failed = [row for row in gates if row["severity"] == "hard" and not row["passed"]]
    warning_failed = [row for row in gates if row["severity"] == "warning" and not row["passed"]]
    decision = choose_decision(evidence, hard_failed)
    queue = build_queue(evidence, hard_failed)
    safe_queue = [
        row
        for row in queue
        if row.get("safe_to_execute_without_operator_public_approval") is True and row.get("command")
    ]
    trigger_state = "RED" if hard_failed else "YELLOW" if warning_failed or decision["kind"] != "public_transfer_floor_cleared" else "GREEN"
    return {
        "policy": "project_theseus_pre_public_generalization_readiness_audit_v1",
        "created_utc": now(),
        "ok": not hard_failed,
        "trigger_state": trigger_state,
        "decision": decision,
        "summary": {
            "decision": decision["kind"],
            "decision_reason": decision["reason"],
            "operator_review_ready": evidence["private"]["operator_review_ready"],
            "private_code_transfer_ready": evidence["private"]["code_transfer_ready"],
            "private_agent_transfer_ready": evidence["private"]["agent_transfer_ready"],
            "teacher_path_ready": evidence["private"]["teacher_path_ready"],
            "public_pass_rate": evidence["public"]["pass_rate"],
            "public_floor": PUBLIC_FLOOR,
            "public_task_count": evidence["public"]["task_count"],
            "cards_below_floor": evidence["public"]["cards_below_floor"],
            "public_calibration_allowed": False,
            "operator_lock_active": evidence["safety"]["operator_lock_active"],
            "forbidden_post_v4_public_artifact_count": len(evidence["safety"]["forbidden_post_v4_public_artifacts_present"]),
            "learned_token_pass_count_total": evidence["private"]["learned_token_pass_count_total"],
            "prototype_pass_count_total": evidence["private"]["prototype_pass_count_total"],
            "agent_lane_terminal_tool_use_cases": evidence["private"]["agent_lane_terminal_tool_use_cases"],
            "agent_lane_cross_domain_capsule_count": evidence["private"]["agent_lane_cross_domain_capsule_count"],
            "residual_ratchet_decision": evidence["private"]["residual_ratchet_decision"],
            "residual_ratchet_pending_queue_item_count": evidence["private"]["residual_ratchet_pending_queue_item_count"],
            "next_safe_private_action": safe_queue[0]["kind"] if safe_queue else evidence["private"]["next_safe_private_action"],
            "hard_failed_gate_count": len(hard_failed),
            "warning_failed_gate_count": len(warning_failed),
            "queue": rel(queue_out),
            "queue_item_count": len(queue),
            "score_semantics": "pre-public readiness audit only; not promotion evidence and not public calibration",
            "external_inference_calls": 0,
        },
        "evidence": evidence,
        "gates": gates,
        "queue": queue,
        "rules": {
            "public_calibration": "This audit never runs or unlocks public calibration. The operator lock must stay active until explicit approval.",
            "training": "Safe unattended actions may use private curricula and private eval feedback only.",
            "teacher": "Teacher evidence is proposal-only readiness; this audit does not call a teacher.",
            "promotion": "Private readiness can justify operator review, but broad promotion remains blocked until honest public transfer clears the floor.",
        },
        "external_inference_calls": 0,
    }


def observe() -> dict[str, Any]:
    return {
        "governor": read_json(GOVERNOR, {}),
        "governor_queue": read_jsonl(GOVERNOR_QUEUE),
        "ratchet": read_json(RATCHET, {}),
        "ratchet_queue": read_jsonl(RATCHET_QUEUE),
        "agent_lane": read_json(AGENT_LANE, {}),
        "agent_refresh": read_json(AGENT_REFRESH, {}),
        "teacher_preflight": read_json(TEACHER_PREFLIGHT, {}),
        "readiness_packet": read_json(READINESS_PACKET, {}),
        "public_calibration": read_json(PUBLIC_CALIBRATION, {}),
        "public_matrix": read_json(PUBLIC_MATRIX, {}),
        "frontier_expander": read_json(FRONTIER_EXPANDER, {}),
    }


def summarize(state: dict[str, Any], *, stale_seconds: int) -> dict[str, Any]:
    governor = state["governor"]
    governor_summary = object_field(governor, "summary")
    governor_evidence = object_field(governor, "evidence")
    governor_private = object_field(governor_evidence, "private")
    governor_downstream = object_field(governor_evidence, "downstream")
    matrix_summary = object_field(state["public_matrix"], "summary")
    public_summary = object_field(state["public_calibration"], "summary")
    teacher = state["teacher_preflight"]
    teacher_summary = object_field(teacher, "summary")
    agent_summary = object_field(state["agent_lane"], "summary")
    ratchet_summary = object_field(state["ratchet"], "summary")
    frontier_expander_summary = object_field(state["frontier_expander"], "summary")
    readiness = state["readiness_packet"]

    public_pass_rate = first_number(
        governor_summary.get("public_pass_rate"),
        public_summary.get("real_public_task_pass_rate"),
        matrix_summary.get("real_public_pass_rate"),
        0.0,
    )
    public_task_count = int(first_number(governor_summary.get("public_task_count"), public_summary.get("public_task_count"), matrix_summary.get("real_public_task_count"), 0))
    cards_below_floor = list_field(governor_summary.get("cards_below_floor") or matrix_summary.get("cards_below_floor"))
    forbidden_present = [rel(path) for path in FORBIDDEN_POST_V4_PUBLIC_ARTIFACTS if path.exists()]

    learned_token_pass_count = int(first_number(governor_summary.get("learned_token_pass_count_total"), 0))
    prototype_pass_count = int(first_number(governor_summary.get("prototype_pass_count_total"), 999))
    code_gate_names = [
        "private_broad_green",
        "semantic_alias_green",
        "novel_composition_green",
        "v4_learned_maturity_ready",
        "post_v4_shadow_learned_maturity_ready",
        "v5_learned_maturity_ready",
        "unseen_transfer_learned_maturity_ready",
        "residual_frontier_green",
        "contract_blind_transfer_green",
        "contract_blind_learned_maturity_ready",
        "architecture_guidance_green",
        "teacher_preflight_ok",
        "causal_architecture_delta_green",
        "student_first_audit_clear",
        "residual_ratchet_green",
    ]
    def governor_bool(name: str) -> bool:
        if name in governor_summary:
            return governor_summary.get(name) is True
        if name in governor_private:
            return governor_private.get(name) is True
        if name in governor_downstream:
            return governor_downstream.get(name) is True
        return False

    code_gate_values = {name: governor_bool(name) for name in code_gate_names}
    learned_decoder_ready = bool(
        learned_token_pass_count >= MIN_LEARNED_TOKEN_PASSES
        and prototype_pass_count == 0
        and governor_summary.get("broad_learned_control_structure_ready") is True
        and governor_summary.get("broad_learned_train_novelty_ready") is True
    )
    code_transfer_ready = bool(all(code_gate_values.values()) and learned_decoder_ready)

    agent_core_ready = governor_summary.get("agent_lane_core_transfer_ready") is True
    agent_breadth_ready = governor_summary.get("agent_lane_breadth_extension_ready") is True
    agent_remaining = list_field(governor_summary.get("agent_lane_remaining_blockers"))
    agent_transfer_ready = bool(agent_core_ready and agent_breadth_ready and not agent_remaining)

    teacher_ready = bool(
        governor_summary.get("teacher_preflight_ok") is True
        and teacher.get("ok") is True
        and int(first_number(teacher_summary.get("blocker_count"), 999)) == 0
        and teacher_summary.get("apply_mode_blocked") is True
        and teacher_summary.get("worker_teacher_invariant") is True
    )
    public_allowed_by_reports = bool(
        governor_summary.get("public_calibration_allowed")
        or governor.get("public_calibration_allowed")
        or readiness.get("public_calibration_allowed")
    )
    external_calls = external_call_total(governor, state["ratchet"], state["agent_lane"], state["agent_refresh"], teacher, readiness)
    hard_safety_ready = bool(
        PUBLIC_LOCK.exists()
        and not public_allowed_by_reports
        and not forbidden_present
        and external_calls == 0
        and not any_true("public_tests_used", state)
        and not any_true("public_solutions_used", state)
    )
    ratchet_pending = [
        row
        for row in state["ratchet_queue"]
        if isinstance(row, dict)
        and row.get("status", "pending") == "pending"
        and row.get("safe_to_execute_without_operator_public_approval") is True
        and row.get("command")
    ]
    governor_queue = [row for row in state["governor_queue"] if isinstance(row, dict)]
    operator_review_ready = bool(code_transfer_ready and agent_transfer_ready and teacher_ready and hard_safety_ready)
    stale = {
        "governor": stale_report(GOVERNOR, stale_seconds),
        "ratchet": stale_report(RATCHET, stale_seconds),
        "agent_lane": stale_report(AGENT_LANE, stale_seconds),
        "agent_refresh": stale_report(AGENT_REFRESH, stale_seconds),
        "teacher_preflight": stale_report(TEACHER_PREFLIGHT, stale_seconds),
        "readiness_packet": stale_report(READINESS_PACKET, stale_seconds),
    }
    return {
        "public": {
            "calibration": rel(PUBLIC_CALIBRATION),
            "matrix": rel(PUBLIC_MATRIX),
            "pass_rate": public_pass_rate,
            "floor": PUBLIC_FLOOR,
            "task_count": public_task_count,
            "cards_below_floor": cards_below_floor,
            "floor_cleared": bool(public_pass_rate >= PUBLIC_FLOOR and not cards_below_floor),
        },
        "private": {
            "code_transfer_ready": code_transfer_ready,
            "code_gate_values": code_gate_values,
            "learned_decoder_ready": learned_decoder_ready,
            "learned_token_pass_count_total": learned_token_pass_count,
            "prototype_pass_count_total": prototype_pass_count,
            "broad_learned_control_structure_ready": governor_summary.get("broad_learned_control_structure_ready") is True,
            "broad_learned_train_novelty_ready": governor_summary.get("broad_learned_train_novelty_ready") is True,
            "agent_transfer_ready": agent_transfer_ready,
            "agent_lane_core_transfer_ready": agent_core_ready,
            "agent_lane_breadth_extension_ready": agent_breadth_ready,
            "agent_lane_remaining_blockers": agent_remaining,
            "agent_lane_terminal_tool_use_cases": int(first_number(governor_summary.get("agent_lane_terminal_tool_use_cases"), 0)),
            "agent_lane_cross_domain_capsule_count": int(first_number(governor_summary.get("agent_lane_cross_domain_capsule_count"), 0)),
            "teacher_path_ready": teacher_ready,
            "teacher_live_status": teacher_summary.get("teacher_live_status"),
            "operator_review_ready": operator_review_ready,
            "residual_ratchet_decision": ratchet_summary.get("decision") or governor_summary.get("residual_ratchet_decision"),
            "residual_ratchet_top_action": ratchet_summary.get("top_private_action") or governor_summary.get("residual_ratchet_top_action"),
            "residual_ratchet_pending_queue_item_count": len(ratchet_pending),
            "frontier_expander_decision": frontier_expander_summary.get("decision") or object_field(state["frontier_expander"], "decision").get("kind", ""),
            "frontier_expander_next_safe_private_action": frontier_expander_summary.get("next_safe_private_action") or "",
            "next_safe_private_action": ratchet_pending[0]["kind"] if ratchet_pending else "",
            "ratchet_pending_queue": ratchet_pending[:3],
            "governor_queue": governor_queue[:3],
        },
        "safety": {
            "operator_lock_active": PUBLIC_LOCK.exists(),
            "public_calibration_allowed_by_reports": public_allowed_by_reports,
            "forbidden_post_v4_public_artifacts_present": forbidden_present,
            "public_tests_used": any_true("public_tests_used", state),
            "public_solutions_used": any_true("public_solutions_used", state),
            "external_inference_calls": external_calls,
            "hard_safety_ready": hard_safety_ready,
            "stale": stale,
        },
        "artifacts": {
            "governor": rel(GOVERNOR),
            "governor_queue": rel(GOVERNOR_QUEUE),
            "ratchet": rel(RATCHET),
            "ratchet_queue": rel(RATCHET_QUEUE),
            "agent_lane": rel(AGENT_LANE),
            "agent_refresh": rel(AGENT_REFRESH),
            "teacher_preflight": rel(TEACHER_PREFLIGHT),
            "readiness_packet": rel(READINESS_PACKET),
            "frontier_expander": rel(FRONTIER_EXPANDER),
        },
    }


def build_gates(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    private = evidence["private"]
    safety = evidence["safety"]
    return [
        gate("public_calibration_operator_lock_active", safety["operator_lock_active"], rel(PUBLIC_LOCK), "hard"),
        gate("public_calibration_not_allowed_by_reports", not safety["public_calibration_allowed_by_reports"], safety["public_calibration_allowed_by_reports"], "hard"),
        gate("forbidden_post_v4_public_artifacts_absent", not safety["forbidden_post_v4_public_artifacts_present"], safety["forbidden_post_v4_public_artifacts_present"], "hard"),
        gate("public_tests_and_solutions_not_used", not safety["public_tests_used"] and not safety["public_solutions_used"], safety, "hard"),
        gate("external_inference_zero", safety["external_inference_calls"] == 0, safety["external_inference_calls"], "hard"),
        gate("governor_current", not safety["stale"]["governor"]["stale"], safety["stale"], "warning"),
        gate("agent_lane_current", not safety["stale"]["agent_lane"]["stale"], safety["stale"], "warning"),
        gate("teacher_preflight_current", not safety["stale"]["teacher_preflight"]["stale"], safety["stale"], "warning"),
        gate("learned_decoder_private_evidence_ready", private["learned_decoder_ready"], private, "warning"),
        gate("private_code_transfer_ready", private["code_transfer_ready"], private["code_gate_values"], "warning"),
        gate("private_agent_transfer_ready", private["agent_transfer_ready"], private, "warning"),
        gate("teacher_path_ready", private["teacher_path_ready"], private, "warning"),
        gate("operator_review_ready", private["operator_review_ready"], private, "warning"),
        gate("public_transfer_floor_cleared", evidence["public"]["floor_cleared"], evidence["public"], "warning"),
    ]


def choose_decision(evidence: dict[str, Any], hard_failed: list[dict[str, Any]]) -> dict[str, Any]:
    if hard_failed:
        return {
            "kind": "stop_hard_safety_or_boundary_failure",
            "reason": "A hard public-boundary, lock, or inference invariant failed.",
            "public_calibration_allowed": False,
        }
    if not evidence["private"]["operator_review_ready"]:
        return {
            "kind": "continue_private_readiness_work",
            "reason": "Private decoder/agent/teacher evidence is not yet complete enough for operator public review.",
            "public_calibration_allowed": False,
        }
    if evidence["private"]["residual_ratchet_pending_queue_item_count"] > 0:
        return {
            "kind": "continue_private_residual_ratchet",
            "reason": "Safe private residual ratchet work remains pending before public review.",
            "public_calibration_allowed": False,
        }
    if not evidence["public"]["floor_cleared"]:
        return {
            "kind": "operator_public_review_required",
            "reason": "Private evidence is ready, but the only honest way to know whether transfer improved is an explicit bounded public calibration.",
            "public_calibration_allowed": False,
        }
    return {
        "kind": "public_transfer_floor_cleared",
        "reason": "Public transfer is already above the floor in the current evidence.",
        "public_calibration_allowed": False,
    }


def build_queue(evidence: dict[str, Any], hard_failed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if hard_failed:
        for failed in hard_failed:
            rows.append(
                queue_item(
                    f"fix_{failed['gate']}",
                    f"Resolve hard gate `{failed['gate']}` before training, review, or calibration.",
                    [],
                    priority=1,
                    safe=False,
                )
            )
        return rows

    private = evidence["private"]
    if private["residual_ratchet_pending_queue_item_count"] > 0:
        rows.append(
            queue_item(
                "execute_one_safe_private_residual_ratchet_action",
                "Execute one pending private residual ratchet action, then refresh the governor and this audit.",
                [
                    sys.executable,
                    "scripts/private_residual_self_improvement_ratchet_v1.py",
                    "--execute",
                    "--max-actions",
                    "1",
                ],
                priority=10,
                safe=True,
            )
        )
    if not private["code_transfer_ready"]:
        rows.append(
            queue_item(
                "refresh_generalization_governor",
                "Refresh private decoder/adapter learned-transfer evidence without running public calibration.",
                [
                    sys.executable,
                    "scripts/theseus_generalization_governor_v1.py",
                    "--out",
                    rel(GOVERNOR),
                    "--markdown-out",
                    "reports/theseus_generalization_governor_v1.md",
                    "--queue-out",
                    rel(GOVERNOR_QUEUE),
                ],
                priority=20,
                safe=True,
            )
        )
    if not private["agent_transfer_ready"]:
        rows.append(
            queue_item(
                "refresh_agent_lane_private_transfer",
                "Refresh private tool-use, RL/conversation, and cross-domain STS agent transfer evidence.",
                [
                    sys.executable,
                    "scripts/agent_lane_private_refresh.py",
                    "--max-tool-cases",
                    str(MIN_AGENT_TOOL_CASES),
                    "--max-capsules",
                    "256",
                ],
                priority=30,
                safe=True,
            )
        )
    if not private["teacher_path_ready"]:
        rows.append(
            queue_item(
                "refresh_teacher_preflight_proposal_only",
                "Refresh teacher readiness while preserving proposal-only and no-apply invariants.",
                [
                    sys.executable,
                    "scripts/full_training_teacher_preflight.py",
                    "--profile",
                    "smoke",
                    "--skip-autonomy-readiness",
                    "--out",
                    rel(TEACHER_PREFLIGHT),
                    "--markdown-out",
                    "reports/full_training_teacher_preflight.md",
                ],
                priority=40,
                safe=True,
            )
        )
    if not rows and private["operator_review_ready"] and not evidence["public"]["floor_cleared"]:
        if private["frontier_expander_next_safe_private_action"]:
            rows.append(
                queue_item(
                    "continue_private_generalization_frontier_expansion",
                    "Public transfer remains below floor; run one safe private frontier expansion before another review pass.",
                    [
                        sys.executable,
                        "scripts/private_generalization_frontier_expander_v1.py",
                        "--execute",
                        "--max-actions",
                        "1",
                    ],
                    priority=80,
                    safe=True,
                )
            )
        rows.append(
            queue_item(
                "operator_review_bounded_public_calibration_locked",
                "Private readiness is complete enough for review, but public calibration remains locked and must be explicitly approved.",
                [],
                priority=90,
                safe=False,
                requires_operator_public_unlock=True,
            )
        )
    return rows


def render_markdown(report: dict[str, Any]) -> str:
    summary = object_field(report, "summary")
    failed = [row for row in report.get("gates", []) if not row.get("passed")]
    lines = [
        "# Pre-Public Generalization Readiness Audit",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- decision: `{summary.get('decision')}`",
        f"- operator_review_ready: `{summary.get('operator_review_ready')}`",
        f"- private_code_transfer_ready: `{summary.get('private_code_transfer_ready')}`",
        f"- private_agent_transfer_ready: `{summary.get('private_agent_transfer_ready')}`",
        f"- teacher_path_ready: `{summary.get('teacher_path_ready')}`",
        f"- public_pass_rate: `{summary.get('public_pass_rate')}` floor=`{summary.get('public_floor')}` tasks=`{summary.get('public_task_count')}`",
        f"- cards_below_floor: `{summary.get('cards_below_floor')}`",
        f"- learned_token_pass_count_total: `{summary.get('learned_token_pass_count_total')}`",
        f"- prototype_pass_count_total: `{summary.get('prototype_pass_count_total')}`",
        f"- residual_ratchet_pending_queue_item_count: `{summary.get('residual_ratchet_pending_queue_item_count')}`",
        f"- next_safe_private_action: `{summary.get('next_safe_private_action')}`",
        f"- frontier_expander_decision: `{object_field(report.get('evidence') or {}, 'private').get('frontier_expander_decision')}`",
        f"- frontier_expander_next_safe_private_action: `{object_field(report.get('evidence') or {}, 'private').get('frontier_expander_next_safe_private_action')}`",
        f"- operator_lock_active: `{summary.get('operator_lock_active')}`",
        f"- public_calibration_allowed: `{summary.get('public_calibration_allowed')}`",
        f"- external_inference_calls: `{summary.get('external_inference_calls')}`",
        "",
        "## Failed Gates",
        "",
    ]
    if failed:
        for row in failed:
            lines.append(f"- `{row.get('gate')}` ({row.get('severity')})")
    else:
        lines.append("- None.")
    lines.extend(["", "## Queue", ""])
    for row in report.get("queue", []) or []:
        command = " ".join(str(item) for item in row.get("command") or [])
        if command:
            lines.append(f"- `{row.get('kind')}`: {row.get('title')} Command: `{command}`")
        else:
            lines.append(f"- `{row.get('kind')}`: {row.get('title')}")
    lines.extend(["", "## Rules", ""])
    for key, value in (report.get("rules") or {}).items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def queue_item(
    kind: str,
    title: str,
    command: list[str],
    *,
    priority: int,
    safe: bool,
    requires_operator_public_unlock: bool = False,
) -> dict[str, Any]:
    return {
        "policy": "project_theseus_pre_public_generalization_queue_item_v1",
        "queue": "pre_public_generalization_readiness_audit",
        "kind": kind,
        "title": title,
        "priority": int(priority),
        "command": command,
        "status": "pending",
        "safe_to_execute_without_operator_public_approval": bool(safe),
        "requires_operator_public_unlock": bool(requires_operator_public_unlock),
        "public_calibration_allowed": False,
    }


def stale_report(path: Path, stale_seconds: int) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "stale": True, "age_seconds": None}
    age = max(0.0, time.time() - path.stat().st_mtime)
    return {"path": rel(path), "exists": True, "stale": age > stale_seconds, "age_seconds": round(age, 3)}


def external_call_total(*reports: dict[str, Any]) -> int:
    total = 0
    for report in reports:
        total += int(first_number(report.get("external_inference_calls"), 0))
        total += int(first_number(object_field(report, "summary").get("external_inference_calls"), 0))
    return total


def any_true(key: str, state: dict[str, Any]) -> bool:
    for value in state.values():
        if isinstance(value, dict):
            if value.get(key) is True:
                return True
            if object_field(value, "summary").get(key) is True:
                return True
    return False


def first_number(*values: Any) -> float:
    for value in values:
        try:
            if value is not None and value != "":
                return float(value)
        except (TypeError, ValueError):
            pass
    return 0.0


def object_field(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key) if isinstance(value, dict) else None
    return item if isinstance(item, dict) else {}


def list_field(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def read_json(path: Path, default: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def read_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"_decode_error": line[:200]})
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
