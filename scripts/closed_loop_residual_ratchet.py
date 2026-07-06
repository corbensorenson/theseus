#!/usr/bin/env python3
"""Closed-loop residual ratchet for Project Theseus.

This script is deliberately a control artifact, not a trainer. It turns the
current public-transfer residual evidence into a private-only repair map,
checks whether same-seed private lift and transfer gates prove the repair, and
emits exactly one machine-readable decision:

    promote | rollback | retry_private | stop_blocker

Public benchmark data stays calibration-only. This report never reads public
tests/solutions, never writes training rows from public content, and never
launches public calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import report_evidence_store  # noqa: E402


REPORTS = ROOT / "reports"
TRAINING = Path("D:/ProjectTheseus/training_data/high_transfer/private_train")
DEFAULT_OUT = REPORTS / "closed_loop_residual_ratchet.json"
DEFAULT_MARKDOWN = REPORTS / "closed_loop_residual_ratchet.md"
POLICY = "project_theseus_closed_loop_residual_ratchet_v1"
PUBLIC_FLOOR = 0.70
MIN_PRIVATE_SEMANTIC_LIFT = 0.03
SAME_FRONTIER_CHURN_THRESHOLD = 12
DECISIONS = {"promote", "rollback", "retry_private", "stop_blocker"}


REPAIR_LIBRARY: dict[str, dict[str, Any]] = {
    "edge_case": {
        "repair_concept": "edge_contract_private_repair",
        "private_sources": [
            TRAINING / "residual_targeted_private_edge_case_contract_v1_residual_code_lm_tasks.jsonl",
            TRAINING / "edge_contract_v2_private_residual_curriculum_residual_code_lm_tasks.jsonl",
            TRAINING / "edge_case_full_body_private_curriculum_v1_residual_code_lm_tasks.jsonl",
            TRAINING / "broad_public_code_transfer_floor_recovery_v1_residual_code_lm_tasks.jsonl",
        ],
        "target_metric": "same_seed_private_edge_semantic_lift",
        "consumer": "broad_transfer_residual_decoder_ablation -> decoder_v2_private_ablation_gate",
    },
    "local_code_generation_adapter_needed": {
        "repair_concept": "adapter_runtime_dependency_private_repair",
        "private_sources": [
            TRAINING / "admissibility_and_interface_residual_code_lm_tasks.jsonl",
            TRAINING / "execution_shaped_programs_residual_code_lm_tasks.jsonl",
            TRAINING / "broad_public_code_transfer_floor_recovery_v1_residual_code_lm_tasks.jsonl",
        ],
        "target_metric": "same_seed_private_local_adapter_semantic_lift",
        "consumer": "private_public_transfer_proof -> broad_transfer_matrix",
    },
    "external_dependency_missing": {
        "repair_concept": "dependency_optional_adapter_private_repair",
        "private_sources": [
            TRAINING / "admissibility_and_interface_residual_code_lm_tasks.jsonl",
            TRAINING / "execution_shaped_programs_residual_code_lm_tasks.jsonl",
            TRAINING / "broad_public_code_transfer_floor_recovery_v1_residual_code_lm_tasks.jsonl",
        ],
        "target_metric": "dependency_safe_local_adapter_coverage",
        "consumer": "staged verifier prefilter -> private_public_transfer_proof",
    },
    "algorithm_choice": {
        "repair_concept": "algorithmic_planning_private_repair",
        "private_sources": [
            TRAINING / "algorithmic_planning_residual_code_lm_tasks.jsonl",
            TRAINING / "broad_public_code_transfer_floor_recovery_v1_residual_code_lm_tasks.jsonl",
        ],
        "target_metric": "same_seed_private_algorithm_semantic_lift",
        "consumer": "ranker/retry policy -> decoder_v2_private_ablation_gate",
    },
    "type_handling": {
        "repair_concept": "type_return_shape_private_repair",
        "private_sources": [
            TRAINING / "type_and_return_shape_residual_code_lm_tasks.jsonl",
            TRAINING / "private_type_shape_receiver_veto_ablation_residual_code_lm_tasks.jsonl",
            TRAINING / "broad_public_code_transfer_floor_recovery_v1_residual_code_lm_tasks.jsonl",
        ],
        "target_metric": "same_seed_private_type_return_shape_semantic_lift",
        "consumer": "decoder_v2_private_ablation_gate -> private_public_transfer_proof",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-packet", default="reports/public_transfer_residual_packet.json")
    parser.add_argument("--floor-recovery", default="reports/broad_public_code_transfer_floor_recovery.json")
    parser.add_argument("--same-seed-ablation", default="reports/broad_transfer_residual_decoder_ablation_next_residual_families_after_rank_patch.json")
    parser.add_argument("--decoder-gate", default="reports/decoder_v2_private_ablation_gate.json")
    parser.add_argument("--transfer-proof", default="reports/private_public_transfer_proof.json")
    parser.add_argument("--broad-matrix", default="reports/broad_transfer_matrix.json")
    parser.add_argument("--maturity", default="reports/maturity_integrity_audit.json")
    parser.add_argument("--asi-governor", default="reports/asi_wall_breaker_governor.json")
    parser.add_argument("--autonomy-watchdog", default="reports/autonomy_watchdog.json")
    parser.add_argument("--operator-lock", default="reports/public_calibration_operator_lock.flag")
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()

    started = time.perf_counter()
    reports = {
        "residual_packet": read_json(resolve(args.residual_packet)),
        "floor_recovery": read_json(resolve(args.floor_recovery)),
        "same_seed_ablation": read_json(resolve(args.same_seed_ablation)),
        "decoder_gate": read_json(resolve(args.decoder_gate)),
        "transfer_proof": read_json(resolve(args.transfer_proof)),
        "broad_matrix": read_json(resolve(args.broad_matrix)),
        "maturity": read_json(resolve(args.maturity)),
        "asi_governor": read_json(resolve(args.asi_governor)),
        "autonomy_watchdog": read_json(resolve(args.autonomy_watchdog)),
    }
    operator_lock = operator_lock_state(resolve(args.operator_lock))
    residual_evidence = residual_packet_summary(reports["residual_packet"], args.residual_packet)
    repair_items = build_repair_items(residual_evidence)
    same_seed = same_seed_summary(reports["floor_recovery"], reports["same_seed_ablation"])
    gates = build_gates(reports, residual_evidence, repair_items, same_seed, operator_lock)
    decision = choose_decision(reports, gates, residual_evidence, repair_items, same_seed, operator_lock)
    churn = same_frontier_churn_state(reports["autonomy_watchdog"], decision)

    payload = {
        "policy": POLICY,
        "created_utc": now(),
        "trigger_state": trigger_state(gates, decision),
        "summary": {
            "decision": decision["kind"],
            "decision_reason": decision["reason"],
            "broad_public_pass_rate": broad_public_pass_rate(reports["broad_matrix"]),
            "public_floor": PUBLIC_FLOOR,
            "floor_cleared": broad_public_pass_rate(reports["broad_matrix"]) >= PUBLIC_FLOOR,
            "dominant_residuals": residual_evidence["dominant_residuals"],
            "repair_item_count": len(repair_items),
            "private_repair_pressure_rows": same_seed["private_pressure_row_count"],
            "same_seed_private_semantic_lift": same_seed["semantic_lift"],
            "decoder_gate_ready": gate_ready(reports["decoder_gate"]),
            "private_public_transfer_ready": transfer_ready(reports["transfer_proof"]),
            "operator_lock_active": operator_lock["active"],
            "same_frontier_churn_demoted": churn["demoted"],
            "external_inference_calls": 0,
        },
        "residual_source": residual_evidence,
        "private_repair_items": repair_items,
        "same_seed_evidence": same_seed,
        "gate_states": gates,
        "decision": decision,
        "same_frontier_churn": churn,
        "rules": {
            "public_data": "Public benchmarks are calibration-only; this report uses aggregate residual families and hashed IDs only.",
            "repair_policy": "Residuals become private curricula or source-level architecture work, never public prompt/test/solution training rows.",
            "ablation_policy": "Promotion claims require same-seed private behavioral lift and refreshed decoder/transfer gates.",
            "decision_policy": "Exactly one of promote, rollback, retry_private, or stop_blocker is emitted per ratchet run.",
            "public_calibration": "This script never launches public calibration; a bounded public run must be separately gated.",
        },
        "gates": gates,
        "external_inference_calls": 0,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }
    report_evidence_store.write_json_report(
        resolve(args.out),
        payload,
        markdown_path=resolve(args.markdown_out),
        markdown_text=render_markdown(payload),
        db_path=report_evidence_store.DEFAULT_DB,
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["trigger_state"] != "RED" else 2


def residual_packet_summary(packet: dict[str, Any], packet_path: str) -> dict[str, Any]:
    public = object_field(object_field(packet, "teacher_packet"), "public_calibration")
    summary = object_field(packet, "summary")
    dominant = public.get("dominant_residuals") or summary.get("dominant_residuals") or []
    residual_counts = normalize_residual_counts(dominant)
    card_summaries = [row for row in as_list(public.get("card_summaries")) if isinstance(row, dict)]
    stage_counts = object_field(public, "residual_stage_counts")
    hashed = [row for row in as_list(public.get("hashed_public_residual_task_ids")) if isinstance(row, dict)]
    return {
        "path": packet_path,
        "exists": bool(packet),
        "trigger_state": packet.get("trigger_state"),
        "calibration_source": public.get("calibration_source") or summary.get("calibration_source"),
        "aggregate_public_pass_rate": first_number(public.get("aggregate_public_pass_rate"), summary.get("aggregate_public_pass_rate")),
        "cards_below_floor": as_str_list(public.get("cards_below_floor") or summary.get("cards_below_floor")),
        "dominant_residuals": residual_counts,
        "residual_stage_counts": {str(k): int(number(v)) for k, v in stage_counts.items()},
        "card_summaries": compact_card_summaries(card_summaries),
        "hashed_public_residual_task_count": len(hashed),
        "hashed_public_residual_task_samples": hashed[:12],
        "public_content_policy": public.get("public_content_policy") or "",
        "public_solutions_used": bool(public.get("public_solutions_used")),
        "public_tests_used": bool(public.get("public_tests_used")),
        "template_like_candidate_count": int(number(public.get("template_like_candidate_count"))),
        "student_candidate_benchmark_integrity_valid": bool(public.get("student_candidate_benchmark_integrity_valid", True)),
    }


def build_repair_items(residual_evidence: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for family, count in residual_evidence["dominant_residuals"]:
        spec = REPAIR_LIBRARY.get(family)
        if not spec:
            spec = {
                "repair_concept": f"{family}_private_repair",
                "private_sources": [TRAINING / "broad_public_code_transfer_floor_recovery_v1_residual_code_lm_tasks.jsonl"],
                "target_metric": f"same_seed_private_{family}_lift",
                "consumer": "closed_loop_residual_ratchet",
            }
        source_rows = []
        for path in spec["private_sources"]:
            row_count = count_jsonl_rows(path)
            source_rows.append(
                {
                    "path": rel_or_abs(path),
                    "exists": path.exists(),
                    "row_count": row_count,
                    "bytes": path.stat().st_size if path.exists() else 0,
                }
            )
        card_counts = family_card_counts(residual_evidence["card_summaries"], family)
        items.append(
            {
                "id": stable_id("repair", family, count, spec["repair_concept"])[:20],
                "residual_family": family,
                "residual_count": int(count),
                "repair_concept": spec["repair_concept"],
                "target_metric": spec["target_metric"],
                "consumer": spec["consumer"],
                "private_sources": source_rows,
                "private_source_row_count": sum(int(row["row_count"]) for row in source_rows),
                "affected_cards": card_counts,
                "public_training_policy": "aggregate residual routing only; no public prompts/tests/solutions become rows",
            }
        )
    items.sort(key=lambda row: (-int(row["residual_count"]), str(row["residual_family"])))
    return items


def same_seed_summary(floor_recovery: dict[str, Any], ablation: dict[str, Any]) -> dict[str, Any]:
    recovery_summary = object_field(floor_recovery, "summary")
    ablation_delta = object_field(ablation, "delta")
    recovery_gates = as_list(floor_recovery.get("gates"))
    ablation_gates = as_list(ablation.get("gates"))
    semantic_lift = first_number(
        recovery_summary.get("same_seed_private_semantic_lift"),
        ablation_delta.get("semantic_test_passed_task_rate_delta"),
        0.0,
    )
    semantic_count_lift = int(
        number(
            first_number(
                ablation_delta.get("semantic_test_passed_task_count_delta"),
                recovery_summary.get("semantic_test_passed_task_count_delta"),
                0,
            )
        )
    )
    return {
        "floor_recovery_state": floor_recovery.get("trigger_state"),
        "same_seed_ablation_state": ablation.get("trigger_state"),
        "private_pressure_row_count": int(number(recovery_summary.get("private_pressure_row_count"))),
        "private_pressure_rows_with_behavior_tests": int(number(recovery_summary.get("private_pressure_rows_with_behavior_tests"))),
        "private_focus_counts": object_field(recovery_summary, "private_focus_counts"),
        "semantic_target_family_counts": object_field(recovery_summary, "semantic_target_family_counts"),
        "same_seed_ablation_status": recovery_summary.get("same_seed_ablation_status"),
        "semantic_lift": float(semantic_lift or 0.0),
        "semantic_count_lift": semantic_count_lift,
        "ablation_private_only": gate_passed(recovery_gates, "same_seed_private_ablation_private_only")
        or gate_passed(ablation_gates, "private_only"),
        "semantic_lift_gate": gate_passed(recovery_gates, "same_seed_private_semantic_lift_required_before_promotion")
        or gate_passed(ablation_gates, "private_semantic_correctness_lift"),
        "candidate_distribution_changed": gate_passed(ablation_gates, "candidate_distribution_changed")
        or bool(first_number(ablation_delta.get("eligible_receiver_inventory_task_count_delta"), 0)),
        "patched_candidates_learned_token_only": bool(recovery_summary.get("patched_candidates_learned_token_only"))
        or gate_passed(ablation_gates, "patched_eligible_candidates_are_grammar_masked_learned_tokens"),
        "public_tests_used": bool(recovery_summary.get("public_tests_used")),
        "public_solutions_used": bool(recovery_summary.get("public_solutions_used")),
        "public_calibration_run": bool(recovery_summary.get("public_calibration_run")),
    }


def build_gates(
    reports: dict[str, dict[str, Any]],
    residual_evidence: dict[str, Any],
    repair_items: list[dict[str, Any]],
    same_seed: dict[str, Any],
    operator_lock: dict[str, Any],
) -> list[dict[str, Any]]:
    broad_rate = broad_public_pass_rate(reports["broad_matrix"])
    maturity_summary = object_field(reports["maturity"], "summary")
    asi_summary = object_field(reports["asi_governor"], "summary")
    gates = [
        gate("residual_packet_loaded", residual_evidence["exists"], residual_evidence["path"]),
        gate(
            "public_residuals_are_sanitized",
            not residual_evidence["public_tests_used"]
            and not residual_evidence["public_solutions_used"]
            and residual_evidence["template_like_candidate_count"] == 0
            and residual_evidence["student_candidate_benchmark_integrity_valid"],
            {
                "public_content_policy": residual_evidence["public_content_policy"],
                "hashed_public_residual_task_count": residual_evidence["hashed_public_residual_task_count"],
            },
        ),
        gate(
            "private_repair_pressure_present",
            bool(repair_items) and sum(int(row["private_source_row_count"]) for row in repair_items) > 0,
            {"repair_item_count": len(repair_items)},
        ),
        gate(
            "same_seed_private_ablation_lift_present",
            same_seed["ablation_private_only"]
            and same_seed["semantic_lift"] >= MIN_PRIVATE_SEMANTIC_LIFT
            and same_seed["semantic_lift_gate"],
            same_seed,
        ),
        gate(
            "decoder_gate_green",
            reports["decoder_gate"].get("trigger_state") == "GREEN" and gate_ready(reports["decoder_gate"]),
            compact_decoder_gate(reports["decoder_gate"]),
        ),
        gate(
            "private_public_transfer_proof_green",
            reports["transfer_proof"].get("trigger_state") == "GREEN" and transfer_ready(reports["transfer_proof"]),
            object_field(reports["transfer_proof"], "summary"),
        ),
        gate(
            "broad_floor_cleared_or_precise_residual_diagnosis",
            broad_rate >= PUBLIC_FLOOR or bool(residual_evidence["dominant_residuals"]),
            {
                "broad_public_pass_rate": broad_rate,
                "public_floor": PUBLIC_FLOOR,
                "dominant_residuals": residual_evidence["dominant_residuals"],
            },
        ),
        gate(
            "promotion_and_growth_stay_locked",
            not bool(maturity_summary.get("model_growth_allowed"))
            and not bool(maturity_summary.get("candidate_promotion_allowed"))
            and not bool(asi_summary.get("model_growth_allowed"))
            and not bool(asi_summary.get("candidate_promotion_allowed")),
            {
                "maturity_model_growth_allowed": maturity_summary.get("model_growth_allowed"),
                "maturity_candidate_promotion_allowed": maturity_summary.get("candidate_promotion_allowed"),
                "governor_model_growth_allowed": asi_summary.get("model_growth_allowed"),
                "governor_candidate_promotion_allowed": asi_summary.get("candidate_promotion_allowed"),
            },
        ),
        gate(
            "public_calibration_not_launched_by_ratchet",
            True,
            {
                "operator_lock_active": operator_lock["active"],
                "rule": "closed_loop_residual_ratchet.py emits a decision only; it never launches public calibration",
            },
        ),
    ]
    return gates


def choose_decision(
    reports: dict[str, dict[str, Any]],
    gates: list[dict[str, Any]],
    residual_evidence: dict[str, Any],
    repair_items: list[dict[str, Any]],
    same_seed: dict[str, Any],
    operator_lock: dict[str, Any],
) -> dict[str, Any]:
    broad_rate = broad_public_pass_rate(reports["broad_matrix"])
    decoder_ready = reports["decoder_gate"].get("trigger_state") == "GREEN" and gate_ready(reports["decoder_gate"])
    proof_ready = reports["transfer_proof"].get("trigger_state") == "GREEN" and transfer_ready(reports["transfer_proof"])
    sanitized = gate_by_name(gates, "public_residuals_are_sanitized").get("passed") is True
    same_seed_ready = gate_by_name(gates, "same_seed_private_ablation_lift_present").get("passed") is True
    private_pressure = gate_by_name(gates, "private_repair_pressure_present").get("passed") is True

    if not sanitized:
        return decision(
            "rollback",
            "Public boundary or template integrity failed; demote the repair and preserve diagnostics only.",
            [],
            {"failed_gate": "public_residuals_are_sanitized"},
        )
    if not residual_evidence["exists"]:
        return decision(
            "stop_blocker",
            "No residual packet is available, so the loop cannot map failure to repair pressure.",
            [
                "python",
                "scripts/public_transfer_residual_packet.py",
                "--out",
                "reports/public_transfer_residual_packet.json",
                "--markdown-out",
                "reports/public_transfer_residual_packet.md",
            ],
            {"missing": residual_evidence["path"]},
        )
    if not private_pressure:
        return decision(
            "retry_private",
            "Residuals exist but no private repair pressure is materialized.",
            ["python", "scripts/broad_public_code_transfer_floor_recovery.py", "--execute-ablation"],
            {"residuals": residual_evidence["dominant_residuals"]},
        )
    if not same_seed_ready:
        return decision(
            "retry_private",
            "Private repair pressure exists, but same-seed private behavioral lift is missing or too weak.",
            [
                "python",
                "scripts/broad_transfer_residual_decoder_ablation.py",
                "--task-limit",
                "96",
                "--candidates-per-task",
                "8",
            ],
            {"same_seed": same_seed},
        )
    if not decoder_ready:
        return decision(
            "retry_private",
            "Same-seed lift exists, but decoder gate is not GREEN after the repair.",
            ["python", "scripts/decoder_v2_private_ablation_gate.py"],
            compact_decoder_gate(reports["decoder_gate"]),
        )
    if not proof_ready:
        return decision(
            "retry_private",
            "Decoder gate is GREEN, but private/public transfer proof is not GREEN.",
            ["python", "scripts/private_public_transfer_proof.py"],
            object_field(reports["transfer_proof"], "summary"),
        )
    if broad_rate >= PUBLIC_FLOOR:
        return decision(
            "promote",
            "Broad public transfer floor is cleared with clean gates; promote architecture-control evidence only.",
            [],
            {"broad_public_pass_rate": broad_rate, "public_floor": PUBLIC_FLOOR},
        )
    if operator_lock["active"]:
        return decision(
            "stop_blocker",
            "Private repair lift and transfer gates are GREEN, but the broad public floor remains below target and the next proof requires one separately gated bounded calibration.",
            [
                "python",
                "scripts/broad_code_calibration_scheduler.py",
                "--min-public-tasks",
                "32",
                "--out",
                "reports/broad_code_calibration_scheduler.json",
                "--markdown-out",
                "reports/broad_code_calibration_scheduler.md",
            ],
            {
                "operator_lock": operator_lock,
                "broad_public_pass_rate": broad_rate,
                "public_floor": PUBLIC_FLOOR,
                "dominant_residuals": residual_evidence["dominant_residuals"],
                "bounded_public_calibration_status": "proposal_only_locked",
            },
        )
    return decision(
        "retry_private",
        "Broad floor is below target; use the current residual packet for another private-only repair before any promotion.",
        ["python", "scripts/broad_public_code_transfer_floor_recovery.py", "--execute-ablation"],
        {
            "broad_public_pass_rate": broad_rate,
            "public_floor": PUBLIC_FLOOR,
            "repair_items": [row["id"] for row in repair_items],
        },
    )


def same_frontier_churn_state(watchdog: dict[str, Any], decision_row: dict[str, Any]) -> dict[str, Any]:
    summary = object_field(watchdog, "summary")
    streak = int(number(summary.get("same_frontier_streak")))
    selected = str(summary.get("board_selected_task_id") or "")
    demoted = streak >= SAME_FRONTIER_CHURN_THRESHOLD and decision_row["kind"] in {"retry_private", "stop_blocker"}
    return {
        "same_frontier_streak": streak,
        "threshold": SAME_FRONTIER_CHURN_THRESHOLD,
        "selected_task_id": selected,
        "demoted": demoted,
        "decision_effect": (
            "same-frontier churn is demoted to closed-loop residual ratchet decision; do not repeat the stale board task without new evidence"
            if demoted
            else "no churn demotion needed"
        ),
    }


def trigger_state(gates: list[dict[str, Any]], decision_row: dict[str, Any]) -> str:
    failed = [row for row in gates if not row.get("passed")]
    if decision_row["kind"] == "rollback":
        return "RED"
    if any(row["name"] in {"public_residuals_are_sanitized", "residual_packet_loaded"} for row in failed):
        return "RED"
    if failed or decision_row["kind"] in {"retry_private", "stop_blocker"}:
        return "YELLOW"
    return "GREEN"


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    decision_row = payload["decision"]
    lines = [
        "# Closed-Loop Residual Ratchet",
        "",
        f"- State: `{payload['trigger_state']}`",
        f"- Decision: `{decision_row['kind']}`",
        f"- Reason: {decision_row['reason']}",
        f"- Broad public pass rate: `{summary.get('broad_public_pass_rate')}` / floor `{summary.get('public_floor')}`",
        f"- Same-seed private semantic lift: `{summary.get('same_seed_private_semantic_lift')}`",
        f"- Decoder gate ready: `{summary.get('decoder_gate_ready')}`",
        f"- Transfer proof ready: `{summary.get('private_public_transfer_ready')}`",
        f"- Operator lock active: `{summary.get('operator_lock_active')}`",
        f"- Same-frontier churn demoted: `{summary.get('same_frontier_churn_demoted')}`",
        "",
        "## Residuals",
    ]
    for family, count in summary.get("dominant_residuals") or []:
        lines.append(f"- `{family}`: `{count}`")
    lines.extend(["", "## Private Repair Items"])
    for item in payload.get("private_repair_items", [])[:12]:
        lines.append(
            f"- `{item['residual_family']}` -> `{item['repair_concept']}` "
            f"({item['private_source_row_count']} private rows, consumer `{item['consumer']}`)"
        )
    lines.extend(["", "## Gates"])
    for row in payload.get("gates", []):
        mark = "PASS" if row.get("passed") else "FAIL"
        lines.append(f"- `{mark}` `{row.get('name')}`")
    if decision_row.get("command"):
        lines.extend(["", "## Proposed Command", "```powershell", " ".join(map(str, decision_row["command"])), "```"])
    lines.extend(["", "## Rules", "- No public tests, solutions, copied prompts, or template bodies are used as training data.", "- Public calibration is not launched by this script."])
    return "\n".join(lines) + "\n"


def compact_card_summaries(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for card in cards:
        rows.append(
            {
                "card_id": str(card.get("card_id") or ""),
                "multi_stream_pass_rate": first_number(card.get("multi_stream_pass_rate")),
                "public_task_count": int(number(card.get("public_task_count"))),
                "residual_family_counts": object_field(card, "residual_family_counts"),
                "clean_evidence": bool(card.get("clean_evidence")),
                "no_cheat_valid": bool(card.get("no_cheat_valid")),
            }
        )
    return rows


def family_card_counts(cards: list[dict[str, Any]], family: str) -> dict[str, int]:
    counts = {}
    for card in cards:
        residuals = object_field(card, "residual_family_counts")
        count = int(number(residuals.get(family)))
        if count > 0:
            counts[str(card.get("card_id") or "unknown")] = count
    return counts


def normalize_residual_counts(raw: Any) -> list[list[Any]]:
    counts: Counter[str] = Counter()
    if isinstance(raw, dict):
        for family, count in raw.items():
            counts[str(family)] += int(number(count))
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                counts[str(item[0])] += int(number(item[1]))
            elif isinstance(item, dict):
                family = str(item.get("family") or item.get("residual_family") or item.get("type") or "")
                if family:
                    counts[family] += int(number(item.get("count") or 1))
    return [[family, count] for family, count in counts.most_common()]


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def gate_by_name(gates: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for row in gates:
        if row.get("name") == name:
            return row
    return {}


def gate_passed(gates: list[Any], *names: str) -> bool:
    wanted = set(names)
    for row in gates:
        if not isinstance(row, dict):
            continue
        name = str(row.get("gate") or row.get("name") or "")
        if name in wanted and row.get("passed") is True:
            return True
    return False


def decision(kind: str, reason: str, command: list[str], evidence: Any) -> dict[str, Any]:
    if kind not in DECISIONS:
        raise ValueError(kind)
    return {
        "kind": kind,
        "reason": reason,
        "command": command,
        "evidence": evidence,
        "allowed_values": sorted(DECISIONS),
    }


def gate_ready(report: dict[str, Any]) -> bool:
    return bool(report.get("ready_for_public_calibration"))


def transfer_ready(report: dict[str, Any]) -> bool:
    return bool(report.get("ready_for_public_calibration")) or bool(object_field(report, "summary").get("ready_for_public_calibration"))


def compact_decoder_gate(report: dict[str, Any]) -> dict[str, Any]:
    summary = object_field(report, "summary")
    return {
        "trigger_state": report.get("trigger_state"),
        "ready_for_public_calibration": report.get("ready_for_public_calibration"),
        "public_actual_token_task_coverage": summary.get("public_actual_token_task_coverage"),
        "public_no_admissible_task_rate": summary.get("public_no_admissible_task_rate"),
        "public_program_synthesis_loop_present_rate": summary.get("public_program_synthesis_loop_present_rate"),
        "public_program_synthesis_promotion_ready_rate": summary.get("public_program_synthesis_promotion_ready_rate"),
    }


def broad_public_pass_rate(report: dict[str, Any]) -> float:
    summary = object_field(report, "summary")
    return float(first_number(summary.get("real_public_pass_rate"), summary.get("aggregate_pass_rate"), 0.0) or 0.0)


def operator_lock_state(path: Path) -> dict[str, Any]:
    return {
        "path": rel_or_abs(path),
        "active": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "mtime": path.stat().st_mtime if path.exists() else None,
    }


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else ROOT / raw


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def object_field(obj: dict[str, Any], key: str) -> dict[str, Any]:
    value = obj.get(key) if isinstance(obj, dict) else {}
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def as_str_list(value: Any) -> list[str]:
    return [str(item) for item in as_list(value) if str(item)]


def first_number(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        parsed = number(value, default=None)
        if parsed is not None:
            return float(parsed)
    return None


def number(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def stable_id(*parts: Any) -> str:
    text = "\n".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
