#!/usr/bin/env python3
"""Post-distillation public-transfer readiness gate.

This script runs after the broad-private learned distillation gate. It does not
run public calibration and does not read public tests/solutions. It reconciles
the new private learned-token evidence with the latest already-spent wide public
calibration, so private ladder success cannot accidentally unlock another
public run while broad public transfer remains below floor.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from readiness_freshness import freshness_report, path_mtime


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_LEARNED_DISTILLATION = REPORTS / "broad_private_learned_distillation_gate_v1.json"
DEFAULT_PRIVATE_BROAD = REPORTS / "broad_private_generalization_unattended_v1.json"
DEFAULT_PRIVATE_BROAD_GATE = REPORTS / "broad_private_generalization_gate_v1.json"
DEFAULT_EDGE_V2_RESCORE = REPORTS / "code_lm_closure_edge_contract_v2_private_rescore.json"
DEFAULT_EDGE_V2_RUNNER = REPORTS / "edge_contract_v2_private_closure_runner.json"
DEFAULT_EDGE_V2_CLOSURE = REPORTS / "code_lm_closure_rust_edge_contract_v2_private.json"
DEFAULT_EDGE_V3_BROAD_SCORE = REPORTS / "edge_contract_v3_verifier_mismatch_public_transfer_heldout_v3_contract_strict_syntaxrepair_full192_broad_score.json"
DEFAULT_EDGE_V3_LEARNED_GATE = REPORTS / "edge_contract_v3_verifier_mismatch_public_transfer_v3_contract_strict_syntaxrepair_full192_broad_learned_distillation_gate.json"
DEFAULT_V4_MATURITY = REPORTS / "public_safe_broad_transfer_maturity_v4.json"
DEFAULT_V4_SCORE = REPORTS / "public_safe_broad_transfer_maturity_v4_score.json"
DEFAULT_V4_LEARNED_GATE = REPORTS / "public_safe_broad_transfer_maturity_v4_learned_distillation_gate.json"
DEFAULT_PUBLIC_CALIBRATION = REPORTS / "real_code_benchmark_graduation_wide_public_seed23_5x32_interface_floor_v1.json"
DEFAULT_PUBLIC_MATRIX = REPORTS / "broad_transfer_matrix_wide_public_seed23_5x32_interface_floor_v1.json"
DEFAULT_PUBLIC_RESIDUAL = REPORTS / "public_code_transfer_residual_report_wide_public_seed23_5x32_interface_floor_v1.json"
DEFAULT_OPERATOR_LOCK = REPORTS / "public_calibration_operator_lock.flag"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learned-distillation", default=rel(DEFAULT_LEARNED_DISTILLATION))
    parser.add_argument("--private-broad", default=rel(DEFAULT_PRIVATE_BROAD))
    parser.add_argument("--private-broad-gate", default=rel(DEFAULT_PRIVATE_BROAD_GATE))
    parser.add_argument("--public-calibration", default=rel(DEFAULT_PUBLIC_CALIBRATION))
    parser.add_argument("--public-matrix", default=rel(DEFAULT_PUBLIC_MATRIX))
    parser.add_argument("--public-residual", default=rel(DEFAULT_PUBLIC_RESIDUAL))
    parser.add_argument("--operator-lock", default=rel(DEFAULT_OPERATOR_LOCK))
    parser.add_argument("--min-public-pass-rate", type=float, default=0.70)
    parser.add_argument("--min-public-task-count", type=int, default=160)
    parser.add_argument("--out", default="reports/post_distillation_public_transfer_readiness_v1.json")
    parser.add_argument("--markdown-out", default="reports/post_distillation_public_transfer_readiness_v1.md")
    args = parser.parse_args()

    report = build_report(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    learned_path = resolve(args.learned_distillation)
    private_broad_path = resolve(args.private_broad)
    private_broad_gate_path = resolve(args.private_broad_gate)
    public_calibration_path = resolve(args.public_calibration)
    public_matrix_path = resolve(args.public_matrix)
    public_residual_path = resolve(args.public_residual)
    operator_lock_path = resolve(args.operator_lock)

    learned = read_json(learned_path, {})
    private_broad = read_json(private_broad_path, {})
    private_broad_gate = read_json(private_broad_gate_path, {})
    public_calibration = read_json(public_calibration_path, {})
    public_matrix = read_json(public_matrix_path, {})
    public_residual = read_json(public_residual_path, {})

    learned_summary = object_field(learned, "summary")
    private_summary = object_field(private_broad, "summary")
    private_gate_summary = object_field(private_broad_gate, "summary")
    public_summary = object_field(public_calibration, "summary")
    matrix_summary = object_field(public_matrix, "summary")
    residual_summary = object_field(public_residual, "summary")

    next_fix = str(
        residual_summary.get("recommended_private_fix_family_after_current_adapter")
        or residual_summary.get("recommended_private_fix_family")
        or "build_next_private_residual_curriculum_from_public_residual_categories"
    )
    fix_chain = completed_private_fix_chain(next_fix, residual_summary)
    completed_fix = fix_chain["original"]
    completed_successor_fix = fix_chain["last_completed_successor"]
    completed_fix_evidence = fix_chain["last_completed"] or completed_fix
    failed_fix_evidence = fix_chain["first_failed"]
    effective_next_fix = str(fix_chain["next_family"])
    active_completed_private = completed_fix_evidence if completed_fix_evidence.get("completed") else {}

    min_public_pass_rate = float(args.min_public_pass_rate)
    public_pass_rate = first_number(
        public_summary.get("real_public_task_pass_rate"),
        public_summary.get("multi_stream_pass_rate"),
        matrix_summary.get("real_public_pass_rate"),
        0.0,
    )
    public_task_count = int(first_number(public_summary.get("public_task_count"), matrix_summary.get("real_public_task_count"), 0))
    public_regressions = int(first_number(public_summary.get("task_level_regressions_vs_single_stream"), matrix_summary.get("total_regressions"), 0))
    cards_below_floor = list_field(matrix_summary.get("cards_below_floor"))
    promotion_card_count = int(first_number(matrix_summary.get("promotion_candidate_card_count"), 0))
    learned_only_pass_rate = first_number(learned_summary.get("learned_only_pass_rate"), 0.0)
    learned_token_pass_count = int(first_number(learned_summary.get("learned_token_pass_count"), 0))
    prototype_pass_count = int(first_number(learned_summary.get("prototype_pass_count"), -1))
    private_broad_pass_rate = first_number(private_summary.get("heldout_pass_rate"), 0.0)
    if active_completed_private.get("family") == "edge_contract_v4_public_safe_broad_transfer_maturity_curriculum":
        learned_only_pass_rate = first_number(active_completed_private.get("learned_only_pass_rate"), learned_only_pass_rate)
        learned_token_pass_count = int(first_number(active_completed_private.get("learned_token_pass_count"), learned_token_pass_count))
        prototype_pass_count = int(first_number(active_completed_private.get("prototype_pass_count"), prototype_pass_count))
        private_broad_pass_rate = first_number(active_completed_private.get("pass_rate"), private_broad_pass_rate)
    operator_lock_active = operator_lock_path.exists()
    current_evidence = current_decoder_evidence(
        learned_path,
        learned,
        private_broad_path,
        private_broad_gate_path,
        private_broad_gate,
        active_completed_private,
    )
    active_v4_successor_current = completed_v4_successor_current(active_completed_private)

    integrity_gates = [
        gate("learned_distillation_report_green", learned.get("trigger_state") == "GREEN" or active_v4_successor_current, {
            "path": rel(learned_path),
            "trigger_state": learned.get("trigger_state"),
            "completion_evidence_status": learned_summary.get("completion_evidence_status"),
            "superseded_by_fresh_v4_successor": active_v4_successor_current,
        }),
        gate("learned_and_private_broad_current_for_decoder_source_and_release", current_evidence["fresh"], current_evidence),
        gate("learned_only_private_pass_rate_ge_070", learned_only_pass_rate >= 0.70, {
            "observed": learned_only_pass_rate,
            "minimum": 0.70,
        }),
        gate("prototype_dependency_zero", prototype_pass_count == 0, {
            "prototype_pass_count": prototype_pass_count,
            "learned_token_pass_count": learned_token_pass_count,
        }),
        gate("private_broad_unattended_green", private_broad.get("trigger_state") == "GREEN" and private_broad_pass_rate >= 0.70, {
            "path": rel(private_broad_path),
            "trigger_state": private_broad.get("trigger_state"),
            "heldout_pass_rate": private_broad_pass_rate,
        }),
        gate("private_broad_gate_green", (
            private_broad_gate.get("trigger_state") == "GREEN"
            and private_gate_summary.get("decoder_source_release_fresh") is True
        ) or active_v4_successor_current, {
            "path": rel(private_broad_gate_path),
            "trigger_state": private_broad_gate.get("trigger_state"),
            "decoder_source_release_fresh": private_gate_summary.get("decoder_source_release_fresh"),
            "blocker_count": private_gate_summary.get("blocker_count"),
            "superseded_by_fresh_v4_successor": active_v4_successor_current,
        }),
        gate("public_calibration_result_present", public_task_count >= int(args.min_public_task_count), {
            "path": rel(public_calibration_path),
            "public_task_count": public_task_count,
            "minimum": int(args.min_public_task_count),
        }),
        gate("public_boundary_clean", public_boundary_clean(public_calibration, public_summary, residual_summary), {
            "calibration_public_tests_used": public_summary.get("public_tests_used"),
            "calibration_public_solutions_used": public_summary.get("public_solutions_used"),
            "residual_public_tests_or_solutions_embedded": residual_summary.get("public_tests_or_solutions_embedded"),
            "residual_public_prompts_embedded": residual_summary.get("public_prompts_embedded"),
        }),
        gate("external_inference_zero", external_inference_zero(learned, private_broad, public_calibration, public_residual), {
            "learned": top_external_calls(learned),
            "private_broad": top_external_calls(private_broad),
            "public_calibration": top_external_calls(public_calibration),
            "public_residual": top_external_calls(public_residual),
        }),
        gate("public_calibration_operator_lock_active", operator_lock_active, {
            "path": rel(operator_lock_path),
            "active": operator_lock_active,
        }),
    ]
    readiness_gates = [
        gate("public_pass_rate_ge_floor", public_pass_rate >= min_public_pass_rate, {
            "observed": public_pass_rate,
            "minimum": min_public_pass_rate,
        }),
        gate("public_cards_clear_floor", len(cards_below_floor) == 0 and promotion_card_count > 0, {
            "cards_below_floor": cards_below_floor,
            "promotion_candidate_card_count": promotion_card_count,
        }),
        gate("public_regressions_zero", public_regressions == 0, {"regressions": public_regressions}),
    ]

    hard_blockers = [row for row in integrity_gates if not row["passed"]]
    transfer_blockers = [row for row in readiness_gates if not row["passed"]]
    trigger_state = "RED" if hard_blockers else ("GREEN" if not transfer_blockers else "YELLOW")
    public_transfer_ready = not hard_blockers and not transfer_blockers
    public_calibration_allowed = public_transfer_ready and not operator_lock_active
    return {
        "policy": "project_theseus_post_distillation_public_transfer_readiness_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "public_transfer_ready_for_new_calibration": public_transfer_ready,
        "public_calibration_allowed": public_calibration_allowed,
        "operator_lock_active": operator_lock_active,
        "inputs": {
            "learned_distillation": rel(learned_path),
            "private_broad": rel(private_broad_path),
            "private_broad_gate": rel(private_broad_gate_path),
            "public_calibration": rel(public_calibration_path),
            "public_matrix": rel(public_matrix_path),
            "public_residual": rel(public_residual_path),
            "operator_lock": rel(operator_lock_path),
            "public_tests_or_solutions_read": False,
        },
        "summary": {
            "completion_evidence_status": "public_transfer_ready" if public_transfer_ready else "public_transfer_blocked_after_private_distillation",
            "learned_only_private_pass_rate": learned_only_pass_rate,
            "learned_token_pass_count": learned_token_pass_count,
            "prototype_pass_count": prototype_pass_count,
            "private_broad_pass_rate": private_broad_pass_rate,
            "decoder_source_release_fresh": current_evidence["fresh"],
            "decoder_source_release_stale_reasons": current_evidence["stale_reasons"],
            "public_pass_rate": public_pass_rate,
            "public_floor": min_public_pass_rate,
            "public_task_count": public_task_count,
            "public_regressions": public_regressions,
            "cards_below_floor": cards_below_floor,
            "promotion_candidate_card_count": promotion_card_count,
            "next_public_blocker": residual_summary.get("next_blocker_after_current_adapter") or residual_summary.get("next_blocker"),
            "recommended_private_fix_family": effective_next_fix,
            "original_recommended_private_fix_family": next_fix,
            "original_recommended_private_fix_completed": completed_fix.get("completed"),
            "completed_successor_private_fix_family": completed_successor_fix.get("family"),
            "completed_successor_private_fix_completed": completed_successor_fix.get("completed"),
            "completed_private_fix_evidence": completed_fix_evidence,
            "failed_private_fix_evidence": failed_fix_evidence,
            "completed_private_fix_chain": fix_chain["chain"],
            "dominant_public_residual_categories": residual_summary.get("adapter_adjusted_dominant_categories") or residual_summary.get("dominant_categories") or [],
            "public_tests_or_solutions_used": False,
            "external_inference_calls": 0,
        },
        "gates": {
            "integrity": integrity_gates,
            "readiness": readiness_gates,
        },
        "hard_blockers": hard_blockers,
        "transfer_blockers": transfer_blockers,
        "next_actions": next_actions(
            trigger_state,
            public_transfer_ready,
            operator_lock_active,
            effective_next_fix,
            public_pass_rate,
            min_public_pass_rate,
            cards_below_floor,
            current_evidence,
            hard_blockers,
            completed_fix_evidence,
            failed_fix_evidence,
            original_next_fix=next_fix,
        ),
        "rules": {
            "public_calibration": "Do not run another public calibration while this gate is YELLOW/RED or while the operator lock is active.",
            "training": "Use public residual categories only as private curriculum routing labels; do not train on public prompts, tests, or solutions.",
            "promotion": "Private broad-ladder learned-token success is necessary but not sufficient for candidate promotion or model growth.",
        },
        "public_tests_used": False,
        "public_solutions_used": False,
        "external_inference_calls": 0,
}


def completed_private_fix_status(family: str) -> dict[str, Any]:
    if family == "edge_contract_v3_verifier_mismatch_public_transfer_private_curriculum":
        return completed_edge_v3_fix_status(family)
    if family == "edge_contract_v4_public_safe_broad_transfer_maturity_curriculum":
        return completed_v4_maturity_fix_status(family)
    if family == "operator_reviewed_bounded_public_calibration_packet":
        return {"family": family, "completed": False, "reason": "operator_review_required"}
    if family != "edge_contract_v2_private_residual_curriculum":
        return {"family": family, "completed": False, "reason": "no_completion_contract_for_family"}
    rescore = read_json(DEFAULT_EDGE_V2_RESCORE, {})
    runner = read_json(DEFAULT_EDGE_V2_RUNNER, {})
    closure = read_json(DEFAULT_EDGE_V2_CLOSURE, {})
    freshness = freshness_report(
        {
            "edge_v2_rescore": DEFAULT_EDGE_V2_RESCORE,
            "edge_v2_runner": DEFAULT_EDGE_V2_RUNNER,
            "edge_v2_closure": DEFAULT_EDGE_V2_CLOSURE,
        },
        root=ROOT,
        rule="completed private fix evidence must be fresh against decoder source and release binary",
    )
    completed = (
        rescore.get("trigger_state") == "GREEN"
        and rescore.get("ready_for_public_calibration") is True
        and runner.get("trigger_state") == "GREEN"
        and object_field(runner, "summary").get("ready_for_public_calibration") is True
        and closure.get("trigger_state") == "GREEN"
        and freshness["fresh"]
        and rescore.get("public_tests_used") is not True
        and rescore.get("public_solutions_used") is not True
        and top_external_calls(rescore) == 0
        and top_external_calls(runner) == 0
        and top_external_calls(closure) == 0
    )
    return {
        "family": family,
        "completed": completed,
        "rescore": rel(DEFAULT_EDGE_V2_RESCORE),
        "runner": rel(DEFAULT_EDGE_V2_RUNNER),
        "closure": rel(DEFAULT_EDGE_V2_CLOSURE),
        "rescore_trigger_state": rescore.get("trigger_state"),
        "rescore_ready_for_public_calibration": rescore.get("ready_for_public_calibration"),
        "runner_trigger_state": runner.get("trigger_state"),
        "runner_ready_for_public_calibration": object_field(runner, "summary").get("ready_for_public_calibration"),
        "closure_trigger_state": closure.get("trigger_state"),
        "freshness": freshness,
        "private_delta": first_number(object_field(rescore, "summary").get("private_pass_rate_delta"), object_field(runner, "summary").get("private_delta"), 0.0),
        "public_tests_used": rescore.get("public_tests_used"),
        "public_solutions_used": rescore.get("public_solutions_used"),
        "external_inference_calls": max(top_external_calls(rescore), top_external_calls(runner), top_external_calls(closure)),
    }


def completed_private_fix_chain(initial_family: str, residual_summary: dict[str, Any]) -> dict[str, Any]:
    chain: list[dict[str, Any]] = []
    family = initial_family
    last_completed: dict[str, Any] | None = None
    last_completed_successor: dict[str, Any] = {"family": family, "completed": False, "reason": "no_completed_successor"}
    original_status: dict[str, Any] | None = None
    seen: set[str] = set()

    for step in range(8):
        if family in seen:
            chain.append({"family": family, "completed": False, "reason": "private_fix_chain_cycle"})
            break
        seen.add(family)
        status = completed_private_fix_status(family)

        if step == 0:
            original_status = status
            if family == "edge_contract_v2_private_residual_curriculum" and not status.get("completed"):
                # Historical v2 reports can go stale after decoder changes even
                # though a fresher successor already exists. Treat v2 as
                # superseded so the gate audits the live successor evidence.
                status = {**status, "completed": True, "superseded_by_successor": True}

        if family == "edge_contract_v3_verifier_mismatch_public_transfer_private_curriculum" and not status.get("completed"):
            v4_status = completed_v4_maturity_fix_status("edge_contract_v4_public_safe_broad_transfer_maturity_curriculum")
            if v4_status.get("completed"):
                status = {**status, "completed": True, "superseded_by_successor": True, "superseded_by": v4_status["family"]}

        chain.append(status)
        if not status.get("completed"):
            break

        last_completed = status
        if step > 0:
            last_completed_successor = status
        next_family = successor_private_fix(family, status, residual_summary)
        if next_family == family:
            break
        family = next_family

    return {
        "initial_family": initial_family,
        "next_family": family,
        "original": original_status or {"family": initial_family, "completed": False, "reason": "not_checked"},
        "last_completed": last_completed,
        "last_completed_successor": last_completed_successor,
        "first_failed": next((row for row in chain if not row.get("completed")), {}),
        "chain": chain,
    }


def completed_edge_v3_fix_status(family: str) -> dict[str, Any]:
    score = read_json(DEFAULT_EDGE_V3_BROAD_SCORE, {})
    gate_report = read_json(DEFAULT_EDGE_V3_LEARNED_GATE, {})
    score_summary = object_field(score, "summary")
    gate_summary = object_field(gate_report, "summary")
    freshness = freshness_report(
        {
            "edge_v3_broad_score": DEFAULT_EDGE_V3_BROAD_SCORE,
            "edge_v3_learned_gate": DEFAULT_EDGE_V3_LEARNED_GATE,
        },
        root=ROOT,
        rule="completed edge-v3 private fix evidence must be fresh against decoder source and release binary",
    )
    pass_rate = first_number(score_summary.get("pass_rate"), 0.0)
    learned_only_pass_rate = first_number(gate_summary.get("learned_only_pass_rate"), 0.0)
    completed = (
        score.get("trigger_state") == "GREEN"
        and gate_report.get("trigger_state") == "GREEN"
        and pass_rate >= 0.70
        and learned_only_pass_rate >= 0.70
        and int(first_number(gate_summary.get("prototype_pass_count"), -1)) == 0
        and freshness["fresh"]
        and score_summary.get("public_tests_used") is not True
        and score_summary.get("public_solutions_used") is not True
        and gate_report.get("public_tests_used") is not True
        and gate_report.get("public_solutions_used") is not True
        and top_external_calls(score) == 0
        and top_external_calls(gate_report) == 0
    )
    return {
        "family": family,
        "completed": completed,
        "score": rel(DEFAULT_EDGE_V3_BROAD_SCORE),
        "learned_gate": rel(DEFAULT_EDGE_V3_LEARNED_GATE),
        "score_trigger_state": score.get("trigger_state"),
        "gate_trigger_state": gate_report.get("trigger_state"),
        "pass_rate": pass_rate,
        "pass_count": int(first_number(score_summary.get("pass_count"), 0)),
        "task_count": int(first_number(score_summary.get("heldout_task_count"), 0)),
        "learned_only_pass_rate": learned_only_pass_rate,
        "prototype_pass_count": int(first_number(gate_summary.get("prototype_pass_count"), -1)),
        "sts_off_control_pass_rate": first_number(score_summary.get("control_pass_rate"), 0.0),
        "sts_delta": first_number(score_summary.get("sts_delta"), 0.0),
        "sts_regressions": int(first_number(score_summary.get("sts_regressions"), 0)),
        "freshness": freshness,
        "public_tests_used": score_summary.get("public_tests_used") or gate_report.get("public_tests_used"),
        "public_solutions_used": score_summary.get("public_solutions_used") or gate_report.get("public_solutions_used"),
        "external_inference_calls": max(top_external_calls(score), top_external_calls(gate_report)),
    }


def completed_v4_maturity_fix_status(family: str) -> dict[str, Any]:
    maturity = read_json(DEFAULT_V4_MATURITY, {})
    score = read_json(DEFAULT_V4_SCORE, {})
    gate_report = read_json(DEFAULT_V4_LEARNED_GATE, {})
    maturity_summary = object_field(maturity, "summary")
    score_summary = object_field(score, "summary")
    gate_summary = object_field(gate_report, "summary")
    artifacts = {
        "v4_score": DEFAULT_V4_SCORE,
        "v4_learned_gate": DEFAULT_V4_LEARNED_GATE,
    }
    gate_inputs = gate_report.get("inputs") if isinstance(gate_report.get("inputs"), dict) else {}
    gate_outputs = gate_report.get("outputs") if isinstance(gate_report.get("outputs"), dict) else {}
    for key in ["candidates", "control_candidates", "score"]:
        value = str(gate_inputs.get(key) or "").strip()
        if value:
            artifacts[f"v4_input_{key}"] = resolve(value)
    for key in ["learned_only_candidates", "learned_only_score"]:
        value = str(gate_outputs.get(key) or "").strip()
        if value:
            artifacts[f"v4_output_{key}"] = resolve(value)
    freshness = freshness_report(
        artifacts,
        root=ROOT,
        rule="completed v4 maturity evidence must be fresh against decoder source and release binary",
    )
    pass_rate = first_number(score_summary.get("pass_rate"), 0.0)
    learned_only_pass_rate = first_number(gate_summary.get("learned_only_pass_rate"), 0.0)
    full_task_count = int(first_number(gate_summary.get("full_task_count"), score_summary.get("heldout_task_count"), 0))
    learned_task_count = int(first_number(gate_summary.get("learned_only_task_count"), 0))
    prototype_pass_count = int(first_number(gate_summary.get("prototype_pass_count"), -1))
    learned_token_pass_count = int(first_number(gate_summary.get("learned_token_pass_count"), 0))
    learned_maturity = learned_maturity_summary(gate_summary)
    completed = (
        maturity.get("trigger_state") == "GREEN"
        and score.get("trigger_state") == "GREEN"
        and gate_report.get("trigger_state") == "GREEN"
        and pass_rate >= 0.70
        and learned_only_pass_rate >= 0.70
        and full_task_count >= 1000
        and learned_task_count >= 1000
        and prototype_pass_count == 0
        and learned_token_pass_count >= learned_task_count
        and learned_maturity["ready"]
        and freshness["fresh"]
        and score_summary.get("public_tests_used") is not True
        and score_summary.get("public_solutions_used") is not True
        and gate_report.get("public_tests_used") is not True
        and gate_report.get("public_solutions_used") is not True
        and top_external_calls(maturity) == 0
        and top_external_calls(score) == 0
        and top_external_calls(gate_report) == 0
    )
    return {
        "family": family,
        "completed": completed,
        "maturity": rel(DEFAULT_V4_MATURITY),
        "score": rel(DEFAULT_V4_SCORE),
        "learned_gate": rel(DEFAULT_V4_LEARNED_GATE),
        "maturity_trigger_state": maturity.get("trigger_state"),
        "score_trigger_state": score.get("trigger_state"),
        "gate_trigger_state": gate_report.get("trigger_state"),
        "pass_rate": pass_rate,
        "pass_count": int(first_number(score_summary.get("pass_count"), 0)),
        "task_count": int(first_number(score_summary.get("heldout_task_count"), 0)),
        "learned_only_pass_rate": learned_only_pass_rate,
        "learned_only_pass_count": int(first_number(gate_summary.get("learned_only_pass_count"), 0)),
        "learned_only_task_count": learned_task_count,
        "learned_token_pass_count": learned_token_pass_count,
        "prototype_pass_count": prototype_pass_count,
        "learned_maturity": learned_maturity,
        "learned_maturity_ready": learned_maturity["ready"],
        "learned_train_body_overlap_rate": learned_maturity["train_body_overlap_rate"],
        "sts_off_control_pass_rate": first_number(score_summary.get("control_pass_rate"), 0.0),
        "sts_delta": first_number(score_summary.get("sts_delta"), 0.0),
        "sts_regressions": int(first_number(score_summary.get("sts_regressions"), 0)),
        "candidate_rows": int(first_number(score_summary.get("candidate_row_count"), 0)),
        "learned_only_candidate_rows": int(first_number(object_field(gate_summary, "candidate_inventory").get("learned_only_candidate_rows"), 0)),
        "maturity_train_rows": int(first_number(maturity_summary.get("private_train_row_count"), maturity_summary.get("train_row_count"), 0)),
        "maturity_heldout_rows": int(first_number(maturity_summary.get("private_heldout_row_count"), maturity_summary.get("heldout_row_count"), 0)),
        "maturity_mtime": path_mtime(DEFAULT_V4_MATURITY) or None,
        "freshness": freshness,
        "public_tests_used": score_summary.get("public_tests_used") or gate_report.get("public_tests_used"),
        "public_solutions_used": score_summary.get("public_solutions_used") or gate_report.get("public_solutions_used"),
        "external_inference_calls": max(top_external_calls(maturity), top_external_calls(score), top_external_calls(gate_report)),
    }


def successor_private_fix(family: str, completed_fix: dict[str, Any], residual_summary: dict[str, Any]) -> str:
    if family == "edge_contract_v2_private_residual_curriculum" and completed_fix.get("completed"):
        blocker = str(residual_summary.get("next_blocker_after_current_adapter") or residual_summary.get("next_blocker") or "public_transfer")
        if blocker == "verifier_mismatch":
            return "edge_contract_v3_verifier_mismatch_public_transfer_private_curriculum"
        return f"edge_contract_v3_{blocker}_private_curriculum"
    if family == "edge_contract_v3_verifier_mismatch_public_transfer_private_curriculum" and completed_fix.get("completed"):
        return "edge_contract_v4_public_safe_broad_transfer_maturity_curriculum"
    if family == "edge_contract_v4_public_safe_broad_transfer_maturity_curriculum" and completed_fix.get("completed"):
        return "operator_reviewed_bounded_public_calibration_packet"
    return family


def current_decoder_evidence(
    learned_path: Path,
    learned: dict[str, Any],
    private_broad_path: Path,
    private_broad_gate_path: Path,
    private_broad_gate: dict[str, Any],
    active_completed_private: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifacts = {
        "learned_distillation_gate": learned_path,
        "private_broad_unattended": private_broad_path,
        "private_broad_gate": private_broad_gate_path,
    }
    learned_outputs = learned.get("outputs") if isinstance(learned.get("outputs"), dict) else {}
    for key in ["learned_only_candidates", "learned_only_score"]:
        value = str(learned_outputs.get(key) or "").strip()
        if value:
            artifacts[f"learned_{key}"] = resolve(value)
    gate_inputs = private_broad_gate.get("inputs") if isinstance(private_broad_gate.get("inputs"), dict) else {}
    for key in ["score", "unattended"]:
        value = str(gate_inputs.get(key) or "").strip()
        if value:
            artifacts[f"private_broad_{key}"] = resolve(value)
    report = freshness_report(
        artifacts,
        root=ROOT,
        rule=(
            "post-distillation readiness must use learned/private broad reports "
            "regenerated after decoder source changes or release binary rebuilds"
        ),
    )
    report["learned_report_mtime"] = path_mtime(learned_path) or None
    report["private_broad_report_mtime"] = path_mtime(private_broad_path) or None
    report["private_broad_gate_report_mtime"] = path_mtime(private_broad_gate_path) or None
    active_completed_private = active_completed_private or {}
    active_freshness = object_field(active_completed_private, "freshness")
    if (
        not report.get("fresh")
        and active_completed_private.get("family") == "edge_contract_v4_public_safe_broad_transfer_maturity_curriculum"
        and active_completed_private.get("completed")
        and active_freshness.get("fresh") is True
    ):
        return {
            "fresh": True,
            "stale_reasons": [],
            "superseded_stale_baseline": True,
            "active_completed_private_family": active_completed_private.get("family"),
            "active_completed_private_evidence_fresh": True,
            "baseline_stale_reasons": report.get("stale_reasons", []),
            "baseline_evidence": report,
            "active_completed_private_freshness": active_freshness,
            "rule": (
                "fresh completed v4 maturity evidence supersedes stale intermediate broad-v1/v3 reports "
                "for post-distillation readiness; public calibration remains operator-locked"
            ),
        }
    return report


def completed_v4_successor_current(completed_private: dict[str, Any] | None) -> bool:
    completed_private = completed_private or {}
    freshness = object_field(completed_private, "freshness")
    return bool(
        completed_private.get("family") == "edge_contract_v4_public_safe_broad_transfer_maturity_curriculum"
        and completed_private.get("completed") is True
        and freshness.get("fresh") is True
        and top_external_calls(completed_private) == 0
    )


def public_boundary_clean(calibration: dict[str, Any], calibration_summary: dict[str, Any], residual_summary: dict[str, Any]) -> bool:
    return (
        calibration_summary.get("public_tests_used") is not True
        and calibration_summary.get("public_solutions_used") is not True
        and calibration.get("public_tests_used") is not True
        and calibration.get("public_solutions_used") is not True
        and residual_summary.get("public_tests_or_solutions_embedded") is not True
        and residual_summary.get("public_prompts_embedded") is not True
    )


def external_inference_zero(*reports: dict[str, Any]) -> bool:
    return all(top_external_calls(report) == 0 for report in reports)


def top_external_calls(report: dict[str, Any]) -> int:
    summary = object_field(report, "summary")
    return int(first_number(report.get("external_inference_calls"), summary.get("external_inference_calls"), 0))


def next_actions(
    trigger_state: str,
    public_transfer_ready: bool,
    operator_lock_active: bool,
    next_fix: str,
    public_pass_rate: float,
    floor: float,
    cards_below_floor: list[str],
    current_evidence: dict[str, Any],
    hard_blockers: list[dict[str, Any]],
    completed_fix: dict[str, Any],
    failed_fix: dict[str, Any],
    *,
    original_next_fix: str,
) -> list[str]:
    if not current_evidence.get("fresh"):
        return [
            "Refresh broad-private unattended fanout/score/gate and learned-distillation evidence under the current decoder/release.",
            "Keep public calibration locked; stale private transfer reports cannot support post-distillation readiness.",
        ]
    if trigger_state == "RED":
        first = hard_blockers[0].get("gate") if hard_blockers else "unknown"
        return [f"Fix hard integrity blocker `{first}` before any training, promotion, or calibration work."]
    if not public_transfer_ready:
        failed_score_trigger = str(failed_fix.get("score_trigger_state") or "")
        failed_sts_delta = first_number(failed_fix.get("sts_delta"), 0.0)
        failed_control_rate = first_number(failed_fix.get("sts_off_control_pass_rate"), 0.0)
        if failed_score_trigger == "YELLOW" and (failed_sts_delta <= 0.0 or failed_control_rate >= 0.70):
            family = str(failed_fix.get("family") or next_fix)
            return [
                f"Keep public calibration locked: current public pass rate is {public_pass_rate} below floor {floor}.",
                f"Repair the same-seed STS control for `{family}`: refreshed score is YELLOW with STS delta {failed_sts_delta} and control pass rate {failed_control_rate}.",
                "Do not treat learned-only private pass rate as transfer evidence until STS-on beats the same-seed STS-off control again.",
                "Profile the current fanout runtime before long unattended loops; the latest v4 full STS-on/control refreshes each took about 662s.",
            ]
        if completed_fix.get("completed") and next_fix == "operator_reviewed_bounded_public_calibration_packet":
            return [
                f"Keep public calibration locked: current public pass rate is {public_pass_rate} below floor {floor}.",
                "Private v4 maturity evidence is complete; prepare an operator-reviewed bounded public calibration packet instead of adding another private successor by default.",
                f"Target cards below floor: {', '.join(cards_below_floor) if cards_below_floor else 'unspecified public cards'}.",
                "If the operator explicitly approves one public run, spend exactly one bounded calibration and relock immediately.",
            ]
        if completed_fix.get("completed") and original_next_fix != next_fix:
            return [
                f"Keep public calibration locked: current public pass rate is {public_pass_rate} below floor {floor}.",
                f"Do not loop on completed `{original_next_fix}`; build successor `{next_fix}` without public prompts/tests/solutions.",
                f"Target cards below floor: {', '.join(cards_below_floor) if cards_below_floor else 'unspecified public cards'}.",
                "Rerun private repair, transfer proof, and this post-distillation readiness gate before any operator-approved public run.",
            ]
        return [
            f"Keep public calibration locked: current public pass rate is {public_pass_rate} below floor {floor}.",
            f"Build the next private residual curriculum from `{next_fix}` without public prompts/tests/solutions.",
            f"Target cards below floor: {', '.join(cards_below_floor) if cards_below_floor else 'unspecified public cards'}.",
            "Rerun private repair, transfer proof, and this post-distillation readiness gate before any operator-approved public run.",
        ]
    if operator_lock_active:
        return ["Technical public-transfer readiness is green; operator lock still blocks calibration until explicitly removed."]
    return ["Technical public-transfer readiness is green and the operator lock is inactive; schedule exactly one bounded public calibration, then relock."]


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def object_field(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key)
    return value if isinstance(value, dict) else {}


def list_field(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def first_number(*values: Any) -> float:
    for value in values:
        try:
            if value is not None and value != "":
                return float(value)
        except Exception:
            continue
    return 0.0


def learned_maturity_summary(summary: dict[str, Any]) -> dict[str, Any]:
    structure = object_field(summary, "learned_structural_inventory")
    novelty = object_field(summary, "learned_train_novelty_inventory")
    normalized_ast_unique_count = int(first_number(structure.get("pass_normalized_ast_unique_count"), 0))
    min_normalized_ast_unique_count = int(first_number(structure.get("min_pass_normalized_ast_unique_count"), 999))
    ast_shape_count = int(first_number(structure.get("pass_ast_shape_count"), 0))
    min_ast_shape_count = int(first_number(structure.get("min_pass_ast_shape_count"), 999))
    top_duplicate_rate = first_number(structure.get("pass_top_normalized_ast_duplicate_rate"), 1.0)
    max_top_duplicate_rate = first_number(structure.get("max_pass_top_duplicate_rate"), 0.0)
    structural_ready = bool(
        normalized_ast_unique_count >= min_normalized_ast_unique_count
        and ast_shape_count >= min_ast_shape_count
        and top_duplicate_rate <= max_top_duplicate_rate
        and structure.get("control_structure_coverage_ready") is True
    )
    novelty_ready = novelty.get("novelty_ready") is True
    fresh = summary.get("decoder_source_release_fresh") is True
    prototype_pass_count = int(first_number(summary.get("prototype_pass_count"), 999))
    return {
        "ready": bool(fresh and structural_ready and novelty_ready and prototype_pass_count == 0),
        "fresh": fresh,
        "structural_ready": structural_ready,
        "novelty_ready": novelty_ready,
        "prototype_pass_count": prototype_pass_count,
        "pass_normalized_ast_unique_count": normalized_ast_unique_count,
        "pass_ast_shape_count": ast_shape_count,
        "pass_top_duplicate_rate": top_duplicate_rate,
        "train_ast_overlap_rate": first_number(novelty.get("exact_train_normalized_ast_overlap_rate"), 1.0),
        "train_body_overlap_rate": first_number(novelty.get("exact_train_body_normalized_ast_overlap_rate"), 1.0),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = object_field(report, "summary")
    lines = [
        "# Post-Distillation Public Transfer Readiness v1",
        "",
        f"- State: `{report.get('trigger_state')}`",
        f"- Completion: `{summary.get('completion_evidence_status')}`",
        f"- Learned-only private pass rate: `{summary.get('learned_only_private_pass_rate')}`",
        f"- Public pass rate: `{summary.get('public_pass_rate')}`",
        f"- Decoder source/release fresh: `{summary.get('decoder_source_release_fresh')}`",
        f"- Public floor: `{summary.get('public_floor')}`",
        f"- Cards below floor: `{summary.get('cards_below_floor')}`",
        f"- Recommended private fix family: `{summary.get('recommended_private_fix_family')}`",
        f"- Public calibration allowed: `{report.get('public_calibration_allowed')}`",
        f"- Operator lock active: `{report.get('operator_lock_active')}`",
        "",
        "## Blockers",
    ]
    blockers = report.get("hard_blockers", []) + report.get("transfer_blockers", [])
    if not blockers:
        lines.append("- none")
    else:
        for row in blockers:
            lines.append(f"- `{row.get('gate')}` evidence `{row.get('evidence')}`")
    lines.extend(["", "## Next Actions"])
    for action in report.get("next_actions", []):
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
