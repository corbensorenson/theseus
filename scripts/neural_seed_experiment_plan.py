#!/usr/bin/env python3
"""Write the first measured neural-seed experiment gate.

This is a planning/gating artifact only. It does not train a model, call the
teacher, fetch data, or spend public calibration. The report is intentionally
usable by teacher-distillation and overnight handoff gates: a spec may be ready
while the neural student itself remains not ready.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="configs/neural_seed_experiment_policy.json")
    parser.add_argument("--out", default="reports/neural_seed_growth_gate.json")
    parser.add_argument("--markdown-out", default="reports/neural_seed_growth_gate.md")
    args = parser.parse_args()

    policy = read_json(resolve(args.policy))
    state = load_state()
    report = build_report(policy, state, args.policy)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 2 if report["trigger_state"] == "RED" else 0


def load_state() -> dict[str, Any]:
    return {
        "external": read_json(REPORTS / "external_inference_audit.json"),
        "teacher_distillation": read_json(REPORTS / "teacher_distillation_gate.json"),
        "model_growth": read_json(REPORTS / "model_growth_gate.json"),
        "overnight": read_json(REPORTS / "overnight_learning_readiness.json"),
        "architecture": read_json(REPORTS / "architecture_experiment_governance.json"),
        "causal_architecture_delta": read_json(REPORTS / "causal_architecture_delta_loop.json"),
        "adapter": read_json(REPORTS / "benchmark_adapter_factory.json"),
        "genesis": read_json(REPORTS / "genesis_kernel" / "report.json"),
        "resource": read_json(REPORTS / "resource_governor.json"),
        "mlx_parity": read_json(REPORTS / "macos_mlx_parity_audit.json"),
        "mlx_work": read_json(REPORTS / "macos_mlx_work_proof.json"),
        "substrate_comparator": read_json(REPORTS / "neural_seed_substrate_comparator.json"),
        "code_proposer_comparator": read_json(REPORTS / "neural_seed_code_proposer_comparator.json"),
        "token_decoder_comparator": read_json(REPORTS / "neural_seed_token_decoder_comparator.json"),
        "token_decoder_multiseed": read_json(REPORTS / "neural_seed_token_decoder_multiseed_smoke.json"),
        "semantic_plan_gap_audit": read_json(REPORTS / "neural_seed_semantic_plan_gap_audit.json"),
        "code_proposer_gap_report": read_json(REPORTS / "neural_seed_code_proposer_gap_report.json"),
        "architecture_sweep": read_json(REPORTS / "neural_seed_architecture_sweep.json"),
        "residual_mining": read_json(REPORTS / "neural_seed_residual_mining.json"),
        "token_decoder_route_ablation": read_json(
            REPORTS / "neural_seed_token_decoder_route_independence_ablation.json"
        ),
        "token_decoder_complementarity": read_json(REPORTS / "neural_seed_token_decoder_complementarity_audit.json"),
        "token_decoder_residual_context": read_json(
            REPORTS / "neural_seed_token_decoder_residual_context_miner.json"
        ),
        "structural_action_ablation": read_json(REPORTS / "neural_seed_structural_action_ablation_report.json"),
    }


def build_report(policy: dict[str, Any], state: dict[str, Any], policy_path: str) -> dict[str, Any]:
    checks = build_checks(policy, state)
    hard = [row for row in checks if row["severity"] == "hard" and not row["passed"]]
    readiness = [row for row in checks if row["severity"] != "hard" and not row["passed"]]
    spec_ready = not hard and not readiness
    distillation_evidence = student_distillation_evidence(policy, state, spec_ready)
    execute_allowed = bool(policy.get("default_execute_allowed")) and bool(state["model_growth"].get("model_growth_allowed"))
    trigger_state = "RED" if hard else "YELLOW"
    if execute_allowed and spec_ready:
        trigger_state = "GREEN"
    arms = policy.get("arms") if isinstance(policy.get("arms"), list) else []
    matched_budget = policy.get("matched_budget") if isinstance(policy.get("matched_budget"), dict) else {}
    report = {
        "policy": "project_theseus_neural_seed_growth_gate_v0",
        "created_utc": now(),
        "config": policy_path,
        "experiment_id": policy.get("experiment_id"),
        "trigger_state": trigger_state,
        "spec_ready": spec_ready,
        "execute_allowed": execute_allowed,
        "neural_student_ready": False,
        "student_distillation_evidence_ready": distillation_evidence["ready"],
        "summary": {
            "spec_ready": spec_ready,
            "execute_allowed": execute_allowed,
            "neural_student_ready": False,
            "student_distillation_evidence_ready": distillation_evidence["ready"],
            "student_distillation_evidence_blockers": distillation_evidence["blockers"],
            "student_distillation_evidence_semantics": (
                "Private neural-seed evidence is ready for a future governed "
                "teacher-distillation manifest. This is not model growth, "
                "promotion, public calibration, or runtime serving readiness."
            ),
            "matched_arms": len(arms),
            "matched_budget_id": matched_budget.get("budget_id"),
            "model_growth_allowed": bool(state["model_growth"].get("model_growth_allowed")),
            "model_growth_missing_evidence": state["model_growth"].get("missing_evidence", []),
            "model_growth_hard_blockers": state["model_growth"].get("hard_blockers", []),
            "adapter_ready_cards": get_path(state, ["adapter", "summary", "ready_cards"], None),
            "genesis_failed_hard_gates": genesis_failed_hard_gates(state["genesis"]),
            "substrate_comparator": substrate_comparator_summary(state["substrate_comparator"]),
            "code_proposer_comparator": code_proposer_comparator_summary(state["code_proposer_comparator"]),
            "token_decoder_comparator": token_decoder_comparator_summary(state["token_decoder_comparator"]),
            "token_decoder_multiseed": token_decoder_multiseed_summary(state["token_decoder_multiseed"]),
            "semantic_plan_gap_audit": semantic_plan_gap_audit_summary(state["semantic_plan_gap_audit"]),
            "code_proposer_gap_report": code_proposer_gap_summary(state["code_proposer_gap_report"]),
            "architecture_sweep": architecture_sweep_summary(state["architecture_sweep"]),
            "residual_mining": residual_mining_summary(state["residual_mining"]),
            "token_decoder_route_ablation": token_decoder_route_ablation_summary(
                state["token_decoder_route_ablation"]
            ),
            "token_decoder_complementarity": token_decoder_complementarity_summary(
                state["token_decoder_complementarity"]
            ),
            "token_decoder_residual_context": token_decoder_residual_context_summary(
                state["token_decoder_residual_context"]
            ),
            "structural_action_ablation": structural_action_ablation_summary(state["structural_action_ablation"]),
            "external_inference_calls": 0,
        },
        "checks": checks,
        "hard_blockers": [row["name"] for row in hard],
        "readiness_blockers": [row["name"] for row in readiness],
        "experiment": {
            "objective": policy.get("objective"),
            "arms": arms,
            "matched_budget": matched_budget,
            "data_contract": policy.get("data_contract", {}),
            "measurement_contract": policy.get("measurement_contract", []),
            "promotion_boundary": policy.get("promotion_boundary", {}),
        },
        "substrate_comparator": substrate_comparator_summary(state["substrate_comparator"]),
        "code_proposer_comparator": code_proposer_comparator_summary(state["code_proposer_comparator"]),
        "token_decoder_comparator": token_decoder_comparator_summary(state["token_decoder_comparator"]),
        "token_decoder_multiseed": token_decoder_multiseed_summary(state["token_decoder_multiseed"]),
        "semantic_plan_gap_audit": semantic_plan_gap_audit_summary(state["semantic_plan_gap_audit"]),
        "code_proposer_gap_report": code_proposer_gap_summary(state["code_proposer_gap_report"]),
        "architecture_sweep": architecture_sweep_summary(state["architecture_sweep"]),
        "residual_mining": residual_mining_summary(state["residual_mining"]),
        "token_decoder_route_ablation": token_decoder_route_ablation_summary(state["token_decoder_route_ablation"]),
        "token_decoder_complementarity": token_decoder_complementarity_summary(
            state["token_decoder_complementarity"]
        ),
        "token_decoder_residual_context": token_decoder_residual_context_summary(
            state["token_decoder_residual_context"]
        ),
        "structural_action_ablation": structural_action_ablation_summary(state["structural_action_ablation"]),
        "student_distillation_evidence": distillation_evidence,
        "macos_constraints": macos_constraints(policy, state),
        "runbook": {
            "generate_gate": "python3 scripts/neural_seed_experiment_plan.py",
            "pre_launch_checks": [
                "python3 scripts/model_growth_gate.py --out reports/model_growth_gate.json",
                "python3 scripts/teacher_distillation_gate.py --out reports/teacher_distillation_gate.json --markdown-out reports/teacher_distillation_gate.md",
                "python3 scripts/overnight_learning_readiness.py --out reports/overnight_learning_readiness.json"
            ],
            "execution_status": "not launched by this report",
        },
        "next_action": next_action(spec_ready, execute_allowed, hard, readiness, state),
        "score_semantics": (
            "Neural-seed gate generation only. This script does not train, call the teacher, "
            "spend public calibration, fetch data, or promote. Comparator fields summarize "
            "reports/neural_seed_substrate_comparator.json and "
            "reports/neural_seed_code_proposer_comparator.json, "
            "reports/neural_seed_code_proposer_gap_report.json, and "
            "reports/neural_seed_token_decoder_comparator.json, "
            "reports/neural_seed_token_decoder_multiseed_smoke.json, "
            "reports/neural_seed_semantic_plan_gap_audit.json, "
            "reports/neural_seed_architecture_sweep.json, "
            "reports/neural_seed_residual_mining.json, "
            "reports/neural_seed_token_decoder_route_independence_ablation.json, "
            "reports/neural_seed_token_decoder_complementarity_audit.json, "
            "reports/neural_seed_token_decoder_residual_context_miner.json, and "
            "reports/neural_seed_structural_action_ablation_report.json when present."
        ),
        "external_inference_calls": 0,
    }
    return report


def student_distillation_evidence(
    policy: dict[str, Any],
    state: dict[str, Any],
    spec_ready: bool,
) -> dict[str, Any]:
    """Separate student evidence from growth/promotion readiness.

    `neural_student_ready` remains a model-growth/promotion flag. Teacher
    distillation needs a narrower signal: whether a small private student path
    exists behind the verifier harness, with no public data, teacher calls, or
    fallback returns. This signal can be true while model growth stays locked.
    """

    token_report = state.get("token_decoder_comparator", {})
    multiseed_report = state.get("token_decoder_multiseed", {})
    route_report = state.get("token_decoder_route_ablation", {})
    complementarity_report = state.get("token_decoder_complementarity", {})
    residual_context_report = state.get("token_decoder_residual_context", {})
    structural_report = state.get("structural_action_ablation", {})

    structural_summary = structural_report.get("summary") if isinstance(structural_report, dict) else {}
    standalone = (
        structural_summary.get("standalone_96eval_summary")
        if isinstance(structural_summary.get("standalone_96eval_summary"), dict)
        else {}
    )
    structural_compiler = get_path(structural_report, ["axes", "line_action_compilation"], {})
    no_cheat = {
        "external_inference_calls": sum(
            int(report.get("external_inference_calls") or 0)
            for report in [
                token_report,
                multiseed_report,
                route_report,
                complementarity_report,
                residual_context_report,
                structural_report,
            ]
            if isinstance(report, dict)
        ),
        "teacher_used": bool(structural_summary.get("teacher_used")),
        "public_training_rows": int(structural_summary.get("public_training_rows") or 0),
        "fallback_return_rows_total": int(standalone.get("fallback_return_rows_total") or 0),
        "line_action_fallback_return_rate_max": float(
            structural_compiler.get("fallback_return_rate_max")
            if structural_compiler.get("fallback_return_rate_max") is not None
            else 1.0
        ),
    }
    gates = {
        "spec_ready": bool(spec_ready),
        "token_decoder_comparator_ready": token_decoder_comparator_ready(token_report),
        "token_decoder_multiseed_ready": token_decoder_multiseed_ready(multiseed_report),
        "token_decoder_route_ablation_ready": token_decoder_route_ablation_ready(route_report),
        "token_decoder_complementarity_ready": token_decoder_complementarity_ready(complementarity_report),
        "token_decoder_residual_context_ready": token_decoder_residual_context_ready(residual_context_report),
        "structural_action_ablation_ready": structural_action_ablation_ready(structural_report),
        "no_external_inference": no_cheat["external_inference_calls"] == 0,
        "no_teacher_used": not no_cheat["teacher_used"],
        "no_public_training_rows": no_cheat["public_training_rows"] == 0,
        "no_fallback_returns": no_cheat["fallback_return_rows_total"] == 0
        and no_cheat["line_action_fallback_return_rate_max"] == 0.0,
    }
    blockers = [name for name, passed in gates.items() if not passed]
    return {
        "ready": not blockers,
        "semantics": (
            "Bounded private student evidence for future governed teacher "
            "distillation only; does not unlock model growth, promotion, "
            "public calibration, or runtime external serving."
        ),
        "gates": gates,
        "blockers": blockers,
        "no_cheat": no_cheat,
        "student_ready_requirements": get_path(policy, ["promotion_boundary", "neural_student_ready_requires"], []),
    }


def build_checks(policy: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    arms = policy.get("arms") if isinstance(policy.get("arms"), list) else []
    budget = policy.get("matched_budget") if isinstance(policy.get("matched_budget"), dict) else {}
    data_contract = policy.get("data_contract") if isinstance(policy.get("data_contract"), dict) else {}
    model_growth = state["model_growth"] if isinstance(state.get("model_growth"), dict) else {}
    teacher = state["teacher_distillation"] if isinstance(state.get("teacher_distillation"), dict) else {}
    external = state["external"] if isinstance(state.get("external"), dict) else {}
    return [
        check(
            "external_inference_audit_ok",
            bool(external.get("ok")),
            "hard",
            {
                "ok": external.get("ok"),
                "total_violations": get_path(external, ["summary", "total_violations"], None),
            },
        ),
        check(
            "teacher_distillation_locked",
            not bool(teacher.get("distillation_allowed")) and int(teacher.get("external_inference_calls") or 0) == 0,
            "hard",
            {
                "trigger_state": teacher.get("trigger_state"),
                "distillation_allowed": teacher.get("distillation_allowed"),
                "missing_or_locked": teacher.get("missing_or_locked", []),
            },
        ),
        check(
            "public_training_sources_forbidden",
            "public benchmark tests or solutions" in set(data_contract.get("forbidden_sources", []))
            and str(data_contract.get("public_calibration", "")).startswith("locked"),
            "hard",
            data_contract,
        ),
        check(
            "no_silent_execute",
            not bool(policy.get("default_execute_allowed")),
            "hard",
            {"default_execute_allowed": policy.get("default_execute_allowed")},
        ),
        check(
            "two_matched_arms_declared",
            len(arms) == 2
            and len({row.get("matched_budget_id") for row in arms if isinstance(row, dict)}) == 1
            and bool(budget.get("budget_id")),
            "readiness",
            {"arms": [row.get("id") for row in arms if isinstance(row, dict)], "matched_budget": budget.get("budget_id")},
        ),
        check(
            "model_growth_evidence_cleared",
            not model_growth.get("hard_blockers") and not model_growth.get("missing_evidence"),
            "readiness",
            {
                "model_growth_allowed": model_growth.get("model_growth_allowed"),
                "hard_blockers": model_growth.get("hard_blockers", []),
                "missing_evidence": model_growth.get("missing_evidence", []),
                "next_action": model_growth.get("next_action"),
            },
        ),
        check(
            "adapter_and_genesis_evidence_present",
            int(get_path(state, ["adapter", "summary", "ready_cards"], 0) or 0) > 0
            and not genesis_failed_hard_gates(state["genesis"]),
            "readiness",
            {
                "adapter_ready_cards": get_path(state, ["adapter", "summary", "ready_cards"], None),
                "genesis_failed_hard_gates": genesis_failed_hard_gates(state["genesis"]),
            },
        ),
        check(
            "macos_mlx_or_cpu_constraints_recorded",
            macos_constraint_evidence_present(state),
            "readiness",
            macos_constraints(policy, state),
        ),
        check(
            "substrate_comparator_smoke_recorded",
            substrate_comparator_ready(state.get("substrate_comparator", {})),
            "readiness",
            substrate_comparator_summary(state.get("substrate_comparator", {})),
        ),
        check(
            "code_proposer_comparator_smoke_recorded",
            code_proposer_comparator_ready(state.get("code_proposer_comparator", {})),
            "readiness",
            code_proposer_comparator_summary(state.get("code_proposer_comparator", {})),
        ),
        check(
            "code_proposer_gap_report_recorded",
            code_proposer_gap_ready(state.get("code_proposer_gap_report", {})),
            "readiness",
            code_proposer_gap_summary(state.get("code_proposer_gap_report", {})),
        ),
        check(
            "token_decoder_comparator_smoke_recorded",
            token_decoder_comparator_ready(state.get("token_decoder_comparator", {})),
            "readiness",
            token_decoder_comparator_summary(state.get("token_decoder_comparator", {})),
        ),
        check(
            "token_decoder_multiseed_smoke_recorded",
            token_decoder_multiseed_ready(state.get("token_decoder_multiseed", {})),
            "readiness",
            token_decoder_multiseed_summary(state.get("token_decoder_multiseed", {})),
        ),
        check(
            "semantic_plan_gap_audit_recorded",
            semantic_plan_gap_audit_ready(state.get("semantic_plan_gap_audit", {})),
            "readiness",
            semantic_plan_gap_audit_summary(state.get("semantic_plan_gap_audit", {})),
        ),
        check(
            "architecture_seed_sweep_recorded",
            architecture_sweep_ready(state.get("architecture_sweep", {})),
            "readiness",
            architecture_sweep_summary(state.get("architecture_sweep", {})),
        ),
        check(
            "residual_mining_recorded",
            residual_mining_ready(state.get("residual_mining", {})),
            "readiness",
            residual_mining_summary(state.get("residual_mining", {})),
        ),
        check(
            "token_decoder_route_ablation_recorded",
            token_decoder_route_ablation_ready(state.get("token_decoder_route_ablation", {})),
            "readiness",
            token_decoder_route_ablation_summary(state.get("token_decoder_route_ablation", {})),
        ),
        check(
            "token_decoder_complementarity_recorded",
            token_decoder_complementarity_ready(state.get("token_decoder_complementarity", {})),
            "readiness",
            token_decoder_complementarity_summary(state.get("token_decoder_complementarity", {})),
        ),
        check(
            "token_decoder_residual_context_recorded",
            token_decoder_residual_context_ready(state.get("token_decoder_residual_context", {})),
            "readiness",
            token_decoder_residual_context_summary(state.get("token_decoder_residual_context", {})),
        ),
        check(
            "structural_action_ablation_recorded",
            structural_action_ablation_ready(state.get("structural_action_ablation", {})),
            "readiness",
            structural_action_ablation_summary(state.get("structural_action_ablation", {})),
        ),
    ]


def substrate_comparator_ready(report: dict[str, Any]) -> bool:
    if not isinstance(report, dict) or not report:
        return False
    hard_failures = [
        row.get("name")
        for row in report.get("gates", [])
        if isinstance(row, dict) and row.get("severity") == "hard" and not row.get("passed")
    ]
    return bool(
        report.get("trigger_state") in {"GREEN", "YELLOW"}
        and get_path(report, ["summary", "substrate_smoke_ready"], False)
        and not hard_failures
        and int(report.get("external_inference_calls") or 0) == 0
    )


def substrate_comparator_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {
            "present": False,
            "substrate_smoke_ready": False,
            "code_proposer_comparison_ready": False,
            "path": "reports/neural_seed_substrate_comparator.json",
        }
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    comparisons = report.get("comparisons") if isinstance(report.get("comparisons"), dict) else {}
    return {
        "present": True,
        "trigger_state": report.get("trigger_state"),
        "comparison_level": summary.get("comparison_level"),
        "substrate_smoke_ready": bool(summary.get("substrate_smoke_ready")),
        "code_proposer_comparison_ready": bool(summary.get("code_proposer_comparison_ready")),
        "discriminative_smoke": bool(summary.get("discriminative_smoke")),
        "trusted_parameter_match": bool(summary.get("trusted_parameter_match")),
        "symliquid_parameter_count": summary.get("symliquid_parameter_count"),
        "transformer_parameter_count": summary.get("transformer_parameter_count"),
        "best_sts_on_arm_by_verifier_pass_rate": summary.get("best_sts_on_arm_by_verifier_pass_rate"),
        "symliquid_minus_transformer_sts_on_verifier_pass_rate": comparisons.get(
            "symliquid_minus_transformer_sts_on_verifier_pass_rate"
        ),
        "external_inference_calls": report.get("external_inference_calls"),
        "path": "reports/neural_seed_substrate_comparator.json",
    }


def code_proposer_comparator_ready(report: dict[str, Any]) -> bool:
    if not isinstance(report, dict) or not report:
        return False
    hard_failures = [
        row.get("name")
        for row in report.get("gates", [])
        if isinstance(row, dict) and row.get("severity") == "hard" and not row.get("passed")
    ]
    return bool(
        report.get("trigger_state") in {"GREEN", "YELLOW"}
        and get_path(report, ["summary", "code_proposer_smoke_ready"], False)
        and not hard_failures
        and int(report.get("external_inference_calls") or 0) == 0
    )


def code_proposer_comparator_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {
            "present": False,
            "code_proposer_smoke_ready": False,
            "path": "reports/neural_seed_code_proposer_comparator.json",
        }
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    comparisons = report.get("comparisons") if isinstance(report.get("comparisons"), dict) else {}
    by_arm = comparisons.get("by_arm") if isinstance(comparisons.get("by_arm"), dict) else {}
    return {
        "present": True,
        "trigger_state": report.get("trigger_state"),
        "comparison_level": summary.get("comparison_level"),
        "code_proposer_smoke_ready": bool(summary.get("code_proposer_smoke_ready")),
        "both_arms_emit_candidate_code_rows": bool(summary.get("both_arms_emit_candidate_code_rows")),
        "same_private_verifier_for_both_arms": bool(summary.get("same_private_verifier_for_both_arms")),
        "trusted_parameter_match": bool(summary.get("trusted_parameter_match")),
        "candidate_rows": summary.get("candidate_rows"),
        "body_template_count": summary.get("body_template_count"),
        "symliquid_parameter_count": summary.get("symliquid_parameter_count"),
        "transformer_parameter_count": summary.get("transformer_parameter_count"),
        "best_sts_on_arm_by_verifier_pass_rate": summary.get("best_sts_on_arm_by_verifier_pass_rate"),
        "symliquid_minus_transformer_sts_on_verifier_pass_rate": comparisons.get(
            "symliquid_minus_transformer_sts_on_verifier_pass_rate"
        ),
        "by_arm": by_arm,
        "external_inference_calls": report.get("external_inference_calls"),
        "path": "reports/neural_seed_code_proposer_comparator.json",
    }


def code_proposer_gap_ready(report: dict[str, Any]) -> bool:
    if not isinstance(report, dict) or not report:
        return False
    hard_failures = [
        row.get("name")
        for row in report.get("gates", [])
        if isinstance(row, dict) and row.get("severity") == "hard" and not row.get("passed")
    ]
    return bool(report.get("trigger_state") in {"GREEN", "YELLOW"} and not hard_failures)


def code_proposer_gap_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {"present": False, "path": "reports/neural_seed_code_proposer_gap_report.json"}
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return {
        "present": True,
        "trigger_state": report.get("trigger_state"),
        "gap_counts": summary.get("gap_counts"),
        "sts_repairs": summary.get("sts_repairs"),
        "sts_regressions": summary.get("sts_regressions"),
        "failure_cause_counts": summary.get("failure_cause_counts"),
        "external_inference_calls": report.get("external_inference_calls"),
        "path": "reports/neural_seed_code_proposer_gap_report.json",
    }


def token_decoder_comparator_ready(report: dict[str, Any]) -> bool:
    if not isinstance(report, dict) or not report:
        return False
    hard_failures = [
        row.get("name")
        for row in report.get("gates", [])
        if isinstance(row, dict) and row.get("severity") == "hard" and not row.get("passed")
    ]
    return bool(
        report.get("trigger_state") in {"GREEN", "YELLOW"}
        and get_path(report, ["summary", "token_decoder_smoke_ready"], False)
        and not hard_failures
        and int(report.get("external_inference_calls") or 0) == 0
    )


def token_decoder_comparator_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {
            "present": False,
            "token_decoder_smoke_ready": False,
            "path": "reports/neural_seed_token_decoder_comparator.json",
        }
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    comparisons = report.get("comparisons") if isinstance(report.get("comparisons"), dict) else {}
    by_arm = comparisons.get("by_arm") if isinstance(comparisons.get("by_arm"), dict) else {}
    sym = by_arm.get("symliquid_style") if isinstance(by_arm.get("symliquid_style"), dict) else {}
    tx = by_arm.get("transformer_control") if isinstance(by_arm.get("transformer_control"), dict) else {}
    return {
        "present": True,
        "trigger_state": report.get("trigger_state"),
        "comparison_level": summary.get("comparison_level"),
        "token_decoder_smoke_ready": bool(summary.get("token_decoder_smoke_ready")),
        "both_arms_emit_token_decoded_candidate_code_rows": bool(summary.get("both_arms_emit_token_decoded_candidate_code_rows")),
        "same_private_verifier_for_both_arms": bool(summary.get("same_private_verifier_for_both_arms")),
        "trusted_parameter_match": bool(summary.get("trusted_parameter_match")),
        "candidate_rows": summary.get("candidate_rows"),
        "target_mode": summary.get("target_mode"),
        "target_vocab_size": summary.get("target_vocab_size"),
        "symliquid_parameter_count": summary.get("symliquid_parameter_count"),
        "transformer_parameter_count": summary.get("transformer_parameter_count"),
        "syntax_pass_rate_sts_on": {
            "symliquid_style": sym.get("syntax_pass_rate_sts_on"),
            "transformer_control": tx.get("syntax_pass_rate_sts_on"),
        },
        "raw_syntax_pass_rate_sts_on": {
            "symliquid_style": sym.get("raw_syntax_pass_rate_sts_on"),
            "transformer_control": tx.get("raw_syntax_pass_rate_sts_on"),
        },
        "grammar_repair_changed_rate_sts_on": {
            "symliquid_style": sym.get("grammar_repair_changed_rate_sts_on"),
            "transformer_control": tx.get("grammar_repair_changed_rate_sts_on"),
        },
        "grammar_repair_fallback_rate_sts_on": {
            "symliquid_style": sym.get("grammar_repair_fallback_rate_sts_on"),
            "transformer_control": tx.get("grammar_repair_fallback_rate_sts_on"),
        },
        "statement_skeleton_render_rate_sts_on": {
            "symliquid_style": sym.get("statement_skeleton_render_rate_sts_on"),
            "transformer_control": tx.get("statement_skeleton_render_rate_sts_on"),
        },
        "semantic_slot_render_rate_sts_on": {
            "symliquid_style": sym.get("semantic_slot_render_rate_sts_on"),
            "transformer_control": tx.get("semantic_slot_render_rate_sts_on"),
        },
        "semantic_plan_supported_rate_sts_on": {
            "symliquid_style": sym.get("semantic_plan_supported_rate_sts_on"),
            "transformer_control": tx.get("semantic_plan_supported_rate_sts_on"),
        },
        "predicted_return_shape_rate_sts_on": {
            "symliquid_style": sym.get("predicted_return_shape_rate_sts_on"),
            "transformer_control": tx.get("predicted_return_shape_rate_sts_on"),
        },
        "best_sts_on_arm_by_verifier_pass_rate": summary.get("best_sts_on_arm_by_verifier_pass_rate"),
        "symliquid_minus_transformer_sts_on_verifier_pass_rate": comparisons.get(
            "symliquid_minus_transformer_sts_on_verifier_pass_rate"
        ),
        "symliquid_gap_vs_body_template": summary.get("symliquid_gap_vs_body_template"),
        "transformer_gap_vs_body_template": summary.get("transformer_gap_vs_body_template"),
        "by_arm": by_arm,
        "external_inference_calls": report.get("external_inference_calls"),
        "path": "reports/neural_seed_token_decoder_comparator.json",
    }


def token_decoder_multiseed_ready(report: dict[str, Any]) -> bool:
    if not isinstance(report, dict) or not report:
        return False
    hard_failures = [
        row.get("name")
        for row in report.get("gates", [])
        if isinstance(row, dict) and row.get("severity") == "hard" and not row.get("passed")
    ]
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return bool(
        report.get("trigger_state") in {"GREEN", "YELLOW"}
        and int(summary.get("completed_seed_count") or 0) >= 5
        and not hard_failures
        and int(report.get("external_inference_calls") or 0) == 0
    )


def token_decoder_multiseed_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {
            "present": False,
            "multiseed_smoke_ready": False,
            "path": "reports/neural_seed_token_decoder_multiseed_smoke.json",
        }
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return {
        "present": True,
        "trigger_state": report.get("trigger_state"),
        "multiseed_smoke_ready": token_decoder_multiseed_ready(report),
        "completed_seed_count": summary.get("completed_seed_count"),
        "requested_seed_count": summary.get("requested_seed_count"),
        "symliquid_sts_on_mean": summary.get("symliquid_sts_on_mean"),
        "symliquid_sts_on_stdev": summary.get("symliquid_sts_on_stdev"),
        "transformer_sts_on_mean": summary.get("transformer_sts_on_mean"),
        "transformer_sts_on_stdev": summary.get("transformer_sts_on_stdev"),
        "symliquid_minus_transformer_sts_on_mean": summary.get("symliquid_minus_transformer_sts_on_mean"),
        "symliquid_minus_transformer_sts_on_stdev": summary.get("symliquid_minus_transformer_sts_on_stdev"),
        "symliquid_expected_plan_match_mean": summary.get("symliquid_expected_plan_match_mean"),
        "transformer_expected_plan_match_mean": summary.get("transformer_expected_plan_match_mean"),
        "symliquid_minus_transformer_expected_plan_match_mean": summary.get("symliquid_minus_transformer_expected_plan_match_mean"),
        "winner_counts": summary.get("winner_counts"),
        "symliquid_gap_closed": bool(summary.get("symliquid_gap_closed")),
        "bottleneck": summary.get("bottleneck"),
        "external_inference_calls": report.get("external_inference_calls"),
        "path": "reports/neural_seed_token_decoder_multiseed_smoke.json",
    }


def semantic_plan_gap_audit_ready(report: dict[str, Any]) -> bool:
    if not isinstance(report, dict) or not report:
        return False
    hard_failures = [
        row.get("name")
        for row in report.get("gates", [])
        if isinstance(row, dict) and row.get("severity") == "hard" and not row.get("passed")
    ]
    return bool(
        report.get("trigger_state") in {"GREEN", "YELLOW"}
        and not hard_failures
        and int(report.get("external_inference_calls") or 0) == 0
    )


def semantic_plan_gap_audit_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {
            "present": False,
            "semantic_plan_gap_audit_ready": False,
            "path": "reports/neural_seed_semantic_plan_gap_audit.json",
        }
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return {
        "present": True,
        "trigger_state": report.get("trigger_state"),
        "semantic_plan_gap_audit_ready": semantic_plan_gap_audit_ready(report),
        "seed": report.get("seed"),
        "eval_rows": summary.get("eval_rows"),
        "candidate_rows": summary.get("candidate_rows"),
        "gap_counts": summary.get("gap_counts"),
        "symliquid_private_eval_pass_rate": summary.get("symliquid_private_eval_pass_rate"),
        "transformer_private_eval_pass_rate": summary.get("transformer_private_eval_pass_rate"),
        "symliquid_private_eval_plan_match_rate": summary.get("symliquid_private_eval_plan_match_rate"),
        "transformer_private_eval_plan_match_rate": summary.get("transformer_private_eval_plan_match_rate"),
        "bottleneck": get_path(summary, ["bottleneck", "label"], None),
        "symliquid_gap_closed": get_path(summary, ["bottleneck", "symliquid_gap_closed"], None),
        "external_inference_calls": report.get("external_inference_calls"),
        "path": "reports/neural_seed_semantic_plan_gap_audit.json",
    }


def architecture_sweep_ready(report: dict[str, Any]) -> bool:
    if not isinstance(report, dict) or not report:
        return False
    hard_failures = [
        row.get("name")
        for row in report.get("gates", [])
        if isinstance(row, dict) and row.get("severity") == "hard" and not row.get("passed")
    ]
    return bool(
        report.get("trigger_state") in {"GREEN", "YELLOW"}
        and int(get_path(report, ["summary", "seed_count"], 0) or 0) >= int(get_path(report, ["summary", "minimum_claim_seed_count"], 5) or 5)
        and not hard_failures
        and int(report.get("external_inference_calls") or 0) == 0
    )


def architecture_sweep_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {
            "present": False,
            "seed_sweep_ready": False,
            "path": "reports/neural_seed_architecture_sweep.json",
        }
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    aggregate = report.get("aggregate") if isinstance(report.get("aggregate"), dict) else {}
    return {
        "present": True,
        "trigger_state": report.get("trigger_state"),
        "seed_sweep_ready": architecture_sweep_ready(report),
        "seed_count": summary.get("seed_count"),
        "run_count": summary.get("run_count"),
        "single_seed_claims_disallowed": bool(summary.get("single_seed_claims_disallowed")),
        "aggregate": aggregate,
        "external_inference_calls": report.get("external_inference_calls"),
        "path": "reports/neural_seed_architecture_sweep.json",
    }


def residual_mining_ready(report: dict[str, Any]) -> bool:
    if not isinstance(report, dict) or not report:
        return False
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return bool(
        report.get("trigger_state") in {"GREEN", "YELLOW"}
        and int(summary.get("task_rows") or 0) > 0
        and int(report.get("external_inference_calls") or 0) == 0
    )


def residual_mining_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {
            "present": False,
            "residual_mining_ready": False,
            "path": "reports/neural_seed_residual_mining.json",
        }
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return {
        "present": True,
        "trigger_state": report.get("trigger_state"),
        "residual_mining_ready": residual_mining_ready(report),
        "symliquid_only_win_count": summary.get("symliquid_only_win_count"),
        "transformer_only_win_count": summary.get("transformer_only_win_count"),
        "both_fail_count": summary.get("both_fail_count"),
        "next_private_pressure": report.get("next_private_pressure", [])[:12],
        "external_inference_calls": report.get("external_inference_calls"),
        "path": "reports/neural_seed_residual_mining.json",
    }


def token_decoder_route_ablation_ready(report: dict[str, Any]) -> bool:
    if not isinstance(report, dict) or not report:
        return False
    hard_failures = [
        row.get("name")
        for row in report.get("gates", [])
        if isinstance(row, dict) and row.get("severity") == "hard" and not row.get("passed")
    ]
    attribution = report.get("attribution") if isinstance(report.get("attribution"), dict) else {}
    variant_rows = report.get("variant_rows") if isinstance(report.get("variant_rows"), list) else []
    return bool(
        report.get("trigger_state") in {"GREEN", "YELLOW"}
        and not hard_failures
        and int(report.get("external_inference_calls") or 0) == 0
        and len(variant_rows) >= 6
        and float(attribution.get("symliquid_beam_off_recovery_delta_vs_pre_beam_reference") or 0.0) > 0.0
        and float(attribution.get("transformer_beam_off_recovery_delta_vs_pre_beam_reference") or 0.0) > 0.0
        and str(attribution.get("likely_primary_source") or "")
    )


def token_decoder_route_ablation_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {
            "present": False,
            "route_ablation_ready": False,
            "path": "reports/neural_seed_token_decoder_route_independence_ablation.json",
        }
    attribution = report.get("attribution") if isinstance(report.get("attribution"), dict) else {}
    variant_rows = report.get("variant_rows") if isinstance(report.get("variant_rows"), list) else []
    best = attribution.get("best_non_collapsed_prototype_reduced_variant")
    return {
        "present": True,
        "trigger_state": report.get("trigger_state"),
        "route_ablation_ready": token_decoder_route_ablation_ready(report),
        "run_count": len(variant_rows),
        "seed_count": len(report.get("seeds") if isinstance(report.get("seeds"), list) else []),
        "symliquid_learned_internal_routing_delta_vs_no_internal_beam_off": attribution.get(
            "symliquid_learned_internal_routing_delta_vs_no_internal_beam_off"
        ),
        "transformer_learned_internal_routing_delta_vs_no_internal_beam_off": attribution.get(
            "transformer_learned_internal_routing_delta_vs_no_internal_beam_off"
        ),
        "symliquid_route_dropout_delta_vs_full_beam_off": attribution.get(
            "symliquid_route_dropout_delta_vs_full_beam_off"
        ),
        "transformer_route_dropout_delta_vs_full_beam_off": attribution.get(
            "transformer_route_dropout_delta_vs_full_beam_off"
        ),
        "symliquid_no_visible_text_memory_mean": attribution.get("symliquid_no_visible_text_memory_mean"),
        "transformer_no_visible_text_memory_mean": attribution.get("transformer_no_visible_text_memory_mean"),
        "best_non_collapsed_prototype_reduced_variant": best if isinstance(best, dict) else None,
        "likely_primary_source": attribution.get("likely_primary_source"),
        "external_inference_calls": report.get("external_inference_calls"),
        "path": "reports/neural_seed_token_decoder_route_independence_ablation.json",
    }


def token_decoder_complementarity_ready(report: dict[str, Any]) -> bool:
    if not isinstance(report, dict) or not report:
        return False
    hard_failures = [
        row.get("name")
        for row in report.get("gates", [])
        if isinstance(row, dict) and row.get("severity") == "hard" and not row.get("passed")
    ]
    recommendation = get_path(report, ["overall", "recommendation"], {})
    return bool(
        report.get("trigger_state") in {"GREEN", "YELLOW"}
        and not hard_failures
        and int(report.get("external_inference_calls") or 0) == 0
        and str(recommendation.get("decision") or "") in {"keep_symliquid_as_discovery_lane", "prefer_transformer_survival_lane"}
        and recommendation.get("full_symliquid_sts_on_mean") is not None
        and recommendation.get("full_transformer_sts_on_mean") is not None
    )


def token_decoder_complementarity_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {
            "present": False,
            "complementarity_ready": False,
            "path": "reports/neural_seed_token_decoder_complementarity_audit.json",
        }
    overall = report.get("overall") if isinstance(report.get("overall"), dict) else {}
    recommendation = overall.get("recommendation") if isinstance(overall.get("recommendation"), dict) else {}
    return {
        "present": True,
        "trigger_state": report.get("trigger_state"),
        "complementarity_ready": token_decoder_complementarity_ready(report),
        "decision": recommendation.get("decision"),
        "rationale": recommendation.get("rationale"),
        "full_symliquid_sts_on_mean": recommendation.get("full_symliquid_sts_on_mean"),
        "full_transformer_sts_on_mean": recommendation.get("full_transformer_sts_on_mean"),
        "full_union_gain_vs_best_single": recommendation.get("full_union_gain_vs_best_single"),
        "stable_full_symliquid_only_task_count": recommendation.get("stable_full_symliquid_only_task_count"),
        "stable_full_transformer_only_task_count": recommendation.get("stable_full_transformer_only_task_count"),
        "symliquid_route_dropout_half_mean": recommendation.get("symliquid_route_dropout_half_mean"),
        "max_union_gain_vs_best_single": overall.get("max_union_gain_vs_best_single"),
        "external_inference_calls": report.get("external_inference_calls"),
        "path": "reports/neural_seed_token_decoder_complementarity_audit.json",
    }


def token_decoder_residual_context_ready(report: dict[str, Any]) -> bool:
    if not isinstance(report, dict) or not report:
        return False
    hard_failures = [
        row.get("name")
        for row in report.get("gates", [])
        if isinstance(row, dict) and row.get("severity") == "hard" and not row.get("passed")
    ]
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    bucket_counts = summary.get("bucket_counts") if isinstance(summary.get("bucket_counts"), dict) else {}
    recommendation = report.get("recommendation") if isinstance(report.get("recommendation"), dict) else {}
    return bool(
        report.get("trigger_state") in {"GREEN", "YELLOW"}
        and not hard_failures
        and int(report.get("external_inference_calls") or 0) == 0
        and sum(int(value or 0) for value in bucket_counts.values()) > 0
        and str(recommendation.get("next_action") or "")
    )


def token_decoder_residual_context_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {
            "present": False,
            "residual_context_ready": False,
            "path": "reports/neural_seed_token_decoder_residual_context_miner.json",
        }
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    recommendation = report.get("recommendation") if isinstance(report.get("recommendation"), dict) else {}
    return {
        "present": True,
        "trigger_state": report.get("trigger_state"),
        "residual_context_ready": token_decoder_residual_context_ready(report),
        "bucket_counts": summary.get("bucket_counts"),
        "top_dropout_regression_families": summary.get("top_dropout_regression_families"),
        "next_action": recommendation.get("next_action"),
        "rationale": recommendation.get("rationale"),
        "current_full_symliquid_mean": recommendation.get("current_full_symliquid_mean"),
        "current_no_visible_symliquid_mean": recommendation.get("current_no_visible_symliquid_mean"),
        "current_route_dropout_symliquid_mean": recommendation.get("current_route_dropout_symliquid_mean"),
        "dropout_regression_symliquid_count": recommendation.get("dropout_regression_symliquid_count"),
        "external_inference_calls": report.get("external_inference_calls"),
        "path": "reports/neural_seed_token_decoder_residual_context_miner.json",
    }


def structural_action_ablation_ready(report: dict[str, Any]) -> bool:
    if not isinstance(report, dict) or not report:
        return False
    hard_failures = [
        row.get("name")
        for row in report.get("gates", [])
        if isinstance(row, dict) and row.get("severity") == "hard" and not row.get("passed")
    ]
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    compiler = get_path(report, ["axes", "line_action_compilation"], {})
    ast_axis = get_path(report, ["axes", "finer_ast_synthesis"], {})
    return bool(
        report.get("trigger_state") in {"GREEN", "YELLOW"}
        and not hard_failures
        and int(summary.get("external_inference_calls") or 0) == 0
        and not bool(summary.get("teacher_used"))
        and int(summary.get("public_training_rows") or 0) == 0
        and int(summary.get("structural_rows") or 0) > 0
        and float(compiler.get("fallback_return_rate_max") if compiler.get("fallback_return_rate_max") is not None else 1.0)
        == 0.0
        and str(ast_axis.get("status") or "") == "separated_but_not_yet_implemented"
    )


def structural_action_ablation_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {
            "present": False,
            "structural_action_ablation_ready": False,
            "path": "reports/neural_seed_structural_action_ablation_report.json",
        }
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    compiler = get_path(report, ["axes", "line_action_compilation"], {})
    ast_axis = get_path(report, ["axes", "finer_ast_synthesis"], {})
    standalone = summary.get("standalone_96eval_summary") if isinstance(summary.get("standalone_96eval_summary"), dict) else {}
    return {
        "present": True,
        "trigger_state": report.get("trigger_state"),
        "structural_action_ablation_ready": structural_action_ablation_ready(report),
        "integrated_trigger_state": summary.get("integrated_trigger_state"),
        "structural_rows": summary.get("structural_rows"),
        "token_rows": summary.get("token_rows"),
        "symliquid_sts_on_pass_rate": summary.get("symliquid_sts_on_pass_rate"),
        "transformer_sts_on_pass_rate": summary.get("transformer_sts_on_pass_rate"),
        "standalone_96eval_multiseed_available": summary.get("standalone_96eval_multiseed_available"),
        "standalone_symliquid_delta_mean": standalone.get("symliquid_delta_mean"),
        "standalone_transformer_delta_mean": standalone.get("transformer_delta_mean"),
        "standalone_fallback_return_rows_total": standalone.get("fallback_return_rows_total"),
        "line_action_compiler": compiler.get("compiler"),
        "line_action_fallback_return_rate_max": compiler.get("fallback_return_rate_max"),
        "finer_ast_synthesis_status": ast_axis.get("status"),
        "external_inference_calls": report.get("external_inference_calls"),
        "path": "reports/neural_seed_structural_action_ablation_report.json",
    }


def macos_constraint_evidence_present(state: dict[str, Any]) -> bool:
    if platform.system() != "Darwin":
        return True
    parity_summary = get_path(state, ["mlx_parity", "summary"], {})
    work_summary = get_path(state, ["mlx_work", "summary"], {})
    return bool(
        int(parity_summary.get("runnable_evidence_missing") or 0) == 0
        and int(work_summary.get("cli_smoke_ok_count") or 0) >= 1
        and int(work_summary.get("worker_smoke_ok_count") or 0) >= 1
    )


def macos_constraints(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    configured = policy.get("macos_constraints") if isinstance(policy.get("macos_constraints"), dict) else {}
    parity_summary = get_path(state, ["mlx_parity", "summary"], {})
    work_summary = get_path(state, ["mlx_work", "summary"], {})
    return {
        "host_platform": platform.system(),
        "host_machine": platform.machine(),
        "primary_apple_silicon_backend": "mlx_apple",
        "intel_mac_backend": "cpu_storage_operator_only_until_non_mlx_backend_profiled",
        "configured": configured,
        "resource_execution_owner": get_path(state, ["resource", "decision", "execution_owner"], None),
        "mlx_parity": {
            "implemented_mlx_cli_bridges": parity_summary.get("implemented_mlx_cli_bridges"),
            "implemented_hive_mlx_tasks": parity_summary.get("implemented_hive_mlx_tasks"),
            "runnable_evidence_missing": parity_summary.get("runnable_evidence_missing"),
            "kernel_parity_pending_count": parity_summary.get("kernel_parity_pending_count"),
        },
        "mlx_work_proof": {
            "cli_smoke_ok_count": work_summary.get("cli_smoke_ok_count"),
            "worker_smoke_ok_count": work_summary.get("worker_smoke_ok_count"),
            "teacher_used": work_summary.get("teacher_used"),
            "external_inference_calls": work_summary.get("external_inference_calls"),
        },
    }


def genesis_failed_hard_gates(report: dict[str, Any]) -> list[str]:
    rows = report.get("release_gates") if isinstance(report.get("release_gates"), list) else []
    return [
        str(row.get("gate"))
        for row in rows
        if isinstance(row, dict) and row.get("severity") == "hard" and not row.get("passed")
    ]


def next_action(
    spec_ready: bool,
    execute_allowed: bool,
    hard: list[dict[str, Any]],
    readiness: list[dict[str, Any]],
    state: dict[str, Any],
) -> str:
    if hard:
        return "Do not proceed; clear hard neural-seed boundary blockers: " + ", ".join(row["name"] for row in hard) + "."
    if readiness:
        return "Finish the neural-seed spec prerequisites: " + ", ".join(row["name"] for row in readiness) + "."
    if not execute_allowed:
        return (
            "Spec is ready, but execution remains locked. "
            f"Model-growth gate says: {state['model_growth'].get('next_action')}"
        )
    return "Execution may run only as a bounded private neural-seed smoke behind verifier/fanout/STS gates."


def render_markdown(report: dict[str, Any]) -> str:
    checks = "\n".join(
        f"- `{row['name']}`: passed=`{row['passed']}` severity=`{row['severity']}`"
        for row in report["checks"]
    )
    arms = "\n".join(
        f"- `{row.get('id')}`: {row.get('student_type')} ({row.get('substrate')})"
        for row in report["experiment"].get("arms", [])
    )
    comparator = report.get("substrate_comparator") if isinstance(report.get("substrate_comparator"), dict) else {}
    comparator_lines = "\n".join(
        [
            f"- present: `{comparator.get('present')}`",
            f"- comparison_level: `{comparator.get('comparison_level')}`",
            f"- substrate_smoke_ready: `{comparator.get('substrate_smoke_ready')}`",
            f"- code_proposer_comparison_ready: `{comparator.get('code_proposer_comparison_ready')}`",
            f"- trusted_parameter_match: `{comparator.get('trusted_parameter_match')}`",
            f"- best_sts_on_arm_by_verifier_pass_rate: `{comparator.get('best_sts_on_arm_by_verifier_pass_rate')}`",
            f"- symliquid_minus_transformer_sts_on_verifier_pass_rate: `{comparator.get('symliquid_minus_transformer_sts_on_verifier_pass_rate')}`",
        ]
    )
    code_comparator = report.get("code_proposer_comparator") if isinstance(report.get("code_proposer_comparator"), dict) else {}
    code_comparator_lines = "\n".join(
        [
            f"- present: `{code_comparator.get('present')}`",
            f"- comparison_level: `{code_comparator.get('comparison_level')}`",
            f"- code_proposer_smoke_ready: `{code_comparator.get('code_proposer_smoke_ready')}`",
            f"- both_arms_emit_candidate_code_rows: `{code_comparator.get('both_arms_emit_candidate_code_rows')}`",
            f"- same_private_verifier_for_both_arms: `{code_comparator.get('same_private_verifier_for_both_arms')}`",
            f"- trusted_parameter_match: `{code_comparator.get('trusted_parameter_match')}`",
            f"- candidate_rows: `{code_comparator.get('candidate_rows')}`",
            f"- best_sts_on_arm_by_verifier_pass_rate: `{code_comparator.get('best_sts_on_arm_by_verifier_pass_rate')}`",
            f"- symliquid_minus_transformer_sts_on_verifier_pass_rate: `{code_comparator.get('symliquid_minus_transformer_sts_on_verifier_pass_rate')}`",
        ]
    )
    gap = report.get("code_proposer_gap_report") if isinstance(report.get("code_proposer_gap_report"), dict) else {}
    gap_lines = "\n".join(
        [
            f"- present: `{gap.get('present')}`",
            f"- gap_counts: `{gap.get('gap_counts')}`",
            f"- sts_repairs: `{gap.get('sts_repairs')}`",
            f"- sts_regressions: `{gap.get('sts_regressions')}`",
            f"- failure_cause_counts: `{gap.get('failure_cause_counts')}`",
        ]
    )
    token_comparator = report.get("token_decoder_comparator") if isinstance(report.get("token_decoder_comparator"), dict) else {}
    token_comparator_lines = "\n".join(
        [
            f"- present: `{token_comparator.get('present')}`",
            f"- comparison_level: `{token_comparator.get('comparison_level')}`",
            f"- token_decoder_smoke_ready: `{token_comparator.get('token_decoder_smoke_ready')}`",
            f"- both_arms_emit_token_decoded_candidate_code_rows: `{token_comparator.get('both_arms_emit_token_decoded_candidate_code_rows')}`",
            f"- trusted_parameter_match: `{token_comparator.get('trusted_parameter_match')}`",
            f"- candidate_rows: `{token_comparator.get('candidate_rows')}`",
            f"- target_mode: `{token_comparator.get('target_mode')}`",
            f"- syntax_pass_rate_sts_on: `{token_comparator.get('syntax_pass_rate_sts_on')}`",
            f"- raw_syntax_pass_rate_sts_on: `{token_comparator.get('raw_syntax_pass_rate_sts_on')}`",
            f"- grammar_repair_fallback_rate_sts_on: `{token_comparator.get('grammar_repair_fallback_rate_sts_on')}`",
            f"- statement_skeleton_render_rate_sts_on: `{token_comparator.get('statement_skeleton_render_rate_sts_on')}`",
            f"- semantic_slot_render_rate_sts_on: `{token_comparator.get('semantic_slot_render_rate_sts_on')}`",
            f"- semantic_plan_supported_rate_sts_on: `{token_comparator.get('semantic_plan_supported_rate_sts_on')}`",
            f"- predicted_return_shape_rate_sts_on: `{token_comparator.get('predicted_return_shape_rate_sts_on')}`",
            f"- best_sts_on_arm_by_verifier_pass_rate: `{token_comparator.get('best_sts_on_arm_by_verifier_pass_rate')}`",
            f"- symliquid_minus_transformer_sts_on_verifier_pass_rate: `{token_comparator.get('symliquid_minus_transformer_sts_on_verifier_pass_rate')}`",
            f"- symliquid_gap_vs_body_template: `{token_comparator.get('symliquid_gap_vs_body_template')}`",
            f"- transformer_gap_vs_body_template: `{token_comparator.get('transformer_gap_vs_body_template')}`",
        ]
    )
    token_multiseed = report.get("token_decoder_multiseed") if isinstance(report.get("token_decoder_multiseed"), dict) else {}
    token_multiseed_lines = "\n".join(
        [
            f"- present: `{token_multiseed.get('present')}`",
            f"- multiseed_smoke_ready: `{token_multiseed.get('multiseed_smoke_ready')}`",
            f"- completed_seed_count: `{token_multiseed.get('completed_seed_count')}`",
            f"- requested_seed_count: `{token_multiseed.get('requested_seed_count')}`",
            f"- symliquid_sts_on_mean: `{token_multiseed.get('symliquid_sts_on_mean')}`",
            f"- transformer_sts_on_mean: `{token_multiseed.get('transformer_sts_on_mean')}`",
            f"- symliquid_minus_transformer_sts_on_mean: `{token_multiseed.get('symliquid_minus_transformer_sts_on_mean')}`",
            f"- symliquid_expected_plan_match_mean: `{token_multiseed.get('symliquid_expected_plan_match_mean')}`",
            f"- transformer_expected_plan_match_mean: `{token_multiseed.get('transformer_expected_plan_match_mean')}`",
            f"- symliquid_gap_closed: `{token_multiseed.get('symliquid_gap_closed')}`",
            f"- bottleneck: `{token_multiseed.get('bottleneck')}`",
            f"- winner_counts: `{token_multiseed.get('winner_counts')}`",
        ]
    )
    semantic_audit = report.get("semantic_plan_gap_audit") if isinstance(report.get("semantic_plan_gap_audit"), dict) else {}
    semantic_audit_lines = "\n".join(
        [
            f"- present: `{semantic_audit.get('present')}`",
            f"- semantic_plan_gap_audit_ready: `{semantic_audit.get('semantic_plan_gap_audit_ready')}`",
            f"- seed: `{semantic_audit.get('seed')}`",
            f"- gap_counts: `{semantic_audit.get('gap_counts')}`",
            f"- symliquid_private_eval_plan_match_rate: `{semantic_audit.get('symliquid_private_eval_plan_match_rate')}`",
            f"- transformer_private_eval_plan_match_rate: `{semantic_audit.get('transformer_private_eval_plan_match_rate')}`",
            f"- bottleneck: `{semantic_audit.get('bottleneck')}`",
        ]
    )
    architecture_sweep = report.get("architecture_sweep") if isinstance(report.get("architecture_sweep"), dict) else {}
    architecture_sweep_lines = "\n".join(
        [
            f"- present: `{architecture_sweep.get('present')}`",
            f"- seed_sweep_ready: `{architecture_sweep.get('seed_sweep_ready')}`",
            f"- seed_count: `{architecture_sweep.get('seed_count')}`",
            f"- run_count: `{architecture_sweep.get('run_count')}`",
            f"- single_seed_claims_disallowed: `{architecture_sweep.get('single_seed_claims_disallowed')}`",
            f"- aggregate: `{architecture_sweep.get('aggregate')}`",
        ]
    )
    residual_mining = report.get("residual_mining") if isinstance(report.get("residual_mining"), dict) else {}
    residual_mining_lines = "\n".join(
        [
            f"- present: `{residual_mining.get('present')}`",
            f"- residual_mining_ready: `{residual_mining.get('residual_mining_ready')}`",
            f"- symliquid_only_win_count: `{residual_mining.get('symliquid_only_win_count')}`",
            f"- transformer_only_win_count: `{residual_mining.get('transformer_only_win_count')}`",
            f"- both_fail_count: `{residual_mining.get('both_fail_count')}`",
            f"- next_private_pressure: `{residual_mining.get('next_private_pressure')}`",
        ]
    )
    route_ablation = (
        report.get("token_decoder_route_ablation")
        if isinstance(report.get("token_decoder_route_ablation"), dict)
        else {}
    )
    route_ablation_lines = "\n".join(
        [
            f"- present: `{route_ablation.get('present')}`",
            f"- route_ablation_ready: `{route_ablation.get('route_ablation_ready')}`",
            f"- run_count: `{route_ablation.get('run_count')}`",
            f"- symliquid_learned_internal_routing_delta_vs_no_internal_beam_off: `{route_ablation.get('symliquid_learned_internal_routing_delta_vs_no_internal_beam_off')}`",
            f"- transformer_learned_internal_routing_delta_vs_no_internal_beam_off: `{route_ablation.get('transformer_learned_internal_routing_delta_vs_no_internal_beam_off')}`",
            f"- symliquid_route_dropout_delta_vs_full_beam_off: `{route_ablation.get('symliquid_route_dropout_delta_vs_full_beam_off')}`",
            f"- symliquid_no_visible_text_memory_mean: `{route_ablation.get('symliquid_no_visible_text_memory_mean')}`",
            f"- likely_primary_source: `{route_ablation.get('likely_primary_source')}`",
        ]
    )
    complementarity = (
        report.get("token_decoder_complementarity")
        if isinstance(report.get("token_decoder_complementarity"), dict)
        else {}
    )
    complementarity_lines = "\n".join(
        [
            f"- present: `{complementarity.get('present')}`",
            f"- complementarity_ready: `{complementarity.get('complementarity_ready')}`",
            f"- decision: `{complementarity.get('decision')}`",
            f"- full_symliquid_sts_on_mean: `{complementarity.get('full_symliquid_sts_on_mean')}`",
            f"- full_transformer_sts_on_mean: `{complementarity.get('full_transformer_sts_on_mean')}`",
            f"- full_union_gain_vs_best_single: `{complementarity.get('full_union_gain_vs_best_single')}`",
            f"- max_union_gain_vs_best_single: `{complementarity.get('max_union_gain_vs_best_single')}`",
            f"- symliquid_route_dropout_half_mean: `{complementarity.get('symliquid_route_dropout_half_mean')}`",
        ]
    )
    residual_context = (
        report.get("token_decoder_residual_context")
        if isinstance(report.get("token_decoder_residual_context"), dict)
        else {}
    )
    residual_context_lines = "\n".join(
        [
            f"- present: `{residual_context.get('present')}`",
            f"- residual_context_ready: `{residual_context.get('residual_context_ready')}`",
            f"- bucket_counts: `{residual_context.get('bucket_counts')}`",
            f"- top_dropout_regression_families: `{residual_context.get('top_dropout_regression_families')}`",
            f"- next_action: `{residual_context.get('next_action')}`",
            f"- current_full_symliquid_mean: `{residual_context.get('current_full_symliquid_mean')}`",
            f"- current_route_dropout_symliquid_mean: `{residual_context.get('current_route_dropout_symliquid_mean')}`",
        ]
    )
    structural_action = (
        report.get("structural_action_ablation")
        if isinstance(report.get("structural_action_ablation"), dict)
        else {}
    )
    structural_action_lines = "\n".join(
        [
            f"- present: `{structural_action.get('present')}`",
            f"- structural_action_ablation_ready: `{structural_action.get('structural_action_ablation_ready')}`",
            f"- structural_rows: `{structural_action.get('structural_rows')}`",
            f"- token_rows: `{structural_action.get('token_rows')}`",
            f"- symliquid_sts_on_pass_rate: `{structural_action.get('symliquid_sts_on_pass_rate')}`",
            f"- transformer_sts_on_pass_rate: `{structural_action.get('transformer_sts_on_pass_rate')}`",
            f"- standalone_symliquid_delta_mean: `{structural_action.get('standalone_symliquid_delta_mean')}`",
            f"- standalone_transformer_delta_mean: `{structural_action.get('standalone_transformer_delta_mean')}`",
            f"- line_action_fallback_return_rate_max: `{structural_action.get('line_action_fallback_return_rate_max')}`",
            f"- finer_ast_synthesis_status: `{structural_action.get('finer_ast_synthesis_status')}`",
        ]
    )
    return "\n".join(
        [
            "# Neural Seed Growth Gate",
            "",
            f"- created_utc: `{report['created_utc']}`",
            f"- trigger_state: `{report['trigger_state']}`",
            f"- spec_ready: `{report['spec_ready']}`",
            f"- execute_allowed: `{report['execute_allowed']}`",
            f"- neural_student_ready: `{report['neural_student_ready']}`",
            f"- student_distillation_evidence_ready: `{report.get('student_distillation_evidence_ready')}`",
            f"- next_action: {report['next_action']}",
            "",
            "## Arms",
            "",
            arms,
            "",
            "## Checks",
            "",
            checks,
            "",
            "## Substrate Comparator",
            "",
            comparator_lines,
            "",
            "## Code-Proposer Comparator",
            "",
            code_comparator_lines,
            "",
            "## Code-Proposer Gap Report",
            "",
            gap_lines,
            "",
            "## Token Decoder Comparator",
            "",
            token_comparator_lines,
            "",
            "## Token Decoder Multi-Seed",
            "",
            token_multiseed_lines,
            "",
            "## Semantic Plan Gap Audit",
            "",
            semantic_audit_lines,
            "",
            "## Architecture Seed Sweep",
            "",
            architecture_sweep_lines,
            "",
            "## Residual Mining",
            "",
            residual_mining_lines,
            "",
            "## Token Decoder Route Ablation",
            "",
            route_ablation_lines,
            "",
            "## Token Decoder Complementarity",
            "",
            complementarity_lines,
            "",
            "## Token Decoder Residual Context",
            "",
            residual_context_lines,
            "",
            "## Structural Action Ablation",
            "",
            structural_action_lines,
            "",
            "## Boundary",
            "",
            report["score_semantics"],
            "",
        ]
    )


def check(name: str, passed: bool, severity: str, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
