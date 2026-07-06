"""Gate parameter/architecture growth behind cheaper evidence.

Project Theseus should get smarter before it gets bigger. This report makes
that rule machine-readable for autonomy, teacher self-edit, and architecture
search: data, adapters, tools, inference, and Rust/CUDA efficiency must be tried
before any parameter or substrate growth is allowed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="configs/model_growth_policy.json")
    parser.add_argument("--out", default="reports/model_growth_gate.json")
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy)
    state = load_state()
    checks = build_checks(state)
    hard_blockers = [item for item in checks if item["severity"] == "hard" and not item["passed"]]
    missing_evidence = [item for item in checks if item["severity"] == "evidence" and not item["passed"]]
    promotion_allowed = not hard_blockers and not missing_evidence and bool(get_path(state, ["architecture", "architecture_change_allowed"], False))
    private_architecture_blockers = [
        item
        for item in hard_blockers
        if item["name"] != "public_transfer_floor_cleared_for_promotion_growth"
    ]
    private_architecture_experiment_allowed = (
        not private_architecture_blockers
        and not missing_evidence
        and bool(get_path(state, ["architecture", "architecture_change_allowed"], False))
    )
    report = {
        "policy": "project_theseus_model_growth_gate_v0",
        "created_utc": now(),
        "config": args.policy,
        "principle": policy.get("principle"),
        "model_growth_allowed": promotion_allowed,
        "promotion_grade_model_growth_allowed": promotion_allowed,
        "private_architecture_experiment_allowed": private_architecture_experiment_allowed,
        "allowed_growth_types": policy.get("allowed_growth_types", []),
        "growth_boundary": {
            "model_growth_allowed_semantics": "promotion_grade_capacity_or_serving_growth_only",
            "private_architecture_experiment_semantics": (
                "bounded private evidence gathering behind verifier/gates; no promotion, "
                "serving, public calibration, or transfer claim implied"
            ),
            "public_transfer_floor": 0.70,
            "current_public_transfer_rate": broad_public_pass_rate(state),
        },
        "neural_seed_growth": policy.get("neural_seed_growth", {}),
        "checks": checks,
        "hard_blockers": [item["name"] for item in hard_blockers],
        "private_architecture_experiment_blockers": [item["name"] for item in private_architecture_blockers],
        "missing_evidence": [item["name"] for item in missing_evidence],
        "next_action": next_action(hard_blockers, missing_evidence, state),
        "external_inference_calls": 0,
    }
    write_json(ROOT / args.out, report)
    print(json.dumps(report, indent=2))
    return 0


def load_state() -> dict[str, Any]:
    reports = ROOT / "reports"
    return {
        "external": read_json(reports / "external_inference_audit.json"),
        "attd": read_json(reports / "attd_report.json"),
        "resource": read_json(reports / "resource_governor.json"),
        "performance": read_json(reports / "performance_optimizer.json"),
        "bottleneck": read_json(reports / "candidate_bottleneck_reducer.json"),
        "adapter": read_json(reports / "benchmark_adapter_factory.json"),
        "synthetic_benchmark_factory": read_json(reports / "synthetic_benchmark_factory.json"),
        "multi_stream_trace_factory": read_json(reports / "multi_stream_trace_factory.json"),
        "multi_stream_code_pressure": read_latest_json(reports, "multi_stream_code_pressure_*_seed*.json"),
        "multi_stream_monitorability_probe": read_json(reports / "multi_stream_monitorability_probe.json"),
        "multi_stream_candidate_gate": read_json(reports / "multi_stream_candidate_gate.json"),
        "loop_promoter": read_json(reports / "loop_closure_tool_promoter.json"),
        "loop_harvester": read_json(reports / "loop_closure_harvester.json"),
        "frontier": read_json(reports / "frontier_policy_status.json"),
        "architecture": read_json(reports / "architecture_experiment_governance.json"),
        "causal_architecture_delta": read_json(reports / "causal_architecture_delta_loop.json"),
        "maturity_integrity": read_json(reports / "maturity_integrity_audit.json"),
        "asi_governor": read_json(reports / "asi_wall_breaker_governor.json"),
        "transfer": read_json(reports / "transfer_eval_suite.json"),
        "token_superposition": read_json(reports / "token_superposition_training.json"),
        "code_residual_forge": read_json(reports / "code_residual_forge.json"),
        "code_repair_organism": read_latest_json(reports, "local_code_repair_organism_*_seed*.json"),
        "self_edit_lane": read_json(reports / "self_edit_experiment_lane.json"),
        "long_horizon_memory": read_json(reports / "long_horizon_memory_probe.json"),
        "genesis": read_json(reports / "genesis_kernel" / "report.json"),
    }


def build_checks(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        check("external_inference_teacher_only", bool(get_path(state, ["external", "ok"], True)), "hard", state.get("external", {}).get("summary")),
        check(
            "maturity_integrity_audit_has_no_hard_blockers",
            bool(state.get("maturity_integrity"))
            and int(get_path(state, ["maturity_integrity", "summary", "hard_blocker_count"], 1)) == 0
            and int(get_path(state, ["maturity_integrity", "summary", "manifest_public_leak_hit_count"], 1)) == 0,
            "hard",
            get_path(state, ["maturity_integrity", "summary"], {}),
        ),
        check(
            "asi_wall_governor_reports_growth_wall",
            bool(state.get("asi_governor"))
            and int(get_path(state, ["asi_governor", "summary", "wall_count"], 0) or 0) > 0
            and int(get_path(state, ["asi_governor", "summary", "external_inference_calls"], state.get("asi_governor", {}).get("external_inference_calls", 0)) or 0) == 0,
            "evidence",
            get_path(state, ["asi_governor", "summary"], {}),
        ),
        check("attd_not_red", str(get_path(state, ["attd", "trigger_state"], "GREEN")) != "RED", "hard", get_path(state, ["attd", "attd_score"], None)),
        check("resource_governor_allows_work", bool(get_path(state, ["resource", "decision", "can_run_requested_profile"], True)), "hard", get_path(state, ["resource", "decision", "throttle_reasons"], [])),
        check("performance_optimizer_green", str(get_path(state, ["performance", "trigger_state"], "GREEN")) != "RED", "hard", get_path(state, ["performance", "score"], None)),
        check("safe_bottleneck_actions_done", int(get_path(state, ["bottleneck", "remaining_safe_auto_action_count"], 0) or 0) == 0, "hard", get_path(state, ["bottleneck", "status"], None)),
        check(
            "public_transfer_floor_cleared_for_promotion_growth",
            public_transfer_floor_cleared(state),
            "hard",
            {
                "broad_public_pass_rate": broad_public_pass_rate(state),
                "floor": 0.70,
                "maturity_state": state.get("maturity_integrity", {}).get("trigger_state"),
                "public_calibration_allowed": get_path(state, ["maturity_integrity", "summary", "public_calibration_allowed"], False),
                "candidate_promotion_allowed": get_path(state, ["maturity_integrity", "summary", "candidate_promotion_allowed"], False),
                "note": "Blocks promotion-grade model/capacity growth only; bounded private architecture experiments may continue separately.",
            },
        ),
        check("adapter_pressure_available", int(get_path(state, ["adapter", "summary", "ready_cards"], 0) or 0) > 0, "evidence", get_path(state, ["adapter", "summary"], {})),
        check("synthetic_benchmark_pressure_available", synthetic_benchmark_factory_attempted(state), "evidence", synthetic_benchmark_factory_evidence(state)),
        check("multi_stream_code_pressure_attempted", multi_stream_code_pressure_attempted(state), "evidence", multi_stream_code_pressure_evidence(state)),
        check("loop_closure_attempted", int(get_path(state, ["loop_promoter", "after_tools"], 0) or 0) >= int(get_path(state, ["loop_promoter", "before_tools"], 0) or 0), "evidence", state.get("loop_promoter", {})),
        check("transfer_eval_available", bool(get_path(state, ["transfer", "summary", "task_count"], 0)), "evidence", get_path(state, ["transfer", "summary"], {})),
        check("frontier_wall_evidence_present", bool(get_path(state, ["frontier", "frontier_pressure", "active_frontier_wall"], False) or get_path(state, ["frontier", "frontier_pressure", "frontier_exhausted"], False)), "evidence", get_path(state, ["frontier", "frontier_pressure"], {})),
        check("code_residual_forge_attempted", code_residual_forge_attempted(state), "evidence", code_residual_forge_evidence(state)),
        check("local_code_repair_organism_attempted", code_repair_organism_attempted(state), "evidence", code_repair_organism_evidence(state)),
        check("self_edit_experiment_lane_attempted", self_edit_lane_attempted(state), "evidence", self_edit_lane_evidence(state)),
        check("long_horizon_memory_probe_attempted", long_horizon_memory_attempted(state), "evidence", long_horizon_memory_evidence(state)),
        check("token_superposition_training_attempted", token_superposition_attempted(state), "evidence", token_superposition_evidence(state)),
        check("genesis_artifact_substrate_ready", genesis_ready(state), "evidence", genesis_evidence(state)),
        check("architecture_ladder_reached", architecture_ladder_reached(state), "evidence", architecture_ladder_evidence(state)),
    ]


def token_superposition_attempted(state: dict[str, Any]) -> bool:
    report = state.get("token_superposition") if isinstance(state.get("token_superposition"), dict) else {}
    policy = str(report.get("policy") or "")
    backend = str(report.get("backend") or "")
    if not policy.startswith("project_theseus_token_superposition_"):
        return False
    baseline = report.get("baseline") if isinstance(report.get("baseline"), dict) else {}
    variants = report.get("variants") if isinstance(report.get("variants"), list) else []
    return bool(baseline and variants and backend in {"rust_cuda", "mlx_apple"})


def synthetic_benchmark_factory_attempted(state: dict[str, Any]) -> bool:
    report = state.get("synthetic_benchmark_factory") if isinstance(state.get("synthetic_benchmark_factory"), dict) else {}
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return bool(
        report.get("policy") == "project_theseus_synthetic_benchmark_factory_v1"
        and report.get("trigger_state") == "GREEN"
        and int(summary.get("ready_cards") or 0) > 0
        and int(summary.get("case_count") or 0) > 0
        and int(report.get("external_inference_calls") or 0) == 0
    )


def synthetic_benchmark_factory_evidence(state: dict[str, Any]) -> dict[str, Any]:
    report = state.get("synthetic_benchmark_factory") if isinstance(state.get("synthetic_benchmark_factory"), dict) else {}
    return {
        "policy": report.get("policy"),
        "trigger_state": report.get("trigger_state"),
        "cards": get_path(report, ["summary", "cards"], None),
        "ready_cards": get_path(report, ["summary", "ready_cards"], None),
        "case_count": get_path(report, ["summary", "case_count"], None),
        "cross_arm_case_count": get_path(report, ["summary", "cross_arm_case_count"], None),
        "quality_score": get_path(report, ["summary", "quality_score"], None),
        "external_inference_calls": report.get("external_inference_calls"),
    }


def architecture_ladder_reached(state: dict[str, Any]) -> bool:
    report = state.get("causal_architecture_delta") if isinstance(state.get("causal_architecture_delta"), dict) else {}
    gates = report.get("gates") if isinstance(report.get("gates"), list) else []
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return bool(
        report.get("policy") == "project_theseus_causal_architecture_delta_loop_v1"
        and report.get("trigger_state") == "GREEN"
        and report.get("status") == "completed_with_capability_delta"
        and all(bool(gate.get("passed")) for gate in gates if isinstance(gate, dict))
        and int(report.get("external_inference_calls") or 0) == 0
        and int(summary.get("public_task_count") or 0) == 0
        and not bool(summary.get("public_tests_or_solutions_used"))
    )


def architecture_ladder_evidence(state: dict[str, Any]) -> dict[str, Any]:
    report = state.get("causal_architecture_delta") if isinstance(state.get("causal_architecture_delta"), dict) else {}
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    failed_gates = [
        gate.get("gate") or gate.get("name")
        for gate in report.get("gates", [])
        if isinstance(gate, dict) and not gate.get("passed")
    ]
    return {
        "policy": report.get("policy"),
        "trigger_state": report.get("trigger_state"),
        "status": report.get("status"),
        "execute": report.get("execute"),
        "best_target_delta": summary.get("best_target_delta"),
        "private_heldout_pass_rate_delta": summary.get("private_heldout_pass_rate_delta"),
        "private_receiver_eligible_task_rate_delta": summary.get("private_receiver_eligible_task_rate_delta"),
        "private_semantic_test_passed_task_rate_delta": summary.get("private_semantic_test_passed_task_rate_delta"),
        "public_task_count": summary.get("public_task_count"),
        "public_tests_or_solutions_used": summary.get("public_tests_or_solutions_used"),
        "failed_gates": failed_gates,
        "architecture_change_allowed": get_path(state, ["architecture", "architecture_change_allowed"], False),
        "external_inference_calls": report.get("external_inference_calls"),
    }


def multi_stream_code_pressure_attempted(state: dict[str, Any]) -> bool:
    factory = state.get("multi_stream_trace_factory") if isinstance(state.get("multi_stream_trace_factory"), dict) else {}
    pressure = state.get("multi_stream_code_pressure") if isinstance(state.get("multi_stream_code_pressure"), dict) else {}
    probe = state.get("multi_stream_monitorability_probe") if isinstance(state.get("multi_stream_monitorability_probe"), dict) else {}
    gate_report = state.get("multi_stream_candidate_gate") if isinstance(state.get("multi_stream_candidate_gate"), dict) else {}
    return bool(
        factory.get("policy") == "project_theseus_multi_stream_trace_factory_v1"
        and factory.get("trigger_state") == "GREEN"
        and pressure.get("policy") == "project_theseus_multi_stream_code_pressure_v1"
        and get_path(pressure, ["verifier", "trigger_state"], "") == "GREEN"
        and probe.get("trigger_state") == "GREEN"
        and gate_report.get("trigger_state") == "GREEN"
        and float(get_path(pressure, ["summary", "pass_rate_delta"], 0.0) or 0.0) > 0.0
        and int(get_path(pressure, ["summary", "task_level_improvements_over_single_stream"], 0) or 0) > 0
        and int(get_path(pressure, ["summary", "task_level_regressions_vs_single_stream"], 0) or 0) == 0
        and int(pressure.get("external_inference_calls") or 0) == 0
    )


def multi_stream_code_pressure_evidence(state: dict[str, Any]) -> dict[str, Any]:
    factory = state.get("multi_stream_trace_factory") if isinstance(state.get("multi_stream_trace_factory"), dict) else {}
    pressure = state.get("multi_stream_code_pressure") if isinstance(state.get("multi_stream_code_pressure"), dict) else {}
    probe = state.get("multi_stream_monitorability_probe") if isinstance(state.get("multi_stream_monitorability_probe"), dict) else {}
    gate_report = state.get("multi_stream_candidate_gate") if isinstance(state.get("multi_stream_candidate_gate"), dict) else {}
    return {
        "factory_trigger_state": factory.get("trigger_state"),
        "case_count": get_path(factory, ["summary", "case_count"], None),
        "pressure_policy": pressure.get("policy"),
        "score": pressure.get("score"),
        "single_stream_pass_rate": get_path(pressure, ["summary", "single_stream_transfer_pass_rate"], None),
        "multi_stream_pass_rate": get_path(pressure, ["summary", "multi_stream_pass_rate"], None),
        "pass_rate_delta": get_path(pressure, ["summary", "pass_rate_delta"], None),
        "task_level_improvements_over_single_stream": get_path(pressure, ["summary", "task_level_improvements_over_single_stream"], None),
        "task_level_regressions_vs_single_stream": get_path(pressure, ["summary", "task_level_regressions_vs_single_stream"], None),
        "patch_stream_synthesis_used_count": get_path(pressure, ["summary", "patch_stream_synthesis_used_count"], None),
        "verifier_trigger_state": get_path(pressure, ["verifier", "trigger_state"], None),
        "monitorability_trigger_state": probe.get("trigger_state"),
        "candidate_gate_trigger_state": gate_report.get("trigger_state"),
        "external_inference_calls": pressure.get("external_inference_calls"),
    }


def code_residual_forge_attempted(state: dict[str, Any]) -> bool:
    report = state.get("code_residual_forge") if isinstance(state.get("code_residual_forge"), dict) else {}
    if report.get("policy") != "project_theseus_code_residual_forge_report_v1":
        return False
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return bool(
        int(summary.get("report_count") or 0) > 0
        and int(summary.get("cluster_count") or 0) > 0
        and int(summary.get("transfer_artifacts") or 0) > 0
        and report.get("trigger_state") != "RED"
    )


def code_residual_forge_evidence(state: dict[str, Any]) -> dict[str, Any]:
    report = state.get("code_residual_forge") if isinstance(state.get("code_residual_forge"), dict) else {}
    return {
        "policy": report.get("policy"),
        "trigger_state": report.get("trigger_state"),
        "active_card_id": get_path(report, ["summary", "active_card_id"], None),
        "active_score": get_path(report, ["summary", "active_score"], None),
        "cluster_count": get_path(report, ["summary", "cluster_count"], None),
        "dominant_residual_class": get_path(report, ["summary", "dominant_residual_class"], None),
        "transfer_artifacts": get_path(report, ["summary", "transfer_artifacts"], None),
        "rotation_decision": get_path(report, ["summary", "rotation_decision"], None),
    }


def code_repair_organism_attempted(state: dict[str, Any]) -> bool:
    report = state.get("code_repair_organism") if isinstance(state.get("code_repair_organism"), dict) else {}
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return bool(
        report.get("policy") == "project_theseus_local_code_repair_organism_v1"
        and summary.get("transfer_loaded")
        and summary.get("transfer_altered_behavior")
        and float(summary.get("pass_rate_delta") or 0.0) > 0.0
        and int(report.get("external_inference_calls") or 0) == 0
    )


def code_repair_organism_evidence(state: dict[str, Any]) -> dict[str, Any]:
    report = state.get("code_repair_organism") if isinstance(state.get("code_repair_organism"), dict) else {}
    return {
        "policy": report.get("policy"),
        "card_id": report.get("card_id"),
        "baseline_pass_rate": get_path(report, ["summary", "baseline_pass_rate"], None),
        "transfer_pass_rate": get_path(report, ["summary", "transfer_pass_rate"], None),
        "pass_rate_delta": get_path(report, ["summary", "pass_rate_delta"], None),
        "transfer_loaded": get_path(report, ["summary", "transfer_loaded"], None),
        "transfer_altered_behavior": get_path(report, ["summary", "transfer_altered_behavior"], None),
        "external_inference_calls": report.get("external_inference_calls"),
    }


def self_edit_lane_attempted(state: dict[str, Any]) -> bool:
    report = state.get("self_edit_lane") if isinstance(state.get("self_edit_lane"), dict) else {}
    return bool(
        report.get("policy") == "project_theseus_self_edit_experiment_lane_v1"
        and report.get("trigger_state") in {"GREEN", "YELLOW"}
        and int(report.get("external_inference_calls") or 0) == 0
        and any(row.get("gate") == "residual_to_patch_contracts_written" and row.get("passed") for row in report.get("gates", []) if isinstance(row, dict))
    )


def self_edit_lane_evidence(state: dict[str, Any]) -> dict[str, Any]:
    report = state.get("self_edit_lane") if isinstance(state.get("self_edit_lane"), dict) else {}
    return {
        "policy": report.get("policy"),
        "trigger_state": report.get("trigger_state"),
        "experiment_count": len(report.get("experiments", []) if isinstance(report.get("experiments"), list) else []),
        "failed_gates": [
            row.get("gate")
            for row in report.get("gates", [])
            if isinstance(row, dict) and not row.get("passed")
        ],
    }


def long_horizon_memory_attempted(state: dict[str, Any]) -> bool:
    report = state.get("long_horizon_memory") if isinstance(state.get("long_horizon_memory"), dict) else {}
    return bool(
        report.get("policy") == "project_theseus_long_horizon_memory_probe_v1"
        and report.get("trigger_state") == "GREEN"
        and float(get_path(report, ["score", "overall"], 0.0) or 0.0) >= 0.90
        and int(report.get("external_inference_calls") or 0) == 0
    )


def long_horizon_memory_evidence(state: dict[str, Any]) -> dict[str, Any]:
    report = state.get("long_horizon_memory") if isinstance(state.get("long_horizon_memory"), dict) else {}
    return {
        "policy": report.get("policy"),
        "trigger_state": report.get("trigger_state"),
        "score": report.get("score"),
        "horizons_hours": report.get("horizons_hours"),
    }


def token_superposition_evidence(state: dict[str, Any]) -> dict[str, Any]:
    report = state.get("token_superposition") if isinstance(state.get("token_superposition"), dict) else {}
    best = report.get("best_variant") if isinstance(report.get("best_variant"), dict) else {}
    return {
        "policy": report.get("policy"),
        "backend": report.get("backend"),
        "status": get_path(report, ["promotion_decision", "status"], None),
        "best_id": best.get("id"),
        "best_nominal_speedup": best.get("nominal_speedup_vs_baseline"),
        "best_train_speedup": best.get("measured_train_speedup_vs_baseline"),
        "best_loss_delta": best.get("combined_loss_delta_vs_baseline"),
    }


def genesis_ready(state: dict[str, Any]) -> bool:
    report = state.get("genesis") if isinstance(state.get("genesis"), dict) else {}
    if report.get("policy") != "project_theseus_genesis_kernel_report_v0":
        return False
    gates = report.get("release_gates") if isinstance(report.get("release_gates"), list) else []
    failed_hard = [
        row
        for row in gates
        if isinstance(row, dict) and row.get("severity") == "hard" and not row.get("passed")
    ]
    return bool(get_path(report, ["summary", "artifact_count"], 0)) and not failed_hard


def genesis_evidence(state: dict[str, Any]) -> dict[str, Any]:
    report = state.get("genesis") if isinstance(state.get("genesis"), dict) else {}
    return {
        "policy": report.get("policy"),
        "trigger_state": get_path(report, ["summary", "trigger_state"], None),
        "artifact_count": get_path(report, ["summary", "artifact_count"], None),
        "claim_count": get_path(report, ["summary", "claim_count"], None),
        "primitive_candidate_count": get_path(report, ["summary", "primitive_candidate_count"], None),
        "failed_hard_gates": [
            row.get("gate")
            for row in report.get("release_gates", [])
            if isinstance(row, dict) and row.get("severity") == "hard" and not row.get("passed")
        ],
    }


def next_action(hard_blockers: list[dict[str, Any]], missing_evidence: list[dict[str, Any]], state: dict[str, Any]) -> str:
    if hard_blockers:
        names = {item["name"] for item in hard_blockers}
        if names == {"public_transfer_floor_cleared_for_promotion_growth"}:
            return (
                "Do not allow promotion-grade model growth yet; continue bounded private transformer/hybrid "
                "survival-lane repair and matched SymLiquid comparison until transfer evidence improves."
            )
        return f"Do not grow the model; clear hard blockers first: {', '.join(item['name'] for item in hard_blockers)}."
    if missing_evidence:
        return f"Do not grow the model yet; exhaust cheaper interventions first: {', '.join(item['name'] for item in missing_evidence)}."
    if not get_path(state, ["architecture", "architecture_change_allowed"], False):
        return "Do not grow the model; architecture experiment governance has not reached that ladder rung."
    return "Growth may be proposed as a small, localized, measured experiment with full regression gates."


def broad_public_pass_rate(state: dict[str, Any]) -> float:
    value = get_path(state, ["maturity_integrity", "summary", "broad_public_pass_rate"], None)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def public_transfer_floor_cleared(state: dict[str, Any]) -> bool:
    return broad_public_pass_rate(state) >= 0.70


def check(name: str, passed: bool, severity: str, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def read_latest_json(directory: Path, pattern: str) -> Any:
    matches = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        return {}
    return read_json(matches[-1])


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
