"""Promotion gate for real SymLiquid ratchet candidates."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import coherence_delirium_gate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REAL_CODE_GRADUATION = "reports/real_code_benchmark_graduation.json"
BABYLM_FRONTIER_FAMILY = "babylm_mutated"
RL_FRONTIER_FAMILY = "rl_local"
PRESSURE_FRONTIER_FAMILIES = {
    "minecraft_rl",
    "drone_rl",
    "coding_local_sandbox",
    "web_agent_local",
    "transfer_eval",
}
CODE_PUBLIC_TASK_FLOOR = 0.70
CANONICAL_CANDIDATE_PROFILE_REPORTS = [
    "reports/training_ratchet_candidate_profile_run.json",
    "reports/training_ratchet_candidate_evidence_profile.json",
]
ALLOWED_REAL_CODE_CANDIDATE_SOURCES = {
    "local_theseus_student_checkpoint",
    "student_learning_checkpoint_v1",
    "student_neural_checkpoint_v1",
    "student_token_generator_checkpoint_v1",
    "student_code_lm_checkpoint_v1",
}
ALLOWED_REAL_CODE_SCORE_CLAIMS = {
    "student_checkpoint_public_task_calibration_only",
    "student_learning_checkpoint_public_task_calibration_only",
    "student_neural_checkpoint_public_task_calibration_only",
    "student_token_generator_checkpoint_public_task_calibration_only",
    "student_code_lm_checkpoint_public_task_calibration_only",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture-gate", default="reports/architecture_gate_report.json")
    parser.add_argument("--rmi", default="reports/ratcheting_modular_intelligence_report.json")
    parser.add_argument(
        "--public-report",
        default="reports/blimp_filtered_train_800k_evalfull_hv16k_lr02_complexnpfix.json",
    )
    parser.add_argument(
        "--seed49-regression",
        default="reports/babylm_mutated_holdout_seed49_stateful_grammar_state_frontier.json",
    )
    parser.add_argument(
        "--seed55-frontier",
        default="reports/babylm_mutated_holdout_seed55_stateful_grammar_state_frontier.json",
    )
    parser.add_argument(
        "--frontier-report",
        default="",
        help="Active frontier report for this candidate/profile run. Defaults to the profile artifact, latest mutated frontier, or historical seed55.",
    )
    parser.add_argument("--residual-escrow", default="reports/residual_escrow.json")
    parser.add_argument(
        "--residual-baseline",
        default="reports/residual_escrow_pre_candidate_baseline.json",
        help="Residual escrow snapshot from before the candidate/frontier run.",
    )
    parser.add_argument("--max-cluster-delta", type=int, default=10)
    parser.add_argument("--max-active-diagnostic-delta", type=int, default=5)
    parser.add_argument("--max-critical-delta", type=int, default=0)
    parser.add_argument("--max-residual-delta", type=float, default=0.05)
    parser.add_argument("--runtime-report", default="reports/preflight_cuda_rollout_smoke.json")
    parser.add_argument("--synthetic-report", default="reports/synthetic_data_curator.json")
    parser.add_argument("--profile-report", default="reports/training_ratchet_profile_run.json")
    parser.add_argument(
        "--allow-profile-step-in-progress",
        default="",
        help="Treat one profile step as present when this gate is run from inside that step.",
    )
    parser.add_argument("--transfer-artifacts", default="reports/arm_transfer_artifacts.json")
    parser.add_argument("--code-residual-forge", default="reports/code_residual_forge.json")
    parser.add_argument("--real-code-graduation", default="reports/real_code_benchmark_graduation.json")
    parser.add_argument("--broad-transfer-matrix", default="reports/broad_transfer_matrix.json")
    parser.add_argument("--student-learning-closure", default="reports/student_learning_closure.json")
    parser.add_argument("--coherence-gate", default="reports/coherence_delirium_gate.json")
    parser.add_argument("--maturity-integrity-audit", default="reports/maturity_integrity_audit.json")
    parser.add_argument("--blind-information-flow-audit", default="reports/blind_information_flow_audit.json")
    parser.add_argument("--candidate-integrity-audit", default="reports/candidate_integrity_audit.json")
    parser.add_argument("--out", default="reports/candidate_promotion_gate.json")
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="Return success when the report is written even if promotion remains blocked. Use for roadmap/CI guard validation, not promotion.",
    )
    args = parser.parse_args()

    architecture = read_json(Path(args.architecture_gate))
    rmi = read_json(Path(args.rmi))
    public = read_json(Path(args.public_report))
    seed49 = read_json(Path(args.seed49_regression))
    requested_profile_run = read_json(Path(args.profile_report))
    frontier_report, frontier_report_source = resolve_frontier_report(args, requested_profile_run)
    frontier = read_json(Path(frontier_report))
    profile_run, profile_report_path = resolve_profile_report(
        args.profile_report,
        args.allow_profile_step_in_progress,
        frontier_report=frontier_report,
        frontier=frontier,
    )
    seed55 = read_json(Path(args.seed55_frontier))
    escrow = read_json(Path(args.residual_escrow))
    residual_baseline = read_json(Path(args.residual_baseline))
    runtime = read_json(Path(args.runtime_report))
    synthetic = read_json(Path(args.synthetic_report))
    transfer_artifacts = read_json(Path(args.transfer_artifacts))
    code_residual_forge = read_json(Path(args.code_residual_forge))
    real_code_graduation, real_code_graduation_path = resolve_real_code_graduation_report(args.real_code_graduation)
    broad_transfer_matrix = read_json(Path(args.broad_transfer_matrix))
    student_learning_closure = read_json(Path(args.student_learning_closure))
    coherence_gate = read_json(Path(args.coherence_gate)) or coherence_delirium_gate.load_gate()
    maturity_integrity = read_json(Path(args.maturity_integrity_audit))
    blind_information_flow = read_json(Path(args.blind_information_flow_audit))
    candidate_integrity = read_json(Path(args.candidate_integrity_audit))
    delta = residual_delta(residual_baseline, escrow)
    synthetic_used = "data/synthetic/" in str(frontier.get("input_path") or "").replace("\\", "/")
    synthetic_benchmark_used = synthetic_benchmark_pressure(frontier)
    multi_stream_used = multi_stream_pressure(frontier)
    frontier_floor_cleared = (accuracy(frontier) or 0.0) >= 0.70

    checks = [
        check(
            "architecture_gate_green",
            bool(architecture.get("ready_for_heavy_training")),
            f"ready_for_heavy_training={architecture.get('ready_for_heavy_training')}",
        ),
        check(
            "rmi_score_green",
            get_path(rmi, ["implementation_score", "score"], 0.0) >= 1.0,
            f"score={get_path(rmi, ['implementation_score', 'score'], None)}",
        ),
        check(
            "public_comparator_present",
            accuracy(public) is not None,
            f"accuracy={accuracy(public)} path={args.public_report}",
        ),
        check(
            "public_comparator_no_regression",
            (accuracy(public) or 0.0) >= 0.90,
            f"accuracy={accuracy(public)} threshold=0.90",
        ),
        check(
            "seed49_regression_holds",
            (accuracy(seed49) or 0.0) >= 0.90,
            f"accuracy={accuracy(seed49)} threshold=0.90 path={args.seed49_regression}",
        ),
        check(
            "active_frontier_exists",
            bool(frontier),
            f"path={frontier_report}",
        ),
        check(
            "active_frontier_clears_floor",
            (accuracy(frontier) or 0.0) >= 0.70,
            f"accuracy={accuracy(frontier)} floor=0.70 path={frontier_report}",
        ),
        check(
            "active_frontier_training_budget_sufficient",
            pressure_budget_sufficient(profile_run, frontier),
            pressure_budget_evidence(profile_run, frontier),
        ),
        check(
            "synthetic_data_governed",
            (not synthetic_used) or synthetic_report_ok(synthetic),
            synthetic_evidence(synthetic, synthetic_used, args.synthetic_report),
        ),
        check(
            "synthetic_benchmark_private_pressure_only",
            not synthetic_benchmark_used,
            synthetic_benchmark_evidence(frontier, synthetic_benchmark_used),
        ),
        check(
            "multi_stream_private_pressure_only",
            not multi_stream_used,
            multi_stream_evidence(frontier, multi_stream_used),
        ),
        check(
            "candidate_profile_evidence_complete",
            candidate_profile_complete(profile_run, args.allow_profile_step_in_progress)
            and candidate_profile_matches_frontier(profile_run, frontier_report, frontier),
            candidate_profile_evidence(
                profile_run,
                profile_report_path,
                args.allow_profile_step_in_progress,
                frontier_report=frontier_report,
                frontier=frontier,
            ),
        ),
        check(
            "residual_escrow_active",
            get_path(escrow, ["summary", "cluster_count"], 0) > 0,
            f"clusters={get_path(escrow, ['summary', 'cluster_count'], 0)}",
        ),
        check(
            "residual_baseline_present",
            bool(residual_baseline),
            f"path={args.residual_baseline}",
        ),
        check(
            "residual_cluster_delta_bounded",
            bool(residual_baseline) and delta["cluster_count_delta"] <= args.max_cluster_delta,
            f"delta={delta['cluster_count_delta']} max={args.max_cluster_delta}",
        ),
        check(
            "active_diagnostic_delta_bounded",
            bool(residual_baseline)
            and delta["active_diagnostic_target_count_delta"] <= args.max_active_diagnostic_delta,
            f"delta={delta['active_diagnostic_target_count_delta']} max={args.max_active_diagnostic_delta}",
        ),
        check(
            "critical_residual_delta_bounded",
            bool(residual_baseline) and delta["critical_cluster_delta"] <= args.max_critical_delta,
            f"delta={delta['critical_cluster_delta']} max={args.max_critical_delta}",
        ),
        check(
            "max_residual_delta_bounded",
            bool(residual_baseline) and delta["max_residual_delta"] <= args.max_residual_delta,
            f"delta={delta['max_residual_delta']:.6f} max={args.max_residual_delta}",
        ),
        check(
            "runtime_cost_reported",
            runtime_cost_reported(runtime),
            runtime_cost_evidence(runtime),
        ),
        check(
            "cuda_no_fallback",
            accelerator_no_bad_fallback(runtime),
            accelerator_fallback_evidence(runtime),
        ),
        check(
            "graduation_transfer_artifact_ready",
            (not frontier_floor_cleared) or graduation_transfer_artifact_ready(transfer_artifacts, profile_run, frontier),
            transfer_artifact_evidence(transfer_artifacts, args.transfer_artifacts, frontier_floor_cleared, profile_run, frontier),
        ),
        check(
            "code_frontier_transfer_artifact_ready",
            code_frontier_transfer_artifact_ready(code_residual_forge, profile_run, frontier),
            code_frontier_transfer_artifact_evidence(code_residual_forge, args.code_residual_forge, profile_run, frontier),
        ),
        check(
            "code_frontier_transfer_consumed",
            code_frontier_transfer_consumed(profile_run, frontier),
            code_frontier_transfer_consumed_evidence(profile_run, frontier),
        ),
        check(
            "real_code_benchmark_graduation_ready",
            real_code_benchmark_graduation_ready(real_code_graduation, profile_run, frontier),
            real_code_benchmark_graduation_evidence(real_code_graduation, real_code_graduation_path, profile_run, frontier),
        ),
        check(
            "broad_public_code_transfer_ready",
            broad_public_code_transfer_ready(broad_transfer_matrix, profile_run, frontier),
            broad_public_code_transfer_evidence(broad_transfer_matrix, args.broad_transfer_matrix, profile_run, frontier),
        ),
        check(
            "student_learning_closure_ready",
            student_learning_closure_ready(student_learning_closure, real_code_graduation, profile_run, frontier),
            student_learning_closure_evidence(student_learning_closure, args.student_learning_closure, real_code_graduation, profile_run, frontier),
        ),
        check(
            "coherence_delirium_candidate_promotion_allowed",
            bool(coherence_gate.get("allows_candidate_promotion")),
            coherence_gate_evidence(coherence_gate, args.coherence_gate),
        ),
        check(
            "maturity_integrity_audit_green",
            maturity_integrity_ready(maturity_integrity),
            maturity_integrity_evidence(maturity_integrity, args.maturity_integrity_audit),
        ),
        check(
            "blind_information_flow_audit_green",
            blind_information_flow_ready(blind_information_flow),
            blind_information_flow_evidence(blind_information_flow, args.blind_information_flow_audit),
        ),
        check(
            "candidate_integrity_viea_receipt_ready",
            candidate_integrity_ready(candidate_integrity),
            candidate_integrity_evidence(candidate_integrity, args.candidate_integrity_audit),
        ),
    ]

    passed = sum(1 for item in checks if item["passed"])
    promotion_integrity = promotion_integrity_summary(candidate_integrity, args.candidate_integrity_audit)
    report = {
        "policy": "local_only_no_external_inference",
        "methodology": "rmi_candidate_promotion_gate",
        "promote": passed == len(checks),
        "passed": passed,
        "total": len(checks),
        "checks": checks,
        "scores": {
            "public_accuracy": accuracy(public),
            "seed49_regression_accuracy": accuracy(seed49),
            "active_frontier_accuracy": accuracy(frontier),
            "seed55_frontier_accuracy": accuracy(seed55),
        },
        "artifacts": {
            "architecture_gate": args.architecture_gate,
            "rmi": args.rmi,
            "public_report": args.public_report,
            "seed49_regression": args.seed49_regression,
            "active_frontier": frontier_report,
            "active_frontier_source": frontier_report_source,
            "active_frontier_family": active_frontier_family(profile_run, frontier),
            "seed55_frontier": args.seed55_frontier,
            "residual_escrow": args.residual_escrow,
            "residual_baseline": args.residual_baseline,
            "runtime_report": args.runtime_report,
            "synthetic_report": args.synthetic_report,
            "profile_report": profile_report_path,
            "transfer_artifacts": args.transfer_artifacts,
            "code_residual_forge": args.code_residual_forge,
            "real_code_graduation": real_code_graduation_path,
            "broad_transfer_matrix": args.broad_transfer_matrix,
            "student_learning_closure": args.student_learning_closure,
            "coherence_gate": args.coherence_gate,
            "maturity_integrity_audit": args.maturity_integrity_audit,
            "blind_information_flow_audit": args.blind_information_flow_audit,
            "candidate_integrity_audit": args.candidate_integrity_audit,
        },
        "promotion_integrity": promotion_integrity,
        "runtime_governance": {
            "coherence_delirium": {
                "trigger_state": coherence_gate.get("trigger_state"),
                "source_trigger_state": coherence_gate.get("source_trigger_state"),
                "coherence_score": coherence_gate.get("coherence_score"),
                "delirium_score": coherence_gate.get("delirium_score"),
                "allows_candidate_promotion": coherence_gate.get("allows_candidate_promotion"),
                "candidate_blockers": coherence_gate.get("candidate_blockers", []),
            }
        },
        "residual_delta": delta,
    }
    write_json(Path(args.out), report)
    print(json.dumps(report, indent=2))
    return 0 if report["promote"] or args.allow_blocked else 1


def check(name: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def runtime_cost_reported(runtime: dict[str, Any]) -> bool:
    if bool(get_path(runtime, ["runtime_profile"], {})) and bool(get_path(runtime, ["timing_breakdown_ms"], {})):
        return True
    if runtime.get("policy") == "project_theseus_macos_training_preflight_v0":
        worker = get_path(runtime, ["execution", "worker_report", "payload"], {})
        metrics = worker.get("metrics") if isinstance(worker, dict) else {}
        return bool(worker.get("runtime_ms") is not None and metrics)
    return False


def runtime_cost_evidence(runtime: dict[str, Any]) -> str:
    if runtime.get("policy") == "project_theseus_macos_training_preflight_v0":
        worker = get_path(runtime, ["execution", "worker_report", "payload"], {})
        metrics = worker.get("metrics") if isinstance(worker, dict) else {}
        return (
            f"macos_preflight=True worker_runtime_ms={worker.get('runtime_ms') if isinstance(worker, dict) else None} "
            f"metric_keys={sorted(metrics.keys())[:8] if isinstance(metrics, dict) else []}"
        )
    return (
        f"runtime_profile={bool(get_path(runtime, ['runtime_profile'], {}))} "
        f"timing={bool(get_path(runtime, ['timing_breakdown_ms'], {}))}"
    )


def accelerator_no_bad_fallback(runtime: dict[str, Any]) -> bool:
    if "cuda_fallback" in runtime:
        return runtime.get("cuda_fallback") is False
    if runtime.get("policy") == "project_theseus_macos_training_preflight_v0":
        worker = get_path(runtime, ["execution", "worker_report", "payload"], {})
        backend = str(worker.get("backend") or "") if isinstance(worker, dict) else ""
        return bool(
            runtime.get("bounded_smoke_allowed")
            and get_path(runtime, ["execution", "ok"], False)
            and backend in {"mlx_apple", "apple_mlx"}
            and int(get_path(runtime, ["execution", "external_inference_calls"], 0) or 0) == 0
        )
    return False


def accelerator_fallback_evidence(runtime: dict[str, Any]) -> str:
    if runtime.get("policy") == "project_theseus_macos_training_preflight_v0":
        worker = get_path(runtime, ["execution", "worker_report", "payload"], {})
        return (
            f"macos_preflight=True execution_ok={get_path(runtime, ['execution', 'ok'], None)} "
            f"bounded_smoke_allowed={runtime.get('bounded_smoke_allowed')} "
            f"backend={worker.get('backend') if isinstance(worker, dict) else None} "
            f"external_inference_calls={get_path(runtime, ['execution', 'external_inference_calls'], None)}"
        )
    return f"cuda_fallback={runtime.get('cuda_fallback')}"


def maturity_integrity_ready(report: dict[str, Any]) -> bool:
    if report.get("policy") != "project_theseus_maturity_integrity_audit_v1":
        return False
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return bool(
        report.get("trigger_state") == "GREEN"
        and int(summary.get("hard_blocker_count") or 0) == 0
        and bool(summary.get("public_calibration_allowed"))
        and bool(summary.get("candidate_promotion_allowed"))
    )


def maturity_integrity_evidence(report: dict[str, Any], path: str) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return (
        f"path={path} state={report.get('trigger_state')} "
        f"hard_blockers={summary.get('hard_blocker_count')} "
        f"maturity_blockers={summary.get('maturity_blocker_count')} "
        f"evidence_blockers={summary.get('evidence_blocker_count')} "
        f"public_calibration_allowed={summary.get('public_calibration_allowed')} "
        f"candidate_promotion_allowed={summary.get('candidate_promotion_allowed')} "
        f"manifest_public_leak_hits={summary.get('manifest_public_leak_hit_count')}"
    )


def blind_information_flow_ready(report: dict[str, Any]) -> bool:
    if report.get("policy") != "project_theseus_blind_information_flow_audit_v1":
        return False
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return bool(report.get("trigger_state") == "GREEN" and int(summary.get("invalid_claim_count") or 0) == 0)


def blind_information_flow_evidence(report: dict[str, Any], path: str) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return (
        f"path={path} state={report.get('trigger_state')} "
        f"invalid_claims={summary.get('invalid_claim_count')} "
        f"static_violations={summary.get('static_information_flow_violation_count')} "
        f"candidate_overclaims={summary.get('candidate_overclaim_count')} "
        f"report_overclaims={summary.get('report_overclaim_count')} "
        f"missing_inputs={summary.get('missing_input_count')}"
    )


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def candidate_integrity_ready(report: dict[str, Any]) -> bool:
    if report.get("policy") != "project_theseus_candidate_integrity_audit_v1":
        return False
    summary = dict_value(report.get("summary"))
    receipt = dict_value(report.get("viea_spine_consumer_receipt"))
    claimed = dict_value(summary.get("claimed_promotion_by_family"))
    verified = dict_value(summary.get("integrity_verified_by_family"))
    non_promotion_claims = {
        family: int(number(count))
        for family, count in claimed.items()
        if family not in set(summary.get("promotion_families") or []) and int(number(count)) > 0
    }
    promotion_verified = sum(
        int(number(count))
        for family, count in verified.items()
        if family in set(summary.get("promotion_families") or [])
    )
    return bool(
        report.get("trigger_state") == "GREEN"
        and receipt.get("ready") is True
        and int(number(summary.get("integrity_mismatch_count"))) == 0
        and promotion_verified > 0
        and not non_promotion_claims
        and int(number(receipt.get("no_cheat_fault_count"))) == 0
    )


def candidate_integrity_evidence(report: dict[str, Any], path: str) -> str:
    summary = dict_value(report.get("summary"))
    receipt = dict_value(report.get("viea_spine_consumer_receipt"))
    verified = dict_value(summary.get("integrity_verified_by_family"))
    promotion_families = set(summary.get("promotion_families") or [])
    promotion_verified = {
        family: int(number(count))
        for family, count in verified.items()
        if family in promotion_families and int(number(count)) > 0
    }
    return (
        f"path={path} state={report.get('trigger_state')} "
        f"viea_ready={receipt.get('ready')} "
        f"viea_record_count={receipt.get('record_count')} "
        f"no_cheat_fault_count={receipt.get('no_cheat_fault_count')} "
        f"candidate_count={summary.get('candidate_count')} "
        f"integrity_verified_candidate_count={summary.get('integrity_verified_candidate_count')} "
        f"integrity_mismatch_count={summary.get('integrity_mismatch_count')} "
        f"promotion_verified_by_family={promotion_verified} "
        f"family_counts={summary.get('family_counts')}"
    )


def promotion_integrity_summary(report: dict[str, Any], path: str) -> dict[str, Any]:
    summary = dict_value(report.get("summary"))
    receipt = dict_value(report.get("viea_spine_consumer_receipt"))
    verified = dict_value(summary.get("integrity_verified_by_family"))
    family_counts = dict_value(summary.get("family_counts"))
    promotion_families = set(summary.get("promotion_families") or [])
    promotion_verified = {
        family: int(number(count))
        for family, count in verified.items()
        if family in promotion_families and int(number(count)) > 0
    }
    claimed = dict_value(summary.get("claimed_promotion_by_family"))
    non_promotion_claims = {
        family: int(number(count))
        for family, count in claimed.items()
        if family not in promotion_families and int(number(count)) > 0
    }
    return {
        "policy": "project_theseus_candidate_promotion_integrity_receipt_v1",
        "source": path,
        "candidate_integrity_policy": report.get("policy"),
        "candidate_integrity_trigger_state": report.get("trigger_state"),
        "viea_spine_consumer_receipt": receipt,
        "candidate_count": summary.get("candidate_count"),
        "audited_candidate_count": summary.get("audited_candidate_count"),
        "family_counts": family_counts,
        "promotion_families": sorted(promotion_families),
        "promotion_verified_by_family": promotion_verified,
        "non_promotion_claims_by_family": non_promotion_claims,
        "integrity_verified_candidate_count": summary.get("integrity_verified_candidate_count"),
        "integrity_mismatch_count": summary.get("integrity_mismatch_count"),
        "learned_generation_claim_boundary": (
            "Only independently recomputed learned_full_body_token, transformer_hybrid, or symliquid "
            "families with integrity verification may support learned-generation promotion. "
            "Fallback/template/router/tool families remain non-promotion evidence."
        ),
        "ready_for_promotion_claims": candidate_integrity_ready(report),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def coherence_gate_evidence(gate_report: dict[str, Any], path: str) -> str:
    return (
        f"path={path} state={gate_report.get('trigger_state')} "
        f"source={gate_report.get('source_trigger_state')} "
        f"coherence={gate_report.get('coherence_score')} delirium={gate_report.get('delirium_score')} "
        f"candidate_blockers={gate_report.get('candidate_blockers', [])}"
    )


def resolve_profile_report(
    path: str,
    allow_step_in_progress: str = "",
    *,
    frontier_report: str = "",
    frontier: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    frontier = frontier or {}
    requested = read_json(Path(path))
    if candidate_profile_complete(requested, allow_step_in_progress) and candidate_profile_matches_frontier(
        requested,
        frontier_report,
        frontier,
    ):
        return requested, path
    matching_incomplete: tuple[dict[str, Any], str] | None = None
    for candidate in CANONICAL_CANDIDATE_PROFILE_REPORTS:
        if candidate == path:
            continue
        candidate_report = read_json(Path(candidate))
        if not candidate_profile_matches_frontier(
            candidate_report,
            frontier_report,
            frontier,
        ):
            continue
        if candidate_profile_complete(candidate_report, allow_step_in_progress):
            return candidate_report, candidate
        if matching_incomplete is None:
            matching_incomplete = (candidate_report, candidate)
    if matching_incomplete is not None:
        return matching_incomplete
    return requested, path


def resolve_frontier_report(args: argparse.Namespace, profile_run: dict[str, Any]) -> tuple[str, str]:
    family = str(profile_run.get("frontier_family") or "")
    explicit_is_babylm = is_babylm_frontier_path(args.frontier_report)
    if args.frontier_report and (family in {"", BABYLM_FRONTIER_FAMILY} or not explicit_is_babylm):
        return args.frontier_report, "explicit_arg"
    if not args.frontier_report:
        canonical_profile = newest_complete_candidate_profile(args.allow_profile_step_in_progress)
        if canonical_profile:
            canonical_run, canonical_path = canonical_profile
            canonical_family = str(canonical_run.get("frontier_family") or "")
            if canonical_family in PRESSURE_FRONTIER_FAMILIES:
                pressure_frontier = existing_profile_artifact(canonical_run, "pressure_runner")
                if pressure_frontier:
                    return pressure_frontier, f"canonical_candidate_profile:{canonical_path}"
    if family in PRESSURE_FRONTIER_FAMILIES:
        policy_frontier = frontier_policy_active_report(family)
        if policy_frontier:
            return policy_frontier, "frontier_policy_status"
        pressure_frontier = existing_profile_artifact(profile_run, "pressure_runner")
        if pressure_frontier:
            return pressure_frontier, "profile_pressure_runner"
    if family == RL_FRONTIER_FAMILY:
        rl_smoke = existing_profile_artifact(profile_run, "rl_frontier_smoke")
        if rl_smoke:
            return rl_smoke, "profile_rl_frontier_smoke"
        rl_train = existing_profile_artifact(profile_run, "rl_frontier_train")
        if rl_train:
            return rl_train, "profile_rl_frontier_train"
    if args.frontier_report:
        return args.frontier_report, "explicit_arg_legacy_fallback"
    profile_frontier = get_path(profile_run, ["artifacts", "mutated_frontier"], "")
    latest = latest_mutated_frontier_report(Path(args.seed55_frontier).parent)
    if profile_frontier:
        profile_seed = seed_from_frontier_path(str(profile_frontier))
        latest_seed = seed_from_frontier_path(str(latest or ""))
        if latest is None or profile_seed >= latest_seed:
            return str(profile_frontier), "profile_artifact"
    if latest:
        return str(latest), "latest_mutated_frontier"
    return args.seed55_frontier, "historical_seed55_fallback"


def newest_complete_candidate_profile(allow_step_in_progress: str = "") -> tuple[dict[str, Any], str] | None:
    candidates: list[tuple[float, dict[str, Any], str]] = []
    for path in CANONICAL_CANDIDATE_PROFILE_REPORTS:
        report_path = Path(path)
        if not report_path.exists():
            continue
        report = read_json(report_path)
        if not candidate_profile_complete(report, allow_step_in_progress):
            continue
        candidates.append((report_path.stat().st_mtime, report, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, report, path = candidates[0]
    return report, path


def frontier_policy_active_report(family: str) -> str:
    policy = read_json(Path("reports/frontier_policy_status.json"))
    candidates: list[Any] = []
    frontier = policy.get("frontier")
    if isinstance(frontier, dict):
        candidates.append(frontier)
    for key in ("active_frontier", "selected"):
        value = policy.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    pressure = policy.get("frontier_pressure") if isinstance(policy.get("frontier_pressure"), dict) else {}
    expected_family = str(policy.get("frontier_family") or pressure.get("next_frontier_family") or family or "")
    expected_card = str(policy.get("pressure_card_id") or pressure.get("next_pressure_card_id") or "")
    ledger = read_json(Path("reports/benchmark_ledger.json"))
    if isinstance(ledger, list):
        candidates.extend(item for item in ledger if isinstance(item, dict))
    for candidate in candidates:
        report = str(candidate.get("best_report") or candidate.get("report") or "")
        benchmark_name = str(candidate.get("benchmark_name") or candidate.get("family") or "")
        if not report or not Path(report).exists():
            continue
        if expected_family and not (
            benchmark_name.startswith(expected_family)
            or expected_family in benchmark_name
            or expected_family in report
        ):
            continue
        if expected_card and expected_card not in report and expected_card not in benchmark_name:
            continue
        return report
    if expected_card:
        report_candidates = sorted(
            Path("reports").glob(f"pressure_{safe_report_glob(expected_card)}_seed*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if report_candidates:
            return str(report_candidates[0])
    return ""


def safe_report_glob(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def existing_profile_artifact(profile_run: dict[str, Any], key: str) -> str:
    value = str(get_path(profile_run, ["artifacts", key], "") or "")
    return value if value and Path(value).exists() else ""


def is_babylm_frontier_path(value: str) -> bool:
    return "babylm_mutated_holdout_seed" in str(value or "").replace("\\", "/")


def active_frontier_family(profile_run: dict[str, Any], frontier: dict[str, Any]) -> str:
    return str(
        frontier.get("frontier_family")
        or profile_run.get("frontier_family")
        or get_path(frontier, ["summary", "suite"], "")
        or BABYLM_FRONTIER_FAMILY
    )


def latest_mutated_frontier_report(reports_dir: Path) -> Path | None:
    candidates: list[tuple[int, float, Path]] = []
    for path in reports_dir.glob("babylm_mutated_holdout_seed*_stateful_grammar_state_frontier.json"):
        match = re.search(r"seed(\d+)_", path.name)
        if not match:
            continue
        seed = int(match.group(1))
        candidates.append((seed, path.stat().st_mtime, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][2]


def seed_from_frontier_path(value: str) -> int:
    match = re.search(r"seed(\d+)", value)
    return int(match.group(1)) if match else -1


def accuracy(report: dict[str, Any]) -> float | None:
    value = get_path(report, ["eval", "summary", "accuracy"], None)
    if value is None:
        value = get_path(report, ["summary", "accuracy"], None)
    if value is None:
        value = get_path(report, ["summary", "score"], None)
    if value is None:
        value = report.get("accuracy")
    if value is None:
        value = report.get("normalized_perf")
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def synthetic_report_ok(report: dict[str, Any]) -> bool:
    return (
        bool(report.get("training_ready"))
        and bool(get_path(report, ["verification", "ok"], False))
        and float(report.get("blend_synthetic_ratio") or 1.0) <= 0.1200001
        and int(report.get("external_inference_calls") or 0) == 0
    )


def synthetic_evidence(report: dict[str, Any], synthetic_used: bool, path: str) -> str:
    if not synthetic_used:
        return "not_used_by_active_frontier"
    return (
        f"path={path} ready={report.get('training_ready')} "
        f"ratio={report.get('blend_synthetic_ratio')} "
        f"verification_ok={get_path(report, ['verification', 'ok'], None)} "
        f"external_inference_calls={report.get('external_inference_calls')}"
    )


def synthetic_benchmark_pressure(frontier: dict[str, Any]) -> bool:
    return bool(
        str(frontier.get("card_id") or "").startswith("synthetic_")
        or str(frontier.get("runner_family") or "") == "synthetic_benchmark_local"
        or "synthetic_benchmark" in str(get_path(frontier, ["summary", "suite"], ""))
        or get_path(frontier, ["metrics", "score_semantics"], "")
        == "private_pressure_readiness_and_unit_test_pass_rate_not_public_comparator_accuracy"
    )


def synthetic_benchmark_evidence(frontier: dict[str, Any], synthetic_used: bool) -> str:
    if not synthetic_used:
        return "not_used_by_active_frontier"
    return (
        f"card_id={frontier.get('card_id')} runner_family={frontier.get('runner_family')} "
        f"score_semantics={get_path(frontier, ['metrics', 'score_semantics'], '')} "
        "rule=private_synthetic_pressure_cannot_promote_without_real_benchmark_regression"
    )


def multi_stream_pressure(frontier: dict[str, Any]) -> bool:
    semantics = str(get_path(frontier, ["metrics", "score_semantics"], ""))
    return bool(
        str(frontier.get("card_id") or "").startswith("multistream_")
        or str(frontier.get("runner_family") or "") == "multi_stream_code_pressure"
        or "multi_stream_code_pressure" in str(get_path(frontier, ["summary", "suite"], ""))
        or "private_multistream_pressure" in semantics
    )


def multi_stream_evidence(frontier: dict[str, Any], multi_stream_used: bool) -> str:
    if not multi_stream_used:
        return "not_used_by_active_frontier"
    return (
        f"card_id={frontier.get('card_id')} runner_family={frontier.get('runner_family')} "
        f"score_semantics={get_path(frontier, ['metrics', 'score_semantics'], '')} "
        "rule=private_multi_stream_pressure_cannot_promote_without_real_benchmark_regression"
    )


def candidate_profile_complete(report: dict[str, Any], allow_step_in_progress: str = "") -> bool:
    if report.get("ok") is not True:
        return False
    step_names = {
        str(step.get("name"))
        for step in report.get("steps", [])
        if isinstance(step, dict) and step.get("returncode") == 0
    }
    if allow_step_in_progress:
        step_names.add(str(allow_step_in_progress))
    family = str(report.get("frontier_family") or BABYLM_FRONTIER_FAMILY)
    if family in PRESSURE_FRONTIER_FAMILIES:
        required = {
            "ablation_matrix",
            "profile_vram_stress",
            "capability_ratchet_refresh",
            "training_preflight_refresh",
        }
        pressure_step_present = any(name.startswith("pressure_runner_") for name in step_names)
        code_forge_present = family != "coding_local_sandbox" or any(
            name.startswith("code_residual_forge_") for name in step_names
        )
        return required.issubset(step_names) and pressure_step_present and code_forge_present
    if family == RL_FRONTIER_FAMILY:
        required = {
            "ablation_matrix",
            "profile_vram_stress",
            "capability_ratchet_refresh",
            "training_preflight_refresh",
        }
        train_present = any(name.startswith("rl_frontier_train_") for name in step_names)
        smoke_present = any(name.startswith("rl_frontier_smoke_") for name in step_names)
        return required.issubset(step_names) and train_present and smoke_present
    if report.get("profile") != "candidate":
        return False
    required = {
        "synthetic_data_curator",
        "ablation_matrix",
        "profile_vram_stress",
        "capability_ratchet_refresh",
        "training_preflight_refresh",
    }
    frontier_step_present = bool({"seed55_frontier", "mutated_frontier"} & step_names)
    return required.issubset(step_names) and frontier_step_present


def candidate_profile_matches_frontier(report: dict[str, Any], frontier_report: str, frontier: dict[str, Any]) -> bool:
    family = str(report.get("frontier_family") or "")
    active_family = active_frontier_family(report, frontier)
    if family and active_family and family != active_family:
        return False
    active_card = active_card_id(frontier_report, frontier)
    profile_card = str(report.get("pressure_card_id") or "")
    if not profile_card:
        profile_card = profile_pressure_card_id(report)
    return not active_card or not profile_card or active_card == profile_card


def active_card_id(frontier_report: str, frontier: dict[str, Any]) -> str:
    value = str(frontier.get("card_id") or frontier.get("pressure_card_id") or "")
    if value:
        return value
    match = re.search(r"pressure_(.+)_seed\d+", str(frontier_report).replace("\\", "/"))
    return match.group(1) if match else ""


def profile_pressure_card_id(report: dict[str, Any]) -> str:
    for step in report.get("steps", []) if isinstance(report.get("steps"), list) else []:
        if not isinstance(step, dict):
            continue
        name = str(step.get("name") or "")
        match = re.match(r"pressure_runner_(.+)_seed\d+$", name)
        if match:
            return match.group(1)
    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
    value = str(artifacts.get("pressure_runner") or artifacts.get("mutated_frontier") or "")
    match = re.search(r"pressure_(.+)_seed\d+", value.replace("\\", "/"))
    return match.group(1) if match else ""


def candidate_profile_evidence(
    report: dict[str, Any],
    path: str,
    allow_step_in_progress: str = "",
    *,
    frontier_report: str = "",
    frontier: dict[str, Any] | None = None,
) -> str:
    frontier = frontier or {}
    step_names = [
        str(step.get("name"))
        for step in report.get("steps", [])
        if isinstance(step, dict) and step.get("returncode") == 0
    ]
    active_card = active_card_id(frontier_report, frontier)
    profile_card = str(report.get("pressure_card_id") or profile_pressure_card_id(report) or "")
    return (
        f"path={path} profile={report.get('profile')} ok={report.get('ok')} "
        f"allow_step_in_progress={allow_step_in_progress or 'none'} "
        f"profile_card={profile_card or 'unknown'} active_card={active_card or 'unknown'} "
        f"frontier_match={candidate_profile_matches_frontier(report, frontier_report, frontier)} "
        f"completed_steps={','.join(step_names)}"
    )


def pressure_budget_sufficient(profile_run: dict[str, Any], frontier: dict[str, Any]) -> bool:
    family = str(profile_run.get("frontier_family") or "")
    if family not in PRESSURE_FRONTIER_FAMILIES:
        return True
    budget = frontier.get("budget") if isinstance(frontier.get("budget"), dict) else {}
    min_evals = int(budget.get("min_train_candidate_evaluations") or 0)
    min_steps = int(budget.get("min_train_env_steps") or 0)
    actual_evals = int(budget.get("train_candidate_evaluations") or 0)
    actual_steps = int(budget.get("train_env_steps_budget") or 0)
    if min_evals <= 0 and min_steps <= 0:
        return False
    budget_ok = actual_evals >= min_evals and actual_steps >= min_steps
    status = str(frontier.get("status") or "")
    if status in {"runtime_blocked", "timeout", "partial_budget_exhausted"}:
        return False
    checks = frontier.get("checks") if isinstance(frontier.get("checks"), list) else []
    blocking_checks = {
        "train_before_eval_candidate_budget",
        "train_before_eval_env_step_budget",
    }
    for item in checks:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("gate") or "")
        if name in blocking_checks and not item.get("passed"):
            return False
    return budget_ok


def pressure_budget_evidence(profile_run: dict[str, Any], frontier: dict[str, Any]) -> str:
    family = str(profile_run.get("frontier_family") or "")
    if family not in PRESSURE_FRONTIER_FAMILIES:
        return "not_pressure_frontier"
    budget = frontier.get("budget") if isinstance(frontier.get("budget"), dict) else {}
    checks = frontier.get("checks") if isinstance(frontier.get("checks"), list) else []
    budget_checks = []
    for item in checks:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("gate") or "")
        if name in {"train_before_eval_candidate_budget", "train_before_eval_env_step_budget"}:
            budget_checks.append(f"{name}={bool(item.get('passed'))}")
    return (
        f"status={frontier.get('status')} "
        f"candidate_evals={budget.get('train_candidate_evaluations')} "
        f"min_candidate_evals={budget.get('min_train_candidate_evaluations')} "
        f"env_steps={budget.get('train_env_steps_budget')} "
        f"min_env_steps={budget.get('min_train_env_steps')} "
        f"checks={','.join(budget_checks) or 'missing'}"
    )


def residual_delta(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    prev_summary = previous.get("summary") or {}
    cur_summary = current.get("summary") or {}
    prev_clusters = previous.get("clusters") or []
    cur_clusters = current.get("clusters") or []
    return {
        "cluster_count_delta": int(cur_summary.get("cluster_count") or 0)
        - int(prev_summary.get("cluster_count") or 0),
        "case_count_delta": int(cur_summary.get("case_count") or 0)
        - int(prev_summary.get("case_count") or 0),
        "active_diagnostic_target_count_delta": int(
            cur_summary.get("active_diagnostic_target_count") or 0
        )
        - int(prev_summary.get("active_diagnostic_target_count") or 0),
        "critical_cluster_delta": critical_cluster_count(cur_clusters)
        - critical_cluster_count(prev_clusters),
        "max_residual_delta": max_residual(cur_clusters) - max_residual(prev_clusters),
    }


def critical_cluster_count(clusters: list[Any]) -> int:
    return sum(1 for cluster in clusters if isinstance(cluster, dict) and cluster.get("critical_failure"))


def max_residual(clusters: list[Any]) -> float:
    values = [
        float(cluster.get("max_residual") or 0.0)
        for cluster in clusters
        if isinstance(cluster, dict)
    ]
    return max(values, default=0.0)


def graduation_transfer_artifact_ready(
    transfer_artifacts: dict[str, Any],
    profile_run: dict[str, Any],
    frontier: dict[str, Any],
) -> bool:
    family = active_frontier_family(profile_run, frontier)
    if not family:
        return False
    if family == "coding_local_sandbox":
        return True
    if get_path(transfer_artifacts, ["summary", "frontier_family"], "") not in {"", family}:
        return False
    artifacts = transfer_artifacts.get("artifacts")
    if not isinstance(artifacts, list):
        return False
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        artifact_family = str(artifact.get("family") or "")
        loads_into = artifact.get("loads_into") if isinstance(artifact.get("loads_into"), list) else []
        if artifact_family not in {"", family}:
            continue
        if not artifact_path_exists(artifact):
            continue
        if family == "drone_rl" and "drone_control_arm" not in loads_into:
            continue
        if family == "minecraft_rl" and "minecraft_world_arm" not in loads_into:
            continue
        return True
    return False


def transfer_artifact_evidence(
    transfer_artifacts: dict[str, Any],
    path: str,
    frontier_floor_cleared: bool,
    profile_run: dict[str, Any] | None = None,
    frontier: dict[str, Any] | None = None,
) -> str:
    if not frontier_floor_cleared:
        return "not_required_until_active_frontier_clears_floor"
    family = active_frontier_family(profile_run or {}, frontier or {})
    if family == "coding_local_sandbox":
        return "handled_by_code_frontier_transfer_artifact_ready_and_code_frontier_transfer_consumed"
    summary = transfer_artifacts.get("summary") if isinstance(transfer_artifacts, dict) else {}
    artifacts = transfer_artifacts.get("artifacts") if isinstance(transfer_artifacts, dict) else []
    usable = [
        str(item.get("path"))
        for item in artifacts
        if isinstance(item, dict) and artifact_path_exists(item)
    ] if isinstance(artifacts, list) else []
    return (
        f"path={path} frontier_family={get_path(summary, ['frontier_family'], None)} "
        f"artifacts={len(artifacts) if isinstance(artifacts, list) else 0} "
        f"usable_artifacts={len(usable)}"
    )


def code_frontier_transfer_artifact_ready(
    code_residual_forge: dict[str, Any],
    profile_run: dict[str, Any],
    frontier: dict[str, Any],
) -> bool:
    family = active_frontier_family(profile_run, frontier)
    if family != "coding_local_sandbox":
        return True
    if code_residual_forge.get("policy") != "project_theseus_code_residual_forge_report_v1":
        return False
    if code_residual_forge.get("trigger_state") == "RED":
        return False
    summary = code_residual_forge.get("summary") if isinstance(code_residual_forge.get("summary"), dict) else {}
    if str(summary.get("family") or "") != family:
        return False
    if int(summary.get("transfer_artifacts") or 0) <= 0:
        return False
    artifacts = code_residual_forge.get("transfer_artifacts") if isinstance(code_residual_forge.get("transfer_artifacts"), dict) else {}
    for artifact in artifacts.get("artifacts", []) if isinstance(artifacts.get("artifacts"), list) else []:
        if not isinstance(artifact, dict):
            continue
        if artifact.get("family") != family:
            continue
        loads_into = artifact.get("loads_into") if isinstance(artifact.get("loads_into"), list) else []
        if "code_repair_arm" not in loads_into:
            continue
        if artifact_path_exists(artifact):
            return True
    return False


def code_frontier_transfer_artifact_evidence(
    code_residual_forge: dict[str, Any],
    path: str,
    profile_run: dict[str, Any],
    frontier: dict[str, Any],
) -> str:
    family = active_frontier_family(profile_run, frontier)
    if family != "coding_local_sandbox":
        return "not_coding_frontier"
    summary = code_residual_forge.get("summary") if isinstance(code_residual_forge, dict) else {}
    artifacts = code_residual_forge.get("transfer_artifacts", {}).get("artifacts", []) if isinstance(code_residual_forge, dict) else []
    usable = [
        str(item.get("path"))
        for item in artifacts
        if isinstance(item, dict) and item.get("family") == "coding_local_sandbox" and artifact_path_exists(item)
    ] if isinstance(artifacts, list) else []
    return (
        f"path={path} policy={code_residual_forge.get('policy') if isinstance(code_residual_forge, dict) else None} "
        f"trigger_state={code_residual_forge.get('trigger_state') if isinstance(code_residual_forge, dict) else None} "
        f"active_card_id={summary.get('active_card_id')} "
        f"transfer_artifacts={summary.get('transfer_artifacts')} "
        f"usable_artifacts={len(usable)}"
    )


def code_frontier_transfer_consumed(profile_run: dict[str, Any], frontier: dict[str, Any]) -> bool:
    family = active_frontier_family(profile_run, frontier)
    if family != "coding_local_sandbox":
        return True
    organism = get_path(frontier, ["metrics", "code_repair_organism"], {})
    if not isinstance(organism, dict):
        return False
    return bool(
        organism.get("ran")
        and organism.get("transfer_loaded")
        and organism.get("transfer_consumed")
        and organism.get("transfer_altered_behavior")
        and float(organism.get("pass_rate_delta") or 0.0) > 0.0
    )


def code_frontier_transfer_consumed_evidence(profile_run: dict[str, Any], frontier: dict[str, Any]) -> str:
    family = active_frontier_family(profile_run, frontier)
    if family != "coding_local_sandbox":
        return "not_coding_frontier"
    organism = get_path(frontier, ["metrics", "code_repair_organism"], {})
    if not isinstance(organism, dict):
        return "missing metrics.code_repair_organism"
    return (
        f"ran={organism.get('ran')} loaded={organism.get('transfer_loaded')} "
        f"consumed={organism.get('transfer_consumed')} altered={organism.get('transfer_altered_behavior')} "
        f"baseline={organism.get('baseline_pass_rate')} transfer={organism.get('transfer_pass_rate')} "
        f"delta={organism.get('pass_rate_delta')} report={organism.get('report')}"
    )


def real_code_benchmark_graduation_ready(
    real_code: dict[str, Any],
    profile_run: dict[str, Any],
    frontier: dict[str, Any],
) -> bool:
    family = active_frontier_family(profile_run, frontier)
    if family != "coding_local_sandbox":
        return True
    embedded = get_path(frontier, ["metrics", "real_code_benchmark_graduation"], {})
    if isinstance(embedded, dict) and embedded.get("ran") and embedded_real_code_ready(embedded):
        return embedded_real_code_ready(embedded)
    return canonical_real_code_ready(real_code)


def canonical_real_code_ready(real_code: dict[str, Any]) -> bool:
    if real_code.get("policy") != "project_theseus_real_code_benchmark_graduation_v1":
        return False
    if real_code.get("trigger_state") not in {"GREEN", "YELLOW"}:
        return False
    if real_code.get("candidate_source") not in ALLOWED_REAL_CODE_CANDIDATE_SOURCES:
        return False
    if real_code.get("public_benchmark_score_claim") not in ALLOWED_REAL_CODE_SCORE_CLAIMS:
        return False
    if int(real_code.get("external_inference_calls") or 0) != 0:
        return False
    summary = real_code.get("summary") if isinstance(real_code.get("summary"), dict) else {}
    return bool(
        int(summary.get("public_task_count") or 0) > 0
        and int(summary.get("total_case_count") or 0) > 0
        and float(summary.get("real_public_task_pass_rate") or 0.0) >= CODE_PUBLIC_TASK_FLOOR
        and int(summary.get("student_candidate_count") or 0) > 0
        and bool(summary.get("student_candidate_provenance_valid"))
        and bool(summary.get("student_candidate_benchmark_integrity_valid"))
        and bool(summary.get("token_level_code_generation_learned"))
        and int(summary.get("benchmark_promotion_eligible_candidate_count") or 0) > 0
        and int(summary.get("integrity_verified_candidate_count") or 0) > 0
        and int(summary.get("functional_promotion_count") or 0) > 0
        and int(summary.get("template_like_candidate_count") or 0) == 0
        and int(summary.get("loop_closure_candidate_count") or 0) == 0
        and int(summary.get("task_level_regressions_vs_single_stream") or 0) == 0
    )


def embedded_real_code_ready(embedded: dict[str, Any]) -> bool:
    return bool(
        embedded.get("candidate_source") in ALLOWED_REAL_CODE_CANDIDATE_SOURCES
        and embedded.get("public_benchmark_score_claim") in ALLOWED_REAL_CODE_SCORE_CLAIMS
        and int(embedded.get("external_inference_calls") or 0) == 0
        and int(embedded.get("public_task_count") or 0) > 0
        and int(embedded.get("total_case_count") or 0) > 0
        and float(embedded.get("real_public_task_pass_rate") or 0.0) >= CODE_PUBLIC_TASK_FLOOR
        and int(embedded.get("student_candidate_count") or 0) > 0
        and bool(embedded.get("student_candidate_provenance_valid"))
        and bool(embedded.get("student_candidate_benchmark_integrity_valid"))
        and bool(embedded.get("token_level_code_generation_learned"))
        and int(embedded.get("benchmark_promotion_eligible_candidate_count") or 0) > 0
        and int(embedded.get("integrity_verified_candidate_count") or 0) > 0
        and int(embedded.get("functional_promotion_count") or 0) > 0
        and int(embedded.get("template_like_candidate_count") or 0) == 0
        and int(embedded.get("loop_closure_candidate_count") or 0) == 0
        and int(embedded.get("task_level_regressions_vs_single_stream") or 0) == 0
    )


def broad_public_code_transfer_ready(
    broad: dict[str, Any],
    profile_run: dict[str, Any],
    frontier: dict[str, Any],
) -> bool:
    family = active_frontier_family(profile_run, frontier)
    if family != "coding_local_sandbox":
        return True
    if broad.get("policy") != "project_theseus_broad_transfer_matrix_v1":
        return False
    summary = broad.get("summary") if isinstance(broad.get("summary"), dict) else {}
    min_tasks = int(summary.get("min_public_tasks_per_promotion_card") or 32)
    requested_cards = int(summary.get("requested_card_count") or 0)
    clean_cards = int(summary.get("clean_covered_card_count") or 0)
    total_tasks = int(summary.get("real_public_task_count") or 0)
    min_total_tasks = max(min_tasks * max(2, min(requested_cards or 2, 3)), min_tasks)
    return bool(
        broad.get("trigger_state") == "GREEN"
        and requested_cards > 0
        and clean_cards == requested_cards
        and total_tasks >= min_total_tasks
        and float(summary.get("real_public_pass_rate") or 0.0) >= CODE_PUBLIC_TASK_FLOOR
        and int(summary.get("total_regressions") or 0) == 0
        and int(summary.get("no_cheat_violation_count") or 0) == 0
        and not summary.get("cards_below_floor")
        and not summary.get("no_clean_student_evidence_cards")
        and not summary.get("loader_only_cards")
        and not summary.get("missing_cards")
        and not summary.get("coverage_warning_cards")
    )


def broad_public_code_transfer_evidence(
    broad: dict[str, Any],
    path: str,
    profile_run: dict[str, Any],
    frontier: dict[str, Any],
) -> str:
    family = active_frontier_family(profile_run, frontier)
    if family != "coding_local_sandbox":
        return "not_coding_frontier"
    summary = broad.get("summary") if isinstance(broad.get("summary"), dict) else {}
    min_tasks = int(summary.get("min_public_tasks_per_promotion_card") or 32)
    requested_cards = int(summary.get("requested_card_count") or 0)
    min_total_tasks = max(min_tasks * max(2, min(requested_cards or 2, 3)), min_tasks)
    return (
        f"path={path} policy={broad.get('policy')} trigger_state={broad.get('trigger_state')} "
        f"requested_cards={summary.get('requested_card_count')} clean_cards={summary.get('clean_covered_card_count')} "
        f"real_public_tasks={summary.get('real_public_task_count')} min_total_tasks={min_total_tasks} "
        f"pass_rate={summary.get('real_public_pass_rate')} floor={CODE_PUBLIC_TASK_FLOOR} "
        f"single_stream={summary.get('real_public_single_stream_pass_rate')} sts_delta={summary.get('real_public_sts_delta')} "
        f"below_floor={summary.get('cards_below_floor')} no_clean={summary.get('no_clean_student_evidence_cards')} "
        f"loader_only={summary.get('loader_only_cards')} missing={summary.get('missing_cards')} "
        f"coverage_warnings={summary.get('coverage_warning_cards')} no_cheat_violations={summary.get('no_cheat_violation_count')} "
        f"promotion_candidate_cards={summary.get('promotion_candidate_card_count')}"
    )


def student_learning_closure_ready(
    closure: dict[str, Any],
    real_code: dict[str, Any],
    profile_run: dict[str, Any],
    frontier: dict[str, Any],
) -> bool:
    family = active_frontier_family(profile_run, frontier)
    if family != "coding_local_sandbox":
        return True
    if real_code.get("candidate_source") in {"student_token_generator_checkpoint_v1", "student_code_lm_checkpoint_v1"}:
        summary = real_code.get("summary") if isinstance(real_code.get("summary"), dict) else {}
        expected_claim = (
            "student_code_lm_checkpoint_public_task_calibration_only"
            if real_code.get("candidate_source") == "student_code_lm_checkpoint_v1"
            else "student_token_generator_checkpoint_public_task_calibration_only"
        )
        return bool(
            real_code.get("public_benchmark_score_claim") == expected_claim
            and int(real_code.get("external_inference_calls") or 0) == 0
            and bool(summary.get("student_candidate_benchmark_integrity_valid"))
            and bool(summary.get("token_level_code_generation_learned"))
            and int(summary.get("benchmark_promotion_eligible_candidate_count") or 0) > 0
            and int(summary.get("integrity_verified_candidate_count") or 0) > 0
            and int(summary.get("functional_promotion_count") or 0) > 0
            and int(summary.get("template_like_candidate_count") or 0) == 0
            and int(summary.get("loop_closure_candidate_count") or 0) == 0
        )
    if closure.get("policy") != "project_theseus_student_learning_closure_v1":
        return False
    if closure.get("trigger_state") != "GREEN":
        return False
    if closure.get("candidate_source_after") != "student_neural_checkpoint_v1":
        return False
    if int(closure.get("external_inference_calls") or 0) != 0:
        return False
    summary = closure.get("summary") if isinstance(closure.get("summary"), dict) else {}
    if int(summary.get("learned_candidate_count") or 0) <= 0:
        return False
    if not bool(summary.get("neural_weight_update")):
        return False
    if float(summary.get("weight_delta_l1_from_zero") or 0.0) <= 0.0:
        return False
    if float(summary.get("pass_rate_delta") or 0.0) <= 0.0 and float(summary.get("all_pass_rate_delta") or 0.0) <= 0.0:
        return False
    if not bool(summary.get("token_level_code_generation_learned")):
        return False
    if real_code.get("candidate_source") != "student_neural_checkpoint_v1":
        return False
    if real_code.get("public_benchmark_score_claim") != "student_neural_checkpoint_public_task_calibration_only":
        return False
    return True


def student_learning_closure_evidence(
    closure: dict[str, Any],
    path: str,
    real_code: dict[str, Any],
    profile_run: dict[str, Any],
    frontier: dict[str, Any],
) -> str:
    family = active_frontier_family(profile_run, frontier)
    if family != "coding_local_sandbox":
        return "not_coding_frontier"
    summary = closure.get("summary") if isinstance(closure.get("summary"), dict) else {}
    if real_code.get("candidate_source") in {"student_token_generator_checkpoint_v1", "student_code_lm_checkpoint_v1"}:
        real_summary = real_code.get("summary") if isinstance(real_code.get("summary"), dict) else {}
        return (
            "satisfied_by_token_level_student_generator "
            f"real_code_candidate_source={real_code.get('candidate_source')} "
            f"real_code_score_claim={real_code.get('public_benchmark_score_claim')} "
            f"token_level_code_generation_learned={real_summary.get('token_level_code_generation_learned')} "
            f"eligible_candidates={real_summary.get('benchmark_promotion_eligible_candidate_count')} "
            f"integrity_verified_candidates={real_summary.get('integrity_verified_candidate_count')} "
            f"functional_promotion_count={real_summary.get('functional_promotion_count')} "
            f"functional_promotion_fraction={real_summary.get('functional_promotion_fraction')} "
            f"template_like_candidates={real_summary.get('template_like_candidate_count')} "
            f"loop_closure_candidates={real_summary.get('loop_closure_candidate_count')}"
        )
    return (
        f"path={path} policy={closure.get('policy')} trigger_state={closure.get('trigger_state')} "
        f"candidate_source_after={closure.get('candidate_source_after')} "
        f"before={summary.get('before_eval_pass_rate')} after={summary.get('after_eval_pass_rate')} "
        f"delta={summary.get('pass_rate_delta')} all_delta={summary.get('all_pass_rate_delta')} "
        f"learned_candidates={summary.get('learned_candidate_count')} "
        f"neural_weight_update={summary.get('neural_weight_update')} "
        f"token_level_code_generation_learned={summary.get('token_level_code_generation_learned')} "
        f"real_code_candidate_source={real_code.get('candidate_source')} "
        f"real_code_score_claim={real_code.get('public_benchmark_score_claim')}"
    )


def real_code_benchmark_graduation_evidence(
    real_code: dict[str, Any],
    path: str,
    profile_run: dict[str, Any],
    frontier: dict[str, Any],
) -> str:
    family = active_frontier_family(profile_run, frontier)
    if family != "coding_local_sandbox":
        return "not_coding_frontier"
    embedded = get_path(frontier, ["metrics", "real_code_benchmark_graduation"], {})
    summary = real_code.get("summary") if isinstance(real_code.get("summary"), dict) else {}
    embedded_ready = embedded_real_code_ready(embedded) if isinstance(embedded, dict) and embedded.get("ran") else None
    canonical_ready = canonical_real_code_ready(real_code)
    return (
        f"path={path} policy={real_code.get('policy')} trigger_state={real_code.get('trigger_state')} "
        f"public_tasks={summary.get('public_task_count')} total_cases={summary.get('total_case_count')} "
        f"real_public_task_pass_rate={summary.get('real_public_task_pass_rate')} "
        f"delta={summary.get('pass_rate_delta')} regressions={summary.get('task_level_regressions_vs_single_stream')} "
        f"candidate_source={real_code.get('candidate_source')} score_claim={real_code.get('public_benchmark_score_claim')} "
        f"student_candidates={summary.get('student_candidate_count')} "
        f"student_provenance_valid={summary.get('student_candidate_provenance_valid')} "
        f"benchmark_integrity_valid={summary.get('student_candidate_benchmark_integrity_valid')} "
        f"token_level_code_generation_learned={summary.get('token_level_code_generation_learned')} "
        f"template_like_candidates={summary.get('template_like_candidate_count')} "
        f"loop_closure_candidates={summary.get('loop_closure_candidate_count')} "
        f"eligible_candidates={summary.get('benchmark_promotion_eligible_candidate_count')} "
        f"integrity_verified_candidates={summary.get('integrity_verified_candidate_count')} "
        f"functional_promotion_count={summary.get('functional_promotion_count')} "
        f"functional_promotion_fraction={summary.get('functional_promotion_fraction')} "
        f"embedded_ran={embedded.get('ran') if isinstance(embedded, dict) else None} "
        f"embedded_ready={embedded_ready} canonical_ready={canonical_ready} "
        f"embedded_public_tasks={embedded.get('public_task_count') if isinstance(embedded, dict) else None} "
        f"embedded_real_public_task_pass_rate={embedded.get('real_public_task_pass_rate') if isinstance(embedded, dict) else None} "
        f"required_public_task_floor={CODE_PUBLIC_TASK_FLOOR}"
    )


def artifact_path_exists(artifact: dict[str, Any]) -> bool:
    value = str(artifact.get("path") or "")
    if not value:
        return False
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.exists()


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_real_code_graduation_report(requested: str) -> tuple[dict[str, Any], str]:
    """Use the broadest clean public report when the caller asks for canonical.

    The canonical report is often a smoke run. Explicit paths remain explicit,
    but the default gate path should not ignore a wider clean calibration.
    """
    requested_path = Path(requested)
    requested_payload = read_json(requested_path)
    if requested.replace("\\", "/") != DEFAULT_REAL_CODE_GRADUATION:
        return requested_payload, requested
    candidates: list[tuple[int, int, str, str, dict[str, Any]]] = []
    for path in (ROOT / "reports").glob("real_code_benchmark_graduation*.json"):
        payload = read_json(path)
        if not canonical_real_code_ready(payload):
            continue
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        candidates.append(
            (
                int(summary.get("public_task_count") or 0),
                int(summary.get("total_case_count") or 0),
                str(payload.get("created_utc") or ""),
                display_path(path),
                payload,
            )
        )
    if not candidates:
        return requested_payload, requested
    candidates.sort(reverse=True)
    _, _, _, selected_path, selected_payload = candidates[0]
    return selected_payload, selected_path


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
