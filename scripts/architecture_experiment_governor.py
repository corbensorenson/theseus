"""Select minimal architecture experiments from current ratchet evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPACE = ROOT / "configs" / "architecture_search_space.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default=str(DEFAULT_SPACE.relative_to(ROOT)))
    parser.add_argument("--out", default="reports/architecture_experiment_governance.json")
    parser.add_argument("--markdown-out", default="reports/architecture_experiment_governance.md")
    args = parser.parse_args()

    space = read_json(ROOT / args.space)
    state = load_state()
    experiments = rank_experiments(space, state)
    recommendation = next(
        (
            row
            for row in experiments
            if row.get("status") in {"ready_for_smoke_or_profile", "evidence_recorded_ready"}
        ),
        experiments[0] if experiments else {},
    )
    report = {
        "policy": "sparkstream_architecture_experiment_governance_v0",
        "created_utc": now(),
        "space": args.space,
        "small_model_principle": space.get("objective"),
        "state": summarize_state(state),
        "global_gates": evaluate_global_gates(space, state),
        "experiments": experiments,
        "recommended_next_experiment": recommendation,
        "architecture_change_allowed": architecture_change_allowed(state),
        "teacher_role": "Teacher may implement the smallest patch only through the guarded self-edit lane after this report identifies a wall.",
        "external_inference_calls": 0,
    }
    write_json(ROOT / args.out, report)
    write_markdown(ROOT / args.markdown_out, report)
    print(json.dumps(report, indent=2))
    return 0


def load_state() -> dict[str, Any]:
    reports = ROOT / "reports"
    return {
        "candidate": read_json(reports / "candidate_promotion_gate.json"),
        "frontier_policy": read_json(reports / "frontier_policy_status.json"),
        "benchmaxx_curriculum": read_json(reports / "benchmaxx_curriculum.json"),
        "benchmark_ledger": read_json(reports / "benchmark_ledger.json"),
        "residual_escrow": read_json(reports / "residual_escrow.json"),
        "resource_governor": read_json(reports / "resource_governor.json"),
        "architecture_gate": read_json(reports / "architecture_gate_report.json"),
        "ablation_matrix": read_json(reports / "ablation_matrix_rtx2060super_report.json"),
        "synthetic": read_json(reports / "synthetic_data_curator.json"),
        "synthetic_benchmark_factory": read_json(reports / "synthetic_benchmark_factory.json"),
        "multi_stream_trace_factory": read_json(reports / "multi_stream_trace_factory.json"),
        "multi_stream_code_pressure": read_latest_json(reports, "multi_stream_code_pressure_*_seed*.json"),
        "multi_stream_monitorability_probe": read_json(reports / "multi_stream_monitorability_probe.json"),
        "multi_stream_candidate_gate": read_json(reports / "multi_stream_candidate_gate.json"),
        "external_inference_audit": read_json(reports / "external_inference_audit.json"),
        "arm_lifecycle": read_json(reports / "arm_lifecycle_governance.json"),
        "attd": read_json(reports / "attd_report.json"),
        "attd_checkpoint": read_json(reports / "attd_dirty_workspace_checkpoint.json"),
        "minecraft_runtime": read_json(reports / "minecraft_runtime_probe.json"),
        "token_superposition": read_json(reports / "token_superposition_training.json"),
        "code_residual_forge": read_json(reports / "code_residual_forge.json"),
        "code_repair_organism": read_latest_json(reports, "local_code_repair_organism_*_seed*.json"),
        "self_edit_lane": read_json(reports / "self_edit_experiment_lane.json"),
        "long_horizon_memory": read_json(reports / "long_horizon_memory_probe.json"),
        "virtual_context_memory": read_json(reports / "virtual_context_memory_probe.json"),
        "virtual_context_memory_status": read_json(reports / "virtual_context_memory_status.json"),
        "virtual_context_memory_consumer_audit": read_json(reports / "virtual_context_memory_consumer_audit.json"),
        "genesis": read_json(reports / "genesis_kernel" / "report.json"),
        "causal_architecture_delta_loop": read_json(reports / "causal_architecture_delta_loop.json"),
        "broad_survival_promotion": read_json(reports / "broad_capability_survival_promotion_gate_v1.json"),
        "private_public_transfer": read_json(reports / "private_public_transfer_proof.json"),
        "broad_transfer": read_json(reports / "broad_transfer_matrix.json"),
        "transfer_generalization": read_json(reports / "transfer_generalization_audit.json"),
    }


def rank_experiments(space: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    priority_rank = {"highest": 0, "high": 1, "medium": 2, "low": 3}
    rows: list[dict[str, Any]] = []
    ladder = space.get("intervention_ladder", [])
    failed = set(failed_gates(state.get("candidate") or {}))
    frontier = active_frontier(state)
    wall = frontier.get("wall_type") or ""
    family = frontier_family(frontier)
    residual_delta = get_path(state, ["candidate", "residual_delta"], {})
    for item in space.get("experiment_families", []):
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind"))
        score = 100 - (priority_rank.get(str(item.get("priority")), 9) * 10)
        reasons: list[str] = []
        if kind.startswith("zero_param"):
            score += 15
            reasons.append("zero-param or tooling first")
        if "candidate_profile_evidence_complete" in failed and item.get("id") in {
            "residual_bridge_data",
            "benchmark_adapter_factory",
        }:
            score += 10
            reasons.append("candidate evidence still incomplete")
        if wall in {"architecture_training_wall", "state_or_rollout_wall", "policy_selected_frontier"}:
            if item.get("id") in {"learned_sequence_state_probe", "cuda_hot_loop_ownership"}:
                score += 12
                reasons.append(f"active wall={wall}")
            if item.get("id") == "drone_controller_transfer_adapter" and family == "drone_rl":
                score += 45
                reasons.append("active drone-control wall")
            if family == "minecraft_rl" and item.get("id") in {"loop_closure_tooling", "benchmark_adapter_factory", "learned_sequence_state_probe"}:
                score += 20
                reasons.append("active Minecraft/Open-World wall")
            if kind == "minimal_architecture_change":
                score += 4
                reasons.append("architecture wall present but lower-cost experiments still preferred")
        if family in {"babylm_mutated", "language"} and get_path(state, ["synthetic", "training_ready"], False) and item.get("id") == "residual_bridge_data":
            score += 8
            reasons.append("synthetic residual bridge is already governed")
        if get_path(state, ["resource_governor", "decision", "can_run_requested_profile"], True):
            score += 4
            reasons.append("resource governor allows current profile")
        if item.get("id") == "token_superposition_training":
            tst_status = get_path(state, ["token_superposition", "promotion_decision", "status"], "")
            if family in {"coding_local_sandbox", "babylm_mutated", "language"}:
                score += 34
                reasons.append("active token/code frontier can use a Rust/CUDA training-efficiency test")
            if tst_status != "eligible_for_training_lane":
                score += 8
                reasons.append("TST has not yet produced passing Rust/CUDA evidence")
        if item.get("id") == "code_residual_forge":
            if family == "coding_local_sandbox":
                score += 55
                reasons.append("active coding frontier must export residual traces and transfer artifacts")
            if code_residual_forge_ready(state):
                score += 6
                reasons.append("Code Residual Forge evidence is already recorded")
        if item.get("id") == "local_code_repair_organism":
            if family == "coding_local_sandbox":
                score += 60
                reasons.append("active coding frontier needs a real repair loop and transfer heredity proof")
            if code_repair_organism_ready(state):
                score += 8
                reasons.append("local code repair heredity evidence is already recorded")
        if item.get("id") == "self_edit_experiment_lane":
            score += 44
            reasons.append("residual clusters should become bounded source-patch experiments before growth")
            if self_edit_lane_ready(state):
                score += 8
                reasons.append("self-edit experiment lane evidence is already recorded")
        if item.get("id") == "long_horizon_memory_probe":
            score += 36
            reasons.append("long-horizon recovery must be proven before broad autonomy")
            if long_horizon_memory_ready(state):
                score += 8
                reasons.append("memory probe evidence is already recorded")
            if virtual_context_memory_ready(state):
                score += 4
                reasons.append("virtual context memory substrate is green")
        if item.get("id") == "genesis_artifact_substrate":
            if genesis_ready(state):
                score += 6
                reasons.append("Genesis artifact substrate evidence is already recorded")
            else:
                score += 50
                reasons.append("live invention artifacts are not yet compiled for governance")
        if item.get("id") == "synthetic_benchmark_factory":
            if synthetic_benchmark_factory_ready(state):
                score += 8
                reasons.append("synthetic benchmark backstop evidence is already recorded")
            else:
                score += 52
                reasons.append("fresh local benchmark pressure should be generated before model growth")
        if item.get("id") == "multi_stream_code_pressure":
            if family == "coding_local_sandbox":
                score += 66
                reasons.append("active coding frontier can test solver/test/critic/patch streams against single-stream repair")
            if multi_stream_code_pressure_ready(state):
                score += 8
                reasons.append("multi-stream code pressure evidence is already recorded")
            else:
                score += 42
                reasons.append("parallel coding streams should be proven before substrate/model growth")
        if isinstance(residual_delta, dict) and residual_delta.get("max_residual_delta", 0) > 0:
            if item.get("id") in {"residual_bridge_data", "loop_closure_tooling"}:
                score += 8
                reasons.append("residual delta worsened")
        row = {
            **item,
            "rank_score": score,
            "why_now": reasons or ["kept in queue for future ladder step"],
            "status": status_for_experiment(item, state),
            "teacher_needed": bool(kind == "minimal_architecture_change" or "architecture" in str(item.get("id"))),
        }
        row.setdefault("private_eval_plan", private_eval_plan_for(item))
        row.setdefault("rollback_rule", rollback_rule_for(item))
        rows.append(row)
    attd_delta = attd_monolith_delta_experiment(state)
    if attd_delta:
        rows.append(attd_delta)
    causal_delta = causal_public_code_transfer_delta_experiment(state)
    if causal_delta:
        rows.append(causal_delta)
    rows.sort(key=lambda row: (-float(row.get("rank_score", 0)), str(row.get("id"))))
    return rows


def causal_public_code_transfer_delta_experiment(state: dict[str, Any]) -> dict[str, Any] | None:
    broad_summary = get_path(state, ["broad_transfer", "summary"], {})
    transfer_summary = get_path(state, ["transfer_generalization", "summary"], {})
    weak_cards = transfer_summary.get("weak_cards") if isinstance(transfer_summary, dict) else None
    if not isinstance(weak_cards, list):
        weak_cards = broad_summary.get("cards_below_floor") if isinstance(broad_summary, dict) else []
    broad_rate = number(broad_summary.get("real_public_pass_rate") if isinstance(broad_summary, dict) else None, default=0.0)
    transfer_ready = bool(transfer_summary.get("transfer_ready")) if isinstance(transfer_summary, dict) else False
    causal = state.get("causal_architecture_delta_loop") if isinstance(state.get("causal_architecture_delta_loop"), dict) else {}
    summary = causal.get("summary") if isinstance(causal.get("summary"), dict) else {}
    evidence_ready = bool(
        causal.get("policy") == "project_theseus_causal_architecture_delta_loop_v1"
        and causal.get("status") == "completed_with_capability_delta"
        and causal.get("promotion_evidence")
        and float(summary.get("best_target_delta") or 0.0) >= 0.01
    )
    if not weak_cards and transfer_ready and not evidence_ready:
        return None
    command = "python scripts/causal_architecture_delta_loop.py"
    if not evidence_ready:
        command += " --execute-ablation"
    weak_text = ",".join(str(card) for card in weak_cards[:4]) if isinstance(weak_cards, list) else "unknown"
    return {
        "id": "causal_public_code_transfer_router_delta",
        "kind": "bounded_architecture_patch_ablation",
        "priority": "highest",
        "profile": "private_same_seed_ablation",
        "hypothesis": (
            "Public code-transfer residuals should be converted into a bounded source-router "
            "patch, same-seed private heldout A/B, and explicit promote/rollback decision."
        ),
        "commands": [command],
        "promotion_gates": [
            "residual_cluster_selected",
            "private_same_seed_ablation_usable",
            "targeted_private_heldout_delta_ge_0_01",
            "decoder_gate_remains_green",
            "private_public_transfer_proof_remains_green",
            "no_public_tests_or_solutions_used",
        ],
        "rank_score": 260,
        "why_now": [
            "architecture experiment delta is the current compounding blocker",
            f"broad public pass rate={broad_rate}",
            f"weak cards={weak_text}",
        ],
        "status": "evidence_recorded_ready" if evidence_ready else "ready_for_smoke_or_profile",
        "teacher_needed": False,
        "residual_cluster": summary.get("residual_cluster")
        or f"public_code_transfer:admissibility_and_interface:weak_cards={weak_text or 'unknown'}",
        "delta_metric": "private_heldout_pass_rate_delta",
        "private_eval_plan": {
            "type": "same_seed_private_heldout_ablation",
            "commands": [command],
            "success_requires": [
                "targeted_private_heldout_delta_ge_0_01",
                "private_only_no_public_rows",
                "decoder_gate_remains_green",
                "private_public_transfer_proof_remains_green",
            ],
            "public_benchmark_use": "none",
            "external_inference_calls": 0,
        },
        "rollback_rule": (
            "Demote/rollback the bounded router patch if same-seed private heldout delta is below "
            "0.01, no-admissible regresses, decoder/transfer gates lose GREEN, or public tests/solutions "
            "enter the run."
        ),
    }


def attd_monolith_delta_experiment(state: dict[str, Any]) -> dict[str, Any] | None:
    checkpoint = state.get("attd_checkpoint") if isinstance(state.get("attd_checkpoint"), dict) else {}
    before = checkpoint.get("attd_before") if isinstance(checkpoint.get("attd_before"), dict) else {}
    attd = state.get("attd") if isinstance(state.get("attd"), dict) else {}
    before_violations = before.get("hard_cap_violations") if isinstance(before.get("hard_cap_violations"), list) else []
    hard_caps = attd.get("hard_caps") if isinstance(attd.get("hard_caps"), dict) else {}
    git_baseline_lines = git_blob_line_count("crates/symliquid-cli/src/code_lm_closure.rs")
    prior_monolith_blocked = "max_source_file_lines" in before_violations or git_baseline_lines > 7000
    if not prior_monolith_blocked:
        return None
    if not bool(hard_caps.get("passed")):
        return None
    item = {
        "id": "attd_monolith_decomposition",
        "kind": "zero_param",
        "priority": "highest",
        "profile": "smoke",
        "hypothesis": "Splitting the Code LM closure monolith into bounded Rust module chunks removes the ATTD hard-cap blocker without changing decoder behavior.",
        "commands": ["python scripts/attd_analyzer.py --out reports/attd_report.json"],
        "promotion_gates": [
            "attd_hard_caps_pass",
            "max_source_file_lines_removed",
            "rust_compile_passed",
            "no_public_benchmark_training_data",
            "external_inference_zero",
        ],
        "rank_score": 220,
        "why_now": [
            "ATTD checkpoint recorded max_source_file_lines as a RED blocker",
            f"git baseline code_lm_closure.rs lines={git_baseline_lines}",
            "current ATTD hard caps now pass, so the architecture experiment needs delta evidence",
        ],
        "status": "ready_for_smoke_or_profile",
        "teacher_needed": False,
        "private_eval_plan": {
            "type": "architecture_hygiene_delta_check",
            "commands": [
                "cargo check -p symliquid-cli",
                "python scripts/attd_analyzer.py --out reports/attd_report.json",
            ],
            "success_requires": [
                "rust_compile_passed",
                "attd_hard_caps_pass",
                "max_source_file_lines_removed",
            ],
            "public_benchmark_use": "none",
            "external_inference_calls": 0,
        },
        "rollback_rule": (
            "If the Rust compile check fails or ATTD hard caps regress, preserve reports as residual evidence and "
            "restore only the code_lm_closure module split paths after human review; do not reset unrelated work."
        ),
    }
    return item


def git_blob_line_count(path: str) -> int:
    try:
        result = subprocess.run(
            ["git", "show", f"HEAD:{path}"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=20,
        )
    except Exception:
        return 0
    if result.returncode != 0:
        return 0
    return len(result.stdout.splitlines())


def private_eval_plan_for(item: dict[str, Any]) -> dict[str, Any]:
    commands = item.get("commands") if isinstance(item.get("commands"), list) else []
    safe_commands = [str(cmd) for cmd in commands[:2]]
    return {
        "type": "bounded_local_smoke_or_profile",
        "commands": safe_commands,
        "success_requires": item.get("promotion_gates") or [],
        "public_benchmark_use": "calibration_only_after_private_gate",
        "external_inference_calls": 0,
    }


def rollback_rule_for(item: dict[str, Any]) -> str:
    paths = item.get("target_files") or item.get("paths") or []
    if not isinstance(paths, list):
        paths = []
    safe_paths = ", ".join(str(path) for path in paths[:8]) or "declared lane-owned files only"
    return (
        "If private eval is flat, regressive, or fails verification, do not promote the experiment; "
        f"preserve the report as residual evidence and restore only lane-owned changed paths ({safe_paths}) after human review. "
        "Never reset the full workspace or revert unrelated user/generated changes."
    )


def status_for_experiment(item: dict[str, Any], state: dict[str, Any]) -> str:
    kind = str(item.get("kind"))
    attd_state = get_path(state, ["attd", "trigger_state"], "MISSING")
    if item.get("id") == "genesis_artifact_substrate":
        if genesis_ready(state):
            return "evidence_recorded_ready"
        return "ready_for_smoke_or_profile"
    if item.get("id") == "synthetic_benchmark_factory":
        if synthetic_benchmark_factory_ready(state):
            return "evidence_recorded_ready"
        return "ready_for_smoke_or_profile"
    if item.get("id") == "multi_stream_code_pressure":
        if multi_stream_code_pressure_ready(state):
            return "evidence_recorded_ready"
        return "ready_for_smoke_or_profile"
    if item.get("id") == "token_superposition_training":
        report = state.get("token_superposition") if isinstance(state.get("token_superposition"), dict) else {}
        if report.get("policy") == "project_theseus_token_superposition_rust_cuda_report_v1":
            status = get_path(report, ["promotion_decision", "status"], "")
            if status == "not_promoted_keep_as_evidence":
                return "evidence_recorded_failed_gates"
    if item.get("id") == "code_residual_forge":
        if code_residual_forge_ready(state):
            return "evidence_recorded_ready"
        return "ready_for_smoke_or_profile"
    if item.get("id") == "local_code_repair_organism":
        if code_repair_organism_ready(state):
            return "evidence_recorded_ready"
        return "ready_for_smoke_or_profile"
    if item.get("id") == "self_edit_experiment_lane":
        if self_edit_lane_ready(state):
            return "evidence_recorded_ready"
        return "ready_for_smoke_or_profile"
    if item.get("id") == "long_horizon_memory_probe":
        if long_horizon_memory_ready(state):
            return "evidence_recorded_ready"
        return "ready_for_smoke_or_profile"
    if attd_state in {"MISSING", "RED"}:
        return "blocked_by_attd_maintenance"
    if kind == "minimal_architecture_change" and not architecture_change_allowed(state):
        return "queued_until_ladder_reaches_architecture"
    if item.get("profile") == "candidate" and not get_path(
        state, ["resource_governor", "decision", "can_run_requested_profile"], True
    ):
        return "blocked_by_resource_governor"
    return "ready_for_smoke_or_profile"


def architecture_change_allowed(state: dict[str, Any]) -> bool:
    frontier = active_frontier(state)
    failed = set(failed_gates(state.get("candidate") or {}))
    architecture_wall = frontier.get("wall_type") in {
        "architecture_training_wall",
        "state_or_rollout_wall",
        "evaluation_frontier_wall",
    }
    seed_regression_ok = "seed49_regression_holds" not in failed
    public_transfer_is_wall = public_transfer_wall(state, failed)
    regression_ok = not {"public_comparator_no_regression"} & failed
    audit_ok = bool(get_path(state, ["external_inference_audit", "ok"], True))
    attd_ok = bool(get_path(state, ["attd", "governance", "allows_architecture_change"], False))
    if public_transfer_is_wall:
        return bool(audit_ok and attd_ok and seed_regression_ok and bounded_architecture_growth_evidence(state))
    return bool(architecture_wall and regression_ok and seed_regression_ok and audit_ok and attd_ok)


def public_transfer_wall(state: dict[str, Any], failed: set[str]) -> bool:
    broad_transfer_rate = get_path(state, ["broad_transfer", "summary", "real_public_pass_rate"], None)
    broad_transfer_wall = broad_transfer_rate is not None and float(broad_transfer_rate or 0.0) < 0.70
    private_transfer_ready = bool(get_path(state, ["private_public_transfer", "summary", "ready_for_public_calibration"], False))
    public_comparator_wall = "public_comparator_no_regression" in failed
    return bool(broad_transfer_wall or public_comparator_wall or not private_transfer_ready)


def bounded_architecture_growth_evidence(state: dict[str, Any]) -> bool:
    causal = state.get("causal_architecture_delta_loop") if isinstance(state.get("causal_architecture_delta_loop"), dict) else {}
    causal_summary = causal.get("summary") if isinstance(causal.get("summary"), dict) else {}
    broad = state.get("broad_survival_promotion") if isinstance(state.get("broad_survival_promotion"), dict) else {}
    broad_no_cheat = broad.get("no_cheat") if isinstance(broad.get("no_cheat"), dict) else {}
    causal_ok = bool(
        causal.get("trigger_state") == "GREEN"
        and causal.get("promotion_evidence") is True
        and causal_summary.get("status") == "completed_with_capability_delta"
        and int(causal_summary.get("public_task_count") or 0) == 0
        and not bool(causal_summary.get("public_tests_or_solutions_used"))
        and int(causal.get("external_inference_calls") or 0) == 0
    )
    broad_ok = bool(
        broad.get("trigger_state") == "GREEN"
        and broad.get("promoted") is True
        and int(broad.get("external_inference_calls") or 0) == 0
        and int(broad_no_cheat.get("public_training_rows") or 0) == 0
        and int(broad_no_cheat.get("fallback_return_rows") or 0) == 0
        and not bool(broad_no_cheat.get("teacher_used"))
        and not bool(get_path(broad, ["active_manifest", "serving_allowed"], False))
    )
    return bool(causal_ok or broad_ok)


def evaluate_global_gates(space: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    failed = set(failed_gates(state.get("candidate") or {}))
    checks = []
    checks.append({"gate": "external_inference_audit_ok", "passed": bool(get_path(state, ["external_inference_audit", "ok"], True))})
    checks.append({"gate": "architecture_gate_green", "passed": bool(get_path(state, ["architecture_gate", "ready_for_heavy_training"], True))})
    checks.append({"gate": "public_comparator_no_regression", "passed": "public_comparator_no_regression" not in failed})
    checks.append({"gate": "seed49_regression_holds", "passed": "seed49_regression_holds" not in failed})
    checks.append({"gate": "resource_cost_reported", "passed": bool(state.get("resource_governor"))})
    checks.append({"gate": "residual_delta_bounded", "passed": not ({"active_diagnostic_delta_bounded", "max_residual_delta_bounded"} & failed)})
    checks.append({"gate": "bounded_architecture_growth_evidence", "passed": bounded_architecture_growth_evidence(state)})
    checks.append({"gate": "architecture_change_allowed", "passed": architecture_change_allowed(state)})
    checks.append({"gate": "attd_report_available", "passed": get_path(state, ["attd", "policy"], "") == "sparkstream_attd_report_v0"})
    checks.append({"gate": "attd_not_red", "passed": get_path(state, ["attd", "trigger_state"], "MISSING") in {"GREEN", "YELLOW"}})
    checks.append({"gate": "code_residual_forge_ready", "passed": code_residual_forge_ready(state)})
    checks.append({"gate": "local_code_repair_organism_ready", "passed": code_repair_organism_ready(state)})
    checks.append({"gate": "self_edit_experiment_lane_ready", "passed": self_edit_lane_ready(state)})
    checks.append({"gate": "long_horizon_memory_probe_ready", "passed": long_horizon_memory_ready(state)})
    checks.append({"gate": "virtual_context_memory_ready", "passed": virtual_context_memory_ready(state)})
    checks.append({"gate": "genesis_artifact_substrate_ready", "passed": genesis_ready(state)})
    checks.append({"gate": "synthetic_benchmark_factory_ready", "passed": synthetic_benchmark_factory_ready(state)})
    checks.append({"gate": "multi_stream_code_pressure_ready", "passed": multi_stream_code_pressure_ready(state)})
    return checks


def summarize_state(state: dict[str, Any]) -> dict[str, Any]:
    candidate = state.get("candidate") or {}
    frontier = active_frontier(state)
    return {
        "candidate_promote": candidate.get("promote"),
        "candidate_passed": f"{candidate.get('passed')}/{candidate.get('total')}",
        "failed_gates": failed_gates(candidate),
        "active_frontier": frontier.get("benchmark_name"),
        "active_frontier_source": frontier.get("frontier_source"),
        "pressure_card_id": frontier.get("pressure_card_id"),
        "frontier_family": frontier_family(frontier),
        "frontier_score": frontier.get("score"),
        "frontier_wall_type": frontier.get("wall_type"),
        "residual_delta": candidate.get("residual_delta"),
        "resource_can_run": get_path(state, ["resource_governor", "decision", "can_run_requested_profile"], None),
        "arm_split_candidates": get_path(state, ["arm_lifecycle", "summary", "split_candidates"], None),
        "attd_trigger_state": get_path(state, ["attd", "trigger_state"], None),
        "attd_score": get_path(state, ["attd", "attd_score"], None),
        "genesis_trigger_state": get_path(state, ["genesis", "summary", "trigger_state"], None),
        "genesis_artifact_count": get_path(state, ["genesis", "summary", "artifact_count"], None),
        "synthetic_benchmark_factory_trigger_state": get_path(state, ["synthetic_benchmark_factory", "trigger_state"], None),
        "synthetic_benchmark_cards": get_path(state, ["synthetic_benchmark_factory", "summary", "cards"], None),
        "synthetic_benchmark_cases": get_path(state, ["synthetic_benchmark_factory", "summary", "case_count"], None),
        "multi_stream_factory_trigger_state": get_path(state, ["multi_stream_trace_factory", "trigger_state"], None),
        "multi_stream_code_pressure_score": get_path(state, ["multi_stream_code_pressure", "score"], None),
        "multi_stream_code_pressure_delta": get_path(state, ["multi_stream_code_pressure", "summary", "pass_rate_delta"], None),
        "multi_stream_monitorability_score": get_path(state, ["multi_stream_monitorability_probe", "summary", "monitorability_score"], None),
        "code_residual_forge_trigger_state": get_path(state, ["code_residual_forge", "trigger_state"], None),
        "code_residual_forge_clusters": get_path(state, ["code_residual_forge", "summary", "cluster_count"], None),
        "code_residual_forge_rotation": get_path(state, ["code_residual_forge", "summary", "rotation_decision"], None),
        "code_repair_organism_delta": get_path(state, ["code_repair_organism", "summary", "pass_rate_delta"], None),
        "self_edit_lane_trigger_state": get_path(state, ["self_edit_lane", "trigger_state"], None),
        "long_horizon_memory_score": get_path(state, ["long_horizon_memory", "score", "overall"], None),
        "virtual_context_memory_state": get_path(state, ["virtual_context_memory", "trigger_state"], None),
        "virtual_context_memory_pages": get_path(state, ["virtual_context_memory", "summary", "semantic_pages"], None),
        "virtual_context_memory_faults": get_path(state, ["virtual_context_memory_status", "summary", "fault_count"], None),
        "virtual_context_memory_consumer_audit": get_path(state, ["virtual_context_memory_consumer_audit", "trigger_state"], None),
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


def code_residual_forge_ready(state: dict[str, Any]) -> bool:
    report = state.get("code_residual_forge") if isinstance(state.get("code_residual_forge"), dict) else {}
    if report.get("policy") != "project_theseus_code_residual_forge_report_v1":
        return False
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return bool(
        report.get("trigger_state") != "RED"
        and int(summary.get("cluster_count") or 0) > 0
        and int(summary.get("transfer_artifacts") or 0) > 0
    )


def synthetic_benchmark_factory_ready(state: dict[str, Any]) -> bool:
    report = state.get("synthetic_benchmark_factory") if isinstance(state.get("synthetic_benchmark_factory"), dict) else {}
    if report.get("policy") != "project_theseus_synthetic_benchmark_factory_v1":
        return False
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return bool(
        report.get("trigger_state") == "GREEN"
        and int(summary.get("ready_cards") or 0) > 0
        and int(summary.get("case_count") or 0) > 0
        and int(report.get("external_inference_calls") or 0) == 0
    )


def multi_stream_code_pressure_ready(state: dict[str, Any]) -> bool:
    factory = state.get("multi_stream_trace_factory") if isinstance(state.get("multi_stream_trace_factory"), dict) else {}
    pressure = state.get("multi_stream_code_pressure") if isinstance(state.get("multi_stream_code_pressure"), dict) else {}
    probe = state.get("multi_stream_monitorability_probe") if isinstance(state.get("multi_stream_monitorability_probe"), dict) else {}
    gate_report = state.get("multi_stream_candidate_gate") if isinstance(state.get("multi_stream_candidate_gate"), dict) else {}
    return bool(
        factory.get("policy") == "project_theseus_multi_stream_trace_factory_v1"
        and factory.get("trigger_state") == "GREEN"
        and pressure.get("policy") == "project_theseus_multi_stream_code_pressure_v1"
        and get_path(pressure, ["verifier", "trigger_state"], "") == "GREEN"
        and float(get_path(pressure, ["summary", "apples_to_apples_overlap"], 0.0) or 0.0) >= 1.0
        and probe.get("trigger_state") == "GREEN"
        and gate_report.get("trigger_state") == "GREEN"
        and int(pressure.get("external_inference_calls") or 0) == 0
    )


def code_repair_organism_ready(state: dict[str, Any]) -> bool:
    report = state.get("code_repair_organism") if isinstance(state.get("code_repair_organism"), dict) else {}
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return bool(
        report.get("policy") == "project_theseus_local_code_repair_organism_v1"
        and summary.get("transfer_loaded")
        and summary.get("transfer_altered_behavior")
        and float(summary.get("pass_rate_delta") or 0.0) > 0.0
    )


def self_edit_lane_ready(state: dict[str, Any]) -> bool:
    report = state.get("self_edit_lane") if isinstance(state.get("self_edit_lane"), dict) else {}
    return bool(
        report.get("policy") == "project_theseus_self_edit_experiment_lane_v1"
        and report.get("trigger_state") in {"GREEN", "YELLOW"}
        and any(row.get("gate") == "residual_to_patch_contracts_written" and row.get("passed") for row in report.get("gates", []) if isinstance(row, dict))
    )


def long_horizon_memory_ready(state: dict[str, Any]) -> bool:
    report = state.get("long_horizon_memory") if isinstance(state.get("long_horizon_memory"), dict) else {}
    return bool(
        report.get("policy") == "project_theseus_long_horizon_memory_probe_v1"
        and report.get("trigger_state") == "GREEN"
        and float(get_path(report, ["score", "overall"], 0.0) or 0.0) >= 0.90
    )


def virtual_context_memory_ready(state: dict[str, Any]) -> bool:
    report = state.get("virtual_context_memory") if isinstance(state.get("virtual_context_memory"), dict) else {}
    status = state.get("virtual_context_memory_status") if isinstance(state.get("virtual_context_memory_status"), dict) else {}
    consumer_audit = state.get("virtual_context_memory_consumer_audit") if isinstance(state.get("virtual_context_memory_consumer_audit"), dict) else {}
    return bool(
        report.get("trigger_state") == "GREEN"
        and get_path(status, ["summary", "vcm_bench_state"], "") == "GREEN"
        and consumer_audit.get("trigger_state") == "GREEN"
    )


def active_frontier(state: dict[str, Any]) -> dict[str, Any]:
    ledger = state.get("benchmark_ledger")
    frontiers = [row for row in ledger if isinstance(row, dict) and row.get("lifecycle") == "frontier"] if isinstance(ledger, list) else []
    preferred_family, pressure_card_id, source = effective_frontier_selection(state)
    if frontiers and (preferred_family or pressure_card_id):
        matches = [
            row
            for row in frontiers
            if policy_matches_frontier(row, preferred_family, pressure_card_id)
        ]
        if matches:
            selected = max(matches, key=lambda row: float(row.get("residual") or 0.0))
            return {
                **selected,
                "frontier_source": source,
                "pressure_card_id": pressure_card_id or None,
            }
    if pressure_card_id:
        return {
            "benchmark_name": f"{preferred_family or 'pressure'}_{pressure_card_id}",
            "benchmark_type": f"frontier_{preferred_family or 'pressure'}",
            "lifecycle": "frontier",
            "score": get_path(state, ["candidate", "scores", "active_frontier_accuracy"], None),
            "residual": None,
            "wall_type": "policy_selected_frontier",
            "frontier_source": f"{source}_synthetic",
            "pressure_card_id": pressure_card_id,
        }
    if not frontiers:
        return {}
    architecture_frontiers = [
        row
        for row in frontiers
        if row.get("wall_type")
        in {"architecture_training_wall", "state_or_rollout_wall", "evaluation_frontier_wall"}
        and below_floor(row)
    ]
    if architecture_frontiers:
        return max(architecture_frontiers, key=lambda row: float(row.get("residual") or 0.0))
    return max(frontiers, key=lambda row: float(row.get("residual") or 0.0))


def effective_frontier_selection(state: dict[str, Any]) -> tuple[str, str, str]:
    curriculum = state.get("benchmaxx_curriculum") if isinstance(state.get("benchmaxx_curriculum"), dict) else {}
    next_frontier = curriculum.get("next_frontier") if isinstance(curriculum.get("next_frontier"), dict) else {}
    runner = str(next_frontier.get("runner_family") or "")
    family = str(next_frontier.get("family") or "")
    runner_map = {
        "minecraft_rl_local": "minecraft_rl",
        "drone_rl_local": "drone_rl",
        "coding_local_sandbox": "coding_local_sandbox",
        "web_agent_local": "web_agent_local",
        "transfer_eval_local": "transfer_eval",
    }
    mapped = runner_map.get(runner, family)
    card = str(next_frontier.get("recommended_env") or "")
    if bool(next_frontier.get("runnable_now")) and mapped:
        return mapped, card, "benchmaxx_curriculum"
    policy = state.get("frontier_policy") if isinstance(state.get("frontier_policy"), dict) else {}
    return str(policy.get("frontier_family") or ""), str(policy.get("pressure_card_id") or ""), "frontier_policy_status"


def policy_matches_frontier(row: dict[str, Any], preferred_family: str, pressure_card_id: str) -> bool:
    name = str(row.get("benchmark_name") or "")
    row_family = frontier_family(row)
    if preferred_family and row_family != preferred_family:
        return False
    if not pressure_card_id:
        return True
    normalized_card = pressure_card_id.removeprefix("source_")
    return pressure_card_id in name or normalized_card in name


def frontier_family(row: dict[str, Any]) -> str:
    name = str(row.get("benchmark_name") or "")
    benchmark_type = str(row.get("benchmark_type") or "")
    if name.startswith("drone_rl_") or "drone_rl" in benchmark_type:
        return "drone_rl"
    if name.startswith("minecraft_rl_") or "minecraft" in name or "minecraft_rl" in benchmark_type:
        return "minecraft_rl"
    if name.startswith("coding_"):
        return "coding_local_sandbox"
    if name.startswith("web_agent_"):
        return "web_agent_local"
    if name.startswith("transfer_") or name.startswith("asi_transfer"):
        return "transfer_eval"
    if name.startswith("ocean-"):
        return "rl_local"
    if "babylm" in name:
        return "babylm_mutated"
    return "general"


def below_floor(row: dict[str, Any]) -> bool:
    score = number(row.get("score"), default=0.0)
    floor = number(get_path(row, ["graduation_policy", "floor_threshold"], 0.70), default=0.70)
    return score < floor


def failed_gates(candidate: dict[str, Any]) -> list[str]:
    return [str(item.get("gate")) for item in candidate.get("checks", []) if isinstance(item, dict) and not item.get("passed")]


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    rows = [
        "# Architecture Experiment Governance",
        "",
        f"Updated: {report.get('created_utc')}",
        "",
        f"Architecture change allowed: {report.get('architecture_change_allowed')}",
        "",
        "## State",
        "",
    ]
    for key, value in (report.get("state") or {}).items():
        rows.append(f"- {key}: {value}")
    rows.extend(["", "## Recommended Queue", ""])
    for exp in report.get("experiments", [])[:12]:
        rows.append(f"- {exp.get('id')}: {exp.get('status')} score={exp.get('rank_score')} kind={exp.get('kind')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


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


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def number(value: Any, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed:
        return default
    return parsed


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
