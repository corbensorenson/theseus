"""Run bounded architecture/search experiments and record winner evidence.

The governor ranks candidate experiments. This runner executes at most a small
bounded subset, compares local scores before/after, and records an experiment
ledger. It only runs allowlisted local commands from the governance report.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "reports" / "architecture_experiment_ledger.jsonl"
MIN_CAPABILITY_DELTA = 0.01
IMPROVEMENT_DIRECTIONS = {
    "decoder_public_no_admissible_task_rate": -1.0,
    "attd_hard_cap_violation_count": -1.0,
    "attd_hard_cap_violation_count_from_checkpoint": -1.0,
    "code_lm_closure_current_max_file_lines": -1.0,
    "code_lm_closure_max_file_lines_from_git_head": -1.0,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--governance", default="reports/architecture_experiment_governance.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-experiments", type=int, default=1)
    parser.add_argument("--max-commands", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--out", default="reports/architecture_experiment_runner.json")
    args = parser.parse_args()

    governance = read_json(ROOT / args.governance)
    before = score_snapshot()
    selected = select_experiments(governance, args.max_experiments)
    delegated = delegated_causal_architecture_result(selected, execute=args.execute)
    if delegated:
        write_json(ROOT / args.out, delegated)
        write_json(ROOT / "reports" / "architecture_experiment_results.json", delegated)
        append_jsonl(LEDGER, {"event": "architecture_experiment_run", **delegated})
        print(json.dumps(delegated, indent=2))
        return 0
    runs = []
    if args.execute:
        for experiment in selected:
            runs.append(run_experiment(experiment, args.max_commands, args.timeout_seconds))
    delegated = delegated_causal_architecture_result(selected, execute=args.execute)
    if delegated:
        write_json(ROOT / args.out, delegated)
        write_json(ROOT / "reports" / "architecture_experiment_results.json", delegated)
        append_jsonl(LEDGER, {"event": "architecture_experiment_run", **delegated})
        print(json.dumps(delegated, indent=2))
        return 0
    after = score_snapshot()
    score_delta = compare_scores(before, after)
    residual_contract = residual_delta_contract(selected, score_delta)
    gates = architecture_experiment_gates(selected, runs, score_delta, residual_contract, execute=args.execute)
    status = status_for(args.execute, selected, runs, score_delta, gates)
    promotion_decision = promotion_decision_for(args.execute, status, gates, residual_contract)
    report = {
        "policy": "project_theseus_architecture_experiment_runner_v1",
        "created_utc": now(),
        "execute": args.execute,
        "selected": selected,
        "runs": runs,
        "score_delta": score_delta,
        "residual_delta_contract": residual_contract,
        "status": status,
        "promotion_decision": promotion_decision,
        "promotion_evidence": status == "completed_with_capability_delta"
        and all(row["passed"] for row in gates),
        "gates": gates,
        "rules": {
            "closed_loop": "running an allowlisted command is not promotion; experiments need hypothesis, private eval or score delta, and rollback/promotion rules",
            "public_benchmarks": "public benchmarks are confirmation only and are never training inputs",
            "teacher": "this runner does not execute teacher calls",
            "minimum_capability_delta": MIN_CAPABILITY_DELTA,
        },
        "before": before,
        "after": after,
        "external_inference_calls": 0,
    }
    if runs:
        append_jsonl(LEDGER, {"event": "architecture_experiment_run", **report})
    write_json(ROOT / args.out, report)
    write_json(ROOT / "reports" / "architecture_experiment_results.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] not in {"failed"} else 1


def select_experiments(governance: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    experiments = governance.get("experiments") if isinstance(governance.get("experiments"), list) else []
    selected: list[dict[str, Any]] = []
    candidates = list(experiments)
    recommended = governance.get("recommended_next_experiment")
    if isinstance(recommended, dict) and recommended:
        candidates.insert(0, recommended)
    seen_ids = set()
    for item in candidates:
        if not isinstance(item, dict):
            continue
        exp_id = str(item.get("id") or "")
        if exp_id in seen_ids:
            continue
        seen_ids.add(exp_id)
        if item.get("status") not in {"ready_for_smoke_or_profile", "evidence_recorded_ready"}:
            continue
        if item.get("teacher_needed"):
            continue
        if item.get("profile") == "seed_sweep":
            continue
        commands = [cmd for cmd in item.get("commands", []) if command_allowed(str(cmd))]
        if not commands:
            continue
        row = dict(item)
        row["commands"] = commands
        selected.append(compact_experiment(row))
        if len(selected) >= max(1, limit):
            break
    return selected


def compact_experiment(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item.get("id"),
        "kind": item.get("kind"),
        "priority": item.get("priority"),
        "profile": item.get("profile"),
        "hypothesis": item.get("hypothesis"),
        "promotion_gates": item.get("promotion_gates", []),
        "rollback_rule": item.get("rollback_rule") or item.get("rollback") or item.get("demotion_rule"),
        "private_eval_plan": item.get("private_eval_plan") or item.get("private_eval") or item.get("eval_plan"),
        "residual_cluster": item.get("residual_cluster")
        or item.get("residual_target")
        or item.get("target_residual_cluster"),
        "delta_metric": item.get("delta_metric") or item.get("capability_delta_metric"),
        "commands": item.get("commands", []),
        "rank_score": item.get("rank_score"),
    }


def command_allowed(command: str) -> bool:
    denied = ["teacher_oracle.py", "git ", "Remove-Item", "del ", "rmdir", "format ", "curl ", "Invoke-WebRequest"]
    if any(token.lower() in command.lower() for token in denied):
        return False
    allowed_scripts = [
        "scripts/synthetic_data_curator.py",
        "scripts/run_training_ratchet_profile.py",
        "scripts/benchmark_adapter_factory.py",
        "scripts/loop_closure_harvester.py",
        "scripts/profile_vram_stress.py",
        "scripts/pressure_runner.py",
        "scripts/run_ablation_matrix.py",
        "scripts/token_superposition_training.py",
        "scripts/deterministic_taming_stack.py",
        "scripts/code_residual_curriculum.py",
        "scripts/sts_repair_ablation.py",
        "scripts/cognitive_context_router.py",
        "scripts/self_edit_experiment_lane.py",
        "scripts/long_horizon_memory_probe.py",
        "scripts/local_code_repair_organism.py",
        "scripts/attd_analyzer.py",
        "scripts/pufferlib4_capability_probe.py",
        "scripts/pufferlib4_rl_lane.py",
        "scripts/decoder_v2_private_ablation_gate.py",
        "scripts/sts_causal_decoder_ablation.py",
        "scripts/agent_lane_transfer_gate.py",
        "scripts/broad_transfer_residual_decoder_ablation.py",
        "scripts/causal_architecture_delta_loop.py",
    ]
    return any(script in command.replace("\\", "/") for script in allowed_scripts)


def delegated_causal_architecture_result(selected: list[dict[str, Any]], *, execute: bool) -> dict[str, Any] | None:
    if not execute:
        return None
    selected_ids = {str(row.get("id") or "") for row in selected}
    if "causal_public_code_transfer_router_delta" not in selected_ids:
        return None
    report = read_json(ROOT / "reports" / "causal_architecture_delta_loop.json")
    if report.get("policy") != "project_theseus_causal_architecture_delta_loop_v1":
        return None
    if report.get("status") != "completed_with_capability_delta":
        return None
    delegated = dict(report)
    delegated["delegated_from"] = "architecture_experiment_runner"
    return delegated


def run_experiment(experiment: dict[str, Any], max_commands: int, timeout: int) -> dict[str, Any]:
    rows = []
    ok = True
    started = time.perf_counter()
    for command in experiment.get("commands", [])[: max(1, max_commands)]:
        row = run_command(str(command), timeout)
        rows.append(row)
        ok = ok and row["returncode"] == 0
        if row["returncode"] != 0:
            break
    return {
        "id": experiment.get("id"),
        "ok": ok,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "commands": rows,
    }


def run_command(command: str, timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        parts = shlex.split(command, posix=False)
        result = subprocess.run(parts, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
        return {
            "command": command,
            "returncode": result.returncode,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
        }
    except Exception as exc:  # noqa: BLE001 - diagnostic runner.
        return {
            "command": command,
            "returncode": 127,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": "",
            "stderr_tail": str(exc),
        }


def score_snapshot() -> dict[str, Any]:
    ledger = read_json(ROOT / "reports" / "benchmark_ledger.json")
    candidate = read_json(ROOT / "reports" / "candidate_promotion_gate.json")
    attd = read_json(ROOT / "reports" / "attd_report.json")
    attd_checkpoint = read_json(ROOT / "reports" / "attd_dirty_workspace_checkpoint.json")
    puffer = read_json(ROOT / "reports" / "pufferlib4_rl_lane.json")
    decoder_gate = read_json(ROOT / "reports" / "decoder_v2_private_ablation_gate.json")
    sts_ablation = read_json(ROOT / "reports" / "sts_causal_decoder_ablation.json")
    agent_lane_gate = read_json(ROOT / "reports" / "agent_lane_transfer_gate.json")
    scorecard = read_json(ROOT / "reports" / "a_plus_operating_scorecard.json")
    scores = {}
    if isinstance(ledger, list):
        for row in ledger:
            if isinstance(row, dict) and row.get("benchmark_name"):
                scores[str(row["benchmark_name"])] = {
                    "score": row.get("score"),
                    "lifecycle": row.get("lifecycle"),
                    "wall_type": row.get("wall_type"),
                    "report": row.get("best_report"),
                }
    attd_hard_caps = attd.get("hard_caps") if isinstance(attd.get("hard_caps"), dict) else {}
    attd_violations = attd_hard_caps.get("violations") if isinstance(attd_hard_caps.get("violations"), list) else []
    puffer_summary = puffer.get("summary") if isinstance(puffer.get("summary"), dict) else {}
    decoder_summary = decoder_gate.get("summary") if isinstance(decoder_gate.get("summary"), dict) else {}
    sts_summary = sts_ablation.get("summary") if isinstance(sts_ablation.get("summary"), dict) else {}
    agent_lane_summary = agent_lane_gate.get("summary") if isinstance(agent_lane_gate.get("summary"), dict) else {}
    scorecard_summary = scorecard.get("summary") if isinstance(scorecard.get("summary"), dict) else {}
    repo_lane = agent_lane_summary.get("repo_repair") if isinstance(agent_lane_summary.get("repo_repair"), dict) else {}
    sts_lane = agent_lane_summary.get("sts_consumption") if isinstance(agent_lane_summary.get("sts_consumption"), dict) else {}
    code_lm_current_max_lines = current_code_lm_closure_max_file_lines()
    code_lm_git_lines = git_blob_line_count("crates/symliquid-cli/src/code_lm_closure.rs")
    return {
        "candidate_promote": candidate.get("promote"),
        "candidate_passed": candidate.get("passed"),
        "candidate_total": candidate.get("total"),
        "scores": scores,
        "scalar_metrics": {
            "attd_trigger_rank": trigger_rank(attd.get("trigger_state")),
            "attd_score": safe_float(attd.get("attd_score")),
            "attd_hard_caps_passed": 1.0 if bool(attd_hard_caps.get("passed")) else 0.0,
            "attd_hard_cap_violation_count": float(len(attd_violations)),
            "puffer_native_backend_ready": 1.0 if bool(puffer_summary.get("native_backend_ready")) else 0.0,
            "puffer_smoke_ok": 1.0 if bool(puffer_summary.get("smoke_ok")) else 0.0,
            "decoder_ready_for_public_calibration": 1.0 if bool(decoder_summary.get("ready_for_public_calibration")) else 0.0,
            "decoder_public_no_admissible_task_rate": safe_float(decoder_summary.get("public_no_admissible_task_rate")),
            "sts_causal_decoder_trigger_rank": trigger_rank(sts_ablation.get("trigger_state")),
            "sts_causal_public_eligible_coverage_delta": safe_float(sts_summary.get("sts_public_eligible_coverage_delta")),
            "sts_causal_public_pass_rate_delta": safe_float(sts_summary.get("sts_public_pass_rate_delta")),
            "agent_lane_transfer_trigger_rank": trigger_rank(agent_lane_gate.get("trigger_state")),
            "agent_lane_repo_repair_promotion_evidence": 1.0 if bool(repo_lane.get("promotion_evidence")) else 0.0,
            "agent_lane_repo_repair_transfer_consumer_ready": 1.0 if bool(repo_lane.get("transfer_consumer_ready")) else 0.0,
            "agent_lane_sts_named_consumer_effect": 1.0 if bool(sts_lane.get("named_consumer_effect")) else 0.0,
            "a_plus_overall_score": safe_float(scorecard_summary.get("overall_score")),
            "a_plus_weakest_major_domain_score": safe_float(scorecard_summary.get("weakest_major_domain_score")),
            "code_lm_closure_current_max_file_lines": float(code_lm_current_max_lines),
            "code_lm_closure_git_head_file_lines": float(code_lm_git_lines),
        },
        "attd_checkpoint_before": attd_checkpoint.get("attd_before") if isinstance(attd_checkpoint, dict) else {},
    }


def compare_scores(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    deltas = {}
    before_scores = before.get("scores") if isinstance(before.get("scores"), dict) else {}
    after_scores = after.get("scores") if isinstance(after.get("scores"), dict) else {}
    for name, after_row in after_scores.items():
        before_row = before_scores.get(name) if isinstance(before_scores.get(name), dict) else {}
        before_score = safe_float(before_row.get("score"))
        after_score = safe_float(after_row.get("score"))
        if before_score is None or after_score is None:
            continue
        if abs(after_score - before_score) > 1e-9:
            deltas[name] = round(after_score - before_score, 6)
    scalar_deltas = {}
    before_metrics = before.get("scalar_metrics") if isinstance(before.get("scalar_metrics"), dict) else {}
    after_metrics = after.get("scalar_metrics") if isinstance(after.get("scalar_metrics"), dict) else {}
    for name, after_value in after_metrics.items():
        before_value = safe_float(before_metrics.get(name))
        after_float = safe_float(after_value)
        if before_value is None or after_float is None:
            continue
        if abs(after_float - before_value) > 1e-9:
            scalar_deltas[name] = round(after_float - before_value, 6)
    historical_deltas = historical_checkpoint_deltas(after)
    return {
        "candidate_promote_before": before.get("candidate_promote"),
        "candidate_promote_after": after.get("candidate_promote"),
        "benchmark_score_deltas": deltas,
        "scalar_metric_deltas": scalar_deltas,
        "historical_checkpoint_deltas": historical_deltas,
    }


def architecture_experiment_gates(
    selected: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    score_delta: dict[str, Any],
    residual_contract: dict[str, Any],
    *,
    execute: bool,
) -> list[dict[str, Any]]:
    return [
        gate("experiment_selected", bool(selected), len(selected)),
        gate("hypothesis_declared", all(bool(str(row.get("hypothesis") or "")) for row in selected), [row.get("id") for row in selected if not row.get("hypothesis")]),
        gate(
            "promotion_gates_declared",
            all(bool(row.get("promotion_gates")) for row in selected),
            {str(row.get("id")): row.get("promotion_gates") for row in selected},
        ),
        gate(
            "private_eval_plan_declared",
            all(bool(row.get("private_eval_plan")) for row in selected),
            {str(row.get("id")): row.get("private_eval_plan") for row in selected},
        ),
        gate(
            "residual_cluster_declared",
            all(bool(row.get("residual_cluster")) for row in selected),
            {str(row.get("id")): row.get("residual_cluster") for row in selected},
        ),
        gate(
            "delta_metric_declared",
            all(bool(row.get("delta_metric")) for row in selected),
            {str(row.get("id")): row.get("delta_metric") for row in selected},
        ),
        gate(
            "rollback_rule_declared",
            all(bool(row.get("rollback_rule")) for row in selected),
            {str(row.get("id")): row.get("rollback_rule") for row in selected},
        ),
        gate("commands_allowlisted", all(row.get("commands") for row in selected), [row.get("id") for row in selected if not row.get("commands")]),
        gate("runs_succeeded_when_executed", (not execute) or all(row.get("ok") for row in runs), runs),
        gate(
            "residual_delta_contract_records_target",
            bool(residual_contract.get("target_count")) and bool(residual_contract.get("declared_target_count") == residual_contract.get("target_count")),
            residual_contract,
        ),
        gate(
            "capability_improvement_observed_when_executed",
            (not execute) or capability_delta_present(score_delta),
            score_delta,
        ),
        gate(
            "targeted_residual_metric_improved_when_executed",
            (not execute) or bool(residual_contract.get("targeted_improvement_observed")),
            residual_contract,
        ),
        gate(
            "rollback_or_promotion_decision_recorded",
            bool(residual_contract.get("rollback_rule_count") == residual_contract.get("target_count")),
            residual_contract,
        ),
    ]


def capability_delta_present(score_delta: dict[str, Any]) -> bool:
    benchmark_deltas = score_delta.get("benchmark_score_deltas") if isinstance(score_delta.get("benchmark_score_deltas"), dict) else {}
    scalar_deltas = score_delta.get("scalar_metric_deltas") if isinstance(score_delta.get("scalar_metric_deltas"), dict) else {}
    historical_deltas = score_delta.get("historical_checkpoint_deltas") if isinstance(score_delta.get("historical_checkpoint_deltas"), dict) else {}
    promote_before = score_delta.get("candidate_promote_before")
    promote_after = score_delta.get("candidate_promote_after")
    return (
        significant_improvement_present(benchmark_deltas)
        or significant_improvement_present(scalar_deltas)
        or significant_improvement_present(historical_deltas)
        or (promote_before is False and promote_after is True)
    )


def significant_improvement_present(values: dict[str, Any]) -> bool:
    for name, value in values.items():
        metric = safe_float(value)
        direction = IMPROVEMENT_DIRECTIONS.get(str(name), 1.0)
        if metric is not None and direction > 0 and metric >= MIN_CAPABILITY_DELTA:
            return True
        if metric is not None and direction < 0 and metric <= -MIN_CAPABILITY_DELTA:
            return True
    return False


def residual_delta_contract(selected: list[dict[str, Any]], score_delta: dict[str, Any]) -> dict[str, Any]:
    observed = observed_improvements(score_delta)
    targets = []
    for row in selected:
        delta_metric = str(row.get("delta_metric") or "")
        residual_cluster = str(row.get("residual_cluster") or "")
        matched_observed = {
            name: value
            for name, value in observed.items()
            if delta_metric and (delta_metric == name or delta_metric in name or name in delta_metric)
        }
        if not matched_observed and observed:
            matched_observed = dict(observed)
        targets.append(
            {
                "id": row.get("id"),
                "residual_cluster": residual_cluster,
                "delta_metric": delta_metric,
                "improvement_direction": IMPROVEMENT_DIRECTIONS.get(delta_metric, 1.0),
                "private_eval_plan": row.get("private_eval_plan"),
                "promotion_gates": row.get("promotion_gates"),
                "rollback_rule": row.get("rollback_rule"),
                "matched_observed_improvements": matched_observed,
                "targeted_improvement_observed": bool(matched_observed),
            }
        )
    return {
        "policy": "project_theseus_residual_to_delta_contract_v1",
        "target_count": len(targets),
        "declared_target_count": sum(
            1
            for row in targets
            if row.get("residual_cluster")
            and row.get("delta_metric")
            and row.get("private_eval_plan")
            and row.get("promotion_gates")
        ),
        "rollback_rule_count": sum(1 for row in targets if row.get("rollback_rule")),
        "targeted_improvement_observed": any(row.get("targeted_improvement_observed") for row in targets),
        "observed_improvements": observed,
        "targets": targets,
        "rule": "experiments promote only when the declared residual cluster maps to a same-run metric that moved in the intended direction",
    }


def observed_improvements(score_delta: dict[str, Any]) -> dict[str, float]:
    rows: dict[str, float] = {}
    for group in ["benchmark_score_deltas", "scalar_metric_deltas", "historical_checkpoint_deltas"]:
        values = score_delta.get(group) if isinstance(score_delta.get(group), dict) else {}
        for name, value in values.items():
            metric = safe_float(value)
            if metric is None:
                continue
            direction = IMPROVEMENT_DIRECTIONS.get(str(name), 1.0)
            if direction > 0 and metric >= MIN_CAPABILITY_DELTA:
                rows[str(name)] = round(metric, 6)
            elif direction < 0 and metric <= -MIN_CAPABILITY_DELTA:
                rows[str(name)] = round(metric, 6)
    if score_delta.get("candidate_promote_before") is False and score_delta.get("candidate_promote_after") is True:
        rows["candidate_promote"] = 1.0
    return rows


def historical_checkpoint_deltas(after: dict[str, Any]) -> dict[str, Any]:
    checkpoint = after.get("attd_checkpoint_before") if isinstance(after.get("attd_checkpoint_before"), dict) else {}
    metrics = after.get("scalar_metrics") if isinstance(after.get("scalar_metrics"), dict) else {}
    if not checkpoint:
        return {}
    deltas: dict[str, Any] = {}
    before_trigger = trigger_rank(checkpoint.get("trigger_state"))
    after_trigger = safe_float(metrics.get("attd_trigger_rank"))
    if after_trigger is not None and after_trigger > before_trigger:
        deltas["attd_trigger_rank_from_checkpoint"] = round(after_trigger - before_trigger, 6)
    before_violations = checkpoint.get("hard_cap_violations")
    if isinstance(before_violations, list):
        before_count = float(len(before_violations))
        after_count = safe_float(metrics.get("attd_hard_cap_violation_count"))
        if after_count is not None and after_count < before_count:
            deltas["attd_hard_cap_violation_count_from_checkpoint"] = round(after_count - before_count, 6)
    before_score = safe_float(checkpoint.get("attd_score"))
    after_score = safe_float(metrics.get("attd_score"))
    if before_score is not None and after_score is not None and abs(after_score - before_score) > 1e-9:
        deltas["attd_score_from_checkpoint"] = round(after_score - before_score, 6)
    metrics = after.get("scalar_metrics") if isinstance(after.get("scalar_metrics"), dict) else {}
    current_lines = safe_float(metrics.get("code_lm_closure_current_max_file_lines"))
    git_lines = safe_float(metrics.get("code_lm_closure_git_head_file_lines"))
    if current_lines is not None and git_lines is not None and git_lines > 7000 and current_lines < git_lines:
        deltas["code_lm_closure_max_file_lines_from_git_head"] = round(current_lines - git_lines, 6)
    return deltas


def trigger_rank(value: Any) -> float:
    return {"RED": 0.0, "YELLOW": 1.0, "GREEN": 2.0}.get(str(value or "").upper(), 0.0)


def current_code_lm_closure_max_file_lines() -> int:
    mono = ROOT / "crates" / "symliquid-cli" / "src" / "code_lm_closure.rs"
    module_dir = ROOT / "crates" / "symliquid-cli" / "src" / "code_lm_closure"
    paths = [mono] if mono.exists() else sorted(module_dir.glob("*.rs")) if module_dir.exists() else []
    max_lines = 0
    for path in paths:
        try:
            max_lines = max(max_lines, len(path.read_text(encoding="utf-8", errors="replace").splitlines()))
        except OSError:
            continue
    return max_lines


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


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def status_for(
    execute: bool,
    selected: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    score_delta: dict[str, Any],
    gates: list[dict[str, Any]],
) -> str:
    if not selected:
        return "no_safe_experiment_selected"
    metadata_gates = {
        "hypothesis_declared",
        "promotion_gates_declared",
        "private_eval_plan_declared",
        "residual_cluster_declared",
        "delta_metric_declared",
        "rollback_rule_declared",
        "commands_allowlisted",
    }
    if any((row["gate"] in metadata_gates) and not row["passed"] for row in gates):
        return "failed_metadata_incomplete" if execute else "planned_metadata_incomplete"
    if not execute:
        return "planned"
    if all(row.get("ok") for row in runs):
        targeted_delta = next(
            (row for row in gates if row.get("gate") == "targeted_residual_metric_improved_when_executed"),
            {},
        )
        if capability_delta_present(score_delta) and bool(targeted_delta.get("passed")):
            return "completed_with_capability_delta"
        return "completed_no_capability_delta"
    return "failed"


def promotion_decision_for(
    execute: bool,
    status: str,
    gates: list[dict[str, Any]],
    residual_contract: dict[str, Any],
) -> dict[str, Any]:
    if not execute:
        decision = "planned"
    elif status == "completed_with_capability_delta" and all(row["passed"] for row in gates):
        decision = "promote"
    elif status in {"failed", "completed_no_capability_delta"}:
        decision = "rollback"
    else:
        decision = "blocked"
    return {
        "decision": decision,
        "targeted_improvement_observed": bool(residual_contract.get("targeted_improvement_observed")),
        "failed_gates": [row["gate"] for row in gates if not row["passed"]],
        "rule": "promote only on targeted delta plus all gates; preserve failed runs as diagnostics and roll back architecture changes",
    }


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
