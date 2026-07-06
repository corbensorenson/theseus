"""Harvest repeated workflows that should become verified tools."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-recurrence", type=int, default=3)
    parser.add_argument("--out", default="reports/loop_closure_harvester.json")
    parser.add_argument("--markdown-out", default="reports/loop_closure_harvester.md")
    args = parser.parse_args()

    traces = read_jsonl(ROOT / "reports" / "workflow_routing_traces.jsonl")
    teacher_repairs = read_jsonl(ROOT / "reports" / "teacher_self_edit_traces.jsonl")
    autonomy = read_jsonl(ROOT / "reports" / "autonomy_ledger.jsonl")
    daemon = read_jsonl(ROOT / "reports" / "sparkstream_daemon_ledger.jsonl")
    candidates = build_candidates(traces, teacher_repairs, autonomy, daemon, args.minimum_recurrence)
    report = {
        "policy": "sparkstream_loop_closure_harvester_v0",
        "created_utc": now(),
        "minimum_recurrence": args.minimum_recurrence,
        "summary": {
            "workflow_traces": len(traces),
            "teacher_repair_traces": len(teacher_repairs),
            "autonomy_cycles": len(autonomy),
            "daemon_events": len(daemon),
            "candidates": len(candidates),
            "ready_for_tool_synthesis": len([item for item in candidates if item.get("status") == "ready_for_tool_synthesis"]),
            "blocked_benchmark_or_eval_task_candidates": len(
                [item for item in candidates if item.get("status") == "blocked_benchmark_or_eval_task_not_tool_distillable"]
            ),
        },
        "candidates": candidates,
        "next_actions": [item["next_action"] for item in candidates[:10]],
        "external_inference_calls": 0,
    }
    write_json(ROOT / args.out, report)
    write_markdown(ROOT / args.markdown_out, report)
    print(json.dumps(report, indent=2))
    return 0


def build_candidates(
    traces: list[dict[str, Any]],
    teacher_repairs: list[dict[str, Any]],
    autonomy: list[dict[str, Any]],
    daemon: list[dict[str, Any]],
    minimum: int,
) -> list[dict[str, Any]]:
    command_counts: Counter[str] = Counter()
    command_success: Counter[str] = Counter()
    command_runtime: defaultdict[str, list[int]] = defaultdict(list)
    command_arms: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        key = normalize_command(str(trace.get("command") or trace.get("task") or "unknown"))
        command_counts[key] += 1
        if trace.get("success"):
            command_success[key] += 1
        command_runtime[key].append(int(trace.get("runtime_ms") or 0))
        for arm in trace.get("selected_arms") or []:
            command_arms[key][str(arm)] += 1

    candidates: list[dict[str, Any]] = []
    for key, count in command_counts.most_common():
        if count < minimum:
            continue
        success_rate = command_success[key] / count if count else 0.0
        mean_runtime = sum(command_runtime[key]) / max(1, len(command_runtime[key]))
        tool_name = infer_tool_name(key)
        benchmark_eval_workflow = is_benchmark_or_eval_workflow(key)
        status = (
            "blocked_benchmark_or_eval_task_not_tool_distillable"
            if benchmark_eval_workflow
            else ("ready_for_tool_synthesis" if success_rate >= 0.8 else "needs_verifier_before_tool")
        )
        candidates.append(
            {
                "tool_name": tool_name,
                "source_workflow": key,
                "recurrence_count": count,
                "success_rate": round(success_rate, 4),
                "mean_runtime_ms": int(mean_runtime),
                "dominant_arms": [arm for arm, _ in command_arms[key].most_common(5)],
                "status": status,
                "tool_distillation_allowed": not benchmark_eval_workflow,
                "blocked_reason": "benchmark_or_eval_task_requires_learning_not_tool_template" if benchmark_eval_workflow else "",
                "parameters_to_discover": infer_parameters(key),
                "preconditions": infer_preconditions(key),
                "verification_plan": [
                    "python compile or command dry-run",
                    "schema check output report",
                    "external inference audit remains zero",
                    "register lifecycle and retirement criteria",
                ],
                "risk_tier": "medium" if "teacher" in key or "candidate" in key else "low",
                "runtime_tier": "E2",
                "next_action": (
                    "Do not distill benchmark/eval task workflows into tools; route failures into student training residuals."
                    if benchmark_eval_workflow
                    else f"Compile {tool_name} from repeated workflow after adding schema/verifier tests."
                ),
            }
        )
    candidates.extend(static_candidates(autonomy, daemon, teacher_repairs, minimum))
    return candidates[:50]


def static_candidates(
    autonomy: list[dict[str, Any]],
    daemon: list[dict[str, Any]],
    teacher_repairs: list[dict[str, Any]],
    minimum: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if len(autonomy) >= 3:
        rows.append(
            {
                "tool_name": "candidate_gate_profile_runner",
                "source_workflow": "autonomy_cycle -> training profile -> capability ratchet -> candidate gate",
                "recurrence_count": len(autonomy),
                "success_rate": round(sum(1 for item in autonomy if item.get("ok")) / max(1, len(autonomy)), 4),
                "mean_runtime_ms": 0,
                "dominant_arms": ["head_router", "benchmark_ratchet_arm", "rust_cuda_systems_arm"],
                "status": "ready_for_tool_synthesis",
                "parameters_to_discover": ["profile", "frontier_family", "frontier_seed", "teacher_allowed"],
                "preconditions": ["resource_governor_green", "external_inference_audit_ok"],
                "verification_plan": ["run smoke profile", "candidate gate schema check", "residual delta check"],
                "risk_tier": "medium",
                "runtime_tier": "E2",
                "next_action": "Convert recurrent profile/gate workflow into a single verified ratchet tool.",
            }
        )
    if len(daemon) >= 3:
        rows.append(
            {
                "tool_name": "sparkstream_daemon_health_tool",
                "source_workflow": "daemon cycle monitor and restart readiness",
                "recurrence_count": len(daemon),
                "success_rate": 1.0,
                "mean_runtime_ms": 0,
                "dominant_arms": ["head_router", "safety_reflex_arm"],
                "status": "ready_for_tool_synthesis",
                "parameters_to_discover": ["stale_minutes", "profile", "restart_policy"],
                "preconditions": ["dashboard_or_status_report_present"],
                "verification_plan": ["status endpoint probe", "process state check", "stop/pause flag check"],
                "risk_tier": "medium",
                "runtime_tier": "E1",
                "next_action": "Compile daemon heartbeat checks into a reusable health tool.",
            }
        )
    attd_repairs = [item for item in teacher_repairs if item.get("reason") == "attd_maintenance"]
    if len(attd_repairs) >= minimum:
        success_rate = sum(1 for item in attd_repairs if item.get("success")) / max(1, len(attd_repairs))
        rows.append(
            {
                "tool_name": "attd_maintenance_repair_tool",
                "source_workflow": "ATTD packets -> teacher self-edit -> local checks -> before/after ATTD trace",
                "recurrence_count": len(attd_repairs),
                "success_rate": round(success_rate, 4),
                "mean_runtime_ms": 0,
                "dominant_arms": ["attd_repo_health_governance", "teacher_architect_arm", "code_repair_verifier"],
                "status": "ready_for_tool_synthesis" if success_rate >= 0.8 else "needs_verifier_before_tool",
                "parameters_to_discover": ["packet_component", "scope_paths", "allowed_change_radius", "verification_checks"],
                "preconditions": ["attd_report_available", "maintenance_packets_present", "clean_worktree_or_branch_handoff"],
                "verification_plan": ["ATTD before/after comparison", "python compile", "relevant unit tests", "candidate/regression gates unchanged"],
                "risk_tier": "medium",
                "runtime_tier": "E1",
                "next_action": "Distill repeated ATTD teacher repairs into a local verified maintenance tool.",
            }
        )
    return rows


def normalize_command(command: str) -> str:
    text = command.replace("\\", "/")
    parts = text.split()
    normalized: list[str] = []
    skip_next = False
    for part in parts:
        if skip_next:
            skip_next = False
            continue
        if part in {"--out", "--report-out", "--eval-input", "--input", "--source", "--dest"}:
            normalized.append(part)
            normalized.append("<path>")
            skip_next = True
        elif part.endswith(".exe") or part.endswith("python.exe"):
            normalized.append("<runtime>")
        else:
            normalized.append(part)
    return " ".join(normalized)[:300]


def infer_tool_name(key: str) -> str:
    if "candidate_promotion_gate.py" in key:
        return "candidate_promotion_gate_tool"
    if "synthetic_data_curator.py" in key:
        return "residual_synthetic_data_tool"
    if "run_capability_ratchet.py" in key:
        return "capability_ratchet_refresh_tool"
    if "analyze_babylm_residuals.py" in key:
        return "babylm_residual_analysis_tool"
    if "run_ablation_matrix.py" in key:
        return "ablation_matrix_tool"
    if "training_preflight.py" in key:
        return "training_preflight_tool"
    if "octopus_router.py" in key:
        return "octopus_router_refresh_tool"
    if "symliquid-cli" in key or "train-babylm-probe" in key:
        return "symliquid_local_training_tool"
    return "closed_loop_workflow_tool"


def infer_parameters(key: str) -> list[str]:
    params = ["out"]
    for flag in ["--profile", "--frontier-seed", "--frontier-family", "--eval-input", "--train-limit", "--hv-dim", "--lr"]:
        if flag in key:
            params.append(flag.lstrip("-").replace("-", "_"))
    return sorted(set(params))


def infer_preconditions(key: str) -> list[str]:
    preconditions = ["workspace_present", "no_external_inference"]
    if "cuda" in key or "symliquid-cli" in key:
        preconditions.append("resource_governor_green")
    if "synthetic" in key:
        preconditions.append("leakage_checks_available")
    if "candidate" in key:
        preconditions.append("regression_reports_present")
    return preconditions


def is_benchmark_or_eval_workflow(key: str) -> bool:
    text = key.lower()
    blocked_tokens = [
        "benchmark",
        "eval",
        "frontier",
        "pressure_runner",
        "real_code_benchmark",
        "humaneval",
        "human_eval",
        "evalplus",
        "bigcodebench",
        "livecodebench",
        "mbpp",
        "swe_bench",
        "codeclash",
        "synthetic_benchmark",
        "multi_stream_code_pressure",
    ]
    allowed_infrastructure = [
        "candidate_promotion_gate.py",
        "benchmark_treadmill.py",
        "training_preflight.py",
        "octopus_router.py",
    ]
    if any(token in text for token in allowed_infrastructure):
        return False
    return any(token in text for token in blocked_tokens)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    rows = [
        "# Loop Closure Harvester",
        "",
        f"Updated: {report.get('created_utc')}",
        "",
        "## Summary",
        "",
    ]
    for key, value in (report.get("summary") or {}).items():
        rows.append(f"- {key}: {value}")
    rows.extend(["", "## Candidates", ""])
    for item in report.get("candidates", [])[:30]:
        rows.append(f"- {item.get('tool_name')}: {item.get('status')} recurrence={item.get('recurrence_count')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
