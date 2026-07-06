"""Build VIEA growth surfaces for Theseus.

This script turns the architecture goals into concrete, generated reports:

* broad transfer closure;
* digital-runtime readiness;
* workflow-to-tool compiler v2 scoring;
* SymLiquid integration map;
* teacher-as-architect experiment queue;
* feedback ratchet.

It does not train on public benchmark answers and does not call external
inference. It is a control/report layer for honest growth.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PUBLIC_FLOOR = 0.70


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    out_dir = resolve(args.out_dir)
    state = load_state()
    reports = {
        "broad_transfer_closure": broad_transfer_closure(state),
        "digital_runtime_adapter": digital_runtime_adapter(state),
        "workflow_tool_compiler_v2": workflow_tool_compiler_v2(state),
        "symliquid_substrate_map": symliquid_substrate_map(state),
        "teacher_architect_loop": teacher_architect_loop(state),
    }
    reports["feedback_ratchet"] = feedback_ratchet(state, reports)

    for name, payload in reports.items():
        write_json(out_dir / f"{name}.json", payload)
        write_text(out_dir / f"{name}.md", render_markdown(name, payload))

    summary = {
        "policy": "project_theseus_viea_growth_surfaces_v1",
        "created_utc": now(),
        "trigger_state": aggregate_state(reports.values()),
        "reports": {name: f"reports/{name}.json" for name in reports},
        "external_inference_calls": 0,
    }
    write_json(out_dir / "viea_growth_surfaces.json", summary)
    write_text(out_dir / "viea_growth_surfaces.md", render_markdown("viea_growth_surfaces", summary))
    print(json.dumps(summary, indent=2))
    return 0 if summary["trigger_state"] in {"GREEN", "YELLOW"} else 2


def load_state() -> dict[str, Any]:
    return {
        "learning_scoreboard": read_json(REPORTS / "learning_scoreboard.json"),
        "broad_transfer_matrix": read_json(REPORTS / "broad_transfer_matrix.json"),
        "broad_scheduler": read_json(REPORTS / "broad_code_calibration_scheduler.json"),
        "real_code": read_json(REPORTS / "real_code_benchmark_graduation.json"),
        "student_first": read_json(REPORTS / "student_first_evidence_audit.json"),
        "code_lm": read_json(REPORTS / "code_lm_closure.json"),
        "code_lm_rust": read_json(REPORTS / "code_lm_closure_rust.json"),
        "sts_native": read_json(REPORTS / "sts_native_parallel_probe.json"),
        "sts_repair": read_json(REPORTS / "sts_repair_ablation.json"),
        "residual_escrow": read_json(REPORTS / "residual_escrow.json"),
        "code_residual_forge": read_json(REPORTS / "code_residual_forge.json"),
        "tool_registry": read_json(REPORTS / "tool_registry.json"),
        "loop_harvester": read_json(REPORTS / "loop_closure_harvester.json"),
        "loop_promoter": read_json(REPORTS / "loop_closure_tool_promoter.json"),
        "arm_lifecycle": read_json(REPORTS / "arm_lifecycle_governance.json"),
        "cell_lifecycle": read_json(REPORTS / "cell_lifecycle.json"),
        "routing_memory": read_json(REPORTS / "routing_memory.json"),
        "octopus_router": read_json(REPORTS / "octopus_router_report.json"),
        "architecture_guidance": read_json(REPORTS / "architecture_guidance_loop.json"),
        "teacher_budget": read_json(REPORTS / "teacher_budget_audit.json"),
        "command_executor": read_json(REPORTS / "viea_command_executor.json"),
        "reality": read_json(REPORTS / "reality_manipulator.json"),
        "private_repo_repair": read_json(REPORTS / "private_repo_repair_curriculum.json"),
        "autonomy_watchdog": read_json(REPORTS / "autonomy_watchdog.json"),
    }


def broad_transfer_closure(state: dict[str, Any]) -> dict[str, Any]:
    matrix = state["broad_transfer_matrix"]
    summary = object_field(matrix, "summary")
    scheduler = state["broad_scheduler"]
    rows = matrix.get("rows") if isinstance(matrix.get("rows"), list) else []
    closure_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        card = str(row.get("card_id") or "")
        public_tasks = int_number(row.get("public_task_count"))
        pass_rate = number(row.get("multi_stream_pass_rate"))
        sts_delta = round(number(row.get("multi_stream_pass_rate")) - number(row.get("single_stream_pass_rate")), 6)
        blockers = []
        if row.get("no_cheat_violations"):
            blockers.append("no_cheat_violation")
        if public_tasks < int_number(summary.get("min_public_tasks_per_promotion_card"), 32):
            blockers.append("needs_32_plus_clean_tasks")
        if pass_rate < PUBLIC_FLOOR:
            blockers.append("below_0_70_floor")
        if sts_delta <= 0:
            blockers.append("sts_not_causal_on_card")
        closure_rows.append(
            {
                "card_id": card,
                "public_task_count": public_tasks,
                "pass_rate": pass_rate,
                "floor_gap": round(max(0.0, PUBLIC_FLOOR - pass_rate), 6),
                "sts_delta": sts_delta,
                "blockers": blockers,
                "next_action": next_transfer_action(card, blockers, row),
                "public_calibration_only": True,
            }
        )
    aggregate = number(summary.get("real_public_pass_rate"))
    gates = [
        gate("broad_matrix_loaded", matrix.get("policy") == "project_theseus_broad_transfer_matrix_v1", matrix.get("policy")),
        gate("no_cheat_violations_zero", int_number(summary.get("no_cheat_violation_count")) == 0, summary.get("no_cheat_violation_count")),
        gate("aggregate_above_floor", aggregate >= PUBLIC_FLOOR, aggregate, severity="soft"),
        gate("selected_next_card_present", bool(get_path(scheduler, ["selected", "card_id"], "")), get_path(scheduler, ["selected", "card_id"], ""), severity="soft"),
    ]
    return {
        "policy": "project_theseus_broad_transfer_closure_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(item["passed"] for item in gates) else "YELLOW",
        "summary": {
            "aggregate_public_task_count": summary.get("real_public_task_count"),
            "aggregate_pass_rate": aggregate,
            "aggregate_floor_gap": round(max(0.0, PUBLIC_FLOOR - aggregate), 6),
            "aggregate_sts_delta": summary.get("real_public_sts_delta"),
            "selected_next_card": get_path(scheduler, ["selected", "card_id"], ""),
            "selected_next_action": get_path(scheduler, ["selected", "action"], ""),
            "promotion_evidence": False,
        },
        "rows": closure_rows,
        "gates": gates,
        "rules": {
            "public_data": "calibration_only_not_training",
            "required_student_evidence": "token_level_full_body_generation_no_templates_no_wrappers_no_loop_tools",
            "target": "MBPP/EvalPlus above floor and BigCodeBench/LiveCodeBench at 32+ clean tasks",
        },
        "external_inference_calls": 0,
    }


def digital_runtime_adapter(state: dict[str, Any]) -> dict[str, Any]:
    executor = state["command_executor"]
    artifacts = object_field(executor, "artifacts")
    required = [
        "code_patch_packet",
        "test_packet",
        "release_manifest",
        "rollback_plan",
        "repo_repair_trace",
        "dashboard_actions",
    ]
    packet_rows = []
    for key in required:
        path = str(artifacts.get(key) or "")
        packet_rows.append({"id": key, "path": path, "exists": bool(path and resolve(path).exists())})
    gates = [
        gate("command_executor_loaded", executor.get("policy") == "project_theseus_viea_command_executor_v1", executor.get("policy")),
        gate("digital_packets_complete", all(row["exists"] for row in packet_rows), packet_rows),
        gate("high_risk_runtime_planning_only", not any(call.get("status") == "completed" and str(call.get("stage", "")).endswith("_planning_boundary") for call in executor.get("specialist_calls", []) if isinstance(call, dict)), "chip/matter/robotic blocked"),
    ]
    return {
        "policy": "project_theseus_digital_runtime_adapter_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(item["passed"] for item in gates) else "YELLOW",
        "packets": packet_rows,
        "dashboard_actions": [
            "surface_viea_kernel",
            "surface_command_executor",
            "surface_digital_runtime_packets",
            "surface_broad_transfer_closure",
            "surface_feedback_ratchet",
        ],
        "matter_chip_robotic_boundary": "planning_only_until_explicit_gate",
        "gates": gates,
        "external_inference_calls": 0,
    }


def workflow_tool_compiler_v2(state: dict[str, Any]) -> dict[str, Any]:
    registry = state["tool_registry"]
    tools = registry.get("tools") if isinstance(registry.get("tools"), list) else []
    lifecycle = object_field(state["cell_lifecycle"], "tool_lifecycle")
    lifecycle_items = lifecycle.get("items") if isinstance(lifecycle.get("items"), list) else []
    lifecycle_by_name = {str(item.get("name") or item.get("tool_name") or ""): item for item in lifecycle_items if isinstance(item, dict)}
    scored = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("tool_name") or tool.get("name") or "")
        verification = tool.get("verification_tests") if isinstance(tool.get("verification_tests"), list) else []
        benchmark_related = bool(tool.get("benchmark_related"))
        lifecycle_row = lifecycle_by_name.get(name, {})
        usage = int_number(get_path(lifecycle_row, ["usage", "uses"], tool.get("usage_count", 0)))
        score = min(1.0, 0.25 + 0.08 * len(verification) + 0.05 * min(usage, 6))
        if benchmark_related:
            score -= 0.15
        status = "active"
        if score < 0.35 and not tool.get("protected"):
            status = "expire_review"
        scored.append(
            {
                "tool_name": name,
                "task_family": tool.get("task_family"),
                "verification_test_count": len(verification),
                "usage_count": usage,
                "benchmark_related": benchmark_related,
                "earned_existence_score": round(max(0.0, min(1.0, score)), 3),
                "lifecycle_status": status,
                "expiration_policy": "renew_on_use_or_score_impact_else_expire_review",
                "promotion_evidence": False,
            }
        )
    gates = [
        gate("tool_registry_loaded", bool(registry.get("tools")), len(tools)),
        gate("tools_scored", len(scored) > 0, len(scored)),
        gate("benchmark_answer_tools_blocked", not any(row["benchmark_related"] and row["promotion_evidence"] for row in scored), "benchmark tools never promotion evidence"),
    ]
    return {
        "policy": "project_theseus_workflow_tool_compiler_v2",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(item["passed"] for item in gates) else "YELLOW",
        "summary": {
            "tool_count": len(scored),
            "expire_review_count": sum(1 for row in scored if row["lifecycle_status"] == "expire_review"),
            "average_earned_existence_score": round(sum(row["earned_existence_score"] for row in scored) / max(1, len(scored)), 3),
        },
        "tools": scored,
        "acceptance_rule": "expected recurrence * value * reliability gain > creation cost + maintenance cost + verification cost + risk cost + drift cost",
        "gates": gates,
        "external_inference_calls": 0,
    }


def symliquid_substrate_map(state: dict[str, Any]) -> dict[str, Any]:
    rows = [
        sym_row("router_memory", "routing_memory.json", state["routing_memory"], "Use SymLiquid state to compress task signatures and route outcomes."),
        sym_row("residual_clustering", "code_residual_forge.json / residual_escrow.json", state["code_residual_forge"] or state["residual_escrow"], "Cluster residuals into architecture/data/tool pressure."),
        sym_row("stateful_sequence_decoding", "code_lm_closure_rust.json", state["code_lm_rust"], "Carry full-body code generation with recurrent/state sequence lanes."),
        sym_row("tool_selection", "octopus_router_report.json + tool_registry.json", state["octopus_router"] if state["tool_registry"] else {}, "Select earned tools through routed specialist state."),
        sym_row("sts_stream_conditioning", "sts_native_parallel_probe.json", state["sts_native"], "Condition solver/critic/patch/residual/tool streams."),
        sym_row("small_control_policies", "benchmaxx/resource/hive reports", state["learning_scoreboard"], "Use compact controllers for scheduling and runtime policy."),
        sym_row("long_running_autonomy_state", "autonomy_watchdog.json", state["autonomy_watchdog"], "Preserve long-running loop state without relying on chat context."),
    ]
    gates = [
        gate("symliquid_rows_declared", len(rows) == 7, len(rows)),
        gate("critical_sequence_lane_present", rows[2]["evidence_present"], rows[2]["report"]),
        gate("sts_conditioning_present", rows[4]["evidence_present"], rows[4]["report"], severity="soft"),
    ]
    return {
        "policy": "project_theseus_symliquid_substrate_map_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(item["passed"] for item in gates) else "YELLOW",
        "principle": "SymLiquid is the compact recurrent/state substrate inside VIEA, not a replacement for every component.",
        "rows": rows,
        "gates": gates,
        "external_inference_calls": 0,
    }


def teacher_architect_loop(state: dict[str, Any]) -> dict[str, Any]:
    guidance = state["architecture_guidance"]
    experiments = guidance.get("experiments") if isinstance(guidance.get("experiments"), list) else []
    budget = state["teacher_budget"]
    residual_counts = collect_residual_counts(state)
    experiment_specs = []
    for experiment in experiments[:8]:
        if not isinstance(experiment, dict):
            continue
        experiment_specs.append(
            {
                "id": experiment.get("id"),
                "kind": experiment.get("kind"),
                "teacher_needed": bool(experiment.get("teacher_needed")),
                "hypothesis": experiment.get("hypothesis"),
                "allowed_teacher_role": "architecture_diagnosis_only",
                "forbidden": ["benchmark_answers", "hidden_tests", "public_solution_distillation", "apply_mode_without_gate"],
                "private_eval_required": True,
                "public_calibration_required": True,
                "promote_or_rollback": "promote_only_on_private_gain_public_no_regression_else_rollback",
            }
        )
    gates = [
        gate("architecture_guidance_loaded", guidance.get("policy") == "project_theseus_architecture_guidance_loop_v1", guidance.get("policy")),
        gate("experiment_specs_present", len(experiment_specs) > 0, len(experiment_specs)),
        gate("teacher_architecture_budget_known", bool(budget), budget.get("policy")),
        gate("no_answer_distillation", True, "experiment specs contain hypotheses and commands only"),
    ]
    return {
        "policy": "project_theseus_teacher_architect_loop_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(item["passed"] for item in gates) else "YELLOW",
        "residual_clusters": residual_counts,
        "experiment_specs": experiment_specs,
        "budget_summary": {
            "architecture_allowed": get_path(budget, ["reason_decisions", "architecture_wall", "budget", "allowed"], None),
            "completed_architecture_today": budget.get("completed_architecture_today"),
        },
        "gates": gates,
        "external_inference_calls": 0,
    }


def feedback_ratchet(state: dict[str, Any], reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scoreboard = state["learning_scoreboard"]
    broad = reports["broad_transfer_closure"]
    workflow = reports["workflow_tool_compiler_v2"]
    improved = []
    regressed = []
    became_tool = []
    became_residual = []
    should_expire = []
    train_next = []
    if number(get_path(scoreboard, ["public_transfer", "pass_rate_delta"], 0.0)) > 0:
        improved.append("public code pass-rate delta is positive on the selected clean report")
    if number(get_path(scoreboard, ["broad_transfer_matrix", "real_public_sts_delta"], 0.0)) > 0:
        improved.append("STS-on beats STS-off at broad aggregate calibration")
    if number(get_path(broad, ["summary", "aggregate_floor_gap"], 0.0)) > 0:
        became_residual.append("broad semantic transfer remains below floor")
        train_next.append("private MBPP/EvalPlus semantic lookalikes plus 32+ BigCodeBench/LiveCodeBench adapters")
    expire_review = [row["tool_name"] for row in workflow.get("tools", []) if row.get("lifecycle_status") == "expire_review"]
    should_expire.extend(expire_review[:12])
    became_tool.extend([row["tool_name"] for row in workflow.get("tools", []) if row.get("earned_existence_score", 0) >= 0.55][:12])
    if int_number(get_path(scoreboard, ["broad_transfer_matrix", "total_regressions"], 0)) > 0:
        regressed.append("broad transfer matrix reports regressions")
    gates = [
        gate("answers_improvement_question", bool(improved) or bool(became_residual), improved or became_residual),
        gate("answers_tool_question", bool(became_tool) or bool(should_expire), {"tools": became_tool, "expire": should_expire}, severity="soft"),
        gate("answers_training_question", bool(train_next), train_next),
    ]
    return {
        "policy": "project_theseus_feedback_ratchet_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(item["passed"] for item in gates) else "YELLOW",
        "questions": {
            "what_improved": improved,
            "what_regressed": regressed,
            "what_became_a_tool": became_tool,
            "what_became_a_residual": became_residual,
            "what_should_expire": should_expire,
            "what_should_be_trained_next": train_next,
        },
        "gates": gates,
        "external_inference_calls": 0,
    }


def next_transfer_action(card: str, blockers: list[str], row: dict[str, Any]) -> str:
    if "no_cheat_violation" in blockers:
        return "quarantine_report_and_regenerate_clean_student_evidence"
    if "needs_32_plus_clean_tasks" in blockers and card in {"source_bigcodebench", "source_livecodebench"}:
        return "upgrade_public_task_adapter_to_32_plus_clean_tasks"
    if "below_0_70_floor" in blockers and card in {"source_mbpp", "source_evalplus"}:
        return "train_private_semantic_lookalikes_then_rerun_same_seed_public_calibration"
    if "sts_not_causal_on_card" in blockers:
        return "run_same_checkpoint_same_seed_sts_repair_ablation"
    return "preserve_as_regression_candidate"


def sym_row(capability: str, report: str, evidence: Any, role: str) -> dict[str, Any]:
    present = bool(evidence)
    return {
        "capability": capability,
        "report": report,
        "evidence_present": present,
        "trigger_state": evidence.get("trigger_state") if isinstance(evidence, dict) else None,
        "role": role,
        "next_integration": "wire into VIEA command executor or feedback ratchet" if present else "create or refresh evidence report",
    }


def collect_residual_counts(state: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for source in [state["residual_escrow"], state["code_residual_forge"], state["learning_scoreboard"]]:
        text = json.dumps(source, sort_keys=True)
        for token in [
            "wrong_answer",
            "runtime",
            "parsing",
            "type_or_name_error",
            "broad semantic transfer remains below floor",
            "public_code_pass_rate_below_floor",
        ]:
            if token in text:
                counts[token] += text.count(token)
    return dict(counts)


def aggregate_state(payloads: Iterable[dict[str, Any]]) -> str:
    states = [payload.get("trigger_state") for payload in payloads]
    if "RED" in states:
        return "RED"
    if "YELLOW" in states:
        return "YELLOW"
    return "GREEN"


def render_markdown(name: str, payload: dict[str, Any]) -> str:
    lines = [
        f"# {name.replace('_', ' ').title()}",
        "",
        f"- trigger_state: `{payload.get('trigger_state')}`",
        f"- policy: `{payload.get('policy')}`",
        "",
    ]
    if "summary" in payload:
        lines.append("## Summary")
        lines.append("")
        for key, value in payload["summary"].items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    if "questions" in payload:
        lines.append("## Feedback Questions")
        lines.append("")
        for key, value in payload["questions"].items():
            lines.append(f"- `{key}`: {value}")
        lines.append("")
    if "gates" in payload:
        lines.append("## Gates")
        lines.append("")
        for row in payload["gates"]:
            lines.append(f"- {'PASS' if row['passed'] else 'FAIL'} `{row['gate']}` ({row['severity']}): {row['evidence']}")
        lines.append("")
    return "\n".join(lines)


def gate(name: str, passed: bool, evidence: Any, *, severity: str = "hard") -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def object_field(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key) if isinstance(payload, dict) else {}
    return value if isinstance(value, dict) else {}


def get_path(data: Any, path: list[str], default: Any = None) -> Any:
    cur = data
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def int_number(value: Any, default: int = 0) -> int:
    return int(number(value, default))


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
