"""Governed cell-death/evolution lifecycle pressure for Project Theseus.

This is the anti-bloat layer for arms, suckers, verifier/rule cells, tools,
systems, and private training data. It is deliberately conservative:

* cells expire into renew/improve/split/retire proposals, not silent deletion;
* training data pruning is a plan by default;
* public benchmarks, evaluation data, source manifests, and provenance are
  protected from pruning;
* teacher guidance is architecture-only and never answer-distillation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_POLICY = ROOT / "configs" / "cell_lifecycle_policy.json"
DEFAULT_OUT = REPORTS / "cell_lifecycle.json"
DEFAULT_MARKDOWN_OUT = REPORTS / "cell_lifecycle.md"
DEFAULT_PRUNE_PLAN_OUT = REPORTS / "cell_lifecycle_prune_plan.json"


SYSTEM_REPORTS = [
    ("code_lm_decoder", "system", REPORTS / "code_lm_closure.json"),
    ("rust_code_lm_decoder", "system", REPORTS / "code_lm_closure_rust.json"),
    ("student_learning_closure", "system", REPORTS / "student_learning_closure.json"),
    ("student_first_evidence_audit", "system", REPORTS / "student_first_evidence_audit.json"),
    ("real_code_benchmark_graduation", "system", REPORTS / "real_code_benchmark_graduation.json"),
    ("sts_parallel_streams", "system", REPORTS / "sts_learning_forge.json"),
    ("sts_native_parallel_probe", "system", REPORTS / "sts_native_parallel_probe.json"),
    ("cognitive_context_spaces", "system", REPORTS / "cognitive_context_router.json"),
    ("architecture_guidance_loop", "system", REPORTS / "architecture_guidance_loop.json"),
    ("long_horizon_programming_curriculum", "system", REPORTS / "long_horizon_programming_curriculum.json"),
    ("deterministic_taming_stack", "system", REPORTS / "deterministic_taming_stack.json"),
    ("learning_scoreboard", "system", REPORTS / "learning_scoreboard.json"),
    ("overnight_learning_readiness", "system", REPORTS / "overnight_learning_readiness.json"),
    ("genesis_kernel", "system", REPORTS / "genesis_kernel" / "report.json"),
    ("resource_pantry", "system", REPORTS / "resource_pantry.json"),
    ("training_data_inventory", "system", REPORTS / "training_data_inventory.json"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN_OUT.relative_to(ROOT)))
    parser.add_argument("--prune-plan-out", default=str(DEFAULT_PRUNE_PLAN_OUT.relative_to(ROOT)))
    parser.add_argument("--execute-quarantine", action="store_true")
    parser.add_argument("--max-quarantine-bytes", type=int, default=0)
    args = parser.parse_args()

    policy = read_json(resolve(args.policy), {})
    state = load_state()
    report = build_report(policy, state)
    prune_plan = report["training_data_prune_plan"]
    if args.execute_quarantine:
        execution = execute_quarantine(policy, prune_plan, args.max_quarantine_bytes)
        report["quarantine_execution"] = execution
        prune_plan["quarantine_execution"] = execution

    write_json(resolve(args.out), report)
    write_json(resolve(args.prune_plan_out), prune_plan)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def load_state() -> dict[str, Any]:
    return {
        "arm_registry": read_json(REPORTS / "arm_registry.json", {}),
        "arm_lifecycle": read_json(REPORTS / "arm_lifecycle_governance.json", {}),
        "arm_sucker_registry": read_json(REPORTS / "arm_sucker_registry.json", {}),
        "grammar_suckers": read_json(REPORTS / "grammar_suckers.json", {}),
        "deterministic_taming": read_json(REPORTS / "deterministic_taming_stack.json", {}),
        "tool_registry": read_json(REPORTS / "tool_registry.json", {}),
        "benchmark_ledger": read_json(REPORTS / "benchmark_ledger.json", []),
        "training_inventory": read_json(REPORTS / "training_data_inventory.json", {}),
        "learning_scoreboard": read_json(REPORTS / "learning_scoreboard.json", {}),
        "benchmaxx": read_json(REPORTS / "benchmaxx_curriculum.json", {}),
        "frontier_policy": read_json(REPORTS / "frontier_policy_status.json", {}),
        "architecture_guidance": read_json(REPORTS / "architecture_guidance_loop.json", {}),
        "daemon_events": read_jsonl_tail(REPORTS / "sparkstream_daemon_ledger.jsonl", 300),
        "autonomy_events": read_jsonl_tail(REPORTS / "autonomy_ledger.jsonl", 300),
        "workflow_traces": read_jsonl_tail(REPORTS / "workflow_routing_traces.jsonl", 500),
        "system_reports": {cell_id: read_json(path, {}) for cell_id, _kind, path in SYSTEM_REPORTS},
    }


def build_report(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    cells = build_cells(policy, state)
    mention_text = usage_text_blob(state)
    benchmarks_by_cell = link_benchmarks(cells, state.get("benchmark_ledger", []))
    arm_usage = arm_usage_index(state)

    for cell in cells:
        cell_id = cell["cell_id"]
        base_usage = int(cell.get("usage_count") or 0)
        cell["usage_count"] = base_usage + arm_usage.get(cell_id, 0) + mention_text.count(cell_id.lower())
        cell["linked_benchmarks"] = benchmarks_by_cell.get(cell_id, [])[:12]
        cell["linked_benchmark_count"] = len(benchmarks_by_cell.get(cell_id, []))
        if benchmarks_by_cell.get(cell_id):
            scores = [as_float(row.get("score"), 0.0) for row in benchmarks_by_cell[cell_id]]
            residuals = [as_float(row.get("residual"), 1.0) for row in benchmarks_by_cell[cell_id]]
            cell["best_linked_score"] = round(max(scores), 6)
            cell["worst_linked_residual"] = round(max(residuals), 6)
        decide_cell(policy, cell)

    cells.sort(key=cell_sort_key)
    prune_plan = build_training_data_prune_plan(policy, state)
    tool_creation_pressure = build_tool_creation_pressure(cells, state, prune_plan)
    summary = summarize(cells, prune_plan)
    summary["tool_creation_pressure_count"] = len(tool_creation_pressure)
    trigger_state = "GREEN"
    if summary["retire_candidates"] or summary["improve_candidates"] or summary["split_or_compress_candidates"]:
        trigger_state = "YELLOW"
    if prune_plan["summary"]["unsafe_prune_requests"] > 0:
        trigger_state = "RED"

    return {
        "policy": "project_theseus_cell_lifecycle_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "mode": policy.get("mode", "report_only"),
        "thesis": "Useful cells are renewed; stale or weak cells create improvement pressure; deletion is review-gated.",
        "governance": {
            "report_only": policy.get("mode", "report_only") == "report_only",
            "cell_deletion_requires_human": bool(get_path(policy, ["governance", "delete_arm_requires_human"], True)),
            "training_data_deletion_requires_human": bool(
                get_path(policy, ["data_pruning", "never_delete_without_human_approval"], True)
            ),
            "teacher_role": get_path(policy, ["governance", "teacher_role"], ""),
            "external_inference_calls": 0,
        },
        "summary": summary,
        "cells": cells[: int(policy.get("max_report_cells") or 240)],
        "tool_creation_pressure": tool_creation_pressure,
        "training_data_prune_plan": prune_plan,
        "teacher_escalation": teacher_escalation(cells),
        "external_inference_calls": 0,
    }


def build_cells(policy: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    protected = {str(item) for item in policy.get("protected_cells", [])}
    cells: dict[str, dict[str, Any]] = {}

    for arm in as_list(state.get("arm_registry", {}).get("arms")):
        cell_id = str(arm.get("arm_name") or "")
        if not cell_id:
            continue
        add_cell(
            cells,
            {
                "cell_id": cell_id,
                "kind": "arm",
                "status": arm.get("lifecycle_status") or "unknown",
                "source": "reports/arm_registry.json",
                "score": as_float(arm.get("reliability_score"), None),
                "bloat_index": as_float(arm.get("bloat_index"), 0.0),
                "benchmark_frontier": as_list(arm.get("benchmark_frontier")),
                "retirement_criteria": as_list(arm.get("retirement_criteria")),
                "last_evidence_utc": latest_report_time(state.get("arm_lifecycle")),
            },
        )

    for core in as_list(state.get("arm_sucker_registry", {}).get("cores")):
        cell_id = str(core.get("arm_name") or "")
        if not cell_id:
            continue
        add_cell(
            cells,
            {
                "cell_id": cell_id,
                "kind": "arm",
                "status": core.get("status") or "unknown",
                "source": "reports/arm_sucker_registry.json",
                "score": None,
                "parent": "",
                "last_evidence_utc": latest_report_time(state.get("arm_sucker_registry")),
            },
        )

    for sucker in as_list(state.get("arm_sucker_registry", {}).get("suckers")):
        cell_id = str(sucker.get("sucker_id") or "")
        if not cell_id:
            continue
        add_cell(
            cells,
            {
                "cell_id": cell_id,
                "kind": "sucker",
                "status": sucker.get("status") or "unknown",
                "source": "reports/arm_sucker_registry.json",
                "score": as_float(sucker.get("latest_score"), None),
                "maturity": as_float(sucker.get("maturity"), None),
                "parent": sucker.get("parent_arm"),
                "frontier_families": as_list(sucker.get("frontier_families")),
                "missing_reports": as_list(sucker.get("missing_reports")),
                "last_evidence_utc": latest_report_time(state.get("arm_sucker_registry")),
            },
        )

    for sucker in as_list(state.get("grammar_suckers", {}).get("suckers")):
        cell_id = str(sucker.get("sucker_id") or "")
        if not cell_id:
            continue
        add_cell(
            cells,
            {
                "cell_id": cell_id,
                "kind": "grammar_sucker",
                "status": sucker.get("status") or "unknown",
                "source": "reports/grammar_suckers.json",
                "score": score_from_grammar_evidence(sucker, state.get("grammar_suckers", {})),
                "parent": sucker.get("parent_arm"),
                "language": sucker.get("language"),
                "last_evidence_utc": latest_report_time(state.get("grammar_suckers")),
            },
        )

    for row in as_list(state.get("deterministic_taming", {}).get("arms")):
        sucker_id = str(row.get("sucker") or "")
        cell_id = sucker_id or f"verifier_{row.get('arm')}"
        add_cell(
            cells,
            {
                "cell_id": cell_id,
                "kind": "verifier_sucker",
                "status": "passed" if row.get("passed") else "failed",
                "source": "reports/deterministic_taming_stack.json",
                "score": 1.0 if row.get("passed") else 0.0,
                "parent": row.get("arm"),
                "last_evidence_utc": latest_report_time(state.get("deterministic_taming")),
            },
        )

    for tool in as_list(state.get("tool_registry", {}).get("tools")):
        cell_id = str(tool.get("tool_name") or "")
        if not cell_id:
            continue
        add_cell(
            cells,
            {
                "cell_id": cell_id,
                "kind": "tool",
                "status": tool.get("lifecycle") or "unknown",
                "source": "reports/tool_registry.json",
                "score": as_float(get_path(tool, ["metrics", "success_rate"], None), None),
                "task_family": tool.get("task_family"),
                "retirement_criteria": as_list(tool.get("retirement_criteria")),
                "last_evidence_utc": latest_report_time(state.get("tool_registry")),
            },
        )

    for cell_id, kind, path in SYSTEM_REPORTS:
        report = state.get("system_reports", {}).get(cell_id, {})
        add_cell(
            cells,
            {
                "cell_id": cell_id,
                "kind": kind,
                "status": report.get("trigger_state") or report.get("status") or ("present" if report else "missing"),
                "source": rel(path),
                "score": score_from_system_report(cell_id, report),
                "last_evidence_utc": latest_report_time(report),
            },
        )

    for cell in cells.values():
        cell["protected"] = cell["cell_id"] in protected
        cell["ttl_days"] = ttl_for(policy, cell["kind"])
        cell["expires_utc"] = expiry_for(cell.get("last_evidence_utc"), cell["ttl_days"])
        cell.setdefault("usage_count", 0)
    return list(cells.values())


def add_cell(cells: dict[str, dict[str, Any]], cell: dict[str, Any]) -> None:
    cell_id = str(cell.get("cell_id") or "")
    if not cell_id:
        return
    existing = cells.get(cell_id)
    if not existing:
        cells[cell_id] = cell
        return
    for key, value in cell.items():
        if value in (None, "", [], {}):
            continue
        if key == "source":
            sources = sorted(set(as_list(existing.get("source")) + [str(value)]))
            existing["source"] = sources if len(sources) > 1 else sources[0]
        elif key == "benchmark_frontier":
            existing[key] = sorted(set(as_list(existing.get(key)) + as_list(value)))
        elif key == "last_evidence_utc":
            existing[key] = latest_iso(existing.get(key), value)
        elif key not in existing or existing.get(key) in (None, "", [], {}):
            existing[key] = value


def decide_cell(policy: dict[str, Any], cell: dict[str, Any]) -> None:
    now_dt = datetime.now(timezone.utc)
    expires = parse_dt(cell.get("expires_utc"))
    expired = bool(expires and expires < now_dt)
    score = cell.get("best_linked_score")
    if score is None:
        score = cell.get("score")
    score_f = as_float(score, None)
    usage = int(cell.get("usage_count") or 0)
    bloat = as_float(cell.get("bloat_index"), 0.0)
    status = str(cell.get("status") or "").lower()
    floor = as_float(get_path(policy, ["renewal", "improve_score_floor"], 0.70), 0.70)
    renew_threshold = as_float(get_path(policy, ["renewal", "renew_score_threshold"], 0.70), 0.70)
    high_bloat = as_float(get_path(policy, ["renewal", "high_bloat_index"], 15), 15.0)
    critical_bloat = as_float(get_path(policy, ["renewal", "critical_bloat_index"], 20), 20.0)

    if cell.get("protected"):
        decision = "protect"
        reason = "protected_core_cell"
        next_action = "renew protection and keep metrics fresh"
    elif "planned" in status or status == "missing":
        decision = "improve"
        reason = "cell_not_yet_wired_or_missing"
        next_action = "wire the cell into a measured lane or retire it if not needed"
    elif bloat >= critical_bloat or (bloat >= high_bloat and usage > 0):
        decision = "split_or_compress"
        reason = "bloat_pressure"
        next_action = "cluster usage/residuals, then split, compress, or retire sub-surfaces"
    elif score_f is not None and score_f < floor and usage > 0:
        decision = "improve"
        reason = f"score_below_floor:{score_f:.4f}<{floor:.4f}"
        next_action = "create private residual pressure and architecture experiment before renewal"
    elif expired and usage == 0:
        decision = "retire_candidate"
        reason = "expired_without_recent_usage"
        next_action = "quarantine from active routing after review; preserve report lineage"
    elif expired:
        decision = "probation"
        reason = "expired_but_recently_used"
        next_action = "refresh validation and either renew or improve within next cycle"
    elif usage >= int(get_path(policy, ["renewal", "minimum_usage_for_renew"], 1)) and (
        score_f is None or score_f >= renew_threshold
    ):
        decision = "renew"
        reason = "recent_usage_and_sufficient_score"
        next_action = "extend expiry after next registry refresh"
    else:
        decision = "observe"
        reason = "insufficient_signal"
        next_action = "keep metrics warm; do not spawn duplicates"

    cell["expired"] = expired
    cell["decision"] = decision
    cell["decision_reason"] = reason
    cell["next_action"] = next_action
    cell["teacher_needed"] = decision in {"improve", "split_or_compress", "probation"} and not cell.get("protected")
    cell["destructive_action_allowed"] = False


def build_training_data_prune_plan(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    data_policy = policy.get("data_pruning") if isinstance(policy.get("data_pruning"), dict) else {}
    protected_roles = {str(role) for role in data_policy.get("protected_roles", [])}
    archive_roles = {str(role) for role in data_policy.get("archive_not_delete_roles", [])}
    min_bytes = int(data_policy.get("min_bytes_for_archive_candidate") or 0)
    grace_days = int(data_policy.get("recent_use_grace_days") or 7)
    mastered_hints = mastered_surface_hints(state, data_policy)
    active_hints = active_surface_hints(state)
    rows = collect_data_rows(data_policy, state)
    seen_hash: dict[str, str] = {}
    plan_rows: list[dict[str, Any]] = []
    unsafe = 0

    for row in rows:
        role = str(row.get("role") or "unknown")
        path = str(row.get("path") or "")
        path_l = path.lower().replace("\\", "/")
        modified = parse_dt(row.get("modified_utc"))
        recent = bool(modified and modified > datetime.now(timezone.utc) - timedelta(days=grace_days))
        active = any(hint and hint in path_l for hint in active_hints)
        mastered = any(hint and hint in path_l for hint in mastered_hints)
        duplicate_of = ""
        sha = str(row.get("sha256") or "")
        if sha:
            duplicate_of = seen_hash.get(sha, "")
            seen_hash.setdefault(sha, path)
        decision = "keep"
        reason = "not_prunable"
        if role in protected_roles or "public_benchmark" in role or "evaluation" in role or "holdout" in role:
            decision = "protect"
            reason = "protected_role_or_public_eval_surface"
        elif active:
            decision = "keep_active"
            reason = "active_frontier_or_training_surface"
        elif duplicate_of:
            decision = "archive_candidate"
            reason = f"duplicate_hash_of:{duplicate_of}"
        elif role in archive_roles and mastered and not recent and int(row.get("bytes") or 0) >= min_bytes:
            decision = "archive_candidate"
            reason = "mastered_private_training_surface_can_be_quarantined_after_review"
        elif role in archive_roles and not recent and int(row.get("bytes") or 0) >= min_bytes:
            decision = "observe"
            reason = "private_training_asset_not_recent_but_surface_not_mastered"
        elif role in archive_roles:
            decision = "keep_train"
            reason = "private_training_asset_recent_or_small"
        if decision in {"delete", "prune_now"}:
            unsafe += 1
        plan_rows.append(
            {
                "path": path,
                "role": role,
                "bytes": int(row.get("bytes") or 0),
                "modified_utc": row.get("modified_utc"),
                "sha256": sha,
                "decision": decision,
                "reason": reason,
                "duplicate_of": duplicate_of,
                "requires_human_approval": decision == "archive_candidate",
                "quarantine_not_delete": decision == "archive_candidate",
            }
        )

    counts = Counter(row["decision"] for row in plan_rows)
    candidate_bytes = sum(row["bytes"] for row in plan_rows if row["decision"] == "archive_candidate")
    return {
        "policy": "project_theseus_cell_lifecycle_training_data_prune_plan_v1",
        "created_utc": now(),
        "mode": data_policy.get("mode", "plan_only"),
        "delete_performed": False,
        "quarantine_root": data_policy.get("quarantine_root"),
        "mastered_surface_hints": sorted(mastered_hints),
        "active_surface_hints": sorted(active_hints),
        "summary": {
            "files_considered": len(plan_rows),
            "archive_candidates": counts.get("archive_candidate", 0),
            "archive_candidate_bytes": candidate_bytes,
            "protected": counts.get("protect", 0),
            "keep_active": counts.get("keep_active", 0),
            "keep_train": counts.get("keep_train", 0),
            "unsafe_prune_requests": unsafe,
        },
        "items": sorted(plan_rows, key=lambda item: (item["decision"] != "archive_candidate", -item["bytes"], item["path"]))[:400],
    }


def build_tool_creation_pressure(
    cells: list[dict[str, Any]],
    state: dict[str, Any],
    prune_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    pressure: list[dict[str, Any]] = []
    for cell in cells:
        decision = str(cell.get("decision") or "")
        if decision not in {"improve", "split_or_compress", "probation", "retire_candidate"}:
            continue
        kind = str(cell.get("kind") or "")
        cell_id = str(cell.get("cell_id") or "")
        if kind in {"grammar_sucker", "verifier_sucker"}:
            pressure.append(
                {
                    "pressure_kind": "linter_or_sucker_improvement",
                    "priority": "high" if decision == "improve" else "medium",
                    "source_cell": cell_id,
                    "reason": cell.get("decision_reason"),
                    "suggested_artifact": f"{cell_id}_rule_pack",
                    "suggested_loop_closure": "turn recurring legality failures into deterministic checks before token decoding",
                    "model_burden_reduced": "surface syntax, schema, and admissibility filtering",
                    "must_not_do": "do not insert benchmark answers or hidden public tests",
                }
            )
        elif kind == "arm" and decision == "split_or_compress":
            pressure.append(
                {
                    "pressure_kind": "arm_split_or_compression_tool",
                    "priority": "high",
                    "source_cell": cell_id,
                    "reason": cell.get("decision_reason"),
                    "suggested_artifact": f"{cell_id}_residual_splitter",
                    "suggested_loop_closure": "cluster residuals and route narrow surfaces to suckers before spawning another broad arm",
                    "model_burden_reduced": "routing ambiguity and overloaded specialist context",
                    "must_not_do": "do not delete the parent arm; quarantine only after measured replacement",
                }
            )
        elif kind == "system" and decision in {"improve", "probation"}:
            pressure.append(
                {
                    "pressure_kind": "system_verifier_or_training_tool",
                    "priority": "high" if "code" in cell_id or "real_code" in cell_id else "medium",
                    "source_cell": cell_id,
                    "reason": cell.get("decision_reason"),
                    "suggested_artifact": f"{cell_id}_measured_repair_tool",
                    "suggested_loop_closure": "compile repeated failures into private residual tasks, verifier gates, or STS conditioning traces",
                    "model_burden_reduced": "retry waste and repeated failure reconstruction",
                    "must_not_do": "do not count private or synthetic gains as public promotion",
                }
            )
        elif kind == "tool" and decision in {"improve", "retire_candidate", "probation"}:
            pressure.append(
                {
                    "pressure_kind": "tool_renewal_or_retirement",
                    "priority": "medium",
                    "source_cell": cell_id,
                    "reason": cell.get("decision_reason"),
                    "suggested_artifact": f"{cell_id}_replacement_or_metric_refresh",
                    "suggested_loop_closure": "refresh tool metrics, add smoke tests, or retire after replacement exists",
                    "model_burden_reduced": "manual orchestration and stale workflow selection",
                    "must_not_do": "do not remove a tool that is still referenced by active profiles",
                }
            )

    for row in high_residual_benchmarks(state):
        name = str(row.get("benchmark_name") or "")
        family = str(row.get("benchmark_type") or "")
        residual = as_float(row.get("residual"), 0.0)
        score = as_float(row.get("score"), 0.0)
        pressure.append(
            {
                "pressure_kind": "benchmark_residual_to_tool_or_sucker",
                "priority": "high" if residual and residual >= 0.5 else "medium",
                "source_cell": name,
                "benchmark_type": family,
                "score": score,
                "residual": residual,
                "suggested_artifact": suggested_tool_for_benchmark(name, family),
                "suggested_loop_closure": "mine failed traces for a deterministic checker, private lookalike generator, or arm-specific sucker",
                "model_burden_reduced": "repeated benchmark-family failure handling",
                "must_not_do": "public benchmark failures may inform categories, not answers",
            }
        )

    archive_candidates = int(get_path(prune_plan, ["summary", "archive_candidates"], 0) or 0)
    if archive_candidates:
        pressure.append(
            {
                "pressure_kind": "training_data_compaction_tool",
                "priority": "medium",
                "source_cell": "training_data_prune_plan",
                "archive_candidates": archive_candidates,
                "suggested_artifact": "mastered_private_data_compactor",
                "suggested_loop_closure": "replace mastered bulky private rows with compact regression manifests and replay seeds",
                "model_burden_reduced": "disk bloat and repeated sampling over mastered private surfaces",
                "must_not_do": "never compact public benchmarks, holdouts, or provenance manifests",
            }
        )

    pressure.sort(key=lambda row: (0 if row.get("priority") == "high" else 1, str(row.get("pressure_kind")), str(row.get("source_cell"))))
    return pressure[:64]


def high_residual_benchmarks(state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in as_list(state.get("benchmark_ledger")):
        if not isinstance(row, dict):
            continue
        residual = as_float(row.get("residual"), 0.0) or 0.0
        score = as_float(row.get("score"), 0.0) or 0.0
        lifecycle = str(row.get("lifecycle") or "")
        if residual >= 0.30 and lifecycle == "frontier" and score < 0.70:
            rows.append(row)
    rows.sort(key=lambda item: (-float(item.get("residual") or 0.0), str(item.get("benchmark_name") or "")))
    return rows[:20]


def suggested_tool_for_benchmark(name: str, family: str) -> str:
    text = f"{name} {family}".lower()
    if any(token in text for token in ["evalplus", "human_eval", "mbpp", "bigcodebench", "livecodebench", "code"]):
        return "code_residual_private_task_generator_and_python_ast_linter"
    if any(token in text for token in ["web", "browser"]):
        return "web_task_schema_sucker_and_dom_plan_verifier"
    if any(token in text for token in ["voice", "speech"]):
        return "speech_io_manifest_verifier_and_phoneme_surface_sucker"
    if any(token in text for token in ["drone", "pyflyt", "pybullet"]):
        return "drone_control_safety_sucker_and_waypoint_trace_linter"
    if any(token in text for token in ["minecraft", "crafter", "craftax", "emulator"]):
        return "game_state_action_schema_sucker_and_replay_verifier"
    return "residual_cluster_to_deterministic_checker"


def collect_data_rows(data_policy: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    rows_by_path: dict[str, dict[str, Any]] = {}
    for row in as_list(state.get("training_inventory", {}).get("files")):
        if isinstance(row, dict) and row.get("path"):
            rows_by_path[str(row["path"])] = dict(row)
    for root_text in as_list(data_policy.get("prunable_roots")):
        root = resolve_path_text(str(root_text))
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or should_skip(path):
                continue
            key = rel(path)
            if key not in rows_by_path:
                rows_by_path[key] = data_row(path)
    return list(rows_by_path.values())


def data_row(path: Path) -> dict[str, Any]:
    stat = path.stat()
    row: dict[str, Any] = {
        "path": rel(path),
        "bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "role": role_guess(path),
    }
    if stat.st_size <= 64 * 1024 * 1024:
        row["sha256"] = sha256(path)
    return row


def execute_quarantine(policy: dict[str, Any], prune_plan: dict[str, Any], max_bytes: int) -> dict[str, Any]:
    data_policy = policy.get("data_pruning") if isinstance(policy.get("data_pruning"), dict) else {}
    quarantine_root = resolve_path_text(str(data_policy.get("quarantine_root") or "D:/ProjectTheseus/runtime/quarantine/cell_lifecycle"))
    allowed_roots = [resolve_path_text(str(root)) for root in as_list(data_policy.get("prunable_roots"))]
    moved: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    moved_bytes = 0
    for item in as_list(prune_plan.get("items")):
        if item.get("decision") != "archive_candidate":
            continue
        source = resolve_path_text(str(item.get("path") or ""))
        size = int(item.get("bytes") or 0)
        if not source.exists() or not is_under_any(source, allowed_roots):
            skipped.append({"path": str(item.get("path")), "reason": "not_under_allowed_root_or_missing"})
            continue
        if max_bytes and moved_bytes + size > max_bytes:
            skipped.append({"path": str(item.get("path")), "reason": "max_quarantine_bytes_reached"})
            continue
        root = first_parent_root(source, allowed_roots)
        rel_to_root = source.resolve().relative_to(root.resolve()) if root else Path(source.name)
        dest = quarantine_root / safe_drive_prefix(source) / rel_to_root
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(dest))
        moved_bytes += size
        moved.append({"from": str(source), "to": str(dest), "bytes": size})
    return {
        "executed_utc": now(),
        "mode": "quarantine_move_only",
        "moved_count": len(moved),
        "moved_bytes": moved_bytes,
        "moved": moved,
        "skipped": skipped,
        "delete_performed": False,
    }


def summarize(cells: list[dict[str, Any]], prune_plan: dict[str, Any]) -> dict[str, Any]:
    decisions = Counter(str(cell.get("decision")) for cell in cells)
    kinds = Counter(str(cell.get("kind")) for cell in cells)
    expired = sum(1 for cell in cells if cell.get("expired"))
    teacher_needed = sum(1 for cell in cells if cell.get("teacher_needed"))
    return {
        "cell_count": len(cells),
        "by_kind": dict(kinds),
        "by_decision": dict(decisions),
        "expired_cells": expired,
        "renewed_or_protected": decisions.get("renew", 0) + decisions.get("protect", 0),
        "improve_candidates": decisions.get("improve", 0),
        "split_or_compress_candidates": decisions.get("split_or_compress", 0),
        "retire_candidates": decisions.get("retire_candidate", 0),
        "probation_cells": decisions.get("probation", 0),
        "teacher_architecture_diagnosis_candidates": teacher_needed,
        "training_data_archive_candidates": get_path(prune_plan, ["summary", "archive_candidates"], 0),
        "training_data_archive_candidate_bytes": get_path(prune_plan, ["summary", "archive_candidate_bytes"], 0),
    }


def teacher_escalation(cells: list[dict[str, Any]]) -> dict[str, Any]:
    targets = [
        {
            "cell_id": cell.get("cell_id"),
            "kind": cell.get("kind"),
            "decision": cell.get("decision"),
            "reason": cell.get("decision_reason"),
            "next_action": cell.get("next_action"),
        }
        for cell in cells
        if cell.get("teacher_needed")
    ][:12]
    return {
        "recommended": bool(targets),
        "mode": "architecture_guidance_only",
        "targets": targets,
        "prompt_contract": "Teacher may diagnose missing architecture, verifier, routing, or data pressure. It may not provide benchmark answers, hidden tests, or direct task solutions.",
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Cell Lifecycle",
        "",
        f"State: **{report.get('trigger_state')}**",
        "",
        "This report is anti-bloat pressure, not silent deletion. Cells expire into renewal, improvement, split/compress, probation, or retirement proposals.",
        "",
        "## Summary",
        "",
        f"- Cells: {summary.get('cell_count')}",
        f"- Renew/protect: {summary.get('renewed_or_protected')}",
        f"- Improve candidates: {summary.get('improve_candidates')}",
        f"- Split/compress candidates: {summary.get('split_or_compress_candidates')}",
        f"- Retire candidates: {summary.get('retire_candidates')}",
        f"- Tool creation pressure items: {summary.get('tool_creation_pressure_count')}",
        f"- Data archive candidates: {summary.get('training_data_archive_candidates')} ({summary.get('training_data_archive_candidate_bytes')} bytes)",
        "",
        "## Highest Pressure Cells",
        "",
    ]
    pressure = [
        cell
        for cell in report.get("cells", [])
        if cell.get("decision") in {"improve", "split_or_compress", "probation", "retire_candidate"}
    ][:16]
    if not pressure:
        lines.append("- No high-pressure cells right now.")
    for cell in pressure:
        lines.append(
            f"- `{cell.get('cell_id')}` ({cell.get('kind')}): {cell.get('decision')} - {cell.get('decision_reason')}"
        )
    lines.extend(["", "## Tool Creation Pressure", ""])
    tool_pressure = report.get("tool_creation_pressure", [])[:16]
    if not tool_pressure:
        lines.append("- No tool creation pressure right now.")
    for row in tool_pressure:
        lines.append(
            f"- `{row.get('source_cell')}` -> `{row.get('suggested_artifact')}` ({row.get('pressure_kind')}, {row.get('priority')})"
        )
    lines.extend(["", "## Data Prune Plan", ""])
    prune = report.get("training_data_prune_plan", {})
    prune_summary = prune.get("summary", {})
    lines.append(
        f"- Archive candidates: {prune_summary.get('archive_candidates')} ({prune_summary.get('archive_candidate_bytes')} bytes)"
    )
    lines.append("- Delete performed: false")
    lines.append("- Default action: quarantine proposal only; human approval required.")
    return "\n".join(lines) + "\n"


def arm_usage_index(state: dict[str, Any]) -> dict[str, int]:
    index: dict[str, int] = {}
    for row in as_list(get_path(state, ["arm_lifecycle", "usage", "per_arm"], [])):
        name = str(row.get("arm_name") or "")
        if name:
            index[name] = int(row.get("total_routes") or 0)
    return index


def usage_text_blob(state: dict[str, Any]) -> str:
    rows = []
    for key in ["daemon_events", "autonomy_events", "workflow_traces"]:
        rows.extend(as_list(state.get(key)))
    return "\n".join(json.dumps(row, sort_keys=True, default=str).lower() for row in rows if isinstance(row, dict))


def link_benchmarks(cells: list[dict[str, Any]], ledger: Any) -> dict[str, list[dict[str, Any]]]:
    rows = as_list(ledger)
    linked: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        keywords = cell_keywords(cell)
        frontier = {str(item).lower() for item in as_list(cell.get("benchmark_frontier"))}
        families = {str(item).lower() for item in as_list(cell.get("frontier_families"))}
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("benchmark_name") or "").lower()
            btype = str(row.get("benchmark_type") or "").lower()
            text = f"{name} {btype}"
            if any(keyword and keyword in text for keyword in keywords | frontier | families):
                linked[cell["cell_id"]].append(
                    {
                        "benchmark_name": row.get("benchmark_name"),
                        "benchmark_type": row.get("benchmark_type"),
                        "lifecycle": row.get("lifecycle"),
                        "score": row.get("score"),
                        "residual": row.get("residual"),
                        "curriculum_status": row.get("curriculum_status"),
                    }
                )
    return linked


def cell_keywords(cell: dict[str, Any]) -> set[str]:
    cell_id = str(cell.get("cell_id") or "").lower()
    kind = str(cell.get("kind") or "").lower()
    words = {part for part in cell_id.replace("-", "_").split("_") if len(part) >= 4}
    if "code" in cell_id or "python" in cell_id:
        words.update({"code", "coding", "evalplus", "human_eval", "mbpp", "bigcodebench", "livecodebench"})
    if "grammar" in cell_id or "english" in cell_id or "sbl" in cell_id:
        words.update({"grammar", "babylm", "blimp", "language"})
    if "minecraft" in cell_id or "crafter" in cell_id:
        words.update({"minecraft", "crafter", "craftax"})
    if "drone" in cell_id or "pyflyt" in cell_id:
        words.update({"drone", "pyflyt", "pybullet"})
    if "voice" in cell_id:
        words.update({"voice", "speech"})
    if kind == "tool":
        words.add(str(cell.get("task_family") or "").lower())
    return words


def score_from_grammar_evidence(sucker: dict[str, Any], grammar: dict[str, Any]) -> float | None:
    sid = str(sucker.get("sucker_id") or "")
    summary = grammar.get("summary") if isinstance(grammar.get("summary"), dict) else {}
    if sid == "python_grammar_sucker":
        return as_float(summary.get("python_parse_pass_rate"), None)
    if sid == "english_surface_grammar_sucker":
        return as_float(summary.get("english_surface_pass_rate"), None)
    if sid == "sbl_semantic_backbone_sucker":
        return 1.0 if summary.get("legacy_sbl_found") else 0.0
    if "planned" in str(sucker.get("status") or ""):
        return 0.0
    return None


def score_from_system_report(cell_id: str, report: dict[str, Any]) -> float | None:
    if not report:
        return 0.0
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    if cell_id in {"code_lm_decoder", "rust_code_lm_decoder"}:
        return as_float(summary.get("private_trained_pass_rate"), None)
    if cell_id == "real_code_benchmark_graduation":
        return as_float(summary.get("real_public_task_pass_rate"), None)
    if cell_id == "sts_native_parallel_probe":
        return as_float(summary.get("after_eval_token_accuracy"), None)
    if report.get("trigger_state") == "GREEN":
        return 1.0
    if report.get("trigger_state") == "YELLOW":
        return 0.65
    if report.get("trigger_state") == "RED":
        return 0.0
    return None


def mastered_surface_hints(state: dict[str, Any], data_policy: dict[str, Any]) -> set[str]:
    threshold = as_float(data_policy.get("mastered_score_threshold"), 0.9)
    hints: set[str] = set()
    for row in as_list(state.get("benchmark_ledger")):
        if not isinstance(row, dict):
            continue
        score = as_float(row.get("score"), 0.0)
        lifecycle = str(row.get("lifecycle") or "").lower()
        saturation = str(row.get("saturation_status") or "").lower()
        curriculum = str(row.get("curriculum_status") or "").lower()
        if score >= threshold or lifecycle in {"regression", "mastered"} or saturation in {"saturated", "mastered"} or "regression" in curriculum:
            hints.update(path_hints(str(row.get("benchmark_name") or "")))
    return {hint for hint in hints if hint}


def active_surface_hints(state: dict[str, Any]) -> set[str]:
    hints = set()
    for value in [
        get_path(state, ["learning_scoreboard", "frontier_truth", "pressure_card_id"], ""),
        get_path(state, ["learning_scoreboard", "frontier_truth", "benchmaxx_next_card"], ""),
        get_path(state, ["frontier_policy", "pressure_card_id"], ""),
        get_path(state, ["benchmaxx", "next_frontier", "recommended_env"], ""),
    ]:
        hints.update(path_hints(str(value)))
    return {hint for hint in hints if hint}


def path_hints(text: str) -> set[str]:
    raw = text.lower().replace("-", "_").replace("/", "_")
    parts = {part for part in raw.split("_") if len(part) >= 4}
    if "evalplus" in raw or "eval_plus" in raw:
        parts.add("evalplus")
    if "human_eval" in raw or "humaneval" in raw:
        parts.update({"human_eval", "humaneval"})
    if "bigcodebench" in raw:
        parts.add("bigcodebench")
    if "livecodebench" in raw:
        parts.add("livecodebench")
    return parts


def role_guess(path: Path) -> str:
    text = rel(path).lower().replace("\\", "/")
    if "d:/projecttheseus/training_data/open_code_pantry/private_train" in text:
        return "permissive_open_code_train_only"
    if "d:/projecttheseus/training_data/open_conversation_pantry/private_train" in text:
        return "permissive_open_conversation_train_only"
    if "d:/projecttheseus/training_data/open_conversation_pantry/sts_streams" in text:
        return "permissive_open_conversation_sts_streams"
    if "d:/projecttheseus/training_data/residual_code_curriculum" in text:
        return "private_residual_code_train_only" if "private_train" in text else "private_residual_code_curriculum_asset"
    if "data/sts_learning" in text:
        if "cognitive_context" in text or "context_spaces" in text:
            return "cognitive_context_sts_train_eval_data"
        return "sts_parallel_stream_train_eval_data"
    if "data/synthetic" in text:
        return "synthetic_training_data"
    return "data_asset"


def ttl_for(policy: dict[str, Any], kind: str) -> int:
    default = policy.get("default_ttl_days") if isinstance(policy.get("default_ttl_days"), dict) else {}
    return int(default.get(kind, default.get("system", 7)))


def expiry_for(created_utc: Any, ttl_days: int) -> str:
    base = parse_dt(created_utc) or datetime.now(timezone.utc)
    return (base + timedelta(days=ttl_days)).isoformat()


def latest_report_time(report: Any) -> str:
    if not isinstance(report, dict):
        return ""
    for key in ["created_utc", "updated_utc", "completed_utc", "timestamp_utc"]:
        if report.get(key):
            return str(report.get(key))
    return ""


def latest_iso(a: Any, b: Any) -> str:
    da = parse_dt(a)
    db = parse_dt(b)
    if da and db:
        return (da if da >= db else db).isoformat()
    return str(a or b or "")


def cell_sort_key(cell: dict[str, Any]) -> tuple[int, int, str]:
    priority = {
        "improve": 0,
        "split_or_compress": 1,
        "probation": 2,
        "retire_candidate": 3,
        "observe": 4,
        "renew": 5,
        "protect": 6,
    }.get(str(cell.get("decision")), 9)
    return (priority, -int(cell.get("usage_count") or 0), str(cell.get("cell_id")))


def parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        if "." in text:
            head, tail = text.split(".", 1)
            digits = []
            rest = ""
            for idx, char in enumerate(tail):
                if char.isdigit():
                    digits.append(char)
                    continue
                rest = tail[idx:]
                break
            else:
                rest = ""
            if len(digits) > 6:
                text = head + "." + "".join(digits[:6]) + rest
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def read_jsonl_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[-limit:]
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def resolve_path_text(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def as_float(value: Any, default: float | None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def should_skip(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return "__pycache__" in parts or ".git" in parts or path.suffix.lower() in {".pyc", ".pyo"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_under_any(path: Path, roots: list[Path]) -> bool:
    resolved = path.resolve()
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def first_parent_root(path: Path, roots: list[Path]) -> Path | None:
    resolved = path.resolve()
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return root
        except ValueError:
            continue
    return None


def safe_drive_prefix(path: Path) -> str:
    drive = path.drive.replace(":", "").replace("\\", "")
    return drive or "workspace"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
