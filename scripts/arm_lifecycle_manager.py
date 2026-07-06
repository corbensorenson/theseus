"""Arm lifecycle governance for SparkStream/RMI.

The Octopus router already emits arm cards and a descriptive lifecycle ledger.
This script turns those artifacts plus real routing traces into an operational
governance report: schema validation, usage telemetry, split/merge/register/
update/deprecate proposals, and long-autonomy readiness signals.

It is report-only by default. The autonomous loop may use the proposals to
queue work or ask the teacher, but this script does not mutate arm definitions.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIFECYCLE_POLICY = ROOT / "configs" / "arm_lifecycle_policy.json"
REQUIRED_ARM_FIELDS = [
    "arm_name",
    "capability_scope",
    "input_schema",
    "output_schema",
    "local_tools",
    "local_memory",
    "permission_tier",
    "permission_boundary",
    "runtime_tier",
    "cost_profile",
    "benchmark_frontier",
    "regression_suite",
    "residual_escrow",
    "reliability_score",
    "lifecycle_status",
    "bloat_index",
    "retirement_criteria",
    "quarantine_domain",
    "dynamic_loading",
]
PROTECTED_ARMS = {"head_router", "safety_reflex_arm"}
HIGH_RISK_PERMISSION_TIERS = {"high", "critical"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-registry", default="reports/arm_registry.json")
    parser.add_argument("--lifecycle-ledger", default="reports/arm_lifecycle_ledger.json")
    parser.add_argument("--routing-memory", default="reports/routing_memory.json")
    parser.add_argument("--real-traces", default="reports/routing_memory_real_traces.jsonl")
    parser.add_argument("--policy", default=str(DEFAULT_LIFECYCLE_POLICY.relative_to(ROOT)))
    parser.add_argument("--out", default="reports/arm_lifecycle_governance.json")
    args = parser.parse_args()

    arm_registry = read_json(ROOT / args.arm_registry, {})
    lifecycle_ledger = read_json(ROOT / args.lifecycle_ledger, {})
    routing_memory = read_json(ROOT / args.routing_memory, {})
    real_traces = read_jsonl(ROOT / args.real_traces)
    lifecycle_policy = read_json(ROOT / args.policy, {})
    protected_arms = set(lifecycle_policy.get("protected_arms") or sorted(PROTECTED_ARMS))

    arms = arm_registry.get("arms") if isinstance(arm_registry, dict) else []
    if not isinstance(arms, list):
        arms = []

    validation = validate_arm_cards(arms)
    usage = build_usage(arms, routing_memory, real_traces)
    proposals = build_proposals(arms, lifecycle_ledger, usage, validation, lifecycle_policy, protected_arms)
    actions = build_arm_actions(arms, lifecycle_ledger, usage, proposals, protected_arms)
    summary = build_summary(arms, lifecycle_ledger, usage, validation, proposals)

    critical_issues = [
        issue
        for issue in validation["issues"]
        if issue.get("severity") == "critical"
    ]
    unknown_selected = usage.get("unknown_selected_arms", [])
    protected_present = protected_arms.issubset({arm.get("arm_name") for arm in arms})
    ready = not critical_issues and not unknown_selected and protected_present

    report = {
        "policy": "sparkstream_arm_lifecycle_governance_v0",
        "created_utc": now(),
        "mode": "report_only",
        "lifecycle_policy": {
            "path": args.policy,
            "mode": lifecycle_policy.get("mode", "report_only"),
            "protected_arms": sorted(protected_arms),
        },
        "ready_for_long_autonomy": ready,
        "ready_for_teacher_enabled_run": ready,
        "summary": summary,
        "validation": validation,
        "usage": usage,
        "proposals": proposals,
        "arm_actions": actions,
        "automation_policy": automation_policy(lifecycle_policy),
        "teacher_escalation": {
            "recommended": any(
                proposal.get("kind") in {"split_arm", "register_arm", "deprecate_arm"}
                and proposal.get("priority") in {"high", "critical"}
                for proposal in proposals
            ),
            "reason": teacher_reason(proposals, validation, usage),
        },
        "external_inference_calls": 0,
    }
    write_json(ROOT / args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if ready else 2


def validate_arm_cards(arms: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_names: list[str] = []
    for idx, arm in enumerate(arms):
        name = str(arm.get("arm_name") or f"arm_index_{idx}")
        if name in seen:
            duplicate_names.append(name)
            issues.append(
                {
                    "severity": "critical",
                    "arm_name": name,
                    "kind": "duplicate_arm_name",
                    "message": "Arm names must be unique for routing, lifecycle, and permissions.",
                }
            )
        seen.add(name)
        missing = [field for field in REQUIRED_ARM_FIELDS if field not in arm]
        if missing:
            issues.append(
                {
                    "severity": "critical",
                    "arm_name": name,
                    "kind": "missing_required_fields",
                    "fields": missing,
                    "message": "Arm card is incomplete.",
                }
            )
        boundary = arm.get("permission_boundary") or {}
        if isinstance(boundary, dict):
            external = boundary.get("external_inference")
            if external not in {"forbidden", "teacher_only_when_allowed_and_budgeted"}:
                issues.append(
                    {
                        "severity": "warning",
                        "arm_name": name,
                        "kind": "external_inference_boundary_unclear",
                        "message": f"external_inference={external!r} should be explicit.",
                    }
                )
            network = boundary.get("network")
            if network not in {"disabled_for_inner_loop", "queued_not_fetched", "off_by_default", None}:
                issues.append(
                    {
                        "severity": "warning",
                        "arm_name": name,
                        "kind": "network_boundary_review",
                        "message": f"network={network!r} should be reviewed before long autonomy.",
                    }
                )
            if arm.get("permission_tier") in HIGH_RISK_PERMISSION_TIERS:
                approvals = set(boundary.get("approval_required_for") or [])
                if not approvals:
                    issues.append(
                        {
                            "severity": "critical",
                            "arm_name": name,
                            "kind": "high_risk_without_approval_gate",
                            "message": "High-risk arms need explicit approval gates.",
                        }
                    )
        else:
            issues.append(
                {
                    "severity": "critical",
                    "arm_name": name,
                    "kind": "permission_boundary_not_object",
                    "message": "permission_boundary must be an object.",
                }
            )

    return {
        "required_fields": REQUIRED_ARM_FIELDS,
        "issue_count": len(issues),
        "critical_count": sum(1 for issue in issues if issue.get("severity") == "critical"),
        "warning_count": sum(1 for issue in issues if issue.get("severity") == "warning"),
        "duplicate_names": duplicate_names,
        "issues": issues,
    }


def build_usage(
    arms: list[dict[str, Any]],
    routing_memory: dict[str, Any],
    real_traces: list[dict[str, Any]],
) -> dict[str, Any]:
    known = {arm.get("arm_name") for arm in arms}
    synthetic_counts: Counter[str] = Counter()
    real_counts: Counter[str] = Counter()
    risk_counts: dict[str, Counter[str]] = defaultdict(Counter)
    last_used: dict[str, str] = {}
    real_trace_success: Counter[str] = Counter()
    real_trace_failure: Counter[str] = Counter()

    for entry in routing_memory.get("entries", []) if isinstance(routing_memory, dict) else []:
        if not isinstance(entry, dict):
            continue
        for arm_name in entry.get("selected_arms") or []:
            synthetic_counts[str(arm_name)] += 1

    for trace in real_traces:
        risk = str(trace.get("risk") or "unknown")
        ok = bool((trace.get("outcome") or {}).get("ok", False))
        created = str(trace.get("created_utc") or "")
        for arm_name in trace.get("selected_arms") or []:
            arm_name = str(arm_name)
            real_counts[arm_name] += 1
            risk_counts[arm_name][risk] += 1
            if created and created > last_used.get(arm_name, ""):
                last_used[arm_name] = created
            if ok:
                real_trace_success[arm_name] += 1
            else:
                real_trace_failure[arm_name] += 1

    all_selected = set(synthetic_counts) | set(real_counts)
    unknown = sorted(arm for arm in all_selected if arm not in known)
    per_arm = []
    for arm in arms:
        name = str(arm.get("arm_name"))
        real = real_counts[name]
        success = real_trace_success[name]
        failure = real_trace_failure[name]
        per_arm.append(
            {
                "arm_name": name,
                "synthetic_routes": synthetic_counts[name],
                "real_routes": real,
                "total_routes": synthetic_counts[name] + real,
                "last_used_utc": last_used.get(name),
                "real_successes": success,
                "real_failures": failure,
                "real_success_rate": safe_div(success, success + failure) if real else None,
                "risk_counts": dict(risk_counts[name]),
            }
        )

    return {
        "routing_memory_entries": len(routing_memory.get("entries", [])) if isinstance(routing_memory, dict) else 0,
        "real_trace_count": len(real_traces),
        "unknown_selected_arms": unknown,
        "selected_arm_counts": dict(Counter(synthetic_counts) + Counter(real_counts)),
        "per_arm": sorted(per_arm, key=lambda item: (-int(item["total_routes"]), item["arm_name"])),
    }


def build_proposals(
    arms: list[dict[str, Any]],
    lifecycle_ledger: dict[str, Any],
    usage: dict[str, Any],
    validation: dict[str, Any],
    lifecycle_policy: dict[str, Any],
    protected_arms: set[str],
) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    arms_by_name = {str(arm.get("arm_name")): arm for arm in arms}
    total_usage = {
        item["arm_name"]: int(item.get("total_routes") or 0)
        for item in usage.get("per_arm", [])
    }

    for issue in validation.get("issues", []):
        if issue.get("severity") == "critical":
            proposals.append(
                {
                    "kind": "repair_arm_card",
                    "priority": "critical",
                    "arm_name": issue.get("arm_name"),
                    "reason": issue.get("kind"),
                    "action": "repair required arm card fields before long autonomous runs",
                    "requires_teacher": False,
                }
            )

    for arm_name in usage.get("unknown_selected_arms", []):
        proposals.append(
            {
                "kind": "register_arm",
                "priority": "high",
                "arm_name": arm_name,
                "reason": "real_or_synthetic_routing_trace_selected_unregistered_arm",
                "action": "create or map an arm card before trusting the route",
                "requires_teacher": True,
            }
        )

    split_rows = []
    split_merge = lifecycle_ledger.get("split_merge_retire") if isinstance(lifecycle_ledger, dict) else {}
    if isinstance(split_merge, dict):
        split_rows = split_merge.get("split_candidates") or []
    lifecycle_rows = lifecycle_ledger.get("lifecycle_rows", []) if isinstance(lifecycle_ledger, dict) else []
    for row in lifecycle_rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("arm_name") or "")
        if row.get("recommended_action") == "inspect_for_split" and name:
            split_rows.append(
                {
                    "arm_name": name,
                    "bloat_index": row.get("bloat_index"),
                    "reason": ",".join(row.get("reasons") or ["inspect_for_split"]),
                }
            )
    seen_split = set()
    for row in split_rows:
        name = str(row.get("arm_name") or "")
        if not name or name in seen_split:
            continue
        seen_split.add(name)
        arm = arms_by_name.get(name, {})
        proposals.append(
            {
                "kind": "split_arm",
                "priority": "high" if total_usage.get(name, 0) > 0 else "medium",
                "arm_name": name,
                "reason": row.get("reason") or "bloat_index_above_threshold",
                "action": "draft sub-arm boundaries, local benchmarks, and migration plan; do not auto-split without review",
                "suggested_children": suggest_split_children(name, arm),
                "requires_teacher": True,
            }
        )

    for row in split_merge.get("merge_inspection", []) if isinstance(split_merge, dict) else []:
        names = [str(name) for name in row.get("arms") or []]
        proposals.append(
            {
                "kind": "merge_inspection",
                "priority": "low",
                "arm_names": names,
                "reason": f"shared_quarantine_domain:{row.get('domain')}",
                "action": "compare usage, tools, memory, and residuals; merge only if specialization value is low",
                "requires_teacher": False,
            }
        )

    for arm in arms:
        name = str(arm.get("arm_name"))
        if name in protected_arms:
            continue
        routes = total_usage.get(name, 0)
        lifecycle = str(arm.get("lifecycle_status") or "")
        reliability = as_float(arm.get("reliability_score"), 1.0)
        if routes == 0 and lifecycle == "active":
            proposals.append(
                {
                    "kind": "monitor_unused_arm",
                    "priority": "low",
                    "arm_name": name,
                    "reason": "no_routes_in_current_window",
                    "action": "keep for now; consider retirement only after a longer unused window and regression review",
                    "requires_teacher": False,
                }
            )
        low_reliability = as_float(
            get_path(lifecycle_policy, ["deprecation_thresholds", "low_reliability"], 0.45),
            0.45,
        )
        if routes == 0 and reliability < low_reliability and lifecycle == "active":
            proposals.append(
                {
                    "kind": "deprecate_arm",
                    "priority": "medium",
                    "arm_name": name,
                    "reason": "unused_and_low_reliability",
                    "action": "move to probation before retirement; preserve benchmarks and residuals",
                    "requires_teacher": True,
                }
            )
        if routes > 0:
            proposals.append(
                {
                    "kind": "update_arm_metrics",
                    "priority": "low",
                    "arm_name": name,
                    "reason": "real_or_synthetic_usage_available",
                    "action": "fold usage counts and success rate into the next arm registry refresh",
                    "requires_teacher": False,
                }
            )
        residual_count = len(arm.get("residual_escrow") or [])
        bloat = as_float(arm.get("bloat_index"), 0.0)
        residual_watch = int(get_path(lifecycle_policy, ["split_thresholds", "watch_residual_count"], 6))
        bloat_watch = as_float(get_path(lifecycle_policy, ["split_thresholds", "watch_bloat_index"], 15), 15.0)
        if residual_count >= residual_watch and bloat >= bloat_watch and name not in seen_split:
            proposals.append(
                {
                    "kind": "watch_split_threshold",
                    "priority": "medium",
                    "arm_name": name,
                    "reason": "large_residual_surface_and_rising_bloat",
                    "action": "cluster residuals; split only if residual families diverge across benchmarks",
                    "requires_teacher": False,
                }
            )

    return proposals[:120]


def build_arm_actions(
    arms: list[dict[str, Any]],
    lifecycle_ledger: dict[str, Any],
    usage: dict[str, Any],
    proposals: list[dict[str, Any]],
    protected_arms: set[str],
) -> list[dict[str, Any]]:
    proposal_by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for proposal in proposals:
        arm_name = proposal.get("arm_name")
        if arm_name:
            proposal_by_arm[str(arm_name)].append(proposal)
        for name in proposal.get("arm_names") or []:
            proposal_by_arm[str(name)].append(proposal)

    rows_by_name = {
        str(row.get("arm_name")): row
        for row in lifecycle_ledger.get("lifecycle_rows", [])
        if isinstance(row, dict)
    } if isinstance(lifecycle_ledger, dict) else {}
    usage_by_name = {
        item["arm_name"]: item
        for item in usage.get("per_arm", [])
        if isinstance(item, dict)
    }
    actions: list[dict[str, Any]] = []
    for arm in arms:
        name = str(arm.get("arm_name"))
        proposals_for_arm = proposal_by_arm.get(name, [])
        kinds = {proposal.get("kind") for proposal in proposals_for_arm}
        if name in protected_arms:
            action = "protect_resident"
        elif "split_arm" in kinds:
            action = "inspect_split"
        elif "deprecate_arm" in kinds:
            action = "probation_before_deprecation"
        elif "monitor_unused_arm" in kinds:
            action = "monitor_unused"
        elif "update_arm_metrics" in kinds:
            action = "refresh_metrics"
        else:
            action = "keep_active"
        actions.append(
            {
                "arm_name": name,
                "action": action,
                "lifecycle_status": arm.get("lifecycle_status"),
                "recommended_action": rows_by_name.get(name, {}).get("recommended_action"),
                "total_routes": usage_by_name.get(name, {}).get("total_routes", 0),
                "real_routes": usage_by_name.get(name, {}).get("real_routes", 0),
                "bloat_index": arm.get("bloat_index"),
                "proposal_kinds": sorted(str(kind) for kind in kinds if kind),
            }
        )
    return actions


def build_summary(
    arms: list[dict[str, Any]],
    lifecycle_ledger: dict[str, Any],
    usage: dict[str, Any],
    validation: dict[str, Any],
    proposals: list[dict[str, Any]],
) -> dict[str, Any]:
    lifecycle_summary = lifecycle_ledger.get("summary", {}) if isinstance(lifecycle_ledger, dict) else {}
    proposal_counts = Counter(str(item.get("kind") or "unknown") for item in proposals)
    priorities = Counter(str(item.get("priority") or "unknown") for item in proposals)
    return {
        "arms": len(arms),
        "active_arms": sum(1 for arm in arms if "active" in str(arm.get("lifecycle_status") or "")),
        "split_candidates": lifecycle_summary.get("split_candidates", 0),
        "merge_inspections": lifecycle_summary.get("merge_inspections", 0),
        "retire_candidates": lifecycle_summary.get("retire_candidates", 0),
        "spawn_recommendations": lifecycle_summary.get("spawn_recommendations", 0),
        "schema_errors": validation.get("critical_count", 0),
        "schema_warnings": validation.get("warning_count", 0),
        "routing_memory_entries": usage.get("routing_memory_entries", 0),
        "real_trace_count": usage.get("real_trace_count", 0),
        "unknown_selected_arm_count": len(usage.get("unknown_selected_arms", [])),
        "proposal_count": len(proposals),
        "proposal_counts": dict(proposal_counts),
        "proposal_priorities": dict(priorities),
    }


def suggest_split_children(name: str, arm: dict[str, Any]) -> list[str]:
    if name == "loop_closure_tool_arm":
        return [
            "trajectory_logger_arm",
            "tool_synthesis_arm",
            "tool_verification_arm",
            "tool_retirement_arm",
        ]
    if name == "babylm_grammar_arm":
        return [
            "babylm_morphology_arm",
            "babylm_binding_arm",
            "babylm_wh_gap_arm",
            "babylm_training_runtime_arm",
        ]
    domain = str(arm.get("quarantine_domain") or "specialist")
    return [f"{domain}_frontier_arm", f"{domain}_verification_arm"]


def teacher_reason(
    proposals: list[dict[str, Any]],
    validation: dict[str, Any],
    usage: dict[str, Any],
) -> str:
    if validation.get("critical_count", 0):
        return "critical_arm_card_validation_issue"
    if usage.get("unknown_selected_arms"):
        return "router_selected_unregistered_arm"
    for proposal in proposals:
        if proposal.get("kind") == "split_arm" and proposal.get("priority") == "high":
            return "high_usage_arm_split_candidate"
    return "no_teacher_needed_for_arm_lifecycle"


def automation_policy(lifecycle_policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "safe_auto_apply": bool(lifecycle_policy.get("safe_auto_apply", False)),
        "allowed_without_teacher": lifecycle_policy.get(
            "allowed_without_teacher",
            [
                "refresh_metrics",
                "monitor_unused_arm",
                "queue_register_unknown_arm",
            ],
        ),
        "requires_teacher_or_human_review": lifecycle_policy.get(
            "requires_teacher_or_human_review",
            [
                "split_arm",
                "merge_arms",
                "deprecate_arm",
                "change_permission_tier",
                "increase_runtime_tier",
            ],
        ),
        "requires_human_approval": lifecycle_policy.get(
            "requires_human_approval",
            [
                "delete_arm",
                "grant_network",
                "grant_external_inference",
                "grant_high_risk_side_effects",
            ],
        ),
    }


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
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


def as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
