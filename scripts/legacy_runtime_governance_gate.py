"""Aggregate predecessor runtime-governance concepts into one launch gate.

BeastBrain's PlanForge, BugBrain's coherence/delirium metric, MoECOT's world
job runtime, and Corben's TaskSpell contracts are all useful only if they affect
autonomy decisions. This report turns those live artifacts into explicit gates.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "legacy_runtime_governance_gate.json"

REQUIRED_PLANFORGE_NODES = {
    "observe_status",
    "proxy_truth_audit",
    "coherence_delirium",
    "active_frontier_pressure",
    "trace_fabric_exchange",
    "taskspell_lock",
    "teacher_self_edit",
    "checkpoint_and_backup",
}
REQUIRED_WORLD_JOB_ACTIONS = {"status", "pause", "resume", "cancel"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--max-delirium", type=float, default=0.35)
    args = parser.parse_args()

    report = build_report(max_delirium=args.max_delirium)
    write_json(resolve(args.out), report)
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(*, max_delirium: float) -> dict[str, Any]:
    planforge = read_json(REPORTS / "planforge_schedule.json")
    coherence = read_json(REPORTS / "coherence_delirium_report.json")
    taskspell = read_json(REPORTS / "taskspell_contracts.json")
    proxy_truth = read_json(REPORTS / "proxy_truth_audit.json")
    world_jobs = read_json(REPORTS / "world_adapter_job_runtime.json")
    temporal_replay = read_json(REPORTS / "temporal_replay_assertions.json")

    plan_nodes = [row for row in planforge.get("nodes", []) if isinstance(row, dict)]
    plan_node_ids = {str(row.get("id") or "") for row in plan_nodes}
    critical_path = [str(item) for item in planforge.get("critical_path", [])]
    contracts = [row for row in taskspell.get("contracts", []) if isinstance(row, dict)]
    jobs = [row for row in world_jobs.get("jobs", []) if isinstance(row, dict)]
    assertions = [row for row in temporal_replay.get("assertions", []) if isinstance(row, dict)]
    delirium_score = float_or(coherence.get("delirium_score"), default=1.0)
    proxy_summary = proxy_truth.get("summary") if isinstance(proxy_truth.get("summary"), dict) else {}

    gates = [
        gate("planforge_ready", planforge.get("status") == "READY", planforge.get("status")),
        gate(
            "planforge_required_nodes_present",
            REQUIRED_PLANFORGE_NODES <= plan_node_ids,
            sorted(REQUIRED_PLANFORGE_NODES - plan_node_ids),
        ),
        gate(
            "planforge_pre_teacher_governance_order",
            appears_before(critical_path, "taskspell_lock", "teacher_self_edit")
            and appears_before(critical_path, "proxy_truth_audit", "teacher_self_edit")
            and appears_before(critical_path, "coherence_delirium", "active_frontier_pressure"),
            critical_path,
        ),
        gate("planforge_no_top_blocker", not planforge.get("top_blocker"), planforge.get("top_blocker")),
        gate("coherence_not_red", coherence.get("trigger_state") != "RED", coherence.get("trigger_state")),
        gate("delirium_below_threshold", delirium_score <= max_delirium, f"{delirium_score} <= {max_delirium}"),
        gate("taskspell_locked", taskspell.get("status") == "LOCKED", taskspell.get("status")),
        gate("taskspell_contract_hash_present", bool(get_path(taskspell, ["summary", "lock_hash"])), get_path(taskspell, ["summary", "lock_hash"])),
        gate("taskspell_acceptance_tests_present", all(row.get("acceptance_tests") for row in contracts), len(contracts)),
        gate("proxy_truth_not_fail_closed", not bool(proxy_truth.get("fail_closed")), proxy_truth.get("fail_closed")),
        gate("proxy_truth_no_failed_artifacts", int_or(proxy_summary.get("failed")) == 0, proxy_summary),
        gate("proxy_truth_external_inference_zero", int_or(proxy_truth.get("external_inference_calls")) == 0, proxy_truth.get("external_inference_calls")),
        gate("world_job_runtime_green_or_yellow", world_jobs.get("trigger_state") in {"GREEN", "YELLOW"}, world_jobs.get("trigger_state")),
        gate("world_jobs_have_control_actions", all(job_actions_ready(row) for row in jobs), [row.get("job_id") for row in jobs if not job_actions_ready(row)]),
        gate("world_jobs_have_replay_ids", world_jobs_have_replay_ids(world_jobs), "adapter replay ids present"),
        gate("live_hardware_not_enabled", not any(row.get("live_hardware_allowed") for row in jobs), [row.get("job_id") for row in jobs if row.get("live_hardware_allowed")]),
        gate("temporal_replay_ready", temporal_replay.get("status") == "READY", temporal_replay.get("status")),
        gate(
            "required_temporal_replay_assertions_ready",
            all(row.get("status") == "READY" for row in assertions if row.get("required")),
            [row for row in assertions if row.get("required") and row.get("status") != "READY"],
        ),
    ]
    failed = [row for row in gates if not row["passed"]]
    warning_count = int_or(proxy_summary.get("warnings")) + (1 if coherence.get("trigger_state") == "YELLOW" else 0)
    trigger_state = "YELLOW" if warning_count else "GREEN"
    if failed:
        trigger_state = "YELLOW"
    if any(row["gate"] in hard_gate_names() for row in failed):
        trigger_state = "RED"
    return {
        "policy": "project_theseus_legacy_runtime_governance_gate_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "ready_for_teacher_work": trigger_state in {"GREEN", "YELLOW"} and not failed,
        "ready_for_candidate_promotion": trigger_state == "GREEN" and warning_count == 0,
        "summary": {
            "planforge_status": planforge.get("status"),
            "planforge_nodes": len(plan_nodes),
            "critical_path": critical_path,
            "coherence_trigger_state": coherence.get("trigger_state"),
            "coherence_score": coherence.get("coherence_score"),
            "delirium_score": delirium_score,
            "taskspell_status": taskspell.get("status"),
            "taskspell_contracts": len(contracts),
            "taskspell_lock_hash": get_path(taskspell, ["summary", "lock_hash"]),
            "proxy_truth_state": proxy_truth.get("trigger_state"),
            "proxy_truth_warnings": int_or(proxy_summary.get("warnings")),
            "world_job_state": world_jobs.get("trigger_state"),
            "world_jobs": len(jobs),
            "temporal_replay_status": temporal_replay.get("status"),
            "temporal_replay_assertions": len(assertions),
            "warning_count": warning_count,
            "failed_gates": [row["gate"] for row in failed],
            "external_inference_calls": 0,
        },
        "gates": gates,
        "runtime_contract": {
            "scheduler": "PlanForge critical path must stage truth/coherence/task locks before teacher or promotion work.",
            "coherence": "High delirium blocks long autonomy and promotion.",
            "taskspell": "Teacher/self-evolution work must carry a stable lock hash and acceptance tests.",
            "world_jobs": "World adapters need pause/resume/cancel/status controls and deterministic replay ids.",
            "temporal_replay": "Required replay assertions must be ready before promotion.",
        },
        "next_actions": next_actions(failed, warning_count),
        "external_inference_calls": 0,
    }


def hard_gate_names() -> set[str]:
    return {
        "planforge_ready",
        "planforge_required_nodes_present",
        "coherence_not_red",
        "delirium_below_threshold",
        "taskspell_locked",
        "taskspell_contract_hash_present",
        "proxy_truth_not_fail_closed",
        "proxy_truth_no_failed_artifacts",
        "world_job_runtime_green_or_yellow",
        "temporal_replay_ready",
        "required_temporal_replay_assertions_ready",
    }


def next_actions(failed: list[dict[str, Any]], warning_count: int) -> list[str]:
    if failed:
        return [f"Fix {row['gate']}: {row['evidence']}" for row in failed[:8]]
    if warning_count:
        return ["Proceed with bounded autonomy, but do not promote candidates until proxy-truth/coherence warnings are cleared."]
    return ["Runtime governance gate is clean for bounded autonomy and candidate-promotion consideration."]


def appears_before(items: list[str], before: str, after: str) -> bool:
    if before not in items or after not in items:
        return False
    return items.index(before) < items.index(after)


def job_actions_ready(job: dict[str, Any]) -> bool:
    return REQUIRED_WORLD_JOB_ACTIONS <= {str(item) for item in job.get("actions", [])}


def world_jobs_have_replay_ids(payload: dict[str, Any]) -> bool:
    matrix = [row for row in payload.get("adapter_coverage_matrix", []) if isinstance(row, dict)]
    return bool(matrix) and all(row.get("deterministic_replay_id") for row in matrix)


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def get_path(payload: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def int_or(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def float_or(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
