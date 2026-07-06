"""Plan legacy-source training pressure against the low-rank adapter bank.

This is a dry-run planner, not a trainer. It joins the admitted legacy tiny
sample, governed trace-fabric rows, TaskSpell lock, runtime governance, and the
low-rank adapter bank into one explicit lane plan so adapter concepts do not
sit as an unconnected manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK = ROOT / "reports" / "low_rank_adapter_bank.json"
DEFAULT_SAMPLE = ROOT / "reports" / "legacy_training_source_sample.json"
DEFAULT_TRACE = ROOT / "reports" / "trace_fabric_capsule_materialization.json"
DEFAULT_TASKSPELL = ROOT / "reports" / "taskspell_contracts.json"
DEFAULT_RUNTIME = ROOT / "reports" / "legacy_runtime_governance_gate.json"
DEFAULT_OUT = ROOT / "reports" / "legacy_adapter_bank_training_plan.json"
DEFAULT_PLAN_OUT = ROOT / "data" / "training_sources" / "legacy_adapter_bank_dry_run_plan.jsonl"

LANE_ADAPTER_HINTS: dict[str, list[str]] = {
    "code_repair": [
        "edge_code_repair_verifier_to_benchmark_adapter_factory",
        "code_repair_prior",
    ],
    "code_agent_trace_governance": [
        "edge_code_repair_verifier_to_benchmark_adapter_factory",
        "code_repair_prior",
    ],
    "semantic_compiler": ["grammar_residual_prior"],
    "tool_api_use": ["web_task_prior"],
    "rl_environment_trace_governance": ["drone_control_prior"],
}

ZERO_PARAM_BASELINE_LANES = {
    "instruction_sft",
    "math_reasoning",
    "paper_synthesis",
    "proof_verification",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", default=str(DEFAULT_BANK.relative_to(ROOT)))
    parser.add_argument("--legacy-sample", default=str(DEFAULT_SAMPLE.relative_to(ROOT)))
    parser.add_argument("--trace-materialization", default=str(DEFAULT_TRACE.relative_to(ROOT)))
    parser.add_argument("--taskspell", default=str(DEFAULT_TASKSPELL.relative_to(ROOT)))
    parser.add_argument("--runtime-governance", default=str(DEFAULT_RUNTIME.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--plan-out", default=str(DEFAULT_PLAN_OUT.relative_to(ROOT)))
    parser.add_argument("--max-train-rows-per-lane", type=int, default=32)
    parser.add_argument("--max-eval-rows-per-lane", type=int, default=16)
    args = parser.parse_args()

    report = build_report(
        bank_path=resolve(args.bank),
        legacy_sample_path=resolve(args.legacy_sample),
        trace_path=resolve(args.trace_materialization),
        taskspell_path=resolve(args.taskspell),
        runtime_path=resolve(args.runtime_governance),
        plan_out=resolve(args.plan_out),
        max_train_rows_per_lane=max(1, args.max_train_rows_per_lane),
        max_eval_rows_per_lane=max(1, args.max_eval_rows_per_lane),
    )
    write_json(resolve(args.out), report)
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(
    *,
    bank_path: Path,
    legacy_sample_path: Path,
    trace_path: Path,
    taskspell_path: Path,
    runtime_path: Path,
    plan_out: Path,
    max_train_rows_per_lane: int,
    max_eval_rows_per_lane: int,
) -> dict[str, Any]:
    bank = read_json(bank_path)
    legacy_sample = read_json(legacy_sample_path)
    trace_materialization = read_json(trace_path)
    taskspell = read_json(taskspell_path)
    runtime = read_json(runtime_path)

    adapters = [row for row in bank.get("adapters", []) if isinstance(row, dict)]
    adapters_by_id = {str(row.get("id")): row for row in adapters if row.get("id")}
    max_allowed_interference = float(get_path(bank, ["summary", "max_interference_allowed"], 0.03) or 0.03)
    max_seen_interference = max_adapter_interference(bank)
    taskspell_lock_hash = str(get_path(taskspell, ["summary", "lock_hash"], "") or "")

    rows = build_lane_rows(
        legacy_sample=legacy_sample,
        trace_materialization=trace_materialization,
        adapters_by_id=adapters_by_id,
        taskspell_lock_hash=taskspell_lock_hash,
        max_train_rows_per_lane=max_train_rows_per_lane,
        max_eval_rows_per_lane=max_eval_rows_per_lane,
    )
    write_jsonl(plan_out, rows)

    adapter_use_counts = Counter(str(row["adapter_use_state"]) for row in rows)
    selected_adapters = sorted(
        {
            str(row.get("selected_adapter_id"))
            for row in rows
            if row.get("selected_adapter_id")
        }
    )
    planned_lanes = [
        row["lane"]
        for row in rows
        if row.get("adapter_status") == "planned"
    ]
    zero_param_lanes = [
        row["lane"]
        for row in rows
        if row.get("adapter_use_state") == "zero_param_baseline_no_adapter"
    ]
    hard_gates = [
        gate("adapter_bank_report_present", bool(bank), rel_or_abs(bank_path)),
        gate("adapter_bank_ready", bank.get("status") == "READY", bank.get("status")),
        gate("zero_param_first_preserved", bank.get("zero_param_first") is True, bank.get("zero_param_first")),
        gate(
            "legacy_training_sample_present",
            legacy_sample.get("trigger_state") in {"GREEN", "YELLOW"}
            and int(get_path(legacy_sample, ["summary", "sample_rows"], 0) or 0) > 0,
            f"state={legacy_sample.get('trigger_state')} rows={get_path(legacy_sample, ['summary', 'sample_rows'], None)}",
        ),
        gate(
            "trace_materialization_governed",
            trace_materialization.get("trigger_state") in {"GREEN", "YELLOW"}
            and int(get_path(trace_materialization, ["summary", "raw_payload_rows"], 0) or 0) == 0
            and int(get_path(trace_materialization, ["summary", "external_inference_calls"], 0) or 0) == 0,
            f"state={trace_materialization.get('trigger_state')} raw={get_path(trace_materialization, ['summary', 'raw_payload_rows'], None)}",
        ),
        gate("taskspell_locked", taskspell.get("status") == "LOCKED", taskspell.get("status")),
        gate("taskspell_lock_hash_present", bool(taskspell_lock_hash), taskspell_lock_hash),
        gate("dry_run_plan_rows_written", len(rows) > 0, f"rows={len(rows)} path={rel(plan_out)}"),
        gate(
            "adapter_interference_below_threshold",
            max_seen_interference <= max_allowed_interference,
            f"{max_seen_interference} <= {max_allowed_interference}",
        ),
        gate(
            "planned_adapters_not_activated",
            all(
                row.get("adapter_status") != "planned"
                or row.get("adapter_weight_activation") == "not_allowed_until_artifact_materialized"
                for row in rows
            ),
            planned_lanes,
        ),
        gate(
            "missing_adapters_reported_as_zero_param",
            all(
                row.get("selected_adapter_id")
                or row.get("adapter_use_state") == "zero_param_baseline_no_adapter"
                for row in rows
            ),
            zero_param_lanes,
        ),
        gate("external_inference_zero", True, "local report joins only"),
    ]
    advisory_gates = [
        gate(
            "runtime_teacher_work_ready_for_teacher_targeting",
            runtime.get("trigger_state") in {"GREEN", "YELLOW"} and bool(runtime.get("ready_for_teacher_work")),
            f"state={runtime.get('trigger_state')} teacher={runtime.get('ready_for_teacher_work')}",
        )
    ]
    hard_failed = [row["gate"] for row in hard_gates if not row["passed"]]
    advisory_failed = [row["gate"] for row in advisory_gates if not row["passed"]]
    ready_for_zero_param_dry_run = not hard_failed
    ready_for_adapter_activation = ready_for_zero_param_dry_run and not planned_lanes and not zero_param_lanes
    trigger_state = "RED" if hard_failed else ("GREEN" if ready_for_adapter_activation else "YELLOW")

    return {
        "policy": "project_theseus_legacy_adapter_bank_training_plan_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "ready_for_zero_param_dry_run": ready_for_zero_param_dry_run,
        "ready_for_adapter_activation": ready_for_adapter_activation,
        "plan_path": rel(plan_out),
        "inputs": {
            "adapter_bank": rel_or_abs(bank_path),
            "legacy_training_sample": rel_or_abs(legacy_sample_path),
            "trace_materialization": rel_or_abs(trace_path),
            "taskspell": rel_or_abs(taskspell_path),
            "runtime_governance": rel_or_abs(runtime_path),
        },
        "summary": {
            "plan_rows": len(rows),
            "legacy_training_rows": int(get_path(legacy_sample, ["summary", "sample_rows"], 0) or 0),
            "trace_materialized_rows": int(get_path(trace_materialization, ["summary", "materialized_rows"], 0) or 0),
            "source_lane_count": len(rows),
            "selected_adapters": selected_adapters,
            "adapter_use_counts": dict(adapter_use_counts),
            "planned_adapter_lanes": planned_lanes,
            "zero_param_lanes": zero_param_lanes,
            "advisory_failed": advisory_failed,
            "max_seen_interference": max_seen_interference,
            "max_allowed_interference": max_allowed_interference,
            "taskspell_lock_hash": taskspell_lock_hash,
            "external_inference_calls": 0,
        },
        "lane_plans": rows,
        "gates": hard_gates + advisory_gates,
        "usage_policy": {
            "dry_run_only": True,
            "zero_param_first": True,
            "internal_training_only": True,
            "not_public_benchmark_claim_evidence": True,
            "adapter_weight_activation_requires_ready_artifact": True,
            "adapter_weight_activation_requires_ablation_and_regression_floors": True,
            "raw_trace_payloads_not_consumed": True,
            "external_inference_calls": 0,
        },
        "next_actions": next_actions(trigger_state, planned_lanes, zero_param_lanes),
        "external_inference_calls": 0,
    }


def build_lane_rows(
    *,
    legacy_sample: dict[str, Any],
    trace_materialization: dict[str, Any],
    adapters_by_id: dict[str, dict[str, Any]],
    taskspell_lock_hash: str,
    max_train_rows_per_lane: int,
    max_eval_rows_per_lane: int,
) -> list[dict[str, Any]]:
    source_rows: list[tuple[str, str, int]] = []
    for lane, count in sorted((get_path(legacy_sample, ["summary", "lane_counts"], {}) or {}).items()):
        source_rows.append(("legacy_training_sample", str(lane), int(count or 0)))
    for lane, count in sorted((get_path(trace_materialization, ["summary", "lane_counts"], {}) or {}).items()):
        source_rows.append(("trace_capsule_materialized", str(lane), int(count or 0)))

    rows = []
    for source_family, lane, source_count in source_rows:
        adapter = choose_adapter(lane, adapters_by_id)
        adapter_status = str(adapter.get("status") or "") if adapter else ""
        selected_adapter_id = str(adapter.get("id") or "") if adapter else ""
        interference_risk = float(adapter.get("interference_risk") or 0.0) if adapter else 0.0
        adapter_use_state = classify_adapter_use(adapter, lane)
        train_budget = min(source_count, max_train_rows_per_lane)
        eval_budget = min(max(1, max(2, source_count // 4)), max_eval_rows_per_lane) if source_count else 0
        rows.append(
            {
                "plan_id": stable_id(f"{source_family}:{lane}:{source_count}:{selected_adapter_id}"),
                "dataset_id": "dataset.legacy_adapter_bank_dry_run_plan.v1",
                "source_type": "legacy_adapter_bank_dry_run_plan",
                "source_family": source_family,
                "lane": lane,
                "source_rows": source_count,
                "train_row_budget": train_budget,
                "eval_row_budget": eval_budget,
                "selected_adapter_id": selected_adapter_id,
                "adapter_status": adapter_status,
                "adapter_use_state": adapter_use_state,
                "adapter_rank": adapter.get("rank") if adapter else None,
                "adapter_features": adapter.get("features") if adapter else [],
                "interference_risk": interference_risk,
                "requires_ablation": bool(adapter),
                "adapter_weight_activation": adapter_activation(adapter_status, adapter_use_state),
                "recommendation": lane_recommendation(lane, source_family, adapter_status, adapter_use_state),
                "governance": {
                    "dry_run_only": True,
                    "zero_param_first": True,
                    "taskspell_lock_hash": taskspell_lock_hash,
                    "internal_training_only": True,
                    "not_public_benchmark_claim_evidence": True,
                    "external_inference_calls": 0,
                    "raw_trace_payloads_consumed": False,
                },
            }
        )
    return rows


def choose_adapter(lane: str, adapters_by_id: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    for adapter_id in LANE_ADAPTER_HINTS.get(lane, []):
        adapter = adapters_by_id.get(adapter_id)
        if adapter:
            return adapter
    return None


def classify_adapter_use(adapter: dict[str, Any] | None, lane: str) -> str:
    if not adapter:
        return "zero_param_baseline_no_adapter"
    status = str(adapter.get("status") or "")
    if status == "ready":
        return "ready_adapter_eval_only"
    if status == "planned":
        return "planned_adapter_manifest_only"
    if lane in ZERO_PARAM_BASELINE_LANES:
        return "zero_param_baseline_no_adapter"
    return "adapter_present_unknown_status"


def adapter_activation(adapter_status: str, adapter_use_state: str) -> str:
    if adapter_use_state == "zero_param_baseline_no_adapter":
        return "not_applicable"
    if adapter_status == "ready":
        return "eval_only_until_ablation_and_regression_floors_pass"
    if adapter_status == "planned":
        return "not_allowed_until_artifact_materialized"
    return "blocked_until_status_is_ready"


def lane_recommendation(lane: str, source_family: str, adapter_status: str, adapter_use_state: str) -> str:
    if adapter_use_state == "zero_param_baseline_no_adapter":
        return "run zero-parameter baseline pressure and treat adapter creation as optional follow-up"
    if adapter_status == "ready":
        return "compare zero-parameter baseline against eval-only adapter routing before any weight activation"
    if adapter_status == "planned":
        return "keep this as a manifest-level prior until the adapter artifact and regression floor are materialized"
    return f"review adapter status before routing {source_family}:{lane}"


def next_actions(trigger_state: str, planned_lanes: list[str], zero_param_lanes: list[str]) -> list[str]:
    if trigger_state == "RED":
        return ["Fix failed adapter-bank planning gates before using legacy source pressure."]
    actions = [
        "Run a zero-parameter tiny dry-run across all planned lanes before activating adapter weights.",
        "Use ready adapters only in eval-only routing until ablations and regression floors pass.",
    ]
    if planned_lanes:
        actions.append("Materialize planned adapter artifacts for " + ", ".join(sorted(set(planned_lanes))) + ".")
    if zero_param_lanes:
        actions.append("Keep no-adapter lanes as control baselines: " + ", ".join(sorted(set(zero_param_lanes))) + ".")
    return actions


def max_adapter_interference(bank: dict[str, Any]) -> float:
    values = [
        float(row.get("estimated_interference") or 0.0)
        for row in bank.get("interference_matrix", [])
        if isinstance(row, dict)
    ]
    adapters = [
        float(row.get("interference_risk") or 0.0)
        for row in bank.get("adapters", [])
        if isinstance(row, dict)
    ]
    return max(values + adapters + [0.0])


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:24]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def rel_or_abs(path: Path) -> str:
    try:
        return rel(path)
    except ValueError:
        return str(path)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
