"""Cross-check old-project ports against live Theseus evidence.

The concept map can say a legacy idea is "done", but that only means it has a
target surface. This audit checks whether the important old-project ports are
still backed by current reports, admissions, gates, and runtime evidence.
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/legacy_port_completion_audit.json")
    parser.add_argument("--markdown-out", default="reports/legacy_port_completion_audit.md")
    args = parser.parse_args()

    state = load_state()
    report = build_report(state)
    write_json(resolve(args.out), report)
    write_markdown(resolve(args.markdown_out), report)
    print(json.dumps(report, indent=2))
    return 0 if report["port_coverage_state"] == "GREEN" else 2


def load_state() -> dict[str, Any]:
    return {
        "concept_map": read_json(ROOT / "configs" / "legacy_concept_port_map.json"),
        "concept_audit": read_json(REPORTS / "legacy_project_concept_audit.json"),
        "mechanisms": read_json(REPORTS / "legacy_port_mechanisms.json"),
        "runtime": read_json(REPORTS / "legacy_port_runtime_enforcement.json"),
        "old_registry": read_json(REPORTS / "old_project_registry_port.json"),
        "training_audit": read_json(REPORTS / "legacy_training_source_audit.json"),
        "training_sample": read_json(REPORTS / "legacy_training_source_sample.json"),
        "rl_admission": read_json(REPORTS / "legacy_rl_environment_admission.json"),
        "rl_smoke_plan": read_json(REPORTS / "legacy_rl_smoke_plan.json"),
        "trace_admission": read_json(REPORTS / "trace_fabric_capsule_admission.json"),
        "trace_materialization": read_json(REPORTS / "trace_fabric_capsule_materialization.json"),
        "adapter_plan": read_json(REPORTS / "legacy_adapter_bank_training_plan.json"),
        "active_inference": read_json(REPORTS / "legacy_active_inference_pilot.json"),
        "launch": read_json(REPORTS / "autonomy_launch_readiness.json"),
    }


def build_report(state: dict[str, Any]) -> dict[str, Any]:
    concept_map = state["concept_map"]
    candidates = concept_map.get("port_candidates") if isinstance(concept_map.get("port_candidates"), list) else []
    status_counts = Counter(str(row.get("status") or "unknown") for row in candidates if isinstance(row, dict))
    priority_counts = Counter(str(row.get("priority") or "unknown") for row in candidates if isinstance(row, dict))
    p0_open = [
        row.get("id")
        for row in candidates
        if isinstance(row, dict) and row.get("priority") == "P0" and row.get("status") != "done"
    ]

    checks = [
        check(
            "legacy_source_projects_present",
            get_path(state, ["concept_audit", "summary", "projects_present"], 0) == get_path(state, ["concept_audit", "summary", "projects_declared"], 1),
            compact_summary(state["concept_audit"], ["projects_present", "projects_declared", "missing_evidence_count"]),
        ),
        check("all_declared_port_candidates_done", not p0_open and status_counts.get("done", 0) == len(candidates), dict(status_counts)),
        check("p0_port_candidates_closed", not p0_open, p0_open),
        check("legacy_mechanisms_no_red", int(get_path(state, ["mechanisms", "summary", "red"], 99) or 0) == 0, compact_summary(state["mechanisms"], ["red", "yellow_or_degraded", "top_blocker"])),
        check("runtime_long_autonomy_ready", bool(state["runtime"].get("ready_for_long_autonomy")), compact_runtime(state["runtime"])),
        check("old_registry_green", state["old_registry"].get("trigger_state") == "GREEN", compact_summary(state["old_registry"], ["cards", "case_count", "reference_answers_redacted", "hash_mismatches"])),
        check("training_sources_admitted_safely", training_sources_safe(state["training_audit"]), compact_summary(state["training_audit"], ["ready_local_verified", "serious_training_ready", "hash_mismatches", "unsafe_ready_sources"])),
        check("tiny_sampler_ready", state["training_sample"].get("trigger_state") == "GREEN" and int(get_path(state, ["training_sample", "summary", "sample_rows"], 0) or 0) > 0, compact_summary(state["training_sample"], ["sample_rows", "selected_sources", "lane_counts"])),
        check("rl_environment_admission_ready", state["rl_admission"].get("trigger_state") == "GREEN", compact_summary(state["rl_admission"], ["environments", "p0_smoke_lane", "drone_envs", "hardware_gated_envs"])),
        check("rl_smoke_plan_cataloged", rl_smoke_plan_cataloged(state["rl_smoke_plan"]), compact_summary(state["rl_smoke_plan"], ["planned_envs", "ready_for_seeded_smoke", "pending_dependency", "source_present_pending_install", "runner_pending_adapter"])),
        check("trace_capsule_admission_ready", state["trace_admission"].get("trigger_state") == "GREEN", compact_summary(state["trace_admission"], ["capsules", "accepted_metadata_only", "quarantined", "raw_payload_key_hits"])),
        check("trace_materialization_governed", trace_materialization_governed(state["trace_materialization"]), compact_summary(state["trace_materialization"], ["materialized_rows", "raw_payload_rows", "lane_counts", "rejections"])),
        check("active_inference_pilot_ready", bool(state["active_inference"].get("ready_for_world_model_training_signal")), compact_summary(state["active_inference"], ["mean_prediction_error", "accepted_belief_updates", "ready_world_jobs"])),
    ]
    activation_checks = [
        check("adapter_bank_zero_param_dry_run_ready", bool(state["adapter_plan"].get("ready_for_zero_param_dry_run")), compact_summary(state["adapter_plan"], ["plan_rows", "selected_adapters", "zero_param_lanes", "max_seen_interference"])),
        check("rl_seeded_smoke_dependencies_ready", int(get_path(state, ["rl_smoke_plan", "summary", "ready_for_seeded_smoke"], 0) or 0) > 0, compact_summary(state["rl_smoke_plan"], ["ready_for_seeded_smoke", "pending_dependency", "source_present_pending_install"])),
        check("runtime_candidate_promotion_ready", bool(state["runtime"].get("ready_for_candidate_promotion")), compact_runtime(state["runtime"])),
        check("runtime_self_evolution_ready", bool(state["runtime"].get("ready_for_self_evolution")), compact_runtime(state["runtime"])),
        check("launch_candidate_promotion_ready", bool(state["launch"].get("ready_for_candidate_promotion")), {"candidate_blockers": state["launch"].get("candidate_blockers", [])}),
    ]

    open_port_gaps = [row for row in checks if not row["passed"]]
    activation_gaps = [row for row in activation_checks if not row["passed"]]
    port_coverage_state = "GREEN" if not open_port_gaps else "RED"
    operational_maturity_state = "GREEN" if not activation_gaps else "YELLOW"

    return {
        "policy": "theseus_legacy_port_completion_audit_v0",
        "created_utc": now(),
        "source_root": concept_map.get("source_root", "D:/old_projects"),
        "port_coverage_state": port_coverage_state,
        "operational_maturity_state": operational_maturity_state,
        "trigger_state": "GREEN" if port_coverage_state == "GREEN" and operational_maturity_state == "GREEN" else "YELLOW" if port_coverage_state == "GREEN" else "RED",
        "summary": {
            "projects_declared": get_path(state, ["concept_audit", "summary", "projects_declared"], 0),
            "projects_present": get_path(state, ["concept_audit", "summary", "projects_present"], 0),
            "port_candidates": len(candidates),
            "priority_counts": dict(priority_counts),
            "status_counts": dict(status_counts),
            "open_port_gap_count": len(open_port_gaps),
            "activation_gap_count": len(activation_gaps),
            "mechanism_red_count": get_path(state, ["mechanisms", "summary", "red"], None),
            "mechanism_yellow_count": get_path(state, ["mechanisms", "summary", "yellow_or_degraded"], None),
            "ready_training_sources": get_path(state, ["training_audit", "summary", "ready_local_verified"], None),
            "rl_environments": get_path(state, ["rl_admission", "summary", "environments"], None),
            "trace_capsules_accepted": get_path(state, ["trace_admission", "summary", "accepted_metadata_only"], None),
        },
        "checks": checks,
        "activation_checks": activation_checks,
        "open_port_gaps": open_port_gaps,
        "activation_gaps": activation_gaps,
        "conclusion": (
            "All declared old-project architecture concepts are ported into governed Theseus surfaces. "
            "Remaining work is operational maturation, not discovery/port coverage."
            if not open_port_gaps
            else "Some declared old-project ports are missing current evidence."
        ),
        "next_actions": activation_next_actions(activation_gaps),
        "external_inference_calls": 0,
    }


def training_sources_safe(report: dict[str, Any]) -> bool:
    return (
        report.get("trigger_state") in {"GREEN", "YELLOW"}
        and int(get_path(report, ["summary", "ready_local_verified"], 0) or 0) > 0
        and int(get_path(report, ["summary", "serious_training_ready"], 0) or 0) > 0
        and int(get_path(report, ["summary", "hash_mismatches"], 1) or 0) == 0
        and int(get_path(report, ["summary", "unsafe_ready_sources"], 1) or 0) == 0
        and int(get_path(report, ["summary", "ready_sources_without_benchmark_exclusions"], 1) or 0) == 0
        and int(get_path(report, ["summary", "public_claim_ready_sources"], 1) or 0) == 0
    )


def rl_smoke_plan_cataloged(report: dict[str, Any]) -> bool:
    return (
        report.get("trigger_state") in {"GREEN", "YELLOW"}
        and int(get_path(report, ["summary", "planned_envs"], 0) or 0) > 0
        and int(get_path(report, ["summary", "runner_pending_adapter"], 1) or 0) == 0
        and int(report.get("external_inference_calls") or 0) == 0
    )


def trace_materialization_governed(report: dict[str, Any]) -> bool:
    return (
        report.get("trigger_state") in {"GREEN", "YELLOW"}
        and int(get_path(report, ["summary", "materialized_rows"], 0) or 0) > 0
        and int(get_path(report, ["summary", "raw_payload_rows"], 1) or 0) == 0
        and int(get_path(report, ["summary", "external_inference_calls"], 1) or 0) == 0
    )


def activation_next_actions(gaps: list[dict[str, Any]]) -> list[str]:
    if not gaps:
        return ["No legacy-port activation gaps remain."]
    actions: list[str] = []
    for row in gaps:
        gate = row["gate"]
        if gate == "adapter_bank_zero_param_dry_run_ready":
            actions.append("Run the adapter-bank zero-parameter dry-run after TaskSpell/runtime hashes are current.")
        elif gate == "rl_seeded_smoke_dependencies_ready":
            actions.append("Install or vendor optional RL environment dependencies so cataloged smoke plans can execute: gymnasium, minigrid, procgen, crafter, textworld, scienceworld, and browsergym lanes as needed.")
        elif gate == "runtime_candidate_promotion_ready":
            actions.append("Clear candidate-promotion-only gates: native bridge smokes, pretraining language lane, and coherence promotion cleanliness.")
        elif gate == "runtime_self_evolution_ready":
            actions.append("Resolve WhiteCell teacher_apply_mode_request and rerun runtime enforcement.")
        elif gate == "launch_candidate_promotion_ready":
            actions.append("Promote only after the active frontier, candidate evidence profile, and coherence candidate gates are clean.")
    return actions


def compact_summary(report: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return {key: summary.get(key) for key in keys}


def compact_runtime(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": get_path(report, ["summary", "trigger_state"], None),
        "ready_for_bounded_autonomy": report.get("ready_for_bounded_autonomy"),
        "ready_for_long_autonomy": report.get("ready_for_long_autonomy"),
        "ready_for_candidate_promotion": report.get("ready_for_candidate_promotion"),
        "ready_for_self_evolution": report.get("ready_for_self_evolution"),
        "blockers": report.get("blockers", []),
        "whitecell_active_blockers": get_path(report, ["summary", "whitecell_active_blockers"], []),
    }


def check(gate: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": gate, "passed": bool(passed), "evidence": evidence}


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# Old Projects Port Completion Audit",
        "",
        f"Generated: {report['created_utc']}",
        f"Source root: `{report['source_root']}`",
        "",
        "## Verdict",
        "",
        f"- Port coverage: `{report['port_coverage_state']}`",
        f"- Operational maturity: `{report['operational_maturity_state']}`",
        f"- Trigger state: `{report['trigger_state']}`",
        f"- Conclusion: {report['conclusion']}",
        "",
        "## Coverage Summary",
        "",
        f"- Projects present: {summary['projects_present']}/{summary['projects_declared']}",
        f"- Declared port candidates: {summary['port_candidates']}",
        f"- Status counts: `{summary['status_counts']}`",
        f"- Mechanisms: red={summary['mechanism_red_count']} yellow/degraded={summary['mechanism_yellow_count']}",
        f"- Ready training sources: {summary['ready_training_sources']}",
        f"- RL environments admitted: {summary['rl_environments']}",
        f"- Trace capsules accepted: {summary['trace_capsules_accepted']}",
        "",
        "## Port Checks",
        "",
    ]
    for row in report["checks"]:
        mark = "PASS" if row["passed"] else "FAIL"
        lines.append(f"- {mark} `{row['gate']}`")
    lines.extend(["", "## Activation Checks", ""])
    for row in report["activation_checks"]:
        mark = "PASS" if row["passed"] else "WAIT"
        lines.append(f"- {mark} `{row['gate']}`")
    lines.extend(["", "## Next Actions", ""])
    for item in report["next_actions"]:
        lines.append(f"- {item}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve(path: str | Path) -> Path:
    parsed = Path(path)
    return parsed if parsed.is_absolute() else ROOT / parsed


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
