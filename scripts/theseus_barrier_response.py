#!/usr/bin/env python3
"""Classify the next Theseus barrier without crossing policy walls.

This is a control-plane fact producer. It reads existing reports, separates
safe engineering work from hard-stop governance walls, and emits the next
private-only action target. It does not execute repairs, run public calibration,
or call external inference.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "theseus_barrier_response.json"
DEFAULT_MARKDOWN = REPORTS / "theseus_barrier_response.md"

SAFE_CLASSES = {
    "runtime_bottleneck",
    "artifact_bloat",
    "source_modularity",
    "stale_report_evidence",
    "duplicate_family_pressure",
    "missing_registry_ownership",
    "private_only_verifier_failure",
}
HARD_STOP_CLASSES = {
    "public_calibration_lock",
    "public_data_contamination_risk",
    "external_inference_runtime_serving_risk",
    "teacher_gate_failure",
    "arbitrary_remote_execution_risk",
    "unknown_policy_risk",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()

    started = time.perf_counter()
    reports = collect_reports()
    barriers = classify_barriers(reports)
    safe_barriers = [row for row in barriers if row["class"] in SAFE_CLASSES]
    hard_stops = [row for row in barriers if row["class"] in HARD_STOP_CLASSES]
    selected = select_barrier(safe_barriers)
    payload = {
        "policy": "project_theseus_barrier_response_v1",
        "created_utc": now(),
        "trigger_state": trigger_state(selected, hard_stops),
        "summary": {
            "selected_barrier_id": selected.get("id", ""),
            "selected_barrier_class": selected.get("class", ""),
            "safe_barrier_count": len(safe_barriers),
            "hard_stop_count": len(hard_stops),
            "top_safe_barrier": safe_barriers[0]["id"] if safe_barriers else "",
            "top_hard_stop": hard_stops[0]["id"] if hard_stops else "",
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "selected_barrier": selected,
        "safe_barriers": safe_barriers,
        "hard_stops": hard_stops,
        "reports_used": reports["sources"],
        "rules": {
            "safe_classes": sorted(SAFE_CLASSES),
            "hard_stop_classes": sorted(HARD_STOP_CLASSES),
            "public_benchmark_boundary": "Public calibration remains calibration-only and locked unless an exact operator unlock exists.",
            "teacher_boundary": "Teacher rows require the governed distillation gate and are never runtime serving tokens.",
            "execution_boundary": "This report recommends commands but does not execute arbitrary repairs.",
        },
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), payload)
    write_text(resolve(args.markdown_out), render_markdown(payload))
    print(json.dumps(payload, indent=2))
    return 0 if payload["trigger_state"] != "RED" else 2


def collect_reports() -> dict[str, Any]:
    sources: dict[str, str] = {}

    def latest(name: str, pattern: str, default: Any) -> Any:
        path = latest_path(pattern)
        if not path:
            sources[name] = ""
            return default
        sources[name] = rel(path)
        return read_json(path, default)

    return {
        "sources": sources,
        "efficiency": latest("efficiency", "system_efficiency_audit*.json", {}),
        "registry": latest("registry", "theseus_project_registry*.json", {}),
        "retention": latest("retention", "theseus_artifact_retention*.json", {}),
        "attd": latest("attd", "attd_report*.json", {}),
        "attd_packets": latest("attd_packets", "attd_maintenance_packets*.json", {}),
        "fanout": latest("fanout", "code_lm_closure_rust_*_current_source_smoke_fanout.json", {}),
        "training_admission": latest("training_admission", "training_data_admission*.json", {}),
        "teacher_gate": latest("teacher_gate", "teacher_distillation_gate*.json", {}),
        "hive_policy": read_json(ROOT / "configs" / "hive_policy.json", {}),
    }


def classify_barriers(reports: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(runtime_barriers(reports))
    rows.extend(registry_barriers(reports))
    rows.extend(artifact_barriers(reports))
    rows.extend(attd_barriers(reports))
    rows.extend(private_verifier_barriers(reports))
    rows.extend(hard_stop_barriers(reports))
    rows.sort(key=lambda row: (-float(row.get("priority", 0.0)), row["id"]))
    return rows


def runtime_barriers(reports: dict[str, Any]) -> list[dict[str, Any]]:
    efficiency = reports["efficiency"] if isinstance(reports.get("efficiency"), dict) else {}
    bottlenecks = efficiency.get("loop_bottlenecks") if isinstance(efficiency.get("loop_bottlenecks"), list) else []
    rows = []
    for item in bottlenecks[:8]:
        if not isinstance(item, dict):
            continue
        priority = 0.8 + float(number(item.get("impact_score", 0.0)))
        rows.append(
            barrier(
                barrier_id=str(item.get("id") or "runtime_bottleneck"),
                barrier_class="runtime_bottleneck",
                severity=str(item.get("severity") or "YELLOW"),
                priority=priority,
                evidence=str(item.get("evidence") or ""),
                action=str(item.get("recommended_action") or "Optimize the measured runtime branch."),
                files=[
                    "crates/symliquid-cli/src/code_lm_closure/candidate_fanout/expression_pool.rs",
                    "crates/symliquid-cli/src/code_lm_closure/candidate_fanout/task_rows.rs",
                    "scripts/code_lm_train_once_fanout.py",
                    "scripts/system_efficiency_audit.py",
                ],
                commands=runtime_commands(),
                rollback="Revert only the fanout hot-path patch if private-only smoke regresses or no-cheat counters change.",
                expected_metrics={
                    "baseline_private_candidate_generation_ms": 5982,
                    "baseline_candidate_expression_generation_ms_total": 21507,
                    "baseline_prompt_contract_transduction_total": 7666,
                    "target": "reduce candidate_expression_generation_ms or prompt-contract transduction without reducing accepted private candidates",
                },
            )
        )
    return rows


def registry_barriers(reports: dict[str, Any]) -> list[dict[str, Any]]:
    registry = reports["registry"] if isinstance(reports.get("registry"), dict) else {}
    summary = registry.get("summary") if isinstance(registry.get("summary"), dict) else {}
    rows = []
    unregistered = int(number(summary.get("unregistered_active_source_count", 0)))
    if unregistered:
        rows.append(
            barrier(
                barrier_id="missing_registry_ownership",
                barrier_class="missing_registry_ownership",
                severity="RED" if unregistered >= 80 else "YELLOW",
                priority=0.92,
                evidence=f"unregistered_active_source_count={unregistered}",
                action="Register active sources under an existing surface or retire them through the registry.",
                files=["configs/project_manifest_registry.json", "scripts/theseus_project_registry.py"],
                commands=["python3 scripts/theseus_project_registry.py"],
                rollback="Remove only the registry entry added in this pass if it misclassifies an active source.",
            )
        )
    stale = int(number(summary.get("stale_report_output_count", 0)))
    if stale:
        rows.append(
            barrier(
                barrier_id="stale_report_evidence",
                barrier_class="stale_report_evidence",
                severity="YELLOW",
                priority=0.7,
                evidence=f"stale_report_output_count={stale}",
                action="Refresh or retire stale registered outputs before using them as current evidence.",
                files=["configs/project_manifest_registry.json"],
                commands=["python3 scripts/theseus_project_registry.py"],
                rollback="Restore the previous registry report if freshness metadata is wrong.",
            )
        )
    total_duplicates = int(number(summary.get("duplicate_family_count", 0)))
    duplicates = int(number(summary.get("unclassified_duplicate_family_count", total_duplicates)))
    threshold = int(number(get_path(summary, ["thresholds", "yellow_duplicate_families"], 8)))
    if duplicates >= threshold:
        duplicate_rows = registry.get("duplicate_families") if isinstance(registry.get("duplicate_families"), list) else []
        unclassified_report = [
            row for row in duplicate_rows
            if isinstance(row, dict) and row.get("root") == "reports" and not row.get("classified")
        ]
        classified_source = [
            row for row in duplicate_rows
            if isinstance(row, dict) and row.get("root") in {"scripts", "configs", "docs"} and row.get("classified")
        ]
        rows.append(
            barrier(
                barrier_id="duplicate_family_pressure",
                barrier_class="duplicate_family_pressure",
                severity="YELLOW",
                priority=0.62,
                evidence=(
                    f"duplicate_family_count={total_duplicates}; "
                    f"unclassified_duplicate_family_count={duplicates}; "
                    f"classified_source_duplicates={len(classified_source)}; "
                    f"unclassified_report_duplicates={len(unclassified_report)}"
                ),
                action="Classify report-history families or archive generated variants; do not create new vN lanes.",
                files=["configs/project_manifest_registry.json", "scripts/theseus_artifact_retention.py"],
                commands=[
                    "python3 scripts/theseus_project_registry.py",
                    "python3 scripts/theseus_artifact_retention.py --min-bytes 20000000 --include-jsonl --include-archived-report-dirs --include-report-snapshots --include-runtime-replay-mirrors --include-dist-artifacts",
                ],
                rollback="Undo only classification entries that incorrectly promote historical reports to active source.",
                expected_metrics={
                    "current_duplicate_family_count": total_duplicates,
                    "current_unclassified_duplicate_family_count": duplicates,
                    "target": "reduce count or make remaining duplicate pressure explicitly classified by root/family",
                },
            )
        )
    generated = int(number(summary.get("generated_source_artifact_count", 0)))
    if generated:
        rows.append(
            barrier(
                barrier_id="generated_source_artifacts",
                barrier_class="artifact_bloat",
                severity="RED" if generated else "YELLOW",
                priority=0.95,
                evidence=f"generated_source_artifact_count={generated}",
                action="Quarantine generated source-path artifacts and add ignore coverage.",
                files=[".gitignore", "configs/project_manifest_registry.json"],
                commands=["python3 scripts/theseus_project_registry.py"],
                rollback="Restore only intentionally tracked source files if a generated-artifact match was too broad.",
            )
        )
    return rows


def artifact_barriers(reports: dict[str, Any]) -> list[dict[str, Any]]:
    retention = reports["retention"] if isinstance(reports.get("retention"), dict) else {}
    summary = retention.get("summary") if isinstance(retention.get("summary"), dict) else {}
    candidate_count = int(number(summary.get("candidate_count", 0)))
    dry_gib = float(number(summary.get("dry_run_candidate_gib", 0.0)))
    if candidate_count <= 0 and dry_gib <= 0:
        return []
    return [
        barrier(
            barrier_id="artifact_bloat",
            barrier_class="artifact_bloat",
            severity="YELLOW",
            priority=0.74 + min(0.2, dry_gib / 20.0),
            evidence=f"retention_candidate_count={candidate_count}; dry_run_candidate_gib={dry_gib}",
            action="Run manifest-backed retention before more long loops.",
            files=["scripts/theseus_artifact_retention.py", "scripts/theseus_archive_resolver.py"],
            commands=[
                "python3 scripts/theseus_artifact_retention.py --min-bytes 20000000 --include-jsonl --include-archived-report-dirs --include-report-snapshots --include-runtime-replay-mirrors --include-dist-artifacts"
            ],
            rollback="Use the retention manifest and archive pointers to restore any artifact needed for replay.",
        )
    ]


def attd_barriers(reports: dict[str, Any]) -> list[dict[str, Any]]:
    attd = reports["attd"] if isinstance(reports.get("attd"), dict) else {}
    summary = attd.get("summary") if isinstance(attd.get("summary"), dict) else {}
    hotspots = int(number(summary.get("hotspot_count", summary.get("packet_count", 0))))
    state = str(attd.get("trigger_state") or summary.get("trigger_state") or "")
    if state not in {"YELLOW", "RED"} and hotspots <= 0:
        return []
    return [
        barrier(
            barrier_id="source_modularity_attd",
            barrier_class="source_modularity",
            severity="RED" if state == "RED" else "YELLOW",
            priority=0.55,
            evidence=f"attd_state={state or 'unknown'}; hotspot_count={hotspots}",
            action="Use ATTD maintenance packets where they overlap active runtime or registry pressure.",
            files=["scripts/attd_analyzer.py", "configs/attd_policy.json"],
            commands=["python3 scripts/attd_analyzer.py", "python3 scripts/system_efficiency_audit.py"],
            rollback="Revert only the source split if py_compile/cargo checks fail.",
        )
    ]


def private_verifier_barriers(reports: dict[str, Any]) -> list[dict[str, Any]]:
    efficiency = reports["efficiency"] if isinstance(reports.get("efficiency"), dict) else {}
    summary = efficiency.get("summary") if isinstance(efficiency.get("summary"), dict) else {}
    no_admissible = float(number(summary.get("current_repair_no_admissible_task_rate", 0.0)))
    if no_admissible <= 0.0:
        return []
    return [
        barrier(
            barrier_id="private_only_verifier_failure",
            barrier_class="private_only_verifier_failure",
            severity="YELLOW",
            priority=0.78,
            evidence=f"current_repair_no_admissible_task_rate={no_admissible}",
            action="Repair private-only verifier/admissibility failures before public calibration.",
            files=["scripts/private_candidate_replay_contract_audit_v1.py", "crates/symliquid-cli/src/code_lm_closure/contract_verifier.rs"],
            commands=["python3 scripts/private_candidate_replay_contract_audit_v1.py"],
            rollback="Revert verifier changes if accepted private rows decrease or fallback/template counters increase.",
        )
    ]


def hard_stop_barriers(reports: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    efficiency = reports["efficiency"] if isinstance(reports.get("efficiency"), dict) else {}
    summary = efficiency.get("summary") if isinstance(efficiency.get("summary"), dict) else {}
    if bool(summary.get("public_calibration_locked", False)):
        rows.append(
            hard_stop(
                "public_calibration_lock",
                "public_calibration_lock",
                "YELLOW",
                "system_efficiency_audit reports public_calibration_locked=true",
                "Do not run public calibration; use private repair/smoke work until an exact operator unlock exists.",
            )
        )
    for key in ["efficiency", "registry", "retention", "fanout", "training_admission"]:
        report = reports.get(key) if isinstance(reports.get(key), dict) else {}
        calls = int(number(report.get("external_inference_calls", 0)))
        if calls:
            rows.append(
                hard_stop(
                    f"external_inference_calls_{key}",
                    "external_inference_runtime_serving_risk",
                    "RED",
                    f"{key} external_inference_calls={calls}",
                    "Stop and audit teacher/runtime boundary before any autonomous continuation.",
                )
            )
    admission = reports.get("training_admission") if isinstance(reports.get("training_admission"), dict) else {}
    adm_summary = admission.get("summary") if isinstance(admission.get("summary"), dict) else {}
    if adm_summary and not bool(adm_summary.get("public_benchmark_payload_admitted_zero", True)):
        rows.append(
            hard_stop(
                "public_data_contamination_risk",
                "public_data_contamination_risk",
                "RED",
                "training admission did not prove public_benchmark_payload_admitted_zero",
                "Stop training row admission until public benchmark payload exclusion is restored.",
            )
        )
    hive_policy = reports.get("hive_policy") if isinstance(reports.get("hive_policy"), dict) else {}
    if arbitrary_shell_enabled(hive_policy):
        rows.append(
            hard_stop(
                "arbitrary_remote_execution_risk",
                "arbitrary_remote_execution_risk",
                "RED",
                "hive policy explicitly enables an arbitrary-shell capability",
                "Stop and restore bounded registered task policy before remote execution is enabled.",
            )
        )
    return rows


def select_barrier(safe_barriers: list[dict[str, Any]]) -> dict[str, Any]:
    return safe_barriers[0] if safe_barriers else {}


def barrier(
    *,
    barrier_id: str,
    barrier_class: str,
    severity: str,
    priority: float,
    evidence: str,
    action: str,
    files: list[str],
    commands: list[str],
    rollback: str,
    expected_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": barrier_id,
        "class": barrier_class,
        "severity": severity,
        "priority": round(priority, 4),
        "automation": "safe_recommendation",
        "evidence": evidence,
        "recommended_action": action,
        "files": files,
        "commands": commands,
        "rollback_boundary": rollback,
        "expected_metrics": expected_metrics or {},
    }


def hard_stop(barrier_id: str, barrier_class: str, severity: str, evidence: str, action: str) -> dict[str, Any]:
    return {
        "id": barrier_id,
        "class": barrier_class,
        "severity": severity,
        "priority": 1.0 if severity == "RED" else 0.5,
        "automation": "hard_stop",
        "evidence": evidence,
        "recommended_action": action,
        "files": [],
        "commands": [],
        "rollback_boundary": "none; do not continue this class of work without resolving the policy wall",
        "expected_metrics": {},
    }


def runtime_commands() -> list[str]:
    return [
        "cargo fmt --package symliquid-cli",
        "cargo check -p symliquid-cli",
        "cargo test -p symliquid-cli",
        "python3 scripts/code_lm_train_once_fanout.py --execute --slug fanout_speed_current_source_private_v1 --refresh-fanout-only --private-only-refresh --refresh-private-eval-limit 96 --refresh-candidates-per-task 2 --refresh-smoke-timeout-seconds 900 --skip-build",
        "python3 scripts/system_efficiency_audit.py",
    ]


def arbitrary_shell_enabled(payload: Any) -> bool:
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = str(key).lower()
            if lowered in {"arbitrary_shell", "can_request_arbitrary_shell", "arbitrary_shell_allowed"}:
                return bool(value)
            if lowered in {"allowed_task_scope", "remote_task_kinds", "task_kinds"}:
                if sequence_contains_enabled_arbitrary_shell(value):
                    return True
            if arbitrary_shell_enabled(value):
                return True
    elif isinstance(payload, list):
        for item in payload:
            if arbitrary_shell_enabled(item):
                return True
    return False


def sequence_contains_enabled_arbitrary_shell(value: Any) -> bool:
    if isinstance(value, dict):
        task = value.get("arbitrary_shell")
        if isinstance(task, dict):
            return str(task.get("enabled", "true")).strip().lower() not in {"0", "false", "off", "blocked"}
        return bool(task)
    if not isinstance(value, list):
        return False
    return any(str(item).strip() == "arbitrary_shell" for item in value)


def trigger_state(selected: dict[str, Any], hard_stops: list[dict[str, Any]]) -> str:
    if any(row.get("severity") == "RED" for row in hard_stops):
        return "RED"
    if selected or hard_stops:
        return "YELLOW"
    return "GREEN"


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Theseus Barrier Response",
        "",
        f"- Status: `{payload['trigger_state']}`",
        f"- Selected: `{summary.get('selected_barrier_id') or 'none'}`",
        f"- Safe barriers: `{summary.get('safe_barrier_count')}`",
        f"- Hard stops: `{summary.get('hard_stop_count')}`",
        "",
    ]
    selected = payload.get("selected_barrier") if isinstance(payload.get("selected_barrier"), dict) else {}
    if selected:
        lines.extend([
            "## Selected Safe Work",
            "",
            f"- Class: `{selected.get('class')}`",
            f"- Evidence: {selected.get('evidence')}",
            f"- Action: {selected.get('recommended_action')}",
            f"- Rollback: {selected.get('rollback_boundary')}",
            "",
            "### Commands",
            "",
        ])
        for command in selected.get("commands", []):
            lines.append(f"- `{command}`")
        lines.append("")
    lines.extend(["## Hard Stops", ""])
    for row in payload.get("hard_stops", []):
        lines.extend([
            f"- `{row.get('severity')}` `{row.get('id')}`: {row.get('evidence')}",
            f"  - {row.get('recommended_action')}",
        ])
    if not payload.get("hard_stops"):
        lines.append("- none")
    lines.extend(["", "## Safe Barrier Queue", ""])
    for row in payload.get("safe_barriers", [])[:8]:
        lines.extend([
            f"- `{row.get('class')}` `{row.get('id')}` priority `{row.get('priority')}`",
            f"  - Evidence: {row.get('evidence')}",
            f"  - Action: {row.get('recommended_action')}",
        ])
    if not payload.get("safe_barriers"):
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def latest_path(pattern: str) -> Path | None:
    paths = [path for path in REPORTS.glob(pattern) if path.is_file()]
    if not paths:
        return None
    return max(paths, key=lambda path: path.stat().st_mtime)


def get_path(payload: Any, keys: list[Any], default: Any = None) -> Any:
    cur = payload
    for key in keys:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
