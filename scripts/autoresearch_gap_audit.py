"""Audit SparkStream/RMI against Karpathy Autoresearch loop invariants.

This does not copy the Autoresearch architecture. It extracts the useful
governance pressure: fixed budget, fixed metric, compact result ledger,
keep/discard/crash decisions, log hygiene, and simplicity preference.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "autoresearch_loop_policy.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--out", default="reports/autoresearch_gap_audit.json")
    parser.add_argument("--markdown-out", default="reports/autoresearch_gap_audit.md")
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy)
    state = load_state(policy)
    checks = build_checks(policy, state)
    gaps = build_gaps(policy, state, checks)
    report = {
        "policy": "sparkstream_autoresearch_gap_audit_v0",
        "created_utc": now(),
        "policy_file": args.policy,
        "source_repo": policy.get("source_repo"),
        "source_observations": policy.get("source_observations", []),
        "summary": summarize(checks, gaps, state),
        "state": summarize_state(policy, state),
        "checks": checks,
        "gaps": gaps,
        "recommendations": recommendations(gaps, state),
        "external_inference_calls": 0,
    }
    write_json(ROOT / args.out, report)
    write_markdown(ROOT / args.markdown_out, report)
    print(json.dumps(report, indent=2))
    return 0 if report["summary"]["trigger_state"] != "RED" else 2


def load_state(policy: dict[str, Any]) -> dict[str, Any]:
    reports = ROOT / "reports"
    configs = ROOT / "configs"
    ledger_path = ROOT / get_path(policy, ["experiment_outcome_ledger", "path"], "reports/autoresearch_experiment_ledger.jsonl")
    return {
        "git": git_state(),
        "policy": policy,
        "self_evolution_policy": read_json(configs / "self_evolution_policy.json"),
        "autonomy_policy": read_json(configs / "autonomy_policy.json"),
        "training_profiles": read_json(configs / "training_profiles_rtx2060super.json"),
        "attd": read_json(reports / "attd_report.json"),
        "candidate": read_json(reports / "candidate_promotion_gate.json"),
        "benchmark_ledger": read_json(reports / "benchmark_ledger.json"),
        "model_ledger": read_json(reports / "model_ledger.json"),
        "resource_governor": read_json(reports / "resource_governor.json"),
        "architecture_experiments": read_json(reports / "architecture_experiment_governance.json"),
        "profile_run": read_json(reports / "training_ratchet_profile_run.json"),
        "ablation_matrix": read_json(reports / "ablation_matrix_rtx2060super_report.json"),
        "teacher_self_edit": read_json(reports / "teacher_self_edit_last.json"),
        "ledger_path": str(ledger_path.relative_to(ROOT)) if ledger_path.is_relative_to(ROOT) else str(ledger_path),
        "ledger_entries": read_jsonl(ledger_path),
        "ledger_exists": ledger_path.exists(),
    }


def build_checks(policy: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    self_edit = get_path(state, ["self_evolution_policy", "guarded_self_edit"], {})
    profiles = get_path(state, ["training_profiles", "profiles"], {})
    timeouts = get_path(state, ["autonomy_policy", "command_timeouts_seconds"], {})
    ledger_entries = state.get("ledger_entries") or []
    benchmark_ledger = state.get("benchmark_ledger") or []
    candidate = state.get("candidate") or {}
    attd = state.get("attd") or {}

    checks.append(check(
        "source_reference_recorded",
        bool(policy.get("source_repo")) and "autoresearch" in str(policy.get("source_repo")),
        "info",
        str(policy.get("source_repo")),
    ))
    checks.append(check(
        "editable_scope_explicit",
        bool(self_edit.get("default_target_scope")) and bool(self_edit.get("forbidden_target_scope")),
        "blocker",
        f"editable={self_edit.get('default_target_scope')} forbidden={self_edit.get('forbidden_target_scope')}",
    ))
    checks.append(check(
        "reports_and_data_not_teacher_editable",
        {"reports", "checkpoints/materialized", "games"} <= set(self_edit.get("forbidden_target_scope") or []),
        "blocker",
        f"forbidden={self_edit.get('forbidden_target_scope')}",
    ))
    checks.append(check(
        "fixed_profile_budgets_declared",
        bool(profiles) and all(isinstance(row, dict) and row.get("expected_runtime_minutes") for row in profiles.values()),
        "blocker",
        f"profiles={list(profiles.keys())}",
    ))
    checks.append(check(
        "timeouts_cover_profiles",
        profile_timeouts_cover(profiles, timeouts),
        "warning",
        f"timeouts={timeouts}",
    ))
    checks.append(check(
        "fixed_metric_surface_available",
        bool(benchmark_ledger) and isinstance(candidate.get("checks"), list),
        "blocker",
        f"benchmarks={len(benchmark_ledger) if isinstance(benchmark_ledger, list) else 0} candidate_checks={len(candidate.get('checks') or [])}",
    ))
    checks.append(check(
        "compact_experiment_ledger_exists",
        bool(state.get("ledger_exists")),
        "warning" if get_path(policy, ["governance", "warn_when_no_experiment_ledger"], True) else "info",
        state.get("ledger_path"),
    ))
    checks.append(check(
        "compact_experiment_ledger_has_baseline",
        any(entry.get("status") == "baseline" for entry in ledger_entries if isinstance(entry, dict)),
        "warning",
        f"entries={len(ledger_entries)}",
    ))
    checks.append(check(
        "keep_discard_crash_statuses_defined",
        {"keep", "discard", "crash"} <= set(get_path(policy, ["experiment_outcome_ledger", "status_values"], [])),
        "blocker",
        str(get_path(policy, ["experiment_outcome_ledger", "status_values"], [])),
    ))
    checks.append(check(
        "simplicity_pressure_active",
        attd.get("policy") == "sparkstream_attd_report_v0"
        and bool(get_path(state, ["self_evolution_policy", "small_model_principle", "rule"], "")),
        "blocker",
        f"attd={attd.get('trigger_state')} small_model={bool(get_path(state, ['self_evolution_policy', 'small_model_principle']))}",
    ))
    checks.append(check(
        "crash_repair_budget_declared",
        int(get_path(policy, ["crash_policy", "max_repair_attempts_per_idea"], 0) or 0) > 0,
        "warning",
        str(policy.get("crash_policy")),
    ))
    checks.append(check(
        "log_hygiene_policy_declared",
        bool(get_path(policy, ["comparability_gates", "log_redirection_required"], False)),
        "warning",
        "verbose training logs should stay in reports/log files; context gets compact metric summaries",
    ))
    checks.append(check(
        "resource_cost_reported",
        bool(state.get("resource_governor")),
        "warning",
        "resource governor report available",
    ))
    return checks


def build_gaps(policy: dict[str, Any], state: dict[str, Any], checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    failed = {row["gate"]: row for row in checks if not row.get("passed")}
    if "compact_experiment_ledger_exists" in failed:
        gaps.append(gap(
            "missing_compact_experiment_ledger",
            "medium",
            "Autoresearch has a simple result ledger. SparkStream has rich reports, but no compact append-only keep/discard/crash ledger yet.",
            "Create the first baseline row in reports/autoresearch_experiment_ledger.jsonl after the next profile run.",
        ))
    elif "compact_experiment_ledger_has_baseline" in failed:
        gaps.append(gap(
            "missing_baseline_row",
            "medium",
            "The compact ledger exists but lacks an explicit baseline row.",
            "Append a baseline row before treating later mutations as keep/discard comparable.",
        ))
    if "timeouts_cover_profiles" in failed:
        gaps.append(gap(
            "profile_timeout_mismatch",
            "low",
            "Some configured profile budgets may not be covered by command timeouts.",
            "Align command_timeouts_seconds with training profile expected_runtime_minutes.",
        ))
    if "log_hygiene_policy_declared" in failed:
        gaps.append(gap(
            "log_hygiene_not_explicit",
            "low",
            "Autoresearch redirects noisy run logs and reads compact summaries.",
            "Require profile runners to write verbose logs to report files and feed only summaries into context packets.",
        ))
    if not gaps:
        gaps.append(gap(
            "no_major_autoresearch_gap",
            "info",
            "Core Autoresearch invariants are represented; maintain the compact outcome ledger as runs accumulate.",
            "Keep refreshing this audit with every autonomy cycle.",
        ))
    return gaps


def recommendations(gaps: list[dict[str, Any]], state: dict[str, Any]) -> list[str]:
    recs = [item["next_action"] for item in gaps if item.get("severity") != "info"]
    if not recs:
        recs.append("Continue running fixed-profile experiments and append compact outcome rows for every keep/discard/crash decision.")
    if not state.get("ledger_entries"):
        recs.append("After the next smoke or inner_loop run, append a baseline row before accepting mutated winners.")
    return recs[:8]


def summarize(checks: list[dict[str, Any]], gaps: list[dict[str, Any]], state: dict[str, Any]) -> dict[str, Any]:
    blockers = [row for row in checks if row["severity"] == "blocker" and not row["passed"]]
    warnings = [row for row in checks if row["severity"] == "warning" and not row["passed"]]
    severe_gaps = [row for row in gaps if row.get("severity") in {"high", "critical"}]
    trigger_state = "RED" if blockers or severe_gaps else ("YELLOW" if warnings else "GREEN")
    return {
        "trigger_state": trigger_state,
        "passed": sum(1 for row in checks if row.get("passed")),
        "total": len(checks),
        "blockers": [row["gate"] for row in blockers],
        "warnings": [row["gate"] for row in warnings],
        "gap_count": len([row for row in gaps if row.get("severity") != "info"]),
        "ledger_entries": len(state.get("ledger_entries") or []),
        "needs_baseline": not any(
            entry.get("status") == "baseline"
            for entry in state.get("ledger_entries", [])
            if isinstance(entry, dict)
        ),
    }


def summarize_state(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    return {
        "branch": get_path(state, ["git", "branch"]),
        "dirty": get_path(state, ["git", "dirty"]),
        "experiment_ledger": state.get("ledger_path"),
        "experiment_ledger_entries": len(state.get("ledger_entries") or []),
        "training_profiles": list(get_path(state, ["training_profiles", "profiles"], {}).keys()),
        "candidate_gate": f"{get_path(state, ['candidate', 'passed'])}/{get_path(state, ['candidate', 'total'])}",
        "candidate_promote": get_path(state, ["candidate", "promote"]),
        "attd": get_path(state, ["attd", "trigger_state"]),
        "primary_metric_contract": "benchmark/candidate gates, not a single val_bpb",
        "source_difference": "SparkStream is multi-benchmark and multi-arm, so keep/discard is multi-gate rather than one scalar.",
    }


def check(gate: str, passed: bool, severity: str, evidence: str) -> dict[str, Any]:
    return {
        "gate": gate,
        "passed": bool(passed),
        "severity": severity,
        "evidence": evidence,
    }


def gap(gap_id: str, severity: str, description: str, next_action: str) -> dict[str, Any]:
    return {
        "id": gap_id,
        "severity": severity,
        "description": description,
        "next_action": next_action,
    }


def profile_timeouts_cover(profiles: Any, timeouts: Any) -> bool:
    if not isinstance(profiles, dict) or not isinstance(timeouts, dict):
        return False
    for name, profile in profiles.items():
        if not isinstance(profile, dict):
            return False
        minutes = float(profile.get("expected_runtime_minutes") or 0)
        timeout = float(timeouts.get(name) or 0)
        if minutes <= 0 or timeout < minutes * 60:
            return False
    return True


def git_state() -> dict[str, Any]:
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
        porcelain = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines()
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        return {
            "available": True,
            "branch": branch,
            "commit": commit,
            "dirty": bool(porcelain),
            "porcelain_count": len(porcelain),
        }
    except Exception as exc:  # pragma: no cover - git may be unavailable in packaged contexts
        return {"available": False, "error": str(exc)}


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Autoresearch Gap Audit",
        "",
        f"Updated: {report.get('created_utc')}",
        "",
        f"Source: {report.get('source_repo')}",
        "",
        f"Trigger state: {get_path(report, ['summary', 'trigger_state'])}",
        "",
        "## Checks",
        "",
    ]
    for row in report.get("checks", []):
        marker = "PASS" if row.get("passed") else "FAIL"
        lines.append(f"- {marker} `{row.get('gate')}` ({row.get('severity')}): {row.get('evidence')}")
    lines.extend(["", "## Gaps", ""])
    for row in report.get("gaps", []):
        lines.append(f"- `{row.get('id')}` ({row.get('severity')}): {row.get('description')} Next: {row.get('next_action')}")
    lines.extend(["", "## Recommendations", ""])
    for item in report.get("recommendations", []):
        lines.append(f"- {item}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
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


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
