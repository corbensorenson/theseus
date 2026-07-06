"""Find and safely reduce candidate bottlenecks before asking the teacher.

The reducer is deliberately boring: it reads generated reports, classifies
runtime/governance/asset blockers, and can apply only policy-approved local
setup work such as creating isolated Python venvs. It never downloads bulk
datasets, never connects to live hardware, and never calls external inference.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "candidate_bottleneck_policy.json"
DEFAULT_OUT = ROOT / "reports" / "candidate_bottleneck_reducer.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--fix", action="store_true")
    parser.add_argument("--include-optional", action="store_true")
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy)
    before = observe()
    bottlenecks = classify_bottlenecks(before, policy)
    actions = plan_actions(bottlenecks, policy, include_optional=args.include_optional)
    applied: list[dict[str, Any]] = []
    if args.fix and get_path(policy, ["safe_auto_fix", "enabled"], True):
        for action in actions:
            if not action.get("safe_auto"):
                continue
            applied.append(run_command(action["command"], action.get("id", "runtime_fix")))
        if applied and get_path(policy, ["safe_auto_fix", "refresh_reports_after_fix"], True):
            for command in policy.get("refresh_commands", []):
                if isinstance(command, list) and command:
                    applied.append(run_command(command, "refresh_reports"))

    after = observe() if applied else before
    remaining_bottlenecks = classify_bottlenecks(after, policy)
    remaining_actions = plan_actions(remaining_bottlenecks, policy, include_optional=args.include_optional)
    report = {
        "policy": "theseus_candidate_bottleneck_reducer_v0",
        "created_utc": now(),
        "config": str(Path(args.policy)).replace("\\", "/"),
        "fix_requested": bool(args.fix),
        "include_optional": bool(args.include_optional),
        "status": status(after),
        "candidate_flow_ready": candidate_flow_ready(after),
        "before": compact_observation(before),
        "after": compact_observation(after),
        "initial_bottlenecks": bottlenecks,
        "remaining_bottlenecks": remaining_bottlenecks,
        "bottlenecks": remaining_bottlenecks,
        "planned_actions": actions,
        "remaining_safe_auto_actions": [action for action in remaining_actions if action.get("safe_auto")],
        "applied_actions": applied,
        "teacher_needed": teacher_needed(after, remaining_bottlenecks),
        "teacher_reason": teacher_reason(after, remaining_bottlenecks),
        "external_inference_calls": 0,
    }
    write_json(ROOT / args.out, report)
    print(json.dumps(report, indent=2))
    return 0


def observe() -> dict[str, Any]:
    return {
        "smoke": read_json(ROOT / "reports" / "benchmark_adapter_smoke_status.json"),
        "factory": read_json(ROOT / "reports" / "benchmark_adapter_factory.json"),
        "curriculum": read_json(ROOT / "reports" / "benchmaxx_curriculum.json"),
        "python_runtime": read_json(ROOT / "reports" / "python_runtime_compatibility.json"),
        "resource_governor": read_json(ROOT / "reports" / "resource_governor.json"),
        "performance": read_json(ROOT / "reports" / "performance_optimizer.json"),
    }


def classify_bottlenecks(observation: dict[str, Any], policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    smoke_cards = observation.get("smoke", {}).get("cards", [])
    remediation = policy.get("runtime_remediation", {}) if isinstance(policy.get("runtime_remediation"), dict) else {}
    for card in smoke_cards if isinstance(smoke_cards, list) else []:
        if not isinstance(card, dict):
            continue
        card_id = str(card.get("card_id") or "")
        status_value = str(card.get("smoke_status") or "")
        failed = [check for check in card.get("checks", []) if isinstance(check, dict) and not check.get("passed")]
        runtime_failed = [check for check in failed if check.get("severity") == "runtime"]
        hard_failed = [check for check in failed if check.get("severity") == "hard"]
        if status_value == "passed":
            continue
        kind = "runtime_dependency" if runtime_failed and not hard_failed else "governance_or_asset"
        policy_row = remediation.get(card_id, {})
        rows.append(
            {
                "id": card_id,
                "name": card.get("name"),
                "category": card.get("category"),
                "adapter_type": card.get("adapter_type"),
                "status": status_value,
                "kind": kind,
                "first_failed_check": failed[0].get("name") if failed else "",
                "first_failed_evidence": failed[0].get("evidence") if failed else "",
                "runtime_failed_checks": [check.get("name") for check in runtime_failed],
                "hard_failed_checks": [check.get("name") for check in hard_failed],
                "safe_auto_fix_available": bool(policy_row.get("safe_auto")),
                "manual_reason": policy_row.get("reason", ""),
            }
        )
    curriculum = observation.get("curriculum", {}).get("next_frontier", {})
    if isinstance(curriculum, dict) and curriculum.get("runnable_now") is False:
        rows.append(
            {
                "id": "next_frontier",
                "name": curriculum.get("family"),
                "category": "curriculum",
                "status": "blocked",
                "kind": "frontier_not_runnable",
                "first_failed_check": "next_frontier_runnable_now",
                "first_failed_evidence": curriculum.get("blocked_reason", ""),
                "safe_auto_fix_available": False,
                "manual_reason": "Frontier rotation is waiting on adapter/runtime readiness.",
            }
        )
    return rows


def plan_actions(bottlenecks: list[dict[str, Any]], policy: dict[str, Any], *, include_optional: bool) -> list[dict[str, Any]]:
    remediation = policy.get("runtime_remediation", {}) if isinstance(policy.get("runtime_remediation"), dict) else {}
    actions = []
    seen: set[str] = set()
    for item in bottlenecks:
        card_id = str(item.get("id") or "")
        rule = remediation.get(card_id, {})
        command = rule.get("command")
        if not isinstance(command, list) or not command:
            continue
        safe_auto = bool(rule.get("safe_auto"))
        if not safe_auto and not include_optional:
            continue
        key = "\0".join(str(part) for part in command)
        if key in seen:
            continue
        seen.add(key)
        actions.append(
            {
                "id": card_id,
                "kind": rule.get("kind", "runtime_remediation"),
                "safe_auto": safe_auto,
                "command": command,
                "expected_to_clear": rule.get("expected_to_clear", []),
                "reason": rule.get("reason", ""),
            }
        )
    return actions


def run_command(command: list[Any], name: str) -> dict[str, Any]:
    command = [str(part) for part in command]
    start = time.time()
    try:
        proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=1800)
        return {
            "name": name,
            "command": command,
            "returncode": proc.returncode,
            "ok": proc.returncode == 0,
            "runtime_ms": int((time.time() - start) * 1000),
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }
    except Exception as exc:  # noqa: BLE001 - report path.
        return {
            "name": name,
            "command": command,
            "returncode": -1,
            "ok": False,
            "runtime_ms": int((time.time() - start) * 1000),
            "stdout_tail": "",
            "stderr_tail": str(exc),
        }


def status(observation: dict[str, Any]) -> str:
    smoke_summary = observation.get("smoke", {}).get("summary", {})
    runtime_blocked = int(number(smoke_summary.get("runtime_blocked", 0)))
    failed = int(number(smoke_summary.get("failed", 0)))
    next_frontier = observation.get("curriculum", {}).get("next_frontier", {})
    if failed:
        return "RED"
    if isinstance(next_frontier, dict) and next_frontier.get("runnable_now") is False:
        return "YELLOW_FRONTIER_SETUP_REQUIRED"
    if runtime_blocked:
        if candidate_flow_ready(observation):
            return "YELLOW_OPTIONAL_RUNTIME_BLOCKERS"
        return "YELLOW_RUNTIME_BLOCKERS"
    return "GREEN"


def candidate_flow_ready(observation: dict[str, Any]) -> bool:
    next_frontier = observation.get("curriculum", {}).get("next_frontier", {})
    if not isinstance(next_frontier, dict):
        return False
    return bool(next_frontier.get("runnable_now"))


def teacher_needed(observation: dict[str, Any], bottlenecks: list[dict[str, Any]]) -> bool:
    # Teacher should only be needed after safe local remediation cannot clear a
    # source bug or architecture wall. Policy-described manual/native runtime
    # blockers are still surfaced, but they are not teacher work by default.
    if status(observation) == "RED":
        return True
    return any(
        item.get("kind") == "runtime_dependency"
        and not item.get("safe_auto_fix_available")
        and not item.get("manual_reason")
        for item in bottlenecks
    )


def teacher_reason(observation: dict[str, Any], bottlenecks: list[dict[str, Any]]) -> str:
    if not teacher_needed(observation, bottlenecks):
        return "local_bottleneck_reduction_or_manual_runtime_setup_sufficient"
    return "runtime_blocker_has_no_safe_local_remediation_or_failed_after_remediation"


def compact_observation(observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "smoke_summary": observation.get("smoke", {}).get("summary", {}),
        "factory_summary": observation.get("factory", {}).get("summary", {}),
        "next_frontier": observation.get("curriculum", {}).get("next_frontier", {}),
        "python_runtime_summary": observation.get("python_runtime", {}).get("summary", {}),
        "performance_summary": observation.get("performance", {}).get("summary", observation.get("performance", {})),
    }


def get_path(row: Any, path: list[str], default: Any = None) -> Any:
    cur = row
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def number(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


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


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
