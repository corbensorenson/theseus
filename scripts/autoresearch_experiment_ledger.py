"""Append compact Autoresearch-style experiment outcome rows.

SparkStream keeps rich RMI ledgers, but this ledger is intentionally small:
one row per comparable experiment with metric, memory, status, and description.
It is ignored by git under reports/.
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
    parser.add_argument("--profile", default="inner_loop")
    parser.add_argument("--status", default="auto")
    parser.add_argument("--description", default="")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--out-ledger", default="")
    parser.add_argument("--summary-out", default="reports/autoresearch_experiment_ledger_summary.json")
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy)
    ledger_path = ROOT / (args.out_ledger or get_path(policy, ["experiment_outcome_ledger", "path"], "reports/autoresearch_experiment_ledger.jsonl"))
    existing = read_jsonl(ledger_path)
    reports = ROOT / "reports"
    candidate = read_json(reports / "candidate_promotion_gate.json")
    resource = read_json(reports / "resource_governor.json")
    attd = read_json(reports / "attd_report.json")
    profile_run = read_json(reports / "training_ratchet_profile_run.json")
    git = git_state()

    row = build_row(args, policy, existing, candidate, resource, attd, profile_run, git)
    if args.append:
        append_jsonl(ledger_path, row)
        existing = [*existing, row]

    summary = {
        "policy": "sparkstream_autoresearch_experiment_ledger_summary_v0",
        "created_utc": now(),
        "ledger_path": str(ledger_path.relative_to(ROOT)) if ledger_path.is_relative_to(ROOT) else str(ledger_path),
        "appended": bool(args.append),
        "last_row": row,
        "entries": len(existing),
        "baseline_present": any(item.get("status") == "baseline" for item in existing if isinstance(item, dict)),
        "best": best_row(existing),
        "external_inference_calls": 0,
    }
    write_json(ROOT / args.summary_out, summary)
    print(json.dumps(summary, indent=2))
    return 0


def build_row(
    args: argparse.Namespace,
    policy: dict[str, Any],
    existing: list[dict[str, Any]],
    candidate: dict[str, Any],
    resource: dict[str, Any],
    attd: dict[str, Any],
    profile_run: dict[str, Any],
    git: dict[str, Any],
) -> dict[str, Any]:
    status = args.status
    if status == "auto":
        status = "baseline" if not existing else ("keep" if candidate.get("promote") else "needs_more_evidence")
    scores = candidate.get("scores") or {}
    primary_name = "active_frontier_accuracy"
    primary_value = scores.get(primary_name)
    if primary_value is None:
        primary_name = "public_accuracy"
        primary_value = scores.get(primary_name)
    memory_mib = get_path(resource, ["current_resources", "gpu", "memory_used_mib"], 0) or get_path(resource, ["profile_budget", "max_vram_mib"], 0) or 0
    description = args.description or default_description(status, candidate, profile_run)
    return {
        "run_id": f"ar-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{git.get('commit', 'nogit')}",
        "created_utc": now(),
        "branch": git.get("branch"),
        "start_commit": git.get("commit"),
        "end_commit": git.get("commit"),
        "profile": args.profile,
        "primary_metric_name": primary_name,
        "primary_metric_value": primary_value,
        "primary_metric_direction": "higher_is_better",
        "memory_gb": round(float(memory_mib) / 1024.0, 3),
        "status": status,
        "description": description,
        "report_paths": [
            "reports/candidate_promotion_gate.json",
            "reports/resource_governor.json",
            "reports/training_ratchet_profile_run.json",
            "reports/attd_report.json",
        ],
        "attd_delta": {
            "current_trigger_state": attd.get("trigger_state"),
            "current_score": attd.get("attd_score"),
        },
        "regression_delta": {
            "public_accuracy": scores.get("public_accuracy"),
            "seed49_regression_accuracy": scores.get("seed49_regression_accuracy"),
        },
        "residual_delta": candidate.get("residual_delta"),
        "candidate_promote": candidate.get("promote"),
        "candidate_gate": f"{candidate.get('passed')}/{candidate.get('total')}",
        "failed_gates": [
            item.get("gate")
            for item in candidate.get("checks", [])
            if isinstance(item, dict) and not item.get("passed")
        ],
        "append_only": bool(get_path(policy, ["experiment_outcome_ledger", "append_only"], True)),
        "external_inference_calls": 0,
    }


def default_description(status: str, candidate: dict[str, Any], profile_run: dict[str, Any]) -> str:
    failed = [
        item.get("gate")
        for item in candidate.get("checks", [])
        if isinstance(item, dict) and not item.get("passed")
    ]
    profile = profile_run.get("profile") or "current profile"
    if status == "baseline":
        return f"baseline snapshot for {profile}; candidate gate {candidate.get('passed')}/{candidate.get('total')}"
    if status == "keep":
        return f"candidate promoted on {profile}"
    if status == "crash":
        return f"crash recorded on {profile}"
    return f"{profile} needs more evidence; failed gates={failed}"


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [
        row
        for row in rows
        if isinstance(row.get("primary_metric_value"), (int, float)) and row.get("status") in {"baseline", "keep", "simplification_keep"}
    ]
    if not scored:
        return {}
    return max(scored, key=lambda row: float(row.get("primary_metric_value")))


def git_state() -> dict[str, Any]:
    try:
        return {
            "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
            "commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip(),
        }
    except Exception as exc:  # pragma: no cover
        return {"branch": "", "commit": "", "error": str(exc)}


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


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
