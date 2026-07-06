#!/usr/bin/env python3
"""Private generalization frontier expander.

When the pre-public audit says private evidence is ready but public calibration
is still locked, the system should not sit idle. This controller queues and can
execute one additional private-only frontier expansion action. It uses existing
guarded private runners and preserves the public calibration boundary.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PUBLIC_FLOOR = 0.70

DEFAULT_OUT = REPORTS / "private_generalization_frontier_expander_v1.json"
DEFAULT_MARKDOWN = REPORTS / "private_generalization_frontier_expander_v1.md"
DEFAULT_QUEUE = REPORTS / "private_generalization_frontier_expander_v1_queue.jsonl"

PUBLIC_LOCK = REPORTS / "public_calibration_operator_lock.flag"
PRE_PUBLIC_AUDIT = REPORTS / "pre_public_generalization_readiness_audit.json"
GOVERNOR = REPORTS / "theseus_generalization_governor_v1.json"
GOVERNOR_MARKDOWN = REPORTS / "theseus_generalization_governor_v1.md"
GOVERNOR_QUEUE = REPORTS / "theseus_generalization_governor_v1_queue.jsonl"
PRE_PUBLIC_AUDIT_MARKDOWN = REPORTS / "pre_public_generalization_readiness_audit.md"
PRE_PUBLIC_AUDIT_QUEUE = REPORTS / "pre_public_generalization_readiness_audit_queue.jsonl"
UNSEEN_CHALLENGE = REPORTS / "private_unseen_transfer_challenge_v1.json"
RESIDUAL_FRONTIER = REPORTS / "private_residual_frontier_v1.json"
V5_REFRESH = REPORTS / "private_ecology_generalization_v5_refresh.json"
AGENT_REFRESH = REPORTS / "agent_lane_private_refresh.json"

FORBIDDEN_POST_V4_PUBLIC_ARTIFACTS = [
    REPORTS / "real_code_benchmark_graduation_post_v4_seed23_5x32.json",
    REPORTS / "real_code_benchmark_traces_post_v4_seed23_5x32.jsonl",
    REPORTS / "student_code_candidates_post_v4_seed23_5x32.jsonl",
    REPORTS / "operator_bounded_public_calibration_post_v4_seed23_5x32.json",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-actions", type=int, default=1)
    parser.add_argument("--allow-battery", action="store_true")
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--queue-out", default=rel(DEFAULT_QUEUE))
    args = parser.parse_args()

    report = build_report(args)
    if args.execute and report["decision"]["kind"] == "run_private_frontier_expansion":
        report["execution"] = execute_queue(report["queue"], max_actions=max(0, int(args.max_actions)))
        report["queue"] = apply_execution_results(report["queue"], report["execution"])
        if any(row.get("returncode") == 0 for row in report["execution"]["actions"]):
            report["downstream_refresh"] = refresh_downstream_reports()
            refresh_boundary_state(report, args)
        refresh_after_execution(report, args)
        refresh_failures = failed_downstream_refresh_count(report)
        if refresh_failures:
            object_field(report, "summary")["downstream_refresh_failed_count"] = refresh_failures
            report["trigger_state"] = "YELLOW"
        if any(row.get("returncode") not in {0, None} for row in report["execution"]["actions"]):
            report["trigger_state"] = "YELLOW"

    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    write_jsonl(resolve(args.queue_out), report["queue"])
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report["trigger_state"] == "RED" else 0


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    audit = read_json(PRE_PUBLIC_AUDIT, {})
    unseen = read_json(UNSEEN_CHALLENGE, {})
    frontier = read_json(RESIDUAL_FRONTIER, {})
    v5_refresh = read_json(V5_REFRESH, {})
    agent_refresh = read_json(AGENT_REFRESH, {})

    audit_summary = object_field(audit, "summary")
    unseen_summary = object_field(unseen, "summary")
    frontier_summary = object_field(frontier, "summary")
    v5_summary = object_field(v5_refresh, "summary")
    forbidden_present = [rel(path) for path in FORBIDDEN_POST_V4_PUBLIC_ARTIFACTS if path.exists()]
    public_pass_rate = first_number(audit_summary.get("public_pass_rate"), 0.0)

    hard_gates = [
        gate("public_calibration_operator_lock_active", PUBLIC_LOCK.exists(), rel(PUBLIC_LOCK), "hard"),
        gate("pre_public_audit_present", PRE_PUBLIC_AUDIT.exists() and audit.get("trigger_state") in {"GREEN", "YELLOW"}, audit.get("trigger_state"), "hard"),
        gate("pre_public_audit_public_calibration_disallowed", audit_summary.get("public_calibration_allowed") is False, audit_summary.get("public_calibration_allowed"), "hard"),
        gate("public_transfer_still_below_floor", public_pass_rate < PUBLIC_FLOOR, {"public_pass_rate": public_pass_rate, "floor": PUBLIC_FLOOR}, "hard"),
        gate("forbidden_post_v4_public_artifacts_absent", not forbidden_present, forbidden_present, "hard"),
        gate("public_tests_and_solutions_not_used", not any_true("public_tests_used", audit) and not any_true("public_solutions_used", audit), audit_summary, "hard"),
        gate("external_inference_zero", external_call_total(audit, unseen, frontier, agent_refresh) == 0, external_call_total(audit, unseen, frontier, agent_refresh), "hard"),
    ]
    hard_failed = [row for row in hard_gates if not row["passed"]]
    queue = [] if hard_failed else build_queue(args, unseen_summary, frontier_summary, v5_summary, agent_refresh)
    decision = choose_decision(hard_failed, queue)
    trigger_state = "RED" if hard_failed else "YELLOW" if queue else "GREEN"
    report = {
        "policy": "project_theseus_private_generalization_frontier_expander_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "decision": decision,
        "summary": {
            "decision": decision["kind"],
            "public_pass_rate": public_pass_rate,
            "public_floor": PUBLIC_FLOOR,
            "operator_lock_active": PUBLIC_LOCK.exists(),
            "public_calibration_allowed": False,
            "unseen_challenge_rows": int(first_number(unseen_summary.get("challenge_row_count"), 0)),
            "unseen_challenge_pass_rate": first_number(unseen_summary.get("pass_rate"), 0.0),
            "unseen_challenge_learned_only_pass_rate": first_number(unseen_summary.get("learned_only_pass_rate"), 0.0),
            "residual_frontier_rows": int(first_number(frontier_summary.get("row_count"), 0)),
            "residual_frontier_spec_count": int(first_number(frontier_summary.get("frontier_spec_count"), 0)),
            "residual_frontier_pass_rate": first_number(frontier_summary.get("pass_rate"), 0.0),
            "residual_frontier_only_pass_rate": first_number(frontier_summary.get("frontier_only_pass_rate"), 0.0),
            "v5_refresh_heldout_rows": int(first_number(v5_summary.get("private_heldout_row_count"), 0)),
            "v5_refresh_learned_token_pass_count": int(first_number(v5_summary.get("learned_token_pass_count"), 0)),
            "v5_refresh_pass_rate": first_number(v5_summary.get("pass_rate"), 0.0),
            "v5_refresh_learned_only_pass_rate": first_number(v5_summary.get("learned_only_pass_rate"), 0.0),
            "queue_item_count": len(queue),
            "next_safe_private_action": queue[0]["kind"] if queue else "",
            "hard_failed_gate_count": len(hard_failed),
            "external_inference_calls": 0,
            "score_semantics": "private frontier expansion only; not promotion evidence and not public calibration",
        },
        "inputs": {
            "execute": bool(args.execute),
            "max_actions": int(args.max_actions),
            "allow_battery": bool(args.allow_battery),
            "pre_public_audit": rel(PRE_PUBLIC_AUDIT),
            "public_calibration": "locked",
        },
        "gates": hard_gates,
        "queue": queue,
        "rules": {
            "public_calibration": "Never executed by this expander; public lock must remain active.",
            "training": "Only existing private-only runners are eligible actions.",
            "operator_review": "If no private frontier action remains, return to the pre-public audit/operator-review path.",
        },
        "external_inference_calls": 0,
    }
    return report


def build_queue(
    args: argparse.Namespace,
    unseen_summary: dict[str, Any],
    frontier_summary: dict[str, Any],
    v5_summary: dict[str, Any],
    agent_refresh: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    unseen_rows = int(first_number(unseen_summary.get("challenge_row_count"), 0))
    frontier_rows = int(first_number(frontier_summary.get("row_count"), 0))
    frontier_spec_count = int(first_number(frontier_summary.get("frontier_spec_count"), 0))
    frontier_token_passes = int(first_number(frontier_summary.get("frontier_token_pass_count"), 0))
    frontier_pass_rate = first_number(frontier_summary.get("pass_rate"), 0.0)
    frontier_only_pass_rate = first_number(frontier_summary.get("frontier_only_pass_rate"), 0.0)
    frontier_hard_failed_gates = int(first_number(frontier_summary.get("hard_failed_gate_count"), 999))
    v5_heldout_rows = int(first_number(v5_summary.get("private_heldout_row_count"), 0))
    v5_learned_token_passes = int(first_number(v5_summary.get("learned_token_pass_count"), 0))
    agent_age = time.time() - AGENT_REFRESH.stat().st_mtime if AGENT_REFRESH.exists() else 10**9

    if unseen_rows < 240:
        rows.append(
            queue_item(
                "expand_private_unseen_transfer_challenge_240",
                "Run a larger private unseen-transfer challenge with semantic keys withheld.",
                [
                    sys.executable,
                    "scripts/private_unseen_transfer_challenge_v1.py",
                    "--execute",
                    "--rows",
                    "240",
                    "--seed",
                    "101",
                    "--max-hours",
                    "4",
                ],
                priority=10,
            )
        )
    elif frontier_spec_count < 21 and frontier_rows < 1008:
        rows.append(
            queue_item(
                "expand_private_residual_frontier_1008",
                "Run a larger aggregate-residual private frontier slice.",
                [
                    sys.executable,
                    "scripts/private_residual_frontier_v1.py",
                    "--execute",
                    "--rows",
                    "1008",
                    "--min-rows",
                    "1008",
                    "--seed",
                    "131",
                    "--max-hours",
                    "6",
                    "--shard-size",
                    "105",
                ],
                priority=20,
            )
        )
    elif unseen_rows < 360:
        rows.append(
            queue_item(
                "expand_private_unseen_transfer_challenge_360",
                "Run a second larger private unseen-transfer challenge to widen OOD coverage.",
                [
                    sys.executable,
                    "scripts/private_unseen_transfer_challenge_v1.py",
                    "--execute",
                    "--rows",
                    "360",
                    "--seed",
                    "151",
                    "--max-hours",
                    "6",
                ],
                priority=30,
            )
        )
    elif frontier_spec_count < 21 and frontier_rows < 1344:
        rows.append(
            queue_item(
                "expand_private_residual_frontier_1344",
                "Run another larger private residual-frontier slice before spending public calibration.",
                [
                    sys.executable,
                    "scripts/private_residual_frontier_v1.py",
                    "--execute",
                    "--rows",
                    "1344",
                    "--min-rows",
                    "1344",
                    "--seed",
                    "173",
                    "--max-hours",
                    "8",
                    "--shard-size",
                    "105",
                ],
                priority=40,
            )
        )
    elif frontier_spec_count < 21 or (
        frontier_spec_count <= 21 and (frontier_rows < 840 or frontier_token_passes < 840)
    ):
        rows.append(
            queue_item(
                "expand_private_residual_frontier_840_spec21",
                "Run the broader 21-spec private residual-frontier shard to stress verifier mismatch, stdin parsing, return-shape, interface, and DP transfer before public review.",
                [
                    sys.executable,
                    "scripts/private_residual_frontier_v1.py",
                    "--execute",
                    "--rows",
                    "840",
                    "--min-rows",
                    "840",
                    "--seed",
                    "197",
                    "--max-hours",
                    "6",
                    "--shard-size",
                    "105",
                ],
                priority=44,
            )
        )
    elif (
        frontier_rows < 1040
        or frontier_spec_count < 26
        or frontier_pass_rate < 1.0
        or frontier_only_pass_rate < 0.70
        or frontier_token_passes <= 0
        or frontier_hard_failed_gates > 0
    ):
        rows.append(
            queue_item(
                "expand_private_residual_frontier_1040_spec26",
                "Run the broader 26-spec private residual-frontier shard to add graph, shortest-path, string-DP, stack, and record-grouping transfer pressure before public review.",
                [
                    sys.executable,
                    "scripts/private_residual_frontier_v1.py",
                    "--execute",
                    "--rows",
                    "1040",
                    "--min-rows",
                    "1040",
                    "--seed",
                    "223",
                    "--max-hours",
                    "8",
                    "--shard-size",
                    "104",
                ],
                priority=45,
            )
        )
    elif v5_heldout_rows < 720 or v5_learned_token_passes < 720:
        rows.append(
            queue_item(
                "expand_private_ecology_v5_720",
                "Run the larger private ecology v5 refresh so workflow/tool/storage/spatial transfer is proven beyond the 480-row slice.",
                [
                    sys.executable,
                    "scripts/private_ecology_generalization_v5_refresh.py",
                    "--execute",
                    "--train-rows",
                    "1800",
                    "--heldout-rows",
                    "720",
                    "--private-eval-limit",
                    "720",
                    "--max-hours",
                    "8",
                ],
                priority=45,
            )
        )
    elif agent_age > 24 * 3600:
        rows.append(
            queue_item(
                "refresh_agent_lane_private_transfer",
                "Refresh private tool-use, RL/conversation, and cross-domain STS agent transfer evidence.",
                [
                    sys.executable,
                    "scripts/agent_lane_private_refresh.py",
                    "--max-tool-cases",
                    "64",
                    "--max-capsules",
                    "256",
                ],
                priority=50,
            )
        )

    if args.allow_battery:
        for row in rows:
            if row["kind"].startswith("expand_private_"):
                row["command"].append("--allow-battery")
    return rows


def choose_decision(hard_failed: list[dict[str, Any]], queue: list[dict[str, Any]]) -> dict[str, Any]:
    if hard_failed:
        return {
            "kind": "stop_hard_safety_or_boundary_failure",
            "reason": "A hard public-boundary invariant failed; do not expand private frontier work.",
            "public_calibration_allowed": False,
        }
    if queue:
        return {
            "kind": "run_private_frontier_expansion",
            "reason": "Public transfer remains below floor and public calibration is locked, so run the next safe private frontier action.",
            "public_calibration_allowed": False,
        }
    return {
        "kind": "no_private_frontier_action_remaining",
        "reason": "Configured private frontier expansion targets are satisfied; return to operator public-review gate.",
        "public_calibration_allowed": False,
    }


def execute_queue(queue: list[dict[str, Any]], *, max_actions: int) -> dict[str, Any]:
    actions = []
    for row in [item for item in queue if item.get("status") == "pending"][:max_actions]:
        start = time.time()
        result = subprocess.run(
            row["command"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10 * 3600,
        )
        actions.append(
            {
                "kind": row.get("kind"),
                "command": row.get("command"),
                "returncode": result.returncode,
                "elapsed_seconds": round(time.time() - start, 3),
                "stdout_tail": result.stdout[-2000:],
                "stderr_tail": result.stderr[-2000:],
            }
        )
    return {
        "created_utc": now(),
        "action_count": len(actions),
        "actions": actions,
        "public_calibration_allowed": False,
        "external_inference_calls": 0,
    }


def apply_execution_results(queue: list[dict[str, Any]], execution: dict[str, Any]) -> list[dict[str, Any]]:
    by_kind = {row.get("kind"): row for row in execution.get("actions", [])}
    out = []
    for row in queue:
        row = dict(row)
        result = by_kind.get(row.get("kind"))
        if result:
            row["status"] = "completed" if result.get("returncode") == 0 else "failed"
            row["returncode"] = result.get("returncode")
            row["elapsed_seconds"] = result.get("elapsed_seconds")
        out.append(row)
    return out


def refresh_downstream_reports() -> dict[str, Any]:
    commands = [
        [
            sys.executable,
            "scripts/theseus_generalization_governor_v1.py",
            "--out",
            rel(GOVERNOR),
            "--markdown-out",
            rel(GOVERNOR_MARKDOWN),
            "--queue-out",
            rel(GOVERNOR_QUEUE),
        ],
        [
            sys.executable,
            "scripts/pre_public_generalization_readiness_audit.py",
            "--out",
            rel(PRE_PUBLIC_AUDIT),
            "--markdown-out",
            rel(PRE_PUBLIC_AUDIT_MARKDOWN),
            "--queue-out",
            rel(PRE_PUBLIC_AUDIT_QUEUE),
        ],
    ]
    rows = []
    for command in commands:
        start = time.time()
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30 * 60,
        )
        rows.append(
            {
                "command": command,
                "returncode": result.returncode,
                "elapsed_seconds": round(time.time() - start, 3),
                "stdout_tail": result.stdout[-2000:],
                "stderr_tail": result.stderr[-2000:],
            }
        )
    return {
        "created_utc": now(),
        "action_count": len(rows),
        "actions": rows,
        "public_calibration_allowed": False,
        "external_inference_calls": 0,
    }


def refresh_boundary_state(report: dict[str, Any], args: argparse.Namespace) -> None:
    current = build_report(args)
    report["decision"] = current["decision"]
    report["gates"] = current["gates"]
    report["rules"] = current["rules"]
    report["trigger_state"] = current["trigger_state"]
    summary = object_field(report, "summary")
    current_summary = object_field(current, "summary")
    for key in [
        "decision",
        "public_pass_rate",
        "public_floor",
        "operator_lock_active",
        "public_calibration_allowed",
        "hard_failed_gate_count",
        "external_inference_calls",
        "score_semantics",
    ]:
        summary[key] = current_summary.get(key)
    report["summary"] = summary


def failed_downstream_refresh_count(report: dict[str, Any]) -> int:
    refresh = report.get("downstream_refresh") or {}
    return sum(1 for row in refresh.get("actions", []) or [] if row.get("returncode") != 0)


def refresh_after_execution(report: dict[str, Any], args: argparse.Namespace) -> None:
    unseen_summary = object_field(read_json(UNSEEN_CHALLENGE, {}), "summary")
    frontier_summary = object_field(read_json(RESIDUAL_FRONTIER, {}), "summary")
    v5_summary = object_field(read_json(V5_REFRESH, {}), "summary")
    pending = build_queue(args, unseen_summary, frontier_summary, v5_summary, read_json(AGENT_REFRESH, {}))
    completed_kinds = {row.get("kind") for row in report.get("queue") or [] if row.get("status") == "completed"}
    report["queue"].extend(row for row in pending if row.get("kind") not in completed_kinds)
    summary = object_field(report, "summary")
    summary["unseen_challenge_rows"] = int(first_number(unseen_summary.get("challenge_row_count"), 0))
    summary["unseen_challenge_pass_rate"] = first_number(unseen_summary.get("pass_rate"), 0.0)
    summary["unseen_challenge_learned_only_pass_rate"] = first_number(unseen_summary.get("learned_only_pass_rate"), 0.0)
    summary["residual_frontier_rows"] = int(first_number(frontier_summary.get("row_count"), 0))
    summary["residual_frontier_spec_count"] = int(first_number(frontier_summary.get("frontier_spec_count"), 0))
    summary["residual_frontier_pass_rate"] = first_number(frontier_summary.get("pass_rate"), 0.0)
    summary["residual_frontier_only_pass_rate"] = first_number(frontier_summary.get("frontier_only_pass_rate"), 0.0)
    summary["v5_refresh_heldout_rows"] = int(first_number(v5_summary.get("private_heldout_row_count"), 0))
    summary["v5_refresh_learned_token_pass_count"] = int(first_number(v5_summary.get("learned_token_pass_count"), 0))
    summary["v5_refresh_pass_rate"] = first_number(v5_summary.get("pass_rate"), 0.0)
    summary["v5_refresh_learned_only_pass_rate"] = first_number(v5_summary.get("learned_only_pass_rate"), 0.0)
    summary["queue_item_count"] = len(report.get("queue") or [])
    summary["completed_queue_item_count"] = sum(1 for row in report.get("queue") or [] if row.get("status") == "completed")
    summary["failed_queue_item_count"] = sum(1 for row in report.get("queue") or [] if row.get("status") == "failed")
    next_pending = next((row for row in report.get("queue") or [] if row.get("status") == "pending" and row.get("command")), {})
    summary["next_safe_private_action"] = next_pending.get("kind", "")
    summary["external_inference_calls"] = 0
    report["summary"] = summary


def queue_item(kind: str, title: str, command: list[str], *, priority: int) -> dict[str, Any]:
    return {
        "policy": "project_theseus_private_generalization_frontier_expander_queue_item_v1",
        "queue": "private_generalization_frontier_expander_v1",
        "kind": kind,
        "title": title,
        "priority": int(priority),
        "command": command,
        "status": "pending",
        "safe_to_execute_without_operator_public_approval": True,
        "requires_operator_public_unlock": False,
        "public_calibration_allowed": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = object_field(report, "summary")
    failed = [row for row in report.get("gates", []) if not row.get("passed")]
    lines = [
        "# Private Generalization Frontier Expander v1",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- decision: `{summary.get('decision')}`",
        f"- public_pass_rate: `{summary.get('public_pass_rate')}` floor=`{summary.get('public_floor')}`",
        f"- unseen_challenge_rows: `{summary.get('unseen_challenge_rows')}`",
        f"- residual_frontier_rows: `{summary.get('residual_frontier_rows')}`",
        f"- v5_refresh_heldout_rows: `{summary.get('v5_refresh_heldout_rows')}`",
        f"- v5_refresh_learned_token_pass_count: `{summary.get('v5_refresh_learned_token_pass_count')}`",
        f"- next_safe_private_action: `{summary.get('next_safe_private_action')}`",
        f"- public_calibration_allowed: `{summary.get('public_calibration_allowed')}`",
        f"- external_inference_calls: `{summary.get('external_inference_calls')}`",
        "",
        "## Failed Gates",
        "",
    ]
    if failed:
        for row in failed:
            lines.append(f"- `{row.get('gate')}` ({row.get('severity')})")
    else:
        lines.append("- None.")
    lines.extend(["", "## Queue", ""])
    for row in report.get("queue", []) or []:
        lines.append(f"- `{row.get('kind')}`: {row.get('title')} Command: `{' '.join(str(item) for item in row.get('command') or [])}`")
    refresh = report.get("downstream_refresh") or {}
    if refresh:
        lines.extend(["", "## Downstream Refresh", ""])
        for row in refresh.get("actions", []) or []:
            lines.append(f"- returncode `{row.get('returncode')}`: `{' '.join(str(item) for item in row.get('command') or [])}`")
    lines.append("")
    return "\n".join(lines)


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def any_true(key: str, report: dict[str, Any]) -> bool:
    if report.get(key) is True:
        return True
    if object_field(report, "summary").get(key) is True:
        return True
    return False


def external_call_total(*reports: dict[str, Any]) -> int:
    total = 0
    for report in reports:
        total += int(first_number(report.get("external_inference_calls"), 0))
        total += int(first_number(object_field(report, "summary").get("external_inference_calls"), 0))
    return total


def first_number(*values: Any) -> float:
    for value in values:
        try:
            if value is not None and value != "":
                return float(value)
        except (TypeError, ValueError):
            pass
    return 0.0


def object_field(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key) if isinstance(value, dict) else None
    return item if isinstance(item, dict) else {}


def read_json(path: Path, default: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
