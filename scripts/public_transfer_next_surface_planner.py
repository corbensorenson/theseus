#!/usr/bin/env python3
"""Plan or materialize the next clean public-transfer calibration surface.

This is a no-score tool. It never runs public calibration, never writes
training rows, and never exports public prompts, tests, solutions, traces, or
answer templates. Its job is to keep the public-transfer lane moving without
rerunning a consumed surface.
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
DEFAULT_POLICY = ROOT / "configs" / "permissive_growth_policy.json"
DEFAULT_REGISTRY = REPORTS / "public_benchmark_run_registry.jsonl"
DEFAULT_CARDS = "source_mbpp,source_evalplus,source_bigcodebench,source_human_eval,source_livecodebench"
LEGACY_CONSUMED_MANIFESTS = [
    "reports/public_wide_slice_manifest_seed23_5x32.jsonl",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=rel(DEFAULT_POLICY))
    parser.add_argument("--registry", default=rel(DEFAULT_REGISTRY))
    parser.add_argument("--cards", default=DEFAULT_CARDS)
    parser.add_argument("--cases-per-card", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=512)
    parser.add_argument("--slug-prefix", default="public_transfer_lift")
    parser.add_argument("--private-recovery", default="reports/full_body_contract_transfer_recovery_v2_private320.json")
    parser.add_argument("--operator-lock", default="reports/public_calibration_operator_lock.flag")
    parser.add_argument("--exclude-manifest", action="append", default=[])
    parser.add_argument("--materialize", action="store_true", help="Write the selected case manifest and frozen packet.")
    parser.add_argument("--selector-timeout-seconds", type=int, default=1200)
    parser.add_argument("--packet-timeout-seconds", type=int, default=1200)
    parser.add_argument("--out", default="reports/public_transfer_next_surface_planner.json")
    parser.add_argument("--markdown-out", default="reports/public_transfer_next_surface_planner.md")
    args = parser.parse_args()

    report = build_report(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report["trigger_state"] == "RED" else 0


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    policy_path = resolve(args.policy)
    registry_path = resolve(args.registry)
    policy = read_json(policy_path, {})
    registry_rows = read_jsonl(registry_path)
    consumed_rows = [row for row in registry_rows if row.get("consumed") is True]
    cards = [card.strip() for card in str(args.cards).split(",") if card.strip()]
    registry_status = public_registry_status(policy, consumed_rows)
    excluded_manifests = sorted(
        {
            *LEGACY_CONSUMED_MANIFESTS,
            *consumed_case_manifest_paths(consumed_rows),
            *[part.strip() for value in args.exclude_manifest for part in str(value).split(",") if part.strip()],
        }
    )
    consumed_run_ids = {
        str(row.get("run_id") or row.get("surface_slug") or "")
        for row in consumed_rows
        if str(row.get("run_id") or row.get("surface_slug") or "")
    }
    consumed_seeds = {
        int(command_arg([str(part) for part in row.get("command", [])], "--seed"))
        for row in consumed_rows
        if command_arg([str(part) for part in row.get("command", [])], "--seed").isdigit()
    }
    proposal = select_proposal(
        cards=cards,
        cases_per_card=max(1, int(args.cases_per_card)),
        seed_start=int(args.seed_start),
        seed_end=int(args.seed_end),
        slug_prefix=str(args.slug_prefix),
        consumed_run_ids=consumed_run_ids,
        consumed_seeds=consumed_seeds,
    )
    materialize_result: dict[str, Any] | None = None
    selector_report: dict[str, Any] = {}
    packet: dict[str, Any] = {}
    if args.materialize and proposal:
        materialize_result = materialize_surface(
            proposal,
            cards=cards,
            excluded_manifests=excluded_manifests,
            private_recovery=str(args.private_recovery),
            operator_lock=str(args.operator_lock),
            policy=str(args.policy),
            registry=str(args.registry),
            selector_timeout_seconds=max(60, int(args.selector_timeout_seconds)),
            packet_timeout_seconds=max(60, int(args.packet_timeout_seconds)),
        )
        selector_report = read_json(resolve(proposal["case_manifest_report"]), {})
        packet = read_json(resolve(proposal["packet"]), {})
    source_capacity = source_capacity_summary(selector_report)

    hard_blockers: list[str] = []
    if not proposal:
        hard_blockers.append("no_unconsumed_surface_candidate_found")
    if args.materialize and materialize_result and materialize_result.get("returncode") not in {0, None}:
        hard_blockers.append("materialization_command_failed")
    if args.materialize and source_capacity["insufficient_card_count"]:
        hard_blockers.append("insufficient_disjoint_public_cases_after_consumed_surfaces")
    if args.materialize and packet and packet.get("trigger_state") != "GREEN" and not source_capacity["insufficient_card_count"]:
        hard_blockers.append("materialized_packet_not_green")

    trigger_state = "RED" if hard_blockers else ("GREEN" if args.materialize and packet.get("trigger_state") == "GREEN" else "YELLOW")
    summary = {
        "mode": "materialize" if args.materialize else "dry_plan",
        "run_registry_execution_enabled": registry_status["run_registry_execution_enabled"],
        "authorization_mode": registry_status["authorization_mode"],
        "time_period_run_cap_enabled": False,
        "calendar_throttle_enabled": False,
        "consumed_run_count_total_for_audit": registry_status["consumed_run_count_total_for_audit"],
        "fresh_surfaces_calendar_throttled": False,
        "fresh_surface_execution_policy": "run_immediately_when_frozen_registry_surface_is_clean",
        "consumed_surface_count": len(consumed_rows),
        "consumed_run_ids": sorted(consumed_run_ids),
        "consumed_seeds": sorted(consumed_seeds),
        "excluded_manifest_count": len(excluded_manifests),
        "proposed_slug": proposal.get("slug") if proposal else "",
        "proposed_seed": proposal.get("seed") if proposal else None,
        "proposed_task_count": len(cards) * max(1, int(args.cases_per_card)),
        "materialized_packet_state": packet.get("trigger_state"),
        "materialized_selector_state": selector_report.get("trigger_state"),
        "source_capacity": source_capacity,
        "no_public_calibration_executed": True,
        "training_rows_written": 0,
        "external_inference_calls": 0,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }
    return {
        "policy": "project_theseus_public_transfer_next_surface_planner_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": summary,
        "run_registry": registry_status,
        "proposal": proposal or {},
        "excluded_manifests": excluded_manifests,
        "materialize_result": materialize_result or {"status": "not_requested"},
        "packet_summary": packet.get("summary") if isinstance(packet.get("summary"), dict) else {},
        "source_capacity": source_capacity,
        "hard_blockers": hard_blockers,
        "next_actions": next_actions(args.materialize, proposal, packet, registry_status, hard_blockers),
        "rules": {
            "public_benchmarks_are_calibration_only": True,
            "rerun_consumed_surface": False,
            "train_on_public_prompts_tests_solutions_traces_or_scores": False,
            "public_calibration_executed_by_this_script": False,
            "candidate_generation_executed_by_this_script": False,
            "run_registry_required_for_metadata_planning": False,
            "run_registry_required_for_duplicate_prevention": True,
        },
        "external_inference_calls": 0,
    }


def select_proposal(
    *,
    cards: list[str],
    cases_per_card: int,
    seed_start: int,
    seed_end: int,
    slug_prefix: str,
    consumed_run_ids: set[str],
    consumed_seeds: set[int],
) -> dict[str, Any]:
    card_count = len(cards)
    for seed in range(max(0, seed_start), max(seed_start, seed_end) + 1):
        if seed in consumed_seeds:
            continue
        slug = clean_slug(f"{slug_prefix}_seed{seed}_{card_count}x{cases_per_card}")
        if slug in consumed_run_ids:
            continue
        paths = {
            "case_manifest": f"reports/public_wide_slice_manifest_{slug}.jsonl",
            "case_manifest_report": f"reports/public_wide_slice_selector_{slug}.json",
            "case_manifest_markdown": f"reports/public_wide_slice_selector_{slug}.md",
            "candidate_manifest": f"reports/student_code_candidates_{slug}.jsonl",
            "calibration_out": f"reports/real_code_benchmark_graduation_{slug}.json",
            "trace_out": f"reports/real_code_benchmark_traces_{slug}.jsonl",
            "transfer_artifact": f"reports/transfer_artifacts/code/real_code_benchmark_graduation_{slug}_transfer_artifact.json",
            "packet": f"reports/public_transfer_readiness_packet_{slug}.json",
            "packet_markdown": f"reports/public_transfer_readiness_packet_{slug}.md",
        }
        if any(resolve(path).exists() for key, path in paths.items() if key in {"candidate_manifest", "calibration_out", "trace_out"}):
            continue
        return {
            "slug": slug,
            "seed": seed,
            "cards": cards,
            "cases_per_card": cases_per_card,
            "total_task_count": len(cards) * cases_per_card,
            **paths,
        }
    return {}


def materialize_surface(
    proposal: dict[str, Any],
    *,
    cards: list[str],
    excluded_manifests: list[str],
    private_recovery: str,
    operator_lock: str,
    policy: str,
    registry: str,
    selector_timeout_seconds: int,
    packet_timeout_seconds: int,
) -> dict[str, Any]:
    selector_cmd = [
        sys.executable,
        "scripts/wide_public_slice_selector.py",
        "--cards",
        ",".join(cards),
        "--seed",
        str(proposal["seed"]),
        "--cases-per-card",
        str(proposal["cases_per_card"]),
        "--out",
        proposal["case_manifest"],
        "--report-out",
        proposal["case_manifest_report"],
        "--markdown-out",
        proposal["case_manifest_markdown"],
    ]
    for path in excluded_manifests:
        selector_cmd.extend(["--exclude-manifest", path])
    selector = run_cmd(selector_cmd, selector_timeout_seconds)
    if selector["returncode"] != 0:
        return {"status": "selector_failed", "selector": selector}

    packet_cmd = [
        sys.executable,
        "scripts/public_transfer_lift_v2_packet.py",
        "--slug",
        proposal["slug"],
        "--seed",
        str(proposal["seed"]),
        "--cases-per-card",
        str(proposal["cases_per_card"]),
        "--cards",
        ",".join(cards),
        "--case-manifest",
        proposal["case_manifest"],
        "--case-manifest-report",
        proposal["case_manifest_report"],
        "--private-recovery",
        private_recovery,
        "--policy",
        policy,
        "--registry",
        registry,
        "--operator-lock",
        operator_lock,
        "--out",
        proposal["packet"],
        "--markdown-out",
        proposal["packet_markdown"],
    ]
    for path in excluded_manifests:
        packet_cmd.extend(["--exclude-manifest", path])
    packet = run_cmd(packet_cmd, packet_timeout_seconds)
    status = "completed" if packet["returncode"] == 0 else "packet_failed"
    return {"status": status, "returncode": packet["returncode"], "selector": selector, "packet": packet}


def run_cmd(command: list[str], timeout_seconds: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout_seconds)
        return {
            "command": command,
            "returncode": result.returncode,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": None,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
            "timed_out": True,
        }


def public_registry_status(policy: dict[str, Any], consumed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    public_policy = as_dict(policy.get("public_benchmarks"))
    execution_default = str(public_policy.get("execution_default") or "")
    enabled = execution_default in {
        "governed_measurement_run_registry",
        "governed_run_registry",
    }
    return {
        "run_registry_execution_enabled": enabled,
        "authorization_mode": "run_registry" if enabled else "legacy_approval_file",
        "time_period_run_cap_enabled": False,
        "calendar_throttle_enabled": False,
        "consumed_run_count_total_for_audit": len(consumed_rows),
        "fresh_surfaces_calendar_throttled": False,
        "fresh_surface_execution_policy": "run_immediately_when_frozen_registry_surface_is_clean",
        "per_surface_max_runs": int_or_zero(public_policy.get("per_surface_max_runs")) or 1,
    }


def consumed_case_manifest_paths(rows: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for row in rows:
        command = row.get("command") if isinstance(row.get("command"), list) else []
        manifest = command_arg([str(part) for part in command], "--case-manifest")
        if (
            manifest
            and manifest.startswith("reports/public_wide_slice_manifest_")
            and manifest.endswith(".jsonl")
            and manifest not in seen
        ):
            paths.append(manifest)
            seen.add(manifest)
    return paths


def command_arg(command: list[str], flag: str) -> str:
    try:
        index = command.index(flag)
    except ValueError:
        return ""
    if index + 1 >= len(command):
        return ""
    return str(command[index + 1])


def next_actions(
    materialize: bool,
    proposal: dict[str, Any],
    packet: dict[str, Any],
    registry_status: dict[str, Any],
    hard_blockers: list[str],
) -> list[str]:
    if "insufficient_disjoint_public_cases_after_consumed_surfaces" in hard_blockers:
        return [
            "Do not rerun the consumed public tasks for a new score; the local disjoint 5x64 surface is exhausted for some cards.",
            "Stage or freeze additional legitimate public calibration sources before another large apples-to-apples measurement, or use the remaining rows only as a clearly labeled small diagnostic.",
            "Focus model work on private residual repair until a fresh public surface with enough unused rows is available.",
        ]
    if hard_blockers:
        return [
            "Resolve the listed hard blockers before running another public calibration measurement.",
            "Do not rerun an already consumed surface.",
        ]
    if materialize and packet.get("trigger_state") == "GREEN":
        return [
            f"Run guarded dry-run against {proposal.get('packet')} to confirm execute gates.",
            "If the dry-run would execute, the guarded runner may record exactly one new public calibration measurement for this slug.",
            "After execution, mine only residual categories; do not train on public payloads.",
        ]
    if registry_status.get("run_registry_execution_enabled") and proposal:
        return [
            "Rerun this planner with --materialize to write the next disjoint manifest and readiness packet.",
            "Then run operator_bounded_public_calibration.py in dry-run mode against that packet.",
        ]
    return ["No clean public-transfer surface is currently available."]


def source_capacity_summary(selector_report: dict[str, Any]) -> dict[str, Any]:
    cards: list[dict[str, Any]] = []
    insufficient: list[dict[str, Any]] = []
    for row in selector_report.get("cards_report", []) if isinstance(selector_report.get("cards_report"), list) else []:
        card = {
            "card_id": row.get("card_id"),
            "available_after_exclusions": row.get("available_probe_task_count"),
            "available_before_exclusions": row.get("available_probe_task_count_before_exclusions"),
            "excluded_probe_task_count": row.get("excluded_probe_task_count"),
            "required_task_count": row.get("required_task_count"),
            "selected_task_count": row.get("selected_task_count"),
            "ready": row.get("ready_for_wide_public_calibration"),
            "blockers": row.get("blockers") if isinstance(row.get("blockers"), list) else [],
        }
        cards.append(card)
        if int_or_zero(card["available_after_exclusions"]) < int_or_zero(card["required_task_count"]):
            insufficient.append(card)
    balanced_max = min((int_or_zero(card["available_after_exclusions"]) for card in cards), default=0)
    return {
        "balanced_max_cases_per_card_after_exclusions": balanced_max,
        "insufficient_card_count": len(insufficient),
        "insufficient_cards": insufficient,
        "cards": cards,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = as_dict(report.get("summary"))
    proposal = as_dict(report.get("proposal"))
    lines = [
        "# Public Transfer Next Surface Planner",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- mode: `{summary.get('mode')}`",
        f"- run registry: `{summary.get('authorization_mode')}`",
        f"- consumed runs: `{summary.get('consumed_surface_count')}`",
        f"- proposed slug: `{summary.get('proposed_slug')}`",
        f"- proposed seed: `{summary.get('proposed_seed')}`",
        f"- proposed tasks: `{summary.get('proposed_task_count')}`",
        "",
        "## Rules",
        "",
        "- This planner does not run public calibration.",
        "- Consumed surfaces are excluded from future selector manifests.",
        "- Public prompts, tests, solutions, traces, score labels, and answer templates are not training data.",
        "",
        "## Artifacts",
        "",
    ]
    if proposal:
        for key in ["case_manifest", "case_manifest_report", "packet"]:
            lines.append(f"- `{key}`: `{proposal.get(key)}`")
    else:
        lines.append("- No proposal was available.")
    lines.extend(["", "## Next Actions", ""])
    for action in report.get("next_actions", []):
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def clean_slug(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value).strip())
    return out.strip("_") or "public_transfer_lift"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
