#!/usr/bin/env python3
"""Fast benchmark measurement status and fresh-surface planner.

This command is deliberately not a training command. It lets Theseus inspect
public benchmark measurement state, stage metadata-only public case manifests,
and report the exact guarded command for a real score run without writing public
benchmark payloads into training rows.
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
DEFAULT_REGISTRY = REPORTS / "public_benchmark_run_registry.jsonl"
DEFAULT_CARDS = "source_mbpp,source_evalplus,source_bigcodebench,source_human_eval,source_livecodebench"
DEFAULT_OUT = REPORTS / "theseus_benchmark_measurement.json"
DEFAULT_MD = REPORTS / "theseus_benchmark_measurement.md"
LEGACY_CONSUMED_MANIFESTS = [
    "reports/public_wide_slice_manifest_seed23_5x32.jsonl",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=rel(DEFAULT_REGISTRY))
    parser.add_argument("--cards", default=DEFAULT_CARDS)
    parser.add_argument("--cases-per-card", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=512)
    parser.add_argument("--slug-prefix", default="public_transfer_measurement")
    parser.add_argument("--refresh-capacity", action="store_true")
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument(
        "--diagnostic-if-needed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If a full balanced slice is unavailable, plan the largest clearly labeled diagnostic slice.",
    )
    parser.add_argument("--execute", action="store_true", help="Run the guarded public measurement after materialization succeeds.")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    args = parser.parse_args()

    report = build_report(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report["trigger_state"] == "RED" else 0


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    cards = [part.strip() for part in str(args.cards).split(",") if part.strip()]
    registry_rows = read_jsonl(resolve(args.registry))
    consumed_rows = [row for row in registry_rows if row.get("consumed") is True]
    latest = latest_consumed(consumed_rows)
    previous = previous_consumed(consumed_rows, latest)

    capacity_report = load_capacity_report()
    capacity_result: dict[str, Any] = {"status": "not_refreshed"}
    if args.refresh_capacity or args.materialize:
        capacity_result = run_capacity_probe(
            cards=cards,
            cases_per_card=max(1, int(args.cases_per_card)),
            seed=max(0, int(args.seed_start)),
            registry_rows=consumed_rows,
        )
        capacity_report = read_json(resolve(capacity_result.get("report_out", "")), {})
    capacity = source_capacity_summary(capacity_report)
    requested_cases = max(1, int(args.cases_per_card))
    measurement_plan = choose_measurement_plan(
        requested_cards=cards,
        capacity=capacity,
        requested_cases=requested_cases,
        diagnostic_if_needed=bool(args.diagnostic_if_needed),
    )
    balanced_max = int_or_zero(measurement_plan.get("balanced_max_cases_per_card_after_exclusions"))
    effective_cases = int_or_zero(measurement_plan.get("effective_cases_per_card")) or requested_cases
    effective_cards = [str(card) for card in measurement_plan.get("effective_cards", [])]
    measurement_kind = str(measurement_plan.get("measurement_kind") or "unavailable")

    plan_result: dict[str, Any] = {"status": "not_requested"}
    planner_report: dict[str, Any] = {}
    operator_report: dict[str, Any] = {}
    if args.materialize and measurement_kind != "unavailable":
        plan_result = run_next_surface_planner(
            cases_per_card=effective_cases,
            seed_start=max(0, int(args.seed_start)),
            seed_end=max(int(args.seed_start), int(args.seed_end)),
            cards=effective_cards,
            slug_prefix=diagnostic_slug_prefix(args.slug_prefix, measurement_kind),
            out="reports/theseus_benchmark_public_plan.json",
            markdown_out="reports/theseus_benchmark_public_plan.md",
            materialize=True,
        )
        planner_report = read_json(REPORTS / "theseus_benchmark_public_plan.json", {})
    proposal = as_dict(planner_report.get("proposal")) if planner_report else {}
    packet_path = str(proposal.get("packet") or "")

    if args.execute:
        if not args.materialize:
            operator_report = {
                "ok": False,
                "status": "not_run",
                "reason": "execute_requires_materialize",
            }
        elif not packet_path:
            operator_report = {
                "ok": False,
                "status": "not_run",
                "reason": "materialized_packet_missing",
            }
        else:
            operator_report = run_operator(
                packet=packet_path,
                timeout_seconds=max(60, int(args.timeout_seconds)),
            )
    executed_run = executed_public_run_summary(operator_report, proposal)
    latest_for_summary = executed_run or public_run_summary(latest)
    previous_for_summary = public_run_summary(latest) if executed_run else public_run_summary(previous)

    public_rows_written = max_public_training_rows(latest, previous, planner_report, operator_report)
    external_calls = max_external_inference(latest, previous, planner_report, operator_report)
    duplicate_surface = duplicate_surface_risk(proposal, consumed_rows)
    hard_failures = []
    if public_rows_written:
        hard_failures.append("public_training_rows_detected")
    if external_calls:
        hard_failures.append("external_inference_detected")
    if duplicate_surface:
        hard_failures.append("proposed_surface_already_consumed")
    if args.execute and not operator_report.get("summary", {}).get("executed"):
        hard_failures.append("execute_requested_but_not_completed")

    summary = {
        "latest_public_run": latest_for_summary,
        "previous_public_run": previous_for_summary,
        "consumed_public_run_count": len(consumed_rows),
        "requested_cases_per_card": requested_cases,
        "balanced_max_cases_per_card_after_exclusions": balanced_max,
        "effective_cases_per_card": effective_cases,
        "requested_cards": cards,
        "effective_cards": effective_cards,
        "omitted_cards": measurement_plan.get("omitted_cards", []),
        "measurement_kind": measurement_kind,
        "fresh_headline_surface_available": balanced_max >= requested_cases,
        "diagnostic_surface_available": measurement_kind in {"diagnostic", "partial_card_diagnostic"},
        "partial_card_diagnostic_available": measurement_kind == "partial_card_diagnostic",
        "materialized": bool(plan_result.get("status") == "completed"),
        "planner_state": planner_report.get("trigger_state"),
        "proposed_slug": proposal.get("slug", ""),
        "proposed_task_count": proposal.get("total_task_count", 0),
        "packet": packet_path,
        "execute_requested": bool(args.execute),
        "executed": bool(operator_report.get("summary", {}).get("executed")),
        "model_only_and_tool_assisted_scores_separate": True,
        "public_training_rows_written": public_rows_written,
        "external_inference_calls": external_calls,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }
    trigger_state = "RED" if hard_failures else ("YELLOW" if measurement_kind != "headline" else "GREEN")
    return {
        "policy": "project_theseus_benchmark_measurement_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": summary,
        "source_capacity": capacity,
        "capacity_result": capacity_result,
        "planner_result": plan_result,
        "planner_report_summary": planner_report.get("summary") if isinstance(planner_report.get("summary"), dict) else {},
        "operator_report_summary": operator_report.get("summary") if isinstance(operator_report.get("summary"), dict) else {},
        "hard_failures": hard_failures,
        "next_actions": next_actions(summary, capacity, proposal, args.execute),
        "rules": {
            "public_benchmarks_are_measurement_only": True,
            "train_on_public_prompts_tests_hidden_tests_solutions_traces_or_answer_templates": False,
            "rerun_consumed_surface_for_score_fishing": False,
            "case_manifests_are_metadata_only": True,
            "teacher_runtime_serving": False,
        },
        "external_inference_calls": external_calls,
        "public_training_rows_written": public_rows_written,
        "ok": not hard_failures,
    }


def run_capacity_probe(*, cards: list[str], cases_per_card: int, seed: int, registry_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = REPORTS / "theseus_benchmark_public_capacity_manifest.jsonl"
    report_out = REPORTS / "theseus_benchmark_public_capacity.json"
    md = REPORTS / "theseus_benchmark_public_capacity.md"
    command = [
        sys.executable,
        "scripts/wide_public_slice_selector.py",
        "--cards",
        ",".join(cards),
        "--seed",
        str(seed),
        "--cases-per-card",
        str(cases_per_card),
        "--out",
        rel(out),
        "--report-out",
        rel(report_out),
        "--markdown-out",
        rel(md),
    ]
    for path in consumed_case_manifest_paths(registry_rows):
        command.extend(["--exclude-manifest", path])
    for path in LEGACY_CONSUMED_MANIFESTS:
        if resolve(path).exists():
            command.extend(["--exclude-manifest", path])
    result = run_command(command, timeout_seconds=1200)
    result["report_out"] = rel(report_out)
    result["manifest_out"] = rel(out)
    result["markdown_out"] = rel(md)
    return result


def run_next_surface_planner(
    *,
    cases_per_card: int,
    seed_start: int,
    seed_end: int,
    cards: list[str],
    slug_prefix: str,
    out: str,
    markdown_out: str,
    materialize: bool,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/public_transfer_next_surface_planner.py",
        "--cards",
        ",".join(cards),
        "--cases-per-card",
        str(cases_per_card),
        "--seed-start",
        str(seed_start),
        "--seed-end",
        str(seed_end),
        "--slug-prefix",
        slug_prefix,
        "--out",
        out,
        "--markdown-out",
        markdown_out,
    ]
    if materialize:
        command.append("--materialize")
    return run_command(command, timeout_seconds=2400)


def run_operator(*, packet: str, timeout_seconds: int) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/operator_bounded_public_calibration.py",
        "--packet",
        packet,
        "--out",
        "reports/theseus_benchmark_public_operator_execute.json",
        "--markdown-out",
        "reports/theseus_benchmark_public_operator_execute.md",
        "--timeout-seconds",
        str(timeout_seconds),
        "--execute",
    ]
    result = run_command(command, timeout_seconds=timeout_seconds + 300)
    report = read_json(REPORTS / "theseus_benchmark_public_operator_execute.json", {})
    return report or result


def load_capacity_report() -> dict[str, Any]:
    for path in [
        REPORTS / "theseus_benchmark_public_capacity.json",
        REPORTS / "public_transfer_next_surface_planner_current.json",
    ]:
        data = read_json(path, {})
        if data.get("cards_report"):
            return data
        capacity = data.get("source_capacity")
        if isinstance(capacity, dict) and capacity.get("cards"):
            return {"cards_report": capacity.get("cards", [])}
    return {}


def source_capacity_summary(selector_report: dict[str, Any]) -> dict[str, Any]:
    rows = selector_report.get("cards_report") if isinstance(selector_report.get("cards_report"), list) else []
    cards = []
    insufficient = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        card = {
            "card_id": row.get("card_id"),
            "available_after_exclusions": first_int(row, "available_probe_task_count", "available_after_exclusions"),
            "available_before_exclusions": first_int(row, "available_probe_task_count_before_exclusions", "available_before_exclusions"),
            "excluded_probe_task_count": first_int(row, "excluded_probe_task_count"),
            "required_task_count": first_int(row, "required_task_count"),
            "selected_task_count": first_int(row, "selected_task_count"),
            "ready": bool(row.get("ready_for_wide_public_calibration", row.get("ready", False))),
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


def choose_measurement_plan(
    *,
    requested_cards: list[str],
    capacity: dict[str, Any],
    requested_cases: int,
    diagnostic_if_needed: bool,
) -> dict[str, Any]:
    rows = [row for row in capacity.get("cards", []) if isinstance(row, dict)]
    by_card = {str(row.get("card_id") or ""): row for row in rows}
    requested_rows = [by_card.get(card, {"card_id": card}) for card in requested_cards]
    balanced_max = min((int_or_zero(row.get("available_after_exclusions")) for row in requested_rows), default=0)
    requested_cases = max(1, int(requested_cases))
    if balanced_max >= requested_cases:
        return {
            "measurement_kind": "headline",
            "balanced_max_cases_per_card_after_exclusions": balanced_max,
            "effective_cases_per_card": requested_cases,
            "effective_cards": requested_cards,
            "omitted_cards": [],
        }
    if diagnostic_if_needed and balanced_max > 0:
        return {
            "measurement_kind": "diagnostic",
            "balanced_max_cases_per_card_after_exclusions": balanced_max,
            "effective_cases_per_card": balanced_max,
            "effective_cards": requested_cards,
            "omitted_cards": [],
        }

    full_case_cards = [
        str(row.get("card_id"))
        for row in requested_rows
        if str(row.get("card_id") or "") and int_or_zero(row.get("available_after_exclusions")) >= requested_cases
    ]
    omitted = [
        {
            "card_id": str(row.get("card_id") or ""),
            "available_after_exclusions": int_or_zero(row.get("available_after_exclusions")),
            "required_for_requested_slice": requested_cases,
            "reason": "insufficient_fresh_rows_after_consumed_surface_exclusions",
        }
        for row in requested_rows
        if str(row.get("card_id") or "") not in set(full_case_cards)
    ]
    if diagnostic_if_needed and full_case_cards:
        return {
            "measurement_kind": "partial_card_diagnostic",
            "balanced_max_cases_per_card_after_exclusions": balanced_max,
            "effective_cases_per_card": requested_cases,
            "effective_cards": full_case_cards,
            "omitted_cards": omitted,
        }

    return {
        "measurement_kind": "unavailable",
        "balanced_max_cases_per_card_after_exclusions": balanced_max,
        "effective_cases_per_card": requested_cases,
        "effective_cards": [],
        "omitted_cards": omitted,
    }


def next_actions(summary: dict[str, Any], capacity: dict[str, Any], proposal: dict[str, Any], execute: bool) -> list[str]:
    if summary.get("fresh_headline_surface_available"):
        if summary.get("materialized"):
            actions = [
                f"Fresh metadata-only public surface materialized: {proposal.get('slug')}.",
                "Run the guarded operator command only when ready to record this exact fresh measurement.",
            ]
            if proposal.get("packet"):
                actions.append(
                    "python3 scripts/operator_bounded_public_calibration.py "
                    f"--packet {proposal.get('packet')} --execute"
                )
            return actions
        return [
            "Fresh headline surface is available. Re-run with --materialize to freeze the packet without scoring.",
            "Use --execute only after the packet is green; public outputs must remain measurement-only.",
        ]
    if summary.get("partial_card_diagnostic_available"):
        cards = ",".join(str(card) for card in summary.get("effective_cards", []))
        actions = [
            "A full 5-card headline slice is not available after consumed-surface exclusions.",
            f"Largest fresh full-per-card subset is {summary.get('effective_cases_per_card')} cases/card on: {cards}.",
            "This is a clearly labeled partial-card diagnostic, not a headline apples-to-apples public-transfer claim.",
        ]
        if summary.get("materialized") and proposal.get("packet"):
            actions.append(
                "Partial diagnostic packet is frozen; guarded execution command: "
                f"python3 scripts/operator_bounded_public_calibration.py --packet {proposal.get('packet')} --execute"
            )
        else:
            actions.append("Re-run with --materialize to freeze this partial diagnostic packet without scoring.")
        return actions
    if summary.get("diagnostic_surface_available"):
        actions = [
            f"A full balanced headline slice is not available; largest fresh balanced diagnostic is {summary.get('effective_cases_per_card')} cases/card.",
            "Use diagnostic scores only as a labeled smoke/regression signal, not as a headline public-transfer claim.",
            "Stage more legitimate public calibration sources before another large apples-to-apples claim.",
        ]
        if summary.get("materialized") and proposal.get("packet"):
            actions.append(
                "Diagnostic packet is frozen; guarded execution command: "
                f"python3 scripts/operator_bounded_public_calibration.py --packet {proposal.get('packet')} --execute"
            )
        return actions
    if execute:
        return ["Execute was requested but no fresh public surface was available."]
    return [
        "No fresh public measurement surface is available from the currently staged local public benchmark rows.",
        "Improve private semantic candidate quality and stage additional legitimate benchmark sources for future measurement.",
    ]


def latest_consumed(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row.get("score_recorded") or row.get("real_public_task_pass_rate") is not None]
    if not scored:
        return rows[-1] if rows else {}
    return sorted(scored, key=lambda row: str(row.get("created_utc") or ""))[-1]


def previous_consumed(rows: list[dict[str, Any]], latest: dict[str, Any]) -> dict[str, Any]:
    if not latest:
        return {}
    scored = [row for row in rows if row is not latest and (row.get("score_recorded") or row.get("real_public_task_pass_rate") is not None)]
    return sorted(scored, key=lambda row: str(row.get("created_utc") or ""))[-1] if scored else {}


def public_run_summary(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    passed = int_or_zero(row.get("multi_stream_passed"))
    total = int_or_zero(row.get("public_task_count"))
    pass_rate = float_or_none(row.get("real_public_task_pass_rate"))
    if pass_rate is None and total:
        pass_rate = passed / total
    return {
        "run_id": row.get("run_id") or row.get("surface_slug"),
        "created_utc": row.get("created_utc"),
        "passed": passed,
        "task_count": total,
        "pass_rate": pass_rate,
        "score_report_path": row.get("score_report_path") or row.get("output_path"),
        "dominant_residual_categories": row.get("dominant_residual_categories") or [],
        "template_like_candidate_count": int_or_zero(row.get("template_like_candidate_count")),
        "fallback_return_count": int_or_zero(row.get("fallback_return_count")),
        "external_inference_calls": int_or_zero(row.get("external_inference_calls")),
    }


def executed_public_run_summary(operator_report: dict[str, Any], proposal: dict[str, Any]) -> dict[str, Any]:
    summary = as_dict(operator_report.get("summary"))
    if summary.get("executed") is not True:
        return {}
    score_path = str(summary.get("output_path") or "")
    score_report = read_json(resolve(score_path), {}) if score_path else {}
    score_summary = as_dict(score_report.get("summary"))
    task_count = int_or_zero(score_summary.get("public_task_count") or score_summary.get("total_case_count"))
    pass_rate = float_or_none(score_summary.get("real_public_task_pass_rate"))
    if pass_rate is None:
        pass_rate = float_or_none(score_summary.get("multi_stream_pass_rate"))
    passed = int(round((pass_rate or 0.0) * task_count)) if task_count else 0
    return {
        "run_id": summary.get("run_id") or proposal.get("slug") or score_report.get("frontier_family"),
        "created_utc": score_report.get("created_utc") or operator_report.get("created_utc"),
        "passed": passed,
        "task_count": task_count,
        "pass_rate": pass_rate,
        "score_report_path": score_path,
        "dominant_residual_categories": [],
        "template_like_candidate_count": int_or_zero(score_summary.get("template_like_candidate_count")),
        "fallback_return_count": int_or_zero(score_summary.get("fallback_return_count")),
        "external_inference_calls": int_or_zero(score_summary.get("external_inference_calls") or score_report.get("external_inference_calls")),
    }


def consumed_case_manifest_paths(rows: list[dict[str, Any]]) -> list[str]:
    paths = []
    seen = set()
    for row in rows:
        command = row.get("command") if isinstance(row.get("command"), list) else []
        manifest = command_arg([str(part) for part in command], "--case-manifest")
        if manifest and manifest not in seen:
            paths.append(manifest)
            seen.add(manifest)
    return paths


def duplicate_surface_risk(proposal: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    slug = str(proposal.get("slug") or "")
    if not slug:
        return False
    return any(str(row.get("run_id") or row.get("surface_slug") or "") == slug and row.get("consumed") is True for row in rows)


def command_arg(command: list[str], flag: str) -> str:
    try:
        index = command.index(flag)
    except ValueError:
        return ""
    if index + 1 >= len(command):
        return ""
    return str(command[index + 1])


def diagnostic_slug_prefix(prefix: str, measurement_kind: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(prefix or "").strip()).strip("_")
    clean = clean or "public_transfer_measurement"
    if measurement_kind == "partial_card_diagnostic" and not clean.endswith("_partial_diagnostic"):
        return f"{clean}_partial_diagnostic"
    if measurement_kind == "diagnostic" and not clean.endswith("_diagnostic"):
        return f"{clean}_diagnostic"
    return clean


def run_command(command: list[str], timeout_seconds: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout_seconds)
        return {
            "status": "completed" if result.returncode == 0 else "failed",
            "command": command,
            "returncode": result.returncode,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timed_out",
            "command": command,
            "returncode": None,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
        }


def max_public_training_rows(*reports: dict[str, Any]) -> int:
    values = []
    for report in reports:
        if not isinstance(report, dict):
            continue
        values.append(int_or_zero(report.get("public_training_rows_written")))
        values.append(int_or_zero(report.get("public_training_rows")))
        values.append(int_or_zero(as_dict(report.get("summary")).get("public_training_rows_written")))
        values.append(int_or_zero(as_dict(report.get("summary")).get("training_rows_written")))
    return max([0, *values])


def max_external_inference(*reports: dict[str, Any]) -> int:
    values = []
    for report in reports:
        if not isinstance(report, dict):
            continue
        values.append(int_or_zero(report.get("external_inference_calls")))
        values.append(int_or_zero(as_dict(report.get("summary")).get("external_inference_calls")))
    return max([0, *values])


def render_markdown(report: dict[str, Any]) -> str:
    summary = as_dict(report.get("summary"))
    latest = as_dict(summary.get("latest_public_run"))
    lines = [
        "# Theseus Benchmark Measurement",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- latest public run: `{latest.get('run_id')}` score `{latest.get('passed')}/{latest.get('task_count')}` pass_rate `{latest.get('pass_rate')}`",
        f"- measurement kind: `{summary.get('measurement_kind')}`",
        f"- requested cases/card: `{summary.get('requested_cases_per_card')}`",
        f"- effective cases/card: `{summary.get('effective_cases_per_card')}`",
        f"- effective cards: `{','.join(str(card) for card in summary.get('effective_cards', []))}`",
        f"- balanced fresh max cases/card: `{summary.get('balanced_max_cases_per_card_after_exclusions')}`",
        f"- materialized: `{summary.get('materialized')}` packet `{summary.get('packet')}`",
        f"- public training rows written: `{summary.get('public_training_rows_written')}`",
        f"- external inference calls: `{summary.get('external_inference_calls')}`",
        "",
        "## Next Actions",
    ]
    for action in report.get("next_actions", []):
        lines.append(f"- {action}")
    omitted = summary.get("omitted_cards")
    if isinstance(omitted, list) and omitted:
        lines.extend(["", "## Omitted Cards"])
        for row in omitted:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"- `{row.get('card_id')}`: available `{row.get('available_after_exclusions')}` / required `{row.get('required_for_requested_slice')}`"
            )
    lines.extend(["", "## Rules"])
    for key, value in as_dict(report.get("rules")).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines).rstrip() + "\n"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
    return value if isinstance(value, dict) else default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return candidate.as_posix()


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def first_int(row: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = row.get(key)
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
