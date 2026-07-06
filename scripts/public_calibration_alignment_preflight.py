#!/usr/bin/env python3
"""No-execute alignment preflight for the next public calibration contract.

This script proves that the proposed public calibration command is bound to a
metadata-only case manifest before any operator-approved execution. It does not
run public calibration, read public tests/solutions, generate candidates, write
training rows, or call external inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_CONTRACT = ROOT / "configs" / "public_benchmark_contract_v1.json"
DEFAULT_SELECTOR_REPORT = REPORTS / "public_wide_slice_selector_industry_code_transfer_seed14_5x64_v1.json"
DEFAULT_OPERATOR_DRY_RUN = REPORTS / "operator_bounded_public_calibration_dry_run.json"
DEFAULT_PACKET = REPORTS / "public_calibration_readiness_packet.json"
DEFAULT_OUT = REPORTS / "public_calibration_alignment_preflight.json"
DEFAULT_MD = REPORTS / "public_calibration_alignment_preflight.md"

CASE_MANIFEST_POLICY = "project_theseus_public_code_case_manifest_v1"
EXPECTED_CARDS = {
    "source_mbpp",
    "source_evalplus",
    "source_bigcodebench",
    "source_human_eval",
    "source_livecodebench",
}
FORBIDDEN_TRUE_FLAGS = {
    "prompts_exported",
    "tests_exported",
    "solutions_exported",
    "candidate_code_exported",
    "private_training_allowed",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default=rel(DEFAULT_CONTRACT))
    parser.add_argument("--selector-report", default=rel(DEFAULT_SELECTOR_REPORT))
    parser.add_argument("--operator-dry-run", default=rel(DEFAULT_OPERATOR_DRY_RUN))
    parser.add_argument("--packet", default=rel(DEFAULT_PACKET))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    args = parser.parse_args()

    report = build_report(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report["trigger_state"] == "RED" else 0


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    contract_path = resolve(args.contract)
    selector_path = resolve(args.selector_report)
    operator_dry_run_path = resolve(args.operator_dry_run)
    packet_path = resolve(args.packet)

    contract = read_json(contract_path, {})
    selector = read_json(selector_path, {})
    operator_dry_run = read_json(operator_dry_run_path, {})
    packet = read_json(packet_path, {})

    stage = object_field(contract, "stage_1_code_generation_surface")
    command = [str(part) for part in as_list(stage.get("command_after_unlock_only"))]
    command_text = " ".join(command)
    case_manifest_path = resolve(str(stage.get("case_manifest") or proposed_arg(command, "--case-manifest") or ""))
    candidate_manifest_path = proposed_arg_path(command, "--student-candidate-manifest")
    output_path = proposed_arg_path(command, "--out")
    trace_path = proposed_arg_path(command, "--trace-out")
    transfer_artifact_path = proposed_arg_path(command, "--transfer-artifact-out")

    manifest_rows = read_jsonl(case_manifest_path)
    manifest_scan = scan_case_manifest(manifest_rows)
    selector_counts = {
        str(row.get("card_id")): int_number(row.get("selected_task_count"))
        for row in as_list(selector.get("cards_report"))
        if isinstance(row, dict)
    }
    expected_cases_per_card = int_number(stage.get("cases_per_card"))
    expected_total = int_number(stage.get("total_task_count"))
    expected_cards = set(str(card) for card in as_list(stage.get("cards")))
    if not expected_cards:
        expected_cards = set(EXPECTED_CARDS)

    operator_summary = object_field(operator_dry_run, "summary")
    packet_summary = object_field(packet, "summary")
    candidate_preexists = bool(candidate_manifest_path and candidate_manifest_path.exists())
    output_absent = bool(output_path and not output_path.exists())
    trace_absent = bool(trace_path is None or not trace_path.exists())

    hard_gates = [
        gate("contract_loaded", contract.get("policy") == "project_theseus_public_benchmark_contract_v1", {
            "path": rel(contract_path),
            "policy": contract.get("policy"),
        }),
        gate("case_manifest_bound_in_contract", bool(stage.get("case_manifest")), {
            "contract_case_manifest": stage.get("case_manifest"),
            "command_case_manifest": proposed_arg(command, "--case-manifest"),
        }),
        gate("case_manifest_bound_in_command", proposed_arg(command, "--case-manifest") == rel(case_manifest_path), {
            "command_case_manifest": proposed_arg(command, "--case-manifest"),
            "expected_case_manifest": rel(case_manifest_path),
        }),
        gate("case_manifest_report_bound_in_contract", str(stage.get("case_manifest_report") or "") == rel(selector_path), {
            "contract_case_manifest_report": stage.get("case_manifest_report"),
            "expected_selector_report": rel(selector_path),
        }),
        gate("case_manifest_exists", case_manifest_path.exists(), rel(case_manifest_path)),
        gate("case_manifest_row_count_matches_contract", len(manifest_rows) == expected_total, {
            "row_count": len(manifest_rows),
            "expected_total": expected_total,
        }),
        gate("case_manifest_card_counts_match_contract", manifest_scan["card_counts"] == {
            card: expected_cases_per_card for card in sorted(expected_cards)
        }, {
            "card_counts": manifest_scan["card_counts"],
            "expected": {card: expected_cases_per_card for card in sorted(expected_cards)},
        }),
        gate("case_manifest_metadata_only", not manifest_scan["metadata_failures"], {
            "metadata_failure_count": len(manifest_scan["metadata_failures"]),
            "sample_failures": manifest_scan["metadata_failures"][:10],
        }),
        gate("case_manifest_no_duplicate_tasks", not manifest_scan["duplicate_task_keys"], {
            "duplicate_task_key_count": len(manifest_scan["duplicate_task_keys"]),
            "duplicates": manifest_scan["duplicate_task_keys"][:10],
        }),
        gate("selector_report_green", selector.get("trigger_state") == "GREEN" and selector.get("ready_for_wide_public_calibration") is True, {
            "path": rel(selector_path),
            "trigger_state": selector.get("trigger_state"),
            "ready_for_wide_public_calibration": selector.get("ready_for_wide_public_calibration"),
        }),
        gate("selector_report_counts_match_manifest", selector_counts == manifest_scan["card_counts"], {
            "selector_counts": selector_counts,
            "manifest_card_counts": manifest_scan["card_counts"],
        }),
        gate("command_expected_runner", command[:2] == ["python3", "scripts/real_code_benchmark_graduation.py"], {
            "command": command,
        }),
        gate("candidate_generation_deferred_to_approved_execute", candidate_generation_contract(command, candidate_manifest_path), {
            "student_candidate_generator": proposed_arg(command, "--student-candidate-generator"),
            "student_candidate_manifest": rel(candidate_manifest_path) if candidate_manifest_path else "",
            "candidate_manifest_preexists_before_run": candidate_preexists,
            "skip_student_candidate_generation": "--skip-student-candidate-generation" in command,
        }),
        gate("candidate_manifest_does_not_preexist", not candidate_preexists, {
            "student_candidate_manifest": rel(candidate_manifest_path) if candidate_manifest_path else "",
            "exists": candidate_preexists,
        }),
        gate("output_and_trace_absent_before_run", output_absent and trace_absent, {
            "output_path": rel(output_path) if output_path else "",
            "output_absent": output_absent,
            "trace_path": rel(trace_path) if trace_path else "",
            "trace_absent": trace_absent,
        }),
        gate("dry_run_or_packet_did_not_execute", dry_run_or_packet_did_not_execute(operator_dry_run, packet), {
            "operator_dry_run_mode": operator_summary.get("mode"),
            "operator_dry_run_executed": operator_summary.get("executed"),
            "packet_public_calibration_allowed": packet.get("public_calibration_allowed"),
        }),
    ]

    warning_gates = [
        gate("operator_dry_run_current_if_present", operator_dry_run_current(operator_dry_run, command), {
            "operator_dry_run_path": rel(operator_dry_run_path),
            "operator_command_text": object_field(operator_dry_run, "command").get("text"),
            "contract_command_text": command_text,
        }, severity="warning"),
        gate("packet_current_if_present", packet_current(packet, stage), {
            "packet_path": rel(packet_path),
            "packet_state": packet.get("trigger_state"),
            "packet_proposed_slug": packet_summary.get("proposed_slug"),
            "contract_slug": stage.get("slug"),
        }, severity="warning"),
    ]

    hard_failed = [row for row in hard_gates if not row["passed"]]
    warning_failed = [row for row in warning_gates if not row["passed"]]
    trigger_state = "RED" if hard_failed else ("YELLOW" if warning_failed else "GREEN")
    alignment_ready = not hard_failed

    report = {
        "policy": "project_theseus_public_calibration_alignment_preflight_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "alignment_preflight_ready": alignment_ready,
        "summary": {
            "contract": rel(contract_path),
            "contract_sha256": sha256_file(contract_path),
            "contract_slug": stage.get("slug"),
            "case_manifest": rel(case_manifest_path),
            "case_manifest_sha256": sha256_file(case_manifest_path),
            "case_manifest_report": rel(selector_path),
            "case_manifest_report_sha256": sha256_file(selector_path),
            "case_manifest_row_count": len(manifest_rows),
            "case_manifest_card_counts": manifest_scan["card_counts"],
            "case_manifest_bound_to_command": proposed_arg(command, "--case-manifest") == rel(case_manifest_path),
            "candidate_manifest_bound_to_case_manifest": candidate_generation_contract(command, candidate_manifest_path),
            "student_candidate_manifest": rel(candidate_manifest_path) if candidate_manifest_path else "",
            "candidate_manifest_preexists_before_run": candidate_preexists,
            "candidate_manifest_generation_deferred_to_execute": bool(candidate_manifest_path),
            "output_path": rel(output_path) if output_path else "",
            "trace_path": rel(trace_path) if trace_path else "",
            "transfer_artifact_path": rel(transfer_artifact_path) if transfer_artifact_path else "",
            "selector_report_ready": selector.get("ready_for_wide_public_calibration") is True,
            "selector_report_state": selector.get("trigger_state"),
            "selector_counts": selector_counts,
            "hard_failed_gate_count": len(hard_failed),
            "warning_failed_gate_count": len(warning_failed),
            "metadata_failure_count": len(manifest_scan["metadata_failures"]),
            "duplicate_task_key_count": len(manifest_scan["duplicate_task_keys"]),
            "public_tests_used": False,
            "public_solutions_used": False,
            "public_prompts_exported": False,
            "candidate_code_exported": False,
            "training_rows_written": 0,
            "external_inference_calls": 0,
        },
        "inputs": {
            "contract": rel(contract_path),
            "selector_report": rel(selector_path),
            "operator_dry_run": rel(operator_dry_run_path),
            "packet": rel(packet_path),
        },
        "command": {
            "text": command_text,
            "argv": command,
            "case_manifest": rel(case_manifest_path),
            "student_candidate_manifest": rel(candidate_manifest_path) if candidate_manifest_path else "",
            "candidate_generation": "deferred until one approved execute by real_code_benchmark_graduation.py",
        },
        "gates": hard_gates + warning_gates,
        "rules": {
            "no_public_calibration": True,
            "no_public_prompt_test_solution_read": True,
            "no_candidate_code_generation": True,
            "no_training_rows": True,
            "public_task_ids_are_selectors_only": True,
        },
        "public_tests_used": False,
        "public_solutions_used": False,
        "training_rows_written": 0,
        "external_inference_calls": 0,
    }
    return report


def scan_case_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    card_counts: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    duplicates: list[str] = []
    metadata_failures: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        card = str(row.get("card_id") or "")
        task_id = str(row.get("task_id") or row.get("source_task_id") or "")
        card_counts[card] += 1
        key = (card, task_id)
        if key in seen:
            duplicates.append(f"{card}:{task_id}")
        seen.add(key)
        failures: list[str] = []
        if row.get("policy") != CASE_MANIFEST_POLICY:
            failures.append("unexpected_policy")
        if row.get("public_calibration_only") is not True:
            failures.append("public_calibration_only_not_true")
        for flag in sorted(FORBIDDEN_TRUE_FLAGS):
            if row.get(flag) is True:
                failures.append(f"{flag}_true")
        if not task_id:
            failures.append("task_id_missing")
        if not card:
            failures.append("card_id_missing")
        if failures:
            metadata_failures.append({"line": idx, "task_id": task_id, "card_id": card, "failures": failures})
    return {
        "card_counts": dict(sorted(card_counts.items())),
        "duplicate_task_keys": duplicates,
        "metadata_failures": metadata_failures,
    }


def candidate_generation_contract(command: list[str], candidate_path: Path | None) -> bool:
    return bool(
        proposed_arg(command, "--student-candidate-generator") == "token"
        and proposed_arg(command, "--case-manifest")
        and proposed_arg(command, "--student-candidate-manifest")
        == "reports/student_code_candidates_industry_code_transfer_seed14_5x64_v1.jsonl"
        and candidate_path is not None
        and candidate_path.parent == REPORTS
        and "--skip-student-candidate-generation" not in command
    )


def dry_run_or_packet_did_not_execute(operator_dry_run: dict[str, Any], packet: dict[str, Any]) -> bool:
    operator_summary = object_field(operator_dry_run, "summary")
    return bool(
        operator_summary.get("executed") in {None, False}
        and object_field(operator_dry_run, "run_result").get("status") in {None, "dry_run_not_executed"}
        and packet.get("public_calibration_allowed") in {None, False}
    )


def operator_dry_run_current(operator_dry_run: dict[str, Any], command: list[str]) -> bool:
    if not operator_dry_run:
        return True
    observed = object_field(operator_dry_run, "command").get("argv")
    if not isinstance(observed, list):
        return True
    return [str(part) for part in observed] == command


def packet_current(packet: dict[str, Any], stage: dict[str, Any]) -> bool:
    if not packet:
        return True
    return (
        packet.get("policy") == "project_theseus_public_calibration_readiness_packet_v1"
        and object_field(packet, "summary").get("proposed_slug") == stage.get("slug")
    )


def gate(name: str, passed: bool, evidence: Any, *, severity: str = "hard") -> dict[str, Any]:
    return {
        "gate": name,
        "passed": bool(passed),
        "severity": severity,
        "evidence": evidence,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = object_field(report, "summary")
    lines = [
        "# Public Calibration Alignment Preflight",
        "",
        f"- Trigger state: `{report.get('trigger_state')}`",
        f"- Alignment ready: `{report.get('alignment_preflight_ready')}`",
        f"- Contract: `{summary.get('contract_slug')}`",
        f"- Case manifest: `{summary.get('case_manifest')}`",
        f"- Case manifest rows: `{summary.get('case_manifest_row_count')}`",
        f"- Card counts: `{summary.get('case_manifest_card_counts')}`",
        f"- Candidate manifest: `{summary.get('student_candidate_manifest')}`",
        f"- Candidate generation deferred: `{summary.get('candidate_manifest_generation_deferred_to_execute')}`",
        f"- Candidate manifest preexists: `{summary.get('candidate_manifest_preexists_before_run')}`",
        f"- Public tests used: `{summary.get('public_tests_used')}`",
        f"- Public solutions used: `{summary.get('public_solutions_used')}`",
        f"- Training rows written: `{summary.get('training_rows_written')}`",
        "",
        "## Gates",
    ]
    for row in as_list(report.get("gates")):
        lines.append(f"- `{row.get('gate')}`: `{row.get('passed')}` ({row.get('severity')})")
    lines.append("")
    return "\n".join(lines)


def read_json(path: Path, default: Any) -> Any:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return data if isinstance(data, dict) else default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    except Exception:
        return []
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def object_field(obj: dict[str, Any], key: str) -> dict[str, Any]:
    value = obj.get(key) if isinstance(obj, dict) else {}
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def int_number(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def proposed_arg_path(command: list[str], flag: str) -> Path | None:
    value = proposed_arg(command, flag)
    if not value:
        return None
    return resolve(value)


def proposed_arg(command: list[str], flag: str) -> str:
    try:
        idx = command.index(flag)
        return command[idx + 1]
    except (ValueError, IndexError):
        return ""


def sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


if __name__ == "__main__":
    raise SystemExit(main())
