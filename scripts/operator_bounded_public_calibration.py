#!/usr/bin/env python3
"""Guarded public calibration runner.

This script is the operator side of the post-v4 readiness packet. Dry-run is
the default and writes only audit/template artifacts. Executing public
calibration uses the governed run registry to prevent duplicate score-fishing;
legacy approval files are still accepted for old contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_PACKET = REPORTS / "public_calibration_readiness_packet.json"
DEFAULT_OPERATOR_LOCK = REPORTS / "public_calibration_operator_lock.flag"
DEFAULT_APPROVAL = REPORTS / "public_calibration_operator_approval_industry_code_transfer_seed14_5x64_v1.json"
DEFAULT_OUT = REPORTS / "operator_bounded_public_calibration_dry_run.json"
DEFAULT_MARKDOWN = REPORTS / "operator_bounded_public_calibration_dry_run.md"
DEFAULT_RUN_LOCK = REPORTS / "operator_bounded_public_calibration.lock"
DEFAULT_POLICY = ROOT / "configs" / "permissive_growth_policy.json"
DEFAULT_REGISTRY = REPORTS / "public_benchmark_run_registry.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", default=rel(DEFAULT_PACKET))
    parser.add_argument("--operator-lock", default=rel(DEFAULT_OPERATOR_LOCK))
    parser.add_argument("--approval-file", default=rel(DEFAULT_APPROVAL))
    parser.add_argument("--run-lock", default=rel(DEFAULT_RUN_LOCK))
    parser.add_argument("--policy", default=rel(DEFAULT_POLICY))
    parser.add_argument("--registry", default=rel(DEFAULT_REGISTRY))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--execute", action="store_true", help="Actually run the one-shot calibration.")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()

    report = build_or_run(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["trigger_state"] == "RED":
        return 2
    if args.execute and report["summary"].get("run_returncode") not in {0, None}:
        return 1
    return 0


def build_or_run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    packet_path = resolve(args.packet)
    approval_path = resolve(args.approval_file)
    lock_path = resolve(args.operator_lock)
    run_lock_path = resolve(args.run_lock)
    policy_path = resolve(args.policy)
    registry_path = resolve(args.registry)
    packet = read_json(packet_path, {})
    policy = read_json(policy_path, {})
    packet_hash = sha256_file(packet_path)
    proposed = as_dict(packet.get("proposed_operator_actions"))
    command_text = str(
        proposed.get("calibration_command_after_run_registry_check")
        or proposed.get("calibration_command_after_unlock_only")
        or ""
    ).strip()
    command = shlex.split(command_text) if command_text else []
    output_path = proposed_output_path(command)
    trace_path = proposed_arg_path(command, "--trace-out")
    candidate_path = proposed_arg_path(command, "--student-candidate-manifest")
    approval = read_json(approval_path, {})
    registry_status = public_registry_status(policy, registry_path, packet)

    packet_ready = packet_is_ready(packet)
    command_safe = command_is_expected(command, packet)
    output_absent = bool(output_path and not output_path.exists())
    trace_absent = bool(trace_path is None or not trace_path.exists())
    candidate_preexists = bool(candidate_path and candidate_path.exists())
    candidate_generation_contract = candidate_generation_contract_expected(command, candidate_path)
    lock_present_before = lock_path.exists()
    legacy_approval_status = approval_check(approval, packet_hash, packet)
    approval_status = (
        {
            "approved": registry_status["allowed"],
            "reasons": registry_status["reasons"],
            "authorization_mode": "run_registry",
            "legacy_approval_valid": legacy_approval_status["approved"],
            "packet_sha256": packet_hash,
            "run_id": registry_status.get("run_id"),
            "registry": rel_or_abs(registry_path),
        }
        if registry_status.get("run_registry_execution_enabled")
        else legacy_approval_status
    )
    can_execute = all(
        [
            packet_ready,
            command_safe,
            candidate_generation_contract,
            output_absent,
            trace_absent,
            approval_status["approved"],
            registry_status["surface_not_consumed"],
        ]
    )
    run_result: dict[str, Any] | None = None
    lock_restored = lock_path.exists()

    if args.execute:
        if not can_execute:
            run_result = {
                "status": "not_run",
                "reason": "execute_requested_but_required_gates_failed",
                "failed_requirements": failed_requirements(
                    packet_ready=packet_ready,
                    command_safe=command_safe,
                    candidate_generation_contract=candidate_generation_contract,
                    output_absent=output_absent,
                    trace_absent=trace_absent,
                    approval_status=approval_status,
                    public_benchmark_run_registry=registry_status["allowed"],
                    surface_not_consumed=registry_status["surface_not_consumed"],
                ),
            }
        else:
            run_result, lock_restored = run_once(
                command,
                lock_path=lock_path,
                run_lock_path=run_lock_path,
                timeout_seconds=max(60, int(args.timeout_seconds)),
            )
            if registry_status.get("run_registry_execution_enabled"):
                append_registry_row(
                    registry_path,
                    packet=packet,
                    packet_hash=packet_hash,
                    policy=policy,
                    command=command,
                    run_result=run_result,
                    output_path=output_path,
                    trace_path=trace_path,
                    candidate_path=candidate_path,
                )

    trigger_state = "GREEN" if packet_ready and command_safe and approval_status["approved"] else "YELLOW"
    if args.execute and (not can_execute or not lock_restored):
        trigger_state = "RED"
    if output_path and output_path.exists() and not args.execute:
        trigger_state = "YELLOW"

    summary = {
        "mode": "execute" if args.execute else "dry_run",
        "packet_ready": packet_ready,
        "packet_hash": packet_hash,
        "frozen_integrity_sha256": get_path(packet, ["frozen_integrity", "sha256"]),
        "approval_file": rel_or_abs(approval_path),
        "approval_valid": approval_status["approved"],
        "authorization_mode": approval_status.get("authorization_mode", "legacy_approval_file"),
        "run_registry_execution_enabled": registry_status["run_registry_execution_enabled"],
        "run_registry_allowed": registry_status["allowed"],
        "run_registry_reasons": registry_status["reasons"],
        "run_registry": rel_or_abs(registry_path),
        "run_id": registry_status.get("run_id"),
        "surface_not_consumed": registry_status["surface_not_consumed"],
        "per_surface_existing_run_count": registry_status["per_surface_existing_run_count"],
        "consumed_run_count_total_for_audit": registry_status["consumed_run_count_total_for_audit"],
        "time_period_run_cap_enabled": False,
        "calendar_throttle_enabled": False,
        "fresh_surfaces_calendar_throttled": False,
        "fresh_surface_execution_policy": "run_immediately_when_frozen_registry_surface_is_clean",
        "operator_lock_present_before": lock_present_before,
        "operator_lock_present_after": lock_path.exists(),
        "command_safe": command_safe,
        "candidate_generation_contract_ready": candidate_generation_contract,
        "candidate_manifest_preexists_before_run": candidate_preexists,
        "candidate_manifest_generation_deferred_to_execute": bool(not args.execute and candidate_path),
        "trace_absent_before_run": trace_absent,
        "ready_for_registry_execute": bool(
            packet_ready
            and command_safe
            and candidate_generation_contract
            and output_absent
            and trace_absent
            and approval_status["approved"]
        ),
        "ready_for_operator_approval": bool(
            packet_ready
            and command_safe
            and candidate_generation_contract
            and output_absent
            and trace_absent
            and approval_status["approved"]
        ),
        "would_execute": bool(not args.execute and can_execute),
        "executed": bool(args.execute and run_result and run_result.get("status") == "completed"),
        "run_returncode": (run_result or {}).get("returncode"),
        "proposed_slug": get_path(packet, ["summary", "proposed_slug"]),
        "output_absent_before_run": output_absent,
        "output_path": rel_or_abs(output_path) if output_path else "",
        "output_exists_after": output_path.exists() if output_path else False,
        "trace_path": rel_or_abs(trace_path) if trace_path else "",
        "trace_exists_after": trace_path.exists() if trace_path else False,
        "student_candidate_manifest": rel_or_abs(candidate_path) if candidate_path else "",
        "student_candidate_manifest_exists_after": candidate_path.exists() if candidate_path else False,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }

    return {
        "policy": "project_theseus_operator_bounded_public_calibration_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": summary,
        "inputs": {
            "packet": rel_or_abs(packet_path),
            "operator_lock": rel_or_abs(lock_path),
            "approval_file": rel_or_abs(approval_path),
            "run_lock": rel_or_abs(run_lock_path),
            "policy": rel_or_abs(policy_path),
            "registry": rel_or_abs(registry_path),
        },
        "approval_template": approval_template(packet, packet_hash),
        "gates": [
            gate("packet_ready", packet_ready, packet_evidence(packet)),
            gate("packet_frozen_integrity_ready", packet_frozen_integrity_ready(packet), {
                "frozen_integrity_sha256": get_path(packet, ["frozen_integrity", "sha256"]),
                "harness_hashes_current": get_path(packet, ["frozen_integrity", "harness_hashes_current"]),
                "evidence_artifacts_hashable": get_path(packet, ["frozen_integrity", "evidence_artifacts_hashable"]),
                "harness_mismatch_count": get_path(packet, ["frozen_integrity", "harness_mismatch_count"]),
                "evidence_missing_count": get_path(packet, ["frozen_integrity", "evidence_missing_count"]),
            }),
            gate("approval_file_valid_for_execute", approval_status["approved"] if args.execute else True, {
                **approval_status,
                "required_now": bool(args.execute),
            }),
            gate("public_benchmark_run_registry_allows_run", registry_status["allowed"], registry_status),
            gate("surface_not_consumed", registry_status["surface_not_consumed"], registry_status),
            gate("legacy_operator_lock_not_required_by_registry", lock_present_before or registry_status["run_registry_execution_enabled"], {
                "present": lock_present_before,
                "authorization_mode": approval_status.get("authorization_mode", "legacy_approval_file"),
                "path": rel_or_abs(lock_path),
            }),
            gate("calibration_command_expected_shape", command_safe, {"command": command}),
            gate("candidate_generation_contract_expected", candidate_generation_contract, {
                "student_candidate_generator": proposed_arg(command, "--student-candidate-generator"),
                "student_candidate_manifest": rel_or_abs(candidate_path) if candidate_path else "",
                "preexisting_candidate_manifest_will_be_replaced_by_runner": candidate_preexists,
                "skip_student_candidate_generation": "--skip-student-candidate-generation" in command,
            }),
            gate("output_artifact_absent_before_run", output_absent, rel_or_abs(output_path) if output_path else ""),
            gate("trace_artifact_absent_before_run", trace_absent, rel_or_abs(trace_path) if trace_path else ""),
            gate("dry_run_does_not_execute_public_calibration", (not args.execute) or bool((run_result or {}).get("status") in {"completed", "failed", "timed_out"}), {
                "execute": bool(args.execute),
                "run_status": (run_result or {}).get("status", "dry_run_not_executed"),
            }),
            gate("legacy_operator_lock_state_after_run", lock_path.exists() or registry_status["run_registry_execution_enabled"], {
                "present": lock_path.exists(),
                "authorization_mode": approval_status.get("authorization_mode", "legacy_approval_file"),
                "path": rel_or_abs(lock_path),
            }),
        ],
        "command": {
            "text": command_text,
            "argv": command,
            "output_path": rel_or_abs(output_path) if output_path else "",
            "trace_path": rel_or_abs(trace_path) if trace_path else "",
            "student_candidate_manifest": rel_or_abs(candidate_path) if candidate_path else "",
            "candidate_generation": "generated inside approved execute by real_code_benchmark_graduation.py; dry run does not materialize public-prompt candidates",
        },
        "run_result": run_result or {"status": "dry_run_not_executed"},
        "next_actions": next_actions(args.execute, can_execute, approval_status, output_path),
        "rules": {
            "dry_run_default": "without --execute this script never launches public calibration",
            "approval": "execute uses the governed run registry when enabled; legacy approval JSON is still supported",
            "frozen_integrity": "execute approval must also include the packet's frozen_integrity_sha256 and one_shot_run_id",
            "one_shot": "execute is blocked if the proposed output or trace artifact already exists, or the run id is already consumed in the registry",
            "calendar_throttle": "disabled; new frozen registry surfaces are not delayed by monthly or time-window quotas",
            "candidate_manifest": "student candidates are generated inside the approved run; stale generated artifacts are removed by the benchmark runner before generation",
            "relock": "a pre-existing legacy operator lock is restored after execute; registry-authorized runs do not create a new lock file",
            "public_boundary": "do not train on public prompts, tests, solutions, traces, or score labels from the run",
        },
        "external_inference_calls": 0,
    }


def run_once(
    command: list[str],
    *,
    lock_path: Path,
    run_lock_path: Path,
    timeout_seconds: int,
) -> tuple[dict[str, Any], bool]:
    run_lock_fd = acquire_run_lock(run_lock_path)
    original_lock_exists = lock_path.exists()
    original_lock_text = read_text(lock_path)
    started = time.perf_counter()
    try:
        try:
            lock_path.unlink()
        except OSError:
            pass
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        status = "completed" if result.returncode == 0 else "failed"
        return (
            {
                "status": status,
                "returncode": result.returncode,
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
                "stdout_tail": result.stdout[-4000:],
                "stderr_tail": result.stderr[-4000:],
            },
            restore_lock(lock_path, original_lock_text, original_lock_exists),
        )
    except subprocess.TimeoutExpired as exc:
        return (
            {
                "status": "timed_out",
                "returncode": None,
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
                "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
                "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            },
            restore_lock(lock_path, original_lock_text, original_lock_exists),
        )
    finally:
        restore_lock(lock_path, original_lock_text, original_lock_exists)
        release_run_lock(run_lock_fd, run_lock_path)


def public_registry_status(policy: dict[str, Any], registry_path: Path, packet: dict[str, Any]) -> dict[str, Any]:
    public_policy = as_dict(policy.get("public_benchmarks"))
    execution_default = str(public_policy.get("execution_default") or "")
    enabled = execution_default in {
        "governed_measurement_run_registry",
        "governed_run_registry",
    }
    run_id = str(get_path(packet, ["summary", "proposed_slug"], "") or "")
    registry_rows = read_jsonl(registry_path)
    matching = [
        row
        for row in registry_rows
        if isinstance(row, dict)
        and str(row.get("run_id") or row.get("surface_slug") or "") == run_id
        and row.get("consumed") is True
    ]
    per_surface_max = int(public_policy.get("per_surface_max_runs") or 1)
    surface_not_consumed = bool(run_id and len(matching) < per_surface_max)
    reasons: list[str] = []
    if not enabled:
        reasons.append("governed_run_registry_policy_not_enabled")
    if not run_id:
        reasons.append("missing_frozen_run_id")
    if not surface_not_consumed:
        reasons.append("surface_consumed_or_per_surface_limit_reached")
    return {
        "run_registry_execution_enabled": enabled,
        "authorization_mode": "run_registry" if enabled else "legacy_approval_file",
        "allowed": enabled and bool(run_id) and surface_not_consumed,
        "reasons": reasons,
        "run_id": run_id,
        "registry": rel_or_abs(registry_path),
        "surface_not_consumed": surface_not_consumed,
        "per_surface_max_runs": per_surface_max,
        "per_surface_existing_run_count": len(matching),
        "time_period_run_cap_enabled": False,
        "calendar_throttle_enabled": False,
        "consumed_run_count_total_for_audit": len([row for row in registry_rows if isinstance(row, dict) and row.get("consumed") is True]),
        "fresh_surfaces_calendar_throttled": False,
        "fresh_surface_execution_policy": "run_immediately_when_frozen_registry_surface_is_clean",
    }


def append_registry_row(
    registry_path: Path,
    *,
    packet: dict[str, Any],
    packet_hash: str,
    policy: dict[str, Any],
    command: list[str],
    run_result: dict[str, Any] | None,
    output_path: Path | None,
    trace_path: Path | None,
    candidate_path: Path | None,
) -> None:
    row = {
        "policy": "project_theseus_public_benchmark_run_registry_v1",
        "created_utc": now(),
        "run_id": get_path(packet, ["summary", "proposed_slug"], ""),
        "surface_slug": get_path(packet, ["summary", "proposed_slug"], ""),
        "packet_sha256": packet_hash,
        "frozen_integrity_sha256": get_path(packet, ["frozen_integrity", "sha256"]),
        "command_sha256": hashlib.sha256(" ".join(command).encode("utf-8")).hexdigest(),
        "command": command,
        "status": (run_result or {}).get("status"),
        "returncode": (run_result or {}).get("returncode"),
        "consumed": True,
        "output_path": rel_or_abs(output_path),
        "trace_path": rel_or_abs(trace_path),
        "student_candidate_manifest": rel_or_abs(candidate_path),
        "score_recorded": False,
        "residual_categories_recorded": False,
        "policy_mode": policy.get("mode"),
        "no_training_on_public_eval_payloads": True,
        "external_inference_calls": 0,
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def packet_is_ready(packet: dict[str, Any]) -> bool:
    stage = calibration_stage(packet)
    summary = as_dict(packet.get("summary"))
    proposed_slug = str(summary.get("proposed_slug") or stage.get("slug") or "").strip()
    cards = stage.get("cards") if isinstance(stage.get("cards"), list) else []
    cases_per_card = int_or_zero(stage.get("cases_per_card"))
    total_task_count = int_or_zero(stage.get("total_task_count"))
    expected_total = len(cards) * cases_per_card if cards and cases_per_card else total_task_count
    registry_enabled = bool(
        get_path(packet, ["summary", "run_registry_execution_enabled"]) is True
        or get_path(packet, ["public_benchmark_contract", "global_rules", "run_registry_execution_enabled"]) is True
    )
    legacy_operator_review = bool(
        packet.get("public_calibration_allowed") is False
        and packet.get("operator_lock_active") is True
    )
    registry_authorized = bool(
        registry_enabled
        and packet.get("public_calibration_allowed") in {True, False}
    )
    return bool(
        packet.get("policy") == "project_theseus_public_calibration_readiness_packet_v1"
        and packet.get("mode") == "post_distillation_v4_operator_review"
        and packet.get("trigger_state") == "GREEN"
        and packet.get("technical_ready_for_one_bounded_public_calibration") is True
        and (legacy_operator_review or registry_authorized)
        and proposed_slug
        and proposed_slug == str(stage.get("slug") or "").strip()
        and total_task_count > 0
        and expected_total == total_task_count
        and int_or_zero(summary.get("proposed_public_surface_task_count")) == total_task_count
        and int_or_zero(summary.get("proposed_public_surface_seed")) == int_or_zero(stage.get("seed"))
        and int_or_zero(summary.get("proposed_public_surface_cases_per_card")) == cases_per_card
        and set(summary.get("proposed_public_surface_cards") or cards) == set(cards)
        and str(stage.get("status") or "") in {"contracted_not_executed", "frozen_unconsumed_contract"}
        and str(stage.get("case_manifest") or "").startswith("reports/")
        and str(stage.get("case_manifest_report") or "").startswith("reports/")
        and packet_frozen_integrity_ready(packet)
    )


def packet_frozen_integrity_ready(packet: dict[str, Any]) -> bool:
    return bool(
        get_path(packet, ["frozen_integrity", "sha256"])
        and get_path(packet, ["frozen_integrity", "harness_hashes_current"]) is True
        and get_path(packet, ["frozen_integrity", "evidence_artifacts_hashable"]) is True
        and int(get_path(packet, ["frozen_integrity", "harness_mismatch_count"], 1) or 0) == 0
        and int(get_path(packet, ["frozen_integrity", "evidence_missing_count"], 1) or 0) == 0
    )


def packet_evidence(packet: dict[str, Any]) -> dict[str, Any]:
    summary = dict(packet.get("summary")) if isinstance(packet.get("summary"), dict) else {}
    for key in list(summary):
        if key.startswith("calendar_or_monthly_"):
            summary.pop(key, None)
    summary.setdefault("calendar_throttle_enabled", False)
    return {
        "mode": packet.get("mode"),
        "trigger_state": packet.get("trigger_state"),
        "technical_ready": packet.get("technical_ready_for_one_bounded_public_calibration"),
        "public_calibration_allowed": packet.get("public_calibration_allowed"),
        "operator_lock_active": packet.get("operator_lock_active"),
        "run_registry_execution_enabled": get_path(packet, ["summary", "run_registry_execution_enabled"]),
        "frozen_integrity_sha256": get_path(packet, ["frozen_integrity", "sha256"]),
        "frozen_integrity_ready": packet_frozen_integrity_ready(packet),
        "summary": summary,
    }


def command_is_expected(command: list[str], packet: dict[str, Any]) -> bool:
    if len(command) < 3:
        return False
    script = command[1] if command[0].endswith("python3") or command[0].endswith("python") else command[0]
    if script != "scripts/real_code_benchmark_graduation.py":
        return False
    contract_command = get_path(
        packet,
        ["public_benchmark_contract", "stage_1_code_generation_surface", "command_after_run_registry_check"],
        [],
    )
    if not contract_command:
        contract_command = get_path(
            packet,
            ["public_benchmark_contract", "stage_1_code_generation_surface", "command_after_unlock_only"],
            [],
        )
    if isinstance(contract_command, list) and contract_command and command != [str(part) for part in contract_command]:
        return False
    stage = calibration_stage(packet)
    slug = str(stage.get("slug") or "").strip()
    cards = stage.get("cards") if isinstance(stage.get("cards"), list) else []
    required = {
        "--cards": ",".join(str(card) for card in cards),
        "--seed": str(int_or_zero(stage.get("seed"))),
        "--max-cases-per-card": str(int_or_zero(stage.get("cases_per_card"))),
        "--case-manifest": str(stage.get("case_manifest") or ""),
        "--student-candidate-generator": "token",
        "--student-candidate-manifest": f"reports/student_code_candidates_{slug}.jsonl",
        "--out": f"reports/real_code_benchmark_graduation_{slug}.json",
        "--trace-out": f"reports/real_code_benchmark_traces_{slug}.jsonl",
        "--transfer-artifact-out": f"reports/transfer_artifacts/code/real_code_benchmark_graduation_{slug}_transfer_artifact.json",
    }
    for flag, expected in required.items():
        observed = proposed_arg(command, flag)
        if observed != expected:
            return False
    return "--skip-student-candidate-generation" not in command


def candidate_generation_contract_expected(command: list[str], candidate_path: Path | None) -> bool:
    case_manifest = proposed_arg(command, "--case-manifest")
    student_manifest = proposed_arg(command, "--student-candidate-manifest")
    return bool(
        proposed_arg(command, "--student-candidate-generator") == "token"
        and case_manifest.startswith("reports/public_wide_slice_manifest_")
        and case_manifest.endswith(".jsonl")
        and student_manifest.startswith("reports/student_code_candidates_")
        and student_manifest.endswith(".jsonl")
        and candidate_path is not None
        and candidate_path.parent == REPORTS
        and "--skip-student-candidate-generation" not in command
    )


def approval_check(approval: dict[str, Any], packet_hash: str, packet: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if approval.get("policy") != "project_theseus_public_calibration_operator_approval_v1":
        reasons.append("approval_policy_missing")
    if approval.get("approved") is not True:
        reasons.append("approved_not_true")
    if approval.get("packet_sha256") != packet_hash:
        reasons.append("packet_sha256_mismatch")
    if approval.get("proposed_slug") != get_path(packet, ["summary", "proposed_slug"]):
        reasons.append("proposed_slug_mismatch")
    if approval.get("one_shot_run_id") != get_path(packet, ["summary", "proposed_slug"]):
        reasons.append("one_shot_run_id_mismatch")
    if approval.get("frozen_integrity_sha256") != get_path(packet, ["frozen_integrity", "sha256"]):
        reasons.append("frozen_integrity_sha256_mismatch")
    if int(approval.get("max_runs") or 0) != 1:
        reasons.append("max_runs_not_one")
    return {
        "approved": not reasons,
        "reasons": reasons,
        "packet_sha256": packet_hash,
        "approval_policy": approval.get("policy"),
        "approval_proposed_slug": approval.get("proposed_slug"),
        "approval_one_shot_run_id": approval.get("one_shot_run_id"),
        "expected_proposed_slug": get_path(packet, ["summary", "proposed_slug"]),
        "approval_frozen_integrity_sha256": approval.get("frozen_integrity_sha256"),
        "expected_frozen_integrity_sha256": get_path(packet, ["frozen_integrity", "sha256"]),
    }


def approval_template(packet: dict[str, Any], packet_hash: str) -> dict[str, Any]:
    return {
        "policy": "project_theseus_public_calibration_operator_approval_v1",
        "approved": False,
        "packet_sha256": packet_hash,
        "frozen_integrity_sha256": get_path(packet, ["frozen_integrity", "sha256"]),
        "proposed_slug": get_path(packet, ["summary", "proposed_slug"]),
        "one_shot_run_id": get_path(packet, ["summary", "proposed_slug"]),
        "max_runs": 1,
        "operator_note": "Set approved to true only when intentionally recording exactly one bounded public calibration measurement.",
    }


def failed_requirements(**values: Any) -> list[str]:
    failed: list[str] = []
    for key, value in values.items():
        if isinstance(value, dict):
            if not value.get("approved"):
                failed.append(key)
        elif not value:
            failed.append(key)
    return failed


def next_actions(
    execute: bool,
    can_execute: bool,
    approval_status: dict[str, Any],
    output_path: Path | None,
) -> list[str]:
    if execute and can_execute:
        return [
            "Inspect the run result and immediately refresh broad_transfer_matrix plus the public residual report.",
            "Keep the operator lock restored and do not train on public traces, tests, solutions, prompts, or score labels.",
        ]
    if execute:
        return [
            "Execution was blocked by guardrails; inspect failed_requirements before trying again.",
            "Do not rerun a consumed surface; fix the failed registry, integrity, output, or contamination gate first.",
        ]
    actions = [
        "Dry run only: no public calibration was launched.",
        "When the run registry, frozen packet, output, trace, and contamination gates are clean, rerun with --execute.",
    ]
    if not approval_status.get("approved"):
        reason_text = ", ".join(approval_status.get("reasons") or [])
        if approval_status.get("authorization_mode") == "run_registry":
            actions.append(f"Run registry is not executable for this exact surface: {reason_text}.")
        else:
            actions.append(f"Legacy approval file is missing or invalid: {reason_text}.")
    if output_path and output_path.exists():
        actions.append("The proposed output artifact already exists; do not rerun this one-shot surface.")
    return actions


def proposed_output_path(command: list[str]) -> Path | None:
    return proposed_arg_path(command, "--out")


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


def acquire_run_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy": "project_theseus_operator_bounded_public_calibration_run_lock_v1",
        "created_utc": now(),
        "pid": os.getpid(),
        "argv": sys.argv,
    }
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"run lock already exists: {rel_or_abs(path)}") from exc
    os.write(fd, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))
    return fd


def release_run_lock(fd: int, path: Path) -> None:
    try:
        os.close(fd)
    except OSError:
        pass
    try:
        path.unlink()
    except OSError:
        pass


def restore_lock(path: Path, original_text: str, original_lock_exists: bool = True) -> bool:
    if not original_lock_exists:
        try:
            path.unlink()
        except OSError:
            pass
        return not path.exists()
    text = original_text.strip() or (
        "Public calibration relocked by operator_bounded_public_calibration.py "
        f"at {now()} after a bounded governed run."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path.exists()


def render_markdown(report: dict[str, Any]) -> str:
    summary = as_dict(report.get("summary"))
    lines = [
        "# Operator Bounded Public Calibration",
        "",
        f"- State: `{report.get('trigger_state')}`",
        f"- Mode: `{summary.get('mode')}`",
        f"- Packet ready: `{summary.get('packet_ready')}`",
        f"- Authorization mode: `{summary.get('authorization_mode')}`",
        f"- Authorization valid: `{summary.get('approval_valid')}`",
        f"- Run registry allowed: `{summary.get('run_registry_allowed')}`",
        f"- Surface not consumed: `{summary.get('surface_not_consumed')}`",
        f"- Run registry: `{summary.get('run_registry')}`",
        f"- Frozen integrity SHA-256: `{summary.get('frozen_integrity_sha256')}`",
        f"- Operator lock present after: `{summary.get('operator_lock_present_after')}`",
        f"- Candidate generation contract ready: `{summary.get('candidate_generation_contract_ready')}`",
        f"- Candidate manifest preexists before run: `{summary.get('candidate_manifest_preexists_before_run')}`",
        f"- Candidate generation deferred to execute: `{summary.get('candidate_manifest_generation_deferred_to_execute')}`",
        f"- Trace absent before run: `{summary.get('trace_absent_before_run')}`",
        f"- Would execute: `{summary.get('would_execute')}`",
        f"- Executed: `{summary.get('executed')}`",
        f"- Output exists after: `{summary.get('output_exists_after')}`",
        "",
        "## Gates",
    ]
    for row in report.get("gates", []):
        lines.append(f"- `{row.get('name')}`: `{row.get('passed')}`")
    lines.extend(["", "## Next Actions"])
    for action in report.get("next_actions", []):
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def gate(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def read_json(path: Path, default: Any) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else default
    except Exception:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def calibration_stage(packet: dict[str, Any]) -> dict[str, Any]:
    return as_dict(get_path(packet, ["public_benchmark_contract", "stage_1_code_generation_surface"], {}))


def int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def get_path(obj: Any, path: list[str], default: Any = None) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def rel_or_abs(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
