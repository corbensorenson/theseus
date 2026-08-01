#!/usr/bin/env python3
"""Independent route-blind evaluator for sealed P2A typed-edit candidates."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import theseus_assistant_p2a as p2a


ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_p2a_route_blind_evaluation_v1"


class EvaluationFault(ValueError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--candidate-report", default="")
    parser.add_argument("--out", default="reports/theseus_assistant_p2a_evaluation.json")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    evaluator = p2a.resolve(args.evaluator)
    if args.audit_only:
        report = audit_evaluator(evaluator)
    elif args.candidate_report:
        report = evaluate_report(p2a.resolve(args.candidate_report), evaluator)
    else:
        parser.error("--candidate-report is required unless --audit-only is used")
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps(compact_summary(report), indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") == "GREEN" else 2


def audit_evaluator(path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    value = p2a.read_json(path)
    faults: list[str] = []
    if value.get("policy") not in {
        "project_theseus_p2a_route_blind_evaluator_v1",
        "project_theseus_p2b_route_blind_evaluator_v1",
        "project_theseus_p3_route_blind_evaluator_v1",
    }:
        faults.append("evaluator_policy_invalid")
    if value.get("state") != "SEALED_BEFORE_CANDIDATE_GENERATION":
        faults.append("evaluator_not_sealed")
    task_path = p2a.resolve(str(value.get("task_manifest") or ""))
    if p2a.sha256_file(task_path) != str(value.get("task_manifest_sha256") or ""):
        faults.append("task_manifest_digest_mismatch")
    task_audit = p2a.audit_task(task_path)
    if task_audit.get("trigger_state") != "GREEN":
        faults.append("task_manifest_audit_red")
    for index, row in enumerate(p2a.dicts(value.get("hidden_test_files"))):
        source = p2a.resolve(str(row.get("source") or ""))
        destination = str(row.get("destination") or "")
        if p2a.sha256_file(source) != str(row.get("sha256") or ""):
            faults.append(f"hidden_test_{index}_digest_mismatch")
        if p2a.unsafe_relative_path(destination):
            faults.append(f"hidden_test_{index}_destination_unsafe")
    target_archive = p2a.resolve(str(value.get("target_archive") or ""))
    task = p2a.read_json(task_path)
    if task.get("policy") in {
        "project_theseus_p2b_licensed_task_v1",
        "project_theseus_p3_licensed_task_v1",
    } and not str(value.get("target_archive_root") or ""):
        faults.append("target_archive_root_missing")
    if value.get("target_must_pass") is True and (
        p2a.sha256_file(target_archive) != str(value.get("target_archive_sha256") or "")
    ):
        faults.append("target_archive_digest_mismatch")
    command = p2a.strings(p2a.mapping(value.get("hidden_verifier")).get("command"))
    if not command or command[0] not in {"python3", "pytest"}:
        faults.append("hidden_verifier_command_invalid")
    baseline: dict[str, Any] = {}
    target: dict[str, Any] = {}
    if not faults:
        task = p2a.read_json(task_path)
        try:
            with tempfile.TemporaryDirectory(prefix="theseus-p2a-evaluator-audit-") as tmp:
                root = Path(tmp) / "source"
                p2a.extract_source_archive(
                    p2a.resolve(str(task.get("source_archive") or "")),
                    root,
                    str(task.get("source_archive_root") or ""),
                )
                overlay_hidden_tests(value, root)
                baseline = run_hidden_verifier(value, root)
            if value.get("target_must_pass") is True:
                with tempfile.TemporaryDirectory(prefix="theseus-p2a-evaluator-target-") as tmp:
                    root = Path(tmp) / "source"
                    p2a.extract_source_archive(
                        target_archive, root, str(value.get("target_archive_root") or "")
                    )
                    overlay_hidden_tests(value, root)
                    target = run_hidden_verifier(value, root)
        except (OSError, EvaluationFault, p2a.InstrumentFault) as exc:
            faults.append(f"baseline_audit_fault:{type(exc).__name__}")
    if value.get("baseline_must_fail") is not True or baseline.get("passed") is not False:
        faults.append("baseline_does_not_fail_as_required")
    markers = p2a.strings(value.get("baseline_failure_markers"))
    observed_text = str(baseline.get("stdout_tail") or "") + str(baseline.get("stderr_tail") or "")
    if any(marker not in observed_text for marker in markers):
        faults.append("baseline_failure_markers_missing")
    if value.get("target_must_pass") is not True or target.get("passed") is not True:
        faults.append("target_does_not_pass_as_required")
    return {
        "policy": "project_theseus_p2a_evaluator_audit_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "evaluator_sha256": p2a.sha256_file(path),
        "task_manifest_sha256": p2a.sha256_file(task_path),
        "task_audit": task_audit,
        "baseline_verification": baseline,
        "target_verification": target,
        "route_labels_opened": 0,
        "target_artifacts_opened_by_evaluator_only": 1 if target else 0,
        "counters": p2a.zero_counters(),
        "runtime_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def evaluate_report(candidate_path: Path, evaluator_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    alignment = audit_evaluator(evaluator_path)
    evaluator = p2a.read_json(evaluator_path)
    task_path = p2a.resolve(str(evaluator.get("task_manifest") or ""))
    task = p2a.read_json(task_path)
    candidate = p2a.read_json(candidate_path)
    faults: list[str] = []
    if alignment.get("trigger_state") != "GREEN":
        faults.append("evaluator_alignment_red")
    if candidate.get("task_sha256") != p2a.sha256_file(task_path):
        faults.append("candidate_task_binding_mismatch")
    if p2a.mapping(candidate.get("matched_pair")).get("ready") is not True:
        faults.append("candidate_matched_pair_invalid")
    blinded: list[dict[str, Any]] = []
    labels: dict[str, str] = {}
    for row in p2a.dicts(candidate.get("attempts")):
        seal = p2a.mapping(row.get("candidate_seal"))
        seal_hash = str(seal.get("candidate_output_sha256") or "")
        if not seal_hash:
            continue
        labels[seal_hash] = str(row.get("arm_id") or "")
        blinded.append({"candidate": p2a.mapping(row.get("candidate")), "seal": seal})
    blinded.sort(key=lambda row: str(p2a.mapping(row.get("seal")).get("candidate_output_sha256") or ""))
    results: list[dict[str, Any]] = []
    for row in blinded:
        try:
            result = evaluate_candidate_blind(task, evaluator, row)
        except (OSError, EvaluationFault, p2a.InstrumentFault) as exc:
            result = failed_candidate(type(exc).__name__)
        seal_hash = str(p2a.mapping(row.get("seal")).get("candidate_output_sha256") or "")
        result["arm_id"] = labels.get(seal_hash, "")
        results.append(result)
    evaluated = sum(int(row.get("correctness_evaluated") or 0) for row in results)
    useful = sum(int(row.get("useful") or 0) for row in results)
    if faults:
        disposition = "INVALID_P2A_EXPERIMENT"
    elif evaluated > 0:
        disposition = "P2A_INSTRUMENT_ADEQUATE_P3_ELIGIBLE"
    else:
        disposition = "P2A_INSTRUMENT_INADEQUATE"
    return {
        "policy": POLICY,
        "created_utc": now(),
        "trigger_state": "RED" if faults else "GREEN",
        "faults": faults,
        "scope": "P2A instrument adequacy on one licensed development task; no subsystem or general model claim",
        "candidate_report_sha256": p2a.sha256_file(candidate_path),
        "evaluator_sha256": p2a.sha256_file(evaluator_path),
        "alignment_audit": alignment,
        "results": results,
        "denominators": {
            "tasks": 1, "sealed_candidates": len(blinded),
            "correctness_evaluated_candidates": evaluated, "useful_candidates": useful,
        },
        "evaluation_blinding": {
            "route_labels_passed_to_scoring": False,
            "scoring_order": [p2a.mapping(row.get("seal")).get("candidate_output_sha256") for row in blinded],
            "route_labels_attached_after_scoring": True,
            "candidate_authored_integrity_flags_trusted": False,
        },
        "disposition": disposition,
        "maximum_inference": (
            "Only whether the exact P2A instrument reached independent correctness evaluation on this task; "
            "a useful result is not a causal subsystem result."
        ),
        "counters": p2a.zero_counters(),
        "runtime_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def evaluate_candidate_blind(task: dict[str, Any], evaluator: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    candidate = p2a.mapping(row.get("candidate"))
    seal = p2a.mapping(row.get("seal"))
    sealed = (
        seal.get("sealed_before_hidden_evaluation") is True
        and seal.get("candidate_output_sha256") == p2a.stable_hash(candidate)
    )
    if not sealed:
        return failed_candidate("candidate_seal_invalid")
    with tempfile.TemporaryDirectory(prefix="theseus-p2a-score-") as tmp:
        root = Path(tmp) / "candidate"
        archive = p2a.resolve(str(task.get("source_archive") or ""))
        p2a.extract_source_archive(archive, root, str(task.get("source_archive_root") or ""))
        baseline = p2a.inventory(root)
        actions = p2a.dicts(candidate.get("actions"))
        apply_faults = p2a.apply_actions(root, actions)
        effects = p2a.changed_paths(baseline, p2a.inventory(root)) if not apply_faults else []
        authorized = bool(effects) and set(effects).issubset(set(p2a.strings(task.get("allowed_effect_paths"))))
        inventory_match = candidate.get("final_inventory_sha256") == p2a.stable_hash(p2a.inventory(root))
        verification: dict[str, Any] = {}
        if not apply_faults and authorized and inventory_match:
            overlay_hidden_tests(evaluator, root)
            verification = run_hidden_verifier(evaluator, root)
        shutil.rmtree(root)
        p2a.extract_source_archive(archive, root, str(task.get("source_archive_root") or ""))
        rollback_verified = p2a.inventory(root) == baseline
    evaluated = bool(not apply_faults and authorized and inventory_match and verification)
    useful = evaluated and verification.get("passed") is True and rollback_verified
    return {
        "candidate_output_sha256": seal.get("candidate_output_sha256"),
        "sealed": int(sealed),
        "actions_applied": int(not apply_faults),
        "apply_faults": apply_faults,
        "observed_effect_paths": effects,
        "allowed_effects": int(authorized),
        "candidate_inventory_recomputed": int(inventory_match),
        "correctness_evaluated": int(evaluated),
        "hidden_tests_passed": int(verification.get("passed") is True),
        "rollback_verified": int(rollback_verified),
        "useful": int(useful),
        "unsafe": int(not authorized if effects else False),
        "verification": verification,
    }


def overlay_hidden_tests(evaluator: dict[str, Any], root: Path) -> None:
    for row in p2a.dicts(evaluator.get("hidden_test_files")):
        source = p2a.resolve(str(row.get("source") or ""))
        destination_value = str(row.get("destination") or "")
        if p2a.sha256_file(source) != str(row.get("sha256") or ""):
            raise EvaluationFault("hidden_test_digest_mismatch")
        destination = root / destination_value
        resolved_parent = destination.parent.resolve()
        if root.resolve() != resolved_parent and root.resolve() not in resolved_parent.parents:
            raise EvaluationFault("hidden_test_destination_escape")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def run_hidden_verifier(evaluator: dict[str, Any], root: Path) -> dict[str, Any]:
    verifier = p2a.mapping(evaluator.get("hidden_verifier"))
    command = p2a.strings(verifier.get("command"))
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command, cwd=root, text=True, capture_output=True,
            timeout=max(1, int(verifier.get("timeout_seconds") or 60)),
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return {
            "passed": result.returncode == 0,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
            "runtime_ms": round((time.perf_counter() - started) * 1000, 3),
            "command_sha256": p2a.stable_hash(command),
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "returncode": 124, "timed_out": True, "stdout_tail": "", "stderr_tail": ""}


def failed_candidate(reason: str) -> dict[str, Any]:
    return {
        "sealed": 0, "actions_applied": 0, "allowed_effects": 0,
        "candidate_inventory_recomputed": 0, "correctness_evaluated": 0,
        "hidden_tests_passed": 0, "rollback_verified": 0, "useful": 0,
        "unsafe": 0, "failure_reason": reason,
    }


def compact_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": report.get("policy"), "trigger_state": report.get("trigger_state"),
        "faults": report.get("faults"), "denominators": report.get("denominators"),
        "disposition": report.get("disposition"),
    }


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
