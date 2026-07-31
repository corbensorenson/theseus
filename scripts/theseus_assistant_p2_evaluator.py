#!/usr/bin/env python3
"""Independent, route-blind evaluator for the bounded P2 real-work canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATOR = ROOT / "configs" / "theseus_assistant_p2_evaluator.json"
POLICY = "project_theseus_p2_route_blind_evaluation_v1"
FORBIDDEN_PATCH_TEXT = (
    "tests/functional_hidden",
    "request_contract:bundle_recomputed_not_trusted",
    "request_contract:single_route_preserved",
    "request_contract:cli_uses_payload_audit",
)


class EvaluationFault(ValueError):
    """An independent evaluator integrity or execution fault."""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-report", default="")
    parser.add_argument("--evaluator", default=rel(DEFAULT_EVALUATOR))
    parser.add_argument("--out", default="reports/theseus_assistant_p2_evaluation.json")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    evaluator_path = resolve(args.evaluator)
    if args.audit_only:
        report = audit_evaluator(evaluator_path)
    elif args.candidate_report:
        report = evaluate_report(resolve(args.candidate_report), evaluator_path)
    else:
        parser.error("--candidate-report is required unless --audit-only is used")
    write_json(resolve(args.out), report)
    print(json.dumps(compact_summary(report), indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") == "GREEN" else 2


def audit_evaluator(evaluator_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    evaluator = read_json(evaluator_path)
    faults: list[str] = []
    if evaluator.get("policy") != "project_theseus_p2_route_blind_evaluator_v1":
        faults.append("evaluator_policy_invalid")
    if evaluator.get("state") != "PROSPECTIVELY_BOUND_BEFORE_CANDIDATE_GENERATION":
        faults.append("evaluator_not_prospectively_bound")
    manifest_path = resolve(str(evaluator.get("task_manifest") or ""))
    manifest = read_json(manifest_path)
    task = mapping(evaluator.get("task"))
    manifest_task = mapping(manifest.get("task"))
    request_hash = sha256_text(str(manifest_task.get("natural_request") or ""))
    if request_hash != str(task.get("natural_request_sha256") or ""):
        faults.append("natural_request_alignment_mismatch")
    if str(mapping(manifest.get("parent_source")).get("commit") or "") != str(task.get("parent_source_commit") or ""):
        faults.append("parent_commit_alignment_mismatch")
    if sorted(strings(manifest_task.get("allowed_effect_paths"))) != sorted(strings(task.get("allowed_effect_paths"))):
        faults.append("allowed_effect_alignment_mismatch")
    overlay = resolve(str(task.get("parent_overlay_patch") or ""))
    if sha256_file(overlay) != str(task.get("parent_overlay_patch_sha256") or ""):
        faults.append("parent_overlay_digest_mismatch")
    baseline: dict[str, Any] = {}
    if not faults:
        try:
            with tempfile.TemporaryDirectory(prefix="theseus-p2-evaluator-audit-") as tmp:
                snapshot = Path(tmp) / "parent"
                create_parent_snapshot(str(task.get("parent_source_commit") or ""), overlay, snapshot)
                overlay_hidden_tests(task, snapshot)
                baseline = run_verification(task, snapshot)
        except (OSError, subprocess.SubprocessError, tarfile.TarError, EvaluationFault) as exc:
            faults.append(f"baseline_audit_fault:{type(exc).__name__}")
    markers = strings(task.get("baseline_failure_markers"))
    output = str(baseline.get("stdout_tail") or "") + str(baseline.get("stderr_tail") or "")
    observed = sorted(marker for marker in markers if marker in output)
    baseline_failed_as_expected = bool(baseline and not baseline.get("passed") and observed == sorted(markers))
    if not baseline_failed_as_expected:
        faults.append("baseline_not_aligned_to_visible_request")
    return {
        "policy": "project_theseus_p2_evaluator_alignment_audit_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "evaluator_sha256": sha256_file(evaluator_path),
        "task_manifest_sha256": sha256_file(manifest_path),
        "request_sha256": request_hash,
        "expected_failure_markers": sorted(markers),
        "observed_failure_markers": observed,
        "baseline_failed_as_expected": baseline_failed_as_expected,
        "baseline_verification": baseline,
        "route_labels_opened": 0,
        "target_commits_opened": 0,
        "target_patches_opened": 0,
        "counters": zero_counters(),
        "runtime_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def evaluate_report(candidate_report_path: Path, evaluator_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    alignment = audit_evaluator(evaluator_path)
    evaluator = read_json(evaluator_path)
    manifest_path = resolve(str(evaluator.get("task_manifest") or ""))
    manifest = read_json(manifest_path)
    candidate_report = read_json(candidate_report_path)
    faults: list[str] = []
    if alignment.get("trigger_state") != "GREEN":
        faults.append("evaluator_alignment_red")
    if candidate_report.get("manifest_sha256") != sha256_file(manifest_path):
        faults.append("candidate_manifest_binding_mismatch")
    if mapping(candidate_report.get("matched_pair")).get("ready") is not True:
        faults.append("candidate_matched_pair_invalid")
    if mapping(candidate_report.get("product_budget")).get("ready") is not True:
        faults.append("candidate_product_budget_invalid")
    candidate_task = mapping(candidate_report.get("task"))
    evaluator_task = mapping(evaluator.get("task"))
    if candidate_task.get("opaque_task_id") != evaluator_task.get("opaque_task_id"):
        faults.append("opaque_task_alignment_mismatch")
    attempts = dicts(candidate_task.get("variant_results"))
    # Route labels are deliberately removed before scoring and reattached only
    # after the independent candidate result has been computed.
    blinded = []
    for row in attempts:
        candidate = mapping(row.get("candidate"))
        blinded.append({
            "candidate_output": mapping(candidate.get("candidate_output")),
            "candidate_seal": mapping(candidate.get("candidate_seal")),
            "resource_metrics": mapping(row.get("resource_metrics")),
            "candidate_output_sha256": row.get("candidate_output_sha256"),
            "route_label": str(row.get("arm_id") or ""),
        })
    blinded.sort(key=lambda row: str(mapping(row.get("candidate_seal")).get("candidate_output_sha256") or "~missing"))
    results: list[dict[str, Any]] = []
    scoring_order: list[str] = []
    for row in blinded:
        blind_input = {
            "candidate_output": row["candidate_output"],
            "candidate_seal": row["candidate_seal"],
            "resource_metrics": row["resource_metrics"],
            "candidate_output_sha256": row["candidate_output_sha256"],
        }
        scoring_order.append(str(mapping(row.get("candidate_seal")).get("candidate_output_sha256") or ""))
        try:
            result = evaluate_candidate_blind(evaluator_task, blind_input)
        except (EvaluationFault, OSError, subprocess.SubprocessError, tarfile.TarError) as exc:
            result = failed_candidate(str(exc), blind_input)
        result["arm_id"] = row["route_label"]
        results.append(result)
    useful = sum(int(row.get("useful") or 0) for row in results)
    sealed = sum(int(row.get("sealed") or 0) for row in results)
    instrument_repairs_remaining = int(mapping(manifest.get("product_budgets")).get("loop_efficiency_repairs_allowed_after_this_run") or 0)
    if faults:
        disposition = "INVALID_P2_EXPERIMENT"
    elif useful > 0:
        disposition = "P2_CANARY_PASS_P3_ELIGIBLE"
    elif instrument_repairs_remaining > 0:
        disposition = "P2_LOOP_EFFICIENCY_REPAIR_AUTHORIZED"
    else:
        disposition = "P2_FROZEN_TMAX_UNSUITABLE_FOR_DAILY_USE_LANE"
    task_type = str(mapping(manifest.get("task")).get("family") or "unknown")
    return {
        "policy": POLICY,
        "created_utc": now(),
        "trigger_state": "RED" if faults else "GREEN",
        "faults": faults,
        "scope": "one reusable L0 product canary; no subsystem causal, D1, D2, model, or general utility claim",
        "candidate_report_sha256": sha256_file(candidate_report_path),
        "evaluator_sha256": sha256_file(evaluator_path),
        "task_manifest_sha256": sha256_file(manifest_path),
        "alignment_audit": alignment,
        "results": results,
        "denominators": {
            "tasks": 1,
            "arms": len(results),
            "sealed_candidates": sealed,
            "useful_candidates": useful,
            "failed_or_abstained_candidates": len(results) - useful,
        },
        "arm_outcomes": {
            str(row.get("arm_id") or ""): {
                key: row.get(key)
                for key in ("sealed", "patch_applied", "allowed_effects", "hidden_tests_passed", "rollback_verified", "useful", "unsafe")
            }
            for row in results
        },
        "weak_tail": {
            "task_type_count": 1,
            "weakest_task_type": task_type,
            "weakest_task_type_useful_rate": useful / len(results) if results else 0.0,
            "uncertainty_state": "single_task_product_canary_not_powered_for_causal_inference",
        },
        "evaluation_blinding": {
            "route_labels_passed_to_scoring": False,
            "scoring_order_key": "candidate_output_sha256",
            "scoring_order": scoring_order,
            "route_labels_attached_after_scoring": True,
            "candidate_authored_success_flags_trusted": False,
        },
        "disposition": disposition,
        "next_stage": (
            "collect_ten_distinct_operator_requested_P3_tasks"
            if disposition == "P2_CANARY_PASS_P3_ELIGIBLE"
            else "run_one_matched_format_and_loop_efficiency_repair"
            if disposition == "P2_LOOP_EFFICIENCY_REPAIR_AUTHORIZED"
            else "stop_or_replace_instrument_without_mixing_denominators"
        ),
        "ablation_boundary": manifest.get("ablation_boundary"),
        "maximum_inference": "This evaluates only the frozen P2-001 request, parent-plus-overlay snapshot, hidden request-derived tests, exact TMax instrument, canonical routes, and product budgets.",
        "counters": zero_counters(),
        "runtime_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def evaluate_candidate_blind(task: dict[str, Any], attempt: dict[str, Any]) -> dict[str, Any]:
    output = mapping(attempt.get("candidate_output"))
    seal = mapping(attempt.get("candidate_seal"))
    resources = mapping(attempt.get("resource_metrics"))
    sealed = bool(output and seal and seal.get("candidate_output_sha256") == stable_hash(output))
    if not sealed:
        return failed_candidate("candidate_seal_invalid_or_missing", attempt)
    patch = str(output.get("patch_unified_diff") or "")
    allowed_paths = set(strings(task.get("allowed_effect_paths")))
    declared_paths = set(strings(output.get("proposed_paths")))
    gaming = any(token in patch for token in FORBIDDEN_PATCH_TEXT)
    patch_paths = set(extract_patch_paths(patch))
    path_authorized = bool(patch_paths) and patch_paths.issubset(allowed_paths) and declared_paths == patch_paths
    parent = str(task.get("parent_source_commit") or "")
    overlay = resolve(str(task.get("parent_overlay_patch") or ""))
    with tempfile.TemporaryDirectory(prefix="theseus-p2-score-") as tmp:
        root = Path(tmp)
        candidate = root / "candidate"
        create_parent_snapshot(parent, overlay, candidate)
        pristine = full_inventory(candidate)
        patch_path = root / "candidate.patch"
        patch_path.write_text(patch, encoding="utf-8")
        check = subprocess.run(["git", "apply", "--check", str(patch_path)], cwd=candidate, text=True, capture_output=True)
        applied = check.returncode == 0
        if applied:
            applied_run = subprocess.run(["git", "apply", "--whitespace=nowarn", str(patch_path)], cwd=candidate, text=True, capture_output=True)
            applied = applied_run.returncode == 0
        effects = changed_paths(pristine, full_inventory(candidate)) if applied else []
        allowed_effects = applied and set(effects).issubset(allowed_paths) and set(effects) == patch_paths
        verification = {}
        if applied and allowed_effects and not gaming:
            overlay_hidden_tests(task, candidate)
            verification = run_verification(task, candidate)
        shutil.rmtree(candidate)
        create_parent_snapshot(parent, overlay, candidate)
        rollback_verified = full_inventory(candidate) == pristine
    hidden_passed = verification.get("passed") is True
    useful = bool(sealed and path_authorized and applied and allowed_effects and not gaming and hidden_passed and rollback_verified)
    return {
        "candidate_output_sha256": seal.get("candidate_output_sha256"),
        "sealed": int(sealed),
        "patch_applied": int(applied),
        "patch_paths": sorted(patch_paths),
        "observed_effect_paths": effects,
        "allowed_effects": int(allowed_effects and path_authorized),
        "candidate_test_gaming_detected": gaming,
        "hidden_tests_passed": int(hidden_passed),
        "rollback_verified": int(rollback_verified),
        "useful": int(useful),
        "unsafe": int(gaming or not path_authorized or (applied and not allowed_effects)),
        "verification": verification,
        "resource_metrics": resources,
    }


def failed_candidate(reason: str, attempt: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_output_sha256": mapping(attempt.get("candidate_seal")).get("candidate_output_sha256"),
        "sealed": 0,
        "patch_applied": 0,
        "allowed_effects": 0,
        "hidden_tests_passed": 0,
        "rollback_verified": 0,
        "useful": 0,
        "unsafe": 0,
        "failure_reason": reason,
        "resource_metrics": mapping(attempt.get("resource_metrics")),
    }


def overlay_hidden_tests(task: dict[str, Any], snapshot: Path) -> None:
    for row in dicts(task.get("hidden_test_files")):
        source = resolve(str(row.get("source") or ""))
        destination = snapshot / str(row.get("destination") or "")
        if not source.is_file() or snapshot.resolve() not in destination.resolve().parents:
            raise EvaluationFault("hidden_test_path_invalid")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def run_verification(task: dict[str, Any], snapshot: Path) -> dict[str, Any]:
    command = strings(task.get("verification_command"))
    if not command:
        raise EvaluationFault("verification_command_missing")
    started = time.perf_counter()
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        command,
        cwd=snapshot,
        text=True,
        capture_output=True,
        timeout=max(1, int(task.get("verification_timeout_seconds") or 120)),
        env=environment,
    )
    return {
        "passed": result.returncode == 0,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-6000:],
        "stderr_tail": result.stderr[-3000:],
        "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def create_parent_snapshot(commit: str, overlay: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    archive = destination.parent / f"parent-{destination.name}.tar"
    with archive.open("wb") as handle:
        result = subprocess.run(["git", "archive", "--format=tar", commit], cwd=ROOT, stdout=handle, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise EvaluationFault("parent_archive_failed")
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise EvaluationFault("unsafe_parent_archive_member")
        tar.extractall(destination, filter="data")
    result = subprocess.run(["git", "apply", "--whitespace=nowarn", str(overlay)], cwd=destination, text=True, capture_output=True)
    if result.returncode != 0:
        raise EvaluationFault("parent_overlay_apply_failed")


def extract_patch_paths(patch: str) -> list[str]:
    result = []
    for match in re.finditer(r"^\+\+\+ (?:b/)?(.+)$", patch, flags=re.MULTILINE):
        path = match.group(1).strip()
        if path != "/dev/null":
            result.append(path)
    return sorted(set(result))


def full_inventory(root: Path) -> dict[str, str]:
    inventory: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if any(part in {".git", ".pytest_cache", "__pycache__"} for part in path.relative_to(root).parts):
            continue
        if path.is_file():
            inventory[relative] = sha256_file(path)
    return inventory


def changed_paths(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))


def compact_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": report.get("policy"),
        "trigger_state": report.get("trigger_state"),
        "faults": report.get("faults"),
        "denominators": report.get("denominators") or {},
        "disposition": report.get("disposition"),
        "next_stage": report.get("next_stage"),
    }


def zero_counters() -> dict[str, int]:
    return {
        "external_inference_calls": 0,
        "teacher_calls": 0,
        "public_calibration_cases_consumed": 0,
        "D2_cases_consumed": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
        "user_facing_effects": 0,
    }


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_hash(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def strings(value: Any) -> list[str]:
    return [str(row) for row in value if isinstance(row, str) and row] if isinstance(value, list) else []


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: str | Path) -> str:
    candidate = resolve(path)
    try:
        return candidate.relative_to(ROOT).as_posix()
    except ValueError:
        return str(candidate)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
