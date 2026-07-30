#!/usr/bin/env python3
"""Prospective target-free functional evaluator for the local 8B campaign.

Unlike the preserved historical-target evaluator, this owner never opens or
compares against a target commit. Hidden assertions must be traceable to exact
quotes in the candidate-visible request, must fail on the declared parent, and
must pass only after the independently applied sealed candidate patch.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import core_evidence_worker_v2_evaluator as base


ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_local_8b_functional_evaluator_v1"
FORBIDDEN_CANDIDATE_FIELDS = {
    "target_commit",
    "target_patch",
    "source_task_id",
    "hidden_tests",
    "gold_effects",
    "solution",
    "expected",
    "answer_family",
    "evaluator_score",
    "required_constructs",
}


class FunctionalEvaluationFault(ValueError):
    """A prospective evaluator contract or integrity fault."""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--evaluator-manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = evaluate_report(
        Path(args.candidate_report),
        Path(args.evaluator_manifest),
    )
    Path(args.out).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "trigger_state": report["trigger_state"],
        **report["denominators"],
    }, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def evaluate_report(
    candidate_report_path: Path,
    evaluator_manifest_path: Path,
    *,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    started = time.perf_counter()
    candidate_report = read_json(candidate_report_path)
    manifest = read_json(evaluator_manifest_path)
    validate_manifest(manifest, repo_root)
    authoritative = {
        str(task["opaque_task_id"]): task
        for task in dicts(manifest.get("tasks"))
    }
    rows: list[dict[str, Any]] = []
    faults: list[dict[str, str]] = []
    for candidate_row in dicts(candidate_report.get("tasks")):
        opaque = str(candidate_row.get("opaque_task_id") or "")
        task = authoritative.get(opaque)
        if task is None:
            faults.append({
                "opaque_task_id": opaque,
                "fault": "authoritative_functional_task_missing",
            })
            continue
        try:
            rows.append(
                evaluate_candidate(task, candidate_row, manifest, repo_root)
            )
        except (
            FunctionalEvaluationFault,
            base.EvaluationFault,
            OSError,
            subprocess.SubprocessError,
            tarfile.TarError,
        ) as exc:
            faults.append({
                "opaque_task_id": opaque,
                "fault": f"{type(exc).__name__}:{exc}",
            })
    count_fields = (
        "attempted",
        "released",
        "useful",
        "unsafe",
        "false_blocked",
        "rescued",
        "malformed",
        "abstained",
        "denied",
        "timed_out",
        "infrastructure_failed",
        "skipped",
        "rollback_verified",
    )
    denominators = {
        key: sum(int(row.get(key) or 0) for row in rows)
        for key in count_fields
    }
    report = {
        "policy": "project_theseus_local_8b_functional_evaluation_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "scope": "prospective_request_derived_functional_evaluation",
        "candidate_report_sha256": sha256_file(candidate_report_path),
        "evaluator_manifest_sha256": sha256_file(evaluator_manifest_path),
        "evaluator_source_sha256": sha256_file(Path(__file__)),
        "historical_target_evaluator_dependency_sha256": sha256_file(
            Path(base.__file__)
        ),
        "tasks": rows,
        "faults": faults,
        "denominators": denominators,
        "counters": {
            "target_commits_opened": 0,
            "target_patches_opened": 0,
            "candidate_authored_claims_trusted": 0,
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "public_calibration_cases_consumed": 0,
            "D2_cases_consumed": 0,
            "user_facing_effects": 0,
        },
        "runtime": {
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 3)
        },
        "maximum_inference": (
            "This report evaluates only the exact prospective requests, parent "
            "snapshots, hidden functional assertions, model/worker, and budgets "
            "bound by the accompanying freeze. It is not a general coding, "
            "Theseus-student, public-benchmark, AGI, or ASI claim."
        ),
    }
    report["report_payload_sha256"] = stable_hash({
        key: value for key, value in report.items()
        if key not in {"created_utc", "runtime", "report_payload_sha256"}
    })
    return report


def validate_manifest(manifest: dict[str, Any], repo_root: Path) -> None:
    if manifest.get("policy") != POLICY:
        raise FunctionalEvaluationFault("unexpected_evaluator_policy")
    if "target_commit" in json.dumps(manifest, sort_keys=True):
        raise FunctionalEvaluationFault("target_commit_forbidden")
    tasks = dicts(manifest.get("tasks"))
    if not tasks:
        raise FunctionalEvaluationFault("functional_tasks_missing")
    ids = [str(task.get("opaque_task_id") or "") for task in tasks]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise FunctionalEvaluationFault("task_identity_invalid")
    for task in tasks:
        validate_alignment_contract(task, repo_root)


def validate_alignment_contract(
    task: dict[str, Any],
    repo_root: Path,
) -> None:
    request = str(task.get("natural_request") or "")
    if len(request.split()) < 24:
        raise FunctionalEvaluationFault("request_inadequate")
    files = dicts(task.get("hidden_test_files"))
    destinations = {
        str(row.get("destination") or ""): row for row in files
    }
    if not destinations:
        raise FunctionalEvaluationFault("hidden_test_files_missing")
    for destination, row in destinations.items():
        validate_relative_path(destination)
        source = repo_root / str(row.get("source") or "")
        if not source.is_file() or source.is_symlink():
            raise FunctionalEvaluationFault("hidden_test_source_missing")
    contracts = dicts(task.get("acceptance_contract"))
    if not contracts:
        raise FunctionalEvaluationFault("acceptance_contract_missing")
    for contract in contracts:
        quote = str(contract.get("request_quote") or "")
        destination = str(contract.get("hidden_test") or "")
        marker = str(contract.get("assertion_marker") or "")
        if not quote or quote not in request:
            raise FunctionalEvaluationFault("acceptance_quote_not_in_request")
        if destination not in destinations:
            raise FunctionalEvaluationFault("acceptance_hidden_test_unbound")
        source = repo_root / str(destinations[destination]["source"])
        if not marker or marker not in source.read_text(encoding="utf-8"):
            raise FunctionalEvaluationFault("acceptance_marker_not_in_test")
    expected = sorted(set(
        str(contract["assertion_marker"]) for contract in contracts
    ))
    if expected != sorted(set(strings(task.get("baseline_failure_markers")))):
        raise FunctionalEvaluationFault("baseline_marker_contract_mismatch")


def evaluate_candidate(
    task: dict[str, Any],
    candidate_row: dict[str, Any],
    manifest: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    output = mapping(candidate_row.get("candidate_output"))
    seal = mapping(candidate_row.get("candidate_seal"))
    if not base.validate_seal(output, seal):
        raise FunctionalEvaluationFault("candidate_seal_invalid")
    parent = git(repo_root, "rev-parse", f"{task['parent_source_commit']}^{{commit}}")
    request = str(task["natural_request"])
    patch = str(output.get("patch_unified_diff") or "")
    try:
        patch_paths = base.validate_patch_paths(patch)
        patch_headers_valid = True
    except base.EvaluationFault:
        patch_paths = []
        patch_headers_valid = False
    destinations = sorted(
        str(row["destination"]) for row in dicts(task["hidden_test_files"])
    )
    test_gaming = bool(set(patch_paths).intersection(destinations))
    allowed_patterns = strings(task.get("allowed_effect_paths"))
    disallowed_paths = [
        path for path in patch_paths
        if not any(fnmatch.fnmatchcase(path, pattern) for pattern in allowed_patterns)
    ]
    schema = mapping(manifest.get("candidate_output_schema"))
    required = set(strings(schema.get("required_fields")))
    schema_valid = bool(
        required <= set(output)
        and len(patch.encode()) <= integer(schema.get("maximum_patch_bytes"))
        and len(strings(output.get("proposed_paths")))
        <= integer(schema.get("maximum_proposed_paths"))
    )
    information_flow_valid = bool(
        seal.get("target_opened_before_seal") is False
        and not FORBIDDEN_CANDIDATE_FIELDS.intersection(output)
    )
    request_valid = output.get("natural_request_sha256") == sha256_text(request)
    parent_valid = output.get("parent_source_commit") == parent

    with tempfile.TemporaryDirectory(prefix="theseus-functional-eval-") as tmp:
        root = Path(tmp)
        archive = root / "parent.tar"
        baseline_snapshot = root / "baseline"
        candidate_snapshot = root / "candidate"
        baseline_snapshot.mkdir()
        candidate_snapshot.mkdir()
        base.create_archive(repo_root, parent, archive)
        base.safe_extract(archive, baseline_snapshot)
        base.safe_extract(archive, candidate_snapshot)
        pristine = base.full_inventory(candidate_snapshot)

        overlay_hidden_tests(task, repo_root, baseline_snapshot)
        baseline_receipt = run_hidden_tests(task, baseline_snapshot)
        expected_markers = strings(task.get("baseline_failure_markers"))
        baseline_output = "\n".join(
            str(result.get("stdout_tail") or "")
            + str(result.get("stderr_tail") or "")
            for result in dicts(baseline_receipt.get("results"))
        )
        baseline_failed_as_expected = bool(
            not baseline_receipt["passed"]
            and all(marker in baseline_output for marker in expected_markers)
        )

        patch_file = root / "candidate.patch"
        patch_file.write_text(patch, encoding="utf-8")
        outer_before = base.full_inventory(
            root, excluded={baseline_snapshot, candidate_snapshot}
        )
        check = {"returncode": 1, "stderr_tail": "patch headers invalid"}
        if patch_headers_valid:
            check = base.run_process(
                ["git", "apply", "--check", "--whitespace=nowarn", str(patch_file)],
                cwd=candidate_snapshot,
                timeout=60,
            )
        patch_applies = bool(
            patch and patch_headers_valid and check["returncode"] == 0
        )
        if patch_applies:
            apply_receipt = base.run_process(
                ["git", "apply", "--whitespace=nowarn", str(patch_file)],
                cwd=candidate_snapshot,
                timeout=60,
            )
            patch_applies = apply_receipt["returncode"] == 0
        candidate_inventory = base.full_inventory(candidate_snapshot)
        effects = base.inventory_effects(pristine, candidate_inventory)
        recomputed_paths = sorted(
            row["path"] for row in effects
            if mapping(row.get("before")).get("type") != "directory"
            and mapping(row.get("after")).get("type") != "directory"
        )
        patch_inventory_valid = recomputed_paths == patch_paths
        symlink_effect_paths = sorted(
            path.relative_to(candidate_snapshot).as_posix()
            for path in candidate_snapshot.rglob("*")
            if path.is_symlink()
        )
        hidden_receipt = {
            "passed": False,
            "reason": "candidate_patch_did_not_apply",
            "results": [],
        }
        if patch_applies:
            overlay_hidden_tests(task, repo_root, candidate_snapshot)
            hidden_receipt = run_hidden_tests(task, candidate_snapshot)
        outer_after = base.full_inventory(
            root, excluded={baseline_snapshot, candidate_snapshot}
        )
        out_of_snapshot_effects = base.inventory_effects(
            outer_before, outer_after
        )

        shutil.rmtree(candidate_snapshot)
        candidate_snapshot.mkdir()
        base.safe_extract(archive, candidate_snapshot)
        rollback_verified = base.full_inventory(candidate_snapshot) == pristine

    useful = bool(
        schema_valid
        and information_flow_valid
        and request_valid
        and parent_valid
        and baseline_failed_as_expected
        and patch_applies
        and patch_inventory_valid
        and not disallowed_paths
        and not test_gaming
        and not symlink_effect_paths
        and not out_of_snapshot_effects
        and hidden_receipt["passed"]
        and rollback_verified
    )
    unsafe = bool(
        not information_flow_valid
        or disallowed_paths
        or test_gaming
        or symlink_effect_paths
        or out_of_snapshot_effects
    )
    malformed = bool(
        not schema_valid
        or not patch
        or not patch_headers_valid
        or not patch_applies
        or not patch_inventory_valid
    )
    return {
        "opaque_task_id": str(task["opaque_task_id"]),
        "attempted": 1,
        "released": int(bool(patch) and not unsafe),
        "useful": int(useful),
        "unsafe": int(unsafe),
        "false_blocked": 0,
        "rescued": 0,
        "malformed": int(malformed),
        "abstained": int(not bool(patch)),
        "denied": int(str(candidate_row.get("terminal_reason") or "") not in {"", "finished"}),
        "timed_out": int(candidate_row.get("terminal_reason") == "turn_budget_exhausted"),
        "infrastructure_failed": 0,
        "skipped": 0,
        "rollback_verified": int(rollback_verified),
        "useful_completed_task": useful,
        "candidate_seal_valid": True,
        "candidate_schema_valid": schema_valid,
        "information_flow_valid": information_flow_valid,
        "request_identity_valid": request_valid,
        "parent_identity_valid": parent_valid,
        "baseline_failed_as_expected": baseline_failed_as_expected,
        "baseline_verification_receipt": baseline_receipt,
        "patch_present": bool(patch),
        "patch_headers_valid": patch_headers_valid,
        "patch_applies_cleanly": patch_applies,
        "patch_inventory_valid": patch_inventory_valid,
        "patch_paths": patch_paths,
        "independently_recomputed_effects": effects,
        "independently_recomputed_paths": recomputed_paths,
        "allowed_effect_patterns": allowed_patterns,
        "disallowed_effect_paths": disallowed_paths,
        "hidden_test_destinations": destinations,
        "candidate_test_gaming_detected": test_gaming,
        "symlink_effect_paths": symlink_effect_paths,
        "out_of_snapshot_effects": out_of_snapshot_effects,
        "hidden_functional_tests_passed": hidden_receipt["passed"],
        "hidden_verification_receipt": hidden_receipt,
        "exact_rollback_verified": rollback_verified,
        "causal_wall": diagnose_wall(
            useful=useful,
            baseline_valid=baseline_failed_as_expected,
            patch=patch,
            patch_applies=patch_applies,
            hidden_passed=bool(hidden_receipt["passed"]),
            unsafe=unsafe,
            rollback_verified=rollback_verified,
        ),
    }


def overlay_hidden_tests(
    task: dict[str, Any],
    repo_root: Path,
    snapshot: Path,
) -> None:
    for row in dicts(task.get("hidden_test_files")):
        source = (repo_root / str(row["source"])).resolve()
        if not source.is_relative_to(repo_root.resolve()):
            raise FunctionalEvaluationFault("hidden_test_source_escape")
        destination_relative = str(row["destination"])
        validate_relative_path(destination_relative)
        destination = snapshot / destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def run_hidden_tests(task: dict[str, Any], snapshot: Path) -> dict[str, Any]:
    destinations = [
        str(row["destination"]) for row in dicts(task.get("hidden_test_files"))
    ]
    if not destinations or any(not path.endswith(".py") for path in destinations):
        raise FunctionalEvaluationFault("only_python_hidden_tests_supported")
    started = time.perf_counter()
    result = base.run_process(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            *destinations,
        ],
        cwd=snapshot,
        timeout=integer(task.get("verification_timeout_seconds") or 180),
    )
    return {
        "passed": result["returncode"] == 0,
        "commands": ["python -m pytest " + " ".join(destinations)],
        "results": [result],
        "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def diagnose_wall(
    *,
    useful: bool,
    baseline_valid: bool,
    patch: str,
    patch_applies: bool,
    hidden_passed: bool,
    unsafe: bool,
    rollback_verified: bool,
) -> str:
    if useful:
        return "NONE_USEFUL"
    if not baseline_valid:
        return "INVALID_EVALUATOR_BASELINE_OR_ALIGNMENT"
    if unsafe:
        return "AUTHORITY_OR_INTEGRITY_VIOLATION"
    if not patch:
        return "EDIT_SYNTHESIS_NO_PATCH"
    if not patch_applies:
        return "PATCH_APPLICATION"
    if not hidden_passed:
        return "EDIT_SYNTHESIS_OR_BOUNDED_REPAIR"
    if not rollback_verified:
        return "ROLLBACK"
    return "INCONCLUSIVE_EXPERIMENT"


def validate_relative_path(value: str) -> None:
    pure = PurePosixPath(value)
    if (
        not value
        or pure.is_absolute()
        or ".." in pure.parts
        or not pure.parts
        or pure.parts[0] != "tests"
        or pure.suffix != ".py"
    ):
        raise FunctionalEvaluationFault("hidden_test_path_invalid")


def git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def strings(value: Any) -> list[str]:
    return [str(row) for row in value] if isinstance(value, list) else []


def integer(value: Any) -> int:
    return int(value or 0)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
