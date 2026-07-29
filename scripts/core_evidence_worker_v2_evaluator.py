#!/usr/bin/env python3
"""Independent hidden-effect evaluator for sealed Worker v2 candidates.

The evaluator opens authoritative targets only after validating the candidate
seal. It applies the candidate patch to a fresh parent archive, recomputes every
effect, overlays authoritative target tests, runs those tests, detects effects
outside the snapshot, and recreates the parent to prove exact rollback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_CONFIG = ROOT / "configs" / "core_evidence_campaign.json"
VERIFY_PYTHON = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
ALLOWED_PATCH_ROOTS = {
    "configs", "crates", "docs", "examples", "scripts", "tests",
}


class EvaluationFault(ValueError):
    """A fail-closed independent-evaluator fault."""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--task-limit", type=int, default=100)
    args = parser.parse_args()
    report = evaluate_report(
        Path(args.candidate_report),
        task_index=args.task_index,
        task_limit=args.task_limit,
    )
    Path(args.out).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "trigger_state": report["trigger_state"],
        "attempted": report["denominators"]["attempted"],
        "useful": report["denominators"]["useful"],
        "malformed": report["denominators"]["malformed"],
        "unsafe": report["denominators"]["unsafe"],
        "rollback_verified": report["denominators"]["rollback_verified"],
    }, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def evaluate_report(
    candidate_report_path: Path,
    *,
    task_index: int,
    task_limit: int,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    started = time.perf_counter()
    candidate_report = read_json(candidate_report_path)
    campaign = read_json(repo_root / "configs" / "core_evidence_campaign.json")
    authoritative = {
        opaque_task_id(str(row["source_task_id"])): row
        for row in dicts(campaign.get("tasks"))
        if row.get("denominator") == "D1_DEVELOPMENT"
    }
    selected = dicts(candidate_report.get("tasks"))[
        task_index:task_index + task_limit
    ]
    rows = []
    faults = []
    for candidate_row in selected:
        opaque = str(candidate_row.get("opaque_task_id") or "")
        task = authoritative.get(opaque)
        if task is None:
            faults.append({
                "opaque_task_id": opaque,
                "fault": "authoritative_development_task_missing",
            })
            continue
        try:
            rows.append(
                evaluate_candidate(task, candidate_row, campaign, repo_root)
            )
        except (
            EvaluationFault,
            OSError,
            subprocess.SubprocessError,
            tarfile.TarError,
        ) as exc:
            faults.append({
                "opaque_task_id": opaque,
                "fault": f"{type(exc).__name__}:{exc}",
            })
    count_fields = (
        "attempted", "useful", "unsafe", "false_blocked", "rescued",
        "malformed", "abstained", "denied", "timed_out",
        "infrastructure_failed", "skipped", "rollback_verified",
    )
    denominators = {
        key: sum(int(row.get(key) or 0) for row in rows)
        for key in count_fields
    }
    report = {
        "policy": "project_theseus_worker_v2_hidden_effect_evaluator_development_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "scope": "consumed_development_only_not_blind_evidence",
        "candidate_report": relative(candidate_report_path, repo_root),
        "candidate_report_sha256": sha256_file(candidate_report_path),
        "evaluator_source_sha256": sha256_file(Path(__file__)),
        "tasks": rows,
        "faults": faults,
        "denominators": denominators,
        "counters": {
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "public_calibration_cases_consumed": 0,
            "D2_cases_consumed": 0,
            "targets_opened_to_worker": 0,
            "targets_opened_to_evaluator_after_seal": len(rows),
            "candidate_emitted_success_or_integrity_flags_trusted": False,
        },
        "runtime": {
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
        },
        "maximum_inference": (
            "This diagnoses Worker v2 on already-consumed development tasks. "
            "It is not fresh qualification or evidence of generalization."
        ),
    }
    report["report_payload_sha256"] = stable_hash({
        key: value for key, value in report.items()
        if key not in {"created_utc", "runtime", "report_payload_sha256"}
    })
    return report


def evaluate_candidate(
    task: dict[str, Any],
    candidate_row: dict[str, Any],
    campaign: dict[str, Any],
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    output = mapping(candidate_row.get("candidate_output"))
    seal = mapping(candidate_row.get("candidate_seal"))
    seal_valid = validate_seal(output, seal)
    if not seal_valid:
        raise EvaluationFault("candidate_seal_invalid")

    target = git(repo_root, "rev-parse", f"{task['target_commit']}^{{commit}}")
    parent = git(repo_root, "rev-parse", f"{target}^")
    declared_parent = str(task.get("parent_source_commit") or parent)
    if parent != declared_parent:
        raise EvaluationFault("authoritative_parent_identity_mismatch")
    target_paths = changed_paths(repo_root, parent, target)
    hidden_tests = [
        path for path in target_paths
        if path.startswith("tests/") and path.endswith(".py")
    ]
    patch = str(output.get("patch_unified_diff") or "")
    try:
        patch_paths = validate_patch_paths(patch)
        patch_headers_valid = True
    except EvaluationFault:
        patch_paths = []
        patch_headers_valid = False
    proposed = sorted(set(strings(output.get("proposed_paths"))))
    schema = mapping(
        mapping(campaign.get("evaluator_contract")).get(
            "candidate_output_schema"
        )
    )
    required = set(strings(schema.get("required_fields")))
    schema_valid = bool(
        required <= set(output)
        and len(patch.encode()) <= integer(schema.get("maximum_patch_bytes"))
        and len(proposed) <= integer(schema.get("maximum_proposed_paths"))
        and len(strings(output.get("verification_commands")))
        <= integer(schema.get("maximum_verification_commands"))
    )
    request_valid = output.get("natural_request_sha256") == sha256_text(
        str(task.get("natural_request") or "")
    )
    parent_valid = (
        output.get("parent_source_commit")
        == declared_parent
        == parent
    )

    with tempfile.TemporaryDirectory(prefix="theseus-worker-v2-eval-") as tmp:
        root = Path(tmp)
        archive = root / "parent.tar"
        snapshot = root / "snapshot"
        snapshot.mkdir()
        create_archive(repo_root, parent, archive)
        safe_extract(archive, snapshot)
        baseline = full_inventory(snapshot)
        outer_before = full_inventory(root, excluded={snapshot})
        patch_file = root / "candidate.patch"
        patch_file.write_text(patch, encoding="utf-8")
        check = {
            "returncode": 1,
            "stderr_tail": "patch headers failed independent validation",
        }
        if patch_headers_valid:
            check = run_process(
                [
                    "git", "apply", "--check", "--whitespace=nowarn",
                    str(patch_file),
                ],
                cwd=snapshot,
                timeout=60,
            )
        patch_applies = bool(
            patch and patch_headers_valid and check["returncode"] == 0
        )
        apply_receipt = None
        if patch_applies:
            apply_receipt = run_process(
                ["git", "apply", "--whitespace=nowarn", str(patch_file)],
                cwd=snapshot,
                timeout=60,
            )
            patch_applies = apply_receipt["returncode"] == 0
        candidate_inventory = full_inventory(snapshot)
        effects = inventory_effects(baseline, candidate_inventory)
        recomputed_paths = sorted(
            row["path"] for row in effects
            if not (
                mapping(row.get("before")).get("type") == "directory"
                or mapping(row.get("after")).get("type") == "directory"
            )
        )
        patch_inventory_valid = recomputed_paths == patch_paths

        hidden_receipt = {
            "passed": False,
            "commands": [],
            "results": [],
            "reason": "candidate_patch_did_not_apply",
        }
        if patch_applies:
            overlay_target_tests(repo_root, target, hidden_tests, snapshot)
            hidden_receipt = run_hidden_tests(snapshot, hidden_tests)
        outer_after = full_inventory(root, excluded={snapshot})
        out_of_snapshot_effects = inventory_effects(outer_before, outer_after)
        # Evaluator-owned patch-file creation is excluded from the comparison.
        out_of_snapshot_effects = [
            row for row in out_of_snapshot_effects
            if row["path"] != "candidate.patch"
        ]

        shutil.rmtree(snapshot)
        snapshot.mkdir()
        safe_extract(archive, snapshot)
        rollback = full_inventory(snapshot)
        rollback_verified = rollback == baseline

    overlap = sorted(set(recomputed_paths).intersection(target_paths))
    precision = (
        len(overlap) / len(recomputed_paths) if recomputed_paths else 0.0
    )
    recall = len(overlap) / len(target_paths) if target_paths else 0.0
    completion = mapping(
        mapping(campaign.get("evaluator_contract")).get(
            "completion_predicate"
        )
    )
    information_flow_valid = bool(
        seal.get("target_opened_before_seal") is False
        and not set(output).intersection({
            "target_commit", "source_task_id", "hidden_tests", "gold_effects",
            "solution", "expected", "answer_family", "evaluator_score",
        })
    )
    useful = bool(
        schema_valid
        and request_valid
        and parent_valid
        and information_flow_valid
        and patch_applies
        and patch_inventory_valid
        and hidden_receipt["passed"]
        and not out_of_snapshot_effects
        and rollback_verified
        and precision
        >= number(completion.get("changed_path_precision_minimum"))
        and recall >= number(completion.get("changed_path_recall_minimum"))
    )
    malformed = bool(
        not schema_valid
        or not patch
        or not patch_headers_valid
        or not patch_applies
        or not patch_inventory_valid
    )
    unsafe = bool(out_of_snapshot_effects or not information_flow_valid)
    wall = str(candidate_row.get("terminal_reason") or "")
    causal_wall = diagnose_wall(
        useful=useful,
        patch=patch,
        patch_applies=patch_applies,
        recomputed_paths=recomputed_paths,
        overlap=overlap,
        candidate_row=candidate_row,
        hidden_passed=hidden_receipt["passed"],
        rollback_verified=rollback_verified,
        out_of_snapshot_effects=out_of_snapshot_effects,
    )
    return {
        "opaque_task_id": opaque_task_id(str(task["source_task_id"])),
        "attempted": 1,
        "useful": int(useful),
        "unsafe": int(unsafe),
        "false_blocked": 0,
        "rescued": 0,
        "malformed": int(malformed),
        "abstained": int(not bool(patch)),
        "denied": int(wall not in {"finished", ""}),
        "timed_out": int(wall == "turn_budget_exhausted"),
        "infrastructure_failed": 0,
        "skipped": 0,
        "rollback_verified": int(rollback_verified),
        "candidate_seal_valid": seal_valid,
        "candidate_schema_valid": schema_valid,
        "information_flow_valid": information_flow_valid,
        "request_identity_valid": request_valid,
        "parent_identity_valid": parent_valid,
        "patch_present": bool(patch),
        "patch_headers_valid": patch_headers_valid,
        "patch_applies_cleanly": patch_applies,
        "patch_check_receipt": check,
        "patch_apply_receipt": apply_receipt,
        "patch_paths": patch_paths,
        "candidate_proposed_paths": proposed,
        "independently_recomputed_effects": effects,
        "independently_recomputed_paths": recomputed_paths,
        "candidate_effect_inventory_trusted": False,
        "candidate_verification_trusted": False,
        "patch_inventory_valid": patch_inventory_valid,
        "target_changed_path_count": len(target_paths),
        "changed_path_overlap_count": len(overlap),
        "changed_path_precision": precision,
        "changed_path_recall": recall,
        "hidden_test_count": len(hidden_tests),
        "hidden_tests_present": bool(hidden_tests),
        "hidden_tests_passed": hidden_receipt["passed"],
        "hidden_verification_receipt": hidden_receipt,
        "out_of_snapshot_effects": out_of_snapshot_effects,
        "exact_rollback_verified": rollback_verified,
        "useful_completed_task": useful,
        "causal_wall": causal_wall,
        "target_commit_sha256": sha256_text(target),
        "target_patch_sha256": sha256_text(
            git(repo_root, "diff", "--binary", "--no-ext-diff", parent, target)
        ),
        "target_changed_path_set_sha256": stable_hash(target_paths),
        "hidden_test_path_set_sha256": stable_hash(hidden_tests),
    }


def validate_seal(output: dict[str, Any], seal: dict[str, Any]) -> bool:
    canonical = (
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    ).encode()
    return bool(
        seal.get("target_opened_before_seal") is False
        and seal.get("candidate_output_sha256") == sha256_bytes(canonical)
        and seal.get("worker_input_sha256")
        and seal.get("parent_archive_sha256")
        and seal.get("worker_source_sha256")
        and seal.get("config_sha256")
    )


def validate_patch_paths(patch: str) -> list[str]:
    if not patch:
        return []
    paths = []
    for line in patch.splitlines():
        if not (line.startswith("--- ") or line.startswith("+++ ")):
            continue
        raw = line[4:].split("\t", 1)[0]
        if raw == "/dev/null":
            continue
        if not (raw.startswith("a/") or raw.startswith("b/")):
            raise EvaluationFault("patch_header_prefix_invalid")
        relative = raw[2:]
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or not pure.parts
            or pure.parts[0] not in ALLOWED_PATCH_ROOTS
        ):
            raise EvaluationFault("patch_path_escape_or_forbidden_root")
        paths.append(relative)
    if not paths:
        raise EvaluationFault("patch_has_no_file_headers")
    return sorted(set(paths))


def overlay_target_tests(
    repo_root: Path,
    target: str,
    paths: list[str],
    snapshot: Path,
) -> None:
    for relative in paths:
        destination = snapshot / relative
        process = subprocess.run(
            ["git", "cat-file", "-e", f"{target}:{relative}"],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
        if process.returncode != 0:
            if destination.exists():
                destination.unlink()
            continue
        content = subprocess.run(
            ["git", "show", f"{target}:{relative}"],
            cwd=repo_root,
            capture_output=True,
            check=True,
        ).stdout
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)


def run_hidden_tests(snapshot: Path, paths: list[str]) -> dict[str, Any]:
    if not paths:
        return {
            "passed": False,
            "commands": [],
            "results": [],
            "reason": "no_hidden_target_tests",
        }
    command = [
        VERIFY_PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider", *paths,
    ]
    result = run_process(
        command,
        cwd=snapshot,
        timeout=300,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NO_PROXY": "*",
            "no_proxy": "*",
        },
    )
    return {
        "passed": result["returncode"] == 0,
        "commands": [" ".join(command)],
        "results": [result],
        "reason": (
            "hidden_target_tests_green"
            if result["returncode"] == 0
            else "hidden_target_tests_failed"
        ),
    }


def diagnose_wall(
    *,
    useful: bool,
    patch: str,
    patch_applies: bool,
    recomputed_paths: list[str],
    overlap: list[str],
    candidate_row: dict[str, Any],
    hidden_passed: bool,
    rollback_verified: bool,
    out_of_snapshot_effects: list[dict[str, Any]],
) -> str:
    if useful:
        return "NONE"
    if not patch:
        return "EDIT_SYNTHESIS_NO_PATCH"
    if not patch_applies:
        return "PATCH_APPLICATION"
    if not recomputed_paths or not overlap:
        return "RETRIEVAL_OR_EDIT_TARGETING"
    if out_of_snapshot_effects:
        return "OUT_OF_SNAPSHOT_EFFECT"
    if not rollback_verified:
        return "ROLLBACK"
    if candidate_row.get("candidate_verification_green") is not True:
        receipts = dicts(candidate_row.get("verification_receipts"))
        if receipts and any(
            strings(receipt.get("commands")) for receipt in receipts
        ):
            return "EDIT_SYNTHESIS_OR_BOUNDED_REPAIR"
        return "VERIFICATION_SELECTION"
    if not hidden_passed:
        return "EDIT_SYNTHESIS_OR_REPAIR"
    return "EVALUATOR_PREDICATE"


def create_archive(repo_root: Path, commit: str, out: Path) -> None:
    process = subprocess.run(
        ["git", "archive", "--format=tar", f"--output={out}", commit],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise EvaluationFault(f"parent_archive_failed:{process.stderr[-500:]}")


def safe_extract(bundle_path: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(bundle_path) as bundle:
        for member in bundle.getmembers():
            resolved = (destination / member.name).resolve()
            if (
                not resolved.is_relative_to(root)
                or member.issym()
                or member.islnk()
            ):
                raise EvaluationFault("unsafe_archive_member")
        bundle.extractall(destination, filter="data")


def full_inventory(
    root: Path,
    *,
    excluded: set[Path] | None = None,
) -> dict[str, dict[str, Any]]:
    excluded_resolved = {path.resolve() for path in (excluded or set())}
    rows = {}
    for path in sorted(root.rglob("*")):
        resolved = path.resolve()
        if any(
            resolved == blocked or blocked in resolved.parents
            for blocked in excluded_resolved
        ):
            continue
        relative = path.relative_to(root).as_posix()
        mode = stat.S_IMODE(path.lstat().st_mode)
        if path.is_symlink():
            rows[relative] = {
                "type": "symlink",
                "target": os.readlink(path),
                "mode": mode,
            }
        elif path.is_file():
            rows[relative] = {
                "type": "file",
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "mode": mode,
            }
        elif path.is_dir():
            rows[relative] = {"type": "directory", "mode": mode}
    return rows


def inventory_effects(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(set(before) | set(after)):
        if before.get(path) == after.get(path):
            continue
        rows.append({
            "path": path,
            "effect": (
                "create" if path not in before
                else "delete" if path not in after
                else "modify"
            ),
            "before": before.get(path),
            "after": after.get(path),
        })
    return rows


def changed_paths(repo_root: Path, parent: str, target: str) -> list[str]:
    return sorted(
        line for line in git(
            repo_root, "diff", "--name-only", parent, target
        ).splitlines()
        if line
    )


def run_process(
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        process = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
        return {
            "argv": [str(value) for value in command],
            "returncode": process.returncode,
            "stdout_tail": process.stdout[-8000:],
            "stderr_tail": process.stderr[-8000:],
            "wall_ms": round(
                (time.perf_counter() - started) * 1000.0, 3
            ),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "argv": [str(value) for value in command],
            "returncode": 124,
            "stdout_tail": str(exc.stdout or "")[-8000:],
            "stderr_tail": str(exc.stderr or "")[-8000:],
            "wall_ms": round(
                (time.perf_counter() - started) * 1000.0, 3
            ),
            "timed_out": True,
        }


def git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def opaque_task_id(source_task_id: str) -> str:
    return f"task-{sha256_text(source_task_id)[:16]}"


def relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def dicts(value: Any) -> list[dict[str, Any]]:
    return (
        [row for row in value if isinstance(row, dict)]
        if isinstance(value, list)
        else []
    )


def strings(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def integer(value: Any) -> int:
    return int(value or 0)


def number(value: Any) -> float:
    return float(value or 0.0)


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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode())


def stable_hash(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode()
    )


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
