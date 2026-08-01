#!/usr/bin/env python3
"""Materialize and audit the sealed ten-task Theseus P3 development pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REGISTRY = ROOT / "configs" / "theseus_p3_task_sources.json"
INSTRUMENT = ROOT / "configs" / "theseus_assistant_p3_instrument.json"
INSTRUMENT_AUDIT = ROOT / "reports" / "theseus_assistant_p3_instrument_audit.json"
INSTRUMENT_COMMIT = "d08bf94653ee3a5ca508a2457ca21d58a4010a98"
EXPECTED_INSTRUMENT_SHA256 = "3462e683e23133b796a7258445e26680dc91d0e1d9ec184ece520391b93255ba"
EXPECTED_INSTRUMENT_AUDIT_SHA256 = "856332be39f88bdc43c109bca6f7b5310406fe76b2194d7f60490836921e5037"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialize-only", action="store_true")
    args = parser.parse_args()
    report = materialize_pool(run_audits=not args.materialize_only)
    print(json.dumps({
        "state": report["state"],
        "task_count": report["task_count"],
        "green_evaluator_audits": report["green_evaluator_audits"],
        "faults": report["faults"],
    }, indent=2, sort_keys=True))
    return 0 if report["state"] == "SEALED_BEFORE_CANDIDATE_GENERATION" else 2


def materialize_pool(*, run_audits: bool) -> dict[str, Any]:
    registry = read_json(SOURCE_REGISTRY)
    faults = audit_registry(registry)
    entries: list[dict[str, Any]] = []
    for source in registry.get("tasks", []):
        if not isinstance(source, dict):
            faults.append("source_row_not_object")
            continue
        entry, entry_faults = materialize_task(source, registry)
        entries.append(entry)
        faults.extend(entry_faults)
    if run_audits and not faults:
        for entry in entries:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "theseus_assistant_p2a_evaluator.py"),
                    "--evaluator",
                    entry["evaluator"],
                    "--audit-only",
                    "--out",
                    entry["evaluator_audit"],
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=180,
            )
            if result.returncode != 0:
                faults.append(f"evaluator_audit_red:{entry['stem']}")
            audit = read_json(ROOT / entry["evaluator_audit"])
            entry["evaluator_audit_sha256"] = sha256_file(ROOT / entry["evaluator_audit"])
            entry["evaluator_audit_trigger_state"] = audit.get("trigger_state")
            entry["baseline_parent_failed"] = (
                audit.get("baseline_verification", {}).get("passed") is False
            )
            entry["upstream_target_passed"] = (
                audit.get("target_verification", {}).get("passed") is True
            )
    else:
        for entry in entries:
            entry["evaluator_audit_sha256"] = ""
            entry["evaluator_audit_trigger_state"] = "NOT_RUN"
            entry["baseline_parent_failed"] = False
            entry["upstream_target_passed"] = False
    green = sum(
        entry.get("evaluator_audit_trigger_state") == "GREEN" for entry in entries
    )
    if run_audits and green != 10:
        faults.append("not_all_evaluator_audits_green")
    pool = {
        "policy": "project_theseus_p3_licensed_task_pool_v1",
        "state": (
            "SEALED_BEFORE_CANDIDATE_GENERATION"
            if run_audits and not faults and green == 10
            else "INVALID_NOT_SEALED"
        ),
        "partition": "development_residual_campaign",
        "sealed_utc": registry.get("sealed_utc"),
        "candidate_generation_opened": False,
        "selection_rule": registry.get("selection_rule"),
        "source_registry": relative(SOURCE_REGISTRY),
        "source_registry_sha256": sha256_file(SOURCE_REGISTRY),
        "instrument": relative(INSTRUMENT),
        "instrument_freeze_commit": INSTRUMENT_COMMIT,
        "instrument_sha256": sha256_file(INSTRUMENT),
        "instrument_audit": relative(INSTRUMENT_AUDIT),
        "instrument_audit_sha256": sha256_file(INSTRUMENT_AUDIT),
        "task_count": len(entries),
        "green_evaluator_audits": green,
        "distinct_repositories": len({row.get("repository") for row in registry.get("tasks", [])}),
        "tasks": entries,
        "faults": sorted(set(faults)),
        "source_disjoint_from": [
            "p2a-typing-qualifier-inheritance-001",
            "p2b-multipart-dynamic-read-wrapper-001",
            "p2c-click-optional-metavar-brackets-001"
        ],
        "exclusions": {
            "public_benchmarks": True,
            "training_use": True,
            "D1_use": True,
            "D2_use": True,
            "target_patch_candidate_visibility": True,
            "hidden_test_candidate_visibility": True,
            "user_task_or_label_dependency": True
        },
        "hosted_reference": {
            "cell": "Luna xhigh direct/integrated",
            "transport_state": "DEFINED_NOT_BOUND",
            "same_sealed_pool_required": True,
            "candidate_generation_opened": False,
            "local_denominators_remain_separate": True
        },
        "counters": {
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "candidate_model_calls": 0,
            "public_calibration_cases_consumed": 0,
            "D2_cases_consumed": 0
        },
        "maximum_inference": (
            "This seal proves only that ten licensed development parents fail and their exact "
            "upstream targets pass ten source-bound, network-free evaluators fixed before candidate "
            "generation. It is not model, subsystem, D1, D2, or ASI Stack support evidence."
        )
    }
    write_json(ROOT / "configs" / "theseus_p3_task_pool.json", pool)
    return pool


def audit_registry(registry: dict[str, Any]) -> list[str]:
    faults: list[str] = []
    if registry.get("policy") != "project_theseus_p3_online_source_selection_v1":
        faults.append("source_registry_policy_invalid")
    if registry.get("state") != "FIXED_BEFORE_CANDIDATE_GENERATION":
        faults.append("source_registry_not_fixed")
    tasks = registry.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 10:
        faults.append("source_registry_task_count_invalid")
        return faults
    indexes = [row.get("campaign_index") for row in tasks if isinstance(row, dict)]
    if indexes != list(range(1, 11)):
        faults.append("campaign_indexes_invalid")
    repositories = [row.get("repository") for row in tasks if isinstance(row, dict)]
    if len(set(repositories)) != 10:
        faults.append("repositories_not_distinct")
    stems = [row.get("stem") for row in tasks if isinstance(row, dict)]
    if len(set(stems)) != 10:
        faults.append("stems_not_distinct")
    if registry.get("boundaries", {}).get("candidate_generation_opened") is not False:
        faults.append("candidate_generation_already_opened")
    if sha256_file(INSTRUMENT) != EXPECTED_INSTRUMENT_SHA256:
        faults.append("instrument_digest_mismatch")
    if sha256_file(INSTRUMENT_AUDIT) != EXPECTED_INSTRUMENT_AUDIT_SHA256:
        faults.append("instrument_audit_digest_mismatch")
    return faults


def materialize_task(
    source: dict[str, Any], registry: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    faults: list[str] = []
    index = int(source["campaign_index"])
    stem = str(source["stem"])
    suffix = stem.removeprefix("p3_")
    parent = ROOT / "tests" / "fixtures" / "theseus_assistant_p3_online" / f"{stem}_parent.tar.gz"
    parent_upstream = parent.with_name(f"{stem}_parent_upstream.tar.gz")
    target = parent.with_name(f"{stem}_target.tar.gz")
    target_upstream = parent.with_name(f"{stem}_target_upstream.tar.gz")
    hidden = parent.with_name(f"{stem}_hidden_test.py")
    parent_sanitizer = ROOT / "reports" / f"theseus_{stem}_parent_archive_sanitization.json"
    target_sanitizer = ROOT / "reports" / f"theseus_{stem}_target_archive_sanitization.json"
    task_path = ROOT / "configs" / f"theseus_p3_task_{suffix}.json"
    evaluator_path = ROOT / "configs" / f"theseus_p3_evaluator_{suffix}.json"
    audit_path = ROOT / "reports" / f"theseus_p3_{suffix}_evaluator_audit.json"
    for path in (
        parent, parent_upstream, target, target_upstream, hidden,
        parent_sanitizer, target_sanitizer,
    ):
        if not path.is_file():
            faults.append(f"missing_source_artifact:{relative(path)}")
    parent_sanitization = read_json(parent_sanitizer)
    target_sanitization = read_json(target_sanitizer)
    faults.extend(audit_sanitization(parent_sanitization, parent, parent_upstream, source["source_root"], "parent"))
    faults.extend(audit_sanitization(target_sanitization, target, target_upstream, source["target_root"], "target"))
    faults.extend(audit_archive(parent, source["source_root"], source["license_path"], source["allowed_effect_paths"], "parent"))
    faults.extend(audit_archive(target, source["target_root"], source["license_path"], source["allowed_effect_paths"], "target"))
    task = {
        "policy": "project_theseus_p3_licensed_task_v1",
        "state": "SEALED_BEFORE_CANDIDATE_GENERATION",
        "campaign_index": index,
        "opaque_task_id": f"p3-licensed-maintenance-{index:02d}",
        "partition": "development_residual_campaign",
        "family": "bounded_python_correctness_repair",
        "natural_request": source["natural_request"],
        "source_archive": relative(parent),
        "source_archive_sha256": sha256_file(parent),
        "source_archive_root": source["source_root"],
        "source_provenance": {
            "repository": source["repository"],
            "url": f"https://github.com/{source['repository']}",
            "revision": source["parent_revision"],
            "retrieved_utc": registry["sealed_utc"],
            "license_spdx": source["license_spdx"],
            "license_path": source["license_path"],
            "upstream_request_url": f"https://github.com/{source['repository']}/pull/{source['pull_request']}",
            "upstream_request_title": source["pull_request_title"],
            "upstream_merged_utc": source["merged_utc"],
            "upstream_archive": relative(parent_upstream),
            "upstream_archive_sha256": sha256_file(parent_upstream),
            "archive_sanitization_report": relative(parent_sanitizer),
            "archive_sanitization_report_sha256": sha256_file(parent_sanitizer)
        },
        "contamination_screen": {
            "public_benchmark": False,
            "previous_theseus_surface": False,
            "source_disjoint_from_p2a_p2b_p2c": True,
            "task_selected_after_p3_instrument_freeze": True,
            "task_selected_before_candidate_generation": True,
            "later_patch_or_tests_candidate_visible": False,
            "development_task_eligible_for_training": False,
            "development_task_eligible_for_D1_or_D2": False,
            "memorization_risk": "public_recent_maintenance_change_not_claim_bearing"
        },
        "allowed_effect_paths": source["allowed_effect_paths"],
        "candidate_visible_context": {
            "searches": source["searches"],
            "reads": source["reads"],
            "maximum_total_characters": 9000
        },
        "visible_verifier": {
            "command": ["python3", "-m", "compileall", "-q", *source["allowed_effect_paths"]],
            "timeout_seconds": 30,
            "answer_specific": False
        },
        "effect_authority": "disposable_snapshot_only",
        "maximum_inference": (
            "This task contributes only one P3 development residual observation. It cannot support "
            "a subsystem, general model, D1, D2, training, or ASI Stack claim."
        )
    }
    write_json(task_path, task)
    evaluator = {
        "policy": "project_theseus_p3_route_blind_evaluator_v1",
        "state": "SEALED_BEFORE_CANDIDATE_GENERATION",
        "task_manifest": relative(task_path),
        "task_manifest_sha256": sha256_file(task_path),
        "baseline_must_fail": True,
        "baseline_failure_markers": source["baseline_failure_markers"],
        "hidden_test_files": [{
            "source": relative(hidden),
            "sha256": sha256_file(hidden),
            "destination": "theseus_p3_hidden_test.py"
        }],
        "target_archive": relative(target),
        "target_archive_sha256": sha256_file(target),
        "target_archive_root": source["target_root"],
        "target_provenance": {
            "revision": source["target_revision"],
            "merge_revision": source["merge_revision"],
            "upstream_archive": relative(target_upstream),
            "upstream_archive_sha256": sha256_file(target_upstream),
            "archive_sanitization_report": relative(target_sanitizer),
            "archive_sanitization_report_sha256": sha256_file(target_sanitizer)
        },
        "target_must_pass": True,
        "hidden_verifier": {
            "command": ["python3", "theseus_p3_hidden_test.py"],
            "timeout_seconds": 60,
            "network": "forbidden"
        },
        "blindness": {
            "candidate_generation_may_read_this_manifest": False,
            "route_label_passed_to_scoring": False,
            "later_patch_candidate_visible": False,
            "hidden_test_candidate_visible": False,
            "candidate_emitted_integrity_flags_trusted": False
        },
        "maximum_inference": (
            "A GREEN audit proves only that this exact parent fails and the exact upstream target "
            "passes this sealed evaluator. It does not establish model or subsystem competence."
        )
    }
    write_json(evaluator_path, evaluator)
    entry = {
        "campaign_index": index,
        "stem": stem,
        "repository": source["repository"],
        "pull_request_url": f"https://github.com/{source['repository']}/pull/{source['pull_request']}",
        "license_spdx": source["license_spdx"],
        "parent_revision": source["parent_revision"],
        "target_revision": source["target_revision"],
        "task": relative(task_path),
        "task_sha256": sha256_file(task_path),
        "evaluator": relative(evaluator_path),
        "evaluator_sha256": sha256_file(evaluator_path),
        "evaluator_audit": relative(audit_path)
    }
    return entry, faults


def audit_sanitization(
    report: dict[str, Any], output: Path, upstream: Path, root: str, label: str
) -> list[str]:
    faults: list[str] = []
    if report.get("trigger_state") != "GREEN":
        faults.append(f"{label}_sanitization_not_green")
    if report.get("omitted_members") != []:
        faults.append(f"{label}_sanitization_omitted_members")
    if report.get("source_archive_root") != root:
        faults.append(f"{label}_sanitization_root_mismatch")
    if report.get("input", {}).get("sha256") != sha256_file(upstream):
        faults.append(f"{label}_upstream_archive_digest_mismatch")
    if report.get("output", {}).get("sha256") != sha256_file(output):
        faults.append(f"{label}_normalized_archive_digest_mismatch")
    return faults


def audit_archive(
    archive: Path, root: str, license_path: str, effect_paths: list[str], label: str
) -> list[str]:
    if not archive.is_file():
        return [f"{label}_archive_missing"]
    faults: list[str] = []
    wanted = {f"{root}/{license_path}", *(f"{root}/{path}" for path in effect_paths)}
    try:
        with tarfile.open(archive) as handle:
            members = handle.getmembers()
            names = {member.name.rstrip("/") for member in members}
            if any(member.issym() or member.islnk() or not (member.isdir() or member.isfile()) for member in members):
                faults.append(f"{label}_archive_has_nonregular_member")
            if any(not (name == root or name.startswith(root + "/")) for name in names):
                faults.append(f"{label}_archive_member_outside_root")
            for path in wanted:
                if path not in names:
                    faults.append(f"{label}_archive_required_path_missing:{path}")
    except tarfile.TarError:
        faults.append(f"{label}_archive_invalid")
    return faults


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
