#!/usr/bin/env python3
"""Independently assemble and audit the frozen VCM source panel.

This owner is deliberately execution-free.  It verifies receipts and policy
boundaries, substitutes only the sealed Task 28 replacement, and refuses panel
admission when the natural-language request is outside the English-only seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
import unicodedata
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import theseus_assistant_p2a as p2a


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_source_panel_audit.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_vcm_source_panel_audit.json"
POLICY = "project_theseus_vcm_source_panel_audit_v1"
ROLES = {"parent_source", "target_source", "parent_verifier", "target_verifier"}
IDENTITY_FIELDS = (
    "opaque_source_id",
    "repository",
    "pull_request",
    "panel",
    "query_language",
    "base_revision",
    "head_revision",
    "license_spdx",
)
ZERO_ACTIVITY_KEYS = (
    "parent_target_or_evaluator_executions",
    "candidate_or_control_calls",
    "local_model_calls",
    "external_inference_calls",
    "teacher_calls",
    "training_rows_written",
    "D1_cases_consumed",
    "D2_cases_consumed",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default=p2a.rel(DEFAULT_OUT))
    args = parser.parse_args()
    report = audit(p2a.resolve(args.config))
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def audit(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = p2a.read_json(config_path)
    faults: list[str] = []
    if config.get("policy") != POLICY:
        faults.append("policy_invalid")
    reports = p2a.mapping(config.get("reports"))
    loaded: dict[str, dict[str, Any]] = {}
    for name, binding_value in reports.items():
        binding = p2a.mapping(binding_value)
        path = p2a.resolve(str(binding.get("path") or ""))
        if not path.is_file() or p2a.sha256_file(path) != str(binding.get("sha256") or ""):
            faults.append(f"report_binding_invalid:{name}")
            loaded[name] = {}
        else:
            loaded[name] = p2a.read_json(path)

    selection = loaded.get("metadata_selection", {})
    materialization = loaded.get("failed_materialization", {})
    replacement = loaded.get("replacement_28", {})
    selected = p2a.dicts(selection.get("selected_source_identities"))
    original_rows = p2a.dicts(materialization.get("rows"))
    replacement_row = p2a.mapping(replacement.get("replacement_materialization"))

    validate_terminal_reports(selection, materialization, replacement, faults)
    authority = p2a.mapping(config.get("authority"))
    if not authority or any(value is not False for value in authority.values()):
        faults.append("audit_authority_boundary_invalid")
    if len(selected) != 62 or len(original_rows) != 62:
        faults.append("source_row_cardinality_invalid")
    if [integer(row.get("index")) for row in original_rows] != list(range(1, 63)):
        faults.append("source_row_index_sequence_invalid")

    rows: list[dict[str, Any]] = []
    for index in range(1, 63):
        original = original_rows[index - 1] if len(original_rows) >= index else {}
        metadata = selected[index - 1] if len(selected) >= index else {}
        for field in IDENTITY_FIELDS:
            if original.get(field) != metadata.get(field):
                faults.append(f"task_{index:02d}:metadata_identity_mismatch:{field}")
        if index == 28:
            if original.get("faults") != ["selected_verifier_bytes_unchanged"]:
                faults.append("task_28:original_failure_not_preserved")
            row = replacement_row
        else:
            if original.get("faults") != []:
                faults.append(f"task_{index:02d}:unexpected_materialization_fault")
            row = original
        if integer(row.get("index")) != index:
            faults.append(f"task_{index:02d}:assembled_index_invalid")
        title = str(row.get("natural_language_request") or "")
        if hashlib.sha256(title.encode()).hexdigest() != row.get("natural_language_request_sha256"):
            faults.append(f"task_{index:02d}:natural_language_request_hash_invalid")
        rows.append(row)

    replacement_metadata = p2a.mapping(replacement.get("replacement_metadata"))
    for field in IDENTITY_FIELDS:
        if replacement_row.get(field) != replacement_metadata.get(field):
            faults.append(f"task_28:replacement_metadata_mismatch:{field}")

    if len(selected) >= 28:
        failed_repository = str(selected[27].get("repository") or "")
        replacement_repository = str(replacement_row.get("repository") or "")
        selected_repositories = {str(row.get("repository") or "") for row in selected}
        if not replacement_repository or replacement_repository in selected_repositories:
            faults.append("task_28:replacement_not_disjoint_from_frozen_panel")
        prior = tracked_prior_repositories(config_path)
        prior_receipt = p2a.mapping(selection.get("prior_repository_denylist"))
        if integer(prior_receipt.get("count")) != len(prior) or str(prior_receipt.get("sha256") or "") != stable_list_hash(prior):
            faults.append("prior_repository_denylist_recomputation_invalid")
        if any(repository in set(prior) for repository in selected_repositories):
            faults.append("frozen_panel_prior_repository_overlap")
        if replacement_repository in prior:
            faults.append("task_28:replacement_not_disjoint_from_prior_sources")
        if replacement_repository == failed_repository:
            faults.append("task_28:failed_repository_reused")
    else:
        prior = []

    expected = p2a.mapping(config.get("expected_panel"))
    archive_paths: list[str] = []
    member_receipts = 0
    source_difference_count = 0
    verifier_difference_count = 0
    total_member_bytes = 0
    for row in rows:
        index = integer(row.get("index"))
        archives = p2a.mapping(row.get("archives"))
        if set(archives) != ROLES or row.get("faults") != []:
            faults.append(f"task_{index:02d}:archive_set_invalid")
            continue
        hashes: dict[str, dict[str, str]] = {}
        for role in sorted(ROLES):
            receipt = p2a.mapping(archives.get(role))
            path = p2a.resolve(str(receipt.get("path") or ""))
            expected_root = f"vcm-claim-{index:02d}"
            expected_path = p2a.resolve(str(expected.get("output_directory") or "")) / f"{expected_root}-{role}.tar.gz"
            if receipt.get("root") != expected_root or path != expected_path:
                faults.append(f"task_{index:02d}:{role}:archive_identity_invalid")
            archive_faults, member_hashes, member_bytes = audit_archive(path, receipt)
            faults.extend(f"task_{index:02d}:{role}:{fault}" for fault in archive_faults)
            hashes[role] = member_hashes
            archive_paths.append(p2a.rel(path))
            member_receipts += len(member_hashes)
            total_member_bytes += member_bytes
            license_path = str(
                (row.get("parent_license_path") if role.startswith("parent_") else row.get("target_license_path"))
                or ""
            )
            if license_path not in member_hashes:
                faults.append(f"task_{index:02d}:{role}:license_member_missing")
        if selected_paths_changed(row, hashes, "source"):
            source_difference_count += 1
        else:
            faults.append(f"task_{index:02d}:selected_source_bytes_unchanged")
        if selected_paths_changed(row, hashes, "verifier"):
            verifier_difference_count += 1
        else:
            faults.append(f"task_{index:02d}:selected_verifier_bytes_unchanged")

    if integer(expected.get("task_count")) != 62:
        faults.append("expected_task_count_policy_invalid")
    expected_archive_paths = sorted(
        p2a.rel(path)
        for path in p2a.resolve(str(expected.get("output_directory") or "")).glob("*.tar.gz")
    )
    if len(archive_paths) != integer(expected.get("archive_count")) or len(set(archive_paths)) != len(archive_paths):
        faults.append("archive_path_cardinality_invalid")
    if sorted(archive_paths) != expected_archive_paths:
        faults.append("archive_directory_membership_invalid")
    if member_receipts != integer(expected.get("member_receipt_count")):
        faults.append("member_receipt_count_invalid")
    if total_member_bytes != integer(expected.get("total_member_bytes")):
        faults.append("total_member_bytes_invalid")
    if source_difference_count != 62:
        faults.append("source_difference_count_invalid")
    if verifier_difference_count != 62:
        faults.append("verifier_difference_count_invalid")

    repositories = [str(row.get("repository") or "") for row in rows]
    if len(set(repositories)) != 62 or any(not repository for repository in repositories):
        faults.append("assembled_repository_uniqueness_invalid")
    actual_quotas = Counter((str(row.get("panel") or ""), str(row.get("query_language") or "")) for row in rows)
    expected_quotas = {
        (str(row.get("panel") or ""), str(row.get("query_language") or "")): integer(row.get("count"))
        for row in p2a.dicts(expected.get("quotas"))
    }
    if dict(actual_quotas) != expected_quotas:
        faults.append("panel_language_quotas_invalid")

    title_items = [
        {"index": integer(row.get("index")), "title_sha256": str(row.get("natural_language_request_sha256") or "")}
        for row in rows
    ]
    title_set_sha256 = stable_hash(title_items)
    language = p2a.mapping(config.get("english_language_review"))
    if language.get("all_62_titles_independently_reviewed") is not True:
        faults.append("english_review_completeness_invalid")
    if title_set_sha256 != str(language.get("reviewed_title_set_sha256") or ""):
        faults.append("english_review_title_set_binding_invalid")
    non_english = language_adjudications(rows, language, faults)
    faults.extend(f"task_{index:02d}:natural_language_out_of_scope" for index in non_english)

    for source_name, report in (("materialization", materialization), ("replacement", replacement)):
        counters = p2a.mapping(report.get("counters"))
        for key in ZERO_ACTIVITY_KEYS:
            if integer(counters.get(key)) != 0:
                faults.append(f"{source_name}_forbidden_activity:{key}")

    integrity_faults = [fault for fault in faults if "natural_language_out_of_scope" not in fault]
    archive_integrity_green = not integrity_faults
    admitted = not faults
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if admitted else "RED",
        "state": "SIXTY_TWO_TASK_SOURCE_PANEL_ADMITTED" if admitted else (
            "SOURCE_PANEL_LANGUAGE_REPLACEMENTS_REQUIRED" if archive_integrity_green and non_english else "SOURCE_PANEL_AUDIT_FAILED"
        ),
        "faults": sorted(set(faults)),
        "source_panel_admitted": admitted,
        "archive_integrity_green": archive_integrity_green,
        "task_28_replacement_bound": replacement_row.get("repository") in repositories,
        "assembled_task_count": len(rows),
        "unique_repository_count": len(set(repositories)),
        "archive_receipt_count": len(archive_paths),
        "member_receipt_count": member_receipts,
        "total_member_bytes": total_member_bytes,
        "selected_source_difference_count": source_difference_count,
        "selected_verifier_difference_count": verifier_difference_count,
        "panel_language_quotas": [
            {"panel": panel, "query_language": language_name, "count": count}
            for (panel, language_name), count in sorted(actual_quotas.items())
        ],
        "prior_repository_count_recomputed": len(prior),
        "title_set_sha256": title_set_sha256,
        "english_eligible_task_count": len(rows) - len(non_english),
        "replacement_slots_required": [
            {
                "index": index,
                "panel": rows[index - 1].get("panel"),
                "query_language": rows[index - 1].get("query_language"),
                "title_sha256": rows[index - 1].get("natural_language_request_sha256"),
            }
            for index in non_english
        ],
        "candidate_packet_materialization_opened": False,
        "parent_target_or_evaluator_executions": 0,
        "local_model_calls": 0,
        "external_reference_calls": 0,
        "D1_cases_consumed": 0,
        "D2_cases_consumed": 0,
        "config": artifact(config_path),
        "reports": {name: artifact(p2a.resolve(str(p2a.mapping(value).get("path") or ""))) for name, value in reports.items()},
        "maximum_inference": config.get("maximum_inference"),
    }


def validate_terminal_reports(selection: dict[str, Any], materialization: dict[str, Any], replacement: dict[str, Any], faults: list[str]) -> None:
    if selection.get("trigger_state") != "GREEN" or selection.get("state") != "SIXTY_TWO_SOURCE_IDENTITIES_FROZEN_BEFORE_CONTENT_RETRIEVAL":
        faults.append("metadata_selection_terminal_state_invalid")
    if materialization.get("trigger_state") != "RED" or materialization.get("state") != "SOURCE_MATERIALIZATION_INCOMPLETE" or materialization.get("faults") != ["task_28:selected_verifier_bytes_unchanged"] or materialization.get("archive_set_admitted") is not False or materialization.get("partial_archive_set_admitted") is not False:
        faults.append("failed_materialization_terminal_state_invalid")
    if replacement.get("trigger_state") != "GREEN" or replacement.get("state") != "TASK_28_PYTHON_CLAIM_REPLACEMENT_SOURCE_BOUND" or replacement.get("replacement_admitted") is not True or replacement.get("checkpoint_artifact_hash_verified_final") is not True:
        faults.append("replacement_28_terminal_state_invalid")
    checkpoint = p2a.mapping(replacement.get("checkpoint"))
    path = p2a.resolve(str(checkpoint.get("path") or ""))
    if not path.is_file() or p2a.sha256_file(path) != str(checkpoint.get("sha256") or ""):
        faults.append("replacement_28_final_checkpoint_binding_invalid")


def audit_archive(path: Path, receipt: dict[str, Any]) -> tuple[list[str], dict[str, str], int]:
    faults: list[str] = []
    hashes: dict[str, str] = {}
    total_bytes = 0
    if not path.is_file() or p2a.sha256_file(path) != str(receipt.get("sha256") or ""):
        return ["archive_hash_invalid"], hashes, total_bytes
    expected = {
        str(row.get("path") or ""): (str(row.get("sha256") or ""), integer(row.get("bytes")))
        for row in p2a.dicts(receipt.get("members"))
    }
    root = str(receipt.get("root") or "")
    try:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                pure = PurePosixPath(member.name)
                if not member.isfile() or member.issym() or member.islnk() or pure.is_absolute() or ".." in pure.parts or not member.name.startswith(root + "/"):
                    faults.append("archive_member_unsafe")
                    continue
                if (member.mtime, member.uid, member.gid, member.mode) != (0, 0, 0, 0o644):
                    faults.append("archive_member_not_normalized")
                logical = member.name[len(root) + 1 :]
                handle = archive.extractfile(member)
                content = handle.read() if handle is not None else b""
                digest = hashlib.sha256(content).hexdigest()
                hashes[logical] = digest
                total_bytes += len(content)
                if expected.get(logical) != (digest, len(content)):
                    faults.append("member_receipt_mismatch")
    except (OSError, tarfile.TarError):
        faults.append("archive_unreadable")
    if set(hashes) != set(expected):
        faults.append("member_set_mismatch")
    return sorted(set(faults)), hashes, total_bytes


def selected_paths_changed(row: dict[str, Any], hashes: dict[str, dict[str, str]], kind: str) -> bool:
    return any(
        hashes.get(f"parent_{kind}", {}).get(path) != hashes.get(f"target_{kind}", {}).get(path)
        for path in p2a.strings(row.get(f"selected_{kind}_paths"))
    )


def language_adjudications(rows: list[dict[str, Any]], policy: dict[str, Any], faults: list[str]) -> list[int]:
    decisions = p2a.dicts(policy.get("non_english_title_adjudications"))
    indices: list[int] = []
    for decision in decisions:
        index = integer(decision.get("index"))
        if index < 1 or index > len(rows):
            faults.append("english_review_index_invalid")
            continue
        row = rows[index - 1]
        if row.get("natural_language_request_sha256") != decision.get("title_sha256") or decision.get("english_eligible") is not False:
            faults.append(f"task_{index:02d}:english_review_binding_invalid")
        if not deterministic_non_english_signal(str(row.get("natural_language_request") or "")):
            faults.append(f"task_{index:02d}:non_english_signal_not_reproducible")
        indices.append(index)
    expected = sorted(integer(value) for value in policy.get("replacement_slots_required") or [])
    if sorted(indices) != expected or len(set(indices)) != len(indices):
        faults.append("english_review_replacement_set_invalid")
    return sorted(set(indices))


def deterministic_non_english_signal(title: str) -> bool:
    lowered = f" {title.casefold()} "
    markers = (
        " für ", " absichern ", " testeingaben ", " ausserhalb ",
        " lücke ", " schließen ", " und verankern ", " locație de start ",
        " tenant-namen in der seitenleiste anzeigen ",
    )
    if any(marker in lowered for marker in markers):
        return True
    return any(
        "CYRILLIC" in unicodedata.name(character, "") or "HANGUL" in unicodedata.name(character, "")
        for character in title
    )


def tracked_prior_repositories(config_path: Path) -> list[str]:
    completed = subprocess.run(["git", "ls-files", "configs/*.json"], cwd=ROOT, check=True, capture_output=True, text=True)
    repositories: set[str] = set()
    for name in completed.stdout.splitlines():
        path = ROOT / name
        if path.resolve() == config_path.resolve():
            continue
        try:
            collect_repositories(json.loads(path.read_text(encoding="utf-8")), repositories)
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(repositories)


def collect_repositories(value: Any, repositories: set[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "repository" and isinstance(child, str) and child.count("/") == 1 and not any(character.isspace() for character in child):
                repositories.add(child)
            collect_repositories(child, repositories)
    elif isinstance(value, list):
        for child in value:
            collect_repositories(child, repositories)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stable_list_hash(values: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(values), separators=(",", ":")).encode()).hexdigest()


def artifact(path: Path) -> dict[str, str]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path) if path.is_file() else ""}


def integer(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in (
        "trigger_state", "state", "source_panel_admitted", "archive_integrity_green",
        "assembled_task_count", "unique_repository_count", "archive_receipt_count",
        "member_receipt_count", "english_eligible_task_count", "replacement_slots_required",
        "faults", "local_model_calls", "external_reference_calls",
    )}


if __name__ == "__main__":
    raise SystemExit(main())
