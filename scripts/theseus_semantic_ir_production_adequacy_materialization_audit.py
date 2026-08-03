#!/usr/bin/env python3
"""Independently audit the sealed S3 source archive set."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_materialization_v3.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_materialization_v3_audit.json"
EXPECTED_REPORT_SHA256 = "2aaf9eb8fe85256d45786f9def49828233956a239e7f998f0b9f4aaf0f7bfa24"
POLICY = "project_theseus_semantic_ir_production_adequacy_materialization_audit_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=relative(DEFAULT_REPORT))
    parser.add_argument("--out", default=relative(DEFAULT_OUT))
    args = parser.parse_args()
    result = audit(resolve(args.report))
    write_json(resolve(args.out), result)
    print(json.dumps(summary(result), indent=2, sort_keys=True))
    return 0 if result["trigger_state"] == "GREEN" else 2


def audit(report_path: Path = DEFAULT_REPORT) -> dict[str, Any]:
    faults: list[str] = []
    if not report_path.is_file() or sha256_file(report_path) != EXPECTED_REPORT_SHA256:
        faults.append("materialization_report_binding_invalid")
        report: dict[str, Any] = {}
    else:
        report = read_json(report_path)
    if report.get("trigger_state") != "GREEN" or report.get("archive_set_admitted") is not True:
        faults.append("materialization_not_green")
    if report.get("faults") != [] or report.get("stage") != "source_materialization_complete":
        faults.append("materialization_state_invalid")
    counters = mapping(report.get("counters"))
    expected_activity = {"network_source_calls": 78, "source_archives_materialized": 36}
    for key, value in counters.items():
        expected = expected_activity.get(key, 0)
        if integer(value) != expected:
            faults.append(f"counter_invalid:{key}")
    rows = dictionaries(report.get("rows"))
    if len(rows) != 18 or [integer(row.get("index")) for row in rows] != list(range(1, 19)):
        faults.append("row_identity_invalid")
    archive_paths: list[str] = []
    member_count = 0
    source_difference_count = 0
    for row in rows:
        archives = mapping(row.get("archives"))
        if set(archives) != {"parent", "target"} or row.get("faults") != []:
            faults.append(f"task_{row.get('index')}:archive_pair_invalid")
            continue
        hashes: dict[str, dict[str, str]] = {}
        for role in ("parent", "target"):
            receipt = mapping(archives.get(role))
            path = resolve(str(receipt.get("path") or ""))
            row_faults, member_hashes = audit_archive(path, receipt)
            faults.extend(f"task_{row.get('index')}:{role}:{fault}" for fault in row_faults)
            hashes[role] = member_hashes
            archive_paths.append(relative(path))
            member_count += len(member_hashes)
        if any(
            hashes["parent"].get(str(path)) != hashes["target"].get(str(path))
            for path in row.get("selected_source_paths") or []
        ):
            source_difference_count += 1
        else:
            faults.append(f"task_{row.get('index')}:selected_source_bytes_unchanged")
    expected_paths = sorted(
        relative(path)
        for path in (ROOT / "tests" / "fixtures" / "theseus_semantic_ir_production_adequacy").glob("*.tar.gz")
    )
    if len(archive_paths) != 36 or len(set(archive_paths)) != 36:
        faults.append("archive_path_cardinality_invalid")
    if sorted(archive_paths) != expected_paths:
        faults.append("archive_directory_membership_invalid")
    if member_count != 76:
        faults.append("member_receipt_count_invalid")
    if source_difference_count != 18:
        faults.append("source_difference_count_invalid")
    return {
        "policy": POLICY,
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "materialization_report": artifact(report_path),
        "archive_receipt_count": len(archive_paths),
        "member_receipt_count": member_count,
        "selected_source_difference_count": source_difference_count,
        "counters": counters,
        "maximum_inference": "A GREEN audit establishes only content-bound, normalized, complete parent/target source archives with differing selected source bytes. It does not establish parent failure, target success, evaluator adequacy, mechanics competence, a subsystem effect, D1, D2, or book support.",
    }


def audit_archive(path: Path, receipt: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    faults: list[str] = []
    hashes: dict[str, str] = {}
    if not path.is_file() or sha256_file(path) != str(receipt.get("sha256") or ""):
        return ["archive_hash_invalid"], hashes
    expected = {
        str(row.get("path")): (str(row.get("sha256")), integer(row.get("bytes")))
        for row in dictionaries(receipt.get("members"))
    }
    root = str(receipt.get("root") or "")
    try:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                pure = PurePosixPath(member.name)
                if (
                    not member.isfile()
                    or member.issym()
                    or member.islnk()
                    or pure.is_absolute()
                    or ".." in pure.parts
                    or not member.name.startswith(root + "/")
                ):
                    faults.append("archive_member_unsafe")
                    continue
                if (member.mtime, member.uid, member.gid, member.mode) != (0, 0, 0, 0o644):
                    faults.append("archive_member_not_normalized")
                logical = member.name[len(root) + 1 :]
                handle = archive.extractfile(member)
                content = handle.read() if handle is not None else b""
                hashes[logical] = hashlib.sha256(content).hexdigest()
                if expected.get(logical) != (hashes[logical], len(content)):
                    faults.append("member_receipt_mismatch")
    except (OSError, tarfile.TarError):
        faults.append("archive_unreadable")
    if set(hashes) != set(expected):
        faults.append("member_set_mismatch")
    return sorted(set(faults)), hashes


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in (
        "trigger_state", "faults", "archive_receipt_count",
        "member_receipt_count", "selected_source_difference_count", "counters"
    )}


def dictionaries(value: Any) -> list[dict[str, Any]]:
    return [row for row in value or [] if isinstance(row, dict)]


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def integer(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def artifact(path: Path) -> dict[str, str]:
    return {"path": relative(path), "sha256": sha256_file(path) if path.is_file() else ""}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
