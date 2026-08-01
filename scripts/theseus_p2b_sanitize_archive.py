#!/usr/bin/env python3
"""Create a deterministic regular-file-only derivative of a source archive."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


POLICY = "project_theseus_p2b_source_archive_sanitizer_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = sanitize(Path(args.input), Path(args.output))
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def sanitize(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    admitted: list[str] = []
    omitted: list[dict[str, str]] = []
    roots: set[str] = set()
    with tarfile.open(source, "r:*") as incoming:
        with destination.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as outgoing:
                    for member in incoming.getmembers():
                        roots.add(member.name.strip("/").split("/", 1)[0])
                        if not safe_member_name(member.name):
                            omitted.append({
                                "path": member.name,
                                "kind": "unsafe_member_path",
                                "linkname": member.linkname,
                            })
                            continue
                        if member.issym() or member.islnk():
                            omitted.append({
                                "path": member.name,
                                "kind": "symbolic_link" if member.issym() else "hard_link",
                                "linkname": member.linkname,
                            })
                            continue
                        if not member.isfile() and not member.isdir():
                            omitted.append({"path": member.name, "kind": "special_member", "linkname": ""})
                            continue
                        normalized = copy.copy(member)
                        normalized.uid = 0
                        normalized.gid = 0
                        normalized.uname = ""
                        normalized.gname = ""
                        normalized.mtime = 0
                        if member.isdir():
                            outgoing.addfile(normalized)
                        else:
                            payload = incoming.extractfile(member)
                            if payload is None:
                                raise ValueError(f"unreadable archive member: {member.name}")
                            outgoing.addfile(normalized, payload)
                        admitted.append(member.name)
    faults: list[str] = []
    if len(roots) != 1:
        faults.append("source_archive_root_count_invalid")
    if not admitted:
        faults.append("source_archive_no_admitted_members")
    if any(row["kind"] == "unsafe_member_path" for row in omitted):
        faults.append("source_archive_unsafe_member_path")
    return {
        "policy": POLICY,
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": faults,
        "input": {"path": str(source), "sha256": sha256_file(source)},
        "output": {"path": str(destination), "sha256": sha256_file(destination)},
        "source_archive_root": next(iter(roots)) if len(roots) == 1 else "",
        "admitted_regular_file_or_directory_count": len(admitted),
        "omitted_members": omitted,
        "normalization": {
            "allowed_types": ["regular_file", "directory"],
            "uid_gid": 0,
            "user_group_names": "empty",
            "mtime": 0,
            "gzip_mtime": 0,
            "member_order": "upstream_archive_order",
        },
        "maximum_inference": "Archive transport normalization only; no task, model, or subsystem claim.",
    }


def safe_member_name(value: str) -> bool:
    if not value or "\x00" in value or "\\" in value or value.startswith("/"):
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and all(
        part not in {"", ".", ".."} for part in path.parts
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
