#!/usr/bin/env python3
"""Audit selected VCM source/verifier bytes against the English-only seed scope."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402

POLICY = "project_theseus_vcm_content_language_audit_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_content_language_audit.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    path = p2a.resolve(args.config)
    report = audit(path)
    p2a.write_json(p2a.resolve(args.out or p2a.read_json(path)["report"]), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def audit(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = p2a.read_json(config_path)
    faults = []
    if config.get("policy") != POLICY: faults.append("policy_invalid")
    panel_path = p2a.resolve(config.get("source_panel", ""))
    if not panel_path.is_file() or p2a.sha256_file(panel_path) != config.get("source_panel_sha256"):
        faults.append("source_panel_binding_invalid"); panel = {}
    else: panel = p2a.read_json(panel_path)
    authority = p2a.mapping(config.get("authority"))
    if authority.get("selected_archive_static_text_read_authorized") is not True or any(value is not False for key,value in authority.items() if key != "selected_archive_static_text_read_authorized"):
        faults.append("authority_boundary_invalid")
    ranges = [(row["name"], int(row["start"],16), int(row["end"],16)) for row in config.get("forbidden_unicode_scripts", [])]
    binary = {str(value).lower() for value in config.get("binary_extensions", [])}
    task_rows = []
    scanned_identities = set()
    skipped_binary = 0
    for task in p2a.dicts(panel.get("assembled_rows")):
        violations = []
        for role, archive in p2a.mapping(task.get("archives")).items():
            archive = p2a.mapping(archive)
            path = p2a.resolve(str(archive.get("path") or ""))
            if not path.is_file() or p2a.sha256_file(path) != archive.get("sha256"):
                faults.append(f"archive_binding_invalid:{task.get('index')}:{role}"); continue
            root = str(archive.get("root") or "")
            with tarfile.open(path, "r:gz") as handle:
                for member in handle.getmembers():
                    if not member.isfile(): continue
                    relative = member.name[len(root)+1:] if member.name.startswith(root+"/") else member.name
                    if relative.lower().startswith(("license", "copying")): continue
                    identity = (task.get("index"), relative, member.size, role.split("_",1)[0])
                    if identity in scanned_identities: continue
                    scanned_identities.add(identity)
                    if Path(relative).suffix.lower() in binary:
                        skipped_binary += 1; continue
                    extracted = handle.extractfile(member); payload = extracted.read() if extracted else b""
                    try: text = payload.decode("utf-8")
                    except UnicodeDecodeError:
                        skipped_binary += 1; continue
                    hits = {}
                    for name, start, end in ranges:
                        chars = [char for char in text if start <= ord(char) <= end]
                        if chars: hits[name] = {"count": len(chars), "sample_codepoints": sorted({f"U+{ord(char):04X}" for char in chars})[:12]}
                    if hits:
                        violations.append({"role":role, "path":relative, "sha256":hashlib.sha256(payload).hexdigest(), "scripts":hits})
        task_rows.append({"index":task.get("index"), "repository":task.get("repository"), "content_language_scope_passed":not violations, "violations":violations})
    violation_indices = [int(row["index"]) for row in task_rows if row["violations"]]
    expected = [int(value) for value in config.get("expected_violation_indices", [])]
    if violation_indices != expected: faults.append("expected_violation_set_mismatch")
    content_faults = [f"non_english_selected_content:{index}" for index in violation_indices]
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "RED" if faults or content_faults else "GREEN",
        "state": "SOURCE_PANEL_CONTENT_LANGUAGE_SCOPE_VIOLATION_REPLACEMENT_REQUIRED" if content_faults else "SOURCE_PANEL_CONTENT_LANGUAGE_SCOPE_GREEN",
        "faults": sorted(set(faults + content_faults)),
        "source_panel_previously_admitted": panel.get("source_panel_admitted") is True,
        "source_panel_admission_remains_valid": not content_faults and not faults,
        "task_count": len(task_rows),
        "violation_task_count": len(violation_indices),
        "violation_indices": violation_indices,
        "selected_text_identities_scanned": len(scanned_identities)-skipped_binary,
        "binary_identities_skipped": skipped_binary,
        "tasks": task_rows,
        "replacement_required_before_evaluator_execution": bool(content_faults),
        "parent_target_or_evaluator_executions": 0,
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
        "maximum_inference": config.get("maximum_inference"),
    }


def summary(r: dict[str, Any]) -> dict[str, Any]: return {k:r.get(k) for k in ("trigger_state","state","source_panel_admission_remains_valid","task_count","violation_task_count","violation_indices","selected_text_identities_scanned","binary_identities_skipped","parent_target_or_evaluator_executions","candidate_or_control_calls","external_reference_calls","faults")}

if __name__ == "__main__": raise SystemExit(main())
