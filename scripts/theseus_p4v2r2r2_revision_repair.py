#!/usr/bin/env python3
"""Acquire corrected effect-commit pairs while retaining the invalid originals."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_p4v2r2r2_fetch_sources as fetch  # noqa: E402


REGISTRY = ROOT / "configs" / "theseus_p4v2r2r2_task_sources.json"
ORIGINAL_FETCH = ROOT / "reports" / "theseus_p4v2r2r2_source_fetch.json"
CORRECTIONS = ROOT / "configs" / "theseus_p4v2r2r2_revision_corrections.json"
FIXTURES = ROOT / "tests" / "fixtures" / "theseus_p4v2r2r2_online"
REPORT = ROOT / "reports" / "theseus_p4v2r2r2_revision_repair_fetch.json"
EXPECTED_REGISTRY_SHA256 = "43ceb81b7790f07d9b60d208c947320fddc4deb511be62a781341360642ac99c"
EXPECTED_ORIGINAL_FETCH_SHA256 = "150a449dd11a32442b9d352b463411065c3d55e1cbda874e86c9fe1f5356d12c"


def audit_corrections() -> dict[str, Any]:
    value = fetch.read_json(CORRECTIONS)
    faults: list[str] = []
    if value.get("policy") != "project_theseus_p4v2r2r2_revision_corrections_v1":
        faults.append("policy_invalid")
    if value.get("state") != "SEALED_AFTER_INDEPENDENT_EVALUATOR_FOUND_NONFAILING_REGISTERED_PARENTS_BEFORE_CANDIDATE_OR_CONTROL_GENERATION":
        faults.append("state_invalid")
    if (
        value.get("source_registry_sha256") != EXPECTED_REGISTRY_SHA256
        or fetch.sha256_file(REGISTRY) != EXPECTED_REGISTRY_SHA256
    ):
        faults.append("registry_binding_invalid")
    if (
        value.get("original_source_fetch_sha256") != EXPECTED_ORIGINAL_FETCH_SHA256
        or fetch.sha256_file(ORIGINAL_FETCH) != EXPECTED_ORIGINAL_FETCH_SHA256
    ):
        faults.append("original_fetch_binding_invalid")
    registry = fetch.read_json(REGISTRY)
    by_stem = {str(row["stem"]): row for row in fetch.dictionaries(registry.get("tasks"))}
    rows = fetch.dictionaries(value.get("corrections"))
    expected_stems = {
        "p4v2r2r2_02_h2_1313",
        "p4v2r2r2_03_pygments_3215",
        "p4v2r2r2_10_xarray_11401",
    }
    if {str(row.get("stem")) for row in rows} != expected_stems:
        faults.append("correction_scope_invalid")
    for row in rows:
        stem = str(row.get("stem") or "")
        source = by_stem.get(stem, {})
        if (
            row.get("repository") != source.get("repository")
            or row.get("registered_parent_revision") != source.get("parent_revision")
            or row.get("registered_target_revision") != source.get("target_revision")
        ):
            faults.append(f"registered_identity_mismatch:{stem}")
        revisions = [
            str(row.get("corrected_parent_revision") or ""),
            str(row.get("corrected_target_revision") or ""),
        ]
        if any(len(revision) != 40 for revision in revisions) or revisions[0] == revisions[1]:
            faults.append(f"corrected_identity_invalid:{stem}")
        if not fetch.p2a_strings(row.get("effect_commit_files")):
            faults.append(f"effect_commit_files_empty:{stem}")
    discovery = value.get("discovery") if isinstance(value.get("discovery"), dict) else {}
    if int(discovery.get("process_executions") or 0) != 24 or int(discovery.get("candidate_or_control_calls") or 0) != 0:
        faults.append("discovery_custody_invalid")
    invariants = value.get("invariants") if isinstance(value.get("invariants"), dict) else {}
    for key in ("candidate_or_control_calls", "D1_cases_consumed", "D2_cases_consumed", "local_or_hosted_model_calls", "training_rows_written"):
        if int(invariants.get(key) or 0) != 0:
            faults.append(f"counter_nonzero:{key}")
    for key in ("allowed_effect_paths_changed", "natural_request_changed", "repository_changed", "task_membership_changed", "user_gate"):
        if invariants.get(key) is not False:
            faults.append(f"invariant_invalid:{key}")
    if invariants.get("revision_identity_repaired") is not True or invariants.get("original_artifacts_and_fetch_receipt_retained") is not True:
        faults.append("repair_retention_invariant_invalid")
    if invariants.get("project_selected_quality_token_cap") is not None:
        faults.append("quality_token_cap_present")
    return {
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "path": fetch.relative(CORRECTIONS),
        "sha256": fetch.sha256_file(CORRECTIONS),
        "correction_count": len(rows),
    }


def _effect_payload(path: Path, root: str, effect_path: str) -> bytes:
    with tarfile.open(path, "r:gz") as handle:
        stream = handle.extractfile(f"{root}/{effect_path}")
        if stream is None:
            raise ValueError("effect member unreadable")
        return stream.read()


def acquire(*, fetch_network: bool) -> dict[str, Any]:
    correction_audit = audit_corrections()
    faults = list(correction_audit["faults"])
    registry = fetch.read_json(REGISTRY)
    tasks = {str(row["stem"]): row for row in fetch.dictionaries(registry.get("tasks"))}
    correction_value = fetch.read_json(CORRECTIONS)
    task_rows: list[dict[str, Any]] = []
    for correction in fetch.dictionaries(correction_value.get("corrections")):
        stem = str(correction["stem"])
        source = tasks[stem]
        artifacts: list[dict[str, Any]] = []
        for label in ("parent", "target"):
            revision = str(correction[f"corrected_{label}_revision"])
            root = str(correction[f"corrected_{label}_root"])
            upstream = FIXTURES / f"{stem}_{label}_revision_corrected_upstream.tar.gz"
            projected = FIXTURES / f"{stem}_{label}_revision_corrected.tar.gz"
            sanitization_path = ROOT / "reports" / f"theseus_{stem}_{label}_revision_corrected_archive_sanitization.json"
            url = f"https://codeload.github.com/{source['repository']}/tar.gz/{revision}"
            if not upstream.is_file() and fetch_network:
                fetch.download(url, upstream)
            if not upstream.is_file():
                faults.append(f"corrected_archive_missing:{stem}:{label}")
                continue
            with tempfile.NamedTemporaryFile(dir=FIXTURES, suffix=".full-sanitized.tar.gz", delete=False) as handle:
                full_sanitized = Path(handle.name)
            try:
                sanitization = fetch.sanitizer.sanitize(upstream, full_sanitized)
                full_sha = fetch.sha256_file(full_sanitized)
                projection = fetch.project_archive(
                    full_sanitized,
                    projected,
                    root=root,
                    relative_paths=fetch.required_paths(source),
                )
            except Exception as exc:  # noqa: BLE001 - transport must fail closed.
                faults.append(f"corrected_projection_failed:{stem}:{label}:{exc}")
                continue
            finally:
                full_sanitized.unlink(missing_ok=True)
            sanitization["transport_sanitized_output_retained"] = False
            sanitization["transport_sanitized_output_sha256"] = full_sha
            sanitization["projected_output"] = {"path": fetch.relative(projected), "sha256": fetch.sha256_file(projected)}
            sanitization["projection"] = projection
            sanitization_path.write_text(json.dumps(sanitization, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if sanitization.get("trigger_state") != "GREEN" or sanitization.get("source_archive_root") != root:
                faults.append(f"corrected_sanitization_invalid:{stem}:{label}")
            artifacts.append(
                {
                    "label": label,
                    "revision": revision,
                    "root": root,
                    "url": url,
                    "upstream": fetch.relative(upstream),
                    "upstream_sha256": fetch.sha256_file(upstream),
                    "projected": fetch.relative(projected),
                    "projected_sha256": fetch.sha256_file(projected),
                    "sanitization_report": fetch.relative(sanitization_path),
                    "sanitization_report_sha256": fetch.sha256_file(sanitization_path),
                    "projection": projection,
                }
            )
        effect_differs = False
        if len(artifacts) == 2:
            effect_path = str(source["allowed_effect_paths"][0])
            effect_differs = _effect_payload(ROOT / artifacts[0]["projected"], artifacts[0]["root"], effect_path) != _effect_payload(ROOT / artifacts[1]["projected"], artifacts[1]["root"], effect_path)
            if not effect_differs:
                faults.append(f"corrected_effect_pair_identical:{stem}")
        task_rows.append({"stem": stem, "repository": source["repository"], "allowed_effect_paths": source["allowed_effect_paths"], "effect_source_differs": effect_differs, "artifacts": artifacts})
    report = {
        "policy": "project_theseus_p4v2r2r2_revision_repair_fetch_v1",
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "revision_corrections": correction_audit,
        "source_registry": {"path": fetch.relative(REGISTRY), "sha256": fetch.sha256_file(REGISTRY)},
        "original_source_fetch": {"path": fetch.relative(ORIGINAL_FETCH), "sha256": fetch.sha256_file(ORIGINAL_FETCH), "retained": True},
        "tasks": task_rows,
        "corrected_task_count": len(task_rows),
        "corrected_artifact_count": sum(len(row["artifacts"]) for row in task_rows),
        "parent_target_evaluator_executions_before_correction": 24,
        "candidate_or_control_calls": 0,
        "local_model_calls": 0,
        "hosted_model_calls": 0,
        "teacher_calls": 0,
        "D1_cases_consumed": 0,
        "D2_cases_consumed": 0,
        "training_rows_written": 0,
        "project_selected_quality_token_cap": None,
        "maximum_inference": "A GREEN receipt establishes corrected immutable archive transport and nonidentical production effect surfaces for three pre-candidate task pairs only. It is not evaluator adequacy or model evidence.",
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    report = acquire(fetch_network=args.fetch)
    print(json.dumps({key: report[key] for key in ("trigger_state", "faults", "corrected_task_count", "corrected_artifact_count")}, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
