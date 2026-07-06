#!/usr/bin/env python3
"""Durable deprecation/archive registry for Theseus cleanup.

The registry classifies cleanup candidates without deleting them. It is the
control-plane source of truth for whether a script, doc, report, checkpoint, or
legacy system is live, retained, migrated, deprecated, archived, or merely a
compatibility wrapper.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import report_evidence_store  # noqa: E402


REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "theseus_deprecation_registry.json"
DEFAULT_MARKDOWN = REPORTS / "theseus_deprecation_registry.md"
DEFAULT_DB = report_evidence_store.DEFAULT_DB
PROJECT_REGISTRY = REPORTS / "theseus_project_registry.json"
STATUS_VALUES = {"live", "compatibility_wrapper", "migrated", "deprecated", "archived", "retained", "generated"}
CANONICAL_DOC = "docs/OLD_PROJECTS_TRANSFER_AUDIT.md"
REDIRECT_DOCS = {
    "deprecated/docs/legacy-transfer/OLD_PROJECTS_DEEP_TRANSFER_AUDIT.md",
    "deprecated/docs/legacy-transfer/LEGACY_PROJECTS_CONCEPT_AUDIT.md",
    "deprecated/docs/legacy-transfer/LEGACY_PORT_MECHANISMS.md",
    "deprecated/docs/legacy-transfer/OLD_PROJECTS_PORT_COMPLETION_AUDIT.md",
    "deprecated/docs/legacy-transfer/OLD_PROJECTS_PORT_REVIEW.md",
}
RETIRED_BACKGROUND_DOCS = {
    "deprecated/docs/background/WHITEPAPER.md": "docs/PROJECT_THESEUS_WHITEPAPER.md",
    "deprecated/docs/background/STANDALONE_PARITY_PLAN.md": "docs/TRAINING_EVALS_BENCHMARKS.md",
}
CANONICAL_FACT_PRODUCERS = {
    "scripts/resource_aware_execution_policy.py",
    "scripts/windows_cuda_doctor.py",
    "scripts/learning_launch_supervisor.py",
    "scripts/autonomy_watchdog.py",
    "scripts/hive_work_board_executor.py",
    "scripts/candidate_promotion_gate.py",
    "scripts/maturity_integrity_audit.py",
    "scripts/asi_wall_breaker_governor.py",
}
LIVE_OPERATIONAL_ARTIFACTS = {
    "reports/context_packets.jsonl",
    "reports/hive_artifact_sync_ledger.jsonl",
    "reports/hive_work_board.sqlite",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_DB.relative_to(ROOT)))
    parser.add_argument("--hygiene", default="reports/theseus_workspace_hygiene_audit.json")
    parser.add_argument("--project-registry", default=str(PROJECT_REGISTRY.relative_to(ROOT)))
    parser.add_argument("--retention-manifest", default="reports/theseus_artifact_retention_manifest.json")
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()

    started = time.perf_counter()
    hygiene = read_json(resolve(args.hygiene), {})
    project_registry = read_json(resolve(args.project_registry), {})
    manifest = read_json(resolve(args.retention_manifest), {})
    entries = build_entries(hygiene, manifest, project_registry)
    summary = build_summary(entries, started)
    trigger_state = "GREEN" if summary["deprecated_count"] or summary["archived_count"] or summary["migrated_count"] else "YELLOW"
    payload = {
        "policy": "project_theseus_deprecation_registry_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": summary,
        "project_registry": project_registry_summary(project_registry, args.project_registry),
        "status_values": sorted(STATUS_VALUES),
        "entries": entries,
        "rules": {
            "non_destructive": True,
            "delete_requires": "separate explicit operator approval plus registry proof that no live consumer exists",
            "archive_requires": "manifest entry plus pointer or resolver path",
            "fact_producers": "overlapping peripheral scripts should emit facts consumed by theseus_control_plane.py, not compete as decision owners",
        },
        "external_inference_calls": 0,
    }
    write_registry_table(resolve(args.db), entries)
    report_evidence_store.write_json_report(
        resolve(args.out),
        payload,
        markdown_path=resolve(args.markdown_out),
        markdown_text=render_markdown(payload),
        db_path=resolve(args.db),
    )
    print(json.dumps(payload, indent=2))
    return 0


def build_entries(hygiene: dict[str, Any], manifest: dict[str, Any], project_registry: dict[str, Any]) -> list[dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    archive_by_original = {
        str(row.get("original_path")): row
        for row in manifest.get("entries", [])
        if isinstance(row, dict) and row.get("original_path")
    }

    for path in sorted(CANONICAL_FACT_PRODUCERS):
        if (ROOT / path).exists():
            entries[path] = entry(path, "script", "live", "canonical_control_plane_fact_producer")
    if (ROOT / CANONICAL_DOC).exists():
        entries[CANONICAL_DOC] = entry(CANONICAL_DOC, "doc", "live", "canonical_old_project_transfer_doc")
    for path in sorted(REDIRECT_DOCS):
        if (ROOT / path).exists():
            entries[path] = entry(path, "doc", "migrated", "redirect_stub_to_canonical_old_project_transfer_doc", replacement=CANONICAL_DOC)
    for path, replacement in sorted(RETIRED_BACKGROUND_DOCS.items()):
        if (ROOT / path).exists():
            entries[path] = entry(path, "doc", "deprecated", "retired_background_draft_superseded_by_current_docs", replacement=replacement)
    for original, archived in sorted(archive_by_original.items()):
        entries[original] = entry(
            original,
            "artifact",
            "archived",
            "manifest_backed_heavy_artifact_archive",
            replacement=str(archived.get("archive_path") or ""),
            evidence=archived,
        )
    for row in project_registry_entries(project_registry):
        entries[row["path"]] = row

    for candidate in hygiene.get("candidates", []) if isinstance(hygiene.get("candidates"), list) else []:
        if not isinstance(candidate, dict):
            continue
        kind = str(candidate.get("kind") or "")
        path = str(candidate.get("path") or "")
        if kind == "legacy_script_review" and path:
            refs = int(
                get_path(
                    candidate,
                    ["evidence", "reference_count_in_active_sources"],
                    get_path(candidate, ["evidence", "reference_count_in_scripts"], 0),
                )
                or 0
            )
            status = "deprecated" if refs == 0 else "compatibility_wrapper"
            entries[path] = entry(
                path,
                "script",
                status,
                "legacy_or_old_project_script_reviewed_by_hygiene_audit",
                evidence=candidate.get("evidence", {}),
                replacement=CANONICAL_DOC if "legacy_port" in path or "old_project" in path else "",
            )
        elif kind == "peripheral_overlap":
            for script in get_path(candidate, ["evidence", "scripts"], []) or []:
                script_path = f"scripts/{script}"
                if script_path in CANONICAL_FACT_PRODUCERS:
                    entries.setdefault(script_path, entry(script_path, "script", "live", "canonical_control_plane_fact_producer"))
                else:
                    entries[script_path] = entry(
                        script_path,
                        "script",
                        "compatibility_wrapper",
                        "overlapping_peripheral_role_should_feed_control_plane_not_own_decisions",
                        evidence=candidate.get("evidence", {}),
                        replacement="scripts/theseus_control_plane.py",
                    )
        elif kind == "doc_consolidation":
            entries[CANONICAL_DOC] = entry(CANONICAL_DOC, "doc", "live", "canonical_consolidated_doc")
            for doc in get_path(candidate, ["evidence", "docs"], []) or []:
                if str(doc) == CANONICAL_DOC:
                    continue
                entries[str(doc)] = entry(str(doc), "doc", "migrated", "consolidated_redirect_stub", replacement=CANONICAL_DOC)
        elif kind == "large_report_retention":
            for item in get_path(candidate, ["evidence", "largest_files"], []) or []:
                original = str(item.get("path") or "")
                if not original:
                    continue
                archived = archive_by_original.get(original)
                if archived:
                    entries[original] = entry(
                        original,
                        "artifact",
                        "archived",
                        "manifest_backed_heavy_artifact_archive",
                        replacement=str(archived.get("archive_path") or ""),
                        evidence=archived,
                    )
                else:
                    reason = (
                        "live_operational_artifact_requires_compaction_or_rotation_not_archive"
                        if original in LIVE_OPERATIONAL_ARTIFACTS
                        else "large_artifact_pending_manifest_backed_archive"
                    )
                    entries[original] = entry(
                        original,
                        "artifact",
                        "retained",
                        reason,
                        evidence=item,
                    )
        elif kind == "dirty_workspace":
            entries["workspace/git_status"] = entry(
                "workspace/git_status",
                "workspace",
                "retained",
                "dirty_workspace_requires_human_or_coherent_commit_review",
                evidence=candidate.get("evidence", {}),
            )

    return sorted(entries.values(), key=lambda row: (row["artifact_type"], row["path"]))


def project_registry_entries(project_registry: dict[str, Any]) -> list[dict[str, Any]]:
    if not project_registry:
        return [
            entry(
                "reports/theseus_project_registry.json",
                "registry",
                "retained",
                "project_manifest_registry_missing_pending_materialization",
                replacement="scripts/theseus_project_registry.py",
            )
        ]
    rows: list[dict[str, Any]] = []
    for surface in project_registry.get("surfaces", []) if isinstance(project_registry.get("surfaces"), list) else []:
        if not isinstance(surface, dict):
            continue
        status = str(surface.get("status") or "live")
        artifact_type = str(surface.get("artifact_type") or "surface")
        if status == "live" and artifact_type not in {"generated_artifact", "deprecated"}:
            continue
        canonical = str(surface.get("canonical") or surface.get("id") or "unknown")
        mapped_status = status if status in STATUS_VALUES else "retained"
        rows.append(
            entry(
                canonical,
                "surface",
                mapped_status,
                "project_manifest_registry_surface_status",
                replacement=str(surface.get("canonical") or ""),
                evidence=surface,
            )
        )
    for item in project_registry.get("root_summaries", []) if isinstance(project_registry.get("root_summaries"), list) else []:
        if not isinstance(item, dict):
            continue
        cleanup_class = str(item.get("cleanup_class") or "")
        if cleanup_class not in {"generated_or_build_state", "deprecated_windows_path_mirror"}:
            continue
        rows.append(
            entry(
                str(item.get("path") or "unknown"),
                "root",
                "generated" if cleanup_class == "generated_or_build_state" else "deprecated",
                cleanup_class,
                evidence=item,
            )
        )
    for item in project_registry.get("unregistered", [])[:200] if isinstance(project_registry.get("unregistered"), list) else []:
        if not isinstance(item, dict):
            continue
        rows.append(
            entry(
                str(item.get("path") or "unknown"),
                "unregistered_source",
                "retained",
                "unregistered_active_source_pending_registry_owner",
                evidence=item,
            )
        )
    for item in project_registry.get("generated_source_artifacts", []) if isinstance(project_registry.get("generated_source_artifacts"), list) else []:
        if not isinstance(item, dict):
            continue
        rows.append(
            entry(
                str(item.get("path") or "unknown"),
                "artifact",
                "generated",
                "generated_artifact_in_source_path_pending_quarantine",
                evidence=item,
            )
        )
    return rows


def entry(
    path: str,
    artifact_type: str,
    status: str,
    reason: str,
    *,
    replacement: str = "",
    evidence: Any = None,
) -> dict[str, Any]:
    if status not in STATUS_VALUES:
        raise ValueError(status)
    return {
        "record_type": "deprecation_registry_entry",
        "path": path,
        "artifact_type": artifact_type,
        "status": status,
        "reason": reason,
        "replacement": replacement,
        "evidence": evidence if evidence is not None else {},
        "updated_utc": now(),
    }


def write_registry_table(db_path: Path, entries: list[dict[str, Any]]) -> None:
    conn = report_evidence_store.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS control_deprecation_registry (
                path TEXT PRIMARY KEY,
                artifact_type TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT NOT NULL,
                replacement TEXT NOT NULL DEFAULT '',
                updated_utc TEXT NOT NULL,
                evidence_json TEXT NOT NULL
            )
            """
        )
        for row in entries:
            conn.execute(
                """
                INSERT INTO control_deprecation_registry (
                    path, artifact_type, status, reason, replacement, updated_utc, evidence_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    artifact_type=excluded.artifact_type,
                    status=excluded.status,
                    reason=excluded.reason,
                    replacement=excluded.replacement,
                    updated_utc=excluded.updated_utc,
                    evidence_json=excluded.evidence_json
                """,
                (
                    row["path"],
                    row["artifact_type"],
                    row["status"],
                    row["reason"],
                    row.get("replacement") or "",
                    row["updated_utc"],
                    json.dumps(row.get("evidence", {}), sort_keys=True),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def build_summary(entries: list[dict[str, Any]], started: float) -> dict[str, Any]:
    counts: dict[str, int] = {status: 0 for status in sorted(STATUS_VALUES)}
    types: dict[str, int] = {}
    for row in entries:
        counts[str(row.get("status"))] = counts.get(str(row.get("status")), 0) + 1
        types[str(row.get("artifact_type"))] = types.get(str(row.get("artifact_type")), 0) + 1
    return {
        "entry_count": len(entries),
        **{f"{status}_count": counts.get(status, 0) for status in sorted(STATUS_VALUES)},
        "artifact_type_counts": dict(sorted(types.items())),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }


def project_registry_summary(project_registry: dict[str, Any], path: str) -> dict[str, Any]:
    summary = project_registry.get("summary") if isinstance(project_registry.get("summary"), dict) else {}
    return {
        "path": path,
        "present": bool(project_registry),
        "trigger_state": project_registry.get("trigger_state") if project_registry else "",
        "surface_count": project_registry.get("surface_count") if project_registry else 0,
        "entry_count": summary.get("entry_count", 0),
        "unregistered_active_source_count": summary.get("unregistered_active_source_count", 0),
        "duplicate_family_count": summary.get("duplicate_family_count", 0),
        "generated_source_artifact_count": summary.get("generated_source_artifact_count", 0),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Theseus Deprecation Registry",
        "",
        f"- trigger_state: `{payload.get('trigger_state')}`",
        f"- entries: `{summary.get('entry_count')}`",
        f"- live: `{summary.get('live_count')}` retained: `{summary.get('retained_count')}`",
        f"- compatibility wrappers: `{summary.get('compatibility_wrapper_count')}`",
        f"- migrated: `{summary.get('migrated_count')}` deprecated: `{summary.get('deprecated_count')}` archived: `{summary.get('archived_count')}`",
        "",
        "## Entries",
        "",
    ]
    for row in payload.get("entries", [])[:80]:
        replacement = f" -> `{row.get('replacement')}`" if row.get("replacement") else ""
        lines.append(f"- `{row.get('status')}` `{row.get('artifact_type')}` `{row.get('path')}`{replacement}")
    return "\n".join(lines) + "\n"


def read_json(path: Path, default: Any) -> Any:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return payload if isinstance(payload, dict) else default


def get_path(payload: Any, path: list[Any], default: Any = None) -> Any:
    current = payload
    for key in path:
        if isinstance(current, dict):
            current = current.get(key, default)
        elif isinstance(current, list) and isinstance(key, int) and 0 <= key < len(current):
            current = current[key]
        else:
            return default
    return current


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
