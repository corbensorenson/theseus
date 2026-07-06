#!/usr/bin/env python3
"""Workspace hygiene/deprecation audit for Project Theseus.

This audit is intentionally non-destructive. It identifies cleanup candidates
and migration targets so the control plane can converge duplicated peripheral
systems without deleting useful evidence or breaking old entrypoints.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import report_evidence_store  # noqa: E402


REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
DEPRECATED = ROOT / "deprecated"
CONTROL_PLANE_OWNERSHIP = ROOT / "configs" / "control_plane_ownership.json"
PROJECT_REGISTRY = REPORTS / "theseus_project_registry.json"
DEFAULT_OUT = REPORTS / "theseus_workspace_hygiene_audit.json"
DEFAULT_MARKDOWN = REPORTS / "theseus_workspace_hygiene_audit.md"
LARGE_REPORT_BYTES = 256 * 1024 * 1024
VERY_LARGE_REPORT_BYTES = 768 * 1024 * 1024
CANONICAL_REPORT_STORAGE = {"report_evidence_store.sqlite"}
LIVE_OPERATIONAL_ARTIFACTS = {
    "reports/context_packets.jsonl",
    "reports/hive_artifact_sync_ledger.jsonl",
    "reports/hive_work_board.sqlite",
    "reports/world_adapter_job_control_ledger.jsonl",
}
DOC_DUPLICATE_KEYWORDS = ("OLD_PROJECTS", "LEGACY", "TRANSFER_AUDIT", "PORT")
CANONICAL_LEGACY_TRANSFER_DOC = "OLD_PROJECTS_TRANSFER_AUDIT.md"
LEGACY_SCRIPT_PREFIXES = ("legacy_", "old_")
CANONICAL_CONTROL_PLANE_SCRIPTS = {
    "theseus_control_plane.py",
    "report_evidence_store.py",
    "resource_aware_execution_policy.py",
    "windows_cuda_doctor.py",
    "system_efficiency_audit.py",
    "asi_wall_breaker_governor.py",
    "attd_analyzer.py",
    "hive_work_board_executor.py",
    "learning_launch_supervisor.py",
    "candidate_promotion_gate.py",
    "maturity_integrity_audit.py",
    "agent_lane_transfer_gate.py",
}
ACTIVE_REFERENCE_ROOTS = (
    ROOT / "README.md",
    ROOT / "scripts",
    ROOT / "configs",
    ROOT / "docs",
    ROOT / "benchmarks" / "cards",
)
ACTIVE_REFERENCE_SUFFIXES = {".py", ".ps1", ".sh", ".json", ".toml", ".md"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-registry", default=str(PROJECT_REGISTRY.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()

    started = time.perf_counter()
    source_refs = active_source_reference_counts()
    legacy_review = legacy_script_review(source_refs)
    ownership = read_json(CONTROL_PLANE_OWNERSHIP, {})
    project_registry = read_json(resolve(args.project_registry), {})
    candidates = []
    candidates.extend(project_registry_candidates(project_registry))
    candidates.extend(legacy_review["candidates"])
    candidates.extend(peripheral_overlap_candidates(ownership))
    candidates.extend(doc_consolidation_candidates())
    candidates.extend(large_report_candidates())
    candidates.extend(dirty_workspace_candidates())
    candidates = sorted(candidates, key=candidate_sort_key)
    summary = build_summary(candidates, started)
    trigger_state = "RED" if summary["high_count"] >= 25 else ("YELLOW" if candidates else "GREEN")
    payload = {
        "policy": "project_theseus_workspace_hygiene_audit_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": summary,
        "canonical_control_plane_scripts": sorted(CANONICAL_CONTROL_PLANE_SCRIPTS),
        "project_registry": project_registry_summary(project_registry, args.project_registry),
        "control_plane_ownership": control_plane_ownership_summary(ownership),
        "retained_legacy_bridge_scripts": legacy_review["retained"],
        "retained_live_large_artifacts": retained_live_large_artifacts(),
        "candidates": candidates,
        "deprecation_policy": {
            "non_destructive": True,
            "default_action": "mark_review_or_migrate_before_delete",
            "safe_retirement_steps": [
                "prove no current control-plane consumer depends on the artifact",
                "move docs/scripts to deprecated/ with a pointer stub or registry entry",
                "preserve historical reports through report_evidence_store snapshots before archiving",
                "rerun theseus_control_plane.py and this audit",
            ],
            "artifact_retention": "large historical checkpoints should move out of reports/ into a retention/archive tier with manifest pointers, not be deleted blindly",
        },
        "external_inference_calls": 0,
    }
    report_evidence_store.write_json_report(
        resolve(args.out),
        payload,
        markdown_path=resolve(args.markdown_out),
        markdown_text=render_markdown(payload),
    )
    print(json.dumps(payload, indent=2))
    return 0 if trigger_state != "RED" else 2


def legacy_script_review(source_refs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    retained = []
    for path in sorted(SCRIPTS.glob("*.py")):
        name = path.name
        if not name.startswith(LEGACY_SCRIPT_PREFIXES):
            continue
        refs = source_refs.get(name, {"count": 0, "consumers": []})
        ref_count = int(refs.get("count") or 0)
        evidence = {
            "reference_count_in_active_sources": ref_count,
            "active_source_consumers": refs.get("consumers", []),
            "bytes": path.stat().st_size,
        }
        if ref_count > 0:
            retained.append(
                {
                    "path": rel(path),
                    "status": "retained_legacy_bridge_producer",
                    "reason": "active_source_references_exist",
                    "evidence": evidence,
                }
            )
            continue
        candidates.append(
            candidate(
                "legacy_script_review",
                f"legacy_script_{path.stem}",
                "high",
                path,
                "Legacy/old project script should be reviewed for migration to canonical lanes or moved under deprecated/ with a pointer.",
                evidence=evidence,
            )
        )
    return {"candidates": candidates, "retained": retained}


def project_registry_candidates(registry: dict[str, Any]) -> list[dict[str, Any]]:
    if not registry:
        return [
            {
                "record_type": "workspace_hygiene_candidate",
                "kind": "project_registry_missing",
                "id": "project_manifest_registry_missing",
                "priority": "high",
                "path": "reports/theseus_project_registry.json",
                "action": "Materialize the project manifest registry before cleanup decisions so every active surface has an owner/status.",
                "evidence": {"expected_command": "python3 scripts/theseus_project_registry.py"},
            }
        ]
    candidates: list[dict[str, Any]] = []
    governance_violations = [row for row in registry.get("governance_violations", []) if isinstance(row, dict)]
    if governance_violations:
        candidates.append(
            {
                "record_type": "workspace_hygiene_candidate",
                "kind": "project_registry_evolution_contract_violation",
                "id": "project_registry_evolution_contract_violations",
                "priority": "high"
                if any(str(row.get("severity") or "") == "hard" for row in governance_violations)
                else "medium",
                "path": "",
                "action": (
                    "Resolve registry evolution contract violations: update canonical registered surfaces first, "
                    "or declare a complete successor/deprecation relationship before adding a new lane."
                ),
                "evidence": {"count": len(governance_violations), "violations": governance_violations[:40]},
            }
        )
    unregistered = [row for row in registry.get("unregistered", []) if isinstance(row, dict)]
    if unregistered:
        candidates.append(
            {
                "record_type": "workspace_hygiene_candidate",
                "kind": "project_registry_gap",
                "id": "unregistered_active_sources",
                "priority": "high" if len(unregistered) >= 25 else "medium",
                "path": "",
                "action": "Assign unregistered active source/config/doc files to an existing manifest surface, add a narrow owner surface, or move them under deprecated/generated state.",
                "evidence": {"count": len(unregistered), "sample": unregistered[:80]},
            }
        )
    duplicates = [row for row in registry.get("duplicate_families", []) if isinstance(row, dict)]
    if duplicates:
        candidates.append(
            {
                "record_type": "workspace_hygiene_candidate",
                "kind": "duplicate_family_consolidation",
                "id": "project_registry_duplicate_families",
                "priority": "high" if len(duplicates) >= 12 else "medium",
                "path": "",
                "action": "Consolidate duplicate vN/seed/current/after families behind canonical owners or mark deliberate compatibility wrappers.",
                "evidence": {"count": len(duplicates), "families": duplicates[:50]},
            }
        )
    stale_or_missing = [
        row
        for row in registry.get("report_outputs", [])
        if isinstance(row, dict) and row.get("status") in {"stale", "missing"}
    ]
    if stale_or_missing:
        candidates.append(
            {
                "record_type": "workspace_hygiene_candidate",
                "kind": "stale_or_missing_registry_report_outputs",
                "id": "project_registry_report_outputs_refresh",
                "priority": "high" if len(stale_or_missing) >= 12 else "medium",
                "path": "reports",
                "action": "Refresh or intentionally retire stale/missing report outputs listed by the project registry before trusting the control plane.",
                "evidence": {"count": len(stale_or_missing), "reports": stale_or_missing[:80]},
            }
        )
    generated_source = [row for row in registry.get("generated_source_artifacts", []) if isinstance(row, dict)]
    if generated_source:
        candidates.append(
            {
                "record_type": "workspace_hygiene_candidate",
                "kind": "generated_artifact_in_source_path",
                "id": "project_registry_generated_source_artifacts",
                "priority": "high",
                "path": "",
                "action": "Quarantine generated cache/scratch files from source paths and add ignore coverage where needed.",
                "evidence": {"count": len(generated_source), "sample": generated_source[:40]},
            }
        )
    return candidates


def peripheral_overlap_candidates(ownership: dict[str, Any]) -> list[dict[str, Any]]:
    groups = {
        "supervisor_overlap": [
            "learning_launch_supervisor.py",
            "vacation_mode_supervisor.py",
            "unattended_autonomy_supervisor.py",
            "autonomy_watchdog.py",
            "autonomy_rotation_governor.py",
        ],
        "resource_overlap": [
            "resource_aware_execution_policy.py",
            "resource_governor.py",
            "training_resource_runway.py",
            "windows_cuda_doctor.py",
        ],
        "promotion_gate_overlap": [
            "candidate_promotion_gate.py",
            "model_growth_gate.py",
            "maturity_integrity_audit.py",
            "coherence_delirium_gate.py",
            "asi_wall_breaker_governor.py",
        ],
    }
    owned = owned_peripheral_groups(ownership)
    candidates = []
    for group, names in groups.items():
        existing = [name for name in names if (SCRIPTS / name).exists()]
        if len(existing) <= 1:
            continue
        owned_scripts = owned.get(group, set())
        if owned_scripts and {f"scripts/{name}" for name in existing}.issubset(owned_scripts):
            continue
        candidates.append(
            {
                "record_type": "workspace_hygiene_candidate",
                "kind": "peripheral_overlap",
                "id": group,
                "priority": "medium",
                "path": "",
                "action": "Unify decision ownership in theseus_control_plane.py; keep old scripts as producers or compatibility wrappers only.",
                "evidence": {"scripts": existing},
            }
        )
    return candidates


def doc_consolidation_candidates() -> list[dict[str, Any]]:
    docs = []
    for path in sorted(DOCS.glob("*.md")):
        if path.name == CANONICAL_LEGACY_TRANSFER_DOC:
            continue
        upper = path.name.upper()
        if any(keyword in upper for keyword in DOC_DUPLICATE_KEYWORDS):
            docs.append(path)
    if not docs:
        return []
    return [
        {
            "record_type": "workspace_hygiene_candidate",
            "kind": "doc_consolidation",
            "id": "old_legacy_port_docs_consolidation",
            "priority": "medium",
            "path": "docs",
            "action": "Consolidate old-project/legacy-port docs into one canonical page plus redirect stubs.",
            "evidence": {
                "doc_count": len(docs),
                "docs": [rel(path) for path in docs[:24]],
                "total_bytes": sum(path.stat().st_size for path in docs),
            },
        }
    ]


def large_report_candidates() -> list[dict[str, Any]]:
    candidates = []
    families: dict[str, list[Path]] = defaultdict(list)
    for path in REPORTS.iterdir() if REPORTS.exists() else []:
        if not path.is_file():
            continue
        if path.name in CANONICAL_REPORT_STORAGE:
            continue
        if rel(path) in LIVE_OPERATIONAL_ARTIFACTS:
            continue
        size = path.stat().st_size
        if size < LARGE_REPORT_BYTES:
            continue
        families[large_report_family(path.name)].append(path)
    for family, paths in sorted(families.items(), key=lambda item: (-sum(p.stat().st_size for p in item[1]), item[0])):
        total = sum(path.stat().st_size for path in paths)
        priority = "high" if total >= VERY_LARGE_REPORT_BYTES else "medium"
        candidates.append(
            {
                "record_type": "workspace_hygiene_candidate",
                "kind": "large_report_retention",
                "id": f"large_report_family_{safe_id(family)}",
                "priority": priority,
                "path": "reports",
                "action": "Move historical heavyweight artifacts out of reports/ latest-view space into a manifest-backed archive tier.",
                "evidence": {
                    "family": family,
                    "file_count": len(paths),
                    "total_gib": round(total / (1024**3), 3),
                    "largest_files": [
                        {"path": rel(path), "gib": round(path.stat().st_size / (1024**3), 3)}
                        for path in sorted(paths, key=lambda item: item.stat().st_size, reverse=True)[:8]
                    ],
                },
            }
        )
    return candidates


def dirty_workspace_candidates() -> list[dict[str, Any]]:
    rows = git_status_rows()
    if not rows:
        return []
    buckets = Counter(row[:2].strip() or "??" for row in rows)
    return [
        {
            "record_type": "workspace_hygiene_candidate",
            "kind": "dirty_workspace",
            "id": "dirty_workspace_review",
            "priority": "high",
            "path": "",
            "action": "Review dirty files, separate intentional architecture changes from generated artifacts, and commit or archive coherent units.",
            "evidence": {"status_counts": dict(sorted(buckets.items())), "sample": rows[:40], "count": len(rows)},
        }
    ]


def active_source_reference_counts() -> dict[str, dict[str, Any]]:
    reference_files = active_reference_files()
    script_paths = [path for path in SCRIPTS.glob("*.py") if path.name.startswith(LEGACY_SCRIPT_PREFIXES)]
    counts: dict[str, dict[str, Any]] = {}
    for script in script_paths:
        count = 0
        consumers: list[str] = []
        patterns = [
            re.escape(script.name),
            re.escape(rel(script)),
            rf"(?<![A-Za-z0-9_]){re.escape(script.stem)}(?![A-Za-z0-9_])",
        ]
        for ref_path in reference_files:
            if ref_path == script:
                continue
            text = ref_path.read_text(encoding="utf-8", errors="ignore")
            hits = sum(len(re.findall(pattern, text)) for pattern in patterns)
            if hits:
                count += hits
                consumers.append(rel(ref_path))
        counts[script.name] = {"count": count, "consumers": consumers[:24]}
    return counts


def active_reference_files() -> list[Path]:
    files: list[Path] = []
    for root in ACTIVE_REFERENCE_ROOTS:
        if root.is_file() and root.suffix.lower() in ACTIVE_REFERENCE_SUFFIXES:
            files.append(root)
        elif root.is_dir():
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in ACTIVE_REFERENCE_SUFFIXES:
                    files.append(path)
    return sorted(set(files))


def owned_peripheral_groups(ownership: dict[str, Any]) -> dict[str, set[str]]:
    owned: dict[str, set[str]] = {}
    for group in ownership.get("groups", []) if isinstance(ownership.get("groups"), list) else []:
        if not isinstance(group, dict):
            continue
        group_id = str(group.get("id") or "")
        decision_owner = str(group.get("decision_owner") or ownership.get("decision_owner") or "")
        if decision_owner != "scripts/theseus_control_plane.py":
            continue
        scripts = {
            str(row.get("path") or "")
            for row in group.get("scripts", [])
            if isinstance(row, dict) and row.get("path")
        }
        if group_id and scripts:
            owned[group_id] = scripts
    return owned


def control_plane_ownership_summary(ownership: dict[str, Any]) -> dict[str, Any]:
    groups = ownership.get("groups", []) if isinstance(ownership.get("groups"), list) else []
    return {
        "manifest": rel(CONTROL_PLANE_OWNERSHIP),
        "present": bool(ownership),
        "policy": ownership.get("policy"),
        "decision_owner": ownership.get("decision_owner"),
        "group_count": len(groups),
        "owned_group_ids": [str(group.get("id") or "") for group in groups if isinstance(group, dict)],
    }


def project_registry_summary(registry: dict[str, Any], path: str) -> dict[str, Any]:
    summary = registry.get("summary") if isinstance(registry.get("summary"), dict) else {}
    return {
        "path": path,
        "present": bool(registry),
        "trigger_state": registry.get("trigger_state") if registry else "",
        "surface_count": registry.get("surface_count") if registry else 0,
        "entry_count": summary.get("entry_count", 0),
        "coverage_ratio": summary.get("coverage_ratio"),
        "unregistered_active_source_count": summary.get("unregistered_active_source_count", 0),
        "duplicate_family_count": summary.get("duplicate_family_count", 0),
        "source_duplicate_family_count": summary.get("source_duplicate_family_count", 0),
        "classified_source_duplicate_family_count": summary.get("classified_source_duplicate_family_count", 0),
        "unclassified_source_duplicate_family_count": summary.get("unclassified_source_duplicate_family_count", 0),
        "stale_report_output_count": summary.get("stale_report_output_count", 0),
        "missing_report_output_count": summary.get("missing_report_output_count", 0),
        "generated_source_artifact_count": summary.get("generated_source_artifact_count", 0),
        "registry_governance_violation_count": summary.get("registry_governance_violation_count", 0),
        "registry_hard_governance_violation_count": summary.get("registry_hard_governance_violation_count", 0),
    }


def retained_live_large_artifacts() -> list[dict[str, Any]]:
    rows = []
    for path_text in sorted(LIVE_OPERATIONAL_ARTIFACTS):
        path = resolve(path_text)
        if not path.exists() or not path.is_file():
            continue
        rows.append(
            {
                "path": path_text,
                "status": "retained_live_operational_artifact",
                "gib": round(path.stat().st_size / (1024**3), 3),
                "reason": "append_only_or_sqlite_runtime_state_not_historical_checkpoint_json",
            }
        )
    return rows


def build_summary(candidates: list[dict[str, Any]], started: float) -> dict[str, Any]:
    by_kind = Counter(str(row.get("kind") or "unknown") for row in candidates)
    by_priority = Counter(str(row.get("priority") or "medium") for row in candidates)
    return {
        "candidate_count": len(candidates),
        "high_count": int(by_priority.get("high", 0)),
        "medium_count": int(by_priority.get("medium", 0)),
        "low_count": int(by_priority.get("low", 0)),
        "kinds": dict(sorted(by_kind.items())),
        "priority_counts": dict(sorted(by_priority.items())),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Theseus Workspace Hygiene Audit",
        "",
        f"- trigger_state: `{payload.get('trigger_state')}`",
        f"- candidates: `{summary.get('candidate_count')}`",
        f"- high: `{summary.get('high_count')}` medium: `{summary.get('medium_count')}` low: `{summary.get('low_count')}`",
        "",
        "## Top Candidates",
        "",
    ]
    for row in payload.get("candidates", [])[:30]:
        lines.append(
            f"- `{row.get('id')}` priority=`{row.get('priority')}` kind=`{row.get('kind')}` path=`{row.get('path')}`"
        )
        action = str(row.get("action") or "")
        if action:
            lines.append(f"  - {action}")
    lines.append("")
    return "\n".join(lines)


def candidate(
    kind: str,
    candidate_id: str,
    priority: str,
    path: Path,
    action: str,
    *,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "record_type": "workspace_hygiene_candidate",
        "kind": kind,
        "id": candidate_id,
        "priority": priority,
        "path": rel(path),
        "action": action,
        "evidence": evidence,
    }


def candidate_sort_key(row: dict[str, Any]) -> tuple[int, str, str]:
    rank = {"high": 0, "medium": 1, "low": 2}.get(str(row.get("priority") or "medium").lower(), 1)
    return rank, str(row.get("kind") or ""), str(row.get("id") or "")


def large_report_family(name: str) -> str:
    for marker in ["_seed", "_v", "_train_once", "_broad_floor", "_private_pressure"]:
        if marker in name:
            return name.split(marker, 1)[0]
    return Path(name).stem


def git_status_rows() -> list[str]:
    try:
        result = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._").lower()[:120] or "unknown"


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
