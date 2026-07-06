"""Audit predecessor projects for Theseus concept ports.

This is intentionally source-only. The old project directory contains very
large archives and generated caches; this script reads the curated port map and
checks only declared project roots and evidence paths.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP = ROOT / "configs" / "legacy_concept_port_map.json"
DEFAULT_OUT = ROOT / "reports" / "legacy_project_concept_audit.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "legacy_project_concept_audit.md"

PRIORITY_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
STATUS_RANK = {"not_ported": 0, "partial": 1, "planned": 2, "done": 3, "retired": 4}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default=str(DEFAULT_MAP.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()

    map_path = resolve_repo_path(args.map)
    config = read_json(map_path)
    report = build_report(config, map_path)
    write_json(resolve_repo_path(args.out), report)
    write_markdown(resolve_repo_path(args.markdown_out), report)
    print(json.dumps(report, indent=2))
    return 0


def build_report(config: dict[str, Any], map_path: Path) -> dict[str, Any]:
    projects = [project_report(item) for item in config.get("projects", []) if isinstance(item, dict)]
    projects_by_id = {str(item.get("id")): item for item in projects}
    candidates = [
        candidate_report(item, projects_by_id)
        for item in config.get("port_candidates", [])
        if isinstance(item, dict)
    ]
    top_queue = sorted(
        [item for item in candidates if item.get("status") not in {"done", "retired"}],
        key=lambda item: (
            PRIORITY_RANK.get(str(item.get("priority")), 99),
            STATUS_RANK.get(str(item.get("status")), 99),
            str(item.get("id")),
        ),
    )
    missing_evidence = [
        {
            "project": project.get("id"),
            "path": item.get("path"),
        }
        for project in projects
        for item in project.get("evidence", [])
        if not item.get("exists")
    ]
    status_counts = count_by(candidates, "status")
    priority_counts = count_by(candidates, "priority")
    p0_open = [
        item
        for item in candidates
        if item.get("priority") == "P0" and item.get("status") not in {"done", "retired"}
    ]
    p0_not_ported = [
        item
        for item in candidates
        if item.get("priority") == "P0" and item.get("status") == "not_ported"
    ]
    trigger_state = "RED" if p0_not_ported else ("YELLOW" if p0_open else "GREEN")
    summary = {
        "source_root": config.get("source_root"),
        "projects_declared": len(projects),
        "projects_present": sum(1 for item in projects if item.get("exists")),
        "port_candidates": len(candidates),
        "priority_counts": priority_counts,
        "status_counts": status_counts,
        "p0_open": len(p0_open),
        "p0_not_ported": len(p0_not_ported),
        "missing_evidence_count": len(missing_evidence),
        "trigger_state": trigger_state,
        "top_candidate": top_queue[0].get("id") if top_queue else None,
    }
    return {
        "policy": "theseus_legacy_project_concept_audit_v0",
        "created_utc": now(),
        "map": str(map_path.relative_to(ROOT)) if is_relative_to(map_path, ROOT) else str(map_path),
        "scan_mode": config.get("scan_policy", {}).get("mode", "source_and_docs_only"),
        "summary": summary,
        "projects": projects,
        "port_candidates": candidates,
        "top_port_queue": top_queue[:12],
        "missing_evidence": missing_evidence[:100],
        "next_actions": next_actions(summary, top_queue),
        "external_inference_calls": 0,
    }


def project_report(project: dict[str, Any]) -> dict[str, Any]:
    project_path = Path(str(project.get("path") or ""))
    evidence = []
    for rel in project.get("source_evidence", []):
        rel_text = str(rel)
        path = Path(rel_text)
        full = path if path.is_absolute() else project_path / rel_text
        evidence.append(
            {
                "path": str(full),
                "relative_path": rel_text,
                "exists": full.exists(),
                "kind": "dir" if full.is_dir() else ("file" if full.is_file() else "missing"),
            }
        )
    return {
        "id": project.get("id"),
        "path": str(project_path),
        "exists": project_path.exists(),
        "role_in_lineage": project.get("role_in_lineage"),
        "evidence": evidence,
        "evidence_present": sum(1 for item in evidence if item.get("exists")),
        "evidence_declared": len(evidence),
    }


def candidate_report(candidate: dict[str, Any], projects_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    project = projects_by_id.get(str(candidate.get("project")), {})
    acceptance = [str(item) for item in candidate.get("acceptance_gates", [])]
    return {
        "id": candidate.get("id"),
        "project": candidate.get("project"),
        "source_project_present": bool(project.get("exists")),
        "priority": candidate.get("priority"),
        "status": candidate.get("status"),
        "concept": candidate.get("concept"),
        "port_goal": candidate.get("port_goal"),
        "theseus_surface": candidate.get("theseus_surface", []),
        "acceptance_gates": acceptance,
        "open_acceptance_gate_count": 0 if candidate.get("status") in {"done", "retired"} else len(acceptance),
    }


def next_actions(summary: dict[str, Any], queue: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    if summary.get("missing_evidence_count"):
        actions.append("Review missing legacy evidence paths before claiming the audit is complete.")
    for item in queue[:4]:
        actions.append(
            f"Port {item.get('id')} from {item.get('project')}: {item.get('port_goal')}"
        )
    if not actions:
        actions.append("All declared legacy port candidates are done or retired; refresh the map if new old-project concepts are found.")
    return actions


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report.get("summary") or {}
    rows = [
        "# Legacy Project Concept Audit",
        "",
        f"Updated: {report.get('created_utc')}",
        "",
        "This report audits the curated predecessor-project concept map without scanning generated caches, archives, target folders, or bulk datasets.",
        "",
        "## Summary",
        "",
        f"- trigger_state: {summary.get('trigger_state')}",
        f"- projects_present: {summary.get('projects_present')}/{summary.get('projects_declared')}",
        f"- port_candidates: {summary.get('port_candidates')}",
        f"- p0_open: {summary.get('p0_open')}",
        f"- p0_not_ported: {summary.get('p0_not_ported')}",
        f"- missing_evidence_count: {summary.get('missing_evidence_count')}",
        "",
        "## Project Evidence",
        "",
        "| Project | Present | Evidence | Role |",
        "| --- | --- | ---: | --- |",
    ]
    for project in report.get("projects", []):
        rows.append(
            f"| {project.get('id')} | {project.get('exists')} | "
            f"{project.get('evidence_present')}/{project.get('evidence_declared')} | "
            f"{escape_pipe(project.get('role_in_lineage'))} |"
        )
    rows.extend(["", "## Top Port Queue", "", "| Priority | Status | Candidate | Goal |", "| --- | --- | --- | --- |"])
    for item in report.get("top_port_queue", []):
        rows.append(
            f"| {item.get('priority')} | {item.get('status')} | {item.get('id')} | {escape_pipe(item.get('port_goal'))} |"
        )
    rows.extend(["", "## Next Actions", ""])
    for action in report.get("next_actions", []):
        rows.append(f"- {action}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def escape_pipe(value: Any) -> str:
    return str(value or "").replace("|", "\\|")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
