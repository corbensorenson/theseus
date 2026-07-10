#!/usr/bin/env python3
"""Materialize the Project Theseus manifest registry.

The registry is the lifecycle map for the repo. It classifies active source
surfaces, generated state, report outputs, compatibility wrappers, and cleanup
targets so ATTD/control-plane tools can stop rediscovering the same clutter.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import sqlite3
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
import viea_spine_records  # noqa: E402


DEFAULT_POLICY = ROOT / "configs" / "project_manifest_registry.json"
DEFAULT_STEWARD_CONFIG = ROOT / "configs" / "project_steward.json"
REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "theseus_project_registry.json"
DEFAULT_MARKDOWN = REPORTS / "theseus_project_registry.md"
SOURCE_ROOTS = ("scripts", "configs", "docs", "crates", "dashboard", "benchmarks/cards", "tests", "src")
SOURCE_EXTENSIONS = {".py", ".rs", ".json", ".toml", ".md", ".html", ".css", ".js", ".ps1", ".sh", ".yml", ".yaml"}
ROUTE_VALIDATOR_SPINE_GROUPS = ("governance_records", "failure_boundaries", "authority_records", "resource_route_records")
ROUTE_EVIDENCE_MODES = {"all", "current_invocation", "not_route_required"}
ROUTE_EVIDENCE_FRESHNESS_MODES = {"source_bound", "ttl", "exists"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--db", default=str(report_evidence_store.DEFAULT_DB.relative_to(ROOT)))
    parser.add_argument("--gate", action="store_true", help="Emit a compact registry gate result for other gates/routers.")
    args = parser.parse_args()

    started = time.perf_counter()
    policy_path = resolve(args.policy)
    policy = read_json(policy_path, {})
    inventory = build_inventory(policy)
    entries, unregistered = classify_inventory(inventory, policy)
    report_status = report_output_status(policy)
    base_abstraction_gaps = abstraction_registry_gaps(policy)
    route_validator_spine_receipt = route_validator_materialized_view_receipt()
    stable_field_gaps = stable_capability_field_gaps(policy, route_validator_spine_receipt)
    abstraction_gaps = base_abstraction_gaps + stable_field_gaps
    duplicate_families = duplicate_family_report(inventory, policy)
    generated_source = generated_source_artifacts(inventory, policy)
    implementation_health = implementation_registry_health(policy, report_status)
    stable_field_health = stable_capability_field_health(policy, stable_field_gaps, implementation_health)
    abstraction_health = abstraction_registry_health(policy, implementation_health, abstraction_gaps)
    routing_eligibility = registry_routing_eligibility(policy, implementation_health)
    cleanup_queue = registry_cleanup_queue(
        abstraction_gaps,
        implementation_health,
        duplicate_families,
        report_status,
        generated_source,
    )
    registry_decisions = registry_decision_report(policy, cleanup_queue, routing_eligibility)
    root_summaries = root_size_summaries(policy)
    steward_coverage = steward_coverage_report(policy)
    governance_violations = registry_governance_violations(
        policy,
        entries,
        unregistered,
        abstraction_gaps,
        duplicate_families,
        report_status,
        generated_source,
        implementation_health,
        steward_coverage,
    )
    summary = build_summary(
        entries,
        unregistered,
        abstraction_gaps,
        duplicate_families,
        report_status,
        generated_source,
        abstraction_health,
        implementation_health,
        routing_eligibility,
        cleanup_queue,
        root_summaries,
        governance_violations,
        stable_field_gaps,
        stable_field_health,
        steward_coverage,
        route_validator_spine_receipt,
        policy,
        started,
    )
    trigger_state = trigger_state_for(summary, policy)
    payload = {
        "policy": "project_theseus_project_registry_v1",
        "created_utc": now(),
        "policy_file": rel(policy_path),
        "trigger_state": trigger_state,
        "summary": summary,
        "root_summaries": root_summaries,
        "surface_count": len(policy.get("surfaces", [])) if isinstance(policy.get("surfaces"), list) else 0,
        "abstraction_count": len(policy.get("abstractions", [])) if isinstance(policy.get("abstractions"), list) else 0,
        "implementation_count": len(policy.get("implementations", [])) if isinstance(policy.get("implementations"), list) else 0,
        "route_evidence_contracts": policy.get("route_evidence_contracts", []),
        "registry_evolution_contract": policy.get("registry_evolution_contract", {}),
        "project_steward_config": steward_coverage.get("config_path", ""),
        "project_steward_coverage": steward_coverage,
        "abstraction_registry_contract": policy.get("abstraction_registry_contract", {}),
        "stable_capability_field_contract": policy.get("stable_capability_field_contract", {}),
        "route_validator_spine_receipt": route_validator_spine_receipt,
        "abstractions": compact_abstractions(policy),
        "implementations": compact_implementations(policy),
        "abstraction_registry_gaps": abstraction_gaps,
        "stable_capability_field_gaps": stable_field_gaps,
        "stable_capability_field_health": stable_field_health,
        "abstraction_health": abstraction_health,
        "implementation_health": implementation_health,
        "routing_eligibility": routing_eligibility,
        "cleanup_queue": cleanup_queue,
        "registry_decisions": registry_decisions,
        "surfaces": compact_surfaces(policy),
        "entries": entries,
        "unregistered": unregistered,
        "report_outputs": report_status,
        "duplicate_families": duplicate_families,
        "generated_source_artifacts": generated_source,
        "governance_violations": governance_violations,
        "rules": {
            "source_of_truth": "configs/project_manifest_registry.json defines owned lifecycle surfaces; this report is the materialized current state.",
            "unregistered_active_source": "active source/config/doc files not matched by a surface need an owner or explicit deprecated/generated status.",
            "duplicates": "families with many vN/seed/current/after variants should be consolidated or registered as deliberate compatibility wrappers.",
            "generated_artifacts": "generated/runtime payloads must live under generated roots or be manifest-backed; source paths should stay clean.",
            "public_benchmark_boundary": "registry work does not unlock public calibration or turn public benchmark payloads into training data.",
            "abstraction_boundary": "stable abstractions own contracts; implementations bind surfaces/backends to those contracts and can be swapped only through registry-visible replacement policy.",
            "stable_capability_field": "capability fields must declare semantic contracts, exact content identity, authority/effect ceilings, state continuity, scoped qualification evidence, caller-bound route validation, leases, adaptation envelopes, observability, lifecycle/recovery, and governance controls before routing.",
            "route_validator_viea_spine": "SCF route validation consumes the materialized VIEA governance/failure/authority/resource view before routable implementations are trusted.",
        },
        "external_inference_calls": 0,
    }
    write_registry_table(resolve(args.db), payload)
    report_evidence_store.write_json_report(
        resolve(args.out),
        payload,
        markdown_path=resolve(args.markdown_out),
        markdown_text=render_markdown(payload),
        db_path=resolve(args.db),
    )
    if args.gate:
        print(json.dumps(registry_gate_summary(payload), indent=2))
    else:
        print(
            json.dumps(
                {
                    "policy": payload["policy"],
                    "created_utc": payload["created_utc"],
                    "trigger_state": payload["trigger_state"],
                    "summary": payload["summary"],
                    "top_unregistered": payload["unregistered"][:20],
                    "top_duplicate_families": payload["duplicate_families"][:20],
                    "routing_blockers": [
                        row for row in payload["routing_eligibility"] if not row.get("routing_eligible")
                    ][:20],
                    "cleanup_queue": payload["cleanup_queue"][:20],
                },
                indent=2,
            )
        )
    return 0 if trigger_state != "RED" else 2


def build_inventory(policy: dict[str, Any]) -> list[dict[str, Any]]:
    tracked = set(git_lines(["git", "ls-files", "--cached"]))
    other = set(git_lines(["git", "ls-files", "--others", "--exclude-standard"]))
    paths: dict[str, dict[str, Any]] = {}
    for raw in sorted(tracked | other):
        path = resolve(raw)
        if path.exists():
            row = safe_inventory_row(path, tracked_state="tracked" if raw in tracked else "untracked")
            if row is not None:
                paths[normalize_path(raw)] = row

    for raw in policy.get("root_files", []) if isinstance(policy.get("root_files"), list) else []:
        path = resolve(str(raw))
        if path.exists():
            row = safe_inventory_row(path, tracked_state="tracked" if str(raw) in tracked else "scanned")
            if row is not None:
                paths[normalize_path(str(raw))] = row

    for root_cfg in policy.get("inventory_roots", []) if isinstance(policy.get("inventory_roots"), list) else []:
        if not isinstance(root_cfg, dict):
            continue
        base = resolve(str(root_cfg.get("path") or ""))
        if not base.exists():
            continue
        recursive = bool(root_cfg.get("recursive"))
        iterator = base.rglob("*") if recursive else base.iterdir()
        if base.is_dir():
            row = safe_inventory_row(base, tracked_state="scanned")
            if row is not None:
                paths.setdefault(rel(base), row)
        for path in iterator:
            if should_skip_path(path):
                continue
            raw = rel(path)
            row = safe_inventory_row(path, tracked_state="tracked" if raw in tracked else ("untracked" if raw in other else "scanned"))
            if row is not None:
                paths[raw] = row

    return sorted(paths.values(), key=lambda row: str(row["path"]))


def safe_inventory_row(path: Path, *, tracked_state: str) -> dict[str, Any] | None:
    try:
        return inventory_row(path, tracked_state=tracked_state)
    except FileNotFoundError:
        return None


def inventory_row(path: Path, *, tracked_state: str) -> dict[str, Any]:
    is_file = path.is_file()
    stat = path.stat()
    return {
        "path": rel(path),
        "kind": "file" if is_file else "directory",
        "extension": path.suffix.lower() if is_file else "",
        "tracked_state": tracked_state,
        "bytes": int(stat.st_size) if is_file else 0,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def should_skip_path(path: Path) -> bool:
    parts = set(path.parts)
    if path.name == ".DS_Store":
        return True
    if "__pycache__" in parts or path.suffix == ".pyc":
        return True
    return False


def classify_inventory(inventory: list[dict[str, Any]], policy: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    surfaces = [surface for surface in policy.get("surfaces", []) if isinstance(surface, dict)]
    entries: list[dict[str, Any]] = []
    unregistered: list[dict[str, Any]] = []
    for row in inventory:
        path = str(row.get("path") or "")
        surface = match_surface(path, surfaces)
        if surface:
            entries.append(entry_for(row, surface))
            continue
        if is_active_source_path(path, row):
            unregistered.append(
                {
                    "record_type": "project_registry_gap",
                    "kind": "unregistered_active_source",
                    "path": path,
                    "extension": row.get("extension"),
                    "tracked_state": row.get("tracked_state"),
                    "recommended_action": "add this path to an existing registry surface, create a narrow owner surface, or move it under deprecated/generated state",
                }
            )
    return sorted(entries, key=lambda item: item["path"]), sorted(unregistered, key=lambda item: item["path"])


def match_surface(path: str, surfaces: list[dict[str, Any]]) -> dict[str, Any]:
    for surface in surfaces:
        for pattern in surface.get("patterns", []) if isinstance(surface.get("patterns"), list) else []:
            if fnmatch.fnmatch(path, normalize_path(str(pattern))):
                return surface
    return {}


def entry_for(row: dict[str, Any], surface: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_type": "project_registry_entry",
        "path": row["path"],
        "kind": row["kind"],
        "tracked_state": row["tracked_state"],
        "bytes": row["bytes"],
        "surface_id": str(surface.get("id") or "unknown"),
        "artifact_type": str(surface.get("artifact_type") or "unknown"),
        "role": str(surface.get("role") or "unknown"),
        "owner": str(surface.get("owner") or "unknown"),
        "status": str(surface.get("status") or "live"),
        "canonical": str(surface.get("canonical") or ""),
        "report_outputs": [str(item) for item in surface.get("report_outputs", []) if item],
        "verification_command": str(surface.get("verification_command") or ""),
        "cleanup_policy": str(surface.get("cleanup_policy") or ""),
    }


def is_active_source_path(path: str, row: dict[str, Any]) -> bool:
    if row.get("kind") != "file":
        return False
    if str(row.get("extension") or "").lower() not in SOURCE_EXTENSIONS:
        return False
    return any(path == root or path.startswith(root + "/") for root in SOURCE_ROOTS) or "/" not in path


def report_output_status(policy: dict[str, Any]) -> list[dict[str, Any]]:
    route_rows_by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for implementation in policy.get("implementations", []) if isinstance(policy.get("implementations"), list) else []:
        if not isinstance(implementation, dict):
            continue
        status = str(implementation.get("status") or "")
        eligibility = (
            implementation.get("routing_eligibility")
            if isinstance(implementation.get("routing_eligibility"), dict)
            else {}
        )
        routing_required = bool(eligibility.get("eligible", status == "live"))
        if status != "live" or not routing_required:
            continue
        contract_result = evaluate_route_evidence_contract(policy, implementation)
        for route_row in contract_result["requirements"]:
            path_text = str(route_row.get("path") or "")
            if path_text:
                route_rows_by_path[path_text].append(route_row)

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for surface in policy.get("surfaces", []) if isinstance(policy.get("surfaces"), list) else []:
        if not isinstance(surface, dict):
            continue
        surface_id = str(surface.get("id") or "unknown")
        default_max_age = float(surface.get("max_report_age_hours") or 24.0)
        report_output_policies = (
            surface.get("report_output_policies")
            if isinstance(surface.get("report_output_policies"), dict)
            else {}
        )
        for raw in surface.get("report_outputs", []) if isinstance(surface.get("report_outputs"), list) else []:
            path_text = normalize_path(str(raw))
            if not path_text or path_text in seen:
                continue
            seen.add(path_text)
            output_policy = (
                report_output_policies.get(path_text)
                if isinstance(report_output_policies.get(path_text), dict)
                else {}
            )
            max_age = float(output_policy.get("max_age_hours") or default_max_age)
            not_applicable = platform_report_not_applicable(path_text)
            path = resolve(path_text)
            exists = path.exists()
            created = report_created_utc(path) if exists and path.is_file() else None
            age = age_hours_since(created) if created else None
            route_rows = route_rows_by_path.get(path_text, [])
            route_blockers = [
                blocker
                for route_row in route_rows
                for blocker in route_row.get("blockers", [])
                if blocker
            ]
            route_required = bool(route_rows)
            stale = bool(route_required and any(bool(route_row.get("stale")) for route_row in route_rows))
            if not_applicable:
                status = "not_applicable"
            elif route_required:
                status = "fresh" if not route_blockers else route_evidence_failure_status(route_rows)
            else:
                status = "available" if exists else "missing_supporting"
            rows.append(
                {
                    "record_type": "project_registry_report_output",
                    "surface_id": surface_id,
                    "path": path_text,
                    "evidence_class": "route_required" if route_required else "supporting",
                    "route_required": route_required,
                    "route_required_by": sorted(
                        {str(route_row.get("implementation_id") or "") for route_row in route_rows if route_row.get("implementation_id")}
                    ),
                    "route_requirement_statuses": route_rows,
                    "route_blockers": sorted(set(route_blockers)),
                    "exists": exists,
                    "not_applicable": not_applicable,
                    "created_utc": created.isoformat().replace("+00:00", "Z") if created else "",
                    "age_hours": round(float(age), 3) if age is not None else None,
                    "max_age_hours": max_age,
                    "freshness_policy": (
                        "route_evidence_contract"
                        if route_required
                        else str(output_policy.get("policy") or surface.get("latest_view_policy") or "supporting_evidence_no_ttl")
                    ),
                    "stale": stale,
                    "status": status,
                }
            )
    return sorted(rows, key=lambda item: (item["status"], item["path"]))


def route_evidence_contracts(policy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("id") or ""): row
        for row in policy.get("route_evidence_contracts", [])
        if isinstance(row, dict) and row.get("id")
    }


def evaluate_route_evidence_contract(policy: dict[str, Any], implementation: dict[str, Any]) -> dict[str, Any]:
    implementation_id = str(implementation.get("id") or "unknown")
    contract_id = str(implementation.get("route_evidence_contract_id") or "")
    contract = route_evidence_contracts(policy).get(contract_id)
    if not contract:
        return {
            "contract_id": contract_id,
            "mode": "missing",
            "requirements": [],
            "blockers": ["route_evidence_contract_missing"],
        }
    mode = str(contract.get("mode") or "all")
    if mode == "current_invocation":
        return {
            "contract_id": contract_id,
            "mode": mode,
            "requirements": [
                {
                    "implementation_id": implementation_id,
                    "requirement_id": "current_invocation",
                    "path": "",
                    "freshness_mode": "current_invocation",
                    "status": "current_invocation",
                    "stale": False,
                    "blockers": [],
                }
            ],
            "blockers": [],
        }
    if mode == "not_route_required":
        return {
            "contract_id": contract_id,
            "mode": mode,
            "requirements": [],
            "blockers": [],
        }
    requirements = [
        evaluate_route_evidence_requirement(implementation, row)
        for row in contract.get("requirements", [])
        if isinstance(row, dict)
    ]
    blockers = sorted(
        {
            blocker
            for requirement in requirements
            for blocker in requirement.get("blockers", [])
            if blocker
        }
    )
    if not requirements:
        blockers.append("route_evidence_requirements_missing")
    return {
        "contract_id": contract_id,
        "mode": mode,
        "requirements": requirements,
        "blockers": blockers,
    }


def evaluate_route_evidence_requirement(
    implementation: dict[str, Any], requirement: dict[str, Any]
) -> dict[str, Any]:
    implementation_id = str(implementation.get("id") or "unknown")
    requirement_id = str(requirement.get("id") or "unnamed")
    path_text = normalize_path(str(requirement.get("path") or ""))
    freshness_mode = str(requirement.get("freshness_mode") or "source_bound")
    path = resolve(path_text) if path_text else ROOT / "__missing_route_evidence_path__"
    not_applicable = bool(path_text and platform_report_not_applicable(path_text))
    exists = bool(path_text and path.exists())
    created = report_created_utc(path) if exists and path.is_file() else None
    age = age_hours_since(created) if created else None
    blockers: list[str] = []
    source_paths: list[str] = []
    newest_source_utc = ""
    content_valid = False
    acceptance_passed = False
    acceptance_actual: Any = None

    if not path_text:
        blockers.append("route_evidence_path_missing")
    elif not_applicable:
        pass
    elif not exists:
        blockers.append("route_evidence_missing")
    else:
        payload = read_json(path, None)
        content_valid = isinstance(payload, (dict, list))
        if path.suffix.lower() == ".json" and not content_valid:
            blockers.append("route_evidence_invalid_json")
        if freshness_mode == "ttl":
            max_age = float(requirement.get("max_age_hours") or 0.0)
            if max_age <= 0:
                blockers.append("route_evidence_ttl_missing")
            elif age is None or age > max_age:
                blockers.append("route_evidence_ttl_expired")
        elif freshness_mode == "source_bound":
            source_paths = route_evidence_source_paths(implementation, requirement)
            source_times: list[datetime] = []
            for source_text in source_paths:
                source = resolve(source_text)
                if not source.exists():
                    blockers.append(f"route_evidence_source_missing:{source_text}")
                    continue
                try:
                    source_times.append(datetime.fromtimestamp(source.stat().st_mtime, tz=timezone.utc))
                except OSError:
                    blockers.append(f"route_evidence_source_unreadable:{source_text}")
            if not source_paths:
                blockers.append("route_evidence_sources_missing")
            if source_times:
                newest_source = max(source_times)
                newest_source_utc = newest_source.isoformat().replace("+00:00", "Z")
                if created is None or created.timestamp() + 1.0 < newest_source.timestamp():
                    blockers.append("route_evidence_source_changed")
        elif freshness_mode != "exists":
            blockers.append(f"route_evidence_unknown_freshness_mode:{freshness_mode}")

        acceptance = requirement.get("acceptance") if isinstance(requirement.get("acceptance"), dict) else {}
        if acceptance:
            field = str(acceptance.get("field") or "")
            allowed = acceptance.get("allowed") if isinstance(acceptance.get("allowed"), list) else []
            acceptance_actual = get_nested(payload, field.split("."), None) if field and isinstance(payload, dict) else None
            acceptance_passed = bool(field and allowed and acceptance_actual in allowed)
            if not acceptance_passed:
                blockers.append("route_evidence_acceptance_rejected")
        else:
            acceptance_passed = content_valid or path.suffix.lower() != ".json"

    stale = any(blocker in {"route_evidence_ttl_expired", "route_evidence_source_changed"} for blocker in blockers)
    status = "not_applicable" if not_applicable else ("fresh" if not blockers else route_evidence_failure_status([{"blockers": blockers}]))
    return {
        "record_type": "project_registry_route_evidence_requirement",
        "implementation_id": implementation_id,
        "requirement_id": requirement_id,
        "path": path_text,
        "freshness_mode": freshness_mode,
        "source_paths": source_paths,
        "newest_source_utc": newest_source_utc,
        "exists": exists,
        "not_applicable": not_applicable,
        "created_utc": created.isoformat().replace("+00:00", "Z") if created else "",
        "age_hours": round(float(age), 3) if age is not None else None,
        "content_valid": content_valid,
        "acceptance": requirement.get("acceptance", {}),
        "acceptance_actual": acceptance_actual,
        "acceptance_passed": acceptance_passed,
        "status": status,
        "stale": stale,
        "blockers": sorted(set(blockers)),
    }


def route_evidence_source_paths(implementation: dict[str, Any], requirement: dict[str, Any]) -> list[str]:
    explicit = requirement.get("source_paths") if isinstance(requirement.get("source_paths"), list) else []
    candidates = [str(item) for item in explicit if item]
    if not candidates:
        canonical = str(implementation.get("canonical_entrypoint") or "")
        if canonical:
            candidates.append(canonical)
        command = str(implementation.get("verification_command") or "")
        candidates.extend(
            re.findall(r"(?:scripts|configs|crates|src|tests)/[A-Za-z0-9_./-]+\.(?:py|json|toml|rs|md|ya?ml)", command)
        )
    return sorted({normalize_path(item) for item in candidates if item})


def route_evidence_failure_status(rows: list[dict[str, Any]]) -> str:
    blockers = {str(blocker) for row in rows for blocker in row.get("blockers", [])}
    if any("missing" in blocker for blocker in blockers):
        return "missing"
    if any("invalid" in blocker or "unknown" in blocker for blocker in blockers):
        return "invalid"
    if any("expired" in blocker or "source_changed" in blocker for blocker in blockers):
        return "stale"
    if any("acceptance_rejected" in blocker for blocker in blockers):
        return "rejected"
    return "blocked"


def platform_report_not_applicable(path_text: str) -> bool:
    if path_text.endswith("windows_cuda_doctor.json") and sys.platform != "win32":
        return True
    return False


def duplicate_family_report(inventory: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    thresholds = policy.get("duplicate_family_thresholds") if isinstance(policy.get("duplicate_family_thresholds"), dict) else {}
    by_root: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in inventory:
        path = str(row.get("path") or "")
        if row.get("kind") != "file":
            continue
        root = duplicate_root(path)
        if not root:
            continue
        family = duplicate_family_key(path)
        by_root[root][family].append(row)

    rows = []
    for root, families in sorted(by_root.items()):
        threshold = int(thresholds.get(root, 3 if root != "reports" else 12))
        for family, items in sorted(families.items(), key=lambda item: (-len(item[1]), item[0])):
            if len(items) < threshold:
                continue
            classification = duplicate_family_classification(policy, root, family)
            rows.append(
                {
                    "record_type": "project_registry_duplicate_family",
                    "root": root,
                    "family": family,
                    "count": len(items),
                    "threshold": threshold,
                    "classified": bool(classification),
                    "classification": classification,
                    "sample_paths": [str(item["path"]) for item in sorted(items, key=lambda row: str(row["path"]))[:16]],
                    "recommended_action": (
                        "follow the registry family classification and improve the canonical surface first"
                        if classification
                        else "pick a canonical owner, mark deliberate compatibility wrappers, and route generated variants through reports/archive instead of adding new source lanes"
                    ),
                }
            )
    return rows[:200]


def duplicate_family_classification(policy: dict[str, Any], root: str, family: str) -> dict[str, Any]:
    for row in policy.get("duplicate_family_classifications", []) if isinstance(policy.get("duplicate_family_classifications"), list) else []:
        if not isinstance(row, dict):
            continue
        family_pattern = str(row.get("family") or "")
        if str(row.get("root") or "") == root and (family_pattern == family or fnmatch.fnmatch(family, family_pattern)):
            return {
                "classification": str(row.get("classification") or ""),
                "canonical_surface": str(row.get("canonical_surface") or ""),
                "canonical_path": str(row.get("canonical_path") or ""),
                "promotion_role": str(row.get("promotion_role") or ""),
                "successor_policy": str(row.get("successor_policy") or ""),
            }
    return {}


def duplicate_root(path: str) -> str:
    for root in ("scripts", "configs", "reports", "docs"):
        if path.startswith(root + "/"):
            return root
    return ""


def duplicate_family_key(path: str) -> str:
    name = Path(path).name
    stem = Path(name).stem
    stem = re.sub(r"(_v|_seed|_after|_current|_goal|_final|_hardened|_private|_public|_smoke|_overnight|_run)[A-Za-z0-9_.-]*$", "", stem)
    stem = re.sub(r"_[0-9]{6,}.*$", "", stem)
    return stem or name


def generated_source_artifacts(inventory: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    patterns = [normalize_path(str(item)) for item in policy.get("generated_source_patterns", []) if item]
    rows = []
    for row in inventory:
        path = str(row.get("path") or "")
        if any(fnmatch.fnmatch(path, pattern) for pattern in patterns):
            rows.append(
                {
                    "record_type": "project_registry_gap",
                    "kind": "generated_artifact_in_source_path",
                    "path": path,
                    "tracked_state": row.get("tracked_state"),
                    "recommended_action": "quarantine generated cache/scratch files and add ignore coverage if needed",
                }
            )
    return rows


def steward_config_path(policy: dict[str, Any]) -> Path:
    configured = str(policy.get("project_steward_config") or "")
    return resolve(configured) if configured else DEFAULT_STEWARD_CONFIG


def steward_coverage_report(policy: dict[str, Any]) -> dict[str, Any]:
    path = steward_config_path(policy)
    payload = read_json(path, {})
    if not isinstance(payload, dict):
        payload = {}
    project_steward = payload.get("project_steward") if isinstance(payload.get("project_steward"), dict) else {}
    work_contracts = list_dicts(payload.get("project_work_contracts"))
    taint_records = list_dicts(payload.get("event_taint_records"))
    module_cards = list_dicts(payload.get("module_cards"))
    decisions = list_dicts(payload.get("steward_decisions"))
    major_surfaces = {
        str(surface.get("id") or "")
        for surface in policy.get("surfaces", [])
        if isinstance(surface, dict)
        and str(surface.get("status") or "") in {"live", "retained"}
        and bool(surface.get("major_surface", True))
        and surface.get("id")
    }
    module_surface_ids = {str(card.get("surface_id") or "") for card in module_cards if card.get("surface_id")}
    missing_major_surface_cards = sorted(major_surfaces - module_surface_ids)
    mandatory_taint_classes = [
        str(item)
        for item in payload.get(
            "mandatory_event_taint_classes",
            [
                "issue_or_pr_text",
                "benchmark_payload",
                "teacher_proposal",
                "browser_or_external_model_note",
                "generated_report",
                "worker_output",
                "public_calibration_artifact",
            ],
        )
        if item
    ]
    taint_classes = {str(row.get("source_class") or "") for row in taint_records}
    missing_taint_classes = sorted([item for item in mandatory_taint_classes if item not in taint_classes])
    active_work_contracts = [
        row
        for row in work_contracts
        if str(row.get("status") or "") in {"active", "ready", "in_progress", "queued"}
    ]

    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not path.exists():
        hard_gaps.append(steward_gap("missing_config", "Project steward config is missing.", {"path": rel(path)}))
    steward_missing = required_field_gaps(
        project_steward,
        [
            "id",
            "artifact_id",
            "mission",
            "non_goals",
            "authority_ceiling",
            "evidence_policy",
            "allowed_work",
            "forbidden_work",
            "sunset_criteria",
        ],
    )
    if steward_missing:
        hard_gaps.append(steward_gap("project_steward_missing_fields", "Project steward charter is incomplete.", {"missing_fields": steward_missing}))
    contract_gaps = record_schema_gaps(
        work_contracts,
        [
            "id",
            "objective",
            "roadmap_refs",
            "allowed_files",
            "allowed_tools",
            "forbidden_tools",
            "required_outputs",
            "acceptance_tests",
            "evidence_requirements",
            "authority_ceiling",
            "rollback_path",
            "non_claims",
            "status",
        ],
        "project_work_contract",
    )
    if contract_gaps:
        hard_gaps.append(steward_gap("work_contract_schema_gaps", "One or more project work contracts are incomplete.", {"records": contract_gaps[:12]}))
    if not active_work_contracts:
        hard_gaps.append(steward_gap("no_active_work_contract", "No active project work contract exists for the next roadmap task.", {"contract_count": len(work_contracts)}))
    taint_gaps = record_schema_gaps(
        taint_records,
        [
            "id",
            "source_class",
            "taint_state",
            "trusted_fields",
            "untrusted_fields",
            "forbidden_control_uses",
            "sanitization_route",
            "training_use_policy",
            "residuals",
        ],
        "event_taint_record",
    )
    if taint_gaps:
        hard_gaps.append(steward_gap("event_taint_schema_gaps", "One or more event-taint records are incomplete.", {"records": taint_gaps[:12]}))
    if missing_taint_classes:
        hard_gaps.append(steward_gap("missing_mandatory_taint_classes", "Mandatory event source classes lack taint records.", {"missing_classes": missing_taint_classes}))
    module_gaps = record_schema_gaps(
        module_cards,
        [
            "id",
            "surface_id",
            "problem",
            "interface",
            "invariants",
            "failure_modes",
            "minimal_implementation",
            "validation_commands",
            "evidence_refs",
            "non_claims",
            "deprecation_route",
        ],
        "module_card",
    )
    if module_gaps:
        hard_gaps.append(steward_gap("module_card_schema_gaps", "One or more module cards are incomplete.", {"records": module_gaps[:12]}))
    if missing_major_surface_cards:
        warnings.append(
            steward_gap(
                "module_card_coverage_incomplete",
                "Not every live major surface has a module DoD card yet.",
                {"missing_surface_ids": missing_major_surface_cards[:40], "missing_count": len(missing_major_surface_cards)},
            )
        )
    decision_gaps = record_schema_gaps(
        decisions,
        [
            "id",
            "target",
            "decision",
            "reason",
            "evidence_refs",
            "allowed_next_actions",
            "status",
        ],
        "steward_decision",
    )
    if decision_gaps:
        hard_gaps.append(steward_gap("steward_decision_schema_gaps", "One or more steward decisions are incomplete.", {"records": decision_gaps[:12]}))
    if len(decisions) < 5:
        warnings.append(steward_gap("steward_decision_floor", "Fewer than five stale/report/source-family steward decisions are recorded.", {"decision_count": len(decisions)}))

    status = "RED" if hard_gaps else ("YELLOW" if warnings else "GREEN")
    return {
        "record_type": "project_steward_coverage",
        "config_path": rel(path),
        "exists": path.exists(),
        "status": status,
        "project_steward_id": str(project_steward.get("id") or ""),
        "work_contract_count": len(work_contracts),
        "active_work_contract_count": len(active_work_contracts),
        "event_taint_record_count": len(taint_records),
        "mandatory_event_taint_class_count": len(mandatory_taint_classes),
        "missing_mandatory_taint_classes": missing_taint_classes,
        "module_card_count": len(module_cards),
        "major_surface_count": len(major_surfaces),
        "major_surface_module_card_coverage_ratio": round(len(major_surfaces & module_surface_ids) / max(1, len(major_surfaces)), 6),
        "missing_major_surface_module_cards": missing_major_surface_cards,
        "steward_decision_count": len(decisions),
        "hard_gap_count": len(hard_gaps),
        "warning_count": len(warnings),
        "hard_gaps": hard_gaps,
        "warnings": warnings,
        "active_work_contracts": [
            {
                "id": row.get("id"),
                "objective": row.get("objective"),
                "roadmap_refs": row.get("roadmap_refs", []),
                "status": row.get("status"),
                "authority_ceiling": row.get("authority_ceiling", ""),
            }
            for row in active_work_contracts[:12]
        ],
        "steward_decisions": [
            {
                "id": row.get("id"),
                "target": row.get("target"),
                "decision": row.get("decision"),
                "status": row.get("status"),
                "source_queue_ids": [str(item) for item in list_values(row.get("source_queue_ids"))],
                "cleanup_kinds": [str(item) for item in list_values(row.get("cleanup_kinds"))],
                "cleanup_scopes": [str(item) for item in list_values(row.get("cleanup_scopes"))],
            }
            for row in decisions[:80]
        ],
    }


def list_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def list_values(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def cleanup_scope_matches(pattern: str, scope: str) -> bool:
    if pattern == "*":
        return True
    if pattern.endswith("*"):
        return scope.startswith(pattern[:-1])
    return pattern == scope


def steward_decision_covers_cleanup(decision: dict[str, Any], item: dict[str, Any]) -> bool:
    if str(decision.get("status") or "") != "active":
        return False
    queue_id = str(item.get("queue_id") or "")
    if queue_id and queue_id in {str(value) for value in list_values(decision.get("source_queue_ids"))}:
        return True
    kinds = {str(value) for value in list_values(decision.get("cleanup_kinds")) if value}
    if kinds and str(item.get("kind") or "") not in kinds:
        return False
    patterns = [str(value) for value in list_values(decision.get("cleanup_scopes")) if value]
    if not patterns:
        return False
    scopes = [str(value) for value in list_values(item.get("scope")) if value]
    return any(cleanup_scope_matches(pattern, scope) for pattern in patterns for scope in scopes)


def cleanup_queue_steward_coverage(
    cleanup_queue: list[dict[str, Any]],
    steward_decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    covered_ids: set[str] = set()
    uncovered_ids: list[str] = []
    for item in cleanup_queue:
        queue_id = str(item.get("queue_id") or "")
        if not queue_id:
            continue
        if any(steward_decision_covers_cleanup(decision, item) for decision in steward_decisions):
            covered_ids.add(queue_id)
        else:
            uncovered_ids.append(queue_id)
    return {
        "covered_ids": sorted(covered_ids),
        "uncovered_ids": sorted(uncovered_ids),
        "coverage_ratio": round(len(covered_ids) / max(1, len(covered_ids) + len(uncovered_ids)), 6),
    }


def record_schema_gaps(rows: list[dict[str, Any]], required: list[str], record_type: str) -> list[dict[str, Any]]:
    gaps = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        record_id = str(row.get("id") or f"index:{index}")
        missing = required_field_gaps(row, required)
        if record_id in seen_ids:
            missing = sorted(set(missing + ["unique_id"]))
        seen_ids.add(record_id)
        if missing:
            gaps.append({"record_type": record_type, "id": record_id, "missing_fields": missing})
    return gaps


def steward_gap(kind: str, message: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_type": "project_steward_gap",
        "kind": kind,
        "message": message,
        "evidence": evidence,
        "recommended_action": "Update configs/project_steward.json and rerun python3 scripts/theseus_project_registry.py --gate.",
    }


def registry_governance_violations(
    policy: dict[str, Any],
    entries: list[dict[str, Any]],
    unregistered: list[dict[str, Any]],
    abstraction_gaps: list[dict[str, Any]],
    duplicate_families: list[dict[str, Any]],
    report_status: list[dict[str, Any]],
    generated_source: list[dict[str, Any]],
    implementation_health: list[dict[str, Any]],
    steward_coverage: dict[str, Any],
) -> list[dict[str, Any]]:
    contract = policy.get("registry_evolution_contract") if isinstance(policy.get("registry_evolution_contract"), dict) else {}
    rules = {
        str(row.get("id") or ""): row
        for row in contract.get("rules", [])
        if isinstance(row, dict) and row.get("id")
    }
    rows: list[dict[str, Any]] = []
    if unregistered:
        rows.append(
            governance_violation(
                rules,
                "registry_first",
                "unregistered_active_sources",
                [str(row.get("path") or "") for row in unregistered[:24]],
                {"count": len(unregistered), "sample": unregistered[:24]},
            )
        )
    source_duplicates = [
        row
        for row in duplicate_families
        if str(row.get("root") or "") in {"scripts", "configs", "docs"} and not row.get("classified")
    ]
    if source_duplicates:
        rows.append(
            governance_violation(
                rules,
                "improve_existing_first",
                "source_duplicate_families",
                [f"{row.get('root')}/{row.get('family')}" for row in source_duplicates[:24]],
                {"count": len(source_duplicates), "families": source_duplicates[:24]},
            )
        )
    stale_or_missing = [
        row
        for row in report_status
        if isinstance(row, dict)
        and row.get("route_required")
        and row.get("status") in {"stale", "missing", "invalid", "rejected", "blocked"}
        and not row.get("not_applicable")
    ]
    if stale_or_missing:
        rows.append(
            governance_violation(
                rules,
                "single_current_self_model",
                "blocked_route_evidence",
                [str(row.get("path") or "") for row in stale_or_missing[:24]],
                {"count": len(stale_or_missing), "reports": stale_or_missing[:24]},
            )
        )
    if generated_source:
        rows.append(
            governance_violation(
                rules,
                "no_orphan_runtime_state",
                "generated_artifacts_in_source_paths",
                [str(row.get("path") or "") for row in generated_source[:24]],
                {"count": len(generated_source), "sample": generated_source[:24]},
            )
        )
    if abstraction_gaps:
        rows.append(
            governance_violation(
                rules,
                "abstraction_boundary",
                "abstraction_registry_gaps",
                [str(row.get("scope") or row.get("id") or row.get("abstraction_id") or row.get("implementation_id") or "") for row in abstraction_gaps[:24]],
                {"count": len(abstraction_gaps), "gaps": abstraction_gaps[:24]},
            )
        )
    if steward_coverage.get("status") == "RED":
        rows.append(
            governance_violation(
                rules,
                "project_steward_boundary",
                "project_steward_coverage_red",
                [str(steward_coverage.get("config_path") or "configs/project_steward.json")],
                {
                    "status": steward_coverage.get("status"),
                    "hard_gap_count": steward_coverage.get("hard_gap_count"),
                    "hard_gaps": steward_coverage.get("hard_gaps", [])[:24],
                },
            )
        )
    blocked_implementations = [row for row in implementation_health if row.get("routing_required") and not row.get("routing_eligible")]
    if blocked_implementations:
        rows.append(
            governance_violation(
                rules,
                "abstraction_boundary",
                "implementation_routing_health_gaps",
                [str(row.get("implementation_id") or "") for row in blocked_implementations[:24]],
                {"count": len(blocked_implementations), "implementations": blocked_implementations[:24]},
            )
        )
    schema_gaps = surface_schema_gaps(policy)
    if schema_gaps:
        rows.append(
            governance_violation(
                rules,
                "registry_first",
                "surface_schema_gaps",
                [str(row.get("surface_id") or "") for row in schema_gaps[:24]],
                {"count": len(schema_gaps), "surfaces": schema_gaps[:24]},
            )
        )
    retained_without_successor = retained_surface_gaps(entries)
    if retained_without_successor:
        rows.append(
            governance_violation(
                rules,
                "successor_not_sprawl",
                "retained_surfaces_need_successor_or_retirement",
                [str(row.get("path") or "") for row in retained_without_successor[:24]],
                {"count": len(retained_without_successor), "sample": retained_without_successor[:24]},
            )
        )
    return rows


def governance_violation(
    rules: dict[str, dict[str, Any]],
    rule_id: str,
    kind: str,
    scope: list[str],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    rule = rules.get(rule_id, {})
    return {
        "record_type": "project_registry_governance_violation",
        "rule_id": rule_id,
        "kind": kind,
        "severity": str(rule.get("severity") or "medium"),
        "rule": str(rule.get("rule") or ""),
        "enforcement": str(rule.get("enforcement") or ""),
        "scope": [item for item in scope if item][:24],
        "recommended_action": recommended_action_for_rule(rule_id),
        "evidence": evidence,
    }


def recommended_action_for_rule(rule_id: str) -> str:
    actions = {
        "registry_first": "Register the file under an existing surface, create a narrow surface with required fields, or move it to deprecated/generated state.",
        "improve_existing_first": "Patch the canonical surface first; if a successor is truly needed, record the successor/deprecation relationship in the registry.",
        "single_current_self_model": "Refresh or repair the minimal blocked route receipt; supporting history does not require periodic regeneration.",
        "no_orphan_runtime_state": "Move generated/cache/build artifacts out of source paths through GC or retention manifests and keep ignore coverage current.",
        "successor_not_sprawl": "Add a replacement, retained reason, or cleanup policy so old and new systems do not compete silently.",
        "no_evidence_deletion_without_manifest": "Use manifest-backed retention/GC and leave archive pointers for moved evidence.",
        "abstraction_boundary": "Bind implementations to declared abstractions, give each live abstraction a canonical live implementation, and keep replacements registry-visible.",
        "project_steward_boundary": "Complete the project steward charter, active work contract, event-taint records, module cards, and steward decisions in configs/project_steward.json.",
    }
    return actions.get(rule_id, "Resolve the registry rule violation and rerun the project registry.")


def abstraction_registry_gaps(policy: dict[str, Any]) -> list[dict[str, Any]]:
    contract = policy.get("abstraction_registry_contract") if isinstance(policy.get("abstraction_registry_contract"), dict) else {}
    abstraction_required = [str(item) for item in contract.get("abstraction_required_fields", []) if item]
    implementation_required = [str(item) for item in contract.get("implementation_required_fields", []) if item]
    valid_states = {str(item) for item in contract.get("valid_lifecycle_states", []) if item}
    abstractions = [row for row in policy.get("abstractions", []) if isinstance(row, dict)]
    implementations = [row for row in policy.get("implementations", []) if isinstance(row, dict)]
    route_contract_rows = [row for row in policy.get("route_evidence_contracts", []) if isinstance(row, dict)]
    route_contract_by_id: dict[str, dict[str, Any]] = {}
    surfaces = {
        str(row.get("id") or ""): row
        for row in policy.get("surfaces", [])
        if isinstance(row, dict) and row.get("id")
    }
    abstraction_by_id: dict[str, dict[str, Any]] = {}
    implementation_by_id: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for route_contract in route_contract_rows:
        route_contract_id = str(route_contract.get("id") or "")
        mode = str(route_contract.get("mode") or "")
        if not route_contract_id:
            rows.append(
                {
                    "record_type": "project_registry_route_evidence_gap",
                    "kind": "route_evidence_contract_id_missing",
                    "scope": "route_evidence_contracts",
                    "recommended_action": "give every route evidence contract a stable id",
                }
            )
            continue
        if route_contract_id in route_contract_by_id:
            rows.append(
                {
                    "record_type": "project_registry_route_evidence_gap",
                    "kind": "duplicate_route_evidence_contract_id",
                    "route_evidence_contract_id": route_contract_id,
                    "scope": route_contract_id,
                    "recommended_action": "merge duplicate route evidence contracts",
                }
            )
        route_contract_by_id[route_contract_id] = route_contract
        if mode not in ROUTE_EVIDENCE_MODES:
            rows.append(
                {
                    "record_type": "project_registry_route_evidence_gap",
                    "kind": "route_evidence_contract_mode_invalid",
                    "route_evidence_contract_id": route_contract_id,
                    "mode": mode,
                    "scope": route_contract_id,
                    "recommended_action": "use an allowed route evidence evaluation mode",
                }
            )
        requirements = [row for row in route_contract.get("requirements", []) if isinstance(row, dict)]
        if mode == "all" and not requirements:
            rows.append(
                {
                    "record_type": "project_registry_route_evidence_gap",
                    "kind": "route_evidence_requirements_missing",
                    "route_evidence_contract_id": route_contract_id,
                    "scope": route_contract_id,
                    "recommended_action": "declare at least one minimal route receipt",
                }
            )
        requirement_ids: set[str] = set()
        for requirement in requirements:
            requirement_id = str(requirement.get("id") or "")
            path_text = normalize_path(str(requirement.get("path") or ""))
            freshness_mode = str(requirement.get("freshness_mode") or "")
            missing_fields = [
                field for field, value in (("id", requirement_id), ("path", path_text), ("freshness_mode", freshness_mode)) if not value
            ]
            if missing_fields:
                rows.append(
                    {
                        "record_type": "project_registry_route_evidence_gap",
                        "kind": "route_evidence_requirement_missing_fields",
                        "route_evidence_contract_id": route_contract_id,
                        "requirement_id": requirement_id,
                        "missing_fields": missing_fields,
                        "scope": route_contract_id,
                        "recommended_action": "complete the route evidence requirement schema",
                    }
                )
            if requirement_id in requirement_ids:
                rows.append(
                    {
                        "record_type": "project_registry_route_evidence_gap",
                        "kind": "duplicate_route_evidence_requirement_id",
                        "route_evidence_contract_id": route_contract_id,
                        "requirement_id": requirement_id,
                        "scope": route_contract_id,
                        "recommended_action": "give requirements unique ids within the contract",
                    }
                )
            requirement_ids.add(requirement_id)
            if freshness_mode and freshness_mode not in ROUTE_EVIDENCE_FRESHNESS_MODES:
                rows.append(
                    {
                        "record_type": "project_registry_route_evidence_gap",
                        "kind": "route_evidence_freshness_mode_invalid",
                        "route_evidence_contract_id": route_contract_id,
                        "requirement_id": requirement_id,
                        "freshness_mode": freshness_mode,
                        "scope": route_contract_id,
                        "recommended_action": "use source_bound, ttl, or exists freshness",
                    }
                )
            if freshness_mode == "ttl" and float(requirement.get("max_age_hours") or 0.0) <= 0:
                rows.append(
                    {
                        "record_type": "project_registry_route_evidence_gap",
                        "kind": "route_evidence_ttl_missing",
                        "route_evidence_contract_id": route_contract_id,
                        "requirement_id": requirement_id,
                        "scope": route_contract_id,
                        "recommended_action": "declare a positive TTL for volatile evidence",
                    }
                )

    for abstraction in abstractions:
        abstraction_id = str(abstraction.get("id") or "")
        abstraction_status = str(abstraction.get("status") or "")
        if valid_states and abstraction_status and abstraction_status not in valid_states:
            rows.append(
                {
                    "record_type": "project_registry_abstraction_gap",
                    "kind": "abstraction_unknown_lifecycle_state",
                    "abstraction_id": abstraction_id or "unknown",
                    "status": abstraction_status,
                    "scope": abstraction_id or "unknown",
                    "recommended_action": "use a lifecycle state declared by abstraction_registry_contract.valid_lifecycle_states",
                }
            )
        missing = required_field_gaps(abstraction, abstraction_required)
        if missing:
            rows.append(
                {
                    "record_type": "project_registry_abstraction_gap",
                    "kind": "abstraction_missing_required_fields",
                    "abstraction_id": abstraction_id or "unknown",
                    "scope": abstraction_id or "unknown",
                    "missing_fields": missing,
                    "recommended_action": "complete the abstraction contract before treating it as part of Theseus' stable self-model",
                }
            )
        if abstraction_id:
            if abstraction_id in abstraction_by_id:
                rows.append(
                    {
                        "record_type": "project_registry_abstraction_gap",
                        "kind": "duplicate_abstraction_id",
                        "abstraction_id": abstraction_id,
                        "scope": abstraction_id,
                        "recommended_action": "merge or rename duplicate abstraction ids so one stable contract owns the behavior",
                    }
                )
            abstraction_by_id[abstraction_id] = abstraction
        for surface_id in abstraction.get("related_surfaces", []) if isinstance(abstraction.get("related_surfaces"), list) else []:
            if str(surface_id) not in surfaces:
                rows.append(
                    {
                        "record_type": "project_registry_abstraction_gap",
                        "kind": "abstraction_references_unknown_surface",
                        "abstraction_id": abstraction_id or "unknown",
                        "surface_id": str(surface_id),
                        "scope": abstraction_id or str(surface_id),
                        "recommended_action": "remove the stale related surface or register the surface before claiming it belongs to this abstraction",
                    }
                )

    for implementation in implementations:
        implementation_id = str(implementation.get("id") or "")
        implementation_status = str(implementation.get("status") or "")
        if valid_states and implementation_status and implementation_status not in valid_states:
            rows.append(
                {
                    "record_type": "project_registry_implementation_gap",
                    "kind": "implementation_unknown_lifecycle_state",
                    "implementation_id": implementation_id or "unknown",
                    "status": implementation_status,
                    "scope": implementation_id or "unknown",
                    "recommended_action": "use a lifecycle state declared by abstraction_registry_contract.valid_lifecycle_states",
                }
            )
        missing = required_field_gaps(implementation, implementation_required)
        if missing:
            rows.append(
                {
                    "record_type": "project_registry_implementation_gap",
                    "kind": "implementation_missing_required_fields",
                    "implementation_id": implementation_id or "unknown",
                    "scope": implementation_id or "unknown",
                    "missing_fields": missing,
                    "recommended_action": "complete the implementation binding before it can satisfy an abstraction",
                }
            )
        if implementation_id:
            if implementation_id in implementation_by_id:
                rows.append(
                    {
                        "record_type": "project_registry_implementation_gap",
                        "kind": "duplicate_implementation_id",
                        "implementation_id": implementation_id,
                        "scope": implementation_id,
                        "recommended_action": "merge or rename duplicate implementation ids so one implementation owns the route",
                    }
                )
            implementation_by_id[implementation_id] = implementation

        abstraction_id = str(implementation.get("abstraction_id") or "")
        surface_id = str(implementation.get("surface_id") or "")
        surface = surfaces.get(surface_id, {})
        abstraction = abstraction_by_id.get(abstraction_id, {})
        if abstraction_id and abstraction_id not in abstraction_by_id:
            rows.append(
                {
                    "record_type": "project_registry_implementation_gap",
                    "kind": "implementation_references_unknown_abstraction",
                    "implementation_id": implementation_id or "unknown",
                    "abstraction_id": abstraction_id,
                    "scope": implementation_id or abstraction_id,
                    "recommended_action": "declare the abstraction first or bind this implementation to the existing abstraction it actually satisfies",
                }
            )
        if surface_id and surface_id not in surfaces:
            rows.append(
                {
                    "record_type": "project_registry_implementation_gap",
                    "kind": "implementation_references_unknown_surface",
                    "implementation_id": implementation_id or "unknown",
                    "surface_id": surface_id,
                    "scope": implementation_id or surface_id,
                    "recommended_action": "register the owning surface or bind this implementation to the correct existing surface",
                }
            )
        if implementation_status == "live" and surface and str(surface.get("status") or "") not in {"live", "retained"}:
            rows.append(
                {
                    "record_type": "project_registry_implementation_gap",
                    "kind": "live_implementation_on_non_live_surface",
                    "implementation_id": implementation_id or "unknown",
                    "surface_id": surface_id,
                    "surface_status": str(surface.get("status") or ""),
                    "scope": implementation_id or surface_id,
                    "recommended_action": "make the surface live/retained with a reason or demote the implementation status",
                }
            )
        if implementation_status == "live" and abstraction and str(abstraction.get("status") or "") not in {"live", "retained"}:
            rows.append(
                {
                    "record_type": "project_registry_implementation_gap",
                    "kind": "live_implementation_on_non_live_abstraction",
                    "implementation_id": implementation_id or "unknown",
                    "abstraction_id": abstraction_id,
                    "abstraction_status": str(abstraction.get("status") or ""),
                    "scope": implementation_id or abstraction_id,
                    "recommended_action": "make the abstraction live/retained with a reason or demote the implementation status",
                }
            )
        eligibility = implementation.get("routing_eligibility") if isinstance(implementation.get("routing_eligibility"), dict) else {}
        route_contract_id = str(implementation.get("route_evidence_contract_id") or "")
        route_contract = route_contract_by_id.get(route_contract_id, {})
        if not route_contract_id or not route_contract:
            rows.append(
                {
                    "record_type": "project_registry_implementation_gap",
                    "kind": "implementation_route_evidence_contract_missing",
                    "implementation_id": implementation_id or "unknown",
                    "route_evidence_contract_id": route_contract_id,
                    "scope": implementation_id or "unknown",
                    "recommended_action": "bind the implementation to a declared route evidence contract",
                }
            )
        elif bool(eligibility.get("eligible", implementation_status == "live")) and route_contract.get("mode") == "not_route_required":
            rows.append(
                {
                    "record_type": "project_registry_implementation_gap",
                    "kind": "routable_implementation_disables_route_evidence",
                    "implementation_id": implementation_id or "unknown",
                    "route_evidence_contract_id": route_contract_id,
                    "scope": implementation_id or "unknown",
                    "recommended_action": "give routable implementations current-invocation or explicit route evidence",
                }
            )
        elif route_contract:
            evidence_outputs = {normalize_path(str(item)) for item in implementation.get("evidence_outputs", []) if item}
            surface_outputs = {
                normalize_path(str(item)) for item in surface.get("report_outputs", []) if item
            } if surface else set()
            for requirement in route_contract.get("requirements", []) if isinstance(route_contract.get("requirements"), list) else []:
                if not isinstance(requirement, dict):
                    continue
                path_text = normalize_path(str(requirement.get("path") or ""))
                if path_text and path_text not in evidence_outputs:
                    rows.append(
                        {
                            "record_type": "project_registry_implementation_gap",
                            "kind": "route_evidence_not_declared_by_implementation",
                            "implementation_id": implementation_id or "unknown",
                            "path": path_text,
                            "scope": implementation_id or path_text,
                            "recommended_action": "add the route receipt to implementation evidence_outputs",
                        }
                    )
                if path_text and path_text not in surface_outputs:
                    rows.append(
                        {
                            "record_type": "project_registry_implementation_gap",
                            "kind": "route_evidence_not_declared_by_surface",
                            "implementation_id": implementation_id or "unknown",
                            "surface_id": surface_id,
                            "path": path_text,
                            "scope": implementation_id or path_text,
                            "recommended_action": "add the route receipt to the owning surface report_outputs",
                        }
                    )
        if implementation_status == "live" and not eligibility:
            rows.append(
                {
                    "record_type": "project_registry_implementation_gap",
                    "kind": "live_implementation_missing_routing_eligibility",
                    "implementation_id": implementation_id or "unknown",
                    "scope": implementation_id or "unknown",
                    "recommended_action": "declare routing eligibility so routers can select or reject this implementation safely",
                }
            )
        if eligibility and not isinstance(eligibility.get("roles"), list):
            rows.append(
                {
                    "record_type": "project_registry_implementation_gap",
                    "kind": "implementation_routing_roles_not_list",
                    "implementation_id": implementation_id or "unknown",
                    "scope": implementation_id or "unknown",
                    "recommended_action": "make routing_eligibility.roles a list of declared routing roles",
                }
            )

    for surface_id, surface in sorted(surfaces.items()):
        surface_status = str(surface.get("status") or "")
        if valid_states and surface_status and surface_status not in valid_states:
            rows.append(
                {
                    "record_type": "project_registry_surface_gap",
                    "kind": "surface_unknown_lifecycle_state",
                    "surface_id": surface_id,
                    "status": surface_status,
                    "scope": surface_id,
                    "recommended_action": "use a lifecycle state declared by abstraction_registry_contract.valid_lifecycle_states",
                }
            )
        if surface_status == "live" and surface.get("major_surface", True):
            abstraction_id = str(surface.get("abstraction_id") or "")
            if not abstraction_id:
                rows.append(
                    {
                        "record_type": "project_registry_surface_gap",
                        "kind": "live_major_surface_missing_abstraction_binding",
                        "surface_id": surface_id,
                        "scope": surface_id,
                        "recommended_action": "bind this live surface to the abstraction contract it serves or mark it non-major with a reason",
                    }
                )
            elif abstraction_id not in abstraction_by_id:
                rows.append(
                    {
                        "record_type": "project_registry_surface_gap",
                        "kind": "surface_references_unknown_abstraction",
                        "surface_id": surface_id,
                        "abstraction_id": abstraction_id,
                        "scope": surface_id,
                        "recommended_action": "bind the surface to a declared abstraction or create the abstraction with a complete contract",
                    }
                )

    canonical_for: dict[str, list[str]] = defaultdict(list)
    live_impls_by_abstraction: dict[str, list[str]] = defaultdict(list)
    for implementation in implementations:
        abstraction_id = str(implementation.get("abstraction_id") or "")
        implementation_id = str(implementation.get("id") or "")
        if abstraction_id and str(implementation.get("status") or "") == "live":
            live_impls_by_abstraction[abstraction_id].append(implementation_id)
        if abstraction_id and implementation_id:
            abstraction = abstraction_by_id.get(abstraction_id, {})
            if str(abstraction.get("canonical_implementation_id") or "") == implementation_id:
                canonical_for[abstraction_id].append(implementation_id)

    for abstraction in abstractions:
        abstraction_id = str(abstraction.get("id") or "")
        if not abstraction_id:
            continue
        canonical_id = str(abstraction.get("canonical_implementation_id") or "")
        abstraction_status = str(abstraction.get("status") or "")
        canonical = implementation_by_id.get(canonical_id, {})
        if abstraction_status == "live":
            if not canonical_id:
                rows.append(
                    {
                        "record_type": "project_registry_abstraction_gap",
                        "kind": "live_abstraction_missing_canonical_implementation",
                        "abstraction_id": abstraction_id,
                        "scope": abstraction_id,
                        "recommended_action": "select one canonical implementation or demote the abstraction until it is implemented",
                    }
                )
            elif canonical_id not in implementation_by_id:
                rows.append(
                    {
                        "record_type": "project_registry_abstraction_gap",
                        "kind": "canonical_implementation_missing",
                        "abstraction_id": abstraction_id,
                        "implementation_id": canonical_id,
                        "scope": abstraction_id,
                        "recommended_action": "add the canonical implementation binding or point the abstraction at an existing live implementation",
                    }
                )
            elif str(canonical.get("status") or "") != "live":
                rows.append(
                    {
                        "record_type": "project_registry_abstraction_gap",
                        "kind": "canonical_implementation_not_live",
                        "abstraction_id": abstraction_id,
                        "implementation_id": canonical_id,
                        "implementation_status": str(canonical.get("status") or ""),
                        "scope": abstraction_id,
                        "recommended_action": "promote a live implementation or mark this abstraction retained/deprecated",
                    }
                )
        if abstraction_status == "live" and not live_impls_by_abstraction.get(abstraction_id):
            rows.append(
                {
                    "record_type": "project_registry_abstraction_gap",
                    "kind": "live_abstraction_has_no_live_implementations",
                    "abstraction_id": abstraction_id,
                    "scope": abstraction_id,
                    "recommended_action": "bind at least one live implementation or demote the abstraction status",
                }
            )
        if len(canonical_for.get(abstraction_id, [])) > 1:
            rows.append(
                {
                    "record_type": "project_registry_abstraction_gap",
                    "kind": "multiple_canonical_implementations",
                    "abstraction_id": abstraction_id,
                    "implementation_ids": canonical_for[abstraction_id],
                    "scope": abstraction_id,
                    "recommended_action": "choose one canonical implementation or declare a split route explicitly in the abstraction contract",
                }
            )
    return rows


def required_field_gaps(row: dict[str, Any], required: list[str]) -> list[str]:
    missing = []
    for field in required:
        value = row.get(field)
        if value in (None, "", [], {}):
            missing.append(field)
    return missing


def missing_true_gaps(row: dict[str, Any], required_paths: list[list[str]]) -> list[str]:
    missing = []
    for path in required_paths:
        if not get_nested(row, path, False):
            missing.append(".".join(path))
    return missing


def route_validator_materialized_view_receipt() -> dict[str, Any]:
    return viea_spine_records.materialized_view_consumer_receipt(
        "project_registry_route_validator",
        required_groups=list(ROUTE_VALIDATOR_SPINE_GROUPS),
    )


def stable_capability_field_gaps(policy: dict[str, Any], route_validator_spine_receipt: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Validate SCF contract/binding records without doing domain work.

    Stable Capability Fields are a semantic ABI. This validator intentionally
    checks mediation metadata only: contract dimensions, authority/state
    boundaries, exact content identity, evidence/claim hooks, contextual route
    validation, leases, adaptation envelopes, migration solvency, evaluator
    overlap, and lifecycle/governance controls.
    """

    contract = policy.get("stable_capability_field_contract") if isinstance(policy.get("stable_capability_field_contract"), dict) else {}
    if not contract:
        return [
            {
                "record_type": "project_registry_stable_capability_field_gap",
                "kind": "stable_capability_field_contract_missing",
                "scope": "stable_capability_field_contract",
                "recommended_action": "declare the SCF registry contract so abstractions are governed as semantic fields, not flat registry names",
            }
        ]
    field_required = [str(item) for item in contract.get("field_required_fields", []) if item]
    binding_required = [str(item) for item in contract.get("implementation_binding_required_fields", []) if item]
    section_required = {
        str(key): [str(item) for item in value if item]
        for key, value in contract.get("field_section_required_fields", {}).items()
        if isinstance(value, list)
    } if isinstance(contract.get("field_section_required_fields"), dict) else {}
    binding_section_required = {
        str(key): [str(item) for item in value if item]
        for key, value in contract.get("implementation_binding_section_required_fields", {}).items()
        if isinstance(value, list)
    } if isinstance(contract.get("implementation_binding_section_required_fields"), dict) else {}

    abstractions = [row for row in policy.get("abstractions", []) if isinstance(row, dict)]
    implementations = [row for row in policy.get("implementations", []) if isinstance(row, dict)]
    abstraction_by_id = {str(row.get("id") or ""): row for row in abstractions if row.get("id")}
    rows: list[dict[str, Any]] = []
    spine_receipt = route_validator_spine_receipt if isinstance(route_validator_spine_receipt, dict) else route_validator_materialized_view_receipt()
    if not spine_receipt.get("ready"):
        rows.append(
            scf_gap(
                "route_validator_viea_spine_view_not_ready",
                "project_registry_route_validator",
                [f"reports/viea_spine_materialized_view.json:{item}" for item in spine_receipt.get("missing_required_groups", [])],
                "run the VIEA spine record gate and require governance, failure, authority, and resource route records before approving routable SCF implementations",
                abstraction_id="project_self_model_registry",
            )
        )

    for abstraction in abstractions:
        abstraction_id = str(abstraction.get("id") or "unknown")
        status = str(abstraction.get("status") or "")
        if status not in {"live", "retained"}:
            continue
        field = abstraction.get("stable_capability_field") if isinstance(abstraction.get("stable_capability_field"), dict) else {}
        missing = required_field_gaps(field, field_required)
        if missing:
            rows.append(scf_gap("field_missing_required_fields", abstraction_id, missing, "complete the stable_capability_field record before this abstraction can govern replaceable implementations"))
            continue
        field_id = str(field.get("field_id") or "")
        if field_id != abstraction_id:
            rows.append(scf_gap("field_id_mismatch", abstraction_id, [f"field_id={field_id or 'missing'}"], "stable_capability_field.field_id must match the abstraction id"))
        for section, required in section_required.items():
            section_payload = field.get(section) if isinstance(field.get(section), dict) else {}
            section_missing = required_field_gaps(section_payload, required)
            if section_missing:
                rows.append(
                    scf_gap(
                        "field_section_missing_required_fields",
                        abstraction_id,
                        [f"{section}.{item}" for item in section_missing],
                        "fill the SCF section so callers can reason about substitution, authority, state, evidence, and recovery",
                    )
                )
        if not get_nested(field, ["resolution_policy", "decision_receipt_required"], False):
            rows.append(scf_gap("field_resolution_missing_decision_receipt", abstraction_id, ["resolution_policy.decision_receipt_required"], "SCF resolution must emit a replayable decision receipt"))
        if not get_nested(field, ["authority_policy", "effect_classes"], []):
            rows.append(scf_gap("field_authority_missing_effect_classes", abstraction_id, ["authority_policy.effect_classes"], "SCF authority policy must declare allowed effect classes"))
        if not get_nested(field, ["qualification_policy", "required_evaluators"], []):
            rows.append(scf_gap("field_qualification_missing_required_evaluators", abstraction_id, ["qualification_policy.required_evaluators"], "SCF qualification must cite evaluator/evidence obligations"))
        bool_missing = missing_true_gaps(
            field,
            [
                ["identity_policy", "exact_content_binding_required"],
                ["evidence_registry_policy", "source_events_append_only"],
                ["route_validation_policy", "validator_receipt_required"],
                ["route_validation_policy", "caller_binding_required"],
                ["adaptation_policy", "sealed_epoch_required"],
                ["adaptation_policy", "pinned_updater_required"],
                ["adaptation_policy", "approved_data_receipts_required"],
                ["composition_policy", "dependency_cycles_bundled"],
                ["governance_policy", "change_classification_required"],
            ],
        )
        if bool_missing:
            rows.append(
                scf_gap(
                    "field_public_release_boolean_clause_missing",
                    abstraction_id,
                    bool_missing,
                    "public-release SCF requires exact identity binding, append-only source events, validator receipts, sealed adaptation, bundled dependency cycles, and classified governance changes",
                )
            )

    for implementation in implementations:
        implementation_id = str(implementation.get("id") or "unknown")
        status = str(implementation.get("status") or "")
        if status not in {"live", "retained"}:
            continue
        abstraction_id = str(implementation.get("abstraction_id") or "")
        abstraction = abstraction_by_id.get(abstraction_id, {})
        binding = implementation.get("stable_capability_binding") if isinstance(implementation.get("stable_capability_binding"), dict) else {}
        missing = required_field_gaps(binding, binding_required)
        if missing:
            rows.append(scf_gap("implementation_binding_missing_required_fields", implementation_id, missing, "complete stable_capability_binding before routers can treat this implementation as field-qualified", implementation_id=implementation_id, abstraction_id=abstraction_id))
            continue
        binding_field = str(binding.get("field_id") or "")
        if binding_field != abstraction_id:
            rows.append(scf_gap("implementation_binding_field_mismatch", implementation_id, [f"field_id={binding_field or 'missing'}"], "implementation binding field_id must match abstraction_id", implementation_id=implementation_id, abstraction_id=abstraction_id))
        abstraction_contract = get_nested(abstraction, ["stable_capability_field", "contract_version"], "")
        binding_contract = str(binding.get("contract_version") or "")
        if abstraction_contract and binding_contract != abstraction_contract:
            rows.append(scf_gap("implementation_binding_contract_version_mismatch", implementation_id, [f"contract_version={binding_contract or 'missing'}"], "implementation binding must target the abstraction's SCF contract version", implementation_id=implementation_id, abstraction_id=abstraction_id))
        for section, required in binding_section_required.items():
            section_payload = binding.get(section) if isinstance(binding.get(section), dict) else {}
            section_missing = required_field_gaps(section_payload, required)
            if section_missing:
                rows.append(
                    scf_gap(
                        "implementation_binding_section_missing_required_fields",
                        implementation_id,
                        [f"{section}.{item}" for item in section_missing],
                        "fill the SCF implementation binding section so runtime resolution can filter by evidence, authority, state, and deployment profile",
                        implementation_id=implementation_id,
                        abstraction_id=abstraction_id,
                    )
                )
        if not get_nested(binding, ["authority_request", "effect_classes"], []):
            rows.append(scf_gap("implementation_binding_missing_effect_classes", implementation_id, ["authority_request.effect_classes"], "implementation authority_request must declare effect classes", implementation_id=implementation_id, abstraction_id=abstraction_id))
        if not get_nested(binding, ["observability_binding", "decision_receipt"], False):
            rows.append(scf_gap("implementation_binding_missing_decision_receipt", implementation_id, ["observability_binding.decision_receipt"], "routable implementations must emit or bind a decision receipt", implementation_id=implementation_id, abstraction_id=abstraction_id))
        bool_missing = missing_true_gaps(
            binding,
            [
                ["content_binding", "exact_hash_binding_required"],
                ["route_binding", "validator_receipt_required"],
                ["claim_binding", "evaluator_overlap_record_required"],
                ["lease_binding", "expiry_required"],
                ["lease_binding", "fail_closed_required"],
                ["migration_binding", "solvency_class_required"],
            ],
        )
        if bool_missing:
            rows.append(
                scf_gap(
                    "implementation_binding_public_release_boolean_clause_missing",
                    implementation_id,
                    bool_missing,
                    "public-release SCF bindings require exact hash identity, validator receipts, evaluator overlap records, expiring fail-closed leases, and migration solvency",
                    implementation_id=implementation_id,
                    abstraction_id=abstraction_id,
                )
            )
    return rows


def scf_gap(
    kind: str,
    scope: str,
    missing_fields: list[str],
    recommended_action: str,
    *,
    implementation_id: str = "",
    abstraction_id: str = "",
) -> dict[str, Any]:
    row = {
        "record_type": "project_registry_stable_capability_field_gap",
        "kind": kind,
        "scope": scope,
        "missing_fields": missing_fields,
        "recommended_action": recommended_action,
    }
    if abstraction_id or (scope and not implementation_id):
        row["abstraction_id"] = abstraction_id or scope
    if implementation_id:
        row["implementation_id"] = implementation_id
    return row


def stable_capability_field_health(
    policy: dict[str, Any],
    stable_field_gaps: list[dict[str, Any]],
    implementation_health: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    gaps_by_abstraction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    gaps_by_implementation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for gap in stable_field_gaps:
        abstraction_id = str(gap.get("abstraction_id") or "")
        implementation_id = str(gap.get("implementation_id") or "")
        if abstraction_id:
            gaps_by_abstraction[abstraction_id].append(gap)
        if implementation_id:
            gaps_by_implementation[implementation_id].append(gap)
    implementations_by_abstraction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in implementation_health:
        implementations_by_abstraction[str(row.get("abstraction_id") or "")].append(row)

    rows: list[dict[str, Any]] = []
    for abstraction in policy.get("abstractions", []) if isinstance(policy.get("abstractions"), list) else []:
        if not isinstance(abstraction, dict):
            continue
        abstraction_id = str(abstraction.get("id") or "")
        field = abstraction.get("stable_capability_field") if isinstance(abstraction.get("stable_capability_field"), dict) else {}
        implementation_rows = implementations_by_abstraction.get(abstraction_id, [])
        implementation_gaps = [
            gap
            for impl in implementation_rows
            for gap in gaps_by_implementation.get(str(impl.get("implementation_id") or ""), [])
        ]
        gaps = gaps_by_abstraction.get(abstraction_id, []) + implementation_gaps
        status = str(abstraction.get("status") or "")
        health_state = "GREEN"
        if gaps:
            health_state = "RED"
        elif status == "retained":
            health_state = "RETAINED"
        elif status not in {"live", "retained"}:
            health_state = "INACTIVE"
        rows.append(
            {
                "record_type": "project_registry_stable_capability_field_health",
                "abstraction_id": abstraction_id,
                "field_id": field.get("field_id", ""),
                "contract_version": field.get("contract_version", ""),
                "criticality": field.get("criticality", ""),
                "status": status,
                "health_state": health_state,
                "effect_classes": get_nested(field, ["authority_policy", "effect_classes"], []),
                "state_classes": get_nested(field, ["state_policy", "state_classes"], []),
                "deployment_profiles": get_nested(field, ["qualification_policy", "deployment_profiles"], []),
                "decision_receipt_required": bool(get_nested(field, ["resolution_policy", "decision_receipt_required"], False)),
                "implementation_binding_count": len(implementation_rows),
                "gaps": gaps,
                "recommended_action": "healthy" if health_state in {"GREEN", "RETAINED", "INACTIVE"} else "repair SCF field or implementation binding gaps before routing",
            }
        )
    return rows


def abstraction_registry_health(
    policy: dict[str, Any],
    implementation_health: list[dict[str, Any]],
    abstraction_gaps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    implementations_by_abstraction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in implementation_health:
        implementations_by_abstraction[str(row.get("abstraction_id") or "")].append(row)
    gaps_by_abstraction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for gap in abstraction_gaps:
        abstraction_id = str(gap.get("abstraction_id") or "")
        if abstraction_id:
            gaps_by_abstraction[abstraction_id].append(gap)
    rows = []
    for abstraction in policy.get("abstractions", []) if isinstance(policy.get("abstractions"), list) else []:
        if not isinstance(abstraction, dict):
            continue
        abstraction_id = str(abstraction.get("id") or "")
        implementations = implementations_by_abstraction.get(abstraction_id, [])
        live = [row for row in implementations if row.get("status") == "live"]
        eligible = [row for row in implementations if row.get("routing_eligible")]
        canonical_id = str(abstraction.get("canonical_implementation_id") or "")
        canonical_health = next((row for row in implementations if row.get("implementation_id") == canonical_id), {})
        gaps = gaps_by_abstraction.get(abstraction_id, [])
        abstraction_status = str(abstraction.get("status") or "")
        health_state = "GREEN"
        if gaps:
            health_state = "RED"
        elif abstraction_status == "live" and (not canonical_health or canonical_health.get("status") != "live"):
            health_state = "RED"
        elif abstraction_status == "retained" and canonical_health and canonical_health.get("status") in {"live", "retained"}:
            health_state = "RETAINED"
        elif not eligible and abstraction_status == "live":
            health_state = "YELLOW"
        rows.append(
            {
                "record_type": "project_registry_abstraction_health",
                "abstraction_id": abstraction_id,
                "title": abstraction.get("title"),
                "status": abstraction.get("status"),
                "health_state": health_state,
                "canonical_implementation_id": canonical_id,
                "canonical_implementation_live": bool(canonical_health and canonical_health.get("status") == "live"),
                "implementation_count": len(implementations),
                "live_implementation_count": len(live),
                "routing_eligible_implementation_count": len(eligible),
                "related_surfaces": abstraction.get("related_surfaces", []),
                "split_route_policy": abstraction.get("split_route_policy", ""),
                "gaps": gaps,
                "recommended_action": (
                    "healthy"
                    if health_state == "GREEN"
                    else "repair abstraction gaps or bind a live routing-eligible implementation"
                ),
            }
        )
    return rows


def implementation_registry_health(policy: dict[str, Any], report_status: list[dict[str, Any]]) -> list[dict[str, Any]]:
    abstractions = {
        str(row.get("id") or ""): row
        for row in policy.get("abstractions", [])
        if isinstance(row, dict) and row.get("id")
    }
    surfaces = {
        str(row.get("id") or ""): row
        for row in policy.get("surfaces", [])
        if isinstance(row, dict) and row.get("id")
    }
    report_by_path = {
        str(row.get("path") or ""): row
        for row in report_status
        if isinstance(row, dict) and row.get("path")
    }
    rows: list[dict[str, Any]] = []
    for implementation in policy.get("implementations", []) if isinstance(policy.get("implementations"), list) else []:
        if not isinstance(implementation, dict):
            continue
        implementation_id = str(implementation.get("id") or "")
        abstraction_id = str(implementation.get("abstraction_id") or "")
        surface_id = str(implementation.get("surface_id") or "")
        status = str(implementation.get("status") or "")
        abstraction = abstractions.get(abstraction_id, {})
        surface = surfaces.get(surface_id, {})
        eligibility = implementation.get("routing_eligibility") if isinstance(implementation.get("routing_eligibility"), dict) else {}
        evidence_outputs = [str(item) for item in implementation.get("evidence_outputs", []) if item]
        route_contract = evaluate_route_evidence_contract(policy, implementation)
        route_evidence_rows = route_contract["requirements"]
        route_evidence_paths = {str(row.get("path") or "") for row in route_evidence_rows if row.get("path")}
        evidence_rows = []
        for output in evidence_outputs:
            record = report_by_path.get(output)
            if record:
                evidence_rows.append(
                    {
                        "path": output,
                        "status": record.get("status"),
                        "evidence_class": "route_required" if output in route_evidence_paths else "supporting",
                        "route_required": output in route_evidence_paths,
                        "age_hours": record.get("age_hours"),
                        "max_age_hours": record.get("max_age_hours"),
                        "freshness_policy": record.get("freshness_policy"),
                        "not_applicable": bool(record.get("not_applicable")),
                    }
                )
            else:
                path = resolve(output)
                evidence_rows.append(
                    {
                        "path": output,
                        "status": "missing_from_surface_report_outputs" if not path.exists() else "untracked_by_surface",
                        "evidence_class": "route_required" if output in route_evidence_paths else "supporting",
                        "route_required": output in route_evidence_paths,
                        "age_hours": None,
                        "not_applicable": False,
                    }
                )
        evidence_blockers = list(route_contract.get("blockers", []))
        routing_required = bool(eligibility.get("eligible", status == "live"))
        routing_roles = [str(item) for item in eligibility.get("roles", []) if item] if isinstance(eligibility.get("roles"), list) else []
        routing_eligible = bool(
            status == "live"
            and abstraction
            and surface
            and str(abstraction.get("status") or "") in {"live", "retained"}
            and str(surface.get("status") or "") in {"live", "retained"}
            and routing_required
            and routing_roles
            and (not bool(eligibility.get("requires_fresh_evidence", True)) or not evidence_blockers)
        )
        if str(surface.get("status") or "") == "retained" and status == "live":
            routing_eligible = bool(routing_eligible and not bool(eligibility.get("runtime_serving_allowed")))
        blockers = []
        if status != "live":
            blockers.append(f"implementation_status={status or 'missing'}")
        if not abstraction:
            blockers.append("unknown_abstraction")
        elif str(abstraction.get("status") or "") not in {"live", "retained"}:
            blockers.append(f"abstraction_status={abstraction.get('status')}")
        if not surface:
            blockers.append("unknown_surface")
        elif str(surface.get("status") or "") not in {"live", "retained"}:
            blockers.append(f"surface_status={surface.get('status')}")
        if not routing_roles:
            blockers.append("routing_roles_missing")
        if evidence_blockers and bool(eligibility.get("requires_fresh_evidence", True)):
            blockers.append("route_evidence_blocked")
        if routing_required and route_contract.get("mode") == "not_route_required":
            blockers.append("route_evidence_contract_disables_required_route")
            routing_eligible = False
        rows.append(
            {
                "record_type": "project_registry_implementation_health",
                "implementation_id": implementation_id,
                "abstraction_id": abstraction_id,
                "surface_id": surface_id,
                "status": status,
                "role": str(implementation.get("role") or ""),
                "trust_tier": str(implementation.get("trust_tier") or ""),
                "implementation_type": str(implementation.get("implementation_type") or ""),
                "backend": str(implementation.get("backend") or ""),
                "canonical_entrypoint": str(implementation.get("canonical_entrypoint") or ""),
                "routing_required": routing_required,
                "routing_eligible": routing_eligible,
                "routing_roles": routing_roles,
                "evidence_outputs": evidence_rows,
                "route_evidence_contract_id": route_contract.get("contract_id"),
                "route_evidence_mode": route_contract.get("mode"),
                "route_evidence": route_evidence_rows,
                "evidence_blocker_count": len(evidence_blockers),
                "evidence_blockers": evidence_blockers,
                "supporting_evidence_output_count": sum(1 for row in evidence_rows if not row.get("route_required")),
                "blockers": blockers,
                "attd_hooks": implementation.get("attd_hooks") if isinstance(implementation.get("attd_hooks"), dict) else {},
                "routing_eligibility_contract": eligibility,
                "recommended_action": implementation_health_action(blockers),
            }
        )
    return rows


def implementation_health_action(blockers: list[str]) -> str:
    if not blockers:
        return "eligible under the registry contract"
    if "route_evidence_blocked" in blockers:
        return "refresh the minimal route receipt, repair its source/acceptance contract, or disable the route explicitly"
    if "unknown_abstraction" in blockers or "unknown_surface" in blockers:
        return "repair the implementation binding to a declared abstraction and surface"
    if "routing_roles_missing" in blockers:
        return "declare routing_eligibility.roles before any router can select this implementation"
    return "resolve the listed registry blockers before routing or promotion"


def registry_routing_eligibility(policy: dict[str, Any], implementation_health: list[dict[str, Any]]) -> list[dict[str, Any]]:
    abstractions = {
        str(row.get("id") or ""): row
        for row in policy.get("abstractions", [])
        if isinstance(row, dict) and row.get("id")
    }
    rows: list[dict[str, Any]] = []
    for health in implementation_health:
        roles = [str(item) for item in health.get("routing_roles", []) if item]
        abstraction = abstractions.get(str(health.get("abstraction_id") or ""), {})
        for role in roles or ["unroutable"]:
            rows.append(
                {
                    "record_type": "project_registry_routing_eligibility",
                    "role": role,
                    "implementation_id": health.get("implementation_id"),
                    "abstraction_id": health.get("abstraction_id"),
                    "surface_id": health.get("surface_id"),
                    "routing_eligible": bool(health.get("routing_eligible")),
                    "trust_tier": health.get("trust_tier"),
                    "backend": health.get("backend"),
                    "canonical": str(abstraction.get("canonical_implementation_id") or "") == str(health.get("implementation_id") or ""),
                    "learned_generation_claim_allowed": bool(
                        get_nested(health, ["routing_eligibility_contract", "learned_generation_claim_allowed"])
                    ),
                    "runtime_serving_allowed": bool(
                        get_nested(health, ["routing_eligibility_contract", "runtime_serving_allowed"])
                    ),
                    "teacher_allowed": bool(get_nested(health, ["routing_eligibility_contract", "teacher_allowed"])),
                    "blockers": health.get("blockers", []),
                }
            )
    return sorted(rows, key=lambda row: (str(row.get("role")), str(row.get("implementation_id"))))


def registry_cleanup_queue(
    abstraction_gaps: list[dict[str, Any]],
    implementation_health: list[dict[str, Any]],
    duplicate_families: list[dict[str, Any]],
    report_status: list[dict[str, Any]],
    generated_source: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gap in abstraction_gaps[:16]:
        rows.append(
            cleanup_item(
                "registry_contract_gap",
                "critical",
                [str(gap.get("scope") or gap.get("abstraction_id") or gap.get("implementation_id") or gap.get("surface_id") or "")],
                str(gap.get("recommended_action") or "repair the abstraction/implementation contract gap"),
                gap,
            )
        )
    for health in implementation_health:
        if health.get("routing_required") and not health.get("routing_eligible"):
            rows.append(
                cleanup_item(
                    "implementation_not_routing_eligible",
                    "high",
                    [str(health.get("implementation_id") or "")],
                    str(health.get("recommended_action") or "repair implementation health before routing"),
                    health,
                )
            )
    for row in duplicate_families[:16]:
        priority = "medium" if row.get("classified") else "high"
        action = "improve the canonical registered surface before adding another version"
        if row.get("classified"):
            classification = row.get("classification") if isinstance(row.get("classification"), dict) else {}
            action = str(classification.get("successor_policy") or row.get("recommended_action") or action)
        rows.append(
            cleanup_item(
                "duplicate_family_pressure",
                priority,
                [f"{row.get('root')}/{row.get('family')}"],
                action,
                row,
            )
        )
    stale_or_missing = [
        row
        for row in report_status
        if isinstance(row, dict)
        and row.get("route_required")
        and row.get("status") in {"stale", "missing", "invalid", "rejected", "blocked"}
        and not row.get("not_applicable")
    ]
    for row in stale_or_missing[:16]:
        rows.append(
            cleanup_item(
                "blocked_route_evidence",
                "medium",
                [str(row.get("path") or "")],
                "refresh or repair the minimal route receipt; do not regenerate unrelated supporting evidence",
                row,
            )
        )
    for row in generated_source[:16]:
        rows.append(
            cleanup_item(
                "generated_source_artifact",
                "high",
                [str(row.get("path") or "")],
                "move generated/cache/build state out of active source paths through retention or ignore coverage",
                row,
            )
        )
    for index, row in enumerate(rows, start=1):
        row["queue_id"] = f"registry_cleanup_{index:03d}_{row['kind']}"
    return rows[:80]


def registry_decision_report(
    policy: dict[str, Any],
    cleanup_queue: list[dict[str, Any]],
    routing_eligibility: list[dict[str, Any]],
) -> dict[str, Any]:
    decisions = []
    for item in cleanup_queue[:24]:
        kind = str(item.get("kind") or "")
        decision = "improve_existing"
        if kind == "registry_contract_gap":
            decision = "repair_contract"
        elif kind == "implementation_not_routing_eligible":
            decision = "repair_or_replace_implementation"
        elif kind == "duplicate_family_pressure":
            decision = "consolidate_or_archive"
        elif kind == "blocked_route_evidence":
            decision = "refresh_or_repair_route_evidence"
        elif kind == "generated_source_artifact":
            decision = "archive_generated_state"
        decisions.append(
            {
                "record_type": "project_registry_decision",
                "decision": decision,
                "source_queue_id": item.get("queue_id"),
                "priority": item.get("priority"),
                "scope": item.get("scope", []),
                "bounded_action": item.get("bounded_action", ""),
                "new_abstraction_allowed": False,
                "reason": "Registry pressure should be resolved by improving, replacing, retiring, or archiving an existing registered surface before adding a new lane.",
            }
        )
    eligible_by_role: dict[str, list[str]] = defaultdict(list)
    for row in routing_eligibility:
        if row.get("routing_eligible"):
            eligible_by_role[str(row.get("role") or "unknown")].append(str(row.get("implementation_id") or ""))
    return {
        "policy": "project_theseus_registry_decision_report_v1",
        "decision_count": len(decisions),
        "top_decision": decisions[0] if decisions else {
            "decision": "improve_existing_registered_implementations",
            "reason": "No registry blockers are present; future work should improve or compare existing implementations before proposing new abstractions.",
            "new_abstraction_allowed": False,
        },
        "decisions": decisions,
        "eligible_implementations_by_role": {key: sorted(set(value)) for key, value in sorted(eligible_by_role.items())},
        "new_abstraction_policy": get_nested(
            policy,
            ["registry_evolution_contract", "decision_order"],
            [],
        ),
        "external_inference_calls": 0,
    }


def cleanup_item(kind: str, priority: str, scope: list[str], action: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_type": "project_registry_cleanup_queue_item",
        "kind": kind,
        "priority": priority,
        "scope": [item for item in scope if item][:12],
        "bounded_action": action,
        "evidence": evidence,
        "verification": [
            "python3 scripts/theseus_project_registry.py --gate",
            "python3 scripts/attd_analyzer.py",
            "python3 scripts/theseus_control_plane.py --no-ingest",
        ],
    }


def surface_schema_gaps(policy: dict[str, Any]) -> list[dict[str, Any]]:
    contract = policy.get("registry_evolution_contract") if isinstance(policy.get("registry_evolution_contract"), dict) else {}
    required = [str(item) for item in contract.get("new_surface_required_fields", []) if item]
    rows: list[dict[str, Any]] = []
    for surface in policy.get("surfaces", []) if isinstance(policy.get("surfaces"), list) else []:
        if not isinstance(surface, dict):
            continue
        missing = []
        for field in required:
            value = surface.get(field)
            if value in (None, "", [], {}):
                missing.append(field)
        if missing:
            rows.append(
                {
                    "record_type": "project_registry_surface_schema_gap",
                    "surface_id": str(surface.get("id") or "unknown"),
                    "missing_fields": missing,
                }
            )
    return rows


def retained_surface_gaps(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in entries:
        if row.get("status") != "retained":
            continue
        cleanup = str(row.get("cleanup_policy") or "")
        canonical = str(row.get("canonical") or "")
        if cleanup and canonical:
            continue
        rows.append(
            {
                "record_type": "project_registry_retained_surface_gap",
                "path": row.get("path"),
                "surface_id": row.get("surface_id"),
                "canonical": canonical,
                "cleanup_policy": cleanup,
            }
        )
    return rows


def root_size_summaries(policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    roots = []
    for item in policy.get("inventory_roots", []) if isinstance(policy.get("inventory_roots"), list) else []:
        if isinstance(item, dict) and item.get("path"):
            roots.append(str(item["path"]))
    for raw in sorted(set(roots)):
        path = resolve(raw)
        if not path.exists():
            continue
        size_kib = du_kib(path)
        rows.append(
            {
                "record_type": "project_registry_root_summary",
                "path": normalize_path(raw),
                "exists": True,
                "kind": "directory" if path.is_dir() else "file",
                "mib": round(size_kib / 1024.0, 3) if size_kib is not None else None,
                "gib": round(size_kib / (1024.0 * 1024.0), 3) if size_kib is not None else None,
                "cleanup_class": cleanup_class_for_root(normalize_path(raw)),
            }
        )
    return rows


def cleanup_class_for_root(root: str) -> str:
    if root in {"reports", "runtime", "archive", "checkpoints", "target", "tmp", "logs", "dist"}:
        return "generated_or_build_state"
    if root in {"data"}:
        return "mixed_training_and_generated_state"
    if root in {"scripts", "configs", "docs", "crates", "dashboard", "tests", "benchmarks/cards"}:
        return "active_source_surface"
    if root in {"D:", "deprecated/windows-drive-mirror"}:
        return "deprecated_windows_path_mirror"
    return "supporting_surface"


def du_kib(path: Path) -> int | None:
    try:
        result = subprocess.run(["du", "-sk", str(path)], cwd=ROOT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    first = result.stdout.splitlines()[0].split()[0] if result.stdout.splitlines() else ""
    return int(first) if first.isdigit() else None


def build_summary(
    entries: list[dict[str, Any]],
    unregistered: list[dict[str, Any]],
    abstraction_gaps: list[dict[str, Any]],
    duplicate_families: list[dict[str, Any]],
    report_status: list[dict[str, Any]],
    generated_source: list[dict[str, Any]],
    abstraction_health: list[dict[str, Any]],
    implementation_health: list[dict[str, Any]],
    routing_eligibility: list[dict[str, Any]],
    cleanup_queue: list[dict[str, Any]],
    root_summaries: list[dict[str, Any]],
    governance_violations: list[dict[str, Any]],
    stable_field_gaps: list[dict[str, Any]],
    stable_field_health: list[dict[str, Any]],
    steward_coverage: dict[str, Any],
    route_validator_spine_receipt: dict[str, Any],
    policy: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    status_counts = Counter(str(row.get("status") or "unknown") for row in entries)
    type_counts = Counter(str(row.get("artifact_type") or "unknown") for row in entries)
    registered_paths = {str(row.get("path")) for row in entries}
    unregistered_active = len(unregistered)
    route_reports = [row for row in report_status if row.get("route_required")]
    supporting_reports = [row for row in report_status if not row.get("route_required")]
    stale_reports = [row for row in route_reports if row.get("stale")]
    missing_reports = [row for row in route_reports if not row.get("exists") and not row.get("not_applicable")]
    blocked_route_reports = [
        row
        for row in route_reports
        if row.get("status") in {"stale", "missing", "invalid", "rejected", "blocked"}
        and not row.get("not_applicable")
    ]
    missing_supporting_reports = [
        row for row in supporting_reports if not row.get("exists") and not row.get("not_applicable")
    ]
    generated_mib = sum(float(row.get("mib") or 0.0) for row in root_summaries if row.get("cleanup_class") == "generated_or_build_state")
    source_mib = sum(float(row.get("mib") or 0.0) for row in root_summaries if row.get("cleanup_class") == "active_source_surface")
    source_duplicates = [row for row in duplicate_families if str(row.get("root") or "") in {"scripts", "configs", "docs"}]
    report_duplicates = [row for row in duplicate_families if str(row.get("root") or "") == "reports"]
    unclassified_source_duplicates = [row for row in source_duplicates if not row.get("classified")]
    classified_source_duplicates = [row for row in source_duplicates if row.get("classified")]
    unclassified_report_duplicates = [row for row in report_duplicates if not row.get("classified")]
    classified_report_duplicates = [row for row in report_duplicates if row.get("classified")]
    unclassified_duplicates = [row for row in duplicate_families if not row.get("classified")]
    hard_governance = [row for row in governance_violations if str(row.get("severity") or "") == "hard"]
    routing_blocked = [row for row in implementation_health if row.get("routing_required") and not row.get("routing_eligible")]
    abstraction_health_red = [row for row in abstraction_health if row.get("health_state") == "RED"]
    abstraction_health_yellow = [row for row in abstraction_health if row.get("health_state") == "YELLOW"]
    routing_eligible = [row for row in routing_eligibility if row.get("routing_eligible")]
    learned_claim_allowed = [row for row in routing_eligibility if row.get("learned_generation_claim_allowed")]
    runtime_serving_allowed = [row for row in routing_eligibility if row.get("runtime_serving_allowed")]
    stable_field_health_red = [row for row in stable_field_health if row.get("health_state") == "RED"]
    stable_field_health_yellow = [row for row in stable_field_health if row.get("health_state") == "YELLOW"]
    stable_field_ready = [row for row in stable_field_health if row.get("health_state") in {"GREEN", "RETAINED"}]
    cleanup_steward_coverage = cleanup_queue_steward_coverage(
        cleanup_queue,
        list_dicts(steward_coverage.get("steward_decisions")),
    )
    covered_cleanup_queue_ids = list_values(cleanup_steward_coverage.get("covered_ids"))
    uncovered_cleanup_queue_ids = list_values(cleanup_steward_coverage.get("uncovered_ids"))
    return {
        "entry_count": len(entries),
        "registered_path_count": len(registered_paths),
        "abstraction_count": len(policy.get("abstractions", [])) if isinstance(policy.get("abstractions"), list) else 0,
        "implementation_count": len(policy.get("implementations", [])) if isinstance(policy.get("implementations"), list) else 0,
        "abstraction_registry_gap_count": len(abstraction_gaps),
        "stable_capability_field_gap_count": len(stable_field_gaps),
        "stable_capability_field_health_red_count": len(stable_field_health_red),
        "stable_capability_field_health_yellow_count": len(stable_field_health_yellow),
        "stable_capability_field_ready_count": len(stable_field_ready),
        "route_validator_viea_spine_view_ready": bool(route_validator_spine_receipt.get("ready")),
        "route_validator_viea_spine_record_count": int(route_validator_spine_receipt.get("record_count") or 0),
        "route_validator_viea_spine_governance_record_count": int(route_validator_spine_receipt.get("governance_record_count") or 0),
        "route_validator_viea_spine_failure_boundary_count": int(route_validator_spine_receipt.get("failure_boundary_count") or 0),
        "route_validator_viea_spine_authority_record_count": int(route_validator_spine_receipt.get("authority_record_count") or 0),
        "route_validator_viea_spine_resource_route_record_count": int(route_validator_spine_receipt.get("resource_route_record_count") or 0),
        "route_validator_viea_spine_missing_group_count": len(route_validator_spine_receipt.get("missing_required_groups") if isinstance(route_validator_spine_receipt.get("missing_required_groups"), list) else []),
        "project_steward_status": steward_coverage.get("status", "MISSING"),
        "project_steward_hard_gap_count": steward_coverage.get("hard_gap_count", 0),
        "project_steward_warning_count": steward_coverage.get("warning_count", 0),
        "project_work_contract_count": steward_coverage.get("work_contract_count", 0),
        "active_project_work_contract_count": steward_coverage.get("active_work_contract_count", 0),
        "event_taint_record_count": steward_coverage.get("event_taint_record_count", 0),
        "module_card_count": steward_coverage.get("module_card_count", 0),
        "major_surface_module_card_coverage_ratio": steward_coverage.get("major_surface_module_card_coverage_ratio", 0.0),
        "steward_decision_count": steward_coverage.get("steward_decision_count", 0),
        "cleanup_queue_steward_decision_count": len(covered_cleanup_queue_ids),
        "cleanup_queue_steward_uncovered_count": len(uncovered_cleanup_queue_ids),
        "cleanup_queue_steward_coverage_ratio": cleanup_steward_coverage.get("coverage_ratio"),
        "cleanup_queue_steward_uncovered_ids": uncovered_cleanup_queue_ids[:24],
        "abstraction_health_red_count": len(abstraction_health_red),
        "abstraction_health_yellow_count": len(abstraction_health_yellow),
        "implementation_routing_blocker_count": len(routing_blocked),
        "routing_eligible_implementation_count": len({str(row.get("implementation_id") or "") for row in routing_eligible}),
        "routing_role_count": len({str(row.get("role") or "") for row in routing_eligibility if row.get("role")}),
        "learned_generation_claim_allowed_count": len(learned_claim_allowed),
        "runtime_serving_allowed_count": len(runtime_serving_allowed),
        "registry_cleanup_queue_count": len(cleanup_queue),
        "registry_decision_count": len(cleanup_queue),
        "unregistered_active_source_count": unregistered_active,
        "coverage_ratio": round(len(registered_paths) / max(1, len(registered_paths) + unregistered_active), 6),
        "status_counts": dict(sorted(status_counts.items())),
        "artifact_type_counts": dict(sorted(type_counts.items())),
        "duplicate_family_count": len(duplicate_families),
        "source_duplicate_family_count": len(source_duplicates),
        "classified_source_duplicate_family_count": len(classified_source_duplicates),
        "unclassified_source_duplicate_family_count": len(unclassified_source_duplicates),
        "report_duplicate_family_count": len(report_duplicates),
        "classified_report_duplicate_family_count": len(classified_report_duplicates),
        "unclassified_report_duplicate_family_count": len(unclassified_report_duplicates),
        "unclassified_duplicate_family_count": len(unclassified_duplicates),
        "stale_report_output_count": len(stale_reports),
        "missing_report_output_count": len(missing_reports),
        "route_evidence_output_count": len(route_reports),
        "blocked_route_evidence_output_count": len(blocked_route_reports),
        "supporting_evidence_output_count": len(supporting_reports),
        "missing_supporting_evidence_output_count": len(missing_supporting_reports),
        "generated_source_artifact_count": len(generated_source),
        "registry_governance_violation_count": len(governance_violations),
        "registry_hard_governance_violation_count": len(hard_governance),
        "generated_or_build_state_mib": round(generated_mib, 3),
        "active_source_surface_mib": round(source_mib, 3),
        "thresholds": policy.get("coverage_thresholds", {}),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }


def trigger_state_for(summary: dict[str, Any], policy: dict[str, Any]) -> str:
    thresholds = policy.get("coverage_thresholds") if isinstance(policy.get("coverage_thresholds"), dict) else {}
    if int(summary.get("registry_hard_governance_violation_count") or 0) > 0:
        return "RED"
    if str(summary.get("project_steward_status") or "") == "RED":
        return "RED"
    if summary.get("route_validator_viea_spine_view_ready") is False:
        return "RED"
    unregistered = int(summary.get("unregistered_active_source_count") or 0)
    if unregistered >= int(thresholds.get("red_unregistered_active_sources", 80)):
        return "RED"
    if unregistered >= int(thresholds.get("yellow_unregistered_active_sources", 20)):
        return "YELLOW"
    if int(summary.get("unclassified_duplicate_family_count") or 0) >= int(thresholds.get("yellow_duplicate_families", 8)):
        return "YELLOW"
    if int(summary.get("stale_report_output_count") or 0) >= int(thresholds.get("yellow_stale_report_outputs", 8)):
        return "YELLOW"
    if int(summary.get("generated_source_artifact_count") or 0) >= int(thresholds.get("yellow_generated_source_artifacts", 1)):
        return "YELLOW"
    if str(summary.get("project_steward_status") or "") == "YELLOW":
        return "YELLOW"
    return "GREEN"


def compact_surfaces(policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for surface in policy.get("surfaces", []) if isinstance(policy.get("surfaces"), list) else []:
        if not isinstance(surface, dict):
            continue
        rows.append(
            {
                "id": surface.get("id"),
                "artifact_type": surface.get("artifact_type"),
                "role": surface.get("role"),
                "owner": surface.get("owner"),
                "status": surface.get("status"),
                "abstraction_id": surface.get("abstraction_id", ""),
                "major_surface": surface.get("major_surface", False),
                "routing_roles": surface.get("routing_roles", []),
                "canonical": surface.get("canonical"),
                "report_outputs": surface.get("report_outputs", []),
                "verification_command": surface.get("verification_command", ""),
            }
        )
    return rows


def compact_abstractions(policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for abstraction in policy.get("abstractions", []) if isinstance(policy.get("abstractions"), list) else []:
        if not isinstance(abstraction, dict):
            continue
        rows.append(
            {
                "id": abstraction.get("id"),
                "title": abstraction.get("title"),
                "status": abstraction.get("status"),
                "owner": abstraction.get("owner"),
                "stable_contract": abstraction.get("stable_contract"),
                "io_contract": abstraction.get("io_contract", {}),
                "stable_capability_field": abstraction.get("stable_capability_field", {}),
                "forbidden_shortcuts": abstraction.get("forbidden_shortcuts", []),
                "canonical_implementation_id": abstraction.get("canonical_implementation_id"),
                "required_evidence": abstraction.get("required_evidence", []),
                "related_surfaces": abstraction.get("related_surfaces", []),
                "split_route_policy": abstraction.get("split_route_policy", ""),
            }
        )
    return rows


def compact_implementations(policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for implementation in policy.get("implementations", []) if isinstance(policy.get("implementations"), list) else []:
        if not isinstance(implementation, dict):
            continue
        rows.append(
            {
                "id": implementation.get("id"),
                "abstraction_id": implementation.get("abstraction_id"),
                "surface_id": implementation.get("surface_id"),
                "status": implementation.get("status"),
                "role": implementation.get("role"),
                "trust_tier": implementation.get("trust_tier"),
                "implementation_type": implementation.get("implementation_type"),
                "backend": implementation.get("backend"),
                "canonical_entrypoint": implementation.get("canonical_entrypoint"),
                "evidence_outputs": implementation.get("evidence_outputs", []),
                "route_evidence_contract_id": implementation.get("route_evidence_contract_id", ""),
                "routing_eligibility": implementation.get("routing_eligibility", {}),
                "stable_capability_binding": implementation.get("stable_capability_binding", {}),
                "verification_command": implementation.get("verification_command", ""),
            }
        )
    return rows


def write_registry_table(db_path: Path, payload: dict[str, Any]) -> None:
    conn = report_evidence_store.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS project_manifest_registry (
                path TEXT PRIMARY KEY,
                surface_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                role TEXT NOT NULL,
                owner TEXT NOT NULL,
                status TEXT NOT NULL,
                canonical TEXT NOT NULL,
                verification_command TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                updated_utc TEXT NOT NULL
            )
            """
        )
        for row in payload.get("entries", []):
            conn.execute(
                """
                INSERT INTO project_manifest_registry (
                    path, surface_id, artifact_type, role, owner, status,
                    canonical, verification_command, payload_json, updated_utc
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    surface_id=excluded.surface_id,
                    artifact_type=excluded.artifact_type,
                    role=excluded.role,
                    owner=excluded.owner,
                    status=excluded.status,
                    canonical=excluded.canonical,
                    verification_command=excluded.verification_command,
                    payload_json=excluded.payload_json,
                    updated_utc=excluded.updated_utc
                """,
                (
                    row["path"],
                    row["surface_id"],
                    row["artifact_type"],
                    row["role"],
                    row["owner"],
                    row["status"],
                    row.get("canonical") or "",
                    row.get("verification_command") or "",
                    json.dumps(row, sort_keys=True),
                    payload["created_utc"],
                ),
            )
        conn.commit()
    finally:
        conn.close()


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Theseus Project Registry",
        "",
        f"- trigger_state: `{payload.get('trigger_state')}`",
        f"- surfaces: `{payload.get('surface_count')}`",
        f"- abstractions: `{summary.get('abstraction_count')}`",
        f"- implementations: `{summary.get('implementation_count')}`",
        f"- abstraction registry gaps: `{summary.get('abstraction_registry_gap_count')}`",
        f"- stable capability field gaps: `{summary.get('stable_capability_field_gap_count')}`",
        f"- stable capability field health red/yellow: `{summary.get('stable_capability_field_health_red_count')}` / `{summary.get('stable_capability_field_health_yellow_count')}`",
        f"- route validator VIEA view ready: `{summary.get('route_validator_viea_spine_view_ready')}` records `{summary.get('route_validator_viea_spine_record_count')}` governance `{summary.get('route_validator_viea_spine_governance_record_count')}` failures `{summary.get('route_validator_viea_spine_failure_boundary_count')}`",
        f"- project steward status: `{summary.get('project_steward_status')}` hard `{summary.get('project_steward_hard_gap_count')}` warnings `{summary.get('project_steward_warning_count')}`",
        f"- project work contracts: `{summary.get('project_work_contract_count')}` active `{summary.get('active_project_work_contract_count')}`",
        f"- event-taint records: `{summary.get('event_taint_record_count')}`",
        f"- module cards: `{summary.get('module_card_count')}` coverage `{summary.get('major_surface_module_card_coverage_ratio')}`",
        f"- steward decisions: `{summary.get('steward_decision_count')}`",
        f"- abstraction health red/yellow: `{summary.get('abstraction_health_red_count')}` / `{summary.get('abstraction_health_yellow_count')}`",
        f"- implementation routing blockers: `{summary.get('implementation_routing_blocker_count')}`",
        f"- routing-eligible implementations: `{summary.get('routing_eligible_implementation_count')}`",
        f"- routing roles: `{summary.get('routing_role_count')}`",
        f"- learned-generation claim routes: `{summary.get('learned_generation_claim_allowed_count')}`",
        f"- runtime-serving routes: `{summary.get('runtime_serving_allowed_count')}`",
        f"- registry cleanup queue: `{summary.get('registry_cleanup_queue_count')}`",
        f"- registry decisions: `{summary.get('registry_decision_count')}`",
        f"- entries: `{summary.get('entry_count')}`",
        f"- coverage_ratio: `{summary.get('coverage_ratio')}`",
        f"- unregistered active sources: `{summary.get('unregistered_active_source_count')}`",
        f"- duplicate families: `{summary.get('duplicate_family_count')}`",
        f"- source duplicate families: `{summary.get('source_duplicate_family_count')}`",
        f"- classified source duplicate families: `{summary.get('classified_source_duplicate_family_count')}`",
        f"- unclassified source duplicate families: `{summary.get('unclassified_source_duplicate_family_count')}`",
        f"- report duplicate families: `{summary.get('report_duplicate_family_count')}`",
        f"- classified report duplicate families: `{summary.get('classified_report_duplicate_family_count')}`",
        f"- unclassified report duplicate families: `{summary.get('unclassified_report_duplicate_family_count')}`",
        f"- unclassified duplicate families: `{summary.get('unclassified_duplicate_family_count')}`",
        f"- route evidence: `{summary.get('route_evidence_output_count')}` blocked `{summary.get('blocked_route_evidence_output_count')}` stale `{summary.get('stale_report_output_count')}` missing `{summary.get('missing_report_output_count')}`",
        f"- supporting evidence outputs: `{summary.get('supporting_evidence_output_count')}` missing `{summary.get('missing_supporting_evidence_output_count')}`",
        f"- generated source artifacts: `{summary.get('generated_source_artifact_count')}`",
        f"- registry governance violations: `{summary.get('registry_governance_violation_count')}` hard `{summary.get('registry_hard_governance_violation_count')}`",
        f"- generated/build state MiB: `{summary.get('generated_or_build_state_mib')}`",
        "",
        "## Registry Evolution Contract",
        "",
    ]
    contract = payload.get("registry_evolution_contract") if isinstance(payload.get("registry_evolution_contract"), dict) else {}
    for index, item in enumerate(contract.get("decision_order", []) if isinstance(contract.get("decision_order"), list) else [], start=1):
        lines.append(f"{index}. `{item}`")
    if contract.get("purpose"):
        lines.extend(["", str(contract.get("purpose")), ""])
    abstraction_contract = payload.get("abstraction_registry_contract") if isinstance(payload.get("abstraction_registry_contract"), dict) else {}
    lines.extend(["## Abstraction Registry", ""])
    if abstraction_contract.get("purpose"):
        lines.extend([str(abstraction_contract.get("purpose")), ""])
    steward = payload.get("project_steward_coverage") if isinstance(payload.get("project_steward_coverage"), dict) else {}
    lines.extend(["## Project Steward Coverage", ""])
    lines.append(
        f"- status: `{steward.get('status', 'MISSING')}` config=`{steward.get('config_path', '')}`"
    )
    lines.append(
        f"- work contracts: `{steward.get('work_contract_count', 0)}` active `{steward.get('active_work_contract_count', 0)}`"
    )
    lines.append(
        f"- event-taint records: `{steward.get('event_taint_record_count', 0)}` missing mandatory `{len(steward.get('missing_mandatory_taint_classes') or [])}`"
    )
    lines.append(
        f"- module cards: `{steward.get('module_card_count', 0)}` major-surface coverage `{steward.get('major_surface_module_card_coverage_ratio', 0.0)}`"
    )
    lines.append(f"- steward decisions: `{steward.get('steward_decision_count', 0)}`")
    for row in steward.get("hard_gaps", [])[:12] if isinstance(steward.get("hard_gaps"), list) else []:
        if isinstance(row, dict):
            lines.append(f"- hard gap `{row.get('kind')}`: {row.get('message')}")
    for row in steward.get("warnings", [])[:12] if isinstance(steward.get("warnings"), list) else []:
        if isinstance(row, dict):
            lines.append(f"- warning `{row.get('kind')}`: {row.get('message')}")
    lines.append("")
    lines.append("### Abstractions")
    lines.append("")
    for row in payload.get("abstractions", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('id')}` status=`{row.get('status')}` canonical_impl=`{row.get('canonical_implementation_id')}`"
        )
    lines.extend(["", "### Implementations", ""])
    for row in payload.get("implementations", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('id')}` abstraction=`{row.get('abstraction_id')}` surface=`{row.get('surface_id')}` status=`{row.get('status')}` role=`{row.get('role')}` backend=`{row.get('backend')}`"
        )
    lines.extend(["", "### Abstraction Registry Gaps", ""])
    for row in payload.get("abstraction_registry_gaps", []):
        if not isinstance(row, dict):
            continue
        lines.append(f"- `{row.get('kind')}` scope=`{row.get('scope')}`")
    if not payload.get("abstraction_registry_gaps"):
        lines.append("- None.")
    lines.extend(["", "### Stable Capability Field Health", ""])
    for row in payload.get("stable_capability_field_health", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('abstraction_id')}` health=`{row.get('health_state')}` contract=`{row.get('contract_version')}` criticality=`{row.get('criticality')}` decision_receipt=`{row.get('decision_receipt_required')}`"
        )
        if row.get("gaps"):
            lines.append(f"  - gaps: `{len(row.get('gaps') or [])}`")
    lines.extend(["", "### Abstraction Health", ""])
    for row in payload.get("abstraction_health", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('abstraction_id')}` health=`{row.get('health_state')}` canonical=`{row.get('canonical_implementation_id')}` live_impls=`{row.get('live_implementation_count')}` eligible_impls=`{row.get('routing_eligible_implementation_count')}` split=`{row.get('split_route_policy')}`"
        )
    lines.extend(["", "### Routing Eligibility", ""])
    for row in payload.get("routing_eligibility", [])[:40]:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- role=`{row.get('role')}` impl=`{row.get('implementation_id')}` eligible=`{row.get('routing_eligible')}` learned_claim=`{row.get('learned_generation_claim_allowed')}` runtime=`{row.get('runtime_serving_allowed')}`"
        )
        if row.get("blockers"):
            lines.append(f"  - blockers: `{', '.join(str(item) for item in row.get('blockers', []))}`")
    lines.extend(["", "### Cleanup Queue", ""])
    for row in payload.get("cleanup_queue", [])[:30]:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('queue_id')}` priority=`{row.get('priority')}` kind=`{row.get('kind')}` scope=`{', '.join(str(item) for item in row.get('scope', []))}`"
        )
        if row.get("bounded_action"):
            lines.append(f"  - {row.get('bounded_action')}")
    if not payload.get("cleanup_queue"):
        lines.append("- None.")
    decisions = payload.get("registry_decisions") if isinstance(payload.get("registry_decisions"), dict) else {}
    lines.extend(["", "### Registry Decisions", ""])
    top_decision = decisions.get("top_decision") if isinstance(decisions.get("top_decision"), dict) else {}
    if top_decision:
        lines.append(f"- top: `{top_decision.get('decision')}` {top_decision.get('reason', '')}")
    for row in decisions.get("decisions", [])[:20] if isinstance(decisions.get("decisions"), list) else []:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('decision')}` priority=`{row.get('priority')}` scope=`{', '.join(str(item) for item in row.get('scope', []))}`"
        )
    lines.extend([""])
    lines.extend(
        [
            "## Governance Violations",
            "",
        ]
    )
    for row in payload.get("governance_violations", [])[:20]:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('rule_id')}` `{row.get('severity')}` `{row.get('kind')}` scope `{len(row.get('scope') or [])}`"
        )
        if row.get("recommended_action"):
            lines.append(f"  - {row.get('recommended_action')}")
    if not payload.get("governance_violations"):
        lines.append("- None.")
    lines.extend(
        [
            "",
        "## Top Unregistered Active Sources",
        "",
        ]
    )
    for row in payload.get("unregistered", [])[:40]:
        lines.append(f"- `{row.get('path')}` tracked=`{row.get('tracked_state')}`")
    lines.extend(["", "## Duplicate Families", ""])
    for row in payload.get("duplicate_families", [])[:24]:
        classification = row.get("classification") if isinstance(row.get("classification"), dict) else {}
        classified = "classified" if row.get("classified") else "unclassified"
        lines.append(
            f"- `{row.get('root')}/{row.get('family')}` count=`{row.get('count')}` `{classified}` canonical=`{classification.get('canonical_path', '')}`"
        )
    lines.extend(["", "## Blocked Route Evidence And Missing Supporting Outputs", ""])
    for row in payload.get("report_outputs", []):
        if row.get("status") not in {"fresh", "available", "not_applicable"}:
            lines.append(
                f"- `{row.get('status')}` `{row.get('path')}` surface=`{row.get('surface_id')}` "
                f"class=`{row.get('evidence_class')}` age=`{row.get('age_hours')}`"
            )
    return "\n".join(lines) + "\n"


def report_created_utc(path: Path) -> datetime | None:
    payload = read_json(path, {})
    if isinstance(payload, dict):
        for key in ("created_utc", "updated_utc"):
            parsed = parse_dt(str(payload.get(key) or ""))
            if parsed:
                return parsed
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None


def age_hours_since(value: datetime | None) -> float | None:
    if not value:
        return None
    return max(0.0, (datetime.now(timezone.utc) - value).total_seconds() / 3600.0)


def parse_dt(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def read_json(path: Path, default: Any) -> Any:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return payload


def get_nested(payload: dict[str, Any], path: list[str], default: Any = None) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def registry_gate_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    blockers = [
        row
        for row in payload.get("implementation_health", [])
        if isinstance(row, dict) and row.get("routing_required") and not row.get("routing_eligible")
    ]
    return {
        "policy": payload.get("policy"),
        "created_utc": payload.get("created_utc"),
        "trigger_state": payload.get("trigger_state"),
        "gate_passed": payload.get("trigger_state") != "RED",
        "abstraction_registry_gap_count": summary.get("abstraction_registry_gap_count"),
        "stable_capability_field_gap_count": summary.get("stable_capability_field_gap_count"),
        "stable_capability_field_health_red_count": summary.get("stable_capability_field_health_red_count"),
        "stable_capability_field_health_yellow_count": summary.get("stable_capability_field_health_yellow_count"),
        "route_validator_viea_spine_view_ready": summary.get("route_validator_viea_spine_view_ready"),
        "route_validator_viea_spine_record_count": summary.get("route_validator_viea_spine_record_count"),
        "route_validator_viea_spine_missing_group_count": summary.get("route_validator_viea_spine_missing_group_count"),
        "project_steward_status": summary.get("project_steward_status"),
        "project_steward_hard_gap_count": summary.get("project_steward_hard_gap_count"),
        "project_steward_warning_count": summary.get("project_steward_warning_count"),
        "active_project_work_contract_count": summary.get("active_project_work_contract_count"),
        "event_taint_record_count": summary.get("event_taint_record_count"),
        "module_card_count": summary.get("module_card_count"),
        "major_surface_module_card_coverage_ratio": summary.get("major_surface_module_card_coverage_ratio"),
        "steward_decision_count": summary.get("steward_decision_count"),
        "abstraction_health_red_count": summary.get("abstraction_health_red_count"),
        "abstraction_health_yellow_count": summary.get("abstraction_health_yellow_count"),
        "implementation_routing_blocker_count": summary.get("implementation_routing_blocker_count"),
        "route_evidence_output_count": summary.get("route_evidence_output_count"),
        "blocked_route_evidence_output_count": summary.get("blocked_route_evidence_output_count"),
        "supporting_evidence_output_count": summary.get("supporting_evidence_output_count"),
        "missing_supporting_evidence_output_count": summary.get("missing_supporting_evidence_output_count"),
        "registry_governance_violation_count": summary.get("registry_governance_violation_count"),
        "registry_hard_governance_violation_count": summary.get("registry_hard_governance_violation_count"),
        "routing_eligible_implementation_count": summary.get("routing_eligible_implementation_count"),
        "routing_role_count": summary.get("routing_role_count"),
        "cleanup_queue_count": summary.get("registry_cleanup_queue_count"),
        "cleanup_queue_steward_decision_count": summary.get("cleanup_queue_steward_decision_count"),
        "cleanup_queue_steward_uncovered_count": summary.get("cleanup_queue_steward_uncovered_count"),
        "cleanup_queue_steward_coverage_ratio": summary.get("cleanup_queue_steward_coverage_ratio"),
        "blockers": [
            {
                "implementation_id": row.get("implementation_id"),
                "abstraction_id": row.get("abstraction_id"),
                "surface_id": row.get("surface_id"),
                "blockers": row.get("blockers", []),
                "recommended_action": row.get("recommended_action", ""),
            }
            for row in blockers[:24]
        ],
        "rules": {
            "router_use": "Routers should select only routing_eligible implementations for the requested role.",
            "claim_boundary": "Learned-generation claims require learned_generation_claim_allowed plus candidate integrity evidence.",
            "stable_capability_field": "Routable implementations must bind to an SCF contract with exact identity, authority, state, evidence, route validation, leases, adaptation, migration, observability, and lifecycle/governance metadata.",
            "route_validator_viea_spine": "Route approval requires a ready materialized VIEA view with governance, failure, authority, and resource route records.",
            "project_steward": "Autonomous roadmap work should have a steward charter, active project work contract, event-taint coverage, module cards, and steward decisions before execution.",
        },
    }


def git_lines(command: list[str]) -> list[str]:
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return [normalize_path(line) for line in result.stdout.splitlines() if line.strip()]


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:16]


def normalize_path(value: str) -> str:
    return value.replace("\\", "/").lstrip("./")


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
