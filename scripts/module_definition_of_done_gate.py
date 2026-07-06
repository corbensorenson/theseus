#!/usr/bin/env python3
"""Book-quality definition-of-done gate for Theseus registry modules."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "module_definition_of_done.json"
DEFAULT_REPORT = ROOT / "reports" / "module_definition_of_done.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "module_definition_of_done.md"
DEFAULT_MODULE_RECORDS = ROOT / "reports" / "module_definition_cards.jsonl"
DEFAULT_CROSSWALK = ROOT / "reports" / "book_to_theseus_crosswalk.json"
DEFAULT_SOURCE_BACKLOG_WORK_CARDS = ROOT / "reports" / "book_to_theseus_backlog_work_cards.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default=rel(DEFAULT_REPORT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--module-records-out", default=rel(DEFAULT_MODULE_RECORDS))
    parser.add_argument("--crosswalk", default=rel(DEFAULT_CROSSWALK))
    parser.add_argument("--source-backlog-work-cards-out", default=rel(DEFAULT_SOURCE_BACKLOG_WORK_CARDS))
    args = parser.parse_args()

    started = time.perf_counter()
    config_path = resolve(args.config)
    config = read_json(config_path)
    report = build_report(config_path, config, started, crosswalk_path=resolve(args.crosswalk))
    write_json(resolve(args.out), report)
    write_jsonl(resolve(args.module_records_out), report["module_records"])
    write_jsonl(resolve(args.source_backlog_work_cards_out), report["source_backlog_routing"]["work_cards"])
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(gate_view(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(config_path: Path, config: dict[str, Any], started: float, *, crosswalk_path: Path) -> dict[str, Any]:
    inputs = dict_value(config.get("inputs"))
    manifest = read_json(resolve(str(inputs.get("project_manifest_registry") or "")))
    steward = read_json(resolve(str(inputs.get("project_steward") or "")))
    registry_report = read_json(resolve(str(inputs.get("registry_report") or "")))
    crosswalk = read_json(crosswalk_path)
    book_sources = audit_book_sources(config)
    surfaces = [row for row in list_dicts(manifest.get("surfaces")) if bool(row.get("major_surface"))]
    cards = list_dicts(steward.get("module_dod_cards")) or list_dicts(steward.get("module_cards"))
    cards_by_surface = {str(row.get("surface_id") or ""): row for row in cards}
    module_records = [audit_module_surface(surface, cards_by_surface.get(str(surface.get("id") or "")), config) for surface in surfaces]
    report_family = audit_report_family_policy(surfaces, registry_report, config)
    steward_decision = audit_steward_decisions(steward, config)
    source_backlog_routing = audit_source_backlog_routing(crosswalk_path, crosswalk)

    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for source in book_sources:
        if not source["present"]:
            warnings.append(item_gap(source["path"], "book_standard_source_missing", source, "warning"))
    for record in module_records:
        hard_gaps.extend(record["hard_gaps"])
        warnings.extend(record["warnings"])
    hard_gaps.extend(report_family["hard_gaps"])
    warnings.extend(report_family["warnings"])
    hard_gaps.extend(steward_decision["hard_gaps"])
    warnings.extend(steward_decision["warnings"])
    hard_gaps.extend(source_backlog_routing["hard_gaps"])
    warnings.extend(source_backlog_routing["warnings"])

    trigger_state = "GREEN"
    if hard_gaps:
        trigger_state = "RED"
    elif warnings:
        trigger_state = "YELLOW"

    summary = {
        "config": rel(config_path),
        "major_surface_count": len(surfaces),
        "module_record_count": len(module_records),
        "module_records_ready": sum(1 for row in module_records if row["ready"]),
        "major_surface_coverage_ratio": round(safe_ratio(sum(1 for row in module_records if row["module_card_present"]), len(surfaces)), 6),
        "book_standard_source_count": len(book_sources),
        "book_standard_source_present_count": sum(1 for row in book_sources if row["present"]),
        "surfaces_over_report_cap": report_family["summary"]["surfaces_over_report_cap"],
        "cleanup_queue_count": report_family["summary"]["cleanup_queue_count"],
        "stale_latest_view_count": report_family["summary"]["stale_latest_view_count"],
        "steward_decision_count": steward_decision["summary"]["decision_count"],
        "negative_evidence_linked": steward_decision["summary"]["negative_evidence_linked"],
        "source_backlog_item_count": source_backlog_routing["summary"]["backlog_item_count"],
        "source_backlog_work_card_count": source_backlog_routing["summary"]["work_card_count"],
        "source_backlog_steward_decision_candidate_count": source_backlog_routing["summary"]["steward_decision_candidate_count"],
        "source_backlog_route_smoke_passed": source_backlog_routing["summary"]["route_smoke_passed"],
        "hard_gap_count": len(hard_gaps),
        "warning_count": len(warnings),
    }
    return {
        "policy": "project_theseus_module_definition_of_done_gate_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": summary,
        "book_standard_sources": book_sources,
        "module_records": module_records,
        "report_family_policy": report_family,
        "steward_decision_policy": steward_decision,
        "source_backlog_routing": source_backlog_routing,
        "hard_gaps": hard_gaps,
        "warnings": warnings,
        "rules": {
            "module_card": "Every major surface needs a module card with problem, interface, invariants, failure modes, validation, evidence, non-claims, and deprecation route.",
            "source_crosswalk": "The card must be normalized into a source crosswalk so AI_book ideas map to Theseus surfaces.",
            "non_claim": "Module-card health is repository quality evidence, not learned-model capability evidence.",
            "retirement": "Stale modules must be merged, deprecated, or retired through visible steward decisions.",
        },
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }


def audit_module_surface(surface: dict[str, Any], card: dict[str, Any] | None, config: dict[str, Any]) -> dict[str, Any]:
    surface_id = str(surface.get("id") or "")
    required_fields = set(str(x) for x in list_values(config.get("required_module_card_fields")))
    minimums = dict_value(config.get("minimum_list_lengths"))
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if card is None:
        hard_gaps.append(item_gap(surface_id, "missing_module_card", {"surface_id": surface_id}))
        card = {}
    missing = sorted(field for field in required_fields if field not in card or empty_value(card.get(field)))
    if missing:
        hard_gaps.append(item_gap(surface_id, "missing_required_module_card_fields", {"missing": missing}))
    for field, min_count in minimums.items():
        actual = len(list_values(card.get(field)))
        if actual < int(min_count):
            hard_gaps.append(item_gap(surface_id, f"{field}_below_minimum", {"actual": actual, "minimum": min_count}))
    evidence_refs = [str(x) for x in list_values(card.get("evidence_refs"))]
    evidence_states = [evidence_state(ref) for ref in evidence_refs]
    missing_evidence = [row for row in evidence_states if row["checked"] and not row["present"]]
    if missing_evidence:
        warnings.append(item_gap(surface_id, "evidence_ref_missing_or_generated_later", {"missing": missing_evidence}, "warning"))
    declared_outputs = list_values(surface.get("report_outputs"))
    source_crosswalk = source_crosswalk_for(surface, config)
    ready = not hard_gaps
    return {
        "id": str(card.get("id") or f"module.{surface_id}"),
        "surface_id": surface_id,
        "owner_field": str(surface.get("abstraction_id") or ""),
        "surface_role": str(surface.get("role") or ""),
        "artifact_type": str(surface.get("artifact_type") or ""),
        "module_card_present": bool(card),
        "ready": ready,
        "problem": str(card.get("problem") or ""),
        "interface": str(card.get("interface") or ""),
        "invariants": list_values(card.get("invariants")),
        "failure_modes": list_values(card.get("failure_modes")),
        "minimal_implementation": str(card.get("minimal_implementation") or ""),
        "validation_commands": list_values(card.get("validation_commands")),
        "evidence_refs": evidence_refs,
        "evidence_states": evidence_states,
        "non_claims": list_values(card.get("non_claims")),
        "deprecation_route": str(card.get("deprecation_route") or ""),
        "source_crosswalk": source_crosswalk,
        "declared_report_output_count": len(declared_outputs),
        "hard_gap_count": len(hard_gaps),
        "warning_count": len(warnings),
        "hard_gaps": hard_gaps,
        "warnings": warnings,
    }


def audit_book_sources(config: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for raw in list_values(config.get("book_standard_sources")):
        path = resolve(str(raw))
        out.append({"path": str(raw), "resolved": str(path), "present": path.exists()})
    return out


def audit_report_family_policy(surfaces: list[dict[str, Any]], registry_report: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    policy = dict_value(config.get("report_family_policy"))
    max_outputs = int(policy.get("max_declared_report_outputs_per_surface") or 80)
    max_cleanup = int(policy.get("max_cleanup_queue_count") or 64)
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    over_cap = []
    for surface in surfaces:
        count = len(list_values(surface.get("report_outputs")))
        if count > max_outputs:
            over_cap.append({"surface_id": surface.get("id"), "declared_report_output_count": count, "max": max_outputs})
    if over_cap:
        hard_gaps.append(item_gap("report_family_policy", "surface_report_output_cap_exceeded", {"surfaces": over_cap}))
    cleanup_queue = int(registry_report.get("cleanup_queue_count") or 0)
    if cleanup_queue > max_cleanup:
        warnings.append(item_gap("report_family_policy", "cleanup_queue_above_policy_target", {"cleanup_queue_count": cleanup_queue, "max": max_cleanup}, "warning"))
    stale = registry_report.get("registry_hard_governance_violation_count")
    stale_count = int(stale or 0)
    if stale_count and bool(policy.get("stale_latest_view_is_hard")):
        hard_gaps.append(item_gap("report_family_policy", "registry_latest_view_hard_violations_present", {"count": stale_count}))
    return {
        "summary": {
            "max_declared_report_outputs_per_surface": max_outputs,
            "surfaces_over_report_cap": len(over_cap),
            "cleanup_queue_count": cleanup_queue,
            "max_cleanup_queue_count": max_cleanup,
            "stale_latest_view_count": stale_count,
        },
        "surfaces_over_report_cap": over_cap,
        "hard_gaps": hard_gaps,
        "warnings": warnings,
    }


def audit_steward_decisions(steward: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    decisions = list_dicts(steward.get("steward_decisions"))
    negative = [row for row in decisions if "negative" in json.dumps(row).lower() or "failed" in json.dumps(row).lower()]
    policy = dict_value(config.get("steward_decision_policy"))
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if bool(policy.get("negative_evidence_must_remain_linked")) and not negative:
        hard_gaps.append(item_gap("steward_decisions", "negative_evidence_decision_missing", {}))
    if bool(policy.get("capability_claim_from_module_card_allowed")):
        hard_gaps.append(item_gap("steward_decisions", "module_card_capability_claim_allowed", {}))
    return {
        "summary": {
            "decision_count": len(decisions),
            "negative_evidence_linked": bool(negative),
            "negative_evidence_decision_count": len(negative),
        },
        "negative_evidence_decisions": negative,
        "hard_gaps": hard_gaps,
        "warnings": warnings,
    }


def source_crosswalk_for(surface: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    surface_id = str(surface.get("id") or "")
    refs = []
    for raw in list_values(config.get("book_standard_sources")):
        refs.append({
            "source": str(raw),
            "mapping": "book_quality_dod_or_project_operating_charter",
            "surface_id": surface_id,
        })
    for path in list_values(surface.get("patterns"))[:8]:
        refs.append({
            "source": str(path),
            "mapping": "surface_owned_pattern",
            "surface_id": surface_id,
        })
    return refs


def audit_source_backlog_routing(crosswalk_path: Path, crosswalk: dict[str, Any]) -> dict[str, Any]:
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    backlog_items = list_dicts(crosswalk.get("roadmap_backlog_items"))
    phase_items = {int_or(row.get("phase"), -1): row for row in list_dicts(crosswalk.get("items"))}
    work_cards = route_source_backlog_work_cards(backlog_items, phase_items)
    steward_decision_candidates = route_source_backlog_steward_decisions(backlog_items, phase_items)
    smoke = source_backlog_route_smoke()
    if not crosswalk:
        warnings.append(item_gap("source_backlog_routing", "book_to_theseus_crosswalk_missing", {"path": rel(crosswalk_path)}, "warning"))
    if not smoke["passed"]:
        hard_gaps.append(item_gap("source_backlog_routing", "route_smoke_failed", smoke))
    if len(backlog_items) != len(work_cards) or len(backlog_items) != len(steward_decision_candidates):
        hard_gaps.append(
            item_gap(
                "source_backlog_routing",
                "backlog_route_count_mismatch",
                {
                    "backlog_item_count": len(backlog_items),
                    "work_card_count": len(work_cards),
                    "steward_decision_candidate_count": len(steward_decision_candidates),
                },
            )
        )
    return {
        "policy": "project_theseus_source_backlog_module_dod_routing_v1",
        "crosswalk": rel(crosswalk_path),
        "summary": {
            "backlog_item_count": len(backlog_items),
            "work_card_count": len(work_cards),
            "steward_decision_candidate_count": len(steward_decision_candidates),
            "route_smoke_passed": smoke["passed"],
        },
        "work_cards": work_cards,
        "steward_decision_candidates": steward_decision_candidates,
        "route_smoke": smoke,
        "hard_gaps": hard_gaps,
        "warnings": warnings,
        "rules": {
            "source_change": "Changed AI_book sources create review work; they do not become Theseus capability claims.",
            "module_dod": "Every stale-source backlog row must route to a module work card with validation and non-claims.",
            "steward": "Every stale-source backlog row must route to a steward-decision candidate before roadmap evidence changes.",
        },
    }


def route_source_backlog_work_cards(backlog_items: list[dict[str, Any]], phase_items: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    cards = []
    for item in backlog_items:
        phase_id = int_or(item.get("phase"), -1)
        phase = phase_items.get(phase_id, {})
        surface_id = str(phase.get("registry_surface_id") or "active_docs")
        changed_sources = [str(path) for path in list_values(item.get("changed_source_paths"))]
        cards.append(
            {
                "record_type": "module_dod_work_card",
                "id": stable_id("module_dod_work_card", phase_id, item.get("backlog_id"), changed_sources),
                "source_backlog_id": str(item.get("backlog_id") or ""),
                "phase": phase_id,
                "surface_id": surface_id,
                "title": f"Review AI_book source drift for phase {phase_id}: {item.get('title') or phase.get('title') or 'Untitled'}",
                "problem": "An authored AI_book source linked to this roadmap phase changed after the previous crosswalk baseline.",
                "interface": "reports/book_to_theseus_crosswalk.json -> module DoD work card -> steward decision before roadmap evidence changes.",
                "changed_source_paths": changed_sources,
                "required_validation": [
                    "python3 scripts/roadmap_implementation_gate.py --gate",
                    "python3 scripts/module_definition_of_done_gate.py",
                    "python3 scripts/theseus_project_registry.py --gate",
                ],
                "required_review": [
                    "check whether the changed AI_book source alters the phase contract",
                    "update the registry-owned matrix fields if implementation scope changed",
                    "leave capability claims unchanged until implementation and tests exist",
                ],
                "non_claims": [
                    "source drift review is not model capability evidence",
                    "book source text is not training data",
                    "module DoD routing is not a public benchmark result",
                ],
                "evidence_refs": [rel(DEFAULT_CROSSWALK), "configs/roadmap_implementation_matrix.json"],
                "status": "queued_for_steward_review",
                "public_training_rows_written": 0,
                "external_inference_calls": 0,
                "fallback_return_count": 0,
            }
        )
    return cards


def route_source_backlog_steward_decisions(backlog_items: list[dict[str, Any]], phase_items: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    decisions = []
    for item in backlog_items:
        phase_id = int_or(item.get("phase"), -1)
        phase = phase_items.get(phase_id, {})
        decisions.append(
            {
                "record_type": "steward_decision_candidate",
                "id": stable_id("steward_decision_candidate", phase_id, item.get("backlog_id")),
                "source_backlog_id": str(item.get("backlog_id") or ""),
                "target": f"roadmap phase {phase_id}: {item.get('title') or phase.get('title') or 'Untitled'}",
                "decision": "review_ai_book_source_change_before_roadmap_claim_change",
                "reason": "Roadmap/source alignment must remain registry-owned and evidence-backed when source checksums change.",
                "allowed_next_actions": [
                    "confirm no contract change and clear backlog",
                    "update matrix missing_items/smallest_next_patch/current_evidence",
                    "open a bounded implementation work contract for the affected surface",
                ],
                "forbidden_next_actions": [
                    "treat source change as implemented capability",
                    "train on public benchmark content",
                    "count router/template/tool behavior as learned generation",
                ],
                "evidence_refs": [rel(DEFAULT_CROSSWALK), "reports/roadmap_implementation_gate.json"],
                "status": "candidate_not_applied",
            }
        )
    return decisions


def source_backlog_route_smoke() -> dict[str, Any]:
    sample = [
        {
            "record_type": "roadmap_backlog_item",
            "backlog_id": "book_to_theseus_backlog-smoke",
            "phase": 19,
            "title": "Book-To-Theseus Backlog And Evidence Synchronization",
            "changed_source_paths": ["chapters/prototype-roadmap.qmd"],
        }
    ]
    phase_items = {
        19: {
            "phase": 19,
            "title": "Book-To-Theseus Backlog And Evidence Synchronization",
            "registry_surface_id": "active_docs",
        }
    }
    cards = route_source_backlog_work_cards(sample, phase_items)
    decisions = route_source_backlog_steward_decisions(sample, phase_items)
    passed = (
        len(cards) == 1
        and len(decisions) == 1
        and cards[0].get("surface_id") == "active_docs"
        and cards[0].get("public_training_rows_written") == 0
        and decisions[0].get("status") == "candidate_not_applied"
    )
    return {
        "policy": "project_theseus_source_backlog_route_smoke_v1",
        "passed": passed,
        "sample_backlog_count": len(sample),
        "sample_work_card_count": len(cards),
        "sample_steward_decision_candidate_count": len(decisions),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def evidence_state(ref: str) -> dict[str, Any]:
    if not ref or ":" in ref and not ref.startswith("/") and not ref.startswith("reports/") and not ref.startswith("docs/") and not ref.startswith("configs/") and not ref.startswith("scripts/"):
        return {"ref": ref, "checked": False, "present": None}
    path = resolve(ref.split(":", 1)[0])
    return {"ref": ref, "checked": True, "present": path.exists(), "resolved": str(path)}


def int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256(json.dumps([str(part) for part in parts], sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def gate_view(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": report["policy"],
        "trigger_state": report["trigger_state"],
        "summary": report["summary"],
        "hard_gaps": report["hard_gaps"][:20],
        "warnings": report["warnings"][:20],
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Module Definition Of Done Gate",
        "",
        f"- State: `{report['trigger_state']}`",
        f"- Major surfaces: `{summary['major_surface_count']}`",
        f"- Ready module records: `{summary['module_records_ready']}` / `{summary['module_record_count']}`",
        f"- Major-surface coverage ratio: `{summary['major_surface_coverage_ratio']}`",
        f"- Book standard sources present: `{summary['book_standard_source_present_count']}` / `{summary['book_standard_source_count']}`",
        f"- Stale latest-view hard violations: `{summary['stale_latest_view_count']}`",
        f"- Cleanup queue count: `{summary['cleanup_queue_count']}`",
        f"- Negative evidence linked: `{summary['negative_evidence_linked']}`",
        f"- Source backlog work cards: `{summary['source_backlog_work_card_count']}`",
        f"- Source backlog steward decision candidates: `{summary['source_backlog_steward_decision_candidate_count']}`",
        f"- Source backlog route smoke passed: `{summary['source_backlog_route_smoke_passed']}`",
        f"- Hard gaps: `{summary['hard_gap_count']}`",
        f"- Warnings: `{summary['warning_count']}`",
        "",
        "## Module Records",
    ]
    for row in report["module_records"]:
        lines.append(f"- `{row['surface_id']}`: ready `{row['ready']}`, owner `{row['owner_field']}`, reports `{row['declared_report_output_count']}`")
    if report["hard_gaps"]:
        lines.extend(["", "## Hard Gaps"])
        for gap in report["hard_gaps"][:30]:
            lines.append(f"- `{gap['item_id']}`: {gap['reason']}")
    if report["warnings"]:
        lines.extend(["", "## Warnings"])
        for gap in report["warnings"][:30]:
            lines.append(f"- `{gap['item_id']}`: {gap['reason']}")
    lines.append("")
    return "\n".join(lines)


def gate(gate_id: str, passed: bool, severity: str, detail: Any) -> dict[str, Any]:
    return {"id": gate_id, "passed": bool(passed), "severity": severity, "detail": detail}


def item_gap(item_id: str, reason: str, detail: Any, severity: str = "hard") -> dict[str, Any]:
    return {"item_id": item_id, "reason": reason, "severity": severity, "detail": detail}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (ROOT / candidate).resolve()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_values(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def list_dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in list_values(value) if isinstance(row, dict)]


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict)):
        return not value
    return False


if __name__ == "__main__":
    raise SystemExit(main())
