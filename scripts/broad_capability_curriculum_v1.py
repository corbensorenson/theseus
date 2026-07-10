#!/usr/bin/env python3
"""Build a broad capability curriculum manifest from admitted sources.

The output is an index over existing admitted rows, not a new synthetic suite.
It is meant to keep the next survival-lane training run broad without letting
public benchmark payloads or fallback/template artifacts slip in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADMISSION = ROOT / "reports" / "training_data_admission_v1.json"
DEFAULT_OUT = ROOT / "reports" / "broad_capability_curriculum_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "broad_capability_curriculum_v1.md"
DEFAULT_INDEX = ROOT / "data" / "training_sources" / "broad_capability_curriculum_v1_index.jsonl"
DEFAULT_TRAINING_SOURCES = ROOT / "data" / "training_sources" / "broad_capability_curriculum_v1_training_sources.json"

CAPABILITY_FAMILIES = {
    "return_type_shape": {
        "tokens": {"return", "type", "shape", "interface", "contract", "typed", "schema"},
        "priority": "critical",
    },
    "io_contracts": {
        "tokens": {"stdin", "io", "parse", "json", "csv", "line", "encoding", "contract"},
        "priority": "critical",
    },
    "parsing_encoding": {
        "tokens": {"parse", "parser", "parsing", "encoding", "decode", "string", "text", "token"},
        "priority": "critical",
    },
    "edge_cases": {
        "tokens": {"edge", "boundary", "empty", "invalid", "none", "singleton", "adversarial"},
        "priority": "critical",
    },
    "algorithm_selection": {
        "tokens": {"algorithm", "planning", "numeric", "prefix", "graph", "sort", "gcd", "state", "loop"},
        "priority": "critical",
    },
    "repo_context_repair": {
        "tokens": {"repair", "repo", "tool", "trace", "capsule", "private", "code", "patch"},
        "priority": "high",
    },
    "long_context_vcm_recovery": {
        "tokens": {"vcm", "memory", "context", "long", "recovery", "conversation"},
        "priority": "high",
    },
    "sts_candidate_ranking": {
        "tokens": {"sts", "rank", "ranker", "candidate", "selector", "fanout"},
        "priority": "high",
    },
    "tool_task_planning": {
        "tokens": {"dogfood", "tool", "task", "planning", "operator", "assistant", "lane"},
        "priority": "high",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", default=rel(DEFAULT_ADMISSION))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--index-out", default=rel(DEFAULT_INDEX))
    parser.add_argument("--training-sources-out", default=rel(DEFAULT_TRAINING_SOURCES))
    parser.add_argument("--max-rows-per-source", type=int, default=640)
    args = parser.parse_args()

    started = time.perf_counter()
    report, index_rows = build_report(args, started=started)
    training_sources = build_training_source_manifest(index_rows)
    write_json(resolve(args.out), report)
    write_jsonl(resolve(args.index_out), index_rows)
    write_json(resolve(args.training_sources_out), training_sources)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps({
        "trigger_state": report.get("trigger_state"),
        "summary": report.get("summary"),
        "training_contract": report.get("training_contract"),
        "failed_hard_gates": [
            row.get("name") for row in report.get("gates", [])
            if row.get("severity") == "hard" and row.get("passed") is not True
        ],
    }, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace, *, started: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    admission_path = resolve(args.admission)
    admission = read_json(admission_path)
    sources = [
        row
        for row in admission.get("source_admissions", [])
        if isinstance(row, dict) and row.get("allowed_for_training")
    ]
    candidate_lineage = admission.get("candidate_lineage") if isinstance(admission.get("candidate_lineage"), dict) else {}
    candidate_lineage_summary = candidate_lineage.get("summary") if isinstance(candidate_lineage.get("summary"), dict) else {}
    units = []
    index_rows = []
    family_counts: Counter[str] = Counter()
    blocked_public_sources = []
    total_row_budget = 0
    max_rows = max(1, int(args.max_rows_per_source))

    for source in sources:
        families = classify_source(source)
        if source.get("public_benchmark_payload_detected"):
            blocked_public_sources.append(source.get("path"))
            continue
        row_budget = min(max_rows, int(source.get("row_count") or 0))
        total_row_budget += row_budget
        family_counts.update(families)
        unit = curriculum_unit(source, families=families, row_budget=row_budget)
        units.append(unit)
        index_rows.append(index_row(source, unit))

    coverage = {
        family: {
            "present": family_counts.get(family, 0) > 0,
            "source_count": family_counts.get(family, 0),
            "priority": spec["priority"],
        }
        for family, spec in CAPABILITY_FAMILIES.items()
    }
    critical_missing = [
        family
        for family, row in coverage.items()
        if row["priority"] == "critical" and not row["present"]
    ]
    high_missing = [
        family
        for family, row in coverage.items()
        if row["priority"] == "high" and not row["present"]
    ]
    gates = [
        gate("admission_report_present", bool(admission), rel(admission_path), "hard"),
        gate("admission_not_red", admission.get("trigger_state") in {"GREEN", "YELLOW"}, admission.get("trigger_state"), "hard"),
        gate("training_sources_available", len(sources) > 0, len(sources), "hard"),
        gate(
            "candidate_level_admission_filter_ready",
            candidate_lineage.get("trigger_state") in {"GREEN", "YELLOW"}
            and candidate_lineage_summary.get("admitted_hash_filter_ready") is True,
            {
                "state": candidate_lineage.get("trigger_state"),
                "admitted_candidate_count": candidate_lineage_summary.get("admitted_candidate_count"),
                "ledger": (candidate_lineage.get("candidate_receipt_ledger") or {}).get("path"),
            },
            "hard",
        ),
        gate("public_benchmark_sources_not_selected", not blocked_public_sources, blocked_public_sources[:20], "hard"),
        gate("critical_capability_coverage_complete", not critical_missing, critical_missing, "hard"),
        gate("high_capability_coverage_recorded", isinstance(high_missing, list), high_missing, "warning"),
        gate("curriculum_index_rows_written", len(index_rows) > 0, len(index_rows), "hard"),
        gate("fallback_returns_zero_by_admission", not any(int(row.get("fallback_return_count") or 0) > 0 for row in sources), 0, "hard"),
        gate("raw_user_text_zero_by_admission", not any(int(row.get("raw_user_text_count") or 0) > 0 for row in sources), 0, "hard"),
        gate("external_inference_zero", sum(int(row.get("external_inference_calls") or 0) for row in sources) == 0, 0, "hard"),
    ]
    hard_failed = [row for row in gates if row["severity"] == "hard" and not row["passed"]]
    warning_failed = [row for row in gates if row["severity"] != "hard" and not row["passed"]]
    trigger_state = "GREEN" if not hard_failed else "RED"
    if trigger_state == "GREEN" and warning_failed:
        trigger_state = "YELLOW"

    report = {
        "policy": "project_theseus_broad_capability_curriculum_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "purpose": (
            "Construct a broad, non-cheating survival-lane curriculum from admitted local sources. "
            "This is an index over existing rows, not synthetic benchmark expansion."
        ),
        "inputs": {"admission": rel(admission_path)},
        "artifacts": {
            "index": rel(resolve(args.index_out)),
            "training_sources": rel(resolve(args.training_sources_out)),
        },
        "architecture_policy": {
            "survival_lane": "transformer_hybrid_structural_student",
            "survival_lane_reason": "matched local evidence currently favors transformer_control over SymLiquid",
            "symliquid_role": "bounded_matched_discovery_comparator_only",
            "symliquid_promotion_priority": False,
            "requires_equal_conditions": [
                "same admitted data",
                "same seeds",
                "same compute budget",
                "same verifier",
                "same STS and VCM views",
                "same candidate budget",
            ],
        },
        "training_contract": {
            "public_benchmark_training_allowed": False,
            "candidate_receipt_hash_filter_required": True,
            "candidate_receipt_ledger": (candidate_lineage.get("candidate_receipt_ledger") or {}).get("path"),
            "public_calibration_allowed": False,
            "teacher_distillation_allowed": False,
            "external_inference_allowed": False,
            "fallback_returns_allowed": False,
            "raw_user_text_allowed": False,
            "model_promotion_allowed_by_this_script": False,
            "intended_next_trainer": "transformer/hybrid structural student with VCM retrieval, STS ranking, grammar-masked full-body generation, and verifier fanout",
        },
        "summary": {
            "admitted_source_count": len(sources),
            "curriculum_unit_count": len(units),
            "curriculum_index_row_count": len(index_rows),
            "total_row_budget": total_row_budget,
            "critical_missing": critical_missing,
            "high_missing": high_missing,
            "family_source_counts": dict(sorted(family_counts.items())),
            "external_inference_calls": 0,
        },
        "capability_coverage": coverage,
        "curriculum_units": units,
        "gates": gates,
        "score_semantics": (
            "Curriculum routing metadata only. This script does not train, fetch public corpora, call a teacher, "
            "unlock public calibration, or promote a model."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }
    return report, index_rows


def build_training_source_manifest(index_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sources = []
    ready_sources = []
    for row in index_rows:
        source_path = resolve(str(row.get("source_path") or ""))
        expected_sha = str(row.get("source_sha256") or "")
        actual_sha = sha256_file(source_path) if source_path.exists() else ""
        verified = bool(expected_sha and actual_sha == expected_sha)
        source = {
            "dataset_id": row.get("source_id"),
            "local_path": rel(source_path),
            "sha256": expected_sha,
            "sha256_verified": verified,
            "training_use_state": "ready_local_verified" if verified else "blocked_hash_mismatch",
            "row_budget": int(row.get("row_budget") or 0),
            "capability_families": row.get("capability_families") if isinstance(row.get("capability_families"), list) else [],
            "public_benchmark_training_allowed": False,
            "fallback_returns_allowed": False,
            "raw_user_text_allowed": False,
            "external_inference_calls": 0,
        }
        sources.append(source)
        if verified:
            ready_sources.append(source)
    return {
        "policy": "project_theseus_broad_capability_training_sources_v1",
        "created_utc": now(),
        "source_project": "local_admitted_private_curriculum",
        "copy_training_data": False,
        "sources": sources,
        "ready_sources": ready_sources,
        "usage_policy": {
            "internal_training_only": True,
            "not_public_benchmark_claim_evidence": True,
            "train_only_when_sha256_and_decontamination_policy_pass": True,
            "public_benchmark_payloads_forbidden": True,
            "raw_user_text_forbidden": True,
            "fallback_returns_forbidden": True,
        },
        "summary": {
            "source_count": len(sources),
            "ready_source_count": len(ready_sources),
            "hash_mismatch_count": len(sources) - len(ready_sources),
            "external_inference_calls": 0,
        },
        "external_inference_calls": 0,
    }


def classify_source(source: dict[str, Any]) -> list[str]:
    tags = {str(tag).lower() for tag in source.get("source_family_tags", []) if str(tag)}
    text = " ".join([str(source.get("path") or ""), " ".join(tags)]).lower()
    out = []
    for family, spec in CAPABILITY_FAMILIES.items():
        tokens = spec["tokens"]
        if tags.intersection(tokens) or any(token in text for token in tokens):
            out.append(family)
    if not out:
        out.append("repo_context_repair")
    return sorted(set(out))


def curriculum_unit(source: dict[str, Any], *, families: list[str], row_budget: int) -> dict[str, Any]:
    source_id = str(source.get("source_id") or stable_hash(source.get("path")))
    return {
        "unit_id": stable_hash({"source_id": source_id, "families": families})[:16],
        "source_id": source_id,
        "path": source.get("path"),
        "source_kind": source.get("source_kind"),
        "capability_families": families,
        "row_count": source.get("row_count"),
        "row_budget": row_budget,
        "sha256": source.get("sha256"),
        "split_hash": source.get("split_hash"),
        "license_status": source.get("license_status"),
        "provenance_status": source.get("provenance_status"),
        "training_weight": training_weight(families, row_budget),
        "allowed_for_training": True,
        "public_benchmark_payload_detected": False,
        "fallback_return_count": source.get("fallback_return_count"),
        "raw_user_text_count": source.get("raw_user_text_count"),
        "external_inference_calls": source.get("external_inference_calls"),
    }


def index_row(source: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": "project_theseus_broad_capability_curriculum_index_row_v1",
        "unit_id": unit["unit_id"],
        "source_id": unit["source_id"],
        "source_path": unit["path"],
        "source_sha256": unit["sha256"],
        "split_hash": unit["split_hash"],
        "capability_families": unit["capability_families"],
        "row_budget": unit["row_budget"],
        "training_lane": "transformer_hybrid_structural_student_survival",
        "symliquid_use": "matched_comparator_only",
        "vcm_context_required": "load vcm_task_contexts before task sampling when available",
        "sts_required": "rank generated candidates with guarded STS selector when available",
        "verifier_required": "private verifier fanout only",
        "public_benchmark_training_allowed": False,
        "fallback_returns_allowed": False,
        "raw_user_text_allowed": False,
        "external_inference_calls": 0,
        "created_utc": now(),
    }


def training_weight(families: list[str], row_budget: int) -> float:
    critical = sum(1 for family in families if CAPABILITY_FAMILIES.get(family, {}).get("priority") == "critical")
    high = sum(1 for family in families if CAPABILITY_FAMILIES.get(family, {}).get("priority") == "high")
    base = 1.0 + 0.15 * critical + 0.05 * high
    if row_budget < 16:
        base *= 0.5
    return round(base, 4)


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# Broad Capability Curriculum v1",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- admitted sources: `{summary.get('admitted_source_count')}`",
        f"- curriculum units: `{summary.get('curriculum_unit_count')}`",
        f"- total row budget: `{summary.get('total_row_budget')}`",
        f"- critical missing: `{summary.get('critical_missing')}`",
        f"- high missing: `{summary.get('high_missing')}`",
        "",
        "## Architecture Policy",
        f"- survival lane: `{report.get('architecture_policy', {}).get('survival_lane')}`",
        f"- SymLiquid role: `{report.get('architecture_policy', {}).get('symliquid_role')}`",
        "",
        "## Capability Coverage",
    ]
    coverage = report.get("capability_coverage") if isinstance(report.get("capability_coverage"), dict) else {}
    for family, row in sorted(coverage.items()):
        lines.append(f"- `{family}`: present=`{row.get('present')}`, sources=`{row.get('source_count')}`, priority=`{row.get('priority')}`")
    lines.extend(["", "## Failed Gates"])
    failed = [row for row in report.get("gates", []) if isinstance(row, dict) and not row.get("passed")]
    if not failed:
        lines.append("- none")
    else:
        for row in failed:
            lines.append(f"- `{row.get('name')}` ({row.get('severity')}): `{row.get('evidence')}`")
    return "\n".join(lines) + "\n"


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
