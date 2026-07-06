"""Govern old-project training sources before they enter Theseus training.

The old projects contain useful local rows, public dataset recipes, benchmark
fixtures, and holdouts. This audit keeps those lanes separate. It reads the
metadata emitted by ``old_project_registry_port.py`` and produces a small,
actionable admission plan for launch readiness, teacher evidence packets, and
future samplers.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = ROOT / "data" / "training_sources" / "old_project_registry_training_sources.json"
DEFAULT_REGISTRY_REPORT = ROOT / "reports" / "old_project_registry_port.json"
DEFAULT_OUT = ROOT / "reports" / "legacy_training_source_audit.json"
DEFAULT_MARKDOWN_OUT = ROOT / "reports" / "legacy_training_source_audit.md"
DEFAULT_ADMISSIONS_OUT = ROOT / "data" / "training_sources" / "legacy_training_admissions.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default=str(DEFAULT_SOURCES.relative_to(ROOT)))
    parser.add_argument("--registry-report", default=str(DEFAULT_REGISTRY_REPORT.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN_OUT.relative_to(ROOT)))
    parser.add_argument("--admissions-out", default=str(DEFAULT_ADMISSIONS_OUT.relative_to(ROOT)))
    args = parser.parse_args()

    sources_path = resolve(args.sources)
    registry_path = resolve(args.registry_report)
    source_payload = read_json(sources_path)
    registry_report = read_json(registry_path)
    report = build_report(source_payload, registry_report, sources_path, registry_path)

    admissions_out = resolve(args.admissions_out)
    write_json(admissions_out, report["admission_plan"])
    report["admission_plan_path"] = rel(admissions_out)

    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(
    source_payload: dict[str, Any],
    registry_report: dict[str, Any],
    sources_path: Path,
    registry_path: Path,
) -> dict[str, Any]:
    sources = [row for row in source_payload.get("sources", []) if isinstance(row, dict)]
    ready = [row for row in sources if row.get("training_use_state") == "ready_local_verified"]
    serious_ready = [
        row
        for row in ready
        if bool(row.get("serious_training_ready")) and bool(row.get("quality_gate_passed"))
    ]
    seed_ready = [
        row
        for row in ready
        if "supervision_seed.v3" in str(row.get("dataset_id") or "")
        or ("seed" in str(row.get("dataset_id") or "") and int(row.get("sample_count") or 0) <= 1000)
    ]
    candidate_after_review = [row for row in ready if row not in serious_ready and row not in seed_ready]
    recipe_only = [
        row
        for row in sources
        if row.get("train_allowed") is True
        and row.get("training_use_state") != "ready_local_verified"
        and not bool(row.get("local_exists"))
    ]
    blocked = [row for row in sources if row.get("train_allowed") is not True]
    hash_mismatches = [
        row
        for row in sources
        if row.get("local_exists")
        and row.get("expected_sha256")
        and row.get("actual_sha256")
        and row.get("expected_sha256") != row.get("actual_sha256")
    ]
    unsafe_ready = [
        row
        for row in ready
        if row.get("train_allowed") is not True
        or not bool(row.get("sha256_verified"))
        or not bool(row.get("decontamination_fail_closed"))
    ]
    public_claim_ready = [row for row in ready if bool(row.get("public_claim_ready"))]
    ready_without_exclusions = [
        row
        for row in ready
        if not isinstance(row.get("protected_benchmark_exclusions"), list)
        or not row.get("protected_benchmark_exclusions")
    ]
    registry_summary = registry_report.get("summary") if isinstance(registry_report.get("summary"), dict) else {}
    reference_answers_seen = int_or(registry_summary.get("reference_answers_seen"))
    reference_answers_redacted = int_or(registry_summary.get("reference_answers_redacted"))
    external_calls = int_or(source_payload.get("external_inference_calls")) + int_or(
        registry_report.get("external_inference_calls")
    )

    gates = [
        gate("sources_manifest_present", bool(sources), f"path={rel_or_abs(sources_path)} sources={len(sources)}"),
        gate("old_registry_report_present", bool(registry_report), rel_or_abs(registry_path)),
        gate("ready_local_sources_found", len(ready) > 0, f"ready={len(ready)}"),
        gate("serious_training_ready_source_found", len(serious_ready) > 0, ids(serious_ready)),
        gate("ready_sources_are_train_allowed_hash_verified_and_decontaminated", not unsafe_ready, ids(unsafe_ready)),
        gate("ready_sources_have_protected_benchmark_exclusions", not ready_without_exclusions, ids(ready_without_exclusions)),
        gate("no_ready_source_can_support_public_benchmark_claims", not public_claim_ready, ids(public_claim_ready)),
        gate("hash_mismatches_zero", not hash_mismatches, ids(hash_mismatches)),
        gate(
            "reference_answers_redacted",
            reference_answers_seen == reference_answers_redacted,
            f"seen={reference_answers_seen} redacted={reference_answers_redacted}",
        ),
        gate("metadata_only_no_bulk_copy", source_payload.get("copy_training_data") is False, "copy_training_data=false"),
        gate("external_inference_zero", external_calls == 0, f"external_inference_calls={external_calls}"),
    ]
    trigger_state = "GREEN" if all(item["passed"] for item in gates) else "RED"
    if trigger_state == "GREEN" and candidate_after_review:
        trigger_state = "YELLOW"

    admission_plan = {
        "policy": "project_theseus_legacy_training_admissions_v1",
        "created_utc": now(),
        "source_manifest": rel_or_abs(sources_path),
        "bulk_copy_training_data": False,
        "default_training_use": "metadata_only_until_sampler_explicitly_selects_a_hash_verified_source",
        "admit_for_tiny_dry_run": [admission_row(row, "tiny_dry_run_primary") for row in serious_ready],
        "seed_for_contract_tests": [admission_row(row, "contract_test_seed") for row in seed_ready],
        "candidate_after_quality_review": [admission_row(row, "quality_review_required") for row in candidate_after_review],
        "recipe_only_pending_fetch_and_hash": [recipe_row(row) for row in recipe_only[:40]],
        "blocked_not_train_allowed": [recipe_row(row) for row in blocked[:40]],
        "hard_invariants": [
            "Do not train on benchmark reference answers or answer digests.",
            "Do not use old-project benchmark case manifests as public score evidence.",
            "Do not bulk-copy legacy data without an explicit source admission record.",
            "Do not admit a source without train_allowed, verified hash when local, and protected benchmark exclusions.",
            "Do not treat external inference output as local learning evidence.",
        ],
        "next_sampler": {
            "preferred_dataset_id": serious_ready[0].get("dataset_id") if serious_ready else "",
            "max_rows_first_pass": 128,
            "must_report_lane_mix": True,
            "must_report_source_sha256": True,
            "must_remain_internal_training_only": True,
        },
        "external_inference_calls": 0,
    }

    return {
        "policy": "project_theseus_legacy_training_source_audit_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "sources_manifest": rel_or_abs(sources_path),
        "old_project_registry_report": rel_or_abs(registry_path),
        "summary": {
            "sources": len(sources),
            "ready_local_verified": len(ready),
            "serious_training_ready": len(serious_ready),
            "contract_test_seeds": len(seed_ready),
            "candidate_after_quality_review": len(candidate_after_review),
            "recipe_only_pending_fetch_and_hash": len(recipe_only),
            "blocked_not_train_allowed": len(blocked),
            "hash_mismatches": len(hash_mismatches),
            "unsafe_ready_sources": len(unsafe_ready),
            "ready_sources_without_benchmark_exclusions": len(ready_without_exclusions),
            "public_claim_ready_sources": len(public_claim_ready),
            "reference_answers_seen": reference_answers_seen,
            "reference_answers_redacted": reference_answers_redacted,
            "external_inference_calls": external_calls,
        },
        "primary_training_candidates": [admission_row(row, "tiny_dry_run_primary") for row in serious_ready],
        "contract_test_candidates": [admission_row(row, "contract_test_seed") for row in seed_ready],
        "review_candidates": [admission_row(row, "quality_review_required") for row in candidate_after_review],
        "gates": gates,
        "admission_plan": admission_plan,
        "external_inference_calls": 0,
    }


def admission_row(row: dict[str, Any], use_state: str) -> dict[str, Any]:
    return {
        "dataset_id": row.get("dataset_id"),
        "use_state": use_state,
        "source_uri": row.get("source_uri"),
        "local_path": row.get("local_path"),
        "sample_count": row.get("sample_count"),
        "family": row.get("family"),
        "modality": row.get("modality"),
        "intended_training_phases": row.get("intended_training_phases") or [],
        "license_spdx": row.get("license_spdx"),
        "usage_restrictions": row.get("usage_restrictions") or [],
        "sha256": row.get("actual_sha256") or row.get("expected_sha256"),
        "decontamination_fail_closed": bool(row.get("decontamination_fail_closed")),
        "protected_benchmark_exclusions": row.get("protected_benchmark_exclusions") or [],
        "not_public_benchmark_claim_evidence": True,
    }


def recipe_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": row.get("dataset_id"),
        "source_uri": row.get("source_uri"),
        "family": row.get("family"),
        "modality": row.get("modality"),
        "training_use_state": row.get("training_use_state"),
        "train_allowed": bool(row.get("train_allowed")),
        "license_spdx": row.get("license_spdx"),
        "sample_count": row.get("sample_count"),
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Legacy Training Source Audit",
        "",
        f"- Trigger state: `{report['trigger_state']}`",
        f"- Sources: `{summary['sources']}`",
        f"- Ready local verified: `{summary['ready_local_verified']}`",
        f"- Serious training ready: `{summary['serious_training_ready']}`",
        f"- Hash mismatches: `{summary['hash_mismatches']}`",
        f"- Reference answers redacted: `{summary['reference_answers_redacted']}/{summary['reference_answers_seen']}`",
        "",
        "## Primary Candidates",
    ]
    for row in report.get("primary_training_candidates", []):
        lines.append(f"- `{row['dataset_id']}` samples=`{row['sample_count']}` phases=`{', '.join(row['intended_training_phases'])}`")
    if not report.get("primary_training_candidates"):
        lines.append("- None.")
    lines.extend(["", "## Gates"])
    for row in report.get("gates", []):
        mark = "PASS" if row["passed"] else "FAIL"
        lines.append(f"- `{mark}` `{row['gate']}`: {row['evidence']}")
    lines.append("")
    return "\n".join(lines)


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def ids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("dataset_id") or row.get("source_uri") or "unknown") for row in rows[:20]]


def int_or(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def resolve(value: str) -> Path:
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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
