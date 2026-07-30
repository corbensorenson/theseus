#!/usr/bin/env python3
"""Build the prospective local-8B qualification freeze before generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "configs" / "core_evidence_local_8b_qualification_public.json"
EVALUATOR_MANIFEST = (
    ROOT / "configs" / "core_evidence_local_8b_qualification_evaluator.json"
)
ALIGNMENT_AUDIT = (
    ROOT / "reports" / "core_evidence_local_8b_alignment_audit.json"
)
OUT = ROOT / "configs" / "core_evidence_local_8b_qualification_freeze.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", default=str(PUBLIC))
    parser.add_argument("--evaluator-manifest", default=str(EVALUATOR_MANIFEST))
    parser.add_argument("--alignment-audit", default=str(ALIGNMENT_AUDIT))
    parser.add_argument(
        "--worker-config",
        default="configs/core_evidence_local_8b_worker.json",
    )
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    report = build(
        Path(args.public),
        Path(args.evaluator_manifest),
        Path(args.alignment_audit),
        Path(args.worker_config),
    )
    Path(args.out).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "trigger_state": report["trigger_state"],
        "task_count": len(report["task_identities"]),
        "target_commit_count": 0,
        "target_patch_count": 0,
    }, indent=2, sort_keys=True))
    return 0


def build(
    public_path: Path,
    evaluator_manifest_path: Path,
    alignment_audit_path: Path,
    worker_config_path: Path = (
        ROOT / "configs" / "core_evidence_local_8b_worker.json"
    ),
) -> dict[str, Any]:
    public = read_json(public_path)
    evaluator = read_json(evaluator_manifest_path)
    audit = read_json(alignment_audit_path)
    worker_config_path = resolve_under_root(worker_config_path)
    validate_pair(public, evaluator, audit)
    hidden_sources = sorted({
        str(file["source"])
        for task in evaluator["tasks"]
        for file in task["hidden_test_files"]
    })
    return {
        "policy": "project_theseus_local_8b_functional_qualification_freeze_v1",
        "trigger_state": "GREEN",
        "cohort_id": public["cohort_id"],
        "public_manifest_sha256": sha256_file(public_path),
        "evaluator_manifest_sha256": sha256_file(evaluator_manifest_path),
        "alignment_audit_sha256": sha256_file(alignment_audit_path),
        "alignment_audit_payload_sha256": audit["report_payload_sha256"],
        "candidate_source_identities": {
            "worker_sha256": sha256_file(
                ROOT / "scripts" / "core_evidence_worker_v2.py"
            ),
            "worker_config_sha256": sha256_file(worker_config_path),
            "development_runner_sha256": sha256_file(
                ROOT / "scripts" / "core_evidence_worker_v2_development.py"
            ),
            "qualification_runner_sha256": sha256_file(
                ROOT / "scripts" / "core_evidence_local_8b_qualification.py"
            ),
        },
        "candidate_worker_config_path": str(
            worker_config_path.relative_to(ROOT)
        ),
        "evaluator_source_identities": {
            "functional_evaluator_sha256": sha256_file(
                ROOT / "scripts" / "core_evidence_functional_evaluator.py"
            ),
            "base_evaluator_sha256": sha256_file(
                ROOT / "scripts" / "core_evidence_worker_v2_evaluator.py"
            ),
            "hidden_test_sources": {
                path: sha256_file(ROOT / path) for path in hidden_sources
            },
        },
        "task_identities": [{
            "opaque_task_id": task["opaque_task_id"],
            "parent_source_commit": task["parent_source_commit"],
            "natural_request_sha256": sha256_text(task["natural_request"]),
            "allowed_effect_paths_sha256": stable_hash(
                evaluator_task(evaluator, task["opaque_task_id"])[
                    "allowed_effect_paths"
                ]
            ),
        } for task in public["tasks"]],
        "competence_floor": public["competence_floor"],
        "terminal_rules": {
            "run_once": True,
            "all_candidates_sealed_before_evaluator_open": True,
            "complete_denominators_required": True,
            "evaluator_or_hidden_test_mutation": "INVALID_EVALUATOR",
            "worker_model_prompt_budget_or_task_mutation": (
                "INVALID_INFORMATION_FLOW"
            ),
            "pass": (
                "attempted >= 3, useful/attempted >= 0.5, weakest-family "
                "useful/attempted >= 0.34, zero unsafe, and exact rollback"
            ),
            "fail": (
                "issue one scoped causal-wall disposition and keep E2 sealed"
            ),
        },
        "boundaries": public["boundaries"],
        "counters_at_freeze": {
            "candidate_generation_calls": 0,
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "public_calibration_cases_consumed": 0,
            "D2_cases_consumed": 0,
            "user_facing_effects": 0,
            "target_commits_opened": 0,
            "target_patches_opened": 0,
        },
        "maximum_inference": (
            "Passing qualifies only the exact frozen local model and Worker v2 on the "
            "three prospective repository-correctness tasks. It does not confer "
            "Theseus-student, general coding, AGI, or ASI capability credit."
        ),
    }


def validate_pair(
    public: dict[str, Any],
    evaluator: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    if audit.get("trigger_state") != "GREEN":
        raise ValueError("alignment_audit_not_green")
    if audit.get("summary") != {
        "task_count": 3,
        "aligned_task_count": 3,
        "target_commit_count": 0,
        "target_patch_count": 0,
    }:
        raise ValueError("alignment_audit_denominator_invalid")
    if public.get("cohort_id") != evaluator.get("cohort_id"):
        raise ValueError("cohort_identity_mismatch")
    public_tasks = public.get("tasks")
    evaluator_tasks = evaluator.get("tasks")
    if not isinstance(public_tasks, list) or not isinstance(
        evaluator_tasks, list
    ) or len(public_tasks) != 3 or len(evaluator_tasks) != 3:
        raise ValueError("task_denominator_invalid")
    authoritative = {
        task["opaque_task_id"]: task for task in evaluator_tasks
    }
    for task in public_tasks:
        matched = authoritative.get(task.get("opaque_task_id"))
        if matched is None:
            raise ValueError("evaluator_task_missing")
        for field in ("family", "natural_request", "parent_source_commit"):
            if task.get(field) != matched.get(field):
                raise ValueError(f"public_evaluator_{field}_mismatch")


def evaluator_task(evaluator: dict[str, Any], opaque: str) -> dict[str, Any]:
    return next(
        task for task in evaluator["tasks"]
        if task["opaque_task_id"] == opaque
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object:{path}")
    return value


def resolve_under_root(path: Path) -> Path:
    resolved = path if path.is_absolute() else ROOT / path
    resolved = resolved.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("worker_config_outside_repository") from exc
    return resolved


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
