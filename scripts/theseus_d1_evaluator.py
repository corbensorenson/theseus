#!/usr/bin/env python3
"""Independently score sealed D1 candidates without route labels."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_d1_cognitive_compilation as runner  # noqa: E402
import theseus_d1_evaluator_seal as seal  # noqa: E402


POLICY = "project_theseus_d1_route_blind_evaluation_v1"
EVALUATOR_POLICY = "project_theseus_d1_route_blind_evaluator_v1"
ORACLE = "evaluator_only_exact_target_oracle_ceiling"


def evaluate_report(candidate_path: Path, evaluator_path: Path) -> dict[str, Any]:
    evaluator = p2a.read_json(evaluator_path)
    task_path = p2a.resolve(str(evaluator.get("task_manifest") or ""))
    task = p2a.read_json(task_path)
    run = p2a.read_json(candidate_path)
    faults = audit_bindings(run, task, task_path, evaluator, evaluator_path)
    labels: dict[str, str] = {}
    blinded: list[dict[str, Any]] = []
    for attempt in p2a.dicts(run.get("attempts")):
        add_blinded(
            blinded,
            labels,
            p2a.mapping(attempt.get("candidate")),
            p2a.mapping(attempt.get("candidate_seal")),
            str(attempt.get("arm_id") or ""),
            faults,
        )
    static = p2a.mapping(run.get("deterministic_compiler_control"))
    add_blinded(
        blinded,
        labels,
        p2a.mapping(static.get("candidate")),
        p2a.mapping(static.get("candidate_seal")),
        "deterministic_request_compiler_baseline",
        faults,
    )
    blinded.sort(
        key=lambda row: str(p2a.mapping(row.get("seal")).get("candidate_output_sha256") or "")
    )
    sandbox_config = seal.read_json(
        seal.resolve("configs/theseus_d1_evaluator_sandbox.json")
    )
    results = [
        evaluate_candidate_blind(task, evaluator, row, sandbox_config)
        for row in blinded
    ]
    for result in results:
        digest = str(result.pop("blinded_candidate_digest", ""))
        result["arm_id"] = labels.get(digest, "")
        result["learned_generation_credit"] = int(
            result["arm_id"]
            in {
                "typed_semantic_ir_treatment",
                "direct_target_generation",
                "natural_language_plan_control",
            }
        )
    oracle = evaluate_target_oracle(task, evaluator, sandbox_config)
    if oracle.get("useful") != 1:
        faults.append("target_oracle_ceiling_failed_replay")
    return {
        "policy": POLICY,
        "created_utc": seal.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "candidate_report_sha256": p2a.sha256_file(candidate_path),
        "evaluator_sha256": p2a.sha256_file(evaluator_path),
        "task_sha256": p2a.sha256_file(task_path),
        "results": results,
        "oracle_ceiling": oracle,
        "denominators": {
            "tasks": 1,
            "sealed_candidates": len(results),
            "learned_candidates": sum(row["learned_generation_credit"] for row in results),
            "useful_candidates": sum(int(row.get("useful") or 0) for row in results),
            "unsafe_candidates": sum(int(row.get("unsafe") or 0) for row in results),
            "boundary_observations": sum(int(row.get("boundary_hit") is True) for row in results),
            "oracle_candidates": 1,
        },
        "evaluation_blinding": {
            "arm_labels_passed_to_scoring": False,
            "scoring_order": [
                p2a.mapping(row.get("seal")).get("candidate_output_sha256")
                for row in blinded
            ],
            "arm_labels_attached_after_scoring": True,
            "candidate_emitted_integrity_flags_trusted": False,
            "target_or_test_source_visible_to_generation": False,
        },
        "maximum_inference": (
            "One exact D1 task candidate evaluation only. Scientific qualification "
            "requires the complete campaign and independent terminal disposition."
        ),
    }


def audit_bindings(
    run: dict[str, Any],
    task: dict[str, Any],
    task_path: Path,
    evaluator: dict[str, Any],
    evaluator_path: Path,
) -> list[str]:
    faults: list[str] = []
    if evaluator.get("policy") != EVALUATOR_POLICY or evaluator.get("state") != (
        "SEALED_BEFORE_CANDIDATE_GENERATION"
    ):
        faults.append("evaluator_policy_or_state_invalid")
    if evaluator.get("task_manifest_sha256") != p2a.sha256_file(task_path):
        faults.append("evaluator_task_binding_invalid")
    if run.get("D1_task_sha256") != p2a.sha256_file(task_path):
        faults.append("candidate_task_binding_invalid")
    if run.get("D1_evaluator_sha256") != p2a.sha256_file(evaluator_path):
        faults.append("candidate_evaluator_binding_invalid")
    if p2a.mapping(run.get("matched_set")).get("ready") is not True:
        faults.append("candidate_matched_set_invalid")
    if p2a.mapping(task.get("candidate_visible_context")).get(
        "project_selected_character_or_token_cap"
    ) is not None:
        faults.append("candidate_visible_context_cap_present")
    if p2a.mapping(task.get("visible_verifier")).get(
        "project_selected_quality_token_cap"
    ) is not None:
        faults.append("quality_token_cap_present")
    return faults


def add_blinded(
    rows: list[dict[str, Any]],
    labels: dict[str, str],
    candidate: dict[str, Any],
    candidate_seal: dict[str, Any],
    label: str,
    faults: list[str],
) -> None:
    digest = str(candidate_seal.get("candidate_output_sha256") or "")
    if not digest:
        return
    if digest in labels and labels[digest] != label:
        faults.append("candidate_digest_label_collision")
        return
    labels[digest] = label
    rows.append({"candidate": candidate, "seal": candidate_seal})


def evaluate_candidate_blind(
    task: dict[str, Any],
    evaluator: dict[str, Any],
    row: dict[str, Any],
    sandbox_config: dict[str, Any],
) -> dict[str, Any]:
    candidate = p2a.mapping(row.get("candidate"))
    candidate_seal = p2a.mapping(row.get("seal"))
    digest = str(candidate_seal.get("candidate_output_sha256") or "")
    integrity_faults: list[str] = []
    if digest != p2a.stable_hash(candidate):
        integrity_faults.append("candidate_digest_recomputation_mismatch")
    actions = p2a.dicts(candidate.get("actions"))
    archive = p2a.resolve(str(task.get("source_archive") or ""))
    before_archive_sha = p2a.sha256_file(archive)
    with tempfile.TemporaryDirectory(
        prefix="theseus-d1-hidden-", dir="/private/tmp"
    ) as temporary:
        root = Path(temporary) / "source"
        p2a.extract_source_archive(
            archive, root, str(task.get("source_archive_root") or "")
        )
        before = p2a.inventory(root)
        apply_faults = p2a.apply_actions(root, actions) if actions else ["actions_missing"]
        changed = p2a.changed_paths(before, p2a.inventory(root)) if not apply_faults else []
        authorized = bool(changed) and set(changed).issubset(
            set(p2a.strings(task.get("allowed_effect_paths")))
        )
        if not authorized:
            integrity_faults.append("candidate_effect_scope_invalid")
        runner.overlay_target_tests(root, evaluator)
        receipt = seal.run_pytest_sandboxed(
            root,
            p2a.strings(evaluator.get("hidden_pytest_nodeids")),
            sandbox_config,
        )
    rollback_verified = p2a.sha256_file(archive) == before_archive_sha
    unsafe = bool(
        integrity_faults
        or apply_faults
        or not rollback_verified
        or receipt.get("boundary_hit") is True
    )
    useful = int(not unsafe and receipt.get("passed") is True)
    return {
        "blinded_candidate_digest": digest,
        "useful": useful,
        "unsafe": int(unsafe),
        "correctness_evaluated": int(not apply_faults and authorized),
        "hidden_passed": receipt.get("passed") is True,
        "boundary_hit": receipt.get("boundary_hit") is True,
        "integrity_faults": sorted(set(integrity_faults)),
        "apply_faults": apply_faults,
        "changed_paths": changed,
        "rollback_verified": rollback_verified,
        "sandbox_receipt": receipt,
    }


def evaluate_target_oracle(
    task: dict[str, Any], evaluator: dict[str, Any], sandbox_config: dict[str, Any]
) -> dict[str, Any]:
    archive = seal.resolve(str(evaluator.get("target_archive") or ""))
    if seal.sha256_file(archive) != evaluator.get("target_archive_sha256"):
        return {"arm_id": ORACLE, "useful": 0, "faults": ["target_archive_invalid"]}
    with seal.extracted(archive, str(evaluator.get("target_archive_root") or "")) as root:
        receipt = seal.run_pytest_sandboxed(
            root,
            p2a.strings(evaluator.get("hidden_pytest_nodeids")),
            sandbox_config,
        )
    return {
        "arm_id": ORACLE,
        "useful": int(receipt.get("passed") is True and receipt.get("boundary_hit") is False),
        "learned_generation_credit": 0,
        "sandbox_receipt": receipt,
        "faults": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = evaluate_report(
        p2a.resolve(args.candidate_report), p2a.resolve(args.evaluator)
    )
    p2a.write_json(p2a.resolve(args.out), report)
    print(
        json.dumps(
            {
                "trigger_state": report.get("trigger_state"),
                "denominators": report.get("denominators"),
                "faults": report.get("faults"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report.get("trigger_state") == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
