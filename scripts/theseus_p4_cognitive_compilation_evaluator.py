#!/usr/bin/env python3
"""Independent route-blind evaluator for P4 cognitive-compilation candidates."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import theseus_assistant_p2a as p2a
import theseus_assistant_p2a_evaluator as p2a_evaluator
import theseus_p4_cognitive_compilation as p4


ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_p4_cognitive_compilation_route_blind_evaluation_v1"
EVALUATOR_POLICY = "project_theseus_p4_cognitive_compilation_evaluator_v1"
STATIC = p4.STATIC
ORACLE = "deterministic_compiler_oracle_ceiling"


class EvaluationFault(ValueError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--candidate-report", default="")
    parser.add_argument("--out", default="reports/theseus_p4_cognitive_compilation_evaluation.json")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    evaluator_path = p2a.resolve(args.evaluator)
    report = (
        audit_evaluator(evaluator_path)
        if args.audit_only
        else evaluate_report(p2a.resolve(args.candidate_report), evaluator_path)
    )
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps({
        "policy": report.get("policy"), "trigger_state": report.get("trigger_state"),
        "faults": report.get("faults"), "denominators": report.get("denominators"),
    }, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") == "GREEN" else 2


def audit_evaluator(path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    value = p2a.read_json(path)
    faults: list[str] = []
    if value.get("policy") != EVALUATOR_POLICY:
        faults.append("evaluator_policy_invalid")
    if value.get("state") != "SEALED_BEFORE_CANDIDATE_GENERATION":
        faults.append("evaluator_not_sealed")
    task_path = p2a.resolve(str(value.get("task_manifest") or ""))
    if p2a.sha256_file(task_path) != str(value.get("task_manifest_sha256") or ""):
        faults.append("task_manifest_digest_mismatch")
    task_audit = p4.audit_task(task_path)
    if task_audit.get("trigger_state") != "GREEN":
        faults.append("task_manifest_audit_red")
    for index, row in enumerate(p2a.dicts(value.get("hidden_test_files"))):
        source = p2a.resolve(str(row.get("source") or ""))
        if p2a.sha256_file(source) != str(row.get("sha256") or ""):
            faults.append(f"hidden_test_{index}_digest_mismatch")
        if p2a.unsafe_relative_path(str(row.get("destination") or "")):
            faults.append(f"hidden_test_{index}_destination_unsafe")
    target_archive = p2a.resolve(str(value.get("target_archive") or ""))
    if p2a.sha256_file(target_archive) != str(value.get("target_archive_sha256") or ""):
        faults.append("target_archive_digest_mismatch")
    oracle_path = p2a.resolve(str(value.get("oracle_ir_file") or ""))
    if p2a.sha256_file(oracle_path) != str(value.get("oracle_ir_sha256") or ""):
        faults.append("oracle_ir_digest_mismatch")
    task = p2a.read_json(task_path)
    baseline: dict[str, Any] = {}
    target: dict[str, Any] = {}
    oracle: dict[str, Any] = {}
    corruptions: dict[str, bool] = {}
    if not faults:
        try:
            baseline = verify_archive(task, value, p2a.resolve(str(task.get("source_archive"))), str(task.get("source_archive_root")))
            target = verify_archive(task, value, target_archive, str(value.get("target_archive_root") or ""))
            oracle, corruptions = verify_oracle(task, value, oracle_path)
        except (OSError, EvaluationFault, p2a.InstrumentFault, p4.P4Fault) as exc:
            faults.append(f"evaluator_audit_fault:{type(exc).__name__}")
    if value.get("baseline_must_fail") is not True or baseline.get("hidden_passed") is not False:
        faults.append("baseline_does_not_fail_as_required")
    baseline_text = str(baseline.get("stdout_tail") or "") + str(baseline.get("stderr_tail") or "")
    if any(marker not in baseline_text for marker in p2a.strings(value.get("baseline_failure_markers"))):
        faults.append("baseline_failure_markers_missing")
    if value.get("target_must_pass") is not True or target.get("hidden_passed") is not True:
        faults.append("target_does_not_pass_as_required")
    if oracle.get("visible_passed") is not True or oracle.get("hidden_passed") is not True:
        faults.append("compiler_oracle_does_not_pass")
    expected_corruptions = {
        "source_identity", "obligation_coverage", "target_identity", "unresolved_loss"
    }
    if set(corruptions) != expected_corruptions or not all(corruptions.values()):
        faults.append("oracle_corruption_interventions_inadequate")
    return {
        "policy": "project_theseus_p4_cognitive_compilation_evaluator_audit_v1",
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "evaluator_sha256": p2a.sha256_file(path),
        "task_manifest_sha256": p2a.sha256_file(task_path),
        "task_audit": task_audit,
        "baseline_verification": baseline,
        "target_verification": target,
        "compiler_oracle_verification": oracle,
        "corruption_intervention_rejections": corruptions,
        "route_labels_opened": 0,
        "target_and_oracle_artifacts_opened_by_evaluator_only": 1 if target and oracle else 0,
        "runtime_ms": round((time.perf_counter() - started) * 1000, 3),
        "counters": p2a.zero_counters(),
    }


def evaluate_report(candidate_path: Path, evaluator_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    alignment = audit_evaluator(evaluator_path)
    evaluator = p2a.read_json(evaluator_path)
    task_path = p2a.resolve(str(evaluator.get("task_manifest") or ""))
    task = p2a.read_json(task_path)
    run = p2a.read_json(candidate_path)
    faults: list[str] = []
    if alignment.get("trigger_state") != "GREEN":
        faults.append("evaluator_alignment_red")
    if run.get("task_sha256") != p2a.sha256_file(task_path):
        faults.append("candidate_task_binding_mismatch")
    if p2a.mapping(run.get("matched_set")).get("ready") is not True:
        faults.append("candidate_matched_set_invalid")
    blinded: list[dict[str, Any]] = []
    labels: dict[str, str] = {}
    for attempt in p2a.dicts(run.get("attempts")):
        candidate = p2a.mapping(attempt.get("candidate"))
        seal = p2a.mapping(attempt.get("candidate_seal"))
        digest = str(seal.get("candidate_output_sha256") or "")
        if not digest:
            continue
        label = str(attempt.get("arm_id") or "")
        if digest in labels and labels[digest] != label:
            faults.append("candidate_digest_label_collision")
            continue
        labels[digest] = label
        blinded.append({"candidate": candidate, "seal": seal})
    static = p2a.mapping(run.get("deterministic_compiler_control"))
    static_seal = p2a.mapping(static.get("candidate_seal"))
    static_digest = str(static_seal.get("candidate_output_sha256") or "")
    if static_digest:
        if static_digest in labels and labels[static_digest] != STATIC:
            faults.append("candidate_digest_label_collision")
        else:
            labels[static_digest] = STATIC
            blinded.append({
                "candidate": p2a.mapping(static.get("candidate")),
                "seal": static_seal,
            })
    try:
        oracle_row = oracle_candidate(task, evaluator)
        oracle_digest = str(p2a.mapping(oracle_row.get("seal")).get("candidate_output_sha256") or "")
        if oracle_digest in labels and labels[oracle_digest] != ORACLE:
            faults.append("candidate_digest_label_collision")
        else:
            labels[oracle_digest] = ORACLE
            blinded.append(oracle_row)
    except (OSError, EvaluationFault, p2a.InstrumentFault, p4.P4Fault) as exc:
        faults.append(f"compiler_oracle_candidate_fault:{type(exc).__name__}")
    blinded.sort(key=lambda row: str(p2a.mapping(row.get("seal")).get("candidate_output_sha256") or ""))
    results: list[dict[str, Any]] = []
    for row in blinded:
        result = evaluate_candidate_blind(task, evaluator, row)
        digest = str(p2a.mapping(row.get("seal")).get("candidate_output_sha256") or "")
        result["arm_id"] = labels.get(digest, "")
        result["learned_generation_credit"] = int(labels.get(digest, "") in p4.ARMS)
        results.append(result)
    compiler_results = [row for row in results if row.get("arm_id") == ORACLE]
    if len(compiler_results) != 1 or compiler_results[0].get("useful") != 1:
        faults.append("compiler_oracle_scoring_invalid")
    evaluated = sum(int(row.get("correctness_evaluated") or 0) for row in results)
    useful = sum(int(row.get("useful") or 0) for row in results)
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "scope": "One P4 cognitive-compilation development task; the prospective no-model compiler is a real control and the target-derived oracle is only a mechanics ceiling; both receive zero learned credit",
        "candidate_report_sha256": p2a.sha256_file(candidate_path),
        "evaluator_sha256": p2a.sha256_file(evaluator_path),
        "alignment_audit": alignment,
        "results": results,
        "denominators": {
            "tasks": 1,
            "learned_arms": 3,
            "sealed_learned_candidates": sum(
                int(row.get("arm_id") in p4.ARMS) for row in results
            ),
            "deterministic_compiler_control_candidates": sum(
                int(row.get("arm_id") == STATIC) for row in results
            ),
            "deterministic_compiler_control_abstentions": int(
                static.get("abstained") == 1
            ),
            "compiler_oracle_candidates": 1,
            "correctness_evaluated_candidates": evaluated,
            "useful_candidates_including_oracle": useful,
            "learned_useful_candidates": sum(
                int(row.get("useful") or 0) for row in results
                if row.get("arm_id") in p4.ARMS
            ),
        },
        "evaluation_blinding": {
            "arm_labels_passed_to_scoring": False,
            "scoring_order": [p2a.mapping(row.get("seal")).get("candidate_output_sha256") for row in blinded],
            "arm_labels_attached_after_scoring": True,
            "candidate_authored_integrity_flags_trusted": False,
            "deterministic_control_target_or_oracle_visibility": False,
            "compiler_oracle_answer_visible_to_generation": False,
        },
        "maximum_inference": (
            "Only candidate correctness on this exact development task. The prospective static compiler "
            "is a no-model control; the target-derived oracle proves lowering reachability. Neither receives "
            "learned credit. No single task supports "
            "or falsifies cognitive compilation."
        ),
        "counters": p2a.zero_counters(),
        "runtime_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def verify_archive(
    task: dict[str, Any], evaluator: dict[str, Any], archive: Path, root_name: str
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="theseus-p4-archive-audit-") as tmp:
        root = Path(tmp) / "source"
        p2a.extract_source_archive(archive, root, root_name)
        p2a_evaluator.overlay_hidden_tests(evaluator, root)
        hidden = p2a_evaluator.run_hidden_verifier(evaluator, root)
        return {
            "hidden_passed": hidden.get("passed") is True,
            "returncode": hidden.get("returncode"),
            "stdout_tail": hidden.get("stdout_tail", ""),
            "stderr_tail": hidden.get("stderr_tail", ""),
            "runtime_ms": hidden.get("runtime_ms"),
        }


def verify_oracle(
    task: dict[str, Any], evaluator: dict[str, Any], oracle_path: Path
) -> tuple[dict[str, Any], dict[str, bool]]:
    with tempfile.TemporaryDirectory(prefix="theseus-p4-oracle-audit-") as tmp:
        root = Path(tmp) / "source"
        p2a.extract_source_archive(
            p2a.resolve(str(task.get("source_archive") or "")), root,
            str(task.get("source_archive_root") or ""),
        )
        text = oracle_path.read_text(encoding="utf-8")
        parsed = p4.parse_semantic_ir(text, task, root)
        if parsed["faults"]:
            raise EvaluationFault("oracle_ir_invalid")
        apply_faults = p2a.apply_actions(root, p2a.dicts(parsed.get("actions")))
        visible = p2a.run_visible_verifier(root, task) if not apply_faults else {}
        p2a_evaluator.overlay_hidden_tests(evaluator, root)
        hidden = p2a_evaluator.run_hidden_verifier(evaluator, root) if not apply_faults else {}
    expected_ids = [str(row.get("id") or "") for row in p2a.dicts(task.get("obligations"))]
    source_match = re.search(r"^SOURCE ([a-f0-9]{64})$", text, flags=re.MULTILINE)
    unit_match = p4.IR_UNIT_RE.search(text)
    if source_match is None or unit_match is None:
        raise EvaluationFault("oracle_identity_fields_missing")
    mutations = {
        "source_identity": text.replace(source_match.group(1), "0" * 64, 1),
        "obligation_coverage": text.replace(
            "OBLIGATIONS " + ",".join(expected_ids),
            "OBLIGATIONS " + ",".join(expected_ids[:-1]),
            1,
        ),
        "target_identity": text.replace(unit_match.group(6), "0" * 64, 1),
        "unresolved_loss": text.replace("LOSS NONE", f"LOSS {expected_ids[0]}", 1),
    }
    corruptions: dict[str, bool] = {}
    for name, mutation in mutations.items():
        with tempfile.TemporaryDirectory(prefix="theseus-p4-corruption-") as tmp:
            root = Path(tmp) / "source"
            p2a.extract_source_archive(
                p2a.resolve(str(task.get("source_archive") or "")), root,
                str(task.get("source_archive_root") or ""),
            )
            corruptions[name] = bool(p4.parse_semantic_ir(mutation, task, root)["faults"])
    return ({
        "parse_faults": parsed["faults"],
        "apply_faults": apply_faults,
        "visible_passed": visible.get("passed") is True,
        "hidden_passed": hidden.get("passed") is True,
        "hidden_returncode": hidden.get("returncode"),
        "hidden_stdout_tail": hidden.get("stdout_tail", ""),
        "hidden_stderr_tail": hidden.get("stderr_tail", ""),
    }, corruptions)


def oracle_candidate(task: dict[str, Any], evaluator: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="theseus-p4-oracle-candidate-") as tmp:
        root = Path(tmp) / "source"
        p2a.extract_source_archive(
            p2a.resolve(str(task.get("source_archive") or "")), root,
            str(task.get("source_archive_root") or ""),
        )
        baseline = p2a.inventory(root)
        oracle_path = p2a.resolve(str(evaluator.get("oracle_ir_file") or ""))
        parsed = p4.parse_semantic_ir(oracle_path.read_text(encoding="utf-8"), task, root)
        if parsed["faults"]:
            raise EvaluationFault("oracle_parse_fault")
        actions = p2a.dicts(parsed.get("actions"))
        apply_faults = p2a.apply_actions(root, actions)
        if apply_faults:
            raise EvaluationFault("oracle_apply_fault")
        changed = p2a.changed_paths(baseline, p2a.inventory(root))
        candidate = {
            "protocol": "theseus_semantic_ir_v1",
            "actions": actions,
            "changed_paths": changed,
            "final_inventory_sha256": p2a.stable_hash(p2a.inventory(root)),
            "visible_verifier": p2a.run_visible_verifier(root, task),
            "semantic_receipt": parsed.get("semantic_receipt", {}),
        }
        seal = {
            "candidate_output_sha256": p2a.stable_hash(candidate),
            "sealed_before_hidden_evaluation": True,
            "compiler_only_oracle": True,
        }
        return {"candidate": candidate, "seal": seal}


def evaluate_candidate_blind(
    task: dict[str, Any], evaluator: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    candidate = p2a.mapping(row.get("candidate"))
    seal = p2a.mapping(row.get("seal"))
    sealed = (
        seal.get("sealed_before_hidden_evaluation") is True
        and seal.get("candidate_output_sha256") == p2a.stable_hash(candidate)
    )
    if not sealed:
        return failed_candidate("candidate_seal_invalid")
    with tempfile.TemporaryDirectory(prefix="theseus-p4-score-") as tmp:
        root = Path(tmp) / "candidate"
        archive = p2a.resolve(str(task.get("source_archive") or ""))
        p2a.extract_source_archive(archive, root, str(task.get("source_archive_root") or ""))
        baseline = p2a.inventory(root)
        actions = p2a.dicts(candidate.get("actions"))
        apply_faults = p2a.apply_actions(root, actions)
        effects = p2a.changed_paths(baseline, p2a.inventory(root)) if not apply_faults else []
        authorized = bool(effects) and set(effects).issubset(set(p2a.strings(task.get("allowed_effect_paths"))))
        inventory_match = candidate.get("final_inventory_sha256") == p2a.stable_hash(p2a.inventory(root))
        verification: dict[str, Any] = {}
        if not apply_faults and authorized and inventory_match:
            p2a_evaluator.overlay_hidden_tests(evaluator, root)
            verification = p2a_evaluator.run_hidden_verifier(evaluator, root)
        shutil.rmtree(root)
        p2a.extract_source_archive(archive, root, str(task.get("source_archive_root") or ""))
        rollback = p2a.inventory(root) == baseline
    evaluated = bool(not apply_faults and authorized and inventory_match and verification)
    useful = evaluated and verification.get("passed") is True and rollback
    return {
        "candidate_output_sha256": seal.get("candidate_output_sha256"),
        "sealed": int(sealed), "actions_applied": int(not apply_faults),
        "apply_faults": apply_faults, "observed_effect_paths": effects,
        "allowed_effects": int(authorized),
        "candidate_inventory_recomputed": int(inventory_match),
        "correctness_evaluated": int(evaluated),
        "hidden_tests_passed": int(verification.get("passed") is True),
        "rollback_verified": int(rollback), "useful": int(useful),
        "unsafe": int(not authorized if effects else False),
        "verification": verification,
    }


def failed_candidate(reason: str) -> dict[str, Any]:
    return {
        "sealed": 0, "actions_applied": 0, "allowed_effects": 0,
        "candidate_inventory_recomputed": 0, "correctness_evaluated": 0,
        "hidden_tests_passed": 0, "rollback_verified": 0, "useful": 0,
        "unsafe": 0, "failure_reason": reason,
    }


if __name__ == "__main__":
    raise SystemExit(main())
