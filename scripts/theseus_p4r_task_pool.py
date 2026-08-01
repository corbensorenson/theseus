#!/usr/bin/env python3
"""Materialize and seal the repaired ten-task P4R pool."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4_task_pool as p4_pool  # noqa: E402


SOURCE_REGISTRY = ROOT / "configs" / "theseus_p4r_task_sources.json"
INSTRUMENT = ROOT / "configs" / "theseus_p4_cognitive_compilation_repaired_instrument_r1.json"
INSTRUMENT_AUDIT = ROOT / "reports" / "theseus_p4_cognitive_compilation_repaired_instrument_r1_audit.json"
PREDECESSOR_POOL = ROOT / "configs" / "theseus_p4_task_pool.json"
FIXTURES = ROOT / "tests" / "fixtures" / "theseus_p4r_online"
HIDDEN_TEST = FIXTURES / "theseus_p4r_hidden_test.py"
VISIBLE_TEST = FIXTURES / "theseus_p4r_visible_test.py"
INSTRUMENT_COMMIT = "0a8f0bec2cd8883fa6b1176e5f9979554c45539f"
SOURCE_SELECTION_COMMIT = "694ddbeb"
EXPECTED_INSTRUMENT_SHA256 = "d1997c4f4555c62d9106dc17a2c59580b6cb38110b7819f8a5a5b4de1849689c"
EXPECTED_INSTRUMENT_AUDIT_SHA256 = "dd5b7b9b24d6c92dde2b49abe8c5f09f4c6f8a92bde99b27522294f6f7ee07b1"
SELECTION_INSTRUMENT_COMMIT = "744d29e8"
SELECTION_INSTRUMENT_SHA256 = "64afd82f5dd44b773b9b3f5fb739fee6b8e9ac6a757aea338aaecb620c0b1794"
SELECTION_INSTRUMENT_AUDIT_SHA256 = "faffc010d292abd07e5e5e1e0097f506fed44abbf09a20fd617a6234c0a83e92"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialize-only", action="store_true")
    args = parser.parse_args()
    report = materialize_pool(run_audits=not args.materialize_only)
    print(json.dumps({
        "state": report["state"],
        "task_count": report["task_count"],
        "green_evaluator_audits": report["green_evaluator_audits"],
        "faults": report["faults"],
    }, indent=2, sort_keys=True))
    return 0 if report["state"] == "SEALED_BEFORE_CANDIDATE_GENERATION" else 2


def materialize_pool(*, run_audits: bool) -> dict[str, Any]:
    registry = p2a.read_json(SOURCE_REGISTRY)
    faults = audit_registry(registry)
    entries: list[dict[str, Any]] = []
    for source in p2a.dicts(registry.get("reused_unopened_tasks")):
        entry, entry_faults = adapt_unopened_task(source)
        entries.append(entry)
        faults.extend(entry_faults)
    original_globals = (
        p4_pool.FIXTURE_DIR, p4_pool.HIDDEN_TEST, p4_pool.VISIBLE_TEST
    )
    p4_pool.FIXTURE_DIR, p4_pool.HIDDEN_TEST, p4_pool.VISIBLE_TEST = (
        FIXTURES, HIDDEN_TEST, VISIBLE_TEST
    )
    try:
        for source in p2a.dicts(registry.get("new_tasks")):
            entry, entry_faults = p4_pool.materialize_task(source, registry)
            entry, repair_faults = rebind_new_task(entry, source)
            entries.append(entry)
            faults.extend(entry_faults)
            faults.extend(repair_faults)
    finally:
        p4_pool.FIXTURE_DIR, p4_pool.HIDDEN_TEST, p4_pool.VISIBLE_TEST = original_globals
    if run_audits and not faults:
        for entry in entries:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "theseus_p4_cognitive_compilation_evaluator.py"),
                    "--evaluator", entry["evaluator"],
                    "--audit-only",
                    "--out", entry["evaluator_audit"],
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=240,
            )
            audit = p2a.read_json(ROOT / entry["evaluator_audit"])
            if result.returncode != 0:
                faults.append(f"evaluator_audit_red:{entry['stem']}:{audit.get('faults')}")
            entry.update({
                "evaluator_audit_sha256": p2a.sha256_file(ROOT / entry["evaluator_audit"]),
                "evaluator_audit_trigger_state": audit.get("trigger_state"),
                "baseline_parent_failed": p2a.mapping(audit.get("baseline_verification")).get("hidden_passed") is False,
                "upstream_target_passed": p2a.mapping(audit.get("target_verification")).get("hidden_passed") is True,
                "compiler_oracle_passed": p2a.mapping(audit.get("compiler_oracle_verification")).get("hidden_passed") is True,
                "four_corruptions_rejected": all(p2a.mapping(audit.get("corruption_intervention_rejections")).values()),
            })
    else:
        for entry in entries:
            entry.update({
                "evaluator_audit_sha256": "",
                "evaluator_audit_trigger_state": "NOT_RUN",
                "baseline_parent_failed": False,
                "upstream_target_passed": False,
                "compiler_oracle_passed": False,
                "four_corruptions_rejected": False,
            })
    green = sum(row.get("evaluator_audit_trigger_state") == "GREEN" for row in entries)
    if run_audits and green != 10:
        faults.append("not_all_evaluator_audits_green")
    state = "SEALED_BEFORE_CANDIDATE_GENERATION" if run_audits and not faults and green == 10 else "INVALID_NOT_SEALED"
    pool = {
        "policy": "project_theseus_p4r_cognitive_compilation_task_pool_v1",
        "state": state,
        "partition": "p4r_cognitive_compilation_development",
        "sealed_utc": registry.get("sealed_utc"),
        "candidate_generation_opened": False,
        "selection_rule": registry.get("selection_rule"),
        "source_registry": p2a.rel(SOURCE_REGISTRY),
        "source_registry_sha256": p2a.sha256_file(SOURCE_REGISTRY),
        "source_selection_commit": SOURCE_SELECTION_COMMIT,
        "instrument": p2a.rel(INSTRUMENT),
        "instrument_freeze_commit": INSTRUMENT_COMMIT,
        "instrument_sha256": p2a.sha256_file(INSTRUMENT),
        "instrument_audit": p2a.rel(INSTRUMENT_AUDIT),
        "instrument_audit_sha256": p2a.sha256_file(INSTRUMENT_AUDIT),
        "post_selection_mechanics_repair": {
            "repair_commit": INSTRUMENT_COMMIT,
            "candidate_or_control_calls_before_repair": 0,
            "task_membership_changed": False,
            "evaluator_or_decision_rule_changed": False,
            "repair": "Bind the inherited runner transport field to the pinned model context window; preserve parser-complete/model-EOS normal completion.",
        },
        "task_count": len(entries),
        "green_evaluator_audits": green,
        "distinct_repositories": len({row.get("repository") for row in entries}),
        "reused_unopened_task_count": 7,
        "new_task_count": 3,
        "tasks": entries,
        "faults": sorted(set(faults)),
        "generation_budget": {
            "project_selected_quality_token_cap": None,
            "normal_completion": ["parser_complete", "model_eos"],
            "physical_boundary": "model_declared_context_window_minus_exact_prompt_tokens",
            "boundary_hit_counts_as_failure": False,
        },
        "source_disjoint_from": {
            "P2_P3_P4": sorted(prior_repositories()),
            "D1": "reserved_fresh_source_disjoint_surface_not_acquired",
            "D2": "independent_neural_surface_not_acquired",
            "training": "all_P4R_tasks_permanently_excluded",
        },
        "information_flow": {
            "natural_request_obligations_source_and_visible_feedback_candidate_visible": True,
            "upstream_target_hidden_test_and_oracle_candidate_visible": False,
            "task_selection_conditioned_on_candidate_or_control_output": False,
            "route_labels_passed_to_scoring": False,
        },
        "counters": {
            "local_model_calls": 0,
            "hosted_model_calls": 0,
            "teacher_calls": 0,
            "deterministic_request_compiler_calls": 0,
            "compiler_oracle_evaluator_audits": green,
            "public_calibration_cases_consumed": 0,
            "D1_cases_consumed": 0,
            "D2_cases_consumed": 0,
            "training_rows_written": 0,
        },
        "maximum_inference": "A GREEN seal establishes source/evaluator/oracle reachability and corruption sensitivity only. It is not candidate, subsystem, D1, D2, serving, training, or ASI Stack support evidence.",
    }
    p2a.write_json(ROOT / "configs" / "theseus_p4r_task_pool.json", pool)
    return pool


def adapt_unopened_task(source: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    faults: list[str] = []
    prior_task_path = ROOT / str(source.get("task") or "")
    prior_evaluator_path = ROOT / str(source.get("evaluator") or "")
    prior_task = p2a.read_json(prior_task_path)
    prior_evaluator = p2a.read_json(prior_evaluator_path)
    index = int(source["campaign_index"])
    suffix = str(source["stem"]).removeprefix("p4r_")
    task_path = ROOT / "configs" / f"theseus_p4r_task_{suffix}.json"
    evaluator_path = ROOT / "configs" / f"theseus_p4r_evaluator_{suffix}.json"
    audit_path = ROOT / "reports" / f"theseus_p4r_{suffix}_evaluator_audit.json"
    task = copy.deepcopy(prior_task)
    task["campaign_index"] = index
    task["opaque_task_id"] = f"p4r-cognitive-compilation-{index:02d}"
    task["partition"] = "p4r_cognitive_compilation_development"
    screen = p2a.mapping(task.get("contamination_screen"))
    screen.update({
        "unopened_predecessor_campaign_index": source.get("predecessor_campaign_index"),
        "never_seen_by_P4_v1_candidate_or_control": True,
        "rebound_after_P4R_instrument_freeze": True,
    })
    task["contamination_screen"] = screen
    p2a.write_json(task_path, task)
    evaluator = copy.deepcopy(prior_evaluator)
    evaluator["task_manifest"] = p2a.rel(task_path)
    evaluator["task_manifest_sha256"] = p2a.sha256_file(task_path)
    p2a.write_json(evaluator_path, evaluator)
    entry = {
        "campaign_index": index,
        "stem": source["stem"],
        "repository": source["repository"],
        "provenance": "unopened_P4_v1_task_rebound_without_candidate_visibility",
        "predecessor_campaign_index": source["predecessor_campaign_index"],
        "task": p2a.rel(task_path),
        "task_sha256": p2a.sha256_file(task_path),
        "evaluator": p2a.rel(evaluator_path),
        "evaluator_sha256": p2a.sha256_file(evaluator_path),
        "oracle_ir": prior_evaluator.get("oracle_ir_file"),
        "oracle_ir_sha256": prior_evaluator.get("oracle_ir_sha256"),
        "evaluator_audit": p2a.rel(audit_path),
    }
    if int(prior_task.get("campaign_index") or 0) != int(source.get("predecessor_campaign_index") or 0):
        faults.append(f"predecessor_index_mismatch:{source['stem']}")
    return entry, faults


def rebind_new_task(entry: dict[str, Any], source: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    faults: list[str] = []
    task_path = ROOT / str(entry.get("task") or "")
    evaluator_path = ROOT / str(entry.get("evaluator") or "")
    task = p2a.read_json(task_path)
    task["opaque_task_id"] = f"p4r-cognitive-compilation-{int(source['campaign_index']):02d}"
    task["partition"] = "p4r_cognitive_compilation_development"
    screen = p2a.mapping(task.get("contamination_screen"))
    screen["task_selected_after_p4r_instrument_freeze"] = True
    screen["task_selected_before_any_P4R_candidate_or_control"] = True
    task["contamination_screen"] = screen
    p2a.write_json(task_path, task)
    evaluator = p2a.read_json(evaluator_path)
    evaluator["task_manifest_sha256"] = p2a.sha256_file(task_path)
    evaluator["baseline_failure_markers"] = [f"P4R_FAIL_{source['case']}"]
    p2a.write_json(evaluator_path, evaluator)
    entry.update({
        "provenance": "new_source_disjoint_task_selected_after_P4R_freeze",
        "task_sha256": p2a.sha256_file(task_path),
        "evaluator_sha256": p2a.sha256_file(evaluator_path),
        "evaluator_audit": f"reports/theseus_{source['stem']}_evaluator_audit.json",
    })
    return entry, faults


def audit_registry(registry: dict[str, Any]) -> list[str]:
    faults: list[str] = []
    if registry.get("policy") != "project_theseus_p4r_online_source_selection_v1":
        faults.append("source_registry_policy_invalid")
    if registry.get("state") != "FIXED_BEFORE_TASK_MATERIALIZATION_OR_CANDIDATE_GENERATION":
        faults.append("source_registry_not_fixed")
    reused = p2a.dicts(registry.get("reused_unopened_tasks"))
    new = p2a.dicts(registry.get("new_tasks"))
    rows = [*reused, *new]
    if len(reused) != 7 or len(new) != 3 or len(rows) != 10 or registry.get("task_count") != 10:
        faults.append("source_registry_task_count_invalid")
    if [row.get("campaign_index") for row in rows] != list(range(1, 11)):
        faults.append("campaign_indexes_invalid")
    repositories = [str(row.get("repository") or "").lower() for row in rows]
    if len(set(repositories)) != 10:
        faults.append("repositories_not_distinct")
    if set(str(row.get("repository") or "").lower() for row in new).intersection(prior_repositories()):
        faults.append("new_task_prior_repository_overlap")
    boundaries = p2a.mapping(registry.get("boundaries"))
    if boundaries.get("candidate_generation_opened") is not False:
        faults.append("candidate_generation_already_opened")
    for key in (
        "local_model_calls", "hosted_model_calls", "deterministic_request_compiler_calls",
        "new_task_parent_target_oracle_calls_before_selection",
    ):
        if int(boundaries.get(key) or 0) != 0:
            faults.append(f"selection_conditioning_counter_nonzero:{key}")
    if boundaries.get("user_task_or_label_dependency") is not False:
        faults.append("user_dependency_not_excluded")
    if registry.get("instrument_freeze_commit") != SELECTION_INSTRUMENT_COMMIT:
        faults.append("selection_instrument_commit_mismatch")
    if registry.get("instrument_sha256") != SELECTION_INSTRUMENT_SHA256:
        faults.append("selection_instrument_digest_mismatch")
    if registry.get("instrument_audit_sha256") != SELECTION_INSTRUMENT_AUDIT_SHA256:
        faults.append("selection_instrument_audit_digest_mismatch")
    if p2a.sha256_file(INSTRUMENT) != EXPECTED_INSTRUMENT_SHA256:
        faults.append("active_instrument_digest_mismatch")
    if p2a.sha256_file(INSTRUMENT_AUDIT) != EXPECTED_INSTRUMENT_AUDIT_SHA256:
        faults.append("active_instrument_audit_digest_mismatch")
    return faults


def prior_repositories() -> set[str]:
    values = {value.lower() for value in p4_pool.P2_REPOSITORIES | p4_pool.p3_repositories()}
    predecessor = p2a.read_json(PREDECESSOR_POOL)
    values.update(str(row.get("repository") or "").lower() for row in p2a.dicts(predecessor.get("tasks")))
    return values


if __name__ == "__main__":
    raise SystemExit(main())
