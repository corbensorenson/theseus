#!/usr/bin/env python3
"""Materialize and seal the mechanics-qualified P4-v2r2 task pool."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_assistant_p2a_evaluator as p2a_evaluator  # noqa: E402
import theseus_p4_cognitive_compilation as p4  # noqa: E402
import theseus_p4_task_pool as base_pool  # noqa: E402
import theseus_p4s_cognitive_compilation as p4s  # noqa: E402
import theseus_p4s_task_pool as p4s_pool  # noqa: E402
import theseus_p4v2r2_cognitive_compilation as p4v2r2  # noqa: E402
import theseus_semantic_ir_v2 as ir_v2  # noqa: E402
import theseus_semantic_ir_v2r2 as ir_v2r2  # noqa: E402


SOURCE_REGISTRY = ROOT / "configs" / "theseus_p4v2r2_task_sources.json"
SOURCE_FETCH = ROOT / "reports" / "theseus_p4v2r2_source_fetch.json"
ORACLE_CORRECTIONS = (
    ROOT / "configs" / "theseus_p4v2r2_oracle_materialization_corrections.json"
)
INSTRUMENT = ROOT / "configs" / "theseus_p4v2r2_cognitive_compilation_instrument.json"
INSTRUMENT_AUDIT = (
    ROOT / "reports" / "theseus_p4v2r2_cognitive_compilation_instrument_audit.json"
)
FIXTURES = ROOT / "tests" / "fixtures" / "theseus_p4v2r2_online"
HIDDEN_TEST = FIXTURES / "theseus_p4v2r2_hidden_test.py"
VISIBLE_TEST = FIXTURES / "theseus_p4v2r2_visible_test.py"
SOURCE_SELECTION_COMMIT = "8cebe4a65bb03965e9f62efa8249f2f9ddb8fc08"
SOURCE_ACQUISITION_COMMIT = "3ea8d770f4d59061f7b9ae128e8877917b8fd570"
EVALUATOR_SURFACE_COMMIT = "c9ab9f1346fa96ebd06ab70cc49fcf087fbd6daa"
EXPECTED_SOURCE_REGISTRY_SHA256 = (
    "7264e2a040092de68e98a8a91b97ec38ca9a04a442f80a3c0551e767b0c68915"
)
EXPECTED_SOURCE_FETCH_SHA256 = (
    "7c42c31e8fd1cc2408b3cf9a297882fadd8f409de1cfa5219d8e33b8c4e9f3eb"
)
EXPECTED_HIDDEN_TEST_SHA256 = (
    "dd1029e9a723100698f02c4e9c05db289f9af83adaea3eca7b1840850d6b8045"
)
EXPECTED_VISIBLE_TEST_SHA256 = (
    "44861c76857b6b3ff5f118ad04b30a1da6e6dd530162fbf8a50f4c93b6c1591a"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialize-only", action="store_true")
    args = parser.parse_args()
    report = materialize_pool(run_audits=not args.materialize_only)
    print(
        json.dumps(
            {
                "state": report["state"],
                "task_count": report["task_count"],
                "green_evaluator_audits": report["green_evaluator_audits"],
                "v2r2_oracle_replays_green": report["v2r2_oracle_replays_green"],
                "dependency_corruptions_rejected": report[
                    "dependency_corruptions_rejected"
                ],
                "faults": report["faults"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["state"] == "SEALED_BEFORE_CANDIDATE_GENERATION" else 2


def materialize_pool(*, run_audits: bool) -> dict[str, Any]:
    registry = p2a.read_json(SOURCE_REGISTRY)
    faults = audit_registry(registry)
    instrument_audit = p4v2r2.audit_instrument(INSTRUMENT)
    p2a.write_json(INSTRUMENT_AUDIT, instrument_audit)
    if instrument_audit.get("trigger_state") != "GREEN":
        faults.append("instrument_audit_red")

    entries: list[dict[str, Any]] = []
    for source in p2a.dicts(registry.get("tasks")):
        entry, entry_faults = materialize_task(source, registry)
        entries.append(entry)
        faults.extend(entry_faults)

    if run_audits and not faults:
        for entry in entries:
            result = subprocess.run(
                [
                    sys.executable,
                    str(
                        ROOT
                        / "scripts"
                        / "theseus_p4_cognitive_compilation_evaluator.py"
                    ),
                    "--evaluator",
                    entry["evaluator"],
                    "--audit-only",
                    "--out",
                    entry["evaluator_audit"],
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=240,
                check=False,
            )
            audit_path = ROOT / entry["evaluator_audit"]
            audit = p2a.read_json(audit_path)
            if result.returncode != 0:
                faults.append(
                    f"evaluator_audit_red:{entry['stem']}:{audit.get('faults')}"
                )
            dependency = audit_dependency_corruption(entry)
            if dependency.get("rejected") is not True:
                faults.append(f"dependency_corruption_not_rejected:{entry['stem']}")
            v2_replay = audit_v2r2_oracle(entry)
            if v2_replay.get("trigger_state") != "GREEN":
                faults.append(f"v2r2_oracle_replay_red:{entry['stem']}")
            entry.update(
                {
                    "evaluator_audit_sha256": p2a.sha256_file(audit_path),
                    "evaluator_audit_trigger_state": audit.get("trigger_state"),
                    "baseline_parent_failed": p2a.mapping(
                        audit.get("baseline_verification")
                    ).get("hidden_passed")
                    is False,
                    "upstream_target_passed": p2a.mapping(
                        audit.get("target_verification")
                    ).get("hidden_passed")
                    is True,
                    "compiler_oracle_v1_passed": p2a.mapping(
                        audit.get("compiler_oracle_verification")
                    ).get("hidden_passed")
                    is True,
                    "four_base_corruptions_rejected": all(
                        p2a.mapping(
                            audit.get("corruption_intervention_rejections")
                        ).values()
                    ),
                    "dependency_corruption": dependency,
                    "v2r2_oracle_replay": v2_replay,
                }
            )
    else:
        for entry in entries:
            entry.update(
                {
                    "evaluator_audit_sha256": "",
                    "evaluator_audit_trigger_state": "NOT_RUN",
                    "baseline_parent_failed": False,
                    "upstream_target_passed": False,
                    "compiler_oracle_v1_passed": False,
                    "four_base_corruptions_rejected": False,
                    "dependency_corruption": {
                        "rejected": False,
                        "faults": ["not_run"],
                    },
                    "v2r2_oracle_replay": {
                        "trigger_state": "NOT_RUN",
                        "faults": ["not_run"],
                    },
                }
            )

    green = sum(
        row.get("evaluator_audit_trigger_state") == "GREEN" for row in entries
    )
    v2_green = sum(
        p2a.mapping(row.get("v2r2_oracle_replay")).get("trigger_state") == "GREEN"
        for row in entries
    )
    dependency_green = sum(
        p2a.mapping(row.get("dependency_corruption")).get("rejected") is True
        for row in entries
    )
    if run_audits and green != 10:
        faults.append("not_all_evaluator_audits_green")
    if run_audits and v2_green != 10:
        faults.append("not_all_v2r2_oracle_replays_green")
    if run_audits and dependency_green != 10:
        faults.append("not_all_dependency_corruptions_rejected")
    state = (
        "SEALED_BEFORE_CANDIDATE_GENERATION"
        if run_audits
        and not faults
        and green == v2_green == dependency_green == 10
        else "INVALID_NOT_SEALED"
    )
    pool = {
        "policy": "project_theseus_p4v2r2_cognitive_compilation_task_pool_v1",
        "state": state,
        "partition": "p4v2r2_cognitive_compilation_decision_development",
        "sealed_utc": p2a.now(),
        "candidate_generation_opened": False,
        "selection_rule": registry.get("selection_rule"),
        "source_registry": p2a.rel(SOURCE_REGISTRY),
        "source_registry_sha256": p2a.sha256_file(SOURCE_REGISTRY),
        "source_selection_commit": SOURCE_SELECTION_COMMIT,
        "source_fetch_report": p2a.rel(SOURCE_FETCH),
        "source_fetch_report_sha256": p2a.sha256_file(SOURCE_FETCH),
        "source_acquisition_commit": SOURCE_ACQUISITION_COMMIT,
        "evaluator_surface_commit": EVALUATOR_SURFACE_COMMIT,
        "instrument": p2a.rel(INSTRUMENT),
        "instrument_freeze_commit": registry.get("instrument_freeze_commit"),
        "instrument_sha256": p2a.sha256_file(INSTRUMENT),
        "instrument_audit": p2a.rel(INSTRUMENT_AUDIT),
        "instrument_audit_sha256": p2a.sha256_file(INSTRUMENT_AUDIT),
        "task_count": len(entries),
        "green_evaluator_audits": green,
        "v2r2_oracle_replays_green": v2_green,
        "dependency_corruptions_rejected": dependency_green,
        "distinct_repositories": len(
            {str(row.get("repository") or "").lower() for row in entries}
        ),
        "tasks": entries,
        "faults": sorted(set(faults)),
        "generation_budget": {
            "project_selected_quality_token_cap": None,
            "normal_completion": ["parser_complete", "model_eos"],
            "physical_boundary": (
                "model_declared_context_window_minus_exact_prompt_tokens"
            ),
            "boundary_hit_invalidates_observation": True,
            "boundary_hit_counts_as_model_or_mechanism_failure": False,
        },
        "source_disjoint_from": {
            "P2_through_P4S": registry.get("source_disjoint_from_repositories"),
            "D1": "reserved_fresh_source_disjoint_surface_not_acquired",
            "D2": "independent_neural_surface_not_acquired",
            "training": "all_P4V2R2_tasks_permanently_excluded",
        },
        "information_flow": {
            "natural_request_obligations_source_and_visible_feedback_candidate_visible": True,
            "upstream_target_hidden_test_and_oracles_candidate_visible": False,
            "task_selection_conditioned_on_candidate_or_control_output": False,
            "route_labels_passed_to_scoring": False,
            "candidate_integrity_recomputed_independently": True,
        },
        "hosted_reference": {
            "model": "gpt-5.6-luna",
            "effort": "xhigh",
            "transport_state": "DEFINED_TRANSPORT_NOT_BOUND",
            "same_sealed_pool_required": True,
            "local_and_hosted_denominators_separate": True,
            "P4V2R2_blocking": False,
        },
        "counters": {
            "local_model_calls": 0,
            "hosted_model_calls": 0,
            "teacher_calls": 0,
            "deterministic_request_compiler_calls": 0,
            "compiler_oracle_evaluator_audits": green,
            "v2r2_transport_oracle_replays": v2_green,
            "public_calibration_cases_consumed": 0,
            "D1_cases_consumed": 0,
            "D2_cases_consumed": 0,
            "training_rows_written": 0,
        },
        "maximum_inference": (
            "A GREEN seal establishes only that ten fresh licensed parents fail, "
            "exact targets pass, evaluator-only v1 mechanics ceilings and equivalent "
            "v2r2 transport oracles replay successfully, and source, obligation, "
            "target, loss, and dependency corruptions are rejected. It is not "
            "candidate, mechanism-survivor, D1, D2, serving, training, or ASI Stack "
            "support evidence."
        ),
    }
    p2a.write_json(ROOT / "configs" / "theseus_p4v2r2_task_pool.json", pool)
    return pool


def audit_registry(registry: dict[str, Any]) -> list[str]:
    faults: list[str] = []
    if registry.get("policy") != "project_theseus_p4v2r2_online_source_selection_v1":
        faults.append("source_registry_policy_invalid")
    if registry.get("state") != (
        "FIXED_BEFORE_ARCHIVE_FETCH_PARENT_TARGET_EXECUTION_OR_CANDIDATE_GENERATION"
    ):
        faults.append("source_registry_not_fixed")
    tasks = p2a.dicts(registry.get("tasks"))
    if len(tasks) != 10 or registry.get("task_count") != 10:
        faults.append("source_registry_task_count_invalid")
    if [row.get("campaign_index") for row in tasks] != list(range(1, 11)):
        faults.append("campaign_indexes_invalid")
    repositories = [str(row.get("repository") or "").lower() for row in tasks]
    if len(set(repositories)) != 10:
        faults.append("repositories_not_distinct")
    boundaries = p2a.mapping(registry.get("boundaries"))
    if boundaries.get("candidate_generation_opened") is not False:
        faults.append("candidate_generation_already_opened")
    for key in (
        "parent_target_oracle_executions",
        "local_model_calls",
        "hosted_model_calls",
        "deterministic_request_compiler_calls",
        "teacher_calls",
        "training_rows_written",
        "D1_cases_consumed",
        "D2_cases_consumed",
    ):
        if int(boundaries.get(key) or 0) != 0:
            faults.append(f"selection_boundary_nonzero:{key}")
    if p2a.sha256_file(SOURCE_REGISTRY) != EXPECTED_SOURCE_REGISTRY_SHA256:
        faults.append("source_registry_digest_mismatch")
    fetch = p2a.read_json(SOURCE_FETCH)
    if (
        p2a.sha256_file(SOURCE_FETCH) != EXPECTED_SOURCE_FETCH_SHA256
        or fetch.get("trigger_state") != "GREEN"
        or fetch.get("faults") != []
        or int(fetch.get("candidate_or_control_calls") or 0) != 0
        or int(fetch.get("parent_target_oracle_executions") or 0) != 0
    ):
        faults.append("source_fetch_receipt_invalid")
    if p2a.sha256_file(HIDDEN_TEST) != EXPECTED_HIDDEN_TEST_SHA256:
        faults.append("hidden_evaluator_digest_mismatch")
    if p2a.sha256_file(VISIBLE_TEST) != EXPECTED_VISIBLE_TEST_SHA256:
        faults.append("visible_evaluator_digest_mismatch")
    corrections = p2a.read_json(ORACLE_CORRECTIONS)
    correction_rows = p2a.dicts(corrections.get("corrections"))
    if (
        corrections.get("policy")
        != "project_theseus_p4v2r2_oracle_materialization_correction_v1"
        or corrections.get("state")
        != "EVALUATOR_MECHANICS_REPAIR_BEFORE_CANDIDATE_GENERATION"
        or corrections.get("source_registry_sha256")
        != EXPECTED_SOURCE_REGISTRY_SHA256
        or {
            (row.get("stem"), row.get("unit_id"), row.get("field"))
            for row in correction_rows
        }
        != {
            ("p4v2r2_03_textual_6592", "U2", "parent_selector"),
            ("p4v2r2_03_textual_6592", "U1", "obligation_ids"),
            ("p4v2r2_03_textual_6592", "*", "unit_order"),
            ("p4v2r2_03_textual_6592", "U2", "include_target_decorators"),
            ("p4v2r2_04_scrapy_7783", "U1", "target_selectors"),
            ("p4v2r2_05_pyflakes_765", "*", "baseline_failure_markers"),
        }
    ):
        faults.append("selector_correction_binding_invalid")
    correction_boundaries = p2a.mapping(corrections.get("boundaries"))
    if correction_boundaries.get("candidate_generation_opened") is not False:
        faults.append("selector_correction_after_candidate_generation")
    for key in (
        "local_model_calls",
        "hosted_model_calls",
        "teacher_calls",
        "deterministic_request_compiler_calls",
        "D1_cases_consumed",
        "D2_cases_consumed",
        "training_rows_written",
    ):
        if int(correction_boundaries.get(key) or 0) != 0:
            faults.append(f"selector_correction_boundary_nonzero:{key}")
    instrument = p2a.read_json(INSTRUMENT)
    if (
        p2a.sha256_file(INSTRUMENT) != str(registry.get("instrument_sha256") or "")
        or instrument.get("runtime_attempt_namespace") != "p4v2r2_attempt1"
    ):
        faults.append("instrument_digest_mismatch")
    return faults


def materialize_task(
    source: dict[str, Any], registry: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    faults: list[str] = []
    index = int(source["campaign_index"])
    stem = str(source["stem"])
    suffix = stem.removeprefix("p4v2r2_")
    parent = FIXTURES / f"{stem}_parent.tar.gz"
    parent_upstream = FIXTURES / f"{stem}_parent_upstream.tar.gz"
    target = FIXTURES / f"{stem}_target.tar.gz"
    target_upstream = FIXTURES / f"{stem}_target_upstream.tar.gz"
    parent_sanitizer = (
        ROOT / "reports" / f"theseus_{stem}_parent_archive_sanitization.json"
    )
    target_sanitizer = (
        ROOT / "reports" / f"theseus_{stem}_target_archive_sanitization.json"
    )
    task_path = ROOT / "configs" / f"theseus_p4v2r2_task_{suffix}.json"
    evaluator_path = ROOT / "configs" / f"theseus_p4v2r2_evaluator_{suffix}.json"
    oracle_v1_path = FIXTURES / f"{stem}_oracle_v1.semantic_ir"
    oracle_v2_path = FIXTURES / f"{stem}_oracle_v2r2.semantic_ir"
    audit_path = ROOT / "reports" / f"theseus_p4v2r2_{suffix}_evaluator_audit.json"
    required = [
        parent,
        parent_upstream,
        target,
        target_upstream,
        parent_sanitizer,
        target_sanitizer,
        HIDDEN_TEST,
        VISIBLE_TEST,
    ]
    for path in required:
        if not path.is_file():
            faults.append(f"missing_source_artifact:{p2a.rel(path)}")
    parent_report = p2a.read_json(parent_sanitizer)
    target_report = p2a.read_json(target_sanitizer)
    faults.extend(
        base_pool.audit_sanitization_pair(
            parent_report,
            target_report,
            parent,
            parent_upstream,
            target,
            target_upstream,
            str(source["source_root"]),
            str(source["target_root"]),
            source,
        )
    )
    faults.extend(
        base_pool.audit_archive(parent, str(source["source_root"]), source, "parent")
    )
    faults.extend(
        base_pool.audit_archive(target, str(source["target_root"]), source, "target")
    )
    task = {
        "policy": p4.TASK_POLICY,
        "state": "SEALED_BEFORE_CANDIDATE_GENERATION",
        "campaign_index": index,
        "opaque_task_id": f"p4v2r2-cognitive-compilation-{index:02d}",
        "partition": "p4v2r2_cognitive_compilation_decision_development",
        "family": "bounded_python_correctness_repair",
        "natural_request": source["natural_request"],
        "source_archive": p2a.rel(parent),
        "source_archive_sha256": p2a.sha256_file(parent),
        "source_archive_root": source["source_root"],
        "source_provenance": {
            "repository": source["repository"],
            "url": f"https://github.com/{source['repository']}",
            "revision": source["parent_revision"],
            "source_selection_sealed_utc": registry["sealed_utc"],
            "license_spdx": source["license_spdx"],
            "license_paths": source["license_paths"],
            "upstream_request_url": (
                f"https://github.com/{source['repository']}/pull/"
                f"{source['pull_request']}"
            ),
            "upstream_request_title": source["pull_request_title"],
            "upstream_merged_utc": source["merged_utc"],
            "upstream_archive": p2a.rel(parent_upstream),
            "upstream_archive_sha256": p2a.sha256_file(parent_upstream),
            "archive_sanitization_report": p2a.rel(parent_sanitizer),
            "archive_sanitization_report_sha256": p2a.sha256_file(
                parent_sanitizer
            ),
        },
        "contamination_screen": {
            "public_benchmark": False,
            "previous_theseus_surface": False,
            "source_disjoint_from_p2_through_p4s": True,
            "task_selected_after_p4v2r2_instrument_freeze": True,
            "task_selected_before_any_p4v2r2_candidate_or_control": True,
            "later_patch_hidden_test_or_oracle_candidate_visible": False,
            "development_task_eligible_for_training": False,
            "development_task_eligible_for_D1_or_D2": False,
            "memorization_risk": "public_maintenance_change_not_claim_bearing",
        },
        "obligations": source["obligations"],
        "obligation_dependencies": source["obligation_dependencies"],
        "allowed_effect_paths": source["allowed_effect_paths"],
        "candidate_visible_context": {
            "searches": source["searches"],
            "reads": source["reads"],
            "maximum_total_characters": 12000,
        },
        "visible_verifier": {
            "command": base_pool.python312_exec_command(VISIBLE_TEST, source["case"]),
            "timeout_seconds": 60,
            "answer_specific": True,
            "candidate_prompt_visibility": False,
        },
        "visible_feedback_map": source["visible_markers"],
        "semantic_ir_contract": {
            "version": "theseus_semantic_ir_v2r2_labeled",
            "maximum_symbol_nodes": 1_000_000,
            "maximum_semantic_scope_nodes": 80,
            "maximum_units": 8,
            "source_target_obligation_loss_and_dependency_identity_required": True,
        },
        "effect_authority": "disposable_snapshot_only",
        "maximum_inference": (
            "One P4-v2r2 development observation only; no D1, D2, serving, "
            "training, or ASI Stack support claim."
        ),
    }
    task["semantic_ir_contract"]["maximum_symbol_nodes"] = (
        p4s_pool.exact_lowerer_inventory_count(task, parent)
    )
    p2a.write_json(task_path, task)
    faults.extend(p4s_pool.audit_task_surface(task, parent))
    try:
        oracle_v1 = build_oracle(source, task, parent, target, transport="v1")
        oracle_v1_path.write_text(oracle_v1, encoding="utf-8")
        oracle_v2 = build_oracle(source, task, parent, target, transport="v2r2")
        oracle_v2_path.write_text(oracle_v2, encoding="utf-8")
    except (OSError, ValueError, p2a.InstrumentFault, p4.P4Fault) as exc:
        faults.append(f"oracle_materialization_fault:{type(exc).__name__}:{exc}")
    evaluator = {
        "policy": "project_theseus_p4_cognitive_compilation_evaluator_v1",
        "state": "SEALED_BEFORE_CANDIDATE_GENERATION",
        "task_manifest": p2a.rel(task_path),
        "task_manifest_sha256": p2a.sha256_file(task_path),
        "baseline_must_fail": True,
        "baseline_failure_markers": baseline_failure_marker_map().get(
            stem, [f"P4V2R2_FAIL_{source['case']}"]
        ),
        "hidden_test_files": [
            {
                "source": p2a.rel(HIDDEN_TEST),
                "sha256": p2a.sha256_file(HIDDEN_TEST),
                "destination": "theseus_p4v2r2_hidden_test.py",
            }
        ],
        "hidden_verifier": {
            "command": base_pool.python312_exec_command(
                Path("theseus_p4v2r2_hidden_test.py"), source["case"]
            ),
            "timeout_seconds": 60,
            "network": "forbidden",
        },
        "target_archive": p2a.rel(target),
        "target_archive_sha256": p2a.sha256_file(target),
        "target_archive_root": source["target_root"],
        "target_provenance": {
            "revision": source["target_revision"],
            "merge_revision": source["merge_revision"],
            "upstream_archive": p2a.rel(target_upstream),
            "upstream_archive_sha256": p2a.sha256_file(target_upstream),
            "archive_sanitization_report": p2a.rel(target_sanitizer),
            "archive_sanitization_report_sha256": p2a.sha256_file(
                target_sanitizer
            ),
        },
        "target_must_pass": True,
        "oracle_ir_file": p2a.rel(oracle_v1_path),
        "oracle_ir_sha256": p2a.sha256_file(oracle_v1_path),
        "treatment_transport_oracle_ir_file": p2a.rel(oracle_v2_path),
        "treatment_transport_oracle_ir_sha256": p2a.sha256_file(oracle_v2_path),
        "blindness": {
            "candidate_generation_may_read_this_manifest": False,
            "route_label_passed_to_scoring": False,
            "later_patch_candidate_visible": False,
            "hidden_test_candidate_visible": False,
            "oracle_candidate_visible": False,
            "candidate_emitted_integrity_flags_trusted": False,
        },
        "maximum_inference": (
            "A GREEN audit establishes evaluator reachability and sensitivity for "
            "one sealed P4-v2r2 task only."
        ),
    }
    p2a.write_json(evaluator_path, evaluator)
    entry = {
        "campaign_index": index,
        "stem": stem,
        "repository": source["repository"],
        "pull_request_url": (
            f"https://github.com/{source['repository']}/pull/{source['pull_request']}"
        ),
        "license_spdx": source["license_spdx"],
        "parent_revision": source["parent_revision"],
        "target_revision": source["target_revision"],
        "task": p2a.rel(task_path),
        "task_sha256": p2a.sha256_file(task_path),
        "evaluator": p2a.rel(evaluator_path),
        "evaluator_sha256": p2a.sha256_file(evaluator_path),
        "oracle_ir": p2a.rel(oracle_v1_path),
        "oracle_ir_sha256": p2a.sha256_file(oracle_v1_path),
        "treatment_transport_oracle_ir": p2a.rel(oracle_v2_path),
        "treatment_transport_oracle_ir_sha256": p2a.sha256_file(oracle_v2_path),
        "evaluator_audit": p2a.rel(audit_path),
    }
    return entry, faults


def build_oracle(
    source: dict[str, Any],
    task: dict[str, Any],
    parent: Path,
    target: Path,
    *,
    transport: str,
) -> str:
    if transport not in {"v1", "v2r2"}:
        raise ValueError("oracle transport invalid")
    with tempfile.TemporaryDirectory(prefix="theseus-p4v2r2-oracle-build-") as tmp:
        parent_root = Path(tmp) / "parent"
        target_root = Path(tmp) / "target"
        p2a.extract_source_archive(parent, parent_root, str(source["source_root"]))
        p2a.extract_source_archive(target, target_root, str(source["target_root"]))
        symbols = p4s.semantic_scope_symbol_table(parent_root, task)
        symbol_by_identity = {
            (row["path"], row["start_line"], row["end_line"], row["sha256"]): row
            for row in p2a.dicts(symbols.get("nodes"))
        }
        if transport == "v1":
            chunks = [
                p4.IR_HEADER,
                f"SOURCE {symbols['source_digest']}",
                "OBLIGATIONS "
                + ",".join(row["id"] for row in source["obligations"]),
            ]
        else:
            chunks = [
                ir_v2r2.HEADER,
                f"SOURCE {symbols['source_digest']}",
                "ALL_OBLIGATIONS "
                + ",".join(row["id"] for row in source["obligations"]),
            ]
        selector_corrections = selector_correction_map()
        obligation_corrections = obligation_correction_map()
        target_selector_corrections = target_selector_correction_map()
        decorator_corrections = target_decorator_correction_set()
        for unit in ordered_oracle_units(source):
            path = str(unit["path"])
            unit_key = (str(source["stem"]), str(unit["id"]))
            parent_selector = selector_corrections.get(
                unit_key,
                p2a.mapping(unit["parent_selector"]),
            )
            obligation_ids = obligation_corrections.get(
                unit_key, p2a.strings(unit["obligation_ids"])
            )
            parent_node = base_pool.select_node(
                parent_root / path, parent_selector
            )
            parent_segment = base_pool.node_segment(
                parent_root / path, parent_node["node"]
            )
            key = (
                path,
                parent_node["node"].lineno,
                parent_node["node"].end_lineno,
                p2a.sha256_text(parent_segment),
            )
            symbol = symbol_by_identity.get(key)
            if not symbol:
                raise ValueError(
                    "oracle parent node absent from semantic-scope table: "
                    + str(unit["id"])
                )
            target_selectors = target_selector_corrections.get(
                unit_key, p2a.dicts(unit["target_selectors"])
            )
            selected_targets = [
                base_pool.select_node(target_root / path, p2a.mapping(selector))
                for selector in target_selectors
            ]
            replacement = join_target_segments(
                target_root / path,
                selected_targets,
                int(symbol.get("start_col") or 0),
                include_decorators=unit_key in decorator_corrections,
            )
            if unit["operation"] in {"INSERT_BEFORE", "INSERT_AFTER"}:
                replacement = (" " * int(symbol.get("start_col") or 0)) + replacement
            if transport == "v1":
                chunks.extend(
                    [
                        "UNIT "
                        + " ".join(
                            [
                                unit["id"],
                                ",".join(obligation_ids),
                                unit["operation"],
                                path,
                                symbol["id"],
                                symbol["sha256"],
                            ]
                        ),
                        "<<<",
                        replacement,
                        ">>>",
                    ]
                )
            else:
                chunks.extend(
                    [
                        f"UNIT {unit['id']}",
                        "OBLIGATIONS " + ",".join(obligation_ids),
                        f"OP {unit['operation']}",
                        f"PATH {path}",
                        f"NODE {symbol['id']}",
                        f"NODE_SHA {symbol['sha256']}",
                        "<<<",
                        replacement,
                        ">>>",
                        "END_UNIT",
                    ]
                )
        chunks.extend(["LOSS NONE", "END"])
        text = "\n".join(chunks) + "\n"
        parsed = (
            p4.parse_semantic_ir(text, task, parent_root)
            if transport == "v1"
            else ir_v2r2.parse(text, task, parent_root)
        )
        if parsed.get("faults"):
            raise ValueError(
                "oracle parse faults: "
                + ",".join(p2a.strings(parsed.get("faults")))
            )
        return text


def selector_correction_map() -> dict[tuple[str, str], dict[str, Any]]:
    value = p2a.read_json(ORACLE_CORRECTIONS)
    return {
        (str(row.get("stem") or ""), str(row.get("unit_id") or "")): p2a.mapping(
            row.get("replacement")
        )
        for row in p2a.dicts(value.get("corrections"))
        if row.get("field") == "parent_selector"
    }


def obligation_correction_map() -> dict[tuple[str, str], list[str]]:
    value = p2a.read_json(ORACLE_CORRECTIONS)
    return {
        (str(row.get("stem") or ""), str(row.get("unit_id") or "")): p2a.strings(
            row.get("replacement")
        )
        for row in p2a.dicts(value.get("corrections"))
        if row.get("field") == "obligation_ids"
    }


def target_selector_correction_map() -> dict[tuple[str, str], list[dict[str, Any]]]:
    value = p2a.read_json(ORACLE_CORRECTIONS)
    return {
        (str(row.get("stem") or ""), str(row.get("unit_id") or "")): p2a.dicts(
            row.get("replacement")
        )
        for row in p2a.dicts(value.get("corrections"))
        if row.get("field") == "target_selectors"
    }


def target_decorator_correction_set() -> set[tuple[str, str]]:
    value = p2a.read_json(ORACLE_CORRECTIONS)
    return {
        (str(row.get("stem") or ""), str(row.get("unit_id") or ""))
        for row in p2a.dicts(value.get("corrections"))
        if row.get("field") == "include_target_decorators"
        and row.get("replacement") is True
    }


def unit_order_correction_map() -> dict[str, list[str]]:
    value = p2a.read_json(ORACLE_CORRECTIONS)
    return {
        str(row.get("stem") or ""): p2a.strings(row.get("replacement"))
        for row in p2a.dicts(value.get("corrections"))
        if row.get("field") == "unit_order"
    }


def baseline_failure_marker_map() -> dict[str, list[str]]:
    value = p2a.read_json(ORACLE_CORRECTIONS)
    return {
        str(row.get("stem") or ""): p2a.strings(row.get("replacement"))
        for row in p2a.dicts(value.get("corrections"))
        if row.get("field") == "baseline_failure_markers"
    }


def ordered_oracle_units(source: dict[str, Any]) -> list[dict[str, Any]]:
    units = p2a.dicts(source.get("oracle_units"))
    order = unit_order_correction_map().get(str(source.get("stem") or ""))
    if not order:
        return units
    by_id = {str(unit.get("id") or ""): unit for unit in units}
    if set(order) != set(by_id):
        raise ValueError("oracle unit-order correction is not a permutation")
    return [by_id[unit_id] for unit_id in order]


def join_target_segments(
    path: Path,
    rows: list[dict[str, Any]],
    parent_col: int,
    *,
    include_decorators: bool,
) -> str:
    values = [
        target_node_segment(path, row["node"], include_decorators=include_decorators)
        for row in rows
    ]
    if not values:
        raise ValueError("oracle target selector list empty")
    separator = (
        "\n\n"
        if all(
            type(row["node"]).__name__
            in {"FunctionDef", "AsyncFunctionDef", "ClassDef"}
            for row in rows
        )
        else "\n"
    )
    return values[0] + "".join(
        separator + (" " * parent_col) + value for value in values[1:]
    )


def target_node_segment(path: Path, node: Any, *, include_decorators: bool) -> str:
    if not include_decorators or not getattr(node, "decorator_list", None):
        return base_pool.node_segment(path, node)
    lines = path.read_text(encoding="utf-8").splitlines()
    start = min(int(item.lineno) for item in node.decorator_list)
    end = int(getattr(node, "end_lineno", node.lineno) or node.lineno)
    start_col = min(int(item.col_offset) for item in node.decorator_list)
    end_col = int(
        getattr(node, "end_col_offset", len(lines[end - 1])) or len(lines[end - 1])
    )
    return "\n".join(
        [
            lines[start - 1][start_col:],
            *lines[start : end - 1],
            lines[end - 1][:end_col],
        ]
    )


def audit_dependency_corruption(entry: dict[str, Any]) -> dict[str, Any]:
    task = p2a.read_json(ROOT / str(entry.get("task") or ""))
    oracle_path = ROOT / str(entry.get("treatment_transport_oracle_ir") or "")
    dependencies = p2a.dicts(task.get("obligation_dependencies"))
    if not dependencies:
        return {"rejected": False, "faults": ["dependency_set_empty"]}
    text = oracle_path.read_text(encoding="utf-8")
    mutation = ""
    removed = ""
    for match in ir_v2.UNIT_RE.finditer(text):
        refs = match.group(2).split(",")
        for dependency in dependencies:
            before = str(dependency.get("before") or "")
            after = str(dependency.get("after") or "")
            if before in refs and after in refs and len(refs) > 1:
                mutated_refs = [value for value in refs if value != before]
                start, end = match.span(2)
                mutation = text[:start] + ",".join(mutated_refs) + text[end:]
                removed = before
                break
        if mutation:
            break
    if not mutation:
        return {"rejected": False, "faults": ["dependency_mutation_unavailable"]}
    with tempfile.TemporaryDirectory(
        prefix="theseus-p4v2r2-dependency-corruption-"
    ) as tmp:
        root = Path(tmp) / "source"
        p2a.extract_source_archive(
            p2a.resolve(str(task.get("source_archive") or "")),
            root,
            str(task.get("source_archive_root") or ""),
        )
        faults = p2a.strings(ir_v2r2.parse(mutation, task, root).get("faults"))
    return {
        "rejected": bool(faults) and "semantic_unit_dependency_not_closed" in faults,
        "removed_prerequisite": removed,
        "faults": faults,
    }


def audit_v2r2_oracle(entry: dict[str, Any]) -> dict[str, Any]:
    task = p2a.read_json(ROOT / str(entry.get("task") or ""))
    evaluator = p2a.read_json(ROOT / str(entry.get("evaluator") or ""))
    v1_text = (ROOT / str(entry.get("oracle_ir") or "")).read_text(encoding="utf-8")
    v2_text = (
        ROOT / str(entry.get("treatment_transport_oracle_ir") or "")
    ).read_text(encoding="utf-8")
    faults: list[str] = []
    with tempfile.TemporaryDirectory(prefix="theseus-p4v2r2-oracle-replay-") as tmp:
        v1_root = Path(tmp) / "v1"
        v2_root = Path(tmp) / "v2"
        archive = p2a.resolve(str(task.get("source_archive") or ""))
        root_name = str(task.get("source_archive_root") or "")
        p2a.extract_source_archive(archive, v1_root, root_name)
        p2a.extract_source_archive(archive, v2_root, root_name)
        parsed_v1 = p4.parse_semantic_ir(v1_text, task, v1_root)
        parsed_v2 = ir_v2r2.parse(v2_text, task, v2_root)
        if parsed_v1.get("faults"):
            faults.append("v1_oracle_parse_fault")
        if parsed_v2.get("faults"):
            faults.append("v2r2_oracle_parse_fault")
        actions_v1 = p2a.dicts(parsed_v1.get("actions"))
        actions_v2 = p2a.dicts(parsed_v2.get("actions"))
        if p2a.stable_hash(actions_v1) != p2a.stable_hash(actions_v2):
            faults.append("v1_v2r2_action_equivalence_mismatch")
        apply_faults = p2a.apply_actions(v2_root, actions_v2) if not faults else []
        faults.extend(f"v2r2_apply:{value}" for value in apply_faults)
        visible = p2a.run_visible_verifier(v2_root, task) if not faults else {}
        p2a_evaluator.overlay_hidden_tests(evaluator, v2_root)
        hidden = (
            p2a_evaluator.run_hidden_verifier(evaluator, v2_root)
            if not faults
            else {}
        )
        if visible.get("passed") is not True:
            faults.append("v2r2_visible_replay_failed")
        if hidden.get("passed") is not True:
            faults.append("v2r2_hidden_replay_failed")
    return {
        "policy": "project_theseus_p4v2r2_transport_oracle_replay_v1",
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": faults,
        "v1_v2r2_actions_equivalent": not any(
            value == "v1_v2r2_action_equivalence_mismatch" for value in faults
        ),
        "visible_passed": visible.get("passed") is True,
        "hidden_passed": hidden.get("passed") is True,
        "candidate_or_control_calls": 0,
        "maximum_inference": (
            "Transport-oracle parse, lowering, action-equivalence, and verifier "
            "mechanics for one sealed task only."
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
