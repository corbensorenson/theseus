#!/usr/bin/env python3
"""Materialize and seal the ten-task mechanics-qualified P4S pool."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4_cognitive_compilation as p4  # noqa: E402
import theseus_p4_task_pool as base_pool  # noqa: E402
import theseus_p4s_cognitive_compilation as p4s  # noqa: E402


SOURCE_REGISTRY = ROOT / "configs" / "theseus_p4s_task_sources.json"
SOURCE_FETCH = ROOT / "reports" / "theseus_p4s_source_fetch.json"
INSTRUMENT = ROOT / "configs" / "theseus_p4s_cognitive_compilation_instrument.json"
INSTRUMENT_AUDIT = ROOT / "reports" / "theseus_p4s_cognitive_compilation_instrument_audit_r2.json"
FIXTURES = ROOT / "tests" / "fixtures" / "theseus_p4s_online"
HIDDEN_TEST = FIXTURES / "theseus_p4s_hidden_test.py"
VISIBLE_TEST = FIXTURES / "theseus_p4s_visible_test.py"
SOURCE_SELECTION_COMMIT = "560df8f437d470e8eea9fbd927ee7854f1b93e74"
SOURCE_ACQUISITION_COMMIT = "d19be2eeced6d3a715c49004311b09960a50d507"
EVALUATOR_SURFACE_COMMIT = "f265e87ff0ce550a2ab5deab761da4cd72ed6824"
EXPECTED_SOURCE_REGISTRY_SHA256 = "4237d99b6c051d2e692c2678539183d28194b10a0a704c323b38b481eaef508f"
EXPECTED_SOURCE_FETCH_SHA256 = "907f03465a4ce645eae5f7db1868501cc27c5b712495920b2e1a0cd8e59b7164"
EXPECTED_SELECTION_INSTRUMENT_SHA256 = "19e546e0256bdbfbaaac2bcfa881358bc770dbff6710b6bf2de29b6d713d0828"
EXPECTED_INSTRUMENT_SHA256 = "6dba153b0d54753c3000fa3c4b949ffdce2ab3e24363084b962f34d035b771b1"
EXPECTED_INSTRUMENT_AUDIT_SHA256 = "6e3d4cd8d10457545e0fc47308fff8afbcff34129b0fd456c2ad0344d93d0535"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialize-only", action="store_true")
    args = parser.parse_args()
    report = materialize_pool(run_audits=not args.materialize_only)
    print(json.dumps({
        "state": report["state"],
        "task_count": report["task_count"],
        "green_evaluator_audits": report["green_evaluator_audits"],
        "dependency_corruptions_rejected": report["dependency_corruptions_rejected"],
        "faults": report["faults"],
    }, indent=2, sort_keys=True))
    return 0 if report["state"] == "SEALED_BEFORE_CANDIDATE_GENERATION" else 2


def materialize_pool(*, run_audits: bool) -> dict[str, Any]:
    registry = p2a.read_json(SOURCE_REGISTRY)
    faults = audit_registry(registry)
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
            audit_path = ROOT / entry["evaluator_audit"]
            audit = p2a.read_json(audit_path)
            if result.returncode != 0:
                faults.append(f"evaluator_audit_red:{entry['stem']}:{audit.get('faults')}")
            dependency = audit_dependency_corruption(entry)
            if dependency.get("rejected") is not True:
                faults.append(f"dependency_corruption_not_rejected:{entry['stem']}")
            entry.update({
                "evaluator_audit_sha256": p2a.sha256_file(audit_path),
                "evaluator_audit_trigger_state": audit.get("trigger_state"),
                "baseline_parent_failed": p2a.mapping(
                    audit.get("baseline_verification")
                ).get("hidden_passed") is False,
                "upstream_target_passed": p2a.mapping(
                    audit.get("target_verification")
                ).get("hidden_passed") is True,
                "compiler_oracle_passed": p2a.mapping(
                    audit.get("compiler_oracle_verification")
                ).get("hidden_passed") is True,
                "four_base_corruptions_rejected": all(
                    p2a.mapping(audit.get("corruption_intervention_rejections")).values()
                ),
                "dependency_corruption": dependency,
            })
    else:
        for entry in entries:
            entry.update({
                "evaluator_audit_sha256": "",
                "evaluator_audit_trigger_state": "NOT_RUN",
                "baseline_parent_failed": False,
                "upstream_target_passed": False,
                "compiler_oracle_passed": False,
                "four_base_corruptions_rejected": False,
                "dependency_corruption": {"rejected": False, "faults": ["not_run"]},
            })
    green = sum(row.get("evaluator_audit_trigger_state") == "GREEN" for row in entries)
    dependency_green = sum(
        p2a.mapping(row.get("dependency_corruption")).get("rejected") is True
        for row in entries
    )
    if run_audits and green != 10:
        faults.append("not_all_evaluator_audits_green")
    if run_audits and dependency_green != 10:
        faults.append("not_all_dependency_corruptions_rejected")
    state = (
        "SEALED_BEFORE_CANDIDATE_GENERATION"
        if run_audits and not faults and green == 10 and dependency_green == 10
        else "INVALID_NOT_SEALED"
    )
    pool = {
        "policy": "project_theseus_p4s_cognitive_compilation_task_pool_v1",
        "state": state,
        "partition": "p4s_cognitive_compilation_decision_development",
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
        "selection_instrument_freeze_commit": "42abb39b",
        "execution_instrument_rebind_commit": "bf4f63b9",
        "instrument_sha256": p2a.sha256_file(INSTRUMENT),
        "instrument_audit": p2a.rel(INSTRUMENT_AUDIT),
        "instrument_audit_sha256": p2a.sha256_file(INSTRUMENT_AUDIT),
        "task_count": len(entries),
        "green_evaluator_audits": green,
        "dependency_corruptions_rejected": dependency_green,
        "distinct_repositories": len({row.get("repository") for row in entries}),
        "tasks": entries,
        "faults": sorted(set(faults)),
        "generation_budget": {
            "project_selected_quality_token_cap": None,
            "normal_completion": ["parser_complete", "model_eos"],
            "physical_boundary": "model_declared_context_window_minus_exact_prompt_tokens",
            "boundary_hit_invalidates_observation": True,
            "boundary_hit_counts_as_model_or_mechanism_failure": False,
        },
        "source_disjoint_from": {
            "P2_P3_P4_P4R": registry.get("source_disjoint_from_repositories"),
            "D1": "reserved_fresh_source_disjoint_surface_not_acquired",
            "D2": "independent_neural_surface_not_acquired",
            "training": "all_P4S_tasks_permanently_excluded",
        },
        "information_flow": {
            "natural_request_obligations_source_and_visible_feedback_candidate_visible": True,
            "upstream_target_hidden_test_and_oracle_candidate_visible": False,
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
            "P4S_blocking": False,
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
        "maximum_inference": (
            "A GREEN seal establishes only that ten fresh licensed parents fail, exact "
            "targets and evaluator-only Semantic-IR oracles pass, and source, obligation, "
            "target, loss, and dependency corruptions are rejected. It is not candidate, "
            "mechanism, D1, D2, serving, training, or ASI Stack support evidence."
        ),
    }
    p2a.write_json(ROOT / "configs" / "theseus_p4s_task_pool.json", pool)
    return pool


def audit_registry(registry: dict[str, Any]) -> list[str]:
    faults: list[str] = []
    if registry.get("policy") != "project_theseus_p4s_online_source_selection_v1":
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
        "parent_target_oracle_executions", "local_model_calls", "hosted_model_calls",
        "deterministic_request_compiler_calls", "teacher_calls", "training_rows_written",
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
    ):
        faults.append("source_fetch_receipt_invalid")
    instrument = p2a.read_json(INSTRUMENT)
    namespace_repair = p2a.mapping(
        instrument.get("invalid_attempt1_runtime_namespace_repair")
    )
    if (
        p2a.sha256_file(INSTRUMENT) != EXPECTED_INSTRUMENT_SHA256
        or instrument.get("runtime_attempt_namespace") != "attempt2"
        or namespace_repair.get("state")
        != "PROSPECTIVELY_BOUND_BEFORE_REPAIRED_ATTEMPT2"
        or namespace_repair.get("runtime_attempt_namespace") != "attempt2"
        or namespace_repair.get("repair_commit")
        != "db86ea89bd1b4d62e096ce5c2cb41a93052aeb6b"
    ):
        faults.append("instrument_digest_mismatch")
    audit = p2a.read_json(INSTRUMENT_AUDIT)
    if (
        p2a.sha256_file(INSTRUMENT_AUDIT) != EXPECTED_INSTRUMENT_AUDIT_SHA256
        or audit.get("trigger_state") != "GREEN"
        or audit.get("faults") != []
    ):
        faults.append("instrument_audit_invalid")
    return faults


def materialize_task(
    source: dict[str, Any], registry: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    faults: list[str] = []
    index = int(source["campaign_index"])
    stem = str(source["stem"])
    suffix = stem.removeprefix("p4s_")
    parent = FIXTURES / f"{stem}_parent.tar.gz"
    parent_upstream = FIXTURES / f"{stem}_parent_upstream.tar.gz"
    target = FIXTURES / f"{stem}_target.tar.gz"
    target_upstream = FIXTURES / f"{stem}_target_upstream.tar.gz"
    parent_sanitizer = ROOT / "reports" / f"theseus_{stem}_parent_archive_sanitization.json"
    target_sanitizer = ROOT / "reports" / f"theseus_{stem}_target_archive_sanitization.json"
    task_path = ROOT / "configs" / f"theseus_p4s_task_{suffix}.json"
    evaluator_path = ROOT / "configs" / f"theseus_p4s_evaluator_{suffix}.json"
    oracle_path = FIXTURES / f"{stem}_oracle.semantic_ir"
    audit_path = ROOT / "reports" / f"theseus_p4s_{suffix}_evaluator_audit.json"
    required = [
        parent, parent_upstream, target, target_upstream, parent_sanitizer,
        target_sanitizer, HIDDEN_TEST, VISIBLE_TEST,
    ]
    for path in required:
        if not path.is_file():
            faults.append(f"missing_source_artifact:{p2a.rel(path)}")
    parent_report = p2a.read_json(parent_sanitizer)
    target_report = p2a.read_json(target_sanitizer)
    faults.extend(base_pool.audit_sanitization_pair(
        parent_report, target_report, parent, parent_upstream, target, target_upstream,
        str(source["source_root"]), str(source["target_root"]), source,
    ))
    faults.extend(base_pool.audit_archive(
        parent, str(source["source_root"]), source, "parent"
    ))
    faults.extend(base_pool.audit_archive(
        target, str(source["target_root"]), source, "target"
    ))
    task = {
        "policy": p4.TASK_POLICY,
        "state": "SEALED_BEFORE_CANDIDATE_GENERATION",
        "campaign_index": index,
        "opaque_task_id": f"p4s-cognitive-compilation-{index:02d}",
        "partition": "p4s_cognitive_compilation_decision_development",
        "family": "bounded_python_correctness_repair",
        "natural_request": source["natural_request"],
        "source_archive": p2a.rel(parent),
        "source_archive_sha256": p2a.sha256_file(parent),
        "source_archive_root": source["source_root"],
        "source_provenance": {
            "repository": source["repository"],
            "url": f"https://github.com/{source['repository']}",
            "revision": source["parent_revision"],
            "retrieved_utc": registry["sealed_utc"],
            "license_spdx": source["license_spdx"],
            "license_paths": source["license_paths"],
            "upstream_request_url": f"https://github.com/{source['repository']}/pull/{source['pull_request']}",
            "upstream_request_title": source["pull_request_title"],
            "upstream_merged_utc": source["merged_utc"],
            "upstream_archive": p2a.rel(parent_upstream),
            "upstream_archive_sha256": p2a.sha256_file(parent_upstream),
            "archive_sanitization_report": p2a.rel(parent_sanitizer),
            "archive_sanitization_report_sha256": p2a.sha256_file(parent_sanitizer),
        },
        "contamination_screen": {
            "public_benchmark": False,
            "previous_theseus_surface": False,
            "source_disjoint_from_p2_p3_p4_p4r": True,
            "task_selected_after_p4s_instrument_freeze": True,
            "task_selected_before_any_p4s_candidate_or_control": True,
            "later_patch_hidden_test_or_oracle_candidate_visible": False,
            "development_task_eligible_for_training": False,
            "development_task_eligible_for_D1_or_D2": False,
            "memorization_risk": "public_recent_maintenance_change_not_claim_bearing",
        },
        "obligations": source["obligations"],
        "obligation_dependencies": source["obligation_dependencies"],
        "allowed_effect_paths": source["allowed_effect_paths"],
        "candidate_visible_context": {
            "searches": source["searches"],
            "reads": source["reads"],
            "maximum_total_characters": 9000,
        },
        "visible_verifier": {
            "command": base_pool.python312_exec_command(VISIBLE_TEST, source["case"]),
            "timeout_seconds": 60,
            "answer_specific": True,
            "candidate_prompt_visibility": False,
        },
        "visible_feedback_map": source["visible_markers"],
        "semantic_ir_contract": {
            "version": "theseus_semantic_ir_v2r1_labeled",
            "maximum_symbol_nodes": 1000000,
            "maximum_semantic_scope_nodes": 80,
            "maximum_units": 8,
            "source_target_obligation_loss_and_dependency_identity_required": True,
        },
        "effect_authority": "disposable_snapshot_only",
        "maximum_inference": (
            "One P4S development observation only; no D1, D2, serving, training, "
            "or ASI Stack support claim."
        ),
    }
    task["semantic_ir_contract"]["maximum_symbol_nodes"] = exact_lowerer_inventory_count(
        task, parent
    )
    p2a.write_json(task_path, task)
    faults.extend(audit_task_surface(task, parent))
    try:
        oracle_text = build_oracle(source, task, parent, target)
        oracle_path.write_text(oracle_text, encoding="utf-8")
    except (OSError, ValueError, p2a.InstrumentFault, p4.P4Fault) as exc:
        faults.append(f"oracle_materialization_fault:{type(exc).__name__}:{exc}")
    evaluator = {
        "policy": "project_theseus_p4_cognitive_compilation_evaluator_v1",
        "state": "SEALED_BEFORE_CANDIDATE_GENERATION",
        "task_manifest": p2a.rel(task_path),
        "task_manifest_sha256": p2a.sha256_file(task_path),
        "baseline_must_fail": True,
        "baseline_failure_markers": [f"P4S_FAIL_{source['case']}"],
        "hidden_test_files": [{
            "source": p2a.rel(HIDDEN_TEST),
            "sha256": p2a.sha256_file(HIDDEN_TEST),
            "destination": "theseus_p4s_hidden_test.py",
        }],
        "hidden_verifier": {
            "command": base_pool.python312_exec_command(
                Path("theseus_p4s_hidden_test.py"), source["case"]
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
            "archive_sanitization_report_sha256": p2a.sha256_file(target_sanitizer),
        },
        "target_must_pass": True,
        "oracle_ir_file": p2a.rel(oracle_path),
        "oracle_ir_sha256": p2a.sha256_file(oracle_path),
        "blindness": {
            "candidate_generation_may_read_this_manifest": False,
            "route_label_passed_to_scoring": False,
            "later_patch_candidate_visible": False,
            "hidden_test_candidate_visible": False,
            "oracle_candidate_visible": False,
            "candidate_emitted_integrity_flags_trusted": False,
        },
        "maximum_inference": (
            "A GREEN audit establishes evaluator reachability and sensitivity for one "
            "sealed P4S task only."
        ),
    }
    p2a.write_json(evaluator_path, evaluator)
    entry = {
        "campaign_index": index,
        "stem": stem,
        "repository": source["repository"],
        "pull_request_url": f"https://github.com/{source['repository']}/pull/{source['pull_request']}",
        "license_spdx": source["license_spdx"],
        "parent_revision": source["parent_revision"],
        "target_revision": source["target_revision"],
        "task": p2a.rel(task_path),
        "task_sha256": p2a.sha256_file(task_path),
        "evaluator": p2a.rel(evaluator_path),
        "evaluator_sha256": p2a.sha256_file(evaluator_path),
        "oracle_ir": p2a.rel(oracle_path),
        "oracle_ir_sha256": p2a.sha256_file(oracle_path),
        "evaluator_audit": p2a.rel(audit_path),
    }
    return entry, faults


def build_oracle(
    source: dict[str, Any], task: dict[str, Any], parent: Path, target: Path
) -> str:
    with tempfile.TemporaryDirectory(prefix="theseus-p4s-oracle-build-") as tmp:
        parent_root = Path(tmp) / "parent"
        target_root = Path(tmp) / "target"
        p2a.extract_source_archive(parent, parent_root, str(source["source_root"]))
        p2a.extract_source_archive(target, target_root, str(source["target_root"]))
        symbols = p4s.semantic_scope_symbol_table(parent_root, task)
        symbol_by_identity = {
            (row["path"], row["start_line"], row["end_line"], row["sha256"]): row
            for row in p2a.dicts(symbols.get("nodes"))
        }
        chunks = [
            p4.IR_HEADER,
            f"SOURCE {symbols['source_digest']}",
            "OBLIGATIONS " + ",".join(row["id"] for row in source["obligations"]),
        ]
        for unit in source["oracle_units"]:
            path = str(unit["path"])
            parent_node = base_pool.select_node(
                parent_root / path, p2a.mapping(unit["parent_selector"])
            )
            parent_segment = base_pool.node_segment(parent_root / path, parent_node["node"])
            key = (
                path,
                parent_node["node"].lineno,
                parent_node["node"].end_lineno,
                p2a.sha256_text(parent_segment),
            )
            symbol = symbol_by_identity.get(key)
            if not symbol:
                raise ValueError(
                    f"oracle parent node absent from P4S semantic-scope table: {unit['id']}"
                )
            selected_targets = [
                base_pool.select_node(target_root / path, p2a.mapping(selector))
                for selector in unit["target_selectors"]
            ]
            replacement = base_pool.join_target_segments(
                target_root / path, selected_targets, int(symbol.get("start_col") or 0)
            )
            if unit["operation"] in {"INSERT_BEFORE", "INSERT_AFTER"}:
                replacement = (" " * int(symbol.get("start_col") or 0)) + replacement
            chunks.extend([
                "UNIT " + " ".join([
                    unit["id"], ",".join(unit["obligation_ids"]), unit["operation"],
                    path, symbol["id"], symbol["sha256"],
                ]),
                "<<<",
                replacement,
                ">>>",
            ])
        chunks.extend(["LOSS NONE", "END"])
        text = "\n".join(chunks) + "\n"
        parsed = p4.parse_semantic_ir(text, task, parent_root)
        if parsed.get("faults"):
            raise ValueError(
                "oracle parse faults: " + ",".join(p2a.strings(parsed.get("faults")))
            )
        return text


def audit_dependency_corruption(entry: dict[str, Any]) -> dict[str, Any]:
    task = p2a.read_json(ROOT / str(entry.get("task") or ""))
    oracle_path = ROOT / str(entry.get("oracle_ir") or "")
    dependencies = p2a.dicts(task.get("obligation_dependencies"))
    if not dependencies:
        return {"rejected": False, "faults": ["dependency_set_empty"]}
    text = oracle_path.read_text(encoding="utf-8")
    mutation = ""
    removed = ""
    for match in p4.IR_UNIT_RE.finditer(text):
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
    with tempfile.TemporaryDirectory(prefix="theseus-p4s-dependency-corruption-") as tmp:
        root = Path(tmp) / "source"
        p2a.extract_source_archive(
            p2a.resolve(str(task.get("source_archive") or "")),
            root,
            str(task.get("source_archive_root") or ""),
        )
        faults = p2a.strings(p4.parse_semantic_ir(mutation, task, root).get("faults"))
    return {
        "rejected": bool(faults) and "semantic_unit_dependency_not_closed" in faults,
        "removed_prerequisite": removed,
        "faults": faults,
    }


def audit_task_surface(task: dict[str, Any], parent: Path) -> list[str]:
    faults: list[str] = []
    with tempfile.TemporaryDirectory(prefix="theseus-p4s-task-surface-") as tmp:
        root = Path(tmp) / "parent"
        p2a.extract_source_archive(
            parent, root, str(task.get("source_archive_root") or "")
        )
        context = p2a.render_visible_context(root, task)
        maximum = int(
            p2a.mapping(task.get("candidate_visible_context")).get(
                "maximum_total_characters"
            ) or 0
        )
        if not context or len(context) > maximum:
            faults.append("candidate_visible_context_budget_invalid")
        symbols = p4s.semantic_scope_symbol_table(root, task)
        maximum_nodes = int(
            p2a.mapping(task.get("semantic_ir_contract")).get(
                "maximum_semantic_scope_nodes"
            ) or 0
        )
        if not symbols.get("source_digest") or not p2a.dicts(symbols.get("nodes")):
            faults.append("semantic_scope_symbol_table_empty")
        if len(p2a.dicts(symbols.get("nodes"))) > maximum_nodes:
            faults.append("semantic_scope_symbol_table_cap_exceeded")
    return faults


def exact_lowerer_inventory_count(task: dict[str, Any], parent: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="theseus-p4s-lowerer-inventory-") as tmp:
        root = Path(tmp) / "parent"
        p2a.extract_source_archive(
            parent, root, str(task.get("source_archive_root") or "")
        )
        table = p4.semantic_symbol_table(root, task)
    count = len(p2a.dicts(table.get("nodes")))
    if count <= 0 or count >= 1_000_000:
        raise ValueError("lowerer inventory count invalid")
    return count


if __name__ == "__main__":
    raise SystemExit(main())
