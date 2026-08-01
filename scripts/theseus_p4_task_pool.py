#!/usr/bin/env python3
"""Materialize and audit the sealed Theseus P4 cognitive-compilation task pool."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4_cognitive_compilation as p4  # noqa: E402


SOURCE_REGISTRY = ROOT / "configs" / "theseus_p4_task_sources.json"
INSTRUMENT = ROOT / "configs" / "theseus_p4_cognitive_compilation_instrument.json"
INSTRUMENT_AUDIT = ROOT / "reports" / "theseus_p4_cognitive_compilation_instrument_audit.json"
INSTRUMENT_COMMIT = "4ef352303a4d4c93288d1db3da659a874663c6d3"
EXPECTED_INSTRUMENT_SHA256 = "e2c65b0424e2fbfe109502f051ce6294ca2ef0f1573e9cd2d1e7cd65e8b55523"
EXPECTED_INSTRUMENT_AUDIT_SHA256 = "e4a37808fdc1588bb9d9d776b0779d09a65a9c54d86587abb2122f8080c82f84"
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "theseus_p4_online"
HIDDEN_TEST = FIXTURE_DIR / "theseus_p4_hidden_test.py"
VISIBLE_TEST = FIXTURE_DIR / "theseus_p4_visible_test.py"
P2_REPOSITORIES = {"python/typing", "urllib3/urllib3", "encode/starlette"}


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
    registry = read_json(SOURCE_REGISTRY)
    faults = audit_registry(registry)
    entries: list[dict[str, Any]] = []
    for source in dict_rows(registry.get("tasks")):
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
            audit = read_json(ROOT / entry["evaluator_audit"])
            if result.returncode != 0:
                faults.append(f"evaluator_audit_red:{entry['stem']}")
            entry.update({
                "evaluator_audit_sha256": sha256_file(ROOT / entry["evaluator_audit"]),
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
    state = (
        "SEALED_BEFORE_CANDIDATE_GENERATION"
        if run_audits and not faults and green == 10
        else "INVALID_NOT_SEALED"
    )
    pool = {
        "policy": "project_theseus_p4_cognitive_compilation_task_pool_v1",
        "state": state,
        "partition": "p4_cognitive_compilation_development",
        "sealed_utc": registry.get("sealed_utc"),
        "candidate_generation_opened": False,
        "selection_rule": registry.get("selection_rule"),
        "source_registry": relative(SOURCE_REGISTRY),
        "source_registry_sha256": sha256_file(SOURCE_REGISTRY),
        "instrument": relative(INSTRUMENT),
        "instrument_freeze_commit": INSTRUMENT_COMMIT,
        "instrument_sha256": sha256_file(INSTRUMENT),
        "instrument_audit": relative(INSTRUMENT_AUDIT),
        "instrument_audit_sha256": sha256_file(INSTRUMENT_AUDIT),
        "task_count": len(entries),
        "green_evaluator_audits": green,
        "distinct_repositories": len({row.get("repository") for row in dict_rows(registry.get("tasks"))}),
        "tasks": entries,
        "faults": sorted(set(faults)),
        "source_disjoint_from": {
            "P2": sorted(P2_REPOSITORIES),
            "P3": sorted(p3_repositories()),
            "D1": "reserved_fresh_source_disjoint_surface_not_acquired",
            "D2": "independent_neural_surface_not_acquired",
            "training": "all_P4_tasks_permanently_excluded"
        },
        "information_flow": {
            "natural_request_obligations_source_and_visible_feedback_candidate_visible": True,
            "upstream_target_hidden_test_and_oracle_candidate_visible": False,
            "task_selection_conditioned_on_learned_output": False,
            "task_selection_conditioned_on_static_compiler_output": False,
            "task_selection_conditioned_on_oracle_outcome": False,
            "route_labels_passed_to_scoring": False,
        },
        "hosted_reference": {
            "model": "gpt-5.6-luna",
            "effort": "xhigh",
            "transport_state": "DEFINED_TRANSPORT_NOT_BOUND",
            "same_sealed_pool_required": True,
            "local_and_hosted_denominators_separate": True,
            "P4_blocking": False,
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
            "A GREEN seal proves only that ten fresh licensed parents fail, their exact upstream "
            "targets pass, and evaluator-only Semantic IR oracles reach the passing behavior while "
            "identity and loss corruptions fail. It is not candidate, subsystem, D1, D2, serving, "
            "training, or ASI Stack support evidence."
        ),
    }
    write_json(ROOT / "configs" / "theseus_p4_task_pool.json", pool)
    return pool


def audit_registry(registry: dict[str, Any]) -> list[str]:
    faults: list[str] = []
    if registry.get("policy") != "project_theseus_p4_online_source_selection_v1":
        faults.append("source_registry_policy_invalid")
    if registry.get("state") != "FIXED_BEFORE_CANDIDATE_GENERATION":
        faults.append("source_registry_not_fixed")
    tasks = dict_rows(registry.get("tasks"))
    if len(tasks) != 10 or registry.get("task_count") != 10:
        faults.append("source_registry_task_count_invalid")
    if [row.get("campaign_index") for row in tasks] != list(range(1, 11)):
        faults.append("campaign_indexes_invalid")
    repositories = [str(row.get("repository") or "").lower() for row in tasks]
    if len(set(repositories)) != 10:
        faults.append("repositories_not_distinct")
    if set(repositories).intersection({value.lower() for value in P2_REPOSITORIES | p3_repositories()}):
        faults.append("prior_development_repository_overlap")
    boundaries = p2a.mapping(registry.get("boundaries"))
    zero_fields = (
        "local_model_calls", "hosted_model_calls", "deterministic_request_compiler_calls",
        "compiler_oracle_calls_before_selection",
    )
    if boundaries.get("candidate_generation_opened") is not False:
        faults.append("candidate_generation_already_opened")
    if any(int(boundaries.get(key) or 0) != 0 for key in zero_fields):
        faults.append("selection_conditioning_call_counter_nonzero")
    if boundaries.get("user_task_or_label_dependency") is not False:
        faults.append("user_dependency_not_excluded")
    if registry.get("instrument_freeze_commit") != INSTRUMENT_COMMIT:
        faults.append("instrument_commit_mismatch")
    try:
        frozen = datetime.fromisoformat(str(registry.get("instrument_frozen_utc") or "").replace("Z", "+00:00"))
        sealed = datetime.fromisoformat(str(registry.get("sealed_utc") or "").replace("Z", "+00:00"))
        if sealed <= frozen:
            faults.append("source_registry_not_after_instrument_freeze")
    except ValueError:
        faults.append("freeze_or_seal_time_invalid")
    if registry.get("instrument_sha256") != EXPECTED_INSTRUMENT_SHA256:
        faults.append("registry_instrument_digest_mismatch")
    if sha256_file(INSTRUMENT) != EXPECTED_INSTRUMENT_SHA256:
        faults.append("instrument_digest_mismatch")
    if sha256_file(INSTRUMENT_AUDIT) != EXPECTED_INSTRUMENT_AUDIT_SHA256:
        faults.append("instrument_audit_digest_mismatch")
    return faults


def materialize_task(source: dict[str, Any], registry: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    faults: list[str] = []
    index = int(source["campaign_index"])
    stem = str(source["stem"])
    suffix = stem.removeprefix("p4_")
    parent = FIXTURE_DIR / f"{stem}_parent.tar.gz"
    parent_upstream = FIXTURE_DIR / f"{stem}_parent_upstream.tar.gz"
    target = FIXTURE_DIR / f"{stem}_target.tar.gz"
    target_upstream = FIXTURE_DIR / f"{stem}_target_upstream.tar.gz"
    parent_sanitizer = ROOT / "reports" / f"theseus_{stem}_parent_archive_sanitization.json"
    target_sanitizer = ROOT / "reports" / f"theseus_{stem}_target_archive_sanitization.json"
    task_path = ROOT / "configs" / f"theseus_p4_task_{suffix}.json"
    evaluator_path = ROOT / "configs" / f"theseus_p4_evaluator_{suffix}.json"
    oracle_path = FIXTURE_DIR / f"{stem}_oracle.semantic_ir"
    audit_path = ROOT / "reports" / f"theseus_p4_{suffix}_evaluator_audit.json"
    required = [
        parent, parent_upstream, target, target_upstream, parent_sanitizer,
        target_sanitizer, HIDDEN_TEST, VISIBLE_TEST,
    ]
    for path in required:
        if not path.is_file():
            faults.append(f"missing_source_artifact:{relative(path)}")
    parent_report = read_json(parent_sanitizer)
    target_report = read_json(target_sanitizer)
    faults.extend(audit_sanitization_pair(
        parent_report, target_report, parent, parent_upstream, target, target_upstream,
        str(source["source_root"]), str(source["target_root"]), source,
    ))
    faults.extend(audit_archive(parent, str(source["source_root"]), source, "parent"))
    faults.extend(audit_archive(target, str(source["target_root"]), source, "target"))
    task = {
        "policy": p4.TASK_POLICY,
        "state": "SEALED_BEFORE_CANDIDATE_GENERATION",
        "campaign_index": index,
        "opaque_task_id": f"p4-cognitive-compilation-{index:02d}",
        "partition": "p4_cognitive_compilation_development",
        "family": "bounded_python_correctness_repair",
        "natural_request": source["natural_request"],
        "source_archive": relative(parent),
        "source_archive_sha256": sha256_file(parent),
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
            "upstream_archive": relative(parent_upstream),
            "upstream_archive_sha256": sha256_file(parent_upstream),
            "archive_sanitization_report": relative(parent_sanitizer),
            "archive_sanitization_report_sha256": sha256_file(parent_sanitizer),
        },
        "contamination_screen": {
            "public_benchmark": False,
            "previous_theseus_surface": False,
            "source_disjoint_from_p2_p3": True,
            "task_selected_after_p4_instrument_freeze": True,
            "task_selected_before_any_candidate_or_control": True,
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
            "command": python312_exec_command(VISIBLE_TEST, source["case"]),
            "timeout_seconds": 60,
            "answer_specific": True,
            "candidate_prompt_visibility": False,
        },
        "visible_feedback_map": [{
            "marker": source["visible_marker"],
            "obligation_ids": source["visible_obligation_ids"],
        }],
        "semantic_ir_contract": {
            "version": "theseus_semantic_ir_v1",
            "maximum_symbol_nodes": 80,
            "maximum_units": 8,
            "source_target_obligation_and_loss_identity_required": True,
        },
        "effect_authority": "disposable_snapshot_only",
        "maximum_inference": "One P4 development observation only; no D1, D2, serving, training, or ASI Stack support claim.",
    }
    write_json(task_path, task)
    faults.extend(audit_task_surface(task, parent))
    try:
        oracle_text = build_oracle(source, task, parent, target)
        oracle_path.write_text(oracle_text, encoding="utf-8")
    except (OSError, ValueError, p2a.InstrumentFault, p4.P4Fault) as exc:
        faults.append(f"oracle_materialization_fault:{type(exc).__name__}:{exc}")
    evaluator = {
        "policy": "project_theseus_p4_cognitive_compilation_evaluator_v1",
        "state": "SEALED_BEFORE_CANDIDATE_GENERATION",
        "task_manifest": relative(task_path),
        "task_manifest_sha256": sha256_file(task_path),
        "baseline_must_fail": True,
        "baseline_failure_markers": [f"P4_FAIL_{source['case']}"],
        "hidden_test_files": [{
            "source": relative(HIDDEN_TEST),
            "sha256": sha256_file(HIDDEN_TEST),
            "destination": "theseus_p4_hidden_test.py",
        }],
        "hidden_verifier": {
            "command": python312_exec_command(Path("theseus_p4_hidden_test.py"), source["case"]),
            "timeout_seconds": 60,
            "network": "forbidden",
        },
        "target_archive": relative(target),
        "target_archive_sha256": sha256_file(target),
        "target_archive_root": source["target_root"],
        "target_provenance": {
            "revision": source["target_revision"],
            "merge_revision": source["merge_revision"],
            "upstream_archive": relative(target_upstream),
            "upstream_archive_sha256": sha256_file(target_upstream),
            "archive_sanitization_report": relative(target_sanitizer),
            "archive_sanitization_report_sha256": sha256_file(target_sanitizer),
        },
        "target_must_pass": True,
        "oracle_ir_file": relative(oracle_path),
        "oracle_ir_sha256": sha256_file(oracle_path),
        "blindness": {
            "candidate_generation_may_read_this_manifest": False,
            "route_label_passed_to_scoring": False,
            "later_patch_candidate_visible": False,
            "hidden_test_candidate_visible": False,
            "oracle_candidate_visible": False,
            "candidate_emitted_integrity_flags_trusted": False,
        },
        "maximum_inference": "A GREEN audit establishes evaluator reachability and sensitivity for one sealed task only.",
    }
    write_json(evaluator_path, evaluator)
    entry = {
        "campaign_index": index,
        "stem": stem,
        "repository": source["repository"],
        "pull_request_url": f"https://github.com/{source['repository']}/pull/{source['pull_request']}",
        "license_spdx": source["license_spdx"],
        "parent_revision": source["parent_revision"],
        "target_revision": source["target_revision"],
        "task": relative(task_path),
        "task_sha256": sha256_file(task_path),
        "evaluator": relative(evaluator_path),
        "evaluator_sha256": sha256_file(evaluator_path),
        "oracle_ir": relative(oracle_path),
        "oracle_ir_sha256": sha256_file(oracle_path),
        "evaluator_audit": relative(audit_path),
    }
    return entry, faults


def build_oracle(source: dict[str, Any], task: dict[str, Any], parent: Path, target: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="theseus-p4-oracle-build-") as tmp:
        parent_root = Path(tmp) / "parent"
        target_root = Path(tmp) / "target"
        p2a.extract_source_archive(parent, parent_root, str(source["source_root"]))
        p2a.extract_source_archive(target, target_root, str(source["target_root"]))
        symbols = p4.semantic_symbol_table(parent_root, task)
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
            path = source["allowed_effect_paths"][0]
            parent_node = select_node(parent_root / path, p2a.mapping(unit["parent_selector"]))
            parent_segment = node_segment(parent_root / path, parent_node["node"])
            key = (
                path, parent_node["node"].lineno, parent_node["node"].end_lineno,
                p2a.sha256_text(parent_segment),
            )
            symbol = symbol_by_identity.get(key)
            if not symbol:
                raise ValueError(f"oracle parent node absent from frozen symbol table: {unit['id']}")
            selected_targets = [
                select_node(target_root / path, p2a.mapping(selector))
                for selector in unit["target_selectors"]
            ]
            replacement = join_target_segments(
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
            raise ValueError("oracle parse faults: " + ",".join(p2a.strings(parsed.get("faults"))))
        return text


def audit_task_surface(task: dict[str, Any], parent: Path) -> list[str]:
    faults: list[str] = []
    with tempfile.TemporaryDirectory(prefix="theseus-p4-task-surface-") as tmp:
        root = Path(tmp) / "parent"
        p2a.extract_source_archive(parent, root, str(task.get("source_archive_root") or ""))
        context = p2a.render_visible_context(root, task)
        maximum = int(p2a.mapping(task.get("candidate_visible_context")).get("maximum_total_characters") or 0)
        if not context or len(context) > maximum:
            faults.append("candidate_visible_context_budget_invalid")
        symbols = p4.semantic_symbol_table(root, task)
        maximum_nodes = int(p2a.mapping(task.get("semantic_ir_contract")).get("maximum_symbol_nodes") or 0)
        if not symbols.get("source_digest") or not p2a.dicts(symbols.get("nodes")):
            faults.append("semantic_symbol_table_empty")
        if len(p2a.dicts(symbols.get("nodes"))) > maximum_nodes:
            faults.append("semantic_symbol_table_cap_exceeded")
    return faults


def select_node(path: Path, selector: dict[str, Any]) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    matches: list[dict[str, Any]] = []

    def visit(node: ast.AST, scope: list[str]) -> None:
        next_scope = scope
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            qualified = ".".join([*scope, node.name])
            next_scope = [*scope, node.name]
        else:
            qualified = ".".join(scope)
        segment = node_segment(path, node) if hasattr(node, "lineno") else ""
        if (
            type(node).__name__ == str(selector.get("node_type") or "")
            and (not selector.get("qualified_name") or qualified == selector.get("qualified_name"))
            and (not selector.get("contains") or str(selector["contains"]) in segment)
            and (not selector.get("starts_with") or segment.strip().startswith(str(selector["starts_with"])))
            and (not selector.get("exact") or str(selector["exact"]) == segment.strip())
        ):
            matches.append({"node": node, "qualified_name": qualified, "segment": segment})
        for child in ast.iter_child_nodes(node):
            visit(child, next_scope)

    visit(tree, [])
    if len(matches) != 1:
        raise ValueError(f"selector expected one node, found {len(matches)}: {selector}")
    return matches[0]


def node_segment(path: Path, node: ast.AST) -> str:
    if not hasattr(node, "lineno"):
        return ""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = int(node.lineno)
    end = int(getattr(node, "end_lineno", start) or start)
    start_col = int(getattr(node, "col_offset", 0) or 0)
    end_col = int(getattr(node, "end_col_offset", len(lines[end - 1])) or len(lines[end - 1]))
    if start == end:
        return lines[start - 1][start_col:end_col]
    return "\n".join([
        lines[start - 1][start_col:], *lines[start:end - 1], lines[end - 1][:end_col],
    ])


def join_target_segments(path: Path, rows: list[dict[str, Any]], parent_col: int) -> str:
    values = [node_segment(path, row["node"]) for row in rows]
    if not values:
        raise ValueError("oracle target selector list empty")
    separator = "\n\n" if all(
        isinstance(row["node"], (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for row in rows
    ) else "\n"
    return values[0] + "".join(separator + (" " * parent_col) + value for value in values[1:])


def audit_sanitization_pair(
    parent_report: dict[str, Any], target_report: dict[str, Any],
    parent: Path, parent_upstream: Path, target: Path, target_upstream: Path,
    parent_root: str, target_root: str, source: dict[str, Any],
) -> list[str]:
    faults: list[str] = []
    pairs = [
        ("parent", parent_report, parent, parent_upstream, parent_root),
        ("target", target_report, target, target_upstream, target_root),
    ]
    required = set(p2a.strings(source.get("license_paths"))) | set(p2a.strings(source.get("allowed_effect_paths")))
    omission_sets: list[set[tuple[str, str, str]]] = []
    for label, report, output, upstream, root in pairs:
        if report.get("trigger_state") != "GREEN" or report.get("faults") != []:
            faults.append(f"{label}_sanitization_not_green")
        if report.get("source_archive_root") != root:
            faults.append(f"{label}_sanitization_root_mismatch")
        if p2a.mapping(report.get("input")).get("sha256") != sha256_file(upstream):
            faults.append(f"{label}_upstream_archive_digest_mismatch")
        if p2a.mapping(report.get("output")).get("sha256") != sha256_file(output):
            faults.append(f"{label}_normalized_archive_digest_mismatch")
        omissions: set[tuple[str, str, str]] = set()
        for row in dict_rows(report.get("omitted_members")):
            path = str(row.get("path") or "")
            prefix = root + "/"
            suffix = path[len(prefix):] if path.startswith(prefix) else path
            kind = str(row.get("kind") or "")
            link = str(row.get("linkname") or "")
            if kind not in {"symbolic_link", "hard_link"}:
                faults.append(f"{label}_unsafe_omission_type")
            if suffix in required or any(item.startswith(suffix + "/") for item in required):
                faults.append(f"{label}_omission_overlaps_required_path:{suffix}")
            omissions.add((suffix, kind, link))
        omission_sets.append(omissions)
    if omission_sets[0] != omission_sets[1]:
        faults.append("parent_target_omission_sets_differ")
    return faults


def audit_archive(archive: Path, root: str, source: dict[str, Any], label: str) -> list[str]:
    if not archive.is_file():
        return [f"{label}_archive_missing"]
    faults: list[str] = []
    required = {
        *(f"{root}/{path}" for path in p2a.strings(source.get("license_paths"))),
        *(f"{root}/{path}" for path in p2a.strings(source.get("allowed_effect_paths"))),
    }
    try:
        with tarfile.open(archive) as handle:
            members = handle.getmembers()
            names = {member.name.rstrip("/") for member in members}
            if any(member.issym() or member.islnk() or not (member.isdir() or member.isfile()) for member in members):
                faults.append(f"{label}_archive_has_nonregular_member")
            if any(not (name == root or name.startswith(root + "/")) for name in names):
                faults.append(f"{label}_archive_member_outside_root")
            for path in required:
                if path not in names:
                    faults.append(f"{label}_archive_required_path_missing:{path}")
    except tarfile.TarError:
        faults.append(f"{label}_archive_invalid")
    return faults


def p3_repositories() -> set[str]:
    registry = read_json(ROOT / "configs" / "theseus_p3_task_sources.json")
    return {str(row.get("repository") or "") for row in dict_rows(registry.get("tasks"))}


def python312_exec_command(script: Path, case: str) -> list[str]:
    code = (
        "import os; os.execv('/usr/local/bin/python3', "
        f"['python3', {str(script)!r}, {case!r}])"
    )
    return ["python3", "-c", code]


def dict_rows(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
