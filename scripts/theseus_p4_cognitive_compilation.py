#!/usr/bin/env python3
"""Frozen-model P4 instrument for typed Semantic IR and cognitive compilation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import theseus_assistant_p2a as p2a
import theseus_assistant_runtime as assistant_runtime
import theseus_local_inference_backend as local_backend


ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_p4_cognitive_compilation_run_v1"
INSTRUMENT_POLICY = "project_theseus_p4_cognitive_compilation_instrument_v1"
TASK_POLICY = "project_theseus_p4_cognitive_compilation_task_v1"
DIRECT = "direct_target_generation"
PLAN = "natural_language_plan_control"
SEMANTIC = "typed_semantic_ir_treatment"
STATIC = "deterministic_request_compiler_baseline"
ARMS = (DIRECT, PLAN, SEMANTIC)
IR_HEADER = "THESEUS_SEMANTIC_IR_V1"
IR_UNIT_RE = re.compile(
    r"UNIT ([A-Z][A-Z0-9_-]*) ([A-Z0-9_,]+) "
    r"(REPLACE|INSERT_BEFORE|INSERT_AFTER) ([^\n ]+) ([A-Z0-9-]+) ([a-f0-9]{64})\n"
    r"<<<\n(.*?)\n>>>",
    flags=re.DOTALL,
)
FORBIDDEN_TASK_FIELDS = p2a.FORBIDDEN_TASK_FIELDS


class P4Fault(ValueError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", default="configs/theseus_p4_cognitive_compilation_instrument.json")
    parser.add_argument("--task", default="")
    parser.add_argument("--out", default="reports/theseus_p4_cognitive_compilation_run.json")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    instrument_path = p2a.resolve(args.instrument)
    report = (
        audit_instrument(instrument_path)
        if args.audit_only
        else run_experiment(instrument_path, p2a.resolve(args.task))
    )
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps({
        "policy": report.get("policy"),
        "trigger_state": report.get("trigger_state"),
        "faults": report.get("faults"),
        "denominators": report.get("denominators"),
        "matched_set_ready": p2a.mapping(report.get("matched_set")).get("ready"),
    }, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") in {"GREEN", "YELLOW"} else 2


def audit_instrument(path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    value = p2a.read_json(path)
    faults: list[str] = []
    if value.get("policy") != INSTRUMENT_POLICY:
        faults.append("instrument_policy_invalid")
    if value.get("state") != "PROSPECTIVELY_BOUND_BEFORE_TASK_ACQUISITION":
        faults.append("instrument_not_prospectively_bound")
    p3_path = p2a.resolve(str(value.get("p3_terminal_disposition") or ""))
    if p2a.sha256_file(p3_path) != str(value.get("p3_terminal_disposition_sha256") or ""):
        faults.append("p3_terminal_disposition_digest_mismatch")
    p3 = p2a.read_json(p3_path)
    if p3.get("scientific_status") != "P3_COMPLETE_RESIDUAL_EXPOSED_NO_USEFULNESS_ROUTE_WINNER":
        faults.append("p3_terminal_state_invalid")
    if p2a.mapping(p3.get("next_stage")).get("selected_claim_id") != (
        "cognitive-compilation-and-semantic-ir.core"
    ):
        faults.append("p4_claim_selection_mismatch")
    base_path = p2a.resolve(str(value.get("base_local_instrument") or ""))
    if p2a.sha256_file(base_path) != str(value.get("base_local_instrument_sha256") or ""):
        faults.append("base_instrument_digest_mismatch")
    base_audit = p2a.audit_instrument(base_path)
    if base_audit.get("trigger_state") != "GREEN":
        faults.append("base_local_instrument_audit_red")
    harness = p2a.mapping(value.get("harness"))
    for name in ("candidate_runner", "blind_evaluator"):
        owner = p2a.resolve(str(harness.get(name) or ""))
        if p2a.sha256_file(owner) != str(harness.get(f"{name}_sha256") or ""):
            faults.append(f"{name}_digest_mismatch")
    contract = p2a.mapping(value.get("matched_arm_contract"))
    if tuple(p2a.strings(contract.get("arms"))) != ARMS:
        faults.append("arm_set_invalid")
    for key in (
        "same_frozen_model", "same_source_snapshot", "same_natural_request",
        "same_obligation_information", "same_symbol_table", "same_visible_verifier",
        "same_hidden_evaluator_after_seal", "same_two_model_calls",
        "same_generation_token_cap", "same_effect_sandbox",
    ):
        if contract.get(key) is not True:
            faults.append(f"matched_contract_false:{key}")
    budgets = p2a.mapping(value.get("budgets"))
    if int(budgets.get("model_calls_per_learned_arm") or 0) != 2:
        faults.append("learned_arm_call_budget_invalid")
    if int(budgets.get("persistent_model_loads_per_task_set") or 0) != 1:
        faults.append("persistent_load_budget_invalid")
    if int(budgets.get("maximum_generation_tokens_per_call") or 0) != 1536:
        faults.append("generation_budget_invalid")
    static_control = p2a.mapping(value.get("compiler_only_control"))
    if (
        static_control.get("label") != STATIC
        or static_control.get("prospectively_fixed") is not True
        or static_control.get("model_calls") != 0
        or static_control.get("target_or_oracle_visibility") is not False
        or static_control.get("explicit_abstention") is not True
    ):
        faults.append("deterministic_compiler_control_invalid")
    oracle = p2a.mapping(value.get("compiler_oracle_ceiling"))
    if (
        oracle.get("label") != "deterministic_compiler_oracle_ceiling"
        or oracle.get("location") != "evaluator_only"
        or oracle.get("candidate_generation_visibility") is not False
        or oracle.get("learned_generation_credit") is not False
    ):
        faults.append("compiler_oracle_ceiling_invalid")
    mechanics = mechanics_audit()
    if mechanics.get("trigger_state") != "GREEN":
        faults.append("semantic_ir_mechanics_red")
    return {
        "policy": "project_theseus_p4_cognitive_compilation_instrument_audit_v1",
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "instrument_sha256": p2a.sha256_file(path),
        "base_instrument_audit": base_audit,
        "semantic_ir_mechanics": mechanics,
        "runtime_ms": round((time.perf_counter() - started) * 1000, 3),
        "counters": p2a.zero_counters(),
    }


def audit_task(path: Path) -> dict[str, Any]:
    task = p2a.read_json(path)
    faults: list[str] = []
    if task.get("policy") != TASK_POLICY:
        faults.append("task_policy_invalid")
    if task.get("state") != "SEALED_BEFORE_CANDIDATE_GENERATION":
        faults.append("task_not_sealed")
    if FORBIDDEN_TASK_FIELDS.intersection(task):
        faults.append("answer_identifying_task_field_present")
    archive = p2a.resolve(str(task.get("source_archive") or ""))
    if p2a.sha256_file(archive) != str(task.get("source_archive_sha256") or ""):
        faults.append("source_archive_digest_mismatch")
    if not str(task.get("source_archive_root") or ""):
        faults.append("source_archive_root_missing")
    provenance = p2a.mapping(task.get("source_provenance"))
    if not all(str(provenance.get(key) or "") for key in ("url", "revision", "license_spdx")):
        faults.append("source_provenance_incomplete")
    obligations = p2a.dicts(task.get("obligations"))
    ids = [str(row.get("id") or "") for row in obligations]
    if len(obligations) < 3 or len(set(ids)) != len(ids) or any(not re.fullmatch(r"O[1-9][0-9]*", value) for value in ids):
        faults.append("obligation_set_invalid")
    if not any(row.get("kind") == "preserve" for row in obligations):
        faults.append("preservation_obligation_missing")
    if any(row.get("kind") not in {"require", "preserve", "non_goal"} for row in obligations):
        faults.append("obligation_kind_invalid")
    dependencies = p2a.dicts(task.get("obligation_dependencies"))
    if any(
        str(row.get("before") or "") not in ids
        or str(row.get("after") or "") not in ids
        or row.get("before") == row.get("after")
        for row in dependencies
    ):
        faults.append("obligation_dependency_invalid")
    if dependency_cycle(ids, dependencies):
        faults.append("obligation_dependency_cycle")
    if not p2a.strings(task.get("allowed_effect_paths")):
        faults.append("allowed_effect_paths_missing")
    context = p2a.mapping(task.get("candidate_visible_context"))
    if not p2a.dicts(context.get("reads")):
        faults.append("candidate_visible_reads_missing")
    verifier = p2a.mapping(task.get("visible_verifier"))
    command = p2a.strings(verifier.get("command"))
    if not command or command[0] not in {"python3", "pytest"}:
        faults.append("visible_verifier_invalid")
    maps = p2a.dicts(task.get("visible_feedback_map"))
    if not maps or any(
        not str(row.get("marker") or "")
        or not set(p2a.strings(row.get("obligation_ids"))).issubset(set(ids))
        for row in maps
    ):
        faults.append("visible_feedback_map_invalid")
    return {
        "policy": "project_theseus_p4_cognitive_compilation_task_audit_v1",
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "task_sha256": p2a.sha256_file(path),
        "source_archive_sha256": p2a.sha256_file(archive),
        "counters": p2a.zero_counters(),
    }


def run_experiment(
    instrument_path: Path,
    task_path: Path,
    *,
    session_factory: Callable[..., Any] = local_backend.PersistentLocalInferenceSession,
) -> dict[str, Any]:
    started = time.perf_counter()
    instrument_audit = audit_instrument(instrument_path)
    task_audit = audit_task(task_path)
    if instrument_audit.get("trigger_state") != "GREEN" or task_audit.get("trigger_state") != "GREEN":
        return {
            "policy": POLICY, "created_utc": p2a.now(), "trigger_state": "RED",
            "faults": ["instrument_or_task_audit_red"],
            "instrument_audit": instrument_audit, "task_audit": task_audit,
            "counters": p2a.zero_counters(),
        }
    instrument = p2a.read_json(instrument_path)
    task = p2a.read_json(task_path)
    base = p2a.read_json(p2a.resolve(str(instrument.get("base_local_instrument") or "")))
    runtime_config = assistant_runtime.load_runtime_config(
        p2a.resolve(str(base.get("runtime_config") or ""))
    )
    runtime_binding = p2a.mapping(base.get("runtime_binding")) or p2a.mapping(
        runtime_config.get("local_inference")
    )
    frozen = p2a.mapping(base.get("frozen_model"))
    maximum = int(p2a.mapping(instrument.get("budgets")).get("maximum_generation_tokens_per_call") or 0)
    session = session_factory(
        worker_config_path=p2a.resolve(str(runtime_binding.get("worker_config") or "")),
        runtime_preflight_path=p2a.resolve(str(runtime_binding.get("runtime_preflight") or "")),
        maximum_tokens=maximum,
        required_repo_id=str(frozen.get("repo_id") or ""),
        required_revision=str(frozen.get("revision") or ""),
        required_snapshot_manifest_sha256=str(frozen.get("snapshot_manifest_sha256") or ""),
        session_id=f"p4-{p2a.safe_slug(str(task.get('opaque_task_id') or 'task'))}",
    )
    if not session.ready:
        return {
            "policy": POLICY, "created_utc": p2a.now(), "trigger_state": "RED",
            "faults": ["persistent_backend_not_ready", *list(session.faults)],
            "instrument_audit": instrument_audit, "task_audit": task_audit,
            "counters": p2a.zero_counters(),
        }
    index = int(task.get("campaign_index") or 0)
    order = arm_order(index)
    attempts: list[dict[str, Any]] = []
    with assistant_runtime.bind_local_inference_runner(session.runtime_runner):
        for arm in order:
            attempts.append(run_arm(arm, instrument, task, session))
    static_control = run_deterministic_compiler_control(task)
    identities = {str(row.get("model_identity_sha256") or "") for row in attempts}
    matched = {
        "ready": (
            len(identities) == 1 and "" not in identities
            and session.model_load_count == 1 and session.inference_calls == 6
            and all(len(p2a.dicts(row.get("runtime_calls"))) == 2 for row in attempts)
            and all(
                all(p2a.mapping(route).get("ready") is True for route in p2a.dicts(row.get("route_integrity_rounds")))
                for row in attempts
            )
            and all(
                all(
                    p2a.mapping(route).get("execution_mode") == p2a.route_integrity.DIRECT_MODE
                    for route in p2a.dicts(row.get("route_integrity_rounds"))
                )
                for row in attempts
            )
        ),
        "same_model_identity": len(identities) == 1 and "" not in identities,
        "persistent_model_load_count": session.model_load_count,
        "persistent_inference_calls": session.inference_calls,
        "two_calls_per_learned_arm": all(
            len(p2a.dicts(row.get("runtime_calls"))) == 2 for row in attempts
        ),
        "all_learned_calls_direct_local": all(
            all(
                p2a.mapping(route).get("execution_mode") == p2a.route_integrity.DIRECT_MODE
                for route in p2a.dicts(row.get("route_integrity_rounds"))
            )
            for row in attempts
        ),
    }
    parseable = sum(int(row.get("parseable_candidate") is True) for row in attempts)
    faults = [] if matched["ready"] else ["matched_set_or_persistence_invalid"]
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "RED" if faults else ("GREEN" if parseable else "YELLOW"),
        "faults": faults,
        "scope": "P4 cognitive-compilation development only; no D1, D2, serving, or book-support claim",
        "instrument_sha256": p2a.sha256_file(instrument_path),
        "task_sha256": p2a.sha256_file(task_path),
        "instrument_audit": instrument_audit,
        "task_audit": task_audit,
        "actual_arm_order": list(order),
        "attempts": attempts,
        "deterministic_compiler_control": static_control,
        "matched_set": matched,
        "denominators": {
            "tasks": 1, "learned_arms": 3, "model_calls": session.inference_calls,
            "model_loads": session.model_load_count, "parseable_candidates": parseable,
            "deterministic_compiler_candidates": int(static_control.get("parseable_candidate") is True),
        },
        "counters": p2a.zero_counters(),
        "runtime_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def arm_order(index: int) -> tuple[str, str, str]:
    if index < 1:
        raise P4Fault("campaign_index_invalid")
    rotations = (
        (DIRECT, PLAN, SEMANTIC),
        (PLAN, SEMANTIC, DIRECT),
        (SEMANTIC, DIRECT, PLAN),
    )
    return rotations[(index - 1) % 3]


def run_arm(arm: str, instrument: dict[str, Any], task: dict[str, Any], session: Any) -> dict[str, Any]:
    task_id = p2a.safe_slug(str(task.get("opaque_task_id") or "task"))
    maximum = int(p2a.mapping(instrument.get("budgets")).get("maximum_generation_tokens_per_call") or 0)
    protocol = p2a.mapping(p2a.read_json(p2a.resolve(str(instrument.get("base_local_instrument")))) .get("candidate_protocol"))
    with tempfile.TemporaryDirectory(prefix=f"theseus-p4-{arm}-") as tmp:
        original = Path(tmp) / "original"
        p2a.extract_source_archive(
            p2a.resolve(str(task.get("source_archive") or "")), original,
            str(task.get("source_archive_root") or ""),
        )
        baseline = p2a.inventory(original)
        common = render_common_context(original, task)
        first_prompt = render_arm_prompt(arm, task, common, protocol)
        first = p2a.runtime_call(
            p2a.route_integrity.DIRECT_MODE, f"p4_{task_id}_{arm}", 1,
            first_prompt, maximum, str(p2a.read_json(p2a.resolve(str(instrument.get("base_local_instrument")))).get("runtime_config") or ""),
        )
        first_candidate = parse_arm_output(arm, first["assistant_text"], task, original, protocol)
        first_verification = verify_provisional(original, baseline, task, first_candidate)
        implicated = implicated_obligations(task, first_candidate, first_verification)
        repair_prompt = render_final_prompt(
            first_prompt, first["assistant_text"], first_candidate, first_verification, implicated
        )
        second = p2a.runtime_call(
            p2a.route_integrity.DIRECT_MODE, f"p4_{task_id}_{arm}", 2,
            repair_prompt, maximum, str(p2a.read_json(p2a.resolve(str(instrument.get("base_local_instrument")))).get("runtime_config") or ""),
        )
        final_candidate = parse_arm_output(arm, second["assistant_text"], task, original, protocol)
        locality_faults: list[str] = []
        if arm == SEMANTIC and not first_candidate["faults"] and not final_candidate["faults"]:
            locality_faults = repair_locality_faults(
                p2a.dicts(first_candidate.get("units")),
                p2a.dicts(final_candidate.get("units")),
                implicated,
            )
        final_candidate["faults"] = sorted(set(p2a.strings(final_candidate.get("faults")) + locality_faults))
        final_verification = verify_provisional(original, baseline, task, final_candidate)
        actions = p2a.dicts(final_candidate.get("actions"))
        changed = p2a.strings(final_verification.get("changed_paths"))
        authorized = bool(changed) and set(changed).issubset(set(p2a.strings(task.get("allowed_effect_paths"))))
        parseable = bool(actions) and not final_candidate["faults"] and not final_verification["apply_faults"] and authorized
        candidate_payload = {
            "protocol": (
                "theseus_semantic_ir_v1" if arm == SEMANTIC else "theseus_line_edit_v1"
            ),
            "actions": actions,
            "changed_paths": changed,
            "final_inventory_sha256": final_verification.get("final_inventory_sha256"),
            "visible_verifier": final_verification.get("visible_verifier"),
            "semantic_receipt": final_candidate.get("semantic_receipt", {}),
        }
        seal = {
            "candidate_output_sha256": p2a.stable_hash(candidate_payload),
            "task_sha256": p2a.stable_hash({
                "opaque_task_id": task.get("opaque_task_id"),
                "natural_request": task.get("natural_request"),
                "source_archive_sha256": task.get("source_archive_sha256"),
            }),
            "sealed_before_hidden_evaluation": True,
        } if parseable else {}
        routes = [
            p2a.mapping(first["runtime_report"]).get("route_integrity"),
            p2a.mapping(second["runtime_report"]).get("route_integrity"),
        ]
        return {
            "arm_id": arm,
            "parseable_candidate": parseable,
            "parse_faults": final_candidate["faults"],
            "apply_faults": final_verification["apply_faults"],
            "candidate": candidate_payload if parseable else {},
            "candidate_seal": seal,
            "runtime_calls": [first["receipt"], second["receipt"]],
            "route_integrity_rounds": routes,
            "provisional": {
                "parse_faults": first_candidate["faults"],
                "apply_faults": first_verification["apply_faults"],
                "visible_verifier": first_verification.get("visible_verifier"),
                "implicated_obligation_ids": sorted(implicated),
            },
            "final_visible_verifier_passed": p2a.mapping(final_verification.get("visible_verifier")).get("passed") is True,
            "model_calls": 2,
            "model_identity_sha256": p2a.mapping(
                p2a.mapping(p2a.mapping(second.get("runtime_report")).get("route_integrity")).get("pair_contract")
            ).get("model_identity_sha256"),
        }


def render_common_context(root: Path, task: dict[str, Any]) -> str:
    source = p2a.render_visible_context(root, task)
    symbols = semantic_symbol_table(root, task)
    obligation_lines = [
        f"{row['id']} {str(row['kind']).upper()}: {row['text']}"
        for row in p2a.dicts(task.get("obligations"))
    ]
    dependency_lines = [
        f"{row['before']} -> {row['after']}"
        for row in p2a.dicts(task.get("obligation_dependencies"))
    ] or ["none"]
    symbol_lines = [
        f"{row['id']} {row['sha256']} {row['node_type']} {row['path']} "
        f"{row['start_line']}:{row['start_col']}-{row['end_line']}:{row['end_col']} {row['label']}"
        for row in symbols["nodes"]
    ]
    return (
        "[INFORMATION_MATCHED_OBLIGATIONS]\n" + "\n".join(obligation_lines)
        + "\n[OBLIGATION_DEPENDENCIES]\n" + "\n".join(dependency_lines)
        + f"\n[SEMANTIC_SOURCE_DIGEST]\n{symbols['source_digest']}"
        + "\n[INFORMATION_MATCHED_SYMBOL_TABLE]\n" + "\n".join(symbol_lines)
        + "\n[CANDIDATE_VISIBLE_REPOSITORY_CONTEXT]\n" + source
    )


def run_deterministic_compiler_control(task: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="theseus-p4-static-compiler-") as tmp:
        root = Path(tmp) / "source"
        p2a.extract_source_archive(
            p2a.resolve(str(task.get("source_archive") or "")), root,
            str(task.get("source_archive_root") or ""),
        )
        baseline = p2a.inventory(root)
        actions, compiler_faults, rule = deterministic_request_compile(task, root)
        apply_faults = p2a.apply_actions(root, actions) if actions else []
        changed = p2a.changed_paths(baseline, p2a.inventory(root)) if not apply_faults else []
        authorized = bool(changed) and set(changed).issubset(set(p2a.strings(task.get("allowed_effect_paths"))))
        visible = p2a.run_visible_verifier(root, task) if actions and not apply_faults else {}
        parseable = bool(actions) and not compiler_faults and not apply_faults and authorized
        candidate = {
            "protocol": "deterministic_request_compiler_v1",
            "actions": actions,
            "changed_paths": changed,
            "final_inventory_sha256": p2a.stable_hash(p2a.inventory(root)),
            "visible_verifier": visible,
            "semantic_receipt": {
                "model_generated_ir": False,
                "deterministic_rule": rule,
                "learned_generation_credit": False,
            },
        }
        seal = {
            "candidate_output_sha256": p2a.stable_hash(candidate),
            "task_sha256": p2a.stable_hash({
                "opaque_task_id": task.get("opaque_task_id"),
                "natural_request": task.get("natural_request"),
                "source_archive_sha256": task.get("source_archive_sha256"),
            }),
            "sealed_before_hidden_evaluation": True,
            "deterministic_compiler": True,
        } if parseable else {}
        return {
            "arm_id": STATIC,
            "parseable_candidate": parseable,
            "compiler_faults": compiler_faults,
            "apply_faults": apply_faults,
            "candidate": candidate if parseable else {},
            "candidate_seal": seal,
            "visible_verifier_passed": visible.get("passed") is True,
            "model_calls": 0,
            "learned_generation_credit": 0,
            "abstained": int(not parseable),
        }


def deterministic_request_compile(
    task: dict[str, Any], root: Path
) -> tuple[list[dict[str, Any]], list[str], str]:
    request = str(task.get("natural_request") or "")
    allowed = p2a.strings(task.get("allowed_effect_paths"))
    replace = re.search(
        r"replace(?: the)?(?: exact)?(?: literal)? `([^`]+)` with `([^`]+)`",
        request,
        flags=re.IGNORECASE,
    )
    if replace:
        old, new = replace.groups()
        matches: list[tuple[str, int, str]] = []
        for path in allowed:
            for number, line in enumerate(
                p2a.checked_source_path(root, path).read_text(encoding="utf-8").splitlines(), 1
            ):
                if old in line:
                    matches.append((path, number, line))
        if len(matches) != 1 or matches[0][2].count(old) != 1:
            return [], ["deterministic_literal_replacement_ambiguous"], "exact_literal_replace"
        path, number, line = matches[0]
        return ([{
            "op": "REPLACE", "path": path, "start_line": number, "end_line": number,
            "replacement": line.replace(old, new, 1),
        }], [], "exact_literal_replace")
    collection = re.search(
        r"(add|remove) `([^`]+)` (?:to|from) `([A-Za-z_][A-Za-z0-9_]*)`",
        request,
        flags=re.IGNORECASE,
    )
    if collection:
        operation, item, symbol_name = collection.groups()
        matches: list[tuple[str, ast.AST, str]] = []
        for path in allowed:
            target = p2a.checked_source_path(root, path)
            text = target.read_text(encoding="utf-8")
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(target_node, ast.Name) and target_node.id == symbol_name for target_node in targets):
                    value = node.value
                    if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
                        matches.append((path, value, text))
        if len(matches) != 1:
            return [], ["deterministic_collection_target_ambiguous"], "collection_literal_edit"
        path, value, text = matches[0]
        elements = [ast.get_source_segment(text, element) or "" for element in value.elts]
        if operation.lower() == "add":
            if item in elements:
                return [], ["deterministic_collection_item_already_present"], "collection_literal_edit"
            elements.append(item)
        else:
            if elements.count(item) != 1:
                return [], ["deterministic_collection_item_not_unique"], "collection_literal_edit"
            elements.remove(item)
        opening, closing = (
            ("[", "]") if isinstance(value, ast.List)
            else ("(", ")") if isinstance(value, ast.Tuple)
            else ("{", "}")
        )
        replacement = opening + ", ".join(elements) + ("," if isinstance(value, ast.Tuple) and len(elements) == 1 else "") + closing
        lines = text.splitlines()
        start, end = int(value.lineno), int(value.end_lineno or value.lineno)
        original = lines[start - 1:end]
        prefix = original[0][: int(value.col_offset)]
        suffix = original[-1][int(value.end_col_offset or len(original[-1])) :]
        return ([{
            "op": "REPLACE", "path": path, "start_line": start, "end_line": end,
            "replacement": prefix + replacement + suffix,
        }], [], "collection_literal_edit")
    return [], ["deterministic_request_pattern_unsupported"], "abstain"


def render_arm_prompt(arm: str, task: dict[str, Any], common: str, protocol: dict[str, Any]) -> str:
    request = str(task.get("natural_request") or "")
    if arm == SEMANTIC:
        instruction = (
            "Return only typed Semantic IR in this grammar:\n"
            "THESEUS_SEMANTIC_IR_V1\n"
            "SOURCE <semantic-source-digest>\n"
            "OBLIGATIONS <comma-separated exact obligation ids>\n"
            "UNIT <unit-id> <comma-separated obligation ids> <REPLACE|INSERT_BEFORE|INSERT_AFTER> <path> <node-id> <node-sha256>\n"
            "<<<\n<replacement source>\n>>>\n"
            "LOSS <NONE|comma-separated unresolved obligation ids>\nEND\n"
            "Every obligation must be listed and referenced by a unit. Use stable node ids and hashes from the symbol table."
        )
    elif arm == PLAN:
        instruction = (
            "Write a short non-executing plan, then a complete typed edit. Return only:\n"
            "THESEUS_PLAN_V1\nPLAN\n<plain plan>\nTARGET\n"
            f"{protocol.get('grammar')}"
        )
    else:
        instruction = (
            "Return only a complete typed edit in this exact grammar:\n"
            f"{protocol.get('grammar')}"
        )
    return (
        f"Implement this repository task: {request}\n\n{instruction}\n"
        "The symbol table and obligations are identical across learned arms. Do not return a Git diff, JSON, or commentary.\n\n"
        + common
    )


def render_final_prompt(
    original: str,
    first_output: str,
    first_candidate: dict[str, Any],
    verification: dict[str, Any],
    implicated: set[str],
) -> str:
    feedback = {
        "parse_or_lower_faults": p2a.strings(first_candidate.get("faults")),
        "apply_faults": p2a.strings(verification.get("apply_faults")),
        "visible_verifier_returncode": p2a.mapping(verification.get("visible_verifier")).get("returncode"),
        "visible_verifier_stdout_tail": str(p2a.mapping(verification.get("visible_verifier")).get("stdout_tail") or "")[-1000:],
        "visible_verifier_stderr_tail": str(p2a.mapping(verification.get("visible_verifier")).get("stderr_tail") or "")[-1000:],
        "dependency_local_repair_obligation_ids": sorted(implicated),
    }
    return (
        original
        + "\n\n[PROVISIONAL_OUTPUT]\n" + first_output[-12000:]
        + "\n\n[ACTUAL_TARGET_VALIDATION_FEEDBACK]\n" + json.dumps(feedback, sort_keys=True)
        + "\n\nReturn one complete final candidate against the ORIGINAL snapshot in the same arm grammar. "
        "Do not emit a delta against the provisional candidate."
    )


def parse_arm_output(
    arm: str, text: str, task: dict[str, Any], root: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    if arm == SEMANTIC:
        return parse_semantic_ir(text, task, root)
    raw = str(text or "")
    if arm == PLAN:
        marker = raw.find(p2a.ACTION_HEADER)
        raw = raw[marker:] if marker >= 0 else raw
    actions, faults = p2a.parse_actions(raw, task, protocol)
    return {"actions": actions, "faults": faults, "units": [], "semantic_receipt": {}}


def parse_semantic_ir(text: str, task: dict[str, Any], root: Path) -> dict[str, Any]:
    raw = str(text or "").strip()
    fenced = re.search(r"```(?:text)?\s*(.*?)\s*```", raw, flags=re.DOTALL)
    raw = (fenced.group(1) if fenced else raw).strip()
    faults: list[str] = []
    if not raw.startswith(IR_HEADER) or not raw.endswith("END"):
        return {"actions": [], "faults": ["semantic_ir_envelope_invalid"], "units": [], "semantic_receipt": {}}
    source_match = re.search(r"^SOURCE ([a-f0-9]{64})$", raw, flags=re.MULTILINE)
    obligations_match = re.search(r"^OBLIGATIONS ([A-Z0-9_,]+)$", raw, flags=re.MULTILINE)
    loss_match = re.search(r"^LOSS ([A-Z0-9_,]+|NONE)$", raw, flags=re.MULTILINE)
    symbols = semantic_symbol_table(root, task)
    if not source_match or source_match.group(1) != symbols["source_digest"]:
        faults.append("semantic_source_identity_invalid")
    expected_ids = [str(row.get("id") or "") for row in p2a.dicts(task.get("obligations"))]
    declared = obligations_match.group(1).split(",") if obligations_match else []
    if declared != expected_ids:
        faults.append("semantic_obligation_identity_or_order_invalid")
    loss = [] if loss_match and loss_match.group(1) == "NONE" else (
        loss_match.group(1).split(",") if loss_match else []
    )
    if not loss_match:
        faults.append("semantic_loss_record_missing")
    elif loss:
        if not set(loss).issubset(set(expected_ids)):
            faults.append("semantic_loss_identity_invalid")
        faults.append("semantic_loss_unresolved")
    symbol_map = {row["id"]: row for row in symbols["nodes"]}
    units: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    covered: set[str] = set()
    ranges: dict[str, list[tuple[int, int]]] = {}
    for match in IR_UNIT_RE.finditer(raw):
        unit_id, refs_raw, operation, path, node_id, node_hash, replacement = match.groups()
        refs = refs_raw.split(",")
        if not set(refs).issubset(set(expected_ids)) or not refs:
            faults.append("semantic_unit_obligation_reference_invalid")
        covered.update(refs)
        symbol = p2a.mapping(symbol_map.get(node_id))
        if not symbol or symbol.get("path") != path or symbol.get("sha256") != node_hash:
            faults.append("semantic_target_identity_invalid")
            continue
        required = dependency_ancestors(refs, p2a.dicts(task.get("obligation_dependencies")))
        if not required.issubset(set(refs)):
            faults.append("semantic_unit_dependency_not_closed")
        start, end = int(symbol["start_line"]), int(symbol["end_line"])
        if any(not (end < prior_start or start > prior_end) for prior_start, prior_end in ranges.setdefault(path, [])):
            faults.append("semantic_units_overlap")
        ranges[path].append((start, end))
        source_lines = p2a.checked_source_path(root, path).read_text(encoding="utf-8").splitlines()
        original = source_lines[start - 1:end]
        lowered = replacement
        if operation == "REPLACE":
            replacement_lines = replacement.splitlines() or [""]
            prefix = original[0][: int(symbol.get("start_col") or 0)]
            suffix = original[-1][int(symbol.get("end_col") or len(original[-1])) :]
            if len(replacement_lines) == 1:
                lowered = prefix + replacement_lines[0] + suffix
            else:
                lowered = "\n".join(
                    [prefix + replacement_lines[0], *replacement_lines[1:-1], replacement_lines[-1] + suffix]
                )
        if operation == "INSERT_BEFORE":
            lowered = replacement + "\n" + "\n".join(original)
        elif operation == "INSERT_AFTER":
            lowered = "\n".join(original) + "\n" + replacement
        action = {"op": "REPLACE", "path": path, "start_line": start, "end_line": end, "replacement": lowered}
        actions.append(action)
        units.append({
            "id": unit_id, "obligation_ids": refs, "operation": operation,
            "path": path, "node_id": node_id, "node_sha256": node_hash,
            "replacement_sha256": p2a.sha256_text(replacement), "action": action,
        })
    if not units:
        faults.append("semantic_units_missing")
    if len({row["id"] for row in units}) != len(units):
        faults.append("semantic_unit_identity_duplicate")
    if covered != set(expected_ids):
        faults.append("semantic_obligation_coverage_incomplete")
    allowed = set(p2a.strings(task.get("allowed_effect_paths")))
    if any(row["path"] not in allowed for row in actions):
        faults.append("semantic_effect_path_unauthorized")
    recognized_spans = [match.span() for match in IR_UNIT_RE.finditer(raw)]
    scrubbed = raw
    for start, end in reversed(recognized_spans):
        scrubbed = scrubbed[:start] + "" + scrubbed[end:]
    scrubbed = re.sub(r"^(THESEUS_SEMANTIC_IR_V1|SOURCE .+|OBLIGATIONS .+|LOSS .+|END)\s*$", "", scrubbed, flags=re.MULTILINE).strip()
    if scrubbed:
        faults.append("semantic_ir_unparsed_text")
    receipt = {
        "semantic_source_digest": symbols["source_digest"],
        "obligation_ids": expected_ids,
        "unit_ids": [row["id"] for row in units],
        "loss_obligation_ids": loss,
        "lowered_action_sha256": p2a.stable_hash(actions),
        "model_generated_ir": True,
        "deterministic_lowerer": True,
    }
    return {
        "actions": actions if not faults else [],
        "faults": sorted(set(faults)),
        "units": units,
        "semantic_receipt": receipt,
    }


def semantic_symbol_table(root: Path, task: dict[str, Any]) -> dict[str, Any]:
    read_ranges: dict[str, list[tuple[int, int]]] = {}
    for row in p2a.dicts(p2a.mapping(task.get("candidate_visible_context")).get("reads")):
        read_ranges.setdefault(str(row.get("path") or ""), []).append(
            (int(row.get("start_line") or 1), int(row.get("end_line") or 10**9))
        )
    nodes: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    node_types = (
        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign,
        ast.Return, ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Expr,
        ast.Call, ast.Compare, ast.List, ast.Tuple, ast.Dict, ast.Set,
    )
    for path in p2a.strings(task.get("allowed_effect_paths")):
        target = p2a.checked_source_path(root, path)
        text = target.read_text(encoding="utf-8")
        source_hashes[path] = p2a.sha256_text(text)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        lines = text.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, node_types) or not hasattr(node, "lineno"):
                continue
            start = int(node.lineno)
            end = int(getattr(node, "end_lineno", start) or start)
            if read_ranges.get(path) and not any(
                start <= range_end and end >= range_start
                for range_start, range_end in read_ranges[path]
            ):
                continue
            start_col = int(getattr(node, "col_offset", 0) or 0)
            end_col = int(getattr(node, "end_col_offset", len(lines[end - 1])) or len(lines[end - 1]))
            if start == end:
                segment = lines[start - 1][start_col:end_col]
            else:
                segment = "\n".join(
                    [lines[start - 1][start_col:], *lines[start:end - 1], lines[end - 1][:end_col]]
                )
            node_hash = p2a.sha256_text(segment)
            node_id = "N-" + p2a.sha256_text(
                f"{path}|{type(node).__name__}|{start}|{end}|{node_hash}"
            )[:16].upper()
            label = str(getattr(node, "name", "")) or compact_ast_label(node)
            nodes.append({
                "id": node_id, "sha256": node_hash, "node_type": type(node).__name__,
                "path": path, "start_line": start, "end_line": end,
                "start_col": start_col, "end_col": end_col,
                "label": re.sub(r"\s+", " ", label)[:100],
            })
    nodes.sort(key=lambda row: (row["path"], row["start_line"], row["end_line"], row["node_type"]))
    cap = int(p2a.mapping(task.get("semantic_ir_contract")).get("maximum_symbol_nodes") or 80)
    nodes = nodes[:cap]
    return {
        "source_digest": p2a.stable_hash(source_hashes),
        "allowed_file_sha256": source_hashes,
        "nodes": nodes,
    }


def compact_ast_label(node: ast.AST) -> str:
    value = ast.dump(node, annotate_fields=False, include_attributes=False)
    return value[:120]


def verify_provisional(
    original: Path, baseline: dict[str, str], task: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="theseus-p4-verify-") as tmp:
        root = Path(tmp) / "candidate"
        p2a.extract_source_archive(
            p2a.resolve(str(task.get("source_archive") or "")), root,
            str(task.get("source_archive_root") or ""),
        )
        actions = p2a.dicts(candidate.get("actions"))
        apply_faults = p2a.apply_actions(root, actions) if actions else []
        changed = p2a.changed_paths(baseline, p2a.inventory(root)) if not apply_faults else []
        verifier = p2a.run_visible_verifier(root, task) if actions and not apply_faults else {}
        return {
            "apply_faults": apply_faults,
            "changed_paths": changed,
            "visible_verifier": verifier,
            "final_inventory_sha256": p2a.stable_hash(p2a.inventory(root)),
        }


def implicated_obligations(
    task: dict[str, Any], candidate: dict[str, Any], verification: dict[str, Any]
) -> set[str]:
    all_ids = {str(row.get("id") or "") for row in p2a.dicts(task.get("obligations"))}
    if p2a.strings(candidate.get("faults")) or p2a.strings(verification.get("apply_faults")):
        return all_ids
    visible = p2a.mapping(verification.get("visible_verifier"))
    if visible.get("passed") is True:
        return all_ids
    text = str(visible.get("stdout_tail") or "") + str(visible.get("stderr_tail") or "")
    selected: set[str] = set()
    for row in p2a.dicts(task.get("visible_feedback_map")):
        if str(row.get("marker") or "") in text:
            selected.update(p2a.strings(row.get("obligation_ids")))
    if not selected:
        return all_ids
    dependencies = p2a.dicts(task.get("obligation_dependencies"))
    changed = True
    while changed:
        changed = False
        for row in dependencies:
            before, after = str(row.get("before")), str(row.get("after"))
            if before in selected or after in selected:
                prior = len(selected)
                selected.update((before, after))
                changed |= len(selected) != prior
    return selected


def repair_locality_faults(
    first: list[dict[str, Any]], final: list[dict[str, Any]], allowed_obligations: set[str]
) -> list[str]:
    before = {str(row.get("id") or ""): row for row in first}
    after = {str(row.get("id") or ""): row for row in final}
    faults: list[str] = []
    before_targets = {
        (row.get("operation"), row.get("path"), row.get("node_id")): row.get("id")
        for row in first
    }
    after_targets = {
        (row.get("operation"), row.get("path"), row.get("node_id")): row.get("id")
        for row in final
    }
    if any(
        target in after_targets and after_targets[target] != unit_id
        for target, unit_id in before_targets.items()
    ):
        faults.append("semantic_repair_unit_identity_churn")
    for unit_id in set(before) | set(after):
        if p2a.stable_hash(before.get(unit_id)) == p2a.stable_hash(after.get(unit_id)):
            continue
        refs = set(p2a.strings(p2a.mapping(after.get(unit_id) or before.get(unit_id)).get("obligation_ids")))
        if not refs or not refs.issubset(allowed_obligations):
            faults.append("semantic_repair_not_dependency_local")
    return sorted(set(faults))


def dependency_ancestors(refs: list[str], dependencies: list[dict[str, Any]]) -> set[str]:
    required = set(refs)
    changed = True
    while changed:
        changed = False
        for row in dependencies:
            before, after = str(row.get("before")), str(row.get("after"))
            if after in required and before not in required:
                required.add(before)
                changed = True
    return required


def dependency_cycle(ids: list[str], dependencies: list[dict[str, Any]]) -> bool:
    edges: dict[str, list[str]] = {value: [] for value in ids}
    for row in dependencies:
        edges.setdefault(str(row.get("before")), []).append(str(row.get("after")))
    visiting: set[str] = set()
    visited: set[str] = set()
    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(visit(child) for child in edges.get(node, [])):
            return True
        visiting.remove(node)
        visited.add(node)
        return False
    return any(visit(node) for node in ids)


def mechanics_audit() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="theseus-p4-mechanics-") as tmp:
        root = Path(tmp)
        source = root / "sample.py"
        source.write_text("VALUES = (1, 2)\n\ndef choose(value):\n    return value in VALUES\n", encoding="utf-8")
        task = {
            "allowed_effect_paths": ["sample.py"],
            "candidate_visible_context": {"reads": [{"path": "sample.py", "start_line": 1, "end_line": 4}]},
            "semantic_ir_contract": {"maximum_symbol_nodes": 40},
            "obligations": [
                {"id": "O1", "kind": "require", "text": "add 3"},
                {"id": "O2", "kind": "preserve", "text": "preserve 1 and 2"},
                {"id": "O3", "kind": "non_goal", "text": "do not change choose"},
            ],
            "obligation_dependencies": [{"before": "O2", "after": "O1"}],
        }
        symbols = semantic_symbol_table(root, task)
        target = next(row for row in symbols["nodes"] if row["node_type"] == "Tuple")
        valid = (
            f"{IR_HEADER}\nSOURCE {symbols['source_digest']}\nOBLIGATIONS O1,O2,O3\n"
            f"UNIT U1 O1,O2,O3 REPLACE sample.py {target['id']} {target['sha256']}\n"
            "<<<\n(1, 2, 3)\n>>>\nLOSS NONE\nEND"
        )
        parsed = parse_semantic_ir(valid, task, root)
        mutations = {
            "source": valid.replace(symbols["source_digest"], "0" * 64),
            "obligation": valid.replace("OBLIGATIONS O1,O2,O3", "OBLIGATIONS O1,O2"),
            "target": valid.replace(target["sha256"], "0" * 64),
            "loss": valid.replace("LOSS NONE", "LOSS O1"),
        }
        rejected = {
            name: bool(parse_semantic_ir(value, task, root)["faults"])
            for name, value in mutations.items()
        }
        ready = not parsed["faults"] and len(parsed["actions"]) == 1 and all(rejected.values())
        return {
            "trigger_state": "GREEN" if ready else "RED",
            "valid_ir_action_count": len(parsed["actions"]),
            "valid_ir_faults": parsed["faults"],
            "corruption_rejections": rejected,
            "ready": ready,
        }


if __name__ == "__main__":
    raise SystemExit(main())
