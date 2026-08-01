#!/usr/bin/env python3
"""Learned non-claim canary for Semantic-IR intervention and repair locality."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_assistant_route_integrity as route_integrity  # noqa: E402
import theseus_assistant_runtime as assistant_runtime  # noqa: E402
import theseus_local_inference_backend as local_backend  # noqa: E402
import theseus_p4_cognitive_compilation as p4  # noqa: E402
import theseus_semantic_ir_v2 as ir_v2  # noqa: E402
import theseus_semantic_ir_v2r1 as ir_v2r1  # noqa: E402


POLICY = "project_theseus_p4r_intervention_locality_canary_v1"
CONFIG_POLICY = "project_theseus_p4r_intervention_locality_canary_config_v1"
MODEL_CONTEXT_TOKENS = 262144
EXPECTED_CASES = 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/theseus_p4r_intervention_locality_canary.json"
    )
    parser.add_argument(
        "--out", default="reports/theseus_p4r_intervention_locality_canary.json"
    )
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    config_path = p2a.resolve(args.config)
    report = audit_config(config_path) if args.audit_only else run_canary(config_path)
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps({
        "trigger_state": report.get("trigger_state"),
        "state": report.get("state"),
        "first_verified": report.get("first_verified"),
        "injected_feedback_verified": report.get("injected_feedback_verified"),
        "final_verified": report.get("final_verified"),
        "dependency_local_repairs": report.get("dependency_local_repairs"),
        "model_calls": report.get("model_calls"),
        "safety_ceiling_hits": report.get("safety_ceiling_hits"),
        "faults": report.get("faults"),
    }, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") == "GREEN" else 2


def audit_config(path: Path) -> dict[str, Any]:
    value = p2a.read_json(path)
    faults: list[str] = []
    if value.get("policy") != CONFIG_POLICY:
        faults.append("config_policy_invalid")
    if value.get("state") != "FROZEN_BEFORE_LEARNED_INTERVENTION_CALLS":
        faults.append("config_not_frozen")
    for name, digest_name in (
        ("parser", "parser_sha256"),
        ("runner", "runner_sha256"),
        ("base_local_instrument", "base_local_instrument_sha256"),
        ("semantic_unit_predecessor", "semantic_unit_predecessor_sha256"),
    ):
        owner = p2a.resolve(str(value.get(name) or ""))
        if p2a.sha256_file(owner) != str(value.get(digest_name) or ""):
            faults.append(f"{name}_digest_mismatch")
    predecessor = p2a.read_json(
        p2a.resolve(str(value.get("semantic_unit_predecessor") or ""))
    )
    if predecessor.get("state") != "SEMANTIC_UNIT_GRANULARITY_MECHANICS_GREEN":
        faults.append("semantic_unit_predecessor_not_green")
    base = p2a.read_json(p2a.resolve(str(value.get("base_local_instrument") or "")))
    frozen = p2a.mapping(base.get("frozen_model"))
    if frozen.get("project_selected_quality_token_cap") is not None:
        faults.append("project_selected_quality_token_cap_present")
    if int(frozen.get("model_declared_context_window_tokens") or 0) != MODEL_CONTEXT_TOKENS:
        faults.append("model_context_binding_invalid")
    cases = p2a.dicts(value.get("cases"))
    if len(cases) != EXPECTED_CASES or len({row.get("case_id") for row in cases}) != EXPECTED_CASES:
        faults.append("case_denominator_invalid")
    if {row.get("injected_unit_id") for row in cases} != {"U1", "U2"}:
        faults.append("intervention_rotation_invalid")
    for row in cases:
        case_id = str(row.get("case_id") or "missing")
        if not all(str(row.get(key) or "") for key in (
            "case_id", "natural_request", "source", "expected_source",
            "injected_unit_id", "feedback_marker",
        )):
            faults.append(f"case_incomplete:{case_id}")
        obligations = p2a.dicts(row.get("obligations"))
        if [item.get("id") for item in obligations] != ["O1", "O2", "O3", "O4"]:
            faults.append(f"obligation_order_invalid:{case_id}")
        units = p2a.dicts(row.get("units"))
        if [item.get("unit_id") for item in units] != ["U1", "U2"]:
            faults.append(f"unit_order_invalid:{case_id}")
        if [p2a.strings(item.get("obligation_ids")) for item in units] != [
            ["O1", "O2"], ["O3", "O4"]
        ]:
            faults.append(f"unit_obligation_partition_invalid:{case_id}")
        injected = str(row.get("injected_unit_id") or "")
        unit = next((item for item in units if item.get("unit_id") == injected), {})
        if p2a.strings(row.get("expected_implicated_obligation_ids")) != p2a.strings(
            unit.get("obligation_ids")
        ):
            faults.append(f"implicated_closure_invalid:{case_id}")
    return {
        "policy": "project_theseus_p4r_intervention_locality_canary_audit_v1",
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "config_sha256": p2a.sha256_file(path),
        "counters": p2a.zero_counters(),
    }


def run_canary(
    config_path: Path,
    *,
    session_factory: Callable[..., Any] = local_backend.PersistentLocalInferenceSession,
) -> dict[str, Any]:
    audit = audit_config(config_path)
    if audit.get("trigger_state") != "GREEN":
        return terminal_failure("CONFIG_INVALID", ["config_audit_red"], audit)
    config = p2a.read_json(config_path)
    static_interventions = run_static_intervention_ladder(p2a.dicts(config.get("cases"))[0])
    if static_interventions.get("trigger_state") != "GREEN":
        return terminal_failure(
            "STATIC_INTERVENTION_LADDER_INVALID",
            ["static_intervention_ladder_red"],
            audit,
            static_interventions=static_interventions,
        )
    base = p2a.read_json(p2a.resolve(str(config.get("base_local_instrument") or "")))
    runtime = assistant_runtime.load_runtime_config(
        p2a.resolve(str(base.get("runtime_config") or ""))
    )
    binding = p2a.mapping(base.get("runtime_binding")) or p2a.mapping(
        runtime.get("local_inference")
    )
    frozen = p2a.mapping(base.get("frozen_model"))
    session = session_factory(
        worker_config_path=p2a.resolve(str(binding.get("worker_config") or "")),
        runtime_preflight_path=p2a.resolve(str(binding.get("runtime_preflight") or "")),
        maximum_tokens=MODEL_CONTEXT_TOKENS,
        required_repo_id=str(frozen.get("repo_id") or ""),
        required_revision=str(frozen.get("revision") or ""),
        required_snapshot_manifest_sha256=str(
            frozen.get("snapshot_manifest_sha256") or ""
        ),
        session_id="p4r-intervention-locality-mechanics",
        completion_predicate=ir_v2r1.complete,
    )
    if not session.ready:
        return terminal_failure(
            "BACKEND_INVALID",
            ["persistent_backend_not_ready", *list(session.faults)],
            audit,
            static_interventions=static_interventions,
        )
    rows: list[dict[str, Any]] = []
    with assistant_runtime.bind_local_inference_runner(session.runtime_runner):
        for case in p2a.dicts(config.get("cases")):
            rows.append(run_case(case, base))
    first_verified = sum(int(row.get("first_verified") is True) for row in rows)
    feedback_verified = sum(int(row.get("injected_feedback_verified") is True) for row in rows)
    final_verified = sum(int(row.get("final_verified") is True) for row in rows)
    local_repairs = sum(int(row.get("dependency_local_repair") is True) for row in rows)
    ceiling_hits = sum(int(row.get("safety_ceiling_hits") or 0) for row in rows)
    passed = (
        first_verified == feedback_verified == final_verified == local_repairs == EXPECTED_CASES
        and session.model_load_count == 1
        and session.inference_calls == EXPECTED_CASES * 2
        and ceiling_hits == 0
        and all(row.get("route_integrity_ready") is True for row in rows)
        and all(row.get("termination_ready") is True for row in rows)
        and static_interventions.get("trigger_state") == "GREEN"
    )
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if passed else "YELLOW",
        "state": (
            "INTERVENTION_AND_DEPENDENCY_LOCAL_REPAIR_MECHANICS_GREEN"
            if passed else "INCONCLUSIVE_IMPLEMENTATION"
        ),
        "faults": [] if passed else ["intervention_or_local_repair_floor_not_met"],
        "config_sha256": p2a.sha256_file(config_path),
        "config_audit": audit,
        "static_intervention_ladder": static_interventions,
        "first_verified": f"{first_verified}/{EXPECTED_CASES}",
        "injected_feedback_verified": f"{feedback_verified}/{EXPECTED_CASES}",
        "final_verified": f"{final_verified}/{EXPECTED_CASES}",
        "dependency_local_repairs": f"{local_repairs}/{EXPECTED_CASES}",
        "model_calls": session.inference_calls,
        "model_loads": session.model_load_count,
        "safety_ceiling_hits": ceiling_hits,
        "cases": rows,
        "scope": (
            "Two hand-authored non-claim mechanics cases with deterministic single-unit "
            "fault injection. Passing establishes only transport intervention sensitivity "
            "and dependency-local repair mechanics; it does not support cognitive compilation."
        ),
        "next_gate": (
            "Freeze a fresh source-disjoint P4 decision denominator; D1 and book support "
            "remain closed until a claim-bearing survivor exists."
        ),
        "counters": p2a.zero_counters(),
    }


def terminal_failure(
    state: str,
    faults: list[str],
    audit: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "RED",
        "state": state,
        "faults": faults,
        "config_audit": audit,
        **extra,
        "counters": p2a.zero_counters(),
    }


def run_case(case: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    case_id = p2a.safe_slug(str(case.get("case_id") or "case"))
    with tempfile.TemporaryDirectory(prefix=f"theseus-p4r-locality-{case_id}-") as tmp:
        root = Path(tmp)
        source_path = root / "sample.py"
        source = str(case.get("source") or "")
        source_path.write_text(source, encoding="utf-8")
        task = build_task(case)
        symbols = p4.semantic_symbol_table(root, task)
        targets = select_targets(symbols, p2a.dicts(case.get("units")))
        first_prompt = render_first_prompt(case, task, symbols, targets, source_path)
        first_call = p2a.runtime_call(
            route_integrity.DIRECT_MODE,
            f"p4r_intervention_locality_{case_id}",
            1,
            first_prompt,
            MODEL_CONTEXT_TOKENS,
            str(base.get("runtime_config") or ""),
        )
        first_text, _ = ir_v2r1.normalize(str(first_call.get("assistant_text") or ""))
        first = ir_v2r1.parse(first_text, task, root)
        first_observed, first_apply_faults = apply_and_restore(
            source_path, source, p2a.dicts(first.get("actions"))
        )
        first_verified = (
            not first.get("faults")
            and not first_apply_faults
            and first_observed == str(case.get("expected_source") or "")
        )
        injected_text = inject_unit_fault(
            first_text,
            str(case.get("injected_unit_id") or ""),
            str(case.get("injected_replacement") or ""),
        )
        injected = ir_v2r1.parse(injected_text, task, root)
        injected_observed, injected_apply_faults = apply_and_restore(
            source_path, source, p2a.dicts(injected.get("actions"))
        )
        marker = str(case.get("feedback_marker") or "")
        visible_feedback = visible_validation(case, injected_observed, injected_apply_faults)
        implicated = p4.implicated_obligations(task, injected, {
            "apply_faults": injected_apply_faults,
            "visible_verifier": visible_feedback,
        })
        expected_implicated = set(p2a.strings(case.get("expected_implicated_obligation_ids")))
        feedback_verified = (
            not injected.get("faults")
            and not injected_apply_faults
            and visible_feedback.get("passed") is False
            and marker in str(visible_feedback.get("stdout_tail") or "")
            and implicated == expected_implicated
        )
        repair_prompt = render_repair_prompt(
            first_prompt, injected_text, visible_feedback, implicated,
            untouched_unit_id(case),
        )
        final_call = p2a.runtime_call(
            route_integrity.DIRECT_MODE,
            f"p4r_intervention_locality_{case_id}",
            2,
            repair_prompt,
            MODEL_CONTEXT_TOKENS,
            str(base.get("runtime_config") or ""),
        )
        final_text, _ = ir_v2r1.normalize(str(final_call.get("assistant_text") or ""))
        final = ir_v2r1.parse(final_text, task, root)
        final_observed, final_apply_faults = apply_and_restore(
            source_path, source, p2a.dicts(final.get("actions"))
        )
        locality_faults = p4.repair_locality_faults(
            p2a.dicts(injected.get("units")),
            p2a.dicts(final.get("units")),
            implicated,
        ) if not injected.get("faults") and not final.get("faults") else [
            "locality_not_evaluable"
        ]
        injected_units = unit_fingerprints(p2a.dicts(injected.get("units")))
        final_units = unit_fingerprints(p2a.dicts(final.get("units")))
        injected_id = str(case.get("injected_unit_id") or "")
        untouched_id = untouched_unit_id(case)
        dependency_local = (
            not locality_faults
            and injected_units.get(injected_id) != final_units.get(injected_id)
            and injected_units.get(untouched_id) == final_units.get(untouched_id)
            and bool(injected_units.get(untouched_id))
        )
        final_verified = (
            not final.get("faults")
            and not final_apply_faults
            and final_observed == str(case.get("expected_source") or "")
        )
        call_rows = [runtime_metrics(first_call), runtime_metrics(final_call)]
        return {
            "case_id": case.get("case_id"),
            "injected_unit_id": injected_id,
            "untouched_unit_id": untouched_id,
            "first_parse_faults": p2a.strings(first.get("faults")),
            "first_apply_faults": first_apply_faults,
            "first_verified": first_verified,
            "injected_parse_faults": p2a.strings(injected.get("faults")),
            "injected_apply_faults": injected_apply_faults,
            "injected_feedback_verified": feedback_verified,
            "feedback_marker": marker,
            "implicated_obligation_ids": sorted(implicated),
            "expected_implicated_obligation_ids": sorted(expected_implicated),
            "final_parse_faults": p2a.strings(final.get("faults")),
            "final_apply_faults": final_apply_faults,
            "final_verified": final_verified,
            "locality_faults": locality_faults,
            "targeted_unit_changed": injected_units.get(injected_id) != final_units.get(injected_id),
            "untouched_unit_byte_stable": injected_units.get(untouched_id) == final_units.get(untouched_id),
            "dependency_local_repair": dependency_local,
            "expected_source_sha256": p2a.sha256_text(str(case.get("expected_source") or "")),
            "final_source_sha256": p2a.sha256_text(final_observed),
            "runtime_calls": call_rows,
            "route_integrity_ready": all(row.get("route_integrity_ready") is True for row in call_rows),
            "termination_ready": all(
                row.get("termination_reason") in {"parser_complete", "model_eos"}
                for row in call_rows
            ),
            "safety_ceiling_hits": sum(int(row.get("safety_ceiling_hit") is True) for row in call_rows),
        }


def build_task(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "allowed_effect_paths": ["sample.py"],
        "candidate_visible_context": {
            "reads": [{"path": "sample.py", "start_line": 1, "end_line": 1000}]
        },
        "semantic_ir_contract": {"maximum_symbol_nodes": 80},
        "obligations": p2a.dicts(case.get("obligations")),
        "obligation_dependencies": p2a.dicts(case.get("obligation_dependencies")),
        "visible_feedback_map": [{
            "marker": str(case.get("feedback_marker") or ""),
            "obligation_ids": p2a.strings(case.get("feedback_obligation_ids")),
        }],
    }


def select_targets(
    symbols: dict[str, Any], units: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    rows = p2a.dicts(symbols.get("nodes"))
    selected: dict[str, dict[str, Any]] = {}
    for unit in units:
        line = int(unit.get("target_line") or 0)
        matches = [
            row for row in rows
            if row.get("node_type") == "Assign" and int(row.get("start_line") or 0) == line
        ]
        if len(matches) != 1:
            raise p4.P4Fault(f"semantic_unit_target_not_unique:{unit.get('unit_id')}:{len(matches)}")
        selected[str(unit.get("unit_id") or "")] = matches[0]
    return selected


def render_first_prompt(
    case: dict[str, Any], task: dict[str, Any], symbols: dict[str, Any],
    targets: dict[str, dict[str, Any]], source_path: Path,
) -> str:
    obligations = "\n".join(
        f"{row['id']} {str(row['kind']).upper()}: {row['text']}"
        for row in p2a.dicts(task.get("obligations"))
    )
    dependencies = "\n".join(
        f"{row['before']} -> {row['after']}"
        for row in p2a.dicts(task.get("obligation_dependencies"))
    )
    unit_lines: list[str] = []
    for unit in p2a.dicts(case.get("units")):
        unit_id = str(unit.get("unit_id") or "")
        target = targets[unit_id]
        unit_lines.append(
            f"{unit_id}: OBLIGATIONS {','.join(p2a.strings(unit.get('obligation_ids')))}; "
            f"OP REPLACE; PATH sample.py; NODE {target['id']}; NODE_SHA {target['sha256']}; "
            f"TARGET_NODE_SOURCE {selected_node_source(source_path, target)!r}"
        )
    return (
        f"FIRST LINE MUST BE EXACTLY: {ir_v2r1.HEADER}\n"
        "Return only one complete labeled Semantic IR artifact.\n"
        f"Request: {case.get('natural_request')}\n\n"
        + ir_v2r1.grammar()
        + "\n\nEmit exactly U1 then U2. Copy every bound identity exactly. Use OP REPLACE. "
        "Each replacement block must contain only the complete replacement assignment for "
        "its bound Assign node. Set LOSS NONE.\n"
        f"SOURCE {symbols['source_digest']}\nALL_OBLIGATIONS O1,O2,O3,O4\n"
        f"OBLIGATIONS\n{obligations}\nDEPENDENCIES\n{dependencies}\n"
        + "\n".join(unit_lines)
        + f"\nSOURCE_TEXT\n{case.get('source')}"
    )


def render_repair_prompt(
    original_prompt: str,
    injected_text: str,
    feedback: dict[str, Any],
    implicated: set[str],
    untouched_unit: str,
) -> str:
    packet = {
        "parse_or_lower_faults": [],
        "apply_faults": [],
        "visible_verifier_returncode": feedback.get("returncode"),
        "visible_verifier_stdout_tail": feedback.get("stdout_tail"),
        "visible_verifier_stderr_tail": feedback.get("stderr_tail"),
        "dependency_local_repair_obligation_ids": sorted(implicated),
    }
    return (
        original_prompt
        + "\n\n[PROVISIONAL_OUTPUT_AFTER_CONTROLLED_SINGLE_UNIT_FAULT_INJECTION]\n"
        + injected_text
        + "\n\n[ACTUAL_TARGET_VALIDATION_FEEDBACK]\n"
        + json.dumps(packet, sort_keys=True)
        + "\n\nReturn one complete corrected artifact against the ORIGINAL snapshot. Repair only "
        "the dependency-closed implicated unit. Copy the complete " + untouched_unit
        + " block byte-for-byte from the provisional artifact. Preserve both unit IDs, "
        "targets, hashes, obligation references, order, and LOSS NONE."
    )


def inject_unit_fault(text: str, unit_id: str, replacement: str) -> str:
    matches = 0

    def replace(match: Any) -> str:
        nonlocal matches
        if match.group(1) != unit_id:
            return match.group(0)
        matches += 1
        groups = match.groups()
        return (
            f"UNIT {groups[0]}\nOBLIGATIONS {groups[1]}\nOP {groups[2]}\n"
            f"PATH {groups[3]}\nNODE {groups[4]}\nNODE_SHA {groups[5]}\n"
            f"<<<\n{replacement}\n>>>\nEND_UNIT"
        )

    mutated = ir_v2.UNIT_RE.sub(replace, text)
    if matches != 1:
        raise p4.P4Fault(f"controlled_fault_injection_target_invalid:{unit_id}:{matches}")
    return mutated


def visible_validation(
    case: dict[str, Any], observed: str, apply_faults: list[str]
) -> dict[str, Any]:
    passed = not apply_faults and observed == str(case.get("expected_source") or "")
    marker = "PASS" if passed else str(case.get("feedback_marker") or "VALIDATION_FAILED")
    return {
        "passed": passed,
        "returncode": 0 if passed else 1,
        "stdout_tail": marker,
        "stderr_tail": "" if not apply_faults else "\n".join(apply_faults),
    }


def apply_and_restore(
    source_path: Path, source: str, actions: list[dict[str, Any]]
) -> tuple[str, list[str]]:
    faults = p2a.apply_actions(source_path.parent, actions) if actions else []
    observed = source_path.read_text(encoding="utf-8")
    source_path.write_text(source, encoding="utf-8")
    return observed, faults


def unit_fingerprints(units: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(row.get("id") or ""): p2a.stable_hash(row)
        for row in units
    }


def untouched_unit_id(case: dict[str, Any]) -> str:
    injected = str(case.get("injected_unit_id") or "")
    return "U2" if injected == "U1" else "U1"


def selected_node_source(path: Path, target: dict[str, Any]) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    start, end = int(target["start_line"]), int(target["end_line"])
    start_col, end_col = int(target["start_col"]), int(target["end_col"])
    if start == end:
        return lines[start - 1][start_col:end_col]
    return "\n".join(
        [lines[start - 1][start_col:], *lines[start:end - 1], lines[end - 1][:end_col]]
    )


def runtime_metrics(call: dict[str, Any]) -> dict[str, Any]:
    runtime = p2a.mapping(call.get("runtime_report"))
    backend_path = p2a.resolve(
        str(p2a.mapping(runtime.get("generation_backend")).get("out") or "")
    )
    metrics = p2a.mapping(p2a.read_json(backend_path).get("metrics"))
    return {
        "runtime_report": p2a.mapping(call.get("receipt")).get("report_path"),
        "runtime_report_sha256": p2a.mapping(call.get("receipt")).get("report_sha256"),
        "backend_report": p2a.rel(backend_path),
        "backend_report_sha256": p2a.sha256_file(backend_path),
        "route_integrity_ready": p2a.mapping(runtime.get("route_integrity")).get("ready") is True,
        "termination_reason": metrics.get("termination_reason"),
        "prompt_tokens": metrics.get("prompt_tokens"),
        "generated_tokens": metrics.get("generated_tokens"),
        "safety_ceiling_hit": metrics.get("safety_ceiling_hit"),
    }


def run_static_intervention_ladder(case: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="theseus-p4r-static-interventions-") as tmp:
        root = Path(tmp)
        source_path = root / "sample.py"
        source_path.write_text(str(case.get("source") or ""), encoding="utf-8")
        task = build_task(case)
        symbols = p4.semantic_symbol_table(root, task)
        targets = select_targets(symbols, p2a.dicts(case.get("units")))
        valid = render_fixture(case, symbols, targets)
        parsed = ir_v2r1.parse(valid, task, root)
        mutations = {
            "source_identity": valid.replace(symbols["source_digest"], "0" * 64, 1),
            "obligation_coverage": valid.replace(
                "ALL_OBLIGATIONS O1,O2,O3,O4", "ALL_OBLIGATIONS O1,O2,O3", 1
            ),
            "target_identity": valid.replace(targets["U1"]["sha256"], "0" * 64, 1),
            "loss_record": valid.replace("LOSS NONE", "LOSS O1", 1),
            "dependency_closure": valid.replace(
                "\nOBLIGATIONS O1,O2\n", "\nOBLIGATIONS O1\n", 1
            ),
        }
        results = {
            name: p2a.strings(ir_v2r1.parse(value, task, root).get("faults"))
            for name, value in mutations.items()
        }
        required_faults = {
            "source_identity": "semantic_source_identity_invalid",
            "obligation_coverage": "semantic_obligation_identity_or_order_invalid",
            "target_identity": "semantic_target_identity_invalid",
            "loss_record": "semantic_loss_unresolved",
            "dependency_closure": "semantic_unit_dependency_not_closed",
        }
        rejected = {
            name: required_fault in results.get(name, [])
            for name, required_fault in required_faults.items()
        }
        ready = not parsed.get("faults") and len(p2a.dicts(parsed.get("actions"))) == 2 and all(
            rejected.values()
        )
        return {
            "trigger_state": "GREEN" if ready else "RED",
            "valid_fixture_parse_faults": p2a.strings(parsed.get("faults")),
            "valid_fixture_action_count": len(p2a.dicts(parsed.get("actions"))),
            "required_corruption_rejections": rejected,
            "corruption_faults": results,
            "learned_generation_credit": False,
        }


def render_fixture(
    case: dict[str, Any],
    symbols: dict[str, Any],
    targets: dict[str, dict[str, Any]],
) -> str:
    lines = [
        ir_v2r1.HEADER,
        f"SOURCE {symbols['source_digest']}",
        "ALL_OBLIGATIONS O1,O2,O3,O4",
    ]
    for unit in p2a.dicts(case.get("units")):
        unit_id = str(unit.get("unit_id") or "")
        target = targets[unit_id]
        lines.extend([
            f"UNIT {unit_id}",
            f"OBLIGATIONS {','.join(p2a.strings(unit.get('obligation_ids')))}",
            "OP REPLACE",
            "PATH sample.py",
            f"NODE {target['id']}",
            f"NODE_SHA {target['sha256']}",
            "<<<",
            str(unit.get("expected_replacement") or ""),
            ">>>",
            "END_UNIT",
        ])
    lines.extend(["LOSS NONE", "END"])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
