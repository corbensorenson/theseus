#!/usr/bin/env python3
"""Adequate frozen-model direct/integrated experimental instrument (P2A)."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import theseus_assistant_route_integrity as route_integrity
import theseus_assistant_runtime as assistant_runtime
import theseus_local_inference_backend as local_backend


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INSTRUMENT = ROOT / "configs" / "theseus_assistant_p2a_instrument.json"
POLICY = "project_theseus_p2a_frozen_model_run_v1"
INSTRUMENT_POLICIES = {
    "project_theseus_p2a_frozen_model_instrument_v1",
    "project_theseus_p2b_frozen_model_instrument_v1",
    "project_theseus_p2c_frozen_model_instrument_v1",
    "project_theseus_p3_frozen_model_campaign_v1",
}
ARMS = (route_integrity.DIRECT_MODE, route_integrity.INTEGRATED_MODE)
FORBIDDEN_TASK_FIELDS = {
    "answer", "category", "expected", "hidden_tests", "required_constructs",
    "return_shape", "solution", "solution_body", "solution_expr", "source_task_id",
    "tests", "type_family",
}
ACTION_HEADER = "THESEUS_EDIT_V1"
ACTION_RE = re.compile(
    r"REPLACE ([^\n ]+) ([1-9][0-9]*) ([1-9][0-9]*)\n<<<\n(.*?)\n>>>",
    flags=re.DOTALL,
)


class InstrumentFault(ValueError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", default=rel(DEFAULT_INSTRUMENT))
    parser.add_argument("--task", default="")
    parser.add_argument("--out", default="reports/theseus_assistant_p2a.json")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    instrument_path = resolve(args.instrument)
    if args.audit_only:
        report = audit_instrument(instrument_path)
    elif args.task:
        report = run_experiment(instrument_path, resolve(args.task))
    else:
        parser.error("--task is required unless --audit-only is used")
    write_json(resolve(args.out), report)
    print(json.dumps(compact_summary(report), indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") in {"GREEN", "YELLOW"} else 2


def audit_instrument(path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    value = read_json(path)
    faults: list[str] = []
    if value.get("policy") not in INSTRUMENT_POLICIES:
        faults.append("instrument_policy_invalid")
    if value.get("state") not in {
        "PROSPECTIVELY_BOUND_BEFORE_TASK_ACQUISITION",
        "P2B_PROSPECTIVELY_BOUND_BEFORE_TASK_ACQUISITION",
        "P2C_PROSPECTIVELY_BOUND_BEFORE_TASK_ACQUISITION",
        "P3_PROSPECTIVELY_BOUND_BEFORE_TASK_POOL_ACQUISITION",
    }:
        faults.append("instrument_not_prospectively_bound")
    runtime_config = assistant_runtime.load_runtime_config(resolve(str(value.get("runtime_config") or "")))
    local = mapping(runtime_config.get("local_inference"))
    runtime_binding = mapping(value.get("runtime_binding")) or local
    frozen = mapping(value.get("frozen_model"))
    maximum = int(frozen.get("maximum_generation_tokens_per_call") or 0)
    contract = route_integrity.load_model_contract(
        str(runtime_binding.get("worker_config") or ""),
        str(runtime_binding.get("runtime_preflight") or ""),
        maximum_tokens=maximum,
        required_repo_id=str(frozen.get("repo_id") or ""),
        required_revision=str(frozen.get("revision") or ""),
        required_snapshot_manifest_sha256=str(frozen.get("snapshot_manifest_sha256") or ""),
    )
    identity = mapping(contract.get("identity"))
    if contract.get("ready") is not True:
        faults.append("frozen_model_contract_not_ready")
    if identity.get("identity_sha256") != frozen.get("identity_sha256"):
        faults.append("frozen_model_identity_mismatch")
    if identity.get("decoder_sha256") != frozen.get("decoder_sha256"):
        faults.append("frozen_decoder_mismatch")
    if maximum < 1024 or maximum > 1536:
        faults.append("generation_budget_outside_p2a_bounds")
    if int(frozen.get("maximum_model_calls_per_arm") or 0) != 2:
        faults.append("repair_call_budget_invalid")
    if int(frozen.get("persistent_model_loads_per_pair") or 0) != 1:
        faults.append("persistent_load_budget_invalid")
    matched = mapping(value.get("matched_arm_contract"))
    if tuple(strings(matched.get("arm_order"))) != ARMS:
        faults.append("arm_order_invalid")
    for key in (
        "same_source_snapshot", "same_natural_request",
        "same_candidate_visible_repository_context", "same_typed_edit_protocol",
        "same_patch_application", "same_visible_verifier",
        "same_hidden_evaluator_after_seal", "same_decoder_and_budget_caps",
        "one_visible_verifier_repair_allowed", "route_label_hidden_from_evaluator",
    ):
        if matched.get(key) is not True and key != "same_typed_edit_protocol":
            faults.append(f"matched_contract_false:{key}")
    if matched.get("same_typed_edit_protocol") != "theseus_line_edit_v1":
        faults.append("typed_edit_protocol_mismatch")
    successor_policy = value.get("policy") in {
        "project_theseus_p2b_frozen_model_instrument_v1",
        "project_theseus_p2c_frozen_model_instrument_v1",
        "project_theseus_p3_frozen_model_campaign_v1",
    }
    grammar_round_trip: dict[str, Any] = {}
    if successor_policy:
        if value.get("candidate_path_namespace") != "repository_root_relative_no_archive_prefix":
            faults.append("p2b_candidate_path_namespace_invalid")
        selection_path = resolve(str(value.get("model_selection_report") or ""))
        if sha256_file(selection_path) != str(value.get("model_selection_report_sha256") or ""):
            faults.append("p2b_model_selection_digest_mismatch")
        selection = read_json(selection_path)
        expected_selection_state = (
            "FROZEN_AS_BEST_RETAINED_P3_DEVELOPMENT_LOCAL_DENOMINATOR_NOT_CAPABILITY_QUALIFIED"
            if value.get("policy") == "project_theseus_p3_frozen_model_campaign_v1"
            else "SELECTED_FOR_P2B_INSTRUMENT_ONLY_NOT_QUALIFIED"
        )
        if selection.get("selection_state") != expected_selection_state:
            faults.append("p2b_model_selection_state_invalid")
        if mapping(selection.get("selected_model_identity")).get("revision") != frozen.get("revision"):
            faults.append("p2b_selected_model_revision_mismatch")
        harness = mapping(value.get("harness"))
        for name in ("candidate_runner", "blind_evaluator", "assistant_runtime"):
            harness_path = resolve(str(harness.get(name) or ""))
            if sha256_file(harness_path) != str(harness.get(f"{name}_sha256") or ""):
                faults.append(f"p2b_{name}_digest_mismatch")
    if value.get("policy") in {
        "project_theseus_p2c_frozen_model_instrument_v1",
        "project_theseus_p3_frozen_model_campaign_v1",
    }:
        protocol = mapping(value.get("candidate_protocol"))
        grammar = str(protocol.get("grammar") or "")
        example = (
            grammar.replace("<repository-relative-path>", "src/example.py")
            .replace("<start-line>", "1")
            .replace("<end-line>", "1")
            .replace("<replacement text>", "value = 1")
        )
        actions, parse_faults = parse_actions(
            example, {"allowed_effect_paths": ["src/example.py"]}, protocol
        )
        rendered = render_candidate_prompt("Change the value.", "[READ src/example.py:1-1]\nvalue = 0", protocol)
        grammar_round_trip = {
            "configured_grammar_contains_actual_newlines": "\n" in grammar,
            "configured_grammar_contains_literal_backslash_n": "\\n" in grammar,
            "configured_grammar_rendered_exactly": grammar in rendered,
            "example_parse_faults": parse_faults,
            "example_action_count": len(actions),
            "ready": (
                "\n" in grammar
                and "\\n" not in grammar
                and grammar in rendered
                and not parse_faults
                and len(actions) == 1
            ),
        }
        if grammar_round_trip["ready"] is not True:
            faults.append("p2c_rendered_grammar_parser_round_trip_invalid")
    if value.get("policy") == "project_theseus_p3_frozen_model_campaign_v1":
        if matched.get("arm_order_policy") != "campaign_index_parity_counterbalance_v1":
            faults.append("p3_arm_order_policy_invalid")
        campaign = mapping(value.get("campaign_contract"))
        if int(campaign.get("task_count") or 0) != 10:
            faults.append("p3_task_count_invalid")
        if campaign.get("task_pool_frozen_before_candidate_generation") is not True:
            faults.append("p3_task_pool_not_prospectively_frozen")
    return {
        "policy": "project_theseus_p2a_instrument_audit_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "instrument_sha256": sha256_file(path),
        "model_identity": identity,
        "grammar_round_trip": grammar_round_trip,
        "runtime_ms": round((time.perf_counter() - started) * 1000, 3),
        "counters": zero_counters(),
    }


def audit_task(path: Path) -> dict[str, Any]:
    task = read_json(path)
    faults: list[str] = []
    if task.get("policy") not in {
        "project_theseus_p2a_licensed_task_v1",
        "project_theseus_p2b_licensed_task_v1",
        "project_theseus_p3_licensed_task_v1",
    }:
        faults.append("task_policy_invalid")
    if task.get("state") != "SEALED_BEFORE_CANDIDATE_GENERATION":
        faults.append("task_not_sealed")
    if FORBIDDEN_TASK_FIELDS.intersection(task):
        faults.append("answer_identifying_task_field_present")
    archive = resolve(str(task.get("source_archive") or ""))
    if sha256_file(archive) != str(task.get("source_archive_sha256") or ""):
        faults.append("source_archive_digest_mismatch")
    source_root = str(task.get("source_archive_root") or "")
    if task.get("policy") in {
        "project_theseus_p2b_licensed_task_v1",
        "project_theseus_p3_licensed_task_v1",
    } and not source_root:
        faults.append("source_archive_root_missing")
    if task.get("policy") == "project_theseus_p3_licensed_task_v1":
        campaign_index = int(task.get("campaign_index") or 0)
        if campaign_index < 1 or campaign_index > 10:
            faults.append("p3_campaign_index_invalid")
    if source_root:
        try:
            with tarfile.open(archive) as handle:
                visible = [strip_source_root(member.name, source_root) for member in handle.getmembers()]
            if not any(path for path in visible):
                faults.append("source_archive_root_empty")
        except (tarfile.TarError, InstrumentFault):
            faults.append("source_archive_root_invalid")
    provenance = mapping(task.get("source_provenance"))
    if not all(str(provenance.get(key) or "").strip() for key in ("url", "revision", "license_spdx")):
        faults.append("source_provenance_incomplete")
    if not str(task.get("natural_request") or "").strip():
        faults.append("natural_request_missing")
    if not strings(task.get("allowed_effect_paths")):
        faults.append("allowed_effect_paths_missing")
    context = mapping(task.get("candidate_visible_context"))
    if not dicts(context.get("reads")) and not dicts(context.get("searches")):
        faults.append("candidate_visible_context_empty")
    candidate_paths = [*strings(task.get("allowed_effect_paths"))]
    candidate_paths.extend(str(row.get("path") or "") for row in dicts(context.get("reads")))
    for row in dicts(context.get("searches")):
        candidate_paths.extend(strings(row.get("paths")))
    if source_root and any(path == source_root or path.startswith(source_root.rstrip("/") + "/") for path in candidate_paths):
        faults.append("archive_prefix_exposed_in_candidate_path_namespace")
    verifier = mapping(task.get("visible_verifier"))
    command = strings(verifier.get("command"))
    if not command or command[0] not in {"python3", "pytest"}:
        faults.append("visible_verifier_command_invalid")
    return {
        "policy": "project_theseus_p2a_task_audit_v1",
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "task_sha256": sha256_file(path),
        "source_archive_sha256": sha256_file(archive),
        "counters": zero_counters(),
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
        return invalid_report(instrument_audit, task_audit)
    instrument = read_json(instrument_path)
    task = read_json(task_path)
    runtime_config = assistant_runtime.load_runtime_config(resolve(str(instrument.get("runtime_config") or "")))
    local = mapping(runtime_config.get("local_inference"))
    runtime_binding = mapping(instrument.get("runtime_binding")) or local
    frozen = mapping(instrument.get("frozen_model"))
    session = session_factory(
        worker_config_path=resolve(str(runtime_binding.get("worker_config") or "")),
        runtime_preflight_path=resolve(str(runtime_binding.get("runtime_preflight") or "")),
        maximum_tokens=int(frozen.get("maximum_generation_tokens_per_call") or 0),
        required_repo_id=str(frozen.get("repo_id") or ""),
        required_revision=str(frozen.get("revision") or ""),
        required_snapshot_manifest_sha256=str(frozen.get("snapshot_manifest_sha256") or ""),
        session_id=f"p2a-{safe_slug(str(task.get('opaque_task_id') or 'task'))}",
    )
    if not session.ready:
        return {
            "policy": POLICY, "created_utc": now(), "trigger_state": "RED",
            "faults": ["persistent_backend_not_ready", *list(session.faults)],
            "instrument_audit": instrument_audit, "task_audit": task_audit,
            "counters": zero_counters(),
        }
    attempts: list[dict[str, Any]] = []
    actual_arm_order = arm_order_for_experiment(instrument, task)
    with assistant_runtime.bind_local_inference_runner(session.runtime_runner):
        for arm in actual_arm_order:
            attempts.append(run_arm(arm, instrument, task, session))
    pair = pair_receipt(attempts, session)
    parseable = sum(int(row.get("parseable_candidate") is True) for row in attempts)
    faults = [] if pair.get("ready") else ["matched_pair_or_persistence_invalid"]
    return {
        "policy": POLICY,
        "created_utc": now(),
        "trigger_state": "RED" if faults else ("GREEN" if parseable else "YELLOW"),
        "faults": faults,
        "scope": "P2A instrument and one development task only; no subsystem, D1, D2, or book-support claim",
        "instrument_sha256": sha256_file(instrument_path),
        "task_sha256": sha256_file(task_path),
        "instrument_audit": instrument_audit,
        "task_audit": task_audit,
        "attempts": attempts,
        "actual_arm_order": list(actual_arm_order),
        "matched_pair": pair,
        "denominators": {
            "tasks": 1, "arms": 2, "parseable_candidates": parseable,
            "model_calls": session.inference_calls, "model_loads": session.model_load_count,
        },
        "instrument_disposition": (
            "P2A_CANDIDATES_READY_FOR_BLIND_EVALUATION" if parseable
            else "P2A_INSTRUMENT_INADEQUATE"
        ),
        "counters": zero_counters(),
        "runtime_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def arm_order_for_experiment(
    instrument: dict[str, Any], task: dict[str, Any]
) -> tuple[str, str]:
    if instrument.get("policy") != "project_theseus_p3_frozen_model_campaign_v1":
        return ARMS
    index = int(task.get("campaign_index") or 0)
    if index < 1 or index > 10:
        raise InstrumentFault("p3_campaign_index_invalid")
    return ARMS if index % 2 else (ARMS[1], ARMS[0])


def run_arm(arm: str, instrument: dict[str, Any], task: dict[str, Any], session: Any) -> dict[str, Any]:
    task_id = safe_slug(str(task.get("opaque_task_id") or "task"))
    protocol = mapping(instrument.get("candidate_protocol"))
    maximum = int(mapping(instrument.get("frozen_model")).get("maximum_generation_tokens_per_call") or 0)
    with tempfile.TemporaryDirectory(prefix=f"theseus-p2a-{arm}-") as tmp:
        candidate = Path(tmp) / "candidate"
        extract_source_archive(
            resolve(str(task.get("source_archive") or "")),
            candidate,
            str(task.get("source_archive_root") or ""),
        )
        baseline = inventory(candidate)
        context = render_visible_context(candidate, task)
        prompt = render_candidate_prompt(str(task.get("natural_request") or ""), context, protocol)
        calls: list[dict[str, Any]] = []
        route_rounds: list[dict[str, Any]] = []
        fault_history: list[dict[str, Any]] = []
        response = runtime_call(
            arm, task_id, 1, prompt, maximum, str(instrument.get("runtime_config") or "")
        )
        calls.append(response["receipt"])
        route_rounds.append(mapping(response["runtime_report"]).get("route_integrity"))
        actions, parse_faults = parse_actions(response["assistant_text"], task, protocol)
        apply_faults = apply_actions(candidate, actions) if actions else []
        accepted_actions = list(actions) if actions and not apply_faults else []
        verification = run_visible_verifier(candidate, task) if actions and not apply_faults else {}
        if parse_faults or apply_faults or verification.get("passed") is not True:
            fault_history.append({"call_number": 1, "parse_faults": parse_faults, "apply_faults": apply_faults})
            feedback = bounded_feedback(parse_faults, apply_faults, verification)
            repair_prompt = render_repair_prompt(prompt, feedback)
            response = runtime_call(
                arm, task_id, 2, repair_prompt, maximum, str(instrument.get("runtime_config") or "")
            )
            calls.append(response["receipt"])
            route_rounds.append(mapping(response["runtime_report"]).get("route_integrity"))
            repair_actions, repair_parse_faults = parse_actions(response["assistant_text"], task, protocol)
            repair_apply_faults = apply_actions(candidate, repair_actions) if repair_actions else []
            if repair_actions and not repair_apply_faults:
                accepted_actions.extend(repair_actions)
            parse_faults = repair_parse_faults
            apply_faults = repair_apply_faults
            verification = run_visible_verifier(candidate, task) if repair_actions and not repair_apply_faults else verification
        changed = changed_paths(baseline, inventory(candidate))
        authorized = bool(changed) and set(changed).issubset(set(strings(task.get("allowed_effect_paths"))))
        parseable = bool(accepted_actions) and not parse_faults and not apply_faults and authorized
        candidate_payload = {
            "protocol": "theseus_line_edit_v1",
            "actions": accepted_actions,
            "changed_paths": changed,
            "final_inventory_sha256": stable_hash(inventory(candidate)),
            "visible_verifier": verification,
        }
        seal = {
            "candidate_output_sha256": stable_hash(candidate_payload),
            "task_sha256": stable_hash({
                "opaque_task_id": task.get("opaque_task_id"),
                "natural_request": task.get("natural_request"),
                "source_archive_sha256": task.get("source_archive_sha256"),
            }),
            "sealed_before_hidden_evaluation": True,
        } if parseable else {}
        return {
            "arm_id": arm,
            "parseable_candidate": parseable,
            "parse_faults": sorted(set(parse_faults)),
            "apply_faults": sorted(set(apply_faults)),
            "candidate": candidate_payload if parseable else {},
            "candidate_seal": seal,
            "runtime_calls": calls,
            "route_integrity_rounds": route_rounds,
            "repair_fault_history": fault_history,
            "visible_verifier_passed": verification.get("passed") is True,
            "model_calls": len(calls),
            "model_identity_sha256": mapping(
                mapping(mapping(response.get("runtime_report")).get("route_integrity")).get("pair_contract")
            ).get("model_identity_sha256"),
            "route_integrity": route_rounds[0],
        }


def runtime_call(
    arm: str,
    task_id: str,
    call_number: int,
    prompt: str,
    maximum: int,
    runtime_config: str,
) -> dict[str, Any]:
    session_id = f"p2a_{task_id}_{arm}_{call_number}"
    base = ROOT / "runtime" / "p2a" / session_id
    argv = [
        "theseus_assistant_runtime.py", "--execution-mode", arm, "--intent", "code",
        "--config", runtime_config,
        "--session-id", session_id, "--prompt", prompt,
        "--local-maximum-tokens", str(maximum),
        "--out", rel(base.with_suffix(".json")),
        "--markdown-out", rel(base.with_suffix(".md")),
        "--events-out", rel(base.with_name(base.name + "_events.jsonl")),
        "--viea-trace-out", rel(base.with_name(base.name + "_viea.jsonl")),
        "--skip-context-refresh", "--skip-dogfood",
    ]
    old_argv = sys.argv
    capture = io.StringIO()
    started = time.perf_counter()
    try:
        sys.argv = argv
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            returncode = assistant_runtime.main()
    finally:
        sys.argv = old_argv
    report = read_json(base.with_suffix(".json"))
    return {
        "assistant_text": str(report.get("assistant_text") or ""),
        "runtime_report": report,
        "receipt": {
            "call_number": call_number,
            "returncode": returncode,
            "runtime_ms": round((time.perf_counter() - started) * 1000, 3),
            "report_path": rel(base.with_suffix(".json")),
            "report_sha256": sha256_file(base.with_suffix(".json")),
            "route_integrity_ready": mapping(report.get("route_integrity")).get("ready") is True,
            "runtime_trigger_state": report.get("trigger_state"),
            "runtime_failed_gates": [
                {"name": row.get("name"), "severity": row.get("severity")}
                for row in dicts(report.get("gates")) if row.get("passed") is False
            ],
            "candidate_output_sha256": sha256_text(str(report.get("assistant_text") or "")),
        },
    }


def parse_actions(text: str, task: dict[str, Any], protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    raw = str(text or "").strip()
    fenced = re.search(r"```(?:text)?\s*(.*?)\s*```", raw, flags=re.DOTALL)
    raw = (fenced.group(1) if fenced else raw).strip()
    faults: list[str] = []
    if not raw.startswith(ACTION_HEADER) or not raw.endswith("END"):
        return [], ["typed_action_envelope_invalid"]
    body = raw[len(ACTION_HEADER):].strip()
    body = body[:-3].rstrip()
    actions: list[dict[str, Any]] = []
    cursor = 0
    for match in ACTION_RE.finditer(body):
        if body[cursor:match.start()].strip():
            faults.append("typed_action_unparsed_text")
        path = match.group(1)
        start, end = int(match.group(2)), int(match.group(3))
        replacement = match.group(4)
        actions.append({"op": "REPLACE", "path": path, "start_line": start, "end_line": end, "replacement": replacement})
        cursor = match.end()
    if body[cursor:].strip():
        faults.append("typed_action_unparsed_text")
    allowed = set(strings(task.get("allowed_effect_paths")))
    if not actions:
        faults.append("typed_actions_missing")
    if len(actions) > int(protocol.get("maximum_actions_per_candidate") or 0):
        faults.append("typed_action_count_exceeded")
    if sum(len(row["replacement"].encode()) for row in actions) > int(protocol.get("maximum_replacement_bytes") or 0):
        faults.append("typed_replacement_budget_exceeded")
    if any(row["path"] not in allowed or unsafe_relative_path(row["path"]) for row in actions):
        faults.append("typed_action_path_unauthorized")
    if any(row["end_line"] < row["start_line"] for row in actions):
        faults.append("typed_action_line_range_invalid")
    return (actions if not faults else []), sorted(set(faults))


def apply_actions(root: Path, actions: list[dict[str, Any]]) -> list[str]:
    faults: list[str] = []
    pending: dict[Path, list[str]] = {}
    for index, action in enumerate(actions):
        path = root / str(action.get("path") or "")
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            faults.append(f"action_{index}_target_missing")
            continue
        if root.resolve() not in resolved.parents or not resolved.is_file() or resolved.is_symlink():
            faults.append(f"action_{index}_target_unsafe")
            continue
        lines = pending.get(path, path.read_text(encoding="utf-8").splitlines())
        start, end = int(action["start_line"]), int(action["end_line"])
        if start < 1 or end > len(lines):
            faults.append(f"action_{index}_line_range_out_of_bounds")
            continue
        replacement = str(action.get("replacement") or "").splitlines()
        pending[path] = lines[: start - 1] + replacement + lines[end:]
    if faults:
        return faults
    for path, lines in pending.items():
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return faults


def render_visible_context(root: Path, task: dict[str, Any]) -> str:
    context = mapping(task.get("candidate_visible_context"))
    blocks: list[str] = []
    for row in dicts(context.get("searches")):
        literal = str(row.get("literal") or "")
        paths = strings(row.get("paths"))
        matches: list[str] = []
        for relative in paths:
            target = checked_source_path(root, relative)
            for number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
                if literal in line:
                    matches.append(f"{relative}:{number}:{line}")
        blocks.append(f"[SEARCH literal={json.dumps(literal)}]\n" + "\n".join(matches[:80]))
    for row in dicts(context.get("reads")):
        relative = str(row.get("path") or "")
        target = checked_source_path(root, relative)
        lines = target.read_text(encoding="utf-8").splitlines()
        start = max(1, int(row.get("start_line") or 1))
        end = min(len(lines), int(row.get("end_line") or len(lines)))
        blocks.append(f"[READ {relative}:{start}-{end}]\n" + "\n".join(lines[start - 1:end]))
    if not blocks:
        raise InstrumentFault("candidate_visible_context_empty")
    return "\n\n".join(blocks)


def render_candidate_prompt(request: str, context: str, protocol: dict[str, Any]) -> str:
    return (
        f"Implement this repository task: {request}\n\n"
        "You have only the source/search results below. Return only compact typed edits in this exact grammar:\n"
        f"{protocol.get('grammar')}\n"
        "Line numbers are 1-based and inclusive. Do not return prose, JSON, Markdown, or a Git diff.\n\n"
        f"[candidate_visible_repository_context]\n{context}"
    )


def render_repair_prompt(original: str, feedback: str) -> str:
    return (
        f"{original}\n\n[visible_verifier_feedback]\n{feedback}\n\n"
        "Return a repair as a complete THESEUS_EDIT_V1 operation set against the current disposable snapshot."
    )


def run_visible_verifier(root: Path, task: dict[str, Any]) -> dict[str, Any]:
    verifier = mapping(task.get("visible_verifier"))
    command = strings(verifier.get("command"))
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command, cwd=root, text=True, capture_output=True,
            timeout=max(1, int(verifier.get("timeout_seconds") or 60)),
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return {
            "passed": result.returncode == 0,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
            "runtime_ms": round((time.perf_counter() - started) * 1000, 3),
            "command_sha256": stable_hash(command),
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "returncode": 124, "timed_out": True, "stdout_tail": "", "stderr_tail": ""}


def bounded_feedback(parse_faults: list[str], apply_faults: list[str], verification: dict[str, Any]) -> str:
    return json.dumps({
        "parse_faults": sorted(set(parse_faults)),
        "apply_faults": sorted(set(apply_faults)),
        "verifier_returncode": verification.get("returncode"),
        "verifier_stdout_tail": str(verification.get("stdout_tail") or "")[-1200:],
        "verifier_stderr_tail": str(verification.get("stderr_tail") or "")[-1200:],
    }, sort_keys=True)


def pair_receipt(attempts: list[dict[str, Any]], session: Any) -> dict[str, Any]:
    direct = next(row for row in attempts if row["arm_id"] == route_integrity.DIRECT_MODE)
    integrated = next(row for row in attempts if row["arm_id"] == route_integrity.INTEGRATED_MODE)
    route_pair = route_integrity.compare_matched_pair(
        mapping(direct.get("route_integrity")), mapping(integrated.get("route_integrity"))
    )
    identities = {str(row.get("model_identity_sha256") or "") for row in attempts}
    load_ready = session.model_load_count == 1
    return {
        "ready": route_pair.get("ready") is True and len(identities) == 1 and "" not in identities and load_ready,
        "route_pair": route_pair,
        "same_model_identity": len(identities) == 1 and "" not in identities,
        "persistent_model_load_count": session.model_load_count,
        "persistent_model_load_ready": load_ready,
        "persistent_inference_calls": session.inference_calls,
    }


def extract_source_archive(archive: Path, destination: Path, source_root: str = "") -> None:
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as handle:
        members = handle.getmembers()
        for member in members:
            relative = strip_source_root(member.name, source_root)
            if relative is None:
                continue
            target = (destination / relative).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise InstrumentFault("unsafe_source_archive_member")
            if member.issym() or member.islnk():
                raise InstrumentFault("source_archive_links_forbidden")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise InstrumentFault("source_archive_special_member_forbidden")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = handle.extractfile(member)
            if source is None:
                raise InstrumentFault("source_archive_member_unreadable")
            target.write_bytes(source.read())
            target.chmod(member.mode & 0o777)


def strip_source_root(member_name: str, source_root: str) -> str | None:
    name = member_name.strip("/")
    root = source_root.strip("/")
    if not root:
        return name
    if name == root:
        return None
    prefix = root + "/"
    if not name.startswith(prefix):
        raise InstrumentFault("source_archive_member_outside_declared_root")
    relative = name[len(prefix):]
    return relative or None


def checked_source_path(root: Path, relative: str) -> Path:
    if unsafe_relative_path(relative):
        raise InstrumentFault("unsafe_relative_source_path")
    path = (root / relative).resolve(strict=True)
    if root.resolve() not in path.parents or not path.is_file() or path.is_symlink():
        raise InstrumentFault("source_path_not_regular_file")
    return path


def unsafe_relative_path(value: str) -> bool:
    path = Path(value)
    return path.is_absolute() or ".." in path.parts or not value or "\\" in value


def inventory(root: Path) -> dict[str, str]:
    ignored_parts = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"}
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and not ignored_parts.intersection(path.parts)
        and path.name != ".coverage"
    }


def changed_paths(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))


def invalid_report(instrument: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": POLICY, "created_utc": now(), "trigger_state": "RED",
        "faults": ["instrument_or_task_audit_red"],
        "instrument_audit": instrument, "task_audit": task, "counters": zero_counters(),
    }


def compact_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": report.get("policy"), "trigger_state": report.get("trigger_state"),
        "faults": report.get("faults"), "denominators": report.get("denominators"),
        "instrument_disposition": report.get("instrument_disposition"),
        "matched_pair_ready": mapping(report.get("matched_pair")).get("ready"),
    }


def zero_counters() -> dict[str, int]:
    return {
        "external_inference_calls": 0, "teacher_calls": 0,
        "public_calibration_cases_consumed": 0, "D2_cases_consumed": 0,
        "public_training_rows_written": 0, "fallback_return_count": 0,
        "user_facing_effects": 0,
    }


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def stable_hash(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def strings(value: Any) -> list[str]:
    return [str(row) for row in value if isinstance(row, str) and row] if isinstance(value, list) else []


def safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._") or "task"


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: str | Path) -> str:
    candidate = resolve(path)
    try:
        return candidate.relative_to(ROOT).as_posix()
    except ValueError:
        return str(candidate)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
