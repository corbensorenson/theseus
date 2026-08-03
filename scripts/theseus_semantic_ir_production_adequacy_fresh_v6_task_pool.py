#!/usr/bin/env python3
"""Seal a uniform fresh compact-protocol 18-task adequacy denominator."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import theseus_assistant_p2a as p2a
import theseus_p4_cognitive_compilation as p4
import theseus_semantic_ir_production_adequacy_fresh_v4_task_pool as prior_owner
import theseus_semantic_ir_production_adequacy_replacement_02_task_pool as packet_audit
import theseus_semantic_ir_production_adequacy_runtime_v5 as runtime


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = ROOT / "configs" / "theseus_semantic_ir_production_adequacy_fresh_v6_sources.json"
SOURCE_REPORT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v6_sources.json"
SOURCE_REPORT_SHA256 = "229af6cd74682117be4f50b179620ad79c1c070e0b63eca37a2a881b991b1403"
FRESH_EVALUATOR = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v6_evaluator.json"
FRESH_EVALUATOR_SHA256 = "2fc4ef949caae9bc55058f276250254c6f6d76963e3592d6dbb4a852fc8c5cdb"
PRIOR_POOL = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v5_task_pool.json"
PRIOR_POOL_SHA256 = "7fb283a784322355b891e6e8ab48049010a48aac05bf5401ceda21d1153d59f2"
RUNTIME_PATH = ROOT / "scripts" / "theseus_semantic_ir_production_adequacy_runtime_v5.py"
RUNTIME_SHA256 = "626462aadc8a765d1ab7520a3d063d87dbf9f530836feced540047bf9629357b"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v6_task_pool.json"
POLICY = "project_theseus_semantic_ir_production_adequacy_fresh_v6_task_pool_v1"
PACKET_POLICY = "project_theseus_semantic_ir_production_adequacy_candidate_packet_v3_compact_128bit"
MODEL_CONTEXT_TOKENS = 262_144
FORBIDDEN_KEYS = packet_audit.FORBIDDEN_KEYS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=p2a.rel(DEFAULT_OUT))
    args = parser.parse_args()
    report = materialize_pool()
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def materialize_pool() -> dict[str, Any]:
    faults = binding_faults()
    source_config = p2a.read_json(SOURCE_CONFIG)
    source_report = p2a.read_json(SOURCE_REPORT)
    evaluator = p2a.read_json(FRESH_EVALUATOR)
    prior_pool = p2a.read_json(PRIOR_POOL)
    if (
        source_report.get("trigger_state") != "GREEN"
        or source_report.get("source_pairs_admitted") is not True
    ):
        faults.append("fresh_source_report_not_green")
    if (
        evaluator.get("trigger_state") != "GREEN"
        or evaluator.get("candidate_packet_materialized") is not False
        or int(evaluator.get("green_task_count") or 0) != 4
    ):
        faults.append("fresh_evaluator_not_green_before_packet")
    if prior_pool.get("trigger_state") != "GREEN" or int(prior_pool.get("task_count") or 0) != 18:
        faults.append("prior_pool_not_green")
    specs = {
        int(row.get("index") or 0): row
        for row in p2a.dicts(source_config.get("sources"))
    }
    fresh_rows = {
        int(row.get("index") or 0): row
        for row in p2a.dicts(source_report.get("rows"))
    }
    prior_rows = {
        int(row.get("index") or 0): row
        for row in p2a.dicts(prior_pool.get("rows"))
    }
    prior_denominator = {
        int(row.get("index") or 0): row
        for row in p2a.dicts(prior_pool.get("source_denominator"))
    }
    rows: list[dict[str, Any]] = []
    denominator: list[dict[str, Any]] = []
    if not faults:
        for index in range(1, 19):
            if index <= 4:
                source_row = fresh_rows[index]
                task = fresh_task(index, source_row, specs[index])
            else:
                source_row = prior_denominator[index]
                prior_task = p2a.resolve(str(prior_rows[index].get("task_manifest") or ""))
                if not prior_task.is_file() or p2a.sha256_file(prior_task) != prior_rows[index].get("task_manifest_sha256"):
                    faults.append(f"task_{index:02d}:prior_task_binding_invalid")
                    continue
                task = copy.deepcopy(p2a.read_json(prior_task))
            receipt, denominator_row, row_faults = build_row(index, source_row, task)
            rows.append(receipt)
            denominator.append(denominator_row)
            faults.extend(f"task_{index:02d}:{fault}" for fault in row_faults)
    repositories = [str(row.get("repository") or "") for row in denominator]
    strata = Counter(str(row.get("stratum") or "") for row in denominator)
    if len(repositories) != 18 or len(set(repositories)) != 18:
        faults.append("repository_source_disjointness_invalid")
    if len(strata) != 6 or any(count != 3 for count in strata.values()):
        faults.append("stratum_balance_invalid")
    sealed = (
        not faults
        and len(rows) == 18
        and all(row.get("trigger_state") == "GREEN" for row in rows)
    )
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if sealed else "RED",
        "state": "SEALED_FRESH_V6_UNIFORM_COMPACT_DENOMINATOR_BEFORE_CANDIDATE_GENERATION"
        if sealed
        else "INVALID_NOT_SEALED",
        "fresh_source_report": artifact(SOURCE_REPORT),
        "fresh_evaluator_qualification": artifact(FRESH_EVALUATOR),
        "prior_unexposed_pool": artifact(PRIOR_POOL),
        "compact_statement_runtime": artifact(RUNTIME_PATH),
        "task_count": len(rows),
        "sealed_packet_count": sum(row.get("trigger_state") == "GREEN" for row in rows),
        "repository_count": len(set(repositories)),
        "stratum_counts": dict(sorted(strata.items())),
        "replacement_indices": [1, 2, 3, 4],
        "uniformly_rebound_unexposed_indices": list(range(5, 19)),
        "consumed_v5_prompt_or_candidate_reused": False,
        "rows": rows,
        "source_denominator": denominator,
        "faults": sorted(set(faults)),
        "information_flow": {
            "candidate_receives_exact_serialized_prompt_only": True,
            "parent_source_only": True,
            "target_source_archive_revision_pr_evaluator_or_oracle_candidate_visible": False,
            "recursive_forbidden_key_audit_required": True,
            "candidate_emitted_integrity_flags_trusted": False,
            "uniform_compact_protocol_for_all_tasks": True,
        },
        "completion_boundary": {
            "model_declared_context_window_tokens": MODEL_CONTEXT_TOKENS,
            "project_selected_quality_token_cap": None,
            "addressability_measure": "exact frozen tokenizer chat-template token count",
            "normal_completion": ["parser_complete", "model_eos"],
            "physical_context_boundary_hit_invalidates_observation": True,
            "host_watchdog_activation_invalidates_observation": True,
        },
        "authority": {
            "local_model_calls_authorized_after_green_campaign_binding": 36,
            "external_inference_authorized": False,
            "hidden_evaluation_authorized": False,
            "teacher_calls_authorized": False,
            "training_rows_authorized": False,
            "D1_authorized": False,
            "D2_authorized": False,
            "book_support_promotion_authorized": False,
            "user_or_operator_gate": False,
        },
        "counters": zero_counters(),
        "maximum_inference": "A GREEN seal establishes only a fresh 18-repository, six-stratum, parent-only, uniform compact-protocol denominator with independently qualified evaluators and exact physical-context addressability. It authorizes no model call until a separate campaign binding is GREEN and provides no host-operability, implementation-adequacy, subsystem-effect, D1, D2, training, serving, or book-support evidence.",
    }


def fresh_task(index: int, source_row: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    parent = p2a.mapping(p2a.mapping(source_row.get("archives")).get("parent"))
    selected = p2a.strings(source_row.get("selected_source_paths"))
    return {
        "policy": p4.TASK_POLICY,
        "state": "SEALED_BEFORE_CANDIDATE_GENERATION",
        "opaque_task_id": spec.get("opaque_task_id"),
        "campaign_index": index,
        "partition": "semantic_ir_production_implementation_adequacy",
        "family": "licensed_source_disjoint_python_causal_repair",
        "natural_request": spec.get("natural_request"),
        "source_archive": parent.get("path"),
        "source_archive_sha256": parent.get("sha256"),
        "source_archive_root": parent.get("root"),
        "source_provenance": {
            "url": "https://github.com/" + str(source_row.get("repository") or ""),
            "revision": source_row.get("parent_revision"),
            "license_spdx": source_row.get("license_spdx"),
        },
        "obligations": spec.get("obligations"),
        "obligation_dependencies": [{"before": "O1", "after": "O2"}],
        "allowed_effect_paths": selected,
        "candidate_visible_context": {
            "reads": [],
            "searches": [],
            "full_selected_parent_sources": True,
            "missing_allowed_effect_paths": [],
            "project_selected_character_or_token_cap": None,
        },
        "visible_verifier": {
            "command": [
                "python3",
                "-c",
                "import ast,sys; [ast.parse(open(p, encoding='utf-8').read(), filename=p) for p in sys.argv[1:]]",
                *selected,
            ],
            "timeout_seconds": 60,
            "answer_specific": False,
            "candidate_prompt_visibility": False,
        },
        "visible_feedback_map": [
            {"marker": "SyntaxError", "obligation_ids": ["O1", "O2", "O3"]}
        ],
        "semantic_ir_contract": {
            "version": runtime.HEADER,
            "maximum_symbol_nodes": 1_000_000,
            "maximum_semantic_scope_nodes": 1_000_000,
            "create_file_allowed_only_for_declared_missing_effect_paths": True,
            "role_partition_source_target_loss_and_dependency_identity_required": True,
        },
        "effect_authority": "disposable_snapshot_only",
        "maximum_inference": "One implementation-adequacy observation only; no treatment, D1, D2, serving, training, or book-support claim.",
    }


def build_row(
    index: int, source_row: dict[str, Any], task: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    faults: list[str] = []
    archives = p2a.mapping(source_row.get("archives"))
    parent = p2a.mapping(archives.get("parent"))
    target = p2a.mapping(archives.get("target"))
    selected = p2a.strings(source_row.get("selected_source_paths"))
    task["state"] = "SEALED_BEFORE_CANDIDATE_GENERATION"
    task["source_archive"] = parent.get("path")
    task["source_archive_sha256"] = parent.get("sha256")
    task["source_archive_root"] = parent.get("root")
    contract = p2a.mapping(task.get("semantic_ir_contract"))
    contract.update(
        {
            "version": runtime.HEADER,
            "maximum_symbol_nodes": 1_000_000,
            "maximum_semantic_scope_nodes": 1_000_000,
            "statement_scope_policy": runtime.STATEMENT_SCOPE_POLICY,
            "inventory_truncation_allowed": False,
            "node_handle_bits": runtime.NODE_HANDLE_HEX_BITS,
            "candidate_copies_full_node_sha256": False,
            "full_node_sha256_resolved_independently": True,
        }
    )
    task["semantic_ir_contract"] = contract
    task_path = ROOT / "configs" / f"theseus_semantic_ir_production_adequacy_fresh_v6_task_{index:02d}.json"
    packet_path = ROOT / "configs" / f"theseus_semantic_ir_production_adequacy_fresh_v6_candidate_packet_{index:02d}.json"
    with tempfile.TemporaryDirectory(prefix=f"theseus-fresh-v6-packet-{index:02d}-") as directory:
        root = Path(directory) / "source"
        p2a.extract_source_archive(
            p2a.resolve(str(task.get("source_archive") or "")),
            root,
            str(task.get("source_archive_root") or ""),
        )
        missing = set(
            p2a.strings(
                p2a.mapping(task.get("candidate_visible_context")).get(
                    "missing_allowed_effect_paths"
                )
            )
        )
        reads = []
        for path in selected:
            if path in missing:
                continue
            source_path = p2a.checked_source_path(root, path)
            reads.append(
                {
                    "path": path,
                    "start_line": 1,
                    "end_line": len(source_path.read_text(encoding="utf-8").splitlines()),
                }
            )
        visible = p2a.mapping(task.get("candidate_visible_context"))
        visible.update(
            {
                "reads": reads,
                "searches": [],
                "full_selected_parent_sources": True,
                "project_selected_character_or_token_cap": None,
            }
        )
        task["candidate_visible_context"] = visible
        p2a.write_json(task_path, task)
        audit = p4.audit_task(task_path)
        if audit.get("trigger_state") != "GREEN":
            faults.extend(p2a.strings(audit.get("faults")))
        symbols = runtime.semantic_scope_symbol_table(root, task)
        node_count = len(p2a.dicts(symbols.get("nodes")))
        task["semantic_ir_contract"]["maximum_symbol_nodes"] = node_count
        task["semantic_ir_contract"]["maximum_semantic_scope_nodes"] = node_count
        p2a.write_json(task_path, task)
        prompt = runtime.render_prompt(
            task, runtime.render_common_context(root, task, symbols)
        )
    packet = {
        "policy": PACKET_POLICY,
        "state": "SEALED_BEFORE_CANDIDATE_GENERATION",
        "opaque_task_id": task.get("opaque_task_id"),
        "serialized_prompt": prompt,
    }
    faults.extend(audit_packet(packet, source_row, target))
    prompt_bytes = len(prompt.encode("utf-8"))
    try:
        prompt_tokens = prior_owner.exact_prompt_tokens(prompt)
    except Exception as exc:
        prompt_tokens = 0
        faults.append(f"exact_frozen_tokenizer_unavailable:{type(exc).__name__}")
    residual = MODEL_CONTEXT_TOKENS - prompt_tokens
    if prompt_tokens <= 0 or residual <= 0:
        faults.append("exact_physical_context_residual_invalid")
    p2a.write_json(packet_path, packet)
    metrics = p2a.mapping(symbols.get("statement_scope_metrics"))
    receipt = {
        "index": index,
        "opaque_task_id": task.get("opaque_task_id"),
        "task_manifest": p2a.rel(task_path),
        "task_manifest_sha256": p2a.sha256_file(task_path),
        "candidate_packet": p2a.rel(packet_path),
        "candidate_packet_sha256": p2a.sha256_file(packet_path),
        "serialized_prompt_sha256": p2a.sha256_text(prompt),
        "serialized_prompt_utf8_bytes": prompt_bytes,
        "exact_prompt_tokens": prompt_tokens,
        "exact_context_residual_tokens": residual,
        "semantic_scope_node_count": node_count,
        "statement_scope_metrics": metrics,
        "compact_integrity_abi": symbols.get("compact_integrity_abi"),
        "target_archive_or_source_visible": False,
        "evaluator_identity_or_output_visible": False,
        "faults": sorted(set(faults)),
        "trigger_state": "GREEN" if not faults else "RED",
    }
    denominator = {
        "index": index,
        "repository": source_row.get("repository"),
        "parent_revision": source_row.get("parent_revision"),
        "target_revision": source_row.get("target_revision"),
        "stratum": source_row.get("stratum"),
        "selected_source_paths": selected,
        "archives": source_row.get("archives"),
    }
    return receipt, denominator, faults


def audit_packet(
    packet: dict[str, Any], source_row: dict[str, Any], target: dict[str, Any]
) -> list[str]:
    faults: list[str] = []
    for path, key in packet_audit.recursive_keys(packet):
        if key.lower() in FORBIDDEN_KEYS:
            faults.append(f"forbidden_candidate_key:{path}")
    serialized = json.dumps(packet, sort_keys=True)
    forbidden_values = [
        str(source_row.get("target_revision") or ""),
        str(source_row.get("merge_revision") or ""),
        str(source_row.get("pull_request_url") or ""),
        str(target.get("path") or ""),
        str(target.get("sha256") or ""),
        p2a.rel(FRESH_EVALUATOR),
        FRESH_EVALUATOR_SHA256,
    ]
    for value in forbidden_values:
        if value and value.lower() in serialized.lower():
            faults.append(
                "forbidden_candidate_value:"
                + hashlib.sha256(value.encode()).hexdigest()[:12]
            )
    if not str(packet.get("serialized_prompt") or ""):
        faults.append("candidate_prompt_missing")
    return sorted(set(faults))


def binding_faults() -> list[str]:
    faults: list[str] = []
    for path, expected, label in (
        (SOURCE_REPORT, SOURCE_REPORT_SHA256, "source_report"),
        (FRESH_EVALUATOR, FRESH_EVALUATOR_SHA256, "fresh_evaluator"),
        (PRIOR_POOL, PRIOR_POOL_SHA256, "prior_pool"),
        (RUNTIME_PATH, RUNTIME_SHA256, "compact_runtime"),
    ):
        if not path.is_file() or p2a.sha256_file(path) != expected:
            faults.append(f"binding_invalid:{label}")
    return faults


def artifact(path: Path) -> dict[str, str]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)}


def zero_counters() -> dict[str, int]:
    return {
        "candidate_or_control_calls": 0,
        "local_model_calls": 0,
        "hidden_evaluator_executions": 0,
        "external_inference_calls": 0,
        "teacher_calls": 0,
        "training_rows_written": 0,
        "D1_cases_consumed": 0,
        "D2_cases_consumed": 0,
    }


def summary(report: dict[str, Any]) -> dict[str, Any]:
    rows = p2a.dicts(report.get("rows"))
    return {
        "trigger_state": report.get("trigger_state"),
        "state": report.get("state"),
        "task_count": report.get("task_count"),
        "sealed_packet_count": report.get("sealed_packet_count"),
        "repository_count": report.get("repository_count"),
        "stratum_counts": report.get("stratum_counts"),
        "maximum_exact_prompt_tokens": max(
            (int(row.get("exact_prompt_tokens") or 0) for row in rows), default=0
        ),
        "minimum_exact_context_residual_tokens": min(
            (int(row.get("exact_context_residual_tokens") or 0) for row in rows),
            default=0,
        ),
        "faults": report.get("faults"),
        "counters": report.get("counters"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
