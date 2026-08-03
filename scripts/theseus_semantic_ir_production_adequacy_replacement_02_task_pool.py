#!/usr/bin/env python3
"""Seal the candidate-visible packet for the independently qualified Task 2 replacement."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import theseus_assistant_p2a as p2a
import theseus_p4_cognitive_compilation as p4
import theseus_semantic_ir_production as semantic
import theseus_semantic_ir_production_canary as canary
import theseus_semantic_ir_production_adequacy_runtime as production


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPORT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_replacement_02_source.json"
SOURCE_REPORT_SHA256 = "b6a92bd9c8d73eae5f48f1ddf47eac8abce80a9b0dba926bcb18d3df27382680"
EVALUATOR_REPORT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_replacement_02_evaluator.json"
EVALUATOR_REPORT_SHA256 = "dafacdca124b3849a808e21c0f9bf23ab57bb8180f28b3f5a8dbf77d7a10ee8e"
TASK_PATH = ROOT / "configs" / "theseus_semantic_ir_production_adequacy_replacement_02_task.json"
PACKET_PATH = ROOT / "configs" / "theseus_semantic_ir_production_adequacy_replacement_02_candidate_packet.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_replacement_02_task_pool.json"
POLICY = "project_theseus_semantic_ir_production_adequacy_replacement_task_pool_v1"
PACKET_POLICY = "project_theseus_semantic_ir_production_adequacy_candidate_packet_v1"
MODEL_CONTEXT_TOKENS = 262_144
FORBIDDEN_KEYS = {
    "category",
    "solution",
    "solution_expr",
    "solution_body",
    "tests",
    "hidden_tests",
    "expected",
    "answer",
    "source_task_id",
    "benchmark_card_label",
    "answer_family",
    "return_shape",
    "type_family",
    "required_constructs",
    "target_revision",
    "evaluator",
    "oracle",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=p2a.rel(DEFAULT_OUT))
    args = parser.parse_args()
    report = materialize()
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def materialize() -> dict[str, Any]:
    faults = binding_faults()
    source_report = p2a.read_json(SOURCE_REPORT)
    evaluator_report = p2a.read_json(EVALUATOR_REPORT)
    if (
        source_report.get("trigger_state") != "GREEN"
        or source_report.get("source_pair_admitted") is not True
        or source_report.get("candidate_packet_materialized") is not False
    ):
        faults.append("replacement_source_not_green")
    if (
        evaluator_report.get("trigger_state") != "GREEN"
        or evaluator_report.get("state") != "QUALIFIED_BEFORE_CANDIDATE_PACKET_MATERIALIZATION"
        or evaluator_report.get("candidate_packet_materialized") is not False
    ):
        faults.append("replacement_evaluator_not_green")
    task: dict[str, Any] = {}
    packet: dict[str, Any] = {}
    prompt = ""
    symbol_count = 0
    if not faults:
        task, packet, prompt, symbol_count, build_faults = build_packet(source_report)
        faults.extend(build_faults)
    sealed = not faults
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if sealed else "RED",
        "state": "SEALED_BEFORE_REPLACEMENT_CANDIDATE_GENERATION" if sealed else "INVALID_NOT_SEALED",
        "source_report": artifact(SOURCE_REPORT),
        "evaluator_qualification": artifact(EVALUATOR_REPORT),
        "task_pool_owner": artifact(Path(__file__).resolve()),
        "replacement_index": 2,
        "opaque_task_id": task.get("opaque_task_id"),
        "task_manifest": p2a.rel(TASK_PATH),
        "task_manifest_sha256": p2a.sha256_file(TASK_PATH),
        "candidate_packet": p2a.rel(PACKET_PATH),
        "candidate_packet_sha256": p2a.sha256_file(PACKET_PATH),
        "serialized_prompt_sha256": p2a.sha256_text(prompt),
        "serialized_prompt_utf8_bytes": len(prompt.encode("utf-8")),
        "conservative_minimum_residual_tokens": MODEL_CONTEXT_TOKENS - len(prompt.encode("utf-8")),
        "semantic_scope_node_count": symbol_count,
        "target_archive_or_source_visible": False,
        "evaluator_identity_or_output_visible": False,
        "faults": sorted(set(faults)),
        "information_flow": {
            "candidate_receives_exact_serialized_packet_only": True,
            "repository_pr_target_revision_target_archive_evaluator_or_oracle_candidate_visible": False,
            "recursive_forbidden_key_audit_required": True,
            "actual_serialized_prompt_audited": True,
            "candidate_emitted_integrity_flags_trusted": False,
        },
        "completion_boundary": {
            "model_declared_context_window_tokens": MODEL_CONTEXT_TOKENS,
            "project_selected_quality_token_cap": None,
            "normal_completion": ["parser_complete", "model_eos"],
            "physical_context_boundary_hit_invalidates_observation": True,
            "host_watchdog_activation_invalidates_observation": True,
        },
        "counters": zero_counters(),
        "maximum_inference": (
            "A GREEN seal establishes only that the independently qualified fresh replacement "
            "has a production-representable, parent-source-only prompt with recursive "
            "anti-cheating checks and physical context headroom. It does not establish model "
            "competence, Semantic-IR adequacy or effect, D1, D2, training value, serving, or "
            "book support."
        ),
    }


def build_packet(source_report: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str, int, list[str]]:
    faults: list[str] = []
    archives = p2a.mapping(source_report.get("archives"))
    parent = p2a.mapping(archives.get("parent"))
    target = p2a.mapping(archives.get("target"))
    metadata = p2a.mapping(source_report.get("metadata"))
    selected = ["business_logic_test.py"]
    task = {
        "policy": p4.TASK_POLICY,
        "state": "SEALED_BEFORE_CANDIDATE_GENERATION",
        "opaque_task_id": "semantic-ir-adequacy-02r1",
        "campaign_index": 2,
        "partition": "semantic_ir_production_implementation_adequacy",
        "family": "licensed_source_disjoint_python_causal_repair",
        "natural_request": (
            "Make the fixed-amount discount conformance assertion compare the applied amount "
            "against the configured expected discount rather than a hard-coded constant. "
            "Preserve assertion semantics and unrelated tests. Modify only business_logic_test.py."
        ),
        "source_archive": str(parent.get("path") or ""),
        "source_archive_sha256": str(parent.get("sha256") or ""),
        "source_archive_root": str(parent.get("root") or ""),
        "source_provenance": {
            "url": "https://github.com/Universal-Commerce-Protocol/conformance",
            "revision": str(metadata.get("parent_revision") or ""),
            "license_spdx": "Apache-2.0",
        },
        "obligations": [
            {
                "id": "O1",
                "kind": "require",
                "text": "The applied fixed-discount amount is compared with the configured expected discount value.",
            },
            {
                "id": "O2",
                "kind": "preserve",
                "text": "Assertion semantics, the configured-discount dataflow, and unrelated tests remain intact.",
            },
            {
                "id": "O3",
                "kind": "non_goal",
                "text": "Modify only the declared allowed effect path; do not alter unrelated behavior or files.",
            },
        ],
        "obligation_dependencies": [{"before": "O2", "after": "O1"}],
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
            "version": production.HEADER,
            "maximum_symbol_nodes": 1_000_000,
            "maximum_semantic_scope_nodes": 1_000_000,
            "create_file_allowed_only_for_declared_missing_effect_paths": True,
            "role_partition_source_target_loss_and_dependency_identity_required": True,
        },
        "effect_authority": "disposable_snapshot_only",
        "maximum_inference": "One implementation-adequacy observation only; no treatment, D1, D2, serving, training, or book-support claim.",
    }
    with tempfile.TemporaryDirectory(prefix="theseus-adequacy-02r1-packet-") as directory:
        root = Path(directory) / "source"
        p2a.extract_source_archive(
            p2a.resolve(task["source_archive"]), root, task["source_archive_root"]
        )
        source_path = p2a.checked_source_path(root, selected[0])
        task["candidate_visible_context"]["reads"] = [
            {"path": selected[0], "start_line": 1, "end_line": len(source_path.read_text(encoding="utf-8").splitlines())}
        ]
        p2a.write_json(TASK_PATH, task)
        task_audit = p4.audit_task(TASK_PATH)
        if task_audit.get("trigger_state") != "GREEN":
            faults.extend(p2a.strings(task_audit.get("faults")))
        symbols = production.semantic_scope_symbol_table(root, task)
        symbol_count = len(p2a.dicts(symbols.get("nodes")))
        task["semantic_ir_contract"]["maximum_symbol_nodes"] = symbol_count
        task["semantic_ir_contract"]["maximum_semantic_scope_nodes"] = symbol_count
        p2a.write_json(TASK_PATH, task)
        common = canary.render_common_context(root, task, symbols)
        prompt = production.render_prompt(task, common)
    packet = {
        "policy": PACKET_POLICY,
        "state": "SEALED_BEFORE_CANDIDATE_GENERATION",
        "opaque_task_id": task["opaque_task_id"],
        "serialized_prompt": prompt,
    }
    faults.extend(audit_packet(packet, source_report, target))
    if len(prompt.encode("utf-8")) >= MODEL_CONTEXT_TOKENS:
        faults.append("physical_context_residual_not_proven_by_utf8_upper_bound")
    p2a.write_json(PACKET_PATH, packet)
    return task, packet, prompt, symbol_count, faults


def audit_packet(packet: dict[str, Any], source_report: dict[str, Any], target: dict[str, Any]) -> list[str]:
    faults: list[str] = []
    for path, key in recursive_keys(packet):
        if key.lower() in FORBIDDEN_KEYS:
            faults.append(f"forbidden_candidate_key:{path}")
    serialized = json.dumps(packet, sort_keys=True)
    metadata = p2a.mapping(source_report.get("metadata"))
    forbidden_values = [
        str(metadata.get("parent_revision") or ""),
        str(metadata.get("target_revision") or ""),
        str(metadata.get("merge_revision") or ""),
        str(metadata.get("pull_request_url") or ""),
        str(target.get("path") or ""),
        str(target.get("sha256") or ""),
        p2a.rel(EVALUATOR_REPORT),
        EVALUATOR_REPORT_SHA256,
    ]
    for value in forbidden_values:
        if value and value.lower() in serialized.lower():
            faults.append("forbidden_candidate_value:" + hashlib.sha256(value.encode()).hexdigest()[:12])
    if not str(packet.get("serialized_prompt") or ""):
        faults.append("candidate_prompt_missing")
    return sorted(set(faults))


def recursive_keys(value: Any, prefix: str = "$") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            path = f"{prefix}.{key}"
            rows.append((path, str(key)))
            rows.extend(recursive_keys(nested, path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            rows.extend(recursive_keys(nested, f"{prefix}[{index}]"))
    return rows


def binding_faults() -> list[str]:
    faults: list[str] = []
    for path, expected, label in (
        (SOURCE_REPORT, SOURCE_REPORT_SHA256, "source_report"),
        (EVALUATOR_REPORT, EVALUATOR_REPORT_SHA256, "evaluator_report"),
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
        "external_inference_calls": 0,
        "teacher_calls": 0,
        "training_rows_written": 0,
        "D1_cases_consumed": 0,
        "D2_cases_consumed": 0,
    }


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": report.get("trigger_state"),
        "state": report.get("state"),
        "opaque_task_id": report.get("opaque_task_id"),
        "serialized_prompt_utf8_bytes": report.get("serialized_prompt_utf8_bytes"),
        "conservative_minimum_residual_tokens": report.get("conservative_minimum_residual_tokens"),
        "faults": report.get("faults"),
        "counters": report.get("counters"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
