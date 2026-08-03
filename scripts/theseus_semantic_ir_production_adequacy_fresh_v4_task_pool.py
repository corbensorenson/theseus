#!/usr/bin/env python3
"""Seal the fresh 18-task statement-granular adequacy denominator."""

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
import theseus_local_inference_backend as local_backend
import theseus_p4_cognitive_compilation as p4
import theseus_semantic_ir_production_canary as canary
import theseus_semantic_ir_production_adequacy_replacement_02_task_pool as pool02
import theseus_semantic_ir_production_adequacy_runtime_v4 as runtime_v4


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = ROOT / "configs" / "theseus_semantic_ir_production_adequacy_fresh_v4_sources.json"
SOURCE_REPORT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v4_sources.json"
SOURCE_REPORT_SHA256 = "fb8bbe4446dec871db53c04aae0e83a53057df39988e102c707dc9ac27496b37"
FRESH_EVALUATOR = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v4_evaluator.json"
FRESH_EVALUATOR_SHA256 = "f3ae736ecae7be1c559181b29b862a90fac0ccd7dbfbba8daa216654b31a9662"
MATERIALIZATION = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_materialization_v4.json"
MATERIALIZATION_SHA256 = "7572e6ebb82ae6b16575298c42450a31d7c50ce2823fd5fc6346b12d6216f122"
BASE_EVALUATOR = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_evaluator_qualification.json"
BASE_EVALUATOR_SHA256 = "448544a147595413b0d8d0c7523d9442571651a7f11b49d38b9e5a5c9eb9c35a"
RUNTIME_PATH = ROOT / "scripts" / "theseus_semantic_ir_production_adequacy_runtime_v4.py"
RUNTIME_SHA256 = "a3d42c6816fe2964a223d7d1209fa12122c06a4ce0c73bcbb48f11d77936624c"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v4_task_pool.json"
POLICY = "project_theseus_semantic_ir_production_adequacy_fresh_v4_task_pool_v1"
PACKET_POLICY = "project_theseus_semantic_ir_production_adequacy_candidate_packet_v2_statement_granular"
MODEL_CONTEXT_TOKENS = 262_144
FORBIDDEN_KEYS = pool02.FORBIDDEN_KEYS
MODEL_CARD = {
    "repo_id": "mlx-community/Tmax-9B-MLX-8bit",
    "revision": "33812d6cf04f88856f25eb828de4f3144a194560",
    "chat_template_kwargs": {"enable_thinking": False},
}
_TOKENIZER: Any | None = None


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
    fresh_evaluator = p2a.read_json(FRESH_EVALUATOR)
    materialization = p2a.read_json(MATERIALIZATION)
    base_evaluator = p2a.read_json(BASE_EVALUATOR)
    if source_report.get("trigger_state") != "GREEN" or source_report.get("source_pairs_admitted") is not True:
        faults.append("fresh_source_report_not_green")
    if fresh_evaluator.get("trigger_state") != "GREEN" or fresh_evaluator.get("candidate_packet_materialized") is not False:
        faults.append("fresh_evaluator_not_green_before_packet")
    if materialization.get("trigger_state") != "GREEN" or len(p2a.dicts(materialization.get("rows"))) != 18:
        faults.append("base_materialization_not_green")
    if base_evaluator.get("trigger_state") != "GREEN" or int(base_evaluator.get("green_task_count") or 0) != 18:
        faults.append("base_evaluator_not_green")
    source_specs = {int(row.get("index") or 0): row for row in p2a.dicts(source_config.get("sources"))}
    fresh_rows = {int(row.get("index") or 0): row for row in p2a.dicts(source_report.get("rows"))}
    base_rows = {int(row.get("index") or 0): row for row in p2a.dicts(materialization.get("rows"))}
    rows: list[dict[str, Any]] = []
    denominator: list[dict[str, Any]] = []
    if not faults:
        for index in range(1, 19):
            source_row = fresh_rows[index] if index <= 4 else base_rows[index]
            spec = source_specs.get(index) if index <= 4 else None
            receipt, denominator_row, row_faults = build_row(index, source_row, spec)
            rows.append(receipt)
            denominator.append(denominator_row)
            faults.extend(f"task_{index:02d}:{fault}" for fault in row_faults)
    repositories = [str(row.get("repository") or "") for row in denominator]
    strata = Counter(str(row.get("stratum") or "") for row in denominator)
    if len(repositories) != 18 or len(set(repositories)) != 18:
        faults.append("repository_source_disjointness_invalid")
    if len(strata) != 6 or any(count != 3 for count in strata.values()):
        faults.append("stratum_balance_invalid")
    sealed = not faults and len(rows) == 18 and all(row.get("trigger_state") == "GREEN" for row in rows)
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if sealed else "RED",
        "state": "SEALED_FRESH_STATEMENT_GRANULAR_DENOMINATOR_BEFORE_CANDIDATE_GENERATION" if sealed else "INVALID_NOT_SEALED",
        "fresh_source_report": artifact(SOURCE_REPORT),
        "fresh_evaluator_qualification": artifact(FRESH_EVALUATOR),
        "base_materialization": artifact(MATERIALIZATION),
        "base_evaluator_qualification": artifact(BASE_EVALUATOR),
        "statement_granular_runtime": artifact(RUNTIME_PATH),
        "task_count": len(rows),
        "sealed_packet_count": sum(row.get("trigger_state") == "GREEN" for row in rows),
        "repository_count": len(set(repositories)),
        "stratum_counts": dict(sorted(strata.items())),
        "rows": rows,
        "source_denominator": denominator,
        "faults": sorted(set(faults)),
        "information_flow": {
            "candidate_receives_exact_serialized_prompt_only": True,
            "parent_source_only": True,
            "target_source_archive_revision_pr_evaluator_or_oracle_candidate_visible": False,
            "recursive_forbidden_key_audit_required": True,
            "candidate_emitted_integrity_flags_trusted": False,
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
        "maximum_inference": "A GREEN seal establishes only an 18-repository, six-stratum, parent-source-only statement-granular denominator with independently qualified hidden evaluators and conservative physical-context headroom. It authorizes no model call until a separate campaign binding is green and provides no implementation-adequacy, subsystem-effect, D1, D2, training, serving, or book-support evidence."
    }


def build_row(index: int, source_row: dict[str, Any], fresh_spec: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    faults: list[str] = []
    archives = p2a.mapping(source_row.get("archives"))
    parent = p2a.mapping(archives.get("parent"))
    target = p2a.mapping(archives.get("target"))
    selected = p2a.strings(source_row.get("selected_source_paths"))
    if fresh_spec is None:
        original_path = ROOT / "configs" / f"theseus_semantic_ir_production_adequacy_task_{index:02d}.json"
        task = copy.deepcopy(p2a.read_json(original_path))
        task["source_archive"] = str(parent.get("path") or "")
        task["source_archive_sha256"] = str(parent.get("sha256") or "")
        task["source_archive_root"] = str(parent.get("root") or "")
        task["state"] = "SEALED_BEFORE_CANDIDATE_GENERATION"
    else:
        task = fresh_task(index, source_row, fresh_spec, parent, selected)
    task_path = ROOT / "configs" / f"theseus_semantic_ir_production_adequacy_fresh_v4_task_{index:02d}.json"
    packet_path = ROOT / "configs" / f"theseus_semantic_ir_production_adequacy_fresh_v4_candidate_packet_{index:02d}.json"
    contract = p2a.mapping(task.get("semantic_ir_contract"))
    contract.update({
        "version": runtime_v4.HEADER,
        "maximum_symbol_nodes": 1_000_000,
        "maximum_semantic_scope_nodes": 1_000_000,
        "statement_scope_policy": runtime_v4.STATEMENT_SCOPE_POLICY,
        "inventory_truncation_allowed": False,
    })
    task["semantic_ir_contract"] = contract
    prompt = ""
    node_count = 0
    metrics: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix=f"theseus-fresh-v4-packet-{index:02d}-") as directory:
        root = Path(directory) / "source"
        p2a.extract_source_archive(p2a.resolve(task["source_archive"]), root, task["source_archive_root"])
        reads: list[dict[str, Any]] = []
        missing = set(p2a.strings(p2a.mapping(task.get("candidate_visible_context")).get("missing_allowed_effect_paths")))
        for path in selected:
            if path in missing:
                continue
            source_path = p2a.checked_source_path(root, path)
            reads.append({"path": path, "start_line": 1, "end_line": len(source_path.read_text(encoding="utf-8").splitlines())})
        visible = p2a.mapping(task.get("candidate_visible_context"))
        visible.update({
            "reads": reads,
            "searches": [],
            "full_selected_parent_sources": True,
            "project_selected_character_or_token_cap": None,
        })
        task["candidate_visible_context"] = visible
        p2a.write_json(task_path, task)
        audit = p4.audit_task(task_path)
        if audit.get("trigger_state") != "GREEN":
            faults.extend(p2a.strings(audit.get("faults")))
        symbols = runtime_v4.semantic_scope_symbol_table(root, task)
        node_count = len(p2a.dicts(symbols.get("nodes")))
        metrics = p2a.mapping(symbols.get("statement_scope_metrics"))
        task["semantic_ir_contract"]["maximum_symbol_nodes"] = node_count
        task["semantic_ir_contract"]["maximum_semantic_scope_nodes"] = node_count
        p2a.write_json(task_path, task)
        prompt = runtime_v4.render_prompt(task, canary.render_common_context(root, task, symbols))
    packet = {
        "policy": PACKET_POLICY,
        "state": "SEALED_BEFORE_CANDIDATE_GENERATION",
        "opaque_task_id": task.get("opaque_task_id"),
        "serialized_prompt": prompt,
    }
    faults.extend(audit_packet(packet, source_row, target))
    prompt_bytes = len(prompt.encode("utf-8"))
    try:
        prompt_tokens = exact_prompt_tokens(prompt)
    except Exception as exc:
        prompt_tokens = 0
        faults.append(f"exact_frozen_tokenizer_unavailable:{type(exc).__name__}")
    context_residual = MODEL_CONTEXT_TOKENS - prompt_tokens
    if prompt_tokens <= 0 or context_residual <= 0:
        faults.append("exact_physical_context_residual_invalid")
    p2a.write_json(packet_path, packet)
    receipt = {
        "index": index,
        "opaque_task_id": task.get("opaque_task_id"),
        "task_manifest": p2a.rel(task_path),
        "task_manifest_sha256": p2a.sha256_file(task_path),
        "candidate_packet": p2a.rel(packet_path),
        "candidate_packet_sha256": p2a.sha256_file(packet_path),
        "serialized_prompt_sha256": p2a.sha256_text(prompt),
        "serialized_prompt_utf8_bytes": prompt_bytes,
        "conservative_minimum_residual_tokens": MODEL_CONTEXT_TOKENS - prompt_bytes,
        "exact_prompt_tokens": prompt_tokens,
        "exact_context_residual_tokens": context_residual,
        "utf8_byte_upper_bound_exceeds_context": prompt_bytes >= MODEL_CONTEXT_TOKENS,
        "semantic_scope_node_count": node_count,
        "statement_scope_metrics": metrics,
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


def fresh_task(index: int, source_row: dict[str, Any], spec: dict[str, Any], parent: dict[str, Any], selected: list[str]) -> dict[str, Any]:
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
            "command": ["python3", "-c", "import ast,sys; [ast.parse(open(p, encoding='utf-8').read(), filename=p) for p in sys.argv[1:]]", *selected],
            "timeout_seconds": 60,
            "answer_specific": False,
            "candidate_prompt_visibility": False,
        },
        "visible_feedback_map": [{"marker": "SyntaxError", "obligation_ids": ["O1", "O2", "O3"]}],
        "semantic_ir_contract": {
            "version": runtime_v4.HEADER,
            "maximum_symbol_nodes": 1_000_000,
            "maximum_semantic_scope_nodes": 1_000_000,
            "create_file_allowed_only_for_declared_missing_effect_paths": True,
            "role_partition_source_target_loss_and_dependency_identity_required": True,
        },
        "effect_authority": "disposable_snapshot_only",
        "maximum_inference": "One implementation-adequacy observation only; no treatment, D1, D2, serving, training, or book-support claim.",
    }


def audit_packet(packet: dict[str, Any], source_row: dict[str, Any], target: dict[str, Any]) -> list[str]:
    faults: list[str] = []
    for path, key in pool02.recursive_keys(packet):
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
        p2a.rel(BASE_EVALUATOR),
        BASE_EVALUATOR_SHA256,
    ]
    for value in forbidden_values:
        if value and value.lower() in serialized.lower():
            faults.append("forbidden_candidate_value:" + hashlib.sha256(value.encode()).hexdigest()[:12])
    if not str(packet.get("serialized_prompt") or ""):
        faults.append("candidate_prompt_missing")
    return sorted(set(faults))


def exact_prompt_tokens(prompt: str) -> int:
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        snapshot = local_backend.local_snapshot(MODEL_CARD)
        if not local_backend.complete_model_snapshot(snapshot):
            raise AssertionError("complete frozen model snapshot missing")
        _TOKENIZER = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    encoded = _TOKENIZER.apply_chat_template(
        [
            {"role": "system", "content": local_backend.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        **dict(MODEL_CARD["chat_template_kwargs"]),
    )
    values = encoded.get("input_ids") if hasattr(encoded, "get") else encoded
    return len(values) if isinstance(values, list) else 0


def binding_faults() -> list[str]:
    faults: list[str] = []
    for path, expected, label in (
        (SOURCE_REPORT, SOURCE_REPORT_SHA256, "source_report"),
        (FRESH_EVALUATOR, FRESH_EVALUATOR_SHA256, "fresh_evaluator"),
        (MATERIALIZATION, MATERIALIZATION_SHA256, "materialization"),
        (BASE_EVALUATOR, BASE_EVALUATOR_SHA256, "base_evaluator"),
        (RUNTIME_PATH, RUNTIME_SHA256, "statement_runtime"),
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
        "maximum_prompt_utf8_bytes": max((int(row.get("serialized_prompt_utf8_bytes") or 0) for row in rows), default=0),
        "maximum_exact_prompt_tokens": max((int(row.get("exact_prompt_tokens") or 0) for row in rows), default=0),
        "minimum_exact_context_residual_tokens": min((int(row.get("exact_context_residual_tokens") or 0) for row in rows), default=0),
        "faults": report.get("faults"),
        "counters": report.get("counters"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
