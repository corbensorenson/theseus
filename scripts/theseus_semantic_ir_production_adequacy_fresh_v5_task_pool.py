#!/usr/bin/env python3
"""Seal fresh v5 by replacing consumed Task 1 and rebinding unexposed Tasks 2-18."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import theseus_assistant_p2a as p2a
import theseus_semantic_ir_production_adequacy_fresh_v4_task_pool as v4


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = ROOT / "configs" / "theseus_semantic_ir_production_adequacy_fresh_v5_sources.json"
SOURCE_REPORT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v5_sources.json"
SOURCE_REPORT_SHA256 = "af9b65f34a7f153f5491e5e1ed4550b7a2616d68b5040084e008ed44784085e9"
FRESH_EVALUATOR = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v5_evaluator.json"
FRESH_EVALUATOR_SHA256 = "2da5f5da77d39301e089622af5051e3d667086997cabda0a7d2241cd7526e052"
PRIOR_POOL = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v4_task_pool.json"
PRIOR_POOL_SHA256 = "23533c38b0b57164bdd2f3e46bc493e128e2136d7fe855781672236fc779f0f6"
RUNTIME_PATH = ROOT / "scripts" / "theseus_semantic_ir_production_adequacy_runtime_v4.py"
RUNTIME_SHA256 = "a3d42c6816fe2964a223d7d1209fa12122c06a4ce0c73bcbb48f11d77936624c"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v5_task_pool.json"
POLICY = "project_theseus_semantic_ir_production_adequacy_fresh_v5_task_pool_v1"


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
    prior_pool = p2a.read_json(PRIOR_POOL)
    if source_report.get("trigger_state") != "GREEN" or source_report.get("source_pairs_admitted") is not True:
        faults.append("fresh_source_report_not_green")
    if fresh_evaluator.get("trigger_state") != "GREEN" or fresh_evaluator.get("candidate_packet_materialized") is not False:
        faults.append("fresh_evaluator_not_green_before_packet")
    if prior_pool.get("trigger_state") != "GREEN" or int(prior_pool.get("task_count") or 0) != 18:
        faults.append("prior_pool_not_green")
    source_specs = p2a.dicts(source_config.get("sources"))
    source_rows = p2a.dicts(source_report.get("rows"))
    prior_rows = {int(row.get("index") or 0): row for row in p2a.dicts(prior_pool.get("rows"))}
    prior_denominator = {int(row.get("index") or 0): row for row in p2a.dicts(prior_pool.get("source_denominator"))}
    rows: list[dict[str, Any]] = []
    denominator: list[dict[str, Any]] = []
    if not faults:
        if len(source_specs) != 1 or len(source_rows) != 1:
            faults.append("replacement_source_cardinality_invalid")
        else:
            receipt, denominator_row, row_faults = build_replacement(source_rows[0], source_specs[0])
            rows.append(receipt)
            denominator.append(denominator_row)
            faults.extend(f"task_01:{fault}" for fault in row_faults)
            for index in range(2, 19):
                receipt, denominator_row, row_faults = rebind_unexposed(index, prior_rows[index], prior_denominator[index])
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
        "state": "SEALED_FRESH_V5_STATEMENT_GRANULAR_DENOMINATOR_BEFORE_CANDIDATE_GENERATION" if sealed else "INVALID_NOT_SEALED",
        "fresh_source_report": artifact(SOURCE_REPORT),
        "fresh_evaluator_qualification": artifact(FRESH_EVALUATOR),
        "prior_unexposed_pool": artifact(PRIOR_POOL),
        "statement_granular_runtime": artifact(RUNTIME_PATH),
        "task_count": len(rows),
        "sealed_packet_count": sum(row.get("trigger_state") == "GREEN" for row in rows),
        "repository_count": len(set(repositories)),
        "stratum_counts": dict(sorted(strata.items())),
        "replacement_indices": [1],
        "rebound_unexposed_indices": list(range(2, 19)),
        "consumed_v4_prompt_sha256_reused": False,
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
            "model_declared_context_window_tokens": v4.MODEL_CONTEXT_TOKENS,
            "project_selected_quality_token_cap": None,
            "addressability_measure": "exact frozen tokenizer chat-template token count",
            "normal_completion": ["parser_complete", "model_eos"],
            "physical_context_boundary_hit_invalidates_observation": True,
            "host_watchdog_activation_invalidates_observation": True,
            "v4_host_throughput_evidence_retained": True,
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
        "counters": v4.zero_counters(),
        "maximum_inference": "A GREEN seal establishes only a fresh 18-repository, six-stratum, parent-only statement-granular denominator after replacing consumed v4 Task 1 and rebinding unexposed Tasks 2-18. It authorizes no model call until a separate v5 campaign binding is green and provides no implementation-adequacy, subsystem-effect, D1, D2, training, serving, or book-support evidence."
    }


def build_replacement(source_row: dict[str, Any], spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    original_root = v4.ROOT
    try:
        with tempfile.TemporaryDirectory(prefix="theseus-fresh-v5-task01-") as directory:
            temporary_root = Path(directory)
            (temporary_root / "configs").mkdir(parents=True)
            v4.ROOT = temporary_root
            prior_receipt, denominator, faults = v4.build_row(1, source_row, spec)
            temporary_task = temporary_root / "configs" / "theseus_semantic_ir_production_adequacy_fresh_v4_task_01.json"
            temporary_packet = temporary_root / "configs" / "theseus_semantic_ir_production_adequacy_fresh_v4_candidate_packet_01.json"
            task = p2a.read_json(temporary_task)
            packet = p2a.read_json(temporary_packet)
    finally:
        v4.ROOT = original_root
    task_path = ROOT / "configs" / "theseus_semantic_ir_production_adequacy_fresh_v5_task_01.json"
    packet_path = ROOT / "configs" / "theseus_semantic_ir_production_adequacy_fresh_v5_candidate_packet_01.json"
    p2a.write_json(task_path, task)
    p2a.write_json(packet_path, packet)
    receipt = copy.deepcopy(prior_receipt)
    receipt.update({
        "task_manifest": p2a.rel(task_path),
        "task_manifest_sha256": p2a.sha256_file(task_path),
        "candidate_packet": p2a.rel(packet_path),
        "candidate_packet_sha256": p2a.sha256_file(packet_path),
    })
    return receipt, denominator, faults


def rebind_unexposed(index: int, prior_receipt: dict[str, Any], denominator: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    faults: list[str] = []
    prior_task_path = p2a.resolve(str(prior_receipt.get("task_manifest") or ""))
    prior_packet_path = p2a.resolve(str(prior_receipt.get("candidate_packet") or ""))
    if not prior_task_path.is_file() or not prior_packet_path.is_file():
        return {}, denominator, ["prior_unexposed_packet_missing"]
    if p2a.sha256_file(prior_task_path) != prior_receipt.get("task_manifest_sha256"):
        faults.append("prior_task_manifest_binding_invalid")
    if p2a.sha256_file(prior_packet_path) != prior_receipt.get("candidate_packet_sha256"):
        faults.append("prior_candidate_packet_binding_invalid")
    task = p2a.read_json(prior_task_path)
    packet = p2a.read_json(prior_packet_path)
    task_path = ROOT / "configs" / f"theseus_semantic_ir_production_adequacy_fresh_v5_task_{index:02d}.json"
    packet_path = ROOT / "configs" / f"theseus_semantic_ir_production_adequacy_fresh_v5_candidate_packet_{index:02d}.json"
    p2a.write_json(task_path, task)
    p2a.write_json(packet_path, packet)
    receipt = copy.deepcopy(prior_receipt)
    receipt.update({
        "task_manifest": p2a.rel(task_path),
        "task_manifest_sha256": p2a.sha256_file(task_path),
        "candidate_packet": p2a.rel(packet_path),
        "candidate_packet_sha256": p2a.sha256_file(packet_path),
    })
    receipt["faults"] = sorted(set(p2a.strings(receipt.get("faults")) + faults))
    receipt["trigger_state"] = "GREEN" if not receipt["faults"] else "RED"
    return receipt, copy.deepcopy(denominator), faults


def binding_faults() -> list[str]:
    faults: list[str] = []
    for path, expected, label in (
        (SOURCE_REPORT, SOURCE_REPORT_SHA256, "source_report"),
        (FRESH_EVALUATOR, FRESH_EVALUATOR_SHA256, "fresh_evaluator"),
        (PRIOR_POOL, PRIOR_POOL_SHA256, "prior_pool"),
        (RUNTIME_PATH, RUNTIME_SHA256, "statement_runtime"),
    ):
        if not path.is_file() or p2a.sha256_file(path) != expected:
            faults.append(f"binding_invalid:{label}")
    return faults


def artifact(path: Path) -> dict[str, str]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)}


def summary(report: dict[str, Any]) -> dict[str, Any]:
    rows = p2a.dicts(report.get("rows"))
    return {
        "trigger_state": report.get("trigger_state"),
        "state": report.get("state"),
        "task_count": report.get("task_count"),
        "sealed_packet_count": report.get("sealed_packet_count"),
        "repository_count": report.get("repository_count"),
        "stratum_counts": report.get("stratum_counts"),
        "replacement_task_prompt_tokens": int(rows[0].get("exact_prompt_tokens") or 0) if rows else 0,
        "maximum_exact_prompt_tokens": max((int(row.get("exact_prompt_tokens") or 0) for row in rows), default=0),
        "minimum_exact_context_residual_tokens": min((int(row.get("exact_context_residual_tokens") or 0) for row in rows), default=0),
        "faults": report.get("faults"),
        "counters": report.get("counters"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
