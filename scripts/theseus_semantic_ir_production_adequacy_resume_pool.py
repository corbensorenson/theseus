#!/usr/bin/env python3
"""Assemble the sealed 18-task denominator after consumed Task 2 replacement."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import theseus_assistant_p2a as p2a
import theseus_p4_cognitive_compilation as p4


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_POOL = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_task_pool.json"
ORIGINAL_POOL_SHA256 = "1586f73c5dde60d8568cac8dad539fc25c50b647e940982d0fbc2ace3e0bae35"
REPLACEMENT_POOL = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_replacement_02_task_pool.json"
REPLACEMENT_POOL_SHA256 = "79617f9a88246ceedf85a4bc9d0e465558c2c734133c97b74bec0f59713ce105"
INTERRUPTION = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_campaign_host_protection_interruption.json"
INTERRUPTION_SHA256 = "c071ab2f4560dd24ffaae56453a13d7f34f02dca058dd3e63f0a73ccafa0393e"
PRESERVED_CANDIDATE = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_candidates.json"
PRESERVED_CANDIDATE_SHA256 = "2e47e251afca4c592578e0469c2fff3fbf850aa6c22d699f5537c32e2d75e7bc"
MATERIALIZATION = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_materialization_v4.json"
MATERIALIZATION_SHA256 = "7572e6ebb82ae6b16575298c42450a31d7c50ce2823fd5fc6346b12d6216f122"
SOURCE_REPLACEMENT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_replacement_02_source.json"
SOURCE_REPLACEMENT_SHA256 = "b6a92bd9c8d73eae5f48f1ddf47eac8abce80a9b0dba926bcb18d3df27382680"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_task_pool_v2.json"
POLICY = "project_theseus_semantic_ir_production_adequacy_resume_pool_v1"
MODEL_CONTEXT_TOKENS = 262_144


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=p2a.rel(DEFAULT_OUT))
    args = parser.parse_args()
    report = assemble()
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def assemble() -> dict[str, Any]:
    faults = binding_faults()
    original = p2a.read_json(ORIGINAL_POOL)
    replacement = p2a.read_json(REPLACEMENT_POOL)
    interrupted = p2a.read_json(INTERRUPTION)
    preserved = p2a.read_json(PRESERVED_CANDIDATE)
    materialization = p2a.read_json(MATERIALIZATION)
    replacement_source = p2a.read_json(SOURCE_REPLACEMENT)
    if (
        original.get("trigger_state") != "GREEN"
        or original.get("state") != "SEALED_BEFORE_CANDIDATE_GENERATION"
        or int(original.get("task_count") or 0) != 18
    ):
        faults.append("original_pool_not_green")
    if (
        replacement.get("trigger_state") != "GREEN"
        or replacement.get("state") != "SEALED_BEFORE_REPLACEMENT_CANDIDATE_GENERATION"
        or int(replacement.get("replacement_index") or 0) != 2
    ):
        faults.append("replacement_pool_not_green")
    if (
        interrupted.get("trigger_state") != "RED"
        or interrupted.get("scientific_status") != "INVALID_INFRASTRUCTURE_INCOMPLETE_DENOMINATOR"
        or int(p2a.mapping(interrupted.get("interrupted_call")).get("task_index") or 0) != 2
        or "replace exposed task 02 with a fresh license-compatible source-disjoint task before any new model exposure"
        not in p2a.strings(interrupted.get("required_repairs"))
    ):
        faults.append("interruption_does_not_authorize_replacement")
    if (
        preserved.get("state") != "RUNNING_CANDIDATE_GENERATION"
        or int(preserved.get("completed_task_count") or 0) != 1
        or preserved.get("hidden_evaluation_opened") is not False
        or int(p2a.mapping(preserved.get("counters")).get("local_model_calls") or 0) != 2
    ):
        faults.append("preserved_candidate_custody_invalid")
    original_rows = {int(row.get("index") or 0): row for row in p2a.dicts(original.get("rows"))}
    replacement_row = {
        key: replacement.get(key)
        for key in (
            "replacement_index",
            "opaque_task_id",
            "task_manifest",
            "task_manifest_sha256",
            "candidate_packet",
            "candidate_packet_sha256",
            "serialized_prompt_sha256",
            "serialized_prompt_utf8_bytes",
            "conservative_minimum_residual_tokens",
            "semantic_scope_node_count",
            "target_archive_or_source_visible",
            "faults",
            "trigger_state",
        )
    }
    replacement_row["index"] = replacement_row.pop("replacement_index", 0)
    replacement_row["full_parent_source_path_count"] = 1
    replacement_row["declared_missing_effect_path_count"] = 0
    rows = [replacement_row if index == 2 else original_rows.get(index, {}) for index in range(1, 19)]
    if [int(row.get("index") or 0) for row in rows] != list(range(1, 19)):
        faults.append("combined_denominator_indices_invalid")
    if original_rows.get(2) in rows:
        faults.append("consumed_task_02_still_present")
    for row in rows:
        index = int(row.get("index") or 0)
        task = p2a.resolve(str(row.get("task_manifest") or ""))
        packet = p2a.resolve(str(row.get("candidate_packet") or ""))
        if p2a.sha256_file(task) != row.get("task_manifest_sha256"):
            faults.append(f"task_binding_invalid:{index:02d}")
        if p2a.sha256_file(packet) != row.get("candidate_packet_sha256"):
            faults.append(f"packet_binding_invalid:{index:02d}")
        if p4.audit_task(task).get("trigger_state") != "GREEN":
            faults.append(f"task_audit_red:{index:02d}")
        packet_value = p2a.read_json(packet)
        prompt = str(packet_value.get("serialized_prompt") or "")
        if p2a.sha256_text(prompt) != row.get("serialized_prompt_sha256"):
            faults.append(f"prompt_binding_invalid:{index:02d}")
        if len(prompt.encode("utf-8")) >= MODEL_CONTEXT_TOKENS:
            faults.append(f"prompt_context_residual_invalid:{index:02d}")
    strata = {
        int(row.get("index") or 0): str(row.get("stratum") or "")
        for row in p2a.dicts(materialization.get("rows"))
    }
    strata[2] = str(p2a.mapping(replacement_source.get("metadata")).get("stratum") or "")
    stratum_counts = Counter(strata.get(index, "") for index in range(1, 19))
    if len(stratum_counts) != 6 or any(count != 3 for count in stratum_counts.values()):
        faults.append("stratum_balance_invalid")
    preserved_rows = p2a.dicts(preserved.get("rows"))
    if len(preserved_rows) != 1 or preserved_rows[0].get("task_manifest") != rows[0].get("task_manifest"):
        faults.append("preserved_task_01_identity_invalid")
    green = not faults
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if green else "RED",
        "state": "SEALED_REPLACEMENT_DENOMINATOR_BEFORE_RESUME" if green else "INVALID_NOT_SEALED",
        "original_pool": artifact(ORIGINAL_POOL),
        "replacement_pool": artifact(REPLACEMENT_POOL),
        "interruption_receipt": artifact(INTERRUPTION),
        "preserved_candidate_custody": artifact(PRESERVED_CANDIDATE),
        "source_materialization": artifact(MATERIALIZATION),
        "replacement_source": artifact(SOURCE_REPLACEMENT),
        "task_count": len(rows),
        "sealed_packet_count": sum(row.get("trigger_state") == "GREEN" for row in rows),
        "preserved_candidate_count": len(preserved_rows),
        "resume_generation_indices": list(range(2, 19)),
        "consumed_task_02_rerun_authorized": False,
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "rows": rows,
        "faults": sorted(set(faults)),
        "counters": zero_counters(),
        "maximum_inference": (
            "A GREEN seal establishes only a source-bound 18-task replacement denominator and "
            "Task-1 custody for prospective resume. It does not authorize hidden evaluation or "
            "establish model competence, Semantic-IR adequacy or effect, D1, D2, training value, "
            "serving, or book support."
        ),
    }


def binding_faults() -> list[str]:
    faults: list[str] = []
    for path, expected, label in (
        (ORIGINAL_POOL, ORIGINAL_POOL_SHA256, "original_pool"),
        (REPLACEMENT_POOL, REPLACEMENT_POOL_SHA256, "replacement_pool"),
        (INTERRUPTION, INTERRUPTION_SHA256, "interruption"),
        (PRESERVED_CANDIDATE, PRESERVED_CANDIDATE_SHA256, "preserved_candidate"),
        (MATERIALIZATION, MATERIALIZATION_SHA256, "materialization"),
        (SOURCE_REPLACEMENT, SOURCE_REPLACEMENT_SHA256, "replacement_source"),
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
    return {
        "trigger_state": report.get("trigger_state"),
        "state": report.get("state"),
        "task_count": report.get("task_count"),
        "sealed_packet_count": report.get("sealed_packet_count"),
        "preserved_candidate_count": report.get("preserved_candidate_count"),
        "resume_generation_indices": report.get("resume_generation_indices"),
        "faults": report.get("faults"),
        "counters": report.get("counters"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
