#!/usr/bin/env python3
"""Assemble the sealed 18-task denominator after consumed Task 4 replacement."""

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
REPLACEMENT_02_POOL = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_replacement_02_task_pool.json"
REPLACEMENT_02_POOL_SHA256 = "79617f9a88246ceedf85a4bc9d0e465558c2c734133c97b74bec0f59713ce105"
REPLACEMENT_04_POOL = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_replacement_04_task_pool.json"
REPLACEMENT_04_POOL_SHA256 = "38c92addb60d945a138bc404d12da25d4cf2d5ad2e9b1b7c4c0c794a37f28c8c"
INTERRUPTION = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_campaign_v2_watchdog_interruption.json"
INTERRUPTION_SHA256 = "19406846e48b4b302dc1fe4f4a924bfe5c08284154971b7a18724049f1a675ca"
PRESERVED_CANDIDATES = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_candidates_v2.json"
PRESERVED_CANDIDATES_SHA256 = "74759b382c31bc1cbc9a6d3854c7a795bf62c19441497147ce5a7125f633779a"
MATERIALIZATION = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_materialization_v4.json"
MATERIALIZATION_SHA256 = "7572e6ebb82ae6b16575298c42450a31d7c50ce2823fd5fc6346b12d6216f122"
SOURCE_02 = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_replacement_02_source.json"
SOURCE_02_SHA256 = "b6a92bd9c8d73eae5f48f1ddf47eac8abce80a9b0dba926bcb18d3df27382680"
SOURCE_04 = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_replacement_04_source.json"
SOURCE_04_SHA256 = "579bb48e49e1ce2c777d97a3e87e2a029fcd51b668e6aee49d26423bfd6a91a4"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_task_pool_v3.json"
POLICY = "project_theseus_semantic_ir_production_adequacy_resume_pool_v3"
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
    replacement_02 = p2a.read_json(REPLACEMENT_02_POOL)
    replacement_04 = p2a.read_json(REPLACEMENT_04_POOL)
    interruption = p2a.read_json(INTERRUPTION)
    preserved = p2a.read_json(PRESERVED_CANDIDATES)
    materialization = p2a.read_json(MATERIALIZATION)
    source_02 = p2a.read_json(SOURCE_02)
    source_04 = p2a.read_json(SOURCE_04)
    if (
        original.get("trigger_state") != "GREEN"
        or original.get("state") != "SEALED_BEFORE_CANDIDATE_GENERATION"
        or int(original.get("task_count") or 0) != 18
    ):
        faults.append("original_pool_not_green")
    for index, replacement in ((2, replacement_02), (4, replacement_04)):
        if (
            replacement.get("trigger_state") != "GREEN"
            or replacement.get("state") != "SEALED_BEFORE_REPLACEMENT_CANDIDATE_GENERATION"
            or int(replacement.get("replacement_index") or 0) != index
        ):
            faults.append(f"replacement_pool_not_green:{index:02d}")
    if (
        interruption.get("trigger_state") != "RED"
        or interruption.get("scientific_status") != "INVALID_INFRASTRUCTURE_INCOMPLETE_DENOMINATOR"
        or "task_04_surface_exposed_and_ineligible_for_rerun" not in p2a.strings(interruption.get("faults"))
        or "replace exposed task 04 with a fresh licensed source-disjoint same-stratum task"
        not in p2a.strings(interruption.get("required_repairs"))
    ):
        faults.append("interruption_does_not_authorize_task_04_replacement")
    preserved_rows = p2a.dicts(preserved.get("rows"))
    if (
        preserved.get("state") != "BLOCKED_INFRASTRUCTURE_REPLACEMENT_REQUIRED"
        or preserved.get("trigger_state") != "RED"
        or preserved.get("hidden_evaluation_opened") is not False
        or int(preserved.get("completed_task_count") or 0) != 3
        or int(p2a.mapping(preserved.get("counters")).get("local_model_calls") or 0) != 6
        or [int(row.get("index") or 0) for row in preserved_rows] != [1, 2, 3]
    ):
        faults.append("preserved_candidate_custody_invalid")
    original_rows = {int(row.get("index") or 0): row for row in p2a.dicts(original.get("rows"))}
    replacement_rows = {
        2: replacement_row(replacement_02),
        4: replacement_row(replacement_04),
    }
    rows = [replacement_rows.get(index, original_rows.get(index, {})) for index in range(1, 19)]
    if [int(row.get("index") or 0) for row in rows] != list(range(1, 19)):
        faults.append("combined_denominator_indices_invalid")
    if original_rows.get(2) in rows or original_rows.get(4) in rows:
        faults.append("consumed_task_still_present")
    for row in rows:
        index = int(row.get("index") or 0)
        task = p2a.resolve(str(row.get("task_manifest") or ""))
        packet = p2a.resolve(str(row.get("candidate_packet") or ""))
        if not task.is_file() or p2a.sha256_file(task) != row.get("task_manifest_sha256"):
            faults.append(f"task_binding_invalid:{index:02d}")
        if not packet.is_file() or p2a.sha256_file(packet) != row.get("candidate_packet_sha256"):
            faults.append(f"packet_binding_invalid:{index:02d}")
        if task.is_file() and p4.audit_task(task).get("trigger_state") != "GREEN":
            faults.append(f"task_audit_red:{index:02d}")
        prompt = str(p2a.read_json(packet).get("serialized_prompt") or "") if packet.is_file() else ""
        if p2a.sha256_text(prompt) != row.get("serialized_prompt_sha256"):
            faults.append(f"prompt_binding_invalid:{index:02d}")
        if len(prompt.encode("utf-8")) >= MODEL_CONTEXT_TOKENS:
            faults.append(f"prompt_context_residual_invalid:{index:02d}")
    strata = {
        int(row.get("index") or 0): str(row.get("stratum") or "")
        for row in p2a.dicts(materialization.get("rows"))
    }
    strata[2] = str(p2a.mapping(source_02.get("metadata")).get("stratum") or "")
    strata[4] = str(p2a.mapping(source_04.get("metadata")).get("stratum") or "")
    stratum_counts = Counter(strata.get(index, "") for index in range(1, 19))
    if len(stratum_counts) != 6 or any(count != 3 for count in stratum_counts.values()):
        faults.append("stratum_balance_invalid")
    for preserved_row, pool_row in zip(preserved_rows, rows[:3], strict=True):
        if (
            preserved_row.get("task_manifest") != pool_row.get("task_manifest")
            or preserved_row.get("candidate_packet") != pool_row.get("candidate_packet")
        ):
            faults.append("preserved_candidate_identity_invalid")
    green = not faults
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if green else "RED",
        "state": "SEALED_V3_REPLACEMENT_DENOMINATOR_BEFORE_RESUME" if green else "INVALID_NOT_SEALED",
        "original_pool": artifact(ORIGINAL_POOL),
        "replacement_02_pool": artifact(REPLACEMENT_02_POOL),
        "replacement_04_pool": artifact(REPLACEMENT_04_POOL),
        "interruption_receipt": artifact(INTERRUPTION),
        "preserved_candidate_custody": artifact(PRESERVED_CANDIDATES),
        "source_materialization": artifact(MATERIALIZATION),
        "replacement_02_source": artifact(SOURCE_02),
        "replacement_04_source": artifact(SOURCE_04),
        "task_count": len(rows),
        "sealed_packet_count": sum(row.get("trigger_state") == "GREEN" for row in rows),
        "preserved_candidate_count": len(preserved_rows),
        "preserved_denominator_model_calls": 6,
        "resume_generation_indices": list(range(4, 19)),
        "consumed_task_02_rerun_authorized": False,
        "consumed_task_04_rerun_authorized": False,
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "rows": rows,
        "faults": sorted(set(faults)),
        "counters": zero_counters(),
        "maximum_inference": (
            "A GREEN seal establishes only a source-bound 18-task denominator and exact "
            "candidate-1-through-3 custody for prospective v3 resume. It does not authorize "
            "hidden evaluation or establish model competence, Semantic-IR adequacy or effect, "
            "D1, D2, training value, serving, or book support."
        ),
    }


def replacement_row(report: dict[str, Any]) -> dict[str, Any]:
    row = {
        key: report.get(key)
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
    row["index"] = row.pop("replacement_index", 0)
    row["full_parent_source_path_count"] = 1
    row["declared_missing_effect_path_count"] = 0
    return row


def binding_faults() -> list[str]:
    faults: list[str] = []
    for path, expected, label in (
        (ORIGINAL_POOL, ORIGINAL_POOL_SHA256, "original_pool"),
        (REPLACEMENT_02_POOL, REPLACEMENT_02_POOL_SHA256, "replacement_02_pool"),
        (REPLACEMENT_04_POOL, REPLACEMENT_04_POOL_SHA256, "replacement_04_pool"),
        (INTERRUPTION, INTERRUPTION_SHA256, "interruption"),
        (PRESERVED_CANDIDATES, PRESERVED_CANDIDATES_SHA256, "preserved_candidates"),
        (MATERIALIZATION, MATERIALIZATION_SHA256, "materialization"),
        (SOURCE_02, SOURCE_02_SHA256, "source_02"),
        (SOURCE_04, SOURCE_04_SHA256, "source_04"),
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
