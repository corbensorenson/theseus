#!/usr/bin/env python3
"""Independently score sealed Semantic-IR adequacy candidates."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import theseus_assistant_p2a as p2a
import theseus_semantic_ir_production_adequacy_evaluator_qualification as evaluator
import theseus_semantic_ir_production_adequacy_runtime as production


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_candidates.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_evaluation.json"
POLICY = "project_theseus_semantic_ir_production_adequacy_evaluation_v1"
MATERIALIZATION = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_materialization_v4.json"
CONSTRUCT_REVIEW = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_source_construct_review_v2.json"
QUALIFICATION = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_evaluator_qualification.json"
CONFIG = ROOT / "configs" / "theseus_semantic_ir_production_adequacy_campaign.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=p2a.rel(DEFAULT_RUN))
    parser.add_argument("--out", default=p2a.rel(DEFAULT_OUT))
    args = parser.parse_args()
    report = score(p2a.resolve(args.run))
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps({
        "trigger_state": report.get("trigger_state"),
        "scientific_status": report.get("scientific_status"),
        "success_count": report.get("success_count"),
        "faults": report.get("faults"),
        "counters": report.get("counters"),
    }, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") in {"GREEN", "YELLOW"} else 2


def score(run_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    run = p2a.read_json(run_path)
    faults: list[str] = []
    if (
        run.get("policy")
        != "project_theseus_semantic_ir_production_adequacy_candidates_v1"
        or run.get("state") != "CANDIDATES_SEALED_BEFORE_HIDDEN_EVALUATION"
        or run.get("trigger_state") != "GREEN"
        or int(run.get("completed_task_count") or 0) != 18
        or run.get("hidden_evaluation_opened") is not False
    ):
        faults.append("candidate_run_not_sealed_for_hidden_evaluation")
    counters = p2a.mapping(run.get("counters"))
    if (
        int(counters.get("local_model_calls") or 0) != 36
        or int(counters.get("external_inference_calls") or 0) != 0
        or int(counters.get("hidden_evaluator_executions") or 0) != 0
    ):
        faults.append("candidate_run_counter_invalid")
    config = p2a.read_json(CONFIG)
    run_config = p2a.mapping(run.get("config"))
    if (
        run_config.get("path") != p2a.rel(CONFIG)
        or run_config.get("sha256") != p2a.sha256_file(CONFIG)
        or config.get("state")
        != "PROSPECTIVELY_BOUND_AFTER_POOL_SEAL_BEFORE_MODEL_EXPOSURE"
    ):
        faults.append("prospective_campaign_binding_invalid")
    embedded_audit = p2a.mapping(run.get("config_audit"))
    if (
        embedded_audit.get("trigger_state") != "GREEN"
        or embedded_audit.get("config_sha256") != p2a.sha256_file(CONFIG)
        or embedded_audit.get("candidate_generation_opened") is not False
    ):
        faults.append("prospective_campaign_audit_invalid")
    pool_path = p2a.resolve(str(config.get("sealed_task_pool") or ""))
    pool = p2a.read_json(pool_path) if pool_path.is_file() else {}
    if (
        p2a.sha256_file(pool_path) != str(config.get("sealed_task_pool_sha256") or "")
        or pool.get("state") != "SEALED_BEFORE_CANDIDATE_GENERATION"
        or pool.get("trigger_state") != "GREEN"
    ):
        faults.append("sealed_task_pool_binding_invalid")
    pool_rows = {
        int(row.get("index") or 0): row for row in p2a.dicts(pool.get("rows"))
    }
    run_rows = p2a.dicts(run.get("rows"))
    if (
        sorted(pool_rows) != list(range(1, 19))
        or sorted(int(row.get("index") or 0) for row in run_rows)
        != list(range(1, 19))
    ):
        faults.append("sealed_task_denominator_invalid")
    for row in run_rows:
        receipt = pool_rows.get(int(row.get("index") or 0), {})
        if (
            row.get("task_manifest") != receipt.get("task_manifest")
            or row.get("candidate_packet") != receipt.get("candidate_packet")
        ):
            faults.append(
                f"sealed_task_identity_invalid:{int(row.get('index') or 0):02d}"
            )
    qualification = p2a.read_json(QUALIFICATION)
    if (
        qualification.get("trigger_state") != "GREEN"
        or int(qualification.get("green_task_count") or 0) != 18
    ):
        faults.append("independent_evaluator_qualification_invalid")
    materialization = p2a.read_json(MATERIALIZATION)
    source_rows = {
        int(row.get("index") or 0): row
        for row in p2a.dicts(materialization.get("rows"))
    }
    if sorted(source_rows) != list(range(1, 19)):
        faults.append("independent_source_denominator_invalid")
    construct = p2a.read_json(CONSTRUCT_REVIEW)
    strata = {
        int(row.get("index") or 0): str(row.get("stratum") or "")
        for row in source_rows.values()
    }
    results: list[dict[str, Any]] = []
    integrity_red = False
    if not faults:
        for row in run_rows:
            result = score_row(row, source_rows.get(int(row.get("index") or 0), {}))
            results.append(result)
            if result.get("integrity_state") != "GREEN":
                integrity_red = True
    if len(results) != 18 and not faults:
        faults.append("scored_denominator_not_18")
    success = sum(result.get("passed") is True for result in results)
    by_stratum: dict[str, dict[str, int]] = {}
    for result in results:
        stratum = strata.get(int(result.get("index") or 0), "")
        receipt = by_stratum.setdefault(stratum, {"successes": 0, "tasks": 0})
        receipt["tasks"] += 1
        receipt["successes"] += int(result.get("passed") is True)
    stratum_floor = bool(by_stratum) and all(
        receipt["tasks"] == 3 and receipt["successes"] >= 2
        for receipt in by_stratum.values()
    )
    overall_floor = success >= 13
    evaluator_invalid = any(
        fault in {
            "independent_evaluator_qualification_invalid",
            "independent_source_denominator_invalid",
        }
        for fault in faults
    )
    if evaluator_invalid:
        scientific_status = "INVALID_EVALUATOR"
        trigger = "RED"
    elif faults or integrity_red:
        scientific_status = "INVALID_INFORMATION_FLOW"
        trigger = "RED"
    elif overall_floor and stratum_floor:
        scientific_status = "ADEQUATE_FOR_ONE_FRESH_CLAIM_CAMPAIGN"
        trigger = "GREEN"
    else:
        scientific_status = "INCONCLUSIVE_IMPLEMENTATION"
        trigger = "YELLOW"
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": trigger,
        "scientific_status": scientific_status,
        "candidate_run": {
            "path": p2a.rel(run_path),
            "sha256": p2a.sha256_file(run_path),
        },
        "evaluator_owner": {
            "path": p2a.rel(Path(evaluator.__file__).resolve()),
            "sha256": p2a.sha256_file(Path(evaluator.__file__).resolve()),
        },
        "evaluator_qualification": {
            "path": p2a.rel(QUALIFICATION),
            "sha256": p2a.sha256_file(QUALIFICATION),
        },
        "construct_review": {
            "path": p2a.rel(CONSTRUCT_REVIEW),
            "sha256": p2a.sha256_file(CONSTRUCT_REVIEW),
            "reviewed_task_count": construct.get("reviewed_task_count"),
        },
        "success_count": success,
        "task_count": len(results),
        "overall_floor_passed": overall_floor,
        "stratum_floor_passed": stratum_floor,
        "strata": by_stratum,
        "rows": results,
        "faults": sorted(set(faults)),
        "route_labels_or_candidate_integrity_flags_passed_to_scoring": False,
        "counters": {
            "hidden_evaluator_executions": len(results),
            "local_model_calls": 0,
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "training_rows_written": 0,
            "D1_cases_consumed": 0,
            "D2_cases_consumed": 0,
        },
        "maximum_inference": (
            "This disposition applies only to the exact frozen TMax model, versioned "
            "production runtime, source panel, candidate protocol, two-call repair regime, "
            "and construct-specific independent evaluator. A failure cannot falsify Semantic "
            "IR broadly; evaluator construct specificity may undercount alternative valid "
            "repairs. A pass opens one fresh matched claim-development campaign only."
        ),
        "runtime_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def score_row(row: dict[str, Any], source_row: dict[str, Any]) -> dict[str, Any]:
    index = int(row.get("index") or 0)
    faults: list[str] = []
    task_path = p2a.resolve(str(row.get("task_manifest") or ""))
    packet_path = p2a.resolve(str(row.get("candidate_packet") or ""))
    task = p2a.read_json(task_path)
    packet = p2a.read_json(packet_path)
    payload = p2a.mapping(row.get("candidate_payload"))
    outputs = p2a.mapping(row.get("model_outputs"))
    seal = p2a.mapping(row.get("candidate_seal"))
    expected_seal = {
        "task_manifest_sha256": p2a.sha256_file(task_path),
        "candidate_packet_sha256": p2a.sha256_file(packet_path),
        "serialized_prompt_sha256": p2a.sha256_text(
            str(packet.get("serialized_prompt") or "")
        ),
        "first_output_sha256": p2a.sha256_text(str(outputs.get("first") or "")),
        "final_output_sha256": p2a.sha256_text(str(outputs.get("final") or "")),
        "candidate_payload_sha256": p2a.stable_hash(payload),
        "sealed_before_hidden_evaluation": True,
    }
    if seal != expected_seal:
        faults.append("candidate_seal_invalid")
    runtime_calls = p2a.dicts(row.get("runtime_calls"))
    telemetry = p2a.dicts(row.get("termination_telemetry"))
    if len(runtime_calls) != 2 or len(telemetry) != 2:
        faults.append("candidate_runtime_denominator_invalid")
    expected_output_hashes = [
        p2a.sha256_text(str(outputs.get("first") or "")),
        p2a.sha256_text(str(outputs.get("final") or "")),
    ]
    for position, receipt in enumerate(runtime_calls):
        report_path = p2a.resolve(str(receipt.get("report_path") or ""))
        if (
            not report_path.is_file()
            or p2a.sha256_file(report_path) != str(receipt.get("report_sha256") or "")
            or int(receipt.get("call_number") or 0) != position + 1
            or receipt.get("candidate_output_sha256") != expected_output_hashes[position]
        ):
            faults.append(f"candidate_runtime_receipt_invalid:{position + 1}")
            continue
        runtime_report = p2a.read_json(report_path)
        if p2a.sha256_text(str(runtime_report.get("assistant_text") or "")) != expected_output_hashes[position]:
            faults.append(f"candidate_runtime_output_replay_invalid:{position + 1}")
    if any(
        row.get("termination_reason") not in {"parser_complete", "model_eos"}
        or row.get("safety_ceiling_hit") is True
        for row in telemetry
    ):
        faults.append("candidate_completion_custody_invalid")
    selected = p2a.strings(source_row.get("selected_source_paths"))
    actions = p2a.dicts(payload.get("actions"))
    sources: dict[str, str] = {}
    changed: list[str] = []
    apply_faults: list[str] = []
    with tempfile.TemporaryDirectory(prefix=f"theseus-adequacy-score-{index:02d}-") as directory:
        root = Path(directory) / "source"
        p2a.extract_source_archive(
            p2a.resolve(str(task.get("source_archive") or "")),
            root,
            str(task.get("source_archive_root") or ""),
        )
        baseline = p2a.inventory(root)
        apply_faults = production.apply_actions(root, actions) if actions else []
        inventory = p2a.inventory(root)
        changed = p2a.changed_paths(baseline, inventory) if not apply_faults else []
        if p2a.stable_hash(inventory) != payload.get("final_inventory_sha256"):
            faults.append("candidate_inventory_replay_mismatch")
        if changed != p2a.strings(payload.get("changed_paths")):
            faults.append("candidate_changed_path_replay_mismatch")
        if apply_faults:
            faults.append("candidate_apply_replay_failed")
        if not set(changed).issubset(set(p2a.strings(task.get("allowed_effect_paths")))):
            faults.append("unauthorized_effect_recomputed")
        for path in selected:
            try:
                sources[path] = p2a.checked_source_path(root, path).read_text(
                    encoding="utf-8"
                )
            except (OSError, p2a.InstrumentFault):
                pass
    passed = False
    evaluator_fault = None
    if not faults:
        try:
            passed = evaluator.evaluate(index, sources, tuple(selected))
        except Exception as exc:
            evaluator_fault = f"{type(exc).__name__}:{exc}"[:1000]
            faults.append("independent_evaluator_execution_failed")
    return {
        "index": index,
        "opaque_task_id": task.get("opaque_task_id"),
        "integrity_state": "GREEN" if not faults else "RED",
        "passed": passed if not faults else False,
        "recomputed_changed_paths": changed,
        "apply_faults": apply_faults,
        "evaluator_fault": evaluator_fault,
        "faults": faults,
    }


if __name__ == "__main__":
    raise SystemExit(main())
