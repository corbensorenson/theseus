#!/usr/bin/env python3
"""Blindly score fresh v5 only after all 18 candidate seals."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import theseus_assistant_p2a as p2a
import theseus_semantic_ir_production_adequacy_evaluator_qualification as base_evaluator
import theseus_semantic_ir_production_adequacy_fresh_v4_evaluator as fresh_v4_evaluator
import theseus_semantic_ir_production_adequacy_fresh_v5_evaluator as replacement_evaluator
import theseus_semantic_ir_production_adequacy_runtime_v4 as runtime_v4


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "theseus_semantic_ir_production_adequacy_campaign_v5.json"
POOL = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v5_task_pool.json"
POOL_SHA256 = "7fb283a784322355b891e6e8ab48049010a48aac05bf5401ceda21d1153d59f2"
REPLACEMENT_QUALIFICATION = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v5_evaluator.json"
REPLACEMENT_QUALIFICATION_SHA256 = "2da5f5da77d39301e089622af5051e3d667086997cabda0a7d2241cd7526e052"
FRESH_V4_QUALIFICATION = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v4_evaluator.json"
FRESH_V4_QUALIFICATION_SHA256 = "f3ae736ecae7be1c559181b29b862a90fac0ccd7dbfbba8daa216654b31a9662"
BASE_QUALIFICATION = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_evaluator_qualification.json"
BASE_QUALIFICATION_SHA256 = "448544a147595413b0d8d0c7523d9442571651a7f11b49d38b9e5a5c9eb9c35a"
DEFAULT_RUN = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_candidates_v5.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_score_v5.json"
POLICY = "project_theseus_semantic_ir_production_adequacy_score_v5"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=p2a.rel(DEFAULT_RUN))
    parser.add_argument("--out", default=p2a.rel(DEFAULT_OUT))
    args = parser.parse_args()
    report = score(p2a.resolve(args.run))
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") in {"GREEN", "YELLOW"} else 2


def score(run_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    faults = evaluator_binding_faults()
    run = p2a.read_json(run_path)
    if (
        run.get("policy") != "project_theseus_semantic_ir_production_adequacy_candidates_v5"
        or run.get("state") != "CANDIDATES_SEALED_BEFORE_HIDDEN_EVALUATION"
        or run.get("trigger_state") != "GREEN"
        or int(run.get("completed_task_count") or 0) != 18
        or int(run.get("new_candidate_count") or 0) != 18
        or int(run.get("preserved_candidate_count") or 0) != 0
        or run.get("hidden_evaluation_opened") is not False
    ):
        faults.append("candidate_run_not_sealed_for_hidden_evaluation")
    counters = p2a.mapping(run.get("counters"))
    if int(counters.get("local_model_calls") or 0) != 36 or int(counters.get("external_inference_calls") or 0) != 0 or int(counters.get("hidden_evaluator_executions") or 0) != 0:
        faults.append("candidate_run_counter_invalid")
    config = p2a.read_json(CONFIG)
    run_config = p2a.mapping(run.get("config"))
    if run_config.get("path") != p2a.rel(CONFIG) or run_config.get("sha256") != p2a.sha256_file(CONFIG) or config.get("state") != "PROSPECTIVELY_BOUND_FRESH_V5_BEFORE_CANDIDATE_GENERATION":
        faults.append("prospective_campaign_binding_invalid")
    embedded_audit = p2a.mapping(run.get("config_audit"))
    if embedded_audit.get("trigger_state") != "GREEN" or embedded_audit.get("config_sha256") != p2a.sha256_file(CONFIG) or embedded_audit.get("candidate_generation_opened") is not False or embedded_audit.get("hidden_evaluation_opened") is not False:
        faults.append("prospective_campaign_audit_invalid")
    pool = p2a.read_json(POOL)
    if pool.get("trigger_state") != "GREEN" or pool.get("state") != "SEALED_FRESH_V5_STATEMENT_GRANULAR_DENOMINATOR_BEFORE_CANDIDATE_GENERATION":
        faults.append("sealed_pool_invalid")
    pool_rows = {int(row.get("index") or 0): row for row in p2a.dicts(pool.get("rows"))}
    source_rows = {int(row.get("index") or 0): row for row in p2a.dicts(pool.get("source_denominator"))}
    run_rows = p2a.dicts(run.get("rows"))
    if sorted(pool_rows) != list(range(1, 19)) or sorted(source_rows) != list(range(1, 19)) or [int(row.get("index") or 0) for row in run_rows] != list(range(1, 19)):
        faults.append("sealed_denominator_invalid")
    results: list[dict[str, Any]] = []
    integrity_red = False
    if not faults:
        for row in run_rows:
            index = int(row.get("index") or 0)
            result = score_row(row, pool_rows[index], source_rows[index])
            results.append(result)
            integrity_red = integrity_red or result.get("integrity_state") != "GREEN"
    successes = sum(result.get("passed") is True for result in results)
    strata: dict[str, dict[str, int]] = {}
    for result in results:
        name = str(source_rows.get(int(result.get("index") or 0), {}).get("stratum") or "")
        receipt = strata.setdefault(name, {"successes": 0, "tasks": 0})
        receipt["tasks"] += 1
        receipt["successes"] += int(result.get("passed") is True)
    overall_floor = successes >= 13
    stratum_floor = bool(strata) and all(value["tasks"] == 3 and value["successes"] >= 2 for value in strata.values())
    if faults or integrity_red:
        scientific_status, trigger = "INVALID_INFORMATION_FLOW_OR_EVALUATOR", "RED"
    elif overall_floor and stratum_floor:
        scientific_status, trigger = "ADEQUATE_FOR_ONE_FRESH_CLAIM_CAMPAIGN", "GREEN"
    else:
        scientific_status, trigger = "INCONCLUSIVE_IMPLEMENTATION", "YELLOW"
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": trigger,
        "scientific_status": scientific_status,
        "candidate_run": artifact(run_path),
        "sealed_task_pool": artifact(POOL),
        "replacement_evaluator_owner": artifact(Path(replacement_evaluator.__file__).resolve()),
        "fresh_v4_evaluator_owner": artifact(Path(fresh_v4_evaluator.__file__).resolve()),
        "base_evaluator_owner": artifact(Path(base_evaluator.__file__).resolve()),
        "replacement_evaluator_qualification": artifact(REPLACEMENT_QUALIFICATION),
        "fresh_v4_evaluator_qualification": artifact(FRESH_V4_QUALIFICATION),
        "base_evaluator_qualification": artifact(BASE_QUALIFICATION),
        "success_count": successes,
        "task_count": len(results),
        "overall_floor_passed": overall_floor,
        "stratum_floor_passed": stratum_floor,
        "strata": strata,
        "rows": results,
        "faults": sorted(set(faults)),
        "route_labels_or_candidate_integrity_flags_passed_to_scoring": False,
        "candidate_integrity_recomputed_independently": True,
        "counters": {
            "hidden_evaluator_executions": len(results),
            "local_model_calls": 0,
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "training_rows_written": 0,
            "D1_cases_consumed": 0,
            "D2_cases_consumed": 0,
        },
        "maximum_inference": "This disposition applies only to the exact frozen TMax model, statement-granular runtime, fresh 18-repository panel, two-call syntax-visible repair regime, independent causal-slice evaluators, and host-safety policy. Failure cannot falsify Semantic IR broadly; a pass opens one fresh matched claim-development campaign only.",
        "runtime_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def score_row(row: dict[str, Any], pool_row: dict[str, Any], source_row: dict[str, Any]) -> dict[str, Any]:
    index = int(row.get("index") or 0)
    faults: list[str] = []
    task_path = p2a.resolve(str(pool_row.get("task_manifest") or ""))
    packet_path = p2a.resolve(str(pool_row.get("candidate_packet") or ""))
    if row.get("task_manifest") != p2a.rel(task_path) or row.get("candidate_packet") != p2a.rel(packet_path):
        faults.append("sealed_task_identity_invalid")
    task = p2a.read_json(task_path)
    packet = p2a.read_json(packet_path)
    payload = p2a.mapping(row.get("candidate_payload"))
    outputs = p2a.mapping(row.get("model_outputs"))
    seal = p2a.mapping(row.get("candidate_seal"))
    expected_seal = {
        "task_manifest_sha256": p2a.sha256_file(task_path),
        "candidate_packet_sha256": p2a.sha256_file(packet_path),
        "serialized_prompt_sha256": p2a.sha256_text(str(packet.get("serialized_prompt") or "")),
        "first_output_sha256": p2a.sha256_text(str(outputs.get("first") or "")),
        "final_output_sha256": p2a.sha256_text(str(outputs.get("final") or "")),
        "candidate_payload_sha256": p2a.stable_hash(payload),
        "sealed_before_hidden_evaluation": True,
    }
    if seal != expected_seal:
        faults.append("candidate_seal_invalid")
    calls = p2a.dicts(row.get("runtime_calls"))
    telemetry = p2a.dicts(row.get("termination_telemetry"))
    if len(calls) != 2 or len(telemetry) != 2:
        faults.append("candidate_runtime_denominator_invalid")
    hashes = [expected_seal["first_output_sha256"], expected_seal["final_output_sha256"]]
    for position, receipt in enumerate(calls):
        report_path = p2a.resolve(str(receipt.get("report_path") or ""))
        if not report_path.is_file() or p2a.sha256_file(report_path) != receipt.get("report_sha256") or int(receipt.get("call_number") or 0) != position + 1 or receipt.get("candidate_output_sha256") != hashes[position]:
            faults.append(f"candidate_runtime_receipt_invalid:{position + 1}")
    if any(receipt.get("termination_reason") not in {"parser_complete", "model_eos"} or receipt.get("safety_ceiling_hit") is True for receipt in telemetry):
        faults.append("candidate_completion_custody_invalid")
    selected = p2a.strings(source_row.get("selected_source_paths"))
    actions = p2a.dicts(payload.get("actions"))
    sources: dict[str, str] = {}
    changed: list[str] = []
    apply_faults: list[str] = []
    with tempfile.TemporaryDirectory(prefix=f"theseus-adequacy-score-v5-{index:02d}-") as directory:
        root = Path(directory) / "source"
        p2a.extract_source_archive(p2a.resolve(str(task.get("source_archive") or "")), root, str(task.get("source_archive_root") or ""))
        baseline = p2a.inventory(root)
        apply_faults = runtime_v4.apply_actions(root, actions) if actions else []
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
                sources[path] = p2a.checked_source_path(root, path).read_text(encoding="utf-8")
            except (OSError, p2a.InstrumentFault):
                pass
    passed = False
    evaluator_fault = None
    if not faults:
        try:
            if index == 1:
                passed = replacement_evaluator.evaluate(sources, tuple(selected))
            elif index <= 4:
                passed = fresh_v4_evaluator.evaluate(index, sources, tuple(selected))
            else:
                passed = base_evaluator.evaluate(index, sources, tuple(selected))
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


def evaluator_binding_faults() -> list[str]:
    faults: list[str] = []
    for path, expected, label in (
        (POOL, POOL_SHA256, "pool"),
        (REPLACEMENT_QUALIFICATION, REPLACEMENT_QUALIFICATION_SHA256, "replacement_qualification"),
        (FRESH_V4_QUALIFICATION, FRESH_V4_QUALIFICATION_SHA256, "fresh_v4_qualification"),
        (BASE_QUALIFICATION, BASE_QUALIFICATION_SHA256, "base_qualification"),
    ):
        if not path.is_file() or p2a.sha256_file(path) != expected:
            faults.append(f"evaluator_binding_invalid:{label}")
    if (
        p2a.read_json(REPLACEMENT_QUALIFICATION).get("trigger_state") != "GREEN"
        or p2a.read_json(FRESH_V4_QUALIFICATION).get("trigger_state") != "GREEN"
        or p2a.read_json(BASE_QUALIFICATION).get("trigger_state") != "GREEN"
    ):
        faults.append("evaluator_qualification_invalid")
    return faults


def artifact(path: Path) -> dict[str, str]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)}


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": report.get("trigger_state"),
        "scientific_status": report.get("scientific_status"),
        "success_count": report.get("success_count"),
        "task_count": report.get("task_count"),
        "overall_floor_passed": report.get("overall_floor_passed"),
        "stratum_floor_passed": report.get("stratum_floor_passed"),
        "faults": report.get("faults"),
        "counters": report.get("counters"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
