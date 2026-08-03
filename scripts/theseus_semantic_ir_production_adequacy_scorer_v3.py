#!/usr/bin/env python3
"""Independently score the sealed two-replacement adequacy denominator."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import theseus_assistant_p2a as p2a
import theseus_semantic_ir_production_adequacy_evaluator_qualification as base_evaluator
import theseus_semantic_ir_production_adequacy_replacement_02_evaluator as evaluator_02
import theseus_semantic_ir_production_adequacy_replacement_04_evaluator as evaluator_04
import theseus_semantic_ir_production_adequacy_runtime as production
import theseus_semantic_ir_production_adequacy_scorer as base_scorer
import theseus_semantic_ir_production_adequacy_scorer_v2 as scorer_v2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_candidates_v3.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_evaluation_v3.json"
POLICY = "project_theseus_semantic_ir_production_adequacy_evaluation_v3"
RUN_POLICY = "project_theseus_semantic_ir_production_adequacy_candidates_v3"
CONFIG = ROOT / "configs" / "theseus_semantic_ir_production_adequacy_campaign_v3.json"
POOL = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_task_pool_v3.json"
MATERIALIZATION = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_materialization_v4.json"
MATERIALIZATION_SHA256 = "7572e6ebb82ae6b16575298c42450a31d7c50ce2823fd5fc6346b12d6216f122"
BASE_QUALIFICATION = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_evaluator_qualification.json"
BASE_QUALIFICATION_SHA256 = "448544a147595413b0d8d0c7523d9442571651a7f11b49d38b9e5a5c9eb9c35a"
SOURCE_02 = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_replacement_02_source.json"
SOURCE_02_SHA256 = "b6a92bd9c8d73eae5f48f1ddf47eac8abce80a9b0dba926bcb18d3df27382680"
QUALIFICATION_02 = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_replacement_02_evaluator.json"
QUALIFICATION_02_SHA256 = "dafacdca124b3849a808e21c0f9bf23ab57bb8180f28b3f5a8dbf77d7a10ee8e"
SOURCE_04 = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_replacement_04_source.json"
SOURCE_04_SHA256 = "579bb48e49e1ce2c777d97a3e87e2a029fcd51b668e6aee49d26423bfd6a91a4"
QUALIFICATION_04 = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_replacement_04_evaluator.json"
QUALIFICATION_04_SHA256 = "aa1a4dce43d1a624dd4bab269732ec7c1b8240fdbf34f5303bbaf0a7315864b5"


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
    run = p2a.read_json(run_path)
    faults: list[str] = []
    if (
        run.get("policy") != RUN_POLICY
        or run.get("state") != "CANDIDATES_SEALED_BEFORE_HIDDEN_EVALUATION"
        or run.get("trigger_state") != "GREEN"
        or int(run.get("completed_task_count") or 0) != 18
        or int(run.get("preserved_candidate_count") or 0) != 3
        or int(run.get("new_candidate_count") or 0) != 15
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
        or config.get("state") != "PROSPECTIVELY_BOUND_AFTER_V3_POOL_SEAL_BEFORE_RESUME"
    ):
        faults.append("prospective_campaign_binding_invalid")
    embedded_audit = p2a.mapping(run.get("config_audit"))
    if (
        embedded_audit.get("trigger_state") != "GREEN"
        or embedded_audit.get("config_sha256") != p2a.sha256_file(CONFIG)
        or embedded_audit.get("candidate_generation_opened") is not False
        or embedded_audit.get("hidden_evaluation_opened") is not False
    ):
        faults.append("prospective_campaign_audit_invalid")
    pool = p2a.read_json(POOL)
    if (
        p2a.sha256_file(POOL) != config.get("sealed_resume_pool_sha256")
        or pool.get("state") != "SEALED_V3_REPLACEMENT_DENOMINATOR_BEFORE_RESUME"
        or pool.get("trigger_state") != "GREEN"
    ):
        faults.append("sealed_resume_pool_binding_invalid")
    pool_rows = {int(row.get("index") or 0): row for row in p2a.dicts(pool.get("rows"))}
    run_rows = p2a.dicts(run.get("rows"))
    if sorted(pool_rows) != list(range(1, 19)) or [int(row.get("index") or 0) for row in run_rows] != list(range(1, 19)):
        faults.append("sealed_task_denominator_invalid")
    for row in run_rows:
        receipt = pool_rows.get(int(row.get("index") or 0), {})
        if row.get("task_manifest") != receipt.get("task_manifest") or row.get("candidate_packet") != receipt.get("candidate_packet"):
            faults.append(f"sealed_task_identity_invalid:{int(row.get('index') or 0):02d}")
    evaluator_faults = evaluator_binding_faults()
    faults.extend(evaluator_faults)
    source_rows, strata = source_denominator()
    if sorted(source_rows) != list(range(1, 19)):
        faults.append("independent_source_denominator_invalid")
    results: list[dict[str, Any]] = []
    integrity_red = False
    if not faults:
        for row in run_rows:
            index = int(row.get("index") or 0)
            if index == 2:
                result = scorer_v2.score_replacement_row(row, source_rows[index])
            elif index == 4:
                result = score_replacement_row(row, source_rows[index], index, evaluator_04)
            else:
                result = base_scorer.score_row(row, source_rows[index])
            results.append(result)
            integrity_red = integrity_red or result.get("integrity_state") != "GREEN"
    success = sum(result.get("passed") is True for result in results)
    by_stratum: dict[str, dict[str, int]] = {}
    for result in results:
        receipt = by_stratum.setdefault(
            strata.get(int(result.get("index") or 0), ""), {"successes": 0, "tasks": 0}
        )
        receipt["tasks"] += 1
        receipt["successes"] += int(result.get("passed") is True)
    stratum_floor = bool(by_stratum) and all(
        receipt["tasks"] == 3 and receipt["successes"] >= 2 for receipt in by_stratum.values()
    )
    overall_floor = success >= 13
    if evaluator_faults or "independent_source_denominator_invalid" in faults:
        scientific_status, trigger = "INVALID_EVALUATOR", "RED"
    elif faults or integrity_red:
        scientific_status, trigger = "INVALID_INFORMATION_FLOW", "RED"
    elif overall_floor and stratum_floor:
        scientific_status, trigger = "ADEQUATE_FOR_ONE_FRESH_CLAIM_CAMPAIGN", "GREEN"
    else:
        scientific_status, trigger = "INCONCLUSIVE_IMPLEMENTATION", "YELLOW"
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": trigger,
        "scientific_status": scientific_status,
        "candidate_run": {"path": p2a.rel(run_path), "sha256": p2a.sha256_file(run_path)},
        "base_evaluator_owner": artifact(Path(base_evaluator.__file__).resolve()),
        "replacement_02_evaluator_owner": artifact(Path(evaluator_02.__file__).resolve()),
        "replacement_04_evaluator_owner": artifact(Path(evaluator_04.__file__).resolve()),
        "base_evaluator_qualification": artifact(BASE_QUALIFICATION),
        "replacement_02_evaluator_qualification": artifact(QUALIFICATION_02),
        "replacement_04_evaluator_qualification": artifact(QUALIFICATION_04),
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
            "This disposition applies only to the exact frozen TMax model, versioned production "
            "runtime, two-replacement source panel, two-call repair regime, host-safety "
            "interlock, and construct-specific independent evaluators. A failure cannot falsify "
            "Semantic IR broadly. A pass opens one fresh matched claim-development campaign only."
        ),
        "runtime_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def evaluator_binding_faults() -> list[str]:
    faults: list[str] = []
    for path, expected, label in (
        (MATERIALIZATION, MATERIALIZATION_SHA256, "materialization"),
        (BASE_QUALIFICATION, BASE_QUALIFICATION_SHA256, "base_qualification"),
        (SOURCE_02, SOURCE_02_SHA256, "source_02"),
        (QUALIFICATION_02, QUALIFICATION_02_SHA256, "qualification_02"),
        (SOURCE_04, SOURCE_04_SHA256, "source_04"),
        (QUALIFICATION_04, QUALIFICATION_04_SHA256, "qualification_04"),
    ):
        if not path.is_file() or p2a.sha256_file(path) != expected:
            faults.append(f"evaluator_binding_invalid:{label}")
    for report, label in ((BASE_QUALIFICATION, "base"), (QUALIFICATION_02, "02"), (QUALIFICATION_04, "04")):
        if p2a.read_json(report).get("trigger_state") != "GREEN":
            faults.append(f"evaluator_qualification_invalid:{label}")
    return faults


def source_denominator() -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    materialization = p2a.read_json(MATERIALIZATION)
    rows = {int(row.get("index") or 0): row for row in p2a.dicts(materialization.get("rows"))}
    for index, report_path, selected in (
        (2, SOURCE_02, ["business_logic_test.py"]),
        (4, SOURCE_04, ["skbio/alignment/_pair.py"]),
    ):
        report = p2a.read_json(report_path)
        metadata = p2a.mapping(report.get("metadata"))
        rows[index] = {
            "index": index,
            "repository": metadata.get("repository"),
            "parent_revision": metadata.get("parent_revision"),
            "target_revision": metadata.get("target_revision"),
            "stratum": metadata.get("stratum"),
            "selected_source_paths": selected,
            "archives": report.get("archives"),
        }
    return rows, {index: str(row.get("stratum") or "") for index, row in rows.items()}


def score_replacement_row(
    row: dict[str, Any], source_row: dict[str, Any], index: int, evaluator: Any
) -> dict[str, Any]:
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
        if (
            not report_path.is_file()
            or p2a.sha256_file(report_path) != receipt.get("report_sha256")
            or int(receipt.get("call_number") or 0) != position + 1
            or receipt.get("candidate_output_sha256") != hashes[position]
        ):
            faults.append(f"candidate_runtime_receipt_invalid:{position + 1}")
    if any(
        receipt.get("termination_reason") not in {"parser_complete", "model_eos"}
        or receipt.get("safety_ceiling_hit") is True
        for receipt in telemetry
    ):
        faults.append("candidate_completion_custody_invalid")
    selected = p2a.strings(source_row.get("selected_source_paths"))
    actions = p2a.dicts(payload.get("actions"))
    sources: dict[str, str] = {}
    changed: list[str] = []
    apply_faults: list[str] = []
    with tempfile.TemporaryDirectory(prefix=f"theseus-adequacy-score-{index:02d}r1-") as directory:
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
                sources[path] = p2a.checked_source_path(root, path).read_text(encoding="utf-8")
            except (OSError, p2a.InstrumentFault):
                pass
    passed = False
    evaluator_fault = None
    if not faults:
        try:
            passed = evaluator.evaluate(sources, tuple(selected))
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


def artifact(path: Path) -> dict[str, str]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)}


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": report.get("trigger_state"),
        "scientific_status": report.get("scientific_status"),
        "success_count": report.get("success_count"),
        "task_count": report.get("task_count"),
        "faults": report.get("faults"),
        "counters": report.get("counters"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
