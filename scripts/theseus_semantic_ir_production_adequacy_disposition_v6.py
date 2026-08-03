#!/usr/bin/env python3
"""Independently dispose the terminal fresh-v6 infrastructure observation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import theseus_assistant_p2a as p2a
import theseus_semantic_ir_production_adequacy_campaign_v2 as custody


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_candidates_v6.json"
CANDIDATES_SHA256 = "81990bfb7fbc8dd29798e536630f4920209617cb64e2ba0a0d0628128141398a"
JOURNAL = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_campaign_v6_journal.json"
JOURNAL_SHA256 = "6a9982313a1171a8d36f5632d2deff66a14d6e25dc2f6924301cf934179bcd56"
TASK2_CHECKPOINT = ROOT / "reports" / "theseus_assistant_checkpoint_chat_p2a_semantic_ir_adequacy_fresh_v6_02_direct_local_model_1_388c2124672d012e.json"
TASK2_CHECKPOINT_SHA256 = "7b35aab09172097f711e3e5f5a40258077ad28806bfd966b50662daf53851c28"
TASK2_RUNTIME = ROOT / "runtime" / "p2a" / "p2a_semantic_ir_adequacy_fresh_v6_02_direct_local_model_1.json"
TASK2_RUNTIME_SHA256 = "db452e29cf4bc3e80700bbeb3298782dafef05da78fe28baf871e5be6fe3febe"
POOL = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v6_task_pool.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_v6_terminal_disposition.json"
POLICY = "project_theseus_semantic_ir_production_adequacy_v6_terminal_disposition_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=p2a.rel(DEFAULT_OUT))
    args = parser.parse_args()
    report = dispose()
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") == "GREEN" else 2


def dispose() -> dict[str, Any]:
    started = time.perf_counter()
    faults: list[str] = []
    for path, expected, label in (
        (CANDIDATES, CANDIDATES_SHA256, "candidate_report"),
        (JOURNAL, JOURNAL_SHA256, "journal"),
        (TASK2_CHECKPOINT, TASK2_CHECKPOINT_SHA256, "task2_checkpoint"),
        (TASK2_RUNTIME, TASK2_RUNTIME_SHA256, "task2_runtime"),
    ):
        if not path.is_file() or p2a.sha256_file(path) != expected:
            faults.append(f"evidence_binding_invalid:{label}")
    run = p2a.read_json(CANDIDATES)
    journal = p2a.read_json(JOURNAL)
    checkpoint = p2a.read_json(TASK2_CHECKPOINT)
    runtime_report = p2a.read_json(TASK2_RUNTIME)
    pool = p2a.read_json(POOL)
    rows = p2a.dicts(run.get("rows"))
    if (
        run.get("policy") != "project_theseus_semantic_ir_production_adequacy_candidates_v6"
        or run.get("state") != "BLOCKED_INFRASTRUCTURE_REPLACEMENT_REQUIRED"
        or run.get("trigger_state") != "RED"
        or run.get("faults") != ["host_watchdog_infrastructure_invalid:task_02:call_1"]
        or int(run.get("completed_task_count") or 0) != 1
        or int(run.get("new_candidate_count") or 0) != 1
        or run.get("hidden_evaluation_opened") is not False
        or len(rows) != 1
    ):
        faults.append("terminal_candidate_report_invalid")
    counters = p2a.mapping(run.get("counters"))
    if (
        int(counters.get("local_model_calls") or 0) != 2
        or int(counters.get("external_inference_calls") or 0) != 0
        or int(counters.get("hidden_evaluator_executions") or 0) != 0
        or int(counters.get("teacher_calls") or 0) != 0
        or int(counters.get("training_rows_written") or 0) != 0
    ):
        faults.append("terminal_counters_invalid")
    if (
        journal.get("state") != "MODEL_CALL_INFRASTRUCTURE_INVALID"
        or int(journal.get("task_index") or 0) != 2
        or int(journal.get("call_number") or 0) != 1
        or journal.get("completed_candidate_indices") != [1]
        or journal.get("fault") != "host_watchdog_infrastructure_invalid:task_02:call_1"
        or journal.get("hidden_evaluation_opened") is not False
    ):
        faults.append("terminal_journal_invalid")
    if not rows or custody.audit_candidate_custody(rows[0], p2a.dicts(pool.get("rows"))[0]):
        faults.append("preserved_task1_candidate_custody_invalid")
    metrics = p2a.mapping(checkpoint.get("metrics"))
    diagnostic = p2a.mapping(checkpoint.get("invalid_observation_diagnostic"))
    if (
        checkpoint.get("trigger_state") != "RED"
        or "instrument_inadequate_host_safety_wall_time" not in p2a.strings(checkpoint.get("faults"))
        or int(metrics.get("exact_prompt_tokens") or 0) != 45_113
        or metrics.get("generated_tokens") != 0
        or metrics.get("termination_reason") != "host_safety_wall_time"
        or metrics.get("host_safety_wall_time_hit") is not True
        or metrics.get("physical_context_boundary_hit") is not False
        or metrics.get("project_selected_quality_token_cap") is not None
        or diagnostic.get("candidate_admission_allowed") is not False
        or diagnostic.get("hidden_evaluation_allowed") is not False
        or int(diagnostic.get("partial_output_chars") or 0) != 0
    ):
        faults.append("task2_infrastructure_telemetry_invalid")
    runtime_summary = p2a.mapping(runtime_report.get("summary"))
    if (
        runtime_report.get("trigger_state") != "RED"
        or runtime_summary.get("route_integrity_release_allowed") is not False
        or int(runtime_summary.get("runtime_external_inference_calls") or 0) != 0
    ):
        faults.append("task2_route_hold_invalid")
    trigger = "GREEN" if not faults else "RED"
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": trigger,
        "scientific_status": "INCONCLUSIVE_EXPERIMENT" if not faults else "INVALID_EVIDENCE_BINDING",
        "implementation_disposition": "FROZEN_FOR_CURRENT_TMAX_HOST_BLOCK" if not faults else "UNDECIDED",
        "claim_id": "cognitive-compilation-and-semantic-ir.core",
        "claim_effect_decision_authorized": False,
        "book_support_effect": "none",
        "preserved_candidate_indices": [1] if not faults else [],
        "consumed_unsealed_indices": [2] if not faults else [],
        "hidden_evaluation_opened": False,
        "observation": {
            "exact_prompt_tokens": metrics.get("exact_prompt_tokens"),
            "generated_tokens": metrics.get("generated_tokens"),
            "generation_wall_ms": metrics.get("generation_wall_ms"),
            "host_safety_wall_seconds": metrics.get("host_safety_wall_seconds"),
            "termination_reason": metrics.get("termination_reason"),
            "physical_context_boundary_hit": metrics.get("physical_context_boundary_hit"),
            "project_selected_quality_token_cap": metrics.get("project_selected_quality_token_cap"),
        },
        "residual": {
            "class": "FULL_SOURCE_PROMPT_INGESTION_NOT_HOST_OPERABLE",
            "evidence": "The compact address ABI removed repeated labels and digests, but the fresh 45,113-token full-source packet produced zero tokens before the 600-second host wall.",
            "not_established": [
                "model incompetence", "Semantic-IR mechanism failure",
                "cognitive-compilation claim falsification", "context-window exhaustion",
            ],
        },
        "portfolio_transition": {
            "next_claim_id": "virtual-context-abi.core",
            "activation_reason": "The terminal residual is model-visible context materialization and prompt-ingestion burden; the VCM claim directly controls source-bound selection, omission, adequacy, and cost under matched information.",
            "semantic_ir_fresh_reseal_authorized_in_current_block": False,
            "next_stage_model_calls_authorized": 0,
            "next_stage_external_inference_authorized": False,
            "next_stage_user_gate": False,
        },
        "evidence": {
            "candidate_report": artifact(CANDIDATES),
            "journal": artifact(JOURNAL),
            "task2_checkpoint": artifact(TASK2_CHECKPOINT),
            "task2_runtime": artifact(TASK2_RUNTIME),
        },
        "counters": {
            "local_model_calls": 2,
            "external_inference_calls": 0,
            "hidden_evaluator_executions": 0,
            "teacher_calls": 0,
            "training_rows_written": 0,
            "D1_cases_consumed": 0,
            "D2_cases_consumed": 0,
        },
        "faults": sorted(set(faults)),
        "maximum_inference": "The prospectively bound v6 campaign could not complete on the exact frozen TMax MLX runtime and current host: one candidate sealed, then a fresh 45,113-token prompt hit the 600-second host wall before producing a token. This is INCONCLUSIVE_EXPERIMENT and freezes this Semantic-IR implementation only for the current model/host block. It does not falsify Semantic IR, cognitive compilation, TMax capability, or the ASI Stack claim.",
        "runtime_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def artifact(path: Path) -> dict[str, str]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)}


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in (
        "trigger_state", "scientific_status", "implementation_disposition",
        "claim_effect_decision_authorized", "preserved_candidate_indices",
        "consumed_unsealed_indices", "portfolio_transition", "faults", "counters",
    )}


if __name__ == "__main__":
    raise SystemExit(main())
