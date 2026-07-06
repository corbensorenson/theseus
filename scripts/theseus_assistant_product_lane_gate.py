#!/usr/bin/env python3
"""Gate the B1 assisted verified assistant product lane.

This gate proves a narrow product-lane claim: the local assistant path is wired,
usable through registered surfaces, and emits VCM/tool/verifier/dogfood/VIEA
receipts without public training rows, runtime external inference, fallback
returns, or learned-generation credit. It intentionally does not claim that the
learned code generator is good; that wall belongs to C1.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_E2E = REPORTS / "theseus_assistant_e2e.json"
DEFAULT_PRODUCT = REPORTS / "theseus_assistant_product_spine_smoke.json"
DEFAULT_TRACE = REPORTS / "theseus_assistant_product_spine_trace.jsonl"
DEFAULT_SCHEMA = ROOT / "configs" / "assistant_trace_schema.json"
DEFAULT_OUT = REPORTS / "theseus_assistant_product_lane_gate.json"

REQUIRED_LANES = {"chat_checkpoint", "code_assistant", "tool_assistant", "planning_assistant"}
REQUIRED_TRACE_RECORDS = {
    "intent_contract",
    "command_contract",
    "context_abi_record",
    "context_transaction",
    "context_adequacy",
    "typed_job",
    "planforge_dag",
    "runtime_adapter_invocation",
    "authority_transition",
    "authority_use_receipt",
    "resource_budget",
    "generation_mode",
    "failure_boundary",
    "artifact_graph_record",
    "claim_record",
    "evidence_transition_record",
    "residual_record",
    "policy_optimization_record",
}
ALLOWED_WARNING_GATES = {"code_case_reports_current_generator_wall"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2e", default=rel(DEFAULT_E2E))
    parser.add_argument("--product", default=rel(DEFAULT_PRODUCT))
    parser.add_argument("--trace", default=rel(DEFAULT_TRACE))
    parser.add_argument("--schema", default=rel(DEFAULT_SCHEMA))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()

    e2e = read_json(resolve(args.e2e))
    product = read_json(resolve(args.product))
    schema = read_json(resolve(args.schema))
    trace_rows = read_jsonl(resolve(args.trace))
    report = build_report(e2e=e2e, product=product, schema=schema, trace_rows=trace_rows, args=args)
    write_json(resolve(args.out), report)
    print(json.dumps({"trigger_state": report["trigger_state"], "summary": report["summary"]}, indent=2, sort_keys=True))
    return 1 if args.gate and report["trigger_state"] != "GREEN" else 0


def build_report(
    *,
    e2e: dict[str, Any],
    product: dict[str, Any],
    schema: dict[str, Any],
    trace_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    e2e_summary = dict_value(e2e.get("summary"))
    product_summary = dict_value(product.get("summary"))
    usefulness = dict_value(e2e.get("usefulness_report"))
    trace_counts = Counter(str(row.get("record_type") or "") for row in trace_rows)
    failed_e2e_warnings = [
        str(gate.get("name") or gate.get("gate") or "")
        for gate in list_dicts(e2e.get("gates"))
        if gate.get("severity") == "warning" and not gate.get("passed")
    ]
    failed_e2e_hard = [
        str(gate.get("name") or gate.get("gate") or "")
        for gate in list_dicts(e2e.get("gates"))
        if gate.get("severity") == "hard" and not gate.get("passed")
    ]
    lane_counts = dict_value(usefulness.get("lane_counts"))
    code_wall = dict_value(e2e_summary.get("current_code_generator_wall") or usefulness.get("current_code_generator_wall"))
    gates = [
        gate("assistant_schema_ready", schema.get("policy") == "project_theseus_assistant_trace_schema_v1" and bool(schema.get("required_event_fields")), schema),
        gate("e2e_not_red", e2e.get("trigger_state") in {"GREEN", "YELLOW"}, e2e.get("trigger_state")),
        gate("e2e_hard_gates_pass", not failed_e2e_hard, failed_e2e_hard),
        gate("only_known_c1_warning_allowed", set(failed_e2e_warnings).issubset(ALLOWED_WARNING_GATES), failed_e2e_warnings),
        gate("all_product_routes_pass", int_or(e2e_summary.get("passed_case_count"), 0) == int_or(e2e_summary.get("case_count"), -1) and int_or(e2e_summary.get("case_count"), 0) >= 4, e2e_summary),
        gate("user_facing_cli_passes", bool(e2e_summary.get("user_facing_cli_case_passed")), e2e_summary.get("user_facing_cli_case_passed")),
        gate("session_memory_passes", bool(e2e_summary.get("session_memory_case_passed")) and int_or(e2e_summary.get("session_memory_history_turns_loaded"), 0) > 0, e2e_summary.get("session_memory_history_turns_loaded")),
        gate("posthoc_feedback_metadata_passes", bool(e2e_summary.get("posthoc_feedback_case_passed")) and bool(e2e_summary.get("posthoc_feedback_event_written")), e2e_summary),
        gate("all_required_lanes_have_events", REQUIRED_LANES.issubset(set(lane_counts)), lane_counts),
        gate("dogfood_metadata_pressure_present", int_or(e2e_summary.get("dogfood_training_rows_written"), 0) > 0 and int_or(usefulness.get("trainable_event_count"), 0) > 0, {"summary_rows": e2e_summary.get("dogfood_training_rows_written"), "trainable": usefulness.get("trainable_event_count")}),
        gate("recent_useful_outcomes_present", int_or(usefulness.get("completed_or_accepted_count"), 0) > 0, usefulness.get("completed_or_accepted_count")),
        gate("vcm_ready_all_e2e_cases", int_or(usefulness.get("vcm_ready_case_count"), -1) == int_or(e2e_summary.get("case_count"), -2), usefulness),
        gate("product_spine_green", product.get("trigger_state") == "GREEN", product.get("trigger_state")),
        gate("product_trace_complete", bool(product_summary.get("assistant_viea_trace_complete")) and int_or(product_summary.get("assistant_viea_trace_record_count"), 0) >= len(REQUIRED_TRACE_RECORDS), product_summary),
        gate("trace_required_records_present", REQUIRED_TRACE_RECORDS.issubset({kind for kind, count in trace_counts.items() if count > 0}), dict(trace_counts)),
        gate("tool_evidence_green_and_separate", e2e_summary.get("tool_evidence_state") == "GREEN" and float_or(e2e_summary.get("tool_evidence_tool_on_solve_rate"), 0.0) >= 1.0, e2e_summary),
        gate("private_verifier_receipts_ready", bool(product_summary.get("private_verifier_spine_ready")) and int_or(product_summary.get("private_verifier_spine_record_count"), 0) > 0, product_summary),
        gate("route_validator_and_materialized_view_ready", bool(product_summary.get("route_validator_receipt_ready")) and bool(product_summary.get("viea_materialized_view_ready")), product_summary),
        gate("code_wall_quarantined_for_c1", bool(code_wall) and int_or(code_wall.get("candidate_integrity_mismatch_count"), -1) == 0 and int_or(code_wall.get("public_boundary_violation_count"), -1) == 0, code_wall),
        gate("no_public_external_or_fallback_faults", no_cheat_clean(e2e, product, trace_rows), no_cheat_counters(e2e, product, trace_rows)),
        gate("no_raw_prompt_persisted_in_trace", all(row.get("raw_prompt_stored") is False for row in trace_rows), {"trace_rows": len(trace_rows)}),
        gate("learned_generation_claims_separated", learned_claims_separated(trace_rows), trace_claim_summary(trace_rows)),
    ]
    hard_gaps = [row for row in gates if not row["passed"]]
    expected_invalid_controls = b1_expected_invalid_controls(
        e2e_summary=e2e_summary,
        usefulness=usefulness,
        product_summary=product_summary,
        trace_rows=trace_rows,
        trace_counts=trace_counts,
        code_wall=code_wall,
        gates=gates,
        e2e=e2e,
        product=product,
    )
    synthetic_support_ready = (
        not hard_gaps
        and int_or(usefulness.get("recent_event_count"), 0) >= 20
        and int_or(usefulness.get("trainable_event_count"), 0) >= 20
        and int_or(usefulness.get("completed_or_accepted_count"), 0) >= 8
        and REQUIRED_LANES.issubset(set(lane_counts))
        and int_or(e2e_summary.get("dogfood_training_rows_written"), 0) > 0
        and bool(e2e_summary.get("posthoc_feedback_case_passed"))
        and bool(e2e_summary.get("session_memory_case_passed"))
        and bool(e2e_summary.get("user_facing_cli_case_passed"))
        and product.get("trigger_state") == "GREEN"
        and bool(product_summary.get("assistant_viea_trace_complete"))
        and all(row["rejected"] for row in expected_invalid_controls)
        and no_cheat_clean(e2e, product, trace_rows)
        and learned_claims_separated(trace_rows)
    )
    real_dogfood_support_ready = (
        synthetic_support_ready
        and int_or(usefulness.get("real_user_event_count"), 0) >= 20
        and int_or(usefulness.get("real_user_day_count"), 0) >= 5
        and int_or(usefulness.get("real_user_completed_or_accepted_count"), 0) >= 8
        and bool(usefulness.get("real_user_trace_evidence_ready"))
    )
    support_state = (
        "empirical-test-backed"
        if real_dogfood_support_ready
        else "synthetic-test-backed"
        if synthetic_support_ready
        else ("prototype-backed" if not hard_gaps else "not_yet_supported")
    )
    summary = {
        "b1_assisted_verified_assistant_product_lane_state": "GREEN" if not hard_gaps else "RED",
        "b1_assisted_verified_assistant_product_lane_support_state": support_state,
        "b1_synthetic_support_ready": synthetic_support_ready,
        "b1_empirical_support_ready": real_dogfood_support_ready,
        "b1_empirical_block_reason": "none" if real_dogfood_support_ready else "requires real multi-day user dogfood trace evidence, not only synthetic/e2e fixture metadata",
        "b1_expected_invalid_control_count": len(expected_invalid_controls),
        "b1_expected_invalid_rejected_count": sum(1 for row in expected_invalid_controls if row["rejected"]),
        "e2e_trigger_state": e2e.get("trigger_state"),
        "product_spine_trigger_state": product.get("trigger_state"),
        "passed_case_count": e2e_summary.get("passed_case_count"),
        "case_count": e2e_summary.get("case_count"),
        "recent_event_count": usefulness.get("recent_event_count"),
        "completed_or_accepted_count": usefulness.get("completed_or_accepted_count"),
        "dogfood_training_rows_written": e2e_summary.get("dogfood_training_rows_written"),
        "tool_evidence_state": e2e_summary.get("tool_evidence_state"),
        "tool_evidence_tool_on_solve_rate": e2e_summary.get("tool_evidence_tool_on_solve_rate"),
        "vcm_ready_case_count": usefulness.get("vcm_ready_case_count"),
        "trace_record_count": len(trace_rows),
        "trace_record_type_count": len([kind for kind, count in trace_counts.items() if count > 0]),
        "code_generator_wall_recorded": bool(code_wall),
        "public_training_rows_written": no_cheat_counters(e2e, product, trace_rows)["public_training_rows_written"],
        "external_inference_calls": no_cheat_counters(e2e, product, trace_rows)["external_inference_calls"],
        "fallback_return_count": no_cheat_counters(e2e, product, trace_rows)["fallback_return_count"],
        "hard_gap_count": len(hard_gaps),
    }
    return {
        "policy": "project_theseus_assistant_product_lane_gate_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if not hard_gaps else "RED",
        "summary": summary,
        "gates": gates,
        "hard_gaps": hard_gaps,
        "support_state_basis": {
            "synthetic_support_ready": synthetic_support_ready,
            "empirical_support_ready": real_dogfood_support_ready,
            "empirical_block_reason": "none" if real_dogfood_support_ready else "requires real multi-day user dogfood trace evidence, not only synthetic/e2e fixture metadata",
            "recent_event_count": usefulness.get("recent_event_count"),
            "trainable_event_count": usefulness.get("trainable_event_count"),
            "completed_or_accepted_count": usefulness.get("completed_or_accepted_count"),
            "real_user_event_count": usefulness.get("real_user_event_count"),
            "real_user_day_count": usefulness.get("real_user_day_count"),
            "real_user_completed_or_accepted_count": usefulness.get("real_user_completed_or_accepted_count"),
            "real_user_trace_evidence_ready": usefulness.get("real_user_trace_evidence_ready"),
            "lane_counts": lane_counts,
            "surface_counts": usefulness.get("surface_counts"),
            "outcome_counts": usefulness.get("outcome_counts"),
            "dogfood_training_rows_written": e2e_summary.get("dogfood_training_rows_written"),
            "posthoc_feedback_case_passed": e2e_summary.get("posthoc_feedback_case_passed"),
            "session_memory_case_passed": e2e_summary.get("session_memory_case_passed"),
            "user_facing_cli_case_passed": e2e_summary.get("user_facing_cli_case_passed"),
            "vcm_ready_case_count": usefulness.get("vcm_ready_case_count"),
            "required_lane_count": len(REQUIRED_LANES),
            "trace_record_count": len(trace_rows),
            "trace_required_record_type_count": len(REQUIRED_TRACE_RECORDS),
            "expected_invalid_control_count": len(expected_invalid_controls),
            "expected_invalid_rejected_count": sum(1 for row in expected_invalid_controls if row["rejected"]),
            "no_cheat_counters": no_cheat_counters(e2e, product, trace_rows),
            "raw_text_training_allowed": usefulness.get("raw_text_training_allowed"),
        },
        "expected_invalid_controls": expected_invalid_controls,
        "evidence_refs": [rel(resolve(args.e2e)), rel(resolve(args.product)), rel(resolve(args.trace)), rel(resolve(args.schema))],
        "non_claims": [
            "B1 synthetic-test-backed means the assisted local product lane has metadata-only fixture/e2e dogfood outcomes and replay receipts, not proven real daily usefulness.",
            "B1 empirical-test-backed requires real multi-day user dogfood trace evidence and is not claimed by this fixture-only gate.",
            "Tool-assisted and deterministic outputs are reported separately from learned-generation capability.",
            "The current code generator semantic wall is preserved as C1 negative evidence, not hidden by the product lane.",
            "No public benchmark payloads enter training rows, and runtime external inference remains forbidden.",
        ],
    }


def b1_expected_invalid_controls(
    *,
    e2e_summary: dict[str, Any],
    usefulness: dict[str, Any],
    product_summary: dict[str, Any],
    trace_rows: list[dict[str, Any]],
    trace_counts: Counter,
    code_wall: dict[str, Any],
    gates: list[dict[str, Any]],
    e2e: dict[str, Any],
    product: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "control": "no_dogfood_outcomes_blocks_empirical_claim",
            "rejected": int_or(usefulness.get("recent_event_count"), 0) >= 20
            and int_or(usefulness.get("completed_or_accepted_count"), 0) >= 8,
            "reason": "synthetic B1 needs metadata-only accepted/completed local outcomes; empirical B1 separately requires real multi-day user trace evidence",
        },
        {
            "control": "missing_lane_coverage_blocks_product_claim",
            "rejected": REQUIRED_LANES.issubset(set(dict_value(usefulness.get("lane_counts")))),
            "reason": "assistant product evidence must cover chat, code, tool, and planning lanes",
        },
        {
            "control": "raw_text_training_blocks_b1",
            "rejected": usefulness.get("raw_text_training_allowed") is False
            and all(row.get("raw_prompt_stored") is False for row in trace_rows),
            "reason": "B1 dogfood pressure is metadata-only; raw prompt text cannot be training evidence",
        },
        {
            "control": "tool_or_router_overclaim_blocks_b1",
            "rejected": learned_claims_separated(trace_rows),
            "reason": "tool-assisted/product usefulness must stay separate from learned-generation claims",
        },
        {
            "control": "hidden_code_wall_blocks_b1",
            "rejected": bool(code_wall)
            and int_or(code_wall.get("candidate_integrity_mismatch_count"), -1) == 0
            and int_or(code_wall.get("public_boundary_violation_count"), -1) == 0
            and bool(code_wall.get("semantic_pass_currently_zero")),
            "reason": "B1 cannot hide the C1 learned-generator wall behind product usefulness",
        },
        {
            "control": "missing_vcm_tool_verifier_trace_blocks_b1",
            "rejected": int_or(usefulness.get("vcm_ready_case_count"), -1) >= 4
            and e2e_summary.get("tool_evidence_state") == "GREEN"
            and float_or(e2e_summary.get("tool_evidence_tool_on_solve_rate"), 0.0) >= 1.0
            and bool(product_summary.get("private_verifier_spine_ready"))
            and REQUIRED_TRACE_RECORDS.issubset({kind for kind, count in trace_counts.items() if count > 0}),
            "reason": "empirical assistant evidence must still carry VCM/tool/verifier/VIEA receipts",
        },
        {
            "control": "public_external_fallback_fault_blocks_b1",
            "rejected": no_cheat_clean(e2e, product, trace_rows),
            "reason": "B1 cannot pass with public-training, runtime-external-inference, or fallback-return counters",
        },
        {
            "control": "failed_product_gate_blocks_b1",
            "rejected": all(row["passed"] for row in gates),
            "reason": "empirical support cannot override a failed product-lane gate",
        },
    ]


def no_cheat_clean(e2e: dict[str, Any], product: dict[str, Any], trace_rows: list[dict[str, Any]]) -> bool:
    counters = no_cheat_counters(e2e, product, trace_rows)
    return all(value == 0 for value in counters.values())


def no_cheat_counters(e2e: dict[str, Any], product: dict[str, Any], trace_rows: list[dict[str, Any]]) -> dict[str, int]:
    public_rows = int_or(e2e.get("public_training_rows_written"), 0) + int_or(product.get("public_training_rows_written"), 0)
    external = int_or(e2e.get("external_inference_calls"), 0) + int_or(product.get("external_inference_calls"), 0)
    fallback = int_or(e2e.get("fallback_return_count"), 0) + int_or(product.get("fallback_return_count"), 0)
    for row in trace_rows:
        public_rows += int_or(row.get("public_training_rows_written"), 0)
        external += int_or(row.get("external_inference_calls"), 0)
        fallback += int_or(row.get("fallback_return_count"), 0)
    return {
        "public_training_rows_written": public_rows,
        "external_inference_calls": external,
        "fallback_return_count": fallback,
    }


def learned_claims_separated(trace_rows: list[dict[str, Any]]) -> bool:
    generation_rows = [row for row in trace_rows if row.get("record_type") == "generation_mode"]
    if not generation_rows:
        return False
    for row in generation_rows:
        content = dict_value(row.get("content"))
        if content.get("learned_generation_claim") is True:
            return False
        if content.get("tool_assisted_claim") is True and content.get("deterministic_tool_credit_separate") is not True:
            return False
    return True


def trace_claim_summary(trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    generation_rows = [dict_value(row.get("content")) for row in trace_rows if row.get("record_type") == "generation_mode"]
    return {
        "generation_mode_count": len(generation_rows),
        "learned_generation_claim_count": sum(1 for row in generation_rows if row.get("learned_generation_claim") is True),
        "tool_assisted_count": sum(1 for row in generation_rows if row.get("tool_assisted_claim") is True),
        "tool_credit_separate_count": sum(1 for row in generation_rows if row.get("deterministic_tool_credit_separate") is True),
    }


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    rows.append(data)
    except OSError:
        return rows
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def list_dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def float_or(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
