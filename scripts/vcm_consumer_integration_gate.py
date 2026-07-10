#!/usr/bin/env python3
"""Audit universal adoption of the canonical VCM consumer ABI."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import vcm_consumer_abi


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "vcm_consumer_integration_gate.json"
DEFAULT_MARKDOWN = REPORTS / "vcm_consumer_integration_gate.md"
REQUIRED_RECORD_TYPES = {
    "context_abi_record",
    "context_transaction",
    "context_adequacy",
    "context_lease_receipt",
    "authority_use_receipt",
    "failure_boundary",
}
SIMPLE_CONSUMERS = (
    ("task_context_bridge", REPORTS / "vcm_task_context_bridge.json", ("vcm_consumer_abi",)),
    ("assistant_and_dogfood", REPORTS / "theseus_assistant_vcm_consumer_abi_smoke.json", ("vcm_consumer_abi",)),
    ("deterministic_tools", REPORTS / "deterministic_tool_substrate.json", ("vcm_context_governor_receipt", "consumer_abi")),
    ("training_admission", REPORTS / "training_data_admission_v1.json", ("vcm_context_governor_receipt", "consumer_abi")),
    ("private_verifier", REPORTS / "private_verifier_spine_smoke.json", ("private_verification", "vcm_context_governor_receipt", "consumer_abi")),
    ("train_once_fanout", REPORTS / "code_lm_train_once_fanout.json", ("vcm_context_governor_receipt", "consumer_abi")),
    ("direct_generator", REPORTS / "neural_seed_token_decoder_comparator.json", ("direct_generator_vcm_smoke", "receipt", "consumer_abi")),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=relative(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=relative(DEFAULT_MARKDOWN))
    args = parser.parse_args()
    report = build_report()
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps({"trigger_state": report["trigger_state"], "summary": report["summary"], "hard_gaps": report["hard_gaps"]}, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def build_report() -> dict[str, Any]:
    audits: list[dict[str, Any]] = []
    packets: list[dict[str, Any]] = []
    for consumer_id, path, key_path in SIMPLE_CONSUMERS:
        payload = read_json(path)
        packet = nested(payload, key_path)
        packet = packet if isinstance(packet, dict) else {}
        packets.append(packet)
        audits.append(audit_packet(consumer_id, path, packet))

    planner_path = REPORTS / "theseus_plan_compiled_dags.json"
    planner = read_json(planner_path)
    planner_nodes = [
        node
        for goal in planner.get("compiled_goals", []) if isinstance(goal, dict)
        for node in goal.get("nodes", []) if isinstance(node, dict)
    ]
    for node in planner_nodes:
        packet = nested(node, ("asi_stack_records", "vcm_consumer_abi"))
        packet = packet if isinstance(packet, dict) else {}
        packets.append(packet)
        audits.append(audit_packet(f"planner:{node.get('node_id')}", planner_path, packet))

    native_path = REPORTS / "vcm_native_runtime_probe.json"
    native = read_json(native_path)
    native_summary = as_dict(native.get("summary"))
    native_ready = bool(
        native.get("trigger_state") == "GREEN"
        and native_summary.get("mlx_lm_model_cache_lifecycle_test_passed") is True
        and native_summary.get("native_runtime_claim_backend") == "mlx_apple"
        and native_summary.get("native_runtime_claim_backend_matches_recommended_execution_backend") is True
        and native_summary.get("scheduler_native_kv_route_allowed_for_recommended_backend") is True
        and native_summary.get("mlx_native_kv_lifecycle_claimed") is True
        and native_summary.get("accelerator_kv_parity_claimed") is False
        and native_summary.get("mlx_native_kv_parity_claimed") is False
        and native_summary.get("cuda_native_kv_parity_claimed") is False
        and native_summary.get("metal_native_kv_parity_claimed") is False
    )
    native_audit = {
        "consumer_id": "native_mlx_runtime",
        "report": relative(native_path),
        "ready": native_ready,
        "packet_id": "",
        "faults": [] if native_ready else ["MLX_MODEL_NATIVE_CACHE_PROOF_INCOMPLETE"],
        "record_types": [],
        "source_ref_count": 0,
    }
    audits.append(native_audit)

    assistant = read_json(REPORTS / "theseus_assistant_vcm_consumer_abi_smoke.json")
    assistant_summary = as_dict(assistant.get("summary"))
    dogfood_training = as_dict(nested(assistant, ("dogfood", "training_bridge")))
    dogfood_training_summary = as_dict(dogfood_training.get("summary"))
    dogfood_ready = bool(
        assistant.get("trigger_state") == "GREEN"
        and assistant_summary.get("dogfood_event_written") is True
        and dogfood_training.get("trigger_state") == "GREEN"
        and dogfood_training_summary.get("raw_text_capture_enabled") is False
        and int(dogfood_training_summary.get("public_training_rows") or 0) == 0
        and int(dogfood_training_summary.get("external_inference_calls") or 0) == 0
    )
    negative_controls = expected_invalid_controls(next((packet for packet in packets if packet), {}))
    hard_gaps = [
        {"kind": "consumer_packet_not_ready", "consumer_id": row["consumer_id"], "faults": row["faults"]}
        for row in audits
        if not row["ready"]
    ]
    if not planner_nodes:
        hard_gaps.append({"kind": "planner_nodes_missing", "report": relative(planner_path)})
    if not dogfood_ready:
        hard_gaps.append({"kind": "assistant_dogfood_vcm_path_not_exercised"})
    if any(not row.get("rejected") for row in negative_controls):
        hard_gaps.append({"kind": "expected_invalid_consumer_packet_not_rejected", "controls": negative_controls})
    trigger_state = "GREEN" if not hard_gaps else "RED"
    return {
        "policy": "project_theseus_vcm_consumer_integration_gate_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "canonical_consumer_count": len(audits),
            "ready_consumer_count": sum(1 for row in audits if row["ready"]),
            "planner_node_count": len(planner_nodes),
            "planner_vcm_consumer_abi_ready_count": sum(1 for row in audits if row["consumer_id"].startswith("planner:") and row["ready"]),
            "assistant_dogfood_vcm_path_ready": dogfood_ready,
            "native_mlx_model_cache_ready": native_ready,
            "expected_invalid_control_count": len(negative_controls),
            "expected_invalid_rejected_count": sum(1 for row in negative_controls if row.get("rejected")),
            "hard_gap_count": len(hard_gaps),
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        },
        "consumer_audits": audits,
        "expected_invalid_controls": negative_controls,
        "hard_gaps": hard_gaps,
        "non_claims": [
            "Universal VCM packet adoption is integration evidence, not model capability.",
            "The MLX cache proof is exact-backend lifecycle evidence, not CUDA or cross-accelerator parity.",
            "Dogfood metadata capture does not imply learning or user satisfaction.",
        ],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def audit_packet(consumer_id: str, path: Path, packet: dict[str, Any]) -> dict[str, Any]:
    validation = vcm_consumer_abi.validate_consumer_packet(packet) if packet else {"passed": False, "faults": ["PACKET_MISSING"]}
    records = packet.get("records") if isinstance(packet.get("records"), list) else []
    record_types = {str(row.get("record_type") or "") for row in records if isinstance(row, dict)}
    certificate = as_dict(packet.get("representation_certificate"))
    branch = as_dict(packet.get("snapshot_branch"))
    faults = list(packet.get("typed_faults") or []) + list(validation.get("faults") or [])
    if not REQUIRED_RECORD_TYPES.issubset(record_types):
        faults.append("VCM_CONSUMER_REQUIRED_RECORD_TYPES_MISSING")
    if not certificate.get("source_refs"):
        faults.append("VCM_CONSUMER_SOURCE_REFS_MISSING")
    if branch.get("copy_on_write") is not True or branch.get("source_mutation_allowed") is not False:
        faults.append("VCM_CONSUMER_COPY_ON_WRITE_INVALID")
    if packet.get("ready") is not True:
        faults.append("VCM_CONSUMER_PACKET_NOT_READY")
    return {
        "consumer_id": consumer_id,
        "report": relative(path),
        "ready": not faults,
        "packet_id": packet.get("packet_id"),
        "faults": sorted(set(faults)),
        "record_types": sorted(record_types),
        "source_ref_count": len(certificate.get("source_refs") or []),
    }


def expected_invalid_controls(packet: dict[str, Any]) -> list[dict[str, Any]]:
    if not packet:
        return [{"control": "packet_fixture_missing", "rejected": False}]
    variants: list[tuple[str, dict[str, Any]]] = []
    widened = copy.deepcopy(packet)
    widened["representation_certificate"]["materialized_authority_labels"].append("network_write")
    variants.append(("authority_widening", widened))
    mutation = copy.deepcopy(packet)
    mutation["snapshot_branch"]["write_set"] = list(mutation["snapshot_branch"]["read_set"])
    variants.append(("source_mutation", mutation))
    taint_drop = copy.deepcopy(packet)
    taint_drop["snapshot_branch"]["propagated_taint_labels"] = []
    variants.append(("taint_drop", taint_drop))
    best_effort = copy.deepcopy(packet)
    best_effort["representation_certificate"]["consumer_policy"]["best_effort_materialization_allowed"] = True
    variants.append(("best_effort_materialization", best_effort))
    no_cheat = copy.deepcopy(packet)
    no_cheat["fallback_return_count"] = 1
    variants.append(("fallback_counter", no_cheat))
    return [
        {
            "control": name,
            "rejected": not vcm_consumer_abi.validate_consumer_packet(candidate)["passed"],
        }
        for name, candidate in variants
    ]


def nested(value: Any, path: tuple[str, ...]) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# VCM Consumer Integration Gate",
        "",
        f"- trigger_state: `{report['trigger_state']}`",
        f"- ready consumers: `{summary['ready_consumer_count']}/{summary['canonical_consumer_count']}`",
        f"- planner nodes: `{summary['planner_vcm_consumer_abi_ready_count']}/{summary['planner_node_count']}`",
        f"- assistant dogfood path: `{summary['assistant_dogfood_vcm_path_ready']}`",
        f"- native MLX model cache: `{summary['native_mlx_model_cache_ready']}`",
        f"- invalid controls rejected: `{summary['expected_invalid_rejected_count']}/{summary['expected_invalid_control_count']}`",
        "",
        "## Consumers",
    ]
    for row in report["consumer_audits"]:
        lines.append(f"- `{row['consumer_id']}`: ready=`{row['ready']}` packet=`{row['packet_id'] or 'n/a'}` faults=`{row['faults']}`")
    return "\n".join(lines) + "\n"


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def relative(path: str | Path) -> str:
    value = resolve(path)
    try:
        return str(value.resolve().relative_to(ROOT))
    except ValueError:
        return str(value)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
