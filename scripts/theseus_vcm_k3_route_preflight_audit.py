#!/usr/bin/env python3
"""Role-separated rederivation of the call-free K3 route preflight."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_k3_route_preflight as producer  # noqa: E402

POLICY = "project_theseus_vcm_k3_route_preflight_audit_v3"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_k3_route_preflight.json"
FORBIDDEN = {"repository", "source_task_id", "target", "target_patch", "tests", "hidden_tests", "selected_source_paths", "selected_verifier_paths", "expected", "answer", "solution"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    path = p2a.resolve(args.config)
    cfg = p2a.read_json(path)
    report = audit(path)
    p2a.write_json(p2a.resolve(args.out or str(cfg.get("audit_report") or "")), report)
    print(json.dumps({key: report.get(key) for key in ("trigger_state", "state", "faults", "audited_packet_count", "local_model_calls", "external_reference_calls")}, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") == "GREEN" else 2


def audit(path: Path = DEFAULT_CONFIG, *, actual_report: dict[str, Any] | None = None, actual_packets: dict[str, Any] | None = None, token_counter=None) -> dict[str, Any]:
    cfg = p2a.read_json(path)
    faults: list[str] = []
    producer.validate_binding(cfg, "audit_owner", "audit_owner_sha256", faults)
    counter = token_counter or producer.exact_token_counter(cfg)
    rederived_report, rederived_packets = producer.build(path, token_counter=counter)
    report = actual_report or p2a.read_json(p2a.resolve(str(cfg.get("report") or "")))
    packets = actual_packets or p2a.read_json(p2a.resolve(str(cfg.get("packets_out") or "")))
    if rederived_report.get("trigger_state") != "GREEN": faults.append("producer_rederivation_not_green")
    for key in ("row_count", "route_count", "packet_count", "host_canary_count", "analysis_contract", "host_canary_plan", "information_flow"):
        if report.get(key) != rederived_report.get(key): faults.append(f"report_rederivation_failed:{key}")
    if p2a.dicts(packets.get("rows")) != p2a.dicts(rederived_packets.get("rows")): faults.append("packet_manifest_rederivation_failed")
    rows = p2a.dicts(packets.get("rows"))
    if len(rows) != 36 or {str(row.get("route")) for row in rows} != set(producer.ROUTES): faults.append("packet_route_denominator_invalid")
    if recursive_forbidden_keys(packets): faults.append("candidate_packet_forbidden_metadata")
    by_request: dict[str, list[dict[str, Any]]] = {}
    for row in rows: by_request.setdefault(str(row.get("request_id")), []).append(row)
    for request_id, arms in by_request.items():
        flat = next((row for row in arms if row.get("route") == "information_matched_flat_direct_context"), {})
        vcm = next((row for row in arms if row.get("route") == "governed_vcm"), {})
        if flat.get("context_information_sha256") != vcm.get("context_information_sha256"): faults.append(f"flat_vcm_information_mismatch:{request_id}")
        if sorted(int(row.get("within_row_order") or 0) for row in arms) != [1,2,3,4,5,6]: faults.append(f"counterbalance_invalid:{request_id}")
        for row in arms:
            if row.get("project_selected_quality_token_cap") is not None: faults.append("quality_cap_present")
            exact = row.get("exact_chat_prompt_tokens")
            if exact is not None:
                if int(exact) + int(row.get("physical_context_residual_tokens") or 0) != 262144: faults.append("token_residual_identity_invalid")
            elif int(row.get("prompt_token_lower_bound") or 0) <= 262144 or row.get("physically_addressable") is not False:
                faults.append("oversized_prompt_ineligibility_not_proved")
    if any(report.get(key) != 0 for key in ("local_model_calls", "external_reference_calls", "teacher_calls", "hidden_evaluator_calls")): faults.append("downstream_call_counter_invalid")
    ready = not faults
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if ready else "RED",
        "state": "K3_CALL_FREE_ROUTE_PREFLIGHT_ROLE_SEPARATELY_REDERIVED" if ready else "K3_ROUTE_PREFLIGHT_AUDIT_FAILED",
        "faults": sorted(set(faults)),
        "config": producer.identity(path),
        "producer_report": producer.identity(p2a.resolve(str(cfg.get("report") or ""))) if actual_report is None else {"in_memory": True},
        "packets_artifact": producer.identity(p2a.resolve(str(cfg.get("packets_out") or ""))) if actual_packets is None else {"in_memory": True},
        "audited_row_count": len(by_request),
        "audited_route_count": len(producer.ROUTES),
        "audited_packet_count": len(rows),
        "physically_ineligible_packet_count": sum(int(row.get("physically_addressable") is not True) for row in rows),
        "audit_kind": "role-separated rederivation",
        "local_model_calls": 0,
        "external_reference_calls": 0,
        "teacher_calls": 0,
        "hidden_evaluator_calls": 0,
        "maximum_inference": cfg.get("audit_maximum_inference"),
    }


def recursive_forbidden_keys(value: Any) -> list[str]:
    hits: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key) in FORBIDDEN: hits.add(str(key))
            hits.update(recursive_forbidden_keys(nested))
    elif isinstance(value, list):
        for nested in value: hits.update(recursive_forbidden_keys(nested))
    return sorted(hits)


if __name__ == "__main__":
    raise SystemExit(main())
