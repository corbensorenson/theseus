#!/usr/bin/env python3
"""Audit the real frozen-model v2 route canary before P4 recovery task calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402


INSTRUMENT = ROOT / "configs" / "theseus_p4v2r2r1_cognitive_compilation_instrument.json"
CANARY = ROOT / "reports" / "theseus_p4v2r2r1_route_canary.json"
OUT = ROOT / "reports" / "theseus_p4v2r2r1_route_canary_audit.json"
INSTRUMENT_SHA256 = "028b1ce44961df62ccd5076787bb61f5dd032faee5fc2431cf8566f8a8bdfb7e"


def audit() -> dict[str, Any]:
    faults: list[str] = []
    if not INSTRUMENT.is_file() or p2a.sha256_file(INSTRUMENT) != INSTRUMENT_SHA256:
        faults.append("instrument_binding_invalid")
    instrument = p2a.read_json(INSTRUMENT)
    canary = p2a.read_json(CANARY) if CANARY.is_file() else {}
    if canary.get("trigger_state") != "GREEN":
        faults.append("canonical_runtime_canary_red")
    summary = p2a.mapping(canary.get("summary"))
    route = p2a.mapping(canary.get("route_integrity"))
    if summary.get("execution_mode") != "direct_local_model":
        faults.append("canary_execution_mode_invalid")
    if summary.get("route_integrity_ready") is not True:
        faults.append("route_integrity_not_ready")
    if summary.get("route_integrity_release_allowed") is not True:
        faults.append("route_release_not_authorized")
    if p2a.strings(summary.get("route_integrity_failed_checks")):
        faults.append("route_integrity_failed_checks_present")
    if route.get("policy") != "project_theseus_live_route_integrity_v2":
        faults.append("route_receipt_policy_invalid")
    if route.get("release_allowed") is not True or route.get("ready") is not True:
        faults.append("route_receipt_not_green")

    checkpoint = p2a.mapping(canary.get("checkpoint_chat"))
    backend_path = ROOT / str(checkpoint.get("out") or "")
    backend = p2a.read_json(backend_path) if backend_path.is_file() else {}
    if backend.get("policy") != "project_theseus_local_inference_backend_v2":
        faults.append("backend_policy_invalid")
    if backend.get("trigger_state") != "GREEN" or p2a.strings(backend.get("faults")):
        faults.append("backend_canary_red")
    metrics = p2a.mapping(backend.get("metrics"))
    if int(metrics.get("local_model_inference_calls") or 0) != 1:
        faults.append("canary_model_call_count_invalid")
    if metrics.get("project_selected_quality_token_cap") is not None:
        faults.append("quality_token_cap_present")
    if metrics.get("physical_context_boundary_hit") is not False:
        faults.append("physical_context_boundary_hit")
    if metrics.get("termination_reason") != "model_eos":
        faults.append("canary_termination_invalid")
    if not backend_path.name.startswith(
        "theseus_assistant_checkpoint_chat_p4v2r2r1_route_canary_"
    ):
        faults.append("per_call_backend_receipt_identity_invalid")
    expected = p2a.mapping(instrument.get("route_canary_contract"))
    if expected.get("report") != p2a.rel(CANARY):
        faults.append("instrument_canary_path_invalid")
    if int(summary.get("runtime_external_inference_calls") or 0) != 0:
        faults.append("external_runtime_inference_nonzero")
    return {
        "policy": "project_theseus_p4v2r2r1_route_canary_audit_v1",
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "instrument": {"path": p2a.rel(INSTRUMENT), "sha256": p2a.sha256_file(INSTRUMENT)},
        "canonical_runtime_canary": {"path": p2a.rel(CANARY), "sha256": p2a.sha256_file(CANARY)},
        "backend_receipt": {"path": p2a.rel(backend_path), "sha256": p2a.sha256_file(backend_path)},
        "route_receipt_sha256": route.get("receipt_sha256"),
        "route_policy": route.get("policy"),
        "backend_policy": backend.get("policy"),
        "termination_reason": metrics.get("termination_reason"),
        "generated_tokens": metrics.get("generated_tokens"),
        "physical_context_boundary_hits": int(metrics.get("physical_context_boundary_hit") is True),
        "project_selected_quality_token_cap": metrics.get("project_selected_quality_token_cap"),
        "route_canary_model_calls": int(metrics.get("local_model_inference_calls") or 0),
        "task_candidate_or_control_calls": 0,
        "external_inference_calls": int(summary.get("runtime_external_inference_calls") or 0),
        "maximum_inference": "A GREEN canary proves only that one non-task frozen TMax call traversed the canonical direct runtime, emitted immutable v2 backend telemetry, and passed the v2 route release gate. It is not task or mechanism evidence.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    report = audit()
    p2a.write_json(OUT, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
