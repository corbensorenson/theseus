#!/usr/bin/env python3
"""Role-separated audit for the VCM K3 non-scoring host canaries."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_k3_host_canaries as owner  # noqa: E402

POLICY = "project_theseus_vcm_k3_non_scoring_host_canaries_audit_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_k3_host_canaries.json"


def audit(path: Path = DEFAULT_CONFIG, actual: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = p2a.read_json(path)
    faults: list[str] = []
    expected, selected = owner.build_plan(path)
    report = actual or p2a.read_json(p2a.resolve(str(cfg.get("report") or "")))
    if expected.get("trigger_state") != "GREEN":
        faults.append("role_rederived_plan_not_green")
    expected_rows = [{k: v for k, v in row.items() if k != "prompt"} for row in selected]
    if report.get("selected_receipts") != expected_rows:
        faults.append("selected_receipts_mismatch")
    calls = p2a.dicts(report.get("calls"))
    if int(report.get("local_model_calls") or 0) != len(calls):
        faults.append("local_call_count_mismatch")
    if int(report.get("external_reference_calls") or 0) != 0 or int(report.get("hidden_evaluator_calls") or 0) != 0:
        faults.append("forbidden_call_observed")
    for ordinal, row in enumerate(calls, start=1):
        if row.get("call_ordinal") != ordinal or row.get("route") != expected_rows[ordinal - 1]["route"]:
            faults.append(f"call_order_mismatch:{ordinal}")
        if row.get("raw_response_stored") is not False or row.get("capability_or_mechanism_evidence") is not False:
            faults.append(f"custody_or_scope_invalid:{ordinal}")
        if row.get("trigger_state") == "GREEN" and row.get("termination_reason") not in owner.NORMAL_TERMINATIONS:
            faults.append(f"green_termination_invalid:{ordinal}")
    all_green = len(calls) == 6 and all(row.get("trigger_state") == "GREEN" for row in calls)
    expected_state = "K3_NONSCORING_PHYSICAL_HOST_CANARIES_GREEN" if all_green else "INCONCLUSIVE_EXPERIMENT_HOST_OPERABILITY"
    if report.get("state") != expected_state:
        faults.append("terminal_state_mismatch")
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": "K3_NONSCORING_HOST_CANARIES_ROLE_SEPARATELY_REDERIVED" if not faults else "K3_NONSCORING_HOST_CANARY_AUDIT_FAILED",
        "faults": sorted(set(faults)),
        "producer_report": owner.identity(p2a.resolve(str(cfg.get("report") or ""))) if actual is None else {},
        "audited_call_count": len(calls),
        "audited_green_call_count": sum(row.get("trigger_state") == "GREEN" for row in calls),
        "producer_terminal_state": report.get("state"),
        "nine_task_screen_authorized": all_green,
        "hidden_evaluator_calls": 0,
        "external_reference_calls": 0,
        "capability_or_mechanism_evidence": False,
        "maximum_inference": "A GREEN audit proves only the custody and physical-host disposition of the exact non-scoring calls. It does not establish model competence, VCM utility, economics, transfer, or ASI Stack support.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    path = p2a.resolve(args.config)
    cfg = p2a.read_json(path)
    report = audit(path)
    p2a.write_json(p2a.resolve(args.out or str(cfg.get("audit_report") or "")), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
