#!/usr/bin/env python3
"""Run the v2r1 mechanics canary under a coarser semantic-unit policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4r_semantic_ir_v2r1_canary as v2r1  # noqa: E402


POLICY = "project_theseus_p4r_semantic_unit_granularity_canary_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/theseus_p4r_semantic_unit_granularity_canary.json"
    )
    parser.add_argument(
        "--out", default="reports/theseus_p4r_semantic_unit_granularity_canary.json"
    )
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    config_path = p2a.resolve(args.config)
    report = (
        v2r1.predecessor.audit_config(config_path)
        if args.audit_only else run_canary(config_path)
    )
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps({
        "trigger_state": report.get("trigger_state"),
        "state": report.get("state"),
        "parse_and_lower": report.get("parse_and_lower"),
        "verified": report.get("verified"),
        "model_calls": report.get("model_calls"),
        "safety_ceiling_hits": report.get("safety_ceiling_hits"),
        "faults": report.get("faults"),
    }, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") == "GREEN" else 2


def run_canary(config_path: Path) -> dict[str, Any]:
    original = v2r1.p2a.runtime_call

    def namespaced_runtime_call(
        arm: str, task_id: str, call_number: int, prompt: str,
        maximum: int, runtime_config: str,
    ) -> dict[str, Any]:
        return original(
            arm,
            task_id.replace("p4r_v2r1_mechanics_", "p4r_semantic_unit_mechanics_"),
            call_number,
            prompt,
            maximum,
            runtime_config,
        )

    v2r1.p2a.runtime_call = namespaced_runtime_call
    try:
        report = v2r1.run_canary(config_path)
    finally:
        v2r1.p2a.runtime_call = original
    report["policy"] = POLICY
    report["predecessor_transport_state"] = report.get("state")
    if report.get("trigger_state") == "GREEN":
        report["state"] = "SEMANTIC_UNIT_GRANULARITY_MECHANICS_GREEN"
        report["next_gate"] = "Intervention sensitivity and dependency-local repair canary."
    report["semantic_unit_policy"] = (
        "Select a complete function/method for nested behavioral edits and a complete "
        "top-level assignment for module-state edits; do not ask the model to replace "
        "fine expression nodes while expecting enclosing semantic-scope source."
    )
    report["scope"] = (
        "Same three hand-authored non-claim mechanics requests under a coarser semantic-unit "
        "selection policy. Passing cannot support cognitive compilation or open fresh P4 alone."
    )
    return report


if __name__ == "__main__":
    raise SystemExit(main())
