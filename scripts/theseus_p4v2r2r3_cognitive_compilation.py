#!/usr/bin/env python3
"""Pre-generation repair adapter for the fresh P4-v2r2-r2 campaign."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

import theseus_assistant_p2a as p2a
import theseus_p4v2r2_cognitive_compilation as causal
import theseus_p4v2r2r2_cognitive_compilation as predecessor


ROOT = Path(__file__).resolve().parents[1]
POLICY = predecessor.POLICY
INSTRUMENT_POLICY = predecessor.INSTRUMENT_POLICY
RUNTIME_ATTEMPT_NAMESPACE = "p4v2r2r3_attempt1"
CAUSAL_AUDIT_NAMESPACE = "p4v2r2r1_attempt1"
INSTRUMENT_STATE = "PROSPECTIVELY_RESEALED_AFTER_PRE_GENERATION_RUNNER_RECURSION"
CAUSAL_AUDIT_STATE = "PROSPECTIVELY_RESEALED_AFTER_ROUTE_IMPLEMENTATION_FAILURE"

# Freeze the non-adapter audit owner before any execution monkeypatching.  The
# failed predecessor adapter resolved a mutable module attribute here and
# recursively called itself before the first model session could be opened.
CAUSAL_AUDIT_INSTRUMENT = causal.audit_instrument

candidate_envelope_complete = predecessor.candidate_envelope_complete
persistent_v2_session = predecessor.persistent_v2_session
render_arm_prompt = predecessor.render_arm_prompt
ir_v2r2 = predecessor.ir_v2r2


def audit_instrument(path: Path) -> dict[str, Any]:
    """Audit the successor through a frozen causal audit function reference."""
    value = p2a.read_json(path)
    namespace = str(value.get("runtime_attempt_namespace") or "")
    namespace_faults: list[str] = []
    if namespace != RUNTIME_ATTEMPT_NAMESPACE:
        namespace_faults.append("runtime_attempt_namespace_invalid")
    if value.get("state") != INSTRUMENT_STATE:
        namespace_faults.append("instrument_state_invalid")
    repair = p2a.mapping(value.get("pre_generation_repair"))
    disposition_owner = p2a.mapping(repair.get("failure_disposition"))
    disposition_path = p2a.resolve(str(disposition_owner.get("path") or ""))
    if (
        not disposition_path.is_file()
        or p2a.sha256_file(disposition_path)
        != str(disposition_owner.get("sha256") or "")
    ):
        namespace_faults.append("failure_disposition_binding_invalid")
    else:
        disposition = p2a.read_json(disposition_path)
        if (
            disposition.get("scientific_status") != "NO_OBSERVATION"
            or int(disposition.get("local_model_calls") or 0) != 0
            or int(disposition.get("runtime_backend_receipts") or 0) != 0
        ):
            namespace_faults.append("failure_disposition_not_reusable")
    if (
        repair.get("candidate_generation_opened") is not False
        or int(repair.get("local_model_calls_consumed") or 0) != 0
        or repair.get("same_task_pool_reuse_authorized") is not True
    ):
        namespace_faults.append("pre_generation_repair_contract_invalid")

    projected = dict(value)
    projected["runtime_attempt_namespace"] = CAUSAL_AUDIT_NAMESPACE
    projected["state"] = CAUSAL_AUDIT_STATE
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        dir=ROOT / "runtime" / "control",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(projected, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        report = CAUSAL_AUDIT_INSTRUMENT(temporary)
    finally:
        temporary.unlink(missing_ok=True)

    faults = sorted(set(p2a.strings(report.get("faults")) + namespace_faults))
    report.update(
        {
            "policy": "project_theseus_p4v2r2r3_instrument_audit_v1",
            "trigger_state": "GREEN" if not faults else "RED",
            "faults": faults,
            "instrument_sha256": p2a.sha256_file(path),
            "runtime_attempt_namespace": namespace,
            "namespace_adapter": {
                "causal_runner_changed": False,
                "causal_audit_owner": p2a.rel(Path(causal.__file__).resolve()),
                "causal_audit_owner_sha256": p2a.sha256_file(
                    Path(causal.__file__).resolve()
                ),
                "frozen_audit_function_reference": True,
                "predecessor_runner": p2a.rel(Path(predecessor.__file__).resolve()),
                "predecessor_runner_sha256": p2a.sha256_file(
                    Path(predecessor.__file__).resolve()
                ),
                "receipt_namespace_and_audit_owner_repair_only": True,
            },
        }
    )
    return report


def run_experiment(
    instrument_path: Path,
    task_path: Path,
    *,
    session_factory: Callable[..., Any] = persistent_v2_session,
) -> dict[str, Any]:
    """Delegate unchanged causal execution with the repaired audit owner."""
    original = predecessor.audit_instrument
    predecessor.audit_instrument = audit_instrument
    try:
        return predecessor.run_experiment(
            instrument_path,
            task_path,
            session_factory=session_factory,
        )
    finally:
        predecessor.audit_instrument = original


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instrument",
        default="configs/theseus_p4v2r2r2_cognitive_compilation_instrument.json",
    )
    parser.add_argument("--task", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    report = (
        audit_instrument(p2a.resolve(args.instrument))
        if args.audit_only
        else run_experiment(p2a.resolve(args.instrument), p2a.resolve(args.task))
    )
    p2a.write_json(p2a.resolve(args.out), report)
    print(
        json.dumps(
            {
                "trigger_state": report.get("trigger_state"),
                "faults": report.get("faults"),
                "denominators": report.get("denominators"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report.get("trigger_state") in {"GREEN", "YELLOW"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
