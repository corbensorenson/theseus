#!/usr/bin/env python3
"""Namespace-isolated adapter for the fresh P4-v2r2-r2 campaign."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

import theseus_assistant_p2a as p2a
import theseus_p4v2r2_cognitive_compilation as predecessor


ROOT = Path(__file__).resolve().parents[1]
POLICY = predecessor.POLICY
INSTRUMENT_POLICY = predecessor.INSTRUMENT_POLICY
RUNTIME_ATTEMPT_NAMESPACE = "p4v2r2r2_attempt1"
PREDECESSOR_RUNTIME_ATTEMPT_NAMESPACE = "p4v2r2r1_attempt1"

# The causal implementation remains the committed v2r2 runner.  Only the
# receipt namespace and its prospective instrument binding change here.
candidate_envelope_complete = predecessor.candidate_envelope_complete
persistent_v2_session = predecessor.persistent_v2_session
render_arm_prompt = predecessor.render_arm_prompt
ir_v2r2 = predecessor.ir_v2r2


def audit_instrument(path: Path) -> dict[str, Any]:
    """Run the inherited audit without weakening its historical namespace set."""
    value = p2a.read_json(path)
    namespace = str(value.get("runtime_attempt_namespace") or "")
    namespace_faults: list[str] = []
    if namespace != RUNTIME_ATTEMPT_NAMESPACE:
        namespace_faults.append("runtime_attempt_namespace_invalid")

    projected = dict(value)
    projected["runtime_attempt_namespace"] = PREDECESSOR_RUNTIME_ATTEMPT_NAMESPACE
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
        report = predecessor.audit_instrument(temporary)
    finally:
        temporary.unlink(missing_ok=True)

    faults = sorted(set(p2a.strings(report.get("faults")) + namespace_faults))
    report.update(
        {
            "policy": "project_theseus_p4v2r2r2_instrument_audit_v1",
            "trigger_state": "GREEN" if not faults else "RED",
            "faults": faults,
            "instrument_sha256": p2a.sha256_file(path),
            "runtime_attempt_namespace": namespace,
            "namespace_adapter": {
                "causal_runner_changed": False,
                "predecessor_runner": p2a.rel(Path(predecessor.__file__).resolve()),
                "predecessor_runner_sha256": p2a.sha256_file(
                    Path(predecessor.__file__).resolve()
                ),
                "receipt_namespace_only": True,
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
    """Delegate execution while substituting only the successor audit owner."""
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
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = run_experiment(p2a.resolve(args.instrument), p2a.resolve(args.task))
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
