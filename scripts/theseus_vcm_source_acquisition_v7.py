#!/usr/bin/env python3
"""Longer bounded recovery and final-hash custody for the frozen VCM selector."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import theseus_assistant_p2a as p2a
import theseus_vcm_source_acquisition as v1
import theseus_vcm_source_acquisition_v5 as v5
import theseus_vcm_source_acquisition_v6 as v6


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_source_acquisition_v7.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_vcm_source_acquisition_v7.json"
DEFAULT_CHECKPOINT = ROOT / "reports" / "theseus_vcm_source_acquisition_v7_checkpoint.json"
EXPECTED_DELAYS = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 45.0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default=p2a.rel(DEFAULT_OUT))
    parser.add_argument("--checkpoint", default=p2a.rel(DEFAULT_CHECKPOINT))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = p2a.resolve(args.config)
    report = preflight(config_path)
    config = p2a.read_json(config_path)
    if args.execute and report["trigger_state"] == "GREEN":
        not_before = v1.parse_time(
            p2a.mapping(config.get("graphql_transport")).get(
                "execution_not_before_utc"
            )
        )
        if datetime.now(timezone.utc) < not_before:
            report = {
                **report,
                "trigger_state": "PAUSED",
                "state": "GITHUB_RATE_WINDOW_NOT_RECOVERED_ZERO_REQUESTS",
                "faults": ["execution_not_before_utc_not_reached"],
            }
        else:
            retry_policy = p2a.mapping(
                config.get("extended_transport_retry_policy")
            )
            ledger = v5.RequestLedger(
                p2a.resolve(args.checkpoint), config_path, retry_policy
            )
            try:
                report = v6.acquire(config_path, ledger, retry_policy)
            except (
                OSError,
                RuntimeError,
                ValueError,
                json.JSONDecodeError,
                subprocess.CalledProcessError,
            ) as exc:
                report = {
                    **report,
                    "trigger_state": "PAUSED",
                    "state": "GRAPHQL_METADATA_EXTENDED_RETRIES_EXHAUSTED_NO_SOURCE_EXPOSURE",
                    "faults": [f"{type(exc).__name__}:{exc}"[:4000]],
                }
            report["transport_repair"] = {
                "owner": p2a.rel(Path(__file__).resolve()),
                "v6_selection_and_transport_owner_reused": True,
                "extended_retry_horizon_seconds": sum(EXPECTED_DELAYS),
                "checkpoint_finalized_before_report_hash": True,
            }
            report = finalize_receipt(report, ledger)
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps(v1.summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "PAUSED"} else 2


def preflight(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    report = v6.preflight(config_path)
    config = p2a.read_json(config_path)
    policy = p2a.mapping(config.get("extended_transport_retry_policy"))
    receipt = p2a.mapping(config.get("receipt_finalization_policy"))
    faults = p2a.strings(report.get("faults"))
    if (
        int(policy.get("maximum_attempts_per_logical_request") or 0) != 8
        or [float(value) for value in policy.get("retry_delays_seconds", [])]
        != EXPECTED_DELAYS
        or set(int(value) for value in policy.get("transient_http_statuses", []))
        != v5.TRANSIENT_STATUSES
        or set(
            int(value)
            for value in policy.get("permanent_candidate_http_statuses", [])
        )
        != v5.PERMANENT_CANDIDATE_STATUSES
        or policy.get("retry_unknown_network_failures") is not True
        or policy.get("host_transport_activation_is_not_task_or_mechanism_failure")
        is not True
        or receipt.get("checkpoint_finalize_before_artifact_hash") is not True
        or receipt.get("no_checkpoint_mutation_after_artifact_hash") is not True
        or receipt.get("embedded_hash_must_equal_final_file_hash") is not True
    ):
        faults.append("v7_retry_or_final_receipt_policy_invalid")
    report["faults"] = sorted(set(faults))
    report["trigger_state"] = "GREEN" if not faults else "RED"
    report["state"] = (
        "METADATA_SELECTION_V7_EXTENDED_RECOVERY_FINAL_HASH_PREFLIGHT_GREEN"
        if not faults
        else "INVALID_PREFLIGHT"
    )
    return report


def finalize_receipt(
    report: dict[str, Any], ledger: v5.RequestLedger
) -> dict[str, Any]:
    ledger.finalize(
        str(report.get("state") or "UNKNOWN"),
        int(report.get("selected_repository_count") or 0),
    )
    finalized = v5.attach_accounting(report, ledger)
    finalized["transport_retry_accounting"] = ledger.summary()
    actual = p2a.sha256_file(ledger.path)
    embedded = str(p2a.mapping(finalized.get("checkpoint")).get("sha256") or "")
    if embedded != actual:
        raise RuntimeError("final_checkpoint_artifact_hash_mismatch")
    finalized["checkpoint_artifact_hash_verified_final"] = True
    return finalized


if __name__ == "__main__":
    raise SystemExit(main())
