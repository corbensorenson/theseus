#!/usr/bin/env python3
"""Retry-bounded, checkpoint-accounted transport for the frozen VCM selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

import theseus_assistant_p2a as p2a
import theseus_vcm_source_acquisition as v1
import theseus_vcm_source_acquisition_v3 as v3
import theseus_vcm_source_acquisition_v4 as v4


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_source_acquisition_v5.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_vcm_source_acquisition_v5.json"
DEFAULT_CHECKPOINT = ROOT / "reports" / "theseus_vcm_source_acquisition_v5_checkpoint.json"
PERMANENT_CANDIDATE_STATUSES = {404, 410}
TRANSIENT_STATUSES = {408, 429, 500, 502, 503, 504}


class CandidateMetadataUnavailable(RuntimeError):
    def __init__(self, status: int | None) -> None:
        super().__init__(f"candidate_metadata_unavailable:{status or 'unknown'}")
        self.status = status


class TransportRetriesExhausted(RuntimeError):
    pass


class RequestLedger:
    def __init__(self, checkpoint_path: Path, config_path: Path, policy: dict[str, Any]) -> None:
        self.path = checkpoint_path
        self.config_path = config_path
        self.policy = policy
        self.lock = threading.Lock()
        self.data: dict[str, Any] = {
            "policy": "project_theseus_vcm_source_acquisition_v5_checkpoint_v1",
            "created_utc": p2a.now(),
            "updated_utc": p2a.now(),
            "state": "RUNNING_PUBLIC_METADATA_ONLY",
            "config": v1.artifact(config_path),
            "retry_policy": policy,
            "logical_request_count": 0,
            "physical_attempt_count": 0,
            "successful_request_count": 0,
            "retry_attempt_count": 0,
            "permanent_candidate_failure_count": 0,
            "transport_failure_count": 0,
            "status_counts": {},
            "response_digests": [],
            "attempt_endpoint_hashes": [],
            "selected_source_identities_retained": False,
            "source_content_retrieval_opened": False,
            "candidate_packet_materialization_opened": False,
            "hidden_evaluation_opened": False,
            "local_model_calls": 0,
            "external_reference_calls": 0,
        }
        self._write_locked()

    def begin(self) -> None:
        with self.lock:
            self.data["logical_request_count"] += 1
            self._write_locked()

    def attempt(
        self,
        endpoint_hash: str,
        attempt_index: int,
        *,
        response_digest: str | None = None,
        status: int | None = None,
        permanent: bool = False,
        terminal_transport_failure: bool = False,
    ) -> None:
        with self.lock:
            self.data["physical_attempt_count"] += 1
            if attempt_index > 1:
                self.data["retry_attempt_count"] += 1
            self.data["attempt_endpoint_hashes"].append(endpoint_hash)
            if response_digest:
                self.data["successful_request_count"] += 1
                self.data["response_digests"].append(response_digest)
            if permanent:
                self.data["permanent_candidate_failure_count"] += 1
            if terminal_transport_failure:
                self.data["transport_failure_count"] += 1
            key = str(status) if status is not None else "unknown"
            counts = self.data["status_counts"]
            counts[key] = int(counts.get(key) or 0) + 1
            self._write_locked()

    def finalize(self, state: str, selected_count: int) -> None:
        with self.lock:
            self.data["state"] = state
            self.data["selected_repository_count"] = selected_count
            self.data["response_digest_chain_sha256"] = v1.stable_list_hash(
                self.data["response_digests"]
            )
            self.data["attempt_endpoint_hash_chain_sha256"] = v1.stable_list_hash(
                self.data["attempt_endpoint_hashes"]
            )
            self._write_locked()

    def summary(self) -> dict[str, Any]:
        with self.lock:
            return {
                key: self.data.get(key)
                for key in (
                    "state", "logical_request_count", "physical_attempt_count",
                    "successful_request_count", "retry_attempt_count",
                    "permanent_candidate_failure_count", "transport_failure_count",
                    "status_counts", "response_digest_chain_sha256",
                    "attempt_endpoint_hash_chain_sha256",
                )
            }

    def _write_locked(self) -> None:
        self.data["updated_utc"] = p2a.now()
        p2a.write_json(self.path, self.data)


class RetryingClient:
    def __init__(
        self,
        original: Callable[[str, dict[str, Any]], tuple[Any, str]],
        ledger: RequestLedger,
        policy: dict[str, Any],
    ) -> None:
        self.original = original
        self.ledger = ledger
        self.max_attempts = int(policy.get("maximum_attempts_per_logical_request") or 0)
        self.delays = [float(value) for value in policy.get("retry_delays_seconds", [])]

    def call(self, resource: str, fields: dict[str, Any]) -> tuple[Any, str]:
        self.ledger.begin()
        endpoint_hash = hashlib.sha256(
            json.dumps([resource, fields], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for attempt_index in range(1, self.max_attempts + 1):
            try:
                payload, digest = self.original(resource, fields)
                self.ledger.attempt(
                    endpoint_hash, attempt_index, response_digest=digest, status=200
                )
                return payload, digest
            except subprocess.CalledProcessError as exc:
                status = http_status(exc)
                permanent = status in PERMANENT_CANDIDATE_STATUSES
                retryable = is_retryable(exc, status)
                terminal = permanent or not retryable or attempt_index >= self.max_attempts
                self.ledger.attempt(
                    endpoint_hash,
                    attempt_index,
                    status=status,
                    permanent=permanent,
                    terminal_transport_failure=terminal and not permanent,
                )
                if permanent:
                    raise CandidateMetadataUnavailable(status) from exc
                if not retryable or attempt_index >= self.max_attempts:
                    raise TransportRetriesExhausted(
                        f"metadata_transport_exhausted:{status or 'unknown'}"
                    ) from exc
                time.sleep(self.delays[attempt_index - 1])
        raise TransportRetriesExhausted("metadata_transport_exhausted:invalid_policy")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default=p2a.rel(DEFAULT_OUT))
    parser.add_argument("--checkpoint", default=p2a.rel(DEFAULT_CHECKPOINT))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = p2a.resolve(args.config)
    report = preflight(config_path)
    if args.execute and report["trigger_state"] == "GREEN":
        config = p2a.read_json(config_path)
        retry_policy = p2a.mapping(config.get("transport_retry_policy"))
        ledger = RequestLedger(p2a.resolve(args.checkpoint), config_path, retry_policy)
        try:
            report = acquire(config_path, ledger, retry_policy)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
            report = {
                **report,
                "trigger_state": "PAUSED",
                "state": "PUBLIC_METADATA_RETRIES_EXHAUSTED_NO_SOURCE_EXPOSURE",
                "faults": [f"{type(exc).__name__}:{exc}"[:4000]],
            }
        report = attach_accounting(report, ledger)
        ledger.finalize(str(report.get("state") or "UNKNOWN"), int(report.get("selected_repository_count") or 0))
        report["transport_retry_accounting"] = ledger.summary()
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps(v1.summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "PAUSED"} else 2


def preflight(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    report = v3.preflight(config_path)
    config = p2a.read_json(config_path)
    policy = p2a.mapping(config.get("transport_retry_policy"))
    checkpoint = p2a.mapping(config.get("checkpoint_policy"))
    faults = p2a.strings(report.get("faults"))
    if (
        int(policy.get("maximum_attempts_per_logical_request") or 0) != 4
        or policy.get("retry_delays_seconds") != [0.25, 0.5, 1.0]
        or set(int(value) for value in policy.get("transient_http_statuses", [])) != TRANSIENT_STATUSES
        or set(int(value) for value in policy.get("permanent_candidate_http_statuses", [])) != PERMANENT_CANDIDATE_STATUSES
        or policy.get("retry_unknown_network_failures") is not True
        or checkpoint.get("write_after_every_physical_attempt") is not True
        or checkpoint.get("candidate_identities_retained") is not False
        or checkpoint.get("source_content_retained") is not False
    ):
        faults.append("transport_retry_or_checkpoint_policy_invalid")
    report["faults"] = sorted(set(faults))
    report["trigger_state"] = "GREEN" if not faults else "RED"
    report["state"] = (
        "METADATA_SELECTION_V5_RETRY_CHECKPOINT_PREFLIGHT_GREEN"
        if not faults
        else "INVALID_PREFLIGHT"
    )
    return report


def acquire(
    config_path: Path,
    ledger: RequestLedger,
    retry_policy: dict[str, Any],
) -> dict[str, Any]:
    original_api = v1.api_json
    original_qualifier = v1.qualify_metadata
    client = RetryingClient(original_api, ledger, retry_policy)
    v1.api_json = client.call
    v1.qualify_metadata = qualify_metadata
    try:
        return v3.acquire(config_path)
    finally:
        v1.api_json = original_api
        v1.qualify_metadata = original_qualifier


def qualify_metadata(
    candidate: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, Any], list[str], int, list[str]]:
    try:
        return v4.qualify_metadata(candidate, config)
    except CandidateMetadataUnavailable as exc:
        repository = str(candidate.get("repository") or "")
        number = int(candidate.get("pull_request") or 0)
        return (
            {
                "opaque_source_id": hashlib.sha256(
                    f"vcm-source:{repository}#{number}".encode()
                ).hexdigest(),
                "repository": repository,
                "pull_request": number,
                "query_language": candidate.get("query_language"),
                "selection_rank_sha256": candidate.get("rank"),
                "metadata_qualified": False,
                "candidate_content_retrieved": False,
                "candidate_packet_materialized": False,
            },
            [f"candidate_metadata_http_{exc.status or 'unknown'}"],
            0,
            [],
        )


def attach_accounting(report: dict[str, Any], ledger: RequestLedger) -> dict[str, Any]:
    summary = ledger.summary()
    counters = dict(p2a.mapping(report.get("counters")))
    counters["public_metadata_requests"] = summary["logical_request_count"]
    counters["public_metadata_request_attempts"] = summary["physical_attempt_count"]
    report["counters"] = counters
    transport = dict(p2a.mapping(report.get("transport")))
    transport["request_count"] = summary["logical_request_count"]
    transport["physical_attempt_count"] = summary["physical_attempt_count"]
    transport["retry_attempt_count"] = summary["retry_attempt_count"]
    transport["response_digest_chain_sha256"] = v1.stable_list_hash(
        ledger.data["response_digests"]
    )
    report["transport"] = transport
    report["checkpoint"] = v1.artifact(ledger.path)
    return report


def http_status(exc: subprocess.CalledProcessError) -> int | None:
    text = f"{exc.stderr or ''}\n{exc.stdout or ''}"
    match = re.search(r"HTTP\s+(\d{3})", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def is_retryable(exc: subprocess.CalledProcessError, status: int | None) -> bool:
    if status in TRANSIENT_STATUSES:
        return True
    text = f"{exc.stderr or ''}\n{exc.stdout or ''}".lower()
    if status == 403 and ("rate limit" in text or "secondary rate" in text):
        return True
    return status is None


if __name__ == "__main__":
    raise SystemExit(main())
