#!/usr/bin/env python3
"""Role-separated rederivation of the six immutable-resolution receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_immutable_resolution_segment as producer  # noqa: E402

POLICY = "project_theseus_vcm_immutable_resolution_segment_audit_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_immutable_resolution_segment.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    path = p2a.resolve(args.config)
    cfg = p2a.read_json(path)
    report = audit(path)
    p2a.write_json(p2a.resolve(args.out or str(cfg.get("audit_report") or "")), report)
    print(json.dumps({key: report.get(key) for key in ("trigger_state", "state", "faults", "audited_task_count", "qualified_task_count", "inconclusive_task_count", "candidate_or_control_calls", "external_reference_calls")}, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") == "GREEN" else 2


def audit(path: Path = DEFAULT_CONFIG, *, execution: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg, bound, faults = producer.preflight(path)
    if cfg.get("audit_policy") != POLICY:
        faults.append("audit_policy_invalid")
    owner = p2a.resolve(str(cfg.get("audit_owner") or ""))
    if owner != Path(__file__).resolve() or not owner.is_file() or p2a.sha256_file(owner) != cfg.get("audit_owner_sha256"):
        faults.append("audit_owner_binding_invalid")
    report = execution if execution is not None else p2a.read_json(p2a.resolve(str(cfg.get("report") or "")))
    if report.get("trigger_state") != "GREEN" or report.get("state") != "K2_05_IMMUTABLE_RESOLUTION_SEGMENT_EXECUTED_WITH_SCOPED_DISPOSITIONS" or report.get("panel_admitted") is not False:
        faults.append("producer_state_invalid")
    configured = {int(row.get("index") or 0): row for row in p2a.dicts(cfg.get("rows"))}
    audited: list[dict[str, Any]] = []
    for actual in p2a.dicts(report.get("rows")):
        index = int(actual.get("index") or 0)
        expected = configured.get(index, {})
        item = bound.get("rows", {}).get(index, {})
        archive = p2a.resolve(str(actual.get("target_archive") or ""))
        if not archive.is_file() or p2a.sha256_file(archive) != actual.get("target_archive_sha256") or archive != item.get("archive"):
            faults.append(f"archive_receipt_invalid:{index}")
        receipt = p2a.mapping(actual.get("receipt"))
        if receipt.get("predecessor_reuse") is not True:
            for stream in ("stdout", "stderr"):
                payload = str(receipt.get(stream) or "").encode("utf-8")
                if receipt and (len(payload) != receipt.get(f"{stream}_bytes") or hashlib.sha256(payload).hexdigest() != receipt.get(f"{stream}_sha256") or receipt.get(f"{stream}_complete") is not True):
                    faults.append(f"diagnostic_receipt_invalid:{index}:{stream}")
        if receipt.get("project_selected_output_cap") is not None:
            faults.append(f"output_cap_invalid:{index}")
        disposition = str(actual.get("disposition") or "")
        if disposition.startswith("RESOLUTION_QUALIFIED_IMMUTABLE_LOCK"):
            lock = p2a.mapping(receipt.get("lock"))
            lock_path = p2a.resolve(str(lock.get("path") or ""))
            independently_parsed, parse_faults = producer.validate_lock(str(expected.get("manager") or ""), lock_path)
            if not lock_path.is_file() or p2a.sha256_file(lock_path) != lock.get("sha256") or lock_path.stat().st_size != lock.get("bytes") or parse_faults or independently_parsed.get("package_count") != lock.get("package_count"):
                faults.append(f"qualified_lock_receipt_invalid:{index}")
            if disposition.endswith("REUSED_FROM_SEALED_PREDECESSOR"):
                previous = p2a.mapping(p2a.mapping(bound.get("predecessor")).get("producer_report"))
                previous_row = next((row for row in p2a.dicts(previous.get("rows")) if int(row.get("index") or 0) == index), {})
                previous_lock = p2a.mapping(p2a.mapping(previous_row.get("receipt")).get("lock"))
                if receipt.get("predecessor_reuse") is not True or not str(previous_row.get("disposition") or "").startswith("RESOLUTION_QUALIFIED_IMMUTABLE_LOCK") or previous_lock != lock:
                    faults.append(f"predecessor_lock_reuse_invalid:{index}")
        elif disposition == "INCONCLUSIVE_EXPERIMENT_DEPENDENCY_RESOLUTION":
            if receipt.get("returncode") in (None, 0) or receipt.get("boundary_hit") is not False:
                faults.append(f"resolution_disposition_invalid:{index}")
        elif disposition == "INCONCLUSIVE_EXPERIMENT_HOST_RESOURCE_BOUNDARY":
            if receipt.get("boundary_hit") is not True:
                faults.append(f"host_boundary_disposition_invalid:{index}")
        elif not disposition.startswith("INCONCLUSIVE_IMPLEMENTATION_") and disposition != "INCONCLUSIVE_EXPERIMENT_IMMUTABLE_OUTPUT_ALREADY_EXISTS":
            faults.append(f"unknown_disposition:{index}")
        audited.append({"index": index, "repository": actual.get("repository"), "manager": actual.get("manager"), "returncode": receipt.get("returncode"), "boundary_hit": receipt.get("boundary_hit"), "disposition": disposition, "lock_sha256": p2a.mapping(receipt.get("lock")).get("sha256")})
    if [row["index"] for row in audited] != producer.EXPECTED_INDICES or set(configured) != {row["index"] for row in audited}:
        faults.append("audited_denominator_invalid")
    qualified = sum(str(row["disposition"]).startswith("RESOLUTION_QUALIFIED_IMMUTABLE_LOCK") for row in audited)
    if report.get("qualified_task_count") != qualified or report.get("inconclusive_task_count") != len(audited) - qualified:
        faults.append("producer_counts_invalid")
    for key in ("package_installations", "source_build_executions", "repository_runner_executions", "parent_target_or_evaluator_executions", "candidate_or_control_calls", "external_reference_calls", "teacher_calls"):
        if report.get(key) != 0:
            faults.append(f"zero_boundary_invalid:{key}")
    return {"policy": POLICY, "created_utc": p2a.now(), "trigger_state": "GREEN" if not faults else "RED", "state": "K2_05_IMMUTABLE_RESOLUTION_ROLE_SEPARATELY_REDERIVED" if not faults else "K2_05_IMMUTABLE_RESOLUTION_AUDIT_FAILED", "faults": sorted(set(faults)), "config": {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)}, "producer_report": {"path": cfg.get("report"), "sha256": p2a.sha256_file(p2a.resolve(str(cfg.get("report") or ""))) if execution is None else None}, "audited_task_count": len(audited), "qualified_task_count": qualified, "inconclusive_task_count": len(audited) - qualified, "rows": audited, "panel_admitted": False, "audit_kind": "role-separated rederivation", "network_or_dependency_execution_performed": False, "parent_target_or_evaluator_executions": 0, "candidate_or_control_calls": 0, "external_reference_calls": 0, "maximum_inference": cfg.get("audit_maximum_inference")}


if __name__ == "__main__":
    raise SystemExit(main())
