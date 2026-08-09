#!/usr/bin/env python3
"""Role-separated receipt audit for the eight-row static evaluator segment."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_static_evaluator_segment as producer  # noqa: E402

POLICY = "project_theseus_vcm_static_evaluator_segment_audit_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_static_evaluator_segment.json"


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
    audit_owner = p2a.resolve(str(cfg.get("audit_owner") or ""))
    if cfg.get("audit_policy") != POLICY or audit_owner != Path(__file__).resolve() or not audit_owner.is_file() or p2a.sha256_file(audit_owner) != cfg.get("audit_owner_sha256"):
        faults.append("audit_policy_or_owner_binding_invalid")
    execution_report = execution if execution is not None else p2a.read_json(p2a.resolve(str(cfg.get("report") or "")))
    if execution_report.get("trigger_state") != "GREEN" or execution_report.get("state") != "K2_05_STATIC_SEGMENT_EXECUTED_WITH_SCOPED_DISPOSITIONS" or execution_report.get("panel_admitted") is not False:
        faults.append("producer_state_invalid")
    configured = {int(row.get("index") or 0): row for row in p2a.dicts(cfg.get("rows"))}
    audited_rows: list[dict[str, Any]] = []
    for actual in p2a.dicts(execution_report.get("rows")):
        index = int(actual.get("index") or 0)
        expected = configured.get(index, {})
        sides = {side: p2a.mapping(actual.get(side)) for side in ("parent", "target")}
        expected_evaluator_sha = target_verifier_sha256(p2a.resolve(str(sides["target"].get("archive") or "")), str(expected.get("selected_verifier_path") or ""))
        for side, receipt in sides.items():
            if receipt.get("selected_verifier_path") != expected.get("selected_verifier_path") or receipt.get("command", [None])[1:] != expected.get("arguments"):
                faults.append(f"command_or_verifier_receipt_invalid:{index}:{side}")
            archive = p2a.resolve(str(receipt.get("archive") or ""))
            if not archive.is_file() or p2a.sha256_file(archive) != receipt.get("archive_sha256"):
                faults.append(f"archive_receipt_invalid:{index}:{side}")
            if receipt.get("project_selected_output_cap") is not None or receipt.get("stdout_complete") is not True or receipt.get("stderr_complete") is not True:
                faults.append(f"output_receipt_invalid:{index}:{side}")
            for stream in ("stdout", "stderr"):
                payload = str(receipt.get(stream) or "").encode("utf-8")
                if len(payload) != receipt.get(f"{stream}_bytes") or hashlib.sha256(payload).hexdigest() != receipt.get(f"{stream}_sha256"):
                    faults.append(f"retained_diagnostic_invalid:{index}:{side}:{stream}")
            if receipt.get("common_target_verifier_sha256") != expected_evaluator_sha or receipt.get("common_target_verifier_transplanted_to_parent") is not True:
                faults.append(f"common_evaluator_identity_invalid:{index}:{side}")
        parent_failed = sides["parent"].get("returncode") not in (None, 0) and not sides["parent"].get("boundary_hit")
        target_passed = sides["target"].get("returncode") == 0 and not sides["target"].get("boundary_hit")
        disposition = "QUALIFIED_PARENT_FAIL_TARGET_PASS" if parent_failed and target_passed else "INCONCLUSIVE_EXPERIMENT_STATIC_EVALUATOR_CONSTRUCT"
        if actual.get("parent_failed") is not parent_failed or actual.get("target_passed") is not target_passed or actual.get("disposition") != disposition:
            faults.append(f"disposition_rederivation_failed:{index}")
        audited_rows.append({"index": index, "parent_returncode": sides["parent"].get("returncode"), "target_returncode": sides["target"].get("returncode"), "parent_failed": parent_failed, "target_passed": target_passed, "disposition": disposition})
    if len(audited_rows) != int(cfg.get("expected_task_count") or 0) or set(configured) != {row["index"] for row in audited_rows}:
        faults.append("audited_denominator_invalid")
    qualified = sum(row["disposition"] == "QUALIFIED_PARENT_FAIL_TARGET_PASS" for row in audited_rows)
    if execution_report.get("qualified_task_count") != qualified or execution_report.get("inconclusive_task_count") != len(audited_rows) - qualified:
        faults.append("producer_count_rederivation_failed")
    return {"policy": POLICY, "created_utc": p2a.now(), "trigger_state": "GREEN" if not faults else "RED", "state": "K2_05_STATIC_SEGMENT_ROLE_SEPARATELY_REDERIVED" if not faults else "K2_05_STATIC_SEGMENT_AUDIT_FAILED", "faults": sorted(set(faults)), "config": {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)}, "producer_report": {"path": cfg.get("report"), "sha256": p2a.sha256_file(p2a.resolve(str(cfg.get("report") or ""))) if execution is None else None}, "audited_task_count": len(audited_rows), "qualified_task_count": qualified, "inconclusive_task_count": len(audited_rows) - qualified, "rows": audited_rows, "panel_admitted": False, "audit_kind": "role-separated rederivation", "network_or_dependency_execution_performed": False, "parent_target_or_evaluator_executions": 0, "candidate_or_control_calls": 0, "external_reference_calls": 0, "maximum_inference": cfg.get("audit_maximum_inference")}


def target_verifier_sha256(archive: Path, verifier: str) -> str:
    with tarfile.open(archive, "r:gz") as handle:
        members = [member for member in handle.getmembers() if member.isfile() and member.name.endswith(f"/{verifier}")]
        if len(members) != 1:
            return ""
        extracted = handle.extractfile(members[0])
        return hashlib.sha256(extracted.read()).hexdigest() if extracted is not None else ""


if __name__ == "__main__":
    raise SystemExit(main())
