#!/usr/bin/env python3
"""Role-separated audit of the exact sdist static risk classification."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_untrusted_build_preflight as producer  # noqa: E402

POLICY = "project_theseus_vcm_untrusted_build_preflight_audit_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_untrusted_build_preflight.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    path = p2a.resolve(args.config)
    cfg = p2a.read_json(path)
    report = audit(path)
    p2a.write_json(p2a.resolve(args.out or str(cfg.get("audit_report") or "")), report)
    print(json.dumps({key: report.get(key) for key in ("trigger_state", "state", "faults", "risk_class", "candidate_or_control_calls", "external_reference_calls")}, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") == "GREEN" else 2


def audit(path: Path = DEFAULT_CONFIG, *, execution: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg, bound, faults = producer.preflight(path)
    if cfg.get("audit_policy") != POLICY:
        faults.append("audit_policy_invalid")
    report = execution if execution is not None else p2a.read_json(p2a.resolve(str(cfg.get("report") or "")))
    if report.get("trigger_state") != "GREEN" or report.get("state") != "EXACT_SDIST_STATIC_PREFLIGHT_QUALIFIED":
        faults.append("producer_state_invalid")
    store: Path = bound["store"]
    package = p2a.mapping(cfg.get("package"))
    retained = p2a.mapping(p2a.mapping(report.get("receipt")).get("retained_sdist"))
    if not store.is_file() or p2a.sha256_file(store) != package.get("sha256") or retained.get("sha256") != package.get("sha256") or retained.get("bytes") != store.stat().st_size:
        faults.append("retained_sdist_identity_invalid")
    inspection, inspection_faults = producer.inspect_sdist(store, p2a.mapping(cfg.get("limits"))) if store.is_file() else ({}, ["retained_sdist_missing"])
    faults.extend(inspection_faults)
    observed = p2a.mapping(p2a.mapping(report.get("receipt")).get("inspection"))
    if inspection != observed:
        faults.append("inspection_rederivation_mismatch")
    risk = str(inspection.get("risk_class") or "")
    if risk != "LOW_COMPLEXITY_LEGACY_SETUP_PY_ELIGIBLE_FOR_NETWORK_DENIED_SANDBOX_CANARY":
        faults.append("risk_class_not_eligible")
    for key in ("source_build_executions", "package_installations", "repository_runner_executions", "parent_target_or_evaluator_executions", "candidate_or_control_calls", "external_reference_calls", "teacher_calls"):
        if report.get(key) != 0:
            faults.append(f"zero_boundary_invalid:{key}")
    return {"policy": POLICY, "created_utc": p2a.now(), "trigger_state": "GREEN" if not faults else "RED", "state": "EXACT_SDIST_STATIC_RISK_ROLE_SEPARATELY_REDERIVED" if not faults else "EXACT_SDIST_STATIC_RISK_AUDIT_FAILED", "faults": sorted(set(faults)), "config": {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)}, "producer_report": {"path": cfg.get("report"), "sha256": p2a.sha256_file(p2a.resolve(str(cfg.get("report") or ""))) if execution is None else None}, "risk_class": risk, "member_receipts_sha256": inspection.get("member_receipts_sha256"), "audit_kind": "role-separated rederivation", "network_or_build_execution_performed": False, "candidate_or_control_calls": 0, "external_reference_calls": 0, "maximum_inference": cfg.get("audit_maximum_inference")}


if __name__ == "__main__":
    raise SystemExit(main())
