#!/usr/bin/env python3
"""Role-separately rederive the committed K2.03 generic risk evidence chain."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402

POLICY = "project_theseus_vcm_instrument_builder_audit_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_instrument_builder_audit.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    config_path = p2a.resolve(args.config)
    report = audit(config_path)
    p2a.write_json(p2a.resolve(args.out or p2a.read_json(config_path)["report"]), report)
    print(json.dumps({key: report.get(key) for key in ("trigger_state", "state", "faults", "audited_attempt_count", "qualified_risk_classes", "candidate_or_control_calls", "external_reference_calls")}, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def audit(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    cfg = p2a.read_json(path)
    faults: list[str] = []
    owner = p2a.resolve(str(cfg.get("owner") or ""))
    if cfg.get("policy") != POLICY:
        faults.append("policy_invalid")
    if owner != Path(__file__).resolve() or not owner.is_file() or p2a.sha256_file(owner) != cfg.get("owner_sha256"):
        faults.append("owner_binding_invalid")
    attempts: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    for binding in p2a.dicts(cfg.get("attempts")):
        attempt_id = str(binding.get("attempt_id") or "")
        payload, error = git_blob(str(binding.get("commit") or ""), str(binding.get("path") or ""))
        if error:
            faults.append(f"git_blob_invalid:{attempt_id}:{error}")
            continue
        sha256 = hashlib.sha256(payload).hexdigest()
        if sha256 != binding.get("sha256"):
            faults.append(f"report_digest_invalid:{attempt_id}")
            continue
        try:
            report = json.loads(payload)
        except json.JSONDecodeError:
            faults.append(f"report_json_invalid:{attempt_id}")
            continue
        attempts[attempt_id] = report
        receipts.append({"attempt_id": attempt_id, "commit": binding.get("commit"), "path": binding.get("path"), "sha256": sha256, "state": report.get("state"), "trigger_state": report.get("trigger_state")})
        for key in ("candidate_or_control_calls", "external_reference_calls", "parent_target_or_evaluator_executions", "repository_runner_executions"):
            if report.get(key) != 0:
                faults.append(f"downstream_counter_invalid:{attempt_id}:{key}")

    v2 = attempts.get("v2", {})
    v2r = v2.get("risk_receipts", {})
    require_command(v2r.get("bun_online"), 0, False, "v2_bun_online", faults)
    require_command(v2r.get("bun_offline"), 0, True, "v2_bun_offline", faults)
    require_command(v2r.get("yarn_online"), 1, False, "v2_yarn_expected_gap", faults)
    if "jsdom@30.0.1" not in str(v2r.get("yarn_online", {}).get("stderr_head") or ""):
        faults.append("v2_yarn_transitive_engine_diagnostic_missing")
    if nested(v2r, "stores", "bun", "identity_sha256") != cfg.get("expected_bun_store_identity_sha256"):
        faults.append("v2_bun_store_identity_invalid")

    v5 = attempts.get("v5", {})
    v5r = v5.get("risk_receipts", {})
    if nested(v5r, "bun_disposable_symlink_rebase", "transformed_link_count") != 216 or nested(v5r, "bun_disposable_symlink_rebase", "broken_link_count_after") != 0:
        faults.append("v5_bun_symlink_rebase_invalid")
    require_command(v5r.get("bun_resume_offline"), 0, True, "v5_bun_resume", faults)
    require_command(v5r.get("yarn_online"), 0, False, "v5_yarn_online", faults)
    require_command(v5r.get("yarn_offline"), 0, True, "v5_yarn_offline", faults)
    require_command(v5r.get("typescript_transpilation"), 2, True, "v5_typescript_expected_nonzero", faults)
    if "rust_compilation" in v5r:
        faults.append("v5_rust_should_not_have_executed")
    if nested(v5r, "stores", "yarn", "identity_sha256") != cfg.get("expected_yarn_store_identity_sha256"):
        faults.append("v5_yarn_store_identity_invalid")

    v6 = attempts.get("v6", {})
    v6r = v6.get("risk_receipts", {})
    require_command(v6r.get("bun_resume_offline"), 0, True, "v6_bun_resume", faults)
    require_command(v6r.get("yarn_resume_offline"), 0, True, "v6_yarn_resume", faults)
    require_command(v6r.get("typescript_transpilation"), 2, True, "v6_typescript_expected_parent_gap", faults)
    require_command(v6r.get("rust_compilation"), 0, True, "v6_rust", faults)
    diagnostic = str(v6r.get("typescript_transpilation", {}).get("stdout_head") or "")
    if diagnostic != "framework/src/index-octane.ts(37,48): error TS2307: Cannot find module './styles.generated.ts' or its corresponding type declarations.\n":
        faults.append("v6_typescript_diagnostic_invalid")
    if v6r.get("qualified_bun_store_before") != v6r.get("qualified_bun_store_after"):
        faults.append("v6_bun_store_mutated")
    if content_projection(v6r.get("qualified_yarn_store_before")) != content_projection(v6r.get("qualified_yarn_store_after")):
        faults.append("v6_yarn_store_content_mutated")

    v7 = attempts.get("v7", {})
    v7r = v7.get("typescript_repair_receipts", {})
    if v7.get("trigger_state") != "GREEN" or v7.get("state") != "K2_03_NARROW_REAL_PARENT_TYPESCRIPT_MECHANICS_QUALIFIED":
        faults.append("v7_terminal_state_invalid")
    require_command(v7r.get("bun_resume_offline"), 0, True, "v7_bun_resume", faults)
    require_command(v7r.get("typescript_narrow_mechanics"), 0, True, "v7_typescript_narrow", faults)
    expected_command = p2a.strings(cfg.get("expected_v7_typescript_command"))
    observed_command = p2a.strings(v7r.get("typescript_narrow_mechanics", {}).get("command"))[1:]
    if observed_command != expected_command:
        faults.append("v7_typescript_command_invalid")
    if v7r.get("source_before") != v7r.get("source_after"):
        faults.append("v7_source_mutated")
    if v7r.get("qualified_bun_store_before") != v7r.get("qualified_bun_store_after"):
        faults.append("v7_bun_store_mutated")

    reserve = int(cfg.get("host_reserve_bytes") or 0)
    for attempt_id, report in (("v5", v5), ("v6", v6)):
        during = nested(report, "risk_receipts", "free_bytes_during")
        if not isinstance(during, int) or during < reserve:
            faults.append(f"host_reserve_invalid:{attempt_id}")
    v7_during = nested(v7, "typescript_repair_receipts", "free_bytes_during")
    if not isinstance(v7_during, int) or v7_during < reserve:
        faults.append("host_reserve_invalid:v7")

    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": "K2_03_GENERIC_ECOSYSTEM_RISK_EVIDENCE_ROLE_SEPARATELY_REDERIVED" if not faults else "K2_03_GENERIC_ECOSYSTEM_RISK_AUDIT_FAILED",
        "faults": sorted(set(faults)),
        "config": {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)},
        "audited_attempt_count": len(attempts),
        "attempt_receipts": receipts,
        "qualified_risk_classes": ["bun_real_lock_install_and_retained_replay", "yarn_real_lock_install_and_retained_replay", "typescript_real_parent_file_strict_no_emit_mechanics", "rust_parent_repository_untrusted_compilation"] if not faults else [],
        "preserved_caveats": ["task61_full_project_typecheck_is_inconclusive_missing_generated_parent_source", "typescript_green_is_narrow_real_file_mechanics_only", "repository_evaluators_and_vcm_packets_not_executed"],
        "audit_kind": "role-separated rederivation",
        "network_or_dependency_execution_performed": False,
        "repository_runner_executions": 0,
        "parent_target_or_evaluator_executions": 0,
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
        "maximum_inference": cfg.get("maximum_inference"),
    }


def git_blob(commit: str, path: str) -> tuple[bytes, str]:
    completed = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True, check=False)
    return completed.stdout, "" if completed.returncode == 0 else completed.stderr.decode("utf-8", "replace")[:500]


def require_command(raw: Any, returncode: int, network_denied: bool, label: str, faults: list[str]) -> None:
    row = raw if isinstance(raw, dict) else {}
    if row.get("returncode") != returncode or row.get("network_denied") is not network_denied or row.get("boundary_hit") is not False:
        faults.append(f"command_receipt_invalid:{label}")


def nested(raw: Any, *keys: str) -> Any:
    value = raw
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def content_projection(raw: Any) -> dict[str, Any]:
    row = raw if isinstance(raw, dict) else {}
    return {key: row.get(key) for key in ("path", "file_count", "bytes", "identity_sha256")}


if __name__ == "__main__":
    raise SystemExit(main())
