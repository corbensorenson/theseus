#!/usr/bin/env python3
"""Executable SCF public-release conformance fragment for Theseus.

This is intentionally a registry/kernel conformance check, not a domain
evaluator. It uses only the Python standard library and validates that the
active project registry exposes the Stable Capability Fields public-release
controls required for safe routing and self-improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "project_manifest_registry.json"
DEFAULT_OUT = ROOT / "reports" / "scf_reference_kernel_v1.json"
DEFAULT_MD_OUT = ROOT / "reports" / "scf_reference_kernel_v1.md"
EXPECTED_SCHEMA = "stable_capability_fields_public_release_1_0"

FIELD_BOOL_PATHS = [
    ["identity_policy", "exact_content_binding_required"],
    ["evidence_registry_policy", "source_events_append_only"],
    ["route_validation_policy", "validator_receipt_required"],
    ["route_validation_policy", "caller_binding_required"],
    ["adaptation_policy", "sealed_epoch_required"],
    ["adaptation_policy", "pinned_updater_required"],
    ["adaptation_policy", "approved_data_receipts_required"],
    ["composition_policy", "dependency_cycles_bundled"],
    ["governance_policy", "change_classification_required"],
]

BINDING_BOOL_PATHS = [
    ["content_binding", "exact_hash_binding_required"],
    ["route_binding", "validator_receipt_required"],
    ["claim_binding", "evaluator_overlap_record_required"],
    ["lease_binding", "expiry_required"],
    ["lease_binding", "fail_closed_required"],
    ["migration_binding", "solvency_class_required"],
]

FIELD_SECTION_EXPECTATIONS = {
    "identity_policy": [
        "exact_content_binding_required",
        "identity_objects",
        "hash_bindings",
        "alias_policy",
        "mutable_alias_policy",
    ],
    "evidence_registry_policy": [
        "source_events_append_only",
        "materialized_view_policy",
        "claim_model",
        "defeater_policy",
        "waiver_policy",
        "transfer_witness_policy",
    ],
    "route_validation_policy": [
        "proposer_trust_model",
        "validator_receipt_required",
        "caller_binding_required",
        "lease_policy",
        "role_constraints",
    ],
    "adaptation_policy": [
        "sealed_epoch_required",
        "pinned_updater_required",
        "approved_data_receipts_required",
        "budget_policy",
        "sentinel_policy",
        "baseline_policy",
    ],
    "composition_policy": [
        "dependency_cycles_bundled",
        "shared_artifact_policy",
        "toxic_composition_policy",
        "emergent_risk_policy",
    ],
    "governance_policy": [
        "change_classification_required",
        "trust_root_policy",
        "threshold_change_policy",
        "timelock_policy",
        "federation_policy",
        "emergency_power_policy",
    ],
}

BINDING_SECTION_EXPECTATIONS = {
    "content_binding": [
        "exact_hash_binding_required",
        "artifact_hash_policy",
        "manifest_hash_policy",
        "contract_hash_policy",
        "dependency_closure_hash_policy",
        "evidence_bundle_hash_policy",
    ],
    "route_binding": [
        "role_profiles",
        "caller_context_binding",
        "validator_receipt_required",
        "request_context_hash_policy",
        "fallback_binding",
    ],
    "claim_binding": [
        "qualification_claims",
        "assumptions",
        "defeaters",
        "waivers",
        "transfer_witnesses",
        "evaluator_overlap_record_required",
    ],
    "lease_binding": [
        "expiry_required",
        "authority_grant_binding",
        "state_binding",
        "consequence_budget",
        "fail_closed_required",
    ],
    "adaptation_binding": [
        "epoch_token_policy",
        "updater_revision_policy",
        "data_receipts_policy",
        "sentinel_binding",
        "journal_policy",
        "baseline_binding",
    ],
    "migration_binding": [
        "solvency_class_required",
        "state_relation_binding",
        "rehearsal_required",
        "rollback_or_compensation",
    ],
}

REQUIRED_HASH_BINDINGS = {
    "artifact_hash",
    "manifest_hash",
    "contract_hash",
    "profile_hash",
    "dependency_closure_hash",
    "evaluator_policy_hash",
    "state_relation_hash",
    "evidence_bundle_hashes",
}

REQUIRED_IDENTITY_OBJECTS = {
    "contract_revision",
    "implementation_revision",
    "qualification_claim",
    "route_authorization",
    "source_event",
    "materialized_view",
}

REQUIRED_HARD_VETOES = {
    "identity_mismatch",
    "materialized_view_tamper",
    "expired_route_lease",
    "missing_evaluator_overlap_record",
    "unbounded_canary_consequence",
    "adaptation_epoch_breach",
    "non_solvent_state_migration",
    "threshold_or_timelock_bypass",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def stable_json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def get_nested(row: dict[str, Any], path: list[str], default: Any = None) -> Any:
    value: Any = row
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def missing_required(row: dict[str, Any], required: list[str]) -> list[str]:
    missing = []
    for key in required:
        value = row.get(key)
        if value in (None, "", [], {}):
            missing.append(key)
    return missing


def missing_true(row: dict[str, Any], paths: list[list[str]]) -> list[str]:
    missing = []
    for path in paths:
        if not get_nested(row, path, False):
            missing.append(".".join(path))
    return missing


def add_check(
    checks: list[dict[str, Any]],
    *,
    check_id: str,
    passed: bool,
    scope: str,
    severity: str,
    details: dict[str, Any] | None = None,
) -> None:
    checks.append(
        {
            "check_id": check_id,
            "passed": bool(passed),
            "scope": scope,
            "severity": severity,
            "details": details or {},
        }
    )


def validate_contract(registry: dict[str, Any], config_path: Path, checks: list[dict[str, Any]]) -> None:
    contract = registry.get("stable_capability_field_contract")
    if not isinstance(contract, dict):
        add_check(
            checks,
            check_id="contract_present",
            passed=False,
            scope="stable_capability_field_contract",
            severity="hard",
            details={"reason": "missing SCF contract"},
        )
        return

    add_check(
        checks,
        check_id="schema_version_public_release",
        passed=contract.get("schema_version") == EXPECTED_SCHEMA,
        scope="stable_capability_field_contract",
        severity="hard",
        details={"schema_version": contract.get("schema_version"), "expected": EXPECTED_SCHEMA},
    )
    source_document = str(contract.get("source_document") or "")
    source_path = (ROOT / source_document).resolve() if source_document else Path("")
    source_exists = bool(source_document and source_path.exists())
    add_check(
        checks,
        check_id="source_document_exists",
        passed=source_exists,
        scope="stable_capability_field_contract.source_document",
        severity="hard",
        details={
            "source_document": source_document,
            "source_hash": file_hash(source_path) if source_exists else "",
        },
    )
    add_check(
        checks,
        check_id="registry_content_hash_bound",
        passed=config_path.exists(),
        scope=str(config_path.relative_to(ROOT)),
        severity="hard",
        details={"registry_hash": file_hash(config_path) if config_path.exists() else ""},
    )
    add_check(
        checks,
        check_id="contract_hash_bound",
        passed=True,
        scope="stable_capability_field_contract",
        severity="info",
        details={"contract_hash": stable_json_hash(contract)},
    )

    field_sections = contract.get("field_section_required_fields") if isinstance(contract.get("field_section_required_fields"), dict) else {}
    for section, required in FIELD_SECTION_EXPECTATIONS.items():
        configured = set(field_sections.get(section, []))
        missing = sorted(set(required) - configured)
        add_check(
            checks,
            check_id="contract_requires_field_section",
            passed=not missing,
            scope=f"field_section_required_fields.{section}",
            severity="hard",
            details={"missing": missing},
        )

    binding_sections = contract.get("implementation_binding_section_required_fields") if isinstance(contract.get("implementation_binding_section_required_fields"), dict) else {}
    for section, required in BINDING_SECTION_EXPECTATIONS.items():
        configured = set(binding_sections.get(section, []))
        missing = sorted(set(required) - configured)
        add_check(
            checks,
            check_id="contract_requires_binding_section",
            passed=not missing,
            scope=f"implementation_binding_section_required_fields.{section}",
            severity="hard",
            details={"missing": missing},
        )

    hard_vetoes = set(contract.get("hard_vetoes", []))
    missing_vetoes = sorted(REQUIRED_HARD_VETOES - hard_vetoes)
    add_check(
        checks,
        check_id="public_release_hard_vetoes_present",
        passed=not missing_vetoes,
        scope="stable_capability_field_contract.hard_vetoes",
        severity="hard",
        details={"missing": missing_vetoes},
    )


def validate_abstractions(registry: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    contract = registry.get("stable_capability_field_contract") if isinstance(registry.get("stable_capability_field_contract"), dict) else {}
    field_required = [str(item) for item in contract.get("field_required_fields", [])]
    section_required = contract.get("field_section_required_fields") if isinstance(contract.get("field_section_required_fields"), dict) else {}

    for abstraction in registry.get("abstractions", []):
        if not isinstance(abstraction, dict) or abstraction.get("status") not in {"live", "retained"}:
            continue
        abstraction_id = str(abstraction.get("id") or "unknown")
        field = abstraction.get("stable_capability_field") if isinstance(abstraction.get("stable_capability_field"), dict) else {}
        missing = missing_required(field, field_required)
        add_check(
            checks,
            check_id="field_required_fields_present",
            passed=not missing,
            scope=abstraction_id,
            severity="hard",
            details={"missing": missing},
        )
        add_check(
            checks,
            check_id="field_id_matches_abstraction",
            passed=field.get("field_id") == abstraction_id,
            scope=abstraction_id,
            severity="hard",
            details={"field_id": field.get("field_id")},
        )
        for section, required in section_required.items():
            section_payload = field.get(section) if isinstance(field.get(section), dict) else {}
            missing = missing_required(section_payload, [str(item) for item in required])
            add_check(
                checks,
                check_id="field_section_complete",
                passed=not missing,
                scope=f"{abstraction_id}.{section}",
                severity="hard",
                details={"missing": missing},
            )
        bool_missing = missing_true(field, FIELD_BOOL_PATHS)
        add_check(
            checks,
            check_id="field_boolean_clauses_true",
            passed=not bool_missing,
            scope=abstraction_id,
            severity="hard",
            details={"missing_or_false": bool_missing},
        )
        identity_policy = field.get("identity_policy") if isinstance(field.get("identity_policy"), dict) else {}
        missing_hashes = sorted(REQUIRED_HASH_BINDINGS - set(identity_policy.get("hash_bindings", [])))
        missing_objects = sorted(REQUIRED_IDENTITY_OBJECTS - set(identity_policy.get("identity_objects", [])))
        add_check(
            checks,
            check_id="field_identity_bindings_cover_public_release_objects",
            passed=not missing_hashes and not missing_objects,
            scope=abstraction_id,
            severity="hard",
            details={"missing_hash_bindings": missing_hashes, "missing_identity_objects": missing_objects},
        )


def validate_implementations(registry: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    contract = registry.get("stable_capability_field_contract") if isinstance(registry.get("stable_capability_field_contract"), dict) else {}
    binding_required = [str(item) for item in contract.get("implementation_binding_required_fields", [])]
    section_required = contract.get("implementation_binding_section_required_fields") if isinstance(contract.get("implementation_binding_section_required_fields"), dict) else {}
    abstractions = {
        str(row.get("id") or ""): row
        for row in registry.get("abstractions", [])
        if isinstance(row, dict) and row.get("id")
    }

    for implementation in registry.get("implementations", []):
        if not isinstance(implementation, dict) or implementation.get("status") not in {"live", "retained"}:
            continue
        implementation_id = str(implementation.get("id") or "unknown")
        abstraction_id = str(implementation.get("abstraction_id") or "")
        abstraction = abstractions.get(abstraction_id, {})
        field_contract = get_nested(abstraction, ["stable_capability_field", "contract_version"], "")
        binding = implementation.get("stable_capability_binding") if isinstance(implementation.get("stable_capability_binding"), dict) else {}
        missing = missing_required(binding, binding_required)
        add_check(
            checks,
            check_id="binding_required_fields_present",
            passed=not missing,
            scope=implementation_id,
            severity="hard",
            details={"missing": missing},
        )
        add_check(
            checks,
            check_id="binding_targets_abstraction_field",
            passed=binding.get("field_id") == abstraction_id and (not field_contract or binding.get("contract_version") == field_contract),
            scope=implementation_id,
            severity="hard",
            details={
                "binding_field_id": binding.get("field_id"),
                "abstraction_id": abstraction_id,
                "binding_contract": binding.get("contract_version"),
                "field_contract": field_contract,
            },
        )
        for section, required in section_required.items():
            section_payload = binding.get(section) if isinstance(binding.get(section), dict) else {}
            missing = missing_required(section_payload, [str(item) for item in required])
            add_check(
                checks,
                check_id="binding_section_complete",
                passed=not missing,
                scope=f"{implementation_id}.{section}",
                severity="hard",
                details={"missing": missing},
            )
        bool_missing = missing_true(binding, BINDING_BOOL_PATHS)
        add_check(
            checks,
            check_id="binding_boolean_clauses_true",
            passed=not bool_missing,
            scope=implementation_id,
            severity="hard",
            details={"missing_or_false": bool_missing},
        )
        if get_nested(implementation, ["routing_eligibility", "eligible"], False):
            route_ok = bool(get_nested(binding, ["route_binding", "validator_receipt_required"], False))
            lease_ok = bool(get_nested(binding, ["lease_binding", "expiry_required"], False)) and bool(get_nested(binding, ["lease_binding", "fail_closed_required"], False))
            add_check(
                checks,
                check_id="routing_eligible_binding_has_validator_and_lease",
                passed=route_ok and lease_ok,
                scope=implementation_id,
                severity="hard",
                details={"validator_receipt_required": route_ok, "expiring_fail_closed_lease": lease_ok},
            )


def build_report(config_path: Path) -> dict[str, Any]:
    registry = read_json(config_path)
    checks: list[dict[str, Any]] = []
    validate_contract(registry, config_path, checks)
    validate_abstractions(registry, checks)
    validate_implementations(registry, checks)
    hard_failures = [row for row in checks if row["severity"] == "hard" and not row["passed"]]
    warn_failures = [row for row in checks if row["severity"] == "warning" and not row["passed"]]
    return {
        "policy": "project_theseus_scf_reference_kernel_v1",
        "created_utc": utc_now(),
        "trigger_state": "RED" if hard_failures else "YELLOW" if warn_failures else "GREEN",
        "summary": {
            "check_count": len(checks),
            "hard_failure_count": len(hard_failures),
            "warning_failure_count": len(warn_failures),
            "abstraction_count": len([row for row in registry.get("abstractions", []) if isinstance(row, dict)]),
            "implementation_count": len([row for row in registry.get("implementations", []) if isinstance(row, dict)]),
            "schema_version": get_nested(registry, ["stable_capability_field_contract", "schema_version"], ""),
            "source_document": get_nested(registry, ["stable_capability_field_contract", "source_document"], ""),
        },
        "checks": checks,
        "failures": hard_failures + warn_failures,
        "limits": {
            "production_cryptography": "not_implemented",
            "distributed_consensus": "not_implemented",
            "remote_attestation": "not_implemented",
            "semantic_state_equivalence": "not_proven",
            "purpose": "executable registry conformance fragment, not an empirical capability claim",
        },
        "external_inference_calls": 0,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# SCF Reference Kernel v1",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- checks: `{summary.get('check_count')}`",
        f"- hard failures: `{summary.get('hard_failure_count')}`",
        f"- warning failures: `{summary.get('warning_failure_count')}`",
        f"- schema: `{summary.get('schema_version')}`",
        f"- source document: `{summary.get('source_document')}`",
        "",
        "## Scope",
        "",
        "This is the executable Stable Capability Fields public-release conformance fragment for Theseus' registry. It validates exact identity policy, source-event evidence policy, route validation, leases, adaptation envelopes, evaluator overlap, migration solvency, and governance clauses. It is not a production cryptography, consensus, remote-attestation, or empirical model-capability claim.",
    ]
    failures = report.get("failures", [])
    if failures:
        lines.extend(["", "## Failures", ""])
        for row in failures[:50]:
            lines.append(f"- `{row.get('check_id')}` on `{row.get('scope')}`: `{row.get('details')}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    report = build_report(config_path)
    out = Path(args.out)
    md_out = Path(args.markdown_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    md_out.write_text(render_markdown(report))
    if args.gate:
        print(json.dumps({k: report[k] for k in ["policy", "created_utc", "trigger_state", "summary"]}, indent=2))
    else:
        print(json.dumps(report, indent=2))
    return 0 if report.get("trigger_state") != "RED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
