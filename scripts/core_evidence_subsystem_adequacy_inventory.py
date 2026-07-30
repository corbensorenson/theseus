#!/usr/bin/env python3
"""Audit canonical subsystem ownership before integrated D1 experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "core_evidence_subsystem_adequacy.json"
DEFAULT_OUT = ROOT / "reports" / "core_evidence_subsystem_adequacy_inventory.json"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_repo_path(raw: str) -> Path:
    path = (ROOT / raw).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError(f"path escapes repository: {raw}")
    return path


def implementation_index(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = registry.get("implementations")
    if not isinstance(rows, list):
        raise ValueError("registry implementations must be a list")
    return {
        str(row["id"]): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }


def audit_owner(
    owner: dict[str, Any],
    implementations: dict[str, dict[str, Any]],
    *,
    run_tests: bool,
) -> dict[str, Any]:
    implementation_ids = owner.get("implementation_ids")
    if not isinstance(implementation_ids, list):
        raise ValueError("owner implementation_ids must be a list")
    required_role = owner.get("required_role")
    findings: list[dict[str, Any]] = []
    bound_implementations: list[dict[str, Any]] = []

    if not implementation_ids:
        findings.append(
            {
                "check": "registered_live_implementation",
                "passed": False,
                "detail": "no canonical implementation id is registered",
            }
        )

    for implementation_id in implementation_ids:
        implementation = implementations.get(str(implementation_id))
        if implementation is None:
            findings.append(
                {
                    "check": "implementation_registered",
                    "passed": False,
                    "detail": implementation_id,
                }
            )
            continue
        entrypoint_raw = implementation.get("canonical_entrypoint")
        entrypoint = (
            resolve_repo_path(entrypoint_raw)
            if isinstance(entrypoint_raw, str)
            else None
        )
        eligible = bool(
            isinstance(implementation.get("routing_eligibility"), dict)
            and implementation["routing_eligibility"].get("eligible") is True
        )
        row = {
            "implementation_id": implementation_id,
            "status": implementation.get("status"),
            "role": implementation.get("role"),
            "canonical_entrypoint": entrypoint_raw,
            "canonical_entrypoint_sha256": (
                sha256_path(entrypoint) if entrypoint and entrypoint.is_file() else None
            ),
            "route_eligible": eligible,
            "verification_command": implementation.get("verification_command"),
        }
        bound_implementations.append(row)
        findings.extend(
            [
                {
                    "check": f"{implementation_id}:live",
                    "passed": implementation.get("status") == "live",
                    "detail": implementation.get("status"),
                },
                {
                    "check": f"{implementation_id}:role",
                    "passed": implementation.get("role") == required_role,
                    "detail": implementation.get("role"),
                },
                {
                    "check": f"{implementation_id}:route_eligible",
                    "passed": eligible,
                    "detail": eligible,
                },
                {
                    "check": f"{implementation_id}:canonical_entrypoint",
                    "passed": bool(entrypoint and entrypoint.is_file()),
                    "detail": entrypoint_raw,
                },
                {
                    "check": f"{implementation_id}:verification_command",
                    "passed": bool(implementation.get("verification_command")),
                    "detail": implementation.get("verification_command"),
                },
            ]
        )

    candidate_entrypoints = owner.get("candidate_entrypoints")
    if isinstance(candidate_entrypoints, list) and candidate_entrypoints:
        dependency_union = {
            str(item)
            for implementation_id in implementation_ids
            for item in (
                implementations.get(str(implementation_id), {}).get("dependencies")
                or []
            )
        }
        for raw in candidate_entrypoints:
            path = resolve_repo_path(str(raw))
            findings.extend(
                [
                    {
                        "check": f"candidate_entrypoint:{raw}:exists",
                        "passed": path.is_file(),
                        "detail": str(raw),
                    },
                    {
                        "check": f"candidate_entrypoint:{raw}:registry_bound",
                        "passed": str(raw) in dependency_union,
                        "detail": sorted(dependency_union),
                    },
                ]
            )

    test_rows: list[dict[str, Any]] = []
    tests = owner.get("tests")
    if not isinstance(tests, list) or not tests:
        findings.append(
            {"check": "focused_tests_declared", "passed": False, "detail": tests}
        )
    else:
        for raw in tests:
            path = resolve_repo_path(str(raw))
            test_rows.append(
                {
                    "path": str(raw),
                    "exists": path.is_file(),
                    "sha256": sha256_path(path) if path.is_file() else None,
                }
            )
        findings.append(
            {
                "check": "focused_tests_exist",
                "passed": all(row["exists"] for row in test_rows),
                "detail": [row["path"] for row in test_rows if not row["exists"]],
            }
        )

    test_receipt: dict[str, Any] = {
        "run": run_tests,
        "returncode": None,
        "passed": None,
        "stdout_tail": "",
        "stderr_tail": "",
    }
    if run_tests and test_rows and all(row["exists"] for row in test_rows):
        command = ["python3", "-m", "pytest", "-q", *[row["path"] for row in test_rows]]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        test_receipt = {
            "run": True,
            "command": command,
            "returncode": completed.returncode,
            "passed": completed.returncode == 0,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }
        findings.append(
            {
                "check": "focused_tests_pass",
                "passed": completed.returncode == 0,
                "detail": completed.returncode,
            }
        )

    mechanics_green = bool(findings) and all(bool(row["passed"]) for row in findings)
    state = (
        "MECHANICS_GREEN_PENDING_CAUSAL_SMOKE"
        if mechanics_green and run_tests
        else "INCONCLUSIVE_IMPLEMENTATION"
        if not mechanics_green
        else "INVENTORY_GREEN_TESTS_NOT_RUN"
    )
    return {
        "owner_id": owner.get("owner_id"),
        "state": state,
        "required_interventions": owner.get("required_interventions", []),
        "candidate_entrypoints": owner.get("candidate_entrypoints", []),
        "implementations": bound_implementations,
        "tests": test_rows,
        "test_receipt": test_receipt,
        "findings": findings,
    }


def build_report(config_path: Path, *, run_tests: bool) -> dict[str, Any]:
    config = read_json(config_path)
    registry_path = resolve_repo_path(str(config["registry"]))
    disposition_path = resolve_repo_path(str(config["qualified_worker_disposition"]))
    terminal_disposition_path = resolve_repo_path(
        str(config["development_terminal_disposition"])
    )
    freeze_path = resolve_repo_path(str(config["qualified_worker_freeze"]))
    worker_source_path = resolve_repo_path(str(config["current_worker_source"]))
    active_worker_config_path = resolve_repo_path(str(config["active_worker_config"]))
    registry = read_json(registry_path)
    disposition = read_json(disposition_path)
    terminal_disposition = read_json(terminal_disposition_path)
    freeze = read_json(freeze_path)
    if disposition.get("disposition") != config.get("required_worker_disposition"):
        raise ValueError("qualified worker disposition does not match config")
    if terminal_disposition.get("disposition") != config.get(
        "required_development_terminal_disposition"
    ):
        raise ValueError("development terminal disposition does not match config")

    implementations = implementation_index(registry)
    frozen_source_identities = (
        freeze.get("candidate_source_identities")
        if isinstance(freeze.get("candidate_source_identities"), dict)
        else {}
    )
    frozen_worker_sha256 = str(frozen_source_identities.get("worker_sha256") or "")
    current_worker_sha256 = sha256_path(worker_source_path)
    worker_identity_current = (
        bool(frozen_worker_sha256)
        and current_worker_sha256 == frozen_worker_sha256
    )
    owner_rows = config.get("owners")
    if not isinstance(owner_rows, list) or not owner_rows:
        raise ValueError("owners must be a non-empty list")
    owners = [
        audit_owner(owner, implementations, run_tests=run_tests)
        for owner in owner_rows
        if isinstance(owner, dict)
    ]
    counts = {
        "owners": len(owners),
        "mechanics_green": sum(
            row["state"] == "MECHANICS_GREEN_PENDING_CAUSAL_SMOKE" for row in owners
        ),
        "inventory_green_tests_not_run": sum(
            row["state"] == "INVENTORY_GREEN_TESTS_NOT_RUN" for row in owners
        ),
        "inconclusive_implementation": sum(
            row["state"] == "INCONCLUSIVE_IMPLEMENTATION" for row in owners
        ),
    }
    all_mechanics_green = counts["mechanics_green"] == counts["owners"]
    all_inventory_green = (
        counts["mechanics_green"] + counts["inventory_green_tests_not_run"]
        == counts["owners"]
    )
    trigger_state = (
        "RED_WORKER_SUCCESSOR_REQUIRED"
        if config.get("active_worker_state")
        == "TERMINAL_FAIL_SELECT_MATERIALLY_STRONGER_LOCAL_SUCCESSOR"
        else "RED_WORKER_REQUALIFICATION_REQUIRED"
        if not worker_identity_current
        else "GREEN_ADVANCE_TO_DEVELOPMENT_CAUSAL_SMOKES"
        if all_mechanics_green
        else "GREEN_INVENTORY_TESTS_NOT_RUN"
        if all_inventory_green and not run_tests
        else "RED_INCONCLUSIVE_IMPLEMENTATION"
    )
    return {
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "policy": config.get("policy"),
        "campaign_id": config.get("campaign_id"),
        "config_sha256": sha256_path(config_path),
        "registry_sha256": sha256_path(registry_path),
        "qualified_worker_disposition_sha256": sha256_path(disposition_path),
        "development_terminal_disposition": str(
            config["development_terminal_disposition"]
        ),
        "development_terminal_disposition_sha256": sha256_path(
            terminal_disposition_path
        ),
        "worker_identity": {
            "historical_freeze": str(config["qualified_worker_freeze"]),
            "historical_frozen_worker_sha256": frozen_worker_sha256,
            "current_worker_source": str(config["current_worker_source"]),
            "current_worker_sha256": current_worker_sha256,
            "historical_identity_current": worker_identity_current,
            "active_worker_config": str(config["active_worker_config"]),
            "active_worker_config_sha256": sha256_path(active_worker_config_path),
            "state": (
                "DEVELOPMENT_SUCCESSOR_TERMINAL_FAIL_SELECT_STRONGER_MODEL"
                if config.get("active_worker_state")
                == "TERMINAL_FAIL_SELECT_MATERIALLY_STRONGER_LOCAL_SUCCESSOR"
                else "CURRENT_QUALIFIED_IDENTITY"
                if worker_identity_current
                else "DEVELOPMENT_SUCCESSOR_REQUALIFICATION_REQUIRED"
            ),
        },
        "boundaries": config.get("boundaries"),
        "owners": owners,
        "counts": counts,
        "trigger_state": trigger_state,
        "E2_heldouts_opened": 0,
        "terminal_rules": config.get("terminal_rules"),
        "maximum_inference": config.get("maximum_inference"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    config_path = resolve_repo_path(args.config)
    out_path = resolve_repo_path(args.out)
    report = build_report(config_path, run_tests=args.run_tests)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"trigger_state": report["trigger_state"], **report["counts"]}, indent=2))
    return 0 if report["trigger_state"].startswith("GREEN") else 2


if __name__ == "__main__":
    raise SystemExit(main())
