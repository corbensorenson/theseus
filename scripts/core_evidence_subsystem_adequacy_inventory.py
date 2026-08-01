#!/usr/bin/env python3
"""Audit canonical subsystem ownership before integrated D1 experiments."""

from __future__ import annotations

import argparse
import ast
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


def literal_assignment(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(node.value)
    return None


def top_level_symbols(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    symbols: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.Assign):
            symbols.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            symbols.add(node.target.id)
    return symbols


def audit_development_dogfood(config: dict[str, Any]) -> dict[str, Any]:
    dogfood = config.get("development_dogfood")
    if not isinstance(dogfood, dict):
        return {
            "state": "BLOCKED_NOT_DECLARED",
            "ready": False,
            "findings": [
                {
                    "check": "development_dogfood_declared",
                    "passed": False,
                    "detail": None,
                }
            ],
        }

    if dogfood.get("state") == "HISTORICAL_PRE_P1_L0_INVENTORY_SUPERSEDED":
        current = config.get("current_authority")
        current_authority = current if isinstance(current, dict) else {}
        required = {
            "canonical_state",
            "canonical_roadmap",
            "current_instrument",
            "current_progress",
        }
        declared = required.issubset(current_authority)
        return {
            "state": "HISTORICAL_SUPERSEDED_NO_CURRENT_AUTHORITY",
            "program_phase": dogfood.get("state"),
            "ready": False,
            "current_authority": current_authority,
            "findings": [
                {
                    "check": "historical_supersession_explicit",
                    "passed": declared,
                    "detail": sorted(required),
                },
                {
                    "check": "current_execution_authority_absent",
                    "passed": current_authority.get("state")
                    == "NO_CURRENT_EXECUTION_OR_CLAIM_AUTHORITY",
                    "detail": current_authority.get("state"),
                },
            ],
            "boundaries": dogfood.get("boundaries"),
            "latest_experiment_disposition": dogfood.get(
                "latest_experiment_disposition"
            ),
            "latest_experiment_state": dogfood.get("latest_experiment_state"),
            "maximum_inference": (
                "This row establishes only that the pre-P1 L0 inventory is "
                "retained and explicitly superseded. It cannot schedule work, "
                "qualify a model, open D1 or D2, establish subsystem efficacy, "
                "or move an ASI Stack claim."
            ),
        }

    integration_recovery = (
        dogfood.get("state")
        == "P1_LOCAL_INFERENCE_BACKEND_AND_ROUTE_INTEGRITY_REQUIRED"
    )
    model_config_path = resolve_repo_path(str(dogfood["model_config"]))
    preflight_path = resolve_repo_path(str(dogfood["runtime_preflight"]))
    runner_raw = str(
        dogfood.get("canonical_product_runtime")
        or dogfood.get("runner")
        or "__missing_canonical_runtime__"
    )
    adapter_raw = str(
        dogfood.get("historical_context_wrapper_adapter")
        or dogfood.get("adapter")
        or "__missing_historical_adapter__"
    )
    runner_path = resolve_repo_path(runner_raw)
    adapter_path = resolve_repo_path(adapter_raw)
    task_manifest_path = resolve_repo_path(
        str(dogfood.get("task_manifest") or "__p2_task_not_yet_bound__")
    )
    evaluator_manifest_path = resolve_repo_path(
        str(dogfood.get("evaluator_manifest") or "__p2_evaluator_not_yet_bound__")
    )
    alignment_report_path = resolve_repo_path(
        str(dogfood.get("alignment_report") or "__p2_alignment_not_yet_bound__")
    )
    event_owner_path = resolve_repo_path(str(dogfood["dogfood_event_owner"]))
    latest_disposition_path = resolve_repo_path(
        str(dogfood.get("latest_experiment_disposition") or "")
    )
    repair_receipt_path = resolve_repo_path(
        str(
            dogfood.get("active_repair_receipt")
            or dogfood.get("maintenance_repair_receipt")
            or "__no_repair_receipt__"
        )
    )
    model_config = read_json(model_config_path) if model_config_path.is_file() else {}
    preflight = read_json(preflight_path) if preflight_path.is_file() else {}
    task_manifest = (
        read_json(task_manifest_path) if task_manifest_path.is_file() else {}
    )
    evaluator_manifest = (
        read_json(evaluator_manifest_path)
        if evaluator_manifest_path.is_file()
        else {}
    )
    alignment_report = (
        read_json(alignment_report_path)
        if alignment_report_path.is_file()
        else {}
    )
    latest_disposition = (
        read_json(latest_disposition_path)
        if latest_disposition_path.is_file()
        else {}
    )
    repair_receipt = (
        read_json(repair_receipt_path)
        if repair_receipt_path.is_file()
        else {}
    )
    required_model = (
        dogfood.get("required_model")
        if isinstance(dogfood.get("required_model"), dict)
        else {}
    )
    observed_model = (
        model_config.get("model")
        if isinstance(model_config.get("model"), dict)
        else {}
    )
    preflight_model = (
        preflight.get("model_identity")
        if isinstance(preflight.get("model_identity"), dict)
        else {}
    )
    boundaries = (
        dogfood.get("boundaries")
        if isinstance(dogfood.get("boundaries"), dict)
        else {}
    )
    arms = dogfood.get("arms") if isinstance(dogfood.get("arms"), list) else []
    arm_ids = {
        str(row.get("id"))
        for row in arms
        if isinstance(row, dict) and row.get("id")
    }
    required_event_fields = (
        dogfood.get("required_event_fields")
        if isinstance(dogfood.get("required_event_fields"), list)
        else []
    )
    implemented_event_fields = (
        literal_assignment(event_owner_path, "PAIRED_EVENT_FIELDS")
        if event_owner_path.is_file()
        else None
    )
    implemented_event_field_set = (
        {str(item) for item in implemented_event_fields}
        if isinstance(implemented_event_fields, (list, tuple, set))
        else set()
    )
    required_runner_symbols = {
        str(item) for item in dogfood.get("required_runner_symbols", [])
    }
    implemented_runner_symbols = (
        top_level_symbols(runner_path) if runner_path.is_file() else set()
    )
    findings = [
        {
            "check": "development_dogfood_authorized",
            "passed": dogfood.get("state")
            in {
                "AUTHORIZED_LOCAL_DEVELOPMENT_ONLY",
                "REPAIR_SINGLE_OWNER_BEFORE_L0_003",
                "L0_003_PROSPECTIVELY_BOUND_READY_TO_RUN",
                "REPAIR_SINGLE_OWNER_BEFORE_L0_003_REPLAY",
                "L0_003_REPLAY_READY_FIXED_MODEL",
                "L0_003_R2_REPLAY_READY_FIXED_MODEL",
                "P1_LOCAL_INFERENCE_BACKEND_AND_ROUTE_INTEGRITY_REQUIRED",
            },
            "detail": dogfood.get("state"),
        },
        {
            "check": "latest_terminal_disposition_bound",
            "passed": (
                integration_recovery
                or dogfood.get("state") == "AUTHORIZED_LOCAL_DEVELOPMENT_ONLY"
                or (
                    latest_disposition_path.is_file()
                    and latest_disposition.get("terminal") is True
                    and latest_disposition.get("state")
                    == dogfood.get("latest_experiment_state")
                    and latest_disposition.get("experiment_id")
                    == dogfood.get("latest_experiment_id")
                )
            ),
            "detail": {
                "path": dogfood.get("latest_experiment_disposition"),
                "experiment_id": latest_disposition.get("experiment_id"),
                "state": latest_disposition.get("state"),
                "terminal": latest_disposition.get("terminal"),
            },
        },
        {
            "check": "single_owner_repair_declared",
            "passed": (
                (
                    integration_recovery
                    and isinstance(
                        dogfood.get("active_single_owner_hypothesis"), dict
                    )
                    and dogfood["active_single_owner_hypothesis"].get("owner")
                    == "theseus_assistant_runtime"
                    and dogfood["active_single_owner_hypothesis"].get(
                        "hypothesis_id"
                    )
                    == "canonical_local_model_runtime_integration"
                )
                or dogfood.get("state") == "AUTHORIZED_LOCAL_DEVELOPMENT_ONLY"
                or (
                    isinstance(dogfood.get("active_single_owner_hypothesis"), dict)
                    and dogfood["active_single_owner_hypothesis"].get("owner")
                    == "theseus_assistant_runtime"
                    and bool(
                        dogfood["active_single_owner_hypothesis"].get(
                            "hypothesis_id"
                        )
                    )
                    and dogfood["active_single_owner_hypothesis"].get(
                        "hypothesis_id"
                    )
                    == (
                        latest_disposition.get(
                            "selected_next_single_owner_hypothesis"
                        )
                        if isinstance(
                            latest_disposition.get(
                                "selected_next_single_owner_hypothesis"
                            ),
                            dict,
                        )
                        else {}
                    ).get("hypothesis_id")
                )
            ),
            "detail": dogfood.get("active_single_owner_hypothesis"),
        },
        {
            "check": "active_repair_mechanics_green",
            "passed": (
                integration_recovery
                or dogfood.get("state")
                not in {
                    "L0_003_PROSPECTIVELY_BOUND_READY_TO_RUN",
                    "L0_003_REPLAY_READY_FIXED_MODEL",
                    "L0_003_R2_REPLAY_READY_FIXED_MODEL",
                }
                or (
                    repair_receipt_path.is_file()
                    and repair_receipt.get("state")
                    == dogfood.get("required_repair_state")
                    and repair_receipt.get("hypothesis_id")
                    == (
                        dogfood.get("active_single_owner_hypothesis")
                        if isinstance(
                            dogfood.get("active_single_owner_hypothesis"), dict
                        )
                        else {}
                    ).get("hypothesis_id")
                )
            ),
            "detail": {
                "path": dogfood.get("active_repair_receipt"),
                "state": repair_receipt.get("state"),
                "hypothesis_id": repair_receipt.get("hypothesis_id"),
            },
        },
        {
            "check": "model_config_exists",
            "passed": model_config_path.is_file(),
            "detail": str(dogfood["model_config"]),
        },
        {
            "check": "runtime_preflight_green",
            "passed": (
                preflight.get("trigger_state")
                == dogfood.get("required_preflight_trigger_state")
            ),
            "detail": preflight.get("trigger_state"),
        },
        {
            "check": "model_config_identity_matches",
            "passed": all(
                observed_model.get(key) == value
                for key, value in required_model.items()
            ),
            "detail": {
                key: observed_model.get(key)
                for key in required_model
            },
        },
        {
            "check": "preflight_identity_matches",
            "passed": all(
                preflight_model.get(key) == value
                for key, value in required_model.items()
            ),
            "detail": {
                key: preflight_model.get(key)
                for key in required_model
            },
        },
        {
            "check": "existing_runner_and_owners_present",
            "passed": all(
                path.is_file()
                for path in (runner_path, adapter_path, event_owner_path)
            ),
            "detail": {
                "runner": runner_raw,
                "adapter": adapter_raw,
                "dogfood_event_owner": str(dogfood["dogfood_event_owner"]),
            },
        },
        {
            "check": (
                "canonical_local_model_route_integrity_implemented"
                if integration_recovery
                else "paired_runner_contract_implemented"
            ),
            "passed": (
                not integration_recovery
                and
                bool(required_runner_symbols)
                and required_runner_symbols.issubset(implemented_runner_symbols)
            ),
            "detail": {
                "current_generation_backend_state": dogfood.get(
                    "current_generation_backend_state"
                ),
                "required_generation_backend_state": dogfood.get(
                    "required_generation_backend_state"
                ),
                "required_route_integrity_properties": dogfood.get(
                    "required_route_integrity_properties", []
                ),
                "required_symbols": sorted(required_runner_symbols),
                "implemented_symbols": sorted(
                    required_runner_symbols.intersection(
                        implemented_runner_symbols
                    )
                ),
            },
        },
        {
            "check": "prospective_L0_task_and_evaluator_bound",
            "passed": (
                integration_recovery
                or (
                task_manifest.get("policy")
                == "project_theseus_l0_real_work_task_manifest_v1"
                and task_manifest.get("state")
                == "PROSPECTIVELY_BOUND_BEFORE_CANDIDATE_GENERATION"
                and evaluator_manifest.get("policy")
                == "project_theseus_local_8b_functional_evaluator_v1"
                and evaluator_manifest.get("state")
                == "PROSPECTIVELY_BOUND_BEFORE_CANDIDATE_GENERATION"
                and task_manifest.get("experiment_id")
                == evaluator_manifest.get("experiment_id")
                and isinstance(task_manifest.get("arm_order"), list)
                and len(task_manifest["arm_order"]) == 2
                and set(map(str, task_manifest["arm_order"]))
                == {"full_theseus", "direct_fixed_worker"}
                )
            ),
            "detail": {
                "deferred_until": "P2" if integration_recovery else None,
                "task_manifest": dogfood.get("task_manifest"),
                "evaluator_manifest": dogfood.get("evaluator_manifest"),
                "experiment_id": task_manifest.get("experiment_id"),
                "arm_order": task_manifest.get("arm_order"),
            },
        },
        {
            "check": "prospective_alignment_green_before_generation",
            "passed": (
                integration_recovery
                or (
                alignment_report.get("trigger_state")
                == dogfood.get("required_alignment_trigger_state")
                and alignment_report.get("evaluator_manifest_sha256")
                == (
                    sha256_path(evaluator_manifest_path)
                    if evaluator_manifest_path.is_file()
                    else None
                )
                and (
                    alignment_report.get("summary")
                    if isinstance(alignment_report.get("summary"), dict)
                    else {}
                ).get("aligned_task_count")
                == 1
                )
            ),
            "detail": {
                "deferred_until": "P2" if integration_recovery else None,
                "alignment_report": dogfood.get("alignment_report"),
                "trigger_state": alignment_report.get("trigger_state"),
                "aligned_task_count": (
                    alignment_report.get("summary")
                    if isinstance(alignment_report.get("summary"), dict)
                    else {}
                ).get("aligned_task_count"),
            },
        },
        {
            "check": "paired_baseline_candidate_and_single_ablation_declared",
            "passed": (
                {
                    "direct_fixed_model_runtime",
                    "integrated_theseus_runtime",
                    "single_owner_ablation",
                }.issubset(arm_ids)
                if integration_recovery
                else {
                    "direct_fixed_worker",
                    "full_theseus",
                    "single_owner_ablation",
                }.issubset(arm_ids)
            ),
            "detail": sorted(arm_ids),
        },
        {
            "check": "paired_event_denominator_implemented",
            "passed": (
                bool(required_event_fields)
                and set(map(str, required_event_fields)).issubset(
                    implemented_event_field_set
                )
            ),
            "detail": {
                "policy": dogfood.get("dogfood_event_policy"),
                "required_fields": required_event_fields,
                "implemented_fields": sorted(implemented_event_field_set),
            },
        },
        {
            "check": "development_rows_excluded_from_fresh_qualification",
            "passed": (
                isinstance(dogfood.get("task_policy"), dict)
                and dogfood["task_policy"].get(
                    "development_rows_eligible_for_fresh_qualification"
                )
                is False
            ),
            "detail": dogfood.get("task_policy"),
        },
        {
            "check": "measurement_boundaries_preserved",
            "passed": (
                boundaries.get("external_inference") == "forbidden"
                and boundaries.get("teacher_calls") == "forbidden"
                and boundaries.get("public_benchmark_consumption") == "forbidden"
                and boundaries.get("D2_consumption") == "forbidden"
                and boundaries.get("E2_heldout_consumption") == "forbidden"
                and boundaries.get("automatic_user_facing_effects") == 0
                and boundaries.get("learned_theseus_student_credit") == 0
                and boundaries.get("ASI_stack_support_state_effect") == "none"
            ),
            "detail": boundaries,
        },
    ]
    ready = all(bool(row["passed"]) for row in findings)
    return {
        "state": (
            "BLOCKED_ON_P1_ROUTE_INTEGRITY"
            if integration_recovery
            else "READY_FOR_L0_003_R2_REPLAY"
            if ready
            and dogfood.get("state")
            == "L0_003_R2_REPLAY_READY_FIXED_MODEL"
            else "READY_FOR_L0_003_REPLAY"
            if ready
            and dogfood.get("state") == "L0_003_REPLAY_READY_FIXED_MODEL"
            else "READY_FOR_L0_003_FIXED_MODEL_PAIR"
            if ready
            and dogfood.get("state")
            == "L0_003_PROSPECTIVELY_BOUND_READY_TO_RUN"
            else "READY_FOR_SINGLE_OWNER_PROTOCOL_REPAIR"
            if ready
            and dogfood.get("state")
            in {
                "REPAIR_SINGLE_OWNER_BEFORE_L0_003",
                "REPAIR_SINGLE_OWNER_BEFORE_L0_003_REPLAY",
            }
            else "READY_FIXED_LOCAL_MODEL_PAIRED_DOGFOOD"
            if ready
            else "BLOCKED_DEVELOPMENT_DOGFOOD_PREFLIGHT"
        ),
        "program_phase": dogfood.get("state"),
        "ready": ready,
        "model_config": str(dogfood["model_config"]),
        "model_config_sha256": (
            sha256_path(model_config_path) if model_config_path.is_file() else None
        ),
        "runtime_preflight": str(dogfood["runtime_preflight"]),
        "runtime_preflight_sha256": (
            sha256_path(preflight_path) if preflight_path.is_file() else None
        ),
        "runner": runner_raw,
        "adapter": adapter_raw,
        "arms": arms,
        "task_policy": dogfood.get("task_policy"),
        "metrics": dogfood.get("metrics"),
        "promotion_policy": dogfood.get("promotion_policy"),
        "latest_experiment_disposition": dogfood.get(
            "latest_experiment_disposition"
        ),
        "latest_experiment_state": dogfood.get("latest_experiment_state"),
        "active_repair_receipt": (
            dogfood.get("active_repair_receipt")
            or dogfood.get("maintenance_repair_receipt")
        ),
        "active_repair_state": repair_receipt.get("state"),
        "active_single_owner_hypothesis": dogfood.get(
            "active_single_owner_hypothesis"
        ),
        "boundaries": boundaries,
        "findings": findings,
        "maximum_inference": (
            "This readiness result authorizes the declared reusable local "
            "development phase only. "
            "It does not qualify the worker, open E2 or D2, establish subsystem "
            "efficacy, create Theseus-student credit, or move an ASI Stack claim."
        ),
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
    development_dogfood = audit_development_dogfood(config)
    historical_superseded = (
        config.get("active_worker_state")
        == "HISTORICAL_SUPERSEDED_BY_SOURCE_BOUND_P1_THROUGH_P4S"
    )
    trigger_state = (
        "HISTORICAL_SUPERSEDED_NO_CURRENT_AUTHORITY"
        if historical_superseded
        else "RED_D1_QUALIFICATION_SEALED"
        if config.get("active_worker_state")
        == "TERMINAL_FAIL_D1_QUALIFICATION_REQUIRES_FRESH_COMPETENT_FREEZE_AFTER_L0_SELECTION"
        else "RED_D1_WORKER_REQUALIFICATION_REQUIRED"
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
                "HISTORICAL_SUPERSEDED_NO_CURRENT_AUTHORITY"
                if historical_superseded
                else "D1_QUALIFICATION_SEALED_PENDING_POST_L0_COMPETENT_FREEZE"
                if config.get("active_worker_state")
                == "TERMINAL_FAIL_D1_QUALIFICATION_REQUIRES_FRESH_COMPETENT_FREEZE_AFTER_L0_SELECTION"
                else "CURRENT_QUALIFIED_IDENTITY"
                if worker_identity_current
                else "DEVELOPMENT_SUCCESSOR_REQUALIFICATION_REQUIRED"
            ),
        },
        "development_dogfood": development_dogfood,
        "current_authority": config.get("current_authority"),
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
    return 0 if report["trigger_state"].startswith(
        ("GREEN", "HISTORICAL_SUPERSEDED")
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
