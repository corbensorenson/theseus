#!/usr/bin/env python3
"""Target-blind repository request adapter for the Theseus causal campaign.

This module does not generate or apply a patch. It converts the four fields
already visible to the qualified local worker into an independently auditable
VCM, planning, routing, authority, and procedural-reuse package. The controller
must honor ``dispatch_allowed`` before invoking the mutating worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_plan_compiler as planner  # noqa: E402
import vcm_consumer_abi as vcm_abi  # noqa: E402
import core_evidence_worker_v2 as worker_contract  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "core_evidence_repository_stack_adapter.json"
VISIBLE_FIELDS = {
    "natural_request",
    "parent_source_commit",
    "allowed_runtime_context",
    "authority_grant",
}
FORBIDDEN_KEYS = {
    "answer",
    "answer_family",
    "category",
    "decoder_fields",
    "evaluator_score",
    "expected",
    "gold_effects",
    "hidden_tests",
    "required_constructs",
    "solution",
    "solution_body",
    "solution_expr",
    "source_task_id",
    "target_commit",
    "target_patch",
    "tests",
}


class AdapterFault(ValueError):
    """A fail-closed adapter fault."""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--variant", default="full_stack")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    packet = adapt_visible_input(
        visible=read_json(Path(args.input)),
        snapshot_root=Path(args.snapshot_root),
        variant_id=args.variant,
        config=read_json(Path(args.config)),
    )
    Path(args.out).write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "dispatch_allowed": packet["dispatch_allowed"],
        "variant_id": packet["variant_id"],
        "typed_faults": packet["typed_faults"],
    }, indent=2))
    return 0 if packet["dispatch_allowed"] else 2


def adapt_visible_input(
    *,
    visible: dict[str, Any],
    snapshot_root: Path,
    variant_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    validate_visible(visible)
    validate_config(config)
    snapshot_root = snapshot_root.resolve()
    if not snapshot_root.is_dir():
        raise AdapterFault("snapshot_root_missing")
    if (snapshot_root / ".git").exists():
        raise AdapterFault("candidate_snapshot_git_metadata_forbidden")
    variant = as_dict(as_dict(config.get("variants")).get(variant_id))
    if not variant:
        raise AdapterFault(f"unknown_variant:{variant_id}")

    authority = authority_receipt(
        str(visible["authority_grant"]),
        as_dict(config["supported_authority_grants"]),
    )
    source_path = select_context_source(snapshot_root, str(visible["natural_request"]))
    request_effect_paths = worker_contract.request_effect_paths(
        worker_contract.text_inventory(snapshot_root),
        str(visible["natural_request"]),
    )
    vcm_packet = build_vcm_packet(
        source_path=source_path,
        snapshot_root=snapshot_root,
        authority=authority,
        intervention=str(variant.get("vcm") or ""),
        config=config,
    )
    compiled_plan = compile_request_plan(
        natural_request=str(visible["natural_request"]),
        config=config,
    )
    reuse = procedural_reuse_receipt(config, str(variant.get("procedural_reuse") or ""))
    facts = semantic_facts(
        visible=visible,
        authority=authority,
        vcm_packet=vcm_packet,
        compiled_plan=compiled_plan,
        reuse=reuse,
        include_plan=variant.get("planning") == "typed",
        context_source_path=str(source_path.relative_to(snapshot_root)),
        request_effect_paths=request_effect_paths,
    )
    route = route_receipt(
        variant=variant,
        authority=authority,
        vcm_packet=vcm_packet,
        compiled_plan=compiled_plan,
        reuse=reuse,
        request_effect_paths=request_effect_paths,
    )
    stack_context = render_stack_context(
        variant=variant,
        facts=facts,
        vcm_packet=vcm_packet,
        compiled_plan=compiled_plan,
        authority=authority,
        route=route,
        reuse=reuse,
    )
    faults = sorted(set(
        list(vcm_packet.get("typed_faults") or [])
        + list(authority.get("typed_faults") or [])
        + list(route.get("typed_faults") or [])
        + ([] if compiled_plan.get("trigger_state") == "GREEN" else ["PLAN_NOT_GREEN"])
        + ([] if reuse.get("ready") or variant.get("procedural_reuse") == "fresh_execution"
           else ["PROCEDURAL_REUSE_NOT_READY"])
    ))
    dispatch_allowed = (
        not faults
        and route.get("selected_route") != "conservative_hold"
        and authority.get("mutation_allowed") is True
    )
    if not authority.get("mutation_allowed"):
        faults = sorted(set(faults + ["MUTATING_WORKER_AUTHORITY_DENIED"]))
    if route.get("selected_route") == "conservative_hold":
        faults = sorted(set(faults + ["CONSERVATIVE_HOLD"]))
    # Never append to the caller-owned list. Campaign runners reuse the task
    # object across matched variants; aliasing here would leak the first arm's
    # context into every later arm and invalidate the comparison.
    worker_context = list(list_value(visible["allowed_runtime_context"]))
    if stack_context is not None:
        worker_context.append(stack_context)
    worker_input = {
        "natural_request": visible["natural_request"],
        "parent_source_commit": visible["parent_source_commit"],
        "allowed_runtime_context": worker_context,
        "authority_grant": visible["authority_grant"],
    }
    packet = {
        "policy": config["policy"],
        "campaign_id": config["campaign_id"],
        "variant_id": variant_id,
        "variant": variant,
        "dispatch_allowed": dispatch_allowed,
        "typed_faults": faults,
        "worker_input": worker_input,
        "audit": {
            "visible_field_names": sorted(visible),
            "forbidden_visible_fields_present": [],
            "natural_request_sha256": sha256_text(str(visible["natural_request"])),
            "parent_source_commit": visible["parent_source_commit"],
            "context_source_path": str(source_path.relative_to(snapshot_root)),
            "semantic_fact_count": len(facts),
            "semantic_fact_set_sha256": stable_hash(
                sorted(facts, key=lambda row: str(row.get("id") or ""))
            ),
            "request_effect_paths": request_effect_paths,
            "candidate_output_consulted": False,
            "target_identity_consulted": False,
            "target_patch_consulted": False,
            "hidden_tests_consulted": False,
            "controller_must_honor_dispatch_allowed": True,
        },
        "authority_receipt": authority,
        "vcm_receipt": vcm_abi.compact_consumer_packet(vcm_packet),
        "compiled_plan": compact_plan(compiled_plan),
        "route_receipt": route,
        "procedural_reuse_receipt": reuse,
        "boundaries": config["boundaries"],
        "counters": {
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "public_calibration_cases_consumed": 0,
            "D2_cases_consumed": 0,
            "E2_heldout_cases_consumed": 0,
            "learned_generation_credit": 0,
            "user_facing_effects": 0,
        },
        "non_claims": [
            "Adapter mechanics are not evidence that a subsystem improves task quality.",
            "Procedural metadata reuse is not learned generation.",
            "A GREEN dispatch packet does not authorize effects outside the disposable snapshot.",
        ],
    }
    assert_no_forbidden_keys(packet)
    return packet


def validate_visible(visible: dict[str, Any]) -> None:
    if set(visible) != VISIBLE_FIELDS:
        raise AdapterFault(f"visible_fields_must_equal:{sorted(VISIBLE_FIELDS)}")
    assert_no_forbidden_keys(visible)
    if not str(visible.get("natural_request") or "").strip():
        raise AdapterFault("natural_request_missing")
    if not str(visible.get("parent_source_commit") or "").strip():
        raise AdapterFault("parent_source_commit_missing")
    if not isinstance(visible.get("allowed_runtime_context"), list):
        raise AdapterFault("allowed_runtime_context_must_be_list")


def validate_config(config: dict[str, Any]) -> None:
    if config.get("policy") != "project_theseus_repository_stack_adapter_v1":
        raise AdapterFault("unexpected_config_policy")
    boundaries = as_dict(config.get("boundaries"))
    required_zero = (
        "external_inference_calls",
        "teacher_calls",
        "public_calibration_cases_consumed",
        "D2_cases_consumed",
        "E2_heldout_cases_consumed",
        "user_facing_effects",
    )
    if boundaries.get("network") != "forbidden":
        raise AdapterFault("network_boundary_mismatch")
    if any(boundaries.get(key) != 0 for key in required_zero):
        raise AdapterFault("no_cheat_boundary_mismatch")
    if boundaries.get("candidate_output_authoritative") is not False:
        raise AdapterFault("candidate_authority_boundary_mismatch")


def authority_receipt(grant: str, grants: dict[str, Any]) -> dict[str, Any]:
    labels = [str(item) for item in list_value(grants.get(grant))]
    supported = bool(labels)
    mutation_allowed = (
        grant == "temporary_effect_with_exact_rollback"
        and {"local_snapshot_write", "exact_rollback"}.issubset(set(labels))
    )
    faults = [] if supported else ["AUTHORITY_GRANT_UNSUPPORTED"]
    return {
        "policy": "project_theseus_repository_authority_receipt_v1",
        "grant": grant,
        "supported": supported,
        "authority_labels": labels,
        "mutation_allowed": mutation_allowed,
        "effect_scope": "disposable_parent_snapshot_only" if mutation_allowed else "none",
        "exact_rollback_required": mutation_allowed,
        "candidate_claims_authoritative": False,
        "typed_faults": faults,
    }


def build_vcm_packet(
    *,
    source_path: Path,
    snapshot_root: Path,
    authority: dict[str, Any],
    intervention: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    inputs = as_dict(config["canonical_inputs"])
    context_refs: list[dict[str, Any]] = [{
        "ref": str(source_path),
        "required": True,
        "exists": True,
    }]
    if intervention == "stale":
        context_refs[0].update({
            "created_utc": "2000-01-01T00:00:00Z",
            "max_age_seconds": 1,
        })
    elif intervention == "omission":
        context_refs = [{
            "ref": str(snapshot_root / "__required_context_omitted__"),
            "required": True,
            "exists": False,
        }]
    authority_labels = list(authority.get("authority_labels") or [])
    if not authority_labels:
        authority_labels = ["denied"]
    return vcm_abi.build_consumer_packet(
        consumer_id="core_evidence.repository_stack_adapter",
        purpose="repository_request",
        read_set=[str(source_path)],
        write_set=[str(snapshot_root / ".theseus_disposable_effect")],
        authority_ceiling=authority_labels,
        materialized_authority_labels=authority_labels,
        permitted_uses=["target_blind_repository_inspection", "local_patch_planning"],
        governor_path=resolve_project(inputs["vcm_governor"]),
        semantic_index_path=resolve_project(inputs["vcm_semantic_index"]),
        context_refs=context_refs,
        denied_taints=[
            "public_benchmark_payload",
            "public_calibration_payload",
            "raw_private_user_text",
            "runtime_external_inference",
            "revoked",
            "deleted",
        ],
        audit_refs=[
            str(inputs["vcm_governor"]),
            str(inputs["vcm_semantic_index"]),
        ],
    )


def compile_request_plan(
    *,
    natural_request: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    inputs = as_dict(config["canonical_inputs"])
    planner_config = read_json(resolve_project(inputs["plan_compiler"]))
    goal = {
        "id": f"repository_request_{sha256_text(natural_request)[:12]}",
        "title": "Complete one target-blind natural repository request",
        "owner_surface": "core_control_plane",
        "priority": "critical",
        "risk_tier": "medium",
        "objective": natural_request,
        "non_goals": [
            "Do not inspect hidden tests or target patches.",
            "Do not write outside the disposable parent snapshot.",
            "Do not treat candidate-authored success flags as evidence.",
        ],
        "outputs": ["candidate_unified_diff", "verification_receipt"],
        "acceptance_tests": [
            "request_derived_behavior_verified",
            "effect_scope_valid",
            "exact_rollback_available",
        ],
        "atoms": request_plan_atoms(),
    }
    return planner.compile_goal(
        goal=goal,
        config=planner_config,
        registry=read_json(resolve_project(inputs["registry"])),
        contexts=read_json(resolve_project(inputs["vcm_contexts"])),
        vcm_governor=read_json(resolve_project(inputs["vcm_governor"])),
        deterministic_tool_registry=read_json(
            resolve_project(inputs["deterministic_tool_registry"])
        ),
        deterministic_tool_report=read_json(
            resolve_project(inputs["deterministic_tool_report"])
        ),
        state=planner.load_state(),
        max_context_pages=3,
    )[0]


def request_plan_atoms() -> list[dict[str, Any]]:
    common = {
        "vcm_family": "planning",
        "fallback_return_allowed": False,
        "training_surface": "none",
        "risk_tier": "medium",
    }
    return [
        {
            **common,
            "id": "inspect",
            "op": "OBSERVE",
            "title": "Inspect request-relevant implementation, analogous context, and tests",
            "depends_on": [],
            "outputs": ["request_grounded_repository_context"],
            "required_capabilities": ["registry_lookup", "report_reading"],
            "allowed_tools": ["registry_lookup", "json_report"],
            "executor_backend": "local_deterministic_tool",
            "worker_tier": "T0",
            "estimated_seconds": 10,
            "acceptance_refs": ["request_derived_behavior_verified"],
        },
        {
            **common,
            "id": "edit",
            "op": "IMPLEMENT",
            "title": "Produce the smallest request-complete repository effect",
            "depends_on": ["inspect"],
            "outputs": ["candidate_unified_diff"],
            "required_capabilities": ["repository_editing"],
            "allowed_tools": [],
            "executor_backend": "theseus_control_plane",
            "worker_tier": "T1",
            "estimated_seconds": 30,
            "acceptance_refs": ["effect_scope_valid"],
        },
        {
            **common,
            "id": "verify",
            "op": "VERIFY",
            "title": "Run request-relevant checks selected from actual changed paths",
            "depends_on": ["edit"],
            "outputs": ["verification_receipt"],
            "required_capabilities": ["trace_validation"],
            "allowed_tools": ["python_script", "json_report"],
            "executor_backend": "autonomy_watchdog",
            "worker_tier": "T0",
            "estimated_seconds": 20,
            "acceptance_refs": ["request_derived_behavior_verified"],
        },
        {
            **common,
            "id": "seal",
            "op": "SEAL",
            "title": "Seal the diff and rollback obligation without judging success",
            "depends_on": ["verify"],
            "outputs": ["sealed_candidate", "rollback_obligation"],
            "required_capabilities": ["trace_validation"],
            "allowed_tools": ["json_report"],
            "executor_backend": "theseus_control_plane",
            "worker_tier": "T0",
            "estimated_seconds": 5,
            "acceptance_refs": ["exact_rollback_available"],
        },
    ]


def procedural_reuse_receipt(config: dict[str, Any], intervention: str) -> dict[str, Any]:
    if intervention == "fresh_execution":
        return {
            "policy": "project_theseus_repository_procedural_reuse_receipt_v1",
            "mode": intervention,
            "ready": True,
            "selected_route_id": "",
            "learned_generation_claim_allowed": False,
            "effect_authority_granted": False,
            "typed_faults": [],
        }
    report = read_json(resolve_project(
        as_dict(config["canonical_inputs"])["procedural_adoption"]
    ))
    routes = [
        row for row in list_value(report.get("default_routes"))
        if isinstance(row, dict)
        and row.get("default_route_adopted") is True
        and as_dict(row.get("continued_regression_guard")).get("armed") is True
    ]
    planning = [
        row for row in routes
        if "planning" in str(row.get("id") or "")
    ]
    selected = planning[0] if planning else {}
    ready = (
        report.get("trigger_state") == "GREEN"
        and bool(selected)
        and selected.get("learned_generation_claim_allowed") is False
    )
    return {
        "policy": "project_theseus_repository_procedural_reuse_receipt_v1",
        "mode": intervention,
        "ready": ready,
        "selected_route_id": selected.get("id", ""),
        "candidate_id": selected.get("candidate_id", ""),
        "guard_id": as_dict(selected.get("continued_regression_guard")).get("guard_id", ""),
        "learned_generation_claim_allowed": False,
        "effect_authority_granted": False,
        "typed_faults": [] if ready else ["PROCEDURAL_REUSE_NOT_READY"],
    }


def semantic_facts(
    *,
    visible: dict[str, Any],
    authority: dict[str, Any],
    vcm_packet: dict[str, Any],
    compiled_plan: dict[str, Any],
    reuse: dict[str, Any],
    include_plan: bool,
    context_source_path: str,
    request_effect_paths: list[str],
) -> list[dict[str, Any]]:
    steps = [
        f"{node.get('atom_id')}:{node.get('title')}"
        for node in list_value(compiled_plan.get("nodes"))
        if isinstance(node, dict)
    ]
    atoms: list[dict[str, Any]] = [
        {"id": "request_sha256", "value": sha256_text(str(visible["natural_request"]))},
        {"id": "parent_source_commit", "value": visible["parent_source_commit"]},
        {"id": "effect_scope", "value": "disposable_parent_snapshot_only"},
        {"id": "request_effect_paths", "value": request_effect_paths},
        {"id": "authority_grant", "value": authority.get("grant")},
        {
            "id": "exact_rollback_required",
            "value": authority.get("exact_rollback_required") is True,
        },
        {"id": "vcm_ready", "value": vcm_packet.get("ready") is True},
        {"id": "vcm_context_source", "value": context_source_path},
        {"id": "candidate_success_claims", "value": "ignored"},
        {
            "id": "verification",
            "value": "independently_selected_from_changed_paths",
        },
        {
            "id": "procedural_route",
            "value": reuse.get("selected_route_id") or "fresh_execution",
        },
        {"id": "procedural_reuse_credit", "value": "none"},
        {
            "id": "workflow_instruction",
            "value": "inspect_then_plan_then_edit_then_verify_then_finish",
        },
    ]
    if include_plan:
        atoms.extend(
            {"id": f"plan_step_{index + 1}", "value": step}
            for index, step in enumerate(steps)
        )
    return atoms


def route_receipt(
    *,
    variant: dict[str, Any],
    authority: dict[str, Any],
    vcm_packet: dict[str, Any],
    compiled_plan: dict[str, Any],
    reuse: dict[str, Any],
    request_effect_paths: list[str],
) -> dict[str, Any]:
    policy = str(variant.get("routing") or "")
    if policy == "conservative_hold":
        selected = "conservative_hold"
    elif not authority.get("mutation_allowed"):
        selected = "conservative_hold"
    elif (
        variant.get("governance") == "full"
        and not request_effect_paths
    ):
        selected = "conservative_hold"
    elif not vcm_packet.get("ready") and variant.get("vcm") != "none":
        selected = "conservative_hold"
    elif compiled_plan.get("trigger_state") != "GREEN" and variant.get("planning") != "none":
        selected = "conservative_hold"
    else:
        selected = "full_governance" if policy == "least_sufficient" else "direct"
    faults = []
    if selected == "conservative_hold" and policy != "conservative_hold":
        faults.append("ROUTE_HELD_BY_FAILED_PRECONDITION")
    if variant.get("governance") == "full" and not request_effect_paths:
        faults.append("REQUEST_EFFECT_SCOPE_EMPTY")
    return {
        "policy": "project_theseus_repository_least_sufficient_route_v1",
        "requested_policy": policy,
        "selected_route": selected,
        "decision_inputs": {
            "mutation_allowed": authority.get("mutation_allowed") is True,
            "vcm_ready": vcm_packet.get("ready") is True,
            "plan_green": compiled_plan.get("trigger_state") == "GREEN",
            "procedural_reuse_ready": reuse.get("ready") is True,
        },
        "routing_grants_effect_authority": False,
        "typed_faults": faults,
    }


def render_stack_context(
    *,
    variant: dict[str, Any],
    facts: list[dict[str, Any]],
    vcm_packet: dict[str, Any],
    compiled_plan: dict[str, Any],
    authority: dict[str, Any],
    route: dict[str, Any],
    reuse: dict[str, Any],
) -> dict[str, Any] | str | None:
    if variant.get("vcm") == "none" and variant.get("planning") == "none":
        return None
    if variant.get("vcm") == "information_matched_untyped":
        return "THESEUS_CONTEXT " + " | ".join(
            f"{row['id']}={json.dumps(row['value'], sort_keys=True)}"
            for row in facts
        )
    if variant.get("vcm") == "information_matched_shuffled":
        return "THESEUS_CONTEXT " + " | ".join(
            f"{row['id']}={json.dumps(row['value'], sort_keys=True)}"
            for row in reversed(facts)
        )
    return {
        "context_type": "theseus_repository_stack_context_v1",
        "atoms": facts,
    }


def compact_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "goal_id": plan.get("goal_id"),
        "trigger_state": plan.get("trigger_state"),
        "contract_hash": plan.get("contract_hash"),
        "acceptance_tests": as_dict(plan.get("contract")).get("acceptance_tests", []),
        "steps": [
            {
                "id": node.get("atom_id"),
                "op": node.get("op"),
                "title": node.get("title"),
                "depends_on": node.get("depends_on"),
                "outputs": node.get("outputs"),
                "acceptance_refs": node.get("acceptance_refs"),
                "context_ready": as_dict(node.get("vcm_context_slice")).get("governed_ready"),
            }
            for node in list_value(plan.get("nodes"))
            if isinstance(node, dict)
        ],
        "hard_failures": as_dict(plan.get("lint")).get("hard_failures", []),
    }


def select_context_source(snapshot_root: Path, natural_request: str) -> Path:
    tokens = {
        token.lower() for token in natural_request.replace("/", " ").split()
        if len(token) >= 3
    }
    candidates = []
    for path in snapshot_root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            relative = path.relative_to(snapshot_root)
        except ValueError:
            continue
        if relative.parts and relative.parts[0] not in {
            "configs", "crates", "docs", "examples", "scripts", "tests"
        }:
            continue
        score = sum(token in str(relative).lower() for token in tokens)
        candidates.append((-score, str(relative), path))
    if not candidates:
        raise AdapterFault("no_request_context_source")
    return sorted(candidates)[0][2]


def assert_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, dict):
        bad = FORBIDDEN_KEYS.intersection(str(key) for key in value)
        if bad:
            raise AdapterFault(f"forbidden_information_field:{sorted(bad)}")
        for item in value.values():
            assert_no_forbidden_keys(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_forbidden_keys(item)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AdapterFault(f"json_object_required:{path}")
    return payload


def resolve_project(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_hash(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":")))


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


if __name__ == "__main__":
    raise SystemExit(main())
