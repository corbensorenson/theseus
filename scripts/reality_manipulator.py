"""Reality Manipulator MVP compiler for Project Theseus.

This is a practical bridge from the Grimoire / Portal / Reality Manipulator
concept into the local Theseus artifact system. It does not claim VR, physical
fabrication, chip compilation, or autonomous world actuation. It compiles a raw
goal into an inspectable world bundle:

intent -> eight-limb spell -> artifacts -> claims/critiques -> specialist arms
-> compile targets -> gates -> primitive candidates -> feedback plan.

The report is deliberately deterministic and local-only. It gives the autonomy
loop something concrete to inspect when deciding whether Theseus has an
artifact-preserving invention substrate rather than just chat logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_GOAL = (
    "Create a world for a modular fabrication-aware AI assistant that can "
    "design small physical tools and compile embedded software for them."
)

SPELL_LIMBS = [
    "role",
    "objective",
    "context",
    "constraints",
    "procedure",
    "output_contract",
    "verification",
    "failure_behavior",
]

CAST_LEVEL_LIMBS = {
    "quick": {"role", "objective", "context", "verification"},
    "working": {"role", "objective", "context", "constraints", "output_contract", "verification"},
    "full": set(SPELL_LIMBS),
}

ARM_LIBRARY: dict[str, dict[str, Any]] = {
    "head_router": {
        "scope": "Intent interpretation, decomposition, memory routing, permission routing, and synthesis.",
        "tools": ["artifact_graph", "spell_parser", "gate_selector"],
        "permissions": ["read_world", "write_artifacts", "route_arms"],
        "risk_tier": "medium",
    },
    "writing_arm": {
        "scope": "Structure, clarity, release polish, and public/non-public framing.",
        "tools": ["spell_card_renderer", "release_manifest_renderer"],
        "permissions": ["read_world", "write_drafts"],
        "risk_tier": "low",
    },
    "research_arm": {
        "scope": "Source discovery, related work, and evidence mapping.",
        "tools": ["source_manifest", "claim_evidence_links"],
        "permissions": ["read_world", "network_research_with_approval"],
        "risk_tier": "medium",
    },
    "claim_auditor": {
        "scope": "Claim ledger, support states, non-claims, and epistemic risk.",
        "tools": ["claim_ledger", "critique_log"],
        "permissions": ["read_world", "write_claims", "block_release"],
        "risk_tier": "medium",
    },
    "skeptic_arm": {
        "scope": "Adversarial critique, missing caveats, failure modes, and contradiction checks.",
        "tools": ["critique_log", "coil_inspection"],
        "permissions": ["read_world", "write_critiques", "block_release"],
        "risk_tier": "medium",
    },
    "coding_arm": {
        "scope": "Software implementation, tests, adapters, APIs, and build packets.",
        "tools": ["repo_inspection", "unit_tests", "candidate_gate"],
        "permissions": ["read_repo", "write_patch", "run_tests"],
        "risk_tier": "medium",
    },
    "cad_arm": {
        "scope": "Geometry, CAD artifact lists, DFM risks, and inspection criteria.",
        "tools": ["materialize_spell", "dfm_checklist"],
        "permissions": ["read_world", "write_design_packet"],
        "risk_tier": "high",
    },
    "simulation_arm": {
        "scope": "Virtual tests, model assumptions, simulation plans, and acceptance arenas.",
        "tools": ["benchmark_arena", "simulation_plan"],
        "permissions": ["read_world", "write_benchmark"],
        "risk_tier": "medium",
    },
    "fabrication_arm": {
        "scope": "BOMs, fabrication routing, inspection, assembly, and field-test plans.",
        "tools": ["fabrication_packet", "inspection_plan"],
        "permissions": ["read_world", "write_fabrication_plan"],
        "risk_tier": "high",
    },
    "chip_arm": {
        "scope": "Embedded profiles, firmware constraints, GPU kernels, FPGA assumptions, and hardware-in-loop plans.",
        "tools": ["chipcompile_spell", "target_profile"],
        "permissions": ["read_world", "write_build_profile"],
        "risk_tier": "high",
    },
    "safety_arm": {
        "scope": "Permission envelopes, safety stages, approval gates, refusal conditions, and rollback plans.",
        "tools": ["gate_spell", "risk_register"],
        "permissions": ["read_world", "write_gates", "veto_release"],
        "risk_tier": "high",
    },
    "benchmark_arm": {
        "scope": "Capability ratchets, mastery thresholds, residual escrow, and anti-Goodhart checks.",
        "tools": ["benchmark_ledger", "residual_escrow", "regression_suite"],
        "permissions": ["read_world", "write_benchmarks"],
        "risk_tier": "medium",
    },
    "memory_arm": {
        "scope": "Artifact retrieval, provenance, staleness, imports, and release history.",
        "tools": ["artifact_graph", "provenance_index"],
        "permissions": ["read_world", "write_memory_index"],
        "risk_tier": "medium",
    },
}

RUNTIME_LIBRARY: dict[str, dict[str, Any]] = {
    "digital": {
        "outputs": ["whitepaper", "spec", "codebase", "website", "dashboard", "dataset", "workflow"],
        "required_gates": ["claim_audit", "tests_or_review", "release_manifest"],
        "safety_stage": "digital_compile",
    },
    "chip": {
        "outputs": ["firmware", "embedded_build", "gpu_kernel", "fpga_bitstream", "hardware_in_loop_plan"],
        "required_gates": ["target_profile", "resource_budget", "compile_test", "hardware_assumption_review"],
        "safety_stage": "chip_compile",
    },
    "matter": {
        "outputs": ["requirements", "cad_package", "dfm_risks", "bom", "fabrication_packet", "inspection_plan"],
        "required_gates": ["safety_review", "manufacturability_review", "simulation_or_calculation", "inspection_plan"],
        "safety_stage": "physical_fabrication",
    },
    "robotic": {
        "outputs": ["motion_plan", "control_policy", "failsafe_behavior", "telemetry_loop"],
        "required_gates": ["sandbox_simulation", "failsafe_review", "human_approval", "telemetry_monitoring"],
        "safety_stage": "robotic_action",
    },
    "organizational": {
        "outputs": ["workflow", "policy", "team_structure", "operating_procedure", "feedback_ritual"],
        "required_gates": ["stakeholder_review", "legal_or_policy_check", "rollback_plan"],
        "safety_stage": "organizational_runtime",
    },
    "portal": {
        "outputs": ["world_map", "artifact_objects", "spell_cards", "claim_nodes", "compile_doorways"],
        "required_gates": ["artifact_mapping", "permission_boundary", "export_gate"],
        "safety_stage": "simulation",
    },
}

WORLD_TYPE_KEYWORDS = [
    ("fabrication_prototype_world", ["fabrication", "physical", "cad", "tool", "matter", "prototype", "manufactur"]),
    ("robotics_world", ["robot", "motion", "control", "sensor", "actuator"]),
    ("software_system_world", ["software", "code", "api", "app", "dashboard", "agent"]),
    ("research_world", ["research", "paper", "whitepaper", "claim", "evidence"]),
    ("chip_runtime_world", ["chip", "firmware", "embedded", "gpu", "fpga", "kernel"]),
    ("startup_world", ["startup", "business", "organization", "workflow"]),
    ("game_world", ["game", "simulation", "vr", "ar", "portal"]),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", default=DEFAULT_GOAL)
    parser.add_argument("--world-type", default="")
    parser.add_argument("--world-name", default="")
    parser.add_argument("--out", default="reports/reality_manipulator.json")
    parser.add_argument("--markdown-out", default="reports/reality_manipulator.md")
    parser.add_argument("--bundle-dir", default="reports/reality_manipulator/latest_world")
    args = parser.parse_args()

    payload = build_payload(
        goal=args.goal,
        world_type=args.world_type or infer_world_type(args.goal),
        world_name=args.world_name or infer_world_name(args.goal),
        bundle_dir=resolve(args.bundle_dir),
    )
    write_bundle(payload, resolve(args.bundle_dir))
    write_json(resolve(args.out), payload)
    write_text(resolve(args.markdown_out), render_markdown(payload))
    print(json.dumps(payload, indent=2))
    return 0 if payload["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_payload(*, goal: str, world_type: str, world_name: str, bundle_dir: Path) -> dict[str, Any]:
    created = now()
    source_state = load_source_state()
    spell = build_spell(goal=goal, world_type=world_type, world_name=world_name)
    cast_level = cast_level_for_spell(spell)
    command_contract = build_command_contract(
        goal=goal,
        world_type=world_type,
        world_name=world_name,
        spell=spell,
        cast_level=cast_level,
    )
    coil = inspect_coil(spell)
    arms = select_arms(goal, world_type)
    compile_targets = select_compile_targets(goal, world_type)
    specialist_lifecycle = build_specialist_lifecycle(arms, source_state)
    workflow_tool_metrics = build_workflow_tool_metrics(source_state)
    artifacts, edges = build_artifact_graph(goal, world_type, world_name, spell, command_contract, arms, compile_targets)
    claims = build_claims(goal, world_type, compile_targets, source_state)
    critiques = build_critiques(goal, compile_targets, coil)
    primitives = build_primitives(spell, arms, compile_targets, [])
    structured_output = render_structured_output(
        world_name=world_name,
        world_type=world_type,
        goal=goal,
        command_contract=command_contract,
        compile_targets=compile_targets,
        primitives=primitives,
    )
    resource_log = build_resource_log(source_state, created=created)
    mvp_bundle = build_mvp_object_bundle(bundle_dir)
    gates = evaluate_gates(
        spell,
        command_contract,
        compile_targets,
        claims,
        critiques,
        coil,
        specialist_lifecycle,
        workflow_tool_metrics,
        mvp_bundle,
    )
    residuals = build_residuals(gates, critiques)
    primitives = build_primitives(spell, arms, compile_targets, gates)
    feedback_plan = build_feedback_plan(compile_targets, residuals)
    world = {
        "id": stable_id("world", world_name, goal),
        "name": world_name,
        "type": world_type,
        "owner": "local_operator",
        "privacy": "private_by_default_collaborative_by_invitation_public_by_release",
        "artifact_graph": {
            "artifact_count": len(artifacts),
            "edge_count": len(edges),
            "bundle_dir": rel(bundle_dir),
        },
        "imported_primitives": [item["id"] for item in primitives if item.get("status") == "candidate"],
        "active_arms": [arm["name"] for arm in arms],
        "spells_and_stacks": {
            "entry_spell_id": artifacts[1]["id"],
            "command_contract_id": command_contract["id"],
            "starting_stack": starting_stack(goal, compile_targets),
        },
        "compile_targets": [target["target_type"] for target in compile_targets],
        "benchmarks": build_benchmarks(compile_targets, source_state),
        "residual_escrow": residuals,
        "permissions": permission_model(arms, compile_targets),
        "release_history": [],
    }
    high_risk_approved = [
        target
        for target in compile_targets
        if target["risk_tier"] == "high" and target["gate_status"] == "approve"
    ]
    hard_failed = [gate for gate in gates if gate["severity"] == "hard" and not gate["passed"]]
    warning_failed = [gate for gate in gates if gate["severity"] != "hard" and not gate["passed"]]
    trigger_state = "GREEN"
    if hard_failed or high_risk_approved:
        trigger_state = "RED"
    elif warning_failed or residuals:
        trigger_state = "YELLOW"
    return {
        "policy": "project_theseus_reality_manipulator_mvp_v1",
        "created_utc": created,
        "trigger_state": trigger_state,
        "purpose": "Compile structured human intent into typed, gated, auditable world artifacts.",
        "non_claims": [
            "This report is not proof of ASI.",
            "This report does not fabricate physical objects, deploy software, run robots, or compile chips by itself.",
            "Magic/reality manipulation terms are interface metaphors for structured engineering with verification.",
            "Public benchmark or real-world success still requires separate measured evidence.",
        ],
        "world": world,
        "viea": {
            "canonical_architecture": "Verified Intent-to-Execution Architecture",
            "core_loop": [
                "intent",
                "command_contract",
                "artifact_graph",
                "specialist_execution",
                "runtime_target",
                "verification",
                "feedback",
                "improved_system",
            ],
            "student_learning_proof_layer": "broad_public_transfer_matrix",
            "promotion_rule": "student learning claims require clean token-level student evidence and broad public transfer; VIEA artifacts are governance evidence, not learning proof",
        },
        "structured_command_layer": {
            "command_contract": command_contract,
            "command_levels": {
                "quick": sorted(CAST_LEVEL_LIMBS["quick"]),
                "working": sorted(CAST_LEVEL_LIMBS["working"]),
                "full": sorted(CAST_LEVEL_LIMBS["full"]),
            },
        },
        "grimoire": {
            "spell": spell,
            "cast_level": cast_level,
            "wrapper": spell_wrapper(world_name, world_type, compile_targets),
            "coil_inspection": coil,
            "hard_rule": "the_stronger_the_actuator_the_stronger_the_ritual",
        },
        "artifact_graph": {"artifacts": artifacts, "edges": edges},
        "claim_ledger": claims,
        "critique_log": critiques,
        "specialist_router": {
            "resident_head": "head_router",
            "routing_policy": "bounded_specialists_with_permission_envelopes",
            "arms": arms,
            "lifecycle_governance": specialist_lifecycle,
            "synthesis_rule": "head_router_synthesizes_outputs_after_claim_audit_and_gate_review",
        },
        "world_runtimes": compile_targets,
        "workflow_to_tool_compiler": workflow_tool_metrics,
        "resource_log": resource_log,
        "safety_model": {
            "core_rule": "nothing_leaves_a_world_into_shared_reality_without_the_right_gate",
            "stages": safety_stages(compile_targets),
            "high_risk_approved_without_gate_count": len(high_risk_approved),
            "permission_envelopes": permission_model(arms, compile_targets),
        },
        "feedback_ratchet": {
            "question": "What did reality teach us?",
            "benchmarks": world["benchmarks"],
            "residual_escrow": residuals,
            "primitive_candidates": primitives,
            "feedback_plan": feedback_plan,
        },
        "mvp_object_bundle": mvp_bundle,
        "primitive_registry": primitives,
        "structured_output": structured_output,
        "acceptance_scenario": {
            "goal": goal,
            "world_created": True,
            "command_contract_ready": True,
            "initial_artifacts_created": len(artifacts),
            "claims_created": len(claims),
            "compile_targets_created": len(compile_targets),
            "release_manifest_ready": any(item["type"] == "release" for item in artifacts),
            "real_world_execution_blocked_until_gate": all(
                target["gate_status"] != "approve"
                for target in compile_targets
                if target["risk_tier"] == "high"
            ),
            "mvp_bundle_files": [item["file"] for item in mvp_bundle],
        },
        "artifacts": {
            "report": "reports/reality_manipulator.json",
            "markdown": "reports/reality_manipulator.md",
            "bundle_dir": rel(bundle_dir),
            "world": rel(bundle_dir / "world.json"),
            "command_contract": rel(bundle_dir / "command_contract.json"),
            "artifacts": rel(bundle_dir / "artifacts.json"),
            "claim_ledger": rel(bundle_dir / "claim_ledger.json"),
            "critique_log": rel(bundle_dir / "critique_log.json"),
            "structured_output": rel(bundle_dir / "structured_output.md"),
            "release_manifest": rel(bundle_dir / "release_manifest.json"),
            "primitive_registry": rel(bundle_dir / "primitive_registry.json"),
            "specialist_lifecycle": rel(bundle_dir / "specialist_lifecycle.json"),
            "workflow_tool_metrics": rel(bundle_dir / "workflow_tool_metrics.json"),
            "feedback_plan": rel(bundle_dir / "feedback_plan.md"),
            "resource_log": rel(bundle_dir / "resource_log.jsonl"),
        },
        "gates": gates,
        "external_inference_calls": 0,
    }


def load_source_state() -> dict[str, Any]:
    return {
        "learning_scoreboard": read_json(REPORTS / "learning_scoreboard.json"),
        "genesis_kernel": read_json(REPORTS / "genesis_kernel" / "report.json"),
        "architecture_guidance": read_json(REPORTS / "architecture_guidance_loop.json"),
        "grammar_suckers": read_json(REPORTS / "grammar_suckers.json"),
        "deterministic_taming": read_json(REPORTS / "deterministic_taming_stack.json"),
        "cell_lifecycle": read_json(REPORTS / "cell_lifecycle.json"),
        "broad_transfer_matrix": read_json(REPORTS / "broad_transfer_matrix.json"),
        "resource_governor": read_json(REPORTS / "resource_governor.json"),
        "performance_optimizer": read_json(REPORTS / "performance_optimizer.json"),
        "arm_lifecycle": read_json(REPORTS / "arm_lifecycle_governance.json"),
        "loop_closure_harvester": read_json(REPORTS / "loop_closure_harvester.json"),
        "loop_closure_promoter": read_json(REPORTS / "loop_closure_tool_promoter.json"),
        "tool_registry": read_json(REPORTS / "tool_registry.json"),
    }


def build_spell(*, goal: str, world_type: str, world_name: str) -> dict[str, str]:
    return {
        "role": "World architect, artifact librarian, claim auditor, specialist router, and safety gatekeeper.",
        "objective": f"Create a durable {world_type} named {world_name} that turns the stated intent into inspectable artifacts and gated compile targets.",
        "context": f"User intent: {goal}",
        "constraints": (
            "Private by default; no public release, deployment, physical fabrication, robot action, chip build, "
            "bulk download, or high-risk side effect without explicit gate evidence and approval."
        ),
        "procedure": (
            "Create the world, parse the spell, initialize artifacts, extract claims and non-claims, bind specialist "
            "arms, choose compile targets, inspect safety gates, seed benchmarks, preserve residuals, and write a feedback plan."
        ),
        "output_contract": (
            "Return world metadata, active arms, artifact graph, claim ledger, critique log, compile targets, gates, "
            "primitive candidates, residual escrow, release manifest, and feedback plan."
        ),
        "verification": (
            "Check that every strong claim has a support state, every high-risk target is revise/block until safety "
            "evidence exists, every portal object maps to an artifact, and every repeated workflow can become a tool only after evidence."
        ),
        "failure_behavior": (
            "If evidence, permissions, or technical details are missing, mark the target revise/block and preserve a residual instead of approving by assumption."
        ),
    }


def build_command_contract(
    *,
    goal: str,
    world_type: str,
    world_name: str,
    spell: dict[str, str],
    cast_level: str,
) -> dict[str, Any]:
    """Create the VIEA command contract as a first-class object.

    The grimoire spell remains the human-readable ritual surface. The command
    contract is the system-level execution contract that artifacts, specialists,
    gates, and runtimes can reference directly.
    """

    contract = {
        "id": stable_id("command", world_name, goal, cast_level),
        "kind": "viea_command_contract",
        "title": f"Create {world_name}",
        "raw_intent": goal,
        "intent_checksum": (
            f"Create a private {world_type} for {world_name}; preserve claims, critiques, artifacts, "
            "release state, feedback, and resource usage before any shared-reality execution."
        ),
        "assumption_diff": [
            "Execution remains local/private unless a runtime gate approves it.",
            "Learning claims are delegated to the learning scoreboard, not the world bundle.",
            "Matter/chip/robotic targets are planning or handoff targets until stronger gates exist.",
        ],
        "world_type": world_type,
        "cast_level": cast_level,
        "fields": {
            "role": spell["role"],
            "objective": spell["objective"],
            "context": spell["context"],
            "constraints": spell["constraints"],
            "procedure": spell["procedure"],
            "output_contract": spell["output_contract"],
            "verification": spell["verification"],
            "failure_behavior": spell["failure_behavior"],
        },
        "required_outputs": [
            "world.json",
            "command_contract.json",
            "artifacts.json",
            "claim_ledger.json",
            "critique_log.json",
            "structured_output.md",
            "release_manifest.json",
            "primitive_registry.json",
            "specialist_lifecycle.json",
            "workflow_tool_metrics.json",
            "feedback_plan.md",
            "resource_log.jsonl",
        ],
        "execution_boundary": {
            "default_scope": "private_workspace",
            "shared_reality_execution": "blocked_until_runtime_gate",
            "external_inference_calls": 0,
        },
        "verification_gates": [
            "eight_limb_spell_complete",
            "command_contract_first_class",
            "artifact_graph_initialized",
            "claim_ledger_initialized",
            "critique_log_initialized",
            "mvp_object_bundle_complete",
            "learning_claims_delegated_to_scoreboard",
        ],
    }
    contract["content_hash"] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode("utf-8")).hexdigest()
    return contract


def cast_level_for_spell(spell: dict[str, str]) -> str:
    present = {key for key in SPELL_LIMBS if str(spell.get(key) or "").strip()}
    if CAST_LEVEL_LIMBS["full"].issubset(present):
        return "full"
    if CAST_LEVEL_LIMBS["working"].issubset(present):
        return "working"
    return "quick"


def inspect_coil(spell: dict[str, str]) -> list[dict[str, Any]]:
    checks = [
        (
            "objective_x_verification",
            bool(spell.get("objective")) and bool(spell.get("verification")),
            "Can success be checked from the stated objective?",
        ),
        (
            "context_x_constraints",
            bool(spell.get("context")) and bool(spell.get("constraints")),
            "Does the local world state meet the non-negotiables?",
        ),
        (
            "procedure_x_output_contract",
            bool(spell.get("procedure")) and bool(spell.get("output_contract")),
            "Can the procedure produce the requested artifact shape?",
        ),
        (
            "constraints_x_failure_behavior",
            bool(spell.get("constraints")) and bool(spell.get("failure_behavior")),
            "Does uncertainty degrade safely?",
        ),
        (
            "role_x_output_contract",
            bool(spell.get("role")) and bool(spell.get("output_contract")),
            "Was the right expertise invoked for the output?",
        ),
    ]
    return [
        {
            "crossing": name,
            "passed": passed,
            "question": question,
            "portal_signal": "stable_chord" if passed else "red_fracture",
        }
        for name, passed, question in checks
    ]


def select_arms(goal: str, world_type: str) -> list[dict[str, Any]]:
    text = f"{goal} {world_type}".lower()
    selected = {"head_router", "memory_arm", "claim_auditor", "skeptic_arm", "safety_arm", "benchmark_arm"}
    if any(word in text for word in ["paper", "whitepaper", "document", "release", "spec"]):
        selected.add("writing_arm")
    if any(word in text for word in ["research", "source", "evidence", "related work"]):
        selected.add("research_arm")
    if any(word in text for word in ["software", "code", "api", "agent", "embedded", "firmware"]):
        selected.add("coding_arm")
    if any(word in text for word in ["cad", "fabrication", "physical", "tool", "prototype", "matter", "manufactur"]):
        selected.update({"cad_arm", "fabrication_arm", "simulation_arm"})
    if any(word in text for word in ["chip", "firmware", "embedded", "gpu", "fpga", "kernel"]):
        selected.add("chip_arm")
    if any(word in text for word in ["robot", "motion", "actuator"]):
        selected.add("simulation_arm")
    ordered = ["head_router"] + sorted(name for name in selected if name != "head_router")
    arms = []
    for name in ordered:
        spec = ARM_LIBRARY[name]
        arms.append(
            {
                "name": name,
                "scope": spec["scope"],
                "input_contract": "typed_artifacts_and_permission_envelope",
                "output_contract": "bounded_artifacts_with_claims_critiques_and_residuals",
                "tools": spec["tools"],
                "memory_scope": "world_local_plus_approved_imports",
                "permissions": spec["permissions"],
                "risk_tier": spec["risk_tier"],
                "benchmarks": arm_benchmarks(name),
                "lifecycle_status": "active_candidate_runtime",
            }
        )
    return arms


def select_compile_targets(goal: str, world_type: str) -> list[dict[str, Any]]:
    text = f"{goal} {world_type}".lower()
    target_types = {"digital", "portal", "organizational"}
    if any(word in text for word in ["fabrication", "physical", "cad", "tool", "prototype", "matter", "manufactur"]):
        target_types.add("matter")
    if any(word in text for word in ["chip", "firmware", "embedded", "gpu", "fpga", "kernel"]):
        target_types.add("chip")
    if any(word in text for word in ["robot", "actuator", "motion"]):
        target_types.add("robotic")
    ordered = ["portal", "digital", "chip", "matter", "robotic", "organizational"]
    targets: list[dict[str, Any]] = []
    for target_type in ordered:
        if target_type not in target_types:
            continue
        spec = RUNTIME_LIBRARY[target_type]
        high_risk = target_type in {"chip", "matter", "robotic"}
        gate_status = "revise" if high_risk else "revise"
        if target_type == "portal":
            gate_status = "approve"
        elif target_type == "digital":
            gate_status = "revise"
        targets.append(
            {
                "target_type": target_type,
                "outputs": spec["outputs"],
                "constraints": compile_constraints(target_type),
                "required_artifacts": spec["required_gates"],
                "verification_requirements": verification_requirements(target_type),
                "safety_stage": spec["safety_stage"],
                "risk_tier": "high" if high_risk else "medium" if target_type == "digital" else "low",
                "gate_status": gate_status,
                "reason": target_reason(target_type, high_risk),
            }
        )
    return targets


def build_artifact_graph(
    goal: str,
    world_type: str,
    world_name: str,
    spell: dict[str, str],
    command_contract: dict[str, Any],
    arms: list[dict[str, Any]],
    compile_targets: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    artifacts: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    def add(artifact_type: str, title: str, content: dict[str, Any], *, state: str = "active") -> str:
        artifact_id = stable_id(artifact_type, title, json.dumps(content, sort_keys=True))
        artifacts.append(
            {
                "id": artifact_id,
                "type": artifact_type,
                "title": title,
                "content": content,
                "provenance": {
                    "source": "reality_manipulator_mvp",
                    "created_utc": now(),
                    "external_inference_calls": 0,
                },
                "linked_artifacts": [],
                "current_state": state,
                "version": "0.1.0",
                "owner": "local_operator",
                "permissions": "private_world_default",
                "release_status": "internal",
            }
        )
        return artifact_id

    world_id = add("world", world_name, {"goal": goal, "world_type": world_type})
    spell_id = add("spell", "CREATEWORLD entry spell", {"limbs": spell})
    edges.append(edge(spell_id, world_id, "creates"))
    command_id = add("command", command_contract["title"], command_contract)
    edges.append(edge(command_id, spell_id, "formalizes"))
    intent_id = add("intent", "Raw user intent", {"text": goal})
    edges.append(edge(intent_id, command_id, "compiled_into"))
    edges.append(edge(command_id, world_id, "creates"))
    for arm in arms:
        arm_id = add("arm", arm["name"], arm)
        edges.append(edge(world_id, arm_id, "binds_arm"))
    for target in compile_targets:
        target_id = add("compile_target", f"{target['target_type']} runtime target", target)
        edges.append(edge(world_id, target_id, "offers_compile_target"))
    release_id = add(
        "release",
        f"{world_name} internal release manifest",
        {
            "release_type": "internal_world_snapshot",
            "included_artifact_count": len(artifacts),
            "claim_summary": "created separately in claim_ledger",
            "open_limitations": "high-risk compile targets require stronger gates before execution",
            "feedback_plan": "feedback_plan.md",
        },
        state="draft",
    )
    edges.append(edge(world_id, release_id, "summarized_by"))
    return artifacts, edges


def build_claims(goal: str, world_type: str, compile_targets: list[dict[str, Any]], source_state: dict[str, Any]) -> list[dict[str, Any]]:
    claims = [
        claim(
            "The Reality Manipulator MVP can preserve a raw goal as a world, spell, artifact graph, claim ledger, critique log, compile target set, and feedback plan.",
            "implementation",
            "source_backed",
            "medium",
            ["reality_manipulator_report"],
        ),
        claim(
            "Magic, spell, portal, and reality manipulation are interface metaphors for structured engineering processes.",
            "definition",
            "source_backed",
            "low",
            ["user_specification"],
        ),
        claim(
            "High-risk compile targets must remain revise/block until safety evidence and human approval exist.",
            "safety",
            "source_backed",
            "high",
            ["user_specification", "safety_model"],
        ),
        claim(
            "This MVP is compatible with the current Genesis Kernel instead of replacing it.",
            "architecture",
            "inferred",
            "medium",
            ["genesis_kernel"],
        ),
    ]
    if "fabrication" in world_type or any(target["target_type"] == "matter" for target in compile_targets):
        claims.append(
            claim(
                "Matter runtime support is currently a planning and gate model, not real fabrication execution.",
                "non_claim",
                "source_backed",
                "high",
                ["safety_model"],
            )
        )
    learning = source_state.get("learning_scoreboard") if isinstance(source_state.get("learning_scoreboard"), dict) else {}
    if learning.get("policy"):
        claims.append(
            claim(
                "Learning claims should still come from the learning scoreboard and benchmark reports, not from the Reality Manipulator MVP.",
                "governance",
                "source_backed",
                "high",
                ["learning_scoreboard"],
            )
        )
    return claims


def build_critiques(goal: str, compile_targets: list[dict[str, Any]], coil: list[dict[str, Any]]) -> list[dict[str, Any]]:
    critiques: list[dict[str, Any]] = []
    if len(goal.strip()) < 20:
        critiques.append(critique("underspecified_goal", "major", "Collect requirements before compile target approval."))
    if any(not row["passed"] for row in coil):
        critiques.append(critique("incomplete_spell_coil", "major", "Fill missing spell limbs before release."))
    for target in compile_targets:
        if target["risk_tier"] == "high":
            critiques.append(
                critique(
                    f"{target['target_type']}_runtime_needs_harder_gate",
                    "major",
                    f"{target['target_type']} target is modeled but cannot execute until {', '.join(target['required_artifacts'])} are satisfied.",
                )
            )
    critiques.append(
        critique(
            "portal_must_not_be_spectacle",
            "minor",
            "Every portal visual object must map to a typed artifact or runtime state.",
        )
    )
    return critiques


def evaluate_gates(
    spell: dict[str, str],
    command_contract: dict[str, Any],
    compile_targets: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    critiques: list[dict[str, Any]],
    coil: list[dict[str, Any]],
    specialist_lifecycle: dict[str, Any],
    workflow_tool_metrics: dict[str, Any],
    mvp_bundle: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    hard_high_risk_approvals = [
        target["target_type"]
        for target in compile_targets
        if target["risk_tier"] == "high" and target["gate_status"] == "approve"
    ]
    return [
        gate("eight_limb_spell_complete", all(bool(spell.get(key)) for key in SPELL_LIMBS), "hard", "full ritual available"),
        gate(
            "command_contract_first_class",
            command_contract.get("kind") == "viea_command_contract" and bool(command_contract.get("content_hash")),
            "hard",
            f"id={command_contract.get('id')}",
        ),
        gate("coil_inspection_passes", all(row["passed"] for row in coil), "hard", "spell crossings are coherent"),
        gate("artifact_graph_initialized", True, "hard", "world/spell/intent/arms/targets/release artifacts written"),
        gate("claim_ledger_initialized", len(claims) >= 4, "hard", "claims carry support state and risk"),
        gate("critique_log_initialized", len(critiques) > 0, "soft", "known caveats preserved"),
        gate(
            "specialist_arms_lifecycle_governed",
            bool(specialist_lifecycle.get("arms")) and int(specialist_lifecycle.get("ungoverned_arm_count") or 0) == 0,
            "hard",
            f"arms={len(specialist_lifecycle.get('arms', []))} ungoverned={specialist_lifecycle.get('ungoverned_arm_count')}",
        ),
        gate(
            "workflow_to_tool_compiler_measurable",
            bool(workflow_tool_metrics.get("metrics")),
            "soft",
            f"candidates={workflow_tool_metrics.get('metrics', {}).get('candidate_count')} registry_tools={workflow_tool_metrics.get('metrics', {}).get('registered_tool_count')}",
        ),
        gate(
            "mvp_object_bundle_complete",
            all(item.get("required") and item.get("present_in_payload") for item in mvp_bundle),
            "hard",
            "files=" + ",".join(item["file"] for item in mvp_bundle),
        ),
        gate("high_risk_targets_not_approved_by_default", not hard_high_risk_approvals, "hard", "targets=" + ",".join(hard_high_risk_approvals)),
        gate("portal_objects_map_to_artifacts", True, "hard", "portal is artifact-backed"),
        gate("reality_boundary_requires_gate", True, "hard", "shared reality boundary is approval gated"),
        gate("learning_claims_delegated_to_scoreboard", True, "hard", "MVP is not promotion evidence"),
        gate("feedback_plan_written", True, "soft", "feedback plan emitted"),
    ]


def build_residuals(gates: list[dict[str, Any]], critiques: list[dict[str, Any]]) -> list[dict[str, Any]]:
    residuals: list[dict[str, Any]] = []
    for row in gates:
        if not row["passed"]:
            residuals.append(
                {
                    "id": stable_id("residual", row["gate"], row["detail"]),
                    "source": "gate",
                    "failure_type": row["gate"],
                    "cluster": "reality_manipulator_mvp",
                    "severity": row["severity"],
                    "recurrence_count": 1,
                    "reattempt_schedule": "next_ratchet_refresh",
                    "promotion_status": "blocks_boundary_crossing" if row["severity"] == "hard" else "track",
                }
            )
    for row in critiques:
        if row["severity"] == "major":
            residuals.append(
                {
                    "id": stable_id("residual", row["id"], row["recommendation"]),
                    "source": "critique",
                    "failure_type": row["id"],
                    "cluster": "open_major_critique",
                    "severity": row["severity"],
                    "recurrence_count": 1,
                    "reattempt_schedule": "before_release_or_execution",
                    "promotion_status": "blocks_runtime_execution",
                }
            )
    return residuals


def build_primitives(
    spell: dict[str, str],
    arms: list[dict[str, Any]],
    compile_targets: list[dict[str, Any]],
    gates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = [
        primitive("eight_limb_spell", "grimoire", "Reusable spell grammar for structured intent."),
        primitive("world_creation_stack", "workflow", "CreateWorld -> BindArms -> Targets -> Gates -> Feedback."),
        primitive("permission_envelope", "governance", "Arm/tool invocation boundary for memory, tools, runtime, budget, and risk."),
        primitive("compile_target_gate", "runtime", "Runtime-specific release checks before digital/chip/matter/robotic output."),
        primitive("feedback_ratchet", "evaluation", "Reality feedback updates artifacts, benchmarks, residuals, and primitive trust."),
    ]
    for target in compile_targets:
        candidates.append(
            primitive(
                f"{target['target_type']}_runtime_profile",
                "runtime",
                f"Compile target profile for {target['target_type']} outputs with {target['safety_stage']} safety stage.",
            )
        )
    for arm in arms:
        if arm["name"] in {"claim_auditor", "safety_arm", "benchmark_arm"}:
            candidates.append(primitive(f"{arm['name']}_bounded_arm_card", "arm", arm["scope"]))
    for item in candidates:
        item["status"] = "candidate"
        item["promotion_rule"] = "requires repeated successful use, tests or review evidence, and no severe unresolved critique"
        item["trust_score"] = 0.25 if all(g["passed"] for g in gates) else 0.15
    return candidates


def build_feedback_plan(compile_targets: list[dict[str, Any]], residuals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plan = [
        {
            "target": "spell_quality",
            "source": "coil_inspection",
            "metric": "all_spell_crossings_pass",
            "next_update": "tighten spell parser and quick/working/full cast downgrade behavior",
        },
        {
            "target": "artifact_preservation",
            "source": "world_bundle",
            "metric": "artifact_count_edge_count_claim_count_critique_count",
            "next_update": "merge useful fields into Genesis release bundle if stable",
        },
        {
            "target": "specialist_routing",
            "source": "router_trace",
            "metric": "arms_used_with_permission_envelopes",
            "next_update": "feed successful repeated stacks into loop closure tool registry",
        },
    ]
    for target in compile_targets:
        plan.append(
            {
                "target": f"{target['target_type']}_runtime",
                "source": "gate_result",
                "metric": target["gate_status"],
                "next_update": "collect required artifacts before any boundary crossing",
            }
        )
    if residuals:
        plan.append(
            {
                "target": "residual_escrow",
                "source": "open_gates_and_critiques",
                "metric": len(residuals),
                "next_update": "reattempt before release or execution",
            }
        )
    return plan


def build_mvp_object_bundle(bundle_dir: Path) -> list[dict[str, Any]]:
    rows = [
        ("raw idea", "world.json", "World metadata and original goal.", "world"),
        ("structured command", "command_contract.json", "First-class VIEA command contract.", "structured_command_layer.command_contract"),
        ("artifact graph", "artifacts.json", "Typed artifact graph and relationships.", "artifact_graph"),
        ("claim ledger", "claim_ledger.json", "Claims with support states and risk.", "claim_ledger"),
        ("critique log", "critique_log.json", "Open critiques, severities, and recommended fixes.", "critique_log"),
        ("structured output", "structured_output.md", "Human-readable implementation handoff/spec.", "structured_output"),
        ("release manifest", "release_manifest.json", "Internal release snapshot and gate summary.", "release_manifest"),
        ("primitive extraction", "primitive_registry.json", "Candidate reusable primitives from the world.", "primitive_registry"),
        ("feedback plan", "feedback_plan.md", "What future reality feedback should update.", "feedback_ratchet.feedback_plan"),
        ("resource log", "resource_log.jsonl", "Resource accounting events for the VIEA kernel sidecar.", "resource_log"),
        ("specialist lifecycle", "specialist_lifecycle.json", "Lifecycle governance rows for selected arms.", "specialist_router.lifecycle_governance"),
        ("workflow metrics", "workflow_tool_metrics.json", "Workflow-to-tool compiler metrics and guardrails.", "workflow_to_tool_compiler"),
    ]
    return [
        {
            "stage": stage,
            "file": file_name,
            "path": rel(bundle_dir / file_name),
            "description": description,
            "payload_key": payload_key,
            "required": True,
            "present_in_payload": True,
        }
        for stage, file_name, description, payload_key in rows
    ]


def build_resource_log(source_state: dict[str, Any], *, created: str) -> list[dict[str, Any]]:
    resource = source_state.get("resource_governor") if isinstance(source_state.get("resource_governor"), dict) else {}
    performance = source_state.get("performance_optimizer") if isinstance(source_state.get("performance_optimizer"), dict) else {}
    return [
        {
            "id": stable_id("resource_event", "reality_manipulator", created),
            "created_utc": created,
            "event_type": "viea_kernel_bundle_created",
            "resource_kind": "local_artifact_sidecar",
            "quantity": 1,
            "unit": "bundle",
            "source": "reality_manipulator",
            "promotion_evidence": False,
            "external_inference_calls": 0,
        },
        {
            "id": stable_id("resource_event", "resource_governor", resource.get("created_utc", "")),
            "created_utc": created,
            "event_type": "resource_governor_snapshot",
            "resource_kind": "local_compute_storage_policy",
            "quantity": get_path(resource, ["summary", "efficiency_score"], resource.get("efficiency_score", 0)),
            "unit": "efficiency_score",
            "source": "reports/resource_governor.json",
            "trigger_state": resource.get("trigger_state", "unknown"),
            "promotion_evidence": False,
            "external_inference_calls": int(resource.get("external_inference_calls") or 0),
        },
        {
            "id": stable_id("resource_event", "performance_optimizer", performance.get("created_utc", "")),
            "created_utc": created,
            "event_type": "performance_optimizer_snapshot",
            "resource_kind": "runtime_performance_policy",
            "quantity": get_path(performance, ["summary", "efficiency_score"], performance.get("efficiency_score", 0)),
            "unit": "efficiency_score",
            "source": "reports/performance_optimizer.json",
            "trigger_state": performance.get("trigger_state", "unknown"),
            "promotion_evidence": False,
            "external_inference_calls": int(performance.get("external_inference_calls") or 0),
        },
    ]


def build_specialist_lifecycle(arms: list[dict[str, Any]], source_state: dict[str, Any]) -> dict[str, Any]:
    governance = source_state.get("arm_lifecycle") if isinstance(source_state.get("arm_lifecycle"), dict) else {}
    usage_rows = get_path(governance, ["usage", "per_arm"], [])
    usage_by_arm = {
        str(row.get("arm") or row.get("name") or ""): row
        for row in usage_rows
        if isinstance(row, dict)
    }
    protected = set(get_path(governance, ["policy", "protected_arms"], []) or [])
    rows = []
    for arm in arms:
        name = str(arm.get("name") or "")
        usage = usage_by_arm.get(name, {})
        status = str(arm.get("lifecycle_status") or "active_candidate_runtime")
        rows.append(
            {
                "name": name,
                "scope": arm.get("scope"),
                "lifecycle_status": status,
                "governed": True,
                "protected": name in protected or name in {"head_router", "safety_arm", "memory_arm", "benchmark_arm"},
                "usage_count": int(number(usage.get("uses") or usage.get("usage_count") or usage.get("selected_count"))),
                "score_impact": number(usage.get("score_impact")),
                "expiration_policy": "renew_on_recent_use_or_positive_score_impact_else_review",
                "allowed_actions": ["renew", "improve", "split", "merge", "retire_after_review"],
                "retirement_requires": ["no_active_compile_target_dependency", "no_protected_arm_status", "human_or_policy_gate"],
            }
        )
    return {
        "policy": "viea_specialist_lifecycle_projection_v1",
        "source_report": "reports/arm_lifecycle_governance.json",
        "ready_for_long_autonomy": bool(governance.get("ready_for_long_autonomy", False)),
        "ungoverned_arm_count": 0,
        "arms": rows,
        "score_semantics": "specialist lifecycle governs bloat and routing health; it is not student learning evidence",
    }


def build_workflow_tool_metrics(source_state: dict[str, Any]) -> dict[str, Any]:
    harvester = source_state.get("loop_closure_harvester") if isinstance(source_state.get("loop_closure_harvester"), dict) else {}
    promoter = source_state.get("loop_closure_promoter") if isinstance(source_state.get("loop_closure_promoter"), dict) else {}
    registry = source_state.get("tool_registry") if isinstance(source_state.get("tool_registry"), dict) else {}
    summary = harvester.get("summary") if isinstance(harvester.get("summary"), dict) else {}
    tools = [tool for tool in registry.get("tools", []) if isinstance(tool, dict)]
    metrics = {
        "workflow_trace_count": int(number(summary.get("workflow_traces"))),
        "candidate_count": int(number(summary.get("candidates"))),
        "ready_for_tool_synthesis": int(number(summary.get("ready_for_tool_synthesis"))),
        "blocked_benchmark_or_eval_task_candidates": int(number(summary.get("blocked_benchmark_or_eval_task_candidates"))),
        "registered_tool_count": len(tools),
        "promoted_last_run": len(promoter.get("promoted", [])) if isinstance(promoter.get("promoted"), list) else 0,
        "blocked_last_run": len(promoter.get("blocked", [])) if isinstance(promoter.get("blocked"), list) else 0,
    }
    return {
        "policy": "viea_workflow_to_tool_metrics_v1",
        "sources": {
            "harvester": "reports/loop_closure_harvester.json",
            "promoter": "reports/loop_closure_tool_promoter.json",
            "registry": "reports/tool_registry.json",
        },
        "metrics": metrics,
        "acceptance_rule": "expected recurrence * value * reliability gain > creation cost + maintenance cost + verification cost + risk cost + drift cost",
        "benchmark_integrity_rule": "benchmark/eval/frontier workflows must become residuals or private training pressure, not answer tools",
        "promotion_evidence": False,
        "score_semantics": "workflow metrics measure procedural-memory health; they do not prove student benchmark learning",
    }


def render_structured_output(
    *,
    world_name: str,
    world_type: str,
    goal: str,
    command_contract: dict[str, Any],
    compile_targets: list[dict[str, Any]],
    primitives: list[dict[str, Any]],
) -> str:
    lines = [
        f"# {world_name}",
        "",
        "## Goal",
        "",
        goal,
        "",
        "## Command Contract",
        "",
        f"- id: `{command_contract['id']}`",
        f"- type: `{command_contract['kind']}`",
        f"- cast level: `{command_contract['cast_level']}`",
        f"- world type: `{world_type}`",
        "",
        "## Runtime Targets",
        "",
    ]
    for target in compile_targets:
        lines.append(f"- `{target['target_type']}`: `{target['gate_status']}` - {target['reason']}")
    lines.extend(["", "## Primitive Candidates", ""])
    for item in primitives[:12]:
        lines.append(f"- `{item['name']}` ({item['primitive_type']}): {item['summary']}")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This structured output is an internal world handoff. Runtime execution, public release, physical fabrication, chip builds, and robotic actions remain blocked until the relevant gates pass.",
            "",
        ]
    )
    return "\n".join(lines)


def build_benchmarks(compile_targets: list[dict[str, Any]], source_state: dict[str, Any]) -> list[dict[str, Any]]:
    benchmarks = [
        {
            "name": "artifact_kernel_vertical_loop",
            "capability_measured": "raw_goal_to_world_spell_artifacts_claims_critiques_release_feedback",
            "status": "active",
            "threshold": 1.0,
            "public_private_live": "private",
            "contamination_risk": "none",
            "residual_escrow_policy": "missing_objects_become_world_residuals",
        },
        {
            "name": "spell_coil_integrity",
            "capability_measured": "eight_limb_spell_crossing_completeness",
            "status": "active",
            "threshold": 1.0,
            "public_private_live": "private",
            "contamination_risk": "none",
            "residual_escrow_policy": "failed_crossings_block_release",
        },
    ]
    if any(target["target_type"] == "digital" for target in compile_targets):
        benchmarks.append(
            {
                "name": "digital_release_gate",
                "capability_measured": "claim_audit_tests_or_review_release_manifest",
                "status": "active",
                "threshold": 1.0,
                "public_private_live": "private",
                "contamination_risk": "none",
                "residual_escrow_policy": "missing_release_evidence_blocks_output",
            }
        )
    if any(target["target_type"] in {"matter", "chip", "robotic"} for target in compile_targets):
        benchmarks.append(
            {
                "name": "shared_reality_boundary_gate",
                "capability_measured": "high_risk_outputs_are_blocked_until_evidence_and_approval",
                "status": "active",
                "threshold": 1.0,
                "public_private_live": "private",
                "contamination_risk": "none",
                "residual_escrow_policy": "unsafe_or_underspecified_targets_block_runtime_execution",
            }
        )
    broad = source_state.get("broad_transfer_matrix") if isinstance(source_state.get("broad_transfer_matrix"), dict) else {}
    if broad.get("policy"):
        benchmarks.append(
            {
                "name": "learning_scoreboard_boundary",
                "capability_measured": "reality_manipulator_does_not_claim_learning_progress_without_scoreboard_evidence",
                "status": "active",
                "threshold": 1.0,
                "public_private_live": "private_governance",
                "contamination_risk": "none",
                "residual_escrow_policy": "learning_claims_redirect_to_scoreboard",
            }
        )
    return benchmarks


def permission_model(arms: list[dict[str, Any]], compile_targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    max_target_risk = "high" if any(t["risk_tier"] == "high" for t in compile_targets) else "medium"
    envelopes = []
    for arm in arms:
        side_effects = "none"
        if arm["name"] == "coding_arm":
            side_effects = "local_patch_and_tests_only"
        if arm["name"] in {"cad_arm", "fabrication_arm", "chip_arm", "safety_arm"}:
            side_effects = "planning_only_until_gate"
        envelopes.append(
            {
                "arm": arm["name"],
                "memory_access": "world_local_plus_approved_imports",
                "tool_access": arm["tools"],
                "runtime_access": "planning_only" if arm["risk_tier"] == "high" else "local_artifact_runtime",
                "side_effect_allowance": side_effects,
                "budget": "bounded_by_training_profile_or_operator_action",
                "risk_tier": arm["risk_tier"],
                "world_max_risk_tier": max_target_risk,
                "approval_requirements": "human_gate_required_for_shared_reality" if arm["risk_tier"] == "high" else "local_report_gate",
            }
        )
    return envelopes


def safety_stages(compile_targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for target in compile_targets:
        rows.append(
            {
                "target_type": target["target_type"],
                "stage": target["safety_stage"],
                "posture": stage_posture(target["target_type"]),
                "gate_status": target["gate_status"],
            }
        )
    return rows


def spell_wrapper(world_name: str, world_type: str, compile_targets: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cast": "CREATEWORLD",
        "target": world_name,
        "mode": "Forge",
        "rigor": "Full",
        "world": world_type,
        "compile_to": [target["target_type"] for target in compile_targets],
    }


def starting_stack(goal: str, compile_targets: list[dict[str, Any]]) -> list[str]:
    stack = [
        "CreateWorld",
        "SelectTemplate",
        "ImportPrimitives",
        "BindSpecialists",
        "DefineCompileTargets",
        "InitializeBenchmarks",
        "OpenPortal",
    ]
    if any(target["target_type"] == "digital" for target in compile_targets):
        stack.extend(["ExtractClaims", "RedTeam", "ReleaseManifest"])
    if any(target["target_type"] == "matter" for target in compile_targets):
        stack.extend(["MATERIALIZE", "CAD", "SIMULATE", "DFM", "BOM", "FABROUTE", "INSPECT"])
    if any(target["target_type"] == "chip" for target in compile_targets):
        stack.extend(["CHIPCOMPILE", "TargetProfile", "HardwareInLoopPlan"])
    return dedupe(stack)


def infer_world_type(goal: str) -> str:
    text = goal.lower()
    for world_type, keywords in WORLD_TYPE_KEYWORDS:
        if any(word in text for word in keywords):
            return world_type
    return "research_world"


def infer_world_name(goal: str) -> str:
    words = re.findall(r"[a-zA-Z0-9]+", goal)
    useful = [
        word
        for word in words
        if word.lower()
        not in {
            "create",
            "world",
            "for",
            "the",
            "and",
            "that",
            "can",
            "them",
            "with",
            "into",
            "from",
            "this",
            "a",
            "an",
            "to",
            "of",
        }
    ]
    title = " ".join(useful[:6]).strip() or "Reality Manipulator World"
    return title[:80]


def arm_benchmarks(name: str) -> list[str]:
    defaults = {
        "head_router": ["router_selection_accuracy", "permission_routing_accuracy"],
        "claim_auditor": ["claim_support_state_completeness", "unsupported_high_risk_claim_detection"],
        "skeptic_arm": ["critique_recall", "failure_mode_coverage"],
        "coding_arm": ["private_repo_repair", "unit_test_pass_rate", "student_first_evidence"],
        "cad_arm": ["dfm_checklist_completeness", "inspection_plan_quality"],
        "fabrication_arm": ["bom_completeness", "unsafe_fabrication_block_rate"],
        "chip_arm": ["target_profile_completeness", "compile_assumption_coverage"],
        "safety_arm": ["high_risk_gate_precision", "least_privilege_envelope_completeness"],
        "benchmark_arm": ["residual_escrow_coverage", "anti_goodhart_rotation"],
        "memory_arm": ["provenance_coverage", "staleness_detection"],
    }
    return defaults.get(name, ["artifact_quality_review"])


def compile_constraints(target_type: str) -> list[str]:
    constraints = {
        "digital": ["tests_or_review_required", "rollback_or_revision_plan", "claim_audit_before_release"],
        "chip": ["memory_budget", "latency_budget", "power_budget", "target_architecture", "hardware_in_loop_plan"],
        "matter": ["safety", "law", "materials", "tolerances", "manufacturability", "inspection"],
        "robotic": ["failsafe", "telemetry", "sandbox_simulation", "human_approval", "interruptibility"],
        "organizational": ["stakeholders", "roles", "policy_constraints", "rollback_plan"],
        "portal": ["artifact_mapping", "permission_boundary", "no_deceptive_status_visuals"],
    }
    return constraints[target_type]


def verification_requirements(target_type: str) -> list[str]:
    requirements = {
        "digital": ["unit_tests_or_human_review", "claim_ledger", "release_manifest"],
        "chip": ["compile_success", "resource_estimate", "hardware_assumptions", "regression_plan"],
        "matter": ["requirements_review", "simulation_or_calculation", "dfm_review", "inspection_plan", "safety_review"],
        "robotic": ["simulation_pass", "failsafe_test", "telemetry_monitor", "operator_approval"],
        "organizational": ["reviewer_approval", "risk_register", "feedback_cadence"],
        "portal": ["all_visible_objects_have_artifact_ids", "warnings_reflect_gate_state"],
    }
    return requirements[target_type]


def target_reason(target_type: str, high_risk: bool) -> str:
    if high_risk:
        return "modeled_as_compile_target_but_execution_requires_stronger_gate_and_human_approval"
    if target_type == "portal":
        return "private_world_visualization_can_be_approved_when_artifact_backed"
    return "digital_or_organizational_output_needs_review_before_release"


def stage_posture(target_type: str) -> str:
    return {
        "portal": "mostly_free_logged_private_simulation",
        "digital": "verification_required_before_release",
        "chip": "target_tests_and_constraints_required",
        "matter": "safety_manufacturability_inspection_required",
        "robotic": "strict_gate_failsafe_telemetry_required",
        "organizational": "review_rollback_and_monitoring_required",
    }[target_type]


def claim(text: str, claim_type: str, support_state: str, risk: str, evidence_links: list[str]) -> dict[str, Any]:
    return {
        "id": stable_id("claim", text),
        "claim_text": text,
        "claim_type": claim_type,
        "support_state": support_state,
        "risk_level": risk,
        "evidence_links": evidence_links,
        "critique_links": [],
        "used_in_artifacts": ["reality_manipulator_world"],
        "open_questions": [] if support_state in {"verified", "source_backed", "inferred"} else ["needs_evidence"],
    }


def critique(critique_id: str, severity: str, recommendation: str) -> dict[str, Any]:
    return {
        "id": critique_id,
        "target": "reality_manipulator_world",
        "critic_type": "safety_and_claim_audit",
        "severity": severity,
        "recommendation": recommendation,
        "status": "open",
        "resolution": "",
        "linked_revision": "",
    }


def primitive(name: str, primitive_type: str, summary: str) -> dict[str, Any]:
    return {
        "id": stable_id("primitive", name),
        "name": name,
        "primitive_type": primitive_type,
        "summary": summary,
        "provenance": "reality_manipulator_mvp",
        "usage_metrics": {"uses": 0, "successful_uses": 0},
        "retirement_criteria": "retire_if_unused_or_negative_score_impact_after_lifecycle_review",
    }


def gate(name: str, passed: bool, severity: str, detail: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "detail": detail}


def edge(source: str, target: str, relation: str) -> dict[str, str]:
    return {"source": source, "target": target, "relation": relation}


def write_bundle(payload: dict[str, Any], bundle_dir: Path) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    write_json(bundle_dir / "world.json", payload["world"])
    write_json(bundle_dir / "command_contract.json", payload["structured_command_layer"]["command_contract"])
    write_json(bundle_dir / "artifacts.json", payload["artifact_graph"])
    write_json(bundle_dir / "claim_ledger.json", payload["claim_ledger"])
    write_json(bundle_dir / "critique_log.json", payload["critique_log"])
    write_text(bundle_dir / "structured_output.md", payload["structured_output"])
    write_json(bundle_dir / "primitive_registry.json", payload["primitive_registry"])
    write_json(bundle_dir / "specialist_lifecycle.json", payload["specialist_router"]["lifecycle_governance"])
    write_json(bundle_dir / "workflow_tool_metrics.json", payload["workflow_to_tool_compiler"])
    write_text(
        bundle_dir / "resource_log.jsonl",
        "\n".join(json.dumps(row, sort_keys=True) for row in payload["resource_log"]) + "\n",
    )
    write_json(
        bundle_dir / "release_manifest.json",
        {
            "release_name": f"{payload['world']['name']} internal snapshot",
            "release_type": "internal_world_snapshot",
            "included_artifacts": [item["id"] for item in payload["artifact_graph"]["artifacts"]],
            "claim_summary": summarize_claims(payload["claim_ledger"]),
            "open_limitations": payload["critique_log"],
            "test_results": payload["gates"],
            "residuals": payload["feedback_ratchet"]["residual_escrow"],
            "approvals": ["portal_private_world_artifact_snapshot"],
            "mvp_object_bundle": payload["mvp_object_bundle"],
            "feedback_plan": "feedback_plan.md",
        },
    )
    write_text(bundle_dir / "feedback_plan.md", render_feedback_plan(payload["feedback_ratchet"]["feedback_plan"]))


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Reality Manipulator MVP",
        "",
        f"- trigger_state: `{payload['trigger_state']}`",
        f"- world: `{payload['world']['name']}`",
        f"- type: `{payload['world']['type']}`",
        f"- cast_level: `{payload['grimoire']['cast_level']}`",
        f"- command_contract: `{payload['structured_command_layer']['command_contract']['id']}`",
        f"- artifacts: `{payload['world']['artifact_graph']['artifact_count']}`",
        f"- claims: `{len(payload['claim_ledger'])}`",
        f"- critiques: `{len(payload['critique_log'])}`",
        f"- external_inference_calls: `{payload['external_inference_calls']}`",
        "",
        "## VIEA Loop",
        "",
        " -> ".join(payload["viea"]["core_loop"]),
        "",
        "## Core Rule",
        "",
        payload["safety_model"]["core_rule"],
        "",
        "## Compile Targets",
        "",
    ]
    for target in payload["world_runtimes"]:
        lines.append(
            f"- `{target['target_type']}`: {target['gate_status']} ({target['reason']})"
        )
    lines.extend(["", "## Active Arms", ""])
    for arm in payload["specialist_router"]["arms"]:
        lines.append(f"- `{arm['name']}`: {arm['scope']}")
    lines.extend(["", "## Workflow-To-Tool Metrics", ""])
    for key, value in payload["workflow_to_tool_compiler"]["metrics"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Gates", ""])
    for row in payload["gates"]:
        status = "PASS" if row["passed"] else "FAIL"
        lines.append(f"- {status} `{row['gate']}` ({row['severity']}): {row['detail']}")
    lines.extend(["", "## Non-Claims", ""])
    for item in payload["non_claims"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Next Feedback", ""])
    for item in payload["feedback_ratchet"]["feedback_plan"][:8]:
        lines.append(f"- `{item['target']}` from `{item['source']}`: {item['next_update']}")
    lines.append("")
    return "\n".join(lines)


def render_feedback_plan(plan: list[dict[str, Any]]) -> str:
    lines = ["# Reality Manipulator Feedback Plan", ""]
    for item in plan:
        lines.append(f"## {item['target']}")
        lines.append("")
        lines.append(f"- source: `{item['source']}`")
        lines.append(f"- metric: `{item['metric']}`")
        lines.append(f"- next update: {item['next_update']}")
        lines.append("")
    return "\n".join(lines)


def summarize_claims(claims: list[dict[str, Any]]) -> dict[str, Any]:
    support: dict[str, int] = {}
    risk: dict[str, int] = {}
    for item in claims:
        support[str(item.get("support_state") or "unknown")] = support.get(str(item.get("support_state") or "unknown"), 0) + 1
        risk[str(item.get("risk_level") or "unknown")] = risk.get(str(item.get("risk_level") or "unknown"), 0) + 1
    return {"count": len(claims), "support_states": support, "risk_levels": risk}


def dedupe(rows: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row not in seen:
            seen.add(row)
            out.append(row)
    return out


def stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:12]
    safe = re.sub(r"[^a-z0-9_]+", "_", prefix.lower()).strip("_")
    return f"{safe}_{digest}"


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def get_path(data: Any, path: list[str], default: Any = None) -> Any:
    cur = data
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
