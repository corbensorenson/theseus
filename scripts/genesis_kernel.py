"""Genesis Kernel compiler for Project Theseus.

This is the local-first vertical slice from the Genesis Engine paper:

raw/live system evidence -> typed artifacts -> claim ledger -> critiques
-> primitive candidates -> release bundle -> feedback plan.

The important design choice is that this script does not invent a parallel
world. It ingests the reports Theseus already trusts and compiles them into a
durable artifact substrate that other gates can inspect.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "genesis_kernel_policy.json"
DEFAULT_OUT = ROOT / "reports" / "genesis_kernel" / "report.json"
DEFAULT_BUNDLE = ROOT / "reports" / "genesis_kernel" / "latest_release"

SUPPORT_STATES = {
    "verified",
    "empirical",
    "source_backed",
    "inferred",
    "speculative",
    "unsupported",
    "contradicted",
    "deprecated",
    "requires_experiment",
    "requires_expert_review",
}
RISKS = {"low", "medium", "high", "severe"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        nargs="?",
        choices=["ingest-theseus", "check"],
        default="ingest-theseus",
    )
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE.relative_to(ROOT)))
    parser.add_argument("--report", default=str(DEFAULT_OUT.relative_to(ROOT)))
    args = parser.parse_args()

    if args.command == "check":
        return check_report(ROOT / args.report)

    policy = read_json(ROOT / args.policy)
    if not isinstance(policy, dict) or not policy:
        raise SystemExit(f"missing or invalid Genesis policy: {args.policy}")
    compiler = GenesisCompiler(policy=policy, bundle_dir=ROOT / args.bundle_dir)
    report = compiler.compile()
    write_json(ROOT / args.out, report)
    print(json.dumps(report_summary(report), indent=2))
    return 0


class GenesisCompiler:
    def __init__(self, *, policy: dict[str, Any], bundle_dir: Path) -> None:
        self.policy = policy
        self.bundle_dir = bundle_dir
        self.now = now()
        self.artifacts: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.source_ids: dict[str, str] = {}
        self.claim_ids: list[str] = []
        self.critique_ids: list[str] = []
        self.primitive_ids: list[str] = []
        self.feedback_ids: list[str] = []

    def compile(self) -> dict[str, Any]:
        state = self.load_sources()
        project_id = self.add_artifact(
            "project",
            "Project Theseus / SparkStream Genesis Kernel Snapshot",
            {
                "thesis": (
                    "Theseus should grow through typed evidence, benchmark pressure, "
                    "small-student interventions, verified primitives, and feedback."
                ),
                "release_type": self.policy.get("release_type"),
                "genesis_role": "umbrella artifact substrate for RMI, Benchmaxxing, Loop Closure, Octopus Router, and SymLiquid",
            },
            support_state="source_backed",
            risk="medium",
            origin_type="user_input",
        )
        self.build_source_artifacts(state, project_id)
        self.build_core_system_artifacts(project_id)
        self.build_claims(state)
        self.build_benchmark_artifacts(state)
        self.build_critiques(state)
        self.build_primitives(state)
        self.build_feedback(state)
        release_id, manifest = self.build_release(project_id)
        debt = self.compute_artifact_debt()
        feedback_plan = self.write_feedback_plan(state, debt)
        paths = self.write_bundle(manifest, debt, feedback_plan)
        report = {
            "policy": "project_theseus_genesis_kernel_report_v0",
            "created_utc": self.now,
            "config": relative(DEFAULT_POLICY),
            "bundle_dir": relative(self.bundle_dir),
            "release_artifact_id": release_id,
            "summary": {
                "artifact_count": len(self.artifacts),
                "edge_count": len(self.edges),
                "claim_count": len(self.claim_ids),
                "critique_count": len(self.critique_ids),
                "primitive_candidate_count": len(self.primitive_ids),
                "feedback_count": len(self.feedback_ids),
                "open_major_critiques": debt["open_major_critiques"],
                "unsupported_high_risk_claims": debt["unsupported_high_risk_claim_count"],
                "artifact_debt_score": debt["artifact_debt_score"],
                "trigger_state": trigger_state(debt, manifest["gates"]),
            },
            "release_gates": manifest["gates"],
            "artifact_debt": debt,
            "next_best_action": next_best_action(state, debt, manifest["gates"]),
            "bundle_paths": paths,
            "external_inference_calls": 0,
        }
        return report

    def load_sources(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        for key, rel_path in (self.policy.get("source_reports") or {}).items():
            path = ROOT / str(rel_path)
            value = read_json(path)
            state[str(key)] = value
        return state

    def build_source_artifacts(self, state: dict[str, Any], project_id: str) -> None:
        required = set(str(item) for item in self.policy.get("required_reports", []))
        for key, rel_path in (self.policy.get("source_reports") or {}).items():
            key = str(key)
            rel_path = str(rel_path)
            path = ROOT / rel_path
            data = state.get(key)
            present = path.exists() and data not in ({}, [])
            if present:
                source_id = self.add_artifact(
                    "source",
                    f"Report source: {key}",
                    {
                        "key": key,
                        "path": rel_path,
                        "sha256": file_sha256(path),
                        "summary": summarize_report(data),
                    },
                    support_state="source_backed",
                    risk="low",
                    origin_type="source_import",
                )
                self.source_ids[key] = source_id
                self.add_edge(source_id, project_id, "derives_from", {"path": rel_path})
            elif key in required:
                critique_id = self.add_critique(
                    title=f"Missing required report: {key}",
                    target_artifact=project_id,
                    severity="major",
                    recommended_action="regenerate",
                    summary=f"Required Genesis source report is missing or empty: {rel_path}",
                    source_keys=[],
                )
                self.add_edge(critique_id, project_id, "blocks", {"reason": "missing_required_report"})

    def build_core_system_artifacts(self, project_id: str) -> None:
        systems = [
            (
                "Theseus Hive",
                "Routed local autonomy, benchmark pressure, residual escrow, teacher-gated self-edit, and compute scheduling.",
                ["benchmark_ratchet", "residual_escrow", "teacher_self_edit", "hive_scheduler"],
            ),
            (
                "RMI Growth Loop",
                "Small-student capability growth through pressure, transfer, loop closure, promotion gates, and feedback.",
                ["benchmaxxing", "octopus_router", "cognitive_loop_closure", "model_growth_gate"],
            ),
            (
                "Genesis Kernel for Theseus",
                "Typed artifact substrate that makes claims, critiques, primitives, releases, and feedback inspectable.",
                ["artifact_graph", "claim_ledger", "critique_tribunal", "release_manifest"],
            ),
        ]
        for name, purpose, components in systems:
            system_id = self.add_artifact(
                "system",
                name,
                {"purpose": purpose, "components": components},
                support_state="inferred",
                risk="medium",
                origin_type="generated_patch",
            )
            self.add_edge(system_id, project_id, "uses", {"role": "subsystem"})

    def build_claims(self, state: dict[str, Any]) -> None:
        frontier = as_dict(state.get("frontier_policy"))
        pressure = as_dict(frontier.get("frontier_pressure"))
        candidate = as_dict(state.get("candidate_gate"))
        architecture = as_dict(state.get("architecture_governance"))
        arch_state = as_dict(architecture.get("state"))
        curriculum = as_dict(state.get("benchmaxx_curriculum"))
        model_growth = as_dict(state.get("model_growth_gate"))
        performance = as_dict(state.get("performance_optimizer"))
        hive = as_dict(state.get("hive_scheduler"))
        code_forge = as_dict(state.get("code_residual_forge"))
        code_transfer = as_dict(state.get("code_transfer_artifacts"))
        code_rotation = as_dict(state.get("code_frontier_rotation"))
        code_repair = as_dict(state.get("local_code_repair_organism"))
        self_edit = as_dict(state.get("self_edit_experiment_lane"))
        memory_probe = as_dict(state.get("long_horizon_memory_probe"))
        tst = as_dict(state.get("token_superposition"))
        loop_promoter = as_dict(state.get("loop_closure_tool_promoter"))
        tools = as_dict(state.get("tool_registry"))
        teacher = as_dict(state.get("teacher_oracle"))
        watchdog = as_dict(state.get("autonomy_watchdog"))

        family = str(frontier.get("frontier_family") or pressure.get("next_frontier_family") or arch_state.get("frontier_family") or "unknown")
        card = str(frontier.get("pressure_card_id") or pressure.get("next_pressure_card_id") or arch_state.get("pressure_card_id") or "unknown")
        active_score = arch_state.get("frontier_score")
        self.add_claim(
            "Current active pressure is the coding-first frontier selected by frontier policy.",
            claim_type="empirical",
            support_state="source_backed",
            risk="medium",
            evidence_keys=["frontier_policy", "architecture_governance", "benchmaxx_curriculum"],
            body={
                "frontier_family": family,
                "pressure_card_id": card,
                "frontier_score": active_score,
                "active_frontier_wall": bool(pressure.get("active_frontier_wall")),
                "attempt_count": pressure.get("active_frontier_attempt_count"),
                "programming_first": get_path(curriculum, ["next_frontier", "programming_first"], None),
            },
        )

        failed_gates = [
            str(row.get("gate"))
            for row in candidate.get("checks", [])
            if isinstance(row, dict) and not row.get("passed")
        ]
        self.add_claim(
            "Candidate promotion is currently blocked by explicit gates rather than by hidden teacher preference.",
            claim_type="empirical",
            support_state="verified" if candidate else "unsupported",
            risk="high",
            evidence_keys=["candidate_gate"],
            body={
                "promote": candidate.get("promote"),
                "passed": candidate.get("passed"),
                "total": candidate.get("total"),
                "failed_gates": failed_gates,
            },
        )

        self.add_claim(
            "Code frontiers must emit residual-transfer artifacts and rotate within the programming family before model growth.",
            claim_type="design",
            support_state="source_backed" if code_forge else "unsupported",
            risk="medium",
            evidence_keys=["code_residual_forge", "code_transfer_artifacts", "code_frontier_rotation", "benchmaxx_curriculum"],
            body={
                "forge_trigger_state": code_forge.get("trigger_state"),
                "active_card_id": get_path(code_forge, ["summary", "active_card_id"], None),
                "dominant_residual_class": get_path(code_forge, ["summary", "dominant_residual_class"], None),
                "cluster_count": get_path(code_forge, ["summary", "cluster_count"], None),
                "transfer_artifacts": get_path(code_forge, ["summary", "transfer_artifacts"], None),
                "rotation_decision": code_rotation.get("decision") or get_path(code_forge, ["summary", "rotation_decision"], None),
                "selected_card_id": code_rotation.get("selected_card_id") or get_path(code_forge, ["summary", "selected_card_id"], None),
                "transfer_index_count": get_path(code_transfer, ["summary", "artifact_count"], None),
            },
        )

        self.add_claim(
            "Code transfer should count only when the next runner loads artifacts and reports a heredity delta.",
            claim_type="empirical",
            support_state="empirical" if code_repair else "unsupported",
            risk="medium",
            evidence_keys=["local_code_repair_organism", "code_transfer_artifacts", "candidate_gate"],
            body={
                "repair_policy": code_repair.get("policy"),
                "card_id": code_repair.get("card_id"),
                "baseline_pass_rate": get_path(code_repair, ["summary", "baseline_pass_rate"], None),
                "transfer_pass_rate": get_path(code_repair, ["summary", "transfer_pass_rate"], None),
                "pass_rate_delta": get_path(code_repair, ["summary", "pass_rate_delta"], None),
                "transfer_loaded": get_path(code_repair, ["summary", "transfer_loaded"], None),
                "transfer_altered_behavior": get_path(code_repair, ["summary", "transfer_altered_behavior"], None),
            },
        )

        self.add_claim(
            "Self-edit is constrained to bounded source-patch experiments with verification and rollback plans.",
            claim_type="design",
            support_state="source_backed" if self_edit else "unsupported",
            risk="high",
            evidence_keys=["self_edit_experiment_lane", "architecture_governance"],
            body={
                "policy": self_edit.get("policy"),
                "trigger_state": self_edit.get("trigger_state"),
                "experiment_count": len(self_edit.get("experiments", []) if isinstance(self_edit.get("experiments"), list) else []),
                "commit_allowed": self_edit.get("commit_allowed"),
                "external_inference_calls": self_edit.get("external_inference_calls"),
            },
        )

        self.add_claim(
            "Long-horizon autonomy memory must preserve goals, reject stale decoys, and recover the same next action.",
            claim_type="empirical",
            support_state="empirical" if memory_probe else "unsupported",
            risk="medium",
            evidence_keys=["long_horizon_memory_probe", "benchmaxx_curriculum", "frontier_policy"],
            body={
                "policy": memory_probe.get("policy"),
                "trigger_state": memory_probe.get("trigger_state"),
                "score": memory_probe.get("score"),
                "horizons_hours": memory_probe.get("horizons_hours"),
            },
        )

        self.add_claim(
            "The local student should remain small until cheaper interventions are exhausted under evidence.",
            claim_type="design",
            support_state="source_backed",
            risk="medium",
            evidence_keys=["model_growth_gate", "architecture_governance"],
            body={
                "model_growth_allowed": model_growth.get("model_growth_allowed"),
                "missing_evidence": model_growth.get("missing_evidence"),
                "hard_blockers": model_growth.get("hard_blockers"),
                "allowed_growth_types": model_growth.get("allowed_growth_types"),
            },
        )

        best_tst = as_dict(tst.get("best_variant"))
        promotion = as_dict(tst.get("promotion_decision"))
        self.add_claim(
            "Token Superposition Training has real Rust/CUDA speed evidence but is not yet a promoted training lane.",
            claim_type="empirical",
            support_state="empirical" if tst else "unsupported",
            risk="medium",
            evidence_keys=["token_superposition"],
            body={
                "backend": tst.get("backend"),
                "cuda_fallback": tst.get("cuda_fallback"),
                "best_variant": best_tst.get("id"),
                "best_train_speedup": best_tst.get("measured_train_speedup_vs_baseline"),
                "best_total_speedup": best_tst.get("measured_total_speedup_vs_baseline"),
                "combined_loss_delta_vs_baseline": best_tst.get("combined_loss_delta_vs_baseline"),
                "code_loss_delta_vs_baseline": best_tst.get("code_loss_delta_vs_baseline"),
                "promotion_status": promotion.get("status"),
            },
        )

        self.add_claim(
            "The system is already converting repeated workflows into tools, but the tool library should keep extracting verified loops.",
            claim_type="empirical",
            support_state="source_backed" if loop_promoter else "unsupported",
            risk="medium",
            evidence_keys=["loop_closure_tool_promoter", "loop_closure_harvester", "tool_registry"],
            body={
                "before_tools": loop_promoter.get("before_tools"),
                "after_tools": loop_promoter.get("after_tools"),
                "promoted": loop_promoter.get("promoted"),
                "registry_health": tools.get("registry_health"),
            },
        )

        self.add_claim(
            "Current compute routing is local CUDA-first with no external inference calls in the Genesis pass.",
            claim_type="empirical",
            support_state="source_backed" if performance or hive else "unsupported",
            risk="medium",
            evidence_keys=["performance_optimizer", "hive_scheduler"],
            body={
                "preferred_training_backend": get_path(performance, ["summary", "preferred_training_backend"], None),
                "cuda_available": get_path(performance, ["summary", "cuda_available"], None),
                "gpu_free_mib": get_path(performance, ["summary", "gpu_free_mib"], None),
                "real_worker_chunks": get_path(hive, ["summary", "real_worker_chunks"], None),
                "node_count": hive.get("node_count"),
            },
        )

        self.add_claim(
            "Genesis should wrap RMI as invention infrastructure rather than merge into a single bloated runtime.",
            claim_type="design",
            support_state="inferred",
            risk="medium",
            evidence_keys=[],
            body={
                "relationship": {
                    "genesis_engine": "artifact memory, claim verification, primitive extraction, release discipline, and feedback",
                    "rmi": "AI-system capability growth architecture",
                    "benchmaxxing": "benchmark curriculum pressure",
                    "loop_closure": "workflow-to-tool procedural memory",
                    "octopus_router": "runtime anatomy for specialists",
                    "symliquid": "local student substrate",
                }
            },
        )

        self.add_claim(
            "Teacher and watchdog labor is logged as bounded artifact-producing correction rather than silent autonomy.",
            claim_type="empirical",
            support_state="source_backed" if teacher or watchdog else "unsupported",
            risk="high",
            evidence_keys=["teacher_oracle", "autonomy_watchdog", "sparkstream_status"],
            body={
                "teacher_status": teacher.get("status"),
                "teacher_completed_utc": teacher.get("completed_utc") or teacher.get("created_utc"),
                "watchdog_trigger_state": watchdog.get("trigger_state"),
                "watchdog_applied_actions": watchdog.get("applied_actions"),
                "sparkstream_phase": get_path(state, ["sparkstream_status", "phase"], None),
            },
        )

    def build_benchmark_artifacts(self, state: dict[str, Any]) -> None:
        ledger = state.get("benchmark_ledger")
        if not isinstance(ledger, list):
            return
        rows = [row for row in ledger if isinstance(row, dict)]
        rows.sort(key=lambda row: float(row.get("residual") or 0.0), reverse=True)
        for row in rows[:16]:
            benchmark_id = self.add_artifact(
                "benchmark",
                str(row.get("benchmark_name") or "unnamed benchmark"),
                {
                    "metric": row.get("metric"),
                    "score": row.get("score"),
                    "residual": row.get("residual"),
                    "lifecycle": row.get("lifecycle"),
                    "benchmark_type": row.get("benchmark_type"),
                    "wall_type": row.get("wall_type"),
                    "recommended_intervention": row.get("recommended_intervention"),
                    "best_report": row.get("best_report"),
                    "graduation_policy": row.get("graduation_policy"),
                },
                support_state="empirical",
                risk="medium" if row.get("lifecycle") == "frontier" else "low",
                origin_type="source_import",
            )
            if "benchmark_ledger" in self.source_ids:
                self.add_edge(self.source_ids["benchmark_ledger"], benchmark_id, "supports", {"kind": "benchmark_row"})

    def build_critiques(self, state: dict[str, Any]) -> None:
        candidate = as_dict(state.get("candidate_gate"))
        project_target = self.find_artifact_by_title("Project Theseus / SparkStream Genesis Kernel Snapshot")
        for row in candidate.get("checks", []):
            if not isinstance(row, dict) or row.get("passed"):
                continue
            gate = str(row.get("gate") or "unknown_gate")
            severity = "major" if gate in {"active_frontier_clears_floor", "candidate_profile_evidence_complete"} else "moderate"
            critique_id = self.add_critique(
                title=f"Candidate gate failed: {gate}",
                target_artifact=project_target,
                severity=severity,
                recommended_action="test",
                summary=str(row.get("evidence") or "Candidate gate did not pass."),
                source_keys=["candidate_gate"],
            )
            self.add_edge(critique_id, project_target, "blocks", {"gate": gate})

        tst = as_dict(state.get("token_superposition"))
        if tst:
            for gate in tst.get("gates", []):
                if not isinstance(gate, dict) or gate.get("passed"):
                    continue
                critique_id = self.add_critique(
                    title=f"TST gate failed: {gate.get('gate')}",
                    target_artifact=project_target,
                    severity="moderate",
                    recommended_action="defer",
                    summary=str(gate.get("evidence") or "Token Superposition Training gate did not pass."),
                    source_keys=["token_superposition"],
                )
                self.add_edge(critique_id, project_target, "blocks", {"gate": gate.get("gate")})

        model_growth = as_dict(state.get("model_growth_gate"))
        for name in model_growth.get("hard_blockers", []) or []:
            critique_id = self.add_critique(
                title=f"Model growth hard blocker: {name}",
                target_artifact=project_target,
                severity="fatal",
                recommended_action="fix",
                summary="Model growth must stay blocked until this hard blocker clears.",
                source_keys=["model_growth_gate"],
            )
            self.add_edge(critique_id, project_target, "blocks", {"gate": name})
        for name in model_growth.get("missing_evidence", []) or []:
            critique_id = self.add_critique(
                title=f"Model growth missing evidence: {name}",
                target_artifact=project_target,
                severity="major",
                recommended_action="test",
                summary="Cheaper interventions need evidence before parameter or substrate growth.",
                source_keys=["model_growth_gate"],
            )
            self.add_edge(critique_id, project_target, "blocks", {"gate": name})

    def build_primitives(self, state: dict[str, Any]) -> None:
        for item in self.policy.get("core_primitives", []) or []:
            if not isinstance(item, dict):
                continue
            self.add_primitive(
                name=str(item.get("name")),
                primitive_type=str(item.get("primitive_type") or "workflow"),
                summary=str(item.get("summary") or ""),
                trust_score=0.65,
                observed_in=list(self.source_ids.values()),
                known_failure_modes=[
                    "false_rigor_if_labels_are_not_backed_by_evidence",
                    "bloat_if_every_phrase_becomes_a_primitive",
                ],
            )

        registry = as_dict(state.get("tool_registry"))
        active_tools = [
            row
            for row in registry.get("tools", []) or []
            if isinstance(row, dict) and str(row.get("lifecycle") or "active") == "active"
        ]
        for row in active_tools:
            if not isinstance(row, dict):
                continue
            self.add_primitive(
                name=str(row.get("tool_name") or "promoted_tool"),
                primitive_type="workflow",
                summary=str(row.get("purpose") or f"Verified tool for {row.get('task_family') or 'unknown task family'}."),
                trust_score=0.72,
                observed_in=[self.source_ids[key] for key in ("tool_registry", "loop_closure_tool_promoter") if key in self.source_ids],
                known_failure_modes=(row.get("retirement_criteria") or ["schema_drift", "runtime_environment_changes"])[:4],
            )

        tst = as_dict(state.get("token_superposition"))
        if tst:
            best = as_dict(tst.get("best_variant"))
            self.add_primitive(
                name="Governed Token Superposition Training Lane",
                primitive_type="compression",
                summary="Early bagged-token training can be tested as a compute-efficiency primitive, but only promoted when ordinary AR recovery and code loss gates pass.",
                trust_score=0.38,
                observed_in=[self.source_ids["token_superposition"]] if "token_superposition" in self.source_ids else [],
                known_failure_modes=[
                    "apparent_speedup_without_quality_recovery",
                    "extra_data_consumption_masking_compute_efficiency",
                    "code_frontier_regression",
                ],
                extra={
                    "status": get_path(tst, ["promotion_decision", "status"], None),
                    "best_variant": best.get("id"),
                    "best_train_speedup": best.get("measured_train_speedup_vs_baseline"),
                },
            )

        self.add_primitive(
            name="Corben Invention Genome v0",
            primitive_type="memory",
            summary="A first-pass compressed map of recurring project patterns: root-system synthesis, artifact ledgers, ratchets, modular specialist anatomy, and self-hosting release discipline.",
            trust_score=0.5,
            observed_in=[],
            known_failure_modes=["overexpansion_before_minimum_viable_loop", "too_many_named_layers_without_runtime_pressure"],
            extra={
                "strengths": [
                    "root_system_synthesis",
                    "cross_domain_pattern_extraction",
                    "benchmark_pressure_design",
                    "governed_autonomy",
                ],
                "counterforces": [
                    "minimum_viable_loop",
                    "release_gates",
                    "artifact_debt_dashboard",
                    "reality_feedback_before_new_abstractions",
                ],
            },
        )

    def build_feedback(self, state: dict[str, Any]) -> None:
        candidate = as_dict(state.get("candidate_gate"))
        tst = as_dict(state.get("token_superposition"))
        frontier = as_dict(state.get("frontier_policy"))
        architecture = as_dict(state.get("architecture_governance"))
        self.add_feedback(
            title="Candidate promotion gate feedback",
            target_title="RMI Growth Loop",
            source="benchmark",
            result="mixed",
            interpretation=(
                "Most promotion gates pass, but the active frontier floor remains open, "
                "so release discipline is correctly preventing premature promotion."
            ),
            evidence_keys=["candidate_gate", "frontier_policy"],
            metrics={
                "promote": candidate.get("promote"),
                "passed": candidate.get("passed"),
                "total": candidate.get("total"),
                "failed_gates": [row.get("gate") for row in candidate.get("checks", []) if isinstance(row, dict) and not row.get("passed")],
            },
        )
        if tst:
            self.add_feedback(
                title="Token Superposition Training feedback",
                target_title="Governed Token Superposition Training Lane",
                source="benchmark",
                result="negative",
                interpretation=(
                    "The Rust/CUDA implementation produced speed evidence, but quality recovery failed, "
                    "so this primitive remains experimental rather than live."
                ),
                evidence_keys=["token_superposition"],
                metrics={
                    "status": get_path(tst, ["promotion_decision", "status"], None),
                    "best_train_speedup": get_path(tst, ["best_variant", "measured_train_speedup_vs_baseline"], None),
                    "combined_loss_delta_vs_baseline": get_path(tst, ["best_variant", "combined_loss_delta_vs_baseline"], None),
                    "code_loss_delta_vs_baseline": get_path(tst, ["best_variant", "code_loss_delta_vs_baseline"], None),
                },
            )
        self.add_feedback(
            title="Frontier policy feedback",
            target_title="Theseus Hive",
            source="observation",
            result="mixed_positive",
            interpretation="The system is staying aligned with programming-first pressure while preserving deferred non-code frontiers.",
            evidence_keys=["frontier_policy", "benchmaxx_curriculum", "architecture_governance"],
            metrics={
                "frontier_family": frontier.get("frontier_family"),
                "pressure_card_id": frontier.get("pressure_card_id"),
                "recommended_next_experiment": architecture.get("recommended_next_experiment"),
            },
        )

    def build_release(self, project_id: str) -> tuple[str, dict[str, Any]]:
        claim_summary: dict[str, int] = {}
        risk_summary: dict[str, int] = {}
        for artifact_id in self.claim_ids:
            artifact = self.artifacts[artifact_id]
            confidence = artifact.get("confidence") or {}
            state = str(confidence.get("support_state") or "unsupported")
            risk = str(confidence.get("risk") or "medium")
            claim_summary[state] = claim_summary.get(state, 0) + 1
            risk_summary[risk] = risk_summary.get(risk, 0) + 1
        gates = self.evaluate_release_gates()
        manifest = {
            "release_name": "Project Theseus Genesis Kernel Self-Hosting Snapshot",
            "release_type": self.policy.get("release_type"),
            "created_utc": self.now,
            "claim_summary": claim_summary,
            "risk_summary": risk_summary,
            "open_limitations": open_limitations(self.artifacts),
            "included_artifacts": sorted(self.artifacts.keys()),
            "bundle_files": [
                "artifacts.json",
                "edges.jsonl",
                "events.jsonl",
                "claim_ledger.csv",
                "critique_log.md",
                "primitive_candidates.json",
                "artifact_debt.json",
                "feedback_plan.md",
                "release_manifest.json",
            ],
            "gates": gates,
            "human_owner": "Corben Sorenson",
            "external_inference_calls": 0,
        }
        release_id = self.add_artifact(
            "release",
            manifest["release_name"],
            manifest,
            support_state="verified",
            risk="medium",
            origin_type="tool_output",
        )
        self.add_edge(release_id, project_id, "derives_from", {"kind": "self_hosting_snapshot"})
        for artifact_id in list(self.artifacts.keys()):
            if artifact_id != release_id:
                self.add_edge(release_id, artifact_id, "uses", {"release_includes": True})
        return release_id, manifest

    def evaluate_release_gates(self) -> list[dict[str, Any]]:
        required_keys = set(str(item) for item in self.policy.get("required_reports", []))
        source_loaded = required_keys <= set(self.source_ids.keys())
        unsupported_high = [
            artifact_id
            for artifact_id in self.claim_ids
            if claim_is_unsupported_high_risk(self.artifacts[artifact_id])
        ]
        candidate = read_json(ROOT / "reports" / "candidate_promotion_gate.json")
        candidate_missing = []
        if isinstance(candidate, dict):
            candidate_missing = [
                row.get("gate")
                for row in candidate.get("checks", [])
                if isinstance(row, dict) and not row.get("passed") and "missing" in str(row.get("gate") or "")
            ]
        candidate_bad_pass = bool(candidate.get("promote") and candidate_missing) if isinstance(candidate, dict) else False
        return [
            gate("source_reports_loaded", source_loaded, "hard", f"loaded={len(self.source_ids)} required={len(required_keys)}"),
            gate("claim_ledger_written", bool(self.claim_ids), "hard", f"claims={len(self.claim_ids)}"),
            gate("critique_log_written", bool(self.critique_ids), "soft", f"critiques={len(self.critique_ids)}"),
            gate("primitive_candidates_written", bool(self.primitive_ids), "hard", f"primitives={len(self.primitive_ids)}"),
            gate("release_manifest_written", True, "hard", "manifest generated"),
            gate("feedback_plan_written", bool(self.feedback_ids), "hard", f"feedback_records={len(self.feedback_ids)}"),
            gate("high_risk_unsupported_claims_accounted", not unsupported_high, "hard", f"unsupported_high_risk={len(unsupported_high)}"),
            gate("candidate_promotion_not_passing_with_missing_gates", not candidate_bad_pass, "hard", f"missing_gates={candidate_missing}"),
        ]

    def compute_artifact_debt(self) -> dict[str, Any]:
        referenced = set()
        for edge_row in self.edges:
            referenced.add(str(edge_row.get("from_artifact")))
            referenced.add(str(edge_row.get("to_artifact")))
        orphan_artifacts = [
            artifact_id
            for artifact_id in self.artifacts
            if artifact_id not in referenced and self.artifacts[artifact_id].get("type") not in {"claim", "critique"}
        ]
        open_major = [
            artifact_id
            for artifact_id in self.critique_ids
            if self.artifacts[artifact_id].get("body", {}).get("severity") in {"major", "fatal"}
            and self.artifacts[artifact_id].get("body", {}).get("status") == "open"
        ]
        unsupported_high = [
            artifact_id for artifact_id in self.claim_ids if claim_is_unsupported_high_risk(self.artifacts[artifact_id])
        ]
        feedback_debt = max(0, len([a for a in self.artifacts.values() if a.get("type") == "system"]) - len(self.feedback_ids))
        primitive_pending = len([artifact_id for artifact_id in self.primitive_ids if self.artifacts[artifact_id].get("status") == "draft"])
        score = (
            len(unsupported_high) * 5
            + len([a for a in open_major if self.artifacts[a].get("body", {}).get("severity") == "fatal"]) * 7
            + len(open_major) * 3
            + len(orphan_artifacts)
            + feedback_debt * 2
            + primitive_pending
        )
        return {
            "unsupported_high_risk_claim_count": len(unsupported_high),
            "unsupported_high_risk_claims": unsupported_high,
            "open_major_critiques": len(open_major),
            "open_major_critique_artifacts": open_major,
            "orphan_artifact_count": len(orphan_artifacts),
            "orphan_artifacts": orphan_artifacts[:20],
            "primitive_candidates_pending": primitive_pending,
            "feedback_debt": feedback_debt,
            "artifact_debt_score": score,
        }

    def write_feedback_plan(self, state: dict[str, Any], debt: dict[str, Any]) -> str:
        frontier = as_dict(state.get("frontier_policy"))
        candidate = as_dict(state.get("candidate_gate"))
        lines = [
            "# Genesis Feedback Plan",
            "",
            "## Immediate Measurements",
            "",
            f"- Re-run the active frontier ({frontier.get('frontier_family')}/{frontier.get('pressure_card_id')}) and compare score, floor, residual, runtime, and generated transfer artifact.",
            f"- Keep candidate promotion blocked while failed gates remain: {failed_candidate_gates(candidate)}.",
            "- Promote a primitive only when it has recurrence, compression value, test evidence, and no severe unresolved critique.",
            "- Feed benchmark residuals into primitive trust updates instead of letting failed cases vanish into logs.",
            "",
            "## Artifact Debt Watch",
            "",
            f"- Unsupported high-risk claims: {debt['unsupported_high_risk_claim_count']}.",
            f"- Open major critiques: {debt['open_major_critiques']}.",
            f"- Orphan artifacts: {debt['orphan_artifact_count']}.",
            f"- Feedback debt: {debt['feedback_debt']}.",
            "",
            "## Next Loop",
            "",
            "1. Refresh source reports after every pressure profile.",
            "2. Compile this Genesis bundle.",
            "3. Treat release-manifest gates as context for teacher self-edit and model-growth governance.",
            "4. Extract only primitives that reduce future work or improve quality under evidence.",
        ]
        return "\n".join(lines) + "\n"

    def write_bundle(self, manifest: dict[str, Any], debt: dict[str, Any], feedback_plan: str) -> dict[str, str]:
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "artifacts": self.bundle_dir / "artifacts.json",
            "edges": self.bundle_dir / "edges.jsonl",
            "events": self.bundle_dir / "events.jsonl",
            "claim_ledger": self.bundle_dir / "claim_ledger.csv",
            "critique_log": self.bundle_dir / "critique_log.md",
            "primitive_candidates": self.bundle_dir / "primitive_candidates.json",
            "artifact_debt": self.bundle_dir / "artifact_debt.json",
            "feedback_plan": self.bundle_dir / "feedback_plan.md",
            "release_manifest": self.bundle_dir / "release_manifest.json",
        }
        write_json(paths["artifacts"], {"artifacts": sorted(self.artifacts.values(), key=lambda row: row["id"])})
        write_jsonl(paths["edges"], self.edges)
        write_jsonl(paths["events"], self.events)
        self.write_claim_ledger(paths["claim_ledger"])
        paths["critique_log"].write_text(self.render_critique_log(), encoding="utf-8")
        write_json(paths["primitive_candidates"], self.render_primitives())
        write_json(paths["artifact_debt"], debt)
        paths["feedback_plan"].write_text(feedback_plan, encoding="utf-8")
        write_json(paths["release_manifest"], manifest)
        return {key: relative(path) for key, path in paths.items()}

    def write_claim_ledger(self, path: Path) -> None:
        fields = [
            "id",
            "claim",
            "claim_type",
            "support_state",
            "risk",
            "evidence",
            "objections",
            "used_in",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for artifact_id in self.claim_ids:
                artifact = self.artifacts[artifact_id]
                body = artifact.get("body") or {}
                confidence = artifact.get("confidence") or {}
                writer.writerow(
                    {
                        "id": artifact_id,
                        "claim": body.get("text") or artifact.get("title"),
                        "claim_type": body.get("claim_type"),
                        "support_state": confidence.get("support_state"),
                        "risk": confidence.get("risk"),
                        "evidence": ";".join(body.get("evidence") or []),
                        "objections": ";".join(body.get("objections") or []),
                        "used_in": ";".join(body.get("used_in") or []),
                    }
                )

    def render_critique_log(self) -> str:
        lines = ["# Genesis Critique Log", ""]
        if not self.critique_ids:
            lines.extend(["No critiques recorded.", ""])
            return "\n".join(lines)
        for artifact_id in self.critique_ids:
            artifact = self.artifacts[artifact_id]
            body = artifact.get("body") or {}
            lines.extend(
                [
                    f"## {artifact.get('title')}",
                    "",
                    f"- ID: {artifact_id}",
                    f"- Severity: {body.get('severity')}",
                    f"- Status: {body.get('status')}",
                    f"- Recommended action: {body.get('recommended_action')}",
                    f"- Target: {body.get('target_artifact')}",
                    f"- Summary: {body.get('summary')}",
                    "",
                ]
            )
        return "\n".join(lines)

    def render_primitives(self) -> dict[str, Any]:
        rows = []
        for artifact_id in self.primitive_ids:
            artifact = self.artifacts[artifact_id]
            body = artifact.get("body") or {}
            rows.append(
                {
                    "id": artifact_id,
                    "name": body.get("name") or artifact.get("title"),
                    "primitive_type": body.get("primitive_type"),
                    "summary": body.get("summary"),
                    "trust_score": body.get("trust_score"),
                    "observed_in": body.get("observed_in"),
                    "known_failure_modes": body.get("known_failure_modes"),
                    "recommended_uses": body.get("recommended_uses"),
                    "status": artifact.get("status"),
                }
            )
        return {
            "policy": "project_theseus_genesis_primitive_candidates_v0",
            "created_utc": self.now,
            "primitive_count": len(rows),
            "primitives": rows,
        }

    def add_claim(
        self,
        text: str,
        *,
        claim_type: str,
        support_state: str,
        risk: str,
        evidence_keys: list[str],
        body: dict[str, Any],
    ) -> str:
        evidence = [self.source_ids[key] for key in evidence_keys if key in self.source_ids]
        artifact_id = self.add_artifact(
            "claim",
            text,
            {
                "text": text,
                "claim_type": claim_type,
                "evidence": evidence,
                "objections": [],
                "used_in": [],
                "details": body,
            },
            support_state=normalize_support(support_state),
            risk=normalize_risk(risk),
            origin_type="tool_output",
        )
        self.claim_ids.append(artifact_id)
        for source_id in evidence:
            self.add_edge(source_id, artifact_id, "supports", {"claim_type": claim_type})
        return artifact_id

    def add_critique(
        self,
        *,
        title: str,
        target_artifact: str,
        severity: str,
        recommended_action: str,
        summary: str,
        source_keys: list[str],
    ) -> str:
        evidence = [self.source_ids[key] for key in source_keys if key in self.source_ids]
        artifact_id = self.add_artifact(
            "critique",
            title,
            {
                "target_artifact": target_artifact,
                "severity": severity,
                "summary": summary,
                "recommended_action": recommended_action,
                "status": "open",
                "evidence": evidence,
            },
            support_state="source_backed" if evidence else "inferred",
            risk="medium" if severity in {"major", "fatal"} else "low",
            origin_type="tool_output",
        )
        self.critique_ids.append(artifact_id)
        for source_id in evidence:
            self.add_edge(source_id, artifact_id, "supports", {"critique": title})
        return artifact_id

    def add_primitive(
        self,
        *,
        name: str,
        primitive_type: str,
        summary: str,
        trust_score: float,
        observed_in: list[str],
        known_failure_modes: list[str],
        extra: dict[str, Any] | None = None,
    ) -> str:
        status = "active" if trust_score >= 0.6 else "draft"
        body = {
            "name": name,
            "primitive_type": primitive_type,
            "summary": summary,
            "observed_in": observed_in,
            "trust_score": trust_score,
            "known_failure_modes": known_failure_modes,
            "recommended_uses": ["internal_governance", "self_hosting_growth_loop"],
        }
        if extra:
            body.update(extra)
        artifact_id = self.add_artifact(
            "primitive",
            name,
            body,
            status=status,
            support_state="source_backed" if observed_in else "inferred",
            risk="medium",
            origin_type="tool_output",
        )
        self.primitive_ids.append(artifact_id)
        for source_id in observed_in:
            self.add_edge(source_id, artifact_id, "generalizes_to", {"primitive": name})
        return artifact_id

    def add_feedback(
        self,
        *,
        title: str,
        target_title: str,
        source: str,
        result: str,
        interpretation: str,
        evidence_keys: list[str],
        metrics: dict[str, Any],
    ) -> str:
        target_id = self.find_artifact_by_title(target_title)
        evidence = [self.source_ids[key] for key in evidence_keys if key in self.source_ids]
        artifact_id = self.add_artifact(
            "feedback",
            title,
            {
                "target": target_id,
                "source": source,
                "result": result,
                "interpretation": interpretation,
                "evidence": evidence,
                "metrics": metrics,
                "updates": ["primitive_trust", "frontier_policy", "release_gates"],
            },
            support_state="empirical" if source == "benchmark" else "source_backed",
            risk="medium",
            origin_type="tool_output",
        )
        self.feedback_ids.append(artifact_id)
        if target_id:
            self.add_edge(target_id, artifact_id, "measured_by", {"feedback": result})
            self.add_edge(artifact_id, target_id, "improves", {"update_scope": "workflow_policy"})
        for source_id in evidence:
            self.add_edge(source_id, artifact_id, "supports", {"feedback": title})
        return artifact_id

    def add_artifact(
        self,
        artifact_type: str,
        title: str,
        body: dict[str, Any],
        *,
        status: str = "active",
        support_state: str = "inferred",
        risk: str = "medium",
        origin_type: str = "tool_output",
        created_by: str = "tool",
    ) -> str:
        artifact_id = stable_id("artifact", artifact_type, title)
        artifact = {
            "id": artifact_id,
            "type": artifact_type,
            "title": title,
            "body": body,
            "status": status,
            "version": "0.1.0",
            "provenance": {
                "origin_type": origin_type,
                "parent_artifacts": [],
                "source_spans": [],
                "event_history": [],
            },
            "created_by": created_by,
            "confidence": {
                "support_state": normalize_support(support_state),
                "risk": normalize_risk(risk),
                "rationale": "Compiled by Genesis Kernel from live local reports.",
            },
            "created_at": self.now,
            "updated_at": self.now,
        }
        existing = self.artifacts.get(artifact_id)
        if existing:
            existing.update(artifact)
        else:
            self.artifacts[artifact_id] = artifact
            event_id = self.add_event("artifact.created", artifact_id, f"Created {artifact_type}: {title}")
            self.artifacts[artifact_id]["provenance"]["event_history"].append(event_id)
        return artifact_id

    def add_edge(self, from_artifact: str, to_artifact: str, edge_type: str, metadata: dict[str, Any]) -> str:
        if not from_artifact or not to_artifact:
            return ""
        edge_id = stable_id("edge", from_artifact, edge_type, to_artifact, json.dumps(metadata, sort_keys=True))
        if any(row.get("id") == edge_id for row in self.edges):
            return edge_id
        self.edges.append(
            {
                "id": edge_id,
                "from_artifact": from_artifact,
                "to_artifact": to_artifact,
                "edge_type": edge_type,
                "metadata": metadata,
            }
        )
        self.add_event("edge.created", edge_id, f"{from_artifact} {edge_type} {to_artifact}")
        return edge_id

    def add_event(self, event_type: str, target_artifact: str, rationale: str) -> str:
        previous_hash = self.events[-1]["new_hash"] if self.events else "genesis"
        payload = {
            "project_id": "project_theseus",
            "actor_type": "tool",
            "actor_id": "scripts/genesis_kernel.py",
            "event_type": event_type,
            "target_artifact": target_artifact,
            "timestamp": self.now,
            "previous_hash": previous_hash,
            "rationale": rationale,
        }
        payload["new_hash"] = sha256_json(payload)
        payload["id"] = stable_id("event", event_type, target_artifact, payload["new_hash"][:12])
        self.events.append(payload)
        return payload["id"]

    def find_artifact_by_title(self, title: str) -> str:
        for artifact_id, artifact in self.artifacts.items():
            if artifact.get("title") == title:
                return artifact_id
        return ""


def check_report(path: Path) -> int:
    report = read_json(path)
    if not isinstance(report, dict) or report.get("policy") != "project_theseus_genesis_kernel_report_v0":
        print(json.dumps({"ok": False, "reason": "missing_or_invalid_report", "path": relative(path)}, indent=2))
        return 1
    failed_hard = [
        gate_row
        for gate_row in report.get("release_gates", [])
        if isinstance(gate_row, dict)
        and gate_row.get("severity") == "hard"
        and not gate_row.get("passed")
    ]
    ok = not failed_hard
    print(
        json.dumps(
            {
                "ok": ok,
                "trigger_state": get_path(report, ["summary", "trigger_state"], "UNKNOWN"),
                "failed_hard_gates": [row.get("gate") for row in failed_hard],
                "artifact_count": get_path(report, ["summary", "artifact_count"], 0),
                "bundle_dir": report.get("bundle_dir"),
            },
            indent=2,
        )
    )
    return 0 if ok else 1


def summarize_report(data: Any) -> dict[str, Any]:
    if isinstance(data, list):
        rows = [row for row in data if isinstance(row, dict)]
        frontiers = [row for row in rows if row.get("lifecycle") == "frontier"]
        frontiers.sort(key=lambda row: float(row.get("residual") or 0.0), reverse=True)
        return {
            "kind": "list",
            "count": len(data),
            "frontier_count": len(frontiers),
            "top_frontiers": [
                {
                    "benchmark_name": row.get("benchmark_name"),
                    "score": row.get("score"),
                    "floor": get_path(row, ["graduation_policy", "floor_threshold"], None),
                    "residual": row.get("residual"),
                    "wall_type": row.get("wall_type"),
                }
                for row in frontiers[:8]
            ],
        }
    if not isinstance(data, dict):
        return {"kind": type(data).__name__, "repr": str(data)[:500]}
    summary = {
        "kind": "dict",
        "policy": data.get("policy"),
        "created_utc": data.get("created_utc") or data.get("updated_utc"),
        "status": data.get("status") or data.get("trigger_state"),
        "summary": data.get("summary"),
        "decision": data.get("decision"),
        "next_action": data.get("next_action"),
        "promote": data.get("promote"),
        "passed": data.get("passed"),
        "total": data.get("total"),
    }
    if isinstance(data.get("checks"), list):
        summary["failed_checks"] = [
            row.get("gate") or row.get("name")
            for row in data.get("checks", [])
            if isinstance(row, dict) and not row.get("passed")
        ]
    if isinstance(data.get("gates"), list):
        summary["failed_gates"] = [
            row.get("gate")
            for row in data.get("gates", [])
            if isinstance(row, dict) and not row.get("passed")
        ]
    return {key: value for key, value in summary.items() if value not in (None, {}, [])}


def open_limitations(artifacts: dict[str, dict[str, Any]]) -> list[str]:
    limitations: list[str] = []
    for artifact in artifacts.values():
        if artifact.get("type") != "critique":
            continue
        body = artifact.get("body") or {}
        if body.get("status") == "open" and body.get("severity") in {"major", "fatal"}:
            limitations.append(str(artifact.get("title")))
    if not limitations:
        limitations.append("No major Genesis release limitations found in current local evidence.")
    return limitations[:20]


def failed_candidate_gates(candidate: dict[str, Any]) -> list[str]:
    return [
        str(row.get("gate"))
        for row in candidate.get("checks", [])
        if isinstance(row, dict) and not row.get("passed")
    ]


def next_best_action(state: dict[str, Any], debt: dict[str, Any], gates: list[dict[str, Any]]) -> str:
    failed_hard = [row.get("gate") for row in gates if row.get("severity") == "hard" and not row.get("passed")]
    if failed_hard:
        return f"Fix Genesis hard release gates first: {', '.join(str(item) for item in failed_hard)}."
    frontier = as_dict(state.get("frontier_policy"))
    family = frontier.get("frontier_family")
    card = frontier.get("pressure_card_id")
    candidate = as_dict(state.get("candidate_gate"))
    failed = failed_candidate_gates(candidate)
    if "active_frontier_clears_floor" in failed:
        return (
            f"Keep programming-first pressure on {family}/{card}, rotate within the coding family when stagnation policy triggers, "
            "and keep residuals/transfer artifacts in the Genesis bundle."
        )
    if debt["open_major_critiques"]:
        return "Resolve, waive, or disclose the open major critiques before treating this as a gold-master release."
    return "Use the Genesis release bundle as context for teacher self-edit, primitive extraction, and model-growth decisions."


def trigger_state(debt: dict[str, Any], gates: list[dict[str, Any]]) -> str:
    if any(row.get("severity") == "hard" and not row.get("passed") for row in gates):
        return "RED"
    if debt["unsupported_high_risk_claim_count"] or debt["open_major_critiques"]:
        return "YELLOW"
    if any(not row.get("passed") for row in gates):
        return "YELLOW"
    return "GREEN"


def claim_is_unsupported_high_risk(artifact: dict[str, Any]) -> bool:
    confidence = artifact.get("confidence") or {}
    support_state = str(confidence.get("support_state") or "unsupported")
    risk = str(confidence.get("risk") or "medium")
    return risk in {"high", "severe"} and support_state in {"unsupported", "contradicted"}


def gate(name: str, passed: bool, severity: str, evidence: str) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def normalize_support(value: str) -> str:
    value = str(value or "unsupported")
    return value if value in SUPPORT_STATES else "unsupported"


def normalize_risk(value: str) -> str:
    value = str(value or "medium")
    return value if value in RISKS else "medium"


def stable_id(*parts: str) -> str:
    prefix = slug("_".join(parts[:2]))[:48] or "artifact"
    digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:14]
    return f"{prefix}_{digest}"


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": report.get("policy"),
        "created_utc": report.get("created_utc"),
        "trigger_state": get_path(report, ["summary", "trigger_state"], None),
        "artifact_count": get_path(report, ["summary", "artifact_count"], None),
        "claim_count": get_path(report, ["summary", "claim_count"], None),
        "critique_count": get_path(report, ["summary", "critique_count"], None),
        "primitive_candidate_count": get_path(report, ["summary", "primitive_candidate_count"], None),
        "artifact_debt_score": get_path(report, ["summary", "artifact_debt_score"], None),
        "next_best_action": report.get("next_best_action"),
        "bundle_dir": report.get("bundle_dir"),
    }


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
