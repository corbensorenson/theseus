#!/usr/bin/env python3
"""Same-seed private-to-public transfer proof for Theseus Code LM.

This script does not run public benchmarks. It compares decoder gate metadata
before and after a private closure so public calibration only unlocks when the
receiver candidate manifest actually improves: more learned-token coverage,
fewer no-admissible tasks, clean provenance, and no quality regression.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_BASELINE = REPORTS / "private_public_transfer_baseline_before_fresh_chain.json"
DEFAULT_CURRENT = REPORTS / "decoder_v2_private_ablation_gate.json"
DEFAULT_OUT = REPORTS / "private_public_transfer_proof.json"
DEFAULT_MARKDOWN = REPORTS / "private_public_transfer_proof.md"
DEFAULT_RESIDUAL_PACKET = REPORTS / "private_public_transfer_residual_packet.json"
DEFAULT_RESIDUAL_MARKDOWN = REPORTS / "private_public_transfer_residual_packet.md"
DEFAULT_SAME_SURFACE_PROOF = REPORTS / "code_lm_same_surface_repair_proof_train_once_v1.json"
DEFAULT_PRIVATE_RECEIVER_EVIDENCE = REPORTS / "eligible_receiver_inventory_router_v1_private_ablation32.json"
DEFAULT_PRIVATE_BRIDGE_SHADOW_EVIDENCE = REPORTS / "private_to_public_receiver_inventory_bridge_shadow_ablation32.json"
DEFAULT_BROAD_TRANSFER_RESIDUAL_ABLATION = REPORTS / "broad_transfer_residual_decoder_ablation.json"
MIN_ACTUAL_COVERAGE_LIFT = 0.05
MIN_ELIGIBLE_COVERAGE_LIFT = 0.03
MIN_NO_ADMISSIBLE_SHRINK = 0.05
MIN_PROGRAM_LOOP_COVERAGE = 0.60
MIN_PROGRAM_PROMOTION_READY_RATE = 0.50
MIN_CONTRACT_GUIDED_CANDIDATE_LIFT = 8
MIN_STS_CONDITIONED_CANDIDATE_LIFT = 4
MIN_ACTUAL_TOKEN_CANDIDATE_LIFT = 8
MAX_SPECIALIZED_BUCKET_REGRESSION_RATE = 0.05
MAX_BRIDGED_PUBLIC_NO_ADMISSIBLE_RATE = 0.125


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE.relative_to(ROOT)))
    parser.add_argument("--current", default=str(DEFAULT_CURRENT.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--residual-packet-out", default=str(DEFAULT_RESIDUAL_PACKET.relative_to(ROOT)))
    parser.add_argument("--residual-markdown-out", default=str(DEFAULT_RESIDUAL_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--same-surface-proof", default=str(DEFAULT_SAME_SURFACE_PROOF.relative_to(ROOT)))
    parser.add_argument("--private-receiver-evidence", default=str(DEFAULT_PRIVATE_RECEIVER_EVIDENCE.relative_to(ROOT)))
    parser.add_argument("--private-bridge-shadow-evidence", default=str(DEFAULT_PRIVATE_BRIDGE_SHADOW_EVIDENCE.relative_to(ROOT)))
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    baseline_path = resolve(args.baseline)
    current_path = resolve(args.current)
    out_path = resolve(args.out)
    markdown_path = resolve(args.markdown_out)
    residual_packet_path = resolve(args.residual_packet_out)
    residual_markdown_path = resolve(args.residual_markdown_out)
    same_surface_proof_path = resolve(args.same_surface_proof)
    private_receiver_evidence_path = resolve(args.private_receiver_evidence)
    private_bridge_shadow_evidence_path = resolve(args.private_bridge_shadow_evidence)
    current_gate = read_json(current_path, {})
    current_snapshot = snapshot_from_gate(current_gate, current_path)
    same_surface_proof = read_json(same_surface_proof_path, {})
    same_surface_ready = same_surface_transfer_ready(same_surface_proof)
    private_receiver_evidence_path, private_receiver_evidence = read_evidence_with_default_fallback(
        private_receiver_evidence_path,
        default_path=DEFAULT_PRIVATE_RECEIVER_EVIDENCE,
        fallback_path=DEFAULT_BROAD_TRANSFER_RESIDUAL_ABLATION,
    )
    private_receiver_ready = private_receiver_inventory_ready(private_receiver_evidence)
    private_bridge_shadow_evidence_path, private_bridge_shadow_evidence = read_evidence_with_default_fallback(
        private_bridge_shadow_evidence_path,
        default_path=DEFAULT_PRIVATE_BRIDGE_SHADOW_EVIDENCE,
        fallback_path=DEFAULT_BROAD_TRANSFER_RESIDUAL_ABLATION,
    )
    private_bridge_shadow_ready = private_bridge_shadow_ready_for_transfer(private_bridge_shadow_evidence)

    baseline_payload = read_json(baseline_path, {})
    if args.write_baseline or not baseline_payload:
        baseline_payload = {
            "policy": "project_theseus_private_public_transfer_baseline_v1",
            "created_utc": now(),
            "baseline_source": rel_or_abs(current_path),
            "snapshot": current_snapshot,
            "rule": "capture the stale/old receiver state before a fresh private closure so the next run must prove delta",
            "external_inference_calls": 0,
        }
        write_json(baseline_path, baseline_payload)

    baseline_snapshot = object_field(baseline_payload, "snapshot")
    deltas = compare_snapshots(baseline_snapshot, current_snapshot)
    same_receiver_surface_ready = same_receiver_surface(baseline_snapshot, current_snapshot)
    baseline_public_task_count = int(baseline_snapshot.get("public_task_count") or 0)
    current_public_task_count = int(current_snapshot.get("public_task_count") or 0)
    no_admissible_comparable_public_surface = (
        same_receiver_surface_ready
        and baseline_public_task_count > 0
        and current_public_task_count > 0
    )
    actual_coverage_saturated = saturated_non_regressive(
        baseline_snapshot.get("public_actual_token_task_coverage"),
        current_snapshot.get("public_actual_token_task_coverage"),
        target=1.0,
    )
    no_admissible_saturated = floor_non_regressive(
        baseline_snapshot.get("public_no_admissible_task_rate"),
        current_snapshot.get("public_no_admissible_task_rate"),
        floor=0.0,
    )
    bridged_no_admissible_surface_ready = (
        not no_admissible_comparable_public_surface
        and baseline_public_task_count == 0
        and current_public_task_count > 0
        and same_surface_ready
        and private_receiver_ready
        and private_bridge_shadow_ready
        and number(current_snapshot.get("public_no_admissible_task_rate"))
        <= MAX_BRIDGED_PUBLIC_NO_ADMISSIBLE_RATE
    )
    no_admissible_transfer_ready = (
        (
            no_admissible_comparable_public_surface
            and (
                deltas["public_no_admissible_task_rate_delta"] <= -MIN_NO_ADMISSIBLE_SHRINK
                or no_admissible_saturated
            )
        )
        or bridged_no_admissible_surface_ready
    )
    actual_token_candidate_inventory_lift = (
        deltas["actual_token_candidate_count_delta"] >= MIN_ACTUAL_TOKEN_CANDIDATE_LIFT
    )
    specialized_bucket_substitution_ready = (
        actual_token_candidate_inventory_lift
        and (
            deltas["public_actual_token_task_coverage_delta"] >= MIN_ACTUAL_COVERAGE_LIFT
            or actual_coverage_saturated
        )
        and no_admissible_transfer_ready
        and current_snapshot.get("public_candidate_quality_gate_pass_rate", 0.0)
        >= max(0.90, baseline_snapshot.get("public_candidate_quality_gate_pass_rate", 0.0))
    )
    contract_guided_inventory_ready = (
        deltas["contract_guided_candidate_count_delta"] >= MIN_CONTRACT_GUIDED_CANDIDATE_LIFT
        or current_snapshot.get("contract_guided_candidate_count", 0.0)
        >= baseline_snapshot.get("contract_guided_candidate_count", 0.0) + MIN_CONTRACT_GUIDED_CANDIDATE_LIFT
        or (
            specialized_bucket_substitution_ready
            and bounded_count_regression(
                baseline_snapshot.get("contract_guided_candidate_count"),
                current_snapshot.get("contract_guided_candidate_count"),
            )
        )
    )
    sts_conditioned_inventory_ready = (
        deltas["sts_conditioned_candidate_count_delta"] >= MIN_STS_CONDITIONED_CANDIDATE_LIFT
        or current_snapshot.get("sts_conditioned_candidate_count", 0.0)
        >= baseline_snapshot.get("sts_conditioned_candidate_count", 0.0) + MIN_STS_CONDITIONED_CANDIDATE_LIFT
        or (
            specialized_bucket_substitution_ready
            and bounded_count_regression(
                baseline_snapshot.get("sts_conditioned_candidate_count"),
                current_snapshot.get("sts_conditioned_candidate_count"),
            )
        )
    )
    gates = [
        gate("baseline_present", bool(baseline_snapshot), rel_or_abs(baseline_path)),
        gate(
            "current_gate_present",
            bool(current_gate),
            {"current": rel_or_abs(current_path), "trigger_state": current_gate.get("trigger_state")},
        ),
        gate(
            "current_gate_train_once_wrapper_current",
            current_snapshot.get("train_once_wrapper_current_when_applicable") is not False,
            {
                "train_once_wrapper_current_when_applicable": current_snapshot.get(
                    "train_once_wrapper_current_when_applicable"
                ),
                "train_once_wrapper_run_status": current_snapshot.get("train_once_wrapper_run_status"),
                "train_once_private_inputs_fresh": current_snapshot.get("train_once_private_inputs_fresh"),
                "rule": (
                    "if the current decoder gate selected a train-once closure, the canonical wrapper must "
                    "not mark that closure stale against private training inputs"
                ),
            },
        ),
        gate(
            "current_newer_than_baseline",
            current_snapshot.get("source_gate_mtime", 0.0) >= baseline_snapshot.get("source_gate_mtime", 0.0),
            {"baseline_mtime": baseline_snapshot.get("source_gate_mtime"), "current_mtime": current_snapshot.get("source_gate_mtime")},
        ),
        gate(
            "current_gate_public_surface_present",
            not bool(current_snapshot.get("latest_closure_private_only"))
            and int(current_snapshot.get("public_task_count") or 0) > 0
            and int(current_snapshot.get("public_candidate_count") or 0) > 0,
            {
                "latest_closure": current_snapshot.get("latest_closure"),
                "latest_closure_private_only": bool(current_snapshot.get("latest_closure_private_only")),
                "public_task_count": current_snapshot.get("public_task_count"),
                "public_candidate_count": current_snapshot.get("public_candidate_count"),
                "rule": (
                    "private-only closures can prove private decoder progress, but they are not "
                    "private-to-public transfer attempts and cannot unlock public calibration"
                ),
            },
        ),
        gate(
            "same_receiver_task_count_or_manifest",
            same_receiver_surface_ready or same_surface_ready,
            {
                "baseline_public_task_count": baseline_snapshot.get("public_task_count"),
                "current_public_task_count": current_snapshot.get("public_task_count"),
                "baseline_latest_closure": baseline_snapshot.get("latest_closure"),
                "current_latest_closure": current_snapshot.get("latest_closure"),
                "same_surface_proof": rel_or_abs(same_surface_proof_path)
                if same_surface_proof_path.exists()
                else None,
                "same_surface_proof_ready": same_surface_ready,
                "same_surface_rule": "a GREEN same-surface repair proof may satisfy receiver-surface compatibility when the architecture baseline intentionally used a wider historical receiver",
            },
        ),
        gate(
            "public_actual_coverage_lift",
            deltas["public_actual_token_task_coverage_delta"] >= MIN_ACTUAL_COVERAGE_LIFT
            or actual_coverage_saturated,
            {
                "delta": deltas["public_actual_token_task_coverage_delta"],
                "minimum": MIN_ACTUAL_COVERAGE_LIFT,
                "baseline": baseline_snapshot.get("public_actual_token_task_coverage"),
                "current": current_snapshot.get("public_actual_token_task_coverage"),
                "saturated_non_regressive": actual_coverage_saturated,
                "rule": "lift is required unless the baseline was already saturated and the current receiver did not regress",
            },
        ),
        gate(
            "public_eligible_coverage_lift",
            deltas["public_eligible_task_coverage_delta"] >= MIN_ELIGIBLE_COVERAGE_LIFT,
            {
                "delta": deltas["public_eligible_task_coverage_delta"],
                "minimum": MIN_ELIGIBLE_COVERAGE_LIFT,
                "baseline": baseline_snapshot.get("public_eligible_task_coverage"),
                "current": current_snapshot.get("public_eligible_task_coverage"),
            },
        ),
        gate(
            "public_no_admissible_shrunk",
            no_admissible_transfer_ready,
            {
                "delta": deltas["public_no_admissible_task_rate_delta"],
                "maximum": -MIN_NO_ADMISSIBLE_SHRINK,
                "baseline": baseline_snapshot.get("public_no_admissible_task_rate"),
                "current": current_snapshot.get("public_no_admissible_task_rate"),
                "baseline_public_task_count": baseline_public_task_count,
                "current_public_task_count": current_public_task_count,
                "comparable_public_surface": no_admissible_comparable_public_surface,
                "saturated_non_regressive": no_admissible_saturated,
                "bridged_same_surface_ready": bridged_no_admissible_surface_ready,
                "max_bridged_public_no_admissible_rate": MAX_BRIDGED_PUBLIC_NO_ADMISSIBLE_RATE,
                "rule": (
                    "shrinkage is required for comparable public surfaces. If the baseline had no public "
                    "surface, a GREEN same-surface private receiver proof plus private bridge-shadow evidence "
                    "may satisfy the gate only while the current public no-admissible rate remains bounded."
                ),
            },
        ),
        gate(
            "provenance_quality_non_regressive",
            current_snapshot.get("public_candidate_quality_gate_pass_rate", 0.0)
            >= max(0.90, baseline_snapshot.get("public_candidate_quality_gate_pass_rate", 0.0)),
            {
                "baseline": baseline_snapshot.get("public_candidate_quality_gate_pass_rate"),
                "current": current_snapshot.get("public_candidate_quality_gate_pass_rate"),
                "minimum": max(0.90, baseline_snapshot.get("public_candidate_quality_gate_pass_rate", 0.0)),
                "rule": "current promotion-candidate quality must preserve or beat the baseline; tolerance does not unlock public calibration",
            },
        ),
        gate(
            "program_synthesis_loop_evidence_present",
            current_snapshot.get("public_program_synthesis_loop_present_rate", 0.0) >= MIN_PROGRAM_LOOP_COVERAGE
            and current_snapshot.get("public_program_synthesis_promotion_ready_rate", 0.0) >= MIN_PROGRAM_PROMOTION_READY_RATE,
            {
                "loop_present_rate": current_snapshot.get("public_program_synthesis_loop_present_rate"),
                "minimum_loop_present_rate": MIN_PROGRAM_LOOP_COVERAGE,
                "promotion_ready_rate": current_snapshot.get("public_program_synthesis_promotion_ready_rate"),
                "minimum_promotion_ready_rate": MIN_PROGRAM_PROMOTION_READY_RATE,
                "rule": "public transfer proof requires the learned candidate inventory to carry the real contract-IR/AST-plan/decode/verifier/ranker loop",
            },
        ),
        gate(
            "contract_guided_candidate_inventory_lift",
            contract_guided_inventory_ready,
            {
                "delta": deltas["contract_guided_candidate_count_delta"],
                "minimum": MIN_CONTRACT_GUIDED_CANDIDATE_LIFT,
                "baseline": baseline_snapshot.get("contract_guided_candidate_count"),
                "current": current_snapshot.get("contract_guided_candidate_count"),
                "actual_token_candidate_count_delta": deltas["actual_token_candidate_count_delta"],
                "actual_token_candidate_lift_minimum": MIN_ACTUAL_TOKEN_CANDIDATE_LIFT,
                "specialized_bucket_substitution_ready": specialized_bucket_substitution_ready,
                "max_specialized_bucket_regression_rate": MAX_SPECIALIZED_BUCKET_REGRESSION_RATE,
                "rule": (
                    "older specialized buckets may stay flat or regress slightly when a newer learned-token "
                    "bucket lifts total inventory, coverage, no-admissible rate, and quality on the same surface"
                ),
            },
        ),
        gate(
            "sts_conditioned_candidate_inventory_lift",
            sts_conditioned_inventory_ready,
            {
                "delta": deltas["sts_conditioned_candidate_count_delta"],
                "minimum": MIN_STS_CONDITIONED_CANDIDATE_LIFT,
                "baseline": baseline_snapshot.get("sts_conditioned_candidate_count"),
                "current": current_snapshot.get("sts_conditioned_candidate_count"),
                "actual_token_candidate_count_delta": deltas["actual_token_candidate_count_delta"],
                "actual_token_candidate_lift_minimum": MIN_ACTUAL_TOKEN_CANDIDATE_LIFT,
                "specialized_bucket_substitution_ready": specialized_bucket_substitution_ready,
                "max_specialized_bucket_regression_rate": MAX_SPECIALIZED_BUCKET_REGRESSION_RATE,
                "rule": (
                    "STS must alter the decoder candidate distribution before public calibration can unlock; "
                    "a newer STS-conditioned learned-token bucket may substitute for a narrow historical STS "
                    "bucket when total learned-token inventory and receiver coverage lift under the same seed"
                ),
            },
        ),
        gate(
            "decoder_gate_ready",
            bool(current_gate.get("ready_for_public_calibration")),
            {"ready_for_public_calibration": current_gate.get("ready_for_public_calibration")},
        ),
        gate(
            "private_receiver_inventory_ready",
            private_receiver_ready,
            {
                "private_receiver_evidence": rel_or_abs(private_receiver_evidence_path)
                if private_receiver_evidence_path.exists()
                else None,
                "trigger_state": private_receiver_evidence.get("trigger_state"),
                "status": private_receiver_evidence.get("status"),
                "private_receiver_eligible_task_rate_delta": object_field(private_receiver_evidence, "delta").get(
                    "private_receiver_eligible_task_rate_delta"
                ),
                "no_admissible_rate_delta": object_field(private_receiver_evidence, "delta").get(
                    "no_admissible_rate_delta"
                ),
                "rule": "public transfer readiness requires same-seed private receiver evidence to be GREEN; YELLOW private ablations remain diagnostic until their failed gates are repaired",
            },
        ),
        gate(
            "private_to_public_bridge_shadow_ready",
            private_bridge_shadow_ready,
            {
                "private_bridge_shadow_evidence": rel_or_abs(private_bridge_shadow_evidence_path)
                if private_bridge_shadow_evidence_path.exists()
                else None,
                "trigger_state": private_bridge_shadow_evidence.get("trigger_state"),
                "status": private_bridge_shadow_evidence.get("status"),
                "bridge_shadow_task_delta": object_field(private_bridge_shadow_evidence, "delta").get(
                    "private_to_public_receiver_bridge_shadow_task_count_delta"
                ),
                "semantic_lift": object_field(private_bridge_shadow_evidence, "delta").get(
                    "semantic_test_passed_task_rate_delta"
                ),
                "public_task_count": object_field(private_bridge_shadow_evidence, "manifest").get(
                    "public_task_count"
                ),
                "rule": (
                    "the private-to-public receiver bridge must be exercised on private heldout "
                    "with empty public manifests before a public metadata fanout can claim bridge readiness"
                ),
            },
        ),
    ]
    ready = all(row["passed"] for row in gates)
    report = {
        "policy": "project_theseus_private_public_transfer_proof_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if ready else "YELLOW",
        "ready_for_public_calibration": ready,
        "summary": {
            "ready_for_public_calibration": ready,
            "baseline_path": rel_or_abs(baseline_path),
            "current_gate_path": rel_or_abs(current_path),
            "public_actual_token_task_coverage_delta": deltas["public_actual_token_task_coverage_delta"],
            "public_eligible_task_coverage_delta": deltas["public_eligible_task_coverage_delta"],
            "public_no_admissible_task_rate_delta": deltas["public_no_admissible_task_rate_delta"],
            "public_actual_token_task_coverage_saturated_non_regressive": actual_coverage_saturated,
            "public_no_admissible_task_rate_saturated_non_regressive": no_admissible_saturated,
            "public_no_admissible_comparable_public_surface": no_admissible_comparable_public_surface,
            "public_no_admissible_bridged_same_surface_ready": bridged_no_admissible_surface_ready,
            "public_program_synthesis_loop_present_rate": current_snapshot.get("public_program_synthesis_loop_present_rate"),
            "public_program_synthesis_promotion_ready_rate": current_snapshot.get("public_program_synthesis_promotion_ready_rate"),
            "actual_token_candidate_count_delta": deltas["actual_token_candidate_count_delta"],
            "actual_token_candidate_inventory_lift": actual_token_candidate_inventory_lift,
            "specialized_bucket_substitution_ready": specialized_bucket_substitution_ready,
            "contract_guided_candidate_count_delta": deltas["contract_guided_candidate_count_delta"],
            "sts_conditioned_candidate_count_delta": deltas["sts_conditioned_candidate_count_delta"],
            "current_decoder_gate_ready": bool(current_gate.get("ready_for_public_calibration")),
            "current_gate_private_only": bool(current_snapshot.get("latest_closure_private_only")),
            "current_gate_train_once_wrapper_current": current_snapshot.get(
                "train_once_wrapper_current_when_applicable"
            ),
            "same_surface_proof_ready": same_surface_ready,
            "private_receiver_inventory_ready": private_receiver_ready,
            "private_to_public_bridge_shadow_ready": private_bridge_shadow_ready,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "baseline": baseline_snapshot,
        "current": current_snapshot,
        "same_surface_repair_proof": same_surface_proof if same_surface_proof else {},
        "private_receiver_inventory_evidence": private_receiver_evidence if private_receiver_evidence else {},
        "private_to_public_bridge_shadow_evidence": private_bridge_shadow_evidence
        if private_bridge_shadow_evidence
        else {},
        "deltas": deltas,
        "gates": gates,
        "rules": {
            "public_benchmarks": "not executed here; this compares receiver metadata only",
            "graduation": "private pass rate alone does not count; receiver coverage/no-admissible gates must move or prove saturated non-regression, and candidate inventory must still lift",
            "contamination": "no public tests or solutions are read or emitted",
        },
        "next_actions": next_actions(ready, gates, private_receiver_ready, current_gate),
        "external_inference_calls": 0,
    }
    if not ready:
        residual_packet = build_residual_packet(
            report=report,
            private_receiver_evidence=private_receiver_evidence,
            private_bridge_shadow_evidence=private_bridge_shadow_evidence,
            residual_packet_path=residual_packet_path,
            residual_markdown_path=residual_markdown_path,
        )
        report["residual_packet"] = {
            "path": rel_or_abs(residual_packet_path),
            "markdown": rel_or_abs(residual_markdown_path),
            "trigger_state": residual_packet.get("trigger_state"),
            "next_source_patch_id": residual_packet.get("summary", {}).get("next_source_patch_id"),
        }
    else:
        residual_packet = build_cleared_residual_packet(
            report=report,
            private_receiver_evidence=private_receiver_evidence,
            private_bridge_shadow_evidence=private_bridge_shadow_evidence,
            residual_packet_path=residual_packet_path,
            residual_markdown_path=residual_markdown_path,
        )
        report["residual_packet"] = {
            "path": rel_or_abs(residual_packet_path),
            "markdown": rel_or_abs(residual_markdown_path),
            "trigger_state": residual_packet.get("trigger_state"),
            "next_source_patch_id": residual_packet.get("summary", {}).get("next_source_patch_id"),
        }
    write_json(out_path, report)
    write_text(markdown_path, render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0


def read_evidence_with_default_fallback(
    path: Path,
    *,
    default_path: Path,
    fallback_path: Path,
) -> tuple[Path, dict[str, Any]]:
    evidence = read_json(path, {})
    if evidence or path.resolve() != default_path.resolve():
        return path, evidence
    fallback = read_json(fallback_path, {})
    if fallback:
        return fallback_path, fallback
    return path, evidence


def snapshot_from_gate(gate_report: dict[str, Any], source_path: Path) -> dict[str, Any]:
    summary = object_field(gate_report, "summary")
    public_provenance = object_field(gate_report, "public_candidate_provenance")
    public_no_admissible = object_field(gate_report, "public_no_admissible_diagnostics")
    return {
        "created_utc": now(),
        "source_gate_path": rel_or_abs(source_path),
        "source_gate_mtime": path_mtime(source_path),
        "trigger_state": gate_report.get("trigger_state"),
        "ready_for_public_calibration": bool(gate_report.get("ready_for_public_calibration")),
        "latest_closure": summary.get("latest_closure"),
        "latest_closure_private_only": bool(summary.get("latest_closure_private_only")),
        "train_once_wrapper_current_when_applicable": summary.get("train_once_wrapper_current_when_applicable"),
        "train_once_wrapper_run_status": get_path(
            gate_report, ["summary", "train_once_wrapper_freshness", "train_once_run_status"]
        ),
        "train_once_private_inputs_fresh": get_path(
            gate_report, ["summary", "train_once_wrapper_freshness", "private_inputs_fresh"]
        ),
        "decoder_relevant_source_fingerprint": summary.get("decoder_relevant_source_fingerprint"),
        "private_candidate_count": number(summary.get("private_candidate_count")),
        "public_candidate_count": number(summary.get("public_candidate_count")),
        "private_closure_pass_delta": number(summary.get("private_closure_pass_delta")),
        "public_task_count": number(public_no_admissible.get("task_count")),
        "public_actual_token_task_coverage": number(summary.get("public_actual_token_task_coverage")),
        "public_eligible_task_coverage": number(summary.get("public_eligible_task_coverage")),
        "public_no_admissible_task_rate": number(summary.get("public_no_admissible_task_rate")),
        "public_candidate_provenance_present_rate": number(summary.get("public_candidate_provenance_present_rate")),
        "public_candidate_quality_gate_pass_rate": number(summary.get("public_candidate_quality_gate_pass_rate")),
        "public_candidate_ast_valid_rate": number(summary.get("public_candidate_ast_valid_rate")),
        "public_candidate_entry_point_match_rate": number(summary.get("public_candidate_entry_point_match_rate")),
        "public_candidate_bogus_return_local_callable_count": number(summary.get("public_candidate_bogus_return_local_callable_count")),
        "public_program_synthesis_loop_present_rate": number(
            summary.get("public_program_synthesis_loop_present_rate")
            if summary.get("public_program_synthesis_loop_present_rate") is not None
            else public_provenance.get("program_synthesis_loop_present_rate")
        ),
        "public_program_synthesis_contract_ir_rate": number(
            summary.get("public_program_synthesis_contract_ir_rate")
            if summary.get("public_program_synthesis_contract_ir_rate") is not None
            else public_provenance.get("program_synthesis_contract_ir_rate")
        ),
        "public_program_synthesis_ast_plan_rate": number(
            summary.get("public_program_synthesis_ast_plan_rate")
            if summary.get("public_program_synthesis_ast_plan_rate") is not None
            else public_provenance.get("program_synthesis_ast_plan_rate")
        ),
        "public_program_synthesis_parser_mask_rate": number(
            summary.get("public_program_synthesis_parser_mask_rate")
            if summary.get("public_program_synthesis_parser_mask_rate") is not None
            else public_provenance.get("program_synthesis_parser_mask_rate")
        ),
        "public_program_synthesis_ranker_rate": number(
            summary.get("public_program_synthesis_ranker_rate")
            if summary.get("public_program_synthesis_ranker_rate") is not None
            else public_provenance.get("program_synthesis_ranker_rate")
        ),
        "public_program_synthesis_promotion_ready_rate": number(
            summary.get("public_program_synthesis_promotion_ready_rate")
            if summary.get("public_program_synthesis_promotion_ready_rate") is not None
            else public_provenance.get("program_synthesis_promotion_ready_rate")
        ),
        "contract_guided_candidate_count": number(summary.get("contract_guided_candidate_count")),
        "contract_guided_verifier_pass_rate": number(summary.get("contract_guided_verifier_pass_rate")),
        "sts_conditioned_candidate_count": number(summary.get("sts_conditioned_candidate_count")),
        "sts_conditioned_verifier_pass_rate": number(summary.get("sts_conditioned_verifier_pass_rate")),
        "public_no_admissible_top_reasons": summary.get("public_no_admissible_top_reasons") or {},
        "public_no_admissible_top_missing_capability_families": summary.get("public_no_admissible_top_missing_capability_families") or {},
        "actual_token_candidate_count": number(public_provenance.get("actual_token_candidate_count")),
        "quality_gate_pass_count": number(public_provenance.get("quality_gate_pass_count")),
    }


def compare_snapshots(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, float]:
    keys = [
        "public_actual_token_task_coverage",
        "public_eligible_task_coverage",
        "public_no_admissible_task_rate",
        "public_candidate_quality_gate_pass_rate",
        "public_candidate_ast_valid_rate",
        "public_candidate_entry_point_match_rate",
        "contract_guided_verifier_pass_rate",
        "sts_conditioned_verifier_pass_rate",
        "public_program_synthesis_loop_present_rate",
        "public_program_synthesis_promotion_ready_rate",
        "actual_token_candidate_count",
        "contract_guided_candidate_count",
        "sts_conditioned_candidate_count",
    ]
    return {f"{key}_delta": round(float(current.get(key) or 0.0) - float(baseline.get(key) or 0.0), 6) for key in keys}


def bounded_count_regression(
    baseline_value: Any,
    current_value: Any,
    *,
    max_regression_rate: float = MAX_SPECIALIZED_BUCKET_REGRESSION_RATE,
) -> bool:
    baseline = float(baseline_value or 0.0)
    current = float(current_value or 0.0)
    if baseline <= 0.0:
        return current >= baseline
    return current >= baseline * (1.0 - max(0.0, max_regression_rate))


def same_receiver_surface(baseline: dict[str, Any], current: dict[str, Any]) -> bool:
    baseline_tasks = int(baseline.get("public_task_count") or 0)
    current_tasks = int(current.get("public_task_count") or 0)
    if baseline_tasks and current_tasks and baseline_tasks == current_tasks:
        return True
    baseline_count = int(baseline.get("public_candidate_count") or 0)
    current_count = int(current.get("public_candidate_count") or 0)
    return bool(baseline_count and current_count and baseline_count == current_count)


def same_surface_transfer_ready(proof: dict[str, Any]) -> bool:
    if not isinstance(proof, dict) or not proof:
        return False
    summary = object_field(proof, "summary")
    rules = object_field(proof, "rules")
    gates = proof.get("gates")
    current_no_admissible_rate = summary.get("current_no_admissible_task_rate")
    if current_no_admissible_rate is None:
        current_no_admissible_rate = 1.0
    receiver_eligible_delta = max(
        float(summary.get("eligible_task_coverage_delta") or 0.0),
        float(summary.get("receiver_eligible_coverage_delta") or 0.0),
    )
    return (
        proof.get("trigger_state") == "GREEN"
        and bool(proof.get("ready_for_transfer_proof_receiver_surface"))
        and bool(summary.get("same_task_surface"))
        and float(summary.get("actual_token_task_coverage_delta") or 0.0) >= MIN_ACTUAL_COVERAGE_LIFT
        and receiver_eligible_delta >= MIN_ELIGIBLE_COVERAGE_LIFT
        and float(summary.get("no_admissible_task_rate_delta") or 0.0) <= -MIN_NO_ADMISSIBLE_SHRINK
        and float(current_no_admissible_rate) == 0.0
        and int(summary.get("current_verifier_failed_count") or 0) == 0
        and int(summary.get("current_guardrail_failed_count") or 0) == 0
        and proof.get("external_inference_calls") == 0
        and "public tests and solutions must not be visible" in str(rules.get("public_boundary") or "")
        and isinstance(gates, list)
        and all(bool(row.get("passed")) for row in gates if isinstance(row, dict))
    )


def private_receiver_inventory_ready(evidence: dict[str, Any]) -> bool:
    if not evidence:
        return False
    delta = object_field(evidence, "delta")
    manifest = object_field(evidence, "manifest")
    gates = evidence.get("gates") if isinstance(evidence.get("gates"), list) else []
    return (
        evidence.get("status") == "GREEN"
        and float(delta.get("private_receiver_eligible_task_rate_delta") or 0.0) >= MIN_ELIGIBLE_COVERAGE_LIFT
        and float(delta.get("no_admissible_rate_delta") or 0.0) <= 0.0
        and not bool(manifest.get("public_prompts_used"))
        and not bool(manifest.get("public_tests_used"))
        and not bool(manifest.get("public_solutions_used"))
        and all(bool(row.get("passed")) for row in gates if isinstance(row, dict))
    )


def private_bridge_shadow_ready_for_transfer(evidence: dict[str, Any]) -> bool:
    if not evidence:
        return False
    delta = object_field(evidence, "delta")
    manifest = object_field(evidence, "manifest")
    gates = evidence.get("gates") if isinstance(evidence.get("gates"), list) else []
    return (
        evidence.get("status") == "GREEN"
        and int(delta.get("private_to_public_receiver_bridge_shadow_task_count_delta") or 0) > 0
        and float(delta.get("semantic_test_passed_task_rate_delta") or 0.0) > 0.0
        and float(delta.get("passed_task_rate_delta") or 0.0) >= 0.0
        and float(delta.get("no_admissible_rate_delta") or 0.0) <= 0.0
        and int(manifest.get("public_task_count") or 0) == 0
        and not bool(manifest.get("public_prompts_used"))
        and not bool(manifest.get("public_tests_used"))
        and not bool(manifest.get("public_solutions_used"))
        and all(bool(row.get("passed")) for row in gates if isinstance(row, dict))
    )


def next_actions(
    ready: bool,
    gates: list[dict[str, Any]],
    private_receiver_ready: bool = False,
    current_gate: dict[str, Any] | None = None,
) -> list[str]:
    if ready:
        return [
            "Transfer proof is GREEN; public calibration may be considered only if the decoder gate is also GREEN.",
            "Run at most one bounded 4-card public calibration, then preserve/rollback based on receiver lift.",
        ]
    failed = {row["name"] for row in gates if not row["passed"]}
    if "current_gate_train_once_wrapper_current" in failed:
        return [
            "Current decoder evidence belongs to a train-once closure whose wrapper reports stale private training inputs.",
            "Run a fresh CUDA train-once private checkpoint before treating decoder or transfer metadata as current evidence.",
        ]
    if "current_gate_public_surface_present" in failed:
        return [
            "Current decoder evidence is private-only, so public transfer remains unattempted and locked; use it to patch the public metadata bridge only after the operator/public gate policy allows a bounded visible-metadata fanout.",
        ]
    decoder_failed = set()
    if isinstance(current_gate, dict):
        decoder_failed = {
            str(row.get("name"))
            for row in current_gate.get("gates", [])
            if isinstance(row, dict) and not bool(row.get("passed"))
        }
    if "decoder_gate_ready" in failed and decoder_failed == {"public_surface_scale_sufficient_for_calibration"}:
        return [
            "Keep public calibration locked until the visible calibration metadata surface reaches at least 32 tasks and 96 candidates.",
            "Build a broader metadata-only public surface, rerun train-once fanout, then rerun decoder_v2_private_ablation_gate and this transfer proof before any public benchmark scoring.",
        ]
    if "current_gate_present" in failed or "decoder_gate_ready" in failed:
        return ["Run the fresh private closure and decoder_v2_private_ablation_gate before any public benchmark work."]
    if "private_receiver_inventory_ready" in failed:
        return [
            "Keep public calibration locked; repair the private receiver ablation first so the same-seed private lift is GREEN without speed or integrity regressions.",
        ]
    if private_receiver_ready and "public_eligible_coverage_lift" in failed:
        return [
            "Private eligible receiver routing is proven; next patch should bridge that private receiver inventory into public metadata fanout using only visible prompt/signature contracts, with no public tests or solutions.",
        ]
    if "public_actual_coverage_lift" in failed or "public_eligible_coverage_lift" in failed:
        return ["Continue candidate coverage recovery; private gains have not increased public receiver inventory enough."]
    if "public_no_admissible_shrunk" in failed:
        return ["Target no-admissible residual families before rerunning public calibration."]
    if "program_synthesis_loop_evidence_present" in failed:
        return ["Keep public calibration locked; candidate rows must carry contract-IR, AST-plan, constrained-decode, verifier-repair, and ranker evidence before transfer can count."]
    if "contract_guided_candidate_inventory_lift" in failed:
        return ["Patch interface/return-shape contract-guided decoding until public receiver candidate inventory grows under the same seed and budget."]
    if "sts_conditioned_candidate_inventory_lift" in failed:
        return ["Patch STS-conditioned decoding so STS changes the candidate distribution, then rerun same-seed private transfer proof."]
    return ["Keep public calibration locked until private-to-public receiver metadata shows real delta."]


def build_residual_packet(
    *,
    report: dict[str, Any],
    private_receiver_evidence: dict[str, Any],
    private_bridge_shadow_evidence: dict[str, Any],
    residual_packet_path: Path,
    residual_markdown_path: Path,
) -> dict[str, Any]:
    failed = [row for row in report.get("gates", []) if isinstance(row, dict) and not row.get("passed")]
    failed_names = [str(row.get("name") or "") for row in failed]
    current = object_field(report, "current")
    deltas = object_field(report, "deltas")
    top_reasons = current.get("public_no_admissible_top_reasons") if isinstance(current.get("public_no_admissible_top_reasons"), dict) else {}
    missing_families = (
        current.get("public_no_admissible_top_missing_capability_families")
        if isinstance(current.get("public_no_admissible_top_missing_capability_families"), dict)
        else {}
    )
    private_receiver_ready = private_receiver_inventory_ready(private_receiver_evidence)
    private_receiver_delta = object_field(private_receiver_evidence, "delta")
    private_bridge_shadow_ready = private_bridge_shadow_ready_for_transfer(private_bridge_shadow_evidence)
    private_bridge_shadow_delta = object_field(private_bridge_shadow_evidence, "delta")
    if "public_eligible_coverage_lift" in failed_names:
        if private_receiver_ready:
            patch_id = "private_to_public_receiver_inventory_bridge_v1"
            mechanism = (
                "Bridge the now-proven private eligible receiver inventory into public metadata fanout without "
                "using public tests or solutions: reuse the private residual-family/router priors, infer visible "
                "argument roles and return-shape obligations from prompt/signature metadata only, and add the "
                "resulting receiver contracts to candidate ranking before public calibration can unlock."
            )
            target_files = [
                "crates/symliquid-cli/src/code_lm_closure/broad_transfer_residual_policy.rs",
                "crates/symliquid-cli/src/code_lm_closure/candidate_fanout/expression_pool.rs",
                "crates/symliquid-cli/src/code_lm_closure/candidate_fanout/task_rows.rs",
                "crates/symliquid-cli/src/code_lm_closure/contract_verifier/scoring.rs",
                "scripts/private_public_transfer_proof.py",
            ]
            private_eval = (
                "Keep eligible_receiver_inventory_router_v1 GREEN on private heldout, then require the transfer "
                "proof to show public eligible coverage lift >= 0.03 and no-admissible non-regression under the "
                "same no-public-tests/no-public-solutions boundary."
            )
        else:
            patch_id = "eligible_receiver_inventory_router_v1"
            mechanism = (
                "Increase eligible receiver inventory, not raw candidate count, by routing interface_fidelity "
                "and parsing/string-indexing residuals into contract-guided full-body candidates that satisfy "
                "visible arguments, return shape, required locals, loop/branch obligations, and parser mask "
                "before verifier scoring."
            )
            target_files = [
                "crates/symliquid-cli/src/code_lm_closure/broad_transfer_residual_policy.rs",
                "crates/symliquid-cli/src/code_lm_closure/candidate_fanout/expression_pool.rs",
                "crates/symliquid-cli/src/code_lm_closure/candidate_fanout/task_rows.rs",
                "crates/symliquid-cli/src/code_lm_closure/contract_verifier/scoring.rs",
            ]
            private_eval = (
                "Generate private heldout parser/interface-fidelity rows for locals+parsing, string_indexing, "
                "visible-argument inference, and str/list/bool return-shape obligations; require same-seed "
                "eligible coverage lift >= 0.03 and no-admissible non-regression before any public calibration."
            )
    elif "public_actual_coverage_lift" in failed_names or "public_no_admissible_shrunk" in failed_names:
        patch_id = "no_admissible_receiver_repair_router_v1"
        mechanism = (
            "Route no-admissible residuals into private retry candidates before public receiver metadata "
            "fanout, with explicit AST-valid full bodies and required construct checks."
        )
        target_files = [
            "crates/symliquid-cli/src/code_lm_closure/broad_transfer_residual_policy.rs",
            "crates/symliquid-cli/src/code_lm_closure/candidate_fanout/expression_pool.rs",
        ]
        private_eval = "Require private no-admissible shrinkage and current decoder gate GREEN before transfer proof."
    else:
        patch_id = "transfer_proof_gate_specific_decoder_patch_v1"
        mechanism = "Patch the failed transfer-proof gate directly, then rerun the same private closure/gate/proof chain."
        target_files = ["crates/symliquid-cli/src/code_lm_closure"]
        private_eval = "Use the failed transfer-proof gate as the private acceptance target."
    packet = {
        "policy": "project_theseus_private_public_transfer_residual_packet_v1",
        "created_utc": now(),
        "trigger_state": "GREEN",
        "summary": {
            "transfer_proof_ready": bool(report.get("ready_for_public_calibration")),
            "failed_transfer_gates": failed_names,
            "next_source_patch_id": patch_id,
            "next_source_patch_mechanism": mechanism,
            "private_eval_acceptance": private_eval,
            "private_receiver_inventory_ready": private_receiver_ready,
            "private_to_public_bridge_shadow_ready": private_bridge_shadow_ready,
        },
        "next_source_level_decoder_patch": {
            "id": patch_id,
            "target_files": target_files,
            "mechanism": mechanism,
            "private_eval_acceptance": private_eval,
            "rollback_rule": "Rollback or demote the patch if private eligible coverage lift is below 0.03, no-admissible regresses, or decoder_v2_private_ablation_gate stops being GREEN.",
        },
        "evidence": {
            "transfer_proof": rel_or_abs(resolve(report.get("summary", {}).get("current_gate_path", "reports/decoder_v2_private_ablation_gate.json"))),
            "public_actual_token_task_coverage_delta": deltas.get("public_actual_token_task_coverage_delta"),
            "public_eligible_task_coverage_delta": deltas.get("public_eligible_task_coverage_delta"),
            "public_no_admissible_task_rate_delta": deltas.get("public_no_admissible_task_rate_delta"),
            "actual_token_candidate_count_delta": deltas.get("actual_token_candidate_count_delta"),
            "contract_guided_candidate_count_delta": deltas.get("contract_guided_candidate_count_delta"),
            "sts_conditioned_candidate_count_delta": deltas.get("sts_conditioned_candidate_count_delta"),
            "top_no_admissible_reasons": top_reasons,
            "top_missing_capability_families": missing_families,
            "private_receiver_inventory_evidence_status": private_receiver_evidence.get("status"),
            "private_receiver_eligible_task_rate_delta": private_receiver_delta.get("private_receiver_eligible_task_rate_delta"),
            "private_receiver_no_admissible_rate_delta": private_receiver_delta.get("no_admissible_rate_delta"),
            "private_receiver_public_boundary_clean": private_receiver_ready,
            "private_to_public_bridge_shadow_status": private_bridge_shadow_evidence.get("status"),
            "private_to_public_bridge_shadow_task_delta": private_bridge_shadow_delta.get(
                "private_to_public_receiver_bridge_shadow_task_count_delta"
            ),
            "private_to_public_bridge_shadow_semantic_lift": private_bridge_shadow_delta.get(
                "semantic_test_passed_task_rate_delta"
            ),
            "private_to_public_bridge_shadow_public_boundary_clean": private_bridge_shadow_ready,
        },
        "safety": {
            "public_tests_or_solutions_used": False,
            "public_benchmark_training_allowed": False,
            "public_content_policy": "Only aggregate receiver metadata, residual families, and gate deltas are included; no public tests or solutions.",
            "external_inference_calls": 0,
        },
        "external_inference_calls": 0,
    }
    write_json(residual_packet_path, packet)
    write_text(residual_markdown_path, render_residual_packet_markdown(packet))
    return packet


def build_cleared_residual_packet(
    *,
    report: dict[str, Any],
    private_receiver_evidence: dict[str, Any],
    private_bridge_shadow_evidence: dict[str, Any],
    residual_packet_path: Path,
    residual_markdown_path: Path,
) -> dict[str, Any]:
    deltas = object_field(report, "deltas")
    private_receiver_delta = object_field(private_receiver_evidence, "delta")
    private_bridge_shadow_delta = object_field(private_bridge_shadow_evidence, "delta")
    packet = {
        "policy": "project_theseus_private_public_transfer_residual_packet_v1",
        "created_utc": now(),
        "trigger_state": "CLEARED",
        "summary": {
            "transfer_proof_ready": True,
            "failed_transfer_gates": [],
            "next_source_patch_id": "",
            "next_source_patch_mechanism": "No transfer-proof decoder patch is active; all transfer-proof gates passed.",
            "private_eval_acceptance": "Keep decoder_v2_private_ablation_gate and private_public_transfer_proof GREEN; do not public-calibrate without operator approval.",
            "private_receiver_inventory_ready": private_receiver_inventory_ready(private_receiver_evidence),
            "private_to_public_bridge_shadow_ready": private_bridge_shadow_ready_for_transfer(private_bridge_shadow_evidence),
        },
        "next_source_level_decoder_patch": {
            "id": "",
            "target_files": [],
            "mechanism": "No active patch required by private_public_transfer_proof.",
            "private_eval_acceptance": "Transfer proof is GREEN.",
            "rollback_rule": "If a future transfer-proof gate fails, regenerate this packet with the failed gate as acceptance target.",
        },
        "evidence": {
            "transfer_proof": rel_or_abs(resolve(report.get("summary", {}).get("current_gate_path", "reports/decoder_v2_private_ablation_gate.json"))),
            "public_actual_token_task_coverage_delta": deltas.get("public_actual_token_task_coverage_delta"),
            "public_eligible_task_coverage_delta": deltas.get("public_eligible_task_coverage_delta"),
            "public_no_admissible_task_rate_delta": deltas.get("public_no_admissible_task_rate_delta"),
            "actual_token_candidate_count_delta": deltas.get("actual_token_candidate_count_delta"),
            "contract_guided_candidate_count_delta": deltas.get("contract_guided_candidate_count_delta"),
            "sts_conditioned_candidate_count_delta": deltas.get("sts_conditioned_candidate_count_delta"),
            "private_receiver_eligible_task_rate_delta": private_receiver_delta.get("private_receiver_eligible_task_rate_delta"),
            "private_receiver_no_admissible_rate_delta": private_receiver_delta.get("no_admissible_rate_delta"),
            "private_receiver_public_boundary_clean": private_receiver_inventory_ready(private_receiver_evidence),
            "private_to_public_bridge_shadow_task_delta": private_bridge_shadow_delta.get(
                "private_to_public_receiver_bridge_shadow_task_count_delta"
            ),
            "private_to_public_bridge_shadow_semantic_lift": private_bridge_shadow_delta.get(
                "semantic_test_passed_task_rate_delta"
            ),
            "private_to_public_bridge_shadow_public_boundary_clean": private_bridge_shadow_ready_for_transfer(
                private_bridge_shadow_evidence
            ),
        },
        "safety": {
            "public_tests_or_solutions_used": False,
            "public_benchmark_training_allowed": False,
            "public_content_policy": "Only aggregate receiver metadata, residual families, and gate deltas are included; no public tests or solutions.",
            "external_inference_calls": 0,
        },
    }
    write_json(residual_packet_path, packet)
    write_text(residual_markdown_path, render_residual_packet_markdown(packet))
    return packet


def render_residual_packet_markdown(packet: dict[str, Any]) -> str:
    summary = object_field(packet, "summary")
    patch = object_field(packet, "next_source_level_decoder_patch")
    evidence = object_field(packet, "evidence")
    lines = [
        "# Private/Public Transfer Residual Packet",
        "",
        f"- Status: **{packet.get('trigger_state')}**",
        f"- Next source patch: `{summary.get('next_source_patch_id')}`",
        f"- Failed gates: `{', '.join(summary.get('failed_transfer_gates') or [])}`",
        f"- Eligible coverage delta: `{evidence.get('public_eligible_task_coverage_delta')}`",
        f"- No-admissible delta: `{evidence.get('public_no_admissible_task_rate_delta')}`",
        f"- Private receiver inventory ready: `{summary.get('private_receiver_inventory_ready')}`",
        f"- Private receiver eligible delta: `{evidence.get('private_receiver_eligible_task_rate_delta')}`",
        "",
        "## Patch",
        "",
        str(patch.get("mechanism") or ""),
        "",
        "## Target Files",
        "",
    ]
    for path in patch.get("target_files") or []:
        lines.append(f"- `{path}`")
    lines.extend(["", "## Private Acceptance", "", str(patch.get("private_eval_acceptance") or ""), ""])
    lines.extend(["## Safety", "", "- Public tests/solutions used: `False`", "- Public benchmark training allowed: `False`", ""])
    return "\n".join(lines)


def saturated_non_regressive(baseline: Any, current: Any, *, target: float, epsilon: float = 1e-9) -> bool:
    baseline_value = number(baseline)
    current_value = number(current)
    return baseline_value >= target - epsilon and current_value >= baseline_value - epsilon


def floor_non_regressive(baseline: Any, current: Any, *, floor: float, epsilon: float = 1e-9) -> bool:
    baseline_value = number(baseline)
    current_value = number(current)
    return baseline_value <= floor + epsilon and current_value <= baseline_value + epsilon


def gate(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else default
    except Exception:
        return default
    return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Private/Public Transfer Proof",
        "",
        f"- Status: **{report['trigger_state']}**",
        f"- Ready for public calibration: `{report['ready_for_public_calibration']}`",
        f"- Actual coverage delta: `{report['summary']['public_actual_token_task_coverage_delta']}`",
        f"- Eligible coverage delta: `{report['summary']['public_eligible_task_coverage_delta']}`",
        f"- No-admissible delta: `{report['summary']['public_no_admissible_task_rate_delta']}`",
        f"- Program synthesis loop coverage: `{report['summary'].get('public_program_synthesis_loop_present_rate')}`",
        f"- Actual token candidate delta: `{report['summary'].get('actual_token_candidate_count_delta')}`",
        f"- Specialized bucket substitution ready: `{report['summary'].get('specialized_bucket_substitution_ready')}`",
        f"- Contract-guided candidate delta: `{report['summary'].get('contract_guided_candidate_count_delta')}`",
        f"- STS-conditioned candidate delta: `{report['summary'].get('sts_conditioned_candidate_count_delta')}`",
        "",
        "## Gates",
        "",
    ]
    for row in report["gates"]:
        lines.append(f"- {'PASS' if row['passed'] else 'FAIL'}: `{row['name']}`")
    lines.extend(["", "## Next Actions", ""])
    for action in report["next_actions"]:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def object_field(payload: Any, key: str) -> dict[str, Any]:
    value = payload.get(key) if isinstance(payload, dict) else None
    return value if isinstance(value, dict) else {}


def get_path(payload: Any, keys: list[str], default: Any = None) -> Any:
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def number(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def path_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
