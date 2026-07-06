#!/usr/bin/env python3
"""Operator readiness packet for one bounded public code calibration.

This script never launches calibration. It only checks whether the latest
private closure, decoder gate, transfer proof, and public-boundary evidence are
strong enough for an operator to intentionally unlock exactly one bounded
public calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_POST_DISTILLATION_READINESS = REPORTS / "post_distillation_public_transfer_readiness_v1.json"
DEFAULT_V4_SCORE = REPORTS / "public_safe_broad_transfer_maturity_v4_score.json"
DEFAULT_V4_LEARNED_GATE = REPORTS / "public_safe_broad_transfer_maturity_v4_learned_distillation_gate.json"
DEFAULT_LATEST_PUBLIC_CALIBRATION = REPORTS / "real_code_benchmark_graduation_wide_public_seed23_5x32_interface_floor_v1.json"
DEFAULT_LATEST_PUBLIC_MATRIX = REPORTS / "broad_transfer_matrix_wide_public_seed23_5x32_interface_floor_v1.json"
DEFAULT_WIDE_PUBLIC_MANIFEST = REPORTS / "public_wide_slice_manifest_seed23_5x32.jsonl"
DEFAULT_PRE_PUBLIC_AUDIT = REPORTS / "pre_public_generalization_readiness_audit.json"
DEFAULT_FRONTIER_EXPANDER = REPORTS / "private_generalization_frontier_expander_v1.json"
DEFAULT_PUBLIC_CONTRACT = ROOT / "configs" / "public_benchmark_contract_v1.json"
DEFAULT_CAPABILITY_TRANSFER_CLOSURE = REPORTS / "capability_transfer_closure_v1.json"
DEFAULT_MATURITY_AUDIT = REPORTS / "maturity_integrity_audit.json"
DEFAULT_PUBLIC_TRANSFER_READINESS = REPORTS / "public_transfer_readiness_refresh_v1.json"
DEFAULT_ALIGNMENT_PREFLIGHT = REPORTS / "public_calibration_alignment_preflight.json"
DEFAULT_PRIVATE_ADMISSIBILITY = REPORTS / "private_full_body_candidate_admissibility_gate_capability_transfer_closure_v1.json"
DEFAULT_PRIVATE_HELDOUT = REPORTS / "private_residual_repair_v3_heldout_score_capability_transfer_closure_v1.json"
DEFAULT_PRIVATE_SHAPE_STRUCTURE = REPORTS / "public_shape_semantic_structure_audit_guard_shape_final.json"
DEFAULT_PUBLIC_SHAPE_STRUCTURE = REPORTS / "public_shape_semantic_structure_audit_v1.json"
DEFAULT_PUBLIC_SHAPE_RETURN = REPORTS / "public_shape_return_contract_audit_v1.json"
DEFAULT_STUDENT_TOKEN_GENERATOR = REPORTS / "student_token_code_generator_capability_transfer_closure_v1.json"
DEFAULT_VCM_RELEASE_CONFORMANCE = REPORTS / "vcm_release_conformance_audit.json"
DEFAULT_MACOS_MLX_SMOKE = REPORTS / "macos_mlx_structural_action_smoke.json"
DEFAULT_TEACHER_GATE = REPORTS / "teacher_distillation_gate.json"
DEFAULT_TEACHER_MANIFEST_AUDIT = REPORTS / "teacher_distillation_manifest_audit.json"
DEFAULT_TRAINING_ADMISSION = REPORTS / "training_data_admission_v1.json"
DEFAULT_ACCEPTED_BOUNDED_PUBLIC_SURFACES = [
    {
        "source_mbpp",
        "source_evalplus",
        "source_bigcodebench",
        "source_human_eval",
        "source_livecodebench",
    },
    {
        "source_mbpp",
        "source_evalplus",
        "source_bigcodebench",
        "source_livecodebench",
    },
    {
        "source_mbpp",
        "source_evalplus",
        "source_bigcodebench",
        "source_human_eval",
    },
]
DEFAULT_SLUG = "private_pressure_private_recovery_broad_floor_train_once_v3"
STALE_BROAD_FLOOR_SLUGS = {
    "private_pressure_private_recovery_broad_floor_train_once_v1",
    "private_pressure_private_recovery_broad_floor_train_once_v2",
    "broad_floor_public4card_calibration_v1",
}
RELEASE_BINARY = ROOT / "target" / "release" / "symliquid-cli"
DECODER_SOURCE_ROOTS = [
    ROOT / "crates" / "symliquid-cli" / "src" / "main.rs",
    ROOT / "crates" / "symliquid-cli" / "src" / "code_lm_closure",
    ROOT / "scripts" / "code_lm_closure.py",
    ROOT / "scripts" / "code_lm_private_verifier.py",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--closure-report",
        default="",
        help=(
            "Fresh private closure report to inspect. Defaults to the latest "
            "reports/code_lm_train_once_fanout.json closure_report path."
        ),
    )
    parser.add_argument("--train-once-report", default="reports/code_lm_train_once_fanout.json")
    parser.add_argument("--decoder-gate", default="reports/decoder_v2_private_ablation_gate.json")
    parser.add_argument("--transfer-proof", default="reports/private_public_transfer_proof.json")
    parser.add_argument("--broad-matrix", default="reports/broad_transfer_matrix.json")
    parser.add_argument("--recovery-report", default="reports/broad_public_code_transfer_floor_recovery.json")
    parser.add_argument("--operator-lock", default="reports/public_calibration_operator_lock.flag")
    parser.add_argument("--out", default="reports/public_calibration_readiness_packet.json")
    parser.add_argument("--markdown-out", default="reports/public_calibration_readiness_packet.md")
    parser.add_argument(
        "--mode",
        choices=["auto", "legacy", "post-distillation"],
        default="auto",
        help="auto prefers the current post-distillation v4 packet when that readiness chain is live.",
    )
    parser.add_argument("--post-distillation-readiness", default=rel_or_abs(DEFAULT_POST_DISTILLATION_READINESS))
    parser.add_argument("--v4-score", default=rel_or_abs(DEFAULT_V4_SCORE))
    parser.add_argument("--v4-learned-gate", default=rel_or_abs(DEFAULT_V4_LEARNED_GATE))
    parser.add_argument("--latest-public-calibration", default=rel_or_abs(DEFAULT_LATEST_PUBLIC_CALIBRATION))
    parser.add_argument("--latest-public-matrix", default=rel_or_abs(DEFAULT_LATEST_PUBLIC_MATRIX))
    parser.add_argument("--wide-public-manifest", default=rel_or_abs(DEFAULT_WIDE_PUBLIC_MANIFEST))
    parser.add_argument("--pre-public-audit", default=rel_or_abs(DEFAULT_PRE_PUBLIC_AUDIT))
    parser.add_argument("--frontier-expander", default=rel_or_abs(DEFAULT_FRONTIER_EXPANDER))
    parser.add_argument("--public-contract", default=rel_or_abs(DEFAULT_PUBLIC_CONTRACT))
    parser.add_argument("--capability-transfer-closure", default=rel_or_abs(DEFAULT_CAPABILITY_TRANSFER_CLOSURE))
    parser.add_argument("--maturity-audit", default=rel_or_abs(DEFAULT_MATURITY_AUDIT))
    parser.add_argument("--public-transfer-readiness", default=rel_or_abs(DEFAULT_PUBLIC_TRANSFER_READINESS))
    parser.add_argument("--alignment-preflight", default=rel_or_abs(DEFAULT_ALIGNMENT_PREFLIGHT))
    parser.add_argument("--private-admissibility", default=rel_or_abs(DEFAULT_PRIVATE_ADMISSIBILITY))
    parser.add_argument("--private-heldout", default=rel_or_abs(DEFAULT_PRIVATE_HELDOUT))
    parser.add_argument("--private-shape-structure", default=rel_or_abs(DEFAULT_PRIVATE_SHAPE_STRUCTURE))
    parser.add_argument("--public-shape-structure", default=rel_or_abs(DEFAULT_PUBLIC_SHAPE_STRUCTURE))
    parser.add_argument("--public-shape-return", default=rel_or_abs(DEFAULT_PUBLIC_SHAPE_RETURN))
    parser.add_argument("--student-token-generator", default=rel_or_abs(DEFAULT_STUDENT_TOKEN_GENERATOR))
    parser.add_argument("--vcm-release-conformance", default=rel_or_abs(DEFAULT_VCM_RELEASE_CONFORMANCE))
    parser.add_argument("--macos-mlx-smoke", default=rel_or_abs(DEFAULT_MACOS_MLX_SMOKE))
    parser.add_argument("--teacher-gate", default=rel_or_abs(DEFAULT_TEACHER_GATE))
    parser.add_argument("--teacher-manifest-audit", default=rel_or_abs(DEFAULT_TEACHER_MANIFEST_AUDIT))
    parser.add_argument("--training-admission", default=rel_or_abs(DEFAULT_TRAINING_ADMISSION))
    parser.add_argument("--proposed-slug", default="")
    args = parser.parse_args()

    started = time.perf_counter()
    if should_use_post_distillation_packet(args):
        report = build_post_distillation_packet(args, started)
        write_json(resolve(args.out), report)
        write_text(resolve(args.markdown_out), render_markdown(report))
        print(json.dumps(report, indent=2))
        return 2 if report["trigger_state"] == "RED" else 0

    train_once_report_path = resolve(args.train_once_report)
    closure_path = infer_closure_report_path(args.closure_report, train_once_report_path)
    decoder_gate_path = resolve(args.decoder_gate)
    transfer_proof_path = resolve(args.transfer_proof)
    matrix_path = resolve(args.broad_matrix)
    recovery_path = resolve(args.recovery_report)
    lock_path = resolve(args.operator_lock)
    operator_lock_text = read_text(lock_path) if lock_path.exists() else ""

    train_once = read_json(train_once_report_path, {})
    closure = read_json(closure_path, {})
    decoder_gate = read_json(decoder_gate_path, {})
    transfer_proof = read_json(transfer_proof_path, {})
    matrix = read_json(matrix_path, {})
    recovery = read_json(recovery_path, {})

    private_curriculum_path = resolve(str(closure.get("private_curriculum") or ""))
    public_manifest_path = resolve(str(closure.get("public_task_manifest") or ""))
    private_candidates_path = resolve(str(closure.get("private_candidate_manifest") or ""))
    public_candidates_path = resolve(str(closure.get("public_candidate_manifest") or ""))
    private_curriculum = scan_rows(private_curriculum_path, scope="private_curriculum")
    public_manifest = scan_rows(public_manifest_path, scope="public_manifest")
    private_candidates = scan_rows(private_candidates_path, scope="private_candidates")
    public_candidates = scan_rows(public_candidates_path, scope="public_candidates")

    latest_closure_from_gate = str(get_path(decoder_gate, ["summary", "latest_closure"], "") or decoder_gate.get("latest_closure") or "")
    current_closure_from_transfer = str(get_path(transfer_proof, ["current", "latest_closure"], "") or "")
    canonical_slug = closure_slug(closure_path)
    canonical_artifacts = canonical_artifact_check(
        canonical_slug=canonical_slug,
        stale_slugs=STALE_BROAD_FLOOR_SLUGS,
        paths={
            "closure_report": closure_path,
            "private_curriculum": private_curriculum_path,
            "public_manifest": public_manifest_path,
            "private_candidates": private_candidates_path,
            "public_candidates": public_candidates_path,
        },
        referenced_paths={
            "decoder_gate_latest_closure": latest_closure_from_gate,
            "transfer_proof_current_latest_closure": current_closure_from_transfer,
        },
    )
    recovery_mtime = path_mtime(recovery_path)
    closure_mtime = path_mtime(closure_path)

    public_card_counts = public_manifest["card_counts"]
    public_cards = set(public_card_counts)
    expected_surface = expected_bounded_public_surface(train_once, closure)
    accepted_surfaces = accepted_bounded_public_surfaces(expected_surface)
    card_surface_ok = (
        len(public_cards) in {4, 5}
        and public_cards in accepted_surfaces
        and all(count > 0 for count in public_card_counts.values())
    )
    no_public_leak = all(
        not item["public_leak_hit_count"]
        for item in [private_curriculum, public_manifest, private_candidates, public_candidates]
    )
    closure_completed = (
        bool(closure)
        and closure.get("run_status") == "completed"
        and closure.get("trigger_state") in {"GREEN", "YELLOW"}
    )
    private_curriculum_mtime = float(private_curriculum.get("mtime") or 0.0)
    closure_fresh = bool(closure_completed and closure_mtime >= private_curriculum_mtime > 0.0)
    decoder_gate_fresh = (
        bool(decoder_gate.get("ready_for_public_calibration"))
        and decoder_gate.get("trigger_state") == "GREEN"
        and same_path(latest_closure_from_gate, closure_path)
        and path_mtime(decoder_gate_path) >= closure_mtime > 0.0
    )
    transfer_proof_fresh = (
        bool(transfer_proof.get("ready_for_public_calibration"))
        and transfer_proof.get("trigger_state") == "GREEN"
        and same_path(current_closure_from_transfer, closure_path)
        and path_mtime(transfer_proof_path) >= path_mtime(decoder_gate_path) > 0.0
    )
    recovery_consumed = private_curriculum["broad_floor_recovery_row_count"] > 0
    checkpoint_backend = str(get_path(closure, ["summary", "checkpoint_backend"], "") or "")
    checkpoint_backend_ready = (
        bool(checkpoint_backend)
        and get_path(closure, ["summary", "checkpoint_trigger_state"], "") == "GREEN"
        and get_path(closure, ["summary", "fanout_trigger_state"], "") == "GREEN"
    )
    source_freshness = decoder_source_release_freshness(closure_mtime)
    public_calibration_not_run = get_path(closure, ["summary", "public_calibration_allowed"], None) is False
    operator_lock_active = lock_path.exists()
    technical_ready = all(
        [
            closure_completed,
            closure_fresh,
            source_freshness["fresh"],
            recovery_consumed,
            checkpoint_backend_ready,
            public_calibration_not_run,
            card_surface_ok,
            no_public_leak,
            canonical_artifacts["passed"],
            decoder_gate_fresh,
            transfer_proof_fresh,
        ]
    )
    public_calibration_allowed = bool(technical_ready and not operator_lock_active)

    gates = [
        gate("fresh_private_closure_completed", closure_completed and closure_fresh, {
            "closure": rel_or_abs(closure_path),
            "closure_mtime": closure_mtime,
            "private_curriculum_mtime": private_curriculum_mtime,
            "recovery_mtime": recovery_mtime,
            "run_status": closure.get("run_status"),
            "trigger_state": closure.get("trigger_state"),
            "rule": "the canonical closure must be newer than the private curriculum it consumed; later private diagnostic/A-B reports do not make the closure stale",
        }),
        gate("closure_current_for_decoder_source_and_release", source_freshness["fresh"], source_freshness),
        gate("broad_floor_recovery_rows_consumed", recovery_consumed, private_curriculum),
        gate("checkpoint_backend_ready", checkpoint_backend_ready, {
            "backend": checkpoint_backend,
            "checkpoint_trigger_state": get_path(closure, ["summary", "checkpoint_trigger_state"], ""),
            "fanout_trigger_state": get_path(closure, ["summary", "fanout_trigger_state"], ""),
            "checkpoint_cuda_readout_used": bool(get_path(closure, ["summary", "checkpoint_cuda_readout_used"], False)),
            "rule": "CUDA, MLX, or Rust CPU readout may satisfy readiness when checkpoint and fanout stages are GREEN on the current node.",
        }),
        gate("public_calibration_not_run_by_closure", public_calibration_not_run, get_path(closure, ["summary"], {})),
        gate("bounded_public_surface", card_surface_ok, {
            "expected": sorted(expected_surface) if expected_surface else None,
            "accepted": [sorted(surface) for surface in accepted_surfaces],
            "observed": dict(sorted(public_card_counts.items())),
        }),
        gate("public_boundary_clean", no_public_leak, {
            "private_curriculum": private_curriculum["public_leak_hit_count"],
            "public_manifest": public_manifest["public_leak_hit_count"],
            "private_candidates": private_candidates["public_leak_hit_count"],
            "public_candidates": public_candidates["public_leak_hit_count"],
        }),
        gate("canonical_current_broad_floor_artifacts", canonical_artifacts["passed"], canonical_artifacts),
        gate("decoder_gate_fresh_and_ready", decoder_gate_fresh, {
            "ready": decoder_gate.get("ready_for_public_calibration"),
            "trigger_state": decoder_gate.get("trigger_state"),
            "latest_closure": latest_closure_from_gate,
            "expected_closure": rel_or_abs(closure_path),
        }),
        gate("private_public_transfer_proof_fresh_and_ready", transfer_proof_fresh, {
            "ready": transfer_proof.get("ready_for_public_calibration"),
            "trigger_state": transfer_proof.get("trigger_state"),
            "current_latest_closure": current_closure_from_transfer,
            "expected_closure": rel_or_abs(closure_path),
        }),
        gate("operator_lock_state_recorded", True, {
            "active": operator_lock_active,
            "path": rel_or_abs(lock_path),
        }),
    ]

    leak_blocked = not no_public_leak
    trigger_state = "RED" if leak_blocked else ("GREEN" if technical_ready else "YELLOW")
    report = {
        "policy": "project_theseus_public_calibration_readiness_packet_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "technical_ready_for_one_bounded_4_card_calibration": technical_ready,
        "technical_ready_for_one_bounded_public_calibration": technical_ready,
        "public_calibration_allowed": public_calibration_allowed,
        "operator_lock_active": operator_lock_active,
        "summary": {
            "technical_ready": technical_ready,
            "public_calibration_allowed": public_calibration_allowed,
            "operator_lock_active": operator_lock_active,
            "broad_public_pass_rate": get_path(matrix, ["summary", "real_public_pass_rate"], None),
            "cards_below_floor": get_path(matrix, ["summary", "cards_below_floor"], []),
            "recovery_status": recovery.get("status"),
            "recovery_private_rows": get_path(recovery, ["summary", "private_pressure_row_count"], 0),
            "closure_report": rel_or_abs(closure_path),
            "decoder_gate": rel_or_abs(decoder_gate_path),
            "transfer_proof": rel_or_abs(transfer_proof_path),
            "public_card_counts": dict(sorted(public_card_counts.items())),
            "public_surface_card_count": len(public_cards),
            "public_surface_task_count": sum(public_card_counts.values()),
            "checkpoint_backend": checkpoint_backend,
            "checkpoint_backend_ready": checkpoint_backend_ready,
            "decoder_source_release_fresh": source_freshness["fresh"],
            "newest_decoder_source_mtime": source_freshness["newest_source_mtime"],
            "release_binary_mtime": source_freshness["release_binary_mtime"],
            "public_tests_or_solutions_visible": not no_public_leak,
            "canonical_slug": canonical_slug,
            "stale_artifact_count": canonical_artifacts["stale_artifact_count"],
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "artifacts": {
            "private_curriculum": private_curriculum,
            "public_manifest": public_manifest,
            "private_candidates": private_candidates,
            "public_candidates": public_candidates,
        },
        "gates": gates,
        "next_actions": next_actions(
            technical_ready,
            operator_lock_active,
            transfer_proof_fresh,
            decoder_gate_fresh,
            source_freshness["fresh"],
            operator_lock_text,
        ),
        "rules": {
            "public_calibration": "never launched by this script; operator must explicitly remove/override the lock for exactly one bounded public calibration",
            "public_boundary": "public prompts/signatures may be calibration metadata; public tests and canonical solutions must remain absent",
            "score_semantics": "readiness packet only, not public benchmark score or promotion evidence",
        },
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 2 if trigger_state == "RED" else 0


def should_use_post_distillation_packet(args: argparse.Namespace) -> bool:
    if args.mode == "legacy":
        return False
    if args.mode == "post-distillation":
        return True
    readiness = read_json(resolve(args.post_distillation_readiness), {})
    summary = object_field(readiness, "summary")
    return (
        readiness.get("policy") == "project_theseus_post_distillation_public_transfer_readiness_v1"
        and summary.get("recommended_private_fix_family") == "operator_reviewed_bounded_public_calibration_packet"
    )


def build_post_distillation_packet(args: argparse.Namespace, started: float) -> dict[str, Any]:
    readiness_path = resolve(args.post_distillation_readiness)
    v4_score_path = resolve(args.v4_score)
    v4_gate_path = resolve(args.v4_learned_gate)
    latest_public_path = resolve(args.latest_public_calibration)
    latest_matrix_path = resolve(args.latest_public_matrix)
    manifest_path = resolve(args.wide_public_manifest)
    pre_public_audit_path = resolve(args.pre_public_audit)
    frontier_expander_path = resolve(args.frontier_expander)
    public_contract_path = resolve(args.public_contract)
    capability_closure_path = resolve(args.capability_transfer_closure)
    maturity_audit_path = resolve(args.maturity_audit)
    public_transfer_readiness_path = resolve(args.public_transfer_readiness)
    alignment_preflight_path = resolve(args.alignment_preflight)
    private_admissibility_path = resolve(args.private_admissibility)
    private_heldout_path = resolve(args.private_heldout)
    private_shape_structure_path = resolve(args.private_shape_structure)
    public_shape_structure_path = resolve(args.public_shape_structure)
    public_shape_return_path = resolve(args.public_shape_return)
    student_token_generator_path = resolve(args.student_token_generator)
    vcm_release_path = resolve(args.vcm_release_conformance)
    macos_mlx_smoke_path = resolve(args.macos_mlx_smoke)
    teacher_gate_path = resolve(args.teacher_gate)
    teacher_manifest_path = resolve(args.teacher_manifest_audit)
    training_admission_path = resolve(args.training_admission)
    lock_path = resolve(args.operator_lock)

    readiness = read_json(readiness_path, {})
    v4_score = read_json(v4_score_path, {})
    v4_gate = read_json(v4_gate_path, {})
    latest_public = read_json(latest_public_path, {})
    latest_matrix = read_json(latest_matrix_path, {})
    pre_public_audit = read_json(pre_public_audit_path, {})
    frontier_expander = read_json(frontier_expander_path, {})
    public_contract = read_json(public_contract_path, {})
    capability_closure = read_json(capability_closure_path, {})
    maturity_audit = read_json(maturity_audit_path, {})
    public_transfer_readiness = read_json(public_transfer_readiness_path, {})
    alignment_preflight = read_json(alignment_preflight_path, {})
    private_admissibility = read_json(private_admissibility_path, {})
    private_heldout = read_json(private_heldout_path, {})
    private_shape_structure = read_json(private_shape_structure_path, {})
    public_shape_structure = read_json(public_shape_structure_path, {})
    public_shape_return = read_json(public_shape_return_path, {})
    student_token_generator = read_json(student_token_generator_path, {})
    vcm_release = read_json(vcm_release_path, {})
    macos_mlx_smoke = read_json(macos_mlx_smoke_path, {})
    teacher_gate = read_json(teacher_gate_path, {})
    teacher_manifest = read_json(teacher_manifest_path, {})
    training_admission = read_json(training_admission_path, {})
    manifest = scan_rows(manifest_path, scope="public_manifest")
    frozen_integrity = build_frozen_integrity_snapshot(
        public_contract=public_contract,
        evidence_paths={
            "post_distillation_readiness": readiness_path,
            "pre_public_audit": pre_public_audit_path,
            "frontier_expander": frontier_expander_path,
            "v4_score": v4_score_path,
            "v4_learned_gate": v4_gate_path,
            "wide_public_manifest": manifest_path,
            "latest_public_calibration": latest_public_path,
            "latest_public_matrix": latest_matrix_path,
            "public_benchmark_contract": public_contract_path,
            "capability_transfer_closure": capability_closure_path,
            "maturity_audit": maturity_audit_path,
            "public_transfer_readiness": public_transfer_readiness_path,
            "alignment_preflight": alignment_preflight_path,
            "private_admissibility": private_admissibility_path,
            "private_heldout": private_heldout_path,
            "private_shape_structure": private_shape_structure_path,
            "public_shape_structure": public_shape_structure_path,
            "public_shape_return": public_shape_return_path,
            "student_token_generator": student_token_generator_path,
            "vcm_release_conformance": vcm_release_path,
            "macos_mlx_smoke": macos_mlx_smoke_path,
            "teacher_gate": teacher_gate_path,
            "teacher_manifest_audit": teacher_manifest_path,
            "training_admission": training_admission_path,
        },
    )

    readiness_summary = object_field(readiness, "summary")
    v4_score_summary = object_field(v4_score, "summary")
    v4_gate_summary = object_field(v4_gate, "summary")
    latest_public_summary = object_field(latest_public, "summary")
    latest_matrix_summary = object_field(latest_matrix, "summary")
    pre_public_summary = object_field(pre_public_audit, "summary")
    frontier_summary = object_field(frontier_expander, "summary")
    public_transfer_summary = object_field(public_transfer_readiness, "summary")
    alignment_summary = object_field(alignment_preflight, "summary")
    gate_inventory = object_field(v4_gate_summary, "candidate_inventory")
    current_guard_shape = current_guard_shape_readiness(
        capability_closure=capability_closure,
        maturity_audit=maturity_audit,
        private_admissibility=private_admissibility,
        private_heldout=private_heldout,
        private_shape_structure=private_shape_structure,
        public_shape_structure=public_shape_structure,
        public_shape_return=public_shape_return,
        student_token_generator=student_token_generator,
        vcm_release=vcm_release,
        macos_mlx_smoke=macos_mlx_smoke,
        teacher_gate=teacher_gate,
        teacher_manifest=teacher_manifest,
        training_admission=training_admission,
    )

    public_cards = set(manifest["card_counts"])
    public_task_count = sum(int(value) for value in manifest["card_counts"].values())
    expected_cards = {
        "source_mbpp",
        "source_evalplus",
        "source_bigcodebench",
        "source_human_eval",
        "source_livecodebench",
    }
    post_integrity_ready = (
        readiness.get("policy") == "project_theseus_post_distillation_public_transfer_readiness_v1"
        and readiness.get("trigger_state") == "YELLOW"
        and len(readiness.get("hard_blockers") or []) == 0
        and readiness_summary.get("recommended_private_fix_family") == "operator_reviewed_bounded_public_calibration_packet"
        and readiness_summary.get("completed_successor_private_fix_family") == "edge_contract_v4_public_safe_broad_transfer_maturity_curriculum"
        and readiness_summary.get("completed_successor_private_fix_completed") is True
        and readiness_summary.get("decoder_source_release_fresh") is True
    )
    pre_public_queue = pre_public_audit.get("queue") if isinstance(pre_public_audit.get("queue"), list) else []
    audit_queue_kinds = [str(row.get("kind") or "") for row in pre_public_queue if isinstance(row, dict)]
    current_pre_public_ready = (
        pre_public_audit.get("policy") == "project_theseus_pre_public_generalization_readiness_audit_v1"
        and pre_public_audit.get("trigger_state") == "YELLOW"
        and pre_public_summary.get("operator_review_ready") is True
        and pre_public_summary.get("private_code_transfer_ready") is True
        and pre_public_summary.get("private_agent_transfer_ready") is True
        and pre_public_summary.get("teacher_path_ready") is True
        and int(first_number(pre_public_summary.get("hard_failed_gate_count"), 999)) == 0
        and pre_public_summary.get("public_calibration_allowed") is False
        and pre_public_summary.get("operator_lock_active") is True
        and first_number(pre_public_summary.get("public_pass_rate")) == first_number(readiness_summary.get("public_pass_rate"), latest_matrix_summary.get("real_public_pass_rate"))
        and int(first_number(pre_public_summary.get("learned_token_pass_count_total"))) >= int(first_number(v4_gate_summary.get("learned_token_pass_count"), 0))
        and frontier_expander.get("policy") == "project_theseus_private_generalization_frontier_expander_v1"
        and frontier_expander.get("trigger_state") == "GREEN"
        and frontier_summary.get("decision") == "no_private_frontier_action_remaining"
        and not frontier_summary.get("next_safe_private_action")
        and audit_queue_kinds == ["operator_review_bounded_public_calibration_locked"]
    )
    public_transfer_failed_gate_names = failed_transfer_gate_names(public_transfer_readiness)
    public_transfer_semantic_not_dead = (
        public_transfer_summary.get("full_body_post_v4_default_semantic_dead") is False
        or first_number(public_transfer_summary.get("full_body_v4_full_body_selected_pass_rate")) > 0.0
        or first_number(public_transfer_summary.get("full_body_v4_strict_novel_learned_only_pass_rate")) > 0.0
        or first_number(public_transfer_summary.get("full_body_semantic_best_selected_pass_rate")) > 0.0
    )
    public_transfer_private_evidence_current = (
        public_transfer_readiness.get("policy") == "project_theseus_public_transfer_readiness_refresh_v1"
        and public_transfer_readiness.get("trigger_state") in {"GREEN", "YELLOW"}
        and int(first_number(public_transfer_summary.get("hard_failed_gate_count"), 999)) == 0
        and int(first_number(public_transfer_summary.get("evidence_failed_gate_count"), 999)) == 0
        and set(public_transfer_failed_gate_names).issubset({"current_public_transfer_floor_cleared"})
        and public_transfer_summary.get("private_fix_review_ready") is True
        and public_transfer_summary.get("calibration_review_ready_after_private_fixes") is True
        and int(first_number(public_transfer_summary.get("private_residual_unresolved_target_count"), 999)) == 0
        and public_transfer_summary.get("alignment_preflight_ready") is True
        and alignment_preflight_ready(alignment_preflight)
        and int(first_number(public_transfer_summary.get("contract_task_count"))) == 320
        and first_number(public_transfer_summary.get("full_body_selected_pass_rate")) > 0.0
        and first_number(public_transfer_summary.get("full_body_pass_if_any_rate")) > 0.0
        and int(first_number(public_transfer_summary.get("full_body_benchmark_promotion_eligible_candidate_count"))) > 0
        and int(first_number(public_transfer_summary.get("full_body_fallback_return_candidate_count"))) == 0
        and int(first_number(public_transfer_summary.get("fallback_return_candidate_count"))) == 0
        and int(first_number(public_transfer_summary.get("full_body_public_leakage_count"))) == 0
        and int(first_number(public_transfer_summary.get("public_leakage_count"))) == 0
        and public_transfer_semantic_not_dead
        and public_transfer_readiness.get("public_tests_used") is False
        and public_transfer_readiness.get("public_solutions_used") is False
        and int(first_number(public_transfer_readiness.get("external_inference_calls"))) == 0
        and public_transfer_readiness.get("public_calibration_allowed") is False
    )
    v4_score_ready = (
        v4_score.get("trigger_state") == "GREEN"
        and first_number(v4_score_summary.get("pass_rate")) >= 0.70
        and int(first_number(v4_score_summary.get("heldout_task_count"))) >= 1000
        and int(first_number(v4_score_summary.get("control_pass_count"))) == 0
        and int(first_number(v4_score_summary.get("sts_regressions"))) == 0
        and v4_score_summary.get("public_tests_used") is False
        and v4_score_summary.get("public_solutions_used") is False
    )
    v4_learned_ready = (
        v4_gate.get("trigger_state") == "GREEN"
        and first_number(v4_gate_summary.get("learned_only_pass_rate")) >= 0.70
        and int(first_number(v4_gate_summary.get("learned_only_task_count"))) >= 1000
        and int(first_number(v4_gate_summary.get("prototype_pass_count"), -1)) == 0
        and int(first_number(v4_gate_summary.get("learned_token_pass_count"))) >= int(first_number(v4_gate_summary.get("learned_only_task_count")))
        and int(first_number(gate_inventory.get("prototype_rows"), -1)) == 0
        and v4_gate_summary.get("decoder_source_release_fresh") is True
        and v4_gate.get("public_tests_used") is False
        and v4_gate.get("public_solutions_used") is False
    )
    latest_public_consumed = (
        latest_public.get("policy") == "project_theseus_real_code_benchmark_graduation_v1"
        and int(first_number(latest_public_summary.get("public_task_count"))) >= 160
        and first_number(
            latest_public_summary.get("real_public_task_pass_rate"),
            latest_public_summary.get("multi_stream_pass_rate"),
        ) == first_number(pre_public_summary.get("public_pass_rate"), readiness_summary.get("public_pass_rate"))
    )
    manifest_ready = (
        manifest["exists"]
        and public_cards == expected_cards
        and public_task_count == 160
        and all(int(count) == 32 for count in manifest["card_counts"].values())
        and manifest["public_leak_hit_count"] == 0
    )
    public_boundary_clean = (
        readiness.get("public_tests_used") is False
        and readiness.get("public_solutions_used") is False
        and not any_true("public_tests_used", pre_public_audit)
        and not any_true("public_solutions_used", pre_public_audit)
        and not any_true("public_tests_used", frontier_expander)
        and not any_true("public_solutions_used", frontier_expander)
        and v4_score_summary.get("public_tests_used") is False
        and v4_score_summary.get("public_solutions_used") is False
        and v4_gate.get("public_tests_used") is False
        and v4_gate.get("public_solutions_used") is False
        and alignment_preflight.get("public_tests_used") is False
        and alignment_preflight.get("public_solutions_used") is False
        and manifest["public_leak_hit_count"] == 0
    )
    external_inference_zero = max(
        int(first_number(readiness.get("external_inference_calls"), readiness_summary.get("external_inference_calls"))),
        int(first_number(pre_public_audit.get("external_inference_calls"), pre_public_summary.get("external_inference_calls"))),
        int(first_number(frontier_expander.get("external_inference_calls"), frontier_summary.get("external_inference_calls"))),
        int(first_number(v4_score.get("external_inference_calls"), v4_score_summary.get("external_inference_calls"))),
        int(first_number(v4_gate.get("external_inference_calls"), v4_gate_summary.get("external_inference_calls"))),
        int(first_number(alignment_preflight.get("external_inference_calls"), alignment_summary.get("external_inference_calls"))),
    ) == 0
    packet_does_not_run_public = True
    next_surface = object_field(public_contract, "stage_1_code_generation_surface")
    consumed_surface = object_field(object_field(public_contract, "global_rules"), "consumed_surface_do_not_rerun")
    next_surface_cards = set(str(card) for card in list_value(next_surface.get("cards")))
    next_surface_command = list_value(next_surface.get("command_after_unlock_only"))
    proposed_slug = str(args.proposed_slug or next_surface.get("slug") or "industry_code_transfer_seed14_5x64_v1")
    next_public_contract_ready = (
        public_contract.get("policy") == "project_theseus_public_benchmark_contract_v1"
        and next_surface.get("status") == "contracted_not_executed"
        and proposed_slug == next_surface.get("slug")
        and next_surface.get("slug") != consumed_surface.get("slug")
        and int(first_number(next_surface.get("total_task_count"))) == 320
        and int(first_number(next_surface.get("cases_per_card"))) == 64
        and int(first_number(next_surface.get("seed"))) == 14
        and next_surface_cards == expected_cards
        and next_surface.get("case_manifest") == "reports/public_wide_slice_manifest_industry_code_transfer_seed14_5x64_v1.jsonl"
        and next_surface.get("case_manifest_report") == "reports/public_wide_slice_selector_industry_code_transfer_seed14_5x64_v1.json"
        and next_surface_command[:2] == ["python3", "scripts/real_code_benchmark_graduation.py"]
        and command_arg(next_surface_command, "--case-manifest") == "reports/public_wide_slice_manifest_industry_code_transfer_seed14_5x64_v1.jsonl"
        and "--skip-student-candidate-generation" not in next_surface_command
    )
    technical_ready = all(
        [
            post_integrity_ready or current_pre_public_ready,
            v4_score_ready,
            v4_learned_ready,
            latest_public_consumed,
            manifest_ready,
            public_boundary_clean,
            external_inference_zero,
            packet_does_not_run_public,
            next_public_contract_ready,
            current_guard_shape["passed"],
            public_transfer_private_evidence_current,
            frozen_integrity["harness_hashes_current"],
            frozen_integrity["evidence_artifacts_hashable"],
        ]
    )
    operator_lock_active = lock_path.exists()
    public_calibration_allowed = bool(technical_ready and not operator_lock_active)
    trigger_state = "RED" if not public_boundary_clean else ("GREEN" if technical_ready else "YELLOW")
    proposed = proposed_public_calibration_commands(args, public_contract, proposed_slug)

    review_chain_ready = post_integrity_ready or current_pre_public_ready
    gates = [
        gate("review_readiness_chain_current", review_chain_ready, {
            "path": rel_or_abs(readiness_path),
            "trigger_state": readiness.get("trigger_state"),
            "hard_blocker_count": len(readiness.get("hard_blockers") or []),
            "legacy_post_distillation_ready": post_integrity_ready,
            "current_pre_public_ready": current_pre_public_ready,
            "recommended_private_fix_family": readiness_summary.get("recommended_private_fix_family"),
            "completed_successor_private_fix_family": readiness_summary.get("completed_successor_private_fix_family"),
            "completed_successor_private_fix_completed": readiness_summary.get("completed_successor_private_fix_completed"),
        }),
        gate("current_pre_public_audit_operator_review_ready", current_pre_public_ready, {
            "path": rel_or_abs(pre_public_audit_path),
            "trigger_state": pre_public_audit.get("trigger_state"),
            "operator_review_ready": pre_public_summary.get("operator_review_ready"),
            "hard_failed_gate_count": pre_public_summary.get("hard_failed_gate_count"),
            "learned_token_pass_count_total": pre_public_summary.get("learned_token_pass_count_total"),
            "frontier_expander": rel_or_abs(frontier_expander_path),
            "frontier_expander_decision": frontier_summary.get("decision"),
            "frontier_expander_next_safe_private_action": frontier_summary.get("next_safe_private_action"),
            "audit_queue_kinds": audit_queue_kinds,
        }),
        gate("v4_private_score_green", v4_score_ready, {
            "path": rel_or_abs(v4_score_path),
            "trigger_state": v4_score.get("trigger_state"),
            "pass_count": v4_score_summary.get("pass_count"),
            "heldout_task_count": v4_score_summary.get("heldout_task_count"),
            "pass_rate": v4_score_summary.get("pass_rate"),
            "control_pass_count": v4_score_summary.get("control_pass_count"),
            "sts_delta": v4_score_summary.get("sts_delta"),
        }),
        gate("v4_learned_only_no_prototype_gate_green", v4_learned_ready, {
            "path": rel_or_abs(v4_gate_path),
            "trigger_state": v4_gate.get("trigger_state"),
            "learned_only_pass_count": v4_gate_summary.get("learned_only_pass_count"),
            "learned_only_task_count": v4_gate_summary.get("learned_only_task_count"),
            "learned_only_pass_rate": v4_gate_summary.get("learned_only_pass_rate"),
            "prototype_pass_count": v4_gate_summary.get("prototype_pass_count"),
            "learned_token_pass_count": v4_gate_summary.get("learned_token_pass_count"),
        }),
        gate("latest_wide_public_calibration_consumed_and_relocked", latest_public_consumed and operator_lock_active, {
            "latest_public_calibration": rel_or_abs(latest_public_path),
            "public_pass_rate": readiness_summary.get("public_pass_rate"),
            "public_task_count": readiness_summary.get("public_task_count"),
            "operator_lock_active": operator_lock_active,
            "operator_lock": rel_or_abs(lock_path),
        }),
        gate("wide_public_manifest_pinned_5x32", manifest_ready, {
            "manifest": rel_or_abs(manifest_path),
            "card_counts": dict(sorted(manifest["card_counts"].items())),
            "row_count": manifest["row_count"],
            "public_leak_hit_count": manifest["public_leak_hit_count"],
        }),
        gate("public_boundary_clean", public_boundary_clean, {
            "readiness_public_tests_used": readiness.get("public_tests_used"),
            "readiness_public_solutions_used": readiness.get("public_solutions_used"),
            "pre_public_audit_public_tests_used": any_true("public_tests_used", pre_public_audit),
            "pre_public_audit_public_solutions_used": any_true("public_solutions_used", pre_public_audit),
            "frontier_expander_public_tests_used": any_true("public_tests_used", frontier_expander),
            "frontier_expander_public_solutions_used": any_true("public_solutions_used", frontier_expander),
            "v4_score_public_tests_used": v4_score_summary.get("public_tests_used"),
            "v4_score_public_solutions_used": v4_score_summary.get("public_solutions_used"),
            "v4_gate_public_tests_used": v4_gate.get("public_tests_used"),
            "v4_gate_public_solutions_used": v4_gate.get("public_solutions_used"),
            "alignment_preflight_public_tests_used": alignment_preflight.get("public_tests_used"),
            "alignment_preflight_public_solutions_used": alignment_preflight.get("public_solutions_used"),
            "manifest_public_leak_hit_count": manifest["public_leak_hit_count"],
        }),
        gate("external_inference_zero", external_inference_zero, {
            "readiness": readiness.get("external_inference_calls"),
            "pre_public_audit": pre_public_audit.get("external_inference_calls"),
            "frontier_expander": frontier_expander.get("external_inference_calls"),
            "v4_score": v4_score.get("external_inference_calls"),
            "v4_learned_gate": v4_gate.get("external_inference_calls"),
            "alignment_preflight": alignment_preflight.get("external_inference_calls"),
        }),
        gate("packet_does_not_run_public_calibration", packet_does_not_run_public, {
            "script": "scripts/public_calibration_readiness_packet.py",
            "rule": "this packet only writes JSON/Markdown readiness evidence; it does not remove the operator lock or run real_code_benchmark_graduation.py",
        }),
        gate("current_guard_shape_transfer_evidence_ready", current_guard_shape["passed"], current_guard_shape),
        gate("public_transfer_readiness_private_evidence_current", public_transfer_private_evidence_current, {
            "path": rel_or_abs(public_transfer_readiness_path),
            "trigger_state": public_transfer_readiness.get("trigger_state"),
            "hard_failed_gate_count": public_transfer_summary.get("hard_failed_gate_count"),
            "evidence_failed_gate_count": public_transfer_summary.get("evidence_failed_gate_count"),
            "transfer_failed_gate_count": public_transfer_summary.get("transfer_failed_gate_count"),
            "failed_transfer_gate_names": public_transfer_failed_gate_names,
            "private_fix_review_ready": public_transfer_summary.get("private_fix_review_ready"),
            "calibration_review_ready_after_private_fixes": public_transfer_summary.get(
                "calibration_review_ready_after_private_fixes"
            ),
            "private_residual_unresolved_target_count": public_transfer_summary.get(
                "private_residual_unresolved_target_count"
            ),
            "private_residual_unresolved_target_category_counts": public_transfer_summary.get(
                "private_residual_unresolved_target_category_counts"
            ),
            "contract_task_count": public_transfer_summary.get("contract_task_count"),
            "alignment_preflight_state": alignment_preflight.get("trigger_state"),
            "alignment_preflight_ready": alignment_preflight.get("alignment_preflight_ready"),
            "alignment_case_manifest": alignment_summary.get("case_manifest"),
            "alignment_case_manifest_row_count": alignment_summary.get("case_manifest_row_count"),
            "alignment_candidate_manifest_bound_to_case_manifest": alignment_summary.get(
                "candidate_manifest_bound_to_case_manifest"
            ),
            "full_body_selected_pass_rate": public_transfer_summary.get("full_body_selected_pass_rate"),
            "full_body_pass_if_any_rate": public_transfer_summary.get("full_body_pass_if_any_rate"),
            "full_body_benchmark_promotion_eligible_candidate_count": public_transfer_summary.get("full_body_benchmark_promotion_eligible_candidate_count"),
            "full_body_fallback_return_candidate_count": public_transfer_summary.get("full_body_fallback_return_candidate_count"),
            "full_body_public_leakage_count": public_transfer_summary.get("full_body_public_leakage_count"),
            "full_body_post_v4_default_semantic_dead": public_transfer_summary.get("full_body_post_v4_default_semantic_dead"),
            "full_body_v4_full_body_selected_pass_rate": public_transfer_summary.get("full_body_v4_full_body_selected_pass_rate"),
            "full_body_v4_strict_novel_learned_only_pass_rate": public_transfer_summary.get("full_body_v4_strict_novel_learned_only_pass_rate"),
            "full_body_semantic_best_selected_pass_rate": public_transfer_summary.get("full_body_semantic_best_selected_pass_rate"),
            "semantic_not_dead": public_transfer_semantic_not_dead,
            "latest_public_pass_rate": public_transfer_summary.get("latest_public_pass_rate"),
            "latest_public_task_count": public_transfer_summary.get("latest_public_task_count"),
            "public_tests_used": public_transfer_readiness.get("public_tests_used"),
            "public_solutions_used": public_transfer_readiness.get("public_solutions_used"),
            "external_inference_calls": public_transfer_readiness.get("external_inference_calls"),
            "public_calibration_allowed": public_transfer_readiness.get("public_calibration_allowed"),
            "rule": "This gate proves only private evidence readiness. It does not clear the public transfer floor or execute calibration.",
        }),
        gate("frozen_next_public_contract_ready", next_public_contract_ready, {
            "contract": rel_or_abs(public_contract_path),
            "contract_status": public_contract.get("status"),
            "consumed_slug": consumed_surface.get("slug"),
            "proposed_slug": proposed_slug,
            "status": next_surface.get("status"),
            "seed": next_surface.get("seed"),
            "cases_per_card": next_surface.get("cases_per_card"),
            "total_task_count": next_surface.get("total_task_count"),
            "case_manifest": next_surface.get("case_manifest"),
            "case_manifest_report": next_surface.get("case_manifest_report"),
            "cards": sorted(next_surface_cards),
            "command": next_surface_command,
        }),
        gate("frozen_harness_hashes_current", frozen_integrity["harness_hashes_current"], {
            "contract": rel_or_abs(public_contract_path),
            "mismatch_count": frozen_integrity["harness_mismatch_count"],
            "missing_count": frozen_integrity["harness_missing_count"],
            "mismatches": frozen_integrity["harness_mismatches"],
            "rule": "public approval must be tied to the same harness file hashes frozen in the public benchmark contract",
        }),
        gate("frozen_evidence_artifacts_hashable", frozen_integrity["evidence_artifacts_hashable"], {
            "missing_count": frozen_integrity["evidence_missing_count"],
            "missing": frozen_integrity["evidence_missing"],
            "sha256": frozen_integrity["sha256"],
            "rule": "the packet records exact evidence artifact hashes so approval is bound to current evidence, not a moving report name",
        }),
    ]

    return {
        "policy": "project_theseus_public_calibration_readiness_packet_v1",
        "mode": "post_distillation_v4_operator_review",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "technical_ready_for_one_bounded_4_card_calibration": technical_ready,
        "technical_ready_for_one_bounded_public_calibration": technical_ready,
        "public_calibration_allowed": public_calibration_allowed,
        "operator_lock_active": operator_lock_active,
        "summary": {
            "technical_ready": technical_ready,
            "public_calibration_allowed": public_calibration_allowed,
            "operator_lock_active": operator_lock_active,
            "post_distillation_readiness": rel_or_abs(readiness_path),
            "recommended_private_fix_family": readiness_summary.get("recommended_private_fix_family"),
            "completed_successor_private_fix_family": readiness_summary.get("completed_successor_private_fix_family"),
            "completed_successor_private_fix_completed": readiness_summary.get("completed_successor_private_fix_completed"),
            "current_pre_public_audit": rel_or_abs(pre_public_audit_path),
            "pre_public_operator_review_ready": pre_public_summary.get("operator_review_ready"),
            "pre_public_learned_token_pass_count_total": pre_public_summary.get("learned_token_pass_count_total"),
            "frontier_expander": rel_or_abs(frontier_expander_path),
            "frontier_expander_decision": frontier_summary.get("decision"),
            "frontier_expander_next_safe_private_action": frontier_summary.get("next_safe_private_action"),
            "current_capability_transfer_closure": rel_or_abs(capability_closure_path),
            "current_closure_trigger_state": capability_closure.get("trigger_state"),
            "current_private_semantic_ready": current_guard_shape["summary"].get("private_semantic_ready"),
            "current_public_candidate_coverage_ready": current_guard_shape["summary"].get("public_candidate_coverage_ready"),
            "current_maturity_audit": rel_or_abs(maturity_audit_path),
            "current_maturity_blockers": current_guard_shape["summary"].get("maturity_blockers"),
            "current_private_selected_pass_rate": current_guard_shape["summary"].get("private_selected_pass_rate"),
            "current_private_pass_if_any_rate": current_guard_shape["summary"].get("private_pass_if_any_rate"),
            "public_transfer_readiness": rel_or_abs(public_transfer_readiness_path),
            "public_transfer_readiness_state": public_transfer_readiness.get("trigger_state"),
            "public_transfer_readiness_private_evidence_current": public_transfer_private_evidence_current,
            "public_transfer_readiness_hard_failed_gate_count": public_transfer_summary.get("hard_failed_gate_count"),
            "public_transfer_readiness_evidence_failed_gate_count": public_transfer_summary.get("evidence_failed_gate_count"),
            "public_transfer_readiness_transfer_failed_gate_count": public_transfer_summary.get("transfer_failed_gate_count"),
            "public_transfer_failed_gate_names": public_transfer_failed_gate_names,
            "public_transfer_private_fix_review_ready": public_transfer_summary.get("private_fix_review_ready"),
            "public_transfer_calibration_review_ready_after_private_fixes": public_transfer_summary.get(
                "calibration_review_ready_after_private_fixes"
            ),
            "public_transfer_contract_task_count": public_transfer_summary.get("contract_task_count"),
            "public_transfer_full_body_selected_pass_rate": public_transfer_summary.get("full_body_selected_pass_rate"),
            "public_transfer_full_body_pass_if_any_rate": public_transfer_summary.get("full_body_pass_if_any_rate"),
            "public_transfer_full_body_benchmark_promotion_eligible_candidate_count": public_transfer_summary.get("full_body_benchmark_promotion_eligible_candidate_count"),
            "public_transfer_full_body_default_semantic_dead": public_transfer_summary.get("full_body_post_v4_default_semantic_dead"),
            "public_transfer_full_body_v4_full_body_selected_pass_rate": public_transfer_summary.get("full_body_v4_full_body_selected_pass_rate"),
            "public_transfer_full_body_v4_strict_novel_learned_only_pass_rate": public_transfer_summary.get("full_body_v4_strict_novel_learned_only_pass_rate"),
            "public_transfer_full_body_semantic_not_dead": public_transfer_semantic_not_dead,
            "alignment_preflight": rel_or_abs(alignment_preflight_path),
            "alignment_preflight_state": alignment_preflight.get("trigger_state"),
            "alignment_preflight_ready": alignment_preflight.get("alignment_preflight_ready"),
            "alignment_case_manifest": alignment_summary.get("case_manifest"),
            "alignment_case_manifest_row_count": alignment_summary.get("case_manifest_row_count"),
            "alignment_candidate_manifest_bound_to_case_manifest": alignment_summary.get(
                "candidate_manifest_bound_to_case_manifest"
            ),
            "current_public_shape_selected_obligation_satisfaction_rate": current_guard_shape["summary"].get("public_shape_selected_obligation_satisfaction_rate"),
            "current_public_shape_selected_return_compatible_rate": current_guard_shape["summary"].get("public_shape_selected_return_compatible_rate"),
            "vcm_runtime_state": current_guard_shape["summary"].get("vcm_runtime_state"),
            "vcm_native_runtime_claimable": current_guard_shape["summary"].get("vcm_native_runtime_claimable"),
            "macos_mlx_state": current_guard_shape["summary"].get("macos_mlx_state"),
            "macos_mlx_used": current_guard_shape["summary"].get("macos_mlx_used"),
            "teacher_distillation_fail_closed": current_guard_shape["summary"].get("teacher_distillation_fail_closed"),
            "teacher_rows_admitted": current_guard_shape["summary"].get("teacher_rows_admitted"),
            "training_data_admission_state": current_guard_shape["summary"].get("training_data_admission_state"),
            "v4_private_pass_rate": v4_score_summary.get("pass_rate"),
            "v4_private_pass_count": v4_score_summary.get("pass_count"),
            "v4_private_task_count": v4_score_summary.get("heldout_task_count"),
            "v4_learned_only_pass_rate": v4_gate_summary.get("learned_only_pass_rate"),
            "v4_learned_only_pass_count": v4_gate_summary.get("learned_only_pass_count"),
            "v4_learned_only_task_count": v4_gate_summary.get("learned_only_task_count"),
            "prototype_pass_count": v4_gate_summary.get("prototype_pass_count"),
            "learned_token_pass_count": v4_gate_summary.get("learned_token_pass_count"),
            "broad_public_pass_rate": pre_public_summary.get("public_pass_rate", readiness_summary.get("public_pass_rate")),
            "latest_public_calibration": rel_or_abs(latest_public_path),
            "latest_public_matrix": rel_or_abs(latest_matrix_path),
            "cards_below_floor": readiness_summary.get("cards_below_floor") or latest_matrix_summary.get("cards_below_floor") or [],
            "public_floor": pre_public_summary.get("public_floor", readiness_summary.get("public_floor")),
            "public_surface_card_count": len(public_cards),
            "public_surface_task_count": public_task_count,
            "public_card_counts": dict(sorted(manifest["card_counts"].items())),
            "latest_consumed_public_surface_task_count": public_task_count,
            "proposed_public_surface_slug": proposed_slug,
            "proposed_public_surface_task_count": next_surface.get("total_task_count"),
            "proposed_public_surface_seed": next_surface.get("seed"),
            "proposed_public_surface_cases_per_card": next_surface.get("cases_per_card"),
            "proposed_public_surface_cards": sorted(next_surface_cards),
            "public_tests_or_solutions_visible": not public_boundary_clean,
            "packet_does_not_run_public_calibration": packet_does_not_run_public,
            "proposed_slug": proposed_slug,
            "frozen_integrity_sha256": frozen_integrity["sha256"],
            "frozen_harness_hashes_current": frozen_integrity["harness_hashes_current"],
            "frozen_evidence_artifacts_hashable": frozen_integrity["evidence_artifacts_hashable"],
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "artifacts": {
            "post_distillation_readiness": rel_or_abs(readiness_path),
            "pre_public_audit": rel_or_abs(pre_public_audit_path),
            "frontier_expander": rel_or_abs(frontier_expander_path),
            "v4_score": rel_or_abs(v4_score_path),
            "v4_learned_gate": rel_or_abs(v4_gate_path),
            "wide_public_manifest": manifest,
            "latest_public_calibration": rel_or_abs(latest_public_path),
            "latest_public_matrix": rel_or_abs(latest_matrix_path),
            "public_benchmark_contract": rel_or_abs(public_contract_path),
            "capability_transfer_closure": rel_or_abs(capability_closure_path),
            "maturity_audit": rel_or_abs(maturity_audit_path),
            "public_transfer_readiness": rel_or_abs(public_transfer_readiness_path),
            "alignment_preflight": rel_or_abs(alignment_preflight_path),
            "private_admissibility": rel_or_abs(private_admissibility_path),
            "private_heldout": rel_or_abs(private_heldout_path),
            "private_shape_structure": rel_or_abs(private_shape_structure_path),
            "public_shape_structure": rel_or_abs(public_shape_structure_path),
            "public_shape_return": rel_or_abs(public_shape_return_path),
            "student_token_generator": rel_or_abs(student_token_generator_path),
            "vcm_release_conformance": rel_or_abs(vcm_release_path),
            "macos_mlx_smoke": rel_or_abs(macos_mlx_smoke_path),
            "teacher_gate": rel_or_abs(teacher_gate_path),
            "teacher_manifest_audit": rel_or_abs(teacher_manifest_path),
            "training_admission": rel_or_abs(training_admission_path),
        },
        "public_benchmark_contract": {
            "path": rel_or_abs(public_contract_path),
            "consumed_surface_do_not_rerun": consumed_surface,
            "stage_1_code_generation_surface": next_surface,
        },
        "frozen_integrity": frozen_integrity,
        "proposed_operator_actions": proposed,
        "gates": gates,
        "next_actions": post_distillation_next_actions(technical_ready, operator_lock_active, public_calibration_allowed),
        "rules": {
            "public_calibration": "this packet never launches calibration; operator approval must be explicit and bounded to one run",
            "operator_lock": "public_calibration_allowed remains false while reports/public_calibration_operator_lock.flag exists",
            "public_boundary": "public task IDs are calibration selectors only; do not train on public prompts, tests, solutions, traces, or scores",
            "frozen_integrity": "operator approval must include the packet hash and frozen_integrity_sha256 for this exact harness/evidence state",
            "score_semantics": "this packet is review readiness only, not a public benchmark score or promotion artifact",
        },
        "external_inference_calls": 0,
    }


def proposed_public_calibration_commands(
    args: argparse.Namespace,
    public_contract: dict[str, Any],
    slug: str,
) -> dict[str, Any]:
    next_surface = object_field(public_contract, "stage_1_code_generation_surface")
    command = list_value(next_surface.get("command_after_unlock_only"))
    command_text = " ".join(str(part) for part in command)
    return {
        "requires_explicit_operator_approval": True,
        "alignment_preflight": (
            "python3 scripts/public_calibration_alignment_preflight.py "
            "--out reports/public_calibration_alignment_preflight.json "
            "--markdown-out reports/public_calibration_alignment_preflight.md"
        ),
        "preflight_packet": (
            "python3 scripts/public_calibration_readiness_packet.py "
            "--mode post-distillation "
            "--out reports/public_calibration_readiness_packet.json "
            "--markdown-out reports/public_calibration_readiness_packet.md"
        ),
        "guarded_runner_dry_run": (
            "python3 scripts/operator_bounded_public_calibration.py "
            "--out reports/operator_bounded_public_calibration_dry_run.json "
            "--markdown-out reports/operator_bounded_public_calibration_dry_run.md"
        ),
        "guarded_runner_execute_after_approval_only": (
            "python3 scripts/operator_bounded_public_calibration.py --execute "
            f"--out reports/operator_bounded_public_calibration_{slug}.json "
            f"--markdown-out reports/operator_bounded_public_calibration_{slug}.md"
        ),
        "calibration_command_after_unlock_only": command_text,
        "mandatory_after_run": [
            "restore reports/public_calibration_operator_lock.flag immediately",
            "run broad transfer matrix and residual report for the new calibration artifact",
            "do not train on the public traces, tests, solutions, prompts, or score labels",
        ],
    }


def build_frozen_integrity_snapshot(
    *,
    public_contract: dict[str, Any],
    evidence_paths: dict[str, Path],
) -> dict[str, Any]:
    contract_snapshot = object_field(public_contract, "contract_snapshot")
    expected_harness = object_field(contract_snapshot, "harness_files_sha256")
    harness_files: dict[str, Any] = {}
    mismatches: list[dict[str, Any]] = []
    missing_harness: list[str] = []

    for raw_path, expected_hash in sorted(expected_harness.items()):
        path = resolve(raw_path)
        observed_hash = sha256_file(path)
        exists = path.exists() and path.is_file()
        matched = bool(exists and observed_hash and observed_hash == str(expected_hash))
        row = {
            "path": rel_or_abs(path),
            "exists": exists,
            "expected_sha256": str(expected_hash),
            "observed_sha256": observed_hash,
            "matched": matched,
            "size_bytes": file_size(path),
        }
        harness_files[str(raw_path)] = row
        if not exists:
            missing_harness.append(str(raw_path))
        elif not matched:
            mismatches.append(row)

    evidence_hashes: dict[str, Any] = {}
    missing_evidence: list[str] = []
    for name, path in sorted(evidence_paths.items()):
        exists = path.exists() and path.is_file()
        row = {
            "path": rel_or_abs(path),
            "exists": exists,
            "sha256": sha256_file(path),
            "size_bytes": file_size(path),
        }
        evidence_hashes[name] = row
        if not exists:
            missing_evidence.append(name)

    stable_payload = {
        "policy": "project_theseus_public_calibration_frozen_integrity_v1",
        "contract_status": public_contract.get("status"),
        "contract_head": object_field(public_contract, "contract_snapshot").get("theseus_git_head"),
        "consumed_surface": object_field(object_field(public_contract, "global_rules"), "consumed_surface_do_not_rerun"),
        "next_surface": object_field(public_contract, "stage_1_code_generation_surface"),
        "harness_files": harness_files,
        "evidence_hashes": evidence_hashes,
    }
    integrity_sha = hashlib.sha256(stable_json(stable_payload).encode("utf-8")).hexdigest()
    return {
        **stable_payload,
        "sha256": integrity_sha,
        "harness_hashes_current": not mismatches and not missing_harness and bool(expected_harness),
        "harness_mismatch_count": len(mismatches),
        "harness_missing_count": len(missing_harness),
        "harness_mismatches": mismatches,
        "harness_missing": missing_harness,
        "evidence_artifacts_hashable": not missing_evidence,
        "evidence_missing_count": len(missing_evidence),
        "evidence_missing": missing_evidence,
    }


def post_distillation_next_actions(
    technical_ready: bool,
    operator_lock_active: bool,
    public_calibration_allowed: bool,
) -> list[str]:
    if not technical_ready:
        return [
            "Fix packet gates before any operator decision; do not run public calibration from incomplete review evidence.",
            "Keep public calibration locked.",
        ]
    if operator_lock_active:
        return [
            "Operator packet is technically ready for review; public calibration remains locked.",
            "If explicitly approved, run exactly one bounded seed14 5x64 public calibration, then relock immediately.",
            "If not approved, continue private-only curriculum expansion and do not spend another public run.",
        ]
    if public_calibration_allowed:
        return ["Run exactly one bounded seed14 5x64 public calibration, then relock immediately."]
    return ["Keep public calibration locked."]


def failed_transfer_gate_names(public_transfer_readiness: dict[str, Any]) -> list[str]:
    transfer = object_field(public_transfer_readiness, "gates").get("transfer")
    if not isinstance(transfer, list):
        return []
    names: list[str] = []
    for row in transfer:
        if isinstance(row, dict) and row.get("passed") is not True:
            names.append(str(row.get("gate") or "unknown"))
    return names


def alignment_preflight_ready(report: dict[str, Any]) -> bool:
    summary = object_field(report, "summary")
    return bool(
        report.get("policy") == "project_theseus_public_calibration_alignment_preflight_v1"
        and report.get("trigger_state") in {"GREEN", "YELLOW"}
        and report.get("alignment_preflight_ready") is True
        and int(first_number(summary.get("case_manifest_row_count"))) == 320
        and summary.get("candidate_manifest_bound_to_case_manifest") is True
        and summary.get("candidate_manifest_preexists_before_run") is False
        and summary.get("public_tests_used") is False
        and summary.get("public_solutions_used") is False
        and int(first_number(summary.get("training_rows_written"))) == 0
        and int(first_number(report.get("external_inference_calls"), summary.get("external_inference_calls"))) == 0
    )


def current_guard_shape_readiness(
    *,
    capability_closure: dict[str, Any],
    maturity_audit: dict[str, Any],
    private_admissibility: dict[str, Any],
    private_heldout: dict[str, Any],
    private_shape_structure: dict[str, Any],
    public_shape_structure: dict[str, Any],
    public_shape_return: dict[str, Any],
    student_token_generator: dict[str, Any],
    vcm_release: dict[str, Any],
    macos_mlx_smoke: dict[str, Any],
    teacher_gate: dict[str, Any],
    teacher_manifest: dict[str, Any],
    training_admission: dict[str, Any],
) -> dict[str, Any]:
    closure_summary = object_field(capability_closure, "summary")
    closure_no_cheat = object_field(closure_summary, "no_cheat")
    public_coverage = object_field(closure_summary, "public_candidate_coverage")
    maturity_summary = object_field(maturity_audit, "summary")
    private_admissibility_summary = object_field(private_admissibility, "summary")
    private_heldout_summary = object_field(private_heldout, "summary")
    private_shape_summary = object_field(private_shape_structure, "summary")
    public_shape_summary = object_field(public_shape_structure, "summary")
    public_return_summary = object_field(public_shape_return, "summary")
    generator_summary = object_field(student_token_generator, "summary")
    vcm_summary = object_field(vcm_release, "summary")
    vcm_profiles = object_field(vcm_summary, "profile_states")
    mlx_summary = object_field(macos_mlx_smoke, "summary")
    teacher_share = object_field(teacher_gate, "teacher_share")
    teacher_manifest_summary = object_field(teacher_manifest, "summary")
    teacher_admission_checks = object_field(teacher_manifest_summary, "admission_checks")
    training_admission_summary = object_field(training_admission, "summary")
    private_task_count = int(first_number(private_admissibility_summary.get("task_count")))
    private_candidate_count = int(first_number(private_admissibility_summary.get("candidate_row_count")))
    private_min_candidate_count = private_task_count * 4 if private_task_count > 0 else 0
    public_task_count = int(first_number(generator_summary.get("task_count")))
    public_candidate_count = int(first_number(generator_summary.get("candidate_count")))
    public_min_candidate_count = public_task_count * 4 if public_task_count > 0 else 0

    no_cheat_counts = {
        "closure_external_inference_calls": int(first_number(closure_no_cheat.get("external_inference_calls"))),
        "closure_fallback_return_count": int(first_number(closure_no_cheat.get("fallback_return_count"))),
        "closure_public_leakage_count": int(first_number(closure_no_cheat.get("public_leakage_count"))),
        "closure_public_training_rows": int(first_number(closure_no_cheat.get("public_training_rows"))),
        "closure_teacher_used_count": int(first_number(closure_no_cheat.get("teacher_used_count"))),
        "closure_template_like_count": int(first_number(closure_no_cheat.get("template_like_count"))),
        "coverage_external_inference_calls": int(first_number(public_coverage.get("external_inference_calls"))),
        "generator_external_inference_calls": int(first_number(generator_summary.get("external_inference_calls"))),
        "generator_expression_memory_fallback_count": int(first_number(generator_summary.get("expression_memory_fallback_count"))),
        "generator_template_like_candidate_count": int(first_number(generator_summary.get("template_like_candidate_count"))),
        "private_admissibility_external_inference_calls": int(first_number(private_admissibility_summary.get("external_inference_calls"))),
        "private_admissibility_fallback_return_candidate_count": int(first_number(private_admissibility_summary.get("fallback_return_candidate_count"))),
        "private_admissibility_template_like_candidate_count": int(first_number(private_admissibility_summary.get("template_like_candidate_count"))),
        "private_heldout_external_inference_calls": int(first_number(private_heldout_summary.get("external_inference_calls"))),
        "private_heldout_fallback_return_candidate_count": int(first_number(private_heldout_summary.get("fallback_return_candidate_count"))),
        "private_shape_external_inference_calls": int(first_number(private_shape_summary.get("external_inference_calls"))),
        "public_shape_external_inference_calls": int(first_number(public_shape_summary.get("external_inference_calls"))),
        "public_return_external_inference_calls": int(first_number(public_return_summary.get("external_inference_calls"))),
        "vcm_external_inference_calls": int(first_number(vcm_summary.get("external_inference_calls"))),
        "vcm_fallback_return_count": int(first_number(vcm_summary.get("fallback_return_count"))),
        "mlx_external_inference_calls": int(first_number(mlx_summary.get("external_inference_calls"))),
        "mlx_fallback_return_rows": int(first_number(mlx_summary.get("fallback_return_rows"))),
        "teacher_gate_external_inference_calls": int(first_number(teacher_gate.get("external_inference_calls"))),
        "teacher_manifest_external_inference_calls": int(first_number(teacher_manifest_summary.get("external_inference_calls"))),
        "teacher_rows_admitted": int(first_number(teacher_manifest_summary.get("teacher_rows_admitted"))),
        "teacher_public_training_rows_written": int(first_number(teacher_manifest_summary.get("public_training_rows_written"))),
        "training_admission_external_inference_calls": int(first_number(training_admission_summary.get("external_inference_calls"))),
    }
    teacher_training_cost_counts = {
        "teacher_gate_external_inference_calls": no_cheat_counts["teacher_gate_external_inference_calls"],
        "teacher_manifest_external_inference_calls": no_cheat_counts["teacher_manifest_external_inference_calls"],
        "teacher_rows_admitted": no_cheat_counts["teacher_rows_admitted"],
    }
    no_cheat_hard_counts = {
        key: value
        for key, value in no_cheat_counts.items()
        if key not in teacher_training_cost_counts
    }
    no_cheat_zero = all(value == 0 for value in no_cheat_hard_counts.values())

    closure_ready = (
        capability_closure.get("policy") == "project_theseus_capability_transfer_closure_v1"
        and capability_closure.get("trigger_state") == "YELLOW"
        and closure_summary.get("private_semantic_ready") is True
        and closure_summary.get("public_candidate_coverage_ready") is True
        and closure_summary.get("no_cheat_clean") is True
        and closure_summary.get("public_promotion_ready") is False
        and first_number(closure_summary.get("private_selected_pass_rate")) >= 1.0
        and first_number(closure_summary.get("private_pass_if_any_rate")) >= 1.0
    )
    maturity_blockers = [str(item) for item in list_value(maturity_audit.get("maturity_blockers"))]
    maturity_ready = (
        maturity_audit.get("policy") == "project_theseus_maturity_integrity_audit_v1"
        and maturity_audit.get("trigger_state") == "YELLOW"
        and int(first_number(maturity_summary.get("hard_blocker_count"))) == 0
        and int(first_number(maturity_summary.get("evidence_blocker_count"))) == 0
        and maturity_blockers == ["public_transfer_floor_cleared"]
        and maturity_summary.get("candidate_promotion_allowed") is False
        and maturity_summary.get("model_growth_allowed") is False
        and maturity_summary.get("public_calibration_allowed") is False
    )
    private_candidate_ready = (
        private_admissibility.get("policy") == "project_theseus_private_full_body_candidate_admissibility_gate_v1"
        and private_admissibility.get("trigger_state") == "GREEN"
        and first_number(private_admissibility_summary.get("selected_pass_rate")) >= 1.0
        and first_number(private_admissibility_summary.get("pass_if_any_rate")) >= 1.0
        and first_number(private_admissibility_summary.get("no_admissible_task_rate")) == 0.0
        and private_task_count >= 240
        and private_candidate_count >= private_min_candidate_count
        and int(first_number(private_admissibility_summary.get("full_body_token_candidate_count"))) >= private_candidate_count
        and int(first_number(private_admissibility_summary.get("learned_token_candidate_count"))) >= private_candidate_count
    )
    private_heldout_ready = (
        private_heldout.get("policy") == "project_theseus_private_residual_repair_v3_heldout_score_v1"
        and private_heldout.get("trigger_state") == "GREEN"
        and private_heldout_summary.get("adapter_off_scoring") is True
        and first_number(private_heldout_summary.get("learned_candidate_task_pass_rate")) >= 1.0
        and first_number(private_heldout_summary.get("private_residual_v3_heldout_pass_rate")) >= 1.0
        and first_number(private_heldout_summary.get("private_residual_v3_sts_delta")) == 0.0
        and private_heldout_summary.get("private_residual_v3_sts_lift_claim_allowed") is False
        and private_heldout_summary.get("public_tests_used") is False
        and private_heldout_summary.get("public_solutions_used") is False
    )
    private_shape_ready = (
        private_shape_structure.get("policy") == "project_theseus_public_shape_semantic_structure_audit_v1"
        and private_shape_structure.get("trigger_state") == "GREEN"
        and first_number(private_shape_summary.get("selected_obligation_satisfaction_rate")) >= 1.0
        and first_number(private_shape_summary.get("selected_task_full_obligation_rate")) >= 1.0
        and int(first_number(private_shape_summary.get("fragmented_candidate_union_only_task_count"))) == 0
    )
    public_shape_ready = (
        public_shape_structure.get("policy") == "project_theseus_public_shape_semantic_structure_audit_v1"
        and public_shape_structure.get("trigger_state") == "GREEN"
        and first_number(public_shape_summary.get("candidate_ast_parse_rate")) >= 1.0
        and first_number(public_shape_summary.get("selected_obligation_satisfaction_rate")) >= 0.99
        and first_number(public_shape_summary.get("selected_task_full_obligation_rate")) >= 0.97
        and first_number(public_shape_summary.get("multi_statement_generated_body_candidate_rate")) >= 0.80
    )
    public_return_ready = (
        public_shape_return.get("policy") == "project_theseus_public_shape_return_contract_audit_v1"
        and public_shape_return.get("trigger_state") == "GREEN"
        and first_number(public_return_summary.get("candidate_ast_parse_rate")) >= 1.0
        and first_number(public_return_summary.get("selected_shape_compatible_task_rate")) >= 1.0
        and first_number(public_return_summary.get("expected_shape_coverage_rate")) >= 0.85
    )
    generator_ready = (
        student_token_generator.get("policy") == "project_theseus_student_token_code_generator_v1"
        and student_token_generator.get("trigger_state") == "GREEN"
        and generator_summary.get("token_level_code_generation_learned") is True
        and generator_summary.get("canonical_solution_seen_by_solver") is False
        and generator_summary.get("public_tests_visible_to_generator") is False
        and public_task_count >= 160
        and public_candidate_count >= public_min_candidate_count
        and int(first_number(generator_summary.get("full_body_token_candidate_count"))) >= public_candidate_count
        and int(first_number(generator_summary.get("grammar_masked_learned_token_candidate_count"))) >= public_candidate_count
    )
    vcm_ready = (
        vcm_release.get("policy") == "project_theseus_vcm_release_conformance_audit_v1"
        and vcm_release.get("trigger_state") == "GREEN"
        and vcm_profiles.get("VCM-Runtime") == "GREEN"
        and vcm_summary.get("native_prefix_kv_lifecycle_test_passed") is True
        and vcm_summary.get("native_runtime_claimable") is True
        and vcm_summary.get("native_runtime_route_metadata_ready") is True
        and int(first_number(vcm_summary.get("native_runtime_blocker_count"))) == 0
    )
    mlx_ready = (
        macos_mlx_smoke.get("policy") == "project_theseus_macos_mlx_structural_action_smoke_v0"
        and macos_mlx_smoke.get("trigger_state") == "GREEN"
        and mlx_summary.get("mlx_available") is True
        and mlx_summary.get("mlx_used") is True
        and first_number(mlx_summary.get("verifier_pass_rate")) >= 1.0
    )
    teacher_fail_closed = (
        teacher_gate.get("policy") == "project_theseus_teacher_distillation_gate_v0"
        and teacher_gate.get("trigger_state") == "YELLOW"
        and teacher_gate.get("distillation_allowed") is False
        and not list_value(teacher_gate.get("hard_blockers"))
        and int(first_number(teacher_share.get("teacher_accepted_rows"))) == 0
        and first_number(teacher_share.get("teacher_accepted_row_share")) == 0.0
    )
    teacher_governed_training_ready = (
        teacher_gate.get("policy") == "project_theseus_teacher_distillation_gate_v0"
        and teacher_gate.get("trigger_state") == "GREEN"
        and teacher_gate.get("distillation_allowed") is True
        and not list_value(teacher_gate.get("hard_blockers"))
        and int(first_number(teacher_share.get("teacher_accepted_rows"))) >= 1
        and first_number(teacher_share.get("teacher_accepted_row_share")) <= first_number(
            teacher_share.get("max_initial_training_ratio"), 1.0
        )
    )
    teacher_manifest_proposal_only = (
        teacher_manifest.get("policy") == "project_theseus_teacher_distillation_manifest_builder_v0"
        and teacher_manifest.get("trigger_state") == "YELLOW"
        and teacher_manifest_summary.get("manifest_ready_for_distillation") is False
        and int(first_number(teacher_manifest_summary.get("row_count"))) == 0
        and int(first_number(teacher_manifest_summary.get("teacher_rows_admitted"))) == 0
        and int(first_number(teacher_manifest_summary.get("public_overlap_hits"))) == 0
        and int(first_number(teacher_manifest_summary.get("holdout_overlap_hits"))) == 0
        and teacher_admission_checks.get("provenance_retained") is True
        and teacher_admission_checks.get("license_checked") is True
        and teacher_admission_checks.get("leakage_audited") is True
        and teacher_admission_checks.get("public_benchmark_excluded") is True
        and teacher_admission_checks.get("runtime_serving_forbidden") is True
    )
    teacher_manifest_governed_training_ready = (
        teacher_manifest.get("policy") == "project_theseus_teacher_distillation_manifest_builder_v0"
        and teacher_manifest.get("trigger_state") == "GREEN"
        and teacher_manifest_summary.get("manifest_ready_for_distillation") is True
        and int(first_number(teacher_manifest_summary.get("row_count"))) >= 1
        and int(first_number(teacher_manifest_summary.get("teacher_rows_admitted"))) >= 1
        and int(first_number(teacher_manifest_summary.get("public_overlap_hits"))) == 0
        and int(first_number(teacher_manifest_summary.get("holdout_overlap_hits"))) == 0
        and int(first_number(teacher_manifest_summary.get("public_training_rows_written"))) == 0
        and teacher_admission_checks.get("provenance_retained") is True
        and teacher_admission_checks.get("license_checked") is True
        and teacher_admission_checks.get("leakage_audited") is True
        and teacher_admission_checks.get("verifier_accepted") is True
        and teacher_admission_checks.get("public_benchmark_excluded") is True
        and teacher_admission_checks.get("runtime_serving_forbidden") is True
    )
    training_admission_clean = (
        training_admission.get("policy") == "project_theseus_training_data_admission_v1"
        and training_admission.get("trigger_state") in {"GREEN", "YELLOW"}
        and int(first_number(training_admission_summary.get("allowed_training_source_count"))) > 0
        and training_admission_summary.get("public_benchmark_payload_admitted") is False
        and training_admission_summary.get("public_benchmark_training_allowed") is False
        and training_admission_summary.get("teacher_used") is False
    )
    public_boundary_clean = (
        no_cheat_zero
        and not any_true("public_tests_used", private_heldout)
        and not any_true("public_solutions_used", private_heldout)
        and not any_true("public_tests_visible_to_generator", student_token_generator)
        and not any_true("canonical_solution_seen_by_solver", student_token_generator)
    )

    checks = [
        gate("current_closure_private_semantic_and_coverage_ready", closure_ready, {
            "trigger_state": capability_closure.get("trigger_state"),
            "private_semantic_ready": closure_summary.get("private_semantic_ready"),
            "public_candidate_coverage_ready": closure_summary.get("public_candidate_coverage_ready"),
            "private_selected_pass_rate": closure_summary.get("private_selected_pass_rate"),
            "private_pass_if_any_rate": closure_summary.get("private_pass_if_any_rate"),
            "public_promotion_ready": closure_summary.get("public_promotion_ready"),
        }),
        gate("maturity_has_only_public_transfer_floor_wall", maturity_ready, {
            "trigger_state": maturity_audit.get("trigger_state"),
            "hard_blocker_count": maturity_summary.get("hard_blocker_count"),
            "evidence_blocker_count": maturity_summary.get("evidence_blocker_count"),
            "maturity_blockers": maturity_blockers,
            "candidate_promotion_allowed": maturity_summary.get("candidate_promotion_allowed"),
            "model_growth_allowed": maturity_summary.get("model_growth_allowed"),
        }),
        gate("private_full_body_admissibility_green", private_candidate_ready, {
            "selected_pass_rate": private_admissibility_summary.get("selected_pass_rate"),
            "pass_if_any_rate": private_admissibility_summary.get("pass_if_any_rate"),
            "no_admissible_task_rate": private_admissibility_summary.get("no_admissible_task_rate"),
            "task_count": private_task_count,
            "candidate_row_count": private_candidate_count,
            "minimum_candidate_row_count": private_min_candidate_count,
            "full_body_token_candidate_count": private_admissibility_summary.get("full_body_token_candidate_count"),
            "learned_token_candidate_count": private_admissibility_summary.get("learned_token_candidate_count"),
        }),
        gate("private_heldout_adapter_off_semantic_green", private_heldout_ready, {
            "learned_candidate_task_pass_rate": private_heldout_summary.get("learned_candidate_task_pass_rate"),
            "private_residual_v3_heldout_pass_rate": private_heldout_summary.get("private_residual_v3_heldout_pass_rate"),
            "private_residual_v3_sts_delta": private_heldout_summary.get("private_residual_v3_sts_delta"),
            "private_residual_v3_sts_lift_claim_allowed": private_heldout_summary.get("private_residual_v3_sts_lift_claim_allowed"),
        }),
        gate("private_shape_structure_green", private_shape_ready, {
            "selected_obligation_satisfaction_rate": private_shape_summary.get("selected_obligation_satisfaction_rate"),
            "selected_task_full_obligation_rate": private_shape_summary.get("selected_task_full_obligation_rate"),
            "fragmented_candidate_union_only_task_count": private_shape_summary.get("fragmented_candidate_union_only_task_count"),
        }),
        gate("public_shape_prompt_only_structure_ready", public_shape_ready, {
            "selected_obligation_satisfaction_rate": public_shape_summary.get("selected_obligation_satisfaction_rate"),
            "selected_task_full_obligation_rate": public_shape_summary.get("selected_task_full_obligation_rate"),
            "multi_statement_generated_body_candidate_rate": public_shape_summary.get("multi_statement_generated_body_candidate_rate"),
        }),
        gate("public_shape_return_contract_ready", public_return_ready, {
            "selected_shape_compatible_task_rate": public_return_summary.get("selected_shape_compatible_task_rate"),
            "expected_shape_coverage_rate": public_return_summary.get("expected_shape_coverage_rate"),
            "candidate_ast_parse_rate": public_return_summary.get("candidate_ast_parse_rate"),
        }),
        gate("student_token_generator_full_body_clean", generator_ready, {
            "candidate_generation_mode": generator_summary.get("candidate_generation_mode"),
            "task_count": public_task_count,
            "candidate_count": public_candidate_count,
            "minimum_candidate_count": public_min_candidate_count,
            "full_body_token_candidate_count": generator_summary.get("full_body_token_candidate_count"),
            "grammar_masked_learned_token_candidate_count": generator_summary.get("grammar_masked_learned_token_candidate_count"),
            "public_tests_visible_to_generator": generator_summary.get("public_tests_visible_to_generator"),
            "canonical_solution_seen_by_solver": generator_summary.get("canonical_solution_seen_by_solver"),
        }),
        gate("vcm_runtime_native_lifecycle_ready", vcm_ready, {
            "profile_states": vcm_profiles,
            "native_prefix_kv_lifecycle_test_passed": vcm_summary.get("native_prefix_kv_lifecycle_test_passed"),
            "native_runtime_claimable": vcm_summary.get("native_runtime_claimable"),
            "native_runtime_recommended_backend": vcm_summary.get("native_runtime_recommended_backend"),
        }),
        gate("macos_mlx_route_green", mlx_ready, {
            "mlx_available": mlx_summary.get("mlx_available"),
            "mlx_used": mlx_summary.get("mlx_used"),
            "mlx_default_device": mlx_summary.get("mlx_default_device"),
            "verifier_pass_rate": mlx_summary.get("verifier_pass_rate"),
            "fallback_return_rows": mlx_summary.get("fallback_return_rows"),
        }),
        gate("teacher_distillation_governed_state_ready", teacher_fail_closed or teacher_governed_training_ready, {
            "trigger_state": teacher_gate.get("trigger_state"),
            "distillation_allowed": teacher_gate.get("distillation_allowed"),
            "hard_blockers": list_value(teacher_gate.get("hard_blockers")),
            "teacher_accepted_rows": teacher_share.get("teacher_accepted_rows"),
            "teacher_accepted_row_share": teacher_share.get("teacher_accepted_row_share"),
            "fail_closed": teacher_fail_closed,
            "governed_training_ready": teacher_governed_training_ready,
        }),
        gate("teacher_manifest_training_rows_governed", teacher_manifest_proposal_only or teacher_manifest_governed_training_ready, {
            "manifest_ready_for_distillation": teacher_manifest_summary.get("manifest_ready_for_distillation"),
            "proposal_rows_retained_not_training": teacher_manifest_summary.get("proposal_rows_retained_not_training"),
            "teacher_rows_admitted": teacher_manifest_summary.get("teacher_rows_admitted"),
            "public_training_rows_written": teacher_manifest_summary.get("public_training_rows_written"),
            "admission_checks": teacher_admission_checks,
            "proposal_only": teacher_manifest_proposal_only,
            "governed_training_ready": teacher_manifest_governed_training_ready,
        }),
        gate("training_data_admission_hard_clean", training_admission_clean, {
            "trigger_state": training_admission.get("trigger_state"),
            "allowed_training_source_count": training_admission_summary.get("allowed_training_source_count"),
            "public_benchmark_payload_admitted": training_admission_summary.get("public_benchmark_payload_admitted"),
            "public_benchmark_training_allowed": training_admission_summary.get("public_benchmark_training_allowed"),
            "teacher_used": training_admission_summary.get("teacher_used"),
        }),
        gate("current_no_cheat_counters_zero", no_cheat_zero and public_boundary_clean, {
            "hard_counts": no_cheat_hard_counts,
            "teacher_training_cost_counts": teacher_training_cost_counts,
        }),
    ]
    passed = all(row["passed"] for row in checks)
    return {
        "passed": passed,
        "summary": {
            "private_semantic_ready": closure_summary.get("private_semantic_ready"),
            "public_candidate_coverage_ready": closure_summary.get("public_candidate_coverage_ready"),
            "private_selected_pass_rate": closure_summary.get("private_selected_pass_rate"),
            "private_pass_if_any_rate": closure_summary.get("private_pass_if_any_rate"),
            "maturity_blockers": maturity_blockers,
            "public_shape_selected_obligation_satisfaction_rate": public_shape_summary.get("selected_obligation_satisfaction_rate"),
            "public_shape_selected_return_compatible_rate": public_return_summary.get("selected_shape_compatible_task_rate"),
            "vcm_runtime_state": vcm_profiles.get("VCM-Runtime"),
            "vcm_native_runtime_claimable": vcm_summary.get("native_runtime_claimable"),
            "macos_mlx_state": macos_mlx_smoke.get("trigger_state"),
            "macos_mlx_used": mlx_summary.get("mlx_used"),
            "teacher_distillation_fail_closed": teacher_fail_closed,
            "teacher_governed_training_ready": teacher_governed_training_ready,
            "teacher_rows_admitted": teacher_manifest_summary.get("teacher_rows_admitted"),
            "training_data_admission_state": training_admission.get("trigger_state"),
            "no_cheat_zero": no_cheat_zero and public_boundary_clean,
            "teacher_training_external_inference_calls": teacher_training_cost_counts[
                "teacher_gate_external_inference_calls"
            ],
        },
        "checks": checks,
        "no_cheat_counts": no_cheat_counts,
        "no_cheat_hard_counts": no_cheat_hard_counts,
        "teacher_training_cost_counts": teacher_training_cost_counts,
        "rule": "Current guard/shape evidence must be bound into the one-shot packet hash before any operator approval. This does not clear the public floor or execute calibration.",
    }


def expected_bounded_public_surface(train_once: dict[str, Any], closure: dict[str, Any]) -> set[str]:
    for candidate in [
        get_path(train_once, ["effective_public_surface", "cards"], []),
        get_path(train_once, ["summary", "effective_public_surface", "cards"], []),
        get_path(closure, ["effective_public_surface", "cards"], []),
        get_path(closure, ["summary", "effective_public_surface", "cards"], []),
    ]:
        if isinstance(candidate, list):
            cards = {str(item) for item in candidate if str(item)}
            if len(cards) in {4, 5}:
                return cards
    return set()


def accepted_bounded_public_surfaces(expected_surface: set[str]) -> list[set[str]]:
    accepted = [set(surface) for surface in DEFAULT_ACCEPTED_BOUNDED_PUBLIC_SURFACES]
    if len(expected_surface) in {4, 5} and expected_surface not in accepted:
        accepted.append(set(expected_surface))
    return accepted


def scan_rows(path: Path, *, scope: str) -> dict[str, Any]:
    card_counts: Counter[str] = Counter()
    leak_hits: list[dict[str, Any]] = []
    broad_floor_count = 0
    row_count = 0
    if not path.exists() or not path.is_file():
        return {
            "scope": scope,
            "path": rel_or_abs(path),
            "exists": False,
            "row_count": 0,
            "card_counts": {},
            "public_leak_hit_count": 0,
            "public_leak_hits": [],
            "broad_floor_recovery_row_count": 0,
        }
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            row_count += 1
            card = str(row.get("card_id") or "")
            if card:
                card_counts[card] += 1
            if row_is_broad_floor_recovery(row):
                broad_floor_count += 1
            reasons = public_leak_reasons(row, scope=scope)
            if reasons and len(leak_hits) < 20:
                leak_hits.append(
                    {
                        "line": line_no,
                        "task_id": row.get("task_id"),
                        "scope": scope,
                        "reasons": reasons,
                    }
                )
    return {
        "scope": scope,
        "path": rel_or_abs(path),
        "exists": True,
        "row_count": row_count,
        "card_counts": dict(card_counts),
        "public_leak_hit_count": len(leak_hits),
        "public_leak_hits": leak_hits,
        "broad_floor_recovery_row_count": broad_floor_count,
        "mtime": path_mtime(path),
    }


def row_is_broad_floor_recovery(row: dict[str, Any]) -> bool:
    tags = [str(item) for item in row.get("tags") or []]
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    return (
        "broad_public_code_transfer_floor_recovery_v1" in tags
        or "private_only_floor_recovery" in tags
        or row.get("source_id") == "local_generated_broad_public_floor_recovery_private_pressure"
        or "broad_public_code_transfer_floor_recovery_v1" in str(row.get("high_transfer_source_jsonl") or "")
        or provenance.get("policy") == "project_theseus_broad_public_code_transfer_floor_recovery_v1"
    )


def public_leak_reasons(row: dict[str, Any], *, scope: str) -> list[str]:
    reasons: list[str] = []
    bool_fields = [
        "public_tests_used",
        "public_solutions_used",
        "canonical_solution_used",
        "canonical_solution_seen_by_solver",
        "public_tests_visible",
    ]
    for field in bool_fields:
        if row.get(field) is True:
            reasons.append(field)
    if scope != "public_manifest":
        for field in ["tests_exported", "canonical_solution_exported"]:
            if row.get(field) is True:
                reasons.append(field)
    contract = row.get("decoder_contract") if isinstance(row.get("decoder_contract"), dict) else {}
    if contract.get("public_tests_used") is True:
        reasons.append("decoder_contract.public_tests_used")
    if contract.get("public_solutions_used") is True:
        reasons.append("decoder_contract.public_solutions_used")
    plan = contract.get("generation_plan") if isinstance(contract.get("generation_plan"), dict) else {}
    if plan.get("public_tests_used") is True:
        reasons.append("decoder_contract.generation_plan.public_tests_used")
    if plan.get("public_solutions_used") is True:
        reasons.append("decoder_contract.generation_plan.public_solutions_used")
    return reasons


def canonical_artifact_check(
    *,
    canonical_slug: str,
    stale_slugs: set[str],
    paths: dict[str, Path],
    referenced_paths: dict[str, str],
) -> dict[str, Any]:
    stale_hits: list[dict[str, str]] = []
    noncanonical_hits: list[dict[str, str]] = []
    inspected: dict[str, str] = {}
    for name, path in paths.items():
        value = rel_or_abs(path)
        inspected[name] = value
        if any(stale in value for stale in stale_slugs):
            stale_hits.append({"name": name, "path": value})
        if "broad_floor" in value and canonical_slug not in value:
            noncanonical_hits.append({"name": name, "path": value})
    for name, value in referenced_paths.items():
        text = str(value or "")
        inspected[name] = text
        if any(stale in text for stale in stale_slugs):
            stale_hits.append({"name": name, "path": text})
        if "broad_floor" in text and canonical_slug not in text:
            noncanonical_hits.append({"name": name, "path": text})
    return {
        "passed": not stale_hits and not noncanonical_hits,
        "canonical_slug": canonical_slug,
        "stale_slugs": sorted(stale_slugs),
        "stale_artifact_count": len(stale_hits),
        "noncanonical_artifact_count": len(noncanonical_hits),
        "stale_artifacts": stale_hits,
        "noncanonical_artifacts": noncanonical_hits,
        "inspected_paths": inspected,
        "rule": "broad-floor governance may read historical broad-floor reports only as explicit baselines, never as the canonical current closure",
    }


def infer_closure_report_path(raw_closure_report: str, train_once_report_path: Path) -> Path:
    requested = str(raw_closure_report or "").strip()
    if requested:
        return resolve(requested)
    train_once = read_json(train_once_report_path, {})
    inferred = str(
        get_path(train_once, ["paths", "closure_report"], "")
        or get_path(train_once, ["summary", "closure_report"], "")
        or ""
    )
    if inferred:
        return resolve(inferred)
    return resolve(f"reports/code_lm_closure_{DEFAULT_SLUG}.json")


def closure_slug(path: Path) -> str:
    stem = path.stem
    prefix = "code_lm_closure_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


def next_actions(
    technical_ready: bool,
    operator_lock_active: bool,
    transfer_proof_fresh: bool,
    decoder_gate_fresh: bool,
    source_fresh: bool,
    operator_lock_text: str = "",
) -> list[str]:
    if not source_fresh:
        return [
            "Regenerate the private closure, decoder gate, transfer proof, and readiness packet after the latest decoder source/release build.",
            "Do not run public calibration from stale candidate artifacts.",
        ]
    if not decoder_gate_fresh:
        return ["Run decoder_v2_private_ablation_gate.py against the fresh broad-floor closure report."]
    if not transfer_proof_fresh:
        return ["Run private_public_transfer_proof.py against the fresh decoder gate and broad-floor baseline."]
    if technical_ready and operator_lock_active:
        if "completed" in operator_lock_text.lower():
            return [
                "The bounded public calibration has already been consumed and relocked.",
                "Do not run another public calibration on this surface; continue private-only residual repair until broad transfer and coherence clear.",
            ]
        return [
            "Technical readiness is present, but public calibration is still operator-locked.",
            "If explicitly approved, run exactly one bounded public calibration, then immediately relock.",
        ]
    if technical_ready:
        return ["Run exactly one bounded public calibration, then immediately relock and refresh broad_transfer_matrix."]
    return ["Finish the fresh private closure, decoder gate, and transfer proof before any public calibration."]


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    if report.get("mode") == "post_distillation_v4_operator_review":
        lines = [
            "# Public Calibration Readiness Packet",
            "",
            f"- Trigger state: `{report.get('trigger_state')}`",
            f"- Technical ready: `{report.get('technical_ready_for_one_bounded_public_calibration')}`",
            f"- Public calibration allowed: `{report.get('public_calibration_allowed')}`",
            f"- Operator lock active: `{report.get('operator_lock_active')}`",
            f"- Public surface: `{summary.get('public_surface_card_count')}` cards / `{summary.get('public_surface_task_count')}` tasks",
            f"- Latest public pass rate: `{summary.get('broad_public_pass_rate')}`",
            f"- Public floor: `{summary.get('public_floor')}`",
            f"- Cards below floor: `{', '.join(summary.get('cards_below_floor') or [])}`",
            f"- v4 private pass rate: `{summary.get('v4_private_pass_rate')}`",
            f"- v4 learned-only pass rate: `{summary.get('v4_learned_only_pass_rate')}`",
            f"- Prototype pass count: `{summary.get('prototype_pass_count')}`",
            f"- Learned-token pass count: `{summary.get('learned_token_pass_count')}`",
            f"- Pre-public learned-token pass count total: `{summary.get('pre_public_learned_token_pass_count_total')}`",
            f"- Frozen integrity SHA-256: `{summary.get('frozen_integrity_sha256')}`",
            f"- Frozen harness hashes current: `{summary.get('frozen_harness_hashes_current')}`",
            f"- Frozen evidence artifacts hashable: `{summary.get('frozen_evidence_artifacts_hashable')}`",
            f"- Frontier expander decision: `{summary.get('frontier_expander_decision')}`",
            f"- Frontier expander next safe private action: `{summary.get('frontier_expander_next_safe_private_action')}`",
            f"- Current private selected/pass-if-any: `{summary.get('current_private_selected_pass_rate')}` / `{summary.get('current_private_pass_if_any_rate')}`",
            f"- Public-transfer readiness: `{summary.get('public_transfer_readiness_state')}`; private evidence current: `{summary.get('public_transfer_readiness_private_evidence_current')}`",
            f"- Public-transfer full-body selected/pass-if-any: `{summary.get('public_transfer_full_body_selected_pass_rate')}` / `{summary.get('public_transfer_full_body_pass_if_any_rate')}`",
            f"- Current public-shape structure/return: `{summary.get('current_public_shape_selected_obligation_satisfaction_rate')}` / `{summary.get('current_public_shape_selected_return_compatible_rate')}`",
            f"- VCM runtime/native claimable: `{summary.get('vcm_runtime_state')}` / `{summary.get('vcm_native_runtime_claimable')}`",
            f"- macOS MLX state/used: `{summary.get('macos_mlx_state')}` / `{summary.get('macos_mlx_used')}`",
            f"- Teacher fail-closed/admitted rows: `{summary.get('teacher_distillation_fail_closed')}` / `{summary.get('teacher_rows_admitted')}`",
            "",
            "## Gates",
        ]
        for item in report.get("gates", []):
            lines.append(f"- `{item.get('name')}`: `{item.get('passed')}`")
        lines.extend(["", "## Next Actions"])
        for action in report.get("next_actions", []):
            lines.append(f"- {action}")
        lines.append("")
        return "\n".join(lines)

    lines = [
        "# Public Calibration Readiness Packet",
        "",
        f"- Trigger state: `{report.get('trigger_state')}`",
        f"- Technical ready: `{report.get('technical_ready_for_one_bounded_public_calibration')}`",
        f"- Public calibration allowed: `{report.get('public_calibration_allowed')}`",
        f"- Operator lock active: `{report.get('operator_lock_active')}`",
        f"- Public surface: `{summary.get('public_surface_card_count')}` cards / `{summary.get('public_surface_task_count')}` tasks",
        f"- Checkpoint backend: `{summary.get('checkpoint_backend')}`",
        f"- Decoder source/release fresh: `{summary.get('decoder_source_release_fresh')}`",
        f"- Broad public pass rate: `{summary.get('broad_public_pass_rate')}`",
        f"- Cards below floor: `{', '.join(summary.get('cards_below_floor') or [])}`",
        f"- Recovery private rows: `{summary.get('recovery_private_rows')}`",
        "",
        "## Gates",
    ]
    for item in report.get("gates", []):
        lines.append(f"- `{item.get('name')}`: `{item.get('passed')}`")
    lines.extend(["", "## Next Actions"])
    for action in report.get("next_actions", []):
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def gate(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return ROOT / value


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path)


def path_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def decoder_source_release_freshness(closure_mtime: float) -> dict[str, Any]:
    source_paths = decoder_source_paths()
    newest_source_mtime = max((path_mtime(path) for path in source_paths), default=0.0)
    release_binary_mtime = path_mtime(RELEASE_BINARY)
    required_mtime = max(newest_source_mtime, release_binary_mtime)
    stale_reasons: list[str] = []
    if not closure_mtime:
        stale_reasons.append("closure_missing_or_unreadable")
    if not RELEASE_BINARY.exists():
        stale_reasons.append("release_binary_missing")
    if newest_source_mtime and closure_mtime < newest_source_mtime:
        stale_reasons.append("decoder_source_newer_than_closure")
    if release_binary_mtime and closure_mtime < release_binary_mtime:
        stale_reasons.append("release_binary_newer_than_closure")
    return {
        "fresh": bool(closure_mtime and RELEASE_BINARY.exists() and closure_mtime >= required_mtime),
        "closure_mtime": closure_mtime,
        "newest_source_mtime": newest_source_mtime or None,
        "release_binary": rel_or_abs(RELEASE_BINARY),
        "release_binary_exists": RELEASE_BINARY.exists(),
        "release_binary_mtime": release_binary_mtime or None,
        "required_mtime": required_mtime or None,
        "source_count": len(source_paths),
        "newest_sources": [
            {"path": rel_or_abs(path), "mtime": path_mtime(path)}
            for path in sorted(source_paths, key=path_mtime, reverse=True)[:8]
        ],
        "stale_reasons": stale_reasons,
        "rule": "calibration-readiness artifacts must be regenerated after decoder source changes or after rebuilding the release binary.",
    }


def decoder_source_paths() -> list[Path]:
    paths: list[Path] = []
    for root in DECODER_SOURCE_ROOTS:
        if root.is_file():
            paths.append(root)
        elif root.is_dir():
            paths.extend(path for path in root.rglob("*.rs") if path.is_file())
    return paths


def same_path(left: str, right: Path) -> bool:
    if not left:
        return False
    try:
        return resolve(left).resolve() == right.resolve()
    except Exception:
        return str(left).replace("\\", "/").endswith(str(right).replace("\\", "/"))


def get_path(obj: Any, path: list[str], default: Any = None) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def object_field(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key)
    return value if isinstance(value, dict) else {}


def list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def command_arg(command: list[Any], flag: str) -> str:
    parts = [str(part) for part in command]
    try:
        index = parts.index(flag)
        return parts[index + 1]
    except (ValueError, IndexError):
        return ""


def any_true(key: str, value: Any) -> bool:
    if isinstance(value, dict):
        return any((item is True) if name == key else any_true(key, item) for name, item in value.items())
    if isinstance(value, list):
        return any(any_true(key, item) for item in value)
    return False


def first_number(*values: Any) -> float:
    for value in values:
        try:
            if value is not None and value != "":
                return float(value)
        except Exception:
            continue
    return 0.0


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
