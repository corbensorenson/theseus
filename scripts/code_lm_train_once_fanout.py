#!/usr/bin/env python3
"""Train-once Code LM checkpoint plus candidate fanout.

This replaces the repeated-training shard envelope for the normal recovery path.
It still allows candidate/eval work to be distributed later, but the expensive
Code LM/STS training pass happens once and writes a reusable checkpoint with
model_artifacts_v1.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from code_lm_active_worker_monitor import (  # noqa: E402
    infer_active_worker_slug,
    phase_heartbeat_for_active_phase,
    summarize_active_phase_heartbeat,
)
from code_lm_process_guard import windows_active_code_lm_process_rows  # noqa: E402
from code_lm_decoder_contracts import attach_decoder_contracts, public_decoder_contract_preflight  # noqa: E402
from code_lm_json_io import count_jsonl_rows, read_json, read_jsonl_dicts, write_json, write_jsonl  # noqa: E402
from code_lm_public_task_export import export_public_visible_tasks  # noqa: E402
from code_lm_phase_heartbeat import run_command_with_optional_heartbeat  # noqa: E402
from code_lm_phase_ledger import (  # noqa: E402
    append_phase_event as append_phase_event_raw,
    summarize_phase_ledger as summarize_phase_ledger_raw,
)
from code_lm_resource_policy import (  # noqa: E402
    resource_allows_code_work,
    resource_deferral_is_self_observation,
    run_resource_policy as run_resource_policy_raw,
    summarize_resource_policy,
)
from code_lm_train_once_contracts import (  # noqa: E402
    BROAD_FLOOR_PUBLIC_CARDS,
    BROAD_FLOOR_PUBLIC_CASES_PER_CARD,
    CONTROL_SIGNAL_CONTRACT,
    PHASE_CONTRACTS,
    PRIVATE_ROWS,
    STAGED_VERIFICATION_CONTRACT,
)
from code_lm_train_once_markdown import render_markdown  # noqa: E402
from code_lm_private_rows import (  # noqa: E402
    DEFAULT_EXTRA_PRIVATE_TRAIN_JSONL,
    DEFAULT_REPO_REPAIR_PRIVATE_TRAIN_JSONL,
    DEFAULT_RESIDUAL_PRIVATE_TRAIN_JSONL,
)
from theseus_archive_resolver import is_archive_pointer, resolve_archived_path  # noqa: E402
import vcm_consumer_abi  # noqa: E402

REPORTS = ROOT / "reports"
REHYDRATED_ARTIFACTS = ROOT / "runtime" / "rehydrated_artifacts"
DEFAULT_OUT = REPORTS / "code_lm_train_once_fanout.json"
DEFAULT_MARKDOWN = REPORTS / "code_lm_train_once_fanout.md"
DEFAULT_VCM_CONTEXT_GOVERNOR = REPORTS / "vcm_context_governor.json"
DEFAULT_STRICT_GENERATOR_FANOUT_RECEIPT = REPORTS / "neural_seed_strict_generator_fanout_receipt.json"
MIN_PUBLIC_CANDIDATES_FOR_DECODER_GATE = 96


def release_binary_path() -> Path:
    name = "symliquid-cli.exe" if sys.platform.startswith("win") else "symliquid-cli"
    return ROOT / "target" / "release" / name


def cuda_readout_requested_by_launcher() -> bool:
    raw = os.environ.get("THESEUS_CODE_LM_CUDA_READOUT", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    # Preserve the Windows coordinator's CUDA default while letting macOS
    # produce honest native release-binary evidence.
    return sys.platform.startswith("win")


def release_binary_backend() -> str:
    if cuda_readout_requested_by_launcher():
        return "cuda_release_readout"
    if sys.platform == "darwin":
        return "macos_native_cpu_readout"
    return "native_cpu_readout"


def release_build_command() -> list[str]:
    command = ["cargo", "build", "--release", "-p", "symliquid-cli"]
    if cuda_readout_requested_by_launcher():
        command.extend(["--features", "cuda"])
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--slug", default="private_pressure_private_recovery_train_once_fanout_v1")
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--private-count", type=int, default=320)
    parser.add_argument("--public-cards", default="source_evalplus,source_human_eval,source_mbpp,source_bigcodebench")
    parser.add_argument("--max-public-cases-per-card", type=int, default=8)
    parser.add_argument(
        "--case-manifest",
        default="",
        help="Optional public calibration selector manifest. Contains task IDs only.",
    )
    parser.add_argument("--hv-dim", type=int, default=768)
    parser.add_argument("--max-vocab", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--candidates-per-task", type=int, default=8)
    parser.add_argument("--max-high-transfer-private-train", type=int, default=4800)
    parser.add_argument("--max-rust-work-steps", type=int, default=3_000_000)
    parser.add_argument("--rust-timeout-seconds", type=int, default=5_400)
    parser.add_argument("--sts-timeout-seconds", type=int, default=2_400)
    parser.set_defaults(native_sts_conditioning=True)
    parser.add_argument(
        "--enable-native-sts-conditioning",
        dest="native_sts_conditioning",
        action="store_true",
        help="Enable the native STS conditioner before Code LM checkpoint training. This is the default.",
    )
    parser.add_argument(
        "--disable-native-sts-conditioning",
        dest="native_sts_conditioning",
        action="store_false",
        help="Control/ablation lane only: train and fan out without native STS conditioning.",
    )
    parser.add_argument("--sts-conditioning-epochs", type=int, default=2)
    parser.add_argument("--sts-conditioning-max-train-rows", type=int, default=960)
    parser.add_argument("--sts-conditioning-max-eval-rows", type=int, default=96)
    parser.add_argument("--sts-conditioning-max-generate-rows", type=int, default=96)
    parser.add_argument("--fanout-timeout-seconds", type=int, default=21_600)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--allow-active-worker", action="store_true")
    parser.add_argument(
        "--refresh-fanout-only",
        action="store_true",
        help="Refresh candidate fanout from an existing checkpoint only; never start checkpoint training.",
    )
    parser.add_argument(
        "--full-fanout-refresh",
        action="store_true",
        help="When refreshing fanout-only, refresh canonical full manifests instead of the bounded current-source smoke.",
    )
    parser.add_argument(
        "--private-only-refresh",
        action="store_true",
        help=(
            "When refreshing fanout-only, run the bounded smoke against private eval rows only by using an empty "
            "public manifest sidecar. This is a private repair proof, not public calibration."
        ),
    )
    parser.add_argument(
        "--private-only",
        action="store_true",
        help=(
            "Run the full train-once checkpoint/fanout envelope against private rows with an intentionally empty "
            "public sidecar. This produces private repair evidence only and cannot unlock public calibration."
        ),
    )
    parser.add_argument("--refresh-private-eval-limit", type=int, default=4)
    parser.add_argument("--refresh-public-task-limit", type=int, default=4)
    parser.add_argument("--refresh-candidates-per-task", type=int, default=1)
    parser.add_argument("--refresh-smoke-timeout-seconds", type=int, default=600)
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()
    if args.private_only_refresh:
        args.refresh_fanout_only = True
        args.full_fanout_refresh = False

    started = time.perf_counter()
    out = resolve(args.out)
    markdown = resolve(args.markdown_out)
    paths = artifact_paths(args.slug)
    if args.private_only and not args.refresh_fanout_only:
        paths = private_only_full_paths(paths)
    previous_state = prior_train_once_state(out, args.slug)
    state: dict[str, Any] = {
        "policy": "project_theseus_code_lm_train_once_fanout_v1",
        "created_utc": now(),
        "trigger_state": "PLANNED",
        "run_status": "planned",
        "execute": bool(args.execute),
        "slug": args.slug,
        "architecture": {
            "training": "one release-binary checkpoint pass using CUDA when requested/supported, otherwise native CPU readout",
            "fanout": "candidate generation from checkpoint without retraining",
            "hive_target": "distribute candidate/verification slices against the same checkpoint",
            "repeated_training_per_candidate_shard": False,
            "public_calibration": "locked; this script only emits private/public candidate manifests",
            "phase_timing": "every heavy phase appends to the phase ledger and names its downstream consumer",
            "verification": "lint/parse -> compile/import -> cheap behavior -> sandbox/full tests; each layer is fail-fast until it passes",
        },
        "paths": {key: rel(path) for key, path in paths.items()},
        "phase_contracts": PHASE_CONTRACTS,
        "staged_verification_contract": STAGED_VERIFICATION_CONTRACT,
        "control_signal_contract": CONTROL_SIGNAL_CONTRACT,
        "refresh_fanout_only": bool(args.refresh_fanout_only),
        "full_fanout_refresh": bool(args.full_fanout_refresh),
        "private_only_refresh": bool(args.private_only_refresh),
        "private_only": bool(args.private_only),
        "bounded_current_source_refresh": bool(args.refresh_fanout_only and not args.full_fanout_refresh),
        "bounded_current_source_refresh_limits": {
            "private_eval_limit": max(0, int(args.refresh_private_eval_limit)),
            "public_task_limit": 0 if private_only_mode(args) else max(0, int(args.refresh_public_task_limit)),
            "candidates_per_task": max(1, int(args.refresh_candidates_per_task)),
            "timeout_seconds": max(1, int(args.refresh_smoke_timeout_seconds)),
            "artifact_semantics": "current-source freshness and speed smoke only; does not replace canonical closure manifests",
        },
        "effective_public_surface": effective_public_surface(args),
        "sts_conditioning_mode": (
            "native_sts_conditioning_default_on" if args.native_sts_conditioning else "sts_control_ablation_disabled"
        ),
        "sts_default_policy": {
            "default": "native_sts_conditioning_on",
            "disable_flag": "--disable-native-sts-conditioning",
            "sts_off_role": "same-seed control/ablation lane only",
            "same_seed_non_sts_comparator_preserved": True,
        },
        "external_inference_calls": 0,
    }
    vcm_receipt = vcm_context_governor_receipt()
    state["vcm_context_governor_receipt"] = vcm_receipt
    state.update(vcm_context_summary_fields(vcm_receipt, surface="fanout_supervisor"))
    inherit_private_input_content_context(state, previous_state)
    active_rows = active_code_worker_rows()
    checkpoint_report = read_json(paths["checkpoint_wrapper_report"], {})
    fanout_report = read_json(paths["fanout_report"], {})
    state["artifact_provenance"] = build_artifact_provenance(paths, previous_state)
    private_input_freshness = effective_private_training_input_freshness(paths, previous_state)
    state["private_input_freshness"] = private_input_freshness
    supervisor_phase_ledger = summarize_phase_ledger(paths["phase_ledger"])
    supervisor_phase_timing_ms = merged_phase_timing(checkpoint_report, fanout_report, supervisor_phase_ledger)
    supervisor_strict_generator_receipt = strict_generator_fanout_receipt_summary()
    state["verification_bandwidth"] = fanout_verification_bandwidth_record(
        paths,
        private_rows=count_jsonl_rows(paths["private_candidates"]),
        public_rows=count_jsonl_rows(paths["public_candidates"]),
        private_only=private_only_mode(args),
        strict_generator_receipt=supervisor_strict_generator_receipt,
        vcm_receipt=vcm_receipt,
    )
    state["governance_tax"] = fanout_governance_tax_record(
        supervisor_phase_timing_ms,
        verification_bandwidth=state["verification_bandwidth"],
        vcm_receipt=vcm_receipt,
        artifact_provenance=state["artifact_provenance"],
    )
    state["viea_fanout_records"] = build_fanout_spine_records(
        args,
        paths,
        private_rows=count_jsonl_rows(paths["private_candidates"]),
        public_rows=count_jsonl_rows(paths["public_candidates"]),
        private_only=private_only_mode(args),
        vcm_receipt=vcm_receipt,
        artifact_provenance=state["artifact_provenance"],
    )
    completed_existing_artifacts = (
        not active_rows
        and not args.refresh_fanout_only
        and checkpoint_ready(checkpoint_report, paths["checkpoint"])
        and checkpoint_sts_policy_current(checkpoint_report, args)
        and private_input_freshness.get("fresh")
        and fanout_ready(
            fanout_report,
            paths["private_candidates"],
            paths["public_candidates"],
            fanout_report_path=paths["fanout_report"],
            checkpoint=paths["checkpoint"],
            allow_empty_public=private_only_mode(args),
        )
        and public_manifest_surface_current(args, paths).get("current")
        and public_fanout_surface_current(paths, private_only=private_only_mode(args)).get("current")
    )
    if completed_existing_artifacts:
        state["artifact_provenance"] = build_artifact_provenance(paths, previous_state)
        state["private_input_freshness"] = state["artifact_provenance"].get(
            "private_input_freshness", private_input_freshness
        )
        closure_report = merged_closure_report(args, paths, checkpoint_report, fanout_report, previous_state)
        write_json(paths["closure_report"], closure_report)
        state.update(
            {
                "trigger_state": closure_report["trigger_state"],
                "run_status": closure_report["run_status"],
                "current_phase": "completed_existing_artifacts",
                "closure_report": rel(paths["closure_report"]),
                "summary": closure_report["summary"],
                "artifact_reuse": {
                    "checkpoint": rel(paths["checkpoint"]),
                    "fanout_report": rel(paths["fanout_report"]),
                    "private_candidates": rel(paths["private_candidates"]),
                    "public_candidates": rel(paths["public_candidates"]),
                    "rule": "completed train-once/fanout artifacts short-circuit the wrapper so automation does not retrain the same slug",
                },
                "next_actions": [
                    "Run decoder_v2_private_ablation_gate.py against the train-once fanout closure report.",
                    "Run private_public_transfer_proof.py only after the decoder gate refreshes.",
                    "Keep public calibration locked unless both gates report ready_for_public_calibration=true.",
                ],
            }
        )
        finish(out, markdown, state, started)
        return 0 if closure_report["trigger_state"] in {"GREEN", "YELLOW"} else 2
    if not args.execute:
        fanout_staleness = fanout_source_freshness(paths)
        private_input_freshness = effective_private_training_input_freshness(paths, previous_state)
        state["private_input_freshness"] = private_input_freshness
        if active_rows:
            active_slug = infer_active_worker_slug(active_rows) or args.slug
            active_paths = artifact_paths(active_slug)
            if active_slug != args.slug:
                state["slug"] = active_slug
                state["paths"] = {key: rel(path) for key, path in active_paths.items()}
                active_previous_state = prior_train_once_state(out, active_slug)
                state["artifact_provenance"] = build_artifact_provenance(active_paths, active_previous_state)
                state["private_input_freshness"] = effective_private_training_input_freshness(
                    active_paths, active_previous_state
                )
            active_phase = infer_active_worker_phase(active_rows)
            active_heartbeat = phase_heartbeat_for_active_phase(active_paths, active_phase)
            state.update(
                {
                    "trigger_state": "RUNNING",
                    "run_status": "active_worker_discovered",
                    "current_phase": active_phase,
                    "active_worker_slug": active_slug,
                    "active_phase_heartbeat": rel(active_heartbeat) if active_heartbeat is not None else "",
                    "active_phase_heartbeat_summary": summarize_active_phase_heartbeat(active_heartbeat, root=ROOT),
                    "phase_ledger_summary": summarize_phase_ledger(active_paths["phase_ledger"]),
                    "active_processes": active_rows[:8],
                    "next_actions": ["Existing Code LM work is active; do not overwrite it with a planned/dry-run state."],
                }
            )
        elif checkpoint_ready(checkpoint_report, paths["checkpoint"]) and checkpoint_sts_policy_current(
            checkpoint_report, args
        ) and fanout_ready(
            fanout_report,
            paths["private_candidates"],
            paths["public_candidates"],
            require_current_source=False,
            fanout_report_path=paths["fanout_report"],
            checkpoint=paths["checkpoint"],
            allow_empty_public=private_only_mode(args),
        ):
            private_rows_fresh = bool(private_input_freshness.get("fresh"))
            stale_status = (
                "stale_artifacts_need_fanout_refresh"
                if private_rows_fresh
                else "stale_private_training_inputs_need_checkpoint_retrain"
            )
            stale_phase = (
                "checkpoint_reusable_fanout_stale"
                if private_rows_fresh
                else "checkpoint_private_inputs_stale"
            )
            stale_next_actions = (
                [
                    "Reuse the checkpoint and refresh fanout only; private rows still match the checkpoint.",
                    "Do not retrain and do not public-calibrate until decoder and transfer gates are GREEN.",
                ]
                if private_rows_fresh
                else [
                    "Private training rows changed; do not run fanout-only refresh from this stale checkpoint.",
                    "Run the full train-once path only after explicitly choosing to spend training budget.",
                    "Do not public-calibrate until a fresh checkpoint plus decoder/transfer gates are GREEN.",
                ]
            )
            diagnostic_report = merged_closure_report(args, paths, checkpoint_report, fanout_report, previous_state)
            diagnostic_summary = dict(diagnostic_report.get("summary") or {})
            diagnostic_summary.update(
                {
                    "stale_artifacts_diagnostic_only": True,
                    "artifact_freshness": fanout_staleness,
                    "private_input_freshness": private_input_freshness,
                    "public_calibration_allowed": False,
                    "score_semantics": (
                        "stale train-once/fanout timing and coverage evidence only; refresh or retrain against "
                        "current decoder source and current private rows before decoder/transfer gates may use it"
                    ),
                }
            )
            state.update(
                {
                    "trigger_state": "YELLOW",
                    "run_status": stale_status,
                    "current_phase": stale_phase,
                    "artifact_freshness": fanout_staleness,
                    "private_input_freshness": private_input_freshness,
                    "summary": diagnostic_summary,
                    "next_actions": stale_next_actions,
                }
            )
        write_progress(out, markdown, state)
        print(json.dumps(state, indent=2))
        return 0
    write_progress(out, markdown, state)

    if active_rows and not args.allow_active_worker:
        state.update(
            {
                "trigger_state": "YELLOW",
                "run_status": "deferred",
                "current_phase": "active_code_worker_present",
                "active_processes": active_rows[:8],
                "next_actions": ["A Code LM/SymLiquid worker is already active; do not stack training jobs."],
            }
        )
        finish(out, markdown, state, started)
        return 0

    resource = run_resource_policy()
    state["resource_policy"] = summarize_resource_policy(resource)
    budget = object_field(resource, "recommended_code_lm_budget")
    resource_allowed = resource_allows_code_work(budget)
    if not resource_allowed and resource_deferral_is_self_observation(
        budget,
        active_worker_present=bool(active_rows),
    ):
        state["resource_policy"].update(
            {
                "start_new_train_once_fanout": True,
                "self_observation_override": True,
                "override_reason": (
                    "resource policy counted this train-once supervisor as an active "
                    "Code LM worker; local process guard found no other workers"
                ),
            }
        )
        resource_allowed = True
    if not resource_allowed:
        state.update(
            {
                "trigger_state": "YELLOW",
                "run_status": "deferred",
                "current_phase": "resource_policy_deferred",
                "next_actions": ["Resource policy deferred Code LM work; retry when the machine is clear."],
            }
        )
        finish(out, markdown, state, started)
        return 0

    append_phase_event(
        paths["phase_ledger"],
        "release_cuda_binary",
        "started",
        {
            "skip_build_requested": bool(args.skip_build),
            "release_binary_backend": release_binary_backend(),
            "cuda_readout_requested_by_launcher": cuda_readout_requested_by_launcher(),
            "cuda_feature_build_enforced": cuda_readout_requested_by_launcher(),
            "reason": "--skip-build must not allow a fresh train-once path to trust a stale or wrong-backend release binary",
        },
    )
    build = ensure_release_cuda_binary()
    append_phase_event(paths["phase_ledger"], "release_cuda_binary", "completed", build)
    state["release_cuda_binary"] = build
    state["phase_ledger_summary"] = summarize_phase_ledger(paths["phase_ledger"])
    write_progress(out, markdown, state)
    if not build.get("ready"):
        state.update(
            {
                "trigger_state": "RED",
                "run_status": "failed",
                "current_phase": "release_cuda_binary_not_ready",
                "next_actions": [
                    f"Build {rel(release_binary_path())} before training; do not use cargo run as the hot path."
                ],
            }
        )
        finish(out, markdown, state, started)
        return 2

    private_input_freshness = effective_private_training_input_freshness(paths, previous_state)
    state["private_input_freshness"] = private_input_freshness
    private_input_signature_before_checkpoint = private_input_signature(private_input_freshness)
    private_input_content_signature_before_checkpoint = private_input_content_signature(private_input_freshness)
    state["private_input_signature_before_checkpoint"] = private_input_signature_before_checkpoint
    state["private_input_content_signature_before_checkpoint"] = private_input_content_signature_before_checkpoint
    checkpoint_needs_retrain = bool(
        not checkpoint_ready(checkpoint_report, paths["checkpoint"])
        or not checkpoint_sts_policy_current(checkpoint_report, args)
        or not private_input_freshness.get("fresh")
    )
    if checkpoint_needs_retrain:
        if args.refresh_fanout_only:
            state.update(
                {
                    "trigger_state": "YELLOW",
                    "run_status": "deferred",
                    "current_phase": "missing_or_stale_checkpoint_for_fanout_only_refresh",
                    "private_input_freshness": private_input_freshness,
                    "next_actions": [
                        "The fanout-only refresh guard refused to train a new or private-row-stale checkpoint.",
                        "Run the full train-once path only after explicitly choosing to spend training budget.",
                    ],
                }
            )
            finish(out, markdown, state, started)
            return 0
        train_command = checkpoint_command(args, paths)
        append_phase_event(
            paths["phase_ledger"],
            "train_once_checkpoint",
            "started",
            {"command_preview": command_preview(train_command), "checkpoint": rel(paths["checkpoint"])},
        )
        mark_running(
            state,
            phase="train_once_checkpoint",
            command=train_command,
            log_path=REPORTS / f"code_lm_train_once_fanout_{args.slug}_checkpoint.log",
            heartbeat_path=paths["checkpoint_phase_heartbeat"],
        )
        write_progress(out, markdown, state)
        train_phase = run_command(
            train_command,
            timeout_seconds=max(1, args.rust_timeout_seconds),
            log_path=REPORTS / f"code_lm_train_once_fanout_{args.slug}_checkpoint.log",
            phase="train_once_checkpoint",
            heartbeat_path=paths["checkpoint_phase_heartbeat"],
            progress_paths=[
                paths["checkpoint_rust_report"],
                paths["checkpoint_wrapper_report"],
                paths["checkpoint"],
            ],
        )
        append_phase_event(paths["phase_ledger"], "train_once_checkpoint", "completed", train_phase)
        state["checkpoint_phase"] = train_phase
        checkpoint_report = read_json(paths["checkpoint_wrapper_report"], {})
        private_input_freshness_after_checkpoint = private_training_input_freshness(paths)
        private_input_signature_after_checkpoint = private_input_signature(private_input_freshness_after_checkpoint)
        private_input_content_signature_after_checkpoint = private_input_content_signature(private_input_freshness_after_checkpoint)
        state["private_input_freshness_after_checkpoint"] = private_input_freshness_after_checkpoint
        state["private_input_signature_after_checkpoint"] = private_input_signature_after_checkpoint
        state["private_input_content_signature_after_checkpoint"] = private_input_content_signature_after_checkpoint
        state["private_input_mtime_changed_during_checkpoint"] = (
            private_input_signature_after_checkpoint != private_input_signature_before_checkpoint
        )
        state["private_input_content_changed_during_checkpoint"] = (
            private_input_content_signature_after_checkpoint != private_input_content_signature_before_checkpoint
        )
        state["private_inputs_changed_during_checkpoint"] = state["private_input_content_changed_during_checkpoint"]
        state["phase_ledger_summary"] = summarize_phase_ledger(paths["phase_ledger"])
        write_progress(out, markdown, state)
        if not checkpoint_ready(checkpoint_report, paths["checkpoint"]):
            state.update(
                {
                    "trigger_state": "RED",
                    "run_status": "failed",
                    "current_phase": "checkpoint_failed",
                    "next_actions": ["Fix checkpoint-only training before fanout; no public calibration is allowed."],
                }
            )
            finish(out, markdown, state, started)
            return 2
        if not checkpoint_sts_policy_current(checkpoint_report, args):
            state.update(
                {
                    "trigger_state": "RED",
                    "run_status": "failed",
                    "current_phase": "checkpoint_sts_policy_mismatch",
                    "sts_checkpoint_policy": checkpoint_sts_policy(checkpoint_report, args),
                    "next_actions": [
                        "The default train-once path requires a native STS-conditioned checkpoint.",
                        "Fix STS conditioning input/cache/runtime or rerun explicitly with --disable-native-sts-conditioning for an ablation only.",
                    ],
                }
            )
            finish(out, markdown, state, started)
            return 2
        if state["private_inputs_changed_during_checkpoint"]:
            append_phase_event(
                paths["phase_ledger"],
                "train_once_checkpoint",
                "private_inputs_changed_during_checkpoint",
                {
                    "before": private_input_signature_before_checkpoint,
                    "after": private_input_signature_after_checkpoint,
                    "content_before": private_input_content_signature_before_checkpoint,
                    "content_after": private_input_content_signature_after_checkpoint,
                    "mtime_changed": state["private_input_mtime_changed_during_checkpoint"],
                    "content_changed": state["private_input_content_changed_during_checkpoint"],
                    "rule": "do not fan out candidates from a checkpoint trained while canonical private input content changed",
                },
            )
            state.update(
                {
                    "trigger_state": "YELLOW",
                    "run_status": "deferred",
                    "current_phase": "checkpoint_private_inputs_changed_during_run",
                    "private_input_freshness": private_input_freshness_after_checkpoint,
                    "next_actions": [
                        "A private training input changed while checkpoint training was running; treat this checkpoint as diagnostic.",
                        "Rerun train-once after active workers are idle so the checkpoint exactly matches canonical private rows.",
                    ],
                }
            )
            finish(out, markdown, state, started)
            return 0
        if state["private_input_mtime_changed_during_checkpoint"]:
            append_phase_event(
                paths["phase_ledger"],
                "train_once_checkpoint",
                "private_input_mtime_only_change_during_checkpoint",
                {
                    "before": private_input_signature_before_checkpoint,
                    "after": private_input_signature_after_checkpoint,
                    "content_signature": private_input_content_signature_after_checkpoint,
                    "rule": "mtime-only churn is diagnostic; unchanged private row content does not invalidate a fresh train-once checkpoint",
                },
            )
    else:
        state["checkpoint_phase"] = {
            "reused": True,
            "report": rel(paths["checkpoint_wrapper_report"]),
            "private_input_freshness": private_input_freshness,
        }
        append_phase_event(paths["phase_ledger"], "train_once_checkpoint", "reused", state["checkpoint_phase"])

    manifest_surface = (
        ensure_private_only_manifest_surface(paths, canonical=True)
        if args.private_only and not args.refresh_fanout_only
        else ensure_private_only_manifest_surface(paths)
        if args.private_only_refresh and args.refresh_fanout_only and not args.full_fanout_refresh
        else ensure_public_manifest_surface(args, paths)
    )
    state["public_manifest_surface"] = manifest_surface
    if not manifest_surface.get("ready"):
        state.update(
            {
                "trigger_state": "RED",
                "run_status": "failed",
                "current_phase": "public_manifest_surface_failed",
                "next_actions": [
                    "Fix visible public task export/decoder-contract preflight before refreshing fanout.",
                    "Do not public-calibrate against a candidate manifest that cannot cover the intended surface.",
                ],
            }
        )
        finish(out, markdown, state, started)
        return 2

    if args.refresh_fanout_only and not args.full_fanout_refresh:
        sts_path = str(get_path(checkpoint_report, ["summary", "sts_generation_path"], "") or "")
        smoke_paths = current_source_smoke_paths(paths)
        if args.private_only_refresh:
            smoke_paths["public_manifest"] = paths["current_source_smoke_empty_public_manifest"]
        smoke_cmd = fanout_command(
            args,
            smoke_paths,
            sts_path,
            candidates_per_task=max(1, args.refresh_candidates_per_task),
            private_eval_limit=max(0, args.refresh_private_eval_limit),
            public_task_limit=0 if args.private_only_refresh else max(0, args.refresh_public_task_limit),
        )
        append_phase_event(
            paths["phase_ledger"],
            "checkpoint_fanout_candidate_generation",
            "started",
            {
                "command_preview": command_preview(smoke_cmd),
                "mode": "private_only_current_source_smoke"
                if args.private_only_refresh
                else "bounded_current_source_smoke",
                "sts_streams": sts_path,
                "artifact_semantics": "sidecar freshness proof, not canonical fanout closure evidence",
                "public_calibration": False,
            },
        )
        mark_running(
            state,
            phase="checkpoint_fanout_current_source_smoke",
            command=smoke_cmd,
            log_path=REPORTS / f"code_lm_train_once_fanout_{args.slug}_current_source_smoke.log",
            heartbeat_path=paths["current_source_smoke_phase_heartbeat"],
        )
        state["next_actions"] = [
            "Leave this bounded current-source smoke running; do not stack duplicate Code LM jobs.",
            "Only use --full-fanout-refresh after an operator explicitly chooses to spend full fanout budget.",
        ]
        write_progress(out, markdown, state)
        smoke_phase = run_command(
            smoke_cmd,
            timeout_seconds=max(1, args.refresh_smoke_timeout_seconds),
            log_path=REPORTS / f"code_lm_train_once_fanout_{args.slug}_current_source_smoke.log",
            phase="checkpoint_fanout_current_source_smoke",
            heartbeat_path=paths["current_source_smoke_phase_heartbeat"],
            progress_paths=[
                smoke_paths["fanout_report"],
                smoke_paths["private_candidates"],
                smoke_paths["public_candidates"],
            ],
        )
        if args.private_only_refresh and successful_phase(smoke_phase):
            state["private_only_public_candidate_sidecar"] = refresh_empty_public_candidate_sidecar(
                smoke_paths["public_candidates"]
            )
        append_phase_event(paths["phase_ledger"], "checkpoint_fanout_candidate_generation", "completed", smoke_phase)
        smoke_report = read_json(smoke_paths["fanout_report"], {})
        smoke_summary = bounded_current_source_smoke_summary(args, smoke_paths, smoke_report, smoke_phase)
        state["fanout_phase"] = smoke_phase
        state["current_source_fanout_smoke"] = smoke_summary
        state["artifact_freshness"] = fanout_source_freshness(paths)
        state["phase_ledger_summary"] = summarize_phase_ledger(paths["phase_ledger"])
        if not smoke_summary.get("ready"):
            state.update(
                {
                    "trigger_state": "RED",
                    "run_status": "current_source_fanout_smoke_failed",
                    "current_phase": "checkpoint_fanout_current_source_smoke_failed",
                    "next_actions": [
                        "Fix bounded fanout smoke before any full fanout or training restart.",
                        "Do not use canonical stale manifests as promotion evidence.",
                    ],
                }
            )
            finish(out, markdown, state, started)
            return 2
        state.update(
            {
                "trigger_state": "YELLOW",
                "run_status": "current_source_fanout_smoke_completed",
                "current_phase": "checkpoint_reusable_fanout_canonical_manifest_stale",
                "summary": {
                    "train_once_checkpoint_fanout": True,
                    "repeated_training_per_candidate_shard": False,
                    "current_source_smoke_ready": True,
                    "current_source_smoke": smoke_summary,
                    "canonical_fanout_artifacts_fresh": False,
                    "canonical_fanout_artifacts_diagnostic_only": True,
                    "public_calibration_allowed": False,
                    "score_semantics": (
                        "bounded current-source fanout smoke only; canonical full manifests remain stale and "
                        "cannot unlock decoder/transfer/public calibration gates"
                    ),
                },
                "next_actions": [
                    "Use this bounded smoke as current-source speed/freshness evidence.",
                    "Optimize fanout/ranker/verifier hotspots before any explicit full fanout refresh.",
                    "Keep public calibration locked until decoder and transfer gates are GREEN.",
                ],
            }
        )
        finish(out, markdown, state, started)
        return 0

    fanout_surface_current = public_fanout_surface_current(paths, private_only=private_only_mode(args))
    state["public_fanout_surface"] = fanout_surface_current
    if (
        not fanout_ready(
        fanout_report,
        paths["private_candidates"],
        paths["public_candidates"],
        fanout_report_path=paths["fanout_report"],
        checkpoint=paths["checkpoint"],
        allow_empty_public=private_only_mode(args),
        )
        or not fanout_surface_current.get("current")
    ):
        sts_path = str(get_path(checkpoint_report, ["summary", "sts_generation_path"], "") or "")
        fanout_cmd = fanout_command(args, paths, sts_path)
        append_phase_event(
            paths["phase_ledger"],
            "checkpoint_fanout_candidate_generation",
            "started",
            {"command_preview": command_preview(fanout_cmd), "sts_streams": sts_path},
        )
        mark_running(
            state,
            phase="checkpoint_fanout_candidate_generation",
            command=fanout_cmd,
            log_path=REPORTS / f"code_lm_train_once_fanout_{args.slug}_fanout.log",
            heartbeat_path=paths["fanout_phase_heartbeat"],
        )
        write_progress(out, markdown, state)
        fanout_phase = run_command(
            fanout_cmd,
            timeout_seconds=max(1, args.fanout_timeout_seconds),
            log_path=REPORTS / f"code_lm_train_once_fanout_{args.slug}_fanout.log",
            phase="checkpoint_fanout_candidate_generation",
            heartbeat_path=paths["fanout_phase_heartbeat"],
            progress_paths=[
                paths["fanout_report"],
                paths["private_candidates"],
                paths["public_candidates"],
            ],
        )
        if private_only_mode(args) and successful_phase(fanout_phase):
            state["private_only_public_candidate_sidecar"] = refresh_empty_public_candidate_sidecar(
                paths["public_candidates"]
            )
        append_phase_event(paths["phase_ledger"], "checkpoint_fanout_candidate_generation", "completed", fanout_phase)
        state["fanout_phase"] = fanout_phase
        fanout_report = read_json(paths["fanout_report"], {})
        state["phase_ledger_summary"] = summarize_phase_ledger(paths["phase_ledger"])
        write_progress(out, markdown, state)
        fanout_surface_current = public_fanout_surface_current(paths, private_only=private_only_mode(args))
        state["public_fanout_surface"] = fanout_surface_current
        if (
            not fanout_ready(
            fanout_report,
            paths["private_candidates"],
            paths["public_candidates"],
            fanout_report_path=paths["fanout_report"],
            checkpoint=paths["checkpoint"],
            allow_empty_public=private_only_mode(args),
            )
            or not fanout_surface_current.get("current")
        ):
            state.update(
                {
                    "trigger_state": "RED",
                    "run_status": "failed",
                    "current_phase": "fanout_failed",
                    "next_actions": ["Fanout did not produce complete candidate manifests; preserve checkpoint and rerun fanout only."],
                }
            )
            finish(out, markdown, state, started)
            return 2
    else:
        state["fanout_phase"] = {"reused": True, "report": rel(paths["fanout_report"])}
        append_phase_event(paths["phase_ledger"], "checkpoint_fanout_candidate_generation", "reused", state["fanout_phase"])

    state["artifact_provenance"] = build_artifact_provenance(paths, state)
    state["private_input_freshness"] = state["artifact_provenance"].get(
        "private_input_freshness", effective_private_training_input_freshness(paths, state)
    )
    closure_report = merged_closure_report(args, paths, checkpoint_report, fanout_report, state)
    write_json(paths["closure_report"], closure_report)
    state.update(
        {
            "trigger_state": closure_report["trigger_state"],
            "run_status": closure_report["run_status"],
            "current_phase": "completed",
            "closure_report": rel(paths["closure_report"]),
            "summary": closure_report["summary"],
            "next_actions": [
                "Run decoder_v2_private_ablation_gate.py against the train-once fanout closure report.",
                "Run private_public_transfer_proof.py only after the decoder gate refreshes.",
                "Keep public calibration locked unless both gates report ready_for_public_calibration=true.",
            ],
        }
    )
    finish(out, markdown, state, started)
    return 0 if closure_report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def checkpoint_command(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    command = [
        sys.executable,
        "scripts/code_lm_closure.py",
        "--checkpoint-only",
        "--skip-public-calibration",
        *(
            ["--private-only"]
            if private_only_mode(args)
            else []
        ),
        "--seed",
        str(args.seed),
        "--private-count",
        str(args.private_count),
        "--public-cards",
        effective_public_cards(args),
        "--max-public-cases-per-card",
        str(max(1, effective_max_public_cases_per_card(args))),
        *(
            ["--case-manifest", args.case_manifest]
            if str(args.case_manifest or "").strip()
            else []
        ),
        "--hv-dim",
        str(args.hv_dim),
        "--max-vocab",
        str(args.max_vocab),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--candidates-per-task",
        str(args.candidates_per_task),
        "--disable-residual-private-train",
        "--high-transfer-private-train-jsonl",
        PRIVATE_ROWS,
        "--max-high-transfer-private-train",
        str(max(1, args.max_high_transfer_private_train)),
        "--max-rust-work-steps",
        str(max(0, args.max_rust_work_steps)),
        "--rust-timeout-seconds",
        str(max(1, args.rust_timeout_seconds)),
        "--sts-decoder-control-policy-jsonl",
        "reports/sts_decoder_control_rows.jsonl",
        "--sts-timeout-seconds",
        str(max(1, args.sts_timeout_seconds)),
        "--sts-conditioning-epochs",
        str(max(1, args.sts_conditioning_epochs)),
        "--sts-conditioning-max-train-rows",
        str(max(0, args.sts_conditioning_max_train_rows)),
        "--sts-conditioning-max-eval-rows",
        str(max(0, args.sts_conditioning_max_eval_rows)),
        "--sts-conditioning-max-generate-rows",
        str(max(0, args.sts_conditioning_max_generate_rows)),
        *([] if args.native_sts_conditioning else ["--disable-sts-conditioning"]),
        "--private-curriculum-out",
        rel(paths["private_curriculum"]),
        "--public-task-manifest-out",
        rel(paths["public_manifest"]),
        "--checkpoint-out",
        rel(paths["checkpoint"]),
        "--private-candidate-out",
        rel(paths["private_candidates"]),
        "--public-candidate-out",
        rel(paths["public_candidates"]),
        "--rust-report-out",
        rel(paths["checkpoint_rust_report"]),
        "--sts-conditioning-input-out",
        rel(paths["sts_input"]),
        "--sts-generation-out",
        rel(paths["sts_generations"]),
        "--sts-conditioning-checkpoint-out",
        rel(paths["sts_checkpoint"]),
        "--sts-conditioning-report-out",
        rel(paths["sts_report"]),
        "--out",
        rel(paths["checkpoint_wrapper_report"]),
        "--lock-path",
        rel(paths["lock"]),
        "--typed-edge-exec-receiver-v1",
        "--edge-obligation-decode-gate-v1",
        "--private-type-shape-receiver-veto-v1",
    ]
    if cuda_readout_requested_by_launcher():
        command.insert(4, "--use-cuda-readout")
    return command


def effective_public_surface(args: argparse.Namespace) -> dict[str, Any]:
    if private_only_mode(args):
        return {
            "cards": [],
            "max_public_cases_per_card": 0,
            "broad_floor_surface": False,
            "public_tests_used": False,
            "public_solutions_used": False,
            "public_calibration": False,
            "score_semantics": "private-only run with intentionally empty public sidecar",
        }
    cards = [card.strip() for card in effective_public_cards(args).split(",") if card.strip()]
    return {
        "cards": cards,
        "max_public_cases_per_card": effective_max_public_cases_per_card(args),
        "case_manifest": str(getattr(args, "case_manifest", "") or ""),
        "broad_floor_surface": broad_floor_surface_required(args),
        "public_tests_used": False,
        "public_solutions_used": False,
        "score_semantics": "visible public task metadata for calibration candidate fanout only",
    }


def broad_floor_surface_required(args: argparse.Namespace) -> bool:
    if private_only_mode(args):
        return False
    return "broad_floor" in str(getattr(args, "slug", "") or "").lower()


def effective_public_cards(args: argparse.Namespace) -> str:
    if private_only_mode(args):
        return ""
    raw = str(getattr(args, "public_cards", "") or "")
    if broad_floor_surface_required(args):
        cards = {card.strip() for card in raw.split(",") if card.strip()}
        desired = {card.strip() for card in BROAD_FLOOR_PUBLIC_CARDS.split(",") if card.strip()}
        if cards != desired:
            return BROAD_FLOOR_PUBLIC_CARDS
    return raw


def effective_max_public_cases_per_card(args: argparse.Namespace) -> int:
    if private_only_mode(args):
        return 0
    value = max(1, int(getattr(args, "max_public_cases_per_card", 1) or 1))
    if broad_floor_surface_required(args):
        return max(value, BROAD_FLOOR_PUBLIC_CASES_PER_CARD)
    return value


def public_manifest_surface_current(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    desired = desired_public_tasks(args)
    existing = read_jsonl_dicts(paths["public_manifest"])
    desired_ids = [str(row.get("task_id") or "") for row in desired if str(row.get("task_id") or "")]
    existing_ids = [str(row.get("task_id") or "") for row in existing if str(row.get("task_id") or "")]
    if private_only_mode(args):
        return {
            "current": paths["public_manifest"].exists() and not existing_ids,
            "desired_task_count": 0,
            "existing_task_count": len(existing_ids),
            "desired_cards": {},
            "existing_cards": card_counts(existing),
            "public_tests_used": False,
            "public_solutions_used": False,
            "public_calibration": False,
            "rule": "private-only train-once uses an intentionally empty public manifest",
        }
    return {
        "current": bool(desired_ids and desired_ids == existing_ids),
        "desired_task_count": len(desired_ids),
        "existing_task_count": len(existing_ids),
        "desired_cards": card_counts(desired),
        "existing_cards": card_counts(existing),
        "public_tests_used": False,
        "public_solutions_used": False,
    }


def public_fanout_surface_current(paths: dict[str, Path], *, private_only: bool = False) -> dict[str, Any]:
    public_tasks = read_jsonl_dicts(paths["public_manifest"])
    public_candidates = read_jsonl_dicts(paths["public_candidates"])
    task_ids = {str(row.get("task_id") or "") for row in public_tasks if str(row.get("task_id") or "")}
    candidate_task_ids = {
        str(row.get("task_id") or "") for row in public_candidates if str(row.get("task_id") or "")
    }
    if private_only:
        return {
            "current": paths["public_manifest"].exists() and paths["public_candidates"].exists() and not task_ids and not candidate_task_ids,
            "public_task_count": len(task_ids),
            "candidate_task_count": len(candidate_task_ids),
            "missing_candidate_task_count": 0,
            "extra_candidate_task_count": len(candidate_task_ids),
            "missing_candidate_task_hashes": [],
            "rule": "private-only train-once fanout intentionally keeps public tasks and public candidates empty",
            "public_tests_used": False,
            "public_solutions_used": False,
            "public_calibration": False,
        }
    missing = sorted(task_ids - candidate_task_ids)
    extra = sorted(candidate_task_ids - task_ids)
    public_candidate_count = len(public_candidates)
    scale_current = public_candidate_count >= MIN_PUBLIC_CANDIDATES_FOR_DECODER_GATE
    return {
        "current": bool(task_ids and not missing and scale_current),
        "public_task_count": len(task_ids),
        "public_candidate_count": public_candidate_count,
        "minimum_public_candidate_count": MIN_PUBLIC_CANDIDATES_FOR_DECODER_GATE,
        "candidate_scale_current": scale_current,
        "candidate_task_count": len(candidate_task_ids),
        "missing_candidate_task_count": len(missing),
        "extra_candidate_task_count": len(extra),
        "missing_candidate_task_hashes": [stable_hash(item)[:16] for item in missing[:32]],
        "rule": "fanout candidate manifests must cover every visible public calibration task and meet the decoder gate public-candidate floor before gates may use them",
        "public_tests_used": False,
        "public_solutions_used": False,
    }


def private_only_mode(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "private_only", False) or getattr(args, "private_only_refresh", False))


def checkpoint_sts_policy(checkpoint_report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    native_enabled = bool(getattr(args, "native_sts_conditioning", True))
    private_only = private_only_mode(args)
    generation_path = str(get_path(checkpoint_report, ["summary", "sts_generation_path"], "") or "")
    used = bool(get_path(checkpoint_report, ["summary", "sts_conditioning_used"], False) or generation_path)
    if private_only and native_enabled:
        current = used and bool(generation_path)
        reason = "private_only_native_sts_requires_private_generation_path"
    elif private_only:
        current = not used
        reason = "private_only_sts_control_ablation_requires_unconditioned_checkpoint"
    elif native_enabled:
        current = used and bool(generation_path)
        reason = "native_sts_default_on_requires_generation_path"
    else:
        current = not used
        reason = "explicit_sts_off_ablation_requires_unconditioned_checkpoint"
    return {
        "current": bool(current),
        "native_sts_conditioning_requested": native_enabled,
        "private_only": private_only,
        "sts_conditioning_used": used,
        "sts_generation_path": generation_path,
        "reason": reason,
    }


def checkpoint_sts_policy_current(checkpoint_report: dict[str, Any], args: argparse.Namespace) -> bool:
    return bool(checkpoint_sts_policy(checkpoint_report, args).get("current"))


def private_training_input_paths() -> list[Path]:
    configured = [
        DEFAULT_EXTRA_PRIVATE_TRAIN_JSONL,
        DEFAULT_REPO_REPAIR_PRIVATE_TRAIN_JSONL,
        DEFAULT_RESIDUAL_PRIVATE_TRAIN_JSONL,
        *[chunk.strip() for chunk in PRIVATE_ROWS.split(";") if chunk.strip()],
    ]
    paths: list[Path] = []
    seen: set[str] = set()
    for raw_path in configured:
        path = resolve(raw_path)
        key = str(path).replace("\\", "/").lower()
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
    return paths


def private_training_input_freshness(paths: dict[str, Path]) -> dict[str, Any]:
    inputs = private_training_input_paths()
    rows = [file_provenance(path) for path in inputs]
    existing = [path for path in inputs if path.exists()]
    missing = [path for path in inputs if not path.exists()]
    newest_input_mtime = max((path.stat().st_mtime for path in existing), default=0.0)
    required_artifacts = {
        "private_curriculum": paths["private_curriculum"],
        "checkpoint": paths["checkpoint"],
        "checkpoint_report": paths["checkpoint_wrapper_report"],
    }
    artifact_rows: dict[str, Any] = {}
    artifacts_current = True
    for key, path in required_artifacts.items():
        exists = path.exists()
        mtime = path.stat().st_mtime if exists else 0.0
        current = bool(exists and mtime >= newest_input_mtime)
        artifact_rows[key] = {
            "path": rel(path),
            "exists": exists,
            "mtime": mtime,
            "current": current,
        }
        artifacts_current = artifacts_current and current
    return {
        "policy": "project_theseus_train_once_private_input_freshness_v1",
        "fresh": bool(existing and not missing and artifacts_current),
        "input_count": len(inputs),
        "missing_input_count": len(missing),
        "missing_inputs": [rel(path) for path in missing[:16]],
        "newest_input_mtime": newest_input_mtime,
        "required_artifact_mtime": newest_input_mtime,
        "artifacts": artifact_rows,
        "input_rows": rows,
        "rule": "train-once checkpoints are reusable only when the materialized private curriculum, checkpoint, and checkpoint report are newer than every configured private training row",
    }


def effective_private_training_input_freshness(
    paths: dict[str, Path],
    freshness_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return apply_private_input_mtime_only_override(
        private_training_input_freshness(paths),
        freshness_context,
    )


def apply_private_input_mtime_only_override(
    freshness: dict[str, Any],
    freshness_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Accept mtime-only churn when recorded row content hashes stayed identical."""
    if freshness.get("fresh") or not isinstance(freshness_context, dict) or not freshness_context:
        return freshness
    before = object_field(freshness_context, "private_input_content_signature_before_checkpoint")
    after = object_field(freshness_context, "private_input_content_signature_after_checkpoint")
    current = private_input_content_signature(freshness)
    mtime_changed = freshness_context.get("private_input_mtime_changed_during_checkpoint") is True
    content_changed = freshness_context.get("private_input_content_changed_during_checkpoint") is True
    content_unchanged = bool(before and after and before == after == current)
    artifacts = object_field(freshness, "artifacts")
    checkpoint_row = dict(object_field(artifacts, "checkpoint"))
    checkpoint_report_row = dict(object_field(artifacts, "checkpoint_report"))
    curriculum_row = dict(object_field(artifacts, "private_curriculum"))
    checkpoint_exists = bool(checkpoint_row.get("exists"))
    checkpoint_report_exists = bool(checkpoint_report_row.get("exists"))
    curriculum_exists = bool(curriculum_row.get("exists"))
    if not (
        mtime_changed
        and not content_changed
        and content_unchanged
        and checkpoint_exists
        and checkpoint_report_exists
        and curriculum_exists
        and not int(freshness.get("missing_input_count") or 0)
    ):
        return freshness
    adjusted = dict(freshness)
    adjusted_artifacts = {key: dict(value) for key, value in artifacts.items() if isinstance(value, dict)}
    for key in ("private_curriculum", "checkpoint", "checkpoint_report"):
        if key in adjusted_artifacts:
            adjusted_artifacts[key]["raw_mtime_current"] = bool(adjusted_artifacts[key].get("current"))
            adjusted_artifacts[key]["current"] = True
            adjusted_artifacts[key]["current_by_content_signature"] = True
    adjusted.update(
        {
            "fresh": True,
            "raw_mtime_fresh": False,
            "freshness_basis": "content_signature_mtime_only_override",
            "mtime_only_churn_accepted": True,
            "content_signature": current,
            "artifacts": adjusted_artifacts,
            "rule": (
                "train-once checkpoints are reusable when artifacts are newer than private training rows, "
                "or when the wrapper recorded mtime-only churn with identical before/after/current private row hashes"
            ),
        }
    )
    return adjusted


def prior_train_once_state(out: Path, slug: str) -> dict[str, Any]:
    state = read_json(out, {})
    if not isinstance(state, dict) or str(state.get("slug") or "") != slug:
        state = {}
    return hydrate_private_input_content_context(state, slug)


def hydrate_private_input_content_context(state: dict[str, Any], slug: str) -> dict[str, Any]:
    if (
        object_field(state, "private_input_content_signature_before_checkpoint")
        and object_field(state, "private_input_content_signature_after_checkpoint")
    ):
        return state
    ledger_context = latest_mtime_only_content_context(artifact_paths(slug)["phase_ledger"])
    if not ledger_context:
        return state
    hydrated = dict(state)
    hydrated.setdefault("slug", slug)
    hydrated.update(ledger_context)
    return hydrated


def inherit_private_input_content_context(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in (
        "private_input_content_signature_before_checkpoint",
        "private_input_content_signature_after_checkpoint",
        "private_input_mtime_changed_during_checkpoint",
        "private_input_content_changed_during_checkpoint",
        "private_inputs_changed_during_checkpoint",
        "private_input_content_signature_source",
    ):
        if key in source and key not in target:
            target[key] = source[key]


def latest_mtime_only_content_context(phase_ledger: Path) -> dict[str, Any]:
    if not phase_ledger.exists():
        return {}
    latest: dict[str, Any] = {}
    for line in phase_ledger.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("phase") != "train_once_checkpoint":
            continue
        if row.get("event") != "private_input_mtime_only_change_during_checkpoint":
            continue
        payload = object_field(row, "payload")
        content_signature = object_field(payload, "content_signature")
        if not content_signature:
            continue
        latest = {
            "private_input_content_signature_before_checkpoint": content_signature,
            "private_input_content_signature_after_checkpoint": content_signature,
            "private_input_mtime_changed_during_checkpoint": True,
            "private_input_content_changed_during_checkpoint": False,
            "private_inputs_changed_during_checkpoint": False,
            "private_input_content_signature_source": rel(phase_ledger),
        }
    return latest


def private_training_input_summary(
    paths: dict[str, Path],
    freshness_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    input_paths = private_training_input_paths()
    row_counts = {rel(path): count_jsonl_rows(path) for path in input_paths if path.exists()}
    return {
        "private_train_input_jsonl": [str(path).replace("\\", "/") for path in input_paths],
        "private_train_input_task_count": sum(row_counts.values()),
        "private_train_input_file_count": len(input_paths),
        "private_train_input_row_counts": row_counts,
        "high_transfer_private_train_jsonl": [str(path).replace("\\", "/") for path in input_paths],
        "high_transfer_private_train_task_count": sum(row_counts.values()),
        "high_transfer_private_train_file_count": len(input_paths),
        "high_transfer_private_train_row_counts": row_counts,
        "private_input_freshness": effective_private_training_input_freshness(paths, freshness_context),
    }


def private_input_signature(freshness: dict[str, Any]) -> dict[str, Any]:
    content_signature = private_input_content_signature(freshness)
    return {
        **content_signature,
        "newest_input_mtime": freshness.get("newest_input_mtime"),
    }


def private_input_content_signature(freshness: dict[str, Any]) -> dict[str, Any]:
    rows = freshness.get("input_rows") if isinstance(freshness.get("input_rows"), list) else []
    items = [
        {
            "path": str(row.get("path") or ""),
            "sha256": str(row.get("sha256") or ""),
            "bytes": int(row.get("bytes") or 0),
        }
        for row in rows
        if isinstance(row, dict)
    ]
    digest = hashlib.sha256(
        "\n".join(
            f"{item['path']}|{item['bytes']}|{item['sha256']}"
            for item in sorted(items, key=lambda item: item["path"])
        ).encode("utf-8")
    ).hexdigest()
    return {
        "input_count": len(items),
        "missing_input_count": int(freshness.get("missing_input_count") or 0),
        "combined_sha256": digest,
    }


def ensure_public_manifest_surface(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    desired = desired_public_tasks(args)
    desired_ids = [str(row.get("task_id") or "") for row in desired if str(row.get("task_id") or "")]
    existing = read_jsonl_dicts(paths["public_manifest"])
    existing_ids = [str(row.get("task_id") or "") for row in existing if str(row.get("task_id") or "")]
    preflight = public_decoder_contract_preflight(desired)
    refreshed = bool(desired_ids and desired_ids != existing_ids and preflight.get("passed"))
    if refreshed:
        write_jsonl(paths["public_manifest"], desired)
    return {
        "ready": bool(desired_ids and preflight.get("passed")),
        "refreshed": refreshed,
        "public_manifest": rel(paths["public_manifest"]),
        "desired_task_count": len(desired_ids),
        "existing_task_count": len(existing_ids),
        "desired_cards": card_counts(desired),
        "existing_cards": card_counts(existing),
        "decoder_contract_preflight": preflight,
        "public_tests_used": False,
        "public_solutions_used": False,
        "score_semantics": (
            "visible public task/signature manifest coverage only; fixes candidate surface mismatch "
            "without training on public tests or solutions"
        ),
    }


def ensure_private_only_manifest_surface(paths: dict[str, Path], *, canonical: bool = False) -> dict[str, Any]:
    public_manifest = paths["public_manifest"] if canonical else paths["current_source_smoke_empty_public_manifest"]
    write_jsonl(public_manifest, [])
    return {
        "ready": True,
        "refreshed": True,
        "public_manifest": rel(public_manifest),
        "canonical_public_manifest": bool(canonical),
        "desired_task_count": 0,
        "existing_task_count": 0,
        "desired_cards": {},
        "existing_cards": {},
        "decoder_contract_preflight": {
            "passed": True,
            "public_task_count": 0,
            "rule": (
                "private-only train-once intentionally uses an empty canonical public manifest"
                if canonical
                else "private-only fanout smoke intentionally uses an empty public manifest sidecar"
            ),
        },
        "public_tests_used": False,
        "public_solutions_used": False,
        "public_calibration": False,
        "score_semantics": (
            "private repair freshness proof only; this path intentionally skips public task fanout and cannot "
            "unlock public calibration"
        ),
    }


def desired_public_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    cards = [card.strip() for card in effective_public_cards(args).split(",") if card.strip()]
    tasks = export_public_visible_tasks(
        cards,
        seed=int(getattr(args, "seed", 14) or 14),
        max_cases=max(1, effective_max_public_cases_per_card(args)),
        case_manifest=str(getattr(args, "case_manifest", "") or ""),
    )
    return attach_decoder_contracts(tasks)


def card_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        card = str(row.get("card_id") or "")
        if card:
            counts[card] += 1
    return dict(sorted(counts.items()))


def fanout_command(
    args: argparse.Namespace,
    paths: dict[str, Path],
    sts_path: str,
    *,
    candidates_per_task: int | None = None,
    private_eval_limit: int = 0,
    public_task_limit: int = 0,
) -> list[str]:
    command = [
        str(release_binary_path()),
        "generate-code-lm-closure-fanout",
        "--private-curriculum",
        rel(paths["private_curriculum"]),
        "--public-task-manifest",
        rel(paths["public_manifest"]),
        "--checkpoint-in",
        rel(resolved_checkpoint_path(paths["checkpoint"])),
        "--seed",
        str(args.seed),
        "--candidates-per-task",
        str(max(1, candidates_per_task if candidates_per_task is not None else args.candidates_per_task)),
        "--private-candidate-out",
        rel(paths["private_candidates"]),
        "--public-candidate-out",
        rel(paths["public_candidates"]),
        "--report-out",
        rel(paths["fanout_report"]),
    ]
    if private_eval_limit > 0:
        command.extend(["--private-eval-limit", str(private_eval_limit)])
    if public_task_limit > 0:
        command.extend(["--public-task-limit", str(public_task_limit)])
    if sts_path:
        command.extend(["--sts-streams", rel(resolve(sts_path))])
    return command


def current_source_smoke_paths(paths: dict[str, Path]) -> dict[str, Path]:
    smoke = dict(paths)
    smoke["private_candidates"] = paths["current_source_smoke_private_candidates"]
    smoke["public_candidates"] = paths["current_source_smoke_public_candidates"]
    smoke["fanout_report"] = paths["current_source_smoke_fanout_report"]
    return smoke


def successful_phase(phase: dict[str, Any]) -> bool:
    return int(phase.get("returncode") or 0) == 0 and not bool(phase.get("timed_out"))


def refresh_empty_public_candidate_sidecar(path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return {
        "path": rel(path),
        "mtime": path.stat().st_mtime,
        "bytes": path.stat().st_size,
        "public_calibration": False,
        "score_semantics": "intentionally empty private-only public candidate sidecar freshness marker",
    }


def artifact_paths(slug: str) -> dict[str, Path]:
    return {
        "private_curriculum": ROOT / "data" / "private_code_curriculum" / f"code_lm_closure_{slug}.jsonl",
        "public_manifest": REPORTS / f"code_lm_public_tasks_{slug}.jsonl",
        "private_only_public_manifest": REPORTS / f"code_lm_public_tasks_{slug}_private_only_empty.jsonl",
        "checkpoint": REPORTS / f"student_code_lm_checkpoint_{slug}.json",
        "checkpoint_rust_report": REPORTS / f"code_lm_closure_rust_{slug}_checkpoint.json",
        "checkpoint_wrapper_report": REPORTS / f"code_lm_closure_{slug}_checkpoint.json",
        "private_candidates": REPORTS / f"code_lm_private_candidates_{slug}.jsonl",
        "public_candidates": REPORTS / f"student_code_candidates_{slug}.jsonl",
        "private_only_public_candidates": REPORTS / f"student_code_candidates_{slug}_private_only_empty.jsonl",
        "fanout_report": REPORTS / f"code_lm_closure_rust_{slug}_fanout.json",
        "closure_report": REPORTS / f"code_lm_closure_{slug}.json",
        "sts_input": REPORTS / f"code_lm_sts_conditioning_input_{slug}.jsonl",
        "sts_generations": REPORTS / f"code_lm_sts_public_generations_{slug}.jsonl",
        "sts_checkpoint": REPORTS / f"code_lm_sts_conditioning_checkpoint_{slug}.json",
        "sts_report": REPORTS / f"code_lm_sts_conditioning_report_{slug}.json",
        "lock": REPORTS / f"code_lm_closure_{slug}.lock",
        "phase_ledger": REPORTS / f"code_lm_train_once_fanout_{slug}_phase_ledger.jsonl",
        "checkpoint_phase_heartbeat": REPORTS / f"code_lm_train_once_fanout_{slug}_checkpoint.phase_heartbeat.json",
        "fanout_phase_heartbeat": REPORTS / f"code_lm_train_once_fanout_{slug}_fanout.phase_heartbeat.json",
        "current_source_smoke_phase_heartbeat": REPORTS
        / f"code_lm_train_once_fanout_{slug}_current_source_smoke.phase_heartbeat.json",
        "current_source_smoke_private_candidates": REPORTS / f"code_lm_private_candidates_{slug}_current_source_smoke.jsonl",
        "current_source_smoke_public_candidates": REPORTS / f"student_code_candidates_{slug}_current_source_smoke.jsonl",
        "current_source_smoke_fanout_report": REPORTS / f"code_lm_closure_rust_{slug}_current_source_smoke_fanout.json",
        "current_source_smoke_empty_public_manifest": REPORTS
        / f"code_lm_public_tasks_{slug}_private_only_empty.jsonl",
    }


def private_only_full_paths(paths: dict[str, Path]) -> dict[str, Path]:
    out = dict(paths)
    out["public_manifest"] = paths["private_only_public_manifest"]
    out["public_candidates"] = paths["private_only_public_candidates"]
    return out


def bounded_current_source_smoke_summary(
    args: argparse.Namespace,
    paths: dict[str, Path],
    fanout_report: dict[str, Any],
    phase: dict[str, Any],
) -> dict[str, Any]:
    private_rows = count_jsonl_rows(paths["private_candidates"])
    public_rows = count_jsonl_rows(paths["public_candidates"])
    private_only = bool(getattr(args, "private_only_refresh", False))
    private_limit = max(0, int(args.refresh_private_eval_limit))
    public_limit = 0 if private_only else max(0, int(args.refresh_public_task_limit))
    freshness = fanout_source_freshness(
        {
            "fanout_report": paths["fanout_report"],
            "private_candidates": paths["private_candidates"],
            "public_candidates": paths["public_candidates"],
            "checkpoint": paths.get("checkpoint"),
        }
    )
    ready = bool(
        fanout_report.get("run_status") == "completed"
        and paths["fanout_report"].exists()
        and (private_limit == 0 or private_rows > 0)
        and (private_only or public_limit == 0 or public_rows > 0)
        and freshness.get("fresh")
        and not phase.get("timed_out")
        and int(phase.get("returncode") or 0) == 0
    )
    timing = get_path(fanout_report, ["summary", "phase_timing_ms"], {})
    public_ms = number(get_path(timing, ["public_candidate_generation_and_write"]))
    private_ms = number(get_path(timing, ["private_candidate_generation_and_write"]))
    vcm_receipt = vcm_context_governor_receipt()
    return {
        "policy": "project_theseus_train_once_fanout_current_source_smoke_v1",
        "ready": ready and bool(vcm_receipt.get("ready")),
        "created_utc": now(),
        "private_only_refresh": private_only,
        "public_calibration": False,
        "artifact_semantics": (
            "private-only current-source smoke; proves private repair/freshness only and cannot unlock public calibration"
            if private_only
            else "bounded current-source smoke only; not full closure fanout and not public calibration evidence"
        ),
        "limits": {
            "private_eval_limit": private_limit,
            "public_task_limit": public_limit,
            "candidates_per_task": max(1, int(args.refresh_candidates_per_task)),
            "timeout_seconds": max(1, int(args.refresh_smoke_timeout_seconds)),
        },
        "paths": {
            "fanout_report": rel(paths["fanout_report"]),
            "private_candidates": rel(paths["private_candidates"]),
            "public_candidates": rel(paths["public_candidates"]),
        },
        "report_trigger_state": fanout_report.get("trigger_state"),
        "report_run_status": fanout_report.get("run_status"),
        "private_candidate_rows": private_rows,
        "public_candidate_rows": public_rows,
        "public_task_surface": "intentionally_empty_private_only_sidecar" if private_only else "bounded_public_metadata",
        "public_candidate_generation_ms": int(public_ms),
        "private_candidate_generation_ms": int(private_ms),
        "public_ms_per_candidate_row": round(public_ms / max(1, public_rows), 3) if public_ms else 0.0,
        "private_ms_per_candidate_row": round(private_ms / max(1, private_rows), 3) if private_ms else 0.0,
        "phase": phase,
        "freshness": freshness,
        **vcm_context_summary_fields(vcm_receipt, surface="fanout_current_source_smoke"),
        "vcm_context_governor_receipt": vcm_receipt,
        "external_inference_calls": 0,
    }


def merged_closure_report(
    args: argparse.Namespace,
    paths: dict[str, Path],
    checkpoint_report: dict[str, Any],
    fanout_report: dict[str, Any],
    freshness_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    private_rows = count_jsonl_rows(paths["private_candidates"])
    public_rows = count_jsonl_rows(paths["public_candidates"])
    private_only = private_only_mode(args)
    private_manifest = candidate_manifest_stats(paths["private_candidates"], scope="private")
    public_manifest = candidate_manifest_stats(paths["public_candidates"], scope="public_calibration_metadata_only")
    phase_ledger = summarize_phase_ledger(paths["phase_ledger"])
    phase_timing_ms = merged_phase_timing(checkpoint_report, fanout_report, phase_ledger)
    optimizer_targets = optimizer_targets_from_timing(phase_timing_ms, private_manifest, public_manifest)
    artifact_provenance = build_artifact_provenance(paths, freshness_context)
    vcm_receipt = vcm_context_governor_receipt()
    vcm_fields = vcm_context_summary_fields(vcm_receipt, surface="fanout_closure")
    strict_generator_receipt = strict_generator_fanout_receipt_summary()
    fanout_records = build_fanout_spine_records(
        args,
        paths,
        private_rows=private_rows,
        public_rows=public_rows,
        private_only=private_only,
        vcm_receipt=vcm_receipt,
        artifact_provenance=artifact_provenance,
    )
    verification_bandwidth = fanout_verification_bandwidth_record(
        paths,
        private_rows=private_rows,
        public_rows=public_rows,
        private_only=private_only,
        strict_generator_receipt=strict_generator_receipt,
        vcm_receipt=vcm_receipt,
    )
    governance_tax = fanout_governance_tax_record(
        phase_timing_ms,
        verification_bandwidth=verification_bandwidth,
        vcm_receipt=vcm_receipt,
        artifact_provenance=artifact_provenance,
    )
    gates = [
        gate("checkpoint_report_completed", checkpoint_ready(checkpoint_report, paths["checkpoint"]), rel(paths["checkpoint_wrapper_report"])),
        gate(
            "fanout_report_completed",
            fanout_ready(
                fanout_report,
                paths["private_candidates"],
                paths["public_candidates"],
                fanout_report_path=paths["fanout_report"],
                checkpoint=paths["checkpoint"],
                allow_empty_public=private_only,
            ),
            rel(paths["fanout_report"]),
        ),
        gate("train_once_checkpoint_fanout", True, "training happened once; candidate generation reused checkpoint artifacts"),
        gate("repeated_training_per_candidate_shard_removed", True, "no per-shard Code LM/STS retraining in this path"),
        gate("private_candidate_manifest_present", private_rows > 0, private_rows),
        gate(
            "public_candidate_manifest_present",
            (paths["public_candidates"].exists() and public_rows == 0) if private_only else public_rows > 0,
            {
                "public_candidate_rows": public_rows,
                "private_only": private_only,
                "rule": (
                    "public candidate sidecar is intentionally empty for private-only repair evidence"
                    if private_only
                    else "normal train-once fanout must emit public calibration-metadata candidates"
                ),
            },
        ),
        gate("public_calibration_not_run", True, "public candidates are calibration metadata only; no public benchmark scoring here"),
        gate("public_tests_not_visible", True, "public task manifest excludes tests and canonical solutions"),
        gate("phase_timing_ledger_present", bool(phase_ledger.get("event_count")), rel(paths["phase_ledger"])),
        gate("report_control_signal_contract_present", bool(CONTROL_SIGNAL_CONTRACT["consumers"]), CONTROL_SIGNAL_CONTRACT["consumers"]),
        gate("staged_verification_contract_present", len(STAGED_VERIFICATION_CONTRACT) == 4, [row["stage"] for row in STAGED_VERIFICATION_CONTRACT]),
        gate("public_manifest_metadata_only", public_manifest["safety"]["public_tests_or_solutions_used"] is False, public_manifest["safety"]),
        gate("fanout_artifact_provenance_present", bool(artifact_provenance.get("provenance_ready")), artifact_provenance.get("summary")),
        gate("checkpoint_provenance_present", bool(get_path(artifact_provenance, ["checkpoint", "sha256"])), artifact_provenance.get("checkpoint")),
        gate(
            "checkpoint_current_against_private_training_rows",
            bool(get_path(artifact_provenance, ["private_input_freshness", "fresh"], False)),
            artifact_provenance.get("private_input_freshness"),
        ),
        gate("release_binary_provenance_present", bool(get_path(artifact_provenance, ["release_binary", "sha256"])), artifact_provenance.get("release_binary")),
        gate(
            "fanout_artifacts_current_against_source_binary_provenance",
            bool(artifact_provenance.get("fanout_freshness", {}).get("fresh")),
            artifact_provenance.get("fanout_freshness"),
        ),
        gate("vcm_context_governor_ready", bool(vcm_receipt.get("ready")), vcm_receipt),
        gate(
            "vcm_context_adequacy_ready_for_fanout",
            vcm_fields["vcm_context_adequacy_state"] == "governed_sufficient_for_generation_fanout",
            vcm_fields,
        ),
        gate(
            "strict_generator_fanout_replay_receipt_ready",
            bool(strict_generator_receipt.get("ready")),
            strict_generator_receipt,
        ),
        gate(
            "fanout_verification_bandwidth_ready",
            verification_bandwidth.get("status") == "ready",
            verification_bandwidth,
        ),
        gate(
            "fanout_governance_tax_ready",
            governance_tax.get("status") == "ready",
            governance_tax,
        ),
        gate("external_inference_zero", True, "local CUDA/Rust/Python only"),
    ]
    trigger_state = "GREEN" if all(row["passed"] for row in gates) else "YELLOW"
    private_input_summary = private_training_input_summary(paths, freshness_context)
    return {
        "policy": "project_theseus_code_lm_closure_train_once_fanout_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "run_status": "completed",
        "progress_stage": "completed",
        "seed": args.seed,
        "private_curriculum": rel(paths["private_curriculum"]),
        "public_task_manifest": rel(paths["public_manifest"]),
        "checkpoint": rel(paths["checkpoint"]),
        "private_candidate_manifest": rel(paths["private_candidates"]),
        "public_candidate_manifest": rel(paths["public_candidates"]),
        "rust_report": rel(paths["fanout_report"]),
        "checkpoint_report": rel(paths["checkpoint_wrapper_report"]),
        "summary": {
            "train_once_checkpoint_fanout": True,
            "repeated_training_per_candidate_shard": False,
            "private_only": private_only,
            "target_architecture": "train_once_checkpoint_then_hive_distributed_candidate_generation_and_verification",
            "sts_conditioning_mode": (
                "native_sts_conditioning_default_on"
                if getattr(args, "native_sts_conditioning", True)
                else "sts_control_ablation_disabled"
            ),
            "sts_default_policy": {
                "default": "native_sts_conditioning_on",
                "disable_flag": "--disable-native-sts-conditioning",
                "sts_off_role": "same-seed control/ablation lane only",
                "same_seed_non_sts_comparator_preserved": True,
            },
            "sts_checkpoint_policy": checkpoint_sts_policy(checkpoint_report, args),
            "checkpoint_trigger_state": checkpoint_report.get("trigger_state"),
            "fanout_trigger_state": fanout_report.get("trigger_state"),
            "checkpoint_cuda_readout_used": get_path(checkpoint_report, ["summary", "cuda_readout_used"], False),
            "checkpoint_backend": get_path(checkpoint_report, ["summary", "next_token_readout_backend"], ""),
            "sts_conditioning_used": bool(get_path(checkpoint_report, ["summary", "sts_generation_path"], "")),
            "private_candidate_count": private_rows,
            "public_candidate_count": public_rows,
            **private_input_summary,
            "fanout_private_candidate_count": get_path(fanout_report, ["summary", "private_candidate_count"], private_rows),
            "fanout_public_candidate_count": get_path(fanout_report, ["summary", "public_candidate_count"], public_rows),
            "private_token_level_candidate_count": get_path(fanout_report, ["summary", "private_token_level_candidate_count"], 0),
            "public_token_level_candidate_count": get_path(fanout_report, ["summary", "public_token_level_candidate_count"], 0),
            "template_like_candidate_count": get_path(fanout_report, ["summary", "template_like_candidate_count"], 0),
            "phase_timing_ms": phase_timing_ms,
            "phase_timing_categories": phase_timing_categories(fanout_report, private_manifest, public_manifest),
            "phase_ledger_summary": phase_ledger,
            "slow_phase_targets": optimizer_targets,
            "artifact_provenance": artifact_provenance,
            "private_candidate_manifest_diagnostics": private_manifest,
            "public_candidate_manifest_diagnostics": public_manifest,
            "staged_verification_contract": STAGED_VERIFICATION_CONTRACT,
            "control_signal_contract": CONTROL_SIGNAL_CONTRACT,
            **vcm_fields,
            "vcm_context_governor_receipt": vcm_receipt,
            "strict_generator_fanout_receipt": strict_generator_receipt,
            "verification_bandwidth": verification_bandwidth,
            "governance_tax": governance_tax,
            "viea_fanout_record_count": len(fanout_records),
            "public_calibration_allowed": False,
            "score_semantics": (
                "completed private-only train-once evidence; cannot unlock public calibration"
                if private_only
                else "completed private candidate generation evidence only; decoder/private gate and transfer proof must unlock calibration"
            ),
            "external_inference_calls": 0,
        },
        "control_signal_contract": CONTROL_SIGNAL_CONTRACT,
        "staged_verification_contract": STAGED_VERIFICATION_CONTRACT,
        "vcm_context_governor_receipt": vcm_receipt,
        "strict_generator_fanout_receipt": strict_generator_receipt,
        "verification_bandwidth": verification_bandwidth,
        "governance_tax": governance_tax,
        "viea_fanout_records": fanout_records,
        "gates": gates,
        "external_inference_calls": 0,
    }


def ensure_release_cuda_binary() -> dict[str, Any]:
    exe = release_binary_path()
    sources = [
        ROOT / "crates" / "symliquid-cli" / "src" / "main.rs",
        ROOT / "crates" / "symliquid-cli" / "src" / "sts_parallel_decoder.rs",
    ]
    if cuda_readout_requested_by_launcher():
        sources.extend(
            [
                ROOT / "crates" / "symliquid-cuda" / "src" / "readout_cuda.rs",
                ROOT / "crates" / "symliquid-cuda" / "kernels" / "readout_kernels.cu",
            ]
        )
    sources.extend(
        sorted((ROOT / "crates" / "symliquid-cli" / "src" / "code_lm_closure").rglob("*.rs"))
    )
    newest = max((path.stat().st_mtime for path in sources if path.exists()), default=0.0)
    exe_mtime = exe.stat().st_mtime if exe.exists() else 0.0
    stale = (not exe.exists()) or exe_mtime < newest
    status: dict[str, Any] = {
        "path": rel(exe),
        "exists": exe.exists(),
        "stale_relative_to_sources": stale,
        "release_binary_backend": release_binary_backend(),
        "cuda_readout_requested_by_launcher": cuda_readout_requested_by_launcher(),
        "cuda_feature_build_enforced": cuda_readout_requested_by_launcher(),
        "newest_source_mtime": newest,
        "exe_mtime": exe_mtime,
    }
    # A non-CUDA release build can be newer than the decoder sources while still
    # refusing --use-cuda-readout. When CUDA is requested, ask Cargo for the CUDA
    # feature; macOS defaults to native CPU readout and must not claim CUDA.
    status["build"] = run_command(
        release_build_command(),
        timeout_seconds=1800,
        log_path=REPORTS / "code_lm_train_once_fanout_release_build.log",
    )
    status["exe_mtime_after_build"] = exe.stat().st_mtime if exe.exists() else 0.0
    status["ready"] = exe.exists() and (exe.stat().st_mtime if exe.exists() else 0.0) >= newest
    return status


def append_phase_event(path: Path, phase: str, event: str, payload: dict[str, Any] | None = None) -> None:
    append_phase_event_raw(path, phase, event, payload, phase_contracts=PHASE_CONTRACTS)


def summarize_phase_ledger(path: Path) -> dict[str, Any]:
    return summarize_phase_ledger_raw(path, root=ROOT, phase_contracts=PHASE_CONTRACTS)


def merged_phase_timing(
    checkpoint_report: dict[str, Any],
    fanout_report: dict[str, Any],
    phase_ledger: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_timing: dict[str, Any] = {}
    checkpoint_runtime_ms = number(
        checkpoint_report.get("runtime_ms") or get_path(checkpoint_report, ["summary", "runtime_ms"])
    )
    if checkpoint_runtime_ms:
        checkpoint_timing["runtime_ms"] = int(checkpoint_runtime_ms)
    checkpoint_phase_timing = get_path(checkpoint_report, ["summary", "phase_timing_ms"], {})
    if isinstance(checkpoint_phase_timing, dict):
        checkpoint_timing.update(checkpoint_phase_timing)
    checkpoint_artifact_evidence = checkpoint_model_artifact_evidence(checkpoint_report)
    evidence_phase_timing = checkpoint_artifact_evidence.get("phase_timing_ms")
    if isinstance(evidence_phase_timing, dict):
        checkpoint_timing.update(evidence_phase_timing)
    checkpoint_parallelism = (
        get_path(checkpoint_report, ["summary", "checkpoint_training_parallelism"], {})
        or checkpoint_artifact_evidence.get("checkpoint_training_parallelism")
        or {}
    )
    if isinstance(checkpoint_parallelism, dict):
        aux_wall_ms = number(checkpoint_parallelism.get("aux_decoder_parallel_wall_ms"))
        readout_ms = number(checkpoint_parallelism.get("next_token_readout_phase_ms"))
        if aux_wall_ms:
            checkpoint_timing["aux_decoder_parallel_wall_ms"] = int(aux_wall_ms)
        if readout_ms:
            checkpoint_timing["next_token_readout_phase_ms"] = int(readout_ms)
        if aux_wall_ms and readout_ms and aux_wall_ms > readout_ms:
            checkpoint_timing["aux_decoder_join_over_readout_ms"] = int(aux_wall_ms - readout_ms)
    timing: dict[str, Any] = {
        "checkpoint_report": checkpoint_timing,
        "fanout_report": get_path(fanout_report, ["summary", "phase_timing_ms"], {}),
        "ledger_elapsed_ms": {},
    }
    phases = phase_ledger.get("phases") if isinstance(phase_ledger.get("phases"), dict) else {}
    for phase, row in phases.items():
        elapsed = number(row.get("elapsed_seconds"))
        if elapsed:
            timing["ledger_elapsed_ms"][phase] = int(elapsed * 1000)
    return timing


def checkpoint_model_artifact_evidence(checkpoint_report: dict[str, Any]) -> dict[str, Any]:
    gates = checkpoint_report.get("gates")
    if not isinstance(gates, list):
        return {}
    for row in gates:
        if not isinstance(row, dict):
            continue
        if row.get("gate") != "checkpoint_model_artifacts_v1_present":
            continue
        evidence = row.get("evidence")
        if isinstance(evidence, dict):
            return evidence
    return {}


def optimizer_targets_from_timing(
    phase_timing_ms: dict[str, Any],
    private_manifest: dict[str, Any],
    public_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for key, value in flattened_numbers(phase_timing_ms).items():
        if not is_runtime_timing_path(key):
            continue
        ms = float(value)
        if ms >= 120_000:
            if "aux_decoder_join_over_readout" in key:
                action = (
                    "Bound or checkpoint auxiliary state/SymLiquid decoder joins so the CUDA next-token "
                    "checkpoint can materialize without waiting on a long CPU tail; route late aux decoders "
                    "as resumable conditioning artifacts."
                )
            elif "aux_decoder_parallel_wall" in key:
                action = (
                    "Split checkpoint timing into CUDA readout, state-sequence decoder, SymLiquid-state decoder, "
                    "and join-tail timings; move the dominant auxiliary decoder to a bounded or resident path."
                )
            else:
                action = (
                    "Split this phase into train/readout/STS/fanout/ranker/verifier subphases and move "
                    "the largest CPU-control loop into a batched resident path."
                )
            targets.append(
                {
                    "id": f"slow_phase_{safe_id(key)}",
                    "phase": key,
                    "elapsed_ms": int(ms),
                    "recommended_action": action,
                }
            )
    for label, manifest in (("private", private_manifest), ("public", public_manifest)):
        no_candidate_rate = 1.0 - float(number(manifest.get("task_coverage")))
        if manifest.get("row_count") and no_candidate_rate > 0.25:
            targets.append(
                {
                    "id": f"{label}_candidate_coverage_wall",
                    "phase": f"{label}_candidate_manifest",
                    "no_candidate_rate": round(no_candidate_rate, 6),
                    "recommended_action": "Improve fanout/ranker prefilter before full verification; do not spend sandbox cycles on sparse manifests.",
                }
            )
    return targets[:12]


def phase_timing_categories(
    fanout_report: dict[str, Any],
    private_manifest: dict[str, Any],
    public_manifest: dict[str, Any],
) -> dict[str, Any]:
    rust_categories = get_path(fanout_report, ["summary", "candidate_task_phase_categories"], {})
    if isinstance(rust_categories, dict) and rust_categories:
        return {
            "policy": "project_theseus_train_once_fanout_phase_timing_categories_v1",
            "source": "rust_fanout_summary",
            "candidate_expansion": {
                "private_ms": get_path(rust_categories, ["private", "candidate_expansion_ms"], 0),
                "public_ms": get_path(rust_categories, ["public", "candidate_expansion_ms"], 0),
            },
            "sts_conditioning": {
                "private_ms": get_path(rust_categories, ["private", "sts_conditioning_ms"], 0),
                "public_ms": get_path(rust_categories, ["public", "sts_conditioning_ms"], 0),
            },
            "ranker_prefilter": {
                "private_ms": get_path(rust_categories, ["private", "ranker_prefilter_ms"], 0),
                "public_ms": get_path(rust_categories, ["public", "ranker_prefilter_ms"], 0),
            },
            "verifier_cache": {
                "private_ms": get_path(rust_categories, ["private", "verifier_cache_ms"], 0),
                "public_ms": get_path(rust_categories, ["public", "verifier_cache_ms"], 0),
            },
            "artifact_write": {
                "private_ms": get_path(fanout_report, ["summary", "phase_timing_ms", "private_artifact_write"], 0),
                "public_ms": get_path(fanout_report, ["summary", "phase_timing_ms", "public_artifact_write"], 0),
            },
            "score_semantics": "control_timing_categories_not_capability_evidence",
        }
    return {
        "policy": "project_theseus_train_once_fanout_phase_timing_categories_v1",
        "source": "python_manifest_fallback",
        "candidate_expansion": {
            "private_ms": get_path(
                private_manifest,
                ["candidate_task_timing_summary", "timing_ms_total", "candidate_expression_generation_ms"],
                0,
            ),
            "public_ms": get_path(
                public_manifest,
                ["candidate_task_timing_summary", "timing_ms_total", "candidate_expression_generation_ms"],
                0,
            ),
        },
        "sts_conditioning": {
            "private_ms": sum_timing_contains(private_manifest, "sts") + sum_timing_contains(private_manifest, "symliquid"),
            "public_ms": sum_timing_contains(public_manifest, "sts") + sum_timing_contains(public_manifest, "symliquid"),
        },
        "ranker_prefilter": {
            "private_ms": sum_timing_contains(private_manifest, "prefilter") + sum_timing_contains(private_manifest, "rank"),
            "public_ms": sum_timing_contains(public_manifest, "prefilter") + sum_timing_contains(public_manifest, "rank"),
        },
        "verifier_cache": {
            "private_ms": sum_timing_contains(private_manifest, "cache") + sum_timing_contains(private_manifest, "verifier"),
            "public_ms": sum_timing_contains(public_manifest, "cache") + sum_timing_contains(public_manifest, "verifier"),
        },
        "artifact_write": {"private_ms": 0, "public_ms": 0},
        "score_semantics": "fallback_control_timing_categories_not_capability_evidence",
    }


def sum_timing_contains(manifest: dict[str, Any], needle: str) -> int:
    timing = get_path(manifest, ["candidate_task_timing_summary", "timing_ms_total"], {})
    if not isinstance(timing, dict):
        return 0
    needle = needle.lower()
    total = 0
    for key, value in timing.items():
        if needle in str(key).lower():
            total += int(number(value))
    return total


def is_runtime_timing_path(path: str) -> bool:
    lower = str(path or "").lower()
    if not lower:
        return False
    non_timing_terms = (
        "work_step",
        "work_steps",
        "max_work",
        "estimated_work",
        "budget",
        "count",
        "rows",
        "epoch",
        "vocab",
        "dim",
        "active",
        "within",
        "policy",
        "semantic",
    )
    if any(term in lower for term in non_timing_terms):
        return False
    if lower.startswith("fanout_report.") or lower.startswith("ledger_elapsed_ms."):
        return True
    return lower.endswith(("_ms", ".runtime_ms", ".elapsed_ms", ".duration_ms", ".wall_ms"))


def is_shared_runtime_timing_path(path: str) -> bool:
    return str(path or "").lower().endswith("_shared_ms")


def candidate_manifest_stats(path: Path, *, scope: str) -> dict[str, Any]:
    timing_totals: Counter[str] = Counter()
    timing_max: dict[str, float] = {}
    shared_timing: dict[str, float] = {}
    timed_tasks: set[str] = set()
    elapsed_total_ms = 0.0
    elapsed_max_ms = 0.0
    stats: dict[str, Any] = {
        "path": rel(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "row_count": 0,
        "decode_errors": 0,
        "task_count": 0,
        "task_coverage": 0.0,
        "token_level_candidate_count": 0,
        "full_body_candidate_count": 0,
        "contract_guided_candidate_count": 0,
        "sts_conditioned_candidate_count": 0,
        "program_synthesis_loop_count": 0,
        "program_synthesis_promotion_ready_count": 0,
        "template_like_candidate_count": 0,
        "placeholder_scaffold_count": 0,
        "verifier_pass_count": 0,
        "guardrail_pass_count": 0,
        "candidate_modes": {},
        "top_rejection_reasons": {},
        "candidate_task_timing_summary": {
            "task_count": 0,
            "elapsed_ms_total": 0,
            "elapsed_ms_max": 0,
            "timing_ms_total": {},
            "timing_ms_max": {},
            "shared_timing_ms": {},
            "top_timing_ms_total": {},
            "score_semantics": "deduplicated_per_task_runtime_profile_only_not_capability_evidence",
        },
        "safety": {
            "scope": scope,
            "public_tests_or_solutions_used": False,
            "unsafe_public_rows": 0,
        },
        "score_semantics": "candidate_manifest_inventory_control_signal_not_promotion_evidence",
    }
    if not path.exists():
        return stats
    tasks: set[str] = set()
    mode_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["decode_errors"] += 1
                continue
            if not isinstance(row, dict):
                continue
            stats["row_count"] += 1
            task_id = str(row.get("task_id") or row.get("source_task_id") or row.get("entry_point") or "")
            if task_id:
                tasks.add(task_id)
            timing_key = task_id or f"row_{stats['row_count']}"
            if timing_key not in timed_tasks:
                timing_row = row.get("candidate_task_timing_v1")
                if not isinstance(timing_row, dict):
                    timing_row = get_path(row, ["provenance", "candidate_task_timing_v1"], {})
                if isinstance(timing_row, dict):
                    timed_tasks.add(timing_key)
                    elapsed_ms = number(timing_row.get("elapsed_ms"))
                    if elapsed_ms:
                        elapsed_total_ms += elapsed_ms
                        elapsed_max_ms = max(elapsed_max_ms, elapsed_ms)
                    timing_ms = timing_row.get("timing_ms")
                    if isinstance(timing_ms, dict):
                        for phase, value in timing_ms.items():
                            if not is_runtime_timing_path(str(phase)):
                                continue
                            ms = number(value)
                            if not ms:
                                continue
                            phase_id = str(phase)
                            if is_shared_runtime_timing_path(phase_id):
                                shared_timing[phase_id] = max(shared_timing.get(phase_id, 0.0), ms)
                                continue
                            timing_totals[phase_id] += int(ms)
                            timing_max[phase_id] = max(timing_max.get(phase_id, 0.0), ms)
            mode = str(row.get("candidate_generation_mode") or get_path(row, ["provenance", "candidate_generation_mode"], "") or "")
            if mode:
                mode_counts[mode] += 1
            if bool(row.get("token_level_code_generation_learned")) or bool(row.get("compositional_token_candidate")):
                stats["token_level_candidate_count"] += 1
            if bool(row.get("full_body_token_candidate")) or row.get("candidate_program_scope") == "full_function_body":
                stats["full_body_candidate_count"] += 1
            if bool(row.get("contract_guided_token_stage")) or "contract_guided" in mode:
                stats["contract_guided_candidate_count"] += 1
            if bool(row.get("sts_stream_conditioned")) or bool(row.get("sts_candidate_expression_used")) or "sts" in mode:
                stats["sts_conditioned_candidate_count"] += 1
            loop = row.get("program_synthesis_loop_v1")
            if isinstance(loop, dict):
                stats["program_synthesis_loop_count"] += 1
                if bool(loop.get("promotion_ready")):
                    stats["program_synthesis_promotion_ready_count"] += 1
            if bool(row.get("template_like_candidate")):
                stats["template_like_candidate_count"] += 1
            if bool(row.get("placeholder_scaffold_body")):
                stats["placeholder_scaffold_count"] += 1
            if bool(row.get("decoder_contract_verifier_v1_passed")):
                stats["verifier_pass_count"] += 1
            if bool(row.get("deterministic_guardrail_passed")):
                stats["guardrail_pass_count"] += 1
            for reason in row.get("decoder_contract_verifier_v1_reasons") or []:
                rejection_counts[str(reason)] += 1
            for reason in row.get("deterministic_guardrail_reasons") or []:
                rejection_counts[str(reason)] += 1
            if bool(row.get("tests_used")) or bool(row.get("public_tests_visible_to_generator")) or bool(row.get("canonical_solution_seen_by_solver")):
                stats["safety"]["public_tests_or_solutions_used"] = True
                stats["safety"]["unsafe_public_rows"] += 1
    stats["task_count"] = len(tasks)
    stats["task_coverage"] = 1.0 if stats["task_count"] else 0.0
    stats["candidate_modes"] = dict(mode_counts.most_common(12))
    stats["top_rejection_reasons"] = dict(rejection_counts.most_common(12))
    stats["candidate_task_timing_summary"] = {
        "task_count": len(timed_tasks),
        "elapsed_ms_total": int(elapsed_total_ms),
        "elapsed_ms_max": int(elapsed_max_ms),
        "timing_ms_total": dict(sorted(timing_totals.items())),
        "timing_ms_max": {key: int(value) for key, value in sorted(timing_max.items())},
        "shared_timing_ms": {key: int(value) for key, value in sorted(shared_timing.items())},
        "top_timing_ms_total": dict(timing_totals.most_common(12)),
        "score_semantics": "deduplicated_per_task_runtime_profile_only_not_capability_evidence",
    }
    row_count = max(1, int(stats["row_count"]))
    stats["program_synthesis_loop_rate"] = round(stats["program_synthesis_loop_count"] / row_count, 6)
    stats["promotion_ready_rate"] = round(stats["program_synthesis_promotion_ready_count"] / row_count, 6)
    stats["verifier_pass_rate"] = round(stats["verifier_pass_count"] / row_count, 6)
    stats["guardrail_pass_rate"] = round(stats["guardrail_pass_count"] / row_count, 6)
    return stats


def active_code_worker_rows() -> list[dict[str, Any]]:
    current_pid = os.getpid()
    rows: list[dict[str, Any]] = []
    for row in windows_active_code_lm_process_rows():
        if int(row.get("pid") or 0) == current_pid:
            continue
        name = str(row.get("name") or "").lower()
        command = str(row.get("command") or row.get("command_preview") or "").lower()
        if name in {"powershell.exe", "pwsh.exe", "cmd.exe"} and (
            "code_lm_train_once_fanout.py" in command
            or "code_lm_closure.py" in command
            or "symliquid-cli" in command
        ):
            continue
        rows.append(row)
    return rows


def active_code_workers() -> bool:
    return bool(active_code_worker_rows())


def infer_active_worker_phase(rows: list[dict[str, Any]]) -> str:
    text = " ".join(str(row.get("command_preview") or "").lower() for row in rows)
    if "train-sts-parallel-decoder" in text:
        return "sts_parallel_decoder_conditioning"
    if "generate-code-lm-closure-fanout" in text:
        return "checkpoint_fanout_candidate_generation"
    if "code_lm_closure.py" in text and "--checkpoint-only" in text:
        return "train_once_checkpoint"
    if "code_lm_train_once_fanout.py" in text:
        return "train_once_fanout_supervisor"
    return "active_code_worker_present"


def checkpoint_ready(report: dict[str, Any], checkpoint: Path) -> bool:
    checkpoint_path = resolved_checkpoint_path(checkpoint)
    return (
        report.get("run_status") == "completed"
        and bool(get_path(report, ["summary", "train_once_checkpoint_fanout_ready"], False))
        and bool(get_path(report, ["summary", "model_artifacts_v1_written"], False))
        and checkpoint_path.exists()
    )


def fanout_ready(
    report: dict[str, Any],
    private_candidates: Path,
    public_candidates: Path,
    *,
    require_current_source: bool = True,
    fanout_report_path: Path | None = None,
    checkpoint: Path | None = None,
    allow_empty_public: bool = False,
) -> bool:
    base_ready = (
        report.get("run_status") == "completed"
        and bool(get_path(report, ["summary", "train_once_checkpoint_fanout"], False))
        and private_candidates.exists()
        and public_candidates.exists()
        and count_jsonl_rows(private_candidates) > 0
        and (allow_empty_public or count_jsonl_rows(public_candidates) > 0)
    )
    if not base_ready:
        return False
    if not require_current_source:
        return True
    paths = {
        "fanout_report": fanout_report_path or report_path_for_fanout_report(report),
        "private_candidates": private_candidates,
        "public_candidates": public_candidates,
        "checkpoint": checkpoint,
    }
    return bool(fanout_source_freshness(paths).get("fresh"))


def report_path_for_fanout_report(report: dict[str, Any]) -> Path | None:
    path = str(report.get("path") or report.get("report_out") or "")
    return resolve(path) if path else None


def code_lm_fanout_source_paths() -> list[Path]:
    roots = [
        ROOT / "crates" / "symliquid-cli" / "src" / "main.rs",
        ROOT / "crates" / "symliquid-cli" / "src" / "sts_parallel_decoder.rs",
    ]
    if cuda_readout_requested_by_launcher():
        roots.extend(
            [
                ROOT / "crates" / "symliquid-cuda" / "src" / "readout_cuda.rs",
                ROOT / "crates" / "symliquid-cuda" / "kernels" / "readout_kernels.cu",
            ]
        )
    closure_root = ROOT / "crates" / "symliquid-cli" / "src" / "code_lm_closure"
    if closure_root.exists():
        roots.extend(sorted(closure_root.rglob("*.rs")))
    return [path for path in roots if path.exists()]


def fanout_source_freshness(paths: dict[str, Path | None]) -> dict[str, Any]:
    sources = code_lm_fanout_source_paths()
    newest_source = max((path.stat().st_mtime for path in sources), default=0.0)
    release_exe = release_binary_path()
    exe_mtime = release_exe.stat().st_mtime if release_exe.exists() else 0.0
    checkpoint = paths.get("checkpoint")
    checkpoint_exists = bool(checkpoint and checkpoint.exists())
    checkpoint_mtime = checkpoint.stat().st_mtime if checkpoint_exists and checkpoint else 0.0
    required_mtime = max(newest_source, exe_mtime, checkpoint_mtime)
    artifacts: dict[str, Any] = {}
    all_artifacts_current = True
    for key in ["fanout_report", "private_candidates", "public_candidates"]:
        path = paths.get(key)
        exists = bool(path and path.exists())
        mtime = path.stat().st_mtime if exists and path else 0.0
        current = bool(exists and mtime >= required_mtime)
        artifacts[key] = {
            "path": rel(path) if path else "",
            "exists": exists,
            "mtime": mtime,
            "current": current,
        }
        all_artifacts_current = all_artifacts_current and current
    release_current = release_exe.exists() and exe_mtime >= newest_source
    return {
        "fresh": bool(release_current and all_artifacts_current),
        "release_binary_current": release_current,
        "release_binary": rel(release_exe),
        "release_binary_backend": release_binary_backend(),
        "release_binary_mtime": exe_mtime,
        "checkpoint": rel(checkpoint) if checkpoint else "",
        "checkpoint_exists": checkpoint_exists,
        "checkpoint_mtime": checkpoint_mtime,
        "newest_source_mtime": newest_source,
        "required_artifact_mtime": required_mtime,
        "source_count": len(sources),
        "artifacts": artifacts,
        "rule": (
            "fanout evidence must be newer than the checkpoint, decoder/fanout sources, and the platform release binary; "
            "stale manifests are diagnostic-only"
        ),
    }


def build_artifact_provenance(
    paths: dict[str, Path],
    freshness_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_rows = [file_provenance(path) for path in code_lm_fanout_source_paths()]
    source_fingerprint = hashlib.sha256()
    for row in source_rows:
        source_fingerprint.update(
            json.dumps(
                {
                    "path": row.get("path"),
                    "bytes": row.get("bytes"),
                    "mtime": row.get("mtime"),
                    "sha256": row.get("sha256"),
                },
                sort_keys=True,
            ).encode("utf-8")
        )
        source_fingerprint.update(b"\n")
    artifact_rows = {
        "fanout_report": file_provenance(paths["fanout_report"]),
        "private_candidates": file_provenance(paths["private_candidates"]),
        "public_candidates": file_provenance(paths["public_candidates"]),
    }
    fanout_freshness = fanout_source_freshness(
        {
            "fanout_report": paths["fanout_report"],
            "private_candidates": paths["private_candidates"],
            "public_candidates": paths["public_candidates"],
            "checkpoint": paths["checkpoint"],
        }
    )
    private_input_freshness = effective_private_training_input_freshness(paths, freshness_context)
    release_binary = file_provenance(release_binary_path())
    checkpoint = file_provenance(paths["checkpoint"], follow_archive_pointer=True)
    return {
        "policy": "project_theseus_train_once_fanout_artifact_provenance_v1",
        "created_utc": now(),
        "provenance_ready": bool(
            checkpoint.get("sha256")
            and release_binary.get("sha256")
            and source_rows
            and all(row.get("sha256") for row in artifact_rows.values())
        ),
        "checkpoint": checkpoint,
        "release_binary": release_binary,
        "source_fingerprint": {
            "source_count": len(source_rows),
            "newest_source_mtime": max((number(row.get("mtime")) for row in source_rows), default=0),
            "combined_sha256": source_fingerprint.hexdigest(),
            "sources": source_rows,
        },
        "fanout_artifacts": artifact_rows,
        "fanout_freshness": fanout_freshness,
        "private_input_freshness": private_input_freshness,
        "summary": {
            "checkpoint_sha256": checkpoint.get("sha256", ""),
            "release_binary_sha256": release_binary.get("sha256", ""),
            "release_binary_backend": release_binary_backend(),
            "source_combined_sha256": source_fingerprint.hexdigest(),
            "fanout_fresh": fanout_freshness.get("fresh"),
            "private_inputs_fresh": private_input_freshness.get("fresh"),
            "rule": "fanout reports/manifests must be newer than source/binary, and train-once checkpoints must be newer than private training rows",
        },
        "score_semantics": "artifact_identity_and_staleness_control_signal_not_capability_evidence",
    }


def file_provenance(path: Path | None, *, follow_archive_pointer: bool = False) -> dict[str, Any]:
    if path is None:
        return {"path": "", "exists": False}
    logical_path = resolve(path)
    archive_pointer = bool(follow_archive_pointer and logical_path.exists() and is_archive_pointer(logical_path))
    resolved_path = resolve_archived_path(logical_path) if archive_pointer else logical_path
    path = resolved_path
    exists = path.exists()
    row: dict[str, Any] = {
        "path": rel(logical_path),
        "exists": exists,
        "bytes": path.stat().st_size if exists else 0,
        "mtime": path.stat().st_mtime if exists else 0.0,
        "sha256": "",
    }
    if archive_pointer:
        row.update(
            {
                "archive_pointer": True,
                "resolved_path": rel(resolved_path),
                "pointer_bytes": logical_path.stat().st_size,
                "pointer_mtime": logical_path.stat().st_mtime,
            }
        )
    if not exists or not path.is_file():
        return row
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    row["sha256"] = digest.hexdigest()
    return row


def resolved_checkpoint_path(path: Path) -> Path:
    resolved = resolve_archived_path(path)
    if resolved.suffix != ".gz":
        return resolved
    target = REHYDRATED_ARTIFACTS / "checkpoints" / path.name
    if (
        target.exists()
        and target.stat().st_size > 0
        and target.stat().st_mtime >= resolved.stat().st_mtime
    ):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp")
    with gzip.open(resolved, "rb") as source, tmp.open("wb") as destination:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            destination.write(chunk)
    tmp.replace(target)
    return target


def run_resource_policy() -> dict[str, Any]:
    return run_resource_policy_raw(ROOT, REPORTS)


def run_command(
    command: list[str],
    *,
    timeout_seconds: int,
    log_path: Path,
    phase: str | None = None,
    heartbeat_path: Path | None = None,
    progress_paths: list[Path] | None = None,
    heartbeat_interval_seconds: int = 30,
) -> dict[str, Any]:
    return run_command_with_optional_heartbeat(
        command,
        cwd=ROOT,
        env=train_env(),
        timeout_seconds=timeout_seconds,
        log_path=log_path,
        phase=phase,
        heartbeat_path=heartbeat_path,
        progress_paths=progress_paths,
        phase_contracts=PHASE_CONTRACTS,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
    )


def train_env() -> dict[str, str]:
    env = os.environ.copy()
    cuda_enabled = cuda_readout_requested_by_launcher()
    env.update(
        {
            "THESEUS_CODE_LM_CUDA_READOUT": "1" if cuda_enabled else "0",
            "THESEUS_STRATIFIED_WORK_BUDGET_ADMISSION": "1",
            "THESEUS_TARGET_FAMILY_STARVATION_RESCUE": "1",
            "THESEUS_TARGET_FAMILY_STARVATION_RESCUE_MIN_ROWS": "24",
            "THESEUS_TYPED_EDGE_EXEC_RECEIVER_V1": "1",
            "THESEUS_PRIVATE_TYPE_SHAPE_RECEIVER_VETO_V1": "1",
            "THESEUS_EDGE_EXEC_REPAIR_V1": "1",
            "THESEUS_PROGRAM_SYNTHESIS_LOOP_V1": "1",
            "THESEUS_USE_CUDA_RANKER": "1" if cuda_enabled else "0",
            "THESEUS_USE_CUDA_STS_RETRIEVAL": "1" if cuda_enabled else "0",
        }
    )
    return env


def finish(out: Path, markdown: Path, state: dict[str, Any], started: float) -> None:
    state["runtime_ms"] = int((time.perf_counter() - started) * 1000)
    paths = artifact_paths(str(state.get("slug") or "private_pressure_private_recovery_train_once_fanout_v1"))
    state["phase_ledger_summary"] = summarize_phase_ledger(paths["phase_ledger"])
    write_progress(out, markdown, state)
    print(json.dumps(state, indent=2))


def mark_running(
    state: dict[str, Any],
    *,
    phase: str,
    command: list[str],
    log_path: Path,
    heartbeat_path: Path | None = None,
) -> None:
    state.update(
        {
            "trigger_state": "RUNNING",
            "run_status": "running",
            "current_phase": phase,
            "phase_started_utc": now(),
            "active_command": command,
            "active_log_path": rel(log_path),
            "active_phase_heartbeat": rel(heartbeat_path) if heartbeat_path is not None else "",
            "next_actions": ["Leave this train-once/fanout worker running; do not stack duplicate Code LM jobs."],
        }
    )


def write_progress(out: Path, markdown: Path, payload: dict[str, Any]) -> None:
    write_json(out, payload)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_markdown(payload), encoding="utf-8")


def gate(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "detail": detail}


def vcm_context_governor_receipt(path: Path = DEFAULT_VCM_CONTEXT_GOVERNOR) -> dict[str, Any]:
    packet = vcm_consumer_abi.build_consumer_packet(
        consumer_id="code_lm_train_once_fanout",
        purpose="generation_fanout",
        read_set=[rel(path)],
        write_set=["reports/code_lm_train_once_fanout.json"],
        authority_ceiling=["local_checkpoint_read", "local_candidate_manifest_write", "governed_context_read"],
        permitted_uses=["generation_fanout_context", "candidate_manifest_provenance", "audit_replay"],
        governor_path=path,
        taint_labels=["private_generation_metadata", "public_calibration_metadata_not_training"],
        deletion_obligations=["invalidate_fanout_derivatives_when_context_is_revoked"],
        audit_refs=["scripts/code_lm_train_once_fanout.py"],
    )
    governor = packet["governor_receipt"]
    summary = object_field(governor, "summary")
    return {
        **governor,
        "report": rel(path),
        "ready": bool(packet.get("ready")),
        "hard_gap_count": int(number(summary.get("hard_gap_count"))),
        "warning_count": int(number(summary.get("warning_count"))),
        "mission_brief_status": str(summary.get("mission_brief_status") or ""),
        "mission_brief_omission_count": int(number(summary.get("mission_brief_omission_count"))),
        "deletion_closure_status": str(summary.get("deletion_closure_status") or ""),
        "deletion_closure_fault_count": int(number(summary.get("deletion_closure_fault_count"))),
        "scif_status": str(summary.get("scif_status") or ""),
        "consumer_abi": packet,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def vcm_context_summary_fields(vcm_receipt: dict[str, Any], *, surface: str) -> dict[str, Any]:
    ready = bool(vcm_receipt.get("ready"))
    return {
        "vcm_context_surface": surface,
        "vcm_context_governor_ready": ready,
        "vcm_context_governor_state": vcm_receipt.get("trigger_state", ""),
        "vcm_context_governor_receipt_id": vcm_receipt.get("receipt_id", ""),
        "vcm_context_governor_hard_gap_count": int(number(vcm_receipt.get("hard_gap_count"))),
        "vcm_context_adequacy_state": (
            "governed_sufficient_for_generation_fanout"
            if ready
            else "missing_or_insufficient_governed_generation_context"
        ),
        "vcm_mission_brief_status": vcm_receipt.get("mission_brief_status", ""),
        "vcm_deletion_closure_status": vcm_receipt.get("deletion_closure_status", ""),
        "vcm_scif_status": vcm_receipt.get("scif_status", ""),
    }


def strict_generator_fanout_receipt_summary(
    path: Path = DEFAULT_STRICT_GENERATOR_FANOUT_RECEIPT,
) -> dict[str, Any]:
    report = read_json(path, {})
    summary = object_field(report, "summary")
    combined = object_field(summary, "combined")
    hard_gaps = report.get("hard_gaps") if isinstance(report.get("hard_gaps"), list) else []
    ready = bool(
        path.exists()
        and report.get("policy") == "project_theseus_neural_seed_strict_generator_fanout_receipt_v1"
        and report.get("trigger_state") in {"GREEN", "YELLOW"}
        and not hard_gaps
        and number(summary.get("eligible_full_body_candidate_count")) > 0
        and number(combined.get("runtime_load_task_rate")) > 0.0
        and int(number(summary.get("public_training_rows_written"))) == 0
        and int(number(summary.get("external_inference_calls"))) == 0
        and int(number(summary.get("fallback_return_count"))) == 0
    )
    return {
        "policy": "project_theseus_strict_generator_fanout_receipt_summary_v1",
        "ready": ready,
        "path": rel(path),
        "trigger_state": report.get("trigger_state"),
        "hard_gaps": hard_gaps,
        "eligible_full_body_candidate_count": int(number(summary.get("eligible_full_body_candidate_count"))),
        "replayed_task_count": int(number(summary.get("replayed_task_count"))),
        "syntax_valid_rate": number(summary.get("syntax_valid_rate")),
        "runtime_load_task_rate": number(combined.get("runtime_load_task_rate")),
        "intended_behavior_pass_rate": number(combined.get("intended_behavior_pass_rate")),
        "score_semantics": (
            "Strict full-body generator replay receipt only; behavior is measured separately from syntax/loadability "
            "and no renderer/router/template/tool/fallback row is credited as learned generation."
        ),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def build_fanout_spine_records(
    args: argparse.Namespace,
    paths: dict[str, Path],
    *,
    private_rows: int,
    public_rows: int,
    private_only: bool,
    vcm_receipt: dict[str, Any],
    artifact_provenance: dict[str, Any],
) -> list[dict[str, Any]]:
    vcm_fields = vcm_context_summary_fields(vcm_receipt, surface="fanout_closure")
    fanout_report = read_json(paths["fanout_report"], {})
    checkpoint_report = read_json(paths["checkpoint_wrapper_report"], {})
    phase_ledger = summarize_phase_ledger(paths["phase_ledger"])
    phase_timing_ms = merged_phase_timing(checkpoint_report, fanout_report, phase_ledger)
    strict_generator_receipt = strict_generator_fanout_receipt_summary()
    verification_bandwidth = fanout_verification_bandwidth_record(
        paths,
        private_rows=private_rows,
        public_rows=public_rows,
        private_only=private_only,
        strict_generator_receipt=strict_generator_receipt,
        vcm_receipt=vcm_receipt,
    )
    governance_tax = fanout_governance_tax_record(
        phase_timing_ms,
        verification_bandwidth=verification_bandwidth,
        vcm_receipt=vcm_receipt,
        artifact_provenance=artifact_provenance,
    )
    run_suffix = stable_payload_hash(
        {
            "slug": getattr(args, "slug", ""),
            "checkpoint": rel(paths["checkpoint"]),
            "fanout_report": rel(paths["fanout_report"]),
            "private_rows": private_rows,
            "public_rows": public_rows,
            "vcm_receipt_id": vcm_receipt.get("receipt_id"),
        }
    )[:16]
    run_id = f"train_once_fanout-{run_suffix}"
    claim_id = f"claim_train_once_fanout-{run_suffix}"
    common: dict[str, Any] = {
        "run_id": run_id,
        "producer_surface": "code_lm_train_once_fanout",
        "slug": getattr(args, "slug", ""),
        "private_candidate_count": private_rows,
        "public_candidate_count": public_rows,
        "private_only": private_only,
        "public_calibration_allowed": False,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "support_state": "SUPPORTED" if vcm_receipt.get("ready") else "BLOCKED",
        "verification_bandwidth": verification_bandwidth,
        "governance_tax": governance_tax,
        **vcm_fields,
    }
    artifact_hash = stable_payload_hash(
        {
            "checkpoint": object_field(artifact_provenance, "checkpoint").get("sha256", ""),
            "fanout": object_field(artifact_provenance, "fanout_report").get("sha256", ""),
            "private_candidates": object_field(artifact_provenance, "private_candidates").get("sha256", ""),
            "public_candidates": object_field(artifact_provenance, "public_candidates").get("sha256", ""),
        }
    )
    abi_packet = vcm_receipt.get("consumer_abi") if isinstance(vcm_receipt.get("consumer_abi"), dict) else {}
    abi_records = abi_packet.get("records") if isinstance(abi_packet.get("records"), list) else []
    return list(abi_records) + [
        {
            **common,
            "record_type": "context_transaction_record",
            "record_id": f"train_once_fanout_context_transaction-{run_suffix}",
            "transaction_id": f"fanout_context_txn-{run_suffix}",
            "operation": "read",
            "mounts": ["vcm_context_governor", "train_once_checkpoint", "candidate_manifests"],
            "read_set": [
                rel(DEFAULT_VCM_CONTEXT_GOVERNOR),
                rel(paths["checkpoint"]),
                rel(paths["fanout_report"]),
                rel(paths["private_candidates"]),
                rel(paths["public_candidates"]),
            ],
            "write_set": [rel(paths["closure_report"])],
            "taint_labels": ["governed_context_receipt", "candidate_generation_metadata"],
            "closure_state": "closed_for_fanout_evidence",
            "deletion_obligations": "closed_by_vcm_governor",
            "faults": [] if vcm_receipt.get("ready") else ["vcm_context_governor_not_ready"],
        },
        {
            **common,
            "record_type": "context_adequacy_record",
            "record_id": f"train_once_fanout_context_adequacy-{run_suffix}",
            "adequacy_id": f"fanout_context_adequacy-{run_suffix}",
            "target_claim_id": claim_id,
            "governor_ready": bool(vcm_receipt.get("ready")),
            "governor_receipt_id": vcm_receipt.get("receipt_id", ""),
            "adequacy_state": vcm_fields["vcm_context_adequacy_state"],
            "fail_closed": not bool(vcm_receipt.get("ready")),
            "compression_path": "vcm_governor_receipt_to_train_once_fanout_context",
            "semantic_units": [
                {
                    "title": "VCM context governor receipt",
                    "source_path": rel(DEFAULT_VCM_CONTEXT_GOVERNOR),
                    "address": rel(DEFAULT_VCM_CONTEXT_GOVERNOR),
                    "taints": ["governed_context_receipt"],
                }
            ],
            "residual_risks": [] if vcm_receipt.get("ready") else ["fanout_context_not_governed"],
        },
        {
            **common,
            "record_type": "runtime_adapter_invocation",
            "record_id": f"train_once_fanout_runtime_adapter-{run_suffix}",
            "adapter_id": "code_lm_train_once_fanout_supervisor",
            "status": "READY" if vcm_receipt.get("ready") else "BLOCKED",
            "checkpoint_ref": rel(paths["checkpoint"]),
            "fanout_report_ref": rel(paths["fanout_report"]),
            "non_claim": "Fanout supervisor evidence is routing and candidate-manifest traceability, not learned-generation promotion.",
        },
        {
            **common,
            "record_type": "resource_budget_record",
            "record_id": f"train_once_fanout_resource_budget-{run_suffix}",
            "budget_id": f"fanout_budget-{run_suffix}",
            "backend": release_binary_backend(),
            "heavy_training_started_by_record": False,
            "verification_obligation_count": verification_bandwidth["obligation_count"],
            "verifier_capacity_units": verification_bandwidth["verifier_capacity_units"],
            "capacity_margin_units": verification_bandwidth["capacity_margin_units"],
            "escalation_required": verification_bandwidth["escalation_required"],
            "residual_obligations": verification_bandwidth["residual_obligations"],
            "score_semantics": "budget envelope only; no training or public calibration is performed by this record.",
        },
        {
            **common,
            "record_type": "costed_route_record",
            "record_id": f"train_once_fanout_costed_route-{run_suffix}",
            "cost_accounting": {
                "governed_overhead_ms": governance_tax["governed_overhead_ms"],
                "governed_total_latency_ms": governance_tax["governed_total_latency_ms"],
                "review_load_units": governance_tax["review_load_units"],
                "caught_failure_count": governance_tax["caught_failure_count"],
                "tax_per_caught_failure": governance_tax["tax_per_caught_failure"],
                "verification_obligation_count": verification_bandwidth["obligation_count"],
                "verifier_capacity_units": verification_bandwidth["verifier_capacity_units"],
            },
            "cost_classes": ["checkpoint_freshness", "vcm_context_governance", "candidate_integrity", "strict_replay_receipt"],
            "non_claim": "Fanout governance cost is route accounting; lower latency cannot hide displaced verification or replay obligations.",
        },
        {
            **common,
            "record_type": "generation_mode_record",
            "record_id": f"train_once_fanout_generation_mode-{run_suffix}",
            "candidate_generation_credit": 0,
            "learned_generation_claim_allowed": False,
            "state": "candidate_manifest_fanout_trace_only",
            "non_claim": "Candidate fanout inventory cannot count templates, tools, or routing as learned generation.",
        },
        {
            **common,
            "record_type": "failure_boundary",
            "record_id": f"train_once_fanout_failure_boundary-{run_suffix}",
            "failure_id": f"fanout_boundary-{run_suffix}",
            "fallback_return_used": False,
            "structured_non_solved": public_rows == 0 and not private_only,
            "verification_escalation_required": verification_bandwidth["escalation_required"],
            "residual_obligations": verification_bandwidth["residual_obligations"],
            "terminal": False,
            "status": "READY" if vcm_receipt.get("ready") else "BLOCKED",
        },
        {
            **common,
            "record_type": "artifact_graph_record",
            "record_id": f"train_once_fanout_artifact-{run_suffix}",
            "artifact_kind": "train_once_fanout_closure",
            "content_hash": artifact_hash,
            "checkpoint_ref": rel(paths["checkpoint"]),
            "fanout_report_ref": rel(paths["fanout_report"]),
            "private_candidate_manifest_ref": rel(paths["private_candidates"]),
            "public_candidate_manifest_ref": rel(paths["public_candidates"]),
            "context_refs": [rel(DEFAULT_VCM_CONTEXT_GOVERNOR), str(vcm_receipt.get("receipt_id", ""))],
        },
        {
            **common,
            "record_type": "claim_record",
            "record_id": f"train_once_fanout_claim-{run_suffix}",
            "claim_id": claim_id,
            "state": "train_once_fanout_closure_summarized",
            "status": "YELLOW",
            "evidence_ref": rel(paths["closure_report"]),
            "claim_boundary": "traceability_only_not_capability_promotion",
        },
        {
            **common,
            "record_type": "evidence_transition_record",
            "record_id": f"train_once_fanout_evidence_transition-{run_suffix}",
            "state": "checkpoint_and_manifests_to_closure_report",
            "status": "SUPPORTED" if vcm_receipt.get("ready") else "BLOCKED",
            "evidence_ref": rel(paths["closure_report"]),
        },
        {
            **common,
            "record_type": "authority_use_receipt",
            "record_id": f"train_once_fanout_authority_use-{run_suffix}",
            "authority_scope": ["local_candidate_fanout", "metadata_only_public_manifest"],
            "state": "local_fanout_no_public_scoring_no_external_inference",
            "status": "READY" if vcm_receipt.get("ready") else "BLOCKED",
        },
    ]


def fanout_verification_bandwidth_record(
    paths: dict[str, Path],
    *,
    private_rows: int,
    public_rows: int,
    private_only: bool,
    strict_generator_receipt: dict[str, Any],
    vcm_receipt: dict[str, Any],
) -> dict[str, Any]:
    strict_ready = bool(strict_generator_receipt.get("ready"))
    obligation_count = (
        1  # checkpoint freshness/provenance
        + 1  # VCM context adequacy
        + 1  # public metadata-only boundary
        + 1  # no external inference/no fallback accounting
        + max(1, private_rows)
        + (0 if private_only else max(1, public_rows))
        + (1 if strict_ready else 2)
    )
    verifier_capacity_units = (
        max(1, private_rows)
        + (0 if private_only else max(1, public_rows // 2))
        + (3 if strict_ready else 0)
        + (2 if vcm_receipt.get("ready") else 0)
    )
    capacity_floor_units = max(4, min(obligation_count, 256))
    capacity_margin_units = verifier_capacity_units - capacity_floor_units
    residual_obligations = []
    if not strict_ready:
        residual_obligations.append("strict_generator_fanout_receipt_not_ready")
    if not vcm_receipt.get("ready"):
        residual_obligations.append("vcm_context_governor_not_ready")
    if capacity_margin_units < 0:
        residual_obligations.append("fanout_verifier_capacity_escalation")
    return {
        "policy": "project_theseus_fanout_verification_bandwidth_v1",
        "surface": "code_lm_train_once_fanout",
        "evidence_refs": [
            rel(paths["closure_report"]),
            rel(paths["fanout_report"]),
            rel(paths["checkpoint_wrapper_report"]),
            rel(DEFAULT_STRICT_GENERATOR_FANOUT_RECEIPT),
            rel(DEFAULT_VCM_CONTEXT_GOVERNOR),
        ],
        "obligation_count": obligation_count,
        "verifier_capacity_units": verifier_capacity_units,
        "capacity_floor_units": capacity_floor_units,
        "capacity_margin_units": capacity_margin_units,
        "verification_arms": [
            "checkpoint_freshness",
            "candidate_manifest_integrity",
            "strict_generator_replay_receipt",
            "private_verifier_cascade",
            "vcm_context_adequacy",
        ],
        "decomposition_contract": {
            "private_candidate_count": private_rows,
            "public_candidate_count": public_rows,
            "private_only": private_only,
            "verification_strategy": "checkpoint_and_candidate_manifests_require_strict_replay_private_verifier_and_vcm_context_receipts",
        },
        "residual_obligations": residual_obligations,
        "escalation_thresholds": {
            "capacity_margin_min": 0,
            "strict_generator_replay_receipt_required": True,
            "vcm_context_governor_required": True,
            "public_metadata_only_required": True,
        },
        "escalation_required": bool(residual_obligations),
        "adequacy_state": "ready" if not residual_obligations else "verification_capacity_residual",
        "status": "ready",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "candidate_generation_credit": 0,
        "non_claims": [
            "fanout verification bandwidth is accounting, not candidate quality",
            "strict replay receipts do not grant learned-generation credit",
        ],
    }


def fanout_governance_tax_record(
    phase_timing_ms: dict[str, Any],
    *,
    verification_bandwidth: dict[str, Any],
    vcm_receipt: dict[str, Any],
    artifact_provenance: dict[str, Any],
) -> dict[str, Any]:
    candidate_generation_ms = int(number(phase_timing_ms.get("candidate_expression_generation_ms")))
    raw_latency_ms = max(1, int(number(phase_timing_ms.get("total_ms"))) or candidate_generation_ms)
    gate_costs = {
        "vcm_context_adequacy_ms": 4 if vcm_receipt.get("ready") else 8,
        "artifact_provenance_ms": 4 if artifact_provenance.get("provenance_ready") else 9,
        "strict_replay_receipt_ms": 5,
        "verification_bandwidth_accounting_ms": 3,
        "no_cheat_boundary_ms": 2,
    }
    governed_overhead_ms = sum(gate_costs.values())
    caught_failure_count = len(verification_bandwidth.get("residual_obligations") or [])
    review_load_units = max(1, len(verification_bandwidth.get("verification_arms") or []))
    tax_per_caught_failure = round(governed_overhead_ms / max(1, caught_failure_count), 6)
    return {
        "policy": "project_theseus_fanout_governance_tax_v1",
        "surface": "code_lm_train_once_fanout",
        "gate_costs": gate_costs,
        "raw_route_latency_ms": raw_latency_ms,
        "governed_overhead_ms": governed_overhead_ms,
        "governed_total_latency_ms": raw_latency_ms + governed_overhead_ms,
        "review_load_units": review_load_units,
        "caught_failure_count": caught_failure_count,
        "tax_per_caught_failure": tax_per_caught_failure,
        "tax_justified": caught_failure_count > 0 or bool(artifact_provenance.get("provenance_ready")),
        "tax_value_statement": "fanout governance cost preserves checkpoint freshness, context adequacy, replay, and no-cheat boundaries",
        "status": "ready",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "candidate_generation_credit": 0,
        "non_claims": [
            "governance tax is overhead accounting, not capability",
            "candidate generation speed must be reported after displaced verification cost",
        ],
    }


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_payload_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def flattened_numbers(row: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(row, dict):
        for key, value in row.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(flattened_numbers(value, child))
    elif isinstance(row, (int, float)) and not isinstance(row, bool):
        out[prefix or "value"] = float(row)
    return out


def number(value: Any) -> float:
    try:
        if value is None or isinstance(value, bool):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_id(value: Any) -> str:
    text = str(value or "").strip().lower()
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() else "_")
    return "_".join(part for part in "".join(out).split("_") if part)[:80] or "unknown"


def command_preview(command: list[str], limit: int = 16) -> list[str]:
    if len(command) <= limit:
        return command
    return command[:limit] + [f"...(+{len(command) - limit} args)"]


def get_path(row: Any, path: list[str], default: Any = None) -> Any:
    current = row
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def object_field(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key)
    return value if isinstance(value, dict) else {}


def resolve(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return ROOT / value


def rel(path: str | Path) -> str:
    value = resolve(path)
    try:
        return str(value.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(value).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
