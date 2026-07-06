"""Code LM Closure v1: private training -> private eval -> public calibration.

This lane is deliberately strict:
- private generated code tasks may include solutions and hidden tests;
- public MBPP/HumanEval/EvalPlus tasks are exported as visible prompts only;
- the Rust/SymLiquid command trains a next-token code readout on private train
  rows only;
- public benchmark tests are used only by the existing calibration harness.
"""

from __future__ import annotations

import argparse
import ast
import atexit
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import real_code_benchmark_graduation as real_code  # noqa: E402
from real_code_benchmark_support import runtime_tmp_dir as benchmark_runtime_tmp_dir  # noqa: E402
from code_lm_private_verifier import (  # noqa: E402
    concept_family,
    concept_residual_label,
    evaluate_private_candidates,
)
from code_lm_private_curriculum import (  # noqa: E402
    build_private_curriculum,
    load_extra_private_train,
    load_extra_private_train_many,
    split_path_list,
)
from code_lm_decoder_contracts import (  # noqa: E402
    attach_decoder_contracts,
    public_decoder_contract_preflight,
)
from code_lm_process_guard import duplicate_artifact_processes  # noqa: E402
from code_lm_public_task_export import (  # noqa: E402
    export_public_visible_tasks,
    load_symliquid_state,
    prioritize_private_rows_for_public_categories,
)
from code_lm_rust_launch import (  # noqa: E402
    build_step_plan,
    private_type_shape_receiver_veto_enabled,
    rust_closure_command,
    symliquid_process_env,
    timeout_arg,
    typed_edge_exec_receiver_enabled,
)
from code_lm_run_lock import acquire_run_lock, release_run_lock  # noqa: E402
from code_lm_source_fingerprint import (  # noqa: E402
    decoder_relevant_source_fingerprint,
    decoder_relevant_source_mtime,
    decoder_source_paths,
)
from code_lm_sts_conditioning import run_sts_conditioning  # noqa: E402
from code_lm_private_rows import (  # noqa: E402
    DEFAULT_EXTRA_PRIVATE_TRAIN_JSONL,
    DEFAULT_REPO_REPAIR_PRIVATE_TRAIN_JSONL,
    DEFAULT_RESIDUAL_PRIVATE_TRAIN_JSONL,
    default_no_admissible_repair_policy_jsonl,
    high_transfer_private_rows_string,
)
from process_tree import run_process_tree  # noqa: E402


DEFAULT_PUBLIC_CARDS = "source_human_eval"
DEFAULT_NO_ADMISSIBLE_REPAIR_POLICY_JSONL = default_no_admissible_repair_policy_jsonl()
DEFAULT_HIGH_TRANSFER_PRIVATE_TRAIN_JSONL = high_transfer_private_rows_string(include_broad_floor_recovery=True)
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--private-count", type=int, default=1200)
    parser.add_argument("--public-cards", default=DEFAULT_PUBLIC_CARDS)
    parser.add_argument("--max-public-cases-per-card", type=int, default=8)
    parser.add_argument(
        "--case-manifest",
        default="",
        help="Optional public calibration selector manifest. Contains task IDs only.",
    )
    parser.add_argument("--hv-dim", type=int, default=768)
    parser.add_argument("--max-vocab", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--candidates-per-task", type=int, default=12)
    parser.add_argument("--extra-private-train-jsonl", default=DEFAULT_EXTRA_PRIVATE_TRAIN_JSONL)
    parser.add_argument("--residual-private-train-jsonl", default=DEFAULT_RESIDUAL_PRIVATE_TRAIN_JSONL)
    parser.add_argument("--repo-repair-private-train-jsonl", default=DEFAULT_REPO_REPAIR_PRIVATE_TRAIN_JSONL)
    parser.add_argument(
        "--high-transfer-private-train-jsonl",
        default=DEFAULT_HIGH_TRANSFER_PRIVATE_TRAIN_JSONL,
        help="Semicolon/comma-separated private high-transfer JSONL files. Public benchmark data must remain calibration-only.",
    )
    parser.add_argument("--max-extra-private-train", type=int, default=2000)
    parser.add_argument("--max-residual-private-train", type=int, default=1200)
    parser.add_argument("--max-repo-repair-private-train", type=int, default=1200)
    parser.add_argument("--max-high-transfer-private-train", type=int, default=4800)
    parser.add_argument("--disable-extra-private-train", action="store_true")
    parser.add_argument("--disable-residual-private-train", action="store_true")
    parser.add_argument("--disable-repo-repair-private-train", action="store_true")
    parser.add_argument("--disable-high-transfer-private-train", action="store_true")
    parser.add_argument("--private-curriculum-out", default="data/private_code_curriculum/code_lm_closure_seed14.jsonl")
    parser.add_argument("--public-task-manifest-out", default="reports/code_lm_public_tasks.jsonl")
    parser.add_argument("--checkpoint-out", default="reports/student_code_lm_checkpoint.json")
    parser.add_argument(
        "--checkpoint-in",
        default="",
        help="Optional train-once checkpoint for generate-only/fanout paths. Empty means train a fresh checkpoint.",
    )
    parser.add_argument("--private-candidate-out", default="reports/code_lm_private_candidates.jsonl")
    parser.add_argument("--public-candidate-out", default="reports/student_code_candidates.jsonl")
    parser.add_argument("--rust-report-out", default="reports/code_lm_closure_rust.json")
    parser.add_argument("--public-report-out", default="reports/real_code_benchmark_graduation.json")
    parser.add_argument("--public-trace-out", default="reports/real_code_benchmark_traces.jsonl")
    parser.add_argument("--public-transfer-artifact-out", default="reports/transfer_artifacts/code/real_code_benchmark_graduation_transfer_artifact.json")
    parser.add_argument("--out", default="reports/code_lm_closure.json")
    parser.add_argument("--skip-public-calibration", action="store_true")
    parser.add_argument(
        "--private-only",
        action="store_true",
        help=(
            "Use an intentionally empty public task/candidate sidecar. This is private repair evidence only "
            "and cannot unlock public calibration."
        ),
    )
    parser.add_argument(
        "--rust-timeout-seconds",
        type=int,
        default=7200,
        help="Wall-clock safety fuse for the Rust learner. Use 0 to disable; run duration is primarily controlled by step-budget knobs.",
    )
    parser.add_argument(
        "--max-rust-work-steps",
        type=int,
        default=0,
        help="Optional hard cap for Rust learner work steps. 0 means derive the full step plan from epochs/private rows/public rows/candidates.",
    )
    parser.add_argument(
        "--use-cuda-readout",
        action="store_true",
        default=os.environ.get("THESEUS_CODE_LM_CUDA_READOUT", "0").strip()
        in {"1", "true", "TRUE", "yes", "YES"},
        help="Require the Rust Code LM next-token readout to use the CUDA fast sparse readout path.",
    )
    parser.add_argument(
        "--checkpoint-only",
        action="store_true",
        help="Train one reusable Code LM checkpoint with model_artifacts_v1, then stop before candidate generation. Use a fanout job for candidates.",
    )
    parser.add_argument("--rust-build-profile", choices=["release", "debug"], default="release")
    parser.add_argument(
        "--public-timeout-seconds",
        type=int,
        default=1800,
        help="Wall-clock safety fuse for public calibration. Use 0 to disable; case count and candidate count define the work step duration.",
    )
    parser.add_argument("--disable-sts-conditioning", action="store_true")
    parser.add_argument("--symliquid-state-engine-report", default="reports/symliquid_state_engine.json")
    parser.add_argument("--sts-training-data", default="data/sts_learning/sts_code_streams_seed14.jsonl")
    parser.add_argument(
        "--no-admissible-repair-policy-jsonl",
        default=DEFAULT_NO_ADMISSIBLE_REPAIR_POLICY_JSONL,
        help="Metadata-only decoder-control rows produced by the private ablation gate. These are STS/control pressure, not code-answer training rows.",
    )
    parser.add_argument(
        "--sts-decoder-control-policy-jsonl",
        default="reports/sts_decoder_control_rows.jsonl",
        help="Metadata-only STS decoder-control rows produced by sts_causal_decoder_ablation.py. These are control pressure, not code-answer training rows.",
    )
    parser.add_argument("--max-no-admissible-control-rows", type=int, default=640)
    parser.add_argument("--max-sts-decoder-control-rows", type=int, default=128)
    parser.add_argument("--sts-conditioning-input-out", default="reports/code_lm_sts_conditioning_input.jsonl")
    parser.add_argument("--sts-generation-out", default="reports/code_lm_sts_public_generations.jsonl")
    parser.add_argument("--sts-conditioning-checkpoint-out", default="reports/code_lm_sts_conditioning_checkpoint.json")
    parser.add_argument("--sts-conditioning-report-out", default="reports/code_lm_sts_conditioning_report.json")
    parser.add_argument("--sts-conditioning-hv-dim", type=int, default=512)
    parser.add_argument("--sts-conditioning-max-vocab", type=int, default=640)
    parser.add_argument("--sts-conditioning-epochs", type=int, default=2)
    parser.add_argument("--sts-conditioning-lr", type=float, default=0.06)
    parser.add_argument("--sts-conditioning-max-generate-steps", type=int, default=48)
    parser.add_argument(
        "--sts-conditioning-max-train-rows",
        type=int,
        default=960,
        help="Optional cap for CPU STS conditioning train rows. 0 keeps the full derived budget.",
    )
    parser.add_argument(
        "--sts-conditioning-max-eval-rows",
        type=int,
        default=96,
        help="Optional cap for CPU STS conditioning eval rows. 0 keeps the full derived budget.",
    )
    parser.add_argument(
        "--sts-conditioning-max-generate-rows",
        type=int,
        default=96,
        help="Optional cap for CPU STS conditioning generation rows. 0 keeps the full derived budget.",
    )
    parser.add_argument(
        "--sts-timeout-seconds",
        type=int,
        default=3600,
        help="Wall-clock safety fuse for STS conditioning. Use 0 to disable; row/epoch/generation counts define the work step duration.",
    )
    parser.add_argument("--lock-path", default="reports/code_lm_closure.lock")
    parser.add_argument(
        "--edge-obligation-decode-gate-v1",
        action="store_true",
        help="Run the private-only edge-obligation verifier over generated private candidates before treating the closure as calibration-ready.",
    )
    parser.add_argument("--edge-obligation-report-out", default="reports/edge_obligation_decode_gate_v1_private.json")
    parser.add_argument("--edge-obligation-markdown-out", default="reports/edge_obligation_decode_gate_v1_private.md")
    parser.add_argument(
        "--typed-edge-exec-receiver-v1",
        action="store_true",
        help="Enable typed-edge executable receiver/reranker scoring inside the Rust decoder.",
    )
    parser.add_argument(
        "--private-type-shape-receiver-veto-v1",
        action="store_true",
        help="Enable the teacher-gated type/return-shape receiver bias after private ablation passes.",
    )
    parser.add_argument(
        "--disable-rust-resume",
        action="store_true",
        help="Always rerun the Rust/SymLiquid stage even when completed fresh artifacts already exist.",
    )
    parser.add_argument(
        "--private-eval-shard-index",
        type=int,
        default=0,
        help="Deterministic private eval shard index for chunked/resumable candidate generation.",
    )
    parser.add_argument(
        "--private-eval-shard-count",
        type=int,
        default=1,
        help="Deterministic private eval shard count. Train rows are preserved; only eval rows are sharded.",
    )
    parser.add_argument(
        "--public-task-shard-index",
        type=int,
        default=0,
        help="Deterministic public visible-task shard index for chunked/resumable candidate generation.",
    )
    parser.add_argument(
        "--public-task-shard-count",
        type=int,
        default=1,
        help="Deterministic public visible-task shard count. Public data remains calibration metadata only.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Generate private/public manifests and run decoder-contract preflight without STS conditioning, Rust training, or public calibration.",
    )
    parser.add_argument("--allow-concurrent", action="store_true")
    args = parser.parse_args()

    lock_fd = None
    if not args.allow_concurrent:
        duplicate_processes = duplicate_artifact_processes(args, ROOT)
        if duplicate_processes:
            skipped = {
                "policy": "project_theseus_code_lm_closure_v1",
                "created_utc": now(),
                "trigger_state": "YELLOW",
                "reason": "duplicate_code_lm_artifact_target_already_running",
                "duplicate_processes": duplicate_processes[:4],
                "public_cards": args.public_cards,
                "seed": args.seed,
                "out": rel(resolve(args.out)),
                "public_candidate_out": rel(resolve(args.public_candidate_out)),
                "external_inference_calls": 0,
            }
            write_json(resolve(args.out), skipped)
            print(json.dumps(skipped, indent=2))
            return 76
        lock_fd = acquire_run_lock(args, ROOT)
        if lock_fd is None:
            skipped = {
                "policy": "project_theseus_code_lm_closure_v1",
                "created_utc": now(),
                "trigger_state": "YELLOW",
                "reason": "code_lm_closure_already_running",
                "lock_path": rel(resolve(args.lock_path)),
                "public_cards": args.public_cards,
                "seed": args.seed,
                "external_inference_calls": 0,
            }
            write_json(resolve(args.out), skipped)
            print(json.dumps(skipped, indent=2))
            return 75
        atexit.register(release_run_lock, lock_fd, resolve(args.lock_path))

    started = time.perf_counter()
    private_rows = build_private_curriculum(seed=args.seed, count=max(20, args.private_count))
    extra_private_rows = []
    if not args.disable_extra_private_train:
        extra_private_rows = load_extra_private_train(
            resolve(args.extra_private_train_jsonl),
            max_rows=max(0, args.max_extra_private_train),
        )
        private_rows.extend(extra_private_rows)
    residual_private_rows = []
    if not args.disable_residual_private_train:
        residual_private_rows = load_extra_private_train(
            resolve(args.residual_private_train_jsonl),
            max_rows=max(0, args.max_residual_private_train),
        )
        private_rows.extend(residual_private_rows)
    repo_repair_private_rows = []
    if not args.disable_repo_repair_private_train:
        repo_repair_private_rows = load_extra_private_train(
            resolve(args.repo_repair_private_train_jsonl),
            max_rows=max(0, args.max_repo_repair_private_train),
        )
        private_rows.extend(repo_repair_private_rows)
    high_transfer_private_rows = []
    if not args.disable_high_transfer_private_train:
        high_transfer_private_rows = load_extra_private_train_many(
            args.high_transfer_private_train_jsonl,
            max_rows=max(0, args.max_high_transfer_private_train),
        )
        private_rows.extend(high_transfer_private_rows)
    high_transfer_private_file_counts = Counter(
        str(row.get("high_transfer_source_jsonl") or "unknown")
        for row in high_transfer_private_rows
    )
    requested_public_cards = [card.strip() for card in args.public_cards.split(",") if card.strip()]
    public_cards = [] if args.private_only else real_code.expand_requested_cards(requested_public_cards)
    public_tasks = (
        []
        if args.private_only
        else export_public_visible_tasks(
            public_cards,
            seed=args.seed,
            max_cases=max(1, args.max_public_cases_per_card),
            case_manifest=args.case_manifest,
        )
    )
    symliquid_state = load_symliquid_state(args)
    private_rows = attach_decoder_contracts(private_rows)
    public_tasks = attach_decoder_contracts(public_tasks)
    private_rows, public_category_priority = prioritize_private_rows_for_public_categories(
        private_rows,
        public_tasks,
        symliquid_state=symliquid_state,
    )
    shard_summary = build_candidate_shard_summary(private_rows, public_tasks, args)
    private_rows = apply_private_eval_shard(private_rows, args)
    public_tasks = apply_public_task_shard(public_tasks, args)
    write_jsonl(resolve(args.private_curriculum_out), private_rows)
    write_jsonl(resolve(args.public_task_manifest_out), public_tasks)
    private_curriculum_stats = jsonl_file_stats(resolve(args.private_curriculum_out))
    public_manifest_stats = jsonl_file_stats(resolve(args.public_task_manifest_out))
    public_contract_preflight = (
        {
            "passed": True,
            "public_task_count": 0,
            "private_only": True,
            "rule": "private-only closure intentionally uses an empty public task manifest",
        }
        if args.private_only
        else public_decoder_contract_preflight(public_tasks)
    )
    public_manifest_serialized = public_manifest_serialized_ok(public_manifest_stats, public_tasks, private_only=args.private_only)
    if not public_contract_preflight["passed"]:
        gates = [
            gate("private_curriculum_generated", len(private_rows) >= 20, f"rows={len(private_rows)}"),
            gate("private_curriculum_serialized", private_curriculum_stats["rows"] == len(private_rows) and private_curriculum_stats["bytes"] > 0, private_curriculum_stats),
            gate("public_task_manifest_serialized", public_manifest_serialized, public_manifest_stats),
            gate("public_decoder_contract_preflight", False, public_contract_preflight),
        ]
        report = {
            "policy": "project_theseus_code_lm_closure_v1",
            "created_utc": now(),
            "trigger_state": "RED",
            "run_status": "failed",
            "progress_stage": "public_decoder_contract_preflight_failed",
            "decoder_relevant_source_fingerprint": decoder_relevant_source_fingerprint(),
            "decoder_relevant_source_mtime": decoder_relevant_source_mtime() or None,
            "hard_operational_failures": ["public_decoder_contract_preflight"],
            "seed": args.seed,
            "private_curriculum": rel(resolve(args.private_curriculum_out)),
            "public_task_manifest": rel(resolve(args.public_task_manifest_out)),
            "summary": {
                "private_task_count": len(private_rows),
                "public_task_count": len(public_tasks),
                "private_curriculum_serialized": private_curriculum_stats,
                "public_task_manifest_serialized": public_manifest_stats,
                "candidate_generation_shard": shard_summary,
                "public_decoder_contract_preflight": public_contract_preflight,
                "external_inference_calls": 0,
            },
            "gates": gates,
            "external_inference_calls": 0,
        }
        write_json(resolve(args.out), report)
        print(json.dumps(report, indent=2))
        return 2
    if args.preflight_only:
        gates = [
            gate("private_curriculum_generated", len(private_rows) >= 20, f"rows={len(private_rows)}"),
            gate("private_curriculum_serialized", private_curriculum_stats["rows"] == len(private_rows) and private_curriculum_stats["bytes"] > 0, private_curriculum_stats),
            gate("public_task_manifest_serialized", public_manifest_serialized, public_manifest_stats),
            gate("public_decoder_contract_preflight", True, public_contract_preflight),
            gate("external_inference_zero", True, "local manifest/preflight only"),
        ]
        report = {
            "policy": "project_theseus_code_lm_closure_preflight_v1",
            "created_utc": now(),
            "trigger_state": "GREEN" if all(row["passed"] for row in gates) else "YELLOW",
            "run_status": "completed",
            "progress_stage": "public_decoder_contract_preflight_completed",
            "decoder_relevant_source_fingerprint": decoder_relevant_source_fingerprint(),
            "decoder_relevant_source_mtime": decoder_relevant_source_mtime() or None,
            "seed": args.seed,
            "private_curriculum": rel(resolve(args.private_curriculum_out)),
            "public_task_manifest": rel(resolve(args.public_task_manifest_out)),
            "summary": {
                "private_task_count": len(private_rows),
                "public_task_count": len(public_tasks),
                "private_curriculum_serialized": private_curriculum_stats,
                "public_task_manifest_serialized": public_manifest_stats,
                "candidate_generation_shard": shard_summary,
                "public_decoder_contract_preflight": public_contract_preflight,
                "external_inference_calls": 0,
            },
            "gates": gates,
            "external_inference_calls": 0,
        }
        write_json(resolve(args.out), report)
        print(json.dumps(report, indent=2))
        return 0
    sts_conditioning = run_sts_conditioning(args, public_tasks, private_rows)
    args.sts_streams_effective = sts_conditioning["generation_path"]
    step_plan = build_step_plan(args, private_rows, public_tasks, sts_conditioning)

    rust_command = rust_closure_command(args)
    rust_timed_out = False
    resume_completed_rust_used = False
    rust_report = read_json(resolve(args.rust_report_out), {})
    if (not args.disable_rust_resume) and rust_artifacts_fresh(
        rust_report,
        args=args,
        public_task_count=len(public_tasks),
        allow_empty_public=args.private_only,
    ):
        resume_completed_rust_used = True
        rust_result = subprocess.CompletedProcess(
            rust_command,
            0,
            stdout="Reused completed fresh Rust/SymLiquid artifacts from a prior interrupted outer run.",
            stderr="",
        )
    else:
        write_outer_progress_report(
            args,
            stage="rust_symliquid_stage_started",
            step_plan=step_plan,
            private_rows=private_rows,
            public_tasks=public_tasks,
            sts_conditioning=sts_conditioning,
        )
        rust_result = run_process_tree(
            rust_command,
            cwd=ROOT,
            env=symliquid_process_env(args),
            timeout_seconds=timeout_arg(args.rust_timeout_seconds),
        )
        if rust_result.returncode == 124:
            after_timeout_report = read_json(resolve(args.rust_report_out), {})
            if rust_report_completed(
                after_timeout_report,
                checkpoint_out=resolve(args.checkpoint_out),
                public_candidate_out=resolve(args.public_candidate_out),
                rust_timed_out=False,
                allow_empty_public=args.private_only,
            ):
                rust_result = subprocess.CompletedProcess(
                    rust_command,
                    0,
                    stdout=(
                        timeout_text(rust_result.stdout)
                        + "\nRecovered completed Rust/SymLiquid artifacts after the timeout fuse."
                    ).strip(),
                    stderr=timeout_text(rust_result.stderr),
                )
            else:
                rust_timed_out = True
                rust_result = subprocess.CompletedProcess(
                    rust_command,
                    124,
                    stdout=timeout_text(rust_result.stdout),
                    stderr=(
                        timeout_text(rust_result.stderr)
                        + f"\nTimed out after safety fuse={args.rust_timeout_seconds} seconds; "
                        + "killed the Rust/SymLiquid process tree; "
                        + "increase the fuse or reduce step-budget knobs, do not treat wall time as learning duration."
                    ).strip(),
                )
                write_json(
                    resolve(args.rust_report_out),
                    {
                        "policy": "project_theseus_code_lm_closure_rust_v1",
                        "created_utc": now(),
                        "trigger_state": "YELLOW",
                        "run_status": "timed_out_process_tree_killed",
                        "runtime_ms": None,
                        "private_candidate_manifest": rel(resolve(args.private_candidate_out)),
                        "public_candidate_manifest": rel(resolve(args.public_candidate_out)),
                        "summary": {
                            "timeout_seconds": args.rust_timeout_seconds,
                            "process_tree_killed": True,
                            "score_semantics": "timeout_marker_only_not_learning_evidence",
                            "partial_checkpoint_exists": resolve(args.checkpoint_out).exists(),
                            "partial_private_candidate_rows": count_jsonl_rows(resolve(args.private_candidate_out)),
                            "partial_public_candidate_rows": count_jsonl_rows(resolve(args.public_candidate_out)),
                            "previous_report_status": after_timeout_report.get("run_status"),
                            "previous_report_trigger_state": after_timeout_report.get("trigger_state"),
                            "previous_report_summary": object_field(after_timeout_report, "summary"),
                        },
                        "stderr_tail": timeout_text(rust_result.stderr),
                        "external_inference_calls": 0,
                    },
                )
        rust_report = read_json(resolve(args.rust_report_out), {})
    if bool(getattr(args, "checkpoint_only", False)):
        rust_completed_ok = (
            rust_result.returncode == 0
            and rust_report.get("run_status") == "completed"
            and rust_report.get("progress_stage") == "checkpoint_only_completed"
        )
        gates = [
            gate("private_curriculum_generated", len(private_rows) >= 20, f"rows={len(private_rows)}"),
            gate("private_curriculum_serialized", private_curriculum_stats["rows"] == len(private_rows) and private_curriculum_stats["bytes"] > 0, private_curriculum_stats),
            gate("public_task_manifest_serialized", public_manifest_serialized, public_manifest_stats),
            gate("public_decoder_contract_preflight", public_contract_preflight["passed"], public_contract_preflight),
            gate("private_split_hygiene", split_hygiene(private_rows), split_hygiene_detail(private_rows)),
            gate("sts_conditioning_safe", sts_conditioning["safe"], sts_conditioning),
            gate(
                "train_once_checkpoint_written",
                rust_completed_ok and resolve(args.checkpoint_out).exists(),
                {
                    "returncode": rust_result.returncode,
                    "rust_run_status": rust_report.get("run_status"),
                    "rust_progress_stage": rust_report.get("progress_stage"),
                    "checkpoint": rel(resolve(args.checkpoint_out)),
                    "checkpoint_exists": resolve(args.checkpoint_out).exists(),
                },
            ),
            gate(
                "checkpoint_model_artifacts_v1_present",
                bool(get_path(rust_report, ["summary", "model_artifacts_v1_written"], False)),
                get_path(rust_report, ["summary"], {}),
            ),
            gate(
                "cuda_readout_used_when_requested",
                (not args.use_cuda_readout) or bool(get_path(rust_report, ["summary", "cuda_readout_used"], False)),
                {
                    "requested": bool(args.use_cuda_readout),
                    "used": get_path(rust_report, ["summary", "cuda_readout_used"], False),
                    "backend": get_path(rust_report, ["summary", "next_token_readout_backend"], ""),
                },
            ),
            gate("public_tasks_visible_only", all("tests" not in row and "canonical_solution" not in row for row in public_tasks), f"public_tasks={len(public_tasks)}"),
            gate("external_inference_zero", True, "local manifest/STS/Rust checkpoint training only"),
        ]
        hard_operational_failures = [
            row
            for row in gates
            if row["gate"]
            in {
                "private_curriculum_serialized",
                "public_task_manifest_serialized",
                "public_decoder_contract_preflight",
                "train_once_checkpoint_written",
                "checkpoint_model_artifacts_v1_present",
                "cuda_readout_used_when_requested",
                "external_inference_zero",
            }
            and not row["passed"]
        ]
        trigger_state = "RED" if hard_operational_failures else ("GREEN" if all(row["passed"] for row in gates) else "YELLOW")
        report = {
            "policy": "project_theseus_code_lm_closure_checkpoint_only_v1",
            "created_utc": now(),
            "trigger_state": trigger_state,
            "run_status": "completed" if not hard_operational_failures else "failed",
            "progress_stage": "checkpoint_only_completed" if not hard_operational_failures else "checkpoint_only_failed",
            "decoder_relevant_source_fingerprint": decoder_relevant_source_fingerprint(),
            "decoder_relevant_source_mtime": decoder_relevant_source_mtime() or None,
            "hard_operational_failures": [row["gate"] for row in hard_operational_failures],
            "seed": args.seed,
            "private_curriculum": rel(resolve(args.private_curriculum_out)),
            "public_task_manifest": rel(resolve(args.public_task_manifest_out)),
            "checkpoint": rel(resolve(args.checkpoint_out)),
            "rust_report": rel(resolve(args.rust_report_out)),
            "summary": {
                "checkpoint_only": True,
                "train_once_checkpoint_fanout_ready": not hard_operational_failures,
                "repeated_training_per_candidate_shard": False,
                "target_architecture": "train_once_checkpoint_then_hive_distributed_candidate_generation_and_verification",
                "private_task_count": len(private_rows),
                "private_train_task_count": sum(1 for row in private_rows if row["split"] == "train"),
                "private_eval_task_count": sum(1 for row in private_rows if row["split"] == "eval"),
                "public_task_count": len(public_tasks),
                "private_curriculum_serialized": private_curriculum_stats,
                "public_task_manifest_serialized": public_manifest_stats,
                "candidate_generation_shard": shard_summary,
                "sts_conditioning_used": bool(sts_conditioning["generation_path"]),
                "sts_conditioning_report": rel(resolve(args.sts_conditioning_report_out)) if sts_conditioning["generation_path"] else "",
                "sts_generation_path": sts_conditioning["generation_path"],
                "symliquid_state_conditioning_used": bool(symliquid_state.get("loaded")),
                "symliquid_state_report": rel(resolve(args.symliquid_state_engine_report)),
                "before_next_token_accuracy": get_path(rust_report, ["summary", "before_next_token_accuracy"], 0.0),
                "after_next_token_accuracy": get_path(rust_report, ["summary", "after_next_token_accuracy"], 0.0),
                "next_token_accuracy_delta": get_path(rust_report, ["summary", "next_token_accuracy_delta"], 0.0),
                "cuda_readout_requested": bool(args.use_cuda_readout),
                "cuda_readout_used": get_path(rust_report, ["summary", "cuda_readout_used"], False),
                "next_token_readout_backend": get_path(rust_report, ["summary", "next_token_readout_backend"], ""),
                "model_artifacts_v1_written": get_path(rust_report, ["summary", "model_artifacts_v1_written"], False),
                "rust_step_duration": get_path(rust_report, ["summary", "step_duration"], {}),
                "rust_work_budget_admission": get_path(rust_report, ["summary", "work_budget_admission"], {}),
                "external_inference_calls": 0,
            },
            "symliquid_state_conditioning": symliquid_state,
            "sts_conditioning": sts_conditioning,
            "step_duration": step_plan,
            "rust_command": rust_command,
            "rust_returncode": rust_result.returncode,
            "rust_completed_ok": rust_completed_ok,
            "rust_resume_completed_artifacts_used": resume_completed_rust_used,
            "rust_timed_out": rust_timed_out,
            "rust_stdout_tail": rust_result.stdout[-1600:],
            "rust_stderr_tail": rust_result.stderr[-1600:],
            "gates": gates,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "external_inference_calls": 0,
        }
        write_json(resolve(args.out), report)
        print(json.dumps(report, indent=2))
        return 0 if trigger_state in {"GREEN", "YELLOW"} else 2
    private_candidates = read_jsonl(resolve(args.private_candidate_out))
    public_candidate_sanitize = sanitize_public_candidate_manifest(resolve(args.public_candidate_out))
    private_eval = evaluate_private_candidates(private_rows, private_candidates)
    edge_obligation_gate = run_edge_obligation_decode_gate(args) if args.edge_obligation_decode_gate_v1 else {}
    rust_report_is_completed = rust_report_completed(
        rust_report,
        checkpoint_out=resolve(args.checkpoint_out),
        public_candidate_out=resolve(args.public_candidate_out),
        rust_timed_out=rust_timed_out,
        allow_empty_public=args.private_only,
    )
    rust_completed_ok = rust_result.returncode == 0 or rust_report_is_completed

    public_report: dict[str, Any] = {}
    public_result: subprocess.CompletedProcess[str] | None = None
    if not args.skip_public_calibration and rust_completed_ok:
        public_command = [
            sys.executable,
            "scripts/real_code_benchmark_graduation.py",
            "--cards",
            ",".join(public_cards),
            "--seed",
            str(args.seed),
            "--max-cases-per-card",
            str(max(1, args.max_public_cases_per_card)),
            "--skip-student-candidate-generation",
            "--student-candidate-manifest",
            rel(resolve(args.public_candidate_out)),
            "--out",
            rel(resolve(args.public_report_out)),
            "--trace-out",
            rel(resolve(args.public_trace_out)),
            "--transfer-artifact-out",
            rel(resolve(args.public_transfer_artifact_out)),
        ]
        if str(args.case_manifest or "").strip():
            public_command.extend(["--case-manifest", args.case_manifest])
        public_result = subprocess.run(
            public_command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout_arg(args.public_timeout_seconds),
        )
        public_report = read_json(resolve(args.public_report_out), {})

    gates = [
        gate("private_curriculum_generated", len(private_rows) >= 20, f"rows={len(private_rows)}"),
        gate("private_curriculum_serialized", private_curriculum_stats["rows"] == len(private_rows) and private_curriculum_stats["bytes"] > 0, private_curriculum_stats),
        gate("public_task_manifest_serialized", public_manifest_serialized, public_manifest_stats),
        gate("public_decoder_contract_preflight", public_contract_preflight["passed"], public_contract_preflight),
        gate("private_split_hygiene", split_hygiene(private_rows), split_hygiene_detail(private_rows)),
        gate("extra_private_train_governed", extra_private_train_governed(extra_private_rows), f"rows={len(extra_private_rows)} source={args.extra_private_train_jsonl}"),
        gate("residual_private_train_governed", extra_private_train_governed(residual_private_rows), f"rows={len(residual_private_rows)} source={args.residual_private_train_jsonl}"),
        gate("repo_repair_private_train_governed", extra_private_train_governed(repo_repair_private_rows), f"rows={len(repo_repair_private_rows)} source={args.repo_repair_private_train_jsonl}"),
        gate("high_transfer_private_train_governed", extra_private_train_governed(high_transfer_private_rows), f"rows={len(high_transfer_private_rows)} source={args.high_transfer_private_train_jsonl}"),
        gate("public_tasks_visible_only", all("tests" not in row and "canonical_solution" not in row for row in public_tasks), f"public_tasks={len(public_tasks)}"),
        gate("sts_conditioning_safe", sts_conditioning["safe"], sts_conditioning),
        gate(
            "rust_code_lm_trained",
            rust_completed_ok,
            {
                "returncode": rust_result.returncode,
                "timeout": rust_timed_out,
                "trigger": rust_report.get("trigger_state"),
                "run_status": rust_report.get("run_status"),
                "report_completed": rust_report_is_completed,
                "checkpoint": rust_report.get("checkpoint"),
                "public_candidate_manifest": rust_report.get("public_candidate_manifest"),
            },
        ),
        gate("next_token_accuracy_improved", float(get_path(rust_report, ["summary", "next_token_accuracy_delta"], 0.0)) > 0.0, get_path(rust_report, ["summary", "next_token_accuracy_delta"], 0.0)),
        gate(
            "rust_loaded_private_train_rows",
            int(get_path(rust_report, ["summary", "private_train_task_count_before_work_budget"], 0)) > 0,
            {
                "outer_private_train_rows": sum(1 for row in private_rows if row.get("split") == "train"),
                "rust_private_train_rows_before_budget": get_path(rust_report, ["summary", "private_train_task_count_before_work_budget"], 0),
                "rust_input_file_status": get_path(rust_report, ["summary", "input_file_status"], {}),
            },
        ),
        gate("private_execution_eval_ran", private_eval["eval_task_count"] > 0, private_eval),
        gate("private_execution_after_improved", private_eval["trained_pass_rate"] > private_eval["baseline_pass_rate"], private_eval),
        gate("private_sts_repair_causal_nonnegative", private_eval["sts_repair_pass_rate_delta"] >= 0.0 and private_eval["sts_repair_task_level_regressions"] == 0, private_eval),
        gate(
            "private_sts_repair_causal_positive_or_open_wall",
            private_eval["sts_repair_pass_rate_delta"] > 0.0
            and private_eval["sts_repair_task_level_regressions"] == 0,
            private_eval,
        ),
        gate(
            "edge_obligation_decode_gate_v1_private",
            (not args.edge_obligation_decode_gate_v1)
            or bool(edge_obligation_gate.get("ready_for_public_calibration")),
            edge_obligation_gate.get("summary", {"enabled": False}),
        ),
        gate("public_calibration_ran", args.skip_public_calibration or bool(public_report), get_path(public_report, ["summary"], {})),
        gate("public_score_claim_quarantined", args.skip_public_calibration or public_report.get("public_benchmark_score_claim") == "student_code_lm_checkpoint_public_task_calibration_only", public_report.get("public_benchmark_score_claim")),
        gate(
            "public_sts_same_seed_delta_positive_or_open_wall",
            args.skip_public_calibration
            or (
                float(public_report.get("pass_rate_delta") or get_path(public_report, ["summary", "pass_rate_delta"], 0.0) or 0.0) > 0.0
                and int(public_report.get("task_level_regressions") or get_path(public_report, ["summary", "task_level_regressions_vs_single_stream"], 0) or 0) == 0
            ),
            {
                "pass_rate_delta": public_report.get("pass_rate_delta", get_path(public_report, ["summary", "pass_rate_delta"], None)),
                "task_level_regressions": public_report.get("task_level_regressions", get_path(public_report, ["summary", "task_level_regressions_vs_single_stream"], None)),
            },
        ),
        gate("external_inference_zero", True, "local Rust/SymLiquid + Python sandbox only"),
    ]
    hard_operational_failures = [
        row
        for row in gates
        if row["gate"]
        in {
            "private_curriculum_serialized",
            "public_task_manifest_serialized",
            "public_decoder_contract_preflight",
            "rust_code_lm_trained",
            "rust_loaded_private_train_rows",
            "private_execution_eval_ran",
            "public_calibration_ran",
            "public_score_claim_quarantined",
            "external_inference_zero",
        }
        and not row["passed"]
    ]
    trigger_state = "RED" if hard_operational_failures else ("GREEN" if all(row["passed"] for row in gates) else "YELLOW")
    report = {
        "policy": "project_theseus_code_lm_closure_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "run_status": "completed" if not hard_operational_failures else "failed",
        "decoder_relevant_source_fingerprint": decoder_relevant_source_fingerprint(),
        "decoder_relevant_source_mtime": decoder_relevant_source_mtime() or None,
        "hard_operational_failures": [row["gate"] for row in hard_operational_failures],
        "seed": args.seed,
        "private_curriculum": rel(resolve(args.private_curriculum_out)),
        "public_task_manifest": rel(resolve(args.public_task_manifest_out)),
        "checkpoint": rel(resolve(args.checkpoint_out)),
        "private_candidate_manifest": rel(resolve(args.private_candidate_out)),
        "public_candidate_manifest": rel(resolve(args.public_candidate_out)),
        "rust_report": rel(resolve(args.rust_report_out)),
        "public_report": rel(resolve(args.public_report_out)) if public_report else "",
        "summary": {
            "private_task_count": len(private_rows),
            "generated_private_task_count": len(private_rows) - len(extra_private_rows) - len(residual_private_rows) - len(repo_repair_private_rows) - len(high_transfer_private_rows),
            "extra_private_train_task_count": len(extra_private_rows),
            "extra_private_train_jsonl": rel(resolve(args.extra_private_train_jsonl)),
            "residual_private_train_task_count": len(residual_private_rows),
            "residual_private_train_jsonl": rel(resolve(args.residual_private_train_jsonl)),
            "repo_repair_private_train_task_count": len(repo_repair_private_rows),
            "repo_repair_private_train_jsonl": rel(resolve(args.repo_repair_private_train_jsonl)),
            "high_transfer_private_train_task_count": len(high_transfer_private_rows),
            "high_transfer_private_train_jsonl": [rel(resolve(path)) for path in split_path_list(args.high_transfer_private_train_jsonl)],
            "high_transfer_private_train_file_counts": dict(sorted(high_transfer_private_file_counts.items())),
            "public_visible_category_priority": public_category_priority,
            "candidate_generation_shard": shard_summary,
            "private_train_task_count": sum(1 for row in private_rows if row["split"] == "train"),
            "private_eval_task_count": sum(1 for row in private_rows if row["split"] == "eval"),
            "private_curriculum_serialized": private_curriculum_stats,
            "public_task_manifest_serialized": public_manifest_stats,
            "decoder_contract_private_task_count": sum(1 for row in private_rows if isinstance(row.get("decoder_contract"), dict)),
            "decoder_contract_public_task_count": sum(1 for row in public_tasks if isinstance(row.get("decoder_contract"), dict)),
            "public_task_count": len(public_tasks),
            "sts_conditioning_used": bool(sts_conditioning["generation_path"]),
            "sts_conditioned_public_task_count": sts_conditioning["conditioned_public_task_count"],
            "sts_conditioning_report": rel(resolve(args.sts_conditioning_report_out)) if sts_conditioning["generation_path"] else "",
            "symliquid_state_conditioning_used": bool(symliquid_state.get("loaded")),
            "symliquid_state_report": rel(resolve(args.symliquid_state_engine_report)),
            "before_next_token_accuracy": get_path(rust_report, ["summary", "before_next_token_accuracy"], 0.0),
            "after_next_token_accuracy": get_path(rust_report, ["summary", "after_next_token_accuracy"], 0.0),
            "next_token_accuracy_delta": get_path(rust_report, ["summary", "next_token_accuracy_delta"], 0.0),
            "private_baseline_pass_rate": private_eval["baseline_pass_rate"],
            "private_sts_off_pass_rate": private_eval["sts_off_pass_rate"],
            "private_trained_pass_rate": private_eval["trained_pass_rate"],
            "private_pass_rate_delta": private_eval["pass_rate_delta"],
            "private_sts_repair_pass_rate_delta": private_eval["sts_repair_pass_rate_delta"],
            "private_sts_repair_task_level_improvements": private_eval["sts_repair_task_level_improvements"],
            "private_sts_repair_task_level_regressions": private_eval["sts_repair_task_level_regressions"],
            "private_concept_residual_counts": private_eval["concept_residual_counts"],
            "private_concept_family_pass_rates": private_eval["concept_family_pass_rates"],
            "public_real_task_pass_rate": get_path(public_report, ["summary", "real_public_task_pass_rate"], None),
            "public_candidate_source": public_report.get("candidate_source"),
            "public_benchmark_score_claim": public_report.get("public_benchmark_score_claim"),
            "token_level_code_generation_learned": get_path(rust_report, ["summary", "token_level_code_generation_learned"], False),
            "full_body_token_candidate_count": get_path(public_report, ["summary", "full_body_token_candidate_count"], get_path(rust_report, ["summary", "full_body_token_candidate_count"], 0)),
            "compositional_token_candidate_count": get_path(public_report, ["summary", "compositional_token_candidate_count"], get_path(rust_report, ["summary", "compositional_token_candidate_count"], 0)),
            "full_body_public_pass_count": get_path(public_report, ["summary", "full_body_public_pass_count"], 0),
            "expression_fallback_public_pass_count": get_path(public_report, ["summary", "expression_fallback_public_pass_count"], 0),
            "template_like_candidate_count": get_path(public_report, ["summary", "template_like_candidate_count"], 0),
            "loop_closure_candidate_count": get_path(public_report, ["summary", "loop_closure_candidate_count"], 0),
            "invalid_public_promotion_candidates_disabled": public_candidate_sanitize["invalid_promotion_candidates_disabled"],
            "rust_resume_completed_artifacts_used": resume_completed_rust_used,
            "rust_run_status": rust_report.get("run_status"),
            "rust_progress_stage": rust_report.get("progress_stage"),
            "rust_step_duration": get_path(rust_report, ["summary", "step_duration"], {}),
            "rust_work_budget_admission": get_path(rust_report, ["summary", "work_budget_admission"], {}),
            "rust_input_file_status": get_path(rust_report, ["summary", "input_file_status"], {}),
            "typed_edge_exec_receiver_v1_enabled": typed_edge_exec_receiver_enabled(args),
            "private_type_shape_receiver_veto_v1_enabled": private_type_shape_receiver_veto_enabled(args),
            "edge_obligation_decode_gate_enabled": bool(args.edge_obligation_decode_gate_v1),
            "edge_obligation_decode_gate_ready": bool(edge_obligation_gate.get("ready_for_public_calibration")),
            "edge_obligation_decode_gate_report": rel(resolve(args.edge_obligation_report_out)) if edge_obligation_gate else "",
            "decoder_relevant_source_fingerprint": decoder_relevant_source_fingerprint(),
            "decoder_relevant_source_mtime": decoder_relevant_source_mtime() or None,
            "external_inference_calls": 0,
        },
        "private_eval": private_eval,
        "edge_obligation_decode_gate": edge_obligation_gate,
        "symliquid_state_conditioning": symliquid_state,
        "sts_conditioning": sts_conditioning,
        "step_duration": step_plan,
        "public_candidate_sanitize": public_candidate_sanitize,
        "rust_command": rust_command,
        "rust_returncode": rust_result.returncode,
        "rust_completed_ok": rust_completed_ok,
        "rust_resume_completed_artifacts_used": resume_completed_rust_used,
        "rust_timed_out": rust_timed_out,
        "rust_stdout_tail": rust_result.stdout[-1600:],
        "rust_stderr_tail": rust_result.stderr[-1600:],
        "public_returncode": public_result.returncode if public_result else None,
        "public_stdout_tail": public_result.stdout[-1600:] if public_result else "",
        "public_stderr_tail": public_result.stderr[-1600:] if public_result else "",
        "gates": gates,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    print(json.dumps(report, indent=2))
    return 0 if trigger_state in {"GREEN", "YELLOW"} else 2


def run_edge_obligation_decode_gate(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/edge_obligation_decode_gate_v1.py",
        "--private-curriculum",
        rel(resolve(args.private_curriculum_out)),
        "--private-candidates",
        rel(resolve(args.private_candidate_out)),
        "--closure-report",
        rel(resolve(args.out)),
        "--out",
        rel(resolve(args.edge_obligation_report_out)),
        "--markdown-out",
        rel(resolve(args.edge_obligation_markdown_out)),
    ]
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=900)
    except subprocess.TimeoutExpired as exc:
        return {
            "policy": "project_theseus_edge_obligation_decode_gate_v1",
            "trigger_state": "YELLOW",
            "ready_for_public_calibration": False,
            "summary": {
                "timed_out": True,
                "timeout_seconds": 900,
                "stdout_tail": timeout_text(exc.stdout)[-600:],
                "stderr_tail": timeout_text(exc.stderr)[-600:],
            },
        }
    report = read_json(resolve(args.edge_obligation_report_out), {})
    if not report:
        report = {
            "policy": "project_theseus_edge_obligation_decode_gate_v1",
            "trigger_state": "YELLOW",
            "ready_for_public_calibration": False,
            "summary": {
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-600:],
                "stderr_tail": result.stderr[-600:],
            },
        }
    report.setdefault("runner", {})
    report["runner"].update(
        {
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-600:],
            "stderr_tail": result.stderr[-600:],
        }
    )
    return report


def sanitize_public_candidate_manifest(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    if not rows:
        return {"candidate_count": 0, "invalid_promotion_candidates_disabled": 0}
    changed = 0
    for row in rows:
        if not bool(row.get("benchmark_promotion_eligible")):
            continue
        ineligible_reason = ""
        try:
            ast.parse(str(row.get("code") or ""))
        except SyntaxError as exc:
            ineligible_reason = "python_syntax_invalid"
            row["python_parse_error"] = f"{exc.__class__.__name__}: {exc.msg}"
        if not ineligible_reason and row.get("decoder_contract_verifier_v1_passed") is False:
            ineligible_reason = "decoder_contract_verifier_failed"
        if not ineligible_reason and row.get("deterministic_guardrail_passed") is False:
            ineligible_reason = "deterministic_guardrail_failed"
        if not ineligible_reason and bool(row.get("placeholder_scaffold_body")):
            ineligible_reason = "placeholder_scaffold_body"
        if not ineligible_reason and bogus_return_attribute_code(str(row.get("code") or "")):
            ineligible_reason = "bogus_return_attribute"
        if not ineligible_reason and bogus_return_local_callable_code(str(row.get("code") or "")):
            ineligible_reason = "bogus_return_local_callable"
        if ineligible_reason:
            row["benchmark_promotion_eligible"] = False
            row["promotion_ineligible_reason"] = ineligible_reason
            provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
            provenance["benchmark_promotion_eligible"] = False
            provenance["promotion_ineligible_reason"] = ineligible_reason
            row["provenance"] = provenance
            changed += 1
    if changed:
        write_jsonl(path, rows)
    return {
        "candidate_count": len(rows),
        "invalid_promotion_candidates_disabled": changed,
        "rule": "malformed Python may remain residual evidence but cannot be promotion evidence",
    }


def bogus_return_attribute_code(code: str) -> bool:
    blocked = {
        "isinstance",
        "list",
        "dict",
        "tuple",
        "str",
        "int",
        "float",
        "bool",
        "set",
        "len",
        "sum",
        "min",
        "max",
        "sorted",
        "range",
        "append",
        "extend",
        "insert",
        "remove",
        "pop",
        "sort",
        "reverse",
        "items",
        "keys",
        "values",
        "get",
        "split",
        "strip",
        "lower",
        "upper",
        "replace",
        "join",
    }
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Attribute):
            if isinstance(node.value.value, ast.Name) and node.value.attr in blocked:
                return True
    return False


def bogus_return_local_callable_code(code: str) -> bool:
    """Reject bodies like `return total(item)` when `total` is a local value.

    These are syntactically valid Python, so ast.parse alone will not catch
    them. They are almost always broken decoder completions where a local
    accumulator/list/dict is accidentally treated as a function.
    """

    allowed_callables = {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "filter",
        "float",
        "int",
        "len",
        "list",
        "map",
        "max",
        "min",
        "pow",
        "range",
        "reversed",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    }
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for fn in [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]:
        assigned: set[str] = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                assigned.add(node.id)
            elif isinstance(node, (ast.For, ast.AsyncFor)) and isinstance(node.target, ast.Name):
                assigned.add(node.target.id)
            elif isinstance(node, ast.With):
                for item in node.items:
                    if isinstance(item.optional_vars, ast.Name):
                        assigned.add(item.optional_vars.id)
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Return)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in assigned
                and node.value.func.id not in allowed_callables
            ):
                return True
    return False


def rust_report_completed(
    report: dict[str, Any],
    *,
    checkpoint_out: Path,
    public_candidate_out: Path,
    rust_timed_out: bool,
    allow_empty_public: bool = False,
) -> bool:
    """Accept a completed Rust report even if Windows returns a flaky status.

    The Rust/SymLiquid lane writes the checkpoint, candidate manifest, and all
    report gates at the end of a successful run. On Windows we have observed a
    nonzero process code after those artifacts are already complete. This check
    does not forgive timeouts or missing artifacts; it only lets public
    calibration continue when the report itself proves the requested run
    completed cleanly.
    """
    if rust_timed_out:
        return False
    if report.get("policy") != "project_theseus_code_lm_closure_rust_v1":
        return False
    if report.get("run_status") == "in_progress":
        return False
    if report.get("trigger_state") not in {"GREEN", "YELLOW"}:
        return False
    if report.get("checkpoint") != rel(checkpoint_out):
        return False
    if report.get("public_candidate_manifest") != rel(public_candidate_out):
        return False
    summary = object_field(report, "summary")
    if not allow_empty_public and int(summary.get("public_candidate_count") or 0) <= 0:
        return False
    if int(summary.get("full_body_token_candidate_count") or 0) <= 0:
        return False
    return public_candidate_out.exists() and public_candidate_out.stat().st_size > 0


def rust_artifacts_fresh(
    report: dict[str, Any],
    *,
    args: argparse.Namespace,
    public_task_count: int,
    allow_empty_public: bool = False,
) -> bool:
    if not rust_report_completed(
        report,
        checkpoint_out=resolve(args.checkpoint_out),
        public_candidate_out=resolve(args.public_candidate_out),
        rust_timed_out=False,
        allow_empty_public=allow_empty_public,
    ):
        return False
    summary = object_field(report, "summary")
    if int(summary.get("public_task_count") or -1) != int(public_task_count):
        return False
    input_paths = [
        resolve(args.private_curriculum_out),
        resolve(args.public_task_manifest_out),
    ]
    if getattr(args, "sts_streams_effective", ""):
        sts_path = resolve(args.sts_streams_effective)
        recorded_sts = get_path(summary, ["input_file_status", "sts_streams"], {})
        recorded_path = str(recorded_sts.get("path") or "").replace("\\", "/") if isinstance(recorded_sts, dict) else ""
        current_path = str(sts_path).replace("\\", "/")
        recorded_bytes = int(recorded_sts.get("bytes") or -1) if isinstance(recorded_sts, dict) else -1
        current_bytes = sts_path.stat().st_size if sts_path.exists() else -2
        if recorded_path != current_path or recorded_bytes != current_bytes:
            input_paths.append(sts_path)
    output_paths = [
        resolve(args.checkpoint_out),
        resolve(args.private_candidate_out),
        resolve(args.public_candidate_out),
        resolve(args.rust_report_out),
    ]
    for path in output_paths:
        if not path.exists():
            return False
        if path == resolve(args.public_candidate_out) and allow_empty_public:
            continue
        if path.stat().st_size <= 0:
            return False
    if allow_empty_public and int(summary.get("public_candidate_count") or 0) != 0:
        return False
        return False
    newest_input = max((path.stat().st_mtime for path in input_paths if path.exists()), default=0.0)
    return all(path.stat().st_mtime >= newest_input for path in output_paths)


def build_candidate_shard_summary(
    private_rows: list[dict[str, Any]],
    public_tasks: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    private_index, private_count = normalized_shard(
        int(getattr(args, "private_eval_shard_index", 0) or 0),
        int(getattr(args, "private_eval_shard_count", 1) or 1),
    )
    public_index, public_count = normalized_shard(
        int(getattr(args, "public_task_shard_index", 0) or 0),
        int(getattr(args, "public_task_shard_count", 1) or 1),
    )
    eval_rows = [row for row in private_rows if row.get("split") == "eval"]
    public_selected = sharded_rows(public_tasks, public_index, public_count)
    eval_selected = sharded_rows(eval_rows, private_index, private_count)
    return {
        "policy": "project_theseus_code_lm_candidate_generation_shard_v1",
        "mode": "deterministic_eval_and_public_task_shards",
        "private_eval_shard_index": private_index,
        "private_eval_shard_count": private_count,
        "private_eval_total_before_shard": len(eval_rows),
        "private_eval_selected": len(eval_selected),
        "public_task_shard_index": public_index,
        "public_task_shard_count": public_count,
        "public_task_total_before_shard": len(public_tasks),
        "public_task_selected": len(public_selected),
        "train_rows_preserved": sum(1 for row in private_rows if row.get("split") == "train"),
        "score_semantics": "chunking/runtime_envelope_only_not_learning_evidence",
    }


def apply_private_eval_shard(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    index, count = normalized_shard(
        int(getattr(args, "private_eval_shard_index", 0) or 0),
        int(getattr(args, "private_eval_shard_count", 1) or 1),
    )
    if count <= 1:
        return rows
    train_rows = [row for row in rows if row.get("split") != "eval"]
    eval_rows = [row for row in rows if row.get("split") == "eval"]
    return train_rows + sharded_rows(eval_rows, index, count)


def apply_public_task_shard(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    index, count = normalized_shard(
        int(getattr(args, "public_task_shard_index", 0) or 0),
        int(getattr(args, "public_task_shard_count", 1) or 1),
    )
    if count <= 1:
        return rows
    return sharded_rows(rows, index, count)


def normalized_shard(index: int, count: int) -> tuple[int, int]:
    count = max(1, int(count))
    index = max(0, int(index))
    if index >= count:
        index = index % count
    return index, count


def sharded_rows(rows: list[dict[str, Any]], index: int, count: int) -> list[dict[str, Any]]:
    if count <= 1:
        return rows
    ordered = sorted(rows, key=stable_task_sort_key)
    return [row for offset, row in enumerate(ordered) if offset % count == index]


def stable_task_sort_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("card_id") or row.get("source_id") or ""),
        str(row.get("category") or ""),
        str(row.get("task_id") or row.get("entry_point") or row.get("source_task_id") or ""),
    )


def write_outer_progress_report(
    args: argparse.Namespace,
    *,
    stage: str,
    step_plan: dict[str, Any],
    private_rows: list[dict[str, Any]],
    public_tasks: list[dict[str, Any]],
    sts_conditioning: dict[str, Any],
) -> None:
    progress = {
        "policy": "project_theseus_code_lm_closure_v1",
        "created_utc": now(),
        "trigger_state": "YELLOW",
        "run_status": "in_progress",
        "progress_stage": stage,
        "decoder_relevant_source_fingerprint": decoder_relevant_source_fingerprint(),
        "decoder_relevant_source_mtime": decoder_relevant_source_mtime() or None,
        "seed": args.seed,
        "private_curriculum": rel(resolve(args.private_curriculum_out)),
        "public_task_manifest": rel(resolve(args.public_task_manifest_out)),
        "rust_report": rel(resolve(args.rust_report_out)),
        "summary": {
            "private_task_count": len(private_rows),
            "private_train_task_count": sum(1 for row in private_rows if row.get("split") == "train"),
            "private_eval_task_count": sum(1 for row in private_rows if row.get("split") == "eval"),
            "public_task_count": len(public_tasks),
            "sts_conditioning_used": bool(sts_conditioning.get("generation_path")),
            "decoder_relevant_source_fingerprint": decoder_relevant_source_fingerprint(),
            "decoder_relevant_source_mtime": decoder_relevant_source_mtime() or None,
            "external_inference_calls": 0,
        },
        "step_duration": step_plan,
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), progress)


def split_hygiene(rows: list[dict[str, Any]]) -> bool:
    train_ids = {row["task_id"] for row in rows if row.get("split") == "train"}
    eval_ids = {row["task_id"] for row in rows if row.get("split") == "eval"}
    return bool(train_ids) and bool(eval_ids) and not (train_ids & eval_ids)


def split_hygiene_detail(rows: list[dict[str, Any]]) -> str:
    train = sum(1 for row in rows if row.get("split") == "train")
    evals = sum(1 for row in rows if row.get("split") == "eval")
    return f"train={train} eval={evals}"


def extra_private_train_governed(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        if row.get("split") != "train":
            return False
        if row.get("public_benchmark") is not False:
            return False
        if str(row.get("benchmark_evidence_level") or "") not in {
            "permissive_open_source_train_only",
            "private_residual_generated_train_only",
            "private_type_contract_decoder_feedback_train_only",
            "private_plan_ir_generated_training_only",
        }:
            return False
        if not str(row.get("license_spdx") or "").strip():
            return False
    return True


def classify_failure(stderr: str) -> str:
    text = stderr.lower()
    if "lint_parse_failed" in text or "candidate_compile_failed" in text or "test_harness_compile_failed" in text:
        return "verification_cascade_compile"
    if "runtime_failed" in text or "sandbox_launch_failed" in text:
        return "runtime_load_failure"
    if "syntaxerror" in text or "indentationerror" in text:
        return "parsing"
    if "typeerror" in text or "attributeerror" in text:
        return "type_handling"
    if "assertionerror" in text:
        return "wrong_answer"
    if "timeout" in text:
        return "timeout"
    return "runtime"


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def ratio(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def object_field(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    return item if isinstance(item, dict) else {}


def safe_name(value: Any) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value or "item")).strip("_") or "item"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def runtime_tmp_dir() -> Path:
    return benchmark_runtime_tmp_dir()


def read_json(path: Path, default: Any = None) -> Any:
    default = {} if default is None else default
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def public_manifest_serialized_ok(
    stats: dict[str, Any],
    public_tasks: list[dict[str, Any]],
    *,
    private_only: bool = False,
) -> bool:
    if int(stats.get("rows") or 0) != len(public_tasks):
        return False
    if private_only:
        return len(public_tasks) == 0
    return int(stats.get("bytes") or 0) > 0


def jsonl_file_stats(path: Path) -> dict[str, Any]:
    rows = 0
    decode_errors = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                json.loads(line)
                rows += 1
            except json.JSONDecodeError:
                decode_errors += 1
    return {
        "path": rel(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "rows": rows,
        "decode_errors": decode_errors,
        "score_semantics": "serialization_integrity_only_not_learning_evidence",
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    if path.exists():
        try:
            if path.read_text(encoding="utf-8") == text:
                return
        except OSError:
            pass
    path.write_text(text, encoding="utf-8")


def timeout_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
