"""Audit Mac/MLX parity against the current Windows/CUDA training surface."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from macos_mlx_parity_audit_support import (
    get_path,
    int_value,
    now,
    platform_report,
    read_json,
    rel,
    render_markdown,
    resolve_repo_path,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports" / "macos_mlx_parity_audit.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "macos_mlx_parity_audit.md"
WORK_PROOF = ROOT / "reports" / "macos_mlx_work_proof.json"
WORK_PROOF_DIR = ROOT / "reports" / "macos_mlx_work_proof"
ACCELERATOR_PARITY_MANIFEST = ROOT / "reports" / "accelerator_parity_manifest.json"
METAL_ROLLOUT_PROOF = ROOT / "reports" / "macos_metal_rollout_hot_loop_proof.json"
METAL_ROLLOUT_CLI_PROOF = ROOT / "reports" / "macos_metal_rollout_cli_proof.json"
METAL_ROLLOUT_FEATURE_PROOF = ROOT / "reports" / "macos_metal_rollout_feature_proof.json"
METAL_ROLLOUT_READOUT_PROOF = ROOT / "reports" / "macos_metal_rollout_readout_proof.json"
METAL_ROLLOUT_READOUT_TRAINING_PROOF = ROOT / "reports" / "macos_metal_rollout_readout_training_proof.json"
METAL_TRAIN_PATH_PROOF = ROOT / "reports" / "macos_metal_train_path_proof.json"
METAL_STATE_TRAINING_PROOF = ROOT / "reports" / "macos_metal_rollout_state_training_proof.json"
METAL_TOKEN_SUPERPOSITION_READOUT_PROOF = ROOT / "reports" / "macos_metal_token_superposition_readout_proof.json"
METAL_TOKEN_SUPERPOSITION_REPORT = ROOT / "reports" / "token_superposition_metal_training.json"
METAL_TOKEN_SUPERPOSITION_ROUTE_POLICY = ROOT / "configs" / "macos_metal_token_superposition_route_policy.json"
METAL_TOKEN_SUPERPOSITION_CANARY_POLICY = ROOT / "configs" / "macos_metal_token_superposition_scheduler_canary_policy.json"
METAL_TOKEN_SUPERPOSITION_LADDER = ROOT / "reports" / "macos_metal_token_superposition_ladder.json"
METAL_TOKEN_SUPERPOSITION_SCHEDULER_CANARY = ROOT / "reports" / "macos_metal_token_superposition_scheduler_canary.json"
METAL_TRAIN_ROLLOUT_REPORT = ROOT / "reports" / "symliquid_rollout_metal_train_report.json"
METAL_TRAIN_STANDALONE_REPORT = ROOT / "reports" / "symliquid_standalone_metal_train_report.json"
METAL_STANDALONE_ROUTE_POLICY = ROOT / "configs" / "macos_metal_standalone_route_policy.json"
METAL_STANDALONE_CANARY_POLICY = ROOT / "configs" / "macos_metal_standalone_scheduler_canary_policy.json"
METAL_STANDALONE_SCHEDULER_CANARY = ROOT / "reports" / "macos_metal_standalone_scheduler_canary.json"
METAL_ROUTE_POLICY = ROOT / "configs" / "macos_metal_route_policy.json"
METAL_CANARY_POLICY = ROOT / "configs" / "macos_metal_scheduler_canary_policy.json"
METAL_SCHEDULER_DRY_RUN = ROOT / "reports" / "macos_metal_scheduler_dry_run.json"
METAL_PARITY_LADDER = ROOT / "reports" / "macos_metal_parity_ladder.json"
METAL_SCHEDULER_CANARY = ROOT / "reports" / "macos_metal_scheduler_canary.json"
METAL_TRAIN_ROLLOUT_SWEEP_REPORT = ROOT / "reports" / "symliquid_rollout_metal_sweep.json"
METAL_ROLLOUT_SWEEP_ROUTE_POLICY = ROOT / "configs" / "macos_metal_rollout_sweep_route_policy.json"
METAL_ROLLOUT_SWEEP_CANARY_POLICY = ROOT / "configs" / "macos_metal_rollout_sweep_scheduler_canary_policy.json"
METAL_ROLLOUT_SWEEP_SCHEDULER_CANARY = ROOT / "reports" / "macos_metal_rollout_sweep_scheduler_canary.json"
METAL_PRODUCTION_ROUTE_READINESS = ROOT / "reports" / "macos_metal_production_route_readiness.json"

WORKER_REPORTS = {
    "mlx_eval_chunk": WORK_PROOF_DIR / "mlx_eval_chunk.json",
    "mlx_training_chunk": WORK_PROOF_DIR / "mlx_training_chunk.json",
    "mlx_rollout_chunk": WORK_PROOF_DIR / "mlx_rollout_chunk.json",
}

CLI_REPORTS = {
    "train-standalone-mlx": WORK_PROOF_DIR / "cli_train_standalone_mlx.json",
    "train-rollout-mlx": WORK_PROOF_DIR / "cli_train_rollout_mlx.json",
    "train-rollout-mlx-sweep": WORK_PROOF_DIR / "cli_train_rollout_mlx_sweep.json",
    "train-token-superposition-mlx": WORK_PROOF_DIR / "cli_train_token_superposition_mlx.json",
}

HIVE_TASK_PARITY = [
    {
        "cuda": "cuda_eval_chunk",
        "mlx": "mlx_eval_chunk",
        "status": "implemented",
        "owner": "scripts/hive_worker_chunk.py",
        "notes": "MLX BabyLM eval chunk is the current Apple Silicon smoke/eval path.",
    },
    {
        "cuda": "cuda_training_chunk",
        "mlx": "mlx_training_chunk",
        "status": "implemented",
        "owner": "scripts/hive_worker_chunk.py",
        "notes": "MLX BabyLM train chunk uses cached features and MLX tensors.",
    },
    {
        "cuda": "cuda_rollout_chunk",
        "mlx": "mlx_rollout_chunk",
        "status": "implemented",
        "owner": "scripts/hive_worker_chunk.py",
        "notes": "MLX rollout/control probe gives Apple Silicon a bounded control lane; Rust/CUDA rollout remains a separate deeper port.",
    },
]

RUST_CLI_PARITY = [
    {
        "cuda": "train-standalone-cuda",
        "mac": "train-standalone-mlx",
        "status": "implemented_python_mlx_bridge",
        "kernel_port_status": "pending_rust_metal_or_rust_mlx",
        "bridge_route": "apple_silicon_mlx_bounded_work",
        "kernel_route_until_ported": "windows_nvidia_cuda",
        "notes": "symliquid-cli now exposes a real MLX-backed readout command via scripts/macos_mlx_training.py; exact Rust CGS CUDA kernel parity is still a deeper port target.",
    },
    {
        "cuda": "train-rollout-cuda",
        "mac": "train-rollout-mlx",
        "status": "implemented_python_mlx_bridge",
        "kernel_port_status": "pending_rust_metal_or_rust_mlx",
        "bridge_route": "apple_silicon_mlx_bounded_work",
        "kernel_route_until_ported": "windows_nvidia_cuda",
        "notes": "symliquid-cli now exposes a bounded MLX rollout/control command; exact Rust/CUDA rollout hot-loop parity is still pending.",
    },
    {
        "cuda": "train-rollout-cuda-sweep",
        "mac": "train-rollout-mlx-sweep",
        "status": "implemented_python_mlx_bridge",
        "kernel_port_status": "pending_rust_metal_or_rust_mlx",
        "bridge_route": "apple_silicon_mlx_bounded_work",
        "kernel_route_until_ported": "windows_nvidia_cuda",
        "notes": "symliquid-cli now exposes a bounded MLX rollout sweep over the MLX rollout worker.",
    },
    {
        "cuda": "train-token-superposition-cuda",
        "mac": "train-token-superposition-mlx",
        "status": "implemented_python_mlx_bridge",
        "kernel_port_status": "pending_rust_metal_or_rust_mlx",
        "bridge_route": "apple_silicon_mlx_bounded_work",
        "kernel_route_until_ported": "windows_nvidia_cuda",
        "notes": "symliquid-cli now exposes a real MLX token-superposition readout lane; Rust/Metal kernel parity remains pending.",
    },
]

PENDING_KERNEL_STATUS = "pending_rust_metal_or_rust_mlx"
GUARDED_METAL_PROOF_STATUS = "rust_metal_guarded_proof_ready"
GUARDED_METAL_STATUS = "rust_metal_guarded_canary_ready"
FULL_PARITY_READY_STATUSES = {"rust_metal_ready", "rust_mlx_ready", "native_kernel_ready"}
NATIVE_KERNEL_READY_STATUSES = {*FULL_PARITY_READY_STATUSES, GUARDED_METAL_PROOF_STATUS, GUARDED_METAL_STATUS}


def guarded_cli_parity_rows(cli_text: str) -> list[dict[str, Any]]:
    rows = [dict(row) for row in RUST_CLI_PARITY]
    standalone_guarded = train_standalone_metal_guarded_proof_ready(cli_text)
    standalone_canary = train_standalone_metal_guarded_canary_ready(cli_text)
    rollout_guarded = train_rollout_metal_guarded_canary_ready(cli_text)
    rollout_sweep_canary = train_rollout_metal_sweep_guarded_canary_ready(cli_text)
    rollout_sweep_guarded = train_rollout_metal_sweep_guarded_proof_ready(cli_text)
    rollout_state_training = train_rollout_metal_state_training_proof_ready(cli_text)
    token_superposition_guarded = train_token_superposition_metal_guarded_canary_ready(cli_text)
    for row in rows:
        if row.get("cuda") == "train-standalone-cuda" and standalone_canary:
            row.update(
                {
                    "status": "implemented_python_mlx_bridge_with_guarded_rust_metal_canary",
                    "kernel_port_status": GUARDED_METAL_STATUS,
                    "kernel_route_until_ported": "apple_metal_guarded_canary_only",
                    "production_scheduler_routing_enabled": False,
                    "guarded_native_kernel_canary_ready": True,
                    "notes": (
                        "symliquid-cli has the MLX bridge plus a guarded Rust/Metal "
                        "train-standalone canary with canonical artifact, route policy, "
                        "scheduler evidence, and no-cheat locks. Production scheduler "
                        "routing and full parity claims remain locked."
                    ),
                }
            )
        elif row.get("cuda") == "train-standalone-cuda" and standalone_guarded:
            row.update(
                {
                    "status": "implemented_python_mlx_bridge_with_guarded_rust_metal_proof",
                    "kernel_port_status": GUARDED_METAL_PROOF_STATUS,
                    "kernel_route_until_ported": "apple_metal_guarded_proof_only",
                    "production_scheduler_routing_enabled": False,
                    "guarded_native_kernel_proof_ready": True,
                    "notes": (
                        "symliquid-cli has the MLX bridge plus a guarded Rust/Metal "
                        "train-standalone proof with canonical artifact and no-cheat evidence. "
                        "Production scheduler routing and full parity claims remain locked."
                    ),
                }
            )
        elif row.get("cuda") == "train-rollout-cuda" and rollout_guarded:
            row.update(
                {
                    "status": "implemented_python_mlx_bridge_with_guarded_rust_metal_canary",
                    "kernel_port_status": GUARDED_METAL_STATUS,
                    "kernel_route_until_ported": "apple_metal_guarded_canary_only",
                    "production_scheduler_routing_enabled": False,
                    "guarded_native_kernel_canary_ready": True,
                    "notes": (
                        "symliquid-cli has the MLX bridge plus a guarded Rust/Metal "
                        "train-rollout canary with artifact, ladder, scheduler, and no-cheat "
                        "evidence. Production scheduler routing and full parity claims remain locked."
                    ),
                }
            )
        elif row.get("cuda") == "train-rollout-cuda-sweep" and rollout_sweep_canary:
            row.update(
                {
                    "status": "implemented_python_mlx_bridge_with_guarded_rust_metal_sweep_canary",
                    "kernel_port_status": GUARDED_METAL_STATUS,
                    "kernel_route_until_ported": "apple_metal_guarded_sweep_canary_only",
                    "production_scheduler_routing_enabled": False,
                    "guarded_native_kernel_canary_ready": True,
                    "state_training_semantics_proof_ready": rollout_state_training,
                    "state_training_native_ported": rollout_state_training,
                    "notes": (
                        "symliquid-cli has the MLX bridge plus a guarded Rust/Metal "
                        "train-rollout sweep canary with child artifacts, route policy, "
                        "scheduler evidence, and no-cheat locks. The bounded Rust/Metal "
                        "state-training proof is recorded separately; production scheduler "
                        "routing and full parity claims remain locked."
                    ),
                }
            )
        elif row.get("cuda") == "train-rollout-cuda-sweep" and rollout_sweep_guarded:
            row.update(
                {
                    "status": "implemented_python_mlx_bridge_with_guarded_rust_metal_sweep_proof",
                    "kernel_port_status": GUARDED_METAL_PROOF_STATUS,
                    "kernel_route_until_ported": "apple_metal_guarded_sweep_proof_only",
                    "production_scheduler_routing_enabled": False,
                    "guarded_native_kernel_proof_ready": True,
                    "state_training_semantics_proof_ready": rollout_state_training,
                    "state_training_native_ported": rollout_state_training,
                    "notes": (
                        "symliquid-cli has the MLX bridge plus a guarded Rust/Metal "
                        "train-rollout sweep proof with child artifacts and no-cheat evidence. "
                        "The bounded Rust/Metal state-training proof is recorded separately; "
                        "production scheduler routing and full parity claims remain locked."
                    ),
                }
            )
        elif row.get("cuda") == "train-token-superposition-cuda" and token_superposition_guarded:
            row.update(
                {
                    "status": "implemented_python_mlx_bridge_with_guarded_rust_metal_canary",
                    "kernel_port_status": GUARDED_METAL_STATUS,
                    "kernel_route_until_ported": "apple_metal_guarded_canary_only",
                    "production_scheduler_routing_enabled": False,
                    "guarded_native_kernel_canary_ready": True,
                    "notes": (
                        "symliquid-cli has the MLX bridge plus a guarded Rust/Metal "
                        "train-token-superposition canary with artifact, ladder, scheduler, "
                        "and no-cheat evidence. Production scheduler routing and full parity "
                        "claims remain locked."
                    ),
                }
            )
    return rows


def train_standalone_metal_guarded_proof_ready(cli_text: str) -> bool:
    report = read_json(METAL_TRAIN_STANDALONE_REPORT, {})
    return all(
        [
            command_present("train-standalone-metal", cli_text),
            report.get("ok"),
            report.get("state") == "GREEN",
            report.get("command") == "train-standalone-metal",
            get_path(report, ["report_contract", "matches_train_standalone_cli_surface"], False),
            validate_train_standalone_metal_artifact(report).get("ok"),
            get_path(report, ["report_contract", "scheduler_routing_enabled"], True) is False,
            get_path(report, ["report_contract", "python_mlx_bridge_used"], True) is False,
            get_path(report, ["guardrails", "no_fallback_returns"], False),
            int(report.get("external_inference_calls") or 0) == 0,
            report.get("teacher_used") is False,
            int(report.get("public_training_rows") or 0) == 0,
            report.get("model_promotion_allowed") is False,
            report.get("train_standalone_parity_claim_allowed") is False,
            report.get("full_cli_parity_claim_allowed") is False,
        ]
    )


def train_standalone_metal_guarded_canary_ready(cli_text: str) -> bool:
    if not train_standalone_metal_guarded_proof_ready(cli_text):
        return False
    canary = read_json(METAL_STANDALONE_SCHEDULER_CANARY, {})
    return bool(validate_train_standalone_metal_scheduler_canary(canary).get("ok"))


def train_rollout_metal_guarded_canary_ready(cli_text: str) -> bool:
    report = read_json(METAL_TRAIN_ROLLOUT_REPORT, {})
    artifact = validate_train_rollout_metal_artifact(report)
    route_policy = validate_train_rollout_metal_route_policy(report, artifact)
    return all(
        [
            command_present("train-rollout-metal", cli_text),
            report.get("ok"),
            report.get("command") == "train-rollout-metal",
            get_path(report, ["report_contract", "matches_train_rollout_cli_surface"], False),
            artifact.get("ok"),
            route_policy.get("ok"),
            validate_train_rollout_metal_scheduler_dry_run(read_json(METAL_SCHEDULER_DRY_RUN, {})).get("ok"),
            validate_train_rollout_metal_parity_ladder(read_json(METAL_PARITY_LADDER, {})).get("ok"),
            validate_train_rollout_metal_scheduler_canary(read_json(METAL_SCHEDULER_CANARY, {})).get("ok"),
        ]
    )


def train_rollout_metal_sweep_guarded_proof_ready(cli_text: str) -> bool:
    report = read_json(METAL_TRAIN_ROLLOUT_SWEEP_REPORT, {})
    validation = validate_train_rollout_metal_sweep(report)
    return all(
        [
            command_present("train-rollout-metal-sweep", cli_text),
            validation.get("ok"),
            report.get("ok"),
            report.get("trigger_state") == "GREEN",
            report.get("command") == "train-rollout-metal-sweep",
            get_path(report, ["report_contract", "matches_train_rollout_sweep_cli_surface"], False),
            get_path(report, ["report_contract", "python_mlx_bridge_used"], True) is False,
            get_path(report, ["report_contract", "scheduler_routing_enabled"], True) is False,
            get_path(report, ["report_contract", "state_training_native_ported"], True) is False,
            get_path(report, ["guardrails", "does_not_claim_cuda_state_training_parity"], False),
            get_path(report, ["guardrails", "no_fallback_returns"], False),
            int(report.get("external_inference_calls") or 0) == 0,
            report.get("teacher_used") is False,
            int(report.get("public_training_rows") or 0) == 0,
            report.get("model_promotion_allowed") is False,
            report.get("train_rollout_sweep_parity_claim_allowed") is False,
            report.get("full_cli_parity_claim_allowed") is False,
        ]
    )


def train_rollout_metal_sweep_guarded_canary_ready(cli_text: str) -> bool:
    if not train_rollout_metal_sweep_guarded_proof_ready(cli_text):
        return False
    canary = read_json(METAL_ROLLOUT_SWEEP_SCHEDULER_CANARY, {})
    route_policy = read_json(METAL_ROLLOUT_SWEEP_ROUTE_POLICY, {})
    return bool(validate_train_rollout_metal_sweep_scheduler_canary(canary, route_policy).get("ok"))


def train_rollout_metal_state_training_proof_ready(cli_text: str) -> bool:
    return bool(
        command_present("rollout-metal-state-training-proof", cli_text)
        and validate_rollout_metal_state_training_proof(
            read_json(METAL_STATE_TRAINING_PROOF, {})
        ).get("ok")
    )


def train_token_superposition_metal_guarded_canary_ready(cli_text: str) -> bool:
    report = read_json(METAL_TOKEN_SUPERPOSITION_REPORT, {})
    route_policy = read_json(METAL_TOKEN_SUPERPOSITION_ROUTE_POLICY, {})
    contract = validate_train_token_superposition_metal_contract(report)
    return all(
        [
            command_present("train-token-superposition-metal", cli_text),
            validate_token_superposition_metal_readout_proof(
                read_json(METAL_TOKEN_SUPERPOSITION_READOUT_PROOF, {})
            ).get("ok"),
            contract.get("ok"),
            validate_train_token_superposition_metal_artifact(report).get("ok"),
            validate_train_token_superposition_metal_ladder(
                read_json(METAL_TOKEN_SUPERPOSITION_LADDER, {}),
                route_policy,
                contract,
            ).get("ok"),
            validate_train_token_superposition_metal_scheduler_canary(
                read_json(METAL_TOKEN_SUPERPOSITION_SCHEDULER_CANARY, {}),
                route_policy,
            ).get("ok"),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Report current Mac/MLX parity against Windows/CUDA surfaces.")
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()

    policy = read_json(ROOT / "configs" / "hive_policy.json", {})
    worker = ROOT / "scripts" / "hive_worker_chunk.py"
    cli = ROOT / "crates" / "symliquid-cli" / "src" / "main.rs"
    worker_text = worker.read_text(encoding="utf-8", errors="ignore") if worker.exists() else ""
    cli_text = cli.read_text(encoding="utf-8", errors="ignore") if cli.exists() else ""
    mlx = mlx_status()
    work_proof = read_json(WORK_PROOF, {})
    task_rows = [task_row(row, policy, worker_text, work_proof) for row in HIVE_TASK_PARITY]
    cli_rows = [cli_row(row, cli_text, work_proof) for row in guarded_cli_parity_rows(cli_text)]
    native_proof = native_hot_loop_proof(cli_rows, cli_text, worker_text, work_proof)
    missing = [row for row in task_rows + cli_rows if row.get("status") in {"missing", "cpu_fallback_only", "missing_policy_or_runner"}]
    kernel_pending = [row for row in cli_rows if row.get("kernel_port_status") == PENDING_KERNEL_STATUS]
    kernel_ready = [row for row in cli_rows if row.get("kernel_port_status") in NATIVE_KERNEL_READY_STATUSES]
    guarded_proof_ready = [row for row in cli_rows if row.get("kernel_port_status") == GUARDED_METAL_PROOF_STATUS]
    guarded_ready = [row for row in cli_rows if row.get("kernel_port_status") == GUARDED_METAL_STATUS]
    production_route_pending = [
        row for row in cli_rows if row.get("kernel_parity_ready") and not row.get("production_route_ready")
    ]
    runnable_missing = [row for row in task_rows + cli_rows if not get_path(row, ["latest_evidence", "ok"], False)]
    routing = routing_decisions(task_rows, cli_rows, mlx)
    report = {
        "ok": True,
        "policy": "project_theseus_macos_mlx_parity_audit_v0",
        "created_utc": now(),
        "platform": platform_report(),
        "mlx": mlx,
        "state": "GREEN"
        if not missing and not kernel_pending and not production_route_pending and mlx.get("available")
        else "YELLOW"
        if mlx.get("available")
        else "RED",
        "summary": {
            "hive_task_rows": len(task_rows),
            "rust_cli_rows": len(cli_rows),
            "implemented_hive_mlx_tasks": len([row for row in task_rows if row.get("status") == "implemented"]),
            "implemented_mlx_cli_bridges": len(
                [row for row in cli_rows if str(row.get("status") or "").startswith("implemented_python_mlx_bridge")]
            ),
            "runnable_mlx_bridge_count": len([row for row in cli_rows if row.get("bridge_ready")]),
            "kernel_parity_ready_count": len(kernel_ready),
            "kernel_parity_guarded_proof_count": len(guarded_proof_ready),
            "kernel_parity_guarded_canary_count": len(guarded_ready),
            "kernel_parity_pending_count": len(kernel_pending),
            "production_route_pending_count": len(production_route_pending),
            "cuda_hot_loop_route_required_count": len([row for row in cli_rows if row.get("cuda_hot_loop_route_required")]),
            "rust_or_metal_kernel_pending": len(kernel_pending),
            "missing_or_cpu_fallback": len(missing),
            "latest_worker_evidence_ok": len([row for row in task_rows if get_path(row, ["latest_evidence", "ok"], False)]),
            "latest_cli_evidence_ok": len([row for row in cli_rows if get_path(row, ["latest_evidence", "ok"], False)]),
            "runnable_evidence_missing": len(runnable_missing),
            "token_superposition_promoted": any(
                get_path(row, ["latest_evidence", "promotion_decision", "promote_to_training_lane"], False)
                for row in cli_rows
                if row.get("mac") == "train-token-superposition-mlx"
            ),
            "native_hot_loop_parity_claim_allowed": bool(native_proof.get("parity_claim_allowed")),
            "native_subkernel_proof_count": int(get_path(native_proof, ["summary", "native_subkernel_proof_count"], 0) or 0),
            "native_cli_proof_count": int(get_path(native_proof, ["summary", "native_cli_proof_count"], 0) or 0),
            "native_feature_proof_count": int(get_path(native_proof, ["summary", "native_feature_proof_count"], 0) or 0),
            "native_readout_proof_count": int(get_path(native_proof, ["summary", "native_readout_proof_count"], 0) or 0),
            "native_readout_training_proof_count": int(get_path(native_proof, ["summary", "native_readout_training_proof_count"], 0) or 0),
            "native_train_path_proof_count": int(get_path(native_proof, ["summary", "native_train_path_proof_count"], 0) or 0),
            "native_train_standalone_cli_contract_count": int(get_path(native_proof, ["summary", "native_train_standalone_cli_contract_count"], 0) or 0),
            "native_train_standalone_artifact_equivalence_count": int(get_path(native_proof, ["summary", "native_train_standalone_artifact_equivalence_count"], 0) or 0),
            "native_train_standalone_scheduler_canary_count": int(get_path(native_proof, ["summary", "native_train_standalone_scheduler_canary_count"], 0) or 0),
            "native_train_rollout_cli_contract_count": int(get_path(native_proof, ["summary", "native_train_rollout_cli_contract_count"], 0) or 0),
            "native_train_rollout_artifact_equivalence_count": int(get_path(native_proof, ["summary", "native_train_rollout_artifact_equivalence_count"], 0) or 0),
            "native_train_rollout_scheduler_guardrail_count": int(get_path(native_proof, ["summary", "native_train_rollout_scheduler_guardrail_count"], 0) or 0),
            "native_train_rollout_scheduler_dry_run_count": int(get_path(native_proof, ["summary", "native_train_rollout_scheduler_dry_run_count"], 0) or 0),
            "native_train_rollout_parity_ladder_count": int(get_path(native_proof, ["summary", "native_train_rollout_parity_ladder_count"], 0) or 0),
            "native_train_rollout_scheduler_canary_count": int(get_path(native_proof, ["summary", "native_train_rollout_scheduler_canary_count"], 0) or 0),
            "native_train_rollout_sweep_cli_contract_count": int(get_path(native_proof, ["summary", "native_train_rollout_sweep_cli_contract_count"], 0) or 0),
            "native_train_rollout_sweep_child_artifact_count": int(get_path(native_proof, ["summary", "native_train_rollout_sweep_child_artifact_count"], 0) or 0),
            "native_train_rollout_sweep_scheduler_canary_count": int(get_path(native_proof, ["summary", "native_train_rollout_sweep_scheduler_canary_count"], 0) or 0),
            "native_train_rollout_state_training_proof_count": int(get_path(native_proof, ["summary", "native_train_rollout_state_training_proof_count"], 0) or 0),
            "native_token_superposition_readout_proof_count": int(get_path(native_proof, ["summary", "native_token_superposition_readout_proof_count"], 0) or 0),
            "native_train_token_superposition_cli_contract_count": int(get_path(native_proof, ["summary", "native_train_token_superposition_cli_contract_count"], 0) or 0),
            "native_train_token_superposition_artifact_equivalence_count": int(get_path(native_proof, ["summary", "native_train_token_superposition_artifact_equivalence_count"], 0) or 0),
            "native_train_token_superposition_ladder_count": int(get_path(native_proof, ["summary", "native_train_token_superposition_ladder_count"], 0) or 0),
            "native_train_token_superposition_scheduler_canary_count": int(get_path(native_proof, ["summary", "native_train_token_superposition_scheduler_canary_count"], 0) or 0),
            "native_metal_production_route_readiness_review_count": int(get_path(native_proof, ["summary", "native_metal_production_route_readiness_review_count"], 0) or 0),
            "native_metal_production_route_ready_count": int(get_path(native_proof, ["summary", "native_metal_production_route_ready_count"], 0) or 0),
            "native_metal_production_route_blocker_count": int(get_path(native_proof, ["summary", "native_metal_production_route_blocker_count"], 0) or 0),
            "accelerator_parity_manifest_count": 1 if accelerator_parity_manifest_ready() else 0,
            "native_hot_loop_targets_pending": int(get_path(native_proof, ["summary", "pending_native_hot_loop_targets"], 0) or 0),
        },
        "hive_task_parity": task_rows,
        "rust_cli_parity": cli_rows,
        "native_hot_loop_proof": native_proof,
        "routing_decisions": routing,
        "latest_work_proof": compact_work_proof(work_proof),
        "next_implementation_targets": next_targets(task_rows, cli_rows, mlx),
        "guardrails": {
            "mac_native_first": "Prefer MLX on Apple Silicon, CPU/storage/operator on Intel Macs.",
            "no_fake_parity": "Do not label CUDA rollout/token-superposition as Mac-native until a real MLX or Metal path exists.",
            "routing": "Scheduler should route CUDA-only hot loops to Windows/NVIDIA and MLX chunks to Apple Silicon.",
        },
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = ROOT / args.markdown_out
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report.get("state") != "RED" else 2


def task_row(row: dict[str, str], policy: dict[str, Any], worker_text: str, work_proof: dict[str, Any]) -> dict[str, Any]:
    allowed = set(get_path(policy, ["worker_chunks", "allowed_task_kinds"], []))
    # Some task allowlists live in registered task policy instead of worker_chunks.
    allowed.update(str(item) for item in get_path(policy, ["tasks", "allowed_kinds"], []) if item)
    mlx = row["mlx"]
    implemented = mlx in allowed or mlx in worker_text
    status = row["status"]
    if row["status"] == "implemented" and not implemented:
        status = "missing_policy_or_runner"
    route = {
        "scheduler_route": "apple_silicon_mlx" if status == "implemented" else "do_not_route_until_implemented",
        "parity_class": "registered_hive_worker_chunk" if status == "implemented" else "missing_worker_chunk",
        "cuda_equivalent_route": "windows_nvidia_cuda",
    }
    return {**row, "status": status, "policy_or_runner_present": implemented, "latest_evidence": worker_evidence(row["mlx"], work_proof), "route": route}


def cli_row(row: dict[str, Any], cli_text: str, work_proof: dict[str, Any]) -> dict[str, Any]:
    cuda_present = command_present(row["cuda"], cli_text)
    mac = row.get("mac") or ""
    mac_present = bool(mac and command_present(mac, cli_text))
    evidence = cli_evidence(mac, row["cuda"], work_proof)
    status = str(row.get("status") or "")
    bridge_ready = bool(mac_present and evidence.get("ok") and status.startswith("implemented_python_mlx_bridge"))
    kernel_status = str(row.get("kernel_port_status") or "")
    kernel_pending = kernel_status == PENDING_KERNEL_STATUS
    kernel_parity_ready = kernel_status in NATIVE_KERNEL_READY_STATUSES
    guarded_proof_ready = kernel_status == GUARDED_METAL_PROOF_STATUS
    guarded_canary_ready = kernel_status == GUARDED_METAL_STATUS
    production_route_ready = kernel_status in FULL_PARITY_READY_STATUSES and bool(
        row.get("production_scheduler_routing_enabled", True)
    )
    cuda_hot_loop_route_required = bool(kernel_pending or (kernel_parity_ready and not production_route_ready))
    if guarded_canary_ready:
        operator_label = (
            "Rust/Metal guarded canary proof is ready; production routing and full parity remain locked."
        )
    elif guarded_proof_ready:
        operator_label = (
            "Rust/Metal guarded proof is ready; production routing and full parity remain locked."
        )
    elif kernel_pending:
        operator_label = "MLX bridge runnable; CUDA-equivalent hot-loop parity still routes to Windows/NVIDIA."
    elif production_route_ready:
        operator_label = "Mac native kernel parity ready for production routing."
    else:
        operator_label = "Mac native kernel proof exists, but production routing is still locked."
    route = {
        "bridge_ready": bridge_ready,
        "bridge_route": row.get("bridge_route") or ("apple_silicon_mlx_bounded_work" if bridge_ready else "do_not_route_until_evidence_refresh"),
        "kernel_parity_ready": kernel_parity_ready,
        "guarded_native_kernel_proof_ready": guarded_proof_ready,
        "guarded_native_kernel_canary_ready": guarded_canary_ready,
        "production_route_ready": production_route_ready,
        "kernel_route_until_ported": row.get("kernel_route_until_ported") or ("windows_nvidia_cuda" if kernel_pending else "apple_silicon_native_kernel"),
        "cuda_hot_loop_route_required": cuda_hot_loop_route_required,
        "operator_label": operator_label,
    }
    return {
        **row,
        "cuda_present": cuda_present,
        "mac_present": mac_present,
        "bridge_ready": bridge_ready,
        "kernel_parity_ready": route["kernel_parity_ready"],
        "guarded_native_kernel_proof_ready": route["guarded_native_kernel_proof_ready"],
        "guarded_native_kernel_canary_ready": route["guarded_native_kernel_canary_ready"],
        "production_route_ready": route["production_route_ready"],
        "cuda_hot_loop_route_required": route["cuda_hot_loop_route_required"],
        "route": route,
        "latest_evidence": evidence,
    }


def routing_decisions(task_rows: list[dict[str, Any]], cli_rows: list[dict[str, Any]], mlx: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": "project_theseus_macos_mlx_routing_truth_v0",
        "mlx_available": bool(mlx.get("available")),
        "worker_chunks": [
            {
                "cuda": row.get("cuda"),
                "mac": row.get("mlx"),
                "scheduler_route": get_path(row, ["route", "scheduler_route"], ""),
                "latest_evidence_ok": get_path(row, ["latest_evidence", "ok"], False),
            }
            for row in task_rows
        ],
        "cli_bridges": [
            {
                "cuda": row.get("cuda"),
                "mac": row.get("mac"),
                "bridge_ready": row.get("bridge_ready"),
                "bridge_route": get_path(row, ["route", "bridge_route"], ""),
                "kernel_parity_ready": row.get("kernel_parity_ready"),
                "guarded_native_kernel_proof_ready": row.get("guarded_native_kernel_proof_ready"),
                "guarded_native_kernel_canary_ready": row.get("guarded_native_kernel_canary_ready"),
                "production_route_ready": row.get("production_route_ready"),
                "kernel_route_until_ported": get_path(row, ["route", "kernel_route_until_ported"], ""),
                "cuda_hot_loop_route_required": row.get("cuda_hot_loop_route_required"),
                "operator_label": get_path(row, ["route", "operator_label"], ""),
            }
            for row in cli_rows
        ],
        "default_training_route": "Apple Silicon receives registered MLX chunks and bounded MLX CLI bridges; guarded Metal canaries are proof-only until a production route gate opens. CUDA-equivalent hot loops still require Windows/NVIDIA or an explicitly production-ready Mac native kernel.",
        "safe_claim": "A runnable Python MLX bridge is useful Mac-native work evidence, but it is not final Rust/Metal hot-loop parity.",
    }


def native_hot_loop_proof(
    cli_rows: list[dict[str, Any]],
    cli_text: str,
    worker_text: str,
    work_proof: dict[str, Any],
) -> dict[str, Any]:
    metal_rollout_proof = read_json(METAL_ROLLOUT_PROOF, {})
    metal_rollout_cli_proof = read_json(METAL_ROLLOUT_CLI_PROOF, {})
    metal_rollout_feature_proof = read_json(METAL_ROLLOUT_FEATURE_PROOF, {})
    metal_rollout_readout_proof = read_json(METAL_ROLLOUT_READOUT_PROOF, {})
    metal_rollout_readout_training_proof = read_json(METAL_ROLLOUT_READOUT_TRAINING_PROOF, {})
    metal_train_path_proof = read_json(METAL_TRAIN_PATH_PROOF, {})
    metal_state_training_proof = read_json(METAL_STATE_TRAINING_PROOF, {})
    metal_token_superposition_readout_proof = read_json(METAL_TOKEN_SUPERPOSITION_READOUT_PROOF, {})
    metal_token_superposition_report = read_json(METAL_TOKEN_SUPERPOSITION_REPORT, {})
    metal_token_superposition_route_policy = read_json(METAL_TOKEN_SUPERPOSITION_ROUTE_POLICY, {})
    metal_token_superposition_ladder_report = read_json(METAL_TOKEN_SUPERPOSITION_LADDER, {})
    metal_token_superposition_scheduler_canary_report = read_json(METAL_TOKEN_SUPERPOSITION_SCHEDULER_CANARY, {})
    metal_train_standalone_report = read_json(METAL_TRAIN_STANDALONE_REPORT, {})
    metal_train_standalone_scheduler_canary_report = read_json(METAL_STANDALONE_SCHEDULER_CANARY, {})
    metal_train_rollout_report = read_json(METAL_TRAIN_ROLLOUT_REPORT, {})
    metal_train_rollout_sweep_report = read_json(METAL_TRAIN_ROLLOUT_SWEEP_REPORT, {})
    metal_train_rollout_sweep_scheduler_canary_report = read_json(METAL_ROLLOUT_SWEEP_SCHEDULER_CANARY, {})
    metal_train_rollout_sweep_route_policy = read_json(METAL_ROLLOUT_SWEEP_ROUTE_POLICY, {})
    metal_production_route_readiness_report = read_json(METAL_PRODUCTION_ROUTE_READINESS, {})
    metal_scheduler_dry_run = read_json(METAL_SCHEDULER_DRY_RUN, {})
    metal_parity_ladder_report = read_json(METAL_PARITY_LADDER, {})
    metal_scheduler_canary_report = read_json(METAL_SCHEDULER_CANARY, {})
    metal_rollout_ok = bool(metal_rollout_proof.get("ok") and metal_rollout_proof.get("state") == "GREEN")
    metal_rollout_cli_ok = bool(
        metal_rollout_cli_proof.get("ok") and metal_rollout_cli_proof.get("state") == "GREEN"
    )
    metal_rollout_feature_ok = bool(
        metal_rollout_feature_proof.get("ok") and metal_rollout_feature_proof.get("state") == "GREEN"
    )
    metal_rollout_readout_ok = bool(
        metal_rollout_readout_proof.get("ok") and metal_rollout_readout_proof.get("state") == "GREEN"
    )
    metal_rollout_readout_training_ok = bool(
        metal_rollout_readout_training_proof.get("ok")
        and metal_rollout_readout_training_proof.get("state") == "GREEN"
    )
    metal_train_path_ok = bool(metal_train_path_proof.get("ok") and metal_train_path_proof.get("state") == "GREEN")
    metal_train_rollout_contract_ok = bool(
        metal_train_rollout_report.get("ok")
        and metal_train_rollout_report.get("command") == "train-rollout-metal"
        and get_path(metal_train_rollout_report, ["report_contract", "matches_train_rollout_cli_surface"], False)
    )
    metal_train_rollout_artifact_equivalence = validate_train_rollout_metal_artifact(metal_train_rollout_report)
    metal_train_rollout_artifact_ok = bool(metal_train_rollout_artifact_equivalence.get("ok"))
    metal_train_rollout_route_guard = validate_train_rollout_metal_route_policy(
        metal_train_rollout_report,
        metal_train_rollout_artifact_equivalence,
    )
    metal_train_rollout_route_guard_ok = bool(metal_train_rollout_route_guard.get("ok"))
    metal_train_rollout_scheduler_dry_run = validate_train_rollout_metal_scheduler_dry_run(metal_scheduler_dry_run)
    metal_train_rollout_scheduler_dry_run_ok = bool(metal_train_rollout_scheduler_dry_run.get("ok"))
    metal_train_rollout_parity_ladder = validate_train_rollout_metal_parity_ladder(metal_parity_ladder_report)
    metal_train_rollout_parity_ladder_ok = bool(metal_train_rollout_parity_ladder.get("ok"))
    metal_train_rollout_scheduler_canary = validate_train_rollout_metal_scheduler_canary(metal_scheduler_canary_report)
    metal_train_rollout_scheduler_canary_ok = bool(metal_train_rollout_scheduler_canary.get("ok"))
    metal_train_rollout_sweep = validate_train_rollout_metal_sweep(metal_train_rollout_sweep_report)
    metal_train_rollout_sweep_ok = bool(metal_train_rollout_sweep.get("ok"))
    metal_state_training = validate_rollout_metal_state_training_proof(metal_state_training_proof)
    metal_state_training_ok = bool(metal_state_training.get("ok"))
    metal_train_rollout_sweep_scheduler_canary = validate_train_rollout_metal_sweep_scheduler_canary(
        metal_train_rollout_sweep_scheduler_canary_report,
        metal_train_rollout_sweep_route_policy,
    )
    metal_train_rollout_sweep_scheduler_canary_ok = bool(metal_train_rollout_sweep_scheduler_canary.get("ok"))
    metal_production_route_readiness = validate_metal_production_route_readiness(
        metal_production_route_readiness_report
    )
    metal_production_route_readiness_ok = bool(metal_production_route_readiness.get("ok"))
    metal_train_standalone_contract_ok = bool(
        metal_train_standalone_report.get("ok")
        and metal_train_standalone_report.get("state") == "GREEN"
        and metal_train_standalone_report.get("command") == "train-standalone-metal"
        and get_path(metal_train_standalone_report, ["report_contract", "matches_train_standalone_cli_surface"], False)
    )
    metal_train_standalone_artifact_equivalence = validate_train_standalone_metal_artifact(
        metal_train_standalone_report
    )
    metal_train_standalone_artifact_ok = bool(metal_train_standalone_artifact_equivalence.get("ok"))
    metal_train_standalone_scheduler_canary = validate_train_standalone_metal_scheduler_canary(
        metal_train_standalone_scheduler_canary_report
    )
    metal_train_standalone_scheduler_canary_ok = bool(metal_train_standalone_scheduler_canary.get("ok"))
    metal_token_superposition_readout = validate_token_superposition_metal_readout_proof(
        metal_token_superposition_readout_proof
    )
    metal_token_superposition_readout_ok = bool(metal_token_superposition_readout.get("ok"))
    metal_token_superposition_contract = validate_train_token_superposition_metal_contract(
        metal_token_superposition_report
    )
    metal_token_superposition_contract_ok = bool(metal_token_superposition_contract.get("ok"))
    metal_token_superposition_artifact_equivalence = validate_train_token_superposition_metal_artifact(
        metal_token_superposition_report
    )
    metal_token_superposition_artifact_ok = bool(metal_token_superposition_artifact_equivalence.get("ok"))
    metal_token_superposition_ladder = validate_train_token_superposition_metal_ladder(
        metal_token_superposition_ladder_report,
        metal_token_superposition_route_policy,
        metal_token_superposition_contract,
    )
    metal_token_superposition_ladder_ok = bool(metal_token_superposition_ladder.get("ok"))
    metal_token_superposition_scheduler_canary = validate_train_token_superposition_metal_scheduler_canary(
        metal_token_superposition_scheduler_canary_report,
        metal_token_superposition_route_policy,
    )
    metal_token_superposition_scheduler_canary_ok = bool(metal_token_superposition_scheduler_canary.get("ok"))
    cli_proof_command_present = command_present("rollout-metal-proof", cli_text)
    feature_proof_command_present = command_present("rollout-metal-feature-proof", cli_text)
    readout_proof_command_present = command_present("rollout-metal-readout-proof", cli_text)
    readout_training_proof_command_present = command_present("rollout-metal-readout-training-proof", cli_text)
    train_path_proof_command_present = command_present("rollout-metal-train-path-proof", cli_text)
    state_training_proof_command_present = command_present("rollout-metal-state-training-proof", cli_text)
    token_superposition_readout_proof_command_present = command_present(
        "token-superposition-metal-readout-proof", cli_text
    )
    train_token_superposition_metal_command_present = command_present(
        "train-token-superposition-metal", cli_text
    )
    train_standalone_metal_command_present = command_present("train-standalone-metal", cli_text)
    train_rollout_metal_command_present = command_present("train-rollout-metal", cli_text)
    train_rollout_metal_sweep_command_present = command_present("train-rollout-metal-sweep", cli_text)
    rust_metal_files = sorted(
        rel(path)
        for path in list((ROOT / "crates").glob("**/*metal*")) + list((ROOT / "crates").glob("**/*Metal*"))
        if path.is_file()
    )
    rust_mlx_files = sorted(
        rel(path)
        for path in list((ROOT / "crates").glob("**/*mlx*")) + list((ROOT / "crates").glob("**/*MLX*"))
        if path.is_file()
    )
    cuda_kernel_files = sorted(rel(path) for path in (ROOT / "crates" / "symliquid-cuda" / "kernels").glob("*.cu"))
    native_surface_present = bool(rust_metal_files or rust_mlx_files)
    pending_rows = [row for row in cli_rows if row.get("kernel_port_status") == PENDING_KERNEL_STATUS]
    ready_rows = [row for row in cli_rows if row.get("kernel_parity_ready")]
    guarded_proof_ready_rows = [row for row in cli_rows if row.get("kernel_port_status") == GUARDED_METAL_PROOF_STATUS]
    guarded_ready_rows = [row for row in cli_rows if row.get("kernel_port_status") == GUARDED_METAL_STATUS]
    production_route_pending_rows = [
        row for row in ready_rows if not row.get("production_route_ready")
    ]
    bridge_rows = [row for row in cli_rows if row.get("bridge_ready")]
    target_rows = []
    for row in cli_rows:
        evidence = row.get("latest_evidence") if isinstance(row.get("latest_evidence"), dict) else {}
        subkernel_proof = {}
        if row.get("cuda") == "train-standalone-cuda" and metal_train_standalone_report:
            subkernel_proof = {
                "path": rel(METAL_TRAIN_STANDALONE_REPORT),
                "command_present": train_standalone_metal_command_present,
                "ok": metal_train_standalone_contract_ok,
                "state": metal_train_standalone_report.get("state"),
                "command": metal_train_standalone_report.get("command"),
                "policy": metal_train_standalone_report.get("policy"),
                "backend": metal_train_standalone_report.get("backend"),
                "implementation": metal_train_standalone_report.get("implementation"),
                "feature_set": metal_train_standalone_report.get("feature_set"),
                "hv_dim": metal_train_standalone_report.get("hv_dim"),
                "labels": metal_train_standalone_report.get("labels"),
                "kernel_launches": metal_train_standalone_report.get("kernel_launches"),
                "metrics": metal_train_standalone_report.get("metrics")
                if isinstance(metal_train_standalone_report.get("metrics"), dict)
                else {},
                "runtime_profile": metal_train_standalone_report.get("runtime_profile")
                if isinstance(metal_train_standalone_report.get("runtime_profile"), dict)
                else {},
                "work_receipt": metal_train_standalone_report.get("work_receipt")
                if isinstance(metal_train_standalone_report.get("work_receipt"), dict)
                else {},
                "report_contract": metal_train_standalone_report.get("report_contract")
                if isinstance(metal_train_standalone_report.get("report_contract"), dict)
                else {},
                "artifact_equivalence": metal_train_standalone_artifact_equivalence,
                "python_mlx_bridge_used": bool(
                    get_path(metal_train_standalone_report, ["report_contract", "python_mlx_bridge_used"], True)
                ),
                "train_standalone_parity_claim_allowed": bool(
                    metal_train_standalone_report.get("train_standalone_parity_claim_allowed")
                ),
                "full_cli_parity_claim_allowed": bool(
                    metal_train_standalone_report.get("full_cli_parity_claim_allowed")
                ),
                "model_promotion_allowed": bool(metal_train_standalone_report.get("model_promotion_allowed")),
                "scheduler_canary": metal_train_standalone_scheduler_canary,
            }
        elif row.get("cuda") == "train-rollout-cuda" and metal_rollout_proof:
            subkernel_proof = {
                "path": rel(METAL_ROLLOUT_PROOF),
                "ok": metal_rollout_ok,
                "state": metal_rollout_proof.get("state"),
                "native_hot_loop": metal_rollout_proof.get("native_hot_loop"),
                "parity_scope": metal_rollout_proof.get("parity_scope"),
                "max_abs_delta": metal_rollout_proof.get("max_abs_delta"),
                "tolerance": metal_rollout_proof.get("tolerance"),
                "full_cli_parity_claim_allowed": bool(metal_rollout_proof.get("full_cli_parity_claim_allowed")),
                "main_cli_proof": {
                    "path": rel(METAL_ROLLOUT_CLI_PROOF),
                    "command_present": cli_proof_command_present,
                    "ok": metal_rollout_cli_ok,
                    "state": metal_rollout_cli_proof.get("state"),
                    "max_abs_delta": metal_rollout_cli_proof.get("max_abs_delta"),
                    "tolerance": metal_rollout_cli_proof.get("tolerance"),
                    "full_cli_parity_claim_allowed": bool(metal_rollout_cli_proof.get("full_cli_parity_claim_allowed")),
                },
                "feature_path_proof": {
                    "path": rel(METAL_ROLLOUT_FEATURE_PROOF),
                    "command_present": feature_proof_command_present,
                    "ok": metal_rollout_feature_ok,
                    "state": metal_rollout_feature_proof.get("state"),
                    "native_path": metal_rollout_feature_proof.get("native_path"),
                    "parity_scope": metal_rollout_feature_proof.get("parity_scope"),
                    "max_abs_delta": metal_rollout_feature_proof.get("max_abs_delta"),
                    "tolerance": metal_rollout_feature_proof.get("tolerance"),
                    "feature_tensor": metal_rollout_feature_proof.get("feature_tensor"),
                    "train_rollout_parity_claim_allowed": bool(metal_rollout_feature_proof.get("train_rollout_parity_claim_allowed")),
                    "full_cli_parity_claim_allowed": bool(metal_rollout_feature_proof.get("full_cli_parity_claim_allowed")),
                },
                "readout_path_proof": {
                    "path": rel(METAL_ROLLOUT_READOUT_PROOF),
                    "command_present": readout_proof_command_present,
                    "ok": metal_rollout_readout_ok,
                    "state": metal_rollout_readout_proof.get("state"),
                    "native_path": metal_rollout_readout_proof.get("native_path"),
                    "native_readout": metal_rollout_readout_proof.get("native_readout"),
                    "parity_scope": metal_rollout_readout_proof.get("parity_scope"),
                    "feature_max_abs_delta": metal_rollout_readout_proof.get("feature_max_abs_delta"),
                    "logits_max_abs_delta": metal_rollout_readout_proof.get("logits_max_abs_delta"),
                    "prediction_agreement": metal_rollout_readout_proof.get("prediction_agreement"),
                    "train_rollout_parity_claim_allowed": bool(metal_rollout_readout_proof.get("train_rollout_parity_claim_allowed")),
                    "full_cli_parity_claim_allowed": bool(metal_rollout_readout_proof.get("full_cli_parity_claim_allowed")),
                },
                "readout_training_path_proof": {
                    "path": rel(METAL_ROLLOUT_READOUT_TRAINING_PROOF),
                    "command_present": readout_training_proof_command_present,
                    "ok": metal_rollout_readout_training_ok,
                    "state": metal_rollout_readout_training_proof.get("state"),
                    "native_path": metal_rollout_readout_training_proof.get("native_path"),
                    "native_trainer": metal_rollout_readout_training_proof.get("native_trainer"),
                    "native_readout": metal_rollout_readout_training_proof.get("native_readout"),
                    "parity_scope": metal_rollout_readout_training_proof.get("parity_scope"),
                    "feature_max_abs_delta": metal_rollout_readout_training_proof.get("feature_max_abs_delta"),
                    "weight_max_abs_delta": metal_rollout_readout_training_proof.get("weight_max_abs_delta"),
                    "bias_max_abs_delta": metal_rollout_readout_training_proof.get("bias_max_abs_delta"),
                    "logits_max_abs_delta": metal_rollout_readout_training_proof.get("logits_max_abs_delta"),
                    "prediction_agreement": metal_rollout_readout_training_proof.get("prediction_agreement"),
                    "readout_training_subpath_proof": bool(metal_rollout_readout_training_proof.get("readout_training_subpath_proof")),
                    "train_rollout_parity_claim_allowed": bool(metal_rollout_readout_training_proof.get("train_rollout_parity_claim_allowed")),
                    "full_cli_parity_claim_allowed": bool(metal_rollout_readout_training_proof.get("full_cli_parity_claim_allowed")),
                },
                "train_path_proof": {
                    "path": rel(METAL_TRAIN_PATH_PROOF),
                    "command_present": train_path_proof_command_present,
                    "ok": metal_train_path_ok,
                    "state": metal_train_path_proof.get("state"),
                    "native_path": metal_train_path_proof.get("native_path"),
                    "native_trainer": metal_train_path_proof.get("native_trainer"),
                    "native_readout": metal_train_path_proof.get("native_readout"),
                    "parity_scope": metal_train_path_proof.get("parity_scope"),
                    "train_rows": metal_train_path_proof.get("train_rows"),
                    "eval_rows": metal_train_path_proof.get("eval_rows"),
                    "kernel_launches": metal_train_path_proof.get("kernel_launches"),
                    "parity_metrics": metal_train_path_proof.get("parity_metrics") if isinstance(metal_train_path_proof.get("parity_metrics"), dict) else {},
                    "train_metrics": metal_train_path_proof.get("train_metrics") if isinstance(metal_train_path_proof.get("train_metrics"), dict) else {},
                    "eval_metrics": metal_train_path_proof.get("eval_metrics") if isinstance(metal_train_path_proof.get("eval_metrics"), dict) else {},
                    "symbolic_fallback": bool(metal_train_path_proof.get("symbolic_fallback")),
                    "python_mlx_bridge_used": bool(get_path(metal_train_path_proof, ["runtime_profile", "python_mlx_bridge_used"], True)),
                    "train_rollout_parity_claim_allowed": bool(metal_train_path_proof.get("train_rollout_parity_claim_allowed")),
                    "full_cli_parity_claim_allowed": bool(metal_train_path_proof.get("full_cli_parity_claim_allowed")),
                },
                "train_rollout_metal_cli_contract": {
                    "path": rel(METAL_TRAIN_ROLLOUT_REPORT),
                    "command_present": train_rollout_metal_command_present,
                    "ok": metal_train_rollout_contract_ok,
                    "state": metal_train_rollout_report.get("state"),
                    "command": metal_train_rollout_report.get("command"),
                    "policy": metal_train_rollout_report.get("policy"),
                    "backend": metal_train_rollout_report.get("backend"),
                    "implementation": metal_train_rollout_report.get("implementation"),
                    "native_path": metal_train_rollout_report.get("native_path"),
                    "parity_scope": metal_train_rollout_report.get("parity_scope"),
                    "metrics": metal_train_rollout_report.get("metrics") if isinstance(metal_train_rollout_report.get("metrics"), dict) else {},
                    "work_receipt": metal_train_rollout_report.get("work_receipt") if isinstance(metal_train_rollout_report.get("work_receipt"), dict) else {},
                    "report_contract": metal_train_rollout_report.get("report_contract") if isinstance(metal_train_rollout_report.get("report_contract"), dict) else {},
                    "artifact_write": metal_train_rollout_report.get("artifact_write") if isinstance(metal_train_rollout_report.get("artifact_write"), dict) else {},
                    "artifact_equivalence": metal_train_rollout_artifact_equivalence,
                    "scheduler_route_guardrail": metal_train_rollout_route_guard,
                    "scheduler_dry_run": metal_train_rollout_scheduler_dry_run,
                    "bounded_parity_ladder": metal_train_rollout_parity_ladder,
                    "scheduler_canary": metal_train_rollout_scheduler_canary,
                    "python_mlx_bridge_used": bool(get_path(metal_train_rollout_report, ["runtime_profile", "python_mlx_bridge_used"], True)),
                    "train_rollout_parity_claim_allowed": bool(metal_train_rollout_report.get("train_rollout_parity_claim_allowed")),
                    "full_cli_parity_claim_allowed": bool(metal_train_rollout_report.get("full_cli_parity_claim_allowed")),
                    "model_promotion_allowed": bool(metal_train_rollout_report.get("model_promotion_allowed")),
                },
            }
        elif row.get("cuda") == "train-rollout-cuda-sweep" and metal_train_rollout_sweep_report:
            subkernel_proof = {
                "path": rel(METAL_TRAIN_ROLLOUT_SWEEP_REPORT),
                "command_present": train_rollout_metal_sweep_command_present,
                "ok": metal_train_rollout_sweep_ok,
                "state": metal_train_rollout_sweep_report.get("state"),
                "policy": metal_train_rollout_sweep_report.get("policy"),
                "command": metal_train_rollout_sweep_report.get("command"),
                "backend": metal_train_rollout_sweep_report.get("backend"),
                "implementation": metal_train_rollout_sweep_report.get("implementation"),
                "parity_for": metal_train_rollout_sweep_report.get("parity_for"),
                "score_semantics": metal_train_rollout_sweep_report.get("score_semantics"),
                "summary": metal_train_rollout_sweep_report.get("summary")
                if isinstance(metal_train_rollout_sweep_report.get("summary"), dict)
                else {},
                "metrics": metal_train_rollout_sweep_report.get("metrics")
                if isinstance(metal_train_rollout_sweep_report.get("metrics"), dict)
                else {},
                "report_contract": metal_train_rollout_sweep_report.get("report_contract")
                if isinstance(metal_train_rollout_sweep_report.get("report_contract"), dict)
                else {},
                "work_receipt": metal_train_rollout_sweep_report.get("work_receipt")
                if isinstance(metal_train_rollout_sweep_report.get("work_receipt"), dict)
                else {},
                "guardrail_validation": metal_train_rollout_sweep,
                "python_mlx_bridge_used": bool(
                    get_path(metal_train_rollout_sweep_report, ["report_contract", "python_mlx_bridge_used"], True)
                ),
                "state_training_native_ported": bool(
                    get_path(metal_train_rollout_sweep_report, ["summary", "state_training_native_ported"], True)
                ),
                "state_training_proof": {
                    "path": rel(METAL_STATE_TRAINING_PROOF),
                    "command_present": state_training_proof_command_present,
                    "ok": metal_state_training_ok,
                    "state": metal_state_training_proof.get("state"),
                    "native_path": metal_state_training_proof.get("native_path"),
                    "native_state_trainers": metal_state_training_proof.get("native_state_trainers")
                    if isinstance(metal_state_training_proof.get("native_state_trainers"), list)
                    else [],
                    "parity_scope": metal_state_training_proof.get("parity_scope"),
                    "state_training_native_ported": metal_state_training_proof.get("state_training_native_ported"),
                    "state_training_semantics_proof": metal_state_training_proof.get("state_training_semantics_proof"),
                    "kernel_launches": metal_state_training_proof.get("kernel_launches"),
                    "metrics": metal_state_training_proof.get("metrics")
                    if isinstance(metal_state_training_proof.get("metrics"), dict)
                    else {},
                    "guardrail_validation": metal_state_training,
                    "cuda_state_training_parity_claim_allowed": bool(
                        metal_state_training_proof.get("cuda_state_training_parity_claim_allowed")
                    ),
                    "train_rollout_sweep_parity_claim_allowed": bool(
                        metal_state_training_proof.get("train_rollout_sweep_parity_claim_allowed")
                    ),
                    "full_cli_parity_claim_allowed": bool(
                        metal_state_training_proof.get("full_cli_parity_claim_allowed")
                    ),
                    "model_promotion_allowed": bool(metal_state_training_proof.get("model_promotion_allowed")),
                },
                "train_rollout_sweep_parity_claim_allowed": bool(
                    metal_train_rollout_sweep_report.get("train_rollout_sweep_parity_claim_allowed")
                ),
                "full_cli_parity_claim_allowed": bool(
                    metal_train_rollout_sweep_report.get("full_cli_parity_claim_allowed")
                ),
                "model_promotion_allowed": bool(metal_train_rollout_sweep_report.get("model_promotion_allowed")),
                "scheduler_canary": metal_train_rollout_sweep_scheduler_canary,
            }
        elif row.get("cuda") == "train-token-superposition-cuda" and metal_token_superposition_readout_proof:
            subkernel_proof = {
                "path": rel(METAL_TOKEN_SUPERPOSITION_READOUT_PROOF),
                "command_present": token_superposition_readout_proof_command_present,
                "ok": metal_token_superposition_readout_ok,
                "state": metal_token_superposition_readout_proof.get("state"),
                "native_path": metal_token_superposition_readout_proof.get("native_path"),
                "native_bag_trainer": metal_token_superposition_readout_proof.get("native_bag_trainer"),
                "native_ar_trainer": metal_token_superposition_readout_proof.get("native_ar_trainer"),
                "native_readout": metal_token_superposition_readout_proof.get("native_readout"),
                "parity_scope": metal_token_superposition_readout_proof.get("parity_scope"),
                "parity_metrics": metal_token_superposition_readout_proof.get("parity_metrics")
                if isinstance(metal_token_superposition_readout_proof.get("parity_metrics"), dict)
                else {},
                "dataset": metal_token_superposition_readout_proof.get("dataset")
                if isinstance(metal_token_superposition_readout_proof.get("dataset"), dict)
                else {},
                "variant": metal_token_superposition_readout_proof.get("variant")
                if isinstance(metal_token_superposition_readout_proof.get("variant"), dict)
                else {},
                "guardrail_validation": metal_token_superposition_readout,
                "python_mlx_bridge_used": False,
                "train_token_superposition_parity_claim_allowed": bool(
                    metal_token_superposition_readout_proof.get("train_token_superposition_parity_claim_allowed")
                ),
                "full_cli_parity_claim_allowed": bool(
                    metal_token_superposition_readout_proof.get("full_cli_parity_claim_allowed")
                ),
                "train_token_superposition_metal_cli_contract": {
                    "path": rel(METAL_TOKEN_SUPERPOSITION_REPORT),
                    "command_present": train_token_superposition_metal_command_present,
                    "ok": metal_token_superposition_contract_ok,
                    "state": metal_token_superposition_report.get("state"),
                    "policy": metal_token_superposition_report.get("policy"),
                    "command": metal_token_superposition_report.get("command"),
                    "backend": metal_token_superposition_report.get("backend"),
                    "implementation": metal_token_superposition_report.get("implementation"),
                    "metrics": metal_token_superposition_report.get("metrics")
                    if isinstance(metal_token_superposition_report.get("metrics"), dict)
                    else {},
                    "work_receipt": metal_token_superposition_report.get("work_receipt")
                    if isinstance(metal_token_superposition_report.get("work_receipt"), dict)
                    else {},
                    "report_contract": metal_token_superposition_report.get("report_contract")
                    if isinstance(metal_token_superposition_report.get("report_contract"), dict)
                    else {},
                    "guardrail_validation": metal_token_superposition_contract,
                    "python_mlx_bridge_used": bool(
                        get_path(metal_token_superposition_report, ["report_contract", "python_mlx_bridge_used"], True)
                    ),
                    "train_token_superposition_parity_claim_allowed": bool(
                        metal_token_superposition_report.get("train_token_superposition_parity_claim_allowed")
                    ),
                    "full_cli_parity_claim_allowed": bool(
                        metal_token_superposition_report.get("full_cli_parity_claim_allowed")
                    ),
                    "model_promotion_allowed": bool(
                        metal_token_superposition_report.get("model_promotion_allowed")
                    ),
                },
                "artifact_equivalence": metal_token_superposition_artifact_equivalence,
                "bounded_token_superposition_ladder": metal_token_superposition_ladder,
                "scheduler_canary": metal_token_superposition_scheduler_canary,
            }
        target_rows.append(
            {
                "cuda_command": row.get("cuda"),
                "mac_command": row.get("mac"),
                "bridge_ready": bool(row.get("bridge_ready")),
                "bridge_implementation": row.get("status"),
                "latest_evidence_ok": bool(evidence.get("ok")),
                "latest_evidence_report": evidence.get("report_path"),
                "kernel_port_status": row.get("kernel_port_status"),
                "kernel_parity_ready": bool(row.get("kernel_parity_ready")),
                "guarded_native_kernel_proof_ready": bool(row.get("guarded_native_kernel_proof_ready")),
                "guarded_native_kernel_canary_ready": bool(row.get("guarded_native_kernel_canary_ready")),
                "production_route_ready": bool(row.get("production_route_ready")),
                "cuda_hot_loop_route_required": bool(row.get("cuda_hot_loop_route_required")),
                "metrics": evidence.get("metrics") if isinstance(evidence.get("metrics"), dict) else {},
                "native_subkernel_proof": subkernel_proof,
            }
        )
    parity_claim_allowed = bool(
        native_surface_present
        and cli_rows
        and len(ready_rows) == len(cli_rows)
        and not pending_rows
        and not production_route_pending_rows
    )
    return {
        "policy": "project_theseus_macos_native_hot_loop_parity_proof_v0",
        "summary": {
            "rust_metal_or_mlx_surface_present": native_surface_present,
            "rust_metal_source_files": len(rust_metal_files),
            "rust_mlx_source_files": len(rust_mlx_files),
            "cuda_kernel_source_files": len(cuda_kernel_files),
            "python_mlx_bridge_commands_ready": len(bridge_rows),
            "native_hot_loop_targets": len(cli_rows),
            "native_hot_loop_targets_ready": len(ready_rows),
            "native_hot_loop_targets_guarded_proof_ready": len(guarded_proof_ready_rows),
            "native_hot_loop_targets_guarded_canary_ready": len(guarded_ready_rows),
            "production_route_pending_native_hot_loop_targets": len(production_route_pending_rows),
            "native_subkernel_proof_count": 1 if metal_rollout_ok else 0,
            "native_cli_proof_count": 1 if cli_proof_command_present and metal_rollout_cli_ok else 0,
            "native_feature_proof_count": 1 if feature_proof_command_present and metal_rollout_feature_ok else 0,
            "native_readout_proof_count": 1 if readout_proof_command_present and metal_rollout_readout_ok else 0,
            "native_readout_training_proof_count": 1 if readout_training_proof_command_present and metal_rollout_readout_training_ok else 0,
            "native_train_path_proof_count": 1 if train_path_proof_command_present and metal_train_path_ok else 0,
            "native_train_standalone_cli_contract_count": 1
            if train_standalone_metal_command_present and metal_train_standalone_contract_ok
            else 0,
            "native_train_standalone_artifact_equivalence_count": 1
            if metal_train_standalone_artifact_ok
            else 0,
            "native_train_standalone_scheduler_canary_count": 1
            if metal_train_standalone_scheduler_canary_ok
            else 0,
            "native_train_rollout_cli_contract_count": 1 if train_rollout_metal_command_present and metal_train_rollout_contract_ok else 0,
            "native_train_rollout_artifact_equivalence_count": 1 if metal_train_rollout_artifact_ok else 0,
            "native_train_rollout_scheduler_guardrail_count": 1 if metal_train_rollout_route_guard_ok else 0,
            "native_train_rollout_scheduler_dry_run_count": 1 if metal_train_rollout_scheduler_dry_run_ok else 0,
            "native_train_rollout_parity_ladder_count": 1 if metal_train_rollout_parity_ladder_ok else 0,
            "native_train_rollout_scheduler_canary_count": 1 if metal_train_rollout_scheduler_canary_ok else 0,
            "native_train_rollout_sweep_cli_contract_count": 1
            if train_rollout_metal_sweep_command_present and metal_train_rollout_sweep_ok
            else 0,
            "native_train_rollout_sweep_child_artifact_count": int(
                metal_train_rollout_sweep.get("artifact_count") or 0
            )
            if metal_train_rollout_sweep_ok
            else 0,
            "native_train_rollout_sweep_scheduler_canary_count": 1
            if metal_train_rollout_sweep_scheduler_canary_ok
            else 0,
            "native_train_rollout_state_training_proof_count": 1
            if state_training_proof_command_present and metal_state_training_ok
            else 0,
            "native_token_superposition_readout_proof_count": 1
            if token_superposition_readout_proof_command_present and metal_token_superposition_readout_ok
            else 0,
            "native_train_token_superposition_cli_contract_count": 1
            if train_token_superposition_metal_command_present and metal_token_superposition_contract_ok
            else 0,
            "native_train_token_superposition_artifact_equivalence_count": 1
            if metal_token_superposition_artifact_ok
            else 0,
            "native_train_token_superposition_ladder_count": 1
            if train_token_superposition_metal_command_present and metal_token_superposition_ladder_ok
            else 0,
            "native_train_token_superposition_scheduler_canary_count": 1
            if metal_token_superposition_scheduler_canary_ok
            else 0,
            "native_metal_production_route_readiness_review_count": 1
            if metal_production_route_readiness_ok
            else 0,
            "native_metal_production_route_ready_count": int(
                metal_production_route_readiness.get("production_route_ready_count") or 0
            )
            if metal_production_route_readiness_ok
            else 0,
            "native_metal_production_route_blocker_count": int(
                metal_production_route_readiness.get("blocker_count") or 0
            )
            if metal_production_route_readiness_ok
            else 0,
            "pending_native_hot_loop_targets": len(pending_rows),
            "parity_claim_allowed": parity_claim_allowed,
            "no_fake_parity": True,
        },
        "parity_claim_allowed": parity_claim_allowed,
        "source_inventory": {
            "rust_metal_files": rust_metal_files,
            "rust_mlx_files": rust_mlx_files,
            "cuda_kernel_files": cuda_kernel_files,
            "python_mlx_bridge": "scripts/macos_mlx_training.py",
            "rust_cli_bridge": "crates/symliquid-cli/src/main.rs",
            "symliquid_cli_native_proof_command_present": cli_proof_command_present,
            "symliquid_cli_feature_proof_command_present": feature_proof_command_present,
            "symliquid_cli_readout_proof_command_present": readout_proof_command_present,
            "symliquid_cli_readout_training_proof_command_present": readout_training_proof_command_present,
            "symliquid_cli_train_path_proof_command_present": train_path_proof_command_present,
            "symliquid_cli_token_superposition_readout_proof_command_present": token_superposition_readout_proof_command_present,
            "symliquid_cli_train_standalone_metal_command_present": train_standalone_metal_command_present,
            "symliquid_cli_train_token_superposition_metal_command_present": train_token_superposition_metal_command_present,
            "symliquid_cli_train_rollout_metal_command_present": train_rollout_metal_command_present,
            "symliquid_cli_train_rollout_metal_sweep_command_present": train_rollout_metal_sweep_command_present,
            "cli_mentions_python_mlx_bridge": "scripts/macos_mlx_training.py" in cli_text,
            "worker_mentions_mlx_chunks": all(name in worker_text for name in ["mlx_babylm_eval", "mlx_babylm_train", "mlx_rollout_probe"]),
        },
        "proof_reports": [
            {
                "path": rel(METAL_ROLLOUT_PROOF),
                "present": bool(metal_rollout_proof),
                "ok": metal_rollout_ok,
                "state": metal_rollout_proof.get("state"),
                "native_hot_loop": metal_rollout_proof.get("native_hot_loop"),
                "parity_scope": metal_rollout_proof.get("parity_scope"),
                "max_abs_delta": metal_rollout_proof.get("max_abs_delta"),
                "tolerance": metal_rollout_proof.get("tolerance"),
                "full_cli_parity_claim_allowed": bool(metal_rollout_proof.get("full_cli_parity_claim_allowed")),
            },
            {
                "path": rel(METAL_ROLLOUT_CLI_PROOF),
                "present": bool(metal_rollout_cli_proof),
                "command_present": cli_proof_command_present,
                "ok": metal_rollout_cli_ok,
                "state": metal_rollout_cli_proof.get("state"),
                "native_hot_loop": metal_rollout_cli_proof.get("native_hot_loop"),
                "parity_scope": metal_rollout_cli_proof.get("parity_scope"),
                "max_abs_delta": metal_rollout_cli_proof.get("max_abs_delta"),
                "tolerance": metal_rollout_cli_proof.get("tolerance"),
                "full_cli_parity_claim_allowed": bool(metal_rollout_cli_proof.get("full_cli_parity_claim_allowed")),
            },
            {
                "path": rel(METAL_ROLLOUT_FEATURE_PROOF),
                "present": bool(metal_rollout_feature_proof),
                "command_present": feature_proof_command_present,
                "ok": metal_rollout_feature_ok,
                "state": metal_rollout_feature_proof.get("state"),
                "native_path": metal_rollout_feature_proof.get("native_path"),
                "native_hot_loop": metal_rollout_feature_proof.get("native_hot_loop"),
                "parity_scope": metal_rollout_feature_proof.get("parity_scope"),
                "max_abs_delta": metal_rollout_feature_proof.get("max_abs_delta"),
                "tolerance": metal_rollout_feature_proof.get("tolerance"),
                "feature_tensor": metal_rollout_feature_proof.get("feature_tensor"),
                "train_rollout_parity_claim_allowed": bool(metal_rollout_feature_proof.get("train_rollout_parity_claim_allowed")),
                "full_cli_parity_claim_allowed": bool(metal_rollout_feature_proof.get("full_cli_parity_claim_allowed")),
            },
            {
                "path": rel(METAL_ROLLOUT_READOUT_PROOF),
                "present": bool(metal_rollout_readout_proof),
                "command_present": readout_proof_command_present,
                "ok": metal_rollout_readout_ok,
                "state": metal_rollout_readout_proof.get("state"),
                "native_path": metal_rollout_readout_proof.get("native_path"),
                "native_readout": metal_rollout_readout_proof.get("native_readout"),
                "parity_scope": metal_rollout_readout_proof.get("parity_scope"),
                "feature_max_abs_delta": metal_rollout_readout_proof.get("feature_max_abs_delta"),
                "logits_max_abs_delta": metal_rollout_readout_proof.get("logits_max_abs_delta"),
                "prediction_agreement": metal_rollout_readout_proof.get("prediction_agreement"),
                "readout": metal_rollout_readout_proof.get("readout"),
                "train_rollout_parity_claim_allowed": bool(metal_rollout_readout_proof.get("train_rollout_parity_claim_allowed")),
                "full_cli_parity_claim_allowed": bool(metal_rollout_readout_proof.get("full_cli_parity_claim_allowed")),
            },
            {
                "path": rel(METAL_ROLLOUT_READOUT_TRAINING_PROOF),
                "present": bool(metal_rollout_readout_training_proof),
                "command_present": readout_training_proof_command_present,
                "ok": metal_rollout_readout_training_ok,
                "state": metal_rollout_readout_training_proof.get("state"),
                "native_path": metal_rollout_readout_training_proof.get("native_path"),
                "native_trainer": metal_rollout_readout_training_proof.get("native_trainer"),
                "native_readout": metal_rollout_readout_training_proof.get("native_readout"),
                "parity_scope": metal_rollout_readout_training_proof.get("parity_scope"),
                "feature_max_abs_delta": metal_rollout_readout_training_proof.get("feature_max_abs_delta"),
                "weight_max_abs_delta": metal_rollout_readout_training_proof.get("weight_max_abs_delta"),
                "bias_max_abs_delta": metal_rollout_readout_training_proof.get("bias_max_abs_delta"),
                "logits_max_abs_delta": metal_rollout_readout_training_proof.get("logits_max_abs_delta"),
                "prediction_agreement": metal_rollout_readout_training_proof.get("prediction_agreement"),
                "eval_loss_delta": metal_rollout_readout_training_proof.get("eval_loss_delta"),
                "readout_training_subpath_proof": bool(metal_rollout_readout_training_proof.get("readout_training_subpath_proof")),
                "train_rollout_parity_claim_allowed": bool(metal_rollout_readout_training_proof.get("train_rollout_parity_claim_allowed")),
                "full_cli_parity_claim_allowed": bool(metal_rollout_readout_training_proof.get("full_cli_parity_claim_allowed")),
            },
            {
                "path": rel(METAL_TRAIN_PATH_PROOF),
                "present": bool(metal_train_path_proof),
                "command_present": train_path_proof_command_present,
                "ok": metal_train_path_ok,
                "state": metal_train_path_proof.get("state"),
                "native_path": metal_train_path_proof.get("native_path"),
                "native_trainer": metal_train_path_proof.get("native_trainer"),
                "native_readout": metal_train_path_proof.get("native_readout"),
                "parity_scope": metal_train_path_proof.get("parity_scope"),
                "train_rows": metal_train_path_proof.get("train_rows"),
                "eval_rows": metal_train_path_proof.get("eval_rows"),
                "kernel_launches": metal_train_path_proof.get("kernel_launches"),
                "parity_metrics": metal_train_path_proof.get("parity_metrics") if isinstance(metal_train_path_proof.get("parity_metrics"), dict) else {},
                "train_metrics": metal_train_path_proof.get("train_metrics") if isinstance(metal_train_path_proof.get("train_metrics"), dict) else {},
                "eval_metrics": metal_train_path_proof.get("eval_metrics") if isinstance(metal_train_path_proof.get("eval_metrics"), dict) else {},
                "symbolic_fallback": bool(metal_train_path_proof.get("symbolic_fallback")),
                "python_mlx_bridge_used": bool(get_path(metal_train_path_proof, ["runtime_profile", "python_mlx_bridge_used"], True)),
                "train_rollout_parity_claim_allowed": bool(metal_train_path_proof.get("train_rollout_parity_claim_allowed")),
                "full_cli_parity_claim_allowed": bool(metal_train_path_proof.get("full_cli_parity_claim_allowed")),
            },
            {
                "path": rel(METAL_TRAIN_STANDALONE_REPORT),
                "present": bool(metal_train_standalone_report),
                "command_present": train_standalone_metal_command_present,
                "ok": metal_train_standalone_contract_ok,
                "state": metal_train_standalone_report.get("state"),
                "policy": metal_train_standalone_report.get("policy"),
                "command": metal_train_standalone_report.get("command"),
                "backend": metal_train_standalone_report.get("backend"),
                "implementation": metal_train_standalone_report.get("implementation"),
                "feature_set": metal_train_standalone_report.get("feature_set"),
                "metrics": metal_train_standalone_report.get("metrics")
                if isinstance(metal_train_standalone_report.get("metrics"), dict)
                else {},
                "report_contract": metal_train_standalone_report.get("report_contract")
                if isinstance(metal_train_standalone_report.get("report_contract"), dict)
                else {},
                "artifact_write": metal_train_standalone_report.get("artifact_write")
                if isinstance(metal_train_standalone_report.get("artifact_write"), dict)
                else {},
                "artifact_equivalence": metal_train_standalone_artifact_equivalence,
                "python_mlx_bridge_used": bool(
                    get_path(metal_train_standalone_report, ["report_contract", "python_mlx_bridge_used"], True)
                ),
                "train_standalone_parity_claim_allowed": bool(
                    metal_train_standalone_report.get("train_standalone_parity_claim_allowed")
                ),
                "full_cli_parity_claim_allowed": bool(
                    metal_train_standalone_report.get("full_cli_parity_claim_allowed")
                ),
                "model_promotion_allowed": bool(metal_train_standalone_report.get("model_promotion_allowed")),
                "scheduler_canary": metal_train_standalone_scheduler_canary,
            },
            {
                "path": rel(METAL_TRAIN_ROLLOUT_SWEEP_REPORT),
                "present": bool(metal_train_rollout_sweep_report),
                "command_present": train_rollout_metal_sweep_command_present,
                "ok": metal_train_rollout_sweep_ok,
                "state": metal_train_rollout_sweep_report.get("state"),
                "policy": metal_train_rollout_sweep_report.get("policy"),
                "command": metal_train_rollout_sweep_report.get("command"),
                "backend": metal_train_rollout_sweep_report.get("backend"),
                "implementation": metal_train_rollout_sweep_report.get("implementation"),
                "metrics": metal_train_rollout_sweep_report.get("metrics")
                if isinstance(metal_train_rollout_sweep_report.get("metrics"), dict)
                else {},
                "summary": metal_train_rollout_sweep_report.get("summary")
                if isinstance(metal_train_rollout_sweep_report.get("summary"), dict)
                else {},
                "report_contract": metal_train_rollout_sweep_report.get("report_contract")
                if isinstance(metal_train_rollout_sweep_report.get("report_contract"), dict)
                else {},
                "guardrail_validation": metal_train_rollout_sweep,
                "train_rollout_sweep_parity_claim_allowed": bool(
                    metal_train_rollout_sweep_report.get("train_rollout_sweep_parity_claim_allowed")
                ),
                "full_cli_parity_claim_allowed": bool(
                    metal_train_rollout_sweep_report.get("full_cli_parity_claim_allowed")
                ),
                "model_promotion_allowed": bool(metal_train_rollout_sweep_report.get("model_promotion_allowed")),
                "scheduler_canary": metal_train_rollout_sweep_scheduler_canary,
            },
            {
                "path": rel(METAL_TOKEN_SUPERPOSITION_READOUT_PROOF),
                "present": bool(metal_token_superposition_readout_proof),
                "command_present": token_superposition_readout_proof_command_present,
                "ok": metal_token_superposition_readout_ok,
                "state": metal_token_superposition_readout_proof.get("state"),
                "native_path": metal_token_superposition_readout_proof.get("native_path"),
                "native_bag_trainer": metal_token_superposition_readout_proof.get("native_bag_trainer"),
                "native_ar_trainer": metal_token_superposition_readout_proof.get("native_ar_trainer"),
                "native_readout": metal_token_superposition_readout_proof.get("native_readout"),
                "parity_scope": metal_token_superposition_readout_proof.get("parity_scope"),
                "parity_metrics": metal_token_superposition_readout_proof.get("parity_metrics")
                if isinstance(metal_token_superposition_readout_proof.get("parity_metrics"), dict)
                else {},
                "dataset": metal_token_superposition_readout_proof.get("dataset")
                if isinstance(metal_token_superposition_readout_proof.get("dataset"), dict)
                else {},
                "guardrail_validation": metal_token_superposition_readout,
                "train_token_superposition_parity_claim_allowed": bool(
                    metal_token_superposition_readout_proof.get("train_token_superposition_parity_claim_allowed")
                ),
                "full_cli_parity_claim_allowed": bool(
                    metal_token_superposition_readout_proof.get("full_cli_parity_claim_allowed")
                ),
            },
            {
                "path": rel(METAL_TOKEN_SUPERPOSITION_REPORT),
                "present": bool(metal_token_superposition_report),
                "command_present": train_token_superposition_metal_command_present,
                "ok": metal_token_superposition_contract_ok,
                "policy": metal_token_superposition_report.get("policy"),
                "command": metal_token_superposition_report.get("command"),
                "backend": metal_token_superposition_report.get("backend"),
                "implementation": metal_token_superposition_report.get("implementation"),
                "metrics": metal_token_superposition_report.get("metrics")
                if isinstance(metal_token_superposition_report.get("metrics"), dict)
                else {},
                "report_contract": metal_token_superposition_report.get("report_contract")
                if isinstance(metal_token_superposition_report.get("report_contract"), dict)
                else {},
                "artifact_write": metal_token_superposition_report.get("artifact_write")
                if isinstance(metal_token_superposition_report.get("artifact_write"), dict)
                else {},
                "artifact_equivalence": metal_token_superposition_artifact_equivalence,
                "guardrail_validation": metal_token_superposition_contract,
                "train_token_superposition_parity_claim_allowed": bool(
                    metal_token_superposition_report.get("train_token_superposition_parity_claim_allowed")
                ),
                "full_cli_parity_claim_allowed": bool(
                    metal_token_superposition_report.get("full_cli_parity_claim_allowed")
                ),
                "model_promotion_allowed": bool(metal_token_superposition_report.get("model_promotion_allowed")),
            },
            {
                "path": rel(METAL_TOKEN_SUPERPOSITION_LADDER),
                "present": bool(metal_token_superposition_ladder_report),
                "command_present": train_token_superposition_metal_command_present,
                "ok": metal_token_superposition_ladder_ok,
                "policy": metal_token_superposition_ladder_report.get("policy"),
                "trigger_state": metal_token_superposition_ladder_report.get("trigger_state"),
                "route_policy": metal_token_superposition_ladder_report.get("route_policy")
                if isinstance(metal_token_superposition_ladder_report.get("route_policy"), dict)
                else {},
                "summary": metal_token_superposition_ladder_report.get("summary")
                if isinstance(metal_token_superposition_ladder_report.get("summary"), dict)
                else {},
                "guardrail_validation": metal_token_superposition_ladder,
                "train_token_superposition_parity_claim_allowed": bool(
                    get_path(
                        metal_token_superposition_ladder_report,
                        ["summary", "train_token_superposition_parity_claim_allowed"],
                        True,
                    )
                ),
                "native_hot_loop_parity_claim_allowed": bool(
                    get_path(
                        metal_token_superposition_ladder_report,
                        ["summary", "native_hot_loop_parity_claim_allowed"],
                        True,
                    )
                ),
                "model_promotion_allowed": bool(
                    get_path(metal_token_superposition_ladder_report, ["summary", "model_promotion_allowed"], True)
                ),
            },
            {
                "path": rel(METAL_TOKEN_SUPERPOSITION_SCHEDULER_CANARY),
                "present": bool(metal_token_superposition_scheduler_canary_report),
                "command_present": train_token_superposition_metal_command_present,
                "ok": metal_token_superposition_scheduler_canary_ok,
                "policy": metal_token_superposition_scheduler_canary_report.get("policy"),
                "mode": metal_token_superposition_scheduler_canary_report.get("mode"),
                "train_report": metal_token_superposition_scheduler_canary_report.get("train_report"),
                "artifact": metal_token_superposition_scheduler_canary_report.get("artifact"),
                "route_policy": metal_token_superposition_scheduler_canary_report.get("route_policy"),
                "canary_policy": metal_token_superposition_scheduler_canary_report.get("canary_policy"),
                "guardrail_validation": metal_token_superposition_scheduler_canary,
                "production_scheduler_routing_enabled": bool(
                    get_path(
                        metal_token_superposition_scheduler_canary_report,
                        ["guardrails", "production_scheduler_routing_enabled"],
                        True,
                    )
                ),
                "remote_task_submitted": bool(
                    get_path(metal_token_superposition_scheduler_canary_report, ["guardrails", "remote_task_submitted"], True)
                ),
                "model_promotion_allowed": bool(
                    get_path(metal_token_superposition_scheduler_canary_report, ["guardrails", "model_promotion_allowed"], True)
                ),
            },
            {
                "path": rel(METAL_PRODUCTION_ROUTE_READINESS),
                "present": bool(metal_production_route_readiness_report),
                "ok": metal_production_route_readiness_ok,
                "trigger_state": metal_production_route_readiness.get("trigger_state"),
                "production_route_allowed": metal_production_route_readiness.get("production_route_allowed"),
                "production_route_ready_count": metal_production_route_readiness.get("production_route_ready_count"),
                "guarded_evidence_ok_count": metal_production_route_readiness.get("guarded_evidence_ok_count"),
                "surface_count": metal_production_route_readiness.get("surface_count"),
                "blocker_count": metal_production_route_readiness.get("blocker_count"),
                "blockers": metal_production_route_readiness.get("blockers"),
                "guardrail_validation": metal_production_route_readiness,
            },
            {
                "path": rel(METAL_TRAIN_ROLLOUT_REPORT),
                "present": bool(metal_train_rollout_report),
                "command_present": train_rollout_metal_command_present,
                "ok": metal_train_rollout_contract_ok,
                "state": metal_train_rollout_report.get("state"),
                "command": metal_train_rollout_report.get("command"),
                "policy": metal_train_rollout_report.get("policy"),
                "backend": metal_train_rollout_report.get("backend"),
                "implementation": metal_train_rollout_report.get("implementation"),
                "native_path": metal_train_rollout_report.get("native_path"),
                "parity_scope": metal_train_rollout_report.get("parity_scope"),
                "report_contract": metal_train_rollout_report.get("report_contract") if isinstance(metal_train_rollout_report.get("report_contract"), dict) else {},
                "artifact_write": metal_train_rollout_report.get("artifact_write") if isinstance(metal_train_rollout_report.get("artifact_write"), dict) else {},
                "artifact_equivalence": metal_train_rollout_artifact_equivalence,
                "scheduler_route_guardrail": metal_train_rollout_route_guard,
                "scheduler_dry_run": metal_train_rollout_scheduler_dry_run,
                "bounded_parity_ladder": metal_train_rollout_parity_ladder,
                "scheduler_canary": metal_train_rollout_scheduler_canary,
                "train_rollout_parity_claim_allowed": bool(metal_train_rollout_report.get("train_rollout_parity_claim_allowed")),
                "full_cli_parity_claim_allowed": bool(metal_train_rollout_report.get("full_cli_parity_claim_allowed")),
                "model_promotion_allowed": bool(metal_train_rollout_report.get("model_promotion_allowed")),
            },
        ],
        "targets": target_rows,
        "latest_work_proof": compact_work_proof(work_proof),
        "next_native_step": {
            "target": "Keep token-superposition Metal production routing locked until full native parity review or port another CUDA hot loop." if metal_token_superposition_scheduler_canary_ok and metal_train_rollout_scheduler_canary_ok else "Prepare reviewed token-superposition Metal scheduler canary." if metal_token_superposition_artifact_ok and metal_token_superposition_ladder_ok and metal_train_rollout_scheduler_canary_ok else "Design operator-reviewed token-superposition Metal route gate or add artifact-equivalence proof." if metal_token_superposition_ladder_ok and metal_train_rollout_scheduler_canary_ok else "Add a bounded train-token-superposition-metal ladder or route guard, or port another CUDA hot loop." if metal_token_superposition_contract_ok and metal_train_rollout_scheduler_canary_ok else "Build the full train-token-superposition-metal CLI contract/report, or port another CUDA hot loop to Rust/Metal." if metal_token_superposition_readout_ok and metal_train_rollout_scheduler_canary_ok else "Port token-superposition readout from Python MLX bridge to Rust/Metal or Rust/MLX." if metal_train_rollout_scheduler_canary_ok else "Prepare and execute a separately reviewed Metal scheduler canary policy." if metal_train_rollout_parity_ladder_ok else "Climb the bounded train-rollout-metal parity ladder before any production route is enabled." if metal_train_rollout_scheduler_dry_run_ok else "Run a bounded scheduler dry-run for train-rollout-metal before enabling any production route." if metal_train_rollout_route_guard_ok else "Add explicit scheduler route policy and rollback guardrails for train-rollout-metal." if metal_train_rollout_artifact_ok else "Prove production checkpoint/artifact equivalence for the Rust-owned Metal train-rollout command, then route that target explicitly.",
            "why": "The reviewed local rollout and token-superposition scheduler canaries are proven with no-cheat locks. This still does not enable production routing or a full native parity claim; the next useful Mac step is a separate operator-reviewed enablement decision or another CUDA hot-loop port." if metal_token_superposition_scheduler_canary_ok and metal_train_rollout_scheduler_canary_ok else "The reviewed local rollout scheduler canary, bounded token-superposition Metal ladder, and token-superposition artifact-equivalence proof are proven with no-cheat locks. This still does not enable production routing or a full native parity claim; the next useful Mac step is a reviewed scheduler canary." if metal_token_superposition_artifact_ok and metal_token_superposition_ladder_ok and metal_train_rollout_scheduler_canary_ok else "The reviewed local rollout scheduler canary and bounded token-superposition Metal ladder are both proven with no-cheat locks. This still does not enable production routing or a full native parity claim; the next useful Mac step is artifact equivalence or an operator-reviewed route gate." if metal_token_superposition_ladder_ok and metal_train_rollout_scheduler_canary_ok else "The reviewed local rollout scheduler canary and full token-superposition Metal CLI contract are both proven with no-cheat locks. This still does not enable production routing or a full native parity claim; the next useful Mac step is bounded ladder/route-guard evidence for token-superposition or another CUDA hot-loop port." if metal_token_superposition_contract_ok and metal_train_rollout_scheduler_canary_ok else "The reviewed local rollout scheduler canary and token-superposition Metal readout subpath are both proven with no-cheat locks. This still does not enable production routing or a full token-superposition parity claim; the next useful Mac step is a full train-token-superposition-metal CLI contract/report." if metal_token_superposition_readout_ok and metal_train_rollout_scheduler_canary_ok else "The reviewed local scheduler canary passed with routing, promotion, parity, teacher/public, external-inference, fallback, and remote-task locks intact. Token-superposition still needs Rust/Metal readout proof before the Python MLX bridge can stop being the Mac path." if metal_train_rollout_scheduler_canary_ok else "The bounded local Metal ladder passed across multiple private synthetic sizes with artifact, routing, promotion, teacher/public, external-inference, and no-fallback locks intact. It still does not justify a full native parity claim; the next step must be separately reviewed before production routing changes." if metal_train_rollout_parity_ladder_ok else "The guarded local scheduler dry-run executed and verified placement, artifact, rollback, and no-cheat locks. Production routing remains disabled; the next proof should expand bounded sizes before route enablement is even considered." if metal_train_rollout_scheduler_dry_run_ok else "Rust/Metal now writes and verifies a canonical train-rollout readout artifact and has an explicit route guard contract; production scheduler routing remains disabled until a bounded dry-run proves the route." if metal_train_rollout_route_guard_ok else "Rust/Metal now writes and verifies a canonical train-rollout readout artifact, but scheduler routing remains disabled until an explicit route policy and rollback guardrails exist." if metal_train_rollout_artifact_ok else "Rust/Metal now proves rollout_state_update, bounded rollout memory features, readout logits scoring, readout SGD/eval, a train/eval report-style proof, and a train-rollout-metal CLI report contract. The remaining blocker is production checkpoint/artifact compatibility and scheduler routing policy.",
            "minimum_done": [
                "train-rollout-metal writes a production-compatible checkpoint or a formal artifact-equivalence proof against the CUDA/MLX rollout report contract.",
                "The scheduler policy has an explicit Metal route with backend, artifact, and rollback guardrails.",
                "macos_mlx_parity_audit marks only train-rollout-cuda ready after native correctness, artifact equivalence, and routing guardrails pass.",
            ],
        },
        "guardrails": {
            "bridge_is_not_kernel_parity": True,
            "do_not_claim_native_parity_until_ready_rows_cover_all_targets": True,
            "public_benchmark_training_used": False,
            "teacher_used": False,
        },
    }


def validate_train_rollout_metal_artifact(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {"ok": False, "reason": "missing_train_rollout_metal_report"}
    artifact_write = report.get("artifact_write") if isinstance(report.get("artifact_write"), dict) else {}
    artifact_path = artifact_write.get("path") if isinstance(artifact_write.get("path"), str) else ""
    resolved_path = (ROOT / artifact_path).resolve() if artifact_path and not Path(artifact_path).is_absolute() else Path(artifact_path)
    artifact = read_json(resolved_path, {}) if artifact_path else {}
    labels = artifact.get("labels") if isinstance(artifact.get("labels"), list) else []
    weights = artifact.get("weights") if isinstance(artifact.get("weights"), list) else []
    bias = artifact.get("bias") if isinstance(artifact.get("bias"), list) else []
    expected_hv_dim = int(get_path(report, ["config", "hv_dim"], report.get("hv_dim") or 0) or 0)
    expected_output_dim = int(get_path(report, ["config", "output_dim"], report.get("labels") or 0) or 0)
    expected_feature_set = report.get("feature_set")
    checks = {
        "report_ok": bool(report.get("ok") and report.get("command") == "train-rollout-metal"),
        "artifact_write_attempted": bool(artifact_write.get("attempted")),
        "artifact_write_kind_canonical": artifact_write.get("kind") == "canonical_readout_artifact",
        "artifact_write_production_checkpoint_compatible": bool(artifact_write.get("production_checkpoint_compatible")),
        "artifact_path_present": bool(artifact_path),
        "artifact_file_present": bool(artifact_path and resolved_path.exists()),
        "artifact_json_loaded": bool(artifact),
        "hv_dim_matches_report": artifact.get("hv_dim") == expected_hv_dim,
        "output_dim_matches_report": artifact.get("output_dim") == expected_output_dim,
        "label_count_matches_output_dim": len(labels) == expected_output_dim,
        "weights_count_matches_shape": len(weights) == expected_hv_dim * expected_output_dim,
        "bias_count_matches_output_dim": len(bias) == expected_output_dim,
        "feature_set_matches_report": bool(expected_feature_set) and artifact.get("feature_set") == expected_feature_set,
        "scheduler_routing_still_locked": get_path(report, ["report_contract", "scheduler_routing_enabled"], True) is False,
        "promotion_still_locked": not bool(report.get("model_promotion_allowed")) and not bool(get_path(report, ["promotion_decision", "promote_to_training_lane"], False)),
        "parity_claim_still_locked": not bool(report.get("parity_claim_allowed")) and not bool(report.get("train_rollout_parity_claim_allowed")) and not bool(report.get("full_cli_parity_claim_allowed")),
        "no_external_inference": int(report.get("external_inference_calls") or 0) == 0,
        "no_teacher": not bool(report.get("teacher_used")),
        "no_public_training_rows": int(report.get("public_training_rows") or 0) == 0,
        "no_fallback_returns": bool(get_path(report, ["guardrails", "no_fallback_returns"], False)),
    }
    return {
        "ok": all(checks.values()),
        "path": rel(resolved_path) if artifact_path else "",
        "schema": artifact_write.get("schema"),
        "production_checkpoint_compatible": bool(artifact_write.get("production_checkpoint_compatible")),
        "scheduler_routing_enabled": bool(get_path(report, ["report_contract", "scheduler_routing_enabled"], False)),
        "expected_hv_dim": expected_hv_dim,
        "expected_output_dim": expected_output_dim,
        "weights": len(weights),
        "bias": len(bias),
        "labels": len(labels),
        "feature_set": artifact.get("feature_set"),
        "checks": checks,
    }


def validate_train_rollout_metal_sweep(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {"ok": False, "reason": "missing_train_rollout_metal_sweep_report"}
    children = report.get("children") if isinstance(report.get("children"), list) else []
    child_validations = []
    for child in children:
        if not isinstance(child, dict):
            child_validations.append({"ok": False, "reason": "child_not_object"})
            continue
        child_path = get_path(child, ["sweep_child", "child_report_path"], "")
        resolved_child = (
            (ROOT / child_path).resolve()
            if isinstance(child_path, str) and child_path and not Path(child_path).is_absolute()
            else Path(child_path)
            if isinstance(child_path, str) and child_path
            else None
        )
        full_child = read_json(resolved_child, {}) if resolved_child else {}
        if not full_child:
            full_child = child
        artifact = validate_train_rollout_metal_artifact(full_child)
        checks = {
            "child_report_present": bool(full_child),
            "child_report_ok": bool(full_child.get("ok") and full_child.get("state") == "GREEN"),
            "child_command_matches": full_child.get("command") == "train-rollout-metal",
            "child_backend_matches": full_child.get("backend") == "apple_metal",
            "child_contract_matches": get_path(full_child, ["report_contract", "matches_train_rollout_cli_surface"], False)
            is True,
            "child_scheduler_routing_disabled": get_path(full_child, ["report_contract", "scheduler_routing_enabled"], True)
            is False,
            "child_python_mlx_not_used": get_path(full_child, ["report_contract", "python_mlx_bridge_used"], True)
            is False,
            "child_artifact_equivalence_ok": bool(artifact.get("ok")),
            "child_sweep_parent_matches": get_path(full_child, ["sweep_child", "parent_command"], "")
            == "train-rollout-metal-sweep",
            "child_sweep_parity_matches": get_path(full_child, ["sweep_child", "parity_for"], "")
            == "train-rollout-cuda-sweep",
            "child_state_training_not_claimed": get_path(full_child, ["sweep_child", "state_training_native_ported"], True)
            is False,
            "child_external_inference_zero": int(full_child.get("external_inference_calls") or 0) == 0,
            "child_teacher_disabled": full_child.get("teacher_used") is False,
            "child_public_training_zero": int(full_child.get("public_training_rows") or 0) == 0,
            "child_no_fallback_returns": get_path(full_child, ["guardrails", "no_fallback_returns"], False) is True,
            "child_promotion_locked": full_child.get("model_promotion_allowed") is False
            and get_path(full_child, ["promotion_decision", "promote_to_training_lane"], True) is False,
            "child_parity_claim_locked": full_child.get("train_rollout_parity_claim_allowed") is False
            and full_child.get("full_cli_parity_claim_allowed") is False,
        }
        child_validations.append(
            {
                "ok": all(checks.values()),
                "path": rel(resolved_child) if resolved_child else "",
                "artifact_equivalence": artifact,
                "checks": checks,
            }
        )
    child_count = len(children)
    child_ok_count = sum(1 for row in child_validations if row.get("ok"))
    checks = {
        "report_ok": bool(report.get("ok") and report.get("trigger_state") == "GREEN"),
        "policy_matches": report.get("policy") == "project_theseus_macos_metal_rollout_sweep_v0",
        "command_matches": report.get("command") == "train-rollout-metal-sweep",
        "parity_for_matches": report.get("parity_for") == "train-rollout-cuda-sweep",
        "backend_matches": report.get("backend") == "apple_metal",
        "implementation_matches": report.get("implementation") == "rust_metal_rollout_sweep_guarded_proof",
        "contract_matches_surface": get_path(report, ["report_contract", "matches_train_rollout_sweep_cli_surface"], False)
        is True,
        "mirrors_mlx_command": get_path(report, ["report_contract", "mirrors_command"], "")
        == "train-rollout-mlx-sweep",
        "child_command_declared": get_path(report, ["report_contract", "child_command"], "")
        == "train-rollout-metal",
        "scheduler_routing_disabled": get_path(report, ["report_contract", "scheduler_routing_enabled"], True)
        is False
        and get_path(report, ["summary", "production_scheduler_routing_enabled"], True) is False,
        "python_mlx_bridge_not_used": get_path(report, ["report_contract", "python_mlx_bridge_used"], True)
        is False,
        "state_training_not_claimed": get_path(report, ["report_contract", "state_training_native_ported"], True)
        is False
        and get_path(report, ["summary", "state_training_native_ported"], True) is False,
        "cuda_state_training_parity_locked": get_path(
            report,
            ["summary", "cuda_state_training_parity_claim_allowed"],
            True,
        )
        is False
        and get_path(report, ["guardrails", "does_not_claim_cuda_state_training_parity"], False) is True,
        "children_present": child_count > 0,
        "all_children_ok": child_count > 0 and child_ok_count == child_count,
        "summary_child_count_matches": int(get_path(report, ["summary", "run_count"], -1) or -1)
        == child_count,
        "summary_child_ok_count_matches": int(get_path(report, ["summary", "child_ok_count"], -1) or -1)
        == child_ok_count,
        "artifact_count_matches": int(get_path(report, ["summary", "artifact_count"], -1) or -1)
        == child_count,
        "kernel_launches_positive": int(get_path(report, ["summary", "total_kernel_launches"], 0) or 0) > 0,
        "work_receipt_matches": get_path(report, ["work_receipt", "backend"], "") == "apple_metal"
        and get_path(report, ["work_receipt", "task_kind"], "") == "train_rollout_metal_sweep_cli"
        and int(get_path(report, ["work_receipt", "claimed_work_units"], 0) or 0)
        == int(get_path(report, ["summary", "total_kernel_launches"], 0) or 0),
        "external_inference_zero": int(report.get("external_inference_calls") or 0) == 0
        and int(get_path(report, ["summary", "external_inference_calls"], 0) or 0) == 0
        and get_path(report, ["guardrails", "no_external_inference"], False) is True,
        "teacher_disabled": report.get("teacher_used") is False
        and get_path(report, ["summary", "teacher_used"], True) is False
        and get_path(report, ["guardrails", "no_teacher"], False) is True,
        "public_training_zero": int(report.get("public_training_rows") or 0) == 0
        and int(get_path(report, ["summary", "public_training_rows"], 0) or 0) == 0
        and get_path(report, ["guardrails", "no_public_training_rows"], False) is True,
        "no_fallback_returns": int(get_path(report, ["summary", "fallback_returns"], 1)) == 0
        and get_path(report, ["guardrails", "no_fallback_returns"], False) is True,
        "public_calibration_not_run": get_path(report, ["guardrails", "no_public_calibration"], False) is True,
        "promotion_locked": report.get("model_promotion_allowed") is False
        and get_path(report, ["promotion_decision", "promote_to_training_lane"], True) is False,
        "parity_claim_locked": report.get("train_rollout_sweep_parity_claim_allowed") is False
        and report.get("full_cli_parity_claim_allowed") is False
        and get_path(report, ["summary", "native_hot_loop_parity_claim_allowed"], True) is False,
    }
    return {
        "ok": all(checks.values()),
        "path": rel(METAL_TRAIN_ROLLOUT_SWEEP_REPORT),
        "child_count": child_count,
        "child_ok_count": child_ok_count,
        "artifact_count": int(get_path(report, ["summary", "artifact_count"], 0) or 0),
        "total_kernel_launches": int(get_path(report, ["summary", "total_kernel_launches"], 0) or 0),
        "state_training_native_ported": get_path(report, ["summary", "state_training_native_ported"], None),
        "production_scheduler_routing_enabled": get_path(
            report,
            ["summary", "production_scheduler_routing_enabled"],
            None,
        ),
        "checks": checks,
        "children": child_validations,
    }


def validate_rollout_metal_state_training_proof(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {
            "ok": False,
            "path": rel(METAL_STATE_TRAINING_PROOF),
            "reason": "missing_rollout_metal_state_training_proof",
        }

    def num(path: list[Any], default: float = float("inf")) -> float:
        value = get_path(report, path, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    effective_tolerance = num(["effective_tolerance"], 5.0e-4)
    checks = {
        "report_ok": bool(report.get("ok") and report.get("state") == "GREEN"),
        "policy_matches": report.get("policy")
        == "project_theseus_macos_metal_rollout_state_training_proof_v0",
        "native_path_matches": report.get("native_path")
        == "rollout_state_update_plus_state_dynamics_training_update",
        "parity_scope_matches": report.get("parity_scope")
        == "bounded_private_synthetic_rollout_state_training_update",
        "parity_for_matches": report.get("parity_for")
        == "train-rollout-cuda-sweep state-training subpath",
        "state_training_semantics_proof": report.get("state_training_semantics_proof") is True,
        "state_training_native_ported": report.get("state_training_native_ported") is True,
        "state_update_on_metal": get_path(report, ["runtime_profile", "state_update_on_metal"], False) is True,
        "rollout_state_on_metal": get_path(report, ["runtime_profile", "rollout_state_on_metal"], False) is True,
        "python_mlx_bridge_not_used": get_path(report, ["runtime_profile", "python_mlx_bridge_used"], True) is False,
        "state_training_attempted": get_path(report, ["state_training", "attempted"], False) is True,
        "params_changed": get_path(report, ["state_training", "params_changed_from_base"], False) is True,
        "decision_matches": get_path(report, ["state_training", "decision_matches"], False) is True,
        "param_delta_within_tolerance": num(["metrics", "param_max_abs_delta_cpu_vs_metal"])
        <= effective_tolerance,
        "feature_delta_within_tolerance": num(["metrics", "feature_max_abs_delta_cpu_vs_metal"])
        <= effective_tolerance,
        "loss_delta_within_tolerance": num(["metrics", "loss_delta_cpu_vs_metal"]) <= effective_tolerance,
        "alignment_delta_within_tolerance": num(["metrics", "alignment_delta_cpu_vs_metal"])
        <= effective_tolerance,
        "kernel_launches_positive": int(report.get("kernel_launches") or 0) > 0,
        "external_inference_zero": int(report.get("external_inference_calls") or 0) == 0,
        "teacher_disabled": report.get("teacher_used") is False,
        "public_training_zero": int(report.get("public_training_rows") or 0) == 0,
        "fallback_returns_zero": int(report.get("fallback_returns") or 0) == 0,
        "no_fallback_returns": get_path(report, ["guardrails", "no_fallback_returns"], False) is True,
        "no_public_calibration": get_path(report, ["guardrails", "no_public_calibration"], False) is True,
        "no_public_training_rows": get_path(report, ["guardrails", "no_public_training_rows"], False) is True,
        "no_teacher": get_path(report, ["guardrails", "no_teacher"], False) is True,
        "no_external_inference": get_path(report, ["guardrails", "no_external_inference"], False) is True,
        "production_scheduler_routing_disabled": report.get("production_scheduler_routing_enabled") is False
        and get_path(report, ["guardrails", "does_not_route_scheduler_to_metal"], False) is True,
        "cuda_state_training_parity_claim_locked": report.get("cuda_state_training_parity_claim_allowed")
        is False,
        "train_rollout_sweep_parity_claim_locked": report.get("train_rollout_sweep_parity_claim_allowed")
        is False,
        "full_cli_parity_claim_locked": report.get("full_cli_parity_claim_allowed") is False,
        "model_promotion_locked": report.get("model_promotion_allowed") is False,
    }
    return {
        "ok": all(checks.values()),
        "path": rel(METAL_STATE_TRAINING_PROOF),
        "state_training_native_ported": report.get("state_training_native_ported"),
        "kernel_launches": report.get("kernel_launches"),
        "effective_tolerance": report.get("effective_tolerance"),
        "param_max_abs_delta": get_path(report, ["metrics", "param_max_abs_delta_cpu_vs_metal"], None),
        "feature_max_abs_delta": get_path(report, ["metrics", "feature_max_abs_delta_cpu_vs_metal"], None),
        "checks": checks,
    }


def validate_train_rollout_metal_sweep_scheduler_canary(
    report: dict[str, Any],
    route_policy: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {
            "ok": False,
            "path": rel(METAL_ROLLOUT_SWEEP_SCHEDULER_CANARY),
            "reason": "missing_train_rollout_sweep_scheduler_canary",
        }
    canary_policy = read_json(METAL_ROLLOUT_SWEEP_CANARY_POLICY, {})
    preflight = get_path(report, ["checks", "preflight"], {})
    result = get_path(report, ["checks", "result"], {})
    guardrails = report.get("guardrails") if isinstance(report.get("guardrails"), dict) else {}
    placement = report.get("planned_placement") if isinstance(report.get("planned_placement"), dict) else {}
    payload = placement.get("payload") if isinstance(placement.get("payload"), dict) else {}
    train_path = report.get("train_report") if isinstance(report.get("train_report"), str) else ""
    train_report = read_json(resolve_repo_path(train_path), {}) if train_path else {}
    sweep_validation = validate_train_rollout_metal_sweep(train_report)
    bounds = canary_policy.get("bounds") if isinstance(canary_policy.get("bounds"), dict) else {}
    requires = canary_policy.get("requires") if isinstance(canary_policy.get("requires"), dict) else {}
    summary = train_report.get("summary") if isinstance(train_report.get("summary"), dict) else {}
    child_guardrails = train_report.get("guardrails") if isinstance(train_report.get("guardrails"), dict) else {}
    child_contract = train_report.get("report_contract") if isinstance(train_report.get("report_contract"), dict) else {}
    checks = {
        "canary_report_ok": bool(report.get("ok")),
        "policy_matches": report.get("policy") == "project_theseus_macos_metal_rollout_sweep_scheduler_canary_v0",
        "execute_mode": report.get("mode") == "execute",
        "execution_attempted": get_path(report, ["execution", "attempted"], False) is True,
        "execution_ok": get_path(report, ["execution", "ok"], False) is True,
        "canary_policy_present": bool(canary_policy),
        "canary_policy_matches": canary_policy.get("policy") == "project_theseus_macos_metal_rollout_sweep_scheduler_canary_policy_v0",
        "route_policy_present": bool(route_policy),
        "route_policy_matches": route_policy.get("policy") == "project_theseus_macos_metal_rollout_sweep_route_policy_v0",
        "route_policy_command_matches": route_policy.get("command") == "train-rollout-metal-sweep",
        "command_matches": canary_policy.get("command") == "train-rollout-metal-sweep",
        "task_kind_matches": report.get("task_kind") == "train_rollout_metal_sweep_local_canary"
        and canary_policy.get("task_kind") == "train_rollout_metal_sweep_local_canary",
        "placement_local_only": placement.get("target") == "local",
        "payload_command_matches": payload.get("command") == "train-rollout-metal-sweep",
        "payload_backend_matches": "apple_metal" in (payload.get("backend_requirements") if isinstance(payload.get("backend_requirements"), list) else []),
        "preflight_all_ok": isinstance(preflight, dict) and all(bool(value) for value in preflight.values()),
        "result_all_ok": isinstance(result, dict) and all(bool(value) for value in result.values()),
        "required_reports_declared": bool(preflight.get("required_reports_declared")),
        "route_policy_production_disabled": route_policy.get("production_scheduler_routing_enabled") is False
        and canary_policy.get("production_scheduler_routing_enabled") is False,
        "remote_task_not_submitted": guardrails.get("remote_task_submitted") is False
        and canary_policy.get("remote_task_submitted") is False,
        "registers_worker_chunk_false": guardrails.get("registers_worker_chunk") is False
        and canary_policy.get("registers_worker_chunk") is False,
        "train_report_loaded": bool(train_report),
        "child_sweep_validation_ok": bool(sweep_validation.get("ok")),
        "child_count_matches_bounds": int(summary.get("run_count") or 0) == int(bounds.get("expected_child_runs") or 0),
        "child_artifacts_match_bounds": int(sweep_validation.get("artifact_count") or 0) == int(bounds.get("expected_child_runs") or 0),
        "kernel_launches_within_canary_cap": 0 < int(summary.get("total_kernel_launches") or 0) <= int(bounds.get("max_kernel_launches") or 0),
        "scheduler_routing_still_disabled": child_contract.get("scheduler_routing_enabled") is False
        and summary.get("production_scheduler_routing_enabled") is False
        and guardrails.get("production_scheduler_routing_enabled") is False,
        "state_training_still_unclaimed": child_contract.get("state_training_native_ported") is False
        and summary.get("state_training_native_ported") is False
        and summary.get("cuda_state_training_parity_claim_allowed") is False
        and guardrails.get("state_training_native_ported") is False
        and guardrails.get("cuda_state_training_parity_claim_allowed") is False
        and requires.get("state_training_native_ported") is False
        and requires.get("cuda_state_training_parity_claim_allowed") is False,
        "promotion_still_locked": guardrails.get("model_promotion_allowed") is False
        and train_report.get("model_promotion_allowed") is False
        and get_path(train_report, ["promotion_decision", "promote_to_training_lane"], True) is False,
        "parity_claim_still_locked": guardrails.get("train_rollout_sweep_parity_claim_allowed") is False
        and guardrails.get("native_hot_loop_parity_claim_allowed") is False
        and train_report.get("train_rollout_sweep_parity_claim_allowed") is False
        and train_report.get("full_cli_parity_claim_allowed") is False
        and summary.get("native_hot_loop_parity_claim_allowed") is False,
        "external_inference_zero": int(guardrails.get("external_inference_calls") or 0) == 0
        and int(train_report.get("external_inference_calls") or 0) == 0
        and int(summary.get("external_inference_calls") or 0) == 0
        and int(requires.get("external_inference_calls") or 0) == 0,
        "teacher_disabled": guardrails.get("teacher_used") is False
        and train_report.get("teacher_used") is False
        and summary.get("teacher_used") is False
        and requires.get("teacher_used") is False,
        "public_training_zero": int(train_report.get("public_training_rows") or 0) == 0
        and int(summary.get("public_training_rows") or 0) == 0
        and int(requires.get("public_training_rows") or 0) == 0,
        "no_fallback_returns": guardrails.get("no_fallback_returns") is True
        and child_guardrails.get("no_fallback_returns") is True
        and int(summary.get("fallback_returns") or 0) == 0
        and requires.get("no_fallback_returns") is True,
        "public_calibration_not_run": guardrails.get("public_calibration_run") is False
        and child_guardrails.get("no_public_calibration") is True
        and requires.get("public_calibration_not_run") is True,
    }
    return {
        "ok": all(checks.values()),
        "path": rel(METAL_ROLLOUT_SWEEP_SCHEDULER_CANARY),
        "canary_policy": rel(METAL_ROLLOUT_SWEEP_CANARY_POLICY),
        "route_policy": rel(METAL_ROLLOUT_SWEEP_ROUTE_POLICY),
        "mode": report.get("mode"),
        "train_report": train_path,
        "artifact_dir": report.get("artifact_dir"),
        "task_kind": report.get("task_kind"),
        "target": placement.get("target"),
        "child_count": summary.get("run_count"),
        "artifact_count": sweep_validation.get("artifact_count"),
        "kernel_launches": summary.get("total_kernel_launches"),
        "state_training_native_ported": summary.get("state_training_native_ported"),
        "cuda_state_training_parity_claim_allowed": summary.get("cuda_state_training_parity_claim_allowed"),
        "production_scheduler_routing_enabled": guardrails.get("production_scheduler_routing_enabled"),
        "remote_task_submitted": guardrails.get("remote_task_submitted"),
        "sweep_validation": sweep_validation,
        "checks": checks,
    }


def validate_train_standalone_metal_artifact(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {"ok": False, "reason": "missing_train_standalone_metal_report"}
    artifact_write = report.get("artifact_write") if isinstance(report.get("artifact_write"), dict) else {}
    artifact_path = artifact_write.get("path") if isinstance(artifact_write.get("path"), str) else ""
    resolved_path = (ROOT / artifact_path).resolve() if artifact_path and not Path(artifact_path).is_absolute() else Path(artifact_path)
    artifact = read_json(resolved_path, {}) if artifact_path else {}
    labels = artifact.get("labels") if isinstance(artifact.get("labels"), list) else []
    weights = artifact.get("weights") if isinstance(artifact.get("weights"), list) else []
    bias = artifact.get("bias") if isinstance(artifact.get("bias"), list) else []
    expected_hv_dim = int(report.get("hv_dim") or 0)
    expected_output_dim = int(report.get("labels") or 0)
    checks = {
        "report_ok": bool(report.get("ok") and report.get("state") == "GREEN"),
        "command_matches": report.get("command") == "train-standalone-metal",
        "parity_for_matches": report.get("parity_for") == "train-standalone-cuda",
        "backend_matches": report.get("backend") == "apple_metal",
        "implementation_matches": report.get("implementation") == "rust_metal_structured_cgs_vsa_readout_cli",
        "cuda_fallback_false": report.get("cuda_fallback") is False,
        "symbolic_fallback_false": report.get("symbolic_fallback") is False,
        "contract_matches_surface": get_path(report, ["report_contract", "matches_train_standalone_cli_surface"], False)
        is True,
        "mirrors_mlx_command": get_path(report, ["report_contract", "mirrors_command"], "")
        == "train-standalone-mlx",
        "scheduler_routing_disabled": get_path(report, ["report_contract", "scheduler_routing_enabled"], True)
        is False,
        "python_mlx_bridge_not_used": get_path(report, ["report_contract", "python_mlx_bridge_used"], True)
        is False
        and get_path(report, ["runtime_profile", "python_mlx_bridge_used"], "true") == "false",
        "native_rust_owned": get_path(report, ["runtime_profile", "native_rust_owned"], "false") == "true",
        "native_readout_subpath_declared": bool(get_path(report, ["report_contract", "native_readout_subpath"], "")),
        "kernel_launches_positive": int(report.get("kernel_launches") or 0) > 0,
        "work_receipt_matches": get_path(report, ["work_receipt", "backend"], "") == "apple_metal"
        and get_path(report, ["work_receipt", "task_kind"], "") == "train_standalone_metal_cli"
        and int(get_path(report, ["work_receipt", "claimed_work_units"], 0) or 0)
        == int(report.get("kernel_launches") or 0),
        "artifact_write_attempted": bool(artifact_write.get("attempted")),
        "artifact_write_kind_canonical": artifact_write.get("kind") == "canonical_readout_artifact",
        "artifact_write_production_checkpoint_compatible": bool(artifact_write.get("production_checkpoint_compatible")),
        "artifact_path_present": bool(artifact_path),
        "artifact_file_present": bool(artifact_path and resolved_path.exists()),
        "artifact_json_loaded": bool(artifact),
        "hv_dim_matches_report": artifact.get("hv_dim") == expected_hv_dim,
        "output_dim_matches_report": artifact.get("output_dim") == expected_output_dim,
        "label_count_matches_output_dim": len(labels) == expected_output_dim,
        "weights_count_matches_shape": len(weights) == expected_hv_dim * expected_output_dim,
        "bias_count_matches_output_dim": len(bias) == expected_output_dim,
        "feature_set_matches_contract": artifact.get("feature_set") == "structured_cgs_vsa_metal_readout"
        and artifact_write.get("feature_set") == artifact.get("feature_set"),
        "promotion_locked": report.get("model_promotion_allowed") is False
        and get_path(report, ["promotion_decision", "promote_to_training_lane"], True) is False
        and artifact_write.get("promotion_allowed") is False,
        "parity_claim_locked": report.get("train_standalone_parity_claim_allowed") is False
        and report.get("full_cli_parity_claim_allowed") is False
        and artifact_write.get("train_standalone_parity_claim_allowed") is False,
        "external_inference_zero": int(report.get("external_inference_calls") or 0) == 0
        and get_path(report, ["guardrails", "no_external_inference"], False) is True,
        "teacher_disabled": report.get("teacher_used") is False
        and get_path(report, ["guardrails", "no_teacher"], False) is True,
        "public_training_zero": int(report.get("public_training_rows") or 0) == 0
        and get_path(report, ["guardrails", "no_public_training_rows"], False) is True,
        "no_fallback_returns": get_path(report, ["guardrails", "no_fallback_returns"], False) is True,
        "public_calibration_not_run": get_path(report, ["guardrails", "no_public_calibration"], False) is True,
    }
    return {
        "ok": all(checks.values()),
        "path": rel(resolved_path) if artifact_path else "",
        "schema": artifact_write.get("schema"),
        "production_checkpoint_compatible": bool(artifact_write.get("production_checkpoint_compatible")),
        "scheduler_routing_enabled": bool(get_path(report, ["report_contract", "scheduler_routing_enabled"], False)),
        "expected_hv_dim": expected_hv_dim,
        "expected_output_dim": expected_output_dim,
        "weights": len(weights),
        "bias": len(bias),
        "labels": len(labels),
        "feature_set": artifact.get("feature_set"),
        "checks": checks,
    }


def validate_train_standalone_metal_scheduler_canary(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {
            "ok": False,
            "path": rel(METAL_STANDALONE_SCHEDULER_CANARY),
            "reason": "missing_standalone_scheduler_canary_report",
        }
    route_policy = read_json(METAL_STANDALONE_ROUTE_POLICY, {})
    canary_policy = read_json(METAL_STANDALONE_CANARY_POLICY, {})
    train_path = report.get("train_report") if isinstance(report.get("train_report"), str) else ""
    artifact_path = report.get("artifact") if isinstance(report.get("artifact"), str) else ""
    train_report = read_json(resolve_repo_path(train_path), {}) if train_path else {}
    artifact = read_json(resolve_repo_path(artifact_path), {}) if artifact_path else {}
    artifact_equivalence = validate_train_standalone_metal_artifact(train_report) if train_report else {"ok": False}
    preflight = get_path(report, ["checks", "preflight"], {})
    result = get_path(report, ["checks", "result"], {})
    placement = report.get("planned_placement") if isinstance(report.get("planned_placement"), dict) else {}
    payload = placement.get("payload") if isinstance(placement.get("payload"), dict) else {}
    guardrails = report.get("guardrails") if isinstance(report.get("guardrails"), dict) else {}
    execution = report.get("execution") if isinstance(report.get("execution"), dict) else {}
    bounds = report.get("canary_bounds") if isinstance(report.get("canary_bounds"), dict) else {}
    runtime = train_report.get("runtime_profile") if isinstance(train_report.get("runtime_profile"), dict) else {}
    child_guardrails = train_report.get("guardrails") if isinstance(train_report.get("guardrails"), dict) else {}
    child_contract = train_report.get("report_contract") if isinstance(train_report.get("report_contract"), dict) else {}
    required_reports = canary_policy.get("required_reports") if isinstance(canary_policy.get("required_reports"), list) else []
    checks = {
        "canary_report_ok": bool(report.get("ok")),
        "policy_matches": report.get("policy") == "project_theseus_macos_metal_standalone_scheduler_canary_v0",
        "execute_mode": report.get("mode") == "execute",
        "execution_attempted": bool(execution.get("attempted")),
        "execution_ok": bool(execution.get("ok")) and int(execution.get("returncode") or 0) == 0,
        "route_policy_present": bool(route_policy),
        "route_policy_matches": route_policy.get("policy") == "project_theseus_macos_metal_standalone_route_policy_v0",
        "route_policy_command_matches": route_policy.get("command") == "train-standalone-metal",
        "route_policy_production_disabled": route_policy.get("production_scheduler_routing_enabled") is False,
        "canary_policy_present": bool(canary_policy),
        "canary_policy_matches": canary_policy.get("policy") == "project_theseus_macos_metal_standalone_scheduler_canary_policy_v0",
        "canary_policy_local_only": canary_policy.get("local_only") is True,
        "canary_policy_production_disabled": canary_policy.get("production_scheduler_routing_enabled") is False,
        "required_reports_declared": all(
            path in required_reports
            for path in [
                "reports/symliquid_standalone_metal_train_report.json",
                "reports/macos_metal_train_standalone_readout_artifact.json",
                "reports/macos_mlx_parity_audit.json",
            ]
        ),
        "preflight_all_ok": bool(preflight) and all(bool(value) for value in preflight.values()),
        "result_all_ok": bool(result) and all(bool(value) for value in result.values()),
        "placement_local_only": placement.get("target") == "local",
        "task_kind_is_canary": report.get("task_kind") == "train_standalone_metal_local_canary",
        "task_kind_not_registered_worker_chunk": bool(preflight.get("canary_not_registered_worker_chunk")),
        "payload_command_matches": payload.get("command") == "train-standalone-metal",
        "payload_backend_matches": "apple_metal" in (payload.get("backend_requirements") if isinstance(payload.get("backend_requirements"), list) else []),
        "train_report_loaded": bool(train_report),
        "artifact_loaded": bool(artifact),
        "train_report_ok": bool(train_report.get("ok") and train_report.get("state") == "GREEN"),
        "train_report_command_matches": train_report.get("command") == "train-standalone-metal",
        "train_report_backend_matches": train_report.get("backend") == "apple_metal" and runtime.get("backend") == "apple_metal",
        "native_rust_owned": runtime.get("native_rust_owned") == "true",
        "python_mlx_bridge_not_used": runtime.get("python_mlx_bridge_used") == "false",
        "artifact_equivalence_ok": bool(artifact_equivalence.get("ok")),
        "artifact_shape_loaded": bool(artifact),
        "kernel_launches_within_canary_cap": 0 < int(train_report.get("kernel_launches") or 0) <= int(bounds.get("max_kernel_launches") or 0),
        "scheduler_routing_still_disabled": child_contract.get("scheduler_routing_enabled") is False
        and child_guardrails.get("does_not_route_scheduler_to_metal") is True
        and guardrails.get("production_scheduler_routing_enabled") is False,
        "remote_task_not_submitted": guardrails.get("remote_task_submitted") is False,
        "registers_worker_chunk_false": guardrails.get("registers_worker_chunk") is False,
        "promotion_still_locked": train_report.get("model_promotion_allowed") is False
        and guardrails.get("model_promotion_allowed") is False
        and get_path(train_report, ["promotion_decision", "promote_to_training_lane"], True) is False,
        "parity_claim_still_locked": train_report.get("train_standalone_parity_claim_allowed") is False
        and train_report.get("full_cli_parity_claim_allowed") is False
        and guardrails.get("train_standalone_parity_claim_allowed") is False
        and guardrails.get("native_hot_loop_parity_claim_allowed") is False,
        "external_inference_zero": int(train_report.get("external_inference_calls") or 0) == 0
        and int(guardrails.get("external_inference_calls") or 0) == 0
        and child_guardrails.get("no_external_inference") is True,
        "teacher_disabled": train_report.get("teacher_used") is False
        and guardrails.get("teacher_used") is False
        and child_guardrails.get("no_teacher") is True,
        "public_training_zero": int(train_report.get("public_training_rows") or 0) == 0
        and guardrails.get("public_benchmark_training_used") is False
        and child_guardrails.get("no_public_training_rows") is True,
        "public_calibration_not_run": guardrails.get("public_calibration_run") is False
        and child_guardrails.get("no_public_calibration") is True,
        "no_fallback_returns": train_report.get("symbolic_fallback") is False
        and guardrails.get("no_fallback_returns") is True
        and child_guardrails.get("no_fallback_returns") is True,
    }
    return {
        "ok": all(checks.values()),
        "path": rel(METAL_STANDALONE_SCHEDULER_CANARY),
        "mode": report.get("mode"),
        "train_report": train_path,
        "artifact": artifact_path,
        "task_kind": report.get("task_kind"),
        "target": placement.get("target"),
        "backend_requirements": payload.get("backend_requirements") if isinstance(payload.get("backend_requirements"), list) else [],
        "max_kernel_launches": bounds.get("max_kernel_launches"),
        "kernel_launches": train_report.get("kernel_launches"),
        "production_scheduler_routing_enabled": guardrails.get("production_scheduler_routing_enabled"),
        "remote_task_submitted": guardrails.get("remote_task_submitted"),
        "artifact_equivalence": artifact_equivalence,
        "checks": checks,
    }


def validate_train_token_superposition_metal_artifact(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {"ok": False, "reason": "missing_train_token_superposition_metal_report"}
    artifact_write = report.get("artifact_write") if isinstance(report.get("artifact_write"), dict) else {}
    artifact_path = artifact_write.get("path") if isinstance(artifact_write.get("path"), str) else ""
    resolved_path = (ROOT / artifact_path).resolve() if artifact_path and not Path(artifact_path).is_absolute() else Path(artifact_path)
    artifact = read_json(resolved_path, {}) if artifact_path else {}
    labels = artifact.get("labels") if isinstance(artifact.get("labels"), list) else []
    weights = artifact.get("weights") if isinstance(artifact.get("weights"), list) else []
    bias = artifact.get("bias") if isinstance(artifact.get("bias"), list) else []
    dataset = report.get("dataset") if isinstance(report.get("dataset"), dict) else {}
    expected_hv_dim = int(dataset.get("hv_dim") or get_path(report, ["config", "hv_dim"], 0) or 0)
    expected_output_dim = int(dataset.get("vocab_size") or 0)
    checks = {
        "report_ok": bool(report.get("ok") and report.get("command") == "train-token-superposition-metal"),
        "artifact_write_attempted": bool(artifact_write.get("attempted")),
        "artifact_write_kind_canonical": artifact_write.get("kind") == "canonical_readout_artifact",
        "artifact_write_production_checkpoint_compatible": bool(artifact_write.get("production_checkpoint_compatible")),
        "artifact_path_present": bool(artifact_path),
        "artifact_file_present": bool(artifact_path and resolved_path.exists()),
        "artifact_json_loaded": bool(artifact),
        "hv_dim_matches_report": artifact.get("hv_dim") == expected_hv_dim,
        "output_dim_matches_report": artifact.get("output_dim") == expected_output_dim,
        "label_count_matches_output_dim": len(labels) == expected_output_dim,
        "weights_count_matches_shape": len(weights) == expected_hv_dim * expected_output_dim,
        "bias_count_matches_output_dim": len(bias) == expected_output_dim,
        "feature_set_matches_contract": artifact.get("feature_set")
        == "metal_token_superposition_readout_private_residual_train_eval"
        and artifact_write.get("feature_set") == artifact.get("feature_set"),
        "scheduler_routing_still_locked": get_path(report, ["report_contract", "scheduler_routing_enabled"], True) is False,
        "promotion_still_locked": not bool(report.get("model_promotion_allowed"))
        and not bool(get_path(report, ["promotion_decision", "promote_to_training_lane"], False))
        and artifact_write.get("promotion_allowed") is False,
        "parity_claim_still_locked": not bool(report.get("train_token_superposition_parity_claim_allowed"))
        and not bool(report.get("full_cli_parity_claim_allowed"))
        and artifact_write.get("train_token_superposition_parity_claim_allowed") is False,
        "no_external_inference": int(report.get("external_inference_calls") or 0) == 0,
        "no_teacher": not bool(report.get("teacher_used")),
        "no_public_training_rows": int(report.get("public_training_rows") or 0) == 0,
        "no_fallback_returns": bool(get_path(report, ["guardrails", "no_fallback_returns"], False)),
    }
    return {
        "ok": all(checks.values()),
        "path": rel(resolved_path) if artifact_path else "",
        "schema": artifact_write.get("schema"),
        "production_checkpoint_compatible": bool(artifact_write.get("production_checkpoint_compatible")),
        "scheduler_routing_enabled": bool(get_path(report, ["report_contract", "scheduler_routing_enabled"], False)),
        "expected_hv_dim": expected_hv_dim,
        "expected_output_dim": expected_output_dim,
        "weights": len(weights),
        "bias": len(bias),
        "labels": len(labels),
        "feature_set": artifact.get("feature_set"),
        "checks": checks,
    }


def validate_train_rollout_metal_route_policy(
    report: dict[str, Any],
    artifact_equivalence: dict[str, Any],
) -> dict[str, Any]:
    policy = read_json(METAL_ROUTE_POLICY, {})
    if not isinstance(policy, dict) or not policy:
        return {"ok": False, "path": rel(METAL_ROUTE_POLICY), "reason": "missing_route_policy"}
    requires = policy.get("requires") if isinstance(policy.get("requires"), dict) else {}
    rollback = policy.get("rollback_guardrails") if isinstance(policy.get("rollback_guardrails"), list) else []
    required_reports = policy.get("required_reports") if isinstance(policy.get("required_reports"), list) else []
    checks = {
        "policy_present": True,
        "command_matches": policy.get("command") == "train-rollout-metal",
        "parity_target_matches": policy.get("parity_for") == "train-rollout-cuda",
        "backend_matches_report": policy.get("backend") == report.get("backend") == "apple_metal",
        "bounded_smoke_only": bool(policy.get("bounded_smoke_route_enabled")) and policy.get("route_state") == "guarded_smoke_only",
        "production_scheduler_route_disabled": policy.get("production_scheduler_routing_enabled") is False,
        "report_scheduler_route_disabled": get_path(report, ["report_contract", "scheduler_routing_enabled"], True) is False,
        "artifact_equivalence_required": requires.get("macos_mlx_parity_audit_artifact_equivalence_ok") is True,
        "artifact_equivalence_ok": bool(artifact_equivalence.get("ok")),
        "canonical_artifact_required": requires.get("canonical_readout_artifact") is True,
        "promotion_locked": requires.get("model_promotion_allowed") is False and not bool(report.get("model_promotion_allowed")),
        "parity_claim_locked": requires.get("train_rollout_parity_claim_allowed") is False and not bool(report.get("train_rollout_parity_claim_allowed")),
        "external_inference_zero": requires.get("external_inference_calls") == 0 and int(report.get("external_inference_calls") or 0) == 0,
        "teacher_disabled": requires.get("teacher_used") is False and not bool(report.get("teacher_used")),
        "public_training_zero": requires.get("public_training_rows") == 0 and int(report.get("public_training_rows") or 0) == 0,
        "fallback_returns_forbidden": requires.get("no_fallback_returns") is True and bool(get_path(report, ["guardrails", "no_fallback_returns"], False)),
        "remote_scope_unchanged": bool(policy.get("does_not_change_hive_remote_task_scope")),
        "no_arbitrary_remote_execution": bool(policy.get("no_arbitrary_remote_execution")),
        "required_reports_declared": all(
            path in required_reports
            for path in [
                "reports/macos_mlx_parity_audit.json",
                "reports/symliquid_rollout_metal_train_report.json",
                "reports/macos_metal_train_rollout_readout_artifact.json",
            ]
        ),
        "rollback_guardrails_declared": all(
            item in rollback
            for item in [
                "artifact_equivalence_failure",
                "external_inference_or_teacher_or_public_training_rows",
                "promotion_or_parity_claim_enabled",
                "scheduler_report_regression",
            ]
        ),
    }
    return {
        "ok": all(checks.values()),
        "path": rel(METAL_ROUTE_POLICY),
        "route_state": policy.get("route_state"),
        "bounded_smoke_route_enabled": bool(policy.get("bounded_smoke_route_enabled")),
        "production_scheduler_routing_enabled": bool(policy.get("production_scheduler_routing_enabled")),
        "next_allowed_step": policy.get("next_allowed_step"),
        "rollback_guardrails": rollback,
        "checks": checks,
    }


def validate_train_rollout_metal_scheduler_dry_run(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {"ok": False, "path": rel(METAL_SCHEDULER_DRY_RUN), "reason": "missing_scheduler_dry_run_report"}
    preflight = get_path(report, ["checks", "preflight"], {})
    result = get_path(report, ["checks", "result"], {})
    placement = report.get("planned_placement") if isinstance(report.get("planned_placement"), dict) else {}
    payload = placement.get("payload") if isinstance(placement.get("payload"), dict) else {}
    guardrails = report.get("guardrails") if isinstance(report.get("guardrails"), dict) else {}
    execution = report.get("execution") if isinstance(report.get("execution"), dict) else {}
    bounds = report.get("dry_run_bounds") if isinstance(report.get("dry_run_bounds"), dict) else {}
    checks = {
        "dry_run_report_ok": bool(report.get("ok")),
        "policy_matches": report.get("policy") == "project_theseus_macos_metal_scheduler_dry_run_v0",
        "execute_mode": report.get("mode") == "execute",
        "execution_attempted": bool(execution.get("attempted")),
        "execution_ok": bool(execution.get("ok")) and int(execution.get("returncode") or 0) == 0,
        "placement_local_only": placement.get("target") == "local",
        "task_kind_is_dry_run": placement.get("task_kind") == "train_rollout_metal_dry_run",
        "dry_run_not_registered_worker_chunk": bool(preflight.get("dry_run_not_registered_worker_chunk")),
        "route_policy_present": bool(preflight.get("route_policy_present")),
        "route_policy_matches": bool(preflight.get("route_policy_command_matches")) and bool(preflight.get("route_policy_backend_matches")),
        "production_scheduler_routing_disabled": bool(preflight.get("production_scheduler_routing_disabled")) and guardrails.get("production_scheduler_routing_enabled") is False,
        "remote_task_not_submitted": guardrails.get("remote_task_submitted") is False,
        "no_remote_task_scope_change": bool(preflight.get("no_remote_task_scope_change")),
        "no_arbitrary_remote_execution": bool(preflight.get("no_arbitrary_remote_execution")),
        "bounded_kernel_cap": int(bounds.get("max_kernel_launches") or 0) <= 64 and bool(result.get("kernel_launches_bounded")),
        "command_matches": payload.get("command") == "train-rollout-metal" and bool(result.get("command_matches")),
        "backend_matches": bool(result.get("backend_matches")),
        "artifact_validated": bool(result.get("artifact_file_loaded")) and bool(result.get("artifact_write_production_compatible")),
        "artifact_shape_validated": bool(result.get("artifact_hv_dim_matches_report")) and bool(result.get("artifact_output_dim_matches_report")) and bool(result.get("artifact_weights_count_matches")) and bool(result.get("artifact_bias_count_matches")),
        "work_receipt_accepted": bool(result.get("work_receipt_accepted")),
        "scheduler_routing_still_disabled": bool(result.get("scheduler_routing_still_disabled")),
        "promotion_still_locked": bool(result.get("promotion_still_locked")) and guardrails.get("model_promotion_allowed") is False,
        "parity_claim_still_locked": bool(result.get("parity_claim_still_locked")) and guardrails.get("train_rollout_parity_claim_allowed") is False,
        "external_inference_zero": bool(result.get("external_inference_zero")) and int(guardrails.get("external_inference_calls") or 0) == 0,
        "teacher_disabled": bool(result.get("teacher_disabled")) and guardrails.get("teacher_used") is False,
        "public_training_zero": bool(result.get("public_training_zero")) and guardrails.get("public_benchmark_training_used") is False,
        "no_fallback_returns": bool(result.get("no_fallback_returns")) and guardrails.get("no_fallback_returns") is True,
    }
    return {
        "ok": all(checks.values()),
        "path": rel(METAL_SCHEDULER_DRY_RUN),
        "mode": report.get("mode"),
        "train_report": report.get("train_report"),
        "artifact": report.get("artifact"),
        "task_kind": placement.get("task_kind"),
        "target": placement.get("target"),
        "backend_requirements": payload.get("backend_requirements") if isinstance(payload.get("backend_requirements"), list) else [],
        "max_kernel_launches": bounds.get("max_kernel_launches"),
        "production_scheduler_routing_enabled": guardrails.get("production_scheduler_routing_enabled"),
        "remote_task_submitted": guardrails.get("remote_task_submitted"),
        "checks": checks,
    }


def validate_train_rollout_metal_parity_ladder(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {"ok": False, "path": rel(METAL_PARITY_LADDER), "reason": "missing_metal_parity_ladder_report"}
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    guardrails = report.get("guardrails") if isinstance(report.get("guardrails"), dict) else {}
    tiers = report.get("tiers") if isinstance(report.get("tiers"), list) else []
    tier_results: list[dict[str, Any]] = []
    for row in tiers:
        if not isinstance(row, dict):
            continue
        config = row.get("config") if isinstance(row.get("config"), dict) else {}
        recorded_checks = row.get("checks") if isinstance(row.get("checks"), dict) else {}
        report_path = row.get("report") if isinstance(row.get("report"), str) else ""
        artifact_path = row.get("artifact") if isinstance(row.get("artifact"), str) else ""
        resolved_report = resolve_repo_path(report_path) if report_path else Path()
        resolved_artifact = resolve_repo_path(artifact_path) if artifact_path else Path()
        child_report = read_json(resolved_report, {}) if report_path else {}
        child_artifact = read_json(resolved_artifact, {}) if artifact_path else {}
        artifact_equivalence = validate_train_rollout_metal_artifact(child_report) if child_report else {"ok": False}
        max_kernel_launches = int(config.get("max_kernel_launches") or 0)
        kernel_launches = int(row.get("kernel_launches") or child_report.get("kernel_launches") or 0)
        expected_hv_dim = int(config.get("hv_dim") or 0)
        expected_output_dim = int(config.get("output_dim") or 0)
        expected_rows = int(config.get("cases_per_task") or 0)
        tolerance = float(config.get("tolerance") or 0.0)
        child_runtime = child_report.get("runtime_profile") if isinstance(child_report.get("runtime_profile"), dict) else {}
        child_contract = child_report.get("report_contract") if isinstance(child_report.get("report_contract"), dict) else {}
        child_guardrails = child_report.get("guardrails") if isinstance(child_report.get("guardrails"), dict) else {}
        tier_checks = {
            "row_ok": bool(row.get("ok")),
            "recorded_checks_present": bool(recorded_checks),
            "recorded_checks_all_ok": bool(recorded_checks) and all(bool(ok) for ok in recorded_checks.values()),
            "report_path_present": bool(report_path),
            "artifact_path_present": bool(artifact_path),
            "child_report_loaded": bool(child_report),
            "child_artifact_loaded": bool(child_artifact),
            "child_report_ok": bool(child_report.get("ok") and child_report.get("state") == "GREEN"),
            "child_command_matches": child_report.get("command") == "train-rollout-metal",
            "child_backend_matches": child_report.get("backend") == "apple_metal" and child_runtime.get("backend") == "apple_metal",
            "native_rust_owned": child_runtime.get("native_rust_owned") is True,
            "python_mlx_bridge_not_used": child_runtime.get("python_mlx_bridge_used") is False,
            "kernel_launches_positive": kernel_launches > 0,
            "kernel_launches_within_tier_cap": max_kernel_launches > 0 and kernel_launches <= max_kernel_launches,
            "train_rows_match": int(child_report.get("train_rows") or 0) == expected_rows,
            "eval_rows_match": int(child_report.get("eval_rows") or 0) == expected_rows,
            "hv_dim_matches": int(child_report.get("hv_dim") or 0) == expected_hv_dim,
            "output_dim_matches": int(child_report.get("labels") or 0) == expected_output_dim,
            "tolerance_declared": abs(float(get_path(child_report, ["args", "tolerance"], 0.0) or 0.0) - tolerance) <= 1.0e-8,
            "tolerance_bounded": 0.0 < tolerance <= 0.0005,
            "artifact_equivalence_ok": bool(artifact_equivalence.get("ok")),
            "artifact_validated": bool(row.get("artifact_validated")),
            "scheduler_routing_still_disabled": child_contract.get("scheduler_routing_enabled") is False
            and child_guardrails.get("does_not_route_scheduler_to_metal") is True,
            "promotion_still_locked": child_report.get("model_promotion_allowed") is False
            and get_path(child_report, ["promotion_decision", "promote_to_training_lane"], True) is False,
            "parity_claim_still_locked": child_report.get("train_rollout_parity_claim_allowed") is False
            and child_report.get("full_cli_parity_claim_allowed") is False
            and child_report.get("parity_claim_allowed") is False,
            "external_inference_zero": int(child_report.get("external_inference_calls") or 0) == 0
            and child_guardrails.get("no_external_inference") is True,
            "teacher_disabled": child_report.get("teacher_used") is False and child_guardrails.get("no_teacher") is True,
            "public_training_zero": int(child_report.get("public_training_rows") or 0) == 0
            and child_guardrails.get("no_public_training_rows") is True,
            "no_fallback_returns": child_report.get("symbolic_fallback") is False
            and child_guardrails.get("no_fallback_returns") is True,
        }
        tier_results.append(
            {
                "tier_id": row.get("tier_id"),
                "ok": all(tier_checks.values()),
                "report": report_path,
                "artifact": artifact_path,
                "kernel_launches": kernel_launches,
                "max_kernel_launches": max_kernel_launches,
                "artifact_equivalence": artifact_equivalence,
                "checks": tier_checks,
            }
        )
    tier_ok_count = sum(1 for row in tier_results if row.get("ok"))
    checks = {
        "policy_matches": report.get("policy") == "project_theseus_macos_metal_parity_ladder_v0",
        "trigger_green": report.get("trigger_state") == "GREEN",
        "execute_true": report.get("execute") is True,
        "tier_count_at_least_three": len(tier_results) >= 3,
        "all_tiers_ok": bool(tier_results) and tier_ok_count == len(tier_results),
        "summary_tier_count_matches": int(summary.get("tier_count") or 0) == len(tier_results),
        "summary_tier_ok_count_matches": int(summary.get("tier_ok_count") or 0) == tier_ok_count,
        "summary_artifact_count_matches": int(summary.get("artifact_count") or 0) == tier_ok_count,
        "production_scheduler_routing_disabled": summary.get("production_scheduler_routing_enabled") is False
        and guardrails.get("production_scheduler_routing_enabled") is False,
        "remote_task_not_submitted": summary.get("remote_task_submitted") is False
        and guardrails.get("remote_task_submitted") is False,
        "model_promotion_locked": summary.get("model_promotion_allowed") is False
        and guardrails.get("model_promotion_allowed") is False,
        "train_rollout_parity_claim_locked": summary.get("train_rollout_parity_claim_allowed") is False
        and guardrails.get("train_rollout_parity_claim_allowed") is False,
        "native_hot_loop_parity_claim_locked": summary.get("native_hot_loop_parity_claim_allowed") is False
        and guardrails.get("native_hot_loop_parity_claim_allowed") is False,
        "external_inference_zero": int(summary.get("external_inference_calls") or 0) == 0
        and int(guardrails.get("external_inference_calls") or 0) == 0,
        "teacher_disabled": summary.get("teacher_used") is False and guardrails.get("teacher_used") is False,
        "public_training_zero": int(summary.get("public_training_rows") or 0) == 0
        and guardrails.get("public_benchmark_training_used") is False,
        "fallback_returns_forbidden": int(summary.get("fallback_returns") or 0) == 0
        and guardrails.get("fallback_returns_allowed") is False,
        "public_calibration_not_run": guardrails.get("public_calibration_run") is False,
    }
    return {
        "ok": all(checks.values()),
        "path": rel(METAL_PARITY_LADDER),
        "trigger_state": report.get("trigger_state"),
        "execute": bool(report.get("execute")),
        "tier_count": len(tier_results),
        "tier_ok_count": tier_ok_count,
        "total_kernel_launches": summary.get("total_kernel_launches"),
        "max_kernel_launches": summary.get("max_kernel_launches"),
        "production_scheduler_routing_enabled": summary.get("production_scheduler_routing_enabled"),
        "remote_task_submitted": summary.get("remote_task_submitted"),
        "model_promotion_allowed": summary.get("model_promotion_allowed"),
        "train_rollout_parity_claim_allowed": summary.get("train_rollout_parity_claim_allowed"),
        "native_hot_loop_parity_claim_allowed": summary.get("native_hot_loop_parity_claim_allowed"),
        "tiers": tier_results,
        "checks": checks,
    }


def validate_train_rollout_metal_scheduler_canary(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {"ok": False, "path": rel(METAL_SCHEDULER_CANARY), "reason": "missing_scheduler_canary_report"}
    canary_policy = read_json(METAL_CANARY_POLICY, {})
    train_path = report.get("train_report") if isinstance(report.get("train_report"), str) else ""
    artifact_path = report.get("artifact") if isinstance(report.get("artifact"), str) else ""
    train_report = read_json(resolve_repo_path(train_path), {}) if train_path else {}
    artifact = read_json(resolve_repo_path(artifact_path), {}) if artifact_path else {}
    artifact_equivalence = validate_train_rollout_metal_artifact(train_report) if train_report else {"ok": False}
    preflight = get_path(report, ["checks", "preflight"], {})
    result = get_path(report, ["checks", "result"], {})
    placement = report.get("planned_placement") if isinstance(report.get("planned_placement"), dict) else {}
    payload = placement.get("payload") if isinstance(placement.get("payload"), dict) else {}
    guardrails = report.get("guardrails") if isinstance(report.get("guardrails"), dict) else {}
    execution = report.get("execution") if isinstance(report.get("execution"), dict) else {}
    bounds = report.get("canary_bounds") if isinstance(report.get("canary_bounds"), dict) else {}
    runtime = train_report.get("runtime_profile") if isinstance(train_report.get("runtime_profile"), dict) else {}
    child_guardrails = train_report.get("guardrails") if isinstance(train_report.get("guardrails"), dict) else {}
    child_contract = train_report.get("report_contract") if isinstance(train_report.get("report_contract"), dict) else {}
    required_reports = canary_policy.get("required_reports") if isinstance(canary_policy.get("required_reports"), list) else []
    checks = {
        "canary_report_ok": bool(report.get("ok")),
        "policy_matches": report.get("policy") == "project_theseus_macos_metal_scheduler_canary_v0",
        "execute_mode": report.get("mode") == "execute",
        "execution_attempted": bool(execution.get("attempted")),
        "execution_ok": bool(execution.get("ok")) and int(execution.get("returncode") or 0) == 0,
        "canary_policy_present": bool(canary_policy),
        "canary_policy_matches": canary_policy.get("policy") == "project_theseus_macos_metal_scheduler_canary_policy_v0",
        "canary_policy_local_only": canary_policy.get("local_only") is True,
        "canary_policy_production_disabled": canary_policy.get("production_scheduler_routing_enabled") is False,
        "required_reports_declared": all(
            path in required_reports
            for path in [
                "reports/macos_metal_scheduler_dry_run.json",
                "reports/macos_metal_parity_ladder.json",
                "reports/macos_mlx_parity_audit.json",
            ]
        ),
        "preflight_all_ok": bool(preflight) and all(bool(value) for value in preflight.values()),
        "result_all_ok": bool(result) and all(bool(value) for value in result.values()),
        "placement_local_only": placement.get("target") == "local",
        "task_kind_is_canary": placement.get("task_kind") == "train_rollout_metal_local_canary",
        "task_kind_not_registered_worker_chunk": bool(preflight.get("canary_not_registered_worker_chunk")),
        "payload_command_matches": payload.get("command") == "train-rollout-metal",
        "payload_backend_matches": "apple_metal" in (payload.get("backend_requirements") if isinstance(payload.get("backend_requirements"), list) else []),
        "train_report_loaded": bool(train_report),
        "artifact_loaded": bool(artifact),
        "train_report_ok": bool(train_report.get("ok") and train_report.get("state") == "GREEN"),
        "train_report_command_matches": train_report.get("command") == "train-rollout-metal",
        "train_report_backend_matches": train_report.get("backend") == "apple_metal" and runtime.get("backend") == "apple_metal",
        "native_rust_owned": runtime.get("native_rust_owned") is True,
        "python_mlx_bridge_not_used": runtime.get("python_mlx_bridge_used") is False,
        "artifact_equivalence_ok": bool(artifact_equivalence.get("ok")),
        "kernel_launches_within_canary_cap": 0 < int(train_report.get("kernel_launches") or 0) <= int(bounds.get("max_kernel_launches") or 0),
        "tolerance_bounded": 0.0 < float(bounds.get("tolerance") or 0.0) <= 0.0005,
        "scheduler_routing_still_disabled": child_contract.get("scheduler_routing_enabled") is False
        and child_guardrails.get("does_not_route_scheduler_to_metal") is True
        and guardrails.get("production_scheduler_routing_enabled") is False,
        "remote_task_not_submitted": guardrails.get("remote_task_submitted") is False,
        "registers_worker_chunk_false": guardrails.get("registers_worker_chunk") is False,
        "promotion_still_locked": train_report.get("model_promotion_allowed") is False
        and guardrails.get("model_promotion_allowed") is False
        and get_path(train_report, ["promotion_decision", "promote_to_training_lane"], True) is False,
        "parity_claim_still_locked": train_report.get("train_rollout_parity_claim_allowed") is False
        and train_report.get("full_cli_parity_claim_allowed") is False
        and train_report.get("parity_claim_allowed") is False
        and guardrails.get("train_rollout_parity_claim_allowed") is False
        and guardrails.get("native_hot_loop_parity_claim_allowed") is False,
        "external_inference_zero": int(train_report.get("external_inference_calls") or 0) == 0
        and int(guardrails.get("external_inference_calls") or 0) == 0
        and child_guardrails.get("no_external_inference") is True,
        "teacher_disabled": train_report.get("teacher_used") is False
        and guardrails.get("teacher_used") is False
        and child_guardrails.get("no_teacher") is True,
        "public_training_zero": int(train_report.get("public_training_rows") or 0) == 0
        and guardrails.get("public_benchmark_training_used") is False
        and child_guardrails.get("no_public_training_rows") is True,
        "no_fallback_returns": train_report.get("symbolic_fallback") is False
        and guardrails.get("no_fallback_returns") is True
        and child_guardrails.get("no_fallback_returns") is True,
    }
    return {
        "ok": all(checks.values()),
        "path": rel(METAL_SCHEDULER_CANARY),
        "mode": report.get("mode"),
        "train_report": train_path,
        "artifact": artifact_path,
        "task_kind": placement.get("task_kind"),
        "target": placement.get("target"),
        "backend_requirements": payload.get("backend_requirements") if isinstance(payload.get("backend_requirements"), list) else [],
        "max_kernel_launches": bounds.get("max_kernel_launches"),
        "kernel_launches": train_report.get("kernel_launches"),
        "tolerance": bounds.get("tolerance"),
        "production_scheduler_routing_enabled": guardrails.get("production_scheduler_routing_enabled"),
        "remote_task_submitted": guardrails.get("remote_task_submitted"),
        "artifact_equivalence": artifact_equivalence,
        "checks": checks,
    }


def validate_token_superposition_metal_readout_proof(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {
            "ok": False,
            "path": rel(METAL_TOKEN_SUPERPOSITION_READOUT_PROOF),
            "reason": "missing_token_superposition_metal_readout_proof",
        }
    guardrails = report.get("guardrails") if isinstance(report.get("guardrails"), dict) else {}
    dataset = report.get("dataset") if isinstance(report.get("dataset"), dict) else {}
    parity = report.get("parity_metrics") if isinstance(report.get("parity_metrics"), dict) else {}
    variant = report.get("variant") if isinstance(report.get("variant"), dict) else {}
    tolerance = float(get_path(report, ["config", "tolerance"], 0.0) or 0.0)
    checks = {
        "proof_ok": bool(report.get("ok")),
        "policy_matches": report.get("policy")
        == "project_theseus_macos_metal_token_superposition_readout_proof_v0",
        "state_green": report.get("state") == "GREEN",
        "native_bag_trainer_present": report.get("native_bag_trainer") == "readout_bag_sgd_samples_kernel",
        "native_ar_trainer_present": report.get("native_ar_trainer") == "readout_sgd_samples_kernel",
        "native_readout_present": report.get("native_readout") == "linear_readout_logits_kernel",
        "private_synthetic_scope": report.get("parity_scope")
        == "bounded_private_synthetic_token_superposition_readout",
        "dataset_private_synthetic": dataset.get("source") == "deterministic_private_synthetic_tokens",
        "public_training_zero": int(report.get("public_training_rows") or 0) == 0
        and int(dataset.get("public_training_rows") or 0) == 0
        and guardrails.get("no_public_training_rows") is True,
        "public_calibration_not_run": guardrails.get("no_public_calibration") is True,
        "external_inference_zero": int(report.get("external_inference_calls") or 0) == 0
        and guardrails.get("no_external_inference") is True,
        "teacher_disabled": report.get("teacher_used") is False and guardrails.get("no_teacher") is True,
        "no_fallback_returns": guardrails.get("no_fallback_returns") is True
        and report.get("symbolic_fallback") is False,
        "promotion_locked": report.get("model_promotion_allowed") is False,
        "parity_claim_locked": report.get("train_token_superposition_parity_claim_allowed") is False
        and report.get("full_cli_parity_claim_allowed") is False
        and guardrails.get("does_not_claim_full_kernel_parity") is True
        and guardrails.get("does_not_claim_training_lane_parity") is True,
        "scheduler_route_locked": guardrails.get("does_not_route_scheduler_to_metal") is True,
        "tolerance_bounded": 0.0 < tolerance <= 0.0005001,
        "baseline_delta_bounded": float(parity.get("baseline_weight_max_abs_delta") or 0.0) <= tolerance
        and float(parity.get("baseline_bias_max_abs_delta") or 0.0) <= tolerance
        and float(parity.get("baseline_loss_delta") or 0.0) <= tolerance,
        "variant_delta_bounded": float(parity.get("variant_weight_max_abs_delta") or 0.0) <= tolerance
        and float(parity.get("variant_bias_max_abs_delta") or 0.0) <= tolerance
        and float(parity.get("variant_logits_max_abs_delta") or 0.0) <= tolerance
        and float(parity.get("variant_loss_delta") or 0.0) <= tolerance,
        "prediction_agreement_exact": float(parity.get("prediction_agreement") or 0.0) == 1.0
        and float(variant.get("prediction_agreement") or 0.0) == 1.0,
    }
    return {
        "ok": all(checks.values()),
        "path": rel(METAL_TOKEN_SUPERPOSITION_READOUT_PROOF),
        "native_path": report.get("native_path"),
        "parity_scope": report.get("parity_scope"),
        "tolerance": tolerance,
        "parity_metrics": parity,
        "guardrails": guardrails,
        "checks": checks,
    }


def validate_train_token_superposition_metal_contract(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {
            "ok": False,
            "path": rel(METAL_TOKEN_SUPERPOSITION_REPORT),
            "reason": "missing_train_token_superposition_metal_report",
        }
    guardrails = report.get("guardrails") if isinstance(report.get("guardrails"), dict) else {}
    contract = report.get("report_contract") if isinstance(report.get("report_contract"), dict) else {}
    dataset = report.get("dataset") if isinstance(report.get("dataset"), dict) else {}
    baseline = report.get("baseline") if isinstance(report.get("baseline"), dict) else {}
    variants = report.get("variants") if isinstance(report.get("variants"), list) else []
    best_variant = report.get("best_variant") if isinstance(report.get("best_variant"), dict) else {}
    work_receipt = report.get("work_receipt") if isinstance(report.get("work_receipt"), dict) else {}
    raw_promotion = report.get("raw_gate_promotion_decision") if isinstance(report.get("raw_gate_promotion_decision"), dict) else {}
    variant_kernel_launches = sum(
        int(row.get("kernel_launches") or 0)
        for row in variants
        if isinstance(row, dict)
    )
    kernel_launches = int(baseline.get("kernel_launches") or 0) + variant_kernel_launches
    checks = {
        "report_ok": bool(report.get("ok")),
        "policy_matches": report.get("policy") == "project_theseus_token_superposition_metal_report_v1",
        "command_matches": report.get("command") == "train-token-superposition-metal",
        "parity_for_matches": report.get("parity_for") == "train-token-superposition-cuda",
        "backend_matches": report.get("backend") == "apple_metal",
        "implementation_matches": report.get("implementation") == "rust_metal_token_superposition_readout_cli",
        "cuda_fallback_false": report.get("cuda_fallback") is False,
        "contract_matches_surface": contract.get("matches_train_token_superposition_cli_surface") is True,
        "mirrors_mlx_command": contract.get("mirrors_command") == "train-token-superposition-mlx",
        "scheduler_routing_disabled": contract.get("scheduler_routing_enabled") is False
        and guardrails.get("does_not_route_scheduler_to_metal") is True,
        "python_mlx_bridge_not_used": contract.get("python_mlx_bridge_used") is False,
        "native_readout_subpath_declared": bool(contract.get("native_readout_subpath")),
        "baseline_present": bool(baseline) and baseline.get("id") == "baseline_ar_metal",
        "variant_present": bool(variants) and bool(best_variant),
        "variant_ids_are_metal": all(
            isinstance(row, dict) and str(row.get("id") or "").endswith("_metal")
            for row in variants
        ),
        "kernel_launches_positive": kernel_launches > 0,
        "work_receipt_matches": work_receipt.get("backend") == "apple_metal"
        and work_receipt.get("task_kind") == "train_token_superposition_metal_cli"
        and int(work_receipt.get("claimed_work_units") or 0) == kernel_launches,
        "dataset_present": bool(dataset)
        and int(dataset.get("train_tokens") or 0) > 0
        and int(dataset.get("vocab_size") or 0) > 0,
        "public_training_zero": int(report.get("public_training_rows") or 0) == 0
        and guardrails.get("no_public_training_rows") is True,
        "public_calibration_not_run": guardrails.get("no_public_calibration") is True,
        "external_inference_zero": int(report.get("external_inference_calls") or 0) == 0
        and guardrails.get("no_external_inference") is True,
        "teacher_disabled": report.get("teacher_used") is False and guardrails.get("no_teacher") is True,
        "no_fallback_returns": guardrails.get("no_fallback_returns") is True,
        "promotion_locked": report.get("model_promotion_allowed") is False
        and get_path(report, ["promotion_decision", "promote_to_training_lane"], True) is False
        and guardrails.get("promotion_locked_by_macos_contract") is True,
        "raw_gate_decision_retained": bool(raw_promotion),
        "parity_claim_locked": report.get("train_token_superposition_parity_claim_allowed") is False
        and report.get("full_cli_parity_claim_allowed") is False
        and guardrails.get("does_not_claim_full_kernel_parity") is True
        and guardrails.get("does_not_claim_training_lane_parity") is True,
    }
    return {
        "ok": all(checks.values()),
        "path": rel(METAL_TOKEN_SUPERPOSITION_REPORT),
        "command": report.get("command"),
        "backend": report.get("backend"),
        "kernel_launches": kernel_launches,
        "variant_count": len(variants),
        "best_variant_id": best_variant.get("id"),
        "scheduler_routing_enabled": contract.get("scheduler_routing_enabled"),
        "model_promotion_allowed": report.get("model_promotion_allowed"),
        "raw_gate_promotion_status": raw_promotion.get("status"),
        "checks": checks,
    }


def validate_train_token_superposition_metal_ladder(
    report: dict[str, Any],
    route_policy: dict[str, Any],
    contract_validation: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {
            "ok": False,
            "path": rel(METAL_TOKEN_SUPERPOSITION_LADDER),
            "reason": "missing_train_token_superposition_metal_ladder",
        }
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    guardrails = report.get("guardrails") if isinstance(report.get("guardrails"), dict) else {}
    route = report.get("route_policy") if isinstance(report.get("route_policy"), dict) else {}
    route_checks = route.get("checks") if isinstance(route.get("checks"), dict) else {}
    policy_requires = route_policy.get("requires") if isinstance(route_policy.get("requires"), dict) else {}
    tiers = report.get("tiers") if isinstance(report.get("tiers"), list) else []
    tier_rows = [row for row in tiers if isinstance(row, dict)]
    tier_checks_ok = all(
        row.get("ok") is True
        and isinstance(row.get("checks"), dict)
        and all(bool(value) for value in row["checks"].values())
        for row in tier_rows
    )
    tier_inputs_private = all(
        row.get("checks", {}).get("private_inputs_only") is True
        and row.get("checks", {}).get("project_code_excluded") is True
        and row.get("checks", {}).get("code_docs_excluded") is True
        for row in tier_rows
        if isinstance(row.get("checks"), dict)
    )
    tier_kernel_bounds = all(
        0 < int(row.get("kernel_launches") or 0) <= int(get_path(row, ["config", "max_kernel_launches"], 0) or 0)
        for row in tier_rows
    )
    checks = {
        "report_ok": report.get("trigger_state") == "GREEN",
        "policy_matches": report.get("policy")
        == "project_theseus_macos_metal_token_superposition_ladder_v0",
        "execute_mode": report.get("execute") is True,
        "route_policy_present": route_policy.get("policy")
        == "project_theseus_macos_metal_token_superposition_route_policy_v0",
        "route_policy_validated": route.get("ok") is True
        and all(bool(value) for value in route_checks.values()),
        "route_policy_command_matches": route_policy.get("command") == "train-token-superposition-metal",
        "route_policy_guarded_ladder_only": route_policy.get("route_state") == "guarded_ladder_only",
        "route_policy_production_disabled": route_policy.get("production_scheduler_routing_enabled") is False
        and route.get("production_scheduler_routing_enabled") is False,
        "route_policy_requires_no_cheat": int(policy_requires.get("external_inference_calls", -1)) == 0
        and policy_requires.get("teacher_used") is False
        and int(policy_requires.get("public_training_rows", -1)) == 0
        and policy_requires.get("no_fallback_returns") is True
        and policy_requires.get("public_calibration_not_run") is True,
        "contract_report_ready": contract_validation.get("ok") is True,
        "tier_count_positive": int(summary.get("tier_count") or 0) >= 3
        and len(tier_rows) == int(summary.get("tier_count") or 0),
        "all_tiers_ok": int(summary.get("tier_ok_count") or 0) == int(summary.get("tier_count") or -1)
        and tier_checks_ok,
        "tier_inputs_private": tier_inputs_private,
        "tier_kernel_launches_bounded": tier_kernel_bounds,
        "total_kernel_launches_positive": int(summary.get("total_kernel_launches") or 0) > 0,
        "scheduler_routing_disabled": summary.get("production_scheduler_routing_enabled") is False
        and guardrails.get("production_scheduler_routing_enabled") is False,
        "remote_task_not_submitted": summary.get("remote_task_submitted") is False
        and guardrails.get("remote_task_submitted") is False,
        "model_promotion_locked": summary.get("model_promotion_allowed") is False
        and guardrails.get("model_promotion_allowed") is False,
        "parity_claim_locked": summary.get("train_token_superposition_parity_claim_allowed") is False
        and summary.get("native_hot_loop_parity_claim_allowed") is False
        and guardrails.get("train_token_superposition_parity_claim_allowed") is False
        and guardrails.get("native_hot_loop_parity_claim_allowed") is False,
        "external_inference_zero": int(summary.get("external_inference_calls") or 0) == 0
        and int(report.get("external_inference_calls") or 0) == 0
        and int(guardrails.get("external_inference_calls") or 0) == 0,
        "teacher_disabled": summary.get("teacher_used") is False and guardrails.get("teacher_used") is False,
        "public_training_zero": int(summary.get("public_training_rows") or 0) == 0,
        "fallback_returns_zero": int(summary.get("fallback_returns") or 0) == 0
        and guardrails.get("fallback_returns_allowed") is False,
        "public_calibration_not_run": guardrails.get("public_calibration_run") is False,
    }
    return {
        "ok": all(checks.values()),
        "path": rel(METAL_TOKEN_SUPERPOSITION_LADDER),
        "route_policy": rel(METAL_TOKEN_SUPERPOSITION_ROUTE_POLICY),
        "trigger_state": report.get("trigger_state"),
        "tier_count": summary.get("tier_count"),
        "tier_ok_count": summary.get("tier_ok_count"),
        "total_kernel_launches": summary.get("total_kernel_launches"),
        "max_kernel_launches": summary.get("max_kernel_launches"),
        "production_scheduler_routing_enabled": summary.get("production_scheduler_routing_enabled"),
        "model_promotion_allowed": summary.get("model_promotion_allowed"),
        "train_token_superposition_parity_claim_allowed": summary.get("train_token_superposition_parity_claim_allowed"),
        "native_hot_loop_parity_claim_allowed": summary.get("native_hot_loop_parity_claim_allowed"),
        "checks": checks,
    }


def validate_train_token_superposition_metal_scheduler_canary(
    report: dict[str, Any],
    route_policy: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {
            "ok": False,
            "path": rel(METAL_TOKEN_SUPERPOSITION_SCHEDULER_CANARY),
            "reason": "missing_train_token_superposition_scheduler_canary",
        }
    canary_policy = read_json(METAL_TOKEN_SUPERPOSITION_CANARY_POLICY, {})
    preflight = get_path(report, ["checks", "preflight"], {})
    result = get_path(report, ["checks", "result"], {})
    guardrails = report.get("guardrails") if isinstance(report.get("guardrails"), dict) else {}
    placement = report.get("planned_placement") if isinstance(report.get("planned_placement"), dict) else {}
    train_path = report.get("train_report") if isinstance(report.get("train_report"), str) else ""
    artifact_path = report.get("artifact") if isinstance(report.get("artifact"), str) else ""
    train_report = read_json(resolve_repo_path(train_path), {}) if train_path else {}
    artifact = read_json(resolve_repo_path(artifact_path), {}) if artifact_path else {}
    contract_validation = validate_train_token_superposition_metal_contract(train_report)
    artifact_validation = validate_train_token_superposition_metal_artifact(train_report)
    bounds = canary_policy.get("bounds") if isinstance(canary_policy.get("bounds"), dict) else {}
    requires = canary_policy.get("requires") if isinstance(canary_policy.get("requires"), dict) else {}
    kernel_launches = int(get_path(train_report, ["metrics", "kernel_launches"], 0) or 0)
    checks = {
        "canary_report_ok": bool(report.get("ok")),
        "policy_matches": report.get("policy")
        == "project_theseus_macos_metal_token_superposition_scheduler_canary_v0",
        "execute_mode": report.get("mode") == "execute",
        "execution_attempted": get_path(report, ["execution", "attempted"], False) is True,
        "execution_ok": get_path(report, ["execution", "ok"], False) is True,
        "canary_policy_present": bool(canary_policy),
        "canary_policy_matches": canary_policy.get("policy")
        == "project_theseus_macos_metal_token_superposition_scheduler_canary_policy_v0",
        "route_policy_present": bool(route_policy),
        "route_policy_matches": route_policy.get("policy")
        == "project_theseus_macos_metal_token_superposition_route_policy_v0",
        "command_matches": canary_policy.get("command") == "train-token-superposition-metal",
        "task_kind_matches": placement.get("task_kind") == "train_token_superposition_metal_local_canary"
        and canary_policy.get("task_kind") == "train_token_superposition_metal_local_canary",
        "placement_local_only": placement.get("target") == "local",
        "preflight_all_ok": isinstance(preflight, dict) and all(bool(value) for value in preflight.values()),
        "result_all_ok": isinstance(result, dict) and all(bool(value) for value in result.values()),
        "required_reports_declared": bool(preflight.get("required_reports_declared")),
        "route_policy_production_disabled": route_policy.get("production_scheduler_routing_enabled") is False
        and canary_policy.get("production_scheduler_routing_enabled") is False,
        "remote_task_not_submitted": guardrails.get("remote_task_submitted") is False
        and canary_policy.get("remote_task_submitted") is False,
        "registers_worker_chunk_false": guardrails.get("registers_worker_chunk") is False
        and canary_policy.get("registers_worker_chunk") is False,
        "train_report_loaded": bool(train_report),
        "artifact_loaded": bool(artifact),
        "child_contract_ok": bool(contract_validation.get("ok")),
        "child_artifact_equivalence_ok": bool(artifact_validation.get("ok")),
        "kernel_launches_within_canary_cap": 0 < kernel_launches <= int(bounds.get("max_kernel_launches") or 0),
        "scheduler_routing_still_disabled": get_path(train_report, ["report_contract", "scheduler_routing_enabled"], True)
        is False
        and guardrails.get("production_scheduler_routing_enabled") is False,
        "promotion_still_locked": guardrails.get("model_promotion_allowed") is False
        and train_report.get("model_promotion_allowed") is False
        and get_path(train_report, ["promotion_decision", "promote_to_training_lane"], True) is False,
        "parity_claim_still_locked": guardrails.get("train_token_superposition_parity_claim_allowed") is False
        and guardrails.get("native_hot_loop_parity_claim_allowed") is False
        and train_report.get("train_token_superposition_parity_claim_allowed") is False
        and train_report.get("full_cli_parity_claim_allowed") is False,
        "external_inference_zero": int(guardrails.get("external_inference_calls") or 0) == 0
        and int(train_report.get("external_inference_calls") or 0) == 0
        and int(requires.get("external_inference_calls") or 0) == 0,
        "teacher_disabled": guardrails.get("teacher_used") is False
        and train_report.get("teacher_used") is False
        and requires.get("teacher_used") is False,
        "public_training_zero": int(train_report.get("public_training_rows") or 0) == 0
        and int(requires.get("public_training_rows") or 0) == 0,
        "no_fallback_returns": guardrails.get("no_fallback_returns") is True
        and get_path(train_report, ["guardrails", "no_fallback_returns"], False) is True
        and requires.get("no_fallback_returns") is True,
        "public_calibration_not_run": guardrails.get("public_calibration_run") is False
        and get_path(train_report, ["guardrails", "no_public_calibration"], False) is True
        and requires.get("public_calibration_not_run") is True,
    }
    return {
        "ok": all(checks.values()),
        "path": rel(METAL_TOKEN_SUPERPOSITION_SCHEDULER_CANARY),
        "canary_policy": rel(METAL_TOKEN_SUPERPOSITION_CANARY_POLICY),
        "route_policy": rel(METAL_TOKEN_SUPERPOSITION_ROUTE_POLICY),
        "mode": report.get("mode"),
        "train_report": train_path,
        "artifact": artifact_path,
        "task_kind": placement.get("task_kind"),
        "target": placement.get("target"),
        "kernel_launches": kernel_launches,
        "production_scheduler_routing_enabled": guardrails.get("production_scheduler_routing_enabled"),
        "remote_task_submitted": guardrails.get("remote_task_submitted"),
        "artifact_equivalence": artifact_validation,
        "checks": checks,
    }


def worker_evidence(task_kind: str, work_proof: dict[str, Any]) -> dict[str, Any]:
    rows = work_proof.get("worker_smokes") if isinstance(work_proof.get("worker_smokes"), list) else []
    for row in rows:
        if not isinstance(row, dict) or row.get("task_kind") != task_kind:
            continue
        return {
            "ok": bool(row.get("ok")),
            "backend": row.get("backend"),
            "report_path": row.get("report_path"),
            "metrics": evidence_metrics(row.get("metrics") if isinstance(row.get("metrics"), dict) else {}),
            "work_receipt": row.get("work_receipt") if isinstance(row.get("work_receipt"), dict) else {},
        }
    direct_path = WORKER_REPORTS.get(task_kind)
    direct = read_json(direct_path, {}) if direct_path else {}
    if direct:
        return {
            "ok": bool(direct.get("ok")),
            "backend": direct.get("backend"),
            "report_path": str(direct_path.relative_to(ROOT)) if direct_path else "",
            "metrics": evidence_metrics(direct.get("metrics") if isinstance(direct.get("metrics"), dict) else {}),
            "work_receipt": direct.get("work_receipt") if isinstance(direct.get("work_receipt"), dict) else {},
            "source": "direct_latest_proof_file",
        }
    return {"ok": False, "missing": True, "reason": "no_latest_worker_smoke_in_reports/macos_mlx_work_proof.json"}


def cli_evidence(command: str, parity_for: str, work_proof: dict[str, Any]) -> dict[str, Any]:
    rows = work_proof.get("cli_smokes") if isinstance(work_proof.get("cli_smokes"), list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("command_name") != command and row.get("parity_for") != parity_for:
            continue
        return {
            "ok": bool(row.get("ok")),
            "backend": row.get("backend"),
            "implementation": row.get("implementation"),
            "report_path": row.get("report_path"),
            "child_report_path": row.get("child_report_path"),
            "metrics": evidence_metrics(row.get("metrics") if isinstance(row.get("metrics"), dict) else {}),
            "child": row.get("child") if isinstance(row.get("child"), dict) else {},
            "promotion_decision": row.get("promotion_decision") if isinstance(row.get("promotion_decision"), dict) else {},
        }
    direct_path = CLI_REPORTS.get(command)
    direct = read_json(direct_path, {}) if direct_path else {}
    if direct:
        child = direct.get("child_report") if isinstance(direct.get("child_report"), dict) else {}
        return {
            "ok": bool(direct.get("ok")),
            "backend": direct.get("backend"),
            "implementation": direct.get("implementation"),
            "report_path": str(direct_path.relative_to(ROOT)) if direct_path else "",
            "child_report_path": direct.get("child_report_path"),
            "metrics": evidence_metrics(direct.get("metrics") if isinstance(direct.get("metrics"), dict) else {}),
            "child": child,
            "promotion_decision": direct.get("promotion_decision") if isinstance(direct.get("promotion_decision"), dict) else {},
            "source": "direct_latest_proof_file",
        }
    return {"ok": False, "missing": True, "reason": "no_latest_cli_smoke_in_reports/macos_mlx_work_proof.json"}


def evidence_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "train_accuracy",
        "eval_accuracy",
        "train_return_proxy",
        "eval_return_proxy",
        "loss_initial",
        "loss_final",
        "examples_per_second",
        "mlx_train_ms",
        "mlx_eval_ms",
        "train_rows",
        "eval_rows",
        "train_cases",
        "eval_cases",
    ]
    return {key: metrics.get(key) for key in keep if key in metrics}


def compact_work_proof(work_proof: dict[str, Any]) -> dict[str, Any]:
    if not work_proof:
        return {"ok": False, "missing": True, "path": str(WORK_PROOF.relative_to(ROOT))}
    return {
        "ok": bool(work_proof.get("ok")),
        "state": work_proof.get("state"),
        "path": str(WORK_PROOF.relative_to(ROOT)),
        "created_utc": work_proof.get("created_utc"),
        "summary": work_proof.get("summary") if isinstance(work_proof.get("summary"), dict) else {},
    }


def command_present(command: str, text: str) -> bool:
    if not command:
        return False
    camel = "".join(part.capitalize() for part in command.split("-"))
    return command in text or camel in text


def mlx_status() -> dict[str, Any]:
    runtimes = [
        {"name": "active_python", "python": Path(sys.executable)},
        {"name": "source_venv", "python": ROOT / ".venv-puffer" / "bin" / "python"},
        {
            "name": "installed_app_venv",
            "python": Path.home()
            / "Library"
            / "Application Support"
            / "Project Theseus Hive"
            / "app"
            / "current"
            / ".venv-puffer"
            / "bin"
            / "python",
        },
    ]
    checks = [mlx_runtime_check(row["name"], row["python"]) for row in runtimes]
    available = any(row.get("available") for row in checks)
    preferred = next((row for row in checks if row.get("available")), checks[0] if checks else {})
    return {
        "available": available,
        "module": "mlx.core",
        "preferred_runtime": preferred.get("name"),
        "runtimes": checks,
    }


def mlx_runtime_check(name: str, python: Path) -> dict[str, Any]:
    if not python.exists():
        return {"name": name, "python": str(python), "available": False, "error": "python_missing"}
    code = "import json, mlx.core as mx; x=mx.array([1.0,2.0]); mx.eval(x); print(json.dumps({'available': True, 'probe': [float(v) for v in x.tolist()]}))"
    try:
        result = subprocess.run([str(python), "-c", code], cwd=ROOT, text=True, capture_output=True, timeout=30)
    except Exception as exc:  # noqa: BLE001 - diagnostics only.
        return {"name": name, "python": str(python), "available": False, "error": type(exc).__name__, "message": str(exc)}
    if result.returncode != 0:
        return {
            "name": name,
            "python": str(python),
            "available": False,
            "returncode": result.returncode,
            "stderr_tail": result.stderr[-500:],
        }
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        payload = {"available": True, "stdout_tail": result.stdout[-500:]}
    return {"name": name, "python": str(python), **payload}


def next_targets(task_rows: list[dict[str, Any]], cli_rows: list[dict[str, Any]], mlx: dict[str, Any]) -> list[dict[str, str]]:
    targets = []
    accelerator_manifest = read_json(ACCELERATOR_PARITY_MANIFEST, {})
    accelerator_manifest_ready = bool(
        accelerator_manifest.get("ok")
        and accelerator_manifest.get("trigger_state") == "GREEN"
        and int_value(accelerator_manifest, ["summary", "hard_failure_count"], -1) == 0
        and int_value(accelerator_manifest, ["summary", "explicit_guardrail_gap_count"], -1) == 0
        and int_value(accelerator_manifest, ["summary", "external_inference_calls"], -1) == 0
        and int_value(accelerator_manifest, ["summary", "public_training_rows"], -1) == 0
        and int_value(accelerator_manifest, ["summary", "model_promotion_allowed_count"], -1) == 0
        and int_value(accelerator_manifest, ["summary", "production_routing_enabled_count"], -1) == 0
    )
    metal_contract = read_json(METAL_TRAIN_ROLLOUT_REPORT, {})
    metal_contract_ready = bool(
        metal_contract.get("ok")
        and metal_contract.get("command") == "train-rollout-metal"
        and get_path(metal_contract, ["report_contract", "matches_train_rollout_cli_surface"], False)
    )
    metal_artifact = validate_train_rollout_metal_artifact(metal_contract)
    metal_artifact_ready = bool(metal_artifact.get("ok"))
    metal_route_guard_ready = bool(validate_train_rollout_metal_route_policy(metal_contract, metal_artifact).get("ok"))
    metal_scheduler_dry_run_ready = bool(validate_train_rollout_metal_scheduler_dry_run(read_json(METAL_SCHEDULER_DRY_RUN, {})).get("ok"))
    metal_parity_ladder_ready = bool(validate_train_rollout_metal_parity_ladder(read_json(METAL_PARITY_LADDER, {})).get("ok"))
    metal_scheduler_canary_ready = bool(validate_train_rollout_metal_scheduler_canary(read_json(METAL_SCHEDULER_CANARY, {})).get("ok"))
    token_superposition_readout_ready = bool(
        validate_token_superposition_metal_readout_proof(
            read_json(METAL_TOKEN_SUPERPOSITION_READOUT_PROOF, {})
        ).get("ok")
    )
    token_superposition_contract_ready = bool(
        validate_train_token_superposition_metal_contract(
            read_json(METAL_TOKEN_SUPERPOSITION_REPORT, {})
        ).get("ok")
    )
    token_superposition_artifact_ready = bool(
        validate_train_token_superposition_metal_artifact(
            read_json(METAL_TOKEN_SUPERPOSITION_REPORT, {})
        ).get("ok")
    )
    token_superposition_ladder_ready = bool(
        validate_train_token_superposition_metal_ladder(
            read_json(METAL_TOKEN_SUPERPOSITION_LADDER, {}),
            read_json(METAL_TOKEN_SUPERPOSITION_ROUTE_POLICY, {}),
            validate_train_token_superposition_metal_contract(read_json(METAL_TOKEN_SUPERPOSITION_REPORT, {})),
        ).get("ok")
    )
    token_superposition_scheduler_canary_ready = bool(
        validate_train_token_superposition_metal_scheduler_canary(
            read_json(METAL_TOKEN_SUPERPOSITION_SCHEDULER_CANARY, {}),
            read_json(METAL_TOKEN_SUPERPOSITION_ROUTE_POLICY, {}),
        ).get("ok")
    )
    if not mlx.get("available"):
        targets.append({"priority": "P0", "target": "Install/repair MLX in the active Mac runtime", "why": "Apple Silicon cannot run native MLX chunks without it."})
    if any(row.get("mlx") == "mlx_rollout_chunk" and row.get("status") == "missing" for row in task_rows):
        targets.append({"priority": "P1", "target": "Add mlx_rollout_chunk", "why": "This is the largest CUDA-only Hive worker gap."})
    if any(row.get("cuda") == "train-rollout-cuda" and row.get("kernel_port_status") == PENDING_KERNEL_STATUS for row in cli_rows):
        if metal_scheduler_canary_ready:
            targets.append({"priority": "P1", "target": "Design operator-reviewed train-rollout-metal production-route gate or port the next CUDA hot loop", "why": "The reviewed local Metal scheduler canary is proven with no-cheat locks intact. Production routing and full parity claims remain locked until a separate operator-reviewed route gate exists."})
        elif metal_parity_ladder_ready:
            targets.append({"priority": "P1", "target": "Prepare reviewed train-rollout-metal scheduler canary or climb a larger bounded ladder", "why": "The local bounded Metal ladder is proven with no-cheat locks intact, but production routing and full parity claims remain locked until a reviewed canary policy exists."})
        elif metal_scheduler_dry_run_ready:
            targets.append({"priority": "P1", "target": "Climb bounded train-rollout-metal parity ladder", "why": "The guarded scheduler dry-run is proven; before any production route, run larger private bounded sizes with the same no-cheat locks and compare against CPU/CUDA-equivalent report contracts."})
        elif metal_route_guard_ready:
            targets.append({"priority": "P1", "target": "Run bounded scheduler dry-run for train-rollout-metal", "why": "The Rust/Metal CLI contract, canonical artifact, and route guard exist; production routing remains locked until a dry-run proves placement, artifact, and rollback behavior."})
        elif metal_artifact_ready:
            targets.append({"priority": "P1", "target": "Add explicit scheduler routing guardrails for train-rollout-metal", "why": "The Rust/Metal CLI contract and canonical readout artifact are proven, but production routing still needs backend, artifact, rollback, and no-promotion safeguards."})
        elif metal_contract_ready:
            targets.append({"priority": "P1", "target": "Prove train-rollout-metal checkpoint/artifact equivalence", "why": "The Rust/Metal CLI contract exists, but production-compatible artifact validation is still pending."})
        else:
            targets.append({"priority": "P1", "target": "Port rollout hot loop from Python MLX bridge to Rust/Metal or Rust/MLX", "why": "The user-facing MLX command exists now, but native kernel parity is still pending."})
    if any(row.get("cuda") == "train-token-superposition-cuda" and row.get("kernel_port_status") == PENDING_KERNEL_STATUS for row in cli_rows):
        if token_superposition_scheduler_canary_ready:
            targets.append({"priority": "P2", "target": "Keep train-token-superposition-metal production routing locked until full native parity review or port another CUDA hot loop", "why": "The reviewed local token-superposition Metal scheduler canary is proven with no-cheat locks intact. Production routing and full parity claims remain intentionally pending until an explicit operator-reviewed enablement step."})
        elif token_superposition_ladder_ready and token_superposition_artifact_ready:
            targets.append({"priority": "P2", "target": "Prepare reviewed train-token-superposition-metal scheduler canary", "why": "The full Rust/Metal token-superposition CLI contract, canonical artifact-equivalence proof, and bounded private ladder are proven with no-cheat locks, but production routing and full parity remain intentionally pending."})
        elif token_superposition_ladder_ready:
            targets.append({"priority": "P2", "target": "Design operator-reviewed train-token-superposition-metal route gate or add artifact-equivalence proof", "why": "The full Rust/Metal token-superposition CLI contract and bounded private ladder are proven with no-cheat locks, but production routing and full parity remain intentionally pending."})
        elif token_superposition_contract_ready:
            targets.append({"priority": "P2", "target": "Add bounded train-token-superposition-metal ladder and route guardrails", "why": "The full Rust/Metal token-superposition CLI contract is proven with no-cheat locks, but production routing and full parity remain intentionally pending."})
        elif token_superposition_readout_ready:
            targets.append({"priority": "P2", "target": "Build full train-token-superposition-metal CLI contract/report", "why": "The bounded Rust/Metal token-superposition readout subpath is proven with no-cheat locks, but the full CLI contract and scheduler route remain intentionally pending."})
        else:
            targets.append({"priority": "P2", "target": "Port token-superposition readout from Python MLX bridge to Rust/Metal or Rust/MLX", "why": "The MLX command is real, but the deeper Rust/CUDA fast path still needs a Mac-native kernel."})
    production_locked = [
        row.get("cuda")
        for row in cli_rows
        if row.get("kernel_parity_ready") and not row.get("production_route_ready")
    ]
    if production_locked:
        production_readiness = validate_metal_production_route_readiness(
            read_json(METAL_PRODUCTION_ROUTE_READINESS, {})
        )
        if production_readiness.get("ok"):
            target = "Resolve macOS Metal production-route readiness blockers while keeping scheduler routing disabled"
            why = (
                "The dedicated production-route readiness review is present and fail-closed. "
                f"It reports {production_readiness.get('production_route_ready_count')} production-ready routes and "
                f"{production_readiness.get('blocker_count')} blockers across "
                + ", ".join(str(name) for name in production_locked)
                + "."
            )
        else:
            target = "Prepare a separate macOS Metal production-route readiness review"
            why = (
                "Guarded native evidence exists for "
                + ", ".join(str(name) for name in production_locked)
                + ", but production scheduler routing and full parity claims are still intentionally locked."
            )
        targets.append(
            {
                "priority": "P1",
                "target": target,
                "why": why,
            }
        )
    pending_cuda = [
        row.get("cuda")
        for row in cli_rows
        if row.get("kernel_port_status") == PENDING_KERNEL_STATUS
    ]
    if pending_cuda:
        targets.append(
            {
                "priority": "P1",
                "target": "Port remaining pending Mac-native CLI hot loops",
                "why": "Still pending Rust/Metal or Rust/MLX parity rows: " + ", ".join(str(name) for name in pending_cuda) + ".",
            }
        )
    if any(not get_path(row, ["latest_evidence", "ok"], False) for row in task_rows + cli_rows):
        targets.append({"priority": "P2", "target": "Refresh runnable MLX parity evidence", "why": "The audit now distinguishes static command coverage from live MLX worker/CLI proof rows."})
    if not accelerator_manifest_ready:
        targets.append({"priority": "P3", "target": "Add benchmark parity reports", "why": "Every CUDA/MLX pair should emit comparable metrics and artifact manifests."})
    return targets


def accelerator_parity_manifest_ready() -> bool:
    report = read_json(ACCELERATOR_PARITY_MANIFEST, {})
    return bool(
        report.get("ok")
        and report.get("trigger_state") == "GREEN"
        and int_value(report, ["summary", "surface_ok_count"], 0)
        == int_value(report, ["summary", "surface_count"], -1)
        and int_value(report, ["summary", "hard_failure_count"], -1) == 0
        and int_value(report, ["summary", "explicit_guardrail_gap_count"], -1) == 0
        and int_value(report, ["summary", "external_inference_calls"], -1) == 0
        and int_value(report, ["summary", "teacher_used_count"], -1) == 0
        and int_value(report, ["summary", "public_training_rows"], -1) == 0
        and int_value(report, ["summary", "model_promotion_allowed_count"], -1) == 0
        and int_value(report, ["summary", "production_routing_enabled_count"], -1) == 0
    )


def validate_metal_production_route_readiness(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {"ok": False, "reason": "missing_macos_metal_production_route_readiness"}
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    guardrails = report.get("guardrails") if isinstance(report.get("guardrails"), dict) else {}
    surfaces = report.get("surfaces") if isinstance(report.get("surfaces"), list) else []
    checks = {
        "policy_matches": report.get("policy") == "project_theseus_macos_metal_production_route_readiness_v0",
        "state_yellow_or_green": report.get("trigger_state") in {"YELLOW", "GREEN"},
        "report_ok": report.get("ok") is True,
        "surfaces_present": bool(surfaces),
        "all_guarded_evidence_ok": int_value(report, ["summary", "guarded_evidence_ok_count"], 0)
        == int_value(report, ["summary", "surface_count"], -1),
        "hard_failures_zero": int_value(report, ["summary", "hard_failure_count"], -1) == 0,
        "production_route_allowed_false": summary.get("production_route_allowed") is False,
        "production_route_ready_zero": int_value(report, ["summary", "production_route_ready_count"], -1) == 0,
        "operator_approval_invalid": summary.get("operator_approval_valid") is False,
        "external_inference_zero": int_value(report, ["summary", "external_inference_calls"], -1) == 0,
        "teacher_zero": int_value(report, ["summary", "teacher_used_count"], -1) == 0,
        "public_training_zero": int_value(report, ["summary", "public_training_rows"], -1) == 0,
        "promotion_zero": int_value(report, ["summary", "model_promotion_allowed_count"], -1) == 0,
        "guardrail_no_scheduler_enable": guardrails.get("does_not_enable_scheduler_routing") is True,
        "guardrail_no_remote_scope_change": guardrails.get("does_not_change_remote_task_scope") is True,
        "guardrail_no_arbitrary_remote_execution": guardrails.get("no_arbitrary_remote_execution") is True,
    }
    return {
        "ok": all(checks.values()),
        "path": rel(METAL_PRODUCTION_ROUTE_READINESS),
        "trigger_state": report.get("trigger_state"),
        "production_route_allowed": summary.get("production_route_allowed"),
        "production_route_ready_count": summary.get("production_route_ready_count"),
        "guarded_evidence_ok_count": summary.get("guarded_evidence_ok_count"),
        "surface_count": summary.get("surface_count"),
        "blocker_count": summary.get("blocker_count"),
        "blockers": report.get("blockers") if isinstance(report.get("blockers"), list) else [],
        "checks": checks,
    }




if __name__ == "__main__":
    raise SystemExit(main())
