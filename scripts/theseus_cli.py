"""Project Theseus Hive command line interface.

This is the terminal/server control surface for the same governed Hive runtime
used by the setup wizard and dashboard. It intentionally delegates to the
existing profile, invite, scheduler, node, checkpoint-chat, and public
contribution modules instead of creating a second control plane.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request as urlrequest
from urllib.error import URLError


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs"
POLICY_PATH = CONFIGS / "hive_policy.json"

sys.path.insert(0, str(ROOT / "scripts"))
import hive_profiles  # noqa: E402
import hive_usb_writer  # noqa: E402
import hive_remote_access  # noqa: E402
import hive_network_doctor  # noqa: E402
import hive_remote_control  # noqa: E402
import hive_storage  # noqa: E402
import hive_training_orchestrator  # noqa: E402
import hive_solo_learning_loop  # noqa: E402
import full_training_teacher_preflight  # noqa: E402
import hive_bootstrap  # noqa: E402
import hive_rented_compute  # noqa: E402
import hive_utilization_manager  # noqa: E402
import hive_voice_following  # noqa: E402
import hive_spatial  # noqa: E402
import hive_users  # noqa: E402
import license_manager  # noqa: E402
import compute_market  # noqa: E402
import openai_compat_server  # noqa: E402
import public_hive_contributor  # noqa: E402
import theseus_runtime  # noqa: E402
import theseus_setup_wizard  # noqa: E402
import update_manager  # noqa: E402
import hive_version_manager  # noqa: E402
import hive_macos_canary  # noqa: E402
from theseus_cli_compact import (  # noqa: E402
    compact_benchmarks,
    compact_candidate,
    compact_dashboard,
    compact_license,
    compact_market,
    compact_node,
    compact_openai,
    compact_peers,
    compact_probe,
    compact_public,
    compact_remote,
    compact_remote_control,
    compact_rented_compute,
    compact_runtime,
    compact_scheduler,
    compact_updates,
    compact_utilization,
    compact_voice,
)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    json_output = "--json" in raw_argv
    raw_argv = [item for item in raw_argv if item != "--json"]

    parser = argparse.ArgumentParser(
        prog="theseus",
        description="Project Theseus Hive CLI for install, status, chat, task routing, and device registration.",
    )
    sub = parser.add_subparsers(dest="command")

    install = sub.add_parser("install", help="Install a user-level theseus CLI wrapper and probe this node.")
    install.add_argument("--target-dir", default="")
    install.add_argument("--skip-probe", action="store_true")

    sub.add_parser("status", help="Show dashboard, Hive, resource, benchmark, and profile status.")

    benchmark = sub.add_parser("benchmark", help="Inspect or run governed benchmark measurement without training contamination.")
    benchmark_sub = benchmark.add_subparsers(dest="benchmark_command")
    benchmark_public = benchmark_sub.add_parser("public", help="Show latest public-transfer score and plan the next fresh measurement surface.")
    benchmark_public.add_argument("--cards", default="source_mbpp,source_evalplus,source_bigcodebench,source_human_eval,source_livecodebench")
    benchmark_public.add_argument("--cases-per-card", type=int, default=64)
    benchmark_public.add_argument("--seed-start", type=int, default=1)
    benchmark_public.add_argument("--seed-end", type=int, default=512)
    benchmark_public.add_argument("--slug-prefix", default="public_transfer_measurement")
    benchmark_public.add_argument("--refresh-capacity", action="store_true")
    benchmark_public.add_argument("--materialize", action="store_true")
    benchmark_public.add_argument("--diagnostic-if-needed", action=argparse.BooleanOptionalAction, default=True)
    benchmark_public.add_argument("--execute", action="store_true")
    benchmark_public.add_argument("--timeout-seconds", type=int, default=7200)
    benchmark_public.add_argument("--out", default="reports/theseus_benchmark_measurement.json")
    benchmark_public.add_argument("--markdown-out", default="reports/theseus_benchmark_measurement.md")

    runtime = sub.add_parser("runtime", help="Configure generated runtime paths and spillover storage.")
    runtime_sub = runtime.add_subparsers(dest="runtime_command")
    runtime_status = runtime_sub.add_parser("status", help="Show runtime path configuration.")
    runtime_status.add_argument("--create", action="store_true")
    runtime_doctor = runtime_sub.add_parser("doctor", help="Diagnose Mac/source/app Python runtime, MLX, LaunchAgents, and runtime roots.")
    runtime_doctor.add_argument("--out", default="reports/macos_runtime_doctor.json")
    runtime_init = runtime_sub.add_parser("init", help="Write local runtime path config.")
    runtime_init.add_argument("--runtime-root", default="")
    runtime_init.add_argument("--no-create", action="store_true")
    runtime_migrate = runtime_sub.add_parser("migrate-junctions", help="Move ignored generated directories to runtime root and leave junctions/symlinks.")
    runtime_migrate.add_argument("--runtime-root", default="")
    runtime_migrate.add_argument("--dirs", default="reports,checkpoints,target")
    runtime_migrate.add_argument("--dry-run", action="store_true")

    mac = sub.add_parser("mac", help="Run macOS canary, training preflight, DMG readiness, role, and join-bundle gates.")
    mac_sub = mac.add_subparsers(dest="mac_command")
    mac_roles = mac_sub.add_parser("roles", help="Assign and verify this Mac's Hive roles.")
    mac_roles.add_argument("--write-local-config", action="store_true")
    mac_roles.add_argument("--out", default="reports/macos_role_assignment.json")
    mac_roles.add_argument("--markdown-out", default="")
    mac_preflight = mac_sub.add_parser("training-preflight", help="Gate long Mac training and optionally queue one local worker proof.")
    mac_preflight.add_argument("--execute", action="store_true")
    mac_preflight.add_argument("--profile", default="smoke")
    mac_preflight.add_argument("--timeout", type=float, default=90.0)
    mac_preflight.add_argument("--min-disk-gib", type=float, default=25.0)
    mac_preflight.add_argument("--min-battery-percent", type=float, default=45.0)
    mac_preflight.add_argument("--allow-battery-smoke", action="store_true")
    mac_preflight.add_argument("--offline", action="store_true")
    mac_preflight.add_argument("--out", default="reports/macos_training_preflight.json")
    mac_preflight.add_argument("--markdown-out", default="")
    mac_dmg = mac_sub.add_parser("dmg-readiness", help="Build/check DMG, package, version catalog, and installed-app update readiness.")
    mac_dmg.add_argument("--execute", action="store_true")
    mac_dmg.add_argument("--skip-build", action="store_true")
    mac_dmg.add_argument("--skip-version-publish", action="store_true")
    mac_dmg.add_argument("--api-url", default="http://127.0.0.1:8791")
    mac_dmg.add_argument("--out", default="reports/macos_dmg_readiness_gate.json")
    mac_dmg.add_argument("--markdown-out", default="")
    mac_join = mac_sub.add_parser("join-bundle", help="Write one-click Mac/iPhone/Watch join artifacts.")
    mac_join.add_argument("--bundle-out", default="dist/macos/ProjectTheseusHive.join.json")
    mac_join.add_argument("--qr-out", default="dist/macos/ProjectTheseusHive.join.svg")
    mac_join.add_argument("--no-token", action="store_true")
    mac_join.add_argument("--coordinator-url", action="append", default=[])
    mac_join.add_argument("--relay-url", action="append", default=[])
    mac_join.add_argument("--operator-token-scope", default="")
    mac_join.add_argument("--out", default="reports/macos_join_bundle_status.json")
    mac_join.add_argument("--markdown-out", default="")
    mac_app = mac_sub.add_parser("app-status", help="Show Joined / Running / Update OK / Training Ready from the installed app/API.")
    mac_app.add_argument("--api-url", default="http://127.0.0.1:8791")
    mac_app.add_argument("--text", action="store_true")
    mac_app.add_argument("--out", default="reports/macos_app_install_status.json")
    mac_app.add_argument("--markdown-out", default="")
    mac_canary = mac_sub.add_parser("canary", help="Run Mac role, app, training, DMG, and join-bundle gates together.")
    mac_canary.add_argument("--execute", action="store_true")
    mac_canary.add_argument("--write-local-config", action="store_true")
    mac_canary.add_argument("--skip-build", action="store_true")
    mac_canary.add_argument("--skip-version-publish", action="store_true")
    mac_canary.add_argument("--skip-training-execute", action="store_true")
    mac_canary.add_argument("--write-join-bundle", action="store_true")
    mac_canary.add_argument("--api-url", default="http://127.0.0.1:8791")
    mac_canary.add_argument("--out", default="reports/macos_canary.json")
    mac_canary.add_argument("--markdown-out", default="")

    cuda = sub.add_parser("cuda", help="Inspect Windows/NVIDIA CUDA readiness.")
    cuda_sub = cuda.add_subparsers(dest="cuda_command")
    cuda_doctor = cuda_sub.add_parser("doctor", help="Run the Windows/CUDA node doctor.")
    cuda_doctor.add_argument("--profile", default="smoke")
    cuda_doctor.add_argument("--refresh", action="store_true")
    cuda_doctor.add_argument("--refresh-preflight", action="store_true")
    cuda_doctor.add_argument("--out", default="reports/windows_cuda_doctor.json")
    cuda_doctor.add_argument("--markdown-out", default="reports/windows_cuda_doctor.md")

    license_cmd = sub.add_parser("license", help="Register this install, import a license, and inspect feature gates.")
    license_sub = license_cmd.add_subparsers(dest="license_command")
    license_sub.add_parser("status", help="Show local registration/license status.")
    license_register = license_sub.add_parser("register", help="Register this local install for free homelab/research or paid-license request.")
    license_register.add_argument("--name", default="")
    license_register.add_argument("--email", default="")
    license_register.add_argument("--organization", default="")
    license_register.add_argument("--usage", choices=["personal_homelab", "research", "startup_free", "company", "public_operator"], default="personal_homelab")
    license_register.add_argument("--seats", type=int, default=1)
    license_register.add_argument("--commercial", action="store_true")
    license_register.add_argument("--accept-terms", action="store_true")
    license_request = license_sub.add_parser("request", help="Create a license request bundle for a paid key.")
    license_request.add_argument("--feature", action="append", default=[])
    license_request.add_argument("--out", default="")
    license_import = license_sub.add_parser("import", help="Import a signed license file.")
    license_import.add_argument("--file", default="")
    license_import.add_argument("--license-json", default="")
    license_check = license_sub.add_parser("check", help="Check whether a licensed feature is allowed.")
    license_check.add_argument("--feature", default="local_research")
    license_check.add_argument("--requested-tier", default="")

    setup = sub.add_parser("setup", help="Start the no-terminal web setup wizard.")
    setup.add_argument("--port", type=int, default=8788)
    setup.add_argument("--host", default="127.0.0.1")
    setup.add_argument("--no-open", action="store_true")

    join = sub.add_parser("join", help="One-step alias for joining a Hive from an invite/bootstrap bundle.")
    add_join_args(join)

    bootstrap = sub.add_parser("bootstrap", help="Write a one-step join bundle and iPhone/Watch QR import profile.")
    add_bootstrap_args(bootstrap)

    start = sub.add_parser("start", help="Start dashboard, Hive node, and optional relay.")
    start.add_argument("--dashboard-port", type=int, default=8787)
    start.add_argument("--hive-port", type=int, default=8791)
    start.add_argument("--relay-port", type=int, default=8793)
    start.add_argument("--relay", action="store_true")
    start.add_argument("--restart", action="store_true")

    stop = sub.add_parser("stop", help="Stop local Theseus dashboard/Hive helper processes.")
    stop.add_argument("--force", action="store_true")

    chat = sub.add_parser("chat", help="Chat through the canonical local assistant runtime.")
    chat.add_argument("message", nargs="*")
    chat.add_argument("--prompt", default="")
    chat.add_argument("--checkpoint-id", default="live")
    chat.add_argument("--session-id", default="cli_default")
    chat.add_argument("--intent", choices=["auto", "chat", "code", "tool", "planning"], default="auto")
    chat.add_argument("--feedback", choices=["accepted", "missed", "ignored", "corrected", "completed"], default="completed")
    chat.add_argument("--surface", default="theseus_cli")
    chat.add_argument("--allow-teacher", action="store_true")

    feedback = sub.add_parser("feedback", help="Mark the latest local assistant answer as accepted, missed, ignored, corrected, or completed.")
    feedback.add_argument("outcome", choices=["accepted", "missed", "ignored", "corrected", "completed"])
    feedback.add_argument("--session-id", default="cli_default")
    feedback.add_argument("--latest-report", default="reports/checkpoint_chat_last.json")
    feedback.add_argument("--artifact-ref", action="append", default=[])
    feedback.add_argument("--assistant-lane", default="")
    feedback.add_argument("--surface", default="theseus_cli_feedback")
    feedback.add_argument("--intent-summary-redacted", default="")
    feedback.add_argument("--error-family", default="")
    feedback.add_argument("--duration-ms", type=int, default=0)
    feedback.add_argument("--skip-training-bridge", action="store_true")
    feedback.add_argument("--out", default="reports/theseus_cli_feedback_latest.json")
    feedback.add_argument("--markdown-out", default="reports/theseus_cli_feedback_latest.md")

    do = sub.add_parser("do", help="Submit a bounded Hive task to a local or remote Hive node.")
    do.add_argument("kind", nargs="?")
    do.add_argument("--kind", dest="kind_flag", default="")
    do.add_argument("--payload-json", default="{}")
    do.add_argument("--peer-url", default="http://127.0.0.1:8791")

    schedule = sub.add_parser("schedule", help="Refresh the resource-aware Hive scheduler report.")
    schedule.add_argument("--execute", action="store_true")
    schedule.add_argument("--probe-peers", action="store_true")
    schedule.add_argument("--worker-chunks", action="store_true")
    schedule.add_argument("--out", default="reports/hive_scheduler.json")

    utilize = sub.add_parser("utilize", help="Keep safe Hive compute fed with bounded work.")
    utilize_sub = utilize.add_subparsers(dest="utilize_command")
    utilize_status = utilize_sub.add_parser("status", help="Show idle/busy Hive slots and queue-fill plan.")
    utilize_status.add_argument("--out", default="reports/hive_utilization_manager.json")
    utilize_sweep = utilize_sub.add_parser("sweep", help="Plan or execute one utilization sweep.")
    utilize_sweep.add_argument("--execute", action="store_true")
    utilize_sweep.add_argument("--profile", default="")
    utilize_sweep.add_argument("--max-new-jobs", type=int, default=0)
    utilize_sweep.add_argument("--offline", action="store_true", help="Flight-safe mode: force local-only work and avoid internet/peer maintenance checks.")
    utilize_sweep.add_argument("--local-only", action="store_true")
    utilize_sweep.add_argument("--allow-wan", action="store_true")
    utilize_sweep.add_argument("--allow-battery", action="store_true", help="Allow bounded local work while on battery; still respects --min-battery-percent.")
    utilize_sweep.add_argument("--min-battery-percent", type=float, default=None)
    utilize_sweep.add_argument("--keep-awake", action="store_true", help="On macOS, hold a caffeinate assertion while this utilization command runs.")
    utilize_sweep.add_argument("--out", default="reports/hive_utilization_manager.json")
    utilize_loop = utilize_sub.add_parser("loop", help="Run repeated utilization sweeps until stopped.")
    utilize_loop.add_argument("--execute", action="store_true")
    utilize_loop.add_argument("--profile", default="")
    utilize_loop.add_argument("--max-new-jobs", type=int, default=0)
    utilize_loop.add_argument("--offline", action="store_true", help="Flight-safe mode: force local-only work and avoid internet/peer maintenance checks.")
    utilize_loop.add_argument("--local-only", action="store_true")
    utilize_loop.add_argument("--allow-wan", action="store_true")
    utilize_loop.add_argument("--allow-battery", action="store_true", help="Allow bounded local work while on battery; still respects --min-battery-percent.")
    utilize_loop.add_argument("--min-battery-percent", type=float, default=None)
    utilize_loop.add_argument("--keep-awake", action="store_true", help="On macOS, hold a caffeinate assertion while this utilization command runs.")
    utilize_loop.add_argument("--cycles", type=int, default=0, help="0 means run until a utilization stop flag is written.")
    utilize_loop.add_argument("--sleep-seconds", type=int, default=60)
    utilize_loop.add_argument("--out", default="reports/hive_utilization_manager.json")

    train = sub.add_parser("train", help="Plan, run, and inspect decentralized Hive training rounds.")
    train_sub = train.add_subparsers(dest="train_command")
    train_status = train_sub.add_parser("status", help="Show active arm leases and promoted training artifacts.")
    train_status.add_argument("--out", default="")
    train_plan = train_sub.add_parser("plan", help="Build an arm-level placement plan without queueing work.")
    add_train_round_args(train_plan)
    train_run = train_sub.add_parser("run", help="Queue a bounded decentralized training round.")
    add_train_round_args(train_run)
    train_run.add_argument("--sync-artifacts", action="store_true")
    train_run.add_argument("--no-execute", action="store_true")
    train_sync = train_sub.add_parser("sync", help="Fetch and merge Hive training artifacts.")
    train_sync.add_argument("--out", default="reports/hive_training_orchestrator.json")
    train_overnight = train_sub.add_parser("overnight", help="Write an auditable overnight training report.")
    train_overnight.add_argument("--hours", type=float, default=12.0)
    train_overnight.add_argument("--out", default="reports/hive_overnight_training_report.json")
    train_overnight.add_argument("--markdown-out", default="reports/hive_overnight_training_report.md")
    train_teacher = train_sub.add_parser("teacher-preflight", help="Verify full-training sparse-teacher readiness.")
    train_teacher.add_argument("--profile", default="smoke")
    train_teacher.add_argument("--policy", default="configs/teacher_policy.json")
    train_teacher.add_argument("--require-teacher-cli", action="store_true")
    train_teacher.add_argument("--allow-teacher-live", action="store_true")
    train_teacher.add_argument("--require-live-teacher", action="store_true")
    train_teacher.add_argument("--live-timeout-seconds", type=int, default=240)
    train_teacher.add_argument("--skip-queue-smoke", action="store_true")
    train_teacher.add_argument("--skip-autonomy-readiness", action="store_true")
    train_teacher.add_argument("--out", default="reports/full_training_teacher_preflight.json")
    train_teacher.add_argument("--markdown-out", default="reports/full_training_teacher_preflight.md")
    train_generalization = train_sub.add_parser("generalization-governor", help="Consolidate private transfer evidence and the locked public-transfer wall.")
    train_generalization.add_argument("--out", default="reports/theseus_generalization_governor_v1.json")
    train_generalization.add_argument("--markdown-out", default="reports/theseus_generalization_governor_v1.md")
    train_generalization.add_argument("--queue-out", default="reports/theseus_generalization_governor_v1_queue.jsonl")
    train_generalization.add_argument("--stale-seconds", type=int, default=72 * 3600)
    train_pre_public = train_sub.add_parser("pre-public-audit", help="Audit private readiness before any operator-approved public calibration.")
    train_pre_public.add_argument("--out", default="reports/pre_public_generalization_readiness_audit.json")
    train_pre_public.add_argument("--markdown-out", default="reports/pre_public_generalization_readiness_audit.md")
    train_pre_public.add_argument("--queue-out", default="reports/pre_public_generalization_readiness_audit_queue.jsonl")
    train_pre_public.add_argument("--stale-seconds", type=int, default=72 * 3600)
    train_frontier_expander = train_sub.add_parser("frontier-expander", help="Queue or run safe private frontier expansion while public calibration is locked.")
    train_frontier_expander.add_argument("--execute", action="store_true")
    train_frontier_expander.add_argument("--max-actions", type=int, default=1)
    train_frontier_expander.add_argument("--allow-battery", action="store_true")
    train_frontier_expander.add_argument("--out", default="reports/private_generalization_frontier_expander_v1.json")
    train_frontier_expander.add_argument("--markdown-out", default="reports/private_generalization_frontier_expander_v1.md")
    train_frontier_expander.add_argument("--queue-out", default="reports/private_generalization_frontier_expander_v1_queue.jsonl")
    train_semantic_alias = train_sub.add_parser("semantic-alias-gate", help="Run the private semantic-alias transfer gate without public calibration.")
    train_semantic_alias.add_argument("--execute", action="store_true")
    train_semantic_alias.add_argument("--task-limit", type=int, default=0, help="0 means all private alias heldout rows.")
    train_semantic_alias.add_argument("--min-alias-rows", type=int, default=1008)
    train_semantic_alias.add_argument("--seed", type=int, default=43)
    train_semantic_alias.add_argument("--candidates-per-task", type=int, default=2)
    train_semantic_alias.add_argument("--checkpoint-in", default="")
    train_semantic_alias.add_argument("--out", default="reports/broad_private_semantic_alias_gate_v1.json")
    train_semantic_alias.add_argument("--markdown-out", default="reports/broad_private_semantic_alias_gate_v1.md")
    train_novel_composition = train_sub.add_parser("novel-composition-gate", help="Run the private novel-composition transfer gate without public calibration.")
    train_novel_composition.add_argument("--execute", action="store_true")
    train_novel_composition.add_argument("--rows", type=int, default=1008)
    train_novel_composition.add_argument("--min-rows", type=int, default=1008)
    train_novel_composition.add_argument("--seed", type=int, default=47)
    train_novel_composition.add_argument("--candidates-per-task", type=int, default=2)
    train_novel_composition.add_argument("--checkpoint-in", default="")
    train_novel_composition.add_argument("--out", default="reports/broad_private_novel_composition_gate_v1.json")
    train_novel_composition.add_argument("--markdown-out", default="reports/broad_private_novel_composition_gate_v1.md")
    train_residual_ratchet = train_sub.add_parser("residual-ratchet", help="Build the private residual self-improvement queue without public calibration.")
    train_residual_ratchet.add_argument("--execute", action="store_true")
    train_residual_ratchet.add_argument("--max-actions", type=int, default=1)
    train_residual_ratchet.add_argument("--out", default="reports/private_residual_self_improvement_ratchet_v1.json")
    train_residual_ratchet.add_argument("--markdown-out", default="reports/private_residual_self_improvement_ratchet_v1.md")
    train_residual_ratchet.add_argument("--queue-out", default="reports/private_residual_self_improvement_ratchet_v1_queue.jsonl")
    train_residual_frontier = train_sub.add_parser("residual-frontier", help="Run private residual-frontier composition pressure without public calibration.")
    train_residual_frontier.add_argument("--execute", action="store_true")
    train_residual_frontier.add_argument("--rows", type=int, default=336)
    train_residual_frontier.add_argument("--min-rows", type=int, default=336)
    train_residual_frontier.add_argument("--seed", type=int, default=89)
    train_residual_frontier.add_argument("--candidates-per-task", type=int, default=3)
    train_residual_frontier.add_argument("--score-timeout-seconds", type=int, default=2)
    train_residual_frontier.add_argument("--max-hours", type=float, default=4.0)
    train_residual_frontier.add_argument("--allow-battery", action="store_true")
    train_residual_frontier.add_argument("--checkpoint-in", default="")
    train_residual_frontier.add_argument("--out", default="reports/private_residual_frontier_v1.json")
    train_residual_frontier.add_argument("--markdown-out", default="reports/private_residual_frontier_v1.md")
    train_residual_frontier.add_argument("--queue-out", default="reports/private_residual_frontier_v1_queue.jsonl")
    train_ecology_v5 = train_sub.add_parser("ecology-v5-refresh", help="Run the full private v5 ecology refresh without public calibration.")
    train_ecology_v5.add_argument("--execute", action="store_true")
    train_ecology_v5.add_argument("--train-rows", type=int, default=1200)
    train_ecology_v5.add_argument("--heldout-rows", type=int, default=480)
    train_ecology_v5.add_argument("--private-eval-limit", type=int, default=480)
    train_ecology_v5.add_argument("--seed", type=int, default=41)
    train_ecology_v5.add_argument("--curriculum-seed", type=int, default=59)
    train_ecology_v5.add_argument("--candidates-per-task", type=int, default=4)
    train_ecology_v5.add_argument("--score-timeout-seconds", type=int, default=2)
    train_ecology_v5.add_argument("--max-hours", type=float, default=6.0)
    train_ecology_v5.add_argument("--allow-battery", action="store_true")
    train_ecology_v5.add_argument("--checkpoint-in", default="")
    train_ecology_v5.add_argument("--out", default="reports/private_ecology_generalization_v5_refresh.json")
    train_ecology_v5.add_argument("--markdown-out", default="reports/private_ecology_generalization_v5_refresh.md")
    train_ecology_v5.add_argument("--queue-out", default="reports/private_ecology_generalization_v5_refresh_queue.jsonl")
    train_unseen_transfer = train_sub.add_parser("unseen-transfer-challenge", help="Run a private OOD transfer challenge with exact semantic keys withheld.")
    train_unseen_transfer.add_argument("--execute", action="store_true")
    train_unseen_transfer.add_argument("--rows", type=int, default=120)
    train_unseen_transfer.add_argument("--seed", type=int, default=73)
    train_unseen_transfer.add_argument("--candidates-per-task", type=int, default=4)
    train_unseen_transfer.add_argument("--score-timeout-seconds", type=int, default=2)
    train_unseen_transfer.add_argument("--max-hours", type=float, default=2.0)
    train_unseen_transfer.add_argument("--allow-battery", action="store_true")
    train_unseen_transfer.add_argument("--checkpoint-in", default="")
    train_unseen_transfer.add_argument("--out", default="reports/private_unseen_transfer_challenge_v1.json")
    train_unseen_transfer.add_argument("--markdown-out", default="reports/private_unseen_transfer_challenge_v1.md")
    train_unseen_transfer.add_argument("--queue-out", default="reports/private_unseen_transfer_challenge_v1_queue.jsonl")

    solo = sub.add_parser("solo", help="Run and inspect a local/offline solo-Mac learning loop.")
    solo_sub = solo.add_subparsers(dest="solo_command")
    solo_status = solo_sub.add_parser("status", help="Show standalone Mac learning, MLX, ledger, controls, and promotion state.")
    solo_status.add_argument("--hours", type=float, default=24.0)
    solo_status.add_argument("--out", default="reports/hive_solo_learning_status.json")
    solo_sweep = solo_sub.add_parser("sweep", help="Run one local/offline utilization sweep and update the solo ledger.")
    add_solo_run_args(solo_sweep)
    solo_sweep.add_argument("--wait-seconds", type=float, default=90.0)
    solo_loop = solo_sub.add_parser("loop", help="Run repeated local/offline sweeps and update the solo ledger.")
    add_solo_run_args(solo_loop)
    solo_loop.add_argument("--cycles", type=int, default=1)
    solo_loop.add_argument("--sleep-seconds", type=int, default=60)
    solo_loop.add_argument("--wait-seconds", type=float, default=90.0)
    solo_overnight = solo_sub.add_parser("overnight", help="Write the solo overnight/offline learning report.")
    solo_overnight.add_argument("--hours", type=float, default=12.0)
    solo_overnight.add_argument("--out", default="reports/hive_solo_overnight_report.json")
    solo_overnight.add_argument("--markdown-out", default="reports/hive_solo_overnight_report.md")

    hive = sub.add_parser("hive", help="Create, join, switch, invite, and upgrade Hive profiles.")
    hive_sub = hive.add_subparsers(dest="hive_command")
    hive_create = hive_sub.add_parser("create", help="Create and activate a private or semi-private Hive.")
    hive_create.add_argument("--name", default="My Project Theseus Hive")
    hive_create.add_argument("--tier", choices=["private", "friends_family", "company"], default="private")
    hive_create.add_argument("--mode", choices=["lan", "relay"], default="lan")
    hive_create.add_argument("--relay-url", default="")
    hive_create.add_argument("--start", action="store_true")

    hive_join = hive_sub.add_parser("join", help="Join a Hive from invite file, invite JSON, or manual details.")
    add_join_args(hive_join)

    hive_sub.add_parser("list", help="List saved Hive profiles.")
    hive_switch = hive_sub.add_parser("switch", help="Activate a saved Hive profile.")
    hive_switch.add_argument("profile_id")
    hive_switch.add_argument("--restart", action="store_true")

    hive_invite = hive_sub.add_parser("invite", help="Create an invite bundle for the active Hive.")
    hive_invite.add_argument("--out", default="")
    hive_invite.add_argument("--no-token", action="store_true")
    hive_bootstrap_cmd = hive_sub.add_parser("bootstrap", help="Write a one-step join bundle and iPhone/Watch QR import profile.")
    add_bootstrap_args(hive_bootstrap_cmd)
    hive_add_user = hive_sub.add_parser("add-user", help="Create a per-user operator token for a phone or family device.")
    hive_add_user.add_argument("--user-id", default="")
    hive_add_user.add_argument("--name", required=True)
    hive_add_user.add_argument("--role", choices=["owner", "operator", "member", "child", "guest"], default="member")
    hive_add_user.add_argument("--device-label", default="")
    hive_add_user.add_argument("--expires-days", type=int, default=0)
    hive_add_user.add_argument("--replace", action="store_true")
    hive_add_user.add_argument("--out", default="")
    hive_sub.add_parser("users", help="List configured Hive users without revealing tokens.")
    hive_revoke_user = hive_sub.add_parser("revoke-user", help="Disable a Hive user token.")
    hive_revoke_user.add_argument("user_id")
    hive_revoke_user.add_argument("--out", default="reports/hive_user_revoke_last.json")

    hive_upgrade = hive_sub.add_parser("upgrade-relay", help="Upgrade the active Hive to relay/multi-network mode.")
    hive_upgrade.add_argument("--relay-url", default="")
    hive_upgrade.add_argument("--start", action="store_true")
    hive_training_link = hive_sub.add_parser("training-link", help="Check Mac<->Windows Hive training task readiness.")
    hive_training_link.add_argument("--refresh", action="store_true")
    hive_training_link.add_argument("--out", default="reports/hive_training_link_doctor.json")
    hive_training_link.add_argument("--markdown-out", default="reports/hive_training_link_doctor.md")
    hive_network = hive_sub.add_parser("network-doctor", help="Diagnose live Hive API, peer, coordinator, and roaming reachability.")
    hive_network.add_argument("--timeout", type=float, default=1.5)
    hive_network.add_argument("--coordinator-url", action="append", default=[])
    hive_network.add_argument("--peer-url", action="append", default=[])
    hive_network.add_argument("--out", default="reports/hive_network_doctor.json")
    hive_network.add_argument("--markdown-out", default="reports/hive_network_doctor.md")
    hive_macos_gate = hive_sub.add_parser("macos-release-gate", help="Run the macOS DMG/pkg/version/update canary gate.")
    hive_macos_gate.add_argument("--execute", action="store_true")
    hive_macos_gate.add_argument("--skip-build", action="store_true")
    hive_macos_gate.add_argument("--skip-version-publish", action="store_true")
    hive_macos_gate.add_argument("--skip-local-install", action="store_true")
    hive_macos_gate.add_argument("--skip-local-converge", action="store_true")
    hive_macos_gate.add_argument("--skip-deps", action="store_true")
    hive_macos_gate.add_argument("--no-require-mlx", action="store_true")
    hive_macos_gate.add_argument("--api-url", default="http://127.0.0.1:8791")
    hive_macos_gate.add_argument("--coordinator-url", action="append", default=[])
    hive_macos_gate.add_argument("--peer-url", action="append", default=[])
    hive_macos_gate.add_argument("--timeout", type=float, default=2.0)
    hive_macos_gate.add_argument("--out", default="reports/hive_macos_release_gate.json")
    hive_macos_gate.add_argument("--markdown-out", default="reports/hive_macos_release_gate.md")

    device = sub.add_parser("device", help="Register new devices and inspect discovered peers.")
    device_sub = device.add_subparsers(dest="device_command")
    device_invite = device_sub.add_parser("invite", help="Alias for hive invite.")
    device_invite.add_argument("--out", default="")
    device_invite.add_argument("--no-token", action="store_true")
    device_register = device_sub.add_parser("register", help="Alias for hive join.")
    device_register.add_argument("--invite", required=True)
    device_register.add_argument("--start", action="store_true")
    device_sub.add_parser("list", help="List discovered Hive devices.")

    remote = sub.add_parser("remote", help="Configure free self-hosted remote access for home, workshop, travel, and phones.")
    remote_sub = remote.add_subparsers(dest="remote_command")
    remote_status = remote_sub.add_parser("status", help="Show free remote-access posture and next actions.")
    remote_status.add_argument("--out", default="reports/hive_remote_access_status.json")
    remote_configure = remote_sub.add_parser("configure-relay", help="Set the active Hive relay URL and write a remote invite.")
    remote_configure.add_argument("--relay-url", default="")
    remote_configure.add_argument("--out", default="reports/hive_invite_remote_private.json")
    remote_configure.add_argument("--start", action="store_true")
    remote_configure.add_argument("--restart", action="store_true")
    remote_guide = remote_sub.add_parser("wireguard-guide", help="Write a no-cost self-hosted WireGuard/private tunnel setup guide.")
    remote_guide.add_argument("--out", default="reports/hive_wireguard_free_setup.md")
    remote_mobile = remote_sub.add_parser("mobile-profile", help="Write an iPhone roaming profile with all known node/relay endpoints.")
    remote_mobile.add_argument("--out", default="reports/hive_mobile_roaming_profile.json")
    remote_mobile.add_argument("--no-token", action="store_true")
    remote_doctor = remote_sub.add_parser("doctor", help="Diagnose coordinator, LAN, relay, and roaming endpoint reachability.")
    remote_doctor.add_argument("--timeout", type=float, default=1.5)
    remote_doctor.add_argument("--coordinator-url", action="append", default=[])
    remote_doctor.add_argument("--peer-url", action="append", default=[])
    remote_doctor.add_argument("--out", default="reports/hive_network_doctor.json")
    remote_doctor.add_argument("--markdown-out", default="reports/hive_network_doctor.md")

    control = sub.add_parser("control", help="Create and launch governed Hive screen/keyboard/mouse takeover sessions.")
    control_sub = control.add_subparsers(dest="control_command")
    control_status = control_sub.add_parser("status", help="Show local remote-control provider readiness.")
    control_status.add_argument("--out", default="reports/hive_remote_control_status.json")
    control_request = control_sub.add_parser("request", help="Create an audited remote-control session handoff.")
    control_request.add_argument("--target-node", default="local")
    control_request.add_argument("--provider", default="auto")
    control_request.add_argument("--mode", choices=["view", "control"], default="control")
    control_request.add_argument("--duration-minutes", type=int, default=60)
    control_request.add_argument("--out", default="reports/hive_remote_control_session.json")
    control_launch = control_sub.add_parser("launch", help="Launch a local remote desktop client.")
    control_launch.add_argument("--provider", choices=["rdp", "vnc", "screen_sharing", "rustdesk", "sunshine_moonlight"], default="rdp")
    control_launch.add_argument("--host", default="")
    control_launch.add_argument("--target-url", default="")
    control_launch.add_argument("--rustdesk-id", default="")
    control_launch.add_argument("--execute", action="store_true")
    control_launch.add_argument("--out", default="reports/hive_remote_control_launch.json")

    storage = sub.add_parser("storage", help="Configure and use bounded Hive storage shares across trusted nodes.")
    storage_sub = storage.add_subparsers(dest="storage_command")
    storage_status = storage_sub.add_parser("status", help="Show local storage shares, NAS/mount candidates, and safety limits.")
    storage_status.add_argument("--out", default="reports/hive_storage_status.json")
    storage_index = storage_sub.add_parser("index", help="Write a bounded index of configured shares.")
    storage_index.add_argument("--out", default="reports/hive_storage_index.json")
    storage_index.add_argument("--limit", type=int, default=500)
    storage_add = storage_sub.add_parser("add-share", help="Expose a local folder or mounted NAS path as a read-only Hive share.")
    storage_add.add_argument("--path", required=True)
    storage_add.add_argument("--name", default="")
    storage_add.add_argument("--share-id", default="")
    storage_add.add_argument("--tag", action="append", default=[])
    storage_add.add_argument("--writable", action="store_true")
    storage_remove = storage_sub.add_parser("remove-share", help="Disable a configured Hive storage share.")
    storage_remove.add_argument("share_id")
    storage_browse = storage_sub.add_parser("browse", help="Browse a configured local storage share.")
    storage_browse.add_argument("--share-id", required=True)
    storage_browse.add_argument("--path", default="")
    storage_browse.add_argument("--limit", type=int, default=200)
    storage_browse.add_argument("--out", default="")
    storage_pull = storage_sub.add_parser("pull", help="Pull a file from a peer storage share into the local Hive storage inbox.")
    storage_pull.add_argument("--peer-url", required=True)
    storage_pull.add_argument("--share-id", required=True)
    storage_pull.add_argument("--path", required=True)
    storage_pull.add_argument("--out", default="")

    voice = sub.add_parser("voice", help="Configure room-aware Hive voice following and response routing.")
    voice_sub = voice.add_subparsers(dest="voice_command")
    voice_status = voice_sub.add_parser("status", help="Show local voice-following mic/speaker readiness.")
    voice_status.add_argument("--out", default="reports/hive_voice_following_status.json")
    voice_room = voice_sub.add_parser("configure-room", help="Set this node's room and opt in mic/speaker role.")
    voice_room.add_argument("--room-id", default="")
    voice_room.add_argument("--room-name", default="")
    voice_mic = voice_room.add_mutually_exclusive_group()
    voice_mic.add_argument("--microphone", dest="microphone", action="store_true")
    voice_mic.add_argument("--no-microphone", dest="microphone", action="store_false")
    voice_room.set_defaults(microphone=None)
    voice_speaker = voice_room.add_mutually_exclusive_group()
    voice_speaker.add_argument("--speaker", dest="speaker", action="store_true")
    voice_speaker.add_argument("--no-speaker", dest="speaker", action="store_false")
    voice_room.set_defaults(speaker=None)
    voice_room.add_argument("--priority", type=int, default=-1)
    voice_presence = voice_sub.add_parser("presence", help="Record a local voice-presence score without storing audio.")
    voice_presence.add_argument("--score", type=float, required=True)
    voice_presence.add_argument("--source", default="manual")
    voice_presence.add_argument("--room-id", default="")
    voice_presence.add_argument("--room-name", default="")
    voice_presence.add_argument("--rms-db", type=float, default=None)
    voice_presence.add_argument("--direction", type=float, default=None)
    voice_presence.add_argument("--out", default="reports/hive_voice_presence_last.json")
    voice_route = voice_sub.add_parser("route", help="Choose the current Hive voice listen/respond route.")
    voice_route.add_argument("--out", default="reports/hive_voice_following_route.json")

    spatial = sub.add_parser("spatial", help="Configure and inspect the privacy-preserving spatial Hive operator scene.")
    spatial_sub = spatial.add_subparsers(dest="spatial_command")
    spatial_status = spatial_sub.add_parser("status", help="Write rooms, nodes, voice route, storage anchors, and work state for spatial clients.")
    spatial_status.add_argument("--out", default="reports/hive_spatial_status.json")
    spatial_configure = spatial_sub.add_parser("configure-node", help="Set this node's room/zone/position for visionOS, Quest, and phone spatial views.")
    spatial_configure.add_argument("--room-id", default="")
    spatial_configure.add_argument("--room-name", default="")
    spatial_configure.add_argument("--zone", default="")
    spatial_configure.add_argument("--x", type=float, default=None)
    spatial_configure.add_argument("--y", type=float, default=None)
    spatial_configure.add_argument("--z", type=float, default=None)
    spatial_configure.add_argument("--yaw", type=float, default=None)
    spatial_display = spatial_configure.add_mutually_exclusive_group()
    spatial_display.add_argument("--display", dest="display", action="store_true")
    spatial_display.add_argument("--no-display", dest="display", action="store_false")
    spatial_configure.set_defaults(display=None)
    spatial_configure.add_argument("--surface", action="append", default=[])
    spatial_configure.add_argument("--device", action="append", default=[])
    spatial_configure.add_argument("--out", default="reports/hive_spatial_status.json")

    usb = sub.add_parser("usb", help="Write universal installer USB bundles for Windows, macOS, and Linux.")
    usb_sub = usb.add_subparsers(dest="usb_command")
    usb_sub.add_parser("list", help="List removable USB targets.")
    usb_write = usb_sub.add_parser("write", help="Write a portable Hive installer folder/zip with the active invite token embedded.")
    usb_write.add_argument("--out", default=str(Path("dist") / "universal-usb" / "ProjectTheseusUniversalUSB"))
    usb_write.add_argument("--target", default="")
    usb_write.add_argument("--confirm-label", default="")
    usb_write.add_argument("--yes", action="store_true")
    usb_write.add_argument("--usb-label", default="THESEUS_HIVE")
    usb_write.add_argument("--coordinator-url", default="")
    usb_write.add_argument("--invite", default="")
    usb_write.add_argument("--hive-mode", choices=["current", "new", "public"], default="current")
    usb_write.add_argument("--new-hive-name", default="Project Theseus Hive USB")
    usb_write.add_argument("--new-hive-tier", choices=["private", "friends_family", "company"], default="private")
    usb_write.add_argument("--no-activate-new-hive", action="store_true")
    usb_write.add_argument("--public-gateway-url", default="")
    usb_write.add_argument("--public-mode", choices=["off", "idle", "always"], default="idle")
    usb_write.add_argument("--public-worker-name", default="")
    usb_write.add_argument("--expires-days", type=int, default=30)
    usb_write.add_argument("--include-heavy-data", action="store_true")
    usb_write.add_argument("--no-zip", action="store_true")
    usb_write.add_argument("--force", action="store_true")
    usb_write.add_argument("--dry-run", action="store_true")

    public = sub.add_parser("public", help="Manage public idle-compute contribution.")
    public_sub = public.add_subparsers(dest="public_command")
    public_sub.add_parser("status", help="Show public contribution gates.")
    public_config = public_sub.add_parser("configure", help="Configure public contribution mode.")
    public_config.add_argument("--mode", choices=["off", "idle", "always"], default="off")
    public_config.add_argument("--gateway-url", default="")
    public_config.add_argument("--worker-name", default="")
    public_config.add_argument("--allow", action="store_true")
    public_sub.add_parser("poll-once", help="Check the public queue once without a daemon.")
    public_sub.add_parser("work-smoke", help="Run the bounded public worker smoke task.")

    market = sub.add_parser("market", help="Quote, settle, and inspect Theseus Work Credit compute accounting.")
    market_sub = market.add_subparsers(dest="market_command")
    market_sub.add_parser("status", help="Show internal work-credit wallet, receipts, and legal posture.")
    market_quote = market_sub.add_parser("quote", help="Quote gas for a bounded Hive task.")
    market_quote.add_argument("--task-kind", default="cuda_eval_chunk")
    market_quote.add_argument("--payload-json", default="{}")
    market_settle = market_sub.add_parser("settle", help="Settle receipts from the worker chunk ledger.")
    market_settle.add_argument("--worker-ledger", default="reports/hive_worker_chunk_ledger.jsonl")
    market_settle.add_argument("--limit", type=int, default=50)
    market_rent = market_sub.add_parser("rent-plan", help="Build a quote-first rent plan for a stronger Hive worker.")
    market_rent.add_argument("--task-kind", default="cuda_eval_chunk")
    market_rent.add_argument("--payload-json", default="{}")
    market_rent.add_argument("--max-gas-micro-twc", type=int, default=0)

    rent = sub.add_parser("rent", help="Plan and explicitly launch rented cloud compute/storage for the private Hive.")
    rent_sub = rent.add_subparsers(dest="rent_command")
    rent_status = rent_sub.add_parser("status", help="Show rented-compute profiles, provider readiness, and last plan.")
    rent_status.add_argument("--out", default="reports/hive_rented_compute_status.json")
    rent_init = rent_sub.add_parser("init", help="Create an ignored local rented-compute profile template.")
    rent_init.add_argument("--provider", default="aws_ec2")
    rent_init.add_argument("--name", default="")
    rent_init.add_argument("--repo-url", default="")
    rent_init.add_argument("--branch", default="")
    rent_init.add_argument("--region", default="us-east-1")
    rent_init.add_argument("--out", default="configs/hive_rented_compute.local.json")
    rent_init.add_argument("--overwrite", action="store_true")
    rent_plan = rent_sub.add_parser("plan", help="Build a dry-run rented compute/storage plan.")
    rent_plan.add_argument("--profile", default="aws-gpu-nightly")
    rent_plan.add_argument("--task-kind", default="cuda_training_chunk")
    rent_plan.add_argument("--hours", type=float, default=4.0)
    rent_plan.add_argument("--estimated-hourly-usd", type=float, default=0.0)
    rent_plan.add_argument("--ignore-conditions", action="store_true")
    rent_plan.add_argument("--out", default="reports/hive_rented_compute_plan.json")
    rent_launch = rent_sub.add_parser("launch", help="Launch a previously approved dry-run plan. Requires --execute.")
    rent_launch.add_argument("--plan", default="reports/hive_rented_compute_plan.json")
    rent_launch.add_argument("--execute", action="store_true")
    rent_launch.add_argument("--out", default="reports/hive_rented_compute_launch.json")
    rent_stop = rent_sub.add_parser("stop", help="Terminate/release rented capacity. Requires --execute.")
    rent_stop.add_argument("--profile", default="")
    rent_stop.add_argument("--provider", default="aws_ec2")
    rent_stop.add_argument("--instance-id", default="")
    rent_stop.add_argument("--bucket", default="")
    rent_stop.add_argument("--resource-name", default="")
    rent_stop.add_argument("--execute", action="store_true")
    rent_stop.add_argument("--out", default="reports/hive_rented_compute_launch.json")

    openai = sub.add_parser("openai", help="Manage the local OpenAI-compatible endpoint shim.")
    openai_sub = openai.add_subparsers(dest="openai_command")
    openai_sub.add_parser("status", help="Show local endpoint status and base URL.")
    openai_start = openai_sub.add_parser("start", help="Enable and start the local endpoint.")
    openai_start.add_argument("--host", default="")
    openai_start.add_argument("--port", type=int, default=0)
    openai_start.add_argument("--model", default="")
    openai_start.add_argument("--checkpoint-id", default="")
    openai_stop = openai_sub.add_parser("stop", help="Stop the local endpoint.")
    openai_stop.add_argument("--disable", action="store_true")
    openai_config = openai_sub.add_parser("configure", help="Configure the local endpoint without starting it.")
    openai_state = openai_config.add_mutually_exclusive_group()
    openai_state.add_argument("--enable", action="store_true")
    openai_state.add_argument("--disable", action="store_true")
    openai_config.add_argument("--host", default="")
    openai_config.add_argument("--port", type=int, default=0)
    openai_config.add_argument("--model", default="")
    openai_config.add_argument("--checkpoint-id", default="")
    openai_config.add_argument("--require-token", action="store_true")
    openai_config.add_argument("--no-token-required", action="store_true")
    openai_config.add_argument("--api-token", default="")

    updates = sub.add_parser("update", help="Check, create, and install accepted-candidate updates.")
    update_sub = updates.add_subparsers(dest="update_command")
    update_sub.add_parser("status", help="Show candidate update status.")
    update_check = update_sub.add_parser("check", help="Check the configured official/private update catalog.")
    update_check.add_argument("--catalog-url", default="")
    update_check.add_argument("--update-id", default="")
    update_check.add_argument("--apply", action="store_true")
    update_check.add_argument("--if-enabled-on-start", action="store_true")
    update_check.add_argument("--respect-interval", action="store_true")
    update_configure = update_sub.add_parser("configure", help="Set manual, notify, or safe auto-update behavior.")
    update_configure.add_argument("--mode", choices=["manual", "notify", "auto_soft", "auto_safe"], default="")
    update_configure.add_argument("--channel", default="")
    update_configure.add_argument("--track", choices=["stable", "beta", "dev"], default="")
    update_configure.add_argument("--catalog-url", default="")
    update_configure.add_argument("--check-on-start", action="store_true")
    update_configure.add_argument("--no-check-on-start", action="store_true")
    update_configure.add_argument("--auto-install-soft", action="store_true")
    update_configure.add_argument("--no-auto-install-soft", action="store_true")
    update_configure.add_argument("--auto-install-hard", action="store_true")
    update_configure.add_argument("--no-auto-install-hard", action="store_true")
    update_configure.add_argument("--allow-prerelease", action="store_true")
    update_configure.add_argument("--no-allow-prerelease", action="store_true")
    update_sub.add_parser("catalog", help="Print the public update catalog this node can share.")
    update_create = update_sub.add_parser("create", help="Create an update offer from a promoted checkpoint.")
    update_create.add_argument("--checkpoint-id", default="")
    update_create.add_argument("--if-promoted", action="store_true")
    update_apply = update_sub.add_parser("apply", help="Apply a soft update or guarded hard update.")
    update_apply.add_argument("--mode", choices=["auto", "soft", "hard"], default="auto")
    update_apply.add_argument("--execute", action="store_true")
    update_apply.add_argument("--allow-hard", action="store_true")
    update_apply.add_argument("--restart", action="store_true")
    update_apply.add_argument("--offer", default="")
    update_sub.add_parser("hive-version", help="Show Hive version and installer convergence status.")
    update_verify_hive = update_sub.add_parser("verify-hive", help="Verify and bless the current Hive version.")
    update_verify_hive.add_argument("--skip-checks", action="store_true")
    update_publish_hive = update_sub.add_parser("publish-hive", help="Publish the verified Hive version as the private catalog.")
    update_publish_hive.add_argument("--skip-checks", action="store_true")
    update_converge_hive = update_sub.add_parser("converge-hive", help="Ask Hive peers to install safe soft updates from the private catalog.")
    update_converge_hive.add_argument("--catalog-url", default="")
    update_converge_hive.add_argument("--peer-url", action="append", default=[])
    update_converge_hive.add_argument("--execute", action="store_true")
    update_converge_hive.add_argument("--allow-hard", action="store_true")

    args = parser.parse_args(raw_argv)
    if not args.command:
        parser.print_help()
        return 2

    handlers = {
        "install": lambda: install_cli(args),
        "status": lambda: status_payload(),
        "benchmark": lambda: benchmark_command(args),
        "license": lambda: license_command(args),
        "runtime": lambda: runtime_command(args),
        "mac": lambda: mac_command(args),
        "cuda": lambda: cuda_command(args),
        "setup": lambda: start_setup_wizard(args),
        "join": lambda: join_command(args),
        "bootstrap": lambda: bootstrap_command(args),
        "start": lambda: start_services(args),
        "stop": lambda: stop_services(args),
        "chat": lambda: chat_checkpoint(args),
        "feedback": lambda: feedback_command(args),
        "do": lambda: submit_task(args),
        "schedule": lambda: run_scheduler(args),
        "utilize": lambda: utilization_command(args),
        "train": lambda: train_command(args),
        "solo": lambda: solo_command(args),
        "hive": lambda: hive_command(args),
        "device": lambda: device_command(args),
        "remote": lambda: remote_command(args),
        "control": lambda: control_command(args),
        "storage": lambda: storage_command(args),
        "voice": lambda: voice_command(args),
        "spatial": lambda: spatial_command(args),
        "usb": lambda: usb_command(args),
        "public": lambda: public_command(args),
        "market": lambda: market_command(args),
        "rent": lambda: rent_command(args),
        "openai": lambda: openai_command(args),
        "update": lambda: update_command(args),
    }
    report = handlers[args.command]()
    emit(report, json_output=json_output, command=args.command)
    return 0 if report.get("ok", True) else 2


def add_train_round_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--round-id", default="")
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--allow-wan", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--out", default="reports/hive_training_orchestrator.json")


def add_solo_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--profile", default="inner_loop")
    parser.add_argument("--max-new-jobs", type=int, default=1)
    parser.add_argument("--allow-battery", action="store_true")
    parser.add_argument("--min-battery-percent", type=float, default=None)
    parser.add_argument("--keep-awake", action="store_true")
    parser.add_argument("--out", default="reports/hive_solo_learning_status.json")


def add_join_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--invite", default="")
    parser.add_argument("--invite-json", default="")
    parser.add_argument("--hive-id", default="")
    parser.add_argument("--join-token", default="")
    parser.add_argument("--relay-url", default="")
    parser.add_argument("--coordinator-url", default="")
    parser.add_argument("--name", default="")
    parser.add_argument("--tier", choices=["private", "friends_family", "company"], default="private")
    parser.add_argument("--start", action="store_true")


def add_bootstrap_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", default="reports/hive_join_bundle.json")
    parser.add_argument("--qr-out", default="reports/hive_join_profile_qr.svg")
    parser.add_argument("--no-token", action="store_true")
    parser.add_argument("--coordinator-url", action="append", default=[])
    parser.add_argument("--relay-url", action="append", default=[])
    parser.add_argument("--operator-token-scope", default="")


def install_cli(args: argparse.Namespace) -> dict[str, Any]:
    target = Path(args.target_dir).expanduser() if args.target_dir else default_cli_target_dir()
    target.mkdir(parents=True, exist_ok=True)
    wrapper = target / ("theseus.cmd" if platform.system() == "Windows" else "theseus")
    if platform.system() == "Windows":
        wrapper.write_text(
            "@echo off\r\n"
            f"cd /d \"{ROOT}\"\r\n"
            f"\"{python_executable()}\" \"{ROOT / 'scripts' / 'theseus_cli.py'}\" %*\r\n",
            encoding="utf-8",
        )
    else:
        wrapper.write_text(
            "#!/usr/bin/env sh\n"
            f"cd \"{ROOT}\" || exit 1\n"
            f"exec \"{python_executable()}\" \"{ROOT / 'scripts' / 'theseus_cli.py'}\" \"$@\"\n",
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
    probe = {} if args.skip_probe else run_json([python_executable(), "scripts/hive_node.py", "probe"])
    scheduler = {} if args.skip_probe else run_json([python_executable(), "scripts/hive_scheduler.py", "--out", "reports/hive_scheduler.json"])
    path_entries = [Path(item).expanduser().resolve() for item in os.environ.get("PATH", "").split(os.pathsep) if item]
    on_path = target.resolve() in path_entries
    report = {
        "ok": True,
        "policy": "project_theseus_cli_install_v0",
        "created_utc": now(),
        "wrapper": str(wrapper),
        "target_dir": str(target),
        "target_dir_on_path": on_path,
        "probe": compact_probe(probe),
        "scheduler_ok": bool(scheduler.get("ok", True)) if scheduler else None,
        "next_commands": [
            "theseus status" if on_path else f"Run {wrapper} status, or add {target} to PATH.",
            "theseus setup",
            "theseus hive create --name Home --tier private --mode lan --start",
        ],
    }
    write_json(REPORTS / "theseus_cli_install.json", report)
    return report


def status_payload() -> dict[str, Any]:
    policy = read_json(POLICY_PATH, {})
    active_private = hive_profiles.active_profile()
    dashboard_live = port_open("127.0.0.1", 8787)
    hive_live = port_open("127.0.0.1", 8791)
    relay_live = port_open("127.0.0.1", 8793)
    node_status = fetch_json("http://127.0.0.1:8791/api/hive/status") if hive_live else read_json(REPORTS / "hive_status.json", {})
    dashboard_status = fetch_json("http://127.0.0.1:8787/api/status") if dashboard_live else read_json(REPORTS / "sparkstream_status.json", {})
    public_status = public_hive_contributor.status_report(policy)
    license_status = license_manager.status_report(write_report=True)
    openai_status = openai_compat_server.status_report(write_report=True)
    update_status = update_manager.status_report(write_report=True)
    market_status = compute_market.status_report(write_report=True)
    rented_compute_status = hive_rented_compute.status_report(policy=policy, write_report=True)
    utilization_status = hive_utilization_manager.build_report(policy, execute=False, args=argparse.Namespace(profile="", max_new_jobs=0, local_only=False, allow_wan=False))
    remote_status = hive_remote_access.status_report(write_report=True)
    remote_control_status = hive_remote_control.status_report(policy=policy, write_report=True)
    voice_status = hive_voice_following.status_report(policy=policy, write_report=True)
    report = {
        "ok": True,
        "policy": "project_theseus_cli_status_v0",
        "created_utc": now(),
        "root": str(ROOT),
        "services": {
            "dashboard": {"live": dashboard_live, "url": "http://127.0.0.1:8787"},
            "hive_node": {"live": hive_live, "url": "http://127.0.0.1:8791/api/hive/status"},
            "hive_relay": {"live": relay_live, "url": "http://127.0.0.1:8793/api/hive/relay/status"},
        },
        "active_hive": hive_profiles.public_profile(active_private) if active_private else {},
        "node": compact_node(node_status),
        "dashboard": compact_dashboard(dashboard_status),
        "candidate": compact_candidate(read_json(REPORTS / "candidate_promotion_gate.json", {})),
        "benchmarks": compact_benchmarks(read_json(REPORTS / "benchmark_ledger.json", [])),
        "peers": compact_peers(read_json(REPORTS / "hive_peers.json", {})),
        "scheduler": compact_scheduler(read_json(REPORTS / "hive_scheduler.json", {})),
        "public_contribution": compact_public(public_status),
        "compute_market": compact_market(market_status),
        "rented_compute": compact_rented_compute(rented_compute_status),
        "utilization": compact_utilization(utilization_status),
        "remote_access": compact_remote(remote_status),
        "remote_control": compact_remote_control(remote_control_status),
        "voice_following": compact_voice(voice_status),
        "license": compact_license(license_status),
        "runtime_paths": compact_runtime(theseus_runtime.runtime_report(create=False, write_report=True)),
        "openai_compat": compact_openai(openai_status),
        "updates": compact_updates(update_status),
    }
    write_json(REPORTS / "theseus_cli_status.json", report)
    return report


def benchmark_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.benchmark_command in {None, "public"}:
        command = [
            python_executable(),
            "scripts/theseus_benchmark_measurement.py",
            "--cards",
            args.cards,
            "--cases-per-card",
            str(args.cases_per_card),
            "--seed-start",
            str(args.seed_start),
            "--seed-end",
            str(args.seed_end),
            "--slug-prefix",
            args.slug_prefix,
            "--out",
            args.out,
            "--markdown-out",
            args.markdown_out,
            "--timeout-seconds",
            str(args.timeout_seconds),
        ]
        if args.refresh_capacity:
            command.append("--refresh-capacity")
        if args.materialize:
            command.append("--materialize")
        command.append("--diagnostic-if-needed" if args.diagnostic_if_needed else "--no-diagnostic-if-needed")
        if args.execute:
            command.append("--execute")
        result = run_json(command, timeout=max(1800, int(args.timeout_seconds) + 600 if args.execute else 1800))
        result.setdefault("ok", result.get("trigger_state") != "RED")
        return result
    return {"ok": False, "error": "benchmark_subcommand_required"}


def license_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.license_command in {None, "status"}:
        return license_manager.status_report(write_report=True)
    if args.license_command == "register":
        return license_manager.register_install(license_manager.read_json(license_manager.POLICY_PATH, {}), args)
    if args.license_command == "request":
        return license_manager.license_request(license_manager.read_json(license_manager.POLICY_PATH, {}), args.feature)
    if args.license_command == "import":
        return license_manager.import_license(license_manager.read_json(license_manager.POLICY_PATH, {}), file_path=args.file, raw=args.license_json)
    if args.license_command == "check":
        result = license_manager.check_feature(args.feature, context={"requested_tier": args.requested_tier}, write_report=True)
        return {**result, "ok": bool(result.get("allowed"))}
    return {"ok": False, "error": "license_subcommand_required"}


def runtime_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.runtime_command in {None, "status"}:
        return theseus_runtime.runtime_report(create=bool(getattr(args, "create", False)), write_report=True)
    if args.runtime_command == "doctor":
        report = theseus_runtime.runtime_doctor_report(write_report=True)
        if getattr(args, "out", ""):
            write_json(resolve_path(args.out), report)
        return report
    if args.runtime_command == "init":
        return theseus_runtime.write_local_config(args.runtime_root, create=not bool(args.no_create))
    if args.runtime_command == "migrate-junctions":
        command = [
            python_executable(),
            "scripts/runtime_paths.py",
            "migrate-junctions",
            "--dirs",
            args.dirs,
        ]
        if args.runtime_root:
            command += ["--runtime-root", args.runtime_root]
        if args.dry_run:
            command.append("--dry-run")
        return run_json(command, timeout=3600)
    return {"ok": False, "error": "runtime_subcommand_required"}


def mac_command(args: argparse.Namespace) -> dict[str, Any]:
    policy = read_json(POLICY_PATH, {})
    if args.mac_command in {None, "canary"}:
        report = hive_macos_canary.canary_report(policy, args)
    elif args.mac_command == "roles":
        report = hive_macos_canary.role_assignment_report(policy, write_local=bool(args.write_local_config))
    elif args.mac_command == "training-preflight":
        report = hive_macos_canary.training_preflight_report(policy, args)
    elif args.mac_command == "dmg-readiness":
        report = hive_macos_canary.dmg_readiness_report(policy, args)
    elif args.mac_command == "join-bundle":
        report = hive_macos_canary.join_bundle_report(policy, args)
    elif args.mac_command == "app-status":
        report = hive_macos_canary.app_status_report(policy, api_url=str(args.api_url or "http://127.0.0.1:8791"))
        if bool(getattr(args, "text", False)):
            report["text"] = hive_macos_canary.app_status_text(report)
    else:
        return {"ok": False, "error": "mac_subcommand_required"}
    out = str(getattr(args, "out", "") or "")
    if out:
        write_json(resolve_path(out), report)
    markdown_out = str(getattr(args, "markdown_out", "") or "")
    if markdown_out:
        resolve_path(markdown_out).write_text(hive_macos_canary.markdown_report(report), encoding="utf-8")
    return report


def cuda_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.cuda_command in {None, "doctor"}:
        command = [
            python_executable(),
            "scripts/windows_cuda_doctor.py",
            "--profile",
            args.profile,
            "--out",
            args.out,
            "--markdown-out",
            args.markdown_out,
        ]
        if args.refresh:
            command.append("--refresh")
        if args.refresh_preflight:
            command.append("--refresh-preflight")
        result = run_json(command, timeout=600)
        result.setdefault("ok", result.get("trigger_state") != "RED")
        return result
    return {"ok": False, "error": "cuda_subcommand_required"}


def train_command(args: argparse.Namespace) -> dict[str, Any]:
    policy = read_json(POLICY_PATH, {})
    if args.train_command in {None, "status"}:
        report = hive_training_orchestrator.status_report(policy=policy)
        if getattr(args, "out", ""):
            write_json(resolve_path(args.out), report)
        return report
    if args.train_command == "sync":
        report = hive_training_orchestrator.sync_artifacts(policy)
        if getattr(args, "out", ""):
            write_json(resolve_path(args.out), report)
        return report
    if args.train_command == "overnight":
        return hive_training_orchestrator.overnight_report(
            policy=policy,
            hours=float(args.hours),
            out=resolve_path(args.out),
            markdown_out=resolve_path(args.markdown_out),
            write_report=True,
        )
    if args.train_command == "teacher-preflight":
        cli_args = [
            "--profile",
            args.profile,
            "--policy",
            args.policy,
            "--out",
            args.out,
            "--markdown-out",
            args.markdown_out,
            "--live-timeout-seconds",
            str(args.live_timeout_seconds),
            "--quiet",
        ]
        if args.require_teacher_cli:
            cli_args.append("--require-teacher-cli")
        if args.allow_teacher_live:
            cli_args.append("--allow-teacher-live")
        if args.require_live_teacher:
            cli_args.append("--require-live-teacher")
        if args.skip_queue_smoke:
            cli_args.append("--skip-queue-smoke")
        if args.skip_autonomy_readiness:
            cli_args.append("--skip-autonomy-readiness")
        rc = full_training_teacher_preflight.main(cli_args)
        report = read_json(resolve_path(args.out), {"ok": rc == 0})
        report.setdefault("ok", rc == 0)
        return report
    if args.train_command == "generalization-governor":
        result = run_json(
            [
                python_executable(),
                "scripts/theseus_generalization_governor_v1.py",
                "--out",
                args.out,
                "--markdown-out",
                args.markdown_out,
                "--queue-out",
                args.queue_out,
                "--stale-seconds",
                str(args.stale_seconds),
            ],
            timeout=120,
        )
        result.setdefault("ok", result.get("trigger_state") != "RED")
        return result
    if args.train_command == "pre-public-audit":
        result = run_json(
            [
                python_executable(),
                "scripts/pre_public_generalization_readiness_audit.py",
                "--out",
                args.out,
                "--markdown-out",
                args.markdown_out,
                "--queue-out",
                args.queue_out,
                "--stale-seconds",
                str(args.stale_seconds),
            ],
            timeout=120,
        )
        result.setdefault("ok", result.get("trigger_state") != "RED")
        return result
    if args.train_command == "frontier-expander":
        command = [
            python_executable(),
            "scripts/private_generalization_frontier_expander_v1.py",
            "--max-actions",
            str(args.max_actions),
            "--out",
            args.out,
            "--markdown-out",
            args.markdown_out,
            "--queue-out",
            args.queue_out,
        ]
        if args.execute:
            command.append("--execute")
        if args.allow_battery:
            command.append("--allow-battery")
        result = run_json(command, timeout=900 if not args.execute else 10 * 3600)
        result.setdefault("ok", result.get("trigger_state") != "RED")
        return result
    if args.train_command == "semantic-alias-gate":
        command = [
            python_executable(),
            "scripts/broad_private_semantic_alias_gate_v1.py",
            "--task-limit",
            str(args.task_limit),
            "--min-alias-rows",
            str(args.min_alias_rows),
            "--seed",
            str(args.seed),
            "--candidates-per-task",
            str(args.candidates_per_task),
            "--out",
            args.out,
            "--markdown-out",
            args.markdown_out,
        ]
        if args.execute:
            command.append("--execute")
        if args.checkpoint_in:
            command.extend(["--checkpoint-in", args.checkpoint_in])
        result = run_json(command, timeout=900)
        result.setdefault("ok", result.get("trigger_state") != "RED")
        return result
    if args.train_command == "novel-composition-gate":
        command = [
            python_executable(),
            "scripts/broad_private_novel_composition_gate_v1.py",
            "--rows",
            str(args.rows),
            "--min-rows",
            str(args.min_rows),
            "--seed",
            str(args.seed),
            "--candidates-per-task",
            str(args.candidates_per_task),
            "--out",
            args.out,
            "--markdown-out",
            args.markdown_out,
        ]
        if args.execute:
            command.append("--execute")
        if args.checkpoint_in:
            command.extend(["--checkpoint-in", args.checkpoint_in])
        result = run_json(command, timeout=900)
        result.setdefault("ok", result.get("trigger_state") != "RED")
        return result
    if args.train_command == "residual-ratchet":
        command = [
            python_executable(),
            "scripts/private_residual_self_improvement_ratchet_v1.py",
            "--max-actions",
            str(args.max_actions),
            "--out",
            args.out,
            "--markdown-out",
            args.markdown_out,
            "--queue-out",
            args.queue_out,
        ]
        if args.execute:
            command.append("--execute")
        result = run_json(command, timeout=900 if not args.execute else 6 * 3600)
        result.setdefault("ok", result.get("trigger_state") != "RED")
        return result
    if args.train_command == "residual-frontier":
        command = [
            python_executable(),
            "scripts/private_residual_frontier_v1.py",
            "--rows",
            str(args.rows),
            "--min-rows",
            str(args.min_rows),
            "--seed",
            str(args.seed),
            "--candidates-per-task",
            str(args.candidates_per_task),
            "--score-timeout-seconds",
            str(args.score_timeout_seconds),
            "--max-hours",
            str(args.max_hours),
            "--out",
            args.out,
            "--markdown-out",
            args.markdown_out,
            "--queue-out",
            args.queue_out,
        ]
        if args.execute:
            command.append("--execute")
        if args.allow_battery:
            command.append("--allow-battery")
        if args.checkpoint_in:
            command.extend(["--checkpoint-in", args.checkpoint_in])
        result = run_json(command, timeout=900 if not args.execute else 6 * 3600)
        result.setdefault("ok", result.get("trigger_state") != "RED")
        return result
    if args.train_command == "ecology-v5-refresh":
        command = [
            python_executable(),
            "scripts/private_ecology_generalization_v5_refresh.py",
            "--train-rows",
            str(args.train_rows),
            "--heldout-rows",
            str(args.heldout_rows),
            "--private-eval-limit",
            str(args.private_eval_limit),
            "--seed",
            str(args.seed),
            "--curriculum-seed",
            str(args.curriculum_seed),
            "--candidates-per-task",
            str(args.candidates_per_task),
            "--score-timeout-seconds",
            str(args.score_timeout_seconds),
            "--max-hours",
            str(args.max_hours),
            "--out",
            args.out,
            "--markdown-out",
            args.markdown_out,
            "--queue-out",
            args.queue_out,
        ]
        if args.execute:
            command.append("--execute")
        if args.allow_battery:
            command.append("--allow-battery")
        if args.checkpoint_in:
            command.extend(["--checkpoint-in", args.checkpoint_in])
        result = run_json(command, timeout=900 if not args.execute else 6 * 3600)
        result.setdefault("ok", result.get("trigger_state") != "RED")
        return result
    if args.train_command == "unseen-transfer-challenge":
        command = [
            python_executable(),
            "scripts/private_unseen_transfer_challenge_v1.py",
            "--rows",
            str(args.rows),
            "--seed",
            str(args.seed),
            "--candidates-per-task",
            str(args.candidates_per_task),
            "--score-timeout-seconds",
            str(args.score_timeout_seconds),
            "--max-hours",
            str(args.max_hours),
            "--out",
            args.out,
            "--markdown-out",
            args.markdown_out,
            "--queue-out",
            args.queue_out,
        ]
        if args.execute:
            command.append("--execute")
        if args.allow_battery:
            command.append("--allow-battery")
        if args.checkpoint_in:
            command.extend(["--checkpoint-in", args.checkpoint_in])
        result = run_json(command, timeout=900 if not args.execute else 2 * 3600)
        result.setdefault("ok", result.get("trigger_state") != "RED")
        return result
    if args.train_command in {"plan", "run"}:
        report = hive_training_orchestrator.orchestrate(
            policy,
            profile=args.profile,
            run_id=args.run_id,
            round_id=args.round_id,
            execute=args.train_command == "run" and not bool(getattr(args, "no_execute", False)),
            sync=bool(getattr(args, "sync_artifacts", False)),
            max_jobs=int(args.max_jobs or 0),
            allow_wan=bool(args.allow_wan),
            local_only=bool(args.local_only),
        )
        if getattr(args, "out", ""):
            write_json(resolve_path(args.out), report)
        return report
    return {"ok": False, "error": "train_subcommand_required"}


def solo_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.solo_command in {None, "status"}:
        report = hive_solo_learning_loop.solo_status(hours=float(getattr(args, "hours", 24.0)), write=True)
        if getattr(args, "out", ""):
            write_json(resolve_path(args.out), report)
        return report
    if args.solo_command == "sweep":
        report = hive_solo_learning_loop.run_sweep(args)
        if getattr(args, "out", ""):
            write_json(resolve_path(args.out), report)
        return report
    if args.solo_command == "loop":
        report = hive_solo_learning_loop.run_loop(args)
        if getattr(args, "out", ""):
            write_json(resolve_path(args.out), report)
        return report
    if args.solo_command == "overnight":
        report = hive_solo_learning_loop.solo_overnight(
            hours=float(args.hours),
            write=True,
            markdown_out=resolve_path(args.markdown_out),
        )
        if getattr(args, "out", ""):
            write_json(resolve_path(args.out), report)
        return report
    return {"ok": False, "error": "solo_subcommand_required"}


def utilization_command(args: argparse.Namespace) -> dict[str, Any]:
    policy = read_json(POLICY_PATH, {})
    if args.utilize_command in {None, "status"}:
        report = hive_utilization_manager.build_report(policy, execute=False, args=args)
        if getattr(args, "out", ""):
            write_json(resolve_path(args.out), report)
        return report
    if args.utilize_command == "sweep":
        keep_awake = hive_utilization_manager.start_keep_awake_assertion(args)
        try:
            report = hive_utilization_manager.build_report(policy, execute=bool(args.execute), args=args)
            if getattr(args, "out", ""):
                write_json(resolve_path(args.out), report)
            hive_utilization_manager.append_jsonl(hive_utilization_manager.LEDGER, hive_utilization_manager.compact_ledger(report))
            return report
        finally:
            hive_utilization_manager.stop_keep_awake_assertion(keep_awake)
    if args.utilize_command == "loop":
        keep_awake = hive_utilization_manager.start_keep_awake_assertion(args)
        try:
            cycles = 0
            last: dict[str, Any] = {}
            while not hive_utilization_manager.stop_requested() and (int(args.cycles) <= 0 or cycles < int(args.cycles)):
                cycles += 1
                last = hive_utilization_manager.build_report(policy, execute=bool(args.execute), args=args)
                if getattr(args, "out", ""):
                    write_json(resolve_path(args.out), last)
                hive_utilization_manager.append_jsonl(hive_utilization_manager.LEDGER, hive_utilization_manager.compact_ledger(last))
                if int(args.cycles) > 0 and cycles >= int(args.cycles):
                    break
                hive_utilization_manager.sleep_or_stop(max(1, int(args.sleep_seconds)))
            return last or {"ok": True, "stopped": hive_utilization_manager.stop_requested()}
        finally:
            hive_utilization_manager.stop_keep_awake_assertion(keep_awake)
    return {"ok": False, "error": "utilize_subcommand_required"}


def openai_command(args: argparse.Namespace) -> dict[str, Any]:
    policy = openai_compat_server.read_json(openai_compat_server.POLICY_PATH, {})
    if args.openai_command in {None, "status"}:
        return openai_compat_server.status_report(policy=policy, write_report=True)
    if args.openai_command == "configure":
        return openai_compat_server.configure_endpoint(policy, args)
    if args.openai_command == "start":
        cfg = openai_compat_server.effective_config(policy)
        cfg["enabled"] = True
        if args.host:
            cfg["host"] = args.host
        if args.port:
            cfg["port"] = args.port
        if args.model:
            cfg["model"] = args.model
        if args.checkpoint_id:
            cfg["checkpoint_id"] = args.checkpoint_id
        cfg["allow_teacher"] = False
        openai_compat_server.enforce_safe_defaults(policy, cfg)
        openai_compat_server.write_json(openai_compat_server.local_config_path(policy), cfg)
        if port_open(str(cfg.get("host") or "127.0.0.1"), int(cfg.get("port") or 8789)):
            return openai_compat_server.status_report(policy=policy, write_report=True)
        process = spawn([python_executable(), "scripts/openai_compat_server.py", "serve"], "openai_compat")
        time.sleep(0.7)
        return {**openai_compat_server.status_report(policy=policy, write_report=True), "started_process": process}
    if args.openai_command == "stop":
        if args.disable:
            cfg = openai_compat_server.effective_config(policy)
            cfg["enabled"] = False
            openai_compat_server.write_json(openai_compat_server.local_config_path(policy), cfg)
        return openai_compat_server.stop_endpoint(policy)
    return {"ok": False, "error": "openai_subcommand_required"}


def update_command(args: argparse.Namespace) -> dict[str, Any]:
    policy = update_manager.read_json(update_manager.POLICY_PATH, {})
    if args.update_command in {None, "status"}:
        return update_manager.status_report(policy=policy, write_report=True)
    if args.update_command == "check":
        check_args = argparse.Namespace(
            catalog_url=args.catalog_url,
            update_id=args.update_id,
            apply=bool(args.apply),
            if_enabled_on_start=bool(args.if_enabled_on_start),
            respect_interval=bool(args.respect_interval),
        )
        return update_manager.check_for_updates(policy, check_args)
    if args.update_command == "configure":
        return update_manager.configure_client(policy, args)
    if args.update_command == "catalog":
        return update_manager.public_catalog(policy)
    if args.update_command == "create":
        create_args = argparse.Namespace(checkpoint_id=args.checkpoint_id, if_promoted=bool(args.if_promoted))
        return update_manager.create_offer(policy, create_args)
    if args.update_command == "apply":
        apply_args = argparse.Namespace(
            mode=args.mode,
            execute=bool(args.execute),
            allow_hard=bool(args.allow_hard),
            restart=bool(args.restart),
            offer=args.offer,
        )
        return update_manager.apply_update(policy, apply_args)
    if args.update_command == "hive-version":
        return hive_version_manager.status_report(write_report=True)
    if args.update_command == "verify-hive":
        return hive_version_manager.verify_current_version(skip_checks=bool(args.skip_checks))
    if args.update_command == "publish-hive":
        return hive_version_manager.publish_catalog(skip_checks=bool(args.skip_checks), release_root="")
    if args.update_command == "converge-hive":
        converge_args = argparse.Namespace(
            catalog_url=args.catalog_url,
            peer_url=args.peer_url,
            execute=bool(args.execute),
            allow_hard=bool(args.allow_hard),
            timeout_seconds=15,
        )
        return hive_version_manager.converge_fleet(converge_args)
    return {"ok": False, "error": "update_subcommand_required"}


def join_command(args: argparse.Namespace) -> dict[str, Any]:
    return join_hive_from_args(args)


def bootstrap_command(args: argparse.Namespace) -> dict[str, Any]:
    report = hive_bootstrap.write_bootstrap_bundle(read_json(POLICY_PATH, {}), args)
    return report


def start_setup_wizard(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        python_executable(),
        "scripts/theseus_setup_wizard.py",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if not args.no_open:
        command.append("--open")
    proc = spawn(command, "setup_wizard")
    return {
        "ok": True,
        "policy": "project_theseus_cli_setup_start_v0",
        "created_utc": now(),
        "url": f"http://{args.host}:{args.port}",
        "process": proc,
    }


def start_services(args: argparse.Namespace) -> dict[str, Any]:
    report = theseus_setup_wizard.start_services(
        {
            "dashboard_port": args.dashboard_port,
            "hive_port": args.hive_port,
            "relay_port": args.relay_port,
            "start_relay": bool(args.relay),
            "restart": bool(args.restart),
        }
    )
    write_json(REPORTS / "theseus_cli_start.json", report)
    return report


def stop_services(args: argparse.Namespace) -> dict[str, Any]:
    if platform.system() == "Windows":
        stopped = theseus_setup_wizard.stop_known_processes(force=bool(args.force))
    else:
        patterns = [
            "scripts/sparkstream_dashboard.py",
            "scripts/hive_node.py",
            "scripts/hive_relay.py",
            "scripts/theseus_setup_wizard.py",
        ]
        if args.force:
            patterns.extend(
                [
                    "scripts/sparkstream_daemon.py",
                    "scripts/autonomy_cycle.py",
                    "scripts/code_lm_closure.py",
                    "scripts/legacy_port_mechanisms.py",
                    "symliquid-cli",
                ]
            )
        stopped_rows = []
        for pattern in patterns:
            result = subprocess.run(["pkill", "-f", pattern], cwd=ROOT, text=True, capture_output=True)
            stopped_rows.append({"pattern": pattern, "returncode": result.returncode})
        stopped = stopped_rows
    time.sleep(0.5)
    report = {"ok": True, "policy": "project_theseus_cli_stop_v0", "created_utc": now(), "stopped": stopped, "force": bool(args.force)}
    write_json(REPORTS / "theseus_cli_stop.json", report)
    return report


def chat_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    prompt = args.prompt or " ".join(args.message).strip()
    if not prompt:
        return {"ok": False, "error": "prompt_required"}
    command = [
        python_executable(),
        "scripts/theseus_assistant_runtime.py",
        "--checkpoint-id",
        args.checkpoint_id,
        "--session-id",
        args.session_id,
        "--prompt",
        prompt,
        "--intent",
        args.intent,
        "--feedback",
        args.feedback,
        "--surface",
        args.surface,
        "--out",
        "reports/checkpoint_chat_last.json",
    ]
    result = run_json(command, timeout=1800)
    if result:
        result["teacher_runtime_requested"] = bool(args.allow_teacher)
        result["teacher_runtime_allowed"] = False
        result["teacher_runtime_reason"] = "canonical assistant runtime does not serve external inference; teacher use remains governed training-only"
        result.setdefault("ok", True)
        return result
    return {"ok": False, "error": "theseus_assistant_runtime_failed"}


def feedback_command(args: argparse.Namespace) -> dict[str, Any]:
    latest_path = resolve_path(args.latest_report)
    latest = read_json(latest_path, {})
    latest_summary = latest.get("summary") if isinstance(latest.get("summary"), dict) else {}
    session_id = str(args.session_id or latest_summary.get("session_id") or "cli_default")
    lane = str(args.assistant_lane or latest_summary.get("assistant_lane") or "chat_checkpoint")
    artifact_refs = [str(item) for item in args.artifact_ref]
    if not artifact_refs and latest_path.exists():
        artifact_refs.append(relative_path(latest_path))
    if not artifact_refs:
        artifact_refs.append(str(args.latest_report))
    intent_summary = str(args.intent_summary_redacted or "").strip()
    if not intent_summary:
        intent = str(latest_summary.get("intent") or "assistant")
        state = str(latest.get("trigger_state") or latest.get("ok") or "unknown")
        intent_summary = f"posthoc_cli_feedback session={session_id} intent={intent} state={state}"
    event_out = REPORTS / "theseus_cli_feedback_event.json"
    event_md = REPORTS / "theseus_cli_feedback_event.md"
    event_cmd = [
        python_executable(),
        "scripts/dogfood_trace_event.py",
        "--execute",
        "--surface",
        str(args.surface),
        "--assistant-lane",
        lane,
        "--outcome",
        str(args.outcome),
        "--intent-summary-redacted",
        intent_summary,
        "--duration-ms",
        str(max(0, int(args.duration_ms or 0))),
        "--out",
        relative_path(event_out),
        "--markdown-out",
        relative_path(event_md),
    ]
    for artifact in artifact_refs:
        event_cmd.extend(["--artifact-ref", artifact])
    if args.error_family:
        event_cmd.extend(["--error-family", str(args.error_family)])
    event_report = run_json(event_cmd, timeout=60)
    bridge_report: dict[str, Any] = {}
    if not args.skip_training_bridge:
        bridge_out = REPORTS / "theseus_cli_feedback_training_bridge.json"
        bridge_md = REPORTS / "theseus_cli_feedback_training_bridge.md"
        bridge_cmd = [
            python_executable(),
            "scripts/dogfood_trace_training_bridge.py",
            "--execute",
            "--compact-existing",
            "--out",
            relative_path(bridge_out),
            "--markdown-out",
            relative_path(bridge_md),
        ]
        bridge_report = run_json(bridge_cmd, timeout=180)
    event_written = bool(event_report.get("event_written"))
    bridge_state = bridge_report.get("trigger_state") if bridge_report else "skipped"
    report = {
        "ok": bool(event_written and (args.skip_training_bridge or bridge_state in {"GREEN", "YELLOW"})),
        "policy": "project_theseus_cli_assistant_feedback_v0",
        "created_utc": now(),
        "session_id": session_id,
        "outcome": args.outcome,
        "assistant_lane": lane,
        "artifact_refs": artifact_refs,
        "event_written": event_written,
        "event_report": relative_path(event_out),
        "training_bridge_state": bridge_state,
        "training_rows_written": bridge_report.get("training_rows_written", 0) if bridge_report else 0,
        "training_bridge_report": "reports/theseus_cli_feedback_training_bridge.json" if bridge_report else "",
        "latest_report": relative_path(latest_path) if latest_path.exists() else str(args.latest_report),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "score_semantics": "Post-hoc local assistant feedback only. Writes redacted metadata and never stores raw prompt/completion text or public benchmark payloads.",
        "event": {
            "trigger_state": event_report.get("trigger_state"),
            "write_blocker": event_report.get("write_blocker"),
            "summary": event_report.get("summary") if isinstance(event_report.get("summary"), dict) else {},
        },
        "training_bridge": {
            "trigger_state": bridge_report.get("trigger_state") if bridge_report else "skipped",
            "summary": bridge_report.get("summary") if isinstance(bridge_report.get("summary"), dict) else {},
        },
    }
    write_json(resolve_path(args.out), report)
    markdown = [
        "# Theseus CLI Assistant Feedback",
        "",
        f"- ok: `{report['ok']}`",
        f"- outcome: `{report['outcome']}`",
        f"- session: `{report['session_id']}`",
        f"- lane: `{report['assistant_lane']}`",
        f"- event written: `{report['event_written']}`",
        f"- training bridge: `{report['training_bridge_state']}` rows `{report['training_rows_written']}`",
        f"- public training rows: `{report['public_training_rows_written']}`",
        f"- external inference calls: `{report['external_inference_calls']}`",
        f"- fallback returns: `{report['fallback_return_count']}`",
        "",
    ]
    resolve_path(args.markdown_out).parent.mkdir(parents=True, exist_ok=True)
    resolve_path(args.markdown_out).write_text("\n".join(markdown), encoding="utf-8")
    return report


def submit_task(args: argparse.Namespace) -> dict[str, Any]:
    kind = args.kind_flag or args.kind
    if not kind:
        return {"ok": False, "error": "task_kind_required"}
    payload = parse_json(args.payload_json, {})
    command = [
        python_executable(),
        "scripts/hive_node.py",
        "submit",
        "--peer-url",
        args.peer_url,
        "--kind",
        kind,
        "--payload-json",
        json.dumps(payload),
        "--out",
        "reports/theseus_cli_task_submit.json",
    ]
    result = run_json(command, env=hive_env())
    result.setdefault("ok", False)
    return result


def run_scheduler(args: argparse.Namespace) -> dict[str, Any]:
    command = [python_executable(), "scripts/hive_scheduler.py", "--out", args.out]
    if args.execute:
        command.append("--execute")
    if args.probe_peers:
        command.append("--probe-peers")
    if args.worker_chunks:
        command.append("--worker-chunks")
    result = run_json(command, timeout=600)
    result.setdefault("ok", True)
    return result


def hive_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.hive_command == "create":
        return theseus_setup_wizard.create_hive(
            {
                "name": args.name,
                "tier": args.tier,
                "mode": args.mode,
                "relay_url": args.relay_url,
                "start_services": bool(args.start),
            }
        )
    if args.hive_command == "join":
        return join_hive_from_args(args)
    if args.hive_command == "list":
        return hive_profiles.load_profiles()
    if args.hive_command == "switch":
        return theseus_setup_wizard.switch_hive({"profile_id": args.profile_id, "restart_services": bool(args.restart)})
    if args.hive_command == "invite":
        return create_active_invite(args.out, include_token=not args.no_token)
    if args.hive_command == "bootstrap":
        return bootstrap_command(args)
    if args.hive_command == "add-user":
        user_args = argparse.Namespace(
            user_id=args.user_id,
            name=args.name,
            role=args.role,
            device_label=args.device_label,
            expires_days=args.expires_days,
            token="",
            replace=bool(args.replace),
            out=args.out,
        )
        report = hive_users.create_user(read_json(POLICY_PATH, {}), user_args)
        if args.out:
            write_json(resolve_path(args.out), report)
        return report
    if args.hive_command == "users":
        return hive_users.list_users(read_json(POLICY_PATH, {}))
    if args.hive_command == "revoke-user":
        report = hive_users.revoke_user(read_json(POLICY_PATH, {}), args.user_id)
        write_json(resolve_path(args.out), report)
        return report
    if args.hive_command == "upgrade-relay":
        return theseus_setup_wizard.upgrade_active_hive_to_relay({"relay_url": args.relay_url, "start_services": bool(args.start)})
    if args.hive_command == "training-link":
        command = [
            python_executable(),
            "scripts/hive_training_link_doctor.py",
            "--out",
            args.out,
            "--markdown-out",
            args.markdown_out,
        ]
        if args.refresh:
            command.append("--refresh")
        result = run_json(command, timeout=300)
        result.setdefault("ok", result.get("state") in {"GREEN", "YELLOW"})
        return result
    if args.hive_command == "network-doctor":
        return hive_network_doctor.doctor_report(
            policy=read_json(POLICY_PATH, {}),
            timeout=float(args.timeout),
            coordinator_urls=[str(item) for item in args.coordinator_url or []],
            peer_urls=[str(item) for item in args.peer_url or []],
            out=resolve_path(args.out),
            markdown_out=resolve_path(args.markdown_out),
            write_report=True,
        )
    if args.hive_command == "macos-release-gate":
        command = [
            python_executable(),
            "scripts/hive_macos_release_gate.py",
            "--api-url",
            args.api_url,
            "--timeout",
            str(args.timeout),
            "--out",
            args.out,
            "--markdown-out",
            args.markdown_out,
        ]
        for flag in [
            "execute",
            "skip_build",
            "skip_version_publish",
            "skip_local_install",
            "skip_local_converge",
            "skip_deps",
            "no_require_mlx",
        ]:
            if bool(getattr(args, flag, False)):
                command.append("--" + flag.replace("_", "-"))
        for url in args.coordinator_url or []:
            command.extend(["--coordinator-url", str(url)])
        for url in args.peer_url or []:
            command.extend(["--peer-url", str(url)])
        result = run_json(command, timeout=7200)
        result.setdefault("ok", bool(result.get("private_canary_ready")))
        return result
    return {"ok": False, "error": "hive_subcommand_required"}


def join_hive_from_args(args: argparse.Namespace) -> dict[str, Any]:
    body: dict[str, Any] = {"start_services": bool(args.start)}
    if args.invite:
        body["invite"] = read_json(resolve_path(args.invite), {})
    elif args.invite_json:
        body["invite_text"] = args.invite_json
    else:
        body.update(
            {
                "hive_id": args.hive_id,
                "join_token": args.join_token,
                "relay_url": args.relay_url,
                "coordinator_url": getattr(args, "coordinator_url", ""),
                "name": args.name,
                "tier": args.tier,
            }
        )
    report = theseus_setup_wizard.join_hive(body)
    if report.get("ok"):
        report["next_commands"] = [
            "theseus start --restart",
            "theseus device list",
            "theseus hive network-doctor",
        ]
    return report


def device_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.device_command == "invite":
        return create_active_invite(args.out, include_token=not args.no_token)
    if args.device_command == "register":
        return theseus_setup_wizard.join_hive({"invite": read_json(resolve_path(args.invite), {}), "start_services": bool(args.start)})
    if args.device_command == "list":
        peers = read_json(REPORTS / "hive_peers.json", {})
        if port_open("127.0.0.1", 8791):
            peers = fetch_json("http://127.0.0.1:8791/api/hive/peers") or peers
        return {"ok": True, "policy": "project_theseus_cli_device_list_v0", "created_utc": now(), **(peers if isinstance(peers, dict) else {})}
    return {"ok": False, "error": "device_subcommand_required"}


def remote_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.remote_command in {None, "status"}:
        report = hive_remote_access.status_report(write_report=True)
        out = str(getattr(args, "out", "") or "")
        if out:
            write_json(resolve_path(out), report)
        return report
    if args.remote_command == "configure-relay":
        report = hive_remote_access.configure_relay_url(args.relay_url, out=args.out)
        if report.get("ok") and args.start:
            report["service_report"] = theseus_setup_wizard.start_services({"start_relay": True, "restart": bool(args.restart)})
        return report
    if args.remote_command == "wireguard-guide":
        return hive_remote_access.write_wireguard_guide(out=args.out)
    if args.remote_command == "mobile-profile":
        return hive_remote_access.write_mobile_roaming_profile(out=args.out, include_token=not bool(args.no_token))
    if args.remote_command == "doctor":
        return hive_network_doctor.doctor_report(
            policy=read_json(POLICY_PATH, {}),
            timeout=float(args.timeout),
            coordinator_urls=[str(item) for item in args.coordinator_url or []],
            peer_urls=[str(item) for item in args.peer_url or []],
            out=resolve_path(args.out),
            markdown_out=resolve_path(args.markdown_out),
            write_report=True,
        )
    return {"ok": False, "error": "remote_subcommand_required"}


def control_command(args: argparse.Namespace) -> dict[str, Any]:
    policy = read_json(POLICY_PATH, {})
    if args.control_command in {None, "status"}:
        report = hive_remote_control.status_report(policy=policy, write_report=True)
        out = str(getattr(args, "out", "") or "")
        if out:
            write_json(resolve_path(out), report)
        return report
    if args.control_command == "request":
        report = hive_remote_control.request_session(
            policy=policy,
            payload={
                "target_node_id": args.target_node,
                "provider": args.provider,
                "mode": args.mode,
                "duration_minutes": args.duration_minutes,
            },
        )
        if args.out:
            write_json(resolve_path(args.out), report)
        return report
    if args.control_command == "launch":
        report = hive_remote_control.launch_client(
            provider=args.provider,
            host=args.host,
            target_url=args.target_url,
            rustdesk_id=args.rustdesk_id,
            execute=bool(args.execute),
        )
        if args.out:
            write_json(resolve_path(args.out), report)
        return report
    return {"ok": False, "error": "control_subcommand_required"}


def storage_command(args: argparse.Namespace) -> dict[str, Any]:
    policy = read_json(POLICY_PATH, {})
    if args.storage_command in {None, "status"}:
        report = hive_storage.status_report(policy=policy, write_report=True)
        out = str(getattr(args, "out", "") or "")
        if out:
            write_json(resolve_path(out), report)
        return report
    if args.storage_command == "index":
        report = hive_storage.index_report(policy=policy, limit=int(args.limit or 500), write_report=True)
        out = str(getattr(args, "out", "") or "")
        if out:
            write_json(resolve_path(out), report)
        return report
    if args.storage_command == "add-share":
        return hive_storage.add_share(
            policy=policy,
            path=args.path,
            name=args.name,
            share_id=args.share_id,
            tags=args.tag,
            writable=bool(args.writable),
        )
    if args.storage_command == "remove-share":
        return hive_storage.remove_share(policy=policy, share_id=args.share_id)
    if args.storage_command == "browse":
        report = hive_storage.browse_share(policy=policy, share_id=args.share_id, rel_path=args.path, limit=int(args.limit or 200))
        if args.out:
            write_json(resolve_path(args.out), report)
        return report
    if args.storage_command == "pull":
        return hive_storage.pull_file(policy=policy, peer_url=args.peer_url, share_id=args.share_id, rel_path=args.path, out=args.out)
    return {"ok": False, "error": "storage_subcommand_required"}


def voice_command(args: argparse.Namespace) -> dict[str, Any]:
    policy = read_json(POLICY_PATH, {})
    if args.voice_command in {None, "status"}:
        report = hive_voice_following.status_report(policy=policy, write_report=True)
        out = str(getattr(args, "out", "") or "")
        if out:
            write_json(resolve_path(out), report)
        return report
    if args.voice_command == "configure-room":
        return hive_voice_following.configure_room(
            policy=policy,
            room_id=args.room_id,
            room_name=args.room_name,
            microphone=args.microphone,
            speaker=args.speaker,
            priority=int(args.priority or -1),
        )
    if args.voice_command == "presence":
        report = hive_voice_following.presence_update(
            policy=policy,
            payload={
                "score": args.score,
                "source": args.source,
                "room_id": args.room_id,
                "room_name": args.room_name,
                "rms_db": args.rms_db,
                "direction_degrees": args.direction,
            },
            requester={"source": "cli"},
            write_report=True,
        )
        out = str(getattr(args, "out", "") or "")
        if out:
            write_json(resolve_path(out), report)
        return report
    if args.voice_command == "route":
        report = hive_voice_following.route_decision(policy=policy, write_report=True)
        out = str(getattr(args, "out", "") or "")
        if out:
            write_json(resolve_path(out), report)
        return report
    return {"ok": False, "error": "voice_subcommand_required"}


def spatial_command(args: argparse.Namespace) -> dict[str, Any]:
    policy = read_json(POLICY_PATH, {})
    if args.spatial_command in {None, "status"}:
        report = hive_spatial.status_report(policy=policy, write_report=True)
        out = str(getattr(args, "out", "") or "")
        if out:
            write_json(resolve_path(out), report)
        return report
    if args.spatial_command == "configure-node":
        report = hive_spatial.configure_node(
            policy=policy,
            room_id=args.room_id,
            room_name=args.room_name,
            zone=args.zone,
            x=args.x,
            y=args.y,
            z=args.z,
            yaw_degrees=args.yaw,
            display=args.display,
            surfaces=args.surface,
            nearby_devices=args.device,
        )
        out = str(getattr(args, "out", "") or "")
        if out:
            write_json(resolve_path(out), hive_spatial.status_report(policy=policy, write_report=True))
        return report
    return {"ok": False, "error": "spatial_subcommand_required"}


def usb_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.usb_command == "list":
        return {"ok": True, "policy": "project_theseus_usb_targets_v0", "created_utc": now(), "targets": hive_usb_writer.list_usb_drives()}
    if args.usb_command in {None, "write"}:
        try:
            report = hive_usb_writer.build_bundle(
                out=resolve_path(args.out),
                coordinator_url=str(args.coordinator_url or ""),
                invite=resolve_path(args.invite) if args.invite else None,
                expires_days=int(args.expires_days or 30),
                include_heavy_data=bool(args.include_heavy_data),
                zip_bundle=not bool(args.no_zip),
                force=bool(args.force),
                hive_mode=str(args.hive_mode),
                new_hive_name=str(args.new_hive_name or ""),
                new_hive_tier=str(args.new_hive_tier),
                activate_new_hive=not bool(args.no_activate_new_hive),
                public_gateway_url=str(args.public_gateway_url or ""),
                public_mode=str(args.public_mode),
                public_worker_name=str(args.public_worker_name or ""),
                target=resolve_path(args.target) if args.target else None,
                confirm_label=str(args.confirm_label or ""),
                yes=bool(args.yes),
                usb_label=str(args.usb_label or ""),
                dry_run=bool(args.dry_run),
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        write_json(REPORTS / "theseus_usb_writer_last.json", report)
        return report
    return {"ok": False, "error": "usb_subcommand_required"}


def public_command(args: argparse.Namespace) -> dict[str, Any]:
    policy = read_json(POLICY_PATH, {})
    if args.public_command in {None, "status"}:
        return public_hive_contributor.status_report(policy)
    if args.public_command == "configure":
        class ConfigureArgs:
            mode = args.mode
            gateway_url = args.gateway_url
            worker_name = args.worker_name
            allow = args.allow

        return public_hive_contributor.configure_contribution(policy, ConfigureArgs())
    if args.public_command == "poll-once":
        return public_hive_contributor.poll_once(policy)
    if args.public_command == "work-smoke":
        return public_hive_contributor.work_smoke(policy)
    return {"ok": False, "error": "public_subcommand_required"}


def market_command(args: argparse.Namespace) -> dict[str, Any]:
    policy = compute_market.read_json(compute_market.POLICY_PATH, {})
    if args.market_command in {None, "status"}:
        return compute_market.status_report(policy=policy, write_report=True)
    if args.market_command == "quote":
        return compute_market.quote_task(
            args.task_kind,
            parse_json(args.payload_json, {}),
            read_json(REPORTS / "hive_status.json", {}),
            policy=policy,
            write_report=True,
        )
    if args.market_command == "settle":
        return compute_market.settle_worker_ledger(resolve_path(args.worker_ledger), limit=int(args.limit or 50), policy=policy, write_report=True)
    if args.market_command == "rent-plan":
        return compute_market.rent_plan(
            args.task_kind,
            parse_json(args.payload_json, {}),
            max_gas_micro_twc=int(args.max_gas_micro_twc or 0),
            policy=policy,
            write_report=True,
        )
    return {"ok": False, "error": "market_subcommand_required"}


def rent_command(args: argparse.Namespace) -> dict[str, Any]:
    policy = read_json(POLICY_PATH, {})
    config = read_json(hive_rented_compute.LOCAL_CONFIG_PATH, {})
    if args.rent_command in {None, "status"}:
        return hive_rented_compute.status_report(policy=policy, config=config, write_report=True, out=args.out)
    if args.rent_command == "init":
        return hive_rented_compute.init_config(args, policy=policy)
    if args.rent_command == "plan":
        return hive_rented_compute.build_plan(
            profile_name=args.profile,
            task_kind=args.task_kind,
            hours=float(args.hours or 0),
            estimated_hourly_usd=float(args.estimated_hourly_usd or 0),
            ignore_conditions=bool(args.ignore_conditions),
            policy=policy,
            config=config,
            out=args.out,
        )
    if args.rent_command == "launch":
        return hive_rented_compute.launch_plan(plan_path=resolve_path(args.plan), execute=bool(args.execute), out=args.out)
    if args.rent_command == "stop":
        return hive_rented_compute.stop_capacity(
            profile_name=args.profile,
            provider=args.provider,
            instance_id=args.instance_id,
            bucket=args.bucket,
            resource_name=args.resource_name,
            execute=bool(args.execute),
            policy=policy,
            config=config,
            out=args.out,
        )
    return {"ok": False, "error": "rent_subcommand_required"}


def create_active_invite(out: str, *, include_token: bool) -> dict[str, Any]:
    active = hive_profiles.active_profile()
    if not active:
        return {"ok": False, "error": "no_active_hive"}
    invite = theseus_setup_wizard.invite_from_profile(active, include_token=include_token)
    invite_path = resolve_path(out) if out else REPORTS / f"hive_invite_{active.get('profile_id', 'active')}.json"
    write_json(invite_path, invite)
    report = {
        "ok": True,
        "policy": "project_theseus_cli_invite_v0",
        "created_utc": now(),
        "invite_path": str(invite_path),
        "hive_id": invite.get("hive_id"),
        "hive_name": invite.get("hive_name"),
        "tier": invite.get("tier"),
        "relay_url": invite.get("relay_url"),
        "token_included": bool(include_token),
        "phone_url": theseus_setup_wizard.phone_join_url(active),
    }
    write_json(REPORTS / "theseus_cli_invite_last.json", report)
    return report


def default_cli_target_dir() -> Path:
    if platform.system() == "Windows":
        return Path.home() / "bin"
    return Path.home() / ".local" / "bin"


def python_executable() -> str:
    return theseus_setup_wizard.python_executable()


def spawn(command: list[str], label: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"cwd": ROOT}
    if platform.system() == "Windows":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
    kwargs["env"] = theseus_runtime.runtime_env()
    proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)
    return {"label": label, "pid": proc.pid, "command": command[:3] + ["..."] if len(command) > 3 else command}


def run_json(command: list[str], *, timeout: int = 300, env: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout, env=env or theseus_runtime.runtime_env())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc), "command": command}
    value = parse_json(result.stdout.strip(), {})
    if isinstance(value, dict) and value:
        value.setdefault("ok", result.returncode == 0)
        if result.returncode != 0:
            value.setdefault("stderr_tail", result.stderr[-2000:])
        return value
    return {
        "ok": result.returncode == 0,
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def hive_env() -> dict[str, str]:
    env = os.environ.copy()
    active = hive_profiles.active_profile()
    if active:
        if active.get("join_token") and not env.get("THESEUS_HIVE_SECRET"):
            env["THESEUS_HIVE_SECRET"] = str(active.get("join_token"))
        if active.get("hive_id") and not env.get("THESEUS_HIVE_ID"):
            env["THESEUS_HIVE_ID"] = str(active.get("hive_id"))
        if active.get("relay_url") and not env.get("THESEUS_HIVE_RELAY_URL"):
            env["THESEUS_HIVE_RELAY_URL"] = str(active.get("relay_url"))
        if active.get("tier") and not env.get("THESEUS_HIVE_TIER"):
            env["THESEUS_HIVE_TIER"] = str(active.get("tier"))
    return env


def fetch_json(url: str) -> dict[str, Any]:
    try:
        with urlrequest.urlopen(url, timeout=2) as response:  # noqa: S310 - local/private Hive endpoints.
            raw = response.read().decode("utf-8")
    except (URLError, TimeoutError, OSError):
        return {}
    value = parse_json(raw, {})
    return value if isinstance(value, dict) else {}


def port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def emit(report: dict[str, Any], *, json_output: bool, command: str) -> None:
    if json_output:
        print(json.dumps(report, indent=2))
        return
    if command == "status":
        print_status(report)
    elif command == "chat":
        response = report.get("response") if isinstance(report.get("response"), dict) else {}
        print(report.get("assistant_text") or response.get("answer") or json.dumps(report, indent=2))
    elif command == "mac" and isinstance(report.get("text"), str):
        print(report["text"])
    else:
        print(json.dumps(report, indent=2))


def print_status(report: dict[str, Any]) -> None:
    services = report.get("services", {})
    active = report.get("active_hive", {})
    candidate = report.get("candidate", {})
    benches = report.get("benchmarks", {})
    public = report.get("public_contribution", {})
    market = report.get("compute_market", {})
    rented = report.get("rented_compute", {})
    utilization = report.get("utilization", {})
    remote = report.get("remote_access", {})
    remote_control = report.get("remote_control", {})
    voice = report.get("voice_following", {})
    openai = report.get("openai_compat", {})
    updates = report.get("updates", {})
    runtime = report.get("runtime_paths", {})
    print("Project Theseus Hive")
    print(f"Root: {report.get('root')}")
    print(
        "Services: "
        f"dashboard={'up' if get_path(services, ['dashboard', 'live'], False) else 'down'}, "
        f"hive={'up' if get_path(services, ['hive_node', 'live'], False) else 'down'}, "
        f"relay={'up' if get_path(services, ['hive_relay', 'live'], False) else 'down'}"
    )
    if active:
        print(f"Active Hive: {active.get('name')} ({active.get('tier')}, {active.get('mode')})")
    else:
        print("Active Hive: none")
    print(f"Benchmarks: {benches.get('frontier', 0)} frontier, {benches.get('regression', 0)} regression, {benches.get('total', 0)} total")
    print(f"Candidate: promote={candidate.get('promote')} passed={candidate.get('passed')}/{candidate.get('total')}")
    if runtime:
        print(f"Runtime: {runtime.get('runtime_root') or '--'}")
    failed = candidate.get("failed_gates") or []
    if failed:
        print("Blocked by: " + ", ".join(failed))
    print(f"Public contribution: {public.get('mode')} ({public.get('next_action')})")
    print(
        "Compute market: "
        f"{market.get('mode') or 'accounting'} "
        f"{format_micro(market.get('available_micro_twc') or 0)} {market.get('currency_symbol') or 'TWC'} available, "
        f"exchange={'on' if market.get('exchange_enabled') else 'off'}"
    )
    print(
        "Rented compute: "
        f"{rented.get('configured_profile_count') or 0} profile(s), "
        f"AWS/GCP/Azure/Vast/curl="
        f"{'y' if rented.get('aws_cli_installed') else 'n'}/"
        f"{'y' if rented.get('gcloud_cli_installed') else 'n'}/"
        f"{'y' if rented.get('azure_cli_installed') else 'n'}/"
        f"{'y' if rented.get('vastai_cli_installed') else 'n'}/"
        f"{'y' if rented.get('curl_installed') else 'n'}, "
        f"last_plan={rented.get('last_plan_decision') or '--'}"
    )
    print(
        "Utilization: "
        f"state={utilization.get('trigger_state') or '--'} "
        f"idle={utilization.get('idle_slots') or {}} "
        f"busy={utilization.get('busy_slots') or {}} "
        f"planned={utilization.get('planned_actions') or 0}"
    )
    print(
        "Remote access: "
        f"{'relay configured' if remote.get('relay_configured') else 'LAN/hotspot only'} "
        f"scope={remote.get('relay_scope') or 'not_configured'} "
        f"paid_dependency={'yes' if remote.get('paid_dependency_required') else 'no'}"
    )
    print(
        "Remote control: "
        f"{remote_control.get('ready_provider_count') or 0} ready "
        f"preferred={remote_control.get('preferred_provider_id') or '--'}"
    )
    print(
        "Voice following: "
        f"{voice.get('room_name') or '--'} "
        f"mic={'ready' if voice.get('microphone_ready') else 'off'} "
        f"speaker={'ready' if voice.get('speaker_ready') else 'off'} "
        f"route={voice.get('route_state') or '--'}"
    )
    print(f"OpenAI-compatible endpoint: {'live' if openai.get('live') else 'off'} ({openai.get('base_url') or 'not configured'})")
    print(
        "Updates: "
        f"{'available' if updates.get('update_available') else 'current/check'} "
        f"soft={updates.get('soft_update_available')} hard={updates.get('hard_update_available')} "
        f"checkpoint={updates.get('offer_checkpoint_id') or updates.get('installed_checkpoint_id') or '--'}"
    )


def format_micro(value: Any) -> str:
    try:
        number = int(value) / 1_000_000.0
    except (TypeError, ValueError):
        number = 0.0
    if abs(number) >= 1:
        return f"{number:.3f}"
    text = f"{number:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def resolve_path(path: str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_json(raw: str, default: Any) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
