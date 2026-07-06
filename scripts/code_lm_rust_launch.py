"""Rust/SymLiquid launch helpers for Code LM closure.

Hot-path launch policy lives here so orchestration, visible task export,
and Rust process setup can evolve independently.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from code_lm_source_fingerprint import decoder_source_paths

ROOT = Path(__file__).resolve().parents[1]


def release_binary(profile: str = "release") -> Path:
    name = "symliquid-cli.exe" if sys.platform.startswith("win") else "symliquid-cli"
    return ROOT / "target" / profile / name


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def rust_closure_command(args: argparse.Namespace) -> list[str]:
    profile = str(getattr(args, "rust_build_profile", "release") or "release")
    override = os.environ.get("THESEUS_SYMLIQUID_CLI", "").strip()
    if override:
        override_path = Path(override)
        if override_path.exists():
            prefix = [str(override_path)]
        else:
            prefix = []
    else:
        prefix = []
    exe = release_binary(profile)
    source_files = [
        ROOT / "crates" / "symliquid-cli" / "src" / "main.rs",
        ROOT / "crates" / "symliquid-cli" / "src" / "sts_parallel_decoder.rs",
        ROOT / "crates" / "symliquid-cuda" / "src" / "lib.rs",
        ROOT / "crates" / "symliquid-cuda" / "src" / "readout_cuda.rs",
        ROOT / "crates" / "symliquid-cuda" / "kernels" / "readout_kernels.cu",
    ] + decoder_source_paths()
    if not prefix:
        exe_fresh = exe.exists() and all(exe.stat().st_mtime >= path.stat().st_mtime for path in source_files if path.exists())
        allow_cargo_hot_path = os.environ.get("THESEUS_ALLOW_CARGO_HOT_PATH", "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if bool(getattr(args, "use_cuda_readout", False)):
            if exe.exists() and cuda_release_exe_ready(exe):
                prefix = [str(exe)]
            elif allow_cargo_hot_path:
                prefix = ["cargo", "run", "--release", "-p", "symliquid-cli", "--features", "cuda", "--"]
            else:
                raise RuntimeError(
                    f"CUDA Code LM hot path requires {rel(exe)}; "
                    "build it explicitly with cargo build --release -p symliquid-cli --features cuda "
                    "or set THESEUS_ALLOW_CARGO_HOT_PATH=1 for a diagnostic fallback."
                )
        elif exe_fresh:
            prefix = [str(exe)]
        elif profile == "release":
            if exe.exists():
                prefix = [str(exe)]
            elif allow_cargo_hot_path:
                prefix = ["cargo", "run", "--release", "-p", "symliquid-cli", "--"]
            else:
                raise RuntimeError(
                    f"Code LM hot path requires {rel(exe)}; "
                    "build it explicitly or set THESEUS_ALLOW_CARGO_HOT_PATH=1 for a diagnostic fallback."
                )
        else:
            if allow_cargo_hot_path:
                prefix = ["cargo", "run", "-p", "symliquid-cli", "--"]
            else:
                raise RuntimeError(
                    "debug cargo hot path disabled; set THESEUS_ALLOW_CARGO_HOT_PATH=1 for diagnostics."
                )
    command = [
        *prefix,
        "train-code-lm-closure",
        "--private-curriculum",
        str(resolve(args.private_curriculum_out)),
        "--public-task-manifest",
        str(resolve(args.public_task_manifest_out)),
        "--seed",
        str(args.seed),
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
        "--checkpoint-out",
        rel(resolve(args.checkpoint_out)),
        "--checkpoint-in",
        rel(resolve(args.checkpoint_in)) if str(args.checkpoint_in or "").strip() else "",
        "--private-candidate-out",
        rel(resolve(args.private_candidate_out)),
        "--public-candidate-out",
        rel(resolve(args.public_candidate_out)),
        "--report-out",
        rel(resolve(args.rust_report_out)),
    ]
    if int(getattr(args, "max_rust_work_steps", 0) or 0) > 0:
        command.extend(["--max-work-steps", str(int(args.max_rust_work_steps))])
    if bool(getattr(args, "use_cuda_readout", False)):
        command.append("--use-cuda-readout")
    if bool(getattr(args, "checkpoint_only", False)):
        command.append("--checkpoint-only")
    sts_streams = str(getattr(args, "sts_streams_effective", "") or "")
    if sts_streams:
        command.extend(["--sts-streams", str(resolve(sts_streams))])
    return command


def cuda_release_exe_ready(exe: Path) -> bool:
    if not exe.exists():
        return False
    try:
        result = subprocess.run([str(exe), "--help"], cwd=ROOT, text=True, capture_output=True, timeout=10)
    except Exception:
        return False
    help_text = f"{result.stdout}\n{result.stderr}".lower()
    return result.returncode == 0 and "train-standalone-cuda" in help_text and "train-code-lm-closure" in help_text


def timeout_arg(seconds: int | float | None) -> float | None:
    """Return subprocess timeout as a safety fuse, never as the run duration.

    Theseus learning lanes are supposed to be controlled by explicit work
    steps: rows, epochs, candidates, and cases. A wall-clock timeout is only a
    runaway guard, so 0 disables it and positive values are used verbatim except
    for a small floor that prevents accidental near-immediate termination.
    """

    value = int(seconds or 0)
    if value <= 0:
        return None
    return float(max(60, value))


def symliquid_process_env(args: argparse.Namespace | None = None) -> dict[str, str]:
    env = os.environ.copy()
    # Large private high-transfer curricula can build deep iterator/JSON stacks
    # on Windows debug builds. Give Rust workers a real stack so verification
    # probes and unattended runs fail on model issues, not launcher limits.
    env.setdefault("RUST_MIN_STACK", str(256 * 1024 * 1024))
    env.setdefault("RUST_BACKTRACE", "1")
    # Keep transfer-critical families from looking "covered" after only a
    # handful of rows. The Rust learner can still evict broad execution rows to
    # stay inside the explicit work-step lease.
    env.setdefault("THESEUS_TARGET_FAMILY_STARVATION_RESCUE_MIN_ROWS", "24")
    # Normal Code LM closures must measure learned token generation, not
    # handcrafted contract/skeleton helpers. Diagnostic scripts may intentionally
    # omit this env var, but production learning and public calibration run
    # template-free by default.
    env.setdefault("THESEUS_TEMPLATE_FREE_STUDENT_CANDIDATES", "1")
    if typed_edge_exec_receiver_enabled(args):
        env["THESEUS_TYPED_EDGE_EXEC_RECEIVER_V1"] = "1"
    if private_type_shape_receiver_veto_enabled(args):
        env["THESEUS_PRIVATE_TYPE_SHAPE_RECEIVER_VETO_V1"] = "1"
    return env


def typed_edge_exec_receiver_enabled(args: argparse.Namespace | None = None) -> bool:
    return bool(
        os.environ.get("THESEUS_TYPED_EDGE_EXEC_RECEIVER_V1")
        or bool(getattr(args, "typed_edge_exec_receiver_v1", False))
    )


def private_type_shape_receiver_veto_enabled(args: argparse.Namespace | None = None) -> bool:
    return bool(
        os.environ.get("THESEUS_PRIVATE_TYPE_SHAPE_RECEIVER_VETO_V1")
        or bool(getattr(args, "private_type_shape_receiver_veto_v1", False))
    )


def build_step_plan(
    args: argparse.Namespace,
    private_rows: list[dict[str, Any]],
    public_tasks: list[dict[str, Any]],
    sts_conditioning: dict[str, Any],
) -> dict[str, Any]:
    train_rows = [row for row in private_rows if row.get("split") == "train"]
    eval_rows = [row for row in private_rows if row.get("split") == "eval"]
    public_task_count = len(public_tasks)
    candidate_budget = max(1, int(args.candidates_per_task))
    epochs = max(1, int(args.epochs))
    estimated_train_token_steps = sum(
        max(1, len(str(row.get("solution_body") or row.get("solution_expr") or "").split()))
        * semantic_training_repeat_cost(row)
        for row in train_rows
    ) * epochs
    estimated_private_candidate_steps = len(eval_rows) * candidate_budget
    estimated_public_candidate_steps = public_task_count * candidate_budget
    estimated_public_sandbox_steps = public_task_count * candidate_budget * 2
    estimated_sts_rows = int(sts_conditioning.get("conditioned_public_task_count") or 0)
    estimated_work_steps = (
        estimated_train_token_steps
        + estimated_private_candidate_steps
        + estimated_public_candidate_steps
        + estimated_public_sandbox_steps
        + estimated_sts_rows
    )
    max_rust_work_steps = int(getattr(args, "max_rust_work_steps", 0) or 0)
    return {
        "policy": "project_theseus_step_duration_v1",
        "duration_control": "step_budget_primary_wall_clock_safety_fuse_only",
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "public_task_count": public_task_count,
        "epochs": epochs,
        "candidates_per_task": candidate_budget,
        "estimated_train_token_steps": estimated_train_token_steps,
        "semantic_repeat_accounting": True,
        "estimated_private_candidate_steps": estimated_private_candidate_steps,
        "estimated_public_candidate_steps": estimated_public_candidate_steps,
        "estimated_public_sandbox_steps": estimated_public_sandbox_steps,
        "estimated_sts_conditioning_rows": estimated_sts_rows,
        "estimated_total_work_steps": estimated_work_steps,
        "max_rust_work_steps": max_rust_work_steps,
        "max_rust_work_steps_active": max_rust_work_steps > 0,
        "wall_clock_safety_fuses": {
            "rust_timeout_seconds": int(args.rust_timeout_seconds),
            "public_timeout_seconds": int(args.public_timeout_seconds),
            "sts_timeout_seconds": int(args.sts_timeout_seconds),
            "zero_disables_wall_clock_fuse": True,
        },
        "score_semantics": "planning metadata only; not learning evidence",
    }


def semantic_training_repeat_cost(row: dict[str, Any]) -> int:
    category = str(row.get("category") or "")
    tags = {str(tag) for tag in row.get("tags", []) if str(tag)}
    hard = {
        "palindrome",
        "caesar_decode_shift5",
        "below_threshold",
        "add_numbers",
        "same_chars",
        "gcd_pair",
        "is_prime",
        "is_anagram",
        "base_digits",
        "median_list",
        "triangle_area_sides",
        "frequency_at_least_value",
        "hex_prime_count",
    }
    if category in hard or "semantic_residual" in tags:
        return 15
    if any(token in category for token in ["recurrence", "vowel", "digit_rotate", "digit_shift"]):
        return 9
    return 3

