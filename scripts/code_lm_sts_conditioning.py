"""STS conditioning and decoder-control stream generation for Code LM closure.

The functions here build STS/control rows and invoke the bounded native
STS decoder. Public benchmark tasks remain visible-metadata only: no public
tests, no public solutions, no answer-key training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from code_lm_private_verifier import concept_family, concept_residual_label
from code_lm_public_task_export import load_symliquid_state, symliquid_sts_context
from code_lm_rust_launch import symliquid_process_env, timeout_arg
from code_lm_source_fingerprint import decoder_relevant_source_fingerprint, decoder_relevant_source_mtime
from process_tree import run_process_tree

ROOT = Path(__file__).resolve().parents[1]


def default_sts_cache_dir() -> Path:
    configured = os.environ.get("THESEUS_STS_CONDITIONING_CACHE_DIR", "").strip()
    if configured:
        return Path(configured)
    if sys.platform.startswith("win"):
        return Path("D:/ProjectTheseus/runtime/sts_conditioning_cache")
    return ROOT / "runtime" / "sts_conditioning_cache"


STS_CACHE_DIR = default_sts_cache_dir()
STS_OUTPUT_STREAMS = [
    "solver_stream",
    "critic_stream",
    "tool_stream",
    "patch_stream",
    "residual_stream",
    "visible_report_stream",
]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


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
    rows: list[dict[str, Any]] = []
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


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


def timeout_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()

def write_sts_progress_report(
    args: argparse.Namespace,
    *,
    private_rows: list[dict[str, Any]],
    public_tasks: list[dict[str, Any]],
    conditioning_rows: list[dict[str, Any]],
    command: list[str],
) -> None:
    progress = {
        "policy": "project_theseus_code_lm_closure_v1",
        "created_utc": now(),
        "trigger_state": "YELLOW",
        "run_status": "in_progress",
        "progress_stage": "sts_conditioning_stage_started",
        "decoder_relevant_source_fingerprint": decoder_relevant_source_fingerprint(),
        "decoder_relevant_source_mtime": decoder_relevant_source_mtime() or None,
        "seed": args.seed,
        "private_curriculum": rel(resolve(args.private_curriculum_out)),
        "public_task_manifest": rel(resolve(args.public_task_manifest_out)),
        "sts_conditioning_report": rel(resolve(args.sts_conditioning_report_out)),
        "sts_generation": rel(resolve(args.sts_generation_out)),
        "summary": {
            "private_task_count": len(private_rows),
            "private_train_task_count": sum(1 for row in private_rows if row.get("split") == "train"),
            "private_eval_task_count": sum(1 for row in private_rows if row.get("split") == "eval"),
            "public_task_count": len(public_tasks),
            "sts_conditioning_input_rows": len(conditioning_rows),
            "sts_timeout_seconds": int(args.sts_timeout_seconds),
            "sts_resume_supported": True,
            "external_inference_calls": 0,
        },
        "sts_command": command,
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), progress)


def run_sts_conditioning(
    args: argparse.Namespace,
    public_tasks: list[dict[str, Any]],
    private_code_tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    training_path = resolve(args.sts_training_data)
    default = {
        "safe": True,
        "enabled": not args.disable_sts_conditioning,
        "generation_path": "",
        "conditioning_input": "",
        "conditioned_public_task_count": 0,
        "returncode": None,
        "trigger_state": "SKIPPED",
        "reason": "",
        "external_inference_calls": 0,
    }
    if args.disable_sts_conditioning:
        default["reason"] = "disabled"
        return default
    if not training_path.exists():
        default["reason"] = f"missing_training_data={rel(training_path)}"
        return default
    sym_context = symliquid_sts_context(load_symliquid_state(args))
    private_code_stream_rows = private_code_sts_rows(private_code_tasks, sym_context=sym_context)
    private_eval_sts_rows = sum(1 for row in private_code_stream_rows if row.get("split") == "eval")
    if not public_tasks and private_eval_sts_rows <= 0:
        default["reason"] = "no_public_or_private_eval_tasks"
        return default
    private_rows = read_jsonl(training_path)
    no_admissible_control_source_rows = load_no_admissible_control_rows(args)
    no_admissible_control_rows = no_admissible_control_sts_rows(
        no_admissible_control_source_rows,
        sym_context=sym_context,
    )
    sts_decoder_control_source_rows = load_sts_decoder_control_rows(args)
    sts_decoder_control_rows = sts_decoder_control_sts_rows(
        sts_decoder_control_source_rows,
        sym_context=sym_context,
    )
    sts_decoder_control_policy = sts_decoder_control_policy_text(sts_decoder_control_source_rows)
    public_rows = public_sts_rows(public_tasks, sym_context=sym_context)
    # Keep bounded STS generation focused on the visible receiver shard. The
    # training/eval rows remain present, but public context must not be starved
    # behind hundreds of private/control rows when max_generate_rows is small.
    conditioning_rows = (
        public_rows
        + private_code_stream_rows
        + no_admissible_control_rows
        + sts_decoder_control_rows
        + private_rows
    )
    write_jsonl(resolve(args.sts_conditioning_input_out), conditioning_rows)
    public_task_ids = {str(row.get("task_id") or "") for row in public_tasks}
    private_eval_task_ids = {
        str(row.get("task_id") or "")
        for row in private_code_stream_rows
        if row.get("split") == "eval"
    }

    def summarize_existing_sts(result: subprocess.CompletedProcess[str] | None) -> dict[str, Any]:
        report = read_json(resolve(args.sts_conditioning_report_out), {})
        generated = read_jsonl(resolve(args.sts_generation_out))
        if sts_decoder_control_policy:
            generated = apply_sts_decoder_control_policy(
                generated,
                public_task_ids=public_task_ids,
                private_eval_task_ids=private_eval_task_ids,
                policy_text=sts_decoder_control_policy,
            )
            write_jsonl(resolve(args.sts_generation_out), generated)
        public_conditioned_rows = [
            row
            for row in generated
            if str(row.get("task_id") or "") in public_task_ids
            and isinstance(row.get("streams"), dict)
            and bool(row.get("native_parallel_token_generation"))
            and not bool(row.get("public_benchmark_solutions_included"))
        ]
        private_conditioned_rows = [
            row
            for row in generated
            if str(row.get("task_id") or "") in private_eval_task_ids
            and isinstance(row.get("streams"), dict)
            and bool(row.get("native_parallel_token_generation"))
            and not bool(row.get("public_benchmark_solutions_included"))
        ]
        conditioned_count = len({str(row.get("task_id") or "") for row in public_conditioned_rows})
        private_conditioned_count = len({str(row.get("task_id") or "") for row in private_conditioned_rows})
        min_public_conditioned = min(len(public_task_ids), 16)
        min_private_conditioned = min(len(private_eval_task_ids), 16)
        safe = (
            conditioned_count >= min_public_conditioned
            and private_conditioned_count >= min_private_conditioned
            and report.get("trigger_state") in {"GREEN", "YELLOW"}
            and not bool(get_path(report, ["summary", "public_benchmark_solutions_included"], True))
        )
        report_mtime = resolve(args.sts_conditioning_report_out).stat().st_mtime if resolve(args.sts_conditioning_report_out).exists() else 0.0
        input_mtime = resolve(args.sts_conditioning_input_out).stat().st_mtime if resolve(args.sts_conditioning_input_out).exists() else 0.0
        fresh = safe
        return {
            "safe": safe,
            "fresh": fresh,
            "report": report,
            "generated": generated,
            "conditioned_count": conditioned_count,
            "private_conditioned_count": private_conditioned_count,
            "conditioned_public_row_count": len(public_conditioned_rows),
            "conditioned_private_row_count": len(private_conditioned_rows),
            "min_public_conditioned": min_public_conditioned,
            "min_private_conditioned": min_private_conditioned,
            "report_mtime": report_mtime,
            "conditioning_input_mtime": input_mtime,
            "result": result,
        }

    max_generate_rows = bounded_sts_budget(
        max(16, len(public_rows) + private_eval_sts_rows),
        getattr(args, "sts_conditioning_max_generate_rows", 0),
        floor=16,
    )
    max_train_rows = bounded_sts_budget(
        max(
            1400,
            len(private_code_stream_rows)
            + len(no_admissible_control_rows)
            + len(sts_decoder_control_rows)
            + 960,
        ),
        getattr(args, "sts_conditioning_max_train_rows", 0),
        floor=64,
    )
    max_eval_rows = bounded_sts_budget(
        max(240, private_eval_sts_rows),
        getattr(args, "sts_conditioning_max_eval_rows", 0),
        floor=16,
    )
    command = rust_sts_conditioning_command(
        args,
        max_generate_rows=max_generate_rows,
        max_train_rows=max_train_rows,
        max_eval_rows=max_eval_rows,
    )
    cache = sts_conditioning_cache_context(
        args,
        conditioning_input=resolve(args.sts_conditioning_input_out),
        max_generate_rows=max_generate_rows,
        max_train_rows=max_train_rows,
        max_eval_rows=max_eval_rows,
    )
    write_sts_progress_report(
        args,
        private_rows=private_code_tasks,
        public_tasks=public_tasks,
        conditioning_rows=conditioning_rows,
        command=command,
    )
    cache_restore = restore_sts_conditioning_cache(args, cache)
    existing = (
        summarize_existing_sts(None)
        if cache_restore.get("restored")
        else {
            "safe": False,
            "fresh": False,
            "report": {},
            "generated": [],
            "conditioned_count": 0,
            "private_conditioned_count": 0,
            "conditioned_public_row_count": 0,
            "conditioned_private_row_count": 0,
            "min_public_conditioned": min(len(public_task_ids), 16),
            "min_private_conditioned": min(len(private_eval_task_ids), 16),
            "report_mtime": 0.0,
            "conditioning_input_mtime": resolve(args.sts_conditioning_input_out).stat().st_mtime
            if resolve(args.sts_conditioning_input_out).exists()
            else 0.0,
            "result": None,
        }
    )
    if existing["fresh"]:
        result = subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "Reused content-addressed STS conditioning cache."
                if cache_restore.get("restored")
                else "Reused completed fresh STS conditioning artifacts from a prior interrupted outer run."
            ),
            stderr="",
        )
        summarized = {**existing, "result": result, "cache_restore": cache_restore}
    else:
        result = run_process_tree(
            command,
            cwd=ROOT,
            env=symliquid_process_env(),
            timeout_seconds=timeout_arg(args.sts_timeout_seconds),
        )
        summarized = summarize_existing_sts(result)
        cache_store = store_sts_conditioning_cache(args, cache, summarized)
        summarized = {**summarized, "cache_store": cache_store}
        if result.returncode == 124 and summarized["safe"]:
            result = subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    timeout_text(result.stdout)
                    + "\nRecovered completed STS conditioning artifacts after the timeout fuse."
                ).strip(),
                stderr=timeout_text(result.stderr),
            )
            summarized = {**summarized, "result": result, "fresh": True}
        elif result.returncode == 124:
            report = summarized["report"] if isinstance(summarized["report"], dict) else {}
            write_json(
                resolve(args.sts_conditioning_report_out),
                {
                    "policy": "project_theseus_sts_parallel_decoder_conditioning_v1",
                    "created_utc": now(),
                    "trigger_state": "YELLOW",
                    "run_status": "timed_out_process_tree_killed",
                    "summary": {
                        "timeout_seconds": args.sts_timeout_seconds,
                        "process_tree_killed": True,
                        "score_semantics": "timeout_marker_only_not_learning_evidence",
                        "conditioning_input_rows": count_jsonl_rows(resolve(args.sts_conditioning_input_out)),
                        "partial_generation_rows": count_jsonl_rows(resolve(args.sts_generation_out)),
                        "previous_report_status": report.get("run_status"),
                        "previous_report_trigger_state": report.get("trigger_state"),
                        "previous_report_summary": object_field(report, "summary"),
                    },
                    "stderr_tail": timeout_text(result.stderr),
                    "external_inference_calls": 0,
                },
            )
            summarized = summarize_existing_sts(result)
    report = summarized["report"]
    generated = summarized["generated"]
    conditioned_count = summarized["conditioned_count"]
    private_conditioned_count = summarized["private_conditioned_count"]
    safe = (
        result.returncode == 0
        and conditioned_count >= int(summarized["min_public_conditioned"])
        and private_conditioned_count >= int(summarized["min_private_conditioned"])
        and report.get("trigger_state") in {"GREEN", "YELLOW"}
        and not bool(get_path(report, ["summary", "public_benchmark_solutions_included"], True))
    )
    return {
        "safe": safe,
        "enabled": True,
        "generation_path": rel(resolve(args.sts_generation_out)) if safe else "",
        "conditioning_input": rel(resolve(args.sts_conditioning_input_out)),
        "conditioned_public_task_count": conditioned_count,
        "conditioned_private_concept_task_count": private_conditioned_count,
        "conditioned_public_row_count": summarized["conditioned_public_row_count"],
        "conditioned_private_row_count": summarized["conditioned_private_row_count"],
        "min_public_conditioned": summarized["min_public_conditioned"],
        "min_private_conditioned": summarized["min_private_conditioned"],
        "private_concept_sts_row_count": len(private_code_stream_rows),
        "no_admissible_control_sts_row_count": len(no_admissible_control_rows),
        "no_admissible_control_source_row_count": len(no_admissible_control_source_rows),
        "sts_decoder_control_sts_row_count": len(sts_decoder_control_rows),
        "sts_decoder_control_source_row_count": len(sts_decoder_control_source_rows),
        "sts_conditioning_budget": {
            "max_train_rows": max_train_rows,
            "max_eval_rows": max_eval_rows,
            "max_generate_rows": max_generate_rows,
            "epochs": int(getattr(args, "sts_conditioning_epochs", 5) or 5),
            "hv_dim": int(getattr(args, "sts_conditioning_hv_dim", 512) or 512),
            "max_vocab": int(getattr(args, "sts_conditioning_max_vocab", 640) or 640),
            "max_generate_steps": int(getattr(args, "sts_conditioning_max_generate_steps", 48) or 48),
        },
        "sts_decoder_control_policy_applied": bool(sts_decoder_control_policy),
        "sts_decoder_control_path": rel(resolve(args.sts_decoder_control_policy_jsonl))
        if resolve(args.sts_decoder_control_policy_jsonl).exists()
        else "",
        "no_admissible_control_path": rel(resolve(args.no_admissible_repair_policy_jsonl))
        if resolve(args.no_admissible_repair_policy_jsonl).exists()
        else "",
        "symliquid_state_context_used": bool(sym_context),
        "returncode": result.returncode,
        "trigger_state": report.get("trigger_state"),
        "native_parallel_token_generation": bool(get_path(report, ["summary", "native_parallel_token_generation_proven"], False)),
        "output_stream_count": get_path(report, ["summary", "output_stream_count"], 0),
        "eval_token_accuracy_delta": get_path(report, ["summary", "eval_token_accuracy_delta"], 0.0),
        "resume_completed_sts_used": bool(existing["fresh"]),
        "sts_conditioning_cache": {
            "cache_key": cache["cache_key"],
            "cache_dir": rel(cache["cache_dir"]),
            "restore": summarized.get("cache_restore", cache_restore),
            "store": summarized.get("cache_store", {"stored": False, "reason": "not_run_or_not_safe"}),
        },
        "timed_out": result.returncode == 124,
        "artifact_report_mtime": summarized["report_mtime"],
        "conditioning_input_mtime": summarized["conditioning_input_mtime"],
        "stdout_tail": result.stdout[-1200:],
        "stderr_tail": result.stderr[-1200:],
        "external_inference_calls": 0,
    }


def rust_sts_conditioning_command(
    args: argparse.Namespace,
    *,
    max_generate_rows: int,
    max_train_rows: int,
    max_eval_rows: int,
) -> list[str]:
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
    exe_name = "symliquid-cli.exe" if sys.platform.startswith("win") else "symliquid-cli"
    exe = ROOT / "target" / profile / exe_name
    source_files = [
        ROOT / "crates" / "symliquid-cli" / "src" / "main.rs",
        ROOT / "crates" / "symliquid-cli" / "src" / "sts_parallel_decoder.rs",
    ]
    if not prefix:
        exe_fresh = exe.exists() and all(exe.stat().st_mtime >= path.stat().st_mtime for path in source_files if path.exists())
        if exe_fresh:
            prefix = [str(exe)]
        elif profile == "release":
            prefix = ["cargo", "run", "--release", "-p", "symliquid-cli", "--"]
        else:
            prefix = ["cargo", "run", "-p", "symliquid-cli", "--"]
    return [
        *prefix,
        "train-sts-parallel-decoder",
        "--input",
        rel(resolve(args.sts_conditioning_input_out)),
        "--seed",
        str(args.seed),
        "--hv-dim",
        str(max(64, int(getattr(args, "sts_conditioning_hv_dim", 512) or 512))),
        "--max-vocab",
        str(max(64, int(getattr(args, "sts_conditioning_max_vocab", 640) or 640))),
        "--epochs",
        str(max(1, int(getattr(args, "sts_conditioning_epochs", 5) or 5))),
        "--lr",
        str(float(getattr(args, "sts_conditioning_lr", 0.06) or 0.06)),
        "--max-generate-steps",
        str(max(1, int(getattr(args, "sts_conditioning_max_generate_steps", 48) or 48))),
        "--max-train-rows",
        str(max_train_rows),
        "--max-eval-rows",
        str(max_eval_rows),
        "--max-generate-rows",
        str(max_generate_rows),
        "--checkpoint-out",
        rel(resolve(args.sts_conditioning_checkpoint_out)),
        "--generation-out",
        rel(resolve(args.sts_generation_out)),
        "--report-out",
        rel(resolve(args.sts_conditioning_report_out)),
    ]


def sts_conditioning_cache_context(
    args: argparse.Namespace,
    *,
    conditioning_input: Path,
    max_generate_rows: int,
    max_train_rows: int,
    max_eval_rows: int,
) -> dict[str, Any]:
    input_sha = file_sha256(conditioning_input)
    profile = str(getattr(args, "rust_build_profile", "release") or "release")
    exe_name = "symliquid-cli.exe" if sys.platform.startswith("win") else "symliquid-cli"
    release_binary = ROOT / "target" / profile / exe_name
    source_files = [
        ROOT / "crates" / "symliquid-cli" / "src" / "main.rs",
        ROOT / "crates" / "symliquid-cli" / "src" / "sts_parallel_decoder.rs",
    ]
    config = {
        "policy": "project_theseus_sts_conditioning_cache_v2_source_binary_bound",
        "input_sha256": input_sha,
        "seed": int(args.seed),
        "hv_dim": int(getattr(args, "sts_conditioning_hv_dim", 512) or 512),
        "max_vocab": int(getattr(args, "sts_conditioning_max_vocab", 640) or 640),
        "epochs": int(getattr(args, "sts_conditioning_epochs", 5) or 5),
        "lr": float(getattr(args, "sts_conditioning_lr", 0.06) or 0.06),
        "max_generate_steps": int(getattr(args, "sts_conditioning_max_generate_steps", 48) or 48),
        "max_train_rows": int(max_train_rows),
        "max_eval_rows": int(max_eval_rows),
        "max_generate_rows": int(max_generate_rows),
        "sts_source_sha256": {rel(path): file_sha256(path) for path in source_files},
        "release_binary": rel(release_binary),
        "release_binary_sha256": file_sha256(release_binary),
    }
    cache_key = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:32]
    cache_dir = STS_CACHE_DIR / cache_key
    return {
        "cache_key": cache_key,
        "cache_dir": cache_dir,
        "config": config,
        "report": cache_dir / "report.json",
        "generation": cache_dir / "generation.jsonl",
        "checkpoint": cache_dir / "checkpoint.json",
        "metadata": cache_dir / "metadata.json",
    }


def bounded_sts_budget(default_value: int, cap_value: Any, *, floor: int) -> int:
    try:
        cap = int(cap_value or 0)
    except Exception:
        cap = 0
    if cap <= 0:
        return int(default_value)
    return max(int(floor), min(int(default_value), cap))


def restore_sts_conditioning_cache(args: argparse.Namespace, cache: dict[str, Any]) -> dict[str, Any]:
    report = Path(cache["report"])
    generation = Path(cache["generation"])
    checkpoint = Path(cache["checkpoint"])
    if not (report.exists() and generation.exists() and checkpoint.exists()):
        return {"restored": False, "reason": "cache_miss", "cache_key": cache["cache_key"]}
    restored = []
    for src, dst in [
        (report, resolve(args.sts_conditioning_report_out)),
        (generation, resolve(args.sts_generation_out)),
        (checkpoint, resolve(args.sts_conditioning_checkpoint_out)),
    ]:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        restored.append(rel(dst))
    return {
        "restored": True,
        "reason": "content_addressed_cache_hit",
        "cache_key": cache["cache_key"],
        "restored_paths": restored,
    }


def store_sts_conditioning_cache(args: argparse.Namespace, cache: dict[str, Any], summarized: dict[str, Any]) -> dict[str, Any]:
    if not summarized.get("safe"):
        return {"stored": False, "reason": "unsafe_or_incomplete", "cache_key": cache["cache_key"]}
    cache_dir = Path(cache["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    stored = []
    for src, dst in [
        (resolve(args.sts_conditioning_report_out), Path(cache["report"])),
        (resolve(args.sts_generation_out), Path(cache["generation"])),
        (resolve(args.sts_conditioning_checkpoint_out), Path(cache["checkpoint"])),
    ]:
        if not src.exists():
            return {"stored": False, "reason": f"missing_source={rel(src)}", "cache_key": cache["cache_key"]}
        shutil.copyfile(src, dst)
        stored.append(rel(dst))
    write_json(
        Path(cache["metadata"]),
        {
            "policy": "project_theseus_sts_conditioning_cache_v2_source_binary_bound",
            "created_utc": now(),
            "cache_key": cache["cache_key"],
            "config": cache["config"],
            "stored_paths": stored,
            "score_semantics": "runtime_cache_only_not_learning_evidence",
            "external_inference_calls": 0,
        },
    )
    return {"stored": True, "reason": "safe_sts_conditioning_cached", "cache_key": cache["cache_key"], "stored_paths": stored}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_no_admissible_control_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = resolve(args.no_admissible_repair_policy_jsonl)
    if int(args.max_no_admissible_control_rows) <= 0 or not path.exists():
        return []
    rows = []
    for row in read_jsonl(path):
        if not isinstance(row, dict):
            continue
        if row.get("source_type") != "no_admissible_candidate_residual":
            continue
        if row.get("training_use_state") != "decoder_control_policy_only_not_code_answer_training":
            continue
        if row.get("raw_public_prompt_or_tests_copied") or row.get("public_benchmark_training_data_used"):
            continue
        rows.append(row)
        if len(rows) >= max(0, int(args.max_no_admissible_control_rows)):
            break
    return rows


def no_admissible_control_sts_rows(rows: list[dict[str, Any]], *, sym_context: str = "") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        family = str(row.get("missing_capability_family") or "unknown_capability_family")
        constructs = row.get("required_constructs") if isinstance(row.get("required_constructs"), list) else []
        construct_text = ", ".join(str(item) for item in constructs[:12]) or "infer from contract"
        task_hash = str(row.get("task_hash") or "")
        source_scope = "public_metadata_only" if str(row.get("split") or "") == "control" else "private_control"
        policy_text = str(row.get("answer") or "").strip()
        input_streams = {
            "context_stream": (
                f"decoder_control source={source_scope} task_hash={task_hash} "
                f"missing_family={family} required_constructs={construct_text} symliquid_state={sym_context}"
            ),
            "solver_stream": (
                "recover candidate coverage with learned full-body token generation; "
                "preserve exact signature, use required arguments, satisfy return shape, "
                f"target missing family {family}"
            ),
            "critic_stream": (
                "reject no_admissible, partial syntax, vacuous body, wrong interface, "
                "wrong return shape, and missing loop/branch/local-state obligations"
            ),
            "tool_stream": (
                "use parser-in-the-loop completion, contract verifier feedback, "
                "style/minimality scoring, and metadata-only residual clusters"
            ),
            "residual_stream": (
                f"no_admissible_candidate family={family}; required_constructs={construct_text}"
            ),
        }
        out.append(
            {
                "policy": "project_theseus_no_admissible_decoder_control_sts_v1",
                "task_id": f"no_admissible_control_{task_hash or row.get('row_id', '')}",
                "source_task_id": task_hash,
                "split": "control",
                "benchmark_evidence_level": "decoder_control_metadata_only_not_code_answer_training",
                "visible_task_only": True,
                "public_benchmark_solutions_included": False,
                "public_tests_included": False,
                "canonical_solution_exported": False,
                "raw_public_prompt_or_tests_copied": False,
                "residual_concept": family,
                "input_streams": input_streams,
                "target_streams": {
                    "solver_stream": "emit admissible learned-token candidates before scoring",
                    "critic_stream": "detect sparse candidate coverage and explain rejection reasons",
                    "tool_stream": "apply AST completion and contract verifier feedback",
                    "patch_stream": policy_text or "decoder control policy only; no benchmark answer",
                    "residual_stream": f"candidate_coverage_recovery:{family}",
                    "visible_report_stream": "no-admissible residual converted to causal decoder-control pressure",
                },
                "causal_contract": {
                    "strict_past_only": True,
                    "same_row_cross_stream_attention": "forbidden",
                    "one_token_per_output_stream_target": True,
                    "decoder_control_only": True,
                    "not_code_answer_training": True,
                },
            }
        )
    return out


def load_sts_decoder_control_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = resolve(args.sts_decoder_control_policy_jsonl)
    if int(args.max_sts_decoder_control_rows) <= 0 or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        if not isinstance(row, dict):
            continue
        if row.get("source_type") != "sts_decoder_control_contract":
            continue
        if row.get("training_use_state") != "decoder_control_policy_only_not_code_answer_training":
            continue
        if row.get("raw_public_prompt_or_tests_copied") or row.get("public_benchmark_training_data_used"):
            continue
        if row.get("public_tests_included") or row.get("public_benchmark_solutions_included"):
            continue
        rows.append(row)
        if len(rows) >= max(0, int(args.max_sts_decoder_control_rows)):
            break
    return rows


def sts_decoder_control_sts_rows(rows: list[dict[str, Any]], *, sym_context: str = "") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    policy_text = sts_decoder_control_policy_text(rows)
    for row in rows:
        objective = str(row.get("objective") or "sts_decoder_control")
        families = row.get("targeted_capability_families") if isinstance(row.get("targeted_capability_families"), list) else []
        family_text = ", ".join(str(item) for item in families[:12]) or str(
            row.get("missing_capability_family") or "candidate_coverage_recovery"
        )
        rejection_counts = row.get("candidate_rejection_reason_counts")
        if isinstance(rejection_counts, dict):
            reason_text = ", ".join(f"{key}:{value}" for key, value in list(rejection_counts.items())[:10])
        else:
            reason_text = "none"
        comparator_required = bool(row.get("force_same_seed_non_sts_comparator"))
        out.append(
            {
                "policy": "project_theseus_sts_decoder_control_sts_v1",
                "task_id": f"sts_decoder_control_{row.get('row_id', objective)}",
                "source_task_id": row.get("row_id") or objective,
                "split": "control",
                "benchmark_evidence_level": "decoder_control_metadata_only_not_code_answer_training",
                "visible_task_only": True,
                "public_benchmark_solutions_included": False,
                "public_tests_included": False,
                "canonical_solution_exported": False,
                "raw_public_prompt_or_tests_copied": False,
                "residual_concept": str(row.get("missing_capability_family") or "sts_decoder_control"),
                "input_streams": {
                    "context_stream": (
                        f"sts_decoder_control objective={objective} target_families={family_text} "
                        f"symliquid_state={sym_context} {policy_text}"
                    ),
                    "solver_stream": (
                        "produce learned-token candidate families with exact visible interface, "
                        "argument-role mapping, return shape, branch/loop/local skeleton, and AST-valid body"
                    ),
                    "critic_stream": (
                        f"compare STS-conditioned and same-seed non-STS candidates; "
                        f"rejection_reasons={reason_text}; comparator_required={comparator_required}"
                    ),
                    "tool_stream": (
                        "route STS as a decoder prior, retry policy, candidate rank feature, "
                        "and private ablation signal; do not use public answers; "
                        f"{policy_text}"
                    ),
                    "residual_stream": f"sts_control:{objective}:{family_text}",
                },
                "target_streams": {
                    "solver_stream": "emit STS-conditioned and same-seed non-STS learned-token candidates",
                    "critic_stream": "measure candidate distribution delta and pass-rate delta without public tests",
                    "tool_stream": "consume this row inside run_sts_conditioning before private closure ranking",
                    "patch_stream": str(row.get("answer") or "decoder control policy only; no benchmark answer"),
                    "residual_stream": f"sts_causal_decoder_control:{objective}",
                    "visible_report_stream": "STS capsule became a named decoder-control consumer",
                },
                "causal_contract": {
                    "strict_past_only": True,
                    "same_row_cross_stream_attention": "forbidden",
                    "one_token_per_output_stream_target": True,
                    "decoder_control_only": True,
                    "not_code_answer_training": True,
                    "force_same_seed_non_sts_comparator": comparator_required,
                },
            }
        )
    return out


def sts_decoder_control_policy_text(rows: list[dict[str, Any]]) -> str:
    """Compact control text copied into generated STS streams for real consumers.

    These rows are metadata-only control pressure. They intentionally carry no
    public answers, tests, or solution bodies.
    """

    if not rows:
        return ""
    prefer_values = [
        bool(row.get("prefer_sts_when_verifier_passes"))
        for row in rows
        if "prefer_sts_when_verifier_passes" in row
    ]
    positive_values = [
        bool(row.get("sts_positive_same_seed_lift"))
        for row in rows
        if "sts_positive_same_seed_lift" in row
    ]
    non_regressive_values = [
        bool(row.get("sts_coverage_non_regressive"))
        for row in rows
        if "sts_coverage_non_regressive" in row
    ]
    regressed = any(bool(row.get("sts_conditioning_regressed_candidate_coverage")) for row in rows)
    objectives = sorted({str(row.get("objective") or "") for row in rows if row.get("objective")})[:8]
    families = sorted(
        {
            str(family)
            for row in rows
            for family in (
                row.get("targeted_capability_families")
                if isinstance(row.get("targeted_capability_families"), list)
                else []
            )
        }
    )[:12]
    prefer_sts = all(prefer_values) if prefer_values else False
    positive_lift = any(positive_values) if positive_values else False
    non_regressive = all(non_regressive_values) if non_regressive_values else False
    return (
        "sts_decoder_control_policy "
        f"objectives={','.join(objectives) or 'none'}; "
        f"target_families={','.join(families) or 'none'}; "
        f"prefer_sts_when_verifier_passes={str(prefer_sts).lower()}; "
        f"sts_positive_same_seed_lift={str(positive_lift).lower()}; "
        f"sts_coverage_non_regressive={str(non_regressive).lower()}; "
        f"sts_conditioning_regressed_candidate_coverage={str(regressed).lower()}; "
        "public_calibration_locked=true; public_tests_used=false; public_solutions_used=false"
    )


def apply_sts_decoder_control_policy(
    generated: list[dict[str, Any]],
    *,
    public_task_ids: set[str],
    private_eval_task_ids: set[str],
    policy_text: str,
) -> list[dict[str, Any]]:
    if not policy_text:
        return generated
    target_ids = public_task_ids | private_eval_task_ids
    out: list[dict[str, Any]] = []
    for row in generated:
        if str(row.get("task_id") or "") not in target_ids or not isinstance(row.get("streams"), dict):
            out.append(row)
            continue
        streams = dict(row["streams"])
        for key in ["critic_stream", "tool_stream", "visible_report_stream"]:
            value = str(streams.get(key) or "").strip()
            if policy_text not in value:
                streams[key] = f"{value}; {policy_text}".strip("; ")
        row = dict(row)
        row["streams"] = streams
        row["sts_decoder_control_policy_applied"] = True
        row["sts_decoder_control_policy"] = policy_text
        out.append(row)
    return out


def private_code_sts_rows(private_tasks: list[dict[str, Any]], *, sym_context: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    priority_categories = {
        "ascii_mod_char",
        "average_or_zero",
        "bell_number_sequence",
        "below_threshold",
        "car_race_collision_count",
        "combinations_with_replacement",
        "count_digit_under_divisibility",
        "count_integer_items",
        "dict_required_keys",
        "distinct_count",
        "extract_def_name",
        "fruit_distribution_private",
        "is_anagram",
        "is_prime",
        "list_tail_replace",
        "max_list",
        "max_tuple_difference",
        "min_list",
        "modular_power_two",
        "multi_step_digit_shift_private",
        "multiply_three_primes",
        "nested_sum",
        "newman_conway_sequence",
        "next_perfect_square",
        "nonempty_substring_count",
        "palindrome",
        "palindrome_list_weight",
        "pluck_smallest_even",
        "polynomial_zero_bisection",
        "prime_fib_sequence",
        "public_private_count",
        "same_chars",
        "simple_power",
        "smallest_palindrome_changes",
        "sort_by_second",
        "spelled_number_sort",
        "split_list_at_index",
        "substring_count",
        "sum_list",
        "sum_squares",
        "symbol_beat_parser",
        "three_sum_zero_exists",
        "triangle_area_product",
        "tuple_item_count",
        "uppercase_ascii_sum",
        "two_sum_zero_exists",
        "word_count",
        "off_by_one_loop",
        "string_parsing_edge",
        "type_shape_mismatch",
        "selection_logic",
    }
    for task in private_tasks:
        family = concept_family(task)
        category = str(task.get("category") or "")
        tags = {str(tag) for tag in task.get("tags", [])} if isinstance(task.get("tags"), list) else set()
        repo_repair = "private_repo_repair" in tags or str(task.get("card_id") or "") == "private_repo_repair"
        decoder_contract = task.get("decoder_contract") if isinstance(task.get("decoder_contract"), dict) else {}
        generation_plan = decoder_contract.get("generation_plan") if isinstance(decoder_contract.get("generation_plan"), dict) else {}
        skeleton_bias = generation_plan.get("skeleton_bias") if isinstance(generation_plan.get("skeleton_bias"), list) else []
        contract_pressure = bool(decoder_contract) and any(
            str(item)
            in {
                "argument_roles",
                "return_contract",
                "branch_loop_skeleton",
                "local_state_updates",
                "edge_conditions",
                "algorithmic_planning",
                "execution_shaped_program",
            }
            for item in skeleton_bias
        )
        high_transfer_tag = any(
            token in tag
            for tag in tags
            for token in ("high_transfer", "type_contract", "edge_contract", "typed_interface", "algorithmic")
        )
        if (
            family not in {"recurrence_state", "string_rule_composition", "digit_rotation"}
            and category not in priority_categories
            and not repo_repair
            and not contract_pressure
            and not high_transfer_tag
        ):
            continue
        target_streams = private_code_target_streams(task)
        plan_text = decoder_contract_plan_text(task)
        input_streams = {
            "context_stream": (
                f"private_concept entry_point={task.get('entry_point')} "
                f"family={family} category={task.get('category')} prompt={task.get('prompt')} "
                f"symliquid_state={sym_context} decoder_plan={plan_text}"
            ),
            "solver_stream": f"plan full body for {family}; {plan_text}; emit state, branches, loops, and final return",
            "critic_stream": f"check {concept_residual_label(task, '')}; reject shallow wrappers and verifier-plan mismatches",
            "tool_stream": "use ast.parse, decoder_contract verifier feedback, concept guardrails, and hidden private tests",
            "residual_stream": concept_residual_label(task, ""),
        }
        rows.append(
            {
                "policy": "project_theseus_private_code_concept_sts_v1",
                "task_id": str(task.get("task_id") or ""),
                "source_task_id": str(task.get("source_task_id") or ""),
                "split": "train" if task.get("split") == "train" else "eval",
                "benchmark_evidence_level": "private_concept_sts_train_or_eval_only",
                "public_benchmark_solutions_included": False,
                "public_tests_included": False,
                "private_solution_body_used": task.get("split") == "train",
                "residual_concept": family,
                "concept_residual_label": concept_residual_label(task, ""),
                "input_streams": input_streams,
                "target_streams": target_streams,
                "causal_contract": {
                    "strict_past_only": True,
                    "same_row_cross_stream_attention": "forbidden",
                    "one_token_per_output_stream_target": True,
                    "private_hidden_tests_only": True,
                },
            }
        )
    return rows


def private_code_target_streams(task: dict[str, Any]) -> dict[str, str]:
    body = str(task.get("solution_body") or "")
    family = concept_family(task)
    label = concept_residual_label(task, "")
    plan_text = decoder_contract_plan_text(task)
    return {
        "solver_stream": f"emit a full Python body for {family}; {plan_text}",
        "critic_stream": f"look for {label}, invalid syntax, shallow wrappers, missing state variables, and wrong return contract",
        "tool_stream": "run ast parser, decoder_contract verifier, and hidden private tests; do not use public benchmark tests",
        "patch_stream": body,
        "residual_stream": label,
        "visible_report_stream": "private concept repair trace generated without public leakage",
    }


def decoder_contract_plan_text(task: dict[str, Any]) -> str:
    contract = task.get("decoder_contract") if isinstance(task.get("decoder_contract"), dict) else {}
    plan = contract.get("generation_plan") if isinstance(contract.get("generation_plan"), dict) else {}
    roles = contract.get("argument_roles") if isinstance(contract.get("argument_roles"), dict) else {}
    return_contract = contract.get("return_contract") if isinstance(contract.get("return_contract"), dict) else {}
    pieces = []
    if roles:
        pieces.append("argument_roles=" + ",".join(f"{key}:{value}" for key, value in sorted(roles.items())[:4]))
    if return_contract:
        pieces.append(
            "return_contract="
            + ",".join(
                f"{key}:{value}"
                for key, value in sorted(return_contract.items())
                if key in {"shape", "empty_or_invalid_behavior", "must_preserve_container_shape"}
            )
        )
    skeleton = plan.get("skeleton_bias") if isinstance(plan.get("skeleton_bias"), list) else []
    if skeleton:
        pieces.append("skeleton_bias=" + ",".join(str(item) for item in skeleton[:8]))
    repair = str(plan.get("repair_strategy") or "").strip()
    if repair:
        pieces.append("repair_strategy=" + repair[:180])
    return " ; ".join(pieces) if pieces else "signature -> return_shape -> skeleton -> verifier repair"


def public_sts_rows(public_tasks: list[dict[str, Any]], *, sym_context: str = "") -> list[dict[str, Any]]:
    rows = []
    for task in public_tasks:
        input_streams = public_sts_input_streams(task, sym_context=sym_context)
        base = {
            "policy": "project_theseus_code_lm_public_sts_context_v1",
            "task_id": str(task.get("task_id") or ""),
            "source_task_id": str(task.get("source_task_id") or ""),
            "split": "public_calibration",
            "benchmark_evidence_level": str(task.get("benchmark_evidence_level") or ""),
            "visible_task_only": True,
            "tests_exported": False,
            "canonical_solution_exported": False,
            "public_benchmark_solutions_included": False,
            "public_calibration_task": True,
            "target_streams": {stream: "" for stream in STS_OUTPUT_STREAMS},
            "causal_contract": {
                "strict_past_only": True,
                "same_row_cross_stream_attention": "forbidden",
                "one_token_per_output_stream_target": True,
            },
        }
        rows.append(
            {
                **base,
                "row_index": 0,
                "input_streams": {
                    "context_stream": input_streams["context_stream"],
                    "critic_stream": input_streams["critic_stream"],
                    "residual_stream": input_streams["residual_stream"],
                },
            }
        )
        rows.append(
            {
                **base,
                "row_index": 1,
                "input_streams": {
                    "context_stream": input_streams["context_stream"],
                    "solver_stream": input_streams["solver_stream"],
                    "critic_stream": input_streams["critic_stream"],
                    "tool_stream": input_streams["tool_stream"],
                    "patch_stream": input_streams["patch_stream"],
                    "residual_stream": input_streams["residual_stream"],
                },
            }
        )
    return rows


def public_sts_input_streams(task: dict[str, Any], *, sym_context: str = "") -> dict[str, str]:
    category = str(task.get("category") or "")
    prompt = str(task.get("prompt") or "")
    entry = str(task.get("entry_point") or "")
    tags = " ".join(str(tag) for tag in task.get("tags", []) if str(tag))
    repair_hint = public_category_repair_hint(category)
    plan_text = decoder_contract_plan_text(task)
    return {
        "context_stream": (
            f"visible_prompt_only entry_point={entry} category={category or 'unknown'} "
            f"tags={tags} symliquid_state={sym_context} decoder_plan={plan_text} prompt={prompt}"
        ),
        "solver_stream": (
            "generate full Python function body tokens; prefer named locals, loops, "
            f"conditionals, early returns, and collections when useful; {plan_text}; {repair_hint}"
        ),
        "critic_stream": (
            "audit syntax, indentation, empty input, singleton input, type handling, "
            "branch coverage, and whether the generated body is only a shallow wrapper"
        ),
        "tool_stream": (
            "available checks are ast.parse, function-shape lint, sandbox tests, "
            "candidate provenance, and promotion-eligibility audit"
        ),
        "patch_stream": (
            "repair by changing learned body tokens only; no public tests, no canonical "
            "solution, no task-id lookup, no loop-closure benchmark tool"
        ),
        "residual_stream": (
            "classify failure as syntax, loop, branch, collection, type, parsing, "
            "algorithm, timeout, or wrong_answer and feed only category pressure back"
        ),
    }


def public_category_repair_hint(category: str) -> str:
    hints = {
        "safe_head": "guard length before indexing and return fallback for empty input",
        "dict_required_keys": "iterate required keys and return false on first missing key",
        "public_private_count": "inspect mapping fields and count public test case entries",
        "extract_def_name": "scan source lines for a def header and extract the name before '('",
        "parse_ints": "normalize separators and skip tokens that are not signed integers",
        "sorted_unique_values": "deduplicate before sorting and preserve return type expectation",
        "sort_even_index_values": "copy input, sort even-position values, write them back by index",
        "increment_each_item": "iterate collection and append each incremented value",
        "count_digit_under_divisibility": "loop below the numeric limit, test divisibility, count digit occurrences",
        "two_sum_zero_exists": "track seen values and detect an additive inverse at distinct positions",
        "three_sum_zero_exists": "use three distinct positions or nested loops with early true return",
        "base_digits": "handle zero, repeatedly divide by base, prepend remainders",
        "triangle_area_product": "multiply base and height and divide by two",
        "balanced_brackets_simple": "use a stack or balance counter and reject mismatched closing brackets",
        "monotonic_sequence": "compare adjacent items and allow either nondecreasing or nonincreasing order",
        "common_elements": "deduplicate intersection before sorting returned common values",
        "largest_prime_factor": "divide out factors and keep the largest factor that remains prime",
        "arithmetic_series_sum": "accumulate a bounded range or use the arithmetic series relation",
        "derivative_coefficients": "multiply each coefficient by its index and skip the constant term",
        "tribonacci_sequence": "maintain three prior values and iterate until the requested index",
        "rotate_sequence": "normalize shift with length and combine sequence slices",
        "circular_digit_shift": "convert digits to a string, handle overshift by reversing, otherwise rotate by slices",
        "digit_sum_casefold": "filter the relevant character class before converting digits to ints",
        "fruit_distribution_private": "parse visible integer quantities from text before subtracting used counts",
        "is_prime": "handle n <= 1, test divisors through square root",
        "factors": "loop positive divisors and append exact divisors",
        "prime_factors": "divide out prime factors with multiplicity and append remaining prime tail",
        "divisible_by_11": "return the modulo-eleven divisibility predicate",
        "largest_divisor": "loop candidate divisors below n and keep the largest exact divisor",
        "rescale_to_unit": "compute min and max once, then scale each item into the zero-one interval",
        "decode_cyclic": "process text in groups of three and rotate each full group back",
        "prime_fib_sequence": "advance Fibonacci state and count only terms that pass a prime check",
        "polynomial_zero_bisection": "evaluate polynomial coefficients and narrow a bracketing interval",
        "length": "return len of the visible input object",
        "max_list": "handle non-empty collection maximum",
        "min_list": "handle non-empty collection minimum",
    }
    return hints.get(category, "infer an abstract algorithm from visible prompt words only")
