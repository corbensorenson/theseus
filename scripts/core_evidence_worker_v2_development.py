#!/usr/bin/env python3
"""Run Worker v2 on consumed development tasks only.

This is an engineering diagnostic, never blind evidence. It intentionally
does not inspect or score target commits; the separate evaluator owns that
after a candidate is sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "core_evidence_local_8b_worker.json"
E0 = ROOT / "reports" / "core_evidence_e0_preregistration.json"
OUT = ROOT / "reports" / "core_evidence_worker_v2_development.json"
EVENTS = ROOT / "runtime" / "core_evidence_worker_v2_development_events.jsonl"
MLX_PYTHON = Path("/Users/corbensorenson/miniforge3/bin/python")


class DevelopmentRunFault(RuntimeError):
    def __init__(self, reason: str, partial_receipt: dict[str, Any]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.partial_receipt = partial_receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--task-limit", type=int, default=1)
    parser.add_argument("--task-manifest", default="")
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--mlx-python", default=str(MLX_PYTHON))
    parser.add_argument("--events-out", default=str(EVENTS))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    report = run(
        args.task_index,
        args.task_limit,
        task_manifest=(
            Path(args.task_manifest) if args.task_manifest else None
        ),
        config_path=Path(args.config),
        mlx_python=Path(args.mlx_python),
        events_path=Path(args.events_out),
    )
    Path(args.out).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "trigger_state": report["trigger_state"],
        "attempted": len(report["tasks"]),
        "patch_count": sum(bool(row.get("patch_unified_diff")) for row in report["tasks"]),
        "verified_count": sum(row.get("candidate_verification_green") is True for row in report["tasks"]),
        "fault_count": len(report["faults"]),
    }, indent=2, sort_keys=True))
    return 0 if not report["faults"] else 2


def run(
    task_index: int,
    task_limit: int,
    *,
    task_manifest: Path | None = None,
    config_path: Path = CONFIG,
    mlx_python: Path = MLX_PYTHON,
    events_path: Path = EVENTS,
) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = config_path.resolve()
    # Preserve the venv entry-point path. Resolving its symlink to the base
    # interpreter discards pyvenv.cfg discovery and loses installed MLX-LM.
    mlx_python = (
        mlx_python
        if mlx_python.is_absolute()
        else ROOT / mlx_python
    )
    events_path = events_path.resolve()
    if task_manifest is not None:
        task_manifest = task_manifest.resolve()
    if events_path.exists():
        events_path.unlink()
    e0 = read_json(E0)
    config = read_json(config_path)
    if task_manifest is None:
        public_tasks = [
            row
            for row in dicts(
                dict_value(e0.get("public_packet")).get("tasks")
            )
            if row.get("partition") == "development"
            and row.get("denominator") == "D1_DEVELOPMENT"
        ]
    else:
        public_tasks = dicts(read_json(task_manifest).get("tasks"))
    selected = public_tasks[task_index:task_index + task_limit]
    rows = []
    faults = []
    for task in selected:
        try:
            rows.append(
                run_task(
                    task,
                    config,
                    config_path=config_path,
                    events_path=events_path,
                    mlx_python=mlx_python,
                )
            )
        except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            faults.append({
                "opaque_task_id": task.get("opaque_task_id"),
                "fault": f"{type(exc).__name__}:{exc}",
            })
    report = {
        "policy": "project_theseus_worker_v2_consumed_development_diagnostic_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "scope": "consumed_development_only_not_blind_evidence",
        "source": {
            "commit": git("rev-parse", "HEAD"),
            "worker_sha256": sha256_file(ROOT / "scripts" / "core_evidence_worker_v2.py"),
            "config_sha256": sha256_file(config_path),
            "E0_preregistration_sha256": e0.get("preregistration_sha256"),
            "task_manifest_sha256": (
                sha256_file(task_manifest)
                if task_manifest is not None
                else None
            ),
            "runtime_python_sha256": sha256_file(mlx_python),
        },
        "tasks": rows,
        "faults": faults,
        "counters": {
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "public_calibration_cases_consumed": 0,
            "D2_cases_consumed": 0,
            "learned_generation_credit": sum(
                int(row.get("learned_generation_credit") or 0) for row in rows
            ),
            "user_facing_effects": 0,
        },
        "runtime": {
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
        },
        "maximum_inference": (
            "This report can diagnose Worker v2 mechanics on consumed development "
            "tasks. It cannot support capability, generalization, or E2 claims."
        ),
    }
    report["report_payload_sha256"] = stable_hash({
        key: value for key, value in report.items()
        if key not in {"created_utc", "runtime", "report_payload_sha256"}
    })
    return report


def run_task(
    task: dict[str, Any],
    config: dict[str, Any],
    *,
    config_path: Path = CONFIG,
    events_path: Path = EVENTS,
    mlx_python: Path = MLX_PYTHON,
    visible_transform: (
        Callable[[dict[str, Any], Path], tuple[dict[str, Any], dict[str, Any]]]
        | None
    ) = None,
) -> dict[str, Any]:
    visible = {
        key: task[key] for key in (
            "natural_request", "parent_source_commit", "allowed_runtime_context",
            "authority_grant",
        )
    }
    with tempfile.TemporaryDirectory(prefix="theseus-worker-v2-dev-") as tmp:
        root = Path(tmp)
        snapshot = root / "snapshot"
        snapshot.mkdir()
        archive = root / "parent.tar"
        worker_input = root / "input.json"
        worker_output = root / "output.json"
        events_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "archive", "--format=tar", f"--output={archive}", str(visible["parent_source_commit"])],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        with tarfile.open(archive) as bundle:
            safe_extract(bundle, snapshot)
        if (snapshot / ".git").exists():
            raise ValueError("git metadata entered worker snapshot")
        input_adapter_receipt: dict[str, Any] = {}
        if visible_transform is not None:
            visible, input_adapter_receipt = visible_transform(visible, snapshot)
            if set(visible) != {
                "natural_request",
                "parent_source_commit",
                "allowed_runtime_context",
                "authority_grant",
            }:
                raise ValueError("visible transform changed worker field boundary")
        worker_input.write_text(
            json.dumps(visible, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        event_offset = events_path.stat().st_size if events_path.exists() else 0
        started_utc = now()
        started = time.perf_counter()
        command = [
            str(mlx_python),
            str(ROOT / "scripts" / "core_evidence_worker_v2.py"),
            "--input", str(worker_input),
            "--snapshot-root", str(snapshot),
            "--out", str(worker_output),
            "--config", str(config_path),
            "--events-out", str(events_path),
        ]
        try:
            process = subprocess.run(
                command,
                cwd=snapshot,
                capture_output=True,
                text=True,
                timeout=1800,
                env={
                    **os.environ,
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "NO_PROXY": "*",
                    "no_proxy": "*",
                    "PYTHONHASHSEED": "0",
                },
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            finished_utc = now()
            worker_wall_ms = round(
                (time.perf_counter() - started) * 1000.0,
                3,
            )
            raise DevelopmentRunFault(
                "worker_process_timeout",
                {
                    "terminal_reason": "worker_process_timeout",
                    "candidate_available": False,
                    "worker_wall_ms": worker_wall_ms,
                    "event_metrics": appended_event_metrics(
                        events_path, event_offset
                    ),
                    "model_identity": {
                        "repo_id": mapping(config.get("model")).get("repo_id"),
                        "revision": mapping(config.get("model")).get("revision"),
                        "runtime": "mlx_lm_local_metal",
                    },
                    "worker_config_sha256": sha256_file(config_path),
                    "started_utc": started_utc,
                    "finished_utc": finished_utc,
                    "timeout_seconds": 1800,
                    "returncode": None,
                    "stdout_tail_sha256": hashlib.sha256(
                        bytes(exc.stdout or b"")
                        if isinstance(exc.stdout, bytes)
                        else str(exc.stdout or "").encode()
                    ).hexdigest(),
                    "stderr_tail_sha256": hashlib.sha256(
                        bytes(exc.stderr or b"")
                        if isinstance(exc.stderr, bytes)
                        else str(exc.stderr or "").encode()
                    ).hexdigest(),
                },
            ) from exc
        finished_utc = now()
        worker_wall_ms = round((time.perf_counter() - started) * 1000.0, 3)
        if process.returncode != 0 or not worker_output.is_file():
            raise ValueError(
                f"worker_failed:returncode={process.returncode}:stderr={process.stderr[-1000:]}"
            )
        candidate = read_json(worker_output)
        event_metrics = appended_event_metrics(events_path, event_offset)
        receipts = dicts(candidate.get("verification_receipts"))
        seal = {
            "candidate_output_sha256": sha256_file(worker_output),
            "worker_input_sha256": sha256_file(worker_input),
            "parent_archive_sha256": sha256_file(archive),
            "worker_source_sha256": sha256_file(
                ROOT / "scripts" / "core_evidence_worker_v2.py"
            ),
            "config_sha256": sha256_file(config_path),
            "started_utc": started_utc,
            "finished_utc": finished_utc,
            "worker_wall_ms": worker_wall_ms,
            "target_opened_before_seal": False,
        }
        return {
            "opaque_task_id": task.get("opaque_task_id"),
            "candidate_output": candidate,
            "candidate_seal": seal,
            "candidate_output_sha256": seal["candidate_output_sha256"],
            "worker_input_sha256": seal["worker_input_sha256"],
            "parent_archive_sha256": seal["parent_archive_sha256"],
            "sealed_before_target_open": True,
            "patch_unified_diff": candidate.get("patch_unified_diff"),
            "proposed_paths": candidate.get("proposed_paths"),
            "candidate_verification_green": bool(receipts and receipts[-1].get("passed") is True),
            "verification_receipts": receipts,
            "effect_inventory": candidate.get("effect_inventory"),
            "action_summary": candidate.get("action_summary"),
            "repair_attempts": candidate.get("repair_attempts"),
            "format_repairs": candidate.get("format_repairs"),
            "terminal_reason": candidate.get("terminal_reason"),
            "residuals": candidate.get("residuals"),
            "learned_generation_credit": candidate.get("learned_generation_credit"),
            "local_model_inference_calls": candidate.get("local_model_inference_calls"),
            "model_identity": candidate.get("model_identity"),
            "event_metrics": event_metrics,
            "input_adapter_receipt": input_adapter_receipt,
            "external_inference_calls": candidate.get("external_inference_calls"),
            "teacher_calls": candidate.get("teacher_calls"),
            "public_calibration_cases_consumed": candidate.get("public_calibration_cases_consumed"),
            "D2_cases_consumed": candidate.get("D2_cases_consumed"),
            "worker_stdout_sha256": hashlib.sha256(process.stdout.encode()).hexdigest(),
            "worker_stderr_sha256": hashlib.sha256(process.stderr.encode()).hexdigest(),
        }


def appended_event_metrics(path: Path, offset: int) -> dict[str, Any]:
    if not path.is_file():
        return {
            "model_calls": 0,
            "generated_tokens": 0,
            "prompt_tokens": 0,
            "uncached_prompt_tokens": 0,
            "tool_calls": 0,
            "verification_count": 0,
            "generation_wall_ms": 0.0,
            "failed_actions": 0,
            "action_counts": {},
        }
    with path.open("rb") as handle:
        handle.seek(max(0, int(offset)))
        payload = handle.read().decode("utf-8")
    rows = []
    for line in payload.splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    action_counts: dict[str, int] = {}
    for row in rows:
        action = str(row.get("action") or "")
        action_counts[action] = action_counts.get(action, 0) + 1
    return {
        "model_calls": len(rows),
        "generated_tokens": sum(int(row.get("generated_tokens") or 0) for row in rows),
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in rows),
        "uncached_prompt_tokens": sum(
            int(row.get("uncached_prompt_tokens") or 0) for row in rows
        ),
        "tool_calls": sum(
            count
            for action, count in action_counts.items()
            if action not in {"", "plan", "abstain"}
        ),
        "verification_count": sum(
            int(row.get("verification_count") or 0) for row in rows
        ),
        "generation_wall_ms": round(
            sum(float(row.get("generation_wall_ms") or 0.0) for row in rows),
            3,
        ),
        "failed_actions": sum(row.get("ok") is not True for row in rows),
        "action_counts": dict(sorted(action_counts.items())),
    }


def safe_extract(bundle: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in bundle.getmembers():
        resolved = (destination / member.name).resolve()
        if not resolved.is_relative_to(root) or member.issym() or member.islnk():
            raise ValueError("unsafe archive member")
    bundle.extractall(destination, filter="data")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return value


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
