#!/usr/bin/env python3
"""Run request-bound stack variants on already-consumed development tasks.

This is an engineering causal smoke, not blind capability evidence. Every
model-visible variant is assembled before generation by the target-blind
adapter. Denied variants stop before local-model inference.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import core_evidence_repository_stack_adapter as adapter  # noqa: E402
import core_evidence_worker_v2_development as development  # noqa: E402


DEFAULT_ADAPTER_CONFIG = (
    ROOT / "configs" / "core_evidence_repository_stack_adapter.json"
)
DEFAULT_WORKER_CONFIG = (
    ROOT / "configs" / "core_evidence_tmax_9b_worker_control_v4_development.json"
)
DEFAULT_E0 = ROOT / "reports" / "core_evidence_e0_preregistration.json"
DEFAULT_TASK_MANIFEST = ROOT / "configs" / "core_evidence_repository_stack_development_public.json"
DEFAULT_OUT = ROOT / "reports" / "core_evidence_repository_stack_development.json"
DEFAULT_EVENTS = ROOT / "runtime" / "core_evidence_repository_stack_development_events.jsonl"
DEFAULT_MLX_PYTHON = ROOT / "runtime" / "venvs" / "mlx-0.32.0-py312" / "bin" / "python"
DEFAULT_VARIANTS = (
    "full_stack",
    "direct",
    "planning_none",
    "vcm_information_matched_untyped",
    "vcm_information_matched_shuffled",
    "vcm_stale",
    "vcm_omission",
    "conservative_hold",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--task-limit", type=int, default=1)
    parser.add_argument("--task-manifest", default="")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--adapter-config", default=str(DEFAULT_ADAPTER_CONFIG))
    parser.add_argument("--worker-config", default=str(DEFAULT_WORKER_CONFIG))
    parser.add_argument("--mlx-python", default=str(DEFAULT_MLX_PYTHON))
    parser.add_argument("--events-out", default=str(DEFAULT_EVENTS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    report = run(
        task_index=args.task_index,
        task_limit=args.task_limit,
        variant_ids=[
            item.strip() for item in args.variants.split(",") if item.strip()
        ],
        adapter_config_path=Path(args.adapter_config),
        worker_config_path=Path(args.worker_config),
        mlx_python=Path(args.mlx_python),
        events_path=Path(args.events_out),
        task_manifest_path=(
            Path(args.task_manifest) if args.task_manifest else None
        ),
    )
    Path(args.out).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "trigger_state": report["trigger_state"],
        "task_count": report["denominators"]["tasks"],
        "variant_attempt_count": report["denominators"]["variant_attempts"],
        "model_dispatch_count": report["denominators"]["model_dispatches"],
        "pre_generation_denial_count": report["denominators"]["pre_generation_denials"],
        "fault_count": len(report["faults"]),
    }, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def run(
    *,
    task_index: int,
    task_limit: int,
    variant_ids: list[str],
    adapter_config_path: Path = DEFAULT_ADAPTER_CONFIG,
    worker_config_path: Path = DEFAULT_WORKER_CONFIG,
    mlx_python: Path = DEFAULT_MLX_PYTHON,
    events_path: Path = DEFAULT_EVENTS,
    task_manifest_path: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    e0 = read_json(DEFAULT_E0)
    adapter_config = read_json(adapter_config_path)
    worker_config = read_json(worker_config_path)
    adapter_config_path = adapter_config_path.resolve()
    worker_config_path = worker_config_path.resolve()
    mlx_python = mlx_python if mlx_python.is_absolute() else (ROOT / mlx_python)
    events_path = events_path.resolve()
    if task_manifest_path is None:
        task_source = dicts(as_dict(e0.get("public_packet")).get("tasks"))
    else:
        task_manifest_path = task_manifest_path.resolve()
        task_manifest = read_json(task_manifest_path)
        if task_manifest.get("policy") != (
            "project_theseus_repository_stack_consumed_development_public_v1"
        ):
            raise ValueError("unexpected_consumed_development_manifest_policy")
        task_source = dicts(task_manifest.get("tasks"))
    tasks = [
        row for row in task_source
        if row.get("partition") == "development"
        and row.get("denominator") == "D1_DEVELOPMENT"
    ][task_index:task_index + task_limit]
    rows: list[dict[str, Any]] = []
    faults: list[dict[str, str]] = []
    for task in tasks:
        variants: list[dict[str, Any]] = []
        for variant_id in variant_ids:
            receipt_holder: dict[str, Any] = {}

            def transform(
                visible: dict[str, Any],
                snapshot: Path,
                *,
                selected_variant: str = variant_id,
            ) -> tuple[dict[str, Any], dict[str, Any]]:
                packet = adapter.adapt_visible_input(
                    visible=visible,
                    snapshot_root=snapshot,
                    variant_id=selected_variant,
                    config=adapter_config,
                )
                receipt_holder["packet"] = packet
                if not packet["dispatch_allowed"]:
                    raise PreGenerationDenial(packet)
                return packet["worker_input"], compact_adapter_receipt(packet)

            try:
                candidate = development.run_task(
                    copy.deepcopy(task),
                    worker_config,
                    config_path=worker_config_path,
                    events_path=events_path,
                    mlx_python=mlx_python,
                    visible_transform=transform,
                )
                variants.append({
                    "variant_id": variant_id,
                    "dispatch_allowed": True,
                    "pre_generation_denied": False,
                    "adapter_receipt": compact_adapter_receipt(
                        receipt_holder["packet"]
                    ),
                    "candidate": candidate,
                })
            except PreGenerationDenial as exc:
                variants.append({
                    "variant_id": variant_id,
                    "dispatch_allowed": False,
                    "pre_generation_denied": True,
                    "adapter_receipt": compact_adapter_receipt(exc.packet),
                    "candidate": None,
                })
            except (
                OSError,
                ValueError,
                RuntimeError,
                json.JSONDecodeError,
            ) as exc:
                faults.append({
                    "opaque_task_id": str(task.get("opaque_task_id") or ""),
                    "variant_id": variant_id,
                    "fault": f"{type(exc).__name__}:{exc}",
                })
        rows.append({
            "opaque_task_id": task.get("opaque_task_id"),
            "family": task.get("family"),
            "authority_grant": task.get("authority_grant"),
            "variant_results": variants,
        })
    flat = [
        variant for row in rows for variant in dicts(row.get("variant_results"))
    ]
    aliased_input_groups = worker_input_aliases(flat)
    if aliased_input_groups:
        faults.append({
            "opaque_task_id": "*",
            "variant_id": "*",
            "fault": (
                "INVALID_INTERVENTION_SHARED_WORKER_INPUT:"
                + json.dumps(aliased_input_groups, sort_keys=True)
            ),
        })
    report = {
        "policy": "project_theseus_repository_stack_consumed_development_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "scope": "consumed_development_causal_smoke_not_blind_evidence",
        "variant_ids": variant_ids,
        "source": {
            "commit": development.git("rev-parse", "HEAD"),
            "adapter_source_sha256": sha256_file(Path(adapter.__file__)),
            "adapter_config_sha256": sha256_file(adapter_config_path),
            "worker_source_sha256": sha256_file(
                ROOT / "scripts" / "core_evidence_worker_v2.py"
            ),
            "worker_config_sha256": sha256_file(worker_config_path),
            "E0_preregistration_sha256": e0.get("preregistration_sha256"),
            "task_manifest_sha256": (
                sha256_file(task_manifest_path)
                if task_manifest_path is not None
                else None
            ),
            "runtime_python_sha256": sha256_file(mlx_python),
        },
        "tasks": rows,
        "faults": faults,
        "denominators": {
            "tasks": len(rows),
            "variant_attempts": len(flat),
            "model_dispatches": sum(
                row.get("dispatch_allowed") is True for row in flat
            ),
            "pre_generation_denials": sum(
                row.get("pre_generation_denied") is True for row in flat
            ),
            "candidate_patches": sum(
                bool(as_dict(row.get("candidate")).get("patch_unified_diff"))
                for row in flat
            ),
            "candidate_verification_green": sum(
                as_dict(row.get("candidate")).get(
                    "candidate_verification_green"
                ) is True
                for row in flat
            ),
        },
        "counters": {
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "public_calibration_cases_consumed": 0,
            "D2_cases_consumed": 0,
            "E2_heldout_cases_consumed": 0,
            "user_facing_effects": 0,
        },
        "runtime": {
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
        },
        "maximum_inference": (
            "This engineering report can expose generic integration defects and "
            "prompt-level behavior differences on already-consumed development "
            "tasks. It cannot support fresh capability or subsystem-efficacy claims."
        ),
    }
    report["report_payload_sha256"] = stable_hash({
        key: value for key, value in report.items()
        if key not in {"created_utc", "runtime", "report_payload_sha256"}
    })
    return report


class PreGenerationDenial(Exception):
    def __init__(self, packet: dict[str, Any]) -> None:
        super().__init__("adapter denied mutating worker before generation")
        self.packet = packet


def compact_adapter_receipt(packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": packet.get("policy"),
        "variant_id": packet.get("variant_id"),
        "dispatch_allowed": packet.get("dispatch_allowed"),
        "typed_faults": packet.get("typed_faults"),
        "audit": packet.get("audit"),
        "authority_receipt": packet.get("authority_receipt"),
        "vcm_receipt": packet.get("vcm_receipt"),
        "compiled_plan": packet.get("compiled_plan"),
        "route_receipt": packet.get("route_receipt"),
        "procedural_reuse_receipt": packet.get("procedural_reuse_receipt"),
        "counters": packet.get("counters"),
    }


def worker_input_aliases(
    variants: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reject distinct model-visible variants that seal the same input bytes."""
    by_hash: dict[str, list[str]] = {}
    for row in variants:
        candidate = as_dict(row.get("candidate"))
        digest = str(candidate.get("worker_input_sha256") or "")
        if row.get("dispatch_allowed") is not True or not digest:
            continue
        by_hash.setdefault(digest, []).append(str(row.get("variant_id") or ""))
    return [
        {"worker_input_sha256": digest, "variant_ids": sorted(ids)}
        for digest, ids in sorted(by_hash.items())
        if len(set(ids)) > 1
    ]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return value


def dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
