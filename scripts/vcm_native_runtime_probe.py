#!/usr/bin/env python3
"""Probe whether VCM can honestly claim native runtime cache integration.

This intentionally separates three states that were previously easy to blur:

1. semantic VCM cache-key/lifecycle proof;
2. hardware-aware route metadata for choosing a runtime owner;
3. real native prefix/KV cache reuse inside a local model runtime.

Only the third state may set native_*_cache_claimed=true. If the local runtime
does not expose a tested prefix/KV cache adapter, this report stays YELLOW and
names the exact blocker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RUNTIME = ROOT / "runtime" / "vcm_runtime_cache"
DEFAULT_OUT = REPORTS / "vcm_native_runtime_probe.json"
DEFAULT_MARKDOWN = REPORTS / "vcm_native_runtime_probe.md"
DEFAULT_DESCRIPTORS = RUNTIME / "native_runtime_route_descriptors.jsonl"
DEFAULT_CLAIMS = REPORTS / "vcm_runtime_materialization_claims.jsonl"
DEFAULT_MLX_DIRECT_DESCRIPTOR = REPORTS / "vcm_mlx_direct_descriptor_probe.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--descriptors-out", default=rel(DEFAULT_DESCRIPTORS))
    parser.add_argument("--python-timeout-seconds", type=int, default=10)
    args = parser.parse_args()

    started = time.perf_counter()
    report, descriptors = build_report(timeout_seconds=max(1, args.python_timeout_seconds), started=started)
    write_json(resolve(args.out), report)
    write_jsonl(resolve(args.descriptors_out), descriptors)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(*, timeout_seconds: int, started: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runtime_readiness = read_json(REPORTS / "vcm_runtime_claim_readiness.json")
    runtime_lifecycle = read_json(REPORTS / "vcm_runtime_cache_lifecycle.json")
    claims = read_jsonl(DEFAULT_CLAIMS)
    mlx = read_json(REPORTS / "macos_mlx_environment_diagnosis.json")
    governor = read_json(REPORTS / "resource_governor.json")

    readiness_summary = dict_value(runtime_readiness.get("summary"))
    lifecycle_summary = dict_value(runtime_lifecycle.get("summary"))
    semantic_ready = (
        runtime_readiness.get("trigger_state") == "GREEN"
        and float(readiness_summary.get("cache_key_complete_rate") or 0.0) >= 1.0
        and int(readiness_summary.get("accepted_semantic_claims") or 0) > 0
        and runtime_lifecycle.get("trigger_state") == "GREEN"
        and float(lifecycle_summary.get("reuse_hit_rate") or 0.0) >= 1.0
        and float(lifecycle_summary.get("snapshot_invalidation_miss_rate") or 0.0) >= 1.0
        and float(lifecycle_summary.get("policy_invalidation_miss_rate") or 0.0) >= 1.0
    )

    route = build_route_descriptor(mlx=mlx, governor=governor)
    python_probe = probe_python_runtime(route.get("python"), timeout_seconds=timeout_seconds)
    augment_python_probe_from_mlx_diagnosis(python_probe, mlx)
    native_python, native_python_probe = choose_native_lifecycle_python(
        route_python=str(route.get("python") or ""),
        route_python_probe=python_probe,
        timeout_seconds=timeout_seconds,
    )
    native_lifecycle = run_native_lifecycle_probe(
        route=route,
        claims=claims,
        python=native_python,
        timeout_seconds=timeout_seconds,
    )
    mlx_tensor_subprocess_lifecycle = run_mlx_tensor_descriptor_lifecycle_probe(
        route=route,
        claims=claims,
        python=str(route.get("python") or ""),
        timeout_seconds=timeout_seconds,
    )
    mlx_tensor_direct_lifecycle = read_mlx_direct_descriptor_lifecycle(route=route, claims=claims)
    mlx_tensor_lifecycle = choose_mlx_tensor_descriptor_lifecycle(
        subprocess_lifecycle=mlx_tensor_subprocess_lifecycle,
        direct_lifecycle=mlx_tensor_direct_lifecycle,
    )
    mlx_model_lifecycle = run_mlx_lm_model_cache_lifecycle_probe(
        route=route,
        claims=claims,
        python=str(route.get("python") or ""),
        timeout_seconds=timeout_seconds,
    )
    selected_native_lifecycle = (
        mlx_model_lifecycle
        if route.get("backend") == "mlx_apple" and mlx_model_lifecycle.get("passed") is True
        else native_lifecycle
    )
    native_adapter = inspect_native_adapter(native_lifecycle=selected_native_lifecycle)
    claim_scope = native_claim_scope(route=route, native_lifecycle=selected_native_lifecycle)
    descriptors = [
        row
        for row in [
            route,
            selected_native_lifecycle.get("runtime_cache_record"),
            mlx_tensor_lifecycle.get("runtime_cache_record"),
        ]
        if isinstance(row, dict) and row
    ]

    no_cheat = {
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
        "teacher_calls": 0,
    }
    native_lifecycle_passed = selected_native_lifecycle.get("passed") is True
    native_claimable = (
        semantic_ready
        and route.get("route_descriptor_complete") is True
        and native_adapter.get("native_cache_adapter_present") is True
        and native_lifecycle_passed
        and claim_scope.get("claim_scope_complete") is True
    )
    recommended_backend_native_runtime_claimable = bool(
        native_claimable and claim_scope.get("claim_backend_matches_recommended_execution_backend") is True
    )
    recommended_backend_tensor_descriptor_claimable = bool(
        semantic_ready
        and route.get("route_descriptor_complete") is True
        and route.get("backend") == "mlx_apple"
        and mlx_tensor_lifecycle.get("passed") is True
    )
    warnings = []
    blockers = []
    if not semantic_ready:
        blockers.append(
            {
                "kind": "semantic_runtime_lifecycle_not_green",
                "detail": "Run vcm_runtime_claim_readiness.py and vcm_runtime_cache_lifecycle.py until semantic key reuse/invalidation are green.",
            }
        )
    if not route.get("route_descriptor_complete"):
        blockers.append({"kind": "hardware_route_descriptor_incomplete", "detail": route})
    if python_probe.get("mlx_core_usable") is True and python_probe.get("mlx_lm_available") is not True:
        warnings.append(
            {
                "kind": "mlx_lm_missing_for_mlx_specific_runtime",
                "detail": "mlx.core is usable, but mlx-lm is not installed in the recommended MLX Python. The native runtime proof uses local Transformers/Torch instead.",
            }
        )
    if native_adapter.get("native_cache_adapter_present") is not True:
        blockers.append(
            {
                "kind": "no_vcm_native_kv_prefix_adapter",
                "detail": "No checked-in VCM adapter exposes a tested native prefix/KV cache lifecycle. Semantic descriptor caching remains the only proven runtime layer.",
            }
        )
    if native_claimable and not recommended_backend_native_runtime_claimable:
        warnings.append(
            {
                "kind": "recommended_backend_native_kv_not_claimed",
                "detail": (
                    "Native cache lifecycle passed for the scoped claim backend, but the hardware-aware "
                    "recommended execution backend is different. Scheduler-facing native KV routing must "
                    "remain disabled for the recommended backend until that exact backend has a lifecycle proof."
                ),
                "claim_backend": claim_scope.get("claim_backend"),
                "recommended_execution_backend": claim_scope.get("recommended_execution_backend"),
            }
        )
    if recommended_backend_tensor_descriptor_claimable and mlx_model_lifecycle.get("passed") is not True:
        warnings.append(
            {
                "kind": "mlx_descriptor_lifecycle_is_not_model_kv_cache",
                "detail": (
                    "MLX core resident tensor descriptor reuse/invalidation passed for complete VCM keys, "
                    "but the model-forward MLX-LM lifecycle is not proven, so model-native MLX KV/prefix cache routing remains disabled."
                ),
            }
        )
    if (
        mlx_tensor_lifecycle.get("launch_mode") == "direct_top_level_report"
        and mlx_tensor_subprocess_lifecycle.get("passed") is not True
    ):
        warnings.append(
            {
                "kind": "mlx_descriptor_requires_direct_launch_boundary",
                "detail": (
                    "The MLX descriptor lifecycle proof passed only as a direct top-level MLX Python process. "
                    "Python-orchestrated nested launch remains disabled for scheduler routing until a fork-safe "
                    "worker boundary is implemented."
                ),
                "subprocess_error": mlx_tensor_subprocess_lifecycle.get("error"),
            }
        )
    if not native_lifecycle_passed:
        blockers.append(
            {
                "kind": "native_prefix_kv_lifecycle_test_absent",
                "detail": native_lifecycle.get("error")
                or native_lifecycle.get("failed_reason")
                or "No local model runtime cache object was created, reused, invalidated, and verified under complete ordered runtime keys.",
            }
        )

    gates = [
        gate("semantic_runtime_lifecycle_green", semantic_ready, {"readiness": runtime_readiness.get("trigger_state"), "lifecycle": runtime_lifecycle.get("trigger_state")}),
        gate("hardware_route_descriptor_complete", route.get("route_descriptor_complete") is True, route),
        gate("python_probe_completed", python_probe.get("ok") is True, python_probe),
        gate("native_runtime_python_probe_completed", native_python_probe.get("ok") is True, native_python_probe),
        gate("native_adapter_present", native_adapter.get("native_cache_adapter_present") is True, native_adapter),
        gate("native_lifecycle_test_passed", native_lifecycle_passed, native_lifecycle),
        gate(
            "mlx_tensor_descriptor_lifecycle_test_passed",
            mlx_tensor_lifecycle.get("passed") is True,
            mlx_tensor_lifecycle,
        ),
        gate(
            "mlx_lm_model_cache_lifecycle_test_passed",
            mlx_model_lifecycle.get("passed") is True,
            mlx_model_lifecycle,
        ),
        gate("native_claim_scope_complete", claim_scope.get("claim_scope_complete") is True, claim_scope),
        gate(
            "native_claim_scope_does_not_widen_across_accelerators",
            claim_scope.get("cuda_native_kv_parity_claimed") is False
            and claim_scope.get("metal_native_kv_parity_claimed") is False
            and claim_scope.get("mlx_native_kv_parity_claimed") is False
            and (
                claim_scope.get("mlx_native_kv_lifecycle_claimed") is False
                or claim_scope.get("claim_backend") == "mlx_apple"
                and claim_scope.get("claim_backend_matches_recommended_execution_backend") is True
            ),
            claim_scope,
        ),
        gate("external_inference_zero", no_cheat["external_inference_calls"] == 0, no_cheat["external_inference_calls"]),
        gate("public_training_zero", no_cheat["public_training_rows_written"] == 0, no_cheat["public_training_rows_written"]),
        gate("fallback_return_zero", no_cheat["fallback_return_count"] == 0, no_cheat["fallback_return_count"]),
        gate("teacher_calls_zero", no_cheat["teacher_calls"] == 0, no_cheat["teacher_calls"]),
    ]

    trigger_state = "GREEN" if native_claimable else ("YELLOW" if semantic_ready and route.get("route_descriptor_complete") else "RED")
    viea_runtime_records = build_viea_runtime_records(
        route=route,
        native_lifecycle=selected_native_lifecycle,
        mlx_tensor_lifecycle=mlx_tensor_lifecycle,
        claim_scope=claim_scope,
        semantic_ready=semantic_ready,
        native_claimable=native_claimable,
        recommended_backend_native_runtime_claimable=recommended_backend_native_runtime_claimable,
        recommended_backend_tensor_descriptor_claimable=recommended_backend_tensor_descriptor_claimable,
        blockers=blockers,
        warnings=warnings,
        no_cheat=no_cheat,
    )
    report = {
        "policy": "project_theseus_vcm_native_runtime_probe_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "semantic_runtime_lifecycle_green": semantic_ready,
            "hardware_aware_runtime_route_metadata_ready": route.get("route_descriptor_complete") is True,
            "runtime_route_descriptor_count": len(descriptors),
            "runtime_profile_claimed": native_claimable,
            "native_kv_cache_claimed": native_claimable,
            "native_prefix_cache_claimed": native_claimable,
            "native_prefix_kv_lifecycle_test_passed": native_lifecycle_passed,
            "native_runtime_claimable": native_claimable,
            "native_runtime_kind": selected_native_lifecycle.get("runtime_kind"),
            "native_runtime_cache_backend": selected_native_lifecycle.get("backend"),
            "native_runtime_key_complete": selected_native_lifecycle.get("runtime_key_complete"),
            "native_runtime_cache_reuse_hit": selected_native_lifecycle.get("cache_reuse_hit"),
            "native_runtime_append_seq_length": selected_native_lifecycle.get("append_seq_length"),
            "native_runtime_invalidation_miss_rate": selected_native_lifecycle.get("invalidation_miss_rate"),
            "mlx_lm_model_cache_lifecycle_test_passed": mlx_model_lifecycle.get("passed") is True,
            "mlx_lm_model_cache_class": mlx_model_lifecycle.get("cache_class"),
            "mlx_lm_model_forward_cache_created": mlx_model_lifecycle.get("native_model_forward_cache_created"),
            "mlx_tensor_descriptor_lifecycle_test_passed": mlx_tensor_lifecycle.get("passed") is True,
            "mlx_tensor_descriptor_launch_mode": mlx_tensor_lifecycle.get("launch_mode"),
            "mlx_tensor_descriptor_runtime_kind": mlx_tensor_lifecycle.get("runtime_kind"),
            "mlx_tensor_descriptor_backend": mlx_tensor_lifecycle.get("backend"),
            "mlx_tensor_descriptor_device": mlx_tensor_lifecycle.get("device"),
            "mlx_tensor_descriptor_key_complete": mlx_tensor_lifecycle.get("runtime_key_complete"),
            "mlx_tensor_descriptor_cache_reuse_hit": mlx_tensor_lifecycle.get("cache_reuse_hit"),
            "mlx_tensor_descriptor_append_seq_length": mlx_tensor_lifecycle.get("append_seq_length"),
            "mlx_tensor_descriptor_invalidation_miss_rate": mlx_tensor_lifecycle.get("invalidation_miss_rate"),
            "recommended_backend_runtime_descriptor_lifecycle_claimable": recommended_backend_tensor_descriptor_claimable,
            "scheduler_vcm_descriptor_route_allowed_for_recommended_backend": recommended_backend_tensor_descriptor_claimable,
            "recommended_backend": route.get("backend"),
            "recommended_execution_backend": claim_scope.get("recommended_execution_backend"),
            "recommended_python": route.get("python"),
            "native_runtime_claim_scope": claim_scope.get("claim_scope"),
            "native_runtime_claim_backend": claim_scope.get("claim_backend"),
            "native_runtime_claim_device": claim_scope.get("claim_device"),
            "native_runtime_claim_backend_matches_recommended_execution_backend": claim_scope.get("claim_backend_matches_recommended_execution_backend"),
            "recommended_backend_native_runtime_claimable": recommended_backend_native_runtime_claimable,
            "scheduler_native_kv_route_allowed_for_recommended_backend": recommended_backend_native_runtime_claimable,
            "scheduler_native_kv_route_fail_closed": not recommended_backend_native_runtime_claimable,
            "accelerator_kv_lifecycle_claimed": claim_scope.get("accelerator_kv_lifecycle_claimed"),
            "mlx_native_kv_lifecycle_claimed": claim_scope.get("mlx_native_kv_lifecycle_claimed"),
            "accelerator_kv_parity_claimed": claim_scope.get("accelerator_kv_parity_claimed"),
            "mlx_native_kv_parity_claimed": claim_scope.get("mlx_native_kv_parity_claimed"),
            "cuda_native_kv_parity_claimed": claim_scope.get("cuda_native_kv_parity_claimed"),
            "metal_native_kv_parity_claimed": claim_scope.get("metal_native_kv_parity_claimed"),
            "native_runtime_python": str(route.get("python") or "") if selected_native_lifecycle is mlx_model_lifecycle else native_python,
            "mlx_core_usable": python_probe.get("mlx_core_usable"),
            "mlx_lm_available": python_probe.get("mlx_lm_available"),
            "route_python_transformers_available": python_probe.get("transformers_available"),
            "route_python_torch_available": python_probe.get("torch_available"),
            "transformers_available": native_python_probe.get("transformers_available"),
            "torch_available": native_python_probe.get("torch_available"),
            "native_cache_adapter_present": native_adapter.get("native_cache_adapter_present"),
            "blocker_count": len(blockers),
            "viea_runtime_record_count": len(viea_runtime_records),
            **no_cheat,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "runtime_route_descriptor": route,
        "native_runtime_claim_scope": claim_scope,
        "python_runtime_probe": python_probe,
        "native_runtime_python_probe": native_python_probe,
        "native_adapter_probe": native_adapter,
        "native_lifecycle_probe": selected_native_lifecycle,
        "cpu_transformers_native_lifecycle_probe": native_lifecycle,
        "mlx_lm_model_cache_lifecycle_probe": mlx_model_lifecycle,
        "mlx_tensor_descriptor_lifecycle_probe": mlx_tensor_lifecycle,
        "mlx_tensor_descriptor_subprocess_probe": mlx_tensor_subprocess_lifecycle,
        "mlx_tensor_descriptor_direct_probe": mlx_tensor_direct_lifecycle,
        "viea_runtime_records": viea_runtime_records,
        "gates": gates,
        "blockers": blockers,
        "warnings": warnings,
        "score_semantics": (
            "Native VCM runtime probe only. GREEN means a real local native prefix/KV cache lifecycle was tested. "
            "The claim scope is backend-specific and does not imply MLX, CUDA, Metal, or cross-accelerator KV parity unless those exact backend lifecycle tests pass. "
            "YELLOW means semantic VCM caching and hardware route metadata are ready, but native model-runtime KV/prefix reuse is not claimed. "
            "This report does not call external inference, train, spend public calibration, or use fallback returns."
        ),
        "external_inference_calls": 0,
    }
    return report, descriptors


def build_route_descriptor(*, mlx: dict[str, Any], governor: dict[str, Any]) -> dict[str, Any]:
    mlx_summary = dict_value(mlx.get("summary"))
    route_decision = dict_value(mlx.get("route_decision"))
    resources = dict_value(governor.get("current_resources"))
    gpu = dict_value(resources.get("gpu"))
    disk = dict_value(resources.get("disk"))
    decision = dict_value(governor.get("decision"))
    backend = str(decision.get("execution_owner") or gpu.get("accelerator") or "cpu")
    python = str(route_decision.get("recommended_python") or mlx_summary.get("recommended_python") or "")
    descriptor = {
        "policy": "project_theseus_vcm_native_runtime_route_descriptor_v1",
        "created_utc": now(),
        "host_system": platform.system(),
        "host_machine": platform.machine(),
        "backend": backend,
        "accelerator": gpu.get("accelerator") or backend,
        "gpu_available": bool(gpu.get("available")),
        "mlx_usable": bool(gpu.get("mlx_usable") or mlx_summary.get("usable_mlx_runtime_count")),
        "metal_usable": bool(gpu.get("metal_usable")),
        "python": python,
        "resource_governor_policy": governor.get("policy"),
        "resource_governor_decision": decision,
        "disk_free_gib": disk.get("free_gib"),
        "runtime_key_order": [
            "source_address",
            "source_content_hash",
            "representation_level",
            "representation_object_hash",
            "certificate_id",
            "model_id",
            "tokenizer_id",
            "adapter_id",
            "policy_hash",
            "principal",
            "permission_view",
            "redaction_view",
            "snapshot",
            "role_layout_hash",
            "materialization_predicate_hash",
            "backend",
            "accelerator",
            "model_runtime_id",
        ],
        "native_kv_cache_claimed": False,
        "native_prefix_cache_claimed": False,
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
    }
    required = ["host_system", "host_machine", "backend", "accelerator"]
    descriptor["route_descriptor_complete"] = all(bool(descriptor.get(key)) for key in required)
    descriptor["descriptor_hash"] = hash_json({key: descriptor.get(key) for key in sorted(descriptor) if key != "descriptor_hash"})
    return descriptor


def probe_python_runtime(python: str, *, timeout_seconds: int) -> dict[str, Any]:
    if not python:
        return {"ok": False, "error": "no_recommended_python"}
    code = r'''
import importlib.metadata
import json
import platform
import sys

def package(name):
    try:
        return {"available": True, "version": importlib.metadata.version(name)}
    except Exception as exc:
        return {"available": False, "error": type(exc).__name__ + ":" + str(exc)}

payload = {
    "executable": sys.executable,
    "version": sys.version.split()[0],
    "machine": platform.machine(),
    "packages": {
        "mlx": package("mlx"),
        "mlx-lm": package("mlx-lm"),
        "transformers": package("transformers"),
        "torch": package("torch"),
    },
}
payload["probe_mode"] = "package_metadata_only_no_native_imports"
print(json.dumps(payload, sort_keys=True))
'''
    try:
        proc = subprocess.run(
            [python, "-c", code],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "python": python, "error": type(exc).__name__ + ":" + str(exc)}
    payload = parse_json_line(proc.stdout)
    if not payload:
        payload = {"ok": False}
    payload.update(
        {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-1000:],
            "mlx_lm_available": bool(dict_value(dict_value(payload.get("packages")).get("mlx-lm")).get("available")),
            "transformers_available": bool(dict_value(dict_value(payload.get("packages")).get("transformers")).get("available")),
            "torch_available": bool(dict_value(dict_value(payload.get("packages")).get("torch")).get("available")),
        }
    )
    return payload


def augment_python_probe_from_mlx_diagnosis(python_probe: dict[str, Any], mlx: dict[str, Any]) -> None:
    summary = dict_value(mlx.get("summary"))
    active_python = str(summary.get("active_python") or summary.get("recommended_python") or "")
    for probe in list_value(mlx.get("python_probes")):
        if not isinstance(probe, dict) or str(probe.get("python") or "") != active_python:
            continue
        python_probe["mlx_core_usable"] = bool(probe.get("mlx_core_usable"))
        python_probe["mlx_native_abort"] = bool(probe.get("native_abort"))
        python_probe["mlx_default_device"] = parse_probe_stdout_field(probe, "default_device")
        python_probe["mlx_probe_source"] = "reports/macos_mlx_environment_diagnosis.json"
        python_probe["mlx_metadata"] = dict_value(probe.get("metadata")).get("mlx")
        if "mlx_lm_available" not in python_probe:
            python_probe["mlx_lm_available"] = bool(dict_value(dict_value(probe.get("metadata")).get("mlx-lm")).get("version"))
        return
    python_probe["mlx_core_usable"] = bool(summary.get("usable_mlx_runtime_count"))
    python_probe["mlx_probe_source"] = "reports/macos_mlx_environment_diagnosis.json:summary"


def choose_native_lifecycle_python(
    *,
    route_python: str,
    route_python_probe: dict[str, Any],
    timeout_seconds: int,
) -> tuple[str, dict[str, Any]]:
    if route_python_probe.get("ok") and route_python_probe.get("torch_available") and route_python_probe.get("transformers_available"):
        return route_python, route_python_probe
    candidates: list[str] = []
    for raw in [
        sys.executable,
        "/Users/corbensorenson/miniforge3/bin/python3",
        str(ROOT / ".venv" / "bin" / "python"),
        route_python,
    ]:
        if not raw:
            continue
        path = Path(raw).expanduser()
        if not path.exists():
            continue
        value = str(path)
        if value not in candidates:
            candidates.append(value)
    fallback_probe = route_python_probe if route_python else {"ok": False, "error": "no_recommended_python"}
    for candidate in candidates:
        probe = probe_python_runtime(candidate, timeout_seconds=timeout_seconds)
        if probe.get("ok") and probe.get("torch_available") and probe.get("transformers_available"):
            probe["selection_reason"] = "torch_transformers_native_cache_runtime_available"
            return candidate, probe
        if candidate == route_python:
            fallback_probe = probe
    fallback_probe["selection_reason"] = "no_candidate_python_with_torch_and_transformers"
    return route_python, fallback_probe


def run_native_lifecycle_probe(
    *,
    route: dict[str, Any],
    claims: list[dict[str, Any]],
    python: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    if not python:
        return {"passed": False, "error": "no_recommended_python", "runtime_kind": "none"}
    accepted_claim = next(
        (
            claim
            for claim in claims
            if isinstance(claim, dict)
            and claim.get("outcome") == "ACCEPTED_SEMANTIC_DESCRIPTOR"
            and claim.get("key_complete") is True
            and isinstance(claim.get("runtime_key"), dict)
        ),
        None,
    )
    if not accepted_claim:
        return {"passed": False, "error": "no_accepted_semantic_claim_with_complete_runtime_key", "runtime_kind": "none"}
    base_key = dict(accepted_claim.get("runtime_key") or {})
    backend = "transformers_tiny_gpt2_dynamic_cache_torch"
    runtime_key = {
        **base_key,
        "backend": backend,
        "accelerator": "cpu",
        "model_runtime_id": "transformers_tiny_gpt2_dynamic_cache_torch_v1",
    }
    runtime_key_order = list_value(route.get("runtime_key_order"))
    missing = [field for field in runtime_key_order if not runtime_key.get(field)]
    payload = {
        "runtime_key": runtime_key,
        "runtime_key_order": runtime_key_order,
        "mutate_fields": ["snapshot", "policy_hash", "permission_view", "redaction_view", "model_runtime_id"],
    }
    code = r'''
import hashlib
import json
import sys

payload = json.loads(sys.stdin.read())

def hash_json(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()

try:
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    from transformers.cache_utils import DynamicCache

    runtime_key = payload["runtime_key"]
    runtime_key_order = payload["runtime_key_order"]
    missing = [field for field in runtime_key_order if not runtime_key.get(field)]
    key_hash = hash_json({field: runtime_key.get(field) for field in runtime_key_order})
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=32,
        n_positions=16,
        n_ctx=16,
        n_embd=16,
        n_layer=1,
        n_head=2,
        use_cache=True,
        bos_token_id=0,
        eos_token_id=1,
    )
    model = GPT2LMHeadModel(config)
    model.eval()

    def seq_length(cache):
        if hasattr(cache, "get_seq_length"):
            return int(cache.get_seq_length(0))
        first_layer = cache[0]
        key_tensor = first_layer[0]
        return int(key_tensor.shape[-2])

    def native_cache(cache):
        if hasattr(cache, "get_seq_length"):
            return cache
        if hasattr(DynamicCache, "from_legacy_cache"):
            return DynamicCache.from_legacy_cache(cache)
        return cache

    with torch.no_grad():
        prefix_ids = torch.tensor([[2, 3, 4]], dtype=torch.long)
        prefix_out = model(input_ids=prefix_ids, use_cache=True)
        cache = native_cache(prefix_out.past_key_values)
    prefix_seq_length = seq_length(cache)
    store = {key_hash: cache}
    reused_cache = store.get(key_hash)
    reuse_hit = reused_cache is cache
    before_append_seq_length = seq_length(reused_cache)
    with torch.no_grad():
        append_ids = torch.tensor([[5, 6]], dtype=torch.long)
        append_out = model(input_ids=append_ids, past_key_values=reused_cache, use_cache=True)
        appended_cache = native_cache(append_out.past_key_values)
    append_seq_length = seq_length(appended_cache)
    mutated_misses = {}
    for field in payload["mutate_fields"]:
        mutated = dict(runtime_key)
        mutated[field] = str(mutated.get(field, "")) + "#mutated"
        mutated_hash = hash_json({name: mutated.get(name) for name in runtime_key_order})
        mutated_misses[field] = mutated_hash not in store
    cache_class = type(appended_cache).__name__
    layer_count = len(appended_cache) if hasattr(appended_cache, "__len__") else 0
    model_param_count = sum(param.numel() for param in model.parameters())
    passed = (
        not missing
        and reuse_hit
        and prefix_seq_length == 3
        and before_append_seq_length == 3
        and append_seq_length == 5
        and all(mutated_misses.values())
    )
    print(json.dumps({
        "ok": True,
        "passed": passed,
        "runtime_kind": "transformers_tiny_gpt2_dynamic_cache_torch",
        "cache_class": cache_class,
        "torch_version": getattr(torch, "__version__", ""),
        "transformers_model_class": type(model).__name__,
        "model_param_count": model_param_count,
        "device": "cpu",
        "runtime_key_hash": key_hash,
        "runtime_key_complete": not missing,
        "missing_runtime_key_fields": missing,
        "cache_reuse_hit": reuse_hit,
        "prefix_seq_length": prefix_seq_length,
        "before_append_seq_length": before_append_seq_length,
        "append_seq_length": append_seq_length,
        "mutated_key_misses": mutated_misses,
        "invalidation_miss_rate": sum(1 for value in mutated_misses.values() if value) / max(1, len(mutated_misses)),
        "cache_layer_count": layer_count,
        "native_kv_cache_object_created": True,
        "native_model_forward_cache_created": True,
        "native_prefix_cache_reused": reuse_hit and append_seq_length == 5,
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
        "teacher_calls": 0,
    }, sort_keys=True))
except Exception as exc:
    print(json.dumps({
        "ok": False,
        "passed": False,
        "runtime_kind": "transformers_tiny_gpt2_dynamic_cache_torch",
        "error": type(exc).__name__ + ":" + str(exc),
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
        "teacher_calls": 0,
    }, sort_keys=True))
'''
    try:
        proc = subprocess.run(
            [python, "-c", code],
            cwd=str(ROOT),
            input=json.dumps(payload, sort_keys=True),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"passed": False, "runtime_kind": backend, "error": type(exc).__name__ + ":" + str(exc)}
    result = parse_json_line(proc.stdout)
    if not result:
        result = {"ok": False, "passed": False, "runtime_kind": backend, "error": "no_json_probe_output"}
    result.update(
        {
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-1000:],
            "backend": backend,
            "accelerator": "cpu",
            "runtime_key_order": runtime_key_order,
            "runtime_key_complete": result.get("runtime_key_complete") is True and not missing,
            "missing_runtime_key_fields": list(dict.fromkeys([*missing, *list_value(result.get("missing_runtime_key_fields"))])),
            "runtime_cache_record": {
                "policy": "project_theseus_vcm_native_runtime_cache_record_v1",
                "created_utc": now(),
                "runtime_kind": result.get("runtime_kind"),
                "backend": backend,
                "accelerator": "cpu",
                "runtime_key_hash": result.get("runtime_key_hash") or hash_json({field: runtime_key.get(field) for field in runtime_key_order}),
                "cache_class": result.get("cache_class"),
                "transformers_model_class": result.get("transformers_model_class"),
                "model_param_count": result.get("model_param_count"),
                "cache_reuse_hit": result.get("cache_reuse_hit"),
                "append_seq_length": result.get("append_seq_length"),
                "invalidation_miss_rate": result.get("invalidation_miss_rate"),
                "native_kv_cache_claimed": result.get("passed") is True,
                "native_prefix_cache_claimed": result.get("passed") is True,
                "external_inference_calls": 0,
                "public_training_rows_written": 0,
                "fallback_return_count": 0,
                "teacher_calls": 0,
            },
        }
    )
    if proc.returncode != 0 and not result.get("error"):
        result["error"] = f"native_runtime_subprocess_returncode_{proc.returncode}"
    if result.get("passed") is not True and not result.get("failed_reason"):
        result["failed_reason"] = "native lifecycle checks did not all pass"
    return result


def run_mlx_tensor_descriptor_lifecycle_probe(
    *,
    route: dict[str, Any],
    claims: list[dict[str, Any]],
    python: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    if not python:
        return {"passed": False, "error": "no_mlx_python", "runtime_kind": "mlx_core_tensor_prefix_descriptor"}
    accepted_claim = next(
        (
            claim
            for claim in claims
            if isinstance(claim, dict)
            and claim.get("outcome") == "ACCEPTED_SEMANTIC_DESCRIPTOR"
            and claim.get("key_complete") is True
            and isinstance(claim.get("runtime_key"), dict)
        ),
        None,
    )
    if not accepted_claim:
        return {"passed": False, "error": "no_accepted_semantic_claim_with_complete_runtime_key", "runtime_kind": "mlx_core_tensor_prefix_descriptor"}
    base_key = dict(accepted_claim.get("runtime_key") or {})
    backend = "mlx_apple"
    runtime_key = {
        **base_key,
        "backend": backend,
        "accelerator": "mlx_apple",
        "model_runtime_id": "mlx_core_tensor_prefix_descriptor_v1",
    }
    runtime_key_order = list_value(route.get("runtime_key_order"))
    missing = [field for field in runtime_key_order if not runtime_key.get(field)]
    payload = {
        "runtime_key": runtime_key,
        "runtime_key_order": runtime_key_order,
        "mutate_fields": ["snapshot", "policy_hash", "permission_view", "redaction_view", "model_runtime_id"],
    }
    code = r'''
import hashlib
import json
import sys

payload = json.loads(sys.stdin.read())

def hash_json(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()

try:
    import mlx.core as mx

    runtime_key = payload["runtime_key"]
    runtime_key_order = payload["runtime_key_order"]
    missing = [field for field in runtime_key_order if not runtime_key.get(field)]
    key_hash = hash_json({field: runtime_key.get(field) for field in runtime_key_order})
    # Descriptor-scoped resident tensors: real MLX arrays, not a model KV cache.
    key_tensor = mx.reshape(mx.arange(24, dtype=mx.float32), (1, 2, 3, 4))
    value_tensor = key_tensor + 100.0
    mx.eval(key_tensor, value_tensor)
    store = {key_hash: {"key": key_tensor, "value": value_tensor}}
    reused = store.get(key_hash)
    reuse_hit = reused is not None and reused["key"] is key_tensor and reused["value"] is value_tensor
    prefix_seq_length = int(reused["key"].shape[2])
    append_key = mx.ones((1, 2, 2, 4), dtype=mx.float32) * 7.0
    append_value = mx.ones((1, 2, 2, 4), dtype=mx.float32) * 17.0
    appended_key = mx.concatenate([reused["key"], append_key], axis=2)
    appended_value = mx.concatenate([reused["value"], append_value], axis=2)
    mx.eval(appended_key, appended_value)
    append_seq_length = int(appended_key.shape[2])
    checksum = float(mx.sum(appended_key).item() + mx.sum(appended_value).item())
    mutated_misses = {}
    for field in payload["mutate_fields"]:
        mutated = dict(runtime_key)
        mutated[field] = str(mutated.get(field, "")) + "#mutated"
        mutated_hash = hash_json({name: mutated.get(name) for name in runtime_key_order})
        mutated_misses[field] = mutated_hash not in store
    passed = (
        not missing
        and reuse_hit
        and prefix_seq_length == 3
        and append_seq_length == 5
        and all(mutated_misses.values())
        and checksum > 0.0
    )
    print(json.dumps({
        "ok": True,
        "passed": passed,
        "runtime_kind": "mlx_core_tensor_prefix_descriptor",
        "backend": "mlx_apple",
        "accelerator": "mlx_apple",
        "device": str(mx.default_device()),
        "mlx_version": getattr(mx, "__version__", ""),
        "runtime_key_hash": key_hash,
        "runtime_key_complete": not missing,
        "missing_runtime_key_fields": missing,
        "cache_reuse_hit": reuse_hit,
        "prefix_seq_length": prefix_seq_length,
        "append_seq_length": append_seq_length,
        "mutated_key_misses": mutated_misses,
        "invalidation_miss_rate": sum(1 for value in mutated_misses.values() if value) / max(1, len(mutated_misses)),
        "resident_tensor_count": 2,
        "resident_tensor_shape": list(key_tensor.shape),
        "appended_tensor_shape": list(appended_key.shape),
        "checksum": checksum,
        "model_native_kv_cache_claimed": False,
        "mlx_lm_cache_claimed": False,
        "native_kv_cache_claimed": False,
        "native_prefix_cache_claimed": False,
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
        "teacher_calls": 0,
    }, sort_keys=True))
except Exception as exc:
    print(json.dumps({
        "ok": False,
        "passed": False,
        "runtime_kind": "mlx_core_tensor_prefix_descriptor",
        "error": type(exc).__name__ + ":" + str(exc),
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
        "teacher_calls": 0,
    }, sort_keys=True))
'''
    try:
        proc = subprocess.run(
            [python, "-c", code],
            cwd=str(ROOT),
            input=json.dumps(payload, sort_keys=True),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"passed": False, "runtime_kind": "mlx_core_tensor_prefix_descriptor", "error": type(exc).__name__ + ":" + str(exc)}
    result = parse_json_line(proc.stdout)
    if not result:
        result = {"ok": False, "passed": False, "runtime_kind": "mlx_core_tensor_prefix_descriptor", "error": "no_json_probe_output"}
    result.update(
        {
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-1000:],
            "backend": backend,
            "accelerator": "mlx_apple",
            "runtime_key_order": runtime_key_order,
            "runtime_key_complete": result.get("runtime_key_complete") is True and not missing,
            "missing_runtime_key_fields": list(dict.fromkeys([*missing, *list_value(result.get("missing_runtime_key_fields"))])),
            "runtime_cache_record": {
                "policy": "project_theseus_vcm_mlx_tensor_descriptor_cache_record_v1",
                "created_utc": now(),
                "runtime_kind": result.get("runtime_kind"),
                "backend": backend,
                "accelerator": "mlx_apple",
                "device": result.get("device"),
                "runtime_key_hash": result.get("runtime_key_hash") or hash_json({field: runtime_key.get(field) for field in runtime_key_order}),
                "cache_reuse_hit": result.get("cache_reuse_hit"),
                "append_seq_length": result.get("append_seq_length"),
                "invalidation_miss_rate": result.get("invalidation_miss_rate"),
                "resident_tensor_count": result.get("resident_tensor_count"),
                "resident_tensor_shape": result.get("resident_tensor_shape"),
                "model_native_kv_cache_claimed": False,
                "native_kv_cache_claimed": False,
                "native_prefix_cache_claimed": False,
                "descriptor_lifecycle_claimed": result.get("passed") is True,
                "external_inference_calls": 0,
                "public_training_rows_written": 0,
                "fallback_return_count": 0,
                "teacher_calls": 0,
            },
        }
    )
    if proc.returncode != 0 and not result.get("error"):
        result["error"] = f"mlx_tensor_descriptor_subprocess_returncode_{proc.returncode}"
    if result.get("passed") is not True and not result.get("failed_reason"):
        result["failed_reason"] = "MLX tensor descriptor lifecycle checks did not all pass"
    return result


def run_mlx_lm_model_cache_lifecycle_probe(
    *,
    route: dict[str, Any],
    claims: list[dict[str, Any]],
    python: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Exercise a real MLX-LM model forward cache under a complete VCM key."""
    if not python:
        return {"passed": False, "error": "no_mlx_python", "runtime_kind": "mlx_lm_tiny_llama_kv_cache"}
    accepted_claim = next(
        (
            claim
            for claim in claims
            if isinstance(claim, dict)
            and claim.get("outcome") == "ACCEPTED_SEMANTIC_DESCRIPTOR"
            and claim.get("key_complete") is True
            and isinstance(claim.get("runtime_key"), dict)
        ),
        None,
    )
    if not accepted_claim:
        return {"passed": False, "error": "no_accepted_semantic_claim_with_complete_runtime_key", "runtime_kind": "mlx_lm_tiny_llama_kv_cache"}
    runtime_key = {
        **dict(accepted_claim.get("runtime_key") or {}),
        "backend": "mlx_apple",
        "accelerator": "mlx_apple",
        "model_runtime_id": "mlx_lm_tiny_llama_kv_cache_v1",
    }
    runtime_key_order = list_value(route.get("runtime_key_order"))
    missing = [field for field in runtime_key_order if not runtime_key.get(field)]
    payload = {
        "runtime_key": runtime_key,
        "runtime_key_order": runtime_key_order,
        "mutate_fields": ["snapshot", "policy_hash", "permission_view", "redaction_view", "model_runtime_id"],
    }
    code = r'''
import hashlib
import json
import sys

payload = json.loads(sys.stdin.read())

def hash_json(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()

try:
    import mlx.core as mx
    import mlx_lm
    from mlx.utils import tree_flatten
    from mlx_lm.models.cache import KVCache
    from mlx_lm.models.llama import Model, ModelArgs

    runtime_key = payload["runtime_key"]
    runtime_key_order = payload["runtime_key_order"]
    missing = [field for field in runtime_key_order if not runtime_key.get(field)]
    key_hash = hash_json({field: runtime_key.get(field) for field in runtime_key_order})
    args = ModelArgs(
        model_type="llama",
        hidden_size=16,
        num_hidden_layers=1,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=32,
    )
    model = Model(args)
    cache = model.make_cache()
    prefix_logits = model(mx.array([[1, 2, 3]]), cache=cache)
    mx.eval(prefix_logits)
    prefix_offset = int(cache[0].offset)
    store = {key_hash: cache}
    reused = store.get(key_hash)
    reuse_hit = reused is cache
    append_logits = model(mx.array([[4, 5]]), cache=reused)
    mx.eval(append_logits)
    append_offset = int(cache[0].offset)
    mutated_misses = {}
    for field in payload["mutate_fields"]:
        mutated = dict(runtime_key)
        mutated[field] = str(mutated.get(field, "")) + "#mutated"
        mutated_hash = hash_json({name: mutated.get(name) for name in runtime_key_order})
        mutated_misses[field] = mutated_hash not in store
    passed = (
        not missing
        and len(cache) == 1
        and isinstance(cache[0], KVCache)
        and prefix_offset == 3
        and append_offset == 5
        and reuse_hit
        and tuple(prefix_logits.shape) == (1, 3, 32)
        and tuple(append_logits.shape) == (1, 2, 32)
        and all(mutated_misses.values())
    )
    print(json.dumps({
        "ok": True,
        "passed": passed,
        "runtime_kind": "mlx_lm_tiny_llama_kv_cache",
        "backend": "mlx_apple",
        "accelerator": "mlx_apple",
        "device": str(mx.default_device()),
        "mlx_version": getattr(mx, "__version__", ""),
        "mlx_lm_version": getattr(mlx_lm, "__version__", ""),
        "runtime_key_hash": key_hash,
        "runtime_key_complete": not missing,
        "missing_runtime_key_fields": missing,
        "cache_class": type(cache[0]).__name__,
        "cache_layer_count": len(cache),
        "cache_reuse_hit": reuse_hit,
        "native_model_forward_cache_created": prefix_offset == 3,
        "native_prefix_cache_reused": reuse_hit,
        "prefix_seq_length": prefix_offset,
        "append_seq_length": append_offset,
        "prefix_logits_shape": list(prefix_logits.shape),
        "append_logits_shape": list(append_logits.shape),
        "mutated_key_misses": mutated_misses,
        "invalidation_miss_rate": sum(1 for value in mutated_misses.values() if value) / max(1, len(mutated_misses)),
        "model_param_count": int(sum(x.size for _, x in tree_flatten(model.parameters()))),
        "model_native_kv_cache_claimed": passed,
        "mlx_lm_cache_claimed": passed,
        "native_kv_cache_claimed": passed,
        "native_prefix_cache_claimed": passed,
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
        "teacher_calls": 0,
    }, sort_keys=True))
except BaseException as exc:
    print(json.dumps({
        "ok": False,
        "passed": False,
        "runtime_kind": "mlx_lm_tiny_llama_kv_cache",
        "error": type(exc).__name__ + ":" + str(exc),
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
        "teacher_calls": 0,
    }, sort_keys=True))
'''
    try:
        proc = subprocess.run(
            [python, "-c", code],
            cwd=str(ROOT),
            input=json.dumps(payload, sort_keys=True),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"passed": False, "runtime_kind": "mlx_lm_tiny_llama_kv_cache", "error": type(exc).__name__ + ":" + str(exc)}
    result = parse_json_line(proc.stdout)
    if not result:
        result = {"ok": False, "passed": False, "runtime_kind": "mlx_lm_tiny_llama_kv_cache", "error": "no_json_probe_output"}
    result.update(
        {
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-1200:],
            "backend": "mlx_apple",
            "accelerator": "mlx_apple",
            "runtime_key_order": runtime_key_order,
            "runtime_key_complete": result.get("runtime_key_complete") is True and not missing,
            "missing_runtime_key_fields": list(dict.fromkeys([*missing, *list_value(result.get("missing_runtime_key_fields"))])),
            "runtime_cache_record": {
                "policy": "project_theseus_vcm_mlx_lm_native_cache_record_v1",
                "created_utc": now(),
                "runtime_kind": result.get("runtime_kind"),
                "backend": "mlx_apple",
                "accelerator": "mlx_apple",
                "device": result.get("device"),
                "runtime_key_hash": result.get("runtime_key_hash") or hash_json({field: runtime_key.get(field) for field in runtime_key_order}),
                "cache_class": result.get("cache_class"),
                "cache_reuse_hit": result.get("cache_reuse_hit"),
                "native_model_forward_cache_created": result.get("native_model_forward_cache_created"),
                "native_prefix_cache_reused": result.get("native_prefix_cache_reused"),
                "append_seq_length": result.get("append_seq_length"),
                "invalidation_miss_rate": result.get("invalidation_miss_rate"),
                "native_kv_cache_claimed": result.get("passed") is True,
                "native_prefix_cache_claimed": result.get("passed") is True,
                "model_native_kv_cache_claimed": result.get("passed") is True,
                "external_inference_calls": 0,
                "public_training_rows_written": 0,
                "fallback_return_count": 0,
                "teacher_calls": 0,
            },
        }
    )
    if proc.returncode != 0 and not result.get("error"):
        result["error"] = f"mlx_lm_model_cache_subprocess_returncode_{proc.returncode}"
    if result.get("passed") is not True and not result.get("failed_reason"):
        result["failed_reason"] = "MLX-LM model-forward KV cache lifecycle checks did not all pass"
    return result


def read_mlx_direct_descriptor_lifecycle(*, route: dict[str, Any], claims: list[dict[str, Any]]) -> dict[str, Any]:
    report = read_json(DEFAULT_MLX_DIRECT_DESCRIPTOR)
    accepted_claim = next(
        (
            claim
            for claim in claims
            if isinstance(claim, dict)
            and claim.get("outcome") == "ACCEPTED_SEMANTIC_DESCRIPTOR"
            and claim.get("key_complete") is True
            and isinstance(claim.get("runtime_key"), dict)
        ),
        None,
    )
    if not accepted_claim:
        return {
            "passed": False,
            "launch_mode": "direct_top_level_report",
            "error": "no_accepted_semantic_claim_with_complete_runtime_key",
            "runtime_kind": "mlx_core_tensor_prefix_descriptor",
        }
    runtime_key = {
        **dict(accepted_claim.get("runtime_key") or {}),
        "backend": "mlx_apple",
        "accelerator": "mlx_apple",
        "model_runtime_id": "mlx_core_tensor_prefix_descriptor_v1",
    }
    runtime_key_order = list_value(route.get("runtime_key_order"))
    expected_hash = hash_json({field: runtime_key.get(field) for field in runtime_key_order})
    if not report:
        return {
            "passed": False,
            "launch_mode": "direct_top_level_report",
            "error": f"missing_direct_mlx_descriptor_report:{rel(DEFAULT_MLX_DIRECT_DESCRIPTOR)}",
            "expected_runtime_key_hash": expected_hash,
            "runtime_kind": "mlx_core_tensor_prefix_descriptor",
        }
    summary = dict_value(report.get("summary"))
    hash_matches = summary.get("runtime_key_hash") == expected_hash
    direct_clean = (
        report.get("policy") == "project_theseus_vcm_mlx_direct_descriptor_probe_v1"
        and report.get("trigger_state") == "GREEN"
        and summary.get("passed") is True
        and summary.get("runtime_key_complete") is True
        and hash_matches
        and summary.get("cache_reuse_hit") is True
        and int(summary.get("append_seq_length") or 0) == 5
        and float(summary.get("invalidation_miss_rate") or 0.0) >= 1.0
        and int(summary.get("external_inference_calls") or 0) == 0
        and int(summary.get("public_training_rows_written") or 0) == 0
        and int(summary.get("fallback_return_count") or 0) == 0
        and int(summary.get("teacher_calls") or 0) == 0
    )
    result = {
        "ok": bool(direct_clean),
        "passed": bool(direct_clean),
        "launch_mode": "direct_top_level_report",
        "runtime_kind": summary.get("runtime_kind") or "mlx_core_tensor_prefix_descriptor",
        "backend": summary.get("backend") or "mlx_apple",
        "accelerator": summary.get("accelerator") or "mlx_apple",
        "device": summary.get("device"),
        "runtime_key_hash": summary.get("runtime_key_hash"),
        "expected_runtime_key_hash": expected_hash,
        "runtime_key_hash_matches_current_key": hash_matches,
        "runtime_key_complete": summary.get("runtime_key_complete") is True,
        "missing_runtime_key_fields": list_value(summary.get("missing_runtime_key_fields")),
        "cache_reuse_hit": summary.get("cache_reuse_hit"),
        "prefix_seq_length": summary.get("prefix_seq_length"),
        "append_seq_length": summary.get("append_seq_length"),
        "invalidation_miss_rate": summary.get("invalidation_miss_rate"),
        "resident_tensor_count": summary.get("resident_tensor_count"),
        "resident_tensor_shape": summary.get("resident_tensor_shape"),
        "model_native_kv_cache_claimed": False,
        "native_kv_cache_claimed": False,
        "native_prefix_cache_claimed": False,
        "external_inference_calls": int(summary.get("external_inference_calls") or 0),
        "public_training_rows_written": int(summary.get("public_training_rows_written") or 0),
        "fallback_return_count": int(summary.get("fallback_return_count") or 0),
        "teacher_calls": int(summary.get("teacher_calls") or 0),
        "source_report": rel(DEFAULT_MLX_DIRECT_DESCRIPTOR),
        "runtime_cache_record": dict_value(report.get("runtime_cache_record")),
    }
    if not direct_clean:
        result["error"] = report.get("failed_reason") or summary.get("error") or "direct_mlx_descriptor_report_not_clean_or_not_current"
        result["failed_reason"] = "Direct MLX descriptor report did not match the current complete VCM key or lifecycle contract"
    return result


def choose_mlx_tensor_descriptor_lifecycle(
    *, subprocess_lifecycle: dict[str, Any], direct_lifecycle: dict[str, Any]
) -> dict[str, Any]:
    if direct_lifecycle.get("passed") is True:
        return direct_lifecycle
    if subprocess_lifecycle.get("passed") is True:
        result = dict(subprocess_lifecycle)
        result.setdefault("launch_mode", "python_subprocess")
        return result
    result = dict(subprocess_lifecycle)
    result.setdefault("launch_mode", "python_subprocess")
    result["direct_top_level_report_error"] = direct_lifecycle.get("error")
    result["direct_top_level_report_passed"] = direct_lifecycle.get("passed") is True
    return result


def parse_probe_stdout_field(probe: dict[str, Any], field: str) -> Any:
    stdout = str(get_path(probe, ["core_probe", "stdout_tail"], ""))
    payload = parse_json_line(stdout)
    return payload.get(field)


def inspect_native_adapter(*, native_lifecycle: dict[str, Any]) -> dict[str, Any]:
    roots = [ROOT / "scripts", ROOT / "crates"]
    matches: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.name == Path(__file__).name or not path.is_file() or path.suffix not in {".py", ".rs"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lowered = text.lower()
            if "vcm" not in lowered:
                continue
            if all(token in lowered for token in ["native", "kv", "cache", "lifecycle"]):
                matches.append({"path": rel(path), "kind": "candidate_text_match"})
    lifecycle_adapter_present = bool(
        native_lifecycle.get("passed") is True
        and native_lifecycle.get("native_model_forward_cache_created") is True
        and native_lifecycle.get("native_prefix_cache_reused") is True
        and (
            native_lifecycle.get("native_kv_cache_object_created") is True
            or native_lifecycle.get("cache_class") == "KVCache"
        )
    )
    return {
        "native_cache_adapter_present": lifecycle_adapter_present,
        "adapter_kind": native_lifecycle.get("runtime_kind") if lifecycle_adapter_present else None,
        "adapter_source": (
            "scripts/vcm_native_runtime_probe.py:run_mlx_lm_model_cache_lifecycle_probe"
            if lifecycle_adapter_present and native_lifecycle.get("backend") == "mlx_apple"
            else "scripts/vcm_native_runtime_probe.py:run_native_lifecycle_probe"
            if lifecycle_adapter_present
            else None
        ),
        "candidate_text_matches": matches[:20],
        "candidate_text_match_count": len(matches),
        "required_adapter_contract": [
            "construct native prefix/KV object from complete ordered VCM runtime key",
            "reuse object under identical key",
            "invalidate object on snapshot/policy/permission/redaction/model/tokenizer change",
            "prove no authority escalation and no fallback returns",
        ],
    }


def native_claim_scope(*, route: dict[str, Any], native_lifecycle: dict[str, Any]) -> dict[str, Any]:
    recommended_execution_backend = str(route.get("backend") or "")
    claim_backend = str(native_lifecycle.get("backend") or native_lifecycle.get("runtime_kind") or "")
    claim_device = str(native_lifecycle.get("device") or native_lifecycle.get("accelerator") or "")
    passed = native_lifecycle.get("passed") is True
    accelerator_backend = recommended_execution_backend in {"mlx_apple", "apple_metal", "cuda", "nvidia_cuda"}
    backend_matches_execution = bool(claim_backend) and claim_backend == recommended_execution_backend
    accelerator_kv_lifecycle_claimed = bool(passed and accelerator_backend and backend_matches_execution)
    mlx_native_kv_lifecycle_claimed = bool(passed and claim_backend == "mlx_apple" and backend_matches_execution)
    return {
        "policy": "project_theseus_vcm_native_runtime_claim_scope_v1",
        "claim_scope_complete": bool(passed and claim_backend and claim_device),
        "claim_scope": (
            "local_transformers_torch_dynamic_cache_cpu_only"
            if claim_backend == "transformers_tiny_gpt2_dynamic_cache_torch"
            else "backend_specific_native_runtime_cache_lifecycle"
        ),
        "recommended_execution_backend": recommended_execution_backend,
        "claim_backend": claim_backend,
        "claim_device": claim_device,
        "claim_backend_matches_recommended_execution_backend": backend_matches_execution,
        "native_kv_cache_claimed_for_backend": passed,
        "native_prefix_cache_claimed_for_backend": passed,
        "accelerator_kv_lifecycle_claimed": accelerator_kv_lifecycle_claimed,
        "mlx_native_kv_lifecycle_claimed": mlx_native_kv_lifecycle_claimed,
        "accelerator_kv_parity_claimed": False,
        "mlx_native_kv_parity_claimed": False,
        "cuda_native_kv_parity_claimed": False,
        "metal_native_kv_parity_claimed": False,
        "not_claimed": [
            "CUDA native KV/prefix cache reuse",
            "Metal native KV/prefix cache reuse",
            "cross-accelerator KV/prefix cache parity",
        ] + ([] if mlx_native_kv_lifecycle_claimed else ["MLX native KV/prefix cache reuse"]),
        "score_semantics": (
            "A native cache lifecycle proof is valid only for the exact claim_backend and claim_device. "
            "Hardware route metadata may recommend a different execution backend for training or inference, "
            "but that does not widen the native KV/prefix cache claim."
        ),
    }


def build_viea_runtime_records(
    *,
    route: dict[str, Any],
    native_lifecycle: dict[str, Any],
    mlx_tensor_lifecycle: dict[str, Any],
    claim_scope: dict[str, Any],
    semantic_ready: bool,
    native_claimable: bool,
    recommended_backend_native_runtime_claimable: bool,
    recommended_backend_tensor_descriptor_claimable: bool,
    blockers: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    no_cheat: dict[str, int],
) -> list[dict[str, Any]]:
    """Normalize runtime-cache readiness into the shared VIEA spine contract."""

    run_id = stable_runtime_record_id(
        "vcm_runtime_probe",
        route.get("descriptor_hash"),
        native_lifecycle.get("runtime_key_hash"),
        mlx_tensor_lifecycle.get("runtime_key_hash"),
        claim_scope.get("claim_backend"),
        claim_scope.get("recommended_execution_backend"),
    )
    runtime_ref = rel(DEFAULT_OUT)
    descriptor_ref = rel(DEFAULT_DESCRIPTORS)
    cpu_native_supported = bool(native_claimable)
    recommended_native_kv_route_allowed = bool(recommended_backend_native_runtime_claimable)
    recommended_descriptor_route_allowed = bool(recommended_backend_tensor_descriptor_claimable)
    fail_closed_reason = "none"
    if not recommended_native_kv_route_allowed:
        fail_closed_reason = (
            "recommended_backend_native_kv_lifecycle_not_proven"
            if cpu_native_supported
            else "native_prefix_kv_lifecycle_not_proven"
        )
    base = {
        "run_id": run_id,
        "source_surface": "vcm_native_runtime_probe",
        "artifact_ref": runtime_ref,
        "backend": route.get("backend"),
        "recommended_execution_backend": claim_scope.get("recommended_execution_backend"),
        "native_runtime_claim_backend": claim_scope.get("claim_backend"),
        "native_runtime_claim_device": claim_scope.get("claim_device"),
        "scheduler_native_kv_route_allowed": recommended_native_kv_route_allowed,
        "scheduler_vcm_descriptor_route_allowed": recommended_descriptor_route_allowed,
        "semantic_runtime_lifecycle_green": semantic_ready,
        "runtime_profile_claimed": cpu_native_supported,
        "native_kv_cache_claimed": cpu_native_supported,
        "native_prefix_cache_claimed": cpu_native_supported,
        "accelerator_kv_lifecycle_claimed": bool(claim_scope.get("accelerator_kv_lifecycle_claimed")),
        "mlx_native_kv_lifecycle_claimed": bool(claim_scope.get("mlx_native_kv_lifecycle_claimed")),
        "accelerator_kv_parity_claimed": bool(claim_scope.get("accelerator_kv_parity_claimed")),
        "mlx_native_kv_parity_claimed": bool(claim_scope.get("mlx_native_kv_parity_claimed")),
        "cuda_native_kv_parity_claimed": bool(claim_scope.get("cuda_native_kv_parity_claimed")),
        "metal_native_kv_parity_claimed": bool(claim_scope.get("metal_native_kv_parity_claimed")),
        "fail_closed_reason": fail_closed_reason,
        "public_training_rows_written": int(no_cheat.get("public_training_rows_written") or 0),
        "external_inference_calls": int(no_cheat.get("external_inference_calls") or 0),
        "fallback_return_count": int(no_cheat.get("fallback_return_count") or 0),
        "teacher_calls": int(no_cheat.get("teacher_calls") or 0),
    }

    def record(record_type: str, suffix: str, **fields: Any) -> dict[str, Any]:
        payload = {
            **base,
            "record_type": record_type,
            **fields,
        }
        payload["record_id"] = stable_runtime_record_id(run_id, record_type, suffix, payload)
        return payload

    return [
        record(
            "authority_use_receipt",
            "authority",
            authority_scope="local_runtime_probe_only",
            authority_mode="same_as_source_no_escalation",
            raw_private_text_stored=False,
            raw_prompt_stored=False,
        ),
        record(
            "context_transaction",
            "context",
            source_claims_ref=rel(DEFAULT_CLAIMS),
            semantic_materialization_claims_ready=semantic_ready,
            runtime_key_complete=bool(native_lifecycle.get("runtime_key_complete")),
            mlx_descriptor_key_complete=bool(mlx_tensor_lifecycle.get("runtime_key_complete")),
        ),
        record(
            "context_adequacy",
            "adequacy",
            state="governed_sufficient_for_runtime_cache_boundary"
            if semantic_ready and route.get("route_descriptor_complete")
            else "insufficient_for_runtime_cache_boundary",
            adequate_for_semantic_descriptor_route=recommended_descriptor_route_allowed,
            adequate_for_recommended_backend_native_kv_route=recommended_native_kv_route_allowed,
            blocked_reason=fail_closed_reason,
        ),
        record(
            "runtime_adapter_invocation",
            "runtime",
            adapter_id="vcm_native_runtime_probe_v1",
            runtime_kind=native_lifecycle.get("runtime_kind"),
            runtime_backend=native_lifecycle.get("backend"),
            runtime_cache_key_hash=native_lifecycle.get("runtime_key_hash"),
            runtime_cache_reuse_hit=bool(native_lifecycle.get("cache_reuse_hit")),
            native_prefix_cache_reused=bool(native_lifecycle.get("native_prefix_cache_reused")),
            mlx_descriptor_runtime_kind=mlx_tensor_lifecycle.get("runtime_kind"),
            mlx_descriptor_lifecycle_passed=bool(mlx_tensor_lifecycle.get("passed")),
        ),
        record(
            "resource_budget",
            "resource",
            backend_requirements=[route.get("backend"), claim_scope.get("claim_backend")],
            python=route.get("python"),
            native_runtime_python=route.get("native_runtime_python"),
            estimated_latency_ms=native_lifecycle.get("runtime_ms"),
            disk_free_gib=route.get("disk_free_gib"),
        ),
        record(
            "costed_route",
            "route",
            route_phase="runtime_cache_readiness",
            task_fit="descriptor_route_only" if not recommended_native_kv_route_allowed else "native_kv_route_allowed",
            scheduler_native_kv_route_fail_closed=not recommended_native_kv_route_allowed,
            scheduler_vcm_descriptor_route_allowed=recommended_descriptor_route_allowed,
            network_class="local",
        ),
        record(
            "generation_mode",
            "mode",
            mode="runtime_probe_not_generation",
            learned_generation_claim_allowed=False,
            candidate_generation_credit=0,
            structured_non_solved=True,
        ),
        record(
            "failure_boundary",
            "failure",
            failure_id=stable_runtime_record_id(run_id, "failure_boundary", fail_closed_reason),
            state="fail_closed" if not recommended_native_kv_route_allowed else "open_for_exact_backend",
            blocked_reason=fail_closed_reason,
            blocker_count=len(blockers),
            warning_count=len(warnings),
            blockers=blockers[:8],
            warnings=warnings[:8],
        ),
        record(
            "artifact_graph_record",
            "artifact",
            artifact_ref=runtime_ref,
            descriptor_ref=descriptor_ref,
            runtime_claims_ref=rel(DEFAULT_CLAIMS),
            runtime_cache_record_ref="runtime/vcm_runtime_cache/semantic_materialization_cache_index.jsonl",
        ),
        record(
            "claim_record",
            "claim",
            claim_id=stable_runtime_record_id(run_id, "claim", claim_scope.get("claim_scope")),
            support_state="SUPPORTED_EXACT_BACKEND_ONLY" if cpu_native_supported else "NOT_SUPPORTED",
            claim_scope=claim_scope.get("claim_scope"),
            verifier_state="recommended_backend_native_kv_route_allowed"
            if recommended_native_kv_route_allowed
            else "recommended_backend_native_kv_route_blocked",
        ),
        record(
            "evidence_transition_record",
            "evidence",
            previous_support_state="semantic_descriptor_cache_ready" if semantic_ready else "semantic_descriptor_cache_not_ready",
            current_support_state="backend_scoped_native_kv_proven_recommended_backend_blocked"
            if cpu_native_supported and not recommended_native_kv_route_allowed
            else "recommended_backend_native_kv_route_allowed"
            if recommended_native_kv_route_allowed
            else "native_kv_not_proven",
            evidence_ref=runtime_ref,
        ),
    ]


def stable_runtime_record_id(*parts: Any) -> str:
    return "vcm-runtime:" + hashlib.sha256(
        json.dumps(parts, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def parse_json_line(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# VCM Native Runtime Probe",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- semantic_runtime_lifecycle_green: `{summary.get('semantic_runtime_lifecycle_green')}`",
        f"- hardware route metadata ready: `{summary.get('hardware_aware_runtime_route_metadata_ready')}`",
        f"- runtime profile claimed: `{summary.get('runtime_profile_claimed')}`",
        f"- native KV cache claimed: `{summary.get('native_kv_cache_claimed')}`",
        f"- native prefix cache claimed: `{summary.get('native_prefix_cache_claimed')}`",
        f"- native prefix/KV lifecycle test passed: `{summary.get('native_prefix_kv_lifecycle_test_passed')}`",
        f"- claim backend / recommended backend: `{summary.get('native_runtime_claim_backend')}` / `{summary.get('recommended_execution_backend')}`",
        f"- scheduler native KV route allowed for recommended backend: `{summary.get('scheduler_native_kv_route_allowed_for_recommended_backend')}`",
        f"- MLX tensor descriptor lifecycle passed: `{summary.get('mlx_tensor_descriptor_lifecycle_test_passed')}`",
        f"- scheduler VCM descriptor route allowed for recommended backend: `{summary.get('scheduler_vcm_descriptor_route_allowed_for_recommended_backend')}`",
        f"- recommended backend: `{summary.get('recommended_backend')}`",
        f"- recommended Python: `{summary.get('recommended_python')}`",
        f"- MLX core usable: `{summary.get('mlx_core_usable')}`",
        f"- MLX-LM available: `{summary.get('mlx_lm_available')}`",
        "",
        "## Blockers",
    ]
    if not report.get("blockers"):
        lines.append("- none")
    else:
        for blocker in report.get("blockers", []):
            lines.append(f"- `{blocker.get('kind')}`: {blocker.get('detail')}")
    return "\n".join(lines) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def hash_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
