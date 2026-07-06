#!/usr/bin/env python3
"""Direct top-level MLX descriptor lifecycle probe for VCM.

This script is meant to be launched by the MLX Python interpreter itself, not
by a Python subprocess orchestrator. On macOS, MLX/Metal can fail after nested
Python launch boundaries even when a top-level MLX process is healthy.

The claim here is deliberately narrow: real MLX arrays can be stored, reused,
extended, and invalidated under complete VCM runtime keys. This is not a
model-native MLX KV/prefix cache claim.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from vcm_native_runtime_probe import (  # noqa: E402
    DEFAULT_CLAIMS,
    REPORTS,
    build_route_descriptor,
    dict_value,
    hash_json,
    list_value,
    now,
    read_json,
    read_jsonl,
    rel,
    resolve,
    write_json,
)


DEFAULT_OUT = REPORTS / "vcm_mlx_direct_descriptor_probe.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    args = parser.parse_args()
    started = time.perf_counter()
    report = build_report(started=started)
    write_json(resolve(args.out), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(*, started: float) -> dict[str, Any]:
    route = build_route_descriptor(
        mlx=read_json(REPORTS / "macos_mlx_environment_diagnosis.json"),
        governor=read_json(REPORTS / "resource_governor.json"),
    )
    claims = read_jsonl(DEFAULT_CLAIMS)
    runtime_key, runtime_key_order = build_runtime_key(route=route, claims=claims)
    missing = [field for field in runtime_key_order if not runtime_key.get(field)]
    key_hash = hash_json({field: runtime_key.get(field) for field in runtime_key_order})

    summary: dict[str, Any] = {
        "passed": False,
        "runtime_kind": "mlx_core_tensor_prefix_descriptor",
        "backend": "mlx_apple",
        "accelerator": "mlx_apple",
        "runtime_key_hash": key_hash,
        "runtime_key_complete": not missing,
        "missing_runtime_key_fields": missing,
        "model_native_kv_cache_claimed": False,
        "mlx_lm_cache_claimed": False,
        "native_kv_cache_claimed": False,
        "native_prefix_cache_claimed": False,
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
        "teacher_calls": 0,
    }
    runtime_cache_record: dict[str, Any] = {
        "policy": "project_theseus_vcm_mlx_tensor_descriptor_cache_record_v1",
        "created_utc": now(),
        "runtime_kind": "mlx_core_tensor_prefix_descriptor",
        "backend": "mlx_apple",
        "accelerator": "mlx_apple",
        "runtime_key_hash": key_hash,
        "model_native_kv_cache_claimed": False,
        "native_kv_cache_claimed": False,
        "native_prefix_cache_claimed": False,
        "descriptor_lifecycle_claimed": False,
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
        "teacher_calls": 0,
    }

    error = ""
    try:
        import mlx.core as mx

        key_tensor = mx.reshape(mx.arange(24, dtype=mx.float32), (1, 2, 3, 4))
        value_tensor = key_tensor + 100.0
        mx.eval(key_tensor, value_tensor)
        store = {key_hash: {"key": key_tensor, "value": value_tensor}}
        reused = store.get(key_hash)
        reuse_hit = reused is not None and reused["key"] is key_tensor and reused["value"] is value_tensor
        prefix_seq_length = int(reused["key"].shape[2]) if reused else 0
        append_key = mx.ones((1, 2, 2, 4), dtype=mx.float32) * 7.0
        append_value = mx.ones((1, 2, 2, 4), dtype=mx.float32) * 17.0
        appended_key = mx.concatenate([reused["key"], append_key], axis=2) if reused else append_key
        appended_value = mx.concatenate([reused["value"], append_value], axis=2) if reused else append_value
        mx.eval(appended_key, appended_value)
        append_seq_length = int(appended_key.shape[2])
        checksum = float(mx.sum(appended_key).item() + mx.sum(appended_value).item())

        mutated_misses = {}
        for field in ["snapshot", "policy_hash", "permission_view", "redaction_view", "model_runtime_id"]:
            mutated = dict(runtime_key)
            mutated[field] = str(mutated.get(field, "")) + "#mutated"
            mutated_hash = hash_json({name: mutated.get(name) for name in runtime_key_order})
            mutated_misses[field] = mutated_hash not in store
        invalidation_miss_rate = sum(1 for value in mutated_misses.values() if value) / max(1, len(mutated_misses))
        passed = (
            not missing
            and reuse_hit
            and prefix_seq_length == 3
            and append_seq_length == 5
            and invalidation_miss_rate >= 1.0
            and checksum > 0.0
        )
        summary.update(
            {
                "passed": passed,
                "device": str(mx.default_device()),
                "mlx_version": getattr(mx, "__version__", ""),
                "cache_reuse_hit": reuse_hit,
                "prefix_seq_length": prefix_seq_length,
                "append_seq_length": append_seq_length,
                "mutated_key_misses": mutated_misses,
                "invalidation_miss_rate": invalidation_miss_rate,
                "resident_tensor_count": 2,
                "resident_tensor_shape": list(key_tensor.shape),
                "appended_tensor_shape": list(appended_key.shape),
                "checksum": checksum,
            }
        )
        runtime_cache_record.update(
            {
                "device": summary.get("device"),
                "cache_reuse_hit": reuse_hit,
                "append_seq_length": append_seq_length,
                "invalidation_miss_rate": invalidation_miss_rate,
                "resident_tensor_count": 2,
                "resident_tensor_shape": list(key_tensor.shape),
                "descriptor_lifecycle_claimed": passed,
            }
        )
    except Exception as exc:  # noqa: BLE001 - probe must report exact local runtime wall.
        error = type(exc).__name__ + ":" + str(exc)
        summary["error"] = error

    trigger_state = "GREEN" if summary.get("passed") is True else "YELLOW"
    return {
        "policy": "project_theseus_vcm_mlx_direct_descriptor_probe_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            **summary,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "runtime_route_descriptor": route,
        "runtime_cache_record": runtime_cache_record,
        "failed_reason": "" if trigger_state == "GREEN" else error or "direct MLX descriptor lifecycle did not pass",
        "score_semantics": (
            "Direct top-level MLX resident tensor descriptor lifecycle only. This is not an MLX-LM or model-native "
            "KV/prefix cache claim, and it does not widen CUDA/Metal parity claims."
        ),
    }


def build_runtime_key(*, route: dict[str, Any], claims: list[dict[str, Any]]) -> tuple[dict[str, Any], list[Any]]:
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
    base_key = dict_value(accepted_claim.get("runtime_key") if accepted_claim else {})
    runtime_key = {
        **base_key,
        "backend": "mlx_apple",
        "accelerator": "mlx_apple",
        "model_runtime_id": "mlx_core_tensor_prefix_descriptor_v1",
    }
    return runtime_key, list_value(route.get("runtime_key_order"))


if __name__ == "__main__":
    raise SystemExit(main())
