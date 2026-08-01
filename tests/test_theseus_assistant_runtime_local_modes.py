from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_assistant_route_integrity_v2 as integrity  # noqa: E402
import theseus_assistant_runtime as runtime  # noqa: E402


def bind_source_route_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    """Supply explicit route receipts instead of ambient ignored reports."""
    context_address = "vcm://p1-fixture/operator-chat"
    monkeypatch.setattr(
        runtime,
        "assistant_materialized_view_receipt",
        lambda: {"ready": True, "receipt_id": "view:p1-fixture", "record_count": 1},
    )
    monkeypatch.setattr(
        runtime,
        "assistant_route_validator_receipt",
        lambda: {
            "ready": True,
            "receipt_id": "route:p1-fixture",
            "resource_route_record_count": 1,
        },
    )
    monkeypatch.setattr(
        runtime,
        "private_verifier_receipt_packet",
        lambda: {"ready": True, "receipt_id": "verifier:p1-fixture"},
    )
    monkeypatch.setattr(
        runtime,
        "vcm_context_governor_packet",
        lambda: {
            "ready": True,
            "trigger_state": "GREEN",
            "adequacy_state": "governed_sufficient",
            "summary": {
                "mission_brief_status": "ready",
                "deletion_closure_status": "closed",
                "deletion_closure_fault_count": 0,
                "scif_status": "ready",
                "hard_gap_count": 0,
                "mission_brief_compression_loss": 0.0,
            },
            "no_cheat": {
                "public_training_rows_written": 0,
                "runtime_external_inference_calls": 0,
                "fallback_return_count": 0,
            },
        },
    )
    monkeypatch.setattr(
        runtime.vcm_consumer_abi,
        "build_consumer_packet",
        lambda **_kwargs: {
            "ready": True,
            "packet_id": "vcm-consumer:p1-fixture",
            "typed_faults": [],
            "records": [],
            "validation": {"passed": True, "faults": []},
        },
    )
    monkeypatch.setattr(
        runtime,
        "teacher_policy_packet",
        lambda: {
            "runtime_external_tokens_forbidden": True,
            "teacher_apply_mode_forbidden": True,
            "public_benchmark_distillation_forbidden": True,
            "teacher_share_metric_ready": True,
            "teacher_share_ledger_state": "GREEN",
            "teacher_share_within_cap": True,
            "distillation_allowed": False,
            "runtime_external_inference_calls": 0,
            "public_training_rows_written": 0,
        },
    )
    original_read_json = runtime.read_json

    def read_json(path: Path, default: object) -> object:
        resolved = Path(path)
        if resolved == runtime.REPORTS / "deterministic_tool_registry.json":
            return {"trigger_state": "GREEN", "tools": [{"tool_id": "fixture.read_only"}]}
        if resolved == runtime.REPORTS / "theseus_plan_compiler.json":
            return {"trigger_state": "GREEN", "summary": {"compiled_goal_count": 1}}
        if resolved == runtime.REPORTS / "vcm_task_contexts.json":
            return {
                "task_contexts": [
                    {
                        "task_family_id": "operator_chat",
                        "ready": True,
                        "selected_pages": [
                            {
                                "address": context_address,
                                "content_hash": runtime.sha256_text("P1 fixture context."),
                                "taints": [],
                                "contradiction_refs": [],
                            }
                        ],
                    }
                ]
            }
        if resolved == runtime.REPORTS / "virtual_context_compiled_context.json":
            return {
                "model_visible_pages": [
                    {
                        "address": context_address,
                        "certificate_id": "certificate:p1-fixture",
                        "title": "P1 fixture",
                        "execution_class": "verified_context",
                        "taints": [],
                        "materialized_text": "P1 fixture context.",
                    }
                ]
            }
        return original_read_json(resolved, default)

    monkeypatch.setattr(runtime, "read_json", read_json)


def args(tmp_path: Path, mode: str) -> argparse.Namespace:
    return argparse.Namespace(
        config="configs/theseus_assistant_runtime.json",
        checkpoint_id="",
        session_id=f"p1_test_{mode}",
        prompt="State the next P1 action.",
        execution_mode=mode,
        surface="local_assistant",
        intent="chat",
        principal="local-user",
        origin="local_user_control",
        unauthenticated=False,
        requested_route="",
        fallback_policy="no_fallback",
        effort="balanced",
        feedback="",
        error_family="",
        out=str(tmp_path / f"{mode}.json"),
        markdown_out=str(tmp_path / f"{mode}.md"),
        events_out=str(tmp_path / "events.jsonl"),
        viea_trace_out=str(tmp_path / "trace.jsonl"),
        skip_context_refresh=True,
        skip_dogfood=True,
        effect_canary=False,
        effect_target="runtime/assistant_effects/default_route_authority.json",
        print_answer=False,
        local_maximum_tokens=0,
    )


def test_canonical_runtime_runs_matched_direct_and_integrated_paths_without_real_model_load(
    tmp_path: Path, monkeypatch
) -> None:
    bind_source_route_readiness(monkeypatch)
    config = json.loads((ROOT / "configs" / "theseus_assistant_runtime.json").read_text(encoding="utf-8"))
    local = config["local_inference"]
    identity = integrity.load_model_contract(
        local["worker_config"],
        local["runtime_preflight"],
        maximum_tokens=local["product_maximum_tokens"],
    )["identity"]
    observed_prompts: dict[str, str] = {}

    def fake_local_inference(**kwargs: object) -> dict:
        mode = str(kwargs["execution_mode"])
        prompt = str(kwargs["prompt"])
        observed_prompts[mode] = prompt
        payload = {
            "policy": "project_theseus_local_inference_backend_v1",
            "trigger_state": "GREEN",
            "backend": {"identity": identity},
            "request": {
                "execution_mode": mode,
                "prompt_sha256": runtime.sha256_text(prompt),
                "route_context_digest": kwargs["route_context_digest"],
            },
            "response": {
                "mode": "frozen_tmax_local_inference",
                "answer": "P1 fake local answer.",
                "teacher_recommended": False,
            },
            "session": {
                "session_id": kwargs["session_id"],
                "history_turns_loaded": 0,
                "session_path": "",
                "persistence": "disabled_no_raw_text_retention",
            },
            "metrics": {"local_model_inference_calls": 1, "generated_tokens": 5},
            "external_inference_calls": 0,
        }
        runtime.write_json(Path(kwargs["out"]), payload)
        return {
            "id": "local_inference",
            "returncode": 0,
            "stderr_tail": "",
            "runtime_ms": 1,
            "command": ["fake-local-inference"],
            "prompt_transport": "stdin_not_argv",
        }

    monkeypatch.setattr(runtime, "run_local_inference", fake_local_inference)
    direct = runtime.build_report(args(tmp_path, integrity.DIRECT_MODE), time.perf_counter())
    integrated = runtime.build_report(args(tmp_path, integrity.INTEGRATED_MODE), time.perf_counter())

    assert direct["trigger_state"] == "GREEN"
    assert integrated["trigger_state"] == "GREEN"
    assert direct["generation_backend"]["id"] == "local_inference"
    assert integrated["generation_backend"]["id"] == "local_inference"
    assert direct["route_integrity"]["ready"] is True
    assert integrated["route_integrity"]["ready"] is True
    assert observed_prompts[integrity.DIRECT_MODE] == "State the next P1 action."
    assert "[theseus_verified_vcm_context]" in observed_prompts[integrity.INTEGRATED_MODE]
    assert "[theseus_executed_route]" in observed_prompts[integrity.INTEGRATED_MODE]
    assert integrity.compare_matched_pair(
        direct["route_integrity"], integrated["route_integrity"]
    )["ready"] is True


def test_runtime_experimental_token_override_is_explicit_and_context_scoped(
    tmp_path: Path, monkeypatch
) -> None:
    bind_source_route_readiness(monkeypatch)
    observed: dict[str, object] = {}

    def fake_runner(**kwargs: object) -> dict:
        observed["maximum_tokens"] = dict(kwargs["config"])["product_maximum_tokens"]
        local = dict(kwargs["config"])
        identity = integrity.load_model_contract(
            local["worker_config"],
            local["runtime_preflight"],
            maximum_tokens=local["product_maximum_tokens"],
        )["identity"]
        payload = {
            "policy": "project_theseus_local_inference_backend_v1",
            "trigger_state": "GREEN",
            "backend": {"identity": identity},
            "request": {
                "execution_mode": kwargs["execution_mode"],
                "prompt_sha256": runtime.sha256_text(str(kwargs["prompt"])),
                "route_context_digest": kwargs["route_context_digest"],
            },
            "response": {"answer": "typed edit", "teacher_recommended": False},
            "session": {"session_id": kwargs["session_id"], "history_turns_loaded": 0},
            "metrics": {"local_model_inference_calls": 1},
            "external_inference_calls": 0,
        }
        runtime.write_json(Path(kwargs["out"]), payload)
        return {"id": "local_inference", "returncode": 0, "stderr_tail": ""}

    request_args = args(tmp_path, integrity.DIRECT_MODE)
    request_args.local_maximum_tokens = 1536
    with runtime.bind_local_inference_runner(fake_runner):
        report = runtime.build_report(request_args, time.perf_counter())

    assert report["trigger_state"] == "GREEN"
    assert observed["maximum_tokens"] == 1536
    assert runtime._LOCAL_INFERENCE_RUNNER.get() is None


def test_runtime_config_overlay_deep_merges_local_model_binding(tmp_path: Path) -> None:
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps({"local_inference": {"worker_config": "old", "timeout_seconds": 1}, "keep": 2}))
    child = tmp_path / "child.json"
    child.write_text(json.dumps({"extends": str(parent), "local_inference": {"worker_config": "new"}}))

    merged = runtime.load_runtime_config(child)

    assert merged["local_inference"] == {"worker_config": "new", "timeout_seconds": 1}
    assert merged["keep"] == 2
