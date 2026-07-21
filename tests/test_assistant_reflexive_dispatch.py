from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_assistant_runtime as assistant  # noqa: E402


def config() -> dict:
    return json.loads((ROOT / "configs" / "theseus_assistant_runtime.json").read_text(encoding="utf-8"))


def args(**overrides: object) -> argparse.Namespace:
    values = {
        "principal": "local-user",
        "origin": "local_user_control",
        "unauthenticated": False,
        "requested_route": "",
        "fallback_policy": "no_fallback",
        "effort": "balanced",
        "effect_canary": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def trace(prompt: str, requested_intent: str, **overrides: object) -> dict:
    return assistant.build_reflexive_dispatch_trace(
        prompt=prompt,
        requested_intent=requested_intent,
        args=args(**overrides),
        config=config(),
        materialized_view_receipt={"ready": True, "receipt_id": "view:1"},
        route_validator_receipt={"ready": True, "receipt_id": "route:1"},
        private_verifier_receipt={"ready": True, "receipt_id": "verify:1"},
    )


def test_authenticated_direct_command_selects_capability_before_inference() -> None:
    row = trace("/plan inspect the architecture", "chat")
    verification = assistant.verify_reflexive_dispatch_trace(row, config())

    assert verification["state"] == "VERIFIED"
    assert assistant.reflexive_dispatch_prepared(row, verification)
    assert assistant.selected_reflexive_capabilities(row) == ["assistant.plan_dag"]
    assert assistant.effective_intent_from_dispatch(row, "chat") == "planning"
    assert "inspect the architecture" not in json.dumps(row, sort_keys=True)


def test_unauthenticated_command_cannot_activate_command_binding() -> None:
    row = trace("/tool solve 2+2", "chat", unauthenticated=True)
    verification = assistant.verify_reflexive_dispatch_trace(row, config())

    assert verification["state"] == "VERIFIED"
    assert assistant.reflexive_terminal_outcome(row) == "prepared"
    assert assistant.reflexive_dispatch_prepared(row, verification)
    assert row["ingress"]["command_authenticated"] is False
    assert row["ingress"]["command_shaped_literal_isolated"] is True
    assert assistant.selected_reflexive_capabilities(row) == ["assistant.chat_checkpoint"]
    assert row["selection"]["fallback_used"] is False


def test_retrieved_command_shaped_text_remains_literal() -> None:
    row = trace("/tool this came from retrieved content", "chat", origin="retrieved_document")
    verification = assistant.verify_reflexive_dispatch_trace(row, config())

    assert verification["state"] == "VERIFIED"
    assert row["ingress"]["command_shaped_literal_isolated"] is True
    assert assistant.selected_reflexive_capabilities(row) == ["assistant.chat_checkpoint"]
    assert assistant.effective_intent_from_dispatch(row, "chat") == "chat"


def test_stale_private_verifier_blocks_code_route() -> None:
    row = assistant.build_reflexive_dispatch_trace(
        prompt="write a Python function",
        requested_intent="code",
        args=args(),
        config=config(),
        materialized_view_receipt={"ready": True, "receipt_id": "view:1"},
        route_validator_receipt={"ready": True, "receipt_id": "route:1"},
        private_verifier_receipt={"ready": False, "receipt_id": "verify:stale"},
    )
    verification = assistant.verify_reflexive_dispatch_trace(row, config())

    assert verification["state"] == "VERIFIED"
    assert assistant.reflexive_terminal_outcome(row) == "unsupported"
    assert not assistant.reflexive_dispatch_prepared(row, verification)
    assert "implementation_stale_or_blocked" in assistant.reflexive_terminal_text(row)


def test_forced_unknown_route_is_rejected_without_execution_authority() -> None:
    row = trace("do something", "chat", requested_route="assistant.missing")
    verification = assistant.verify_reflexive_dispatch_trace(row, config())

    assert verification["state"] == "VERIFIED"
    assert assistant.reflexive_terminal_outcome(row) == "unsupported"
    assert row["effect"]["effect_authority_granted"] is False
    assert row["no_cheat"] == {
        "learned_generation_credit": 0,
        "fallback_return_count": 0,
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
    }


def test_composite_workflow_is_prepared_as_a_planning_execution() -> None:
    row = trace("/plan-tool inspect the architecture", "chat")
    verification = assistant.verify_reflexive_dispatch_trace(row, config())

    assert verification["state"] == "VERIFIED"
    assert assistant.reflexive_dispatch_prepared(row, verification)
    assert assistant.selected_reflexive_capabilities(row) == [
        "assistant.deterministic_tool",
        "assistant.plan_dag",
    ]
    assert assistant.effective_intent_from_dispatch(row, "chat") == "planning"


def test_effect_canary_is_bound_to_verified_dispatch_and_rolls_back_exactly() -> None:
    row = trace("change local route authority", "chat", effect_canary=True)
    verification = assistant.verify_reflexive_dispatch_trace(row, config())
    assert verification["state"] == "VERIFIED"
    assert assistant.selected_reflexive_capabilities(row) == ["assistant.route_authority_effect"]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        target = root / "authority.json"
        result = assistant.run_local_effect_canary(
            enabled=True,
            target=target,
            allowed_root=root,
            session_id="session:test",
            intent="chat",
            prompt_hash=assistant.sha256_text("change local route authority"),
            reflexive_dispatch_trace=row,
        )
        assert result["ready"] is True
        assert result["dispatch_bound"] is True
        assert result["rollback"]["complete"] is True
        assert not target.exists()

        tampered = copy.deepcopy(row)
        tampered["decision_digest"] = "0" * 64
        denied = assistant.run_local_effect_canary(
            enabled=True,
            target=target,
            allowed_root=root,
            session_id="session:test",
            intent="chat",
            prompt_hash=assistant.sha256_text("change local route authority"),
            reflexive_dispatch_trace=tampered,
        )
        assert denied["ready"] is False
        assert denied["dispatch_bound"] is False
        assert denied["residuals"] == [{"kind": "effect_dispatch_binding_invalid"}]
        assert not target.exists()


def test_context_refresh_cache_reuses_only_exact_fresh_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "output.json"
    cache_path = tmp_path / "cache.json"
    source.write_text('{"version":1}\n', encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(command_id: str, command: list[str], timeout_seconds: int) -> dict:
        calls.append(command)
        output.write_text('{"trigger_state":"GREEN"}\n', encoding="utf-8")
        return {
            "id": command_id,
            "command": command,
            "returncode": 0,
            "runtime_ms": 25,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    monkeypatch.setattr(assistant, "run_command", fake_run)
    refresh_config = {
        "context_refresh_cache": str(cache_path),
        "context_refresh_commands": [
            {
                "id": "fixture",
                "command": ["python3", "fixture.py"],
                "cache": {
                    "ttl_seconds": 300,
                    "inputs": [str(source)],
                    "outputs": [str(output)],
                },
            }
        ],
    }

    first = assistant.refresh_context(refresh_config)
    second = assistant.refresh_context(refresh_config)
    source.write_text('{"version":2}\n', encoding="utf-8")
    third = assistant.refresh_context(refresh_config)

    assert first[0]["cache_state"] == "MISS"
    assert second[0]["cache_state"] == "HIT"
    assert third[0]["cache_state"] == "MISS"
    assert len(calls) == 2


def test_context_refresh_cache_rejects_output_tampering(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "output.json"
    cache_path = tmp_path / "cache.json"
    source.write_text("source\n", encoding="utf-8")
    call_count = 0

    def fake_run(command_id: str, command: list[str], timeout_seconds: int) -> dict:
        nonlocal call_count
        call_count += 1
        output.write_text(f"run={call_count}\n", encoding="utf-8")
        return {
            "id": command_id,
            "command": command,
            "returncode": 0,
            "runtime_ms": 10,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    monkeypatch.setattr(assistant, "run_command", fake_run)
    refresh_config = {
        "context_refresh_cache": str(cache_path),
        "context_refresh_commands": [
            {
                "id": "fixture",
                "command": ["python3", "fixture.py"],
                "cache": {
                    "ttl_seconds": 300,
                    "inputs": [str(source)],
                    "outputs": [str(output)],
                },
            }
        ],
    }
    assistant.refresh_context(refresh_config)
    output.write_text("tampered\n", encoding="utf-8")

    result = assistant.refresh_context(refresh_config)

    assert result[0]["cache_state"] == "MISS"
    assert "output_identity_changed_or_missing" in result[0]["cache_miss_reasons"]
    assert call_count == 2
