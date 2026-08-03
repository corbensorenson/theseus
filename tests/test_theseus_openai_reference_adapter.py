from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SCRIPT = ROOT / "scripts" / "theseus_openai_reference_adapter.py"
SPEC = importlib.util.spec_from_file_location("theseus_openai_reference_adapter", SCRIPT)
assert SPEC and SPEC.loader
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)

AUDIT_SCRIPT = ROOT / "scripts" / "external_inference_audit.py"
AUDIT_SPEC = importlib.util.spec_from_file_location("external_inference_audit_reference_test", AUDIT_SCRIPT)
assert AUDIT_SPEC and AUDIT_SPEC.loader
external_audit = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(external_audit)


def policy() -> dict:
    return json.loads(
        (ROOT / "configs" / "theseus_external_reference_control.json").read_text(
            encoding="utf-8"
        )
    )


def packet_and_seal() -> tuple[dict, dict]:
    value = policy()
    packet = adapter.qualification_packet(value)
    return packet, adapter.qualification_seal(value, packet)


def test_offline_qualification_is_green_and_makes_zero_calls() -> None:
    value = policy()

    report = adapter.qualify_adapter(
        value,
        policy_path=ROOT / "configs" / "theseus_external_reference_control.json",
    )

    assert report["trigger_state"] == "GREEN"
    assert report["hard_gaps"] == []
    assert all(report["negative_controls"].values())
    assert report["counters"]["reference_calls_made"] == 0


def test_request_is_exact_luna_xhigh_responses_shape_without_output_cap() -> None:
    value = policy()
    packet, seal = packet_and_seal()

    request = adapter.build_request(value, packet, seal)

    assert request["model"] == "gpt-5.6-luna"
    assert request["reasoning"] == {"effort": "xhigh", "context": "current_turn"}
    assert request["store"] is False
    assert request["tools"] == []
    assert request["parallel_tool_calls"] is False
    assert "max_output_tokens" not in request
    assert request["metadata"]["role"] == "measurement_only_reference_control"


def test_hidden_target_metadata_is_rejected_recursively() -> None:
    value = policy()
    packet, seal = packet_and_seal()
    packet["allowed_runtime_context"] = [{"nested": {"hidden_tests": ["no"]}}]

    faults = adapter.validate_execution_authority(
        value,
        packet,
        seal,
        require_calls_authorized=False,
    )

    assert "candidate_hidden_field_present:hidden_tests" in faults


def test_execute_fails_before_transport_when_calls_are_not_authorized(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    value = policy()
    packet, seal = packet_and_seal()
    transport_used = False

    def forbidden_transport(*args: object, **kwargs: object) -> None:
        nonlocal transport_used
        transport_used = True
        raise AssertionError("network transport must not be reached")

    monkeypatch.setattr(adapter.urllib.request, "urlopen", forbidden_transport)

    with pytest.raises(adapter.ReferenceGateError, match="reference_calls_not_authorized"):
        adapter.execute_reference_call(
            policy=value,
            packet=packet,
            seal=seal,
            candidate_out=tmp_path / "candidate.txt",
        )

    assert transport_used is False


def test_spend_preflight_prices_physical_boundary_without_capping_generation() -> None:
    value = policy()

    assert adapter.worst_case_call_cost_usd(value) == pytest.approx(0.6504)
    assert adapter.observed_cost_usd(
        value,
        {
            "input_tokens": 1000,
            "cached_input_tokens": 200,
            "output_tokens": 500,
        },
    ) == pytest.approx(0.000764)

    assert adapter.observed_cost_usd(
        value,
        {
            "input_tokens": 1000,
            "cached_input_tokens": 200,
            "cache_write_input_tokens": 100,
            "output_tokens": 500,
        },
    ) == pytest.approx(0.000769)


def test_response_receipt_extracts_usage_and_output_without_serving_it() -> None:
    response = {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "candidate artifact"}],
            }
        ],
        "usage": {
            "input_tokens": 12,
            "input_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 0},
            "output_tokens": 7,
            "output_tokens_details": {"reasoning_tokens": 2},
            "total_tokens": 19,
        },
    }

    assert adapter.extract_output_text(response) == "candidate artifact"
    assert adapter.normalize_usage(response["usage"]) == {
        "input_tokens": 12,
        "cached_input_tokens": 3,
        "cache_write_input_tokens": 0,
        "output_tokens": 7,
        "reasoning_tokens": 2,
        "total_tokens": 19,
    }
    assert adapter.response_finish_reason(response) == "completed"


def test_external_inference_audit_allows_only_the_sealed_reference_surface() -> None:
    allowed_patterns = {
        "openai_api_endpoint",
        "openai_api_key",
        "generic_bearer_auth",
    }
    for path in (
        Path("scripts/theseus_openai_reference_adapter.py"),
        Path("configs/theseus_external_reference_control.json"),
    ):
        for pattern in allowed_patterns:
            assert external_audit.classify(path, pattern) == "allowed_measurement_reference"

    assert (
        external_audit.classify(Path("scripts/unregistered_remote.py"), "openai_api_endpoint")
        == "violation"
    )
