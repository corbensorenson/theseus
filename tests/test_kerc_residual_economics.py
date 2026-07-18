from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from kerc_residual_economics import (
    ResidualEconomicsFault,
    allocate_rate_distortion,
    build_residual_codec,
    build_structural_rate_distortion_allocation,
    calibrate_allocation_lambda,
    decode_conditional_payload,
    encode_conditional_payload,
    promotion_economics,
    validate_promotion_economics,
    validate_residual_codec,
)
from vcm_semantic_memory import (  # noqa: E402
    HRLStateFault,
    apply_hierarchical_residual_delta,
    create_hierarchical_residual_state,
)


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"a",
        b"Kernel English residual residual residual",
        bytes(range(256)),
        bytes(range(255, -1, -1)) * 3,
    ],
)
def test_conditional_arithmetic_codec_is_exact(payload: bytes) -> None:
    conditioning = b'KERNEL:{"operator":"ASSERT"}'
    record = encode_conditional_payload(payload, conditioning)
    assert decode_conditional_payload(record, conditioning) == payload
    assert record["exact_roundtrip"] is True


def test_conditional_codec_rejects_wrong_context_and_tampering() -> None:
    record = encode_conditional_payload(b"residual", b"kernel-a")
    with pytest.raises(ResidualEconomicsFault, match="CONDITION_MISMATCH"):
        decode_conditional_payload(record, b"kernel-b")
    tampered = copy.deepcopy(record)
    tampered["encoded_bits"] += 1
    with pytest.raises(ResidualEconomicsFault, match="RECORD_INVALID"):
        decode_conditional_payload(tampered, b"kernel-a")


def test_codec_golden_vector_prevents_silent_wire_drift() -> None:
    record = encode_conditional_payload(
        b"Kernel English residual residual residual",
        b'KERNEL:{"operator":"ASSERT"}',
    )
    assert record["encoded_base64"] == (
        "S9fAwiuHws9ADJuE8WGlQO6K6XQznz8nxVMZroHW35g6JNNCqZQ="
    )
    assert record["encoded_bits"] == 302


def test_codec_fuzz_roundtrip_across_payload_and_condition_shapes() -> None:
    rng = random.Random(20260717)
    for _ in range(64):
        payload = rng.randbytes(rng.randrange(0, 2049))
        conditioning = rng.randbytes(rng.randrange(0, 1025))
        record = encode_conditional_payload(payload, conditioning)
        assert decode_conditional_payload(record, conditioning) == payload


def test_hierarchical_codec_replays_every_channel() -> None:
    arguments = {
        "kernel_program": {"nodes": [{"operator": "ASSERT"}]},
        "global_state": {"dialect": "en-US", "term": "Kernel English"},
        "segment_residual": {"voice": "active"},
        "token_residuals": [{"position": 1, "realization": "authorize"}],
        "exact_objects": {"@Q1": {"bytes": "Do not change me."}},
    }
    record = build_residual_codec(**arguments)
    receipt = validate_residual_codec(record, **arguments)
    assert receipt["state"] == "READY"
    assert receipt["total_encoded_bits"] > 0
    assert set(record["channels"]) == {"interaction", "segment", "token", "exact"}
    assert record["wire_schema"] == "kerc_residual_wire_v1"
    assert record["encoded_storage_bytes"] > 0
    assert record["cleartext_abi_storage_bytes"] > 0
    assert record["cleartext_abi_copy_charged_to_wire_bits"] is False
    assert record["cleartext_abi_copy_charged_to_storage"] is True


def test_rate_distortion_allocator_honors_hard_minimum() -> None:
    rows = [
        {
            "fidelity": fidelity,
            "encoded_bits": bits,
            "distortion": distortion,
            "evidence_sha256": f"sha256:{index:064x}",
        }
        for index, (fidelity, bits, distortion) in enumerate(
            (
                ("semantic", 0, 1.0),
                ("faithful", 8, 0.3),
                ("lexical", 16, 0.1),
                ("exact", 64, 0.0),
            )
        )
    ]
    ordinary = allocate_rate_distortion(
        rows, importance=1.0, lambda_value=20.0
    )
    protected = allocate_rate_distortion(
        rows, importance=1.0, lambda_value=20.0, minimum_fidelity="exact"
    )
    assert ordinary["selected_fidelity"] == "faithful"
    assert protected["selected_fidelity"] == "exact"
    assert all(
        row["hard_blocked"]
        for row in protected["candidates"]
        if row["fidelity"] != "exact"
    )


def test_structural_allocation_measures_candidates_and_protects_exact_objects() -> None:
    allocation = build_structural_rate_distortion_allocation(
        kernel_program={"nodes": [{"operator": "ASSERT"}]},
        global_state={"language": "en"},
        segment_residual={"frame_name": "Statement"},
        token_residuals=[{"tag": "TERM:KERC"}],
        exact_objects={"@Q1": {"object_type": "QUOTE", "value": "exact"}},
        importance=0.9,
        lambda_value=512.0,
    )
    assert allocation["minimum_fidelity"] == "exact"
    assert allocation["selected_fidelity"] == "exact"
    assert set(allocation["candidate_evidence"]) == {
        "semantic",
        "faithful",
        "lexical",
        "exact",
    }
    assert allocation["distortion_authority"].endswith("not_semantic_utility")
    assert allocation["capability_or_efficiency_claim"] is False


def test_lambda_calibration_uses_smallest_dev_value_meeting_distortion_ceiling() -> None:
    unprotected = build_structural_rate_distortion_allocation(
        kernel_program={"nodes": [{"operator": "ASSERT"}]},
        global_state={"language": "en"},
        segment_residual={"frame_name": "Statement"},
        token_residuals=[{"tag": "TERM:KERC"}],
        exact_objects={},
        importance=1.0,
        lambda_value=1.0,
    )
    calibration = calibrate_allocation_lambda(
        [unprotected],
        lambda_grid=[1.0, 100.0, 10000.0],
        maximum_importance_weighted_distortion=0.0,
    )
    assert calibration["fit_split"] == "private_dev"
    assert calibration["final_evaluation_used_for_selection"] is False
    assert calibration["selected_lambda"] == 10000.0
    assert calibration["curve"][-1]["passes"] is True


def test_promotion_requires_strict_measured_break_even() -> None:
    before = promotion_economics(
        definition_bits=100, direct_bits=20, reference_bits=4, observed_uses=6
    )
    after = promotion_economics(
        definition_bits=100, direct_bits=20, reference_bits=4, observed_uses=7
    )
    assert before["minimum_uses_strict_break_even"] == 7
    assert before["should_promote"] is False
    assert after["should_promote"] is True
    assert validate_promotion_economics(after) == after


def test_hrl_promotion_transaction_enforces_measured_break_even() -> None:
    state = create_hierarchical_residual_state(
        "codec-test",
        scope={
            "user": "test",
            "project": "theseus",
            "conversation": "codec-test",
        },
    )
    before = promotion_economics(
        definition_bits=100, direct_bits=20, reference_bits=4, observed_uses=6
    )
    with pytest.raises(HRLStateFault, match="PROMOTION_BEFORE_BREAK_EVEN"):
        apply_hierarchical_residual_delta(
            state,
            [
                {
                    "op": "PROMOTE_SHARED_RESIDUAL",
                    "key": "APPROVE",
                    "value": "authorize",
                    "economics": before,
                }
            ],
            expected_state_hash=state["state_hash"],
            actor_authority="compiler",
            actor_id="kerc-residual-policy-v1",
            provenance={"source": "private-test"},
        )
    after = promotion_economics(
        definition_bits=100, direct_bits=20, reference_bits=4, observed_uses=7
    )
    updated, delta = apply_hierarchical_residual_delta(
        state,
        [
            {
                "op": "PROMOTE_SHARED_RESIDUAL",
                "key": "APPROVE",
                "value": "authorize",
                "economics": after,
            }
        ],
        expected_state_hash=state["state_hash"],
        actor_authority="compiler",
        actor_id="kerc-residual-policy-v1",
        provenance={"source": "private-test"},
    )
    entry = updated["global"]["terminology"]["APPROVE"]
    assert entry["promotion_economics"]["should_promote"] is True
    assert delta["applied"][0]["status"] == "applied"
