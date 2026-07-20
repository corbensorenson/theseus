from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from kerc_residual_economics import (  # noqa: E402
    ResidualEconomicsFault,
    allocate_rate_distortion,
    build_residual_codec,
    build_residual_unit_packet,
    build_structural_rate_distortion_allocation,
    calibrate_allocation_lambda,
    decode_conditional_payload,
    encode_conditional_payload,
    promotion_economics,
    validate_promotion_economics,
    validate_residual_codec,
    validate_residual_unit_packet,
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


def test_per_unit_packet_separates_units_payloads_render_plans_and_constraints() -> None:
    arguments = {
        "source_record_sha256": "sha256:" + "1" * 64,
        "residual_mode": "SOURCE_RECONSTRUCTION",
        "kernel_program": {"nodes": [{"operator": "ASSERT"}]},
        "global_state": {
            "language": "en",
            "terminology": {"APPROVE": "authorize"},
        },
        "segment_residual": {
            "schema": "frame_v1",
            "frames": [{"frame_name": "Statement", "frame_roles": ["TOPIC"]}],
        },
        "token_residuals": [
            {
                "tag": "ENTITY_MENTION:PLACE",
                "source_span": [0, 5],
                "authority": "licensed_manual_annotation",
            }
        ],
        "concept_capsules": {
            "@C0": {
                "stable_identity": "concept.example",
                "preferred_realization": "Example",
            }
        },
        "exact_objects": {
            "@Q1": {
                "object_type": "QUOTE",
                "copy_policy": "EXACT",
                "inline_bytes_b64": "RXhhY3Q=",
            }
        },
    }
    packet = build_residual_unit_packet(**arguments)
    assert validate_residual_unit_packet(packet, **arguments) == packet
    assert packet["source_residual_and_render_plan_separated"] is True
    assert packet["packet_wide_fidelity_drives_training"] is False
    assert packet["learned_allocator_claimed"] is False
    assert set(packet["summary"]["unit_counts_by_kind"]) == {
        "interaction_entry",
        "segment_frame",
        "token_residue",
        "concept_realization",
        "exact_object",
    }
    assert all(
        unit["source_residual"]["policy"]
        != unit["render_plan"]["policy"]
        for unit in packet["units"]
    )
    assert packet["distortion_dimensions"] == [
        "semantic_proposition",
        "entity_identity",
        "value_unit_precision",
        "scope",
        "polarity",
        "modality",
        "temporal",
        "causal",
        "attribution",
        "quote",
        "terminology",
        "style",
        "byte",
    ]
    assert all(
        len(candidate["distortion_vector"]) == len(packet["distortion_dimensions"])
        and all(
            value is None or 0.0 <= value <= 1.0
            for value in candidate["distortion_vector"]
        )
        for unit in packet["units"]
        for candidate in unit["candidates"]
    )
    exact = next(unit for unit in packet["units"] if unit["unit_kind"] == "exact_object")
    assert exact["minimum_fidelity"] == "exact"
    assert exact["selected_fidelity"] == "exact"
    assert all(
        candidate["hard_blocked"]
        for candidate in exact["candidates"]
        if candidate["fidelity"] != "exact"
    )
    unrelated = [
        unit for unit in packet["units"] if unit["unit_kind"] != "exact_object"
    ]
    assert any(unit["minimum_fidelity"] != "exact" for unit in unrelated)
    assert any(unit["selected_fidelity"] != "exact" for unit in unrelated)


def test_per_unit_packet_is_replay_stable_and_tamper_evident() -> None:
    arguments = {
        "source_record_sha256": "sha256:" + "2" * 64,
        "residual_mode": "SOURCE_RECONSTRUCTION",
        "kernel_program": {"nodes": [{"operator": "ASSERT"}]},
        "global_state": {"language": "en"},
        "segment_residual": {},
        "token_residuals": [
            {
                "tag": "ENTITY:PERSON",
                "source_span": [0, 4],
                "authority": "licensed_manual_annotation",
            }
        ],
        "concept_capsules": {},
        "exact_objects": {},
    }
    first = build_residual_unit_packet(**arguments)
    second = build_residual_unit_packet(**arguments)
    assert first == second
    assert len({unit["unit_id"] for unit in first["units"]}) == len(first["units"])
    tampered = copy.deepcopy(first)
    tampered["units"][0]["selected_encoded_bits"] += 1
    with pytest.raises(ResidualEconomicsFault, match="UNIT_PACKET_INVALID"):
        validate_residual_unit_packet(tampered, **arguments)


def test_per_unit_codec_conditioning_cannot_include_the_priced_unit() -> None:
    arguments = {
        "source_record_sha256": "sha256:" + "3" * 64,
        "residual_mode": "SOURCE_RECONSTRUCTION",
        "kernel_program": {"nodes": [{"operator": "ASSERT"}]},
        "global_state": {"language": "en"},
        "segment_residual": {},
        "token_residuals": [],
        "concept_capsules": {},
        "exact_objects": {},
    }
    first = build_residual_unit_packet(**arguments)
    changed = build_residual_unit_packet(
        **{**arguments, "global_state": {"language": "changed-secret-value"}}
    )
    first_unit = first["units"][0]
    changed_unit = changed["units"][0]
    assert first["conditioning_excludes_current_unit"] is True
    assert first_unit["unit_id"] == changed_unit["unit_id"]
    assert first_unit["condition_sha256"] == changed_unit["condition_sha256"]
    assert (
        first_unit["source_residual"]["payload_sha256"]
        != changed_unit["source_residual"]["payload_sha256"]
    )


def test_per_unit_codec_uses_only_strictly_higher_residual_state() -> None:
    arguments = {
        "source_record_sha256": "sha256:" + "4" * 64,
        "residual_mode": "SOURCE_RECONSTRUCTION",
        "kernel_program": {"nodes": [{"operator": "ASSERT"}]},
        "global_state": {"language": "en"},
        "segment_residual": {"frames": [{"frame_name": "Statement"}]},
        "token_residuals": [
            {
                "tag": "ENTITY:PERSON",
                "source_span": [0, 4],
                "authority": "licensed_manual_annotation",
            }
        ],
        "concept_capsules": {},
        "exact_objects": {},
    }
    first = build_residual_unit_packet(**arguments)
    changed = build_residual_unit_packet(
        **{
            **arguments,
            "segment_residual": {"frames": [{"frame_name": "Question"}]},
        }
    )
    first_segment = next(
        unit for unit in first["units"] if unit["unit_kind"] == "segment_frame"
    )
    changed_segment = next(
        unit for unit in changed["units"] if unit["unit_kind"] == "segment_frame"
    )
    first_token = next(
        unit for unit in first["units"] if unit["unit_kind"] == "token_residue"
    )
    changed_token = next(
        unit for unit in changed["units"] if unit["unit_kind"] == "token_residue"
    )
    assert first_segment["condition_sha256"] == changed_segment["condition_sha256"]
    assert first_token["condition_sha256"] != changed_token["condition_sha256"]


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
