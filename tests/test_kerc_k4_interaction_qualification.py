from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import kerc_k4_interaction_qualification as k4
import kerc_k4_candidate_producer as producer


def fixture(tmp_path: Path) -> tuple[dict, Path]:
    config = json.loads(
        (ROOT / "configs/kerc_k4_interaction_qualification_v2.json").read_text(
            encoding="utf-8"
        )
    )
    config_path = tmp_path / "configs/kerc_k4_interaction_qualification_v2.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config, config_path


def expected_output(metadata: dict) -> str:
    values = [str(value) for value in metadata["expected"].get("expected_terms") or []]
    values.extend(str(group[0]) for group in metadata["expected"].get("expected_any") or [] if group)
    return " ".join(values) or "supported answer"


def write_producer_evidence(
    config: dict,
    *,
    root: Path,
    packet_path: Path,
    outputs_path: Path,
) -> None:
    contract = config["kerc_k4_qualification"]
    outputs, faults = k4.load_outputs(outputs_path)
    assert not faults
    producer_path = ROOT / "scripts/kerc_k4_candidate_producer.py"
    generator_path = ROOT / "scripts/moecot_language_arm_training.py"
    report_path = root / contract["candidate_producer_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "policy": "project_theseus_kerc_k4_candidate_producer_v2",
        "surface_id": contract["surface_id"],
        "trigger_state": "GREEN",
        "packet": {"sha256": k4.sha256(packet_path)},
        "outputs": {"sha256": k4.sha256(outputs_path)},
        "row_count": len(outputs),
        "candidate_visible_fields": ["prompt"],
        "answer_identifying_metadata_exposed": False,
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "fallback_template_router_tool_credit": 0,
        "observations": [
            {
                "opaque_id": opaque_id,
                "output_sha256": k4.hashlib.sha256(output.encode()).hexdigest(),
            }
            for opaque_id, output in outputs.items()
        ],
        "source_artifacts": {
            "producer": {
                "path": str(producer_path),
                "sha256": k4.sha256(producer_path),
            },
            "generator": {
                "path": str(generator_path),
                "sha256": k4.sha256(generator_path),
            },
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    guard_path = root / contract["candidate_producer_guard_receipt"]
    guard_path.write_text(
        json.dumps(
            {
                "passed": True,
                "returncode": 0,
                "maximum_swapout_growth_mib": 0.0,
                "limits": {"maximum_swapout_growth_mib": 16},
            }
        ),
        encoding="utf-8",
    )


def test_prompt_packet_exposes_only_prompt_seed_and_opaque_identity(
    tmp_path: Path,
) -> None:
    config, config_path = fixture(tmp_path)
    packet, hidden = k4.build_prompt_packet(
        config, config_path=config_path, root=tmp_path
    )

    rows = [
        row for group in packet["checkpoint_groups"] for row in group["rows"]
    ]
    assert packet["row_count"] == 108
    assert len(hidden) == 108
    assert len({row["opaque_id"] for row in rows}) == 108
    assert all(set(row) == {"opaque_id", "prompt"} for row in rows)
    serialized_rows = json.dumps(rows, sort_keys=True)
    assert '"case_id"' not in serialized_rows
    assert '"intervention"' not in serialized_rows
    assert '"expected_terms"' not in serialized_rows
    assert '"model_seed"' not in serialized_rows
    assert packet["information_flow"]["answer_identifying_metadata_exposed"] is False
    assert packet["information_flow"]["candidate_visible_fields"] == ["prompt"]


def test_shuffled_and_stale_controls_are_distinct_for_every_case(
    tmp_path: Path,
) -> None:
    config, _config_path = fixture(tmp_path)
    contract = k4.validate_contract(config)
    cases = k4.case_map(config)
    for case_id in contract["selected_case_ids"]:
        prompts = k4.intervention_prompts(
            case_id,
            cases=cases,
            contract=contract,
        )
        assert prompts["context_present"] != prompts["context_shuffled"]
        assert prompts["context_present"] != prompts["stale_state"]
        assert prompts["context_shuffled"] != prompts["stale_state"]


def test_candidate_producer_accepts_only_grouped_prompt_only_rows(
    tmp_path: Path,
) -> None:
    config, config_path = fixture(tmp_path)
    packet, _hidden = k4.build_prompt_packet(
        config, config_path=config_path, root=tmp_path
    )
    by_seed = producer.validate_packet(packet)
    assert set(by_seed) == {20260722, 20260723, 20260724}
    assert all(len(rows) == 36 for rows in by_seed.values())
    assert all(set(row) == {"opaque_id", "prompt"} for rows in by_seed.values() for row in rows)

    tampered = copy.deepcopy(packet)
    tampered["checkpoint_groups"][0]["rows"][0]["model_seed"] = 20260722
    with pytest.raises(ValueError, match="candidate row schema"):
        producer.validate_packet(tampered)


def test_complete_direct_outputs_can_pass_frozen_k4_causal_gate(
    tmp_path: Path,
) -> None:
    config, config_path = fixture(tmp_path)
    packet, hidden = k4.build_prompt_packet(
        config, config_path=config_path, root=tmp_path
    )
    packet_path = tmp_path / "reports/kerc_k4_prompt_packet.json"
    packet_path.parent.mkdir(parents=True)
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    outputs_path = tmp_path / "reports/kerc_k4_candidate_outputs.jsonl"
    rows = []
    for opaque_id, metadata in hidden.items():
        output = (
            expected_output(metadata)
            if metadata["intervention"] in {"context_present", "expansion_replay"}
            else "I do not have enough supported context."
        )
        rows.append(json.dumps({"opaque_id": opaque_id, "output": output}))
    outputs_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    write_producer_evidence(
        config,
        root=tmp_path,
        packet_path=packet_path,
        outputs_path=outputs_path,
    )

    report = k4.evaluate(
        config,
        config_path=config_path,
        packet_path=packet_path,
        outputs_path=outputs_path,
        root=tmp_path,
    )

    assert report["trigger_state"] == "GREEN"
    assert report["source_family_disjoint"] is True
    assert report["seed_count"] == 3
    assert report["case_count"] == 6
    assert report["effect"]["confidence_interval_lower"] > 0
    assert report["wrong_user_donor_leakage_rate"] == 0
    assert report["forbidden_term_output_rate"] == 0


def test_candidate_emitted_metadata_and_missing_rows_fail_closed(
    tmp_path: Path,
) -> None:
    config, config_path = fixture(tmp_path)
    packet, hidden = k4.build_prompt_packet(
        config, config_path=config_path, root=tmp_path
    )
    packet_path = tmp_path / "reports/kerc_k4_prompt_packet.json"
    packet_path.parent.mkdir(parents=True)
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    first_id = next(iter(hidden))
    outputs_path = tmp_path / "reports/kerc_k4_candidate_outputs.jsonl"
    outputs_path.write_text(
        json.dumps(
            {
                "opaque_id": first_id,
                "output": "candidate says green",
                "passed": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = k4.evaluate(
        config,
        config_path=config_path,
        packet_path=packet_path,
        outputs_path=outputs_path,
        root=tmp_path,
    )

    assert report["trigger_state"] == "RED"
    assert any(fault.startswith("candidate_output_schema_invalid") for fault in report["faults"])
    assert "candidate_output_identity_set_mismatch" in report["faults"]


def test_complete_empty_model_outputs_are_recorded_as_behavioral_failure(
    tmp_path: Path,
) -> None:
    config, config_path = fixture(tmp_path)
    packet, hidden = k4.build_prompt_packet(
        config, config_path=config_path, root=tmp_path
    )
    packet_path = tmp_path / "reports/kerc_k4_prompt_packet.json"
    packet_path.parent.mkdir(parents=True)
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    outputs_path = tmp_path / "reports/kerc_k4_candidate_outputs.jsonl"
    outputs_path.write_text(
        "".join(
            json.dumps({"opaque_id": opaque_id, "output": ""}) + "\n"
            for opaque_id in hidden
        ),
        encoding="utf-8",
    )

    report = k4.evaluate(
        config,
        config_path=config_path,
        packet_path=packet_path,
        outputs_path=outputs_path,
        root=tmp_path,
    )

    assert report["trigger_state"] == "RED"
    assert report["missing_output_count"] == 0
    assert report["acceptance_predicates"]["outputs_complete"] is True
    assert report["acceptance_predicates"]["context_present"] is False
