from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from neural_seed_functional_utility import (
    audit_candidate_provenance,
    build_blind_english_packet,
    build_freeze,
    build_manifest,
    evaluate_bundle,
    validate_freeze,
)


CONFIG_PATH = ROOT / "configs/neural_seed_functional_utility.json"
CONFIG = json.loads(CONFIG_PATH.read_text())


def test_manifest_is_green_disjoint_and_candidate_packet_is_blind() -> None:
    manifest = build_manifest(CONFIG, CONFIG_PATH)

    assert manifest["trigger_state"] == "GREEN"
    assert manifest["source_disjoint_audit"]["state"] == "GREEN"
    assert manifest["source_disjoint_audit"]["rows_scanned"] > 0
    assert manifest["candidate_packet"]["row_count"] == 160
    assert manifest["candidate_packet"]["evaluator_metadata_present"] is False
    assert all(set(row) == {"case_id", "arm_id", "prompt"} for row in manifest["candidate_packet"]["rows"])


def test_blind_english_packet_binds_content_without_model_identity(monkeypatch) -> None:
    manifest = build_manifest(CONFIG, CONFIG_PATH)
    freeze = build_freeze(manifest, CONFIG_PATH)
    bundle = {
        "candidates": [
            {
                "case_id": case["case_id"],
                "output": f"candidate for {case['case_id']}",
            }
            for case in manifest["evaluator_cases"]
        ]
    }
    monkeypatch.setattr(
        "neural_seed_functional_utility.audit_candidate_provenance",
        lambda *_args, **_kwargs: {"state": "GREEN", "hard_gaps": []},
    )

    packet = build_blind_english_packet(CONFIG, manifest, bundle, freeze)

    assert packet["trigger_state"] == "GREEN"
    assert packet["item_count"] == 32
    assert packet["model_identity_present"] is False
    assert packet["checkpoint_identity_present"] is False
    assert packet["reference_answer_present"] is False
    assert "target_id" not in packet
    assert all(
        set(item)
        == {
            "blind_item_id",
            "case_id",
            "prompt",
            "candidate_output",
            "candidate_sha256",
            "dimensions",
            "score_scale",
        }
        for item in packet["items"]
    )


def test_freeze_detects_compiler_or_contract_mutation() -> None:
    manifest = build_manifest(CONFIG, CONFIG_PATH)
    freeze = build_freeze(manifest, CONFIG_PATH)
    assert validate_freeze(manifest, freeze) == []
    assert len(freeze["generation_wrapper_sha256"]) == 64
    assert len(freeze["training_generator_sha256"]) == 64

    mutated = copy.deepcopy(freeze)
    mutated["verifier_sha256"] = "0" * 64
    assert "freeze_identity_mismatch:verifier_sha256" in validate_freeze(manifest, mutated)


def test_candidate_bundle_fails_closed_on_missing_duplicate_and_fake_flags() -> None:
    manifest = build_manifest(CONFIG, CONFIG_PATH)
    freeze = build_freeze(manifest, CONFIG_PATH)
    first = manifest["evaluator_cases"][0]
    bundle = {
        "policy": "project_theseus_direct_model_candidate_bundle_v1",
        "case_contract_sha256": freeze["case_contract_sha256"],
        "generation_function": "moecot_language_arm_training.generate_model_text",
        "templates_renderers_routers_tools_credit": 0,
        "checkpoint_artifacts": [],
        "candidates": [
            {"case_id": first["case_id"], "output": "fake", "passed": True, "learned": True},
            {"case_id": first["case_id"], "output": "fake", "passed": True, "learned": True},
        ],
    }
    result = evaluate_bundle(CONFIG, manifest, bundle, freeze, [])

    assert result["trigger_state"] == "RED"
    assert "duplicate_candidate_case" in result["hard_gaps"]
    assert "candidate_case_set_mismatch" in result["hard_gaps"]
    assert "checkpoint_artifacts_missing" in result["hard_gaps"]
    assert result["boundaries"]["candidate_self_declared_flags_trusted"] is False


def test_candidate_provenance_binds_receipt_output_and_target(tmp_path: Path) -> None:
    target = "dense_active_parameter"
    directory = tmp_path / "checkpoints/moecot_language_seed_v8" / target
    directory.mkdir(parents=True)
    checkpoint = directory / "weights.npz"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    freeze = {
        "case_contract_sha256": "c" * 64,
        "candidate_packet_sha256": "p" * 64,
        "generation_wrapper_sha256": "g" * 64,
        "training_generator_sha256": "t" * 64,
        "v8_plan_sha256": "a" * 64,
        "v8_stage_signature": "s" * 64,
    }
    (directory / "training_receipt.json").write_text(
        json.dumps(
            {
                "complete": True,
                "plan_sha256": freeze["v8_plan_sha256"],
                "stage_signature": freeze["v8_stage_signature"],
                "checkpoint": str(checkpoint.relative_to(tmp_path)),
                "checkpoint_sha256": checkpoint_hash,
            }
        )
    )
    output = "answer"
    bundle = {
        "policy": "project_theseus_direct_model_candidate_bundle_v1",
        "target_id": target,
        "case_contract_sha256": freeze["case_contract_sha256"],
        "candidate_packet_sha256": freeze["candidate_packet_sha256"],
        "generation_function": "moecot_language_arm_training.generate_model_text",
        "generation_wrapper_sha256": freeze["generation_wrapper_sha256"],
        "training_generator_sha256": freeze["training_generator_sha256"],
        "templates_renderers_routers_tools_credit": 0,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "timing": {
            "clock": "time.perf_counter",
            "checkpoint_load_duration_ms_by_target": {target: 5.0},
            "checkpoint_load_duration_ms_total": 5.0,
            "generation_duration_ms_total": 10.0,
            "wall_duration_ms": 15.0,
        },
        "checkpoint_artifacts": [
            {"target_id": target, "path": str(checkpoint.relative_to(tmp_path)), "sha256": checkpoint_hash}
        ],
        "candidates": [
            {
                "case_id": "case-1",
                "target_id": target,
                "output": output,
                "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                "generation_duration_ms": 10.0,
            }
        ],
    }
    cases = {"case-1": {"case_id": "case-1", "arm_id": "python"}}

    assert audit_candidate_provenance(bundle, freeze, cases, root=tmp_path)["state"] == "GREEN"

    mutated = copy.deepcopy(bundle)
    mutated["candidates"][0]["output"] = "changed"
    result = audit_candidate_provenance(mutated, freeze, cases, root=tmp_path)
    assert result["state"] == "RED"
    assert "candidate_output_identity_mismatch:case-1" in result["hard_gaps"]

    mutated_timing = copy.deepcopy(bundle)
    mutated_timing["candidates"][0]["generation_duration_ms"] = -1
    result = audit_candidate_provenance(mutated_timing, freeze, cases, root=tmp_path)
    assert result["state"] == "RED"
    assert "candidate_generation_timing_invalid:case-1" in result["hard_gaps"]
    assert "candidate_generation_timing_total_mismatch" in result["hard_gaps"]


def test_accepted_output_rate_includes_generation_verification_and_cold_load(monkeypatch) -> None:
    manifest = build_manifest(CONFIG, CONFIG_PATH)
    freeze = build_freeze(manifest, CONFIG_PATH)
    candidates = [
        {
            "case_id": case["case_id"],
            "target_id": "dense_active_parameter",
            "output": "candidate",
            "generation_duration_ms": 10.0,
        }
        for case in manifest["evaluator_cases"]
    ]
    bundle = {
        "target_id": "dense_active_parameter",
        "candidates": candidates,
        "timing": {
            "checkpoint_load_duration_ms_by_target": {
                "dense_active_parameter": 1000.0,
            }
        },
    }
    monkeypatch.setattr(
        "neural_seed_functional_utility.audit_candidate_provenance",
        lambda *_args, **_kwargs: {"state": "GREEN", "hard_gaps": []},
    )
    monkeypatch.setattr(
        "neural_seed_functional_utility.verify_candidate",
        lambda case, _output, _config: {
            "case_id": case["case_id"],
            "arm_id": case["arm_id"],
            "passed": case["arm_id"] != "english",
            "duration_ms": 20.0,
            "fault": "" if case["arm_id"] != "english" else "not_scored_here",
        },
    )

    result = evaluate_bundle(CONFIG, manifest, bundle, freeze, [])

    assert result["trigger_state"] == "YELLOW"
    assert result["summary"]["code_checkpoint_load_duration_ms"] == 1000.0
    assert result["summary"]["code_generation_duration_ms"] == 1280.0
    assert result["summary"]["code_verification_duration_ms"] == 2560.0
    assert result["summary"]["accepted_verified_output_per_second"] == 128 / 4.84
    assert result["summary"]["accepted_verified_output_per_second_warm"] == 128 / 3.84
