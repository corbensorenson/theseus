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
        "checkpoint_artifacts": [
            {"target_id": target, "path": str(checkpoint.relative_to(tmp_path)), "sha256": checkpoint_hash}
        ],
        "candidates": [
            {"case_id": "case-1", "target_id": target, "output": output, "output_sha256": hashlib.sha256(output.encode()).hexdigest()}
        ],
    }
    cases = {"case-1": {"case_id": "case-1", "arm_id": "python"}}

    assert audit_candidate_provenance(bundle, freeze, cases, root=tmp_path)["state"] == "GREEN"

    mutated = copy.deepcopy(bundle)
    mutated["candidates"][0]["output"] = "changed"
    result = audit_candidate_provenance(mutated, freeze, cases, root=tmp_path)
    assert result["state"] == "RED"
    assert "candidate_output_identity_mismatch:case-1" in result["hard_gaps"]
