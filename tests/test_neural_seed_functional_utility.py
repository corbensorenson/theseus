from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from neural_seed_functional_utility import (
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
