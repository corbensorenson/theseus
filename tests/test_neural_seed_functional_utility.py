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
    audit_local_english_judgments,
    build_blind_english_packet,
    build_freeze,
    build_manifest,
    compare_qualifications,
    evaluate_bundle,
    validate_freeze,
)
from neural_seed_functional_cases import stable_hash


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


def qualification(target: str, rates: dict[str, float], throughput: float) -> dict:
    return {
        "policy": "project_theseus_private_functional_utility_qualification_v1",
        "trigger_state": "GREEN",
        "evaluation_complete": True,
        "candidate_provenance": {
            "state": "GREEN",
            "bundle_target_id": target,
        },
        "by_arm": {
            arm: {
                "passed": int(round(rates[arm] * 32)),
                "scored": 32,
                "expected": 32,
                "functional_pass_rate": rates[arm],
            }
            for arm in CONFIG["arms"]
        },
        "summary": {
            "accepted_verified_output_per_second": throughput,
            "accepted_verified_output_per_second_warm": throughput,
        },
        "boundaries": {
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "templates_renderers_routers_tools_credit": 0,
        },
    }


def exact_diagnostic(freeze: dict) -> dict:
    return {
        "policy": "project_theseus_moecot_dense_exact_recovery_diagnostic_v8",
        "trigger_state": "GREEN",
        "publication_ready": True,
        "freeze_identity": {
            "functional_case_contract_sha256": freeze["case_contract_sha256"],
        },
        "boundaries": {
            "public_benchmark_payload_count": 0,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "templates_renderers_routers_tools_credit": 0,
        },
    }


def bind_freeze(rows: list[dict], freeze: dict) -> list[dict]:
    for row in rows:
        row["freeze_sha256"] = stable_hash(freeze)
    return rows


def test_architecture_verdict_falsifies_all_code_zero_before_pareto() -> None:
    manifest = build_manifest(CONFIG, CONFIG_PATH)
    freeze = build_freeze(manifest, CONFIG_PATH)
    rates = {arm: (0.25 if arm == "english" else 0.0) for arm in CONFIG["arms"]}
    rows = bind_freeze(
        [
            qualification("moecot_system", rates, 2.0),
            qualification("dense_active_parameter", rates, 3.0),
            qualification("dense_total_parameter", rates, 4.0),
        ],
        freeze,
    )

    result = compare_qualifications(CONFIG, rows, exact_diagnostic(freeze), freeze)

    assert result["trigger_state"] == "GREEN"
    assert result["decision"] == "FALSIFY_10_8M_ACTIVE_SCALE_RUNG"
    assert result["architecture_selected"] is False
    assert result["route_replacement_authorized"] is False


def test_architecture_verdict_requires_dense_confirmation_on_strict_pareto_gain() -> None:
    manifest = build_manifest(CONFIG, CONFIG_PATH)
    freeze = build_freeze(manifest, CONFIG_PATH)
    sparse_rates = {arm: 0.25 for arm in CONFIG["arms"]}
    dense_rates = {arm: 0.5 for arm in CONFIG["arms"]}
    rows = bind_freeze(
        [
            qualification("moecot_system", sparse_rates, 2.0),
            qualification("dense_active_parameter", dense_rates, 3.0),
            qualification("dense_total_parameter", sparse_rates, 1.0),
        ],
        freeze,
    )

    result = compare_qualifications(CONFIG, rows, exact_diagnostic(freeze), freeze)

    assert result["trigger_state"] == "GREEN"
    assert result["decision"] == "DENSE_HYBRID_CONFIRMATION_REQUIRED"
    assert result["pareto"]["dense_active_over_moecot"] is True
    assert result["route_replacement_authorized"] is False
    assert "rows" not in result["functional_results"]["dense_active_parameter"]


def test_architecture_verdict_rejects_incomplete_or_mismatched_evidence() -> None:
    manifest = build_manifest(CONFIG, CONFIG_PATH)
    freeze = build_freeze(manifest, CONFIG_PATH)
    rates = {arm: 0.25 for arm in CONFIG["arms"]}
    rows = bind_freeze(
        [
            qualification("moecot_system", rates, 2.0),
            qualification("dense_active_parameter", rates, 2.0),
            qualification("dense_total_parameter", rates, 2.0),
        ],
        freeze,
    )
    rows[1]["evaluation_complete"] = False
    rows[2]["freeze_sha256"] = "0" * 64

    result = compare_qualifications(CONFIG, rows, exact_diagnostic(freeze), freeze)

    assert result["trigger_state"] == "RED"
    assert result["decision"] == "INVALID_EVIDENCE"
    assert "qualification_incomplete:dense_active_parameter" in result["hard_gaps"]
    assert "qualification_freeze_mismatch:dense_total_parameter" in result["hard_gaps"]


def test_architecture_verdict_requires_repeatable_moecot_pareto_gain() -> None:
    manifest = build_manifest(CONFIG, CONFIG_PATH)
    freeze = build_freeze(manifest, CONFIG_PATH)
    sparse_rates = {arm: 0.5 for arm in CONFIG["arms"]}
    dense_rates = {arm: 0.25 for arm in CONFIG["arms"]}
    rows = bind_freeze(
        [
            qualification("moecot_system", sparse_rates, 4.0),
            qualification("dense_active_parameter", dense_rates, 3.0),
            qualification("dense_total_parameter", dense_rates, 2.0),
        ],
        freeze,
    )

    result = compare_qualifications(CONFIG, rows, exact_diagnostic(freeze), freeze)

    assert result["trigger_state"] == "GREEN"
    assert result["decision"] == "MOECOT_CONFIRMATION_REQUIRED"
    assert result["pareto"]["moecot_over_dense_active"] is True
    assert result["pareto"]["moecot_over_dense_total"] is True


def test_architecture_verdict_preserves_quality_cost_tradeoff_as_unresolved() -> None:
    manifest = build_manifest(CONFIG, CONFIG_PATH)
    freeze = build_freeze(manifest, CONFIG_PATH)
    sparse_rates = {arm: 0.5 for arm in CONFIG["arms"]}
    dense_rates = {arm: 0.25 for arm in CONFIG["arms"]}
    rows = bind_freeze(
        [
            qualification("moecot_system", sparse_rates, 1.0),
            qualification("dense_active_parameter", dense_rates, 4.0),
            qualification("dense_total_parameter", dense_rates, 3.0),
        ],
        freeze,
    )

    result = compare_qualifications(CONFIG, rows, exact_diagnostic(freeze), freeze)

    assert result["trigger_state"] == "GREEN"
    assert result["decision"] == "UNRESOLVED_CONFIRMATION_REQUIRED"
    assert not any(result["pareto"].values())


def test_local_english_judgments_require_bound_receipt_and_pinned_models(monkeypatch) -> None:
    manifest = build_manifest(CONFIG, CONFIG_PATH)
    freeze = build_freeze(manifest, CONFIG_PATH)
    bundle = {
        "candidates": [
            {"case_id": case["case_id"], "output": "candidate"}
            for case in manifest["evaluator_cases"]
        ]
    }
    monkeypatch.setattr(
        "neural_seed_functional_utility.audit_candidate_provenance",
        lambda *_args, **_kwargs: {"state": "GREEN", "hard_gaps": []},
    )
    blind = build_blind_english_packet(CONFIG, manifest, bundle, freeze)
    rater_config = json.loads(
        (ROOT / "configs/neural_seed_local_english_raters.json").read_text()
    )
    scores = {dimension: 3 for dimension in CONFIG["english_scoring"]["dimensions"]}
    judgments = [
        {
            "case_id": item["case_id"],
            "blind_item_id": item["blind_item_id"],
            "candidate_sha256": item["candidate_sha256"],
            "rater_id": card["rater_id"],
            "scores": scores,
        }
        for item in blind["items"]
        for card in rater_config["primary_raters"]
    ]
    identity = {"path": "reports/judgments.jsonl", "sha256": "j" * 64, "row_count": 64}
    receipt = {
        "policy": "project_theseus_local_blind_english_judgment_receipt_v1",
        "trigger_state": "GREEN",
        "config_sha256": freeze["local_english_rater_config_sha256"],
        "implementation_sha256": freeze["local_english_rater_implementation_sha256"],
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "judgments_admitted_to_training": False,
        "raw_model_responses_retained": False,
        "local_evaluator_inference_calls": 64,
        "model_receipts": [
            {
                "rater_id": card["rater_id"],
                "repo_id": card["repo_id"],
                "revision": card["revision"],
                "snapshot_identity": {"manifest_sha256": card["revision"]},
            }
            for card in rater_config["primary_raters"]
        ],
        "judgment_files": [
            {
                "label": "opaque_a",
                **identity,
                "blind_packet_contract_sha256": blind["packet_sha256"],
            }
        ],
    }

    audit = audit_local_english_judgments(
        CONFIG, manifest, bundle, freeze, judgments, receipt, identity, "opaque_a"
    )
    assert audit["state"] == "GREEN"

    tampered = copy.deepcopy(receipt)
    tampered["judgment_files"][0]["sha256"] = "0" * 64
    audit = audit_local_english_judgments(
        CONFIG, manifest, bundle, freeze, judgments, tampered, identity, "opaque_a"
    )
    assert audit["state"] == "RED"
    assert "local_judgment_file_identity_mismatch:sha256" in audit["hard_gaps"]
