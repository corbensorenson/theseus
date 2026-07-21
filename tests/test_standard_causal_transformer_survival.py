from __future__ import annotations

import json
import hashlib
import subprocess
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import semantic_ir
import standard_causal_transformer_survival as survival
import standard_causal_transformer_corpus as corpus
from standard_causal_transformer_model import (
    CausalTransformerConfig,
    build_model,
    parameter_count,
)
from standard_causal_transformer_survival import (
    EXECUTABLE_STATE_ROLES,
    beam_rank_score,
    balanced_binary_class_weights,
    balanced_categorical_class_weights,
    batched_beam_advance,
    build_data_model_scaling_contract,
    cache_arrays,
    causal_loss,
    completion_pool_target,
    canonical_model_signature,
    compare_attention_policy_canaries,
    compare_target_mode_canaries,
    encode_sft_training_examples,
    encode_model_source,
    executable_state_role_lookup,
    executable_state_token_roles,
    extend_target_vocab_for_mode,
    generation_prefix_complete,
    hierarchical_beam_rank_score,
    materialize_canonical_mixed_corpus_receipt,
    materialize_stage_only_receipt,
    assign_body_balanced_sampling_weights,
    normalized_sampling_probabilities,
    prepare_semantic_plan_labels,
    prepare_ordered_plan_sequences,
    publish_candidate_artifact,
    phase_target_positions,
    prune_active_beams,
    prune_complete_beams,
    render_visible_signature,
    scaling_contract_sha256,
    semantic_plan_feature_contract,
    semantic_plan_labels_for_body,
    semantic_plan_metrics_from_logits,
    semantic_stage_source,
    sequence_partition_audit,
    select_family_disjoint_eval,
    select_preference_train_rows,
    standalone_sft_contract_decision,
    source_token_offset,
    source_tokens,
    stage_materialization_lock,
    stage_signature,
    target_token_offset,
    training_heartbeat_record,
    training_target_tokens,
    training_callable_signature,
    training_targets_complete,
    validate_config,
    visible_eval_source,
)
from blind_information_flow_audit import audit_source
from candidate_integrity import recompute_candidate_integrity
from standard_causal_transformer_conditioning import (
    deranged_source_arrays,
    inspect_bindings as inspect_conditioning_bindings,
    publish_completed_checkpoint,
    validate_config as validate_conditioning_config,
)
from code_lm_private_verifier import evaluate_all_private_candidates


def test_pretrain_stage_and_optimizer_targets_are_distinct_and_legacy_safe() -> None:
    current = {
        "pretrain_stage_unique_positions": 422_334_331,
        "pretrain_optimizer_target_positions": 1_146_808_520,
    }
    assert survival.pretrain_stage_position_target(current) == 422_334_331
    assert survival.pretrain_optimizer_position_target(current) == 1_146_808_520
    legacy = {"pretrain_target_token_positions": 215_552_020}
    assert survival.pretrain_stage_position_target(legacy) == 215_552_020
    assert survival.pretrain_optimizer_position_target(legacy) == 215_552_020


def test_resumed_training_heartbeat_reports_phase_and_cumulative_positions() -> None:
    row = training_heartbeat_record(
        phase_name="pretrain",
        global_step=525,
        phase_step=24,
        phase_positions_consumed=184_750,
        phase_positions_requested=1_092_889_688,
        position_offset=3_845_232,
        position_target_total=1_096_734_920,
        latest_loss=2.3966214,
        elapsed_seconds=401.2349,
    )
    assert row["policy"] == "standard_causal_transformer_training_heartbeat_v2"
    assert row["phase_target_positions_consumed"] == 184_750
    assert row["phase_target_positions_requested"] == 1_092_889_688
    assert row["position_offset"] == 3_845_232
    assert row["target_positions_consumed"] == 4_029_982
    assert row["target_positions_requested"] == 1_096_734_920
    assert row["latest_loss"] == 2.396621


def test_training_heartbeat_rejects_impossible_cumulative_positions() -> None:
    with pytest.raises(ValueError, match="exceed target"):
        training_heartbeat_record(
            phase_name="pretrain",
            global_step=1,
            phase_step=1,
            phase_positions_consumed=11,
            phase_positions_requested=10,
            latest_loss=1.0,
            elapsed_seconds=1.0,
        )


def test_materialization_targets_may_exceed_but_not_weaken_scale_minima() -> None:
    config = json.loads(
        (ROOT / "configs" / "standard_causal_transformer_survival.json").read_text()
    )
    contract = config["data_model_scaling_contract"]
    minimums = corpus.minimum_category_targets(contract)
    targets = corpus.category_targets(contract)
    assert sum(minimums.values()) == contract["required_unique_positions"]
    assert sum(targets.values()) == contract["materialization_unique_positions"]
    assert all(targets[key] >= minimums[key] for key in targets)

    invalid = json.loads(json.dumps(contract))
    invalid["materialization_category_positions"]["html_css"] = 1
    with pytest.raises(ValueError, match="cannot fall below"):
        corpus.category_targets(invalid)


def test_capacity_identity_refresh_allows_only_nonmeasurement_config_changes(
    tmp_path: Path,
) -> None:
    index = tmp_path / "index.sqlite3"
    with sqlite3.connect(index) as connection:
        connection.execute("CREATE TABLE documents (category TEXT, digest TEXT)")
        connection.executemany(
            "INSERT INTO documents VALUES (?, ?)",
            [("english_broad", "a" * 64), ("python", "b" * 64)],
        )
    prior = {
        "canonical_corpus": {"policy": "fixture"},
        "tokenization": {},
        "sources": {},
        "evaluation": {},
        "training": {"pretrain_target_token_positions": 10},
    }
    current = json.loads(json.dumps(prior))
    current["training"] = {
        "pretrain_stage_unique_positions": 10,
        "pretrain_optimizer_target_positions": 20,
    }
    prior_path = tmp_path / "prior.json"
    current_path = tmp_path / "current.json"
    prior_path.write_text(json.dumps(prior), encoding="utf-8")
    current_path.write_text(json.dumps(current), encoding="utf-8")
    selected = [f"english_broad:{'a' * 64}", f"python:{'b' * 64}"]
    report_path = tmp_path / "capacity.json"
    report_path.write_text(
        json.dumps(
            {
                "policy": "project_theseus_admitted_index_exact_capacity_measurement_v1",
                "index": str(index),
                "index_sha256": survival.file_content_sha256(index),
                "selected_document_count": 2,
                "selected_document_digest": survival.sha("\n".join(selected)),
                "config_sha256": survival.file_content_sha256(prior_path),
                "positions_by_category": {},
                "total_unique_positions": 10,
            }
        ),
        encoding="utf-8",
    )
    migrated = survival.refresh_capacity_identity(
        report_path,
        current_config_path=current_path,
        prior_config_path=prior_path,
    )
    assert migrated["config_sha256"] == survival.file_content_sha256(current_path)
    assert migrated["identity_migration"]["measurement_contract_unchanged"] is True
    assert migrated["identity_migration"]["exact_index_bytes_replayed"] is True
    assert migrated["identity_migration"]["indexed_document_inventory_replayed"] is True
    assert migrated["identity_migration"]["selected_documents_recomputed"] is False
    assert migrated["identity_migration"]["position_counts_recomputed"] is False

    changed = json.loads(json.dumps(current))
    changed["canonical_corpus"]["policy"] = "changed"
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="full remeasurement"):
        survival.refresh_capacity_identity(
            report_path,
            current_config_path=changed_path,
            prior_config_path=prior_path,
        )
from standard_causal_transformer_preference import (
    PreferenceArrays,
    build_preference_pairs,
    reward_removed_pairs,
    train_dpo,
)
from standard_causal_transformer_survival_gate import (
    audit_generation_mode_canary,
    audit_latent_ordered_plan_ablation,
    audit_ordered_plan_ablation,
    audit_preference_canary,
    audit_semantic_plan_head_ablation,
    audit_slot_ordered_plan_ablation,
    audit_sft_contract_admission,
    audit_state_memory_ablation,
    audit_state_memory_continuation,
    is_scaling_readiness_gap,
    scaling_shortfall_summary,
    audit_teacher_residual_ablation,
    audit_target_mode_comparison,
    evaluation_replay_is_content_bound,
)
from generation_mode_gate import audit_comparison, read_report_ref
from policy_optimization_gate import extract_behavior_metrics, summarize_behavior_evidence
from code_lm_decoder_contracts import visible_arg_count_hint_for_task
from broad_private_generalization_ladder_v1 import row_from_template, template_bank


def test_stage_only_receipt_materializes_without_model_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.json"
    config = {
        "training": {"pretrain_target_token_positions": 8},
    }
    config_path.write_text(json.dumps(config))
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    metadata = stage_dir / "stage_metadata_v1.json"
    metadata.write_text('{"content_bound":true}')
    canonical = {
        "policy": "project_theseus_canonical_balanced_pretrain_stage_v1",
        "target_positions": 8,
        "window_count": 2,
        "max_sequence_tokens": 4,
        "non_overlapping_windows": True,
        "category_positions": {"english_broad": 8},
        "arm_views": {"hidden_generalist_fallback": "forbidden"},
        "array_artifacts": {"inputs": {"sha256": "a" * 64}},
        "index": {"sha256": "b" * 64},
        "tokenizer_audit": {"roundtrip_failure_count": 0},
    }
    fake_stage = SimpleNamespace(
        summary={
            "stage_signature": "stage-v2",
            "cache_status": "miss",
            "canonical_pretrain_stage": canonical,
            "public_training_rows": 0,
            "external_inference_calls": 0,
        }
    )
    monkeypatch.setattr(survival, "validate_config", lambda _config: None)
    monkeypatch.setattr(
        survival,
        "build_data_model_scaling_contract",
        lambda _config: {
            "training_authorized": True,
            "hard_gaps": [],
            "selected_rung": {"id": "isolated"},
            "required_unique_positions": 8,
        },
    )
    monkeypatch.setattr(survival, "materialize_stage", lambda *_args, **_kwargs: fake_stage)

    receipt = materialize_stage_only_receipt(
        config,
        config_path=str(config_path),
        stage_dir=stage_dir,
        force=False,
    )

    assert receipt["trigger_state"] == "GREEN"
    assert receipt["model_training_performed"] is False
    assert receipt["capability_credit"] == "NONE"
    assert receipt["canonical_pretrain_stage"]["target_positions"] == 8

    canonical["target_positions"] = 7
    rejected = materialize_stage_only_receipt(
        config,
        config_path=str(config_path),
        stage_dir=stage_dir,
        force=False,
    )
    assert rejected["trigger_state"] == "RED"
    assert rejected["hard_gaps"] == ["materialized_position_count_mismatch"]


def test_dry_run_cannot_overwrite_candidate_evidence(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    sentinel = '{"candidate_id":"retained"}\n'
    path.write_text(sentinel)

    assert publish_candidate_artifact(execute=False, path=path, candidates=[]) is False
    assert path.read_text() == sentinel

    replacement = [{"candidate_id": "executed"}]
    assert publish_candidate_artifact(
        execute=True, path=path, candidates=replacement
    ) is True
    assert [json.loads(line) for line in path.read_text().splitlines()] == replacement


def test_frozen_scaling_contract_reports_noncanonical_shortfall_without_authorizing_training(tmp_path: Path) -> None:
    config = json.loads((ROOT / "configs" / "standard_causal_transformer_survival.json").read_text())
    config["data_model_scaling_contract"]["canonical_corpus_receipt"] = {
        "path": str(tmp_path / "absent-canonical-corpus.json")
    }

    contract = build_data_model_scaling_contract(config)

    assert contract["training_authorized"] is False
    valid_planning_positions = sum(
        row["declared_positions"]
        for row in contract["planning_receipts"]
        if row["valid_planning_receipt"]
    )
    assert contract["planning_estimate_positions"] == valid_planning_positions
    active_parameters = contract["selected_rung"]["active_parameter_count"]
    assert contract["planning_estimate_tokens_per_active_parameter"] == round(
        valid_planning_positions / active_parameters, 6
    )
    assert contract["planning_estimate_shortfall_positions"] == (
        contract["required_unique_positions"] - valid_planning_positions
    )
    assert contract["planning_estimate_is_training_authority"] is False
    assert "canonical_mixed_corpus_receipt_missing" in contract["hard_gaps"]


def test_scaling_gate_separates_readiness_from_integrity_and_reports_canonical_shortfall() -> None:
    assert is_scaling_readiness_gap("canonical_unique_position_floor_not_met")
    assert is_scaling_readiness_gap("domain_minimum_not_met:code_total")
    assert not is_scaling_readiness_gap("canonical_corpus_contract_identity_mismatch")
    summary = scaling_shortfall_summary(
        {
            "required_unique_positions": 100,
            "planning_estimate_shortfall_positions": 90,
            "canonical_corpus_receipt": {
                "content_bound": True,
                "unique_model_visible_positions": 60,
            },
        }
    )
    assert summary == {
        "canonical_shortfall_positions": 40,
        "planning_estimate_shortfall_positions": 90,
    }


def test_canonical_scaling_receipt_requires_all_dimensions_and_rejects_repetition_inflation(tmp_path: Path) -> None:
    config = json.loads((ROOT / "configs" / "standard_causal_transformer_survival.json").read_text())
    scaling = config["data_model_scaling_contract"]
    required = scaling["required_unique_positions"]
    planning_path = tmp_path / "planning.json"
    planning_path.write_text(
        json.dumps(
            {
                "trigger_state": "GREEN",
                "summary": {
                    "checkpoint_sha256": "c" * 64,
                    "data_exposure": {"one_pass_total_token_positions": 1},
                },
            }
        )
    )
    scaling["planning_receipts"] = [
        {
            "id": "isolated_test_planning_receipt",
            "domain": "code_test",
            "path": str(planning_path),
            "one_pass_positions": 1,
            "accounting_abi": "isolated_test_v1",
            "canonical_accounting": False,
        }
    ]
    receipt_path = tmp_path / "canonical-corpus.json"
    receipt = {
        "policy": "project_theseus_canonical_mixed_corpus_receipt_v1",
        "trigger_state": "GREEN",
        "summary": {
            "tokenizer_abi": scaling["selected_rung"]["tokenizer_abi"],
            "active_parameter_count": scaling["selected_rung"]["active_parameter_count"],
            "unique_model_visible_positions": required,
            "optimizer_token_positions": required * 4,
            "optimizer_repetition_counted_as_unique_data": False,
            "domain_unique_positions": {
                **scaling["domain_minimum_positions"],
                **scaling["subset_minimum_positions"],
            },
            "code_language_unique_positions": scaling["code_language_minimum_positions"],
            "evidence_dimensions": {key: True for key in scaling["required_evidence_dimensions"]},
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "source_content_identity_verified": True,
            "source_manifest_digest": "a" * 64,
            "contract_sha256": scaling_contract_sha256(scaling),
            "language_scope": {
                "natural_languages": ["en"],
                "programming_languages": [
                    "python",
                    "javascript_typescript",
                    "html_css",
                    "rust",
                ],
                "non_allowed_action": "quarantine",
            },
            "code_quality_policy": {
                "policy": "project_theseus_curated_code_quality_v1",
                "curated_repo_config_sha256": "b" * 64,
                "curated_repo_count": 1,
            },
        },
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }

    def bind(payload: dict) -> None:
        identity_keys = (
            "tokenizer_abi",
            "active_parameter_count",
            "unique_model_visible_positions",
            "domain_unique_positions",
            "code_language_unique_positions",
            "evidence_dimensions",
            "source_manifest_digest",
            "contract_sha256",
            "language_scope",
            "code_quality_policy",
        )
        payload["identity_payload"] = {key: payload["summary"][key] for key in identity_keys}
        payload["receipt_identity_sha256"] = hashlib.sha256(
            json.dumps(payload["identity_payload"], sort_keys=True).encode()
        ).hexdigest()
        receipt_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        scaling["canonical_corpus_receipt"] = {
            "path": str(receipt_path),
        }

    bind(receipt)
    accepted = build_data_model_scaling_contract(config)
    assert accepted["training_authorized"] is True, accepted["hard_gaps"]
    assert accepted["canonical_corpus_receipt"]["optimizer_repetition_factor"] == 4.0

    receipt["summary"]["optimizer_token_positions"] = required * 4 + 1
    receipt["summary"]["evidence_dimensions"]["semantic_deduplication"] = False
    bind(receipt)
    rejected = build_data_model_scaling_contract(config)
    assert rejected["training_authorized"] is False
    gaps = rejected["canonical_corpus_receipt"]["hard_gaps"]
    assert "optimizer_repetition_above_predeclared_maximum" in gaps
    assert any(gap.startswith("required_evidence_dimensions_missing:") for gap in gaps)


def canonical_corpus_fixture(tmp_path: Path) -> tuple[dict, Path, Path]:
    config = json.loads((ROOT / "configs" / "standard_causal_transformer_survival.json").read_text())
    code_path = tmp_path / "sample.py"
    code_path.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    code_manifest = tmp_path / "code-manifest.json"
    code_manifest.write_text(
        json.dumps(
            {
                "policy": "project_theseus_narrow_corpus_manifest_ladder_v1",
                "sources": [
                    {
                        "admitted": True,
                        "license_allowed": True,
                        "public_benchmark_payload_detected": False,
                        "eval_overlap_detected": False,
                        "path": str(code_path),
                        "sha256": hashlib.sha256(code_path.read_bytes()).hexdigest(),
                        "content_type": "code_python",
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    conversation_root = tmp_path / "conversation"
    shard_path = conversation_root / "private_train" / "shard.jsonl"
    shard_path.parent.mkdir(parents=True)
    valid_row = {
        "causal_text": "user: hello\nassistant: hello there",
        "target_message": {"role": "assistant", "content": "hello there"},
        "license_spdx": "apache-2.0",
        "data_admission_receipt_id": "receipt-1",
        "public_benchmark": False,
        "external_inference_calls": 0,
    }
    public_row = dict(valid_row, public_benchmark=True, data_admission_receipt_id="receipt-public")
    shard_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in (valid_row, valid_row, public_row)) + "\n",
        encoding="utf-8",
    )
    conversation_manifest = conversation_root / "manifest.json"
    conversation_manifest.write_text(
        json.dumps(
            {
                "policy": "project_theseus_governed_conversation_stream_state_v1",
                "shards": [
                    {
                        "train_path": "private_train/shard.jsonl",
                        "train_sha256": hashlib.sha256(shard_path.read_bytes()).hexdigest(),
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    broad_text_root = tmp_path / "broad-text"
    broad_text_root.mkdir()
    broad_text_manifest = broad_text_root / "manifest.json"
    broad_text_manifest.write_text(
        json.dumps(
            {
                "policy": "project_theseus_governed_document_stream_state_v1",
                "shards": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    intake_policy = tmp_path / "english-intake-policy.json"
    intake_policy.write_text(
        json.dumps(
            {
                "sources": [
                    {"id": "conversation", "enabled": True, "required_langs": ["en"]}
                ],
                "broad_text_sources": [
                    {"id": "documents", "enabled": True, "required_language": "en"}
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    repo_policy = tmp_path / "repo-policy.json"
    repo_policy.write_text(
        json.dumps({"repos": [{"repo": "example/project"}]}, sort_keys=True),
        encoding="utf-8",
    )
    config["canonical_corpus"] = {
        "policy": "project_theseus_canonical_mixed_corpus_materialization_v1",
        "natural_language_scope": {
            "allowed_languages": ["en"],
            "non_allowed_action": "quarantine",
            "intake_policy": str(intake_policy),
        },
        "programming_language_scope": [
            "python",
            "javascript_typescript",
            "html_css",
            "rust",
        ],
        "code_quality_policy": {
            "policy": "project_theseus_curated_code_quality_v1",
            "curated_repo_config": str(repo_policy),
            "minimum_logical_tokens": 1,
            "maximum_line_characters": 2000,
            "maximum_mean_nonempty_line_characters": 300,
            "minimum_unique_token_ratio": 0.01,
            "excluded_path_parts": ["vendor"],
            "excluded_name_markers": [".min."],
            "generated_text_markers": ["do not edit"],
        },
        "code_manifests": [str(code_manifest)],
        "conversation_manifest": str(conversation_manifest),
        "conversation_root": str(conversation_root),
        "broad_text_manifest": str(broad_text_manifest),
        "broad_text_root": str(broad_text_root),
        "near_duplicate_hamming_distance": 3,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    return config, code_manifest, shard_path


def test_canonical_corpus_materializer_counts_only_content_bound_unique_governed_rows(tmp_path: Path) -> None:
    config, _, _ = canonical_corpus_fixture(tmp_path)

    receipt = materialize_canonical_mixed_corpus_receipt(config)

    assert receipt["hard_gaps"] == []
    assert receipt["summary"]["document_counts"] == {
        "code_total": 1,
        "english_natural_language_total": 1,
        "english_conversation_instruction": 1,
    }
    assert receipt["summary"]["exact_duplicate_counts"] == {
        "english_natural_language_total": 1,
    }
    assert receipt["summary"]["excluded_counts"] == {
        "conversation_governance_or_completeness": 1,
    }
    assert receipt["summary"]["source_content_identity_verified"] is True
    assert receipt["summary"]["public_training_rows_written"] == 0
    assert receipt["summary"]["external_inference_calls"] == 0
    assert receipt["summary"]["fallback_return_count"] == 0
    assert receipt["trigger_state"] == "YELLOW"


def test_canonical_corpus_materializer_excludes_stale_code_without_position_credit(tmp_path: Path) -> None:
    config, code_manifest, _ = canonical_corpus_fixture(tmp_path)
    manifest = json.loads(code_manifest.read_text())
    manifest["sources"][0]["sha256"] = "0" * 64
    code_manifest.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    receipt = materialize_canonical_mixed_corpus_receipt(config)

    assert receipt["hard_gaps"] == []
    assert receipt["summary"]["document_counts"].get("code_total", 0) == 0
    assert receipt["summary"]["code_language_unique_positions"].get("python", 0) == 0
    assert receipt["summary"]["excluded_counts"]["code_source_identity_mismatch"] == 1


def test_canonical_corpus_materializer_excludes_incomplete_code_but_keeps_retained_completeness(tmp_path: Path) -> None:
    config, code_manifest, _ = canonical_corpus_fixture(tmp_path)
    invalid_path = tmp_path / "invalid.py"
    invalid_path.write_text("def broken(:\n", encoding="utf-8")
    manifest = json.loads(code_manifest.read_text())
    manifest["sources"].append(
        {
            "admitted": True,
            "license_allowed": True,
            "public_benchmark_payload_detected": False,
            "eval_overlap_detected": False,
            "path": str(invalid_path),
            "sha256": hashlib.sha256(invalid_path.read_bytes()).hexdigest(),
            "content_type": "code_python",
        }
    )
    code_manifest.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    receipt = materialize_canonical_mixed_corpus_receipt(config)

    assert receipt["summary"]["document_counts"]["code_total"] == 1
    assert receipt["summary"]["excluded_counts"]["code_incomplete"] == 1
    assert receipt["summary"]["evidence_dimensions"]["executable_or_dialogue_completeness"] is True


def test_canonical_corpus_materializer_rejects_stale_conversation_shard(tmp_path: Path) -> None:
    config, _, shard_path = canonical_corpus_fixture(tmp_path)
    shard_path.write_text(shard_path.read_text() + "{}\n", encoding="utf-8")

    receipt = materialize_canonical_mixed_corpus_receipt(config)

    assert receipt["trigger_state"] == "RED"
    assert any(gap.startswith("conversation_shard_identity_mismatch:") for gap in receipt["hard_gaps"])
    assert receipt["summary"]["document_counts"].get("english_conversation_instruction", 0) == 0


def test_canonical_corpus_materializer_counts_content_bound_multilingual_code_shard(tmp_path: Path) -> None:
    config, _, _ = canonical_corpus_fixture(tmp_path)
    rows = [
        ("web/app.js", "javascript", "export function add(a, b) { return a + b; }"),
        ("web/view.tsx", "typescript", "export const Label = (p: {text: string}) => <span>{p.text}</span>;"),
        ("web/index.html", "html", "<main><h1>Theseus</h1></main>"),
        ("web/style.css", "css", "main { display: grid; gap: 1rem; }"),
        ("src/lib.rs", "rust", "pub fn add(a: i64, b: i64) -> i64 { a + b }"),
    ]
    shard_path = tmp_path / "code-shard.jsonl"
    shard_rows = []
    for path, language, text in rows:
        shard_rows.append(
            {
                "repo": "example/project",
                "path": path,
                "language": language,
                "license_spdx": "MIT",
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "public_benchmark": False,
                "public_benchmark_solutions_included": False,
                "public_tests_included": False,
                "benchmark_excluded": True,
            }
        )
    shard_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in shard_rows) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "code-shard-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "policy": "project_theseus_open_code_canonical_shard_manifest_v1",
                "sample_jsonl": shard_path.name,
                "sample_jsonl_sha256": hashlib.sha256(shard_path.read_bytes()).hexdigest(),
                "allowed_licenses": ["MIT"],
                "admitted_sources": [{"repo": "example/project", "license_spdx": "MIT"}],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    config["canonical_corpus"]["code_shard_manifests"] = [str(manifest_path)]

    receipt = materialize_canonical_mixed_corpus_receipt(config)

    positions = receipt["summary"]["code_language_unique_positions"]
    assert positions["javascript_typescript"] > 0
    assert positions["html_css"] > 0
    assert positions["rust"] > 0
    assert receipt["hard_gaps"] == []

    shard_path.write_text(shard_path.read_text() + "{}\n", encoding="utf-8")
    tampered = materialize_canonical_mixed_corpus_receipt(config)
    assert tampered["trigger_state"] == "RED"
    assert any(gap.startswith("code_shard_identity_mismatch:") for gap in tampered["hard_gaps"])


def test_decoder_is_causal_and_cached_decode_matches_full_decode() -> None:
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(11)
    config = CausalTransformerConfig(
        vocab_size=64,
        d_model=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ff_dim=64,
    )
    model = build_model(config, mx=mx, nn=nn)
    prefix_a = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    prefix_b = mx.array([[1, 2, 3, 9]], dtype=mx.int32)
    logits_a, _ = model(prefix_a)
    logits_b, _ = model(prefix_b)
    mx.eval(logits_a, logits_b)
    assert bool(mx.allclose(logits_a[:, :3], logits_b[:, :3], atol=1e-5))

    prefill, cache = model(prefix_a[:, :3])
    cached, _ = model(prefix_a[:, 3:], cache)
    full, _ = model(prefix_a)
    mx.eval(prefill, cached, full)
    assert bool(mx.allclose(cached[:, -1], full[:, -1], atol=1e-4))


def test_attention_policy_configuration_fails_closed() -> None:
    common = {
        "vocab_size": 64,
        "d_model": 32,
        "num_layers": 2,
        "num_heads": 4,
        "num_kv_heads": 2,
        "ff_dim": 64,
    }
    with pytest.raises(ValueError, match="attention policy"):
        CausalTransformerConfig(**common, attention_policy="bidirectional").validate()
    with pytest.raises(ValueError, match="separator token"):
        CausalTransformerConfig(
            **common, source_target_separator_token_id=64
        ).validate()
    with pytest.raises(ValueError, match="prefix-LM attention"):
        CausalTransformerConfig(
            **common,
            attention_policy="prefix_lm",
            state_memory_mode="semantic_roles",
            state_memory_slots=8,
        ).validate()


def test_mlx_native_grouped_query_attention_matches_explicit_kv_tiling() -> None:
    import mlx.core as mx

    mx.random.seed(20260720)
    query = mx.random.normal((2, 8, 7, 16))
    key = mx.random.normal((2, 2, 7, 16))
    value = mx.random.normal((2, 2, 7, 16))
    native = mx.fast.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=16 ** -0.5,
        mask="causal",
    )
    explicit = mx.fast.scaled_dot_product_attention(
        query,
        mx.repeat(key, repeats=4, axis=1),
        mx.repeat(value, repeats=4, axis=1),
        scale=16 ** -0.5,
        mask="causal",
    )
    mx.eval(native, explicit)

    assert tuple(native.shape) == tuple(explicit.shape) == (2, 8, 7, 16)
    assert bool(mx.allclose(native, explicit, atol=1e-5, rtol=1e-5))


def test_prefix_lm_mask_is_bidirectional_only_inside_source_partition() -> None:
    import mlx.core as mx
    import mlx.nn as nn

    config = CausalTransformerConfig(
        vocab_size=64,
        d_model=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        ff_dim=64,
        attention_policy="prefix_lm",
    )
    model = build_model(config, mx=mx, nn=nn)
    mask = np.asarray(
        model.sequence_attention_mask(
            mx.array([[1, 10, 11, 2, 20, 21]], dtype=mx.int32), None
        )
    )[0, 0]
    assert np.all(mask[:4, :4] == 0.0)
    assert np.all(mask[:4, 4:] < -1e8)
    assert np.all(mask[4, :5] == 0.0)
    assert mask[4, 5] < -1e8
    assert np.all(mask[5] == 0.0)

    no_separator = np.asarray(
        model.sequence_attention_mask(
            mx.array([[1, 10, 11, 12]], dtype=mx.int32), None
        )
    )[0, 0]
    assert np.all(no_separator[np.tril_indices(4)] == 0.0)
    assert np.all(no_separator[np.triu_indices(4, 1)] < -1e8)


def test_prefix_lm_has_source_lookahead_without_target_leakage() -> None:
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(29)
    config = CausalTransformerConfig(
        vocab_size=64,
        d_model=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ff_dim=64,
        attention_policy="prefix_lm",
    )
    model = build_model(config, mx=mx, nn=nn)
    base = mx.array([[1, 10, 11, 2, 20, 21]], dtype=mx.int32)
    later_source_changed = mx.array([[1, 10, 19, 2, 20, 21]], dtype=mx.int32)
    future_target_changed = mx.array([[1, 10, 11, 2, 20, 31]], dtype=mx.int32)
    base_logits, _ = model(base)
    source_logits, _ = model(later_source_changed)
    target_logits, _ = model(future_target_changed)
    mx.eval(base_logits, source_logits, target_logits)
    assert not bool(mx.allclose(base_logits[:, :2], source_logits[:, :2], atol=1e-6))
    assert bool(mx.allclose(base_logits[:, :5], target_logits[:, :5], atol=1e-6))


def test_prefix_lm_cache_partition_matches_full_sequence() -> None:
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(31)
    config = CausalTransformerConfig(
        vocab_size=64,
        d_model=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ff_dim=64,
        attention_policy="prefix_lm",
    )
    model = build_model(config, mx=mx, nn=nn)
    sequence = mx.array([[1, 10, 11, 2, 20, 21, 22]], dtype=mx.int32)
    full_logits, full_cache = model(sequence)
    _prefill_logits, prefill_cache = model(sequence[:, :5])
    cached_logits, cached_state = model(sequence[:, 5:], prefill_cache)
    mx.eval(
        full_logits,
        cached_logits,
        *cache_arrays(full_cache),
        *cache_arrays(cached_state),
    )
    assert bool(mx.allclose(cached_logits, full_logits[:, 5:], atol=1e-4))


def test_prefix_lm_is_parameter_neutral_and_loads_causal_checkpoint(
    tmp_path: Path,
) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.utils as mlx_utils

    common = {
        "vocab_size": 64,
        "d_model": 32,
        "num_layers": 2,
        "num_heads": 4,
        "num_kv_heads": 2,
        "ff_dim": 64,
    }
    mx.random.seed(37)
    causal = build_model(CausalTransformerConfig(**common), mx=mx, nn=nn)
    prefix = build_model(
        CausalTransformerConfig(**common, attention_policy="prefix_lm"),
        mx=mx,
        nn=nn,
    )
    assert parameter_count(causal, mlx_utils) == parameter_count(prefix, mlx_utils)
    checkpoint = tmp_path / "causal_checkpoint.npz"
    causal.save_weights(str(checkpoint))
    prefix.load_weights(str(checkpoint))
    no_separator = mx.array([[1, 10, 11, 12]], dtype=mx.int32)
    causal_logits, _ = causal(no_separator)
    prefix_logits, _ = prefix(no_separator)
    mx.eval(causal_logits, prefix_logits)
    assert bool(mx.allclose(causal_logits, prefix_logits, atol=1e-6))


def test_encoder_decoder_configuration_fails_closed() -> None:
    common = {
        "vocab_size": 64,
        "d_model": 32,
        "num_layers": 2,
        "num_heads": 4,
        "num_kv_heads": 2,
        "ff_dim": 64,
    }
    with pytest.raises(ValueError, match="requires source encoder"):
        CausalTransformerConfig(
            **common, attention_policy="encoder_decoder"
        ).validate()
    with pytest.raises(ValueError, match="require encoder-decoder"):
        CausalTransformerConfig(**common, source_encoder_layers=1).validate()
    with pytest.raises(ValueError, match="not yet compatible"):
        CausalTransformerConfig(
            **common,
            attention_policy="encoder_decoder",
            source_encoder_layers=1,
            state_memory_mode="semantic_roles",
            state_memory_slots=8,
        ).validate()
    with pytest.raises(ValueError, match="source copying requires"):
        CausalTransformerConfig(
            **common, source_copy_mode="pointer_generator"
        ).validate()


def test_pointer_generator_requires_lookup_and_only_copies_source_partition() -> None:
    import mlx.core as mx
    import mlx.nn as nn

    config = CausalTransformerConfig(
        vocab_size=64,
        d_model=16,
        num_layers=1,
        num_heads=4,
        num_kv_heads=1,
        ff_dim=32,
        attention_policy="encoder_decoder",
        source_encoder_layers=1,
        source_copy_mode="pointer_generator",
    )
    with pytest.raises(ValueError, match="source-to-target lookup"):
        build_model(config, mx=mx, nn=nn)
    lookup = np.full(64, -1, dtype=np.int32)
    lookup[10] = 40
    lookup[11] = 41
    model = build_model(
        config, mx=mx, nn=nn, source_to_target_lookup=lookup
    )
    model.copy_gate.weight = mx.zeros_like(model.copy_gate.weight)
    model.copy_gate.bias = mx.full(model.copy_gate.bias.shape, -20.0)
    tokens = mx.array([[1, 10, 11, 2, 20]], dtype=mx.int32)
    logits, cache = model(tokens)
    changed_target = mx.array([[1, 10, 11, 2, 33]], dtype=mx.int32)
    changed_logits, _ = model(changed_target)
    mx.eval(logits, changed_logits, *cache_arrays(cache))
    assert float(logits[0, -1, 40].item()) > float(logits[0, -1, 42].item())
    assert float(logits[0, -1, 41].item()) > float(logits[0, -1, 42].item())
    assert bool(mx.allclose(logits[:, :4], changed_logits[:, :4], atol=1e-6))


def test_kerc_configuration_requires_complete_modular_architecture() -> None:
    common = {
        "vocab_size": 96,
        "d_model": 32,
        "num_layers": 1,
        "num_heads": 4,
        "num_kv_heads": 2,
        "ff_dim": 64,
        "attention_policy": "encoder_decoder",
        "source_encoder_layers": 1,
        "source_copy_mode": "pointer_generator",
        "kerc_surface_token_start": 20,
        "kerc_surface_token_end": 40,
        "kerc_kernel_token_start": 40,
        "kerc_kernel_token_end": 60,
        "kerc_pointer_token_start": 60,
        "kerc_pointer_token_end": 90,
        "kerc_end_token_id": 3,
    }
    with pytest.raises(ValueError, match="four distinct"):
        CausalTransformerConfig(
            **common,
            kerc_task_token_ids=(10, 10, 11, 12),
            kerc_stage_adapter_dim=8,
            kerc_residual_choice_count=4,
            kerc_residual_bottleneck_dim=8,
            kerc_verifier_dim=8,
            kerc_decision_bottleneck_dim=8,
            kerc_decision_output_dim=4,
        ).validate()
    with pytest.raises(ValueError, match="copy-aware"):
        CausalTransformerConfig(
            **{**common, "source_copy_mode": "none"},
            kerc_task_token_ids=(10, 11, 12, 13),
            kerc_stage_adapter_dim=8,
            kerc_residual_choice_count=4,
            kerc_residual_bottleneck_dim=8,
            kerc_verifier_dim=8,
            kerc_decision_bottleneck_dim=8,
            kerc_decision_output_dim=4,
        ).validate()
    with pytest.raises(ValueError, match="trusted task tokens"):
        CausalTransformerConfig(**common, kerc_stage_adapter_dim=8).validate()


def test_kerc_per_unit_allocator_is_content_sensitive_and_hard_constrained() -> None:
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(71)
    config = CausalTransformerConfig(
        vocab_size=96,
        d_model=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        ff_dim=64,
        attention_policy="encoder_decoder",
        source_encoder_layers=1,
        source_copy_mode="pointer_generator",
        kerc_task_token_ids=(10, 11, 12, 13),
        kerc_stage_adapter_dim=8,
        kerc_residual_choice_count=4,
        kerc_residual_bottleneck_dim=8,
        kerc_residual_unit_kind_count=5,
        kerc_residual_unit_feature_dim=18,
        kerc_residual_unit_byte_vocab_size=257,
        kerc_verifier_dim=8,
        kerc_verifier_output_dim=5,
        kerc_decision_bottleneck_dim=8,
        kerc_decision_output_dim=4,
        kerc_surface_token_start=20,
        kerc_surface_token_end=40,
        kerc_kernel_token_start=40,
        kerc_kernel_token_end=60,
        kerc_pointer_token_start=60,
        kerc_pointer_token_end=90,
        kerc_end_token_id=3,
    )
    model = build_model(
        config,
        mx=mx,
        nn=nn,
        source_to_target_lookup=np.arange(96, dtype=np.int32),
    )
    unit_bytes = mx.array([[[1, 2, 3, 256], [4, 5, 256, 256]]], dtype=mx.int32)
    byte_mask = mx.array([[[1, 1, 1, 0], [1, 1, 0, 0]]], dtype=mx.float32)
    kinds = mx.array([[0, 4]], dtype=mx.int32)
    features = mx.zeros((1, 2, 4, 18), dtype=mx.float32)
    unit_mask = mx.ones((1, 2), dtype=mx.float32)
    hard = mx.array(
        [[[0, 0, 0, 0], [1, 1, 1, 0]]], dtype=mx.bool_
    )
    first = model.kerc_allocate_units(
        unit_byte_ids=unit_bytes,
        unit_byte_mask=byte_mask,
        unit_kind_ids=kinds,
        unit_candidate_features=features,
        unit_mask=unit_mask,
        unit_hard_block_mask=hard,
        source_summary=mx.zeros((1, 32), dtype=mx.float32),
    )
    ragged = model.kerc_allocate_units(
        unit_byte_ids=mx.array([1, 2, 3, 4, 5], dtype=mx.int32),
        unit_byte_mask=None,
        unit_byte_offsets=mx.array([[[0, 3], [3, 5]]], dtype=mx.int64),
        unit_kind_ids=kinds,
        unit_candidate_features=features,
        unit_mask=unit_mask,
        unit_hard_block_mask=hard,
        source_summary=mx.zeros((1, 32), dtype=mx.float32),
    )
    changed_bytes = mx.array([[[1, 9, 3, 256], [4, 5, 256, 256]]], dtype=mx.int32)
    second = model.kerc_allocate_units(
        unit_byte_ids=changed_bytes,
        unit_byte_mask=byte_mask,
        unit_kind_ids=kinds,
        unit_candidate_features=features,
        unit_mask=unit_mask,
        unit_hard_block_mask=hard,
        source_summary=mx.zeros((1, 32), dtype=mx.float32),
    )
    mx.eval(
        first["logits"],
        first["confidence_logits"],
        first["residual_context"],
        ragged["logits"],
        second["logits"],
    )
    assert tuple(first["logits"].shape) == (1, 2, 4)
    assert tuple(first["confidence_logits"].shape) == (1, 2)
    assert float(first["logits"][0, 1, 3].item()) > -1e8
    assert all(float(first["logits"][0, 1, index].item()) < -1e8 for index in range(3))
    assert bool(mx.allclose(first["logits"], ragged["logits"], atol=1e-5))
    assert not bool(mx.allclose(first["logits"][0, 0], second["logits"][0, 0]))


def test_kerc_per_unit_allocator_receives_gradient_without_legacy_channel_loss() -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils as mlx_utils

    mx.random.seed(73)
    config = CausalTransformerConfig(
        vocab_size=96,
        d_model=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        ff_dim=64,
        attention_policy="encoder_decoder",
        source_encoder_layers=1,
        source_copy_mode="pointer_generator",
        kerc_task_token_ids=(10, 11, 12, 13),
        kerc_stage_adapter_dim=8,
        kerc_residual_choice_count=4,
        kerc_residual_bottleneck_dim=8,
        kerc_residual_unit_kind_count=5,
        kerc_residual_unit_feature_dim=18,
        kerc_residual_unit_byte_vocab_size=257,
        kerc_verifier_dim=8,
        kerc_verifier_output_dim=5,
        kerc_decision_bottleneck_dim=8,
        kerc_decision_output_dim=4,
        kerc_surface_token_start=20,
        kerc_surface_token_end=40,
        kerc_kernel_token_start=40,
        kerc_kernel_token_end=60,
        kerc_pointer_token_start=60,
        kerc_pointer_token_end=90,
        kerc_end_token_id=3,
    )
    model = build_model(
        config,
        mx=mx,
        nn=nn,
        source_to_target_lookup=np.arange(96, dtype=np.int32),
    )
    inputs = mx.array([[1, 11, 20, 2, 45], [1, 13, 21, 2, 31]], dtype=mx.int32)
    labels = mx.zeros(inputs.shape, dtype=mx.int32)
    token_mask = mx.zeros(inputs.shape, dtype=mx.float32)
    unit_bytes = mx.array(
        [[[1, 2, 256], [3, 4, 256]], [[5, 6, 256], [7, 8, 256]]],
        dtype=mx.int32,
    )
    byte_mask = mx.array(
        [[[1, 1, 0], [1, 1, 0]], [[1, 1, 0], [1, 1, 0]]],
        dtype=mx.float32,
    )
    kinds = mx.array([[0, 1], [2, 4]], dtype=mx.int32)
    features = mx.zeros((2, 2, 4, 18), dtype=mx.float32)
    unit_mask = mx.ones((2, 2), dtype=mx.float32)
    hard = mx.array(
        [
            [[0, 0, 0, 0], [1, 0, 0, 0]],
            [[0, 0, 0, 0], [1, 1, 1, 0]],
        ],
        dtype=mx.bool_,
    )
    targets = mx.array([[0, 1], [2, 3]], dtype=mx.int32)
    authority = mx.ones((2, 2), dtype=mx.float32)
    confidence = mx.array([[0.9, 0.8], [0.7, 1.0]], dtype=mx.float32)
    before = {
        name: np.asarray(value).copy()
        for name, value in mlx_utils.tree_flatten(model.parameters())
        if "kerc_unit" in name
    }
    optimizer = optim.SGD(learning_rate=0.05)
    loss_and_grad = nn.value_and_grad(model, causal_loss)
    loss, gradients = loss_and_grad(
        model,
        inputs,
        labels,
        token_mask,
        mx,
        nn,
        kerc_unit_residual_labels=targets,
        kerc_unit_residual_weight=1.0,
        kerc_unit_residual_loss_mask=authority,
        kerc_unit_confidence_targets=confidence,
        kerc_unit_byte_ids=unit_bytes,
        kerc_unit_byte_mask=byte_mask,
        kerc_unit_kind_ids=kinds,
        kerc_unit_candidate_features=features,
        kerc_unit_mask=unit_mask,
        kerc_unit_hard_block_mask=hard,
    )
    optimizer.update(model, gradients)
    mx.eval(model.parameters(), optimizer.state, loss)
    after = {
        name: np.asarray(value).copy()
        for name, value in mlx_utils.tree_flatten(model.parameters())
        if "kerc_unit" in name
    }
    changed = {name for name in before if not np.array_equal(before[name], after[name])}
    assert float(loss.item()) > 0.0
    assert any("candidate" in name for name in changed)
    assert any("content" in name or "byte" in name for name in changed)
    assert any("confidence" in name for name in changed)


def test_kerc_verifier_zero_ablation_preserves_configured_output_contract() -> None:
    import mlx.core as mx
    import mlx.nn as nn

    config = CausalTransformerConfig(
        vocab_size=96,
        d_model=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        ff_dim=64,
        attention_policy="encoder_decoder",
        source_encoder_layers=1,
        source_copy_mode="pointer_generator",
        kerc_task_token_ids=(10, 11, 12, 13),
        kerc_stage_adapter_dim=8,
        kerc_residual_choice_count=4,
        kerc_residual_bottleneck_dim=8,
        kerc_verifier_dim=8,
        kerc_verifier_output_dim=5,
        kerc_decision_bottleneck_dim=8,
        kerc_decision_output_dim=4,
        kerc_verifier_ablation="zero",
        kerc_surface_token_start=20,
        kerc_surface_token_end=40,
        kerc_kernel_token_start=40,
        kerc_kernel_token_end=60,
        kerc_pointer_token_start=60,
        kerc_pointer_token_end=90,
        kerc_end_token_id=3,
    )
    model = build_model(
        config,
        mx=mx,
        nn=nn,
        source_to_target_lookup=np.arange(96, dtype=np.int32),
    )
    logits = model.kerc_verifier_logits(
        mx.array([[1, 10, 20, 2, 30]], dtype=mx.int32)
    )
    mx.eval(logits)

    assert tuple(logits.shape) == (1, 5)
    assert bool(mx.all(logits == 0.0))


def test_kerc_stage_residual_and_cache_are_source_bound() -> None:
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(47)
    config = CausalTransformerConfig(
        vocab_size=96,
        d_model=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ff_dim=64,
        attention_policy="encoder_decoder",
        source_encoder_layers=1,
        source_copy_mode="pointer_generator",
        kerc_task_token_ids=(10, 11, 12, 13),
        kerc_stage_adapter_dim=8,
        kerc_residual_choice_count=4,
        kerc_residual_bottleneck_dim=8,
        kerc_verifier_dim=8,
        kerc_decision_bottleneck_dim=8,
        kerc_decision_output_dim=4,
        kerc_surface_token_start=20,
        kerc_surface_token_end=40,
        kerc_kernel_token_start=40,
        kerc_kernel_token_end=60,
        kerc_pointer_token_start=60,
        kerc_pointer_token_end=90,
        kerc_end_token_id=3,
    )
    model = build_model(
        config,
        mx=mx,
        nn=nn,
        source_to_target_lookup=np.arange(96, dtype=np.int32),
    )
    compiler = mx.array([[1, 11, 20, 21, 2, 45, 46]], dtype=mx.int32)
    target_changed = mx.array([[1, 11, 20, 21, 2, 45, 44]], dtype=mx.int32)
    logits, _cache, aux = model(compiler, return_training_aux=True)
    changed_logits, _changed_cache, changed_aux = model(
        target_changed, return_training_aux=True
    )
    mx.eval(
        logits,
        changed_logits,
        aux["kerc"]["stage_weights"],
        aux["kerc"]["residual_logits"],
        changed_aux["kerc"]["residual_logits"],
    )
    assert np.asarray(aux["kerc"]["stage_weights"]).tolist() == [
        [0.0, 1.0, 0.0, 0.0]
    ]
    assert bool(mx.allclose(
        aux["kerc"]["residual_logits"],
        changed_aux["kerc"]["residual_logits"],
        atol=1e-6,
    ))
    assert bool(mx.allclose(logits[:, :6], changed_logits[:, :6], atol=1e-6))
    assert float(logits[0, -1, 25].item()) < -1e8
    assert np.isfinite(float(logits[0, -1, 45].item()))
    renderer_logits, _renderer_cache = model(
        mx.array([[1, 13, 20, 21, 2, 32]], dtype=mx.int32)
    )
    mx.eval(renderer_logits)
    assert np.isfinite(float(renderer_logits[0, -1, 25].item()))
    assert float(renderer_logits[0, -1, 45].item()) < -1e8
    assert bool(mx.allclose(
        aux["copy_aux"]["generator_gate"],
        mx.ones_like(aux["copy_aux"]["generator_gate"]),
        atol=1e-7,
    ))

    renderer = mx.array([[1, 13, 20, 21, 2, 30, 31]], dtype=mx.int32)
    _renderer_logits, _renderer_cache, renderer_aux = model(
        renderer, return_training_aux=True
    )
    mx.eval(renderer_aux["copy_aux"]["generator_gate"])
    assert not bool(mx.allclose(
        renderer_aux["copy_aux"]["generator_gate"],
        mx.ones_like(renderer_aux["copy_aux"]["generator_gate"]),
        atol=1e-7,
    ))

    _prefill, cache = model(compiler[:, :6])
    cached, _next = model(compiler[:, 6:], cache)
    full, _full_cache = model(compiler)
    mx.eval(cached, full)
    assert bool(mx.allclose(cached[:, -1], full[:, -1], atol=1e-4))

    malformed = mx.array([[1, 10, 11, 20, 2, 30]], dtype=mx.int32)
    _logits, _cache, malformed_aux = model(malformed, return_training_aux=True)
    mx.eval(malformed_aux["kerc"]["stage_weights"])
    assert np.asarray(malformed_aux["kerc"]["stage_weights"]).tolist() == [
        [0.0, 0.0, 0.0, 0.0]
    ]


def test_kerc_joint_loss_updates_modules_and_checkpoint_reloads(tmp_path: Path) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils as mlx_utils

    mx.random.seed(53)
    config = CausalTransformerConfig(
        vocab_size=96,
        d_model=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        ff_dim=64,
        attention_policy="encoder_decoder",
        source_encoder_layers=1,
        source_copy_mode="pointer_generator",
        kerc_task_token_ids=(10, 11, 12, 13),
        kerc_stage_adapter_dim=8,
        kerc_residual_choice_count=4,
        kerc_residual_bottleneck_dim=8,
        kerc_verifier_dim=8,
        kerc_decision_bottleneck_dim=8,
        kerc_decision_output_dim=4,
        kerc_surface_token_start=20,
        kerc_surface_token_end=40,
        kerc_kernel_token_start=40,
        kerc_kernel_token_end=60,
        kerc_pointer_token_start=60,
        kerc_pointer_token_end=90,
        kerc_end_token_id=3,
    )
    lookup = np.arange(96, dtype=np.int32)
    model = build_model(config, mx=mx, nn=nn, source_to_target_lookup=lookup)
    inputs = mx.array(
        [[1, 11, 20, 2, 45, 46], [1, 13, 22, 2, 32, 33]], dtype=mx.int32
    )
    labels = mx.array(
        [[11, 20, 2, 45, 46, 0], [13, 22, 2, 32, 33, 0]], dtype=mx.int32
    )
    mask = mx.array(
        [[0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 0]], dtype=mx.float32
    )
    residual_labels = mx.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=mx.int32)
    verifier_labels = mx.array([[0, 0, 0, 0], [1, 0, 1, 0]], dtype=mx.float32)
    decision_labels = mx.array([0, 2], dtype=mx.int32)
    optimizer = optim.SGD(learning_rate=0.05)
    value_and_grad = nn.value_and_grad(model, causal_loss)
    before = {
        name: np.asarray(value).copy()
        for name, value in mlx_utils.tree_flatten(model.parameters())
        if "kerc_" in name
    }
    loss, gradients = value_and_grad(
        model,
        inputs,
        labels,
        mask,
        mx,
        nn,
        kerc_residual_labels=residual_labels,
        kerc_residual_weight=0.5,
        kerc_verifier_labels=verifier_labels,
        kerc_verifier_weight=0.5,
        kerc_decision_labels=decision_labels,
        kerc_decision_weight=0.5,
        kerc_decision_class_weights=mx.ones((4,), dtype=mx.float32),
        kerc_decision_loss_mask=mx.ones((2,), dtype=mx.float32),
    )
    optimizer.update(model, gradients)
    mx.eval(model.parameters(), optimizer.state, loss)
    after = {
        name: np.asarray(value).copy()
        for name, value in mlx_utils.tree_flatten(model.parameters())
        if "kerc_" in name
    }
    assert before.keys() == after.keys()
    assert before and all(np.isfinite(value).all() for value in after.values())
    changed = {name for name in before if not np.array_equal(before[name], after[name])}
    assert any("kerc_residual" in name for name in changed)
    assert any("kerc_verifier" in name for name in changed)
    assert any("kerc_decision" in name for name in changed)
    assert any(
        "kerc_stage" in name or "kerc_kernel_output" in name
        or "kerc_surface_output" in name
        for name in changed
    )

    checkpoint = tmp_path / "kerc_mechanics.npz"
    model.save_weights(str(checkpoint))
    restored = build_model(config, mx=mx, nn=nn, source_to_target_lookup=lookup)
    restored.load_weights(str(checkpoint), strict=True)
    expected, _ = model(inputs)
    observed, _ = restored(inputs)
    mx.eval(expected, observed)
    assert bool(mx.allclose(expected, observed, atol=1e-6))


def test_kerc_verifier_class_weights_balance_sparse_corruptions() -> None:
    labels = np.ones((8, 4), dtype=np.float32)
    for index in range(4):
        labels[index, index] = 0.0
    positive, negative, receipt = balanced_binary_class_weights(labels)

    assert np.allclose(positive, np.full(4, 4.0 / 7.0, dtype=np.float32))
    assert np.allclose(negative, np.full(4, 4.0, dtype=np.float32))
    assert receipt["positive_counts"] == [7, 7, 7, 7]
    assert receipt["negative_counts"] == [1, 1, 1, 1]
    for feature in range(4):
        positive_mass = float(labels[:, feature].sum() * positive[feature])
        negative_mass = float((1.0 - labels[:, feature]).sum() * negative[feature])
        assert positive_mass == pytest.approx(negative_mass)


def test_kerc_verifier_class_weights_fail_closed_without_both_classes() -> None:
    with pytest.raises(ValueError, match="requires both classes"):
        balanced_binary_class_weights(np.ones((8, 4), dtype=np.float32))


def test_kerc_residual_class_weights_balance_each_channel() -> None:
    labels = np.asarray(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 2, 3],
        ],
        dtype=np.int32,
    )

    weights, receipt = balanced_categorical_class_weights(
        labels,
        class_count=4,
        maximum=16.0,
        require_two_classes_per_feature=True,
    )

    assert weights.shape == (4, 4)
    assert weights[0, 1] > weights[0, 0]
    assert weights[1, 1] > weights[1, 0]
    assert weights[2, 2] > weights[2, 0]
    assert weights[3, 3] > weights[3, 0]
    assert receipt["observed_class_count_by_feature"] == [2, 2, 2, 2]
    assert receipt["weight_sha256"] == hashlib.sha256(weights.tobytes()).hexdigest()


def test_kerc_residual_loss_mask_removes_verifier_only_rows() -> None:
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(43)
    config = CausalTransformerConfig(
        vocab_size=96,
        d_model=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        ff_dim=64,
        attention_policy="encoder_decoder",
        source_encoder_layers=1,
        source_copy_mode="pointer_generator",
        kerc_task_token_ids=(10, 11, 12, 13),
        kerc_stage_adapter_dim=8,
        kerc_residual_choice_count=4,
        kerc_residual_bottleneck_dim=8,
        kerc_verifier_dim=8,
        kerc_verifier_output_dim=5,
        kerc_decision_bottleneck_dim=8,
        kerc_decision_output_dim=4,
        kerc_surface_token_start=20,
        kerc_surface_token_end=40,
        kerc_kernel_token_start=40,
        kerc_kernel_token_end=60,
        kerc_pointer_token_start=60,
        kerc_pointer_token_end=90,
        kerc_end_token_id=3,
    )
    model = build_model(
        config,
        mx=mx,
        nn=nn,
        source_to_target_lookup=np.arange(96, dtype=np.int32),
    )
    inputs = mx.array([[1, 10, 20, 2, 30, 31]], dtype=mx.int32)
    labels = mx.array([[10, 20, 2, 30, 31, 0]], dtype=mx.int32)
    zero_token_mask = mx.zeros((1, 6), dtype=mx.float32)
    loss = causal_loss(
        model,
        inputs,
        labels,
        zero_token_mask,
        mx,
        nn,
        kerc_residual_labels=mx.array([[0, 1, 2, 3]], dtype=mx.int32),
        kerc_residual_weight=1.0,
        kerc_residual_loss_mask=mx.array([0.0], dtype=mx.float32),
    )
    mx.eval(loss)
    assert float(loss.item()) == pytest.approx(0.0)


def test_encoder_decoder_source_memory_excludes_target_values() -> None:
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(41)
    config = CausalTransformerConfig(
        vocab_size=64,
        d_model=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ff_dim=64,
        attention_policy="encoder_decoder",
        source_encoder_layers=2,
    )
    model = build_model(config, mx=mx, nn=nn)
    base = mx.array([[1, 10, 11, 2, 20, 21]], dtype=mx.int32)
    changed_target = mx.array([[1, 10, 11, 2, 33, 34]], dtype=mx.int32)
    changed_source = mx.array([[1, 10, 19, 2, 20, 21]], dtype=mx.int32)
    base_memory, base_mask, _, _ = model.encode_source(base)
    target_memory, target_mask, _, _ = model.encode_source(changed_target)
    source_memory, _source_mask, _, _ = model.encode_source(changed_source)
    mx.eval(base_memory, target_memory, source_memory, base_mask, target_mask)
    assert bool(mx.allclose(base_memory, target_memory, atol=1e-7))
    assert bool(mx.all(base_mask == target_mask))
    assert not bool(mx.allclose(base_memory, source_memory, atol=1e-7))


def test_encoder_decoder_is_causal_and_cached_generation_matches_full() -> None:
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(43)
    config = CausalTransformerConfig(
        vocab_size=64,
        d_model=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ff_dim=64,
        attention_policy="encoder_decoder",
        source_encoder_layers=2,
    )
    model = build_model(config, mx=mx, nn=nn)
    sequence = mx.array([[1, 10, 11, 2, 20, 21, 22]], dtype=mx.int32)
    future_changed = mx.array([[1, 10, 11, 2, 20, 21, 31]], dtype=mx.int32)
    source_changed = mx.array([[1, 10, 19, 2, 20, 21, 22]], dtype=mx.int32)
    full, full_cache = model(sequence)
    future, _ = model(future_changed)
    source, _ = model(source_changed)
    _prefill, prefill_cache = model(sequence[:, :5])
    cached, cached_state = model(sequence[:, 5:], prefill_cache)
    mx.eval(
        full,
        future,
        source,
        cached,
        *cache_arrays(full_cache),
        *cache_arrays(cached_state),
    )
    assert bool(mx.allclose(full[:, :6], future[:, :6], atol=1e-6))
    assert not bool(mx.allclose(full[:, 4:], source[:, 4:], atol=1e-6))
    assert bool(mx.allclose(cached, full[:, 5:], atol=1e-4))


def test_encoder_decoder_preserves_separator_free_causal_pretraining_cache() -> None:
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(47)
    config = CausalTransformerConfig(
        vocab_size=64,
        d_model=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ff_dim=64,
        attention_policy="encoder_decoder",
        source_encoder_layers=1,
    )
    model = build_model(config, mx=mx, nn=nn)
    sequence = mx.array([[1, 10, 11, 12]], dtype=mx.int32)
    full, _ = model(sequence)
    _prefill, cache = model(sequence[:, :3])
    cached, next_cache = model(sequence[:, 3:], cache)
    mx.eval(full, cached, *cache_arrays(next_cache))
    assert len(cache) == config.num_layers
    assert len(next_cache) == config.num_layers
    assert bool(mx.allclose(cached[:, -1], full[:, -1], atol=1e-4))


def test_encoder_decoder_separator_free_pretraining_forward_is_compilable() -> None:
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(49)
    model = build_model(
        CausalTransformerConfig(
            vocab_size=64,
            d_model=32,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            ff_dim=64,
            attention_policy="encoder_decoder",
            source_encoder_layers=1,
        ),
        mx=mx,
        nn=nn,
    )
    mx.eval(model.parameters())

    def forward(tokens):
        return model(tokens, source_conditioning=False)[0]

    compiled_forward = mx.compile(forward)
    for sequence in (
        mx.array([[1, 10, 11, 12, 13, 14]], dtype=mx.int32),
        mx.array([[1, 10, 11, 12]], dtype=mx.int32),
    ):
        eager = forward(sequence)
        compiled = compiled_forward(sequence)
        mx.eval(eager, compiled)
        assert bool(mx.allclose(eager, compiled, atol=1e-6))


def test_compiled_microbatch_pretraining_matches_eager_full_batch(
    tmp_path: Path,
) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils as mlx_utils

    config = CausalTransformerConfig(
        vocab_size=64,
        d_model=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ff_dim=64,
    )
    mx.random.seed(51)
    compiled_model = build_model(config, mx=mx, nn=nn)
    mx.eval(compiled_model.parameters())
    eager_model = build_model(config, mx=mx, nn=nn)
    eager_model.load_weights(
        list(mlx_utils.tree_flatten(compiled_model.parameters()))
    )
    mx.eval(eager_model.parameters())
    rng = np.random.default_rng(19)
    inputs = rng.integers(3, 64, size=(8, 16), dtype=np.int32)
    labels = np.roll(inputs, -1, axis=1)
    mask = np.ones_like(inputs, dtype=np.float32)
    common = {
        "inputs": inputs,
        "labels": labels,
        "mask": mask,
        "progress_mask": mask,
        "ordered_plan_loss_weight": 1.0,
        "sample_weights": None,
        "plan_labels": None,
        "plan_label_mode": "none",
        "plan_auxiliary_weight": 0.0,
        "plan_shuffle_seed": 0,
        "plan_loss_mode": "binary_multilabel",
        "plan_slot_count": 0,
        "plan_factor_group_sizes": (),
        "phase_name": "compiled-parity",
        "target_positions": int(mask.sum()),
        "batch_size": 8,
        "gradient_clip": 1.0,
        "seed": 23,
        "max_steps": 1,
        "checkpoint": tmp_path / "unused.npz",
        "checkpoint_every": 99,
        "heartbeat": tmp_path / "heartbeat.json",
        "global_step_offset": 0,
        "mx": mx,
        "optim": optim,
    }
    compiled_report = survival.train_phase(
        compiled_model,
        optim.AdamW(learning_rate=1e-3),
        nn.value_and_grad(compiled_model, causal_loss),
        source_conditioning=False,
        **common,
    )
    eager_report = survival.train_phase(
        eager_model,
        optim.AdamW(learning_rate=1e-3),
        nn.value_and_grad(eager_model, causal_loss),
        source_conditioning=False,
        training_step_mode="eager",
        **common,
    )
    compiled_parameters = dict(mlx_utils.tree_flatten(compiled_model.parameters()))
    eager_parameters = dict(mlx_utils.tree_flatten(eager_model.parameters()))
    maximum_delta = max(
        float(mx.max(mx.abs(compiled_parameters[name] - eager_parameters[name])).item())
        for name in compiled_parameters
    )

    assert compiled_report["training_step_execution"] == (
        "mlx_compiled_shape_bucket_v1"
    )
    assert eager_report["training_step_execution"] == (
        "mlx_eager_auxiliary_objective_v1"
    )
    assert compiled_report["training_step_mode_requested"] == "auto"
    assert eager_report["training_step_mode_requested"] == "eager"
    assert compiled_report["final_loss"] == pytest.approx(
        eager_report["final_loss"], abs=2e-6
    )
    assert maximum_delta < 5e-6


def test_zero_initialized_expert_adapter_preserves_trunk_and_freezes_exactly() -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.utils as mlx_utils

    common = {
        "vocab_size": 64,
        "d_model": 32,
        "num_layers": 2,
        "num_heads": 4,
        "num_kv_heads": 2,
        "ff_dim": 64,
    }
    mx.random.seed(53)
    trunk = build_model(CausalTransformerConfig(**common), mx=mx, nn=nn)
    mx.random.seed(53)
    expert = build_model(
        CausalTransformerConfig(**common, expert_adapter_dim=8), mx=mx, nn=nn
    )
    expert.load_weights(
        list(mlx_utils.tree_flatten(trunk.parameters())), strict=False
    )
    sequence = mx.array([[1, 10, 11, 12]], dtype=mx.int32)
    trunk_logits, _ = trunk(sequence)
    expert_logits, _ = expert(sequence)
    mx.eval(trunk_logits, expert_logits)
    assert bool(mx.allclose(trunk_logits, expert_logits, atol=1e-6))
    expert.freeze_to_expert_adapter()
    trainable = {
        name for name, _value in mlx_utils.tree_flatten(expert.trainable_parameters())
    }
    assert trainable
    assert all(".expert_adapter." in name for name in trainable)
    assert parameter_count(expert, mlx_utils) > parameter_count(trunk, mlx_utils)


def test_source_conditioned_expert_scope_owns_only_specialist_modules() -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.utils as mlx_utils

    model = build_model(
        CausalTransformerConfig(
            vocab_size=64,
            d_model=32,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            ff_dim=64,
            attention_policy="encoder_decoder",
            source_encoder_layers=1,
            source_copy_mode="pointer_generator",
            source_copy_auxiliary_loss_weight=0.25,
            expert_adapter_dim=8,
        ),
        mx=mx,
        nn=nn,
        source_to_target_lookup=mx.arange(64, dtype=mx.int32),
    )
    model.freeze_to_language_expert("source_conditioned_delta")
    trainable = {
        name for name, _value in mlx_utils.tree_flatten(model.trainable_parameters())
    }
    assert trainable
    assert any(".expert_adapter." in name for name in trainable)
    assert any(".source_attention." in name for name in trainable)
    assert any(name.startswith("source_layers.") for name in trainable)
    assert {"copy_query.weight", "copy_key.weight", "copy_gate.weight", "copy_gate.bias"}.issubset(
        trainable
    )
    assert "token_embedding.weight" not in trainable
    assert not any(
        ".attention.q_proj." in name
        and ".source_attention." not in name
        and not name.startswith("source_layers.")
        for name in trainable
    )
    assert not any(".feed_forward." in name and not name.startswith("source_layers.") for name in trainable)


def test_low_rank_source_expert_scope_stays_inside_data_bounded_adapters() -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.utils as mlx_utils

    model = build_model(
        CausalTransformerConfig(
            vocab_size=64,
            d_model=32,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            ff_dim=64,
            attention_policy="encoder_decoder",
            source_encoder_layers=1,
            source_copy_mode="pointer_generator",
            source_copy_auxiliary_loss_weight=0.25,
            expert_adapter_dim=8,
            source_expert_adapter_dim=4,
        ),
        mx=mx,
        nn=nn,
        source_to_target_lookup=mx.arange(64, dtype=mx.int32),
    )
    model.freeze_to_language_expert("low_rank_source_adapters")
    trainable = {
        name for name, _value in mlx_utils.tree_flatten(model.trainable_parameters())
    }
    assert any(".expert_adapter." in name for name in trainable)
    assert any(".source_expert_adapter." in name for name in trainable)
    assert any(
        name.startswith("source_layers.") and ".expert_adapter." in name
        for name in trainable
    )
    assert {"copy_gate.weight", "copy_gate.bias"}.issubset(trainable)
    assert "copy_query.weight" not in trainable
    assert "copy_key.weight" not in trainable
    assert not any(".source_attention." in name for name in trainable)
    assert not any(
        name.startswith("source_layers.")
        and ".expert_adapter." not in name
        for name in trainable
    )


def test_expert_adapter_configuration_fails_closed() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        CausalTransformerConfig(vocab_size=64, expert_adapter_dim=-1).validate()
    with pytest.raises(ValueError, match="cannot be negative"):
        CausalTransformerConfig(vocab_size=64, source_expert_adapter_dim=-1).validate()
    with pytest.raises(ValueError, match="require encoder-decoder"):
        CausalTransformerConfig(vocab_size=64, source_expert_adapter_dim=4).validate()


def test_sequence_partition_audit_rejects_boundary_corruption() -> None:
    valid_inputs = np.array([[1, 10, 11, 2, 20, 21, 0]], dtype=np.int32)
    valid_mask = np.array([[0, 0, 0, 0, 1, 1, 0]], dtype=np.float32)
    assert sequence_partition_audit(
        valid_inputs, valid_mask, require_separator=True
    )["valid"]

    target_leak = valid_mask.copy()
    target_leak[0, 2] = 1.0
    leak_receipt = sequence_partition_audit(
        valid_inputs, target_leak, require_separator=True
    )
    assert not leak_receipt["valid"]
    assert leak_receipt["target_not_strictly_after_separator_row_count"] == 1

    duplicate = valid_inputs.copy()
    duplicate[0, 1] = 2
    duplicate_receipt = sequence_partition_audit(
        duplicate, valid_mask, require_separator=True
    )
    assert not duplicate_receipt["valid"]
    assert duplicate_receipt["multiple_separator_row_count"] == 1

    raw_code_receipt = sequence_partition_audit(
        valid_inputs, valid_mask, require_separator=False
    )
    assert not raw_code_receipt["valid"]
    assert raw_code_receipt["unexpected_separator_row_count"] == 1


def test_attention_policy_ablation_requires_matched_behavior_gain() -> None:
    def report(policy: str, passed: int, coverage: int, reward: float) -> dict:
        return {
            "policy": "project_theseus_standard_causal_transformer_survival_v1",
            "seed": 41,
            "architecture": {
                "family": "standard_decoder_only_causal_transformer",
                "attention_policy": policy,
                "parameter_count": 1000,
                "config": {
                    "vocab_size": 64,
                    "d_model": 32,
                    "attention_policy": policy,
                    "source_target_separator_token_id": 2,
                },
            },
            "stage": {
                "stage_signature": "same-stage",
                "holdout_families": ["a", "b"],
                "sequence_partition_audit": {
                    "pretrain": {"valid": True},
                    "sft": {"valid": True},
                    "eval": {"valid": True},
                },
            },
            "training": {
                "complete": True,
                "eval_loss_after": 1.0,
                "phases": [
                    {
                        "phase": "prompt_signature_body_sft",
                        "optimizer_steps": 10,
                        "target_positions_consumed": 1000,
                    }
                ],
            },
            "private_verifier": {
                "summary": {"passed_task_count": passed},
                "private_verification": {"mean_verification_reward": reward},
            },
            "summary": {
                "candidate_task_count": coverage,
                "syntax_valid_candidate_count": coverage,
                "public_training_rows": 0,
                "external_inference_calls": 0,
                "fallback_return_count": 0,
                "template_renderer_router_tool_credit_count": 0,
            },
            "runtime_ms": 100,
        }

    causal = report("causal", 1, 20, 0.25)
    improved = compare_attention_policy_canaries(
        causal, report("prefix_lm", 2, 20, 0.30)
    )
    assert improved["trigger_state"] == "GREEN"
    assert improved["adoption_state"] == "ADOPTED"

    loss_only = report("prefix_lm", 1, 20, 0.30)
    loss_only["training"]["eval_loss_after"] = 0.5
    rejected = compare_attention_policy_canaries(causal, loss_only)
    assert rejected["trigger_state"] == "GREEN"
    assert rejected["adoption_state"] == "NOT_ADOPTED"
    assert "no_family_disjoint_verifier_pass_gain" in rejected["rejection_reasons"]

    mismatched = report("prefix_lm", 2, 20, 0.30)
    mismatched["seed"] = 42
    invalid = compare_attention_policy_canaries(causal, mismatched)
    assert invalid["trigger_state"] == "RED"
    assert invalid["adoption_state"] == "NOT_ADOPTED"
    assert "seed" in invalid["hard_gaps"]


def test_semantic_plan_head_is_source_only_and_cache_partition_invariant() -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import semantic_ir

    mx.random.seed(13)
    config = CausalTransformerConfig(
        vocab_size=96,
        d_model=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ff_dim=64,
        semantic_plan_feature_count=len(semantic_ir.plan_obligation_features()),
        semantic_plan_separator_token_id=2,
    )
    model = build_model(config, mx=mx, nn=nn)
    sequence_a = mx.array([[1, 12, 13, 2, 21, 22]], dtype=mx.int32)
    sequence_b = mx.array([[1, 12, 13, 2, 44, 45]], dtype=mx.int32)
    logits_a, cache_a, plan_a = model(sequence_a, return_plan_logits=True)
    _logits_b, _cache_b, plan_b = model(sequence_b, return_plan_logits=True)
    sequence_c = mx.array([[1, 14, 15, 2, 21, 22]], dtype=mx.int32)
    _logits_c, _cache_c, plan_c = model(sequence_c, return_plan_logits=True)
    prefix_logits, prefix_cache, prefix_plan = model(
        sequence_a[:, :4], return_plan_logits=True
    )
    cached_logits, cached_state = model(sequence_a[:, 4:], prefix_cache)
    mx.eval(
        logits_a,
        plan_a,
        plan_b,
        plan_c,
        prefix_logits,
        prefix_plan,
        cached_logits,
        *cache_arrays(cache_a),
        *cache_arrays(cached_state),
    )
    assert bool(mx.allclose(plan_a, plan_b, atol=1e-6))
    assert not bool(mx.allclose(plan_a, plan_c, atol=1e-6))
    assert bool(mx.allclose(plan_a, prefix_plan, atol=1e-6))
    assert bool(mx.allclose(cached_logits, logits_a[:, 4:], atol=1e-4))
    assert len(cache_a) == config.num_layers + 1


def test_low_rank_semantic_plan_bottleneck_is_source_only_and_trainable() -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    mx.random.seed(17)
    config = CausalTransformerConfig(
        vocab_size=96,
        d_model=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ff_dim=64,
        semantic_plan_feature_count=48,
        semantic_plan_separator_token_id=2,
        semantic_plan_bottleneck_dim=8,
    )
    model = build_model(config, mx=mx, nn=nn)
    tokens = mx.array([[1, 12, 13, 2, 21, 22]], dtype=mx.int32)
    labels = mx.array([[0, 1, 0, 0, 0, 0] * 8], dtype=mx.float32)
    target = mx.array([[12, 13, 2, 21, 22, 0]], dtype=mx.int32)
    mask = mx.array([[0, 0, 0, 1, 1, 0]], dtype=mx.float32)
    loss_and_grad = nn.value_and_grad(model, causal_loss)
    loss, grads = loss_and_grad(
        model, tokens, target, mask, mx, nn, labels, 0.25, None
    )
    optimizer = optim.AdamW(learning_rate=1e-3)
    optimizer.update(model, grads)
    mx.eval(loss, model.parameters(), optimizer.state)
    assert float(loss.item()) > 0.0
    assert tuple(model.semantic_plan_classifier.weight.shape) == (48, 8)
    assert tuple(model.semantic_plan_features.weight.shape) == (48, 8)


def test_slot_plan_attention_is_source_only_and_cache_partition_invariant() -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.utils as mlx_utils

    mx.random.seed(23)
    config = CausalTransformerConfig(
        vocab_size=96,
        d_model=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ff_dim=64,
        semantic_plan_feature_count=48,
        semantic_plan_separator_token_id=2,
        semantic_plan_bottleneck_dim=8,
        semantic_plan_slot_count=4,
        semantic_plan_conditioning_mode="slot_attention",
    )
    model = build_model(config, mx=mx, nn=nn)
    sequence_a = mx.array([[1, 12, 13, 2, 21, 22]], dtype=mx.int32)
    sequence_b = mx.array([[1, 12, 13, 2, 44, 45]], dtype=mx.int32)
    sequence_c = mx.array([[1, 14, 15, 2, 21, 22]], dtype=mx.int32)
    logits_a, cache_a, plan_a = model(sequence_a, return_plan_logits=True)
    _logits_b, _cache_b, plan_b = model(sequence_b, return_plan_logits=True)
    _logits_c, _cache_c, plan_c = model(sequence_c, return_plan_logits=True)
    prefix_logits, prefix_cache, prefix_plan = model(
        sequence_a[:, :4], return_plan_logits=True
    )
    cached_logits, cached_state = model(sequence_a[:, 4:], prefix_cache)
    mx.eval(
        logits_a,
        plan_a,
        plan_b,
        plan_c,
        prefix_logits,
        prefix_plan,
        cached_logits,
        *cache_arrays(cache_a),
        *cache_arrays(cached_state),
    )
    assert bool(mx.allclose(plan_a, plan_b, atol=1e-6))
    assert not bool(mx.allclose(plan_a, plan_c, atol=1e-6))
    assert bool(mx.allclose(plan_a, prefix_plan, atol=1e-6))
    assert bool(mx.allclose(cached_logits, logits_a[:, 4:], atol=1e-4))
    assert tuple(cache_a[-1][0].shape) == (1, 4, 32)

    common = dict(
        vocab_size=96,
        d_model=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ff_dim=64,
    )
    mx.random.seed(31)
    body_model = build_model(CausalTransformerConfig(**common), mx=mx, nn=nn)
    mx.random.seed(31)
    slot_model = build_model(
        CausalTransformerConfig(
            **common,
            semantic_plan_feature_count=48,
            semantic_plan_separator_token_id=2,
            semantic_plan_bottleneck_dim=8,
            semantic_plan_slot_count=4,
            semantic_plan_conditioning_mode="slot_attention",
        ),
        mx=mx,
        nn=nn,
    )
    body_logits, _body_cache = body_model(sequence_a)
    slot_logits, _slot_cache = slot_model(sequence_a)
    mx.eval(body_logits, slot_logits)
    assert bool(mx.allclose(body_logits, slot_logits, atol=1e-6))

    target = mx.array([[12, 13, 2, 21, 22, 0]], dtype=mx.int32)
    mask = mx.array([[0, 0, 0, 1, 1, 0]], dtype=mx.float32)
    value_and_grad = nn.value_and_grad(slot_model, causal_loss)
    body_loss, gradients = value_and_grad(
        slot_model, sequence_a, target, mask, mx, nn, None, 0.0, None
    )
    mx.eval(body_loss, gradients)
    plan_output_gradients = [
        value
        for name, value in mlx_utils.tree_flatten(gradients)
        if "plan_attention.out_proj.weight" in name
    ]
    assert len(plan_output_gradients) == config.num_layers
    assert all(float(mx.sum(mx.abs(value)).item()) > 0.0 for value in plan_output_gradients)


def test_slot_categorical_plan_objective_handles_empty_slots_and_rejects_multihot() -> None:
    logits = np.asarray(
        [
            [4.0, -2.0, -3.0, -3.0, -2.0, -1.0],
            [-2.0, 5.0, -3.0, -2.0, -3.0, 6.0],
        ],
        dtype=np.float32,
    )
    targets = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    metrics = semantic_plan_metrics_from_logits(
        logits, targets, loss_mode="slot_categorical", slot_count=2
    )
    assert metrics["micro_accuracy"] == 1.0
    assert metrics["micro_f1"] == 1.0
    assert metrics["exact_row_accuracy"] == 1.0
    assert metrics["empty_slot_accuracy"] == 1.0
    assert metrics["active_slot_count"] == 3
    assert metrics["predicted_active_slot_count"] == 3

    malformed = targets.copy()
    malformed[0, 1] = 1.0
    with pytest.raises(ValueError, match="zero-or-one-hot"):
        semantic_plan_metrics_from_logits(
            logits, malformed, loss_mode="slot_categorical", slot_count=2
        )

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.utils as mlx_utils

    mx.random.seed(47)
    model = build_model(
        CausalTransformerConfig(
            vocab_size=96,
            d_model=32,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            ff_dim=64,
            semantic_plan_feature_count=48,
            semantic_plan_separator_token_id=2,
            semantic_plan_bottleneck_dim=8,
            semantic_plan_slot_count=4,
            semantic_plan_conditioning_mode="slot_attention",
            semantic_plan_probability_mode="slot_categorical",
        ),
        mx=mx,
        nn=nn,
    )
    sequence = mx.array([[1, 12, 13, 2, 21, 22]], dtype=mx.int32)
    token_targets = mx.array([[12, 13, 2, 21, 22, 0]], dtype=mx.int32)
    body_mask = mx.array([[0, 0, 0, 1, 1, 0]], dtype=mx.float32)
    plan_targets = np.zeros((1, 48), dtype=np.float32)
    plan_targets[0, 0] = 1.0
    plan_targets[0, 13] = 1.0
    value_and_grad = nn.value_and_grad(model, causal_loss)
    loss, gradients = value_and_grad(
        model,
        sequence,
        token_targets,
        body_mask,
        mx,
        nn,
        mx.array(plan_targets),
        0.25,
        None,
        "slot_categorical",
        4,
    )
    mx.eval(loss, gradients)
    classifier_gradients = [
        value
        for name, value in mlx_utils.tree_flatten(gradients)
        if "semantic_plan_classifier" in name
    ]
    assert classifier_gradients
    assert any(
        float(mx.sum(mx.abs(value)).item()) > 0.0 for value in classifier_gradients
    )


def test_factorized_step_plan_objective_is_group_closed_and_cache_invariant() -> None:
    groups = (1, 2, 3)
    logits = np.asarray(
        [[3.0, -2.0, 4.0, -3.0, -2.0, 5.0, -4.0, 2.0, -2.0, 1.0, 0.0, -1.0]],
        dtype=np.float32,
    )
    targets = np.asarray(
        [[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    metrics = semantic_plan_metrics_from_logits(
        logits,
        targets,
        loss_mode="factorized_step_categorical",
        slot_count=2,
        factor_group_sizes=groups,
    )
    assert metrics["micro_f1"] == 1.0
    assert metrics["exact_row_accuracy"] == 1.0
    assert metrics["empty_slot_accuracy"] == 1.0
    assert metrics["factor_accuracy_on_active_slots"] == 1.0

    malformed = targets.copy()
    malformed[0, 2] = 0.0
    with pytest.raises(ValueError, match="one category per active factor"):
        semantic_plan_metrics_from_logits(
            logits,
            malformed,
            loss_mode="factorized_step_categorical",
            slot_count=2,
            factor_group_sizes=groups,
        )

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.utils as mlx_utils

    model_groups = (1, 2, 3, 6)
    mx.random.seed(53)
    model = build_model(
        CausalTransformerConfig(
            vocab_size=96,
            d_model=32,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            ff_dim=64,
            semantic_plan_feature_count=48,
            semantic_plan_separator_token_id=2,
            semantic_plan_bottleneck_dim=8,
            semantic_plan_slot_count=4,
            semantic_plan_conditioning_mode="slot_attention",
            semantic_plan_probability_mode="factorized_step",
            semantic_plan_factor_group_sizes=model_groups,
        ),
        mx=mx,
        nn=nn,
    )
    sequence = mx.array([[1, 12, 13, 2, 21, 22]], dtype=mx.int32)
    full_logits, full_cache, _plan_logits = model(
        sequence, return_plan_logits=True
    )
    _prefix_logits, prefix_cache, _prefix_plan = model(
        sequence[:, :4], return_plan_logits=True
    )
    cached_logits, cached_state = model(sequence[:, 4:], prefix_cache)
    mx.eval(
        full_logits,
        cached_logits,
        *cache_arrays(full_cache),
        *cache_arrays(cached_state),
    )
    assert bool(mx.allclose(cached_logits, full_logits[:, 4:], atol=1e-4))

    token_targets = mx.array([[12, 13, 2, 21, 22, 0]], dtype=mx.int32)
    body_mask = mx.array([[0, 0, 0, 1, 1, 0]], dtype=mx.float32)
    plan_targets = np.zeros((1, 48), dtype=np.float32)
    offset = 0
    plan_targets[0, offset] = 1.0
    offset += 1
    for width in model_groups[1:]:
        plan_targets[0, offset] = 1.0
        offset += width
    value_and_grad = nn.value_and_grad(model, causal_loss)
    loss, gradients = value_and_grad(
        model,
        sequence,
        token_targets,
        body_mask,
        mx,
        nn,
        mx.array(plan_targets),
        0.25,
        None,
        "factorized_step_categorical",
        4,
        model_groups,
    )
    mx.eval(loss, gradients)
    classifier_gradients = [
        value
        for name, value in mlx_utils.tree_flatten(gradients)
        if "semantic_plan_classifier" in name
    ]
    assert classifier_gradients
    assert any(
        float(mx.sum(mx.abs(value)).item()) > 0.0 for value in classifier_gradients
    )


def test_semantic_plan_auxiliary_loss_updates_plan_parameters() -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils as mlx_utils
    import semantic_ir

    mx.random.seed(29)
    model = build_model(
        CausalTransformerConfig(
            vocab_size=96,
            d_model=32,
            num_layers=1,
            num_heads=4,
            num_kv_heads=2,
            ff_dim=64,
            semantic_plan_feature_count=len(semantic_ir.plan_obligation_features()),
            semantic_plan_separator_token_id=2,
        ),
        mx=mx,
        nn=nn,
    )
    inputs = mx.array([[1, 12, 13, 2, 21, 22], [1, 14, 15, 2, 23, 24]], dtype=mx.int32)
    labels = mx.array([[12, 13, 2, 21, 22, 0], [14, 15, 2, 23, 24, 0]], dtype=mx.int32)
    mask = mx.array([[0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 0]], dtype=mx.float32)
    plan_targets = np.zeros(
        (2, len(semantic_ir.plan_obligation_features())), dtype=np.float32
    )
    plan_targets[:, 0] = 1.0
    plan_labels = mx.array(plan_targets, dtype=mx.float32)
    optimizer = optim.SGD(learning_rate=0.05)
    value_and_grad = nn.value_and_grad(model, causal_loss)
    before = {
        name: np.asarray(value).copy()
        for name, value in mlx_utils.tree_flatten(model.parameters())
        if "semantic_plan" in name
    }
    loss, gradients = value_and_grad(
        model, inputs, labels, mask, mx, nn, plan_labels, 0.5
    )
    optimizer.update(model, gradients)
    mx.eval(model.parameters(), optimizer.state, loss)
    after = {
        name: np.asarray(value).copy()
        for name, value in mlx_utils.tree_flatten(model.parameters())
        if "semantic_plan" in name
    }
    assert before.keys() == after.keys()
    assert before
    assert all(np.isfinite(value).all() for value in after.values())
    assert any(not np.array_equal(before[name], after[name]) for name in before)


def test_zero_initialized_plan_residual_starts_body_logit_neutral() -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import semantic_ir

    common = dict(
        vocab_size=96,
        d_model=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        ff_dim=64,
    )
    mx.random.seed(37)
    body_model = build_model(CausalTransformerConfig(**common), mx=mx, nn=nn)
    mx.random.seed(37)
    plan_model = build_model(
        CausalTransformerConfig(
            **common,
            semantic_plan_feature_count=len(semantic_ir.plan_obligation_features()),
            semantic_plan_separator_token_id=2,
        ),
        mx=mx,
        nn=nn,
    )
    tokens = mx.array([[1, 12, 13, 2, 21, 22]], dtype=mx.int32)
    body_logits, _body_cache = body_model(tokens)
    plan_logits, _plan_cache, obligations = plan_model(
        tokens, return_plan_logits=True
    )
    mx.eval(body_logits, plan_logits, obligations)
    assert bool(mx.allclose(body_logits, plan_logits, atol=1e-6))


def test_semantic_plan_shuffled_control_is_deranged_and_mass_matched() -> None:
    labels = np.eye(8, dtype=np.float32)
    semantic, semantic_receipt = prepare_semantic_plan_labels(
        labels, mode="semantic", seed=17
    )
    shuffled, shuffled_receipt = prepare_semantic_plan_labels(
        labels, mode="shuffled", seed=17
    )
    assert np.array_equal(semantic, labels)
    assert shuffled is not None
    assert not any(np.array_equal(shuffled[index], labels[index]) for index in range(len(labels)))
    assert np.array_equal(shuffled.sum(axis=0), labels.sum(axis=0))
    assert semantic_receipt["positive_label_count"] == shuffled_receipt["positive_label_count"]
    assert shuffled_receipt["fixed_point_count"] == 0
    assert shuffled_receipt["label_sha256"] != semantic_receipt["label_sha256"]


def test_latent_ordered_plan_field_is_closed_low_rank_and_body_stream_preserving() -> None:
    import semantic_ir

    config = json.loads((ROOT / "configs" / "standard_causal_transformer_survival.json").read_text())
    slot_count = 16
    feature_count = slot_count * len(semantic_ir.plan_protocol_tokens())
    config["model"].update(
        {
            "semantic_plan_feature_count": feature_count,
            "semantic_plan_bottleneck_dim": 32,
        }
    )
    config["semantic_plan_training"] = {
        "enabled": True,
        "label_mode": "semantic",
        "auxiliary_loss_weight": 0.25,
        "shuffle_seed": 101,
        "target": "ordered_plan_slot_token_field",
        "ordered_slot_count": slot_count,
    }
    validate_config(config)
    features = semantic_plan_feature_contract(config)
    labels = semantic_plan_labels_for_body("return data", config)
    assert len(features) == len(labels) == feature_count
    assert sum(labels) == len(semantic_ir.body_to_plan_tokens("return data", max_tokens=slot_count))
    assert config["tokenization"]["target_mode"] == "body_tokens"

    slot_attention = json.loads(json.dumps(config))
    slot_attention["model"].update(
        {
            "semantic_plan_slot_count": slot_count,
            "semantic_plan_conditioning_mode": "slot_attention",
        }
    )
    validate_config(slot_attention)
    mismatched_slots = json.loads(json.dumps(slot_attention))
    mismatched_slots["model"]["semantic_plan_slot_count"] = slot_count - 1
    with pytest.raises(ValueError, match="must match ordered plan slots"):
        validate_config(mismatched_slots)

    categorical = json.loads(json.dumps(slot_attention))
    categorical["model"]["semantic_plan_probability_mode"] = "slot_categorical"
    categorical["semantic_plan_training"]["loss_mode"] = "slot_categorical"
    validate_config(categorical)
    mismatched_objective = json.loads(json.dumps(categorical))
    mismatched_objective["model"]["semantic_plan_probability_mode"] = (
        "independent_sigmoid"
    )
    with pytest.raises(ValueError, match="requires categorical slot probabilities"):
        validate_config(mismatched_objective)

    factorized = json.loads(json.dumps(slot_attention))
    factor_slot_count = semantic_ir.PLAN_MAX_STEPS
    factorized["model"].update(
        {
            "semantic_plan_feature_count": len(
                semantic_ir.ordered_plan_step_features(factor_slot_count)
            ),
            "semantic_plan_slot_count": factor_slot_count,
            "semantic_plan_probability_mode": "factorized_step",
            "semantic_plan_factor_group_sizes": list(
                semantic_ir.ordered_plan_step_factor_group_sizes()
            ),
        }
    )
    factorized["semantic_plan_training"].update(
        {
            "target": "ordered_plan_step_factor_field",
            "ordered_slot_count": factor_slot_count,
            "loss_mode": "factorized_step_categorical",
        }
    )
    validate_config(factorized)
    mismatched_factors = json.loads(json.dumps(factorized))
    mismatched_factors["model"]["semantic_plan_factor_group_sizes"][-1] -= 1
    with pytest.raises(ValueError, match="must match the registered IR field"):
        validate_config(mismatched_factors)

    semantic, semantic_receipt = prepare_semantic_plan_labels(
        np.asarray([labels, labels], dtype=np.float32), mode="semantic", seed=7
    )
    dropped, dropout_receipt = prepare_semantic_plan_labels(
        np.asarray([labels, labels], dtype=np.float32), mode="dropout", seed=7
    )
    assert semantic is not None and dropped is not None
    assert int(semantic.sum()) > 0
    assert int(dropped.sum()) == 0
    assert semantic_receipt["label_mode"] == "semantic"
    assert dropout_receipt["label_mode"] == "dropout"


def test_ordered_plan_gate_requires_exact_gain_and_invalid_controls(tmp_path: Path) -> None:
    modes = {
        "body_only": ("body_tokens", "none"),
        "semantic": ("typed_semantic_ir_plan_body_tokens_v1", "semantic"),
        "shuffled": ("typed_semantic_ir_plan_body_tokens_v1", "shuffled"),
        "dropout": ("typed_semantic_ir_plan_body_tokens_v1", "dropout"),
    }
    directories: dict[str, Path] = {}
    for name, (target_mode, label_mode) in modes.items():
        directory = tmp_path / name
        directory.mkdir()
        directories[name] = directory
        config = {
            "seed": 1,
            "sources": {"private": "x"},
            "model": {"d_model": 32},
            "tokenization": {
                "target_mode": target_mode,
                "max_source_tokens": 32,
                "sequence_plan_reserve_tokens": 33,
            },
            "training": {"sft_target_token_positions": 100},
            "evaluation": {"holdout_family_count": 24},
        }
        if label_mode != "none":
            config["tokenization"]["semantic_plan_max_tokens"] = 32
            config["ordered_plan_training"] = {
                "label_mode": label_mode,
                "plan_loss_weight": 0.25,
                "shuffle_seed": 7,
            }
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
        candidates = directory / "candidates.jsonl"
        candidates.write_text("{}\n", encoding="utf-8")
        receipt = {
            "label_mode": label_mode,
            "row_count": 10,
            "fixed_point_count": 10 if label_mode == "semantic" else 0,
            "encoded_plan_position_count": 0 if label_mode == "none" else 200,
            "encoded_body_position_count": 500,
            "plan_sha256": "" if label_mode == "none" else f"hash-{label_mode}",
        }
        report = {
            "architecture": {"parameter_count": 100 if name == "body_only" else 120},
            "stage": {
                "family_disjoint_eval_task_count": 24,
                "unique_body_target_positions": 500,
                "unique_sft_body_count": 50,
                "train_holdout_family_overlap_count": 0,
                "train_eval_prompt_overlap_count": 0,
                "train_eval_body_overlap_count": 0,
                "holdout_families": [f"family-{index}" for index in range(24)],
                "ordered_plan_label_receipt": receipt,
            },
            "training": {
                "eval_loss_after": 1.0,
                "ordered_plan_eval_after": (
                    {"state": "NOT_APPLICABLE"}
                    if name == "body_only"
                    else {
                        "state": "MEASURED",
                        "teacher_forced_loss": {
                            "semantic": 0.5,
                            "shuffled": 0.8,
                            "dropout": 0.9,
                        }[name],
                    }
                ),
                "phases": [
                    {
                        "phase": "prompt_signature_body_sft",
                        "optimizer_body_positions_consumed": 100,
                    }
                ],
            },
            "summary": {
                "model_only_passed_task_count": 1 if name == "semantic" else 0,
                "candidate_task_count": 8 if name == "semantic" else 7,
                "candidate_count": 12 if name == "semantic" else 10,
            },
            "private_verifier": {
                "private_verification": {
                    "mean_verification_reward": 0.6 if name == "semantic" else 0.5
                }
            },
            "decode": {"runtime_ms": 10},
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        }
        (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
        (directory / "integrity.json").write_text(
            json.dumps(
                {
                    "source": str(candidates),
                    "trigger_state": "GREEN",
                    "summary": {"candidate_count": 1, "integrity_mismatch_count": 0},
                }
            ),
            encoding="utf-8",
        )
        (directory / "blind_audit.json").write_text(
            json.dumps({"trigger_state": "GREEN", "summary": {"invalid_claim_count": 0}}),
            encoding="utf-8",
        )

    adopted = audit_ordered_plan_ablation(directories)
    assert adopted["state"] == "GREEN"
    assert adopted["adoption_state"] == "ADOPTED"

    semantic_report_path = directories["semantic"] / "report.json"
    semantic_report = json.loads(semantic_report_path.read_text())
    semantic_report["summary"]["model_only_passed_task_count"] = 0
    semantic_report_path.write_text(json.dumps(semantic_report), encoding="utf-8")
    rejected = audit_ordered_plan_ablation(directories)
    assert rejected["state"] == "GREEN"
    assert rejected["adoption_state"] == "NOT_ADOPTED"
    assert "no_family_disjoint_verifier_pass_gain" in rejected["adoption_rejection_reasons"]


def test_evaluation_replay_requires_distinct_content_bound_training_receipt(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.npz"
    checkpoint.write_bytes(b"retained-weights")
    prior = tmp_path / "completed-training.json"
    prior.write_text(json.dumps({"training": {"complete": True}}), encoding="utf-8")
    live = tmp_path / "report.json"
    report = {
        "training": {"evaluation_only_replay": True},
        "artifacts": {
            "checkpoint": str(checkpoint),
            "prior_training_receipt": str(prior),
        },
        "conditioning": {
            "evaluation_base_checkpoint_sha256": hashlib.sha256(
                checkpoint.read_bytes()
            ).hexdigest(),
            "prior_training_receipt_sha256": hashlib.sha256(prior.read_bytes()).hexdigest(),
            "evaluation_replay_contract": "content_bound_checkpoint_and_training_receipt_v1",
        },
    }
    assert evaluation_replay_is_content_bound(report, live)

    report["conditioning"]["prior_training_receipt_sha256"] = "0" * 64
    assert not evaluation_replay_is_content_bound(report, live)

    report["conditioning"]["prior_training_receipt_sha256"] = hashlib.sha256(
        prior.read_bytes()
    ).hexdigest()
    report["artifacts"]["prior_training_receipt"] = str(live)
    live.write_text(json.dumps(report), encoding="utf-8")
    report["conditioning"]["prior_training_receipt_sha256"] = hashlib.sha256(
        live.read_bytes()
    ).hexdigest()
    assert not evaluation_replay_is_content_bound(report, live)


def test_latent_ordered_plan_gate_requires_semantic_body_gain(tmp_path: Path) -> None:
    directories: dict[str, Path] = {}
    for name in ("body_only", "semantic", "shuffled", "dropout"):
        directory = tmp_path / name
        directory.mkdir()
        directories[name] = directory
        config = {
            "seed": 1,
            "model": {"d_model": 32},
            "tokenization": {"target_mode": "body_tokens"},
            "training": {"sft_target_token_positions": 100},
            "evaluation": {"holdout_family_count": 24},
        }
        if name != "body_only":
            config["model"].update(
                {
                    "semantic_plan_feature_count": 48,
                    "semantic_plan_separator_token_id": 2,
                    "semantic_plan_bottleneck_dim": 8,
                }
            )
            config["semantic_plan_training"] = {
                "enabled": True,
                "label_mode": name,
                "auxiliary_loss_weight": 0.25,
                "target": "ordered_plan_slot_token_field",
                "ordered_slot_count": 8,
            }
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
        candidates = directory / "candidates.jsonl"
        candidates.write_text("{}\n", encoding="utf-8")
        report = {
            "architecture": {
                "parameter_count": 100 if name == "body_only" else 120,
                "semantic_plan_head": {
                    "feature_count": 0 if name == "body_only" else 48,
                    "feature_contract_sha256": "" if name == "body_only" else "field-hash",
                },
            },
            "stage": {
                "sft_example_count": 50,
                "unique_body_target_positions": 500,
                "unique_sft_body_count": 40,
                "train_holdout_family_overlap_count": 0,
                "train_eval_prompt_overlap_count": 0,
                "train_eval_body_overlap_count": 0,
                "unique_semantic_eval_task_count": 24,
                "holdout_families": [f"family-{index}" for index in range(24)],
            },
            "training": {
                "complete": True,
                "eval_loss_after": 1.0,
                "semantic_plan_eval_after": {
                    "micro_f1": {"body_only": 0.0, "semantic": 0.8, "shuffled": 0.4, "dropout": 0.0}[name]
                },
                "phases": [{"optimizer_body_positions_consumed": 100}],
            },
            "summary": {
                "model_only_passed_task_count": 1 if name == "semantic" else 0,
                "candidate_task_count": 8 if name == "semantic" else 7,
                "candidate_count": 12 if name == "semantic" else 10,
            },
            "private_verifier": {
                "private_verification": {
                    "mean_verification_reward": 0.6 if name == "semantic" else 0.5
                }
            },
            "decode": {"runtime_ms": 10},
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        }
        (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
        (directory / "integrity.json").write_text(
            json.dumps(
                {
                    "source": str(candidates),
                    "trigger_state": "GREEN",
                    "summary": {"candidate_count": 1, "integrity_mismatch_count": 0},
                }
            ),
            encoding="utf-8",
        )
        (directory / "blind_audit.json").write_text(
            json.dumps({"trigger_state": "GREEN", "summary": {"invalid_claim_count": 0}}),
            encoding="utf-8",
        )

    adopted = audit_latent_ordered_plan_ablation(directories)
    assert adopted["state"] == "GREEN"
    assert adopted["adoption_state"] == "ADOPTED"

    for name in ("semantic", "shuffled", "dropout"):
        config_path = directories[name] / "config.json"
        config = json.loads(config_path.read_text())
        config["model"].update(
            {
                "semantic_plan_slot_count": 8,
                "semantic_plan_conditioning_mode": "slot_attention",
                "semantic_plan_probability_mode": "factorized_step",
                "semantic_plan_factor_group_sizes": [1, 2, 3],
            }
        )
        config["semantic_plan_training"].update(
            {
                "loss_mode": "factorized_step_categorical",
                "target": "ordered_plan_step_factor_field",
            }
        )
        config_path.write_text(json.dumps(config), encoding="utf-8")
    plan_labels = np.zeros((24, 48), dtype=np.float32)
    for row in range(24):
        for slot, code in ((0, row % 6), (1, row // 6)):
            offset = slot * 6
            plan_labels[row, offset] = 1.0
            plan_labels[row, offset + 1 + code % 2] = 1.0
            plan_labels[row, offset + 3 + code // 2] = 1.0
    stage_dir = directories["semantic"] / "stage"
    stage_dir.mkdir()
    np.savez_compressed(stage_dir / "stage_arrays_v1.npz", eval_plan_labels=plan_labels)
    slot_adopted = audit_slot_ordered_plan_ablation(directories)
    assert slot_adopted["state"] == "GREEN", slot_adopted["hard_gaps"]
    assert slot_adopted["adoption_state"] == "ADOPTED"

    shuffled_config_path = directories["shuffled"] / "config.json"
    shuffled_config = json.loads(shuffled_config_path.read_text())
    shuffled_config["model"]["semantic_plan_slot_count"] = 7
    shuffled_config_path.write_text(json.dumps(shuffled_config), encoding="utf-8")
    invalid_slots = audit_slot_ordered_plan_ablation(directories)
    assert invalid_slots["state"] == "RED"
    assert any(
        row["kind"] == "slot_ordered_plan_conditioning_contract_matches_failed"
        for row in invalid_slots["hard_gaps"]
    )
    shuffled_config["model"]["semantic_plan_slot_count"] = 8
    shuffled_config_path.write_text(json.dumps(shuffled_config), encoding="utf-8")

    semantic = directories["semantic"] / "report.json"
    value = json.loads(semantic.read_text())
    value["summary"]["model_only_passed_task_count"] = 0
    semantic.write_text(json.dumps(value), encoding="utf-8")
    rejected = audit_slot_ordered_plan_ablation(directories)
    assert rejected["adoption_state"] == "NOT_ADOPTED"
    assert "no_family_disjoint_verifier_pass_gain" in rejected[
        "adoption_rejection_reasons"
    ]

def test_batched_beam_advance_matches_serial_cached_model_calls() -> None:
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(13)
    config = CausalTransformerConfig(vocab_size=64, d_model=32, num_layers=2, num_heads=4, num_kv_heads=2, ff_dim=64)
    model = build_model(config, mx=mx, nn=nn)
    prefix = mx.array([[1, 2, 3]], dtype=mx.int32)
    logits, cache = model(prefix)
    mx.eval(logits, *cache_arrays(cache))
    beam = {"tokens": ["x"], "score": -0.5, "cache": cache, "logits": logits[0, -1]}
    specs = [
        {"beam": beam, "local_id": 4, "token": "a", "log_probability": -0.1},
        {"beam": beam, "local_id": 5, "token": "b", "log_probability": -0.2},
    ]
    batched = batched_beam_advance(model, specs, target_offset=10, mx=mx)
    for index, spec in enumerate(specs):
        serial_logits, serial_cache = model(
            mx.array([[10 + spec["local_id"]]], dtype=mx.int32), cache
        )
        mx.eval(serial_logits, *cache_arrays(serial_cache))
        assert bool(mx.allclose(batched[index]["logits"], serial_logits[0, -1], atol=1e-5))
        for (batch_key, batch_value), (serial_key, serial_value) in zip(
            batched[index]["cache"], serial_cache
        ):
            assert bool(mx.allclose(batch_key, serial_key, atol=1e-5))
            assert bool(mx.allclose(batch_value, serial_value, atol=1e-5))


def test_hierarchical_beam_ranks_body_conditionally() -> None:
    body_better = {
        "tokens": ["plan-a", "body-a", "body-b"],
        "score": -10.0,
        "plan_score": -4.0,
        "plan_token_count": 1,
    }
    flat_better = {
        "tokens": ["plan-b", "body-a", "body-b"],
        "score": -9.0,
        "plan_score": -1.0,
        "plan_token_count": 1,
    }
    assert hierarchical_beam_rank_score(body_better, 0.0, 0.0) == -6.0
    assert hierarchical_beam_rank_score(flat_better, 0.0, 0.0) == -8.0
    assert prune_active_beams(
        [flat_better, body_better],
        limit=1,
        length_penalty=0.0,
        hierarchical=True,
        plan_score_weight=0.0,
    ) == [body_better]


def test_executable_state_role_lookup_has_matched_semantic_and_hash_density() -> None:
    base = {
        "model": {"state_memory_slots": len(EXECUTABLE_STATE_ROLES)},
        "tokenization": {"shared_source_target_vocabulary": False},
    }
    source_vocab = {"walk": 0, "return": 1, "value": 2}
    target_vocab = {"NAME:for": 0, "NAME:return": 1, "OP:+=": 2}
    semantic = executable_state_role_lookup(
        {**base, "model": {**base["model"], "state_memory_mode": "semantic_roles"}},
        source_vocab,
        target_vocab,
    )
    hashed = executable_state_role_lookup(
        {**base, "model": {**base["model"], "state_memory_mode": "hash_control"}},
        source_vocab,
        target_vocab,
    )
    assert semantic is not None and hashed is not None
    assert semantic.shape == hashed.shape
    assert np.array_equal(semantic.sum(axis=1), hashed.sum(axis=1))
    assert not np.array_equal(semantic, hashed)
    assert executable_state_token_roles("NAME:return") == ("return_closure",)
    assert "traversal" in executable_state_token_roles("walk")
    assert "state_update" in executable_state_token_roles("OP:+=")


def test_executable_state_memory_is_causal_and_cache_partition_invariant() -> None:
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(29)
    slots = len(EXECUTABLE_STATE_ROLES)
    lookup = np.zeros((64, slots), dtype=np.float32)
    for token_id in range(64):
        lookup[token_id, token_id % slots] = 1.0
    config = CausalTransformerConfig(
        vocab_size=64,
        d_model=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ff_dim=64,
        state_memory_slots=slots,
        state_memory_chunk_size=2,
        state_memory_local_window=4,
        state_memory_mode="semantic_roles",
    )
    model = build_model(config, mx=mx, nn=nn, state_role_lookup=lookup)
    sequence_a = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    sequence_b = mx.array([[1, 2, 3, 4, 9]], dtype=mx.int32)
    logits_a, _ = model(sequence_a)
    logits_b, _ = model(sequence_b)
    mx.eval(logits_a, logits_b)
    assert bool(mx.allclose(logits_a[:, :4], logits_b[:, :4], atol=1e-5))

    prefix_logits, cache = model(sequence_a[:, :3])
    cached_logits, cached_state = model(sequence_a[:, 3:], cache)
    full_logits, full_state = model(sequence_a)
    mx.eval(prefix_logits, cached_logits, full_logits, *cache_arrays(cached_state), *cache_arrays(full_state))
    assert bool(mx.allclose(cached_logits, full_logits[:, 3:], atol=1e-4))
    for cached_layer, full_layer in zip(cached_state, full_state):
        for cached_value, full_value in zip(cached_layer, full_layer):
            assert bool(mx.allclose(cached_value, full_value, atol=1e-4))


def test_structured_and_hash_state_models_are_parameter_matched() -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.utils as mlx_utils

    slots = len(EXECUTABLE_STATE_ROLES)
    lookup = np.eye(slots, dtype=np.float32)[np.arange(64) % slots]
    common = dict(
        vocab_size=64,
        d_model=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ff_dim=64,
        state_memory_slots=slots,
        state_memory_chunk_size=2,
        state_memory_local_window=4,
    )
    semantic = build_model(
        CausalTransformerConfig(**common, state_memory_mode="semantic_roles"),
        mx=mx,
        nn=nn,
        state_role_lookup=lookup,
    )
    control = build_model(
        CausalTransformerConfig(**common, state_memory_mode="hash_control"),
        mx=mx,
        nn=nn,
        state_role_lookup=lookup,
    )
    semantic_count = sum(value.size for _name, value in mlx_utils.tree_flatten(semantic.parameters()))
    control_count = sum(value.size for _name, value in mlx_utils.tree_flatten(control.parameters()))
    assert semantic_count == control_count


def test_role_shuffle_changes_reads_before_any_state_write_commits() -> None:
    import mlx.core as mx
    import mlx.nn as nn

    slots = len(EXECUTABLE_STATE_ROLES)
    lookup = np.eye(slots, dtype=np.float32)[np.arange(64) % slots]
    common = dict(
        vocab_size=64,
        d_model=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ff_dim=64,
        state_memory_slots=slots,
        state_memory_chunk_size=4,
        state_memory_local_window=8,
        state_memory_mode="semantic_roles",
        state_memory_read_policy="role_dependency",
    )
    mx.random.seed(41)
    normal = build_model(
        CausalTransformerConfig(**common, state_memory_ablation="none"),
        mx=mx,
        nn=nn,
        state_role_lookup=lookup,
    )
    mx.random.seed(41)
    shuffled = build_model(
        CausalTransformerConfig(**common, state_memory_ablation="shuffle"),
        mx=mx,
        nn=nn,
        state_role_lookup=lookup,
    )
    tokens = mx.array([[1, 2]], dtype=mx.int32)
    normal_logits, _ = normal(tokens)
    shuffled_logits, _ = shuffled(tokens)
    mx.eval(normal_logits, shuffled_logits)
    assert not bool(mx.allclose(normal_logits, shuffled_logits, atol=1e-6))


def test_visible_eval_source_does_not_read_solution_or_tests() -> None:
    row = {
        "prompt": "Return the input length.",
        "entry_point": "length",
        "solution_body": "SECRET_SOLUTION",
        "tests": "SECRET_TEST",
        "decoder_contract": {"visible_arg_count_hint": 1},
    }
    visible = visible_eval_source(row)
    assert visible == (
        "Return the input length.\n"
        "signature def solve(data=None, other=None, *extra):"
    )
    assert "SECRET" not in visible
    changed_hidden = {
        **row,
        "solution_body": "DIFFERENT_SECRET",
        "tests": "DIFFERENT_TEST",
        "concept_residual_label": "answer_identifying_family",
        "decoder_contract": {
            "visible_arg_count_hint": 1,
            "return_shape": "forbidden_hidden_shape",
            "required_constructs": ["forbidden_hidden_construct"],
        },
    }
    assert visible_eval_source(changed_hidden) == visible


def test_empty_body_is_rejected_instead_of_receiving_fallback() -> None:
    with pytest.raises(ValueError, match="empty body"):
        render_visible_signature("def solve(data):", "")


def test_family_disjoint_split_is_deterministic_and_nonempty() -> None:
    config = json.loads((ROOT / "configs" / "standard_causal_transformer_survival.json").read_text())
    rows_a, families_a = select_family_disjoint_eval(config)
    rows_b, families_b = select_family_disjoint_eval(config)
    assert families_a == families_b
    assert len(families_a) == config["evaluation"]["holdout_family_count"]
    assert len(rows_a) == len(rows_b) == len(families_a) * config["evaluation"]["rows_per_family"]
    assert {row["concept_residual_label"] for row in rows_a} == set(families_a)
    assert len({(row["prompt"], row["solution_body"]) for row in rows_a}) == len(rows_a) == 24
    target_limit = config["tokenization"]["max_target_tokens"] - 2
    from neural_seed_token_decoder_support import body_tokens

    assert max(len(body_tokens(row["solution_body"])) for row in rows_a) <= target_limit
    assert config["evaluation"]["decode_max_target_tokens"] >= max(
        len(body_tokens(row["solution_body"])) for row in rows_a
    )


def test_config_rejects_fallback_or_external_inference() -> None:
    config = json.loads((ROOT / "configs" / "standard_causal_transformer_survival.json").read_text())
    validate_config(config)
    bad = json.loads(json.dumps(config))
    bad["boundaries"]["fallback_returns_allowed"] = True
    with pytest.raises(ValueError, match="fallback"):
        validate_config(bad)

    thin = json.loads(json.dumps(config))
    thin["evaluation"]["holdout_family_count"] = 6
    thin["evaluation"]["rows_per_family"] = 4
    with pytest.raises(ValueError, match="24 distinct families"):
        validate_config(thin)

    missing_vocab_mode = json.loads(json.dumps(config))
    del missing_vocab_mode["tokenization"]["shared_source_target_vocabulary"]
    with pytest.raises(ValueError, match="explicitly boolean"):
        validate_config(missing_vocab_mode)

    task_named = json.loads(json.dumps(config))
    task_named["tokenization"]["canonical_model_signature_name"] = "task_name"
    with pytest.raises(ValueError, match="must remain solve"):
        validate_config(task_named)

    invalid_plan = json.loads(json.dumps(config))
    invalid_plan["tokenization"]["target_mode"] = "typed_semantic_ir_plan_body_tokens_v1"
    invalid_plan["tokenization"]["semantic_plan_max_tokens"] = 4
    with pytest.raises(ValueError, match="semantic plan token budget"):
        validate_config(invalid_plan)

    plan_head = json.loads(json.dumps(config))
    import semantic_ir

    plan_head["model"]["semantic_plan_feature_count"] = len(
        semantic_ir.plan_obligation_features()
    )
    plan_head["model"]["semantic_plan_separator_token_id"] = 2
    plan_head["semantic_plan_training"] = {
        "enabled": True,
        "label_mode": "semantic",
        "auxiliary_loss_weight": 0.2,
        "shuffle_seed": 19,
    }
    validate_config(plan_head)
    mismatched_features = json.loads(json.dumps(plan_head))
    mismatched_features["model"]["semantic_plan_feature_count"] -= 1
    with pytest.raises(ValueError, match="fixed registered IR contract"):
        validate_config(mismatched_features)
    missing_control = json.loads(json.dumps(plan_head))
    missing_control["semantic_plan_training"]["label_mode"] = "none"
    with pytest.raises(ValueError, match="semantic, shuffled, or dropout"):
        validate_config(missing_control)

    ordered = json.loads(json.dumps(config))
    ordered["tokenization"].update(
        {
            "target_mode": "typed_semantic_ir_plan_body_tokens_v1",
            "semantic_plan_max_tokens": 32,
            "sequence_plan_reserve_tokens": 33,
        }
    )
    ordered["ordered_plan_training"] = {
        "label_mode": "semantic",
        "plan_loss_weight": 0.25,
        "shuffle_seed": 31,
    }
    validate_config(ordered)
    invalid_ordered = json.loads(json.dumps(ordered))
    invalid_ordered["ordered_plan_training"]["label_mode"] = "random"
    with pytest.raises(ValueError, match="semantic, shuffled, or dropout"):
        validate_config(invalid_ordered)


def test_standard_transformer_plan_body_target_is_learned_and_closed_vocab() -> None:
    import semantic_ir
    from neural_seed_token_decoder_rendering import PLAN_BODY_START_TOKEN
    from neural_seed_token_decoder_support import body_tokens as tokenize_body

    config = json.loads((ROOT / "configs" / "standard_causal_transformer_survival.json").read_text())
    config["tokenization"]["target_mode"] = semantic_ir.PLAN_BODY_TARGET_MODE
    config["tokenization"]["semantic_plan_max_tokens"] = 32
    config["tokenization"]["sequence_plan_reserve_tokens"] = 33
    body = "out = []\nfor value in data:\n    out.append(value + 1)\nreturn out"
    tokens = training_target_tokens(body, config)
    assert tokens[0] == semantic_ir.PLAN_BEGIN
    assert PLAN_BODY_START_TOKEN in tokens
    assert tokens[tokens.index(PLAN_BODY_START_TOKEN) + 1 :] == tokenize_body(body)
    assert generation_prefix_complete(tokens, target_mode=semantic_ir.PLAN_BODY_TARGET_MODE)

    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "NAME:out": 4}
    summary = extend_target_vocab_for_mode(config, vocab)
    assert summary["target_independent_closed_protocol"] is True
    assert summary["added_token_count"] == len(semantic_ir.plan_protocol_tokens()) + 1
    assert set(semantic_ir.plan_protocol_tokens()) <= set(vocab)
    assert PLAN_BODY_START_TOKEN in vocab


def test_ordered_plan_controls_preserve_token_mass_and_derange_rows() -> None:
    import semantic_ir

    config = json.loads((ROOT / "configs" / "standard_causal_transformer_survival.json").read_text())
    config["tokenization"].update(
        {
            "target_mode": semantic_ir.PLAN_BODY_TARGET_MODE,
            "semantic_plan_max_tokens": 32,
            "sequence_plan_reserve_tokens": 33,
        }
    )
    config["ordered_plan_training"] = {
        "label_mode": "semantic",
        "plan_loss_weight": 0.25,
        "shuffle_seed": 41,
    }
    examples = [
        {"body": "return data"},
        {"body": "out = []\nfor item in data:\n    out.append(item)\nreturn out"},
        {"body": "total = 0\nfor item in data:\n    total += item\nreturn total"},
    ]
    semantic, semantic_receipt = prepare_ordered_plan_sequences(
        examples, config, mode="semantic"
    )
    shuffled, shuffled_receipt = prepare_ordered_plan_sequences(
        examples, config, mode="shuffled"
    )
    dropped, dropped_receipt = prepare_ordered_plan_sequences(
        examples, config, mode="dropout"
    )
    assert semantic_receipt["token_count"] == shuffled_receipt["token_count"]
    assert semantic_receipt["token_count"] == dropped_receipt["token_count"]
    assert shuffled_receipt["fixed_point_count"] == 0
    assert all(left != right for left, right in zip(semantic, shuffled))
    assert all(len(left) == len(right) for left, right in zip(semantic, dropped))
    assert semantic_receipt["plan_sha256"] != shuffled_receipt["plan_sha256"]
    assert semantic_receipt["plan_sha256"] != dropped_receipt["plan_sha256"]


def test_ordered_plan_stage_counts_body_exposure_separately() -> None:
    import semantic_ir
    from neural_seed_token_decoder_support import body_tokens as tokenize_body

    config = json.loads((ROOT / "configs" / "standard_causal_transformer_survival.json").read_text())
    config["tokenization"].update(
        {
            "target_mode": semantic_ir.PLAN_BODY_TARGET_MODE,
            "semantic_plan_max_tokens": 32,
            "sequence_plan_reserve_tokens": 33,
            "max_sequence_tokens": 128,
            "max_source_tokens": 32,
            "max_target_tokens": 96,
        }
    )
    config["ordered_plan_training"] = {
        "label_mode": "semantic",
        "plan_loss_weight": 0.25,
        "shuffle_seed": 43,
    }
    body = "out = []\nfor item in data:\n    out.append(item)\nreturn out"
    source = "Copy each item into a new list.\nsignature def solve(data):"
    examples = [{"source_text": source, "body": body, "sampling_weight": 1.0}]
    source_vocab = {
        token: index
        for index, token in enumerate(
            ["<pad>", "<bos>", "<eos>", "<unk>", *dict.fromkeys(source_tokens(source))]
        )
    }
    target_stream = training_target_tokens(body, config)
    target_vocab = {
        token: index
        for index, token in enumerate(
            ["<pad>", "<bos>", "<eos>", "<unk>", *dict.fromkeys(target_stream)]
        )
    }
    _x, _y, all_mask, _weights, _labels, body_mask, receipt = (
        encode_sft_training_examples(
            config,
            examples,
            source_vocab,
            target_vocab,
            ordered_plan_mode="semantic",
        )
    )
    assert int(body_mask.sum()) == len(tokenize_body(body)) + 1
    assert int(all_mask.sum()) > int(body_mask.sum())
    assert receipt["encoded_body_position_count"] == int(body_mask.sum())
    assert receipt["encoded_plan_position_count"] == int(
        np.maximum(all_mask - body_mask, 0.0).sum()
    )


def test_matched_target_mode_comparison_rejects_loss_only_plan_improvement() -> None:
    common_config = {
        "seed": 1,
        "sources": {"private": "x"},
        "model": {"d_model": 8},
        "training": {"pretrain_target_token_positions": 10, "sft_target_token_positions": 20},
        "evaluation": {"holdout_family_count": 24},
        "preference": {"max_train_tasks": 8},
        "boundaries": {"public_training_rows": 0},
        "tokenization": {
            "target_mode": "body_tokens",
            "semantic_plan_max_tokens": 0,
            "max_source_tokens": 32,
        },
    }

    def report(mode: str, loss: float, candidates: int, tasks: int, reward: float) -> dict:
        return {
            "stage": {"target_mode": mode},
            "architecture": {"parameter_count": 100},
            "training": {"complete": True, "eval_loss_after": loss},
            "summary": {"candidate_count": candidates, "candidate_task_count": tasks},
            "decode": {"runtime_ms": 10},
            "private_verifier": {
                "summary": {"passed_task_count": 0},
                "private_verification": {"mean_verification_reward": reward},
            },
            "runtime_ms": 20,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        }

    plan_config = json.loads(json.dumps(common_config))
    plan_config["tokenization"]["target_mode"] = "typed_semantic_ir_plan_body_tokens_v1"
    plan_config["tokenization"]["semantic_plan_max_tokens"] = 32
    control_config = json.loads(json.dumps(common_config))
    clean_integrity = {"summary": {"integrity_mismatch_count": 0}}
    result = compare_target_mode_canaries(
        report("typed_semantic_ir_plan_body_tokens_v1", 1.6, 4, 2, 0.2),
        report("body_tokens", 1.8, 8, 4, 0.4),
        plan_config=plan_config,
        control_config=control_config,
        plan_integrity=clean_integrity,
        control_integrity=clean_integrity,
    )
    assert result["trigger_state"] == "GREEN"
    assert result["adoption_state"] == "NOT_ADOPTED"
    assert result["deltas"]["candidate_task_count"] == -2
    assert "no_verifier_pass_gain" in result["adoption_rejection_reasons"]


def test_matched_target_mode_comparison_rejects_non_target_config_drift() -> None:
    config = {
        "seed": 1,
        "sources": {"private": "x"},
        "tokenization": {"target_mode": "body_tokens", "max_source_tokens": 32},
        "model": {"d_model": 8},
        "training": {"pretrain_target_token_positions": 10},
        "evaluation": {"holdout_family_count": 24},
        "preference": {"max_train_tasks": 8},
        "boundaries": {"public_training_rows": 0},
    }
    plan_config = json.loads(json.dumps(config))
    plan_config["tokenization"]["target_mode"] = "typed_semantic_ir_plan_body_tokens_v1"
    control_config = json.loads(json.dumps(config))
    control_config["tokenization"]["max_source_tokens"] = 64
    plan_report = {
        "stage": {"target_mode": "typed_semantic_ir_plan_body_tokens_v1"},
        "architecture": {"parameter_count": 100},
        "training": {"complete": True, "eval_loss_after": 1.0},
        "summary": {"candidate_count": 1, "candidate_task_count": 1},
        "decode": {"runtime_ms": 1},
        "private_verifier": {
            "summary": {"passed_task_count": 1},
            "private_verification": {"mean_verification_reward": 1.0},
        },
        "runtime_ms": 1,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    control_report = json.loads(json.dumps(plan_report))
    control_report["stage"]["target_mode"] = "body_tokens"
    integrity = {"summary": {"integrity_mismatch_count": 0}}
    result = compare_target_mode_canaries(
        plan_report,
        control_report,
        plan_config=plan_config,
        control_config=control_config,
        plan_integrity=integrity,
        control_integrity=integrity,
    )
    assert result["trigger_state"] == "RED"
    assert result["matched_checks"]["tokenization_except_target_mode"] is False


def test_target_mode_gate_replays_artifact_bindings_and_adoption(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"bound")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    artifact_names = {
        "plan_report",
        "plan_config",
        "plan_candidates",
        "plan_checkpoint",
        "plan_integrity",
        "control_report",
        "control_config",
        "control_candidates",
        "control_checkpoint",
        "control_integrity",
    }
    comparison = {
        "policy": "project_theseus_standard_causal_target_mode_matched_comparison_v1",
        "trigger_state": "GREEN",
        "adoption_state": "NOT_ADOPTED",
        "adoption_rejection_reasons": ["no_verifier_pass_gain"],
        "boundaries_clean": True,
        "matched_checks": {"seed": True, "target_modes_expected": True},
        "plan": {
            "passed_task_count": 0,
            "candidate_task_count": 2,
            "mean_verification_reward": 0.2,
            "integrity_mismatch_count": 0,
        },
        "control": {
            "passed_task_count": 0,
            "candidate_task_count": 4,
            "mean_verification_reward": 0.4,
            "integrity_mismatch_count": 0,
        },
        "deltas": {"candidate_task_count": -2},
        "artifacts": {
            name: {"path": str(artifact), "sha256": digest, "bytes": 5}
            for name in artifact_names
        },
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(comparison), encoding="utf-8")
    audit = audit_target_mode_comparison(path)
    assert audit["state"] == "GREEN"
    assert audit["adoption_state"] == "NOT_ADOPTED"
    assert all(item["matches"] for item in audit["receipt"]["artifact_receipts"].values())

    comparison["adoption_state"] = "ADOPTED"
    path.write_text(json.dumps(comparison), encoding="utf-8")
    invalid = audit_target_mode_comparison(path)
    assert invalid["state"] == "RED"
    assert any(
        gap["kind"] == "target_mode_adoption_decision_mismatch"
        for gap in invalid["hard_gaps"]
    )


def test_sft_contract_admission_audit_retains_behavior_negative_result(tmp_path: Path) -> None:
    filtered_config = {
        "policy": "candidate",
        "model": {"d_model": 32},
        "training": {"positions": 100},
        "sft_contract_admission": {
            "require_self_contained_body": True,
            "private_sampling_probability_target": 0.25,
        },
    }
    control_config = {key: value for key, value in filtered_config.items() if key != "sft_contract_admission"}
    filtered_config_path = tmp_path / "filtered-config.json"
    control_config_path = tmp_path / "control-config.json"
    filtered_config_path.write_text(json.dumps(filtered_config), encoding="utf-8")
    control_config_path.write_text(json.dumps(control_config), encoding="utf-8")

    def report(config_path: Path, *, candidates: int, tasks: int, reward: float) -> dict[str, object]:
        return {
            "artifacts": {"config": str(config_path)},
            "stage": {
                "sft_contract_admission": {
                    "target_body_fields_added_to_model_source": 0,
                    "heldout_rows_read_by_filter": 0,
                }
            },
            "summary": {
                "model_only_passed_task_count": 0,
                "candidate_task_count": tasks,
                "candidate_count": candidates,
            },
            "private_verifier": {"private_verification": {"mean_verification_reward": reward}},
            "training": {"eval_loss_after": 1.0},
            "decode": {"runtime_ms": 100},
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        }

    filtered_report_path = tmp_path / "filtered-report.json"
    control_report_path = tmp_path / "control-report.json"
    integrity_path = tmp_path / "integrity.json"
    blind_path = tmp_path / "blind.json"
    filtered_report_path.write_text(
        json.dumps(report(filtered_config_path, candidates=3, tasks=2, reward=0.2)),
        encoding="utf-8",
    )
    control_report_path.write_text(
        json.dumps(report(control_config_path, candidates=5, tasks=4, reward=0.4)),
        encoding="utf-8",
    )
    integrity_path.write_text(
        json.dumps({"trigger_state": "GREEN", "summary": {"integrity_mismatch_count": 0}}),
        encoding="utf-8",
    )
    blind_path.write_text(
        json.dumps({"trigger_state": "GREEN", "summary": {"invalid_claim_count": 0}}),
        encoding="utf-8",
    )
    audit = audit_sft_contract_admission(
        report_path=filtered_report_path,
        integrity_path=integrity_path,
        blind_audit_path=blind_path,
        control_report_path=control_report_path,
    )
    assert audit["state"] == "GREEN"
    assert audit["adoption_state"] == "NOT_ADOPTED"
    assert audit["deltas"]["candidate_task_count"] == -2
    assert audit["receipt"]["matched_config_except_contract_admission"] is True


def test_sft_contract_admission_audit_is_not_run_when_local_evidence_is_absent(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.json"
    audit = audit_sft_contract_admission(
        report_path=missing,
        integrity_path=missing,
        blind_audit_path=missing,
        control_report_path=missing,
    )
    assert audit["state"] == "NOT_RUN"
    assert audit["hard_gaps"] == []


def test_state_memory_ablation_requires_behavior_gain_and_both_causal_controls(
    tmp_path: Path,
) -> None:
    modes = {
        "body_only": ("none", "none"),
        "semantic": ("semantic_roles", "none"),
        "hash_control": ("hash_control", "none"),
        "zero": ("semantic_roles", "zero"),
        "shuffle": ("semantic_roles", "shuffle"),
    }
    metrics = {
        "body_only": (0, 20, 70, 0.40, 1.7),
        "semantic": (1, 24, 81, 0.50, 1.0),
        "hash_control": (0, 21, 72, 0.41, 1.6),
        "zero": (0, 23, 76, 0.43, 1.7),
        "shuffle": (0, 23, 78, 0.44, 1.2),
    }
    arm_dirs: dict[str, Path] = {}
    semantic_checkpoint = tmp_path / "semantic.npz"
    semantic_checkpoint.write_bytes(b"semantic")
    for name, (mode, ablation) in modes.items():
        directory = tmp_path / name
        directory.mkdir()
        arm_dirs[name] = directory
        model = {"d_model": 32}
        if name != "body_only":
            model.update(
                {
                    "state_memory_slots": 8,
                    "state_memory_chunk_size": 4,
                    "state_memory_local_window": 8,
                    "state_memory_mode": mode,
                    "state_memory_ablation": ablation,
                }
            )
        (directory / "config.json").write_text(
            json.dumps({"seed": 1, "model": model, "training": {"positions": 100}}),
            encoding="utf-8",
        )
        candidates = directory / "candidates.jsonl"
        candidates.write_text("{}\n", encoding="utf-8")
        checkpoint = semantic_checkpoint if name in {"semantic", "zero", "shuffle"} else directory / "model.npz"
        if not checkpoint.exists():
            checkpoint.write_bytes(name.encode())
        passed, tasks, count, reward, loss = metrics[name]
        (directory / "report.json").write_text(
            json.dumps(
                {
                    "artifacts": {"checkpoint": str(checkpoint)},
                    "architecture": {"parameter_count": 200 if name != "body_only" else 150},
                    "stage": {"stage_signature": "same"},
                    "training": {
                        "eval_loss_after": loss,
                        "phases": [{"optimizer_body_positions_consumed": 100}],
                    },
                    "summary": {
                        "model_only_passed_task_count": passed,
                        "candidate_task_count": tasks,
                        "candidate_count": count,
                    },
                    "private_verifier": {
                        "private_verification": {"mean_verification_reward": reward}
                    },
                    "decode": {"runtime_ms": 10},
                    "public_training_rows_written": 0,
                    "external_inference_calls": 0,
                    "fallback_return_count": 0,
                }
            ),
            encoding="utf-8",
        )
        (directory / "integrity.json").write_text(
            json.dumps(
                {
                    "source": str(candidates),
                    "trigger_state": "GREEN",
                    "summary": {"candidate_count": 1, "integrity_mismatch_count": 0},
                }
            ),
            encoding="utf-8",
        )
        (directory / "blind_audit.json").write_text(
            json.dumps({"trigger_state": "GREEN", "summary": {"invalid_claim_count": 0}}),
            encoding="utf-8",
        )

    adopted = audit_state_memory_ablation(arm_dirs)
    assert adopted["state"] == "GREEN"
    assert adopted["adoption_state"] == "ADOPTED"
    assert adopted["receipt"]["optimizer_body_positions"] == {name: 100 for name in modes}

    shuffle_report = json.loads((arm_dirs["shuffle"] / "report.json").read_text())
    shuffle_report["training"]["eval_loss_after"] = 0.9
    (arm_dirs["shuffle"] / "report.json").write_text(json.dumps(shuffle_report), encoding="utf-8")
    rejected = audit_state_memory_ablation(arm_dirs)
    assert rejected["state"] == "GREEN"
    assert rejected["adoption_state"] == "NOT_ADOPTED"
    assert "role_shuffle_did_not_causally_degrade" in rejected["adoption_rejection_reasons"]


def test_state_continuation_requires_exact_gain_and_improved_loss(tmp_path: Path) -> None:
    modes = {"body_only": "none", "semantic": "semantic_roles", "hash_control": "hash_control"}
    arm_dirs: dict[str, Path] = {}
    for name, mode in modes.items():
        directory = tmp_path / name
        directory.mkdir()
        arm_dirs[name] = directory
        model = {"d_model": 32}
        if mode != "none":
            model.update(
                {
                    "state_memory_slots": 8,
                    "state_memory_chunk_size": 4,
                    "state_memory_local_window": 8,
                    "state_memory_mode": mode,
                    "state_memory_ablation": "none",
                    "state_memory_read_policy": "unrestricted",
                }
            )
        (directory / "config.json").write_text(
            json.dumps({"seed": 1, "model": model, "training": {"positions": 200}}), encoding="utf-8"
        )
        candidate_path = directory / "candidates.jsonl"
        candidate_path.write_text("{}\n", encoding="utf-8")
        prior_checkpoint = directory / "prior.npz"
        prior_checkpoint.write_bytes(name.encode())
        prior_report = directory / "prior.json"
        prior_report.write_text(
            json.dumps(
                {
                    "artifacts": {"checkpoint": str(prior_checkpoint)},
                    "summary": {"model_only_passed_task_count": 0, "candidate_task_count": 20, "candidate_count": 60},
                    "private_verifier": {"private_verification": {"mean_verification_reward": 0.4}},
                    "training": {"eval_loss_after": 1.5},
                    "decode": {"runtime_ms": 20},
                }
            ),
            encoding="utf-8",
        )
        passed = 1 if name == "semantic" else 0
        loss = 1.2 if name == "semantic" else 1.4
        report = {
            "artifacts": {"prior_training_receipt": str(prior_report)},
            "architecture": {"parameter_count": 200 if mode != "none" else 150},
            "conditioning": {"resume_base_checkpoint_sha256": hashlib.sha256(prior_checkpoint.read_bytes()).hexdigest()},
            "stage": {"stage_signature": "same"},
            "training": {
                "eval_loss_after": loss,
                "phases": [
                    {"phase": "prompt_signature_body_sft", "optimizer_body_positions_consumed": 100},
                    {"phase": "prompt_signature_body_sft_continuation", "optimizer_body_positions_consumed": 100},
                ],
            },
            "summary": {"model_only_passed_task_count": passed, "candidate_task_count": 21, "candidate_count": 70},
            "private_verifier": {"private_verification": {"mean_verification_reward": 0.5}},
            "decode": {"runtime_ms": 10},
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        }
        (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
        (directory / "integrity.json").write_text(
            json.dumps(
                {
                    "source": str(candidate_path),
                    "trigger_state": "GREEN",
                    "summary": {"candidate_count": 1, "integrity_mismatch_count": 0},
                }
            ),
            encoding="utf-8",
        )
        (directory / "blind_audit.json").write_text(
            json.dumps({"trigger_state": "GREEN", "summary": {"invalid_claim_count": 0}}), encoding="utf-8"
        )

    adopted = audit_state_memory_continuation(arm_dirs)
    assert adopted["state"] == "GREEN"
    assert adopted["adoption_state"] == "ADOPTED"

    semantic_path = arm_dirs["semantic"] / "report.json"
    semantic = json.loads(semantic_path.read_text())
    semantic["training"]["eval_loss_after"] = 1.6
    semantic_path.write_text(json.dumps(semantic), encoding="utf-8")
    rejected = audit_state_memory_continuation(arm_dirs)
    assert rejected["state"] == "GREEN"
    assert rejected["adoption_state"] == "NOT_ADOPTED"
    assert "semantic_heldout_loss_worsened" in rejected["adoption_rejection_reasons"]


def test_teacher_residual_ablation_requires_exact_private_behavior_gain(tmp_path: Path) -> None:
    teacher_gate = tmp_path / "teacher_gate.json"
    teacher_gate.write_text(
        json.dumps({"trigger_state": "GREEN", "distillation_allowed": True}),
        encoding="utf-8",
    )
    provider_audit = tmp_path / "provider_audit.json"
    provider_audit.write_text(
        json.dumps(
            {
                "ok": True,
                "summary": {
                    "teacher_receipt_violations": 0,
                    "teacher_provider_counts": {"codex_cli/gpt-5.6-sol": 2},
                },
            }
        ),
        encoding="utf-8",
    )
    arm_dirs: dict[str, Path] = {}
    for name, state_mode in {"body_only": "none", "semantic": "semantic_roles"}.items():
        directory = tmp_path / name
        directory.mkdir()
        arm_dirs[name] = directory
        model = {"d_model": 32}
        if state_mode != "none":
            model.update(
                {
                    "state_memory_slots": 8,
                    "state_memory_chunk_size": 4,
                    "state_memory_local_window": 8,
                    "state_memory_mode": state_mode,
                    "state_memory_ablation": "none",
                    "state_memory_read_policy": "unrestricted",
                }
            )
        (directory / "config.json").write_text(
            json.dumps(
                {
                    "seed": 1,
                    "model": model,
                    "training": {"positions": 200},
                    "teacher_distillation": {
                        "minimum_code_lm_rows_for_sampling": 2,
                        "teacher_sampling_probability_target": 0.1,
                    },
                }
            ),
            encoding="utf-8",
        )
        candidates = directory / "candidates.jsonl"
        candidates.write_text("{}\n", encoding="utf-8")
        checkpoint = directory / "prior.npz"
        checkpoint.write_bytes(name.encode())
        prior = directory / "prior.json"
        prior.write_text(
            json.dumps(
                {
                    "artifacts": {"checkpoint": str(checkpoint)},
                    "summary": {
                        "model_only_passed_task_count": 0,
                        "candidate_task_count": 20,
                        "candidate_count": 60,
                    },
                    "private_verifier": {"private_verification": {"mean_verification_reward": 0.4}},
                    "training": {"eval_loss_after": 1.5},
                    "decode": {"runtime_ms": 20},
                }
            ),
            encoding="utf-8",
        )
        passed = 1 if name == "semantic" else 0
        report = {
            "artifacts": {"prior_training_receipt": str(prior)},
            "conditioning": {
                "resume_base_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            },
            "stage": {
                "stage_signature": "same",
                "governed_teacher_prompt_pair_count": 2,
                "governed_teacher_unique_body_count": 2,
                "teacher_sampling_probability": 0.1,
                "governed_teacher_source_summary": {"gate_green": True, "tranche_ready": True},
                "train_holdout_family_overlap_count": 0,
                "train_eval_prompt_overlap_count": 0,
                "train_eval_body_overlap_count": 0,
                "governed_teacher_current_holdout_rejected_count": 0,
                "governed_teacher_eval_overlap_rejected_count": 0,
            },
            "training": {
                "eval_loss_after": 1.2 if name == "semantic" else 1.4,
                "phases": [
                    {"phase": "prompt_signature_body_sft", "optimizer_body_positions_consumed": 100},
                    {"phase": "prompt_signature_body_sft_continuation", "optimizer_body_positions_consumed": 100},
                ],
            },
            "summary": {
                "model_only_passed_task_count": passed,
                "candidate_task_count": 21,
                "candidate_count": 70,
            },
            "private_verifier": {"private_verification": {"mean_verification_reward": 0.5}},
            "decode": {"runtime_ms": 10},
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        }
        (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
        (directory / "integrity.json").write_text(
            json.dumps(
                {
                    "source": str(candidates),
                    "trigger_state": "GREEN",
                    "summary": {"candidate_count": 1, "integrity_mismatch_count": 0},
                }
            ),
            encoding="utf-8",
        )
        (directory / "blind_audit.json").write_text(
            json.dumps({"trigger_state": "GREEN", "summary": {"invalid_claim_count": 0}}),
            encoding="utf-8",
        )

    adopted = audit_teacher_residual_ablation(
        arm_dirs,
        teacher_gate_path=teacher_gate,
        provider_audit_path=provider_audit,
    )
    assert adopted["state"] == "GREEN"
    assert adopted["adoption_state"] == "ADOPTED_STATE_SHADOW"

    semantic_path = arm_dirs["semantic"] / "report.json"
    semantic = json.loads(semantic_path.read_text())
    semantic["summary"]["model_only_passed_task_count"] = 0
    semantic["training"]["eval_loss_after"] = 1.6
    semantic_path.write_text(json.dumps(semantic), encoding="utf-8")
    rejected = audit_teacher_residual_ablation(
        arm_dirs,
        teacher_gate_path=teacher_gate,
        provider_audit_path=provider_audit,
    )
    assert rejected["state"] == "GREEN"
    assert rejected["adoption_state"] == "NOT_ADOPTED"
    assert "no_family_disjoint_verifier_pass_gain" in rejected["adoption_rejection_reasons"]


def test_semantic_plan_head_ablation_requires_shuffled_control_and_exact_gain(
    tmp_path: Path,
) -> None:
    arm_dirs: dict[str, Path] = {}
    for name in ("body_only", "semantic", "shuffled"):
        directory = tmp_path / name
        directory.mkdir()
        arm_dirs[name] = directory
        config = {
            "seed": 7,
            "model": {"d_model": 32},
            "training": {"sft_target_token_positions": 100},
        }
        if name != "body_only":
            config["model"].update(
                {
                    "semantic_plan_feature_count": 181,
                    "semantic_plan_separator_token_id": 2,
                }
            )
            config["semantic_plan_training"] = {
                "enabled": True,
                "label_mode": name,
                "auxiliary_loss_weight": 0.2,
                "shuffle_seed": 19,
            }
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (directory / "candidates.jsonl").write_text("{}\n", encoding="utf-8")
        passed = 1 if name == "semantic" else 0
        reward = 0.5 if name == "semantic" else 0.4
        f1 = 0.7 if name == "semantic" else (0.3 if name == "shuffled" else 0.0)
        binary_loss = 0.4 if name == "semantic" else (0.8 if name == "shuffled" else 0.0)
        label_receipt = {
            "label_mode": name if name != "body_only" else "none",
            "row_count": 20 if name != "body_only" else 0,
            "positive_label_count": 50 if name != "body_only" else 0,
            "fixed_point_count": 20 if name == "semantic" else 0,
            "label_sha256": name if name != "body_only" else "",
        }
        report = {
            "architecture": {"parameter_count": 200 if name != "body_only" else 150},
            "stage": {"stage_signature": "same"},
            "training": {
                "phases": [
                    {
                        "phase": "prompt_signature_body_sft",
                        "optimizer_body_positions_consumed": 100,
                        "semantic_plan_labels": label_receipt,
                        "semantic_plan_positive_weights": {
                            "weight_sha256": "matched" if name != "body_only" else ""
                        },
                    }
                ],
                "eval_loss_after": 1.0,
                "semantic_plan_eval_after": (
                    {
                        "source_contract": "prompt_signature_only_before_separator",
                        "target_labels_visible_at_inference": False,
                        "micro_f1": f1,
                        "binary_cross_entropy": binary_loss,
                    }
                    if name != "body_only"
                    else {"state": "NOT_APPLICABLE"}
                ),
            },
            "summary": {
                "model_only_passed_task_count": passed,
                "candidate_task_count": 24,
                "candidate_count": 60,
            },
            "private_verifier": {
                "private_verification": {"mean_verification_reward": reward}
            },
            "decode": {"runtime_ms": 10},
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        }
        (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
        (directory / "integrity.json").write_text(
            json.dumps(
                {
                    "source": str(directory / "candidates.jsonl"),
                    "trigger_state": "GREEN",
                    "summary": {"candidate_count": 1, "integrity_mismatch_count": 0},
                }
            ),
            encoding="utf-8",
        )
        (directory / "blind_audit.json").write_text(
            json.dumps(
                {"trigger_state": "GREEN", "summary": {"invalid_claim_count": 0}}
            ),
            encoding="utf-8",
        )

    adopted = audit_semantic_plan_head_ablation(arm_dirs)
    assert adopted["state"] == "GREEN"
    assert adopted["adoption_state"] == "ADOPTED"
    assert adopted["receipt"]["semantic_label_causal_gain"] is True

    shuffled_path = arm_dirs["shuffled"] / "report.json"
    shuffled = json.loads(shuffled_path.read_text())
    shuffled["training"]["semantic_plan_eval_after"]["micro_f1"] = 0.8
    shuffled["training"]["semantic_plan_eval_after"]["binary_cross_entropy"] = 0.3
    shuffled_path.write_text(json.dumps(shuffled), encoding="utf-8")
    rejected = audit_semantic_plan_head_ablation(arm_dirs)
    assert rejected["state"] == "GREEN"
    assert rejected["adoption_state"] == "NOT_ADOPTED"
    assert "semantic_labels_did_not_beat_shuffled_labels" in rejected[
        "adoption_rejection_reasons"
    ]


def test_candidate_integrity_recomputes_direct_plan_body_trace_instead_of_trusting_flags() -> None:
    import semantic_ir
    from neural_seed_token_decoder_rendering import PLAN_BODY_START_TOKEN
    from neural_seed_token_decoder_support import body_tokens as tokenize_body

    body = "out = []\nfor value in data:\n    out.append(value + 1)\nreturn out"
    code = "def solve(data):\n" + "\n".join(f"    {line}" for line in body.splitlines()) + "\n"
    tokens = [
        *semantic_ir.body_to_plan_tokens(body, max_tokens=32),
        PLAN_BODY_START_TOKEN,
        *tokenize_body(body),
    ]
    row = {
        "candidate_generation_mode": "direct_decoder_only_causal_semantic_plan_body_tokens",
        "candidate_source": "standard_causal_transformer_survival",
        "substrate_arm": "transformer_hybrid_survival",
        "code": code,
        "decoded_target_tokens": tokens,
        "decoded_token_sha256": hashlib.sha256(" ".join(tokens).encode()).hexdigest(),
        "provenance": {"candidate_family": "structural_adapter"},
    }
    verified = recompute_candidate_integrity(row)
    assert verified["recomputed_candidate_family"] == "transformer_hybrid"
    assert verified["direct_plan_body_trace"]["valid"] is True

    corrupted = json.loads(json.dumps(row))
    corrupted["decoded_target_tokens"][-2] = "NAME:wrong"
    corrupted["decoded_token_sha256"] = hashlib.sha256(
        " ".join(corrupted["decoded_target_tokens"]).encode()
    ).hexdigest()
    rejected = recompute_candidate_integrity(corrupted)
    assert rejected["recomputed_candidate_family"] == "structural_adapter"
    assert rejected["direct_plan_body_trace"]["valid"] is False
    assert "decoded_body_trace_code_mismatch" in rejected["direct_plan_body_trace"]["faults"]

    hierarchical = json.loads(json.dumps(row))
    hierarchical["candidate_generation_mode"] = (
        "direct_decoder_only_hierarchical_semantic_plan_body_tokens"
    )
    hierarchical_verified = recompute_candidate_integrity(hierarchical)
    assert hierarchical_verified["direct_plan_body_trace"]["valid"] is True
    assert hierarchical_verified["recomputed_candidate_family"] == "transformer_hybrid"


def test_evaluate_only_requires_execute_before_any_report_write() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "standard_causal_transformer_survival.py"), "--evaluate-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--resume and --evaluate-only require --execute" in result.stderr


def test_blind_audit_inspects_the_new_inference_surface(tmp_path: Path) -> None:
    violating = tmp_path / "violating_generator.py"
    violating.write_text(
        "def visible_eval_source(row):\n"
        "    return row.get('solution_body')\n",
        encoding="utf-8",
    )
    result = audit_source(violating)
    assert result["violation_count"] == 1
    assert result["violations"][0]["field"] == "solution_body"

    clean = audit_source(ROOT / "scripts" / "standard_causal_transformer_survival.py")
    assert clean["violation_count"] == 0


def test_source_derangement_keeps_targets_and_changes_sources() -> None:
    import numpy as np

    inputs = np.array(
        [
            [1, 10, 2, 20, 21, 0],
            [1, 11, 12, 2, 20, 22],
        ],
        dtype=np.int32,
    )
    labels = np.array(
        [
            [10, 2, 20, 21, 23, 0],
            [11, 12, 2, 20, 22, 23],
        ],
        dtype=np.int32,
    )
    mask = np.array(
        [
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1],
        ],
        dtype=np.float32,
    )
    negative_inputs, negative_labels, negative_mask = deranged_source_arrays(inputs, labels, mask, seed=7)
    assert list(negative_inputs[0, :3]) == [1, 11, 12]
    assert 2 in negative_inputs[0]
    assert list(negative_labels[0][negative_mask[0] > 0]) == [21, 23]
    assert list(negative_labels[1][negative_mask[1] > 0]) == [22, 23]
    assert list(negative_mask.sum(axis=1)) == list(mask.sum(axis=1))


def test_conditioning_config_forbids_canonical_report_overwrite() -> None:
    config = json.loads((ROOT / "configs" / "standard_causal_transformer_conditioning.json").read_text())
    validate_conditioning_config(config)
    unsafe = json.loads(json.dumps(config))
    unsafe["report"] = "reports/standard_causal_transformer_survival.json"
    with pytest.raises(ValueError, match="cannot overwrite"):
        validate_conditioning_config(unsafe)


def test_conditioning_preflight_rejects_stale_stage_and_incomplete_training(tmp_path: Path) -> None:
    config = json.loads((ROOT / "configs" / "standard_causal_transformer_conditioning.json").read_text())
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    (stage_dir / "stage_arrays_v1.npz").write_bytes(b"not-loaded-by-preflight")
    (stage_dir / "stage_metadata_v1.json").write_text(json.dumps({"stage_signature": "stale"}))
    base_report = tmp_path / "base_report.json"
    base_report.write_text(
        json.dumps(
            {
                "artifacts": {
                    "config": config["base_config"],
                    "checkpoint": config["base_checkpoint"],
                    "stage_dir": str(stage_dir),
                },
                "stage": {"stage_signature": "stale"},
                "training": {"complete": False, "evaluation_only_replay": True},
            }
        )
    )
    config["stage_dir"] = str(stage_dir)
    config["stage_signature"] = "stale"
    config["base_report"] = str(base_report)
    config["report"] = str(tmp_path / "conditioning.json")
    bindings = inspect_conditioning_bindings(config)
    fault_codes = {row["code"] for row in bindings["binding_faults"]}
    blocker_codes = {row["code"] for row in bindings["training_blockers"]}
    assert "stage_logic_or_source_mismatch" in fault_codes
    assert "base_training_receipt_incomplete" in blocker_codes
    assert "base_training_receipt_is_evaluation_replay" in blocker_codes
    assert bindings["ready_for_measure"] is False
    assert bindings["ready_for_train"] is False


def test_conditioning_checkpoint_publication_is_atomic(tmp_path: Path) -> None:
    final = tmp_path / "checkpoint.npz"
    partial = tmp_path / "checkpoint.partial.npz"
    final.write_bytes(b"old-complete")
    partial.write_bytes(b"new-complete")
    assert final.read_bytes() == b"old-complete"
    publish_completed_checkpoint(partial, final)
    assert final.read_bytes() == b"new-complete"
    assert not partial.exists()


def test_conditioning_rejects_ambiguous_legacy_execute_flag() -> None:
    canonical = ROOT / "reports" / "standard_causal_transformer_survival.json"
    before = canonical.read_bytes()
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "standard_causal_transformer_conditioning.py"), "--execute"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "unrecognized arguments: --execute" in result.stderr
    assert canonical.read_bytes() == before


def test_conditioning_train_mode_fails_before_optimizer_without_complete_receipt(tmp_path: Path) -> None:
    config = json.loads((ROOT / "configs" / "standard_causal_transformer_conditioning.json").read_text())
    config["conditioned_checkpoint_dir"] = str(tmp_path / "checkpoint")
    config["report"] = str(tmp_path / "report.json")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "standard_causal_transformer_conditioning.py"),
            "--config",
            str(config_path),
            "--mode",
            "train",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["trigger_state"] == "RED"
    assert {
        "base_training_receipt_incomplete",
        "base_training_receipt_is_evaluation_replay",
    } <= {row["code"] for row in report["typed_faults"]}
    assert not (tmp_path / "checkpoint").exists()


def test_semantic_stage_source_rejects_placeholders_and_strips_metadata() -> None:
    placeholder, placeholder_audit = semantic_stage_source(
        {
            "function": "foo",
            "source_text": "Implement Python function foo.\nfoo\nvisible_intent_tags intent_list\nsignature def foo(data):",
        }
    )
    assert placeholder == ""
    assert placeholder_audit["placeholder"] is True

    semantic, semantic_audit = semantic_stage_source(
        {
            "function": "dedupe",
            "source_text": (
                "Return unique input values in their original order while preserving the first occurrence.\n"
                "dedupe\nvisible_intent_tags intent_collection\n"
                "prompt_operation_hints op_stable_dedup\nsignature def dedupe(data):"
            ),
        }
    )
    assert semantic.startswith("Return unique input values")
    assert "visible_intent_tags" not in semantic
    assert "prompt_operation_hints" not in semantic
    assert semantic.endswith("signature def solve(data):")
    assert semantic_audit == {"placeholder": False, "too_short": False, "metadata_tagged": True}


def test_finished_beam_pool_does_not_collapse_to_fanout_size() -> None:
    config = {"fanout": 4, "completion_pool_multiplier": 8}
    assert completion_pool_target(config) == 32
    rows = [
        {"tokens": [f"token_{index}"], "score": -float(index + 1)}
        for index in range(40)
    ]
    retained = prune_complete_beams(rows, limit=completion_pool_target(config), length_penalty=0.7)
    assert len(retained) == 32
    assert beam_rank_score(retained[0], 0.7) >= beam_rank_score(retained[-1], 0.7)


def test_private_callable_signature_is_invariant_to_hidden_tests_body_and_decoder_metadata() -> None:
    base = {
        "entry_point": "pair_sum",
        "solution_body": "return data + other",
        "tests": "assert pair_sum(2, 3) == 5\n",
        "decoder_contract": {"visible_arg_count_hint": 2, "return_shape": "number"},
    }
    changed_hidden = {
        **base,
        "solution_body": "return None",
        "tests": "assert pair_sum(1) == 99\n",
        "decoder_contract": {"visible_arg_count_hint": 1, "return_shape": "dict"},
    }
    expected = (
        "def pair_sum(data=None, other=None, *extra):",
        {"source": "generic_prompt_only_interface", "arity": "variable"},
    )
    assert training_callable_signature(base) == expected
    assert training_callable_signature(changed_hidden) == expected


def test_explicit_callable_signature_is_preserved_without_hidden_derivation() -> None:
    signature, receipt = training_callable_signature(
        {
            "entry_point": "pair_sum",
            "callable_signature": "def pair_sum(left, right):",
            "tests": "assert pair_sum(1) == 99\n",
            "solution_body": "return None",
        }
    )
    assert signature == "def pair_sum(left, right):"
    assert receipt == {"source": "explicit_callable_signature", "arity": 2}


def test_model_signature_canonicalizes_name_but_preserves_declared_arguments() -> None:
    config = {"tokenization": {"canonical_model_signature_name": "solve"}}
    assert canonical_model_signature("def task_123(left, right, *extra):", config) == (
        "def solve(left, right, *extra):"
    )


def test_shared_source_encoding_uses_target_vocabulary_id_space() -> None:
    config = {"tokenization": {"shared_source_target_vocabulary": True}}
    source_vocab = {"Return": 0}
    target_vocab = json.loads(
        (ROOT / "checkpoints" / "strict_generator_mlx_dense_body_open_vocab" /
         "strict_generator_mlx_strict_generator_dense_right_sized_v1_vocab.json").read_text()
    )["target_vocab"]
    ids, receipt = encode_model_source("Return data", source_vocab, target_vocab, config)
    assert ids
    assert receipt["unknown_token_count"] == 0
    assert receipt["model_stream"] == "shared_source_target"
    assert source_token_offset(config, source_vocab) == 3
    assert target_token_offset(config, source_vocab) == 3


def test_split_source_encoding_uses_disjoint_source_embedding_segment() -> None:
    config = {"tokenization": {"shared_source_target_vocabulary": False}}
    source_vocab = {"Return": 0, "data": 1}
    target_vocab = {"<bos>": 0, "return": 1, "data": 2}
    ids, receipt = encode_model_source("Return data", source_vocab, target_vocab, config)
    assert ids == [0, 1]
    assert receipt["model_stream"] == "split_source_target"
    assert source_token_offset(config, source_vocab) == 3
    assert target_token_offset(config, source_vocab) == 5
    embedded_source_ids = [source_token_offset(config, source_vocab) + value for value in ids]
    assert max(embedded_source_ids) < target_token_offset(config, source_vocab)


def test_broad_private_rows_declare_prompt_visible_signature_from_template_contract() -> None:
    by_category = {row.category: row for row in template_bank()}
    one_arg = row_from_template(
        by_category["bpg_gcd_positive"], split="eval", task_index=1, variant=1
    )
    two_arg = row_from_template(
        by_category["bpg_lcs_length"], split="eval", task_index=2, variant=2
    )
    four_arg = row_from_template(
        by_category["bpg_shortest_hops"], split="eval", task_index=3, variant=3
    )
    assert one_arg["callable_signature"].endswith("(data):")
    assert two_arg["callable_signature"].endswith("(data, other):")
    assert four_arg["callable_signature"].endswith("(data, other, *extra):")
    assert all(
        row["provenance"]["callable_signature_source"]
        == "template_declared_argument_contract"
        for row in (one_arg, two_arg, four_arg)
    )


def test_decoder_interface_hint_ignores_private_tests_and_solution() -> None:
    row = {
        "category": "",
        "prompt": "Transform the input sequence.",
        "tests": "assert solve(1, 2, 3, 4) == 10",
        "solution_body": "return data + other + extra[0] + extra[1]",
        "public_benchmark": False,
    }
    assert visible_arg_count_hint_for_task(row) is None
    row["prompt"] = "Compare two strings and return the shorter one."
    assert visible_arg_count_hint_for_task(row) == 2


def test_blind_audit_rejects_hidden_test_derived_signature_helper(tmp_path: Path) -> None:
    source = tmp_path / "leaky_signature.py"
    source.write_text(
        "def training_callable_signature(row):\n"
        "    tests = row.get('tests')\n"
        "    return 'def solve(data):' if tests else 'def solve():'\n"
    )
    audit = audit_source(source)
    assert audit["violation_count"] == 1
    assert audit["violations"][0]["kind"] == "forbidden_field_in_inference_path"
    assert audit["violations"][0]["field"] == "tests"


def test_family_disjoint_eval_freezes_normalized_callable_signature() -> None:
    config = json.loads((ROOT / "configs" / "standard_causal_transformer_survival.json").read_text())
    rows, _families = select_family_disjoint_eval(config)
    lcs = next(row for row in rows if row["concept_residual_label"] == "bpg_lcs_length")
    assert lcs["callable_signature"].endswith("(data, other):")
    assert lcs["callable_signature_receipt"] == {
        "source": "explicit_callable_signature",
        "arity": 2,
    }
    assert visible_eval_source(lcs).endswith("signature def solve(data, other):")


def test_private_prompt_variants_share_fixed_body_sampling_mass() -> None:
    examples = [
        {"source": "licensed_function", "source_text": "licensed", "body": "return data"},
        {"source": "governed_private", "source_text": "first", "body": "return data + 1"},
        {"source": "governed_private", "source_text": "second", "body": "return data + 1"},
        {"source": "governed_private", "source_text": "third", "body": "return data - 1"},
    ]
    weighted, audit = assign_body_balanced_sampling_weights(examples, private_body_weight=16.0)
    assert weighted[0]["sampling_weight"] == 1.0
    assert weighted[1]["sampling_weight"] == weighted[2]["sampling_weight"] == 8.0
    assert weighted[3]["sampling_weight"] == 16.0
    assert audit["private_sampling_mass"] == 32.0
    probabilities = normalized_sampling_probabilities(
        np.array([row["sampling_weight"] for row in weighted]), len(weighted)
    )
    assert probabilities is not None
    assert float(probabilities.sum()) == pytest.approx(1.0)


def test_coverage_first_plan_is_deterministic_and_covers_overlapping_groups() -> None:
    labels = (
        ("objective:a", "verifier:positive"),
        ("objective:a", "decision:clarify", "verifier:negative:semantic"),
        ("objective:b", "decision:abstain", "verifier:negative:decision"),
        ("objective:b", "verifier:positive"),
    )
    required = (
        "objective:a",
        "objective:b",
        "decision:clarify",
        "decision:abstain",
        "verifier:positive",
        "verifier:negative:semantic",
        "verifier:negative:decision",
    )
    first = survival.coverage_first_plan(
        labels, required, row_count=len(labels), capacity=4
    )
    second = survival.coverage_first_plan(
        labels, required, row_count=len(labels), capacity=4
    )
    assert first == second
    assert first["state"] == "PLANNED"
    observed = {
        label
        for index in first["selected_indices"]
        for label in labels[index]
    }
    assert set(required) <= observed
    assert len(first["selected_indices"]) == 3


def test_coverage_first_plan_fails_closed_when_capacity_is_too_small() -> None:
    with pytest.raises(ValueError, match="bounded run permits"):
        survival.coverage_first_plan(
            (("a",), ("b",)), ("a", "b"), row_count=2, capacity=1
        )


def test_private_sampling_mass_can_match_a_frozen_probability() -> None:
    examples = [
        {"source": "licensed_function", "source_text": "licensed-a", "body": "return data"},
        {"source": "licensed_function", "source_text": "licensed-b", "body": "return other"},
        {"source": "governed_private", "source_text": "private-a", "body": "return data + 1"},
        {"source": "governed_private", "source_text": "private-b", "body": "return data - 1"},
    ]
    weighted, audit = assign_body_balanced_sampling_weights(
        examples,
        private_body_weight=16.0,
        private_sampling_probability_target=0.25,
    )
    assert audit["configured_private_body_sampling_weight"] == 16.0
    assert audit["private_sampling_probability"] == pytest.approx(0.25)
    assert sum(row["sampling_weight"] for row in weighted[2:]) == pytest.approx(2 / 3)


def test_licensed_curriculum_reweights_without_dropping_rows_or_private_mass() -> None:
    examples = [
        {
            "source": "licensed_function",
            "source_text": "standalone",
            "body": "return data",
            "sampling_multiplier": 3.0,
        },
        {
            "source": "licensed_function",
            "source_text": "context dependent",
            "body": "return module.value",
            "sampling_multiplier": 0.5,
        },
        {
            "source": "governed_private",
            "source_text": "private-a",
            "body": "return data + 1",
        },
        {
            "source": "governed_private",
            "source_text": "private-b",
            "body": "return data - 1",
        },
    ]
    weighted, audit = assign_body_balanced_sampling_weights(
        examples,
        private_body_weight=16.0,
        private_sampling_probability_target=0.25,
    )
    assert len(weighted) == len(examples)
    assert [row["sampling_weight"] for row in weighted[:2]] == [3.0, 0.5]
    assert audit["licensed_sampling_mass"] == 3.5
    assert audit["private_sampling_probability"] == pytest.approx(0.25)


def test_standalone_sft_contract_accepts_imports_and_lexical_closures() -> None:
    source = "Compute a rounded square root.\nsignature def solve(data):"
    body = (
        "import math\n"
        "def rounded(value):\n"
        "    return round(math.sqrt(value), 3)\n"
        "return rounded(data)"
    )
    decision = standalone_sft_contract_decision(source, body)
    assert decision["accepted"] is True
    assert decision["unresolved_names"] == []
    assert decision["target_derived_source_field_count"] == 0


def test_standalone_sft_contract_rejects_hidden_module_context() -> None:
    source = "Normalize numeric input.\nsignature def solve(data):"
    decision = standalone_sft_contract_decision(source, "return np.asarray(data)")
    assert decision["accepted"] is False
    assert decision["reject_reasons"] == ["unresolved_module_context"]
    assert decision["unresolved_names"] == ["np"]
    assert decision["model_source_unchanged"] is True


def test_standalone_sft_contract_requires_declared_signature() -> None:
    decision = standalone_sft_contract_decision("Compute a sum.", "return sum(data)")
    assert decision["accepted"] is False
    assert "declared_signature_missing" in decision["reject_reasons"]


def test_teacher_sampling_target_is_exact_and_body_balanced() -> None:
    examples = [
        {"source": "licensed_function", "source_text": "a", "body": "return 1"},
        {"source": "licensed_function", "source_text": "b", "body": "return 2"},
        {"source": "governed_private", "source_text": "c", "body": "return 3"},
        {"source": "governed_openai_teacher", "source_text": "d", "body": "return 4"},
        {"source": "governed_openai_teacher", "source_text": "e", "body": "return 4"},
        {"source": "governed_openai_teacher", "source_text": "f", "body": "return 5"},
    ]
    weighted, summary = assign_body_balanced_sampling_weights(
        examples,
        private_body_weight=2.0,
        teacher_sampling_probability_target=0.2,
    )
    teacher = [row for row in weighted if row["source"] == "governed_openai_teacher"]
    duplicate_body_mass = sum(
        row["sampling_weight"] for row in teacher if row["body"] == "return 4"
    )
    unique_body_mass = sum(
        row["sampling_weight"] for row in teacher if row["body"] == "return 5"
    )
    assert duplicate_body_mass == unique_body_mass
    assert summary["teacher_sampling_probability"] == 0.2


def test_training_completion_uses_consumed_positions_not_estimated_steps() -> None:
    config = {
        "pretrain_target_token_positions": 100,
        "sft_target_token_positions": 50,
    }
    reports = [
        {"phase": "licensed_module_causal_pretraining", "target_positions_consumed": 100},
        {"phase": "prompt_signature_body_sft", "target_positions_consumed": 49},
    ]
    assert phase_target_positions(reports, "prompt_signature_body_sft") == 49
    assert training_targets_complete(reports, config) is False
    reports.append(
        {"phase": "prompt_signature_body_sft_continuation", "target_positions_consumed": 1}
    )
    assert training_targets_complete(reports, config) is True


def test_stage_signature_ignores_decode_only_knobs_but_tracks_staging_contract() -> None:
    config = json.loads((ROOT / "configs" / "standard_causal_transformer_survival.json").read_text())
    baseline = stage_signature(config)
    decode_only = json.loads(json.dumps(config))
    decode_only["evaluation"]["beam_width"] += 4
    decode_only["preference"]["optimizer_steps"] += 1
    assert stage_signature(decode_only) == baseline

    staging_change = json.loads(json.dumps(config))
    staging_change["tokenization"]["max_source_tokens"] += 1
    assert stage_signature(staging_change) != baseline

    contract_change = json.loads(json.dumps(config))
    contract_change["sft_contract_admission"] = {
        "require_self_contained_body": True,
        "private_sampling_probability_target": 0.25,
    }
    assert stage_signature(contract_change) != baseline


def test_stage_signature_binds_each_admitted_training_source(tmp_path: Path) -> None:
    config = json.loads((ROOT / "configs" / "standard_causal_transformer_survival.json").read_text())
    source = tmp_path / "private.jsonl"
    source.write_text('{"prompt":"p","solution_body":"return data"}\n')
    admission = tmp_path / "admission.json"
    admission.write_text(json.dumps({"train_admitted_sources": [{"path": str(source)}]}))
    config["sources"]["training_admission"] = str(admission)
    baseline = stage_signature(config)
    source.write_text('{"prompt":"p","solution_body":"return data + 1"}\n')
    assert stage_signature(config) != baseline


def test_stage_signature_binds_teacher_admission_inputs(tmp_path: Path) -> None:
    config = json.loads((ROOT / "configs" / "standard_causal_transformer_survival.json").read_text())
    manifest = tmp_path / "teacher_manifest.json"
    gate = tmp_path / "teacher_gate.json"
    manifest.write_text(json.dumps({"rows": []}), encoding="utf-8")
    gate.write_text(json.dumps({"trigger_state": "GREEN"}), encoding="utf-8")
    config["teacher_distillation"]["manifest"] = str(manifest)
    config["teacher_distillation"]["gate"] = str(gate)
    baseline = stage_signature(config)
    manifest.write_text(json.dumps({"rows": [{"row_id": "changed"}]}), encoding="utf-8")
    assert stage_signature(config) != baseline
    manifest.write_text(json.dumps({"rows": []}), encoding="utf-8")
    assert stage_signature(config) == baseline
    config["teacher_distillation"]["teacher_sampling_probability_target"] = 0.15
    assert stage_signature(config) != baseline


def test_stage_materialization_lock_blocks_duplicate_writer_and_recovers_stale_owner(
    tmp_path: Path,
) -> None:
    with stage_materialization_lock(tmp_path, timeout_seconds=0.1):
        with pytest.raises(TimeoutError):
            with stage_materialization_lock(tmp_path, timeout_seconds=0.05):
                pass
    lock_path = tmp_path / ".materialize.lock"
    lock_path.write_text(json.dumps({"owner_token": "dead", "pid": 99999999, "created_epoch": 0}))
    with stage_materialization_lock(tmp_path, timeout_seconds=0.1):
        assert lock_path.exists()
    assert not lock_path.exists()


def test_preference_rows_are_private_verifier_bearing_and_eval_disjoint() -> None:
    config = json.loads((ROOT / "configs" / "standard_causal_transformer_survival.json").read_text())
    eval_rows, holdout_families = select_family_disjoint_eval(config)
    rows, audit = select_preference_train_rows(
        config,
        holdout_families=holdout_families,
        eval_prompt_hashes={__import__("hashlib").sha256(row["prompt"].encode()).hexdigest() for row in eval_rows},
        eval_body_hashes={__import__("hashlib").sha256(row["solution_body"].encode()).hexdigest() for row in eval_rows},
    )
    assert len(rows) == config["preference"]["max_train_tasks"]
    assert all(row["split"] == "eval" and row["tests"] and row["callable_signature"] for row in rows)
    assert not ({row["concept_residual_label"] for row in rows} & set(holdout_families))
    assert audit["train_eval_family_overlap_count"] == 0
    assert audit["train_eval_prompt_overlap_count"] == 0
    assert audit["train_eval_body_overlap_count"] == 0


def test_preference_pairs_require_exact_pass_and_hard_negative() -> None:
    tasks = [{"task_id": "task-a", "prompt": "p", "callable_signature": "def solve(data):"}]
    candidates = [
        {"task_id": "task-a", "candidate_sha256": "pass", "code": "def solve(data):\n    return len(data)"},
        {"task_id": "task-a", "candidate_sha256": "fail", "code": "def solve(data):\n    return data"},
    ]
    verifier = {
        "verification_attempt_labels": [
            {
                "task_id": "task-a",
                "phase": "private_eval",
                "candidate_sha256": "fail",
                "passed": False,
                "verification_reward": 0.7,
                "verification_stage": "runtime_loaded",
                "rank_score": -1.0,
                "semantic_ir_state": "READY",
            },
            {
                "task_id": "task-a",
                "phase": "private_eval",
                "candidate_sha256": "pass",
                "passed": True,
                "verification_reward": 1.0,
                "verification_stage": "intended_behavior_passed",
                "rank_score": -2.0,
                "semantic_ir_state": "READY",
            },
        ]
    }
    pairs, summary = build_preference_pairs(tasks, candidates, verifier, max_pairs=4, seed=3)
    assert len(pairs) == 1
    assert pairs[0]["chosen"]["candidate_sha256"] == "pass"
    assert pairs[0]["rejected"]["candidate_sha256"] == "fail"
    assert summary["selected_pair_count"] == summary["semantic_ir_ready_pair_count"] == 1


def test_reward_removed_control_has_identically_zero_pair_signal() -> None:
    pairs = [
        {"pair_id": str(index), "chosen": f"c{index}", "rejected": f"r{index}", "chosen_reward": 1.0, "rejected_reward": 0.0, "chosen_stage": "pass", "rejected_stage": "fail"}
        for index in range(4)
    ]
    control, summary = reward_removed_pairs(pairs, seed=9)
    assert summary == {
        "pair_count": 4,
        "zero_reward_pair_count": 4,
        "all_pair_margins_identically_zero": True,
        "verifier_direction_available_to_control": False,
    }
    assert all(row["chosen"] == row["rejected"] for row in control)
    assert all(row["chosen_reward"] == row["rejected_reward"] for row in control)


def test_tiny_dpo_update_increases_verifier_preference_margin() -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    mx.random.seed(17)
    cfg = CausalTransformerConfig(vocab_size=32, d_model=16, num_layers=1, num_heads=4, num_kv_heads=2, ff_dim=32)
    reference = build_model(cfg, mx=mx, nn=nn)
    policy = build_model(cfg, mx=mx, nn=nn)
    policy.update(reference.parameters())
    arrays = PreferenceArrays(
        chosen_inputs=np.array([[1, 4, 5, 6], [1, 7, 8, 9]], dtype=np.int32),
        chosen_labels=np.array([[4, 5, 6, 10], [7, 8, 9, 11]], dtype=np.int32),
        chosen_mask=np.ones((2, 4), dtype=np.float32),
        rejected_inputs=np.array([[1, 4, 5, 12], [1, 7, 8, 13]], dtype=np.int32),
        rejected_labels=np.array([[4, 5, 12, 14], [7, 8, 13, 15]], dtype=np.int32),
        rejected_mask=np.ones((2, 4), dtype=np.float32),
    )
    report = train_dpo(
        policy,
        reference,
        arrays,
        optimizer_steps=6,
        batch_size=2,
        learning_rate=1e-3,
        beta=0.2,
        gradient_clip_norm=1.0,
        seed=5,
        mx=mx,
        nn=nn,
        optim=optim,
    )
    assert report["state"] == "TRAINED"
    assert report["preference_margin_delta"] > 0
    assert report["public_training_rows_written"] == report["external_inference_calls"] == 0


def test_all_candidate_preference_labeling_does_not_stop_after_first_pass() -> None:
    task = {
        "task_id": "private-pref",
        "entry_point": "solve",
        "category": "private",
        "concept_residual_label": "private_preference_test",
        "split": "eval",
        "tests": "assert solve([1, 2, 3]) == 3\n",
    }
    candidates = [
        {
            "task_id": "private-pref",
            "entry_point": "solve",
            "phase": "private_eval",
            "rank": 1,
            "candidate_sha256": "pass",
            "code": "def solve(data):\n    return len(data)",
        },
        {
            "task_id": "private-pref",
            "entry_point": "solve",
            "phase": "private_eval",
            "rank": 2,
            "candidate_sha256": "fail",
            "code": "def solve(data):\n    return data",
        },
    ]
    result = evaluate_all_private_candidates([task], candidates)
    labels = result["verification_attempt_labels"]
    assert len(labels) == 2
    assert {row["candidate_sha256"]: row["passed"] for row in labels} == {
        "pass": True,
        "fail": False,
    }
    assert result["uses_eval_tests_or_solutions_for_generation"] is False


def test_preference_gate_rejects_margin_only_behavior_regression_promotion() -> None:
    canary = {
        "state": "GREEN",
        "adoption_state": "NOT_ADOPTED",
        "reward_improves_behavior": False,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "preference_pair_summary": {"selected_pair_count": 1},
        "reward_present_training": {"preference_margin_delta": 1.8},
        "reward_removed_training": {"preference_margin_delta": 0.0},
        "base_heldout": {"passed_task_count": 1, "rank1_passed_task_count": 0},
        "reward_present_heldout": {"passed_task_count": 0, "rank1_passed_task_count": 0, "integrity_mismatch_count": 0},
        "reward_removed_heldout": {"passed_task_count": 1, "rank1_passed_task_count": 0, "integrity_mismatch_count": 0},
        "artifacts": {"reward_checkpoint": "checkpoints/reward.npz", "control_checkpoint": "checkpoints/control.npz"},
    }
    clean = audit_preference_canary(canary, ROOT / "checkpoints" / "canonical.npz")
    assert clean["hard_gaps"] == []
    assert clean["reward_behavior_delta"] == -1

    canary["adoption_state"] = "QUALIFIED_SHADOW"
    invalid = audit_preference_canary(canary, ROOT / "checkpoints" / "canonical.npz")
    assert "preference_adoption_state_mismatch" in {row["kind"] for row in invalid["hard_gaps"]}


def test_generation_mode_gate_recomputes_speed_and_nonregression() -> None:
    canary = {
        "state": "GREEN",
        "adoption_state": "BATCHED_DEFAULT",
        "candidate_manifest_equal": True,
        "behavior_non_regression": True,
        "integrity_non_regression": True,
        "generation_speedup": 2.5,
        "serial": {"generation_runtime_ms": 1000, "passed_task_count": 1, "rank1_passed_task_count": 0, "integrity_mismatch_count": 0},
        "batched": {"generation_runtime_ms": 400, "passed_task_count": 1, "rank1_passed_task_count": 0, "integrity_mismatch_count": 0},
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    audit = audit_generation_mode_canary(canary)
    assert audit["hard_gaps"] == []
    assert audit["speedup"] == 2.5

    canary["batched"]["passed_task_count"] = 0
    invalid = audit_generation_mode_canary(canary)
    kinds = {row["kind"] for row in invalid["hard_gaps"]}
    assert "generation_mode_behavior_decision_mismatch" in kinds
    assert "generation_mode_adoption_state_mismatch" in kinds

    canary["serial"]["passed_task_count"] = 0
    canary["batched"]["passed_task_count"] = 0
    canary["behavior_non_regression"] = True
    canary["adoption_state"] = "BATCHED_RUNTIME_ONLY"
    runtime_only = audit_generation_mode_canary(canary)
    assert runtime_only["hard_gaps"] == []
    assert runtime_only["adoption_state"] == "BATCHED_RUNTIME_ONLY"


def test_policy_gate_treats_margin_only_preference_regression_as_negative_evidence() -> None:
    payload = {
        "preference_canary": {
            "adoption_state": "NOT_ADOPTED",
            "reward_improves_behavior": False,
            "preference_pair_summary": {"selected_pair_count": 1},
            "reward_present_training": {"preference_margin_delta": 1.8},
            "reward_removed_training": {"preference_margin_delta": 0.0},
            "base_heldout": {"passed_task_count": 1},
            "reward_present_heldout": {"passed_task_count": 0, "integrity_mismatch_count": 0},
            "reward_removed_heldout": {"passed_task_count": 1},
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        }
    }
    metrics = extract_behavior_metrics(payload)
    evidence = [{"path": "synthetic.json", "present": True, "behavior_metrics": metrics}]
    summary = summarize_behavior_evidence(evidence)
    assert metrics["preference_reward_behavior_delta"] == -1
    assert summary["has_behavior_lift"] is False
    assert summary["has_loss_only_lift"] is True
    assert summary["behavior_regression_refs"] == ["synthetic.json"]


def test_generation_registry_reads_nested_same_report_modes_and_requires_parity(tmp_path: Path) -> None:
    report_path = tmp_path / "mode_report.json"
    report_path.write_text(
        json.dumps(
            {
                "trigger_state": "GREEN",
                "summary": {
                    "family_disjoint_eval_task_count": 24,
                    "public_training_rows": 0,
                    "external_inference_calls": 0,
                    "fallback_return_count": 0,
                },
                "generation_mode_canary": {
                    "candidate_manifest_equal": True,
                    "public_training_rows_written": 0,
                    "external_inference_calls": 0,
                    "serial": {
                        "generation_runtime_ms": 1000,
                        "candidate_count": 8,
                        "passed_task_count": 1,
                        "integrity_mismatch_count": 0,
                        "accepted_verified_output_per_second": 1.0,
                    },
                    "batched": {
                        "generation_runtime_ms": 400,
                        "candidate_count": 8,
                        "passed_task_count": 1,
                        "integrity_mismatch_count": 0,
                        "accepted_verified_output_per_second": 2.5,
                    },
                },
            }
        )
    )
    serial = read_report_ref(str(report_path), metric_path="generation_mode_canary.serial")
    batched = read_report_ref(str(report_path), metric_path="generation_mode_canary.batched")
    assert serial["metric_path_present"] and batched["metric_path_present"]
    assert serial["metrics"]["task_count"] == batched["metrics"]["task_count"] == 24
    assert serial["metrics"]["candidate_manifest_equal"] is True
    modes = {
        "serial": {"metrics": serial["metrics"]},
        "batched": {"metrics": batched["metrics"]},
    }
    comparison = audit_comparison(
        {"id": "same-report", "baseline_mode_id": "serial", "candidate_mode_id": "batched"},
        modes,
    )
    assert comparison["promotable"] is True

    modes["batched"]["metrics"]["candidate_manifest_equal"] = False
    assert audit_comparison(
        {"id": "no-parity", "baseline_mode_id": "serial", "candidate_mode_id": "batched"},
        modes,
    )["promotable"] is False
