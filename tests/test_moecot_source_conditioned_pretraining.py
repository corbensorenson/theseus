from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from moecot_source_conditioned_pretraining import (  # noqa: E402
    attach_grounded_context_counterfactuals,
    decode_kerc_global_target,
    encode_kerc_global_target,
    contract_sha256,
    delete_spans,
    denoising_rows,
    inspect_kernel_english,
    kernel_english_split_overlap,
    kerc_code_space,
    kerc_code_tokens,
    load_kerc_semantic_source_catalog,
    materialize_kernel_english,
    select_kernel_records,
    source_rejection,
    source_conditioning_dependencies,
    validate_config,
    validate_kernel_english_config,
    validate_kernel_record_learned_abi,
    validate_manifest,
)
import kernel_english_protocol as kernel  # noqa: E402
from kerc_residual_economics import (  # noqa: E402
    build_structural_rate_distortion_allocation,
    residual_unit_allocation_receipt,
)
import vcm_semantic_memory as memory  # noqa: E402
from neural_seed_open_vocab import reserve_byte_fallback_tokens  # noqa: E402


def config() -> dict:
    return {
        "source_conditioned_pretraining": {
            "policy": "project_theseus_moecot_source_conditioned_pretraining_v1",
            "rows_by_arm": {
                "english": 0,
                "python": 2,
                "javascript_typescript": 2,
                "html_css": 2,
                "rust": 2,
            },
            "maximum_windows_per_document": 2,
            "deletion_fraction": 0.2,
            "maximum_deletion_spans": 2,
            "minimum_target_logical_tokens": 8,
            "maximum_target_logical_tokens": 24,
            "maximum_source_encoded_tokens": 200,
            "maximum_target_encoded_tokens": 100,
            "seed": 7,
            "allowed_licenses": ["mit"],
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        }
    }


def kernel_config(tmp_path: Path) -> dict:
    return {
        "stage_dir": str(tmp_path / "stage"),
        "kernel_english_training": {
            "policy": "project_theseus_moecot_kernel_english_stage_v1",
            "required": True,
            "stage_root": str(tmp_path / "kernel-stage"),
            "report": str(tmp_path / "kernel-report.json"),
            "records_jsonl": str(tmp_path / "records.jsonl"),
            "verification_ledger_jsonl": str(tmp_path / "verification-ledger.jsonl"),
            "semantic_source_catalog_json": str(
                tmp_path / "semantic-source-catalog.json"
            ),
            "objective_order": list(kernel.TRAINING_OBJECTIVES),
            "records_by_split": {
                "private_train": 1,
                "private_dev": 1,
                "private_eval": 1,
            },
            "selection": {
                "policy": "project_theseus_kerc_constraint_aware_selection_v1",
                "ranking_seed": "fixture-selection-v1",
                "selection_visible_fields": [
                    "record_sha256",
                    "raw_source_sha256",
                    "split",
                    "semantic_supervision.claim_authority",
                    "semantic_supervision.objective_authority",
                    "interaction_annotation.kind",
                ],
                "answer_text_visible": False,
                "model_outcomes_visible": False,
                "exclude_raw_sources_present_in_multiple_splits": True,
                "preserve_grounded_question_quota": True,
                "preserve_decision_grade_objective_floors": True,
                "fill_policy": "content_hash_rank_after_constraints",
            },
            "semantic_supervision": {
                "policy": "project_theseus_kerc_semantic_supervision_program_v1",
                "tiers": {
                    tier: {
                        "claim_authority": contract["claim_authority"],
                        "maximum_optimizer_sampling_weight": float(
                            contract["maximum_optimizer_sampling_weight"]
                        ),
                        "training_only": contract["allowed_splits"]
                        == {"private_train"},
                    }
                    for tier, contract in kernel.SEMANTIC_EVIDENCE_TIERS.items()
                },
                "minimum_decision_grade_records_by_split_and_objective": {
                    split: {objective: 1 for objective in kernel.TRAINING_OBJECTIVES}
                    for split in ("private_train", "private_dev", "private_eval")
                },
                "maximum_train_record_share_by_tier": {
                    "local_parser_silver": 0.8,
                    "governed_openai_residual": 0.1,
                },
                "maximum_train_optimizer_probability_by_tier": {
                    "governed_openai_residual": 0.02,
                },
                "source_qualification": [
                    {
                        "source_id": "private-kerc-fixture-v1",
                        "intended_tier": "audited_human_gold",
                        "disposition": "eligible_after_dual_review_and_content_addressing",
                        "public_benchmark_surface": False,
                        "license_spdx": "CC0-1.0",
                        "source_url": "local://private-kerc-fixture-v1",
                    }
                ],
                "public_semantic_benchmarks_training_forbidden": True,
                "silver_can_satisfy_decision_grade_floor": False,
            },
            "semantic_corpus_materialization": {
                "policy": "project_theseus_kerc_semantic_corpus_materialization_v1",
                "output_root": str(tmp_path / "kernel-output"),
                "candidate_records_jsonl": str(tmp_path / "candidate-records.jsonl"),
                "producer_manifest_json": str(tmp_path / "producer-manifest.json"),
                "content_cache": {
                    "policy": "project_theseus_kerc_content_addressed_run_cache_v1",
                    "root": str(tmp_path / "content-cache"),
                    "enabled": True,
                    "producer_role": "fixture-producer",
                    "verifier_role": "fixture-verifier",
                    "family_identity_policy": "project_theseus_kerc_source_family_identity_v1",
                    "producer_candidate_layer": "candidate_record_v1",
                    "producer_finalization_layer": "candidate_finalization_v1",
                    "verifier_semantic_layer": "semantic_admission_v1",
                    "common_change_invalidates_all_families": True,
                    "family_local_change_invalidates_only_that_family": True,
                },
                "dolly": {
                    "path": str(tmp_path / "dolly.jsonl"),
                    "dataset_id": "fixture-dolly",
                    "dataset_revision": "fixture-v1",
                    "source_url": "https://example.test/dolly",
                    "license_evidence_url": "https://example.test/dolly-license",
                    "content_sha256": "sha256:" + "1" * 64,
                    "license_spdx": "CC0-1.0",
                    "records_by_split": {
                        "private_train": 1,
                        "private_dev": 0,
                        "private_eval": 0,
                    },
                    "grounded_question_records_by_split": {
                        "private_train": 0,
                        "private_dev": 0,
                        "private_eval": 0,
                    },
                    "grounded_question_required_forms": [
                        "what",
                        "who",
                        "where",
                        "when",
                    ],
                    "grounded_question_claim_scope": "fixture extractive support only",
                    "grounded_question_allowed_objectives": list(
                        kernel.TRAINING_OBJECTIVES
                    ),
                    "allowed_objectives": list(kernel.TRAINING_OBJECTIVES),
                },
                "masc": {
                    "archive_path": str(tmp_path / "masc.tgz"),
                    "extracted_root": str(tmp_path / "masc"),
                    "dataset_id": "fixture-masc",
                    "dataset_revision": "fixture-v1",
                    "source_url": "https://example.test/masc",
                    "license_evidence_url": "https://example.test/masc-license",
                    "content_sha256": "sha256:" + "2" * 64,
                    "license_spdx": "CC0-1.0",
                    "records_by_split": {
                        "private_train": 0,
                        "private_dev": 1,
                        "private_eval": 1,
                    },
                    "composite_semantic_records_by_split": {
                        "private_train": 0,
                        "private_dev": 0,
                        "private_eval": 0,
                    },
                    "composite_semantic_minimum_frames": 2,
                    "composite_semantic_maximum_frames": 8,
                    "composite_semantic_unique_source_credit": 0,
                    "composite_semantic_claim_scope": "fixture multi-frame annotations only",
                    "decision_semantic_records_by_split": {
                        "private_train": 0,
                        "private_dev": 0,
                        "private_eval": 0,
                    },
                    "decision_semantic_minimum_annotations": 2,
                    "decision_semantic_maximum_annotations": 24,
                    "decision_semantic_unique_source_credit": 0,
                    "decision_semantic_claim_scope": "fixture decision semantics only",
                    "event_coreference": {
                        "policy": "project_theseus_kerc_masc_manual_event_coreference_v1",
                        "alignment_contract": "complete_named_gate_group_dual_independent_token_alignment_v1",
                        "source_compaction_contract": "uniform_radius_mention_centered_source_windows_v1",
                        "original_event_root": str(tmp_path / "masc-events"),
                        "document_map": {
                            "fixture-dev.xml": "fixture/dev",
                            "fixture-eval.xml": "fixture/eval",
                        },
                        "expected_observed_group_count": 0,
                        "expected_observed_mention_count": 0,
                        "expected_admitted_group_count": 0,
                        "expected_admitted_mention_count": 0,
                        "records_by_split": {
                            "private_train": 0,
                            "private_dev": 0,
                            "private_eval": 0,
                        },
                        "mentions_by_split": {
                            "private_train": 0,
                            "private_dev": 0,
                            "private_eval": 0,
                        },
                        "expected_rejected_group_count": 0,
                        "expected_rejected_groups": [],
                        "unique_source_credit": 0,
                        "claim_scope": "fixture manual event-coreference membership only",
                    },
                    "mpqa_relations": {
                        "policy": "project_theseus_kerc_masc_manual_mpqa_relation_v1",
                        "relation_contract": "complete_manual_mpqa_expression_attitude_target_source_chain_v1",
                        "source_compaction_contract": "uniform_radius_relation_member_source_windows_v1",
                        "original_mpqa_root": str(tmp_path / "masc-mpqa"),
                        "private_dev_documents": ["fixture-dev"],
                        "private_eval_documents": ["fixture-eval"],
                        "expected_observed_linked_expression_count": 0,
                        "expected_admitted_relation_count": 0,
                        "expected_admitted_source_member_count": 1,
                        "expected_admitted_attitude_count": 1,
                        "expected_admitted_target_count": 1,
                        "records_by_split": {
                            "private_train": 0,
                            "private_dev": 0,
                            "private_eval": 0,
                        },
                        "expected_rejection_reason_counts": {
                            "fixture_rejection": 0,
                        },
                        "unique_source_credit": 0,
                        "claim_scope": "fixture complete manual MPQA links only",
                    },
                    "allowed_objectives": list(kernel.TRAINING_OBJECTIVES),
                    "document_groups": {
                        "private_dev": ["fixture/dev"],
                        "private_eval": ["fixture/eval"],
                    },
                    "contextual_frame_ambiguity": {
                        "policy": "project_theseus_kerc_masc_train_only_contextual_frame_ambiguity_v1",
                        "fit_split": "private_train",
                        "minimum_distinct_frames": 2,
                        "minimum_total_occurrences": 2,
                        "claim_scope": "fixture contextual frame selection only",
                    },
                },
                "oasst2": {
                    "dataset_id": "fixture-oasst2",
                    "dataset_revision": "fixture-v1",
                    "source_url": "https://example.test/oasst2",
                    "license_evidence_url": "https://example.test/oasst2-license",
                    "content_sha256": "sha256:" + "3" * 64,
                    "license_spdx": "CC0-1.0",
                    "files": {
                        "train": {
                            "path": str(tmp_path / "oasst2-train.parquet"),
                            "content_sha256": "sha256:" + "4" * 64,
                        },
                        "validation": {
                            "path": str(tmp_path / "oasst2-validation.parquet"),
                            "content_sha256": "sha256:" + "5" * 64,
                        },
                    },
                    "records_by_split": {
                        "private_train": 0,
                        "private_dev": 0,
                        "private_eval": 0,
                    },
                    "explicit_behavior_records_by_split": {
                        "private_train": {"CLARIFY": 0, "ABSTAIN": 0},
                        "private_dev": {"CLARIFY": 0, "ABSTAIN": 0},
                        "private_eval": {"CLARIFY": 0, "ABSTAIN": 0},
                    },
                    "explicit_behavior_claim_scope": "fixture surface behavior only",
                    "allowed_objectives": list(kernel.TRAINING_OBJECTIVES),
                    "minimum_quality": 0.5,
                    "maximum_label_values": {
                        "spam": 0.5,
                        "lang_mismatch": 0.5,
                        "pii": 0.5,
                        "not_appropriate": 0.5,
                    },
                    "maximum_current_characters": 1024,
                    "maximum_response_characters": 1024,
                    "maximum_context_characters": 2048,
                    "maximum_compiled_context_bytes": 4096,
                    "minimum_prior_turns": 2,
                    "maximum_prior_turns": 4,
                    "required_valid_realization_ranks": [0, 1],
                },
                "minimum_source_groups_by_split": {
                    "private_train": 1,
                    "private_dev": 1,
                    "private_eval": 1,
                },
                "minimum_source_sentences_by_split": {
                    "private_train": 1,
                    "private_dev": 1,
                    "private_eval": 1,
                },
                "maximum_source_characters": 2048,
                "residual_economics": {
                    "policy": "project_theseus_kerc_residual_economics_v1",
                    "allocation_lambda_grid_bits": [4096.0],
                    "maximum_dev_importance_weighted_structural_distortion": 1.0,
                    "importance_fit_split": "private_train",
                    "importance_calibration_split": "private_dev",
                    "importance_final_evaluation_split": "private_eval",
                    "utility_claim": False,
                },
                "public_benchmark_payload_count": 0,
                "external_inference_calls": 0,
                "fallback_return_count": 0,
                "template_credit": 0,
            },
            "maximum_sequence_tokens": 16384,
            "sequence_buckets": {
                "policy": "project_theseus_kerc_exact_sequence_buckets_v1",
                "routing": "encoded_length_only_without_target_semantic_metadata",
                "buckets": [
                    {
                        "bucket_id": "standard_8k",
                        "maximum_sequence_tokens": 8192,
                        "maximum_batch_size": 2,
                    },
                    {
                        "bucket_id": "exact_high_fan_in_16k",
                        "maximum_sequence_tokens": 16384,
                        "maximum_batch_size": 1,
                    },
                ],
                "truncation_allowed": False,
                "row_drop_allowed": False,
                "long_bucket_capability_credit": False,
            },
            "batch_size": 1,
            "materialization_execution": {
                "policy": "project_theseus_kerc_bounded_parallel_materialization_v1",
                "validation_workers": 1,
                "compilation_workers": 1,
                "batch_rows": 2,
                "deterministic_input_order": True,
                "raw_line_content_binding_required": True,
                "worker_failure_behavior": (
                    "reject_record_or_abort_stage_without_partial_trust"
                ),
            },
            "materialization_cache": {
                "policy": "project_theseus_kerc_content_addressed_run_cache_v1",
                "enabled": True,
                "cache_root": str(tmp_path / "kernel-stage-cache"),
                "role": "canonical_kerc_stage_materializer",
                "exact_dependency_binding_required": True,
                "output_identity_revalidation_required": True,
                "cache_miss_behavior": (
                    "recompute_complete_stage_and_publish_atomic_receipt"
                ),
            },
            "code_vocabulary": {
                "policy": "project_theseus_kerc_dual_code_vocabulary_v1",
                "fit_split": "private_train",
                "kernel_max_vocab": 512,
                "pointer_max_vocab": 512,
                "surface_vocabulary_owner": "canonical_moecot_target_vocab",
                "byte_fallback_required": True,
                "dev_eval_vocabulary_fit_forbidden": True,
            },
            "allowed_licenses": ["cc0-1.0"],
            "public_training_rows_written": 0,
            "public_benchmark_payload_count": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "template_credit": 0,
            "deterministic_renderer_credit": 0,
        },
    }


def kernel_record(split: str, index: int) -> dict:
    source = f"Approval request {index} may proceed."
    scope = {
        "user": "training-user",
        "project": "theseus",
        "conversation": f"training-{split}-{index}",
        "privacy": "private_local",
    }
    state = memory.create_hierarchical_residual_state(
        f"training-{split}-{index}", scope=scope
    )
    program = {
        "roots": ["k0"],
        "nodes": [
            {
                "node_id": "k0",
                "operator": "PROCEED",
                "modality": "POSSIBLE",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 0.8,
                "derivation": "compiler_inference",
                "source_spans": [[0, len(source)]],
                "arguments": [],
            }
        ],
    }
    packet = kernel.build_kernel_packet(
        source,
        program,
        hrl_state=state,
        provenance={"source": "private_test_fixture"},
    )
    importance = {
        "policy": "project_theseus_kerc_calibrated_importance_policy_v1",
        "policy_sha256": "sha256:" + "a" * 64,
        "source_visible_features_sha256": "sha256:" + "b" * 64,
        "scores": {
            "semantic_importance": 1.0,
            "surface_importance": 0.0,
            "identity_anchoring": 0.0,
        },
        "allocation_importance": 1.0,
        "target_fields_visible_to_policy": [],
        "fallback_return_count": 0,
    }
    importance["receipt_sha256"] = kernel.stable_hash(importance)
    residual = packet["residual"]
    allocation = build_structural_rate_distortion_allocation(
        kernel_program=packet["program"],
        global_state=state["global"],
        segment_residual=residual["segment_frame"],
        token_residuals=residual["token_tags"],
        exact_objects=packet["protected_objects"],
        importance=importance["allocation_importance"],
        lambda_value=4096.0,
        exact_codec=residual["codec"],
    )
    packet = kernel.revise_kernel_packet_fidelity(
        packet,
        allocation["selected_fidelity"],
        local_hrl_state=state,
    )
    answer = {
        "claims": [
            {
                "claim_id": "claim-1",
                "predicate": "PROCEED",
                "modality": "POSSIBLE",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 0.8,
                "arguments": [],
            }
        ],
        "required_terms": [],
        "required_caveats": ["The outcome remains uncertain."],
        "style": {"register": "plain"},
    }
    record = {
        "policy": kernel.TRAINING_RECORD_POLICY,
        "split": split,
        "language": "en",
        "source_text": source,
        "kernel_packet": packet,
        "hrl_state": state,
        "hrl_deltas": [],
        "answer_packet": answer,
        "surface_target": f"Request {index} may proceed, but the outcome remains uncertain.",
        "provenance": {
            "source_id": f"source-{split}-{index}",
            "source_group": f"group-{split}-{index}",
            "license_spdx": "CC0-1.0",
            "permitted_use": "model_training",
            "dataset_id": "private-kerc-fixture-v1",
            "dataset_revision": "fixture-revision-v1",
        },
        "semantic_supervision": {
            "policy": kernel.SEMANTIC_SUPERVISION_POLICY,
            "evidence_tier": "audited_human_gold",
            "producer_kind": "human_annotation",
            "producer_id": "fixture-annotator-v1",
            "producer_artifact_sha256": "sha256:" + "e" * 64,
            "annotation_source_id": "private-kerc-fixture-v1",
            "annotation_source_sha256": "sha256:" + "d" * 64,
            "claim_authority": "decision_grade_reference",
            "model_derived": False,
            "public_calibration_surface": False,
            "benchmark_payload_used": False,
            "objective_authority": {
                objective: True for objective in kernel.TRAINING_OBJECTIVES
            },
            "optimizer_sampling_weight": 1.0,
        },
        "residual_supervision": {
            "policy": "project_theseus_kerc_residual_supervision_v1",
            "labels_by_channel": {
                "interaction": 0,
                "segment": 0,
                "token": 0,
                "exact": 3,
            },
            "record_fidelity_label": kernel.KERC_FIDELITY_LABELS[
                allocation["selected_fidelity"]
            ],
            "record_fidelity_label_training_authority": False,
            "packet_wide_fidelity_drives_training": False,
            "residual_unit_allocation": residual_unit_allocation_receipt(
                packet["residual"]["unit_packet"]
            ),
            "annotator_independent_of_model": True,
            "evidence_sha256": "sha256:" + "c" * 64,
            "allocation_target_authority": "measured_structural_rate_distortion_with_calibrated_source_visible_importance",
            "rate_distortion_optimality_claimed": False,
            "importance": importance,
            "rate_distortion_allocation": allocation,
        },
        "verification_receipt": {
            "policy": kernel.TRAINING_VERIFICATION_POLICY,
            "receipt_id": f"receipt-{split}-{index}",
            "accepted": True,
            "verifier_id": "fixture-verifier-v1",
            "reviewer_independent_of_record_producer": True,
            "method": "human_dual_review",
            "evidence_sha256": "sha256:" + f"{index:064x}"[-64:],
        },
        "public_benchmark": False,
        "public_tests_included": False,
        "public_benchmark_solutions_included": False,
        "external_inference": False,
        "fallback_return_count": 0,
        "template_credit": 0,
        "deterministic_renderer_credit": 0,
        "candidate_generation_credit": 0,
    }
    record["verification_receipt"]["semantic_payload_sha256"] = (
        kernel.training_semantic_payload_sha256(record)
    )
    return record


def test_compact_allocator_authority_requires_an_enabled_unit() -> None:
    assert kernel.compact_allocator_targets_have_authority(
        {"targets": [{"allocator_loss_enabled": False}]}
    ) is False
    assert kernel.compact_allocator_targets_have_authority(
        {
            "targets": [
                {"allocator_loss_enabled": False},
                {"allocator_loss_enabled": True},
            ]
        }
    ) is True


def test_config_rejects_cross_arm_data_and_nonzero_boundaries() -> None:
    cfg = config()
    validate_config(cfg)
    cfg["source_conditioned_pretraining"]["rows_by_arm"]["english"] = 1
    with pytest.raises(ValueError, match="English arm"):
        validate_config(cfg)
    cfg = config()
    cfg["source_conditioned_pretraining"]["public_training_rows_written"] = 1
    with pytest.raises(ValueError, match="no-cheat"):
        validate_config(cfg)


def test_grounded_context_counterfactuals_remove_stale_hash_shortcuts() -> None:
    first_hash = "sha256:" + "1" * 64
    donor_hash = "sha256:" + "2" * 64
    first = {
        "record_sha256": "sha256:" + "a" * 64,
        "surface_target": "Alpha",
        "provenance": {"source_group": "first"},
        "interaction_annotation": {
            "kind": "licensed_grounded_question_context",
            "context_sha256": first_hash,
        },
        "source_annotation": {
            "context": "Alpha is supported by this source.",
            "response": "Alpha",
        },
    }
    donor = {
        "record_sha256": "sha256:" + "b" * 64,
        "surface_target": "Beta",
        "provenance": {"source_group": "donor"},
        "interaction_annotation": {
            "kind": "licensed_grounded_question_context",
            "context_sha256": donor_hash,
        },
        "source_annotation": {
            "context": "Beta is supported by a separate source.",
            "response": "Beta",
        },
    }
    encoded_first_hash = __import__("base64").b64encode(first_hash.encode()).decode()
    program = {
        "record_type": "kernel_program",
        "kernel_version": "KE-1.0",
        "roots": ["k0"],
        "nodes": [
            {
                "node_id": "k0",
                "operator": "ANSWER_FROM_CONTEXT",
                "arguments": [
                    {
                        "role": "CONTEXT_SHA256",
                        "value": {
                            "type": "byte_literal",
                            "value": encoded_first_hash,
                        },
                    }
                ],
            }
        ],
    }
    program["program_sha256"] = kernel.stable_hash(program)
    packet = {
        "claims": [
            {
                "claim_id": "claim-1",
                "arguments": [
                    {
                        "role": "ANSWER_SPAN",
                        "value": {
                            "type": "byte_literal",
                            "value": "QWxwaGE=",
                        },
                    },
                    {
                        "role": "CONTEXT_SHA256",
                        "value": {
                            "type": "byte_literal",
                            "value": encoded_first_hash,
                        },
                    },
                ],
            }
        ]
    }
    packet["answer_packet_sha256"] = kernel.stable_hash(packet)
    core_prompt = kernel.canonical_json(
        {
            "program": program,
            "residual": {
                "interaction": [
                    [
                        "document_context",
                        "content",
                        first["source_annotation"]["context"],
                    ],
                    ["question_contract", "context_sha256", first_hash],
                ]
            },
        }
    )
    direct_prompt = kernel.canonical_json(
        {
            "current_surface": "Which value is supported?",
            "interaction": [
                ["document_context", "content", first["source_annotation"]["context"]],
                ["question_contract", "context_sha256", first_hash],
            ],
        }
    )
    views = {
        "private_train": [
            {
                "objective": "surface_direct_control_v1",
                "source_record_sha256": first["record_sha256"],
                "prompt": direct_prompt,
                "target": "Alpha",
                "target_sha256": kernel.stable_hash(b"Alpha"),
                "evaluator_only_fields": [],
            },
            {
                "objective": "kernel_program_to_answer_packet_v1",
                "source_record_sha256": first["record_sha256"],
                "prompt": core_prompt,
                "target": kernel.canonical_json(packet),
                "target_sha256": kernel.stable_hash(
                    kernel.canonical_json(packet).encode()
                ),
                "evaluator_only_fields": [],
            },
        ]
    }

    receipt = attach_grounded_context_counterfactuals(
        views, {"private_train": [first, donor]}
    )

    assert receipt["total_count"] == 4
    assert receipt["missing_donor_record_sha256"] == []
    for view in views["private_train"]:
        by_strategy = {
            item["strategy"]: item for item in view["kerc_context_counterfactuals"]
        }
        withheld = json.loads(by_strategy["context_withheld"]["prompt"])
        interaction = withheld.get("interaction") or withheld["residual"]["interaction"]
        assert [
            "document_context",
            "content",
            first["source_annotation"]["context"],
        ] not in interaction
        shuffled = by_strategy["context_shuffled"]
        assert first_hash not in shuffled["prompt"]
        assert encoded_first_hash not in shuffled["prompt"]
        assert (
            first["surface_target"].casefold()
            not in donor["source_annotation"]["context"].casefold()
        )
        if view["objective"] == "kernel_program_to_answer_packet_v1":
            shuffled_prompt = json.loads(shuffled["prompt"])
            unsigned_program = {
                key: value
                for key, value in shuffled_prompt["program"].items()
                if key != "program_sha256"
            }
            assert shuffled_prompt["program"]["program_sha256"] == kernel.stable_hash(
                unsigned_program
            )
            shuffled_packet = json.loads(shuffled["target"])
            unsigned_packet = {
                key: value
                for key, value in shuffled_packet.items()
                if key != "answer_packet_sha256"
            }
            assert shuffled_packet["answer_packet_sha256"] == kernel.stable_hash(
                unsigned_packet
            )
        assert shuffled["labels"] == [0, 1, 1, 1, 0]
        assert shuffled["generator_loss_enabled"] is False
        assert shuffled["unique_source_credit"] == 0


def test_corruption_is_deterministic_bounded_and_not_identity() -> None:
    tokens = list("abcdefghijklmnopqrstuvwxyz")
    cfg = config()["source_conditioned_pretraining"]
    first = delete_spans(tokens, cfg, "1" * 64)
    second = delete_spans(tokens, cfg, "1" * 64)
    assert first == second
    assert first != tokens
    assert len(first) == len(tokens) - round(len(tokens) * cfg["deletion_fraction"])


def test_denoising_rows_keep_target_out_of_prompt_and_preserve_provenance() -> None:
    cfg = config()["source_conditioned_pretraining"]
    cfg["maximum_source_encoded_tokens"] = 1000
    cfg["maximum_target_encoded_tokens"] = 500
    source_vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    target_vocab = dict(source_vocab)
    reserve_byte_fallback_tokens(source_vocab, max_vocab=270, stream="source")
    reserve_byte_fallback_tokens(target_vocab, max_vocab=270, stream="target")
    source = {
        "text": 'fn main() { let value = 42; println!("{}", value); }\n' * 3,
        "text_sha256": "a" * 64,
        "language": "rust",
        "repo": "example/repo",
        "path": "src/main.rs",
        "license_spdx": "MIT",
    }
    rows = denoising_rows(source, "rust", cfg, source_vocab, target_vocab)
    assert rows
    assert all(row["target"] not in row["prompt"] for row in rows)
    assert all(row["public_benchmark"] is False for row in rows)
    assert all(row["source_identity"]["text_sha256"] == "a" * 64 for row in rows)
    assert all(0.0 < row["target_token_copy_fraction"] < 1.0 for row in rows)


def test_source_rejection_fails_closed_on_license_and_public_payloads() -> None:
    cfg = config()["source_conditioned_pretraining"]
    clean = {
        "text": "fn main() {}",
        "license_spdx": "MIT",
        "public_benchmark": False,
        "public_tests_included": False,
        "public_benchmark_solutions_included": False,
    }
    assert source_rejection(clean, cfg) == ""
    assert (
        source_rejection({**clean, "license_spdx": "unknown"}, cfg)
        == "license_not_allowed"
    )
    assert source_rejection({**clean, "public_benchmark": True}, cfg).startswith(
        "public_"
    )


def test_source_conditioned_manifest_binds_all_mutable_dependencies(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text('{"text":"alpha"}\n', encoding="utf-8")
    stage = tmp_path / "stage"
    stage.mkdir()
    metadata = stage / "stage_metadata_v1.json"
    metadata.write_text(
        '{"source_vocab":{"a":1},"target_vocab":{"a":1}}\n', encoding="utf-8"
    )
    supervision = tmp_path / "supervision"
    (supervision / "private_train").mkdir(parents=True)
    supervision_manifest = supervision / "manifest.json"
    supervision_manifest.write_text('{"policy":"fixture"}\n', encoding="utf-8")
    supervision_rows = supervision / "private_train" / "python.jsonl"
    supervision_rows.write_text('{"target_sha256":"abc"}\n', encoding="utf-8")
    artifact = tmp_path / "python.jsonl"
    artifact.write_text('{"row_id":"one"}\n{"row_id":"two"}\n', encoding="utf-8")

    full_config = config()
    cfg = full_config["source_conditioned_pretraining"]
    cfg["source_jsonl"] = str(source)
    cfg["stage_root"] = str(tmp_path / "source-conditioned")
    cfg["rows_by_arm"] = {
        "english": 0,
        "python": 2,
        "javascript_typescript": 0,
        "html_css": 0,
        "rust": 0,
    }
    full_config["stage_dir"] = str(stage)
    full_config["supervision"] = {"stage_root": str(supervision)}
    payload = {
        "policy": cfg["policy"],
        "contract_sha256": contract_sha256(cfg),
        "dependencies": source_conditioning_dependencies(full_config),
        "artifacts": {
            "python": {
                "path": str(artifact),
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "row_count": 2,
            }
        },
        "public_training_rows_written": 0,
        "public_benchmark_payload_count": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    assert validate_manifest(payload, cfg, full_config) == []

    source.write_text('{"text":"changed"}\n', encoding="utf-8")
    gaps = validate_manifest(payload, cfg, full_config)
    assert "dependency_identity_mismatch:source_jsonl" in gaps
    source.write_text('{"text":"alpha"}\n', encoding="utf-8")

    metadata.write_text(
        '{"source_vocab":{"b":2},"target_vocab":{"a":1}}\n', encoding="utf-8"
    )
    gaps = validate_manifest(payload, cfg, full_config)
    assert "dependency_identity_mismatch:stage_metadata" in gaps
    metadata.write_text(
        '{"source_vocab":{"a":1},"target_vocab":{"a":1}}\n', encoding="utf-8"
    )

    supervision_rows.write_text('{"target_sha256":"changed"}\n', encoding="utf-8")
    gaps = validate_manifest(payload, cfg, full_config)
    assert "dependency_identity_mismatch:supervision_stage" in gaps


def test_kerc_cache_requires_selective_family_invalidation_contract(
    tmp_path: Path,
) -> None:
    cfg = kernel_config(tmp_path)
    validate_kernel_english_config(cfg)
    cache = cfg["kernel_english_training"]["semantic_corpus_materialization"][
        "content_cache"
    ]
    for key, invalid in (
        ("family_identity_policy", "wrong-policy"),
        ("producer_candidate_layer", "whole-file-cache"),
        ("producer_finalization_layer", "unchecked-finalization"),
        ("verifier_semantic_layer", "shared-authority"),
        ("common_change_invalidates_all_families", False),
        ("family_local_change_invalidates_only_that_family", False),
    ):
        mutated = json.loads(json.dumps(cfg))
        mutated["kernel_english_training"]["semantic_corpus_materialization"][
            "content_cache"
        ][key] = invalid
        with pytest.raises(ValueError, match="content-addressed cache contract"):
            validate_kernel_english_config(mutated)
    assert cache["common_change_invalidates_all_families"] is True


def test_kerc_gum_entity_coreference_contract_fails_closed(tmp_path: Path) -> None:
    del tmp_path
    cfg = json.loads(
        (ROOT / "configs" / "moecot_language_arm_training.json").read_text()
    )
    kernel_cfg = cfg["kernel_english_training"]
    deferred = kernel_cfg["disposition"]
    kernel_cfg["required"] = True
    kernel_cfg["records_by_split"] = kernel_cfg[
        "deferred_candidate_records_by_split"
    ]
    kernel_cfg["disposition"] = {
        "policy": deferred["policy"],
        "state": "CANDIDATE_REQUIRED",
        "qualification_scope": "faithful_full_compiler_core_renderer_candidate",
        "basis": "adequacy_audit_reopened_after_toy_proxy",
        "full_kerc_training_enabled": True,
        "general_kerc_falsification_claimed": False,
        "learned_capability_claimed": False,
        "retained_mechanisms": [],
        "superseded_proxy_evidence": deferred["superseded_proxy_evidence"],
        "non_claims": deferred["non_claims"],
    }
    validate_kernel_english_config(cfg)
    for path, invalid in (
        (("relation_contract",), "pairwise_proxy"),
        (("source_compaction_contract",), "truncate_mentions"),
        (("records_by_split", "private_eval"), 675),
        (("bridge_records_by_split", "private_train"), 1312),
        (("content_sha256",), "sha256:unbound"),
    ):
        mutated = json.loads(json.dumps(cfg))
        contract = mutated["kernel_english_training"][
            "semantic_corpus_materialization"
        ]["gum"]["entity_coreference"]
        if len(path) == 1:
            contract[path[0]] = invalid
        else:
            contract[path[0]][path[1]] = invalid
        with pytest.raises(
            ValueError, match="GUM discourse/coreference contract invalid"
        ):
            validate_kernel_english_config(mutated)


def test_kernel_stage_materializes_replays_and_cleans_atomic_files(
    tmp_path: Path,
) -> None:
    cfg = kernel_config(tmp_path)
    cfg["kernel_english_training"]["materialization_execution"].update(
        {"validation_workers": 2, "compilation_workers": 2}
    )
    validate_kernel_english_config(cfg)
    source_vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    target_vocab = dict(source_vocab)
    for token in kernel.TRAINING_TASK_TAGS.values():
        source_vocab[token] = len(source_vocab)
    reserve_byte_fallback_tokens(source_vocab, max_vocab=274, stream="source")
    reserve_byte_fallback_tokens(target_vocab, max_vocab=270, stream="target")
    stage = Path(cfg["stage_dir"])
    stage.mkdir(parents=True)
    (stage / "stage_metadata_v1.json").write_text(
        json.dumps({"source_vocab": source_vocab, "target_vocab": target_vocab}),
        encoding="utf-8",
    )
    records = [
        kernel_record("private_train", 1),
        kernel_record("private_dev", 2),
        kernel_record("private_eval", 3),
    ]
    records_path = Path(cfg["kernel_english_training"]["records_jsonl"])
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    ledger_path = Path(cfg["kernel_english_training"]["verification_ledger_jsonl"])
    ledger_path.write_text(
        "".join(
            json.dumps(row["verification_receipt"], sort_keys=True) + "\n"
            for row in records
        ),
        encoding="utf-8",
    )
    catalog_path = Path(cfg["kernel_english_training"]["semantic_source_catalog_json"])
    catalog_path.write_text(
        json.dumps(
            {
                "policy": "project_theseus_kerc_semantic_source_catalog_v1",
                "sources": [
                    {
                        "dataset_id": "private-kerc-fixture-v1",
                        "dataset_revision": "fixture-revision-v1",
                        "content_sha256": "sha256:" + "d" * 64,
                        "license_spdx": "CC0-1.0",
                        "permitted_use": "model_training",
                        "training_allowed": True,
                        "public_benchmark_surface": False,
                        "public_benchmark_payload": False,
                        "allowed_evidence_tiers": ["audited_human_gold"],
                        "allowed_objectives": list(kernel.TRAINING_OBJECTIVES),
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = materialize_kernel_english(cfg, tmp_path / "config.json")
    assert report["trigger_state"] == "GREEN"
    assert report["verifier_corruption_count"] == 12
    assert report["verifier_corruptions_receive_generator_loss"] is False
    assert report["unique_raw_source_count"] == 3
    assert report["derived_view_unique_data_credit"] == 0
    assert report["derived_view_optimizer_exposure_count"] == 12
    assert report["materialization_execution"]["validation_workers"] == 2
    assert report["materialization_execution"]["compilation_workers"] == 2
    assert report["materialization_execution"]["compiled_record_count"] == 3
    assert report["materialization_execution"]["compiled_view_count"] == 12
    intervention = report["materialization_execution"]["unit_intervention_targets"]
    assert intervention["record_count"] == 3
    assert intervention["unit_count"] == 6
    assert intervention["candidate_intervention_count"] == 24
    assert intervention["allocator_authority_unit_count"] == 3
    assert intervention["withheld_uncertain_unit_count"] == 3
    assert intervention["target_producer_is_final_evaluator"] is False
    assert intervention["public_or_hidden_target_used"] is False
    assert report["semantic_supervision"][
        "decision_grade_record_counts_by_split_and_objective"
    ] == {
        split: {objective: 1 for objective in kernel.TRAINING_OBJECTIVES}
        for split in ("private_train", "private_dev", "private_eval")
    }
    assert all(
        value == 3 for value in report["compiled_view_count_by_objective"].values()
    )
    codebook_path = Path(report["code_vocabulary"]["path"])
    if not codebook_path.is_absolute():
        codebook_path = ROOT / codebook_path
    codebook = json.loads(codebook_path.read_text())
    assert codebook["fit_split"] == "private_train"
    assert codebook["dev_eval_vocabulary_fit_count"] == 0
    assert codebook["verifier_corruption_vocabulary_fit_count"] == 0
    assert codebook["kernel_vocab"] != codebook["pointer_vocab"]
    assert codebook["source_view_count"] == 2
    train_artifact = Path(report["artifacts"]["english:private_train"]["path"])
    if not train_artifact.is_absolute():
        train_artifact = ROOT / train_artifact
    train_views = [json.loads(line) for line in train_artifact.read_text().splitlines()]
    compiler_view = next(
        row for row in train_views if row["objective"] == "surface_to_kernel_program_v1"
    )
    assert compiler_view["kerc_residual_unit_allocator_loss_enabled"] is False
    assert all(
        bool(row["kerc_residual_unit_targets"])
        == row["kerc_residual_unit_allocator_loss_enabled"]
        == (row["objective"] == "kernel_program_to_answer_packet_v1")
        for row in train_views
    )
    assert compiler_view["semantic_evidence_tier"] == "audited_human_gold"
    assert compiler_view["decision_grade_reference"] is True
    assert compiler_view["optimizer_sampling_weight"] == 1.0
    encoded, encoded_receipt = encode_kerc_global_target(
        compiler_view["target"],
        code_vocabulary=codebook,
        kernel_offset=1000,
        pointer_offset=2000,
    )
    decoded, decoded_receipt = decode_kerc_global_target(
        encoded,
        code_vocabulary=codebook,
        kernel_offset=1000,
        pointer_offset=2000,
    )
    assert encoded_receipt["unknown_token_count"] == 0
    assert decoded_receipt["state"] == "READY"
    assert decoded == compiler_view["target"]
    assert (
        inspect_kernel_english(cfg, tmp_path / "config.json")["trigger_state"]
        == "GREEN"
    )
    assert not list(
        Path(cfg["kernel_english_training"]["stage_root"]).glob("*.partial")
    )

    manifest_path = Path(cfg["kernel_english_training"]["stage_root"]) / "manifest.json"
    manifest_mtime_ns = manifest_path.stat().st_mtime_ns
    replayed_report = materialize_kernel_english(cfg, tmp_path / "config.json")
    assert replayed_report == report
    assert manifest_path.stat().st_mtime_ns == manifest_mtime_ns
    cache_root = Path(
        cfg["kernel_english_training"]["materialization_cache"]["cache_root"]
    )
    assert len(list(cache_root.rglob("*.json"))) == 1

    train_path = Path(report["artifacts"]["english:private_train"]["path"])
    if not train_path.is_absolute():
        train_path = ROOT / train_path
    train_path.write_text(
        train_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8"
    )
    replay = inspect_kernel_english(cfg, tmp_path / "config.json")
    assert replay["trigger_state"] == "RED"
    assert any("artifact_identity" in gap for gap in replay["hard_gaps"])


def test_kernel_stage_preflights_learned_abi_before_bounded_selection() -> None:
    record = kernel_record("private_train", 99)
    record["kernel_packet"]["residual"]["segment_frame"] = {
        "frame_name": "Progress",
        "frame_roles": ["ENTITY"],
    }
    record["kernel_packet"]["residual"]["token_tags"] = [
        {
            "tag": "FRAME_ROLE:UNDECLARED_ROLE",
            "source_span": [0, 8],
            "authority": "licensed_manual_annotation",
        }
    ]

    with pytest.raises(
        ValueError,
        match=r"learned ABI rejected governed record at line 41.*UNDECLARED_ROLE",
    ):
        validate_kernel_record_learned_abi(record, line_number=41)


def test_kernel_code_tokenizer_preserves_compact_transport_as_exact_atoms() -> None:
    payload = kernel.canonical_json(
        {
            "policy": kernel.LEARNED_PROGRAM_TRANSPORT_POLICY,
            "tokens": [
                "PNODE_BEGIN",
                "KOP:ENTITY_RELATION_COREF",
                "PBYTE:YWJjZA==",
                "PPROGRAM_END",
            ],
        }
    )
    tokens = kerc_code_tokens(payload)

    assert "".join(tokens) == payload
    assert '"KOP:ENTITY_RELATION_COREF"' in tokens
    assert kerc_code_space('"KOP:ENTITY_RELATION_COREF"') == "V_K"
    assert kerc_code_space('"PNODE_BEGIN"') == "V_P"
    assert len(tokens) < 24


def selection_record(
    split: str,
    identity: str,
    *,
    objectives: tuple[str, ...],
    raw_source: str | None = None,
    grounded: bool = False,
) -> dict:
    return {
        "record_sha256": "sha256:" + hashlib.sha256(identity.encode()).hexdigest(),
        "raw_source_sha256": "sha256:"
        + hashlib.sha256((raw_source or identity).encode()).hexdigest(),
        "split": split,
        "semantic_supervision": {
            "claim_authority": "decision_grade_reference",
            "objective_authority": {
                objective: objective in objectives
                for objective in kernel.TRAINING_OBJECTIVES
            },
        },
        "interaction_annotation": (
            {"kind": "licensed_grounded_question_context"} if grounded else {}
        ),
    }


def selection_config(tmp_path: Path) -> dict:
    base = kernel_config(tmp_path)["kernel_english_training"]
    base["records_by_split"] = {
        "private_train": 3,
        "private_dev": 1,
        "private_eval": 1,
    }
    base["semantic_supervision"][
        "minimum_decision_grade_records_by_split_and_objective"
    ] = {
        "private_train": {objective: 1 for objective in kernel.TRAINING_OBJECTIVES},
        "private_dev": {objective: 1 for objective in kernel.TRAINING_OBJECTIVES},
        "private_eval": {objective: 1 for objective in kernel.TRAINING_OBJECTIVES},
    }
    base["semantic_corpus_materialization"]["dolly"][
        "grounded_question_records_by_split"
    ] = {"private_train": 1, "private_dev": 0, "private_eval": 0}
    return base


def test_kernel_selection_preserves_constraints_and_is_order_invariant(
    tmp_path: Path,
) -> None:
    compiler_pair = (
        "surface_to_kernel_program_v1",
        "kernel_program_to_answer_packet_v1",
    )
    all_objectives = tuple(kernel.TRAINING_OBJECTIVES)
    candidates = {
        "private_train": [
            selection_record(
                "private_train",
                "grounded-direct",
                objectives=("surface_direct_control_v1",),
                grounded=True,
            ),
            selection_record(
                "private_train",
                "answer",
                objectives=("answer_packet_to_surface_v1",),
            ),
            selection_record(
                "private_train", "compiler-core", objectives=compiler_pair
            ),
            selection_record(
                "private_train",
                "cross-split-train",
                raw_source="duplicate-source",
                objectives=all_objectives,
            ),
        ],
        "private_dev": [
            selection_record("private_dev", "dev", objectives=all_objectives),
            selection_record(
                "private_dev",
                "cross-split-dev",
                raw_source="duplicate-source",
                objectives=all_objectives,
            ),
        ],
        "private_eval": [
            selection_record("private_eval", "eval", objectives=all_objectives)
        ],
    }
    cfg = selection_config(tmp_path)

    selected, receipt = select_kernel_records(candidates, cfg)
    reversed_selected, reversed_receipt = select_kernel_records(
        {split: list(reversed(rows)) for split, rows in candidates.items()}, cfg
    )

    assert receipt == reversed_receipt
    assert {
        split: [row["record_sha256"] for row in rows]
        for split, rows in selected.items()
    } == {
        split: [row["record_sha256"] for row in rows]
        for split, rows in reversed_selected.items()
    }
    assert receipt["cross_split_raw_source_exclusion_count"] == 1
    assert receipt["cross_split_raw_source_excluded_record_count"] == 2
    assert receipt["grounded_question_count_by_split"]["private_train"] == 1
    assert all(
        count >= 1
        for counts in receipt["decision_grade_objective_count_by_split"].values()
        for count in counts.values()
    )
    assert not {
        row["raw_source_sha256"] for rows in selected.values() for row in rows
    } & {"sha256:" + hashlib.sha256(b"duplicate-source").hexdigest()}


def test_kernel_selection_fails_closed_when_declared_floor_is_infeasible(
    tmp_path: Path,
) -> None:
    cfg = selection_config(tmp_path)
    candidates = {
        "private_train": [
            selection_record(
                "private_train",
                f"train-{index}",
                objectives=("surface_to_kernel_program_v1",),
                grounded=index == 0,
            )
            for index in range(3)
        ],
        "private_dev": [
            selection_record(
                "private_dev", "dev", objectives=tuple(kernel.TRAINING_OBJECTIVES)
            )
        ],
        "private_eval": [
            selection_record(
                "private_eval", "eval", objectives=tuple(kernel.TRAINING_OBJECTIVES)
            )
        ],
    }

    with pytest.raises(ValueError, match="objective selection floor infeasible"):
        select_kernel_records(candidates, cfg)


def test_semantic_source_catalog_rejects_public_calibration_sources(
    tmp_path: Path,
) -> None:
    cfg = kernel_config(tmp_path)["kernel_english_training"]
    catalog = tmp_path / "semantic-source-catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "policy": "project_theseus_kerc_semantic_source_catalog_v1",
                "sources": [
                    {
                        "dataset_id": "public-semantic-benchmark",
                        "dataset_revision": "frozen-v1",
                        "content_sha256": "sha256:" + "f" * 64,
                        "license_spdx": "CC0-1.0",
                        "permitted_use": "model_training",
                        "training_allowed": True,
                        "public_benchmark_surface": True,
                        "public_benchmark_payload": True,
                        "allowed_evidence_tiers": ["licensed_human_task_gold"],
                        "allowed_objectives": list(kernel.TRAINING_OBJECTIVES),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    sources, gaps = load_kerc_semantic_source_catalog(catalog, cfg)
    assert sources == {}
    assert gaps == [
        "kernel_semantic_source_catalog_contract_invalid:public-semantic-benchmark"
    ]


def test_deferred_kernel_stage_needs_no_records_and_writes_zero_exposure_receipt(
    tmp_path: Path,
) -> None:
    canonical = json.loads(
        (ROOT / "configs" / "moecot_language_arm_training.json").read_text()
    )["kernel_english_training"]
    canonical["required"] = False
    canonical["records_by_split"] = {
        "private_train": 0,
        "private_dev": 0,
        "private_eval": 0,
    }
    canonical["stage_root"] = str(tmp_path / "deferred-kernel-stage")
    canonical["report"] = str(tmp_path / "deferred-kernel-report.json")
    canonical["records_jsonl"] = str(tmp_path / "must-not-exist-records.jsonl")
    canonical["verification_ledger_jsonl"] = str(
        tmp_path / "must-not-exist-ledger.jsonl"
    )
    cfg = {"stage_dir": str(tmp_path / "stage"), "kernel_english_training": canonical}

    validate_kernel_english_config(cfg)
    inspected = inspect_kernel_english(cfg, tmp_path / "config.json")
    assert inspected["trigger_state"] == "GREEN"
    assert inspected["full_kerc_training_enabled"] is False
    assert inspected["artifacts"] == {}
    assert inspected["verification_ledger_required"] is False

    materialized = materialize_kernel_english(cfg, tmp_path / "config.json")
    assert materialized["mode"] == "deferred_from_first_long_run"
    assert materialized["selected_record_count_by_split"] == {
        "private_train": 0,
        "private_dev": 0,
        "private_eval": 0,
    }
    assert not Path(canonical["records_jsonl"]).exists()
    assert not Path(canonical["verification_ledger_jsonl"]).exists()
    assert (Path(canonical["stage_root"]) / "manifest.json").is_file()


def test_kernel_split_overlap_rejects_group_and_content_reuse() -> None:
    train = kernel.validate_training_record(kernel_record("private_train", 10))
    dev = kernel.validate_training_record(kernel_record("private_dev", 11))
    dev["provenance"]["source_group"] = train["provenance"]["source_group"]
    dev["raw_source_sha256"] = train["raw_source_sha256"]
    audit = kernel_english_split_overlap(
        {"private_train": [train], "private_dev": [dev], "private_eval": []}
    )
    assert audit["content_bound_disjoint"] is False
    assert audit["source_group_overlap_count"] == 1
    assert audit["raw_source_overlap_count"] == 1
