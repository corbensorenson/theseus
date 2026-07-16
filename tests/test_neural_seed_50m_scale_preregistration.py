from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import neural_seed_50m_scale_preregistration as scale  # noqa: E402
from standard_causal_transformer_model import (  # noqa: E402
    CausalTransformerConfig,
    build_model,
)


def test_preregistered_candidate_and_controls_are_mechanically_matched() -> None:
    config = scale.read_json(ROOT / "configs" / "neural_seed_50m_scale_preregistration.json")
    vocab = scale.read_json(ROOT / "runtime" / "moecot_language_seed_v1" / "exact_language_vocab.json")
    observed = scale.architecture_contract(config, vocab)
    assert 50_000_000 <= observed["active_parameter_count_per_request"] <= 100_000_000
    assert observed["total_parameter_count"] > observed["active_parameter_count_per_request"]
    for control in ("dense_active_parameter", "dense_total_parameter"):
        assert abs(observed[control]["delta"]) <= observed[control]["ff_width_parameter_increment"]
    assert observed["router_parameter_count"] == 0


def test_task_contract_requires_replayable_content_identity(tmp_path: Path) -> None:
    ledger = tmp_path / "units.jsonl.gz"
    with gzip.open(ledger, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"unit_id": "one"}) + "\n")
    report = {
        "policy": "project_theseus_task_complete_training_units_v1",
        "contract_state": "GREEN",
        "coverage_state": "YELLOW",
        "summary": {"contract_hard_gap_count": 0},
        "ledger_receipt": {
            "path": str(ledger),
            "sha256": scale.file_sha256(ledger),
            "replay_valid": True,
        },
        "coverage": {},
    }
    observed = scale.task_complete_contract(report)
    assert observed["contract_ready"] is True
    assert observed["coverage_ready"] is False
    with ledger.open("ab") as handle:
        handle.write(b"tamper")
    assert scale.task_complete_contract(report)["contract_ready"] is False


def test_scale_preregistration_rejects_stale_admission_tcb(tmp_path: Path) -> None:
    task_report = tmp_path / "task.json"
    tcb_path = tmp_path / "tcb.json"
    task_report.write_text('{"policy":"task"}')
    tcb = {
        "policy": "project_theseus_training_admission_epistemic_tcb_v1",
        "trigger_state": "GREEN",
        "hard_gaps": [],
        "summary": {"mutation_count": 17, "surviving_mutant_count": 0},
        "input_artifacts": {
            "task_report": {
                "path": scale.relative(task_report),
                "sha256": scale.file_sha256(task_report),
            }
        },
    }
    tcb_path.write_text(json.dumps(tcb))
    admission = {
        "policy": "project_theseus_training_data_admission_v1",
        "trigger_state": "YELLOW",
        "gates": [],
        "summary": {"training_admission_epistemic_tcb_qualified": True},
        "training_admission_epistemic_tcb": {
            "path": scale.relative(tcb_path),
            "sha256": scale.file_sha256(tcb_path),
            "qualified": True,
        },
    }
    assert scale.training_admission_contract(
        admission, tcb, task_report_path=task_report, tcb_path=tcb_path
    )["contract_ready"] is True
    task_report.write_text('{"policy":"tampered"}')
    assert scale.training_admission_contract(
        admission, tcb, task_report_path=task_report, tcb_path=tcb_path
    )["contract_ready"] is False


def test_canary_encoding_has_target_only_loss_mask() -> None:
    vocab = scale.read_json(ROOT / "runtime" / "moecot_language_seed_v1" / "exact_language_vocab.json")
    unit = {
        "visible_context": '{"messages":[{"role":"user","content":"Say hello"}]}',
        "target": "Hello.",
    }
    batch = scale.encode_canary_batch(unit, vocab, sequence_length=64)
    assert batch["inputs"].shape == (1, 64)
    assert batch["labels"].shape == (1, 64)
    assert 0 < int(batch["mask"].sum()) < 64
    separator = list(batch["inputs"][0]).index(2)
    first_supervised = int(next(index for index, value in enumerate(batch["mask"][0]) if value))
    assert first_supervised > separator


def test_capacity_receipt_does_not_treat_repetition_as_unique_data() -> None:
    observed = scale.canonical_capacity({
        "data_model_scaling_contract": {
            "canonical_corpus_receipt": {
                "valid": True,
                "content_bound": True,
                "unique_model_visible_positions": 123,
                "optimizer_token_positions": 999999,
                "hard_gaps": [],
            }
        }
    })
    assert observed["receipt_valid"] is True
    assert observed["unique_model_visible_positions"] == 123
    assert observed["optimizer_repetition_counted_as_unique_data"] is False


def test_specialist_data_support_fails_closed_per_parameter_owner() -> None:
    architecture = {"expert_parameter_count_per_arm": 10}
    capacity = {
        "domain_unique_positions": {"english_natural_language_total": 200},
        "code_language_unique_positions": {
            "python": 200,
            "javascript_typescript": 200,
            "html_css": 199,
            "rust": 200,
        },
    }
    rejected = scale.specialist_data_support(
        architecture, capacity, minimum_ratio=20.0
    )
    assert rejected["ready"] is False
    assert rejected["shortfall_arms"] == ["html_css"]
    assert rejected["arms"]["html_css"]["shortfall_positions"] == 1
    assert rejected["optimizer_repetition_counted_as_unique_data"] is False

    capacity["code_language_unique_positions"]["html_css"] = 200
    accepted = scale.specialist_data_support(
        architecture, capacity, minimum_ratio=20.0
    )
    assert accepted["ready"] is True
    assert accepted["shortfall_arms"] == []


def test_pointer_generator_forward_and_checkpoint_replay_are_deterministic(
    tmp_path: Path,
) -> None:
    import mlx.core as mx
    import mlx.nn as nn

    lookup = np.full(96, -1, dtype=np.int32)
    lookup[10] = 0
    lookup[11] = 40
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
        expert_adapter_dim=8,
        source_expert_adapter_dim=4,
    )

    def instantiate() -> object:
        return build_model(
            config,
            mx=mx,
            nn=nn,
            source_to_target_lookup=mx.array(lookup, dtype=mx.int32),
        )

    mx.random.seed(20260715)
    model = instantiate()
    tokens = mx.array(
        [[1, 10, 20, 21, 11, 22, 23, 24, 25, 26, 2, 40, 41]],
        dtype=mx.int32,
    )
    reference, _ = model(tokens)
    mx.eval(reference)
    for _ in range(8):
        repeated, _ = model(tokens)
        mx.eval(repeated)
        assert bool(mx.array_equal(reference, repeated))

    checkpoint = tmp_path / "pointer_generator.safetensors"
    model.save_weights(str(checkpoint))
    restored = instantiate()
    restored.load_weights(str(checkpoint))
    replayed, _ = restored(tokens)
    mx.eval(replayed)
    assert bool(mx.array_equal(reference, replayed))
