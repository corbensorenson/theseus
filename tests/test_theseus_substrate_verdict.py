from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_substrate_verdict as verdict  # noqa: E402


def _train_payload(*, sparse: bool) -> dict:
    specialist_total = 60_000 if sparse else 4_000
    specialist_active = 4_000
    total = 100_000 if sparse else 44_000
    return {
        "summary": {
            "parameter_count": total,
            "optimizer_token_positions_consumed": 100_000,
            "heldout_lm_loss_after": 3.7 if sparse else 3.6,
            "training_tokens_per_second": 800 if sparse else 2300,
            "specialist_core": {
                "specialist_total_parameter_count": specialist_total,
                "specialist_active_parameter_count_per_token": specialist_active,
            },
            "specialist_routing": {"active_expert_count": 20},
        },
        "budget": {
            "source_vocab_sha256": "source",
            "target_vocab_sha256": "target",
            "row_summary": {"encoded_source_rows": 1000},
            "public_training_rows": 0,
            "external_inference_calls": 0,
            "fallback_template_router_tool_credit_count": 0,
        },
    }


def _decode_payload(candidate_rows: int, behavior_passes: int) -> dict:
    return {
        "gates": [
            {"name": "candidate_rows_emitted", "evidence": candidate_rows},
            {
                "name": "functional_pass_moved_above_zero",
                "evidence": {"family_disjoint": behavior_passes, "broad_private_heldout": 0},
            },
        ]
    }


def test_sparse_comparison_prefers_dense_without_behavior_win(tmp_path, monkeypatch) -> None:
    payloads = {
        "sparse_train": _train_payload(sparse=True),
        "dense_train": _train_payload(sparse=False),
        "sparse_decode": _decode_payload(0, 0),
        "dense_decode": _decode_payload(2, 0),
    }
    paths = {}
    for name, payload in payloads.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = str(path)
    monkeypatch.setattr(verdict, "SPECIALIST_REPORTS", paths)
    comparison = verdict.load_specialist_comparison()
    assert comparison["matched_contract"] is True
    assert comparison["active_parameter_relative_gap"] == 0.0
    assert comparison["adoption_state"] == "NOT_ADOPTED"
    assert comparison["practical_route"] == "dense_active_control"
    assert comparison["no_cheat_clean"] is True
