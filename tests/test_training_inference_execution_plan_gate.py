from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from theseus_archive_resolver import is_archive_pointer, resolve_archived_path
from training_inference_execution_plan_gate import (
    check_lane_shapes,
    check_current_t2_training_smoke_if_present,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smoke(checkpoint: Path, vocab: Path) -> dict:
    return {
        "trigger_state": "GREEN",
        "backend": "mlx_high_level_transformer",
        "device": "Device(gpu, 0)",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(resolve_archived_path(checkpoint)),
        "vocab": str(vocab),
        "vocab_sha256": sha256(vocab),
        "training_plan": {"target_token_positions": 100},
        "optimizer_token_positions_consumed": 100,
        "parameter_update_fraction": 1.0,
        "parameter_tensor_update_fraction": 1.0,
        "heldout_lm_improved": True,
        "training_tokens_per_second": 1.0,
        "open_or_pretrained_model_weights_used": False,
        "public_training_rows": 0,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "fallback_template_router_tool_credit_count": 0,
    }


def archive_with_sidecar(tmp_path: Path) -> tuple[Path, Path]:
    logical = tmp_path / "checkpoint.npz"
    archived = tmp_path / "archive" / "checkpoint.npz"
    archived.parent.mkdir()
    archived.write_bytes(b"checkpoint-weights")
    pointer = logical.with_name(logical.name + ".archive-pointer.json")
    pointer.write_text(
        json.dumps(
            {
                "policy": "project_theseus_archived_artifact_pointer_v1",
                "original_path": str(logical),
                "archive_path": str(archived),
                "original_sha256": sha256(archived),
            }
        ),
        encoding="utf-8",
    )
    return logical, archived


def test_t2_gate_resolves_and_hashes_retention_sidecar(tmp_path: Path) -> None:
    logical, archived = archive_with_sidecar(tmp_path)
    vocab = tmp_path / "vocab.json"
    vocab.write_text('{"tokens": []}', encoding="utf-8")
    result = check_current_t2_training_smoke_if_present(
        {"t2_private_training_smoke": smoke(logical, vocab)}
    )
    assert result["passed"]
    assert result["evidence"]["checkpoint_from_archive"] is True
    assert Path(result["evidence"]["checkpoint_resolved"]).name == archived.name
    assert is_archive_pointer(logical)
    assert resolve_archived_path(logical) == archived


def test_t2_gate_rejects_corrupted_archived_payload(tmp_path: Path) -> None:
    logical, archived = archive_with_sidecar(tmp_path)
    vocab = tmp_path / "vocab.json"
    vocab.write_text('{"tokens": []}', encoding="utf-8")
    receipt = smoke(logical, vocab)
    archived.write_bytes(b"corrupted")
    result = check_current_t2_training_smoke_if_present(
        {"t2_private_training_smoke": receipt}
    )
    assert not result["passed"]
    assert "checkpoint_sha256_mismatch" in result["evidence"]["faults"]


def test_live_artifact_wins_over_stale_sidecar(tmp_path: Path) -> None:
    logical, _archived = archive_with_sidecar(tmp_path)
    logical.write_bytes(b"new-live-checkpoint")
    assert not is_archive_pointer(logical)
    assert resolve_archived_path(logical) == logical


def test_five_arm_smoke_and_completed_lane_are_operational() -> None:
    rows = [
        {
            "target_id": target,
            "state": "GREEN",
            "optimizer_steps": 1,
            "capability_claim": "NOT_EVALUATED",
        }
        for target in (
            "english",
            "python",
            "javascript_typescript",
            "html_css",
            "rust",
            "dense_total_parameter",
            "dense_active_parameter",
        )
    ]
    report = {
        "policy": "project_theseus_moecot_language_arm_training_plan_v1",
        "trigger_state": "GREEN",
        "checkpoint_inventory": {
            "state": "GREEN",
            "all_targets_smoke_ready": True,
            "valid_smoke_count": 7,
            "distinct_checkpoint_digest_count": 7,
            "distinct_optimizer_digest_count": 7,
            "capability_claim": "NOT_EVALUATED",
            "rows": rows,
        },
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "templates_renderers_routers_tools_credit": 0,
    }
    result = check_current_t2_training_smoke_if_present(
        {"t2_private_training_smoke": report}
    )
    assert result["passed"]
    lane = {
        "id": "T2_private_training_smoke",
        "title": "Private Training Smoke",
        "status": "completed",
        "goal": "bounded smoke",
        "entry_criteria": ["ready"],
        "required_gates": ["gate"],
        "allowed_inputs": ["private"],
        "forbidden_inputs": ["public benchmark payloads"],
        "evidence_outputs": ["report"],
        "exit_criteria": ["seven receipts"],
        "rollback_or_stop": "quarantine",
        "no_claims": ["not capability"],
    }
    assert check_lane_shapes({"lanes": [lane]})["passed"]
