from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from moecot_dense_exact_recovery_diagnostic import ARMS, build_diagnostic


PLAN = "a" * 64
STAGE = "b" * 64


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def seed_target(root: Path, target: str, *, complete: bool = True) -> None:
    directory = root / "checkpoints/moecot_language_seed_v8" / target
    checkpoint = directory / "weights.bin"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes((target * 3).encode())
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    receipt = {
        "complete": complete,
        "plan_sha256": PLAN,
        "stage_signature": STAGE,
        "checkpoint": str(checkpoint.relative_to(root)),
        "checkpoint_sha256": digest,
        "parameter_count": 100,
        "trainable_parameter_count": 10,
        "optimizer_steps": 2,
        "optimizer_positions": 20,
        "wall_seconds": 1.0,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "templates_renderers_routers_tools_credit": 0,
    }
    if target in ARMS:
        shared = root / "checkpoints/moecot_language_seed_v8/shared_trunk/weights.bin"
        receipt["shared_trunk_checkpoint"] = str(shared.relative_to(root))
        receipt["shared_trunk_checkpoint_sha256"] = hashlib.sha256(shared.read_bytes()).hexdigest()
    write_json(directory / "training_receipt.json", receipt)
    if target == "shared_trunk":
        return
    by_arm = {}
    arms = [target] if target in ARMS else list(ARMS)
    for arm in arms:
        by_arm[arm] = {
            "row_count": 4,
            "exact_match_count": 1,
            "nonempty_count": 4,
            "byte_serialization_valid_count": 4,
            "syntax_checked_count": 4 if arm == "python" else 0,
            "syntax_valid_count": 3 if arm == "python" else 0,
        }
    write_json(
        directory / "evaluation_private_dev_receipt.json",
        {
            "checkpoint_sha256": digest,
            "candidate_family": "direct_autoregressive_model_text",
            "target_visible_to_generator": False,
            "by_arm": by_arm,
            "summary": next(iter(by_arm.values())),
            "public_training_rows_written": 0,
            "public_benchmark_payload_count": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "templates_renderers_routers_tools_credit": 0,
        },
    )


def test_complete_diagnostic_defers_architecture_verdict(tmp_path: Path) -> None:
    for target in ("shared_trunk", *ARMS, "dense_active_parameter", "dense_total_parameter"):
        seed_target(tmp_path, target)

    report = build_diagnostic(tmp_path, {"v8_plan_sha256": PLAN, "v8_stage_signature": STAGE, "case_contract_sha256": "c" * 64, "evaluation_state": "NOT_EVALUATED"})

    assert report["publication_ready"] is True
    assert report["trigger_state"] == "GREEN"
    assert report["architecture_verdict"] == "DEFER_TO_FROZEN_FUNCTIONAL_UTILITY"
    assert report["boundaries"]["functional_utility_claimed"] is False
    assert report["moecot"]["summary"]["row_count"] == 20
    assert report["dense_controls"]["dense_active_parameter"]["summary"]["row_count"] == 20


def test_incomplete_control_fails_publication_closed(tmp_path: Path) -> None:
    for target in ("shared_trunk", *ARMS, "dense_active_parameter", "dense_total_parameter"):
        seed_target(tmp_path, target, complete=target != "dense_total_parameter")

    report = build_diagnostic(tmp_path, {"v8_plan_sha256": PLAN, "v8_stage_signature": STAGE})

    assert report["publication_ready"] is False
    assert report["trigger_state"] == "YELLOW"
    assert "dense_total_parameter:training_incomplete" in report["hard_gaps"]


def test_shared_trunk_mutation_invalidates_every_expert(tmp_path: Path) -> None:
    for target in ("shared_trunk", *ARMS, "dense_active_parameter", "dense_total_parameter"):
        seed_target(tmp_path, target)
    shared = tmp_path / "checkpoints/moecot_language_seed_v8/shared_trunk/weights.bin"
    shared.write_bytes(b"mutated")

    report = build_diagnostic(tmp_path, {"v8_plan_sha256": PLAN, "v8_stage_signature": STAGE})

    assert report["publication_ready"] is False
    assert "shared_trunk:checkpoint_identity_mismatch" in report["hard_gaps"]
    assert all(
        f"{arm}:shared_checkpoint_identity_mismatch" in report["hard_gaps"]
        for arm in ARMS
    )
