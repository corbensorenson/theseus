from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pre_long_run_replacement_freeze as freeze


def test_replacement_freeze_config_has_exact_resume_boundary() -> None:
    config = json.loads(
        (
            ROOT / "configs/pre_long_run_replacement_freeze.json"
        ).read_text(encoding="utf-8")
    )
    assert config["decision"] == (
        "AUTHORIZE_EXACT_STEP_11416_RESUME_UNCHANGED_AFTER_OPERATOR_REMOVES_HOLD"
    )
    assert config["expected"]["optimizer_steps"] == 11416
    assert config["expected"]["optimizer_positions"] == 87441996
    assert config["boundaries"]["long_training_started_or_resumed"] is False
    assert config["boundaries"]["hold_removed_by_freeze"] is False


def test_current_replacement_freeze_is_green_deterministic_and_held() -> None:
    first = freeze.execute(
        ROOT / "configs/pre_long_run_replacement_freeze.json",
        publish_report=False,
    )
    second = freeze.execute(
        ROOT / "configs/pre_long_run_replacement_freeze.json",
        publish_report=False,
    )
    assert first["trigger_state"] == "GREEN"
    assert first["failed_gates"] == []
    assert first["missing_gates"] == []
    assert all(row["passed"] for row in first["gates"].values())
    assert first["package_identity"] == second["package_identity"]
    assert freeze.verify_package_identity(first)
    assert first["checkpoint_custody"]["optimizer_steps"] == 11416
    assert first["checkpoint_custody"]["lineage"]["manifest_count"] == 37
    assert (
        first["checkpoint_custody"]["lineage"][
            "pre_anchor_full_chain_available"
        ]
        is False
    )
    assert first["functional_surface"]["consumed_case_count"] == 0
    assert first["resume_authority"]["operator_hold_present"] is True
    assert first["resume_authority"]["operator_hold_removed_by_package"] is False
    assert first["resume_authority"]["long_training_authorized_now"] is False
    assert first["resume_authority"]["long_training_started_or_resumed"] is False


def test_package_identity_rejects_tampering() -> None:
    report = freeze.execute(
        ROOT / "configs/pre_long_run_replacement_freeze.json",
        publish_report=False,
    )
    tampered = copy.deepcopy(report)
    tampered["checkpoint_custody"]["optimizer_steps"] += 1
    assert freeze.verify_package_identity(tampered) is False


def test_selected_recipe_is_current_control_not_rejected_candidate() -> None:
    report = freeze.execute(
        ROOT / "configs/pre_long_run_replacement_freeze.json",
        publish_report=False,
    )
    recipe = report["selected_architecture"]["execution_recipe"]
    assert recipe["training_step_mode"] == "compiled"
    assert recipe["compute_dtype"] == "float32"
    assert recipe["optimizer_id"] == "adamw_mlx"
    assert recipe["self_attention_projection"] == "separate"
    assert recipe["feed_forward_activation"] == "swiglu"
    assert recipe["residual_policy"] == "sequential_unscaled"
    assert recipe["per_head_muon"] is False
