from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pre_long_run_replacement_freeze as freeze  # noqa: E402


DIRTY_SOURCE_STATE = {
    "commit": "a" * 40,
    "branch": "main",
    "clean_at_generation": False,
    "dirty_path_count": 1,
    "dirty_paths": [" M maintenance.py"],
}
GREEN_SOURCE_STATE = {
    "commit": "b" * 40,
    "branch": "main",
    "clean_at_generation": True,
    "dirty_path_count": 0,
    "dirty_paths": [],
}


def test_replacement_freeze_config_has_exact_resume_boundary() -> None:
    config = json.loads(
        (ROOT / "configs/pre_long_run_replacement_freeze.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["decision"] == (
        "AUTHORIZE_EXACT_STEP_11416_RESUME_UNCHANGED_THROUGH_MACHINE_PREDICATE_LEASE"
    )
    assert config["expected"]["optimizer_steps"] == 11416
    assert config["expected"]["optimizer_positions"] == 87441996
    assert config["boundaries"]["long_training_started_or_resumed"] is False
    assert config["boundaries"]["machine_authority_bypassed_by_freeze"] is False
    assert "source_binding" in config["required_gates"]
    assert "machine_authority_boundary" in config["required_gates"]


def test_dirty_source_replacement_freeze_is_blocked_deterministic_and_held() -> None:
    first = freeze.execute(
        ROOT / "configs/pre_long_run_replacement_freeze.json",
        publish_report=False,
        source_state_override=DIRTY_SOURCE_STATE,
    )
    second = freeze.execute(
        ROOT / "configs/pre_long_run_replacement_freeze.json",
        publish_report=False,
        source_state_override=DIRTY_SOURCE_STATE,
    )
    assert first["trigger_state"] == "RED"
    assert first["failed_gates"] == ["source_binding"]
    assert first["missing_gates"] == []
    assert first["gates"]["functional_surface"]["passed"] is True
    assert first["gates"]["functional_surface_freshness"]["passed"] is True
    assert first["gates"]["source_binding"]["passed"] is False
    assert first["package_identity"] == second["package_identity"]
    assert freeze.verify_package_identity(first)
    assert first["checkpoint_custody"]["optimizer_steps"] == 11416
    assert first["checkpoint_custody"]["lineage"]["manifest_count"] == 37
    assert (
        first["checkpoint_custody"]["lineage"]["pre_anchor_full_chain_available"]
        is False
    )
    assert first["functional_surface"]["consumed_case_count"] == 0
    assert (
        first["functional_surface"]["integrity"]["state"]
        == "VALID_FRESH_PRIVATE_SURFACE"
    )
    assert first["resume_authority"]["machine_authority_boundary_present"] is True
    assert first["resume_authority"]["machine_authority_bypassed_by_package"] is False
    assert first["resume_authority"]["user_or_operator_approval_required"] is False
    assert first["resume_authority"]["long_training_authorized_now"] is False
    assert first["resume_authority"]["long_training_started_or_resumed"] is False


def test_clean_source_replacement_freeze_is_green_but_preserves_machine_authority() -> None:
    report = freeze.execute(
        ROOT / "configs/pre_long_run_replacement_freeze.json",
        publish_report=False,
        source_state_override=GREEN_SOURCE_STATE,
    )
    assert report["trigger_state"] == "GREEN"
    assert report["failed_gates"] == []
    assert report["gates"]["source_binding"]["passed"] is True
    assert report["source_binding"] == GREEN_SOURCE_STATE
    assert set(report["source_artifacts"]) == {
        "configs/project_manifest_registry.json",
        "configs/roadmap_implementation_matrix.json",
        "configs/neural_seed_training_availability.json",
        "configs/neural_seed_d2_autonomous_evaluation_controller.json",
        "configs/pre_long_run_acceleration_residual_audit.json",
        "configs/pre_long_run_independent_readiness_audit.json",
        "configs/pre_long_run_replacement_freeze.json",
        "scripts/neural_seed_training_campaign.py",
        "scripts/neural_seed_d2_autonomous_evaluation_controller.py",
        "scripts/pre_long_run_acceleration_residual_audit.py",
        "scripts/pre_long_run_independent_readiness_audit.py",
        "scripts/pre_long_run_replacement_freeze.py",
        "tests/test_pre_long_run_acceleration_residual_audit.py",
        "tests/test_pre_long_run_independent_readiness_audit.py",
        "tests/test_pre_long_run_replacement_freeze.py",
    }
    assert report["resume_authority"]["machine_authority_boundary_present"] is True
    assert report["resume_authority"]["machine_authority_bypassed_by_package"] is False
    assert report["resume_authority"]["user_or_operator_approval_required"] is False
    assert report["resume_authority"]["long_training_authorized_now"] is False
    assert report["resume_authority"]["long_training_started_or_resumed"] is False


def test_machine_authority_boundary_rejects_emergency_stop_and_human_gate() -> None:
    training = freeze.read_json(
        ROOT / "configs/neural_seed_training_availability.json"
    )
    d2 = freeze.read_json(
        ROOT / "configs/neural_seed_d2_autonomous_evaluation_controller.json"
    )
    d2["authority"]["user_or_operator_approval_required"] = True
    report = freeze.machine_authority_boundary(
        training,
        d2,
        emergency_yield_present=True,
        active_d2_lease_present=True,
    )
    assert report["passed"] is False
    assert "emergency_yield_requested" in report["faults"]
    assert "d2_forbidden_authority_present" in report["faults"]
    assert "active_d2_evaluation_lease_present" in report["faults"]


def test_package_identity_rejects_tampering() -> None:
    report = freeze.execute(
        ROOT / "configs/pre_long_run_replacement_freeze.json",
        publish_report=False,
        source_state_override=GREEN_SOURCE_STATE,
    )
    tampered = copy.deepcopy(report)
    tampered["checkpoint_custody"]["optimizer_steps"] += 1
    assert freeze.verify_package_identity(tampered) is False


def test_selected_recipe_is_current_control_not_rejected_candidate() -> None:
    report = freeze.execute(
        ROOT / "configs/pre_long_run_replacement_freeze.json",
        publish_report=False,
        source_state_override=GREEN_SOURCE_STATE,
    )
    recipe = report["selected_architecture"]["execution_recipe"]
    assert recipe["training_step_mode"] == "compiled"
    assert recipe["compute_dtype"] == "float32"
    assert recipe["optimizer_id"] == "adamw_mlx"
    assert recipe["self_attention_projection"] == "separate"
    assert recipe["feed_forward_activation"] == "swiglu"
    assert recipe["residual_policy"] == "sequential_unscaled"
    assert recipe["per_head_muon"] is False
