from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import selected_route_sustained_qualification as sustained


def test_sustained_config_requires_real_two_hour_window() -> None:
    config = json.loads(
        (ROOT / "configs" / "selected_route_sustained_qualification.json").read_text()
    )
    sustained.validate_config(config)
    assert config["require_resource_availability"] is True
    assert config["availability_config"] == (
        "configs/neural_seed_training_availability.json"
    )
    config["minimum_contiguous_child_wall_seconds"] = 7199
    try:
        sustained.validate_config(config)
    except ValueError as exc:
        assert "two hours" in str(exc)
    else:
        raise AssertionError("short sustained window was accepted")


def test_first_middle_last_uses_joined_position_throughput() -> None:
    rows = []
    for index, rate in enumerate((100.0, 90.0, 80.0, 70.0, 60.0), start=1):
        wall = 10.0
        positions = int(rate * wall)
        rows.append(
            {
                "segment_index": index,
                "optimizer_position_delta": positions,
                "child_wall_seconds": wall,
                "device_step_seconds": 8.0,
                "host_resource_safety": {
                    "minimum_reclaimable_available_mib": 3000.0,
                    "maximum_process_rss_mib": 1000.0,
                    "maximum_inferred_unified_memory_mib": 4000.0,
                    "maximum_swapout_growth_mib": 0.0,
                },
                "machine_state_after": {
                    "power": {"on_ac_power": True},
                    "thermal": {"warning_detected": False},
                },
            }
        )
    windows = sustained.first_middle_last(rows, 0.2)
    assert windows["first"]["joined_positions_per_second"] == 100.0
    assert windows["middle"]["joined_positions_per_second"] == 80.0
    assert windows["last"]["joined_positions_per_second"] == 60.0


def test_selected_recipe_matches_canonical_training_config() -> None:
    training_config = json.loads(
        (ROOT / "configs" / "moecot_language_arm_training.json").read_text()
    )
    selector = json.loads(
        (ROOT / "reports" / "training_acceleration_final_selector.json").read_text()
    )
    assert sustained.selected_recipe_from_training(training_config) == selector[
        "selected_recipe"
    ]


def test_availability_pause_is_incomplete_not_training_failure(
    tmp_path: Path,
) -> None:
    config_path = ROOT / "configs" / "selected_route_sustained_qualification.json"
    config = json.loads(config_path.read_text())
    selector_path = ROOT / config["selector_report"]
    selector = json.loads(selector_path.read_text())
    training_config_path = ROOT / config["training_config"]
    training_config = json.loads(training_config_path.read_text())
    canonical = {
        "checkpoint": {"sha256": "a" * 64},
        "optimizer": {"sha256": "b" * 64},
        "receipt": {"sha256": "c" * 64},
    }
    availability = {
        "trigger_state": "PAUSED",
        "failed_gates": ["ac_power"],
    }
    report = sustained.report_for(
        config_path=config_path,
        config=config,
        selector_path=selector_path,
        selector=selector,
        training_config_path=training_config_path,
        training_config=training_config,
        plan={"plan_sha256": "d" * 64},
        canonical_before=canonical,
        canonical_after=canonical,
        rows=[],
        availability_checks=[availability],
        interruption={
            "fault": "availability_paused",
            "segment_index": 1,
            "failed_gates": ["ac_power"],
        },
    )
    assert report["trigger_state"] == "PAUSED"
    assert report["support_state"] == "INCOMPLETE"
    assert report["hard_gaps"] == []


def test_new_sustained_namespace_initializes_before_resolving_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    initialized = False
    target = {"target_id": "shared_trunk"}

    monkeypatch.setattr(
        sustained.training,
        "scratch_target_contract",
        lambda _target, root: {
            "receipt": str(root / "shared_trunk" / "training_receipt.json")
        },
    )

    def initialize(_target, _root):
        nonlocal initialized
        initialized = True

    def target_paths(_target):
        assert initialized
        return {"receipt": tmp_path / "training_receipt.json"}

    monkeypatch.setattr(sustained.fresh, "initialize_scratch", initialize)
    monkeypatch.setattr(sustained.fresh, "target_paths", target_paths)
    paths = sustained.prepare_scratch_paths(
        target,
        tmp_path,
        initialize=True,
    )
    assert paths == {"receipt": tmp_path / "training_receipt.json"}
