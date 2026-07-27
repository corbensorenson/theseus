from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import selected_route_working_set_calibration as calibration


def write_pair(root: Path, index: int, *, peak: float, passed: bool = True):
    route = root / f"route-{index}.json"
    host = root / f"host-{index}.json"
    route.write_text(
        json.dumps(
            {
                "optimizer_steps": 64,
                "resume_validation": "GREEN",
                "checkpoint_sha256": f"checkpoint-{index}",
                "optimizer_state_sha256": f"optimizer-{index}",
            }
        )
    )
    host.write_text(
        json.dumps(
            {
                "passed": passed,
                "host_resource_safety": {
                    "passed": passed,
                    "fault": "",
                    "command": [
                        "python",
                        "child.py",
                        "--config",
                        "/repo/configs/moecot_language_arm_training.json",
                        "--steps",
                        "64",
                    ],
                    "maximum_inferred_unified_memory_mib": peak,
                    "minimum_reclaimable_available_mib": 1500.0,
                    "maximum_swapout_growth_mib": 0.0,
                },
            }
        )
    )
    return route, host


def test_calibration_derives_envelope_from_two_successful_receipts(
    tmp_path: Path,
) -> None:
    report = calibration.calibrate(
        [
            write_pair(tmp_path, 1, peak=4200.0),
            write_pair(tmp_path, 2, peak=4800.0),
        ]
    )
    assert report["passed"] is True
    assert report["maximum_inferred_unified_memory_mib"] == 4800.0
    assert report["successful_receipt_count"] == 2
    assert "no fixed remaining-memory floor" in report["selection_rule"]


def test_failed_receipt_cannot_calibrate_working_set(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="successful exact"):
        calibration.calibrate(
            [
                write_pair(tmp_path, 1, peak=4200.0),
                write_pair(tmp_path, 2, peak=4800.0, passed=False),
            ]
        )
