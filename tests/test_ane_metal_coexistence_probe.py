from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import ane_metal_coexistence_probe as coexistence


def test_command_validation_rejects_shell_string() -> None:
    with pytest.raises(coexistence.CoexistenceFault):
        coexistence.validate_command("python3 -c pass", "gpu")


def test_two_sleeping_processes_prove_harness_overlap_only() -> None:
    command = [
        sys.executable,
        "-c",
        "import time; time.sleep(0.08)",
    ]

    report = coexistence.execute(command, command, rounds=2)

    assert report["all_commands_green"] is True
    assert report["actual_overlap_observed"] is True
    assert report["overlap_speedup_vs_serial_sum"] > 1.2
    assert report["checkpoint_mutated"] is False
    assert "does not prove zero-copy" in report["claim_scope"]
