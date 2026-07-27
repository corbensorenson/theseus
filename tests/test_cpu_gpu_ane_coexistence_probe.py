import json
import sys

import pytest

from scripts.cpu_gpu_ane_coexistence_probe import (
    CoexistenceFault,
    execute,
    final_json,
    validate_command,
)


def command(label: str) -> list[str]:
    payload = (
        {"trigger_state": "GREEN", "mean_milliseconds": 1.0}
        if label == "cpu"
        else {
            "trigger_state": "GREEN_OPERATOR",
            "runtime": {"mean_milliseconds": 1.0},
        }
    )
    return [
        sys.executable,
        "-c",
        f"import json,time; time.sleep(0.01); print(json.dumps({payload!r}))",
    ]


def test_final_json_ignores_non_json_prefix() -> None:
    payload = final_json(b"progress\n{\"trigger_state\":\"GREEN\"}\n")
    assert payload == {"trigger_state": "GREEN"}


def test_validate_command_rejects_shell_string() -> None:
    with pytest.raises(CoexistenceFault, match="cpu_must_be_nonempty"):
        validate_command("python worker.py", "cpu")


def test_three_worker_execute_reports_overlap_and_slowdowns() -> None:
    report = execute(
        {label: command(label) for label in ("cpu", "gpu", "ane")},
        rounds=2,
    )
    assert report["all_commands_green"] is True
    assert report["actual_overlap_observed"] is True
    assert report["trigger_state"] == "GREEN_THREE_ENGINE_MECHANICAL_OVERLAP"
    assert report["overlap_speedup_vs_serial_sum"] > 1.0
    assert set(report["workers"]) == {"cpu", "gpu", "ane"}
    assert all(
        report["workers"][label]["kernel_slowdown"] > 0.0
        for label in report["workers"]
    )
    assert all(
        "worker_payload" not in row
        for label in ("cpu", "gpu", "ane")
        for row in report["raw"]["standalone"][label]
    )
