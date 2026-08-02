from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_p4v2r2r3_autonomous_launch as launcher  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "theseus_p4v2r2r3_autonomous_launch.json"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_launch_bindings_are_exact_and_no_human_or_token_gate_exists() -> None:
    value = config()

    assert launcher.validate_config(value) == []
    assert launcher.audit_bindings(value)["passed"] is True
    assert value["authority"]["user_or_operator_approval_required"] is False
    assert value["authority"]["project_selected_quality_token_cap_allowed"] is False


def test_machine_predicates_authorize_only_the_fresh_zero_call_campaign() -> None:
    report = launcher.preflight(
        config(),
        config_path=CONFIG_PATH,
        overrides={
            "power": {"external_connected": True, "discharging": False},
            "memory": {"passed": True},
            "disk": {"passed": True},
            "runtime": {"passed": True},
            "metal": {"passed": True},
            "jobs": [],
            "lease_exists": False,
            "campaign": {
                "trigger_state": "GREEN",
                "pending_tasks": 10,
                "model_calls_retained": 0,
            },
        },
    )

    assert report["trigger_state"] == "GREEN"
    assert report["launch_authorized"] is True
    assert report["failed_gates"] == []
