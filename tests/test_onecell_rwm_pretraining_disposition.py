from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import onecell_rwm_pretraining_disposition as onecell


def test_first_language_campaign_retirement_is_binding() -> None:
    report = onecell.build_report(onecell.load_config())
    assert report["trigger_state"] == "GREEN"
    assert report["disposition"] == "retired_from_first_language_campaign"
    assert report["summary"]["route_authorized"] is False
    assert report["summary"]["optimizer_exposure_steps"] == 0
    assert report["summary"]["checkpoint_member"] is False


def test_substrate_neutral_abi_and_reentry_are_complete() -> None:
    report = onecell.build_report(onecell.load_config())
    assert set(report["cognitive_kernel_abi"]["methods"]) == {
        "initialize", "propose", "accept_receipt", "checkpoint", "restore",
        "parameter_accounting", "resource_accounting",
    }
    assert report["summary"]["objective_term_count"] == 9
    assert report["summary"]["checkpoint_group_count"] == 8
    assert report["summary"]["successor_prerequisite_count"] >= 10


def test_disposition_mutations_fail_closed() -> None:
    controls = onecell.mutation_controls(onecell.load_config())
    assert controls["case_count"] >= 10
    assert controls["passed_count"] == controls["case_count"]
