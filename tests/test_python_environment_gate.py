from __future__ import annotations

import copy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import python_environment_gate as gate


def test_canonical_cpu_contract_and_requirements_are_exactly_synchronized() -> None:
    contract = gate.load_contract()
    expected = {
        row["distribution"].lower().replace("_", "-"): row["version"]
        for row in gate.profile_packages(contract, "cpu")
    }
    actual = gate.requirement_pins(ROOT / contract["profiles"]["cpu"]["requirements"])

    assert actual == expected


def test_every_profile_binds_an_exact_complete_lock() -> None:
    contract = gate.load_contract()
    for profile, row in contract["profiles"].items():
        requirements = ROOT / row["requirements"]
        lock = ROOT / row["lock"]
        direct = gate.requirement_pins(requirements)
        closure = gate.requirement_pins(lock)

        assert lock.resolve() in gate.declared_constraints(requirements)
        assert gate.sha256_file(lock) == row["lock_sha256"]
        assert direct.items() <= closure.items(), profile


def test_gate_reports_precise_missing_package_and_install_command() -> None:
    contract = gate.load_contract()
    versions = {
        row["distribution"]: row["version"]
        for row in gate.profile_packages(contract, "cpu")
    }
    versions["pyarrow"] = None
    report = gate.audit(
        contract,
        "cpu",
        python_version=(3, 12),
        version_lookup=lambda name: versions[name],
        import_lookup=lambda _module: {"passed": True, "returncode": 0, "stderr": ""},
        inventory_lookup=lambda: {name.lower().replace("_", "-"): version for name, version in versions.items() if version},
    )

    assert report["trigger_state"] == "RED"
    assert report["summary"]["missing_or_mismatched"] == ["pyarrow"]
    assert report["summary"]["install_command"].endswith(
        "-m pip install -r " + str(ROOT / "requirements/theseus-py312-cpu.txt")
    )


def test_terminal_mlx_profile_cannot_hide_requirement_drift() -> None:
    contract = gate.load_contract()
    broken = copy.deepcopy(contract)
    broken["profiles"]["mlx"]["packages"][0]["version"] = "999.0.0"
    versions = {
        row["distribution"]: row["version"]
        for row in gate.profile_packages(broken, "mlx")
    }
    report = gate.audit(
        broken,
        "mlx",
        python_version=(3, 12),
        version_lookup=lambda name: versions[name],
        import_lookup=lambda _module: {"passed": True, "returncode": 0, "stderr": ""},
        inventory_lookup=lambda: {name.lower().replace("_", "-"): version for name, version in versions.items() if version},
    )

    assert report["trigger_state"] == "RED"
    assert report["requirements"]["matches_contract"] is False


def test_lock_digest_drift_fails_closed() -> None:
    contract = gate.load_contract()
    broken = copy.deepcopy(contract)
    broken["profiles"]["cpu"]["lock_sha256"] = "0" * 64
    closure = gate.requirement_pins(ROOT / contract["profiles"]["cpu"]["lock"])
    report = gate.audit(
        broken,
        "cpu",
        python_version=(3, 12),
        version_lookup=lambda name: closure[name.lower().replace("_", "-")],
        import_lookup=lambda _module: {"passed": True, "returncode": 0, "stderr": ""},
        inventory_lookup=lambda: closure,
    )

    assert report["trigger_state"] == "RED"
    assert report["lock"]["digest_matches"] is False
