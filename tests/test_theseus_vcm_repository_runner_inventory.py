import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_repository_runner_inventory as owner

REPORT = owner.audit(ROOT / "configs/theseus_vcm_repository_runner_inventory.json")


def test_inventory_is_static_and_covers_exact_closures():
    report = REPORT
    assert report["trigger_state"] == "GREEN"
    assert report["observations"]["task_count"] == 62
    assert report["observations"]["tasks_with_parent_and_target_source_closure"] == 62
    assert report["parent_target_or_evaluator_executions"] == 0
    assert report["candidate_or_control_calls"] == 0


def test_runner_receipts_are_source_bound_not_hand_authored():
    report = REPORT
    receipts = [receipt for row in report["rows"] for receipt in row["target"]["runner_receipts"]]
    assert receipts
    assert all(receipt["kind"] in {"ci_run_line", "package_json_script", "cargo_manifest_standard", "python_manifest_pytest"} for receipt in receipts)
    assert all(len(receipt["receipt_sha256"]) == 64 for receipt in receipts)
    assert report["runner_derivation_policy"]["hand_authored_repository_specific_commands"] is False


def test_catomic_exact_license_case_has_cargo_manifest_lock_and_runner():
    report = REPORT
    row = report["rows"][31]
    assert row["repository"] == "maelguimet/catomic"
    assert "Cargo.toml" in row["target"]["root_manifests"]
    assert "Cargo.lock" in row["target"]["root_locks"]
    assert any(receipt["command"] == "cargo test" for receipt in row["target"]["runner_receipts"])
