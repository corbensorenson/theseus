from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_six_row_environment_preflight as owner  # noqa: E402
import theseus_vcm_six_row_environment_preflight_audit as audit_owner  # noqa: E402


def test_six_row_environment_preflight_is_ready_and_call_free() -> None:
    report = owner.evaluate(ROOT / "configs/theseus_vcm_six_row_environment_preflight.json")
    assert report["trigger_state"] == "GREEN"
    assert report["execution_ready"] is True
    assert report["task_count"] == 6
    assert report["untrusted_build_risk_classes_qualified"] is True
    assert report["network_or_dependency_execution_performed"] is False
    assert report["candidate_or_control_calls"] == 0
    assert report["panel_admitted"] is False


def test_six_row_environment_preflight_is_role_separately_rederived() -> None:
    report = audit_owner.audit(ROOT / "configs/theseus_vcm_six_row_environment_preflight.json")
    assert report["trigger_state"] == "GREEN"
    assert report["execution_ready_rederived"] is True
    assert report["required_incremental_peak_bytes"] == 5807060056
    assert report["network_or_dependency_execution_performed"] is False
