import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_repository_closure_materialization_audit as owner


def test_revised_closures_role_separately_rederive_all_rows():
    report = owner.audit(ROOT / "configs" / "theseus_vcm_repository_closure_materialization_audit.json")
    assert report["trigger_state"] == "GREEN"
    assert report["task_count"] == 62
    assert report["archive_artifact_count"] == 124
    assert report["replayed_unchanged_task_count"] == 59
    assert report["replacement_task_count"] == 3
    assert report["replacement_indices"] == [12, 13, 35]
    assert all(not row["faults"] for row in report["audited_rows"])
    assert report["network_calls"] == 0
    assert report["parent_target_or_evaluator_executions"] == 0
    assert report["local_model_calls"] == 0
