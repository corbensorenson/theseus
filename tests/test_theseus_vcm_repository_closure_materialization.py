import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_repository_closure_materialization as owner


def test_panel_transforms_to_exact_62_task_registry():
    panel = json.loads((ROOT / "reports/theseus_vcm_source_panel_audit_v2.json").read_text())
    registry = owner.transform_panel(panel)
    assert registry["task_count"] == 62
    assert owner.d1.audit_registry(registry) == []
    assert len({row["repository"] for row in registry["tasks"]}) == 62
    assert registry["tasks"][0]["parent_revision"] == panel["assembled_rows"][0]["base_revision"]
    assert registry["tasks"][0]["target_revision"] == panel["assembled_rows"][0]["head_revision"]


def test_changed_status_is_derived_from_bound_parent_and_target_members():
    panel = json.loads((ROOT / "reports/theseus_vcm_source_panel_audit_v2.json").read_text())
    registry = owner.transform_panel(panel)
    allowed = {"added", "modified", "removed", "changed"}
    assert all(file["status"] in allowed for task in registry["tasks"] for file in task["changed_files"])
    assert all(task["changed_files"] for task in registry["tasks"])


def test_preflight_opens_only_exact_archive_fetch():
    path = ROOT / "configs/theseus_vcm_repository_closure_materialization.json"
    config = json.loads(path.read_text())
    report = owner.preflight(config, path)
    assert report["trigger_state"] == "GREEN"
    assert report["execution_authorized"] is True
    assert report["task_count"] == 62
    assert report["planned_archive_count"] == 124
    assert report["parent_target_or_evaluator_executions"] == 0
    assert report["candidate_or_control_calls"] == 0


def test_storage_policy_is_physical_and_preserves_ten_gib_reserve():
    config = json.loads((ROOT / "configs/theseus_vcm_repository_closure_materialization.json").read_text())
    assert config["physical_storage_policy"]["minimum_free_bytes_after_download"] >= 10 * 1024**3
    assert config["authority"]["user_or_operator_gate"] is False
    assert config["authority"]["untrusted_repository_execution_authorized"] is False
    assert config["retention_policy"]["upstream_bytes_retained_after_normalization"] is False
    assert config["retention_policy"]["upstream_sha256_receipt_retained"] is True
    assert Path(config["tls_ca_bundle"]["path"]).is_file()
    assert owner.sha256_file(Path(config["tls_ca_bundle"]["path"])) == config["tls_ca_bundle"]["sha256"]
