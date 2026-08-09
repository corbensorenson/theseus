from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_immutable_resolution_segment as owner  # noqa: E402
import theseus_vcm_immutable_resolution_segment_audit as audit_owner  # noqa: E402

CONFIG = ROOT / "configs" / "theseus_vcm_immutable_resolution_segment.json"


def test_preflight_binds_exact_six_immutable_resolution_rows() -> None:
    cfg, bound, faults = owner.preflight(CONFIG)
    assert faults == []
    assert [row["index"] for row in cfg["rows"]] == [12, 13, 16, 25, 35, 56]
    assert {row["manager"] for row in cfg["rows"]} == {"uv", "cargo"}
    assert set(bound["tools"]) == {"uv", "python", "python_3_14", "cargo"}
    assert len(cfg["local_wheels"]) == 4


def test_resolution_contract_has_no_execution_or_model_authority() -> None:
    cfg = owner.p2a.read_json(CONFIG)
    assert cfg["authority"]["package_installation_authorized"] is False
    assert cfg["authority"]["source_build_authorized"] is False
    assert cfg["authority"]["repository_runner_execution_authorized"] is False
    assert cfg["authority"]["parent_target_evaluator_execution_authorized"] is False
    assert cfg["authority"]["local_model_calls_authorized"] is False
    assert cfg["authority"]["external_reference_calls_authorized"] is False
    assert cfg["partial_panel_admission_forbidden"] is True
    assert owner.p2a.resolve(cfg["audit_owner"]) == Path(audit_owner.__file__).resolve()


def test_resolution_outputs_are_one_generic_manifest_driven_family(tmp_path: Path) -> None:
    cfg = owner.p2a.read_json(CONFIG)
    assert cfg["expected_task_count"] == 6
    assert len({row["output_name"] for row in cfg["rows"]}) == 6
    assert all("task-" in row["output_name"] for row in cfg["rows"])
    assert "do not guess" in cfg["declared_dependency_gap_policy"]
    _, bound, faults = owner.preflight(CONFIG)
    assert faults == []
    command, _, _ = owner.resolution_command(cfg, bound, cfg["rows"][0], tmp_path, tmp_path / "cache", tmp_path)
    assert "--no-build" in command
    assert "--only-binary" not in command
    task13 = next(row for row in cfg["rows"] if row["index"] == 13)
    command13, _, _ = owner.resolution_command(cfg, bound, task13, tmp_path, tmp_path / "cache", tmp_path)
    assert str(owner.p2a.resolve(cfg["tools"]["python_3_14"]["path"])) in command13
    assert "--find-links" in command13
    assert str(owner.p2a.resolve(task13["find_links"])) in command13
