from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import phase0_reproducibility_gate as gate


def test_source_manifest_is_content_and_path_bound(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    source = tmp_path / "scripts" / "example.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    first = gate.source_manifest(tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    second = gate.source_manifest(tmp_path)

    assert first["file_count"] == 1
    assert first["sha256"] != second["sha256"]


def test_verify_rejects_source_drift(monkeypatch) -> None:
    monkeypatch.setattr(gate, "source_manifest", lambda: {"sha256": "new", "file_count": 1})
    report = {
        "policy": "project_theseus_phase0_reproducibility_gate_v1",
        "trigger_state": "GREEN",
        "source_manifest": {"sha256": "old"},
        "python_environment": {"trigger_state": "GREEN"},
        "python_non_accelerator_suite": {"passed": True},
        "rust_workspace_suite": {"passed": True},
    }

    result = gate.verify(report)

    assert result["trigger_state"] == "RED"
    assert result["faults"] == ["source_manifest_drift"]


def test_missing_report_fails_closed(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["phase0_reproducibility_gate.py", "--gate", "--out", str(tmp_path / "missing.json")])

    assert gate.main() == 2
    assert json.loads(capsys.readouterr().out)["faults"] == ["qualification_report_missing"]
