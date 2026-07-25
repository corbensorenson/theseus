from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import hive_node


def test_report_path_supports_workspace_relative_and_external_absolute_paths(
    tmp_path: Path,
) -> None:
    assert hive_node.rel_path(ROOT / "reports" / "status.json") == "reports/status.json"
    assert hive_node.rel_path(tmp_path / "status.json") == (tmp_path / "status.json").as_posix()
