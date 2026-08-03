from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_yarn_bootstrap as owner  # noqa: E402


def test_preflight_binds_exact_archive_and_denies_execution() -> None:
    path = ROOT / "configs" / "theseus_vcm_yarn_bootstrap.json"
    report = owner.preflight(json.loads(path.read_text()), path)
    assert report["trigger_state"] == "PAUSED"
    assert report["network_requests"] == 0
    assert report["tool_execution_performed"] is False
    assert report["repository_execution_authorized"] is False
    assert report["dependency_installation_authorized"] is False


def test_target_inspection_requires_exact_yarn_identity(tmp_path: Path) -> None:
    target = tmp_path / "yarn"
    (target / "bin").mkdir(parents=True)
    (target / "bin" / "yarn.js").write_text("// trusted test\n")
    (target / "package.json").write_text(json.dumps({"name": "yarn", "version": "1.22.22"}))
    config = json.loads((ROOT / "configs" / "theseus_vcm_yarn_bootstrap.json").read_text())
    receipt, faults = owner.inspect_target(target, config)
    assert faults == []
    assert receipt["version"] == "1.22.22"
