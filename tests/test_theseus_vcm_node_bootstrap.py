from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_node_bootstrap as owner  # noqa: E402


def test_preflight_binds_exact_runtime_without_dependency_or_repository_authority() -> None:
    path = ROOT / "configs" / "theseus_vcm_node_bootstrap.json"
    report = owner.preflight(json.loads(path.read_text()), path)
    assert report["trigger_state"] == "PAUSED"
    assert report["network_requests"] == 0
    assert report["dependency_installations"] == 0
    assert report["repository_executions"] == 0
    assert report["candidate_or_control_calls"] == 0


def test_member_and_symlink_validation_stays_inside_archive_root() -> None:
    root = "node-v22.20.0-darwin-arm64"
    assert owner.normalize_member(f"{root}/bin/node", root) == f"{root}/bin/node"
    assert owner.normalize_member(f"{root}/../escape", root) is None
    assert owner.normalize_member("other/bin/node", root) is None
    assert owner.safe_symlink_target(f"{root}/bin/npm", "../lib/node_modules/npm/bin/npm-cli.js", root)
    assert not owner.safe_symlink_target(f"{root}/bin/npm", "../../escape", root)
