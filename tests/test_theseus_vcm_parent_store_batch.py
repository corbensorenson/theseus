from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_parent_only_materializer_audit as base_audit  # noqa: E402
import theseus_vcm_parent_store_batch as owner  # noqa: E402

CONFIG = ROOT / "configs" / "theseus_vcm_parent_store_batch.json"


def test_batch_manifest_and_resource_preflight_are_safe_and_complete() -> None:
    cfg = owner.p2a.read_json(CONFIG)
    faults: list[str] = []
    manifest, rows = owner.load_manifest(cfg, faults)
    resource = owner.resource_preflight(rows, cfg, faults)
    assert faults == []
    assert manifest["policy"] == "project_theseus_vcm_parent_only_batch_manifest_v1"
    assert len(rows) == 62
    assert all(set(row) == base_audit.ROW_KEYS for row in rows)
    assert resource["parent_archive_count"] == 62
    assert resource["execution_gate"] == "OPEN"
    assert resource["content_payload_duplication"] is False


def test_batch_config_preserves_zero_call_and_nonadmission_contract() -> None:
    cfg = owner.p2a.read_json(CONFIG)
    assert cfg["broad_parent_effect_root"] == "repository"
    assert cfg["resource_contract"]["host_reserve_bytes"] == 10 * 1024**3
    assert "model" in cfg["maximum_inference"]
    assert "Luna" in cfg["maximum_inference"]
