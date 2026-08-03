from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_semantic_ir_production_adequacy_replacement_02_task_pool as pool  # noqa: E402


def test_replacement_packet_is_sealed_and_parent_only() -> None:
    report = pool.materialize()
    assert report["trigger_state"] == "GREEN"
    assert report["opaque_task_id"] == "semantic-ir-adequacy-02r1"
    assert report["target_archive_or_source_visible"] is False
    assert report["evaluator_identity_or_output_visible"] is False
    assert report["conservative_minimum_residual_tokens"] > 0
    assert report["completion_boundary"]["project_selected_quality_token_cap"] is None
    assert report["counters"]["local_model_calls"] == 0


def test_packet_has_no_forbidden_keys_or_hidden_identities() -> None:
    packet = p2a.read_json(pool.PACKET_PATH)
    source = p2a.read_json(pool.SOURCE_REPORT)
    target = p2a.mapping(p2a.mapping(source["archives"])["target"])
    assert pool.audit_packet(packet, source, target) == []
    serialized = json.dumps(packet, sort_keys=True)
    metadata = p2a.mapping(source["metadata"])
    assert metadata["target_revision"] not in serialized
    assert metadata["merge_revision"] not in serialized
    assert pool.EVALUATOR_REPORT_SHA256 not in serialized


def test_task_has_only_declared_effect_authority() -> None:
    task = p2a.read_json(pool.TASK_PATH)
    assert task["allowed_effect_paths"] == ["business_logic_test.py"]
    assert task["effect_authority"] == "disposable_snapshot_only"
    assert task["candidate_visible_context"]["full_selected_parent_sources"] is True
    assert task["candidate_visible_context"]["project_selected_character_or_token_cap"] is None
