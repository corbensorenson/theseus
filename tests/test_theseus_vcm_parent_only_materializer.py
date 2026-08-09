from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_parent_only_materializer as owner  # noqa: E402
import theseus_vcm_parent_only_materializer_audit as audit_owner  # noqa: E402

CONFIG = ROOT / "configs" / "theseus_vcm_parent_only_materializer.json"


def test_parent_only_store_and_production_abi_materialize() -> None:
    report, store = owner.materialize(CONFIG)
    assert report["trigger_state"] == "GREEN"
    assert report["row_count"] == 4
    assert report["regular_file_count"] > 0
    assert report["text_page_count"] > 0
    assert report["information_flow"]["complete_parent_text_frontier_retained_without_selection_cap"] is True
    for row in report["rows"]:
        assert row["allowed_effect_paths_present"] is False
        assert row["candidate_surface"]["broad_parent_effect_root"] == "repository"
        assert row["vcm_consumer_abi"]["ready"] is True
    for row in store["rows"]:
        assert row["selector"]["selected_page_or_byte_cap"] is None
        assert len(row["selector"]["frontier"]) == sum(
            item["content_class"] == "utf8_text" for item in row["inventory"]
        )


def test_role_separated_audit_rederives_candidate_bytes_and_selector() -> None:
    producer, store = owner.materialize(CONFIG)
    report = audit_owner.audit(CONFIG, producer=producer, store=store)
    assert report["trigger_state"] == "GREEN"
    assert report["audit_kind"] == "role-separated rederivation"
    assert report["audited_row_count"] == 4
    assert report["audited_candidate_visible_field_count"] == 16
    assert all(report["conclusions"].values())
    assert report["candidate_or_control_calls"] == 0
    assert report["external_reference_calls"] == 0


def test_archive_backed_page_read_revalidates_content() -> None:
    _, store = owner.materialize(CONFIG)
    row = store["rows"][0]
    page = next(item for item in row["inventory"] if item["content_class"] == "utf8_text")
    payload = owner.read_parent_page(row, page["path"])
    assert owner.sha256_bytes(payload) == page["sha256"]
    assert len(payload) == page["bytes"]
