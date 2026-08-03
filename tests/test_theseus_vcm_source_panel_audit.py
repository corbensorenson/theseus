from __future__ import annotations

import hashlib
import io
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_vcm_source_panel_audit as audit  # noqa: E402


def test_real_panel_fails_closed_only_for_six_language_replacements() -> None:
    report = audit.audit()
    assert report["trigger_state"] == "RED"
    assert report["state"] == "SOURCE_PANEL_LANGUAGE_REPLACEMENTS_REQUIRED"
    assert report["source_panel_admitted"] is False
    assert report["archive_integrity_green"] is True
    assert report["assembled_task_count"] == 62
    assert report["unique_repository_count"] == 62
    assert report["archive_receipt_count"] == 248
    assert report["member_receipt_count"] == 695
    assert report["selected_source_difference_count"] == 62
    assert report["selected_verifier_difference_count"] == 62
    assert [row["index"] for row in report["replacement_slots_required"]] == [1, 12, 19, 48, 51, 56]
    assert report["faults"] == [
        f"task_{index:02d}:natural_language_out_of_scope"
        for index in [1, 12, 19, 48, 51, 56]
    ]
    assert report["local_model_calls"] == 0
    assert report["external_reference_calls"] == 0


def test_task_28_replacement_archive_receipt_is_independently_green() -> None:
    report = audit.p2a.read_json(ROOT / "reports" / "theseus_vcm_source_replacement_28.json")
    row = report["replacement_materialization"]
    for receipt in row["archives"].values():
        faults, hashes, total_bytes = audit.audit_archive(ROOT / receipt["path"], receipt)
        assert faults == []
        assert len(hashes) == len(receipt["members"])
        assert total_bytes == sum(member["bytes"] for member in receipt["members"])


def test_archive_hash_tamper_fails_before_tar_trust(tmp_path: Path) -> None:
    path = tmp_path / "bad.tar.gz"
    with tarfile.open(fileobj=io.BytesIO(), mode="w:gz"):
        pass
    path.write_bytes(b"not-the-bound-archive")
    faults, hashes, total_bytes = audit.audit_archive(
        path,
        {"sha256": hashlib.sha256(b"different").hexdigest(), "members": [], "root": "x"},
    )
    assert faults == ["archive_hash_invalid"]
    assert hashes == {}
    assert total_bytes == 0


def test_language_signals_cover_reviewed_non_english_families() -> None:
    assert audit.deterministic_non_english_signal("Login-Schutzschild für den GKE-Rollout absichern")
    assert audit.deterministic_non_english_signal("Tenant-Namen in der Seitenleiste anzeigen")
    assert audit.deterministic_non_english_signal("리뷰 스킬의 실행 요구 제거")
    assert audit.deterministic_non_english_signal("модель выбирает проект")
    assert not audit.deterministic_non_english_signal("Fix the compatibility window")
