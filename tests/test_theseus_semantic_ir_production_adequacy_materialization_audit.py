from __future__ import annotations

import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_semantic_ir_production_adequacy_materialization_audit as audit  # noqa: E402


def test_sealed_materialization_audit_is_green() -> None:
    report = audit.audit()
    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["archive_receipt_count"] == 36
    assert report["member_receipt_count"] == 76
    assert report["selected_source_difference_count"] == 18


def test_archive_hash_tamper_fails_before_tar_trust(tmp_path: Path) -> None:
    path = tmp_path / "bad.tar.gz"
    path.write_bytes(b"not an archive")
    faults, hashes = audit.audit_archive(
        path,
        {"sha256": hashlib.sha256(b"different").hexdigest(), "members": [], "root": "x"},
    )
    assert faults == ["archive_hash_invalid"]
    assert hashes == {}
