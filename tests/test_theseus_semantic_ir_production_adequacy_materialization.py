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

import theseus_semantic_ir_production_adequacy_materialization as materialization  # noqa: E402


def test_materialization_preflight_is_green_and_zero_call() -> None:
    report = materialization.preflight()
    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["archive_set_admitted"] is False
    assert set(report["counters"].values()) == {0}


def test_deterministic_archive_is_byte_stable_and_normalized(tmp_path: Path) -> None:
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    files = {"src/example.py": b"value = 1\n", "LICENSE": b"license\n"}
    materialization.write_deterministic_archive(first, "opaque-root", files)
    materialization.write_deterministic_archive(second, "opaque-root", files)
    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(fileobj=io.BytesIO(first.read_bytes()), mode="r:gz") as archive:
        members = archive.getmembers()
        assert [row.name for row in members] == [
            "opaque-root/LICENSE",
            "opaque-root/src/example.py",
        ]
        assert all(row.mtime == 0 and row.uid == 0 and row.gid == 0 for row in members)
        assert all(row.mode == 0o644 and row.isfile() for row in members)


def test_unsafe_archive_member_is_rejected(tmp_path: Path) -> None:
    for value in ("../escape", "/absolute", "bad\\windows"):
        try:
            materialization.write_deterministic_archive(
                tmp_path / "bad.tar.gz", "root", {value: b"x"}
            )
        except ValueError as exc:
            assert "unsafe_archive_member" in str(exc)
        else:
            raise AssertionError(value)
