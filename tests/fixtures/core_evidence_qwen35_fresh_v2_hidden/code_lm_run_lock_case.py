from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import code_lm_run_lock as run_lock  # noqa: E402


def args() -> SimpleNamespace:
    return SimpleNamespace(
        lock_path="runtime/qualification.lock",
        rust_timeout_seconds=1,
        public_timeout_seconds=1,
        sts_timeout_seconds=1,
    )


def test_release_removes_the_acquired_lock(tmp_path: Path) -> None:
    fd = run_lock.acquire_run_lock(args(), tmp_path)
    path = run_lock.resolve_path(args().lock_path, tmp_path)
    assert fd is not None and path.is_file()
    run_lock.release_run_lock(fd, path)
    assert not path.exists()


def test_release_preserves_a_replacement_lock(tmp_path: Path) -> None:
    fd = run_lock.acquire_run_lock(args(), tmp_path)
    path = run_lock.resolve_path(args().lock_path, tmp_path)
    assert fd is not None
    path.unlink()
    replacement = {"pid": 999999, "owner": "replacement"}
    path.write_text(json.dumps(replacement) + "\n", encoding="utf-8")

    run_lock.release_run_lock(fd, path)

    assert path.is_file() and json.loads(
        path.read_text(encoding="utf-8")
    ) == replacement, "request_contract:replacement_run_lock_preserved"


def test_release_does_not_unlink_a_different_path(tmp_path: Path) -> None:
    fd = run_lock.acquire_run_lock(args(), tmp_path)
    acquired = run_lock.resolve_path(args().lock_path, tmp_path)
    other = tmp_path / "runtime" / "other.lock"
    other.write_text('{"owner":"other"}\n', encoding="utf-8")
    assert fd is not None

    run_lock.release_run_lock(fd, other)

    assert other.is_file(), "request_contract:different_run_lock_preserved"
    acquired.unlink(missing_ok=True)
