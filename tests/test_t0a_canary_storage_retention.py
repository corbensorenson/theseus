from __future__ import annotations

import importlib.util
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "t0a_canary_storage_retention",
    ROOT / "scripts" / "t0a_canary_storage_retention.py",
)
assert SPEC and SPEC.loader
retention = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retention)


def make_cache(root: Path, name: str = "run") -> Path:
    cache = root / name / retention.CACHE_BASENAME
    cache.mkdir(parents=True)
    (cache / "tensor-a.npy").write_bytes(b"\x93NUMPY" + b"a" * 128)
    (cache / "tensor-b.npy").write_bytes(b"\x93NUMPY" + b"b" * 64)
    old = 1_700_000_000
    for path in (cache, *cache.iterdir()):
        os.utime(path, (old, old))
    return cache


def test_discovers_only_old_flat_npy_cache(tmp_path: Path) -> None:
    root = tmp_path / "runtime" / "t0a_canaries"
    cache = make_cache(root)

    candidates, rejected = retention.discover(
        root,
        min_age_hours=24,
        tracked=set(),
        now_timestamp=1_700_100_000,
    )

    assert rejected == []
    assert len(candidates) == 1
    assert candidates[0]["file_count"] == 2
    assert candidates[0]["classification"].startswith("regenerable_process_local")
    assert cache.exists()


def test_rejects_symlinks_non_npy_and_tracked_files(tmp_path: Path) -> None:
    root = tmp_path / "runtime" / "t0a_canaries"
    symlink_cache = make_cache(root, "symlink")
    (symlink_cache / "link.npy").symlink_to(symlink_cache / "tensor-a.npy")
    non_npy_cache = make_cache(root, "non-npy")
    (non_npy_cache / "receipt.json").write_text("{}", encoding="utf-8")
    tracked_cache = make_cache(root, "tracked")
    tracked = {retention.relative(tracked_cache / "tensor-a.npy")}

    candidates, rejected = retention.discover(
        root,
        min_age_hours=0,
        tracked=tracked,
        now_timestamp=1_700_100_000,
    )

    assert candidates == []
    assert {row["reason"] for row in rejected} == {
        "cache_contains_symlink",
        "cache_contains_non_npy_file",
        "cache_contains_git_tracked_file",
    }


def test_execute_removes_exact_cache_and_preserves_protected_tree(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime" / "t0a_canaries"
    cache = make_cache(root)
    protected = tmp_path / "checkpoints" / "active"
    protected.mkdir(parents=True)
    (protected / "weights.safetensors").write_bytes(b"model")
    before = retention.fingerprint_tree(protected)
    candidates, rejected = retention.discover(
        root,
        min_age_hours=0,
        tracked=set(),
        now_timestamp=1_700_100_000,
    )

    actions = [retention.remove_cache(root, candidates[0])]
    after = retention.fingerprint_tree(protected)

    assert rejected == []
    assert actions[0]["status"] == "deleted"
    assert not cache.exists()
    assert before == after


def test_changed_cache_is_not_removed(tmp_path: Path) -> None:
    root = tmp_path / "runtime" / "t0a_canaries"
    cache = make_cache(root)
    candidates, _ = retention.discover(
        root,
        min_age_hours=0,
        tracked=set(),
        now_timestamp=1_700_100_000,
    )
    (cache / "tensor-a.npy").write_bytes(b"changed")

    action = retention.remove_cache(root, candidates[0])

    assert action["status"] == "failed"
    assert "cache_changed_after_manifest" in action["error"]
    assert cache.exists()
