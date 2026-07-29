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


def make_checkpoint_directory(root: Path, name: str = "run") -> Path:
    checkpoint = root / name / "english_kerc"
    checkpoint.mkdir(parents=True)
    for step in (7, 8):
        (checkpoint / f"weights.step-{step:08d}.safetensors").write_bytes(
            f"weights-{step}".encode()
        )
        (checkpoint / f"optimizer.step-{step:08d}.safetensors").write_bytes(
            f"optimizer-{step}".encode()
        )
        (
            checkpoint / f"optimizer.step-{step:08d}.mlx-rng.safetensors"
        ).write_bytes(f"rng-{step}".encode())
    (checkpoint / "weights.safetensors").write_bytes(b"weights-8")
    (checkpoint / "optimizer.safetensors").write_bytes(b"optimizer-8")
    (checkpoint / "optimizer.mlx-rng.safetensors").write_bytes(b"rng-8")
    (checkpoint / "weights.step-00000007.merged-fp32.safetensors").write_bytes(
        b"diagnostic"
    )
    (checkpoint / "training_receipt.json").write_text("{}", encoding="utf-8")
    old = 1_700_000_000
    for path in checkpoint.iterdir():
        os.utime(path, (old, old))
    return checkpoint


def test_checkpoint_compaction_preserves_terminal_aliases_and_diagnostics(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime" / "t0a_canaries"
    checkpoint = make_checkpoint_directory(root)

    candidates, rejected, preserved = (
        retention.discover_superseded_checkpoints(
            root,
            min_age_hours=0,
            tracked=set(),
            now_timestamp=1_700_100_000,
        )
    )

    assert rejected == []
    assert len(candidates) == 3
    assert {row["step"] for row in candidates} == {7}
    assert preserved[0]["terminal_step"] == 8
    assert preserved[0]["terminal_alias_parity"] is True
    assert preserved[0]["merged_diagnostic_file_count"] == 1

    actions = [
        retention.remove_checkpoint_generation(root, row) for row in candidates
    ]
    assert {row["status"] for row in actions} == {"deleted"}
    assert not (checkpoint / "weights.step-00000007.safetensors").exists()
    assert (checkpoint / "weights.step-00000008.safetensors").exists()
    assert (checkpoint / "weights.safetensors").read_bytes() == b"weights-8"
    assert (
        checkpoint / "weights.step-00000007.merged-fp32.safetensors"
    ).exists()
    assert (checkpoint / "training_receipt.json").exists()


def test_checkpoint_compaction_rejects_terminal_alias_mismatch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime" / "t0a_canaries"
    checkpoint = make_checkpoint_directory(root)
    (checkpoint / "optimizer.safetensors").write_bytes(b"wrong")

    candidates, rejected, preserved = (
        retention.discover_superseded_checkpoints(
            root,
            min_age_hours=0,
            tracked=set(),
            now_timestamp=1_700_100_000,
        )
    )

    assert candidates == []
    assert preserved == []
    assert rejected == [
        {
            "path": str(checkpoint),
            "reason": "terminal_alias_digest_mismatch:optimizer",
        }
    ]


def test_changed_checkpoint_generation_is_not_removed(tmp_path: Path) -> None:
    root = tmp_path / "runtime" / "t0a_canaries"
    checkpoint = make_checkpoint_directory(root)
    candidates, _, _ = retention.discover_superseded_checkpoints(
        root,
        min_age_hours=0,
        tracked=set(),
        now_timestamp=1_700_100_000,
    )
    candidate = next(row for row in candidates if row["kind"] == "weights")
    (checkpoint / "weights.step-00000007.safetensors").write_bytes(b"changed")

    action = retention.remove_checkpoint_generation(root, candidate)

    assert action["status"] == "failed"
    assert "checkpoint_generation_changed_after_manifest" in action["error"]


def test_closed_canary_run_requires_no_canonical_reference(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime" / "t0a_canaries"
    checkpoint = make_checkpoint_directory(root / "family", "run")
    run = checkpoint.parent
    canonical = [
        {
            "source": "configs/example.json",
            "line": "7",
            "reference": retention.relative(run),
        }
    ]

    protected = retention.inspect_canary_run(
        root,
        run,
        min_age_hours=0,
        now_timestamp=1_700_100_000,
        tracked=set(),
        authority_references=canonical,
        evidence_references=[],
    )
    candidate = retention.inspect_canary_run(
        root,
        run,
        min_age_hours=0,
        now_timestamp=1_700_100_000,
        tracked=set(),
        authority_references=[],
        evidence_references=[
            {
                "source": "reports/result.json",
                "line": "9",
                "reference": retention.relative(run) + "/english_kerc",
            }
        ],
    )

    assert protected["authority_references"] == canonical
    assert candidate["authority_references"] == []
    assert candidate["evidence_reference_files"] == ["reports/result.json"]
    assert candidate["terminal_checkpoints"][0]["terminal_alias_parity"] is True


def test_closed_canary_run_exact_manifest_removal(tmp_path: Path) -> None:
    root = tmp_path / "runtime" / "t0a_canaries"
    checkpoint = make_checkpoint_directory(root / "family", "run")
    run = checkpoint.parent
    candidate = retention.inspect_canary_run(
        root,
        run,
        min_age_hours=0,
        now_timestamp=1_700_100_000,
        tracked=set(),
        authority_references=[],
        evidence_references=[],
    )

    action = retention.remove_canary_run(root, candidate)

    assert action["status"] == "deleted"
    assert not run.exists()
