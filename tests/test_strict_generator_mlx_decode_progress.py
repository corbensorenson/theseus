from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from strict_generator_mlx_decode_reporting import (  # noqa: E402
    bind_decode_progress_split,
    commit_decode_progress_batch,
    initialize_decode_progress,
)


def _contract(value: str = "a") -> dict[str, object]:
    return {
        "policy": "strict_mlx_decode_run_contract_v1",
        "checkpoint_sha256": value * 64,
        "vocab_sha256": "b" * 64,
        "decode_cache_mode": "incremental",
    }


def test_progress_commit_is_atomic_private_and_resumable(tmp_path: Path) -> None:
    path = tmp_path / "decode.progress.json"
    progress = initialize_decode_progress(path, run_contract=_contract(), resume=False)
    split = bind_decode_progress_split(
        progress,
        path,
        split_name="family_disjoint",
        task_input_hashes=["task-a", "task-b"],
    )
    assert split["completed"] == {}

    commit_decode_progress_batch(
        progress,
        path,
        split_name="family_disjoint",
        records=[
            {
                "task_index": 0,
                "task_input_hash": "task-a",
                "decoded": [{"body": "return value"}],
                "diagnostic": {"accepted_candidate_rows": 1},
            }
        ],
        batch_receipt={"decode_cache_receipt": {"mode": "incremental"}},
    )

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["splits"]["family_disjoint"]["completed"]["0"]["decoded"]
    assert persisted["public_training_rows"] == 0
    assert persisted["external_inference_calls"] == 0
    assert "prompt" not in json.dumps(persisted).lower()
    assert not list(tmp_path.glob("*.tmp"))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    resumed = initialize_decode_progress(path, run_contract=_contract(), resume=True)
    assert resumed["resume_count"] == 1
    assert len(resumed["splits"]["family_disjoint"]["completed"]) == 1


def test_progress_rejects_stale_contract_and_task_inventory(tmp_path: Path) -> None:
    path = tmp_path / "decode.progress.json"
    progress = initialize_decode_progress(path, run_contract=_contract(), resume=False)
    bind_decode_progress_split(
        progress,
        path,
        split_name="family_disjoint",
        task_input_hashes=["task-a"],
    )
    with pytest.raises(ValueError, match="run contract mismatch"):
        initialize_decode_progress(path, run_contract=_contract("c"), resume=True)
    with pytest.raises(ValueError, match="task inventory mismatch"):
        bind_decode_progress_split(
            progress,
            path,
            split_name="family_disjoint",
            task_input_hashes=["task-other"],
        )


def test_progress_rejects_corruption_and_symlink_paths(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt or unreadable"):
        initialize_decode_progress(corrupt, run_contract=_contract(), resume=True)

    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    symlink = tmp_path / "linked.json"
    symlink.symlink_to(target)
    with pytest.raises(ValueError, match="symlinked"):
        initialize_decode_progress(symlink, run_contract=_contract(), resume=False)


def test_progress_rejects_wrong_task_hash_before_commit(tmp_path: Path) -> None:
    path = tmp_path / "decode.progress.json"
    progress = initialize_decode_progress(path, run_contract=_contract(), resume=False)
    bind_decode_progress_split(
        progress,
        path,
        split_name="family_disjoint",
        task_input_hashes=["task-a"],
    )
    with pytest.raises(ValueError, match="task hash mismatch"):
        commit_decode_progress_batch(
            progress,
            path,
            split_name="family_disjoint",
            records=[
                {
                    "task_index": 0,
                    "task_input_hash": "wrong",
                    "decoded": [],
                    "diagnostic": {},
                }
            ],
            batch_receipt={},
        )
