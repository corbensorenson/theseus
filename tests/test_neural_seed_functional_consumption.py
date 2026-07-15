from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from neural_seed_functional_consumption import (
    ConsumptionError,
    complete_reservation,
    fail_reservation,
    read_registry,
    require_completed_artifact,
    reserve_once,
)


def test_reservation_is_append_only_and_exact_identity_cannot_repeat(tmp_path: Path) -> None:
    registry = tmp_path / "consumption.jsonl"
    identity = {"freeze_sha256": "f" * 64, "target_id": "dense_active_parameter"}
    reservation = reserve_once(registry, stage="candidate_generation", identity=identity)
    completed = complete_reservation(
        registry,
        reservation,
        artifact={"path": "candidate.json", "sha256": "a" * 64},
    )

    assert completed["event"] == "completed"
    assert [row["event"] for row in read_registry(registry)] == ["reserved", "completed"]
    assert require_completed_artifact(
        registry,
        stage="candidate_generation",
        artifact_sha256="a" * 64,
    )["reservation_id"] == reservation["reservation_id"]
    with pytest.raises(ConsumptionError, match="already consumed or reserved"):
        reserve_once(registry, stage="candidate_generation", identity=identity)

    with pytest.raises(ConsumptionError, match="expected one completed artifact"):
        require_completed_artifact(
            registry,
            stage="candidate_generation",
            artifact_sha256="b" * 64,
        )


def test_different_stage_is_a_distinct_preregistered_consumption(tmp_path: Path) -> None:
    registry = tmp_path / "consumption.jsonl"
    identity = {"freeze_sha256": "f" * 64, "candidate_sha256": "c" * 64}
    first = reserve_once(registry, stage="code_verification", identity=identity)
    complete_reservation(registry, first, artifact={"sha256": "1" * 64})
    second = reserve_once(registry, stage="final_qualification", identity=identity)
    complete_reservation(registry, second, artifact={"sha256": "2" * 64})
    assert len(read_registry(registry)) == 4


def test_failed_reservation_remains_consumed(tmp_path: Path) -> None:
    registry = tmp_path / "consumption.jsonl"
    identity = {"surface": "frozen"}
    reservation = reserve_once(registry, stage="blind_scoring", identity=identity)
    fail_reservation(registry, reservation, fault="model_fault")
    with pytest.raises(ConsumptionError, match="already consumed or reserved"):
        reserve_once(registry, stage="blind_scoring", identity=identity)


def test_malformed_history_fails_closed(tmp_path: Path) -> None:
    registry = tmp_path / "consumption.jsonl"
    registry.write_text('{"policy":"wrong"}\n', encoding="utf-8")
    with pytest.raises(ConsumptionError, match="invalid consumption registry"):
        reserve_once(registry, stage="candidate_generation", identity={"x": 1})


def _race_reservation(registry: str, start: multiprocessing.Event, queue: multiprocessing.Queue) -> None:
    start.wait()
    try:
        reserve_once(Path(registry), stage="candidate_generation", identity={"same": True})
        queue.put("reserved")
    except ConsumptionError:
        queue.put("refused")


def test_concurrent_duplicate_reservation_has_one_winner(tmp_path: Path) -> None:
    registry = tmp_path / "consumption.jsonl"
    start = multiprocessing.Event()
    queue: multiprocessing.Queue = multiprocessing.Queue()
    workers = [
        multiprocessing.Process(target=_race_reservation, args=(str(registry), start, queue))
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    start.set()
    for worker in workers:
        worker.join(timeout=10)
        assert worker.exitcode == 0
    assert sorted(queue.get(timeout=2) for _ in workers) == ["refused", "reserved"]
