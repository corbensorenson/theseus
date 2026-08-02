from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4v2r2r2_cognitive_compilation as successor  # noqa: E402


PREDECESSOR_INSTRUMENT = (
    ROOT / "configs" / "theseus_p4v2r2r1_cognitive_compilation_instrument.json"
)


def successor_instrument(tmp_path: Path, *, namespace: str) -> Path:
    value = p2a.read_json(PREDECESSOR_INSTRUMENT)
    value["runtime_attempt_namespace"] = namespace
    value["state"] = successor.INSTRUMENT_STATE
    value["harness"]["candidate_runner"] = p2a.rel(Path(successor.__file__).resolve())
    value["harness"]["candidate_runner_sha256"] = p2a.sha256_file(
        Path(successor.__file__).resolve()
    )
    path = tmp_path / "instrument.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_successor_namespace_is_green_without_changing_causal_runner(tmp_path: Path) -> None:
    path = successor_instrument(
        tmp_path, namespace=successor.RUNTIME_ATTEMPT_NAMESPACE
    )
    report = successor.audit_instrument(path)

    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["instrument_sha256"] == p2a.sha256_file(path)
    assert report["namespace_adapter"]["causal_runner_changed"] is False
    assert report["namespace_adapter"]["receipt_namespace_only"] is True


def test_successor_rejects_every_other_namespace(tmp_path: Path) -> None:
    path = successor_instrument(tmp_path, namespace="p4v2r2r3_attempt1")
    report = successor.audit_instrument(path)

    assert report["trigger_state"] == "RED"
    assert report["faults"] == ["runtime_attempt_namespace_invalid"]


def test_execution_delegates_with_temporary_successor_audit(monkeypatch) -> None:
    observed = {}
    original = successor.predecessor.audit_instrument

    def delegated(instrument_path, task_path, *, session_factory):
        observed["audit"] = successor.predecessor.audit_instrument
        observed["instrument"] = instrument_path
        observed["task"] = task_path
        observed["session_factory"] = session_factory
        return {"trigger_state": "GREEN"}

    monkeypatch.setattr(successor.predecessor, "run_experiment", delegated)
    sentinel = object()
    result = successor.run_experiment(
        Path("instrument.json"), Path("task.json"), session_factory=sentinel
    )

    assert result == {"trigger_state": "GREEN"}
    assert observed["audit"] is successor.audit_instrument
    assert observed["session_factory"] is sentinel
    assert successor.predecessor.audit_instrument is original
