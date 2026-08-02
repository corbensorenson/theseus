from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4v2r2r3_cognitive_compilation as successor  # noqa: E402


PREDECESSOR_INSTRUMENT = (
    ROOT / "configs" / "theseus_p4v2r2r2_cognitive_compilation_instrument.json"
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


def test_successor_audit_is_green_with_frozen_non_adapter_owner(tmp_path: Path) -> None:
    path = successor_instrument(
        tmp_path, namespace=successor.RUNTIME_ATTEMPT_NAMESPACE
    )
    report = successor.audit_instrument(path)

    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["namespace_adapter"]["frozen_audit_function_reference"] is True
    assert report["namespace_adapter"]["causal_runner_changed"] is False


def test_successor_rejects_every_other_namespace(tmp_path: Path) -> None:
    path = successor_instrument(tmp_path, namespace="p4v2r2r4_attempt1")
    report = successor.audit_instrument(path)

    assert report["trigger_state"] == "RED"
    assert report["faults"] == ["runtime_attempt_namespace_invalid"]


def test_execution_path_calls_the_repaired_audit_without_recursion(
    monkeypatch, tmp_path: Path
) -> None:
    path = successor_instrument(
        tmp_path, namespace=successor.RUNTIME_ATTEMPT_NAMESPACE
    )
    original = successor.predecessor.audit_instrument

    def delegated(instrument_path, task_path, *, session_factory):
        audit = successor.predecessor.audit_instrument(instrument_path)
        return {"trigger_state": audit["trigger_state"], "faults": audit["faults"]}

    monkeypatch.setattr(successor.predecessor, "run_experiment", delegated)
    result = successor.run_experiment(path, Path("task.json"), session_factory=object())

    assert result == {"trigger_state": "GREEN", "faults": []}
    assert successor.predecessor.audit_instrument is original
