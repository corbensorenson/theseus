from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_d1_cognitive_compilation as d1  # noqa: E402


INSTRUMENT = ROOT / "configs" / "theseus_p4v2r2r3_prompt_continuity_repair.json"


def write_archive(path: Path, root: str, files: dict[str, str]) -> None:
    with tarfile.open(path, "w:gz") as handle:
        for name, value in files.items():
            payload = value.encode()
            member = tarfile.TarInfo(f"{root}/{name}")
            member.size = len(payload)
            handle.addfile(member, io.BytesIO(payload))


def test_D1_adapter_audits_complete_artifact_uncapped_P4_owner() -> None:
    report = d1.audit_instrument(INSTRUMENT)

    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["D1_runtime_attempt_namespace"] == (
        "d1_fresh_qualification_attempt1"
    )
    assert report["causal_mechanism_changed"] is False
    instrument = json.loads(INSTRUMENT.read_text(encoding="utf-8"))
    continuity = instrument["prompt_continuity_repair"]
    assert continuity["complete_visible_verifier_feedback_visible_to_second_call"] is True
    assert continuity["project_selected_verifier_feedback_character_cap"] is None


def test_visible_verifier_overlays_evaluator_only_target_tests(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "pkg.py").write_text("VALUE = 1\n", encoding="utf-8")
    archive = tmp_path / "target.tar.gz"
    write_archive(
        archive,
        "target-root",
        {"tests/test_pkg.py": "def test_value():\n    assert True\n"},
    )
    evaluator = {
        "target_archive": str(archive),
        "target_archive_sha256": p2a.sha256_file(archive),
        "target_archive_root": "target-root",
        "test_overlay_paths": ["tests/test_pkg.py"],
    }
    observed: dict[str, object] = {}

    def sandboxed(root: Path, nodeids: list[str], _: dict) -> dict:
        observed["test"] = (root / "tests/test_pkg.py").read_text(encoding="utf-8")
        observed["nodeids"] = nodeids
        return {"passed": True, "boundary_hit": False}

    monkeypatch.setattr(d1.seal, "run_pytest_sandboxed", sandboxed)
    receipt = d1.run_verifier_on_candidate(
        candidate,
        ["tests/test_pkg.py::test_value"],
        evaluator,
        {},
    )

    assert receipt["passed"] is True
    assert observed["test"] == "def test_value():\n    assert True\n"
    assert observed["nodeids"] == ["tests/test_pkg.py::test_value"]
