from __future__ import annotations

import json
import sys
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_assistant_p2a_evaluator as evaluator  # noqa: E402


def write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    (source / "example.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as handle:
        handle.add(source / "example.py", arcname="example.py")
    target_source = tmp_path / "target_source"
    target_source.mkdir()
    (target_source / "example.py").write_text("def value():\n    return 2\n", encoding="utf-8")
    target_archive = tmp_path / "target.tar"
    with tarfile.open(target_archive, "w") as handle:
        handle.add(target_source / "example.py", arcname="example.py")
    task = {
        "policy": "project_theseus_p2a_licensed_task_v1",
        "state": "SEALED_BEFORE_CANDIDATE_GENERATION",
        "opaque_task_id": "blind-fixture",
        "natural_request": "Make value() return 2.",
        "source_archive": str(archive),
        "source_archive_sha256": p2a.sha256_file(archive),
        "source_provenance": {
            "url": "https://example.invalid/repository",
            "revision": "abc123",
            "license_spdx": "MIT",
        },
        "allowed_effect_paths": ["example.py"],
        "candidate_visible_context": {
            "reads": [{"path": "example.py", "start_line": 1, "end_line": 2}],
            "searches": [],
        },
        "visible_verifier": {"command": ["python3", "-m", "compileall", "-q", "."]},
    }
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    hidden = tmp_path / "hidden_test.py"
    hidden.write_text("from example import value\nassert value() == 2\n", encoding="utf-8")
    evaluator_payload = {
        "policy": "project_theseus_p2a_route_blind_evaluator_v1",
        "state": "SEALED_BEFORE_CANDIDATE_GENERATION",
        "task_manifest": str(task_path),
        "task_manifest_sha256": p2a.sha256_file(task_path),
        "baseline_must_fail": True,
        "baseline_failure_markers": ["AssertionError"],
        "hidden_test_files": [
            {"source": str(hidden), "sha256": p2a.sha256_file(hidden), "destination": "hidden_test.py"}
        ],
        "target_archive": str(target_archive),
        "target_archive_sha256": p2a.sha256_file(target_archive),
        "target_must_pass": True,
        "hidden_verifier": {"command": ["python3", "hidden_test.py"], "timeout_seconds": 10},
    }
    evaluator_path = tmp_path / "evaluator.json"
    evaluator_path.write_text(json.dumps(evaluator_payload), encoding="utf-8")
    return task_path, evaluator_path, archive


def test_evaluator_audit_recomputes_failing_baseline(tmp_path: Path) -> None:
    _, evaluator_path, _ = write_fixture(tmp_path)

    report = evaluator.audit_evaluator(evaluator_path)

    assert report["trigger_state"] == "GREEN"
    assert report["baseline_verification"]["passed"] is False
    assert report["target_verification"]["passed"] is True
    assert report["route_labels_opened"] == 0


def test_sealed_candidate_is_scored_blind_and_reaches_correctness(tmp_path: Path) -> None:
    task_path, evaluator_path, archive = write_fixture(tmp_path)
    action = {
        "op": "REPLACE", "path": "example.py", "start_line": 2, "end_line": 2,
        "replacement": "    return 2",
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "candidate"
        p2a.extract_source_archive(archive, root)
        assert p2a.apply_actions(root, [action]) == []
        candidate_payload = {
            "protocol": "theseus_line_edit_v1",
            "actions": [action],
            "changed_paths": ["example.py"],
            "final_inventory_sha256": p2a.stable_hash(p2a.inventory(root)),
            "visible_verifier": {"passed": True},
        }
    seal = {
        "candidate_output_sha256": p2a.stable_hash(candidate_payload),
        "sealed_before_hidden_evaluation": True,
    }
    candidate_report = {
        "task_sha256": p2a.sha256_file(task_path),
        "matched_pair": {"ready": True},
        "attempts": [
            {"arm_id": "integrated_local_model", "candidate": candidate_payload, "candidate_seal": seal}
        ],
    }
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(candidate_report), encoding="utf-8")

    report = evaluator.evaluate_report(candidate_path, evaluator_path)

    assert report["trigger_state"] == "GREEN"
    assert report["disposition"] == "P2A_INSTRUMENT_ADEQUATE_P3_ELIGIBLE"
    assert report["denominators"]["correctness_evaluated_candidates"] == 1
    assert report["denominators"]["useful_candidates"] == 1
    assert report["evaluation_blinding"]["route_labels_passed_to_scoring"] is False
