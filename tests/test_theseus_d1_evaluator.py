from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_d1_evaluator as evaluator  # noqa: E402
import theseus_d1_evaluator_seal as seal  # noqa: E402


def write_archive(path: Path, root: str, source: str) -> None:
    with tarfile.open(path, "w:gz") as handle:
        files = {
            "LICENSE": "MIT\n",
            "pkg/core.py": source,
            "tests/test_core.py": "def test_repair():\n    assert True\n",
        }
        for name, text in files.items():
            payload = text.encode()
            member = tarfile.TarInfo(f"{root}/{name}")
            member.size = len(payload)
            handle.addfile(member, io.BytesIO(payload))


def surface(tmp_path: Path) -> tuple[Path, Path, Path]:
    parent = tmp_path / "parent.tar.gz"
    target = tmp_path / "target.tar.gz"
    write_archive(parent, "repo-parent", "def repair(value):\n    return value\n")
    write_archive(target, "repo-target", "def repair(value):\n    return abs(value)\n")
    task = {
        "source_archive": str(parent),
        "source_archive_sha256": p2a.sha256_file(parent),
        "source_archive_root": "repo-parent",
        "allowed_effect_paths": ["pkg/core.py"],
        "candidate_visible_context": {"project_selected_character_or_token_cap": None},
        "visible_verifier": {"project_selected_quality_token_cap": None},
    }
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task) + "\n", encoding="utf-8")
    manifest = {
        "policy": evaluator.EVALUATOR_POLICY,
        "state": "SEALED_BEFORE_CANDIDATE_GENERATION",
        "task_manifest": str(task_path),
        "task_manifest_sha256": p2a.sha256_file(task_path),
        "target_archive": str(target),
        "target_archive_sha256": p2a.sha256_file(target),
        "target_archive_root": "repo-target",
        "test_overlay_paths": ["tests/test_core.py"],
        "hidden_pytest_nodeids": ["tests/test_core.py::test_repair"],
    }
    evaluator_path = tmp_path / "evaluator.json"
    evaluator_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return task_path, evaluator_path, parent


def test_blind_evaluator_recomputes_candidate_integrity_and_rolls_back(
    tmp_path: Path, monkeypatch
) -> None:
    task_path, evaluator_path, parent = surface(tmp_path)
    candidate = {
        "actions": [
            {
                "op": "REPLACE",
                "path": "pkg/core.py",
                "start_line": 1,
                "end_line": 2,
                "replacement": "def repair(value):\n    return abs(value)",
            }
        ]
    }
    digest = p2a.stable_hash(candidate)
    run = {
        "D1_task_sha256": p2a.sha256_file(task_path),
        "D1_evaluator_sha256": p2a.sha256_file(evaluator_path),
        "matched_set": {"ready": True},
        "attempts": [
            {
                "arm_id": "typed_semantic_ir_treatment",
                "candidate": candidate,
                "candidate_seal": {"candidate_output_sha256": digest},
            }
        ],
        "deterministic_compiler_control": {},
    }
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps(run) + "\n", encoding="utf-8")
    parent_sha = p2a.sha256_file(parent)

    monkeypatch.setattr(
        seal,
        "run_pytest_sandboxed",
        lambda root, nodeids, config: {
            "passed": True,
            "returncode": 0,
            "boundary_hit": False,
            "nodeids_sha256": p2a.stable_hash(nodeids),
        },
    )
    report = evaluator.evaluate_report(run_path, evaluator_path)
    assert report["trigger_state"] == "GREEN"
    assert report["results"][0]["useful"] == 1
    assert report["results"][0]["arm_id"] == "typed_semantic_ir_treatment"
    assert report["evaluation_blinding"]["arm_labels_passed_to_scoring"] is False
    assert p2a.sha256_file(parent) == parent_sha


def test_digest_collision_across_labels_fails_closed() -> None:
    rows: list[dict] = []
    labels: dict[str, str] = {}
    faults: list[str] = []
    candidate = {"actions": []}
    candidate_seal = {"candidate_output_sha256": "a" * 64}
    evaluator.add_blinded(rows, labels, candidate, candidate_seal, "arm-a", faults)
    evaluator.add_blinded(rows, labels, candidate, candidate_seal, "arm-b", faults)
    assert faults == ["candidate_digest_label_collision"]
