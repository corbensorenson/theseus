from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = json.loads(
    (ROOT / "configs" / "theseus_p4v2r2_task_sources.json").read_text(
        encoding="utf-8"
    )
)
FIXTURES = ROOT / "tests" / "fixtures" / "theseus_p4v2r2_online"
VISIBLE = FIXTURES / "theseus_p4v2r2_visible_test.py"
HIDDEN = FIXTURES / "theseus_p4v2r2_hidden_test.py"
PYTHON = Path("/usr/local/bin/python3")


def extract(archive: Path, destination: Path, expected_root: str) -> Path:
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(destination, filter="data")
    root = destination / expected_root
    assert root.is_dir()
    return root


def evaluate(script: Path, case: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(PYTHON), str(script), case],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize("task", REGISTRY["tasks"], ids=lambda row: row["case"])
def test_exact_targets_pass_and_parents_fail_independent_evaluators(
    task: dict, tmp_path: Path
) -> None:
    stem = task["stem"]
    parent = extract(
        FIXTURES / f"{stem}_parent.tar.gz", tmp_path / "parent", task["source_root"]
    )
    target = extract(
        FIXTURES / f"{stem}_target.tar.gz", tmp_path / "target", task["target_root"]
    )

    for script in (VISIBLE, HIDDEN):
        parent_result = evaluate(script, task["case"], parent)
        target_result = evaluate(script, task["case"], target)
        assert parent_result.returncode != 0, (
            task["case"],
            script.name,
            parent_result.stdout,
            parent_result.stderr,
        )
        assert target_result.returncode == 0, (
            task["case"],
            script.name,
            target_result.stdout,
            target_result.stderr,
        )


def test_visible_markers_match_prospectively_bound_registry() -> None:
    text = VISIBLE.read_text(encoding="utf-8")
    expected = {
        marker["marker"]
        for task in REGISTRY["tasks"]
        for marker in task["visible_markers"]
    }
    assert all(marker in text for marker in expected)
    assert "target_revision" not in text
    assert "merge_revision" not in text


def test_evaluators_have_no_network_or_project_runtime_dependency() -> None:
    for path in (VISIBLE, HIDDEN):
        text = path.read_text(encoding="utf-8")
        assert "urllib" not in text
        assert "requests" not in text
        assert "theseus_" not in text.lower().replace("p4v2r2", "")
        compile(text, str(path), "exec")
    assert PYTHON.is_file()
