from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = json.loads(
    (ROOT / "configs" / "theseus_p4v2r2r2_task_sources.json").read_text(encoding="utf-8")
)
REVISION_CORRECTIONS = json.loads(
    (ROOT / "configs" / "theseus_p4v2r2r2_revision_corrections.json").read_text(
        encoding="utf-8"
    )
)
CORRECTIONS_BY_STEM = {row["stem"]: row for row in REVISION_CORRECTIONS["corrections"]}
FIXTURES = ROOT / "tests" / "fixtures" / "theseus_p4v2r2r2_online"
CORE = FIXTURES / "theseus_p4v2r2r2_evaluator_core.py"
VISIBLE = FIXTURES / "theseus_p4v2r2r2_visible_test.py"
HIDDEN = FIXTURES / "theseus_p4v2r2r2_hidden_test.py"


def extract(archive: Path, destination: Path, expected_root: str) -> Path:
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(destination, filter="data")
    root = destination / expected_root
    assert root.is_dir()
    return root


def run(script: Path, case: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), case],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize("task", REGISTRY["tasks"], ids=lambda row: row["case"])
def test_exact_targets_pass_and_parents_fail_both_evaluator_layers(
    task: dict, tmp_path: Path
) -> None:
    stem = task["stem"]
    correction = CORRECTIONS_BY_STEM.get(stem)
    suffix = "_revision_corrected" if correction else ""
    parent_root = correction["corrected_parent_root"] if correction else task["source_root"]
    target_root = correction["corrected_target_root"] if correction else task["target_root"]
    parent = extract(
        FIXTURES / f"{stem}_parent{suffix}.tar.gz", tmp_path / "parent", parent_root
    )
    target = extract(
        FIXTURES / f"{stem}_target{suffix}.tar.gz", tmp_path / "target", target_root
    )
    for script in (VISIBLE, HIDDEN):
        parent_result = run(script, task["case"], parent)
        target_result = run(script, task["case"], target)
        assert parent_result.returncode != 0, (task["case"], script.name, parent_result.stdout, parent_result.stderr)
        assert target_result.returncode == 0, (task["case"], script.name, target_result.stdout, target_result.stderr)


def test_evaluator_surface_is_dependency_free_and_bound_to_no_target_identity() -> None:
    for path in (CORE, VISIBLE, HIDDEN):
        text = path.read_text(encoding="utf-8")
        compile(text, str(path), "exec")
        assert "target_revision" not in text
        assert "merge_revision" not in text
        assert "requests" not in text
        assert "urllib" not in text
