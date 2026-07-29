from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import core_evidence_worker_v2_evaluator as evaluator  # noqa: E402


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_patch_path_validation_rejects_escape() -> None:
    assert evaluator.validate_patch_paths(
        "--- a/scripts/x.py\n+++ b/scripts/x.py\n@@ -1 +1 @@\n-x=1\n+x=2\n"
    ) == ["scripts/x.py"]
    with pytest.raises(evaluator.EvaluationFault, match="escape"):
        evaluator.validate_patch_paths(
            "--- a/scripts/x.py\n+++ b/../escape.py\n"
        )


def test_seal_is_recomputed_from_candidate_bytes() -> None:
    output = {"worker_id": "local", "abstained": True}
    encoded = (
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    ).encode()
    seal = {
        "candidate_output_sha256": hashlib.sha256(encoded).hexdigest(),
        "worker_input_sha256": "a",
        "parent_archive_sha256": "b",
        "worker_source_sha256": "c",
        "config_sha256": "d",
        "target_opened_before_seal": False,
    }
    assert evaluator.validate_seal(output, seal)
    assert not evaluator.validate_seal(
        {**output, "abstained": False}, seal
    )


def test_failed_selected_verification_is_a_repair_wall() -> None:
    wall = evaluator.diagnose_wall(
        useful=False,
        patch="diff",
        patch_applies=True,
        recomputed_paths=["scripts/x.py"],
        overlap=["scripts/x.py"],
        candidate_row={
            "candidate_verification_green": False,
            "verification_receipts": [{
                "commands": ["python -m pytest tests/test_x.py"],
                "passed": False,
            }],
        },
        hidden_passed=False,
        rollback_verified=True,
        out_of_snapshot_effects=[],
    )
    assert wall == "EDIT_SYNTHESIS_OR_BOUNDED_REPAIR"


def test_independent_evaluator_applies_patch_runs_hidden_test_and_rolls_back(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test")
    (repo / "scripts").mkdir()
    (repo / "tests").mkdir()
    (repo / "scripts" / "calc.py").write_text(
        "def value():\n    return 1\n", encoding="utf-8"
    )
    (repo / "tests" / "test_calc.py").write_text(
        "from scripts.calc import value\n\n"
        "def test_value():\n    assert value() == 1\n",
        encoding="utf-8",
    )
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "parent")
    parent = git(repo, "rev-parse", "HEAD")
    (repo / "scripts" / "calc.py").write_text(
        "def value():\n    return 2\n", encoding="utf-8"
    )
    (repo / "tests" / "test_calc.py").write_text(
        "from scripts.calc import value\n\n"
        "def test_value():\n    assert value() == 2\n",
        encoding="utf-8",
    )
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "Raise value")
    target = git(repo, "rev-parse", "HEAD")
    patch = git(
        repo, "diff", parent, target, "--", "scripts/calc.py"
    ) + "\n"
    request = "Raise value"
    output = {
        "worker_id": "theseus_local_repository_worker_v2",
        "natural_request_sha256": hashlib.sha256(
            request.encode()
        ).hexdigest(),
        "parent_source_commit": parent,
        "patch_unified_diff": patch,
        "proposed_paths": ["scripts/calc.py"],
        "verification_commands": ["python -m py_compile scripts/calc.py"],
        "abstained": False,
    }
    encoded = (
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    ).encode()
    seal = {
        "candidate_output_sha256": hashlib.sha256(encoded).hexdigest(),
        "worker_input_sha256": "a",
        "parent_archive_sha256": "b",
        "worker_source_sha256": "c",
        "config_sha256": "d",
        "target_opened_before_seal": False,
    }
    task = {
        "source_task_id": "synthetic-1",
        "target_commit": target,
        "parent_source_commit": parent,
        "natural_request": request,
    }
    campaign = {
        "evaluator_contract": {
            "candidate_output_schema": {
                "required_fields": list(output),
                "maximum_patch_bytes": 100000,
                "maximum_proposed_paths": 8,
                "maximum_verification_commands": 8,
            },
            "completion_predicate": {
                "changed_path_precision_minimum": 0.5,
                "changed_path_recall_minimum": 0.25,
            },
        }
    }
    result = evaluator.evaluate_candidate(
        task,
        {
            "candidate_output": output,
            "candidate_seal": seal,
            "candidate_verification_green": True,
            "terminal_reason": "finished",
        },
        campaign,
        repo,
    )
    assert result["patch_applies_cleanly"], result
    assert result["hidden_tests_passed"]
    assert result["exact_rollback_verified"]
    assert result["out_of_snapshot_effects"] == []
    assert result["useful_completed_task"]
