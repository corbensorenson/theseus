from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p2_evaluator", ROOT / "scripts" / "theseus_assistant_p2_evaluator.py")
assert SPEC and SPEC.loader
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def test_evaluator_alignment_proves_hidden_tests_fail_on_frozen_parent() -> None:
    report = evaluator.audit_evaluator(ROOT / "configs" / "theseus_assistant_p2_evaluator.json")
    assert report["trigger_state"] == "GREEN", report["faults"]
    assert report["baseline_failed_as_expected"] is True
    assert report["route_labels_opened"] == 0
    assert report["target_commits_opened"] == 0


def test_patch_path_extraction_is_fail_closed() -> None:
    patch = "--- a/scripts/a.py\n+++ b/scripts/a.py\n@@ -1 +1 @@\n-a\n+b\n--- a/tests/x.py\n+++ b/tests/x.py\n"
    assert evaluator.extract_patch_paths(patch) == ["scripts/a.py", "tests/x.py"]


def test_missing_candidate_is_valid_negative_not_unsafe() -> None:
    result = evaluator.failed_candidate("candidate_seal_invalid_or_missing", {})
    assert result["useful"] == 0
    assert result["unsafe"] == 0
    assert result["sealed"] == 0
