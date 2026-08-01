from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_p4s_list_normalization_mechanics as mechanics  # noqa: E402


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_retained_p4s_diagnostic_is_mechanics_only_and_green() -> None:
    report = mechanics.build_report()

    assert report["trigger_state"] == "GREEN"
    assert report["retained_attempt2_treatment_call_count"] == 20
    assert all(row["state"] == "GREEN" for row in report["rows"])
    assert all(row["list_fields_canonical_after_normalization"] for row in report["rows"])
    assert report["unrelated_non_list_syntax_residual_count"] == 2
    assert report["custody"]["task_evaluator_invocations"] == 0
    assert report["custody"]["p4s_scores_recomputed"] is False
    assert report["custody"]["p4s_disposition_modified"] is False
    assert report["custody"]["consumed_runtime_receipts_modified"] == 0


def test_non_list_projection_does_not_mask_replacement_content() -> None:
    original = (
        "THESEUS_SEMANTIC_IR_V2\nALL_OBLIGATIONS [O1]\n<<<\n"
        "OBLIGATIONS [O9]\n>>>\nEND"
    )
    changed = original.replace("O9", "O8")

    assert mechanics.non_list_projection(original) != mechanics.non_list_projection(
        changed
    )


def test_v2r2_mechanics_binding_matches_exact_sources_and_retained_evidence() -> None:
    config = json.loads(
        (ROOT / "configs" / "theseus_semantic_ir_v2r2_mechanics.json").read_text(
            encoding="utf-8"
        )
    )

    assert config["state"] == "PROSPECTIVE_LIST_NORMALIZATION_MECHANICS_GREEN"
    for path_field, digest_field in (
        ("parser", "parser_sha256"),
        ("mechanics_report", "mechanics_report_sha256"),
        ("retained_p4s_terminal_disposition", "retained_p4s_terminal_disposition_sha256"),
    ):
        assert file_sha256(ROOT / config[path_field]) == config[digest_field]
    for path_field, digest_field in (
        ("parser_adversarial_test", "parser_adversarial_test_sha256"),
        ("retained_mechanics_test", "retained_mechanics_test_sha256"),
    ):
        assert file_sha256(ROOT / config["tests"][path_field]) == config["tests"][digest_field]
