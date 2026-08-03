from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_semantic_ir_production_adequacy_replacement_02_evaluator as evaluator  # noqa: E402


def sources(role: str) -> dict[str, str]:
    report = p2a.read_json(evaluator.SOURCE_REPORT)
    receipt = p2a.mapping(p2a.mapping(report["archives"])[role])
    return evaluator.archive_sources(receipt)


def test_frozen_parent_fails_and_target_passes() -> None:
    assert evaluator.evaluate(sources("parent"), (evaluator.EXPECTED_PATH,)) is False
    assert evaluator.evaluate(sources("target"), (evaluator.EXPECTED_PATH,)) is True


def test_literal_and_wrong_dataflow_controls_fail() -> None:
    target = sources("target")
    literal = dict(target)
    literal[evaluator.EXPECTED_PATH] = evaluator.replace_assert_expected_with_literal(
        target[evaluator.EXPECTED_PATH]
    )
    assert evaluator.evaluate(literal, (evaluator.EXPECTED_PATH,)) is False
    wrong_source = dict(target)
    wrong_source[evaluator.EXPECTED_PATH] = target[evaluator.EXPECTED_PATH].replace(
        "expected_discount = self.fixture_ctx.get_expected_fixed_discount_reduction()",
        "expected_discount = 500",
        1,
    )
    assert evaluator.evaluate(wrong_source, (evaluator.EXPECTED_PATH,)) is False


def test_path_integrity_fails_closed() -> None:
    target = sources("target")
    assert evaluator.evaluate({}, (evaluator.EXPECTED_PATH,)) is False
    target["other.py"] = "value = 1\n"
    assert evaluator.evaluate(target, (evaluator.EXPECTED_PATH,)) is False


def test_full_qualification_is_green_before_candidate_packet() -> None:
    report = evaluator.qualify()
    assert report["trigger_state"] == "GREEN"
    assert report["candidate_packet_materialized"] is False
    assert all(report["checks"].values())
    assert report["counters"]["local_model_calls"] == 0
