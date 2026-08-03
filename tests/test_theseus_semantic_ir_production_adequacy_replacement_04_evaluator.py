from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_semantic_ir_production_adequacy_replacement_04_evaluator as evaluator  # noqa: E402


def sources(role: str) -> dict[str, str]:
    report = p2a.read_json(evaluator.SOURCE_REPORT)
    receipt = p2a.mapping(p2a.mapping(report["archives"])[role])
    return evaluator.archive_sources(receipt)


def test_frozen_parent_fails_and_target_passes() -> None:
    assert evaluator.evaluate(sources("parent"), (evaluator.EXPECTED_PATH,)) is False
    assert evaluator.evaluate(sources("target"), (evaluator.EXPECTED_PATH,)) is True


def test_truthiness_cast_dataflow_default_and_order_controls_fail() -> None:
    target = sources("target")
    source = target[evaluator.EXPECTED_PATH]
    mutations = [
        source.replace("if atol is None:", "if not atol:", 1),
        source.replace("    atol = dtype(atol)\n", "", 1),
        source.replace("        atol = 0.0\n", "        atol = 0.5\n", 1),
        source.replace("    atol = dtype(atol)\n", "    atol = dtype(0.0)\n", 1),
    ]
    block = "    if atol is None:\n        atol = 0.0\n    atol = dtype(atol)\n\n"
    anchor = "    gap_open, gap_extend = prep_gapcost(gap_cost, dtype=dtype)\n"
    mutations.append(source.replace(block, "", 1).replace(anchor, anchor + block, 1))
    for mutation in mutations:
        assert evaluator.evaluate({evaluator.EXPECTED_PATH: mutation}, (evaluator.EXPECTED_PATH,)) is False


def test_benign_comment_and_path_integrity_controls() -> None:
    target = sources("target")
    benign = dict(target)
    benign[evaluator.EXPECTED_PATH] += "\n# benign\n"
    assert evaluator.evaluate(benign, (evaluator.EXPECTED_PATH,)) is True
    assert evaluator.evaluate({}, (evaluator.EXPECTED_PATH,)) is False
    target["other.py"] = "value = 1\n"
    assert evaluator.evaluate(target, (evaluator.EXPECTED_PATH,)) is False


def test_full_qualification_is_green_before_candidate_packet() -> None:
    report = evaluator.qualify()
    assert report["trigger_state"] == "GREEN"
    assert report["candidate_packet_materialized"] is False
    assert all(report["checks"].values())
    assert report["counters"]["local_model_calls"] == 0
    assert report["counters"]["parent_target_evaluator_executions"] == 10
