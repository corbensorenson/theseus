from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from neural_seed_functional_cases import materialize_cases
from neural_seed_functional_verifiers import score_english_judgments, verify_candidate


CONFIG = json.loads((ROOT / "configs/neural_seed_functional_utility.json").read_text())
CASES = materialize_cases(CONFIG)


def case(arm: str, family: str, variant: int = 0) -> dict:
    return next(row for row in CASES if row["arm_id"] == arm and row["task_family"] == family and row["variant"] == variant)


def test_python_known_good_bad_and_side_effect_candidates() -> None:
    row = case("python", "stable_unique")
    name = row["verifier"]["function_name"]
    good = f"def {name}(values):\n    return list(dict.fromkeys(values))\n"
    bad = f"def {name}(values):\n    return sorted(set(values))\n"
    side_effect = f"import socket\ndef {name}(values):\n    return list(dict.fromkeys(values))\n"

    assert verify_candidate(row, good, CONFIG)["passed"] is True
    assert verify_candidate(row, bad, CONFIG)["passed"] is False
    rejected = verify_candidate(row, side_effect, CONFIG)
    assert rejected["passed"] is False
    assert rejected["fault"] == "prohibited_side_effect"


def test_deno_known_good_and_bad_candidates() -> None:
    row = case("javascript_typescript", "stable_unique")
    name = row["verifier"]["function_name"]
    good = f"export function {name}(values) {{ return [...new Set(values)]; }}\n"
    bad = f"export function {name}(values) {{ return [...new Set(values)].sort(); }}\n"

    assert verify_candidate(row, good, CONFIG)["passed"] is True
    assert verify_candidate(row, bad, CONFIG)["passed"] is False


def test_rust_known_good_and_bad_candidates() -> None:
    row = case("rust", "stable_unique")
    name = row["verifier"]["function_name"]
    good = f"use std::collections::HashSet;\npub fn {name}(values: &[i32]) -> Vec<i32> {{ let mut s=HashSet::new(); values.iter().copied().filter(|v| s.insert(*v)).collect() }}\n"
    bad = f"pub fn {name}(values: &[i32]) -> Vec<i32> {{ let mut x=values.to_vec(); x.sort(); x.dedup(); x }}\n"

    assert verify_candidate(row, good, CONFIG)["passed"] is True
    assert verify_candidate(row, bad, CONFIG)["passed"] is False


def test_html_requires_dom_contract_and_real_render() -> None:
    row = case("html_css", "status_alert")
    good = '<!doctype html><html><head><style>section{border:1px solid red}button:focus-visible{outline:2px solid blue}</style></head><body><section role="alert"><h1>Sync 1 failed</h1><button type="button">Retry</button></section></body></html>'
    external = good.replace("</body>", '<script src="https://example.com/x.js"></script></body>')

    passed = verify_candidate(row, good, CONFIG)
    assert passed["passed"] is True
    assert passed["render"]["screenshot_bytes"] >= 512
    rejected = verify_candidate(row, external, CONFIG)
    assert rejected["passed"] is False
    assert "javascript_forbidden" in rejected["failures"]


def test_english_requires_blind_independent_raters_and_adjudication() -> None:
    english_cases = [row for row in CASES if row["arm_id"] == "english"]
    outputs = {row["case_id"]: "A concise grounded answer." for row in english_cases}
    dimensions = CONFIG["english_scoring"]["dimensions"]
    one_rater = [
        {"case_id": row["case_id"], "rater_id": "r1", "scores": {dimension: 3 for dimension in dimensions}}
        for row in english_cases
    ]
    result = score_english_judgments(CASES, outputs, one_rater, CONFIG)
    assert result["valid"] is False
    assert any(fault.startswith("insufficient_raters") for fault in result["faults"])

    exposed = one_rater + [
        {"case_id": row["case_id"], "rater_id": "r2", "model_id": "revealed", "scores": {dimension: 3 for dimension in dimensions}}
        for row in english_cases
    ]
    result = score_english_judgments(CASES, outputs, exposed, CONFIG)
    assert result["valid"] is False
    assert any(fault.startswith("identity_or_reference_exposed") for fault in result["faults"])
