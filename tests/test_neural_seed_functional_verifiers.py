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


def test_python_candidate_cannot_inspect_hidden_assertion_source() -> None:
    row = case("python", "stable_unique")
    name = row["verifier"]["function_name"]
    introspective = f"""import inspect
def {name}(values):
    needle=''.join(chr(x) for x in [97,115,115,101,114,116,32,109,46])
    contexts=' '.join(''.join(frame.code_context or []) for frame in inspect.stack())
    if needle in contexts:
        return list(dict.fromkeys(values))
    return []
"""

    result = verify_candidate(row, introspective, CONFIG)
    assert result["passed"] is False
    assert result["hidden_expected_visible_to_candidate"] is False


def test_deno_known_good_and_bad_candidates() -> None:
    row = case("javascript_typescript", "stable_unique")
    name = row["verifier"]["function_name"]
    good = f"export function {name}(values) {{ return [...new Set(values)]; }}\n"
    bad = f"export function {name}(values) {{ return [...new Set(values)].sort(); }}\n"

    assert verify_candidate(row, good, CONFIG)["passed"] is True
    assert verify_candidate(row, bad, CONFIG)["passed"] is False


def test_deno_candidate_cannot_replace_host_comparison_or_protocol() -> None:
    row = case("javascript_typescript", "stable_unique")
    name = row["verifier"]["function_name"]
    adversarial = f"""console.log = () => {{}};
export function {name}(_values: number[]) {{
  (JSON as any).stringify = () => 'same';
  return [];
}}
"""

    result = verify_candidate(row, adversarial, CONFIG)
    assert result["passed"] is False
    assert result["hidden_expected_visible_to_candidate"] is False


def test_rust_known_good_and_bad_candidates() -> None:
    row = case("rust", "stable_unique")
    name = row["verifier"]["function_name"]
    good = f"use std::collections::HashSet;\npub fn {name}(values: &[i32]) -> Vec<i32> {{ let mut s=HashSet::new(); values.iter().copied().filter(|v| s.insert(*v)).collect() }}\n"
    bad = f"pub fn {name}(values: &[i32]) -> Vec<i32> {{ let mut x=values.to_vec(); x.sort(); x.dedup(); x }}\n"

    assert verify_candidate(row, good, CONFIG)["passed"] is True
    assert verify_candidate(row, bad, CONFIG)["passed"] is False

    include_probe = f'pub fn {name}(_values: &[i32]) -> Vec<i32> {{ let _ = include_str!("../tests/functional.rs"); vec![] }}\n'
    rejected = verify_candidate(row, include_probe, CONFIG)
    assert rejected["passed"] is False
    assert rejected["fault"] == "prohibited_side_effect"


def test_html_requires_dom_contract_and_real_render() -> None:
    row = case("html_css", "status_alert")
    good = '<!doctype html><html><head><title>Status</title><style>section{border:1px solid red}button:focus-visible{outline:2px solid blue}</style></head><body><section role="alert"><h1>Sync 1 failed</h1><button type="button">Retry</button></section></body></html>'
    external = good.replace("</body>", '<script src="https://example.com/x.js"></script></body>')

    passed = verify_candidate(row, good, CONFIG)
    assert passed["passed"] is True
    assert len(passed["renders"]) == 2
    assert all(render["screenshot_bytes"] >= 512 for render in passed["renders"])
    assert all(render["browser_assertions"]["visible_alert"] for render in passed["renders"])
    rejected = verify_candidate(row, external, CONFIG)
    assert rejected["passed"] is False
    assert "javascript_forbidden" in rejected["failures"]


def test_html_responsive_behavior_is_computed_at_both_viewports() -> None:
    row = case("html_css", "responsive_cards")
    body = '<main><h1>Projects 1</h1><section class="cards" aria-label="Projects"><article><h2>A</h2></article><article><h2>B</h2></article><article><h2>C</h2></article></section></main>'
    good_css = '.cards{display:grid;grid-template-columns:repeat(3,1fr)}@media (max-width:48rem){.cards{grid-template-columns:1fr}}'
    bad_css = '.cards{display:grid;grid-template-columns:repeat(3,1fr)}@media (max-width:48rem){.cards{grid-template-columns:repeat(3,1fr)}}'
    page = lambda css: f'<!doctype html><html><head><title>Projects</title><style>{css}</style></head><body>{body}</body></html>'

    passed = verify_candidate(row, page(good_css), CONFIG)
    assert passed["passed"] is True
    assert passed["renders"][0]["browser_audit"]["articleColumnCount"] == 3
    assert passed["renders"][1]["browser_audit"]["articleColumnCount"] == 1

    rejected = verify_candidate(row, page(bad_css), CONFIG)
    assert rejected["passed"] is False
    assert "browser_behavior_failure:narrow" in rejected["failures"]


def test_html_rejects_local_resource_reads() -> None:
    row = case("html_css", "status_alert")
    candidate = '<!doctype html><html><head><title>Status</title><style>@import "file:///etc/passwd";section{border:1px solid red}button:focus-visible{outline:2px solid blue}</style></head><body><section role="alert"><h1>Sync 1 failed</h1><button type="button">Retry</button></section></body></html>'

    rejected = verify_candidate(row, candidate, CONFIG)
    assert rejected["passed"] is False
    assert "local_resource_forbidden" in rejected["failures"]


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
