from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import kerc_source_span_transport_audit as audit  # noqa: E402


def test_source_span_population_audit_is_zero_credit_and_exact_on_one_admitted_row() -> None:
    report = audit.audit_artifact(
        audit.DEFAULT_ARTIFACT,
        maximum_compiler_rows=1,
    )

    assert report["full_population"] is False
    assert report["qualification"] == "NOT_QUALIFIED"
    assert report["counts"]["compiler_rows"] == 1
    assert report["counts"]["target_variants"] == 2
    assert report["counts"]["roundtrip_failures"] == 0
    assert report["counts"]["exact_v3_semantic_equivalence_targets"] == 2
    assert report["candidate_generation_credit"] == 0
    assert report["deterministic_materialization_generation_credit"] == 0
    assert report["deterministic_materialization_capability_credit"] == 0
    assert report["public_evaluation_rows_consumed"] == 0
    assert report["private_evaluation_rows_consumed"] == 0
