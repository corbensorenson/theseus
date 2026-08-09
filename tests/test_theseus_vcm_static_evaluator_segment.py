from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_static_evaluator_segment as owner  # noqa: E402
import theseus_vcm_static_evaluator_segment_audit as audit_owner  # noqa: E402

CONFIG = ROOT / "configs" / "theseus_vcm_static_evaluator_segment.json"


def test_static_segment_preflight_binds_exact_eight_dependency_free_rows() -> None:
    cfg, bound, faults = owner.preflight(CONFIG)
    assert faults == []
    assert [row["index"] for row in cfg["rows"]] == [1, 10, 20, 22, 23, 27, 57, 58]
    assert set(bound["executables"]) == {"python", "node", "sandbox_exec"}
    assert all(row["runtime"] in {"python", "node"} for row in cfg["rows"])


def test_static_segment_has_no_dependency_or_model_authority() -> None:
    cfg = owner.p2a.read_json(CONFIG)
    assert "no dependency installation" in cfg["maximum_inference"]
    assert "local-model" in cfg["maximum_inference"]
    assert "Luna" in cfg["maximum_inference"]
    assert "partial panel" in cfg["maximum_inference"]
    audit = owner.p2a.resolve(cfg["audit_owner"])
    assert audit == Path(audit_owner.__file__).resolve()
    assert owner.p2a.sha256_file(audit) == cfg["audit_owner_sha256"]
