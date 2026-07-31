from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("p2_canary", ROOT / "scripts" / "theseus_assistant_p2_canary.py")
assert SPEC and SPEC.loader
p2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p2)


def test_manifest_audit_binds_frozen_parent_model_and_visible_source() -> None:
    report = p2.audit_manifest(ROOT / "configs" / "theseus_assistant_p2_canary.json")
    assert report["trigger_state"] == "GREEN", report["faults"]
    assert report["context_receipt"]["visible_source_characters"] > 1000
    assert report["model_identity"]["identity_sha256"] == "580ab3bd8df4c2af0790ed015342fe7dbaa4ecad7e2fdc69b9fc4ef170f88a85"


def test_candidate_parser_accepts_only_authorized_sealed_shape() -> None:
    payload = {
        "patch_unified_diff": "--- a/scripts/theseus_assistant_route_integrity.py\n+++ b/scripts/theseus_assistant_route_integrity.py\n@@ -1 +1 @@\n-old\n+new\n",
        "proposed_paths": ["scripts/theseus_assistant_route_integrity.py"],
        "verification_commands": ["python3 -m pytest -q"],
        "abstained": False,
    }
    parsed, faults = p2.parse_candidate_output(
        json.dumps(payload),
        ["scripts/theseus_assistant_route_integrity.py"],
        {"required_fields": list(payload), "maximum_patch_bytes": 65536, "maximum_proposed_paths": 1},
    )
    assert not faults
    assert parsed == payload
    payload["proposed_paths"] = ["roadmap.md"]
    parsed, faults = p2.parse_candidate_output(
        json.dumps(payload),
        ["scripts/theseus_assistant_route_integrity.py"],
        {"required_fields": list(payload), "maximum_patch_bytes": 65536, "maximum_proposed_paths": 1},
    )
    assert not parsed
    assert "candidate_path_outside_authority" in faults


def test_budget_receipt_fails_closed() -> None:
    attempts = [{"arm_id": "direct_local_model", "resource_metrics": {"model_calls": 7, "arm_wall_ms": 1}}]
    receipt = p2.product_budget_receipt(
        attempts,
        {"maximum_model_calls_per_arm": 6, "maximum_arm_wall_ms": 720000, "maximum_pair_wall_ms": 1500000},
        p2.time.perf_counter(),
    )
    assert receipt["ready"] is False


def test_repair_amendment_is_bound_to_the_retained_negative() -> None:
    report = p2.audit_repair_amendment(
        ROOT / "configs" / "theseus_assistant_p2_canary.json",
        ROOT / "configs" / "theseus_assistant_p2_canary_repair_r1.json",
    )
    assert report["trigger_state"] == "GREEN", report["faults"]


def test_raw_diff_repair_parser_uses_a_deterministic_envelope() -> None:
    patch = "--- a/scripts/theseus_assistant_route_integrity.py\n+++ b/scripts/theseus_assistant_route_integrity.py\n@@ -1 +1 @@\n-old\n+new\n"
    parsed, faults = p2.parse_candidate_output(
        patch,
        ["scripts/theseus_assistant_route_integrity.py"],
        {"maximum_patch_bytes": 65536, "maximum_proposed_paths": 1},
        output_encoding="raw_git_unified_diff",
    )
    assert not faults
    assert parsed["patch_unified_diff"] == patch
    assert parsed["proposed_paths"] == ["scripts/theseus_assistant_route_integrity.py"]
