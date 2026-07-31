from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_route_integrity as integrity


def receipt(mode: str) -> dict:
    row = {
        "policy": integrity.ROUTE_POLICY,
        "execution_mode": mode,
        "ready": True,
        "failed_checks": [],
        "request_binding": {"user_prompt_sha256": "a" * 64},
        "pair_contract": {
            "model_identity_sha256": "b" * 64,
            "structural_verifier_id": integrity.STRUCTURAL_VERIFIER,
            "effect_sandbox_id": integrity.EFFECT_SANDBOX,
            "maximum_model_calls": 1,
            "automatic_effects": 0,
        },
    }
    row["receipt_sha256"] = integrity.receipt_digest(row)
    return row


def bundle() -> dict:
    direct = receipt(integrity.DIRECT_MODE)
    integrated = receipt(integrity.INTEGRATED_MODE)
    pair = integrity.compare_matched_pair(direct, integrated)
    return {
        "ready": True,
        "receipts": {
            integrity.DIRECT_MODE: direct,
            integrity.INTEGRATED_MODE: integrated,
        },
        "matched_pair": pair,
        "external_inference_calls": 0,
        "teacher_calls": 0,
        "public_calibration_cases_consumed": 0,
        "D2_cases_consumed": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
        "user_facing_effects": 0,
    }


def test_bundle_is_recomputed_instead_of_trusted() -> None:
    audit = getattr(integrity, "audit_report_payload", None)
    assert callable(audit), "request_contract:bundle_recomputed_not_trusted"
    report = bundle()
    result = audit(report)
    assert result["ready"] is True, "request_contract:bundle_recomputed_not_trusted"
    assert result["recomputed_matched_pair"] == report["matched_pair"], "request_contract:bundle_recomputed_not_trusted"
    corrupted = copy.deepcopy(report)
    corrupted["matched_pair"]["ready"] = False
    assert audit(corrupted)["ready"] is False, "request_contract:bundle_recomputed_not_trusted"
    nonzero = copy.deepcopy(report)
    nonzero["teacher_calls"] = 1
    assert audit(nonzero)["ready"] is False, "request_contract:bundle_recomputed_not_trusted"


def test_single_route_behavior_is_preserved() -> None:
    audit = getattr(integrity, "audit_report_payload", None)
    assert callable(audit), "request_contract:single_route_preserved"
    result = audit({"route_integrity": receipt(integrity.DIRECT_MODE)})
    assert result["policy"] == "project_theseus_route_integrity_receipt_audit_v1", "request_contract:single_route_preserved"
    assert result["ready"] is True, "request_contract:single_route_preserved"


def test_cli_report_path_uses_payload_audit(tmp_path: Path) -> None:
    report_path = tmp_path / "bundle.json"
    report_path.write_text(json.dumps(bundle()), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "scripts/theseus_assistant_route_integrity.py", "--report", str(report_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, "request_contract:cli_uses_payload_audit"
    output = json.loads(result.stdout)
    assert output["policy"] == "project_theseus_route_integrity_bundle_audit_v1", "request_contract:cli_uses_payload_audit"
    assert output["ready"] is True, "request_contract:cli_uses_payload_audit"
