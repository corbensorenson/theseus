from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import report_evidence_integrity as integrity  # noqa: E402


def build_pack(tmp_path: Path, monkeypatch) -> dict:
    monkeypatch.setattr(integrity, "ROOT", tmp_path)
    gate = tmp_path / "reports" / "gate.json"
    gate.parent.mkdir(parents=True, exist_ok=True)
    gate.write_text(
        json.dumps(
            {
                "policy": "fixture_gate_v1",
                "trigger_state": "GREEN",
                "summary": {"pass_count": 1},
                "non_claims": ["fixture scope only"],
                "public_training_rows_written": 0,
                "external_inference_calls": 0,
                "fallback_return_count": 0,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return integrity.build_standard_evidence_packs(
        [gate],
        commands={"reports/gate.json": ["python3 fixture_gate.py --gate"]},
    )[0]


def test_standard_pack_has_valid_payload_hash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pack = build_pack(tmp_path, monkeypatch)
    assert isinstance(pack.get("pack_payload_sha256"), str) and len(
        pack["pack_payload_sha256"]
    ) == 64, "request_contract:evidence_pack_payload_hash_present"
    assert integrity.validate_evidence_pack(pack)


def test_command_projection_tampering_breaks_payload_hash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pack = build_pack(tmp_path, monkeypatch)
    pack["commands"] = ["python3 attacker.py"]
    assert not integrity.validate_evidence_pack(
        pack
    ), "request_contract:evidence_pack_command_tamper_rejected"


def test_receipt_projection_tampering_breaks_payload_hash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pack = build_pack(tmp_path, monkeypatch)
    pack["baseline_receipt"] = {"fabricated": True}
    assert not integrity.validate_evidence_pack(
        pack
    ), "request_contract:evidence_pack_receipt_tamper_rejected"
