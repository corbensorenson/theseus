import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import teacher_distillation_gate as gate  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def policy() -> dict:
    return {
        "ledger_path": "reports/teacher_distillation_ledger.jsonl",
        "teacher_share": {
            "accounting_ledger_path": (
                "runtime/data_governance/teacher_share_accounting_ledger.jsonl"
            ),
            "max_initial_training_ratio": 0.1,
            "target_trend": "decrease",
        },
    }


def prepare_root(root: Path) -> None:
    teacher_ledger = root / "reports/teacher_distillation_ledger.jsonl"
    teacher_ledger.parent.mkdir(parents=True)
    teacher_ledger.write_text(
        json.dumps(
            {
                "accepted": True,
                "source_kind": "teacher_distillation",
                "created_utc": "2026-07-01T00:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(
        root / "reports/training_data_admission_v1.json",
        {
            "source_admissions": [
                {
                    "training_use": "allowed",
                    "allowed_for_training": True,
                    "row_count": 99,
                    "teacher_row_count": 0,
                }
            ],
            "candidate_lineage": {
                "candidate_receipt_ledger": {
                    "path": "runtime/data_governance/receipts.jsonl.gz",
                    "sha256": "a" * 64,
                    "receipt_count": 99,
                    "replay_valid": True,
                }
            },
        },
    )


def test_share_requires_content_bound_durable_snapshot(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    prepare_root(tmp_path)
    state = gate.load_state(policy())
    before = gate.teacher_share_summary(policy(), state)
    assert before["teacher_accepted_row_share"] == 0.01
    assert before["base_metric_ready"] is True
    assert before["metric_ready"] is False

    assert gate.record_teacher_share_snapshot(policy(), state, before) is True
    state = gate.load_state(policy())
    after = gate.teacher_share_summary(policy(), state)
    assert after["metric_ready"] is True
    assert after["accounting_ledger_replay_valid"] is True
    assert after["current_snapshot_recorded"] is True
    assert gate.record_teacher_share_snapshot(policy(), state, after) is False


def test_share_ledger_replay_rejects_tampering(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    prepare_root(tmp_path)
    state = gate.load_state(policy())
    share = gate.teacher_share_summary(policy(), state)
    gate.record_teacher_share_snapshot(policy(), state, share)

    ledger = tmp_path / "runtime/data_governance/teacher_share_accounting_ledger.jsonl"
    row = json.loads(ledger.read_text(encoding="utf-8"))
    row["snapshot"]["teacher_accepted_rows"] = 9
    ledger.write_text(json.dumps(row) + "\n", encoding="utf-8")
    state = gate.load_state(policy())
    after = gate.teacher_share_summary(policy(), state)
    assert after["metric_ready"] is False
    assert after["accounting_ledger_replay_valid"] is False
    assert any("hash_mismatch" in item for item in after["accounting_ledger_faults"])


def test_new_admission_report_requires_new_snapshot(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    prepare_root(tmp_path)
    state = gate.load_state(policy())
    share = gate.teacher_share_summary(policy(), state)
    gate.record_teacher_share_snapshot(policy(), state, share)

    admission = tmp_path / "reports/training_data_admission_v1.json"
    value = json.loads(admission.read_text(encoding="utf-8"))
    value["source_admissions"][0]["row_count"] = 199
    write_json(admission, value)
    state = gate.load_state(policy())
    changed = gate.teacher_share_summary(policy(), state)
    assert changed["teacher_accepted_row_share"] == 0.005
    assert changed["metric_ready"] is False
    assert changed["current_snapshot_recorded"] is False
