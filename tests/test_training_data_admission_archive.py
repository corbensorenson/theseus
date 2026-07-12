from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_archive_resolver  # noqa: E402
import training_data_admission_v1 as admission  # noqa: E402


def test_teacher_manifest_admission_follows_retention_pointer(
    tmp_path: Path, monkeypatch
) -> None:
    reports = tmp_path / "reports"
    archive = tmp_path / "archive" / "teacher.json.gz"
    reports.mkdir(parents=True)
    archive.parent.mkdir(parents=True)
    manifest = {
        "summary": {
            "manifest_ready_for_distillation": True,
            "row_count": 1,
            "admission_safety_checks_clean": True,
            "public_overlap_hits": 0,
            "holdout_overlap_hits": 0,
        },
        "admission_checks": {"runtime_serving_forbidden": True},
        "rows": [{"row_id": "accepted-1"}],
    }
    with gzip.open(archive, "wt", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    (reports / "teacher_distillation_manifest.json").write_text(
        json.dumps(
            {
                "policy": "project_theseus_archived_artifact_pointer_v1",
                "archive_path": str(archive),
            }
        )
    )
    (reports / "teacher_distillation_gate.json").write_text(
        json.dumps({"trigger_state": "GREEN", "summary": {"distillation_allowed": True}})
    )
    (reports / "teacher_distillation_ledger.jsonl").write_text("")
    monkeypatch.setattr(admission, "ROOT", tmp_path)
    monkeypatch.setattr(theseus_archive_resolver, "ROOT", tmp_path)

    result = admission.audit_teacher_distillation_gate()

    assert result["manifest_ready_for_distillation"] is True
    assert result["accepted_manifest_row_count"] == 1
    assert result["runtime_serving_forbidden"] is True
