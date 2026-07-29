from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import vcm_consumer_integration_gate as gate  # noqa: E402


def test_read_json_follows_retention_pointer(tmp_path: Path) -> None:
    archive = tmp_path / "consumer.json.gz"
    payload = {"trigger_state": "GREEN", "vcm_consumer_abi": {"ready": True}}
    with gzip.open(archive, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    pointer = tmp_path / "consumer.json"
    pointer.write_text(
        json.dumps(
            {
                "policy": "project_theseus_archived_artifact_pointer_v1",
                "archive_path": str(archive),
            }
        ),
        encoding="utf-8",
    )

    assert gate.read_json(pointer) == payload
