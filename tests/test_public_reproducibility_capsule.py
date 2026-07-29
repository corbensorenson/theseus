from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import public_reproducibility_capsule as capsule  # noqa: E402


def test_public_capsule_trains_resumes_verifies_and_hashes(tmp_path: Path) -> None:
    report = capsule.execute(
        ROOT / "examples" / "public_repro_capsule" / "config.json",
        tmp_path,
    )

    assert report["trigger_state"] == "GREEN"
    assert report["training"]["resume_exact_match"] is True
    assert report["verification"]["passed"] is True
    assert report["inputs"]["license"] == "Apache-2.0"
    assert len(report["artifacts"]) == 4
    assert all(len(row["sha256"]) == 64 for row in report["artifacts"])
    assert report["boundaries"]["private_training_rows"] == 0
    assert report["boundaries"]["user_data_rows"] == 0
    assert report["boundaries"]["external_inference_calls"] == 0
