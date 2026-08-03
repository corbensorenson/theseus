from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_p4_cognitive_compilation as p4  # noqa: E402
import theseus_semantic_ir_production_canary as canary  # noqa: E402


def test_non_claim_canary_config_and_task_are_green_and_uncapped() -> None:
    config = canary.audit_config(
        ROOT / "configs" / "theseus_semantic_ir_production.json"
    )
    task = p4.audit_task(
        ROOT / "configs" / "theseus_semantic_ir_production_non_claim_task_01.json"
    )

    assert config["trigger_state"] == "GREEN"
    assert config["faults"] == []
    assert task["trigger_state"] == "GREEN"
    assert config["counters"]["local_model_calls"] == 0
