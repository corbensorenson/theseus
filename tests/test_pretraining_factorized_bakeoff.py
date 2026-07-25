from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pretraining_factorized_bakeoff as bakeoff
import rdc_kerc_resource_disposition


def kerc_report(*, adopted: bool) -> dict:
    source = ROOT / "configs" / "rdc_kerc_matched_adequacy.json"
    artifact = {
        "path": str(source.relative_to(ROOT)),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "bytes": source.stat().st_size,
    }
    checks = {
        "seed_count": True,
        "matched_parameter_count": True,
        "equal_optimizer_steps": True,
        "mean_exact_match_gain": adopted,
        "exact_seed_win_fraction": adopted,
        "mean_target_similarity_gain": adopted,
        "nonempty_nonregression": True,
        "byte_serialization_nonregression": True,
        "lifecycle_cost": True,
    }
    return {
        "disposition": (
            "ADOPT_RDC_KERC_FIRST_CAMPAIGN"
            if adopted
            else "SCOPED_DISCOVERY_EXCLUDED_FROM_FIRST_CAMPAIGN"
        ),
        "scientific_falsification_claimed": False,
        "metrics": {"seed_count": 3},
        "checks": checks,
        "source_artifacts": {f"artifact_{index}": artifact for index in range(8)},
        "public_training_rows": 0,
        "public_evaluation_rows": 0,
        "external_inference_calls": 0,
        "fallback_template_router_tool_credit": 0,
        "production_checkpoint_mutations": 0,
        "claim_boundary": "bounded private engineering selection",
        "reentry_condition": "larger matched verifier-bearing campaign",
    }


def test_kerc_adoption_requires_all_three_seed_gates() -> None:
    config = bakeoff.load_config()
    result = bakeoff.evaluate_kerc(config, kerc_report(adopted=True))

    assert result["disposition"] == "ADOPT_RDC_KERC_FIRST_CAMPAIGN"
    assert result["scientific_falsification_claimed"] is False
    assert all(result["checks"].values())


def test_kerc_loss_is_scoped_campaign_exclusion_not_falsification() -> None:
    config = bakeoff.load_config()
    result = bakeoff.evaluate_kerc(config, kerc_report(adopted=False))

    assert result["disposition"] == "SCOPED_DISCOVERY_EXCLUDED_FROM_FIRST_CAMPAIGN"
    assert result["scientific_falsification_claimed"] is False
    assert result["reentry_condition"]


def test_kerc_resource_deferral_selects_control_without_falsification() -> None:
    config = bakeoff.load_config()
    result = bakeoff.evaluate_kerc(
        config, rdc_kerc_resource_disposition.build_report()
    )

    assert result["disposition"] == "RESOURCE_DEFERRED_ON_THIS_HOST"
    assert result["scientific_falsification_claimed"] is False
    assert result["metrics"]["long_panel_peak_mib"] > 6144


def test_kerc_stale_source_or_assisted_credit_fails_closed() -> None:
    config = bakeoff.load_config()
    report = kerc_report(adopted=False)
    report["source_artifacts"]["artifact_0"] = {
        **report["source_artifacts"]["artifact_0"],
        "sha256": "0" * 64,
    }
    with pytest.raises(bakeoff.BakeoffFault, match="source_artifact_stale"):
        bakeoff.evaluate_kerc(config, report)

    report = kerc_report(adopted=False)
    report["fallback_template_router_tool_credit"] = 1
    with pytest.raises(bakeoff.BakeoffFault, match="no_cheat_counter_nonzero"):
        bakeoff.evaluate_kerc(config, report)


def test_config_rejects_any_nonzero_no_cheat_boundary(tmp_path: Path) -> None:
    config = copy.deepcopy(bakeoff.load_config())
    config["hard_boundaries"]["public_training_rows"] = 1
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(bakeoff.BakeoffFault, match="hard_boundary_nonzero"):
        bakeoff.load_config(path)
