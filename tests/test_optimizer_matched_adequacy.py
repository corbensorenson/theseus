from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import optimizer_matched_adequacy as adequacy


def test_config_matches_tuning_opportunity_and_blocks_confirmation_use() -> None:
    config = adequacy.load_config()
    assert {row["id"] for row in config["candidates"]} == {
        "adafactor_mlx",
        "adamw_mlx",
        "muon_mlx",
        "schedule_free_adamw_mlx",
    }
    assert {len(row["profiles"]) for row in config["candidates"]} == {3}
    assert config["hard_boundaries"]["confirmation_surface_consumption"] == 0
    assert config["hard_boundaries"]["heldout_labels_visible_to_tuning"] is False
    assert config["prospective_decision"]["maximum_next_update_absolute_error"] == 1e-7
    assert config["width_transfer"]["source_width"] == 48
    assert config["width_transfer"]["target_width"] == 512


def test_optimizer_policy_cards_are_complete_content_bound_and_tamper_evident() -> None:
    config = adequacy.load_config()
    cards = adequacy.optimizer_policy_cards(config)
    assert adequacy.validate_optimizer_policy_cards(cards, config) == []
    assert cards["adafactor_mlx"]["eligibility_groups_and_fallback"] == {
        "factored": "every trainable tensor with ndim >= 2",
        "fallback": "unfactored second moment for vectors and scalars",
    }
    assert (
        cards["schedule_free_adamw_mlx"]["full_checkpoint_state"][-1]
        == "publication mode restored to training y after evaluation"
    )
    assert "required_observations" in cards["muon_mlx"][
        "approximation_cadence_and_precision"
    ]
    cards["adafactor_mlx"]["clipping"][
        "optimizer_update_rms_threshold"
    ] = 9.0
    assert adequacy.validate_optimizer_policy_cards(cards, config) == [
        "policy_card_digest_invalid:adafactor_mlx"
    ]


def test_tune_split_is_deterministic_and_disjoint() -> None:
    rows = {
        arm: [{"source_identity": f"{arm}-{index}"} for index in range(8)]
        for arm in ("english", "python")
    }
    first = adequacy.split_tuning_rows(rows, tune_rows_per_arm=2, namespace="x")
    second = adequacy.split_tuning_rows(rows, tune_rows_per_arm=2, namespace="x")
    assert first == second
    assert not adequacy.source_sets(first[0]) & adequacy.source_sets(first[1])


def test_profile_selection_uses_only_tune_loss() -> None:
    runs = []
    for profile, loss in (("slow", 2.0), ("best", 1.0), ("other", 1.5)):
        for seed in (1, 2, 3):
            runs.append(
                {
                    "candidate_id": "adamw_mlx",
                    "profile": {"id": profile, "learning_rate": 0.1},
                    "seed": seed,
                    "final_heldout": {"ntp_loss": loss},
                }
            )
    selected = adequacy.select_tuned_profiles(runs)
    assert selected["adamw_mlx"]["profile"]["id"] == "best"


def test_reference_is_retained_when_challenger_misses_pareto_floor() -> None:
    config = adequacy.load_config()
    arms = {arm: {"ntp_loss": 1.0} for arm in config["scoped_arms"]}
    runs = []
    for candidate_id, loss in (
        ("adamw_mlx", 1.0),
        ("muon_mlx", 1.001),
        ("schedule_free_adamw_mlx", 1.002),
    ):
        for seed in config["seeds"]:
            runs.append(
                {
                    "candidate_id": candidate_id,
                    "seed": seed,
                    "training_wall_seconds": 1.0,
                    "optimizer_state_bytes": 100,
                    "final_heldout": {"ntp_loss": loss, "by_arm": arms},
                }
            )
    _comparisons, disposition = adequacy.compare_final(runs, config)
    assert disposition["selected_optimizer"] == "adamw_mlx"
    assert disposition["scientific_falsification_claimed"] is False


def test_target_width_transfer_requires_every_seed_loss_progress_and_exact_resume() -> None:
    config = adequacy.load_config()
    runs = []
    for seed in config["width_transfer"]["seeds"]:
        runs.append(
            {
                "candidate_id": "adamw_mlx",
                "seed": seed,
                "model_config": {"d_model": 512},
                "initial_heldout": {"ntp_loss": 2.0},
                "final_heldout": {"ntp_loss": 1.5},
                "first_gradient_l1": 1.0,
                "checkpoint_reload_exact_state": True,
                "checkpoint_next_update_numerically_equivalent": True,
                "checkpoint_next_update_max_absolute_error": 0.0,
            }
        )
    report = adequacy.assess_width_transfer(runs, config, "adamw_mlx")
    assert report["trigger_state"] == "GREEN"
    assert report["disposition"] == "SELECTED_RECIPE_TRANSFERS_TO_TARGET_WIDTH"
    runs[-1]["final_heldout"]["ntp_loss"] = 2.1
    rejected = adequacy.assess_width_transfer(runs, config, "adamw_mlx")
    assert rejected["trigger_state"] == "RED"


def test_loaded_optimizer_binding_preserves_existing_state() -> None:
    class Dummy:
        def __init__(self) -> None:
            self.calls = 0

        def init(self, parameters) -> None:
            assert parameters == {"weight": "bound"}
            self.calls += 1

    optimizer = Dummy()
    adequacy.bind_loaded_optimizer_state(optimizer, {"weight": "bound"})
    assert optimizer.calls == 1


def test_mutated_public_boundary_fails_closed() -> None:
    config = json.loads(adequacy.DEFAULT_CONFIG.read_text())
    config["hard_boundaries"]["public_training_rows"] = 1
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "bad.json"
        path.write_text(json.dumps(config))
        try:
            adequacy.load_config(path)
        except adequacy.OptimizerAdequacyFault as exc:
            assert "hard_boundary_nonzero" in str(exc)
        else:
            raise AssertionError("public training mutation was accepted")
