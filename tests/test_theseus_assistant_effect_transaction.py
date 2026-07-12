from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_assistant_runtime as runtime  # noqa: E402
from viea_spine_records import audit_effect_complete_transaction  # noqa: E402


def route_packet() -> dict:
    return {
        "ready": True,
        "selected_route": {"id": "route.procedural.planning.v1"},
    }


class AssistantEffectTransactionTests(unittest.TestCase):
    def setUp(self) -> None:
        test_effect_root = ROOT / "runtime" / "assistant_effects"
        test_effect_root.mkdir(parents=True, exist_ok=True)
        self.tempdir = tempfile.TemporaryDirectory(dir=test_effect_root)
        self.allowed_root = Path(self.tempdir.name)
        self.target = self.allowed_root / "default_route_authority.json"
        self.outside = self.allowed_root.with_name(f"{self.allowed_root.name}-outside.json")

    def tearDown(self) -> None:
        self.outside.unlink(missing_ok=True)
        self.tempdir.cleanup()

    def run_canary(self, target: Path | None = None) -> dict:
        return runtime.run_local_effect_canary(
            enabled=True,
            target=target or self.target,
            allowed_root=self.allowed_root,
            session_id="test-session",
            intent="planning",
            prompt_hash="a" * 64,
            procedural_default_route=route_packet(),
        )

    def test_new_route_authority_file_is_observed_then_removed(self) -> None:
        result = self.run_canary()

        self.assertTrue(result["ready"])
        self.assertTrue(result["observation"]["matches_intent"])
        self.assertTrue(result["rollback"]["complete"])
        self.assertTrue(result["rollback"]["removed_new_path"])
        self.assertEqual(result["rollback"]["residual_count"], 0)
        self.assertFalse(self.target.exists())

    def test_existing_bytes_and_mode_are_restored_exactly(self) -> None:
        self.target.parent.mkdir(parents=True, exist_ok=True)
        prior = b"prior-route-state\n"
        self.target.write_bytes(prior)
        os.chmod(self.target, 0o640)

        result = self.run_canary()

        self.assertTrue(result["ready"])
        self.assertTrue(result["rollback"]["restored_prior_bytes"])
        self.assertEqual(self.target.read_bytes(), prior)
        self.assertEqual(self.target.stat().st_mode & 0o777, 0o640)
        self.assertEqual(result["rollback"]["before_identity"], result["rollback"]["final_identity"])

    def test_path_escape_and_symlink_are_denied_without_effect(self) -> None:
        outside = self.outside
        escaped = self.run_canary(outside)
        self.assertFalse(escaped["ready"])
        self.assertEqual(escaped["residuals"][0]["kind"], "effect_target_denied")
        self.assertFalse(outside.exists())

        self.allowed_root.mkdir(parents=True, exist_ok=True)
        outside.write_text("unchanged", encoding="utf-8")
        self.target.symlink_to(outside)
        linked = self.run_canary()
        self.assertFalse(linked["ready"])
        self.assertEqual(outside.read_text(encoding="utf-8"), "unchanged")

    def test_missing_or_unready_route_cannot_pass_effect_observation(self) -> None:
        result = runtime.run_local_effect_canary(
            enabled=True,
            target=self.target,
            allowed_root=self.allowed_root,
            session_id="test-session",
            intent="planning",
            prompt_hash="a" * 64,
            procedural_default_route={"ready": False, "selected_route": {}},
        )

        self.assertFalse(result["ready"])
        self.assertFalse(result["observation"]["matches_intent"])
        self.assertTrue(result["rollback"]["complete"])
        self.assertFalse(self.target.exists())

    def test_canonical_effect_receipt_passes_independent_audit_and_mutations_fail(self) -> None:
        effect = self.run_canary()
        report = audit_report(effect)
        route_id = effect["observation"]["expected_route_id"]

        audit = audit_effect_complete_transaction(report, expected_route_ids={route_id})

        self.assertTrue(audit["valid"])
        self.assertEqual(audit["support_state"], "replayable-reference-backed")
        self.assertEqual(audit["expected_invalid_control_count"], 8)
        self.assertEqual(audit["expected_invalid_rejected_count"], 8)

    def test_effect_audit_rejects_route_outside_adopted_set(self) -> None:
        report = audit_report(self.run_canary())

        audit = audit_effect_complete_transaction(report, expected_route_ids={"default.other.route"})

        self.assertFalse(audit["valid"])
        self.assertIn("route_is_expected", audit["hard_gaps"])


def audit_report(effect: dict) -> dict:
    transaction_id = effect["transaction_id"]
    counters = {
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    inventory_id = "effect-inventory-test"
    observation_id = "effect-observation-test"
    trace = [
        {
            "record_id": inventory_id,
            "record_type": "effect_inventory",
            "content": {
                "transaction_id": transaction_id,
                "declared_effects": effect["effect_inventory"],
                "proposer_id": effect["proposer_id"],
                "undeclared_effects_permitted": False,
            },
            **counters,
        },
        {
            "record_id": observation_id,
            "record_type": "effect_observation_record",
            "content": {
                "transaction_id": transaction_id,
                "effect_inventory_record_id": inventory_id,
                "observation": effect["observation"],
                "observer_id": effect["observer_id"],
                "observer_independent_from_proposer": True,
            },
            **counters,
        },
        {
            "record_id": "effect-rollback-test",
            "record_type": "rollback_completeness_record",
            "content": {
                "transaction_id": transaction_id,
                "effect_inventory_record_id": inventory_id,
                "effect_observation_record_id": observation_id,
                "rollback": effect["rollback"],
                "evaluator_id": effect["evaluator_id"],
                "evaluator_independent_from_proposer_and_observer": True,
                "ready": True,
                "residuals": [],
            },
            **counters,
        },
    ]
    return {
        "trigger_state": "GREEN",
        "summary": {
            "effect_canary_enabled": True,
            "effect_canary_ready": True,
            "effect_canary_transaction_id": transaction_id,
            "effect_canary_first_effect_identity": effect["rollback"]["first_effect_identity"],
            "effect_canary_final_effect_identity": effect["rollback"]["final_identity"],
            "effect_canary_rollback_complete": True,
            **counters,
        },
        "effect_canary": effect,
        "assistant_viea_trace": trace,
        **counters,
    }


if __name__ == "__main__":
    unittest.main()
