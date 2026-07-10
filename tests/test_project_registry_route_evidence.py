from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_project_registry as registry  # noqa: E402
import theseus_control_plane as control_plane  # noqa: E402


class RouteEvidenceContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        (self.root / "reports").mkdir()
        (self.root / "scripts").mkdir()
        self.root_patch = mock.patch.object(registry, "ROOT", self.root)
        self.root_patch.start()

    def tearDown(self) -> None:
        self.root_patch.stop()
        self.tempdir.cleanup()

    def write_json(self, relative: str, payload: dict) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def policy(self, requirement: dict, *, mode: str = "all") -> dict:
        return {
            "route_evidence_contracts": [
                {"id": "route.test", "mode": mode, "requirements": [requirement] if mode == "all" else []}
            ],
            "implementations": [
                {
                    "id": "impl.test",
                    "status": "live",
                    "canonical_entrypoint": "scripts/implementation.py",
                    "route_evidence_contract_id": "route.test",
                    "evidence_outputs": ["reports/gate.json"],
                    "routing_eligibility": {"eligible": True},
                }
            ],
            "surfaces": [
                {
                    "id": "surface.test",
                    "report_outputs": ["reports/gate.json", "reports/history.json"],
                }
            ],
        }

    def test_old_supporting_evidence_does_not_expire_route(self) -> None:
        self.write_json("reports/history.json", {"created_utc": "2000-01-01T00:00:00Z"})
        policy = {"implementations": [], "route_evidence_contracts": [], "surfaces": self.policy({})["surfaces"]}

        rows = registry.report_output_status(policy)
        history = next(row for row in rows if row["path"] == "reports/history.json")

        self.assertEqual(history["evidence_class"], "supporting")
        self.assertEqual(history["status"], "available")
        self.assertFalse(history["stale"])

    def test_retained_non_routeable_implementation_does_not_authorize_report(self) -> None:
        self.write_json(
            "reports/gate.json",
            {"created_utc": datetime.now(timezone.utc).isoformat(), "trigger_state": "RED"},
        )
        requirement = {
            "id": "gate",
            "path": "reports/gate.json",
            "freshness_mode": "source_bound",
            "source_paths": ["scripts/implementation.py"],
            "acceptance": {"field": "trigger_state", "allowed": ["GREEN"]},
        }
        policy = self.policy(requirement)
        policy["implementations"][0]["status"] = "retained"
        policy["implementations"][0]["routing_eligibility"] = {"eligible": False}

        gate = next(
            row for row in registry.report_output_status(policy) if row["path"] == "reports/gate.json"
        )

        self.assertEqual(gate["evidence_class"], "supporting")
        self.assertEqual(gate["status"], "available")
        self.assertFalse(gate["route_required"])

    def test_source_change_invalidates_route_receipt(self) -> None:
        source = self.root / "scripts/implementation.py"
        source.write_text("VALUE = 2\n", encoding="utf-8")
        self.write_json(
            "reports/gate.json",
            {"created_utc": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(), "trigger_state": "GREEN"},
        )
        requirement = {
            "id": "gate",
            "path": "reports/gate.json",
            "freshness_mode": "source_bound",
            "source_paths": ["scripts/implementation.py"],
            "acceptance": {"field": "trigger_state", "allowed": ["GREEN"]},
        }

        result = registry.evaluate_route_evidence_contract(self.policy(requirement), self.policy(requirement)["implementations"][0])

        self.assertIn("route_evidence_source_changed", result["blockers"])
        self.assertEqual(result["requirements"][0]["status"], "stale")

    def test_current_source_bound_receipt_is_replayably_fresh(self) -> None:
        source = self.root / "scripts/implementation.py"
        source.write_text("VALUE = 1\n", encoding="utf-8")
        old = (datetime.now(timezone.utc) - timedelta(hours=1)).timestamp()
        os.utime(source, (old, old))
        self.write_json(
            "reports/gate.json",
            {"created_utc": datetime.now(timezone.utc).isoformat(), "trigger_state": "GREEN"},
        )
        requirement = {
            "id": "gate",
            "path": "reports/gate.json",
            "freshness_mode": "source_bound",
            "source_paths": ["scripts/implementation.py"],
            "acceptance": {"field": "trigger_state", "allowed": ["GREEN"]},
        }
        policy = self.policy(requirement)

        first = registry.evaluate_route_evidence_contract(policy, policy["implementations"][0])
        second = registry.evaluate_route_evidence_contract(policy, policy["implementations"][0])

        self.assertEqual(first["blockers"], [])
        self.assertEqual(second["blockers"], [])
        self.assertEqual(first["requirements"][0]["status"], "fresh")
        self.assertEqual(second["requirements"][0]["status"], "fresh")

    def test_fresh_but_rejected_receipt_cannot_authorize_route(self) -> None:
        source = self.root / "scripts/implementation.py"
        source.write_text("VALUE = 1\n", encoding="utf-8")
        old = (datetime.now(timezone.utc) - timedelta(hours=1)).timestamp()
        os.utime(source, (old, old))
        self.write_json(
            "reports/gate.json",
            {"created_utc": datetime.now(timezone.utc).isoformat(), "trigger_state": "RED"},
        )
        requirement = {
            "id": "gate",
            "path": "reports/gate.json",
            "freshness_mode": "source_bound",
            "source_paths": ["scripts/implementation.py"],
            "acceptance": {"field": "trigger_state", "allowed": ["GREEN"]},
        }
        policy = self.policy(requirement)

        result = registry.evaluate_route_evidence_contract(policy, policy["implementations"][0])

        self.assertIn("route_evidence_acceptance_rejected", result["blockers"])
        self.assertEqual(result["requirements"][0]["status"], "rejected")

    def test_ttl_is_reserved_for_volatile_route_evidence(self) -> None:
        self.write_json(
            "reports/gate.json",
            {"created_utc": (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat(), "trigger_state": "GREEN"},
        )
        requirement = {
            "id": "volatile",
            "path": "reports/gate.json",
            "freshness_mode": "ttl",
            "max_age_hours": 24,
        }
        policy = self.policy(requirement)

        result = registry.evaluate_route_evidence_contract(policy, policy["implementations"][0])

        self.assertIn("route_evidence_ttl_expired", result["blockers"])
        self.assertEqual(result["requirements"][0]["status"], "stale")

    def test_current_invocation_contract_avoids_self_dependency(self) -> None:
        policy = self.policy({}, mode="current_invocation")

        result = registry.evaluate_route_evidence_contract(policy, policy["implementations"][0])

        self.assertEqual(result["blockers"], [])
        self.assertEqual(result["requirements"][0]["status"], "current_invocation")


class LiveRegistryContractTests(unittest.TestCase):
    def replacement_policy(self) -> dict:
        predecessor = {
            "id": "impl.old",
            "abstraction_id": "field.test",
            "status": "retained",
            "superseded_by_implementation_id": "impl.new",
        }
        successor = {
            "id": "impl.new",
            "abstraction_id": "field.test",
            "status": "live",
            "supersedes_implementation_id": "impl.old",
        }
        transaction = {
            "id": "replacement.test",
            "abstraction_id": "field.test",
            "predecessor_implementation_id": "impl.old",
            "successor_implementation_id": "impl.new",
            "decision": "adopt_canonical",
            "state": "qualified",
            "checks": {
                "contract_compatible": True,
                "independent_integrity_green": True,
                "blind_information_flow_green": True,
                "private_replay_above_zero": True,
                "predecessor_retained": True,
            },
            "content_bindings": {
                "test_source": {
                    "path": "tests/test_project_registry_route_evidence.py",
                    "sha256": registry.file_sha256(ROOT / "tests/test_project_registry_route_evidence.py"),
                }
            },
            "evidence_refs": ["tests/test_project_registry_route_evidence.py"],
            "no_cheat_counters": {
                "public_training_rows_written": 0,
                "external_inference_calls": 0,
                "fallback_return_count": 0,
            },
            "rollback": {"armed": True, "mode": "contain_and_requalify_predecessor"},
            "claim_boundary": {
                "supported_claim": "private proposer floor",
                "non_claims": ["not public transfer"],
            },
        }
        return {
            "abstraction_registry_contract": {
                "replacement_transaction_required_fields": [
                    "id",
                    "abstraction_id",
                    "predecessor_implementation_id",
                    "successor_implementation_id",
                    "decision",
                    "state",
                    "checks",
                    "content_bindings",
                    "evidence_refs",
                    "no_cheat_counters",
                    "rollback",
                    "claim_boundary",
                ],
                "replacement_transaction_required_checks": [
                    "contract_compatible",
                    "independent_integrity_green",
                    "blind_information_flow_green",
                    "private_replay_above_zero",
                    "predecessor_retained",
                ],
            },
            "abstractions": [
                {
                    "id": "field.test",
                    "canonical_implementation_id": "impl.new",
                }
            ],
            "implementations": [predecessor, successor],
            "implementation_replacement_transactions": [transaction],
        }

    def test_valid_replacement_transaction_preserves_lineage_and_route_decision(self) -> None:
        policy = self.replacement_policy()

        gaps = registry.implementation_replacement_gaps(policy)

        self.assertEqual(gaps, [])

    def test_missing_replacement_transaction_blocks_declared_successor(self) -> None:
        policy = self.replacement_policy()
        policy["implementation_replacement_transactions"] = []

        gaps = registry.implementation_replacement_gaps(policy)

        self.assertIn(
            "canonical_successor_replacement_transaction_count_invalid",
            {row["kind"] for row in gaps},
        )

    def test_failed_independent_check_or_nonzero_no_cheat_counter_blocks_replacement(self) -> None:
        policy = self.replacement_policy()
        transaction = policy["implementation_replacement_transactions"][0]
        transaction["checks"]["private_replay_above_zero"] = False
        transaction["no_cheat_counters"]["fallback_return_count"] = 1

        gaps = registry.implementation_replacement_gaps(policy)
        kinds = {row["kind"] for row in gaps}

        self.assertIn("replacement_required_checks_failed", kinds)
        self.assertIn("replacement_no_cheat_counters_invalid", kinds)

    def test_forged_content_hash_blocks_replacement(self) -> None:
        policy = self.replacement_policy()
        policy["implementation_replacement_transactions"][0]["content_bindings"]["test_source"]["sha256"] = "0" * 64

        gaps = registry.implementation_replacement_gaps(policy)

        self.assertIn("replacement_exact_content_bindings_invalid", {row["kind"] for row in gaps})

    def test_registered_verification_commands_do_not_pass_source_modules_as_runtime_arguments(self) -> None:
        policy = json.loads((ROOT / "configs/project_manifest_registry.json").read_text(encoding="utf-8"))
        commands: list[str] = []

        def collect(value: object) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "verification_command" and isinstance(item, str):
                        commands.append(item)
                    else:
                        collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        collect(policy)
        positional_source_pattern = re.compile(
            r"python3 scripts/\S+\.py(?:(?!&&).)* "
            r"scripts/neural_seed_candidate_generation\.py scripts/neural_seed_report_io\.py"
        )
        invalid = [command for command in commands if positional_source_pattern.search(command)]

        self.assertEqual(invalid, [])

    def test_control_plane_fails_closed_on_blocked_route_evidence(self) -> None:
        registry_summary = {
            "abstraction_registry_gap_count": 0,
            "stable_capability_field_gap_count": 0,
            "stable_capability_field_health_red_count": 0,
            "implementation_routing_blocker_count": 0,
            "blocked_route_evidence_output_count": 1,
            "registry_hard_governance_violation_count": 0,
            "routing_eligible_implementation_count": 3,
            "route_evidence_output_count": 4,
            "supporting_evidence_output_count": 20,
        }
        payloads = {
            "theseus_project_registry": {"trigger_state": "GREEN", "summary": registry_summary},
            "viea_spine_materialized_view": {
                "trigger_state": "GREEN",
                "summary": {
                    "record_count": 1,
                    "claim_ledger_entry_count": 1,
                    "semantic_ir_record_count": 1,
                    "simulation_fidelity_record_count": 1,
                    "governance_record_count": 1,
                    "failure_boundary_count": 1,
                    "no_cheat_fault_count": 0,
                },
            },
            "teacher_share_ledger_summary": {
                "trigger_state": "GREEN",
                "summary": {
                    "metric_ready": True,
                    "teacher_share_within_cap": True,
                    "runtime_external_inference_calls": 0,
                    "public_training_rows_written": 0,
                },
            },
        }

        gates = control_plane.build_gates(payloads, [], [], {"current_unstored_count": 0, "current_truncated_without_snapshot_count": 0})

        self.assertFalse(gates["registry_governance_ready"]["passed"])
        self.assertEqual(gates["registry_governance_ready"]["evidence"]["blocked_route_evidence_output_count"], 1)

    def test_control_plane_does_not_consume_stale_payloads(self) -> None:
        records = [
            {"id": "fresh", "stale": False, "missing": False, "payload": {"trigger_state": "GREEN"}},
            {"id": "stale", "stale": True, "missing": False, "payload": {"trigger_state": "GREEN"}},
            {"id": "missing", "stale": False, "missing": True, "payload": {"trigger_state": "GREEN"}},
        ]

        trusted = control_plane.trusted_report_payloads(records)

        self.assertEqual(trusted, {"fresh": {"trigger_state": "GREEN"}})


if __name__ == "__main__":
    unittest.main()
