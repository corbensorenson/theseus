from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import relational_dimension_compiler as rdc
import semantic_ir


class RelationalDimensionCompilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = rdc.reference_fixture()

    def test_semantic_ir_owns_relational_lowering(self) -> None:
        contract = semantic_ir.relational_dimension_contract()
        self.assertEqual("semantic_ir", contract["owner"])
        self.assertEqual(rdc.POLICY, contract["policy"])
        self.assertEqual("none", contract["effect_authority"])

    def test_role_reification_roundtrips_finite_nary_relation(self) -> None:
        relation = self.fixture["relation"]
        self.assertEqual(4, len(relation.incidences))
        self.assertEqual(
            {
                "object": ("report",),
                "recipient": ("bob",),
                "sender": ("alice",),
                "time": ("noon",),
            },
            rdc.reconstruct_roles(relation),
        )

    def test_branch_leakage_and_false_actualization_fail_closed(self) -> None:
        branch = rdc.BranchRecord("hypothesis:1", "actual", "simulation", False, 1)
        entities = {
            "a": rdc.EntityRecord("a", "agent", "hypothesis:1", "p:a"),
            "b": rdc.EntityRecord("b", "agent", "actual", "p:b"),
        }
        schema = rdc.RelationSchema(
            "link",
            1,
            (rdc.RoleSpec("left", "agent"), rdc.RoleSpec("right", "agent")),
            2,
            "ordered",
            (),
        )
        with self.assertRaisesRegex(rdc.RDCFault, "relation_branch_leakage"):
            rdc.reify_relation(
                relation_id="link:1",
                schema=schema,
                participants={"left": ["a"], "right": ["b"]},
                entities=entities,
                branch=branch,
                provenance_digests=("p",),
                confidence=0.5,
                uncertainty_state="ambiguous",
            )
        relation = self.fixture["relation"]
        proposed = rdc.replace(relation, branch_id="hypothesis:1", lifecycle_state="qualified")
        with self.assertRaisesRegex(
            rdc.RDCFault, "hypothetical_branch_actualization_forbidden"
        ):
            rdc.transition_relation(
                proposed,
                next_state="observed",
                branch=branch,
                event_kind="observation",
                evidence_digest="e",
            )

    def test_proposal_denominator_and_recall_are_explicit(self) -> None:
        receipt = rdc.proposal_receipt(
            request={"q": 1},
            denominator_ids=("r1", "r2", "r3", "r4"),
            proposed_ids=("r1", "r3"),
            qualified_ids=("r1",),
            known_relevant_ids=("r1", "r2"),
            proposal_sources=("schema",),
        )
        self.assertEqual(4, receipt.denominator_count)
        self.assertEqual(0.5, receipt.proposal_recall)
        with self.assertRaisesRegex(rdc.RDCFault, "proposal_outside_denominator"):
            rdc.proposal_receipt(
                request={},
                denominator_ids=("r1",),
                proposed_ids=("oracle",),
                qualified_ids=(),
                proposal_sources=("invalid",),
            )

    def test_operator_router_selects_least_total_cost_or_abstains(self) -> None:
        operators = (
            rdc.OperatorCard("pair", 2, ("s",), "exact", "i", "o", "ordered", 2, 1, 1, 1, ()),
            rdc.OperatorCard("triad", 3, ("s",), "learned", "i", "o", "ordered", 8, 4, 4, 2, ()),
        )
        selected = rdc.select_operator(
            request={"x": 1},
            schema_id="s",
            minimum_primitive_arity=2,
            operators=operators,
            maximum_total_cost=10,
            weights={},
        )
        self.assertEqual("pair", selected.operator_id)
        abstained = rdc.select_operator(
            request={"x": 1},
            schema_id="s",
            minimum_primitive_arity=3,
            operators=operators,
            maximum_total_cost=5,
            weights={},
        )
        self.assertTrue(abstained.abstained)

    def test_contraction_is_query_relative_and_reversible(self) -> None:
        certificate = rdc.contract_relations(
            relations=(self.fixture["relation"],),
            macro_entity_id="macro:1",
            query_family_ids=("q:covered",),
            environment_class="env:1",
            discrepancy_tolerance=0.05,
            boundary_entity_ids=("alice", "bob"),
            expansion_triggers=("evidence_revoked",),
        )
        use = rdc.resolve_contraction(
            certificate,
            query_family_id="q:covered",
            environment_class="env:1",
            observed_discrepancy=0.01,
        )
        self.assertTrue(use["use_macro_object"])
        expand = rdc.resolve_contraction(
            certificate,
            query_family_id="q:other",
            environment_class="env:1",
            observed_discrepancy=0.01,
        )
        self.assertTrue(expand["expand_source_complex"])
        self.assertEqual(("transfer:1",), certificate.source_relation_ids)

    def test_columnar_lowering_is_content_bound_and_rejects_mutation(self) -> None:
        packet = self.fixture["lowering"]
        receipt = rdc.validate_lowering(packet)
        self.assertEqual(4, receipt["entity_count"])
        self.assertEqual(4, receipt["incidence_count"])
        mutated = copy.deepcopy(packet)
        mutated["incidences"]["participant_id"][0] = "unknown"
        with self.assertRaisesRegex(rdc.RDCFault, "lowering_digest_mismatch"):
            rdc.validate_lowering(mutated)

    def test_schema_migration_roundtrips_and_rejects_mutated_target(self) -> None:
        source = self.fixture["schema"]
        target = rdc.RelationSchema(
            schema_id="transfer",
            version=2,
            roles=(
                rdc.RoleSpec("agent_from", "agent"),
                rdc.RoleSpec("payload", "artifact"),
                rdc.RoleSpec("agent_to", "agent"),
                rdc.RoleSpec("at_time", "time"),
            ),
            semantic_arity=4,
            symmetry="ordered",
            dimensional_axes=source.dimensional_axes,
        )
        migration = rdc.SchemaMigration(
            schema_id="transfer",
            from_version=1,
            to_version=2,
            role_map=(
                ("sender", "agent_from"),
                ("object", "payload"),
                ("recipient", "agent_to"),
                ("time", "at_time"),
            ),
        )
        migrated, receipt = rdc.migrate_relation_schema(
            self.fixture["relation"],
            source_schema=source,
            target_schema=target,
            migration=migration,
            entities=self.fixture["entities"],
            branch=self.fixture["branch"],
        )
        self.assertEqual(2, migrated.schema_version)
        self.assertTrue(receipt.exact_rollback_verified)
        restored = rdc.rollback_relation_schema(
            migrated,
            source_schema=source,
            target_schema=target,
            migration=migration,
            receipt=receipt,
            entities=self.fixture["entities"],
            branch=self.fixture["branch"],
        )
        self.assertEqual(self.fixture["relation"], restored)
        with self.assertRaisesRegex(rdc.RDCFault, "schema_migration_receipt_mismatch"):
            rdc.rollback_relation_schema(
                rdc.replace(migrated, confidence=0.1),
                source_schema=source,
                target_schema=target,
                migration=migration,
                receipt=receipt,
                entities=self.fixture["entities"],
                branch=self.fixture["branch"],
            )

    def test_monitored_specialist_retains_and_rolls_back_to_slow_path(self) -> None:
        certificate = rdc.contract_relations(
            relations=(self.fixture["relation"],),
            macro_entity_id="macro:1",
            query_family_ids=("q:covered",),
            environment_class="env:1",
            discrepancy_tolerance=0.05,
            boundary_entity_ids=("alice", "bob"),
            expansion_triggers=("evidence_revoked",),
        )
        observations = tuple(
            rdc.SpecialistObservation(f"q:{index}", f"a:{index}", f"a:{index}", True, 2.0)
            for index in range(4)
        )
        specialist = rdc.compile_monitored_specialist(
            certificate=certificate,
            schema_id="transfer",
            schema_version=1,
            slow_path_operator_id="rdc_slow_path",
            observations=observations,
            current_revision=10,
            lifetime_revisions=5,
            maximum_discrepancy_rate=0.0,
            minimum_verified_observations=4,
        )
        routed = rdc.route_compiled_specialist(
            specialist,
            query_family_id="q:covered",
            environment_class="env:1",
            schema_id="transfer",
            schema_version=1,
            current_revision=11,
            source_complex_digest=certificate.source_complex_digest,
        )
        self.assertTrue(routed["use_specialist"])
        self.assertTrue(routed["retained_slow_path"])
        rollback = rdc.route_compiled_specialist(
            specialist,
            query_family_id="q:covered",
            environment_class="env:1",
            schema_id="transfer",
            schema_version=1,
            current_revision=11,
            source_complex_digest=certificate.source_complex_digest,
            monitoring_observations=(
                rdc.SpecialistObservation("q:new", "wrong", "slow", True, 1.0),
            ),
        )
        self.assertFalse(rollback["use_specialist"])
        self.assertEqual("rdc_slow_path", rollback["operator_id"])
        self.assertIn("monitoring_discrepancy_exceeded", rollback["rollback_reasons"])

    def test_sparse_factorized_lowering_has_numpy_mlx_parity(self) -> None:
        batch = rdc.lower_sparse_relation_batch(
            self.fixture["lowering"],
            entity_type_vocabulary=("agent", "artifact", "time"),
            schema_vocabulary=("transfer",),
            role_vocabulary=("object", "recipient", "sender", "time"),
        )
        receipt = rdc.validate_sparse_relation_batch(batch)
        self.assertEqual(4, receipt["incidence_count"])
        rng = np.random.default_rng(7)
        entities = rng.normal(size=(4, 8)).astype(np.float32)
        roles = rng.normal(size=(4, 8)).astype(np.float32)
        schemas = rng.normal(size=(1, 8)).astype(np.float32)
        reference = rdc.factorized_relation_pool_numpy(
            batch,
            entity_features=entities,
            role_factors=roles,
            schema_factors=schemas,
        )
        candidate = np.asarray(
            rdc.factorized_relation_pool_mlx(
                batch,
                entity_features=entities,
                role_factors=roles,
                schema_factors=schemas,
            )
        )
        np.testing.assert_allclose(reference, candidate, rtol=1e-5, atol=1e-6)
        mutated = copy.deepcopy(batch)
        mutated["incidence_role_id"][0] = 99
        with self.assertRaisesRegex(rdc.RDCFault, "sparse_lowering_digest_mismatch"):
            rdc.validate_sparse_relation_batch(mutated)

    def test_reference_suite_is_green_without_truth_or_capability_claim(self) -> None:
        report = rdc.run_reference_suite()
        self.assertEqual("GREEN", report["trigger_state"])
        self.assertEqual(0, report["no_cheat_counters"]["public_training_rows"])
        self.assertIn("not_relation_truth", report["claim_boundary"])


if __name__ == "__main__":
    unittest.main()
