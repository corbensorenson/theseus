from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import kerc_implementation_fidelity_gate as fidelity  # noqa: E402


class KercImplementationFidelityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract_path = ROOT / "configs" / "kerc_implementation_fidelity.json"
        cls.contract = json.loads(cls.contract_path.read_text(encoding="utf-8"))

    def test_current_contract_is_source_bound_and_honest(self) -> None:
        report = fidelity.audit_contract(
            copy.deepcopy(self.contract),
            contract_path=self.contract_path,
        )

        self.assertEqual(report["trigger_state"], "GREEN")
        self.assertEqual(report["faults"], [])
        self.assertEqual(report["summary"]["record_count"], 22580)
        self.assertEqual(
            report["corpus_audit"]["observed"]["multi_node_program_count"], 11156
        )
        self.assertEqual(
            report["corpus_audit"]["observed"]["multi_claim_answer_count"], 1700
        )
        self.assertEqual(
            report["corpus_audit"]["observed"]["byte_literal_value_count"], 18759
        )
        self.assertEqual(report["summary"]["hypothesis_evidence_active_count"], 0)
        self.assertFalse(report["summary"]["byte_literal_mutation_rejected"])
        self.assertFalse(report["summary"]["interaction_label_depends_on_global_dictionary"])

    def test_per_unit_allocator_cannot_be_relabelled_faithful_without_units(self) -> None:
        contract = copy.deepcopy(self.contract)
        row = next(item for item in contract["mechanisms"] if item["id"] == "kerc.learned_per_unit_allocator")
        row["status"] = "faithful"
        report = fidelity.audit_mechanisms(
            contract,
            root=ROOT,
            corpus_report={"observed": contract["observed_corpus_contract"]},
        )

        self.assertIn("per_unit_allocator_overclaimed", [item["kind"] for item in report["faults"]])

    def test_empty_interaction_dictionary_cannot_support_amortization_claim(self) -> None:
        contract = copy.deepcopy(self.contract)
        row = next(item for item in contract["mechanisms"] if item["id"] == "kerc.interaction_amortization")
        row["status"] = "faithful"
        report = fidelity.audit_mechanisms(
            contract,
            root=ROOT,
            corpus_report={"observed": contract["observed_corpus_contract"]},
        )

        self.assertIn("interaction_amortization_overclaimed", [item["kind"] for item in report["faults"]])

    def test_hypothesis_cannot_activate_before_matched_campaign(self) -> None:
        contract = copy.deepcopy(self.contract)
        contract["hypotheses"][0]["status"] = "faithful"
        report = fidelity.audit_hypotheses(contract)

        self.assertEqual(report["active_count"], 1)
        self.assertIn("hypothesis_evidence_activated_before_k8", [item["kind"] for item in report["faults"]])

    def test_missing_mechanism_and_unknown_status_fail_closed(self) -> None:
        contract = copy.deepcopy(self.contract)
        contract["mechanisms"] = contract["mechanisms"][1:]
        contract["mechanisms"][0]["status"] = "complete"
        report = fidelity.audit_mechanisms(
            contract,
            root=ROOT,
            corpus_report={"observed": contract["observed_corpus_contract"]},
        )
        kinds = [item["kind"] for item in report["faults"]]

        self.assertIn("missing_required_mechanisms", kinds)
        self.assertIn("mechanism_status_invalid", kinds)

    def test_behavior_probes_recompute_known_construct_gaps(self) -> None:
        with (ROOT / "runtime/kerc_semantic_corpus/candidate_records.jsonl").open("r", encoding="utf-8") as handle:
            sample = json.loads(next(handle))
        report = fidelity.audit_behavioral_claim_probes(sample)

        self.assertEqual(report["interaction_label_empty_segments"], 0)
        self.assertEqual(report["interaction_label_segment_state_only"], 1)
        self.assertFalse(report["interaction_label_depends_on_global_dictionary"])
        self.assertFalse(report["byte_literal_mutation_rejected"])
        self.assertEqual(report["faults"], [])


if __name__ == "__main__":
    unittest.main()
