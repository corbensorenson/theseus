from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_supply_chain as supply_chain  # noqa: E402


class AIBOMTests(unittest.TestCase):
    def test_requested_resolved_and_observed_identity_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "scripts").mkdir()
            (root / "scripts" / "worker.py").write_text("import json\nVALUE = 1\n", encoding="utf-8")
            policy = {
                "surfaces": [{"id": "worker", "artifact_type": "runtime"}],
                "implementations": [
                    {
                        "id": "impl.worker.v1",
                        "abstraction_id": "worker",
                        "status": "live",
                        "backend": "python",
                        "canonical_entrypoint": "scripts/worker.py",
                        "dependencies": ["scripts/worker.py"],
                        "evidence_outputs": ["reports/missing.json"],
                    }
                ]
            }
            entries = [{"kind": "file", "path": "scripts/worker.py", "surface_id": "worker", "bytes": 22}]

            report = supply_chain.build_aibom(root, policy, entries)

            source = next(row for row in report["artifacts"] if row["artifact_kind"] == "code")
            self.assertEqual(source["requested_identity"]["locator"], "scripts/worker.py")
            self.assertEqual(source["resolved_identity"]["locator"], "scripts/worker.py")
            self.assertTrue(source["observed_identity"]["sha256"].startswith("sha256:"))
            self.assertEqual(report["summary"]["missing_identity_count"], 0)
            self.assertEqual(report["summary"]["not_materialized_evidence_count"], 1)
            self.assertIn("runtime", report["summary"]["artifact_domain_counts"])
            self.assertFalse(report["claims"]["signed_supply_chain"])
            self.assertFalse(report["claims"]["advisory_freshness"])
            self.assertFalse(report["claims"]["reproducible_build"])

    def test_substituted_dependency_invalidates_all_descendants(self) -> None:
        dependencies = [
            {"dependency_artifact_id": "source", "dependent_artifact_id": "implementation"},
            {"dependency_artifact_id": "implementation", "dependent_artifact_id": "checkpoint"},
            {"dependency_artifact_id": "checkpoint", "dependent_artifact_id": "release"},
        ]

        record = supply_chain.descendant_invalidation(
            {"source", "implementation", "checkpoint", "release"},
            dependencies,
            {"source"},
        )

        self.assertTrue(record["closure_complete"])
        self.assertEqual(
            set(record["invalidated_artifact_ids"]),
            {"source", "implementation", "checkpoint", "release"},
        )

    def test_unknown_descendant_fails_closure(self) -> None:
        record = supply_chain.descendant_invalidation(
            {"source"},
            [{"dependency_artifact_id": "source", "dependent_artifact_id": "unregistered-release"}],
            {"source"},
        )

        self.assertFalse(record["closure_complete"])
        self.assertEqual(record["unknown_artifact_ids"], ["unregistered-release"])


if __name__ == "__main__":
    unittest.main()
