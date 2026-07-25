from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import soap_full_shape_resource_preflight as preflight


class SoapFullShapeResourcePreflightTests(unittest.TestCase):
    def test_shape_accounting_matches_frozen_57m_checkpoint(self) -> None:
        config = preflight.load_config()
        rows = preflight.safetensor_shapes(preflight.resolve(config["checkpoint"]))
        accounting = preflight.full_shape_accounting(rows, config)
        self.assertEqual(197, accounting["tensor_count"])
        self.assertEqual(154, accounting["matrix_count"])
        self.assertEqual(54_836_746, accounting["parameter_elements"])
        self.assertEqual(422_314_025, accounting["full_shape_factor_elements"])
        self.assertEqual(8195, accounting["largest_factor_dimension"])

    def test_projection_is_monotonic_in_matrix_dimension(self) -> None:
        timing = {"cpu_timings": [{"dimension": 128, "median_seconds": 0.01}]}
        small = preflight.project_refresh_seconds([[128, 128]], timing, 4.0)
        large = preflight.project_refresh_seconds([[256, 256]], timing, 4.0)
        self.assertGreater(
            large["optimistic_full_refresh_seconds"],
            small["optimistic_full_refresh_seconds"],
        )

    def test_execution_is_resource_disposition_not_quality_falsification(self) -> None:
        report = preflight.execute()
        self.assertEqual("GREEN", report["trigger_state"])
        self.assertIn(
            report["disposition"],
            {
                "FORMALLY_SCOPE_REMOVED_FULL_SHAPE_SOAP_UNECONOMIC_M1_MLX",
                "ELIGIBLE_FOR_MATCHED_QUALITY_CANARY",
            },
        )
        if report["disposition"].startswith("FORMALLY_SCOPE_REMOVED"):
            self.assertEqual(
                "REMOVED_FROM_FIRST_CAMPAIGN_FINITE_DOCKET",
                report["finite_docket_membership"],
            )
            self.assertEqual(
                "REMOVE_FULL_SHAPE_SOAP_FROM_FIRST_CAMPAIGN",
                report["engineering_scope_decision"],
            )
            self.assertTrue(report["reentry_condition"])
        self.assertEqual("NOT_EVALUATED", report["scientific_optimizer_quality_claim"])
        self.assertEqual(0, report["public_training_rows"])
        self.assertFalse(report["production_checkpoint_mutation"])


if __name__ == "__main__":
    unittest.main()
