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


def route_packet() -> dict:
    return {
        "ready": True,
        "selected_route": {"id": "route.procedural.planning.v1"},
    }


class AssistantEffectTransactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.allowed_root = Path(self.tempdir.name) / "assistant_effects"
        self.target = self.allowed_root / "default_route_authority.json"

    def tearDown(self) -> None:
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
        self.target.parent.mkdir(parents=True)
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
        outside = Path(self.tempdir.name) / "outside.json"
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


if __name__ == "__main__":
    unittest.main()
