from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "teacher_distillation_manifest_builder",
    ROOT / "scripts" / "teacher_distillation_manifest_builder.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

SMOKE_SPEC = importlib.util.spec_from_file_location(
    "teacher_distillation_admission_smoke",
    ROOT / "scripts" / "teacher_distillation_admission_smoke.py",
)
assert SMOKE_SPEC and SMOKE_SPEC.loader
SMOKE = importlib.util.module_from_spec(SMOKE_SPEC)
SMOKE_SPEC.loader.exec_module(SMOKE)

AUDIT_SPEC = importlib.util.spec_from_file_location(
    "external_inference_audit",
    ROOT / "scripts" / "external_inference_audit.py",
)
assert AUDIT_SPEC and AUDIT_SPEC.loader
AUDIT = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(AUDIT)

PROVIDER_SPEC = importlib.util.spec_from_file_location(
    "teacher_provider_policy",
    ROOT / "scripts" / "teacher_provider_policy.py",
)
assert PROVIDER_SPEC and PROVIDER_SPEC.loader
PROVIDER = importlib.util.module_from_spec(PROVIDER_SPEC)
PROVIDER_SPEC.loader.exec_module(PROVIDER)

GATE_SPEC = importlib.util.spec_from_file_location(
    "teacher_distillation_gate",
    ROOT / "scripts" / "teacher_distillation_gate.py",
)
assert GATE_SPEC and GATE_SPEC.loader
GATE = importlib.util.module_from_spec(GATE_SPEC)
GATE_SPEC.loader.exec_module(GATE)


class TeacherProviderPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = json.loads((ROOT / "configs" / "teacher_distillation_policy.json").read_text())

    def test_openai_codex_teacher_is_allowed(self) -> None:
        result = MODULE.teacher_provider_decision(
            self.policy, {"provider": "openai", "model": "codex-current-session"}
        )
        self.assertTrue(result["accepted"])
        self.assertEqual([], result["reject_reasons"])

    def test_anthropic_teacher_is_rejected(self) -> None:
        result = MODULE.teacher_provider_decision(
            self.policy, {"provider": "anthropic", "model": "claude-sonnet"}
        )
        self.assertFalse(result["accepted"])
        self.assertIn("teacher_provider_not_approved", result["reject_reasons"])
        self.assertIn("forbidden_teacher_provider_or_model", result["reject_reasons"])

    def test_missing_identity_fails_closed(self) -> None:
        result = MODULE.teacher_provider_decision(self.policy, {})
        self.assertFalse(result["accepted"])
        self.assertIn("teacher_provider_missing", result["reject_reasons"])
        self.assertIn("teacher_model_missing", result["reject_reasons"])

    def test_policy_cannot_be_silently_widened(self) -> None:
        policy = copy.deepcopy(self.policy)
        policy["provider_policy"]["allowed_providers"].append("anthropic")
        policy["provider_policy"]["allowed_model_prefixes"].append("claude")
        policy["provider_policy"]["forbidden_markers"] = []
        result = MODULE.teacher_provider_decision(
            policy, {"provider": "anthropic", "model": "claude-opus"}
        )
        self.assertFalse(result["accepted"])
        self.assertIn("forbidden_teacher_provider_or_model", result["reject_reasons"])

    def test_oracle_launch_requires_codex_cli_and_executable(self) -> None:
        teacher_policy = json.loads((ROOT / "configs" / "teacher_policy.json").read_text())
        approved = PROVIDER.teacher_launch_decision(self.policy, teacher_policy)
        self.assertTrue(approved["accepted"])

        forbidden = copy.deepcopy(teacher_policy)
        forbidden["provider"] = "anthropic"
        forbidden["model"] = "claude-opus"
        forbidden["codex_command"] = "claude"
        rejected = PROVIDER.teacher_launch_decision(self.policy, forbidden)
        self.assertFalse(rejected["accepted"])
        self.assertIn("teacher_oracle_requires_codex_executable", rejected["reject_reasons"])
        self.assertIn("forbidden_teacher_provider_or_model", rejected["reject_reasons"])

        disguised = copy.deepcopy(teacher_policy)
        disguised["codex_command"] = "/tmp/codex"
        disguised_rejected = PROVIDER.teacher_launch_decision(self.policy, disguised)
        self.assertFalse(disguised_rejected["accepted"])
        self.assertIn(
            "teacher_oracle_requires_codex_executable",
            disguised_rejected["reject_reasons"],
        )

    def test_manifest_admits_openai_and_rejects_anthropic(self) -> None:
        approved = SMOKE.build_fixture_call()
        approved_report = MODULE.build_manifest(
            policy=self.policy,
            policy_path=ROOT / "configs" / "teacher_distillation_policy.json",
            teacher_calls_path=ROOT / "reports" / "teacher_calls.jsonl",
            teacher_calls=[approved],
        )
        self.assertEqual("GREEN", approved_report["trigger_state"])
        self.assertEqual(1, approved_report["summary"]["row_count"])

        forbidden = copy.deepcopy(approved)
        forbidden["provider"] = "anthropic"
        forbidden["model"] = "claude-sonnet"
        forbidden_report = MODULE.build_manifest(
            policy=self.policy,
            policy_path=ROOT / "configs" / "teacher_distillation_policy.json",
            teacher_calls_path=ROOT / "reports" / "teacher_calls.jsonl",
            teacher_calls=[forbidden],
        )
        self.assertEqual("RED", forbidden_report["trigger_state"])
        self.assertEqual(0, forbidden_report["summary"]["row_count"])
        self.assertEqual(1, forbidden_report["summary"]["teacher_provider_violation_count"])
        reasons = forbidden_report["manifest"]["rejected_candidates"][0]["reject_reasons"]
        self.assertIn("forbidden_teacher_provider_or_model", reasons)

    def test_real_receipt_binds_codex_executable_and_model(self) -> None:
        row = {
            "provider": "codex_cli",
            "model": "gpt-5.5",
            "status": "completed",
            "external_inference_calls": 1,
            "command": [
                "/Applications/Codex.app/Contents/Resources/codex",
                "exec",
                "-m",
                "gpt-5.5",
                "-s",
                "read-only",
                "-",
            ],
        }
        result = PROVIDER.teacher_receipt_decision(self.policy, row)
        self.assertTrue(result["accepted"])
        self.assertEqual("codex", result["executable"])
        self.assertEqual("gpt-5.5", result["command_model"])
        self.assertTrue(result["command_sha256"])

    def test_relabelled_non_openai_receipt_is_rejected(self) -> None:
        row = {
            "provider": "codex_cli",
            "model": "gpt-5.5",
            "status": "completed",
            "external_inference_calls": 1,
            "command": ["claude", "exec", "-m", "gpt-5.5", "-"],
        }
        result = PROVIDER.teacher_receipt_decision(self.policy, row)
        self.assertFalse(result["accepted"])
        self.assertIn("external_teacher_receipt_executable_not_codex", result["reject_reasons"])
        self.assertIn("forbidden_teacher_executable_or_argument", result["reject_reasons"])

    def test_real_receipt_without_command_fails_closed(self) -> None:
        row = {
            "provider": "codex_cli",
            "model": "gpt-5.5",
            "status": "completed",
            "external_inference_calls": 1,
        }
        result = PROVIDER.teacher_receipt_decision(self.policy, row)
        self.assertFalse(result["accepted"])
        self.assertIn("external_teacher_receipt_command_missing", result["reject_reasons"])

    def test_receipt_model_must_match_command(self) -> None:
        row = {
            "provider": "codex_cli",
            "model": "gpt-5.5",
            "status": "completed",
            "external_inference_calls": 1,
            "command": ["codex", "exec", "-m", "gpt-5.4", "-"],
        }
        result = PROVIDER.teacher_receipt_decision(self.policy, row)
        self.assertFalse(result["accepted"])
        self.assertIn("external_teacher_receipt_model_mismatch", result["reject_reasons"])

    def test_relabelled_nested_provider_provenance_is_rejected(self) -> None:
        row = {
            "provider": "codex_cli",
            "model": "gpt-5.5",
            "status": "completed",
            "external_inference_calls": 1,
            "command": ["codex", "exec", "-m", "gpt-5.5", "-"],
            "transport_receipt": {
                "upstream_provider": "anthropic",
                "upstream_model": "claude-sonnet",
            },
        }
        result = PROVIDER.teacher_receipt_decision(self.policy, row)
        self.assertFalse(result["accepted"])
        self.assertIn(
            "forbidden_teacher_identity_in_receipt_provenance",
            result["reject_reasons"],
        )
        self.assertEqual(2, len(result["forbidden_identity_values"]))

    def test_external_audit_recomputes_teacher_provider_summary(self) -> None:
        approved = {
            "provider": "codex_cli",
            "model": "gpt-5.5",
            "status": "completed",
            "external_inference_calls": 1,
            "command": ["codex", "exec", "-m", "gpt-5.5", "-"],
        }
        violations, summary = AUDIT.audit_teacher_receipt_rows([approved], self.policy)
        self.assertEqual([], violations)
        self.assertEqual(1, summary["scanned_teacher_receipts"])
        self.assertEqual({"codex_cli/gpt-5.5": 1}, summary["teacher_provider_counts"])

        forbidden = copy.deepcopy(approved)
        forbidden["provider_metadata"] = {"vendor": "Anthropic"}
        violations, _summary = AUDIT.audit_teacher_receipt_rows([forbidden], self.policy)
        self.assertEqual(1, len(violations))
        self.assertIn(
            "forbidden_teacher_identity_in_receipt_provenance",
            violations[0]["reject_reasons"],
        )

    def test_gate_rejects_accepted_teacher_ledger_row_without_receipt_proof(self) -> None:
        row = {
            "ledger_event_id": "unproven_teacher_row",
            "source_kind": "teacher_distillation",
            "accepted": True,
            "teacher_provider": "codex_cli",
            "teacher_model": "gpt-5.5",
        }
        violations = GATE.teacher_ledger_provider_violations(
            self.policy,
            {"ledger_rows": [row]},
        )
        self.assertEqual(1, len(violations))
        self.assertEqual(
            "teacher_receipt_provenance_missing_or_invalid",
            violations[0]["violation"],
        )

    def test_external_audit_detects_claude_cli_invocation(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="teacher_provider_audit_", dir=ROOT / "scripts", delete=False
        ) as handle:
            handle.write('import subprocess\nsubprocess.run(["claude", "--print", "forbidden"])\n')
            path = Path(handle.name)
        try:
            _hits, violations, _delegates = AUDIT.scan_code([path])
        finally:
            path.unlink(missing_ok=True)
        self.assertTrue(
            any(item["kind"] == "claude_cli_invocation_forbidden" for item in violations)
        )


if __name__ == "__main__":
    unittest.main()
