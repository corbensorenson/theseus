from __future__ import annotations

import base64
import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_artifact_admission as admission  # noqa: E402
import neural_seed_token_model_backend as token_backend  # noqa: E402
import standard_causal_transformer_survival as transformer_survival  # noqa: E402


def sign(payload: dict, private_key: Ed25519PrivateKey, key_id: str = "test-key") -> dict:
    row = dict(payload)
    value = private_key.sign(admission.canonical_payload(row))
    row["signature"] = {
        "alg": "ed25519",
        "key_id": key_id,
        "value": base64.urlsafe_b64encode(value).decode("ascii").rstrip("="),
    }
    return row


class ArtifactAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.artifact = self.root / "model.bin"
        self.artifact.write_bytes(b"theseus-model-weights")
        os.chmod(self.artifact, 0o600)
        self.private_key = Ed25519PrivateKey.generate()
        public_hex = self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ).hex()
        self.now = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)
        self.policy = {
            "purpose": "local_private_model_runtime",
            "max_advisory_age_hours": 24,
            "trusted_public_keys": {"test-key": public_hex},
            "revoked_key_ids": [],
            "revoked_artifact_hashes": [],
            "minimum_generation_by_artifact": {"model.main": 3},
        }
        self.advisory = sign(
            {
                "policy": "test_advisory_v1",
                "created_utc": self.now.isoformat(),
                "revoked_artifact_hashes": [],
            },
            self.private_key,
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def attestation(self, **overrides: object) -> dict:
        payload = {
            "policy": "test_attestation_v1",
            "logical_artifact_id": "model.main",
            "artifact_sha256": admission.file_sha256(self.artifact),
            "generation": 3,
            "issued_utc": (self.now - timedelta(hours=1)).isoformat(),
            "expires_utc": (self.now + timedelta(hours=1)).isoformat(),
            "advisory_snapshot_sha256": admission.payload_sha256(self.advisory),
            "allowed_purposes": ["local_private_model_runtime"],
            "value_class": "development",
            "build_identity": "sha256:build",
            "training_identity": "sha256:training",
            "custody": {},
        }
        payload.update(overrides)
        return sign(payload, self.private_key)

    def admit(self, attestation: dict | None = None, advisory: dict | None = None, policy: dict | None = None, path: Path | None = None) -> dict:
        return admission.admit_artifact(
            path or self.artifact,
            attestation=attestation or self.attestation(),
            advisory_snapshot=advisory or self.advisory,
            policy=policy or self.policy,
            now_utc=self.now,
        )

    def test_valid_identity_survives_relocation(self) -> None:
        relocated = self.root / "relocated" / "model-renamed.bin"
        relocated.parent.mkdir()
        shutil.copy2(self.artifact, relocated)
        os.chmod(relocated, 0o600)

        result = self.admit(path=relocated)

        self.assertTrue(result["admitted"])
        self.assertEqual(result["artifact"]["observed_sha256"], admission.file_sha256(self.artifact))

    def test_tampered_artifact_fails_before_deserialization(self) -> None:
        attestation = self.attestation()
        self.artifact.write_bytes(b"tampered")

        result = self.admit(attestation=attestation)

        self.assertFalse(result["admitted"])
        self.assertIn("artifact_hash_mismatch", result["reasons"])

    def test_stale_advisory_revoked_key_and_rollback_fail(self) -> None:
        stale = sign(
            {
                "policy": "test_advisory_v1",
                "created_utc": (self.now - timedelta(days=2)).isoformat(),
                "revoked_artifact_hashes": [],
            },
            self.private_key,
        )
        stale_result = self.admit(attestation=self.attestation(advisory_snapshot_sha256=admission.payload_sha256(stale)), advisory=stale)
        self.assertIn("advisory_snapshot_stale", stale_result["reasons"])

        revoked_policy = dict(self.policy)
        revoked_policy["revoked_key_ids"] = ["test-key"]
        self.assertIn("signing_key_revoked", self.admit(policy=revoked_policy)["reasons"])

        self.assertIn("artifact_generation_rollback", self.admit(attestation=self.attestation(generation=2))["reasons"])

    def test_valuable_weight_requires_observed_custody_and_key_release(self) -> None:
        result = self.admit(attestation=self.attestation(value_class="valuable_weight"))

        self.assertFalse(result["admitted"])
        self.assertIn("valuable_weight_encrypted_storage_unproven", result["reasons"])
        self.assertIn("valuable_weight_key_release_missing", result["reasons"])
        self.assertIn("valuable_weight_anti_rollback_state_missing", result["reasons"])

    def test_checkpoint_loader_denies_before_deserialization(self) -> None:
        class TorchMustNotLoad:
            def load(self, *args: object, **kwargs: object) -> object:
                raise AssertionError("deserializer reached before admission")

        result = token_backend.load_pretraining_initializer_for_arm(
            arm_id="transformer",
            tokenizer={"vocab": {"<pad>": 0, "<unk>": 1}, "merges": []},
            tokenizer_path=self.root / "tokenizer.json",
            checkpoint_path=self.artifact,
            torch=TorchMustNotLoad(),
            device="cpu",
            admission_config={
                "artifact_admission": {
                    "required": True,
                    "policy": str(self.root / "missing-policy.json"),
                    "attestation": str(self.root / "missing-attestation.json"),
                    "advisory_snapshot": str(self.root / "missing-advisory.json"),
                }
            },
        )

        self.assertFalse(result["active"])
        self.assertTrue(result["reason"].startswith("artifact_admission_rejected:"))

        class ModelMustNotLoad:
            def load_weights(self, path: str) -> None:
                raise AssertionError("MLX/NumPy loader reached before admission")

        with self.assertRaisesRegex(ValueError, "artifact admission rejected before load"):
            transformer_survival.admitted_load_weights(
                ModelMustNotLoad(),
                self.artifact,
                {
                    "artifact_admission": {
                        "required": True,
                        "policy": str(self.root / "missing-policy.json"),
                        "attestation": str(self.root / "missing-attestation.json"),
                        "advisory_snapshot": str(self.root / "missing-advisory.json"),
                    }
                },
            )


if __name__ == "__main__":
    unittest.main()
