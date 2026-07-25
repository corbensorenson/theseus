from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import dynamic_byte_patch_codec as codec


class DynamicBytePatchCodecTests(unittest.TestCase):
    def test_arbitrary_binary_round_trips_exactly(self) -> None:
        payload = bytes(range(256)) + "Theseus: λ".encode("utf-8")
        probabilities = [0.9 if index % 11 == 10 else 0.1 for index in range(len(payload))]
        boundaries = codec.boundaries_from_probabilities(
            probabilities, threshold=0.5, max_patch_bytes=16
        )
        envelope = codec.encode_lossless(
            payload,
            boundaries,
            codec_revision="test",
            boundary_source="test_probabilities",
        )
        self.assertEqual(payload, codec.decode_lossless(envelope))

    def test_protected_span_is_never_split(self) -> None:
        payload = b"prefix SECRET-VALUE suffix"
        protected = ((7, 19),)
        probabilities = [1.0] * len(payload)
        boundaries = codec.boundaries_from_probabilities(
            probabilities,
            threshold=0.5,
            max_patch_bytes=16,
            protected_spans=protected,
        )
        self.assertFalse(any(7 < point < 19 for point in boundaries))
        envelope = codec.encode_lossless(
            payload,
            boundaries,
            codec_revision="test",
            boundary_source="test",
            protected_spans=protected,
        )
        self.assertTrue(any(patch.protected for patch in envelope.patches))

    def test_corrupt_patch_fails_closed(self) -> None:
        payload = b"bounded evidence"
        envelope = codec.encode_lossless(
            payload,
            (0, 7, len(payload)),
            codec_revision="test",
            boundary_source="test",
        )
        record = dataclasses.asdict(envelope)
        record["patches"][0]["payload_hex"] = b"changed".hex()
        with self.assertRaisesRegex(codec.DynamicPatchFault, "patch_length_mismatch|patch_checksum_mismatch"):
            codec.decode_lossless(record)

    def test_invalid_or_impossible_boundary_policy_fails_closed(self) -> None:
        with self.assertRaisesRegex(codec.DynamicPatchFault, "boundary_policy_invalid"):
            codec.boundaries_from_probabilities([0.1], threshold=2.0, max_patch_bytes=4)
        with self.assertRaisesRegex(codec.DynamicPatchFault, "protected_span_exceeds_patch_budget"):
            codec.boundaries_from_probabilities(
                [0.1] * 20,
                threshold=0.5,
                max_patch_bytes=4,
                protected_spans=((2, 12),),
            )

    def test_learned_causal_codec_mechanics_train_intervene_and_reload(self) -> None:
        receipt = codec.mlx_dynamic_patch_adequacy_canary(optimizer_steps=64)
        self.assertTrue(receipt["available"], receipt.get("stderr_tail"))
        self.assertTrue(receipt["passed"], receipt)
        self.assertEqual(0, receipt["public_training_rows"])
        self.assertEqual("NOT_EVALUATED", receipt["capability_claim"])
        self.assertTrue(receipt["checks"]["within_patch_local_causality"])

    def test_single_byte_decoder_uses_previous_byte_not_current_target(self) -> None:
        import mlx.core as mx
        import mlx.nn as nn

        mx.random.seed(19)
        model = codec.build_dynamic_patch_model(
            d_model=16,
            patch_hidden_dim=24,
            max_patches=8,
            max_patch_bytes=8,
            mx=mx,
            nn=nn,
        )
        latent = mx.ones((1, 16), dtype=mx.float32)
        position = mx.array([2], dtype=mx.int32)
        previous_a = mx.array([17], dtype=mx.int32)
        previous_b = mx.array([18], dtype=mx.int32)
        first = model.decode_next_byte(latent, position, previous_a)
        same = model.decode_next_byte(latent, position, previous_a)
        changed = model.decode_next_byte(latent, position, previous_b)
        mx.eval(first, same, changed)
        self.assertEqual(0.0, float(mx.max(mx.abs(first - same)).item()))
        self.assertGreater(float(mx.max(mx.abs(first - changed)).item()), 0.0)

        patch_latents = mx.stack([latent, latent * 2.0], axis=1)
        patch_ids = mx.array([[0, 0, 1]], dtype=mx.int32)
        positions = mx.array([[0, 1, 0]], dtype=mx.int32)
        byte_ids = mx.array([[10, 11, 12]], dtype=mx.int32)
        teacher_forced = model.decode_patch_latents(
            patch_latents, patch_ids, positions, byte_ids
        )
        sequential_middle = model.decode_next_byte(
            patch_latents[:, 0, :],
            mx.array([1], dtype=mx.int32),
            mx.array([10], dtype=mx.int32),
        )
        sequential_patch_start = model.decode_next_byte(
            patch_latents[:, 1, :],
            mx.array([0], dtype=mx.int32),
            None,
        )
        mx.eval(teacher_forced, sequential_middle, sequential_patch_start)
        self.assertLessEqual(
            float(
                mx.max(mx.abs(teacher_forced[:, 1, :] - sequential_middle)).item()
            ),
            1e-6,
        )
        self.assertLessEqual(
            float(
                mx.max(
                    mx.abs(teacher_forced[:, 2, :] - sequential_patch_start)
                ).item()
            ),
            1e-6,
        )

    def test_patch_core_contracts_to_active_capacity_and_masks_padding(self) -> None:
        import mlx.core as mx
        import mlx.nn as nn

        mx.random.seed(23)
        model = codec.build_dynamic_patch_causal_candidate(
            d_model=16,
            patch_hidden_dim=24,
            max_patches=64,
            max_patch_bytes=8,
            vocab_size=256,
            mx=mx,
            nn=nn,
        )
        byte_ids = mx.array([[10, 11, 12, 13, 99, 98]], dtype=mx.int32)
        patch_ids = mx.array([[0, 0, 1, 1, 1, 1]], dtype=mx.int32)
        positions = mx.array([[0, 1, 0, 1, 0, 0]], dtype=mx.int32)
        uncertainty = mx.ones(byte_ids.shape, dtype=mx.float32)
        mask = mx.array([[1, 1, 1, 1, 0, 0]], dtype=mx.float32)
        first = model(byte_ids, patch_ids, positions, uncertainty, mask)
        altered = mx.array([[10, 11, 12, 13, 4, 5]], dtype=mx.int32)
        second = model(altered, patch_ids, positions, uncertainty, mask)
        mx.eval(first, second)
        self.assertEqual(2, int(first[2].shape[1]))
        self.assertLessEqual(
            float(mx.max(mx.abs(first[0][:, :4, :] - second[0][:, :4, :])).item()),
            1e-6,
        )

    def test_reference_suite_is_green_without_superiority_claim(self) -> None:
        report = codec.run_reference_suite()
        self.assertEqual("GREEN", report["trigger_state"])
        self.assertTrue(report["lossless_roundtrip"])


if __name__ == "__main__":
    unittest.main()
