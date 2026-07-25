from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import target_authoritative_drafting as drafting


class TargetAuthoritativeDraftingTests(unittest.TestCase):
    def manifest(self, **overrides: object) -> drafting.DraftManifest:
        values = {
            "schema_version": "1.0.0",
            "mode": "medusa_tree",
            "draft_revision": "draft:1",
            "draft_checkpoint_digest": "draft-sha",
            "target_model_revision": "target:1",
            "target_parameter_digest": "target-sha",
            "codec_digest": "codec-sha",
            "cache_schema_version": "target_tree_kv_v1",
            "max_draft_depth": 4,
        }
        values.update(overrides)
        return drafting.DraftManifest(**values)

    def test_manifest_binds_exact_target_codec_and_cache_policy(self) -> None:
        receipt = drafting.validate_manifest(
            self.manifest(),
            target_model_revision="target:1",
            target_parameter_digest="target-sha",
            codec_digest="codec-sha",
        )
        self.assertTrue(receipt["valid"])
        for key, value, fault in (
            ("target_model_revision", "wrong", "draft_target_revision_mismatch"),
            ("target_parameter_digest", "wrong", "draft_target_parameter_mismatch"),
            ("codec_digest", "wrong", "draft_codec_mismatch"),
            ("cache_commit_policy", "all_proposals", "draft_cache_commit_policy_invalid"),
        ):
            with self.assertRaisesRegex(drafting.DraftAuthorityFault, fault):
                drafting.validate_manifest(
                    self.manifest(**{key: value}),
                    target_model_revision="target:1",
                    target_parameter_digest="target-sha",
                    codec_digest="codec-sha",
                )

    def test_mismatch_discards_unaccepted_suffix_and_commits_target(self) -> None:
        prefix = [2, 5, 7]
        first = drafting.reference_target_next(tuple(prefix))
        prefix_after_first = tuple(prefix + [first])
        target_second = drafting.reference_target_next(prefix_after_first)
        receipt = drafting.verify_draft_branch(
            prefix,
            [first, (target_second + 1) % 41, 9, 10],
            drafting.reference_target_next,
        )
        self.assertEqual((first,), receipt.accepted_draft_tokens)
        self.assertEqual(3, len(receipt.rejected_draft_tokens))
        self.assertEqual((first, target_second), receipt.committed_tokens)
        self.assertEqual(1, receipt.rollback_count)

    def test_full_decode_is_exactly_target_equivalent_under_bad_drafts(self) -> None:
        prefix = [2, 5, 7]
        canonical = list(prefix)
        for _ in range(20):
            canonical.append(drafting.reference_target_next(tuple(canonical)))

        def proposer(current: tuple[int, ...], budget: int) -> list[int]:
            scratch = list(current)
            proposals = []
            for index in range(budget):
                token = drafting.reference_target_next(tuple(scratch))
                if index == 2:
                    token = (token + 5) % 41
                proposals.append(token)
                scratch.append(token)
            return proposals

        decoded, receipts = drafting.target_authoritative_decode(
            prefix,
            proposer,
            drafting.reference_target_next,
            max_new_tokens=20,
            max_draft_depth=5,
        )
        self.assertEqual(canonical, decoded)
        self.assertGreater(sum(receipt.rollback_count for receipt in receipts), 0)

    def test_empty_draft_and_budget_overflow_fail_safely(self) -> None:
        decoded, receipts = drafting.target_authoritative_decode(
            [1],
            lambda _prefix, _budget: [],
            drafting.reference_target_next,
            max_new_tokens=3,
            max_draft_depth=2,
        )
        self.assertEqual(4, len(decoded))
        self.assertEqual([], receipts)
        with self.assertRaisesRegex(
            drafting.DraftAuthorityFault, "drafter_exceeded_depth_budget"
        ):
            drafting.target_authoritative_decode(
                [1],
                lambda _prefix, budget: [2] * (budget + 1),
                drafting.reference_target_next,
                max_new_tokens=3,
                max_draft_depth=2,
            )

    def test_medusa_and_eagle_train_against_frozen_target_and_reload(self) -> None:
        receipt = drafting.mlx_drafting_adequacy_canary(optimizer_steps=48)
        self.assertTrue(receipt["available"], receipt.get("stderr_tail"))
        self.assertTrue(receipt["passed"], receipt)
        self.assertEqual(
            {"medusa_tree", "eagle_feature", "separate_draft"},
            set(receipt["observed"]["results"]),
        )
        self.assertEqual(0, receipt["public_training_rows"])
        self.assertEqual("NOT_EVALUATED", receipt["capability_claim"])

    def test_real_mlx_target_kv_rejects_suffix_and_matches_canonical_greedy(self) -> None:
        import mlx.core as mx
        import mlx.nn as nn
        from standard_causal_transformer_model import CausalTransformerConfig, build_model

        mx.random.seed(441)
        model = build_model(
            CausalTransformerConfig(
                vocab_size=43,
                d_model=24,
                num_layers=2,
                num_heads=4,
                num_kv_heads=2,
                ff_dim=64,
            ),
            mx=mx,
            nn=nn,
        )
        prefix = [1, 5, 9]

        def canonical(count: int) -> list[int]:
            state = drafting.prefill_target_kv(
                model, prefix, mx=mx, target_revision="target:test"
            )
            result = list(prefix)
            for _ in range(count):
                state, _receipt = drafting.verify_draft_branch_with_target_kv(
                    state,
                    [],
                    model,
                    mx=mx,
                    target_revision="target:test",
                )
                result.append(state.committed_prefix[-1])
            return result

        expected = canonical(12)

        def proposer(current: tuple[int, ...], budget: int) -> list[int]:
            state = drafting.prefill_target_kv(
                model, current, mx=mx, target_revision="target:test"
            )
            values = []
            for index in range(budget):
                state, _receipt = drafting.verify_draft_branch_with_target_kv(
                    state,
                    [],
                    model,
                    mx=mx,
                    target_revision="target:test",
                )
                token = state.committed_prefix[-1]
                values.append((token + 1) % 43 if index == 2 else token)
            return values

        observed, receipts = drafting.target_authoritative_kv_decode(
            model,
            prefix,
            proposer,
            mx=mx,
            target_revision="target:test",
            max_new_tokens=12,
            max_draft_depth=4,
        )
        self.assertEqual(expected, observed)
        self.assertGreater(sum(row.rollback_count for row in receipts), 0)
        self.assertTrue(all(not row.rejected_suffix_cache_committed for row in receipts))
        self.assertTrue(all(row.exact_target_authority for row in receipts))
        self.assertTrue(
            all(
                row.committed_cache_length
                == len(row.initial_prefix) + len(row.committed_tokens)
                for row in receipts
            )
        )

    def test_target_kv_revision_and_truncation_fail_closed(self) -> None:
        class CacheArray:
            shape = (1, 2, 3, 4)

            def __getitem__(self, _key: object) -> "CacheArray":
                return self

        cache = [(CacheArray(), CacheArray())]
        self.assertEqual(3, drafting.target_cache_length(cache))
        with self.assertRaisesRegex(
            drafting.DraftAuthorityFault, "target_cache_truncation_out_of_bounds"
        ):
            drafting.truncate_target_cache(cache, 4)

    def test_reference_suite_is_green_without_capability_claim(self) -> None:
        report = drafting.run_reference_suite()
        self.assertEqual("GREEN", report["trigger_state"])
        self.assertTrue(report["target_parity"])
        self.assertGreater(report["rollback_count"], 0)
        self.assertTrue(report["mlx_target_kv_authority"]["passed"])
        self.assertEqual(
            "NOT_EVALUATED",
            report["mlx_target_kv_authority"]["speed_claim"],
        )


if __name__ == "__main__":
    unittest.main()
