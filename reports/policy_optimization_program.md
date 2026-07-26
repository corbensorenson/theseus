# Policy Optimization Program

- trigger_state: `GREEN`
- records: `4`
- defaults allowed: `0`
- defaults with behavior lift: `0`
- hard gaps: `0` warnings: `0`
- mean reward-probe coverage: `1.0`

## Records

| id | layer | status | default | probe coverage | behavior lift | recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| strict_generator_pairwise_replay_preference_v1 | candidate_generator | quarantined | False | 1.0 | False | eligible_for_shadow_or_candidate_tracking |
| sts_ranker_policy_v1 | candidate_ranker | guarded | False | 1.0 | True | may_prepare_guarded_default_review |
| vcm_context_policy_v1 | context_memory | candidate | False | 1.0 | False | eligible_for_shadow_or_candidate_tracking |
| octopus_router_policy_v1 | planning_router | candidate | False | 1.0 | False | eligible_for_shadow_or_candidate_tracking |

## Hard Gaps

- None.

## Warnings

- None.

## Rules

- `default_policy`: A policy update may become default only with clean authority boundaries, full reward-hacking probe coverage, and behavioral verifier/accepted-output lift.
- `loss_boundary`: LM loss, selector score, or reward value is not capability evidence unless behavioral verification improves.
- `claim_boundary`: Policy updates do not support learned-generation, public-transfer, substrate-win, or runtime-serving claims unless those claims have independent evidence.
- `public_boundary`: Public benchmark artifacts are calibration-only and never become training rows.
- `objective_boundary`: DPO, IPO, ORPO, KTO, SimPO, GRPO, RLOO, ReMax, and RLVR share one executable disabled-by-default contract; numerical movement cannot activate or select an objective.
